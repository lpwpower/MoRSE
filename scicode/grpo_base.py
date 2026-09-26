from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

SCRIPT_PATH = Path(__file__).resolve()
SCICODE_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from morse.mole.mole_generator import GenerationConfig, MoLEGenerator  # noqa: E402
from morse.mole.mole_lora import LoRAConfig, inject_mole_lora  # noqa: E402
from morse.mole.router import SubtaskRouter, SubtaskRouterConfig  # noqa: E402

from scicode import pipeline as scipipe  # noqa: E402
from scicode.sft import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATASET,
    DEFAULT_H5PY_FILE,
    SCICODE_ROOT as _SCICODE_ROOT_SENTINEL,
    StepSFTSample,
    TitleEmbedder,
    _build_sft_samples,
    _extract_step_code_for_eval,
    _iter_selected_problems,
    _load_backbone,
    _ordered_step_ids,
    _read_jsonl,
    _seed_everything,
    _select_prompt_template,
    _write_json,
    evaluate_all_training_samples,
    load_mole_checkpoint,
    save_mole_checkpoint,
)

# Sanity check that both scripts resolve the same root.
if _SCICODE_ROOT_SENTINEL != SCICODE_ROOT:
    raise RuntimeError("SCICODE_ROOT mismatch between SFT and GRPO scripts.")

DEFAULT_RUN_ROOT = SCICODE_ROOT / "runs" / "scicode_mole_grpo"


@dataclass
class GRPOCandidate:
    idx: int
    text: str
    python_code: str
    parsed_function: str
    prompt_ids: torch.Tensor
    gen_ids: torch.Tensor
    expert_ids: torch.Tensor
    logp_router: torch.Tensor
    reward: float
    metrics: Dict[str, Any]


def _sanitize(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_") or "item"


def _ast_node_type_counts(code: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    try:
        tree = ast.parse(code)
    except Exception:
        return counts
    for node in ast.walk(tree):
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _ast_similarity(a: str, b: str) -> float:
    ca = _ast_node_type_counts(a)
    cb = _ast_node_type_counts(b)
    if not ca or not cb:
        return 0.0
    keys = set(ca.keys()) | set(cb.keys())
    inter = 0.0
    union = 0.0
    for k in keys:
        va = float(ca.get(k, 0))
        vb = float(cb.get(k, 0))
        inter += min(va, vb)
        union += max(va, vb)
    if union <= 0:
        return 0.0
    return float(inter / union)


def _clip_advantage(adv: float, clip: float) -> float:
    c = float(clip)
    if c <= 0.0:
        return float(adv)
    if adv > c:
        return float(c)
    if adv < -c:
        return float(-c)
    return float(adv)


def _safe_std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / float(len(vals))
    var = sum((x - mean) ** 2 for x in vals) / float(len(vals))
    return float(var ** 0.5)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tail_text(value: str, max_chars: int = 1200) -> str:
    text = str(value or "")
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _write_candidate_snapshot(
    *,
    snapshot_root: Path,
    rank: int,
    global_step_next: int,
    item_idx: int,
    candidate_idx: int,
    problem: dict,
    step_id: str,
    prompt: str,
    response_text: str,
    python_code: str,
    parsed_function: str,
    oracle_ancestors: Dict[str, str],
    metrics: Dict[str, Any],
    step_result: Optional[scipipe.ScriptRunResult],
) -> Dict[str, str]:
    problem_id = str(problem.get("problem_id") or "")
    base = (
        snapshot_root
        / f"rank_{rank:02d}"
        / f"{_sanitize(problem_id)}_{_sanitize(step_id)}"
        / f"gs_{int(global_step_next):07d}_item_{int(item_idx):05d}_cand_{int(candidate_idx):03d}"
    )
    base.mkdir(parents=True, exist_ok=True)

    prompt_path = base / "prompt.txt"
    response_path = base / "response.txt"
    python_path = base / "python.py"
    parsed_fn_path = base / "parsed_function.py"
    assembled_path = base / "assembled_program.py"
    step_test_path = base / "step_test_result.json"
    metrics_path = base / "metrics.json"

    prompt_path.write_text(str(prompt or ""), encoding="utf-8")
    response_path.write_text(str(response_text or ""), encoding="utf-8")
    python_path.write_text(str(python_code or "") + "\n", encoding="utf-8")
    parsed_fn_path.write_text(str(parsed_function or "") + "\n", encoding="utf-8")

    assembled_code = ""
    if str(python_code or "").strip():
        try:
            assembled_code = scipipe._assemble_program_code(
                dependencies=str(problem.get("required_dependencies") or ""),
                ancestor_step_ids=list(oracle_ancestors.keys()),
                solved_functions=oracle_ancestors,
                current_python_code=python_code,
            )
        except Exception as exc:
            assembled_code = f"# assemble_failed: {exc}\n" + str(python_code or "")
    assembled_path.write_text(str(assembled_code), encoding="utf-8")

    _write_json(
        step_test_path,
        {
            "status": step_result.status if step_result is not None else "skipped",
            "passed": bool(step_result.passed) if step_result is not None else False,
            "return_code": int(step_result.return_code) if step_result is not None else None,
            "elapsed_ms": int(step_result.elapsed_ms) if step_result is not None else None,
            "script_path": str(step_result.script_path) if step_result is not None else "",
            "stdout_tail": _tail_text(step_result.stdout) if step_result is not None else "",
            "stderr_tail": _tail_text(step_result.stderr) if step_result is not None else "",
        },
    )
    _write_json(metrics_path, dict(metrics))

    return {
        "snapshot_dir": str(base.resolve()),
        "snapshot_prompt_path": str(prompt_path.resolve()),
        "snapshot_response_path": str(response_path.resolve()),
        "snapshot_python_path": str(python_path.resolve()),
        "snapshot_parsed_function_path": str(parsed_fn_path.resolve()),
        "snapshot_assembled_program_path": str(assembled_path.resolve()),
        "snapshot_step_test_result_path": str(step_test_path.resolve()),
        "snapshot_metrics_path": str(metrics_path.resolve()),
    }


def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_gather_float_list(local_vals: List[float], *, device: torch.device, world_size: int) -> List[float]:
    if world_size <= 1:
        return [float(v) for v in local_vals]
    local = torch.tensor(local_vals, dtype=torch.float32, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    merged: List[float] = []
    for tensor in gathered:
        merged.extend([float(x) for x in tensor.detach().cpu().tolist()])
    return merged


def _global_best_metrics(
    *,
    local_best_reward: float,
    local_best_step: float,
    local_best_shape: float,
    local_best_gt: float,
    device: torch.device,
    world_size: int,
) -> Tuple[float, float, float, float]:
    vec = torch.tensor(
        [float(local_best_reward), float(local_best_step), float(local_best_shape), float(local_best_gt)],
        dtype=torch.float32,
        device=device,
    )
    if world_size <= 1:
        out = vec.detach().cpu().tolist()
        return float(out[0]), float(out[1]), float(out[2]), float(out[3])
    gathered = [torch.empty_like(vec) for _ in range(world_size)]
    dist.all_gather(gathered, vec)
    best = max((g.detach().cpu().tolist() for g in gathered), key=lambda x: float(x[0]))
    return float(best[0]), float(best[1]), float(best[2]), float(best[3])


def _average_gradients(params: List[torch.nn.Parameter], world_size: int) -> None:
    if world_size <= 1:
        return
    scale = 1.0 / float(world_size)
    for p in params:
        if p.grad is None:
            # Ensure every rank has a real grad tensor so all_reduce can propagate
            # non-zero gradients from other ranks to this rank.
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.mul_(scale)


def _build_oracle_ancestors(problem: dict) -> Dict[str, Dict[str, str]]:
    """Map step_id -> solved_functions dict of all earlier GT steps."""
    step_by_id, step_order = scipipe._build_step_maps(problem)
    ordered = _ordered_step_ids(step_order)
    oracle_for_step: Dict[str, Dict[str, str]] = {}
    gt_so_far: Dict[str, str] = {}
    for step_id in ordered:
        oracle_for_step[step_id] = dict(gt_so_far)
        step = step_by_id[step_id]
        gt = str(step.get("ground_truth_code") or "").strip()
        if gt:
            gt_so_far[step_id] = gt
    return oracle_for_step


def _evaluate_candidate_reward(
    *,
    problem: dict,
    step: dict,
    step_id: str,
    python_code: str,
    parsed_function: str,
    oracle_ancestors: Dict[str, str],
    sample_eval_dir: Path,
    h5py_file: Path,
    timeout_s: int,
    env: Dict[str, str],
    w_step: float,
    w_shape: float,
    w_gt: float,
    pass_bonus: float,
) -> Tuple[float, Dict[str, Any], Optional[scipipe.ScriptRunResult]]:
    parse_ok = bool(python_code.strip())
    header_ok = bool(parsed_function.strip())
    ast_ok = 0.0
    if parse_ok:
        try:
            ast.parse(python_code)
            ast_ok = 1.0
        except Exception:
            ast_ok = 0.0

    if not parse_ok:
        return 0.0, {
            "parse_ok": False,
            "header_ok": False,
            "ast_ok": 0.0,
            "step_score": 0.0,
            "shape_score": 0.0,
            "reward": 0.0,
        }, None

    assembled_code = scipipe._assemble_program_code(
        dependencies=str(problem.get("required_dependencies") or ""),
        ancestor_step_ids=list(oracle_ancestors.keys()),
        solved_functions=oracle_ancestors,
        current_python_code=python_code,
    )
    tests = list(step.get("test_cases") or [])
    step_result = scipipe._run_step_test(
        sample_dir=sample_eval_dir,
        step_id=step_id,
        assembled_code=assembled_code,
        test_cases=tests,
        h5py_file=h5py_file,
        timeout_s=timeout_s,
        env=env,
    )

    step_score = 1.0 if step_result.passed else 0.0
    shape_score = (0.4 * float(parse_ok)) + (0.4 * float(header_ok)) + (0.2 * float(ast_ok))

    gate = 1.0 if (parse_ok and header_ok) else 0.0
    reward = gate * (float(w_step) * step_score + float(w_shape) * shape_score)
    if step_score >= 1.0:
        reward += float(pass_bonus)

    return float(reward), {
        "parse_ok": bool(parse_ok),
        "header_ok": bool(header_ok),
        "ast_ok": float(ast_ok),
        "step_score": float(step_score),
        "shape_score": float(shape_score),
        "reward": float(reward),
    }, step_result


def _build_training_step_entries(
    *,
    problems: List[dict],
    with_background: bool,
    prompt_template: str,
    max_steps_per_problem: int,
    graph_root: Optional[Path],
) -> List[Tuple[dict, dict, str, str, Dict[str, str], str]]:
    """Return tuples: (problem, step, step_id, prompt, oracle_ancestors, subtask_text)."""
    entries: List[Tuple[dict, dict, str, str, Dict[str, str], str]] = []
    graph_index: Dict[str, Path] = {}
    if graph_root is not None:
        graph_index = scipipe._build_graph_root_index(graph_root)

    for problem in problems:
        step_by_id, step_order = scipipe._build_step_maps(problem)
        if not step_by_id:
            continue
        problem_id = str(problem.get("problem_id") or "")

        plan: List[Tuple[str, List[str]]] = []
        graph_dir = graph_index.get(problem_id)
        graph_path = (graph_dir / "task_graph.json") if graph_dir else None
        if graph_path is not None and graph_path.exists():
            try:
                spec = scipipe.convert_taskgraph(graph_path)
                predecessors = scipipe._build_predecessors(spec)
                for node_id in sorted(spec.node_metadata):
                    step_id = str(spec.node_metadata[node_id].original_id)
                    if step_id not in step_by_id:
                        continue
                    ancestor_node_ids = scipipe._collect_ancestors(node_id, predecessors)
                    ancestor_step_ids = [str(spec.node_metadata[a].original_id) for a in sorted(ancestor_node_ids)]
                    ancestor_step_ids = [sid for sid in ancestor_step_ids if sid in step_by_id]
                    ancestor_step_ids.sort(key=lambda sid: step_order.get(sid, 10**9))
                    plan.append((step_id, ancestor_step_ids))
            except Exception:
                plan = []

        if not plan:
            ordered = _ordered_step_ids(step_order)
            plan = [(sid, [aid for aid in ordered if step_order[aid] < step_order[sid]]) for sid in ordered]

        if max_steps_per_problem > 0:
            plan = plan[:max_steps_per_problem]

        for step_id, ancestor_step_ids in plan:
            step = step_by_id[step_id]
            target_code = str(step.get("ground_truth_code") or "").strip()
            if not target_code:
                continue
            oracle_ancestors: Dict[str, str] = {}
            for anc_id in ancestor_step_ids:
                anc_step = step_by_id.get(anc_id)
                if anc_step is None:
                    continue
                anc_gt = str(anc_step.get("ground_truth_code") or "").strip()
                if anc_gt:
                    oracle_ancestors[anc_id] = anc_gt
            rendered_ancestors = [sid for sid in ancestor_step_ids if sid in oracle_ancestors]
            prompt, _ = scipipe._render_prompt(
                problem=problem,
                step=step,
                ancestor_step_ids=rendered_ancestors,
                solved_functions=oracle_ancestors,
                step_by_id=step_by_id,
                with_background=with_background,
                prompt_template=prompt_template,
            )
            desc = str(step.get("step_description_prompt") or "").strip()
            bg = str(step.get("step_background") or "").strip()
            subtask_text = (desc + "\n" + bg).strip() if bg else desc
            entries.append((problem, step, step_id, prompt, oracle_ancestors, subtask_text or f"{problem_id}:{step_id}"))
    return entries


def train_grpo(args: argparse.Namespace) -> None:
    if args.gpus and str(args.gpus).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus).strip()

    use_dist = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_dist and not _is_dist_ready():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    world_size = int(dist.get_world_size()) if _is_dist_ready() else 1
    rank = int(dist.get_rank()) if _is_dist_ready() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.device)))

    if torch.cuda.is_available():
        cuda_idx = int(local_rank) if world_size > 1 else int(args.device)
        torch.cuda.set_device(cuda_idx)
        device = torch.device("cuda", cuda_idx)
    else:
        device = torch.device("cpu")

    _seed_everything(int(args.seed) + (rank * 100003))
    is_main = rank == 0

    if int(args.group_size) <= 0:
        raise ValueError("--group-size must be > 0.")
    if world_size > 1 and (int(args.group_size) % int(world_size) != 0):
        raise ValueError(
            f"--group-size ({int(args.group_size)}) must be divisible by WORLD_SIZE ({int(world_size)}) for DDP training."
        )
    local_group_size = int(args.group_size) // int(world_size)
    local_group_size = max(1, local_group_size)

    init_ckpt: Optional[Path] = None
    if args.init_checkpoint is not None:
        init_ckpt = Path(args.init_checkpoint).resolve()
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-checkpoint not found: {init_ckpt}")

    run_name = str(args.run_name or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_root / run_name).resolve()
    ckpt_root = (args.ckpt_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    problems = _iter_selected_problems(_read_jsonl(args.dataset), int(args.max_problems))
    if not problems:
        raise RuntimeError("No training problems selected.")

    prompt_template = _select_prompt_template(str(args.prompt_template), bool(args.with_background))
    train_entries = _build_training_step_entries(
        problems=problems,
        with_background=bool(args.with_background),
        prompt_template=prompt_template,
        max_steps_per_problem=int(args.max_steps_per_problem),
        graph_root=(Path(args.graph_root).resolve() if args.graph_root is not None else None),
    )
    if not train_entries:
        raise RuntimeError("No GRPO training entries built.")

    model, tokenizer = _load_backbone(model_name=args.model_name, torch_dtype=args.torch_dtype, device=device)

    lora_cfg = LoRAConfig(
        num_experts=int(args.num_subtask_experts),
        top_k=int(args.subtask_top_k),
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        target_modules=("q_proj", "v_proj", "o_proj"),
        last_n_layers=int(args.lora_last_n_layers),
    )
    inject_mole_lora(model, cfg=lora_cfg)

    router_cfg = SubtaskRouterConfig(num_experts=int(args.num_subtask_experts), top_k=int(args.subtask_top_k))
    router = SubtaskRouter(router_cfg).to(device)
    title_embedder = TitleEmbedder(model=model, tokenizer=tokenizer, out_dim=router_cfg.title_emb_dim).to(device)

    opt_lora = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)],
        lr=float(args.lora_lr),
    )
    opt_router_params = [p for p in list(router.parameters()) + list(title_embedder.parameters()) if p.requires_grad]
    opt_router = torch.optim.AdamW(opt_router_params, lr=float(args.router_lr))

    loaded_state: Dict[str, Any] = {}
    if init_ckpt is not None:
        loaded_state = load_mole_checkpoint(
            ckpt_dir=init_ckpt,
            device=device,
            router=router,
            title_embedder=title_embedder,
            model=model,
            opt_router=None,
            opt_lora=None,
        )

    gen_cfg = GenerationConfig(
        model_name=args.model_name,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        torch_dtype=str(args.torch_dtype),
        device=int(device.index or 0),
    )
    mole_gen = MoLEGenerator(model=model, tokenizer=tokenizer, device=device, gen_cfg=gen_cfg)

    scipipe._check_eval_prerequisites(args.h5py_file)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not py_path else str(REPO_ROOT) + os.pathsep + py_path

    tmp_eval_root = run_dir / "train_step_eval_tmp"
    tmp_eval_root.mkdir(parents=True, exist_ok=True)
    detailed_log_root = run_dir / "detailed_logs"
    detailed_log_root.mkdir(parents=True, exist_ok=True)
    rank_detail_jsonl = detailed_log_root / f"rank_{rank:02d}_step_records.jsonl"
    snapshot_root = detailed_log_root / "snapshots"
    sample_perf_jsonl = detailed_log_root / "sample_performance.jsonl"

    history: List[dict] = []
    global_step = int(loaded_state.get("global_step", 0))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params.extend([p for p in router.parameters() if p.requires_grad])
    trainable_params.extend([p for p in title_embedder.parameters() if p.requires_grad])

    for epoch in range(1, int(args.epochs) + 1):
        epoch_entries = list(train_entries)
        random.Random(int(args.seed) + int(epoch)).shuffle(epoch_entries)

        for item_idx, (problem, step, step_id, prompt, oracle_ancestors, subtask_text) in enumerate(epoch_entries, start=1):
            title_emb = title_embedder(subtask_text)
            logits = router(title_emb=title_emb)

            cands: List[GRPOCandidate] = []
            sample_eval_dir = (
                tmp_eval_root
                / f"rank_{rank:02d}"
                / f"{_sanitize(str(problem.get('problem_id') or 'p'))}_{_sanitize(step_id)}"
            )
            sample_eval_dir.mkdir(parents=True, exist_ok=True)

            for local_idx in range(1, int(local_group_size) + 1):
                cand_idx = (rank * int(local_group_size)) + local_idx
                expert_ids, logp_router = router.sample_topk(logits)
                text, prompt_ids, gen_ids = mole_gen.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
                python_code = scipipe._extract_python_script(text)
                parsed_function = _extract_step_code_for_eval(step, python_code)

                reward, metrics, step_result = _evaluate_candidate_reward(
                    problem=problem,
                    step=step,
                    step_id=step_id,
                    python_code=python_code,
                    parsed_function=parsed_function,
                    oracle_ancestors=oracle_ancestors,
                    sample_eval_dir=sample_eval_dir,
                    h5py_file=args.h5py_file,
                    timeout_s=int(args.test_timeout_s),
                    env=env,
                    w_step=float(args.reward_w_step),
                    w_shape=float(args.reward_w_shape),
                    w_gt=float(args.reward_w_gt),
                    pass_bonus=float(args.reward_pass_bonus),
                )
                metrics["step_test_status"] = step_result.status if step_result is not None else "skipped"
                metrics["did_retry"] = False
                metrics["retry_count"] = 0
                metrics["attempt_count"] = 1
                snapshot_paths: Dict[str, str] = {}
                if bool(args.save_candidate_snapshots):
                    snapshot_paths = _write_candidate_snapshot(
                        snapshot_root=snapshot_root,
                        rank=int(rank),
                        global_step_next=int(global_step + 1),
                        item_idx=int(item_idx),
                        candidate_idx=int(cand_idx),
                        problem=problem,
                        step_id=str(step_id),
                        prompt=prompt,
                        response_text=text,
                        python_code=python_code,
                        parsed_function=parsed_function,
                        oracle_ancestors=oracle_ancestors,
                        metrics=metrics,
                        step_result=step_result,
                    )

                _append_jsonl(
                    rank_detail_jsonl,
                    {
                        "global_step_next": int(global_step + 1),
                        "epoch": int(epoch),
                        "item_idx": int(item_idx),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "problem_id": str(problem.get("problem_id") or ""),
                        "problem_name": str(problem.get("problem_name") or ""),
                        "step_id": str(step_id),
                        "candidate_idx_global": int(cand_idx),
                        "candidate_idx_local": int(local_idx),
                        "group_size_global": int(args.group_size),
                        "group_size_local": int(local_group_size),
                        "did_retry": False,
                        "retry_count": 0,
                        "attempt_count": 1,
                        "reward": float(reward),
                        "parse_ok": bool(metrics.get("parse_ok", False)),
                        "header_ok": bool(metrics.get("header_ok", False)),
                        "ast_ok": float(metrics.get("ast_ok", 0.0)),
                        "step_score": float(metrics.get("step_score", 0.0)),
                        "shape_score": float(metrics.get("shape_score", 0.0)),
                        "step_test_status": str(metrics.get("step_test_status", "skipped")),
                        "step_test_passed": bool(step_result.passed) if step_result is not None else False,
                        "step_test_return_code": int(step_result.return_code) if step_result is not None else None,
                        "step_test_elapsed_ms": int(step_result.elapsed_ms) if step_result is not None else None,
                        "step_test_script_path": str(step_result.script_path) if step_result is not None else "",
                        "step_test_stdout_tail": _tail_text(step_result.stdout) if step_result is not None else "",
                        "step_test_stderr_tail": _tail_text(step_result.stderr) if step_result is not None else "",
                        "sample_eval_dir": str(sample_eval_dir.resolve()),
                        "expert_ids": [int(x) for x in expert_ids.detach().cpu().tolist()],
                        **snapshot_paths,
                    },
                )

                cands.append(
                    GRPOCandidate(
                        idx=int(cand_idx),
                        text=text,
                        python_code=python_code,
                        parsed_function=parsed_function,
                        prompt_ids=prompt_ids,
                        gen_ids=gen_ids,
                        expert_ids=expert_ids,
                        logp_router=logp_router,
                        reward=float(reward),
                        metrics=metrics,
                    )
                )

            local_rewards = [float(c.reward) for c in cands]
            rewards = _all_gather_float_list(local_rewards, device=device, world_size=world_size)
            reward_mean = float(sum(rewards) / float(len(rewards))) if rewards else 0.0
            reward_std = float(_safe_std(rewards))
            step_scores = _all_gather_float_list(
                [float(c.metrics.get("step_score", 0.0)) for c in cands], device=device, world_size=world_size
            )
            shape_scores = _all_gather_float_list(
                [float(c.metrics.get("shape_score", 0.0)) for c in cands], device=device, world_size=world_size
            )
            parse_scores = _all_gather_float_list(
                [1.0 if bool(c.metrics.get("parse_ok", False)) else 0.0 for c in cands],
                device=device,
                world_size=world_size,
            )
            header_scores = _all_gather_float_list(
                [1.0 if bool(c.metrics.get("header_ok", False)) else 0.0 for c in cands],
                device=device,
                world_size=world_size,
            )
            ast_scores = _all_gather_float_list(
                [float(c.metrics.get("ast_ok", 0.0)) for c in cands], device=device, world_size=world_size
            )

            did_update = True
            if bool(args.grpo_skip_update_if_allzero) and rewards and (max(rewards) - min(rewards) == 0.0):
                did_update = False

            loss_value = 0.0
            if did_update:
                opt_router.zero_grad(set_to_none=True)
                opt_lora.zero_grad(set_to_none=True)
                total_loss = torch.tensor(0.0, device=device)
                has_grad_term = False

                for cand in cands:
                    adv = float(cand.reward) - float(reward_mean)
                    if bool(args.grpo_adv_normalize):
                        adv = adv / float(reward_std + float(args.grpo_adv_eps)) if reward_std > 0.0 else 0.0
                    adv = _clip_advantage(float(adv), float(args.advantage_clip))
                    if float(adv) == 0.0:
                        continue

                    logp_mole_sum = mole_gen.logprob_of_generation(
                        prompt_ids=cand.prompt_ids,
                        gen_ids=cand.gen_ids,
                        expert_ids=cand.expert_ids,
                    )
                    gen_len = max(1.0, float(cand.gen_ids.numel()))
                    logp_mole_mean = logp_mole_sum / gen_len

                    loss_i = -(torch.tensor(float(adv), device=device) * logp_mole_mean)
                    loss_router_i = -(torch.tensor(float(adv), device=device) * float(args.alpha_router) * cand.logp_router)
                    total_loss = total_loss + loss_i + loss_router_i
                    has_grad_term = True

                local_has_grad = 1 if has_grad_term else 0
                global_has_grad = local_has_grad
                if world_size > 1:
                    has_grad_t = torch.tensor([float(local_has_grad)], dtype=torch.float32, device=device)
                    dist.all_reduce(has_grad_t, op=dist.ReduceOp.SUM)
                    global_has_grad = 1 if has_grad_t.item() > 0.0 else 0

                if has_grad_term:
                    total_loss.backward()
                if global_has_grad:
                    _average_gradients(trainable_params, world_size)
                    opt_router.step()
                    opt_lora.step()
                    loss_value = float(total_loss.detach().cpu().item())
                    if world_size > 1:
                        loss_t = torch.tensor([loss_value], dtype=torch.float32, device=device)
                        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                        loss_value = float(loss_t.item() / float(world_size))

            best = max(cands, key=lambda c: float(c.reward))
            best_reward, best_step_score, best_shape_score, _ = _global_best_metrics(
                local_best_reward=float(best.reward),
                local_best_step=float(best.metrics.get("step_score", 0.0)),
                local_best_shape=float(best.metrics.get("shape_score", 0.0)),
                local_best_gt=0.0,
                device=device,
                world_size=world_size,
            )
            global_step += 1
            sample_perf = {
                "global_step": int(global_step),
                "epoch": int(epoch),
                "item_idx": int(item_idx),
                "problem_id": str(problem.get("problem_id") or ""),
                "problem_name": str(problem.get("problem_name") or ""),
                "step_id": str(step_id),
                "num_candidates": int(len(rewards)),
                "did_retry": False,
                "retry_count": 0,
                "reward_mean": float(reward_mean),
                "reward_std": float(reward_std),
                "reward_max": float(max(rewards)) if rewards else 0.0,
                "reward_min": float(min(rewards)) if rewards else 0.0,
                "step_pass_rate": float(sum(step_scores) / float(len(step_scores))) if step_scores else 0.0,
                "shape_score_mean": float(sum(shape_scores) / float(len(shape_scores))) if shape_scores else 0.0,
                "parse_ok_rate": float(sum(parse_scores) / float(len(parse_scores))) if parse_scores else 0.0,
                "header_ok_rate": float(sum(header_scores) / float(len(header_scores))) if header_scores else 0.0,
                "ast_ok_rate": float(sum(ast_scores) / float(len(ast_scores))) if ast_scores else 0.0,
                "best_reward": float(best_reward),
                "best_step_score": float(best_step_score),
                "best_shape_score": float(best_shape_score),
                "did_update": bool(did_update),
                "loss": float(loss_value),
            }
            if is_main:
                row = {
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "item_idx": int(item_idx),
                    "problem_id": str(problem.get("problem_id") or ""),
                    "problem_name": str(problem.get("problem_name") or ""),
                    "step_id": str(step_id),
                    "reward_mean": float(reward_mean),
                    "reward_std": float(reward_std),
                    "best_reward": float(best_reward),
                    "best_step_score": float(best_step_score),
                    "best_shape_score": float(best_shape_score),
                    "did_update": bool(did_update),
                    "did_retry": False,
                    "retry_count": 0,
                    "loss": float(loss_value),
                    "group_rewards": rewards,
                }
                history.append(row)
                _append_jsonl(sample_perf_jsonl, sample_perf)

            if int(args.save_every_steps) > 0 and (global_step % int(args.save_every_steps) == 0):
                if is_main:
                    save_mole_checkpoint(
                        ckpt_dir=ckpt_root / "step_latest",
                        router=router,
                        title_embedder=title_embedder,
                        model=model,
                        opt_router=opt_router,
                        opt_lora=opt_lora,
                        trainer_state={
                            "run_name": run_name,
                            "global_step": int(global_step),
                            "epoch": int(epoch),
                        },
                    )
                if world_size > 1:
                    dist.barrier()

        if is_main:
            _write_json(run_dir / f"train_epoch_{epoch:03d}.json", {"epoch": epoch, "global_step": global_step})
            # Rolling checkpoint updated every epoch.
            save_mole_checkpoint(
                ckpt_dir=ckpt_root / "last",
                router=router,
                title_embedder=title_embedder,
                model=model,
                opt_router=opt_router,
                opt_lora=opt_lora,
                trainer_state={
                    "run_name": run_name,
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                },
            )
        if world_size > 1:
            dist.barrier()
        if int(args.save_every_epochs) > 0 and (epoch % int(args.save_every_epochs) == 0):
            if is_main:
                save_mole_checkpoint(
                    ckpt_dir=ckpt_root / f"epoch_{epoch:03d}",
                    router=router,
                    title_embedder=title_embedder,
                    model=model,
                    opt_router=opt_router,
                    opt_lora=opt_lora,
                    trainer_state={
                        "run_name": run_name,
                        "global_step": int(global_step),
                        "epoch": int(epoch),
                    },
                )
            if world_size > 1:
                dist.barrier()

    if is_main:
        save_mole_checkpoint(
            ckpt_dir=ckpt_root / "final",
            router=router,
            title_embedder=title_embedder,
            model=model,
            opt_router=opt_router,
            opt_lora=opt_lora,
            trainer_state={
                "run_name": run_name,
                "global_step": int(global_step),
                "epochs": int(args.epochs),
            },
        )

        _write_json(run_dir / "train_history.json", {"rows": history})
        _write_json(
            run_dir / "train_summary.json",
            {
                "run_name": run_name,
                "num_entries": len(train_entries),
                "epochs": int(args.epochs),
                "global_step": int(global_step),
                "checkpoint_final": str((ckpt_root / "final").resolve()),
                "init_checkpoint": (str(init_ckpt) if init_ckpt is not None else ""),
                "train_init_mode": ("resume_from_checkpoint" if init_ckpt is not None else "scratch"),
                "graph_root": (str(Path(args.graph_root).resolve()) if args.graph_root is not None else ""),
                "world_size": int(world_size),
                "group_size": int(args.group_size),
                "local_group_size": int(local_group_size),
                "detailed_step_log_dir": str(detailed_log_root.resolve()),
                "detailed_step_log_files": [
                    str((detailed_log_root / f"rank_{r:02d}_step_records.jsonl").resolve()) for r in range(world_size)
                ],
                "sample_performance_log": str(sample_perf_jsonl.resolve()),
                "snapshot_root": str(snapshot_root.resolve()),
                "save_candidate_snapshots": bool(args.save_candidate_snapshots),
            },
        )
    if world_size > 1:
        dist.barrier()

    if bool(args.eval_after_train) and is_main:
        eval_gen_cfg = GenerationConfig(
            model_name=args.model_name,
            max_new_tokens=int(args.eval_max_new_tokens),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            torch_dtype=str(args.torch_dtype),
            device=int(device.index or 0),
        )
        eval_gen = MoLEGenerator(model=model, tokenizer=tokenizer, device=device, gen_cfg=eval_gen_cfg)
        eval_summary = evaluate_all_training_samples(
            problems=problems,
            run_dir=run_dir,
            mole_gen=eval_gen,
            router=router,
            title_embedder=title_embedder,
            with_background=bool(args.with_background),
            prompt_template_mode=str(args.prompt_template),
            h5py_file=args.h5py_file,
            test_timeout_s=int(args.test_timeout_s),
        )
        _write_json(run_dir / "eval_summary.json", eval_summary)

    if is_main:
        print(str(run_dir))
        print(str((ckpt_root / "final").resolve()))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SciCode MoLE GRPO training from SFT checkpoint.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--graph-root",
        type=Path,
        default=None,
        help="Optional task-graph root (sample dirs with task_graph.json + sample.json); when provided, use graph ancestors instead of chain.",
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--run-name", type=str, default="")

    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional path to SFT (or previous GRPO) checkpoint directory. If omitted, train from scratch.",
    )

    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--torch-dtype", type=str, default="bfloat16")

    p.add_argument("--num-subtask-experts", type=int, default=4)
    p.add_argument("--subtask-top-k", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-last-n-layers", type=int, default=8)

    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--router-lr", type=float, default=5e-5)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Global candidates per GRPO update. Under torchrun, this is split evenly across ranks.",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every-steps", type=int, default=50)
    p.add_argument("--save-every-epochs", type=int, default=0, help="Save checkpoint at every N epochs (0 disables).")

    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--max-steps-per-problem", type=int, default=0)

    p.add_argument("--with-background", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--prompt-template",
        choices=["auto", "multistep", "background_comment"],
        default="auto",
    )

    p.add_argument("--alpha-router", type=float, default=0.2)
    p.add_argument("--grpo-adv-normalize", action="store_true")
    p.add_argument("--grpo-adv-eps", type=float, default=1e-6)
    p.add_argument("--advantage-clip", type=float, default=5.0)
    p.add_argument("--grpo-skip-update-if-allzero", action="store_true")

    p.add_argument("--reward-w-step", type=float, default=0.75)
    p.add_argument("--reward-w-shape", type=float, default=0.15)
    p.add_argument("--reward-w-gt", type=float, default=0.10)
    p.add_argument("--reward-pass-bonus", type=float, default=0.10)

    p.add_argument("--h5py-file", type=Path, default=DEFAULT_H5PY_FILE)
    p.add_argument("--test-timeout-s", type=int, default=180)
    p.add_argument(
        "--save-candidate-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-candidate snapshot files (prompt/response/python/assembled/test result) under run_dir/detailed_logs/snapshots.",
    )

    p.add_argument("--eval-after-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-max-new-tokens", type=int, default=2048)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-top-p", type=float, default=1.0)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        train_grpo(args)
    finally:
        if _is_dist_ready():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
