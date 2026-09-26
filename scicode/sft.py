from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_PATH = Path(__file__).resolve()
SCICODE_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCICODE_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morse.mole.mole_generator import GenerationConfig, MoLEGenerator  # noqa: E402
from morse.mole.mole_lora import LoRAConfig, inject_mole_lora, lora_parameters, set_active_experts  # noqa: E402
from morse.mole.router import SubtaskRouter, SubtaskRouterConfig, normalize_title_embedding  # noqa: E402

from scicode import pipeline as scipipe  # noqa: E402


PROMPT_TEMPLATE_MULTISTEP = (SCICODE_ROOT / "eval" / "data" / "multistep_template.txt").read_text(encoding="utf-8")
PROMPT_TEMPLATE_BG_COMMENT = (SCICODE_ROOT / "eval" / "data" / "background_comment_template.txt").read_text(encoding="utf-8")

DEFAULT_DATASET = SCICODE_ROOT / "data" / "problems_dev.jsonl"
DEFAULT_H5PY_FILE = SCICODE_ROOT / "eval" / "data" / "test_data.h5"
DEFAULT_RUN_ROOT = SCICODE_ROOT / "runs" / "scicode_mole_sft_runs"
DEFAULT_CKPT_ROOT = SCICODE_ROOT / "checkpoints"


@dataclass
class StepSFTSample:
    problem_id: str
    problem_name: str
    step_id: str
    subtask_text: str
    prompt: str
    target_code: str


class TitleEmbedder(nn.Module):
    """Embed subtask text using frozen token embeddings + trainable projection."""

    def __init__(self, *, model: nn.Module, tokenizer, out_dim: int):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        hidden = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
        if hidden <= 0:
            raise ValueError("Could not determine hidden size for TitleEmbedder.")
        self.proj = nn.Linear(hidden, int(out_dim))

    @torch.no_grad()
    def _mean_token_emb(self, text: str) -> torch.Tensor:
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
        input_ids = enc["input_ids"].to(self._model.device)
        emb = self._model.get_input_embeddings()(input_ids)
        if self._tokenizer.pad_token_id is None:
            return emb.mean(dim=1)
        mask = (input_ids != self._tokenizer.pad_token_id).float().unsqueeze(-1)
        den = torch.clamp(mask.sum(dim=1), min=1.0)
        return (emb * mask).sum(dim=1) / den

    def forward(self, text: str) -> torch.Tensor:
        mean_emb = self._mean_token_emb(text)
        out = self.proj(mean_emb)
        return normalize_title_embedding(out)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sanitize(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_") or "item"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_prompt_template(mode: str, with_background: bool) -> str:
    if mode == "multistep":
        return PROMPT_TEMPLATE_MULTISTEP
    if mode == "background_comment":
        return PROMPT_TEMPLATE_BG_COMMENT
    return PROMPT_TEMPLATE_MULTISTEP if with_background else PROMPT_TEMPLATE_BG_COMMENT


def _iter_selected_problems(problems: List[dict], max_problems: int) -> List[dict]:
    if max_problems > 0:
        return problems[:max_problems]
    return problems


def _ordered_step_ids(step_order: Dict[str, int]) -> List[str]:
    return [sid for sid, _ in sorted(step_order.items(), key=lambda item: item[1])]


def _build_sft_samples(
    *,
    problems: List[dict],
    with_background: bool,
    prompt_template: str,
    max_steps_per_problem: int,
) -> List[StepSFTSample]:
    out: List[StepSFTSample] = []
    for problem in problems:
        problem_id = str(problem.get("problem_id") or "")
        problem_name = str(problem.get("problem_name") or f"problem_{problem_id}")
        step_by_id, step_order = scipipe._build_step_maps(problem)
        ordered_ids = _ordered_step_ids(step_order)
        gt_so_far: Dict[str, str] = {}

        if max_steps_per_problem > 0:
            ordered_ids = ordered_ids[:max_steps_per_problem]

        for step_id in ordered_ids:
            step = step_by_id[step_id]
            target_code = str(step.get("ground_truth_code") or "").strip()
            if not target_code:
                continue
            ancestor_ids = [sid for sid in ordered_ids if step_order[sid] < step_order[step_id] and sid in gt_so_far]
            prompt, _ = scipipe._render_prompt(
                problem=problem,
                step=step,
                ancestor_step_ids=ancestor_ids,
                solved_functions=gt_so_far,
                step_by_id=step_by_id,
                with_background=with_background,
                prompt_template=prompt_template,
            )
            desc = str(step.get("step_description_prompt") or "").strip()
            bg = str(step.get("step_background") or "").strip()
            subtask_text = (desc + "\n" + bg).strip() if bg else desc
            out.append(
                StepSFTSample(
                    problem_id=problem_id,
                    problem_name=problem_name,
                    step_id=step_id,
                    subtask_text=subtask_text or f"{problem_name}:{step_id}",
                    prompt=prompt,
                    target_code=target_code,
                )
            )
            gt_so_far[step_id] = target_code
    return out


def _build_training_tensors(
    *,
    tokenizer,
    prompt: str,
    target_code: str,
    max_length: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    target_text = "\n\n```python\n" + target_code.strip() + "\n```"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    if not target_ids:
        return None

    if len(prompt_ids) + len(target_ids) > max_length:
        keep_prompt = max(64, max_length - len(target_ids))
        prompt_ids = prompt_ids[-keep_prompt:]

    if len(prompt_ids) + len(target_ids) > max_length:
        keep_target = max(8, max_length - len(prompt_ids))
        target_ids = target_ids[:keep_target]

    if len(prompt_ids) + len(target_ids) <= 1:
        return None

    full = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids[:]

    input_ids = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
    return input_ids, attention_mask, labels_t


def _load_backbone(*, model_name: str, torch_dtype: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    try:
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
    except Exception:
        pass
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    except Exception:
        pass
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


def _collect_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_A" in name or "lora_B" in name):
            state[name] = p.detach().cpu()
    return state


def save_mole_checkpoint(
    *,
    ckpt_dir: Path,
    router: nn.Module,
    title_embedder: nn.Module,
    model: nn.Module,
    opt_router: Optional[torch.optim.Optimizer],
    opt_lora: Optional[torch.optim.Optimizer],
    trainer_state: Dict[str, Any],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), ckpt_dir / "router.pt")
    # Only persist the trainable `proj` submodule. TitleEmbedder also stores a
    # reference to the full backbone (`self._model`), which PyTorch registers
    # as a child module regardless of the underscore prefix, so saving the full
    # state_dict would dump ~16 GB of frozen base-LM weights every time and
    # repeatedly trigger GPFS stream-writer failures during sample-level saves.
    torch.save(title_embedder.proj.state_dict(), ckpt_dir / "title_embedder.pt")
    torch.save(_collect_lora_state(model), ckpt_dir / "lora_state.pt")
    if opt_router is not None:
        torch.save(opt_router.state_dict(), ckpt_dir / "opt_router.pt")
    if opt_lora is not None:
        torch.save(opt_lora.state_dict(), ckpt_dir / "opt_lora.pt")
    torch.save(trainer_state, ckpt_dir / "trainer_state.pt")


def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_mole_checkpoint(
    *,
    ckpt_dir: Path,
    device: torch.device,
    router: nn.Module,
    title_embedder: nn.Module,
    model: nn.Module,
    opt_router: Optional[torch.optim.Optimizer],
    opt_lora: Optional[torch.optim.Optimizer],
) -> Dict[str, Any]:
    router.load_state_dict(torch.load(ckpt_dir / "router.pt", map_location=device))
    te_sd = torch.load(ckpt_dir / "title_embedder.pt", map_location=device)
    if isinstance(te_sd, dict) and any(
        k.startswith("_model.") or k.startswith("proj.") for k in te_sd.keys()
    ):
        # Backward-compat: legacy checkpoints stored the full TitleEmbedder
        # state_dict (proj.* + _model.*). Restore only proj.* and ignore the
        # frozen backbone dump.
        proj_sd = {
            k[len("proj."):]: v for k, v in te_sd.items() if k.startswith("proj.")
        }
        title_embedder.proj.load_state_dict(proj_sd)
    else:
        title_embedder.proj.load_state_dict(te_sd)

    lora_state = torch.load(ckpt_dir / "lora_state.pt", map_location="cpu")
    name_to_param = dict(model.named_parameters())
    for name, tensor in lora_state.items():
        p = name_to_param.get(name)
        if p is None:
            continue
        p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))

    if opt_router is not None and (ckpt_dir / "opt_router.pt").exists():
        opt_router.load_state_dict(torch.load(ckpt_dir / "opt_router.pt", map_location="cpu"))
        _move_optimizer_state_to_device(opt_router, device)
    if opt_lora is not None and (ckpt_dir / "opt_lora.pt").exists():
        opt_lora.load_state_dict(torch.load(ckpt_dir / "opt_lora.pt", map_location="cpu"))
        _move_optimizer_state_to_device(opt_lora, device)

    state_path = ckpt_dir / "trainer_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        return dict(state) if isinstance(state, dict) else {}
    return {}


def _extract_step_code_for_eval(step: dict, raw_python: str) -> str:
    python_code = raw_python.strip()
    if not python_code:
        return ""
    try:
        fn_name = scipipe._extract_function_name(str(step.get("function_header") or ""))
    except Exception:
        return python_code
    parsed = scipipe._get_function_from_code(python_code, fn_name)
    return (parsed or "").strip()


def _build_eval_prompt_template(mode: str, with_background: bool) -> str:
    return _select_prompt_template(mode, with_background)


def generate_step_code(
    *,
    mole_gen: MoLEGenerator,
    router: SubtaskRouter,
    title_embedder: TitleEmbedder,
    subtask_text: str,
    prompt: str,
) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    title_emb = title_embedder(subtask_text)
    logits = router(title_emb=title_emb)
    expert_ids, _ = router.greedy_topk(logits)
    text, prompt_ids, gen_ids = mole_gen.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
    python_code = scipipe._extract_python_script(text)
    return python_code, expert_ids, prompt_ids, gen_ids


def evaluate_all_training_samples(
    *,
    problems: List[dict],
    run_dir: Path,
    mole_gen: MoLEGenerator,
    router: SubtaskRouter,
    title_embedder: TitleEmbedder,
    with_background: bool,
    prompt_template_mode: str,
    h5py_file: Path,
    test_timeout_s: int,
) -> Dict[str, Any]:
    scipipe._check_eval_prerequisites(h5py_file)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not py_path else str(REPO_ROOT) + os.pathsep + py_path

    prompt_template = _build_eval_prompt_template(prompt_template_mode, with_background)
    rows: List[dict] = []
    eval_root = run_dir / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    for sample_idx, problem in enumerate(problems, start=1):
        problem_id = str(problem.get("problem_id") or sample_idx)
        sample_name = scipipe._sample_dir_name(sample_idx, problem)
        sample_dir = eval_root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        log_dir = sample_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        step_by_id, step_order = scipipe._build_step_maps(problem)
        ordered_ids = _ordered_step_ids(step_order)
        solved_functions: Dict[str, str] = {}
        solved_full: Dict[str, str] = {}
        failed = False
        step_passed = 0
        tested_steps = 0

        for node_idx, step_id in enumerate(ordered_ids):
            step = step_by_id[step_id]
            node_dir = log_dir / f"node_{node_idx:02d}_{_sanitize(step_id)}"
            node_dir.mkdir(parents=True, exist_ok=True)

            ancestor_step_ids = [sid for sid in ordered_ids if step_order[sid] < step_order[step_id] and sid in solved_functions]
            prompt, _ = scipipe._render_prompt(
                problem=problem,
                step=step,
                ancestor_step_ids=ancestor_step_ids,
                solved_functions=solved_functions,
                step_by_id=step_by_id,
                with_background=with_background,
                prompt_template=prompt_template,
            )
            (node_dir / "prompt_attempt_1.txt").write_text(prompt, encoding="utf-8")
            desc = str(step.get("step_description_prompt") or "").strip()
            bg = str(step.get("step_background") or "").strip()
            subtask_text = (desc + "\n" + bg).strip() if bg else desc

            python_code, expert_ids, _, _ = generate_step_code(
                mole_gen=mole_gen,
                router=router,
                title_embedder=title_embedder,
                subtask_text=subtask_text or f"{problem_id}:{step_id}",
                prompt=prompt,
            )
            (node_dir / "python_attempt_1.py").write_text(python_code + "\n", encoding="utf-8")
            (node_dir / "expert_ids_attempt_1.json").write_text(
                json.dumps([int(x) for x in expert_ids.detach().cpu().tolist()], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            parsed_function = _extract_step_code_for_eval(step, python_code)
            if not parsed_function:
                failed = True
                _write_json(
                    node_dir / "step_test_attempt_1.json",
                    {
                        "status": "fail",
                        "return_code": 1,
                        "elapsed_ms": 0,
                        "stdout_tail": "",
                        "stderr_tail": "empty parsed function code",
                        "script_path": "",
                    },
                )
                continue

            assembled = scipipe._assemble_program_code(
                dependencies=str(problem.get("required_dependencies") or ""),
                ancestor_step_ids=ancestor_step_ids,
                solved_functions=solved_functions,
                current_python_code=python_code,
            )
            (sample_dir / "generated_code").mkdir(parents=True, exist_ok=True)
            (sample_dir / "generated_code" / f"{step_id}.py").write_text(assembled, encoding="utf-8")

            tests = list(step.get("test_cases") or [])
            tested_steps += 1
            step_result = scipipe._run_step_test(
                sample_dir=sample_dir,
                step_id=step_id,
                assembled_code=assembled,
                test_cases=tests,
                h5py_file=h5py_file,
                timeout_s=test_timeout_s,
                env=env,
            )
            _write_json(
                node_dir / "step_test_attempt_1.json",
                {
                    "status": step_result.status,
                    "return_code": step_result.return_code,
                    "elapsed_ms": step_result.elapsed_ms,
                    "stdout_tail": step_result.stdout[-4000:],
                    "stderr_tail": step_result.stderr[-4000:],
                    "script_path": str(step_result.script_path),
                },
            )
            if step_result.passed:
                solved_functions[step_id] = parsed_function
                solved_full[step_id] = python_code.strip()
                step_passed += 1
            else:
                failed = True

        general_status = "skipped"
        general_result = None
        if solved_functions:
            final_ids = [sid for sid in ordered_ids if sid in solved_functions]
            final_code = (
                "\n\n".join(
                    [str(problem.get("required_dependencies") or "").strip()] + [solved_functions[sid] for sid in final_ids]
                ).strip()
                + "\n"
            )
            general_tests = list(problem.get("general_tests") or [])
            if general_tests:
                sub_steps = list(problem.get("sub_steps") or [])
                target_group = str(sub_steps[-1].get("step_number") or problem_id) if sub_steps else problem_id
                general_result = scipipe._run_general_test(
                    sample_dir=sample_dir,
                    problem_id=problem_id,
                    general_target_group=target_group,
                    assembled_code=final_code,
                    general_tests=general_tests,
                    h5py_file=h5py_file,
                    timeout_s=test_timeout_s,
                    env=env,
                )
                general_status = general_result.status
                _write_json(
                    log_dir / "general_test.json",
                    {
                        "status": general_result.status,
                        "return_code": general_result.return_code,
                        "elapsed_ms": general_result.elapsed_ms,
                        "stdout_tail": general_result.stdout[-4000:],
                        "stderr_tail": general_result.stderr[-4000:],
                        "script_path": str(general_result.script_path),
                    },
                )
                if not general_result.passed:
                    failed = True

        sample_metrics = {
            "problem_id": problem_id,
            "problem_name": str(problem.get("problem_name") or ""),
            "status": "ok" if not failed else "failed",
            "tested_steps": tested_steps,
            "step_passed": step_passed,
            "step_pass_rate": float(step_passed / tested_steps) if tested_steps else 0.0,
            "general_status": general_status,
        }
        _write_json(log_dir / "sample_metrics.json", sample_metrics)
        rows.append(sample_metrics)

    summary = scipipe._summarize_samples(rows)
    _write_json(eval_root / "summary.json", summary)
    _write_json(eval_root / "rows.json", {"rows": rows})
    return summary


def train_sft(args: argparse.Namespace) -> None:
    _seed_everything(int(args.seed))
    if args.gpus and str(args.gpus).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus).strip()

    device = torch.device("cuda", int(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    run_name = str(args.run_name or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_root / run_name).resolve()
    ckpt_root = (args.ckpt_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    problems = _iter_selected_problems(_read_jsonl(args.dataset), int(args.max_problems))
    if not problems:
        raise RuntimeError("No training problems selected.")

    prompt_template = _select_prompt_template(str(args.prompt_template), bool(args.with_background))
    samples = _build_sft_samples(
        problems=problems,
        with_background=bool(args.with_background),
        prompt_template=prompt_template,
        max_steps_per_problem=int(args.max_steps_per_problem),
    )
    if not samples:
        raise RuntimeError("No SFT samples produced from dataset (check ground_truth_code fields).")

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

    opt_lora = torch.optim.AdamW(list(lora_parameters(model)), lr=float(args.lora_lr))
    opt_router = torch.optim.AdamW(list(router.parameters()) + list(title_embedder.parameters()), lr=float(args.router_lr))

    global_step = 0
    losses: List[float] = []

    for epoch in range(1, int(args.epochs) + 1):
        epoch_samples = list(samples)
        random.shuffle(epoch_samples)
        epoch_loss = 0.0
        epoch_updates = 0

        for sample in epoch_samples:
            prepared = _build_training_tensors(
                tokenizer=tokenizer,
                prompt=sample.prompt,
                target_code=sample.target_code,
                max_length=int(args.max_length),
                device=device,
            )
            if prepared is None:
                continue
            input_ids, attention_mask, labels = prepared

            title_emb = title_embedder(sample.subtask_text)
            logits = router(title_emb=title_emb)

            pseudo_label = abs(hash(sample.subtask_text)) % int(args.num_subtask_experts)
            pseudo_target = torch.tensor([pseudo_label], dtype=torch.long, device=device)
            router_loss = F.cross_entropy(logits, pseudo_target)

            expert_ids = torch.topk(F.softmax(logits[0], dim=-1), k=min(int(args.subtask_top_k), int(args.num_subtask_experts))).indices
            set_active_experts(model, expert_ids)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            lm_loss = out.loss
            loss = lm_loss + float(args.router_supervision_weight) * router_loss

            opt_router.zero_grad(set_to_none=True)
            opt_lora.zero_grad(set_to_none=True)
            loss.backward()
            opt_router.step()
            opt_lora.step()

            loss_v = float(loss.detach().cpu().item())
            losses.append(loss_v)
            epoch_loss += loss_v
            epoch_updates += 1
            global_step += 1

            if int(args.save_every_steps) > 0 and (global_step % int(args.save_every_steps) == 0):
                save_mole_checkpoint(
                    ckpt_dir=ckpt_root / "step_latest",
                    router=router,
                    title_embedder=title_embedder,
                    model=model,
                    opt_router=opt_router,
                    opt_lora=opt_lora,
                    trainer_state={
                        "run_name": run_name,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                )

        epoch_mean = float(epoch_loss / epoch_updates) if epoch_updates else 0.0
        _write_json(
            run_dir / f"train_epoch_{epoch:03d}.json",
            {
                "epoch": epoch,
                "global_step": global_step,
                "num_updates": epoch_updates,
                "mean_loss": epoch_mean,
            },
        )

    save_mole_checkpoint(
        ckpt_dir=ckpt_root / "final",
        router=router,
        title_embedder=title_embedder,
        model=model,
        opt_router=opt_router,
        opt_lora=opt_lora,
        trainer_state={
            "run_name": run_name,
            "global_step": global_step,
            "epochs": int(args.epochs),
            "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        },
    )

    _write_json(
        run_dir / "train_summary.json",
        {
            "run_name": run_name,
            "num_samples": len(samples),
            "epochs": int(args.epochs),
            "global_step": global_step,
            "mean_loss": float(sum(losses) / len(losses)) if losses else 0.0,
            "checkpoint_final": str((ckpt_root / "final").resolve()),
        },
    )

    if bool(args.eval_after_train):
        gen_cfg = GenerationConfig(
            model_name=args.model_name,
            max_new_tokens=int(args.eval_max_new_tokens),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            torch_dtype=str(args.torch_dtype),
            device=int(args.device),
        )
        mole_gen = MoLEGenerator(model=model, tokenizer=tokenizer, device=device, gen_cfg=gen_cfg)
        eval_summary = evaluate_all_training_samples(
            problems=problems,
            run_dir=run_dir,
            mole_gen=mole_gen,
            router=router,
            title_embedder=title_embedder,
            with_background=bool(args.with_background),
            prompt_template_mode=str(args.prompt_template),
            h5py_file=args.h5py_file,
            test_timeout_s=int(args.test_timeout_s),
        )
        _write_json(run_dir / "eval_summary.json", eval_summary)

    print(str(run_dir))
    print(str((ckpt_root / "final").resolve()))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SciCode MoLE SFT training + checkpoint + eval.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--run-name", type=str, default="")

    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--torch-dtype", type=str, default="bfloat16")

    p.add_argument("--num-subtask-experts", type=int, default=4)
    p.add_argument("--subtask-top-k", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-last-n-layers", type=int, default=8)

    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--router-lr", type=float, default=1e-4)
    p.add_argument("--router-supervision-weight", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--save-every-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-problems", type=int, default=0, help="0 means all problems in dataset")
    p.add_argument("--max-steps-per-problem", type=int, default=0, help="0 means all steps")

    p.add_argument("--with-background", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--prompt-template",
        choices=["auto", "multistep", "background_comment"],
        default="auto",
        help="Prompt template used for SFT input rendering.",
    )

    p.add_argument("--eval-after-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--h5py-file", type=Path, default=DEFAULT_H5PY_FILE)
    p.add_argument("--test-timeout-s", type=int, default=180)
    p.add_argument("--eval-max-new-tokens", type=int, default=2048)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-top-p", type=float, default=1.0)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    train_sft(args)


if __name__ == "__main__":
    main()
