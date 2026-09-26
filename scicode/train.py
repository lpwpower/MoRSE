from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

SCRIPT_PATH = Path(__file__).resolve()
SCICODE_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCICODE_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morse.mole.mole_generator import GenerationConfig, MoLEGenerator  # noqa: E402
from morse.mole.mole_lora import LoRAConfig, inject_mole_lora  # noqa: E402
from morse.mole.router import SubtaskRouter, SubtaskRouterConfig, normalize_title_embedding  # noqa: E402

from scicode import pipeline as scipipe  # noqa: E402
from scicode import pipeline_srddstyle as srddpipe  # noqa: E402
from scicode import grpo_base  # noqa: E402
from scicode.sft import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATASET,
    DEFAULT_H5PY_FILE,
    SCICODE_ROOT as _SCICODE_ROOT_SENTINEL,
    TitleEmbedder,
    _extract_step_code_for_eval,
    _iter_selected_problems,
    _load_backbone,
    _read_jsonl,
    _seed_everything,
    _write_json,
    evaluate_all_training_samples,
    load_mole_checkpoint,
    save_mole_checkpoint,
)

if _SCICODE_ROOT_SENTINEL != SCICODE_ROOT:
    raise RuntimeError("SCICODE_ROOT mismatch between SFT and SRDD-style GRPO scripts.")

DEFAULT_RUN_ROOT = SCICODE_ROOT / "runs" / "scicode_mole_grpo_srddstyle_runs"
DEFAULT_CURRICULUM_DIFFICULTY_FILE = SCICODE_ROOT / "data" / "domain_difficulty_stats.tsv"

DIFFICULTY_RANK: Dict[str, int] = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "very_hard": 3,
    "unknown": 4,
}

ROLE_EXPERT_IDS: Dict[str, int] = {
    "execute": 0,
    "aggregate": 1,
}


@dataclass
class SRDDStyleEntry:
    kind: str  # execute | aggregate
    problem: dict
    step: dict
    step_id: str
    prompt: str
    subtask_text: str
    oracle_ancestors: Dict[str, str]
    expected_step_ids: List[str]
    parent_step_ids: List[str]
    parent_count: int
    target_gt_code: str


@dataclass
class SRDDStyleCandidate:
    idx: int
    route_local_idx: int
    route_global_idx: int
    text: str
    python_code: str
    parsed_function: str
    prompt_ids: torch.Tensor
    gen_ids: torch.Tensor
    expert_ids: torch.Tensor
    logp_router: torch.Tensor
    role_expert_id: int
    subtask_expert_ids: List[int]
    tf_target_ids: torch.Tensor
    tf_enabled: bool
    reward: float
    metrics: Dict[str, Any]


@dataclass
class CandidateEvalResult:
    reward: float
    metrics: Dict[str, Any]
    step_result: Optional[scipipe.ScriptRunResult]
    assembled_code: str



def _tail_text(value: str, max_chars: int = 1200) -> str:
    text = str(value or "")
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]



def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")



def _safe_mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.resolve())
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return str(value)


def _summarize_dataset_gt_stats(problems: Sequence[dict]) -> Dict[str, Any]:
    num_problems = int(len(problems))
    num_problems_with_gt = 0
    num_steps = 0
    num_steps_with_gt = 0
    for problem in problems:
        has_problem_gt = False
        for step in list(problem.get("sub_steps") or []):
            num_steps += 1
            if str(step.get("ground_truth_code") or "").strip():
                num_steps_with_gt += 1
                has_problem_gt = True
        if has_problem_gt:
            num_problems_with_gt += 1
    return {
        "num_problems": int(num_problems),
        "num_problems_with_gt": int(num_problems_with_gt),
        "num_steps": int(num_steps),
        "num_steps_with_gt": int(num_steps_with_gt),
        "problem_gt_rate": (
            float(num_problems_with_gt) / float(num_problems) if num_problems > 0 else 0.0
        ),
        "step_gt_rate": (float(num_steps_with_gt) / float(num_steps) if num_steps > 0 else 0.0),
    }


def _apply_disable_gt_code_signals(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "disable_gt_code_signals", False)):
        return
    # One switch to hard-disable every GT-dependent reward/loss path.
    args.reward_w_gt = 0.0
    args.enable_gt_stage_schedule = False
    args.stage_wgt_zero_epoch = 1
    args.stage_wgt_epoch1 = 0.0
    args.stage_wgt_epoch2 = 0.0

    args.enable_tf_ce = False
    args.tf_ce_weight = 0.0
    args.enable_tf_ce_stage_schedule = False
    args.tf_ce_zero_epoch = 1
    args.tf_ce_epoch1 = 0.0
    args.tf_ce_epoch2 = 0.0
    args.tf_ce_include_aggregate = False

    args.enable_tf_reward = False
    args.tf_reward_weight = 0.0


def _srdd_step_text(step: dict, *, with_background: bool) -> str:
    desc = str(step.get("step_description_prompt") or "").strip()
    if with_background:
        bg = str(step.get("step_background") or "").strip()
        if bg:
            return f"{desc}\n{bg}".strip()
    return desc


def _build_router_subtask_text(
    *,
    entry_kind: str,
    problem: dict,
    step_id: str,
    step: dict,
    node_title: str,
    with_background: bool,
    aggregate_subtask_prefix: str,
    router_text_source: str,
) -> str:
    source = str(router_text_source or "description").strip().lower()
    problem_id = str(problem.get("problem_id") or "").strip()
    desc = str(step.get("step_description_prompt") or "").strip()
    bg = str(step.get("step_background") or "").strip()
    title = str(node_title or "").strip()
    header = str(step.get("function_header") or "").strip().splitlines()[0] if str(step.get("function_header") or "").strip() else ""

    if entry_kind == "aggregate":
        prefix = (
            f"{aggregate_subtask_prefix} "
            f"problem {problem_id} step {step_id}"
        ).strip()
        if source == "title":
            return (f"{prefix}\n{title}" if title else prefix).strip()
        if source == "title_header":
            parts = [prefix]
            if title:
                parts.append(title)
            if header:
                parts.append(header)
            return "\n".join(parts).strip()
        return prefix

    if source == "title":
        return title or desc or f"{problem_id}:{step_id}:execute"
    if source == "title_header":
        parts = [title or desc]
        if header:
            parts.append(header)
        return "\n".join([p for p in parts if p]).strip() or f"{problem_id}:{step_id}:execute"

    # Default: rich semantic description with optional background.
    return _srdd_step_text(step, with_background=with_background) or f"{problem_id}:{step_id}:execute"


def _apply_prefill_python_fence(prompt: str, *, enabled: bool) -> str:
    if not enabled:
        return str(prompt or "").rstrip() + "\n"
    return str(prompt or "").rstrip() + "\n\n```python\n"


def _build_execute_prompt_nonchat(
    *,
    problem: dict,
    step: dict,
    step_id: str,
    ancestor_step_ids: Sequence[str],
    solved_functions: Dict[str, str],
    step_by_id: Dict[str, dict],
    with_background: bool,
    prompt_style: str,
    prefill_python_fence: bool,
) -> str:
    problem_id = str(problem.get("problem_id") or "").strip()
    problem_name = str(problem.get("problem_name") or f"problem_{problem_id or 'unknown'}").strip()
    problem_desc = str(problem.get("problem_description_main") or "").strip()
    problem_bg = str(problem.get("problem_background_main") or "").strip()
    deps = str(problem.get("required_dependencies") or "").strip()

    step_desc = _srdd_step_text(step, with_background=with_background)
    step_header = str(step.get("function_header") or "").strip()
    step_return = str(step.get("return_line") or "").strip()

    upstream_blocks: List[str] = []
    for sid in ancestor_step_ids:
        prev_step = step_by_id.get(str(sid))
        if prev_step is None:
            continue
        prev_header = str(prev_step.get("function_header") or "").strip()
        prev_code = str(solved_functions.get(str(sid)) or "").strip()
        if not prev_code:
            continue
        block = [f"Step ID: {sid}"]
        if prev_header:
            block.extend(["Function Header:", prev_header])
        block.extend(["Solved Function Code (reference only):", prev_code])
        upstream_blocks.append("\n".join(block).strip())
    upstream_text = "\n\n-----\n\n".join(upstream_blocks) if upstream_blocks else "(none)"

    background_section = f"\nProblem background:\n{problem_bg}\n" if problem_bg else ""
    prompt = textwrap.dedent(
        f"""
        You are solving one SciCode step in a single response.

        Problem name: {problem_name}
        Problem id: {problem_id}
        Main description:
        {problem_desc}
        {background_section}
        Required dependencies:
        {deps}

        Previously solved steps (reference only; do NOT output these functions):
        {upstream_text}

        Current step to implement:
        Step ID: {step_id}
        Description:
        {step_desc}

        Function header:
        {step_header}
        """
    ).strip()
    if step_return:
        prompt += f"\n\nReturn line:\n{step_return}"

    prompt += "\n\n" + textwrap.dedent(
        """

        Requirements:
        - Implement the current step only (you may add small local helpers if needed).
        - Do not output previous-step function code.
        - Use only the required dependencies listed above.
        - Do not include test code or example usage.
        - Output format is strict: output exactly one markdown Python code block.
        - Do not output any text before the code block.
        - Do not output any text after the code block.
        - Do not include explanations, reasoning, or instruction discussion anywhere in the response.
        """
    ).strip()
    if str(prompt_style).strip().lower() == "minimal":
        prompt += "\n\n" + textwrap.dedent(
            """

            Additional efficiency constraints:
            - Prefer concise implementations; avoid long comments and long docstrings.
            - Keep code minimal but complete for the required function header.
            """
        ).strip()
    return _apply_prefill_python_fence(prompt, enabled=bool(prefill_python_fence))


def _build_aggregate_prompt_nonchat(
    *,
    problem: dict,
    next_step_id: str,
    expected_step_ids: Sequence[str],
    step_by_id: Dict[str, dict],
    parent_snapshots: Sequence[Tuple[int, Dict[str, str]]],
    prompt_style: str,
    prefill_python_fence: bool,
) -> str:
    deps = str(problem.get("required_dependencies") or "").strip()
    blocks: List[str] = []
    for parent_id, snapshot in parent_snapshots:
        chunk = [f"Parent node {parent_id}"]
        if not snapshot:
            chunk.append("(empty snapshot)")
            blocks.append("\n".join(chunk))
            continue
        for sid in expected_step_ids:
            code = str(snapshot.get(sid) or "").strip()
            if not code:
                continue
            chunk.append(f"# step {sid}")
            chunk.append(code)
            chunk.append("------")
        blocks.append("\n".join(chunk).strip())
    snapshots_text = "\n\n".join(blocks) if blocks else "(none)"

    expected_lines: List[str] = []
    for sid in expected_step_ids:
        header = ""
        step = step_by_id.get(str(sid))
        if step is not None:
            header = str(step.get("function_header") or "").strip().splitlines()[0]
        expected_lines.append(f"- step {sid}: {header}")
    expected_text = "\n".join(expected_lines) if expected_lines else "(none)"

    prompt = textwrap.dedent(
        f"""
        You are the aggregate agent for SciCode task-graph execution.
        Merge multiple parent snapshots into ONE coherent upstream function set.

        Required dependencies:
        {deps}

        Next node step id: {next_step_id}

        Expected upstream steps to preserve:
        {expected_text}

        Parent snapshots:
        {snapshots_text}

        Requirements:
        - Return merged upstream functions only.
        - Preserve function signatures compatible with provided function headers.
        - Do not add tests or example usage.
        - Output format is strict: output exactly one markdown Python code block.
        - Do not output any text before the code block.
        - Do not output any text after the code block.
        - Do not include explanations, reasoning, or instruction discussion anywhere in the response.
        """
    ).strip()
    if str(prompt_style).strip().lower() == "minimal":
        prompt += "\n\n" + textwrap.dedent(
            """

            Additional efficiency constraints:
            - Keep only the required upstream function/class implementations.
            - Prefer concise code and avoid long comments/docstrings.
            """
        ).strip()
    return _apply_prefill_python_fence(prompt, enabled=bool(prefill_python_fence))


def _subtask_router_reg_loss(router: SubtaskRouter, l2_weight: float, ortho_weight: float) -> torch.Tensor:
    device = router.prototypes.device
    l2 = torch.tensor(0.0, device=device)
    ortho = torch.tensor(0.0, device=device)
    if float(l2_weight) > 0.0:
        l2 = (router.prototypes ** 2).mean()
    if float(ortho_weight) > 0.0:
        proto = normalize_title_embedding(router.prototypes)
        gram = proto @ proto.t()
        ident = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        ortho = ((gram - ident) ** 2).mean()
    return (float(l2_weight) * l2) + (float(ortho_weight) * ortho)


def _title_embedder_trainable_parameters(title_embedder: TitleEmbedder) -> List[torch.nn.Parameter]:
    """Return only TitleEmbedder-owned trainable params (exclude attached backbone submodule params)."""
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()

    proj = getattr(title_embedder, "proj", None)
    if isinstance(proj, nn.Module):
        for p in proj.parameters():
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))

    if not params:
        for name, p in title_embedder.named_parameters():
            if name.startswith("_model."):
                continue
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
    return params


def _unique_trainable_parameters(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    out: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for p in params:
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _resolve_local_route_layout(*, local_group_size: int, requested_local_routes: int) -> Tuple[int, int]:
    """Compute (num_routes_local, candidates_per_route_local) for hierarchical credit assignment."""
    if int(local_group_size) <= 0:
        raise ValueError("local_group_size must be > 0.")

    if int(requested_local_routes) <= 0:
        # Auto: keep at least 2 samples per route when possible.
        route_count = max(1, int(local_group_size) // 2)
    else:
        route_count = int(requested_local_routes)

    if route_count > int(local_group_size):
        raise ValueError(
            f"hierarchical-local-routes ({route_count}) cannot exceed local_group_size ({int(local_group_size)})."
        )
    if int(local_group_size) % int(route_count) != 0:
        raise ValueError(
            "local_group_size must be divisible by hierarchical-local-routes. "
            f"Got local_group_size={int(local_group_size)}, hierarchical-local-routes={int(route_count)}."
        )
    per_route = int(local_group_size) // int(route_count)
    if per_route <= 0:
        raise ValueError("Invalid per-route candidate count computed for hierarchical layout.")
    return int(route_count), int(per_route)


def _select_execute_experts(
    *,
    router: SubtaskRouter,
    title_embedder: TitleEmbedder,
    device: torch.device,
    subtask_text: str,
    subtask_expert_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, List[int]]:
    role_id = int(ROLE_EXPERT_IDS["execute"])
    logits = router(title_emb=title_embedder(subtask_text))
    subtask_ids, logp_router = router.sample_topk(logits)
    subtask_ids = subtask_ids + int(subtask_expert_offset)
    role_tensor = torch.tensor([role_id], dtype=torch.long, device=device)
    expert_ids = torch.cat([role_tensor, subtask_ids.to(device=device)])
    subtask_local = [int(x) - int(subtask_expert_offset) for x in expert_ids.detach().cpu().tolist()[1:]]
    return expert_ids, logp_router, role_id, subtask_local


def _select_aggregate_experts(
    *,
    device: torch.device,
    parent_subtask_experts: List[int],
    subtask_expert_offset: int,
    merge_use_parent_experts: bool,
    randomize_parent_experts: bool = False,
    max_parent_experts: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, List[int]]:
    role_id = int(ROLE_EXPERT_IDS["aggregate"])
    role_tensor = torch.tensor([role_id], dtype=torch.long, device=device)
    logp_router = torch.tensor(0.0, dtype=torch.float32, device=device)

    if not bool(merge_use_parent_experts):
        return role_tensor, logp_router, role_id, []

    selected_pool = sorted({int(x) for x in parent_subtask_experts if int(x) >= 0})
    if not selected_pool:
        return role_tensor, logp_router, role_id, []

    selected: List[int] = list(selected_pool)
    if bool(randomize_parent_experts):
        upper = int(max_parent_experts) if int(max_parent_experts) > 0 else len(selected_pool)
        upper = max(1, min(upper, len(selected_pool)))
        k = random.randint(1, upper)
        selected = sorted(random.sample(selected_pool, k=k))

    subtask_ids = torch.tensor(
        [int(subtask_expert_offset) + int(x) for x in selected],
        dtype=torch.long,
        device=device,
    )
    expert_ids = torch.cat([role_tensor, subtask_ids])
    return expert_ids, logp_router, role_id, selected


def _build_children_from_predecessors(predecessors: Dict[int, List[int]]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {nid: [] for nid in predecessors}
    for child, parents in predecessors.items():
        for parent in parents:
            children.setdefault(parent, []).append(child)
    for nid in children:
        children[nid] = sorted(set(children[nid]))
    return children



def _parse_snapshot_functions_for_steps(
    *,
    python_code: str,
    step_by_id: Dict[str, dict],
    allowed_step_ids: Sequence[str],
) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    code = str(python_code or "").strip()
    if not code:
        return parsed
    try:
        ast.parse(code)
    except Exception:
        return parsed

    for sid in allowed_step_ids:
        step = step_by_id.get(sid)
        if step is None:
            continue
        try:
            fn_name = scipipe._extract_function_name(str(step.get("function_header") or ""))
        except Exception:
            continue
        fn_code = scipipe._get_function_from_code(code, fn_name)
        fn_code = str(fn_code or "").strip()
        if fn_code:
            parsed[sid] = fn_code
    return parsed



def _build_graph_plan_for_problem(
    *,
    problem: dict,
    step_by_id: Dict[str, dict],
    step_order: Dict[str, int],
    graph_root: Optional[Path],
) -> Tuple[List[int], Dict[int, List[int]], Dict[int, str], Dict[int, str]]:
    problem_id = str(problem.get("problem_id") or "")
    if graph_root is not None:
        graph_index = scipipe._build_graph_root_index(graph_root)
        graph_dir = graph_index.get(problem_id)
    else:
        graph_index = {}
        graph_dir = None

    if graph_dir is not None and (graph_dir / "task_graph.json").exists():
        try:
            spec = scipipe.convert_taskgraph(graph_dir / "task_graph.json")
            predecessors = scipipe._build_predecessors(spec)
            node_ids = sorted(spec.node_metadata)
            node_to_step = {int(nid): str(spec.node_metadata[nid].original_id) for nid in node_ids}
            node_to_title = {int(nid): str(getattr(spec.node_metadata[nid], "title", "") or "").strip() for nid in node_ids}
            node_ids = [nid for nid in node_ids if node_to_step.get(nid) in step_by_id]
            predecessors = {int(nid): [int(p) for p in predecessors.get(int(nid), [])] for nid in node_ids}
            return (
                node_ids,
                predecessors,
                {nid: node_to_step[nid] for nid in node_ids},
                {nid: node_to_title.get(nid, "") for nid in node_ids},
            )
        except Exception:
            pass

    # Chain fallback
    ordered_step_ids = [sid for sid, _ in sorted(step_order.items(), key=lambda item: item[1]) if sid in step_by_id]
    node_ids = list(range(1, len(ordered_step_ids) + 1))
    predecessors = {nid: ([nid - 1] if nid > 1 else []) for nid in node_ids}
    node_to_step = {nid: ordered_step_ids[nid - 1] for nid in node_ids}
    node_to_title = {
        nid: str((step_by_id.get(node_to_step[nid], {}) or {}).get("step_description_prompt") or "").strip()
        for nid in node_ids
    }
    return node_ids, predecessors, node_to_step, node_to_title



def _build_srddstyle_training_entries(
    *,
    problems: List[dict],
    with_background: bool,
    prompt_style: str,
    prefill_python_fence: bool,
    max_steps_per_problem: int,
    graph_root: Optional[Path],
    include_execute_entries: bool,
    include_aggregate_entries: bool,
    aggregate_min_parents: int,
    aggregate_subtask_prefix: str,
    router_text_source: str,
) -> Tuple[List[SRDDStyleEntry], Dict[str, int]]:
    entries: List[SRDDStyleEntry] = []
    counts = {
        "execute_entries": 0,
        "aggregate_entries": 0,
    }

    for problem in problems:
        step_by_id, step_order = scipipe._build_step_maps(problem)
        if not step_by_id:
            continue

        node_ids, predecessors, node_to_step, node_to_title = _build_graph_plan_for_problem(
            problem=problem,
            step_by_id=step_by_id,
            step_order=step_order,
            graph_root=graph_root,
        )
        if max_steps_per_problem > 0:
            node_ids = node_ids[:max_steps_per_problem]
        if not node_ids:
            continue

        node_snapshots_gt: Dict[int, Dict[str, str]] = {}

        for node_id in node_ids:
            step_id = str(node_to_step.get(node_id) or "")
            step = step_by_id.get(step_id)
            if step is None:
                continue
            node_title = str(node_to_title.get(node_id) or "").strip()

            parent_ids = [pid for pid in sorted(predecessors.get(node_id, [])) if pid in node_snapshots_gt]
            parent_step_ids = [str(node_to_step.get(pid) or "") for pid in parent_ids if str(node_to_step.get(pid) or "")]
            parent_snapshots = [dict(node_snapshots_gt[pid]) for pid in parent_ids]

            base_snapshot = srddpipe._hard_merge_snapshots(
                parent_snapshots=parent_snapshots,
                step_order=step_order,
            )
            ancestor_step_ids = srddpipe._sorted_step_ids(base_snapshot.keys(), step_order)

            target_gt_code = str(step.get("ground_truth_code") or "").strip()
            if include_aggregate_entries and (len(parent_ids) >= int(aggregate_min_parents)):
                expected_step_ids = srddpipe._sorted_step_ids(
                    {sid for snap in parent_snapshots for sid in snap.keys()},
                    step_order,
                )
                parent_payload = [(pid, node_snapshots_gt[pid]) for pid in parent_ids]
                aggregate_prompt = _build_aggregate_prompt_nonchat(
                    problem=problem,
                    next_step_id=step_id,
                    expected_step_ids=expected_step_ids,
                    step_by_id=step_by_id,
                    parent_snapshots=parent_payload,
                    prompt_style=prompt_style,
                    prefill_python_fence=prefill_python_fence,
                )
                agg_oracle = srddpipe._hard_merge_snapshots(parent_snapshots=parent_snapshots, step_order=step_order)
                entries.append(
                    SRDDStyleEntry(
                        kind="aggregate",
                        problem=problem,
                        step=step,
                        step_id=step_id,
                        prompt=aggregate_prompt,
                        subtask_text=_build_router_subtask_text(
                            entry_kind="aggregate",
                            problem=problem,
                            step_id=step_id,
                            step=step,
                            node_title=node_title,
                            with_background=with_background,
                            aggregate_subtask_prefix=aggregate_subtask_prefix,
                            router_text_source=router_text_source,
                        ),
                        oracle_ancestors=dict(agg_oracle),
                        expected_step_ids=list(expected_step_ids),
                        parent_step_ids=list(parent_step_ids),
                        parent_count=len(parent_ids),
                        target_gt_code=target_gt_code,
                    )
                )
                counts["aggregate_entries"] += 1

            if include_execute_entries:
                prompt = _build_execute_prompt_nonchat(
                    problem=problem,
                    step=step,
                    step_id=step_id,
                    ancestor_step_ids=ancestor_step_ids,
                    solved_functions=base_snapshot,
                    step_by_id=step_by_id,
                    with_background=with_background,
                    prompt_style=prompt_style,
                    prefill_python_fence=prefill_python_fence,
                )
                subtask_text = _build_router_subtask_text(
                    entry_kind="execute",
                    problem=problem,
                    step_id=step_id,
                    step=step,
                    node_title=node_title,
                    with_background=with_background,
                    aggregate_subtask_prefix=aggregate_subtask_prefix,
                    router_text_source=router_text_source,
                )
                entries.append(
                    SRDDStyleEntry(
                        kind="execute",
                        problem=problem,
                        step=step,
                        step_id=step_id,
                        prompt=prompt,
                        subtask_text=subtask_text or f"{problem.get('problem_id')}:{step_id}:execute",
                        oracle_ancestors=dict(base_snapshot),
                        expected_step_ids=list(ancestor_step_ids),
                        parent_step_ids=list(parent_step_ids),
                        parent_count=len(parent_ids),
                        target_gt_code=target_gt_code,
                    )
                )
                counts["execute_entries"] += 1

            # Update GT snapshot propagated by this node.
            new_snapshot = dict(base_snapshot)
            if target_gt_code:
                new_snapshot[step_id] = target_gt_code
            node_snapshots_gt[node_id] = new_snapshot

    return entries, counts



def _normalize_difficulty_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if text in {"easy", "medium", "hard"}:
        return text
    if text in {"very hard", "veryhard", "very-hard"}:
        return "very_hard"
    return "unknown"


def _load_problem_difficulty_rank_map(path: Optional[Path]) -> Dict[str, int]:
    if path is None:
        return {}
    fpath = Path(path).resolve()
    if not fpath.exists():
        return {}
    mapping: Dict[str, int] = {}
    with fpath.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            problem_id = str((row or {}).get("problem_id") or "").strip()
            if not problem_id:
                continue
            diff_label = _normalize_difficulty_label((row or {}).get("difficulty"))
            mapping[problem_id] = int(DIFFICULTY_RANK.get(diff_label, DIFFICULTY_RANK["unknown"]))
    return mapping


def _order_entries_for_epoch(
    *,
    train_entries: Sequence[SRDDStyleEntry],
    shuffle_entries: bool,
    seed: int,
    epoch: int,
    epochs: int,
    curriculum_mode: str,
    problem_difficulty_rank: Dict[str, int],
    curriculum_progressive_min_fraction: float,
) -> List[SRDDStyleEntry]:
    # Shuffle only at problem granularity. Keep each problem's internal
    # taskgraph entry order unchanged (execute/aggregate scheduling order).
    entries = list(train_entries)

    by_problem: Dict[str, List[SRDDStyleEntry]] = {}
    problem_order: List[str] = []
    for entry in entries:
        problem_id = str(entry.problem.get("problem_id") or "")
        if not problem_id:
            problem_id = f"__unknown_problem_{len(problem_order):06d}"
        if problem_id not in by_problem:
            by_problem[problem_id] = []
            problem_order.append(problem_id)
        by_problem[problem_id].append(entry)

    mode = str(curriculum_mode or "none").strip().lower()
    if mode == "none":
        if bool(shuffle_entries):
            random.Random(int(seed) + int(epoch)).shuffle(problem_order)
    else:
        if mode not in {"easy_to_hard", "hard_to_easy", "progressive_easy_to_hard"}:
            raise ValueError(f"Unsupported --curriculum-mode: {curriculum_mode}")
        asc = mode in {"easy_to_hard", "progressive_easy_to_hard"}
        buckets: Dict[int, List[str]] = {}
        for pid in problem_order:
            rank = int(problem_difficulty_rank.get(str(pid), DIFFICULTY_RANK["unknown"]))
            buckets.setdefault(rank, []).append(pid)
        ordered_problem_ids: List[str] = []
        rng = random.Random(int(seed) + int(epoch))
        for rank in sorted(buckets.keys(), reverse=(not asc)):
            bucket = list(buckets.get(rank, []))
            if bool(shuffle_entries):
                rng.shuffle(bucket)
            ordered_problem_ids.extend(bucket)
        if mode == "progressive_easy_to_hard":
            total = len(ordered_problem_ids)
            if total > 0:
                if int(epochs) <= 1:
                    frac = 1.0
                else:
                    frac = float(curriculum_progressive_min_fraction) + (
                        (1.0 - float(curriculum_progressive_min_fraction))
                        * float(int(epoch) - 1)
                        / float(int(epochs) - 1)
                    )
                frac = min(1.0, max(0.05, float(frac)))
                keep = max(1, int(math.ceil(float(total) * float(frac))))
                ordered_problem_ids = ordered_problem_ids[:keep]
        problem_order = ordered_problem_ids

    ordered: List[SRDDStyleEntry] = []
    for problem_id in problem_order:
        ordered.extend(by_problem.get(problem_id, []))
    return ordered


def _apply_epoch_entry_schedule(
    *,
    entries: Sequence[SRDDStyleEntry],
    epoch: int,
    execute_only_epochs: int,
    aggregate_start_epoch: int,
) -> Tuple[List[SRDDStyleEntry], Dict[str, int]]:
    scheduled = list(entries)
    e = int(epoch)
    execute_only_n = max(0, int(execute_only_epochs))
    aggregate_start = max(1, int(aggregate_start_epoch))

    if e <= execute_only_n:
        scheduled = [x for x in scheduled if x.kind == "execute"]
    elif e < aggregate_start:
        scheduled = [x for x in scheduled if x.kind != "aggregate"]

    info = {
        "epoch": int(e),
        "execute_only_epochs": int(execute_only_n),
        "aggregate_start_epoch": int(aggregate_start),
        "num_entries_before": int(len(entries)),
        "num_entries_after": int(len(scheduled)),
        "num_execute_after": int(sum(1 for x in scheduled if x.kind == "execute")),
        "num_aggregate_after": int(sum(1 for x in scheduled if x.kind == "aggregate")),
    }
    return scheduled, info


def _stage_epoch_w_gt(args: argparse.Namespace, epoch: int) -> float:
    base = max(0.0, float(getattr(args, "reward_w_gt", 0.0)))
    if not bool(getattr(args, "enable_gt_stage_schedule", True)):
        return float(base)

    e = int(epoch)
    zero_epoch = max(1, int(getattr(args, "stage_wgt_zero_epoch", 3)))
    w1_cfg = float(getattr(args, "stage_wgt_epoch1", -1.0))
    w2_cfg = float(getattr(args, "stage_wgt_epoch2", -1.0))
    w1 = float(base if w1_cfg < 0.0 else max(0.0, w1_cfg))
    w2 = float((w1 * 0.5) if w2_cfg < 0.0 else max(0.0, w2_cfg))

    if e >= zero_epoch:
        return 0.0
    if e <= 1:
        return float(w1)
    if e == 2:
        return float(w2)

    if zero_epoch <= 3:
        return float(w2)
    remain = float(max(0, zero_epoch - e))
    total = float(max(1, zero_epoch - 2))
    return float(max(0.0, w2 * (remain / total)))


def _stage_epoch_tf_ce(args: argparse.Namespace, epoch: int) -> float:
    base = max(0.0, float(getattr(args, "tf_ce_weight", 0.0)))
    if not bool(getattr(args, "enable_tf_ce_stage_schedule", True)):
        return float(base)

    e = int(epoch)
    zero_epoch = max(1, int(getattr(args, "tf_ce_zero_epoch", 3)))
    w1_cfg = float(getattr(args, "tf_ce_epoch1", -1.0))
    w2_cfg = float(getattr(args, "tf_ce_epoch2", -1.0))
    w1 = float(base if w1_cfg < 0.0 else max(0.0, w1_cfg))
    w2 = float((w1 * 0.5) if w2_cfg < 0.0 else max(0.0, w2_cfg))

    if e >= zero_epoch:
        return 0.0
    if e <= 1:
        return float(w1)
    if e == 2:
        return float(w2)

    if zero_epoch <= 3:
        return float(w2)
    remain = float(max(0, zero_epoch - e))
    total = float(max(1, zero_epoch - 2))
    return float(max(0.0, w2 * (remain / total)))


def _build_teacher_target_text(entry: SRDDStyleEntry, *, include_aggregate: bool) -> str:
    if str(entry.kind) == "execute":
        return str(entry.target_gt_code or "").strip()
    if str(entry.kind) != "aggregate" or not bool(include_aggregate):
        return ""

    blocks: List[str] = []
    for sid in list(entry.expected_step_ids):
        code = str(entry.oracle_ancestors.get(str(sid)) or "").strip()
        if code:
            blocks.append(code)
    return "\n\n".join(blocks).strip()


def _encode_teacher_target_ids(
    *,
    tokenizer: Any,
    target_text: str,
    device: torch.device,
    max_tokens: int,
) -> torch.Tensor:
    text = str(target_text or "").strip()
    if not text:
        return torch.empty(0, dtype=torch.long, device=device)
    try:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    except Exception:
        return torch.empty(0, dtype=torch.long, device=device)
    ids = enc.get("input_ids")
    if ids is None or int(ids.numel()) <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    target = ids[0].to(device=device, dtype=torch.long)
    if int(max_tokens) > 0 and int(target.numel()) > int(max_tokens):
        target = target[: int(max_tokens)]
    return target


def _compute_ddp_timeout_s(args: argparse.Namespace, *, world_size_env: int) -> int:
    configured = int(getattr(args, "dist_timeout_s", 0))
    if configured > 0:
        return int(configured)

    ws = max(1, int(world_size_env))
    group_size = max(1, int(getattr(args, "group_size", 1)))
    local_group_size = max(1, int(math.ceil(float(group_size) / float(ws))))
    test_timeout_s = max(1, int(getattr(args, "test_timeout_s", 180)))
    buffer_s = max(300, int(getattr(args, "dist_timeout_buffer_s", 900)))
    # One slow rank can spend close to one full local candidate batch inside step tests
    # before reaching the next collective. Size the process-group timeout to that skew.
    return int(max(900, (local_group_size * test_timeout_s) + buffer_s))


def _evaluate_execute_entry(
    *,
    entry: SRDDStyleEntry,
    python_code: str,
    parsed_function: str,
    sample_eval_dir: Path,
    h5py_file: Path,
    timeout_s: int,
    env: Dict[str, str],
    w_step: float,
    w_shape: float,
    w_gt: float,
    pass_bonus: float,
) -> CandidateEvalResult:
    reward, metrics, step_result = grpo_base._evaluate_candidate_reward(
        problem=entry.problem,
        step=entry.step,
        step_id=entry.step_id,
        python_code=python_code,
        parsed_function=parsed_function,
        oracle_ancestors=entry.oracle_ancestors,
        sample_eval_dir=sample_eval_dir,
        h5py_file=h5py_file,
        timeout_s=timeout_s,
        env=env,
        w_step=w_step,
        w_shape=w_shape,
        w_gt=0.0,
        pass_bonus=pass_bonus,
    )
    assembled_code = scipipe._assemble_program_code(
        dependencies=str(entry.problem.get("required_dependencies") or ""),
        ancestor_step_ids=list(entry.oracle_ancestors.keys()),
        solved_functions=entry.oracle_ancestors,
        current_python_code=python_code,
    )
    metrics = dict(metrics)
    target_gt = str(entry.target_gt_code or entry.step.get("ground_truth_code") or "").strip()
    has_gt = bool(target_gt)
    gt_score = 0.0
    gt_term = 0.0

    metrics["entry_kind"] = "execute"
    metrics["has_gt"] = bool(has_gt)
    metrics["gt_score"] = float(gt_score)
    metrics["gt_term"] = float(gt_term)
    metrics["reward"] = float(reward)
    return CandidateEvalResult(
        reward=float(reward),
        metrics=metrics,
        step_result=step_result,
        assembled_code=assembled_code,
    )



def _evaluate_aggregate_entry(
    *,
    entry: SRDDStyleEntry,
    python_code: str,
    step_by_id: Dict[str, dict],
    step_order: Dict[str, int],
    sample_eval_dir: Path,
    h5py_file: Path,
    timeout_s: int,
    env: Dict[str, str],
    w_step: float,
    w_shape: float,
    w_gt: float,
    pass_bonus: float,
) -> CandidateEvalResult:
    parse_ok = bool(str(python_code or "").strip())
    ast_ok = 0.0
    if parse_ok:
        try:
            ast.parse(python_code)
            ast_ok = 1.0
        except Exception:
            ast_ok = 0.0

    parsed_map = _parse_snapshot_functions_for_steps(
        python_code=python_code,
        step_by_id=step_by_id,
        allowed_step_ids=entry.expected_step_ids,
    )
    expected_count = len(entry.expected_step_ids)
    parsed_count = len(parsed_map)
    coverage = (float(parsed_count) / float(expected_count)) if expected_count > 0 else 1.0

    ancestor_step_ids = srddpipe._sorted_step_ids(parsed_map.keys(), step_order)
    assembled_code = scipipe._assemble_program_code(
        dependencies=str(entry.problem.get("required_dependencies") or ""),
        ancestor_step_ids=ancestor_step_ids,
        solved_functions=parsed_map,
        current_python_code=python_code,
    )

    step_result = scipipe._run_step_test(
        sample_dir=sample_eval_dir,
        step_id=entry.step_id,
        assembled_code=assembled_code,
        test_cases=list(entry.step.get("test_cases") or []),
        h5py_file=h5py_file,
        timeout_s=timeout_s,
        env=env,
    )
    step_score = 1.0 if step_result.passed else 0.0

    # Aggregate reward should optimize merge quality only:
    # - coverage: how many expected upstream functions are retained
    # - ast_ok: merged code is syntactically valid
    # - parse_ok: non-empty code was produced
    merge_score = (0.70 * float(coverage)) + (0.20 * float(ast_ok)) + (0.10 * float(parse_ok))
    shape_score = merge_score

    has_gt = bool(
        any(str(entry.oracle_ancestors.get(str(sid)) or "").strip() for sid in entry.expected_step_ids)
    )
    gt_score = 0.0
    gt_term = 0.0

    gate = 1.0 if parse_ok else 0.0
    reward = gate * ((0.45 * float(step_score)) + (0.25 * float(coverage)) + (0.20 * float(ast_ok)))

    metrics = {
        "entry_kind": "aggregate",
        "parse_ok": bool(parse_ok),
        "header_ok": bool(parsed_count > 0),
        "ast_ok": float(ast_ok),
        "step_score": float(step_score),
        "shape_score": float(shape_score),
        "reward": float(reward),
        "has_gt": bool(has_gt),
        "gt_score": float(gt_score),
        "gt_term": float(gt_term),
        "expected_upstream_count": int(expected_count),
        "parsed_upstream_count": int(parsed_count),
        "coverage": float(coverage),
    }

    return CandidateEvalResult(
        reward=float(reward),
        metrics=metrics,
        step_result=step_result,
        assembled_code=assembled_code,
    )



def _evaluate_entry_candidate(
    *,
    entry: SRDDStyleEntry,
    python_code: str,
    parsed_function: str,
    step_by_id: Dict[str, dict],
    step_order: Dict[str, int],
    sample_eval_dir: Path,
    h5py_file: Path,
    timeout_s: int,
    env: Dict[str, str],
    w_step: float,
    w_shape: float,
    w_gt: float,
    pass_bonus: float,
) -> CandidateEvalResult:
    if entry.kind == "execute":
        return _evaluate_execute_entry(
            entry=entry,
            python_code=python_code,
            parsed_function=parsed_function,
            sample_eval_dir=sample_eval_dir,
            h5py_file=h5py_file,
            timeout_s=timeout_s,
            env=env,
            w_step=w_step,
            w_shape=w_shape,
            w_gt=w_gt,
            pass_bonus=pass_bonus,
        )

    return _evaluate_aggregate_entry(
        entry=entry,
        python_code=python_code,
        step_by_id=step_by_id,
        step_order=step_order,
        sample_eval_dir=sample_eval_dir,
        h5py_file=h5py_file,
        timeout_s=timeout_s,
        env=env,
        w_step=w_step,
        w_shape=w_shape,
        w_gt=w_gt,
        pass_bonus=pass_bonus,
    )



def _write_srddstyle_candidate_snapshot(
    *,
    snapshot_root: Path,
    global_step_next: int,
    item_idx: int,
    candidate_idx: int,
    entry: SRDDStyleEntry,
    prompt: str,
    response_text: str,
    python_code: str,
    parsed_function: str,
    assembled_code: str,
    metrics: Dict[str, Any],
    step_result: Optional[scipipe.ScriptRunResult],
) -> Dict[str, str]:
    base = _srddstyle_candidate_snapshot_dir(
        snapshot_root=snapshot_root,
        global_step_next=global_step_next,
        item_idx=item_idx,
        candidate_idx=candidate_idx,
        entry=entry,
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
    assembled_path.write_text(str(assembled_code or ""), encoding="utf-8")

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


def _srddstyle_candidate_snapshot_dir(
    *,
    snapshot_root: Path,
    global_step_next: int,
    item_idx: int,
    candidate_idx: int,
    entry: SRDDStyleEntry,
) -> Path:
    problem_bucket = grpo_base._sanitize(str(entry.problem.get("problem_id") or "p"))
    step_bucket = f"{grpo_base._sanitize(str(entry.step_id or 'step'))}_{entry.kind}"
    return (
        snapshot_root
        / problem_bucket
        / step_bucket
        / f"gs_{int(global_step_next):07d}_item_{int(item_idx):05d}_cand_{int(candidate_idx):03d}"
    )



def train_grpo(args: argparse.Namespace) -> None:
    if args.gpus and str(args.gpus).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus).strip()
    if str(getattr(args, "hf_use_chat_template", "")).strip():
        os.environ["HF_USE_CHAT_TEMPLATE"] = str(args.hf_use_chat_template).strip().lower()
    os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", str(1 << 20))
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_timeout_s = _compute_ddp_timeout_s(args, world_size_env=world_size_env)
    use_dist = world_size_env > 1
    if use_dist and not grpo_base._is_dist_ready():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=int(ddp_timeout_s)),
        )

    world_size = int(dist.get_world_size()) if grpo_base._is_dist_ready() else 1
    rank = int(dist.get_rank()) if grpo_base._is_dist_ready() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.device)))

    visible_cuda_uuids: List[str] = []
    selected_cuda_uuid = ""
    if torch.cuda.is_available():
        visible_cuda_count = int(torch.cuda.device_count())
        if visible_cuda_count <= 0:
            raise RuntimeError("CUDA is available but no visible CUDA devices were detected.")
        for vidx in range(int(visible_cuda_count)):
            try:
                visible_cuda_uuids.append(str(getattr(torch.cuda.get_device_properties(vidx), "uuid", "")))
            except Exception:
                visible_cuda_uuids.append("")
        if world_size > 1:
            if int(local_rank) >= int(visible_cuda_count):
                cuda_idx = int(local_rank) % int(visible_cuda_count)
            else:
                cuda_idx = int(local_rank)
        else:
            cuda_idx = int(args.device)
            if cuda_idx < 0 or cuda_idx >= int(visible_cuda_count):
                raise RuntimeError(
                    f"--device={cuda_idx} is invalid for {visible_cuda_count} visible CUDA devices."
                )
        torch.cuda.set_device(cuda_idx)
        device = torch.device("cuda", cuda_idx)
        try:
            selected_cuda_uuid = str(getattr(torch.cuda.get_device_properties(int(cuda_idx)), "uuid", ""))
        except Exception:
            selected_cuda_uuid = ""
    else:
        device = torch.device("cpu")

    print(
        f"[rank {rank}] local_rank={local_rank} world_size={world_size} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"selected_device={device} "
        f"selected_uuid={selected_cuda_uuid} "
        f"visible_uuids={','.join(visible_cuda_uuids)} "
        f"visible_cuda_count={int(torch.cuda.device_count()) if torch.cuda.is_available() else 0} "
        f"ddp_timeout_s={int(ddp_timeout_s)}",
        flush=True,
    )

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
    local_route_count, local_candidates_per_route = _resolve_local_route_layout(
        local_group_size=int(local_group_size),
        requested_local_routes=int(getattr(args, "hierarchical_local_routes", 0)),
    )
    if int(local_route_count) * int(local_candidates_per_route) != int(local_group_size):
        raise RuntimeError("Invalid hierarchical route layout: local routes do not cover local group size.")
    if is_main:
        print(
            "[hier-credit] "
            f"group_size={int(args.group_size)} world_size={int(world_size)} "
            f"local_group_size={int(local_group_size)} local_routes={int(local_route_count)} "
            f"candidates_per_route_local={int(local_candidates_per_route)}",
            flush=True,
        )

    # Rolling checkpoint cadence is sample-based (global_step).
    save_every_samples = int(getattr(args, "save_every_samples", 0))

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
    if is_main:
        _write_json(
            run_dir / "launch_config.json",
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "script_path": str(SCRIPT_PATH),
                "cwd": str(Path.cwd().resolve()),
                "argv": list(sys.argv),
                "args": {str(k): _json_safe_value(v) for k, v in vars(args).items()},
            },
        )

    problems = _iter_selected_problems(_read_jsonl(args.dataset), int(args.max_problems))
    if not problems:
        raise RuntimeError("No training problems selected.")
    dataset_gt_stats = _summarize_dataset_gt_stats(problems)
    if is_main:
        print(
            "[dataset] "
            f"path={str(Path(args.dataset).resolve())} "
            f"problems={int(dataset_gt_stats.get('num_problems', 0))} "
            f"problems_with_gt={int(dataset_gt_stats.get('num_problems_with_gt', 0))} "
            f"steps_with_gt={int(dataset_gt_stats.get('num_steps_with_gt', 0))}/"
            f"{int(dataset_gt_stats.get('num_steps', 0))}",
            flush=True,
        )
        print(
            "[gt-signals] "
            f"disable_gt_code_signals={bool(getattr(args, 'disable_gt_code_signals', False))} "
            f"reward_w_gt={float(args.reward_w_gt):.6f} "
            f"enable_gt_stage_schedule={bool(args.enable_gt_stage_schedule)} "
            f"enable_tf_ce={bool(args.enable_tf_ce)} "
            f"tf_ce_weight={float(args.tf_ce_weight):.6f} "
            f"enable_tf_reward={bool(args.enable_tf_reward)} "
            f"tf_reward_weight={float(args.tf_reward_weight):.6f}",
            flush=True,
        )
        print(
            "[router] "
            f"router_text_source={str(getattr(args, 'router_text_source', 'description'))}",
            flush=True,
        )

    train_entries, entry_counts = _build_srddstyle_training_entries(
        problems=problems,
        with_background=bool(args.with_background),
        prompt_style=str(args.prompt_style),
        prefill_python_fence=bool(args.prefill_python_fence),
        max_steps_per_problem=int(args.max_steps_per_problem),
        graph_root=(Path(args.graph_root).resolve() if args.graph_root is not None else None),
        include_execute_entries=bool(args.include_execute_entries),
        include_aggregate_entries=bool(args.include_aggregate_entries),
        aggregate_min_parents=int(args.aggregate_min_parents),
        aggregate_subtask_prefix=str(args.aggregate_subtask_prefix),
        router_text_source=str(getattr(args, "router_text_source", "description")),
    )
    if not train_entries:
        raise RuntimeError("No SRDD-style GRPO entries built.")

    curriculum_mode = str(getattr(args, "curriculum_mode", "none") or "none").strip().lower()
    if curriculum_mode not in {"none", "easy_to_hard", "hard_to_easy", "progressive_easy_to_hard"}:
        raise ValueError(f"Unsupported --curriculum-mode: {curriculum_mode}")
    curriculum_progressive_min_fraction = float(getattr(args, "curriculum_progressive_min_fraction", 0.4))
    curriculum_progressive_min_fraction = min(1.0, max(0.05, float(curriculum_progressive_min_fraction)))
    curriculum_difficulty_file = (
        Path(args.curriculum_difficulty_file).resolve() if args.curriculum_difficulty_file is not None else None
    )
    problem_difficulty_rank: Dict[str, int] = {}
    if curriculum_mode != "none":
        problem_difficulty_rank = _load_problem_difficulty_rank_map(curriculum_difficulty_file)
        if not problem_difficulty_rank:
            raise RuntimeError(
                f"Curriculum mode '{curriculum_mode}' enabled, but no difficulty mapping loaded from: "
                f"{curriculum_difficulty_file}"
            )
    train_problem_ids = sorted(
        {
            str(e.problem.get("problem_id") or "")
            for e in train_entries
            if str(e.problem.get("problem_id") or "").strip()
        }
    )
    curriculum_problem_match_count = int(
        sum(1 for pid in train_problem_ids if str(pid) in problem_difficulty_rank)
    )

    step_by_problem: Dict[str, Dict[str, dict]] = {}
    step_order_by_problem: Dict[str, Dict[str, int]] = {}
    for problem in problems:
        pid = str(problem.get("problem_id") or "")
        sb, so = scipipe._build_step_maps(problem)
        step_by_problem[pid] = sb
        step_order_by_problem[pid] = so

    model, tokenizer = _load_backbone(model_name=args.model_name, torch_dtype=args.torch_dtype, device=device)
    try:
        first_param = next(model.parameters())
        print(
            f"[rank {rank}] model_first_param_device={str(first_param.device)} "
            f"model_first_param_dtype={str(first_param.dtype)}",
            flush=True,
        )
    except StopIteration:
        print(f"[rank {rank}] model has no parameters", flush=True)

    num_role_experts = int(len(ROLE_EXPERT_IDS))
    num_subtask_experts = int(args.num_subtask_experts)
    if num_subtask_experts <= 0:
        raise ValueError("--num-subtask-experts must be > 0.")
    num_total_experts = int(num_role_experts + num_subtask_experts)
    subtask_expert_offset = int(num_role_experts)

    lora_cfg = LoRAConfig(
        num_experts=int(num_total_experts),
        top_k=int(args.subtask_top_k) + 1,
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        target_modules=("q_proj", "v_proj", "o_proj"),
        last_n_layers=int(args.lora_last_n_layers),
    )
    inject_mole_lora(model, cfg=lora_cfg)

    router_cfg = SubtaskRouterConfig(num_experts=int(num_subtask_experts), top_k=int(args.subtask_top_k))
    router = SubtaskRouter(router_cfg).to(device)
    title_embedder = TitleEmbedder(model=model, tokenizer=tokenizer, out_dim=router_cfg.title_emb_dim).to(device)

    lora_trainable_params = _unique_trainable_parameters(
        [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    )
    router_trainable_params = _unique_trainable_parameters([p for p in router.parameters() if p.requires_grad])
    title_trainable_params = _unique_trainable_parameters(_title_embedder_trainable_parameters(title_embedder))

    opt_lora = torch.optim.AdamW(lora_trainable_params, lr=float(args.lora_lr))
    opt_router_params = _unique_trainable_parameters(list(router_trainable_params) + list(title_trainable_params))
    opt_router = torch.optim.AdamW(opt_router_params, lr=float(args.router_lr))

    loaded_state: Dict[str, Any] = {}
    if init_ckpt is not None:
        try:
            loaded_state = load_mole_checkpoint(
                ckpt_dir=init_ckpt,
                device=device,
                router=router,
                title_embedder=title_embedder,
                model=model,
                opt_router=opt_router,
                opt_lora=opt_lora,
            )
        except Exception as exc:
            if is_main:
                print(
                    f"[warn] optimizer-state load failed ({exc}); reloading checkpoint weights without optimizer states.",
                    flush=True,
                )
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
    sample_final_perf_jsonl = detailed_log_root / "sample_final_performance.jsonl"

    history: List[dict] = []
    global_step = int(loaded_state.get("global_step", 0))
    resume_epoch = int(loaded_state.get("epoch", 0)) if init_ckpt is not None else 0
    resume_item_idx = int(loaded_state.get("item_idx", 0)) if init_ckpt is not None else 0
    resume_strict_step = bool(getattr(args, "resume_strict_step", False))
    if init_ckpt is not None and resume_strict_step:
        start_epoch = max(1, resume_epoch)
    else:
        start_epoch = max(1, resume_epoch + 1)
    if start_epoch > int(args.epochs):
        if is_main:
            print(
                f"[resume] checkpoint epoch={resume_epoch} already >= target epochs={int(args.epochs)}; nothing to train."
            )
        return

    trainable_params = _unique_trainable_parameters(
        list(lora_trainable_params) + list(router_trainable_params) + list(title_trainable_params)
    )

    role_expert_counts: Dict[str, int] = {"execute": 0, "aggregate": 0}
    subtask_expert_counts: Dict[int, int] = {i: 0 for i in range(int(num_subtask_experts))}
    enforce_local_expert_diversity = bool(getattr(args, "enforce_local_expert_diversity", True))
    enforce_local_output_diversity = bool(getattr(args, "enforce_local_output_diversity", True))
    diversity_max_resample = max(1, int(getattr(args, "diversity_max_resample", 8)))
    aggregate_parent_max_experts = int(getattr(args, "aggregate_parent_max_experts", 0))
    if aggregate_parent_max_experts <= 0:
        aggregate_parent_max_experts = int(args.subtask_top_k)
    aggregate_random_parent_experts = bool(getattr(args, "aggregate_random_parent_experts", True))

    for epoch in range(start_epoch, int(args.epochs) + 1):
        epoch_entries = _order_entries_for_epoch(
            train_entries=train_entries,
            shuffle_entries=bool(getattr(args, "shuffle_entries", False)),
            seed=int(args.seed),
            epoch=int(epoch),
            epochs=int(args.epochs),
            curriculum_mode=curriculum_mode,
            problem_difficulty_rank=problem_difficulty_rank,
            curriculum_progressive_min_fraction=float(curriculum_progressive_min_fraction),
        )
        epoch_entries, stage_info = _apply_epoch_entry_schedule(
            entries=epoch_entries,
            epoch=int(epoch),
            execute_only_epochs=int(getattr(args, "stage_execute_only_epochs", 1)),
            aggregate_start_epoch=int(getattr(args, "stage_aggregate_start_epoch", 2)),
        )
        # Keep the original w_gt schedule for compatibility/logging.
        # GT supervision is now provided by TF-CE (AST shaping removed).
        epoch_reward_w_gt = _stage_epoch_w_gt(args, int(epoch))
        if bool(getattr(args, "enable_tf_ce", True)):
            epoch_tf_ce_weight = _stage_epoch_tf_ce(args, int(epoch))
        else:
            epoch_tf_ce_weight = 0.0
        if not epoch_entries:
            raise RuntimeError(
                f"No train entries remain after epoch schedule filtering at epoch={epoch}. "
                "Check --include-* flags and stage schedule arguments."
            )
        if is_main:
            print(
                "[stage] "
                f"epoch={int(epoch)} "
                f"entries={int(stage_info.get('num_entries_after', 0))} "
                f"execute={int(stage_info.get('num_execute_after', 0))} "
                f"aggregate={int(stage_info.get('num_aggregate_after', 0))} "
                "context=oracle_prefix_only "
                f"w_gt={float(epoch_reward_w_gt):.6f} "
                f"tf_ce={float(epoch_tf_ce_weight):.6f}",
                flush=True,
            )
        entry_start_idx = 1
        if init_ckpt is not None and resume_strict_step and int(epoch) == int(start_epoch):
            if int(resume_item_idx) <= 0 and int(resume_epoch) > 0 and int(global_step) > 0:
                inferred_item_idx = int(global_step) - max(0, int(resume_epoch) - 1) * int(len(epoch_entries))
                if 1 <= int(inferred_item_idx) <= int(len(epoch_entries)):
                    resume_item_idx = int(inferred_item_idx)
                    if is_main:
                        print(
                            f"[resume] inferred item_idx={resume_item_idx} from global_step={global_step}, "
                            f"epoch={resume_epoch}, entries_per_epoch={len(epoch_entries)}",
                            flush=True,
                        )
            if int(resume_item_idx) > 0:
                entry_start_idx = min(int(len(epoch_entries)) + 1, int(resume_item_idx) + 1)
                if is_main:
                    print(
                        f"[resume] strict-step mode: epoch={epoch}, resume_item_idx={resume_item_idx}, "
                        f"starting from entry index {entry_start_idx}.",
                        flush=True,
                    )
        step_subtask_expert_cache: Dict[Tuple[str, str], List[int]] = {}
        # Oracle-prefix-only mode: prompts/ancestors always come from dataset GT context.
        # Keep runtime snapshots only for logging/debug artifacts.
        runtime_problem_snapshot: Dict[str, Dict[str, str]] = {}
        runtime_step_snapshot: Dict[Tuple[str, str], Dict[str, str]] = {}
        role_expert_counts = {"execute": 0, "aggregate": 0}
        subtask_expert_counts = {i: 0 for i in range(int(num_subtask_experts))}
        problem_entry_remaining: Dict[str, int] = {}
        problem_final_state: Dict[str, Dict[str, Any]] = {}
        if is_main:
            for e in epoch_entries:
                pid = str(e.problem.get("problem_id") or "")
                problem_entry_remaining[pid] = int(problem_entry_remaining.get(pid, 0)) + 1

        for item_idx, entry in enumerate(epoch_entries, start=1):
            if int(item_idx) < int(entry_start_idx):
                continue
            cands: List[SRDDStyleCandidate] = []
            sample_eval_dir = (
                tmp_eval_root
                / f"rank_{rank:02d}"
                / f"{grpo_base._sanitize(str(entry.problem.get('problem_id') or 'p'))}_{grpo_base._sanitize(entry.step_id)}_{entry.kind}"
            )
            sample_eval_dir.mkdir(parents=True, exist_ok=True)

            problem_id = str(entry.problem.get("problem_id") or "")
            sb = step_by_problem.get(problem_id, {})
            so = step_order_by_problem.get(problem_id, {})

            context_mode = "oracle_prefix_only"
            use_oracle_prefix = True
            # Always use pre-built GT-prefix context from entry construction.
            entry_runtime = entry
            base_snapshot_runtime = dict(entry.oracle_ancestors)

            tf_target_text = _build_teacher_target_text(
                entry_runtime,
                include_aggregate=bool(getattr(args, "tf_ce_include_aggregate", False)),
            )
            tf_target_ids_entry = _encode_teacher_target_ids(
                tokenizer=tokenizer,
                target_text=tf_target_text,
                device=device,
                max_tokens=int(getattr(args, "tf_ce_max_target_tokens", 768)),
            )
            tf_target_ids_empty = torch.empty(0, dtype=torch.long, device=device)

            parent_subtask_union: List[int] = []
            if entry_runtime.kind != "execute":
                parent_subtask_union = sorted(
                    {
                        int(sid)
                        for parent_step_id in entry.parent_step_ids
                        for sid in step_subtask_expert_cache.get((problem_id, parent_step_id), [])
                    }
                )

            used_expert_signatures: set[Tuple[int, ...]] = set()
            used_output_texts: set[str] = set()
            route_groups: List[Dict[str, Any]] = []
            candidate_detail_rows: Dict[int, Dict[str, Any]] = {}
            for route_local_idx in range(1, int(local_route_count) + 1):
                route_global_idx = (rank * int(local_route_count)) + int(route_local_idx)
                expert_ids = torch.empty(0, dtype=torch.long, device=device)
                logp_router = torch.tensor(0.0, dtype=torch.float32, device=device)
                role_expert_id = int(ROLE_EXPERT_IDS.get(entry_runtime.kind, 0))
                chosen_subtask_expert_ids: List[int] = []
                chosen_signature: Tuple[int, ...] = tuple()
                for _ in range(int(diversity_max_resample)):
                    if entry_runtime.kind == "execute":
                        expert_ids, logp_router, role_expert_id, chosen_subtask_expert_ids = _select_execute_experts(
                            router=router,
                            title_embedder=title_embedder,
                            device=device,
                            subtask_text=entry_runtime.subtask_text,
                            subtask_expert_offset=subtask_expert_offset,
                        )
                    else:
                        expert_ids, logp_router, role_expert_id, chosen_subtask_expert_ids = _select_aggregate_experts(
                            device=device,
                            parent_subtask_experts=parent_subtask_union,
                            subtask_expert_offset=subtask_expert_offset,
                            merge_use_parent_experts=bool(args.merge_use_parent_experts),
                            randomize_parent_experts=bool(aggregate_random_parent_experts),
                            max_parent_experts=int(aggregate_parent_max_experts),
                        )

                    chosen_signature = tuple(int(x) for x in expert_ids.detach().cpu().tolist())
                    if (not enforce_local_expert_diversity) or (chosen_signature not in used_expert_signatures):
                        break
                used_expert_signatures.add(chosen_signature)

                route_usage_count = int(local_candidates_per_route)
                role_expert_counts[entry.kind] = int(role_expert_counts.get(entry.kind, 0)) + int(route_usage_count)
                for sid in chosen_subtask_expert_ids:
                    if 0 <= int(sid) < int(num_subtask_experts):
                        subtask_expert_counts[int(sid)] = int(subtask_expert_counts.get(int(sid), 0)) + int(route_usage_count)

                route_cands: List[SRDDStyleCandidate] = []
                for route_member_idx in range(1, int(local_candidates_per_route) + 1):
                    local_idx = ((int(route_local_idx) - 1) * int(local_candidates_per_route)) + int(route_member_idx)
                    cand_idx = (rank * int(local_group_size)) + int(local_idx)

                    text = ""
                    prompt_ids = torch.empty(0, dtype=torch.long, device=device)
                    gen_ids = torch.empty(0, dtype=torch.long, device=device)
                    for _ in range(int(diversity_max_resample)):
                        text, prompt_ids, gen_ids = mole_gen.generate_with_experts(
                            prompt=entry_runtime.prompt,
                            expert_ids=expert_ids,
                        )
                        text_sig = str(text or "").strip()
                        if (not enforce_local_output_diversity) or (text_sig not in used_output_texts):
                            used_output_texts.add(text_sig)
                            break
                    python_code = scipipe._extract_python_script(text)
                    parsed_function = (
                        _extract_step_code_for_eval(entry_runtime.step, python_code) if entry_runtime.kind == "execute" else ""
                    )

                    eval_res = _evaluate_entry_candidate(
                        entry=entry_runtime,
                        python_code=python_code,
                        parsed_function=parsed_function,
                        step_by_id=sb,
                        step_order=so,
                        sample_eval_dir=sample_eval_dir,
                        h5py_file=args.h5py_file,
                        timeout_s=int(args.test_timeout_s),
                        env=env,
                        w_step=float(args.reward_w_step),
                        w_shape=float(args.reward_w_shape),
                        w_gt=float(epoch_reward_w_gt),
                        pass_bonus=float(args.reward_pass_bonus),
                    )

                    metrics = dict(eval_res.metrics)
                    step_result = eval_res.step_result
                    metrics["step_test_status"] = step_result.status if step_result is not None else "skipped"
                    metrics["did_retry"] = False
                    metrics["retry_count"] = 0
                    metrics["attempt_count"] = 1
                    metrics["role_expert_id"] = int(role_expert_id)
                    metrics["subtask_expert_ids"] = [int(x) for x in chosen_subtask_expert_ids]
                    tf_enabled = bool(getattr(args, "enable_tf_ce", True)) and bool(tf_target_ids_entry.numel() > 0)
                    if tf_enabled and bool(getattr(args, "tf_ce_execute_only", True)) and entry_runtime.kind != "execute":
                        tf_enabled = False
                    if tf_enabled and bool(getattr(args, "tf_ce_only_on_fail", True)):
                        tf_enabled = float(metrics.get("step_score", 0.0)) < 1.0
                    tf_target_ids = tf_target_ids_entry if tf_enabled else tf_target_ids_empty
                    tf_reward_enabled = bool(getattr(args, "enable_tf_reward", False)) and bool(tf_enabled)
                    tf_nll_value: Optional[float] = None
                    if tf_reward_enabled and int(tf_target_ids.numel()) > 0:
                        with torch.no_grad():
                            tf_logp_sum_probe = mole_gen.logprob_of_generation(
                                prompt_ids=prompt_ids,
                                gen_ids=tf_target_ids,
                                expert_ids=expert_ids,
                            )
                            tf_len_probe = max(1.0, float(tf_target_ids.numel()))
                            tf_nll_probe = -(tf_logp_sum_probe / tf_len_probe)
                            tf_nll_value = float(tf_nll_probe.detach().cpu().item())
                            if not math.isfinite(float(tf_nll_value)):
                                tf_nll_value = None
                    metrics["tf_nll"] = float(tf_nll_value) if tf_nll_value is not None else None
                    metrics["tf_reward_enabled"] = bool(tf_reward_enabled)
                    metrics["reward_before_tf"] = float(eval_res.reward)
                    metrics["teacher_reward_z"] = 0.0
                    metrics["teacher_reward_term"] = 0.0
                    metrics["reward_after_tf"] = float(eval_res.reward)

                    snapshot_paths: Dict[str, str] = {}
                    if bool(args.save_candidate_snapshots):
                        snapshot_paths = _write_srddstyle_candidate_snapshot(
                            snapshot_root=snapshot_root,
                            global_step_next=int(global_step + 1),
                            item_idx=int(item_idx),
                            candidate_idx=int(cand_idx),
                            entry=entry_runtime,
                            prompt=entry_runtime.prompt,
                            response_text=text,
                            python_code=python_code,
                            parsed_function=parsed_function,
                            assembled_code=eval_res.assembled_code,
                            metrics=metrics,
                            step_result=step_result,
                        )

                    candidate_detail_rows[int(cand_idx)] = {
                        "global_step_next": int(global_step + 1),
                        "epoch": int(epoch),
                        "item_idx": int(item_idx),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "entry_kind": entry.kind,
                        "parent_count": int(entry.parent_count),
                        "problem_id": str(entry.problem.get("problem_id") or ""),
                        "problem_name": str(entry.problem.get("problem_name") or ""),
                        "step_id": str(entry.step_id),
                        "parent_step_ids": list(entry.parent_step_ids),
                        "candidate_idx_global": int(cand_idx),
                        "candidate_idx_local": int(local_idx),
                        "group_size_global": int(args.group_size),
                        "group_size_local": int(local_group_size),
                        "route_idx_local": int(route_local_idx),
                        "route_idx_global": int(route_global_idx),
                        "route_member_idx_local": int(route_member_idx),
                        "routes_local": int(local_route_count),
                        "candidates_per_route_local": int(local_candidates_per_route),
                        "context_mode": str(context_mode),
                        "use_oracle_prefix": bool(use_oracle_prefix),
                        "did_retry": False,
                        "retry_count": 0,
                        "attempt_count": 1,
                        "reward": float(eval_res.reward),
                        "reward_base": float(eval_res.reward),
                        "parse_ok": bool(metrics.get("parse_ok", False)),
                        "header_ok": bool(metrics.get("header_ok", False)),
                        "ast_ok": float(metrics.get("ast_ok", 0.0)),
                        "step_score": float(metrics.get("step_score", 0.0)),
                        "shape_score": float(metrics.get("shape_score", 0.0)),
                        "has_gt": bool(metrics.get("has_gt", False)),
                        "gt_score": float(metrics.get("gt_score", 0.0)),
                        "gt_term": float(metrics.get("gt_term", 0.0)),
                        "coverage": float(metrics.get("coverage", 0.0)),
                        "expected_upstream_count": int(metrics.get("expected_upstream_count", 0)),
                        "parsed_upstream_count": int(metrics.get("parsed_upstream_count", 0)),
                        "step_test_status": str(metrics.get("step_test_status", "skipped")),
                        "step_test_passed": bool(step_result.passed) if step_result is not None else False,
                        "step_test_return_code": int(step_result.return_code) if step_result is not None else None,
                        "step_test_elapsed_ms": int(step_result.elapsed_ms) if step_result is not None else None,
                        "step_test_script_path": str(step_result.script_path) if step_result is not None else "",
                        "step_test_stdout_tail": _tail_text(step_result.stdout) if step_result is not None else "",
                        "step_test_stderr_tail": _tail_text(step_result.stderr) if step_result is not None else "",
                        "sample_eval_dir": str(sample_eval_dir.resolve()),
                        "expert_ids": [int(x) for x in expert_ids.detach().cpu().tolist()],
                        "role_expert_id": int(role_expert_id),
                        "subtask_expert_ids": [int(x) for x in chosen_subtask_expert_ids],
                        "logp_router": float(logp_router.detach().cpu().item()),
                        "reward_w_gt_epoch": float(epoch_reward_w_gt),
                        "tf_ce_weight_epoch": float(epoch_tf_ce_weight),
                        "tf_enabled": bool(tf_enabled),
                        "tf_target_tokens": int(tf_target_ids.numel()),
                        "tf_reward_enabled": bool(tf_reward_enabled),
                        "tf_nll": (float(tf_nll_value) if tf_nll_value is not None else None),
                        "teacher_reward_z": 0.0,
                        "teacher_reward_term": 0.0,
                        **snapshot_paths,
                    }

                    cand = SRDDStyleCandidate(
                        idx=int(cand_idx),
                        route_local_idx=int(route_local_idx),
                        route_global_idx=int(route_global_idx),
                        text=text,
                        python_code=python_code,
                        parsed_function=parsed_function,
                        prompt_ids=prompt_ids,
                        gen_ids=gen_ids,
                        expert_ids=expert_ids,
                        logp_router=logp_router,
                        role_expert_id=int(role_expert_id),
                        subtask_expert_ids=[int(x) for x in chosen_subtask_expert_ids],
                        tf_target_ids=tf_target_ids,
                        tf_enabled=bool(tf_enabled),
                        reward=float(eval_res.reward),
                        metrics=metrics,
                    )
                    cands.append(cand)
                    route_cands.append(cand)

                route_groups.append(
                    {
                        "route_local_idx": int(route_local_idx),
                        "route_global_idx": int(route_global_idx),
                        "logp_router": logp_router,
                        "cands": route_cands,
                        "reward_mean_local": 0.0,
                        "reward_std_local": 0.0,
                    }
                )

            teacher_reward_enabled = bool(getattr(args, "enable_tf_reward", False))
            teacher_reward_weight = max(0.0, float(getattr(args, "tf_reward_weight", 0.0)))
            teacher_reward_zclip = max(0.0, float(getattr(args, "tf_reward_zclip", 2.0)))
            teacher_reward_eps = max(1e-12, float(getattr(args, "tf_reward_eps", 1e-6)))
            teacher_reward_apply_count = 0
            teacher_reward_bonus_sum = 0.0
            teacher_reward_bonus_abs_sum = 0.0
            teacher_reward_z_sum = 0.0
            teacher_reward_nll_mean = 0.0
            teacher_reward_nll_std = 0.0

            if teacher_reward_enabled and teacher_reward_weight > 0.0 and cands:
                local_tf_nll_vals: List[float] = []
                local_tf_nll_masks: List[float] = []
                for cand in cands:
                    tf_nll_local = cand.metrics.get("tf_nll", None)
                    tf_reward_ok = bool(cand.metrics.get("tf_reward_enabled", False))
                    tf_valid = bool(tf_reward_ok and isinstance(tf_nll_local, (int, float)) and math.isfinite(float(tf_nll_local)))
                    local_tf_nll_vals.append(float(tf_nll_local) if tf_valid else 0.0)
                    local_tf_nll_masks.append(1.0 if tf_valid else 0.0)

                tf_nll_vals = grpo_base._all_gather_float_list(local_tf_nll_vals, device=device, world_size=world_size)
                tf_nll_masks = grpo_base._all_gather_float_list(local_tf_nll_masks, device=device, world_size=world_size)
                tf_nll_valid = [float(v) for v, m in zip(tf_nll_vals, tf_nll_masks) if float(m) > 0.5 and math.isfinite(float(v))]
                if tf_nll_valid:
                    teacher_reward_nll_mean = float(sum(tf_nll_valid) / float(len(tf_nll_valid)))
                    teacher_reward_nll_std = float(grpo_base._safe_std(tf_nll_valid)) if len(tf_nll_valid) > 1 else 0.0

                for cand, tf_nll_local, tf_mask_local in zip(cands, local_tf_nll_vals, local_tf_nll_masks):
                    z = 0.0
                    bonus = 0.0
                    if float(tf_mask_local) > 0.5 and float(teacher_reward_nll_std) > 0.0:
                        z = (float(teacher_reward_nll_mean) - float(tf_nll_local)) / float(teacher_reward_nll_std + teacher_reward_eps)
                        z = float(grpo_base._clip_advantage(float(z), float(teacher_reward_zclip)))
                        bonus = float(teacher_reward_weight) * float(z)
                    if float(bonus) != 0.0:
                        cand.reward = float(cand.reward) + float(bonus)
                        teacher_reward_apply_count += 1
                        teacher_reward_bonus_sum += float(bonus)
                        teacher_reward_bonus_abs_sum += abs(float(bonus))
                        teacher_reward_z_sum += float(z)
                    cand.metrics["teacher_reward_z"] = float(z)
                    cand.metrics["teacher_reward_term"] = float(bonus)
                    cand.metrics["reward_after_tf"] = float(cand.reward)

                if world_size > 1:
                    tf_reward_stats_t = torch.tensor(
                        [
                            float(teacher_reward_apply_count),
                            float(teacher_reward_bonus_sum),
                            float(teacher_reward_bonus_abs_sum),
                            float(teacher_reward_z_sum),
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    dist.all_reduce(tf_reward_stats_t, op=dist.ReduceOp.SUM)
                    teacher_reward_apply_count = int(tf_reward_stats_t[0].item())
                    teacher_reward_bonus_sum = float(tf_reward_stats_t[1].item())
                    teacher_reward_bonus_abs_sum = float(tf_reward_stats_t[2].item())
                    teacher_reward_z_sum = float(tf_reward_stats_t[3].item())

            for cand in cands:
                detail_row = candidate_detail_rows.get(int(cand.idx))
                if detail_row is None:
                    continue
                detail_row["reward"] = float(cand.reward)
                detail_row["teacher_reward_z"] = float(cand.metrics.get("teacher_reward_z", 0.0))
                detail_row["teacher_reward_term"] = float(cand.metrics.get("teacher_reward_term", 0.0))
                _append_jsonl(rank_detail_jsonl, detail_row)

            local_rewards = [float(c.reward) for c in cands]
            rewards = grpo_base._all_gather_float_list(local_rewards, device=device, world_size=world_size)
            reward_mean = float(sum(rewards) / float(len(rewards))) if rewards else 0.0
            reward_std = float(grpo_base._safe_std(rewards))
            teacher_reward_apply_rate = (
                float(teacher_reward_apply_count) / float(len(rewards)) if rewards else 0.0
            )
            teacher_reward_bonus_mean = (
                float(teacher_reward_bonus_sum) / float(teacher_reward_apply_count)
                if teacher_reward_apply_count > 0
                else 0.0
            )
            teacher_reward_bonus_abs_mean = (
                float(teacher_reward_bonus_abs_sum) / float(teacher_reward_apply_count)
                if teacher_reward_apply_count > 0
                else 0.0
            )
            teacher_reward_z_mean = (
                float(teacher_reward_z_sum) / float(teacher_reward_apply_count)
                if teacher_reward_apply_count > 0
                else 0.0
            )
            local_route_reward_means: List[float] = []
            for route_group in route_groups:
                route_rewards_local = [float(c.reward) for c in route_group["cands"]]
                route_mean_local = float(sum(route_rewards_local) / float(len(route_rewards_local))) if route_rewards_local else 0.0
                route_std_local = float(grpo_base._safe_std(route_rewards_local))
                route_group["reward_mean_local"] = float(route_mean_local)
                route_group["reward_std_local"] = float(route_std_local)
                local_route_reward_means.append(float(route_mean_local))
            route_reward_means = grpo_base._all_gather_float_list(
                local_route_reward_means, device=device, world_size=world_size
            )
            route_reward_mean = (
                float(sum(route_reward_means) / float(len(route_reward_means))) if route_reward_means else 0.0
            )
            route_reward_std = float(grpo_base._safe_std(route_reward_means))
            step_scores = grpo_base._all_gather_float_list(
                [float(c.metrics.get("step_score", 0.0)) for c in cands], device=device, world_size=world_size
            )
            shape_scores = grpo_base._all_gather_float_list(
                [float(c.metrics.get("shape_score", 0.0)) for c in cands], device=device, world_size=world_size
            )
            parse_scores = grpo_base._all_gather_float_list(
                [1.0 if bool(c.metrics.get("parse_ok", False)) else 0.0 for c in cands],
                device=device,
                world_size=world_size,
            )
            header_scores = grpo_base._all_gather_float_list(
                [1.0 if bool(c.metrics.get("header_ok", False)) else 0.0 for c in cands],
                device=device,
                world_size=world_size,
            )
            ast_scores = grpo_base._all_gather_float_list(
                [float(c.metrics.get("ast_ok", 0.0)) for c in cands], device=device, world_size=world_size
            )
            coverage_scores = grpo_base._all_gather_float_list(
                [float(c.metrics.get("coverage", 0.0)) for c in cands], device=device, world_size=world_size
            )
            gt_scores = grpo_base._all_gather_float_list(
                [float(c.metrics.get("gt_score", 0.0)) for c in cands], device=device, world_size=world_size
            )
            gt_terms = grpo_base._all_gather_float_list(
                [float(c.metrics.get("gt_term", 0.0)) for c in cands], device=device, world_size=world_size
            )
            has_gt_scores = grpo_base._all_gather_float_list(
                [1.0 if bool(c.metrics.get("has_gt", False)) else 0.0 for c in cands],
                device=device,
                world_size=world_size,
            )
            tf_enabled_scores = grpo_base._all_gather_float_list(
                [1.0 if bool(c.tf_enabled) else 0.0 for c in cands],
                device=device,
                world_size=world_size,
            )
            tf_target_token_scores = grpo_base._all_gather_float_list(
                [float(c.tf_target_ids.numel()) for c in cands],
                device=device,
                world_size=world_size,
            )

            did_update = True
            update_skip_reason = ""
            if bool(args.grpo_skip_update_if_allzero) and rewards and (max(rewards) - min(rewards) == 0.0):
                did_update = False
                update_skip_reason = "all_rewards_equal"
            if (
                did_update
                and bool(args.grpo_skip_update_if_low_std)
                and rewards
                and float(reward_std) < float(args.grpo_min_reward_std)
            ):
                did_update = False
                update_skip_reason = "low_reward_std"
            update_forced_by_tf = False
            if (
                (not did_update)
                and float(epoch_tf_ce_weight) > 0.0
                and bool(tf_enabled_scores)
                and (sum(tf_enabled_scores) > 0.0)
            ):
                did_update = True
                update_skip_reason = ""
                update_forced_by_tf = True

            loss_value = 0.0
            loss_grpo_value = 0.0
            loss_router_value = 0.0
            loss_tf_ce_value = 0.0
            tf_ce_apply_count = 0
            tf_ce_token_count = 0
            if did_update:
                opt_router.zero_grad(set_to_none=True)
                opt_lora.zero_grad(set_to_none=True)
                total_loss = torch.tensor(0.0, device=device)
                has_grad_term = False

                for route_group in route_groups:
                    route_mean_local = float(route_group.get("reward_mean_local", 0.0))
                    route_std_local = float(route_group.get("reward_std_local", 0.0))

                    for cand in route_group["cands"]:
                        # Inner-layer credit for generation policy: within-route relative advantage.
                        adv = float(cand.reward) - float(route_mean_local)
                        if bool(args.grpo_adv_normalize):
                            adv = adv / float(route_std_local + float(args.grpo_adv_eps)) if route_std_local > 0.0 else 0.0
                        adv = grpo_base._clip_advantage(float(adv), float(args.advantage_clip))
                        cand_has_grad = False
                        if float(adv) != 0.0:
                            logp_mole_sum = mole_gen.logprob_of_generation(
                                prompt_ids=cand.prompt_ids,
                                gen_ids=cand.gen_ids,
                                expert_ids=cand.expert_ids,
                            )
                            gen_len = max(1.0, float(cand.gen_ids.numel()))
                            logp_mole_mean = logp_mole_sum / gen_len

                            adv_t = torch.tensor(float(adv), device=device)
                            loss_i = -(adv_t * logp_mole_mean)
                            total_loss = total_loss + loss_i
                            loss_grpo_value += float(loss_i.detach().cpu().item())
                            cand_has_grad = True

                        if (
                            float(epoch_tf_ce_weight) > 0.0
                            and bool(cand.tf_enabled)
                            and int(cand.tf_target_ids.numel()) > 0
                        ):
                            tf_logp_sum = mole_gen.logprob_of_generation(
                                prompt_ids=cand.prompt_ids,
                                gen_ids=cand.tf_target_ids,
                                expert_ids=cand.expert_ids,
                            )
                            tf_len = max(1.0, float(cand.tf_target_ids.numel()))
                            tf_nll = -(tf_logp_sum / tf_len)
                            loss_tf_i = torch.tensor(float(epoch_tf_ce_weight), device=device) * tf_nll
                            total_loss = total_loss + loss_tf_i
                            loss_tf_ce_value += float(loss_tf_i.detach().cpu().item())
                            tf_ce_apply_count += 1
                            tf_ce_token_count += int(cand.tf_target_ids.numel())
                            cand_has_grad = True

                        if cand_has_grad:
                            has_grad_term = True

                    # Outer-layer credit for router: across-route relative advantage.
                    route_adv = float(route_mean_local) - float(route_reward_mean)
                    if bool(args.grpo_adv_normalize):
                        route_adv = (
                            route_adv / float(route_reward_std + float(args.grpo_adv_eps))
                            if route_reward_std > 0.0
                            else 0.0
                        )
                    route_adv = grpo_base._clip_advantage(float(route_adv), float(args.advantage_clip))
                    if float(route_adv) != 0.0:
                        route_adv_t = torch.tensor(float(route_adv), device=device)
                        loss_router_i = -(route_adv_t * float(args.alpha_router) * route_group["logp_router"])
                        total_loss = total_loss + loss_router_i
                        loss_router_value += float(loss_router_i.detach().cpu().item())
                        has_grad_term = True

                local_has_grad = 1 if has_grad_term else 0
                global_has_grad = local_has_grad
                if world_size > 1:
                    has_grad_t = torch.tensor([float(local_has_grad)], dtype=torch.float32, device=device)
                    dist.all_reduce(has_grad_t, op=dist.ReduceOp.SUM)
                    global_has_grad = 1 if has_grad_t.item() > 0.0 else 0

                if has_grad_term:
                    reg_loss = _subtask_router_reg_loss(
                        router,
                        float(args.subtask_proto_l2),
                        float(args.subtask_proto_ortho),
                    )
                    if float(reg_loss.detach().cpu().item()) != 0.0:
                        total_loss = total_loss + reg_loss
                    total_loss.backward()
                if global_has_grad:
                    grpo_base._average_gradients(trainable_params, world_size)
                    opt_router.step()
                    opt_lora.step()
                    loss_value = float(total_loss.detach().cpu().item())
                    if world_size > 1:
                        loss_t = torch.tensor([loss_value], dtype=torch.float32, device=device)
                        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                        loss_value = float(loss_t.item() / float(world_size))

            if world_size > 1:
                tf_stats_t = torch.tensor(
                    [
                        float(tf_ce_apply_count),
                        float(tf_ce_token_count),
                        float(loss_tf_ce_value),
                        float(loss_grpo_value),
                        float(loss_router_value),
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                dist.all_reduce(tf_stats_t, op=dist.ReduceOp.SUM)
                tf_ce_apply_count = int(tf_stats_t[0].item())
                tf_ce_token_count = int(tf_stats_t[1].item())
                loss_tf_ce_value = float(tf_stats_t[2].item()) / float(world_size)
                loss_grpo_value = float(tf_stats_t[3].item()) / float(world_size)
                loss_router_value = float(tf_stats_t[4].item()) / float(world_size)

            best = max(cands, key=lambda c: float(c.reward))
            best_reward, best_step_score, best_shape_score, _ = grpo_base._global_best_metrics(
                local_best_reward=float(best.reward),
                local_best_step=float(best.metrics.get("step_score", 0.0)),
                local_best_shape=float(best.metrics.get("shape_score", 0.0)),
                local_best_gt=0.0,
                device=device,
                world_size=world_size,
            )
            best_candidate_idx_global = (
                int(max(range(len(rewards)), key=lambda i: float(rewards[i]))) + 1 if rewards else int(best.idx)
            )
            winner_snapshot_dir = _srddstyle_candidate_snapshot_dir(
                snapshot_root=snapshot_root,
                global_step_next=int(global_step + 1),
                item_idx=int(item_idx),
                candidate_idx=int(best_candidate_idx_global),
                entry=entry_runtime,
            )
            winner_python = ""
            winner_parsed_function = ""
            winner_metrics: Dict[str, Any] = {}
            try:
                py_path = winner_snapshot_dir / "python.py"
                if py_path.exists():
                    winner_python = str(py_path.read_text(encoding="utf-8"))
            except Exception:
                winner_python = ""
            try:
                parsed_path = winner_snapshot_dir / "parsed_function.py"
                if parsed_path.exists():
                    winner_parsed_function = str(parsed_path.read_text(encoding="utf-8"))
            except Exception:
                winner_parsed_function = ""
            try:
                metrics_path = winner_snapshot_dir / "metrics.json"
                if metrics_path.exists():
                    winner_metrics = dict(json.loads(metrics_path.read_text(encoding="utf-8")))
            except Exception:
                winner_metrics = {}

            if int(best_candidate_idx_global) == int(best.idx):
                if not winner_python:
                    winner_python = str(best.python_code or "")
                if not winner_parsed_function:
                    winner_parsed_function = str(best.parsed_function or "")
                if not winner_metrics:
                    winner_metrics = dict(best.metrics or {})

            merged_runtime_snapshot = dict(base_snapshot_runtime)
            if entry_runtime.kind == "execute":
                parsed_clean = str(winner_parsed_function or "").strip()
                if parsed_clean:
                    merged_runtime_snapshot[str(entry_runtime.step_id)] = parsed_clean
                winner_subtask_experts = [
                    int(x)
                    for x in list(winner_metrics.get("subtask_expert_ids") or [])
                    if isinstance(x, (int, float))
                ]
                winner_parse_ok = bool(winner_metrics.get("parse_ok", False))
                winner_step_ok = float(winner_metrics.get("step_score", 0.0)) > 0.0
                if winner_parse_ok and winner_step_ok and winner_subtask_experts:
                    step_subtask_expert_cache[(problem_id, entry_runtime.step_id)] = winner_subtask_experts
            else:
                parsed_map_runtime = _parse_snapshot_functions_for_steps(
                    python_code=str(winner_python or ""),
                    step_by_id=sb,
                    allowed_step_ids=entry_runtime.expected_step_ids,
                )
                for sid, code in parsed_map_runtime.items():
                    clean = str(code or "").strip()
                    if clean:
                        merged_runtime_snapshot[str(sid)] = clean

            runtime_step_snapshot[(problem_id, entry_runtime.step_id)] = dict(merged_runtime_snapshot)
            current_problem_snapshot = dict(runtime_problem_snapshot.get(problem_id, {}))
            current_problem_snapshot.update(merged_runtime_snapshot)
            runtime_problem_snapshot[problem_id] = current_problem_snapshot
            global_step += 1

            sample_perf = {
                "global_step": int(global_step),
                "epoch": int(epoch),
                "item_idx": int(item_idx),
                "entry_kind": entry_runtime.kind,
                "parent_count": int(entry_runtime.parent_count),
                "context_mode": str(context_mode),
                "use_oracle_prefix": bool(use_oracle_prefix),
                "disable_gt_code_signals": bool(getattr(args, "disable_gt_code_signals", False)),
                "problem_id": str(entry_runtime.problem.get("problem_id") or ""),
                "problem_name": str(entry_runtime.problem.get("problem_name") or ""),
                "step_id": str(entry_runtime.step_id),
                "num_candidates": int(len(rewards)),
                "num_routes_global": int(len(route_reward_means)),
                "num_routes_local": int(local_route_count),
                "candidates_per_route_local": int(local_candidates_per_route),
                "reward_mean": float(reward_mean),
                "reward_std": float(reward_std),
                "route_reward_mean": float(route_reward_mean),
                "route_reward_std": float(route_reward_std),
                "reward_max": float(max(rewards)) if rewards else 0.0,
                "reward_min": float(min(rewards)) if rewards else 0.0,
                "step_pass_rate": float(sum(step_scores) / float(len(step_scores))) if step_scores else 0.0,
                "shape_score_mean": float(sum(shape_scores) / float(len(shape_scores))) if shape_scores else 0.0,
                "parse_ok_rate": float(sum(parse_scores) / float(len(parse_scores))) if parse_scores else 0.0,
                "header_ok_rate": float(sum(header_scores) / float(len(header_scores))) if header_scores else 0.0,
                "ast_ok_rate": float(sum(ast_scores) / float(len(ast_scores))) if ast_scores else 0.0,
                "coverage_mean": float(sum(coverage_scores) / float(len(coverage_scores))) if coverage_scores else 0.0,
                "has_gt_rate": float(sum(has_gt_scores) / float(len(has_gt_scores))) if has_gt_scores else 0.0,
                "gt_score_mean": float(sum(gt_scores) / float(len(gt_scores))) if gt_scores else 0.0,
                "gt_term_mean": float(sum(gt_terms) / float(len(gt_terms))) if gt_terms else 0.0,
                "reward_w_gt_epoch": float(epoch_reward_w_gt),
                "tf_ce_weight_epoch": float(epoch_tf_ce_weight),
                "tf_enabled_rate": float(sum(tf_enabled_scores) / float(len(tf_enabled_scores)))
                if tf_enabled_scores
                else 0.0,
                "tf_target_tokens_mean": float(sum(tf_target_token_scores) / float(len(tf_target_token_scores)))
                if tf_target_token_scores
                else 0.0,
                "tf_ce_apply_count": int(tf_ce_apply_count),
                "tf_ce_token_count": int(tf_ce_token_count),
                "teacher_reward_enabled": bool(teacher_reward_enabled and teacher_reward_weight > 0.0),
                "teacher_reward_weight": float(teacher_reward_weight),
                "teacher_reward_apply_count": int(teacher_reward_apply_count),
                "teacher_reward_apply_rate": float(teacher_reward_apply_rate),
                "teacher_reward_bonus_mean": float(teacher_reward_bonus_mean),
                "teacher_reward_bonus_abs_mean": float(teacher_reward_bonus_abs_mean),
                "teacher_reward_z_mean": float(teacher_reward_z_mean),
                "teacher_reward_nll_mean": float(teacher_reward_nll_mean),
                "teacher_reward_nll_std": float(teacher_reward_nll_std),
                "loss_grpo": float(loss_grpo_value),
                "loss_router": float(loss_router_value),
                "loss_tf_ce": float(loss_tf_ce_value),
                "best_reward": float(best_reward),
                "best_step_score": float(best_step_score),
                "best_shape_score": float(best_shape_score),
                "role_expert_id": int(best.role_expert_id),
                "subtask_expert_ids": [int(x) for x in best.subtask_expert_ids],
                "logp_router": float(best.logp_router.detach().cpu().item()),
                "did_update": bool(did_update),
                "update_skip_reason": str(update_skip_reason),
                "update_forced_by_tf": bool(update_forced_by_tf),
                "loss": float(loss_value),
            }

            if is_main:
                row = {
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "item_idx": int(item_idx),
                    "entry_kind": entry_runtime.kind,
                    "parent_count": int(entry_runtime.parent_count),
                    "context_mode": str(context_mode),
                    "use_oracle_prefix": bool(use_oracle_prefix),
                    "disable_gt_code_signals": bool(getattr(args, "disable_gt_code_signals", False)),
                    "problem_id": str(entry_runtime.problem.get("problem_id") or ""),
                    "problem_name": str(entry_runtime.problem.get("problem_name") or ""),
                    "step_id": str(entry_runtime.step_id),
                    "num_routes_global": int(len(route_reward_means)),
                    "num_routes_local": int(local_route_count),
                    "candidates_per_route_local": int(local_candidates_per_route),
                    "reward_mean": float(reward_mean),
                    "reward_std": float(reward_std),
                    "route_reward_mean": float(route_reward_mean),
                    "route_reward_std": float(route_reward_std),
                    "best_reward": float(best_reward),
                    "best_step_score": float(best_step_score),
                    "best_shape_score": float(best_shape_score),
                    "role_expert_id": int(best.role_expert_id),
                    "subtask_expert_ids": [int(x) for x in best.subtask_expert_ids],
                    "reward_w_gt_epoch": float(epoch_reward_w_gt),
                    "tf_ce_weight_epoch": float(epoch_tf_ce_weight),
                    "tf_enabled_rate": float(sum(tf_enabled_scores) / float(len(tf_enabled_scores)))
                    if tf_enabled_scores
                    else 0.0,
                    "tf_ce_apply_count": int(tf_ce_apply_count),
                    "tf_ce_token_count": int(tf_ce_token_count),
                    "teacher_reward_enabled": bool(teacher_reward_enabled and teacher_reward_weight > 0.0),
                    "teacher_reward_weight": float(teacher_reward_weight),
                    "teacher_reward_apply_count": int(teacher_reward_apply_count),
                    "teacher_reward_apply_rate": float(teacher_reward_apply_rate),
                    "teacher_reward_bonus_mean": float(teacher_reward_bonus_mean),
                    "teacher_reward_bonus_abs_mean": float(teacher_reward_bonus_abs_mean),
                    "teacher_reward_z_mean": float(teacher_reward_z_mean),
                    "teacher_reward_nll_mean": float(teacher_reward_nll_mean),
                    "teacher_reward_nll_std": float(teacher_reward_nll_std),
                    "did_update": bool(did_update),
                    "update_skip_reason": str(update_skip_reason),
                    "update_forced_by_tf": bool(update_forced_by_tf),
                    "loss_grpo": float(loss_grpo_value),
                    "loss_router": float(loss_router_value),
                    "loss_tf_ce": float(loss_tf_ce_value),
                    "loss": float(loss_value),
                    "group_rewards": rewards,
                }
                history.append(row)
                _append_jsonl(sample_perf_jsonl, sample_perf)

                state = problem_final_state.get(problem_id)
                if state is None:
                    state = {
                        "problem": entry_runtime.problem,
                        "final_snapshot": {},
                        "best_path": [],
                    }
                    problem_final_state[problem_id] = state

                best_cand_dir = winner_snapshot_dir
                best_python = str(winner_python or "")
                best_parsed_function = str(winner_parsed_function or "")

                final_snapshot = state["final_snapshot"]
                if entry_runtime.kind == "execute":
                    parsed_clean = str(best_parsed_function or "").strip()
                    if parsed_clean:
                        final_snapshot[str(entry_runtime.step_id)] = parsed_clean
                else:
                    parsed_map = _parse_snapshot_functions_for_steps(
                        python_code=str(best_python or ""),
                        step_by_id=sb,
                        allowed_step_ids=entry_runtime.expected_step_ids,
                    )
                    if parsed_map:
                        final_snapshot.update(parsed_map)

                state["best_path"].append(
                    {
                        "global_step": int(global_step),
                        "epoch": int(epoch),
                        "item_idx": int(item_idx),
                        "entry_kind": str(entry_runtime.kind),
                        "problem_id": str(problem_id),
                        "step_id": str(entry_runtime.step_id),
                        "candidate_idx_global": int(best_candidate_idx_global),
                        "best_reward": float(best_reward),
                        "best_step_score": float(best_step_score),
                        "best_shape_score": float(best_shape_score),
                        "snapshot_dir": str(best_cand_dir.resolve()),
                    }
                )

                problem_entry_remaining[problem_id] = int(problem_entry_remaining.get(problem_id, 1)) - 1
                if int(problem_entry_remaining.get(problem_id, 0)) <= 0:
                    problem_obj = dict(state.get("problem") or {})
                    final_snapshot = dict(state.get("final_snapshot") or {})
                    final_step_ids = srddpipe._sorted_step_ids(
                        final_snapshot.keys(),
                        step_order_by_problem.get(problem_id, {}),
                    )
                    final_code = ""
                    if final_step_ids:
                        final_code = (
                            "\n\n".join(
                                [str(problem_obj.get("required_dependencies") or "").strip()]
                                + [str(final_snapshot[sid]).strip() for sid in final_step_ids]
                            ).strip()
                            + "\n"
                        )

                    general_tests = list(problem_obj.get("general_tests") or [])
                    general_status = "skipped"
                    general_pass = 0
                    general_payload: Dict[str, Any] = {
                        "status": "skipped",
                        "reason": "",
                        "target_group": "",
                    }
                    if not final_code:
                        general_status = "no_final_snapshot"
                        general_payload = {
                            "status": "no_final_snapshot",
                            "reason": "No parseable functions in best-candidate path.",
                            "target_group": "",
                        }
                    elif not general_tests:
                        general_status = "no_general_tests"
                        general_payload = {
                            "status": "no_general_tests",
                            "reason": "Problem has no general_tests.",
                            "target_group": "",
                        }
                    else:
                        final_eval_dir = (
                            run_dir
                            / "train_final_eval"
                            / f"epoch_{epoch:03d}"
                            / f"{grpo_base._sanitize(str(problem_id))}_{grpo_base._sanitize(str(problem_obj.get('problem_name') or 'p'))}"
                        )
                        final_eval_dir.mkdir(parents=True, exist_ok=True)
                        sub_steps = list(problem_obj.get("sub_steps") or [])
                        general_target_group = (
                            str(sub_steps[-1].get("step_number") or problem_id) if sub_steps else str(problem_id)
                        )
                        general_result = scipipe._run_general_test(
                            sample_dir=final_eval_dir,
                            problem_id=str(problem_id),
                            general_target_group=str(general_target_group),
                            assembled_code=str(final_code),
                            general_tests=general_tests,
                            h5py_file=args.h5py_file,
                            timeout_s=int(args.test_timeout_s),
                            env=env,
                        )
                        general_status = str(general_result.status)
                        general_pass = 1 if bool(general_result.passed) else 0
                        general_payload = {
                            "status": str(general_result.status),
                            "passed": bool(general_result.passed),
                            "return_code": int(general_result.return_code),
                            "elapsed_ms": int(general_result.elapsed_ms),
                            "stdout_tail": _tail_text(general_result.stdout),
                            "stderr_tail": _tail_text(general_result.stderr),
                            "target_group": str(general_target_group),
                            "script_path": str(general_result.script_path),
                            "eval_dir": str(final_eval_dir.resolve()),
                        }

                    final_snapshot_dir = (
                        snapshot_root
                        / grpo_base._sanitize(str(problem_id))
                        / f"final_epoch_{epoch:03d}"
                    )
                    final_snapshot_dir.mkdir(parents=True, exist_ok=True)
                    (final_snapshot_dir / "final_program.py").write_text(str(final_code or ""), encoding="utf-8")
                    _write_json(final_snapshot_dir / "final_snapshot.json", final_snapshot)
                    _write_json(final_snapshot_dir / "best_path.json", list(state.get("best_path") or []))
                    _write_json(final_snapshot_dir / "general_test_result.json", general_payload)

                    _append_jsonl(
                        sample_final_perf_jsonl,
                        {
                            "epoch": int(epoch),
                            "problem_id": str(problem_id),
                            "problem_name": str(problem_obj.get("problem_name") or ""),
                            "num_entries": int(len(state.get("best_path") or [])),
                            "final_snapshot_size": int(len(final_snapshot)),
                            "final_step_ids": list(final_step_ids),
                            "general_test_status": str(general_status),
                            "general_test_pass": int(general_pass),
                            "final_snapshot_dir": str(final_snapshot_dir.resolve()),
                            "final_program_path": str((final_snapshot_dir / "final_program.py").resolve()),
                            "general_test_result_path": str((final_snapshot_dir / "general_test_result.json").resolve()),
                        },
                    )
                    problem_final_state.pop(problem_id, None)

            if int(save_every_samples) > 0 and (global_step % int(save_every_samples) == 0):
                if is_main:
                    save_mole_checkpoint(
                        ckpt_dir=ckpt_root / "sample_latest",
                        router=router,
                        title_embedder=title_embedder,
                        model=model,
                        opt_router=opt_router,
                        opt_lora=opt_lora,
                        trainer_state={
                            "run_name": run_name,
                            "global_step": int(global_step),
                            "epoch": int(epoch),
                            "item_idx": int(item_idx),
                        },
                    )
                if world_size > 1:
                    dist.barrier()

        if is_main:
            _write_json(run_dir / f"train_epoch_{epoch:03d}.json", {"epoch": epoch, "global_step": global_step})
            if bool(args.save_last):
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
                        "item_idx": int(len(epoch_entries)),
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
                        "item_idx": int(len(epoch_entries)),
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
                "epoch": int(epoch),
                "item_idx": int(len(epoch_entries)),
                "epochs": int(args.epochs),
            },
        )

        _write_json(run_dir / "train_history.json", {"rows": history})
        _write_json(
            run_dir / "train_summary.json",
            {
                "run_name": run_name,
                "dataset": str(Path(args.dataset).resolve()),
                "dataset_gt_stats": dict(dataset_gt_stats),
                "disable_gt_code_signals": bool(getattr(args, "disable_gt_code_signals", False)),
                "num_entries": len(train_entries),
                "entry_counts": entry_counts,
                "epochs": int(args.epochs),
                "global_step": int(global_step),
                "checkpoint_final": str((ckpt_root / "final").resolve()),
                "init_checkpoint": (str(init_ckpt) if init_ckpt is not None else ""),
                "train_init_mode": ("resume_from_checkpoint" if init_ckpt is not None else "scratch"),
                "graph_root": (str(Path(args.graph_root).resolve()) if args.graph_root is not None else ""),
                "world_size": int(world_size),
                "group_size": int(args.group_size),
                "local_group_size": int(local_group_size),
                "hierarchical_local_routes": int(local_route_count),
                "candidates_per_route_local": int(local_candidates_per_route),
                "save_every_samples_resolved": int(save_every_samples),
                "sample_latest_checkpoint_dir": str((ckpt_root / "sample_latest").resolve()),
                "save_last": bool(args.save_last),
                "dist_timeout_s": int(args.dist_timeout_s),
                "dist_timeout_buffer_s": int(args.dist_timeout_buffer_s),
                "detailed_step_log_dir": str(detailed_log_root.resolve()),
                "detailed_step_log_files": [
                    str((detailed_log_root / f"rank_{r:02d}_step_records.jsonl").resolve()) for r in range(world_size)
                ],
                "sample_performance_log": str(sample_perf_jsonl.resolve()),
                "sample_final_performance_log": str(sample_final_perf_jsonl.resolve()),
                "snapshot_root": str(snapshot_root.resolve()),
                "save_candidate_snapshots": bool(args.save_candidate_snapshots),
                "entry_kinds": sorted({e.kind for e in train_entries}),
                "aggregate_min_parents": int(args.aggregate_min_parents),
                "num_role_experts": int(num_role_experts),
                "num_subtask_experts": int(num_subtask_experts),
                "num_total_experts": int(num_total_experts),
                "subtask_expert_offset": int(subtask_expert_offset),
                "merge_use_parent_experts": bool(args.merge_use_parent_experts),
                "shuffle_entries": bool(args.shuffle_entries),
                "shuffle_entries_scope": "problem_order_only",
                "training_context_mode": "oracle_prefix_only",
                "router_text_source": str(getattr(args, "router_text_source", "description")),
                "enable_gt_stage_schedule": bool(args.enable_gt_stage_schedule),
                "stage_execute_only_epochs": int(args.stage_execute_only_epochs),
                "stage_aggregate_start_epoch": int(args.stage_aggregate_start_epoch),
                "stage_wgt_zero_epoch": int(args.stage_wgt_zero_epoch),
                "stage_wgt_epoch1": float(args.stage_wgt_epoch1),
                "stage_wgt_epoch2": float(args.stage_wgt_epoch2),
                "enable_tf_ce": bool(args.enable_tf_ce),
                "tf_ce_weight": float(args.tf_ce_weight),
                "tf_ce_only_on_fail": bool(args.tf_ce_only_on_fail),
                "tf_ce_execute_only": bool(args.tf_ce_execute_only),
                "tf_ce_include_aggregate": bool(args.tf_ce_include_aggregate),
                "tf_ce_max_target_tokens": int(args.tf_ce_max_target_tokens),
                "enable_tf_ce_stage_schedule": bool(args.enable_tf_ce_stage_schedule),
                "tf_ce_zero_epoch": int(args.tf_ce_zero_epoch),
                "tf_ce_epoch1": float(args.tf_ce_epoch1),
                "tf_ce_epoch2": float(args.tf_ce_epoch2),
                "enable_tf_reward": bool(args.enable_tf_reward),
                "tf_reward_weight": float(args.tf_reward_weight),
                "tf_reward_zclip": float(args.tf_reward_zclip),
                "tf_reward_eps": float(args.tf_reward_eps),
                "curriculum_mode": str(curriculum_mode),
                "curriculum_difficulty_file": (
                    str(curriculum_difficulty_file) if curriculum_difficulty_file is not None else ""
                ),
                "curriculum_progressive_min_fraction": float(curriculum_progressive_min_fraction),
                "curriculum_problem_match_count": int(curriculum_problem_match_count),
                "curriculum_problem_total": int(len(train_problem_ids)),
                "enforce_local_expert_diversity": bool(enforce_local_expert_diversity),
                "enforce_local_output_diversity": bool(enforce_local_output_diversity),
                "diversity_max_resample": int(diversity_max_resample),
                "aggregate_random_parent_experts": bool(aggregate_random_parent_experts),
                "aggregate_parent_max_experts": int(aggregate_parent_max_experts),
                "role_expert_counts_last_epoch": {
                    "execute": int(role_expert_counts.get("execute", 0)),
                    "aggregate": int(role_expert_counts.get("aggregate", 0)),
                },
                "subtask_expert_counts_last_epoch": {
                    str(i): int(subtask_expert_counts.get(i, 0)) for i in range(int(num_subtask_experts))
                },
            },
        )
    if world_size > 1:
        dist.barrier()

    if bool(args.eval_after_train) and is_main:
        _write_json(
            run_dir / "eval_summary.json",
            {
                "skipped": True,
                "reason": (
                    "eval_after_train currently uses chain-style single-agent evaluator and is incompatible with "
                    "role+subtask expert routing (fixed role expert + offset subtask experts)."
                ),
            },
        )

    if is_main:
        print(str(run_dir))
        print(str((ckpt_root / "final").resolve()))



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SciCode MoLE GRPO training (oracle-prefix-only context) with SRDD-style taskgraph entries."
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--graph-root",
        type=Path,
        default=None,
        help="Optional task-graph root (sample dirs with task_graph.json + sample.json).",
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--run-name", type=str, default="")

    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional path to SFT (or previous GRPO) checkpoint directory.",
    )
    p.add_argument(
        "--resume-strict-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When resuming from checkpoint, continue from saved item index in the same epoch "
            "(or infer from global_step) instead of jumping directly to the next epoch."
        ),
    )

    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--torch-dtype", type=str, default="bfloat16")

    p.add_argument("--num-subtask-experts", type=int, default=4)
    p.add_argument("--subtask-top-k", type=int, default=2)
    p.add_argument("--merge-use-parent-experts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--subtask-proto-l2", type=float, default=0.0)
    p.add_argument("--subtask-proto-ortho", type=float, default=0.0)
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
    p.add_argument(
        "--hierarchical-local-routes",
        type=int,
        default=0,
        help=(
            "Number of route groups per rank for two-layer hierarchical credit assignment "
            "(<=0 uses auto layout)."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--save-every-samples",
        type=int,
        default=50,
        help="Save rolling checkpoint every N training samples (global_step).",
    )
    p.add_argument("--save-every-epochs", type=int, default=0)
    p.add_argument(
        "--save-last",
        action="store_true",
        help="Also save ckpt_root/last at each epoch end (default: off).",
    )

    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--max-steps-per-problem", type=int, default=0)
    p.add_argument(
        "--shuffle-entries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Shuffle problem order each epoch while preserving per-problem taskgraph step order. "
            "If curriculum is enabled, shuffling is applied within each difficulty bucket."
        ),
    )
    p.add_argument(
        "--curriculum-mode",
        choices=["none", "easy_to_hard", "hard_to_easy", "progressive_easy_to_hard"],
        default="easy_to_hard",
        help="Problem-level curriculum scheduling by difficulty rank.",
    )
    p.add_argument(
        "--curriculum-progressive-min-fraction",
        type=float,
        default=0.4,
        help="For progressive_easy_to_hard, fraction of easiest problems kept at epoch 1.",
    )
    p.add_argument(
        "--curriculum-difficulty-file",
        type=Path,
        default=DEFAULT_CURRICULUM_DIFFICULTY_FILE,
        help="TSV with columns: problem_id, difficulty.",
    )

    p.add_argument("--with-background", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--prompt-style",
        type=str,
        default="strict",
        choices=("strict", "minimal"),
        help="Non-chat prompt policy aligned with single-LLM ablation.",
    )
    p.add_argument(
        "--prefill-python-fence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append trailing ```python to prompt so model continues directly as code.",
    )
    p.add_argument(
        "--hf-use-chat-template",
        type=str,
        choices=("auto", "on", "off"),
        default="off",
        help="Forwarded to HF_USE_CHAT_TEMPLATE. Keep off for non-chat prompting.",
    )
    p.add_argument(
        "--prompt-template",
        choices=["auto", "multistep", "background_comment"],
        default="auto",
        help="Legacy compatibility flag (kept for CLI compatibility; non-chat prompts are controlled by --prompt-style).",
    )

    p.add_argument("--include-execute-entries", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-aggregate-entries", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aggregate-min-parents", type=int, default=2)
    p.add_argument("--aggregate-subtask-prefix", type=str, default="aggregate merge")
    p.add_argument(
        "--router-text-source",
        type=str,
        default="description",
        choices=("description", "title", "title_header"),
        help=(
            "Text source used by router/title_embedder for expert selection: "
            "description=step description(+background), title=node title, "
            "title_header=node title + function header."
        ),
    )
    p.add_argument(
        "--aggregate-random-parent-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For aggregate entries, sample a random subset of parent subtask experts per candidate.",
    )
    p.add_argument(
        "--aggregate-parent-max-experts",
        type=int,
        default=0,
        help="Upper bound for sampled parent experts in aggregate (<=0 uses subtask-top-k).",
    )
    p.add_argument("--alpha-router", type=float, default=0.2)
    p.add_argument("--grpo-adv-normalize", action="store_true")
    p.add_argument("--grpo-adv-eps", type=float, default=1e-6)
    p.add_argument("--advantage-clip", type=float, default=5.0)
    p.add_argument("--grpo-skip-update-if-allzero", action="store_true")
    p.add_argument(
        "--grpo-skip-update-if-low-std",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip optimizer update when candidate reward std is too low (weak learning signal).",
    )
    p.add_argument(
        "--grpo-min-reward-std",
        type=float,
        default=0.03,
        help="Minimum reward std required to perform GRPO update when --grpo-skip-update-if-low-std is enabled.",
    )
    p.add_argument(
        "--enforce-local-expert-diversity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample per-candidate expert routing to reduce duplicate expert signatures in one local group.",
    )
    p.add_argument(
        "--enforce-local-output-diversity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample generation to reduce duplicate outputs in one local group.",
    )
    p.add_argument("--diversity-max-resample", type=int, default=8)

    p.add_argument("--reward-w-step", type=float, default=1.00)
    p.add_argument("--reward-w-shape", type=float, default=0.02)
    p.add_argument("--reward-w-gt", type=float, default=0.10)
    p.add_argument("--reward-pass-bonus", type=float, default=0.25)
    p.add_argument(
        "--disable-gt-code-signals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Hard-disable all GT-dependent reward/loss signals. "
            "When enabled: reward_w_gt=0, TF-CE off, TF reward off."
        ),
    )
    p.add_argument(
        "--enable-tf-ce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable teacher-forcing auxiliary CE loss on GT targets during GRPO updates.",
    )
    p.add_argument(
        "--tf-ce-weight",
        type=float,
        default=0.05,
        help="Base coefficient for auxiliary teacher-forcing CE loss.",
    )
    p.add_argument(
        "--tf-ce-only-on-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply TF-CE only when step test fails (step_score < 1).",
    )
    p.add_argument(
        "--tf-ce-execute-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict TF-CE to execute entries.",
    )
    p.add_argument(
        "--tf-ce-include-aggregate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow aggregate entries to build GT targets for TF-CE.",
    )
    p.add_argument(
        "--tf-ce-max-target-tokens",
        type=int,
        default=768,
        help="Truncate TF-CE GT target length to this many tokens.",
    )
    p.add_argument(
        "--enable-tf-ce-stage-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable epoch-wise annealing schedule for TF-CE coefficient.",
    )
    p.add_argument(
        "--tf-ce-zero-epoch",
        type=int,
        default=3,
        help="Set TF-CE coefficient to 0 at epochs >= this value.",
    )
    p.add_argument(
        "--tf-ce-epoch1",
        type=float,
        default=-1.0,
        help="Optional explicit TF-CE coefficient for epoch 1. Negative uses --tf-ce-weight.",
    )
    p.add_argument(
        "--tf-ce-epoch2",
        type=float,
        default=-1.0,
        help="Optional explicit TF-CE coefficient for epoch 2. Negative uses 0.5 * epoch1 coefficient.",
    )
    p.add_argument(
        "--enable-tf-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject normalized teacher signal into candidate reward: "
            "reward += tf_reward_weight * z(-tf_nll), computed over valid candidates in the current GRPO group."
        ),
    )
    p.add_argument(
        "--tf-reward-weight",
        type=float,
        default=0.05,
        help="Coefficient for normalized teacher-reward bonus when --enable-tf-reward is enabled.",
    )
    p.add_argument(
        "--tf-reward-zclip",
        type=float,
        default=2.0,
        help="Clip bound for normalized teacher z-score before scaling by --tf-reward-weight.",
    )
    p.add_argument(
        "--tf-reward-eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon added to teacher-reward std normalization denominator.",
    )
    p.add_argument(
        "--enable-gt-stage-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable staged training schedule for execute/aggregate and epoch-wise GT reward annealing.",
    )
    p.add_argument(
        "--stage-execute-only-epochs",
        type=int,
        default=1,
        help="Epochs <= this value train execute entries only.",
    )
    p.add_argument(
        "--stage-aggregate-start-epoch",
        type=int,
        default=2,
        help="Aggregate entries are allowed from this epoch onward.",
    )
    p.add_argument(
        "--stage-wgt-zero-epoch",
        type=int,
        default=3,
        help="Set GT reward weight to zero at epochs >= this value.",
    )
    p.add_argument(
        "--stage-wgt-epoch1",
        type=float,
        default=-1.0,
        help="Optional explicit GT reward weight for epoch 1. Negative uses --reward-w-gt.",
    )
    p.add_argument(
        "--stage-wgt-epoch2",
        type=float,
        default=-1.0,
        help="Optional explicit GT reward weight for epoch 2. Negative uses 0.5 * epoch1 weight.",
    )

    p.add_argument("--h5py-file", type=Path, default=DEFAULT_H5PY_FILE)
    p.add_argument("--test-timeout-s", type=int, default=180)
    p.add_argument(
        "--dist-timeout-s",
        type=int,
        default=0,
        help="DDP/NCCL process-group timeout in seconds. <=0 uses an auto-sized timeout based on local group size and test timeout.",
    )
    p.add_argument(
        "--dist-timeout-buffer-s",
        type=int,
        default=900,
        help="Extra slack added to auto-sized DDP timeout to absorb generation / file I/O skew between ranks.",
    )
    p.add_argument(
        "--save-candidate-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-candidate snapshot files under run_dir/detailed_logs/snapshots.",
    )

    p.add_argument("--eval-after-train", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval-max-new-tokens", type=int, default=2048)
    p.add_argument("--eval-temperature", type=float, default=0.0)
    p.add_argument("--eval-top-p", type=float, default=1.0)

    args = p.parse_args(argv)
    if not bool(args.include_execute_entries) and not bool(args.include_aggregate_entries):
        raise ValueError("At least one of --include-execute-entries/--include-aggregate-entries must be enabled.")
    if int(args.aggregate_min_parents) < 2:
        args.aggregate_min_parents = 2
    if int(args.num_subtask_experts) <= 0:
        raise ValueError("--num-subtask-experts must be > 0.")
    if int(args.subtask_top_k) <= 0:
        raise ValueError("--subtask-top-k must be > 0.")
    if int(args.subtask_top_k) > int(args.num_subtask_experts):
        raise ValueError("--subtask-top-k cannot exceed --num-subtask-experts.")
    if int(args.aggregate_parent_max_experts) > int(args.num_subtask_experts):
        raise ValueError("--aggregate-parent-max-experts cannot exceed --num-subtask-experts.")
    if int(args.diversity_max_resample) <= 0:
        raise ValueError("--diversity-max-resample must be > 0.")
    if float(args.grpo_min_reward_std) < 0.0:
        raise ValueError("--grpo-min-reward-std must be >= 0.")
    if int(args.hierarchical_local_routes) < 0:
        raise ValueError("--hierarchical-local-routes must be >= 0.")
    if float(args.tf_ce_weight) < 0.0:
        raise ValueError("--tf-ce-weight must be >= 0.")
    if int(args.tf_ce_max_target_tokens) <= 0:
        raise ValueError("--tf-ce-max-target-tokens must be > 0.")
    if int(args.tf_ce_zero_epoch) <= 0:
        raise ValueError("--tf-ce-zero-epoch must be >= 1.")
    if float(args.tf_reward_weight) < 0.0:
        raise ValueError("--tf-reward-weight must be >= 0.")
    if float(args.tf_reward_zclip) < 0.0:
        raise ValueError("--tf-reward-zclip must be >= 0.")
    if float(args.tf_reward_eps) <= 0.0:
        raise ValueError("--tf-reward-eps must be > 0.")
    if int(args.dist_timeout_s) < 0:
        raise ValueError("--dist-timeout-s must be >= 0.")
    if int(args.dist_timeout_buffer_s) < 0:
        raise ValueError("--dist-timeout-buffer-s must be >= 0.")
    if int(args.save_every_samples) < 0:
        raise ValueError("--save-every-samples must be >= 0.")
    if int(args.save_every_epochs) < 0:
        raise ValueError("--save-every-epochs must be >= 0.")
    if int(args.stage_execute_only_epochs) < 0:
        raise ValueError("--stage-execute-only-epochs must be >= 0.")
    if int(args.stage_aggregate_start_epoch) <= 0:
        raise ValueError("--stage-aggregate-start-epoch must be >= 1.")
    if int(args.stage_wgt_zero_epoch) <= 0:
        raise ValueError("--stage-wgt-zero-epoch must be >= 1.")
    args.curriculum_progressive_min_fraction = min(1.0, max(0.05, float(args.curriculum_progressive_min_fraction)))
    _apply_disable_gt_code_signals(args)
    return args



def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        train_grpo(args)
    finally:
        if grpo_base._is_dist_ready():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
