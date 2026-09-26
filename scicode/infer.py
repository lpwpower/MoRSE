"""SciCode TaskGraph pipeline with SRDD/BioPlanner-style scheduling.

Key behavior:
- execute agent runs on every node (current step generation);
- aggregate agent runs only when a node has multiple parents (and optionally for multi-leaf final merge);
- SciCode step/general tests remain the evaluator.
- prompt construction is train-aligned (strict non-chat template).
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from scicode import pipeline as scipipe


DEFAULT_OUTPUT_ROOT = scipipe.SCICODE_ROOT / "runs" / "scicode_taskgraph_runs_srddstyle"


@dataclass
class AggregateResult:
    snapshot: Dict[str, str]
    status: str
    used_agent: bool
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    error: str


class RetryMixMoleTextGenerator(scipipe.MoleTextGenerator):
    """MoLE generator with retry-aware router selection policy.

    Attempt policy:
    1) attempt #1: greedy top-1 pair (stable baseline via greedy_topk)
    2) attempt #2: top-k pool similarity-probability sampling
    3) attempt #3+: top-k pool sampling with pair de-dup against previous attempts in same retry context
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        device: torch.device,
        gen_cfg,
        router,
        title_embedder,
        num_role_experts: int = 0,
        subtask_expert_offset: int = 0,
        retry_sample_pool_k: int = 4,
        retry_sampling_temperature: float = 1.0,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            gen_cfg=gen_cfg,
            router=router,
            title_embedder=title_embedder,
            num_role_experts=int(num_role_experts),
            subtask_expert_offset=int(subtask_expert_offset),
        )
        self._retry_sample_pool_k = max(int(retry_sample_pool_k), 1)
        self._retry_sampling_temperature = max(float(retry_sampling_temperature), 1e-6)
        self._retry_key = ""
        self._retry_attempt = 1
        self._retry_history: Dict[str, List[Tuple[int, ...]]] = {}

    def set_retry_context(self, *, retry_key: str, attempt: int, reset: bool = False) -> None:
        key = str(retry_key or "").strip()
        if key and bool(reset):
            self._retry_history.pop(key, None)
        self._retry_key = key
        self._retry_attempt = max(int(attempt), 1)

    @staticmethod
    def _canonical_pair(ids: torch.Tensor) -> Tuple[int, ...]:
        return tuple(sorted(int(v) for v in ids.detach().cpu().tolist()))

    def _remember_pair(self, ids: torch.Tensor) -> None:
        key = str(self._retry_key or "").strip()
        if not key:
            return
        hist = self._retry_history.setdefault(key, [])
        hist.append(self._canonical_pair(ids))
        if len(hist) > 8:
            del hist[:-8]

    def _sample_weighted_topk(
        self,
        *,
        logits: torch.Tensor,
        avoid_pairs: Optional[Set[Tuple[int, ...]]] = None,
    ) -> torch.Tensor:
        if logits.dim() != 2 or logits.size(0) != 1:
            raise ValueError("retry sampling expects router logits shape [1, num_experts].")

        logits_1d = logits[0]
        num_experts = int(logits_1d.numel())
        if num_experts <= 0:
            raise ValueError("router has no experts.")

        select_k = min(int(self._router.cfg.top_k), num_experts)
        pool_k = min(max(int(self._retry_sample_pool_k), select_k), num_experts)

        top_vals, top_ids = torch.topk(logits_1d, k=pool_k, dim=-1)
        probs = torch.softmax(top_vals / float(self._retry_sampling_temperature), dim=-1)
        avoid = set(avoid_pairs or set())

        chosen: Optional[torch.Tensor] = None
        for _ in range(64):
            local_ids = torch.multinomial(probs, num_samples=select_k, replacement=False)
            sampled = top_ids[local_ids]
            sort_idx = torch.argsort(logits_1d[sampled], descending=True)
            sampled = sampled[sort_idx]
            chosen = sampled
            if self._canonical_pair(sampled) not in avoid:
                return sampled

        if avoid:
            best_ids: Optional[torch.Tensor] = None
            best_score: Optional[float] = None
            for combo in itertools.combinations(range(pool_k), select_k):
                idx_tensor = top_ids[list(combo)]
                if self._canonical_pair(idx_tensor) in avoid:
                    continue
                score = float(logits_1d[idx_tensor].sum().item())
                if best_score is None or score > best_score:
                    best_score = score
                    best_ids = idx_tensor
            if best_ids is not None:
                sort_idx = torch.argsort(logits_1d[best_ids], descending=True)
                return best_ids[sort_idx]

        if chosen is not None:
            return chosen
        greedy_ids, _ = self._router.greedy_topk(logits)
        return greedy_ids

    def generate(self, prompt: str) -> str:
        subtask_text = self._current_subtask_text or prompt
        with torch.no_grad():
            title_emb = self._title_embedder(subtask_text)
            logits = self._router(title_emb=title_emb)

            attempt = max(int(self._retry_attempt), 1)
            if attempt == 1:
                subtask_ids, _ = self._router.greedy_topk(logits)
            elif attempt == 2:
                subtask_ids = self._sample_weighted_topk(logits=logits, avoid_pairs=None)
            else:
                key = str(self._retry_key or "").strip()
                seen_pairs = set(self._retry_history.get(key, [])) if key else set()
                subtask_ids = self._sample_weighted_topk(logits=logits, avoid_pairs=seen_pairs)

            self._remember_pair(subtask_ids)

            if self._num_role_experts > 0:
                role_id = 1 if (str(subtask_text).strip().lower().startswith("aggregate_for_") and self._num_role_experts >= 2) else 0
                role_tensor = torch.tensor([int(role_id)], dtype=subtask_ids.dtype, device=subtask_ids.device)
                expert_ids = torch.cat([role_tensor, subtask_ids + int(self._subtask_expert_offset)], dim=0)
            else:
                expert_ids = subtask_ids

            text, _prompt_ids, _gen_ids = self._mole.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
        return text



def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_prefill_python_fence(prompt: str, *, enabled: bool) -> str:
    if not enabled:
        return str(prompt or "").rstrip() + "\n"
    return str(prompt or "").rstrip() + "\n\n```python\n"


def _srdd_step_text(step: dict, *, with_background: bool) -> str:
    desc = str(step.get("step_description_prompt") or "").strip()
    if with_background:
        bg = str(step.get("step_background") or "").strip()
        if bg:
            return f"{desc}\n{bg}".strip()
    return desc


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



def _extract_step_code_for_eval(step: dict, raw_python: str) -> str:
    python_code = str(raw_python or "").strip()
    if not python_code:
        return ""
    try:
        fn_name = scipipe._extract_function_name(str(step.get("function_header") or ""))
    except Exception:
        return python_code
    parsed = scipipe._get_function_from_code(python_code, fn_name)
    return (parsed or "").strip()



def _sorted_step_ids(step_ids: Iterable[str], step_order: Dict[str, int]) -> List[str]:
    return sorted(set(str(s) for s in step_ids), key=lambda sid: step_order.get(sid, 10**9))



def _hard_merge_snapshots(
    *,
    parent_snapshots: Sequence[Dict[str, str]],
    step_order: Dict[str, int],
) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for snapshot in parent_snapshots:
        for sid in _sorted_step_ids(snapshot.keys(), step_order):
            code = str(snapshot.get(sid) or "").strip()
            if sid not in merged and code:
                merged[sid] = code
    return merged



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



def _make_children_map(predecessors: Dict[int, List[int]]) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {nid: [] for nid in predecessors}
    for child, parents in predecessors.items():
        for parent in parents:
            children.setdefault(parent, []).append(child)
    for nid in children:
        children[nid] = sorted(set(children[nid]))
    return children



def _summarize_samples(rows: Sequence[dict]) -> dict:
    total = len(rows)
    ok = sum(1 for row in rows if row.get("overall_status", row.get("status")) == "ok")
    failed = sum(1 for row in rows if row.get("overall_status", row.get("status")) == "failed")
    skipped = sum(1 for row in rows if row.get("overall_status", row.get("status")) == "skipped")
    problem_correct_samples = 0
    general_pass_samples = 0
    for row in rows:
        tested_steps = row.get("tested_steps")
        passed_steps = row.get("passed_steps")
        default_problem_correctness = 1 if isinstance(tested_steps, (int, float)) and tested_steps > 0 and passed_steps == tested_steps else 0
        if int(row.get("problem_correctness", default_problem_correctness)) == 1:
            problem_correct_samples += 1
        if str(row.get("general_test_status", "")).strip().lower() == "pass":
            general_pass_samples += 1

    def _avg(values: Iterable[float]) -> float:
        values = list(values)
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "total_samples": total,
        "ok_samples": ok,
        "failed_samples": failed,
        "skipped_samples": skipped,
        "problem_correct_samples": problem_correct_samples,
        "problem_correct_rate": (float(problem_correct_samples) / float(total)) if total else 0.0,
        "general_pass_samples": general_pass_samples,
        "general_pass_rate": (float(general_pass_samples) / float(total)) if total else 0.0,
        "mean_step_pass_rate": _avg(
            float(row.get("step_pass_rate", 0.0))
            for row in rows
            if isinstance(row.get("step_pass_rate"), (int, float))
        ),
        "mean_prompt_tokens": _avg(
            float(row.get("total_prompt_tokens", 0.0))
            for row in rows
            if isinstance(row.get("total_prompt_tokens"), (int, float))
        ),
        "mean_completion_tokens": _avg(
            float(row.get("total_completion_tokens", 0.0))
            for row in rows
            if isinstance(row.get("total_completion_tokens"), (int, float))
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _set_mole_subtask(generator, text: str) -> None:
    if isinstance(generator, scipipe.MoleTextGenerator):
        generator.set_subtask_text(str(text or "").strip())


def _set_mole_retry_context(generator, *, retry_key: str, attempt: int, reset: bool = False) -> None:
    if hasattr(generator, "set_retry_context"):
        try:
            generator.set_retry_context(retry_key=str(retry_key or ""), attempt=int(attempt), reset=bool(reset))
        except Exception:
            return


def _append_retry_hint(prompt: str, hint_block: str, *, prefill_python_fence: bool) -> str:
    if not hint_block:
        return prompt
    if bool(prefill_python_fence):
        suffix = "\n\n```python\n"
        if str(prompt).endswith(suffix):
            return str(prompt)[: -len(suffix)] + "\n\n" + str(hint_block).strip() + suffix
    return str(prompt) + "\n\n" + str(hint_block).strip()


def _resolve_base_snapshot(
    *,
    args: argparse.Namespace,
    generator,
    node_dir: Path,
    problem: dict,
    step_id: str,
    expected_step_ids: Sequence[str],
    step_by_id: Dict[str, dict],
    parent_ids: Sequence[int],
    parent_snapshots: Sequence[Dict[str, str]],
    step_order: Dict[str, int],
) -> AggregateResult:
    if not parent_ids:
        return AggregateResult(snapshot={}, status="root", used_agent=False, attempts=0, prompt_tokens=0, completion_tokens=0, error="")

    if len(parent_ids) == 1:
        return AggregateResult(
            snapshot=dict(parent_snapshots[0]),
            status="single_parent_passthrough",
            used_agent=False,
            attempts=0,
            prompt_tokens=0,
            completion_tokens=0,
            error="",
        )

    if not bool(args.aggregate_enabled):
        if str(args.no_aggregate_parent_strategy) == "hard_merge":
            merged = _hard_merge_snapshots(parent_snapshots=parent_snapshots, step_order=step_order)
            return AggregateResult(
                snapshot=merged,
                status="aggregate_disabled_hard_merge",
                used_agent=False,
                attempts=0,
                prompt_tokens=0,
                completion_tokens=0,
                error="",
            )
        return AggregateResult(
            snapshot=dict(parent_snapshots[0]),
            status="aggregate_disabled_first_parent",
            used_agent=False,
            attempts=0,
            prompt_tokens=0,
            completion_tokens=0,
            error="",
        )

    baseline = _hard_merge_snapshots(parent_snapshots=parent_snapshots, step_order=step_order)
    if str(args.aggregate_mode) == "hard":
        return AggregateResult(
            snapshot=baseline,
            status="hard_merge",
            used_agent=False,
            attempts=1,
            prompt_tokens=0,
            completion_tokens=0,
            error="",
        )

    parent_payload = list(zip(parent_ids, parent_snapshots))
    total_prompt_tokens = 0
    total_completion_tokens = 0
    last_error = ""

    for attempt in range(1, max(int(args.aggregate_max_attempts), 1) + 1):
        prompt = _build_aggregate_prompt_nonchat(
            problem=problem,
            next_step_id=step_id,
            expected_step_ids=expected_step_ids,
            step_by_id=step_by_id,
            parent_snapshots=parent_payload,
            prompt_style=str(args.prompt_style),
            prefill_python_fence=bool(args.prefill_python_fence),
        )
        if last_error:
            prompt = _append_retry_hint(
                prompt,
                "Previous aggregate attempt failed to produce useful merged functions.\n"
                + f"Failure hint:\n{last_error}",
                prefill_python_fence=bool(args.prefill_python_fence),
            )

        if attempt == 1:
            (node_dir / "aggregate_prompt.txt").write_text(prompt, encoding="utf-8")
        (node_dir / f"aggregate_prompt_attempt_{attempt}.txt").write_text(prompt, encoding="utf-8")

        _set_mole_retry_context(
            generator,
            retry_key=f"aggregate::{node_dir}::{step_id}",
            attempt=attempt,
            reset=(attempt == 1),
        )
        _set_mole_subtask(generator, f"aggregate_for_{step_id}")
        prompt_tokens = generator.count_tokens(prompt)
        response = generator.generate(prompt)
        completion_tokens = generator.count_tokens(response)
        total_prompt_tokens += int(prompt_tokens)
        total_completion_tokens += int(completion_tokens)

        (node_dir / f"aggregate_response_attempt_{attempt}.txt").write_text(response, encoding="utf-8")
        python_code = scipipe._extract_python_script(response)
        (node_dir / f"aggregate_python_attempt_{attempt}.py").write_text(python_code + "\n", encoding="utf-8")

        parsed = _parse_snapshot_functions_for_steps(
            python_code=python_code,
            step_by_id=step_by_id,
            allowed_step_ids=list(baseline.keys()),
        )
        if not parsed:
            last_error = "no recognized upstream functions extracted from aggregate output"
            continue

        merged = dict(baseline)
        for sid, code in parsed.items():
            if str(code or "").strip():
                merged[sid] = str(code).strip()

        return AggregateResult(
            snapshot=merged,
            status="aggregate_llm_ok",
            used_agent=True,
            attempts=attempt,
            prompt_tokens=int(total_prompt_tokens),
            completion_tokens=int(total_completion_tokens),
            error="",
        )

    return AggregateResult(
        snapshot=baseline,
        status="aggregate_llm_fallback_hard_merge",
        used_agent=True,
        attempts=max(int(args.aggregate_max_attempts), 1),
        prompt_tokens=int(total_prompt_tokens),
        completion_tokens=int(total_completion_tokens),
        error=last_error,
    )


def _make_generator(args: argparse.Namespace):
    if args.mole_checkpoint is not None:
        device = torch.device("cuda", int(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        ckpt_dir = scipipe._resolve_mole_ckpt_dir(Path(args.mole_checkpoint))
        inferred_subtask_experts = scipipe._infer_subtask_expert_count_from_router(ckpt_dir)
        inferred_total_experts = scipipe._infer_total_expert_count_from_lora(ckpt_dir)
        num_subtask_experts = int(
            inferred_subtask_experts
            if inferred_subtask_experts is not None
            else int(args.mole_num_subtask_experts)
        )
        num_role_experts = int(
            max(0, int(inferred_total_experts) - int(num_subtask_experts))
            if inferred_total_experts is not None
            else max(0, int(getattr(args, "mole_num_role_experts", 0)))
        )
        num_total_experts = int(
            inferred_total_experts
            if inferred_total_experts is not None
            else int(num_role_experts + num_subtask_experts)
        )
        model, tokenizer = scipipe._load_backbone(
            model_name=args.code_model_name,
            torch_dtype=str(args.torch_dtype),
            device=device,
        )
        lora_cfg = scipipe.LoRAConfig(
            num_experts=int(num_total_experts),
            top_k=int(args.mole_subtask_top_k),
            rank=int(args.mole_lora_rank),
            alpha=float(args.mole_lora_alpha),
            target_modules=("q_proj", "v_proj", "o_proj"),
            last_n_layers=int(args.mole_lora_last_n_layers),
        )
        scipipe.inject_mole_lora(model, cfg=lora_cfg)
        router_cfg = scipipe.SubtaskRouterConfig(
            num_experts=int(num_subtask_experts),
            top_k=int(args.mole_subtask_top_k),
        )
        router = scipipe.SubtaskRouter(router_cfg).to(device)
        title_embedder = scipipe.TitleEmbedder(model=model, tokenizer=tokenizer, out_dim=router_cfg.title_emb_dim).to(device)
        scipipe._load_mole_checkpoint(
            ckpt_dir=ckpt_dir,
            device=device,
            router=router,
            title_embedder=title_embedder,
            model=model,
        )
        gen_cfg = scipipe.MoLEGenerationConfig(
            model_name=args.code_model_name,
            max_new_tokens=int(args.code_max_new_tokens),
            temperature=float(args.code_temperature),
            top_p=0.95,
            torch_dtype=str(args.torch_dtype),
            device=int(args.device),
        )
        return RetryMixMoleTextGenerator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            gen_cfg=gen_cfg,
            router=router,
            title_embedder=title_embedder,
            num_role_experts=int(num_role_experts),
            subtask_expert_offset=int(num_role_experts),
            retry_sample_pool_k=int(args.mole_retry_sample_pool_k),
            retry_sampling_temperature=float(args.mole_retry_sample_temperature),
        )

    cfg = scipipe.HFGenerationConfig(
        model_name=args.code_model_name,
        device=args.device,
        device_map=(None if str(args.device_map).strip().lower() == "none" else args.device_map),
        max_new_tokens=args.code_max_new_tokens,
        temperature=args.code_temperature,
        torch_dtype=args.torch_dtype,
    )
    return scipipe.HFSubprocessTextGenerator(cfg) if args.code_subprocess else scipipe.HFTextGenerator(cfg)



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SciCode TaskGraph SRDD-style pipeline (merge-only aggregate).")

    p.add_argument("--dataset", type=Path, default=scipipe.DEFAULT_DATASET)
    p.add_argument("--graph-root", type=Path, default=scipipe.DEFAULT_GRAPH_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--timestamp", type=str, default=None)
    p.add_argument("--problem-id", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-existing-any-status", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fallback-chain-graph", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--code-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--code-max-new-tokens", type=int, default=4096)
    p.add_argument("--code-temperature", type=float, default=0.0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--torch-dtype", type=str, default="bfloat16")
    p.add_argument("--code-subprocess", action="store_true")
    p.add_argument("--keep-model-loaded", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--mole-checkpoint", type=Path, default=None)
    p.add_argument("--mole-num-subtask-experts", type=int, default=4)
    p.add_argument("--mole-subtask-top-k", type=int, default=2)
    p.add_argument("--mole-lora-rank", type=int, default=8)
    p.add_argument("--mole-lora-alpha", type=float, default=16.0)
    p.add_argument("--mole-lora-last-n-layers", type=int, default=8)
    p.add_argument(
        "--mole-retry-sample-pool-k",
        type=int,
        default=4,
        help="Retry#2/#3 router sampling pool size (choose from top-k experts by similarity).",
    )
    p.add_argument(
        "--mole-retry-sample-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for retry#2/#3 similarity-probability sampling.",
    )

    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument(
        "--propagate-failed-candidate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If a node fails after retries, still propagate its last parseable candidate to downstream nodes.",
    )

    p.add_argument("--aggregate-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aggregate-mode", choices=["llm", "hard"], default="llm")
    p.add_argument("--aggregate-max-attempts", type=int, default=2)
    p.add_argument(
        "--no-aggregate-parent-strategy",
        choices=["first", "hard_merge"],
        default="first",
        help="When --no-aggregate-enabled and parent_count>1, choose baseline parent context policy.",
    )
    p.add_argument(
        "--aggregate-stop-on-failure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If aggregate agent fails (llm mode), treat sample as failed immediately.",
    )
    p.add_argument(
        "--final-aggregate-leaves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If multiple leaf snapshots remain, run the same merge logic to build final snapshot.",
    )

    p.add_argument(
        "--with-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror SciCode gencode mode switch for prompt construction.",
    )
    p.add_argument(
        "--prompt-template",
        choices=["auto", "background", "default"],
        default="auto",
        help="Legacy compatibility flag (ignored in train-aligned prompt mode).",
    )
    p.add_argument(
        "--prompt-style",
        type=str,
        default="strict",
        choices=("strict", "minimal"),
        help="Train-aligned non-chat prompt policy.",
    )
    p.add_argument(
        "--prefill-python-fence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append trailing ```python to prompt so model continues directly as code.",
    )

    p.add_argument("--no-eval", action="store_false", dest="eval", help="Skip SciCode step/general tests.")
    p.set_defaults(eval=True)
    p.add_argument("--h5py-file", type=Path, default=scipipe.DEFAULT_H5PY_FILE)
    p.add_argument("--test-timeout-s", type=int, default=180)

    return p.parse_args(argv)



def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.gpus and str(args.gpus).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus).strip()

    problems = scipipe._read_jsonl(args.dataset)
    if args.problem_id is not None:
        problems = [row for row in problems if str(row.get("problem_id")) == str(args.problem_id)]
        if not problems:
            raise ValueError(f"problem_id={args.problem_id} not found in {args.dataset}")
    if args.max_samples and args.max_samples > 0:
        problems = problems[: args.max_samples]
    if not problems:
        raise RuntimeError("No problems selected.")

    if args.eval:
        scipipe._check_eval_prerequisites(args.h5py_file)

    graph_index = scipipe._build_graph_root_index(args.graph_root) if args.graph_root else {}

    timestamp = (args.timestamp or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    sample_rows: List[dict] = []
    generator = _make_generator(args) if args.keep_model_loaded else None

    run_env = os.environ.copy()
    existing_pp = run_env.get("PYTHONPATH", "")
    scicode_src = str(scipipe.SCICODE_ROOT / "src")
    run_env["PYTHONPATH"] = f"{scicode_src}:{existing_pp}" if existing_pp else scicode_src

    for sample_idx, problem in enumerate(problems, start=1):
        sample_name = scipipe._sample_dir_name(sample_idx, problem)
        sample_dir = run_root / sample_name
        log_dir = sample_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        sample_metrics_path = log_dir / "sample_metrics.json"

        if args.skip_existing_any_status and sample_metrics_path.exists():
            try:
                payload = json.loads(sample_metrics_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if payload:
                sample_rows.append(payload)
            continue

        if args.skip_existing and sample_metrics_path.exists():
            try:
                payload = json.loads(sample_metrics_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if payload.get("status") == "ok":
                sample_rows.append(payload)
                continue

        problem_id = str(problem.get("problem_id") or "")
        problem_name = str(problem.get("problem_name") or f"problem_{problem_id}")
        step_by_id, step_order = scipipe._build_step_maps(problem)

        (sample_dir / "sample.json").write_text(
            json.dumps(
                {
                    "problem_id": problem_id,
                    "problem_name": problem_name,
                    "num_sub_steps": len(problem.get("sub_steps") or []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        graph_path = sample_dir / "task_graph.json"
        graph_source = "missing"
        src_dir = graph_index.get(problem_id)
        if src_dir and (src_dir / "task_graph.json").exists():
            graph_source = str(src_dir / "task_graph.json")
            graph_path.write_text((src_dir / "task_graph.json").read_text(encoding="utf-8"), encoding="utf-8")
            graph_gen_src = src_dir / "graph_generation.json"
            if graph_gen_src.exists():
                (sample_dir / "graph_generation.json").write_text(graph_gen_src.read_text(encoding="utf-8"), encoding="utf-8")
        elif args.fallback_chain_graph:
            graph_source = "chain_fallback"
            _write_json(graph_path, scipipe._build_chain_graph_json(problem))
            _write_json(sample_dir / "graph_generation.json", {"status": "ok", "edge_source": "chain_fallback"})
        else:
            row = {
                "status": "failed",
                "problem_id": problem_id,
                "problem_name": problem_name,
                "error": "task_graph_missing",
                "graph_source": graph_source,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _write_json(sample_metrics_path, row)
            sample_rows.append(row)
            if args.stop_on_failure:
                break
            continue

        spec = scipipe.convert_taskgraph(graph_path)
        predecessors = scipipe._build_predecessors(spec)
        children = _make_children_map(predecessors)

        node_snapshots: Dict[int, Dict[str, str]] = {}
        node_logs: List[dict] = []
        failed = False
        skipped_count = 0
        tested_steps = 0
        step_passed = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        aggregate_calls = 0
        aggregate_agent_calls = 0
        aggregate_fallback_calls = 0
        merge_nodes = 0
        run_start = time.time()

        if generator is None:
            generator = _make_generator(args)

        for node_id in sorted(spec.node_metadata):
            node_meta = spec.node_metadata[node_id]
            step_id = str(node_meta.original_id)
            step = step_by_id.get(step_id)
            node_dir = log_dir / f"node_{node_id:02d}_{scipipe._sanitize(step_id)}"
            node_dir.mkdir(parents=True, exist_ok=True)

            if step is None:
                node_logs.append(
                    {
                        "node_id": node_id,
                        "step_id": step_id,
                        "status": "failed",
                        "error": "missing_step_in_dataset",
                    }
                )
                failed = True
                if args.stop_on_failure:
                    break
                node_snapshots[node_id] = {}
                continue

            parent_ids = sorted(predecessors.get(node_id, []))
            if len(parent_ids) > 1:
                merge_nodes += 1
            missing_parents = [pid for pid in parent_ids if pid not in node_snapshots]
            if missing_parents:
                node_logs.append(
                    {
                        "node_id": node_id,
                        "step_id": step_id,
                        "status": "failed",
                        "error": f"missing_parent_snapshots:{missing_parents}",
                    }
                )
                failed = True
                if args.stop_on_failure:
                    break
                node_snapshots[node_id] = {}
                continue

            parent_snapshots = [dict(node_snapshots[pid]) for pid in parent_ids]
            expected_step_ids = _sorted_step_ids({sid for snap in parent_snapshots for sid in snap.keys()}, step_order)

            agg_res = _resolve_base_snapshot(
                args=args,
                generator=generator,
                node_dir=node_dir,
                problem=problem,
                step_id=step_id,
                expected_step_ids=expected_step_ids,
                step_by_id=step_by_id,
                parent_ids=parent_ids,
                parent_snapshots=parent_snapshots,
                step_order=step_order,
            )
            base_snapshot = dict(agg_res.snapshot)
            total_prompt_tokens += int(agg_res.prompt_tokens)
            total_completion_tokens += int(agg_res.completion_tokens)
            if len(parent_ids) > 1:
                aggregate_calls += 1
                if agg_res.used_agent:
                    aggregate_agent_calls += 1
                if "fallback" in str(agg_res.status):
                    aggregate_fallback_calls += 1
            if agg_res.error:
                _write_json(node_dir / "aggregate_error.json", {"error": agg_res.error, "status": agg_res.status})
                if bool(args.aggregate_stop_on_failure) and len(parent_ids) > 1 and bool(args.aggregate_enabled):
                    failed = True
                    node_logs.append(
                        {
                            "node_id": node_id,
                            "step_id": step_id,
                            "status": "failed",
                            "error": f"aggregate_failed:{agg_res.error}",
                        }
                    )
                    if args.stop_on_failure:
                        break

            if scipipe._should_use_fixed_step(problem_id, step_id):
                fixed_code = scipipe._load_fixed_step(problem_id, step_id)
                new_snapshot = dict(base_snapshot)
                new_snapshot[step_id] = fixed_code
                node_snapshots[node_id] = new_snapshot
                skipped_count += 1
                (sample_dir / "generated_code").mkdir(parents=True, exist_ok=True)
                assembled_fixed = scipipe._assemble_program_code(
                    dependencies=str(problem.get("required_dependencies") or ""),
                    ancestor_step_ids=_sorted_step_ids(base_snapshot.keys(), step_order),
                    solved_functions=base_snapshot,
                    current_python_code=fixed_code,
                )
                (sample_dir / "generated_code" / f"{step_id}.py").write_text(assembled_fixed, encoding="utf-8")
                node_logs.append(
                    {
                        "node_id": node_id,
                        "step_id": step_id,
                        "status": "fixed_step",
                        "parent_ids": parent_ids,
                        "aggregate_status": agg_res.status,
                        "aggregate_used_agent": bool(agg_res.used_agent),
                    }
                )
                continue

            ancestor_node_ids = scipipe._collect_ancestors(node_id, predecessors)
            ancestor_step_ids = [str(spec.node_metadata[a].original_id) for a in sorted(ancestor_node_ids)]
            ancestor_step_ids = [sid for sid in ancestor_step_ids if sid in base_snapshot]
            ancestor_step_ids = _sorted_step_ids(ancestor_step_ids, step_order)

            prompt_base = _build_execute_prompt_nonchat(
                problem=problem,
                step=step,
                step_id=step_id,
                ancestor_step_ids=ancestor_step_ids,
                solved_functions=base_snapshot,
                step_by_id=step_by_id,
                with_background=bool(args.with_background),
                prompt_style=str(args.prompt_style),
                prefill_python_fence=bool(args.prefill_python_fence),
            )

            tested_this_node = False
            last_error = ""
            last_parsed = ""
            last_python = ""

            for attempt in range(1, max(int(args.max_attempts), 1) + 1):
                prompt = prompt_base
                if last_error:
                    prompt = _append_retry_hint(
                        prompt,
                        "Previous attempt failed.\n"
                        + scipipe.SCICODE_RETRY_GUIDELINES
                        + "\n"
                        + f"Failure hint:\n{last_error}",
                        prefill_python_fence=bool(args.prefill_python_fence),
                    )
                (node_dir / f"prompt_attempt_{attempt}.txt").write_text(prompt, encoding="utf-8")

                _set_mole_retry_context(
                    generator,
                    retry_key=f"execute::{node_dir}::{step_id}",
                    attempt=attempt,
                    reset=(attempt == 1),
                )
                _set_mole_subtask(generator, _srdd_step_text(step, with_background=bool(args.with_background)))
                prompt_tokens = generator.count_tokens(prompt)
                response = generator.generate(prompt)
                completion_tokens = generator.count_tokens(response)
                total_prompt_tokens += int(prompt_tokens)
                total_completion_tokens += int(completion_tokens)

                (node_dir / f"response_attempt_{attempt}.txt").write_text(response, encoding="utf-8")
                python_code = scipipe._extract_python_script(response)
                last_python = python_code
                (node_dir / f"python_attempt_{attempt}.py").write_text(python_code + "\n", encoding="utf-8")

                parsed_function = _extract_step_code_for_eval(step, python_code)
                last_parsed = parsed_function or last_parsed
                if not parsed_function:
                    last_error = "empty parsed function code"
                    continue

                assembled_code = scipipe._assemble_program_code(
                    dependencies=str(problem.get("required_dependencies") or ""),
                    ancestor_step_ids=ancestor_step_ids,
                    solved_functions=base_snapshot,
                    current_python_code=python_code,
                )
                (sample_dir / "generated_code").mkdir(parents=True, exist_ok=True)
                (sample_dir / "generated_code" / f"{step_id}.py").write_text(assembled_code, encoding="utf-8")

                step_test_result: Optional[scipipe.ScriptRunResult] = None
                if args.eval:
                    if not tested_this_node:
                        tested_steps += 1
                        tested_this_node = True
                    step_test_result = scipipe._run_step_test(
                        sample_dir=sample_dir,
                        step_id=step_id,
                        assembled_code=assembled_code,
                        test_cases=list(step.get("test_cases") or []),
                        h5py_file=args.h5py_file,
                        timeout_s=int(args.test_timeout_s),
                        env=run_env,
                    )
                    _write_json(
                        node_dir / f"step_test_attempt_{attempt}.json",
                        {
                            "status": step_test_result.status,
                            "return_code": int(step_test_result.return_code),
                            "elapsed_ms": int(step_test_result.elapsed_ms),
                            "stdout_tail": step_test_result.stdout[-4000:],
                            "stderr_tail": step_test_result.stderr[-4000:],
                            "script_path": str(step_test_result.script_path),
                        },
                    )
                    if not step_test_result.passed:
                        last_error = (
                            f"step test {step_test_result.status}; rc={step_test_result.return_code}; "
                            f"stderr_tail={step_test_result.stderr[-500:]}"
                        )
                        continue
                    step_passed += 1

                accepted_snapshot = dict(base_snapshot)
                accepted_snapshot[step_id] = parsed_function
                node_snapshots[node_id] = accepted_snapshot
                node_logs.append(
                    {
                        "node_id": node_id,
                        "step_id": step_id,
                        "status": "ok",
                        "attempt": attempt,
                        "parent_ids": parent_ids,
                        "ancestor_steps": ancestor_step_ids,
                        "prompt_tokens": int(prompt_tokens),
                        "completion_tokens": int(completion_tokens),
                        "aggregate_status": agg_res.status,
                        "aggregate_used_agent": bool(agg_res.used_agent),
                        "tested": bool(args.eval),
                        "test_status": (step_test_result.status if step_test_result is not None else "skipped"),
                    }
                )
                break
            else:
                failed = True
                fallback_snapshot = dict(base_snapshot)
                propagated = False
                if bool(args.propagate_failed_candidate) and str(last_parsed or "").strip():
                    fallback_snapshot[step_id] = str(last_parsed).strip()
                    propagated = True
                    assembled_code = scipipe._assemble_program_code(
                        dependencies=str(problem.get("required_dependencies") or ""),
                        ancestor_step_ids=ancestor_step_ids,
                        solved_functions=base_snapshot,
                        current_python_code=last_python,
                    )
                    (sample_dir / "generated_code").mkdir(parents=True, exist_ok=True)
                    (sample_dir / "generated_code" / f"{step_id}.py").write_text(assembled_code, encoding="utf-8")

                node_snapshots[node_id] = fallback_snapshot
                node_logs.append(
                    {
                        "node_id": node_id,
                        "step_id": step_id,
                        "status": "failed",
                        "attempts": max(int(args.max_attempts), 1),
                        "error": last_error or "unknown",
                        "propagated_failed_candidate": bool(propagated),
                        "parent_ids": parent_ids,
                        "ancestor_steps": ancestor_step_ids,
                        "aggregate_status": agg_res.status,
                        "aggregate_used_agent": bool(agg_res.used_agent),
                    }
                )
                if args.stop_on_failure:
                    break

        # Resolve final snapshot (single sink or merged leaves).
        final_snapshot: Dict[str, str] = {}
        solved_node_ids = sorted(node_snapshots.keys())
        leaf_ids = [nid for nid in solved_node_ids if len(children.get(nid, [])) == 0]

        final_agg_meta = {
            "status": "skipped",
            "used_agent": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "leaf_count": len(leaf_ids),
        }

        if not leaf_ids and solved_node_ids:
            final_snapshot = dict(node_snapshots[solved_node_ids[-1]])
            final_agg_meta["status"] = "fallback_last_node"
        elif len(leaf_ids) == 1:
            final_snapshot = dict(node_snapshots[leaf_ids[0]])
            final_agg_meta["status"] = "single_leaf"
        elif len(leaf_ids) > 1:
            leaf_snaps = [dict(node_snapshots[nid]) for nid in leaf_ids]
            if bool(args.final_aggregate_leaves):
                final_res = _resolve_base_snapshot(
                    args=args,
                    generator=generator,
                    node_dir=log_dir,
                    problem=problem,
                    step_id="final",
                    expected_step_ids=_sorted_step_ids({sid for snap in leaf_snaps for sid in snap.keys()}, step_order),
                    step_by_id=step_by_id,
                    parent_ids=leaf_ids,
                    parent_snapshots=leaf_snaps,
                    step_order=step_order,
                )
                final_snapshot = dict(final_res.snapshot)
                final_agg_meta = {
                    "status": f"final_{final_res.status}",
                    "used_agent": bool(final_res.used_agent),
                    "prompt_tokens": int(final_res.prompt_tokens),
                    "completion_tokens": int(final_res.completion_tokens),
                    "leaf_count": len(leaf_ids),
                }
                total_prompt_tokens += int(final_res.prompt_tokens)
                total_completion_tokens += int(final_res.completion_tokens)
            else:
                final_snapshot = _hard_merge_snapshots(parent_snapshots=leaf_snaps, step_order=step_order)
                final_agg_meta["status"] = "final_hard_merge"

        general_result: Optional[scipipe.ScriptRunResult] = None
        general_status = "skipped"
        if args.eval and not failed and final_snapshot:
            final_step_ids = _sorted_step_ids(final_snapshot.keys(), step_order)
            final_code = (
                "\n\n".join(
                    [str(problem.get("required_dependencies") or "").strip()] + [str(final_snapshot[sid]).strip() for sid in final_step_ids]
                ).strip()
                + "\n"
            )
            general_tests = list(problem.get("general_tests") or [])
            if general_tests:
                sub_steps = list(problem.get("sub_steps") or [])
                general_target_group = str(sub_steps[-1].get("step_number") or problem_id) if sub_steps else problem_id
                general_result = scipipe._run_general_test(
                    sample_dir=sample_dir,
                    problem_id=problem_id,
                    general_target_group=general_target_group,
                    assembled_code=final_code,
                    general_tests=general_tests,
                    h5py_file=args.h5py_file,
                    timeout_s=int(args.test_timeout_s),
                    env=run_env,
                )
                general_status = general_result.status
                _write_json(
                    log_dir / "general_test.json",
                    {
                        "status": general_result.status,
                        "return_code": int(general_result.return_code),
                        "elapsed_ms": int(general_result.elapsed_ms),
                        "stdout_tail": general_result.stdout[-4000:],
                        "stderr_tail": general_result.stderr[-4000:],
                        "target_group": general_target_group,
                        "script_path": str(general_result.script_path),
                    },
                )
                if not general_result.passed:
                    failed = True
            else:
                general_status = "no_general_tests"

        elapsed_ms = int((time.time() - run_start) * 1000)
        step_pass_rate = (float(step_passed) / float(tested_steps)) if tested_steps else 0.0
        overall_status = "failed" if failed else "ok"
        problem_correctness = 1 if tested_steps > 0 and step_passed == tested_steps else 0

        sample_row = {
            "status": overall_status,
            "overall_status": overall_status,
            "problem_id": problem_id,
            "problem_name": problem_name,
            "graph_source": graph_source,
            "mode": "srddstyle_taskgraph",
            "aggregate_enabled": bool(args.aggregate_enabled),
            "aggregate_mode": str(args.aggregate_mode),
            "total_steps": len(step_order),
            "fixed_steps": skipped_count,
            "tested_steps": tested_steps,
            "passed_steps": step_passed,
            "step_pass_rate": step_pass_rate,
            "problem_correctness": problem_correctness,
            "general_test_status": general_status,
            "general_test_pass": 1 if str(general_status).lower() == "pass" else 0,
            "merge_nodes": int(merge_nodes),
            "aggregate_calls": int(aggregate_calls),
            "aggregate_agent_calls": int(aggregate_agent_calls),
            "aggregate_fallback_calls": int(aggregate_fallback_calls),
            "final_aggregate": final_agg_meta,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "elapsed_ms": elapsed_ms,
            "node_logs": node_logs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(sample_metrics_path, sample_row)
        sample_rows.append(sample_row)

        if not args.keep_model_loaded:
            generator = None
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if failed and args.stop_on_failure:
            break

    summary = {
        "dataset": str(args.dataset),
        "graph_root": str(args.graph_root),
        "output_root": str(run_root),
        "model_name": args.code_model_name,
        "eval_enabled": bool(args.eval),
        "mode": "srddstyle_taskgraph",
        "prompt_style": str(args.prompt_style),
        "prefill_python_fence": bool(args.prefill_python_fence),
        "aggregate_enabled": bool(args.aggregate_enabled),
        "aggregate_mode": str(args.aggregate_mode),
        "samples": sample_rows,
        "aggregate": _summarize_samples(sample_rows),
    }
    _write_json(run_root / "summary.json", summary)
    print(f"SciCode SRDD-style taskgraph pipeline complete. Output: {run_root}")


if __name__ == "__main__":
    main()
