"""SciCode TaskGraph pipeline with SRDD/BioPlanner-style scheduling.

Key behavior:
- execute agent runs on every node (current step generation);
- aggregate agent runs only when a node has multiple parents (and optionally for multi-leaf final merge);
- SciCode step/general tests remain the evaluator.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scicode import pipeline as scipipe


DEFAULT_OUTPUT_ROOT = scipipe.SCICODE_ROOT / "runs" / "scicode_taskgraph_runs_srddstyle"

AGGREGATE_GUIDELINES = (
    "You are the aggregate agent for SciCode task-graph execution.\n"
    "Merge multiple parent snapshots into ONE coherent upstream function set.\n"
    "Rules:\n"
    "1) Output Python code only (single ```python``` block).\n"
    "2) Keep only function/class implementations for already-completed upstream steps.\n"
    "3) Do not add tests or example code.\n"
    "4) Preserve function signatures compatible with provided function headers when possible."
)


@dataclass
class AggregateResult:
    snapshot: Dict[str, str]
    status: str
    used_agent: bool
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    error: str



def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def _select_prompt_template(prompt_template: str, with_background: bool) -> str:
    if prompt_template == "background":
        return scipipe.BACKGOUND_PROMPT_TEMPLATE
    if prompt_template == "default":
        return scipipe.DEFAULT_PROMPT_TEMPLATE
    return scipipe.BACKGOUND_PROMPT_TEMPLATE if with_background else scipipe.DEFAULT_PROMPT_TEMPLATE



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



def _build_aggregate_prompt(
    *,
    problem: dict,
    next_step_id: str,
    expected_step_ids: Sequence[str],
    step_by_id: Dict[str, dict],
    parent_snapshots: Sequence[Tuple[int, Dict[str, str]]],
) -> str:
    blocks: List[str] = [AGGREGATE_GUIDELINES, ""]

    deps = str(problem.get("required_dependencies") or "").strip()
    if deps:
        blocks.extend(["[Dependencies]", deps, ""])

    blocks.append(f"[Next node step id] {next_step_id}")
    blocks.append("")

    if expected_step_ids:
        blocks.append("[Expected upstream steps to preserve]")
        for sid in expected_step_ids:
            header = ""
            step = step_by_id.get(sid)
            if step is not None:
                header = str(step.get("function_header") or "").strip().splitlines()[0]
            blocks.append(f"- step {sid}: {header}")
        blocks.append("")

    blocks.append("[Parent snapshots]")
    for parent_id, snapshot in parent_snapshots:
        blocks.append(f"=== Parent node {parent_id} ===")
        if not snapshot:
            blocks.append("(empty snapshot)")
            continue
        for sid in expected_step_ids:
            code = str(snapshot.get(sid) or "").strip()
            if not code:
                continue
            blocks.append(f"# step {sid}")
            blocks.append(code)
            blocks.append("------")
    blocks.append("")
    blocks.append("Return merged upstream functions only.")
    return "\n".join(blocks).strip() + "\n"



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
        prompt = _build_aggregate_prompt(
            problem=problem,
            next_step_id=step_id,
            expected_step_ids=expected_step_ids,
            step_by_id=step_by_id,
            parent_snapshots=parent_payload,
        )
        if last_error:
            prompt = (
                prompt
                + "\n\nPrevious aggregate attempt failed to produce useful merged functions.\n"
                + f"Failure hint:\n{last_error}\n"
            )

        if attempt == 1:
            (node_dir / "aggregate_prompt.txt").write_text(prompt, encoding="utf-8")
        (node_dir / f"aggregate_prompt_attempt_{attempt}.txt").write_text(prompt, encoding="utf-8")

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
    prompt_template = _select_prompt_template(args.prompt_template, bool(args.with_background))

    timestamp = (args.timestamp or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    sample_rows: List[dict] = []
    generator = scipipe._make_generator(args) if args.keep_model_loaded else None

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
            generator = scipipe._make_generator(args)

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

            prompt_base, _ = scipipe._render_prompt(
                problem=problem,
                step=step,
                ancestor_step_ids=ancestor_step_ids,
                solved_functions=base_snapshot,
                step_by_id=step_by_id,
                with_background=bool(args.with_background),
                prompt_template=prompt_template,
            )

            tested_this_node = False
            last_error = ""
            last_parsed = ""
            last_python = ""

            for attempt in range(1, max(int(args.max_attempts), 1) + 1):
                prompt = prompt_base
                if last_error:
                    prompt = (
                        prompt
                        + "\n\nPrevious attempt failed.\n"
                        + scipipe.SCICODE_RETRY_GUIDELINES
                        + "\n"
                        + f"Failure hint:\n{last_error}\n"
                    )
                (node_dir / f"prompt_attempt_{attempt}.txt").write_text(prompt, encoding="utf-8")

                _set_mole_subtask(generator, scipipe._step_prompt_text(step, bool(args.with_background)))
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
        "aggregate_enabled": bool(args.aggregate_enabled),
        "aggregate_mode": str(args.aggregate_mode),
        "samples": sample_rows,
        "aggregate": _summarize_samples(sample_rows),
    }
    _write_json(run_root / "summary.json", summary)
    print(f"SciCode SRDD-style taskgraph pipeline complete. Output: {run_root}")


if __name__ == "__main__":
    main()
