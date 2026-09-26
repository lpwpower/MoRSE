"""Batch-generate SciCode dev task graphs with parsed subproblem nodes + LLM dependencies.

Design goals:
1) Node content comes directly from benchmark sub_steps (avoid LLM rewriting descriptions).
2) LLM is only used to infer dependency links among existing subproblem IDs.
3) Resume/retry behavior mirrors SRDD task-graph batch scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
SCICODE_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from morse.taskgraph.generator import DEFAULT_MODEL_NAME  # noqa: E402
from morse.taskgraph.graph import SubtaskGraph, SubtaskNode  # noqa: E402


DEFAULT_DATASET_PATH = SCICODE_ROOT / "data" / "problems_dev.jsonl"
DEFAULT_OUTPUT_DIR = SCICODE_ROOT / "data" / "taskgraphs"


class DependencyGenerationError(RuntimeError):
    """Raised when the dependency planner cannot produce valid dependency JSON."""


def _sanitize(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "item"


def _resolve_dtype(alias: Optional[str]) -> Optional[object]:
    if not alias:
        return None
    alias = alias.strip().lower()
    import torch

    mapping = {
        "float32": ("float32", "float"),
        "fp32": ("float32", "float"),
        "float16": ("float16", "half"),
        "fp16": ("float16", "half"),
        "bfloat16": ("bfloat16",),
        "bf16": ("bfloat16",),
    }
    if alias not in mapping:
        raise ValueError(f"Unsupported torch dtype alias: {alias}")
    for attr in mapping[alias]:
        dtype = getattr(torch, attr, None)
        if dtype is not None:
            return dtype
    return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _graph_ok(sample_dir: Path) -> bool:
    graph_path = sample_dir / "task_graph.json"
    status_path = sample_dir / "graph_generation.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
        if status.get("status") == "ok":
            return True
        if status.get("status") == "failed":
            return False
    if not graph_path.exists():
        return False
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("nodes"))


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    stack: List[int] = []
    start_idx: int | None = None
    for idx, char in enumerate(text):
        if char == "{":
            if not stack:
                start_idx = idx
            stack.append(idx)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
    raise DependencyGenerationError("LLM response did not contain a valid JSON object.")


def _read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _derive_step_title(step_description_prompt: str, fallback: str) -> str:
    text = (step_description_prompt or "").strip()
    if not text:
        return fallback
    first_line = text.splitlines()[0].strip()
    first_line = re.sub(r"\s+", " ", first_line)
    return first_line[:120] if first_line else fallback


def _build_nodes(problem: dict, *, include_background: bool) -> Tuple[List[SubtaskNode], List[str]]:
    sub_steps = list(problem.get("sub_steps") or [])
    nodes: List[SubtaskNode] = []
    order: List[str] = []
    for idx, step in enumerate(sub_steps, start=1):
        step_id = str(step.get("step_number") or f"{problem.get('problem_id', 'p')}.{idx}")
        step_desc = str(step.get("step_description_prompt") or "").strip()
        func_header = str(step.get("function_header") or "").strip()
        return_line = str(step.get("return_line") or "").strip()
        background = str(step.get("step_background") or "").strip()
        title = _derive_step_title(step_desc, fallback=f"Step {step_id}")

        desc_parts = [
            f"step_number: {step_id}",
            "step_description_prompt:",
            step_desc,
            "",
            "function_header:",
            func_header,
        ]
        if return_line:
            desc_parts.extend(["", "return_line:", return_line])
        if include_background and background:
            desc_parts.extend(["", "step_background:", background])

        nodes.append(
            SubtaskNode(
                node_id=step_id,
                title=title,
                description="\n".join(desc_parts).strip(),
                depends_on=[],
            )
        )
        order.append(step_id)
    if not nodes:
        raise ValueError(f"Problem {problem.get('problem_id')} has no sub_steps.")
    return nodes, order


def _build_dependency_prompt(problem: dict, step_order: List[str]) -> str:
    task_name = str(problem.get("problem_name") or f"problem_{problem.get('problem_id', 'unknown')}").strip()
    task_desc = str(problem.get("problem_description_main") or "").strip()
    sub_steps = list(problem.get("sub_steps") or [])

    step_blocks: List[str] = []
    for idx, step_id in enumerate(step_order):
        step = sub_steps[idx] if idx < len(sub_steps) else {}
        step_desc = str(step.get("step_description_prompt") or "").strip()
        func_header = str(step.get("function_header") or "").strip()
        step_blocks.append(
            "\n".join(
                [
                    f"Step ID: {step_id}",
                    "Step Description:",
                    step_desc,
                    "Function Header:",
                    func_header,
                ]
            )
        )
    step_section = "\n\n-----\n\n".join(step_blocks)

    return textwrap.dedent(
        f"""
        You are given an existing decomposition of a coding task into ordered subproblems (steps).
        Your only job is to infer dependency links among these existing step IDs.

        Important constraints:
        - Do NOT rewrite, rename, or merge steps.
        - Use only listed step IDs.
        - A step can depend only on earlier steps in the listed order.
        - Return direct prerequisites only (do not add redundant transitive parents).
        - Prefer a sparse, valid DAG.

        Return JSON only (no markdown, no explanation), with exactly this schema:
        {{
          "dependencies": [
            {{"step": "<step_id>", "depends_on": ["<step_id>", "..."]}}
          ]
        }}

        Task name: {task_name}
        Task description:
        {task_desc}

        Ordered steps:
        {step_section}

        JSON:
        """
    ).strip()


def _parse_dependency_response(response_text: str, step_order: List[str]) -> Dict[str, List[str]]:
    parsed = _extract_first_json_object(response_text)
    raw_deps = parsed.get("dependencies")
    if not isinstance(raw_deps, list):
        raise DependencyGenerationError("JSON must contain a list field 'dependencies'.")

    by_step: Dict[str, List[str]] = {}
    for row in raw_deps:
        if not isinstance(row, dict):
            continue
        step = str(row.get("step") or row.get("id") or row.get("node_id") or "").strip()
        if not step:
            continue
        deps_raw = row.get("depends_on") or row.get("dependencies") or []
        if isinstance(deps_raw, str):
            deps = [part.strip() for part in deps_raw.split(",") if part.strip()]
        else:
            deps = [str(item).strip() for item in deps_raw if str(item).strip()]
        by_step[step] = deps

    # Ensure all steps are present in the mapping even if omitted by the LLM.
    for step in step_order:
        by_step.setdefault(step, [])
    return by_step


def _sanitize_dependencies(predicted: Dict[str, List[str]], step_order: List[str]) -> Dict[str, List[str]]:
    order_index = {step_id: idx for idx, step_id in enumerate(step_order)}
    cleaned: Dict[str, List[str]] = {}
    for step_id in step_order:
        idx = order_index[step_id]
        allowed_parents = set(step_order[:idx])
        deps: List[str] = []
        for parent in predicted.get(step_id, []):
            if parent == step_id:
                continue
            if parent not in allowed_parents:
                continue
            if parent not in deps:
                deps.append(parent)
        if idx > 0 and not deps:
            deps = [step_order[idx - 1]]
        cleaned[step_id] = deps
    return cleaned


def _chain_dependencies(step_order: List[str]) -> Dict[str, List[str]]:
    dep: Dict[str, List[str]] = {}
    for idx, step_id in enumerate(step_order):
        dep[step_id] = [step_order[idx - 1]] if idx > 0 else []
    return dep


class HFDependencyPlanner:
    """Infer step dependencies using a local HF causal/text2text model."""

    def __init__(
        self,
        *,
        model_name: str,
        pipeline_task: str,
        max_new_tokens: int,
        temperature: float,
        device: int,
        device_map: Optional[str],
        torch_dtype: Optional[str],
        hf_use_chat_template: str,
        hf_chat_enable_thinking: str,
        hf_system_prompt: str,
        trust_remote_code: bool,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise DependencyGenerationError(
                "transformers is required. Please install transformers in the active environment."
            ) from exc

        dtype_hint = _resolve_dtype(torch_dtype)
        self.model_name = model_name
        self.pipeline_task = pipeline_task
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.last_prompt: str | None = None
        self.last_model_prompt: str | None = None
        self.last_response_text: str | None = None

        model_kwargs: Dict[str, Any] = {}
        if dtype_hint is not None:
            model_kwargs["torch_dtype"] = dtype_hint

        use_device_map = bool(device_map and device_map.strip().lower() != "none")

        tok_kwargs: Dict[str, Any] = {"trust_remote_code": bool(trust_remote_code)}
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self._tokenizer = tokenizer
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_load_kwargs: Dict[str, Any] = dict(model_kwargs)
        model_load_kwargs["trust_remote_code"] = bool(trust_remote_code)
        if use_device_map:
            model_load_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

        if not use_device_map:
            import torch

            if torch.cuda.is_available():
                model = model.to(f"cuda:{int(device)}")

        self._pipeline = pipeline(
            task=pipeline_task,
            model=model,
            tokenizer=tokenizer,
        )
        chat_mode = str(hf_use_chat_template or "").strip().lower()
        if not chat_mode:
            chat_mode = os.environ.get("HF_USE_CHAT_TEMPLATE", "false").strip().lower()
        self._chat_template_mode = chat_mode
        thinking_mode = str(hf_chat_enable_thinking or "").strip().lower()
        if not thinking_mode:
            thinking_mode = os.environ.get("HF_CHAT_ENABLE_THINKING", "").strip().lower()
        if thinking_mode in {"1", "true", "yes", "on"}:
            self._chat_enable_thinking: Optional[bool] = True
        elif thinking_mode in {"0", "false", "no", "off"}:
            self._chat_enable_thinking = False
        else:
            self._chat_enable_thinking = None
        self._chat_system_prompt = str(hf_system_prompt or "").strip()
        if not self._chat_system_prompt:
            self._chat_system_prompt = os.environ.get("HF_SYSTEM_PROMPT", "").strip()

    def _should_use_chat_template(self) -> bool:
        if self.pipeline_task != "text-generation":
            return False
        if self._chat_template_mode in {"0", "false", "no", "off"}:
            return False
        tok = self._tokenizer
        if tok is None or not hasattr(tok, "apply_chat_template"):
            return False
        if self._chat_template_mode == "auto" and not getattr(tok, "chat_template", None):
            return False
        return True

    def _format_prompt_for_model(self, prompt: str) -> str:
        if not self._should_use_chat_template():
            return prompt
        tok = self._tokenizer
        messages: List[Dict[str, Any]] = []
        if self._chat_system_prompt:
            messages.append({"role": "system", "content": self._chat_system_prompt})
        messages.append({"role": "user", "content": prompt})
        chat_kwargs: Dict[str, Any] = {}
        if self._chat_enable_thinking is not None:
            chat_kwargs["enable_thinking"] = self._chat_enable_thinking
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)
        except Exception:
            # Some chat templates expect multimodal-style content blocks.
            try:
                messages_mm: List[Dict[str, Any]] = []
                if self._chat_system_prompt:
                    messages_mm.append({"role": "system", "content": [{"type": "text", "text": self._chat_system_prompt}]})
                messages_mm.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
                return tok.apply_chat_template(messages_mm, tokenize=False, add_generation_prompt=True, **chat_kwargs)
            except Exception:
                return prompt

    def infer_dependencies(self, problem: dict, step_order: List[str]) -> Dict[str, List[str]]:
        prompt = _build_dependency_prompt(problem, step_order)
        self.last_prompt = prompt
        model_prompt = self._format_prompt_for_model(prompt)
        self.last_model_prompt = model_prompt
        do_sample = self.temperature > 0
        call_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": do_sample,
        }
        if self.pipeline_task == "text-generation":
            call_kwargs["return_full_text"] = False
        if do_sample:
            call_kwargs["temperature"] = self.temperature
        else:
            call_kwargs["temperature"] = 1.0
            call_kwargs["top_p"] = 1.0
            call_kwargs["top_k"] = 50

        response_text = self._pipeline(model_prompt, **call_kwargs)[0]["generated_text"]
        if self.pipeline_task == "text-generation" and model_prompt in response_text:
            response_text = response_text.split(model_prompt, 1)[1].lstrip()
        self.last_response_text = response_text
        return _parse_dependency_response(response_text, step_order)


def _sample_dir(output_dir: Path, idx: int, problem: dict) -> Path:
    problem_id = str(problem.get("problem_id") or idx)
    problem_name = _sanitize(str(problem.get("problem_name") or f"problem_{problem_id}"))
    return output_dir / f"{idx:03d}_{problem_id}_{problem_name}"


def _record_metadata(sample_dir: Path, idx: int, problem: dict) -> None:
    payload = {
        "index": idx,
        "problem_id": str(problem.get("problem_id") or ""),
        "problem_name": str(problem.get("problem_name") or ""),
        "problem_description_main": str(problem.get("problem_description_main") or ""),
        "num_sub_steps": len(problem.get("sub_steps") or []),
    }
    _write_json(sample_dir / "sample.json", payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SciCode dev task graphs from parsed subproblems, with LLM dependency inference."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--pipeline-task",
        type=str,
        choices=["text-generation", "text2text-generation"],
        default="text-generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HF trust_remote_code when loading model/tokenizer/pipeline.",
    )
    parser.add_argument(
        "--hf-use-chat-template",
        type=str,
        default="false",
        help=(
            "Prompt formatting mode for text-generation models: "
            "false (default, raw prompt), true (force chat template), or auto."
        ),
    )
    parser.add_argument(
        "--hf-system-prompt",
        type=str,
        default="",
        help="Optional system prompt used only when chat-template formatting is enabled.",
    )
    parser.add_argument(
        "--hf-chat-enable-thinking",
        type=str,
        default="",
        help=(
            "Optional thinking switch passed to tokenizer.apply_chat_template when chat-template mode is enabled: "
            "true/false. Empty means tokenizer default behavior."
        ),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument(
        "--index-offset",
        type=int,
        default=0,
        help="Offset added to 1-based sample index when naming output directories.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip samples where graph_generation.json already reports status=ok.",
    )
    parser.add_argument(
        "--include-background-in-node-desc",
        action="store_true",
        help="Append step_background to each node description.",
    )
    parser.add_argument(
        "--fallback-chain-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When LLM dependency inference fails, write a linear chain graph instead of marking failure.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problems = _read_jsonl(args.dataset)
    if not problems:
        raise RuntimeError(f"No records found in dataset: {args.dataset}")
    if args.max_samples and args.max_samples > 0:
        problems = problems[: args.max_samples]

    planner = HFDependencyPlanner(
        model_name=args.model_name,
        pipeline_task=args.pipeline_task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        hf_use_chat_template=args.hf_use_chat_template,
        hf_chat_enable_thinking=args.hf_chat_enable_thinking,
        hf_system_prompt=args.hf_system_prompt,
        trust_remote_code=bool(args.trust_remote_code),
    )

    failures: List[str] = []
    index_offset = max(int(args.index_offset), 0)
    for idx_local, problem in enumerate(problems, start=1):
        idx = idx_local + index_offset
        sample_dir = _sample_dir(args.output_dir, idx, problem)
        if args.skip_existing and _graph_ok(sample_dir):
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)
        _record_metadata(sample_dir, idx, problem)

        nodes, step_order = _build_nodes(problem, include_background=bool(args.include_background_in_node_desc))
        graph_path = sample_dir / "task_graph.json"
        status_path = sample_dir / "graph_generation.json"

        problem_label = f"{problem.get('problem_id', idx)}:{problem.get('problem_name', 'unknown')}"
        last_error: Exception | None = None
        edge_source = "llm"

        for attempt in range(1, max(int(args.max_retries), 1) + 1):
            try:
                predicted = planner.infer_dependencies(problem, step_order)
                cleaned = _sanitize_dependencies(predicted, step_order)
                for node in nodes:
                    node.depends_on = cleaned.get(node.node_id, [])

                graph = SubtaskGraph(
                    task_name=str(problem.get("problem_name") or f"problem_{problem.get('problem_id', idx)}").strip(),
                    task_description=str(problem.get("problem_description_main") or "").strip(),
                    nodes=nodes,
                )
                graph.save_json(graph_path)
                _write_json(
                    status_path,
                    {
                        "status": "ok",
                        "attempts": attempt,
                        "edge_source": edge_source,
                        "model_name": args.model_name,
                        "pipeline_task": args.pipeline_task,
                        "device": args.device,
                        "device_map": args.device_map,
                        "torch_dtype": args.torch_dtype,
                        "hf_use_chat_template": str(args.hf_use_chat_template),
                        "hf_chat_enable_thinking": str(args.hf_chat_enable_thinking),
                        "trust_remote_code": bool(args.trust_remote_code),
                        "node_count": len(graph.nodes),
                        "edge_count": len(list(graph.edges())),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                break
            except Exception as exc:
                last_error = exc
                prompt = planner.last_prompt or ""
                response = planner.last_response_text or ""
                (sample_dir / f"dependency_error_attempt_{attempt}.txt").write_text(str(exc), encoding="utf-8")
                if prompt:
                    (sample_dir / f"dependency_prompt_attempt_{attempt}.txt").write_text(prompt, encoding="utf-8")
                if response:
                    (sample_dir / f"dependency_response_attempt_{attempt}.txt").write_text(response, encoding="utf-8")
                logging.warning("Dependency inference retry %d/%d failed for %s: %s", attempt, args.max_retries, problem_label, exc)
        else:
            if args.fallback_chain_on_failure:
                edge_source = "chain_fallback"
                chain_dep = _chain_dependencies(step_order)
                for node in nodes:
                    node.depends_on = chain_dep.get(node.node_id, [])
                graph = SubtaskGraph(
                    task_name=str(problem.get("problem_name") or f"problem_{problem.get('problem_id', idx)}").strip(),
                    task_description=str(problem.get("problem_description_main") or "").strip(),
                    nodes=nodes,
                )
                graph.save_json(graph_path)
                _write_json(
                    status_path,
                    {
                        "status": "ok",
                        "attempts": max(int(args.max_retries), 1),
                        "edge_source": edge_source,
                        "model_name": args.model_name,
                        "pipeline_task": args.pipeline_task,
                        "device": args.device,
                        "device_map": args.device_map,
                        "torch_dtype": args.torch_dtype,
                        "hf_use_chat_template": str(args.hf_use_chat_template),
                        "hf_chat_enable_thinking": str(args.hf_chat_enable_thinking),
                        "trust_remote_code": bool(args.trust_remote_code),
                        "node_count": len(graph.nodes),
                        "edge_count": len(list(graph.edges())),
                        "error": str(last_error) if last_error else "unknown",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                logging.warning("Fell back to linear chain dependencies for %s.", problem_label)
            else:
                failures.append(problem_label)
                _write_json(
                    status_path,
                    {
                        "status": "failed",
                        "attempts": max(int(args.max_retries), 1),
                        "edge_source": "none",
                        "model_name": args.model_name,
                        "pipeline_task": args.pipeline_task,
                        "device": args.device,
                        "device_map": args.device_map,
                        "torch_dtype": args.torch_dtype,
                        "hf_use_chat_template": str(args.hf_use_chat_template),
                        "hf_chat_enable_thinking": str(args.hf_chat_enable_thinking),
                        "trust_remote_code": bool(args.trust_remote_code),
                        "error": str(last_error) if last_error else "unknown",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )

    if failures:
        raise RuntimeError(f"Dependency generation failed for {len(failures)} samples: {failures[:5]}")

    logging.info("Done. Output dir: %s", args.output_dir)


if __name__ == "__main__":
    main()
