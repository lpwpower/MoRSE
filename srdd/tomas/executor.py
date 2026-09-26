"""A camel-free executor that mirrors the essential MacNet DAG pipeline."""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from srdd.tomas.codes_relaxed import Codes
from morse.llm.hf_llm import HFTextGenerator
from srdd.tomas.review_test import SmokeTestResult, smoke_test_repo


def _stdlib_modules() -> set[str]:
	if hasattr(sys, "stdlib_module_names"):
		return set(sys.stdlib_module_names)  # py>=3.10
	# Fallback: best-effort list.
	return {
		"argparse",
		"asyncio",
		"base64",
		"collections",
		"concurrent",
		"csv",
		"dataclasses",
		"datetime",
		"functools",
		"glob",
		"hashlib",
		"heapq",
		"html",
		"http",
		"io",
		"itertools",
		"json",
		"logging",
		"math",
		"os",
		"pathlib",
		"queue",
		"random",
		"re",
		"shlex",
		"signal",
		"sqlite3",
		"statistics",
		"string",
		"subprocess",
		"sys",
		"tempfile",
		"textwrap",
		"threading",
		"time",
		"typing",
		"unittest",
		"urllib",
		"uuid",
		"xml",
		"zipfile",
	}


_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
_PASS_RE = re.compile(r"^\s*pass(?:\s|#|$)", re.MULTILINE)


def _contains_format_placeholders(text: str) -> bool:
	lowered = text.lower()
	return any(
		token in lowered
		for token in (
			"filename.py",
			"<full code>",
			"docstring",
			"code\n```",
		)
	)


def validate_codes(
	codes: Codes,
	*,
	require_main: bool = True,
	stdlib_only: bool = True,
	allow_pass_todo: bool = False,
	allow_format_placeholders: bool = False,
) -> None:
	if not codes.codebooks:
		raise ValueError("Model output did not contain any parseable code blocks/files.")

	for filename in codes.codebooks:
		if not filename.endswith(".py"):
			raise ValueError(f"Non-Python file generated: {filename}")
		if (not allow_format_placeholders) and filename.lower() == "filename.py":
			raise ValueError("Placeholder filename 'FILENAME.py' detected; use real filenames.")

	if require_main and "main.py" not in codes.codebooks:
		raise ValueError("Generated repository must include main.py")

	for filename, content in codes.codebooks.items():
		lowered = content.lower()
		if (not allow_format_placeholders) and _contains_format_placeholders(content):
			raise ValueError(f"Format placeholder text detected in {filename}.")
		if not allow_pass_todo and (_PASS_RE.search(content) or "todo" in lowered):
			raise ValueError(f"Placeholders detected in {filename} (pass/todo).")
		if (not allow_format_placeholders) and re.search(r"^\s*(?:from|import)\s+FILENAME\b", content, re.MULTILINE):
			raise ValueError(f"Placeholder import 'FILENAME' detected in {filename}.")

	if stdlib_only:
		stdlib = _stdlib_modules()
		local_modules = {Path(f).stem.lower() for f in codes.codebooks}
		for filename, content in codes.codebooks.items():
			for match in _IMPORT_RE.finditer(content):
				module = match.group(1).split(".")[0]
				module_key = module.strip()
				if module_key in {"__future__", ""}:
					continue
				# Allow local imports that refer to other files in the repo (case-insensitive).
				if module_key.lower() in local_modules:
					continue
				if module_key not in stdlib:
					raise ValueError(f"Non-stdlib import '{module_key}' detected in {filename}.")


def _summarize_validation_hints(codes: Codes) -> str:
	"""Best-effort hints to help the model repair its output."""
	lines: list[str] = []
	non_py = sorted([f for f in codes.codebooks if not f.endswith(".py")])
	if non_py:
		lines.append(f"- Remove/rename non-.py files: {', '.join(non_py[:10])}")
	if "main.py" not in codes.codebooks:
		lines.append("- Add a runnable main.py (must not rely on non-stdlib imports).")
	pass_files = []
	todo_files = []
	placeholder_files = []
	for filename, content in codes.codebooks.items():
		if _PASS_RE.search(content):
			pass_files.append(filename)
		if "todo" in content.lower():
			todo_files.append(filename)
		if _contains_format_placeholders(content):
			placeholder_files.append(filename)
	if pass_files:
		lines.append(f"- Remove ALL `pass` statements by implementing or deleting stubs: {', '.join(sorted(pass_files)[:10])}")
	if todo_files:
		lines.append(f"- Remove ALL TODO markers by implementing the logic: {', '.join(sorted(todo_files)[:10])}")
	if placeholder_files:
		lines.append(f"- Remove placeholder template text: {', '.join(sorted(placeholder_files)[:10])}")

	stdlib = _stdlib_modules()
	local = {Path(f).stem for f in codes.codebooks}
	bad_imports: dict[str, set[str]] = {}
	for filename, content in codes.codebooks.items():
		for match in _IMPORT_RE.finditer(content):
			module = match.group(1).split(".")[0]
			if module in {"__future__", ""}:
				continue
			if module in local:
				continue
			if module not in stdlib:
				bad_imports.setdefault(filename, set()).add(module)
	if bad_imports:
		preview = []
		for filename in sorted(bad_imports):
			preview.append(f"{filename}: {', '.join(sorted(bad_imports[filename]))}")
		lines.append("- Remove non-stdlib imports:\n  - " + "\n  - ".join(preview[:10]))

	return "\n".join(lines) if lines else "- Regenerate output strictly following constraints."


@dataclass
class ExecutionResult:
	project_name: str
	run_dir: Path
	code_dir: Path
	sample_metrics_path: Path


class ForcedRetryError(RuntimeError):
	"""Signals a forced retry without treating the attempt as a real failure."""


@dataclass
class ReviewToolResult:
	hard_issues: list[str]
	validation_error: str | None
	smoke: SmokeTestResult | None
	report: str


def _write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_ts() -> str:
	return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _time_ms() -> int:
	return int(time.time() * 1000)


def _truncate_report(text: str, limit: int = 2000) -> str:
	if not text:
		return ""
	if len(text) <= limit:
		return text
	return text[: max(0, limit - 20)] + "...[truncated]..."


def _run_smoke_test(
	*,
	codes: Codes,
	layer_dir: Path,
	node_id: int,
	stage: str,
	attempt: int,
	timeout_s: int = 3,
) -> SmokeTestResult:
	repo_dir = layer_dir / f"{stage}_repo_attempt_{attempt}"
	if repo_dir.exists():
		shutil.rmtree(repo_dir)
	codes.write_to_directory(repo_dir)

	result = smoke_test_repo(repo_dir, timeout_s=timeout_s)
	_write_text(
		layer_dir / f"{stage}_smoke_attempt_{attempt}.txt",
		f"passed: {result.passed}\n"
		f"elapsed_ms: {result.elapsed_ms}\n"
		"details:\n"
		f"{result.details}\n",
	)
	return result


def _format_review_tool_report(result: ReviewToolResult) -> str:
	lines: list[str] = []
	if result.hard_issues:
		lines.append("Hard issues:")
		for issue in result.hard_issues:
			lines.append(f"- {_truncate_report(issue)}")
	else:
		lines.append("Hard issues: none")
	if result.smoke is None:
		lines.append("Smoke test: skipped")
	else:
		lines.append(
			f"Smoke test: passed={result.smoke.passed} elapsed_ms={result.smoke.elapsed_ms}"
		)
		if result.smoke.details:
			lines.append(f"Smoke details: {_truncate_report(result.smoke.details)}")
	return "\n".join(lines) + "\n"


def _run_review_tool(
	*,
	codes: Codes | None,
	tool_dir: Path,
	stage: str,
	attempt: int,
	allow_pass_todo: bool,
	allow_format_placeholders: bool,
	timeout_s: int = 3,
) -> ReviewToolResult:
	tool_dir.mkdir(parents=True, exist_ok=True)
	hard_issues: list[str] = []
	validation_error: str | None = None
	smoke: SmokeTestResult | None = None

	if codes is None or not codes.codebooks:
		validation_error = "Model output did not contain any parseable code blocks/files."
		hard_issues.append(validation_error)
	else:
		try:
			validate_codes(
				codes,
				allow_pass_todo=allow_pass_todo,
				allow_format_placeholders=allow_format_placeholders,
			)
		except Exception as exc:
			validation_error = str(exc)
			hard_issues.append(validation_error)

	if not hard_issues and codes is not None and codes.codebooks:
		repo_dir = tool_dir / f"{stage}_tool_repo_attempt_{attempt}"
		if repo_dir.exists():
			shutil.rmtree(repo_dir)
		codes.write_to_directory(repo_dir)
		smoke = smoke_test_repo(repo_dir, timeout_s=timeout_s)
		if not smoke.passed:
			hard_issues.append(f"Smoke test failed: {smoke.details}".strip())

	base = ReviewToolResult(
		hard_issues=hard_issues,
		validation_error=validation_error,
		smoke=smoke,
		report="",
	)
	result = ReviewToolResult(
		hard_issues=hard_issues,
		validation_error=validation_error,
		smoke=smoke,
		report=_format_review_tool_report(base),
	)
	_write_text(tool_dir / f"{stage}_review_tool_attempt_{attempt}.txt", result.report)
	return result


def build_executor_prompt(
	*,
	task_description: str,
	subtask_title: str,
	subtask_description: str,
	repo_snapshot: str,
	preamble: str | None = None,
) -> str:
	parts = [
		"You are an expert in Python programming. Your task is to implement complete runnable Python-only software project with minimal dependencies.\n"
		"Overall task description:\n"
		f"{task_description}\n\n"
	]
	if preamble:
		parts.append(preamble.strip() + "\n\n")
	parts.append(
		"Now complete THIS subtask and update the repository accordingly:\n"
		f"Subtask title: {subtask_title}\n"
		f"Subtask description: {subtask_description}\n\n"
		"Constraints:\n"
		"- Output ONLY ONE file: main.py. Do NOT emit any explanation text.\n"
		"- Output NOTHING else: no preface, no bullet points, no notes, no prose.\n"
		"- The FIRST line must be exactly: main.py\n"
		"- The SECOND line must be exactly: ```python\n"
		"- The LAST line must be the closing ``` fence.\n"
		"- Do NOT output any other filenames or file blocks.\n"
		"- Python 3 standard library only; no external deps or GUI frameworks.\n"
		"- Use ASCII-only characters; no smart quotes or non-ASCII punctuation.\n"
		"- Use 4-space indentation only; do NOT use tabs.\n"
		"- No placeholders: TODO, pass, <full code>, or empty blocks.\n"
		"- main.py must include if __name__ == \"__main__\": entry point.\n"
		"- Must run without extra files; keep it minimal.\n\n"
		"Output format (exactly once):\n"
		"main.py\n"
		"```python\n"
		"<full code>\n"
		"```\n\n"
		"Current repository snapshot (all files):\n"
		f"{repo_snapshot}\n\n"
	)
	return "".join(parts)


def build_aggregate_prompt(
	*,
	task_description: str,
	parent_snapshots: Sequence[str],
	preamble: str | None = None,
) -> str:
	joined = "\n\n".join(
		[f"=== Candidate Solution {idx} ===\n{snap}" for idx, snap in enumerate(parent_snapshots, start=1)]
	)
	parts = [
     	"You are an expert in Python programming. Your task is to implement complete runnable Python-only software project with minimal dependencies.\n"
		"You are merging multiple Python project repositories for the SAME software program.\n"
		"Overall task description:\n"
		f"{task_description}\n\n"
		"You are given multiple candidate repositories produced by different predecessor nodes, and each one may implement DIFFERENT subtasks.\n"
		"Synthesize ONE unified repository that includes ALL completed subtasks from ALL parents, resolves conflicts, and keeps the project runnable end-to-end.\n"
		"If two parents implement overlapping parts differently, merge them or choose the option that best satisfies the overall task while preserving as much functionality as possible.\n"
		"Constraints:\n"
		"- Output ONLY ONE file: main.py. Do NOT emit any explanation text.\n"
		"- Output NOTHING else: no preface, no bullet points, no notes, no prose.\n"
		"- The FIRST line must be exactly: main.py\n"
		"- The SECOND line must be exactly: ```python\n"
		"- The LAST line must be the closing ``` fence.\n"
		"- Do NOT output any other filenames or file blocks.\n"
		"- Python 3 standard library only; no external deps or GUI frameworks.\n"
		"- Use ASCII-only characters; no smart quotes or non-ASCII punctuation.\n"
		"- Use 4-space indentation only; do NOT use tabs.\n"
		"- No placeholders: TODO, pass, <full code>, or empty blocks.\n"
		"- main.py must include if __name__ == \"__main__\": entry point.\n"
		"- Must run without extra files; keep it minimal.\n\n"
		"Output format (exactly once):\n"
		"main.py\n"
		"```python\n"
		"<full code>\n"
		"```\n\n"
	]
	if preamble:
		parts.append(preamble.strip() + "\n\n")
	parts.append("Candidate repositories:\n" f"{joined}\n\n")
	return "".join(parts)


def execute_task_graph(
	*,
	spec,
	generator: HFTextGenerator,
	output_root: Path,
	warehouse_root: Path,
	sample_name: str,
	require_aggregation: bool = True,
	max_attempts: int = 3,
	stop_on_failure: bool = False,
	run_dir: Path | None = None,
	code_dir: Path | None = None,
	use_verifier: bool = False,
) -> ExecutionResult:
	safe_sample = re.sub(r"[^a-zA-Z0-9]+", "_", sample_name).strip("_") or "sample"
	run_slug = f"{spec.slug()}_{safe_sample}_{time.strftime('%Y%m%d_%H%M%S')}"
	resolved_run_dir = Path(run_dir) if run_dir is not None else (output_root / run_slug)
	resolved_run_dir.mkdir(parents=True, exist_ok=True)

	_write_text(resolved_run_dir / "task_description.txt", spec.task_description)
	(resolved_run_dir / "task_graph.json").write_text(json.dumps(spec.__dict__, default=str, indent=2), encoding="utf-8")

	solutions: Dict[int, Codes] = {}
	predecessors: Dict[int, List[int]] = {nid: [] for nid in spec.node_metadata}
	for edge in spec.edge_strings:
		src, dst = edge.split("->", 1)
		predecessors[int(dst)].append(int(src))

	node_ids = sorted(spec.node_metadata)
	relaxed_until = len(node_ids) // 3  # first 1/3 nodes allow placeholder stubs

	sample_start_ms = _time_ms()
	sample_start_ts = _now_ts()
	sample_calls: List[Dict[str, Any]] = []
	sample_nodes: List[Dict[str, Any]] = []
	if use_verifier:
		raise NotImplementedError(
			"use_verifier (MacNet-style reviewer) is not included in this release; "
			"the paper pipeline uses the rule-based smoke test only."
		)
	reviewer = None

	for node_index, node_id in enumerate(node_ids):
		node_meta = spec.node_metadata[node_id]
		parent_ids = sorted(predecessors.get(node_id, []))
		layer_dir = resolved_run_dir / f"node_{node_id:02d}"
		layer_dir.mkdir(parents=True, exist_ok=True)
		allow_pass_todo = node_index < relaxed_until
		allow_format_placeholders = node_index < relaxed_until

		def _require_solution(parent_id: int) -> Codes:
			try:
				return solutions[parent_id]
			except KeyError as exc:
				raise RuntimeError(
					f"Missing parent solution for node {node_id}: parent_id={parent_id}, parents={parent_ids}, solved={sorted(solutions)}"
				) from exc

		# Base snapshot; keep base_codes for failure fallback.
		base_codes: Codes | None = None
		aggregate_ran = len(parent_ids) > 1
		last_agg_response: str | None = None
		if not parent_ids:
			base_snapshot = ""
		elif len(parent_ids) == 1:
			base_codes = _require_solution(parent_ids[0])
			base_snapshot = base_codes.snapshot()
		else:
			parent_snaps = [_require_solution(parent).snapshot() for parent in parent_ids]
			base_codes = _require_solution(parent_ids[0])
			base_snapshot = base_codes.snapshot()

			agg_last_exc: Exception | None = None
			agg_last_hint: str | None = None
			last_agg_codes: Codes | None = None
			agg_last_reviewer: str | None = None
			agg_done = False
			if aggregate_ran:
				for attempt in range(1, max_attempts + 1):
					try:
						candidate_snaps = list(parent_snaps)
						if last_agg_codes is not None and last_agg_codes.codebooks:
							candidate_snaps.append(last_agg_codes.snapshot())
						preamble = ""
						if agg_last_reviewer:
							preamble = (
								preamble
								+ "Reviewer feedback to incorporate:\n"
								+ agg_last_reviewer
								+ "\n\n"
							)
						if agg_last_exc is not None and not isinstance(agg_last_exc, ForcedRetryError):
							preamble = (
								preamble
								+ "Previous attempt failed validation/review with error:\n"
								+ str(agg_last_exc)
								+ "\n\nRepair checklist:\n"
								+ (agg_last_hint or "- Regenerate output strictly following constraints.")
								+ "\n\nNow regenerate the FULL repository meeting ALL constraints.\n"
							)
						prompt = build_aggregate_prompt(
							task_description=spec.task_description,
							parent_snapshots=candidate_snaps,
							preamble=preamble or None,
						)
						if attempt == 1:
							_write_text(layer_dir / "aggregate_prompt.txt", prompt)
						_write_text(layer_dir / f"aggregate_prompt_attempt_{attempt}.txt", prompt)
						prompt_tokens = generator.count_tokens(prompt)
						call_start_ms = _time_ms()
						agg_response = generator.generate(prompt)
						last_agg_response = agg_response
						call_end_ms = _time_ms()
						completion_tokens = generator.count_tokens(agg_response)
						_write_text(layer_dir / f"aggregate_response_attempt_{attempt}.txt", agg_response)
						agg_codes = Codes(agg_response)
						last_agg_codes = agg_codes

						if reviewer is None:
							validation_error = None
							try:
								validate_codes(
									agg_codes,
									allow_pass_todo=allow_pass_todo,
									allow_format_placeholders=allow_format_placeholders,
								)
							except Exception as exc:
								validation_error = exc
							if validation_error is not None:
								raise validation_error
							smoke = _run_smoke_test(
								codes=agg_codes,
								layer_dir=layer_dir,
								node_id=node_id,
								stage="aggregate",
								attempt=attempt,
							)
							sample_calls.append(
								{
									"ts": _now_ts(),
									"node_id": node_id,
									"kind": "smoke",
									"stage": "aggregate",
									"attempt": attempt,
									"elapsed_ms": smoke.elapsed_ms,
									"passed": smoke.passed,
									"details": smoke.details,
								}
							)
							if not smoke.passed:
								raise RuntimeError(
									f"Smoke test failed for node {node_id} (aggregate) attempt {attempt}: {smoke.details}"
								)
						else:
							agg_reviewer_dir = layer_dir / "aggregate_reviewer"
							agg_reviewer_dir.mkdir(parents=True, exist_ok=True)
							tool_result = _run_review_tool(
								codes=agg_codes,
								tool_dir=agg_reviewer_dir,
								stage="aggregate",
								attempt=attempt,
								allow_pass_todo=allow_pass_todo,
								allow_format_placeholders=allow_format_placeholders,
							)
							parent_lines = []
							for pid in parent_ids:
								meta = spec.node_metadata.get(pid)
								if meta is None:
									parent_lines.append(f"- Node {pid}: (missing metadata)")
									continue
								parent_lines.append(f"- Node {pid}: {meta.title} - {meta.description}")
							parent_summary = "Parent subtasks:\n" + "\n".join(parent_lines)
							try:
								if attempt == 1 or tool_result.hard_issues:
									feedback, events = reviewer.review(
										task_description=spec.task_description,
										subtask_title="Aggregate from parents",
										subtask_description=parent_summary,
										candidate=last_agg_codes,
										previous_error=str(agg_last_exc) if agg_last_exc else None,
										tool_report=tool_result.report,
										layer_dir=agg_reviewer_dir,
										attempt=attempt,
									)
									if feedback:
										agg_last_reviewer = feedback
									for e in events:
										sample_calls.append(
											{"ts": _now_ts(), "node_id": node_id, "stage": "aggregate", **e}
										)
								sample_calls.append(
									{
										"ts": _now_ts(),
										"node_id": node_id,
										"kind": "review_tool",
										"stage": "aggregate",
										"attempt": attempt,
										"hard_issue_count": len(tool_result.hard_issues),
										"smoke_passed": (tool_result.smoke.passed if tool_result.smoke else None),
									}
								)
							except Exception as vexc:
								_write_text(
									agg_reviewer_dir / f"reviewer_error_attempt_{attempt}.txt",
									str(vexc),
								)
								raise
							if attempt == 1 and max_attempts > 1:
								raise ForcedRetryError("Forced retry after initial review.")
							if tool_result.hard_issues:
								raise RuntimeError(
									"Review tool found hard issues: "
									+ "; ".join(tool_result.hard_issues)
								)

						base_codes = agg_codes
						base_snapshot = agg_codes.snapshot()
						agg_last_exc = None
						agg_done = True
						sample_calls.append(
							{
								"ts": _now_ts(),
								"node_id": node_id,
								"kind": "aggregate",
								"attempt": attempt,
								"prompt_tokens": prompt_tokens,
								"completion_tokens": completion_tokens,
								"elapsed_ms": call_end_ms - call_start_ms,
							}
						)
						break
					except Exception as exc:
						agg_last_exc = exc
						try:
							sample_calls.append(
								{
									"ts": _now_ts(),
									"node_id": node_id,
									"kind": "aggregate",
									"attempt": attempt,
									"prompt_tokens": generator.count_tokens(prompt),
									"completion_tokens": generator.count_tokens(agg_response)
									if "agg_response" in locals()
									else 0,
									"elapsed_ms": (_time_ms() - call_start_ms)
									if "call_start_ms" in locals()
									else None,
									"error": str(exc),
								}
							)
						except Exception:
							pass
						try:
							agg_last_hint = _summarize_validation_hints(last_agg_codes or Codes(""))
						except Exception:
							agg_last_hint = None
						_write_text(layer_dir / f"aggregate_error_attempt_{attempt}.txt", str(exc))

			if agg_last_exc is not None or not agg_done:
				_write_text(
					layer_dir / "aggregate_failure.txt",
					f"Aggregation failed after {max_attempts} attempts.\nError: {agg_last_exc}\n",
				)
				if stop_on_failure and require_aggregation:
					raise RuntimeError(
						f"Node {node_id} aggregation failed after {max_attempts} attempts: {agg_last_exc}"
					) from agg_last_exc
				if last_agg_codes is not None and last_agg_codes.codebooks:
					base_codes = last_agg_codes
					base_snapshot = last_agg_codes.snapshot()
				else:
					base_codes = _require_solution(parent_ids[0])
					base_snapshot = base_codes.snapshot()

		current_snapshot = base_snapshot
		_write_text(
			layer_dir / "prompt.txt",
			build_executor_prompt(
				task_description=spec.task_description,
				subtask_title=node_meta.title,
				subtask_description=node_meta.description,
				repo_snapshot=current_snapshot,
			),
		)
		last_exc = None
		last_hint = None
		last_reviewer: str | None = None
		codes = None
		last_candidate = None
		last_smoke: SmokeTestResult | None = None
		last_response: str | None = None
		node_done = False
		for attempt in range(1, max_attempts + 1):
			try:
				preamble = ""
				if last_reviewer:
					preamble = (
						preamble
						+ "Reviewer feedback to incorporate:\n"
						+ last_reviewer
						+ "\n\n"
					)
				if last_exc is not None and not isinstance(last_exc, ForcedRetryError):
					preamble = (
						preamble
						+ "Your previous output failed validation/review with error:\n"
						+ str(last_exc)
						+ "\n\nRepair checklist:\n"
						+ (last_hint or "- Regenerate output strictly following constraints.")
						+ "\n\nNow regenerate the FULL repository meeting ALL constraints.\n"
					)
				attempt_prompt = build_executor_prompt(
					task_description=spec.task_description,
					subtask_title=node_meta.title,
					subtask_description=node_meta.description,
					repo_snapshot=current_snapshot,
					preamble=preamble or None,
				)
				_write_text(layer_dir / f"prompt_attempt_{attempt}.txt", attempt_prompt)
				prompt_tokens = generator.count_tokens(attempt_prompt)
				call_start_ms = _time_ms()
				response = generator.generate(attempt_prompt)
				last_response = response
				call_end_ms = _time_ms()
				completion_tokens = generator.count_tokens(response)
				_write_text(layer_dir / f"response_attempt_{attempt}.txt", response)
				candidate = Codes(response)
				last_candidate = candidate
				if not candidate.codebooks:
					raise ValueError("Model output did not contain any parseable code blocks/files.")
				current_snapshot = candidate.snapshot()

				if reviewer is None:
					validation_error = None
					try:
						validate_codes(
							candidate,
							allow_pass_todo=allow_pass_todo,
							allow_format_placeholders=allow_format_placeholders,
						)
					except Exception as exc:
						validation_error = exc
					if validation_error is not None:
						raise validation_error
					smoke = _run_smoke_test(
						codes=candidate,
						layer_dir=layer_dir,
						node_id=node_id,
						stage="execute",
						attempt=attempt,
					)
					last_smoke = smoke
					if not smoke.passed:
						raise RuntimeError(
							f"Smoke test failed for node {node_id} (execute) attempt {attempt}: {smoke.details}"
						)
				else:
					try:
						tool_result = _run_review_tool(
							codes=candidate,
							tool_dir=layer_dir,
							stage="execute",
							attempt=attempt,
							allow_pass_todo=allow_pass_todo,
							allow_format_placeholders=allow_format_placeholders,
						)
						if attempt == 1 or tool_result.hard_issues:
							feedback, events = reviewer.review(
								task_description=spec.task_description,
								subtask_title=node_meta.title,
								subtask_description=node_meta.description,
								candidate=last_candidate,
								previous_error=str(last_exc) if last_exc else None,
								tool_report=tool_result.report,
								layer_dir=layer_dir,
								attempt=attempt,
							)
							if feedback:
								last_reviewer = feedback
							for e in events:
								sample_calls.append({"ts": _now_ts(), "node_id": node_id, **e})
						sample_calls.append(
							{
								"ts": _now_ts(),
								"node_id": node_id,
								"kind": "review_tool",
								"stage": "execute",
								"attempt": attempt,
								"hard_issue_count": len(tool_result.hard_issues),
								"smoke_passed": (tool_result.smoke.passed if tool_result.smoke else None),
							}
						)
					except Exception as vexc:
						_write_text(layer_dir / f"reviewer_error_attempt_{attempt}.txt", str(vexc))
						raise
					if attempt == 1 and max_attempts > 1:
						raise ForcedRetryError("Forced retry after initial review.")
					if tool_result.hard_issues:
						raise RuntimeError(
							"Review tool found hard issues: "
							+ "; ".join(tool_result.hard_issues)
						)

				codes = candidate
				last_exc = None
				node_done = True
				sample_calls.append(
					{
						"ts": _now_ts(),
						"node_id": node_id,
						"kind": "execute",
						"attempt": attempt,
						"prompt_tokens": prompt_tokens,
						"completion_tokens": completion_tokens,
						"elapsed_ms": call_end_ms - call_start_ms,
					}
				)
				if reviewer is None and last_smoke is not None:
					sample_calls.append(
						{
							"ts": _now_ts(),
							"node_id": node_id,
							"kind": "smoke",
							"stage": "execute",
							"attempt": attempt,
							"elapsed_ms": last_smoke.elapsed_ms,
							"passed": last_smoke.passed,
							"details": last_smoke.details,
						}
					)
				break
			except Exception as exc:
				last_exc = exc
				try:
					sample_calls.append(
						{
							"ts": _now_ts(),
							"node_id": node_id,
							"kind": "execute",
							"attempt": attempt,
							"prompt_tokens": generator.count_tokens(attempt_prompt),
							"completion_tokens": generator.count_tokens(response) if "response" in locals() else 0,
							"elapsed_ms": (_time_ms() - call_start_ms) if "call_start_ms" in locals() else None,
							"error": str(exc),
						}
					)
				except Exception:
					pass
				try:
					last_hint = _summarize_validation_hints(last_candidate or Codes(""))
				except Exception:
					last_hint = None
				_write_text(layer_dir / f"error_attempt_{attempt}.txt", str(exc))
		if last_exc is not None or codes is None or not node_done:
			_write_text(
				layer_dir / "node_failure.txt",
				f"Node generation failed after {max_attempts} attempts.\nError: {last_exc}\n",
			)
			if stop_on_failure:
				raise RuntimeError(f"Node {node_id} failed after {max_attempts} attempts: {last_exc}") from last_exc

			if last_candidate is not None and last_candidate.codebooks:
				codes = last_candidate
			elif base_snapshot.strip():
				codes = Codes(base_snapshot)
			else:
				codes = Codes(
					"main.py\n```python\n"
					"def main():\n"
					"    print('Pipeline fallback: upstream node failed to generate parseable code.')\n\n"
					"if __name__ == '__main__':\n"
					"    main()\n"
					"```\n"
				)

		solutions[node_id] = codes
		_write_text(layer_dir / "snapshot.txt", codes.snapshot())
		_write_text(
			layer_dir / "execute_response.txt",
			(last_response or "(no execute response)").strip() + "\n",
		)
		_write_text(
			layer_dir / "reviewer_response.txt",
			(last_reviewer or "SKIPPED: reviewer not invoked").strip() + "\n",
		)
		if aggregate_ran:
			agg_payload = last_agg_response or "NO AGGREGATION RESPONSE"
		else:
			agg_payload = "SKIPPED: aggregation not invoked"
		_write_text(layer_dir / "aggregate_response.txt", agg_payload.strip() + "\n")
		sample_nodes.append(
			{
				"node_id": node_id,
				"title": node_meta.title,
				"allow_pass_todo": allow_pass_todo,
				"allow_format_placeholders": allow_format_placeholders,
				"parent_count": len(parent_ids),
				"failed": bool(last_exc),
				"smoke_passed": (last_smoke.passed if last_smoke is not None else None),
				"smoke_details": (last_smoke.details if last_smoke is not None else None),
				"smoke_elapsed_ms": (last_smoke.elapsed_ms if last_smoke is not None else None),
				"reviewer_enabled": bool(use_verifier),
				"reviewer_last_feedback": (last_reviewer if use_verifier else None),
			}
		)

	project_name = run_slug
	resolved_code_dir = Path(code_dir) if code_dir is not None else (warehouse_root / project_name)
	final_node = max(solutions)
	solutions[final_node].write_to_directory(resolved_code_dir)

	sample_end_ms = _time_ms()
	sample_metrics = {
		"task_name": getattr(spec, "task_name", None),
		"sample_name": sample_name,
		"started_at": sample_start_ts,
		"ended_at": _now_ts(),
		"elapsed_ms": sample_end_ms - sample_start_ms,
		"total_prompt_tokens": sum((c.get("prompt_tokens") or 0) for c in sample_calls),
		"total_completion_tokens": sum((c.get("completion_tokens") or 0) for c in sample_calls),
		"max_prompt_tokens": max([c.get("prompt_tokens") or 0 for c in sample_calls], default=0),
		"max_completion_tokens": max([c.get("completion_tokens") or 0 for c in sample_calls], default=0),
		"calls": sample_calls,
		"nodes": sample_nodes,
	}
	sample_metrics_path = resolved_run_dir / "sample_metrics.json"
	_write_json(sample_metrics_path, sample_metrics)

	return ExecutionResult(
		project_name=project_name,
		run_dir=resolved_run_dir,
		code_dir=resolved_code_dir,
		sample_metrics_path=sample_metrics_path,
	)
