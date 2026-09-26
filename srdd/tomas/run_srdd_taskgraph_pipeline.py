"""SRDD → TaskGraph → Execution pipeline.

Workflow:
1) Read SRDD sample(s) from CSV.
2) Generate a Task-Action graph using `task_graph` utilities.
3) Execute the graph using the current mini-MacNet executor.
4) Save logs + repo under a timestamp/category/sample directory layout.
5) Optionally run SRDD evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# REPO_ROOT is the repo root: the directory that CONTAINS morse/, scicode/, srdd/.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from srdd.eval import srdd_evaluator  # noqa: E402

from morse.taskgraph.generator import GraphGenerationError, LLMTaskGraphGenerator  # noqa: E402
from morse.llm.converter import convert_taskgraph  # noqa: E402
from srdd.tomas.executor import execute_task_graph  # noqa: E402
from morse.llm.hf_llm import HFGenerationConfig, HFSubprocessTextGenerator, HFTextGenerator  # noqa: E402


@dataclass
class SRDDSample:
	name: str
	description: str
	category: str


def read_srdd_samples(csv_path: Path) -> List[SRDDSample]:
	samples: List[SRDDSample] = []
	with csv_path.open(newline="", encoding="utf-8") as handle:
		for row in csv.DictReader(handle):
			name = (row.get("Name") or "").strip()
			desc = (row.get("Description") or "").strip()
			cat = (row.get("Category") or "Unknown").strip()
			if name and desc:
				samples.append(SRDDSample(name=name, description=desc, category=cat))
	return samples


def iter_first_n_by_category(samples: Iterable[SRDDSample], n: int) -> List[SRDDSample]:
	by_cat: Dict[str, List[SRDDSample]] = {}
	for s in samples:
		by_cat.setdefault(s.category, []).append(s)
	picked: List[SRDDSample] = []
	for cat in sorted(by_cat):
		picked.extend(by_cat[cat][:n])
	return picked


def iter_slice_by_category(
	samples: Iterable[SRDDSample],
	*,
	offset: int,
	limit: int,
	category_parity: str | None = None,
) -> List[SRDDSample]:
	by_cat: Dict[str, List[SRDDSample]] = {}
	for s in samples:
		by_cat.setdefault(s.category, []).append(s)
	picked: List[SRDDSample] = []

	categories = sorted(by_cat)
	if category_parity in {"odd", "even"}:
		want_odd = category_parity == "odd"
		categories = [cat for idx, cat in enumerate(categories, start=1) if (idx % 2 == 1) == want_odd]

	for cat in categories:
		start = max(offset, 0)
		end = start + max(limit, 0)
		picked.extend(by_cat[cat][start:end])
	return picked


def sanitize(name: str) -> str:
	return "".join([c if c.isalnum() else "_" for c in name]).strip("_") or "item"


def build_graph_root_index(graph_root: Path) -> Dict[Tuple[str, str], Path]:
	index: Dict[Tuple[str, str], Path] = {}
	if not graph_root or not graph_root.exists():
		return index
	for sample_json in graph_root.rglob("sample.json"):
		try:
			payload = json.loads(sample_json.read_text(encoding="utf-8"))
		except Exception:
			continue
		name = sanitize(str(payload.get("name") or ""))
		category = sanitize(str(payload.get("category") or ""))
		if not name or not category:
			continue
		index[(category, name)] = sample_json.parent
	return index


def parse_srdd_report(report_path: Path) -> Dict[str, float]:
	"""Parse ChatDev_macnet/srdd_evaluator.py output report."""
	metrics: Dict[str, float] = {}
	if not report_path.exists():
		return metrics
	for line in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
		parts = line.split(":", 1)
		if len(parts) != 2:
			continue
		key = parts[0].strip().lower()
		value = parts[1].strip()
		if key in {"completeness", "executability", "consistency", "consistency_stripped", "eci_product", "eci_mean"}:
			try:
				metrics[key] = float(value)
			except ValueError:
				continue
	return metrics


def read_embedding_metrics(metrics_path: Path) -> Dict[str, float]:
	if not metrics_path.exists():
		return {}
	try:
		payload = json.loads(metrics_path.read_text(encoding="utf-8"))
	except Exception:
		return {}
	out: Dict[str, float] = {}
	for key in ("consistency_embedding", "consistency_embedding_stripped"):
		val = payload.get(key)
		if isinstance(val, (int, float)):
			out[key] = float(val)
	return out


def read_sample_metrics(sample_metrics_path: Path) -> Dict[str, Any]:
	if not sample_metrics_path.exists():
		return {}
	try:
		return json.loads(sample_metrics_path.read_text(encoding="utf-8"))
	except Exception:
		return {}


def mean(values: List[float]) -> float:
	return sum(values) / len(values) if values else 0.0


def collect_results(timestamp_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
	all_rows: List[Dict[str, Any]] = []
	by_category: Dict[str, List[Dict[str, Any]]] = {}
	for category_dir in sorted([p for p in timestamp_root.iterdir() if p.is_dir()]):
		category_key = category_dir.name
		for sample_dir in sorted([p for p in category_dir.iterdir() if p.is_dir()]):
			report_path = sample_dir / "srdd_report.txt"
			metrics = parse_srdd_report(report_path)
			metrics.update(read_embedding_metrics(sample_dir / "srdd_embedding_metrics.json"))
			sample_metrics_path = sample_dir / "log" / "sample_metrics.json"
			sm = read_sample_metrics(sample_metrics_path)
			graph_status_path = sample_dir / "graph_generation.json"
			graph_status = {}
			if graph_status_path.exists():
				try:
					graph_status = json.loads(graph_status_path.read_text(encoding="utf-8"))
				except Exception:
					graph_status = {}

			# Ignore samples that never reached execution/evaluation (e.g., graph generation failed).
			if not report_path.exists() and not sample_metrics_path.exists():
				continue
			if graph_status.get("status") == "failed":
				continue
			row = {
				"category": category_key,
				"sample": sample_dir.name,
				"srdd": metrics,
				"tokens": {
					"total_prompt_tokens": sm.get("total_prompt_tokens"),
					"total_completion_tokens": sm.get("total_completion_tokens"),
					"max_prompt_tokens": sm.get("max_prompt_tokens"),
					"max_completion_tokens": sm.get("max_completion_tokens"),
				},
				"timing": {
					"elapsed_ms": sm.get("elapsed_ms"),
				},
				"paths": {
					"sample_dir": str(sample_dir),
					"repo_dir": str(sample_dir / "repo"),
					"log_dir": str(sample_dir / "log"),
				},
			}
			all_rows.append(row)
			by_category.setdefault(category_key, []).append(row)
	return all_rows, by_category


def compute_means(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
	srdd_keys = [
		"executability",
		"completeness",
		"consistency",
		"consistency_stripped",
		"consistency_embedding",
		"consistency_embedding_stripped",
	]
	srdd_means = {
		k: mean([r.get("srdd", {}).get(k) for r in rows if r.get("srdd", {}).get(k) is not None])
		for k in srdd_keys
	}
	srdd_means["eci_mean"] = (
		float(srdd_means.get("executability", 0.0)) + float(srdd_means.get("completeness", 0.0)) + float(srdd_means.get("consistency", 0.0))
	) / 3.0
	srdd_means["eci_product"] = (
		float(srdd_means.get("executability", 0.0))
		* float(srdd_means.get("completeness", 0.0))
		* float(srdd_means.get("consistency_stripped", 0.0))
	)
	srdd_means["eci_mean_embedding"] = (
		float(srdd_means.get("executability", 0.0))
		+ float(srdd_means.get("completeness", 0.0))
		+ float(srdd_means.get("consistency_embedding", 0.0))
	) / 3.0
	srdd_means["eci_product_embedding"] = (
		float(srdd_means.get("executability", 0.0))
		* float(srdd_means.get("completeness", 0.0))
		* float(srdd_means.get("consistency_embedding_stripped", 0.0))
	)

	token_keys = ["total_prompt_tokens", "total_completion_tokens", "max_prompt_tokens", "max_completion_tokens"]
	token_means = {
		k: mean([float(r.get("tokens", {}).get(k)) for r in rows if r.get("tokens", {}).get(k) is not None])
		for k in token_keys
	}
	token_means["total_tokens"] = token_means["total_prompt_tokens"] + token_means["total_completion_tokens"]

	time_means = {
		"elapsed_ms": mean([float(r.get("timing", {}).get("elapsed_ms")) for r in rows if r.get("timing", {}).get("elapsed_ms") is not None])
	}
	return {
		"count": len(rows),
		"srdd_mean": srdd_means,
		"token_mean": token_means,
		"time_mean": time_means,
	}


def update_summaries(timestamp_root: Path) -> None:
	@contextmanager
	def _lock(path: Path):
		try:
			import fcntl  # unix only

			handle = path.open("w", encoding="utf-8")
			try:
				fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
				yield
			finally:
				try:
					fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
				except Exception:
					pass
				handle.close()
		except Exception:
			# Best-effort fallback when file locking is unavailable.
			yield

	with _lock(timestamp_root / ".summary.lock"):
		all_rows, by_category = collect_results(timestamp_root)
		overall = compute_means(all_rows)

		summary = {
			"timestamp_root": str(timestamp_root),
			"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
			"overall": overall,
			"categories": {cat: compute_means(rows) for cat, rows in by_category.items()},
		}
		(timestamp_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

		for cat, rows in by_category.items():
			cat_dir = timestamp_root / cat
			payload = {
				"category": cat,
				"updated_at": summary["updated_at"],
				"stats": compute_means(rows),
			}
			(cat_dir / "category_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="SRDD batch pipeline: generate task graph then execute it.")
	parser.add_argument("--srdd-csv", type=Path, default=REPO_ROOT / "srdd" / "data" / "SRDD.csv")
	parser.add_argument(
		"--sample-name",
		type=str,
		default=None,
		help="Run exactly one SRDD sample by Name (ignores per-category selection).",
	)
	parser.add_argument("--per-category", type=int, default=2, help="How many samples to run per category.")
	parser.add_argument("--per-category-offset", type=int, default=0, help="Skip this many samples per category before selecting.")
	parser.add_argument(
		"--category-parity",
		choices=["odd", "even"],
		default=None,
		help="Run only odd/even categories by sorted category index (1-based).",
	)
	parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "srdd" / "tomas" / "srdd_runs")
	parser.add_argument("--timestamp", type=str, default=None, help="Optional fixed timestamp folder name (useful for resume).")
	parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose output folder already contains done.txt.")
	parser.add_argument(
		"--reuse-existing-graph",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="When resuming with --skip-existing, reuse an existing task_graph.json if graph_generation.json reports status=ok.",
	)
	parser.add_argument(
		"--graph-root",
		type=Path,
		default=None,
		help="Optional root containing precomputed graphs (expects sample.json + task_graph.json).",
	)
	parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Visible GPUs for the whole run (default: 0-3).")

	parser.add_argument("--graph-model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
	parser.add_argument("--graph-max-new-tokens", type=int, default=2048)
	parser.add_argument("--graph-temperature", type=float, default=0.2)
	parser.add_argument(
		"--graph-device-map",
		type=str,
		default="auto",
		help="Transformers device_map for task-graph generation (default: auto = all visible GPUs). Use 'none' to disable.",
	)
	parser.add_argument(
		"--graph-device",
		type=str,
		default="0",
		help="Device for task-graph generation pipeline: -1 for CPU or a CUDA device index (default: 0).",
	)
	parser.add_argument("--graph-torch-dtype", type=str, default="bfloat16", help="Torch dtype for graph model (default: bfloat16).")
	parser.add_argument("--graph-retries", type=int, default=3, help="Retry count for task-graph JSON generation (max 3).")
	parser.add_argument("--code-use-gpus", type=str, default="0,1,2,3", help="Which visible GPUs to allow for code model placement (default: 0,1,2,3).")

	parser.add_argument("--code-model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
	parser.add_argument("--code-max-new-tokens", type=int, default=4096)
	parser.add_argument("--code-temperature", type=float, default=0.2)
	parser.add_argument("--device-map", type=str, default="auto")
	parser.add_argument("--device", type=int, default=0)
	parser.add_argument("--torch-dtype", type=str, default=None)
	parser.add_argument(
		"--code-subprocess",
		action="store_true",
		help="Run code-generation LLM in a subprocess and auto-restart on CUDA failures.",
	)
	parser.add_argument(
		"--keep-models-loaded",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Keep graph/code models loaded for the entire run (avoid re-loading per sample).",
	)

	parser.add_argument("--max-attempts", type=int, default=3)
	parser.add_argument("--stop-on-failure", action="store_true")
	parser.add_argument(
		"--use-verifier",
		action="store_true",
		help="Enable a MacNet-style reviewer agent to provide repair feedback (and optionally request compile/run checks) between attempts.",
	)
	parser.add_argument("--no-eval-srdd", action="store_false", dest="eval_srdd")
	parser.set_defaults(eval_srdd=True)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.gpus and args.gpus.strip():
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

	graph_device = int(args.graph_device)

	samples = read_srdd_samples(args.srdd_csv)
	if args.sample_name:
		selected = [s for s in samples if s.name == args.sample_name]
		if not selected:
			raise ValueError(f"Sample '{args.sample_name}' not found in {args.srdd_csv}")
	else:
		selected = iter_slice_by_category(
			samples, offset=args.per_category_offset, limit=args.per_category, category_parity=args.category_parity
		)

	graph_root_index: Dict[Tuple[str, str], Path] = {}
	if args.graph_root:
		graph_root_index = build_graph_root_index(args.graph_root)

	timestamp = (args.timestamp or "").strip() or time.strftime("%Y%m%d_%H%M%S")
	root = args.output_root / timestamp
	root.mkdir(parents=True, exist_ok=True)

	dtype_kwargs: Dict[str, object] = {}
	try:
		import torch
		if args.graph_torch_dtype and hasattr(torch, args.graph_torch_dtype):
			dtype_kwargs["torch_dtype"] = getattr(torch, args.graph_torch_dtype)
	except Exception:
		dtype_kwargs = {}

	allowed = [x.strip() for x in (args.code_use_gpus or "").split(",") if x.strip()]
	max_memory: Dict[object, str] = {}
	try:
		import torch
		if torch.cuda.is_available():
			allowed_set = set(allowed) if allowed else {str(i) for i in range(torch.cuda.device_count())}
			for idx in range(torch.cuda.device_count()):
				# accelerate expects integer device ids (0,1,2,...) for CUDA, not "cuda:0".
				max_memory[idx] = ("80GiB" if str(idx) in allowed_set else "0GiB")
			max_memory["cpu"] = "0GiB"
	except Exception:
		max_memory = {}

	def _release_model_memory() -> None:
		try:
			import gc
			import torch

			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		except Exception:
			return

	def _as_device_map(raw: str | None) -> str | None:
		text = (raw or "").strip()
		if not text or text.lower() == "none":
			return None
		return text

	graph_device_map = _as_device_map(args.graph_device_map)
	code_device_map = _as_device_map(args.device_map)

	# Optionally keep models alive across the entire run (faster for many samples).
	graph_generator: Optional[LLMTaskGraphGenerator] = None
	code_generator: Optional[HFTextGenerator] = None
	if args.keep_models_loaded:
		if not args.graph_root:
			graph_generator = LLMTaskGraphGenerator(
				model_name=args.graph_model_name,
				max_new_tokens=args.graph_max_new_tokens,
				temperature=args.graph_temperature,
				device=graph_device,
				device_map=graph_device_map,
				pipeline_task="text-generation",
				model_kwargs=dtype_kwargs,
			)

		code_cfg = HFGenerationConfig(
			model_name=args.code_model_name,
			device=args.device,
			device_map=code_device_map,
			max_memory=max_memory or None,
			max_new_tokens=args.code_max_new_tokens,
			temperature=args.code_temperature,
			torch_dtype=args.torch_dtype,
		)
		code_generator = HFSubprocessTextGenerator(code_cfg) if args.code_subprocess else HFTextGenerator(code_cfg)

	for sample in selected:
		category_dir = root / sanitize(sample.category)
		sample_dir = category_dir / sanitize(sample.name)
		log_dir = sample_dir / "log"
		repo_dir = sample_dir / "repo"
		sample_dir.mkdir(parents=True, exist_ok=True)
		category_dir.mkdir(parents=True, exist_ok=True)

		if args.skip_existing:
			if (sample_dir / "skipped.txt").exists():
				continue
			if (log_dir / "sample_metrics.json").exists():
				continue
			if (sample_dir / "done.txt").exists() and (log_dir / "sample_metrics.json").exists():
				continue

		# 1) Generate task graph then immediately execute the sample.
		graph_path = sample_dir / "task_graph.json"
		graph_metrics_path = sample_dir / "graph_generation.json"

		reuse_graph = False
		if graph_root_index:
			source_dir = graph_root_index.get((category_dir.name, sample_dir.name))
			if source_dir and (source_dir / "task_graph.json").exists():
				shutil.copy2(source_dir / "task_graph.json", graph_path)
				if (source_dir / "graph_generation.json").exists():
					shutil.copy2(source_dir / "graph_generation.json", graph_metrics_path)
				else:
					graph_metrics_path.write_text(
						json.dumps({"status": "ok", "source": str(source_dir)}, ensure_ascii=False, indent=2),
						encoding="utf-8",
					)
				reuse_graph = True
		if args.skip_existing and args.reuse_existing_graph and graph_path.exists() and graph_metrics_path.exists():
			try:
				payload = json.loads(graph_metrics_path.read_text(encoding="utf-8"))
				if isinstance(payload, dict) and payload.get("status") == "ok":
					reuse_graph = True
			except Exception:
				reuse_graph = False

		last_error = None
		if reuse_graph:
			last_error = None
		else:
			for attempt in range(1, min(max(args.graph_retries, 1), 3) + 1):
				try:
					if graph_generator is None:
						graph_generator = LLMTaskGraphGenerator(
							model_name=args.graph_model_name,
							max_new_tokens=args.graph_max_new_tokens,
							temperature=args.graph_temperature,
							device=graph_device,
							device_map=graph_device_map,
							pipeline_task="text-generation",
							model_kwargs=dtype_kwargs,
						)

					start = time.time()
					graph = graph_generator.generate(sample.name, sample.description)
					elapsed_ms = int((time.time() - start) * 1000)
					graph.save_json(graph_path)

					tokenizer = getattr(getattr(graph_generator, "_pipeline", None), "tokenizer", None)
					prompt = graph_generator.last_prompt or ""
					response = graph_generator.last_response_text or ""
					prompt_tokens = len(tokenizer.encode(prompt)) if tokenizer is not None else None
					completion_tokens = len(tokenizer.encode(response)) if tokenizer is not None else None
					graph_metrics_path.write_text(
						json.dumps(
							{
								"status": "ok",
								"attempt": attempt,
								"elapsed_ms": elapsed_ms,
								"prompt_tokens": prompt_tokens,
								"completion_tokens": completion_tokens,
							},
							ensure_ascii=False,
							indent=2,
						),
						encoding="utf-8",
					)
					last_error = None
					break
				except GraphGenerationError as exc:
					last_error = exc
					(sample_dir / f"graph_error_attempt_{attempt}.txt").write_text(str(exc), encoding="utf-8")
					if getattr(graph_generator, "last_response_text", None):
						(sample_dir / f"graph_response_attempt_{attempt}.txt").write_text(
							graph_generator.last_response_text or "", encoding="utf-8"
						)
		if not args.keep_models_loaded:
			try:
				del graph_generator
			except Exception:
				pass
			graph_generator = None
			_release_model_memory()

		if last_error is not None:
			graph_metrics_path.write_text(
				json.dumps(
					{"status": "failed", "error": str(last_error), "attempts": min(max(args.graph_retries, 1), 3)},
					ensure_ascii=False,
					indent=2,
				),
				encoding="utf-8",
			)
			(sample_dir / "skipped.txt").write_text(
				"Skipped: task graph generation failed after retries (no fallback).\n", encoding="utf-8"
			)
			update_summaries(root)
			continue

		if code_generator is None:
			code_cfg = HFGenerationConfig(
				model_name=args.code_model_name,
				device=args.device,
				device_map=code_device_map,
				max_memory=max_memory or None,
				max_new_tokens=args.code_max_new_tokens,
				temperature=args.code_temperature,
				torch_dtype=args.torch_dtype,
			)
			code_generator = HFSubprocessTextGenerator(code_cfg) if args.code_subprocess else HFTextGenerator(code_cfg)

		log_dir.mkdir(parents=True, exist_ok=True)
		repo_dir.mkdir(parents=True, exist_ok=True)

		spec = convert_taskgraph(graph_path)
		result = execute_task_graph(
			spec=spec,
			generator=code_generator,
			output_root=log_dir,
			warehouse_root=repo_dir.parent,
			sample_name=sample.name,
			require_aggregation=True,
			max_attempts=args.max_attempts,
			stop_on_failure=args.stop_on_failure,
			use_verifier=bool(args.use_verifier),
			run_dir=log_dir,
			code_dir=repo_dir,
		)

		if not args.keep_models_loaded:
			try:
				del code_generator
			except Exception:
				pass
			code_generator = None
			_release_model_memory()

		if args.eval_srdd:
			report_path = sample_dir / "srdd_report.txt"
			try:
				srdd_evaluator.evaluate(sample.name, str(args.srdd_csv), str(result.code_dir), str(report_path))
			except Exception as exc:
				(sample_dir / "srdd_eval_error.txt").write_text(str(exc), encoding="utf-8")

		(sample_dir / "done.txt").write_text("ok\n", encoding="utf-8")
		update_summaries(root)


if __name__ == "__main__":
	main()
