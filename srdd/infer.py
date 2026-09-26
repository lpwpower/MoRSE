from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# REPO_ROOT is the directory that CONTAINS morse/, scicode/, srdd/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from srdd.eval import srdd_evaluator  # noqa: E402

from morse.taskgraph.generator import GraphGenerationError, LLMTaskGraphGenerator  # noqa: E402
from srdd.tomas.codes import Codes  # noqa: E402
from morse.llm.converter import TaskGraphSpecification, convert_taskgraph  # noqa: E402
from srdd.tomas.executor import (  # noqa: E402
	_summarize_validation_hints,
	build_aggregate_prompt,
	build_executor_prompt,
	validate_codes,
)
from srdd.tomas.review_test import smoke_test_repo  # noqa: E402
from srdd.tomas.run_srdd_taskgraph_pipeline import update_summaries  # noqa: E402

from morse.mole.mole_generator import GenerationConfig, MoLEGenerator  # noqa: E402
from morse.mole.mole_lora import LoRAConfig, inject_mole_lora  # noqa: E402
from morse.mole.roles import Role  # noqa: E402
from morse.mole.router import SubtaskRouter, SubtaskRouterConfig  # noqa: E402
from srdd.role_subtask import TitleEmbedder  # noqa: E402


ROLE_EXPERT_IDS = {
	Role.EXECUTE: 0,
	Role.AGGREGATE: 1,
}


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


def iter_slice_by_category(samples: List[SRDDSample], *, offset: int, limit: int) -> List[SRDDSample]:
	by_cat: Dict[str, List[SRDDSample]] = {}
	for s in samples:
		by_cat.setdefault(s.category, []).append(s)
	picked: List[SRDDSample] = []
	for cat in sorted(by_cat):
		start = max(offset, 0)
		end = start + max(limit, 0)
		picked.extend(by_cat[cat][start:end])
	return picked


def iter_slice_by_category_parity(
	samples: List[SRDDSample],
	*,
	offset: int,
	limit: int,
	parity: str,
) -> List[SRDDSample]:
	"""Pick a per-category slice, optionally filtering categories by (even/odd) index.

	Category index is defined by sorting category names lexicographically and assigning
	0..N-1. 'even' selects index%2==0, 'odd' selects index%2==1, 'none' selects all.
	"""
	parity_key = (parity or "none").strip().lower()
	if parity_key not in {"none", "even", "odd"}:
		raise ValueError(f"Invalid parity: {parity}")
	by_cat: Dict[str, List[SRDDSample]] = {}
	for s in samples:
		by_cat.setdefault(s.category, []).append(s)
	categories = sorted(by_cat)
	keep: set[str]
	if parity_key == "none":
		keep = set(categories)
	else:
		want_even = parity_key == "even"
		keep = {cat for idx, cat in enumerate(categories) if (idx % 2 == 0) == want_even}
	picked: List[SRDDSample] = []
	for cat in categories:
		if cat not in keep:
			continue
		start = max(offset, 0)
		end = start + max(limit, 0)
		picked.extend(by_cat[cat][start:end])
	return picked


def iter_slice_by_category_shard(
	samples: List[SRDDSample],
	*,
	offset: int,
	limit: int,
	shards: int,
	shard_idx: int,
) -> List[SRDDSample]:
	"""Pick a per-category slice, filtering categories by index%shards==shard_idx.

	Category index is defined by sorting category names lexicographically and assigning
	0..N-1. This is a generalization of odd/even splitting to N shards.
	"""
	n = int(shards)
	k = int(shard_idx)
	if n <= 0:
		raise ValueError(f"Invalid shards: {shards}")
	if k < 0 or k >= n:
		raise ValueError(f"Invalid shard_idx: {shard_idx} (shards={shards})")
	by_cat: Dict[str, List[SRDDSample]] = {}
	for s in samples:
		by_cat.setdefault(s.category, []).append(s)
	categories = sorted(by_cat)
	keep = {cat for idx, cat in enumerate(categories) if (idx % n) == k}
	picked: List[SRDDSample] = []
	for cat in categories:
		if cat not in keep:
			continue
		start = max(offset, 0)
		end = start + max(limit, 0)
		picked.extend(by_cat[cat][start:end])
	return picked


def sanitize(name: str) -> str:
	return "".join([c if c.isalnum() else "_" for c in name]).strip("_") or "item"


def _write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _time_ms() -> int:
	return int(time.time() * 1000)


def _now_ts() -> str:
	return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _count_tokens(tokenizer, text: str) -> int:
	try:
		return int(len(tokenizer.encode(text)))
	except Exception:
		return 0


def _try_acquire_lock(path: Path, *, stale_seconds: int) -> bool:
	path.parent.mkdir(parents=True, exist_ok=True)
	for _ in range(2):
		try:
			fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
		except FileExistsError:
			try:
				age_s = time.time() - path.stat().st_mtime
			except FileNotFoundError:
				continue
			if stale_seconds > 0 and age_s > float(stale_seconds):
				try:
					path.unlink()
				except FileNotFoundError:
					pass
				continue
			return False
		else:
			with os.fdopen(fd, "w", encoding="utf-8") as handle:
				handle.write(f"pid={os.getpid()}\nstarted_at={_now_ts()}\n")
			return True
	return False


def _release_lock(path: Path) -> None:
	try:
		path.unlink()
	except FileNotFoundError:
		return


def _ensure_symlink_dir(*, link_root: Path, target_dir: Path) -> None:
	link_root.mkdir(parents=True, exist_ok=True)
	link_path = link_root / target_dir.name
	if link_path.exists() or link_path.is_symlink():
		try:
			if link_path.is_symlink() and link_path.resolve() == target_dir.resolve():
				return
		except Exception:
			return
		return
	try:
		os.symlink(str(target_dir), str(link_path), target_is_directory=True)
	except FileExistsError:
		return


def _as_device_map(raw: str | None) -> str | None:
	text = (raw or "").strip()
	if not text or text.lower() == "none":
		return None
	return text


def _load_mole_from_checkpoint(
	*,
	checkpoint_dir: Path,
	exec_model_name: str,
	torch_dtype: str,
	device: torch.device,
	num_role_experts: int,
	num_subtask_experts: int,
	subtask_top_k: int,
	lora_rank: int,
	lora_alpha: float,
	lora_last_n_layers: int,
	max_new_tokens: int,
	no_ckpt_load: bool = False,
) -> Tuple[MoLEGenerator, SubtaskRouter, TitleEmbedder, object]:
	from transformers import AutoModelForCausalLM, AutoTokenizer

	if no_ckpt_load:
		ckpt = None
	else:
		ckpt = Path(checkpoint_dir)
		if not ckpt.exists():
			raise FileNotFoundError(f"Checkpoint dir not found: {ckpt}")
		required = ["router.pt", "title_embedder.pt", "lora_state.pt"]
		for name in required:
			if not (ckpt / name).exists():
				raise FileNotFoundError(f"Missing checkpoint file: {ckpt / name}")

	tok = AutoTokenizer.from_pretrained(exec_model_name)
	if getattr(tok, "pad_token_id", None) is None:
		tok.pad_token = tok.eos_token
	dtype = getattr(torch, str(torch_dtype), None)
	dtype = dtype if dtype is not None else torch.bfloat16
	model = AutoModelForCausalLM.from_pretrained(exec_model_name, torch_dtype=dtype)
	model.to(device)
	model.eval()
	try:
		if getattr(model, "config", None) is not None:
			model.config.use_cache = True  # type: ignore[attr-defined]
	except Exception:
		pass
	for p in model.parameters():
		p.requires_grad = False

	lora_cfg = LoRAConfig(
		num_experts=int(num_role_experts + num_subtask_experts),
		top_k=int(subtask_top_k) + 1,
		rank=int(lora_rank),
		alpha=float(lora_alpha),
		target_modules=("q_proj", "v_proj", "o_proj"),
		last_n_layers=int(lora_last_n_layers),
	)
	inject_mole_lora(model, cfg=lora_cfg)

	router_cfg = SubtaskRouterConfig(num_experts=int(num_subtask_experts), top_k=int(subtask_top_k))
	router = SubtaskRouter(router_cfg).to(device)
	title_embedder = TitleEmbedder(model=model, tokenizer=tok, out_dim=router_cfg.title_emb_dim).to(device)

	if ckpt is not None:
		router.load_state_dict(torch.load(ckpt / "router.pt", map_location=device))
		_te_sd = torch.load(ckpt / "title_embedder.pt", map_location=device)
		_missing, _unexpected = title_embedder.load_state_dict(_te_sd, strict=False)
		if "proj.weight" in _unexpected:
			raise RuntimeError("ckpt has proj.weight marked unexpected — class mismatch")
		if "proj.bias" in _missing and hasattr(title_embedder.proj, "bias") and title_embedder.proj.bias is not None:
			with torch.no_grad():
				title_embedder.proj.bias.zero_()
		lora_state = torch.load(ckpt / "lora_state.pt", map_location="cpu")
		name_to_param = dict(model.named_parameters())
		for name, tensor in lora_state.items():
			p = name_to_param.get(name)
			if p is None:
				continue
			p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))

	gen_cfg = GenerationConfig(
		model_name=str(exec_model_name),
		max_new_tokens=int(max_new_tokens),
		temperature=0.2,
		top_p=0.95,
		torch_dtype=str(torch_dtype),
		device=int(getattr(device, "index", 0) or 0),
	)
	mole_gen = MoLEGenerator(model=model, tokenizer=tok, device=device, gen_cfg=gen_cfg)
	return mole_gen, router, title_embedder, tok


def _subtask_text(title: str, description: str) -> str:
	title = (title or "").strip()
	desc = (description or "").strip()
	if title and desc:
		return f"{title}: {desc}"
	return title or desc or "none"


def _select_experts(
	*,
	router: SubtaskRouter,
	title_embedder: TitleEmbedder,
	device: torch.device,
	role: Role,
	subtask_text: str,
	mode: str,
	subtask_expert_offset: int,
) -> Tuple[torch.Tensor, List[int]]:
	role_expert_id = ROLE_EXPERT_IDS.get(role)
	if role_expert_id is None:
		raise ValueError(f"Unknown role for expert routing: {role}")
	title_emb = title_embedder(subtask_text)
	logits = router(title_emb=title_emb)
	mode_key = (mode or "greedy").strip().lower()
	if mode_key == "sample":
		subtask_ids, _ = router.sample_topk(logits)
	else:
		subtask_ids, _ = router.greedy_topk(logits)
	subtask_ids = subtask_ids + int(subtask_expert_offset)
	role_tensor = torch.tensor([int(role_expert_id)], device=device, dtype=torch.long)
	expert_ids = torch.cat([role_tensor, subtask_ids.to(device=device)])
	chosen_subtask = [int(x) - int(subtask_expert_offset) for x in subtask_ids.detach().cpu().tolist()]
	return expert_ids.to(device=device, dtype=torch.long), chosen_subtask


def _select_aggregate_experts(
	*,
	device: torch.device,
	parent_subtask_experts: List[int],
	subtask_expert_offset: int,
) -> torch.Tensor:
	role_expert_id = ROLE_EXPERT_IDS.get(Role.AGGREGATE)
	if role_expert_id is None:
		raise ValueError(f"Unknown role for expert routing: {Role.AGGREGATE}")
	role_tensor = torch.tensor([int(role_expert_id)], device=device, dtype=torch.long)
	if not parent_subtask_experts:
		return role_tensor
	subtask_ids = torch.tensor(
		[int(subtask_expert_offset) + int(idx) for idx in parent_subtask_experts],
		device=device,
		dtype=torch.long,
	)
	return torch.cat([role_tensor, subtask_ids])


def _run_smoke_test(
	*,
	codes: Codes,
	layer_dir: Path,
	stage: str,
	attempt: int,
	timeout_s: int,
) -> Tuple[bool, int, str]:
	import shutil

	repo_dir = layer_dir / f"{stage}_repo_attempt_{attempt}"
	if repo_dir.exists():
		shutil.rmtree(repo_dir)
	codes.write_to_directory(repo_dir)
	result = smoke_test_repo(repo_dir, timeout_s=int(timeout_s))
	_write_text(
		layer_dir / f"{stage}_smoke_attempt_{attempt}.txt",
		f"passed: {result.passed}\n" f"elapsed_ms: {result.elapsed_ms}\n" "details:\n" f"{result.details}\n",
	)
	return bool(result.passed), int(result.elapsed_ms), str(result.details)


def execute_task_graph_mole(
	*,
	spec: TaskGraphSpecification,
	mole_gen: MoLEGenerator,
	router: SubtaskRouter,
	title_embedder: TitleEmbedder,
	tokenizer,
	log_dir: Path,
	repo_dir: Path,
	sample_name: str,
	router_mode: str,
	max_attempts: int,
	smoke_timeout_s: int,
	subtask_expert_offset: int,
) -> None:
	log_dir.mkdir(parents=True, exist_ok=True)
	repo_dir.mkdir(parents=True, exist_ok=True)

	_write_text(log_dir / "task_description.txt", spec.task_description)
	(log_dir / "task_graph.json").write_text(json.dumps(spec.__dict__, default=str, indent=2), encoding="utf-8")

	solutions: Dict[int, Codes] = {}
	predecessors: Dict[int, List[int]] = {nid: [] for nid in spec.node_metadata}
	for edge in spec.edge_strings:
		src, dst = edge.split("->", 1)
		predecessors[int(dst)].append(int(src))
	node_ids = sorted(spec.node_metadata)
	relaxed_until = len(node_ids) // 3

	sample_start_ms = _time_ms()
	sample_start_ts = _now_ts()
	sample_calls: List[Dict[str, Any]] = []
	sample_nodes: List[Dict[str, Any]] = []
	node_subtask_experts: Dict[int, List[int]] = {}

	for node_index, node_id in enumerate(node_ids):
		node_meta = spec.node_metadata[node_id]
		parent_ids = sorted([pid for pid in predecessors.get(node_id, []) if pid in solutions])
		layer_dir = log_dir / f"node_{node_id:02d}"
		layer_dir.mkdir(parents=True, exist_ok=True)
		allow_pass_todo = node_index < relaxed_until
		allow_format_placeholders = node_index < relaxed_until

		base_codes: Codes | None = None
		node_subtask_text = _subtask_text(node_meta.title, node_meta.description)
		if not parent_ids:
			base_snapshot = ""
		elif len(parent_ids) == 1:
			base_codes = solutions[parent_ids[0]]
			base_snapshot = base_codes.snapshot()
		else:
			parent_snaps = [solutions[parent].snapshot() for parent in parent_ids]
			base_codes = solutions[parent_ids[0]]
			base_snapshot = base_codes.snapshot()
			parent_subtask_union = sorted({idx for pid in parent_ids for idx in node_subtask_experts.get(pid, [])})
			expert_ids = _select_aggregate_experts(
				device=mole_gen.device,
				parent_subtask_experts=parent_subtask_union,
				subtask_expert_offset=subtask_expert_offset,
			)

			last_exc = None
			last_hint = None
			last_agg_codes = None
			last_smoke: Tuple[bool, int, str] | None = None
			for attempt in range(1, int(max_attempts) + 1):
				try:
					candidate_snaps = list(parent_snaps)
					if last_agg_codes is not None and last_agg_codes.codebooks:
						candidate_snaps.append(last_agg_codes.snapshot())
					prompt = build_aggregate_prompt(task_description=spec.task_description, parent_snapshots=candidate_snaps)
					if last_exc is not None:
						prompt = (
							prompt
							+ "\n\nPrevious attempt failed validation/smoke test with error:\n"
							+ str(last_exc)
							+ "\n\nRepair checklist:\n"
							+ (last_hint or "- Regenerate output strictly following constraints.")
							+ "\n\nNow regenerate the FULL repository meeting ALL constraints.\n"
						)
					if attempt == 1:
						_write_text(layer_dir / "aggregate_prompt.txt", prompt)
					_write_text(layer_dir / f"aggregate_prompt_attempt_{attempt}.txt", prompt)

					prompt_tokens = _count_tokens(tokenizer, prompt)
					call_start_ms = _time_ms()
					agg_response, _, _ = mole_gen.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
					call_end_ms = _time_ms()
					completion_tokens = _count_tokens(tokenizer, agg_response)
					_write_text(layer_dir / f"aggregate_response_attempt_{attempt}.txt", agg_response)
					agg_codes = Codes(agg_response)
					last_agg_codes = agg_codes
					validate_codes(
						agg_codes,
						allow_pass_todo=allow_pass_todo,
						allow_format_placeholders=allow_format_placeholders,
					)
					smoke_ok, smoke_ms, smoke_details = _run_smoke_test(
						codes=agg_codes,
						layer_dir=layer_dir,
						stage="aggregate",
						attempt=attempt,
						timeout_s=int(smoke_timeout_s),
					)
					last_smoke = (smoke_ok, smoke_ms, smoke_details)

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
					sample_calls.append(
						{
							"ts": _now_ts(),
							"node_id": node_id,
							"kind": "smoke",
							"stage": "aggregate",
							"attempt": attempt,
							"elapsed_ms": smoke_ms,
							"passed": smoke_ok,
							"details": smoke_details,
						}
					)
					if not smoke_ok:
						raise RuntimeError(f"Smoke test failed for node {node_id} (aggregate) attempt {attempt}: {smoke_details}")

					base_codes = agg_codes
					base_snapshot = agg_codes.snapshot()
					last_exc = None
					break
				except Exception as exc:
					last_exc = exc
					try:
						last_hint = _summarize_validation_hints(last_agg_codes or Codes(""))
					except Exception:
						last_hint = None
					_write_text(layer_dir / f"aggregate_error_attempt_{attempt}.txt", str(exc))

			if last_exc is not None:
				_write_text(layer_dir / "aggregate_failure.txt", f"Aggregation failed after {int(max_attempts)} attempts.\nError: {last_exc}\n")
				if last_agg_codes is not None and last_agg_codes.codebooks:
					base_codes = last_agg_codes
					base_snapshot = last_agg_codes.snapshot()
				else:
					base_codes = solutions[parent_ids[0]]
					base_snapshot = base_codes.snapshot()

		# Execute node.
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

		expert_ids, subtask_ids = _select_experts(
			router=router,
			title_embedder=title_embedder,
			device=mole_gen.device,
			role=Role.EXECUTE,
			subtask_text=node_subtask_text,
			mode=router_mode,
			subtask_expert_offset=subtask_expert_offset,
		)
		node_subtask_experts[node_id] = subtask_ids

		last_exc: Exception | None = None
		last_hint: str | None = None
		codes: Codes | None = None
		last_candidate: Codes | None = None
		last_smoke: Tuple[bool, int, str] | None = None

		for attempt in range(1, int(max_attempts) + 1):
			attempt_prompt = build_executor_prompt(
				task_description=spec.task_description,
				subtask_title=node_meta.title,
				subtask_description=node_meta.description,
				repo_snapshot=current_snapshot,
			)
			if last_exc is not None:
				attempt_prompt = (
					attempt_prompt
					+ "\n\nYour previous output failed validation/smoke test with error:\n"
					+ str(last_exc)
					+ "\n\nRepair checklist:\n"
					+ (last_hint or "- Regenerate output strictly following constraints.")
					+ "\n\nNow regenerate the FULL repository meeting ALL constraints.\n"
				)
			_write_text(layer_dir / f"prompt_attempt_{attempt}.txt", attempt_prompt)

			try:
				prompt_tokens = _count_tokens(tokenizer, attempt_prompt)
				call_start_ms = _time_ms()
				response, _, _ = mole_gen.generate_with_experts(prompt=attempt_prompt, expert_ids=expert_ids)
				call_end_ms = _time_ms()
				completion_tokens = _count_tokens(tokenizer, response)
				_write_text(layer_dir / f"response_attempt_{attempt}.txt", response)
				candidate = Codes(response)
				last_candidate = candidate
				validate_codes(
					candidate,
					allow_pass_todo=allow_pass_todo,
					allow_format_placeholders=allow_format_placeholders,
				)
				smoke_ok, smoke_ms, smoke_details = _run_smoke_test(
					codes=candidate,
					layer_dir=layer_dir,
					stage="exec",
					attempt=attempt,
					timeout_s=int(smoke_timeout_s),
				)
				last_smoke = (smoke_ok, smoke_ms, smoke_details)

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
				sample_calls.append(
					{
						"ts": _now_ts(),
						"node_id": node_id,
						"kind": "smoke",
						"stage": "exec",
						"attempt": attempt,
						"elapsed_ms": smoke_ms,
						"passed": smoke_ok,
						"details": smoke_details,
					}
				)
				if not smoke_ok:
					raise RuntimeError(f"Smoke test failed for node {node_id} attempt {attempt}: {smoke_details}")
				codes = candidate
				last_exc = None
				break
			except Exception as exc:
				last_exc = exc
				last_hint = _summarize_validation_hints(last_candidate or Codes(""))
				if "did not contain any parseable code blocks/files" in str(exc).lower():
					last_hint = (
						(last_hint + "\n" if last_hint else "")
						+ "- Output must be a sequence of files. Each file must start with '<name>.py' on its own line, "
						+ "followed by a fenced python code block."
					)
				_write_text(layer_dir / f"error_attempt_{attempt}.txt", str(exc))

		if last_exc is not None or codes is None:
			_write_text(layer_dir / "node_failure.txt", f"Node generation failed after {int(max_attempts)} attempts.\nError: {last_exc}\n")
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
		sample_nodes.append(
			{
				"node_id": node_id,
				"title": node_meta.title,
				"allow_pass_todo": allow_pass_todo,
				"allow_format_placeholders": allow_format_placeholders,
				"parent_count": len(parent_ids),
				"failed": bool(last_exc),
				"smoke_passed": (last_smoke[0] if last_smoke is not None else None),
				"smoke_details": (last_smoke[2] if last_smoke is not None else None),
				"smoke_elapsed_ms": (last_smoke[1] if last_smoke is not None else None),
			}
		)

	final_node = max(solutions) if solutions else None
	if final_node is not None:
		solutions[final_node].write_to_directory(repo_dir)

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
	_write_json(log_dir / "sample_metrics.json", sample_metrics)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="MoLE inference-only SRDD TaskGraph runner (scratch-friendly, resume + forward/reverse order).")
	p.add_argument("--srdd-csv", type=Path, default=REPO_ROOT / "srdd" / "data" / "SRDD.csv")
	p.add_argument("--sample-name", type=str, default=None)
	p.add_argument("--per-category", type=int, default=1000000)
	p.add_argument("--per-category-offset", type=int, default=0)
	p.add_argument("--order", type=str, default="forward", choices=["forward", "reverse"])
	p.add_argument("--category-parity", type=str, default="none", choices=["none", "even", "odd"])
	p.add_argument("--category-shards", type=int, default=1, help="Split categories into N shards by idx%N. Requires --category-parity none.")
	p.add_argument("--category-shard-idx", type=int, default=0, help="Which shard to run (0..N-1). Requires --category-parity none.")
	p.add_argument("--skip-existing", action="store_true")

	p.add_argument("--output-root", type=Path, required=True, help="Actual output root (recommended: scratch).")
	p.add_argument("--link-root", type=Path, default=None, help="Create a symlink under this dir pointing to --output-root.")
	p.add_argument("--timestamp", type=str, default=None, help="Fixed timestamp folder (for resume).")

	# Graph model.
	p.add_argument("--graph-model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
	p.add_argument("--graph-max-new-tokens", type=int, default=2048)
	p.add_argument("--graph-temperature", type=float, default=0.0)
	p.add_argument("--graph-device-map", type=str, default="none")
	p.add_argument("--graph-device", type=int, default=-1, help="Graph model device for transformers.pipeline (-1=CPU).")
	p.add_argument("--graph-torch-dtype", type=str, default="bfloat16")
	p.add_argument("--graph-retries", type=int, default=3)

	# MoLE execute model + checkpoint.
	p.add_argument("--checkpoint-dir", type=Path, required=False, default=None, help="Checkpoint dir containing router.pt/title_embedder.pt/lora_state.pt. Omit with --no-ckpt-load for base-model baseline.")
	p.add_argument("--no-ckpt-load", action="store_true", help="Baseline mode: skip router/title_embedder/lora_state loading. LoRA B is zero-init so output==base model; router picks random experts but contributes 0.")
	p.add_argument("--exec-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
	p.add_argument("--torch-dtype", type=str, default="bfloat16")
	p.add_argument("--device", type=int, default=0, help="Execute model device index within CUDA_VISIBLE_DEVICES.")
	p.add_argument("--num-subtask-experts", type=int, default=6)
	p.add_argument("--subtask-top-k", type=int, default=2)
	p.add_argument("--max-new-tokens", type=int, default=12288)
	p.add_argument("--lora-rank", type=int, default=8)
	p.add_argument("--lora-alpha", type=float, default=16.0)
	p.add_argument("--lora-last-n-layers", type=int, default=8)
	p.add_argument("--router-mode", type=str, default="greedy", choices=["greedy", "sample"])
	p.add_argument("--max-attempts", type=int, default=3)
	p.add_argument("--smoke-timeout-s", type=int, default=3)

	# Concurrency locks.
	p.add_argument("--lock-stale-seconds", type=int, default=21600, help="Treat in_progress.lock as stale after this many seconds.")
	p.add_argument("--summary-lock-stale-seconds", type=int, default=600)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	random.seed(0)
	torch.manual_seed(0)

	output_root = Path(args.output_root).resolve()
	output_root.mkdir(parents=True, exist_ok=True)
	if args.link_root is not None:
		_ensure_symlink_dir(link_root=Path(args.link_root).resolve(), target_dir=output_root)

	timestamp = (args.timestamp or "").strip() or time.strftime("%Y%m%d_%H%M%S")
	timestamp_root = output_root / timestamp
	timestamp_root.mkdir(parents=True, exist_ok=True)

	graph_device_map = _as_device_map(args.graph_device_map)
	dtype_kwargs: Dict[str, object] = {}
	try:
		if args.graph_torch_dtype and hasattr(torch, str(args.graph_torch_dtype)):
			dtype_kwargs["torch_dtype"] = getattr(torch, str(args.graph_torch_dtype))
	except Exception:
		dtype_kwargs = {}

	graph_generator = LLMTaskGraphGenerator(
		model_name=args.graph_model_name,
		max_new_tokens=int(args.graph_max_new_tokens),
		temperature=float(args.graph_temperature),
		device=int(args.graph_device),
		device_map=graph_device_map,
		pipeline_task="text-generation",
		model_kwargs=dtype_kwargs,
	)

	device = torch.device("cuda", int(args.device)) if torch.cuda.is_available() else torch.device("cpu")
	num_role_experts = len(ROLE_EXPERT_IDS)
	num_subtask_experts = int(args.num_subtask_experts)
	if num_subtask_experts <= 0:
		raise ValueError("--num-subtask-experts must be > 0.")
	subtask_expert_offset = int(num_role_experts)
	mole_gen, router, title_embedder, tokenizer = _load_mole_from_checkpoint(
		checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir is not None else Path("."),
		exec_model_name=str(args.exec_model_name),
		torch_dtype=str(args.torch_dtype),
		device=device,
		num_role_experts=num_role_experts,
		num_subtask_experts=num_subtask_experts,
		subtask_top_k=int(args.subtask_top_k),
		lora_rank=int(args.lora_rank),
		lora_alpha=float(args.lora_alpha),
		lora_last_n_layers=int(args.lora_last_n_layers),
		max_new_tokens=int(args.max_new_tokens),
		no_ckpt_load=bool(args.no_ckpt_load),
	)

	samples = read_srdd_samples(Path(args.srdd_csv))
	if args.sample_name:
		selected = [s for s in samples if s.name == args.sample_name]
		if not selected:
			raise ValueError(f"Sample '{args.sample_name}' not found in {args.srdd_csv}")
	else:
		parity = str(args.category_parity)
		shards = int(args.category_shards)
		shard_idx = int(args.category_shard_idx)
		if parity.strip().lower() != "none":
			if shards != 1 or shard_idx != 0:
				raise ValueError("--category-shards/--category-shard-idx require --category-parity none.")
			selected = iter_slice_by_category_parity(
				samples,
				offset=int(args.per_category_offset),
				limit=int(args.per_category),
				parity=parity,
			)
		elif shards > 1:
			selected = iter_slice_by_category_shard(
				samples,
				offset=int(args.per_category_offset),
				limit=int(args.per_category),
				shards=shards,
				shard_idx=shard_idx,
			)
		else:
			selected = iter_slice_by_category_parity(
				samples,
				offset=int(args.per_category_offset),
				limit=int(args.per_category),
				parity="none",
			)
	if str(args.order).strip().lower() == "reverse":
		selected = list(reversed(selected))

	for sample in selected:
		category_dir = timestamp_root / sanitize(sample.category)
		sample_dir = category_dir / sanitize(sample.name)
		log_dir = sample_dir / "log"
		repo_dir = sample_dir / "repo"
		sample_dir.mkdir(parents=True, exist_ok=True)
		category_dir.mkdir(parents=True, exist_ok=True)

		if bool(args.skip_existing) and (sample_dir / "done.txt").exists():
			continue

		lock_path = sample_dir / "in_progress.lock"
		if not _try_acquire_lock(lock_path, stale_seconds=int(args.lock_stale_seconds)):
			continue
		try:
			# 1) Graph generation.
			graph_path = sample_dir / "task_graph.json"
			graph_metrics_path = sample_dir / "graph_generation.json"
			last_error: Exception | None = None
			# If a pre-populated task_graph.json already exists (e.g. copied from
			# the training-time taskgraph root), skip LLM graph generation.
			_pregen = graph_path.exists() and graph_path.stat().st_size > 0
			if _pregen and not graph_metrics_path.exists():
				_write_json(graph_metrics_path, {"status": "ok", "source": "pre_populated"})
			for attempt in range(1, min(max(int(args.graph_retries), 1), 3) + 1) if not _pregen else []:
				try:
					start = time.time()
					graph = graph_generator.generate(sample.name, sample.description)
					elapsed_ms = int((time.time() - start) * 1000)
					graph.save_json(graph_path)

					prompt = graph_generator.last_prompt or ""
					response = graph_generator.last_response_text or ""
					tokenizer_graph = getattr(getattr(graph_generator, "_pipeline", None), "tokenizer", None)
					prompt_tokens = len(tokenizer_graph.encode(prompt)) if tokenizer_graph is not None else None
					completion_tokens = len(tokenizer_graph.encode(response)) if tokenizer_graph is not None else None
					_write_json(
						graph_metrics_path,
						{
							"status": "ok",
							"attempt": attempt,
							"elapsed_ms": elapsed_ms,
							"prompt_tokens": prompt_tokens,
							"completion_tokens": completion_tokens,
						},
					)
					last_error = None
					break
				except GraphGenerationError as exc:
					last_error = exc
					_write_text(sample_dir / f"graph_error_attempt_{attempt}.txt", str(exc))
					if getattr(graph_generator, "last_response_text", None):
						_write_text(sample_dir / f"graph_response_attempt_{attempt}.txt", graph_generator.last_response_text or "")

			if last_error is not None:
				_write_json(
					graph_metrics_path,
					{
						"status": "failed",
						"error": str(last_error),
						"attempts": min(max(int(args.graph_retries), 1), 3),
					},
				)
				_write_text(sample_dir / "skipped.txt", "Skipped: task graph generation failed after retries.\n")
				continue

			# 2) Execute graph with MoLE.
			spec = convert_taskgraph(graph_path)
			execute_task_graph_mole(
				spec=spec,
				mole_gen=mole_gen,
				router=router,
				title_embedder=title_embedder,
				tokenizer=tokenizer,
				log_dir=log_dir,
				repo_dir=repo_dir,
				sample_name=sample.name,
				router_mode=str(args.router_mode),
				max_attempts=int(args.max_attempts),
				smoke_timeout_s=int(args.smoke_timeout_s),
				subtask_expert_offset=subtask_expert_offset,
			)

			# 3) SRDD eval.
			report_path = sample_dir / "srdd_report.txt"
			try:
				srdd_evaluator.evaluate(sample.name, str(args.srdd_csv), str(repo_dir), str(report_path))
			except Exception as exc:
				_write_text(sample_dir / "srdd_eval_error.txt", str(exc))

			_write_text(sample_dir / "done.txt", "ok\n")
		finally:
			_release_lock(lock_path)

		# Best-effort: update summaries with a coarse lock to avoid concurrent writes.
		summary_lock = timestamp_root / ".summary.lock"
		if _try_acquire_lock(summary_lock, stale_seconds=int(args.summary_lock_stale_seconds)):
			try:
				update_summaries(timestamp_root)
			finally:
				_release_lock(summary_lock)


if __name__ == "__main__":
	main()
