from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.multiprocessing as mp

# REPO_ROOT is the directory that CONTAINS morse/, scicode/, srdd/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from srdd.eval import srdd_evaluator  # noqa: E402

from morse.taskgraph.generator import GraphPlanner  # noqa: E402
from srdd.tomas.codes import Codes  # noqa: E402
from morse.llm.converter import convert_taskgraph  # noqa: E402
from srdd.tomas.executor import (  # noqa: E402
	_summarize_validation_hints as summarize_validation_hints,
	build_aggregate_prompt,
	build_executor_prompt,
	validate_codes,
)
from srdd.tomas.review_test import smoke_test_repo  # noqa: E402

from morse.mole.mole_generator import GenerationConfig, MoLEGenerator  # noqa: E402
from morse.mole.mole_lora import LoRAConfig, inject_mole_lora, lora_parameters  # noqa: E402
from srdd.eval.metrics_logger import MetricsLogger, WandBConfig  # noqa: E402
from srdd.eval.reward import completeness_from_code, compute_reward, consistency_stripped_from_code  # noqa: E402
from morse.mole.roles import Role  # noqa: E402
from morse.mole.router import SubtaskRouter, SubtaskRouterConfig, normalize_title_embedding  # noqa: E402


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


def iter_slice_by_category(samples: Iterable[SRDDSample], *, offset: int, limit: int) -> List[SRDDSample]:
	by_cat: Dict[str, List[SRDDSample]] = {}
	for s in samples:
		by_cat.setdefault(s.category, []).append(s)
	picked: List[SRDDSample] = []
	for cat in sorted(by_cat):
		start = max(offset, 0)
		end = start + max(limit, 0)
		picked.extend(by_cat[cat][start:end])
	return picked


def sanitize(name: str) -> str:
	return "".join([c if c.isalnum() else "_" for c in name]).strip("_") or "item"


ROLE_EXPERT_IDS = {
	Role.EXECUTE: 0,
	Role.AGGREGATE: 1,
}

MERGE_USE_PARENT_EXPERTS: bool = True


def _subtask_text(title: str, description: str) -> str:
	title = (title or "").strip()
	desc = (description or "").strip()
	if title and desc:
		return f"{title}: {desc}"
	return title or desc or "none"


def _record_expert_selection(
	*,
	role_counts: Dict[Role, int],
	subtask_counts: Dict[int, int],
	role: Role,
	expert_ids: torch.Tensor,
	subtask_expert_offset: int,
	num_subtask_experts: int,
) -> None:
	role_counts[role] = int(role_counts.get(role, 0)) + 1
	try:
		ids = [int(x) for x in expert_ids.detach().cpu().tolist()]
	except Exception:
		return
	for eid in ids:
		if eid < int(subtask_expert_offset):
			continue
		idx = int(eid) - int(subtask_expert_offset)
		if 0 <= idx < int(num_subtask_experts):
			subtask_counts[idx] = int(subtask_counts.get(idx, 0)) + 1


def _write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding="utf-8")


def _write_text_best_effort(path: Path, text: str) -> bool:
	"""Write text, but don't crash training on quota/full-disk for debug artifacts."""
	try:
		_write_text(path, text)
		return True
	except OSError as exc:
		if getattr(exc, "errno", None) in (28, 122):  # ENOSPC / EDQUOT
			print(f"[warn] could not write {path} ({exc}); skipping.", file=sys.stderr)
			return False
		raise


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json_best_effort(path: Path) -> Dict[str, Any]:
	if not path.exists():
		return {}
	try:
		out = json.loads(path.read_text(encoding="utf-8"))
		return dict(out) if isinstance(out, dict) else {}
	except Exception:
		return {}


def _load_taskgraph_index(root: Path) -> Dict[Tuple[str, str], Path]:
	index: Dict[Tuple[str, str], Path] = {}
	for sample_path in root.rglob("sample.json"):
		try:
			payload = json.loads(sample_path.read_text(encoding="utf-8"))
		except Exception:
			continue
		if not isinstance(payload, dict):
			continue
		category = str(payload.get("category", "") or "").strip()
		name = str(payload.get("name", "") or "").strip()
		if not category or not name:
			continue
		graph_path = sample_path.parent / "task_graph.json"
		if not graph_path.exists():
			continue
		index[(category, name)] = graph_path
	return index


def _mean(values: List[float]) -> float:
	return sum(values) / float(len(values)) if values else 0.0


@contextmanager
def _file_lock(lock_path: Path):
	try:
		import fcntl  # unix only

		lock_path.parent.mkdir(parents=True, exist_ok=True)
		handle = lock_path.open("w", encoding="utf-8")
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


def _truncate_metrics_jsonl_to_state(*, jsonl_path: Path, attempt_step_max: int, sample_step_max: int) -> None:
	"""Truncate metrics.jsonl to match a checkpointed (attempt_step, sample_step) state.

	This ensures that when resuming from a checkpoint (typically saved at sample boundaries),
	any partially-logged next-sample attempts are discarded so that new logs re-use the same
	`attempt_step` range without duplicates.
	"""
	jsonl_path = Path(jsonl_path)
	if not jsonl_path.exists():
		return
	tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
	with jsonl_path.open("r", encoding="utf-8", errors="ignore") as src, tmp_path.open("w", encoding="utf-8") as dst:
		for line in src:
			if not line.strip():
				continue
			try:
				obj = json.loads(line)
			except Exception:
				# Keep unparseable lines (best-effort), but never past the first truncation point.
				dst.write(line)
				continue
			if not isinstance(obj, dict):
				dst.write(line)
				continue
			typ = obj.get("type")
			if typ == "attempt":
				step = obj.get("attempt_step")
				if isinstance(step, int) and step > int(attempt_step_max):
					break
			elif typ == "sample":
				step = obj.get("sample_step")
				if isinstance(step, int) and step > int(sample_step_max):
					break
			dst.write(line)
	tmp_path.replace(jsonl_path)


def _compute_srdd_means(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
	def _vals(key: str) -> List[float]:
		out: List[float] = []
		for r in rows:
			v = (r.get("srdd", {}) or {}).get(key)
			if isinstance(v, (int, float)):
				out.append(float(v))
		return out

	exec_m = _mean(_vals("executability"))
	comp_m = _mean(_vals("completeness"))
	cons_m = _mean(_vals("consistency"))
	cons_str_m = _mean(_vals("consistency_stripped"))
	cons_emb_m = _mean(_vals("consistency_embedding"))
	cons_emb_str_m = _mean(_vals("consistency_embedding_stripped"))

	srdd_mean: Dict[str, Any] = {
		"executability": exec_m,
		"completeness": comp_m,
		"consistency": cons_m,
		"consistency_stripped": cons_str_m,
		"consistency_embedding": cons_emb_m,
		"consistency_embedding_stripped": cons_emb_str_m,
	}
	srdd_mean["eci_mean"] = (exec_m + comp_m + cons_m) / 3.0
	srdd_mean["eci_product"] = exec_m * comp_m * cons_str_m
	srdd_mean["eci_mean_embedding"] = (exec_m + comp_m + cons_emb_m) / 3.0
	srdd_mean["eci_product_embedding"] = exec_m * comp_m * cons_emb_str_m
	return srdd_mean


def _compute_train_means(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
	def _vals(key: str) -> List[float]:
		out: List[float] = []
		for r in rows:
			v = (r.get("train", {}) or {}).get(key)
			if isinstance(v, (int, float)):
				out.append(float(v))
		return out

	def _bool_rate(key: str) -> float:
		vals: List[float] = []
		for r in rows:
			v = (r.get("train", {}) or {}).get(key)
			if isinstance(v, bool):
				vals.append(1.0 if v else 0.0)
		return _mean(vals)

	return {
		"reward": _mean(_vals("reward")),
		"loss": _mean(_vals("loss")),
		"advantage": _mean(_vals("advantage")),
		"main_gate_pass_rate": _bool_rate("main_gate_pass"),
	}


def _compute_time_means(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
	vals: List[float] = []
	for r in rows:
		v = (r.get("timing", {}) or {}).get("elapsed_ms")
		if isinstance(v, (int, float)):
			vals.append(float(v))
	return {"elapsed_ms": _mean(vals)}


def _collect_completed_samples(run_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
	all_rows: List[Dict[str, Any]] = []
	by_cat: Dict[str, List[Dict[str, Any]]] = {}
	for cat_dir in sorted([p for p in run_root.iterdir() if p.is_dir()]):
		cat = cat_dir.name
		for sample_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
			m = _read_json_best_effort(sample_dir / "train_metrics.json")
			if not m:
				continue
			row = {
				"category": cat,
				"sample": sample_dir.name,
				"srdd": {
					"executability": m.get("executability"),
					"completeness": m.get("completeness"),
					"consistency": m.get("consistency_srdd"),
					"consistency_stripped": m.get("consistency_stripped_srdd"),
					"consistency_embedding": m.get("consistency_embedding"),
					"consistency_embedding_stripped": m.get("consistency_embedding_stripped"),
				},
				"train": {
					"reward": m.get("reward"),
					"loss": m.get("loss"),
					"advantage": m.get("advantage"),
					"main_gate_pass": m.get("main_gate_pass"),
				},
				"timing": {"elapsed_ms": m.get("elapsed_ms")},
				"paths": {
					"sample_dir": str(sample_dir),
					"repo_dir": str(sample_dir / "repo"),
					"log_dir": str(sample_dir / "log"),
				},
			}
			all_rows.append(row)
			by_cat.setdefault(cat, []).append(row)
	return all_rows, by_cat


def _update_training_summaries(run_root: Path) -> Dict[str, Any]:
	"""Update run-level `summary.json`/`category_summary.json` like inference does."""
	run_root = Path(run_root)
	with _file_lock(run_root / ".summary.lock"):
		all_rows, by_cat = _collect_completed_samples(run_root)
		overall = {
			"count": len(all_rows),
			"srdd_mean": _compute_srdd_means(all_rows),
			"train_mean": _compute_train_means(all_rows),
			"time_mean": _compute_time_means(all_rows),
		}
		summary = {
			"timestamp_root": str(run_root),
			"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
			"overall": overall,
			"categories": {
				cat: {
					"count": len(rows),
					"srdd_mean": _compute_srdd_means(rows),
					"train_mean": _compute_train_means(rows),
					"time_mean": _compute_time_means(rows),
				}
				for cat, rows in by_cat.items()
			},
		}
		(run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

		for cat, rows in by_cat.items():
			cat_dir = run_root / cat
			payload = {
				"category": cat,
				"updated_at": summary["updated_at"],
				"stats": {
					"count": len(rows),
					"srdd_mean": _compute_srdd_means(rows),
					"train_mean": _compute_train_means(rows),
					"time_mean": _compute_time_means(rows),
				},
			}
			(cat_dir / "category_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		return summary


def _repo_text(codes: Codes) -> str:
	parts = []
	for filename in sorted(codes.codebooks):
		parts.append(codes.codebooks[filename])
	return "\n".join(parts)


def _proxy_reward(
	*,
	task_description: str,
	subtask_description: str,
	codes: Codes,
	parse_ok: bool,
	validate_ok: bool,
	smoke_ok: bool,
	w_smoke: float,
	w_comp: float,
	w_cons_strip: float,
	w_cons_task: float,
	w_cons_subtask: float,
) -> Tuple[float, Dict[str, Any]]:
	if not parse_ok or not validate_ok:
		return 0.0, {
			"parse_ok": bool(parse_ok),
			"validate_ok": bool(validate_ok),
			"smoke_ok": bool(smoke_ok),
			"proxy_completeness": 0.0,
			"proxy_consistency_task_stripped": 0.0,
			"proxy_consistency_subtask_stripped": 0.0,
			"proxy_consistency_stripped": 0.0,
			"proxy_reward": 0.0,
		}

	code_text = _repo_text(codes)
	comp = float(completeness_from_code(code_text))
	cons_task = float(consistency_stripped_from_code(task_description, code_text))
	cons_subtask = float(consistency_stripped_from_code(subtask_description, code_text))
	cons_mix = float(w_cons_task) * float(cons_task) + float(w_cons_subtask) * float(cons_subtask)
	r = float(w_smoke) * float(bool(smoke_ok)) + float(w_comp) * comp + float(w_cons_strip) * cons_mix
	return float(r), {
		"parse_ok": bool(parse_ok),
		"validate_ok": bool(validate_ok),
		"smoke_ok": bool(smoke_ok),
		"proxy_completeness": float(comp),
		"proxy_consistency_task_stripped": float(cons_task),
		"proxy_consistency_subtask_stripped": float(cons_subtask),
		"proxy_consistency_stripped": float(cons_mix),
		"proxy_reward": float(r),
	}


@dataclass
class GRPOCandidate:
	idx: int
	text: str
	codes: Codes
	prompt_ids: torch.Tensor
	gen_ids: torch.Tensor
	parse_ok: bool
	validate_ok: bool
	smoke_ok: bool
	error: Exception | None
	proxy_reward: float
	proxy_metrics: Dict[str, Any]
	logp_mole: torch.Tensor | None


def _safe_std(values: List[float]) -> float:
	if not values:
		return 0.0
	mean = sum(values) / float(len(values))
	var = sum((x - mean) ** 2 for x in values) / float(len(values))
	return float(math.sqrt(var))


def _clip_advantage(adv: float, clip: float) -> float:
	"""Clip advantage magnitude to stabilize policy-gradient updates."""
	try:
		c = float(clip)
	except Exception:
		return float(adv)
	if c <= 0.0:
		return float(adv)
	if adv > c:
		return float(c)
	if adv < -c:
		return float(-c)
	return float(adv)


def _subtask_router_reg_loss(
	router: nn.Module, l2_weight: float, ortho_weight: float
) -> torch.Tensor:
	if not isinstance(router, SubtaskRouter):
		return torch.tensor(0.0)
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
	return float(l2_weight) * l2 + float(ortho_weight) * ortho


def _shared_lora_state_from_model(model: nn.Module) -> Dict[str, torch.Tensor]:
	"""Create shared-memory CPU tensors mirroring trainable LoRA params."""
	state: Dict[str, torch.Tensor] = {}
	for name, p in model.named_parameters():
		if not p.requires_grad:
			continue
		t = p.detach().to(device="cpu")
		state[name] = t.clone().share_memory_()
	return state


def _refresh_shared_lora_state(shared: Dict[str, torch.Tensor], model: nn.Module) -> None:
	"""Copy current trainable LoRA params from model to shared CPU tensors."""
	for name, p in model.named_parameters():
		if not p.requires_grad:
			continue
		dst = shared.get(name)
		if dst is None:
			continue
		dst.copy_(p.detach().to(device="cpu"))


def _load_shared_lora_state_into_model(shared: Dict[str, torch.Tensor], model: nn.Module, device: torch.device) -> None:
	"""Copy shared CPU LoRA params into model parameters (on device)."""
	name_to_param = dict(model.named_parameters())
	for name, src in shared.items():
		p = name_to_param.get(name)
		if p is None:
			continue
		p.data.copy_(src.to(device=p.device, dtype=p.dtype))


def _parse_device_list(s: str) -> List[int]:
	raw = (s or "").strip()
	if raw == "":
		return []
	out: List[int] = []
	for part in raw.split(","):
		part = part.strip()
		if part == "":
			continue
		out.append(int(part))
	return out


def _gen_worker_loop(
	*,
	worker_device: int,
	model_name: str,
	torch_dtype: str,
	gen_cfg: GenerationConfig,
	lora_cfg: LoRAConfig,
	shared_lora: Dict[str, torch.Tensor],
	task_q: mp.Queue,
	result_q: mp.Queue,
) -> None:
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	device = torch.device("cuda", int(worker_device)) if torch.cuda.is_available() else torch.device("cpu")
	model, tokenizer = _load_backbone(model_name=model_name, torch_dtype=torch_dtype, device=device)
	inject_mole_lora(model, cfg=lora_cfg)
	_load_shared_lora_state_into_model(shared_lora, model, device)

	worker_gen_cfg = GenerationConfig(
		model_name=str(gen_cfg.model_name),
		max_new_tokens=int(gen_cfg.max_new_tokens),
		temperature=float(gen_cfg.temperature),
		top_p=float(gen_cfg.top_p),
		torch_dtype=str(gen_cfg.torch_dtype),
		device=int(worker_device),
	)
	mole_gen = MoLEGenerator(model=model, tokenizer=tokenizer, device=device, gen_cfg=worker_gen_cfg)

	while True:
		msg = task_q.get()
		if not isinstance(msg, dict):
			continue
		if msg.get("type") == "stop":
			break
		if msg.get("type") == "generate":
			# Ensure worker uses latest LoRA before generating.
			_load_shared_lora_state_into_model(shared_lora, model, device)
			prompt = str(msg["prompt"])
			expert_ids = torch.tensor(msg["expert_ids"], device=device, dtype=torch.long)
			num = int(msg.get("num_samples", 1))
			cand_start = int(msg.get("candidate_start", 1))
			seed = msg.get("seed")
			if seed is not None:
				try:
					torch.manual_seed(int(seed))
				except Exception:
					pass
			outs = mole_gen.generate_n_with_experts(prompt=prompt, expert_ids=expert_ids, num_samples=num)
			samples: List[Dict[str, Any]] = []
			for i, (text, prompt_ids, gen_ids) in enumerate(outs):
				cidx = cand_start + i
				samples.append(
					{
						"candidate_idx": int(cidx),
						"text": text,
						"prompt_ids": prompt_ids.detach().cpu().tolist(),
						"gen_ids": gen_ids.detach().cpu().tolist(),
					}
				)
			result_q.put({"type": "generated", "samples": samples, "worker_device": int(worker_device)})


class GenWorkerPool:
	def __init__(
		self,
		*,
		worker_devices: List[int],
		model_name: str,
		torch_dtype: str,
		gen_cfg: GenerationConfig,
		lora_cfg: LoRAConfig,
		shared_lora: Dict[str, torch.Tensor],
	) -> None:
		self.worker_devices = list(worker_devices)
		self._ctx = mp.get_context("spawn")
		self._task_queues: Dict[int, mp.Queue] = {}
		self._result_q: mp.Queue = self._ctx.Queue()
		self._procs: List[mp.Process] = []
		for dev in self.worker_devices:
			tq = self._ctx.Queue()
			self._task_queues[int(dev)] = tq
			p = self._ctx.Process(
				target=_gen_worker_loop,
				kwargs={
					"worker_device": int(dev),
					"model_name": str(model_name),
					"torch_dtype": str(torch_dtype),
					"gen_cfg": gen_cfg,
					"lora_cfg": lora_cfg,
					"shared_lora": shared_lora,
					"task_q": tq,
					"result_q": self._result_q,
				},
			)
			p.daemon = True
			p.start()
			self._procs.append(p)

	def close(self) -> None:
		for dev, q in self._task_queues.items():
			try:
				q.put({"type": "stop"})
			except Exception:
				pass
		for p in self._procs:
			try:
				p.join(timeout=5)
			except Exception:
				pass
			if p.is_alive():
				try:
					p.terminate()
				except Exception:
					pass

	def dispatch_generate(
		self,
		*,
		prompt: str,
		expert_ids: List[int],
		num_samples_per_worker: int,
		candidate_start_by_device: Dict[int, int],
		seed: int | None = None,
	) -> None:
		for dev in self.worker_devices:
			cstart = int(candidate_start_by_device[int(dev)])
			self._task_queues[int(dev)].put(
				{
					"type": "generate",
					"prompt": prompt,
					"expert_ids": [int(x) for x in expert_ids],
					"num_samples": int(num_samples_per_worker),
					"candidate_start": int(cstart),
					"seed": int(seed) if seed is not None else None,
				}
			)
		return None

	def collect_generated(self, *, expected_workers: int | None = None) -> List[Dict[str, Any]]:
		expected = int(expected_workers) if expected_workers is not None else len(self.worker_devices)
		collected: List[Dict[str, Any]] = []
		while expected > 0:
			msg = self._result_q.get()
			if isinstance(msg, dict) and msg.get("type") == "generated":
				collected.extend(list(msg.get("samples") or []))
				expected -= 1
		return collected


class TitleEmbedder(nn.Module):
	"""Embed node titles using frozen token embeddings + a small trainable projection."""

	def __init__(self, *, model, tokenizer, out_dim: int):
		super().__init__()
		self._model = model
		self._tokenizer = tokenizer
		hidden = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
		if hidden <= 0:
			raise ValueError("Could not determine base model hidden size for TitleEmbedder.")
		self.proj = nn.Linear(hidden, int(out_dim))

	@torch.no_grad()
	def _mean_token_emb(self, text: str) -> torch.Tensor:
		enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
		input_ids = enc["input_ids"].to(self._model.device)
		emb = self._model.get_input_embeddings()(input_ids)  # [1, T, H]
		mask = (input_ids != self._tokenizer.pad_token_id).float().unsqueeze(-1) if self._tokenizer.pad_token_id is not None else None
		if mask is None:
			return emb.mean(dim=1)
		den = torch.clamp(mask.sum(dim=1), min=1.0)
		return (emb * mask).sum(dim=1) / den

	def forward(self, title: str) -> torch.Tensor:
		mean_emb = self._mean_token_emb(title)  # [1, H], no_grad
		out = self.proj(mean_emb)  # [1, out_dim], trainable proj
		return normalize_title_embedding(out)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train MoLE with fixed role experts + subtask router on SRDD TaskGraph execution.")
	p.add_argument("--srdd-csv", type=Path, default=REPO_ROOT / "srdd" / "data" / "SRDD.csv")
	p.add_argument("--per-category", type=int, default=2)
	p.add_argument("--per-category-offset", type=int, default=0)
	p.add_argument("--output-root", type=Path, default=REPO_ROOT / "srdd" / "runs")
	p.add_argument("--ckpt-root", type=Path, default=REPO_ROOT / "srdd" / "checkpoints")
	p.add_argument("--run-ts", type=str, default="", help="Override run timestamp folder name (for tmux resume scripts).")

	p.add_argument(
		"--save-candidate-artifacts",
		action="store_true",
		help="Persist per-candidate response texts + smoke-test repos/logs (can create huge numbers of files).",
	)
	p.add_argument(
		"--skip-existing-samples",
		action="store_true",
		help="Skip samples that already have train_metrics.json under the run directory (useful for resuming without a checkpoint).",
	)

	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES for training (default: 0).")
	p.add_argument("--device", type=int, default=0)
	p.add_argument("--torch-dtype", type=str, default="bfloat16")

	# Graph generation (fixed; not trained in this MVP).
	p.add_argument("--graph-provider", type=str, default="hf", choices=["hf", "huggingface", "gemini", "heuristic"])
	p.add_argument("--graph-model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
	p.add_argument("--graph-max-new-tokens", type=int, default=2048)
	p.add_argument("--graph-temperature", type=float, default=0.0)
	p.add_argument("--graph-retries", type=int, default=3, help="Retry count for task-graph generation (default: 3).")
	p.add_argument("--graph-device-map", type=str, default="auto")
	p.add_argument("--graph-device", type=int, default=0)
	p.add_argument(
		"--graph-gpus",
		type=str,
		default="",
		help="Comma-separated CUDA device indices (within CUDA_VISIBLE_DEVICES) for graph model sharding; default uses all visible GPUs except --device.",
	)
	p.add_argument(
		"--graph-max-memory-fraction",
		type=float,
		default=0.90,
		help="When --graph-gpus is set (or inferred), allocate at most this fraction of each selected GPU's total memory.",
	)
	p.add_argument("--graph-fallback-only", action="store_true", help="Use heuristic planner only (no LLM).")
	p.add_argument(
		"--taskgraph-root",
		type=Path,
		default=None,
		help="Root directory containing pre-generated SRDD task_graph.json files (skip graph LMs).",
	)

	# Execution / ablations.
	p.add_argument(
		"--disable-aggregate",
		action="store_true",
		help="Ablation: disable MergeAgent at join points (use the first parent snapshot as base).",
	)
	p.add_argument(
		"--merge-use-parent-experts",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="When aggregating multiple parents, include the union of parent subtask experts in the expert set (default: enabled).",
	)

	# Code model (MoLE backbone).
	p.add_argument("--model-name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
	p.add_argument("--max-new-tokens", type=int, default=4096)
	p.add_argument("--temperature", type=float, default=0.2)
	p.add_argument("--top-p", type=float, default=0.95)

	# MoLE / LoRA.
	p.add_argument(
		"--num-subtask-experts",
		type=int,
		default=4,
		help="Number of subtask experts (total experts = 2 role experts + this value).",
	)
	p.add_argument("--subtask-top-k", type=int, default=2, help="Number of subtask experts to select per node.")
	p.add_argument("--subtask-proto-l2", type=float, default=0.0, help="L2 regularization weight for subtask prototypes.")
	p.add_argument("--subtask-proto-ortho", type=float, default=0.0, help="Orthogonality regularization weight for subtask prototypes.")
	p.add_argument("--lora-rank", type=int, default=8)
	p.add_argument("--lora-alpha", type=float, default=16.0)
	p.add_argument("--lora-last-n-layers", type=int, default=8)

	# Training.
	p.add_argument("--router-lr", type=float, default=1e-4)
	p.add_argument("--lora-lr", type=float, default=2e-4)
	p.add_argument("--alpha-router", type=float, default=0.1, help="Weight for router logp term in RL loss.")
	p.add_argument("--baseline-momentum", type=float, default=0.9)
	p.add_argument("--max-attempts", type=int, default=3, help="Retry count per generation call when validation fails.")
	p.add_argument("--smoke-timeout-s", type=int, default=10, help="Smoke-test timeout for main.py (match inference default: 10s).")
	p.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N samples.")
	p.add_argument("--save-at-end", action="store_true", help="Save a checkpoint before exiting.")

	# Per-attempt RL (optional): update on each generation attempt using a cheap proxy reward.
	p.add_argument(
		"--update-per-attempt",
		action="store_true",
		help="Update Router/LoRA after every generation attempt using proxy reward; sample end still evaluates SRDD but does not update.",
	)
	p.add_argument("--attempt-reward", type=str, default="proxy", choices=["proxy"])
	p.add_argument("--proxy-w-smoke", type=float, default=0.5)
	p.add_argument("--proxy-w-comp", type=float, default=0.5)
	p.add_argument("--proxy-w-cons-strip", type=float, default=1.0)
	p.add_argument("--proxy-cons-task-weight", type=float, default=0.7)
	p.add_argument("--proxy-cons-subtask-weight", type=float, default=0.3)
	p.add_argument(
		"--reset-sample-cursor",
		action="store_true",
		help="When resuming, ignore the saved sample_cursor (useful when changing per-category slices).",
	)

	# GRPO (grouped policy gradient) for per-attempt updates.
	p.add_argument("--use-grpo", action="store_true", help="Use GRPO-style group advantages for per-attempt updates.")
	p.add_argument("--grpo-group-size", type=int, default=8, help="GRPO group size per attempt.")
	p.add_argument("--grpo-group-size-max", type=int, default=8, help="Max GRPO group size when adaptive expansion triggers.")
	p.add_argument("--grpo-adaptive", action="store_true", help="Expand group size up to --grpo-group-size-max if all rewards are zero.")
	p.add_argument(
		"--grpo-skip-update-if-allzero",
		action="store_true",
		help="Skip optimizer update if group rewards are all equal (e.g., all zero).",
	)
	p.add_argument("--grpo-adv-normalize", action="store_true", help="Normalize group advantages by (std + eps).")
	p.add_argument("--grpo-adv-eps", type=float, default=1e-6)
	p.add_argument(
		"--advantage-clip",
		type=float,
		default=5.0,
		help="Clip advantages to [-A, A] for stability (0 disables). Applies to both GRPO and non-GRPO updates.",
	)

	# Parallel generation (multi-GPU) for GRPO groups.
	p.add_argument("--gen-parallel", action="store_true", help="Parallelize candidate generation across multiple GPUs (main + workers).")
	p.add_argument("--gen-worker-devices", type=str, default="1,2,3", help="Comma-separated worker device indices (within CUDA_VISIBLE_DEVICES).")
	p.add_argument("--gen-batch-per-device", type=int, default=2, help="Number of candidates to generate per worker GPU per attempt.")
	p.add_argument("--gen-main-batch", type=int, default=2, help="Number of candidates to generate on the main training GPU per attempt.")

	# Experiment tracking.
	p.add_argument(
		"--wandb",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Log metrics to Weights & Biases (default: enabled; falls back to JSONL if wandb is unavailable).",
	)
	p.add_argument("--wandb-project", type=str, default="srdd")
	p.add_argument("--wandb-entity", type=str, default="")
	p.add_argument("--wandb-run-name", type=str, default="")
	p.add_argument("--wandb-mode", type=str, default="offline", choices=["offline", "online"])

	# Checkpointing / resume.
	p.add_argument("--resume-from", type=Path, default=None, help="Resume from a checkpoint dir (e.g., srdd/checkpoints/<ts>/<step_dir>).")
	p.add_argument("--checkpoint-overwrite", action="store_true", help="Always overwrite a single checkpoint dir instead of creating step_xxxxxx dirs.")
	p.add_argument("--checkpoint-name", type=str, default="step_latest", help="Checkpoint directory name under srdd/checkpoints/<ts>/ when --checkpoint-overwrite is set.")
	return p.parse_args()


def _seed_everything(seed: int) -> None:
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _load_backbone(*, model_name: str, torch_dtype: str, device: torch.device):
	from transformers import AutoModelForCausalLM, AutoTokenizer

	tok = AutoTokenizer.from_pretrained(model_name)
	if tok.pad_token_id is None:
		tok.pad_token = tok.eos_token
	dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.bfloat16
	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
	model.to(device)
	# Match inference behavior: disable dropout for more stable formatting, while still allowing grads for LoRA.
	model.eval()
	# Reduce training-time memory. We'll explicitly enable caching during `generate(...)` calls.
	try:
		if getattr(model, "config", None) is not None:
			model.config.use_cache = False  # type: ignore[attr-defined]
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


def _graph_planner(args: argparse.Namespace) -> GraphPlanner:
	if args.graph_provider == "heuristic" or args.graph_fallback_only:
		return GraphPlanner(fallback_only=True)
	provider = "gemini" if args.graph_provider == "gemini" else "hf"
	device_map = (args.graph_device_map or "").strip()
	device_map = None if device_map.lower() in {"", "none"} else device_map

	model_kwargs: Dict[str, Any] = {}
	# Keep graph model off the execute GPU by using max_memory to exclude it.
	if torch.cuda.is_available():
		visible = torch.cuda.device_count()
		graph_gpus_raw = (args.graph_gpus or "").strip()
		if graph_gpus_raw:
			graph_gpus = [int(x) for x in graph_gpus_raw.split(",") if x.strip() != ""]
		else:
			graph_gpus = [i for i in range(visible) if i != int(args.device)]
		graph_gpus = [i for i in graph_gpus if 0 <= i < visible]
		if graph_gpus:
			max_mem: Dict[Any, Any] = {}
			for i in range(visible):
				if i in graph_gpus:
					total = int(torch.cuda.get_device_properties(i).total_memory)
					allowed = max(0, int(float(args.graph_max_memory_fraction) * float(total)))
					max_mem[i] = allowed
				else:
					max_mem[i] = 0
			model_kwargs["max_memory"] = max_mem

	dtype = getattr(torch, str(args.torch_dtype), None)
	if dtype is not None:
		model_kwargs["torch_dtype"] = dtype

	return GraphPlanner(
		provider=provider,
		model_name=args.graph_model_name,
		max_new_tokens=int(args.graph_max_new_tokens),
		temperature=float(args.graph_temperature),
		model_kwargs=model_kwargs or None,
		device=int(args.graph_device),
		device_map=device_map,
		allow_fallback=True,
	)


def _build_predecessors(edge_strings: List[str], node_ids: List[int]) -> Dict[int, List[int]]:
	predecessors: Dict[int, List[int]] = {nid: [] for nid in node_ids}
	for edge in edge_strings:
		src, dst = edge.split("->", 1)
		predecessors[int(dst)].append(int(src))
	return predecessors


def _select_experts(
	*,
	router: SubtaskRouter,
	title_embedder: TitleEmbedder,
	device: torch.device,
	role: Role,
	subtask_text: str,
	subtask_expert_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
	role_expert_id = ROLE_EXPERT_IDS.get(role)
	if role_expert_id is None:
		raise ValueError(f"Unknown role for expert routing: {role}")
	title_emb = title_embedder(subtask_text)
	logits = router(title_emb=title_emb)
	subtask_ids, logp_router = router.sample_topk(logits)
	subtask_ids = subtask_ids + int(subtask_expert_offset)
	role_tensor = torch.tensor([int(role_expert_id)], device=device, dtype=torch.long)
	expert_ids = torch.cat([role_tensor, subtask_ids.to(device=device)])
	return expert_ids, logp_router


def _select_aggregate_experts(
	*,
	device: torch.device,
	parent_subtask_experts: List[int],
	subtask_expert_offset: int,
	merge_use_parent_experts: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
	if merge_use_parent_experts is None:
		merge_use_parent_experts = MERGE_USE_PARENT_EXPERTS
	if not merge_use_parent_experts:
		parent_subtask_experts = []
	role_expert_id = ROLE_EXPERT_IDS.get(Role.AGGREGATE)
	if role_expert_id is None:
		raise ValueError(f"Unknown role for expert routing: {Role.AGGREGATE}")
	role_tensor = torch.tensor([int(role_expert_id)], device=device, dtype=torch.long)
	logp_router = torch.tensor(0.0, device=device, requires_grad=True)
	if not parent_subtask_experts:
		return role_tensor, logp_router
	subtask_ids = torch.tensor(
		[int(subtask_expert_offset) + int(idx) for idx in parent_subtask_experts],
		device=device,
		dtype=torch.long,
	)
	expert_ids = torch.cat([role_tensor, subtask_ids])
	return expert_ids, logp_router


def _run_smoke_test(
	*,
	codes: Codes,
	layer_dir: Path,
	stage: str,
	attempt: int,
	timeout_s: int,
	candidate_idx: int | None = None,
	save_candidate_artifacts: bool = False,
) -> None:
	"""Mirror inference: write repo attempt dir, run main.py smoke test, raise on failure."""
	import shutil

	if candidate_idx is None:
		repo_dir = layer_dir / f"{stage}_repo_attempt_{attempt}"
		if repo_dir.exists():
			shutil.rmtree(repo_dir)
		codes.write_to_directory(repo_dir)
	else:
		# Avoid creating many per-candidate repos by default (can hit inode/disk quotas).
		repo_dir = layer_dir / f"{stage}_repo_tmp"
		repo_dir.mkdir(parents=True, exist_ok=True)
		codes.write_to_directory(repo_dir)

	result = smoke_test_repo(repo_dir, timeout_s=int(timeout_s))
	if candidate_idx is None:
		smoke_log = layer_dir / f"{stage}_smoke_attempt_{attempt}.txt"
		_write_text(
			smoke_log,
			f"passed: {result.passed}\nelapsed_ms: {result.elapsed_ms}\ndetails:\n{result.details}\n",
		)
	elif save_candidate_artifacts:
		# Optional: persist per-candidate repo + smoke log for debugging.
		persist_dir = layer_dir / f"{stage}_repo_attempt_{attempt}_cand_{int(candidate_idx)}"
		if persist_dir.exists():
			shutil.rmtree(persist_dir)
		shutil.copytree(repo_dir, persist_dir)
		smoke_log = layer_dir / f"{stage}_smoke_attempt_{attempt}_cand_{int(candidate_idx)}.txt"
		_write_text_best_effort(
			smoke_log,
			f"passed: {result.passed}\nelapsed_ms: {result.elapsed_ms}\ndetails:\n{result.details}\n",
		)
	if not result.passed:
		raise RuntimeError(f"Smoke test failed ({stage}) attempt {attempt}: {result.details}")


def _sample_grpo_candidate(
	*,
	stage: str,
	candidate_idx: int,
	prompt: str,
	mole_gen: MoLEGenerator,
	expert_ids: torch.Tensor,
	layer_dir: Path,
	attempt: int,
	task_description: str,
	subtask_description: str,
	allow_pass_todo: bool,
	allow_format_placeholders: bool,
	args: argparse.Namespace,
	generated: Dict[str, Any] | None = None,
) -> GRPOCandidate:
	if generated is None:
		text, prompt_ids, gen_ids = mole_gen.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
	else:
		text = str(generated.get("text", ""))
		prompt_ids = torch.tensor(list(generated.get("prompt_ids") or []), dtype=torch.long, device=mole_gen.device)
		gen_ids = torch.tensor(list(generated.get("gen_ids") or []), dtype=torch.long, device=mole_gen.device)

	if bool(getattr(args, "save_candidate_artifacts", False)):
		if stage == "aggregate":
			_write_text_best_effort(layer_dir / f"aggregate_response_attempt_{attempt}_cand_{int(candidate_idx)}.txt", text)
		else:
			_write_text_best_effort(layer_dir / f"response_attempt_{attempt}_cand_{int(candidate_idx)}.txt", text)

	codes = Codes(text)
	parse_ok = bool(codes.codebooks)
	validate_ok = False
	smoke_ok = False
	error: Exception | None = None

	try:
			validate_codes(
				codes,
				require_main=True,
				stdlib_only=True,
				allow_pass_todo=allow_pass_todo,
				allow_format_placeholders=allow_format_placeholders,
			)
			validate_ok = True
			_run_smoke_test(
				codes=codes,
				layer_dir=layer_dir,
				stage=stage,
				attempt=attempt,
				timeout_s=int(args.smoke_timeout_s),
				candidate_idx=int(candidate_idx),
				save_candidate_artifacts=bool(getattr(args, "save_candidate_artifacts", False)),
			)
			smoke_ok = True
	except Exception as exc:
		error = exc

	r_proxy, proxy_metrics = _proxy_reward(
		task_description=task_description,
		subtask_description=subtask_description,
		codes=codes,
		parse_ok=parse_ok,
		validate_ok=validate_ok,
		smoke_ok=smoke_ok,
		w_smoke=float(args.proxy_w_smoke),
		w_comp=float(args.proxy_w_comp),
		w_cons_strip=float(args.proxy_w_cons_strip),
		w_cons_task=float(args.proxy_cons_task_weight),
		w_cons_subtask=float(args.proxy_cons_subtask_weight),
	)

	return GRPOCandidate(
		idx=int(candidate_idx),
		text=text,
		codes=codes,
		prompt_ids=prompt_ids,
		gen_ids=gen_ids,
		parse_ok=bool(parse_ok),
		validate_ok=bool(validate_ok),
		smoke_ok=bool(smoke_ok),
		error=error,
		proxy_reward=float(r_proxy),
		proxy_metrics=proxy_metrics,
		logp_mole=None,
	)


def _select_grpo_candidate(cands: List[GRPOCandidate]) -> GRPOCandidate:
	if not cands:
		raise ValueError("No GRPO candidates to select from.")
	passed = [c for c in cands if c.error is None and c.validate_ok and c.smoke_ok]
	if passed:
		return max(passed, key=lambda c: float(c.proxy_reward))
	parsable = [c for c in cands if c.parse_ok]
	if parsable:
		return max(parsable, key=lambda c: float(c.proxy_reward))
	return cands[0]

def _move_optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
	for state in opt.state.values():
		for k, v in list(state.items()):
			if torch.is_tensor(v):
				state[k] = v.to(device=device)


def _load_checkpoint_dir(path: Path) -> Path:
	p = Path(path)
	if p.is_file():
		raise ValueError(f"--resume-from must be a directory, got file: {p}")
	if not p.exists():
		raise FileNotFoundError(f"Checkpoint directory not found: {p}")
	return p


def _save_checkpoint(
	*,
	ckpt_dir: Path,
	router: nn.Module,
	title_embedder: nn.Module,
	model: nn.Module,
	opt_router: torch.optim.Optimizer,
	opt_lora: torch.optim.Optimizer,
	trainer_state: Dict[str, Any],
) -> None:
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	torch.save(router.state_dict(), ckpt_dir / "router.pt")
	torch.save(title_embedder.state_dict(), ckpt_dir / "title_embedder.pt")
	lora_state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
	torch.save(lora_state, ckpt_dir / "lora_state.pt")
	torch.save(opt_router.state_dict(), ckpt_dir / "opt_router.pt")
	torch.save(opt_lora.state_dict(), ckpt_dir / "opt_lora.pt")
	torch.save(trainer_state, ckpt_dir / "trainer_state.pt")


def _load_checkpoint(
	*,
	ckpt_dir: Path,
	device: torch.device,
	router: nn.Module,
	title_embedder: nn.Module,
	model: nn.Module,
	opt_router: torch.optim.Optimizer,
	opt_lora: torch.optim.Optimizer,
) -> Dict[str, Any]:
	router.load_state_dict(torch.load(ckpt_dir / "router.pt", map_location=device))
	title_embedder.load_state_dict(torch.load(ckpt_dir / "title_embedder.pt", map_location=device))
	lora_state = torch.load(ckpt_dir / "lora_state.pt", map_location="cpu")
	name_to_param = dict(model.named_parameters())
	for name, tensor in lora_state.items():
		p = name_to_param.get(name)
		if p is None:
			continue
		p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))

	opt_router.load_state_dict(torch.load(ckpt_dir / "opt_router.pt", map_location="cpu"))
	opt_lora.load_state_dict(torch.load(ckpt_dir / "opt_lora.pt", map_location="cpu"))
	_move_optimizer_state_to_device(opt_router, device)
	_move_optimizer_state_to_device(opt_lora, device)

	state = torch.load(ckpt_dir / "trainer_state.pt", map_location="cpu")
	return dict(state) if isinstance(state, dict) else {}


def main() -> None:
	args = parse_args()
	global MERGE_USE_PARENT_EXPERTS
	MERGE_USE_PARENT_EXPERTS = bool(getattr(args, "merge_use_parent_experts", True))
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	if args.gpus and args.gpus.strip():
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

	_seed_everything(int(args.seed))
	device = torch.device("cuda", int(args.device)) if torch.cuda.is_available() else torch.device("cpu")

	resume_dir: Path | None = None
	if getattr(args, "resume_from", None):
		resume_dir = _load_checkpoint_dir(Path(args.resume_from))
		# Expect: <ckpt_root>/<run_ts>/<step_dir>
		run_ts = resume_dir.parent.name
	else:
		requested = str(getattr(args, "run_ts", "") or "").strip()
		if requested:
			# Keep names safe for filesystem paths.
			run_ts = re.sub(r"[^a-zA-Z0-9._-]+", "_", requested).strip("_") or time.strftime("%Y%m%d_%H%M%S")
		else:
			run_ts = time.strftime("%Y%m%d_%H%M%S")
	run_root = (args.output_root / run_ts).resolve()
	ckpt_root = (args.ckpt_root / run_ts).resolve()
	run_root.mkdir(parents=True, exist_ok=True)
	ckpt_root.mkdir(parents=True, exist_ok=True)

	cfg_for_log: Dict[str, Any] = {}
	for k, v in vars(args).items():
		cfg_for_log[k] = str(v) if isinstance(v, Path) else v
	# W&B init happens after resume loads (to reuse run_id).
	logger: MetricsLogger | None = None

	# Load backbone and inject MoLE LoRA.
	model, tokenizer = _load_backbone(model_name=args.model_name, torch_dtype=args.torch_dtype, device=device)
	num_role_experts = len(ROLE_EXPERT_IDS)
	num_subtask_experts = int(args.num_subtask_experts)
	if num_subtask_experts <= 0:
		raise ValueError("--num-subtask-experts must be > 0.")
	num_experts = int(num_role_experts + num_subtask_experts)
	subtask_expert_offset = int(num_role_experts)

	lora_cfg = LoRAConfig(
		num_experts=int(num_experts),
		top_k=int(args.subtask_top_k) + 1,
		rank=int(args.lora_rank),
		alpha=float(args.lora_alpha),
		target_modules=("q_proj", "v_proj", "o_proj"),
		last_n_layers=int(args.lora_last_n_layers),
	)
	inject_mole_lora(model, cfg=lora_cfg)

	# Subtask router (+ title projection).
	router_cfg = SubtaskRouterConfig(num_experts=int(num_subtask_experts), top_k=int(args.subtask_top_k))
	router = SubtaskRouter(router_cfg).to(device)
	title_embedder = TitleEmbedder(model=model, tokenizer=tokenizer, out_dim=router_cfg.title_emb_dim).to(device)

	# Optimizers.
	opt_lora = torch.optim.AdamW(list(lora_parameters(model)), lr=float(args.lora_lr))
	opt_router = torch.optim.AdamW(list(router.parameters()) + list(title_embedder.parameters()), lr=float(args.router_lr))

	gen_cfg = GenerationConfig(
		model_name=args.model_name,
		max_new_tokens=int(args.max_new_tokens),
		temperature=float(args.temperature),
		top_p=float(args.top_p),
		torch_dtype=str(args.torch_dtype),
		device=int(args.device),
	)
	mole_gen = MoLEGenerator(model=model, tokenizer=tokenizer, device=device, gen_cfg=gen_cfg)

	update_per_attempt = bool(getattr(args, "update_per_attempt", False))
	shared_lora: Dict[str, torch.Tensor] | None = None
	gen_pool: GenWorkerPool | None = None
	if bool(getattr(args, "gen_parallel", False)) and bool(getattr(args, "use_grpo", False)) and bool(update_per_attempt):
		shared_lora = _shared_lora_state_from_model(model)
		_refresh_shared_lora_state(shared_lora, model)
		worker_devices = [d for d in _parse_device_list(str(getattr(args, "gen_worker_devices", ""))) if d != int(args.device)]
		if worker_devices:
			gen_pool = GenWorkerPool(
				worker_devices=worker_devices,
				model_name=str(args.model_name),
				torch_dtype=str(args.torch_dtype),
				gen_cfg=gen_cfg,
				lora_cfg=lora_cfg,
				shared_lora=shared_lora,
			)

	taskgraph_root = Path(args.taskgraph_root) if getattr(args, "taskgraph_root", None) else None
	taskgraph_index: Dict[Tuple[str, str], Path] | None = None
	if taskgraph_root is not None:
		if not taskgraph_root.exists():
			raise FileNotFoundError(f"--taskgraph-root not found: {taskgraph_root}")
		taskgraph_index = _load_taskgraph_index(taskgraph_root)
		if not taskgraph_index:
			raise RuntimeError(f"No task graphs found under --taskgraph-root: {taskgraph_root}")

	planner = None if taskgraph_index is not None else _graph_planner(args)

	samples = read_srdd_samples(args.srdd_csv)
	selected = iter_slice_by_category(samples, offset=int(args.per_category_offset), limit=int(args.per_category))

	checkpoint_overwrite = bool(getattr(args, "checkpoint_overwrite", False))
	checkpoint_name = str(getattr(args, "checkpoint_name", "step_latest") or "step_latest")
	baseline = 0.0
	baseline_proxy = 0.0
	attempt_step = 0
	global_step = 0
	sample_cursor = 0
	wandb_run_id = ""
	last_saved_step = -1

	# Resume: load model/router/optimizers/scalars + RNG states + sample cursor.
	if resume_dir is not None:
		state = _load_checkpoint(
			ckpt_dir=resume_dir,
			device=device,
			router=router,
			title_embedder=title_embedder,
			model=model,
			opt_router=opt_router,
			opt_lora=opt_lora,
		)
		baseline = float(state.get("baseline", 0.0))
		baseline_proxy = float(state.get("baseline_proxy", 0.0))
		attempt_step = int(state.get("attempt_step", 0))
		global_step = int(state.get("global_step", 0))
		sample_cursor = int(state.get("sample_cursor", 0))
		wandb_run_id = str(state.get("wandb_run_id", "") or "")
		last_saved_step = int(global_step)
		try:
			random.setstate(state.get("py_random_state"))
		except Exception:
			pass
		try:
			torch.set_rng_state(state.get("torch_rng_state"))
		except Exception:
			pass
		try:
			if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
				torch.cuda.set_rng_state_all(state.get("cuda_rng_state_all"))
		except Exception:
			pass
		if bool(getattr(args, "reset_sample_cursor", False)):
			sample_cursor = 0

	# If we are resuming from a checkpoint, discard any partially-logged next-sample metrics so that
	# reruns overwrite the previous partial tail and `attempt_step` stays monotonic in metrics.jsonl.
	if resume_dir is not None:
		try:
			_truncate_metrics_jsonl_to_state(
				jsonl_path=run_root / "metrics.jsonl",
				attempt_step_max=int(attempt_step),
				sample_step_max=int(global_step),
			)
		except Exception as exc:
			print(f"[warn] failed to truncate metrics.jsonl to checkpoint state: {exc}", file=sys.stderr)

	wandb_cfg = None
	if bool(getattr(args, "wandb", False)):
		wandb_cfg = WandBConfig(
			project=str(args.wandb_project),
			entity=str(args.wandb_entity or ""),
			run_name=str(args.wandb_run_name or run_ts),
			mode=str(args.wandb_mode),
			run_id=wandb_run_id,
			resume="allow" if (wandb_run_id and resume_dir is not None) else "",
		)
	logger = MetricsLogger(run_root=run_root, jsonl_path=run_root / "metrics.jsonl", wandb=wandb_cfg, config=cfg_for_log)
	if wandb_cfg is not None and wandb_cfg.run_id and getattr(logger, "_wandb_run", None) is not None:
		try:
			wandb_run_id = str(logger._wandb_run.id)  # type: ignore[attr-defined]
		except Exception:
			pass

	try:
		for sample in selected[sample_cursor:]:
			sample_start_t = time.time()
			sample_dir = run_root / sanitize(sample.category) / sanitize(sample.name)
			if bool(getattr(args, "skip_existing_samples", False)) and (sample_dir / "train_metrics.json").exists():
				print(f"[skip] already completed: {sample.category}/{sample.name}", file=sys.stderr)
				sample_cursor += 1
				continue
			log_dir = sample_dir / "log"
			repo_dir = sample_dir / "repo"
			log_dir.mkdir(parents=True, exist_ok=True)
			repo_dir.mkdir(parents=True, exist_ok=True)
			_write_text(log_dir / "task_description.txt", sample.description)

			# 1) Graph generation (fixed).
			graph_path = sample_dir / "task_graph.json"
			graph = None
			last_error: Exception | None = None
			if taskgraph_index is not None:
				src_graph = taskgraph_index.get((sample.category, sample.name))
				if src_graph is None or not src_graph.exists():
					last_error = RuntimeError(f"Precomputed task graph not found for {sample.category}/{sample.name}.")
					_write_json(
						sample_dir / "graph_generation.json",
						{
							"status": "failed",
							"attempts": 0,
							"backend": "precomputed",
							"used_fallback": False,
							"source": None,
							"error": str(last_error),
							"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
						},
					)
				else:
					if not graph_path.exists():
						graph_path.write_text(src_graph.read_text(encoding="utf-8"), encoding="utf-8")
					_write_json(
						sample_dir / "graph_generation.json",
						{
							"status": "ok",
							"attempts": 0,
							"backend": "precomputed",
							"used_fallback": False,
							"source": str(src_graph),
							"error": None,
							"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
						},
					)
					graph = object()
			else:
				max_attempts = min(max(int(args.graph_retries), 1), 3)
				for attempt in range(1, max_attempts + 1):
					try:
						graph = planner.generate(sample.name, sample.description)
						backend = getattr(planner, "last_backend", None)
						used_fallback = bool(getattr(planner, "used_fallback", False))
						# Treat fallback as a "failed attempt" for LLM planning, so we can retry.
						if used_fallback:
							last_error = planner.last_error or RuntimeError("GraphPlanner fell back to heuristic.")
							_write_text(sample_dir / f"graph_error_attempt_{attempt}.txt", str(last_error))
							if planner.last_response_text:
								_write_text(sample_dir / f"graph_response_attempt_{attempt}.txt", planner.last_response_text or "")
							continue
						graph.save_json(graph_path)
						_write_json(
							sample_dir / "graph_generation.json",
							{
								"status": "ok",
								"attempt": attempt,
								"backend": backend,
								"used_fallback": used_fallback,
								"error": str(planner.last_error) if planner.last_error else None,
							},
						)
						last_error = None
						break
					except Exception as exc:
						last_error = exc
						_write_text(sample_dir / f"graph_error_attempt_{attempt}.txt", str(exc))
						if planner.last_response_text:
							_write_text(sample_dir / f"graph_response_attempt_{attempt}.txt", planner.last_response_text or "")

			if graph is None or last_error is not None or not graph_path.exists():
				_write_json(
					sample_dir / "graph_generation.json",
					{"status": "failed", "attempts": max_attempts, "error": str(last_error) if last_error else "unknown"},
				)
				try:
					_update_training_summaries(run_root)
				except Exception:
					pass
				# Still checkpoint progress for robust resuming.
				sample_cursor += 1
				trainer_state = {
					"run_ts": run_ts,
					"global_step": int(global_step),
					"attempt_step": int(attempt_step),
					"baseline": float(baseline),
					"baseline_proxy": float(baseline_proxy),
					"sample_cursor": int(sample_cursor),
					"wandb_run_id": str(wandb_run_id),
					"py_random_state": random.getstate(),
					"torch_rng_state": torch.get_rng_state(),
					"cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
				}
				if int(args.save_every) > 0 and checkpoint_overwrite:
					_save_checkpoint(
						ckpt_dir=ckpt_root / checkpoint_name,
						router=router,
						title_embedder=title_embedder,
						model=model,
						opt_router=opt_router,
						opt_lora=opt_lora,
						trainer_state=trainer_state,
					)
				continue

				spec = convert_taskgraph(graph_path)
				node_ids = sorted(spec.node_metadata)
				predecessors = _build_predecessors(spec.edge_strings, node_ids)
				if bool(getattr(args, "disable_aggregate", False)):
					for nid, preds in predecessors.items():
						if len(preds) > 1:
							predecessors[nid] = sorted(preds)[:1]
				relaxed_until = len(node_ids) // 3  # mirror inference: early nodes allow placeholder stubs.

			# 2) Execute DAG with MoLE; collect rollout logps (per-sample mode).
			solutions: Dict[int, Codes] = {}
			rollout: List[Tuple[torch.Tensor, torch.Tensor]] = []
			role_expert_counts: Dict[Role, int] = {Role.EXECUTE: 0, Role.AGGREGATE: 0}
			subtask_expert_counts: Dict[int, int] = {i: 0 for i in range(int(num_subtask_experts))}
			subtask_consistency_vals: List[float] = []
			node_subtask_experts: Dict[int, List[int]] = {}

			for node_index, node_id in enumerate(node_ids):
				node_meta = spec.node_metadata[node_id]
				node_subtask_text = _subtask_text(node_meta.title, node_meta.description)
				layer_dir = log_dir / f"node_{node_id:02d}"
				layer_dir.mkdir(parents=True, exist_ok=True)
				allow_pass_todo = node_index < relaxed_until
				allow_format_placeholders = node_index < relaxed_until

				parent_ids = predecessors.get(node_id, [])
				parent_ids = sorted([pid for pid in parent_ids if pid in solutions])

				# Base snapshot: either merged from parents (aggregate) or a single parent snapshot.
				if not parent_ids:
					base_snapshot = ""
					base_codes: Codes | None = None
				elif len(parent_ids) == 1:
					base_codes = solutions[parent_ids[0]]
					base_snapshot = base_codes.snapshot()
				else:
					parent_snaps = [solutions[p].snapshot() for p in parent_ids]
					parent_subtask_union = sorted(
						{idx for pid in parent_ids for idx in node_subtask_experts.get(pid, [])}
					)
					agg_expert_ids, agg_logp_router = _select_aggregate_experts(
						device=device,
						parent_subtask_experts=parent_subtask_union,
						subtask_expert_offset=subtask_expert_offset,
					)

					last_exc: Exception | None = None
					last_hint: str | None = None
					last_agg_codes: Codes | None = None
					base_codes = solutions[parent_ids[0]]
					base_snapshot = base_codes.snapshot()
					for attempt in range(1, int(args.max_attempts) + 1):
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

						attempt_error: Exception | None = None
						validate_ok = False
						smoke_ok = False
						parse_ok = False
						start_t = time.time()

						expert_ids = agg_expert_ids
						logp_router = agg_logp_router

						_record_expert_selection(
							role_counts=role_expert_counts,
							subtask_counts=subtask_expert_counts,
							role=Role.AGGREGATE,
							expert_ids=expert_ids,
							subtask_expert_offset=subtask_expert_offset,
							num_subtask_experts=int(num_subtask_experts),
						)

						if update_per_attempt and bool(getattr(args, "use_grpo", False)):
							group_size = max(int(getattr(args, "grpo_group_size", 8)), 1)
							group_max = max(int(getattr(args, "grpo_group_size_max", 8)), group_size)
							cands: List[GRPOCandidate] = []

							use_parallel = (
								gen_pool is not None
								and shared_lora is not None
								and bool(getattr(args, "gen_parallel", False))
							)
							if use_parallel:
								main_n = max(int(getattr(args, "gen_main_batch", 2)), 0)
								per_worker = max(int(getattr(args, "gen_batch_per_device", 2)), 1)
								total = int(main_n) + int(per_worker) * int(len(gen_pool.worker_devices))
								if total != int(group_size):
									use_parallel = False

							if use_parallel:
								seed_base = random.randint(0, 2**31 - 1)
								candidate_start_by_device: Dict[int, int] = {}
								next_idx = int(main_n) + 1
								for dev in gen_pool.worker_devices:
									candidate_start_by_device[int(dev)] = int(next_idx)
									next_idx += int(per_worker)
								gen_pool.dispatch_generate(
									prompt=prompt,
									expert_ids=[int(x) for x in expert_ids.detach().cpu().tolist()],
									num_samples_per_worker=int(per_worker),
									candidate_start_by_device=candidate_start_by_device,
									seed=int(seed_base),
								)

								local_samples: List[Dict[str, Any]] = []
								if main_n > 0:
									torch.manual_seed(int(seed_base) + 99991)
									local_outs = mole_gen.generate_n_with_experts(
										prompt=prompt, expert_ids=expert_ids, num_samples=int(main_n)
									)
									for i, (text_i, prompt_ids_i, gen_ids_i) in enumerate(local_outs):
										local_samples.append(
											{
												"candidate_idx": int(i + 1),
												"text": text_i,
												"prompt_ids": prompt_ids_i.detach().cpu().tolist(),
												"gen_ids": gen_ids_i.detach().cpu().tolist(),
											}
										)

								worker_samples = gen_pool.collect_generated()
								all_samples = list(local_samples) + list(worker_samples)
								all_samples = [s for s in all_samples if isinstance(s, dict) and "candidate_idx" in s]
								all_samples.sort(key=lambda s: int(s.get("candidate_idx", 0)))
								for s in all_samples[: int(group_size)]:
									cands.append(
										_sample_grpo_candidate(
											stage="aggregate",
											candidate_idx=int(s["candidate_idx"]),
											prompt=prompt,
											mole_gen=mole_gen,
											expert_ids=expert_ids,
											layer_dir=layer_dir,
											attempt=int(attempt),
											task_description=spec.task_description,
											subtask_description=node_subtask_text,
											allow_pass_todo=allow_pass_todo,
											allow_format_placeholders=allow_format_placeholders,
											args=args,
											generated=s,
										)
									)
							else:
								for cand_idx in range(1, group_size + 1):
									cands.append(
										_sample_grpo_candidate(
											stage="aggregate",
											candidate_idx=int(cand_idx),
											prompt=prompt,
											mole_gen=mole_gen,
											expert_ids=expert_ids,
											layer_dir=layer_dir,
											attempt=int(attempt),
											task_description=spec.task_description,
											subtask_description=node_subtask_text,
											allow_pass_todo=allow_pass_todo,
											allow_format_placeholders=allow_format_placeholders,
											args=args,
										)
									)

							adaptive_triggered = False
							if bool(getattr(args, "grpo_adaptive", False)) and group_size < group_max:
								if all(float(c.proxy_reward) == 0.0 for c in cands):
									adaptive_triggered = True
									for cand_idx in range(group_size + 1, group_max + 1):
										cands.append(
											_sample_grpo_candidate(
												stage="aggregate",
												candidate_idx=int(cand_idx),
												prompt=prompt,
												mole_gen=mole_gen,
												expert_ids=expert_ids,
												layer_dir=layer_dir,
												attempt=int(attempt),
												task_description=spec.task_description,
												subtask_description=node_subtask_text,
												allow_pass_todo=allow_pass_todo,
												allow_format_placeholders=allow_format_placeholders,
												args=args,
											)
										)

							selected_cand = _select_grpo_candidate(cands)
							subtask_consistency_vals.append(
								float(selected_cand.proxy_metrics.get("proxy_consistency_subtask_stripped", 0.0))
							)
							_write_text(layer_dir / f"aggregate_response_attempt_{attempt}.txt", selected_cand.text)
							agg_codes = selected_cand.codes
							last_agg_codes = agg_codes
							parse_ok = bool(selected_cand.parse_ok)
							validate_ok = bool(selected_cand.validate_ok)
							smoke_ok = bool(selected_cand.smoke_ok)
							attempt_error = selected_cand.error
							elapsed_s = float(time.time() - start_t)

							rewards = [float(c.proxy_reward) for c in cands]
							reward_mean = float(sum(rewards) / float(len(rewards))) if rewards else 0.0
							reward_std = float(_safe_std(rewards))
							alpha_router = float(args.alpha_router)

							baseline_before = float(baseline_proxy)
							baseline_proxy = float(args.baseline_momentum) * float(baseline_proxy) + (1.0 - float(args.baseline_momentum)) * float(reward_mean)

							did_update = True
							loss_value = 0.0
							adv_selected_raw = float(selected_cand.proxy_reward) - float(reward_mean)
							adv_selected = float(adv_selected_raw)
							if bool(getattr(args, "grpo_adv_normalize", False)):
								adv_selected = adv_selected / float(reward_std + float(getattr(args, "grpo_adv_eps", 1e-6))) if reward_std > 0.0 else 0.0
							adv_selected = _clip_advantage(float(adv_selected), float(getattr(args, "advantage_clip", 0.0)))

							if bool(getattr(args, "grpo_skip_update_if_allzero", False)) and rewards and (max(rewards) - min(rewards) == 0.0):
								did_update = False
							selected_logp_mole_sum = 0.0
							selected_logp_mole_mean = 0.0
							selected_gen_len = int(selected_cand.gen_ids.numel()) if hasattr(selected_cand, "gen_ids") else 0
							if did_update and rewards:
								opt_router.zero_grad(set_to_none=True)
								opt_lora.zero_grad(set_to_none=True)

								loss_total = torch.tensor(0.0, device=device)
								sum_adv = 0.0
								for c in cands:
									adv = float(c.proxy_reward) - float(reward_mean)
									if bool(getattr(args, "grpo_adv_normalize", False)):
										adv = adv / float(reward_std + float(getattr(args, "grpo_adv_eps", 1e-6))) if reward_std > 0.0 else 0.0
									adv = _clip_advantage(float(adv), float(getattr(args, "advantage_clip", 0.0)))
									sum_adv += float(adv)
									if float(adv) == 0.0:
										continue

									logp_mole_sum = mole_gen.logprob_of_generation(prompt_ids=c.prompt_ids, gen_ids=c.gen_ids, expert_ids=expert_ids)
									gen_len = int(c.gen_ids.numel())
									logp_mole_mean = logp_mole_sum / max(1.0, float(gen_len))
									logp_mole = logp_mole_sum
									c.logp_mole = logp_mole
									if int(c.idx) == int(selected_cand.idx):
										selected_logp_mole_sum = float(logp_mole_sum.detach().cpu().item())
										selected_logp_mole_mean = float(logp_mole_mean.detach().cpu().item())
										selected_gen_len = int(gen_len)

									loss_i = -(torch.tensor(float(adv), device=device) * logp_mole_mean)
									loss_i.backward()
									loss_total = loss_total + loss_i.detach()

								if float(sum_adv) != 0.0:
									loss_router = -(torch.tensor(float(sum_adv), device=device) * alpha_router * logp_router)
									loss_router.backward()
									loss_total = loss_total + loss_router.detach()

								if float(args.subtask_proto_l2) > 0.0 or float(args.subtask_proto_ortho) > 0.0:
									reg_loss = _subtask_router_reg_loss(
										router, float(args.subtask_proto_l2), float(args.subtask_proto_ortho)
									)
									reg_loss.backward()
									loss_total = loss_total + reg_loss.detach()

								opt_router.step()
								opt_lora.step()
								if shared_lora is not None:
									_refresh_shared_lora_state(shared_lora, model)
								loss_value = float(loss_total.detach().cpu().item())

							attempt_step += 1
							logger.log_attempt(
								attempt_step=int(attempt_step),
								metrics={
									"category": sample.category,
									"sample_name": sample.name,
									"node_id": int(node_id),
									"role": str(Role.AGGREGATE.value),
									"attempt_idx": int(attempt),
									"elapsed_s": float(elapsed_s),
									"expert_ids": [int(x) for x in expert_ids.detach().cpu().tolist()],
									"logp_mole": float(selected_logp_mole_sum),
									"logp_mole_mean": float(selected_logp_mole_mean),
									"gen_len": int(selected_gen_len),
									"logp_router": float(logp_router.detach().cpu().item()),
									**selected_cand.proxy_metrics,
									"baseline_before": float(baseline_before),
									"advantage_raw": float(adv_selected_raw),
									"advantage": float(adv_selected),
									"baseline_after": float(baseline_proxy),
									"loss": float(loss_value),
									"error": str(attempt_error) if attempt_error is not None else "",
									"grpo": True,
									"grpo_group_size_target": int(group_size),
									"grpo_group_size_used": int(len(cands)),
									"grpo_adaptive_triggered": bool(adaptive_triggered),
									"grpo_rewards": rewards,
									"grpo_reward_mean": float(reward_mean),
									"grpo_reward_std": float(reward_std),
									"grpo_selected_idx": int(selected_cand.idx),
									"grpo_selected_reward": float(selected_cand.proxy_reward),
									"grpo_did_update": bool(did_update),
								},
							)
						else:
							text, prompt_ids, gen_ids = mole_gen.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
							_write_text(layer_dir / f"aggregate_response_attempt_{attempt}.txt", text)
							agg_codes = Codes(text)
							last_agg_codes = agg_codes
							parse_ok = bool(agg_codes.codebooks)

							try:
								validate_codes(
									agg_codes,
									require_main=True,
									stdlib_only=True,
									allow_pass_todo=allow_pass_todo,
									allow_format_placeholders=allow_format_placeholders,
								)
								validate_ok = True
								_run_smoke_test(codes=agg_codes, layer_dir=layer_dir, stage="aggregate", attempt=attempt, timeout_s=int(args.smoke_timeout_s))
								smoke_ok = True
							except Exception as exc:
								attempt_error = exc

							elapsed_s = float(time.time() - start_t)

							if update_per_attempt or attempt_error is None:
								logp_mole_sum = mole_gen.logprob_of_generation(prompt_ids=prompt_ids, gen_ids=gen_ids, expert_ids=expert_ids)
								gen_len = int(gen_ids.numel())
								logp_mole_mean = logp_mole_sum / max(1.0, float(gen_len))

							if update_per_attempt:
								r_proxy, proxy_metrics = _proxy_reward(
									task_description=spec.task_description,
									subtask_description=node_subtask_text,
									codes=agg_codes,
									parse_ok=parse_ok,
									validate_ok=validate_ok,
									smoke_ok=smoke_ok,
									w_smoke=float(args.proxy_w_smoke),
									w_comp=float(args.proxy_w_comp),
									w_cons_strip=float(args.proxy_w_cons_strip),
									w_cons_task=float(args.proxy_cons_task_weight),
									w_cons_subtask=float(args.proxy_cons_subtask_weight),
								)
								subtask_consistency_vals.append(
									float(proxy_metrics.get("proxy_consistency_subtask_stripped", 0.0))
								)
								baseline_before = float(baseline_proxy)
								advantage_raw = float(r_proxy) - float(baseline_before)
								advantage = _clip_advantage(float(advantage_raw), float(getattr(args, "advantage_clip", 0.0)))
								baseline_proxy = float(args.baseline_momentum) * float(baseline_proxy) + (1.0 - float(args.baseline_momentum)) * float(r_proxy)

								opt_router.zero_grad(set_to_none=True)
								opt_lora.zero_grad(set_to_none=True)
								alpha_router = float(args.alpha_router)
								loss = -(torch.tensor(float(advantage), device=device) * (logp_mole_mean + alpha_router * logp_router))
								loss_total = loss
								if float(args.subtask_proto_l2) > 0.0 or float(args.subtask_proto_ortho) > 0.0:
									reg_loss = _subtask_router_reg_loss(
										router, float(args.subtask_proto_l2), float(args.subtask_proto_ortho)
									)
									loss_total = loss_total + reg_loss
								loss_total.backward()
								opt_router.step()
								opt_lora.step()
								if shared_lora is not None:
									_refresh_shared_lora_state(shared_lora, model)

								attempt_step += 1
								logger.log_attempt(
									attempt_step=int(attempt_step),
									metrics={
										"category": sample.category,
										"sample_name": sample.name,
										"node_id": int(node_id),
										"role": str(Role.AGGREGATE.value),
										"attempt_idx": int(attempt),
										"elapsed_s": float(elapsed_s),
										"expert_ids": [int(x) for x in expert_ids.detach().cpu().tolist()],
										"logp_mole": float(logp_mole_sum.detach().cpu().item()),
										"logp_mole_mean": float(logp_mole_mean.detach().cpu().item()),
										"gen_len": int(gen_len),
										"logp_router": float(logp_router.detach().cpu().item()),
										**proxy_metrics,
										"baseline_before": float(baseline_before),
										"advantage_raw": float(advantage_raw),
										"advantage": float(advantage),
										"baseline_after": float(baseline_proxy),
										"loss": float(loss_total.detach().cpu().item()),
										"error": str(attempt_error) if attempt_error is not None else "",
										"grpo": False,
									},
								)
							else:
								if attempt_error is None:
									rollout.append((logp_mole_mean, logp_router))

						if attempt_error is None:
							base_codes = agg_codes
							base_snapshot = agg_codes.snapshot()
							last_exc = None
							break

						last_exc = attempt_error
						try:
							last_hint = summarize_validation_hints(last_agg_codes or Codes(""))
						except Exception:
							last_hint = None
						if "did not contain any parseable code blocks/files" in str(attempt_error).lower():
							last_hint = (
								(last_hint + "\n" if last_hint else "")
								+ "- Output must be a sequence of files. Each file must start with '<name>.py' on its own line, "
								+ "followed by a fenced python code block."
							)
						_write_text(layer_dir / f"aggregate_error_attempt_{attempt}.txt", str(attempt_error))

					if last_exc is not None:
						_write_text(layer_dir / "aggregate_failure.txt", f"Aggregation failed after {int(args.max_attempts)} attempts.\nError: {last_exc}\n")
						if last_agg_codes is not None and last_agg_codes.codebooks:
							base_codes = last_agg_codes
							base_snapshot = last_agg_codes.snapshot()
						else:
							base_codes = solutions[parent_ids[0]]
							base_snapshot = base_codes.snapshot()

				# Execute node: mirror inference retry mechanism.
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

				if not update_per_attempt:
					expert_ids, logp_router = _select_experts(
						router=router,
						title_embedder=title_embedder,
						device=device,
						role=Role.EXECUTE,
						subtask_text=node_subtask_text,
						subtask_expert_offset=subtask_expert_offset,
					)

				last_exc: Exception | None = None
				last_hint: str | None = None
				codes: Codes | None = None
				last_candidate: Codes | None = None

				for attempt in range(1, int(args.max_attempts) + 1):
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

					attempt_error: Exception | None = None
					validate_ok = False
					smoke_ok = False
					parse_ok = False
					start_t = time.time()

					if update_per_attempt:
						expert_ids, logp_router = _select_experts(
							router=router,
							title_embedder=title_embedder,
							device=device,
							role=Role.EXECUTE,
							subtask_text=node_subtask_text,
							subtask_expert_offset=subtask_expert_offset,
						)

					_record_expert_selection(
						role_counts=role_expert_counts,
						subtask_counts=subtask_expert_counts,
						role=Role.EXECUTE,
						expert_ids=expert_ids,
						subtask_expert_offset=subtask_expert_offset,
						num_subtask_experts=int(num_subtask_experts),
					)

					if update_per_attempt and bool(getattr(args, "use_grpo", False)):
						group_size = max(int(getattr(args, "grpo_group_size", 8)), 1)
						group_max = max(int(getattr(args, "grpo_group_size_max", 8)), group_size)
						cands: List[GRPOCandidate] = []

						use_parallel = (
							gen_pool is not None
							and shared_lora is not None
							and bool(getattr(args, "gen_parallel", False))
						)
						if use_parallel:
							main_n = max(int(getattr(args, "gen_main_batch", 2)), 0)
							per_worker = max(int(getattr(args, "gen_batch_per_device", 2)), 1)
							total = int(main_n) + int(per_worker) * int(len(gen_pool.worker_devices))
							if total != int(group_size):
								use_parallel = False

						if use_parallel:
							seed_base = random.randint(0, 2**31 - 1)
							candidate_start_by_device: Dict[int, int] = {}
							next_idx = int(main_n) + 1
							for dev in gen_pool.worker_devices:
								candidate_start_by_device[int(dev)] = int(next_idx)
								next_idx += int(per_worker)
							gen_pool.dispatch_generate(
								prompt=attempt_prompt,
								expert_ids=[int(x) for x in expert_ids.detach().cpu().tolist()],
								num_samples_per_worker=int(per_worker),
								candidate_start_by_device=candidate_start_by_device,
								seed=int(seed_base),
							)

							local_samples: List[Dict[str, Any]] = []
							if main_n > 0:
								torch.manual_seed(int(seed_base) + 42421)
								local_outs = mole_gen.generate_n_with_experts(
									prompt=attempt_prompt, expert_ids=expert_ids, num_samples=int(main_n)
								)
								for i, (text_i, prompt_ids_i, gen_ids_i) in enumerate(local_outs):
									local_samples.append(
										{
											"candidate_idx": int(i + 1),
											"text": text_i,
											"prompt_ids": prompt_ids_i.detach().cpu().tolist(),
											"gen_ids": gen_ids_i.detach().cpu().tolist(),
										}
									)

							worker_samples = gen_pool.collect_generated()
							all_samples = list(local_samples) + list(worker_samples)
							all_samples = [s for s in all_samples if isinstance(s, dict) and "candidate_idx" in s]
							all_samples.sort(key=lambda s: int(s.get("candidate_idx", 0)))
							for s in all_samples[: int(group_size)]:
								cands.append(
									_sample_grpo_candidate(
										stage="execute",
										candidate_idx=int(s["candidate_idx"]),
										prompt=attempt_prompt,
										mole_gen=mole_gen,
										expert_ids=expert_ids,
										layer_dir=layer_dir,
										attempt=int(attempt),
										task_description=spec.task_description,
										subtask_description=node_subtask_text,
										allow_pass_todo=allow_pass_todo,
										allow_format_placeholders=allow_format_placeholders,
										args=args,
										generated=s,
									)
								)
						else:
							for cand_idx in range(1, group_size + 1):
								cands.append(
									_sample_grpo_candidate(
										stage="execute",
										candidate_idx=int(cand_idx),
										prompt=attempt_prompt,
										mole_gen=mole_gen,
										expert_ids=expert_ids,
										layer_dir=layer_dir,
										attempt=int(attempt),
										task_description=spec.task_description,
										subtask_description=node_subtask_text,
										allow_pass_todo=allow_pass_todo,
										allow_format_placeholders=allow_format_placeholders,
										args=args,
									)
								)

						adaptive_triggered = False
						if bool(getattr(args, "grpo_adaptive", False)) and group_size < group_max:
							if all(float(c.proxy_reward) == 0.0 for c in cands):
								adaptive_triggered = True
								for cand_idx in range(group_size + 1, group_max + 1):
									cands.append(
										_sample_grpo_candidate(
											stage="execute",
											candidate_idx=int(cand_idx),
											prompt=attempt_prompt,
											mole_gen=mole_gen,
											expert_ids=expert_ids,
											layer_dir=layer_dir,
											attempt=int(attempt),
											task_description=spec.task_description,
											subtask_description=node_subtask_text,
											allow_pass_todo=allow_pass_todo,
											allow_format_placeholders=allow_format_placeholders,
											args=args,
										)
									)

						selected_cand = _select_grpo_candidate(cands)
						subtask_consistency_vals.append(
							float(selected_cand.proxy_metrics.get("proxy_consistency_subtask_stripped", 0.0))
						)
						_write_text(layer_dir / f"response_attempt_{attempt}.txt", selected_cand.text)

						candidate = selected_cand.codes
						last_candidate = candidate
						parse_ok = bool(selected_cand.parse_ok)
						if parse_ok:
							current_snapshot = candidate.snapshot()
						validate_ok = bool(selected_cand.validate_ok)
						smoke_ok = bool(selected_cand.smoke_ok)
						attempt_error = selected_cand.error
						elapsed_s = float(time.time() - start_t)

						rewards = [float(c.proxy_reward) for c in cands]
						reward_mean = float(sum(rewards) / float(len(rewards))) if rewards else 0.0
						reward_std = float(_safe_std(rewards))
						alpha_router = float(args.alpha_router)

						baseline_before = float(baseline_proxy)
						baseline_proxy = float(args.baseline_momentum) * float(baseline_proxy) + (1.0 - float(args.baseline_momentum)) * float(reward_mean)

						did_update = True
						loss_value = 0.0
						adv_selected_raw = float(selected_cand.proxy_reward) - float(reward_mean)
						adv_selected = float(adv_selected_raw)
						if bool(getattr(args, "grpo_adv_normalize", False)):
							adv_selected = adv_selected / float(reward_std + float(getattr(args, "grpo_adv_eps", 1e-6))) if reward_std > 0.0 else 0.0
						adv_selected = _clip_advantage(float(adv_selected), float(getattr(args, "advantage_clip", 0.0)))

						if bool(getattr(args, "grpo_skip_update_if_allzero", False)) and rewards and (max(rewards) - min(rewards) == 0.0):
							did_update = False
						selected_logp_mole_sum = 0.0
						selected_logp_mole_mean = 0.0
						selected_gen_len = int(selected_cand.gen_ids.numel()) if hasattr(selected_cand, "gen_ids") else 0
						if did_update and rewards:
							opt_router.zero_grad(set_to_none=True)
							opt_lora.zero_grad(set_to_none=True)

							loss_total = torch.tensor(0.0, device=device)
							sum_adv = 0.0
							for c in cands:
								adv = float(c.proxy_reward) - float(reward_mean)
								if bool(getattr(args, "grpo_adv_normalize", False)):
									adv = adv / float(reward_std + float(getattr(args, "grpo_adv_eps", 1e-6))) if reward_std > 0.0 else 0.0
								adv = _clip_advantage(float(adv), float(getattr(args, "advantage_clip", 0.0)))
								sum_adv += float(adv)
								if float(adv) == 0.0:
									continue

								logp_mole_sum = mole_gen.logprob_of_generation(
									prompt_ids=c.prompt_ids, gen_ids=c.gen_ids, expert_ids=expert_ids
								)
								gen_len = int(c.gen_ids.numel())
								logp_mole_mean = logp_mole_sum / max(1.0, float(gen_len))
								c.logp_mole = logp_mole_sum
								if int(c.idx) == int(selected_cand.idx):
									selected_logp_mole_sum = float(logp_mole_sum.detach().cpu().item())
									selected_logp_mole_mean = float(logp_mole_mean.detach().cpu().item())
									selected_gen_len = int(gen_len)

								loss_i = -(torch.tensor(float(adv), device=device) * logp_mole_mean)
								loss_i.backward()
								loss_total = loss_total + loss_i.detach()

							# Router term (typically cancels out when advantages are mean-centered).
							if float(sum_adv) != 0.0:
								loss_router = -(torch.tensor(float(sum_adv), device=device) * alpha_router * logp_router)
								loss_router.backward()
								loss_total = loss_total + loss_router.detach()

							if float(args.subtask_proto_l2) > 0.0 or float(args.subtask_proto_ortho) > 0.0:
								reg_loss = _subtask_router_reg_loss(
									router, float(args.subtask_proto_l2), float(args.subtask_proto_ortho)
								)
								reg_loss.backward()
								loss_total = loss_total + reg_loss.detach()

							opt_router.step()
							opt_lora.step()
							if shared_lora is not None:
								_refresh_shared_lora_state(shared_lora, model)
							loss_value = float(loss_total.detach().cpu().item())

						attempt_step += 1
						logger.log_attempt(
							attempt_step=int(attempt_step),
							metrics={
								"category": sample.category,
								"sample_name": sample.name,
								"node_id": int(node_id),
								"role": str(Role.EXECUTE.value),
								"attempt_idx": int(attempt),
								"elapsed_s": float(elapsed_s),
								"expert_ids": [int(x) for x in expert_ids.detach().cpu().tolist()],
								"logp_mole": float(selected_logp_mole_sum),
								"logp_mole_mean": float(selected_logp_mole_mean),
								"gen_len": int(selected_gen_len),
								"logp_router": float(logp_router.detach().cpu().item()),
								**selected_cand.proxy_metrics,
								"baseline_before": float(baseline_before),
								"advantage_raw": float(adv_selected_raw),
								"advantage": float(adv_selected),
								"baseline_after": float(baseline_proxy),
								"loss": float(loss_value),
								"error": str(attempt_error) if attempt_error is not None else "",
								"grpo": True,
								"grpo_group_size_target": int(group_size),
								"grpo_group_size_used": int(len(cands)),
								"grpo_adaptive_triggered": bool(adaptive_triggered),
								"grpo_rewards": rewards,
								"grpo_reward_mean": float(reward_mean),
								"grpo_reward_std": float(reward_std),
								"grpo_selected_idx": int(selected_cand.idx),
								"grpo_selected_reward": float(selected_cand.proxy_reward),
								"grpo_did_update": bool(did_update),
							},
						)
					else:
						text, prompt_ids, gen_ids = mole_gen.generate_with_experts(prompt=attempt_prompt, expert_ids=expert_ids)
						_write_text(layer_dir / f"response_attempt_{attempt}.txt", text)

						candidate = Codes(text)
						last_candidate = candidate
						parse_ok = bool(candidate.codebooks)
						if parse_ok:
							current_snapshot = candidate.snapshot()

						try:
							validate_codes(
								candidate,
								require_main=True,
								stdlib_only=True,
								allow_pass_todo=allow_pass_todo,
								allow_format_placeholders=allow_format_placeholders,
							)
							validate_ok = True
							_run_smoke_test(
								codes=candidate,
								layer_dir=layer_dir,
								stage="execute",
								attempt=attempt,
								timeout_s=int(args.smoke_timeout_s),
							)
							smoke_ok = True
						except Exception as exc:
							attempt_error = exc

						elapsed_s = float(time.time() - start_t)

						if update_per_attempt or attempt_error is None:
							logp_mole_sum = mole_gen.logprob_of_generation(prompt_ids=prompt_ids, gen_ids=gen_ids, expert_ids=expert_ids)
							gen_len = int(gen_ids.numel())
							logp_mole_mean = logp_mole_sum / max(1.0, float(gen_len))

						if update_per_attempt:
							r_proxy, proxy_metrics = _proxy_reward(
								task_description=spec.task_description,
								subtask_description=node_subtask_text,
								codes=candidate,
								parse_ok=parse_ok,
								validate_ok=validate_ok,
								smoke_ok=smoke_ok,
								w_smoke=float(args.proxy_w_smoke),
								w_comp=float(args.proxy_w_comp),
								w_cons_strip=float(args.proxy_w_cons_strip),
								w_cons_task=float(args.proxy_cons_task_weight),
								w_cons_subtask=float(args.proxy_cons_subtask_weight),
							)
							subtask_consistency_vals.append(
								float(proxy_metrics.get("proxy_consistency_subtask_stripped", 0.0))
							)
							baseline_before = float(baseline_proxy)
							advantage_raw = float(r_proxy) - float(baseline_before)
							advantage = _clip_advantage(float(advantage_raw), float(getattr(args, "advantage_clip", 0.0)))
							baseline_proxy = float(args.baseline_momentum) * float(baseline_proxy) + (1.0 - float(args.baseline_momentum)) * float(r_proxy)

							opt_router.zero_grad(set_to_none=True)
							opt_lora.zero_grad(set_to_none=True)
							alpha_router = float(args.alpha_router)
							loss = -(torch.tensor(float(advantage), device=device) * (logp_mole_mean + alpha_router * logp_router))
							loss_total = loss
							if float(args.subtask_proto_l2) > 0.0 or float(args.subtask_proto_ortho) > 0.0:
								reg_loss = _subtask_router_reg_loss(
									router, float(args.subtask_proto_l2), float(args.subtask_proto_ortho)
								)
								loss_total = loss_total + reg_loss
							loss_total.backward()
							opt_router.step()
							opt_lora.step()
							if shared_lora is not None:
								_refresh_shared_lora_state(shared_lora, model)

							attempt_step += 1
							logger.log_attempt(
								attempt_step=int(attempt_step),
								metrics={
									"category": sample.category,
									"sample_name": sample.name,
									"node_id": int(node_id),
									"role": str(Role.EXECUTE.value),
									"attempt_idx": int(attempt),
									"elapsed_s": float(elapsed_s),
									"expert_ids": [int(x) for x in expert_ids.detach().cpu().tolist()],
									"logp_mole": float(logp_mole_sum.detach().cpu().item()),
									"logp_mole_mean": float(logp_mole_mean.detach().cpu().item()),
									"gen_len": int(gen_len),
									"logp_router": float(logp_router.detach().cpu().item()),
									**proxy_metrics,
									"baseline_before": float(baseline_before),
									"advantage_raw": float(advantage_raw),
									"advantage": float(advantage),
									"baseline_after": float(baseline_proxy),
									"loss": float(loss_total.detach().cpu().item()),
									"error": str(attempt_error) if attempt_error is not None else "",
									"grpo": False,
								},
							)
						else:
							if attempt_error is None:
								rollout.append((logp_mole_mean, logp_router))

					if attempt_error is None:
						codes = candidate
						try:
							chosen = [int(x) for x in expert_ids.detach().cpu().tolist()]
							subtask_ids = [eid - int(subtask_expert_offset) for eid in chosen if eid >= int(subtask_expert_offset)]
						except Exception:
							subtask_ids = []
						node_subtask_experts[node_id] = subtask_ids
						last_exc = None
						break

					last_exc = attempt_error
					try:
						last_hint = summarize_validation_hints(last_candidate or Codes(""))
					except Exception:
						last_hint = None
					if "did not contain any parseable code blocks/files" in str(attempt_error).lower():
						last_hint = (
							(last_hint + "\n" if last_hint else "")
							+ "- Output must be a sequence of files. Each file must start with '<name>.py' on its own line, "
							+ "followed by a fenced python code block."
						)
					_write_text(layer_dir / f"error_attempt_{attempt}.txt", str(attempt_error))

				if last_exc is not None or codes is None:
					_write_text(
						layer_dir / "node_failure.txt",
						f"Node generation failed after {int(args.max_attempts)} attempts.\nError: {last_exc}\n",
					)
					if last_candidate is not None and last_candidate.codebooks:
						codes = last_candidate
					elif base_codes is not None and base_codes.codebooks:
						codes = base_codes
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

			final_codes = solutions[sorted(solutions)[-1]] if solutions else Codes("")
			final_codes.write_to_directory(repo_dir)

			# 3) Evaluate SRDD and compute reward.
			report_path = sample_dir / "srdd_report.txt"
			try:
				srdd_evaluator.evaluate(sample.name, str(args.srdd_csv), str(repo_dir), str(report_path))
			except Exception as exc:
				_write_text(log_dir / "srdd_eval_error.txt", str(exc))
				report_path.write_text("Executability: 0.0\nCompleteness: 0.0\nConsistency: 0.0\n", encoding="utf-8")

			reward, metrics = compute_reward(task_description=sample.description, repo_dir=repo_dir, srdd_report_path=report_path)
			metrics["update_per_attempt"] = bool(update_per_attempt)
			metrics["attempt_reward"] = str(args.attempt_reward)
			metrics["attempt_step_last"] = int(attempt_step)
			metrics["attempt_baseline_last"] = float(baseline_proxy)
			metrics["elapsed_ms"] = float((time.time() - float(sample_start_t)) * 1000.0)

			# 4) RL update: Router + LoRA experts (backbone stays frozen).
			if not update_per_attempt:
				metrics["baseline_before"] = float(baseline)
				advantage_raw = float(reward) - float(baseline)
				advantage = _clip_advantage(float(advantage_raw), float(getattr(args, "advantage_clip", 0.0)))
				baseline = float(args.baseline_momentum) * float(baseline) + (1.0 - float(args.baseline_momentum)) * float(reward)
				metrics["advantage_raw"] = float(advantage_raw)
				metrics["advantage"] = float(advantage)
				metrics["baseline_after"] = float(baseline)

				if rollout:
					opt_router.zero_grad(set_to_none=True)
					opt_lora.zero_grad(set_to_none=True)
					alpha_router = float(args.alpha_router)
					# logp_mole carries grads for LoRA; logp_router carries grads for Router.
					logp_sum = sum((lm + alpha_router * lr) for (lm, lr) in rollout)
					loss = -(torch.tensor(float(advantage), device=device) * logp_sum)
					loss_total = loss
					if float(args.subtask_proto_l2) > 0.0 or float(args.subtask_proto_ortho) > 0.0:
						reg_loss = _subtask_router_reg_loss(
							router, float(args.subtask_proto_l2), float(args.subtask_proto_ortho)
						)
						loss_total = loss_total + reg_loss
					loss_total.backward()
					opt_router.step()
					opt_lora.step()
					metrics["loss"] = float(loss_total.detach().cpu().item())
				else:
					metrics["loss"] = 0.0
			else:
				metrics["baseline_before"] = None
				metrics["advantage"] = None
				metrics["baseline_after"] = None
				metrics["loss"] = 0.0

			_write_json(sample_dir / "train_metrics.json", metrics)
			_write_json(
				sample_dir / "eval_results.json",
				{
					"category": sample.category,
					"sample_name": sample.name,
					"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
					"srdd": {
						"executability": metrics.get("executability"),
						"completeness": metrics.get("completeness"),
						"consistency": metrics.get("consistency_srdd"),
						"consistency_stripped": metrics.get("consistency_stripped_srdd"),
					},
					"embedding": {
						"consistency_embedding": metrics.get("consistency_embedding"),
						"consistency_embedding_stripped": metrics.get("consistency_embedding_stripped"),
					},
					"train": {
						"reward": metrics.get("reward"),
						"loss": metrics.get("loss"),
						"advantage": metrics.get("advantage"),
						"main_gate_pass": metrics.get("main_gate_pass"),
						"attempt_step_last": metrics.get("attempt_step_last"),
					},
					"timing": {"elapsed_ms": metrics.get("elapsed_ms")},
				},
			)

			summary = None
			try:
				summary = _update_training_summaries(run_root)
			except Exception:
				summary = None

			subtask_consistency_mean = float(_mean(subtask_consistency_vals))
			subtask_total = int(sum(subtask_expert_counts.values()))
			subtask_expert_metrics: Dict[str, Any] = {}
			for idx in range(int(num_subtask_experts)):
				count = int(subtask_expert_counts.get(idx, 0))
				subtask_expert_metrics[f"subtask_expert_count_{idx}"] = count
				subtask_expert_metrics[f"subtask_expert_frac_{idx}"] = float(count) / float(subtask_total) if subtask_total > 0 else 0.0

			logger.log_sample(
				sample_step=int(global_step + 1),
				metrics={
					"category": sample.category,
					"sample_name": sample.name,
					"update_per_attempt": bool(update_per_attempt),
					"train_loss": float(metrics.get("loss", 0.0) or 0.0),
					"train_advantage": float(metrics.get("advantage", 0.0) or 0.0) if not update_per_attempt else None,
					"train_baseline_after": float(metrics.get("baseline_after", 0.0) or 0.0) if not update_per_attempt else None,
					"srdd_reward": float(metrics.get("reward", reward)),
					"srdd_executability": float(metrics.get("executability", 0.0)),
					"srdd_completeness": float(metrics.get("completeness", 0.0)),
					"srdd_consistency_stripped": float(metrics.get("consistency_stripped", 0.0)),
					"srdd_consistency_report": float(metrics.get("consistency_srdd", 0.0)),
					"srdd_consistency_stripped_report": float(metrics.get("consistency_stripped_srdd", 0.0)),
					"consistency_embedding": float(metrics.get("consistency_embedding", 0.0)),
					"consistency_embedding_stripped": float(metrics.get("consistency_embedding_stripped", 0.0)),
					"proxy_subtask_consistency_mean": subtask_consistency_mean,
					"role_expert_count_execute": int(role_expert_counts.get(Role.EXECUTE, 0)),
					"role_expert_count_aggregate": int(role_expert_counts.get(Role.AGGREGATE, 0)),
					"subtask_expert_total": int(subtask_total),
					**subtask_expert_metrics,
					"main_gate_pass": bool(metrics.get("main_gate_pass", False)),
					"elapsed_ms": float(metrics.get("elapsed_ms", 0.0) or 0.0),
					"attempt_step_last": int(attempt_step),
					"attempt_baseline_last": float(baseline_proxy),
					"overall_count": int(summary.get("overall", {}).get("count", 0)) if isinstance(summary, dict) else None,
					"overall_reward_mean": float(summary.get("overall", {}).get("train_mean", {}).get("reward", 0.0)) if isinstance(summary, dict) else None,
				},
			)

			global_step += 1
			sample_cursor += 1
			if int(args.save_every) > 0 and (global_step % int(args.save_every) == 0):
				trainer_state = {
					"run_ts": run_ts,
					"global_step": int(global_step),
					"attempt_step": int(attempt_step),
					"baseline": float(baseline),
					"baseline_proxy": float(baseline_proxy),
					"sample_cursor": int(sample_cursor),
					"wandb_run_id": str(wandb_run_id),
					"py_random_state": random.getstate(),
					"torch_rng_state": torch.get_rng_state(),
					"cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
				}
				ckpt_dir = (ckpt_root / checkpoint_name) if checkpoint_overwrite else (ckpt_root / f"step_{global_step:06d}")
				_save_checkpoint(
					ckpt_dir=ckpt_dir,
					router=router,
					title_embedder=title_embedder,
					model=model,
					opt_router=opt_router,
					opt_lora=opt_lora,
					trainer_state=trainer_state,
				)
				last_saved_step = int(global_step)

		if bool(getattr(args, "save_at_end", False)) and int(global_step) > int(last_saved_step):
			trainer_state = {
				"run_ts": run_ts,
				"global_step": int(global_step),
				"attempt_step": int(attempt_step),
				"baseline": float(baseline),
				"baseline_proxy": float(baseline_proxy),
				"sample_cursor": int(sample_cursor),
				"wandb_run_id": str(wandb_run_id),
				"py_random_state": random.getstate(),
				"torch_rng_state": torch.get_rng_state(),
				"cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
			}
			ckpt_dir = (ckpt_root / checkpoint_name) if checkpoint_overwrite else (ckpt_root / f"step_{global_step:06d}")
			_save_checkpoint(
				ckpt_dir=ckpt_dir,
				router=router,
				title_embedder=title_embedder,
				model=model,
				opt_router=opt_router,
				opt_lora=opt_lora,
				trainer_state=trainer_state,
			)
			last_saved_step = int(global_step)

		print(str(run_root))
	finally:
		if gen_pool is not None:
			try:
				gen_pool.close()
			except Exception:
				pass
		if logger is not None:
			logger.close()


if __name__ == "__main__":
	main()
