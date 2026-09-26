"""Shared embedding-based consistency for SRDD-style evaluation.

The approach:
- compute cosine similarity between embeddings of (description/text) and code,
- provide a "stripped" variant where code comments/docstrings are removed,
- default to the locally configured embedding model (if available).

The embedding model can be selected via environment variables (see
``_load_embedding_model_spec``) or, optionally, via a YAML config file pointed
to by ``SRDD_EMBEDDING_CONFIG_YAML`` (expects an ``embedding`` section with
``local_model_path`` or ``model``). If nothing is configured, a small
sentence-transformers model is used as a safe fallback.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import yaml
from transformers import AutoModel, AutoTokenizer

# ── Compatibility shim: gte-Qwen2-7B-instruct's remote modeling code calls
# DynamicCache.get_usable_length(), which was removed in transformers >=4.50.
# Restore it as an alias for get_seq_length so the embedder forward pass works.
try:
	from transformers.cache_utils import DynamicCache as _DynCache  # type: ignore
	if not hasattr(_DynCache, "get_usable_length"):
		def _dc_get_usable_length(self, new_seq_length: int = 0, layer_idx: int = 0) -> int:  # noqa: E501
			try:
				return int(self.get_seq_length(layer_idx))
			except Exception:
				try:
					return int(self.get_seq_length())
				except Exception:
					return 0
		_DynCache.get_usable_length = _dc_get_usable_length  # type: ignore[attr-defined]
except Exception:
	pass

_TRIPLE_RE = re.compile(r"'''|\"\"\"")


def strip_comments_docstrings(py: str) -> str:
	"""Best-effort removal of # comments and triple-quoted blocks."""
	lines: List[str] = []
	in_triple = False
	triple: Optional[str] = None
	for raw in (py or "").splitlines():
		line = raw.rstrip("\n")
		if not in_triple:
			if line.lstrip().startswith("#"):
				continue
			m = _TRIPLE_RE.search(line)
			if m:
				triple = m.group(0)
				if line.count(triple) >= 2:
					continue
				in_triple = True
				continue
			lines.append(line)
		else:
			if triple and triple in line:
				in_triple = False
				triple = None
	return "\n".join(lines)


def _resolve_hf_model_path(path: str) -> str:
	if not path:
		raise ValueError("Embedding model path is empty.")
	if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
		return path
	snapshots_dir = os.path.join(path, "snapshots")
	if os.path.isdir(snapshots_dir):
		subdirs = [
			os.path.join(snapshots_dir, d)
			for d in os.listdir(snapshots_dir)
			if os.path.isdir(os.path.join(snapshots_dir, d))
		]
		subdirs.sort()
		for d in reversed(subdirs):
			if os.path.isfile(os.path.join(d, "config.json")):
				return d
	# If it's not a local folder, assume HF repo id and let transformers handle it.
	if "/" in path and not os.path.isabs(path):
		return path
	return path


def _default_config_paths() -> List[Path]:
	# Optional YAML config: only honored when explicitly provided via env var.
	cfg_env = os.environ.get("SRDD_EMBEDDING_CONFIG_YAML", "")
	if cfg_env:
		return [Path(cfg_env).expanduser()]
	return []


def _load_embedding_model_spec() -> Tuple[str, int, str, Optional[str]]:
	"""Return (model_or_path, max_length, dtype, device_map)."""
	# Environment overrides.
	env_model = (os.environ.get("SRDD_EMBEDDING_MODEL") or "").strip()
	env_local = (os.environ.get("SRDD_EMBEDDING_LOCAL_PATH") or "").strip()
	env_max_len = (os.environ.get("SRDD_EMBEDDING_MAX_LENGTH") or "").strip()
	env_dtype = (os.environ.get("SRDD_EMBEDDING_DTYPE") or "").strip().lower()
	env_device_map = (os.environ.get("SRDD_EMBEDDING_DEVICE_MAP") or "").strip() or None

	model_spec = env_local or env_model
	# `SRDD_EMBEDDING_MAX_LENGTH=0|none|-1` means: don't impose an extra truncation cap;
	# use the embedding model's own max context length.
	if env_max_len:
		if env_max_len.lower() in {"0", "-1", "none", "null", "no_truncate", "no-truncate"}:
			max_length = 0
		else:
			max_length = int(env_max_len)
	else:
		max_length = 0
	dtype = env_dtype or "auto"

	if model_spec:
		return model_spec, max_length, dtype, env_device_map

	for cfg_path in _default_config_paths():
		if not cfg_path or str(cfg_path) in {".", ""}:
			continue
		if not cfg_path.exists():
			continue
		try:
			cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
			emb = cfg.get("embedding", {}) or {}
			model_spec = (emb.get("local_model_path") or emb.get("model") or "").strip()
			if model_spec:
				return model_spec, max_length, dtype, env_device_map
		except Exception:
			continue

	# Safe fallback that doesn't require external repos on disk (may download if not cached).
	return "sentence-transformers/all-MiniLM-L6-v2", max_length, dtype, env_device_map


def _pick_device() -> str:
	override = (os.environ.get("SRDD_EMBEDDING_DEVICE") or "").strip()
	if override:
		return override
	if not torch.cuda.is_available():
		return "cpu"
	# Under torchrun/distributed, pin each rank's embedder to its own local GPU
	# so ranks 1..N-1 don't all land on cuda:0.
	local_rank = (os.environ.get("LOCAL_RANK") or "").strip()
	if local_rank:
		try:
			return f"cuda:{int(local_rank)}"
		except ValueError:
			pass
	return "cuda"


def _parse_dtype(dtype: str, device: str) -> torch.dtype:
	dtype = (dtype or "auto").lower()
	if dtype == "auto":
		return torch.bfloat16 if device.startswith("cuda") else torch.float32
	if dtype == "float16":
		return torch.float16
	if dtype == "bfloat16":
		return torch.bfloat16
	if dtype == "float32":
		return torch.float32
	raise ValueError(f"Unsupported dtype: {dtype}")

def _infer_model_max_length(tokenizer: object, model: object, *, fallback: int = 1024) -> int:
	candidates: List[int] = []
	try:
		tok_max = int(getattr(tokenizer, "model_max_length", 0) or 0)
		# Many tokenizers use a huge sentinel when max length is "unknown".
		if 0 < tok_max < 1_000_000:
			candidates.append(tok_max)
	except Exception:
		pass

	try:
		cfg = getattr(model, "config", None)
		if cfg is not None:
			for key in ("max_position_embeddings", "n_positions", "seq_length"):
				val = getattr(cfg, key, None)
				if isinstance(val, int) and val > 0:
					candidates.append(int(val))
	except Exception:
		pass

	return min(candidates) if candidates else int(fallback)


@dataclass
class Embedder:
	model_name: str
	device: str
	input_device: str
	max_length: int
	tokenizer: object
	model: object

	def embed_texts(self, texts: List[str], *, batch_size: int = 4) -> torch.Tensor:
		vecs: List[torch.Tensor] = []
		for i in range(0, len(texts), batch_size):
			batch = texts[i : i + batch_size]
			batch = [t if (t or "").strip() else "none" for t in batch]
			enc = self.tokenizer(
				[t.replace("\n", " ") for t in batch],
				padding=True,
				truncation=True,
				max_length=self.max_length,
				return_tensors="pt",
			)
			enc = {k: v.to(self.input_device) for k, v in enc.items()}
			with torch.no_grad():
				out = self.model(**enc)
				last = out.last_hidden_state
				mask = enc.get("attention_mask")
				if mask is None:
					mask = torch.ones(last.shape[:2], device=last.device, dtype=torch.long)
				else:
					mask = mask.to(last.device)
				mask = mask.unsqueeze(-1)
				pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
				pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
				vecs.append(pooled.detach().cpu())
		return torch.cat(vecs, dim=0) if vecs else torch.empty((0, 0))


def _truthy_env(name: str) -> bool:
	return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
	model_spec, max_length, dtype_str, device_map = _load_embedding_model_spec()
	device = _pick_device()
	torch_dtype = _parse_dtype(dtype_str, device=device)

	use_int8 = _truthy_env("SRDD_EMBEDDING_INT8")
	use_4bit = _truthy_env("SRDD_EMBEDDING_4BIT")
	trust_remote = _truthy_env("SRDD_EMBEDDING_TRUST_REMOTE_CODE") or True

	model_name = model_spec
	if os.path.isdir(model_spec):
		model_name = _resolve_hf_model_path(model_spec)

	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
	if getattr(tokenizer, "pad_token_id", None) is None:
		if getattr(tokenizer, "eos_token_id", None) is not None:
			tokenizer.pad_token = tokenizer.eos_token
		elif getattr(tokenizer, "sep_token_id", None) is not None:
			tokenizer.pad_token = tokenizer.sep_token
		else:
			tokenizer.add_special_tokens({"pad_token": "[PAD]"})

	input_device = device
	quant_config = None
	if use_int8 or use_4bit:
		from transformers import BitsAndBytesConfig  # lazy; only if requested
		if use_4bit:
			quant_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_compute_dtype=torch_dtype,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_use_double_quant=True,
			)
		else:
			quant_config = BitsAndBytesConfig(load_in_8bit=True)

		# Quantized models cannot use .to(); must pin via device_map.
		if device.startswith("cuda"):
			if ":" in device:
				pin_device = device
			else:
				pin_device = f"cuda:{torch.cuda.current_device()}"
			device_map = {"": pin_device}
			input_device = pin_device
		else:
			device_map = {"": device}
	elif device_map is None and device.startswith("cuda"):
		device_map = "auto"

	if quant_config is not None:
		model = AutoModel.from_pretrained(
			model_name,
			quantization_config=quant_config,
			device_map=device_map,
			trust_remote_code=trust_remote,
		)
	elif device_map:
		model = AutoModel.from_pretrained(
			model_name,
			dtype=torch_dtype,
			device_map=device_map,
			trust_remote_code=trust_remote,
		)
		hf_map = getattr(model, "hf_device_map", None)
		if isinstance(hf_map, dict):
			for mapped in hf_map.values():
				if isinstance(mapped, str) and mapped.startswith("cuda"):
					input_device = mapped
					break
	else:
		model = AutoModel.from_pretrained(
			model_name,
			dtype=torch_dtype,
			trust_remote_code=trust_remote,
		)
		model.to(device)

	model.eval()
	try:
		model.resize_token_embeddings(len(tokenizer))
	except Exception:
		pass

	# If max_length=0, don't cap below the model/tokenizer limits.
	if max_length <= 0:
		max_length = _infer_model_max_length(tokenizer, model, fallback=1024)

	return Embedder(
		model_name=str(model_name),
		device=device,
		input_device=input_device,
		max_length=max_length,
		tokenizer=tokenizer,
		model=model,
	)


_embed_lock = threading.Lock()


def cosine_similarity(text_a: str, text_b: str, *, batch_size: int = 4) -> float:
	with _embed_lock:
		embedder = get_embedder()
		vecs = embedder.embed_texts([text_a or "", text_b or ""], batch_size=batch_size)
		if vecs.shape[0] != 2:
			return 0.0
		return float(torch.dot(vecs[0], vecs[1]).item())


def consistency(description: str, code: str, *, stripped: bool) -> float:
	if stripped:
		code = strip_comments_docstrings(code)
	return cosine_similarity(description or "", code or "")


def read_code_from_repo(repo_dir: Path, *, exts: Iterable[str] = (".py",)) -> str:
	parts: List[str] = []
	for path in sorted(repo_dir.rglob("*")):
		if not path.is_file():
			continue
		if path.suffix not in set(exts):
			continue
		try:
			parts.append(path.read_text(encoding="utf-8", errors="ignore"))
		except Exception:
			continue
	return "\n".join(parts)
