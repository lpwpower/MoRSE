from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class WandBConfig:
	project: str
	entity: str = ""
	run_name: str = ""
	mode: str = "offline"  # offline | online
	run_id: str = ""
	resume: str = ""  # "", "allow", "must"


class MetricsLogger:
	def __init__(
		self,
		*,
		run_root: Path,
		jsonl_path: Path,
		wandb: Optional[WandBConfig] = None,
		config: Optional[Dict[str, Any]] = None,
	) -> None:
		self._jsonl_path = Path(jsonl_path)
		self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
		self._jsonl_handle = self._jsonl_path.open("a", encoding="utf-8")

		self._wandb_enabled = wandb is not None
		self._wandb = None
		self._wandb_run = None
		if wandb is not None:
			try:
				import wandb as _wandb  # type: ignore
			except Exception as exc:
				print(
					f"[warn] wandb enabled but not installed/working ({exc}); continuing without wandb. "
					"Install with: pip install wandb",
					file=sys.stderr,
				)
				self._wandb_enabled = False
				return
			init_kwargs: Dict[str, Any] = {
				"project": str(wandb.project),
				"dir": str(run_root),
				"mode": str(wandb.mode),
			}
			if wandb.entity:
				init_kwargs["entity"] = str(wandb.entity)
			if wandb.run_name:
				init_kwargs["name"] = str(wandb.run_name)
			if wandb.run_id:
				init_kwargs["id"] = str(wandb.run_id)
			if wandb.resume:
				init_kwargs["resume"] = str(wandb.resume)
			if config is not None:
				init_kwargs["config"] = config
			self._wandb = _wandb
			self._wandb_run = _wandb.init(**init_kwargs)
			_wandb.define_metric("attempt/*", step_metric="attempt_step")
			_wandb.define_metric("sample/*", step_metric="sample_step")
			_wandb.define_metric("attempt_step")
			_wandb.define_metric("sample_step")

	def close(self) -> None:
		try:
			self._jsonl_handle.close()
		finally:
			if self._wandb_run is not None:
				try:
					self._wandb_run.finish()
				except Exception:
					pass

	def _write_jsonl(self, payload: Dict[str, Any]) -> None:
		payload = dict(payload)
		payload.setdefault("time", float(time.time()))
		self._jsonl_handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
		self._jsonl_handle.flush()

	def log_attempt(self, *, attempt_step: int, metrics: Dict[str, Any]) -> None:
		record = {"type": "attempt", "attempt_step": int(attempt_step), **metrics}
		self._write_jsonl(record)
		if self._wandb_run is None:
			return
		wlog = {"attempt_step": int(attempt_step)}
		for k, v in metrics.items():
			wlog[f"attempt/{k}"] = v
		self._wandb.log(wlog)

	def log_sample(self, *, sample_step: int, metrics: Dict[str, Any]) -> None:
		record = {"type": "sample", "sample_step": int(sample_step), **metrics}
		self._write_jsonl(record)
		if self._wandb_run is None:
			return
		wlog = {"sample_step": int(sample_step)}
		for k, v in metrics.items():
			wlog[f"sample/{k}"] = v
		self._wandb.log(wlog)
