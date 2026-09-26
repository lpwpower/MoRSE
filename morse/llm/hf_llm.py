"""HuggingFace Transformers LLM wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass
class HFGenerationConfig:
	model_name: str
	pipeline_task: str = "text-generation"
	device: int = -1
	device_map: Optional[str] = None
	max_memory: Optional[Dict[Union[int, str], str]] = None
	torch_dtype: Optional[str] = None
	max_new_tokens: int = 2048
	temperature: float = 0.2
	top_p: float = 0.95
	repetition_penalty: float = 1.0
	no_repeat_ngram_size: int = 0


class HFTextGenerator:
	def __init__(self, cfg: HFGenerationConfig):
		try:
			from transformers import pipeline
		except ModuleNotFoundError as exc:  # pragma: no cover
			raise ModuleNotFoundError(
				"Missing dependency: transformers. Install requirements.txt or `pip install transformers`."
			) from exc

		model_kwargs: Dict[str, Any] = {}
		if cfg.torch_dtype:
			try:
				import torch
			except ModuleNotFoundError:
				torch = None
			if torch is not None and hasattr(torch, cfg.torch_dtype):
				model_kwargs["torch_dtype"] = getattr(torch, cfg.torch_dtype)
		if cfg.max_memory:
			model_kwargs["max_memory"] = cfg.max_memory

		self.cfg = cfg
		trust_remote_code = os.environ.get("HF_TRUST_REMOTE_CODE", "1").strip().lower() not in {"0", "false", "no"}
		pipe_kwargs: Dict[str, Any] = {
			"task": cfg.pipeline_task,
			"model": cfg.model_name,
			"model_kwargs": model_kwargs,
			"trust_remote_code": trust_remote_code,
		}
		if cfg.device_map:
			pipe_kwargs["device_map"] = cfg.device_map
		else:
			pipe_kwargs["device"] = cfg.device

		try:
			self._pipe = pipeline(**pipe_kwargs)
		except TypeError:
			pipe_kwargs.pop("device_map", None)
			pipe_kwargs["device"] = cfg.device
			self._pipe = pipeline(**pipe_kwargs)
		if cfg.pipeline_task == "text-generation":
			self._call_kwargs = {"return_full_text": False}
		else:
			self._call_kwargs = {}

		tokenizer = getattr(self._pipe, "tokenizer", None)
		self.tokenizer = tokenizer
		if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
			tokenizer.pad_token = tokenizer.eos_token
		self._chat_template_mode = os.environ.get("HF_USE_CHAT_TEMPLATE", "auto").strip().lower()
		self._chat_system_prompt = os.environ.get("HF_SYSTEM_PROMPT", "").strip()

	def _should_use_chat_template(self) -> bool:
		if self.cfg.pipeline_task != "text-generation":
			return False
		if self._chat_template_mode in {"0", "false", "no", "off"}:
			return False
		tok = getattr(self, "tokenizer", None)
		if tok is None or not hasattr(tok, "apply_chat_template"):
			return False
		if self._chat_template_mode == "auto" and not getattr(tok, "chat_template", None):
			return False
		return True

	def _format_prompt_for_model(self, prompt: str) -> str:
		if not self._should_use_chat_template():
			return prompt
		tok = self.tokenizer
		messages = []
		if self._chat_system_prompt:
			messages.append({"role": "system", "content": self._chat_system_prompt})
		messages.append({"role": "user", "content": prompt})
		try:
			return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		except Exception:
			# Some multimodal templates expect list-of-content blocks.
			try:
				messages_mm = []
				if self._chat_system_prompt:
					messages_mm.append({"role": "system", "content": [{"type": "text", "text": self._chat_system_prompt}]})
				messages_mm.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
				return tok.apply_chat_template(messages_mm, tokenize=False, add_generation_prompt=True)
			except Exception:
				return prompt

	def generate(self, prompt: str) -> str:
		model_prompt = self._format_prompt_for_model(prompt)
		do_sample = float(self.cfg.temperature) > 0.0
		call_kwargs = dict(self._call_kwargs)
		if do_sample:
			call_kwargs["temperature"] = self.cfg.temperature
			call_kwargs["top_p"] = self.cfg.top_p
		else:
			call_kwargs["temperature"] = 1.0
			call_kwargs["top_p"] = 1.0
			call_kwargs["top_k"] = 50
		out = self._pipe(
			model_prompt,
			max_new_tokens=self.cfg.max_new_tokens,
			do_sample=do_sample,
			repetition_penalty=self.cfg.repetition_penalty,
			no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
			**call_kwargs,
		)
		if not out:
			return ""
		if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
			return out[0]["generated_text"] or ""
		if isinstance(out, list) and out and isinstance(out[0], str):
			return out[0]
		return str(out)

	def count_tokens(self, text: str) -> int:
		tok = getattr(self, "tokenizer", None)
		if tok is None:
			return 0
		try:
			return len(tok.encode(text))
		except Exception:
			return 0


class HFSubprocessTextGenerator:
	"""Run HF generation in a child process and restart on CUDA failures."""

	def __init__(
		self,
		cfg: HFGenerationConfig,
		*,
		restart_on_cuda: bool = True,
		startup_timeout_s: int = 300,
	):
		self.cfg = cfg
		self._restart_on_cuda = restart_on_cuda
		self._startup_timeout_s = startup_timeout_s
		self._proc: subprocess.Popen[str] | None = None
		self._next_id = 1

		# Token counting in parent (cheap, CPU).
		self.tokenizer = None
		try:
			from transformers import AutoTokenizer

			tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
			if getattr(tok, "pad_token_id", None) is None:
				tok.pad_token = tok.eos_token
			self.tokenizer = tok
		except Exception:
			self.tokenizer = None

		self._start_worker()

	def _start_worker(self) -> None:
		if self._proc is not None:
			return
		env = os.environ.copy()
		env.setdefault("TOKENIZERS_PARALLELISM", "false")
		env.setdefault("PYTHONUNBUFFERED", "1")

		max_memory_json = None
		if self.cfg.max_memory:
			# Ensure JSON-serializable keys (ints become strings).
			payload: Dict[str, str] = {}
			for k, v in self.cfg.max_memory.items():
				payload[str(k)] = str(v)
			max_memory_json = json.dumps(payload)

		cmd = [
			sys.executable,
			"-u",
			"-m",
			"morse.llm.llm_worker",
			"--model-name",
			self.cfg.model_name,
			"--pipeline-task",
			self.cfg.pipeline_task,
			"--device",
			str(self.cfg.device),
			"--max-new-tokens",
			str(self.cfg.max_new_tokens),
			"--temperature",
			str(self.cfg.temperature),
			"--top-p",
			str(self.cfg.top_p),
			"--repetition-penalty",
			str(self.cfg.repetition_penalty),
			"--no-repeat-ngram-size",
			str(self.cfg.no_repeat_ngram_size),
		]
		if self.cfg.device_map:
			cmd += ["--device-map", str(self.cfg.device_map)]
		if self.cfg.torch_dtype:
			cmd += ["--torch-dtype", str(self.cfg.torch_dtype)]
		if max_memory_json:
			cmd += ["--max-memory-json", max_memory_json]

		self._proc = subprocess.Popen(
			cmd,
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=None,  # inherit
			text=True,
			bufsize=1,
			env=env,
		)

		# Ping for readiness (model load can take minutes).
		deadline = time.time() + max(self._startup_timeout_s, 10)
		while True:
			try:
				self._rpc({"op": "ping"})
				break
			except Exception:
				if time.time() > deadline:
					self.restart()
					raise RuntimeError("LLM worker failed to start within timeout.")
				time.sleep(1)

	def restart(self) -> None:
		proc = self._proc
		self._proc = None
		if proc is None:
			self._start_worker()
			return
		try:
			if proc.stdin:
				proc.stdin.write(json.dumps({"op": "shutdown", "id": -1}) + "\n")
				proc.stdin.flush()
		except Exception:
			pass
		try:
			proc.terminate()
		except Exception:
			pass
		try:
			proc.wait(timeout=5)
		except Exception:
			try:
				proc.kill()
			except Exception:
				pass
		self._start_worker()

	def _rpc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
		if self._proc is None:
			self._start_worker()
		assert self._proc is not None
		if self._proc.poll() is not None:
			# Worker already dead.
			self.restart()
			raise RuntimeError("LLM worker exited unexpectedly; restarted.")
		payload = dict(payload)
		payload.setdefault("id", self._next_id)
		self._next_id += 1
		line = json.dumps(payload, ensure_ascii=False)
		assert self._proc.stdin is not None
		assert self._proc.stdout is not None
		try:
			self._proc.stdin.write(line + "\n")
			self._proc.stdin.flush()
		except BrokenPipeError:
			self.restart()
			raise RuntimeError("LLM worker pipe broken; restarted.")

		resp_line = self._proc.stdout.readline()
		if not resp_line:
			# EOF
			self.restart()
			raise RuntimeError("LLM worker returned EOF; restarted.")
		try:
			resp = json.loads(resp_line)
		except Exception:
			raise RuntimeError(f"Invalid worker response: {resp_line[:200]}")
		return resp

	def generate(self, prompt: str) -> str:
		resp = self._rpc({"op": "generate", "prompt": prompt})
		if resp.get("ok") is True:
			return resp.get("text") or ""
		err = str(resp.get("error") or "unknown error")
		fatal = bool(resp.get("fatal"))
		if fatal and self._restart_on_cuda:
			self.restart()
			raise RuntimeError(f"LLM worker CUDA failure; restarted. Error: {err}")
		raise RuntimeError(err)

	def count_tokens(self, text: str) -> int:
		tok = getattr(self, "tokenizer", None)
		if tok is None:
			return 0
		try:
			return len(tok.encode(text))
		except Exception:
			return 0

	def __del__(self) -> None:  # pragma: no cover
		try:
			if self._proc is not None:
				self._proc.terminate()
		except Exception:
			pass
