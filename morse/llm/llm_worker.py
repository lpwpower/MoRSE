"""LLM inference worker process.

This module is started as a separate Python process to isolate CUDA context.
Parent communicates via line-delimited JSON over stdin/stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional


CUDA_FATAL_MARKERS = (
	"cuda error",
	"unspecified launch failure",
	"device-side assert",
	"cublas",
	"cudnn",
)


def _as_max_memory(raw: Optional[str]) -> Optional[Dict[object, str]]:
	if not raw:
		return None
	try:
		parsed = json.loads(raw)
	except Exception:
		return None
	if not isinstance(parsed, dict):
		return None
	out: Dict[object, str] = {}
	for k, v in parsed.items():
		key: object = k
		if isinstance(k, str) and k.isdigit():
			key = int(k)
		out[key] = str(v)
	return out


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="TaskGraph MacNet runner LLM worker (subprocess).")
	p.add_argument("--model-name", type=str, required=True)
	p.add_argument("--pipeline-task", type=str, default="text-generation")
	p.add_argument("--device", type=int, default=-1)
	p.add_argument("--device-map", type=str, default=None)
	p.add_argument("--torch-dtype", type=str, default=None)
	p.add_argument("--max-memory-json", type=str, default=None)
	p.add_argument("--max-new-tokens", type=int, default=2048)
	p.add_argument("--temperature", type=float, default=0.2)
	p.add_argument("--top-p", type=float, default=0.95)
	p.add_argument("--repetition-penalty", type=float, default=1.1)
	p.add_argument("--no-repeat-ngram-size", type=int, default=3)
	return p.parse_args()


def main() -> None:
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

	args = parse_args()

	# Lazy imports inside worker to keep parent light.
	from morse.llm.hf_llm import HFGenerationConfig, HFTextGenerator  # noqa: E402

	cfg = HFGenerationConfig(
		model_name=args.model_name,
		pipeline_task=args.pipeline_task,
		device=args.device,
		device_map=args.device_map,
		torch_dtype=args.torch_dtype,
		max_memory=_as_max_memory(args.max_memory_json),
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		repetition_penalty=args.repetition_penalty,
		no_repeat_ngram_size=args.no_repeat_ngram_size,
	)
	generator = HFTextGenerator(cfg)

	for line in sys.stdin:
		line = line.strip()
		if not line:
			continue
		try:
			req = json.loads(line)
		except Exception:
			continue
		req_id = req.get("id")
		op = req.get("op")
		if op == "shutdown":
			break
		if op == "ping":
			sys.stdout.write(json.dumps({"id": req_id, "ok": True, "pong": True}, ensure_ascii=False) + "\n")
			sys.stdout.flush()
			continue
		if op == "count_tokens":
			text = req.get("text") or ""
			sys.stdout.write(
				json.dumps({"id": req_id, "ok": True, "count": generator.count_tokens(text)}, ensure_ascii=False) + "\n"
			)
			sys.stdout.flush()
			continue
		if op != "generate":
			sys.stdout.write(json.dumps({"id": req_id, "ok": False, "error": f"unknown op: {op}"}, ensure_ascii=False) + "\n")
			sys.stdout.flush()
			continue

		prompt = req.get("prompt") or ""
		try:
			# Allow per-call overrides.
			if isinstance(req.get("max_new_tokens"), int):
				generator.cfg.max_new_tokens = int(req["max_new_tokens"])
			if isinstance(req.get("temperature"), (int, float)):
				generator.cfg.temperature = float(req["temperature"])
			if isinstance(req.get("top_p"), (int, float)):
				generator.cfg.top_p = float(req["top_p"])
			text = generator.generate(prompt)
			sys.stdout.write(json.dumps({"id": req_id, "ok": True, "text": text}, ensure_ascii=False) + "\n")
			sys.stdout.flush()
		except Exception as exc:  # pragma: no cover
			msg = str(exc)
			lower = msg.lower()
			fatal = any(m in lower for m in CUDA_FATAL_MARKERS)
			sys.stdout.write(json.dumps({"id": req_id, "ok": False, "error": msg, "fatal": fatal}, ensure_ascii=False) + "\n")
			sys.stdout.flush()
			if fatal:
				# Exit so parent can restart a fresh CUDA context.
				raise


if __name__ == "__main__":
	try:
		main()
	except Exception:
		# Non-zero exit for parent to detect restart condition.
		raise
