from __future__ import annotations

import os
import math
import re
import subprocess
import sys
import py_compile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from srdd.eval import srdd_embedding  # noqa: E402


_WORD_RE = re.compile(r"[a-zA-Z]+")
_TRIPLE_RE = re.compile(r"'''|\"\"\"")


def tokenize(text: str) -> Counter:
	return Counter(_WORD_RE.findall((text or "").lower()))


def cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
	if not counter_a or not counter_b:
		return 0.0
	intersection = set(counter_a) & set(counter_b)
	numerator = sum(counter_a[token] * counter_b[token] for token in intersection)
	sum_a = sum(value**2 for value in counter_a.values())
	sum_b = sum(value**2 for value in counter_b.values())
	return numerator / math.sqrt(sum_a * sum_b) if sum_a and sum_b else 0.0


def strip_comments_docstrings(py: str) -> str:
	lines = []
	in_triple = False
	triple = None
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

def completeness_from_code(code: str) -> float:
	"""Heuristic completeness proxy (same rule as SRDD evaluator): penalize 'pass'/'todo' markers."""
	lines = (code or "").splitlines()
	filtered = []
	for line in lines:
		lowered = line.lower()
		# Avoid false positives.
		if any(keyword in lowered for keyword in ("password", "passenger", "passed", "passes")):
			continue
		filtered.append(lowered)
	markers = [line for line in filtered if "pass" in line or "todo" in line]
	return 0.0 if markers else 1.0


def consistency_stripped_from_code(task_description: str, code: str) -> float:
	try:
		return srdd_embedding.consistency(task_description, code or "", stripped=True)
	except Exception:
		stripped = strip_comments_docstrings(code or "")
		return cosine_similarity(tokenize(task_description), tokenize(stripped))

def consistency_embedding_from_code(task_description: str, code: str) -> float:
	"""Embedding-based consistency on raw code (no stripping)."""
	try:
		return srdd_embedding.consistency(task_description, code or "", stripped=False)
	except Exception:
		return cosine_similarity(tokenize(task_description), tokenize(code or ""))


def read_repo_code(repo_dir: Path) -> str:
	parts = []
	for path in sorted(repo_dir.rglob("*.py")):
		try:
			parts.append(path.read_text(encoding="utf-8", errors="ignore"))
		except Exception:
			continue
	return "\n".join(parts)


def consistency_stripped(task_description: str, repo_dir: Path) -> float:
	code = read_repo_code(repo_dir)
	return consistency_stripped_from_code(task_description, code)


def parse_srdd_report(report_path: Path) -> Dict[str, float]:
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


def main_py_gate(repo_dir: Path, *, timeout_s: float = 2.0) -> Tuple[bool, Dict[str, Any]]:
	"""Hard gate: main.py exists + compiles + does not crash immediately when executed.

	Notes:
	- Many SRDD tasks are interactive; if execution times out, we treat it as "can run"
	  (it started without an immediate crash).
	"""
	main_path = repo_dir / "main.py"
	details: Dict[str, Any] = {
		"main_exists": False,
		"main_compile_ok": False,
		"main_run_ok": False,
		"main_run_timeout": False,
		"main_run_returncode": None,
	}
	if not main_path.exists():
		return False, details
	details["main_exists"] = True

	try:
		py_compile.compile(str(main_path), doraise=True)
		details["main_compile_ok"] = True
	except Exception:
		return False, details

	env = dict(os.environ)
	env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
	env["PYTHONPATH"] = str(repo_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

	try:
		proc = subprocess.run(
			[sys.executable, str(main_path)],
			cwd=str(repo_dir),
			env=env,
			stdin=subprocess.DEVNULL,
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
			timeout=float(timeout_s),
			check=False,
		)
		details["main_run_returncode"] = int(proc.returncode)
		if proc.returncode == 0:
			details["main_run_ok"] = True
			return True, details
		return False, details
	except subprocess.TimeoutExpired:
		details["main_run_timeout"] = True
		# Consider timeout a pass: main.py did not crash immediately.
		return True, details


def compute_reward(
	*,
	task_description: str,
	repo_dir: Path,
	srdd_report_path: Path,
	w_exec: float = 0.5,
	w_comp: float = 0.5,
	w_cons_strip: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
	m = parse_srdd_report(srdd_report_path)
	exec_score = float(m.get("executability", 0.0))
	comp_score = float(m.get("completeness", 0.0))
	cons_srdd = float(m.get("consistency", 0.0))
	cons_strip_srdd = float(m.get("consistency_stripped", 0.0))

	code_text = read_repo_code(repo_dir)
	cons_embed = float(consistency_embedding_from_code(task_description, code_text))
	cons_strip_embed = float(consistency_stripped_from_code(task_description, code_text))
	gate_pass, gate_details = main_py_gate(repo_dir)

	r = w_exec * exec_score + w_comp * comp_score + w_cons_strip * cons_strip_embed
	if not gate_pass:
		r = 0.0

	metrics: Dict[str, Any] = {
		"executability": exec_score,
		"completeness": comp_score,
		# Consistency from SRDD evaluator report.
		"consistency_srdd": cons_srdd,
		"consistency_stripped_srdd": cons_strip_srdd,
		# Embedding-based consistency (raw + stripped).
		"consistency_embedding": cons_embed,
		"consistency_embedding_stripped": cons_strip_embed,
		# Backward-compatible fields: training reward uses stripped embedding.
		"consistency": cons_embed,
		"consistency_stripped": cons_strip_embed,
		"eci_mean": (float(exec_score) + float(comp_score) + float(cons_strip_embed)) / 3.0,
		"reward": float(r),
	}
	metrics.update(gate_details)
	metrics["main_gate_pass"] = bool(gate_pass)
	return float(r), metrics
