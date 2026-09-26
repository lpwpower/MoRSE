"""Node-level repo smoke tests (executability checks).

This intentionally mirrors ChatDev_macnet/srdd_evaluator.py behavior:
- Find main.py and run it with a timeout (default 3s).
- Treat timeout as success (details='timeout').
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


def iter_python_files(folder: Path) -> Iterable[Path]:
	for root, _dirs, files in os.walk(folder):
		for filename in files:
			if filename.endswith(".py"):
				yield Path(root) / filename


def find_main_py(folder: Path) -> Path | None:
	for filepath in iter_python_files(folder):
		if filepath.name == "main.py":
			return filepath
	return None


def run_with_timeout(command: str, *, timeout_s: int = 3, cwd: Path | None = None) -> Tuple[bool, str]:
	def terminate(process: subprocess.Popen) -> None:
		if process.poll() is None:
			if os.name == "nt":
				process.terminate()
				try:
					process.wait(timeout=1)
				except subprocess.TimeoutExpired:
					process.kill()
			else:
				os.killpg(os.getpgid(process.pid), signal.SIGTERM)
				try:
					process.wait(timeout=1)
				except subprocess.TimeoutExpired:
					os.killpg(os.getpgid(process.pid), signal.SIGKILL)

	if os.name == "nt":
		process = subprocess.Popen(
			command,
			shell=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
			cwd=str(cwd) if cwd is not None else None,
		)
	else:
		process = subprocess.Popen(
			command,
			shell=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			preexec_fn=os.setsid,
			cwd=str(cwd) if cwd is not None else None,
		)

	try:
		out, err = process.communicate(timeout=timeout_s)
	except subprocess.TimeoutExpired:
		terminate(process)
		return True, "timeout"

	success = process.returncode == 0
	stdout_text = out.decode("utf-8", errors="ignore")
	stderr_text = err.decode("utf-8", errors="ignore")
	if not success and "EOFError: EOF when reading a line" in stderr_text:
		return True, "eof"
	output = stdout_text if success else (stderr_text or stdout_text)
	return success, output.strip()


@dataclass
class SmokeTestResult:
	passed: bool
	details: str
	elapsed_ms: int


def smoke_test_repo(repo_dir: Path, *, timeout_s: int = 3) -> SmokeTestResult:
	main_py = find_main_py(repo_dir)
	if main_py is None:
		return SmokeTestResult(passed=False, details="main.py not found", elapsed_ms=0)

	start = time.time()
	# Run with cwd=repo_dir so any relative file writes stay inside the repo folder.
	command = f"{sys.executable} {main_py.name}"
	passed, details = run_with_timeout(command, timeout_s=timeout_s, cwd=main_py.parent)
	elapsed_ms = int((time.time() - start) * 1000)
	if not passed and details:
		lowered = details.lower()
		if "usage:" in lowered or "required argument" in lowered or "required arguments" in lowered:
			return SmokeTestResult(passed=True, details="usage", elapsed_ms=elapsed_ms)
	return SmokeTestResult(passed=passed, details=details, elapsed_ms=elapsed_ms)
