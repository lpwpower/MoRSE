import argparse
import csv
import os
import re
import shlex
import signal
import subprocess
import time
import sys
from pathlib import Path
from typing import Iterable, Tuple

_TRIPLE_RE = re.compile(r"'''|\"\"\"")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from srdd.eval import srdd_embedding  # noqa: E402


def read_sample(csv_path: str, sample_name: str) -> Tuple[str, str]:
	with open(csv_path, newline='', encoding='utf-8') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row.get("Name") == sample_name:
				return row.get("Description", ""), row.get("Category", "")
	raise ValueError(f"Sample '{sample_name}' not found in {csv_path}")


def list_python_files(folder: str) -> Iterable[str]:
	for root, _dirs, files in os.walk(folder):
		for filename in files:
			if filename.endswith('.py'):
				yield os.path.join(root, filename)


def read_code(folder: str) -> str:
	contents = []
	for filepath in sorted(list_python_files(folder)):
		with open(filepath, 'r', encoding='utf-8') as handle:
			contents.append(handle.read())
	return "\n".join(contents)


def strip_comments_docstrings(py: str) -> str:
	return srdd_embedding.strip_comments_docstrings(py)


def completeness_score(code: str) -> float:
	lines = code.splitlines()
	filtered = []
	for line in lines:
		lowered = line.lower()
		if any(keyword in lowered for keyword in ("password", "passenger", "passed", "passes")):
			continue
		filtered.append(lowered)
	markers = [line for line in filtered if re.search(r"\bpass\b", line) or re.search(r"\btodo\b", line)]
	return 0.0 if markers else 1.0


def run_with_timeout(command: str, timeout: int = 3, cwd: str | None = None) -> Tuple[bool, str]:
	def terminate(process: subprocess.Popen) -> None:
		if process.poll() is None:
			if os.name == 'nt':
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

	if os.name == 'nt':
		process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
								   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, cwd=cwd)
	else:
		process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
								   preexec_fn=os.setsid, cwd=cwd)
	try:
		out, err = process.communicate(timeout=timeout)
	except subprocess.TimeoutExpired:
		terminate(process)
		return True, "timeout"
	success = process.returncode == 0
	stdout_text = out.decode('utf-8', errors='ignore')
	stderr_text = err.decode('utf-8', errors='ignore')
	if not success and "EOFError: EOF when reading a line" in stderr_text:
		return True, "eof"
	output = stdout_text if success else stderr_text
	return success, output.strip()


def executability_score(folder: str) -> Tuple[float, str]:
	main_path = None
	for filepath in list_python_files(folder):
		if os.path.basename(filepath) == "main.py":
			main_path = filepath
			break
	if not main_path:
		return 0.0, "main.py not found"
	# `folder` may be a relative path; always run from an absolute CWD and pass a path
	# relative to that CWD to avoid duplicating prefixes like `<cwd>/<folder>/<folder>/main.py`.
	folder_abs = os.path.abspath(folder)
	main_abs = os.path.abspath(main_path)
	main_rel = os.path.relpath(main_abs, start=folder_abs)
	command = f"python3 {shlex.quote(main_rel)}"
	success, details = run_with_timeout(command, cwd=folder_abs)
	return (1.0 if success else 0.0), details


def consistency_score(description: str, code: str) -> float:
	return srdd_embedding.consistency(description, code, stripped=False)


def consistency_stripped_score(description: str, code: str) -> float:
	return srdd_embedding.consistency(description, code, stripped=True)


def evaluate(sample_name: str, csv_path: str, code_folder: str, report_path: str) -> None:
	description, category = read_sample(csv_path, sample_name)
	if not os.path.isdir(code_folder):
		raise FileNotFoundError(f"Code directory '{code_folder}' does not exist")

	code = read_code(code_folder)
	completeness = completeness_score(code)
	executability, execution_details = executability_score(code_folder)
	consistency = consistency_score(description, code)
	consistency_stripped = consistency_stripped_score(description, code)
	# Use consistency_stripped (code with docstrings removed) for both ECI variants to
	# prevent models from gaming cosine similarity by copying the task description into
	# docstrings. The raw `consistency` field is still reported below for reference.
	eci_mean = (float(executability) + float(completeness) + float(consistency_stripped)) / 3.0
	eci_product = float(executability) * float(completeness) * float(consistency_stripped)

	os.makedirs(os.path.dirname(report_path), exist_ok=True)
	with open(report_path, 'w', encoding='utf-8') as report:
		report.write(f"Sample: {sample_name}\n")
		report.write(f"Category: {category}\n")
		report.write(f"Completeness: {completeness:.3f}\n")
		report.write(f"Executability: {executability:.3f}\n")
		report.write(f"Consistency: {consistency:.3f}\n")
		report.write(f"Consistency_stripped: {consistency_stripped:.3f}\n")
		report.write(f"ECI_mean: {eci_mean:.3f}\n")
		report.write(f"ECI_product: {eci_product:.3f}\n")
		report.write("Execution details:\n")
		report.write(execution_details or "(no output)")


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate MacNet output against SRDD metrics")
	parser.add_argument("sample", help="Sample name in SRDD.csv")
	parser.add_argument("csv_path", help="Path to SRDD.csv")
	parser.add_argument("code_dir", help="Directory containing generated code")
	parser.add_argument("report_path", help="Path to write evaluation report")
	args = parser.parse_args()

	evaluate(args.sample, args.csv_path, args.code_dir, args.report_path)


if __name__ == "__main__":
	main()
