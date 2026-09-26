"""Lenient code container/parser with fallback heuristics.

Primary format:
FILENAME.py
```python
<code>
```
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


_STRICT_BLOCK_RE = re.compile(r"(.+?)\n```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_FILENAME_RE = re.compile(r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)")
_CODE_START_RE = re.compile(r"(?m)^\s*(def |class |import |from )")
_MAIN_PY_RE = re.compile(r"\bmain\s*[.\-_]?\s*p\s*y\b", re.IGNORECASE)
_MAX_BASENAME_LEN = 200


def _safe_filename_for_fs(filename: str, *, max_basename_len: int = _MAX_BASENAME_LEN, force_hash: bool = False) -> str:
	"""Return a filesystem-safe filename.

	Some model outputs can include extremely long file basenames (e.g. 100s of chars),
	which will raise OSError(36) on write. We keep the extension and append a stable hash.
	"""
	normalized = (filename or "").replace("\\", "/")
	p = Path(normalized)
	basename = p.name

	if not force_hash and len(basename) <= max_basename_len:
		return normalized

	stem = basename
	suffix = ""
	if "." in basename:
		stem, suffix = basename.rsplit(".", 1)
		suffix = "." + suffix

	digest = hashlib.sha1(basename.encode("utf-8")).hexdigest()[:12]
	keep = max_basename_len - len(suffix) - len(digest) - 2  # "__"
	if keep < 1:
		keep = 1
	new_basename = f"{stem[:keep]}__{digest}{suffix}".lower()

	if p.parent and str(p.parent) not in (".", ""):
		return str(p.parent / new_basename).replace("\\", "/")
	return new_basename


def _normalize_code(code: str) -> str:
	lines = [line.rstrip("\n") for line in code.splitlines()]
	# Keep blank lines, but trim trailing whitespace-only lines at ends.
	while lines and not lines[0].strip():
		lines.pop(0)
	while lines and not lines[-1].strip():
		lines.pop()
	return "\n".join(lines) + ("\n" if lines else "")


def _extract_filename_from_header(header: str) -> str:
	filename = ""
	for cand in _FILENAME_RE.findall(header):
		filename = cand.lower()
	return filename


def _looks_like_main(text: str) -> bool:
	if not text:
		return False
	return bool(_MAIN_PY_RE.search(text))


def _guess_filename(header: str, body: str) -> str:
	filename = _extract_filename_from_header(header)
	if filename:
		return filename
	if _looks_like_main(header) or _looks_like_main(body):
		return "main.py"
	lowered = body.lower()
	if "__name__" in lowered or "def main" in lowered:
		return "main.py"
	return ""


def _header_before(text: str, start: int, *, max_lines: int = 4) -> str:
	if start <= 0:
		return ""
	prefix = text[:start]
	lines = prefix.splitlines()
	if not lines:
		return ""
	return "\n".join(lines[-max_lines:])


def _iter_fenced_blocks(text: str) -> Iterator[Tuple[int, str]]:
	for match in _FENCE_RE.finditer(text):
		yield match.start(), match.group(1)

	# If there are no closed fences, try to capture an unclosed trailing fence.
	if "```" in text and not _FENCE_RE.search(text):
		start = text.find("```")
		if start != -1:
			line_end = text.find("\n", start)
			if line_end != -1 and line_end + 1 < len(text):
				yield start, text[line_end + 1 :]


def _score_code(code: str) -> int:
	score = 0
	lowered = code.lower()
	if "__name__" in lowered:
		score += 50
	if "def main" in lowered:
		score += 20
	if "class " in lowered:
		score += 5
	score += min(len(code), 10000) // 100
	return score


def _extract_loose_code(text: str) -> str:
	lines = text.splitlines()
	for idx, line in enumerate(lines):
		if _CODE_START_RE.search(line):
			return "\n".join(lines[idx:])
	return text


def _fallback_parse(text: str) -> Dict[str, str]:
	candidates: list[tuple[str, str, int]] = []
	for start, body in _iter_fenced_blocks(text):
		code = _normalize_code(body)
		if not code.strip():
			continue
		header = _header_before(text, start)
		filename = _guess_filename(header, code)
		candidates.append((filename, code, _score_code(code)))

	codebooks: Dict[str, str] = {}
	if candidates:
		for filename, code, score in candidates:
			if not filename:
				continue
			prev = codebooks.get(filename)
			if prev is None or score > _score_code(prev):
				codebooks[filename] = code
		if codebooks:
			return codebooks
		best = max(candidates, key=lambda item: (item[2], len(item[1])))
		codebooks["main.py"] = best[1]
		return codebooks

	if _CODE_START_RE.search(text):
		codebooks["main.py"] = _normalize_code(_extract_loose_code(text))
	return codebooks


@dataclass
class Codes:
	"""Stores a repository snapshot as a mapping of filename -> content."""

	generated_content: str = ""
	codebooks: Dict[str, str] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not self.generated_content:
			return
		for match in _STRICT_BLOCK_RE.finditer(self.generated_content):
			header = match.group(1).strip()
			body = match.group(2)
			filename = _extract_filename_from_header(header)
			if not filename:
				continue
			self.codebooks[filename] = _normalize_code(body)
		if not self.codebooks or all(not content.strip() for content in self.codebooks.values()):
			self.codebooks = _fallback_parse(self.generated_content)

	def snapshot(self) -> str:
		"""Render the repository back into the same multi-file markdown format."""
		parts = []
		for filename in sorted(self.codebooks):
			parts.append(f"{filename}\n```python\n{self.codebooks[filename]}```\n")
		return "\n".join(parts).strip() + "\n"

	def snapshot_for_prompt(self) -> str:
		"""Render a repository snapshot for prompts (no markdown fences to avoid fence-collapse)."""
		parts = []
		for filename in sorted(self.codebooks):
			parts.append(f"=== {filename} ===\n{self.codebooks[filename]}\n")
		return "\n".join(parts).strip() + ("\n" if parts else "")

	def write_to_directory(self, directory: Path) -> Path:
		directory.mkdir(parents=True, exist_ok=True)
		written: set[str] = set()
		for filename, content in self.codebooks.items():
			target = directory / filename
			target.parent.mkdir(parents=True, exist_ok=True)
			try:
				target.write_text(content, encoding="utf-8")
				written.add(filename.replace("\\", "/").lower())
			except OSError as exc:
				if getattr(exc, "errno", None) != 36:
					raise
				safe_name = _safe_filename_for_fs(filename, force_hash=True)
				try:
					safe_target = directory / safe_name
					safe_target.parent.mkdir(parents=True, exist_ok=True)
					safe_target.write_text(content, encoding="utf-8")
					written.add(safe_name.replace("\\", "/").lower())
					print(f"[Codes] filename too long; wrote {filename!r} as {safe_name!r}", file=sys.stderr)
				except OSError:
					print(f"[Codes] filename too long; skipped {filename!r}", file=sys.stderr)

		# Remove stale files/subdirs.
		for existing in directory.rglob("*"):
			if existing.is_dir():
				continue
			rel = str(existing.relative_to(directory)).replace("\\", "/").lower()
			if rel not in written:
				existing.unlink(missing_ok=True)
		for existing in sorted(directory.glob("*")):
			if existing.is_dir() and not any(existing.rglob("*")):
				shutil.rmtree(existing, ignore_errors=True)
		return directory
