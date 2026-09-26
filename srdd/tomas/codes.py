"""Minimal code container/parser compatible with our prompts.

Parses a multi-file response formatted as:

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
from typing import Dict


_BLOCK_RE = re.compile(r"(.+?)\n```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)
_FILENAME_RE = re.compile(r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)")
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


@dataclass
class Codes:
	"""Stores a repository snapshot as a mapping of filename -> content."""

	generated_content: str = ""
	codebooks: Dict[str, str] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not self.generated_content:
			return
		for match in _BLOCK_RE.finditer(self.generated_content):
			header = match.group(1).strip()
			body = match.group(2)
			filename = ""
			for cand in _FILENAME_RE.findall(header):
				filename = cand.lower()
			if not filename:
				continue
			self.codebooks[filename] = _normalize_code(body)

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
