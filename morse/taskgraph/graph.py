"""Data structures for representing and serializing task graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List


class GraphVisualizationError(RuntimeError):
	"""Raised when rendering a task graph diagram fails."""


def _resolve_output_path(path: Path, fmt: str) -> Path:
	"""Ensure the output path contains the correct suffix for the format."""

	if path.suffix:
		return path
	return path.with_suffix(f".{fmt}")


@dataclass
class SubtaskNode:
	"""Represents a single subtask node in the graph."""

	node_id: str
	title: str
	description: str
	depends_on: List[str] = field(default_factory=list)

	def to_dict(self) -> Dict[str, object]:
		data = asdict(self)
		data["depends_on"] = list(dict.fromkeys(self.depends_on or []))
		return data


@dataclass
class SubtaskGraph:
	"""Container for a task and its decomposed subtasks."""

	task_name: str
	task_description: str
	nodes: List[SubtaskNode]

	def edges(self) -> Iterable[Dict[str, str]]:
		"""Yield directed edges implied by dependency lists."""
		for node in self.nodes:
			for parent in node.depends_on:
				yield {"from": parent, "to": node.node_id, "relation": "depends_on"}

	def to_dict(self) -> Dict[str, object]:
		return {
			"task_name": self.task_name,
			"task_description": self.task_description,
			"nodes": [node.to_dict() for node in self.nodes],
			"edges": list(self.edges()),
		}

	def save_json(self, path: Path) -> Path:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		with path.open("w", encoding="utf-8") as fp:
			json.dump(self.to_dict(), fp, ensure_ascii=False, indent=2)
		return path

	def short_summary(self) -> str:
		node_count = len(self.nodes)
		edge_count = sum(1 for _ in self.edges())
		return f"SubtaskGraph(nodes={node_count}, edges={edge_count})"


def render_graph_diagram(
	graph: SubtaskGraph,
	output_path: Path,
	*,
	fmt: str | None = None,
	rankdir: str = "LR",
) -> Path:
	"""Render the given ``SubtaskGraph`` to an image using graphviz.

	Parameters
	----------
	graph:
		Graph to visualize.
	output_path:
		Target path for the rendered artifact. If no suffix is provided it
		defaults to ``.png`` or the provided ``fmt``.
	fmt:
		Optional explicit graphviz output format (png, pdf, svg, ...).
	rankdir:
		graphviz rank direction directive, defaults to left-to-right.
	"""

	try:
		import graphviz
	except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
		raise GraphVisualizationError(
			"The 'graphviz' package is required to draw task graphs. Install it via 'pip install graphviz'."
		) from exc

	resolved_fmt = fmt or (output_path.suffix.lstrip(".") if output_path.suffix else "png")
	if not resolved_fmt:
		resolved_fmt = "png"

	target_path = _resolve_output_path(Path(output_path), resolved_fmt)
	target_path.parent.mkdir(parents=True, exist_ok=True)

	dot = graphviz.Digraph(comment=graph.task_name, format=resolved_fmt)
	dot.attr(rankdir=rankdir)
	dot.attr("node", shape="box", style="rounded,filled", fillcolor="#eef5ff", color="#4f81bd")
	dot.attr("edge", color="#4f81bd")

	for node in graph.nodes:
		label = f"{node.node_id}\\n{node.title}" if node.title else node.node_id
		dot.node(node.node_id, label=label)

	for edge in graph.edges():
		dot.edge(edge["from"], edge["to"], label=edge.get("relation", ""))

	data = dot.pipe(format=resolved_fmt)
	target_path.write_bytes(data)
	return target_path
