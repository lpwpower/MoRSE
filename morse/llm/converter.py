"""Utilities for turning Task-Action graphs into MacNet-friendly configs."""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class NodeMetadata:
	"""Structured information about a single Task-Action node."""

	node_id: int
	original_id: str
	title: str
	description: str

	def to_prompt(self, task_description: str) -> str:
		"""Craft the per-node system prompt."""
		safe_title = self.title or f"Node {self.node_id}"
		safe_desc = self.description or "Follow the project requirements."
		return "\n".join(
			[
				"You are the sole Python engineer responsible for a subtask in a directed task graph.",
				f"Overall SRDD task: {task_description.strip()}",
				f"Your assigned subtask ({self.original_id} - {safe_title}): {safe_desc}",
				"Always build from the provided repository snapshot; respect previous functionality while extending it.",
				"Output the entire repository content, file by file, strictly in markdown code blocks as required by MacNet.",
				"Every file must end with the '.py' suffix, include executable Python code, and rely on the Python standard library only.",
				"If upstream code is empty, start a fresh but runnable Python project that satisfies your subtask and lays groundwork for downstream tasks.",
				"Never emit TODO/PASS placeholders or pseudocode comments—implement the full logic.",
			]
		)


@dataclass
class TaskGraphSpecification:
	"""Aggregated metadata derived from a Task-Action graph."""

	task_name: str
	task_description: str
	node_order: List[str]
	edge_strings: List[str]
	node_metadata: Dict[int, NodeMetadata]

	def slug(self) -> str:
		"""Return a filesystem-safe slug for run directories."""
		base = self.task_name or "task"
		slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_")
		return slug.lower() or "task"


def _load_graph(path: Path) -> Dict[str, object]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _collect_edges(node_entries: Iterable[dict]) -> List[Tuple[str, str]]:
	"""Collect all edges from `depends_on` lists."""
	edges: set[Tuple[str, str]] = set()
	for node in node_entries:
		node_id = str(node.get("node_id") or node.get("id") or "").strip()
		if not node_id:
			continue
		for parent in node.get("depends_on") or []:
			parent_id = str(parent).strip()
			if parent_id:
				edges.add((parent_id, node_id))
	return sorted(edges)


def _topological_order(nodes: Dict[str, dict], edges: List[Tuple[str, str]]) -> List[str]:
	adjacency: Dict[str, List[str]] = defaultdict(list)
	indegree: Dict[str, int] = {node_id: 0 for node_id in nodes}
	for parent, child in edges:
		adjacency[parent].append(child)
		indegree[child] = indegree.get(child, 0) + 1

	queue = deque(sorted([node_id for node_id, deg in indegree.items() if deg == 0]))
	ordering: List[str] = []
	while queue:
		current = queue.popleft()
		ordering.append(current)
		for successor in sorted(adjacency.get(current, [])):
			indegree[successor] -= 1
			if indegree[successor] == 0:
				queue.append(successor)

	if len(ordering) != len(nodes):
		raise ValueError("Task graph contains cycles or disconnected references.")
	return ordering


def convert_taskgraph(graph_path: Path) -> TaskGraphSpecification:
	"""Convert a Task-Action graph JSON to a specification consumable by MacNet."""
	payload = _load_graph(graph_path)
	node_entries = payload.get("nodes", [])
	nodes_by_id = {
		str(entry.get("node_id") or entry.get("id") or entry.get("label")).strip(): entry
		for entry in node_entries
		if entry.get("node_id") or entry.get("id") or entry.get("label")
	}

	edges = _collect_edges(node_entries)
	ordering = _topological_order(nodes_by_id, edges)
	id_map = {node_id: idx for idx, node_id in enumerate(ordering)}

	edge_strings = [f"{id_map[parent]}->{id_map[child]}" for parent, child in edges]
	metadata: Dict[int, NodeMetadata] = {}
	for original_id, idx in id_map.items():
		entry = nodes_by_id.get(original_id, {})
		metadata[idx] = NodeMetadata(
			node_id=idx,
			original_id=original_id,
			title=str(entry.get("title") or entry.get("name") or original_id),
			description=str(entry.get("description") or "").strip(),
		)

	return TaskGraphSpecification(
		task_name=str(payload.get("task_name") or "Task").strip(),
		task_description=str(payload.get("task_description") or "").strip(),
		node_order=ordering,
		edge_strings=edge_strings,
		node_metadata=metadata,
	)
