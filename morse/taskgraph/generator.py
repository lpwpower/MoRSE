"""Graph generation backends."""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from .graph import SubtaskGraph, SubtaskNode

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET_SAMPLE = "Mystic_Maze"
DEFAULT_GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-06-05"


class GraphGenerationError(RuntimeError):
	"""Raised when a backend cannot produce a well-formed graph."""


def _extract_first_json_object(text: str) -> Dict[str, Any]:
	stack: List[int] = []
	start_idx: int | None = None
	for idx, char in enumerate(text):
		if char == "{":
			if not stack:
				start_idx = idx
			stack.append(idx)
		elif char == "}":
			if stack:
				stack.pop()
				if not stack and start_idx is not None:
					candidate = text[start_idx : idx + 1]
					try:
						return json.loads(candidate)
					except json.JSONDecodeError:
						continue
	raise GraphGenerationError("LLM response did not contain a valid JSON object")


def _normalize_nodes(raw_nodes: Iterable[Dict[str, Any]]) -> List[SubtaskNode]:
	nodes: List[SubtaskNode] = []
	seen_ids: set[str] = set()
	for idx, raw in enumerate(raw_nodes, start=1):
		if not isinstance(raw, dict):
			continue
		node_id = str(
			raw.get("id")
			or raw.get("node_id")
			or raw.get("label")
			or f"n{idx}"
		).strip()
		if not node_id:
			node_id = f"n{idx}"
		if node_id in seen_ids:
			node_id = f"{node_id}_{idx}"
		seen_ids.add(node_id)
		title = str(raw.get("title") or raw.get("name") or node_id).strip()
		description = str(raw.get("description") or title).strip()
		deps_raw = raw.get("depends_on") or raw.get("dependencies") or []
		if isinstance(deps_raw, str):
			depends_on = [dep.strip() for dep in deps_raw.split(",") if dep.strip()]
		else:
			depends_on = [str(dep).strip() for dep in deps_raw if dep]
		nodes.append(
			SubtaskNode(
				node_id=node_id,
				title=title,
				description=description,
				depends_on=depends_on,
			)
		)
	if not nodes:
		raise GraphGenerationError("No nodes found in generated JSON")
	return nodes


def _build_prompt(task_name: str, task_description: str) -> str:
	return textwrap.dedent(
		f"""
		You are an expert in software engineering planning. Your task is to create detailed fine-grained subtasks for developing a pythonic software following the general task description.
		Break the job into 5-8 engineering subtasks that describe what developers must do and how to cooperate by division of labor to produce that artifact.

		For every subtask include:
		  - id: short identifier (n1, n2, ...)
		  - title: concise engineering action (e.g., \"Design ECS architecture\", \"Implement asset loader\")
		  - description: 1-2 sentences describing the activity, demand, and expected intermediate deliverable for this subtask (keep it short)
		  - depends_on: list of parent ids that must be finished first (each id must refer to another node in this JSON; root subtasks use an empty list; avoid cycles).

		Rules:
		  - Subtasks must be actionable developer work items (requirements analysis, architecture, coding, testing, packaging, documentation).
		  - Assume the project must be standard-library-only Python. If the task implies graphics/UI, plan for a terminal/ASCII implementation (avoid pygame/PyQt/tkinter and other third-party engines).
		  - If a node fans out to multiple downstream tasks, make sure the parent produces shared artifacts/guidance so the child subtasks can execute in parallel consistently.
		  - If a node depends on multiple parents, its work should directly require and unify concrete artifacts from every upstream dependency (it should not be a generic or unrelated task).
		  - Respect dependencies so later implementation steps reference earlier analysis/design steps when appropriate.
		  - Output MUST be valid JSON only: start immediately with '{{' and end immediately after the final '}}'.
		  - Do NOT output markdown, code fences (```), commentary, leading/trailing text, or multiple JSON objects.
		  - Stop immediately after emitting the closing brace of the JSON object.
		  - Ensure the combined subtask descriptions fully cover every requirement and detail stated in the original task description, and make it clear in which subtask each major requirement is addressed.

		Output schema (example, placeholders shown; follow this exact structure):
		{{
		  "task_name": "Sample_Task",
		  "task_description": "Complete description here.",
		  "nodes": [
			{{"id": "n1", "title": "Summary of subtask1 here.", "description": "Concrete activity, demand, and expected intermediate deliverable for this subtask.", "depends_on": []}},
			{{"id": "n2", "title": "Summary of subtask1 here.", "description": "Concrete activity, demand, and expected intermediate deliverable for this subtask.", "depends_on": ["n1"]}}
		  ]
		}}

		The example is purely to show formatting; generate fresh subtasks tailored to the provided task.
		Return JSON only. Start with '{{' on the first character, end with '}}' on the last character.

		Task name: {task_name}
		Task description: {task_description}
		JSON:
		"""
	).strip()


class LLMTaskGraphGenerator:
	"""LLM-backed planner that prompts a seq2seq checkpoint via transformers."""

	def __init__(
		self,
		model_name: str = DEFAULT_MODEL_NAME,
		max_new_tokens: int = 384,
		temperature: float = 0.2,
		device: int = -1,
		device_map: str | None = None,
		pipeline_task: str = "text-generation",
		model_kwargs: Dict[str, Any] | None = None,
		prompt_builder=None,
	) -> None:
		try:
			from transformers import pipeline
		except ModuleNotFoundError as exc:  # pragma: no cover
			raise GraphGenerationError(
				"transformers is required for the LLM planner. Install Task-Action-MAS/requirements.txt"
			) from exc

		self.model_name = model_name
		self.max_new_tokens = max_new_tokens
		self.temperature = temperature
		self.pipeline_task = pipeline_task
		# Injectable decomposition prompt: defaults to the software-engineering
		# prompt (SRDD); pass prompt_builder(task_name, task_description)->str for
		# other domains (e.g. SciCode scientific computing).
		self._prompt_builder = prompt_builder or _build_prompt
		self.last_prompt: str | None = None
		self.last_response_text: str | None = None
		self.device_map = device_map
		self._call_kwargs: Dict[str, Any] = {}
		if pipeline_task == "text-generation":
			self._call_kwargs["return_full_text"] = False
		pipeline_kwargs: Dict[str, Any] = {
			"task": pipeline_task,
			"model": model_name,
			"model_kwargs": dict(model_kwargs or {}),
		}
		if device_map:
			# transformers.pipeline forbids passing device_map twice:
			# - as pipeline(..., device_map=...)
			# - and inside model_kwargs={"device_map": ...}
			#
			# Some callers may include device_map in model_kwargs for older patterns,
			# so we defensively strip it here.
			pipeline_kwargs["model_kwargs"].pop("device_map", None)
			pipeline_kwargs["device_map"] = device_map
		else:
			pipeline_kwargs["device"] = device
		try:
			self._pipeline = pipeline(**pipeline_kwargs)
		except TypeError:
			pipeline_kwargs.pop("device_map", None)
			pipeline_kwargs["device"] = device
			self._pipeline = pipeline(**pipeline_kwargs)
		tokenizer = getattr(self._pipeline, "tokenizer", None)
		if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
			tokenizer.pad_token = tokenizer.eos_token

	def generate(self, task_name: str, task_description: str) -> SubtaskGraph:
		prompt = self._prompt_builder(task_name, task_description)
		self.last_prompt = prompt
		do_sample = self.temperature > 0
		call_kwargs: Dict[str, Any] = dict(self._call_kwargs)
		call_kwargs.update(
			{
				"max_new_tokens": self.max_new_tokens,
				"do_sample": do_sample,
			}
		)
		# Avoid warnings when sampling is disabled: set sampling-only flags to defaults.
		if do_sample:
			call_kwargs["temperature"] = self.temperature
		else:
			call_kwargs["temperature"] = 1.0
			call_kwargs["top_p"] = 1.0
			call_kwargs["top_k"] = 50

		response_text = self._pipeline(prompt, **call_kwargs)[0]["generated_text"]
		if self.pipeline_task == "text-generation" and prompt in response_text:
			response_text = response_text.split(prompt, 1)[1].lstrip()
		self.last_response_text = response_text
		try:
			parsed = _extract_first_json_object(response_text)
			nodes = _normalize_nodes(parsed.get("nodes", []))
			task_title = parsed.get("task_name") or task_name
			task_desc = parsed.get("task_description") or task_description
			return SubtaskGraph(task_title.strip(), task_desc.strip(), nodes)
		except Exception as exc:
			snippet = (response_text or "").strip().replace("\n", "\\n")
			if len(snippet) > 800:
				snippet = snippet[:800] + "..."
			raise GraphGenerationError(
				f"LLM graph generation failed to produce valid JSON: {exc}. Response snippet: {snippet}"
			) from exc


class GeminiTaskGraphGenerator:
	"""Planner backed by Google Gemini via google-generativeai."""

	def __init__(
		self,
		model_name: str = DEFAULT_GEMINI_MODEL_NAME,
		*,
		api_key: str | None = None,
		generation_config: Dict[str, Any] | None = None,
	) -> None:
		try:
			import google.generativeai as genai
		except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
			raise GraphGenerationError(
				"google-generativeai is required for the Gemini planner. Install it via 'pip install google-generativeai'."
			) from exc

		self.model_name = model_name or DEFAULT_GEMINI_MODEL_NAME
		self._api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
		if not self._api_key:
			raise GraphGenerationError(
				"Google API key not provided. Pass google_api_key or set the GOOGLE_API_KEY/GEMINI_API_KEY environment variable."
			)
		genai.configure(api_key=self._api_key)
		self._model = genai.GenerativeModel(self.model_name)
		self._generation_config = generation_config or {}

	def _coalesce_text(self, response: Any) -> str:
		text = getattr(response, "text", None)
		if text:
			return text
		fragments: List[str] = []
		candidates = getattr(response, "candidates", None) or []
		for candidate in candidates:
			content = getattr(candidate, "content", None)
			parts = getattr(content, "parts", None) if content is not None else None
			if parts is None:
				parts = getattr(candidate, "parts", None)
			if not parts:
				continue
			for part in parts:
				part_text = getattr(part, "text", None)
				if part_text:
					fragments.append(part_text)
				elif isinstance(part, dict) and part.get("text"):
					fragments.append(str(part["text"]))
		return "\n".join(fragments).strip()

	def generate(self, task_name: str, task_description: str) -> SubtaskGraph:
		prompt = _build_prompt(task_name, task_description)
		gen_kwargs: Dict[str, Any] = {}
		if self._generation_config:
			gen_kwargs["generation_config"] = self._generation_config
		try:
			response = self._model.generate_content(prompt, **gen_kwargs)
		except Exception as exc:  # pragma: no cover - network call
			raise GraphGenerationError(f"Gemini generation failed: {exc}") from exc
		response_text = self._coalesce_text(response)
		if not response_text:
			raise GraphGenerationError("Gemini response did not contain any text output")
		parsed = _extract_first_json_object(response_text)
		nodes = _normalize_nodes(parsed.get("nodes", []))
		task_title = parsed.get("task_name") or task_name
		task_desc = parsed.get("task_description") or task_description
		return SubtaskGraph(task_title.strip(), task_desc.strip(), nodes)


@dataclass
class HeuristicTaskGraphGenerator:
	"""Fallback planner that uses deterministic heuristics."""

	default_steps: Sequence[str] = (
		"Context Analysis",
		"Mechanic Design",
		"Content Production",
		"Implementation",
		"Playtesting",
	)

	def generate(self, task_name: str, task_description: str) -> SubtaskGraph:
		nodes: List[SubtaskNode] = []
		previous_id: str | None = None
		for idx, title in enumerate(self.default_steps, start=1):
			node_id = f"n{idx}"
			deps = [previous_id] if previous_id else []
			description = (
				f"{title} for {task_name}. "
				f"Grounded in the description: {task_description[:140]}..."
			)
			nodes.append(
				SubtaskNode(
					node_id=node_id,
					title=title,
					description=description,
					depends_on=deps,
				)
			)
			previous_id = node_id
		return SubtaskGraph(task_name, task_description, nodes)


class GraphPlanner:
	"""High-level orchestrator that prefers LLM output but can fall back."""

	def __init__(
		self,
		model_name: str = DEFAULT_MODEL_NAME,
		allow_fallback: bool = True,
		fallback_only: bool = False,
		pipeline_task: str = "text-generation",
		model_kwargs: Dict[str, Any] | None = None,
		provider: str = "hf",
		google_api_key: str | None = None,
		gemini_generation_config: Dict[str, Any] | None = None,
		**llm_kwargs: Any,
	) -> None:
		self.allow_fallback = allow_fallback
		self.provider = provider.lower()
		self._heuristic = HeuristicTaskGraphGenerator()
		self._llm: Any | None = None
		self._llm_error: Exception | None = None
		self._last_backend: str = "heuristic"
		self._used_fallback: bool = False
		if not fallback_only:
			try:
				if self.provider == "gemini":
					actual_model_name = (
						model_name if model_name != DEFAULT_MODEL_NAME else DEFAULT_GEMINI_MODEL_NAME
					)
					self._llm = GeminiTaskGraphGenerator(
						model_name=actual_model_name,
						api_key=google_api_key,
						generation_config=gemini_generation_config,
					)
				elif self.provider in {"hf", "huggingface"}:
					self._llm = LLMTaskGraphGenerator(
						model_name=model_name,
						pipeline_task=pipeline_task,
						model_kwargs=model_kwargs,
						**llm_kwargs,
					)
				else:  # pragma: no cover - defensive guard
					raise ValueError(f"Unsupported LLM provider: {provider}")
			except Exception as exc:  # pragma: no cover
				self._llm_error = exc
				logger.warning("LLM planner unavailable (%s)", exc)
		if fallback_only or self._llm is None:
			logger.info("Using heuristic planner only")

	def generate(self, task_name: str, task_description: str) -> SubtaskGraph:
		if self._llm is not None:
			try:
				graph = self._llm.generate(task_name, task_description)
				self._last_backend = "llm"
				self._used_fallback = False
				return graph
			except Exception as exc:
				self._llm_error = exc
				logger.warning("LLM planner failed: %s", exc)
				if not self.allow_fallback:
					raise
		logger.info("Falling back to heuristic planner")
		self._last_backend = "heuristic"
		self._used_fallback = True
		return self._heuristic.generate(task_name, task_description)

	@property
	def last_error(self) -> Exception | None:
		return self._llm_error

	@property
	def last_prompt(self) -> str | None:
		llm = self._llm
		return getattr(llm, "last_prompt", None) if llm is not None else None

	@property
	def last_response_text(self) -> str | None:
		llm = self._llm
		return getattr(llm, "last_response_text", None) if llm is not None else None

	@property
	def last_backend(self) -> str:
		"""One of: 'llm' or 'heuristic' for the most recent generate() call."""
		return self._last_backend

	@property
	def used_fallback(self) -> bool:
		"""True when last_backend=='heuristic' due to LLM failure/unavailable."""
		return self._used_fallback
