"""SciCode -> TaskGraph -> DAG execution pipeline.

This script mirrors the SRDD task-graph runner structure, but swaps in
SciCode-style step prompts (template-based) and SciCode step/general tests.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn


SCRIPT_PATH = Path(__file__).resolve()
SCICODE_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCICODE_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morse.llm.converter import convert_taskgraph  # noqa: E402
from morse.llm.hf_llm import (  # noqa: E402
    HFGenerationConfig,
    HFSubprocessTextGenerator,
    HFTextGenerator,
)
from morse.mole.mole_generator import GenerationConfig as MoLEGenerationConfig  # noqa: E402
from morse.mole.mole_generator import MoLEGenerator  # noqa: E402
from morse.mole.mole_lora import LoRAConfig, inject_mole_lora  # noqa: E402
from morse.mole.router import SubtaskRouter, SubtaskRouterConfig, normalize_title_embedding  # noqa: E402


BACKGOUND_PROMPT_TEMPLATE = (SCICODE_ROOT / "eval" / "data" / "multistep_template.txt").read_text(encoding="utf-8")
DEFAULT_PROMPT_TEMPLATE = (SCICODE_ROOT / "eval" / "data" / "background_comment_template.txt").read_text(encoding="utf-8")

DEFAULT_DATASET = SCICODE_ROOT / "data" / "problems_dev.jsonl"
DEFAULT_GRAPH_ROOT = SCICODE_ROOT / "data" / "taskgraphs"
DEFAULT_OUTPUT_ROOT = SCICODE_ROOT / "runs" / "scicode_taskgraph_runs"
DEFAULT_H5PY_FILE = SCICODE_ROOT / "eval" / "data" / "test_data.h5"

SPECIAL_STEP_FILES: Dict[Tuple[str, str], Path] = {
    ("13", "13.6"): SCICODE_ROOT / "eval" / "data" / "13.6.txt",
    ("62", "62.1"): SCICODE_ROOT / "eval" / "data" / "62.1.txt",
    ("76", "76.3"): SCICODE_ROOT / "eval" / "data" / "76.3.txt",
}

SCICODE_RETRY_GUIDELINES = (
    "Regenerate the current step following original SciCode requirements only:\n"
    "1) Implement only the current step according to the provided function header.\n"
    "2) Do not include previous-step function code.\n"
    "3) Do not include example usage or test code.\n"
    "4) Keep dependencies limited to the provided dependency list and do not place those dependencies at the top of the code.\n"
    "5) Return Python code in a single ```python``` block."
)


@dataclass
class ScriptRunResult:
    passed: bool
    status: str
    return_code: int
    elapsed_ms: int
    stdout: str
    stderr: str
    script_path: Path


class TitleEmbedder(nn.Module):
    """Embed subtask text with frozen token embeddings + trainable projection."""

    def __init__(self, *, model: nn.Module, tokenizer, out_dim: int):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        hidden = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "n_embd", 0))
        if hidden <= 0:
            raise ValueError("Could not determine hidden size for TitleEmbedder.")
        self.proj = nn.Linear(hidden, int(out_dim))

    @torch.no_grad()
    def _mean_token_emb(self, text: str) -> torch.Tensor:
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
        input_ids = enc["input_ids"].to(self._model.device)
        emb = self._model.get_input_embeddings()(input_ids)
        if self._tokenizer.pad_token_id is None:
            return emb.mean(dim=1)
        mask = (input_ids != self._tokenizer.pad_token_id).float().unsqueeze(-1)
        den = torch.clamp(mask.sum(dim=1), min=1.0)
        return (emb * mask).sum(dim=1) / den

    def forward(self, text: str) -> torch.Tensor:
        mean_emb = self._mean_token_emb(text)
        out = self.proj(mean_emb)
        return normalize_title_embedding(out)


class MoleTextGenerator:
    """Generator adapter exposing HF-like (generate, count_tokens) API."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        device: torch.device,
        gen_cfg: MoLEGenerationConfig,
        router: SubtaskRouter,
        title_embedder: TitleEmbedder,
        num_role_experts: int = 0,
        subtask_expert_offset: int = 0,
    ):
        self._mole = MoLEGenerator(model=model, tokenizer=tokenizer, device=device, gen_cfg=gen_cfg)
        self._tokenizer = tokenizer
        self._router = router
        self._title_embedder = title_embedder
        self._current_subtask_text = ""
        self._num_role_experts = max(int(num_role_experts), 0)
        self._subtask_expert_offset = max(int(subtask_expert_offset), 0)

    def set_subtask_text(self, text: str) -> None:
        self._current_subtask_text = str(text or "").strip()

    def generate(self, prompt: str) -> str:
        subtask_text = self._current_subtask_text or prompt
        with torch.no_grad():
            title_emb = self._title_embedder(subtask_text)
            logits = self._router(title_emb=title_emb)
            subtask_ids, _ = self._router.greedy_topk(logits)
            if self._num_role_experts > 0:
                role_id = 1 if (str(subtask_text).strip().lower().startswith("aggregate_for_") and self._num_role_experts >= 2) else 0
                role_tensor = torch.tensor([int(role_id)], dtype=subtask_ids.dtype, device=subtask_ids.device)
                expert_ids = torch.cat([role_tensor, subtask_ids + int(self._subtask_expert_offset)], dim=0)
            else:
                expert_ids = subtask_ids
            text, _prompt_ids, _gen_ids = self._mole.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
        return text

    def count_tokens(self, text: str) -> int:
        try:
            return len(self._tokenizer.encode(text))
        except Exception:
            return 0


def _sanitize(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "item"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_graph_root_index(graph_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not graph_root.exists():
        return index
    # Use os.walk(followlinks=True) so merged graph roots built from symlinked
    # shard folders (e.g., all_80/* -> shardX/*) are indexed correctly.
    for root, _dirs, files in os.walk(graph_root, followlinks=True):
        if "sample.json" not in files:
            continue
        sample_json = Path(root) / "sample.json"
        try:
            payload = json.loads(sample_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        problem_id = str(payload.get("problem_id") or "").strip()
        if problem_id:
            index[problem_id] = sample_json.parent
    return index


def _select_prompt_template(prompt_template: str, with_background: bool) -> str:
    if prompt_template == "background":
        return BACKGOUND_PROMPT_TEMPLATE
    if prompt_template == "default":
        return DEFAULT_PROMPT_TEMPLATE
    return BACKGOUND_PROMPT_TEMPLATE if with_background else DEFAULT_PROMPT_TEMPLATE


def _strip_dependency_imports(script: str) -> str:
    # Keep behavior aligned with SciCode official scripts.
    return re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", script, flags=re.MULTILINE)


def _is_valid_python(script: str) -> bool:
    if not script.strip():
        return False
    try:
        ast.parse(script)
        return True
    except (SyntaxError, RecursionError, MemoryError, ValueError):
        return False
    except Exception:
        # Never let parser edge-cases crash the whole pipeline worker.
        return False


def _extract_fenced_blocks(response: str) -> List[Tuple[str, str]]:
    # Keep language tag so we can prioritize python blocks and ignore reasoning-only blocks.
    pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", flags=re.DOTALL)
    out: List[Tuple[str, str]] = []
    for match in pattern.finditer(response):
        lang = str(match.group(1) or "").strip().lower()
        block = str(match.group(2) or "").strip()
        if not block:
            continue
        out.append((lang, block))
    return out


def _contains_reasoning_markers(script: str) -> bool:
    text = str(script or "").lower()
    if "<think>" in text or "</think>" in text:
        return True
    if text.startswith("assistant"):
        return True
    # Common traces of chain-of-thought that are not valid program output.
    markers = ("let's ", "i need to", "i should", "step by step", "re-read")
    return any(marker in text for marker in markers)


def _looks_like_code_start(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    code_prefixes = (
        "def ",
        "class ",
        "import ",
        "from ",
        "@",
        "if ",
        "for ",
        "while ",
        "try:",
        "with ",
        "async def ",
        "async for ",
        "async with ",
        "#",
        "'''",
        '"""',
    )
    if stripped.startswith(code_prefixes):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
        return True
    return False


def _code_likeness(script: str) -> Tuple[float, int]:
    lines = [line for line in script.splitlines() if line.strip()]
    if not lines:
        return (0.0, 0)
    score = 0
    for line in lines:
        stripped = line.lstrip()
        if _looks_like_code_start(stripped):
            score += 2
        elif any(ch in line for ch in ("=", "(", ")", "[", "]", "{", "}", ":", ".")):
            score += 1
    return (float(score) / float(len(lines)), len(lines))


def _has_def_or_class(script: str) -> bool:
    return bool(re.search(r"^\s*(def|class)\s+[A-Za-z_][A-Za-z0-9_]*", script, flags=re.MULTILINE))


def _largest_compilable_prefix(script: str) -> str:
    lines = script.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if _is_valid_python(candidate):
            return candidate
    return ""


def _select_best_candidate(candidates: Sequence[str]) -> str:
    if not candidates:
        return ""
    ranked = []
    for candidate in candidates:
        cleaned = candidate.strip()
        valid = _is_valid_python(cleaned)
        code_ratio, nlines = _code_likeness(cleaned)
        has_def = _has_def_or_class(cleaned)
        not_reasoning = 0 if _contains_reasoning_markers(cleaned) else 1
        # Prefer syntactically valid and longer candidates; tiny valid snippets are often truncation artifacts.
        ranked.append((1 if valid else 0, 1 if has_def else 0, not_reasoning, nlines, len(cleaned), code_ratio, cleaned))
    ranked.sort(reverse=True)
    return ranked[0][-1]


def _extract_from_plaintext(response: str) -> str:
    lines = response.splitlines()
    starts = [idx for idx, line in enumerate(lines) if _looks_like_code_start(line)]
    # Keep this bounded; very long reasoning can contain many false positives.
    starts = starts[:32]
    if 0 not in starts:
        starts = [0] + starts

    valid_candidates: List[str] = []
    fallback_candidates: List[str] = []
    for start in starts:
        snippet = "\n".join(lines[start:]).strip()
        if not snippet:
            continue
        snippet = _strip_dependency_imports(snippet).strip()
        if not snippet:
            continue
        if _is_valid_python(snippet):
            valid_candidates.append(snippet)
            continue
        trimmed = _largest_compilable_prefix(snippet)
        if trimmed:
            valid_candidates.append(trimmed)
        fallback_candidates.append(snippet)

    if valid_candidates:
        return _select_best_candidate(valid_candidates)
    if fallback_candidates:
        return _select_best_candidate(fallback_candidates)
    return ""


def _extract_python_script(response: str) -> str:
    response = (response or "").replace("\r\n", "\n").replace("\r", "\n")
    # Prefill-continuation style: prompt ends with ```python and model starts directly with code,
    # then emits a closing fence before any extra content.
    if "```" in response and not response.lstrip().startswith("```"):
        prefill_head = response.split("```", 1)[0].strip()
        if prefill_head:
            prefill_head = _strip_dependency_imports(prefill_head).strip()
            if prefill_head:
                if _is_valid_python(prefill_head):
                    return prefill_head
                trimmed = _largest_compilable_prefix(prefill_head)
                if trimmed and _is_valid_python(trimmed):
                    return trimmed

    fenced = _extract_fenced_blocks(response)
    if fenced:
        cleaned_blocks: List[str] = []
        for lang, block in fenced:
            # Skip explicit reasoning/assistant blocks; they frequently contain think traces.
            if lang in {"assistant", "analysis", "thought", "reasoning"}:
                continue
            cleaned = _strip_dependency_imports(block).strip()
            if not cleaned:
                continue
            if _is_valid_python(cleaned):
                cleaned_blocks.append(cleaned)
                continue
            trimmed = _largest_compilable_prefix(cleaned)
            cleaned_blocks.append(trimmed if trimmed else cleaned)
        # If we dropped everything (e.g., only assistant fence), fall back to all fenced blocks.
        if not cleaned_blocks:
            for _lang, block in fenced:
                cleaned = _strip_dependency_imports(block).strip()
                if not cleaned:
                    continue
                if _is_valid_python(cleaned):
                    cleaned_blocks.append(cleaned)
                    continue
                trimmed = _largest_compilable_prefix(cleaned)
                cleaned_blocks.append(trimmed if trimmed else cleaned)
        best = _select_best_candidate(cleaned_blocks)
        if best:
            return best.strip()

    plain = _extract_from_plaintext(response)
    if plain:
        return plain.strip()

    # Last resort: preserve previous behavior instead of returning empty.
    return _strip_dependency_imports(response).strip()


def _extract_function_name(function_header: str) -> str:
    pattern = r"\bdef\s+(\w+)\s*\("
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    pattern = r"\bclass\s+(\w+)\s*\("
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    raise ValueError("Function/class name not found in function_header.")


def _get_function_from_code(code_string: str, function_name: str) -> str:
    if not code_string.strip():
        return code_string
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception:
        return code_string
    return code_string


def _extract_step_number(step: dict, fallback_idx: int) -> str:
    return str(step.get("step_number") or f"step_{fallback_idx}")


def _build_step_maps(problem: dict) -> Tuple[Dict[str, dict], Dict[str, int]]:
    by_id: Dict[str, dict] = {}
    order: Dict[str, int] = {}
    sub_steps = list(problem.get("sub_steps") or [])
    for idx, step in enumerate(sub_steps, start=1):
        sid = _extract_step_number(step, idx)
        by_id[sid] = step
        order[sid] = idx
    return by_id, order


def _should_use_fixed_step(problem_id: str, step_id: str) -> bool:
    return (problem_id, step_id) in SPECIAL_STEP_FILES


def _load_fixed_step(problem_id: str, step_id: str) -> str:
    path = SPECIAL_STEP_FILES[(problem_id, step_id)]
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed SciCode step file: {path}")
    return path.read_text(encoding="utf-8").strip()


def _step_prompt_text(step: dict, with_background: bool) -> str:
    desc = str(step.get("step_description_prompt") or "").strip()
    if with_background:
        bg = str(step.get("step_background") or "").strip()
        if bg:
            return f"{desc}\n{bg}".strip()
    return desc


def _step_function_stub(step: dict) -> str:
    header = str(step.get("function_header") or "").strip()
    ret = str(step.get("return_line") or "").strip()
    if header and ret:
        return f"{header}\n\n{ret}"
    return header or ret


def _render_prompt(
    *,
    problem: dict,
    step: dict,
    ancestor_step_ids: Sequence[str],
    solved_functions: Dict[str, str],
    step_by_id: Dict[str, dict],
    with_background: bool,
    prompt_template: str,
) -> Tuple[str, str]:
    previous_lines: List[str] = []
    previous_code: List[str] = []
    for sid in ancestor_step_ids:
        prev_step = step_by_id[sid]
        previous_lines.append(_step_prompt_text(prev_step, with_background))
        previous_lines.append(solved_functions[sid])
        previous_lines.append("------")
        previous_code.append(solved_functions[sid])

    problem_steps_str = "\n\n".join(previous_lines[:-1]) if previous_lines else ""
    next_step_str = "\n\n".join([_step_prompt_text(step, with_background), _step_function_stub(step)]).strip()
    dependencies = str(problem.get("required_dependencies") or "").strip()

    prompt = prompt_template.format(
        problem_steps_str=problem_steps_str,
        next_step_str=next_step_str,
        dependencies=dependencies,
    )
    prefix = "\n".join([dependencies, "\n".join(previous_code)]).strip() + "\n"
    return prompt, prefix


def _build_chain_graph_json(problem: dict) -> dict:
    nodes: List[dict] = []
    previous: Optional[str] = None
    for idx, step in enumerate(problem.get("sub_steps") or [], start=1):
        sid = _extract_step_number(step, idx)
        nodes.append(
            {
                "node_id": sid,
                "title": str(step.get("step_description_prompt") or f"Step {sid}")[:120],
                "description": str(step.get("step_description_prompt") or "").strip(),
                "depends_on": [previous] if previous else [],
            }
        )
        previous = sid
    edges = [{"from": parent, "to": node["node_id"], "relation": "depends_on"} for node in nodes for parent in node["depends_on"]]
    return {
        "task_name": str(problem.get("problem_name") or f"problem_{problem.get('problem_id', 'unknown')}").strip(),
        "task_description": str(problem.get("problem_description_main") or "").strip(),
        "nodes": nodes,
        "edges": edges,
    }


def _build_predecessors(spec) -> Dict[int, List[int]]:
    preds: Dict[int, List[int]] = {nid: [] for nid in spec.node_metadata}
    for edge in spec.edge_strings:
        src, dst = edge.split("->", 1)
        preds[int(dst)].append(int(src))
    for node_id in preds:
        preds[node_id] = sorted(set(preds[node_id]))
    return preds


def _collect_ancestors(node_id: int, predecessors: Dict[int, List[int]]) -> Set[int]:
    visited: Set[int] = set()
    stack = list(predecessors.get(node_id, []))
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(predecessors.get(cur, []))
    return visited


def _assemble_program_code(
    *,
    dependencies: str,
    ancestor_step_ids: Sequence[str],
    solved_functions: Dict[str, str],
    current_python_code: str,
) -> str:
    parts: List[str] = []
    if dependencies.strip():
        parts.append(dependencies.strip())
    for sid in ancestor_step_ids:
        code = solved_functions.get(sid, "").strip()
        if code:
            parts.append(code)
    if current_python_code.strip():
        parts.append(current_python_code.strip())
    return "\n\n".join(parts).strip() + "\n"


H5_HELPER_CODE = textwrap.dedent(
    """
    import h5py
    import scipy
    import numpy as np

    def process_hdf5_list(group):
        lst = []
        for key in group.keys():
            lst.append(group[key][()])
        return lst

    def process_hdf5_sparse_matrix(group):
        data = group['data'][()]
        shape = tuple(group['shape'][()])
        if 'row' in group and 'col' in group:
            row = group['row'][()]
            col = group['col'][()]
            return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
        if 'blocksize' in group:
            indices = group['indices'][()]
            indptr = group['indptr'][()]
            blocksize = tuple(group['blocksize'][()])
            return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

    def process_hdf5_dict(group):
        dct = {}
        for key, obj in group.items():
            if isinstance(obj, h5py.Group):
                dct[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
            elif isinstance(obj[()], bytes):
                dct[key] = obj[()].decode('utf-8', errors='strict')
            else:
                try:
                    tmp = float(key)
                    dct[tmp] = obj[()]
                except ValueError:
                    dct[key] = obj[()]
        return dct

    def process_hdf5_datagroup(group):
        for key in group.keys():
            if key == "list":
                return process_hdf5_list(group[key])
            if key == "sparse_matrix":
                return process_hdf5_sparse_matrix(group[key])
            return process_hdf5_dict(group)

    def process_hdf5_to_tuple(step_id, test_num, h5py_file):
        data_lst = []
        with h5py.File(h5py_file, 'r') as f:
            for test_id in range(test_num):
                group_path = f'{step_id}/test{test_id + 1}'
                if not isinstance(f[group_path], h5py.Group):
                    raise FileNotFoundError(f'Path {group_path} not found in the file.')
                group = f[group_path]
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
        return data_lst
    """
).strip()


def _render_test_script(
    *,
    program_code: str,
    target_group: str,
    test_cases: Sequence[str],
    h5py_file: Path,
) -> str:
    lines: List[str] = [
        "import os",
        "import sys",
        f"sys.path.insert(0, {repr(str(SCICODE_ROOT / 'src'))})",
        "",
        H5_HELPER_CODE,
        "",
        program_code.strip(),
        "",
        f"targets = process_hdf5_to_tuple({repr(target_group)}, {len(test_cases)}, {repr(str(h5py_file))})",
    ]
    for idx, test in enumerate(test_cases):
        lines.append(f"target = targets[{idx}]")
        lines.append("")
        lines.extend((test or "").splitlines())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _run_python_script(script_path: Path, timeout_s: int, env: Dict[str, str]) -> ScriptRunResult:
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        passed = proc.returncode == 0
        status = "pass" if passed else "fail"
        return ScriptRunResult(
            passed=passed,
            status=status,
            return_code=int(proc.returncode),
            elapsed_ms=elapsed_ms,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            script_path=script_path,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.time() - start) * 1000)
        return ScriptRunResult(
            passed=False,
            status="timeout",
            return_code=124,
            elapsed_ms=elapsed_ms,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            script_path=script_path,
        )


def _check_eval_prerequisites(h5py_file: Path) -> None:
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required for SciCode step/general tests. Install it first.") from exc
    if not h5py_file.exists():
        raise FileNotFoundError(f"SciCode numeric target file missing: {h5py_file}")


def _sample_dir_name(sample_idx: int, problem: dict) -> str:
    pid = str(problem.get("problem_id") or sample_idx)
    pname = _sanitize(str(problem.get("problem_name") or f"problem_{pid}"))
    return f"{sample_idx:03d}_{pid}_{pname}"


def _run_step_test(
    *,
    sample_dir: Path,
    step_id: str,
    assembled_code: str,
    test_cases: Sequence[str],
    h5py_file: Path,
    timeout_s: int,
    env: Dict[str, str],
) -> ScriptRunResult:
    test_dir = sample_dir / "log" / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    script_path = test_dir / f"{step_id}.py"
    script_path.write_text(
        _render_test_script(
            program_code=assembled_code,
            target_group=step_id,
            test_cases=test_cases,
            h5py_file=h5py_file,
        ),
        encoding="utf-8",
    )
    return _run_python_script(script_path, timeout_s, env)


def _run_general_test(
    *,
    sample_dir: Path,
    problem_id: str,
    general_target_group: str,
    assembled_code: str,
    general_tests: Sequence[str],
    h5py_file: Path,
    timeout_s: int,
    env: Dict[str, str],
) -> ScriptRunResult:
    test_dir = sample_dir / "log" / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    script_path = test_dir / f"{problem_id}.general.py"
    script_path.write_text(
        _render_test_script(
            program_code=assembled_code,
            target_group=general_target_group,
            test_cases=general_tests,
            h5py_file=h5py_file,
        ),
        encoding="utf-8",
    )
    return _run_python_script(script_path, timeout_s, env)


def _load_backbone(*, model_name: str, torch_dtype: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


def _resolve_mole_ckpt_dir(path: Path) -> Path:
    p = path.resolve()
    if (p / "router.pt").exists():
        return p
    if (p / "final" / "router.pt").exists():
        return p / "final"
    raise FileNotFoundError(f"Invalid --mole-checkpoint (router.pt missing): {p}")


def _infer_subtask_expert_count_from_router(ckpt_dir: Path) -> Optional[int]:
    try:
        state = torch.load(ckpt_dir / "router.pt", map_location="cpu")
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    for key, value in state.items():
        if str(key).endswith("prototypes") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return int(value.shape[0])
    return None


def _infer_total_expert_count_from_lora(ckpt_dir: Path) -> Optional[int]:
    try:
        state = torch.load(ckpt_dir / "lora_state.pt", map_location="cpu")
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.ndim >= 3:
            return int(value.shape[0])
    return None


def _load_mole_checkpoint(
    *,
    ckpt_dir: Path,
    device: torch.device,
    router: nn.Module,
    title_embedder: nn.Module,
    model: nn.Module,
) -> None:
    router.load_state_dict(torch.load(ckpt_dir / "router.pt", map_location=device))
    title_embedder.load_state_dict(torch.load(ckpt_dir / "title_embedder.pt", map_location=device))
    lora_state = torch.load(ckpt_dir / "lora_state.pt", map_location="cpu")
    name_to_param = dict(model.named_parameters())
    for name, tensor in lora_state.items():
        p = name_to_param.get(name)
        if p is None:
            continue
        p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SciCode DAG pipeline with SciCode prompt templates.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--problem-id", type=str, default=None, help="Run one problem_id only.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all selected samples.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip-existing-any-status",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If sample_metrics.json exists, skip the sample regardless of stored status.",
    )
    parser.add_argument(
        "--fallback-chain-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If graph missing in --graph-root, synthesize a linear chain graph from sub_steps.",
    )

    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--code-model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--code-max-new-tokens", type=int, default=4096)
    parser.add_argument("--code-temperature", type=float, default=0.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--code-subprocess", action="store_true")
    parser.add_argument(
        "--keep-model-loaded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep code model loaded across all samples.",
    )
    parser.add_argument(
        "--mole-checkpoint",
        type=Path,
        default=None,
        help="Enable MoLE inference mode and load router/title/LORA from this checkpoint dir.",
    )
    parser.add_argument("--mole-num-subtask-experts", type=int, default=4)
    parser.add_argument("--mole-subtask-top-k", type=int, default=2)
    parser.add_argument("--mole-lora-rank", type=int, default=8)
    parser.add_argument("--mole-lora-alpha", type=float, default=16.0)
    parser.add_argument("--mole-lora-last-n-layers", type=int, default=8)

    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--with-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror SciCode gencode mode switch for prompt construction.",
    )
    parser.add_argument(
        "--prompt-template",
        choices=["auto", "background", "default"],
        default="auto",
        help="'background' => BACKGOUND_PROMPT_TEMPLATE, 'default' => DEFAULT_PROMPT_TEMPLATE.",
    )

    parser.add_argument("--no-eval", action="store_false", dest="eval", help="Skip SciCode step/general tests.")
    parser.set_defaults(eval=True)
    parser.add_argument("--h5py-file", type=Path, default=DEFAULT_H5PY_FILE)
    parser.add_argument("--test-timeout-s", type=int, default=180)
    return parser.parse_args(argv)


def _make_generator(args: argparse.Namespace):
    if args.mole_checkpoint is not None:
        device = torch.device("cuda", int(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        ckpt_dir = _resolve_mole_ckpt_dir(Path(args.mole_checkpoint))
        inferred_subtask_experts = _infer_subtask_expert_count_from_router(ckpt_dir)
        inferred_total_experts = _infer_total_expert_count_from_lora(ckpt_dir)
        num_subtask_experts = int(
            inferred_subtask_experts
            if inferred_subtask_experts is not None
            else int(args.mole_num_subtask_experts)
        )
        num_role_experts = int(
            max(0, int(inferred_total_experts) - int(num_subtask_experts))
            if inferred_total_experts is not None
            else max(0, int(getattr(args, "mole_num_role_experts", 0)))
        )
        num_total_experts = int(
            inferred_total_experts
            if inferred_total_experts is not None
            else int(num_role_experts + num_subtask_experts)
        )
        model, tokenizer = _load_backbone(
            model_name=args.code_model_name,
            torch_dtype=str(args.torch_dtype),
            device=device,
        )
        lora_cfg = LoRAConfig(
            num_experts=int(num_total_experts),
            top_k=int(args.mole_subtask_top_k),
            rank=int(args.mole_lora_rank),
            alpha=float(args.mole_lora_alpha),
            target_modules=("q_proj", "v_proj", "o_proj"),
            last_n_layers=int(args.mole_lora_last_n_layers),
        )
        inject_mole_lora(model, cfg=lora_cfg)
        router_cfg = SubtaskRouterConfig(
            num_experts=int(num_subtask_experts),
            top_k=int(args.mole_subtask_top_k),
        )
        router = SubtaskRouter(router_cfg).to(device)
        title_embedder = TitleEmbedder(model=model, tokenizer=tokenizer, out_dim=router_cfg.title_emb_dim).to(device)
        _load_mole_checkpoint(
            ckpt_dir=ckpt_dir,
            device=device,
            router=router,
            title_embedder=title_embedder,
            model=model,
        )
        gen_cfg = MoLEGenerationConfig(
            model_name=args.code_model_name,
            max_new_tokens=int(args.code_max_new_tokens),
            temperature=float(args.code_temperature),
            top_p=0.95,
            torch_dtype=str(args.torch_dtype),
            device=int(args.device),
        )
        return MoleTextGenerator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            gen_cfg=gen_cfg,
            router=router,
            title_embedder=title_embedder,
            num_role_experts=int(num_role_experts),
            subtask_expert_offset=int(num_role_experts),
        )

    cfg = HFGenerationConfig(
        model_name=args.code_model_name,
        device=args.device,
        device_map=(None if str(args.device_map).strip().lower() == "none" else args.device_map),
        max_new_tokens=args.code_max_new_tokens,
        temperature=args.code_temperature,
        torch_dtype=args.torch_dtype,
    )
    return HFSubprocessTextGenerator(cfg) if args.code_subprocess else HFTextGenerator(cfg)


def _summarize_samples(rows: Sequence[dict]) -> dict:
    total = len(rows)
    ok = sum(1 for row in rows if row.get("overall_status", row.get("status")) == "ok")
    failed = sum(1 for row in rows if row.get("overall_status", row.get("status")) == "failed")
    skipped = sum(1 for row in rows if row.get("overall_status", row.get("status")) == "skipped")
    problem_correct_samples = 0
    general_pass_samples = 0
    for row in rows:
        tested_steps = row.get("tested_steps")
        passed_steps = row.get("passed_steps")
        default_problem_correctness = 1 if isinstance(tested_steps, (int, float)) and tested_steps > 0 and passed_steps == tested_steps else 0
        if int(row.get("problem_correctness", default_problem_correctness)) == 1:
            problem_correct_samples += 1
        if str(row.get("general_test_status", "")).strip().lower() == "pass":
            general_pass_samples += 1

    def _avg(values: Iterable[float]) -> float:
        values = list(values)
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "total_samples": total,
        "ok_samples": ok,
        "failed_samples": failed,
        "skipped_samples": skipped,
        "problem_correct_samples": problem_correct_samples,
        "problem_correct_rate": (float(problem_correct_samples) / float(total)) if total else 0.0,
        "general_pass_samples": general_pass_samples,
        "general_pass_rate": (float(general_pass_samples) / float(total)) if total else 0.0,
        "mean_step_pass_rate": _avg(
            float(row.get("step_pass_rate", 0.0))
            for row in rows
            if isinstance(row.get("step_pass_rate"), (int, float))
        ),
        "mean_prompt_tokens": _avg(
            float(row.get("total_prompt_tokens", 0.0))
            for row in rows
            if isinstance(row.get("total_prompt_tokens"), (int, float))
        ),
        "mean_completion_tokens": _avg(
            float(row.get("total_completion_tokens", 0.0))
            for row in rows
            if isinstance(row.get("total_completion_tokens"), (int, float))
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.gpus and str(args.gpus).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus).strip()

    problems = _read_jsonl(args.dataset)
    if args.problem_id is not None:
        problems = [row for row in problems if str(row.get("problem_id")) == str(args.problem_id)]
        if not problems:
            raise ValueError(f"problem_id={args.problem_id} not found in {args.dataset}")
    if args.max_samples and args.max_samples > 0:
        problems = problems[: args.max_samples]
    if not problems:
        raise RuntimeError("No problems selected.")

    if args.eval:
        _check_eval_prerequisites(args.h5py_file)

    graph_index = _build_graph_root_index(args.graph_root) if args.graph_root else {}
    prompt_template = _select_prompt_template(args.prompt_template, bool(args.with_background))

    timestamp = (args.timestamp or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    sample_rows: List[dict] = []
    generator = _make_generator(args) if args.keep_model_loaded else None

    run_env = os.environ.copy()
    existing_pp = run_env.get("PYTHONPATH", "")
    scicode_src = str(SCICODE_ROOT / "src")
    run_env["PYTHONPATH"] = f"{scicode_src}:{existing_pp}" if existing_pp else scicode_src

    for sample_idx, problem in enumerate(problems, start=1):
        sample_name = _sample_dir_name(sample_idx, problem)
        sample_dir = run_root / sample_name
        log_dir = sample_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        sample_metrics_path = log_dir / "sample_metrics.json"

        if args.skip_existing_any_status and sample_metrics_path.exists():
            try:
                payload = json.loads(sample_metrics_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if payload:
                sample_rows.append(payload)
            continue

        if args.skip_existing and sample_metrics_path.exists():
            try:
                payload = json.loads(sample_metrics_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if payload.get("status") == "ok":
                sample_rows.append(payload)
                continue

        problem_id = str(problem.get("problem_id") or "")
        problem_name = str(problem.get("problem_name") or f"problem_{problem_id}")
        step_by_id, step_order = _build_step_maps(problem)

        (sample_dir / "sample.json").write_text(
            json.dumps(
                {
                    "problem_id": problem_id,
                    "problem_name": problem_name,
                    "num_sub_steps": len(problem.get("sub_steps") or []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        graph_path = sample_dir / "task_graph.json"
        graph_source = "missing"
        src_dir = graph_index.get(problem_id)
        if src_dir and (src_dir / "task_graph.json").exists():
            graph_source = str(src_dir / "task_graph.json")
            graph_path.write_text((src_dir / "task_graph.json").read_text(encoding="utf-8"), encoding="utf-8")
            graph_gen_src = src_dir / "graph_generation.json"
            if graph_gen_src.exists():
                (sample_dir / "graph_generation.json").write_text(graph_gen_src.read_text(encoding="utf-8"), encoding="utf-8")
        elif args.fallback_chain_graph:
            graph_source = "chain_fallback"
            _write_json(graph_path, _build_chain_graph_json(problem))
            _write_json(sample_dir / "graph_generation.json", {"status": "ok", "edge_source": "chain_fallback"})
        else:
            row = {
                "status": "failed",
                "problem_id": problem_id,
                "problem_name": problem_name,
                "error": "task_graph_missing",
                "graph_source": graph_source,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _write_json(sample_metrics_path, row)
            sample_rows.append(row)
            if args.stop_on_failure:
                break
            continue

        spec = convert_taskgraph(graph_path)
        predecessors = _build_predecessors(spec)
        solved_functions: Dict[str, str] = {}
        solved_step_full: Dict[str, str] = {}
        node_logs: List[dict] = []
        failed = False
        skipped_count = 0
        step_passed = 0
        tested_steps = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        run_start = time.time()

        if generator is None:
            generator = _make_generator(args)

        for node_id in sorted(spec.node_metadata):
            node_meta = spec.node_metadata[node_id]
            step_id = str(node_meta.original_id)
            step = step_by_id.get(step_id)
            node_dir = log_dir / f"node_{node_id:02d}_{_sanitize(step_id)}"
            node_dir.mkdir(parents=True, exist_ok=True)

            if step is None:
                node_logs.append({"node_id": node_id, "step_id": step_id, "status": "failed", "error": "missing_step_in_dataset"})
                failed = True
                if args.stop_on_failure:
                    break
                continue

            if _should_use_fixed_step(problem_id, step_id):
                fixed_code = _load_fixed_step(problem_id, step_id)
                solved_functions[step_id] = fixed_code
                solved_step_full[step_id] = fixed_code
                skipped_count += 1
                node_logs.append({"node_id": node_id, "step_id": step_id, "status": "fixed_step"})
                (sample_dir / "generated_code").mkdir(parents=True, exist_ok=True)
                (sample_dir / "generated_code" / f"{step_id}.py").write_text(fixed_code + "\n", encoding="utf-8")
                continue

            ancestor_node_ids = _collect_ancestors(node_id, predecessors)
            ancestor_step_ids = [spec.node_metadata[a].original_id for a in sorted(ancestor_node_ids)]
            ancestor_step_ids = [sid for sid in ancestor_step_ids if sid in solved_functions]
            ancestor_step_ids.sort(key=lambda sid: step_order.get(sid, 10**9))

            prompt_base, _ = _render_prompt(
                problem=problem,
                step=step,
                ancestor_step_ids=ancestor_step_ids,
                solved_functions=solved_functions,
                step_by_id=step_by_id,
                with_background=bool(args.with_background),
                prompt_template=prompt_template,
            )

            last_error: Optional[str] = None
            for attempt in range(1, max(int(args.max_attempts), 1) + 1):
                prompt = prompt_base
                if last_error:
                    prompt = (
                        prompt
                        + "\n\nPrevious attempt failed.\n"
                        + SCICODE_RETRY_GUIDELINES
                        + "\n"
                        + f"Failure hint:\n{last_error}\n"
                    )
                prompt_file = node_dir / f"prompt_attempt_{attempt}.txt"
                prompt_file.write_text(prompt, encoding="utf-8")

                if isinstance(generator, MoleTextGenerator):
                    generator.set_subtask_text(_step_prompt_text(step, bool(args.with_background)))

                prompt_tokens = generator.count_tokens(prompt)
                call_start = time.time()
                response = generator.generate(prompt)
                call_elapsed_ms = int((time.time() - call_start) * 1000)
                completion_tokens = generator.count_tokens(response)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                (node_dir / f"response_attempt_{attempt}.txt").write_text(response, encoding="utf-8")
                python_code = _extract_python_script(response)
                (node_dir / f"python_attempt_{attempt}.py").write_text(python_code + "\n", encoding="utf-8")

                try:
                    fn_name = _extract_function_name(str(step.get("function_header") or ""))
                except Exception:
                    fn_name = ""
                parsed_function = _get_function_from_code(python_code, fn_name) if fn_name else python_code
                parsed_function = (parsed_function or "").strip()
                if not parsed_function:
                    last_error = "empty parsed function code"
                    continue

                assembled_code = _assemble_program_code(
                    dependencies=str(problem.get("required_dependencies") or ""),
                    ancestor_step_ids=ancestor_step_ids,
                    solved_functions=solved_functions,
                    current_python_code=python_code,
                )
                (sample_dir / "generated_code").mkdir(parents=True, exist_ok=True)
                (sample_dir / "generated_code" / f"{step_id}.py").write_text(assembled_code, encoding="utf-8")

                step_test_result: Optional[ScriptRunResult] = None
                if args.eval:
                    tests = list(step.get("test_cases") or [])
                    tested_steps += 1
                    step_test_result = _run_step_test(
                        sample_dir=sample_dir,
                        step_id=step_id,
                        assembled_code=assembled_code,
                        test_cases=tests,
                        h5py_file=args.h5py_file,
                        timeout_s=args.test_timeout_s,
                        env=run_env,
                    )
                    _write_json(
                        node_dir / f"step_test_attempt_{attempt}.json",
                        {
                            "status": step_test_result.status,
                            "return_code": step_test_result.return_code,
                            "elapsed_ms": step_test_result.elapsed_ms,
                            "stdout_tail": step_test_result.stdout[-4000:],
                            "stderr_tail": step_test_result.stderr[-4000:],
                            "script_path": str(step_test_result.script_path),
                        },
                    )
                    if not step_test_result.passed:
                        last_error = (
                            f"step test {step_test_result.status}; "
                            f"rc={step_test_result.return_code}; stderr_tail={step_test_result.stderr[-500:]}"
                        )
                        continue
                    step_passed += 1

                solved_functions[step_id] = parsed_function
                solved_step_full[step_id] = python_code.strip()
                node_logs.append(
                    {
                        "node_id": node_id,
                        "step_id": step_id,
                        "status": "ok",
                        "attempt": attempt,
                        "elapsed_ms": call_elapsed_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "ancestor_steps": ancestor_step_ids,
                        "tested": bool(args.eval),
                        "test_status": step_test_result.status if step_test_result is not None else "skipped",
                    }
                )
                break
            else:
                failed = True
                node_logs.append(
                    {
                        "node_id": node_id,
                        "step_id": step_id,
                        "status": "failed",
                        "attempts": max(int(args.max_attempts), 1),
                        "error": last_error or "unknown",
                    }
                )
                if args.stop_on_failure:
                    break

        general_result: Optional[ScriptRunResult] = None
        general_status = "skipped"
        if args.eval and not failed:
            final_step_ids = [sid for sid, _rank in sorted(step_order.items(), key=lambda item: item[1]) if sid in solved_functions]
            final_code = "\n\n".join(
                [str(problem.get("required_dependencies") or "").strip()] + [solved_functions[sid] for sid in final_step_ids]
            ).strip() + "\n"
            general_tests = list(problem.get("general_tests") or [])
            if general_tests:
                sub_steps = list(problem.get("sub_steps") or [])
                general_target_group = str(sub_steps[-1].get("step_number") or problem_id) if sub_steps else problem_id
                general_result = _run_general_test(
                    sample_dir=sample_dir,
                    problem_id=problem_id,
                    general_target_group=general_target_group,
                    assembled_code=final_code,
                    general_tests=general_tests,
                    h5py_file=args.h5py_file,
                    timeout_s=args.test_timeout_s,
                    env=run_env,
                )
                general_status = general_result.status
                _write_json(
                    log_dir / "general_test.json",
                    {
                        "status": general_result.status,
                        "return_code": general_result.return_code,
                        "elapsed_ms": general_result.elapsed_ms,
                        "stdout_tail": general_result.stdout[-4000:],
                        "stderr_tail": general_result.stderr[-4000:],
                        "target_group": general_target_group,
                        "script_path": str(general_result.script_path),
                    },
                )
                if not general_result.passed:
                    failed = True
            else:
                general_status = "no_general_tests"

        elapsed_ms = int((time.time() - run_start) * 1000)
        total_effective_steps = len([sid for sid in step_order if not _should_use_fixed_step(problem_id, sid)])
        step_pass_rate = (float(step_passed) / float(tested_steps)) if tested_steps else 0.0
        overall_status = "failed" if failed else "ok"
        problem_correctness = 1 if tested_steps > 0 and step_passed == tested_steps else 0

        sample_row = {
            "status": overall_status,
            "overall_status": overall_status,
            "problem_id": problem_id,
            "problem_name": problem_name,
            "graph_source": graph_source,
            "total_steps": len(step_order),
            "effective_steps": total_effective_steps,
            "fixed_steps": skipped_count,
            "solved_steps": len(solved_functions),
            "tested_steps": tested_steps,
            "passed_steps": step_passed,
            "step_pass_rate": step_pass_rate,
            "problem_correctness": problem_correctness,
            "general_test_status": general_status,
            "general_test_pass": 1 if str(general_status).lower() == "pass" else 0,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "elapsed_ms": elapsed_ms,
            "node_logs": node_logs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(sample_metrics_path, sample_row)
        sample_rows.append(sample_row)

        if not args.keep_model_loaded:
            generator = None
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if failed and args.stop_on_failure:
            break

    summary = {
        "dataset": str(args.dataset),
        "graph_root": str(args.graph_root),
        "output_root": str(run_root),
        "model_name": args.code_model_name,
        "eval_enabled": bool(args.eval),
        "with_background": bool(args.with_background),
        "prompt_template": args.prompt_template,
        "samples": sample_rows,
        "aggregate": _summarize_samples(sample_rows),
    }
    _write_json(run_root / "summary.json", summary)
    print(f"SciCode taskgraph pipeline complete. Output: {run_root}")


if __name__ == "__main__":
    main()
