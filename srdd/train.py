"""
train_mole_srdd_hgrpo.py  –  SRDD HGRPO Training Script
=========================================================
SRDD 数据集上的分层 GRPO（HGRPO）训练脚本。

主要特性：
  - 在线 DAG 执行：按拓扑顺序逐节点训练（execute + aggregate 均参与）
  - 三种消融模式（exp1/exp2/exp3）：通过 CLI 参数切换
  - 多 GPU 分布式：每个 rank 处理同一样本的同一节点，
    通过 all-reduce 共享奖励，梯度经 all-reduce 平均后更新
  - 奖励：smoke_test_repo 二值 pass/fail（0.0 / 1.0）
  - 上下文同步：每个节点更新后，rank 0 的最优候选代码
    通过 broadcast 同步到全部 rank，供下游节点使用

与 SciCode v3 脚本的关键差异：
  1. 数据集：SRDD CSV（category/name/description） 而非 SciCode JSONL
  2. 任务图：从 taskgraph_root 加载预生成的 task_graph.json（SRDD 格式）
  3. 训练粒度：在线 DAG 执行（顺序） 而非预构建条目列表
  4. 奖励：smoke_test_repo 二值 0/1
  5. 同时训练 execute 和 aggregate 两种 expert（SciCode 只训练 execute）
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ─────────────────────────────────────────────────────────────────
# REPO_ROOT is the directory that CONTAINS morse/, scicode/, srdd/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── SRDD & task-graph imports ──────────────────────────────────────────────────
from srdd.tomas.codes import Codes                            # noqa
from morse.llm.converter import convert_taskgraph            # noqa
from srdd.tomas.executor import (                             # noqa
    build_aggregate_prompt,
    build_executor_prompt,
)
from srdd.tomas.review_test import smoke_test_repo            # noqa
from srdd.eval.reward import completeness_from_code, consistency_stripped_from_code  # noqa

# ── MoLE imports ───────────────────────────────────────────────────────────────
from morse.mole.mole_generator import GenerationConfig, MoLEGenerator   # noqa
from morse.mole.mole_lora import LoRAConfig, inject_mole_lora           # noqa
from morse.mole.router import (                                          # noqa
    SubtaskRouter,
    SubtaskRouterConfig,
    normalize_title_embedding,
)

# ── Optional grpo_base utilities ───────────────────────────────────────────────
try:
    from morse.hgrpo import grpo_utils as _grpo_base  # type: ignore
    _HAS_GRPO_BASE = True
except ImportError:
    _HAS_GRPO_BASE = False

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SRDD_CSV = REPO_ROOT / "srdd" / "data" / "SRDD.csv"
DEFAULT_RUN_ROOT = REPO_ROOT / "srdd" / "runs"
DEFAULT_CKPT_ROOT = REPO_ROOT / "srdd" / "checkpoints"

ROLE_EXPERT_IDS: Dict[str, int] = {
    "execute": 0,
    "aggregate": 1,
}


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SRDDSample:
    name: str
    description: str
    category: str


@dataclass
class SRDDHGRPOCandidate:
    idx: int
    route_local_idx: int
    route_global_idx: int
    text: str
    codes: Optional[Codes]
    prompt_ids: torch.Tensor
    gen_ids: torch.Tensor
    expert_ids: torch.Tensor
    expert_weights: Optional[torch.Tensor]
    logp_router: torch.Tensor
    role_expert_id: int
    subtask_expert_ids: List[int]
    route_is_anchor: bool
    route_selection_mode: str
    reward: float
    smoke_passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# SRDD data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def read_srdd_samples(csv_path: Path) -> List[SRDDSample]:
    samples: List[SRDDSample] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Name") or "").strip()
            desc = (row.get("Description") or "").strip()
            cat  = (row.get("Category") or "Unknown").strip()
            if name and desc:
                samples.append(SRDDSample(name=name, description=desc, category=cat))
    return samples


def _load_taskgraph_index(root: Path) -> Dict[Tuple[str, str], Path]:
    """Map (category, name) -> task_graph.json path."""
    index: Dict[Tuple[str, str], Path] = {}
    for sample_json in root.rglob("sample.json"):
        try:
            payload = json.loads(sample_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        cat  = str(payload.get("category", "") or "").strip()
        name = str(payload.get("name", "") or "").strip()
        if not cat or not name:
            continue
        graph_path = sample_json.parent / "task_graph.json"
        if graph_path.exists():
            index[(cat, name)] = graph_path
    return index


def _build_predecessors(edge_strings: List[str], node_ids: List[int]) -> Dict[int, List[int]]:
    preds: Dict[int, List[int]] = {nid: [] for nid in node_ids}
    for edge in edge_strings:
        src, dst = edge.split("->", 1)
        preds[int(dst)].append(int(src))
    return preds


def _topological_order(node_ids: List[int], predecessors: Dict[int, List[int]]) -> List[int]:
    """Kahn's algorithm – returns nodes in topological order."""
    in_degree: Dict[int, int] = {n: len(predecessors.get(n, [])) for n in node_ids}
    queue = [n for n in node_ids if in_degree[n] == 0]
    order: List[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in node_ids:
            if node in predecessors.get(child, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    return order if len(order) == len(node_ids) else list(node_ids)


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "item"


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SRDD reward evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _repo_text(codes: Codes) -> str:
    parts = []
    for filename in sorted(codes.codebooks):
        parts.append(codes.codebooks[filename])
    return "\n".join(parts)


def _evaluate_srdd_candidate(
    *,
    codes: Optional[Codes],
    node_dir: Path,
    timeout_s: int,
    candidate_idx: int,
    task_description: str = "",
    subtask_description: str = "",
    w_smoke: float = 0.5,
    w_comp: float = 0.5,
    w_cons_strip: float = 1.0,
    w_cons_task: float = 0.7,
    w_cons_subtask: float = 0.3,
) -> Tuple[float, bool, str, Dict[str, Any]]:
    """
    Proxy reward for an SRDD candidate:
      reward = w_smoke * smoke_ok + w_comp * completeness
               + w_cons_strip * (w_cons_task * cons_task + w_cons_subtask * cons_subtask)

    Returns (reward: float, passed: bool, detail: str, proxy_metrics: dict).
    """
    empty_metrics: Dict[str, Any] = {
        "smoke_passed": False, "proxy_completeness": 0.0,
        "proxy_consistency_task": 0.0, "proxy_consistency_subtask": 0.0,
        "proxy_reward": 0.0,
    }
    if codes is None or not codes.codebooks:
        return 0.0, False, "empty_codes", empty_metrics

    repo_dir = node_dir / f"cand_{candidate_idx:03d}_repo"
    passed = False
    detail = ""
    try:
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        codes.write_to_directory(repo_dir)
        result = smoke_test_repo(repo_dir, timeout_s=int(timeout_s))
        passed = bool(result.passed)
        detail = str(result.details or "")
    except Exception as exc:
        detail = str(exc)[:200]
    finally:
        try:
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
        except Exception:
            pass

    code_text = _repo_text(codes)
    comp = float(completeness_from_code(code_text))
    cons_task = float(consistency_stripped_from_code(task_description, code_text)) if task_description else 0.0
    cons_subtask = float(consistency_stripped_from_code(subtask_description, code_text)) if subtask_description else 0.0
    cons_mix = float(w_cons_task) * cons_task + float(w_cons_subtask) * cons_subtask
    reward = float(w_smoke) * float(passed) + float(w_comp) * comp + float(w_cons_strip) * cons_mix
    proxy_metrics: Dict[str, Any] = {
        "smoke_passed": bool(passed),
        "proxy_completeness": float(comp),
        "proxy_consistency_task": float(cons_task),
        "proxy_consistency_subtask": float(cons_subtask),
        "proxy_reward": float(reward),
    }
    return float(reward), bool(passed), detail, proxy_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Distributed utilities  (inline – no dependency on grpo_base)
# ══════════════════════════════════════════════════════════════════════════════

def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _safe_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(max(var, 0.0))


def _clip_advantage(adv: float, clip: float) -> float:
    if clip <= 0.0:
        return adv
    return max(-clip, min(clip, adv))


def _all_gather_float_list(
    local_vals: List[float],
    *,
    device: torch.device,
    world_size: int,
) -> List[float]:
    """Gather float lists from all ranks; returns concatenated list."""
    if _HAS_GRPO_BASE:
        try:
            return _grpo_base._all_gather_float_list(local_vals, device=device, world_size=world_size)
        except Exception:
            pass
    if world_size <= 1 or not _is_dist_ready():
        return list(local_vals)
    n = len(local_vals)
    t = torch.tensor(local_vals, dtype=torch.float32, device=device)
    gathered = [torch.zeros(n, dtype=torch.float32, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    out: List[float] = []
    for g in gathered:
        out.extend(g.cpu().tolist())
    return out


def _average_gradients(params: List[torch.nn.Parameter], world_size: int) -> None:
    if _HAS_GRPO_BASE:
        try:
            _grpo_base._average_gradients(params, world_size)
            return
        except Exception:
            pass
    if world_size <= 1 or not _is_dist_ready():
        return
    # All ranks must participate in the same set of collectives. If a rank has
    # p.grad=None (e.g. it sampled a different LoRA expert subset), replace it
    # with zeros so dist.all_reduce is symmetric across ranks.
    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def _broadcast_text(text: str, *, src: int, device: torch.device) -> str:
    """Broadcast a UTF-8 string from rank `src` to all other ranks."""
    if not _is_dist_ready():
        return text
    encoded = text.encode("utf-8")
    length_t = torch.tensor([len(encoded)], dtype=torch.long, device=device)
    dist.broadcast(length_t, src=src)
    length = int(length_t.item())
    if length == 0:
        return ""
    buf = torch.zeros(length, dtype=torch.uint8, device=device)
    if dist.get_rank() == src:
        buf[:] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
    dist.broadcast(buf, src=src)
    return buf.cpu().numpy().tobytes().decode("utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════════════
# TitleEmbedder
# ══════════════════════════════════════════════════════════════════════════════

class TitleEmbedder(nn.Module):
    """Projects a title/subtask string to a fixed-size embedding for the router."""

    def __init__(self, *, model: nn.Module, tokenizer, out_dim: int) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        hidden_size: int = model.config.hidden_size  # type: ignore[attr-defined]
        self.proj = nn.Linear(hidden_size, out_dim, bias=False)

    def _mean_token_emb(self, title: str) -> torch.Tensor:
        inputs = self._tokenizer(
            title,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=64,
        )
        input_ids: torch.Tensor = inputs["input_ids"].to(self._model.device)
        with torch.no_grad():
            emb = self._model.get_input_embeddings()(input_ids)
        pad_id = self._tokenizer.pad_token_id
        if pad_id is not None:
            mask = (input_ids != pad_id).float().unsqueeze(-1)
            den = torch.clamp(mask.sum(dim=1), min=1.0)
            return (emb * mask).sum(dim=1) / den
        return emb.mean(dim=1)

    def forward(self, title: str) -> torch.Tensor:  # type: ignore[override]
        return normalize_title_embedding(self.proj(self._mean_token_emb(title)))


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint utilities
# ══════════════════════════════════════════════════════════════════════════════

def _collect_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad and ("lora_A" in name or "lora_B" in name)
    }


def save_checkpoint(
    *,
    ckpt_dir: Path,
    router: nn.Module,
    title_embedder: nn.Module,
    model: nn.Module,
    opt_router: Optional[torch.optim.Optimizer],
    opt_lora: Optional[torch.optim.Optimizer],
    trainer_state: Dict[str, Any],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(router.state_dict(), ckpt_dir / "router.pt")
    torch.save(title_embedder.state_dict(), ckpt_dir / "title_embedder.pt")
    torch.save(_collect_lora_state(model), ckpt_dir / "lora_state.pt")
    if opt_router is not None:
        torch.save(opt_router.state_dict(), ckpt_dir / "opt_router.pt")
    if opt_lora is not None:
        torch.save(opt_lora.state_dict(), ckpt_dir / "opt_lora.pt")
    torch.save(trainer_state, ckpt_dir / "trainer_state.pt")
    print(f"[ckpt] saved → {ckpt_dir}", flush=True)


def _move_opt_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def load_checkpoint(
    *,
    ckpt_dir: Path,
    device: torch.device,
    router: nn.Module,
    title_embedder: nn.Module,
    model: nn.Module,
    opt_router: Optional[torch.optim.Optimizer],
    opt_lora: Optional[torch.optim.Optimizer],
) -> Dict[str, Any]:
    router.load_state_dict(torch.load(ckpt_dir / "router.pt", map_location=device))
    title_embedder.load_state_dict(torch.load(ckpt_dir / "title_embedder.pt", map_location=device))
    lora_state = torch.load(ckpt_dir / "lora_state.pt", map_location="cpu")
    name_to_param = dict(model.named_parameters())
    for name, tensor in lora_state.items():
        p = name_to_param.get(name)
        if p is not None:
            p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))
    if opt_router is not None and (ckpt_dir / "opt_router.pt").exists():
        opt_router.load_state_dict(torch.load(ckpt_dir / "opt_router.pt", map_location="cpu"))
        _move_opt_state_to_device(opt_router, device)
    if opt_lora is not None and (ckpt_dir / "opt_lora.pt").exists():
        opt_lora.load_state_dict(torch.load(ckpt_dir / "opt_lora.pt", map_location="cpu"))
        _move_opt_state_to_device(opt_lora, device)
    state_path = ckpt_dir / "trainer_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        return dict(state) if isinstance(state, dict) else {}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# HGRPO route/expert helpers  (adapted from SciCode v3)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_local_route_layout(
    *,
    local_group_size: int,
    requested_local_routes: int,
) -> Tuple[int, int]:
    """Return (num_routes_local, candidates_per_route_local)."""
    if local_group_size <= 0:
        raise ValueError("local_group_size must be > 0.")
    route_count = (
        max(1, local_group_size // 2) if requested_local_routes <= 0 else requested_local_routes
    )
    if route_count > local_group_size:
        raise ValueError(
            f"hierarchical-local-routes ({route_count}) > local_group_size ({local_group_size})."
        )
    if local_group_size % route_count != 0:
        raise ValueError(
            f"local_group_size ({local_group_size}) must be divisible by "
            f"hierarchical-local-routes ({route_count})."
        )
    return int(route_count), int(local_group_size // route_count)


def _select_execute_experts(
    *,
    router: SubtaskRouter,
    title_embedder: TitleEmbedder,
    device: torch.device,
    subtask_text: str,
    subtask_expert_offset: int,
    selection_mode: str,  # "greedy" or "sample"
) -> Tuple[torch.Tensor, torch.Tensor, int, List[int], torch.Tensor, torch.Tensor]:
    """
    Select expert IDs for an execute entry.
    Returns (expert_ids, logp_router, role_expert_id, subtask_local_ids,
             router_logits, router_probs).
    """
    role_id = int(ROLE_EXPERT_IDS["execute"])
    logits = router(title_emb=title_embedder(subtask_text))
    if selection_mode == "greedy":
        subtask_ids, logp_router = router.greedy_topk(logits)
    else:
        subtask_ids, logp_router = router.sample_topk(logits)
    subtask_ids = subtask_ids + int(subtask_expert_offset)
    role_tensor = torch.tensor([role_id], dtype=torch.long, device=device)
    expert_ids = torch.cat([role_tensor, subtask_ids.to(device=device)])
    subtask_local = [int(x) - int(subtask_expert_offset) for x in expert_ids.detach().cpu().tolist()[1:]]
    router_logits = logits[0]
    router_probs = F.softmax(router_logits, dim=-1)
    return expert_ids, logp_router, role_id, subtask_local, router_logits, router_probs


def _select_aggregate_experts(
    *,
    device: torch.device,
    parent_subtask_experts: List[int],
    subtask_expert_offset: int,
    merge_use_parent_experts: bool,
    randomize_parent_experts: bool = True,
    max_parent_experts: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, List[int]]:
    """
    Select expert IDs for an aggregate entry.
    Returns (expert_ids, logp_router, role_expert_id, subtask_local_ids).
    logp_router is 0 for aggregate (router not trained on aggregate expert selection).
    """
    role_id = int(ROLE_EXPERT_IDS["aggregate"])
    role_tensor = torch.tensor([role_id], dtype=torch.long, device=device)
    logp_router = torch.tensor(0.0, dtype=torch.float32, device=device)

    if not merge_use_parent_experts or not parent_subtask_experts:
        return role_tensor, logp_router, role_id, []

    pool = sorted({int(x) for x in parent_subtask_experts if int(x) >= 0})
    selected = list(pool)
    if randomize_parent_experts:
        upper = max(1, min(int(max_parent_experts) if max_parent_experts > 0 else len(pool), len(pool)))
        k = random.randint(1, upper)
        selected = sorted(random.sample(pool, k=k))

    subtask_ids = torch.tensor(
        [int(subtask_expert_offset) + int(x) for x in selected],
        dtype=torch.long,
        device=device,
    )
    expert_ids = torch.cat([role_tensor, subtask_ids])
    return expert_ids, logp_router, role_id, selected


def _subtask_router_reg_loss(
    router: SubtaskRouter,
    l2_weight: float,
    ortho_weight: float,
) -> torch.Tensor:
    device = router.prototypes.device
    l2 = torch.tensor(0.0, device=device)
    ortho = torch.tensor(0.0, device=device)
    if l2_weight > 0.0:
        l2 = (router.prototypes ** 2).mean()
    if ortho_weight > 0.0:
        proto = normalize_title_embedding(router.prototypes)
        gram = proto @ proto.t()
        ident = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        ortho = ((gram - ident) ** 2).mean()
    return (l2_weight * l2) + (ortho_weight * ortho)


def _unique_trainable_parameters(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    seen: set[int] = set()
    out: List[torch.nn.Parameter] = []
    for p in params:
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        out.append(p)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Backbone & model utilities
# ══════════════════════════════════════════════════════════════════════════════

def _load_backbone(*, model_name: str, torch_dtype: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    try:
        if model.config is not None:
            model.config.use_cache = False
    except Exception:
        pass
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    except Exception:
        pass
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# Chat-template wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _wrap_prompt_chat(tokenizer, prompt: str) -> str:
    """Wrap a raw prompt string in the tokenizer's chat template.

    For instruction-tuned models (Qwen3, Llama-3-Instruct, etc.) this ensures
    the model generates an EOS token when done rather than running to
    max_new_tokens.  Falls back to raw prompt if the tokenizer has no template.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking=False: disable Qwen3 chain-of-thought reasoning tokens
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # tokenizer doesn't support enable_thinking (non-Qwen3 model)
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        return prompt


# ══════════════════════════════════════════════════════════════════════════════
# SRDD context prompt builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_execute_prompt_srdd(
    *,
    spec,
    node_id: int,
    parent_codes: Optional[Codes],
) -> str:
    """Build executor prompt for a DAG node."""
    node_meta = spec.node_metadata[node_id]
    subtask_title = str(node_meta.title or "").strip()
    subtask_desc = str(node_meta.description or node_meta.title or "").strip()
    repo_snapshot = ""
    if parent_codes is not None and parent_codes.codebooks:
        repo_snapshot = parent_codes.snapshot_for_prompt()
    return build_executor_prompt(
        task_description=str(spec.task_description or "").strip(),
        subtask_title=subtask_title,
        subtask_description=subtask_desc,
        repo_snapshot=repo_snapshot,
    )


def _build_aggregate_prompt_srdd(
    *,
    spec,
    node_id: int,
    parent_codes_list: List[Codes],
) -> str:
    """Build aggregate prompt merging multiple parent snapshots."""
    parent_snapshots = [c.snapshot_for_prompt() for c in parent_codes_list if c.codebooks]
    return build_aggregate_prompt(
        task_description=str(spec.task_description or "").strip(),
        parent_snapshots=parent_snapshots,
    )


def _subtask_text_for_router(spec, node_id: int) -> str:
    """Short text describing the node for the router's title embedder."""
    meta = spec.node_metadata[node_id]
    title = str(meta.title or "").strip()
    desc  = str(meta.description or "").strip()
    if title and desc:
        return f"{title}: {desc[:120]}"
    return title or desc or "subtask"


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train_grpo(args: argparse.Namespace) -> None:  # noqa: C901
    # ── Distributed setup ──────────────────────────────────────────────────────
    if args.gpus and str(args.gpus).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus).strip()
    os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", str(1 << 20))
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_timeout_s = int(getattr(args, "dist_timeout_s", 21600))
    use_dist = world_size_env > 1

    # 必须在 init_process_group 之前 set_device，否则 NCCL 看到所有 rank 在同一 GPU
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.device)))
    if torch.cuda.is_available():
        visible_count = torch.cuda.device_count()
        cuda_idx = int(local_rank) % int(visible_count) if use_dist else int(args.device)
        torch.cuda.set_device(cuda_idx)
        device = torch.device("cuda", cuda_idx)
    else:
        device = torch.device("cpu")

    if use_dist and not _is_dist_ready():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs: Dict[str, Any] = dict(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=ddp_timeout_s),
        )
        if backend == "nccl" and torch.cuda.is_available():
            init_kwargs["device_id"] = device  # 显式告诉 NCCL 每个 rank 的 GPU
        dist.init_process_group(**init_kwargs)

    world_size = int(dist.get_world_size()) if _is_dist_ready() else 1
    rank = int(dist.get_rank()) if _is_dist_ready() else 0
    is_main = rank == 0

    _seed_everything(int(args.seed) + rank * 100003)

    # ── Route layout ───────────────────────────────────────────────────────────
    group_size = int(args.group_size)
    if world_size > 1 and group_size % world_size != 0:
        raise ValueError(f"--group-size ({group_size}) must be divisible by WORLD_SIZE ({world_size}).")
    local_group_size = max(1, group_size // world_size)
    local_route_count, local_cands_per_route = _resolve_local_route_layout(
        local_group_size=local_group_size,
        requested_local_routes=int(getattr(args, "hierarchical_local_routes", 0)),
    )
    if is_main:
        print(
            f"[hier-credit] group_size={group_size} world_size={world_size} "
            f"local_group_size={local_group_size} local_routes={local_route_count} "
            f"cands_per_route={local_cands_per_route}",
            flush=True,
        )

    # ── Run directory ──────────────────────────────────────────────────────────
    run_name = str(args.run_name or "").strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_root / run_name).resolve()
    ckpt_root = (args.ckpt_root / run_name).resolve()
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)
    if _is_dist_ready():
        dist.barrier()

    detailed_log_dir = run_dir / "detailed_logs"
    if is_main:
        detailed_log_dir.mkdir(parents=True, exist_ok=True)
    if _is_dist_ready():
        dist.barrier()

    sample_perf_jsonl = detailed_log_dir / "sample_performance.jsonl"

    # ── Load SRDD samples ──────────────────────────────────────────────────────
    srdd_csv = Path(args.srdd_csv)
    if not srdd_csv.exists():
        raise FileNotFoundError(f"--srdd-csv not found: {srdd_csv}")
    all_samples = read_srdd_samples(srdd_csv)

    max_samples = int(getattr(args, "max_samples", 0))
    if max_samples > 0:
        all_samples = all_samples[:max_samples]

    if not all_samples:
        raise RuntimeError("No SRDD samples loaded.")

    if is_main:
        print(f"[dataset] srdd_csv={srdd_csv} samples={len(all_samples)}", flush=True)

    # ── Taskgraph index ────────────────────────────────────────────────────────
    taskgraph_root = Path(args.taskgraph_root) if getattr(args, "taskgraph_root", None) else None
    taskgraph_index: Optional[Dict[Tuple[str, str], Path]] = None
    if taskgraph_root is not None:
        if not taskgraph_root.exists():
            raise FileNotFoundError(f"--taskgraph-root not found: {taskgraph_root}")
        taskgraph_index = _load_taskgraph_index(taskgraph_root)
        if not taskgraph_index:
            raise RuntimeError(f"No taskgraphs found under: {taskgraph_root}")
        if is_main:
            print(f"[taskgraph] index size={len(taskgraph_index)}", flush=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    model, tokenizer = _load_backbone(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device=device,
    )

    num_role_experts = len(ROLE_EXPERT_IDS)
    num_subtask_experts = int(args.num_subtask_experts)
    if num_subtask_experts <= 0:
        raise ValueError("--num-subtask-experts must be > 0.")
    num_total_experts = num_role_experts + num_subtask_experts
    subtask_expert_offset = num_role_experts
    subtask_top_k = int(args.subtask_top_k)

    lora_cfg = LoRAConfig(
        num_experts=num_total_experts,
        top_k=subtask_top_k + 1,   # +1 for role expert
        rank=int(args.lora_rank),
        alpha=float(args.lora_alpha),
        target_modules=("q_proj", "v_proj", "o_proj"),
        last_n_layers=int(args.lora_last_n_layers),
    )
    inject_mole_lora(model, cfg=lora_cfg)

    router_cfg = SubtaskRouterConfig(num_experts=num_subtask_experts, top_k=subtask_top_k)
    router = SubtaskRouter(router_cfg).to(device)
    title_embedder = TitleEmbedder(
        model=model, tokenizer=tokenizer, out_dim=router_cfg.title_emb_dim
    ).to(device)

    lora_params = _unique_trainable_parameters(
        [p for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    )
    router_params = _unique_trainable_parameters(list(router.parameters()))
    title_params = _unique_trainable_parameters(
        [p for n, p in title_embedder.named_parameters() if not n.startswith("_model.") and p.requires_grad]
    )

    opt_lora = torch.optim.AdamW(lora_params, lr=float(args.lora_lr))
    opt_router = torch.optim.AdamW(
        _unique_trainable_parameters(router_params + title_params),
        lr=float(args.router_lr),
    )

    all_trainable = _unique_trainable_parameters(lora_params + router_params + title_params)

    # ── Init / resume checkpoint ───────────────────────────────────────────────
    loaded_state: Dict[str, Any] = {}
    init_ckpt_path = getattr(args, "init_checkpoint", None)
    if init_ckpt_path is not None:
        init_ckpt = Path(init_ckpt_path)
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-checkpoint not found: {init_ckpt}")
        loaded_state = load_checkpoint(
            ckpt_dir=init_ckpt,
            device=device,
            router=router,
            title_embedder=title_embedder,
            model=model,
            opt_router=opt_router,
            opt_lora=opt_lora,
        )
        if is_main:
            print(f"[resume] loaded from {init_ckpt}", flush=True)

    gen_cfg = GenerationConfig(
        model_name=args.model_name,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        torch_dtype=str(args.torch_dtype),
        device=int(device.index or 0),
    )
    mole_gen = MoLEGenerator(model=model, tokenizer=tokenizer, device=device, gen_cfg=gen_cfg)

    # ── Training-loop state ────────────────────────────────────────────────────
    global_step = int(loaded_state.get("global_step", 0))
    start_epoch = int(loaded_state.get("epoch", 0)) + 1
    if start_epoch > int(args.epochs):
        if is_main:
            print(f"[resume] already at epoch {start_epoch - 1} >= {args.epochs}; done.")
        return

    # ── HGRPO hyper-parameters (read once) ────────────────────────────────────
    credit_mode = str(getattr(args, "credit_mode", "hierarchical")).strip().lower()
    router_adv_mode = str(getattr(args, "router_adv_mode", "mean_center")).strip().lower()
    execute_route_policy = str(getattr(args, "execute_route_policy", "sample")).strip().lower()
    router_contrast_weight = max(0.0, float(getattr(args, "router_contrast_weight", 0.0)))
    sampled_route_weight = max(0.0, float(getattr(args, "sampled_route_weight", 1.0)))
    sampled_positive_adv_only = bool(getattr(args, "sampled_positive_adv_only", False))
    sampled_drop_below_anchor_delta = float(getattr(args, "sampled_drop_below_anchor_delta", -1.0))
    router_diversity_reg = float(getattr(args, "router_diversity_reg", 0.02))
    alpha_router = float(getattr(args, "alpha_router", 0.15))
    grpo_adv_normalize = bool(getattr(args, "grpo_adv_normalize", True))
    grpo_adv_eps = float(getattr(args, "grpo_adv_eps", 1e-6))
    advantage_clip = float(getattr(args, "advantage_clip", 5.0))
    grpo_skip_allzero = bool(getattr(args, "grpo_skip_update_if_allzero", False))
    grpo_skip_low_std = bool(getattr(args, "grpo_skip_update_if_low_std", False))
    grpo_min_reward_std = float(getattr(args, "grpo_min_reward_std", 0.0005))
    enforce_expert_diversity = bool(getattr(args, "enforce_local_expert_diversity", True))
    enforce_output_diversity = bool(getattr(args, "enforce_local_output_diversity", False))
    enforce_cross_rank_diversity = bool(getattr(args, "enforce_cross_rank_route_diversity", True))
    diversity_max_resample = max(1, int(getattr(args, "diversity_max_resample", 16)))
    merge_use_parent_experts = bool(getattr(args, "merge_use_parent_experts", True))
    aggregate_random_parent_experts = bool(getattr(args, "aggregate_random_parent_experts", True))
    aggregate_parent_max_experts = int(getattr(args, "aggregate_parent_max_experts", 0))
    if aggregate_parent_max_experts <= 0:
        aggregate_parent_max_experts = subtask_top_k
    include_execute = bool(getattr(args, "include_execute_entries", True))
    include_aggregate = bool(getattr(args, "include_aggregate_entries", True))
    aggregate_min_parents = int(getattr(args, "aggregate_min_parents", 2))
    smoke_timeout_s = int(getattr(args, "smoke_timeout_s", 15))
    save_every_epochs = int(getattr(args, "save_every_epochs", 1))
    save_every_samples = int(getattr(args, "save_every_samples", 0))

    # Proxy reward weights (matching old train_mole_srdd_taskgraph_role_subtask defaults)
    reward_w_smoke      = float(getattr(args, "proxy_w_smoke",          0.5))
    reward_w_comp       = float(getattr(args, "proxy_w_comp",           0.5))
    reward_w_cons_strip = float(getattr(args, "proxy_w_cons_strip",     1.0))
    reward_w_cons_task  = float(getattr(args, "proxy_cons_task_weight", 0.7))
    reward_w_cons_sub   = float(getattr(args, "proxy_cons_subtask_weight", 0.3))

    if is_main:
        print(
            f"[hgrpo] credit_mode={credit_mode} router_adv_mode={router_adv_mode} "
            f"execute_route_policy={execute_route_policy} "
            f"sampled_positive_adv_only={sampled_positive_adv_only} "
            f"sampled_route_weight={sampled_route_weight} "
            f"router_contrast_weight={router_contrast_weight}",
            flush=True,
        )
        print(
            f"[entries] include_execute={include_execute} "
            f"include_aggregate={include_aggregate} "
            f"aggregate_min_parents={aggregate_min_parents}",
            flush=True,
        )

    # ════════════════════════════════════════════════════════════════════════════
    # Training epochs
    # ════════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, int(args.epochs) + 1):
        # Shuffle sample order consistently across ranks
        epoch_samples = list(all_samples)
        if bool(getattr(args, "shuffle_samples", True)):
            rng = random.Random(int(args.seed) + epoch * 31337)
            rng.shuffle(epoch_samples)

        epoch_pass_rewards: List[float] = []
        epoch_smoke_rates:  List[float] = []
        epoch_comp_means:   List[float] = []
        epoch_cons_strip_means: List[float] = []
        epoch_global_steps_start = int(global_step)

        for sample_idx, sample in enumerate(epoch_samples, start=1):
            if is_main:
                print(
                    f"[epoch {epoch}/{args.epochs}] sample {sample_idx}/{len(epoch_samples)}: "
                    f"{sample.category}/{sample.name}",
                    flush=True,
                )

            # ── Find task graph ────────────────────────────────────────────────
            graph_path: Optional[Path] = None
            if taskgraph_index is not None:
                graph_path = taskgraph_index.get((sample.category, sample.name))
                if graph_path is None:
                    if is_main:
                        print(
                            f"  [skip] no taskgraph for {sample.category}/{sample.name}",
                            flush=True,
                        )
                    continue
            else:
                # Dynamic graph generation not supported in HGRPO mode
                if is_main:
                    print(
                        "  [skip] --taskgraph-root required for HGRPO training.",
                        flush=True,
                    )
                continue

            # ── Parse taskgraph ────────────────────────────────────────────────
            try:
                spec = convert_taskgraph(graph_path)
            except Exception as exc:
                if is_main:
                    print(f"  [skip] taskgraph parse error: {exc}", flush=True)
                continue

            node_ids_all = sorted(spec.node_metadata)
            predecessors = _build_predecessors(spec.edge_strings, node_ids_all)
            node_order = _topological_order(node_ids_all, predecessors)

            # ── Per-sample state ───────────────────────────────────────────────
            # solutions[node_id] = Codes of the best candidate for that node
            # (rank-0's best, broadcast to all ranks after each node)
            solutions: Dict[int, Codes] = {}
            # subtask experts used by each node's winner (for aggregate context)
            node_subtask_experts: Dict[int, List[int]] = {}

            sample_node_dir = run_dir / _sanitize(sample.category) / _sanitize(sample.name)

            for node_id in node_order:
                node_meta = spec.node_metadata[node_id]
                node_task_desc    = str(spec.task_description or sample.description or "").strip()
                node_subtask_desc = str(node_meta.description or node_meta.title or "").strip()
                parent_ids = [
                    pid for pid in predecessors.get(node_id, []) if pid in solutions
                ]
                n_parents = len(parent_ids)

                # Decide whether this is execute or aggregate
                is_aggregate = n_parents >= aggregate_min_parents
                kind = "aggregate" if is_aggregate else "execute"

                if is_aggregate and not include_aggregate:
                    # Not training aggregate – still need to run greedily for context
                    parent_codes_list = [solutions[pid] for pid in parent_ids]
                    prompt = _wrap_prompt_chat(tokenizer, _build_aggregate_prompt_srdd(
                        spec=spec, node_id=node_id, parent_codes_list=parent_codes_list
                    ))
                    subtask_text = _subtask_text_for_router(spec, node_id)
                    parent_subtask_union = sorted(
                        {x for pid in parent_ids for x in node_subtask_experts.get(pid, [])}
                    )
                    expert_ids, _, _, _ = _select_aggregate_experts(
                        device=device,
                        parent_subtask_experts=parent_subtask_union,
                        subtask_expert_offset=subtask_expert_offset,
                        merge_use_parent_experts=merge_use_parent_experts,
                    )
                    text, _, _ = mole_gen.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
                    codes = Codes(text)
                    solutions[node_id] = codes
                    node_subtask_experts[node_id] = [
                        int(x) - subtask_expert_offset
                        for x in expert_ids.detach().cpu().tolist()[1:]
                        if 0 <= int(x) - subtask_expert_offset < num_subtask_experts
                    ]
                    continue

                if not is_aggregate and not include_execute:
                    # Not training execute – run greedily for context
                    parent_codes = solutions[parent_ids[0]] if parent_ids else None
                    prompt = _wrap_prompt_chat(tokenizer, _build_execute_prompt_srdd(
                        spec=spec, node_id=node_id, parent_codes=parent_codes
                    ))
                    subtask_text = _subtask_text_for_router(spec, node_id)
                    expert_ids, _, _, _, _, _ = _select_execute_experts(
                        router=router,
                        title_embedder=title_embedder,
                        device=device,
                        subtask_text=subtask_text,
                        subtask_expert_offset=subtask_expert_offset,
                        selection_mode="greedy",
                    )
                    text, _, _ = mole_gen.generate_with_experts(prompt=prompt, expert_ids=expert_ids)
                    solutions[node_id] = Codes(text)
                    continue

                # ── Build prompt for this node ─────────────────────────────────
                if is_aggregate:
                    parent_codes_list = [solutions[pid] for pid in parent_ids]
                    prompt = _wrap_prompt_chat(tokenizer, _build_aggregate_prompt_srdd(
                        spec=spec, node_id=node_id, parent_codes_list=parent_codes_list
                    ))
                else:
                    parent_codes = solutions[parent_ids[0]] if parent_ids else None
                    prompt = _wrap_prompt_chat(tokenizer, _build_execute_prompt_srdd(
                        spec=spec, node_id=node_id, parent_codes=parent_codes
                    ))

                subtask_text = _subtask_text_for_router(spec, node_id)
                node_dir = sample_node_dir / f"node_{node_id:02d}"
                node_dir.mkdir(parents=True, exist_ok=True)

                parent_subtask_union = sorted(
                    {x for pid in parent_ids for x in node_subtask_experts.get(pid, [])}
                )

                # ──────────────────────────────────────────────────────────────
                # HGRPO: generate candidates for all local routes
                # ──────────────────────────────────────────────────────────────
                # NOTE: used_expert_signatures is treated as a "globally committed"
                # set when enforce_cross_rank_diversity is on — after each
                # route_local_idx iteration we all_gather every rank's final sig
                # and add them all here, so the next iteration's sampling (and
                # forced-fallback) naturally avoid cross-rank duplicates.
                used_expert_signatures: set[Tuple[int, ...]] = set()
                route_groups: List[Dict[str, Any]] = []
                local_cands: List[SRDDHGRPOCandidate] = []

                _cross_rank_active = (
                    enforce_cross_rank_diversity
                    and enforce_expert_diversity
                    and world_size > 1
                    and _is_dist_ready()
                    and execute_route_policy != "greedy"
                )

                def _sample_execute_sig(
                    _avoid: set,
                    _mode: str,
                ) -> Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor, int, List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
                    """Sample once with resample loop + forced fallback, avoiding `_avoid`."""
                    _expert_ids = torch.empty(0, dtype=torch.long, device=device)
                    _logp_router = torch.tensor(0.0, device=device)
                    _role_expert_id = 0
                    _chosen_subtask_local: List[int] = []
                    _router_logits: Optional[torch.Tensor] = None
                    _router_probs: Optional[torch.Tensor] = None
                    _sig: Tuple[int, ...] = ()
                    for _ in range(diversity_max_resample):
                        (
                            _expert_ids, _logp_router, _role_expert_id,
                            _chosen_subtask_local, _router_logits, _router_probs,
                        ) = _select_execute_experts(
                            router=router,
                            title_embedder=title_embedder,
                            device=device,
                            subtask_text=subtask_text,
                            subtask_expert_offset=subtask_expert_offset,
                            selection_mode=_mode,
                        )
                        _sig = tuple(int(x) for x in _expert_ids.detach().cpu().tolist())
                        if (not enforce_expert_diversity) or (_sig not in _avoid):
                            break
                    # Forced fallback (skip for greedy/anchor — deterministic pick)
                    if (
                        enforce_expert_diversity
                        and _mode != "greedy"
                        and _sig in _avoid
                        and _router_logits is not None
                        and num_subtask_experts > 0
                        and subtask_top_k > 0
                    ):
                        all_combos = list(itertools.combinations(range(num_subtask_experts), subtask_top_k))
                        _probs = torch.softmax(_router_logits[0].detach(), dim=-1)
                        unused = [
                            c for c in all_combos
                            if tuple(sorted(
                                [int(ROLE_EXPERT_IDS["execute"])]
                                + [subtask_expert_offset + x for x in c]
                            ))
                            not in _avoid
                        ]
                        if unused:
                            best_combo = max(unused, key=lambda c: sum(float(_probs[i]) for i in c))
                            forced_ids = torch.tensor(
                                [int(ROLE_EXPERT_IDS["execute"])] + [subtask_expert_offset + x for x in best_combo],
                                dtype=torch.long, device=device,
                            )
                            _logp_router = sum(torch.log(_probs[i] + 1e-12) for i in best_combo)
                            _expert_ids = forced_ids
                            _chosen_subtask_local = list(best_combo)
                            _sig = tuple(int(x) for x in _expert_ids.detach().cpu().tolist())
                    return (
                        _sig, _expert_ids, _logp_router, _role_expert_id,
                        _chosen_subtask_local, _router_logits, _router_probs,
                    )

                for route_local_idx in range(1, local_route_count + 1):
                    route_global_idx = rank * local_route_count + route_local_idx

                    # ── Expert selection ───────────────────────────────────────
                    expert_ids = torch.empty(0, dtype=torch.long, device=device)
                    logp_router = torch.tensor(0.0, device=device)
                    role_expert_id = 0
                    chosen_subtask_local: List[int] = []
                    route_is_anchor = False
                    route_selection_mode = "sample"
                    router_logits: Optional[torch.Tensor] = None
                    router_probs: Optional[torch.Tensor] = None
                    chosen_sig: Tuple[int, ...] = ()

                    if kind == "execute":
                        if execute_route_policy == "greedy":
                            route_is_anchor = True
                            route_selection_mode = "greedy"
                        elif execute_route_policy == "hybrid":
                            # route_global_idx == 1 is the anchor (rank 0, route 1)
                            route_is_anchor = route_global_idx == 1
                            route_selection_mode = "greedy" if route_is_anchor else "sample"
                        else:  # "sample" (exp2, exp3)
                            route_is_anchor = False
                            route_selection_mode = "sample"

                        (
                            chosen_sig, expert_ids, logp_router, role_expert_id,
                            chosen_subtask_local, router_logits, router_probs,
                        ) = _sample_execute_sig(used_expert_signatures, route_selection_mode)

                        # ── Cross-rank collision resolution ──
                        # All ranks participate in the collectives; anchor ranks
                        # (rank 0 in hybrid at route_local_idx==1) naturally keep
                        # their sig since range(rank=0) is empty → my_collision=False.
                        # Non-anchor ranks whose sig matches any lower-rank sig
                        # resample against the merged avoid set.
                        if _cross_rank_active:
                            for _sync_round in range(diversity_max_resample):
                                sigs_per_rank: List[Optional[Tuple[int, ...]]] = [None] * world_size
                                dist.all_gather_object(sigs_per_rank, chosen_sig)
                                lower_sigs = {
                                    sigs_per_rank[r] for r in range(rank)
                                    if sigs_per_rank[r] is not None
                                }
                                # Anchor rank must keep its greedy sig: it doesn't
                                # signal collision even if a (lower) rank somehow
                                # landed on the same combo — but anchor is always
                                # rank 0 here so lower_sigs is empty anyway.
                                my_collision = (
                                    (not route_is_anchor)
                                    and (chosen_sig in lower_sigs)
                                )
                                collision_t = torch.tensor(
                                    [1 if my_collision else 0],
                                    device=device, dtype=torch.int32,
                                )
                                dist.all_reduce(collision_t, op=dist.ReduceOp.MAX)
                                if int(collision_t.item()) == 0:
                                    break
                                if my_collision:
                                    avoid = used_expert_signatures | lower_sigs
                                    (
                                        chosen_sig, expert_ids, logp_router, role_expert_id,
                                        chosen_subtask_local, router_logits, router_probs,
                                    ) = _sample_execute_sig(avoid, route_selection_mode)
                    else:  # aggregate — cross-rank dedup doesn't apply (parent-driven)
                        route_is_anchor = False
                        route_selection_mode = "aggregate"
                        router_logits = None
                        router_probs = None
                        for _ in range(diversity_max_resample):
                            expert_ids, logp_router, role_expert_id, chosen_subtask_local = (
                                _select_aggregate_experts(
                                    device=device,
                                    parent_subtask_experts=parent_subtask_union,
                                    subtask_expert_offset=subtask_expert_offset,
                                    merge_use_parent_experts=merge_use_parent_experts,
                                    randomize_parent_experts=aggregate_random_parent_experts,
                                    max_parent_experts=aggregate_parent_max_experts,
                                )
                            )
                            chosen_sig = tuple(int(x) for x in expert_ids.detach().cpu().tolist())
                            if (not enforce_expert_diversity) or (chosen_sig not in used_expert_signatures):
                                break

                    # Commit: update local set; for execute w/ cross-rank active,
                    # also add every rank's final sig so the next route iteration
                    # (and any other ranks' future samples) avoid them all.
                    if _cross_rank_active and kind == "execute":
                        final_sigs: List[Optional[Tuple[int, ...]]] = [None] * world_size
                        dist.all_gather_object(final_sigs, chosen_sig)
                        for s in final_sigs:
                            if s is not None:
                                used_expert_signatures.add(s)
                    else:
                        used_expert_signatures.add(chosen_sig)

                    # ── Generate candidates for this route ─────────────────────
                    route_cands: List[SRDDHGRPOCandidate] = []
                    used_texts: set[str] = set()

                    # Batch generate
                    _t_gen0 = time.time()
                    batch_gen: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
                    if local_cands_per_route > 1:
                        batch_gen = list(mole_gen.generate_n_with_experts(
                            prompt=prompt,
                            expert_ids=expert_ids,
                            num_samples=local_cands_per_route,
                        ))
                        if enforce_output_diversity:
                            _seen: set[str] = set()
                            _colliding: List[int] = []
                            for _i, (txt, _, _) in enumerate(batch_gen):
                                sig = txt.strip()
                                if sig not in _seen:
                                    _seen.add(sig)
                                else:
                                    _colliding.append(_i)
                            for _ in range(diversity_max_resample):
                                if not _colliding:
                                    break
                                retried = list(mole_gen.generate_n_with_experts(
                                    prompt=prompt,
                                    expert_ids=expert_ids,
                                    num_samples=len(_colliding),
                                ))
                                still = []
                                for _j, _slot in enumerate(_colliding):
                                    rtxt = retried[_j][0].strip()
                                    if rtxt not in _seen:
                                        batch_gen[_slot] = retried[_j]
                                        _seen.add(rtxt)
                                    else:
                                        still.append(_slot)
                                _colliding = still

                    _t_gen1 = time.time()
                    if is_main:
                        _gen_lens = [batch_gen[s][2].numel() if batch_gen else 0 for s in range(local_cands_per_route)]
                        _prompt_len = batch_gen[0][1].numel() if batch_gen else 0
                        print(f"  [timing] rank0 node={node_id} gen={_t_gen1-_t_gen0:.1f}s prompt_len={_prompt_len} gen_lens={_gen_lens}", flush=True)

                    # ── Collect generation data and evaluate candidates ────────────
                    if batch_gen:
                        # All texts already generated in batch; run smoke evaluations
                        # for all slots in parallel (4 threads → 1× timeout instead of 4×).
                        _slot_gen = []
                        for _s in range(local_cands_per_route):
                            _local_idx = (route_local_idx - 1) * local_cands_per_route + _s + 1
                            _cand_idx  = rank * local_group_size + _local_idx
                            _text, _pids, _gids = batch_gen[_s]
                            used_texts.add(_text.strip())
                            _slot_gen.append((_s, _cand_idx, _text, _pids, _gids))

                        # Capture loop-local vars for the closure.
                        _nd, _sto, _ntd, _nsd = node_dir, smoke_timeout_s, node_task_desc, node_subtask_desc
                        _ws, _wc, _wcs, _wct, _wcsub = (
                            reward_w_smoke, reward_w_comp, reward_w_cons_strip,
                            reward_w_cons_task, reward_w_cons_sub,
                        )

                        def _eval_slot(args):
                            _s, _ci, _txt, _pids, _gids = args
                            _codes = Codes(_txt)
                            _rew, _pass, _det, _pm = _evaluate_srdd_candidate(
                                codes=_codes,
                                node_dir=_nd,
                                timeout_s=_sto,
                                candidate_idx=_ci,
                                task_description=_ntd,
                                subtask_description=_nsd,
                                w_smoke=_ws,
                                w_comp=_wc,
                                w_cons_strip=_wcs,
                                w_cons_task=_wct,
                                w_cons_subtask=_wcsub,
                            )
                            return _s, _ci, _txt, _pids, _gids, _codes, _rew, _pass, _pm

                        with ThreadPoolExecutor(max_workers=len(_slot_gen)) as _pool:
                            _eval_results = list(_pool.map(_eval_slot, _slot_gen))

                        for _s, cand_idx, text, prompt_ids, gen_ids, codes, reward, passed, proxy_metrics in _eval_results:
                            top1_prob = 0.0
                            top12_margin = 0.0
                            if router_probs is not None and router_probs.numel() > 0:
                                top1_prob = float(torch.max(router_probs).detach().cpu().item())
                            if router_logits is not None and router_logits.numel() >= 2:
                                top2 = torch.topk(router_logits, k=2).values
                                top12_margin = float((top2[0] - top2[1]).detach().cpu().item())
                            cand = SRDDHGRPOCandidate(
                                idx=cand_idx,
                                route_local_idx=route_local_idx,
                                route_global_idx=route_global_idx,
                                text=text,
                                codes=codes,
                                prompt_ids=prompt_ids,
                                gen_ids=gen_ids,
                                expert_ids=expert_ids,
                                expert_weights=None,
                                logp_router=logp_router,
                                role_expert_id=role_expert_id,
                                subtask_expert_ids=list(chosen_subtask_local),
                                route_is_anchor=route_is_anchor,
                                route_selection_mode=route_selection_mode,
                                reward=float(reward),
                                smoke_passed=bool(passed),
                                metrics={
                                    **proxy_metrics,
                                    "reward": float(reward),
                                    "step_score": float(reward),
                                    "router_top1_prob": float(top1_prob),
                                    "router_top12_margin": float(top12_margin),
                                    "router_logits": router_logits,
                                    "router_probs": router_probs,
                                },
                            )
                            local_cands.append(cand)
                            route_cands.append(cand)
                    else:
                        # Fallback: sequential generation with diversity resampling.
                        for slot in range(local_cands_per_route):
                            local_idx = (route_local_idx - 1) * local_cands_per_route + slot + 1
                            cand_idx = rank * local_group_size + local_idx
                            for _ in range(diversity_max_resample):
                                text, prompt_ids, gen_ids = mole_gen.generate_with_experts(
                                    prompt=prompt, expert_ids=expert_ids
                                )
                                if (not enforce_output_diversity) or (text.strip() not in used_texts):
                                    break
                            used_texts.add(text.strip())
                            codes = Codes(text)
                            reward, passed, _detail, proxy_metrics = _evaluate_srdd_candidate(
                                codes=codes,
                                node_dir=node_dir,
                                timeout_s=smoke_timeout_s,
                                candidate_idx=cand_idx,
                                task_description=node_task_desc,
                                subtask_description=node_subtask_desc,
                                w_smoke=reward_w_smoke,
                                w_comp=reward_w_comp,
                                w_cons_strip=reward_w_cons_strip,
                                w_cons_task=reward_w_cons_task,
                                w_cons_subtask=reward_w_cons_sub,
                            )
                            top1_prob = 0.0
                            top12_margin = 0.0
                            if router_probs is not None and router_probs.numel() > 0:
                                top1_prob = float(torch.max(router_probs).detach().cpu().item())
                            if router_logits is not None and router_logits.numel() >= 2:
                                top2 = torch.topk(router_logits, k=2).values
                                top12_margin = float((top2[0] - top2[1]).detach().cpu().item())
                            cand = SRDDHGRPOCandidate(
                                idx=cand_idx,
                                route_local_idx=route_local_idx,
                                route_global_idx=route_global_idx,
                                text=text,
                                codes=codes,
                                prompt_ids=prompt_ids,
                                gen_ids=gen_ids,
                                expert_ids=expert_ids,
                                expert_weights=None,
                                logp_router=logp_router,
                                role_expert_id=role_expert_id,
                                subtask_expert_ids=list(chosen_subtask_local),
                                route_is_anchor=route_is_anchor,
                                route_selection_mode=route_selection_mode,
                                reward=float(reward),
                                smoke_passed=bool(passed),
                                metrics={
                                    **proxy_metrics,
                                    "reward": float(reward),
                                    "step_score": float(reward),
                                    "router_top1_prob": float(top1_prob),
                                    "router_top12_margin": float(top12_margin),
                                    "router_logits": router_logits,
                                    "router_probs": router_probs,
                                },
                            )
                            local_cands.append(cand)
                            route_cands.append(cand)

                    route_mean_local = float(
                        sum(c.reward for c in route_cands) / len(route_cands)
                    ) if route_cands else 0.0
                    route_std_local = _safe_std([c.reward for c in route_cands])

                    route_groups.append({
                        "route_local_idx": route_local_idx,
                        "route_global_idx": route_global_idx,
                        "logp_router": logp_router,
                        "cands": route_cands,
                        "route_is_anchor": route_is_anchor,
                        "route_selection_mode": route_selection_mode,
                        "router_logits": router_logits,
                        "router_probs": router_probs,
                        "reward_mean_local": route_mean_local,
                        "reward_std_local": route_std_local,
                    })

                # ── All-reduce rewards across ranks ────────────────────────────
                local_rewards = [float(c.reward) for c in local_cands]
                rewards_global = _all_gather_float_list(local_rewards, device=device, world_size=world_size)
                reward_mean = float(sum(rewards_global) / len(rewards_global)) if rewards_global else 0.0
                reward_std = _safe_std(rewards_global)

                # ── Proxy component metrics (benchmark-aligned: smoke / comp / cons_strip) ──
                local_smoke     = [float(bool(c.metrics.get("smoke_passed", False))) for c in local_cands]
                local_comp      = [float(c.metrics.get("proxy_completeness", 0.0)) for c in local_cands]
                local_cons_task = [float(c.metrics.get("proxy_consistency_task", 0.0)) for c in local_cands]
                local_cons_sub  = [float(c.metrics.get("proxy_consistency_subtask", 0.0)) for c in local_cands]
                smoke_global     = _all_gather_float_list(local_smoke,     device=device, world_size=world_size)
                comp_global      = _all_gather_float_list(local_comp,      device=device, world_size=world_size)
                cons_task_global = _all_gather_float_list(local_cons_task, device=device, world_size=world_size)
                cons_sub_global  = _all_gather_float_list(local_cons_sub,  device=device, world_size=world_size)
                _ncg = max(1, len(smoke_global))
                smoke_pass_rate  = float(sum(smoke_global)     / _ncg)
                comp_mean        = float(sum(comp_global)      / _ncg)
                cons_task_mean   = float(sum(cons_task_global) / _ncg)
                cons_sub_mean    = float(sum(cons_sub_global)  / _ncg)
                cons_strip_mean  = float(reward_w_cons_task * cons_task_mean
                                         + reward_w_cons_sub * cons_sub_mean)

                local_route_means = [float(rg["reward_mean_local"]) for rg in route_groups]
                route_means_global = _all_gather_float_list(local_route_means, device=device, world_size=world_size)
                route_reward_mean = float(sum(route_means_global) / len(route_means_global)) if route_means_global else 0.0
                route_reward_std = _safe_std(route_means_global)

                # Global anchor/sampled means (for anchor_vs_others mode)
                anc_means_local = [float(rg["reward_mean_local"]) for rg in route_groups if rg["route_is_anchor"]]
                smp_means_local = [float(rg["reward_mean_local"]) for rg in route_groups if not rg["route_is_anchor"]]
                anc_valid_local = [1.0 if rg["route_is_anchor"] else 0.0 for rg in route_groups]
                smp_valid_local = [1.0 if not rg["route_is_anchor"] else 0.0 for rg in route_groups]

                g_anc = _all_gather_float_list(
                    [float(anc_means_local[0]) if anc_means_local else 0.0], device=device, world_size=world_size
                )
                g_anc_valid = _all_gather_float_list(
                    [1.0 if anc_means_local else 0.0], device=device, world_size=world_size
                )
                g_smp = _all_gather_float_list(
                    [float(smp_means_local[0]) if smp_means_local else 0.0], device=device, world_size=world_size
                )
                g_smp_valid = _all_gather_float_list(
                    [1.0 if smp_means_local else 0.0], device=device, world_size=world_size
                )
                _valid_anc = [m for m, v in zip(g_anc, g_anc_valid) if v > 0.5]
                _valid_smp = [m for m, v in zip(g_smp, g_smp_valid) if v > 0.5]
                anchor_route_mean_global: Optional[float] = (
                    float(sum(_valid_anc) / len(_valid_anc)) if _valid_anc else None
                )
                sampled_route_mean_global: Optional[float] = (
                    float(sum(_valid_smp) / len(_valid_smp)) if _valid_smp else None
                )

                # ── Skip-update checks ─────────────────────────────────────────
                did_update = True
                update_skip_reason = ""
                if grpo_skip_allzero and rewards_global and (max(rewards_global) - min(rewards_global) == 0.0):
                    did_update = False
                    update_skip_reason = "all_rewards_equal"
                if did_update and grpo_skip_low_std and rewards_global and reward_std < grpo_min_reward_std:
                    did_update = False
                    update_skip_reason = "low_reward_std"

                # ── HGRPO credit assignment & backward ─────────────────────────
                loss_value = 0.0
                loss_grpo_value = 0.0
                loss_router_value = 0.0

                # NOTE: do not gate on `local_cands` here. `did_update` is
                # computed from globally all-gathered rewards and is identical
                # on every rank. `local_cands` is per-rank; gating on it would
                # make one rank skip the block (and its collectives) while
                # others enter, causing NCCL deadlock. If local_cands is empty
                # the inner loops simply produce no loss terms and has_grad
                # stays False (synced below).
                if did_update:
                    opt_router.zero_grad(set_to_none=True)
                    opt_lora.zero_grad(set_to_none=True)
                    total_loss = torch.tensor(0.0, device=device)
                    has_grad = False

                    for route_group in route_groups:
                        route_mean_local = float(route_group["reward_mean_local"])
                        route_std_local = float(route_group["reward_std_local"])
                        route_is_anchor = bool(route_group["route_is_anchor"])

                        for cand in route_group["cands"]:
                            if credit_mode == "flat":
                                # Exp3: standard GRPO global advantage
                                adv = float(cand.reward) - reward_mean
                                if grpo_adv_normalize:
                                    adv = adv / (reward_std + grpo_adv_eps) if reward_std > 0.0 else 0.0
                                adv = _clip_advantage(adv, advantage_clip)
                                lora_adv = adv
                                router_adv_flat = adv
                            else:
                                # Exp1/Exp2: hierarchical credit
                                route_lora_w = 1.0
                                if kind == "execute" and not route_is_anchor:
                                    route_lora_w = sampled_route_weight
                                    if (
                                        sampled_drop_below_anchor_delta >= 0.0
                                        and anchor_route_mean_global is not None
                                        and route_mean_local < anchor_route_mean_global - sampled_drop_below_anchor_delta
                                    ):
                                        route_lora_w = 0.0
                                adv = float(cand.reward) - route_mean_local
                                if grpo_adv_normalize:
                                    adv = adv / (route_std_local + grpo_adv_eps) if route_std_local > 0.0 else 0.0
                                adv = _clip_advantage(adv, advantage_clip)
                                if (
                                    kind == "execute"
                                    and not cand.route_is_anchor
                                    and sampled_positive_adv_only
                                    and adv < 0.0
                                ):
                                    adv = 0.0
                                lora_adv = adv * route_lora_w
                                router_adv_flat = None

                            # LoRA update
                            if abs(lora_adv) > 1e-12:
                                logp_sum = mole_gen.logprob_of_generation(
                                    prompt_ids=cand.prompt_ids,
                                    gen_ids=cand.gen_ids,
                                    expert_ids=cand.expert_ids,
                                )
                                gen_len = max(1.0, float(cand.gen_ids.numel()))
                                logp_mean = logp_sum / gen_len
                                loss_i = -(torch.tensor(lora_adv, device=device) * logp_mean)
                                total_loss = total_loss + loss_i
                                loss_grpo_value += float(loss_i.detach().cpu().item())
                                has_grad = True

                            # Flat-mode router update (per-candidate)
                            if (
                                credit_mode == "flat"
                                and router_adv_flat is not None
                                and abs(float(router_adv_flat)) > 1e-12
                            ):
                                lp_r = route_group.get("logp_router")
                                if lp_r is not None:
                                    n_in_route = max(1, len(route_group["cands"]))
                                    r_loss = -(
                                        torch.tensor(float(router_adv_flat), device=device)
                                        * alpha_router * lp_r / n_in_route
                                    )
                                    total_loss = total_loss + r_loss
                                    loss_router_value += float(r_loss.detach().cpu().item())
                                    has_grad = True

                        # Hierarchical router update (per-route)
                        if credit_mode != "flat" and kind == "execute":
                            if (
                                router_adv_mode == "anchor_vs_others"
                                and anchor_route_mean_global is not None
                                and sampled_route_mean_global is not None
                            ):
                                if route_is_anchor:
                                    route_adv = route_mean_local - float(sampled_route_mean_global)
                                else:
                                    route_adv = router_contrast_weight * (
                                        route_mean_local - float(anchor_route_mean_global)
                                    )
                            else:
                                route_adv = route_mean_local - route_reward_mean
                            if grpo_adv_normalize:
                                route_adv = route_adv / (route_reward_std + grpo_adv_eps) if route_reward_std > 0.0 else 0.0
                            route_adv = _clip_advantage(route_adv, advantage_clip)
                            if abs(route_adv) > 1e-12:
                                r_loss = -(
                                    torch.tensor(route_adv, device=device)
                                    * alpha_router * route_group["logp_router"]
                                )
                                total_loss = total_loss + r_loss
                                loss_router_value += float(r_loss.detach().cpu().item())
                                has_grad = True

                    # Router diversity regularization
                    if router_diversity_reg > 0.0 and kind == "execute":
                        for rg in route_groups:
                            probs = rg.get("router_probs")
                            if probs is not None:
                                entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)))
                                total_loss = total_loss - float(router_diversity_reg) * entropy
                                has_grad = True

                    # Prototype regularization
                    if float(args.subtask_proto_l2) > 0.0 or float(args.subtask_proto_ortho) > 0.0:
                        reg = _subtask_router_reg_loss(router, float(args.subtask_proto_l2), float(args.subtask_proto_ortho))
                        if float(reg.item()) != 0.0:
                            total_loss = total_loss + reg
                            has_grad = True

                    # Sync has_grad across ranks: if ANY rank has gradients,
                    # ALL ranks must enter _average_gradients so NCCL collectives
                    # stay symmetric (ranks with no local grad contribute zeros).
                    any_has_grad = has_grad
                    if _is_dist_ready():
                        hg_t = torch.tensor([1 if has_grad else 0], dtype=torch.long, device=device)
                        dist.all_reduce(hg_t, op=dist.ReduceOp.MAX)
                        any_has_grad = bool(hg_t.item())

                    if any_has_grad:
                        _t_bwd0 = time.time()
                        if has_grad:
                            total_loss.backward()
                        _t_bwd1 = time.time()
                        if is_main:
                            print(f"  [timing] backward={_t_bwd1-_t_bwd0:.1f}s", flush=True)
                        _average_gradients(all_trainable, world_size)
                        if has_grad:
                            opt_router.step()
                            opt_lora.step()
                            loss_value = float(total_loss.detach().cpu().item())

                # ── Select best candidate and sync context ─────────────────────
                best_local = max(local_cands, key=lambda c: c.reward) if local_cands else None
                best_reward_local = float(best_local.reward) if best_local else 0.0

                # All-reduce to find which rank has the best candidate
                best_reward_global: float = best_reward_local
                if _is_dist_ready():
                    reward_t = torch.tensor([best_reward_local], dtype=torch.float32, device=device)
                    dist.all_reduce(reward_t, op=dist.ReduceOp.MAX)
                    best_reward_global = float(reward_t.item())

                # Identify which rank owns the best candidate
                is_best_rank = abs(best_reward_local - best_reward_global) < 1e-6
                # Rank 0 wins tiebreaker
                winner_rank = 0
                if _is_dist_ready():
                    cand_rank_t = torch.tensor(
                        [rank if is_best_rank else world_size],
                        dtype=torch.long, device=device,
                    )
                    dist.all_reduce(cand_rank_t, op=dist.ReduceOp.MIN)
                    winner_rank = int(cand_rank_t.item())
                    if winner_rank >= world_size:
                        winner_rank = 0

                # Broadcast winner's code from winner_rank to all ranks
                winner_codes_json = ""
                if _is_dist_ready():
                    if rank == winner_rank and best_local is not None and best_local.codes is not None:
                        winner_codes_json = json.dumps(
                            {"codebooks": best_local.codes.codebooks}, ensure_ascii=False
                        )
                    winner_codes_json = _broadcast_text(winner_codes_json, src=winner_rank, device=device)
                else:
                    if best_local is not None and best_local.codes is not None:
                        winner_codes_json = json.dumps(
                            {"codebooks": best_local.codes.codebooks}, ensure_ascii=False
                        )

                # Reconstruct winner codes from JSON
                winner_codes: Optional[Codes] = None
                if winner_codes_json:
                    try:
                        payload = json.loads(winner_codes_json)
                        winner_codes = Codes(codebooks=payload.get("codebooks", {}))
                    except Exception:
                        pass

                if winner_codes is not None:
                    solutions[node_id] = winner_codes
                elif best_local is not None and best_local.codes is not None:
                    solutions[node_id] = best_local.codes  # fallback

                # Update subtask expert cache for this node
                if best_local is not None:
                    node_subtask_experts[node_id] = list(best_local.subtask_expert_ids)

                # ── Logging ────────────────────────────────────────────────────
                global_step += 1
                pass_rate = float(sum(rewards_global) / len(rewards_global)) if rewards_global else 0.0
                epoch_pass_rewards.append(pass_rate)
                epoch_smoke_rates.append(smoke_pass_rate)
                epoch_comp_means.append(comp_mean)
                epoch_cons_strip_means.append(cons_strip_mean)

                if is_main:
                    step_log = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "sample": f"{sample.category}/{sample.name}",
                        "node_id": node_id,
                        "node_title": str(node_meta.title or ""),
                        "kind": kind,
                        "n_parents": n_parents,
                        "num_candidates": len(rewards_global),
                        "reward_mean": float(reward_mean),
                        "reward_std": float(reward_std),
                        "step_pass_rate": float(pass_rate),
                        "route_reward_mean": float(route_reward_mean),
                        "route_reward_std": float(route_reward_std),
                        "best_reward": float(best_reward_global),
                        "did_update": bool(did_update),
                        "update_skip_reason": str(update_skip_reason),
                        "loss": float(loss_value),
                        "loss_grpo": float(loss_grpo_value),
                        "loss_router": float(loss_router_value),
                        "credit_mode": str(credit_mode),
                        "router_adv_mode": str(router_adv_mode),
                        "execute_route_policy": str(execute_route_policy),
                        "smoke_pass_rate":    float(smoke_pass_rate),
                        "comp_mean":          float(comp_mean),
                        "cons_strip_mean":    float(cons_strip_mean),
                        "cons_task_mean":     float(cons_task_mean),
                        "cons_subtask_mean":  float(cons_sub_mean),
                    }
                    _append_jsonl(sample_perf_jsonl, step_log)
                    if global_step % 10 == 0 or not did_update:
                        print(
                            f"  step={global_step} kind={kind} node={node_id} "
                            f"reward_mean={reward_mean:.3f} std={reward_std:.3f} "
                            f"pass_rate={pass_rate:.3f} "
                            f"smoke={smoke_pass_rate:.2f} comp={comp_mean:.2f} cons_strip={cons_strip_mean:.2f} "
                            f"loss={loss_value:.4f} "
                            f"{'[SKIP]' if not did_update else ''}",
                            flush=True,
                        )

            # ── End of sample ──────────────────────────────────────────────────
            # Per-N-samples latest checkpoint (overwrites step_latest each time)
            if is_main and save_every_samples > 0 and sample_idx % save_every_samples == 0:
                latest_ckpt_dir = ckpt_root / "step_latest"
                trainer_state_latest = {
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "sample_idx": int(sample_idx),
                }
                save_checkpoint(
                    ckpt_dir=latest_ckpt_dir,
                    router=router,
                    title_embedder=title_embedder,
                    model=model,
                    opt_router=opt_router,
                    opt_lora=opt_lora,
                    trainer_state=trainer_state_latest,
                )

        # ── End of epoch ──────────────────────────────────────────────────────
        epoch_pass = float(sum(epoch_pass_rewards) / len(epoch_pass_rewards)) if epoch_pass_rewards else 0.0
        epoch_smoke = float(sum(epoch_smoke_rates) / len(epoch_smoke_rates)) if epoch_smoke_rates else 0.0
        epoch_comp = float(sum(epoch_comp_means) / len(epoch_comp_means)) if epoch_comp_means else 0.0
        epoch_cons = float(sum(epoch_cons_strip_means) / len(epoch_cons_strip_means)) if epoch_cons_strip_means else 0.0
        if is_main:
            print(
                f"[epoch {epoch}] pass_rate={epoch_pass:.4f} "
                f"smoke={epoch_smoke:.3f} comp={epoch_comp:.3f} cons_strip={epoch_cons:.3f} "
                f"steps={global_step - epoch_global_steps_start}",
                flush=True,
            )

        # ── Save checkpoint ────────────────────────────────────────────────────
        if is_main and save_every_epochs > 0 and epoch % save_every_epochs == 0:
            ckpt_dir = ckpt_root / f"epoch_{epoch:03d}"
            trainer_state = {
                "global_step": int(global_step),
                "epoch": int(epoch),
            }
            save_checkpoint(
                ckpt_dir=ckpt_dir,
                router=router,
                title_embedder=title_embedder,
                model=model,
                opt_router=opt_router,
                opt_lora=opt_lora,
                trainer_state=trainer_state,
            )

    if is_main:
        print("[train] done.", flush=True)
    if _is_dist_ready():
        dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# Argparse
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRDD HGRPO training – both execute and aggregate experts trained."
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--srdd-csv", type=Path, default=DEFAULT_SRDD_CSV)
    p.add_argument(
        "--taskgraph-root", type=Path, default=None,
        help="Root containing pre-generated SRDD task_graph.json files.",
    )
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max number of SRDD samples to use (0 = all).")
    p.add_argument("--shuffle-samples", action=argparse.BooleanOptionalAction, default=True)

    # ── Outputs ────────────────────────────────────────────────────────────────
    p.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--init-checkpoint", type=Path, default=None)

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--gpus", type=str, default="")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--torch-dtype", type=str, default="bfloat16")

    # ── MoLE architecture ──────────────────────────────────────────────────────
    p.add_argument("--num-subtask-experts", type=int, default=4)
    p.add_argument("--subtask-top-k", type=int, default=2)
    p.add_argument("--merge-use-parent-experts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aggregate-random-parent-experts",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aggregate-parent-max-experts", type=int, default=0)
    p.add_argument("--subtask-proto-l2", type=float, default=1e-4)
    p.add_argument("--subtask-proto-ortho", type=float, default=0.05)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-last-n-layers", type=int, default=8)
    p.add_argument("--lora-lr", type=float, default=8e-5)
    p.add_argument("--router-lr", type=float, default=3e-5)

    # ── Training schedule ─────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--group-size", type=int, default=16,
                   help="Total candidates per GRPO update (across all ranks).")
    p.add_argument(
        "--hierarchical-local-routes", type=int, default=1,
        help="Number of local routes per rank (1 GPU = 1 route with 4 candidates by default).",
    )
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every-epochs", type=int, default=1)
    p.add_argument("--save-every-samples", type=int, default=0,
                   help="Also save a rolling 'step_latest' checkpoint every N samples (0 = disabled).")

    # ── Proxy reward weights ───────────────────────────────────────────────────
    p.add_argument("--proxy-w-smoke",           type=float, default=0.5)
    p.add_argument("--proxy-w-comp",            type=float, default=0.5)
    p.add_argument("--proxy-w-cons-strip",      type=float, default=1.0)
    p.add_argument("--proxy-cons-task-weight",  type=float, default=0.7)
    p.add_argument("--proxy-cons-subtask-weight", type=float, default=0.3)
    p.add_argument("--smoke-timeout-s", type=int, default=15,
                   help="Timeout (seconds) for each smoke-test evaluation.")

    # ── Entry inclusion ────────────────────────────────────────────────────────
    p.add_argument(
        "--include-execute-entries", action=argparse.BooleanOptionalAction, default=True,
        help="Train on execute (single-parent) nodes.",
    )
    p.add_argument(
        "--include-aggregate-entries", action=argparse.BooleanOptionalAction, default=True,
        help="Train on aggregate (multi-parent) nodes. SRDD default: True.",
    )
    p.add_argument("--aggregate-min-parents", type=int, default=2,
                   help="Minimum number of parents for a node to be treated as aggregate.")

    # ── Expert diversity ───────────────────────────────────────────────────────
    p.add_argument("--enforce-local-expert-diversity",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enforce-local-output-diversity",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enforce-cross-rank-route-diversity",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="If on, sync expert signatures across ranks so each GPU's route "
                        "picks a different (role_expert, subtask_experts) combo. "
                        "Tie-break: lower rank wins; higher ranks resample. "
                        "Default off: empirically hurts early training (off-policy bias "
                        "from forced fallback). Pass --enforce-cross-rank-route-diversity "
                        "explicitly to enable.")
    p.add_argument("--diversity-max-resample", type=int, default=16)
    p.add_argument("--router-diversity-reg", type=float, default=0.02)

    # ── HGRPO credit mode (exp1/exp2/exp3 switch) ──────────────────────────────
    p.add_argument(
        "--credit-mode", type=str, default="hierarchical",
        choices=["hierarchical", "flat"],
        help=(
            "hierarchical = Exp1/Exp2 (within-route LoRA, across-route router); "
            "flat = Exp3 (standard GRPO baseline)."
        ),
    )
    p.add_argument(
        "--execute-route-policy", type=str, default="sample",
        choices=["sample", "hybrid", "greedy"],
        help=(
            "sample = Exp2/Exp3 (all sampled); "
            "hybrid = Exp1 (rank-0 anchor greedy, others sampled)."
        ),
    )
    p.add_argument(
        "--router-adv-mode", type=str, default="mean_center",
        choices=["mean_center", "anchor_vs_others"],
        help=(
            "mean_center = Exp2/Exp3; "
            "anchor_vs_others = Exp1 (anchor route advantage vs sampled routes)."
        ),
    )
    p.add_argument("--router-contrast-weight", type=float, default=0.0,
                   help="Weight for sampled-route router contrast loss (Exp1: 0.4; Exp2/3: 0.0).")
    p.add_argument("--alpha-router", type=float, default=0.15,
                   help="Router log-prob weight in router loss term.")
    p.add_argument("--sampled-route-weight", type=float, default=1.0,
                   help="Weight applied to sampled-route LoRA advantage (Exp1: 0.5; Exp2: 1.0).")
    p.add_argument("--sampled-positive-adv-only", action=argparse.BooleanOptionalAction, default=False,
                   help="For sampled routes, only use positive advantages (Exp1: True; Exp2/3: False).")
    p.add_argument("--sampled-drop-below-anchor-delta", type=float, default=-1.0)

    # ── GRPO update ───────────────────────────────────────────────────────────
    p.add_argument("--grpo-adv-normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--advantage-clip", type=float, default=5.0)
    p.add_argument("--grpo-skip-update-if-allzero", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grpo-skip-update-if-low-std", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grpo-min-reward-std", type=float, default=0.0005)

    # ── Distributed ───────────────────────────────────────────────────────────
    p.add_argument("--dist-timeout-s", type=int, default=21600)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    train_grpo(args)


if __name__ == "__main__":
    main()
