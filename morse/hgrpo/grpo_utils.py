"""Shared GRPO utilities (distributed + numeric helpers) for HGRPO training.

Backbone- and benchmark-agnostic helpers used by both the SRDD and SciCode
training drivers. Extracted verbatim from the original GRPO base so the two
drivers depend only on ``morse`` and not on each other. The underscore-prefixed
names match the drivers' historical ``grpo_base._all_gather_float_list`` calls.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist


def _sanitize(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_") or "item"


def _clip_advantage(adv: float, clip: float) -> float:
    c = float(clip)
    if c <= 0.0:
        return float(adv)
    if adv > c:
        return float(c)
    if adv < -c:
        return float(-c)
    return float(adv)


def _safe_std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / float(len(vals))
    var = sum((x - mean) ** 2 for x in vals) / float(len(vals))
    return float(var ** 0.5)


def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_gather_float_list(local_vals: List[float], *, device: torch.device, world_size: int) -> List[float]:
    if world_size <= 1:
        return [float(v) for v in local_vals]
    local = torch.tensor(local_vals, dtype=torch.float32, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    merged: List[float] = []
    for tensor in gathered:
        merged.extend([float(x) for x in tensor.detach().cpu().tolist()])
    return merged


def _global_best_metrics(
    *,
    local_best_reward: float,
    local_best_step: float,
    local_best_shape: float,
    local_best_gt: float,
    device: torch.device,
    world_size: int,
) -> Tuple[float, float, float, float]:
    vec = torch.tensor(
        [float(local_best_reward), float(local_best_step), float(local_best_shape), float(local_best_gt)],
        dtype=torch.float32,
        device=device,
    )
    if world_size <= 1:
        out = vec.detach().cpu().tolist()
        return float(out[0]), float(out[1]), float(out[2]), float(out[3])
    gathered = [torch.empty_like(vec) for _ in range(world_size)]
    dist.all_gather(gathered, vec)
    best = max((g.detach().cpu().tolist() for g in gathered), key=lambda x: float(x[0]))
    return float(best[0]), float(best[1]), float(best[2]), float(best[3])


def _average_gradients(params: List[torch.nn.Parameter], world_size: int) -> None:
    if world_size <= 1:
        return
    scale = 1.0 / float(world_size)
    for p in params:
        if p.grad is None:
            # Ensure every rank has a real grad tensor so all_reduce can propagate
            # non-zero gradients from other ranks to this rank.
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.mul_(scale)
