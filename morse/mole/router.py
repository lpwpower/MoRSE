from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterConfig:
	num_experts: int = 8
	top_k: int = 3
	role_emb_dim: int = 32
	title_emb_dim: int = 128
	hidden_dim: int = 256


@dataclass
class SubtaskRouterConfig:
	num_experts: int = 4
	top_k: int = 2
	title_emb_dim: int = 128
	prototype_init_std: float = 0.02


class RoleTitleRouter(nn.Module):
	"""A small MLP router: (role embedding + node-title embedding) -> expert logits."""

	def __init__(self, cfg: RouterConfig):
		super().__init__()
		self.cfg = cfg
		self.role_emb = nn.Embedding(2, cfg.role_emb_dim)
		in_dim = cfg.role_emb_dim + cfg.title_emb_dim
		self.mlp = nn.Sequential(
			nn.Linear(in_dim, cfg.hidden_dim),
			nn.GELU(),
			nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
			nn.GELU(),
			nn.Linear(cfg.hidden_dim, cfg.num_experts),
		)

	def forward(self, *, role_id: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
		# role_id: [B]
		# title_emb: [B, title_emb_dim]
		role_vec = self.role_emb(role_id)
		x = torch.cat([role_vec, title_emb], dim=-1)
		return self.mlp(x)

	def sample_topk(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Sample a set of experts (Top-K without replacement) and return (ids, logp_sum).

		logp_sum is an approximation: sum log p(id_i) under the softmax over all experts.
		"""
		if logits.dim() != 2 or logits.size(0) != 1:
			raise ValueError("sample_topk currently supports batch size 1 for simplicity.")
		probs = F.softmax(logits[0], dim=-1)
		k = min(int(self.cfg.top_k), int(self.cfg.num_experts))
		ids = torch.multinomial(probs, num_samples=k, replacement=False)
		logp = torch.log(torch.clamp(probs[ids], min=1e-12)).sum()
		return ids, logp

	def greedy_topk(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if logits.dim() != 2 or logits.size(0) != 1:
			raise ValueError("greedy_topk currently supports batch size 1 for simplicity.")
		probs = F.softmax(logits[0], dim=-1)
		k = min(int(self.cfg.top_k), int(self.cfg.num_experts))
		ids = torch.topk(probs, k=k, dim=-1).indices
		logp = torch.log(torch.clamp(probs[ids], min=1e-12)).sum()
		return ids, logp


class SubtaskRouter(nn.Module):
	"""Prototype router: (subtask embedding) -> expert logits via cosine similarity."""

	def __init__(self, cfg: SubtaskRouterConfig):
		super().__init__()
		self.cfg = cfg
		self.prototypes = nn.Parameter(torch.empty(cfg.num_experts, cfg.title_emb_dim))
		nn.init.normal_(self.prototypes, mean=0.0, std=float(cfg.prototype_init_std))

	def forward(self, *, title_emb: torch.Tensor) -> torch.Tensor:
		if title_emb.dim() != 2:
			raise ValueError("SubtaskRouter expects title_emb with shape [B, D].")
		emb = normalize_title_embedding(title_emb)
		proto = F.normalize(self.prototypes, dim=-1)
		return emb @ proto.t()

	def sample_topk(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if logits.dim() != 2 or logits.size(0) != 1:
			raise ValueError("sample_topk currently supports batch size 1 for simplicity.")
		probs = F.softmax(logits[0], dim=-1)
		k = min(int(self.cfg.top_k), int(self.cfg.num_experts))
		ids = torch.multinomial(probs, num_samples=k, replacement=False)
		logp = torch.log(torch.clamp(probs[ids], min=1e-12)).sum()
		return ids, logp

	def greedy_topk(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if logits.dim() != 2 or logits.size(0) != 1:
			raise ValueError("greedy_topk currently supports batch size 1 for simplicity.")
		probs = F.softmax(logits[0], dim=-1)
		k = min(int(self.cfg.top_k), int(self.cfg.num_experts))
		ids = torch.topk(probs, k=k, dim=-1).indices
		logp = torch.log(torch.clamp(probs[ids], min=1e-12)).sum()
		return ids, logp


def normalize_title_embedding(title_emb: torch.Tensor) -> torch.Tensor:
	# Keep magnitude stable across variable-length titles.
	norm = torch.norm(title_emb, dim=-1, keepdim=True)
	return title_emb / torch.clamp(norm, min=1e-6)
