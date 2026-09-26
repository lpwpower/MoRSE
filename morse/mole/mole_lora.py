from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
	num_experts: int = 8
	top_k: int = 3
	rank: int = 8
	alpha: float = 16.0
	target_modules: Sequence[str] = ("q_proj", "v_proj", "o_proj")
	last_n_layers: int = 8


class MoLELinear(nn.Module):
	"""A Linear layer with K LoRA experts, activated by an external expert set."""

	def __init__(self, base: nn.Linear, *, cfg: LoRAConfig):
		super().__init__()
		if not isinstance(base, nn.Linear):
			raise TypeError("MoLELinear expects an nn.Linear")
		self.base = base
		self.cfg = cfg
		self.in_features = base.in_features
		self.out_features = base.out_features
		self.rank = int(cfg.rank)
		self.scaling = float(cfg.alpha) / float(max(1, self.rank))

		# Freeze base weights.
		for p in self.base.parameters():
			p.requires_grad = False

		# Expert parameters: A [K, r, in], B [K, out, r]
		k = int(cfg.num_experts)
		r = self.rank
		# Match dtype/device to the base layer to avoid matmul dtype mismatches (e.g., bf16 vs fp32).
		base_weight = getattr(self.base, "weight", None)
		device = base_weight.device if base_weight is not None else None
		dtype = base_weight.dtype if base_weight is not None else None
		self.lora_A = nn.Parameter(torch.zeros(k, r, self.in_features, device=device, dtype=dtype))
		self.lora_B = nn.Parameter(torch.zeros(k, self.out_features, r, device=device, dtype=dtype))
		# Init: A ~ N(0, 0.02), B = 0 (common LoRA init)
		nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
		nn.init.zeros_(self.lora_B)

		# Active experts set externally per generation step.
		self._active_experts: Optional[torch.Tensor] = None  # [top_k]
		self._active_weights: Optional[torch.Tensor] = None  # [top_k]
		# Cached effective delta for inference (invalidated on expert change).
		# Shape [out, in]: precomputed sum_k(w_k * scaling * B_k @ A_k).
		self._cached_delta: Optional[torch.Tensor] = None

	def set_active(self, expert_ids: Optional[torch.Tensor], weights: Optional[torch.Tensor] = None) -> None:
		self._active_experts = expert_ids
		self._active_weights = weights
		self._cached_delta = None  # invalidate on expert change

	def _make_delta(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
		"""Compute effective LoRA delta: sum_k(w_k * scaling * B_k @ A_k). Shape: [out, in]."""
		ids = self._active_experts
		weights = self._active_weights
		if weights is None:
			w = torch.ones(ids.numel(), dtype=dtype, device=device) / float(ids.numel())
		else:
			w = weights.to(dtype=dtype, device=device)
		scale = w * self.scaling  # [top_k]
		A = self.lora_A[ids].to(dtype=dtype)  # [top_k, r, in]
		B = self.lora_B[ids].to(dtype=dtype)  # [top_k, out, r]
		# Fused: scale_k * B_k @ A_k, summed over k → [out, in]
		return torch.einsum("k,kor,kri->oi", scale, B, A)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.base(x)
		ids = self._active_experts
		if ids is None or ids.numel() == 0:
			return out
		if torch.is_grad_enabled():
			# Gradient path (logprob): always recompute to maintain autograd graph.
			delta = self._make_delta(x.dtype, x.device)
		else:
			# Inference path (generation): cache delta across token steps.
			if self._cached_delta is None:
				self._cached_delta = self._make_delta(x.dtype, x.device)
			delta = self._cached_delta
		# x: [..., in],  delta: [out, in]  →  x @ delta.T: [..., out]
		return out + x.matmul(delta.t())


def _find_transformer_layers(model: nn.Module) -> List[nn.Module]:
	# Try common attribute names.
	for attr in ("model", "transformer"):
		if hasattr(model, attr):
			m = getattr(model, attr)
			if hasattr(m, "layers") and isinstance(m.layers, (list, nn.ModuleList)):
				return list(m.layers)
	if hasattr(model, "layers") and isinstance(model.layers, (list, nn.ModuleList)):
		return list(model.layers)
	raise ValueError("Could not locate transformer layers on model; add a custom locator for this architecture.")


def inject_mole_lora(model: nn.Module, *, cfg: LoRAConfig) -> List[MoLELinear]:
	"""Replace target Linear modules with MoLELinear in the last N transformer blocks."""
	layers = _find_transformer_layers(model)
	if int(cfg.last_n_layers) <= 0:
		raise ValueError("--last-n-layers must be > 0")
	target_layers = layers[-int(cfg.last_n_layers) :]
	target_suffixes = set(cfg.target_modules)
	replaced: List[MoLELinear] = []

	for layer in target_layers:
		for name, module in list(layer.named_modules()):
			# Replace leaf Linear modules whose local name ends with a target suffix.
			if not isinstance(module, nn.Linear):
				continue
			last = name.split(".")[-1]
			if last not in target_suffixes:
				continue
			# Find parent module to swap attribute.
			parent = layer
			parts = name.split(".")
			for part in parts[:-1]:
				parent = getattr(parent, part)
			attr = parts[-1]
			current = getattr(parent, attr)
			if not isinstance(current, nn.Linear):
				continue
			wrapped = MoLELinear(current, cfg=cfg)
			setattr(parent, attr, wrapped)
			replaced.append(wrapped)

	return replaced


def set_active_experts(model: nn.Module, expert_ids: Optional[torch.Tensor], weights: Optional[torch.Tensor] = None) -> None:
	for mod in model.modules():
		if isinstance(mod, MoLELinear):
			mod.set_active(expert_ids, weights)


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
	for mod in model.modules():
		if isinstance(mod, MoLELinear):
			yield mod.lora_A
			yield mod.lora_B
