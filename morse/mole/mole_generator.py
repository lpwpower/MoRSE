from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .mole_lora import set_active_experts


@dataclass
class GenerationConfig:
	model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
	max_new_tokens: int = 2048
	temperature: float = 0.2
	top_p: float = 0.95
	torch_dtype: str = "bfloat16"
	device: int = 0


class MoLEGenerator:
	"""Wraps a frozen HF causal LM + MoLE LoRA layers to support (generate, logp) pairs."""

	def __init__(self, *, model, tokenizer, device: torch.device, gen_cfg: GenerationConfig):
		self.model = model
		self.tokenizer = tokenizer
		self.device = device
		self.gen_cfg = gen_cfg

	def _encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
		enc = self.tokenizer(text, return_tensors="pt")
		input_ids = enc["input_ids"].to(self.device)
		attention_mask = enc.get("attention_mask")
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids, device=self.device)
		else:
			attention_mask = attention_mask.to(self.device)
		return input_ids, attention_mask

	def _encode_batch(self, texts: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
		enc = self.tokenizer(texts, return_tensors="pt", padding=True)
		input_ids = enc["input_ids"].to(self.device)
		attention_mask = enc.get("attention_mask")
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids, device=self.device)
		else:
			attention_mask = attention_mask.to(self.device)
		return input_ids, attention_mask

	def generate_with_experts(
		self,
		*,
		prompt: str,
		expert_ids: torch.Tensor,
		expert_weights: Optional[torch.Tensor] = None,
	) -> Tuple[str, torch.Tensor, torch.Tensor]:
		"""Return (text, prompt_ids, gen_ids). Generation is done under no_grad."""
		set_active_experts(self.model, expert_ids, expert_weights)
		input_ids, attention_mask = self._encode(prompt)
		do_sample = float(self.gen_cfg.temperature) > 0.0
		# Collect all valid EOS token ids (base eos + chat end-of-turn token if present)
		eos_id = self.tokenizer.eos_token_id
		_extra_eos = getattr(self.tokenizer, "additional_special_tokens_ids", [])
		_turn_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
		if isinstance(_turn_end, int) and _turn_end != self.tokenizer.unk_token_id:
			_extra_eos = list(_extra_eos) + [_turn_end]
		eos_ids = list({eos_id} | set(_extra_eos)) if _extra_eos else [eos_id]
		gen_kwargs = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"do_sample": do_sample,
			"use_cache": True,
			"max_new_tokens": int(self.gen_cfg.max_new_tokens),
			"pad_token_id": eos_id,
			"eos_token_id": eos_ids,
			"return_dict_in_generate": True,
		}
		# Avoid warnings when sampling is disabled: set sampling-only flags to defaults.
		if do_sample:
			gen_kwargs["temperature"] = float(self.gen_cfg.temperature)
			gen_kwargs["top_p"] = float(self.gen_cfg.top_p)
		else:
			gen_kwargs["temperature"] = 1.0
			gen_kwargs["top_p"] = 1.0
			gen_kwargs["top_k"] = 50
		with torch.no_grad():
			out = self.model.generate(**gen_kwargs)
		seq = out.sequences[0]
		prompt_len = input_ids.shape[1]
		gen_ids = seq[prompt_len:]
		text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
		return text, input_ids[0], gen_ids

	def generate_n_with_experts(
		self,
		*,
		prompt: str,
		expert_ids: torch.Tensor,
		num_samples: int,
		expert_weights: Optional[torch.Tensor] = None,
	) -> list[Tuple[str, torch.Tensor, torch.Tensor]]:
		"""Generate N samples for the same prompt under the same active experts.

		Returns a list of (text, prompt_ids, gen_ids) tuples, one per sample.
		"""
		n = int(num_samples)
		if n <= 0:
			return []
		set_active_experts(self.model, expert_ids, expert_weights)
		input_ids, attention_mask = self._encode_batch([prompt] * n)
		do_sample = float(self.gen_cfg.temperature) > 0.0
		eos_id = self.tokenizer.eos_token_id
		_extra_eos = getattr(self.tokenizer, "additional_special_tokens_ids", [])
		_turn_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
		if isinstance(_turn_end, int) and _turn_end != self.tokenizer.unk_token_id:
			_extra_eos = list(_extra_eos) + [_turn_end]
		eos_ids = list({eos_id} | set(_extra_eos)) if _extra_eos else [eos_id]
		gen_kwargs = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"do_sample": do_sample,
			"use_cache": True,
			"max_new_tokens": int(self.gen_cfg.max_new_tokens),
			"pad_token_id": eos_id,
			"eos_token_id": eos_ids,
			"return_dict_in_generate": True,
		}
		if do_sample:
			gen_kwargs["temperature"] = float(self.gen_cfg.temperature)
			gen_kwargs["top_p"] = float(self.gen_cfg.top_p)
		else:
			gen_kwargs["temperature"] = 1.0
			gen_kwargs["top_p"] = 1.0
			gen_kwargs["top_k"] = 50
		with torch.no_grad():
			out = self.model.generate(**gen_kwargs)
		sequences = out.sequences  # [B, T]
		prompt_len = input_ids.shape[1]

		pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
		results: list[Tuple[str, torch.Tensor, torch.Tensor]] = []
		for i in range(sequences.shape[0]):
			seq = sequences[i]
			gen = seq[prompt_len:]
			# Trim padding; include EOS token if present.
			if eos_id is not None:
				eos_pos = (gen == eos_id).nonzero(as_tuple=False)
				if eos_pos.numel() > 0:
					gen = gen[: int(eos_pos[0].item()) + 1]
			# If not terminated, trim trailing pad tokens (generate pads to max length in batch).
			if pad_id is not None and gen.numel() > 0:
				non_pad = (gen != pad_id).nonzero(as_tuple=False)
				if non_pad.numel() > 0:
					last = int(non_pad[-1].item()) + 1
					gen = gen[:last]
				else:
					gen = gen[:0]
			text = self.tokenizer.decode(gen, skip_special_tokens=True)
			results.append((text, input_ids[i], gen))
		return results

	def logprob_of_generation(
		self,
		*,
		prompt_ids: torch.Tensor,
		gen_ids: torch.Tensor,
		expert_ids: torch.Tensor,
		expert_weights: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""Compute sum log p(gen_ids | prompt_ids) with gradients (teacher forcing)."""
		set_active_experts(self.model, expert_ids, expert_weights)
		full = torch.cat([prompt_ids, gen_ids], dim=0).unsqueeze(0).to(self.device)
		# Predict token t from logits at position t-1.
		logits = self.model(full, use_cache=False).logits  # [1, T, V]
		logits = logits[:, :-1, :]  # [1, T-1, V]
		target = full[:, 1:]  # [1, T-1]
		# We only score the generated segment (exclude prompt continuation).
		prompt_len = int(prompt_ids.numel())
		gen_start = max(0, prompt_len - 1)
		logits = logits[:, gen_start:, :]  # [1, G, V]
		target = target[:, gen_start:]  # [1, G]
		# log p(y) = logit[y] - logsumexp(logits)
		tlogits = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, G]
		lse = torch.logsumexp(logits, dim=-1)  # [1, G]
		return (tlogits - lse).sum()
