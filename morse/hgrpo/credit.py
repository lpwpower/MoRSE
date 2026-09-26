"""Two-layer credit assignment for HGRPO (shared by SRDD and SciCode).

Implements the within-route / cross-route advantages of the HGRPO objective in the
paper. For a node, we sample B routes (each fixing an expert combination) and M
candidates per route, with step rewards ``u``:

    A_within^{(b,m)} = (u^{(b,m)} - mu^{(b)}) / (sigma^{(b)} + eps)   -> updates LoRA experts
    A_cross^{(b)}    = (mu^{(b)} - mu_bar)    / (sigma_C   + eps)     -> updates the router

where ``mu^{(b)}, sigma^{(b)}`` are the within-route (per expert-combination) reward
mean / std over the M candidates, ``mu_bar`` is the mean of the route means over all
B routes (the across-route baseline), and ``sigma_C`` is their std. In a distributed
"1 route per GPU" layout, ``mu_bar`` / ``sigma_C`` must be computed over route means
all-gathered across ranks so that the baseline spans all B routes.

This is the standard two-layer credit of the paper; there is no anchor / reference-
route variant here.
"""
from __future__ import annotations


def clip_advantage(adv: float, clip: float) -> float:
    """Clamp an advantage to [-clip, clip]. ``clip <= 0`` disables clipping."""
    adv = float(adv)
    if clip and clip > 0.0:
        if adv > clip:
            return float(clip)
        if adv < -clip:
            return -float(clip)
    return adv


def within_route_advantage(
    reward: float,
    route_mean: float,
    route_std: float,
    *,
    normalize: bool = True,
    eps: float = 1e-6,
    clip: float = 0.0,
) -> float:
    """A_within: a candidate's advantage relative to its own route mean.

    Updates the LoRA experts under the fixed expert combination of that route. The
    within-route baseline ``route_mean`` removes the cross-route variance component
    from the expert gradient (Proposition in the paper).
    """
    adv = float(reward) - float(route_mean)
    if normalize:
        adv = adv / (float(route_std) + float(eps)) if route_std > 0.0 else 0.0
    return clip_advantage(adv, clip)


def cross_route_advantage(
    route_mean: float,
    route_reward_mean: float,
    route_reward_std: float,
    *,
    normalize: bool = True,
    eps: float = 1e-6,
    clip: float = 0.0,
) -> float:
    """A_cross: a route's advantage relative to the across-route baseline ``mu_bar``.

    Updates the router via the route log-likelihood. ``route_reward_mean`` is
    ``mu_bar`` (mean of all route means) and ``route_reward_std`` is ``sigma_C``;
    both must be aggregated over all B routes (all-gathered across ranks under a
    1-route-per-GPU layout).
    """
    adv = float(route_mean) - float(route_reward_mean)
    if normalize:
        adv = adv / (float(route_reward_std) + float(eps)) if route_reward_std > 0.0 else 0.0
    return clip_advantage(adv, clip)
