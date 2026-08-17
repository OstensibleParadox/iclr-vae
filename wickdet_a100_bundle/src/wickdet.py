from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

Tensor = torch.Tensor
OperatorFn = Callable[[Tensor], Tensor]


def _event_dims(x: Tensor) -> Tuple[int, ...]:
    if x.ndim < 2:
        raise ValueError("features must have shape (batch, ...event dimensions...)")
    return tuple(range(1, x.ndim))


def rademacher_probes(
    features: Tensor,
    num_probes: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Return Rademacher probes with the same event shape as ``features``."""
    if num_probes <= 0:
        raise ValueError("num_probes must be positive")

    probes = torch.empty(
        (num_probes, *features.shape[1:]),
        device=features.device,
        dtype=features.dtype,
    )
    probes.bernoulli_(0.5, generator=generator)
    return probes.mul_(2).sub_(1)


def hutchinson_trace(
    operator_fn: OperatorFn,
    probes: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Estimate Tr(P) from matrix-free products Pz."""
    dims = _event_dims(probes)
    pz = operator_fn(probes)
    if pz.shape != probes.shape:
        raise ValueError("operator_fn(probes) must have the same shape as probes")
    trace_est = (probes * pz).sum(dim=dims).mean()
    return trace_est, pz


def hutchinson_hs2(
    pz: Tensor,
) -> Tensor:
    """Estimate ||P||_{S2}^2; for self-adjoint P this is Tr(P^2)."""
    dims = _event_dims(pz)
    return (pz * pz).sum(dim=dims).mean()


def neg_logdet2_series(
    operator_fn: OperatorFn,
    probes: Tensor,
    *,
    K: int = 5,
    first_power: Optional[Tensor] = None,
) -> Tensor:
    r"""Estimate -log det_2(I-P) = sum_{k>=2} Tr(P^k)/k.

    The truncated power series is intended for self-adjoint P with spectral
    radius strictly below one.  ``K`` is the largest retained power.

    A warning is emitted when successive trace estimates suggest the spectral
    radius is near or above 1, where the truncated polynomial no longer
    approximates the barrier functional.
    """
    if K < 2:
        raise ValueError("K must be at least 2 for a det_2 estimate")

    dims = _event_dims(probes)
    p_power_z = first_power if first_power is not None else operator_fn(probes)
    neg_logdet2 = probes.new_zeros(())

    trace_prev: Optional[Tensor] = None
    trace_k: Optional[Tensor] = None

    for k in range(2, K + 1):
        p_power_z = operator_fn(p_power_z)
        if p_power_z.shape != probes.shape:
            raise ValueError("operator_fn must preserve the probe shape")
        trace_k = (probes * p_power_z).sum(dim=dims).mean()
        neg_logdet2 = neg_logdet2 + trace_k / k

        if trace_prev is not None:
            ratio = (trace_k.abs() / (trace_prev.abs() + 1e-30)).clamp(max=100.0)
            if ratio > 0.92:
                warnings.warn(
                    f"neg_logdet2_series: |Tr(P^{k})| / |Tr(P^{k-1})| ≈ {float(ratio):.3f}. "
                    f"Spectral radius may be near or above 1; K={K} truncation is unreliable. "
                    f"The series sum_{{m>=2}} Tr(P^m)/m only converges for ||P|| < 1. "
                    f"Increase K or clamp eigenvalues below 1.",
                    stacklevel=2,
                )
                break  # warn once, don't spam for every subsequent k
        trace_prev = trace_k

    return neg_logdet2


def schatten_cf_penalty(
    operator_fn: OperatorFn,
    features: Tensor,
    K: int = 5,
    num_probes: int = 10,
    *,
    mode: str = "penalty",
    probes: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_diagnostics: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]:
    r"""Matrix-free Wick--Carleman--Fredholm correction.

    ``operator_fn`` represents the linear map P by vector products.  Autograd
    flows through every call to ``operator_fn``; the random probes themselves do
    not require gradients.

    Modes:
      - ``"penalty"`` returns
        1/2(<X,PX>-Tr P) + 1/2[-log det_2(I-P)].
      - ``"log_density"`` returns the Gaussian log-Radon--Nikodym convention
        1/2:<X,PX>: + 1/2 log det_2(I-P).
      - ``"negative_log_density"`` returns the negative of ``"log_density"``.
    """
    dims = _event_dims(features)
    if probes is None:
        probes = rademacher_probes(features, num_probes, generator=generator)
    elif probes.shape[1:] != features.shape[1:]:
        raise ValueError("probes must have shape (num_probes, *features.shape[1:])")

    trace_est, pz = hutchinson_trace(operator_fn, probes)
    hs2_est = hutchinson_hs2(pz)

    # ------------------------------------------------------------------
    # Spectral-radius guard:  ρ(P) ≤ √Tr(P²), so  hs2_est / |Tr(P)|
    # (when available) gives a rough lower-bound on the largest
    # eigenvalue.  When this exceeds ~0.8 the K-term truncation misses a
    # significant fraction of the series tail and the polynomial no
    # longer encodes the barrier at ||P|| = 1.
    # ------------------------------------------------------------------
    _trace_abs = trace_est.abs() + 1e-30
    rho_est = (hs2_est / _trace_abs).clamp(max=0.999)
    if float(rho_est) > 0.75:
        # Conservative tail bound for the largest-eigenvalue component
        tail_bound = rho_est ** (K + 1) / ((K + 1) * (1.0 - rho_est) + 1e-30)
        warnings.warn(
            f"schatten_cf_penalty: rough spectral-radius estimate ρ≈{float(rho_est):.3f} "
            f"(from Tr(P²)/|Tr(P)|).  K={K} truncation may be unreliable — "
            f"remainder bound ≈ {float(tail_bound):.3f}.  "
            f"The penalty functional ½⟨:X,PX:⟩ + ½[−log det₂(I−P)] is only valid "
            f"when the series converges (||P|| < 1).  "
            f"Clamp eigenvalues or increase K.",
            stacklevel=2,
        )

    px = operator_fn(features)
    if px.shape != features.shape:
        raise ValueError("operator_fn(features) must have the same shape as features")

    quadratic = (features * px).sum(dim=dims).mean()
    wick_quadratic = 0.5 * (quadratic - trace_est)

    neg_logdet2 = neg_logdet2_series(
        operator_fn,
        probes,
        K=K,
        first_power=pz,
    )

    if mode == "penalty":
        value = wick_quadratic + 0.5 * neg_logdet2
    elif mode == "log_density":
        value = wick_quadratic - 0.5 * neg_logdet2
    elif mode == "negative_log_density":
        value = -wick_quadratic + 0.5 * neg_logdet2
    else:
        raise ValueError(
            "mode must be one of 'penalty', 'log_density', or "
            "'negative_log_density'"
        )

    if not return_diagnostics:
        return value, hs2_est

    diagnostics = {
        "trace": trace_est,
        "hs2": hs2_est,
        "wick_quadratic": wick_quadratic,
        "neg_logdet2": neg_logdet2,
    }
    return value, diagnostics


class WickCarlemanPenalty(nn.Module):
    """Small nn.Module wrapper around ``schatten_cf_penalty``."""

    def __init__(
        self,
        *,
        K: int = 5,
        num_probes: int = 10,
        mode: str = "penalty",
    ) -> None:
        super().__init__()
        self.K = K
        self.num_probes = num_probes
        self.mode = mode

    def forward(
        self,
        operator_fn: OperatorFn,
        features: Tensor,
        *,
        probes: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]:
        return schatten_cf_penalty(
            operator_fn,
            features,
            K=self.K,
            num_probes=self.num_probes,
            mode=self.mode,
            probes=probes,
            return_diagnostics=return_diagnostics,
        )


__all__ = [
    "WickCarlemanPenalty",
    "hutchinson_hs2",
    "hutchinson_trace",
    "neg_logdet2_series",
    "rademacher_probes",
    "schatten_cf_penalty",
]
