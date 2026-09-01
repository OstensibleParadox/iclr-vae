from __future__ import annotations

import warnings
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

Tensor = torch.Tensor
OperatorFn = Callable[[Tensor], Tensor]
KLDirection = Literal["q_to_p", "p_to_q"]


# Canonical relative-Gaussian contract used throughout this repository:
#
#   amplitude B: eta = B xi, xi ~ N(0, I), hence Cov(eta) = K = B B*;
#   covariance K: q_K = N(0, I + K), p = N(0, I);
#   precision H:  H = I - (I + K)^(-1) = K(I + K)^(-1).
#
# Consequently T2 always means ||K||_HS^2, not ||B||_HS^2 and not
# ||H||_HS^2.  The Wick representation uses H, while the forward KL and the
# q-centered log-density variance use K.


def _event_dims(x: Tensor) -> Tuple[int, ...]:
    if x.ndim < 2:
        raise ValueError("features must have shape (batch, ...event dimensions...)")
    return tuple(range(1, x.ndim))


def _check_matrix(matrix: Tensor, name: str, *, square: bool = True) -> None:
    if matrix.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if square and matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"{name} must be square in its final two dimensions")
    if not (matrix.dtype.is_floating_point or matrix.dtype.is_complex):
        raise TypeError(f"{name} must have a floating-point or complex dtype")


def _adjoint(matrix: Tensor) -> Tensor:
    return matrix.transpose(-2, -1).conj()


def _check_self_adjoint(matrix: Tensor, name: str) -> None:
    _check_matrix(matrix, name)
    if not torch.allclose(matrix, _adjoint(matrix), rtol=1e-6, atol=1e-8):
        raise ValueError(f"{name} must be self-adjoint")


def _identity_like(matrix: Tensor) -> Tensor:
    n = matrix.shape[-1]
    return torch.eye(n, dtype=matrix.dtype, device=matrix.device).expand(
        *matrix.shape[:-2], n, n
    )


def covariance_from_amplitude(amplitude_b: Tensor) -> Tensor:
    r"""Return the relative covariance perturbation ``K = B B*``.

    ``amplitude_b`` may be rectangular with shape ``(..., dimension, rank)``.
    Squaring the amplitude here is part of the contract: a spectral amplitude
    ``beta_j`` produces covariance eigenvalue ``beta_j**2`` and contributes
    ``beta_j**4`` to :math:`T_2`.
    """
    _check_matrix(amplitude_b, "amplitude_b", square=False)
    return amplitude_b @ _adjoint(amplitude_b)


def sample_covariance_perturbation(
    amplitude_b: Tensor,
    num_samples: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    r"""Draw ``eta = B xi`` with covariance ``K = B B*``.

    This deliberately samples the injected perturbation, not ``q_K`` itself.
    ``amplitude_b`` is currently required to be an unbatched real matrix so
    the returned shape is unambiguous: ``(num_samples, dimension)``.
    """
    _check_matrix(amplitude_b, "amplitude_b", square=False)
    if amplitude_b.ndim != 2:
        raise ValueError("sample_covariance_perturbation expects an unbatched B")
    if amplitude_b.dtype.is_complex:
        raise TypeError("sample_covariance_perturbation currently expects real B")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    xi = torch.randn(
        num_samples,
        amplitude_b.shape[-1],
        dtype=amplitude_b.dtype,
        device=amplitude_b.device,
        generator=generator,
    )
    return xi @ amplitude_b.transpose(-2, -1)


def sample_q_from_amplitude(
    amplitude_b: Tensor,
    num_samples: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    r"""Draw from ``q_K = N(0, I + K)`` for ``K = B B*``."""
    perturbation = sample_covariance_perturbation(
        amplitude_b, num_samples, generator=generator
    )
    base = torch.randn(
        perturbation.shape,
        dtype=perturbation.dtype,
        device=perturbation.device,
        generator=generator,
    )
    return base + perturbation


def covariance_t2(covariance_k: Tensor) -> Tensor:
    r"""Return canonical ``T2 = ||K||_HS^2 = Tr(K* K)``."""
    _check_self_adjoint(covariance_k, "covariance_k")
    return (covariance_k.conj() * covariance_k).real.sum(dim=(-2, -1))


def covariance_logdet2(covariance_k: Tensor) -> Tensor:
    r"""Return ``log det_2(I + K) = sum_j(log(1+k_j)-k_j)``.

    The eigenvalue form is stable for small perturbations and accepts signed
    self-adjoint ``K`` whenever ``I + K`` is positive definite.
    """
    _check_self_adjoint(covariance_k, "covariance_k")
    eigenvalues = torch.linalg.eigvalsh(covariance_k)
    if bool(torch.any(eigenvalues <= -1)):
        raise ValueError("I + covariance_k must be positive definite")
    return (torch.log1p(eigenvalues) - eigenvalues).sum(dim=-1)


def precision_logdet2(precision_h: Tensor) -> Tensor:
    r"""Return ``log det_2(I - H) = sum_j(log(1-h_j)+h_j)``."""
    _check_self_adjoint(precision_h, "precision_h")
    eigenvalues = torch.linalg.eigvalsh(precision_h)
    if bool(torch.any(eigenvalues >= 1)):
        raise ValueError("I - precision_h must be positive definite")
    return (torch.log1p(-eigenvalues) + eigenvalues).sum(dim=-1)


def precision_from_covariance(covariance_k: Tensor) -> Tensor:
    r"""Map covariance coordinates to precision coordinates.

    ``H = I - (I + K)^(-1) = K(I + K)^(-1)``.
    """
    _check_self_adjoint(covariance_k, "covariance_k")
    identity = _identity_like(covariance_k)
    precision_h = torch.linalg.solve(identity + covariance_k, covariance_k)
    return 0.5 * (precision_h + _adjoint(precision_h))


def covariance_from_precision(precision_h: Tensor) -> Tensor:
    r"""Map precision coordinates to covariance coordinates.

    ``K = H(I - H)^(-1)``; the domain requires ``I - H`` positive definite.
    """
    _check_self_adjoint(precision_h, "precision_h")
    if bool(torch.any(torch.linalg.eigvalsh(precision_h) >= 1)):
        raise ValueError("I - precision_h must be positive definite")
    identity = _identity_like(precision_h)
    covariance_k = torch.linalg.solve(identity - precision_h, precision_h)
    return 0.5 * (covariance_k + _adjoint(covariance_k))


def kl_q_to_p_from_covariance(covariance_k: Tensor) -> Tensor:
    r"""Return ``KL(q_K || p) = -1/2 log det_2(I + K)``."""
    return -0.5 * covariance_logdet2(covariance_k)


def kl_p_to_q_from_covariance(covariance_k: Tensor) -> Tensor:
    r"""Return ``KL(p || q_K)`` using the corresponding precision ``H``."""
    return kl_p_to_q_from_precision(precision_from_covariance(covariance_k))


def kl_p_to_q_from_precision(precision_h: Tensor) -> Tensor:
    r"""Return ``KL(p || q) = -1/2 log det_2(I - H)``."""
    return -0.5 * precision_logdet2(precision_h)


def kl_q_to_p_from_precision(precision_h: Tensor) -> Tensor:
    r"""Return ``KL(q || p)`` using ``K = H(I-H)^(-1)``."""
    return kl_q_to_p_from_covariance(covariance_from_precision(precision_h))


def precision_wick_log_q_over_p(
    precision_h: Tensor,
    features: Tensor,
) -> Tensor:
    r"""Evaluate the exact dense Wick form of ``log(dq/dp)`` per sample.

    With ``p=N(0,I)`` and ``H=I-(I+K)^(-1)``,

    ``log(dq/dp)(x) = 1/2 (x*Hx - Tr H) + 1/2 log det_2(I-H)``.

    ``features`` may have arbitrary leading batch dimensions and must end in
    the event dimension.  Expectations under ``q`` give ``KL(q||p)``;
    expectations of the negative expression under ``p`` give ``KL(p||q)``.
    """
    _check_self_adjoint(precision_h, "precision_h")
    if precision_h.ndim != 2:
        raise ValueError("precision_wick_log_q_over_p expects an unbatched H")
    if features.shape[-1] != precision_h.shape[-1]:
        raise ValueError("features and precision_h have incompatible dimensions")
    quadratic = torch.einsum("...i,ij,...j->...", features, precision_h, features)
    trace_h = torch.diagonal(precision_h).sum()
    return 0.5 * (quadratic - trace_h) + 0.5 * precision_logdet2(precision_h)


def precision_wick_log_p_over_q(
    precision_h: Tensor,
    features: Tensor,
) -> Tensor:
    r"""Evaluate ``log(dp/dq)`` per sample; the negative of the Wick form."""
    return -precision_wick_log_q_over_p(precision_h, features)


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


def precision_neg_logdet2_series(
    precision_operator_fn: OperatorFn,
    probes: Tensor,
    *,
    series_order: int = 5,
    first_power: Optional[Tensor] = None,
) -> Tensor:
    r"""Estimate ``-log det_2(I-H) = sum_{k>=2} Tr(H^k)/k``.

    ``operator_fn`` is the relative precision perturbation ``H``.  The
    truncated power series is intended for self-adjoint ``H`` with spectral
    radius strictly below one.  ``series_order`` is the largest retained
    power; it is intentionally not named ``K``, which is reserved for the
    relative covariance perturbation.

    A warning is emitted when successive trace estimates suggest the spectral
    radius is near or above 1, where the truncated polynomial no longer
    approximates the barrier functional.
    """
    if series_order < 2:
        raise ValueError("series_order must be at least 2 for a det_2 estimate")

    dims = _event_dims(probes)
    p_power_z = (
        first_power
        if first_power is not None
        else precision_operator_fn(probes)
    )
    neg_logdet2 = probes.new_zeros(())

    trace_prev: Optional[Tensor] = None
    trace_k: Optional[Tensor] = None

    for k in range(2, series_order + 1):
        p_power_z = precision_operator_fn(p_power_z)
        if p_power_z.shape != probes.shape:
            raise ValueError("precision_operator_fn must preserve the probe shape")
        trace_k = (probes * p_power_z).sum(dim=dims).mean()
        neg_logdet2 = neg_logdet2 + trace_k / k

        if trace_prev is not None:
            ratio = (trace_k.abs() / (trace_prev.abs() + 1e-30)).clamp(max=100.0)
            ratio_value = float(ratio.detach())
            if ratio_value > 0.92:
                warnings.warn(
                    "precision_neg_logdet2_series: "
                    f"|Tr(H^{k})| / |Tr(H^{k-1})| ≈ {ratio_value:.3f}. "
                    "Spectral radius may be near or above 1; "
                    f"series_order={series_order} truncation is unreliable. "
                    "The series only converges for ||H|| < 1. Increase "
                    "series_order or clamp eigenvalues below 1.",
                    stacklevel=2,
                )
                break  # warn once, don't spam for every subsequent k
        trace_prev = trace_k

    return neg_logdet2


def neg_logdet2_series(
    operator_fn: OperatorFn,
    probes: Tensor,
    *,
    K: int = 5,
    first_power: Optional[Tensor] = None,
) -> Tensor:
    r"""Compatibility wrapper for :func:`precision_neg_logdet2_series`.

    Historical callers use ``K`` for the series order.  New code should use
    ``series_order`` so that ``K`` remains unambiguously the covariance
    perturbation.
    """
    return precision_neg_logdet2_series(
        operator_fn,
        probes,
        series_order=K,
        first_power=first_power,
    )


def matrix_free_precision_wick_log_q_over_p(
    precision_operator_fn: OperatorFn,
    features: Tensor,
    series_order: int = 5,
    num_probes: int = 10,
    *,
    probes: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_diagnostics: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]:
    r"""Estimate the batch mean of ``log(dq/dp)`` from precision products.

    ``precision_operator_fn(x)`` must apply
    ``H = I - (I + K)^(-1)``.  This API never accepts an amplitude ``B`` or a
    covariance ``K``.  The estimator is

    ``1/2(<x,Hx>-Tr H) + 1/2 log det_2(I-H)``.

    Its expectation is ``KL(q||p)`` only when ``features`` are sampled from
    ``q``.  The returned Hilbert--Schmidt diagnostic is ``||H||_HS^2`` and is
    therefore named ``precision_hs2``; it is not the canonical covariance
    quantity ``T2=||K||_HS^2``.
    """
    dims = _event_dims(features)
    if probes is None:
        probes = rademacher_probes(features, num_probes, generator=generator)
    elif probes.shape[1:] != features.shape[1:]:
        raise ValueError("probes must have shape (num_probes, *features.shape[1:])")

    trace_est, hz = hutchinson_trace(precision_operator_fn, probes)
    precision_hs2_est = hutchinson_hs2(hz)

    # For the intended positive H, Tr(H^2)/Tr(H) is a conservative scale
    # diagnostic.  It is not a replacement for an operator-norm estimate.
    trace_abs = trace_est.abs() + 1e-30
    rho_scale = (precision_hs2_est / trace_abs).clamp(max=0.999)
    rho_value = float(rho_scale.detach())
    if rho_value > 0.75:
        tail_bound = rho_scale ** (series_order + 1) / (
            (series_order + 1) * (1.0 - rho_scale) + 1e-30
        )
        tail_value = float(tail_bound.detach())
        warnings.warn(
            "matrix_free_precision_wick_log_q_over_p: "
            f"Tr(H^2)/|Tr(H)|≈{rho_value:.3f}; "
            f"series_order={series_order} may leave a "
            f"non-negligible series tail (single-mode bound ≈{tail_value:.3f}). "
            "Require ||H||op<1 and increase series_order or tighten the gate.",
            stacklevel=2,
        )

    hx = precision_operator_fn(features)
    if hx.shape != features.shape:
        raise ValueError(
            "precision_operator_fn(features) must have the same shape as features"
        )

    quadratic = (features * hx).sum(dim=dims).mean()
    wick_quadratic = 0.5 * (quadratic - trace_est)
    neg_logdet2_i_minus_h = precision_neg_logdet2_series(
        precision_operator_fn,
        probes,
        series_order=series_order,
        first_power=hz,
    )
    log_q_over_p = wick_quadratic - 0.5 * neg_logdet2_i_minus_h

    if not return_diagnostics:
        return log_q_over_p, precision_hs2_est

    diagnostics = {
        "precision_trace": trace_est,
        "precision_hs2": precision_hs2_est,
        "p_centered_wick_quadratic": wick_quadratic,
        "neg_logdet2_i_minus_h": neg_logdet2_i_minus_h,
        # Compatibility keys.  Their precise meanings are given above.
        "trace": trace_est,
        "hs2": precision_hs2_est,
        "wick_quadratic": wick_quadratic,
        "neg_logdet2": neg_logdet2_i_minus_h,
    }
    return log_q_over_p, diagnostics


def matrix_free_precision_wick_kl(
    precision_operator_fn: OperatorFn,
    features: Tensor,
    *,
    direction: KLDirection,
    series_order: int = 5,
    num_probes: int = 10,
    probes: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_diagnostics: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]:
    r"""Estimate an explicitly directed KL using the precision Wick form.

    For ``direction="q_to_p"``, ``features`` must be sampled from ``q`` and
    the estimator returns ``log(dq/dp)``.  For ``direction="p_to_q"``, they
    must be sampled from ``p`` and it returns ``log(dp/dq)``.  The function
    cannot infer or verify the sampling distribution.
    """
    if direction not in ("q_to_p", "p_to_q"):
        raise ValueError("direction must be 'q_to_p' or 'p_to_q'")
    value, diagnostic = matrix_free_precision_wick_log_q_over_p(
        precision_operator_fn,
        features,
        series_order=series_order,
        num_probes=num_probes,
        probes=probes,
        generator=generator,
        return_diagnostics=return_diagnostics,
    )
    if direction == "p_to_q":
        value = -value
    return value, diagnostic


def schatten_cf_penalty(
    operator_fn: OperatorFn,
    features: Tensor,
    K: int = 5,
    num_probes: int = 10,
    *,
    mode: str = "legacy_penalty",
    probes: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_diagnostics: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Dict[str, Tensor]]]:
    r"""Compatibility wrapper for the precision-H Wick estimator.

    New code should use :func:`matrix_free_precision_wick_log_q_over_p` or
    :func:`matrix_free_precision_wick_kl`.  Here ``operator_fn`` always means
    precision ``H``; passing amplitude ``B`` or covariance ``K`` is invalid.

    Modes:
      - ``"log_q_over_p"`` or ``"kl_q_to_p"``: the ``log(dq/dp)`` form.
      - ``"log_p_over_q"`` or ``"kl_p_to_q"``: its negative.
      - ``"legacy_penalty"``: the historical non-likelihood expression
        ``1/2(<X,HX>-Tr H) + 1/2[-log det_2(I-H)]``.

    The old names ``"penalty"``, ``"log_density"`` and
    ``"negative_log_density"`` remain aliases and emit a deprecation warning.
    """
    aliases = {
        "penalty": "legacy_penalty",
        "log_density": "log_q_over_p",
        "negative_log_density": "log_p_over_q",
    }
    if mode in aliases:
        canonical_mode = aliases[mode]
        warnings.warn(
            f"mode={mode!r} is ambiguous and deprecated; use "
            f"mode={canonical_mode!r}",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = canonical_mode

    log_q_over_p, diagnostics = matrix_free_precision_wick_log_q_over_p(
        operator_fn,
        features=features,
        series_order=K,
        num_probes=num_probes,
        probes=probes,
        generator=generator,
        return_diagnostics=True,
    )

    if mode in ("log_q_over_p", "kl_q_to_p"):
        value = log_q_over_p
    elif mode in ("log_p_over_q", "kl_p_to_q"):
        value = -log_q_over_p
    elif mode == "legacy_penalty":
        value = (
            diagnostics["p_centered_wick_quadratic"]
            + 0.5 * diagnostics["neg_logdet2_i_minus_h"]
        )
    else:
        raise ValueError(
            "mode must be one of 'log_q_over_p', 'log_p_over_q', "
            "'kl_q_to_p', 'kl_p_to_q', or 'legacy_penalty'"
        )

    if not return_diagnostics:
        return value, diagnostics["precision_hs2"]
    return value, diagnostics


class WickCarlemanPenalty(nn.Module):
    """Legacy nn.Module wrapper; prefer explicit precision-Wick functions."""

    def __init__(
        self,
        *,
        K: int = 5,
        num_probes: int = 10,
        mode: str = "legacy_penalty",
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
    "covariance_from_amplitude",
    "covariance_from_precision",
    "covariance_logdet2",
    "covariance_t2",
    "hutchinson_hs2",
    "hutchinson_trace",
    "kl_p_to_q_from_covariance",
    "kl_p_to_q_from_precision",
    "kl_q_to_p_from_covariance",
    "kl_q_to_p_from_precision",
    "matrix_free_precision_wick_kl",
    "matrix_free_precision_wick_log_q_over_p",
    "neg_logdet2_series",
    "precision_neg_logdet2_series",
    "precision_from_covariance",
    "precision_logdet2",
    "precision_wick_log_p_over_q",
    "precision_wick_log_q_over_p",
    "rademacher_probes",
    "sample_covariance_perturbation",
    "sample_q_from_amplitude",
    "schatten_cf_penalty",
]
