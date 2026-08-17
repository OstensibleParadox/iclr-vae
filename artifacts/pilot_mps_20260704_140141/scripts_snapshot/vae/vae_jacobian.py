"""
vae_jacobian.py — Real SD-VAE Jacobian Spectral Analysis

Verifies that the pretrained SD-VAE encoder acts as an operator-ideal
(Schatten) filter: its Jacobian's singular value spectrum decays rapidly
enough to place the induced latent-space covariance operator in S₂.

Usage:
    python3 experiments/vae/vae_jacobian.py
    python3 experiments/vae/vae_jacobian.py --seed 0

Appendix-level reproducibility details:
    Model      : stabilityai/sd-vae-ft-mse (AutoencoderKL, 84M params, frozen)
    Encoder    : vae.encoder + vae.quant_conv — outputs latent mean μ (4 channels)
    VAE scaling: 0.18215 is NOT applied (we analyze raw encoder geometry)
    Input range: [-1, 1]  (SD convention: 2 * img_uint8 / 255 - 1)
    Crop/resize: center-crop to square, bilinear resize to target resolution
    Probes     : Rademacher ±1 in pixel space (flattened R^D)
    Freezing   : All encoder parameters require_grad=False
    Jacobian   : reverse-mode VJP (one backward per latent dim) for ≤256×256
                 Hutchinson estimator (m=50 probes) for 512×512
"""
from __future__ import annotations

import csv
import gc
import math
import os
import sys
import argparse
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gaussian-hilbert-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gaussian-hilbert-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "vae"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device selection: prefer MPS (Apple Silicon), then CUDA, then CPU
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = select_device()
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Load & freeze SD-VAE
# ---------------------------------------------------------------------------

def load_frozen_vae() -> nn.Module:
    from diffusers import AutoencoderKL
    print("Loading stabilityai/sd-vae-ft-mse …")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32,
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    return vae


def encode_mean(vae: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Encode x → latent mean μ.  x must be in [-1, 1], shape (1, 3, H, W)."""
    # We use vae.encode which returns a DiagonalGaussianDistribution.
    # We take .mean (the μ output of quant_conv) — no reparameterization.
    return vae.encode(x).latent_dist.mean


# ---------------------------------------------------------------------------
# Input construction (3 types)
# ---------------------------------------------------------------------------

def make_inputs(resolution: int, seed: int | None = None) -> dict[str, torch.Tensor]:
    """Return dict of {name: (1, 3, H, W)} tensors in [-1, 1]."""
    H = W = resolution
    inputs: dict[str, torch.Tensor] = {}

    # 1. Gaussian white noise clipped to [-1, 1]
    if seed is None:
        x_noise = torch.randn(1, 3, H, W)
    else:
        noise_rng = torch.Generator().manual_seed(int(seed))
        x_noise = torch.randn(1, 3, H, W, generator=noise_rng)

    x_noise = x_noise.clamp(-1.0, 1.0)
    inputs["white_noise"] = x_noise

    # 2. Constant gray (zero)
    inputs["constant_gray"] = torch.zeros(1, 3, H, W)

    # 3. Structured (low-frequency) image: sinusoidal pattern
    #    This simulates a "real image"-like input without needing to download data.
    yy = torch.linspace(-math.pi, math.pi, H).unsqueeze(1).expand(H, W)
    xx = torch.linspace(-math.pi, math.pi, W).unsqueeze(0).expand(H, W)
    pattern = 0.5 * (torch.sin(2 * yy) + torch.cos(3 * xx))  # in (-1, 1)
    # Make 3-channel with slight RGB variation
    x_struct = torch.stack([pattern, pattern.roll(H // 4, 0), pattern.roll(W // 4, 1)], dim=0).unsqueeze(0)
    inputs["structured"] = x_struct.clamp(-1.0, 1.0)

    return {k: v.to(DEVICE) for k, v in inputs.items()}


def input_seed(resolution: int, input_name: str, seed_base: int = 0) -> int:
    """Deterministic seed for each (resolution, input_name, base seed)."""
    offsets = {"white_noise": 0, "constant_gray": 1, "structured": 2}
    if input_name not in offsets:
        raise ValueError(f"Unknown input_type: {input_name}")
    return int(seed_base) * 1_000_000 + resolution * 1_000 + offsets[input_name]


# ---------------------------------------------------------------------------
# Jacobian computation (exact, via VJP)
# ---------------------------------------------------------------------------

def compute_jacobian_exact(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    latent_dim: int,
) -> torch.Tensor:
    """
    Compute the full Jacobian J ∈ R^{d × D} via reverse-mode AD.

    fn  : R^D → R^d  (flattened input/output)
    x0  : (1, 3, H, W) input tensor, will be detached and requires_grad set

    Returns J as a CPU float32 tensor.
    """
    D = x0.numel()
    d = latent_dim
    J = torch.zeros(d, D, dtype=torch.float32)

    x0_flat = x0.detach().reshape(1, -1)  # (1, D)

    for i in range(d):
        xi = x0_flat.clone().requires_grad_(True)
        out = fn(xi.reshape_as(x0))  # (1, 4, h, w)
        out_flat = out.reshape(-1)    # (d,)
        out_flat[i].backward()
        J[i] = xi.grad.detach().reshape(-1).cpu()
        if (i + 1) % max(1, d // 10) == 0:
            print(f"    Jacobian row {i+1}/{d} done", end="\r", flush=True)

    print()
    return J


# ---------------------------------------------------------------------------
# Hutchinson estimation (for large resolutions)
# ---------------------------------------------------------------------------

def hutchinson_jjt_trace(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    num_probes: int = 50,
) -> tuple[float, float]:
    """
    Estimate Tr(J J^T) and Tr((J J^T)²) without forming J explicitly.

      Tr(J J^T) = E[||J z||²]   where z is Rademacher in R^D
      Tr((J J^T)²) = E[||J^T (J z)||²]  (bilinear, two-pass)

    Returns (tr_estimate, t2_estimate).
    """
    D = x0.numel()
    tr_samples = []
    t2_samples = []

    for r in range(num_probes):
        # --- forward pass: compute Jz via JVP (forward-mode) ---
        z = torch.randint(0, 2, (1, 3, *x0.shape[2:]), device=DEVICE).float() * 2 - 1  # Rademacher

        # JVP: d/dt f(x0 + t*z)|_{t=0}
        x0_req = x0.detach().clone().requires_grad_(False)
        z_req = z.clone()
        with torch.no_grad():
            # Numerical JVP (finite difference, 2nd order)
            eps = 1e-3
            out_plus  = fn(x0_req + eps * z_req).detach().reshape(-1)
            out_minus = fn(x0_req - eps * z_req).detach().reshape(-1)
            Jz = (out_plus - out_minus) / (2 * eps)   # (d,)

        tr_samples.append(Jz.pow(2).sum().item())

        # J^T (Jz): backward through fn at x0 with cotangent = Jz
        x0_back = x0.detach().clone().requires_grad_(True)
        out_back = fn(x0_back).reshape(-1)
        out_back.backward(Jz.to(DEVICE))
        JtJz = x0_back.grad.detach().reshape(-1)   # (D,)
        t2_samples.append(JtJz.pow(2).sum().item())

        if (r + 1) % 10 == 0:
            print(f"    Hutchinson probe {r+1}/{num_probes}", end="\r", flush=True)

    print()
    tr_est = float(np.mean(tr_samples))
    t2_est = float(np.mean(t2_samples))
    return tr_est, t2_est


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

def fit_power_law(singular_values: np.ndarray) -> tuple[float, float]:
    """Fit s_i ~ C * i^{-alpha} via log-log OLS.  Returns (alpha, C)."""
    n = len(singular_values)
    i = np.arange(1, n + 1, dtype=float)
    log_i = np.log(i)
    log_s = np.log(singular_values + 1e-30)
    # OLS: log_s = log_C - alpha * log_i
    valid = singular_values > 1e-10
    if valid.sum() < 3:
        return float("nan"), float("nan")
    A = np.stack([np.ones(valid.sum()), -log_i[valid]], axis=1)
    b = log_s[valid]
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    log_C, alpha = coef
    return float(alpha), float(math.exp(log_C))


def effective_rank(singular_values: np.ndarray) -> float:
    """r_eff = (Σ s²)² / Σ s⁴"""
    s2 = singular_values ** 2
    s4 = singular_values ** 4
    denom = s4.sum()
    if denom < 1e-30:
        return 0.0
    return float(s2.sum() ** 2 / denom)


# ---------------------------------------------------------------------------
# Finite-difference sanity check
# ---------------------------------------------------------------------------

def fd_sanity_check(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    J: torch.Tensor,
    num_checks: int = 5,
    eps: float = 1e-3,
) -> float:
    """
    Check ||J @ e_j - (f(x+eps*e_j) - f(x-eps*e_j))/(2eps)||_2 / ||J e_j||_2
    for a few random pixels j.  Returns max relative error.
    """
    D = x0.numel()
    errors = []
    with torch.no_grad():
        f0 = fn(x0).reshape(-1).cpu()
        for _ in range(num_checks):
            j = int(torch.randint(0, D, ()).item())
            e_j = torch.zeros(D, device=DEVICE)
            e_j[j] = 1.0
            delta = e_j.reshape_as(x0) * eps
            fp = fn(x0 + delta).reshape(-1).cpu()
            fm = fn(x0 - delta).reshape(-1).cpu()
            fd_col = (fp - fm) / (2 * eps)
            J_col = J[:, j]
            err = (J_col - fd_col).norm() / (J_col.norm() + 1e-30)
            errors.append(err.item())
    return float(max(errors))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

class ResolutionResult(NamedTuple):
    resolution: int
    input_type: str
    pixel_dim: int
    latent_dim: int
    tr_jjt: float         # Tr(J J^T) = Σ s²
    t2_jjt: float         # Tr((J J^T)²) = Σ s⁴
    alpha: float          # power-law exponent
    eff_rank: float       # effective rank
    condition_number: float
    s_max: float
    s_min_nonzero: float
    hutchinson_only: bool
    fd_rel_error: float   # NaN if hutchinson_only


def run_experiment(
    vae: nn.Module,
    resolutions: list[int] = (64, 128, 256),
    hutchinson_resolutions: list[int] = (512,),
    num_probes_hutchinson: int = 50,
    save_spectra: bool = True,
    seed: int = 0,
) -> list[ResolutionResult]:

    results = []
    spectra: dict[tuple[int, str], np.ndarray] = {}  # for plotting

    # Define encoder function (CPU → latent mean)
    def encode_fn(x: torch.Tensor) -> torch.Tensor:
        return encode_mean(vae, x)

    for res in list(resolutions) + list(hutchinson_resolutions):
        hutchinson_only = res in hutchinson_resolutions
        inputs = {
            name: make_inputs(res, seed=input_seed(res, name, seed))[name]
            for name in ("white_noise", "constant_gray", "structured")
        }
        latent_h = res // 8  # SD-VAE downsamples by factor 8
        latent_w = res // 8
        latent_dim = 4 * latent_h * latent_w  # 4 channels
        pixel_dim = 3 * res * res

        print(f"\n{'='*60}")
        print(f"Resolution {res}×{res}  |  D={pixel_dim}  d={latent_dim}  {'[Hutchinson only]' if hutchinson_only else '[Exact Jacobian]'}")
        print(f"{'='*60}")

        for input_name, x0 in inputs.items():
            print(f"\n  Input type: {input_name}")

            if hutchinson_only:
                # Hutchinson estimation only
                tr_est, t2_est = hutchinson_jjt_trace(encode_fn, x0, num_probes=num_probes_hutchinson)
                print(f"    Hutchinson: Tr(JJ^T) = {tr_est:.4f}  T₂ = {t2_est:.4f}")
                results.append(ResolutionResult(
                    resolution=res,
                    input_type=input_name,
                    pixel_dim=pixel_dim,
                    latent_dim=latent_dim,
                    tr_jjt=tr_est,
                    t2_jjt=t2_est,
                    alpha=float("nan"),
                    eff_rank=float("nan"),
                    condition_number=float("nan"),
                    s_max=float("nan"),
                    s_min_nonzero=float("nan"),
                    hutchinson_only=True,
                    fd_rel_error=float("nan"),
                ))
            else:
                # Exact Jacobian via VJP
                print(f"    Computing Jacobian ({latent_dim} VJPs) …")
                J = compute_jacobian_exact(encode_fn, x0, latent_dim)

                # SVD
                print("    Computing SVD …")
                U, S, Vh = torch.linalg.svd(J, full_matrices=False)
                S_np = S.numpy()

                # Diagnostics
                tr_val = float(S_np.sum() ** 0 * (S_np ** 2).sum())  # Σ s²
                # Careful: Tr(JJ^T) = Σ s_i²
                tr_jjt = float((S_np ** 2).sum())
                t2_jjt = float((S_np ** 4).sum())
                alpha, C = fit_power_law(S_np)
                r_eff = effective_rank(S_np)
                s_max = float(S_np.max())
                nonzero = S_np[S_np > S_np.max() * 1e-6]
                s_min_nz = float(nonzero.min()) if len(nonzero) > 0 else 0.0
                cond = s_max / s_min_nz if s_min_nz > 0 else float("inf")

                print(f"    Tr(JJ^T) = {tr_jjt:.4f}")
                print(f"    T₂       = {t2_jjt:.4f}")
                print(f"    α (fit)  = {alpha:.4f}")
                print(f"    r_eff    = {r_eff:.2f}")
                print(f"    cond     = {cond:.2f}")

                # Finite-difference sanity check (skip for MPS — slow but do it)
                print("    Finite-difference sanity check …")
                fd_err = fd_sanity_check(encode_fn, x0, J, num_checks=5)
                print(f"    FD rel error (max over 5 pixels) = {fd_err:.4e}")

                if save_spectra:
                    spectra[(res, input_name)] = S_np

                results.append(ResolutionResult(
                    resolution=res,
                    input_type=input_name,
                    pixel_dim=pixel_dim,
                    latent_dim=latent_dim,
                    tr_jjt=tr_jjt,
                    t2_jjt=t2_jjt,
                    alpha=alpha,
                    eff_rank=r_eff,
                    condition_number=cond,
                    s_max=s_max,
                    s_min_nonzero=s_min_nz,
                    hutchinson_only=False,
                    fd_rel_error=fd_err,
                ))

                # free memory
                del J, U, S, Vh
                gc.collect()

    return results, spectra


# ---------------------------------------------------------------------------
# Hutchinson vs Exact cross-validation (at 128×128)
# ---------------------------------------------------------------------------

def hutchinson_vs_exact_check(vae: nn.Module, seed: int = 0) -> dict:
    """At 128×128, compare Hutchinson Tr estimate vs exact Σ s²."""
    res = 128
    inputs = make_inputs(res, seed=input_seed(res, "white_noise", seed))
    x0 = inputs["white_noise"]
    latent_dim = 4 * (res // 8) ** 2

    encode_fn = lambda x: encode_mean(vae, x)

    print("\nHutchinson vs Exact cross-validation (128×128, white noise) …")
    J = compute_jacobian_exact(encode_fn, x0, latent_dim)
    _, S, _ = torch.linalg.svd(J, full_matrices=False)
    exact_tr = float((S.numpy() ** 2).sum())
    exact_t2 = float((S.numpy() ** 4).sum())
    del J, S

    tr_hutch, t2_hutch = hutchinson_jjt_trace(encode_fn, x0, num_probes=50)

    rel_err_tr = abs(tr_hutch - exact_tr) / exact_tr
    rel_err_t2 = abs(t2_hutch - exact_t2) / exact_t2

    print(f"  Exact Tr(JJ^T)    = {exact_tr:.6f}")
    print(f"  Hutchinson Tr     = {tr_hutch:.6f}   rel err = {rel_err_tr:.4f}")
    print(f"  Exact T₂          = {exact_t2:.6f}")
    print(f"  Hutchinson T₂     = {t2_hutch:.6f}   rel err = {rel_err_t2:.4f}")

    return {
        "exact_tr": exact_tr, "hutch_tr": tr_hutch, "rel_err_tr": rel_err_tr,
        "exact_t2": exact_t2, "hutch_t2": t2_hutch, "rel_err_t2": rel_err_t2,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    results: list[ResolutionResult],
    spectra: dict,
    cross_val: dict,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        input_types = ["white_noise", "constant_gray", "structured"]
        colors = {"white_noise": "#e74c3c", "constant_gray": "#3498db", "structured": "#2ecc71"}
        labels = {"white_noise": "White noise", "constant_gray": "Constant gray", "structured": "Structured (sinusoidal)"}

        # ---------------------------------------------------------------
        # Figure 1: Main figure (3 panels)
        # ---------------------------------------------------------------
        fig = plt.figure(figsize=(18, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # Panel A: Singular value spectrum at 256×256 (or largest available)
        ax_a = fig.add_subplot(gs[0])
        spec_res = max(r for (r, _) in spectra.keys())
        for itype in input_types:
            key = (spec_res, itype)
            if key not in spectra:
                continue
            S_np = spectra[key]
            i_idx = np.arange(1, len(S_np) + 1)
            ax_a.plot(i_idx, S_np, color=colors[itype], alpha=0.8, linewidth=1.2, label=labels[itype])
            # Power-law fit overlay
            alpha, C = fit_power_law(S_np)
            if not math.isnan(alpha):
                fit_vals = C * i_idx ** (-alpha)
                ax_a.plot(i_idx, fit_vals, "--", color=colors[itype], alpha=0.5, linewidth=1.0)
                ax_a.annotate(f"α={alpha:.2f}", xy=(i_idx[len(i_idx)//2], fit_vals[len(i_idx)//2]),
                              fontsize=8, color=colors[itype])
        ax_a.set_xscale("log")
        ax_a.set_yscale("log")
        ax_a.set_xlabel("Singular value index $i$")
        ax_a.set_ylabel("Singular value $s_i$")
        ax_a.set_title(f"Panel A: $s_i$ vs $i$ (log–log)\nResolution {spec_res}×{spec_res}", fontsize=11)
        ax_a.legend(fontsize=8)
        ax_a.grid(True, alpha=0.3)

        # Panel B: T₂ = Σ s⁴ vs resolution
        ax_b = fig.add_subplot(gs[1])
        exact_results = [r for r in results if not r.hutchinson_only]
        hutch_results = [r for r in results if r.hutchinson_only]
        for itype in input_types:
            res_list = sorted(set(r.resolution for r in exact_results if r.input_type == itype))
            t2_list = [r.t2_jjt for r in exact_results if r.input_type == itype and r.resolution in res_list]
            res_list_s = sorted(res_list)
            if res_list_s:
                ax_b.plot(res_list_s, t2_list, "o-", color=colors[itype], linewidth=2, label=labels[itype])
            # Hutchinson points (dashed, star marker)
            res_h = sorted(set(r.resolution for r in hutch_results if r.input_type == itype))
            t2_h = [r.t2_jjt for r in hutch_results if r.input_type == itype]
            if res_h:
                ax_b.plot(res_h, t2_h, "*", color=colors[itype], markersize=12, label=f"{labels[itype]} (Hutch.)")
        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
        ax_b.set_xlabel("Resolution $H = W$")
        ax_b.set_ylabel("$T_2 = \\mathrm{Tr}((JJ^\\top)^2) = \\sum s_i^4$")
        ax_b.set_title("Panel B: Schatten $T_2$ diagnostic vs resolution\n$T_2 \\in \\mathcal{S}_2$ iff $\\sum s_i^4 < \\infty$", fontsize=11)
        ax_b.legend(fontsize=7)
        ax_b.grid(True, alpha=0.3)

        # Panel C: Effective rank vs resolution
        ax_c = fig.add_subplot(gs[2])
        for itype in input_types:
            res_list = sorted(set(r.resolution for r in exact_results if r.input_type == itype))
            reff_list = [r.eff_rank for r in exact_results if r.input_type == itype and r.resolution in res_list]
            if res_list:
                ax_c.plot(res_list, reff_list, "o-", color=colors[itype], linewidth=2, label=labels[itype])
        ax_c.set_xscale("log")
        ax_c.set_xlabel("Resolution $H = W$")
        ax_c.set_ylabel("Effective rank $r_{\\mathrm{eff}}$")
        ax_c.set_title("Panel C: Effective rank vs resolution\n$r_{\\mathrm{eff}} = (\\sum s_i^2)^2 / \\sum s_i^4$", fontsize=11)
        ax_c.legend(fontsize=8)
        ax_c.grid(True, alpha=0.3)

        fig.suptitle(
            "SD-VAE Encoder Jacobian Spectral Analysis\n"
            "(stabilityai/sd-vae-ft-mse, frozen, input ∈ [−1, 1])",
            fontsize=13, fontweight="bold",
            y=0.98,
        )
        fig.subplots_adjust(top=0.84)
        out = EXP_DIR / "vae_jacobian_results.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"\nMain figure saved: {out}")
        plt.close(fig)

        # ---------------------------------------------------------------
        # Figure 2: Spectrum detail (all resolutions for white noise)
        # ---------------------------------------------------------------
        resolutions = sorted(set(r for (r, _) in spectra.keys()))
        if len(resolutions) > 1:
            fig2, ax = plt.subplots(figsize=(9, 5))
            cmap = plt.get_cmap("plasma")
            for k, res in enumerate(resolutions):
                key = (res, "white_noise")
                if key not in spectra:
                    continue
                S_np = spectra[key]
                i_idx = np.arange(1, len(S_np) + 1)
                c = cmap(k / max(len(resolutions) - 1, 1))
                ax.plot(i_idx, S_np, color=c, linewidth=1.5, label=f"{res}×{res} (d={len(S_np)})")
                alpha, C = fit_power_law(S_np)
                if not math.isnan(alpha):
                    fit_vals = C * i_idx ** (-alpha)
                    ax.plot(i_idx, fit_vals, "--", color=c, alpha=0.4, linewidth=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Singular value index $i$")
            ax.set_ylabel("Singular value $s_i$")
            ax.set_title("SD-VAE Jacobian Singular Value Spectrum (white noise input)\nDashed: power-law fit $s_i \\sim C \\cdot i^{-\\alpha}$")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig2.tight_layout()
            out2 = EXP_DIR / "vae_jacobian_spectrum.png"
            fig2.savefig(out2, dpi=200, bbox_inches="tight")
            print(f"Spectrum figure saved: {out2}")
            plt.close(fig2)

    except ImportError as e:
        print(f"matplotlib not available, skipping plots: {e}")


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> list[ResolutionResult]:
    """Load results from CSV back into ResolutionResult namedtuples."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(ResolutionResult(
                resolution=int(row["resolution"]),
                input_type=row["input_type"],
                pixel_dim=int(row["pixel_dim"]),
                latent_dim=int(row["latent_dim"]),
                tr_jjt=float(row["tr_jjt"]),
                t2_jjt=float(row["t2_jjt"]),
                alpha=float(row["alpha"]),
                eff_rank=float(row["eff_rank"]),
                condition_number=float(row["condition_number"]),
                s_max=float(row["s_max"]),
                s_min_nonzero=float(row["s_min_nonzero"]),
                hutchinson_only=row["hutchinson_only"] == "True",
                fd_rel_error=float(row["fd_rel_error"]),
            ))
    return rows


def load_crossval_csv(cv_path: Path) -> dict:
    """Load cross-validation results from CSV."""
    with open(cv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    row = rows[0]
    return {
        "exact_tr": float(row["exact_tr"]),
        "hutch_tr": float(row["hutch_tr"]),
        "rel_err_tr": float(row["rel_err_tr"]),
        "exact_t2": float(row["exact_t2"]),
        "hutch_t2": float(row["hutch_t2"]),
        "rel_err_t2": float(row["rel_err_t2"]),
    }


def save_csv(results: list[ResolutionResult], cross_val: dict) -> None:
    csv_path = EXP_DIR / "vae_jacobian_results.csv"
    fieldnames = list(ResolutionResult._fields)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r._asdict())
    print(f"CSV saved: {csv_path}")

    # Cross-validation appendix
    cv_path = EXP_DIR / "vae_jacobian_crossval.csv"
    with open(cv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cross_val.keys()))
        writer.writeheader()
        writer.writerow(cross_val)
    print(f"Cross-validation CSV saved: {cv_path}")


# ---------------------------------------------------------------------------
# Print appendix table
# ---------------------------------------------------------------------------

def print_appendix_table(results: list[ResolutionResult]) -> None:
    print("\n" + "="*100)
    print("APPENDIX TABLE: SD-VAE Jacobian Schatten Diagnostics")
    print("="*100)
    header = f"{'Res':>6} {'Input':<18} {'D':>8} {'d':>6} {'Tr(JJ^T)':>12} {'T₂':>12} {'α':>8} {'r_eff':>8} {'cond':>10} {'Hutch?':>6} {'FD err':>10}"
    print(header)
    print("-" * 100)
    for r in results:
        hutch_str = "yes" if r.hutchinson_only else "no"
        fd_str = "—" if math.isnan(r.fd_rel_error) else f"{r.fd_rel_error:.2e}"
        alpha_str = "—" if math.isnan(r.alpha) else f"{r.alpha:.3f}"
        reff_str = "—" if math.isnan(r.eff_rank) else f"{r.eff_rank:.1f}"
        cond_str = "—" if math.isnan(r.condition_number) else f"{r.condition_number:.1f}"
        print(
            f"{r.resolution:>6}×{r.resolution:<2} "
            f"{r.input_type:<18} "
            f"{r.pixel_dim:>8} "
            f"{r.latent_dim:>6} "
            f"{r.tr_jjt:>12.4f} "
            f"{r.t2_jjt:>12.4f} "
            f"{alpha_str:>8} "
            f"{reff_str:>8} "
            f"{cond_str:>10} "
            f"{hutch_str:>6} "
            f"{fd_str:>10}"
        )
    print("="*100)

    # S₂ membership summary
    print("\nSchatten S₂ membership: P_lat ∈ S₂  ⟺  α > 0.5  (i.e. Σ s_i⁴ < ∞ in d→∞ limit)")
    exact = [r for r in results if not r.hutchinson_only and not math.isnan(r.alpha)]
    if exact:
        alphas = [r.alpha for r in exact]
        print(f"  α values: {[f'{a:.3f}' for a in alphas]}")
        print(f"  Mean α = {np.mean(alphas):.3f}  ±  {np.std(alphas):.3f}")
        verdict = "✓ CONFIRMED: S₂ membership satisfied (α > 0.5)" if np.mean(alphas) > 0.5 else "✗ WARNING: α ≤ 0.5, S₂ membership not confirmed"
        print(f"  {verdict}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SD-VAE Jacobian spectral experiment.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for input generation (default: 0).")
    parser.add_argument("--plot-only", action="store_true", help="Skip computation, regenerate plots from saved CSVs/spectra.")
    args = parser.parse_args()

    if args.plot_only:
        import numpy as np
        csv_path = EXP_DIR / "vae_jacobian_results.csv"
        cv_path = EXP_DIR / "vae_jacobian_crossval.csv"
        spectra_path = EXP_DIR / "vae_jacobian_spectra.npz"
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}. Run without --plot-only first.")
        print(f"Plot-only mode: loading from {csv_path}, {spectra_path}")
        results = load_csv(csv_path)
        cross_val = load_crossval_csv(cv_path) if cv_path.exists() else []
        spectra_loaded = np.load(spectra_path, allow_pickle=True) if spectra_path.exists() else {}
        spectra = {}
        for key_str, arr in spectra_loaded.items():
            parts = key_str.split("_", 1)
            res = int(parts[0])
            itype = parts[1]
            spectra[(res, itype)] = arr
        plot_results(results, spectra, cross_val)
        print("\n✓ Plots regenerated.")
        raise SystemExit(0)

    vae = load_frozen_vae()
    seed = args.seed

    # Hutchinson-vs-exact alignment seed controls the white-noise probe used
    # in both main table and cross-validation.

    # Hutchinson vs Exact cross-validation at 128×128
    cross_val = hutchinson_vs_exact_check(vae, seed=seed)

    # Main experiment
    results, spectra = run_experiment(
        vae,
        resolutions=[64, 128, 256],
        hutchinson_resolutions=[512],
        num_probes_hutchinson=50,
        save_spectra=True,
        seed=seed,
    )

    # Save outputs
    save_csv(results, cross_val)
    # Save spectra for --plot-only reuse
    spectra_save = {f"{res}_{itype}": arr for (res, itype), arr in spectra.items()}
    np.savez(EXP_DIR / "vae_jacobian_spectra.npz", **spectra_save)
    print(f"Spectra saved: {EXP_DIR / 'vae_jacobian_spectra.npz'}")
    print_appendix_table(results)
    plot_results(results, spectra, cross_val)

    print("\n✓ Experiment complete.")
