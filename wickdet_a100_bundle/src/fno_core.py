from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_DIR = Path(__file__).resolve().parent

# Keep matplotlib writes working in sandboxed environments with read-only caches.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gaussian-hilbert-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gaussian-hilbert-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from wickdet import schatten_cf_penalty


DEFAULT_BASELINE_ORDER = (
    "ordinary_noise",
    "cf_unfiltered",
    "cf_schatten_filtered",
)


@dataclass(frozen=True)
class ExperimentConfig:
    resolutions: tuple[int, ...] = (64, 128, 256)
    include_512: bool = False
    train_samples: int = 64
    val_samples: int = 16
    epochs: int = 20
    batch_size: int = 4
    width: int = 24
    learning_rate: float = 2.0e-3
    weight_decay: float = 0.0
    seed: int = 2026
    max_beta: float = 0.45
    alpha_filter: float = 1.25
    noise_lr_penalty_weight: float = 1.0e-3
    jacobi_iterations: int = 150
    jacobi_relaxation: float = 0.8
    modes_divisor: int = 4
    output_csv: Path = EXPERIMENT_DIR / "fno_resolution_scaling_results.csv"
    output_timeseries_png: Path = EXPERIMENT_DIR / "fno_resolution_scaling_timeseries.png"
    output_pareto_png: Path = EXPERIMENT_DIR / "fno_resolution_scaling_pareto.png"
    include_scalar_schedule: bool = False
    device: str = "auto"
    seed_repeats: int = 3
    resolution_subset: tuple[int, ...] = (128, 256)
    t2_norm_mode: str = "per_mode"
    pareto_x_metric: str = "t2_norm"

    @property
    def baseline_order(self) -> tuple[str, ...]:
        if self.include_scalar_schedule:
            return DEFAULT_BASELINE_ORDER + ("scalar_schedule",)
        return DEFAULT_BASELINE_ORDER

    @property
    def scalar_schedule_reference_resolution(self) -> int:
        return max(self.resolutions)


@dataclass(frozen=True)
class ResolutionMeta:
    resolution: int
    modes: int
    resolution_seed: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def parse_resolutions(raw: str, include_512: bool) -> tuple[int, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"resolution must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one resolution must be specified")
    values = sorted(set(values))
    if include_512 and 512 not in values:
        values.append(512)
        values = sorted(set(values))
    return tuple(values)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def coordinate_channels(resolution: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, resolution + 1, device=device, dtype=torch.float32)[:-1]
    y = torch.linspace(0.0, 1.0, resolution + 1, device=device, dtype=torch.float32)[:-1]
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0)


def radial_mask_2d(resolution: int, alpha: float, *, device: torch.device) -> torch.Tensor:
    fx = torch.fft.fftfreq(resolution, d=1.0, device=device)
    fy = torch.fft.rfftfreq(resolution, d=1.0, device=device)
    kx, ky = torch.meshgrid(fx, fy, indexing="ij")
    radius = torch.sqrt(kx * kx + ky * ky)
    rmax = float(radius.max().item()) if float(radius.max().item()) > 0 else 1.0
    return 1.0 / (1.0 + (radius / rmax).pow(alpha))


def sample_lowpass_logk(
    resolution: int,
    generator: torch.Generator,
    *,
    device: torch.device,
    power: float = 3.0,
) -> torch.Tensor:
    ky = torch.fft.rfftfreq(resolution, d=1.0, device=device)
    kx = torch.fft.fftfreq(resolution, d=1.0, device=device)
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
    freq_radius = torch.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)

    decay = 1.0 / (1.0 + (freq_radius * resolution).pow(power) + 1e-6)
    mask = torch.exp(-((freq_radius * resolution / (0.25 * resolution)) ** 2))
    coeff_real = torch.randn((len(ky), resolution), generator=generator, device=device)
    coeff_imag = torch.randn((len(ky), resolution), generator=generator, device=device)
    coeff = torch.complex(coeff_real, coeff_imag) * decay * mask
    coeff[0, 0] = coeff[0, 0].real + 0j
    if resolution % 2 == 0:
        coeff[0, -1] = coeff[0, -1].real + 0j
    logk = torch.fft.irfft2(coeff, s=(resolution, resolution), dim=(-2, -1), norm="ortho")
    logk = logk - logk.mean()
    logk = logk / (logk.std() + 1e-6)
    return logk


def deterministic_rhs(resolution: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, resolution, device=device)
    y = torch.linspace(0.0, 1.0, resolution, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rhs = torch.sin(2.0 * torch.pi * xx) * torch.cos(2.0 * torch.pi * yy)
    rhs = rhs - rhs.mean()
    return rhs


def solve_periodic_darcy(
    logk: torch.Tensor,
    rhs: torch.Tensor,
    *,
    iterations: int,
    relaxation: float,
) -> torch.Tensor:
    # Solve
    #   - div( exp(logk) grad u ) = rhs
    # with periodic wrap via a simple damped Jacobi-like iteration.
    a = torch.exp(logk)
    u = torch.zeros_like(rhs)
    h2 = 1.0 / (rhs.shape[-1] * rhs.shape[-1])

    for _ in range(iterations):
        a_e = 0.5 * (a + torch.roll(a, shifts=(-1,), dims=-1))
        a_w = 0.5 * (a + torch.roll(a, shifts=(1,), dims=-1))
        a_n = 0.5 * (a + torch.roll(a, shifts=(-1,), dims=-2))
        a_s = 0.5 * (a + torch.roll(a, shifts=(1,), dims=-2))

        u_e = torch.roll(u, shifts=(-1,), dims=-1)
        u_w = torch.roll(u, shifts=(1,), dims=-1)
        u_n = torch.roll(u, shifts=(-1,), dims=-2)
        u_s = torch.roll(u, shifts=(1,), dims=-2)

        denom = a_e + a_w + a_n + a_s + 1e-8
        u_new = (a_e * u_e + a_w * u_w + a_n * u_n + a_s * u_s - rhs * h2) / denom
        u = relaxation * u_new + (1.0 - relaxation) * u

    return u - u.mean()


def build_dataset(
    resolution: int,
    seed: int,
    *,
    train_samples: int,
    val_samples: int,
    jacobi_iterations: int,
    jacobi_relaxation: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = coordinate_channels(resolution, device)
    rhs = deterministic_rhs(resolution, device)

    total = train_samples + val_samples

    logk_all = []
    u_all = []
    with torch.no_grad():
        for idx in range(total):
            sample_seed = seed + idx * 9973
            local_gen = torch.Generator(device=device)
            local_gen.manual_seed(sample_seed)
            logk = sample_lowpass_logk(
                resolution,
                local_gen,
                device=device,
            )
            u = solve_periodic_darcy(
                logk,
                rhs,
                iterations=jacobi_iterations,
                relaxation=jacobi_relaxation,
            )
            logk_all.append(logk)
            u_all.append(u)

    logk_all = torch.stack(logk_all)
    u_all = torch.stack(u_all)

    coord = coords.unsqueeze(0).expand(total, -1, -1, -1)
    X = torch.cat((logk_all.unsqueeze(1), coord), dim=1)
    y = u_all.unsqueeze(1)

    train_X = X[:train_samples]
    train_y = y[:train_samples]
    val_X = X[train_samples:]
    val_y = y[train_samples:]
    return train_X.float(), train_y.float(), val_X.float(), val_y.float()


def t2_multiplier_matrix(resolution: int, device: torch.device) -> torch.Tensor:
    mult_x = torch.full((resolution,), 2.0, device=device)
    mult_x[0] = 1.0
    if resolution % 2 == 0:
        mult_x[resolution // 2] = 1.0

    mult_y = torch.full((resolution // 2 + 1,), 2.0, device=device)
    mult_y[0] = 1.0
    if resolution % 2 == 0:
        mult_y[-1] = 1.0

    return mult_x[:, None] * mult_y[None, :]


def t2_continuum_weights(resolution: int, device: torch.device) -> torch.Tensor:
    mult = t2_multiplier_matrix(resolution, device)
    # Frequency domain grid for rfft2 coordinates (requested form).
    kx = 2.0 * torch.pi * torch.fft.fftfreq(resolution, d=1.0, device=device)
    ky = 2.0 * torch.pi * torch.fft.rfftfreq(resolution, d=1.0, device=device)
    _ = torch.meshgrid(kx, ky, indexing="ij")
    return mult / float(resolution * resolution)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        scale = 0.02
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        self.weight1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            (batch, self.out_ch, h, w // 2 + 1),
            device=x.device,
            dtype=torch.cfloat,
        )
        upper = slice(0, self.modes1)
        lower = slice(-self.modes1, None)
        right = slice(0, self.modes2)

        out_ft[:, :, upper, right] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, upper, right], self.weight1
        )
        out_ft[:, :, lower, right] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, lower, right], self.weight2
        )

        return torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1), norm="ortho")


class FNOLayer(nn.Module):
    def __init__(self, channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FNO2D(nn.Module):
    def __init__(self, modes: int, width: int, num_layers: int = 4) -> None:
        super().__init__()
        # channels: permeability + (x,y) coordinates
        self.lift = nn.Conv2d(3, width, kernel_size=1)
        self.layers = nn.ModuleList(
            [FNOLayer(width, modes, modes) for _ in range(num_layers)]
        )
        self.proj = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.lift(x))
        for layer in self.layers:
            out = out + layer(out)
        return self.proj(out)


class LearnableSpectralNoise(nn.Module):
    def __init__(
        self,
        resolution: int,
        width: int,
        *,
        max_beta: float,
        alpha_filter: float | None,
        apply_filter: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.width = width
        self.max_beta = max_beta
        self.apply_filter = apply_filter
        self.modes2 = resolution // 2 + 1

        base_scale = 0.05
        self.logits = nn.Parameter(
            torch.full((1, resolution, self.modes2), torch.tensor(base_scale).logit(), device=device)
        )

        self.register_buffer(
            "base_filter",
            radial_mask_2d(resolution, alpha_filter or 1.0, device=device)
            if apply_filter and alpha_filter is not None
            else torch.ones((resolution, self.modes2), device=device),
        )
        self.register_buffer("mult", t2_multiplier_matrix(resolution, device))

    def eigenvalues(self) -> torch.Tensor:
        raw = self.max_beta * torch.sigmoid(self.logits)
        return raw * self.base_filter

    def operator(self, noise: torch.Tensor) -> torch.Tensor:
        beta = self.eigenvalues()
        # broadcast over width and batch; beta depends only on frequency
        beta = beta.view(1, 1, self.resolution, self.modes2)
        z_ft = torch.fft.rfft2(noise, norm="ortho")
        return torch.fft.irfft2(z_ft * beta, s=(self.resolution, self.resolution), norm="ortho")

    def t2_exact(self) -> float:
        beta = self.eigenvalues().to(torch.float64)
        mult = self.mult.to(torch.float64)
        t2 = (beta.square() * mult).sum() * float(self.width)
        return t2.item()

    def t2_discrete(self) -> float:
        return self.t2_exact()

    def t2_continuum(self) -> float:
        beta = self.eigenvalues().to(torch.float64)
        weights = t2_continuum_weights(self.resolution, beta.device).to(torch.float64)
        return (beta.square() * weights).sum().item()

    def t2_normalized(self, mode: str) -> float:
        if mode == "none":
            return float("nan")
        if mode == "per_mode":
            return self.t2_discrete() / (float(self.resolution * self.resolution) * float(self.width))
        if mode == "continuum":
            return self.t2_continuum()
        raise ValueError(f"unknown t2 norm mode: {mode}")

    def summary(self) -> dict[str, float]:
        beta = self.eigenvalues().detach().float().view(-1)
        return {
            "beta_mean": float(beta.mean().item()),
            "beta_std": float(beta.std(unbiased=False).item()),
            "beta_min": float(beta.min().item()),
            "beta_max": float(beta.max().item()),
        }


class OrdinaryGaussianNoise(nn.Module):
    def __init__(self, resolution: int, width: int, *, init_sigma: float = 0.05) -> None:
        super().__init__()
        self.resolution = resolution
        self.width = width
        self.log_scale = nn.Parameter(torch.log(torch.tensor(init_sigma)))

    def sample(self, features: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_scale)
        return torch.randn_like(features) * sigma

    def t2_exact(self) -> float:
        sigma = torch.exp(self.log_scale)
        return float(self.width * (self.resolution**2) * sigma.item() ** 2)

    def t2_discrete(self) -> float:
        return self.t2_exact()

    def t2_continuum(self) -> float:
        sigma = torch.exp(self.log_scale)
        return sigma.item() ** 2

    def t2_normalized(self, mode: str) -> float:
        if mode == "none":
            return float("nan")
        if mode == "per_mode":
            return self.t2_discrete() / (float(self.resolution * self.resolution) * float(self.width))
        if mode == "continuum":
            return self.t2_continuum()
        raise ValueError(f"unknown t2 norm mode: {mode}")

    def summary(self) -> dict[str, float]:
        sigma = torch.exp(self.log_scale).item()
        return {
            "sigma": float(sigma),
            "sigma2": float(sigma * sigma),
            "t2": self.t2_exact(),
        }


class ScalarScheduleNoise(OrdinaryGaussianNoise):
    def __init__(
        self,
        resolution: int,
        width: int,
        *,
        base_resolution: int,
        init_sigma: float = 0.07,
        decay_power: float = 1.0,
    ) -> None:
        super().__init__(resolution, width, init_sigma=init_sigma)
        self.base_resolution = float(base_resolution)
        self.decay_power = decay_power

    def sample(self, features: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_scale)
        sigma = sigma * (torch.tensor(self.base_resolution / float(self.resolution), device=features.device) ** self.decay_power)
        return torch.randn_like(features) * sigma

    def t2_exact(self) -> float:
        sigma = torch.exp(self.log_scale) * (self.base_resolution / float(self.resolution)) ** self.decay_power
        return float(self.width * (self.resolution**2) * sigma.item() ** 2)

    def t2_discrete(self) -> float:
        return self.t2_exact()

    def t2_continuum(self) -> float:
        sigma = torch.exp(self.log_scale) * (self.base_resolution / float(self.resolution)) ** self.decay_power
        return sigma.item() ** 2

    def t2_normalized(self, mode: str) -> float:
        if mode == "none":
            return float("nan")
        if mode == "per_mode":
            return self.t2_discrete() / (float(self.resolution * self.resolution) * float(self.width))
        if mode == "continuum":
            return self.t2_continuum()
        raise ValueError(f"unknown t2 norm mode: {mode}")


class NoiseWrapper(nn.Module):
    def __init__(
        self,
        baseline: str,
        resolution: int,
        width: int,
        *,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.resolution = resolution
        self.width = width
        self.config = config

        if baseline == "ordinary_noise":
            self.layer = OrdinaryGaussianNoise(resolution, width).to(device)
            self.use_cf = False
        elif baseline == "cf_unfiltered":
            self.layer = LearnableSpectralNoise(
                resolution=resolution,
                width=width,
                max_beta=config.max_beta,
                alpha_filter=None,
                apply_filter=False,
                device=device,
            )
            self.use_cf = True
        elif baseline == "cf_schatten_filtered":
            self.layer = LearnableSpectralNoise(
                resolution=resolution,
                width=width,
                max_beta=config.max_beta,
                alpha_filter=config.alpha_filter,
                apply_filter=True,
                device=device,
            )
            self.use_cf = True
        elif baseline == "scalar_schedule":
            self.layer = ScalarScheduleNoise(
                resolution=resolution,
                width=width,
                base_resolution=config.scalar_schedule_reference_resolution,
                init_sigma=0.07,
                decay_power=0.75,
            )
            self.use_cf = False
        else:
            raise ValueError(f"unknown baseline: {baseline}")

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.layer, (OrdinaryGaussianNoise, ScalarScheduleNoise)):
            perturb = self.layer.sample(features)
            return perturb, torch.tensor(0.0, device=features.device)

        perturb_input = torch.randn_like(features)
        if self.use_cf:
            perturb = self.layer.operator(perturb_input)
            cf_penalty, _ = schatten_cf_penalty(
                self.layer.operator,
                perturb_input,
                K=6,
                num_probes=4,
                mode="penalty",
                return_diagnostics=True,
            )
            return perturb, cf_penalty

        perturb = self.layer.operator(perturb_input)
        return perturb, torch.tensor(0.0, device=features.device)

    def t2_exact(self) -> float:
        if hasattr(self.layer, "t2_exact"):
            return float(self.layer.t2_exact())
        return 0.0

    def t2_discrete(self) -> float:
        if hasattr(self.layer, "t2_discrete"):
            return float(self.layer.t2_discrete())
        return 0.0

    def t2_continuum(self) -> float:
        if hasattr(self.layer, "t2_continuum"):
            return float(self.layer.t2_continuum())
        return 0.0

    def t2_norm(self, mode: str) -> float:
        if hasattr(self.layer, "t2_normalized"):
            return float(self.layer.t2_normalized(mode))
        return float("nan")

    def summary(self) -> dict[str, float]:
        if isinstance(self.layer, (LearnableSpectralNoise,)):
            out = dict(self.layer.summary())
            out["t2"] = float(self.t2_exact())
            return out
        return self.layer.summary()  # type: ignore[union-attr]


class FNOWithNoise(nn.Module):
    def __init__(
        self,
        resolution: int,
        width: int,
        *,
        modes: int,
        baseline: str,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.width = width
        self.baseline = baseline
        self.backbone = FNO2D(modes=modes, width=width)
        self.noise = NoiseWrapper(
            baseline=baseline,
            resolution=resolution,
            width=width,
            config=config,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.backbone(x)
        perturb, penalty = self.noise(latent)
        latent = latent + perturb
        pred = latent
        return pred, penalty


def row_metrics(
    epoch: int,
    resolution: int,
    baseline: str,
    train_mse: float,
    val_relative_l2: float,
    grad_norm: float,
    t2_exact: float,
    t2_discrete: float,
    t2_norm: float,
    t2_continuum: float,
    seed: int,
    hfreq_ratio: float,
    noise_summary: dict[str, float],
) -> dict[str, float | int | str]:
    return {
        "epoch": epoch,
        "resolution": resolution,
        "baseline": baseline,
        "seed": seed,
        "train_mse": train_mse,
        "val_relative_l2": val_relative_l2,
        "grad_norm": grad_norm,
        "t2_exact": t2_exact,
        "t2_discrete": t2_discrete,
        "t2_norm": t2_norm,
        "t2_continuum": t2_continuum,
        "high_frequency_energy_ratio": hfreq_ratio,
        "noise_amplitude_summary": json.dumps(noise_summary, separators=(",", ":")),
    }


def high_frequency_ratio(u: torch.Tensor, *, fraction: float = 0.58) -> float:
    # mean_{batch}(energy_high / energy_total) on a single field.
    # u: (B, 1, H, W)
    if u.ndim != 4:
        raise ValueError("u must have shape (B, C, H, W)")

    h = u.shape[-2]
    w = u.shape[-1]
    freq = torch.fft.rfftn(u, dim=(-2, -1), norm="ortho")
    power = torch.abs(freq).square()

    kx = torch.fft.fftfreq(h, d=1.0, device=u.device)
    ky = torch.fft.rfftfreq(w, d=1.0, device=u.device)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    radius = torch.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)
    cutoff = fraction * float(radius.max())

    mask = (radius >= cutoff).to(u.dtype)
    mask = mask[None, None, ...]

    high = (power * mask).sum(dim=(1, 2, 3))
    total = power.sum(dim=(1, 2, 3))
    ratio = (high / (total + 1e-12)).mean().item()
    return float(ratio)


def grad_norm(model: nn.Module) -> float:
    total = torch.tensor(0.0)
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.data
        if torch.is_complex(g):
            total = total + g.real.float().pow(2).sum() + g.imag.float().pow(2).sum()
        else:
            total = total + g.float().pow(2).sum()
    return torch.sqrt(total).item()


def run_baseline(
    resolution: int,
    baseline: str,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> list[dict[str, float | int | str]]:
    set_seed(seed)
    modes = max(2, min(config.modes_divisor and (resolution // config.modes_divisor) or resolution, 16))

    model = FNOWithNoise(
        resolution=resolution,
        width=config.width,
        modes=modes,
        baseline=baseline,
        config=config,
        device=device,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_inputs, train_targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 101),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_inputs, val_targets),
        batch_size=config.batch_size,
        shuffle=False,
    )

    rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        grad_norm_total = 0.0
        train_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred, penalty = model(x_batch)
            mse = F.mse_loss(pred, y_batch)
            loss = mse + config.noise_lr_penalty_weight * penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += mse.item()
            grad_norm_total += grad_norm(model)
            train_batches += 1

        train_loss /= max(train_batches, 1)
        epoch_grad_norm = grad_norm_total / max(train_batches, 1)

        model.eval()
        with torch.no_grad():
            val_rel = 0.0
            val_batches = 0
            val_high_ratio = 0.0
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                pred, _ = model(x_val)
                mse_num = ((pred - y_val) ** 2).sum(dim=(1, 2, 3)).sqrt()
                denom = (y_val ** 2).sum(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
                val_rel += (mse_num / denom).mean().item()
                val_high_ratio += high_frequency_ratio(pred)
                val_batches += 1

            val_rel /= max(val_batches, 1)
            val_high_ratio /= max(val_batches, 1)

        noise_summary = model.noise.summary()
        current_t2 = model.noise.t2_exact()
        current_t2_discrete = model.noise.t2_discrete()
        current_t2_continuum = model.noise.t2_continuum()
        current_t2_norm = model.noise.t2_norm(config.t2_norm_mode)

        rows.append(
            row_metrics(
                epoch=epoch,
                resolution=resolution,
                baseline=baseline,
                train_mse=train_loss,
                val_relative_l2=val_rel,
                grad_norm=epoch_grad_norm,
                t2_exact=current_t2,
                t2_discrete=current_t2_discrete,
                t2_norm=current_t2_norm,
                t2_continuum=current_t2_continuum,
                seed=seed,
                hfreq_ratio=float(val_high_ratio),
                noise_summary=noise_summary,
            )
        )

    return rows


def plot_timeseries(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
    *,
    aggregate_resolutions: tuple[int, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    baselines = sorted({str(row["baseline"]) for row in rows})
    resolutions = sorted({int(row["resolution"]) for row in rows})

    metric_names = [
        "val_relative_l2",
        "train_mse",
        "grad_norm",
        "t2_norm",
        "t2_continuum",
        "high_frequency_energy_ratio",
    ]
    metric_titles = [
        "Validation relative L2",
        "Training loss (MSE)",
        "Gradient norm",
        r"$T_2^{\mathrm{norm}}$",
        r"$T_2^{\mathrm{cont}}$",
        "High-frequency energy ratio",
    ]

    fig, axes = plt.subplots(
        len(metric_names),
        len(resolutions),
        figsize=(4.5 * len(resolutions), 3.0 * len(metric_names)),
        squeeze=False,
    )

    palette = {
        "ordinary_noise": "#2a9d8f",
        "cf_unfiltered": "#f4a261",
        "cf_schatten_filtered": "#264653",
        "scalar_schedule": "#e76f51",
    }

    for res_idx, res in enumerate(resolutions):
        res_rows = [row for row in rows if int(row["resolution"]) == res]
        aggregate = res in aggregate_resolutions
        for baseline in baselines:
            br = [row for row in res_rows if str(row["baseline"]) == baseline]
            if not br:
                continue
            br = sorted(br, key=lambda row: int(row["epoch"]))
            color = palette.get(baseline, None)
            by_epoch: dict[int, dict[str, list[float]]] = {}
            for row in br:
                epoch = int(row["epoch"])
                entry = by_epoch.setdefault(
                    epoch,
                    {"val_relative_l2": [], "train_mse": [], "grad_norm": [], "t2_norm": [], "t2_continuum": [], "high_frequency_energy_ratio": []},
                )
                for metric in metric_names:
                    entry[metric].append(float(row[metric]))

            epochs = sorted(by_epoch.keys())
            if not epochs:
                continue

            for metric_idx, metric in enumerate(metric_names):
                y = [float(torch.tensor(by_epoch[e][metric]).mean().item()) for e in epochs]
                y_std = [float(torch.tensor(by_epoch[e][metric]).std(unbiased=False).item()) for e in epochs]
                axes[metric_idx][res_idx].plot(
                    epochs,
                    y,
                    marker="o",
                    linewidth=1.6,
                    label=baseline,
                    color=color,
                )
                if aggregate:
                    y_lower = [v - s for v, s in zip(y, y_std)]
                    y_upper = [v + s for v, s in zip(y, y_std)]
                    axes[metric_idx][res_idx].fill_between(
                        epochs,
                        y_lower,
                        y_upper,
                        alpha=0.18,
                        color=color,
                    )
                axes[metric_idx][res_idx].set_title(f"res={res}")
                axes[metric_idx][res_idx].set_xlabel("epoch")
                axes[metric_idx][res_idx].set_ylabel(metric_titles[metric_idx])

    for metric_idx in range(len(metric_names)):
        for res_idx in range(len(resolutions)):
            axes[metric_idx][res_idx].legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pareto(
    rows: list[dict[str, float | int | str]],
    out_path: Path,
    *,
    x_metric: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    baselines = sorted({str(row["baseline"]) for row in rows})
    resolutions = sorted({int(row["resolution"]) for row in rows})

    palette = {
        "ordinary_noise": "#2a9d8f",
        "cf_unfiltered": "#f4a261",
        "cf_schatten_filtered": "#264653",
        "scalar_schedule": "#e76f51",
    }

    fig, axes = plt.subplots(1, len(resolutions), figsize=(4.8 * len(resolutions), 4.5), squeeze=False)
    axes = axes[0]

    for res_idx, res in enumerate(resolutions):
        ax = axes[res_idx]
        sub = [row for row in rows if int(row["resolution"]) == res]

        for baseline in baselines:
            br = sorted([r for r in sub if str(r["baseline"]) == baseline], key=lambda r: int(r["epoch"]))
            if not br:
                continue
            by_epoch: dict[int, dict[str, list[float]]] = {}
            for row in br:
                epoch = int(row["epoch"])
                entry = by_epoch.setdefault(epoch, {"x": [], "y": []})
                entry["x"].append(float(row[x_metric]))
                entry["y"].append(float(row["val_relative_l2"]))

            epochs = sorted(by_epoch.keys())
            if not epochs:
                continue

            x: list[float] = []
            x_std: list[float] = []
            y: list[float] = []
            y_std: list[float] = []
            for e in epochs:
                x_values = torch.tensor(by_epoch[e]["x"], dtype=torch.float64)
                y_values = torch.tensor(by_epoch[e]["y"], dtype=torch.float64)
                x.append(float(x_values.mean().item()))
                y.append(float(y_values.mean().item()))
                x_std.append(float(x_values.std(unbiased=False).item()))
                y_std.append(float(y_values.std(unbiased=False).item()))
            color = palette.get(baseline, None)

            ax.plot(x, y, marker="o", linewidth=1.6, label=baseline, color=color)
            ax.errorbar(
                x,
                y,
                xerr=x_std,
                yerr=y_std,
                fmt="none",
                ecolor=color,
                alpha=0.25,
            )
            ax.errorbar(
                [x[0]],
                [y[0]],
                xerr=[[x_std[0]], [x_std[0]]],
                yerr=[[y_std[0]], [y_std[0]]],
                fmt="o",
                markerfacecolor="none",
                color=color,
            )
            ax.errorbar(
                [x[-1]],
                [y[-1]],
                xerr=[[x_std[-1]], [x_std[-1]]],
                yerr=[[y_std[-1]], [y_std[-1]]],
                fmt="o",
                color=color,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        if x_metric == "t2_norm":
            ax.set_xlabel(r"$T_2^{\mathrm{norm}}$")
        elif x_metric == "t2_continuum":
            ax.set_xlabel(r"$T_2^{\mathrm{cont}}$")
        else:
            ax.set_xlabel(r"$T_2$")
        ax.set_ylabel("Validation relative L2")
        ax.set_title(f"res={res}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_rows(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(
    rows: list[dict[str, float | int | str]],
    required_resolutions: Sequence[int],
    required_baselines: Sequence[str],
    seed_repeats: int,
) -> None:
    if not rows:
        raise RuntimeError("no rows to validate")

    metric_names = {
        "resolution",
        "baseline",
        "train_mse",
        "val_relative_l2",
        "grad_norm",
        "t2_exact",
        "t2_discrete",
        "t2_norm",
        "t2_continuum",
        "high_frequency_energy_ratio",
        "noise_amplitude_summary",
        "seed",
    }
    missing = metric_names - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"result rows missing required metrics: {sorted(missing)}")

    for res in required_resolutions:
        for baseline in required_baselines:
            filtered = [
                row
                for row in rows
                if int(row["resolution"]) == int(res)
                and str(row["baseline"]) == baseline
            ]
            if not filtered:
                raise RuntimeError(
                    f"missing rows for resolution={res}, baseline={baseline}"
                )
            seeds = sorted({int(row["seed"]) for row in filtered})
            if len(seeds) != seed_repeats:
                raise RuntimeError(
                    f"expected {seed_repeats} seeds for resolution={res}, baseline={baseline}, got {len(seeds)}"
                )
            for row in filtered:
                ratio = float(row["high_frequency_energy_ratio"])
                if ratio < 0.0 or not torch.isfinite(torch.tensor(ratio)).item():
                    raise RuntimeError(
                        f"invalid high-frequency ratio for {res} {baseline} epoch {row['epoch']}"
                    )

    final_t2_unfiltered = []
    final_t2_filtered = []
    for res in sorted(required_resolutions):
        unfiltered_rows = [
            row for row in rows if int(row["resolution"]) == int(res) and str(row["baseline"]) == "cf_unfiltered"
        ]
        filtered_rows = [
            row for row in rows if int(row["resolution"]) == int(res) and str(row["baseline"]) == "cf_schatten_filtered"
        ]
        if unfiltered_rows and filtered_rows:
            last_unfiltered_epoch = max(int(row["epoch"]) for row in unfiltered_rows)
            last_filtered_epoch = max(int(row["epoch"]) for row in filtered_rows)
            final_t2_unfiltered.append(
                (
                    res,
                    float(
                        torch.tensor(
                            [
                                float(r["t2_exact"])
                                for r in unfiltered_rows
                                if int(r["epoch"]) == last_unfiltered_epoch
                            ],
                            dtype=torch.float64,
                        ).mean().item()
                    ),
                )
            )
            final_t2_filtered.append(
                (
                    res,
                    float(
                        torch.tensor(
                            [
                                float(r["t2_exact"])
                                for r in filtered_rows
                                if int(r["epoch"]) == last_filtered_epoch
                            ],
                            dtype=torch.float64,
                        ).mean().item()
                    ),
                )
            )

    if len(final_t2_unfiltered) >= 2 and len(final_t2_filtered) >= 2:
        r0, t2u0 = final_t2_unfiltered[0]
        r_last, t2u_last = final_t2_unfiltered[-1]
        _, t2f_last = final_t2_filtered[-1]
        growth_unfiltered = t2u_last / (t2u0 + 1e-12)
        growth_filtered = t2f_last / (final_t2_filtered[0][1] + 1e-12)
        if growth_filtered > growth_unfiltered:
            raise RuntimeError(
                "cf_schatten_filtered does not show lower T2 growth than cf_unfiltered "
                f"(growth_filtered={growth_filtered:.3f}, growth_unfiltered={growth_unfiltered:.3f})"
            )


def run_experiment(config: ExperimentConfig) -> list[dict[str, float | int | str]]:
    device = choose_device(config.device)
    all_rows: list[dict[str, float | int | str]] = []

    for resolution in config.resolutions:
        set_seed(config.seed + resolution)
        model_seed = config.seed + resolution * 1000
        train_X, train_y, val_X, val_y = build_dataset(
            resolution,
            seed=config.seed + resolution * 31,
            train_samples=config.train_samples,
            val_samples=config.val_samples,
            jacobi_iterations=config.jacobi_iterations,
            jacobi_relaxation=config.jacobi_relaxation,
            device=device,
        )

        train_X = train_X.to(device)
        train_y = train_y.to(device)
        val_X = val_X.to(device)
        val_y = val_y.to(device)

        for baseline in config.baseline_order:
            for seed_idx in range(config.seed_repeats):
                run_seed = model_seed + seed_idx
                print(
                    f"resolution={resolution}, baseline={baseline}, repeat={seed_idx}, seed={run_seed}, starting..."
                )
                baseline_rows = run_baseline(
                    resolution=resolution,
                    baseline=baseline,
                    train_inputs=train_X,
                    train_targets=train_y,
                    val_inputs=val_X,
                    val_targets=val_y,
                    config=config,
                    device=device,
                    seed=run_seed,
                )
                all_rows.extend(baseline_rows)

    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FNO resolution scaling experiment for 2D Darcy operator."
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default="64,128,256",
        help="Comma-separated list of resolutions.",
    )
    parser.add_argument(
        "--include-512",
        action="store_true",
        help="Add 512 to the resolution list.",
    )
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--output-csv", type=Path, default=ExperimentConfig().output_csv)
    parser.add_argument("--output-timeseries-png", type=Path, default=ExperimentConfig().output_timeseries_png)
    parser.add_argument("--output-pareto-png", type=Path, default=ExperimentConfig().output_pareto_png)
    parser.add_argument(
        "--seed-repeats",
        type=int,
        default=3,
        help="How many random seeds per resolution/baseline.",
    )
    parser.add_argument(
        "--t2-norm-mode",
        type=str,
        default="per_mode",
        choices=("per_mode", "continuum", "none"),
        help="How to normalize t2_norm for the output CSV and plots.",
    )
    parser.add_argument(
        "--aggregate-res",
        nargs="+",
        type=int,
        default=[128, 256],
        help="Resolutions used for mean ± std aggregation plots.",
    )
    parser.add_argument(
        "--pareto-x",
        type=str,
        default="t2_norm",
        choices=("t2_discrete", "t2_norm", "t2_continuum"),
        help="Metric used on x-axis of pareto plot.",
    )
    parser.add_argument("--include-scalar-schedule", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def build_config_from_args() -> ExperimentConfig:
    args = parse_args()

    if args.smoke:
        cfg = ExperimentConfig(
            resolutions=(32, 64),
            train_samples=8,
            val_samples=4,
            epochs=2,
            seed_repeats=args.seed_repeats,
            output_csv=EXPERIMENT_DIR / "fno_resolution_scaling_results.csv",
            output_timeseries_png=EXPERIMENT_DIR / "fno_resolution_scaling_timeseries.png",
            output_pareto_png=EXPERIMENT_DIR / "fno_resolution_scaling_pareto.png",
            resolution_subset=tuple(args.aggregate_res),
            t2_norm_mode=args.t2_norm_mode,
            pareto_x_metric=args.pareto_x,
            include_scalar_schedule=args.include_scalar_schedule,
            device=args.device,
        )
        return cfg

    return ExperimentConfig(
        resolutions=parse_resolutions(args.resolutions, include_512=args.include_512),
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        width=args.width,
        learning_rate=args.lr,
        seed_repeats=args.seed_repeats,
        resolution_subset=tuple(args.aggregate_res),
        t2_norm_mode=args.t2_norm_mode,
        pareto_x_metric=args.pareto_x,
        output_csv=args.output_csv,
        output_timeseries_png=args.output_timeseries_png,
        output_pareto_png=args.output_pareto_png,
        include_scalar_schedule=args.include_scalar_schedule,
        device=args.device,
    )


def main() -> None:
    config = build_config_from_args()
    all_rows = run_experiment(config)
    save_rows(all_rows, config.output_csv)
    plot_timeseries(
        all_rows,
        config.output_timeseries_png,
        aggregate_resolutions=config.resolution_subset,
    )
    plot_pareto(
        all_rows,
        config.output_pareto_png,
        x_metric=config.pareto_x_metric,
    )

    validate_rows(
        all_rows,
        config.resolutions,
        config.baseline_order,
        config.seed_repeats,
    )

    print(f"Wrote {config.output_csv}")
    print(f"Wrote {config.output_timeseries_png}")
    print(f"Wrote {config.output_pareto_png}")


if __name__ == "__main__":
    main()
