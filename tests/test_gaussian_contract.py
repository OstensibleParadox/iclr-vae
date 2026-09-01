from __future__ import annotations

import importlib.util
import unittest
import warnings
from pathlib import Path

import torch

import wickdet


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_MODULE_PATH = ROOT / "wickdet_a100_bundle" / "src" / "wickdet.py"


def _load_bundle_module():
    spec = importlib.util.spec_from_file_location("a100_bundle_wickdet", BUNDLE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BUNDLE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GaussianContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = torch.float64
        self.amplitude_b = torch.tensor(
            [
                [0.45, 0.00],
                [0.10, 0.35],
                [-0.20, 0.25],
            ],
            dtype=self.dtype,
        )

    def test_a100_copy_exports_the_same_contract(self) -> None:
        bundle = _load_bundle_module()
        public_contract = (
            "covariance_from_amplitude",
            "covariance_t2",
            "kl_q_to_p_from_covariance",
            "precision_from_covariance",
            "matrix_free_precision_wick_kl",
        )
        for name in public_contract:
            with self.subTest(name=name):
                self.assertTrue(hasattr(bundle, name))

        root_source = (ROOT / "wickdet.py").read_text(encoding="utf-8")
        bundle_source = BUNDLE_MODULE_PATH.read_text(encoding="utf-8")
        self.assertEqual(root_source, bundle_source)

    def test_amplitude_samples_have_covariance_b_b_star(self) -> None:
        covariance_k = wickdet.covariance_from_amplitude(self.amplitude_b)
        expected_k = self.amplitude_b @ self.amplitude_b.T
        torch.testing.assert_close(covariance_k, expected_k, rtol=0.0, atol=0.0)

        generator = torch.Generator().manual_seed(31415)
        perturbations = wickdet.sample_covariance_perturbation(
            self.amplitude_b,
            100_000,
            generator=generator,
        )
        empirical_k = perturbations.T @ perturbations / perturbations.shape[0]
        torch.testing.assert_close(empirical_k, covariance_k, rtol=0.02, atol=0.001)

    def test_t2_det2_and_both_kl_directions_are_exact(self) -> None:
        covariance_k = wickdet.covariance_from_amplitude(self.amplitude_b)
        identity = torch.eye(covariance_k.shape[-1], dtype=self.dtype)
        precision_h = wickdet.precision_from_covariance(covariance_k)

        expected_t2 = torch.trace(covariance_k @ covariance_k)
        torch.testing.assert_close(
            wickdet.covariance_t2(covariance_k), expected_t2, rtol=1e-13, atol=1e-13
        )

        _, logdet_i_plus_k = torch.linalg.slogdet(identity + covariance_k)
        expected_logdet2 = logdet_i_plus_k - torch.trace(covariance_k)
        expected_q_to_p = -0.5 * expected_logdet2
        expected_p_to_q = 0.5 * (
            logdet_i_plus_k
            + torch.trace(torch.linalg.inv(identity + covariance_k))
            - covariance_k.shape[-1]
        )

        torch.testing.assert_close(
            wickdet.covariance_logdet2(covariance_k),
            expected_logdet2,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            wickdet.kl_q_to_p_from_covariance(covariance_k),
            expected_q_to_p,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            wickdet.kl_q_to_p_from_precision(precision_h),
            expected_q_to_p,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            wickdet.kl_p_to_q_from_covariance(covariance_k),
            expected_p_to_q,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            wickdet.kl_p_to_q_from_precision(precision_h),
            expected_p_to_q,
            rtol=1e-12,
            atol=1e-12,
        )

        recovered_k = wickdet.covariance_from_precision(precision_h)
        torch.testing.assert_close(recovered_k, covariance_k, rtol=1e-12, atol=1e-12)

    def test_q_centered_log_density_variance_is_half_covariance_t2(self) -> None:
        covariance_k = wickdet.covariance_from_amplitude(self.amplitude_b)
        precision_h = wickdet.precision_from_covariance(covariance_k)
        generator = torch.Generator().manual_seed(2718)
        q_samples = wickdet.sample_q_from_amplitude(
            self.amplitude_b,
            150_000,
            generator=generator,
        )
        log_q_over_p = wickdet.precision_wick_log_q_over_p(
            precision_h, q_samples
        )

        expected_mean = wickdet.kl_q_to_p_from_covariance(covariance_k)
        expected_variance = 0.5 * wickdet.covariance_t2(covariance_k)
        identity = torch.eye(covariance_k.shape[-1], dtype=self.dtype)
        # For x~q, Var[1/2 x*Hx] = 1/2 Tr(((I+K)H)^2), and
        # (I+K)H=K.  This is the exact algebra behind the empirical check.
        covariance_times_precision = (identity + covariance_k) @ precision_h
        torch.testing.assert_close(
            covariance_times_precision,
            covariance_k,
            rtol=1e-12,
            atol=1e-12,
        )
        analytic_variance = 0.5 * torch.trace(
            covariance_times_precision @ covariance_times_precision
        )
        torch.testing.assert_close(
            analytic_variance, expected_variance, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            log_q_over_p.mean(), expected_mean, rtol=0.03, atol=0.0005
        )
        torch.testing.assert_close(
            log_q_over_p.var(unbiased=False),
            expected_variance,
            rtol=0.02,
            atol=0.0005,
        )

        # The Wick form is algebraically identical to the ordinary finite-
        # dimensional density ratio; the equality itself is deterministic.
        direct = (
            0.5
            * torch.einsum("bi,ij,bj->b", q_samples[:32], precision_h, q_samples[:32])
            - 0.5 * torch.linalg.slogdet(torch.eye(3, dtype=self.dtype) + covariance_k)[1]
        )
        torch.testing.assert_close(log_q_over_p[:32], direct, rtol=1e-12, atol=1e-12)

    def test_matrix_free_precision_api_names_direction_and_diagnostic(self) -> None:
        precision_h = torch.diag(
            torch.tensor([0.10, 0.20, 0.30], dtype=self.dtype)
        )
        features = torch.tensor(
            [
                [0.5, -1.0, 0.25],
                [1.5, 0.0, -0.5],
                [-0.25, 0.75, 1.0],
                [0.1, -0.2, 0.3],
            ],
            dtype=self.dtype,
        )
        probes = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]], dtype=self.dtype
        )

        def apply_h(x: torch.Tensor) -> torch.Tensor:
            return x @ precision_h.T

        exact_log_q_over_p = wickdet.precision_wick_log_q_over_p(
            precision_h, features
        ).mean()
        q_to_p, diagnostics = wickdet.matrix_free_precision_wick_kl(
            apply_h,
            features,
            direction="q_to_p",
            series_order=32,
            probes=probes,
            return_diagnostics=True,
        )
        p_to_q, _ = wickdet.matrix_free_precision_wick_kl(
            apply_h,
            features,
            direction="p_to_q",
            series_order=32,
            probes=probes,
        )
        torch.testing.assert_close(q_to_p, exact_log_q_over_p, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(p_to_q, -exact_log_q_over_p, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            diagnostics["precision_hs2"],
            torch.trace(precision_h @ precision_h),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertIn("neg_logdet2_i_minus_h", diagnostics)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old_value, _ = wickdet.schatten_cf_penalty(
                apply_h,
                features,
                K=32,
                probes=probes,
                mode="log_density",
            )
        torch.testing.assert_close(old_value, q_to_p, rtol=0.0, atol=0.0)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))


if __name__ == "__main__":
    unittest.main()
