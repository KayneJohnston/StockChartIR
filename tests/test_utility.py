"""Tests for CRRA, Epstein-Zin and shortfall metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src import utility as ut


def constant_bundle(level: float, n_paths: int = 50, horizon: int = 10,
                    with_bequest: bool = False, shift: float = 1.0
                    ) -> ut.ConsumptionBundle:
    consumption = np.full((n_paths, horizon), level)
    bequest = np.full(n_paths, level - shift) if with_bequest else None
    return ut.ConsumptionBundle(consumption, bequest, bequest_shift=shift)


class TestDiscountWeights:
    def test_geometric_series(self) -> None:
        np.testing.assert_allclose(ut.discount_weights(4, 0.5),
                                   [1.0, 0.5, 0.25, 0.125])

    def test_bequest_term_is_appended(self) -> None:
        weights = ut.discount_weights(3, 0.5, bequest_weight=2.0,
                                      include_bequest=True)
        assert weights.size == 4
        assert weights[-1] == pytest.approx(2.0 * 0.5 ** 3)


class TestCRRA:
    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_certain_stream_returns_its_own_level(self, gamma: float) -> None:
        bundle = constant_bundle(1.7)
        cec = ut.crra_certainty_equivalent(bundle, gamma, beta=0.96)
        assert cec == pytest.approx(1.7, rel=1e-10)

    @pytest.mark.parametrize("gamma", [1.0, 2.0, 5.0])
    def test_certain_stream_with_bequest_returns_its_level(self, gamma) -> None:
        # A bequest of (c - kappa) enters the aggregator as exactly c.
        bundle = constant_bundle(2.0, with_bequest=True, shift=1.0)
        cec = ut.crra_certainty_equivalent(bundle, gamma, beta=0.96,
                                           bequest_weight=2.0,
                                           include_bequest=True)
        assert cec == pytest.approx(2.0, rel=1e-10)

    @pytest.mark.parametrize("gamma", [2.0, 5.0, 10.0])
    def test_homogeneous_of_degree_one(self, gamma: float) -> None:
        rng = np.random.default_rng(0)
        consumption = np.exp(rng.normal(0, 0.4, (400, 12)))
        base = ut.ConsumptionBundle(consumption)
        scaled = ut.ConsumptionBundle(consumption * 3.0)
        assert (ut.crra_certainty_equivalent(scaled, gamma, 0.96)
                == pytest.approx(
                    3.0 * ut.crra_certainty_equivalent(base, gamma, 0.96),
                    rel=1e-10))

    def test_risk_lowers_the_certainty_equivalent(self) -> None:
        rng = np.random.default_rng(1)
        risky = np.exp(rng.normal(0, 0.5, (5000, 8)))
        certain = np.full((5000, 8), float(risky.mean()))
        cec_risky = ut.crra_certainty_equivalent(
            ut.ConsumptionBundle(risky), 5.0, 0.96)
        cec_certain = ut.crra_certainty_equivalent(
            ut.ConsumptionBundle(certain), 5.0, 0.96)
        assert cec_risky < cec_certain

    def test_more_risk_aversion_lowers_the_certainty_equivalent(self) -> None:
        rng = np.random.default_rng(2)
        bundle = ut.ConsumptionBundle(np.exp(rng.normal(0, 0.4, (4000, 8))))
        values = [ut.crra_certainty_equivalent(bundle, g, 0.96)
                  for g in (1.0, 2.0, 5.0, 10.0)]
        assert all(a > b for a, b in zip(values, values[1:]))

    def test_dominant_consumption_raises_the_certainty_equivalent(self) -> None:
        rng = np.random.default_rng(3)
        base = np.exp(rng.normal(0, 0.3, (500, 10)))
        better = ut.ConsumptionBundle(base + 0.10)
        assert (ut.crra_certainty_equivalent(better, 5.0, 0.96)
                > ut.crra_certainty_equivalent(ut.ConsumptionBundle(base),
                                               5.0, 0.96))

    def test_log_utility_limit_is_continuous(self) -> None:
        rng = np.random.default_rng(4)
        bundle = ut.ConsumptionBundle(np.exp(rng.normal(0, 0.3, (2000, 6))))
        at_one = ut.crra_certainty_equivalent(bundle, 1.0, 0.96)
        near_one = ut.crra_certainty_equivalent(bundle, 1.0 + 1e-6, 0.96)
        assert at_one == pytest.approx(near_one, rel=1e-4)


class TestEpsteinZin:
    @pytest.mark.parametrize("gamma", [2.0, 5.0, 10.0])
    def test_nests_crra_when_psi_equals_one_over_gamma(self, gamma: float
                                                       ) -> None:
        rng = np.random.default_rng(5)
        bundle = ut.ConsumptionBundle(np.exp(rng.normal(0, 0.35, (3000, 15))))
        crra = ut.crra_certainty_equivalent(bundle, gamma, 0.96)
        ez = ut.epstein_zin_certainty_equivalent(bundle, gamma, 1.0 / gamma,
                                                 0.96)
        assert ez == pytest.approx(crra, rel=1e-9)

    def test_nests_crra_with_a_bequest(self) -> None:
        rng = np.random.default_rng(6)
        consumption = np.exp(rng.normal(0, 0.3, (2000, 10)))
        bequest = np.exp(rng.normal(0, 0.5, 2000))
        bundle = ut.ConsumptionBundle(consumption, bequest, bequest_shift=1.0)
        gamma = 4.0
        crra = ut.crra_certainty_equivalent(bundle, gamma, 0.96, 2.0, True)
        ez = ut.epstein_zin_certainty_equivalent(bundle, gamma, 1.0 / gamma,
                                                 0.96, 2.0, True)
        assert ez == pytest.approx(crra, rel=1e-9)

    @pytest.mark.parametrize("psi", [0.3, 0.5, 1.5, 2.0])
    def test_certain_stream_returns_its_own_level(self, psi: float) -> None:
        bundle = constant_bundle(2.4)
        assert ut.epstein_zin_certainty_equivalent(
            bundle, 5.0, psi, 0.96) == pytest.approx(2.4, rel=1e-10)

    def test_homogeneous_of_degree_one(self) -> None:
        rng = np.random.default_rng(7)
        consumption = np.exp(rng.normal(0, 0.4, (600, 12)))
        base = ut.ConsumptionBundle(consumption)
        scaled = ut.ConsumptionBundle(consumption * 2.5)
        assert (ut.epstein_zin_certainty_equivalent(scaled, 5.0, 1.5, 0.96)
                == pytest.approx(2.5 * ut.epstein_zin_certainty_equivalent(
                    base, 5.0, 1.5, 0.96), rel=1e-10))

    def test_higher_ies_raises_the_index_for_a_tilted_path(self) -> None:
        # A path that is smooth in levels but uneven over time: a higher IES
        # means the investor minds the unevenness less.
        consumption = np.tile(np.array([0.5, 1.5] * 6), (200, 1))
        bundle = ut.ConsumptionBundle(consumption)
        low = ut.epstein_zin_certainty_equivalent(bundle, 5.0, 0.3, 0.96)
        high = ut.epstein_zin_certainty_equivalent(bundle, 5.0, 2.0, 0.96)
        assert high > low

    def test_unit_ies_limit_is_continuous(self) -> None:
        rng = np.random.default_rng(8)
        bundle = ut.ConsumptionBundle(np.exp(rng.normal(0, 0.3, (1000, 8))))
        at_one = ut.epstein_zin_certainty_equivalent(bundle, 3.0, 1.0, 0.96)
        near_one = ut.epstein_zin_certainty_equivalent(bundle, 3.0, 1.0 + 1e-7,
                                                       0.96)
        assert at_one == pytest.approx(near_one, rel=1e-4)


class TestBundleValidation:
    def test_rejects_one_dimensional_consumption(self) -> None:
        with pytest.raises(ValueError, match="n_paths, horizon"):
            ut.ConsumptionBundle(np.ones(10))

    def test_rejects_mismatched_bequest(self) -> None:
        with pytest.raises(ValueError, match="n_paths"):
            ut.ConsumptionBundle(np.ones((4, 3)), np.ones(5))

    def test_zero_bequest_stays_finite(self) -> None:
        bundle = ut.ConsumptionBundle(np.ones((10, 5)), np.zeros(10),
                                      bequest_shift=1.0)
        cec = ut.crra_certainty_equivalent(bundle, 10.0, 0.96, 2.0, True)
        assert np.isfinite(cec) and cec > 0

    def test_consumption_is_floored(self) -> None:
        bundle = ut.ConsumptionBundle(np.zeros((5, 4)), floor=1e-3)
        assert bundle.matrix(False).min() == pytest.approx(1e-3)


class TestShortfallMetrics:
    def test_reports_ruin_and_percentiles(self) -> None:
        rng = np.random.default_rng(9)
        consumption = np.exp(rng.normal(0, 0.2, (1000, 10)))
        bequest = np.clip(rng.normal(5.0, 4.0, 1000), 0, None)
        bundle = ut.ConsumptionBundle(consumption, bequest)
        ruin = bequest <= 0
        metrics = ut.shortfall_metrics(bundle, ruin, np.full(1000, 20.0),
                                       slice(0, 10), consumption_target=1.0)
        assert metrics["prob_ruin"] == pytest.approx(ruin.mean())
        assert metrics["median_wealth_at_retirement"] == pytest.approx(20.0)
        assert 0.0 <= metrics["prob_consumption_below_target"] <= 1.0
        assert metrics["mean_consumption_target"] == pytest.approx(1.0)

    def test_shortfall_is_zero_when_everyone_beats_the_target(self) -> None:
        bundle = ut.ConsumptionBundle(np.full((100, 5), 2.0))
        metrics = ut.shortfall_metrics(bundle, np.zeros(100, dtype=bool),
                                       np.ones(100), slice(0, 5),
                                       consumption_target=1.0)
        assert metrics["mean_consumption_shortfall"] == pytest.approx(0.0)
        assert metrics["prob_consumption_below_target"] == pytest.approx(0.0)

    def test_per_path_target_is_supported(self) -> None:
        consumption = np.tile(np.array([1.0, 1.0, 1.0]), (4, 1))
        bundle = ut.ConsumptionBundle(consumption)
        target = np.array([0.5, 0.9, 1.5, 2.0])
        metrics = ut.shortfall_metrics(bundle, np.zeros(4, dtype=bool),
                                       np.ones(4), slice(0, 3),
                                       consumption_target=target)
        assert metrics["prob_consumption_below_target"] == pytest.approx(0.5)
        assert metrics["mean_consumption_shortfall"] == pytest.approx(
            (0.0 + 0.0 + 0.5 + 1.0) / 4)
