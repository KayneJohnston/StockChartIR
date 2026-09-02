"""Tests for the correlated-human-capital sweep.

The parameter has to be a *correlation* and not a loading. If raising it also
raised income volatility, the sweep would confound two things and the section
would be measuring the wrong one. That property, and the requirement that
zero correlation leaves every existing result untouched, are what these tests
pin down.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import humancapital as hc
from src import lifecycle as lc


@pytest.fixture()
def spec() -> lc.LifecycleSpec:
    return lc.LifecycleSpec(age_start=25, age_retire=63, age_death=93)


@pytest.fixture()
def market(spec) -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(0.07, 0.20, (4000, spec.n_working))


class TestTheParameterIsACorrelation:
    def test_zero_is_bit_identical_with_and_without_the_market(self, spec,
                                                               market) -> None:
        # The guarantee every other result in the project depends on.
        plain = lc.simulate_income(spec, 4000, np.random.default_rng(3))
        with_market = lc.simulate_income(spec, 4000, np.random.default_rng(3),
                                         dom_eq=market)
        assert np.array_equal(plain, with_market)

    @pytest.mark.parametrize("rho", [0.1, 0.3, 0.6])
    def test_the_requested_correlation_is_the_realised_one(self, spec, market,
                                                            rho) -> None:
        z = np.random.default_rng(5).standard_normal(market.shape)
        rotated = rho * lc._standardise(market) + np.sqrt(1 - rho ** 2) * z
        assert np.corrcoef(rotated.ravel(), market.ravel())[0, 1] \
            == pytest.approx(rho, abs=0.02)

    def test_income_volatility_does_not_move_with_the_correlation(self, spec,
                                                                   market
                                                                   ) -> None:
        spreads = []
        for rho in (0.0, 0.3, 0.6):
            tweaked = dataclasses.replace(spec, income_return_correlation=rho)
            income = lc.simulate_income(tweaked, 4000,
                                        np.random.default_rng(3),
                                        dom_eq=market)
            spreads.append(float(np.log(income).std()))
        assert max(spreads) - min(spreads) < 0.01

    def test_a_correlation_without_returns_is_an_error(self, spec) -> None:
        tweaked = dataclasses.replace(spec, income_return_correlation=0.3)
        with pytest.raises(ValueError, match="no domestic equity"):
            lc.simulate_income(tweaked, 100, np.random.default_rng(0))

    def test_an_impossible_correlation_is_rejected(self, spec) -> None:
        with pytest.raises(ValueError, match="income_return_correlation"):
            dataclasses.replace(spec, income_return_correlation=1.4)

    def test_a_constant_market_standardises_to_zero_not_to_nan(self) -> None:
        assert np.all(lc._standardise(np.full((5, 5), 0.07)) == 0.0)


class TestSweep:
    @staticmethod
    def _summary(gap_pct):
        return pd.DataFrame([
            {"strategy": "international_equity", "label": "Intl",
             "cec_crra_gamma5": 1.0 + gap_pct / 100.0},
            {"strategy": "balanced_all_equity", "label": "50/50",
             "cec_crra_gamma5": 1.0},
            {"strategy": "domestic_equity", "label": "Dom",
             "cec_crra_gamma5": 0.9},
        ])

    def test_every_level_is_tagged_with_the_spec_it_used(self, spec) -> None:
        seen = []

        def summarise(tweaked, rho):
            seen.append(tweaked.income_return_correlation)
            return self._summary(5.0)

        frame = hc.sweep(summarise, spec, [0.0, 0.2, 0.4])
        assert seen == [0.0, 0.2, 0.4]
        assert sorted(frame["correlation"].unique()) == [0.0, 0.2, 0.4]

    def test_the_gap_curve_reads_the_pair(self, spec) -> None:
        frame = hc.sweep(lambda s, r: self._summary(4.0 + 10 * r), spec,
                         [0.0, 0.2, 0.4])
        curve = hc.gap_curve(frame,
                             ("international_equity", "balanced_all_equity"))
        assert curve["gap_pct"].iloc[0] == pytest.approx(4.0)
        assert curve["gap_pct"].is_monotonic_increasing

    def test_the_slope_is_per_tenth_of_correlation(self, spec) -> None:
        frame = hc.sweep(lambda s, r: self._summary(4.0 + 10 * r), spec,
                         [0.0, 0.2, 0.4])
        curve = hc.gap_curve(frame,
                             ("international_equity", "balanced_all_equity"))
        assert hc.sensitivity(curve)["slope_per_10pp"] == pytest.approx(1.0)


class TestVerdict:
    @staticmethod
    def _curve(gaps, winner="international_equity", ranks=None):
        n = len(gaps)
        return pd.DataFrame({
            "correlation": np.linspace(0.0, 0.6, n),
            "gap_pct": gaps,
            "winner": [winner] * n,
            "domestic_rank": ranks or [3] * n,
        })

    def test_a_widening_lead_is_reported_as_widening(self) -> None:
        found = hc.verdict(self._curve([4.0, 6.0, 9.0]),
                           {"slope_per_10pp": 0.8},
                           ("international_equity", "balanced_all_equity"))
        assert found["widens_with_correlation"]
        assert not found["winner_ever_changes"]
        assert found["change_pp"] == pytest.approx(5.0)

    def test_a_changed_winner_is_the_headline(self) -> None:
        curve = self._curve([4.0, 1.0, -2.0])
        curve.loc[2, "winner"] = "domestic_equity"
        found = hc.verdict(curve, {"slope_per_10pp": -1.0},
                           ("international_equity", "balanced_all_equity"))
        assert found["winner_ever_changes"]
        assert not found["winner_is_expected_throughout"]

    def test_an_improving_domestic_rank_is_noticed(self) -> None:
        found = hc.verdict(self._curve([4.0, 4.5, 5.0], ranks=[4, 3, 2]),
                           {"slope_per_10pp": 0.2},
                           ("international_equity", "balanced_all_equity"))
        assert found["domestic_ever_improves_rank"]


class TestCorrelationModes:
    """Three readings of a correlated pay cheque, and what separates them."""

    def test_home_mode_leaves_the_foreign_correlation_unspecified(self) -> None:
        built = hc.specs(lc.LifecycleSpec(), [0.0, 0.3], mode="home")
        assert all(s.income_intl_correlation is None for s in built.values())

    def test_strict_mode_pins_the_foreign_correlation_to_zero(self) -> None:
        built = hc.specs(lc.LifecycleSpec(), [0.3], mode="strict")
        assert built[0.3].income_intl_correlation == 0.0
        assert built[0.3].income_return_correlation == pytest.approx(0.3)

    def test_diagonal_mode_correlates_with_both_equally(self) -> None:
        built = hc.specs(lc.LifecycleSpec(), [0.4], mode="diagonal")
        assert built[0.4].income_intl_correlation == pytest.approx(0.4)
        assert built[0.4].income_return_correlation == pytest.approx(0.4)

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            hc.specs(lc.LifecycleSpec(), [0.1], mode="wishful")

    def test_mode_comparison_measures_each_reading_separately(self) -> None:
        curve = pd.DataFrame({
            "mode": ["home", "home", "diagonal", "diagonal"],
            "correlation": [0.0, 0.6, 0.0, 0.6],
            "gap_pct": [5.0, 10.0, 5.0, 6.0],
            "winner": ["a", "a", "a", "a"],
        })
        out = hc.mode_comparison(curve, ("a", "b")).set_index("mode")
        assert float(out.loc["home", "change_pp"]) == pytest.approx(5.0)
        assert float(out.loc["diagonal", "change_pp"]) == pytest.approx(1.0)
        # The objection, quantified: correlating with both markets keeps a
        # fifth of what correlating with home alone delivered.
        assert float(out.loc["diagonal", "share_of_home_effect"]) == \
            pytest.approx(0.2)

    def test_verdict_flags_when_the_diagonal_cancels_most_of_it(self) -> None:
        curve = pd.DataFrame({
            "mode": ["home", "home", "diagonal", "diagonal"],
            "correlation": [0.0, 0.6, 0.0, 0.6],
            "gap_pct": [5.0, 10.0, 5.0, 6.0],
            "winner": ["a", "a", "a", "a"],
            "domestic_rank": [4, 4, 4, 4],
        })
        comparison = hc.mode_comparison(curve, ("a", "b"))
        found = hc.verdict(curve, {"slope_per_10pp": 0.0}, ("a", "b"),
                           mode="home", comparison=comparison)
        assert found["diagonal_mostly_cancels"]
        assert found["diagonal_same_sign"]
        assert found["change_pp"] == pytest.approx(5.0)

    def test_ranking_refuses_to_average_across_modes(self) -> None:
        frame = pd.DataFrame({
            "mode": ["home", "diagonal"], "correlation": [0.0, 0.0],
            "strategy": ["a", "a"], "label": ["a", "a"],
            "cec_crra_gamma5": [1.0, 2.0]})
        with pytest.raises(ValueError, match="several modes"):
            hc.ranking(frame)
        assert len(hc.ranking(frame, mode="home")) == 1


class TestRotation:
    """The two-regressor rotation that pins both correlations at once."""

    @staticmethod
    def _markets(n: int = 40_000, c: float = 0.6):
        rng = np.random.default_rng(7)
        d = rng.standard_normal(n)
        f = c * d + np.sqrt(1 - c ** 2) * rng.standard_normal(n)
        return lc._standardise(d), lc._standardise(f)

    def test_hits_both_targets(self) -> None:
        d, f = self._markets()
        rng = np.random.default_rng(11)
        a, b, resid = lc._rotation(d, f, 0.3, 0.3)
        z = a * d + b * f + resid * rng.standard_normal(d.size)
        assert np.corrcoef(z, d)[0, 1] == pytest.approx(0.3, abs=0.02)
        assert np.corrcoef(z, f)[0, 1] == pytest.approx(0.3, abs=0.02)

    def test_preserves_unit_variance(self) -> None:
        """The whole point: the sweep must not smuggle in extra income risk."""
        d, f = self._markets()
        rng = np.random.default_rng(13)
        a, b, resid = lc._rotation(d, f, 0.5, 0.2)
        z = a * d + b * f + resid * rng.standard_normal(d.size)
        assert float(z.var()) == pytest.approx(1.0, abs=0.02)

    def test_pinning_foreign_to_zero_needs_a_negative_loading(self) -> None:
        """Why `strict` is a stronger claim than leaving foreign unspecified.

        Markets that move together mean a pay cheque correlated with home is
        automatically correlated with abroad. Forcing that to zero requires
        shorting the foreign market inside the innovation.
        """
        d, f = self._markets()
        a, b, _ = lc._rotation(d, f, 0.4, 0.0)
        assert a > 0.0
        assert b < 0.0

    def test_infeasible_correlations_are_rejected(self) -> None:
        d, f = self._markets(c=0.9)
        with pytest.raises(ValueError, match="infeasible"):
            lc._rotation(d, f, 0.99, -0.99)

    def test_diagonal_loads_on_the_average_of_the_two(self) -> None:
        """rho on both markets reduces to rho/(1+c) on each."""
        d, f = self._markets(c=0.6)
        a, b, _ = lc._rotation(d, f, 0.3, 0.3)
        assert a == pytest.approx(b, abs=1e-6)
        assert a == pytest.approx(0.3 / 1.6, abs=0.01)
