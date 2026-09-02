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
