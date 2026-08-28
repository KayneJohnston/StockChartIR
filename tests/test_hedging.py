"""Tests for the currency-hedged international leg and the hedging sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import hedging as hg


class TestHedgedLeg:
    def test_matches_covered_interest_parity(self) -> None:
        # Two countries, no inflation. Country 0's hedged foreign return is
        # country 1's equity return scaled by the interest differential.
        eq_tr = np.array([[0.05, 0.20]])
        bill = np.array([[0.04, 0.01]])
        inflation = np.zeros((1, 2))
        out = dl.build_hedged_international_leg(eq_tr, bill, inflation)
        expected = 1.20 * (1.04 / 1.01) - 1.0
        assert out[0, 0] == pytest.approx(expected)

    def test_leaves_out_the_investors_own_market(self) -> None:
        eq_tr = np.array([[0.10, 0.20, 0.30]])
        bill = np.zeros((1, 3))
        inflation = np.zeros((1, 3))
        out = dl.build_hedged_international_leg(eq_tr, bill, inflation)
        np.testing.assert_allclose(out[0, 0], (1.20 + 1.30) / 2 - 1.0)

    def test_hedging_cost_is_a_flat_drag(self) -> None:
        eq_tr = np.array([[0.0, 0.10]])
        bill = np.zeros((1, 2))
        inflation = np.zeros((1, 2))
        free = dl.build_hedged_international_leg(eq_tr, bill, inflation, 0.0)
        charged = dl.build_hedged_international_leg(eq_tr, bill, inflation, 0.01)
        assert (1.0 + charged[0, 0]) == pytest.approx(
            (1.0 + free[0, 0]) * 0.99)

    def test_domestic_inflation_deflates_the_result(self) -> None:
        eq_tr = np.array([[0.0, 0.10]])
        bill = np.zeros((1, 2))
        inflation = np.array([[0.10, 0.0]])
        out = dl.build_hedged_international_leg(eq_tr, bill, inflation)
        assert out[0, 0] == pytest.approx(1.10 / 1.10 - 1.0)

    def test_single_country_has_no_foreign_leg(self) -> None:
        out = dl.build_hedged_international_leg(
            np.array([[0.1]]), np.zeros((1, 1)), np.zeros((1, 1)))
        assert np.isnan(out).all()


class TestBlend:
    def test_zero_ratio_returns_the_unhedged_leg(self) -> None:
        unhedged = np.array([[0.1, 0.2]])
        hedged = np.array([[0.5, 0.6]])
        np.testing.assert_allclose(
            dl.blend_international_legs(unhedged, hedged, 0.0), unhedged)

    def test_full_ratio_returns_the_hedged_leg(self) -> None:
        unhedged = np.array([[0.1, 0.2]])
        hedged = np.array([[0.5, 0.6]])
        np.testing.assert_allclose(
            dl.blend_international_legs(unhedged, hedged, 1.0), hedged)

    def test_partial_ratio_is_linear(self) -> None:
        unhedged = np.array([[0.0]])
        hedged = np.array([[1.0]])
        np.testing.assert_allclose(
            dl.blend_international_legs(unhedged, hedged, 0.25), [[0.25]])

    def test_falls_back_where_the_hedge_is_not_computable(self) -> None:
        # A missing hedged value must never remove a usable country-year, or
        # the hedge ratio would change which blocks the bootstrap can draw.
        unhedged = np.array([[0.1, 0.2]])
        hedged = np.array([[np.nan, 0.6]])
        out = dl.blend_international_legs(unhedged, hedged, 1.0)
        assert out[0, 0] == pytest.approx(0.1)
        assert out[0, 1] == pytest.approx(0.6)

    def test_clips_an_out_of_range_ratio(self) -> None:
        unhedged = np.array([[0.0]])
        hedged = np.array([[1.0]])
        np.testing.assert_allclose(
            dl.blend_international_legs(unhedged, hedged, 3.0), [[1.0]])


class TestBreakEven:
    def _frame(self, values, costs=(0.0, 0.005, 0.01, 0.02),
               baseline=1.0) -> pd.DataFrame:
        rows = [{"strategy": "balanced_all_equity", "hedge_ratio": 0.0,
                 "hedge_cost": 0.0, "cec": baseline}]
        for cost, value in zip(costs, values):
            rows.append({"strategy": "balanced_all_equity", "hedge_ratio": 0.5,
                         "hedge_cost": cost, "cec": value})
        return pd.DataFrame.from_records(rows)

    def test_interpolates_the_crossing(self) -> None:
        # Advantage goes +0.02, +0.01, -0.01, -0.03: crosses between 0.005
        # and 0.01.
        out = hg.break_even_costs(self._frame([1.02, 1.01, 0.99, 0.97]), "cec")
        value = float(out["break_even_annual_cost"].iloc[0])
        assert 0.005 < value < 0.01

    def test_reports_nan_when_hedging_never_wins(self) -> None:
        out = hg.break_even_costs(self._frame([0.99, 0.98, 0.97, 0.96]), "cec")
        assert np.isnan(out["break_even_annual_cost"].iloc[0])

    def test_reports_infinity_when_hedging_always_wins(self) -> None:
        out = hg.break_even_costs(self._frame([1.05, 1.04, 1.03, 1.02]), "cec")
        assert np.isinf(out["break_even_annual_cost"].iloc[0])

    def test_reports_the_zero_cost_gain(self) -> None:
        out = hg.break_even_costs(self._frame([1.02, 1.01, 0.99, 0.97]), "cec")
        assert out["gain_at_zero_cost_pct"].iloc[0] == pytest.approx(2.0)


class TestOptimalRatio:
    def test_picks_the_best_ratio_at_each_cost(self) -> None:
        rows = [{"strategy": "s", "hedge_ratio": 0.0, "hedge_cost": 0.0,
                 "cec": 1.00}]
        for ratio, values in ((0.25, [1.02, 0.99]), (0.5, [1.01, 0.98])):
            for cost, value in zip([0.0, 0.01], values):
                rows.append({"strategy": "s", "hedge_ratio": ratio,
                             "hedge_cost": cost, "cec": value})
        out = hg.optimal_ratio_by_cost(pd.DataFrame.from_records(rows), "cec",
                                       strategy="s")
        assert out.loc[out.hedge_cost == 0.0,
                       "optimal_hedge_ratio"].iloc[0] == pytest.approx(0.25)
        # At the higher cost every hedge loses, so the unhedged sleeve wins.
        assert out.loc[out.hedge_cost == 0.01,
                       "optimal_hedge_ratio"].iloc[0] == pytest.approx(0.0)


class TestPanelIntegration:
    @pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
    def test_hedging_never_changes_availability(self, ratio) -> None:
        cfg = dl.load_config()
        base = dl.build_panel(cfg, "jst16", hedge_ratio=0.0)
        hedged = dl.build_panel(cfg, "jst16", hedge_ratio=ratio,
                                hedge_cost=0.005)
        np.testing.assert_array_equal(hedged.available, base.available)
        assert hedged.countries == base.countries

    def test_zero_ratio_reproduces_the_unhedged_panel(self) -> None:
        cfg = dl.load_config()
        base = dl.build_panel(cfg, "jst16")
        same = dl.build_panel(cfg, "jst16", hedge_ratio=0.0, hedge_cost=0.02)
        np.testing.assert_allclose(same.intl_eq, base.intl_eq, equal_nan=True)

    def test_a_higher_cost_lowers_the_hedged_leg(self) -> None:
        cfg = dl.load_config()
        cheap = dl.build_panel(cfg, "jst16", hedge_ratio=1.0, hedge_cost=0.0)
        dear = dl.build_panel(cfg, "jst16", hedge_ratio=1.0, hedge_cost=0.02)
        mask = cheap.available
        assert dear.intl_eq[mask].mean() < cheap.intl_eq[mask].mean()

    def test_panel_moments_are_reported(self) -> None:
        cfg = dl.load_config()
        moments = hg.panel_moments(dl.build_panel(cfg, "jst16"))
        assert set(moments) >= {"intl_mean", "intl_sd",
                                "corr_intl_domestic_equity"}
        assert -1.0 <= moments["corr_intl_domestic_equity"] <= 1.0
