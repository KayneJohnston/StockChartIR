"""Foreign dividend withholding tax on the international sleeve.

Three properties carry this section. The tax must be charged in the exact
form the source data's return convention implies, it must fall on the foreign
leg and nothing else, and a rate of zero must leave the panel untouched.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import withholding as wh


def _panel(intl: np.ndarray, name: str = "p"):
    """A stand-in carrying only the fields apply_withholding reads."""
    @dataclasses.dataclass
    class _P:
        intl_eq: np.ndarray
        dom_eq: np.ndarray
        bond: np.ndarray
        bill: np.ndarray
        name: str
    zeros = np.zeros_like(intl)
    return _P(intl_eq=intl, dom_eq=zeros + 0.07, bond=zeros + 0.02,
              bill=zeros + 0.01, name=name)


def _jst(rows) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [{"iso": iso, "year": year, "eq_dp": dp} for iso, year, dp in rows])


class TestDividendShare:
    def test_is_the_dividend_share_of_the_ending_value(self) -> None:
        """The source compounds: 1+total = (1+gain)(1+dividend).

        So the dividend's share of the gross factor is dp/(1+dp) -- which
        needs no reference to the total return and therefore cannot be thrown
        off by the panel's hyperinflations.
        """
        out = wh.dividend_share(_jst([("AUS", 1900, 0.04)]), ["AUS"],
                                np.array([1900]))
        assert out[0, 0] == pytest.approx(0.04 / 1.04)

    def test_drops_impossible_values_rather_than_clipping(self) -> None:
        out = wh.dividend_share(
            _jst([("AUS", 1900, -0.5), ("AUS", 1901, 0.03)]),
            ["AUS"], np.array([1900, 1901]))
        assert np.isnan(out[0, 0])
        assert np.isfinite(out[1, 0])

    def test_missing_country_years_are_nan(self) -> None:
        out = wh.dividend_share(_jst([("AUS", 1900, 0.04)]), ["AUS", "GBR"],
                                np.array([1900, 1901]))
        assert np.isfinite(out[0, 0])
        assert np.isnan(out[1, 0]) and np.isnan(out[0, 1])


class TestSleeve:
    def test_leaves_the_country_itself_out(self) -> None:
        share = np.array([[0.02, 0.04, 0.06]])
        out = wh.sleeve_dividend_share(share)
        assert out[0, 0] == pytest.approx(0.05)
        assert out[0, 1] == pytest.approx(0.04)

    def test_a_year_with_no_other_market_is_nan(self) -> None:
        out = wh.sleeve_dividend_share(np.array([[0.03, np.nan, np.nan]]))
        assert np.isnan(out[0, 0])
        assert out[0, 1] == pytest.approx(0.03)


class TestCharging:
    def test_a_tax_is_a_fee_of_rate_times_dividend_share(self) -> None:
        """The translation the whole section rests on.

        Withholding at tau leaves (1-tau) of the dividend, so the after-tax
        gross factor is (1+cg)(1+(1-tau)dp), which is exactly (1+r)(1-tau*q).
        """
        dp = 0.04
        q = dp / (1.0 + dp)
        cg = 0.06
        gross = (1.0 + cg) * (1.0 + dp) - 1.0
        tau = 0.30
        exact = (1.0 + cg) * (1.0 + (1.0 - tau) * dp) - 1.0
        via_fee = (1.0 + gross) * (1.0 - tau * q) - 1.0
        assert via_fee == pytest.approx(exact)

    def test_zero_rate_returns_the_panel_untouched(self) -> None:
        panel = _panel(np.full((3, 2), 0.08))
        assert wh.apply_withholding(panel, 0.0, np.full((3, 2), 0.04)) is panel

    def test_charges_the_foreign_leg_and_nothing_else(self) -> None:
        panel = _panel(np.full((3, 2), 0.08))
        taxed = wh.apply_withholding(panel, 0.30, np.full((3, 2), 0.04))
        assert (taxed.intl_eq < panel.intl_eq).all()
        assert np.array_equal(taxed.dom_eq, panel.dom_eq)
        assert np.array_equal(taxed.bond, panel.bond)
        assert np.array_equal(taxed.bill, panel.bill)

    def test_missing_dividend_cells_are_charged_nothing(self) -> None:
        """Dropping them would change which blocks the sampler can draw and
        confound the tax with a change of sample."""
        sleeve = np.array([[0.04, np.nan]])
        taxed = wh.apply_withholding(_panel(np.array([[0.08, 0.08]])), 0.30,
                                     sleeve)
        assert taxed.intl_eq[0, 0] < 0.08
        assert taxed.intl_eq[0, 1] == pytest.approx(0.08)

    def test_rejects_an_impossible_rate(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            wh.effective_fee(1.5, np.zeros((2, 2)))


class TestDrag:
    def test_falls_as_dividend_yields_fall(self) -> None:
        years = np.arange(1890, 2021)
        sleeve = np.where(years[:, None] < 1950, 0.05, 0.02)
        out = wh.realised_drag(0.15, sleeve, years)
        eras = out[out["era"] != "whole panel"]
        assert eras["drag_bp"].iloc[0] > eras["drag_bp"].iloc[-1]

    def test_the_whole_panel_row_is_the_mean(self) -> None:
        years = np.arange(1890, 2021)
        sleeve = np.full((years.size, 2), 0.04)
        out = wh.realised_drag(0.15, sleeve, years)
        whole = out[out["era"] == "whole panel"]
        assert float(whole["drag_bp"].iloc[0]) == pytest.approx(0.15 * 0.04 * 1e4)


class TestCrossings:
    @staticmethod
    def _curve(points) -> pd.DataFrame:
        return pd.DataFrame([
            {"rate": r, "rate_pct": r * 100.0,
             "lead_over_rival_pct": lead, "winner": "x"}
            for r, lead in points])

    def test_interpolates_the_crossing(self) -> None:
        curve = self._curve([(0.0, 10.0), (0.2, -10.0)])
        assert wh.crossing(curve, "rival") == pytest.approx(0.10, rel=0.05)

    def test_infinite_when_the_lead_never_closes(self) -> None:
        assert np.isinf(wh.crossing(self._curve([(0.0, 10.0), (0.5, 4.0)]),
                                    "rival"))

    def test_zero_when_the_lead_was_never_there(self) -> None:
        assert wh.crossing(self._curve([(0.0, -1.0), (0.5, -4.0)]),
                           "rival") == 0.0

    def test_reports_the_crossing_in_basis_points(self) -> None:
        curve = self._curve([(0.0, 10.0), (0.2, -10.0)])
        out = wh.crossings(curve, ["rival"], np.full((4, 2), 0.04))
        row = out.iloc[0]
        assert bool(row["reached_on_grid"])
        # 10% of a 4% dividend share is 40 bp a year.
        assert float(row["equivalent_drag_bp"]) == pytest.approx(40.0, rel=0.06)


class TestVerdict:
    @staticmethod
    def _inputs(first_pct: float):
        curve = pd.DataFrame([
            {"rate": 0.0, "rate_pct": 0.0, "winner": "a"},
            {"rate": 0.5, "rate_pct": 50.0, "winner": "b"}])
        crossed = pd.DataFrame([{"rival": "b", "crossing_rate": first_pct / 100,
                                 "crossing_pct": first_pct,
                                 "equivalent_drag_bp": 100.0,
                                 "reached_on_grid": True,
                                 "lead_at_zero_pct": 5.0}])
        optima = pd.DataFrame([
            {"rate": 0.0, "optimal_domestic_share": 0.1,
             "margin_over_runner_up_pct": 0.5},
            {"rate": 0.5, "optimal_domestic_share": 0.4,
             "margin_over_runner_up_pct": 0.5}])
        drag = pd.DataFrame([{"era": "whole panel", "drag_bp": 57.0},
                             {"era": "1890-1949", "drag_bp": 66.0},
                             {"era": "1990-2020", "drag_bp": 42.0}])
        return curve, crossed, optima, drag

    def test_sees_a_crossing_inside_the_statutory_rate(self) -> None:
        found = wh.verdict(*self._inputs(29.0), challenger="a")
        assert found["any_rival_overtakes"]
        assert found["crossing_within_statutory"]
        assert not found["crossing_within_treaty"]

    def test_sees_one_outside_it(self) -> None:
        found = wh.verdict(*self._inputs(45.0), challenger="a")
        assert not found["crossing_within_statutory"]

    def test_reports_the_optimum_walking_home(self) -> None:
        found = wh.verdict(*self._inputs(29.0), challenger="a")
        assert found["optimum_moves_home"]
        assert found["optimal_domestic_shift"] == pytest.approx(0.3)

    def test_notes_the_drag_falling_over_time(self) -> None:
        found = wh.verdict(*self._inputs(29.0), challenger="a")
        assert found["drag_falls_over_time"]

    def test_empty_curve_is_not_a_crash(self) -> None:
        assert wh.verdict(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                          pd.DataFrame(), "a") == {"levels": 0}
