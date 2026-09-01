"""Tests for the fee sweeps.

Two things carry the section. A fee must be charged on *assets*, not on
returns, or sixty-eight years of compounding come out wrong; and a fee level
must not change which blocks the bootstrap can draw, or the sweep stops being
a paired comparison and becomes two different experiments.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import fees as fee

PAIR = ("international_equity", "balanced_all_equity")


def _summary(intl: float, fifty: float) -> pd.DataFrame:
    return pd.DataFrame.from_records([
        {"strategy": "international_equity", "label": "100% International",
         "cec_crra_gamma5": intl, "prob_ruin": 0.09},
        {"strategy": "balanced_all_equity", "label": "50/50",
         "cec_crra_gamma5": fifty, "prob_ruin": 0.12},
    ])


class TestNetOfFee:
    def test_a_fee_is_charged_on_assets_not_on_returns(self) -> None:
        # 10% gross, 1% fee: (1.10)(0.99) - 1 = 8.9%, not 9%.
        assert fee.net_of_fee(0.10, 0.01) == pytest.approx(0.089)

    def test_the_difference_from_naive_subtraction_compounds(self) -> None:
        gross = np.full(68, 0.08)
        exact = np.prod(1.0 + fee.net_of_fee(gross, 0.005))
        naive = np.prod(1.0 + (gross - 0.005))
        # Over a lifetime the two disagree by more than a rounding error.
        assert abs(exact / naive - 1.0) > 0.01

    def test_a_zero_fee_is_the_identity(self) -> None:
        gross = np.array([0.1, -0.2, 0.0])
        assert np.allclose(fee.net_of_fee(gross, 0.0), gross)

    def test_a_fee_never_turns_a_loss_into_a_gain(self) -> None:
        assert fee.net_of_fee(-0.30, 0.01) < -0.30

    def test_a_negative_fee_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="subsidy"):
            fee.net_of_fee(0.05, -0.001)


class TestApplyFees:
    def test_availability_is_untouched(self, real_config_or_skip) -> None:
        # If a fee level changed which blocks are drawable, the sweep would
        # compare two different histories rather than two cost levels.
        panel = dl.build_tier_a(real_config_or_skip)
        netted = fee.apply_fees(panel, {"intl_eq": 0.005})
        assert np.array_equal(panel.available, netted.available)
        assert panel.countries == netted.countries
        assert np.array_equal(panel.years, netted.years)

    def test_only_the_named_series_moves(self, real_config_or_skip) -> None:
        panel = dl.build_tier_a(real_config_or_skip)
        netted = fee.apply_fees(panel, {"intl_eq": 0.005})
        assert not np.array_equal(panel.intl_eq, netted.intl_eq,
                                  equal_nan=True)
        for series in ("dom_eq", "bond", "bill", "inflation"):
            assert np.array_equal(getattr(panel, series),
                                  getattr(netted, series), equal_nan=True)

    def test_a_zero_fee_returns_the_same_panel(self, real_config_or_skip
                                               ) -> None:
        panel = dl.build_tier_a(real_config_or_skip)
        assert fee.apply_fees(panel, {"intl_eq": 0.0}) is panel

    def test_the_name_records_what_was_charged(self, real_config_or_skip
                                                ) -> None:
        panel = dl.build_tier_a(real_config_or_skip)
        netted = fee.apply_fees(panel, {"intl_eq": 0.0025})
        assert "25bp" in netted.name

    def test_missing_values_stay_missing(self, real_config_or_skip) -> None:
        panel = dl.build_tier_a(real_config_or_skip)
        netted = fee.apply_fees(panel, {"intl_eq": 0.01})
        assert np.array_equal(np.isnan(panel.intl_eq),
                              np.isnan(netted.intl_eq))

    def test_an_unchargeable_series_is_rejected(self, real_config_or_skip
                                                 ) -> None:
        panel = dl.build_tier_a(real_config_or_skip)
        with pytest.raises(ValueError, match="not chargeable"):
            fee.apply_fees(panel, {"inflation": 0.01})


class TestGapCurve:
    @staticmethod
    def _frame():
        rows = []
        for level, (i, f) in {0.0: (1.10, 1.00), 0.001: (1.05, 1.00),
                              0.002: (0.98, 1.00)}.items():
            block = _summary(i, f)
            block.insert(0, "differential", level)
            rows.append(block)
        return pd.concat(rows, ignore_index=True)

    def test_the_curve_has_one_row_per_level(self) -> None:
        curve = fee.gap_curve(self._frame(), "differential", PAIR)
        assert len(curve) == 3
        assert float(curve.loc[0, "gap_pct"]) == pytest.approx(10.0)

    def test_the_leader_flips_when_the_gap_goes_negative(self) -> None:
        curve = fee.gap_curve(self._frame(), "differential", PAIR)
        assert curve.loc[0, "leader"] == "international_equity"
        assert curve.loc[2, "leader"] == "balanced_all_equity"


class TestBreakEven:
    def test_it_interpolates_between_the_straddling_points(self) -> None:
        curve = pd.DataFrame({"differential": [0.0, 0.001, 0.002],
                              "gap_pct": [10.0, 5.0, -5.0]})
        # Crosses zero halfway between 0.001 and 0.002.
        assert fee.break_even(curve, "differential") == pytest.approx(0.0015)

    def test_a_gap_that_never_closes_is_infinite(self) -> None:
        curve = pd.DataFrame({"differential": [0.0, 0.01],
                              "gap_pct": [10.0, 4.0]})
        assert np.isinf(fee.break_even(curve, "differential"))

    def test_a_gap_already_gone_is_zero(self) -> None:
        curve = pd.DataFrame({"differential": [0.0, 0.01],
                              "gap_pct": [-1.0, -4.0]})
        assert fee.break_even(curve, "differential") == 0.0

    def test_it_finds_the_first_crossing_not_the_last(self) -> None:
        curve = pd.DataFrame({"differential": [0.0, 0.001, 0.002, 0.003],
                              "gap_pct": [5.0, -1.0, 1.0, -2.0]})
        assert fee.break_even(curve, "differential") < 0.002

    def test_an_unsorted_curve_gives_the_same_answer(self) -> None:
        rows = {"differential": [0.002, 0.0, 0.001], "gap_pct": [-5.0, 10.0, 5.0]}
        assert fee.break_even(pd.DataFrame(rows), "differential") \
            == pytest.approx(0.0015)


class TestAnchors:
    def test_each_anchor_is_interpolated_onto_the_grid(self) -> None:
        curve = pd.DataFrame({"differential": [0.0, 0.001, 0.002],
                              "gap_pct": [10.0, 5.0, 0.0]})
        out = fee.anchor_table(curve, "differential",
                               {"cheap": 0.0005, "dear": 0.0015})
        by = {r["label"]: r for _, r in out.iterrows()}
        assert float(by["cheap"]["gap_pct"]) == pytest.approx(7.5)
        assert float(by["dear"]["gap_pct"]) == pytest.approx(2.5)

    def test_anchors_are_reported_in_basis_points(self) -> None:
        curve = pd.DataFrame({"differential": [0.0, 0.01],
                              "gap_pct": [10.0, 0.0]})
        out = fee.anchor_table(curve, "differential", {"x": 0.0025})
        assert float(out["basis_points"].iloc[0]) == pytest.approx(25.0)


class TestVerdict:
    @staticmethod
    def _sweeps(diff_gaps):
        levels = [0.0, 0.005, 0.010]
        common = pd.concat(
            [_summary(1.10, 1.00).assign(fee=v) for v in levels],
            ignore_index=True)
        blocks = []
        for level, g in zip(levels, diff_gaps):
            block = _summary(1.0 + g / 100.0, 1.00)
            block.insert(0, "differential", level)
            blocks.append(block)
        return common, pd.concat(blocks, ignore_index=True)

    def test_a_common_fee_that_never_closes_the_gap_is_flagged_neutral(self
                                                                       ) -> None:
        common, differential = self._sweeps([6.0, 3.0, -1.0])
        anchors = fee.anchor_table(
            fee.gap_curve(differential, "differential", PAIR), "differential",
            {"cheap": 0.0005})
        out = fee.verdict(common, differential, PAIR, anchors)
        assert out["common_is_near_neutral"]

    def test_a_break_even_outside_the_anchors_is_reported_as_such(self
                                                                  ) -> None:
        common, differential = self._sweeps([6.0, 3.0, -1.0])
        anchors = fee.anchor_table(
            fee.gap_curve(differential, "differential", PAIR), "differential",
            {"cheap": 0.0005, "dear": 0.001})
        out = fee.verdict(common, differential, PAIR, anchors)
        assert out["differential_closes_the_gap"]
        assert not out["inside_historic_range"]
        assert out["anchors_that_close_it"] == []

    def test_a_break_even_below_the_cheapest_anchor_is_flagged(self) -> None:
        common, differential = self._sweeps([1.0, -3.0, -6.0])
        anchors = fee.anchor_table(
            fee.gap_curve(differential, "differential", PAIR), "differential",
            {"cheap": 0.004, "dear": 0.008})
        out = fee.verdict(common, differential, PAIR, anchors)
        assert out["below_cheapest_anchor"]
        assert out["cheapest_anchor_that_closes_it"] == "cheap"
