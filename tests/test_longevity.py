"""Tests for the three-way sweep under an uncertain horizon.

The section exists because a fixed death age is not neutral between
withdrawal rules, so most of these check that the machinery can *tell the
two objectives apart* -- a bug that silently scored both the same way would
produce a section concluding, plausibly and wrongly, that nothing changes.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from src import longevity as lv


class TestCombination:
    def test_a_rate_rule_carries_its_rate_into_the_label(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.1,
                               rule="constant_real", rate=0.04)
        assert "4.0%" in combo.label()
        assert "100% equity" in combo.label()

    def test_a_horizon_rule_has_no_rate_in_its_label(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.1, rule="gompertz")
        assert "%" in combo.label()          # the allocation still shows
        assert "@" not in combo.label()      # but no rate does

    def test_a_variant_is_named_by_its_suffix(self) -> None:
        combo = lv.Combination(equity=1.0, domestic=0.1,
                               rule="guyton_klinger", rate=0.04,
                               suffix="tight band")
        assert combo.rule_label == "guyton_klinger (tight band)"

    def test_it_builds_the_rule_it_names(self) -> None:
        from src import spending as spg

        built = lv.Combination(equity=1.0, domestic=0.1,
                               rule="constant_real", rate=0.045).build()
        assert isinstance(built, spg.ConstantRealRule)
        assert built.rate == pytest.approx(0.045)

    def test_params_reach_the_rule(self) -> None:
        built = lv.Combination(equity=1.0, domestic=0.1, rule="gompertz",
                               params={"buffer_years": 5.0}).build()
        assert built.buffer_years == pytest.approx(5.0)


class TestGrids:
    def test_the_allocation_grid_is_the_cross_product(self) -> None:
        grid = lv.allocation_grid([0.8, 1.0], [0.0, 0.1, 0.2])
        assert len(grid) == 6
        assert (1.0, 0.2) in grid

    def test_only_rate_rules_are_crossed_with_the_rates(self) -> None:
        """A rule that derives its level from a planning horizon has no rate
        to sweep, and crossing it with one would score the same policy many
        times and let it win on repetition."""
        specs = [{"key": "constant_real"}, {"key": "gompertz"}]
        plans = lv.plan_grid(specs, [0.03, 0.04, 0.05])
        rates = [r for k, r, _, _ in plans if k == "constant_real"]
        horizon = [r for k, r, _, _ in plans if k == "gompertz"]
        assert rates == [0.03, 0.04, 0.05]
        assert horizon == [None]

    def test_variants_keep_their_params_and_suffix(self) -> None:
        specs = [{"key": "gompertz", "params": {"buffer_years": 5.0},
                  "suffix": "+5y buffer"}]
        (key, rate, params, suffix), = lv.plan_grid(specs, [0.04])
        assert key == "gompertz" and rate is None
        assert params == {"buffer_years": 5.0}
        assert suffix == "+5y buffer"


def _outcome(consumption: float, ruin: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        consumption=np.full((4, 10), float(consumption)),
        ruin=np.full(4, float(ruin)))


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"equity": e, "domestic": d, "rule": r, "rule_label": r,
         "rate": rate, "has_rate": not pd.isna(rate),
         "label": f"{r}-{e}-{d}-{rate}",
         lv.FIXED: cf, lv.MORTALITY: cm,
         "ruin_fixed": rf, "ruin_mortality": rm, "mean_consumption": 1.0}
        for e, d, r, rate, cf, cm, rf, rm in rows])


class TestSweep:
    def test_both_objectives_come_off_one_outcome(self) -> None:
        """The difference between the two scores has to be the aggregation
        and nothing else, so the sweep must not simulate twice."""
        calls = []

        def simulate(combo):
            calls.append(combo)
            return _outcome(1.0, 0.1)

        combos = [lv.Combination(equity=1.0, domestic=0.1,
                                 rule="constant_real", rate=0.04),
                  lv.Combination(equity=0.8, domestic=0.1, rule="gompertz")]
        frame = lv.sweep(simulate, lambda o: 1.0, lambda o: 0.9,
                         lambda o: 0.05, combos, log_every=0)
        assert len(calls) == 2
        assert len(frame) == 2
        assert set(frame[lv.FIXED]) == {1.0}
        assert set(frame[lv.MORTALITY]) == {0.9}

    def test_a_horizon_rule_records_no_rate(self) -> None:
        frame = lv.sweep(lambda c: _outcome(1.0, 0.0), lambda o: 1.0,
                         lambda o: 1.0, lambda o: 0.0,
                         [lv.Combination(equity=1.0, domestic=0.1,
                                         rule="gompertz")], log_every=0)
        assert not bool(frame["has_rate"].iloc[0])
        assert np.isnan(frame["rate"].iloc[0])


class TestOptimumAndRanking:
    ROWS = [
        (1.0, 0.1, "constant_real", 0.04, 1.00, 0.80, 0.12, 0.05),
        (1.0, 0.1, "constant_real", 0.05, 0.95, 0.85, 0.20, 0.09),
        (1.0, 0.1, "gompertz", np.nan, 0.90, 0.95, 0.00, 0.00),
        (0.8, 0.1, "gompertz", np.nan, 0.88, 0.92, 0.00, 0.00),
    ]

    def test_each_objective_can_pick_a_different_winner(self) -> None:
        frame = _frame(self.ROWS)
        assert lv.optimum(frame, lv.FIXED)["rule"] == "constant_real"
        assert lv.optimum(frame, lv.MORTALITY)["rule"] == "gompertz"

    def test_by_objective_reports_both_winners(self) -> None:
        table = lv.by_objective(_frame(self.ROWS))
        assert list(table["objective"]) == ["fixed horizon",
                                            "survival-weighted"]
        assert table["rule"].tolist() == ["constant_real", "gompertz"]

    def test_each_rule_is_ranked_at_its_own_best_settings(self) -> None:
        """Otherwise a rule is penalised for a rate chosen to suit a
        different horizon, which is the comparison this section exists to
        avoid making."""
        shift = lv.ranking_shift(_frame(self.ROWS))
        row = shift[shift["rule_label"] == "constant_real"].iloc[0]
        assert row[lv.FIXED] == pytest.approx(1.00)   # its best, not its last
        assert row[lv.MORTALITY] == pytest.approx(0.85)

    def test_the_rank_change_is_signed_toward_promotion(self) -> None:
        shift = lv.ranking_shift(_frame(self.ROWS))
        gompertz = shift[shift["rule_label"] == "gompertz"].iloc[0]
        assert gompertz["rank_change"] > 0        # promoted by a real horizon

    def test_an_empty_frame_has_no_optimum(self) -> None:
        with pytest.raises(ValueError, match="no combinations"):
            lv.optimum(pd.DataFrame())


class TestAblation:
    ROWS = [
        # the fixed-horizon winner, and a poor performer once survival-weighted
        (1.0, 0.1, "constant_real", 0.04, 1.00, 0.80, 0.12, 0.05),
        # freeing the rate alone
        (1.0, 0.1, "constant_real", 0.06, 0.90, 0.84, 0.30, 0.14),
        # freeing the allocation alone
        (0.8, 0.1, "constant_real", 0.04, 0.92, 0.83, 0.10, 0.04),
        # freeing the rule alone
        (1.0, 0.1, "gompertz", np.nan, 0.85, 0.90, 0.00, 0.00),
        # all three
        (0.8, 0.3, "gompertz", np.nan, 0.80, 0.97, 0.00, 0.00),
    ]

    def test_the_baseline_is_the_fixed_choice_scored_the_new_way(self
                                                                 ) -> None:
        out = lv.ablation(_frame(self.ROWS))
        base = out.iloc[0]
        assert base["freed"].startswith("nothing")
        assert base["cec"] == pytest.approx(0.80)
        assert base["gain_pct"] == pytest.approx(0.0)

    def test_freeing_one_decision_holds_the_others(self) -> None:
        out = lv.ablation(_frame(self.ROWS)).set_index("freed")
        assert out.loc["rate", "cec"] == pytest.approx(0.84)
        assert out.loc["allocation", "cec"] == pytest.approx(0.83)
        assert out.loc["rule", "cec"] == pytest.approx(0.90)

    def test_freeing_everything_finds_the_overall_best(self) -> None:
        out = lv.ablation(_frame(self.ROWS)).set_index("freed")
        assert out.loc["all three", "cec"] == pytest.approx(0.97)

    def test_gains_are_measured_against_the_baseline(self) -> None:
        out = lv.ablation(_frame(self.ROWS)).set_index("freed")
        assert out.loc["rule", "gain_pct"] == pytest.approx(
            100.0 * (0.90 / 0.80 - 1.0))


class TestVerdict:
    def test_it_names_what_changed(self) -> None:
        frame = _frame(TestOptimumAndRanking.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert found["rule_changes"]
        assert found["anything_changes"]

    def test_a_winner_with_no_rate_is_flagged_not_printed_as_nan(self
                                                                 ) -> None:
        frame = _frame(TestOptimumAndRanking.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert found["winner_sets_no_rate"]
        assert "rate_changes" not in found

    def test_ruin_is_compared_only_on_rules_that_can_run_out(self) -> None:
        """The winning rule often cannot deplete, and zero over zero would
        say nothing about how much a fixed horizon overstates the risk."""
        frame = _frame(TestOptimumAndRanking.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert not found["optimum_can_deplete"]
        assert found["best_depleting_rule"] == "constant_real"
        assert found["best_depleting_ratio"] == pytest.approx(0.20 / 0.09)

    def test_a_fixed_horizon_that_changes_nothing_is_reported_as_such(self
                                                                      ) -> None:
        """The control: if both objectives pick the same combination the
        section has to say so rather than manufacture a difference."""
        rows = [(1.0, 0.1, "gompertz", np.nan, 1.00, 1.00, 0.0, 0.0),
                (0.8, 0.1, "gompertz", np.nan, 0.90, 0.90, 0.0, 0.0)]
        frame = _frame(rows)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert not found["anything_changes"]

    def test_separability_is_judged_against_the_joint_gain(self) -> None:
        frame = _frame(TestAblation.ROWS)
        found = lv.verdict(frame, lv.ranking_shift(frame),
                           lv.ablation(frame))
        assert "interaction_pct" in found
        assert found["interaction_pct"] == pytest.approx(
            found["joint_gain_pct"] - sum(found["single_gains_pct"].values()))

    def test_an_empty_frame_reports_nothing(self) -> None:
        assert lv.verdict(pd.DataFrame(), pd.DataFrame(),
                          pd.DataFrame()) == {"measured": False}


class TestDescribe:
    def test_a_rate_rule_reads_with_its_rate(self) -> None:
        assert lv.describe("constant real", 0.04) == "constant real at 4.0%"

    def test_a_horizon_rule_says_so_rather_than_printing_nan(self) -> None:
        assert "no rate" in lv.describe("gompertz", float("nan"))
        assert "nan" not in lv.describe("gompertz", float("nan")).lower()

    def test_none_is_handled_like_a_missing_rate(self) -> None:
        assert "no rate" in lv.describe("gompertz", None)
