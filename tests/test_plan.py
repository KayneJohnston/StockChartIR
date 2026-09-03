"""The plan and the portfolio, solved together.

The claim under test is `docs/06`'s: that the spending rule and the asset
allocation "can be chosen independently". It is true of rankings and false of
optima, and the machinery here has to be able to tell those two apart -- so
the tests pin the distinction, the mechanism behind it, and the arithmetic of
the interaction term that measures it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import plan as pl
from src import spending as spg


class TestPlan:
    def test_a_rated_rule_carries_its_rate(self) -> None:
        rule = pl.Plan("constant_percent", 0.07, 63).build()
        assert isinstance(rule, spg.SpendingRule)
        assert "7" in pl.Plan("constant_percent", 0.07, 63).label()

    def test_a_rateless_rule_ignores_the_rate(self) -> None:
        """Sweeping a rate over a horizon rule would score one plan twice."""
        assert not pl.Plan("life_expectancy", 0.07, 63).rated
        assert "at 7" not in pl.Plan("life_expectancy", 0.07, 63).label()
        # ...and it still builds, rather than failing on the unused argument.
        assert isinstance(pl.Plan("life_expectancy", 0.07, 63).build(),
                          spg.SpendingRule)

    def test_the_grid_lists_a_rateless_rule_once(self) -> None:
        grid = pl.plan_grid(["constant_real", "life_expectancy"],
                            [0.03, 0.04, 0.05], [63])
        rateless = [p for p in grid if p.rule == "life_expectancy"]
        rated = [p for p in grid if p.rule == "constant_real"]
        assert len(rateless) == 1
        assert len(rated) == 3

    def test_the_grid_crosses_every_age(self) -> None:
        grid = pl.plan_grid(["constant_real"], [0.04], [60, 63, 66])
        assert sorted(p.retire_age for p in grid) == [60, 63, 66]

    def test_plans_are_comparable_by_value(self) -> None:
        """`alternate` detects its fixed point by equality, so this matters."""
        assert pl.Plan("constant_real", 0.04, 63) == \
            pl.Plan("constant_real", 0.04, 63)
        assert pl.Plan("constant_real", 0.04, 63) != \
            pl.Plan("constant_real", 0.05, 63)

    def test_keys_are_distinct_across_the_grid(self) -> None:
        grid = pl.plan_grid(["constant_real", "constant_percent",
                             "life_expectancy"], [0.03, 0.04], [63, 66])
        assert len({p.key() for p in grid}) == len(grid)

    def test_spec_for_moves_only_the_retirement_date(self) -> None:
        from src import lifecycle as lc

        spec = lc.LifecycleSpec()
        moved = pl.spec_for(spec, 68)
        assert moved.age_retire == 68
        assert moved.age_start == spec.age_start
        assert moved.age_death == spec.age_death
        assert moved.horizon == spec.horizon         # only the split moves
        assert moved.n_working > spec.n_working


class TestCanDeplete:
    """A rule that spends a fraction of what is left cannot reach zero."""

    def test_the_fixed_real_rule_can_run_out(self) -> None:
        assert pl.CAN_DEPLETE["constant_real"]

    def test_the_percentage_rule_cannot(self) -> None:
        assert not pl.CAN_DEPLETE["constant_percent"]
        assert not pl.CAN_DEPLETE["life_expectancy"]

    def test_every_registered_rule_is_classified(self) -> None:
        assert set(spg.REGISTRY) <= set(pl.CAN_DEPLETE)

    def test_the_default_rules_span_both_kinds(self) -> None:
        """A comparison with only one kind could not show the mechanism."""
        kinds = {pl.CAN_DEPLETE[k] for k, _ in pl.DEFAULT_RULES}
        assert kinds == {True, False}


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def _sweep(shares_by_rule: dict, ruin_by_rule: dict | None = None
           ) -> tuple[pd.DataFrame, dict]:
    """A crossed sweep whose argmax per rule is exactly what was asked for."""
    grid = [0.0, 0.1, 0.2, 0.3]
    param = {f"d{int(s * 100):03d}": s for s in grid}
    rows = []
    for rule, best in shares_by_rule.items():
        for key, share in param.items():
            rows.append({
                "rule": rule,
                "can_deplete": pl.CAN_DEPLETE.get(rule, True),
                "strategy": key,
                "cec": 1.0 - abs(share - best),
                "prob_ruin": (abs(share - ruin_by_rule[rule])
                              if ruin_by_rule else 0.0),
                "n_paths": 1000,
            })
    return _frame(rows), param


class TestSeparability:
    def test_it_separates_the_ranking_from_the_optimum(self) -> None:
        """The distinction the whole section turns on.

        Two rules can order every candidate identically and still peak in
        different places only if the peak is inside the ordering -- so this
        fixture gives them genuinely different argmaxes and checks both
        answers are reported, not merged.
        """
        frame, param = _sweep({"constant_real": 0.2,
                               "constant_percent": 0.0})
        optima = pl.optimum_by_rule(frame, param, "domestic_share")
        found = pl.separability(optima, frame, param, "domestic_share")
        assert found["optimum_moves_with_the_rule"]
        assert found["optimum_low"] == pytest.approx(0.0)
        assert found["optimum_high"] == pytest.approx(0.2)
        assert found["optimum_spread_pp"] == pytest.approx(20.0)

    def test_identical_optima_are_reported_as_separable(self) -> None:
        frame, param = _sweep({"constant_real": 0.2,
                               "constant_percent": 0.2})
        optima = pl.optimum_by_rule(frame, param, "domestic_share")
        found = pl.separability(optima, frame, param, "domestic_share")
        assert not found["optimum_moves_with_the_rule"]
        assert found["rankings_identical"]
        assert not found["ranking_separable_but_optimum_is_not"]

    def test_the_headline_case_is_recognised(self) -> None:
        """Rankings identical, optimum not: the finding, when it holds."""
        frame, param = _sweep({"a": 0.2, "b": 0.2})
        # Make b's ordering identical but shift its peak by a hair, so the
        # argmax moves while the sort order does not.
        frame.loc[(frame["rule"] == "b") & (frame["strategy"] == "d010"),
                  "cec"] = 1.5
        optima = pl.optimum_by_rule(frame, param, "domestic_share")
        found = pl.separability(optima, frame, param, "domestic_share")
        assert found["optimum_moves_with_the_rule"]

    def test_an_empty_frame_reports_nothing(self) -> None:
        assert pl.separability(pd.DataFrame(), pd.DataFrame(), {}) == \
            {"measured": False}


class TestMechanism:
    def test_it_reports_where_the_two_objectives_agree(self) -> None:
        frame, param = _sweep({"constant_real": 0.2},
                              {"constant_real": 0.2})
        out = pl.ruin_minimum_by_rule(frame, param, "domestic_share")
        row = out.iloc[0]
        assert row["cec_optimal_domestic_share"] == pytest.approx(0.2)
        assert row["ruin_optimal_domestic_share"] == pytest.approx(0.2)
        assert bool(row["agree"])

    def test_it_reports_where_they_part_company(self) -> None:
        frame, param = _sweep({"constant_real": 0.2},
                              {"constant_real": 0.0})
        out = pl.ruin_minimum_by_rule(frame, param, "domestic_share")
        assert not bool(out.iloc[0]["agree"])

    def test_a_rule_that_cannot_ruin_is_flagged(self) -> None:
        """Its ruin column is zero everywhere, so the argmin means nothing."""
        frame, param = _sweep({"constant_percent": 0.0})
        out = pl.ruin_minimum_by_rule(frame, param, "domestic_share")
        assert not bool(out.iloc[0]["ruin_is_possible"])
        assert not bool(out.iloc[0]["can_deplete"])


class TestAblationArithmetic:
    """The interaction term, which is the section's headline number."""

    @staticmethod
    def _table(base: float, alloc: float, plan: float, both: float
               ) -> pd.DataFrame:
        frame = _frame([
            {"freedom": "neither", "plan": "b", "cec": base,
             "mean_equity": 1.0, "mean_domestic": 0.1},
            {"freedom": "allocation only", "plan": "b", "cec": alloc,
             "mean_equity": 1.0, "mean_domestic": 0.2},
            {"freedom": "plan only", "plan": "p", "cec": plan,
             "mean_equity": 1.0, "mean_domestic": 0.1},
            {"freedom": "both", "plan": "p", "cec": both,
             "mean_equity": 1.0, "mean_domestic": 0.2},
        ])
        frame["gain_over_neither_pct"] = (frame["cec"] / base - 1.0) * 100.0
        return frame

    @staticmethod
    def _joint(cec: float) -> dict:
        return {"plan": pl.Plan("constant_percent", 0.07, 63),
                "equity": np.ones(68), "domestic": np.full(68, 0.2),
                "cec": cec, "rounds": pd.DataFrame([{"round": 0}]),
                "converged": True}

    def test_additive_outcomes_give_a_zero_interaction(self) -> None:
        from src import lifecycle as lc

        # 1.00 -> 1.10 (allocation), -> 1.20 (plan), -> 1.30 (both):
        # gains 10, 20, 30, so the interaction is exactly zero.
        table = self._table(1.0, 1.1, 1.2, 1.3)
        found = pl.verdict(self._joint(1.3), table, [], None, lc.LifecycleSpec())
        assert found["interaction_pct"] == pytest.approx(0.0)
        assert found["decisions_separate"]

    def test_a_superadditive_joint_gain_is_a_positive_interaction(self) -> None:
        from src import lifecycle as lc

        table = self._table(1.0, 1.1, 1.2, 1.4)
        found = pl.verdict(self._joint(1.4), table, [], None, lc.LifecycleSpec())
        assert found["interaction_pct"] == pytest.approx(10.0)
        assert not found["decisions_separate"]

    def test_it_says_which_decision_is_worth_more(self) -> None:
        from src import lifecycle as lc

        table = self._table(1.0, 1.1, 1.2, 1.3)
        found = pl.verdict(self._joint(1.3), table, [], None, lc.LifecycleSpec())
        assert found["plan_beats_allocation"]
        assert found["gain_plan_pct"] > found["gain_allocation_pct"]

    def test_it_flags_a_rate_or_age_sitting_on_the_grid_edge(self) -> None:
        """A corner means the grid, not the model, chose the answer."""
        from src import lifecycle as lc

        plans = pl.plan_grid(["constant_percent"], [0.04, 0.07], [60, 63])
        joint = self._joint(1.3)
        found = pl.verdict(joint, self._table(1.0, 1.1, 1.2, 1.3), plans,
                           None, lc.LifecycleSpec())
        assert found["rate_at_ceiling"]
        assert found["retire_age_at_ceiling"]

    def test_an_unmeasured_search_reports_nothing(self) -> None:
        from src import lifecycle as lc

        assert pl.verdict({}, pd.DataFrame(), [], None,
                          lc.LifecycleSpec()) == {"measured": False}


class TestRuinIsReadOffThePathsNotTheLabel:
    """The prior about which rules can run out is easy to get wrong.

    A *smoothed* percentage rule spends a fraction of a lagged balance, so it
    can overshoot a bad year and deplete even though the unsmoothed version
    cannot. The analysis therefore classifies from the simulated ruin column
    and reports where that disagrees with the prior, rather than trusting the
    rule's description.
    """

    def test_a_flat_zero_ruin_column_gives_no_argmin(self) -> None:
        frame, param = _sweep({"constant_percent": 0.1})
        frame["prob_ruin"] = 0.0
        out = pl.ruin_minimum_by_rule(frame, param, "domestic_share").iloc[0]
        assert not bool(out["ruin_is_possible"])
        assert np.isnan(out["ruin_optimal_domestic_share"])
        # ...and "agree" must not be True by accident on a meaningless argmin.
        assert not bool(out["agree"])

    def test_a_rule_that_does_ruin_gets_a_real_argmin(self) -> None:
        frame, param = _sweep({"constant_real": 0.2}, {"constant_real": 0.2})
        out = pl.ruin_minimum_by_rule(frame, param, "domestic_share").iloc[0]
        assert bool(out["ruin_is_possible"])
        assert out["ruin_optimal_domestic_share"] == pytest.approx(0.2)
        assert bool(out["agree"])

    def test_a_prior_contradicted_by_the_paths_is_flagged(self) -> None:
        frame, param = _sweep({"constant_percent": 0.1})
        frame["prob_ruin"] = 0.01           # it does deplete after all
        out = pl.ruin_minimum_by_rule(frame, param, "domestic_share").iloc[0]
        assert bool(out["ruin_is_possible"])
        assert not bool(out["can_deplete"])
        assert not bool(out["prior_matches_observed"])

    def test_the_smoothed_rule_is_on_the_depleting_side_of_the_prior(self
                                                                    ) -> None:
        """Observed at 0.18% ruin in the production run; the prior said no."""
        assert pl.CAN_DEPLETE["endowment"]
