"""Sequence-of-returns risk, isolated by permutation.

Four properties carry this section and each is pinned here. The permutation
must move every series of a year together, or the shuffle dismantles the
joint draw the block bootstrap exists to preserve. A phase-restricted shuffle
must leave the other phase alone. The decomposition must be the law of total
variance and nothing else. And the control -- shuffling nothing -- must come
back at exactly zero, because a decomposition whose null is not null is
measuring something other than the order.

The fifth is the premise of the whole section: without cash flows, order does
not matter at all. If that invariant ever fails, every number here is
measuring a bug rather than sequence risk.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import bootstrap as bs
from src import lifecycle as lc
from src import sequence as seq


def _paths(n_paths: int = 6, horizon: int = 12, seed: int = 0
           ) -> bs.BootstrapPaths:
    """A chunk of paths whose every series is distinguishable from the others.

    The values are deliberately unequal across fields: a test that all eight
    move together cannot pass by accident if they carry the same numbers.
    """
    rng = np.random.default_rng(seed)
    shape = (n_paths, horizon)
    return bs.BootstrapPaths(
        dom_eq=rng.normal(0.07, 0.18, shape),
        intl_eq=rng.normal(0.06, 0.17, shape),
        bond=rng.normal(0.02, 0.08, shape),
        bill=rng.normal(0.01, 0.04, shape),
        inflation=rng.normal(0.03, 0.04, shape),
        domestic_country=rng.integers(0, 3, shape),
        calendar_index=np.tile(np.arange(horizon), (n_paths, 1)),
        block_id=rng.integers(0, 4, shape),
    )


class _Spec:
    """The three attributes :mod:`src.sequence` reads off a lifecycle spec."""

    def __init__(self, n_working: int = 7, horizon: int = 12) -> None:
        self.n_working = n_working
        self.horizon = horizon
        self.n_retired = horizon - n_working

    @property
    def retirement_slice(self) -> slice:
        return slice(self.n_working, self.horizon)


class TestPhaseBounds:
    def test_none_moves_nothing(self) -> None:
        assert seq.phase_bounds("none", _Spec()) == (0, 0)

    def test_the_phases_partition_the_lifetime(self) -> None:
        spec = _Spec()
        lo_a, hi_a = seq.phase_bounds("accumulation", spec)
        lo_r, hi_r = seq.phase_bounds("retirement", spec)
        assert (lo_a, hi_a) == (0, spec.n_working)
        assert (lo_r, hi_r) == (spec.n_working, spec.horizon)
        # Contiguous and exhaustive: no year belongs to both or to neither.
        assert hi_a == lo_r
        assert seq.phase_bounds("both", spec) == (lo_a, hi_r)

    def test_an_unknown_phase_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown phase"):
            seq.phase_bounds("midlife", _Spec())


class TestPermutation:
    def test_the_control_is_the_identity(self) -> None:
        spec = _Spec()
        order = seq.permutation(50, spec.horizon, "none", spec,
                                np.random.default_rng(0))
        assert np.array_equal(order,
                              np.tile(np.arange(spec.horizon), (50, 1)))

    def test_each_row_is_a_permutation_of_the_years(self) -> None:
        spec = _Spec()
        order = seq.permutation(200, spec.horizon, "both", spec,
                                np.random.default_rng(1))
        assert np.array_equal(np.sort(order, axis=1),
                              np.tile(np.arange(spec.horizon), (200, 1)))

    def test_a_phase_shuffle_leaves_the_other_phase_alone(self) -> None:
        spec = _Spec()
        rng = np.random.default_rng(2)
        acc = seq.permutation(200, spec.horizon, "accumulation", spec, rng)
        ret = seq.permutation(200, spec.horizon, "retirement", spec, rng)
        untouched = np.arange(spec.n_working, spec.horizon)
        assert np.array_equal(acc[:, spec.n_working:],
                              np.tile(untouched, (200, 1)))
        assert np.array_equal(ret[:, :spec.n_working],
                              np.tile(np.arange(spec.n_working), (200, 1)))

    def test_paths_are_shuffled_independently(self) -> None:
        """The decomposition needs variation *within* a bag of returns.

        One shared ordering across paths would put every bit of the dispersion
        into the between-bag term and report a sequence share of zero.
        """
        spec = _Spec()
        order = seq.permutation(500, spec.horizon, "both", spec,
                                np.random.default_rng(3))
        assert len(np.unique(order, axis=0)) > 400

    def test_most_paths_actually_move(self) -> None:
        spec = _Spec()
        order = seq.permutation(1000, spec.horizon, "both", spec,
                               np.random.default_rng(4))
        identity = np.tile(np.arange(spec.horizon), (1000, 1))
        moved = (order != identity).any(axis=1)
        assert moved.mean() > 0.95


class TestPermute:
    def test_every_field_moves_together(self) -> None:
        """The year equity fell must stay the year inflation rose.

        Permuting the assets independently would break the cross-asset
        covariance the block bootstrap is built to preserve, and would measure
        a world nobody lives in.
        """
        paths = _paths()
        spec = _Spec()
        order = seq.permutation(paths.n_paths, spec.horizon, "both", spec,
                                np.random.default_rng(5))
        out = seq.permute(paths, order)
        rows = np.arange(paths.n_paths)[:, None]
        for field in dataclasses.fields(paths):
            original = np.asarray(getattr(paths, field.name))
            assert np.array_equal(np.asarray(getattr(out, field.name)),
                                  original[rows, order]), field.name

    def test_the_calendar_index_records_where_each_year_came_from(self) -> None:
        """``calendar_index`` starts as 0..H-1, so it reads back the ordering."""
        paths = _paths()
        spec = _Spec()
        order = seq.permutation(paths.n_paths, spec.horizon, "both", spec,
                                np.random.default_rng(6))
        out = seq.permute(paths, order)
        assert np.array_equal(out.calendar_index, order)

    def test_the_multiset_of_returns_is_untouched(self) -> None:
        paths = _paths()
        spec = _Spec()
        out = seq.permute(paths, seq.permutation(
            paths.n_paths, spec.horizon, "both", spec,
            np.random.default_rng(7)))
        for key in ("dom_eq", "intl_eq", "bond", "bill", "inflation"):
            assert np.allclose(np.sort(out.series(key), axis=1),
                               np.sort(paths.series(key), axis=1))
            # The mean is the thing a return distribution can describe, and it
            # is exactly what a permutation must not change.
            assert np.allclose(out.series(key).mean(axis=1),
                               paths.series(key).mean(axis=1))

    def test_the_identity_order_returns_the_same_numbers(self) -> None:
        paths = _paths()
        out = seq.permute(paths, np.tile(np.arange(paths.horizon),
                                         (paths.n_paths, 1)))
        for field in dataclasses.fields(paths):
            assert np.array_equal(np.asarray(getattr(out, field.name)),
                                  np.asarray(getattr(paths, field.name)))


class TestOrderInvarianceWithoutFlows:
    """The premise of the section, and the thing that makes it non-trivial.

    A buy-and-hold investor with no contributions and no withdrawals ends at
    the same wealth whatever order the returns arrived in, because the product
    of gross returns is commutative. Sequence risk therefore exists *only*
    through cash flows. If this ever fails, the numbers in ``docs/29`` are
    measuring a defect rather than the order.
    """

    def test_compounded_wealth_does_not_depend_on_the_order(self) -> None:
        paths = _paths(n_paths=200, horizon=40, seed=11)
        spec = _Spec(n_working=20, horizon=40)
        shuffled = seq.permute(paths, seq.permutation(
            paths.n_paths, spec.horizon, "both", spec,
            np.random.default_rng(12)))
        before = np.prod(1.0 + paths.dom_eq, axis=1)
        after = np.prod(1.0 + shuffled.dom_eq, axis=1)
        assert np.allclose(before, after, rtol=0.0, atol=1e-12)

    def test_a_single_contribution_breaks_the_invariance(self) -> None:
        """The contrast that shows the test above is not vacuous."""
        returns = np.array([[0.5, -0.3], [0.5, -0.3]])
        flipped = returns[:, ::-1]
        # One unit in at the start, one more after the first year.
        def _terminal(r: np.ndarray) -> np.ndarray:
            w = np.ones(r.shape[0])
            for h in range(r.shape[1]):
                w = (w + 1.0) * (1.0 + r[:, h])
            return w
        assert not np.allclose(_terminal(returns), _terminal(flipped))


class TestDecompose:
    def test_it_is_the_law_of_total_variance(self) -> None:
        rng = np.random.default_rng(20)
        values = rng.normal(1.0, 0.4, (400, 6))
        out = seq.decompose(values)
        assert (out["variance_sequence"] + out["variance_level"]
                == pytest.approx(out["sd_total"] ** 2))
        assert out["sequence_share"] == pytest.approx(
            out["variance_sequence"]
            / (out["variance_sequence"] + out["variance_level"]))

    def test_identical_columns_are_pure_level_risk(self) -> None:
        """No variation within a bag means no sequence risk, by construction."""
        column = np.random.default_rng(21).normal(1.0, 0.3, (300, 1))
        out = seq.decompose(np.tile(column, (1, 5)))
        assert out["sequence_share"] == pytest.approx(0.0, abs=1e-12)
        assert out["sd_sequence"] == pytest.approx(0.0, abs=1e-12)

    def test_identical_rows_are_pure_sequence_risk(self) -> None:
        """Every bag with the same mean leaves only the ordering to explain."""
        row = np.random.default_rng(22).normal(1.0, 0.3, (1, 8))
        out = seq.decompose(np.tile(row, (300, 1)))
        assert out["sequence_share"] == pytest.approx(1.0)

    def test_a_single_ordering_cannot_estimate_sequence_risk(self) -> None:
        """With one column the within-bag term is unidentified, not zero-ish."""
        out = seq.decompose(np.random.default_rng(23).normal(1.0, 0.3, (50, 1)))
        assert out["variance_sequence"] == 0.0
        assert out["sequence_share"] == pytest.approx(0.0)

    def test_a_degenerate_outcome_does_not_divide_by_zero(self) -> None:
        out = seq.decompose(np.ones((10, 3)))
        assert out["sequence_share"] == 0.0

    def test_the_shape_contract_is_enforced(self) -> None:
        with pytest.raises(ValueError, match=r"\(n_paths, n_reps\)"):
            seq.decompose(np.ones(10))


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def _row(phase: str, strategy: str, share: float, cec: float,
         ruin: float = 0.1, **extra) -> dict:
    row = {"phase": phase, "strategy": strategy, "sequence_share": share,
           "cec": cec, "prob_ruin": ruin, "sd_sequence": share,
           "sd_level": 1.0 - share, "sd_total": 1.0,
           "wealth_sequence_share": share, "mean_retirement_consumption": cec,
           "n_paths": 100, "n_reps": 4}
    row.update(extra)
    return row


class TestVerdict:
    def test_a_clean_control_is_recognised(self) -> None:
        found = seq.verdict(_frame([
            _row("none", "eq", 0.0, 1.0), _row("both", "eq", 0.4, 0.98)]),
            "eq", ["none", "both"])
        assert found["control_is_clean"]

    def test_a_dirty_control_is_caught(self) -> None:
        """The check that stops the study reporting noise as ordering."""
        found = seq.verdict(_frame([
            _row("none", "eq", 1e-4, 1.0), _row("both", "eq", 0.4, 0.98)]),
            "eq", ["none", "both"])
        assert not found["control_is_clean"]
        assert found["control_share"] == pytest.approx(1e-4)

    def test_it_classifies_where_the_risk_sits_rather_than_asserting(self
                                                                    ) -> None:
        rows = [_row("none", "eq", 0.0, 1.0), _row("accumulation", "eq", 0.1, 1.0),
                _row("retirement", "eq", 0.3, 1.0), _row("both", "eq", 0.4, 1.0)]
        assert seq.verdict(_frame(rows), "eq")["retirement_dominates"]
        rows[1]["sequence_share"], rows[2]["sequence_share"] = 0.3, 0.1
        found = seq.verdict(_frame(rows), "eq")
        assert not found["retirement_dominates"]
        assert found["retirement_over_accumulation"] == pytest.approx(1 / 3)

    def test_the_majority_finding_is_read_off_the_number(self) -> None:
        rows = [_row("none", "eq", 0.0, 1.0), _row("both", "eq", 0.51, 1.0)]
        assert seq.verdict(_frame(rows), "eq", ["none", "both"])[
            "ordering_is_most_of_the_risk"]
        rows[1]["sequence_share"] = 0.49
        assert not seq.verdict(_frame(rows), "eq", ["none", "both"])[
            "ordering_is_most_of_the_risk"]

    def test_the_cost_of_ordering_is_a_ratio_of_certainty_equivalents(self
                                                                     ) -> None:
        found = seq.verdict(_frame([
            _row("none", "eq", 0.0, 1.0), _row("both", "eq", 0.4, 0.95)]),
            "eq", ["none", "both"])
        assert found["cec_cost_of_ordering_pct"] == pytest.approx(-5.0)

    def test_an_absent_strategy_reports_nothing_rather_than_guessing(self
                                                                     ) -> None:
        assert seq.verdict(_frame([_row("none", "eq", 0.0, 1.0)]),
                           "missing") == {"measured": False}


class TestRankingHolds:
    def test_the_lead_is_reported_under_every_phase(self) -> None:
        rows = []
        for phase, a, b in (("none", 1.10, 1.00), ("both", 1.02, 1.00)):
            rows += [_row(phase, "challenger", 0.0, a),
                     _row(phase, "incumbent", 0.0, b)]
        out = seq.ranking_holds(_frame(rows), "challenger", "incumbent")
        assert list(out["phase"]) == ["none", "both"]
        assert out["lead_pct"].tolist() == pytest.approx([10.0, 2.0])
        assert set(out["winner"]) == {"challenger"}

    def test_a_reversal_is_reported_as_one(self) -> None:
        rows = [_row("both", "challenger", 0.0, 0.98),
                _row("both", "incumbent", 0.0, 1.00)]
        out = seq.ranking_holds(_frame(rows), "challenger", "incumbent")
        assert out["lead_pct"].iloc[0] < 0
        assert out["winner"].iloc[0] == "incumbent"

    def test_a_phase_missing_the_pair_is_skipped(self) -> None:
        rows = [_row("none", "challenger", 0.0, 1.0),
                _row("both", "challenger", 0.0, 1.0),
                _row("both", "incumbent", 0.0, 1.0)]
        out = seq.ranking_holds(_frame(rows), "challenger", "incumbent")
        assert list(out["phase"]) == ["both"]


def _rule_rows(rule: str, label: str, acc: float, ret: float,
               ruin_none: float = 0.10, ruin_both: float = 0.10) -> list[dict]:
    return [
        dict(_row("none", "eq", 0.0, 1.0, ruin_none), rule=rule, rule_label=label),
        dict(_row("accumulation", "eq", acc, 1.0), rule=rule, rule_label=label),
        dict(_row("retirement", "eq", ret, 1.0), rule=rule, rule_label=label),
        dict(_row("both", "eq", acc + ret, 0.98, ruin_both),
             rule=rule, rule_label=label),
    ]


class TestRuleComparison:
    def test_the_ratio_is_retirement_over_accumulation(self) -> None:
        out = seq.rule_comparison(
            _frame(_rule_rows("constant_real", "Fixed real", 0.20, 0.02)), "eq")
        assert out["retirement_over_accumulation"].iloc[0] == pytest.approx(0.1)

    def test_a_rule_that_passes_returns_through_relocates_the_risk(self) -> None:
        rows = (_rule_rows("constant_real", "Fixed real", 0.20, 0.02)
                + _rule_rows("constant_percent", "Percentage", 0.16, 0.14))
        found = seq.rule_verdict(seq.rule_comparison(_frame(rows), "eq"))
        assert found["rule_relocates_the_risk"]
        assert found["fixed_real_ratio"] == pytest.approx(0.1)
        assert found["percentage_ratio"] == pytest.approx(0.875)

    def test_a_ratio_that_barely_moves_is_not_called_a_relocation(self) -> None:
        rows = (_rule_rows("constant_real", "Fixed real", 0.20, 0.02)
                + _rule_rows("constant_percent", "Percentage", 0.20, 0.03))
        found = seq.rule_verdict(seq.rule_comparison(_frame(rows), "eq"))
        assert not found["rule_relocates_the_risk"]

    def test_the_ruin_trade_is_classified_not_asserted(self) -> None:
        """A fixed real rule should pay in ruin what it refuses to pay in
        consumption -- but the doc only says so when the numbers do."""
        rows = (_rule_rows("constant_real", "Fixed real", 0.20, 0.02,
                           ruin_none=0.10, ruin_both=0.12)
                + _rule_rows("constant_percent", "Percentage", 0.16, 0.14,
                             ruin_none=0.0, ruin_both=0.0))
        found = seq.rule_verdict(seq.rule_comparison(_frame(rows), "eq"))
        assert found["fixed_real_trades_ruin_for_smoothness"]
        assert found["fixed_real_ruin_cost_pp"] == pytest.approx(2.0)
        assert found["percentage_ruin_cost_pp"] == pytest.approx(0.0)

        reversed_rows = (_rule_rows("constant_real", "Fixed real", 0.20, 0.02,
                                    ruin_none=0.10, ruin_both=0.10)
                         + _rule_rows("constant_percent", "Percentage", 0.16,
                                      0.14, ruin_none=0.0, ruin_both=0.05))
        assert not seq.rule_verdict(seq.rule_comparison(
            _frame(reversed_rows), "eq"))["fixed_real_trades_ruin_for_smoothness"]

    def test_an_empty_comparison_reports_nothing(self) -> None:
        assert seq.rule_verdict(pd.DataFrame()) == {"measured": False}

    def test_the_default_rules_span_insulated_to_pass_through(self) -> None:
        keys = [k for k, _ in seq.DEFAULT_RULES]
        assert keys[0] == "constant_real"
        assert keys[-1] == "constant_percent"


class TestEndToEnd:
    """The whole machinery on the toy panel, including the zero control."""

    @staticmethod
    def _run(toy_panel, toy_config, phase: str, n_reps: int = 3):
        spec = lc.spec_from_config(toy_config)
        strategies = lc.build_strategies(toy_config, spec)
        sampler = bs.from_config(toy_panel, toy_config)
        return seq.run(sampler, strategies, spec, toy_config, 120, 60,
                       phase, n_reps, seed=99, income_seed=7)

    def test_shuffling_nothing_reproduces_every_path_bit_for_bit(
            self, toy_panel, toy_config) -> None:
        """The control the step refuses to run without.

        Every ordering is the identity, so every replication must reproduce
        the same lifetime exactly. That is asserted on the outcomes rather
        than on the variance, because the variance of identical floats is not
        exactly zero -- ``(v + v + v) / 3`` need not round back to ``v`` -- so
        the share is checked at the same tolerance the step enforces.
        """
        spec = lc.spec_from_config(toy_config)
        results = self._run(toy_panel, toy_config, "none")
        for fields in results.values():
            for name in ("utility", "consumption", "wealth", "ruin"):
                block = fields[name]
                assert np.array_equal(block, block[:, [0]] * np.ones_like(block))
        frame = seq.summarise(results, "none", spec, toy_config)
        assert (frame["sequence_share"] < 1e-9).all()
        found = seq.verdict(frame, "all_equity", ["none"])
        assert found["control_is_clean"]

    def test_shuffling_the_lifetime_produces_real_sequence_risk(
            self, toy_panel, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        frame = seq.summarise(
            self._run(toy_panel, toy_config, "both"), "both", spec, toy_config)
        assert (frame["sequence_share"] > 0).all()
        assert (frame["sequence_share"] <= 1.0).all()

    def test_shuffling_only_the_retired_years_leaves_wealth_untouched(
            self, toy_panel, toy_config) -> None:
        """Wealth at retirement is set before a retirement-phase shuffle bites.

        This is the sharpest available check that the phase bounds are wired
        to the right years: get them off by one and this number moves.
        """
        spec = lc.spec_from_config(toy_config)
        held = self._run(toy_panel, toy_config, "retirement")
        for fields in held.values():
            wealth = fields["wealth"]
            assert np.array_equal(wealth, wealth[:, [0]] * np.ones_like(wealth))
        frame = seq.summarise(held, "retirement", spec, toy_config)
        assert (frame["wealth_sequence_share"] < 1e-9).all()
        # ...while shuffling the working years moves it by a visible margin.
        moved = seq.summarise(self._run(toy_panel, toy_config, "accumulation"),
                              "accumulation", spec, toy_config)
        assert (moved["wealth_sequence_share"] > 0.01).all()

    def test_the_bag_of_returns_is_held_fixed_across_replications(
            self, toy_panel, toy_config) -> None:
        """Two phases must draw the same lifetimes, or nothing is comparable.

        The sampler and the income draw are re-seeded identically for every
        replication and every phase; only the order differs. The identity
        control is the observable consequence: its outcome is the unshuffled
        lifetime, so it must match across independent calls.
        """
        first = self._run(toy_panel, toy_config, "none", n_reps=1)
        second = self._run(toy_panel, toy_config, "none", n_reps=2)
        for key, fields in first.items():
            assert np.allclose(fields["utility"][:, 0],
                               second[key]["utility"][:, 0])

    def test_the_certainty_equivalent_matches_the_utility_layer(
            self, toy_panel, toy_config) -> None:
        """``certainty_equivalent`` must not reimplement the utility module."""
        from src import utility as ut

        spec = lc.spec_from_config(toy_config)
        results = self._run(toy_panel, toy_config, "none", n_reps=1)
        key = next(iter(results))
        utility = results[key]["utility"][:, 0]
        util_cfg = toy_config["utility"]
        weights = ut.discount_weights(
            spec.n_retired, float(util_cfg["discount_factor"]),
            float(util_cfg["bequest_weight"]),
            bool(util_cfg["bequest_enabled"]))
        expected = ut._inverse_felicity(
            utility.mean() / weights.sum(),
            float(util_cfg["baseline_risk_aversion"]))
        assert seq.certainty_equivalent(
            results[key]["utility"][:, :1], spec, toy_config
        ) == pytest.approx(float(expected))
