"""Tests for endogenous retirement timing."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src import bootstrap as bs
from src import lifecycle as lc
from src import retirement as rt
from src import spending as sp


@pytest.fixture()
def setup(toy_panel, toy_config):
    spec = lc.spec_from_config(toy_config)
    strategy = lc.build_strategies(toy_config, spec)["all_equity"]
    paths = bs.from_config(toy_panel, toy_config,
                           horizon_years=spec.horizon).sample(400, chunk_size=200)
    rng = np.random.default_rng(5)
    shocks = lc.draw_income_shocks(400, spec.horizon, rng)
    income = rt.extended_income(spec, 400, shocks=shocks)
    return toy_config, spec, strategy, paths, income, shocks


def state(age, wealth, income, working=None) -> rt.RetirementState:
    wealth = np.asarray(wealth, dtype=float)
    income = np.asarray(income, dtype=float)
    if working is None:
        working = np.ones(wealth.size, dtype=bool)
    return rt.RetirementState(age=age, wealth=wealth, current_income=income,
                              still_working=np.asarray(working))


class TestFixedAgeRule:
    def test_does_not_retire_before_the_age(self) -> None:
        rule = rt.FixedAgeRule(age=63)
        assert not rule.should_retire(state(62, [1e9], [1.0])).any()

    def test_retires_everyone_on_the_age(self) -> None:
        rule = rt.FixedAgeRule(age=63)
        assert rule.should_retire(state(63, [0.0, 1e9], [1.0, 1.0])).all()

    def test_ignores_paths_already_retired(self) -> None:
        rule = rt.FixedAgeRule(age=63)
        out = rule.should_retire(state(70, [1.0, 1.0], [1.0, 1.0],
                                       working=[False, True]))
        np.testing.assert_array_equal(out, [False, True])


class TestWealthMultipleRule:
    def test_waits_for_the_window_to_open(self) -> None:
        rule = rt.WealthMultipleRule(multiple=25, min_age=60, max_age=70)
        assert not rule.should_retire(state(59, [1e9], [1.0])).any()

    def test_retires_once_the_multiple_is_reached(self) -> None:
        rule = rt.WealthMultipleRule(multiple=25, min_age=60, max_age=70)
        out = rule.should_retire(state(62, [24.0, 26.0], [1.0, 1.0]))
        np.testing.assert_array_equal(out, [False, True])

    def test_scales_the_target_with_income(self) -> None:
        rule = rt.WealthMultipleRule(multiple=25, min_age=60, max_age=70)
        out = rule.should_retire(state(62, [50.0, 50.0], [1.0, 3.0]))
        np.testing.assert_array_equal(out, [True, False])

    def test_forces_retirement_when_the_window_closes(self) -> None:
        rule = rt.WealthMultipleRule(multiple=25, min_age=60, max_age=70)
        assert rule.should_retire(state(70, [0.0], [1.0])).all()

    def test_a_higher_multiple_retires_people_later(self, setup) -> None:
        cfg, spec, strategy, paths, income, _ = setup
        low = rt.simulate_flexible(paths, strategy, spec, income,
                                   rt.WealthMultipleRule(1.0, spec.age_start,
                                                         spec.age_death - 1))
        high = rt.simulate_flexible(paths, strategy, spec, income,
                                    rt.WealthMultipleRule(50.0, spec.age_start,
                                                          spec.age_death - 1))
        assert high.retire_age.mean() >= low.retire_age.mean()


class TestRegistry:
    def test_builds_registered_rules(self) -> None:
        assert isinstance(rt.build("fixed_age", age=60), rt.FixedAgeRule)
        assert isinstance(rt.build("wealth_multiple", multiple=20),
                          rt.WealthMultipleRule)

    def test_rejects_an_unknown_rule(self) -> None:
        with pytest.raises(ValueError, match="unknown retirement rule"):
            rt.build("when_bored")


class TestExtendedIncome:
    def test_covers_the_whole_horizon(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        assert income.shape == (400, spec.horizon)

    def test_prefix_matches_the_fixed_date_profile(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        standard = lc.simulate_income(spec, 400, shocks=shocks)
        np.testing.assert_allclose(income[:, :spec.n_working], standard,
                                   rtol=0, atol=0)


class TestWorkingIncomeFloor:
    def test_defaults_to_no_floor(self) -> None:
        assert lc.LifecycleSpec().working_income_floor == 0.0

    def test_raises_the_left_tail_of_income(self) -> None:
        spec = lc.LifecycleSpec()
        floored = dataclasses.replace(spec, working_income_floor=0.5)
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(1)
        plain = lc.simulate_income(spec, 5000, rng_a)
        with_floor = lc.simulate_income(floored, 5000, rng_b)
        assert with_floor.min() > plain.min()
        assert (with_floor >= plain - 1e-12).all()

    def test_a_zero_floor_changes_nothing(self) -> None:
        spec = lc.LifecycleSpec()
        zero = dataclasses.replace(spec, working_income_floor=0.0)
        a = lc.simulate_income(spec, 500, np.random.default_rng(2))
        b = lc.simulate_income(zero, 500, np.random.default_rng(2))
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


class TestFlexibleSimulator:
    def test_reproduces_the_fixed_date_engine(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        standard = lc.simulate_income(spec, 400, shocks=shocks)
        fixed = lc.simulate(paths, strategy, spec, standard)
        flexible = rt.simulate_flexible(paths, strategy, spec, income,
                                        rt.FixedAgeRule(age=spec.age_retire))
        # Career-average earnings accumulate sequentially here and by a
        # pairwise mean there, so social security -- and through it retirement
        # consumption -- agrees to floating point rather than bit for bit.
        np.testing.assert_allclose(flexible.consumption, fixed.consumption,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(flexible.wealth, fixed.wealth,
                                   rtol=0, atol=0)
        np.testing.assert_allclose(flexible.bequest, fixed.bequest,
                                   rtol=0, atol=0)
        np.testing.assert_array_equal(flexible.ruin, fixed.ruin)
        assert (flexible.retire_age == spec.age_retire).all()
        assert (flexible.years_worked == spec.n_working).all()

    def test_honours_a_custom_spending_rule(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        rule = sp.ConstantPercentRule(rate=0.06)
        standard = lc.simulate_income(spec, 400, shocks=shocks)
        fixed = lc.simulate(paths, strategy, spec, standard, rule)
        flexible = rt.simulate_flexible(paths, strategy, spec, income,
                                        rt.FixedAgeRule(age=spec.age_retire),
                                        rule)
        np.testing.assert_allclose(flexible.consumption, fixed.consumption,
                                   rtol=1e-12, atol=1e-12)

    def test_wealth_never_goes_negative(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income,
            rt.WealthMultipleRule(5.0, spec.age_start + 1, spec.age_death - 1))
        assert (outcome.wealth >= -1e-12).all()
        assert np.isfinite(outcome.consumption).all()
        assert (outcome.consumption > 0).all()

    def test_retire_age_and_years_worked_agree(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income,
            rt.WealthMultipleRule(10.0, spec.age_start + 2, spec.age_death - 1))
        np.testing.assert_array_equal(
            outcome.retire_age - spec.age_start, outcome.years_worked)

    def test_rejects_income_that_stops_at_the_nominal_retirement_age(
            self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        short = lc.simulate_income(spec, 400, shocks=shocks)
        with pytest.raises(ValueError, match="every year"):
            rt.simulate_flexible(paths, strategy, spec, short,
                                 rt.FixedAgeRule(age=spec.age_retire))


class TestAnalysis:
    def test_evaluate_reports_timing_and_preference_columns(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(paths, strategy, spec, income,
                                       rt.FixedAgeRule(age=spec.age_retire))
        row = rt.evaluate(outcome, cfg, spec, gammas=[5.0])
        assert row["cec_gamma5"] > 0
        assert row["median_retire_age"] == pytest.approx(spec.age_retire)
        assert 0.0 <= row["prob_ruin"] <= 1.0

    def test_lottery_returns_deciles_and_a_decomposition(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income,
            rt.WealthMultipleRule(8.0, spec.age_start + 2, spec.age_death - 1))
        table, stats = rt.retirement_lottery(outcome, spec, before=2, after=2,
                                             n_buckets=5)
        assert len(table) == 5
        assert table["mean_window_return"].is_monotonic_increasing
        assert 0.0 <= stats["r2_retirement_window"] <= 1.0
        assert 0.0 <= stats["r2_whole_lifetime"] <= 1.0

    def test_better_windows_deliver_more_consumption(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(paths, strategy, spec, income,
                                       rt.FixedAgeRule(age=spec.age_retire))
        table, _ = rt.retirement_lottery(outcome, spec, before=2, after=2,
                                         n_buckets=4)
        assert (table["median_retirement_consumption"].iloc[-1]
                > table["median_retirement_consumption"].iloc[0])

    def test_bull_market_test_reports_both_halves(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income,
            rt.WealthMultipleRule(8.0, spec.age_start + 2, spec.age_death - 1))
        stats = rt.bull_market_test(outcome, spec, before=2)
        for key in ("corr_runup_vs_subsequent_return",
                    "corr_retire_age_vs_runup"):
            value = stats[key]
            assert np.isnan(value) or -1.0 <= value <= 1.0
        assert np.isfinite(stats["mean_runup_early_retirees"])

    def test_bull_market_test_is_clean_when_everyone_retires_together(
            self, setup) -> None:
        # A fixed-age rule leaves the retirement-age correlations undefined by
        # construction; they should come back as NaN, not as a numpy warning.
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(paths, strategy, spec, income,
                                       rt.FixedAgeRule(age=spec.age_retire))
        with np.errstate(all="raise"):
            stats = rt.bull_market_test(outcome, spec, before=2)
        assert np.isnan(stats["corr_retire_age_vs_runup"])
        assert np.isfinite(stats["mean_runup_early_retirees"])

    def test_safe_corr_handles_a_constant_series(self) -> None:
        assert np.isnan(rt._safe_corr(np.ones(10), np.arange(10.0)))
        assert rt._safe_corr(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
