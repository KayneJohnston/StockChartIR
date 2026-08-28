"""Tests for savings-rate rules and the shape optimiser."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import bootstrap as bs
from src import lifecycle as lc
from src import retirement as rt
from src import saving as sav


@pytest.fixture()
def setup(toy_panel, toy_config):
    spec = lc.spec_from_config(toy_config)
    strategy = lc.build_strategies(toy_config, spec)["all_equity"]
    paths = bs.from_config(toy_panel, toy_config,
                           horizon_years=spec.horizon).sample(300, chunk_size=150)
    shocks = lc.draw_income_shocks(300, spec.horizon, np.random.default_rng(4))
    income = rt.extended_income(spec, 300, shocks=shocks)
    return toy_config, spec, strategy, paths, income, shocks


def state(age=40, year=15, wealth=(10.0,), income=(1.0,), last_return=(0.05,)):
    wealth = np.asarray(wealth, dtype=float)
    return sav.SavingState(age=age, year=year, wealth=wealth,
                           current_income=np.asarray(income, dtype=float),
                           last_return=np.asarray(last_return, dtype=float),
                           still_working=np.ones(wealth.size, dtype=bool))


class TestSavingState:
    def test_wealth_to_income_ratio(self) -> None:
        assert state(wealth=[20.0], income=[2.0]).wealth_to_income[0] \
            == pytest.approx(10.0)

    def test_zero_income_does_not_divide_by_zero(self) -> None:
        assert np.isfinite(state(wealth=[1.0], income=[0.0]).wealth_to_income).all()


class TestConstantAndProfile:
    def test_constant_rate_ignores_state(self) -> None:
        rule = sav.ConstantRateRule(rate_value=0.12)
        np.testing.assert_allclose(rule.rate(state(wealth=[0.0, 1e6])), 0.12)

    def test_age_profile_indexes_by_year(self) -> None:
        schedule = np.linspace(0.0, 0.3, 10)
        rule = sav.AgeProfileRule(schedule)
        assert rule.rate(state(year=3))[0] == pytest.approx(schedule[3])

    def test_age_profile_clamps_past_the_end(self) -> None:
        rule = sav.AgeProfileRule(np.array([0.1, 0.2]))
        assert rule.rate(state(year=99))[0] == pytest.approx(0.2)


class TestOnTrackRule:
    def _rule(self, k=0.01, base=0.10, target=10.0, horizon=68):
        return sav.OnTrackRule(target=np.full(horizon, target),
                               base=np.full(horizon, base), sensitivity=k)

    def test_saves_more_when_behind(self) -> None:
        rule = self._rule()
        behind = rule.rate(state(wealth=[5.0], income=[1.0]))[0]
        assert behind > 0.10

    def test_saves_less_when_ahead(self) -> None:
        rule = self._rule()
        ahead = rule.rate(state(wealth=[20.0], income=[1.0]))[0]
        assert ahead < 0.10

    def test_matches_the_base_rate_exactly_on_target(self) -> None:
        rule = self._rule()
        assert rule.rate(state(wealth=[10.0], income=[1.0]))[0] \
            == pytest.approx(0.10)

    def test_zero_sensitivity_is_the_base_profile(self) -> None:
        rule = self._rule(k=0.0)
        np.testing.assert_allclose(
            rule.rate(state(wealth=[0.0, 1e4])), 0.10)

    def test_respects_the_floor_and_cap(self) -> None:
        rule = sav.OnTrackRule(target=np.full(68, 1e6), base=np.full(68, 0.10),
                               sensitivity=1.0, floor=0.0, cap=0.25)
        assert rule.rate(state(wealth=[0.0]))[0] == pytest.approx(0.25)
        low = sav.OnTrackRule(target=np.zeros(68), base=np.full(68, 0.10),
                              sensitivity=1.0, floor=0.02, cap=0.4)
        assert low.rate(state(wealth=[1e6]))[0] == pytest.approx(0.02)


class TestReturnResponsiveRule:
    def test_saves_more_after_a_bad_year(self) -> None:
        rule = sav.ReturnResponsiveRule(base=np.full(68, 0.10), sensitivity=0.5)
        assert rule.rate(state(last_return=[-0.20]))[0] > 0.10

    def test_saves_less_after_a_good_year(self) -> None:
        rule = sav.ReturnResponsiveRule(base=np.full(68, 0.10), sensitivity=0.5)
        assert rule.rate(state(last_return=[0.20]))[0] < 0.10

    def test_zero_sensitivity_is_the_base_profile(self) -> None:
        rule = sav.ReturnResponsiveRule(base=np.full(68, 0.10), sensitivity=0.0)
        assert rule.rate(state(last_return=[-0.5]))[0] == pytest.approx(0.10)


class TestRegistry:
    def test_builds_every_registered_rule(self) -> None:
        for key in sav.REGISTRY:
            assert isinstance(sav.build(key), sav.SavingRule)

    def test_rejects_an_unknown_rule(self) -> None:
        with pytest.raises(ValueError, match="unknown saving rule"):
            sav.build("save_everything")


class TestNormalisation:
    def test_hits_the_target_average(self) -> None:
        out = sav.normalise_to_mean(np.array([1.0, 2.0, 3.0, 4.0]), 4, 0.10,
                                    0.0, 0.40)
        assert out[:4].mean() == pytest.approx(0.10, abs=1e-9)

    def test_preserves_the_relative_shape(self) -> None:
        out = sav.normalise_to_mean(np.array([1.0, 2.0, 3.0, 4.0]), 4, 0.10,
                                    0.0, 0.40)
        assert out[1] / out[0] == pytest.approx(2.0)

    def test_still_hits_the_target_when_clipping_binds(self) -> None:
        out = sav.normalise_to_mean(np.array([1.0, 1.0, 1.0, 50.0]), 4, 0.10,
                                    0.0, 0.15)
        assert out.max() <= 0.15 + 1e-12
        assert out[:4].mean() == pytest.approx(0.10, abs=1e-3)

    def test_flat_multipliers_give_a_flat_schedule(self) -> None:
        out = sav.normalise_to_mean(np.ones(10), 10, 0.12, 0.0, 0.4)
        np.testing.assert_allclose(out, 0.12)


class TestSimulatorIntegration:
    def test_constant_rule_reproduces_the_fixed_engine(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        standard = lc.simulate_income(spec, 300, shocks=shocks)
        fixed = lc.simulate(paths, strategy, spec, standard)
        flexible = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(spec.age_retire),
            saving=sav.ConstantRateRule(rate_value=spec.savings_rate))
        np.testing.assert_allclose(flexible.consumption, fixed.consumption,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(flexible.wealth, fixed.wealth, rtol=0, atol=0)

    def test_default_saving_matches_an_explicit_constant_rule(self, setup
                                                              ) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        a = rt.simulate_flexible(paths, strategy, spec, income,
                                 rt.FixedAgeRule(spec.age_retire))
        b = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(spec.age_retire),
            saving=sav.ConstantRateRule(rate_value=spec.savings_rate))
        np.testing.assert_allclose(a.consumption, b.consumption, rtol=0, atol=0)

    def test_reports_the_realised_savings_rate(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(spec.age_retire),
            saving=sav.ConstantRateRule(rate_value=0.2))
        np.testing.assert_allclose(outcome.mean_savings_rate, 0.2)
        assert (outcome.total_saved > 0).all()

    def test_a_higher_rate_builds_more_wealth(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        low, high = (rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(spec.age_retire),
            saving=sav.ConstantRateRule(rate_value=r)) for r in (0.05, 0.25))
        assert np.median(high.wealth_at_retirement) > \
            np.median(low.wealth_at_retirement)


class TestOptimiser:
    def test_shape_search_holds_the_average_fixed(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup

        def objective(schedule):
            outcome = rt.simulate_flexible(
                paths, strategy, spec, income,
                rt.FixedAgeRule(spec.age_retire),
                saving=sav.AgeProfileRule(np.asarray(schedule)))
            return float(np.median(outcome.wealth_at_retirement))

        schedule, best = sav.optimise_shape_at_fixed_mean(
            objective, spec.n_working, spec.horizon, 0.10,
            [0.5, 1.0, 2.0], n_sweeps=1)
        assert schedule[:spec.n_working].mean() == pytest.approx(0.10, abs=1e-6)
        assert np.isfinite(best)

    def test_shape_search_improves_on_the_flat_start(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup

        def objective(schedule):
            outcome = rt.simulate_flexible(
                paths, strategy, spec, income,
                rt.FixedAgeRule(spec.age_retire),
                saving=sav.AgeProfileRule(np.asarray(schedule)))
            return float(np.median(outcome.wealth_at_retirement))

        flat = np.full(spec.horizon, 0.10)
        before = objective(flat)
        _, after = sav.optimise_shape_at_fixed_mean(
            objective, spec.n_working, spec.horizon, 0.10,
            [0.5, 1.0, 2.0], n_sweeps=1)
        assert after >= before

    def test_matched_rate_comparison_strips_out_saving_more(self) -> None:
        summary = pd.DataFrame([
            {"variant": "Constant 5%", "mean_savings_rate": 0.05, "cec": 1.00},
            {"variant": "Constant 15%", "mean_savings_rate": 0.15, "cec": 1.20},
            # Saves like a 10% constant rate but scores above the interpolated
            # 1.10, so the excess is shape rather than level.
            {"variant": "Shaped", "mean_savings_rate": 0.10, "cec": 1.15},
        ])
        out = sav.matched_rate_comparison(summary, "cec")
        row = out.iloc[0]
        assert row["matched_constant_rate_cec"] == pytest.approx(1.10)
        assert row["value_of_shape_pct"] == pytest.approx(
            (1.15 / 1.10 - 1.0) * 100.0)
        assert not bool(row["extrapolated"])

    def test_matched_rate_comparison_flags_extrapolation(self) -> None:
        summary = pd.DataFrame([
            {"variant": "Constant 5%", "mean_savings_rate": 0.05, "cec": 1.00},
            {"variant": "Constant 15%", "mean_savings_rate": 0.15, "cec": 1.20},
            {"variant": "Shaped", "mean_savings_rate": 0.30, "cec": 1.15},
        ])
        assert bool(sav.matched_rate_comparison(summary, "cec")
                    .iloc[0]["extrapolated"])


class TestTargetAndDeviation:
    def test_target_uses_income_not_consumption(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(spec.age_retire),
            saving=sav.ConstantRateRule(rate_value=0.10))
        target = sav.wealth_to_income_target(outcome, income, spec.horizon)
        expected = np.median(outcome.wealth[:, 5] / income[:, 5])
        assert target[5] == pytest.approx(expected)

    def test_target_rises_with_age(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(spec.age_retire),
            saving=sav.ConstantRateRule(rate_value=0.10))
        target = sav.wealth_to_income_target(outcome, income, spec.horizon)
        assert target[spec.n_working - 1] > target[1]

    def test_deviation_profile_has_a_row_per_working_year(self, setup) -> None:
        cfg, spec, strategy, paths, income, shocks = setup

        def objective(schedule):
            outcome = rt.simulate_flexible(
                paths, strategy, spec, income,
                rt.FixedAgeRule(spec.age_retire),
                saving=sav.AgeProfileRule(np.asarray(schedule)))
            return float(np.median(outcome.wealth_at_retirement))

        frame = sav.deviation_profile(objective, np.full(spec.horizon, 0.10),
                                      spec.n_working, spec.ages)
        assert len(frame) == spec.n_working
        # A flat schedule reset to its own average changes nothing.
        np.testing.assert_allclose(frame["cost_of_resetting_bp"], 0.0,
                                   atol=1e-6)
