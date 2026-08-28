"""Tests for the accumulation-signal study.

The rules here are all small pure functions of state, so most of these pin
down sign conventions and boundary behaviour: a rule that responds the wrong
way round to a shortfall would still produce a plausible-looking sweep, and
the sweeps are expensive enough that nobody would notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import accumulation as acc
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
    return toy_config, spec, strategy, paths, income


def state(*, age=45, year=20, wealth=(5.0,), income=(1.0,), last_return=(0.05,),
          history=None, contributed=None):
    wealth = np.asarray(wealth, dtype=float)
    return sav.SavingState(
        age=age, year=year, wealth=wealth,
        current_income=np.asarray(income, dtype=float),
        last_return=np.asarray(last_return, dtype=float),
        still_working=np.ones(wealth.size, dtype=bool),
        returns_history=None if history is None
        else np.asarray(history, dtype=float),
        contributed=None if contributed is None
        else np.asarray(contributed, dtype=float))


class TestSavingStateExtensions:
    def test_trailing_return_is_the_geometric_mean(self) -> None:
        history = [[0.10, -0.10, 0.20]]
        got = state(history=history).trailing_return(3)[0]
        expected = np.expm1(np.log1p([0.10, -0.10, 0.20]).mean())
        assert got == pytest.approx(expected)

    def test_trailing_return_uses_only_the_last_n_years(self) -> None:
        history = [[5.0, 0.10, 0.10]]
        assert state(history=history).trailing_return(2)[0] \
            == pytest.approx(0.10)

    def test_trailing_return_truncates_to_available_history(self) -> None:
        history = [[0.20]]
        assert state(history=history).trailing_return(10)[0] \
            == pytest.approx(0.20)

    def test_trailing_return_is_zero_without_history(self) -> None:
        assert state().trailing_return(5)[0] == 0.0
        assert state(history=np.zeros((1, 0))).trailing_return(5)[0] == 0.0

    def test_investment_gain_is_wealth_over_contributions(self) -> None:
        got = state(wealth=[12.0], contributed=[10.0]).investment_gain[0]
        assert got == pytest.approx(0.2)

    def test_investment_gain_is_zero_before_any_contribution(self) -> None:
        assert state(wealth=[5.0], contributed=[0.0]).investment_gain[0] == 0.0

    def test_history_slice_never_includes_the_current_year(self, setup) -> None:
        """The rule must not be handed a return it has not lived through yet."""
        cfg, spec, strategy, paths, income = setup
        seen: list[tuple[int, int]] = []

        class Spy(sav.SavingRule):
            key, label = "spy", "Spy"

            def rate(self, s: sav.SavingState) -> np.ndarray:
                seen.append((s.year, 0 if s.returns_history is None
                             else s.returns_history.shape[1]))
                return np.full(s.wealth.shape, 0.10)

        rt.simulate_flexible(paths, strategy, spec, income,
                             rt.FixedAgeRule(age=spec.age_retire), saving=Spy())
        assert seen == [(h, h) for h, _ in seen]


class TestFundedGap:
    @pytest.mark.parametrize("form", acc.GAP_FORMS)
    def test_positive_when_behind(self, form: str) -> None:
        assert acc.funded_gap(np.array([2.0]), 4.0, form)[0] > 0

    @pytest.mark.parametrize("form", acc.GAP_FORMS)
    def test_negative_when_ahead(self, form: str) -> None:
        assert acc.funded_gap(np.array([8.0]), 4.0, form)[0] < 0

    @pytest.mark.parametrize("form", acc.GAP_FORMS)
    def test_zero_on_target(self, form: str) -> None:
        assert acc.funded_gap(np.array([4.0]), 4.0, form)[0] == pytest.approx(0.0)

    @pytest.mark.parametrize("form", acc.GAP_FORMS)
    def test_no_signal_against_a_zero_target(self, form: str) -> None:
        assert acc.funded_gap(np.array([0.0, 3.0]), 0.0, form).tolist() == [0.0, 0.0]

    def test_scale_free_forms_are_bounded_by_the_ratio_clip(self) -> None:
        far_behind = acc.funded_gap(np.array([1e-9]), 10.0, "proportional")[0]
        far_ahead = acc.funded_gap(np.array([1e9]), 10.0, "proportional")[0]
        lo, hi = acc.RATIO_CLIP
        assert far_behind == pytest.approx(1.0 - lo)
        assert far_ahead == pytest.approx(1.0 - hi)

    def test_level_form_is_not_scale_free(self) -> None:
        """Same proportional shortfall, different age: the level gap differs."""
        young = acc.funded_gap(np.array([0.5]), 1.0, "level")[0]
        old = acc.funded_gap(np.array([5.0]), 10.0, "level")[0]
        assert young != pytest.approx(old)
        assert acc.funded_gap(np.array([0.5]), 1.0, "proportional")[0] \
            == pytest.approx(acc.funded_gap(np.array([5.0]), 10.0,
                                            "proportional")[0])

    def test_rejects_an_unknown_form(self) -> None:
        with pytest.raises(ValueError, match="unknown gap form"):
            acc.funded_gap(np.array([1.0]), 2.0, "quadratic")


class TestFundedRatioRule:
    def rule(self, **kwargs) -> acc.FundedRatioRule:
        return acc.FundedRatioRule(target=np.full(68, 4.0),
                                   base=np.full(68, 0.10), **kwargs)

    def test_zero_sensitivity_is_the_base_profile(self) -> None:
        assert self.rule().rate(state(wealth=[1.0]))[0] == pytest.approx(0.10)

    def test_saves_more_when_behind(self) -> None:
        assert self.rule(k_behind=0.2).rate(state(wealth=[1.0]))[0] > 0.10

    def test_saves_less_when_ahead(self) -> None:
        assert self.rule(k_behind=0.2).rate(state(wealth=[8.0]))[0] < 0.10

    def test_k_ahead_none_is_symmetric(self) -> None:
        symmetric = self.rule(k_behind=0.2, k_ahead=0.2)
        implicit = self.rule(k_behind=0.2)
        for wealth in (1.0, 4.0, 9.0):
            assert implicit.rate(state(wealth=[wealth]))[0] \
                == pytest.approx(symmetric.rate(state(wealth=[wealth]))[0])

    def test_catch_up_only_never_eases_off(self) -> None:
        rule = self.rule(k_behind=0.2, k_ahead=0.0)
        assert rule.rate(state(wealth=[1.0]))[0] > 0.10
        assert rule.rate(state(wealth=[20.0]))[0] == pytest.approx(0.10)

    def test_ease_off_only_never_catches_up(self) -> None:
        rule = self.rule(k_behind=0.0, k_ahead=0.2)
        assert rule.rate(state(wealth=[1.0]))[0] == pytest.approx(0.10)
        assert rule.rate(state(wealth=[20.0]))[0] < 0.10

    def test_respects_the_floor_and_cap(self) -> None:
        rule = self.rule(k_behind=10.0, floor=0.02, cap=0.25)
        assert rule.rate(state(wealth=[0.0]))[0] == pytest.approx(0.25)
        assert rule.rate(state(wealth=[40.0]))[0] == pytest.approx(0.02)

    def test_outside_the_age_window_it_is_the_base_profile(self) -> None:
        rule = self.rule(k_behind=0.2, min_age=50, max_age=60)
        assert rule.rate(state(age=45, wealth=[1.0]))[0] == pytest.approx(0.10)
        assert rule.rate(state(age=65, wealth=[1.0]))[0] == pytest.approx(0.10)
        assert rule.rate(state(age=55, wealth=[1.0]))[0] > 0.10


class TestBandedRule:
    def rule(self, **kwargs) -> acc.BandedRule:
        return acc.BandedRule(target=np.full(68, 4.0), base=np.full(68, 0.10),
                              band=0.20, step=0.03, **kwargs)

    def test_inside_the_band_nothing_happens(self) -> None:
        assert self.rule().rate(state(wealth=[4.4]))[0] == pytest.approx(0.10)
        assert self.rule().rate(state(wealth=[3.6]))[0] == pytest.approx(0.10)

    def test_below_the_band_it_steps_up(self) -> None:
        assert self.rule().rate(state(wealth=[2.0]))[0] == pytest.approx(0.13)

    def test_above_the_band_it_steps_down(self) -> None:
        assert self.rule().rate(state(wealth=[6.0]))[0] == pytest.approx(0.07)

    def test_the_step_is_flat_however_far_off_target(self) -> None:
        """Unlike the continuous rule, distance past the band buys nothing."""
        near = self.rule().rate(state(wealth=[2.0]))[0]
        far = self.rule().rate(state(wealth=[0.01]))[0]
        assert near == pytest.approx(far)

    def test_a_zero_target_disables_it(self) -> None:
        rule = acc.BandedRule(target=np.zeros(68), base=np.full(68, 0.10))
        assert rule.rate(state(wealth=[5.0]))[0] == pytest.approx(0.10)


class TestSignals:
    def test_every_labelled_signal_can_be_built(self) -> None:
        for name in acc.SIGNAL_LABELS:
            fn = acc.make_signal(name, target=np.full(68, 4.0),
                                 income_profile=np.full(68, 1.0))
            assert fn(state()).shape == (1,)

    def test_every_signal_has_a_family(self) -> None:
        assert set(acc.SIGNAL_FAMILY) == set(acc.SIGNAL_LABELS)

    def test_return_signals_say_save_more_after_a_bad_year(self) -> None:
        fn = acc.make_signal("return_1y")
        assert fn(state(last_return=[-0.30]))[0] > 0
        assert fn(state(last_return=[0.30]))[0] < 0

    def test_trailing_signals_use_the_window(self) -> None:
        five = acc.make_signal("return_5y")
        history = [[-0.20] * 5 + [0.50]]
        assert five(state(history=history))[0] > 0

    def test_funded_ratio_signal_says_save_more_when_behind(self) -> None:
        fn = acc.make_signal("funded_ratio", target=np.full(68, 4.0))
        assert fn(state(wealth=[1.0]))[0] > 0
        assert fn(state(wealth=[9.0]))[0] < 0

    def test_income_signal_says_save_more_when_paid_above_expectation(self
                                                                     ) -> None:
        fn = acc.make_signal("income_shock", income_profile=np.full(68, 1.0))
        assert fn(state(income=[1.5]))[0] > 0
        assert fn(state(income=[0.5]))[0] < 0

    def test_investment_gain_signal_says_save_more_after_losses(self) -> None:
        fn = acc.make_signal("investment_gain")
        assert fn(state(wealth=[5.0], contributed=[10.0]))[0] > 0
        assert fn(state(wealth=[20.0], contributed=[10.0]))[0] < 0

    def test_the_null_signal_is_identically_zero(self) -> None:
        fn = acc.make_signal("none")
        assert fn(state(wealth=[1.0]))[0] == 0.0

    def test_rejects_an_unknown_signal(self) -> None:
        with pytest.raises(ValueError, match="unknown signal"):
            acc.make_signal("astrology")

    def test_funded_ratio_signal_requires_a_target(self) -> None:
        with pytest.raises(ValueError, match="needs a target"):
            acc.make_signal("funded_ratio")

    def test_income_signal_requires_a_profile(self) -> None:
        with pytest.raises(ValueError, match="needs an expected income path"):
            acc.make_signal("income_shock")


class TestSignalAndCombinedRules:
    def test_signal_rule_with_zero_sensitivity_is_the_base(self) -> None:
        rule = acc.SignalRule(base=np.full(68, 0.10),
                              signal=acc.make_signal("return_1y"),
                              sensitivity=0.0)
        assert rule.rate(state(last_return=[-0.5]))[0] == pytest.approx(0.10)

    def test_combined_rule_adds_both_responses(self) -> None:
        first = acc.make_signal("funded_ratio", target=np.full(68, 4.0))
        second = acc.make_signal("return_1y")
        s = state(wealth=[2.0], last_return=[-0.20])
        base = np.full(68, 0.10)
        only_first = acc.CombinedRule(base=base, first=first, second=second,
                                      k_first=0.2, k_second=0.0).rate(s)[0]
        only_second = acc.CombinedRule(base=base, first=first, second=second,
                                       k_first=0.0, k_second=0.2).rate(s)[0]
        both = acc.CombinedRule(base=base, first=first, second=second,
                                k_first=0.2, k_second=0.2).rate(s)[0]
        assert both == pytest.approx(only_first + only_second - 0.10)

    def test_combined_rule_respects_the_cap(self) -> None:
        first = acc.make_signal("funded_ratio", target=np.full(68, 4.0))
        rule = acc.CombinedRule(base=np.full(68, 0.10), first=first,
                                second=first, k_first=5.0, k_second=5.0,
                                cap=0.30)
        assert rule.rate(state(wealth=[0.0]))[0] == pytest.approx(0.30)


class TestTargets:
    def test_ladder_is_non_decreasing_with_age(self, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        ladder = acc.ladder_target(spec)
        assert np.all(np.diff(ladder) >= -1e-12)

    def test_ladder_hits_its_anchors(self, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        anchors = {spec.age_start + 2: 1.0, spec.age_start + 4: 5.0}
        ladder = acc.ladder_target(spec, anchors)
        assert ladder[2] == pytest.approx(1.0)
        assert ladder[4] == pytest.approx(5.0)

    def test_flat_target_has_no_age_content(self, toy_config) -> None:
        spec = lc.spec_from_config(toy_config)
        assert np.ptp(acc.flat_target(spec, 8.0)) == 0.0

    def test_scaling_is_multiplicative(self) -> None:
        assert acc.scale_target(np.array([1.0, 2.0]), 1.5).tolist() == [1.5, 3.0]


class TestMatchedScorer:
    def scorer(self) -> acc.MatchedScorer:
        frontier = pd.DataFrame({"savings_rate": [0.05, 0.10, 0.20],
                                 "cec": [1.0, 1.2, 1.1]})
        return acc.MatchedScorer.from_frontier(frontier, "cec")

    def test_a_constant_rule_scores_zero_against_itself(self) -> None:
        assert self.scorer().value_pct(1.2, 0.10) == pytest.approx(0.0)

    def test_saving_more_is_not_credited_as_skill(self) -> None:
        """A rule that drifts up the frontier scores on the drift, not the CEC."""
        assert self.scorer().value_pct(1.2, 0.20) > 0.0
        assert self.scorer().value_pct(1.1, 0.20) == pytest.approx(0.0)

    def test_interpolates_between_grid_points(self) -> None:
        assert self.scorer().matched(0.075) == pytest.approx(1.1)

    def test_flags_extrapolation(self) -> None:
        assert self.scorer().extrapolated(0.30)
        assert not self.scorer().extrapolated(0.10)


class TestDiagnostics:
    def test_equivalent_rate_move_is_zero_at_zero_sensitivity(self) -> None:
        assert acc.equivalent_rate_move("proportional", 0.0, 4.0) == 0.0

    def test_equivalent_rate_move_scales_with_the_target_for_level(self) -> None:
        small = acc.equivalent_rate_move("level", 0.01, 2.0)
        large = acc.equivalent_rate_move("level", 0.01, 8.0)
        assert large == pytest.approx(4.0 * small)

    def test_equivalent_rate_move_is_target_free_for_the_funded_ratio(self
                                                                     ) -> None:
        assert acc.equivalent_rate_move("proportional", 0.2, 2.0) \
            == pytest.approx(acc.equivalent_rate_move("proportional", 0.2, 8.0))

    def test_policy_curve_crosses_the_base_rate_on_target(self) -> None:
        rule = acc.FundedRatioRule(target=np.full(68, 4.0),
                                   base=np.full(68, 0.10), k_behind=0.2)
        rates = acc.policy_curve(rule, 20, 4.0, [0.5, 1.0, 1.5])
        assert rates[1] == pytest.approx(0.10)
        assert rates[0] > rates[1] > rates[2]

    def test_increment_over_subtracts_the_reference(self) -> None:
        frame = pd.DataFrame({"matched_value_pct": [1.0, 3.0]})
        out = acc.increment_over(frame, 0.5)
        assert out["increment_pct"].tolist() == [0.5, 2.5]

    def test_at_grid_edge_detects_truncated_searches(self) -> None:
        grid = [0.0, 0.1, 0.2]
        assert acc.at_grid_edge(grid, 0.2)
        assert acc.at_grid_edge(grid, 0.0)
        assert not acc.at_grid_edge(grid, 0.1)
        assert not acc.at_grid_edge([], 0.1)

    def test_common_strength_ranks_at_equal_reach(self) -> None:
        """A form whose grid reaches further must not win on reach alone."""
        frame = pd.DataFrame({
            "form": ["a", "a", "b", "b"],
            "rate_move_pp": [0.0, 4.0, 0.0, 8.0],
            "matched_value_pct": [0.0, 2.0, 0.0, 3.0],
        })
        out = acc.value_at_common_strength(frame)
        assert set(out["common_strength"]) == {4.0}
        row_a = out[out["form"] == "a"].iloc[0]
        row_b = out[out["form"] == "b"].iloc[0]
        assert row_a["value_at_common"] == pytest.approx(2.0)
        assert row_b["value_at_common"] == pytest.approx(1.5)
        assert row_b["own_best"] > row_a["own_best"]
        assert out["form"].iloc[0] == "a"

    def test_partition_windows_tiles_without_overlap(self) -> None:
        frame = pd.DataFrame({
            "min_age": [25, 40, 50, 25, 40, 25],
            "max_age": [39, 49, 62, 49, 62, 62],
            "window": ["25-39", "40-49", "50-62", "25-49", "40-62", "25-62"],
        })
        assert list(acc.partition_windows(frame)["window"]) \
            == ["25-39", "40-49", "50-62"]

    def test_partition_windows_handles_a_single_span(self) -> None:
        frame = pd.DataFrame({"min_age": [25], "max_age": [62],
                              "window": ["25-62"]})
        assert list(acc.partition_windows(frame)["window"]) == ["25-62"]

    def test_best_by_keeps_one_row_per_group(self) -> None:
        frame = pd.DataFrame({"signal": ["a", "a", "b"],
                              "matched_value_pct": [1.0, 2.0, 0.5]})
        best = acc.best_by(frame, "signal")
        assert len(best) == 2
        assert best["matched_value_pct"].tolist() == [0.5, 2.0]

    def test_rate_fan_covers_the_working_years(self, setup) -> None:
        cfg, spec, strategy, paths, income = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(age=spec.age_retire),
            saving=sav.ConstantRateRule(0.10))
        fan = acc.rate_fan(outcome, spec)
        assert len(fan) == spec.n_working
        assert np.allclose(fan["q50"], 0.10)

    def test_activity_profile_is_zero_against_itself(self, setup) -> None:
        cfg, spec, strategy, paths, income = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(age=spec.age_retire),
            saving=sav.ConstantRateRule(0.10))
        profile = acc.activity_profile(outcome, outcome, spec)
        assert np.allclose(profile["mean_abs_deviation"], 0.0)

    def test_quantile_gain_is_zero_against_itself(self, setup) -> None:
        cfg, spec, strategy, paths, income = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(age=spec.age_retire),
            saving=sav.ConstantRateRule(0.10))
        gain = acc.quantile_gain(outcome, outcome, [0.1, 0.5, 0.9])
        assert np.allclose(gain["gain_pct"], 0.0)

    def test_retirement_consumption_matches_the_evaluator(self, setup) -> None:
        cfg, spec, strategy, paths, income = setup
        outcome = rt.simulate_flexible(
            paths, strategy, spec, income, rt.FixedAgeRule(age=spec.age_retire),
            saving=sav.ConstantRateRule(0.10))
        row = rt.evaluate(outcome, cfg, spec, [5.0])
        assert float(np.median(acc.retirement_consumption(outcome))) \
            == pytest.approx(row["median_retirement_consumption"])


class TestSweepsAgainstTheEngine:
    """The sweeps are thin, but a k = 0 row must reproduce the base exactly."""

    def run_fn(self, setup):
        cfg, spec, strategy, paths, income = setup

        def run(rule):
            return rt.evaluate(
                rt.simulate_flexible(paths, strategy, spec, income,
                                     rt.FixedAgeRule(age=spec.age_retire),
                                     saving=rule), cfg, spec, [5.0])
        return run, cfg, spec

    def test_zero_sensitivity_reproduces_the_age_profile(self, setup) -> None:
        run, cfg, spec = self.run_fn(setup)
        base = np.full(spec.horizon, 0.10)
        target = np.full(spec.horizon, 4.0)
        reference = run(sav.AgeProfileRule(base))["cec_gamma5"]
        conditioned = run(acc.FundedRatioRule(target=target, base=base,
                                              k_behind=0.0))["cec_gamma5"]
        assert conditioned == pytest.approx(reference, rel=0, abs=0)

    def test_response_sweep_has_a_row_per_grid_point(self, setup) -> None:
        run, cfg, spec = self.run_fn(setup)
        frame = acc.sweep_response_forms(
            run, "cec_gamma5",
            acc.MatchedScorer(np.array([0.0, 0.2]), np.array([0.5, 1.5])),
            np.full(spec.horizon, 0.10), np.full(spec.horizon, 4.0),
            {"proportional": [0.0, 0.1], "level": [0.0, 0.01]})
        assert len(frame) == 4
        assert set(frame["form"]) == {"proportional", "level"}
        assert frame["rate_move_pp"].iloc[0] == 0.0

    def test_age_window_covering_everything_matches_the_open_rule(self, setup
                                                                  ) -> None:
        run, cfg, spec = self.run_fn(setup)
        base = np.full(spec.horizon, 0.10)
        target = np.full(spec.horizon, 4.0)
        whole = run(acc.FundedRatioRule(target=target, base=base,
                                        k_behind=0.2))["cec_gamma5"]
        windowed = run(acc.FundedRatioRule(
            target=target, base=base, k_behind=0.2,
            min_age=spec.age_start, max_age=spec.age_death))["cec_gamma5"]
        assert windowed == pytest.approx(whole, rel=0, abs=0)

    def test_feasibility_at_zero_width_pins_the_rate(self, setup) -> None:
        """A band of zero width leaves nothing to condition on, so the rule
        collapses onto the constant rate it is scored against."""
        run, cfg, spec = self.run_fn(setup)
        rates = [0.05, 0.10, 0.20]
        scorer = acc.MatchedScorer(
            np.array(rates),
            np.array([run(sav.ConstantRateRule(r))["cec_gamma5"]
                      for r in rates]))
        frame = acc.feasibility_frontier(
            run, "cec_gamma5", scorer,
            np.full(spec.horizon, 0.10), np.full(spec.horizon, 4.0),
            k=0.2, widths=[0.0], target_mean=0.10)
        assert float(frame["mean_savings_rate"].iloc[0]) == pytest.approx(0.10)
        assert float(frame["matched_value_pct"].iloc[0]) == pytest.approx(0.0)
