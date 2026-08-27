"""Tests for the retirement spending rules."""

from __future__ import annotations

import numpy as np
import pytest

from src import lifecycle as lc
from src import spending as sp
from tests.test_lifecycle import constant_paths


def state(wealth, prev=None, initial=None, year=0, age=63, remaining=30,
          last_return=0.05, last_inflation=0.02) -> sp.SpendingState:
    wealth = np.asarray(wealth, dtype=float)
    n = wealth.size
    return sp.SpendingState(
        year=year, age=age, years_remaining=remaining, wealth=wealth,
        prev_withdrawal=np.full(n, prev if prev is not None else 4.0),
        initial_withdrawal=np.full(n, initial if initial is not None else 4.0),
        wealth_at_retirement=np.full(n, 100.0),
        last_return=np.full(n, last_return),
        last_inflation=np.full(n, last_inflation),
    )


class TestConstantRules:
    def test_constant_real_ignores_current_wealth(self) -> None:
        rule = sp.ConstantRealRule(rate=0.04)
        out = rule.desired(state([100.0, 40.0, 500.0], initial=4.0))
        np.testing.assert_allclose(out, 4.0)

    def test_constant_real_initial_is_rate_times_wealth(self) -> None:
        rule = sp.ConstantRealRule(rate=0.04)
        got = rule.initial_withdrawal(np.array([100.0, 250.0]), 30, 63)
        np.testing.assert_allclose(got, [4.0, 10.0])

    def test_constant_percent_tracks_current_wealth(self) -> None:
        rule = sp.ConstantPercentRule(rate=0.05)
        np.testing.assert_allclose(
            rule.desired(state([100.0, 40.0])), [5.0, 2.0])

    def test_constant_percent_never_asks_for_more_than_it_has(self) -> None:
        rule = sp.ConstantPercentRule(rate=0.05)
        wealth = np.array([1e-6, 1.0, 1e6])
        assert np.all(rule.desired(state(wealth)) < wealth)


class TestGuytonKlinger:
    def test_holds_spending_flat_inside_the_guardrails(self) -> None:
        rule = sp.GuytonKlingerRule(rate=0.05, guardrail=0.20)
        # 5/100 = 5% is exactly the initial rate: no adjustment.
        out = rule.desired(state([100.0], prev=5.0, last_return=0.05))
        np.testing.assert_allclose(out, [5.0])

    def test_cuts_when_the_withdrawal_rate_runs_high(self) -> None:
        rule = sp.GuytonKlingerRule(rate=0.05, guardrail=0.20, adjustment=0.10)
        # 5/70 = 7.1% > 6% upper guardrail -> capital-preservation cut.
        out = rule.desired(state([70.0], prev=5.0, last_return=0.05,
                                 remaining=30))
        np.testing.assert_allclose(out, [4.5])

    def test_raises_when_the_withdrawal_rate_runs_low(self) -> None:
        rule = sp.GuytonKlingerRule(rate=0.05, guardrail=0.20, adjustment=0.10)
        # 5/140 = 3.6% < 4% lower guardrail -> prosperity raise.
        out = rule.desired(state([140.0], prev=5.0, last_return=0.05))
        np.testing.assert_allclose(out, [5.5])

    def test_suspends_the_cut_near_the_end_of_the_plan(self) -> None:
        rule = sp.GuytonKlingerRule(rate=0.05, guardrail=0.20, adjustment=0.10,
                                    preservation_cutoff_years=15)
        out = rule.desired(state([70.0], prev=5.0, last_return=0.05,
                                 remaining=10))
        np.testing.assert_allclose(out, [5.0])

    def test_inflation_rule_cuts_real_spending_after_a_down_year(self) -> None:
        rule = sp.GuytonKlingerRule(rate=0.05, guardrail=0.50,
                                    apply_inflation_rule=True)
        out = rule.desired(state([100.0], prev=5.0, last_return=-0.10,
                                 last_inflation=0.04))
        np.testing.assert_allclose(out, [5.0 / 1.04])

    def test_inflation_rule_is_silent_after_an_up_year(self) -> None:
        rule = sp.GuytonKlingerRule(rate=0.05, guardrail=0.50,
                                    apply_inflation_rule=True)
        out = rule.desired(state([100.0], prev=5.0, last_return=0.10,
                                 last_inflation=0.04))
        np.testing.assert_allclose(out, [5.0])

    def test_inflation_rule_can_be_switched_off(self) -> None:
        rule = sp.GuytonKlingerRule(rate=0.05, guardrail=0.50,
                                    apply_inflation_rule=False)
        out = rule.desired(state([100.0], prev=5.0, last_return=-0.10,
                                 last_inflation=0.04))
        np.testing.assert_allclose(out, [5.0])

    def test_deflation_does_not_raise_real_spending(self) -> None:
        rule = sp.GuytonKlingerRule(rate=0.05, guardrail=0.50)
        out = rule.desired(state([100.0], prev=5.0, last_return=-0.10,
                                 last_inflation=-0.03))
        np.testing.assert_allclose(out, [5.0])


class TestVanguardDynamic:
    def test_caps_the_year_on_year_rise(self) -> None:
        rule = sp.VanguardDynamicRule(rate=0.05, ceiling=0.05, floor=-0.025)
        out = rule.desired(state([300.0], prev=5.0))   # target 15 -> capped
        np.testing.assert_allclose(out, [5.25])

    def test_floors_the_year_on_year_fall(self) -> None:
        rule = sp.VanguardDynamicRule(rate=0.05, ceiling=0.05, floor=-0.025)
        out = rule.desired(state([20.0], prev=5.0))    # target 1 -> floored
        np.testing.assert_allclose(out, [4.875])

    def test_passes_the_target_through_inside_the_band(self) -> None:
        rule = sp.VanguardDynamicRule(rate=0.05, ceiling=0.05, floor=-0.025)
        out = rule.desired(state([102.0], prev=5.0))
        np.testing.assert_allclose(out, [5.1])


class TestEndowment:
    def test_nests_constant_percent_at_zero_smoothing(self) -> None:
        smooth = sp.EndowmentRule(rate=0.05, smoothing=0.0)
        plain = sp.ConstantPercentRule(rate=0.05)
        s = state([137.0], prev=9.0)
        np.testing.assert_allclose(smooth.desired(s), plain.desired(s))

    def test_nests_constant_spending_at_full_smoothing(self) -> None:
        rule = sp.EndowmentRule(rate=0.05, smoothing=1.0)
        np.testing.assert_allclose(rule.desired(state([300.0], prev=5.0)), [5.0])

    def test_blends_the_two(self) -> None:
        rule = sp.EndowmentRule(rate=0.05, smoothing=0.7)
        out = rule.desired(state([200.0], prev=5.0))
        np.testing.assert_allclose(out, [0.7 * 5.0 + 0.3 * 10.0])


class TestHorizonRules:
    def test_life_expectancy_divides_by_years_remaining(self) -> None:
        rule = sp.LifeExpectancyRule()
        np.testing.assert_allclose(
            rule.desired(state([100.0], remaining=25)), [4.0])

    def test_life_expectancy_spends_everything_in_the_final_year(self) -> None:
        rule = sp.LifeExpectancyRule()
        np.testing.assert_allclose(
            rule.desired(state([37.0], remaining=1)), [37.0])

    def test_buffer_years_lower_the_rate(self) -> None:
        plain = sp.LifeExpectancyRule()
        buffered = sp.LifeExpectancyRule(buffer_years=5)
        s = state([100.0], remaining=20)
        assert buffered.desired(s)[0] < plain.desired(s)[0]

    def test_gompertz_expectancy_falls_with_age(self) -> None:
        values = [sp.gompertz_life_expectancy(a) for a in (60, 70, 80, 90, 100)]
        assert all(a > b for a, b in zip(values, values[1:]))
        assert all(v > 0 for v in values)

    def test_gompertz_expectancy_is_plausible_at_retirement(self) -> None:
        assert 18.0 < sp.gompertz_life_expectancy(63) < 26.0

    def test_gompertz_expectancy_is_zero_past_the_max_age(self) -> None:
        assert sp.gompertz_life_expectancy(125) == 0.0

    def test_gompertz_rule_spends_more_than_a_fixed_horizon(self) -> None:
        # At 63, actuarial expectancy is shorter than the 30 years to age 93,
        # so the actuarial rule front-loads spending.
        fixed = sp.LifeExpectancyRule()
        actuarial = sp.GompertzRule()
        s = state([100.0], age=63, remaining=30)
        assert actuarial.desired(s)[0] > fixed.desired(s)[0]

    def test_amortisation_nests_life_expectancy_at_zero_return(self) -> None:
        amort = sp.AmortisationRule(assumed_return=0.0)
        rmd = sp.LifeExpectancyRule()
        s = state([100.0], remaining=20)
        np.testing.assert_allclose(amort.desired(s), rmd.desired(s))

    def test_amortisation_matches_the_annuity_formula(self) -> None:
        rule = sp.AmortisationRule(assumed_return=0.03)
        n, r = 20, 0.03
        expected = 100.0 * r / (1.0 - (1.0 + r) ** (-n))
        np.testing.assert_allclose(rule.desired(state([100.0], remaining=n)),
                                   [expected])

    def test_higher_assumed_return_front_loads_spending(self) -> None:
        s = state([100.0], remaining=25)
        low = sp.AmortisationRule(assumed_return=0.01).desired(s)[0]
        high = sp.AmortisationRule(assumed_return=0.05).desired(s)[0]
        assert high > low


class TestRegistry:
    def test_builds_every_registered_rule(self) -> None:
        for key in sp.REGISTRY:
            rule = sp.build(key)
            assert isinstance(rule, sp.SpendingRule)
            assert rule.key == key

    def test_rejects_an_unknown_rule(self) -> None:
        with pytest.raises(ValueError, match="unknown spending rule"):
            sp.build("spend_it_all")

    def test_from_spec_maps_the_legacy_names(self) -> None:
        assert isinstance(sp.from_spec("fixed_real_rule", 0.04),
                          sp.ConstantRealRule)
        assert isinstance(sp.from_spec("fixed_percentage", 0.05),
                          sp.ConstantPercentRule)

    def test_from_spec_rejects_anything_else(self) -> None:
        with pytest.raises(ValueError, match="unknown retirement_rule"):
            sp.from_spec("guardrails", 0.04)

    def test_describe_exposes_the_parameters(self) -> None:
        described = sp.build("guyton_klinger", rate=0.055).describe()
        assert described["key"] == "guyton_klinger"
        assert described["rate"] == pytest.approx(0.055)


class TestIntegrationWithTheSimulator:
    def _run(self, rule, dom_eq=0.03):
        spec = lc.LifecycleSpec(age_start=25, age_retire=35, age_death=60,
                                income_shocks_enabled=False)
        strategy = lc.Strategy("eq", "Eq",
                               np.tile([1.0, 0.0, 0.0, 0.0], (spec.horizon, 1)))
        paths = constant_paths(50, spec.horizon, dom_eq=dom_eq)
        income = np.tile(spec.deterministic_income(), (50, 1))
        return spec, lc.simulate(paths, strategy, spec, income, rule)

    @pytest.mark.parametrize("key", sorted(sp.REGISTRY))
    def test_every_rule_produces_finite_positive_consumption(self, key) -> None:
        _, outcome = self._run(sp.build(key))
        assert np.isfinite(outcome.consumption).all()
        assert (outcome.consumption > 0).all()
        assert np.isfinite(outcome.bequest).all()
        assert (outcome.wealth >= -1e-9).all()

    @pytest.mark.parametrize("key", ["constant_percent", "life_expectancy",
                                     "gompertz"])
    def test_rules_that_cannot_deplete_never_ruin(self, key) -> None:
        _, outcome = self._run(sp.build(key), dom_eq=-0.05)
        assert not outcome.ruin.any()

    def test_a_fixed_amount_can_ruin_on_bad_returns(self) -> None:
        _, outcome = self._run(sp.ConstantRealRule(rate=0.15), dom_eq=-0.05)
        assert outcome.ruin.all()

    def test_life_expectancy_leaves_no_bequest(self) -> None:
        _, outcome = self._run(sp.LifeExpectancyRule())
        np.testing.assert_allclose(outcome.bequest, 0.0, atol=1e-9)

    def test_default_rule_matches_the_legacy_path(self) -> None:
        # LifecycleSpec defaults to fixed_real_rule at 4%, so passing that
        # rule explicitly must reproduce the pre-refactor behaviour exactly.
        _, explicit = self._run(sp.ConstantRealRule(rate=0.04))
        _, implicit = self._run(None)
        np.testing.assert_allclose(explicit.consumption, implicit.consumption)
        np.testing.assert_allclose(explicit.bequest, implicit.bequest)
        np.testing.assert_array_equal(explicit.ruin, implicit.ruin)
