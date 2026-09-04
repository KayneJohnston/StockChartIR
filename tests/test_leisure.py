"""The cost of working, and the retirement date it prices.

Three things carry this section. The leisure adjustment must be a pure
markdown of the working years and leave everything else alone, so that L = 1
reproduces the rest of the paper exactly. The claiming adjustment must be
actuarially fair against the model's *own* survival curve, or an unreduced
pension starting whenever work stops decides the answer before leisure gets a
say. And the break-even must be measured against the date an investor would
choose anyway, or it reports the model's lean toward early retirement rather
than the value of leisure.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import leisure as le
from src import lifecycle as lc


class TestLeisureFromDrop:
    def test_no_drop_is_no_value(self) -> None:
        assert le.leisure_from_drop(0.0) == pytest.approx(1.0)

    def test_the_arithmetic_inverts(self) -> None:
        """A drop of d at constant welfare means a retired dollar does the
        work of 1/(1-d) working ones."""
        for drop in (0.10, 0.20, 0.35):
            value = le.leisure_from_drop(drop)
            assert value * (1.0 - drop) == pytest.approx(1.0)

    def test_a_total_drop_is_refused(self) -> None:
        for drop in (1.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                le.leisure_from_drop(drop)

    def test_the_anchor_table_is_derived_not_written_down(self) -> None:
        frame = le.anchor_table()
        for _, row in frame.iterrows():
            assert float(row["leisure"]) == pytest.approx(
                le.leisure_from_drop(float(row["consumption_drop"])))


def _outcome(spec: lc.LifecycleSpec, n: int = 4) -> lc.LifecycleOutcome:
    horizon = spec.horizon
    return lc.LifecycleOutcome(
        strategy="s", label="s",
        consumption=np.full((n, horizon), 2.0),
        wealth=np.full((n, horizon + 1), 10.0),
        portfolio_return=np.zeros((n, horizon)),
        wealth_at_retirement=np.full(n, 10.0),
        bequest=np.full(n, 5.0),
        ruin=np.zeros(n, dtype=bool),
        ruin_age=np.full(n, spec.age_death),
        social_security=np.full(n, 0.4),
        career_average_income=np.full(n, 1.0))


class TestRescale:
    def test_it_marks_down_the_working_years_only(self) -> None:
        spec = lc.LifecycleSpec()
        out = le.rescale(_outcome(spec), spec, 1.25)
        assert np.allclose(out.consumption[:, :spec.n_working], 2.0 / 1.25)
        assert np.allclose(out.consumption[:, spec.n_working:], 2.0)

    def test_one_is_the_identity(self) -> None:
        """L = 1 must reproduce every other result in the paper exactly."""
        spec = lc.LifecycleSpec()
        outcome = _outcome(spec)
        assert le.rescale(outcome, spec, 1.0) is outcome

    def test_it_leaves_wealth_and_ruin_alone(self) -> None:
        """How the investor feels about working does not move the portfolio."""
        spec = lc.LifecycleSpec()
        outcome = _outcome(spec)
        out = le.rescale(outcome, spec, 1.5)
        assert np.array_equal(out.wealth, outcome.wealth)
        assert np.array_equal(out.bequest, outcome.bequest)
        assert np.array_equal(out.ruin, outcome.ruin)

    def test_it_does_not_mutate_the_original(self) -> None:
        spec = lc.LifecycleSpec()
        outcome = _outcome(spec)
        le.rescale(outcome, spec, 1.5)
        assert np.allclose(outcome.consumption, 2.0)

    def test_valuing_work_above_retirement_is_refused(self) -> None:
        spec = lc.LifecycleSpec()
        with pytest.raises(ValueError, match="below 1.0"):
            le.rescale(_outcome(spec), spec, 0.9)

    def test_a_later_retirement_marks_down_more_years(self) -> None:
        early = lc.LifecycleSpec(age_retire=55)
        late = lc.LifecycleSpec(age_retire=70)
        a = le.rescale(_outcome(early), early, 1.5).consumption
        b = le.rescale(_outcome(late), late, 1.5).consumption
        assert (a < 2.0).sum() < (b < 2.0).sum()


class TestFairClaimFactor:
    @staticmethod
    def _survival(spec: lc.LifecycleSpec) -> np.ndarray:
        from src import mortality as mort
        return mort.survival(spec)

    def test_the_reference_age_is_unadjusted(self) -> None:
        spec = lc.LifecycleSpec()
        factors = le.fair_claim_factor(spec, self._survival(spec), 0.96, 63,
                                       [55, 63, 70])
        assert factors[63] == pytest.approx(1.0)

    def test_claiming_early_reduces_and_deferring_raises(self) -> None:
        spec = lc.LifecycleSpec()
        factors = le.fair_claim_factor(spec, self._survival(spec), 0.96, 63,
                                       [55, 60, 63, 67, 70])
        assert factors[55] < factors[60] < factors[63] < factors[67] \
            < factors[70]
        assert factors[55] < 1.0 < factors[70]

    def test_it_leaves_the_expected_benefit_invariant(self) -> None:
        """That is what "actuarially fair" means, and it is the whole point.

        A factor that did not equalise the discounted expected benefit would
        leave a subsidy or a penalty on the claiming date, and the section
        would be measuring that rather than the value of leisure.
        """
        spec = lc.LifecycleSpec()
        survive = self._survival(spec)
        beta = 0.96
        ages = [55, 58, 60, 63, 67, 70]
        factors = le.fair_claim_factor(spec, survive, beta, 63, ages)
        values = [factors[a] * le.annuity_factor(spec, survive, beta, a)
                  for a in ages]
        assert np.allclose(values, values[0])

    def test_an_age_outside_the_horizon_is_refused(self) -> None:
        spec = lc.LifecycleSpec()
        survive = self._survival(spec)
        for age in (24, 93, 100):
            with pytest.raises(ValueError, match="outside the simulated"):
                le.annuity_factor(spec, survive, 0.96, age)

    def test_the_per_year_rate_lands_where_real_schedules_do(self) -> None:
        """Not fitted to any statute; the agreement is a check on the law.

        US social security reduces a benefit roughly 6.7% a year for early
        claiming and raises it 8% a year for deferral, so a derived rate an
        order of magnitude away from that would mean the survival curve or
        the discount factor was wrong.
        """
        spec = lc.LifecycleSpec()
        table = le.claim_factor_table(
            le.fair_claim_factor(spec, self._survival(spec), 0.96, 63,
                                 [55, 60, 67, 70]), 63)
        rates = table["per_year_pct"].abs()
        assert rates.between(4.0, 12.0).all(), table


class TestSpecCarriesTheClaimFactor:
    def test_the_default_changes_nothing(self) -> None:
        """Every other section must stay bit-identical."""
        spec = lc.LifecycleSpec()
        assert spec.ss_claim_factor == 1.0
        career = np.array([1.0, 2.0])
        assert np.allclose(spec.social_security_benefit(career),
                           dataclasses.replace(spec, ss_claim_factor=1.0)
                           .social_security_benefit(career))

    def test_it_scales_the_progressive_benefit(self) -> None:
        spec = lc.LifecycleSpec()
        career = np.array([0.5, 1.0, 2.0])
        halved = dataclasses.replace(spec, ss_claim_factor=0.5)
        assert np.allclose(halved.social_security_benefit(career),
                           0.5 * spec.social_security_benefit(career))

    def test_it_scales_the_flat_benefit_too(self) -> None:
        spec = lc.LifecycleSpec(social_security_formula="flat")
        career = np.array([1.0, 2.0])
        doubled = dataclasses.replace(spec, ss_claim_factor=2.0)
        assert np.allclose(doubled.social_security_benefit(career),
                           2.0 * spec.social_security_benefit(career))


def _surface(peaks: dict) -> pd.DataFrame:
    """A swept frame whose argmax per leisure value is what was asked for."""
    rows = []
    for leisure, best in peaks.items():
        for age in (55, 58, 60, 63, 67, 70):
            rows.append({"leisure": float(leisure), "retire_age": int(age),
                         "cec": 1.0 - 0.01 * abs(age - best)})
    return pd.DataFrame.from_records(rows)


class TestReadingTheSurface:
    def test_the_optimum_is_found_at_each_leisure_value(self) -> None:
        frame = _surface({1.0: 63, 1.2: 60, 1.5: 55})
        out = le.optimal_age(frame).set_index("leisure")
        assert int(out.loc[1.0, "optimal_age"]) == 63
        assert int(out.loc[1.5, "optimal_age"]) == 55

    def test_an_optimum_on_the_grid_edge_is_flagged(self) -> None:
        """It would be the grid's answer rather than the model's."""
        out = le.optimal_age(_surface({1.0: 55})).iloc[0]
        assert bool(out["at_grid_floor"])
        assert not bool(out["at_grid_ceiling"])
        out = le.optimal_age(_surface({1.0: 70})).iloc[0]
        assert bool(out["at_grid_ceiling"])

    def test_the_reference_is_the_zero_leisure_optimum(self) -> None:
        """Not the oldest age. This model already leans early before leisure
        enters, and a break-even against the ceiling would report that lean."""
        frame = _surface({1.0: 63, 1.5: 55})
        assert le.zero_leisure_optimum(frame) == 63
        out = le.break_even(frame)
        assert set(out["reference_age"]) == {63}

    def test_the_crossing_interpolates(self) -> None:
        rows = []
        for leisure, a, b in ((1.0, 0.90, 1.00), (1.2, 1.10, 1.00)):
            rows += [{"leisure": leisure, "retire_age": 55, "cec": a},
                     {"leisure": leisure, "retire_age": 63, "cec": b}]
        frame = pd.DataFrame.from_records(rows)
        assert le.crossing(frame, 55, 63) == pytest.approx(1.1)

    def test_a_date_that_never_wins_is_infinite_not_missing(self) -> None:
        rows = []
        for leisure in (1.0, 1.5):
            rows += [{"leisure": leisure, "retire_age": 55, "cec": 0.9},
                     {"leisure": leisure, "retire_age": 63, "cec": 1.0}]
        frame = pd.DataFrame.from_records(rows)
        assert le.crossing(frame, 55, 63) == float("inf")

    def test_a_date_already_winning_needs_no_leisure(self) -> None:
        rows = []
        for leisure in (1.0, 1.5):
            rows += [{"leisure": leisure, "retire_age": 55, "cec": 1.1},
                     {"leisure": leisure, "retire_age": 63, "cec": 1.0}]
        frame = pd.DataFrame.from_records(rows)
        assert le.crossing(frame, 55, 63) == pytest.approx(1.0)

    def test_the_break_even_reports_the_implied_drop(self) -> None:
        rows = []
        for leisure, a in ((1.0, 0.90), (1.25, 1.00)):
            rows += [{"leisure": leisure, "retire_age": 55, "cec": a},
                     {"leisure": leisure, "retire_age": 63, "cec": 1.0}]
        out = le.break_even(pd.DataFrame.from_records(rows)).iloc[0]
        assert out["break_even_leisure"] == pytest.approx(1.25)
        assert out["implied_consumption_drop"] == pytest.approx(0.2)


class TestVerdict:
    @staticmethod
    def _inputs(peaks: dict):
        frame = _surface(peaks)
        optima = le.optimal_age(frame)
        return frame, optima, le.break_even(frame), le.anchor_table()

    def test_a_moving_date_is_recognised(self) -> None:
        found = le.verdict(*self._inputs({1.0: 63, 1.2: 60, 1.5: 55}))
        assert found["date_moves_with_leisure"]
        assert found["optimal_age_at_zero"] == 63
        assert found["optimal_age_at_top"] == 55

    def test_a_static_date_is_recognised(self) -> None:
        found = le.verdict(*self._inputs({1.0: 63, 1.2: 63, 1.5: 63}))
        assert not found["date_moves_with_leisure"]

    def test_the_unpriced_corner_is_the_control(self) -> None:
        """With nothing charged for working the date should sit at the
        ceiling, reproducing Section #plan. If it does not, something other
        than leisure is moving the answer -- which is exactly what the
        unadjusted pension turned out to be doing."""
        found = le.verdict(*self._inputs({1.0: 70, 1.5: 55}))
        assert found["corner_without_leisure"]
        found = le.verdict(*self._inputs({1.0: 60, 1.5: 55}))
        assert not found["corner_without_leisure"]

    def test_an_empty_surface_reports_nothing(self) -> None:
        assert le.verdict(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                          pd.DataFrame()) == {"measured": False}


class TestPensionSystems:
    """Australia's Age Pension differs from US social security in two ways
    that pull against each other, so the sweep has to separate them."""

    @staticmethod
    def _cfg() -> dict:
        from src import data_loader as dl
        return dl.load_config("config.yaml")

    def test_the_us_arm_overrides_nothing(self) -> None:
        """It is the paper's own baseline, and must stay bit-identical."""
        overrides, adjusted = le.system_overrides("us", self._cfg())
        assert overrides == {}
        assert adjusted

    def test_the_australian_arms_gate_the_pension_by_age(self) -> None:
        cfg = self._cfg()
        for key in ("au_pension_only", "au_as_legislated"):
            overrides, adjusted = le.system_overrides(key, cfg)
            assert overrides["social_security_formula"] == "means_tested"
            assert overrides["benefit_start_age"] == \
                int(cfg["leisure"]["age_pension_age"])
            # No actuarial adjustment: the Age Pension is an age you reach,
            # not a claiming choice, so adjusting it would model a system
            # nobody lives under.
            assert not adjusted

    def test_only_the_legislated_arm_carries_the_guarantee(self) -> None:
        """That is what separates the eligibility gate from the extra saving."""
        cfg = self._cfg()
        gate, _ = le.system_overrides("au_pension_only", cfg)
        full, _ = le.system_overrides("au_as_legislated", cfg)
        assert "super_guarantee_rate" not in gate
        assert full["super_guarantee_rate"] > 0.0
        assert {k: v for k, v in full.items()
                if k != "super_guarantee_rate"
                and k != "super_contributions_tax"} == gate

    def test_an_unknown_system_names_the_real_ones(self) -> None:
        with pytest.raises(ValueError, match="unknown pension system"):
            le.system_overrides("norway", self._cfg())

    def test_the_comparison_separates_gate_from_guarantee(self) -> None:
        optima = {
            "us": pd.DataFrame({"leisure": [1.0, 1.5], "optimal_age": [60, 55],
                                "cec_at_optimum": [0.76, 0.53]}),
            "au_pension_only": pd.DataFrame(
                {"leisure": [1.0, 1.5], "optimal_age": [65, 62],
                 "cec_at_optimum": [0.62, 0.47]}),
            "au_as_legislated": pd.DataFrame(
                {"leisure": [1.0, 1.5], "optimal_age": [62, 58],
                 "cec_at_optimum": [0.71, 0.51]}),
        }
        empty = pd.DataFrame(columns=["is_earlier", "reached_on_grid",
                                      "years_earlier", "break_even_pct"])
        found = le.system_verdict(le.system_comparison(
            optima, {k: empty for k in optima}))
        assert found["gate_pushes_later"]
        assert found["gate_years_later"] == pytest.approx(5.0)
        assert found["super_buys_back"]
        assert found["super_years_earlier"] == pytest.approx(3.0)
        assert found["australia_retires_later"]
        assert found["legislated_vs_us_years"] == pytest.approx(2.0)

    def test_an_empty_comparison_reports_nothing(self) -> None:
        assert le.system_verdict(pd.DataFrame()) == {"measured": False}

    @staticmethod
    def _priced(us: float, legislated: float) -> pd.DataFrame:
        return pd.DataFrame({
            "system": ["us", "au_pension_only", "au_as_legislated"],
            "age_at_zero_leisure": [60, 67, 62],
            "cost_per_year_pct": [us, 8.0, legislated],
        })

    def test_two_systems_charging_the_same_are_called_similar(self) -> None:
        found = le.system_verdict(self._priced(5.0, 5.2))
        assert found["cost_similar"]
        assert found["cost_ratio"] == pytest.approx(1.04)

    def test_a_system_charging_half_again_is_not(self) -> None:
        """The band exists so that a real difference in slope is reported as
        one, rather than waved through as 'similar'."""
        found = le.system_verdict(self._priced(4.0, 6.0))
        assert not found["cost_similar"]
        assert found["australia_dearer_per_year"]
        assert found["cost_ratio"] == pytest.approx(1.5)

    def test_the_band_is_two_sided(self) -> None:
        found = le.system_verdict(self._priced(6.0, 4.0))
        assert not found["cost_similar"]
        assert not found["australia_dearer_per_year"]

    def test_the_band_separates_both_sides(self) -> None:
        """Probed either side of the edge rather than on it: which way an
        exactly-`COST_SIMILAR_BAND` ratio falls is float noise, and a test
        that pinned it would be pinning the noise."""
        inside = le.COST_SIMILAR_BAND * 0.9
        outside = le.COST_SIMILAR_BAND * 1.1
        for gap in (-inside, inside):
            assert le.system_verdict(
                self._priced(5.0, 5.0 * (1.0 + gap)))["cost_similar"], gap
        for gap in (-outside, outside):
            assert not le.system_verdict(
                self._priced(5.0, 5.0 * (1.0 + gap)))["cost_similar"], gap

    def test_an_unpriced_system_leaves_the_verdict_silent(self) -> None:
        """Rather than defaulting to 'similar', which would print a claim the
        run never measured."""
        frame = self._priced(5.0, 5.5)
        frame.loc[frame["system"] == "us", "cost_per_year_pct"] = np.nan
        found = le.system_verdict(frame)
        assert "cost_similar" not in found
        assert "cost_ratio" not in found


class TestBenefitEligibilityAge:
    """The pension's start date is not the retirement date, and once the
    retirement date can move that stops being a detail."""

    def test_the_default_is_the_retirement_date(self) -> None:
        spec = lc.LifecycleSpec()
        assert spec.benefit_start_age is None
        assert spec.benefit_start_index == spec.n_working

    def test_a_later_age_delays_the_benefit(self) -> None:
        spec = dataclasses.replace(lc.LifecycleSpec(age_retire=55),
                                   benefit_start_age=67)
        assert spec.benefit_start_index == 67 - spec.age_start
        assert spec.benefit_start_index > spec.n_working

    def test_an_age_already_past_pays_from_retirement(self) -> None:
        spec = dataclasses.replace(lc.LifecycleSpec(age_retire=70),
                                   benefit_start_age=67)
        assert spec.benefit_start_index < spec.n_working

    def test_an_impossible_age_is_refused(self) -> None:
        for age in (20, 100):
            with pytest.raises(ValueError, match="benefit_start_age"):
                dataclasses.replace(lc.LifecycleSpec(), benefit_start_age=age)

    def test_a_safety_net_share_outside_zero_to_one_is_refused(self) -> None:
        for share in (-0.1, 1.5):
            with pytest.raises(ValueError,
                               match="pre_eligibility_benefit_share"):
                dataclasses.replace(lc.LifecycleSpec(),
                                    pre_eligibility_benefit_share=share)

    def test_the_bridge_shows_up_in_consumption(self, toy_panel, toy_config
                                                ) -> None:
        """The point of the whole arm: gate the pension and the years before
        it are funded by the portfolio alone."""
        from src import bootstrap as bs

        spec = lc.spec_from_config(toy_config)
        late = dataclasses.replace(spec, benefit_start_age=spec.age_death - 1,
                                   pre_eligibility_benefit_share=0.0)
        sampler = bs.from_config(toy_panel, toy_config)
        chunk = next(iter(sampler.chunks(200, 200)))
        strategies = lc.build_strategies(toy_config, spec)
        income = lc.simulate_income(spec, chunk.n_paths,
                                    np.random.default_rng(0))
        base = lc.simulate_all(chunk, strategies, spec, income)["all_equity"]
        gated = lc.simulate_all(chunk, strategies, late, income)["all_equity"]
        # Same portfolio either way -- a pension does not touch wealth --
        # but consumption before the gate is lower by exactly the benefit.
        assert np.allclose(base.wealth, gated.wealth)
        first = spec.n_working
        assert (gated.consumption[:, first] < base.consumption[:, first]).all()

    def test_the_safety_net_lifts_the_bridge_off_the_floor(self, toy_panel,
                                                           toy_config) -> None:
        from src import bootstrap as bs

        spec = dataclasses.replace(
            lc.spec_from_config(toy_config),
            social_security_formula="means_tested",
            benefit_start_age=lc.spec_from_config(toy_config).age_death - 1)
        netted = dataclasses.replace(spec, pre_eligibility_benefit_share=0.5)
        sampler = bs.from_config(toy_panel, toy_config)
        chunk = next(iter(sampler.chunks(200, 200)))
        strategies = lc.build_strategies(toy_config, spec)
        income = lc.simulate_income(spec, chunk.n_paths,
                                    np.random.default_rng(0))
        bare = lc.simulate_all(chunk, strategies, spec, income)["all_equity"]
        soft = lc.simulate_all(chunk, strategies, netted, income)["all_equity"]
        first = spec.n_working
        assert (soft.consumption[:, first]
                >= bare.consumption[:, first]).all()
        assert (soft.consumption[:, first] > bare.consumption[:, first]).any()
