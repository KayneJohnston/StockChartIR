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
