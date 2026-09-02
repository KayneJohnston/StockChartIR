"""Tests for the survival-weighted certainty equivalent.

The claim the module rests on is that re-weighting an existing set of paths
is *exact* for a policy that does not condition on the death age. The way to
check that is to feed it a degenerate mortality law -- everyone dies at
ninety-three -- and require it to reproduce the project's own certainty
equivalent to machine precision. If that holds, the generalisation is a
generalisation and not a second metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import lifecycle as lc
from src import mortality as mo
from src import utility as ut


class _Outcome:
    """The fields :mod:`src.mortality` reads off a lifecycle outcome."""

    def __init__(self, consumption: np.ndarray, wealth: np.ndarray,
                 ruin_age: np.ndarray | None = None,
                 ruin: np.ndarray | None = None) -> None:
        n = consumption.shape[0]
        self.consumption = consumption
        self.wealth = wealth
        self.bequest = wealth[:, -1]
        self.ruin_age = (np.full(n, 93) if ruin_age is None else ruin_age)
        self.ruin = (np.zeros(n, dtype=bool) if ruin is None else ruin)
        self.label = "test"


@pytest.fixture()
def spec() -> lc.LifecycleSpec:
    return lc.LifecycleSpec(age_start=25, age_retire=63, age_death=93)


@pytest.fixture()
def cfg() -> dict:
    return {"utility": {"discount_factor": 0.96, "bequest_weight": 1.0,
                        "bequest_shift": 1.0, "consumption_window": "retirement"}}


def _flat(spec, level=2.0, n=4):
    consumption = np.full((n, spec.horizon), level)
    wealth = np.full((n, spec.horizon + 1), level - 1.0)
    return _Outcome(consumption, wealth)


class TestSurvival:
    def test_it_starts_at_one_and_never_rises(self, spec) -> None:
        s = mo.survival(spec, 88.0, 10.0)
        assert s[0] == pytest.approx(1.0)
        assert np.all(np.diff(s) <= 1e-12)

    def test_a_later_mode_means_more_survivors(self, spec) -> None:
        assert mo.survival(spec, 93.0, 10.0)[-1] > mo.survival(spec, 83.0,
                                                               10.0)[-1]

    def test_the_death_distribution_sums_to_one(self, spec) -> None:
        for modal in (83.0, 88.0, 93.0):
            probs = mo.death_probabilities(mo.survival(spec, modal, 10.0))
            assert probs.sum() == pytest.approx(1.0)
            assert np.all(probs >= -1e-15)

    def test_the_surviving_mass_is_carried_not_dropped(self, spec) -> None:
        # About a fifth of a cohort is still alive at 93 under the baseline
        # law; losing it would understate life expectancy and lose
        # probability, which is the bug this test exists for.
        s = mo.survival(spec, 88.0, 10.0)
        assert float(s[-1]) > 0.1
        assert mo.death_probabilities(s)[-1] == pytest.approx(float(s[-1]))

    def test_a_later_mode_means_a_longer_expected_life(self, spec) -> None:
        short = mo.life_expectancy(spec, mo.survival(spec, 83.0, 10.0))
        long = mo.life_expectancy(spec, mo.survival(spec, 93.0, 10.0))
        assert spec.age_start < short < long <= spec.age_death

    def test_death_ages_line_up_with_the_probabilities(self, spec) -> None:
        s = mo.survival(spec, 88.0, 10.0)
        assert mo.death_ages(spec, s).shape == mo.death_probabilities(s).shape


class TestExactness:
    def test_a_certain_death_reproduces_the_projects_own_cec(self, spec, cfg
                                                             ) -> None:
        rng = np.random.default_rng(0)
        consumption = np.exp(rng.normal(0.0, 0.3, (500, spec.horizon)))
        wealth = np.exp(rng.normal(0.0, 0.4, (500, spec.horizon + 1)))
        outcome = _Outcome(consumption, wealth)

        certain = np.ones(spec.horizon + 1)
        certain[-1] = 0.0
        mine = mo.certainty_equivalent(outcome, spec, cfg, 5.0, certain)

        bundle = ut.bundle_from_outcome(outcome, cfg, spec)
        theirs = ut.crra_certainty_equivalent(
            bundle, 5.0, cfg["utility"]["discount_factor"],
            cfg["utility"]["bequest_weight"], include_bequest=True)
        assert mine == pytest.approx(theirs, rel=1e-12)

    @pytest.mark.parametrize("gamma", [1.0, 2.0, 5.0, 10.0])
    def test_a_certain_stream_returns_itself(self, spec, cfg, gamma) -> None:
        outcome = _flat(spec, level=2.0)
        for survive in (np.append(np.ones(spec.horizon), 0.0),
                        mo.survival(spec, 88.0, 10.0)):
            assert mo.certainty_equivalent(outcome, spec, cfg, gamma,
                                           survive) == pytest.approx(2.0)

    def test_the_window_is_the_projects_window(self, spec, cfg) -> None:
        # Working-life consumption is identical on every strategy, so it must
        # not enter: doubling it may not move the certainty equivalent.
        outcome = _flat(spec)
        louder = _flat(spec)
        louder.consumption[:, :spec.n_working] *= 4.0
        survive = mo.survival(spec, 88.0, 10.0)
        assert mo.certainty_equivalent(outcome, spec, cfg, 5.0, survive) \
            == pytest.approx(
                mo.certainty_equivalent(louder, spec, cfg, 5.0, survive))

    def test_the_full_window_does_see_working_life(self, spec) -> None:
        cfg = {"utility": {"discount_factor": 0.96, "bequest_weight": 1.0,
                           "bequest_shift": 1.0, "consumption_window": "full"}}
        outcome, louder = _flat(spec), _flat(spec)
        louder.consumption[:, :spec.n_working] *= 4.0
        survive = mo.survival(spec, 88.0, 10.0)
        assert mo.certainty_equivalent(louder, spec, cfg, 5.0, survive) \
            > mo.certainty_equivalent(outcome, spec, cfg, 5.0, survive)

    def test_an_unknown_window_is_rejected(self, spec) -> None:
        with pytest.raises(ValueError, match="consumption_window"):
            mo.window(spec, {"utility": {"consumption_window": "nonsense"}})


class TestRuin:
    def test_dying_early_cannot_raise_the_chance_of_outliving_the_money(
            self, spec, cfg) -> None:
        n = 200
        outcome = _flat(spec, n=n)
        outcome.ruin = np.ones(n, dtype=bool)
        outcome.ruin_age = np.full(n, 85)
        certain = np.append(np.ones(spec.horizon), 0.0)
        under_law = mo.probability_of_ruin(
            outcome, spec, mo.survival(spec, 83.0, 10.0), cfg)
        assert under_law < mo.probability_of_ruin(outcome, spec, certain, cfg)

    def test_a_path_that_never_ruins_never_counts(self, spec, cfg) -> None:
        outcome = _flat(spec)
        assert mo.probability_of_ruin(
            outcome, spec, mo.survival(spec, 88.0, 10.0), cfg) == 0.0


class TestVerdict:
    def test_an_unchanged_ordering_is_reported_as_unchanged(self, spec, cfg
                                                            ) -> None:
        rng = np.random.default_rng(1)
        outcomes = {}
        for i, key in enumerate(("good", "bad")):
            level = 2.0 - 0.4 * i
            noise = np.exp(rng.normal(0.0, 0.1, (300, spec.horizon)))
            outcomes[key] = _Outcome(level * noise,
                                     np.full((300, spec.horizon + 1), level))
            outcomes[key].label = key
        frame = mo.compare(outcomes, spec, cfg, 5.0)
        curve = mo.gap_curve(frame, ("good", "bad"))
        found = mo.verdict(frame, curve, ("good", "bad"))
        assert not found["ordering_ever_changes"]
        assert found["winner_is_expected_throughout"]
