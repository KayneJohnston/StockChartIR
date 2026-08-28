"""Tests for the full four-asset simplex solver.

Most of these pin down the geometry: a search over the simplex is only valid
if every candidate it generates is feasible, and a candidate set that quietly
drifts off the simplex would still produce a plausible-looking schedule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import allocation as al
from src import bootstrap as bs
from src import glidepath as gp
from src import lifecycle as lc


@pytest.fixture()
def evaluator(toy_panel, toy_config):
    spec = lc.spec_from_config(toy_config)
    paths = bs.from_config(toy_panel, toy_config,
                           horizon_years=spec.horizon).sample(200, chunk_size=100)
    income = lc.simulate_income(spec, 200, rng=np.random.default_rng(3))
    return gp.BatchEvaluator(paths, spec, income, toy_config), spec, toy_config


class TestSimplexLattice:
    @pytest.mark.parametrize("step,expected", [(0.5, 10), (0.25, 35),
                                               (0.2, 56), (0.1, 286)])
    def test_lattice_size_is_the_composition_count(self, step, expected) -> None:
        assert len(al.simplex_lattice(step)) == expected

    def test_every_lattice_point_is_on_the_simplex(self) -> None:
        lattice = al.simplex_lattice(0.25)
        assert np.allclose(lattice.sum(axis=1), 1.0)
        assert (lattice >= 0.0).all()

    def test_lattice_contains_every_corner(self) -> None:
        lattice = al.simplex_lattice(0.25)
        for i in range(al.N_ASSETS):
            corner = np.zeros(al.N_ASSETS)
            corner[i] = 1.0
            assert any(np.allclose(row, corner) for row in lattice)

    def test_lattice_points_are_unique(self) -> None:
        lattice = al.simplex_lattice(0.25)
        assert len(np.unique(lattice, axis=0)) == len(lattice)

    def test_rejects_a_non_positive_step(self) -> None:
        with pytest.raises(ValueError, match="positive fraction"):
            al.simplex_lattice(0.0)


class TestExchangeNeighbourhood:
    def test_stays_on_the_simplex(self) -> None:
        nbhd = al.exchange_neighbourhood(np.array([0.4, 0.3, 0.2, 0.1]), 0.05)
        assert np.allclose(nbhd.sum(axis=1), 1.0)
        assert (nbhd >= -1e-12).all()

    def test_includes_the_incumbent(self) -> None:
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        nbhd = al.exchange_neighbourhood(weights, 0.05)
        assert np.allclose(nbhd[0], weights)

    def test_interior_point_has_all_twelve_exchanges(self) -> None:
        assert len(al.exchange_neighbourhood(
            np.array([0.25, 0.25, 0.25, 0.25]), 0.05)) == 1 + len(al.PAIRS)

    def test_a_corner_can_only_move_outward(self) -> None:
        """From a corner only the three exchanges that leave it are feasible."""
        nbhd = al.exchange_neighbourhood(np.array([1.0, 0.0, 0.0, 0.0]), 0.05)
        assert len(nbhd) == 1 + (al.N_ASSETS - 1)

    def test_a_weight_smaller_than_the_step_cannot_be_drawn_down(self) -> None:
        nbhd = al.exchange_neighbourhood(np.array([0.98, 0.02, 0.0, 0.0]), 0.05)
        assert (nbhd >= -1e-12).all()
        # only the large weight can give anything away
        assert len(nbhd) == 1 + (al.N_ASSETS - 1)


class TestSolver:
    def test_the_solution_stays_on_the_simplex(self, evaluator) -> None:
        ev, spec, cfg = evaluator
        schedule, _, _ = al.optimise_full_simplex(
            ev, 5.0, coarse_sweeps=1, fine_sweeps=1)
        assert schedule.shape == (spec.horizon, al.N_ASSETS)
        assert np.allclose(schedule.sum(axis=1), 1.0)
        assert (schedule >= -1e-12).all()

    def test_the_objective_never_falls(self, evaluator) -> None:
        """Coordinate ascent under common random numbers must be monotone."""
        ev, spec, cfg = evaluator
        _, best, trace = al.optimise_full_simplex(
            ev, 5.0, coarse_sweeps=1, fine_sweeps=2)
        assert list(trace["cec"]) == sorted(trace["cec"])
        assert best == pytest.approx(float(trace["cec"].iloc[-1]))

    def test_it_beats_the_starting_point(self, evaluator) -> None:
        ev, spec, cfg = evaluator
        start = [0.0, 0.0, 0.0, 1.0]
        _, best, trace = al.optimise_full_simplex(
            ev, 5.0, start=start, coarse_sweeps=1, fine_sweeps=1)
        assert best > float(trace["cec"].iloc[0])

    def test_the_reported_cec_matches_a_fresh_evaluation(self, evaluator) -> None:
        ev, spec, cfg = evaluator
        schedule, best, _ = al.optimise_full_simplex(
            ev, 5.0, coarse_sweeps=1, fine_sweeps=1)
        assert float(ev.cec(schedule[None], 5.0)[0]) == pytest.approx(best)

    def test_a_start_off_the_simplex_is_normalised(self, evaluator) -> None:
        ev, spec, cfg = evaluator
        schedule, _, _ = al.optimise_full_simplex(
            ev, 5.0, start=[2.0, 2.0, 2.0, 2.0], coarse_sweeps=1,
            fine_sweeps=0)
        assert np.allclose(schedule.sum(axis=1), 1.0)


class TestReporting:
    def test_schedule_frame_has_a_row_per_age(self, evaluator) -> None:
        ev, spec, cfg = evaluator
        schedule = np.tile([0.4, 0.3, 0.2, 0.1], (spec.horizon, 1))
        frame = al.schedule_frame(schedule, spec, 5.0)
        assert len(frame) == spec.horizon
        assert set(frame["phase"]) == {"working", "retired"}
        assert np.allclose(frame["equity"], 0.7)
        assert np.allclose(frame["domestic_share_of_equity"], 0.4 / 0.7)
        assert np.allclose(frame["bond_share_of_fixed"], 0.2 / 0.3)

    def test_schedule_frame_survives_an_all_equity_row(self, evaluator) -> None:
        """The fixed-income share can be zero; the ratio must not blow up."""
        ev, spec, cfg = evaluator
        schedule = np.tile([0.5, 0.5, 0.0, 0.0], (spec.horizon, 1))
        frame = al.schedule_frame(schedule, spec, 5.0)
        assert np.isfinite(frame["bond_share_of_fixed"]).all()

    def test_phase_summary_weights_still_sum_to_one(self, evaluator) -> None:
        ev, spec, cfg = evaluator
        frame = al.schedule_frame(
            np.tile([0.4, 0.3, 0.2, 0.1], (spec.horizon, 1)), spec, 5.0)
        phases = al.phase_summary(frame)
        assert len(phases) == 2
        total = phases[list(lc.ASSETS)].sum(axis=1)
        assert np.allclose(total, 1.0)

    def test_deviation_profile_is_zero_for_a_constant_schedule(self, evaluator
                                                               ) -> None:
        """Resetting an age to the average changes nothing if all ages agree."""
        ev, spec, cfg = evaluator
        schedule = np.tile([0.4, 0.3, 0.2, 0.1], (spec.horizon, 1))
        profile = al.deviation_profile(ev, schedule, 5.0, spec)
        assert len(profile) == spec.horizon
        assert np.allclose(profile["cost_of_resetting_bp"], 0.0, atol=1e-6)

    def test_deviation_profile_is_positive_at_a_solved_optimum(self, evaluator
                                                               ) -> None:
        ev, spec, cfg = evaluator
        schedule, _, _ = al.optimise_full_simplex(
            ev, 5.0, coarse_sweeps=1, fine_sweeps=1)
        profile = al.deviation_profile(ev, schedule, 5.0, spec)
        assert profile["cost_of_resetting_bp"].max() >= -1e-6

    def test_comparison_includes_the_solved_schedule(self, evaluator) -> None:
        ev, spec, cfg = evaluator
        strategies = lc.build_strategies(cfg, spec)
        solved = {5.0: np.tile([0.5, 0.5, 0.0, 0.0], (spec.horizon, 1))}
        frame = al.compare_to_benchmarks(ev, solved, strategies, [5.0])
        assert "full_simplex_optimal" in set(frame["strategy"])
        assert float(frame["gap_to_best_pct"].max()) == pytest.approx(0.0)
        assert (frame["gap_to_best_pct"] <= 1e-9).all()

    def test_restart_check_reports_every_start(self, evaluator) -> None:
        ev, spec, cfg = evaluator
        starts = [[0.25] * 4, [0.5, 0.5, 0.0, 0.0]]
        frame, schedule, best = al.restart_check(
            ev, 5.0, starts, coarse_sweeps=1, fine_sweeps=0)
        assert len(frame) == len(starts)
        assert float(frame["gap_to_best_pct"].max()) == pytest.approx(0.0)
        assert best == pytest.approx(float(frame["solved_cec"].max()))
        assert np.allclose(schedule.sum(axis=1), 1.0)
