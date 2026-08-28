"""Tests for the levered evaluator and the cost-of-credit sweeps.

The load-bearing test is the first one: a levered evaluator at L = 1 must
reproduce the unlevered one exactly, because every comparison in ``docs/13``
is between a levered configuration and an unlevered baseline computed by a
different class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import allocation as al
from src import bootstrap as bs
from src import glidepath as gp
from src import leverage as lv
from src import lifecycle as lc


@pytest.fixture()
def setup(toy_panel, toy_config):
    spec = lc.spec_from_config(toy_config)
    paths = bs.from_config(toy_panel, toy_config,
                           horizon_years=spec.horizon).sample(200, chunk_size=100)
    income = lc.simulate_income(spec, 200, rng=np.random.default_rng(3))
    return paths, spec, income, toy_config


def constant(spec: lc.LifecycleSpec, weights) -> np.ndarray:
    return np.tile(np.asarray(weights, dtype=float), (spec.horizon, 1))[None]


class TestLeveredReturns:
    def test_unit_leverage_is_the_sleeve_return(self) -> None:
        sleeve = np.array([0.10, -0.20, 0.0])
        bill = np.array([0.01, 0.01, 0.01])
        assert np.allclose(lv.levered_returns(sleeve, bill, 1.0, 0.03), sleeve)

    def test_the_spread_is_only_paid_on_borrowed_money(self) -> None:
        sleeve, bill = np.array([0.10]), np.array([0.01])
        free = lv.levered_returns(sleeve, bill, 2.0, 0.0)
        costly = lv.levered_returns(sleeve, bill, 2.0, 0.02)
        assert float((free - costly)[0]) == pytest.approx(0.02)

    def test_it_matches_the_formula(self) -> None:
        sleeve, bill, L, c = np.array([0.10]), np.array([0.01]), 2.5, 0.015
        expected = L * 0.10 - (L - 1.0) * (0.01 + c)
        assert float(lv.levered_returns(sleeve, bill, L, c)[0]) \
            == pytest.approx(expected)

    def test_losses_are_capped_at_total_loss(self) -> None:
        got = lv.levered_returns(np.array([-0.6]), np.array([0.0]), 3.0, 0.02)
        assert float(got[0]) == pytest.approx(lv.MIN_RETURN)

    def test_leverage_broadcasts_over_ages(self) -> None:
        got = lv.levered_returns(np.array([0.1, 0.1]), np.array([0.0, 0.0]),
                                 np.array([1.0, 2.0]), 0.0)
        assert np.allclose(got, [0.1, 0.2])


class TestLeveredEvaluator:
    def test_unit_leverage_reproduces_the_unlevered_evaluator(self, setup
                                                              ) -> None:
        paths, spec, income, cfg = setup
        weights = constant(spec, [0.4, 0.3, 0.2, 0.1])
        plain = gp.BatchEvaluator(paths, spec, income, cfg)
        levered = lv.make_evaluator(paths, spec, income, cfg, leverage=1.0,
                                    spread=0.03)
        assert float(levered.cec(weights, 5.0)[0]) \
            == pytest.approx(float(plain.cec(weights, 5.0)[0]), rel=0, abs=0)

    def test_leverage_raises_the_mean_and_the_spread_lowers_it(self, setup
                                                               ) -> None:
        paths, spec, income, cfg = setup
        weights = constant(spec, [0.5, 0.5, 0.0, 0.0])
        free = lv.make_evaluator(paths, spec, income, cfg, leverage=2.0,
                                 spread=0.0)
        costly = lv.make_evaluator(paths, spec, income, cfg, leverage=2.0,
                                   spread=0.05)
        _, free_bequest, _ = free.simulate(weights)
        _, costly_bequest, _ = costly.simulate(weights)
        assert float(np.median(free_bequest)) > float(np.median(costly_bequest))

    def test_set_leverage_accepts_a_schedule(self, setup) -> None:
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg)
        ev.set_leverage(np.linspace(2.0, 1.0, spec.horizon))
        assert ev.leverage.shape == (spec.horizon,)

    def test_a_schedule_of_the_wrong_length_is_rejected(self, setup) -> None:
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg)
        with pytest.raises(ValueError, match="scalar or"):
            ev.set_leverage(np.ones(spec.horizon + 1))

    def test_a_block_of_schedules_is_accepted(self, setup) -> None:
        """(H, K) leverage is what makes the per-age search batched."""
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg)
        ev.set_leverage(np.ones((spec.horizon, 3)) * [1.0, 1.5, 2.0])
        weights = np.repeat(constant(spec, [0.5, 0.5, 0.0, 0.0]), 3, axis=0)
        scores = ev.cec(weights, 5.0)
        assert scores.shape == (3,)
        # Column k must match a run with that constant leverage on its own.
        # To floating-point tolerance rather than bit for bit: a K = 3 matmul
        # and a K = 1 matmul take different BLAS paths and so sum in a
        # different order, exactly as src.glidepath already documents for its
        # own equivalence check.
        ev.set_leverage(1.5)
        alone = float(ev.cec(constant(spec, [0.5, 0.5, 0.0, 0.0]), 5.0)[0])
        assert float(scores[1]) == pytest.approx(alone, rel=1e-12)

    def test_deleveraging_below_one_is_rejected(self, setup) -> None:
        """Holding cash is a bill weight, not negative borrowing."""
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg)
        with pytest.raises(ValueError, match="below 1"):
            ev.set_leverage(0.5)

    def test_wipeout_is_impossible_unlevered(self, setup) -> None:
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg, leverage=1.0)
        weights = constant(spec, [0.5, 0.5, 0.0, 0.0])
        assert float(ev.wipeout_frequency(weights)[0]) == 0.0

    def test_wipeout_rises_with_leverage(self, setup) -> None:
        paths, spec, income, cfg = setup
        weights = constant(spec, [0.5, 0.5, 0.0, 0.0])
        low = lv.make_evaluator(paths, spec, income, cfg, leverage=1.5)
        high = lv.make_evaluator(paths, spec, income, cfg, leverage=8.0)
        assert float(high.wipeout_frequency(weights)[0]) \
            >= float(low.wipeout_frequency(weights)[0])


class TestSweeps:
    def test_the_sweep_covers_the_whole_grid(self, setup) -> None:
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg)
        sweep = lv.sweep_cost_and_leverage(
            ev, 5.0, [1.0, 1.5], [0.0, 0.02], al.simplex_lattice(0.5))
        assert len(sweep) == 4
        assert np.allclose(
            sweep[np.isclose(sweep["leverage"], 1.0)]["vs_unlevered_pct"], 0.0)

    def test_weights_reported_by_the_sweep_are_feasible(self, setup) -> None:
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg)
        sweep = lv.sweep_cost_and_leverage(
            ev, 5.0, [1.0, 2.0], [0.0], al.simplex_lattice(0.5))
        assert np.allclose(sweep[list(lc.ASSETS)].sum(axis=1), 1.0)

    def test_optimal_by_cost_returns_one_row_per_spread(self, setup) -> None:
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg)
        sweep = lv.sweep_cost_and_leverage(
            ev, 5.0, [1.0, 1.5, 2.0], [0.0, 0.01, 0.02],
            al.simplex_lattice(0.5))
        best = lv.optimal_by_cost(sweep)
        assert len(best) == 3
        assert list(best["spread"]) == sorted(best["spread"])


class TestBreakEven:
    def frame(self, spreads, advantage, leverage) -> pd.DataFrame:
        return pd.DataFrame({"spread": spreads, "leverage": leverage,
                             "cec": np.ones(len(spreads)),
                             "vs_unlevered_pct": advantage})

    def test_it_finds_the_first_crossing_not_the_last(self) -> None:
        """The advantage flattens at zero, so a search on that axis misleads."""
        frame = self.frame([0.0, 0.01, 0.03], [3.1, 0.0, 0.0], [1.5, 1.0, 1.0])
        assert lv.break_even_spread(frame) == pytest.approx(0.01)

    def test_it_interpolates_between_grid_points(self) -> None:
        frame = self.frame([0.0, 0.01, 0.02], [4.0, 2.0, -2.0],
                           [2.0, 1.5, 1.0])
        assert lv.break_even_spread(frame) == pytest.approx(0.015)

    def test_never_worth_it_is_zero(self) -> None:
        frame = self.frame([0.0, 0.01], [0.0, 0.0], [1.0, 1.0])
        assert lv.break_even_spread(frame) == 0.0

    def test_always_worth_it_is_infinite(self) -> None:
        frame = self.frame([0.0, 0.01], [3.0, 1.0], [2.0, 2.0])
        assert lv.break_even_spread(frame) == float("inf")


class TestSchedule:
    def test_the_solved_schedule_is_monotone_in_the_objective(self, setup
                                                              ) -> None:
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg)
        weights = constant(spec, [0.5, 0.5, 0.0, 0.0])
        schedule, best, trace = lv.optimise_leverage_schedule(
            ev, 5.0, weights, [1.0, 1.5, 2.0], 0.0, n_sweeps=1)
        assert schedule.shape == (spec.horizon,)
        assert list(trace["cec"]) == sorted(trace["cec"])
        assert best == pytest.approx(float(trace["cec"].iloc[-1]))

    def test_outcome_detail_reports_the_expected_fields(self, setup) -> None:
        paths, spec, income, cfg = setup
        ev = lv.make_evaluator(paths, spec, income, cfg, leverage=1.5)
        weights = np.tile([0.5, 0.5, 0.0, 0.0], (spec.horizon, 1))
        row = lv.outcome_detail(ev, weights, spec, cfg, [2.0, 5.0])
        for key in ("prob_ruin", "median_retirement_consumption",
                    "p5_retirement_consumption", "cec_gamma5",
                    "wipeout_share_of_years"):
            assert key in row
        assert 0.0 <= row["prob_ruin"] <= 1.0
