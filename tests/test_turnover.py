"""What the schedules trade, and whether the cost arithmetic is right."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import lifecycle as lc
from src import turnover as tn


class _Paths:
    """A deterministic stand-in for BootstrapPaths with known returns."""

    def __init__(self, returns: dict, n_paths: int = 3) -> None:
        self._r = {k: np.tile(np.asarray(v, dtype=float), (n_paths, 1))
                   for k, v in returns.items()}
        self.n_paths = n_paths

    @property
    def horizon(self) -> int:
        return int(next(iter(self._r.values())).shape[1])

    @property
    def inflation(self) -> np.ndarray:
        """Zero real inflation: the feedback spending rules read it, and a
        deterministic stub has nothing to say about it."""
        return np.zeros((self.n_paths, self.horizon))

    def series(self, name: str) -> np.ndarray:
        return self._r[name]


def _flat_paths(horizon: int = 5, **rates) -> _Paths:
    return _Paths({a: [rates.get(a, 0.0)] * horizon for a in lc.ASSETS})


def _strategy(weights) -> lc.Strategy:
    return lc.Strategy(key="s", label="s",
                       weights=np.asarray(weights, dtype=float))


class TestScheduleTurnover:
    def test_constant_weights_never_move(self) -> None:
        w = np.tile([0.5, 0.5, 0.0, 0.0], (6, 1))
        assert np.all(tn.schedule_turnover(w) == 0.0)

    def test_first_year_is_free(self) -> None:
        w = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        assert tn.schedule_turnover(w)[0] == 0.0

    def test_a_full_switch_counts_once(self) -> None:
        """One-way turnover: selling A to buy B is one trade, not two."""
        w = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        assert tn.schedule_turnover(w)[1] == pytest.approx(1.0)


class TestPortfolioTurnover:
    def test_no_drift_when_everything_returns_the_same(self) -> None:
        paths = _flat_paths(5, dom_eq=0.05, intl_eq=0.05, bond=0.05, bill=0.05)
        turn = lc.portfolio_turnover(paths, _strategy(
            np.tile([0.25, 0.25, 0.25, 0.25], (5, 1))))
        assert np.allclose(turn, 0.0)

    def test_single_asset_never_trades(self) -> None:
        paths = _flat_paths(5, dom_eq=0.20, bond=-0.05)
        turn = lc.portfolio_turnover(paths, _strategy(
            np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))))
        assert np.allclose(turn, 0.0)

    def test_drift_is_the_weight_that_ran_away(self) -> None:
        """50/50 with one leg up 20% and the other flat.

        The legs are worth 0.6 and 0.5 of a portfolio now worth 1.1, so the
        weights are 6/11 and 5/11 and restoring 50/50 is a one-way trade of
        1/22. Renormalising by the grown total is the step worth pinning: an
        implementation that compared 0.6 against 0.5 would say 0.1.
        """
        paths = _flat_paths(3, dom_eq=0.20, bond=0.0)
        turn = lc.portfolio_turnover(paths, _strategy(
            np.tile([0.5, 0.0, 0.5, 0.0], (3, 1))))
        assert turn[0, 1] == pytest.approx(1.0 / 22.0)

    def test_constant_weights_make_total_equal_drift(self) -> None:
        """The identity that validates the decomposition.

        A portfolio whose target never moves can only be trading because of
        drift, so the two columns have to agree exactly.
        """
        rng = np.random.default_rng(0)
        paths = _Paths({a: rng.normal(0.04, 0.15, 8) for a in lc.ASSETS})
        strat = _strategy(np.tile([0.4, 0.3, 0.2, 0.1], (8, 1)))
        assert np.allclose(lc.portfolio_turnover(paths, strat),
                           tn.drift_turnover(paths, strat.weights))

    def test_drift_is_not_a_lagged_schedule(self) -> None:
        """A moving schedule must show an excess over its drift floor.

        Measuring drift as "last year's schedule" instead of "last year's
        weights held" would carry the schedule's own move a year late and
        make any schedule look free. This is that regression.
        """
        paths = _flat_paths(4, dom_eq=0.05, intl_eq=0.05, bond=0.05, bill=0.05)
        weights = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0],
                            [1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        total = lc.portfolio_turnover(paths, _strategy(weights)).mean()
        drift = tn.drift_turnover(paths, weights).mean()
        assert drift == pytest.approx(0.0)
        assert total > 0.5


class TestCosts:
    def test_zero_cost_is_the_gross_return_exactly(self) -> None:
        paths = _flat_paths(5, dom_eq=0.10, bond=0.02)
        strat = _strategy(np.tile([0.5, 0.0, 0.5, 0.0], (5, 1)))
        net, turn = lc.net_portfolio_returns(paths, strat, 0.0)
        assert np.array_equal(net, lc.portfolio_returns(paths, strat))
        assert np.all(turn == 0.0)

    def test_cost_compounds_as_one_minus_k_t(self) -> None:
        paths = _flat_paths(6, dom_eq=0.10, bond=0.02)
        strat = _strategy(np.tile([0.5, 0.0, 0.5, 0.0], (6, 1)))
        gross = lc.portfolio_returns(paths, strat)
        net, turn = lc.net_portfolio_returns(paths, strat, 0.01)
        assert np.allclose(net, (1 - 0.01 * turn) * (1 + gross) - 1)

    def test_a_spec_with_no_cost_leaves_the_simulator_untouched(self) -> None:
        assert lc.LifecycleSpec().trading_cost == 0.0

    def test_negative_cost_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="trading_cost"):
            lc.LifecycleSpec(trading_cost=-0.01)


class TestReading:
    def _frame(self, points) -> pd.DataFrame:
        rows = []
        for cost, (a, b) in points.items():
            rows.append({"trading_cost": cost, "strategy": "solved_simplex",
                         "label": "s", "cec_crra_gamma5": a})
            rows.append({"trading_cost": cost, "strategy": "international_equity",
                         "label": "i", "cec_crra_gamma5": b})
        return pd.DataFrame.from_records(rows)

    def test_break_even_interpolates(self) -> None:
        curve = tn.gap_curve(
            self._frame({0.0: (1.10, 1.00), 0.01: (0.90, 1.00)}),
            "solved_simplex", "international_equity")
        # +10% at 0 bp, -10% at 100 bp: the crossing is halfway.
        assert tn.break_even(curve) == pytest.approx(50.0, rel=0.05)

    def test_break_even_is_infinite_when_the_lead_holds(self) -> None:
        curve = tn.gap_curve(
            self._frame({0.0: (1.10, 1.00), 0.01: (1.05, 1.00)}),
            "solved_simplex", "international_equity")
        assert np.isinf(tn.break_even(curve))

    def test_best_fixed_excludes_the_challenger(self) -> None:
        frame = self._frame({0.0: (9.99, 1.00), 0.01: (9.99, 1.00)})
        assert tn.best_fixed(frame, "solved_simplex") == "international_equity"

    def test_best_fixed_is_settled_at_zero_cost(self) -> None:
        """The incumbent must not slide to meet the challenger."""
        rows = []
        for cost, cecs in {0.0: {"a": 1.0, "b": 0.9},
                           0.01: {"a": 0.1, "b": 0.9}}.items():
            for key, value in cecs.items():
                rows.append({"trading_cost": cost, "strategy": key,
                             "label": key, "cec_crra_gamma5": value})
        assert tn.best_fixed(pd.DataFrame.from_records(rows), "x") == "a"

    def test_verdict_reports_the_busiest_and_quietest(self) -> None:
        measured = pd.DataFrame({
            "strategy": ["solved_simplex", "bills_only"],
            "label": ["s", "b"], "turnover_total": [0.15, 0.0],
            "excess_over_drift": [0.13, 0.0]})
        curve = tn.gap_curve(
            self._frame({0.0: (1.10, 1.00), 0.01: (0.90, 1.00)}),
            "solved_simplex", "international_equity")
        found = tn.verdict(curve, measured, {"measured": False},
                           "solved_simplex", "international_equity")
        assert found["busiest_strategy"] == "solved_simplex"
        assert found["quietest_strategy"] == "bills_only"
        assert found["winner_ever_changes"]
        assert not found["survives_whole_grid"]

    def test_a_single_asset_benchmark_has_no_ratio(self) -> None:
        """A portfolio that never rebalances makes the ratio undefined, and
        the prose has to say so rather than print an infinity."""
        measured = pd.DataFrame({
            "strategy": ["solved_simplex", "international_equity"],
            "turnover_total": [0.08, 0.0], "excess_over_drift": [0.07, 0.0]})
        cross = tn.cost_of_the_schedule(measured, "solved_simplex",
                                        "international_equity")
        assert cross["fixed_trades_nothing"]
        assert cross["extra_turnover"] == pytest.approx(0.08)

    def test_cost_of_the_schedule_flags_self_rebalancing(self) -> None:
        measured = pd.DataFrame({
            "strategy": ["target_date_fund", "balanced_all_equity"],
            "turnover_total": [0.04, 0.03], "excess_over_drift": [-0.001, 0.0]})
        cross = tn.cost_of_the_schedule(measured, "target_date_fund",
                                        "balanced_all_equity")
        assert cross["measured"]
        assert cross["solved_is_self_rebalancing"]
        assert cross["extra_turnover"] == pytest.approx(0.01)
