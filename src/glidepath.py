"""Solving for the optimal lifecycle glide path.

``docs/04`` tests a *given* target-date glide path and finds it loses to a
static all-equity portfolio. That leaves the more interesting question
unasked: what shape actually wins? This module answers it by optimising the
age-by-asset weight schedule directly rather than assuming one.

Two searches are run.

**Parametric.** Equity share is a piecewise-linear function of age through a
handful of free knots. The result is a glide path an actual fund could
implement, and its shape is directly comparable to the industry paths in
``docs/03``.

**Free-form.** Every age gets its own allocation, with no smoothness imposed
at all -- 68 free equity shares and 68 free domestic splits. This is the
unrestricted problem, and it is the honest test: if the optimum is genuinely
a declining glide path, an unconstrained search will find one.

Both use **coordinate ascent on a grid**. Under common random numbers the
objective is a deterministic function of the weights, so a grid search over
one coordinate at a time is exact for that coordinate and the sweep is
monotone in the objective -- no gradients, no restarts, no tolerance to tune.
It cannot escape a local optimum in principle; section 4 of ``docs/07``
reports the restart check that was run to look for one.

The cost of the free-form search is the reason :func:`batch_cec` exists: it
evaluates a whole grid of candidate schedules in one vectorised pass, which
is roughly an order of magnitude faster than looping.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import lifecycle as lc
from . import spending as spg
from . import utility as ut
from .bootstrap import BootstrapPaths

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batched evaluation
# ---------------------------------------------------------------------------
class BatchEvaluator:
    """Evaluates many candidate weight schedules against one set of paths.

    The coordinate search calls this tens of thousands of times, so the
    layout is chosen for it. Returns are cached once as ``(H, N, 4)`` and the
    portfolio return for ``K`` candidates is a single batched matmul
    ``(H, N, 4) @ (H, 4, K)`` -- BLAS work rather than a strided ``einsum``,
    which profiling showed was most of the runtime. The whole recursion then
    runs in ``(N, K)`` layout so nothing needs transposing mid-loop.
    """

    def __init__(self, paths: BootstrapPaths, spec: lc.LifecycleSpec,
                 income: np.ndarray, cfg: Mapping[str, Any],
                 spending: spg.SpendingRule | None = None) -> None:
        if paths.horizon < spec.horizon:
            raise ValueError("bootstrap horizon is shorter than the lifecycle")
        self.spec = spec
        self.cfg = cfg
        self.income = income
        self.n_paths = paths.n_paths
        self.rule = spending or spg.from_spec(spec.retirement_rule,
                                              spec.rule_rate)
        horizon = spec.horizon
        # (H, N, 4), contiguous: the batched matmul reads it in this order.
        self._returns = np.ascontiguousarray(
            np.stack([paths.series(a)[:, :horizon] for a in lc.ASSETS], axis=-1)
            .transpose(1, 0, 2))
        self._inflation = np.ascontiguousarray(
            paths.inflation[:, :horizon].T)                     # (H, N)
        self._benefit = spec.social_security_benefit(income.mean(axis=1))
        util = cfg["utility"]
        self._window = str(util.get("consumption_window", "retirement"))
        self._floor = float(util.get("consumption_floor", ut.DEFAULT_FLOOR))
        self._shift = float(util.get("bequest_shift", 1.0))
        self._beta = float(util["discount_factor"])
        self._bequest_weight = float(util["bequest_weight"])
        self._include_bequest = bool(util["bequest_enabled"])

    # -- core recursion -----------------------------------------------------
    def simulate(self, weights: np.ndarray, consumption_from: int = 0
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run ``K`` schedules; returns ``(N, K, H')``, ``(N, K)``, ``(N, K)``.

        Mirrors :func:`src.lifecycle.simulate`. The two agree to
        floating-point tolerance rather than bit-for-bit: the batched path
        accumulates the portfolio return through a BLAS matmul and the
        reference through an ``einsum``, so summation order differs and
        results diverge in the last couple of digits (relative error of order
        1e-15 on consumption, with ruin flags identical). The agreement is
        asserted in ``tests/test_glidepath.py``, which is what stops the two
        implementations from drifting apart in substance.
        """
        spec = self.spec
        if weights.ndim != 3 or weights.shape[2] != len(lc.ASSETS):
            raise ValueError(f"weights must be (K, horizon, {len(lc.ASSETS)})")
        if weights.shape[1] != spec.horizon:
            raise ValueError(
                f"weights horizon {weights.shape[1]} != {spec.horizon}")
        n_k, horizon = weights.shape[0], spec.horizon
        n_paths = self.n_paths

        rp = self._portfolio_returns(weights)

        wealth = np.zeros((n_paths, n_k))
        consumption = np.zeros((n_paths, n_k, horizon - consumption_from))

        for h in range(spec.n_working):
            contribution = spec.savings_rate * self.income[:, h]
            if h >= consumption_from:
                consumption[:, :, h - consumption_from] = \
                    (self.income[:, h] - contribution)[:, None]
            wealth = (wealth + contribution[:, None]) * (1.0 + rp[h])

        wealth_at_retirement = wealth.copy()
        benefit = self._benefit[:, None]
        initial = self.rule.initial_withdrawal(wealth_at_retirement,
                                               spec.n_retired, spec.age_retire)
        prev = initial
        last_return = rp[spec.n_working - 1]
        last_inflation = np.broadcast_to(
            self._inflation[spec.n_working - 1][:, None], (n_paths, n_k))
        ruined = np.zeros((n_paths, n_k), dtype=bool)

        for h in range(spec.n_working, horizon):
            state = spg.SpendingState(
                year=h - spec.n_working,
                age=spec.age_start + h,
                years_remaining=horizon - h,
                wealth=wealth,
                prev_withdrawal=prev,
                initial_withdrawal=initial,
                wealth_at_retirement=wealth_at_retirement,
                last_return=last_return,
                last_inflation=last_inflation,
            )
            desired = np.maximum(self.rule.desired(state), 0.0)
            withdrawal = np.minimum(desired, np.maximum(wealth, 0.0))
            consumption[:, :, h - consumption_from] = benefit + withdrawal
            new_wealth = np.maximum(wealth - withdrawal, 0.0) * (1.0 + rp[h])
            exhausted = (new_wealth <= 0.0) & (h + 1 < horizon)
            ruined |= (~ruined) & exhausted
            wealth = new_wealth
            prev = withdrawal
            last_return = rp[h]
            last_inflation = np.broadcast_to(self._inflation[h][:, None],
                                             (n_paths, n_k))

        return consumption, wealth, ruined

    def _portfolio_returns(self, weights: np.ndarray) -> np.ndarray:
        """Per-year portfolio return for each candidate: ``(H, N, K)``.

        Isolated from :meth:`simulate` so that a subclass can change how a
        weight vector becomes a return without duplicating the recursion.
        :class:`src.leverage.LeveredEvaluator` is the reason it exists: a
        levered portfolio scales the sleeve return and subtracts a borrowing
        cost, but the wealth, withdrawal and utility arithmetic around it is
        identical.
        """
        # (H, N, 4) @ (H, 4, K) -> (H, N, K)
        return np.matmul(self._returns,
                         np.ascontiguousarray(weights.transpose(1, 2, 0)))

    # -- objective ----------------------------------------------------------
    def cec(self, weights: np.ndarray, gamma: float) -> np.ndarray:
        """Certainty equivalent consumption for each of ``K`` schedules."""
        spec = self.spec
        start = spec.n_working if self._window == "retirement" else 0
        consumption, bequest, _ = self.simulate(weights, consumption_from=start)

        block = np.maximum(consumption, self._floor)
        if self._include_bequest:
            beq = np.maximum(self._shift + bequest, self._floor)[:, :, None]
            block = np.concatenate([block, beq], axis=2)
        discount = ut.discount_weights(consumption.shape[2], self._beta,
                                       self._bequest_weight,
                                       self._include_bequest)
        # Aggregate over dates, then over paths -- the same arithmetic as
        # ut.crra_certainty_equivalent, done for all K candidates at once.
        utility = np.tensordot(ut._felicity(block, float(gamma)), discount,
                               axes=([2], [0]))                 # (N, K)
        return np.asarray(
            ut._inverse_felicity(utility.mean(axis=0) / discount.sum(),
                                 float(gamma)), dtype=float)


    def cec_chunked(self, weights: np.ndarray, gamma: float,
                    max_batch: int = 12) -> np.ndarray:
        """:meth:`cec` in slices, for candidate sets too large to hold at once."""
        out = np.empty(weights.shape[0])
        for lo in range(0, weights.shape[0], max_batch):
            hi = min(lo + max_batch, weights.shape[0])
            out[lo:hi] = self.cec(weights[lo:hi], gamma)
        return out


def batch_simulate(
    paths: BootstrapPaths,
    weights: np.ndarray,
    spec: lc.LifecycleSpec,
    income: np.ndarray,
    spending: spg.SpendingRule | None = None,
    consumption_from: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot wrapper around :class:`BatchEvaluator`, in ``(K, N, H)`` order."""
    evaluator = BatchEvaluator(paths, spec, income, {"utility": {
        "consumption_window": "full", "discount_factor": 0.96,
        "bequest_weight": 0.0, "bequest_enabled": False}}, spending)
    consumption, bequest, ruined = evaluator.simulate(weights, consumption_from)
    return (consumption.transpose(1, 0, 2), bequest.T, ruined.T)


def batch_cec(
    paths: BootstrapPaths,
    weights: np.ndarray,
    spec: lc.LifecycleSpec,
    income: np.ndarray,
    cfg: Mapping[str, Any],
    gamma: float,
    spending: spg.SpendingRule | None = None,
) -> np.ndarray:
    """One-shot wrapper around :meth:`BatchEvaluator.cec`."""
    return BatchEvaluator(paths, spec, income, cfg, spending).cec(weights, gamma)


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------
def weights_from_shares(equity: np.ndarray, domestic: np.ndarray,
                        bond_share: float = 0.7) -> np.ndarray:
    """Assemble an ``(H, 4)`` weight matrix from per-age shares.

    ``equity[h]`` is the share of the portfolio in equities at age index
    ``h``; ``domestic[h]`` is the share *of that equity sleeve* held in the
    home market. The fixed-income sleeve is split ``bond_share`` / rest
    between bonds and bills.
    """
    equity = np.clip(np.asarray(equity, dtype=float), 0.0, 1.0)
    domestic = np.clip(np.asarray(domestic, dtype=float), 0.0, 1.0)
    fixed = 1.0 - equity
    return np.column_stack([
        equity * domestic,
        equity * (1.0 - domestic),
        fixed * bond_share,
        fixed * (1.0 - bond_share),
    ])


@dataclasses.dataclass(frozen=True)
class GlideParameterisation:
    """Piecewise-linear equity share through a set of free knots."""

    knot_ages: Tuple[int, ...]
    domestic_share: float = 0.15
    bond_share: float = 0.7

    def build(self, knot_values: Sequence[float],
              spec: lc.LifecycleSpec) -> np.ndarray:
        if len(knot_values) != len(self.knot_ages):
            raise ValueError(
                f"expected {len(self.knot_ages)} knot values, "
                f"got {len(knot_values)}")
        equity = np.interp(spec.ages, np.asarray(self.knot_ages, dtype=float),
                           np.clip(np.asarray(knot_values, dtype=float), 0.0, 1.0))
        domestic = np.full(spec.horizon, self.domestic_share)
        return weights_from_shares(equity, domestic, self.bond_share)

    def strategy(self, knot_values: Sequence[float], spec: lc.LifecycleSpec,
                 key: str = "optimal_parametric",
                 label: str = "Optimal parametric glide path") -> lc.Strategy:
        return lc.Strategy(key=key, label=label,
                           weights=self.build(knot_values, spec))


# ---------------------------------------------------------------------------
# Coordinate ascent
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class OptimisationTrace:
    """What the optimiser did, for the convergence table in docs/07."""

    rows: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def record(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows)


def optimise_parametric(
    evaluator: BatchEvaluator,
    gamma: float,
    parameterisation: GlideParameterisation,
    grid: Sequence[float],
    start: Sequence[float] | None = None,
    n_sweeps: int = 4,
    trace: OptimisationTrace | None = None,
) -> Tuple[np.ndarray, float]:
    """Coordinate ascent over the knot values of a piecewise-linear path."""
    spec = evaluator.spec
    values = (np.full(len(parameterisation.knot_ages), 0.6) if start is None
              else np.array(start, dtype=float))
    grid = np.asarray(grid, dtype=float)
    best = -np.inf
    for sweep in range(n_sweeps):
        improved = False
        for k in range(len(values)):
            candidates = np.repeat(values[None, :], grid.size, axis=0)
            candidates[:, k] = grid
            tensor = np.stack([parameterisation.build(row, spec)
                               for row in candidates])
            scores = evaluator.cec(tensor, gamma)
            pick = int(np.argmax(scores))
            if scores[pick] > best + 1e-12:
                best = float(scores[pick])
                values[k] = grid[pick]
                improved = True
            if trace is not None:
                trace.record(gamma=gamma, sweep=sweep,
                             coordinate=f"knot_age_{parameterisation.knot_ages[k]}",
                             value=float(values[k]), cec=best)
        LOGGER.info("parametric sweep %d (gamma=%.1f): CEC=%.6f", sweep, gamma,
                    best)
        if not improved:
            break
    return values, best


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _band_index(horizon: int, band_years: int) -> np.ndarray:
    """Map each age index to a band, for coarsely parameterised coordinates."""
    return np.minimum(np.arange(horizon) // band_years,
                      (horizon - 1) // band_years)


def optimise_free_form_banded(
    evaluator: BatchEvaluator,
    gamma: float,
    equity_grid: Sequence[float],
    domestic_grid: Sequence[float],
    start_equity: float = 0.85,
    start_domestic: float = 0.2,
    bond_share: float = 0.7,
    domestic_band_years: int = 5,
    n_sweeps: int = 3,
    tolerance: float = 1e-4,
    min_improvement: float = 1e-6,
    trace: OptimisationTrace | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Free equity share at every age; domestic share on multi-year bands.

    The equity share is the decision the literature argues about, so it gets
    a free parameter per year. The domestic split is given coarser bands: it
    moves the answer far less (``docs/05`` section 3.1), and giving it 68 free
    parameters triples the search cost to chase a flat part of the surface.

    ``min_improvement`` is a *relative* threshold a coordinate move must clear
    to be accepted. Without it the search wanders across genuinely flat parts
    of the surface -- late-retirement ages, where the allocation touches only
    a lightly weighted bequest -- and reports single-year spikes that are
    worth a fraction of a basis point. Those spikes look like structure in a
    plotted schedule and are not. :func:`deviation_profile` measures what each
    age is actually worth, rather than asking the reader to trust the shape.
    """
    spec = evaluator.spec
    horizon = spec.horizon
    equity_grid = np.asarray(equity_grid, dtype=float)
    domestic_grid = np.asarray(domestic_grid, dtype=float)
    # Snap the starting point onto the grid. With `min_improvement` in play a
    # coordinate that finds nothing meaningfully better keeps whatever value
    # it started with, so an off-grid start would survive into the reported
    # schedule and read as structure rather than as "did not move".
    equity = np.full(horizon, float(
        equity_grid[np.argmin(np.abs(equity_grid - float(start_equity)))]))
    domestic = np.full(horizon, float(
        domestic_grid[np.argmin(np.abs(domestic_grid - float(start_domestic)))]))
    bands = _band_index(horizon, domestic_band_years)

    best = float(evaluator.cec(
        weights_from_shares(equity, domestic, bond_share)[None], gamma)[0])
    LOGGER.info("free-form start (gamma=%.1f): CEC=%.6f", gamma, best)

    for sweep in range(n_sweeps):
        opening = best
        for h in range(horizon):
            tiled = np.repeat(equity[None, :], equity_grid.size, axis=0)
            tiled[:, h] = equity_grid
            tensor = np.stack([weights_from_shares(row, domestic, bond_share)
                               for row in tiled])
            scores = evaluator.cec(tensor, gamma)
            pick = int(np.argmax(scores))
            if scores[pick] > best * (1.0 + min_improvement):
                best = float(scores[pick])
                equity[h] = equity_grid[pick]

        for band in np.unique(bands):
            mask = bands == band
            tiled = np.repeat(domestic[None, :], domestic_grid.size, axis=0)
            tiled[:, mask] = domestic_grid[:, None]
            tensor = np.stack([weights_from_shares(equity, row, bond_share)
                               for row in tiled])
            scores = evaluator.cec(tensor, gamma)
            pick = int(np.argmax(scores))
            if scores[pick] > best * (1.0 + min_improvement):
                best = float(scores[pick])
                domestic[mask] = domestic_grid[pick]

        gain = (best / opening - 1.0) * 100.0
        LOGGER.info("free-form sweep %d (gamma=%.1f): CEC=%.6f (+%.4f%%)",
                    sweep, gamma, best, gain)
        if trace is not None:
            trace.record(gamma=gamma, sweep=sweep, cec=best,
                         mean_equity=float(equity.mean()),
                         mean_domestic=float(domestic.mean()),
                         gain_pct=gain)
        if gain < tolerance:
            break
    return equity, domestic, best


def schedule_frame(equity: np.ndarray, domestic: np.ndarray,
                   spec: lc.LifecycleSpec, gamma: float,
                   kind: str) -> pd.DataFrame:
    """Tidy age-by-age description of a solved schedule."""
    return pd.DataFrame({
        "kind": kind,
        "risk_aversion": float(gamma),
        "age": spec.ages,
        "equity_share": equity,
        "domestic_share_of_equity": domestic,
    })


def compare_to_benchmarks(
    evaluator: BatchEvaluator,
    solved: Mapping[str, np.ndarray],
    benchmarks: Mapping[str, lc.Strategy],
    gamma: float,
) -> pd.DataFrame:
    """CEC of every solved schedule and every fixed benchmark, side by side."""
    names = list(solved) + list(benchmarks)
    tensor = np.stack([*solved.values(),
                       *(s.weights for s in benchmarks.values())])
    scores = evaluator.cec(tensor, gamma)
    frame = pd.DataFrame({"strategy": names, "risk_aversion": float(gamma),
                          "cec": scores})
    best = frame["cec"].max()
    frame["gap_to_best_pct"] = (frame["cec"] / best - 1.0) * 100.0
    return frame.sort_values("cec", ascending=False).reset_index(drop=True)


def deviation_profile(
    evaluator: BatchEvaluator,
    equity: np.ndarray,
    domestic: np.ndarray,
    gamma: float,
    bond_share: float = 0.7,
    reference_equity: float = 1.0,
) -> pd.DataFrame:
    """What each age's allocation is actually worth.

    For every age, holds the solved schedule fixed and forces that one year to
    ``reference_equity``, reporting the certainty-equivalent cost in basis
    points. A solved schedule can look highly structured when almost all of
    its deviations from a flat line are worth less than a basis point; this is
    the table that separates the two cases, and it is what section 4 of
    ``docs/07`` rests on rather than the eye.
    """
    spec = evaluator.spec
    base = weights_from_shares(equity, domestic, bond_share)
    base_cec = float(evaluator.cec(base[None], gamma)[0])

    candidates = np.repeat(equity[None, :], spec.horizon, axis=0)
    for h in range(spec.horizon):
        candidates[h, h] = reference_equity
    tensor = np.stack([weights_from_shares(row, domestic, bond_share)
                       for row in candidates])
    forced = evaluator.cec_chunked(tensor, gamma)

    return pd.DataFrame({
        "risk_aversion": float(gamma),
        "age": spec.ages,
        "solved_equity_share": equity,
        "cec_solved": base_cec,
        "cec_if_forced_to_reference": forced,
        "cost_of_forcing_bp": (base_cec / forced - 1.0) * 1e4,
    })
