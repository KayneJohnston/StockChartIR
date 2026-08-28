"""Borrowing to invest, and what it is worth at each price of credit.

Every allocation in this project so far is long-only and fully invested: the
weights are non-negative and sum to one. That is a constraint, not a result,
and it rules out a policy with a serious literature behind it. If the case for
equities rests on a horizon long enough for diversification across countries
and decades to work, then a young investor holding only a small financial
balance is under-exposed to the very risk they are being told to take -- and
the natural remedy is to borrow.

The question is never whether leverage raises expected wealth; it obviously
does when the expected asset return exceeds the borrowing rate. The question
is what it is worth to a risk-averse investor **at the price they can
actually borrow at**. This module sweeps that price.

**The mechanics.** An allocation ``x`` over the four assets sums to one. A
leverage ratio ``L`` means holding ``L`` units of that sleeve per unit of
equity capital, funding the difference at the real bill rate plus a spread
``c``:

    r_p = L · (x · r) − (L − 1) · (r_bill + c)

Two modelling choices in that line are load-bearing and are stated here rather
than buried. The borrowing rate floats with the realised real bill return, so
an investor who borrows is exposed to the same rate their cash earns; and the
portfolio return is **clipped at −100%**, which imposes limited liability. The
clip is a margin call: the lender takes what is left and the investor's equity
goes to zero, but they never owe more than they have. That assumption is
generous to leverage, so :func:`wipeout_frequency` reports how often it binds
rather than leaving it implicit.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import allocation as al
from . import glidepath as gp
from . import lifecycle as lc

LOGGER = logging.getLogger(__name__)

#: Index of the bill series within :data:`src.lifecycle.ASSETS`.
BILL = lc.ASSETS.index("bill")

#: A levered portfolio cannot lose more than everything: the lender takes the
#: remainder and the investor's equity is wiped out.
MIN_RETURN = -1.0


def levered_returns(sleeve_return: np.ndarray, bill_return: np.ndarray,
                    leverage: Any, spread: Any) -> np.ndarray:
    """``L·(x·r) − (L−1)·(r_bill + c)``, floored at total loss.

    Every argument is broadcast, so the same function serves the single-path
    reference, the batched evaluator and a per-age leverage schedule.
    """
    leverage = np.asarray(leverage, dtype=float)
    gross = (leverage * sleeve_return
             - (leverage - 1.0) * (bill_return + np.asarray(spread,
                                                            dtype=float)))
    return np.maximum(gross, MIN_RETURN)


class LeveredEvaluator(gp.BatchEvaluator):
    """A :class:`~src.glidepath.BatchEvaluator` that borrows to invest.

    Only the map from weights to a portfolio return changes; the wealth
    recursion, the withdrawal rule and the utility arithmetic are inherited
    unchanged, which is what makes a levered result comparable with every
    unlevered one in this project.

    ``leverage`` may be a scalar or a per-age array, so an age-varying
    leverage schedule costs no extra machinery.
    """

    def __init__(self, *args: Any, leverage: float | np.ndarray = 1.0,
                 spread: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.spread = float(spread)
        self.set_leverage(leverage)

    def set_leverage(self, leverage: float | np.ndarray) -> None:
        """Update the leverage ratio in place, keeping the cached returns.

        Accepts a scalar, a per-age ``(H,)`` schedule, or a ``(H, K)`` block of
        ``K`` competing schedules. The last form is what lets the per-age
        leverage search run as one batched evaluation rather than one
        simulation per candidate.
        """
        array = np.asarray(leverage, dtype=float)
        horizon = self.spec.horizon
        if array.ndim == 0:
            array = np.full(horizon, float(array))
        if array.ndim not in (1, 2) or array.shape[0] != horizon:
            raise ValueError(
                f"leverage must be a scalar or ({horizon},) or ({horizon}, K); "
                f"got {array.shape}")
        if np.any(array < 1.0):
            raise ValueError("leverage below 1 is a cash holding, not "
                             "borrowing; express it through the bill weight")
        self.leverage = array

    def _leverage_block(self) -> np.ndarray:
        """Leverage shaped to broadcast against an ``(H, N, K)`` return array."""
        if self.leverage.ndim == 1:
            return self.leverage[:, None, None]
        return self.leverage[:, None, :]

    def _portfolio_returns(self, weights: np.ndarray) -> np.ndarray:
        sleeve = super()._portfolio_returns(weights)             # (H, N, K)
        return levered_returns(sleeve, self._bill_returns(),
                               self._leverage_block(), self.spread)

    def _bill_returns(self) -> np.ndarray:
        """The real bill series as ``(H, N, 1)``, ready to broadcast over K."""
        return self._returns[:, :, BILL][:, :, None]

    def wipeout_frequency(self, weights: np.ndarray) -> np.ndarray:
        """Share of path-years in which the limited-liability clip binds.

        The clip is the modelling assumption that most favours leverage, so it
        is measured rather than assumed away.
        """
        sleeve = gp.BatchEvaluator._portfolio_returns(self, weights)
        leverage = self._leverage_block()
        gross = (leverage * sleeve
                 - (leverage - 1.0) * (self._bill_returns() + self.spread))
        return (gross < MIN_RETURN).mean(axis=(0, 1))


def make_evaluator(paths: Any, spec: lc.LifecycleSpec, income: np.ndarray,
                   cfg: Mapping[str, Any], leverage: float = 1.0,
                   spread: float = 0.0) -> LeveredEvaluator:
    """Construct a levered evaluator with the project's standard inputs."""
    return LeveredEvaluator(paths, spec, income, cfg, leverage=leverage,
                            spread=spread)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def _constant_schedules(allocations: np.ndarray, horizon: int) -> np.ndarray:
    """``(K, H, 4)`` from ``(K, 4)`` constant allocations."""
    return np.repeat(np.asarray(allocations, dtype=float)[:, None, :],
                     horizon, axis=1)


def sweep_cost_and_leverage(
    evaluator: LeveredEvaluator,
    gamma: float,
    leverage_grid: Sequence[float],
    spread_grid: Sequence[float],
    allocations: np.ndarray,
    max_batch: int = 12,
) -> pd.DataFrame:
    """Best constant allocation at every (borrowing spread, leverage) pair.

    The allocation is held constant over the lifecycle here rather than
    solved per age: the object the question asks for is a leverage ratio and
    the portfolio that goes with it, and a per-age schedule would confound the
    two. :func:`optimise_leverage_schedule` relaxes that separately.
    """
    horizon = evaluator.spec.horizon
    tensor = _constant_schedules(allocations, horizon)
    rows: List[Dict[str, Any]] = []
    for spread in spread_grid:
        evaluator.spread = float(spread)
        for leverage in leverage_grid:
            evaluator.set_leverage(float(leverage))
            scores = evaluator.cec_chunked(tensor, gamma, max_batch=max_batch)
            pick = int(np.argmax(scores))
            weights = np.asarray(allocations)[pick]
            wipe = float(evaluator.wipeout_frequency(tensor[pick][None])[0])
            row: Dict[str, Any] = {
                "spread": float(spread), "leverage": float(leverage),
                "cec": float(scores[pick]),
                "wipeout_share_of_years": wipe}
            row.update({a: float(weights[i])
                        for i, a in enumerate(lc.ASSETS)})
            row["equity"] = float(weights[0] + weights[1])
            rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    unlevered = frame[np.isclose(frame["leverage"], 1.0)] \
        .set_index("spread")["cec"]
    frame["vs_unlevered_pct"] = [
        (r.cec / float(unlevered.loc[r.spread]) - 1.0) * 100.0
        for r in frame.itertuples()]
    return frame


def optimal_by_cost(sweep: pd.DataFrame) -> pd.DataFrame:
    """The best leverage ratio at each borrowing spread."""
    idx = sweep.groupby("spread")["cec"].idxmax()
    out = sweep.loc[idx].sort_values("spread").reset_index(drop=True)
    return out


def break_even_spread(sweep: pd.DataFrame,
                      tolerance: float = 1e-9) -> float:
    """The lowest borrowing spread at which leverage stops being worth having.

    Walks the spread grid in order and interpolates between the last point at
    which the best levered portfolio still beats the unlevered one and the
    first at which it does not. Interpolating on the *advantage* axis instead
    would be wrong: the advantage flattens at exactly zero once the optimum
    has already dropped to 1x, so a search over that axis lands on the last
    tied point rather than the first one.
    """
    best = optimal_by_cost(sweep).sort_values("spread")
    advantage = best["vs_unlevered_pct"].to_numpy(dtype=float)
    spreads = best["spread"].to_numpy(dtype=float)
    if advantage.size == 0 or advantage[0] <= tolerance:
        return 0.0
    crossing = np.flatnonzero(advantage <= tolerance)
    if crossing.size == 0:
        return float("inf")
    i = int(crossing[0])
    lo_a, hi_a = advantage[i - 1], advantage[i]
    if lo_a == hi_a:
        return float(spreads[i])
    weight = lo_a / (lo_a - hi_a)
    return float(spreads[i - 1] + weight * (spreads[i] - spreads[i - 1]))


def optimise_leverage_schedule(
    evaluator: LeveredEvaluator,
    gamma: float,
    weights: np.ndarray,
    leverage_grid: Sequence[float],
    spread: float,
    n_sweeps: int = 2,
    min_improvement: float = 1e-6,
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """Solve for a leverage ratio at every age, holding the allocation fixed.

    This is the Ayres–Nalebuff question: a young investor's financial balance
    is small relative to the lifetime saving still to come, so a constant
    *share* of a small balance is a small share of lifetime exposure. If that
    argument holds, the solved schedule should lever early and delever later —
    and if it does not, the schedule will say so.
    """
    horizon = evaluator.spec.horizon
    grid = np.asarray(leverage_grid, dtype=float)
    schedule = np.full(horizon, float(grid.min()))
    evaluator.spread = float(spread)
    single = weights[None] if weights.ndim == 2 else weights
    # The whole grid for one age is evaluated in a single batched pass: the
    # allocation is identical across candidates, so only the leverage column
    # differs and the (H, K) form of `set_leverage` carries it.
    tensor = np.repeat(single[:1], grid.size, axis=0)

    evaluator.set_leverage(schedule)
    best = float(evaluator.cec(single, gamma)[0])
    rows: List[Dict[str, Any]] = [
        {"sweep": -1, "cec": best, "mean_leverage": float(schedule.mean())}]
    for sweep in range(n_sweeps):
        opening = best
        for h in range(horizon):
            candidates = np.repeat(schedule[:, None], grid.size, axis=1)
            candidates[h, :] = grid
            evaluator.set_leverage(candidates)
            scores = evaluator.cec(tensor, gamma)
            pick = int(np.argmax(scores))
            if scores[pick] > best * (1.0 + min_improvement):
                best = float(scores[pick])
                schedule[h] = grid[pick]
        evaluator.set_leverage(schedule)
        gain = (best / opening - 1.0) * 100.0
        rows.append({"sweep": sweep, "cec": best, "gain_pct": gain,
                     "mean_leverage": float(schedule.mean())})
        LOGGER.info("leverage schedule sweep %d (gamma=%.1f): CEC=%.6f "
                    "(+%.4f%%)", sweep, gamma, best, gain)
        if gain <= 1e-9:
            break
    return schedule, best, pd.DataFrame.from_records(rows)


def outcome_detail(evaluator: LeveredEvaluator, weights: np.ndarray,
                   spec: lc.LifecycleSpec, cfg: Mapping[str, Any],
                   gammas: Sequence[float], extra: Mapping[str, Any] | None = None
                   ) -> Dict[str, Any]:
    """Distributional detail for one levered configuration.

    A certainty equivalent alone cannot show what leverage does to the shape
    of the outcome, and the shape is the whole argument: the levered
    distribution should be wider on both sides, and the question is whether
    the left side is wide enough to matter.
    """
    tensor = weights[None] if weights.ndim == 2 else weights
    consumption, bequest, ruined = evaluator.simulate(
        tensor, consumption_from=spec.n_working)
    retirement = consumption[:, 0, :].mean(axis=1)
    row: Dict[str, Any] = dict(extra or {})
    row.update({
        "prob_ruin": float(ruined[:, 0].mean()),
        "wipeout_share_of_years": float(
            evaluator.wipeout_frequency(tensor)[0]),
        "median_retirement_consumption": float(np.median(retirement)),
        "mean_retirement_consumption": float(retirement.mean()),
        "median_bequest": float(np.median(bequest[:, 0])),
        "prob_zero_bequest": float((bequest[:, 0] <= 0.0).mean()),
    })
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        row[f"p{q}_retirement_consumption"] = float(
            np.percentile(retirement, q))
    for gamma in gammas:
        row[f"cec_gamma{float(gamma):g}"] = float(
            evaluator.cec(tensor, float(gamma))[0])
    return row


def schedule_by_decade(schedule: pd.DataFrame) -> pd.DataFrame:
    """Average solved leverage by decade of age.

    The per-age solution is jittery: the surface is flat enough that a
    coordinate search finds tiny improvements moving one year between adjacent
    grid values, and a raw plot of it reads as structure it does not have.
    Aggregating to decades reports the trend the schedule genuinely carries
    without asking the reader to trust the year-to-year pattern.
    """
    frame = schedule.copy()
    frame["decade"] = (frame["age"] // 10 * 10).astype(int)
    out = (frame.groupby(["spread", "decade"])
           .agg(mean_leverage=("leverage", "mean"),
                min_leverage=("leverage", "min"),
                max_leverage=("leverage", "max"),
                years=("leverage", "size"))
           .reset_index())
    return out.sort_values(["spread", "decade"]).reset_index(drop=True)

