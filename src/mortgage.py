"""A mortgage on the housing sleeve, solved as a loan-to-value ratio by age.

`src.housing` prices housing as an unlevered asset. That is not how households
hold it. A house is the one asset an ordinary person can borrow eighty percent
against at a rate close to the government's, and leaving that out understates
what housing does to a lifetime as surely as leaving the holding cost out
overstates it.

**The parameterisation.** The decision variable is the loan-to-value ratio,
because that is the number a lender quotes and a borrower chooses. An LVR of
``L/V = lambda`` on a property worth ``V`` funded with equity ``E = (1 -
lambda)V`` returns, on that equity,

    r_E = (r_H - lambda * i) / (1 - lambda)

where ``r_H`` is the real return on the property and ``i`` the real mortgage
rate. That is algebraically the leverage multiple ``1 / (1 - lambda)`` applied
to housing alone, so :func:`src.leverage.levered_returns` does the arithmetic
and the two studies stay consistent.

**The rate.** The mortgage is priced off the borrower's *own* country's real
short rate plus a spread, drawn on the same block as every other series, so a
lifetime that lives through high real rates pays them. The spread is swept
rather than assumed: a mortgage is not free, the right margin differs by
country and era, and reporting the whole curve is more honest than defending
one number.

**Limited liability.** Equity is wiped out, not driven negative: the levered
return is floored at total loss, which is the non-recourse assumption. In
recourse jurisdictions the borrower stays liable beyond the property, so this
is the assumption most favourable to borrowing, and
:meth:`MortgageEvaluator.negative_equity_frequency` measures how often it
binds instead of leaving it buried.

**What this is not.** The schedule is rebalanced annually, like everything
else in this project, which for a mortgage means costlessly redrawing the loan
each year to hit a target LVR. Real mortgages amortise on a fixed schedule,
cost several percent of the property to refinance, and are called on missed
payments rather than on a drifting LVR. The result below is the value of the
leverage, not a financing plan.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import glidepath as gp
from . import housing as hs
from . import leverage as lev
from . import lifecycle as lc

LOGGER = logging.getLogger(__name__)

#: The regulatory and practical ceiling most lenders impose on an
#: owner-occupier without mortgage insurance. Above it the product changes
#: rather than merely getting dearer, so the grid stops here.
LVR_CAP = 0.80

#: Loan-to-value ratios the search may choose from. Fine enough to locate a
#: corner, coarse enough that a per-age search stays affordable.
DEFAULT_LVR_GRID: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                       0.7, 0.8)


def leverage_multiple(lvr: Any) -> np.ndarray:
    """``1 / (1 - lambda)``: the gross exposure one unit of equity carries."""
    array = np.asarray(lvr, dtype=float)
    if np.any(array < 0.0) or np.any(array >= 1.0):
        raise ValueError(
            "loan-to-value must lie in [0, 1); an LVR of 1 is infinite "
            "leverage and above it the borrower has no equity at all")
    return 1.0 / (1.0 - array)


class MortgageEvaluator(gp.BatchEvaluator):
    """A five-asset evaluator whose housing sleeve carries a mortgage.

    Only the housing column is levered. The portfolio return is therefore
    split rather than formed in one matmul: the other four assets go through
    the inherited batched product, and the housing term is added separately so
    that a *per-age, per-candidate* loan-to-value schedule costs one broadcast
    rather than one simulation each.
    """

    def __init__(self, *args: Any, spread: float = 0.0,
                 lvr: float | np.ndarray = 0.0,
                 rate_base: str = "bill", **kwargs: Any) -> None:
        kwargs.setdefault("assets", hs.ASSETS)
        super().__init__(*args, **kwargs)
        if hs.HOUSING not in self.assets:
            raise ValueError(
                f"a mortgage needs a housing sleeve to secure; assets are "
                f"{self.assets}")
        self.housing_index = list(self.assets).index(hs.HOUSING)
        if rate_base not in ("bill", "bond"):
            raise ValueError(
                f"rate_base must be 'bill' or 'bond'; got {rate_base!r}")
        self.rate_base = rate_base
        self.spread = float(spread)
        self._base_index = list(self.assets).index(rate_base)
        # (H, N), cached once: the unlevered housing return and the rate the
        # loan against it is priced off.
        self._housing = np.ascontiguousarray(
            self._returns[:, :, self.housing_index])
        self._base_rate = np.ascontiguousarray(
            self._returns[:, :, self._base_index])
        self.set_lvr(lvr)

    # -- the decision variable ---------------------------------------------
    def set_lvr(self, lvr: float | np.ndarray) -> None:
        """Set the loan-to-value schedule in place, keeping cached returns.

        Accepts a scalar, a per-age ``(H,)`` schedule, or an ``(H, K)`` block
        of competing schedules -- the last is what lets the per-age search run
        as one batched evaluation.
        """
        array = np.asarray(lvr, dtype=float)
        horizon = self.spec.horizon
        if array.ndim == 0:
            array = np.full(horizon, float(array))
        if array.ndim not in (1, 2) or array.shape[0] != horizon:
            raise ValueError(
                f"lvr must be a scalar or ({horizon},) or ({horizon}, K); "
                f"got {array.shape}")
        if np.any(array < 0.0):
            raise ValueError("a negative loan-to-value is a deposit, not a "
                             "mortgage; express it through the bill weight")
        if np.any(array > LVR_CAP + 1e-12):
            raise ValueError(
                f"loan-to-value above the {LVR_CAP:.0%} cap; no ordinary "
                "borrower is offered that without mortgage insurance, which "
                "this model does not price")
        self.lvr = array

    def _lvr_block(self) -> np.ndarray:
        """LVR shaped to broadcast against an ``(H, N, K)`` return array."""
        if self.lvr.ndim == 1:
            return self.lvr[:, None, None]
        return self.lvr[:, None, :]

    def levered_housing(self) -> np.ndarray:
        """``(H, N, K)`` return on the housing *equity*, after the mortgage."""
        lvr = self._lvr_block()
        multiple = 1.0 / (1.0 - lvr)
        return lev.levered_returns(self._housing[:, :, None],
                                   self._base_rate[:, :, None],
                                   multiple, self.spread)

    # -- the only thing that changes ---------------------------------------
    def _portfolio_returns(self, weights: np.ndarray) -> np.ndarray:
        w = np.ascontiguousarray(weights.transpose(1, 2, 0))     # (H, A, K)
        without_housing = w.copy()
        without_housing[:, self.housing_index, :] = 0.0
        other = np.matmul(self._returns, without_housing)        # (H, N, K)
        housing_weight = w[:, self.housing_index, :]             # (H, K)
        return other + self.levered_housing() * housing_weight[:, None, :]

    # -- diagnostics --------------------------------------------------------
    def negative_equity_frequency(self) -> np.ndarray:
        """Share of path-years in which the limited-liability floor binds.

        The floor is the assumption most favourable to borrowing, so it is
        measured. A high number means the reported certainty equivalent is
        being propped up by the borrower's right to walk away.
        """
        lvr = self._lvr_block()
        multiple = 1.0 / (1.0 - lvr)
        gross = (multiple * self._housing[:, :, None]
                 - (multiple - 1.0) * (self._base_rate[:, :, None]
                                       + self.spread))
        return (gross < lev.MIN_RETURN).mean(axis=(0, 1))

    def cec_over_lvr(self, weights: np.ndarray, schedules: np.ndarray,
                     gamma: float, max_batch: int = 12) -> np.ndarray:
        """Score ``K`` loan-to-value schedules against one weight schedule.

        ``schedules`` is ``(K, H)``. The weights are tiled to match so that the
        inherited recursion sees a consistent ``K``, which costs a few hundred
        kilobytes and saves writing a second simulator.
        """
        schedules = np.atleast_2d(np.asarray(schedules, dtype=float))
        n_k = schedules.shape[0]
        tiled = np.repeat(np.asarray(weights, dtype=float)[None, :, :],
                          n_k, axis=0)
        out = np.empty(n_k)
        for lo in range(0, n_k, max_batch):
            hi = min(lo + max_batch, n_k)
            self.set_lvr(schedules[lo:hi].T)                     # (H, k)
            out[lo:hi] = self.cec(tiled[lo:hi], gamma)
        return out


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------
def best_constant_lvr(evaluator: MortgageEvaluator, weights: np.ndarray,
                      gamma: float,
                      grid: Sequence[float] = DEFAULT_LVR_GRID,
                      max_batch: int = 12
                      ) -> Tuple[float, float, pd.DataFrame]:
    """The single loan-to-value ratio, held for life, that scores best.

    Returns the ratio, its certainty equivalent, and the whole curve -- the
    curve matters more than the maximum, because a peak that is flat either
    side is a different recommendation from one that is sharp.
    """
    horizon = evaluator.spec.horizon
    schedules = np.array([np.full(horizon, float(v)) for v in grid])
    scores = evaluator.cec_over_lvr(weights, schedules, gamma, max_batch)
    rows: List[Dict[str, Any]] = []
    for value, score in zip(grid, scores):
        evaluator.set_lvr(float(value))
        rows.append({
            "lvr": float(value),
            "leverage_multiple": float(leverage_multiple(value)),
            "cec": float(score),
            "negative_equity_share": float(
                evaluator.negative_equity_frequency()[0]),
        })
    frame = pd.DataFrame.from_records(rows)
    frame["gain_vs_unlevered_pct"] = (
        frame["cec"] / float(frame.loc[frame["lvr"].idxmin(), "cec"]) - 1.0
    ) * 100.0
    pick = int(np.argmax(scores))
    return float(grid[pick]), float(scores[pick]), frame


def optimise_lvr_schedule(evaluator: MortgageEvaluator, weights: np.ndarray,
                          gamma: float,
                          grid: Sequence[float] = DEFAULT_LVR_GRID,
                          start: np.ndarray | None = None,
                          sweeps: int = 3,
                          min_improvement: float = 1e-5,
                          max_batch: int = 12,
                          label: str = ""
                          ) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """Coordinate ascent on the loan-to-value ratio, one age at a time.

    Under common random numbers the objective is deterministic in the
    schedule, so sweeping one age at a time is exact for that age and every
    sweep is monotone. Seeded at the best constant ratio, so the search cannot
    report a schedule worse than the flat one it generalises.
    """
    horizon = evaluator.spec.horizon
    if start is None:
        flat, _, _ = best_constant_lvr(evaluator, weights, gamma, grid,
                                       max_batch)
        schedule = np.full(horizon, flat)
    else:
        schedule = np.asarray(start, dtype=float).copy()

    best = float(evaluator.cec_over_lvr(weights, schedule[None, :], gamma,
                                        max_batch)[0])
    trace: List[Dict[str, Any]] = [
        {"sweep": -1, "cec": best, "evaluations": 1,
         "mean_lvr": float(schedule.mean())}]
    flat_start = bool(np.allclose(schedule, schedule[0]))
    LOGGER.info("%sLVR start: CEC=%.6f at %s",
                label, best,
                f"a flat {100.0 * schedule[0]:.0f}%" if flat_start
                else f"mean {100.0 * schedule.mean():.0f}%")

    evaluations = 1
    for sweep in range(int(sweeps)):
        opening = best
        for age in range(horizon):
            candidates = np.repeat(schedule[None, :], len(grid), axis=0)
            candidates[:, age] = list(grid)
            scores = evaluator.cec_over_lvr(weights, candidates, gamma,
                                            max_batch)
            evaluations += len(grid)
            pick = int(np.argmax(scores))
            if scores[pick] > best * (1.0 + min_improvement):
                best = float(scores[pick])
                schedule[age] = float(grid[pick])
        gain = (best / opening - 1.0) * 100.0
        trace.append({"sweep": sweep, "cec": best, "gain_pct": gain,
                      "evaluations": evaluations,
                      "mean_lvr": float(schedule.mean())})
        LOGGER.info("%sLVR sweep %d: CEC=%.6f (+%.4f%%), mean LVR %.0f%%",
                    label, sweep, best, gain, 100.0 * schedule.mean())
        if gain <= 1e-9:
            break
    evaluator.set_lvr(schedule)
    return schedule, best, pd.DataFrame.from_records(trace)


def alternate(evaluator: MortgageEvaluator, gamma: float,
              grid: Sequence[float] = DEFAULT_LVR_GRID,
              rounds: int = 3,
              coarse_step: float = 0.10,
              fine_step: float = 0.025,
              label: str = "") -> Dict[str, Any]:
    """Alternate between the allocation and the mortgage until neither moves.

    The two decisions are coupled -- how much house to own depends on how
    cheaply it can be financed, and vice versa -- so neither can be solved
    once and left. Each half is exact given the other under common random
    numbers, so the alternation is monotone and terminates.
    """
    weights, cec, _ = hs.constant_mix_optimum(
        evaluator, gamma, coarse_step, fine_step, label=f"{label}[alloc 0] ")
    schedule = np.zeros(evaluator.spec.horizon)
    history: List[Dict[str, Any]] = [
        {"round": 0, "stage": "allocation", "cec": cec,
         "mean_lvr": 0.0, **_weight_row(evaluator, weights)}]

    for round_ in range(1, int(rounds) + 1):
        schedule, cec, _ = optimise_lvr_schedule(
            evaluator, _as_schedule(weights, evaluator.spec.horizon), gamma,
            grid, start=schedule, label=f"{label}[lvr {round_}] ")
        history.append({"round": round_, "stage": "mortgage", "cec": cec,
                        "mean_lvr": float(schedule.mean()),
                        **_weight_row(evaluator, weights)})

        evaluator.set_lvr(schedule)
        new_weights, cec, _ = hs.constant_mix_optimum(
            evaluator, gamma, coarse_step, fine_step,
            label=f"{label}[alloc {round_}] ")
        moved = float(np.abs(new_weights - weights).max())
        weights = new_weights
        history.append({"round": round_, "stage": "allocation", "cec": cec,
                        "mean_lvr": float(schedule.mean()),
                        **_weight_row(evaluator, weights)})
        if moved < 1e-9:
            LOGGER.info("%salternation converged after round %d", label, round_)
            break

    evaluator.set_lvr(schedule)
    return {"weights": weights, "lvr": schedule, "cec": cec,
            "history": pd.DataFrame.from_records(history)}


def _as_schedule(weights: np.ndarray, horizon: int) -> np.ndarray:
    """``(H, A)`` from a constant ``(A,)`` allocation."""
    return np.repeat(np.asarray(weights, dtype=float)[None, :], horizon,
                     axis=0)


def _weight_row(evaluator: MortgageEvaluator,
                weights: np.ndarray) -> Dict[str, float]:
    return {f"w_{a}": float(weights[i])
            for i, a in enumerate(evaluator.assets)}


def sweep_spread(paths: Any, spec: Any, income: np.ndarray,
                 cfg: Mapping[str, Any], gross: np.ndarray,
                 spreads: Sequence[float], gamma: float,
                 holding_cost: float = 0.0,
                 grid: Sequence[float] = DEFAULT_LVR_GRID,
                 rate_base: str = "bill",
                 coarse_step: float = 0.10,
                 fine_step: float = 0.025) -> pd.DataFrame:
    """The joint optimum at each price of mortgage credit.

    One row per spread over the domestic short rate. Everything else is held
    fixed across rows -- same paths, same income draws, same search -- so a
    difference between two rows is the price of the loan and nothing else.
    """
    net = hs.net_of_cost(gross, holding_cost)
    rows: List[Dict[str, Any]] = []
    for spread in spreads:
        evaluator = MortgageEvaluator(
            paths, spec, income, cfg, extra={hs.HOUSING: net},
            spread=float(spread), rate_base=rate_base)
        solved = alternate(evaluator, gamma, grid,
                           coarse_step=coarse_step, fine_step=fine_step,
                           label=f"[spread {spread:.1%}] ")
        weights, schedule = solved["weights"], solved["lvr"]
        housing_weight = float(weights[evaluator.housing_index])
        mean_lvr = float(schedule.mean())
        evaluator.set_lvr(schedule)
        row: Dict[str, Any] = {
            "spread": float(spread),
            "holding_cost": float(holding_cost),
            "cec": float(solved["cec"]),
            "mean_lvr": mean_lvr,
            "lvr_working": float(schedule[:spec.n_working].mean()),
            "lvr_retired": float(schedule[spec.n_working:].mean()),
            "housing_weight": housing_weight,
            # What the household actually owns: equity in housing, grossed up
            # by the loan. This is the number that sounds alarming and is the
            # honest description of an eighty-percent mortgage.
            "gross_housing_exposure": float(
                housing_weight * np.mean(1.0 / (1.0 - schedule))),
            "negative_equity_share": float(
                evaluator.negative_equity_frequency()[0]),
        }
        row.update(_weight_row(evaluator, weights))

        # The matched control: same everything, no mortgage.
        evaluator.set_lvr(0.0)
        unlevered_weights, unlevered_cec, _ = hs.constant_mix_optimum(
            evaluator, gamma, coarse_step, fine_step,
            label=f"[spread {spread:.1%}, no mortgage] ")
        row["cec_unlevered"] = float(unlevered_cec)
        row["gain_vs_unlevered_pct"] = (
            float(solved["cec"]) / float(unlevered_cec) - 1.0) * 100.0
        row["housing_weight_unlevered"] = float(
            unlevered_weights[evaluator.housing_index])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def break_even_spread(frame: pd.DataFrame,
                      column: str = "mean_lvr",
                      spread_column: str = "spread",
                      threshold: float = 0.01) -> float:
    """The mortgage spread at which the optimal loan-to-value reaches zero.

    Linear interpolation between the bracketing rows; NaN when the sweep never
    crosses, because extrapolating would invent the number it failed to find.
    """
    return hs.break_even_cost(frame, weight_column=column,
                              cost_column=spread_column, threshold=threshold)


def schedule_frame(schedule: np.ndarray, weights: np.ndarray,
                   spec: Any, assets: Sequence[str]) -> pd.DataFrame:
    """Age-by-age description of a solved mortgage, for plotting and tables."""
    housing_index = list(assets).index(hs.HOUSING)
    housing_weight = float(weights[housing_index])
    rows: List[Dict[str, Any]] = []
    for h in range(spec.horizon):
        lvr = float(schedule[h])
        rows.append({
            "age": int(spec.ages[h]),
            "phase": "working" if h < spec.n_working else "retired",
            "lvr": lvr,
            "leverage_multiple": float(leverage_multiple(lvr)),
            "housing_equity_weight": housing_weight,
            "gross_housing_exposure": housing_weight / (1.0 - lvr),
        })
    return pd.DataFrame.from_records(rows)


def terminal_option_check(schedule: np.ndarray, spec: Any,
                          tail_years: int = 5) -> Dict[str, Any]:
    """Is the solved mortgage exploiting the right to die owing money?

    Limited liability makes borrowing at the very end of life close to a free
    option: the loan is floored at total loss of equity, the borrower does not
    live to repay it, and the bequest term cannot go below its own floor. A
    solver rewarded for that will lever up in the last few years for reasons
    that have nothing to do with housing.

    This measures the tell rather than assuming it away -- a final-years LVR
    well above the working-life average is the signature -- so the reader can
    discount the tail of the schedule accordingly.
    """
    schedule = np.asarray(schedule, dtype=float)
    tail = schedule[-int(tail_years):]
    body = schedule[:-int(tail_years)]
    retired = schedule[spec.n_working:]
    lift = float(tail.mean() - body.mean())
    return {
        "tail_years": int(tail_years),
        "lvr_final_years": float(tail.mean()),
        "lvr_before_that": float(body.mean()),
        "lvr_retired": float(retired.mean()) if retired.size else float("nan"),
        "terminal_lift": lift,
        # A lift of more than a fifth of the grid's range is not a preference
        # for housing; it is the option.
        "looks_like_the_option": bool(lift > 0.15),
    }


def lvr_deviation_profile(evaluator: MortgageEvaluator, weights: np.ndarray,
                          schedule: np.ndarray, gamma: float, spec: Any,
                          reference: float | None = None,
                          max_batch: int = 12) -> pd.DataFrame:
    """What each age's loan-to-value is actually worth, in basis points.

    The same discipline `src.allocation` applies to a solved weight schedule.
    A coordinate search over a coarse grid produces a jagged line, and most of
    the jaggedness sits on a flat part of the surface: moving one age's ratio
    costs nothing, so the search moves it for free and the plotted schedule
    looks far more structured than the evidence supports.

    This holds the solved schedule fixed, resets one age to the schedule's own
    average, and reports the certainty equivalent given up. Ages whose cost is
    near zero carry no information and should not be read as advice.
    """
    schedule = np.asarray(schedule, dtype=float)
    flat = float(schedule.mean()) if reference is None else float(reference)
    base = float(evaluator.cec_over_lvr(weights, schedule[None, :], gamma,
                                        max_batch)[0])
    variants = np.repeat(schedule[None, :], spec.horizon, axis=0)
    for h in range(spec.horizon):
        variants[h, h] = flat
    forced = evaluator.cec_over_lvr(weights, variants, gamma, max_batch)
    evaluator.set_lvr(schedule)

    rows: List[Dict[str, Any]] = []
    for h in range(spec.horizon):
        rows.append({
            "age": int(spec.ages[h]),
            "phase": "working" if h < spec.n_working else "retired",
            "lvr": float(schedule[h]),
            "reference_lvr": flat,
            "cec_solved": base,
            "cec_if_reset_to_average": float(forced[h]),
            "cost_of_resetting_bp": (base / float(forced[h]) - 1.0) * 1e4,
        })
    return pd.DataFrame.from_records(rows)


def profile_summary(profile: pd.DataFrame,
                    material_bp: float = 5.0) -> Dict[str, Any]:
    """How much of a solved schedule is real, and where it lives."""
    if profile.empty:
        return {"ages": 0}
    cost = profile["cost_of_resetting_bp"]
    material = profile[cost >= float(material_bp)]
    working = material[material["phase"] == "working"]
    return {
        "ages": int(len(profile)),
        "material_ages": int(len(material)),
        "material_share": float(len(material) / len(profile)),
        "material_bp": float(material_bp),
        "max_cost_bp": float(cost.max()),
        "median_cost_bp": float(cost.median()),
        "material_in_working_life": int(len(working)),
        "first_material_age": int(material["age"].min()) if len(material) else -1,
        "last_material_age": int(material["age"].max()) if len(material) else -1,
    }
