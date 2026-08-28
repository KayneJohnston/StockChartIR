"""Solving the whole allocation, not just the equity share.

``docs/07`` solves for the equity share at every age and for the domestic
split on five-year bands, but it holds the fixed-income sleeve at a fixed
70/30 bond/bill mix. That is a real restriction and it was made for cost
rather than for principle: an investor choosing between domestic equity,
international equity, long bonds and bills faces a three-dimensional decision
at every age, and fixing one of the dimensions in advance can only understate
what the unrestricted optimum is worth.

This module removes the restriction. The decision variable is the full weight
simplex at every year of the lifecycle -- 68 points in the 3-simplex, 204 free
parameters -- solved directly against certainty-equivalent consumption.

**The search.** Under common random numbers the objective is a deterministic
function of the schedule, so a search over one age at a time is exact for that
age and every sweep is monotone. Two stages are used because a lattice fine
enough to be precise is too large to sweep and a local search alone is too
easy to trap:

1. a *coarse lattice* sweep, evaluating every composition of the simplex at a
   step of :data:`COARSE_STEP` (35 candidates) at each age in turn; then
2. a *fine local* sweep, evaluating the twelve single-step pairwise exchanges
   around the incumbent, repeated until nothing improves.

Both stages accept a move only if it clears a relative improvement threshold,
for the reason given in :mod:`src.glidepath`: without one the search wanders
over flat parts of the surface and reports year-to-year jitter that reads as
structure in a plotted schedule. :func:`deviation_profile` then measures what
each age's allocation is actually worth, so the shape can be described from
evidence rather than from appearance.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import glidepath as gp
from . import lifecycle as lc

LOGGER = logging.getLogger(__name__)

N_ASSETS = len(lc.ASSETS)
ASSET_LABELS: Mapping[str, str] = {
    "dom_eq": "Domestic equity",
    "intl_eq": "International equity",
    "bond": "Bonds",
    "bill": "Bills",
}

COARSE_STEP = 0.25
FINE_STEP = 0.05

#: The twelve ordered asset pairs a single-step exchange can move weight along.
PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    (i, j) for i in range(N_ASSETS) for j in range(N_ASSETS) if i != j)


def simplex_lattice(step: float = COARSE_STEP,
                    n_assets: int = N_ASSETS) -> np.ndarray:
    """Every allocation whose weights are multiples of ``step`` and sum to one.

    Returns ``(K, n_assets)``. At ``step = 0.25`` this is the 35 compositions
    of four units into four assets, which is coarse enough to sweep at every
    age and fine enough to land in the right region of the simplex.
    """
    if not 0.0 < float(step) <= 1.0:
        raise ValueError(
            f"step must be a positive fraction of one; got {step!r}")
    units = int(round(1.0 / float(step)))
    rows = [np.array(c, dtype=float) / units
            for c in _compositions(units, n_assets)]
    return np.asarray(rows, dtype=float)


def _compositions(total: int, parts: int) -> Iterable[Tuple[int, ...]]:
    """Weak compositions of ``total`` into ``parts`` non-negative integers."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def exchange_neighbourhood(weights: np.ndarray, step: float = FINE_STEP
                           ) -> np.ndarray:
    """Single-step pairwise exchanges around one allocation, plus itself.

    Moving ``step`` from asset *i* to asset *j* keeps the weights on the
    simplex by construction, which is what makes this a valid local search
    there. Exchanges that would drive a weight negative are dropped rather
    than clipped, so every candidate returned is feasible.
    """
    weights = np.asarray(weights, dtype=float)
    out = [weights.copy()]
    for i, j in PAIRS:
        if weights[i] < step - 1e-12:
            continue
        move = weights.copy()
        move[i] -= step
        move[j] += step
        out.append(np.clip(move, 0.0, 1.0))
    return np.asarray(out, dtype=float)


def _schedule_variants(schedule: np.ndarray, age: int,
                       candidates: np.ndarray) -> np.ndarray:
    """``(K, H, 4)``: the schedule repeated with one age set to each candidate."""
    tiled = np.repeat(schedule[None, :, :], candidates.shape[0], axis=0)
    tiled[:, age, :] = candidates
    return tiled


def optimise_full_simplex(
    evaluator: gp.BatchEvaluator,
    gamma: float,
    start: Sequence[float] | None = None,
    coarse_step: float = COARSE_STEP,
    fine_step: float = FINE_STEP,
    coarse_sweeps: int = 2,
    fine_sweeps: int = 4,
    min_improvement: float = 1e-6,
    max_batch: int = 12,
    label: str = "",
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """Solve the ``(H, 4)`` weight schedule by two-stage coordinate ascent.

    Returns the schedule, its certainty equivalent, and a trace of the search
    so that convergence can be shown rather than asserted.
    """
    horizon = evaluator.spec.horizon
    lattice = simplex_lattice(coarse_step)
    opening = np.asarray(start if start is not None
                         else [0.25] * N_ASSETS, dtype=float)
    opening = opening / opening.sum()
    schedule = np.repeat(opening[None, :], horizon, axis=0)

    best = float(evaluator.cec(schedule[None], gamma)[0])
    trace: List[Dict[str, Any]] = [
        {"stage": "start", "sweep": -1, "cec": best,
         "evaluations": 1, **_mean_weights(schedule)}]
    LOGGER.info("%sfull-simplex start (gamma=%.1f): CEC=%.6f",
                label, gamma, best)

    evaluations = 1
    for stage, sweeps, candidates_for in (
            ("coarse", coarse_sweeps, lambda _w: lattice),
            ("fine", fine_sweeps,
             lambda w: exchange_neighbourhood(w, fine_step))):
        for sweep in range(sweeps):
            start_cec = best
            for age in range(horizon):
                candidates = candidates_for(schedule[age])
                if len(candidates) <= 1:
                    continue
                scores = evaluator.cec_chunked(
                    _schedule_variants(schedule, age, candidates), gamma,
                    max_batch=max_batch)
                evaluations += len(candidates)
                pick = int(np.argmax(scores))
                if scores[pick] > best * (1.0 + min_improvement):
                    best = float(scores[pick])
                    schedule[age] = candidates[pick]
            gain = (best / start_cec - 1.0) * 100.0
            trace.append({"stage": stage, "sweep": sweep, "cec": best,
                          "gain_pct": gain, "evaluations": evaluations,
                          **_mean_weights(schedule)})
            LOGGER.info("%s%s sweep %d (gamma=%.1f): CEC=%.6f (+%.4f%%)",
                        label, stage, sweep, gamma, best, gain)
            if gain <= 1e-9:
                break
    return schedule, best, pd.DataFrame.from_records(trace)


def _mean_weights(schedule: np.ndarray) -> Dict[str, float]:
    return {f"mean_{a}": float(schedule[:, i].mean())
            for i, a in enumerate(lc.ASSETS)}


def schedule_frame(schedule: np.ndarray, spec: lc.LifecycleSpec, gamma: float,
                   kind: str = "full simplex") -> pd.DataFrame:
    """Tidy age-by-asset description of a solved schedule."""
    rows = []
    for h in range(spec.horizon):
        row: Dict[str, Any] = {"kind": kind, "risk_aversion": float(gamma),
                               "age": int(spec.ages[h]),
                               "phase": "working" if h < spec.n_working
                               else "retired"}
        row.update({a: float(schedule[h, i]) for i, a in enumerate(lc.ASSETS)})
        row["equity"] = float(schedule[h, 0] + schedule[h, 1])
        row["fixed_income"] = float(schedule[h, 2] + schedule[h, 3])
        row["domestic_share_of_equity"] = float(
            schedule[h, 0] / max(row["equity"], 1e-12))
        row["bond_share_of_fixed"] = float(
            schedule[h, 2] / max(row["fixed_income"], 1e-12))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def phase_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Average solved weights by lifecycle phase and risk aversion."""
    rows = []
    for (gamma, phase), block in frame.groupby(["risk_aversion", "phase"]):
        row: Dict[str, Any] = {"risk_aversion": float(gamma), "phase": phase,
                               "years": int(len(block))}
        row.update({a: float(block[a].mean()) for a in lc.ASSETS})
        row["equity"] = float(block["equity"].mean())
        rows.append(row)
    order = {"working": 0, "retired": 1}
    return (pd.DataFrame.from_records(rows)
            .sort_values(["risk_aversion", "phase"],
                         key=lambda s: s.map(order) if s.name == "phase" else s)
            .reset_index(drop=True))


def deviation_profile(evaluator: gp.BatchEvaluator, schedule: np.ndarray,
                      gamma: float, spec: lc.LifecycleSpec,
                      reference: np.ndarray | None = None,
                      max_batch: int = 12) -> pd.DataFrame:
    """What each age's allocation is actually worth, in basis points.

    Holds the solved schedule fixed, resets one age to ``reference`` (the
    schedule's own average allocation by default) and reports the
    certainty-equivalent cost. A solved schedule can look highly structured
    while most of its structure sits on a flat part of the surface; this
    separates the shape that matters from the search noise around it.
    """
    reference = (schedule.mean(axis=0) if reference is None
                 else np.asarray(reference, dtype=float))
    reference = reference / reference.sum()
    base = float(evaluator.cec(schedule[None], gamma)[0])

    variants = np.repeat(schedule[None, :, :], spec.horizon, axis=0)
    for h in range(spec.horizon):
        variants[h, h, :] = reference
    forced = evaluator.cec_chunked(variants, gamma, max_batch=max_batch)

    rows = []
    for h in range(spec.horizon):
        row: Dict[str, Any] = {
            "risk_aversion": float(gamma), "age": int(spec.ages[h]),
            "phase": "working" if h < spec.n_working else "retired",
            "cec_solved": base, "cec_if_reset_to_average": float(forced[h]),
            "cost_of_resetting_bp": (base / float(forced[h]) - 1.0) * 1e4}
        row.update({a: float(schedule[h, i]) for i, a in enumerate(lc.ASSETS)})
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def compare_to_benchmarks(evaluator: gp.BatchEvaluator,
                          solved: Mapping[float, np.ndarray],
                          strategies: Mapping[str, lc.Strategy],
                          gammas: Sequence[float],
                          extra: Mapping[str, np.ndarray] | None = None
                          ) -> pd.DataFrame:
    """Score the solved schedules against the benchmark strategies."""
    rows: List[Dict[str, Any]] = []
    for gamma in gammas:
        candidates: Dict[str, np.ndarray] = {
            key: strat.weights for key, strat in strategies.items()}
        candidates.update({k: v for k, v in (extra or {}).items()})
        if float(gamma) in solved:
            candidates["full_simplex_optimal"] = solved[float(gamma)]
        names = list(candidates)
        tensor = np.stack([candidates[n] for n in names])
        scores = evaluator.cec_chunked(tensor, float(gamma))
        best = float(np.max(scores))
        for name, score in zip(names, scores):
            rows.append({"strategy": name, "risk_aversion": float(gamma),
                         "cec": float(score),
                         "gap_to_best_pct": (float(score) / best - 1.0) * 100.0})
    return pd.DataFrame.from_records(rows).sort_values(
        ["risk_aversion", "cec"], ascending=[True, False]).reset_index(drop=True)


def restart_check(evaluator: gp.BatchEvaluator, gamma: float,
                  starts: Sequence[Sequence[float]], **kwargs: Any
                  ) -> Tuple[pd.DataFrame, np.ndarray, float]:
    """Re-solve from several corners of the simplex to look for local optima."""
    rows: List[Dict[str, Any]] = []
    best_schedule: np.ndarray | None = None
    best_cec = -np.inf
    for start in starts:
        schedule, cec, _ = optimise_full_simplex(
            evaluator, gamma, start=start,
            label=f"restart {np.round(start, 2).tolist()} ", **kwargs)
        rows.append({"start": " / ".join(f"{v:.2f}" for v in start),
                     "solved_cec": cec, **_mean_weights(schedule)})
        if cec > best_cec:
            best_cec, best_schedule = cec, schedule
    frame = pd.DataFrame.from_records(rows)
    frame["gap_to_best_pct"] = (frame["solved_cec"] / best_cec - 1.0) * 100.0
    assert best_schedule is not None
    return frame, best_schedule, best_cec
