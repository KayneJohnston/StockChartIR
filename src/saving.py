"""Conditioning the savings rate on age and on portfolio state.

`docs/09` found that conditioning the *retirement date* on the portfolio is
worth about 3% of certainty equivalent consumption. This is the accumulation
side of the same question: should the savings rate vary, and on what?

There are two quite different reasons it might, and a comparison that does not
separate them will credit the wrong one.

**Shape.** The savings rate could vary with age alone. Labour income here is
hump-shaped, peaking around 50, and a *fixed* rate makes consumption track
income exactly -- so a 25-year-old consumes least precisely when they are
poorest. A CRRA investor dislikes that, and an age profile that saves less
early and more in peak-earning years smooths it away. This is pure
consumption smoothing; it uses no market information at all.

**Conditioning.** The savings rate could also respond to *state*: whether
wealth is ahead of or behind an age-appropriate target, or what markets have
just done. This is what "you should have six times salary by fifty" advice is
reaching for.

The two are measured separately: the best deterministic age profile is solved
first, and conditioning rules are then scored against *that*, not against a
flat rate. Everything is also reported at a matched average lifetime savings
rate, which separates saving *smarter* from simply saving *more*.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import lifecycle as lc


@dataclasses.dataclass(frozen=True)
class SavingState:
    """What a savings rule may condition on, at the start of one working year."""

    age: int
    year: int                       # 0-based index into the horizon
    wealth: np.ndarray              # (N,) financial wealth
    current_income: np.ndarray      # (N,) this year's real labour income
    last_return: np.ndarray         # (N,) portfolio return of the year just gone
    still_working: np.ndarray       # (N,) bool

    @property
    def wealth_to_income(self) -> np.ndarray:
        return self.wealth / np.maximum(self.current_income, 1e-12)


class SavingRule(abc.ABC):
    """A policy for how much of labour income to save."""

    key: str = "rule"
    label: str = "Rule"

    @abc.abstractmethod
    def rate(self, state: SavingState) -> np.ndarray:
        """Savings rate for each path this year, as a share of labour income."""

    def describe(self) -> Dict[str, Any]:
        fields = {f.name: getattr(self, f.name)
                  for f in dataclasses.fields(self)} \
            if dataclasses.is_dataclass(self) else {}
        return {"key": self.key, "label": self.label, **fields}


@dataclasses.dataclass(frozen=True)
class ConstantRateRule(SavingRule):
    """Save the same share of income every year -- the project's baseline."""

    rate_value: float = 0.10
    key: str = dataclasses.field(default="constant", init=False)
    label: str = dataclasses.field(default="Constant rate", init=False)

    def rate(self, state: SavingState) -> np.ndarray:
        return np.full(state.wealth.shape, self.rate_value)


@dataclasses.dataclass(frozen=True)
class AgeProfileRule(SavingRule):
    """A deterministic savings rate for each age -- shape without conditioning.

    ``schedule`` is indexed by year of the horizon, so entries beyond the
    retirement age are simply never consulted.
    """

    schedule: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(68, 0.10))
    key: str = dataclasses.field(default="age_profile", init=False)
    label: str = dataclasses.field(default="Age profile", init=False)

    def rate(self, state: SavingState) -> np.ndarray:
        index = min(state.year, self.schedule.size - 1)
        return np.full(state.wealth.shape, float(self.schedule[index]))


@dataclasses.dataclass(frozen=True)
class OnTrackRule(SavingRule):
    """Save more when wealth is behind an age-appropriate target, less when ahead.

    ``s_h = clip(base_h + sensitivity * (target_h - W_h / Y_h), floor, cap)``

    ``target`` is a wealth-to-income multiple for each year of the horizon --
    the "you should have six times salary by fifty" heuristic, made explicit
    and made responsive. ``base`` may itself be an age profile, which is what
    lets conditioning be measured *on top of* the best deterministic shape
    rather than instead of it.
    """

    target: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(68))
    base: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(68, 0.10))
    sensitivity: float = 0.01
    floor: float = 0.0
    cap: float = 0.40
    key: str = dataclasses.field(default="on_track", init=False)
    label: str = dataclasses.field(default="On-track", init=False)

    def rate(self, state: SavingState) -> np.ndarray:
        index = min(state.year, self.base.size - 1)
        gap = float(self.target[min(state.year, self.target.size - 1)]) \
            - state.wealth_to_income
        return np.clip(float(self.base[index]) + self.sensitivity * gap,
                       self.floor, self.cap)


@dataclasses.dataclass(frozen=True)
class ReturnResponsiveRule(SavingRule):
    """Save more after a bad year, less after a good one.

    ``s_h = clip(base_h - sensitivity * last_return, floor, cap)``

    A crude counter-cyclical saver. It uses market information without any
    reference to whether the investor is actually on track, which makes it the
    natural foil for :class:`OnTrackRule`: if wealth-conditioning beats
    return-conditioning, the useful signal is the investor's own position
    rather than the market's direction.
    """

    base: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(68, 0.10))
    sensitivity: float = 0.20
    floor: float = 0.0
    cap: float = 0.40
    key: str = dataclasses.field(default="return_responsive", init=False)
    label: str = dataclasses.field(default="Return-responsive", init=False)

    def rate(self, state: SavingState) -> np.ndarray:
        index = min(state.year, self.base.size - 1)
        return np.clip(float(self.base[index])
                       - self.sensitivity * state.last_return,
                       self.floor, self.cap)


REGISTRY: Mapping[str, Any] = {
    "constant": ConstantRateRule,
    "age_profile": AgeProfileRule,
    "on_track": OnTrackRule,
    "return_responsive": ReturnResponsiveRule,
}


def build(key: str, **params: Any) -> SavingRule:
    """Instantiate a savings rule by name."""
    if key not in REGISTRY:
        raise ValueError(
            f"unknown saving rule {key!r}; expected one of {sorted(REGISTRY)}")
    return REGISTRY[key](**params)


# ---------------------------------------------------------------------------
# Solving the age profile
# ---------------------------------------------------------------------------
def optimise_age_profile(
    evaluate_fn: Any,
    n_working: int,
    horizon: int,
    grid: Sequence[float],
    start: float = 0.10,
    n_sweeps: int = 2,
    min_improvement: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Coordinate ascent over the savings rate at each working year.

    ``evaluate_fn`` takes a full-horizon schedule array and returns the
    objective. As in :mod:`src.glidepath`, common random numbers make the
    objective a deterministic function of the schedule, so a grid search over
    one year at a time is exact for that year and each sweep is monotone.

    ``min_improvement`` is a relative threshold a move must clear. Late
    working years sit on a nearly flat part of the surface, and without it the
    search reports year-to-year jitter that is worth a fraction of a basis
    point but reads as structure in a plotted profile.
    """
    grid = np.asarray(grid, dtype=float)
    schedule = np.full(horizon, float(
        grid[np.argmin(np.abs(grid - float(start)))]))
    best = float(evaluate_fn(schedule))
    for _ in range(n_sweeps):
        opening = best
        for h in range(n_working):
            incumbent = schedule[h]
            for candidate in grid:
                if candidate == incumbent:
                    continue
                schedule[h] = candidate
                score = float(evaluate_fn(schedule))
                if score > best * (1.0 + min_improvement):
                    best = score
                    incumbent = candidate
                schedule[h] = incumbent
        if best <= opening * (1.0 + 1e-9):
            break
    return schedule, best


def wealth_to_income_target(outcome: Any, income: np.ndarray,
                            horizon: int) -> np.ndarray:
    """Median wealth-to-income by age, for use as an on-track target.

    Taking the target from the baseline run's own realised median keeps the
    rule self-consistent: "on track" means at the median of what this model
    actually produces, rather than importing an outside rule of thumb the
    panel has no view on.

    The denominator must be *income*, not consumption: :class:`OnTrackRule`
    compares against ``wealth / current_income``, and dividing by consumption
    instead would inflate the target by ``1 / (1 - s)`` and silently
    mis-calibrate the rule.
    """
    wealth = outcome.wealth[:, :horizon]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = wealth / np.maximum(income[:, :horizon], 1e-12)
    return np.nan_to_num(np.median(ratio, axis=0))


def deviation_profile(evaluate_fn: Any, schedule: np.ndarray, n_working: int,
                      ages: np.ndarray, reference: float | None = None
                      ) -> pd.DataFrame:
    """What each year's savings rate is actually worth.

    Holds the solved schedule fixed, resets one year to ``reference`` (the
    schedule's own average by default) and reports the certainty-equivalent
    cost in basis points. A solved profile can look highly structured when
    most of its deviations sit on a flat part of the surface; this separates
    the shape that matters from the search noise around it.
    """
    reference = float(schedule[:n_working].mean()) if reference is None \
        else float(reference)
    base = float(evaluate_fn(schedule))
    rows = []
    for h in range(n_working):
        trial = schedule.copy()
        trial[h] = reference
        forced = float(evaluate_fn(trial))
        rows.append({
            "age": int(ages[h]),
            "solved_savings_rate": float(schedule[h]),
            "cec_if_reset_to_average": forced,
            "cost_of_resetting_bp": (base / forced - 1.0) * 1e4,
        })
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Separating "save smarter" from "save more"
# ---------------------------------------------------------------------------
def matched_rate_comparison(summary: pd.DataFrame, metric: str,
                            constant_prefix: str = "Constant") -> pd.DataFrame:
    """Score each rule against a constant rate matched on its own average.

    A rule that happens to save more will look better simply for having saved
    more. Interpolating the constant-rate frontier at each rule's realised
    mean savings rate strips that out, leaving the part attributable to *when*
    and *on what* it saved.
    """
    constant = summary[summary["variant"].str.startswith(constant_prefix)] \
        .sort_values("mean_savings_rate")
    flexible = summary[~summary["variant"].str.startswith(constant_prefix)]
    if len(constant) < 2 or flexible.empty:
        return pd.DataFrame()

    rates = constant["mean_savings_rate"].to_numpy(dtype=float)
    values = constant[metric].to_numpy(dtype=float)
    rows = []
    for _, row in flexible.iterrows():
        rate = float(row["mean_savings_rate"])
        matched = float(np.interp(rate, rates, values))
        rows.append({
            "variant": row["variant"],
            "mean_savings_rate": rate,
            "cec": float(row[metric]),
            "matched_constant_rate_cec": matched,
            "value_of_shape_pct": (float(row[metric]) / matched - 1.0) * 100.0,
            "extrapolated": not (rates.min() <= rate <= rates.max()),
        })
    return pd.DataFrame.from_records(rows)


def profile_frame(schedule: np.ndarray, spec: lc.LifecycleSpec,
                  label: str) -> pd.DataFrame:
    """Tidy age-by-age description of a savings schedule."""
    n = spec.n_working
    return pd.DataFrame({
        "variant": label,
        "age": spec.ages[:n],
        "savings_rate": schedule[:n],
    })


def normalise_to_mean(multipliers: np.ndarray, n_working: int,
                      target_mean: float, floor: float, cap: float,
                      n_iterations: int = 8) -> np.ndarray:
    """Turn free positive multipliers into a schedule with a fixed average rate.

    ``s_h = clip(target * v_h / mean(v), floor, cap)``, rescaled a few times
    so that clipping does not pull the realised mean away from the target.
    Pinning the average is what makes the *shape* question answerable
    separately from the *level* question: the level is set by the discount
    factor and risk aversion, which the return panel has no view on.
    """
    weights = np.maximum(np.asarray(multipliers, dtype=float), 1e-9)
    scale = target_mean / max(weights[:n_working].mean(), 1e-12)
    schedule = np.clip(weights * scale, floor, cap)
    for _ in range(n_iterations):
        realised = schedule[:n_working].mean()
        if abs(realised - target_mean) < 1e-9 or realised <= 0:
            break
        schedule = np.clip(schedule * (target_mean / realised), floor, cap)
    return schedule


def optimise_shape_at_fixed_mean(
    evaluate_fn: Any,
    n_working: int,
    horizon: int,
    target_mean: float,
    multiplier_grid: Sequence[float],
    floor: float = 0.0,
    cap: float = 0.40,
    n_sweeps: int = 2,
    min_improvement: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Solve the *shape* of the savings profile with its average pinned.

    Coordinate ascent over a per-age multiplier, renormalised after every move
    so the average savings rate never drifts. What comes back answers "given
    that you save this much over a career, when should you save it?" without
    the answer being swallowed by the model's view on how much to save at all.
    """
    grid = np.asarray(multiplier_grid, dtype=float)
    multipliers = np.ones(horizon)
    schedule = normalise_to_mean(multipliers, n_working, target_mean, floor, cap)
    best = float(evaluate_fn(schedule))
    for _ in range(n_sweeps):
        opening = best
        for h in range(n_working):
            incumbent = multipliers[h]
            for candidate in grid:
                if candidate == incumbent:
                    continue
                multipliers[h] = candidate
                trial = normalise_to_mean(multipliers, n_working, target_mean,
                                          floor, cap)
                score = float(evaluate_fn(trial))
                if score > best * (1.0 + min_improvement):
                    best = score
                    incumbent = candidate
                multipliers[h] = incumbent
        if best <= opening * (1.0 + 1e-9):
            break
    return normalise_to_mean(multipliers, n_working, target_mean, floor, cap), best
