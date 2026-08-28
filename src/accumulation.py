"""How hard the savings rate should lean on the balance already accumulated.

`docs/10` established two things and left a third open. It established that
the model cannot identify the *level* of the savings rate, so every comparison
must pin the average; and that at a matched average, conditioning the rate on
whether wealth is ahead of or behind an age-appropriate target is worth about
3% of certainty equivalent consumption, while conditioning on last year's
return is worth almost nothing.

What it did not establish is anything about *how* to lean on that balance. It
tested one functional form (a linear response to the gap in income multiples),
one target (the model's own median path), one symmetric coefficient, no limit
on how far the rate could move, and no view on which years of a career the
signal is actually informative in. Each of those is a real modelling choice,
and a 3% number that survives only one of them is not worth acting on.

This module takes the accumulation signal apart:

* **Functional form.** A gap measured in income multiples grows with age
  mechanically -- being "two times salary behind" means something very
  different at 30 and at 60. Scale-free alternatives (a funded *ratio*, or its
  log) are tested against it.
* **Asymmetry.** Saving more when behind and saving less when ahead are
  separate policies with separate coefficients, and there is no reason they
  should be equal. Which half carries the value is the question people
  actually face.
* **Which signal.** The funded ratio races the return-based signals, the
  investor's own investment gain, the raw balance, and the income shock -- and
  the two strongest are then run together, because two signals that each beat
  the baseline need not beat it jointly.
* **Feasibility.** A rule nobody can follow is worth nothing. The value is
  re-measured with the rate confined to progressively narrower bands around
  its average, and with a coarse guardrail version that moves in steps.
* **Where the value lands.** Whether this is a mean improvement or left-tail
  insurance, and at which ages and which preferences it pays.
* **What it interacts with.** Equity exposure and labour-income risk.

Everything is scored at a matched average savings rate, because a rule that
drifts to saving more will otherwise be credited for saving more. Two smaller
guards run throughout: :func:`at_grid_edge` flags an "optimum" that is really
the boundary of its own grid, and :func:`value_at_common_strength` compares
functional forms at the same response strength rather than at their own best
coefficients, which separates "this form is better" from "this form's grid let
it go further".
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import lifecycle as lc
from . import saving as sav

#: The funded ratio is clipped before any scale-free transform. Early in a
#: career wealth is a rounding error against the target, and an unclipped
#: log would hand a 26-year-old a signal two orders of magnitude larger than
#: anything a 60-year-old ever sees.
RATIO_CLIP: Tuple[float, float] = (0.05, 5.0)

#: A target multiple below this is treated as no target at all: you cannot be
#: behind on a balance you were never expected to have.
MIN_TARGET = 1e-3

GAP_FORMS: Tuple[str, ...] = ("level", "proportional", "log")

FORM_LABELS: Mapping[str, str] = {
    "level": "Level gap (target − W/Y, income multiples)",
    "proportional": "Funded ratio (1 − W/Y ÷ target)",
    "log": "Log funded ratio (log target ÷ W/Y)",
}


def funded_gap(wealth_to_income: np.ndarray, target: float,
               form: str = "proportional") -> np.ndarray:
    """How far behind the target this path is, positive when behind.

    The three forms differ in what "behind" is measured in, which is the whole
    point of comparing them: ``level`` is in income multiples and so grows with
    age, while ``proportional`` and ``log`` are scale-free and mean the same
    thing at every age.
    """
    if form not in GAP_FORMS:
        raise ValueError(f"unknown gap form {form!r}; expected one of {GAP_FORMS}")
    if target < MIN_TARGET:
        return np.zeros(np.shape(wealth_to_income))
    if form == "level":
        return target - wealth_to_income
    ratio = np.clip(wealth_to_income / target, *RATIO_CLIP)
    if form == "proportional":
        return 1.0 - ratio
    return -np.log(ratio)


@dataclasses.dataclass(frozen=True)
class FundedRatioRule(sav.SavingRule):
    """Lean on the balance, with separate coefficients for behind and ahead.

    ``s_h = clip(base_h + k * gap_h, floor, cap)`` where ``k`` is
    ``k_behind`` when the path is short of target and ``k_ahead`` when it is
    over. Setting them equal recovers a symmetric rule; setting ``k_ahead`` to
    zero gives a pure catch-up rule that never eases off.

    ``min_age``/``max_age`` confine the conditioning to part of the career,
    which is how the "when is this signal informative?" question is asked:
    outside the window the rule falls back to ``base`` exactly.
    """

    target: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(68))
    base: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(68, 0.10))
    k_behind: float = 0.0
    k_ahead: float | None = None
    form: str = "proportional"
    floor: float = 0.0
    cap: float = 0.40
    min_age: int | None = None
    max_age: int | None = None
    key: str = dataclasses.field(default="funded_ratio", init=False)
    label: str = dataclasses.field(default="Funded ratio", init=False)

    def rate(self, state: sav.SavingState) -> np.ndarray:
        index = min(state.year, self.base.size - 1)
        base = float(self.base[index])
        if (self.min_age is not None and state.age < self.min_age) or \
                (self.max_age is not None and state.age > self.max_age):
            return np.full(state.wealth.shape, base)
        target = float(self.target[min(state.year, self.target.size - 1)])
        gap = funded_gap(state.wealth_to_income, target, self.form)
        ahead = self.k_behind if self.k_ahead is None else float(self.k_ahead)
        response = np.where(gap > 0.0, self.k_behind * gap, ahead * gap)
        return np.clip(base + response, self.floor, self.cap)


@dataclasses.dataclass(frozen=True)
class BandedRule(sav.SavingRule):
    """Guardrails for saving: do nothing until the funded ratio leaves a band.

    The spending side of this project already found (``docs/06``) that coarse
    guardrail rules give up surprisingly little against continuous ones. This
    is the accumulation-side analogue, and it is the version a person could
    actually be told to follow: check once a year, and move your contribution
    by a fixed step only if you are more than ``band`` off target.
    """

    target: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(68))
    base: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(68, 0.10))
    band: float = 0.20
    step: float = 0.03
    floor: float = 0.0
    cap: float = 0.40
    key: str = dataclasses.field(default="banded", init=False)
    label: str = dataclasses.field(default="Guardrail", init=False)

    def rate(self, state: sav.SavingState) -> np.ndarray:
        index = min(state.year, self.base.size - 1)
        base = float(self.base[index])
        target = float(self.target[min(state.year, self.target.size - 1)])
        if target < MIN_TARGET:
            return np.full(state.wealth.shape, base)
        ratio = state.wealth_to_income / target
        move = np.where(ratio < 1.0 - self.band, self.step,
                        np.where(ratio > 1.0 + self.band, -self.step, 0.0))
        return np.clip(base + move, self.floor, self.cap)


@dataclasses.dataclass(frozen=True)
class SignalRule(sav.SavingRule):
    """Respond linearly to an arbitrary standardised signal.

    ``signal`` returns a per-path number that is positive when the investor
    should save *more*, so every entry in :data:`SIGNAL_LABELS` shares a sign
    convention and the sweeps over ``sensitivity`` are directly comparable.
    """

    base: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(68, 0.10))
    signal: Callable[[sav.SavingState], np.ndarray] | None = None
    signal_name: str = "none"
    sensitivity: float = 0.0
    floor: float = 0.0
    cap: float = 0.40
    key: str = dataclasses.field(default="signal", init=False)
    label: str = dataclasses.field(default="Signal", init=False)

    def rate(self, state: sav.SavingState) -> np.ndarray:
        index = min(state.year, self.base.size - 1)
        base = float(self.base[index])
        if self.signal is None or self.sensitivity == 0.0:
            return np.full(state.wealth.shape, base)
        return np.clip(base + self.sensitivity * self.signal(state),
                       self.floor, self.cap)


@dataclasses.dataclass(frozen=True)
class CombinedRule(sav.SavingRule):
    """Respond to two signals at once, each with its own coefficient.

    Two signals that each beat the baseline need not beat it together: if they
    are reading the same underlying state, layering them just doubles the
    response and the clipping does the rest. ``docs/10`` already found exactly
    that between the savings-side and retirement-side rules, so it cannot be
    assumed away here.
    """

    base: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(68, 0.10))
    first: Callable[[sav.SavingState], np.ndarray] | None = None
    second: Callable[[sav.SavingState], np.ndarray] | None = None
    k_first: float = 0.0
    k_second: float = 0.0
    floor: float = 0.0
    cap: float = 0.40
    key: str = dataclasses.field(default="combined", init=False)
    label: str = dataclasses.field(default="Combined", init=False)

    def rate(self, state: sav.SavingState) -> np.ndarray:
        index = min(state.year, self.base.size - 1)
        response = np.zeros(state.wealth.shape)
        if self.first is not None and self.k_first != 0.0:
            response = response + self.k_first * self.first(state)
        if self.second is not None and self.k_second != 0.0:
            response = response + self.k_second * self.second(state)
        return np.clip(float(self.base[index]) + response, self.floor, self.cap)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
#: A widely published wealth-to-salary ladder ("1x by 30, 3x by 40, ...").
#: Included because it is what an investor is actually likely to be told, and
#: the interesting question is how much of the model-implied target's value a
#: rule of thumb captures.
RULE_OF_THUMB_LADDER: Mapping[int, float] = {
    30: 1.0, 35: 2.0, 40: 3.0, 45: 4.0,
    50: 6.0, 55: 7.0, 60: 8.0, 67: 10.0,
}


def ladder_target(spec: lc.LifecycleSpec,
                  anchors: Mapping[int, float] | None = None) -> np.ndarray:
    """Interpolate a published wealth-to-salary ladder onto every horizon year."""
    anchors = dict(anchors or RULE_OF_THUMB_LADDER)
    ages = np.array(sorted(anchors), dtype=float)
    values = np.array([anchors[int(a)] for a in ages], dtype=float)
    horizon_ages = np.asarray(spec.ages, dtype=float)[:spec.horizon]
    return np.interp(horizon_ages, ages, values, left=0.0, right=values[-1])


def flat_target(spec: lc.LifecycleSpec, multiple: float) -> np.ndarray:
    """A single wealth-to-income multiple at every age -- a deliberate straw man.

    It has no age content at all, so comparing it against the other two
    separates "the target should rise with age" from "there should be a
    target".
    """
    return np.full(spec.horizon, float(multiple))


def scale_target(target: np.ndarray, factor: float) -> np.ndarray:
    """Aim higher or lower by a constant factor."""
    return np.asarray(target, dtype=float) * float(factor)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
#: Which kind of information each signal is. The distinction matters: a stock
#: signal tells the investor where they *are*, a flow signal tells them what
#: just *happened*, and the pay-cheque signal is about the other side of the
#: balance sheet entirely. The cumulative investment gain is a stock despite
#: being about the market -- it is a property of the balance, not of the year.
SIGNAL_FAMILY: Mapping[str, str] = {
    "funded_ratio": "stock (balance vs target)",
    "wealth_level": "stock (balance)",
    "investment_gain": "stock (market's share of the balance)",
    "return_1y": "flow (market)",
    "return_5y": "flow (market)",
    "return_10y": "flow (market)",
    "income_shock": "income",
    "none": "none",
}

SIGNAL_LABELS: Mapping[str, str] = {
    "funded_ratio": "Funded ratio (wealth vs age target)",
    "wealth_level": "Raw balance (wealth ÷ income)",
    "investment_gain": "Investment gain (balance vs contributions)",
    "return_1y": "Last year's return",
    "return_5y": "Trailing 5-year return",
    "return_10y": "Trailing 10-year return",
    "income_shock": "Income vs its expected path",
    "none": "No conditioning (age profile only)",
}


def make_signal(name: str, *, target: np.ndarray | None = None,
                income_profile: np.ndarray | None = None,
                form: str = "proportional"
                ) -> Callable[[sav.SavingState], np.ndarray]:
    """Build one of :data:`SIGNAL_LABELS`, positive meaning "save more".

    Every signal is bounded, either by construction or by an explicit clip, so
    that a single sensitivity grid is meaningful across all of them and no
    signal wins the race merely by having a larger variance.
    """
    if name not in SIGNAL_LABELS:
        raise ValueError(
            f"unknown signal {name!r}; expected one of {sorted(SIGNAL_LABELS)}")

    if name == "funded_ratio":
        if target is None:
            raise ValueError("the funded-ratio signal needs a target")
        tgt = np.asarray(target, dtype=float)

        def funded(state: sav.SavingState) -> np.ndarray:
            t = float(tgt[min(state.year, tgt.size - 1)])
            return funded_gap(state.wealth_to_income, t, form)
        return funded

    if name == "wealth_level":
        # Divided by ten so the coefficient is the same order of magnitude as
        # the scale-free signals; a raw multiple would need a tiny k.
        return lambda state: -np.clip(state.wealth_to_income / 10.0, 0.0, 5.0)

    if name == "investment_gain":
        return lambda state: -np.clip(state.investment_gain, -2.0, 5.0)

    if name in ("return_1y", "return_5y", "return_10y"):
        years = int(name.split("_")[1].rstrip("y"))
        if years == 1:
            return lambda state: -np.clip(state.last_return, -0.9, 2.0)
        return lambda state, n=years: -np.clip(state.trailing_return(n), -0.9, 2.0)

    if name == "income_shock":
        if income_profile is None:
            raise ValueError("the income signal needs an expected income path")
        profile = np.asarray(income_profile, dtype=float)

        def shock(state: sav.SavingState) -> np.ndarray:
            expected = float(profile[min(state.year, profile.size - 1)])
            if expected <= 1e-12:
                return np.zeros(state.wealth.shape)
            return np.clip(state.current_income / expected - 1.0, -1.0, 3.0)
        return shock

    return lambda state: np.zeros(state.wealth.shape)


# ---------------------------------------------------------------------------
# Matched-rate scoring
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MatchedScorer:
    """Score a rule against a constant rate matched on its own average.

    Conditioning rules do not hold their average savings rate fixed -- a rule
    that saves more when behind will, in a model where most paths end up
    behind, quietly save more overall. Interpolating the constant-rate
    frontier at each rule's realised mean strips that out, so what is left is
    attributable to *when* and *on what* the money was saved.
    """

    rates: np.ndarray
    values: np.ndarray

    @classmethod
    def from_frontier(cls, frontier: pd.DataFrame, metric: str,
                      rate_column: str = "savings_rate") -> "MatchedScorer":
        block = frontier.sort_values(rate_column)
        return cls(rates=block[rate_column].to_numpy(dtype=float),
                   values=block[metric].to_numpy(dtype=float))

    def matched(self, mean_rate: float) -> float:
        return float(np.interp(float(mean_rate), self.rates, self.values))

    def value_pct(self, cec: float, mean_rate: float) -> float:
        """Percentage gain over a constant rate that saves the same amount."""
        reference = self.matched(mean_rate)
        if not np.isfinite(reference) or reference <= 0.0:
            return float("nan")
        return (float(cec) / reference - 1.0) * 100.0

    def extrapolated(self, mean_rate: float) -> bool:
        return not (self.rates.min() <= float(mean_rate) <= self.rates.max())


def score_rule(run: Callable[[sav.SavingRule], Mapping[str, Any]],
               rule: sav.SavingRule, metric: str, scorer: MatchedScorer,
               **extra: Any) -> Dict[str, Any]:
    """Simulate one rule and package the numbers every sweep reports.

    Every sweep in this module goes through here, so each one carries the same
    matched-rate comparison, the same ruin probability and the same left-tail
    consumption number -- which is what makes the tables in ``docs/11``
    stackable against each other.
    """
    row = dict(run(rule))
    mean_rate = float(row["mean_savings_rate"])
    out: Dict[str, Any] = dict(extra)
    out.update({
        "cec": float(row[metric]),
        "mean_savings_rate": mean_rate,
        "matched_value_pct": scorer.value_pct(float(row[metric]), mean_rate),
        "extrapolated": scorer.extrapolated(mean_rate),
        "prob_ruin": float(row.get("prob_ruin", np.nan)),
        "p5_retirement_consumption": float(
            row.get("p5_retirement_consumption", np.nan)),
        "median_retirement_consumption": float(
            row.get("median_retirement_consumption", np.nan)),
    })
    return out


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def equivalent_rate_move(form: str, k: float, target_value: float,
                         shortfall: float = 0.25) -> float:
    """Percentage points the rate moves for a path ``shortfall`` short of target.

    The three functional forms measure the gap in different units, so their
    coefficients are not comparable and a plot with ``k`` on the x-axis would
    be meaningless. Translating each into "how much does this actually move
    the contribution for someone a quarter behind" puts all three on one axis,
    which is the only honest way to draw them together.
    """
    gap = funded_gap(np.array([(1.0 - float(shortfall)) * float(target_value)]),
                     float(target_value), form)
    return float(k * gap[0] * 100.0)


def sweep_response_forms(run: Callable[..., Mapping[str, Any]],
                         metric: str, scorer: MatchedScorer,
                         base: np.ndarray, target: np.ndarray,
                         grids: Mapping[str, Sequence[float]],
                         reference_year: int = 20, shortfall: float = 0.25,
                         floor: float = 0.0, cap: float = 0.40) -> pd.DataFrame:
    """Value of a symmetric response, one functional form at a time.

    ``reference_year`` picks the age at which coefficients are translated into
    a common "percentage points of income" scale; mid-career is the natural
    choice because that is where the level form's units are neither trivially
    small nor at their largest.
    """
    reference_target = float(
        np.asarray(target, dtype=float)[min(reference_year,
                                            np.size(target) - 1)])
    rows: List[Dict[str, Any]] = []
    for form, grid in grids.items():
        for k in grid:
            rule = FundedRatioRule(target=target, base=base, k_behind=float(k),
                                   form=form, floor=floor, cap=cap)
            rows.append(score_rule(run, rule, metric, scorer,
                                   form=form, form_label=FORM_LABELS[form],
                                   sensitivity=float(k),
                                   rate_move_pp=equivalent_rate_move(
                                   form, float(k), reference_target, shortfall)))
    return pd.DataFrame.from_records(rows)


def sweep_asymmetry(run: Callable[..., Mapping[str, Any]], metric: str,
                    scorer: MatchedScorer, base: np.ndarray,
                    target: np.ndarray, behind_grid: Sequence[float],
                    ahead_grid: Sequence[float], form: str = "proportional",
                    floor: float = 0.0, cap: float = 0.40) -> pd.DataFrame:
    """Separate coefficients for being behind and being ahead of target."""
    rows: List[Dict[str, Any]] = []
    for kb in behind_grid:
        for ka in ahead_grid:
            rule = FundedRatioRule(target=target, base=base,
                                   k_behind=float(kb), k_ahead=float(ka),
                                   form=form, floor=floor, cap=cap)
            rows.append(score_rule(run, rule, metric, scorer,
                                   k_behind=float(kb), k_ahead=float(ka),
                                   symmetric=bool(np.isclose(kb, ka))))
    return pd.DataFrame.from_records(rows)


def sweep_bands(run: Callable[..., Mapping[str, Any]], metric: str,
                scorer: MatchedScorer, base: np.ndarray, target: np.ndarray,
                band_grid: Sequence[float], step_grid: Sequence[float],
                floor: float = 0.0, cap: float = 0.40) -> pd.DataFrame:
    """Coarse guardrails: how much does a continuous response actually buy?"""
    rows: List[Dict[str, Any]] = []
    for band in band_grid:
        for step in step_grid:
            rule = BandedRule(target=target, base=base, band=float(band),
                              step=float(step), floor=floor, cap=cap)
            rows.append(score_rule(run, rule, metric, scorer,
                                   band=float(band), step=float(step)))
    return pd.DataFrame.from_records(rows)


def sweep_targets(run: Callable[..., Mapping[str, Any]], metric: str,
                  scorer: MatchedScorer, base: np.ndarray,
                  targets: Mapping[str, np.ndarray], factors: Sequence[float],
                  k: float, form: str = "proportional",
                  floor: float = 0.0, cap: float = 0.40) -> pd.DataFrame:
    """Does the target have to be right, and does aiming higher help?"""
    rows: List[Dict[str, Any]] = []
    for name, base_target in targets.items():
        for factor in factors:
            rule = FundedRatioRule(target=scale_target(base_target, factor),
                                   base=base, k_behind=float(k), form=form,
                                   floor=floor, cap=cap)
            rows.append(score_rule(run, rule, metric, scorer, target=name,
                                   factor=float(factor),
                                   median_target_multiple=float(
                                   np.median(scale_target(base_target, factor)))))
    return pd.DataFrame.from_records(rows)


def signal_race(run: Callable[..., Mapping[str, Any]], metric: str,
                scorer: MatchedScorer, base: np.ndarray,
                signals: Mapping[str, Callable[[sav.SavingState], np.ndarray]],
                grid: Sequence[float], floor: float = 0.0,
                cap: float = 0.40) -> pd.DataFrame:
    """Sweep the same sensitivity grid over every candidate signal."""
    rows: List[Dict[str, Any]] = []
    for name, fn in signals.items():
        for k in grid:
            rule = SignalRule(base=base, signal=fn, signal_name=name,
                              sensitivity=float(k), floor=floor, cap=cap)
            rows.append(score_rule(run, rule, metric, scorer, signal=name,
                                   signal_label=SIGNAL_LABELS.get(name, name),
                                   family=SIGNAL_FAMILY.get(name, "other"),
                                   sensitivity=float(k)))
    return pd.DataFrame.from_records(rows)


def best_by(frame: pd.DataFrame, group: str,
            metric: str = "matched_value_pct") -> pd.DataFrame:
    """Best row per group, sorted worst to best -- the horse-race summary."""
    if frame.empty:
        return frame
    idx = frame.groupby(group)[metric].idxmax()
    return frame.loc[idx].sort_values(metric).reset_index(drop=True)


def sweep_combination(run: Callable[..., Mapping[str, Any]], metric: str,
                      scorer: MatchedScorer, base: np.ndarray,
                      first: Callable[[sav.SavingState], np.ndarray],
                      second: Callable[[sav.SavingState], np.ndarray],
                      first_grid: Sequence[float],
                      second_grid: Sequence[float],
                      first_name: str = "first",
                      second_name: str = "second",
                      floor: float = 0.0, cap: float = 0.40) -> pd.DataFrame:
    """Both signals together, over the product of their sensitivity grids."""
    rows: List[Dict[str, Any]] = []
    for k1 in first_grid:
        for k2 in second_grid:
            rule = CombinedRule(base=base, first=first, second=second,
                                k_first=float(k1), k_second=float(k2),
                                floor=floor, cap=cap)
            rows.append(score_rule(run, rule, metric, scorer,
                                   k_first=float(k1), k_second=float(k2),
                                   first_signal=first_name,
                                   second_signal=second_name))
    return pd.DataFrame.from_records(rows)


def value_at_common_strength(frame: pd.DataFrame, group: str = "form",
                             x: str = "rate_move_pp",
                             y: str = "matched_value_pct") -> pd.DataFrame:
    """Compare groups at the same response strength, not at their own optima.

    Each functional form is swept over its own coefficient grid, and those
    grids do not reach the same distance. Ranking the forms by their best rows
    therefore confounds "this form is better" with "this form's grid let it go
    further". Interpolating every form's curve at the strongest response *all*
    of them can produce separates the two.
    """
    if frame.empty:
        return frame
    reach = frame.groupby(group)[x].max()
    common = float(reach.min())
    rows = []
    for name, block in frame.groupby(group):
        block = block.sort_values(x)
        rows.append({
            group: name,
            "common_strength": common,
            "value_at_common": float(np.interp(
                common, block[x].to_numpy(dtype=float),
                block[y].to_numpy(dtype=float))),
            "own_best": float(block[y].max()),
            "own_reach": float(block[x].max()),
        })
    return pd.DataFrame.from_records(rows).sort_values("value_at_common",
                                                       ascending=False)


def at_grid_edge(grid: Sequence[float], best: float,
                 tolerance: float = 1e-9) -> bool:
    """Is the chosen value the largest or smallest the sweep was offered?

    An optimum sitting on the boundary of its own grid is not an optimum, it
    is a truncation, and every claim made from one has to be hedged. This is
    cheap to check and easy to forget.
    """
    values = np.asarray(grid, dtype=float)
    if values.size == 0:
        return False
    return bool(abs(float(best) - values.min()) <= tolerance
                or abs(float(best) - values.max()) <= tolerance)


def partition_windows(frame: pd.DataFrame, min_col: str = "min_age",
                      max_col: str = "max_age") -> pd.DataFrame:
    """The non-overlapping windows that tile the career, in age order.

    The age-window sweep deliberately includes overlapping spans (halves as
    well as thirds) because each answers a different question. Comparing a
    25-year window against a 13-year one says more about length than about
    timing, so the prose is built from the disjoint tiling instead.
    """
    if frame.empty:
        return frame
    rows = frame.sort_values([min_col, max_col])
    picked: List[int] = []
    cursor = int(rows[min_col].min())
    while True:
        candidates = rows[rows[min_col] == cursor]
        if candidates.empty:
            break
        chosen = candidates[max_col].idxmin()
        if int(rows.loc[chosen, max_col]) < cursor:
            break               # malformed span; would not advance the cursor
        picked.append(chosen)
        cursor = int(rows.loc[chosen, max_col]) + 1
    return frame.loc[picked]


def increment_over(frame: pd.DataFrame, reference: float,
                   column: str = "matched_value_pct",
                   name: str = "increment_pct") -> pd.DataFrame:
    """Value net of what the deterministic age profile already earned.

    Conditioning is layered on a solved age profile that is itself worth
    something against a constant rate, so a raw matched-rate number credits the
    signal with the shape's value too. This subtracts it.
    """
    out = frame.copy()
    out[name] = out[column] - float(reference)
    return out


def feasibility_frontier(run: Callable[..., Mapping[str, Any]], metric: str,
                         scorer: MatchedScorer, base: np.ndarray,
                         target: np.ndarray, k: float,
                         widths: Sequence[float], target_mean: float,
                         form: str = "proportional") -> pd.DataFrame:
    """Re-price the rule with the savings rate confined to ±``width``.

    An unconstrained optimum that asks a 55-year-old to swing between 0% and
    40% of income is not advice. Narrowing the band is the cheapest possible
    test of whether the finding survives contact with a real household budget.
    """
    rows: List[Dict[str, Any]] = []
    for width in widths:
        lo = max(target_mean - float(width), 0.0)
        hi = target_mean + float(width)
        rule = FundedRatioRule(target=target, base=np.clip(base, lo, hi),
                               k_behind=float(k), form=form, floor=lo, cap=hi)
        rows.append(score_rule(run, rule, metric, scorer, width=float(width),
                               rate_floor=lo, rate_cap=hi))
    return pd.DataFrame.from_records(rows)


def age_window_value(run: Callable[..., Mapping[str, Any]], metric: str,
                     scorer: MatchedScorer, base: np.ndarray,
                     target: np.ndarray, k: float,
                     windows: Sequence[Tuple[int, int]],
                     form: str = "proportional", floor: float = 0.0,
                     cap: float = 0.40) -> pd.DataFrame:
    """Value when the rule is switched on only for part of the career."""
    rows: List[Dict[str, Any]] = []
    for lo, hi in windows:
        rule = FundedRatioRule(target=target, base=base, k_behind=float(k),
                               form=form, floor=floor, cap=cap,
                               min_age=int(lo), max_age=int(hi))
        rows.append(score_rule(run, rule, metric, scorer, min_age=int(lo),
                               max_age=int(hi), window=f"{int(lo)}-{int(hi)}"))
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Diagnostics on a simulated outcome
# ---------------------------------------------------------------------------
def retirement_consumption(outcome: Any) -> np.ndarray:
    """Average real consumption over each path's own retirement years."""
    return np.array([outcome.consumption[n, outcome.years_worked[n]:].mean()
                     for n in range(outcome.n_paths)])


def quantile_gain(baseline: Any, conditioned: Any,
                  quantiles: Sequence[float]) -> pd.DataFrame:
    """Where in the distribution the conditioning gain actually lands.

    A rule that raises the mean and a rule that lifts the bottom decile are
    very different products, and a single certainty equivalent does not
    distinguish them.
    """
    base = np.sort(retirement_consumption(baseline))
    cond = np.sort(retirement_consumption(conditioned))
    base_bequest = np.sort(baseline.bequest)
    cond_bequest = np.sort(conditioned.bequest)
    rows = []
    for q in quantiles:
        b = float(np.quantile(base, q))
        c = float(np.quantile(cond, q))
        rows.append({
            "quantile": float(q),
            "baseline_consumption": b,
            "conditioned_consumption": c,
            "gain_pct": (c / b - 1.0) * 100.0 if b > 0 else float("nan"),
            "baseline_bequest": float(np.quantile(base_bequest, q)),
            "conditioned_bequest": float(np.quantile(cond_bequest, q)),
        })
    return pd.DataFrame.from_records(rows)


def rate_fan(outcome: Any, spec: lc.LifecycleSpec,
             quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)
             ) -> pd.DataFrame:
    """Cross-sectional spread of the realised savings rate at each age.

    Only paths still working are counted, so the late-career rows are not
    dragged to zero by paths that have already retired.
    """
    rows = []
    rates = outcome.savings_rate_path
    for h in range(min(spec.n_working, rates.shape[1])):
        working = h < outcome.years_worked
        column = rates[working, h]
        if column.size == 0:
            continue
        row: Dict[str, Any] = {"age": int(spec.ages[h]),
                               "n_working": int(column.size)}
        for q in quantiles:
            row[f"q{int(round(q * 100)):02d}"] = float(np.quantile(column, q))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def activity_profile(baseline: Any, conditioned: Any,
                     spec: lc.LifecycleSpec) -> pd.DataFrame:
    """How far the rule moves the rate away from its base, age by age.

    Read alongside :func:`age_window_value`: a rule can be very *active* at an
    age where the activity is worth nothing, and the two together separate
    "the signal moves the rate" from "the signal moves the outcome".
    """
    rows = []
    base_rates = baseline.savings_rate_path
    cond_rates = conditioned.savings_rate_path
    for h in range(min(spec.n_working, cond_rates.shape[1])):
        working = h < conditioned.years_worked
        if not np.any(working):
            continue
        deviation = cond_rates[working, h] - base_rates[working, h]
        rows.append({
            "age": int(spec.ages[h]),
            "mean_abs_deviation": float(np.abs(deviation).mean()),
            "mean_deviation": float(deviation.mean()),
            "share_saving_more": float((deviation > 1e-9).mean()),
            "share_saving_less": float((deviation < -1e-9).mean()),
        })
    return pd.DataFrame.from_records(rows)


def policy_curve(rule: sav.SavingRule, year: int, target_value: float,
                 ratios: Sequence[float], income: float = 1.0,
                 age: int | None = None) -> np.ndarray:
    """The rate a rule prescribes as a function of the funded ratio.

    Evaluating the policy directly, rather than inferring it from a simulated
    cloud of points, is what makes the difference between the functional forms
    visible: they are all fitted on the same signal but bend differently.

    ``target_value`` must be the rule's own target at ``year`` -- the rule
    reads its schedule by year, so passing anything else would draw a curve
    the rule does not actually implement.
    """
    ratios = np.asarray(ratios, dtype=float)
    wealth = ratios * float(target_value) * float(income)
    state = sav.SavingState(
        age=int(year if age is None else age), year=int(year), wealth=wealth,
        current_income=np.full(ratios.shape, float(income)),
        last_return=np.zeros(ratios.shape),
        still_working=np.ones(ratios.shape, dtype=bool))
    return rule.rate(state)
