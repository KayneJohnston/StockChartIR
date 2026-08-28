"""Endogenous retirement timing, and the retirement-date lottery.

Every other document in this project retires the investor on a fixed
birthday. Real people do not. They retire when their balance looks big
enough -- which means, mechanically, that they retire disproportionately
*after* good markets. That is worth testing rather than assuming, because a
balance that looks big after a bull run is also a balance bought at high
valuations, and the same market move that triggers the decision may be
lowering the returns that have to fund it.

This module makes the retirement date a **path-dependent decision** and asks
two questions.

1. **Does a wealth trigger beat a fixed date?** Retiring when wealth reaches a
   multiple of income is what people do and what "you need 25x expenses"
   advice recommends. It is not obvious it is better than a date.
2. **How much does the decade around retirement explain?** Conditioning
   outcomes on the market a person happens to retire into is the mechanism
   underneath every glide-path argument, and it can be measured directly.

Because retirement timing changes how many years the investor works, the
comparison here evaluates utility over the **whole lifetime**, not the
retirement window used elsewhere: a rule that retires people early buys them
leisure that a retirement-only window would not see it paying for.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import lifecycle as lc
from . import spending as spg
from . import utility as ut
from .bootstrap import BootstrapPaths


# ---------------------------------------------------------------------------
# Retirement rules
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RetirementState:
    """What a retirement rule may condition on, at the start of one year."""

    age: int
    wealth: np.ndarray              # (N,) financial wealth
    current_income: np.ndarray      # (N,) this year's real labour income
    still_working: np.ndarray       # (N,) bool


class RetirementRule(abc.ABC):
    """A policy for deciding when to stop working."""

    key: str = "rule"
    label: str = "Rule"

    @abc.abstractmethod
    def should_retire(self, state: RetirementState) -> np.ndarray:
        """Boolean mask of paths retiring at the start of this year."""

    def describe(self) -> Dict[str, Any]:
        fields = {f.name: getattr(self, f.name)
                  for f in dataclasses.fields(self)} \
            if dataclasses.is_dataclass(self) else {}
        return {"key": self.key, "label": self.label, **fields}


@dataclasses.dataclass(frozen=True)
class FixedAgeRule(RetirementRule):
    """Retire on a birthday, whatever the portfolio has done."""

    age: int = 63
    key: str = dataclasses.field(default="fixed_age", init=False)
    label: str = dataclasses.field(default="Fixed age", init=False)

    def should_retire(self, state: RetirementState) -> np.ndarray:
        if state.age < self.age:
            return np.zeros_like(state.still_working)
        return state.still_working.copy()


@dataclasses.dataclass(frozen=True)
class WealthMultipleRule(RetirementRule):
    """Retire once wealth reaches a multiple of current annual income.

    Bounded to ``[min_age, max_age]``: nobody retires before the window
    opens, and everyone is retired by the time it closes whether or not the
    target was hit. A wide window is the "retire when you can afford it"
    policy; a narrow one around a target date is the flexible-by-a-few-years
    option that most people actually hold.
    """

    multiple: float = 25.0
    min_age: int = 55
    max_age: int = 70
    key: str = dataclasses.field(default="wealth_multiple", init=False)
    label: str = dataclasses.field(default="Wealth multiple", init=False)

    def should_retire(self, state: RetirementState) -> np.ndarray:
        if state.age < self.min_age:
            return np.zeros_like(state.still_working)
        if state.age >= self.max_age:
            return state.still_working.copy()
        target = self.multiple * np.maximum(state.current_income, 1e-12)
        return state.still_working & (state.wealth >= target)


REGISTRY: Mapping[str, Any] = {
    "fixed_age": FixedAgeRule,
    "wealth_multiple": WealthMultipleRule,
}


def build(key: str, **params: Any) -> RetirementRule:
    """Instantiate a retirement rule by name."""
    if key not in REGISTRY:
        raise ValueError(
            f"unknown retirement rule {key!r}; expected one of {sorted(REGISTRY)}")
    return REGISTRY[key](**params)


# ---------------------------------------------------------------------------
# Path-dependent simulator
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FlexibleOutcome:
    """Per-path results when the retirement date is a decision, not a date."""

    rule: str
    label: str
    consumption: np.ndarray            # (N, H)
    wealth: np.ndarray                 # (N, H + 1)
    portfolio_return: np.ndarray       # (N, H)
    retire_age: np.ndarray             # (N,)
    wealth_at_retirement: np.ndarray   # (N,)
    career_average_income: np.ndarray  # (N,)
    social_security: np.ndarray        # (N,)
    bequest: np.ndarray                # (N,)
    ruin: np.ndarray                   # (N,) bool
    years_worked: np.ndarray           # (N,)

    @property
    def n_paths(self) -> int:
        return int(self.consumption.shape[0])

    @property
    def years_retired(self) -> np.ndarray:
        return self.consumption.shape[1] - self.years_worked


def simulate_flexible(
    paths: BootstrapPaths,
    strategy: lc.Strategy,
    spec: lc.LifecycleSpec,
    income: np.ndarray,
    retirement: RetirementRule,
    spending: spg.SpendingRule | None = None,
) -> FlexibleOutcome:
    """Lifecycle simulation in which each path chooses its own retirement date.

    The structure mirrors :func:`src.lifecycle.simulate` but cannot reuse its
    two-phase loop, because at any calendar step some paths are working and
    others are drawing down. Everything that the fixed-date engine computes
    once -- wealth at retirement, career-average earnings, the social-security
    benefit, the opening withdrawal -- is instead recorded per path on the
    year that path retires.

    With :class:`FixedAgeRule` at ``spec.age_retire`` this reproduces the
    fixed-date engine exactly, which is asserted in
    ``tests/test_retirement.py``.
    """
    n_paths = paths.n_paths
    horizon = spec.horizon
    if income.shape[0] != n_paths:
        raise ValueError("income and paths disagree on n_paths")
    if income.shape[1] < horizon:
        raise ValueError(
            "flexible retirement needs labour income defined for every year "
            f"of the horizon ({horizon}); got {income.shape[1]}")

    rule = spending or spg.from_spec(spec.retirement_rule, spec.rule_rate)
    rp = lc.portfolio_returns(paths, strategy)
    inflation = paths.inflation[:, :horizon]

    wealth = np.zeros((n_paths, horizon + 1))
    consumption = np.zeros((n_paths, horizon))
    retired = np.zeros(n_paths, dtype=bool)
    retire_age = np.full(n_paths, spec.age_death, dtype=int)
    wealth_at_retirement = np.zeros(n_paths)
    career_average = np.zeros(n_paths)
    benefit = np.zeros(n_paths)
    initial_withdrawal = np.zeros(n_paths)
    prev_withdrawal = np.zeros(n_paths)
    income_sum = np.zeros(n_paths)
    years_worked = np.zeros(n_paths)
    ruined = np.zeros(n_paths, dtype=bool)
    last_return = np.zeros(n_paths)
    last_inflation = np.zeros(n_paths)

    for h in range(horizon):
        age = spec.age_start + h
        available = wealth[:, h]
        working = ~retired

        newly = retirement.should_retire(
            RetirementState(age=age, wealth=available,
                            current_income=income[:, h],
                            still_working=working))
        newly = newly & working
        if newly.any():
            idx = np.flatnonzero(newly)
            retired[idx] = True
            retire_age[idx] = age
            wealth_at_retirement[idx] = available[idx]
            worked = np.maximum(years_worked[idx], 1.0)
            career_average[idx] = income_sum[idx] / worked
            benefit[idx] = spec.social_security_benefit(career_average[idx])
            years_left = max(spec.age_death - age, 1)
            initial_withdrawal[idx] = rule.initial_withdrawal(
                available[idx], years_left, age)
            prev_withdrawal[idx] = initial_withdrawal[idx]

        still_working = ~retired
        contribution = np.where(still_working,
                                spec.savings_rate * income[:, h], 0.0)
        income_sum += np.where(still_working, income[:, h], 0.0)
        years_worked += still_working.astype(float)

        state = spg.SpendingState(
            year=h, age=age, years_remaining=max(spec.age_death - age, 1),
            wealth=available, prev_withdrawal=prev_withdrawal,
            initial_withdrawal=initial_withdrawal,
            wealth_at_retirement=wealth_at_retirement,
            last_return=last_return, last_inflation=last_inflation,
        )
        desired = np.where(retired, np.maximum(rule.desired(state), 0.0), 0.0)
        withdrawal = np.minimum(desired, np.maximum(available, 0.0))

        consumption[:, h] = np.where(
            still_working,
            (1.0 - spec.savings_rate) * income[:, h],
            benefit + withdrawal)
        wealth[:, h + 1] = np.maximum(
            available + contribution - withdrawal, 0.0) * (1.0 + rp[:, h])

        exhausted = retired & (wealth[:, h + 1] <= 0.0) & (h + 1 < horizon)
        ruined |= (~ruined) & exhausted
        prev_withdrawal = np.where(retired, withdrawal, prev_withdrawal)
        last_return = rp[:, h]
        last_inflation = inflation[:, h]

    return FlexibleOutcome(
        rule=retirement.key,
        label=retirement.label,
        consumption=consumption,
        wealth=wealth,
        portfolio_return=rp,
        retire_age=retire_age,
        wealth_at_retirement=wealth_at_retirement,
        career_average_income=career_average,
        social_security=benefit,
        bequest=wealth[:, horizon].copy(),
        ruin=ruined,
        years_worked=years_worked.astype(int),
    )


def extended_income(spec: lc.LifecycleSpec, n_paths: int,
                    shocks: Tuple[np.ndarray, np.ndarray] | None = None,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Labour income defined for *every* year of the horizon, not just to 63.

    A path that keeps working past the nominal retirement age still needs a
    wage, so the deterministic profile is extended to the full horizon. The
    first ``spec.n_working`` columns are identical to what
    :func:`src.lifecycle.simulate_income` produces for the fixed-date engine
    -- the profile is a function of the year index and the permanent shock is
    a cumulative sum, so a longer draw has the shorter one as its prefix. That
    is what keeps the fixed-age comparison exact.

    Note that the economy-wide average earnings used by the social-security
    bend points still come from the *original* spec: the schedule is
    calibrated to the economy, not to how long one investor chose to work.
    """
    # age_retire must stay strictly below age_death, so the working phase is
    # stretched to the whole horizon by pushing the death age out by a year.
    long_spec = dataclasses.replace(spec, age_retire=spec.age_death,
                                    age_death=spec.age_death + 1)
    return lc.simulate_income(long_spec, n_paths, rng=rng, shocks=shocks)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(outcome: FlexibleOutcome, cfg: Mapping[str, Any],
             spec: lc.LifecycleSpec, gammas: Sequence[float] | None = None,
             extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Summarise one retirement rule.

    Utility is evaluated over the **whole lifetime**, not the retirement
    window used elsewhere in this project. Retirement timing changes how many
    years the investor works, so working-life consumption is no longer
    identical across the policies being compared -- a rule that retires people
    early buys them leisure, and a retirement-only window would credit it with
    the cost and none of the benefit.
    """
    util = cfg["utility"]
    gammas = list(gammas if gammas is not None else util["risk_aversions"])
    bundle = ut.ConsumptionBundle(
        consumption=outcome.consumption,
        bequest=outcome.bequest,
        floor=float(util.get("consumption_floor", ut.DEFAULT_FLOOR)),
        bequest_shift=float(util.get("bequest_shift", 1.0)))
    row: Dict[str, Any] = dict(extra or {})
    row.update({"rule": outcome.rule, "label": outcome.label})
    for gamma in gammas:
        row[f"cec_gamma{float(gamma):g}"] = ut.crra_certainty_equivalent(
            bundle, float(gamma), float(util["discount_factor"]),
            float(util["bequest_weight"]), bool(util["bequest_enabled"]))
    retirement_consumption = np.array([
        outcome.consumption[n, outcome.years_worked[n]:].mean()
        for n in range(outcome.n_paths)])
    row.update({
        "median_retire_age": float(np.median(outcome.retire_age)),
        "mean_retire_age": float(outcome.retire_age.mean()),
        "p10_retire_age": float(np.percentile(outcome.retire_age, 10)),
        "p90_retire_age": float(np.percentile(outcome.retire_age, 90)),
        "sd_retire_age": float(outcome.retire_age.std(ddof=1)),
        "prob_ruin": float(outcome.ruin.mean()),
        "median_wealth_at_retirement": float(
            np.median(outcome.wealth_at_retirement)),
        "median_bequest": float(np.median(outcome.bequest)),
        "median_retirement_consumption": float(np.median(retirement_consumption)),
        "p5_retirement_consumption": float(
            np.percentile(retirement_consumption, 5)),
    })
    return row


def _window_return(outcome: FlexibleOutcome, spec: lc.LifecycleSpec,
                   before: int, after: int) -> np.ndarray:
    """Annualised real portfolio return over the window around retirement.

    The window follows each path's *own* retirement date, which is the point:
    under a wealth trigger the date is itself a function of the market, so a
    fixed calendar window would not capture the experience being measured.
    """
    horizon = outcome.consumption.shape[1]
    log_returns = np.log1p(np.clip(outcome.portfolio_return, -0.999999, None))
    out = np.empty(outcome.n_paths)
    for n in range(outcome.n_paths):
        centre = int(outcome.years_worked[n])
        lo = max(centre - before, 0)
        hi = min(centre + after, horizon)
        out[n] = np.expm1(log_returns[n, lo:hi].mean()) if hi > lo else np.nan
    return out


def retirement_lottery(outcome: FlexibleOutcome, spec: lc.LifecycleSpec,
                       before: int = 5, after: int = 5,
                       n_buckets: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """How much of the outcome does the decade around retirement explain?

    Returns a decile table and a small variance decomposition comparing the
    explanatory power of the retirement window against that of the whole
    lifetime. If a single decade explains most of what a 68-year lifetime
    delivers, that is the sequence-of-returns risk every glide-path argument
    appeals to, measured rather than asserted.
    """
    horizon = outcome.consumption.shape[1]
    window = _window_return(outcome, spec, before, after)
    lifetime = np.expm1(np.log1p(
        np.clip(outcome.portfolio_return, -0.999999, None)).mean(axis=1))
    consumption = np.array([
        outcome.consumption[n, outcome.years_worked[n]:].mean()
        for n in range(outcome.n_paths)])

    ok = np.isfinite(window) & np.isfinite(consumption) & (consumption > 0)
    target = np.log(consumption[ok])

    def r_squared(x: np.ndarray) -> float:
        design = np.column_stack([np.ones(x.size), x])
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ coef
        total = ((target - target.mean()) ** 2).sum()
        return float(1.0 - (resid ** 2).sum() / total) if total > 0 else np.nan

    stats = {
        "r2_retirement_window": r_squared(window[ok]),
        "r2_whole_lifetime": r_squared(lifetime[ok]),
        "window_years": float(before + after),
        "corr_window_lifetime": float(np.corrcoef(window[ok], lifetime[ok])[0, 1]),
    }
    stats["share_of_lifetime_r2"] = (
        stats["r2_retirement_window"] / stats["r2_whole_lifetime"]
        if stats["r2_whole_lifetime"] else np.nan)

    edges = np.percentile(window[ok], np.linspace(0, 100, n_buckets + 1))
    bucket = np.clip(np.digitize(window[ok], edges[1:-1]), 0, n_buckets - 1)
    rows = []
    for b in range(n_buckets):
        mask = bucket == b
        if not mask.any():
            continue
        rows.append({
            "decile": b + 1,
            "mean_window_return": float(window[ok][mask].mean()),
            "median_retirement_consumption": float(
                np.median(consumption[ok][mask])),
            "prob_ruin": float(outcome.ruin[ok][mask].mean()),
            "median_retire_age": float(np.median(outcome.retire_age[ok][mask])),
            "median_wealth_at_retirement": float(
                np.median(outcome.wealth_at_retirement[ok][mask])),
        })
    return pd.DataFrame.from_records(rows), stats


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation that returns NaN cleanly when a series has no variance.

    A fixed-age rule gives every path the same retirement age, so several of
    the statistics below are undefined by construction. Letting numpy divide
    by a zero standard deviation would produce the same NaN plus a warning;
    this makes the degenerate case explicit.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def bull_market_test(outcome: FlexibleOutcome, spec: lc.LifecycleSpec,
                     before: int = 5) -> Dict[str, float]:
    """Do people who retire after a bull run retire into worse returns?

    The folk observation is that retirements cluster after good markets. If
    good markets also mean richer valuations and thinner forward returns, a
    wealth trigger would be systematically retiring people at the wrong
    moment. This measures both halves directly: the correlation between the
    run-up to each path's retirement and the returns that follow it, and
    whether early retirees saw better run-ups than late ones.
    """
    horizon = outcome.consumption.shape[1]
    log_returns = np.log1p(np.clip(outcome.portfolio_return, -0.999999, None))
    run_up = np.full(outcome.n_paths, np.nan)
    after = np.full(outcome.n_paths, np.nan)
    for n in range(outcome.n_paths):
        centre = int(outcome.years_worked[n])
        lo = max(centre - before, 0)
        if centre > lo:
            run_up[n] = np.expm1(log_returns[n, lo:centre].mean())
        if centre < horizon:
            after[n] = np.expm1(log_returns[n, centre:].mean())
    early = outcome.retire_age <= np.median(outcome.retire_age)

    def group_mean(values: np.ndarray, mask: np.ndarray) -> float:
        selected = values[mask]
        selected = selected[np.isfinite(selected)]
        return float(selected.mean()) if selected.size else float("nan")

    return {
        "corr_runup_vs_subsequent_return": _safe_corr(run_up, after),
        "mean_runup_early_retirees": group_mean(run_up, early),
        "mean_runup_late_retirees": group_mean(run_up, ~early),
        "mean_subsequent_return_early": group_mean(after, early),
        "mean_subsequent_return_late": group_mean(after, ~early),
        "corr_retire_age_vs_runup": _safe_corr(
            outcome.retire_age.astype(float), run_up),
    }


def value_of_conditioning(summary: pd.DataFrame, metric: str,
                          fixed_prefix: str = "Fixed age") -> pd.DataFrame:
    """Isolate the value of *conditioning on the market* from the value of
    simply retiring earlier.

    This model prices consumption and nothing else: there is no disutility of
    work and no utility of leisure. Retiring early therefore costs only the
    consumption forgone, and since retirement consumption is close to
    working-life consumption at these ages, earlier is mechanically better.
    Comparing a wealth trigger against a fixed age 63 mixes that artefact into
    the answer.

    The fix is to compare each flexible rule against a fixed date **matched on
    its own mean retirement age**, interpolated along the fixed-age frontier.
    What survives is the part attributable to conditioning the date on the
    portfolio rather than to shifting it.
    """
    fixed = summary[summary["variant"].str.startswith(fixed_prefix)] \
        .sort_values("mean_retire_age")
    flexible = summary[~summary["variant"].str.startswith(fixed_prefix)]
    if len(fixed) < 2 or flexible.empty:
        return pd.DataFrame()

    ages = fixed["mean_retire_age"].to_numpy(dtype=float)
    values = fixed[metric].to_numpy(dtype=float)
    rows = []
    for _, row in flexible.iterrows():
        matched = float(np.interp(float(row["mean_retire_age"]), ages, values))
        rows.append({
            "variant": row["variant"],
            "mean_retire_age": float(row["mean_retire_age"]),
            "cec": float(row[metric]),
            "matched_fixed_date_cec": matched,
            "value_of_conditioning_pct": (float(row[metric]) / matched - 1.0)
            * 100.0,
            "extrapolated": not (ages.min() <= float(row["mean_retire_age"])
                                 <= ages.max()),
        })
    return pd.DataFrame.from_records(rows)
