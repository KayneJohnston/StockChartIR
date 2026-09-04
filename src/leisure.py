"""What is a year of retirement worth, and when should it start?

Section #plan solves the withdrawal rule and the allocation together and finds
that a third decision -- the retirement date -- cannot be solved at all. Left
free, it runs to the oldest age on the grid, monotonically, because nothing in
this model charges the investor for the years they spend working. Another year
of employment is more contributions, a shorter drawdown and a larger social
security benefit, at no cost whatever. The optimiser takes every one it is
offered, and the answer is a statement about the model rather than about
retirement.

This section supplies the missing side of the ledger.

**The cost of working, in the units the rest of the paper uses.** Rather than
invent a disutility of labour in utils -- a quantity nothing here can
calibrate -- the cost is expressed as a *consumption equivalent*. Let ``L`` be
the number such that a retired year at consumption ``c`` is worth exactly as
much as a working year at ``L * c``. Then the felicity of a working year is
evaluated at ``c / L`` and the retired years are left alone:

    effective consumption = c / L   while working
                          = c       once retired

``L = 1`` charges nothing for working and reproduces every other result in
this paper exactly, which is the control. ``L = 1.2`` says a year of
retirement is worth twenty per cent more than the same money earned while
employed -- the commute, the constraint on when and where to be, the
unpurchasable hours.

Normalising on the *working* years rather than the retired ones is deliberate:
it leaves the retirement-window certainty equivalents on the same footing as
the rest of the paper, so a reader can carry numbers between sections.

**Why this form.** It is one parameter, it is interpretable without a
utility-theory glossary, and it maps onto something economists have tried to
measure. Consumption is observed to fall at retirement -- the "retirement
consumption puzzle" -- without a matching fall in reported wellbeing, and one
reading of that gap is precisely this: retirees substitute time for money,
producing at home what they used to buy. A fall of ``d`` with welfare held
constant implies ``L = 1 / (1 - d)``. That reading is contested, and this
project cannot test it, so it is used only to put the swept grid on a scale a
reader recognises.

**What is actually reported.** Not a recommended retirement age -- the
calibration would have to be believed for that -- but the *break-even*: for
each candidate date, the value of leisure at which it overtakes working
longer. The question becomes one the reader can answer for themselves. "To
justify stopping at sixty rather than seventy, you must value a year of your
own time at least this highly." That is derived here from end to end; only the
anchors beside it come from outside.

**And the horizon has to be the whole life.** Every other section scores
consumption from the retirement date. That window cannot price a retirement
date: a rule that retires people early buys them leisure, and a
retirement-only window would charge them the cost and credit them none of the
benefit. `docs/09` makes the same point about the wealth-trigger comparison.
So the aggregation here runs from twenty-five, and the numbers are not
comparable with the retirement-window certainty equivalents elsewhere.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Consumption-equivalent value of a retired year against a working one.
#: ``1.0`` charges nothing for working, which is what every other section
#: assumes. The grid runs well past anything the literature suggests, because
#: the point is to locate a break-even rather than to defend a calibration.
DEFAULT_LEISURE: Tuple[float, ...] = (
    1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.00)

#: Retirement dates scored. Wide on both sides on purpose: an optimum sitting
#: on either end of the grid would be the grid's answer rather than the
#: model's, and ``verdict`` says so when it happens.
DEFAULT_AGES: Tuple[int, ...] = (50, 52, 55, 58, 60, 62, 63, 65, 67, 70)

#: Labelled points on the grid, as arithmetic on a stated consumption drop
#: rather than as findings. The drop range is from the retirement-consumption
#: literature and cannot be checked against this project's data; it is here to
#: put the swept grid on a recognisable scale and nothing below depends on it.
ANCHORS: Dict[str, float] = {
    "no value on leisure (every other section)": 0.0,
    "a 10% consumption drop at retirement": 0.10,
    "a 20% consumption drop at retirement": 0.20,
    "a 30% consumption drop at retirement": 0.30,
}


def leisure_from_drop(drop: float) -> float:
    """``L`` implied by a consumption fall of ``drop`` at constant welfare.

    If spending falls by ``d`` at retirement and the retiree is no worse off,
    then ``u(c(1 - d))`` retired equals ``u(c)`` working, so a retired dollar
    does the work of ``1 / (1 - d)`` working ones.
    """
    drop = float(drop)
    if not 0.0 <= drop < 1.0:
        raise ValueError("a consumption drop must lie in [0, 1)")
    return 1.0 / (1.0 - drop)


def anchor_table(anchors: Mapping[str, float] = ANCHORS) -> pd.DataFrame:
    """Each labelled drop and the leisure value it implies."""
    rows = [{"label": label, "consumption_drop": float(drop),
             "leisure": leisure_from_drop(drop)}
            for label, drop in anchors.items()]
    frame = pd.DataFrame.from_records(rows)
    frame["leisure_pct"] = (frame["leisure"] - 1.0) * 100.0
    return frame


#: The pension systems compared, as the two things that actually differ.
#:
#: ``us``
#:     Earnings-related, payable from the day work stops, and adjusted for
#:     the age it starts at -- so the claiming date is worth nothing either
#:     way and only the leisure preference moves the answer.
#: ``au_pension_only``
#:     A flat means-tested pension payable at a fixed age however early
#:     somebody stopped, at the same voluntary saving rate. Isolates the
#:     eligibility gate.
#: ``au_as_legislated``
#:     The same gate with the Superannuation Guarantee on top, which is what
#:     an Australian actually faces. Separates the gate from the extra saving
#:     that pays for crossing it.
SYSTEMS: Tuple[str, ...] = ("us", "au_pension_only", "au_as_legislated")

#: How far apart two systems' per-year costs of retiring earlier may be and
#: still be reported as the same slope. Ten per cent of the cheaper one: wide
#: enough that a bootstrap wobble does not flip the sentence, narrow enough
#: that a system charging half again as much is not called "similar".
COST_SIMILAR_BAND: float = 0.10


def system_overrides(system: str, cfg: Mapping[str, Any],
                     ) -> Tuple[Dict[str, Any], bool]:
    """``(spec overrides, whether the claiming date is actuarially adjusted)``.

    Australia's Age Pension carries no actuarial adjustment: it is not a
    claiming choice at all, but an age you reach. Applying one would model a
    system nobody lives under, so the flag rides with the overrides rather
    than being set separately and forgotten.
    """
    from . import pension as pn

    block = cfg.get("leisure", {})
    if system == "us":
        return {}, True
    au = pn.from_config(cfg)
    au.pop("pension_free_area_non_homeowner", None)
    shared: Dict[str, Any] = {
        "social_security_formula": "means_tested",
        "benefit_start_age": int(block.get("age_pension_age", 67)),
        "pre_eligibility_benefit_share": float(
            block.get("pre_pension_safety_net", 0.0)),
        **au,
    }
    if system == "au_pension_only":
        return shared, False
    if system == "au_as_legislated":
        pension = cfg.get("pension", {})
        return {**shared,
                "super_guarantee_rate": float(
                    pension.get("sg_rate", pn.SG_RATE)),
                "super_contributions_tax": float(
                    pension.get("sg_contributions_tax",
                                pn.SG_CONTRIBUTIONS_TAX))}, False
    raise ValueError(f"unknown pension system {system!r}; expected one of "
                     f"{SYSTEMS}")


# ---------------------------------------------------------------------------
# Paying for the pension you claim early
# ---------------------------------------------------------------------------
def annuity_factor(spec: Any, survive: np.ndarray, beta: float,
                   retire_age: int) -> float:
    """Discounted expected years of benefit for someone retiring at that age.

    ``sum_h beta^h S(h)`` over the retirement years, where ``S`` is the
    model's own survival curve. It is the price of a real annuity starting on
    that birthday, in the model's own units.
    """
    start = int(retire_age) - int(spec.age_start)
    horizon = int(spec.horizon)
    if not 0 <= start < horizon:
        raise ValueError(
            f"retiring at {retire_age} is outside the simulated horizon")
    years = np.arange(start, horizon)
    return float((float(beta) ** years * np.asarray(survive)[start:horizon]).sum())


def fair_claim_factor(spec: Any, survive: np.ndarray, beta: float,
                      reference_age: int, ages: Sequence[int],
                      ) -> Dict[int, float]:
    """Benefit multipliers that make the claiming age worth nothing either way.

    Every other section fixes the retirement date, so the benefit starts on
    the same birthday for every strategy and its start date cancels. It does
    not cancel here. Left alone, this model pays whoever stops work at
    fifty-five a full unreduced pension for thirty-eight years, which is not a
    preference for leisure but a gift, and it is large enough to decide the
    answer on its own.

    Real systems reduce a benefit claimed early and increase one deferred,
    roughly enough to leave its expected present value alone. The factor that
    does that exactly is the ratio of annuity factors,
    ``A(reference) / A(age)`` -- derived here from the model's own Gompertz
    survival and discount factor rather than taken from a statute, so it is
    fair *by construction in this model* rather than approximately fair in
    someone else's.
    """
    base = annuity_factor(spec, survive, beta, int(reference_age))
    return {int(age): base / annuity_factor(spec, survive, beta, int(age))
            for age in ages}


def claim_factor_table(factors: Mapping[int, float], reference_age: int
                       ) -> pd.DataFrame:
    """The multipliers, as a table a reader can check against a real schedule."""
    rows = [{"retire_age": int(age), "reference_age": int(reference_age),
             "claim_factor": float(f),
             "adjustment_pct": (float(f) - 1.0) * 100.0,
             # Undefined at the reference age itself -- there is no year to
             # spread the adjustment over -- and NaN rather than zero so a
             # min/max over the column does not report a rate nobody faces.
             "per_year_pct": (((float(f) ** (1.0 / (int(reference_age) - int(age)))
                                - 1.0) * 100.0)
                              if int(age) != int(reference_age)
                              else float("nan"))}
            for age, f in sorted(factors.items())]
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Charging for the working years
# ---------------------------------------------------------------------------
def rescale(outcome: Any, spec: Any, leisure: float) -> Any:
    """The same outcome with its working years marked down to ``c / L``.

    Only ``consumption`` moves. Wealth, bequests and ruin are facts about the
    portfolio and are not re-denominated by how the investor feels about
    working.
    """
    leisure = float(leisure)
    if leisure < 1.0:
        raise ValueError(
            f"leisure value {leisure} is below 1.0, which would say a year of "
            "work is worth more than the same money in retirement")
    if leisure == 1.0:
        return outcome
    consumption = np.array(outcome.consumption, dtype=float, copy=True)
    consumption[:, :int(spec.n_working)] /= leisure
    return dataclasses.replace(outcome, consumption=consumption)


def _lifetime_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """``cfg`` with the aggregation widened to the whole life.

    A retirement-window objective cannot price a retirement date: it charges
    the investor for the years they worked and credits them nothing for the
    ones they did not.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    out["utility"] = dict(cfg["utility"])
    out["utility"]["consumption_window"] = "full"
    return out


def score(outcome: Any, spec: Any, cfg: Mapping[str, Any], gamma: float,
          leisure: float, survive: np.ndarray | None = None) -> Dict[str, Any]:
    """Lifetime certainty equivalent once working years are charged for."""
    from . import mortality as mort
    from . import utility as ut

    lifetime = _lifetime_cfg(cfg)
    adjusted = rescale(outcome, spec, leisure)
    util = lifetime["utility"]
    row: Dict[str, Any] = {
        "leisure": float(leisure),
        "cec": float(ut.crra_certainty_equivalent(
            ut.bundle_from_outcome(adjusted, lifetime, spec), gamma,
            float(util["discount_factor"]), float(util["bequest_weight"]),
            bool(util["bequest_enabled"]))),
        "prob_ruin": float(np.asarray(outcome.ruin).mean()),
        "mean_retirement_consumption": float(
            outcome.consumption[:, spec.retirement_slice].mean()),
        "n_paths": int(outcome.n_paths),
    }
    if survive is not None:
        row["cec_survival_weighted"] = float(mort.certainty_equivalent(
            adjusted, spec, lifetime, gamma, survive))
    return row


def sweep(simulate: Callable[[int, float], Any], spec: Any,
          cfg: Mapping[str, Any], gamma: float,
          ages: Sequence[int] = DEFAULT_AGES,
          leisures: Sequence[float] = DEFAULT_LEISURE,
          survive: np.ndarray | None = None,
          claim: Mapping[int, float] | None = None) -> pd.DataFrame:
    """Every retirement date scored at every value of leisure.

    ``simulate`` takes a retirement age and a benefit multiplier and returns
    one :class:`~src.lifecycle.LifecycleOutcome`. The lifetime is simulated
    once per date and then re-scored at each leisure value, because ``L`` only
    rescales consumption that has already been computed -- so the grid costs
    one simulation per age rather than one per cell.

    ``claim`` supplies the benefit multiplier per age. Passing ``None`` leaves
    the pension unreduced however early it starts, which is the model as every
    other section has it and is reported here as the diagnostic it is.
    """
    from . import plan as pl

    rows: List[Dict[str, Any]] = []
    for age in ages:
        factor = 1.0 if claim is None else float(claim[int(age)])
        LOGGER.info("simulating a career ending at %d (benefit x%.3f)",
                    int(age), factor)
        aged = pl.spec_for(spec, int(age))
        outcome = simulate(int(age), factor)
        for leisure in leisures:
            row = score(outcome, aged, cfg, gamma, float(leisure), survive)
            row["retire_age"] = int(age)
            row["working_years"] = int(aged.n_working)
            row["retired_years"] = int(aged.n_retired)
            row["claim_factor"] = factor
            row["mean_benefit"] = float(np.asarray(
                outcome.social_security).mean())
            rows.append(row)
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Reading the surface
# ---------------------------------------------------------------------------
def optimal_age(frame: pd.DataFrame, column: str = "cec") -> pd.DataFrame:
    """The best retirement date at each value of leisure."""
    rows: List[Dict[str, Any]] = []
    for leisure, block in frame.groupby("leisure", sort=True):
        ordered = block.sort_values(column, ascending=False)
        best = ordered.iloc[0]
        runner = ordered.iloc[1] if len(ordered) > 1 else best
        rows.append({
            "leisure": float(leisure),
            "leisure_pct": (float(leisure) - 1.0) * 100.0,
            "optimal_age": int(best["retire_age"]),
            "cec_at_optimum": float(best[column]),
            "runner_up_age": int(runner["retire_age"]),
            "margin_over_runner_up_pct": (
                (float(best[column]) / float(runner[column]) - 1.0) * 100.0
                if float(runner[column]) else float("nan")),
            "at_grid_floor": bool(int(best["retire_age"])
                                  == int(block["retire_age"].min())),
            "at_grid_ceiling": bool(int(best["retire_age"])
                                    == int(block["retire_age"].max())),
        })
    return pd.DataFrame.from_records(rows)


def crossing(frame: pd.DataFrame, age: int, reference: int,
             column: str = "cec") -> float:
    """The leisure value at which retiring at ``age`` overtakes ``reference``.

    Linear interpolation between the straddling grid points. Zero where
    ``age`` already wins with no value on leisure at all, and infinity where
    it never does -- both are answers, and both beat a missing value.
    """
    a = frame[frame["retire_age"] == int(age)].sort_values("leisure")
    b = frame[frame["retire_age"] == int(reference)].sort_values("leisure")
    if not len(a) or not len(b):
        return float("nan")
    grid = a["leisure"].to_numpy(dtype=float)
    gap = a[column].to_numpy(dtype=float) - b[column].to_numpy(dtype=float)
    if not np.isfinite(gap).any():
        return float("nan")
    if gap[0] >= 0.0:
        return float(grid[0])
    above = np.flatnonzero(gap >= 0.0)
    if above.size == 0:
        return float("inf")
    i = int(above[0])
    lo, hi = gap[i - 1], gap[i]
    if hi == lo:
        return float(grid[i])
    return float(grid[i - 1] + (grid[i] - grid[i - 1]) * (-lo) / (hi - lo))


def zero_leisure_optimum(frame: pd.DataFrame, column: str = "cec") -> int:
    """The best date when working costs nothing -- the natural comparison.

    Not the oldest age on the grid. In this model retirement consumption
    exceeds working consumption, so the lifetime objective already leans
    early before leisure enters, and a break-even measured against the grid's
    ceiling would report that lean rather than the value of leisure. The
    honest reference is the date an investor who enjoys work exactly as much
    as retirement would already choose.
    """
    block = frame.sort_values("leisure")
    lowest = float(block["leisure"].min())
    at_lowest = block[np.isclose(block["leisure"], lowest)]
    return int(at_lowest.loc[at_lowest[column].idxmax(), "retire_age"])


def break_even(frame: pd.DataFrame, reference: int | None = None,
               column: str = "cec") -> pd.DataFrame:
    """For each date, the leisure value that justifies it over working on.

    The section's deliverable. It turns an answer nobody can calibrate --
    "retire at sixty-one" -- into a question the reader can settle for
    themselves: is a year of your own time worth at least this much?

    The reference defaults to the date that wins when leisure is worth
    nothing, so what is priced is the *earlier* retirement rather than the
    model's own preference for it.
    """
    ages = sorted(int(a) for a in frame["retire_age"].unique())
    reference = int(reference if reference is not None
                    else zero_leisure_optimum(frame, column))
    rows: List[Dict[str, Any]] = []
    for age in ages:
        if age == reference:
            continue
        value = crossing(frame, age, reference, column)
        rows.append({
            "retire_age": age,
            "reference_age": reference,
            "years_earlier": reference - age,
            "is_earlier": bool(age < reference),
            "break_even_leisure": value,
            "break_even_pct": ((value - 1.0) * 100.0 if np.isfinite(value)
                               else float("inf")),
            "implied_consumption_drop": (1.0 - 1.0 / value
                                         if np.isfinite(value) and value > 0
                                         else float("nan")),
            "reached_on_grid": bool(np.isfinite(value)),
        })
    return pd.DataFrame.from_records(rows)


def system_comparison(optima: Mapping[str, pd.DataFrame],
                      crossings: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per pension system: where the date settles and what moving it
    costs.

    The break-evens are not comparable across systems as absolute ages -- each
    is measured against its own zero-leisure date, which is the point -- so
    the row carries that reference alongside them.
    """
    rows: List[Dict[str, Any]] = []
    for name, frame in optima.items():
        ordered = frame.sort_values("leisure")
        cross = crossings.get(name, pd.DataFrame())
        early = cross[cross["is_earlier"]] if len(cross) else cross
        reached = early[early["reached_on_grid"]] if len(early) else early
        row: Dict[str, Any] = {
            "system": name,
            "age_at_zero_leisure": int(ordered["optimal_age"].iloc[0]),
            "age_at_top": int(ordered["optimal_age"].iloc[-1]),
            "ages_chosen": len(set(ordered["optimal_age"])),
            "cec_at_zero_leisure": float(ordered["cec_at_optimum"].iloc[0]),
            "earlier_dates_reachable": int(len(reached)),
            "earlier_dates_offered": int(len(early)),
        }
        if len(reached):
            nearest = reached.loc[reached["years_earlier"].idxmin()]
            row.update({
                "nearest_years_earlier": int(nearest["years_earlier"]),
                "nearest_break_even_pct": float(nearest["break_even_pct"]),
                "cost_per_year_pct": (float(nearest["break_even_pct"])
                                      / max(int(nearest["years_earlier"]), 1)),
            })
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def system_verdict(comparison: pd.DataFrame) -> Dict[str, Any]:
    """What the shape of the pension does to the retirement date."""
    if not len(comparison):
        return {"measured": False}
    indexed = comparison.set_index("system")

    def _get(system: str, column: str) -> Any:
        return (indexed.loc[system, column] if system in indexed.index
                and column in indexed.columns else float("nan"))

    us = _get("us", "age_at_zero_leisure")
    gate = _get("au_pension_only", "age_at_zero_leisure")
    legislated = _get("au_as_legislated", "age_at_zero_leisure")
    found: Dict[str, Any] = {
        "measured": True,
        "systems": int(len(comparison)),
        "us_age": us, "gated_age": gate, "legislated_age": legislated,
    }
    if np.isfinite(us) and np.isfinite(gate):
        found["gate_pushes_later"] = bool(gate > us)
        found["gate_years_later"] = float(gate - us)
    if np.isfinite(gate) and np.isfinite(legislated):
        # The Superannuation Guarantee is the other half of the Australian
        # system, and it pushes the opposite way to the eligibility gate.
        found["super_buys_back"] = bool(legislated < gate)
        found["super_years_earlier"] = float(gate - legislated)
    if np.isfinite(us) and np.isfinite(legislated):
        found["legislated_vs_us_years"] = float(legislated - us)
        found["australia_retires_later"] = bool(legislated > us)
    if "cost_per_year_pct" in comparison.columns:
        priced = comparison.dropna(subset=["cost_per_year_pct"])
        if len(priced):
            found["cheapest_system"] = str(
                priced.loc[priced["cost_per_year_pct"].idxmin(), "system"])
            found["dearest_system"] = str(
                priced.loc[priced["cost_per_year_pct"].idxmax(), "system"])
        us_cost = _get("us", "cost_per_year_pct")
        au_cost = _get("au_as_legislated", "cost_per_year_pct")
        if np.isfinite(us_cost) and np.isfinite(au_cost) and us_cost > 0:
            # Two systems can move the *date* without moving the *slope*, and
            # the reader cannot tell which happened from the dates alone. The
            # ratio is what separates them, and it is classified here rather
            # than eyeballed: within a tenth either way is one story, outside
            # it is another.
            found["us_cost_per_year"] = float(us_cost)
            found["legislated_cost_per_year"] = float(au_cost)
            found["cost_ratio"] = float(au_cost / us_cost)
            found["cost_similar"] = bool(
                abs(au_cost / us_cost - 1.0) <= COST_SIMILAR_BAND)
            found["australia_dearer_per_year"] = bool(au_cost > us_cost)
    return found


def verdict(swept: pd.DataFrame, optima: pd.DataFrame,
            crossings: pd.DataFrame, anchors: pd.DataFrame,
            column: str = "cec") -> Dict[str, Any]:
    """What pricing the cost of working does to the retirement date."""
    if not len(optima):
        return {"measured": False}
    ordered = optima.sort_values("leisure")
    at_one = ordered[np.isclose(ordered["leisure"], 1.0)]
    reached = crossings[crossings["reached_on_grid"]] if len(crossings) \
        else crossings
    found: Dict[str, Any] = {
        "measured": True,
        "levels": int(len(ordered)),
        "highest_leisure": float(ordered["leisure"].iloc[-1]),
        "optimal_age_at_zero": (int(at_one["optimal_age"].iloc[0])
                                if len(at_one) else int("0")),
        "optimal_age_at_top": int(ordered["optimal_age"].iloc[-1]),
        # The control: with nothing charged for working, the date must run to
        # the ceiling, reproducing Section #plan's corner. If it does not,
        # something other than leisure is moving the answer.
        "corner_without_leisure": bool(len(at_one)
                                       and bool(at_one["at_grid_ceiling"].iloc[0])),
        "date_moves_with_leisure": bool(ordered["optimal_age"].nunique() > 1),
        "ages_chosen": sorted({int(a) for a in ordered["optimal_age"]}),
        "smallest_margin_pct": float(ordered["margin_over_runner_up_pct"].min()),
    }
    if len(reached):
        cheapest = reached.loc[reached["break_even_pct"].idxmin()]
        found.update({
            "cheapest_early_age": int(cheapest["retire_age"]),
            "cheapest_break_even_pct": float(cheapest["break_even_pct"]),
            "cheapest_implied_drop": float(cheapest["implied_consumption_drop"]),
        })
    if len(anchors):
        # Where each labelled calibration lands, by interpolation on the
        # solved surface rather than by assertion.
        placed = []
        for _, row in anchors.iterrows():
            value = float(row["leisure"])
            block = ordered[ordered["leisure"] <= value]
            placed.append({
                "label": str(row["label"]), "leisure": value,
                "optimal_age": (int(block["optimal_age"].iloc[-1])
                                if len(block)
                                else int(ordered["optimal_age"].iloc[0])),
            })
        found["anchor_ages"] = placed
        found["anchor_age_range"] = [min(p["optimal_age"] for p in placed),
                                     max(p["optimal_age"] for p in placed)]
    if "cec_survival_weighted" in swept.columns:
        alive = optimal_age(swept, "cec_survival_weighted").sort_values("leisure")
        # Compared across the whole grid rather than at one end: mortality
        # buys a year or two in the middle and can run out of grid at the
        # extremes, so a single-point comparison understates it.
        paired = ordered.merge(alive, on="leisure", suffixes=("", "_alive"))
        earlier = paired["optimal_age_alive"] < paired["optimal_age"]
        found.update({
            "survival_optimal_age_at_zero": int(alive["optimal_age"].iloc[0]),
            "survival_optimal_age_at_top": int(alive["optimal_age"].iloc[-1]),
            "survival_levels_earlier": int(earlier.sum()),
            "survival_levels": int(len(paired)),
            "survival_max_years_earlier": int(
                (paired["optimal_age"] - paired["optimal_age_alive"]).max()),
            "survival_pulls_earlier": bool(earlier.any()),
            "survival_corner_without_leisure": bool(
                bool(alive["at_grid_ceiling"].iloc[0])),
        })
    return found
