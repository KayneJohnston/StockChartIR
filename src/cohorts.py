"""The same lifetimes, without the bootstrap.

Every number elsewhere in this project comes from resampled histories. That
is the right way to get a distribution out of 131 years, but it leaves one
question open that no amount of resampling can answer: *is the result an
artefact of the resampling?* This module answers it by not resampling at all.

A **cohort** is one country and one birth year. The investor is 25 in the
first year of the window, holds the strategy through the realised returns of
that country and of the sleeve of every other market, in calendar order, and
dies 68 years later. Nothing is drawn. There is exactly one lifetime per
(country, start year) pair that the panel can support.

**What this buys and what it costs.** It buys a check on the sampler: if the
ordering survives with no resampling, it is not a property of the block
bootstrap. It costs almost all of the statistical power, for three reasons a
reader has to be told about rather than left to work out:

1. **The windows overlap.** A 131-year record yields 64 start years, but
   adjacent cohorts share 67 of their 68 years -- and once one 68-year
   lifetime is taken out of 131 years, the 63 that remain are too few for a
   second. Non-overlapping, the panel holds *one lifetime per market*, not
   the several hundred rows the table has. :func:`effective_sample` reports
   both counts, and the ratio between them is close to forty to one.
2. **The cohorts are cross-sectionally dependent.** Two world wars and one
   depression land inside almost every window. Inference is therefore by
   *cluster* bootstrap over countries (:func:`cluster_bootstrap`), never by a
   standard error over cohorts.
3. **A market closure removes a whole cohort, not a year.** The bootstrap
   simply refuses blocks that span the German 1944-49 gap and keeps drawing;
   a cohort that spans it cannot be run at all. Germany contributes four
   cohorts and the United States sixty-four. The countries whose domestic
   history was worst are the ones this design is least able to see, so it is
   structurally *kinder* to domestic equity than the bootstrap is.
   :func:`census` reports the count per country so the imbalance is on the
   page rather than in a footnote.

**Labour income is deterministic here.** Elsewhere it carries permanent and
transitory shocks. With a few hundred realised histories, income noise would
be a large share of the cross-sectional spread and none of it would be about
returns, which is the only thing this section is asking about.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import bootstrap as bs
from . import data_loader as dl
from . import lifecycle as lc
from . import utility as ut

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerating what the panel can support
# ---------------------------------------------------------------------------
def enumerate_cohorts(panel: dl.Panel, horizon: int) -> pd.DataFrame:
    """Every (country, start year) whose whole lifetime is on the record.

    A cohort is admissible only if the domestic country has an unbroken run
    of ``horizon`` usable years starting at that year -- the same rule the
    bootstrap applies to a block, applied to a lifetime.
    """
    rows: List[Dict[str, Any]] = []
    for c, iso in enumerate(panel.countries):
        usable = panel.available[:, c]
        for t in range(panel.years.size - horizon + 1):
            if usable[t:t + horizon].all():
                rows.append({
                    "iso": iso,
                    "country_index": c,
                    "start_index": t,
                    "start_year": int(panel.years[t]),
                    "end_year": int(panel.years[t + horizon - 1]),
                })
    return pd.DataFrame.from_records(
        rows, columns=["iso", "country_index", "start_index",
                       "start_year", "end_year"])


def census(panel: dl.Panel, horizon: int) -> pd.DataFrame:
    """Per country: usable years, longest unbroken run, cohorts supported.

    The last column is the one that matters. A country with a war-time gap in
    the middle of its record can have a hundred usable years and still support only
    a handful of lifetimes, because a lifetime cannot step over the gap.
    """
    cohorts = enumerate_cohorts(panel, horizon)
    counts = cohorts.groupby("iso").size() if len(cohorts) else pd.Series(dtype=int)
    rows: List[Dict[str, Any]] = []
    for c, iso in enumerate(panel.countries):
        usable = panel.available[:, c]
        longest = best = 0
        for flag in usable:
            best = best + 1 if flag else 0
            longest = max(longest, best)
        rows.append({
            "iso": iso,
            "usable_years": int(usable.sum()),
            "longest_unbroken": int(longest),
            "cohorts": int(counts.get(iso, 0)),
            "first_start": int(cohorts[cohorts["iso"] == iso]["start_year"].min())
            if counts.get(iso, 0) else -1,
            "last_start": int(cohorts[cohorts["iso"] == iso]["start_year"].max())
            if counts.get(iso, 0) else -1,
        })
    return pd.DataFrame.from_records(rows).sort_values(
        "cohorts", ascending=False).reset_index(drop=True)


def effective_sample(panel: dl.Panel, horizon: int) -> Dict[str, Any]:
    """How many *independent* lifetimes the overlapping cohorts amount to.

    The overlapping count is what the table has; the non-overlapping count is
    what the evidence is worth. Reporting only the first would overstate the
    precision by more than an order of magnitude.
    """
    cohorts = enumerate_cohorts(panel, horizon)
    independent = 0
    for c in range(len(panel.countries)):
        usable = panel.available[:, c]
        run = 0
        for flag in usable:
            run = run + 1 if flag else 0
            if run >= horizon:          # a whole lifetime fits; start a new one
                independent += 1
                run = 0
    return {
        "n_cohorts": int(len(cohorts)),
        "n_independent": int(independent),
        "overlap_ratio": float(len(cohorts) / independent) if independent else
        float("nan"),
        "n_countries": int(cohorts["iso"].nunique()) if len(cohorts) else 0,
        "horizon": int(horizon),
    }


# ---------------------------------------------------------------------------
# Realised lifetimes as bootstrap paths
# ---------------------------------------------------------------------------
def cohort_paths(panel: dl.Panel, cohorts: pd.DataFrame,
                 horizon: int) -> bs.BootstrapPaths:
    """One realised lifetime per row, in the shape the simulator expects.

    Returning a :class:`~src.bootstrap.BootstrapPaths` rather than a bespoke
    container means the lifecycle simulator, the utility layer and the
    summariser used everywhere else run over these paths unmodified. The only
    difference from a bootstrap chunk is how the rows were built: a slice of
    calendar history instead of a resample of one.
    """
    if not len(cohorts):
        raise ValueError("no admissible cohorts; the panel is too short")
    idx = cohorts["start_index"].to_numpy(dtype=int)
    ctry = cohorts["country_index"].to_numpy(dtype=int)
    offsets = np.arange(horizon)[None, :]
    rows = idx[:, None] + offsets                      # (n_cohorts, horizon)
    cols = np.repeat(ctry[:, None], horizon, axis=1)

    def take(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=float)[rows, cols]

    return bs.BootstrapPaths(
        dom_eq=take(panel.dom_eq),
        intl_eq=take(panel.intl_eq),
        bond=take(panel.bond),
        bill=take(panel.bill),
        inflation=take(panel.inflation),
        domestic_country=cols.astype(int),
        calendar_index=rows.astype(int),
        # One unbroken block per lifetime: that is the whole point.
        block_id=np.repeat(np.arange(len(cohorts))[:, None], horizon, axis=1),
    )


def run(panel: dl.Panel, cohorts: pd.DataFrame, spec: lc.LifecycleSpec,
        strategies: Mapping[str, lc.Strategy]
        ) -> Dict[str, lc.LifecycleOutcome]:
    """Every strategy over every realised cohort, on identical income."""
    paths = cohort_paths(panel, cohorts, spec.horizon)
    deterministic = dataclasses.replace(spec, income_shocks_enabled=False)
    income = lc.simulate_income(deterministic, paths.n_paths)
    return lc.simulate_all(paths, strategies, deterministic, income)


# ---------------------------------------------------------------------------
# Reading the cohorts
# ---------------------------------------------------------------------------
def _cec(outcome: lc.LifecycleOutcome, cfg: Mapping[str, Any],
         spec: lc.LifecycleSpec, gamma: float) -> float:
    bundle = ut.bundle_from_outcome(outcome, cfg, spec)
    return ut.crra_certainty_equivalent(
        bundle, gamma, float(cfg["utility"]["discount_factor"]),
        float(cfg["utility"]["bequest_weight"]), include_bequest=True)


def summarise(outcomes: Mapping[str, lc.LifecycleOutcome],
              cfg: Mapping[str, Any], spec: lc.LifecycleSpec,
              gamma: float) -> pd.DataFrame:
    """Certainty equivalent and tail statistics over the realised cohorts."""
    retirement = spec.retirement_slice
    rows: List[Dict[str, Any]] = []
    for key, outcome in outcomes.items():
        mean_retired = outcome.consumption[:, retirement].mean(axis=1)
        rows.append({
            "strategy": key,
            "label": outcome.label,
            f"cec_crra_gamma{gamma:g}": _cec(outcome, cfg, spec, gamma),
            "median_retirement_consumption": float(np.median(mean_retired)),
            "p5_retirement_consumption": float(np.percentile(mean_retired, 5)),
            "prob_ruin": float(outcome.ruin.mean()),
            "median_bequest": float(np.median(outcome.bequest)),
        })
    column = f"cec_crra_gamma{gamma:g}"
    return pd.DataFrame.from_records(rows).sort_values(
        column, ascending=False).reset_index(drop=True)


def per_cohort(outcomes: Mapping[str, lc.LifecycleOutcome],
               cohorts: pd.DataFrame, spec: lc.LifecycleSpec,
               pair: Tuple[str, str]) -> pd.DataFrame:
    """One row per realised lifetime: what each arm delivered, and the gap.

    The comparison is within a cohort -- the same country, the same calendar
    years, the same labour income -- so the difference is the allocation and
    nothing else.
    """
    first, second = pair
    if first not in outcomes or second not in outcomes:
        raise KeyError(f"outcomes lack one of the pair {pair}")
    retirement = spec.retirement_slice
    a = outcomes[first].consumption[:, retirement].mean(axis=1)
    b = outcomes[second].consumption[:, retirement].mean(axis=1)
    frame = cohorts[["iso", "start_year", "end_year"]].copy()
    frame[first] = a
    frame[second] = b
    frame["gap_pct"] = (a / b - 1.0) * 100.0
    frame["first_wins"] = a > b
    frame["ruin_first"] = outcomes[first].ruin
    frame["ruin_second"] = outcomes[second].ruin
    return frame.reset_index(drop=True)


def by_country(detail: pd.DataFrame, pair: Tuple[str, str]) -> pd.DataFrame:
    """The cohort record collapsed to one row per market."""
    grouped = detail.groupby("iso")
    out = pd.DataFrame({
        "cohorts": grouped.size(),
        "mean_gap_pct": grouped["gap_pct"].mean(),
        "median_gap_pct": grouped["gap_pct"].median(),
        "win_rate": grouped["first_wins"].mean(),
        "first_start": grouped["start_year"].min(),
        "last_start": grouped["start_year"].max(),
    })
    return out.sort_values("mean_gap_pct").reset_index()


def long_run_returns(panel: dl.Panel, cohorts: pd.DataFrame,
                     horizon: int) -> pd.DataFrame:
    """The realised 68-year compound return of each leg, cohort by cohort.

    This is the diagnostic for why a realised lifetime need not agree with a
    resampled one in *size*. A bootstrap lifetime is a mosaic of blocks drawn
    from different decades, so a market that underperformed for forty years
    running contributes only a few of those blocks and the rest of the
    lifetime is spliced in from better eras. A cohort cannot splice. If the
    realised spread between the two legs is wider than the resampled one,
    the sampler is diluting persistence, and the bootstrap headline is the
    conservative number.
    """
    paths = cohort_paths(panel, cohorts, horizon)

    def compound(matrix: np.ndarray) -> np.ndarray:
        return np.exp(np.log1p(matrix).sum(axis=1) / horizon) - 1.0

    frame = cohorts[["iso", "start_year", "end_year"]].copy()
    frame["domestic_annualised"] = compound(paths.dom_eq)
    frame["sleeve_annualised"] = compound(paths.intl_eq)
    frame["excess_pp"] = (frame["sleeve_annualised"]
                          - frame["domestic_annualised"]) * 100.0
    return frame.reset_index(drop=True)


def dispersion(realised: pd.DataFrame) -> Dict[str, Any]:
    """Spread of the realised long-run legs, for comparison with the sampler."""
    return {
        "domestic_sd_pp": float(realised["domestic_annualised"].std(ddof=1)
                                * 100.0),
        "sleeve_sd_pp": float(realised["sleeve_annualised"].std(ddof=1) * 100.0),
        "excess_sd_pp": float(realised["excess_pp"].std(ddof=1)),
        "mean_excess_pp": float(realised["excess_pp"].mean()),
        "share_sleeve_ahead": float((realised["excess_pp"] > 0).mean()),
        "worst_domestic_pp": float(realised["domestic_annualised"].min() * 100.0),
        "best_domestic_pp": float(realised["domestic_annualised"].max() * 100.0),
        "worst_sleeve_pp": float(realised["sleeve_annualised"].min() * 100.0),
        "best_sleeve_pp": float(realised["sleeve_annualised"].max() * 100.0),
    }


# ---------------------------------------------------------------------------
# Inference that respects the dependence
# ---------------------------------------------------------------------------
def cluster_bootstrap(detail: pd.DataFrame, n_boot: int = 2000,
                      seed: int = 20260901,
                      column: str = "gap_pct") -> Dict[str, Any]:
    """Resample *countries*, not cohorts, and rebuild the mean gap.

    Cohorts within a country overlap almost completely and cohorts across
    countries share the same wars, so neither is an independent draw. A
    country is the coarsest unit this panel offers, and resampling at that
    level is the honest interval. It is wide, and it should be.
    """
    isos = sorted(detail["iso"].unique())
    groups = {iso: detail.loc[detail["iso"] == iso, column].to_numpy(dtype=float)
              for iso in isos}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    equal_draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        picked = rng.choice(len(isos), size=len(isos), replace=True)
        chosen = [groups[isos[i]] for i in picked]
        draws[b] = float(np.concatenate(chosen).mean())
        # The same resample, read the other way: each drawn country gets one
        # vote regardless of how many cohorts it contributed.
        equal_draws[b] = float(np.mean([c.mean() for c in chosen]))
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    eq_lo, eq_hi = (float(v) for v in np.percentile(equal_draws, [2.5, 97.5]))
    point = float(detail[column].mean())
    per_country = np.array([groups[iso].mean() for iso in isos])
    equal_point = float(per_country.mean())
    # A percentile interval on this few clusters is known to under-cover, so
    # the Student-t interval on the between-country spread is reported beside
    # it. Where the two disagree the wider one is the one to believe.
    n = len(isos)
    t_crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    t_se = float(per_country.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "mean_gap_pct": point,
        "ci_low": lo,
        "ci_high": hi,
        "se": float(draws.std(ddof=1)),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        # Each market one vote. The pooled mean above is implicitly weighted
        # by how much unbroken history a country happens to have, which gives
        # the five markets with a full run sixteen times the weight of Germany.
        "equal_weighted_gap_pct": equal_point,
        "equal_ci_low": eq_lo,
        "equal_ci_high": eq_hi,
        "equal_excludes_zero": bool(eq_lo > 0.0 or eq_hi < 0.0),
        "weighting_shift_pp": equal_point - point,
        # And the parametric cross-check on the same clusters.
        "t_ci_low": equal_point - t_crit * t_se,
        "t_ci_high": equal_point + t_crit * t_se,
        "t_excludes_zero": bool(equal_point - t_crit * t_se > 0.0
                                or equal_point + t_crit * t_se < 0.0),
        "n_boot": int(n_boot),
        "n_clusters": n,
    }


def sign_test(detail: pd.DataFrame) -> Dict[str, Any]:
    """How often the first arm won, counted by country rather than by cohort.

    A cohort-level win rate would count the United States sixty-four times
    and Germany four. Averaging within a country first, and then across
    countries, gives each market one vote -- which is the number of
    independent votes this panel actually contains.
    """
    per_country = detail.groupby("iso")["first_wins"].mean()
    return {
        "cohort_win_rate": float(detail["first_wins"].mean()),
        "country_win_rate": float(per_country.mean()),
        "countries_won": int((per_country > 0.5).sum()),
        "countries_total": int(per_country.size),
    }


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
def verdict(summary: pd.DataFrame, detail: pd.DataFrame,
            interval: Mapping[str, Any], signs: Mapping[str, Any],
            effective: Mapping[str, Any],
            pair: Tuple[str, str]) -> Dict[str, Any]:
    """What the realised cohorts say, classified rather than asserted."""
    column = [c for c in summary.columns if c.startswith("cec_crra_")][0]
    ordered = list(summary["strategy"])
    values = dict(zip(summary["strategy"], summary[column]))
    gap = ((values[pair[0]] / values[pair[1]] - 1.0) * 100.0
           if pair[0] in values and pair[1] in values else float("nan"))
    worst = detail.loc[detail["gap_pct"].idxmin()] if len(detail) else None
    best = detail.loc[detail["gap_pct"].idxmax()] if len(detail) else None
    return {
        "winner": ordered[0] if ordered else "",
        "winner_is_expected": bool(ordered[:1] == [pair[0]]),
        "cec_gap_pct": gap,
        "gap_has_same_sign_as_bootstrap": bool(gap > 0.0),
        "mean_cohort_gap_pct": float(interval["mean_gap_pct"]),
        "interval_excludes_zero": bool(interval["excludes_zero"]),
        "ci_low": float(interval["ci_low"]),
        "ci_high": float(interval["ci_high"]),
        "cohort_win_rate": float(signs["cohort_win_rate"]),
        "country_win_rate": float(signs["country_win_rate"]),
        "every_country_favours_first": bool(
            signs["countries_won"] == signs["countries_total"]),
        "countries_won": int(signs["countries_won"]),
        "countries_total": int(signs["countries_total"]),
        "n_cohorts": int(effective["n_cohorts"]),
        "n_independent": int(effective["n_independent"]),
        "worst_cohort": ("" if worst is None else
                         f"{worst['iso']} {int(worst['start_year'])}"),
        "worst_gap_pct": float("nan") if worst is None else float(worst["gap_pct"]),
        "best_cohort": ("" if best is None else
                        f"{best['iso']} {int(best['start_year'])}"),
        "best_gap_pct": float("nan") if best is None else float(best["gap_pct"]),
    }
