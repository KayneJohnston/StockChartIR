"""How much of the headline rests on any one country, or on any one era?

The sensitivity sweep varies the preferences and the lifecycle. The sleeve
study varies how the international leg is weighted. Neither touches the thing
a sceptical reader asks about first: the panel is **sixteen developed markets
that survived**, and a result assembled from sixteen histories could be one
history wearing a disguise. If dropping the United States overturns the
ranking, the paper is about the United States.

Three questions, answered on the same machinery:

**Influence.** Rebuild the panel sixteen times, each time with one country
removed, and re-run the headline. The removal has to be genuine: a dropped
market must vanish from every other country's international sleeve as well as
from the set its own domestic leg is drawn from, which is why this uses
:func:`src.data_loader.build_tier_a` with a restricted country list rather
than slicing a panel that has already been built.

**Uncertainty.** Those sixteen runs are a delete-one jackknife, so they also
give a standard error for the headline gap that reflects the *panel's* size
rather than the Monte Carlo's. That distinction matters: 100,000 bootstrap
paths make the simulation error tiny while leaving the sixteen-country
sampling error exactly where it was, and reporting only the first would be
a precision the data does not support.

**Stability.** Re-run on expanding windows -- 1890 to 1950, to 1970, to 1990,
to the end -- and on the two halves of the sample. An investor in 1970 had
only the first of those. If the ranking is a late-century artefact, this is
where it shows.

Every comparison needs a noise floor, because a delete-one run draws its own
bootstrap paths and will differ from the full panel even if the country it
dropped was irrelevant. :func:`noise_floor` re-runs the *unmodified* panel
under several seeds to measure that, so a shift can be read against something
rather than assumed meaningful.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl

LOGGER = logging.getLogger(__name__)

#: The two strategies whose ordering is this project's divergence from the
#: study it re-implements.
DEFAULT_PAIR: Tuple[str, str] = ("international_equity", "balanced_all_equity")

#: Expanding windows, as end years. Each asks what an investor standing at
#: that date would have concluded from the record available to them.
DEFAULT_WINDOWS: Tuple[int, ...] = (1950, 1970, 1990, 2020)

#: Extra seeds for the noise floor. Three is enough to bound it; the point is
#: a scale, not a precise variance.
DEFAULT_SEEDS: Tuple[int, ...] = (101, 202, 303)

#: A summariser takes a panel, a path count and optionally a configuration to
#: override the caller's, and returns the headline comparison table. It is
#: passed in rather than reimplemented here so that every run in this module
#: is scored by exactly the code that produces the headline itself.
Summariser = Callable[..., pd.DataFrame]


def _cec_column(frame: pd.DataFrame) -> str:
    matches = [c for c in frame.columns if c.startswith("cec_crra_")]
    if not matches:
        raise KeyError("no CRRA certainty-equivalent column in the summary")
    return matches[0]


def gap(frame: pd.DataFrame, pair: Tuple[str, str] = DEFAULT_PAIR) -> float:
    """Percentage lead of the first strategy over the second."""
    cec = _cec_column(frame)
    rows = {r["strategy"]: float(r[cec]) for _, r in frame.iterrows()}
    challenger, incumbent = pair
    if challenger not in rows or incumbent not in rows:
        return float("nan")
    return (rows[challenger] / rows[incumbent] - 1.0) * 100.0


def _tag(frame: pd.DataFrame, **columns: Any) -> pd.DataFrame:
    out = frame.copy()
    for i, (name, value) in enumerate(columns.items()):
        out.insert(i, name, value)
    return out


# ---------------------------------------------------------------------------
# Influence: one country at a time
# ---------------------------------------------------------------------------
def leave_one_out(cfg: Mapping[str, Any], summarise: Summariser,
                  n_paths: int, countries: Sequence[str],
                  weighting: str | None = None) -> pd.DataFrame:
    """Re-run the headline once per country, with that country removed.

    The removal is from the panel and from the sleeve pool alike, so each run
    answers "what would this paper say if that market's history had never been
    recorded" rather than the weaker "what if we had not invested there".
    """
    frames: List[pd.DataFrame] = []
    for dropped in countries:
        kept = [c for c in countries if c != dropped]
        LOGGER.info("leave-one-out: dropping %s (%d markets left)",
                    dropped, len(kept))
        panel = dl.build_tier_a(cfg, weighting=weighting, countries=kept)
        frames.append(_tag(summarise(panel, n_paths), dropped=dropped,
                           n_markets=len(kept)))
    return pd.concat(frames, ignore_index=True)


def influence(loco: pd.DataFrame, full: pd.DataFrame,
              pair: Tuple[str, str] = DEFAULT_PAIR) -> pd.DataFrame:
    """Per-country influence on the headline gap, most negative first.

    ``shift_pct`` is what removing that country does to the lead. A large
    negative number means the country was holding the result up; a large
    positive one means it was holding it down.
    """
    cec = _cec_column(full)
    baseline = gap(full, pair)
    challenger, incumbent = pair
    rows: List[Dict[str, Any]] = []
    for dropped in loco["dropped"].unique():
        block = loco[loco["dropped"] == dropped]
        by_key = {r["strategy"]: r for _, r in block.iterrows()}
        this = gap(block, pair)
        rows.append({
            "dropped": str(dropped),
            "gap_pct": this,
            "shift_pct": this - baseline,
            "challenger_cec": float(by_key[challenger][cec])
            if challenger in by_key else float("nan"),
            "incumbent_cec": float(by_key[incumbent][cec])
            if incumbent in by_key else float("nan"),
            "ordering_holds": bool(np.isfinite(this) and this > 0.0),
        })
    frame = pd.DataFrame.from_records(rows)
    frame.attrs["baseline_gap_pct"] = baseline
    return frame.sort_values("shift_pct").reset_index(drop=True)


def jackknife(influence_frame: pd.DataFrame,
              baseline_gap: float | None = None) -> Dict[str, Any]:
    """Delete-one jackknife standard error and bias for the headline gap.

    With ``n`` countries and ``g_i`` the gap computed without country ``i``,
    the jackknife variance is ``(n-1)/n * sum (g_i - mean(g))^2``. It is the
    sampling error the *panel* carries, and it is the number a reader should
    weigh the headline against -- not the Monte Carlo error, which 100,000
    paths drive close to zero without adding a single country of evidence.
    """
    values = influence_frame["gap_pct"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return {"n": int(n), "standard_error": float("nan")}
    mean = float(values.mean())
    variance = (n - 1) / n * float(((values - mean) ** 2).sum())
    se = float(np.sqrt(variance))
    base = float(baseline_gap
                 if baseline_gap is not None
                 else influence_frame.attrs.get("baseline_gap_pct", mean))
    out: Dict[str, Any] = {
        "n": int(n),
        "jackknife_mean": mean,
        "standard_error": se,
        "bias": (n - 1) * (mean - base),
        "baseline_gap_pct": base,
        "ci_low": base - 1.96 * se,
        "ci_high": base + 1.96 * se,
    }
    out["excludes_zero"] = bool(out["ci_low"] > 0.0)
    # A binary "excludes zero" hides how close the call is: an interval that
    # clears zero by a hundredth of a point and one that clears it by five
    # both read as a pass. The t-statistic does not hide it.
    out["t_stat"] = base / se if se > 0 else float("inf")
    out["marginal"] = bool(1.96 <= abs(out["t_stat"]) < 2.5)
    return out


# ---------------------------------------------------------------------------
# Stability: one era at a time
# ---------------------------------------------------------------------------
def restrict_years(panel: dl.Panel, start: int, end: int) -> dl.Panel:
    """The panel with country-years outside ``[start, end]`` unavailable.

    Only ``available`` changes, so the sleeve each year is exactly the sleeve
    that year had. Masking is the honest restriction here: rebuilding the
    international leg from the window alone would leave a 1930 sleeve unable
    to see 1929, which is not what an investor of the period faced.
    """
    mask = (panel.years >= int(start)) & (panel.years <= int(end))
    return dataclasses.replace(
        panel, available=panel.available & mask[:, None],
        name=f"{panel.name}[{start}-{end}]")


def subperiods(panel: dl.Panel, summarise: Summariser, n_paths: int,
               windows: Sequence[int] = DEFAULT_WINDOWS,
               split: bool = True) -> pd.DataFrame:
    """Re-run the headline on expanding windows, and on the two halves."""
    first, last = int(panel.years[0]), int(panel.years[-1])
    spans: List[Tuple[str, int, int]] = [
        (f"{first}-{min(int(end), last)}", first, min(int(end), last))
        for end in windows]
    if split:
        middle = first + (last - first) // 2
        spans += [(f"{first}-{middle} (first half)", first, middle),
                  (f"{middle + 1}-{last} (second half)", middle + 1, last)]
    frames: List[pd.DataFrame] = []
    for label, start, end in spans:
        restricted = restrict_years(panel, start, end)
        years = int(restricted.available.any(axis=1).sum())
        observations = int(restricted.available.sum())
        LOGGER.info("sub-period %s: %d years, %d country-years",
                    label, years, observations)
        frames.append(_tag(summarise(restricted, n_paths), window=label,
                           start_year=start, end_year=end,
                           years=years, country_years=observations))
    return pd.concat(frames, ignore_index=True)


def period_summary(periods: pd.DataFrame,
                   pair: Tuple[str, str] = DEFAULT_PAIR) -> pd.DataFrame:
    """One row per window: the gap, and whether the ordering survives it."""
    rows: List[Dict[str, Any]] = []
    for label in periods["window"].unique():
        block = periods[periods["window"] == label]
        this = gap(block, pair)
        rows.append({
            "window": str(label),
            "start_year": int(block["start_year"].iloc[0]),
            "end_year": int(block["end_year"].iloc[0]),
            "country_years": int(block["country_years"].iloc[0]),
            "gap_pct": this,
            "ordering_holds": bool(np.isfinite(this) and this > 0.0),
        })
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# The noise floor every comparison above is read against
# ---------------------------------------------------------------------------
def noise_floor(cfg: Mapping[str, Any], summarise: Summariser, n_paths: int,
                seeds: Sequence[int] = DEFAULT_SEEDS,
                weighting: str | None = None) -> pd.DataFrame:
    """The unmodified panel, re-run under several bootstrap seeds.

    A delete-one run draws its own paths, so it differs from the full panel
    even when the country it dropped carried no information. Without this
    control every shift looks like a finding.
    """
    panel = dl.build_tier_a(cfg, weighting=weighting)
    frames: List[pd.DataFrame] = []
    for seed in seeds:
        scoped = dict(cfg)
        scoped["bootstrap"] = {**cfg["bootstrap"], "seed": int(seed)}
        LOGGER.info("noise floor: full panel at seed %d", seed)
        frames.append(_tag(summarise(panel, n_paths, scoped), seed=int(seed)))
    return pd.concat(frames, ignore_index=True)


def floor_summary(floor: pd.DataFrame,
                  pair: Tuple[str, str] = DEFAULT_PAIR) -> Dict[str, Any]:
    """Spread of the headline gap across seeds, on an unchanged panel."""
    values = [gap(floor[floor["seed"] == s], pair)
              for s in floor["seed"].unique()]
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return {"seeds": 0, "range_pct": float("nan")}
    return {
        "seeds": int(values.size),
        "mean_gap_pct": float(values.mean()),
        "min_gap_pct": float(values.min()),
        "max_gap_pct": float(values.max()),
        "range_pct": float(values.max() - values.min()),
        "sd_pct": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Verdicts, classified rather than asserted
# ---------------------------------------------------------------------------
def verdict(influence_frame: pd.DataFrame, jack: Mapping[str, Any],
            floor: Mapping[str, Any],
            period: pd.DataFrame | None = None) -> Dict[str, Any]:
    """What the three studies jointly say about the headline."""
    holds = influence_frame["ordering_holds"]
    worst = influence_frame.iloc[0] if len(influence_frame) else None
    best = influence_frame.iloc[-1] if len(influence_frame) else None
    noise = float(floor.get("range_pct", float("nan")))
    material = influence_frame[
        influence_frame["shift_pct"].abs() > noise] \
        if np.isfinite(noise) else influence_frame
    out: Dict[str, Any] = {
        "n_countries": int(len(influence_frame)),
        "survives_every_deletion": bool(holds.all()) if len(holds) else False,
        "n_deletions_that_break_it": int((~holds).sum()) if len(holds) else 0,
        "worst_country": str(worst["dropped"]) if worst is not None else "",
        "worst_gap_pct": float(worst["gap_pct"]) if worst is not None else float("nan"),
        "worst_shift_pct": float(worst["shift_pct"]) if worst is not None else float("nan"),
        "best_country": str(best["dropped"]) if best is not None else "",
        "best_shift_pct": float(best["shift_pct"]) if best is not None else float("nan"),
        "noise_range_pct": noise,
        "n_material_countries": int(len(material)),
        "material_countries": [str(v) for v in material["dropped"]],
        "standard_error": float(jack.get("standard_error", float("nan"))),
        "ci_low": float(jack.get("ci_low", float("nan"))),
        "ci_high": float(jack.get("ci_high", float("nan"))),
        "excludes_zero": bool(jack.get("excludes_zero", False)),
    }
    if period is not None and len(period):
        out["n_windows"] = int(len(period))
        out["windows_holding"] = int(period["ordering_holds"].sum())
        out["all_windows_hold"] = bool(period["ordering_holds"].all())
        weakest = period.loc[period["gap_pct"].idxmin()]
        out["weakest_window"] = str(weakest["window"])
        out["weakest_window_gap_pct"] = float(weakest["gap_pct"])
    return out
