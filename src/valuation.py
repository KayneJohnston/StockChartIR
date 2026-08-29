"""Starting valuation, and what it does to a lifetime.

The bootstrap of `docs/02` draws calendar windows without regard to how
expensive markets were when the window opened. A lifetime beginning at a market
peak is therefore statistically identical to one beginning at a trough, which is
a strong assumption and an unrealistic one: dividend yields predict subsequent
long-horizon real equity returns in this panel, and the investor reading this
knows what today's yield is.

This module supplies the missing state variable and does so under one hard
constraint -- **nothing may use information the investor could not have had.**

**The observable.** The workbook's ``eq_dp`` is a dividend *return*: the
dividend paid during year `t` over the price at the start of it, D_t / P_{t-1}.
Conditioning on that would leak, because D_t is unknown until year `t` is over.
The quantity an investor standing at the start of year `t` actually observes is
the trailing yield on the current price,

    y_{t-1} = D_{t-1} / P_{t-1} = eq_dp_{t-1} / (1 + eq_capgain_{t-1})

which uses only the dividend paid and the price reached in the year that has
already finished. :func:`trailing_yield` builds exactly that. The check is structural rather
than statistical: :func:`depends_only_on_past` tampers with one year's raw
inputs and confirms that no earlier row moves. A correlation test could not do
this job, since a leak and an honest signal both show up as predictive power.

**The international sleeve.** The international leg is an equal-weighted
leave-one-out average of other countries' equity. For a portfolio holding equal
*money* in each market, the portfolio dividend yield is the plain mean of the
constituent yields -- not the median, and not a value weighting -- so
:func:`international_yield` mirrors the leg's own construction. The median is
reported alongside as a robustness check, since one country trading at a
distressed yield can pull a 15-country mean.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl

LOGGER = logging.getLogger(__name__)

#: Quantile edges for the valuation buckets. Terciles keep enough paths in each
#: bucket for the certainty equivalent to be estimated precisely; quintiles
#: thin the extremes to the point where the comparison is noise.
DEFAULT_EDGES: Tuple[float, ...] = (0.0, 1 / 3, 2 / 3, 1.0)

#: Names for those buckets, cheapest last: a high dividend yield is a low price.
BUCKET_LABELS: Tuple[str, ...] = ("Expensive", "Middling", "Cheap")


def trailing_yield(jst: pd.DataFrame, isos: Sequence[str],
                   years: np.ndarray) -> np.ndarray:
    """``(T, C)`` trailing dividend yield, observable at the start of each year.

    ``y_{t-1} = D_{t-1} / P_{t-1}``, built from the previous year's dividend
    return and capital gain. The value stored at row ``t`` is what an investor
    could see on the first day of year ``t``, so pairing it with the return of
    year ``t`` onwards involves no look-ahead.
    """
    out = np.full((years.size, len(isos)), np.nan)
    for j, iso in enumerate(isos):
        block = jst[jst["iso"] == iso].set_index("year")
        if block.empty:
            continue
        dp = block["eq_dp"].reindex(years).to_numpy(dtype=float)
        gain = block["eq_capgain"].reindex(years).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            current = dp / (1.0 + gain)
        # Shift by one: the yield an investor sees entering year t is the one
        # that had been established by the close of year t-1.
        lagged = np.roll(current, 1)
        lagged[0] = np.nan
        out[:, j] = np.where(np.isfinite(lagged), lagged, np.nan)
    return out


def international_yield(domestic: np.ndarray) -> np.ndarray:
    """Leave-one-out equal-weighted mean yield, matching the international leg.

    The leg holds equal money in every other country with data that year, and
    the dividend yield of an equally weighted portfolio is the mean of its
    constituents' yields. Countries missing a yield that year are excluded from
    the average rather than treated as zero.
    """
    finite = np.isfinite(domestic)
    filled = np.where(finite, domestic, 0.0)
    total = filled.sum(axis=1, keepdims=True)
    count = finite.sum(axis=1, keepdims=True)
    others = total - filled
    n_others = count - finite
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = others / n_others
    return np.where((n_others > 0) & finite, mean, np.nan)


def international_yield_median(domestic: np.ndarray) -> np.ndarray:
    """The same leave-one-out average taken as a median, for robustness."""
    n_years, n_countries = domestic.shape
    out = np.full_like(domestic, np.nan)
    for i in range(n_countries):
        others = np.delete(domestic, i, axis=1)
        with np.errstate(invalid="ignore"):
            out[:, i] = np.nanmedian(others, axis=1)
    return np.where(np.isfinite(domestic), out, np.nan)


def blended_yield(domestic: np.ndarray, international: np.ndarray,
                  domestic_share: float = 0.5) -> np.ndarray:
    """The yield on the headline 50/50 portfolio, for a single ranking axis."""
    share = float(domestic_share)
    return share * domestic + (1.0 - share) * international


def depends_only_on_past(jst: pd.DataFrame, isos: Sequence[str],
                         years: np.ndarray, probe_year: int,
                         builder: Any = None) -> bool:
    """Structural proof that the yield at year ``t`` uses no year-``t`` data.

    Correlations cannot establish this. A yield built from the current year's
    dividend would still *predict* the current year's return, and a correctly
    lagged one still correlates with the prior year's return because a bad year
    lowers the price in the denominator. Both are expected; neither
    distinguishes a leak from an honest signal.

    So the property is tested by construction instead: overwrite everything the
    workbook records for ``probe_year``, rebuild, and check the row for that
    year did not move. If year-``t`` data reached the year-``t`` yield, it must.

    ``builder`` is the series constructor under test, defaulting to
    :func:`trailing_yield`. It is a parameter so that the test suite can hand
    in a deliberately leaking construction and confirm this check rejects it --
    a check that has never been shown to fail is not evidence of anything.
    """
    build = trailing_yield if builder is None else builder
    baseline = build(jst, isos, years)
    tampered = jst.copy()
    mask = tampered["year"] == int(probe_year)
    for column in ("eq_dp", "eq_capgain", "eq_tr"):
        if column in tampered.columns:
            tampered.loc[mask, column] = 99.0
    after = build(tampered, isos, years)
    row = int(np.flatnonzero(np.asarray(years) == int(probe_year))[0])
    return bool(np.allclose(baseline[row], after[row], equal_nan=True))


def predictive_power(yields: np.ndarray, returns: np.ndarray,
                     horizons: Sequence[int] = (1, 10, 20, 30)) -> pd.DataFrame:
    """Does the observable yield predict subsequent real equity returns?

    One row per horizon: the correlation between the yield an investor could
    see and the annualised real return over the years that follow it, plus the
    means in the cheapest and dearest thirds of the distribution. This is a
    finding rather than a check -- the conditioning is only worth doing if the
    answer is yes.
    """
    rows: List[Dict[str, Any]] = []
    n_years, n_countries = returns.shape
    for h in horizons:
        pairs: List[Tuple[float, float]] = []
        for j in range(n_countries):
            for t in range(n_years - h):
                window = returns[t:t + h, j]
                if np.isfinite(yields[t, j]) and np.isfinite(window).all():
                    annualised = float(np.exp(np.mean(np.log1p(window))) - 1.0)
                    pairs.append((float(yields[t, j]), annualised))
        if len(pairs) < 50:
            continue
        frame = pd.DataFrame(pairs, columns=["yield", "forward"])
        low, high = frame["yield"].quantile(1 / 3), frame["yield"].quantile(2 / 3)
        dear = float(frame[frame["yield"] <= low]["forward"].mean())
        cheap = float(frame[frame["yield"] >= high]["forward"].mean())
        rows.append({
            "horizon_years": int(h),
            "observations": int(len(frame)),
            "correlation": float(frame["yield"].corr(frame["forward"])),
            "forward_return_expensive": dear,
            "forward_return_cheap": cheap,
            "gap": cheap - dear,
        })
    return pd.DataFrame.from_records(rows)


def path_starting_yield(paths: Any, blended: np.ndarray) -> np.ndarray:
    """The yield each simulated lifetime began at, one number per path.

    A path is a chain of calendar windows. Only the first is a starting
    condition -- the rest are the future, which the investor does not choose --
    so the state variable is the blended yield at the first drawn country-year.
    """
    first_year = np.asarray(paths.calendar_index)[:, 0]
    first_country = np.asarray(paths.domestic_country)[:, 0]
    return blended[first_year, first_country]


def bucket_paths(starting: np.ndarray,
                 edges: Sequence[float] = DEFAULT_EDGES,
                 labels: Sequence[str] = BUCKET_LABELS) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Split paths into valuation buckets by quantile of the starting yield.

    Quantiles of the *drawn* distribution rather than fixed yield levels, so
    the buckets stay balanced however the sampler weights countries. Returns
    the bucket index per path and the yield cut-points that produced it.
    """
    finite = starting[np.isfinite(starting)]
    cuts = np.quantile(finite, list(edges)[1:-1]) if finite.size else np.array([])
    index = np.digitize(starting, cuts)
    index = np.where(np.isfinite(starting), index, -1)
    counts = [int((index == i).sum()) for i in range(len(labels))]
    return index, {"cuts": [float(c) for c in cuts],
                   "counts": counts,
                   "labels": list(labels),
                   "unassigned": int((index < 0).sum())}


def locate(value: float, reference: np.ndarray) -> float:
    """Where one yield sits in the panel's distribution, as a percentile."""
    finite = reference[np.isfinite(reference)]
    if not finite.size or not np.isfinite(value):
        return float("nan")
    return float((finite < value).mean() * 100.0)


def current_position(blended: np.ndarray, domestic: np.ndarray,
                     years: np.ndarray, countries: Sequence[str],
                     iso: str = "USA") -> Dict[str, Any]:
    """Where the panel's final year sits, so a reader can place themselves.

    Reported for one named market and for the blended portfolio, because the
    question a reader actually has is not "was the twentieth century "
    "expensive" but "how does where I am starting compare".
    """
    j = list(countries).index(iso) if iso in countries else 0
    column = domestic[:, j]
    last = int(np.flatnonzero(np.isfinite(column))[-1])
    return {
        "iso": iso,
        "year": int(np.asarray(years)[last]),
        "domestic_yield": float(column[last]),
        "domestic_percentile": locate(float(column[last]), domestic),
        "blended_yield": float(blended[last, j]),
        "blended_percentile": locate(float(blended[last, j]), blended),
        "panel_median_yield": float(np.nanmedian(domestic)),
    }


def outcome_subset(outcome: Any, mask: np.ndarray) -> Any:
    """One :class:`~src.lifecycle.LifecycleOutcome` restricted to some paths."""
    from . import lifecycle as lc
    return lc.LifecycleOutcome(
        strategy=outcome.strategy, label=outcome.label,
        consumption=outcome.consumption[mask], wealth=outcome.wealth[mask],
        portfolio_return=outcome.portfolio_return[mask],
        wealth_at_retirement=outcome.wealth_at_retirement[mask],
        bequest=outcome.bequest[mask], ruin=outcome.ruin[mask],
        ruin_age=outcome.ruin_age[mask],
        social_security=outcome.social_security[mask],
        career_average_income=outcome.career_average_income[mask])


def by_bucket(results: Mapping[str, Any], index: np.ndarray,
              labels: Sequence[str], cfg: Mapping[str, Any],
              spec: Any) -> pd.DataFrame:
    """Headline metrics for every strategy within every valuation bucket."""
    from . import utility as ut
    util = cfg["utility"]
    gammas = [float(g) for g in util["risk_aversions"]]
    rows: List[Dict[str, Any]] = []
    for i, label in enumerate(labels):
        mask = index == i
        if mask.sum() < 100:
            continue
        for key, outcome in results.items():
            sub = outcome_subset(outcome, mask)
            bundle = ut.bundle_from_outcome(sub, cfg, spec)
            row: Dict[str, Any] = {
                "bucket": label, "bucket_index": i,
                "n_paths": int(mask.sum()),
                "strategy": key, "label": outcome.label,
            }
            for gamma in gammas:
                row[f"cec_crra_gamma{gamma:g}"] = ut.crra_certainty_equivalent(
                    bundle, gamma, float(util["discount_factor"]),
                    float(util["bequest_weight"]),
                    bool(util["bequest_enabled"]))
            row["prob_ruin"] = float(sub.ruin.mean())
            row["median_retirement_consumption"] = float(
                np.median(sub.consumption[:, spec.retirement_slice]))
            row["p5_retirement_consumption"] = float(
                np.percentile(sub.consumption[:, spec.retirement_slice], 5))
            row["median_wealth_at_retirement"] = float(
                np.median(sub.wealth_at_retirement))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def advantage_by_bucket(frame: pd.DataFrame, challenger: str, incumbent: str,
                        column: str) -> pd.DataFrame:
    """The challenger's lead over the incumbent, bucket by bucket."""
    rows: List[Dict[str, Any]] = []
    for label, block in frame.groupby("bucket", sort=False):
        indexed = block.set_index("strategy")
        if challenger not in indexed.index or incumbent not in indexed.index:
            continue
        lead = (float(indexed.loc[challenger, column])
                / float(indexed.loc[incumbent, column]) - 1.0) * 100.0
        rows.append({
            "bucket": label,
            "n_paths": int(indexed.loc[challenger, "n_paths"]),
            "challenger_cec": float(indexed.loc[challenger, column]),
            "incumbent_cec": float(indexed.loc[incumbent, column]),
            "advantage_pct": lead,
            "challenger_ruin": float(indexed.loc[challenger, "prob_ruin"]),
            "incumbent_ruin": float(indexed.loc[incumbent, "prob_ruin"]),
        })
    return pd.DataFrame.from_records(rows)


def sleeve_comparison(domestic: np.ndarray, starting_mean: np.ndarray,
                      paths_blended_median: np.ndarray,
                      edges: Sequence[float], labels: Sequence[str],
                      ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Does taking the sleeve as a median rather than a mean change anything?

    The mean is the right answer on construction grounds -- the leg holds
    equal money in each market, and an equally weighted portfolio's dividend
    yield is the mean of its constituents' -- but "right on construction
    grounds" is an argument, not a measurement. This measures it: how far the
    two rankings diverge, and how many lifetimes actually land in a different
    valuation bucket because of the choice.
    """
    mean_index, mean_meta = bucket_paths(starting_mean, edges, labels)
    median_index, median_meta = bucket_paths(paths_blended_median, edges,
                                             labels)
    both = (mean_index >= 0) & (median_index >= 0)
    agree = float((mean_index[both] == median_index[both]).mean() * 100.0) \
        if both.any() else float("nan")
    finite = np.isfinite(starting_mean) & np.isfinite(paths_blended_median)
    correlation = float(np.corrcoef(starting_mean[finite],
                                    paths_blended_median[finite])[0, 1]) \
        if finite.sum() > 2 else float("nan")
    rows = [{
        "sleeve": "mean (used)",
        "cut_low": mean_meta["cuts"][0] if mean_meta["cuts"] else float("nan"),
        "cut_high": mean_meta["cuts"][-1] if mean_meta["cuts"] else float("nan"),
        **{f"n_{lab}": n for lab, n in zip(labels, mean_meta["counts"])},
    }, {
        "sleeve": "median (check)",
        "cut_low": median_meta["cuts"][0] if median_meta["cuts"] else float("nan"),
        "cut_high": median_meta["cuts"][-1] if median_meta["cuts"] else float("nan"),
        **{f"n_{lab}": n for lab, n in zip(labels, median_meta["counts"])},
    }]
    return pd.DataFrame.from_records(rows), {
        "agreement_pct": agree,
        "correlation": correlation,
        "reassigned": int((~(mean_index == median_index) & both).sum()),
        "n_compared": int(both.sum()),
    }
