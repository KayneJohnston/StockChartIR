"""Squeezing every observed series out of the primary sources.

`docs/14` found that 22 of the 38 countries have entirely simulated equity,
bond and bill returns, and that this reaches 38% of an observed investor's
international leg. The obvious remedy is more real data. This module is what
came of trying to get it.

**What is reachable.** Outbound access from this environment is governed by an
egress policy that denies the Jordà–Schularick–Taylor host itself, along with
FRED, the OECD, the World Bank and every other bulk macro-data host tested.
The Python package index is reachable but carries only API clients for those
same blocked hosts, not bundled data. So no new *external* source could be
obtained, and guessing at redistributed mirrors would have produced exactly
the unverifiable provenance `docs/14` exists to warn about.

**What was still on the table.** The primary files already in the repository
carry more than the pipeline was reading:

* Canada and Ireland appear in the Jordà–Schularick–Taylor macro file with
  long-term bond yields, short-term rates and consumer prices, but no return
  series -- which is why they were simulated. A bond return can be built from
  a yield and a duration assumption, and a bill return is a short rate. Both
  are then observed rather than generated.
* The Clio-Infra bond-yield file covers 42 countries, of which New Zealand and
  Austria carry enough history to do the same.
* The macro file also carries housing total returns for all sixteen observed
  countries -- an entire asset class, fully empirical, that nothing in the
  project reads.

None of that closes the equity gap, which is the one that matters most, and
this module does not pretend otherwise. What it does is stop simulating the
series that do not have to be simulated, and record precisely which cells are
which so `docs/14` can measure the result instead of asserting it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Countries carrying interest rates but no return series in the macro file.
#: Both are in the eighteen-country macro database and absent from the
#: sixteen-country return database, which is a property of the published
#: source rather than of this pipeline.
JST_RATE_ONLY: Tuple[str, ...] = ("CAN", "IRL")

#: Clio-Infra column -> ISO, for countries with enough bond-yield history to
#: build a return series. The threshold is deliberately conservative: below
#: about forty years a country contributes almost no admissible long blocks.
CLIO_BOND_COUNTRIES: Mapping[str, str] = {
    "New Zealand": "NZL",
    "Austria": "AUT",
}

MIN_YIELD_YEARS = 40


def bond_return_from_yield(yields: np.ndarray, duration: float) -> np.ndarray:
    """Nominal total return on a constant-maturity bond, from its yield alone.

    ``r_t = y_{t-1} + D · (y_{t-1} − y_t)``: one year of carry at last year's
    yield, plus the capital gain a duration-``D`` bond makes when the yield
    falls. It is the standard first-order approximation and it ignores
    convexity, which matters only for large yield moves on long durations.

    The first year is undefined -- there is no previous yield to earn carry at
    -- and is returned as NaN rather than filled, so a country's series starts
    where its data start.
    """
    yields = np.asarray(yields, dtype=float)
    previous = np.roll(yields, 1)
    previous[0] = np.nan
    return previous + float(duration) * (previous - yields)


def _real(nominal: np.ndarray, inflation: np.ndarray) -> np.ndarray:
    """Deflate a nominal return by realised inflation."""
    nominal = np.asarray(nominal, dtype=float)
    inflation = np.asarray(inflation, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        real = (1.0 + nominal) / (1.0 + inflation) - 1.0
    return np.where(np.isfinite(real), real, np.nan)


def _aligned(frame: pd.DataFrame, column: str, years: np.ndarray) -> np.ndarray:
    """Pull one column onto the panel's calendar, NaN where absent."""
    if column not in frame.columns:
        return np.full(years.size, np.nan)
    series = frame.set_index("year")[column]
    return series.reindex(years).to_numpy(dtype=float)


def rates_from_jst(jst: pd.DataFrame, iso: str, years: np.ndarray,
                   duration: float) -> Dict[str, np.ndarray]:
    """Real bond and bill returns for a country that has rates but no returns.

    Returns a mapping with ``bond`` and ``bill`` arrays on the panel calendar,
    NaN wherever the underlying rate or price index is missing. Inflation is
    recomputed here from the country's own consumer price index rather than
    taken from elsewhere, so the deflator is the country's own.
    """
    block = jst[jst["iso"] == iso].sort_values("year")
    if block.empty:
        return {"bond": np.full(years.size, np.nan),
                "bill": np.full(years.size, np.nan)}
    cpi = _aligned(block, "cpi", years)
    with np.errstate(invalid="ignore", divide="ignore"):
        inflation = cpi / np.roll(cpi, 1) - 1.0
    inflation[0] = np.nan

    long_rate = _aligned(block, "ltrate", years) / 100.0
    short_rate = _aligned(block, "stir", years) / 100.0
    return {
        "bond": _real(bond_return_from_yield(long_rate, duration), inflation),
        "bill": _real(np.roll(short_rate, 1), inflation),
    }


def bond_from_clio(clio: pd.DataFrame, column: str, years: np.ndarray,
                   inflation: np.ndarray, duration: float) -> np.ndarray:
    """Real bond return for a country with a Clio-Infra long-term yield series.

    ``clio`` is the year-indexed wide frame :func:`src.data_loader.load_clio_wide`
    returns, whose loader has *already* rescaled percent to decimals -- so the
    yields arrive here as ``0.04`` for four percent and must not be divided
    again. ``inflation`` is supplied by the caller because Clio's yield file
    carries no price index; the panel already sources that country's inflation
    from Clio's own CPI file, so the two come from the same project.
    """
    if column not in clio.columns:
        return np.full(years.size, np.nan)
    yields = clio[column].reindex(years).to_numpy(dtype=float)
    return _real(bond_return_from_yield(yields, duration), inflation)


def housing_returns(jst: pd.DataFrame, isos: Sequence[str],
                    years: np.ndarray) -> np.ndarray:
    """Real housing total returns, ``(T, C)``, for the observed countries.

    Housing is the fourth asset of the "Rate of Return on Everything" project
    and nothing in this pipeline reads it. It is extracted here so that the
    data are captured, audited and available; it is deliberately *not* added
    to the investable asset set, because appraisal smoothing means a raw
    housing series is not comparable with a traded one until it has been
    de-smoothed.
    """
    if "housing_tr" not in jst.columns:
        raise KeyError(
            "the frame has no 'housing_tr' column, so this would return "
            "nothing but NaN; pass the workbook frame from "
            "`data_loader.load_jst`, which retains it"
        )
    out = np.full((years.size, len(isos)), np.nan)
    for j, iso in enumerate(isos):
        block = jst[jst["iso"] == iso].sort_values("year")
        if block.empty:
            continue
        cpi = _aligned(block, "cpi", years)
        with np.errstate(invalid="ignore", divide="ignore"):
            inflation = cpi / np.roll(cpi, 1) - 1.0
        inflation[0] = np.nan
        out[:, j] = _real(_aligned(block, "housing_tr", years), inflation)
    return out


def first_order_autocorrelation(series: np.ndarray) -> float:
    """Lag-one autocorrelation of a series, ignoring gaps."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def desmooth(series: np.ndarray, rho: float | None = None) -> np.ndarray:
    """Undo first-order appraisal smoothing, Geltner-style.

    An appraisal-based index reports ``r_t = (1 − a)·r*_t + a·r_{t-1}``, where
    ``r*`` is the return a traded market would have shown. Inverting that gives
    ``r*_t = (r_t − a·r_{t-1}) / (1 − a)``, which restores the volatility the
    smoothing removed while leaving the mean essentially untouched.

    ``rho`` defaults to the series' own lag-one autocorrelation. A series that
    is not positively autocorrelated is returned unchanged, since there is no
    smoothing to undo.
    """
    values = np.asarray(series, dtype=float)
    if rho is None:
        rho = first_order_autocorrelation(values)
    if not np.isfinite(rho) or rho <= 0.0:
        return values.copy()
    out = np.full_like(values, np.nan)
    out[1:] = (values[1:] - rho * values[:-1]) / (1.0 - rho)
    return out


def coverage_report(observed: Mapping[str, np.ndarray],
                    iso: str) -> Dict[str, Any]:
    """How many years each recovered series actually covers."""
    row: Dict[str, Any] = {"iso": iso}
    for key, values in observed.items():
        row[f"{key}_years"] = int(np.isfinite(values).sum())
    return row


def summarise(recovered: Mapping[str, Mapping[str, np.ndarray]]
              ) -> pd.DataFrame:
    """One row per country describing what was recovered and from where."""
    rows = []
    for iso, series in recovered.items():
        row = coverage_report(series, iso)
        row["source"] = ("Jordà–Schularick–Taylor rates"
                         if iso in JST_RATE_ONLY else "Clio-Infra bond yields")
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values("iso") \
        .reset_index(drop=True) if rows else pd.DataFrame()
