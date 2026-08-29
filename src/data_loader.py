"""Panel ingestion and real-return construction.

This module turns primary long-horizon macro-financial sources into the
calendar-aligned real return panel consumed by :mod:`src.bootstrap`.

The panel has five core series per country-year, all expressed in the
*domestic investor's real consumption units*:

``dom_eq``   real total return on the domestic equity market
``intl_eq``  real total return on an equally weighted portfolio of the
             *other* developed markets, translated into domestic currency
             and deflated by domestic CPI
``bond``     real total return on domestic long-term government bonds
``bill``     real total return on domestic short-term government bills
``inflation``domestic CPI inflation

Provenance is tracked per country with a ``tier`` label:

``A``  every series is empirical (Jorda-Schularick-Taylor / JKKST)
``B``  inflation (and, where available, bond yields) are empirical, while
       equity/bond/bill total returns are *calibrated proxies* generated
       from a factor model estimated on the Tier-A cross-section.

See ``docs/01_country_dataset_and_sources.md`` for the full lineage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from . import observed as obs

LOGGER = logging.getLogger(__name__)

#: Core series carried by the panel, in canonical order.
CORE_SERIES: Tuple[str, ...] = ("dom_eq", "intl_eq", "bond", "bill", "inflation")

#: Gross real returns are floored here so that log utilities stay finite.
#: Only Germany 1923 (a total real wipeout of nominal bonds) binds.
GROSS_RETURN_FLOOR: float = 1.0e-6

#: The 16 countries for which the JST "Rate of Return on Everything" file
#: carries complete equity, bond and bill total-return histories.
TIER_A_ISO: Tuple[str, ...] = (
    "AUS", "BEL", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA",
    "GBR", "ITA", "JPN", "NLD", "NOR", "PRT", "SWE", "USA",
)

#: Countries that were once carried in this panel with *generated* returns and
#: have since been removed. They are recorded rather than deleted outright so
#: the removal is documented and cannot be undone by accident.
#:
#: The panel formerly ran to thirty-eight developed markets. Twenty-two of them
#: had no equity, bond or bill return series in any source available here, and
#: their returns were drawn from a single-factor model fitted to a randomly
#: assigned observed donor. That is a simulation, not a measurement, and no
#: result should rest on it -- so the whole block is gone and the panel is now
#: exactly the countries whose returns were recorded.
#:
#: Four of them (Austria, Canada, Ireland, New Zealand) do have real
#: interest-rate histories, and `src.observed` still recovers them for the
#: provenance audit. They cannot re-enter the panel: a lifecycle investor needs
#: a domestic *equity* return, no source available here carries one for them,
#: and inventing it is the practice this removal exists to end.
REMOVED_SIMULATED: Tuple[str, ...] = (
    "AUT", "CAN", "HRV", "CYP", "CZE", "EST", "GRC", "HKG", "ISL", "IRL",
    "ISR", "KOR", "LVA", "LTU", "LUX", "MLT", "NZL", "POL", "SGP", "SVK",
    "SVN", "TWN",
)

#: Display names for every country this project names, whether or not it is in
#: the panel -- the audit reports on the removed ones too.
REMOVED_NAMES: Mapping[str, str] = {
    "AUT": "Austria", "CAN": "Canada", "HRV": "Croatia", "CYP": "Cyprus",
    "CZE": "Czech Republic", "EST": "Estonia", "GRC": "Greece",
    "HKG": "Hong Kong SAR", "ISL": "Iceland", "IRL": "Ireland",
    "ISR": "Israel", "KOR": "Korea", "LVA": "Latvia", "LTU": "Lithuania",
    "LUX": "Luxembourg", "MLT": "Malta", "NZL": "New Zealand",
    "POL": "Poland", "SGP": "Singapore", "SVK": "Slovakia",
    "SVN": "Slovenia", "TWN": "Taiwan",
}

ISO_TO_NAME: Dict[str, str] = {
    "AUS": "Australia", "BEL": "Belgium", "CHE": "Switzerland",
    "DEU": "Germany", "DNK": "Denmark", "ESP": "Spain", "FIN": "Finland",
    "FRA": "France", "GBR": "United Kingdom", "ITA": "Italy",
    "JPN": "Japan", "NLD": "Netherlands", "NOR": "Norway",
    "PRT": "Portugal", "SWE": "Sweden", "USA": "United States",
    **REMOVED_NAMES,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Read the YAML run configuration into a plain dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):  # pragma: no cover - defensive
        raise ValueError(f"config at {path} did not parse to a mapping")
    return cfg


# ---------------------------------------------------------------------------
# Panel container
# ---------------------------------------------------------------------------
#: Provenance tiers, derived from the per-cell observation masks rather than
#: asserted, so a label can never drift from the data it describes.
TIER_LABELS: Mapping[str, str] = {
    "A": "observed returns",
    "B": "partly observed returns",
    "C": "simulated returns",
}

#: The three investable return series a tier describes. Inflation is excluded:
#: it is empirical for almost every country and would flatter the label.
TIERED_SERIES: Tuple[str, ...] = ("dom_eq", "bond", "bill")


@dataclasses.dataclass(frozen=True)
class Panel:
    """Calendar-aligned real return panel.

    All matrices are ``(T, C)`` with ``T`` calendar years and ``C``
    countries, indexed consistently by :attr:`years` and :attr:`countries`.
    Entries are ``np.nan`` where the country has no usable observation.
    """

    years: np.ndarray
    countries: Tuple[str, ...]
    tier: Tuple[str, ...]
    dom_eq: np.ndarray
    intl_eq: np.ndarray
    bond: np.ndarray
    bill: np.ndarray
    inflation: np.ndarray
    real_exchange_rate: np.ndarray
    available: np.ndarray
    name: str = "panel"
    provenance: Tuple[str, ...] = ()
    #: ``series -> (T, C)`` boolean, True where the cell was *observed* rather
    #: than generated. May be empty, in which case :meth:`observed_mask` falls
    #: back to the country tier -- see there for what that means.
    observed: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)

    # -- basic geometry -----------------------------------------------------
    @property
    def n_years(self) -> int:
        return int(self.years.shape[0])

    @property
    def n_countries(self) -> int:
        return len(self.countries)

    def series(self, key: str) -> np.ndarray:
        """Return the ``(T, C)`` matrix for one of :data:`CORE_SERIES`."""
        if key not in CORE_SERIES:
            raise KeyError(f"unknown series {key!r}; expected one of {CORE_SERIES}")
        return getattr(self, key)

    def stacked(self) -> np.ndarray:
        """Return a ``(T, C, 5)`` tensor ordered as :data:`CORE_SERIES`."""
        return np.stack([self.series(k) for k in CORE_SERIES], axis=-1)

    def country_index(self, iso: str) -> int:
        return self.countries.index(iso)

    def tier_mask(self, tier: str) -> np.ndarray:
        return np.array([t == tier for t in self.tier], dtype=bool)

    def observed_mask(self, key: str) -> np.ndarray:
        """``(T, C)`` True where ``key`` was observed rather than generated.

        A panel carrying no explicit mask falls back to its country labels,
        which are the coarser statement of the same thing: a Tier-A country's
        returns are observed and no other country's are. That default is what
        makes the empirical-only panel audit correctly without special-casing,
        and it keeps the mask from ever contradicting the tier beside it.

        Inflation is exempt from the fallback because it is empirical for
        nearly every country regardless of tier; where it is not, the panel
        that generated it records the fact per cell.
        """
        if key not in CORE_SERIES:
            raise KeyError(f"unknown series {key!r}; expected one of {CORE_SERIES}")
        if self.observed:
            return np.asarray(self.observed[key], dtype=bool) & self.available
        if key not in TIERED_SERIES:
            return self.available.copy()
        return self.available & self.tier_mask("A")[np.newaxis, :]

    def subset(self, isos: Sequence[str], name: str | None = None) -> "Panel":
        """Return a new panel restricted to ``isos`` (international legs are
        *not* recomputed; use :func:`build_panel` for that)."""
        idx = [self.country_index(i) for i in isos]
        return dataclasses.replace(
            self,
            countries=tuple(isos),
            tier=tuple(self.tier[i] for i in idx),
            dom_eq=self.dom_eq[:, idx],
            intl_eq=self.intl_eq[:, idx],
            bond=self.bond[:, idx],
            bill=self.bill[:, idx],
            inflation=self.inflation[:, idx],
            real_exchange_rate=self.real_exchange_rate[:, idx],
            available=self.available[:, idx],
            name=name or self.name,
            provenance=tuple(self.provenance[i] for i in idx)
            if self.provenance else (),
            observed={k: v[:, idx] for k, v in self.observed.items()},
        )

    # -- persistence --------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Flatten the panel to tidy long format."""
        n_t, n_c = self.dom_eq.shape
        frame = pd.DataFrame(
            {
                "year": np.repeat(self.years, n_c),
                "iso": np.tile(np.array(self.countries, dtype=object), n_t),
                "tier": np.tile(np.array(self.tier, dtype=object), n_t),
                "dom_eq": self.dom_eq.reshape(-1),
                "intl_eq": self.intl_eq.reshape(-1),
                "bond": self.bond.reshape(-1),
                "bill": self.bill.reshape(-1),
                "inflation": self.inflation.reshape(-1),
                "real_exchange_rate": self.real_exchange_rate.reshape(-1),
                "available": self.available.reshape(-1),
            }
        )
        if self.provenance:
            lookup = dict(zip(self.countries, self.provenance))
            frame["provenance"] = frame["iso"].map(lookup)
        frame["country"] = frame["iso"].map(ISO_TO_NAME).fillna(frame["iso"])
        return frame

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            years=self.years,
            countries=np.array(self.countries),
            tier=np.array(self.tier),
            dom_eq=self.dom_eq,
            intl_eq=self.intl_eq,
            bond=self.bond,
            bill=self.bill,
            inflation=self.inflation,
            real_exchange_rate=self.real_exchange_rate,
            available=self.available,
            name=np.array(self.name),
            provenance=np.array(self.provenance if self.provenance
                                else [''] * len(self.countries)),
            # Per-cell provenance, one array per series under an "observed_"
            # prefix. Dropping it here would silently downgrade a reloaded
            # panel to the coarser country-level fallback, which is exactly
            # the confusion the masks exist to remove.
            **{f"observed_{k}": np.asarray(v, dtype=bool)
               for k, v in self.observed.items()},
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Panel":
        with np.load(path, allow_pickle=False) as blob:
            return cls(
                years=blob["years"],
                countries=tuple(str(c) for c in blob["countries"]),
                tier=tuple(str(t) for t in blob["tier"]),
                dom_eq=blob["dom_eq"],
                intl_eq=blob["intl_eq"],
                bond=blob["bond"],
                bill=blob["bill"],
                inflation=blob["inflation"],
                real_exchange_rate=blob["real_exchange_rate"],
                available=blob["available"],
                observed={key[len("observed_"):]: blob[key]
                          for key in blob.files
                          if key.startswith("observed_")},
                name=str(blob["name"]),
                provenance=tuple(str(v) for v in blob["provenance"])
                if "provenance" in blob else (),
            )

    def fingerprint(self) -> str:
        """Stable content hash, used to invalidate bootstrap caches."""
        digest = hashlib.sha256()
        digest.update(self.name.encode())
        digest.update(np.ascontiguousarray(self.years).tobytes())
        digest.update("|".join(self.countries).encode())
        for key in CORE_SERIES:
            arr = np.nan_to_num(self.series(key), nan=-9.99e9)
            digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Primary source ingestion
# ---------------------------------------------------------------------------
#: Parsed workbooks, keyed by ``(path, sheet)``.  The hedging sweep rebuilds
#: the panel dozens of times with different hedge ratios; re-parsing a 1.4 MB
#: workbook each time dominated the runtime and the file cannot change during
#: a run.
_WORKBOOK_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def load_jst(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Load the Jorda-Schularick-Taylor Macrohistory workbook.

    Returns the raw country-year frame restricted to the columns this
    project uses.  Nominal returns are decimal (``0.0723`` = 7.23%);
    ``cpi`` is an index; ``xrusd`` is *local currency units per USD*.
    """
    data_cfg = cfg["data"]
    key = (str(data_cfg["jst_workbook"]), str(data_cfg["jst_sheet"]))
    if key not in _WORKBOOK_CACHE:
        _WORKBOOK_CACHE[key] = pd.read_excel(key[0], sheet_name=key[1])
    frame = _WORKBOOK_CACHE[key].copy()
    keep = ["year", "country", "iso", "cpi", "xrusd", "eq_tr", "bond_tr",
            "bill_rate", "eq_dp", "eq_capgain", "bond_rate", "stir", "ltrate",
            "housing_tr", "wage"]
    missing = [c for c in keep if c not in frame.columns]
    if missing:
        raise ValueError(f"JST workbook is missing expected columns: {missing}")
    frame = frame.loc[:, keep].copy()
    frame["year"] = frame["year"].astype(int)
    for col in keep[3:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values(["iso", "year"]).reset_index(drop=True)


def load_clio_wide(path: str | Path, scale: float = 0.01) -> pd.DataFrame:
    """Load a Clio-Infra wide CSV (``year`` + one column per country name).

    Values are stored in percent; ``scale`` converts them to decimals.
    """
    frame = pd.read_csv(path)
    frame = frame.rename(columns={frame.columns[0]: "year"})
    frame["year"] = frame["year"].astype(int)
    value_cols = [c for c in frame.columns if c != "year"]
    frame[value_cols] = frame[value_cols].apply(pd.to_numeric, errors="coerce") * scale
    return frame.set_index("year")


# ---------------------------------------------------------------------------
# Real return construction
# ---------------------------------------------------------------------------
def _pct_change(series: pd.Series) -> pd.Series:
    """Simple period-over-period change without any forward filling."""
    return series / series.shift(1) - 1.0


def deflate(nominal_return: np.ndarray, inflation: np.ndarray) -> np.ndarray:
    """Convert a nominal return to a real return.

    ``(1 + R_nominal) / (1 + pi) - 1``.  The gross real return is floored at
    :data:`GROSS_RETURN_FLOOR` so that hyperinflation wipe-outs stay
    representable in logs.
    """
    gross = (1.0 + np.asarray(nominal_return, dtype=float)) / (
        1.0 + np.asarray(inflation, dtype=float)
    )
    gross = np.where(np.isfinite(gross), gross, np.nan)
    gross = np.where(np.isnan(gross), np.nan, np.maximum(gross, GROSS_RETURN_FLOOR))
    return gross - 1.0


def add_real_returns(jst: pd.DataFrame) -> pd.DataFrame:
    """Attach inflation and real domestic asset returns to the JST frame."""
    out = jst.sort_values(["iso", "year"]).copy()
    out["inflation"] = out.groupby("iso", sort=False)["cpi"].transform(_pct_change)
    out["dom_eq"] = deflate(out["eq_tr"].to_numpy(), out["inflation"].to_numpy())
    out["bond"] = deflate(out["bond_tr"].to_numpy(), out["inflation"].to_numpy())
    out["bill"] = deflate(out["bill_rate"].to_numpy(), out["inflation"].to_numpy())
    return out


def _usd_gross_equity(jst: pd.DataFrame) -> pd.DataFrame:
    """Gross *nominal USD* return on each country's equity market.

    ``(1 + eq_tr_j) * (xrusd_{j,t-1} / xrusd_{j,t})`` -- the local total
    return converted at the change in the USD exchange rate.
    """
    out = jst.sort_values(["iso", "year"]).copy()
    prev_fx = out.groupby("iso", sort=False)["xrusd"].shift(1)
    fx_ratio = prev_fx / out["xrusd"]
    gross = (1.0 + out["eq_tr"]) * fx_ratio
    out["usd_gross_eq"] = np.where(np.isfinite(gross) & (gross > 0), gross, np.nan)
    out["fx_gain"] = np.where(np.isfinite(fx_ratio) & (fx_ratio > 0), fx_ratio, np.nan)
    return out


def build_international_leg(
    wide_usd_gross: np.ndarray,
    wide_fx_gain: np.ndarray,
    wide_inflation: np.ndarray,
    weighting: str = "equal",
    winsor_pct: float = 0.0,
) -> np.ndarray:
    """Real return on the *rest of the world* equity portfolio.

    Parameters
    ----------
    wide_usd_gross:
        ``(T, C)`` gross nominal USD equity returns.
    wide_fx_gain:
        ``(T, C)`` gross ``xrusd_{t-1}/xrusd_{t}`` for each country, i.e. the
        USD appreciation of the local currency.
    wide_inflation:
        ``(T, C)`` domestic CPI inflation.
    weighting:
        ``"equal"`` is the only weighting currently implemented; it averages
        *gross* USD returns across the available foreign markets.
    winsor_pct:
        If positive, the resulting real returns are winsorised at the pooled
        ``winsor_pct`` and ``100 - winsor_pct`` percentiles.  This targets a
        handful of war-time country-years (Italy 1942, France 1946, ...) in
        which administered exchange rates -- not tradable prices -- drive the
        currency conversion.  Every affected observation is listed in
        docs/01.  Set to ``0`` to disable.

    Returns
    -------
    ``(T, C)`` real return, in each country's own consumption units, of an
    equally weighted portfolio of the *other* countries' equity markets.
    """
    if weighting != "equal":
        raise NotImplementedError(f"weighting {weighting!r} is not implemented")

    valid = np.isfinite(wide_usd_gross)
    filled = np.where(valid, wide_usd_gross, 0.0)
    row_sum = filled.sum(axis=1, keepdims=True)
    row_count = valid.sum(axis=1, keepdims=True)

    # Leave-one-out mean of gross USD returns across the *other* markets.
    others_sum = row_sum - filled
    others_count = row_count - valid.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        others_mean = np.where(others_count > 0, others_sum / others_count, np.nan)

    # Translate USD -> local currency, then deflate by domestic inflation.
    with np.errstate(invalid="ignore", divide="ignore"):
        local_gross = others_mean / wide_fx_gain
        real_gross = local_gross / (1.0 + wide_inflation)
    real_gross = np.where(np.isfinite(real_gross), real_gross, np.nan)
    real_gross = np.where(np.isnan(real_gross), np.nan,
                          np.maximum(real_gross, GROSS_RETURN_FLOOR))
    result = real_gross - 1.0
    if winsor_pct and winsor_pct > 0:
        finite = result[np.isfinite(result)]
        if finite.size:
            lo, hi = np.percentile(finite, [winsor_pct, 100.0 - winsor_pct])
            result = np.where(np.isfinite(result), np.clip(result, lo, hi), result)
    return result


def build_hedged_international_leg(
    wide_eq_tr: np.ndarray,
    wide_bill_nominal: np.ndarray,
    wide_inflation: np.ndarray,
    hedge_cost: float = 0.0,
    winsor_pct: float = 0.0,
) -> np.ndarray:
    """Real return on a *currency-hedged* rest-of-the-world equity portfolio.

    Under covered interest parity, fully hedging a foreign equity position
    back to the domestic currency converts the foreign local-currency return
    at the forward rate, which pays the interest-rate differential:

    ```
    hedged_gross_ij = (1 + eq_tr_j) * (1 + r_i) / (1 + r_j)
    ```

    with ``r`` the nominal short rate.  The hedged investor therefore earns
    the foreign *asset* return plus the domestic short rate and gives up the
    foreign one -- they hold the equity risk and shed the currency risk.
    Averaging over the available foreign markets, charging an annual
    ``hedge_cost`` and deflating by domestic inflation gives the series
    returned here.

    Two honest caveats, both stated in docs/08.  CIP is an identity only when
    it holds: it broke down during both world wars, under capital controls,
    and again after 2008, and this construction imposes it throughout.  And a
    hedge in practice is rolled short-dated forwards with basis and margin
    risk, which ``hedge_cost`` represents as a flat drag rather than
    modelling directly -- which is exactly why the cost is swept rather than
    assumed in docs/08.
    """
    gross_local = 1.0 + wide_eq_tr
    short = 1.0 + wide_bill_nominal
    with np.errstate(invalid="ignore", divide="ignore"):
        adjusted = gross_local / short
    valid = np.isfinite(adjusted)
    filled = np.where(valid, adjusted, 0.0)

    row_sum = filled.sum(axis=1, keepdims=True)
    row_count = valid.sum(axis=1, keepdims=True)
    others_sum = row_sum - filled
    others_count = row_count - valid.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        others_mean = np.where(others_count > 0, others_sum / others_count,
                               np.nan)
        hedged_gross = others_mean * short * (1.0 - hedge_cost)
        real_gross = hedged_gross / (1.0 + wide_inflation)

    real_gross = np.where(np.isfinite(real_gross), real_gross, np.nan)
    real_gross = np.where(np.isnan(real_gross), np.nan,
                          np.maximum(real_gross, GROSS_RETURN_FLOOR))
    result = real_gross - 1.0
    if winsor_pct and winsor_pct > 0:
        finite = result[np.isfinite(result)]
        if finite.size:
            lo, hi = np.percentile(finite, [winsor_pct, 100.0 - winsor_pct])
            result = np.where(np.isfinite(result), np.clip(result, lo, hi),
                              result)
    return result


def blend_international_legs(unhedged: np.ndarray, hedged: np.ndarray,
                             hedge_ratio: float) -> np.ndarray:
    """Return on holding ``hedge_ratio`` of the international sleeve hedged.

    A linear blend of the two returns is exactly right for an investor who
    splits the sleeve between a hedged and an unhedged share class and
    rebalances annually -- which is the choice actually on offer.

    Where the hedged series cannot be computed (a country-year with no usable
    short rate) the unhedged value is carried through, so that varying the
    hedge ratio never changes which country-years are available and therefore
    never changes which blocks the bootstrap draws.  That keeps every hedge
    ratio a paired comparison on identical history.
    """
    ratio = float(np.clip(hedge_ratio, 0.0, 1.0))
    if ratio == 0.0:
        return unhedged
    usable = np.isfinite(hedged)
    return np.where(usable, (1.0 - ratio) * unhedged + ratio * hedged,
                    unhedged)


def build_real_exchange_rate(
    wide_cpi: np.ndarray, wide_xrusd: np.ndarray
) -> np.ndarray:
    """Real exchange rate of each country against the world basket.

    The price level of country ``i`` expressed in USD is ``cpi_i / xrusd_i``.
    CPI base years differ across countries, so only *changes* are
    comparable: the index below cumulates
    ``dlog(cpi_i / xrusd_i) - dlog(world basket)`` from 1.0 at each
    country's first observation, where the world basket is the equally
    weighted geometric mean across the countries available in that year.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        log_p_usd = np.log(wide_cpi / wide_xrusd)
    log_p_usd = np.where(np.isfinite(log_p_usd), log_p_usd, np.nan)

    d_log = np.full_like(log_p_usd, np.nan)
    d_log[1:] = log_p_usd[1:] - log_p_usd[:-1]

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        world = np.nanmean(np.where(np.isfinite(d_log), d_log, np.nan), axis=1)
    world = np.where(np.isfinite(world), world, 0.0)

    rel = d_log - world[:, None]
    rel_filled = np.where(np.isfinite(rel), rel, 0.0)
    index = np.exp(np.cumsum(rel_filled, axis=0))
    index = np.where(np.isfinite(d_log) | np.isfinite(log_p_usd), index, np.nan)
    # Renormalise each country so its first valid observation equals 1.0.
    for col in range(index.shape[1]):
        finite = np.flatnonzero(np.isfinite(index[:, col]))
        if finite.size:
            index[:, col] = index[:, col] / index[finite[0], col]
    return index


# ---------------------------------------------------------------------------
# Tier A: fully empirical JST panel
# ---------------------------------------------------------------------------
def _pivot(frame: pd.DataFrame, column: str, years: np.ndarray,
           isos: Sequence[str]) -> np.ndarray:
    """Pivot a tidy frame to a ``(len(years), len(isos))`` matrix."""
    wide = frame.pivot(index="year", columns="iso", values=column)
    wide = wide.reindex(index=years, columns=list(isos))
    return wide.to_numpy(dtype=float)


def build_tier_a(cfg: Mapping[str, Any], hedge_ratio: float = 0.0,
                 hedge_cost: float = 0.0) -> Panel:
    """Construct the fully empirical 16-country JST panel.

    ``hedge_ratio`` blends a currency-hedged international equity leg into
    ``intl_eq`` (see :func:`build_hedged_international_leg`).  It deliberately
    does not affect ``available``, so every hedge ratio draws exactly the same
    blocks and the comparison in docs/08 is paired on identical history.
    """
    data_cfg = cfg["data"]
    jst = load_jst(cfg)
    jst = add_real_returns(jst)
    jst = _usd_gross_equity(jst)

    years = np.arange(int(data_cfg["start_year"]), int(data_cfg["end_year"]) + 1)
    isos = list(TIER_A_ISO)
    window = jst[jst["year"].between(years[0] - 1, years[-1])]

    # The international leg must be built on *all* JST countries that have
    # equity data, then read off for the Tier-A columns.
    all_isos = sorted(window.loc[window["usd_gross_eq"].notna(), "iso"].unique())
    usd_gross = _pivot(window, "usd_gross_eq", years, all_isos)
    fx_gain = _pivot(window, "fx_gain", years, all_isos)
    infl_all = _pivot(window, "inflation", years, all_isos)
    winsor = float(data_cfg.get("international_winsor_pct", 0.0))
    intl_all = build_international_leg(
        usd_gross, fx_gain, infl_all,
        weighting=data_cfg["international_weighting"],
        winsor_pct=winsor,
    )
    if hedge_ratio > 0.0:
        hedged_all = build_hedged_international_leg(
            _pivot(window, "eq_tr", years, all_isos),
            _pivot(window, "bill_rate", years, all_isos),
            infl_all, hedge_cost=hedge_cost, winsor_pct=winsor)
        intl_all = blend_international_legs(intl_all, hedged_all, hedge_ratio)
    intl_lookup = {iso: intl_all[:, k] for k, iso in enumerate(all_isos)}

    dom_eq = _pivot(window, "dom_eq", years, isos)
    bond = _pivot(window, "bond", years, isos)
    bill = _pivot(window, "bill", years, isos)
    inflation = _pivot(window, "inflation", years, isos)
    cpi = _pivot(window, "cpi", years, isos)
    xrusd = _pivot(window, "xrusd", years, isos)
    intl_eq = np.column_stack(
        [intl_lookup.get(iso, np.full(years.size, np.nan)) for iso in isos]
    )
    rer = build_real_exchange_rate(cpi, xrusd)

    available = (
        np.isfinite(dom_eq) & np.isfinite(intl_eq) & np.isfinite(bond)
        & np.isfinite(bill) & np.isfinite(inflation)
    )
    keep = available.sum(axis=0) >= int(data_cfg["min_observations"])
    if not keep.all():
        dropped = [iso for iso, k in zip(isos, keep) if not k]
        LOGGER.warning("dropping Tier-A countries with too few years: %s", dropped)
    idx = np.flatnonzero(keep)
    isos = [isos[i] for i in idx]

    provenance = tuple(
        f"JST/JKKST empirical {int(years[available[:, i]][0])}"
        f"-{int(years[available[:, i]][-1])}; inflation 100% empirical"
        if available[:, i].any() else "JST/JKKST empirical"
        for i in idx
    )
    return Panel(
        years=years,
        countries=tuple(isos),
        tier=tuple("A" for _ in isos),
        dom_eq=dom_eq[:, idx],
        intl_eq=intl_eq[:, idx],
        bond=bond[:, idx],
        bill=bill[:, idx],
        inflation=inflation[:, idx],
        real_exchange_rate=rer[:, idx],
        available=available[:, idx],
        name="observed",
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------
def derive_tiers(observed: Mapping[str, np.ndarray], available: np.ndarray,
                 base: Sequence[str]) -> List[str]:
    """Relabel each country by how much of its return history is observed.

    ``A`` every available cell of every return series is an observation;
    ``C`` none of them is; ``B`` in between -- a country whose interest rates
    survive in a source but whose equity market has to be generated.

    ``base`` supplies the label for a country the masks say nothing about, so
    a panel built without them keeps whatever it was already called.
    """
    labels: List[str] = []
    for i, fallback in enumerate(base):
        column = available[:, i]
        total = int(column.sum())
        if total == 0 or not observed:
            labels.append(str(fallback))
            continue
        seen = sum(int((np.asarray(observed[k], dtype=bool)[:, i] & column).sum())
                   for k in TIERED_SERIES if k in observed)
        span = total * sum(1 for k in TIERED_SERIES if k in observed)
        labels.append("A" if seen == span else "C" if seen == 0 else "B")
    return labels


def _fx_gain_from_levels(xrusd: np.ndarray) -> np.ndarray:
    gain = np.full_like(xrusd, np.nan)
    gain[1:] = xrusd[:-1] / xrusd[1:]
    return np.where(np.isfinite(gain) & (gain > 0), gain, np.nan)


def build_panel(cfg: Mapping[str, Any], mode: str | None = None,
                hedge_ratio: float | None = None,
                hedge_cost: float | None = None) -> Panel:
    """Build the country panel.

    Every country in it has a recorded equity, bond, bill and inflation
    history. There is no generated block and no ``mode`` that produces one:
    the panel used to carry twenty-two developed markets whose returns were
    drawn from a factor model, and those are gone. See
    :data:`REMOVED_SIMULATED`.

    Parameters
    ----------
    mode:
        Accepted for backward compatibility and for the robustness harness,
        which names panels by string. ``"jst16"`` and ``"observed"`` both
        mean the only panel there is; ``"dev38"`` names the simulated panel
        and is refused with an explanation rather than silently aliased, so a
        stale config cannot quietly resurrect it.
    """
    mode = mode or cfg["bootstrap"]["panel"]
    if mode == "dev38":
        raise ValueError(
            "panel 'dev38' no longer exists: its twenty-two extra countries "
            "had factor-model returns rather than recorded ones and were "
            "removed. Use 'observed' (or its old name 'jst16')."
        )
    if mode not in ("jst16", "observed"):
        raise ValueError(
            f"unknown panel mode {mode!r}; expected 'observed' or 'jst16'")
    data_cfg = cfg["data"]
    hedge_ratio = float(data_cfg.get("hedge_ratio", 0.0)
                        if hedge_ratio is None else hedge_ratio)
    hedge_cost = float(data_cfg.get("hedge_cost", 0.0)
                       if hedge_cost is None else hedge_cost)
    return build_tier_a(cfg, hedge_ratio, hedge_cost)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _autocorr(values: np.ndarray, lag: int = 1) -> float:
    values = values[np.isfinite(values)]
    if values.size <= lag + 2:
        return float("nan")
    a, b = values[:-lag], values[lag:]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def summary_statistics(panel: Panel) -> pd.DataFrame:
    """Per-country, per-series moments used in docs/01."""
    records: List[Dict[str, Any]] = []
    for c, iso in enumerate(panel.countries):
        mask = panel.available[:, c]
        for key in CORE_SERIES:
            values = panel.series(key)[:, c][mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            log_gross = np.log1p(np.clip(values, GROSS_RETURN_FLOOR - 1.0, None))
            records.append({
                "iso": iso,
                "country": ISO_TO_NAME.get(iso, iso),
                "tier": panel.tier[c],
                "series": key,
                "n_years": int(values.size),
                "first_year": int(panel.years[mask][0]),
                "last_year": int(panel.years[mask][-1]),
                "mean": float(values.mean()),
                "geometric_mean": float(np.expm1(log_gross.mean())),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
                "skew": float(pd.Series(values).skew()),
                "kurtosis": float(pd.Series(values).kurtosis()),
                "ar1": _autocorr(values, 1),
                "ar2": _autocorr(values, 2),
            })
    return pd.DataFrame.from_records(records)


def coverage_matrix(panel: Panel, decade: bool = True) -> pd.DataFrame:
    """Country x year (or decade) coverage of usable observations.

    By decade the value is the *share* of that decade's years with complete
    data, not the count. The panel's last bucket is usually a partial decade --
    2020 on its own, say -- and a raw count there tops out at one against a
    scale of ten, which reads as an absence of data rather than as a bucket
    with one year in it.
    """
    frame = pd.DataFrame(panel.available.astype(float),
                         index=panel.years, columns=list(panel.countries))
    if not decade:
        return frame.T
    buckets = (frame.index // 10) * 10
    grouped = frame.groupby(buckets).mean()
    grouped.index.name = "decade"
    return grouped.T


def correlation_matrices(panel: Panel) -> Dict[str, pd.DataFrame]:
    """Pooled cross-asset and cross-country correlation matrices."""
    stacked = panel.stacked()
    flat = stacked.reshape(-1, len(CORE_SERIES))
    ok = np.all(np.isfinite(flat), axis=1)
    cross_asset = pd.DataFrame(
        np.corrcoef(flat[ok].T), index=list(CORE_SERIES), columns=list(CORE_SERIES)
    )

    eq = pd.DataFrame(panel.dom_eq, index=panel.years, columns=list(panel.countries))
    cross_country = eq.corr(min_periods=20)
    return {"cross_asset": cross_asset, "cross_country_equity": cross_country}


# ---------------------------------------------------------------------------
# Monthly disaggregation (auxiliary output)
# ---------------------------------------------------------------------------
def to_monthly(panel: Panel, seed: int = 991) -> Dict[str, np.ndarray]:
    """Disaggregate the annual panel into a monthly array.

    Each annual log return is split into twelve monthly log increments that
    sum *exactly* to the annual figure, with the within-year dispersion drawn
    from a mean-zero Gaussian bridge.  This adds no information -- it is a
    presentational convenience so that downstream code can consume a monthly
    array -- and the headline engine deliberately runs on the annual panel.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    n_t, n_c = panel.dom_eq.shape
    for key in CORE_SERIES:
        annual = panel.series(key)
        log_annual = np.log1p(np.clip(annual, GROSS_RETURN_FLOOR - 1.0, None))
        noise = rng.standard_normal((n_t, n_c, 12))
        noise -= noise.mean(axis=2, keepdims=True)
        scale = np.nanstd(log_annual, axis=0, keepdims=True)[..., None] / np.sqrt(12.0)
        monthly_log = log_annual[..., None] / 12.0 + noise * np.nan_to_num(scale)
        # Restore the exact annual aggregate.
        monthly_log = monthly_log - monthly_log.sum(axis=2, keepdims=True) / 12.0 \
            + log_annual[..., None] / 12.0
        # (year, country, month) -> (year x month, country), so that row
        # ``12 * t + m`` is month ``m`` of calendar year ``years[t]``.
        out[key] = np.expm1(
            monthly_log.transpose(0, 2, 1).reshape(n_t * 12, n_c))
    out["month_index"] = np.repeat(panel.years, 12) + np.tile(
        np.arange(12) / 12.0, n_t
    )
    return out


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------
def prepare_panels(cfg: Mapping[str, Any]) -> Dict[str, Panel]:
    """Build, persist and return every panel the run needs."""
    modes = {cfg["bootstrap"]["panel"], *cfg["report"].get("robustness_panels", [])}
    processed = Path(cfg["run"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    panels: Dict[str, Panel] = {}
    for mode in sorted(modes):
        panel = build_panel(cfg, mode=mode)
        panel.save(processed / f"panel_{mode}.npz")
        panel.to_frame().to_csv(processed / f"panel_{mode}.csv", index=False)
        panels[mode] = panel
        LOGGER.info("built panel %s: %d countries x %d years",
                    mode, panel.n_countries, panel.n_years)
    return panels
