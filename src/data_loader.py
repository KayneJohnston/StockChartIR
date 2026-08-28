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

#: The 22 remaining developed markets used to reach the paper's 38-country
#: cross-section.  Membership rule (see docs/01, section 2):  IMF "advanced
#: economies" that host an investable equity market -- i.e. excluding
#: Andorra, Macao SAR, Puerto Rico and San Marino -- plus Poland, which
#: FTSE Russell reclassified as a Developed Market in September 2018.
#:
#: Each entry maps ISO-3 -> (display name, Clio-Infra column or None,
#: earliest plausible year of an organised domestic equity market).
TIER_B_SPEC: Mapping[str, Tuple[str, str | None, int]] = {
    "AUT": ("Austria", "Austria", 1890),
    "CAN": ("Canada", "Canada", 1890),
    "HRV": ("Croatia", "Croatia", 1993),
    "CYP": ("Cyprus", "Cyprus", 1996),
    "CZE": ("Czech Republic", "Czech Republic", 1993),
    "EST": ("Estonia", "Estonia", 1996),
    "GRC": ("Greece", "Greece", 1890),
    "HKG": ("Hong Kong SAR", None, 1947),
    "ISL": ("Iceland", "Iceland", 1993),
    "IRL": ("Ireland", "Ireland", 1922),
    "ISR": ("Israel", "Israel", 1949),
    "KOR": ("Korea", "South Korea", 1956),
    "LVA": ("Latvia", "Latvia", 1996),
    "LTU": ("Lithuania", "Lithuania", 1993),
    "LUX": ("Luxembourg", "Luxembourg", 1929),
    "MLT": ("Malta", "Malta", 1992),
    "NZL": ("New Zealand", "New Zealand", 1890),
    "POL": ("Poland", "Poland", 1991),
    "SGP": ("Singapore", "Singapore", 1965),
    "SVK": ("Slovakia", "Slovakia", 1993),
    "SVN": ("Slovenia", "Slovenia", 1990),
    "TWN": ("Taiwan", None, 1962),
}

ISO_TO_NAME: Dict[str, str] = {
    "AUS": "Australia", "BEL": "Belgium", "CHE": "Switzerland",
    "DEU": "Germany", "DNK": "Denmark", "ESP": "Spain", "FIN": "Finland",
    "FRA": "France", "GBR": "United Kingdom", "ITA": "Italy",
    "JPN": "Japan", "NLD": "Netherlands", "NOR": "Norway",
    "PRT": "Portugal", "SWE": "Sweden", "USA": "United States",
    **{iso: spec[0] for iso, spec in TIER_B_SPEC.items()},
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
    #: than generated. Empty for a panel that is empirical throughout, in
    #: which case every available cell is observed by construction.
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
            "bill_rate", "eq_dp", "bond_rate", "stir", "ltrate", "housing_tr",
            "wage"]
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
        name="jst16",
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Tier B: calibrated developed-market extension
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FactorFit:
    """Country-level factor loadings estimated on the Tier-A cross-section."""

    iso: str
    alpha: Dict[str, float]
    beta: Dict[str, float]
    resid_cov: np.ndarray
    resid_keys: Tuple[str, ...]


def world_factors(panel: Panel) -> Dict[str, np.ndarray]:
    """Equally weighted cross-country mean of each core series, by year."""
    out: Dict[str, np.ndarray] = {}
    for key in CORE_SERIES:
        arr = np.where(panel.available, panel.series(key), np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[key] = np.nanmean(arr, axis=1)
    return out


def fit_factor_model(panel: Panel) -> List[FactorFit]:
    """Regress each Tier-A country's series on the corresponding world factor.

    The residual covariance is retained so that synthetic countries inherit
    the empirical cross-asset correlation structure (equity/bond/bill/CPI)
    of a real developed market rather than an assumed one.
    """
    factors = world_factors(panel)
    keys = CORE_SERIES
    fits: List[FactorFit] = []
    for c, iso in enumerate(panel.countries):
        alpha: Dict[str, float] = {}
        beta: Dict[str, float] = {}
        residuals: Dict[str, np.ndarray] = {}
        mask_all = panel.available[:, c]
        for key in keys:
            if key == "intl_eq":
                # The international leg is rebuilt from scratch for the full
                # panel, so it needs no country-specific factor fit.
                alpha[key], beta[key] = 0.0, 1.0
                residuals[key] = np.zeros(int(mask_all.sum()))
                continue
            y = panel.series(key)[:, c]
            x = factors[key]
            mask = mask_all & np.isfinite(y) & np.isfinite(x)
            if mask.sum() < 10:
                alpha[key], beta[key] = float(np.nanmean(y)), 1.0
                residuals[key] = np.zeros(int(mask_all.sum()))
                continue
            design = np.column_stack([np.ones(mask.sum()), x[mask]])
            coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
            alpha[key], beta[key] = float(coef[0]), float(coef[1])
            resid = y[mask] - design @ coef
            residuals[key] = resid
        # Align residual vectors on the common set of usable years.
        common = mask_all & np.all(
            [np.isfinite(panel.series(k)[:, c]) for k in keys], axis=0
        )
        stack = []
        for key in keys:
            y = panel.series(key)[:, c]
            x = factors[key]
            stack.append(y[common] - (alpha[key] + beta[key] * x[common]))
        resid_mat = np.vstack(stack) if common.sum() > 5 else np.zeros((len(keys), 2))
        cov = np.cov(resid_mat) if resid_mat.shape[1] > 2 else np.eye(len(keys)) * 1e-4
        cov = np.atleast_2d(cov)
        fits.append(FactorFit(iso=iso, alpha=alpha, beta=beta,
                              resid_cov=cov, resid_keys=keys))
    return fits


def _clio_inflation_for(iso: str, clio: pd.DataFrame,
                        years: np.ndarray) -> np.ndarray:
    column = TIER_B_SPEC[iso][1]
    if column is None or column not in clio.columns:
        return np.full(years.size, np.nan)
    series = clio[column].reindex(years)
    return series.to_numpy(dtype=float)


def _jst_inflation_for(iso: str, jst: pd.DataFrame,
                       years: np.ndarray) -> np.ndarray:
    sub = jst[jst["iso"] == iso]
    if sub.empty:
        return np.full(years.size, np.nan)
    return (
        sub.set_index("year")["inflation"].reindex(years).to_numpy(dtype=float)
    )


def build_tier_b(
    tier_a: Panel,
    cfg: Mapping[str, Any],
    jst: pd.DataFrame,
) -> Tuple[np.ndarray, ...]:
    """Generate the calibrated Tier-B block.

    Returns ``(isos, dom_eq, bond, bill, inflation, cpi, xrusd, usd_gross,
    fx_gain)`` for the Tier-B countries, aligned to ``tier_a.years``.

    Construction (documented in docs/01 section 4):

    1. Inflation is empirical wherever a source carries it -- JST CPI for
       Canada and Ireland, Clio-Infra CPI inflation for the remainder.
       Where no source exists the world inflation factor plus a donor-scaled
       idiosyncratic shock is used.
    2. Real equity/bond/bill returns are drawn from the single-factor model
       ``x_{i,t} = alpha_i + beta_i * f_t + eps_{i,t}`` whose parameters are
       resampled from the empirical Tier-A cross-section (a "typical
       developed market" draw) and whose residuals inherit that donor's
       cross-asset covariance.
    3. Coverage starts at the later of the panel start and a documented
       market-inception year, so the synthetic cross-section reproduces the
       staggered entry of the real developed-market universe.
    """
    data_cfg = cfg["data"]
    years = tier_a.years
    rng = np.random.default_rng(int(data_cfg["extension"]["seed"]))
    factors = world_factors(tier_a)
    fits = fit_factor_model(tier_a)
    clio = load_clio_wide(data_cfg["clio_inflation"])
    clio_yields = load_clio_wide(data_cfg["clio_bond_yield"])

    isos = list(TIER_B_SPEC)
    notes: List[str] = []
    n_t, n_b = years.size, len(isos)
    dom_eq = np.full((n_t, n_b), np.nan)
    bond = np.full((n_t, n_b), np.nan)
    bill = np.full((n_t, n_b), np.nan)
    inflation = np.full((n_t, n_b), np.nan)
    #: True where that country-year's inflation came from a published price
    #: index rather than from the factor model. It gates the observed flag on
    #: every *real* return built here: a genuine nominal yield deflated by a
    #: drawn CPI is not an observation.
    empirical_inflation = np.zeros((n_t, n_b), dtype=bool)

    world_infl = factors["inflation"]
    keys = list(CORE_SERIES)
    idx_of = {k: i for i, k in enumerate(keys)}

    for b, iso in enumerate(isos):
        donor = fits[int(rng.integers(len(fits)))]
        start_year = max(int(years[0]), int(TIER_B_SPEC[iso][2]))
        active = years >= start_year

        # --- inflation: empirical where a primary source exists -----------
        infl = _jst_inflation_for(iso, jst, years)
        if not np.isfinite(infl).any():
            infl = _clio_inflation_for(iso, clio, years)
        infl_missing = ~np.isfinite(infl) & active
        if infl_missing.any():
            sd = float(np.sqrt(max(donor.resid_cov[idx_of["inflation"],
                                                   idx_of["inflation"]], 1e-8)))
            draw = (donor.alpha["inflation"]
                    + donor.beta["inflation"] * np.nan_to_num(world_infl)
                    + rng.normal(0.0, sd, size=n_t))
            infl = np.where(infl_missing, draw, infl)
        inflation[:, b] = np.where(active, infl, np.nan)
        empirical_inflation[:, b] = active & ~infl_missing & np.isfinite(infl)
        n_active = int(active.sum())
        n_empirical = int((active & ~infl_missing).sum()) if n_active else 0
        share = n_empirical / n_active if n_active else 0.0
        source = "JST CPI" if iso in ("CAN", "IRL") else (
            "Clio-Infra CPI" if TIER_B_SPEC[iso][1] else "none")
        notes.append(
            f"Tier-B calibrated: inflation {source} "
            f"({share:.0%} of {n_active} active years empirical, remainder "
            f"factor-model); equity/bond/bill simulated from donor "
            f"{donor.iso}; market inception {start_year}"
        )

        # --- asset returns: factor model with donor residual covariance ---
        sub_keys = ["dom_eq", "bond", "bill"]
        sub_idx = [idx_of[k] for k in sub_keys]
        cov = donor.resid_cov[np.ix_(sub_idx, sub_idx)]
        cov = cov + np.eye(len(sub_keys)) * 1e-10
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate donors
            chol = np.diag(np.sqrt(np.clip(np.diag(cov), 1e-10, None)))
        shocks = rng.standard_normal((n_t, len(sub_keys))) @ chol.T
        for j, key in enumerate(sub_keys):
            fitted = (donor.alpha[key]
                      + donor.beta[key] * np.nan_to_num(factors[key])
                      + shocks[:, j])
            fitted = np.maximum(fitted, GROSS_RETURN_FLOOR - 1.0)
            target = {"dom_eq": dom_eq, "bond": bond, "bill": bill}[key]
            target[:, b] = np.where(active & np.isfinite(factors[key]),
                                    fitted, np.nan)

    # ---- replace simulated series with observed ones where they exist -----
    # Four of these countries do have real interest-rate histories, in the
    # macro file (Canada, Ireland) or in Clio-Infra (New Zealand, Austria).
    # A bond return follows from a long yield and a duration; a bill return is
    # a short rate. Simulating a series the sources can supply would be
    # indefensible, so those cells are overwritten and recorded.
    duration = float(data_cfg.get("bond_duration_years", 7.0))
    observed_mask = {key: np.zeros((n_t, n_b), dtype=bool)
                     for key in ("dom_eq", "bond", "bill", "inflation")}
    for b, iso in enumerate(isos):
        recovered: Dict[str, np.ndarray] = {}
        if iso in obs.JST_RATE_ONLY:
            recovered = obs.rates_from_jst(jst, iso, years, duration)
        else:
            column = next((c for c, code in obs.CLIO_BOND_COUNTRIES.items()
                           if code == iso), None)
            if column is not None:
                recovered = {"bond": obs.bond_from_clio(
                    clio_yields, column, years, inflation[:, b], duration)}
        # A real return is observed only if *both* its nominal series and its
        # deflator are. Years whose CPI came from the factor model keep the
        # simulated return rather than a half-observed one.
        active = empirical_inflation[:, b]
        for key, values in recovered.items():
            target = {"dom_eq": dom_eq, "bond": bond, "bill": bill}[key]
            usable = np.isfinite(values) & active
            if not usable.any():
                continue
            target[usable, b] = values[usable]
            observed_mask[key][usable, b] = True
            source = ("JST rates" if iso in obs.JST_RATE_ONLY
                      else "Clio-Infra yields")
            notes[b] += (f"; {key} OBSERVED from {source} "
                         f"({int(usable.sum())} years)")
    observed_mask["inflation"] = empirical_inflation.copy()

    # Nominal CPI level and USD exchange rate are needed to fold Tier-B
    # countries into the international equity leg.  CPI is cumulated from
    # the inflation series; the USD rate is cumulated from a PPP-consistent
    # rule (relative inflation against the world basket), which keeps the
    # real exchange rate stationary by construction for synthetic markets.
    cpi = np.full((n_t, n_b), np.nan)
    xrusd = np.full((n_t, n_b), np.nan)
    world_infl_filled = np.nan_to_num(world_infl)
    for b in range(n_b):
        active = np.isfinite(inflation[:, b])
        if not active.any():
            continue
        first = int(np.flatnonzero(active)[0])
        level, fx = 100.0, 1.0
        for t in range(first, n_t):
            if not active[t]:
                break
            level *= (1.0 + inflation[t, b])
            fx *= (1.0 + inflation[t, b]) / (1.0 + world_infl_filled[t])
            cpi[t, b] = level
            xrusd[t, b] = fx
    return (np.array(isos, dtype=object), dom_eq, bond, bill,
            inflation, cpi, xrusd, tuple(notes), observed_mask)


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
    """Build the requested panel.

    Parameters
    ----------
    mode:
        ``"jst16"`` returns the fully empirical panel; ``"dev38"`` appends
        the calibrated developed-market extension and *recomputes* the
        international equity leg over the enlarged cross-section.  Defaults
        to ``cfg["bootstrap"]["panel"]``.
    """
    mode = mode or cfg["bootstrap"]["panel"]
    data_cfg = cfg["data"]
    hedge_ratio = float(data_cfg.get("hedge_ratio", 0.0)
                        if hedge_ratio is None else hedge_ratio)
    hedge_cost = float(data_cfg.get("hedge_cost", 0.0)
                       if hedge_cost is None else hedge_cost)
    tier_a = build_tier_a(cfg, hedge_ratio, hedge_cost)
    if mode == "jst16":
        return tier_a
    if mode != "dev38":
        raise ValueError(f"unknown panel mode {mode!r}")
    if not cfg["data"]["extension"]["enabled"]:
        raise ValueError("dev38 requested but data.extension.enabled is false")

    jst = add_real_returns(load_jst(cfg))
    (b_iso, b_eq, b_bond, b_bill, b_infl, b_cpi, b_fx,
     b_notes, b_observed) = build_tier_b(tier_a, cfg, jst)

    years = tier_a.years
    isos = list(tier_a.countries) + [str(i) for i in b_iso]
    tiers = ["A"] * tier_a.n_countries + ["B"] * len(b_iso)
    notes = list(tier_a.provenance) + list(b_notes)

    dom_eq = np.column_stack([tier_a.dom_eq, b_eq])
    bond = np.column_stack([tier_a.bond, b_bond])
    bill = np.column_stack([tier_a.bill, b_bill])
    inflation = np.column_stack([tier_a.inflation, b_infl])

    # Re-derive the nominal inputs needed for the international leg.
    jst_fx = _usd_gross_equity(jst)
    window = jst_fx[jst_fx["year"].between(years[0] - 1, years[-1])]
    a_usd_gross = _pivot(window, "usd_gross_eq", years, tier_a.countries)
    a_fx_gain = _pivot(window, "fx_gain", years, tier_a.countries)
    a_cpi = _pivot(window, "cpi", years, tier_a.countries)
    a_xrusd = _pivot(window, "xrusd", years, tier_a.countries)

    b_fx_gain = _fx_gain_from_levels(b_fx)
    b_nominal_eq = (1.0 + b_eq) * (1.0 + b_infl) - 1.0
    b_usd_gross = (1.0 + b_nominal_eq) * b_fx_gain

    usd_gross = np.column_stack([a_usd_gross, b_usd_gross])
    fx_gain = np.column_stack([a_fx_gain, b_fx_gain])
    cpi = np.column_stack([a_cpi, b_cpi])
    xrusd = np.column_stack([a_xrusd, b_fx])

    winsor = float(cfg["data"].get("international_winsor_pct", 0.0))
    intl_eq = build_international_leg(
        usd_gross, fx_gain, inflation,
        weighting=cfg["data"]["international_weighting"],
        winsor_pct=winsor,
    )
    if hedge_ratio > 0.0:
        a_eq_tr = _pivot(window, "eq_tr", years, tier_a.countries)
        a_bill_nom = _pivot(window, "bill_rate", years, tier_a.countries)
        b_bill_nominal = (1.0 + b_bill) * (1.0 + b_infl) - 1.0
        hedged = build_hedged_international_leg(
            np.column_stack([a_eq_tr, b_nominal_eq]),
            np.column_stack([a_bill_nom, b_bill_nominal]),
            inflation, hedge_cost=hedge_cost, winsor_pct=winsor)
        intl_eq = blend_international_legs(intl_eq, hedged, hedge_ratio)
    rer = build_real_exchange_rate(cpi, xrusd)

    available = (
        np.isfinite(dom_eq) & np.isfinite(intl_eq) & np.isfinite(bond)
        & np.isfinite(bill) & np.isfinite(inflation)
    )
    # Tier-A cells are observed by construction; the simulated block carries
    # its own per-cell mask, with the international leg counted as observed
    # only where it is an average over observed markets -- which, being a
    # cross-country average, it never wholly is. It is therefore recorded
    # separately by `src.provenance` rather than forced into a binary here.
    n_a = tier_a.n_countries
    observed = {}
    for key in ("dom_eq", "bond", "bill", "inflation"):
        block = np.zeros((years.size, len(isos)), dtype=bool)
        block[:, :n_a] = True
        block[:, n_a:] = b_observed[key]
        observed[key] = block
    observed["intl_eq"] = np.zeros((years.size, len(isos)), dtype=bool)

    tiers = derive_tiers(observed, available, tiers)

    keep = available.sum(axis=0) >= int(cfg["data"]["min_observations"])
    idx = np.flatnonzero(keep)
    if idx.size != len(isos):
        LOGGER.warning(
            "dropping countries with < %s usable years: %s",
            cfg["data"]["min_observations"],
            [isos[i] for i in range(len(isos)) if i not in set(idx.tolist())],
        )

    return Panel(
        years=years,
        countries=tuple(isos[i] for i in idx),
        tier=tuple(tiers[i] for i in idx),
        dom_eq=dom_eq[:, idx],
        intl_eq=intl_eq[:, idx],
        bond=bond[:, idx],
        bill=bill[:, idx],
        inflation=inflation[:, idx],
        real_exchange_rate=rer[:, idx],
        available=available[:, idx],
        name="dev38",
        provenance=tuple(notes[i] for i in idx),
        observed={k: v[:, idx] for k, v in observed.items()},
    )


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
    """Country x year (or decade) count of usable observations."""
    frame = pd.DataFrame(panel.available.astype(int),
                         index=panel.years, columns=list(panel.countries))
    if not decade:
        return frame.T
    frame["decade"] = (frame.index // 10) * 10
    grouped = frame.groupby("decade").sum()
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
