"""Where every number in the panel actually comes from.

`docs/01` labels each country Tier A or Tier B and says Tier B is
"constructed". That is true but far too soft, and this module exists because
the softness was hiding something a reader needs to know: for a Tier-B country
the equity, bond and bill series are not derived from anything that country
experienced. They are draws from a single-factor model fitted to a randomly
assigned Tier-A donor, plus Gaussian noise with that donor's residual
covariance. Only inflation is empirical, and only where a source carries it.

So the honest description is not "constructed from documented mappings". It is
**simulated**. This module measures exactly how much of the panel that covers,
how much of it reaches a simulated lifetime through the bootstrap, and — the
part that matters most for the headline result — how much of the
*international* leg is simulated even for the sixteen countries whose own data
are real.

It also audits the empirical tier rather than taking it on trust:

* the internal accounting identity ``eq_tr = (1 + eq_capgain)(1 + eq_dp) − 1``
  is checked observation by observation;
* a handful of returns whose values are independently known are compared
  against the source;
* and the last few years of every series are tested for the variance collapse
  that would indicate a redistributed file had been extended or smoothed by
  someone other than the original compilers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import observed as obs

LOGGER = logging.getLogger(__name__)

#: Series whose provenance is at issue. Inflation is treated separately
#: because it is empirical for most Tier-B countries while the return series
#: never are.
RETURN_SERIES: Tuple[str, ...] = ("dom_eq", "bond", "bill")

#: Independently known annual total returns, used to test that the workbook is
#: the real Jordà–Schularick–Taylor file rather than something reconstructed.
#: Each is a widely published figure for a year whose outcome is not in doubt.
KNOWN_ANCHORS: Tuple[Dict[str, Any], ...] = (
    {"iso": "USA", "year": 1931, "series": "eq_tr", "expected": -0.43,
     "tolerance": 0.06, "what": "US equities in the worst year of the slump"},
    {"iso": "USA", "year": 2008, "series": "eq_tr", "expected": -0.37,
     "tolerance": 0.05, "what": "US equities in the global financial crisis"},
    {"iso": "USA", "year": 1974, "series": "eq_tr", "expected": -0.26,
     "tolerance": 0.06, "what": "US equities in the 1974 bear market"},
    {"iso": "USA", "year": 1933, "series": "eq_tr", "expected": 0.54,
     "tolerance": 0.08, "what": "US equities rebounding in 1933"},
    {"iso": "USA", "year": 2013, "series": "eq_tr", "expected": 0.32,
     "tolerance": 0.06, "what": "US equities in 2013"},
)


def file_digest(path: str | Path) -> Dict[str, Any]:
    """Size and SHA-256 of a source file, so a run can be tied to its inputs."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False, "bytes": 0, "sha256": ""}
    payload = path.read_bytes()
    return {"path": str(path), "exists": True, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def source_digests(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Fingerprint every raw input the panel is built from."""
    data = cfg["data"]
    rows = [file_digest(data[key]) for key in
            ("jst_workbook", "clio_inflation", "clio_bond_yield")
            if key in data]
    frame = pd.DataFrame.from_records(rows)
    frame["file"] = [Path(p).name for p in frame["path"]]
    return frame[["file", "exists", "bytes", "sha256", "path"]]


# ---------------------------------------------------------------------------
# What is empirical and what is simulated
# ---------------------------------------------------------------------------
def load_raw_workbook(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """The JST workbook exactly as distributed, with no columns dropped.

    :func:`src.data_loader.load_jst` keeps only what the pipeline consumes.
    The audit needs the discarded columns -- the capital-gain and dividend
    components in particular -- because they are what make the internal
    accounting identity testable.
    """
    data = cfg["data"]
    return pd.read_excel(data["jst_workbook"], sheet_name=data["jst_sheet"])


#: The three investable return series whose provenance this module measures.
#: Inflation is empirical for nearly every country and would flatter any
#: aggregate it entered; the international leg is measured separately because
#: it is a cross-country average and so never wholly one thing or the other.
RETURN_SERIES: Tuple[str, ...] = ("dom_eq", "bond", "bill")


def observed_cells(panel: dl.Panel) -> Dict[str, np.ndarray]:
    """``series -> (T, C)`` boolean, True where the cell is a real observation.

    Counting cells rather than countries is what lets this module report the
    recovered interest-rate histories honestly: Canada's bonds are observed
    for 130 years while its equity is simulated throughout, and a country-level
    label cannot express that.
    """
    return {key: panel.observed_mask(key) & panel.available
            for key in RETURN_SERIES}


def _cell_counts(panel: dl.Panel) -> Tuple[np.ndarray, np.ndarray]:
    """``(observed, total)`` return-cells per country, summed over series."""
    cells = observed_cells(panel)
    observed = np.zeros(panel.n_countries, dtype=int)
    for values in cells.values():
        observed += values.sum(axis=0).astype(int)
    total = panel.available.sum(axis=0).astype(int) * len(RETURN_SERIES)
    return observed, total


def country_provenance(panel: dl.Panel) -> pd.DataFrame:
    """One row per country: tier, coverage, and the source of each series.

    The ``returns_source`` column is the one that matters. "JST/JKKST" means
    the country's own recorded history; "factor model" means the series was
    generated and no observation in it happened.
    """
    available = panel.available
    years = np.asarray(panel.years)
    cells = observed_cells(panel)
    rows: List[Dict[str, Any]] = []
    for i, iso in enumerate(panel.countries):
        column = available[:, i]
        n = int(column.sum())
        note = panel.provenance[i] if i < len(panel.provenance) else ""
        tier = panel.tier[i]
        donor = ""
        if "donor" in note:
            donor = note.split("donor")[1].split(";")[0].strip()
        seen = {key: int(values[:, i].sum()) for key, values in cells.items()}
        observed_years = sum(seen.values())
        total_years = n * len(RETURN_SERIES)
        rows.append({
            "iso": iso,
            "country": dl.ISO_TO_NAME.get(iso, iso),
            "tier": tier,
            "tier_label": dl.TIER_LABELS.get(tier, tier),
            "usable_years": n,
            "first_year": int(years[column].min()) if n else 0,
            "last_year": int(years[column].max()) if n else 0,
            "returns_source": _returns_source(tier, seen),
            "returns_empirical_years": observed_years,
            "returns_simulated_years": total_years - observed_years,
            "share_returns_observed": (observed_years / total_years)
            if total_years else 0.0,
            "dom_eq_observed_years": seen["dom_eq"],
            "bond_observed_years": seen["bond"],
            "bill_observed_years": seen["bill"],
            "inflation_empirical_share": _inflation_share(note),
            "donor": donor,
            "note": note,
        })
    return pd.DataFrame.from_records(rows)


def housing_audit(jst: pd.DataFrame, panel: dl.Panel) -> pd.DataFrame:
    """Audit the housing total returns the panel holds but does not invest in.

    Housing is the fourth asset of the "Rate of Return on Everything" project
    and is fully empirical for all sixteen observed countries -- the largest
    block of genuine data in the sources that no part of this pipeline reads.
    It is audited here rather than added to the investable set, because an
    appraisal-based index is not comparable with a traded one until it has been
    de-smoothed, and the de-smoothing is itself an assumption.

    One row per country: coverage, the raw moments, the lag-one
    autocorrelation that measures the smoothing, and the standard deviation
    that survives undoing it. The equity column is the same country's own
    domestic equity over the same calendar, so the comparison is like for like.
    """
    years = np.asarray(panel.years)
    isos = [iso for iso, tier in zip(panel.countries, panel.tier)
            if tier == "A"]
    if not isos:
        return pd.DataFrame()
    housing = obs.housing_returns(jst, isos, years)
    rows: List[Dict[str, Any]] = []
    for j, iso in enumerate(isos):
        column = housing[:, j]
        finite = np.isfinite(column)
        if finite.sum() < 10:
            continue
        equity = panel.dom_eq[:, panel.country_index(iso)]
        rho = obs.first_order_autocorrelation(column)
        smoothed = obs.desmooth(column, rho)
        rows.append({
            "iso": iso,
            "country": dl.ISO_TO_NAME.get(iso, iso),
            "years": int(finite.sum()),
            "first_year": int(years[finite].min()),
            "last_year": int(years[finite].max()),
            "mean": float(np.nanmean(column)),
            "sd": float(np.nanstd(column, ddof=1)),
            "autocorrelation": rho,
            "sd_desmoothed": float(np.nanstd(smoothed, ddof=1)),
            "equity_mean": float(np.nanmean(equity)),
            "equity_sd": float(np.nanstd(equity, ddof=1)),
            "equity_autocorrelation": obs.first_order_autocorrelation(equity),
        })
    return pd.DataFrame.from_records(rows)


def housing_summary(audit: pd.DataFrame) -> Dict[str, Any]:
    """Headline numbers from :func:`housing_audit`, for the prose to quote."""
    if audit.empty:
        return {"countries": 0, "country_years": 0}
    return {
        "countries": int(len(audit)),
        "country_years": int(audit["years"].sum()),
        "first_year": int(audit["first_year"].min()),
        "last_year": int(audit["last_year"].max()),
        "mean": float(audit["mean"].median()),
        "sd": float(audit["sd"].median()),
        "sd_desmoothed": float(audit["sd_desmoothed"].median()),
        "autocorrelation": float(audit["autocorrelation"].median()),
        "equity_mean": float(audit["equity_mean"].median()),
        "equity_sd": float(audit["equity_sd"].median()),
        "equity_autocorrelation": float(
            audit["equity_autocorrelation"].median()),
        "n_more_autocorrelated": int(
            (audit["autocorrelation"] > audit["equity_autocorrelation"]).sum()),
    }


#: Shortest wage history worth taking a geometric mean of. Below about thirty
#: years the mean is dominated by which decade the series happens to cover.
MIN_WAGE_YEARS = 30

#: Years in which a wage index is measuring occupation, rationing, suppressed
#: prices and post-war repricing at least as much as it measures wages. They
#: are not excluded from the headline number -- a worker alive then lived
#: through them -- but the audit reports both so a reader can see how much of
#: the answer they carry.
WAR_YEARS: Tuple[Tuple[int, int], ...] = ((1914, 1919), (1939, 1946))


def _war_mask(years: np.ndarray,
              windows: Sequence[Tuple[int, int]] = WAR_YEARS) -> np.ndarray:
    """True for calendar years inside any of ``windows``, inclusive."""
    years = np.asarray(years)
    mask = np.zeros(years.shape, dtype=bool)
    for low, high in windows:
        mask |= (years >= low) & (years <= high)
    return mask


def wage_audit(jst: pd.DataFrame, panel: dl.Panel, cfg: Mapping[str, Any],
               min_years: int = MIN_WAGE_YEARS) -> pd.DataFrame:
    """Measured real wage growth against what the income model assumes.

    The lifecycle income profile is ``log Y(a) = log Y0 + b1·(a−a0) +
    b2·(a−a0)²`` -- a pure *age* effect, capturing the career progression a
    worker earns by getting older. It carries no term for the productivity
    growth that lifts the entire wage distribution regardless of age, and the
    macro file measures exactly that, for all eighteen of its countries.

    One row per country: the geometric mean real wage growth it recorded and
    what a career's worth of it compounds to. The two components are additive
    in a fully specified income process, so this is a calibration gap rather
    than a duplicate of the hump.
    """
    if "wage" not in jst.columns:
        return pd.DataFrame()
    years = np.asarray(panel.years)
    isos = sorted({str(i) for i in jst["iso"].dropna().unique()})
    growth = obs.wage_growth(jst, isos, years)
    war = _war_mask(years)
    career = int(cfg["lifecycle"]["age_retire"]) - int(
        cfg["lifecycle"]["age_start"])
    rows: List[Dict[str, Any]] = []
    for j, iso in enumerate(isos):
        column = growth[:, j]
        finite = np.isfinite(column)
        if finite.sum() < int(min_years):
            continue
        # Geometric, because it is what compounds across a career.
        geometric = float(np.exp(np.mean(np.log1p(column[finite]))) - 1.0)
        peacetime = finite & ~war
        ex_war = (float(np.exp(np.mean(np.log1p(column[peacetime]))) - 1.0)
                  if peacetime.sum() >= int(min_years) else float("nan"))
        rows.append({
            "iso": iso,
            "country": dl.ISO_TO_NAME.get(iso, iso),
            "years": int(finite.sum()),
            "first_year": int(years[finite].min()),
            "last_year": int(years[finite].max()),
            "mean": float(np.mean(column[finite])),
            "geometric_mean": geometric,
            "geometric_mean_ex_war": ex_war,
            "sd": float(np.std(column[finite], ddof=1)),
            "career_multiple": float((1.0 + geometric) ** career),
            "in_return_panel": iso in panel.countries,
        })
    return pd.DataFrame.from_records(rows)


def income_profile_growth(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """What the model's own age profile implies, for the same comparison."""
    life = cfg["lifecycle"]
    income = life["income"]
    start, retire = int(life["age_start"]), int(life["age_retire"])
    ages = np.arange(start, retire)
    offset = ages - start
    profile = np.exp(np.log(float(income["initial_real_income"]))
                     + float(income["b1"]) * offset
                     + float(income["b2"]) * offset ** 2)
    span = profile.size - 1
    b1, b2 = float(income["b1"]), float(income["b2"])
    return {
        "career_years": span,
        "peak_age": float(start - b1 / (2.0 * b2)) if b2 else float("nan"),
        "peak_multiple": float(profile.max() / profile[0]),
        "end_multiple": float(profile[-1] / profile[0]),
        "implied_growth": float((profile[-1] / profile[0]) ** (1.0 / span)
                                - 1.0) if span else float("nan"),
    }


def wage_summary(audit: pd.DataFrame, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Headline numbers for the prose, including the gap being reported."""
    model = income_profile_growth(cfg)
    if audit.empty:
        return {"countries": 0, **{f"model_{k}": v for k, v in model.items()}}
    measured = float(audit["geometric_mean"].median())
    ex_war = float(audit["geometric_mean_ex_war"].median())
    combined = (1.0 + measured) * (1.0 + model["implied_growth"]) - 1.0
    return {
        "measured_ex_war": ex_war,
        "war_shifted_by": ex_war - measured,
        "war_years": " and ".join(f"{low}-{high}" for low, high in WAR_YEARS),
        "countries": int(len(audit)),
        "country_years": int(audit["years"].sum()),
        "first_year": int(audit["first_year"].min()),
        "last_year": int(audit["last_year"].max()),
        "measured": measured,
        "lowest": float(audit["geometric_mean"].min()),
        "highest": float(audit["geometric_mean"].max()),
        "lowest_country": str(audit.loc[audit["geometric_mean"].idxmin(),
                                        "country"]),
        "highest_country": str(audit.loc[audit["geometric_mean"].idxmax(),
                                         "country"]),
        "career_multiple": float((1.0 + measured) ** model["career_years"]),
        "outside_return_panel": int((~audit["in_return_panel"]).sum()),
        "combined_growth": combined,
        **{f"model_{k}": v for k, v in model.items()},
    }


def recovered_series(panel: dl.Panel) -> pd.DataFrame:
    """What was rebuilt from published rates rather than generated.

    One row per recovered country-series: the source, the span, and how many
    country-years stopped being simulated because of it. Empty when nothing was
    recovered, which is the correct output for a wholly empirical panel.
    """
    years = np.asarray(panel.years)
    cells = observed_cells(panel)
    rows: List[Dict[str, Any]] = []
    for i, iso in enumerate(panel.countries):
        if panel.tier[i] != "B":
            continue
        note = panel.provenance[i] if i < len(panel.provenance) else ""
        source = ("Jordà–Schularick–Taylor yields and short rates"
                  if "JST rates" in note else "Clio-Infra bond yields")
        for key, values in cells.items():
            column = values[:, i]
            n = int(column.sum())
            if not n:
                continue
            rows.append({
                "iso": iso,
                "country": dl.ISO_TO_NAME.get(iso, iso),
                "series": key,
                "source": source,
                "first_year": int(years[column].min()),
                "last_year": int(years[column].max()),
                "observed_years": n,
            })
    if not rows:
        return pd.DataFrame(columns=["iso", "country", "series", "source",
                                     "first_year", "last_year",
                                     "observed_years"])
    return pd.DataFrame.from_records(rows).sort_values(["iso", "series"]) \
        .reset_index(drop=True)


def _returns_source(tier: str, seen: Mapping[str, int]) -> str:
    """Name the source of a country's return history in a few words."""
    if tier == "A":
        return "JST/JKKST"
    recovered = ", ".join(k for k in RETURN_SERIES if seen.get(k))
    if not recovered:
        return "factor model"
    return f"factor model; {recovered} recovered from published rates"


def _inflation_share(note: str) -> float:
    """Pull the empirical-inflation share out of a Tier-B provenance note."""
    if "% of" not in note:
        return 1.0
    try:
        return float(note.split("(")[1].split("%")[0]) / 100.0
    except (IndexError, ValueError):       # pragma: no cover - defensive
        return float("nan")


def panel_summary(panel: dl.Panel) -> Dict[str, Any]:
    """Headline shares: how much of the panel, and of the sampler, is simulated.

    Everything here counts *cells* -- one country, one year, one return series
    -- because that is the unit the bootstrap actually draws. A country-level
    count would have to call Canada either wholly real or wholly generated,
    and it is neither.
    """
    tier = np.asarray(panel.tier)
    available = panel.available
    observed, total_by_country = _cell_counts(panel)
    total = int(total_by_country.sum())
    simulated = total - int(observed.sum())
    history = available.sum(axis=0).astype(float)
    weight = history / history.sum()
    #: A draw lands on a country-year, so weight each country's simulated share
    #: of its own cells by how often the sampler reaches it.
    per_country = np.divide(observed, total_by_country,
                            out=np.zeros(len(observed), dtype=float),
                            where=total_by_country > 0)
    return {
        "n_countries": len(panel.countries),
        "n_observed_countries": int((tier == "A").sum()),
        "n_partial_countries": int((tier == "B").sum()),
        "n_simulated_countries": int((tier == "C").sum()),
        # Retained under their old names: every consumer reads these, and a
        # country with any simulated return still contaminates a lifetime.
        "n_empirical_countries": int((tier == "A").sum()),
        "country_years": int(available.sum()),
        "return_cells": total,
        "return_cells_empirical": int(observed.sum()),
        "return_cells_simulated": simulated,
        "country_years_empirical": int(available[:, tier == "A"].sum()),
        "country_years_simulated": int(available[:, tier != "A"].sum()),
        "share_country_years_simulated": float(
            available[:, tier != "A"].sum() / available.sum())
        if available.sum() else 0.0,
        "share_cells_simulated": simulated / total if total else 0.0,
        "share_draws_simulated": float((weight * (1.0 - per_country)).sum()),
    }


def simulated_share_by_era(panel: dl.Panel, edges: Sequence[int] =
                           (1890, 1930, 1970, 2000, 2021)) -> pd.DataFrame:
    """How the simulated share of the cross-section moves through time."""
    years = np.asarray(panel.years)
    cells = observed_cells(panel)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (years >= lo) & (years < hi)
        block = panel.available[mask]
        total = int(block.sum()) * len(RETURN_SERIES)
        seen = sum(int(values[mask].sum()) for values in cells.values())
        rows.append({
            "era": f"{lo}-{hi - 1}",
            "country_years": int(block.sum()),
            "return_cells": total,
            "simulated": total - seen,
            "share_simulated": float((total - seen) / total) if total else 0.0,
            "mean_countries_available": float(block.sum(axis=1).mean())
            if len(block) else 0.0,
        })
    return pd.DataFrame.from_records(rows)


def international_leg_contamination(panel: dl.Panel,
                                    edges: Sequence[int] =
                                    (1890, 1930, 1970, 2000, 2021)
                                    ) -> pd.DataFrame:
    """How much of an *empirical* investor's international leg is simulated.

    This is the number that bears hardest on the headline result. The
    international leg is a leave-one-out average across every country with
    data that year, so a Tier-A investor's "international diversification" is
    partly diversification into countries that do not exist.
    """
    tier = np.asarray(panel.tier)
    empirical = tier == "A"
    #: The leg is an average of *equity* returns, so a country contaminates it
    #: exactly when its own equity for that year was generated -- which stays
    #: true of the four countries whose interest rates were recovered.
    equity_seen = panel.observed_mask("dom_eq")
    years = np.asarray(panel.years)
    rows = []
    for lo, hi in zip((edges[0],) + tuple(edges[:-1]),
                      (edges[-1],) + tuple(edges[1:])):
        mask = (years >= lo) & (years < hi)
        shares: List[float] = []
        for t in np.flatnonzero(mask):
            column = panel.available[t]
            synthetic = column & ~equity_seen[t]
            for i in np.flatnonzero(column & empirical):
                others = column.copy()
                others[i] = False
                if others.sum():
                    shares.append(float((synthetic & others).sum()
                                        / others.sum()))
        rows.append({
            "era": "whole panel" if (lo, hi) == (edges[0], edges[-1])
            else f"{lo}-{hi - 1}",
            "observations": len(shares),
            "mean_synthetic_share_of_intl_leg": float(np.mean(shares))
            if shares else float("nan"),
        })
    return pd.DataFrame.from_records(rows).drop_duplicates("era")


# ---------------------------------------------------------------------------
# Auditing the empirical tier
# ---------------------------------------------------------------------------
def identity_check(jst: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """Test ``eq_tr = (1 + eq_capgain)(1 + eq_dp) − 1`` observation by observation.

    A file assembled by hand from a published table would satisfy this exactly.
    The real database does not, because the capital-gain and dividend
    components are sometimes spliced from different underlying indices, so a
    handful of large violations is a sign of authenticity rather than of
    error. What would be damning is the opposite: a file that satisfies it
    everywhere to machine precision has had its components back-solved from
    its totals.
    """
    frame = jst.dropna(subset=["eq_capgain", "eq_dp", "eq_tr"]).copy()
    implied = (1.0 + frame["eq_capgain"]) * (1.0 + frame["eq_dp"]) - 1.0
    frame["identity_error"] = (implied - frame["eq_tr"]).abs()
    breaches = frame[frame["identity_error"] > tolerance]
    return pd.DataFrame([{
        "observations": int(len(frame)),
        "violations_above_tolerance": int(len(breaches)),
        "share_violating": float(len(breaches) / len(frame)) if len(frame) else 0.0,
        "median_error": float(frame["identity_error"].median()),
        "max_error": float(frame["identity_error"].max()),
        "worst_iso": str(frame.loc[frame["identity_error"].idxmax(), "iso"]),
        "worst_year": int(frame.loc[frame["identity_error"].idxmax(), "year"]),
    }])


def anchor_check(jst: pd.DataFrame,
                 anchors: Sequence[Mapping[str, Any]] = KNOWN_ANCHORS
                 ) -> pd.DataFrame:
    """Compare the workbook against returns whose values are independently known."""
    rows: List[Dict[str, Any]] = []
    for anchor in anchors:
        block = jst[(jst["iso"] == anchor["iso"])
                    & (jst["year"] == anchor["year"])]
        value = float(block[anchor["series"]].iloc[0]) if len(block) \
            else float("nan")
        rows.append({
            "what": anchor["what"], "iso": anchor["iso"],
            "year": int(anchor["year"]), "series": anchor["series"],
            "workbook": value, "independently_known": float(anchor["expected"]),
            "difference": value - float(anchor["expected"]),
            "within_tolerance": bool(abs(value - float(anchor["expected"]))
                                     <= float(anchor["tolerance"])),
        })
    return pd.DataFrame.from_records(rows)


def tail_variance_test(jst: pd.DataFrame, series: str = "eq_tr",
                       tail_start: int = 2016, reference_start: int = 1950,
                       reference_end: int = 2015) -> pd.DataFrame:
    """Is the last stretch of every series implausibly smooth?

    A redistributed copy that has been extended past the compilers' own end
    date — by interpolation, by splicing a different index, or by carrying a
    smoothed estimate forward — shows up as a variance collapse in the tail
    that appears in *every* country at once. One country would be noise;
    all of them pointing the same way is a signature.

    Reported as a sign test, because a five-year standard deviation is far too
    noisy to interpret country by country.
    """
    rows: List[Dict[str, Any]] = []
    for iso, block in jst.groupby("iso"):
        block = block.set_index("year")
        if series not in block.columns:
            continue
        reference = block.loc[reference_start:reference_end, series].dropna()
        tail = block.loc[tail_start:, series].dropna()
        if len(reference) < 20 or len(tail) < 3:
            continue
        rows.append({
            "iso": str(iso), "n_reference": int(len(reference)),
            "n_tail": int(len(tail)),
            "sd_reference": float(reference.std()),
            "sd_tail": float(tail.std()),
            "ratio": float(tail.std() / reference.std())
            if reference.std() > 0 else float("nan"),
        })
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame
    frame["tail_smoother"] = frame["ratio"] < 1.0
    return frame.sort_values("ratio").reset_index(drop=True)


def sign_test_p_value(successes: int, trials: int) -> float:
    """Two-sided exact binomial p-value for a fair-coin null."""
    if trials <= 0:
        return float("nan")
    from math import comb
    tail = min(successes, trials - successes)
    cumulative = sum(comb(trials, k) for k in range(tail + 1))
    return float(min(1.0, 2.0 * cumulative / (2.0 ** trials)))


def tail_verdict(frame: pd.DataFrame) -> Dict[str, Any]:
    """Summarise the tail test into something a document can state."""
    if frame.empty:
        return {"countries": 0, "smoother": 0, "p_value": float("nan"),
                "median_ratio": float("nan")}
    smoother = int(frame["tail_smoother"].sum())
    return {
        "countries": int(len(frame)),
        "smoother": smoother,
        "median_ratio": float(frame["ratio"].median()),
        "p_value": sign_test_p_value(smoother, int(len(frame))),
    }


def coverage_by_series(jst: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Which countries in the raw workbook carry which series, and for how long."""
    block = jst[(jst["year"] >= start) & (jst["year"] <= end)]
    columns = [c for c in ("eq_tr", "bond_tr", "bill_rate", "cpi", "xrusd")
               if c in block.columns]
    rows: List[Dict[str, Any]] = []
    for iso, group in block.groupby("iso"):
        row: Dict[str, Any] = {"iso": str(iso),
                               "country": dl.ISO_TO_NAME.get(str(iso), str(iso)),
                               "years_in_file": int(len(group))}
        for column in columns:
            row[column] = int(group[column].notna().sum())
        row["complete_return_years"] = int(
            group[["eq_tr", "bond_tr", "bill_rate", "cpi"]].notna()
            .all(axis=1).sum())
        row["usable_for_returns"] = row["complete_return_years"] > 0
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(
        "complete_return_years", ascending=False).reset_index(drop=True)
