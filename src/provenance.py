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


def country_provenance(panel: dl.Panel) -> pd.DataFrame:
    """One row per country: tier, coverage, and the source of each series.

    The ``returns_source`` column is the one that matters. "JST/JKKST" means
    the country's own recorded history; "factor model" means the series was
    generated and no observation in it happened.
    """
    available = panel.available
    years = np.asarray(panel.years)
    rows: List[Dict[str, Any]] = []
    for i, iso in enumerate(panel.countries):
        column = available[:, i]
        n = int(column.sum())
        note = panel.provenance[i] if i < len(panel.provenance) else ""
        synthetic = panel.tier[i] == "B"
        donor = ""
        if "donor" in note:
            donor = note.split("donor")[1].split(";")[0].strip()
        inflation_share = _inflation_share(note)
        rows.append({
            "iso": iso,
            "country": dl.ISO_TO_NAME.get(iso, iso),
            "tier": panel.tier[i],
            "usable_years": n,
            "first_year": int(years[column].min()) if n else 0,
            "last_year": int(years[column].max()) if n else 0,
            "returns_source": "factor model" if synthetic else "JST/JKKST",
            "returns_empirical_years": 0 if synthetic else n,
            "returns_simulated_years": n if synthetic else 0,
            "inflation_empirical_share": inflation_share,
            "donor": donor,
            "note": note,
        })
    return pd.DataFrame.from_records(rows)


def _inflation_share(note: str) -> float:
    """Pull the empirical-inflation share out of a Tier-B provenance note."""
    if "% of" not in note:
        return 1.0
    try:
        return float(note.split("(")[1].split("%")[0]) / 100.0
    except (IndexError, ValueError):       # pragma: no cover - defensive
        return float("nan")


def panel_summary(panel: dl.Panel) -> Dict[str, Any]:
    """Headline shares: how much of the panel, and of the sampler, is simulated."""
    tier = np.asarray(panel.tier)
    available = panel.available
    synthetic = tier == "B"
    total = int(available.sum())
    simulated = int(available[:, synthetic].sum())
    history = available.sum(axis=0).astype(float)
    weight = history / history.sum()
    return {
        "n_countries": len(panel.countries),
        "n_empirical_countries": int((~synthetic).sum()),
        "n_simulated_countries": int(synthetic.sum()),
        "country_years": total,
        "country_years_empirical": total - simulated,
        "country_years_simulated": simulated,
        "share_country_years_simulated": simulated / total if total else 0.0,
        "share_draws_simulated": float(weight[synthetic].sum()),
    }


def simulated_share_by_era(panel: dl.Panel, edges: Sequence[int] =
                           (1890, 1930, 1970, 2000, 2021)) -> pd.DataFrame:
    """How the simulated share of the cross-section moves through time."""
    tier = np.asarray(panel.tier)
    synthetic = tier == "B"
    years = np.asarray(panel.years)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (years >= lo) & (years < hi)
        block = panel.available[mask]
        total = int(block.sum())
        rows.append({
            "era": f"{lo}-{hi - 1}",
            "country_years": total,
            "simulated": int(block[:, synthetic].sum()),
            "share_simulated": float(block[:, synthetic].sum() / total)
            if total else 0.0,
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
    synthetic = tier == "B"
    empirical = ~synthetic
    years = np.asarray(panel.years)
    rows = []
    for lo, hi in zip((edges[0],) + tuple(edges[:-1]),
                      (edges[-1],) + tuple(edges[1:])):
        mask = (years >= lo) & (years < hi)
        shares: List[float] = []
        for t in np.flatnonzero(mask):
            column = panel.available[t]
            for i in np.flatnonzero(column & empirical):
                others = column.copy()
                others[i] = False
                if others.sum():
                    shares.append(float(synthetic[others].sum() / others.sum()))
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
