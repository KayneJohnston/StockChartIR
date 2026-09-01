"""Does the equal-weighted international sleeve manufacture the headline?

Every result in this project rests on an international equity leg built as a
**leave-one-out equal-weighted** average of the other fifteen markets. That is
a defensible construction and an unusually favourable one. An equal-weighted
portfolio of sixteen national markets is a more diversified object than any
index a person could have bought: it holds as much Portugal as it holds the
United States, it rebalances into whatever has fallen, and it never lets a
single market grow to dominate. Real investors hold something closer to a
capitalisation-weighted index, which is concentrated by construction and grows
*more* concentrated exactly when one market has run.

That matters because the headline finding here -- 100% international beating a
50/50 domestic/international split, where Anarkulova, Cederburg and O'Doherty
find 50/50 -- is a claim about how much diversification the foreign sleeve
delivers. If the equal weighting is doing the work, the divergence is an
artefact of panel construction rather than a finding about the world.

This module rebuilds the panel under a GDP weighting and re-runs the headline
comparison through exactly the same code, so the two answers differ in the
weighting scheme and in nothing else. Three properties make the comparison
paired rather than merely parallel:

* ``available`` is untouched by the weighting, so both panels admit the same
  blocks and the bootstrap draws the same calendar history for each.
* The summary statistics are computed by the caller's own summariser -- the
  one that produces the headline table -- rather than reimplemented here.
* The GDP weights are **lagged one year**, so the sleeve is one an investor
  could have held rather than one assembled with hindsight.

What GDP weighting is and is not: it is a proxy for capitalisation weights,
not a substitute for them. It reproduces the concentration that makes a real
index a real index, and :func:`concentration` measures how much. It does not
reproduce the wedge between an economy's size and its listed market's size,
and being PPP-based it understates markets whose currency is temporarily
strong. It is the closer of the two constructions to what a person could buy,
and still not the thing itself.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import data_loader as dl

LOGGER = logging.getLogger(__name__)

#: The two constructions compared. ``equal`` is the project's headline panel.
WEIGHTINGS: Tuple[str, ...] = ("equal", "gdp")

#: Human labels, so the tables never print a bare config key.
LABELS: Dict[str, str] = {
    "equal": "Equal-weighted (leave-one-out)",
    "gdp": "GDP-weighted (lagged, leave-one-out)",
}


def build_panels(cfg: Mapping[str, Any],
                 weightings: Sequence[str] = WEIGHTINGS
                 ) -> Dict[str, dl.Panel]:
    """One panel per weighting scheme, identical in everything else.

    The panels share ``years``, ``countries`` and ``available`` by
    construction -- only ``intl_eq`` differs -- which is what makes the
    downstream comparison paired on identical history.
    """
    panels = {w: dl.build_tier_a(cfg, weighting=w) for w in weightings}
    reference = panels[list(panels)[0]]
    for name, panel in panels.items():
        if panel.countries != reference.countries:
            raise ValueError(f"panel {name!r} has different countries")
        if not np.array_equal(panel.available, reference.available):
            raise ValueError(
                f"panel {name!r} has different availability, so the "
                "comparison would not be paired on the same history")
    return panels


def concentration(cfg: Mapping[str, Any], countries: Sequence[str],
                  years: np.ndarray) -> pd.DataFrame:
    """How concentrated the GDP weights are, year by year.

    Reports the Herfindahl index and its reciprocal, the *effective number of
    markets*. The equal-weighted sleeve holds that number fixed at the count
    of markets; a real index does not, and the gap between the two is the
    quantity this whole section is about.
    """
    jst = dl.add_real_returns(dl.load_jst(cfg))
    window = jst[jst["year"].between(int(years[0]) - 1, int(years[-1]))]
    size = dl.economy_size(window, np.asarray(years), list(countries))
    rows: List[Dict[str, Any]] = []
    for i, year in enumerate(years):
        row = size[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0 or finite.sum() <= 0:
            continue
        share = finite / finite.sum()
        hhi = float((share ** 2).sum())
        # Index the largest market off the full row rather than the compacted
        # one, so a missing country never shifts the name off its weight.
        largest = int(np.nanargmax(np.where(np.isfinite(row), row, -np.inf)))
        rows.append({
            "year": int(year),
            "markets": int(finite.size),
            "herfindahl": hhi,
            "effective_markets": 1.0 / hhi,
            "largest_share": float(share.max()),
            "largest_market": str(countries[largest]),
            "top3_share": float(np.sort(share)[::-1][:3].sum()),
        })
    return pd.DataFrame.from_records(rows)


def sleeve_moments(panels: Mapping[str, dl.Panel]) -> pd.DataFrame:
    """Pooled moments of each sleeve, and its correlation with the home market.

    The correlation column is the mechanism. An international leg earns its
    place by being different from the domestic one, so if GDP weighting moves
    the headline it should move this first.
    """
    rows: List[Dict[str, Any]] = []
    for name, panel in panels.items():
        mask = panel.available
        intl = panel.intl_eq[mask]
        dom = panel.dom_eq[mask]
        both = np.isfinite(intl) & np.isfinite(dom)
        rows.append({
            "weighting": name,
            "label": LABELS.get(name, name),
            "observations": int(both.sum()),
            "mean": float(intl[both].mean()),
            "sd": float(intl[both].std(ddof=1)),
            "return_per_unit_risk": float(intl[both].mean()
                                          / intl[both].std(ddof=1)),
            "correlation_with_domestic": float(
                np.corrcoef(intl[both], dom[both])[0, 1]),
        })
    return pd.DataFrame.from_records(rows)


def compare_headline(
    cfg: Mapping[str, Any],
    panels: Mapping[str, dl.Panel],
    summarise: Callable[..., pd.DataFrame],
    n_paths: int,
) -> pd.DataFrame:
    """Run the headline strategy comparison once per weighting.

    ``summarise`` is the caller's own headline summariser rather than a copy
    of it living here: the point of this section is that the two panels are
    scored by identical code, and passing the function in is the only way to
    guarantee that as the summariser changes.
    """
    frames: List[pd.DataFrame] = []
    for name, panel in panels.items():
        LOGGER.info("sleeve study: simulating the %s panel", name)
        frame = summarise(panel, n_paths).copy()
        frame.insert(0, "weighting", name)
        frame.insert(1, "weighting_label", LABELS.get(name, name))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _cec_column(frame: pd.DataFrame) -> str:
    """The CRRA certainty-equivalent column, whatever gamma it was run at."""
    matches = [c for c in frame.columns if c.startswith("cec_crra_")]
    if not matches:
        raise KeyError("no CRRA certainty-equivalent column in the comparison")
    return matches[0]


def ranking_shift(comparison: pd.DataFrame) -> pd.DataFrame:
    """Every strategy's certainty equivalent under both weightings, side by side.

    The ``cost_of_gdp_weighting_pct`` column is the quantity the section
    exists to report: what the headline gives up when the sleeve stops being
    an equal-weighted portfolio nobody could have bought.
    """
    cec = _cec_column(comparison)
    wide = comparison.pivot(index=["strategy", "label"], columns="weighting",
                            values=cec).reset_index()
    if "equal" in wide and "gdp" in wide:
        wide["change_pct"] = (wide["gdp"] / wide["equal"] - 1.0) * 100.0
    ruin = comparison.pivot(index=["strategy", "label"], columns="weighting",
                            values="prob_ruin").reset_index()
    for scheme in ("equal", "gdp"):
        if scheme in ruin:
            wide[f"ruin_{scheme}"] = ruin[scheme].to_numpy()
    ordered = wide.sort_values("equal", ascending=False) \
        if "equal" in wide else wide
    return ordered.reset_index(drop=True)


def verdict(comparison: pd.DataFrame, pair: Tuple[str, str] =
            ("international_equity", "balanced_all_equity")) -> Dict[str, Any]:
    """Classify what the weighting does to the finding, rather than assert it.

    ``pair`` names the two strategies whose ordering is the divergence from
    the replicated paper: the concentrated all-international sleeve against
    the 50/50 split those authors land on. The question is not whether the
    certainty equivalents move -- they will -- but whether the *ranking* does.
    """
    cec = _cec_column(comparison)
    challenger, incumbent = pair
    out: Dict[str, Any] = {"cec_column": cec}
    for scheme in sorted(comparison["weighting"].unique()):
        block = comparison[comparison["weighting"] == scheme]
        ranked = block.sort_values(cec, ascending=False)
        top = ranked.iloc[0]
        rows = {r["strategy"]: r for _, r in block.iterrows()}
        gap = float("nan")
        if challenger in rows and incumbent in rows:
            gap = (float(rows[challenger][cec]) / float(rows[incumbent][cec])
                   - 1.0) * 100.0
        out[scheme] = {
            "winner": str(top["strategy"]),
            "winner_label": str(top["label"]),
            "winner_cec": float(top[cec]),
            "gap_pct": gap,
            "challenger_leads": bool(np.isfinite(gap) and gap > 0.0),
        }
    schemes = [k for k in out if k != "cec_column"]
    out["winner_changes"] = len({out[s]["winner"] for s in schemes}) > 1
    out["ordering_changes"] = len({out[s]["challenger_leads"]
                                   for s in schemes}) > 1
    out["survives"] = not out["ordering_changes"]
    return out
