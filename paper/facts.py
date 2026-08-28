"""Every number the paper quotes, loaded from the pipeline's own output.

The paper is written once but rebuilt whenever the pipeline reruns, so no
figure in the text is typed by hand. Anything quoted in prose is resolved here
from ``results/tables/*.csv`` or from ``config.yaml``; if a table is missing
the build fails loudly rather than silently printing a placeholder.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import yaml

TABLES = Path("results/tables")
FIGURES = Path("results/figures")


def pct(value: float, digits: int = 1) -> str:
    """A percentage with an explicit sign, for value comparisons."""
    return f"{value:+.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


@dataclasses.dataclass
class Facts:
    """Lazy access to the pipeline's tables plus the derived scalars."""

    config_path: str = "config.yaml"
    _cache: Dict[str, pd.DataFrame] = dataclasses.field(
        default_factory=dict, repr=False)

    @functools.cached_property
    def cfg(self) -> Mapping[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = TABLES / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing; run `python main.py` before building "
                    "the paper so every quoted number comes from a live result")
            self._cache[name] = pd.read_csv(path)
        return self._cache[name]

    def figure(self, name: str) -> str:
        path = FIGURES / f"{name}.png"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; rerun the pipeline")
        return str(path)

    # -- convenience selectors --------------------------------------------
    @property
    def baseline_gamma(self) -> float:
        return float(self.cfg["utility"]["baseline_risk_aversion"])

    @property
    def headline(self) -> pd.DataFrame:
        return self.table("headline_lifecycle_metrics")

    def strategy_row(self, key: str) -> pd.Series:
        block = self.headline[self.headline["strategy"] == key]
        if block.empty:
            raise KeyError(f"strategy {key!r} not in the headline table")
        return block.iloc[0]

    def cec(self, key: str, gamma: float | None = None) -> float:
        g = self.baseline_gamma if gamma is None else gamma
        return float(self.strategy_row(key)[f"cec_crra_gamma{g:g}"])

    def advantage(self, challenger: str, incumbent: str,
                  gamma: float | None = None) -> float:
        """Percentage certainty-equivalent advantage of one strategy over another."""
        return (self.cec(challenger, gamma) / self.cec(incumbent, gamma) - 1.0) * 100.0

    # -- panel -------------------------------------------------------------
    @functools.cached_property
    def panel(self) -> Dict[str, Any]:
        summary = self.table("panel_summary_statistics")
        countries = summary[["iso", "country", "tier"]].drop_duplicates()
        equity = summary[summary["series"] == "dom_eq"]
        return {
            "n_countries": int(countries["iso"].nunique()),
            "n_tier_a": int((countries["tier"] == "A").sum()),
            "n_tier_b": int((countries["tier"] == "B").sum()),
            "first_year": int(summary["first_year"].min()),
            "last_year": int(summary["last_year"].max()),
            "country_years": int(equity["n_years"].sum()),
            "median_years": float(equity["n_years"].median()),
            "min_years": int(equity["n_years"].min()),
            "mean_real_equity": float(equity["mean"].mean()),
            "mean_geometric_equity": float(equity["geometric_mean"].mean()),
            "mean_equity_sd": float(equity["std"].mean()),
        }

    # -- bootstrap ---------------------------------------------------------
    @functools.cached_property
    def blocks(self) -> Dict[str, float]:
        frame = self.table("bootstrap_blocks")
        return {str(r.statistic): float(r.value) for r in frame.itertuples()}

    # -- the accumulation study -------------------------------------------
    @functools.cached_property
    def shape_value(self) -> float:
        """What the solved age profile alone earns, so conditioning can net it out."""
        best = self.table("acc_signal_best")
        row = best[best["signal"] == "none"]
        return float(row["matched_value_pct"].iloc[0]) if len(row) else 0.0

    def acc_net(self, frame: pd.DataFrame, mask: Any = None) -> float:
        block = frame if mask is None else frame[mask]
        return float(block["matched_value_pct"].max()) - self.shape_value

    @functools.cached_property
    def signals(self) -> pd.DataFrame:
        best = self.table("acc_signal_best").copy()
        best["net"] = best["matched_value_pct"] - self.shape_value
        return best.sort_values("net", ascending=False)

    def signal_net(self, key: str) -> float:
        row = self.signals[self.signals["signal"] == key]
        if row.empty:
            raise KeyError(f"signal {key!r} not in the horse race")
        return float(row["net"].iloc[0])

    def signal_label(self, key: str) -> str:
        row = self.signals[self.signals["signal"] == key]
        return str(row["signal_label"].iloc[0])
