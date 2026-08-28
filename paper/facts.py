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

    # -- the full-simplex allocation solve ---------------------------------
    @functools.cached_property
    def allocation(self) -> Dict[str, Any]:
        schedules = self.table("allocation_solved_schedules")
        comparison = self.table("allocation_comparison")
        deviation = self.table("allocation_deviation_profile")
        restarts = self.table("allocation_restarts")
        gamma = self.baseline_gamma
        base = schedules[np.isclose(schedules["risk_aversion"], gamma)]
        block = comparison[np.isclose(comparison["risk_aversion"], gamma)] \
            .sort_values("cec", ascending=False)
        solved = block[block["strategy"] == "full_simplex_optimal"]
        rival = block[block["strategy"] != "full_simplex_optimal"].iloc[0]
        dev = deviation[np.isclose(deviation["risk_aversion"], gamma)]
        working = base[base["phase"] == "working"]
        return {
            "n_ages": int(len(base)),
            "free_parameters": int(len(base) * 3),
            "lead_pct": (float(solved["cec"].iloc[0]) / float(rival["cec"])
                         - 1.0) * 100.0 if len(solved) else float("nan"),
            "runner_up": str(rival["strategy"]),
            "mean_equity": float(base["equity"].mean()),
            "mean_working_equity": float(working["equity"].mean()),
            "mean_dom_eq": float(base["dom_eq"].mean()),
            "mean_intl_eq": float(base["intl_eq"].mean()),
            "mean_bond": float(base["bond"].mean()),
            "mean_bill": float(base["bill"].mean()),
            "n_material_ages": int((dev["cost_of_resetting_bp"].abs() > 1.0).sum()),
            "restart_spread_pct": float(restarts["gap_to_best_pct"].abs().max()),
        }

    # -- the leverage study -------------------------------------------------
    @functools.cached_property
    def leverage(self) -> Dict[str, Any]:
        sweep = self.table("leverage_sweep")
        optimal = self.table("leverage_optimal_by_cost").sort_values("spread")
        detail = self.table("leverage_outcome_detail")
        free = optimal[np.isclose(optimal["spread"], 0.0)]
        free_row = free.iloc[0] if len(free) else optimal.iloc[0]
        unlevered = optimal[np.isclose(optimal["leverage"], 1.0)]
        base = detail[np.isclose(detail["leverage"], 1.0)].iloc[0]
        top = detail.loc[detail["leverage"].idxmax()]
        # The lowest levered row, which is the one closest to any ratio an
        # investor would actually run.
        levered = detail[detail["leverage"] > 1.0].sort_values("leverage")
        moderate = levered.iloc[0] if len(levered) else top

        # Recomputed here rather than read back, so the paper and the pipeline
        # cannot disagree about where the crossing is.
        advantage = optimal["vs_unlevered_pct"].to_numpy(dtype=float)
        spreads = optimal["spread"].to_numpy(dtype=float)
        crossing = np.flatnonzero(advantage <= 1e-9)
        if advantage.size == 0 or advantage[0] <= 1e-9:
            break_even = 0.0
        elif crossing.size == 0:
            break_even = float("inf")
        else:
            i = int(crossing[0])
            lo, hi = advantage[i - 1], advantage[i]
            break_even = float(spreads[i] if lo == hi else
                               spreads[i - 1] + lo / (lo - hi)
                               * (spreads[i] - spreads[i - 1]))
        return {
            "optimal_at_zero": float(free_row["leverage"]),
            "value_at_zero_spread": float(free_row["vs_unlevered_pct"]),
            "equity_at_zero": float(free_row["equity"]),
            "effective_equity_at_zero": float(free_row["equity"])
            * float(free_row["leverage"]),
            "break_even_spread": break_even,
            # Where the advantage stops being worth the trouble, which can be
            # far below where it formally reaches zero.
            "negligible_spread": float(
                optimal[optimal["vs_unlevered_pct"] < 0.10]["spread"].min())
            if (optimal["vs_unlevered_pct"] < 0.10).any() else float("inf"),
            "first_unlevered_spread": float(unlevered["spread"].min())
            if len(unlevered) else float("nan"),
            # Across both tables: the sweep re-optimises the allocation at
            # every ratio and so never gets wiped out, while the detail table
            # holds one allocation fixed and does. Quoting only the first
            # would contradict the second on the page.
            "max_wipeout": float(max(sweep["wipeout_share_of_years"].max(),
                                     detail["wipeout_share_of_years"].max())),
            "moderate_leverage": float(moderate["leverage"]),
            "moderate_p5_pct": (float(moderate["p5_retirement_consumption"])
                                / float(base["p5_retirement_consumption"])
                                - 1.0) * 100.0,
            "moderate_median_pct": (
                float(moderate["median_retirement_consumption"])
                / float(base["median_retirement_consumption"]) - 1.0) * 100.0,
            "top_leverage": float(top["leverage"]),
            "p5_change_pct": (float(top["p5_retirement_consumption"])
                              / float(base["p5_retirement_consumption"]) - 1.0)
            * 100.0,
            "median_change_pct": (float(top["median_retirement_consumption"])
                                  / float(base["median_retirement_consumption"])
                                  - 1.0) * 100.0,
            "p95_change_pct": (float(top["p95_retirement_consumption"])
                               / float(base["p95_retirement_consumption"]) - 1.0)
            * 100.0,
            "ruin_base": float(base["prob_ruin"]),
            "ruin_top": float(top["prob_ruin"]),
        }

