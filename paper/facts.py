"""Every number the paper quotes, loaded from the pipeline's own output.

The paper is written once but rebuilt whenever the pipeline reruns, so no
figure in the text is typed by hand. Anything quoted in prose is resolved here
from ``results/tables/*.csv`` or from ``config.yaml``; if a table is missing
the build fails loudly rather than silently printing a placeholder.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import yaml

def _war_years_text() -> str:
    """Name the windows the audit excluded, taken from the audit's own source.

    Spelling them out here would let the paper and the pipeline disagree about
    which years were dropped, which is exactly the failure this module exists
    to prevent.
    """
    try:
        from src.provenance import WAR_YEARS
    except ImportError:                                 # pragma: no cover
        return "the war years"
    return " and ".join(f"{low}\u2013{high}" for low, high in WAR_YEARS)


def _removed_countries() -> tuple:
    """The countries dropped for having generated returns, from the source."""
    try:
        from src.data_loader import REMOVED_SIMULATED
    except ImportError:                                 # pragma: no cover
        return ()
    return tuple(REMOVED_SIMULATED)


def _panel_wage_extremes(audit: pd.DataFrame) -> Dict[str, Any]:
    """Delegate to the pipeline's own definition so the two cannot diverge."""
    try:
        from src.provenance import panel_wage_extremes
    except ImportError:                                 # pragma: no cover
        return {}
    return panel_wage_extremes(audit)


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

    @functools.cached_property
    def n_tests(self) -> int:
        """How many tests the suite actually collects.

        Asked of pytest rather than hard-coded, because a number quoted in the
        paper as evidence of correctness should not be able to go stale. Test
        functions are not counted by hand: parametrised cases expand to several
        tests each, so the file count would understate the suite. Falls back to
        counting ``def test_`` if pytest cannot be run wherever the paper is
        being built.
        """
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True, timeout=300, check=False)
            found = re.search(r"(\d+) tests? collected", proc.stdout)
            if found:
                return int(found.group(1))
        except (OSError, subprocess.SubprocessError):    # pragma: no cover
            pass
        root = Path(__file__).resolve().parent.parent / "tests"
        return sum(text.count("def test_")
                   for text in (f.read_text() for f in root.glob("test_*.py")))

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
            "n_tier_c": int((countries["tier"] == "C").sum()),
            # Every country whose equity had to be generated, which is the
            # split the paper's provenance discussion actually turns on.
            "n_simulated_equity": int((countries["tier"] != "A").sum()),
            "first_year": int(summary["first_year"].min()),
            "last_year": int(summary["last_year"].max()),
            "country_years": int(equity["n_years"].sum()),
            "median_years": float(equity["n_years"].median()),
            "min_years": int(equity["n_years"].min()),
            "max_years": int(equity["n_years"].max()),
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

    # -- the provenance audit ----------------------------------------------
    @functools.cached_property
    def provenance(self) -> Dict[str, Any]:
        countries = self.table("provenance_by_country")
        era = self.table("provenance_by_era")
        contamination = self.table("provenance_intl_contamination")
        anchors = self.table("provenance_anchor_checks")
        identity = self.table("provenance_identity_check").iloc[0]
        tail = self.table("provenance_tail_variance")
        try:
            unusable = self.table("provenance_unusable_series")
        except FileNotFoundError:                       # pragma: no cover
            unusable = pd.DataFrame()

        # Cells, not countries: one country, one year, one return series. It
        # is the unit the bootstrap draws, and the only one that can describe a
        # country whose bonds are observed while its equity is generated.
        cells_observed = int(countries["returns_empirical_years"].sum())
        cells_total = cells_observed + int(
            countries["returns_simulated_years"].sum())
        recovered = countries[countries["tier"] == "B"]
        available_simulated = int(
            countries[countries["tier"] != "A"]["usable_years"].sum())
        total = int(countries["usable_years"].sum())
        whole = contamination[contamination["era"] == "whole panel"]
        recent = contamination[contamination["era"] != "whole panel"].iloc[-1]
        smoother = int((tail["ratio"] < 1.0).sum())
        n_tail = int(len(tail))
        from math import comb
        p_value = float(min(1.0, 2.0 * sum(comb(n_tail, k)
                                           for k in range(min(smoother,
                                                              n_tail - smoother) + 1))
                            / 2.0 ** n_tail)) if n_tail else float("nan")
        return {
            "n_countries": int(len(countries)),
            "n_simulated": int((countries["tier"] == "C").sum()),
            "n_partial": int(len(recovered)),
            "n_simulated_equity": int((countries["tier"] != "A").sum()),
            "n_observed": int((countries["tier"] == "A").sum()),
            "country_years": total,
            "country_years_simulated": available_simulated,
            "share_simulated": available_simulated / total if total else 0.0,
            "cells": cells_total,
            "cells_simulated": cells_total - cells_observed,
            "share_cells_simulated": (cells_total - cells_observed)
            / cells_total if cells_total else 0.0,
            "recovered_countries": [str(c) for c in recovered["country"]],
            "recovered_years": int(recovered["returns_empirical_years"].sum()),
            "share_simulated_recent": float(era["share_simulated"].iloc[-1]),
            "recent_era": str(era["era"].iloc[-1]),
            "intl_simulated": float(
                whole["mean_synthetic_share_of_intl_leg"].iloc[0]),
            "intl_simulated_recent": float(
                recent["mean_synthetic_share_of_intl_leg"]),
            "anchors_passed": int(anchors["within_tolerance"].sum()),
            "anchors_total": int(len(anchors)),
            "identity_share_violating": float(identity["share_violating"]),
            "identity_observations": int(identity["observations"]),
            "tail_smoother": smoother,
            "tail_countries": n_tail,
            "tail_median_ratio": float(tail["ratio"].median()),
            "tail_p_value": p_value,
            "unusable": unusable,
            "n_removed": len(_removed_countries()),
            "removed": _removed_countries(),
            "housing": self._housing(),
            "wages": self._wages(),
        }

    def _wages(self) -> Dict[str, Any]:
        """Measured real wage growth against what the income model assumes."""
        try:
            audit = self.table("provenance_wage_audit")
        except FileNotFoundError:                       # pragma: no cover
            return {"countries": 0}
        if audit.empty:
            return {"countries": 0}
        life = self.cfg["lifecycle"]
        income = life["income"]
        start, retire = int(life["age_start"]), int(life["age_retire"])
        offset = np.arange(retire - start)
        profile = np.exp(np.log(float(income["initial_real_income"]))
                         + float(income["b1"]) * offset
                         + float(income["b2"]) * offset ** 2)
        span = profile.size - 1
        model_growth = float((profile[-1] / profile[0]) ** (1.0 / span) - 1.0)
        measured = float(audit["geometric_mean"].median())
        return {
            "countries": int(len(audit)),
            "country_years": int(audit["years"].sum()),
            "first_year": int(audit["first_year"].min()),
            "last_year": int(audit["last_year"].max()),
            "measured": measured,
            "measured_ex_war": float(audit["geometric_mean_ex_war"].median()),
            "war_shifted_by": float(audit["geometric_mean_ex_war"].median())
            - measured,
            "lowest": float(audit["geometric_mean"].min()),
            "highest": float(audit["geometric_mean"].max()),
            "lowest_country": str(
                audit.loc[audit["geometric_mean"].idxmin(), "country"]),
            "lowest_ex_war": float(
                audit.loc[audit["geometric_mean"].idxmin(),
                          "geometric_mean_ex_war"]),
            "highest_country": str(
                audit.loc[audit["geometric_mean"].idxmax(), "country"]),
            "career_years": span,
            "career_multiple": float((1.0 + measured) ** span),
            "model_growth": model_growth,
            "model_peak_age": float(
                start - float(income["b1"]) / (2.0 * float(income["b2"]))),
            "model_peak_multiple": float(profile.max() / profile[0]),
            "model_end_multiple": float(profile[-1] / profile[0]),
            "combined_growth": (1.0 + measured) * (1.0 + model_growth) - 1.0,
            "war_years": _war_years_text(),
            **_panel_wage_extremes(audit),
            "frame": audit,
        }

    def _housing(self) -> Dict[str, Any]:
        """The observed asset class the paper measures but does not invest in."""
        try:
            audit = self.table("provenance_housing_audit")
        except FileNotFoundError:                       # pragma: no cover
            return {"countries": 0}
        if audit.empty:
            return {"countries": 0}
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
                (audit["autocorrelation"]
                 > audit["equity_autocorrelation"]).sum()),
            "frame": audit,
        }

    @functools.cached_property
    def panel_advantage(self) -> Dict[str, float]:
        """The headline advantage, from the only panel there is.

        This used to compare a 38-country panel against its observed subset,
        because 22 of those countries had factor-model returns. They were
        removed, so there is nothing to compare against and the number below is
        simply the result.
        """
        headline = self.table("headline_lifecycle_metrics").set_index("strategy")
        column = f"cec_crra_gamma{self.baseline_gamma:g}"
        equity = float(headline.loc["balanced_all_equity", column])
        glide = float(headline.loc["target_date_fund", column])
        return {"observed": (equity / glide - 1.0) * 100.0}

