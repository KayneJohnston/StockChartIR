"""End-to-end pipeline for the ACO lifecycle replication.

Runs the four steps in order -- panel extraction, block bootstrap, lifecycle
simulation, reporting -- and writes every table, figure and Markdown document
the project promises.

    python main.py                      # full run using config.yaml
    python main.py --config other.yaml  # alternative configuration
    python main.py --quick              # small N, for smoke-testing
    python main.py --steps 1 2          # run only some steps
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src import accumulation as acc
from src import allocation as al
from src import bootstrap as bs
from src import data_loader as dl
from src import fees as fee
from src import glidepath as gp
from src import hedging as hg
from src import housing as hsg
from src import leverage as lev
from src import lifecycle as lc
from src import mortgage as mgg
from src import observed as obs
from src import panel_robustness as pr
from src import plots
from src import provenance as pvn
from src import valuation as vln
from src import report as rp
from src import retirement as rt
from src import sleeve as slv
from src import saving as sav
from src import sensitivity as sn
from src import spending as spg
from src import utility as ut

LOGGER = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _save_table(frame: pd.DataFrame, directory: str | Path, name: str,
                index: bool = False) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=index)
    LOGGER.info("wrote table %s", path)
    return path


def _apply_quick(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg["bootstrap"]["n_paths"] = 5000
    cfg["bootstrap"]["chunk_size"] = 2500
    cfg["bootstrap"]["diagnostics"]["n_paths"] = 4000
    if "sensitivity" in cfg:
        cfg["sensitivity"]["n_paths"] = 4000
        cfg["sensitivity"]["redraw_n_paths"] = 2000
    if "saving" in cfg:
        cfg["saving"]["n_paths"] = 2000
        cfg["saving"]["shape_sweeps"] = 1
    if "allocation" in cfg:
        cfg["allocation"]["n_paths"] = 2000
        cfg["allocation"]["coarse_sweeps"] = 1
        cfg["allocation"]["fine_sweeps"] = 1
        cfg["allocation"]["restart_starts"] = [[0.25, 0.25, 0.25, 0.25],
                                               [0.5, 0.5, 0.0, 0.0]]
    if "leverage" in cfg:
        cfg["leverage"]["n_paths"] = 2000
        cfg["leverage"]["leverage_grid"] = [1.0, 1.5, 2.0]
        cfg["leverage"]["spread_grid"] = [0.0, 0.01, 0.03]
        cfg["leverage"]["detail_leverage"] = [1.0, 1.5, 2.0]
        cfg["leverage"]["schedule"]["spreads"] = [0.0]
        cfg["leverage"]["schedule"]["grid"] = [1.0, 1.5, 2.0]
        cfg["leverage"]["schedule"]["sweeps"] = 1
    if "housing" in cfg:
        cfg["housing"]["n_paths"] = 2000
        cfg["housing"]["holding_costs"] = [0.0, 0.02, 0.04]
        cfg["housing"]["coarse_step"] = 0.25
        cfg["housing"]["fine_step"] = 0.05
        cfg["housing"]["age_varying_costs"] = [0.0]
    if "mortgage" in cfg:
        cfg["mortgage"]["n_paths"] = 2000
        cfg["mortgage"]["spreads"] = [0.0, 0.02, 0.04]
        cfg["mortgage"]["lvr_grid"] = [0.0, 0.2, 0.4, 0.6, 0.8]
        cfg["mortgage"]["coarse_step"] = 0.25
        cfg["mortgage"]["fine_step"] = 0.05
        cfg["mortgage"]["rounds"] = 1
    if "sleeve" in cfg:
        cfg["sleeve"]["n_paths"] = 2000
    if "fees" in cfg:
        cfg["fees"]["n_paths"] = 2000
        cfg["fees"]["common_grid"] = [0.0, 0.005]
        cfg["fees"]["differential_grid"] = [0.0, 0.002, 0.005]
    if "panel_robustness" in cfg:
        cfg["panel_robustness"]["n_paths"] = 1000
        cfg["panel_robustness"]["seeds"] = [101, 202]
        cfg["panel_robustness"]["windows"] = [1950, 2020]
    if "accumulation" in cfg:
        cfg["accumulation"]["n_paths"] = 2000
        cfg["accumulation"]["response_grids"] = {
            "level": [0.0, 0.01], "proportional": [0.0, 0.1],
            "log": [0.0, 0.1]}
        cfg["accumulation"]["asymmetry"] = {"behind": [0.0, 0.1],
                                            "ahead": [-0.05, 0.0, 0.1]}
        cfg["accumulation"]["bands"] = [0.20]
        cfg["accumulation"]["band_steps"] = [0.02, 0.05]
        cfg["accumulation"]["target_factors"] = [1.0, 1.5]
        cfg["accumulation"]["signal_grid"] = [0.0, 0.1]
        cfg["accumulation"]["combination_grid"] = [0.0, 0.1]
        cfg["accumulation"]["feasibility_widths"] = [0.0, 0.05, 0.30]
        cfg["accumulation"]["age_windows"] = [[25, 39], [40, 62], [25, 62]]
        cfg["accumulation"]["constant_rate_grid"] = [0.0, 0.05, 0.10, 0.20]
        cfg["accumulation"]["strategy_interaction"] = ["balanced_all_equity",
                                                       "bills_only"]
        cfg["accumulation"]["income_volatility_factors"] = [0.5, 1.5]
    if "retirement_timing" in cfg:
        cfg["retirement_timing"]["n_paths"] = 4000
    if "hedging" in cfg:
        cfg["hedging"]["n_paths"] = 4000
    if "glide_path" in cfg:
        cfg["glide_path"]["n_paths"] = 2000
        cfg["glide_path"]["n_sweeps"] = 1
        cfg["glide_path"]["restart_equity_starts"] = [0.6]
    LOGGER.warning("quick mode: n_paths reduced to %s",
                   cfg["bootstrap"]["n_paths"])
    return cfg


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------
def find_source_gaps(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Contiguous runs of missing JST return data, by country."""
    jst = dl.add_real_returns(dl.load_jst(cfg))
    start, end = int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"])
    jst = jst[jst["year"].between(start, end)]
    rows: List[Dict[str, Any]] = []
    for iso in dl.TIER_A_ISO:
        sub = jst[jst["iso"] == iso].sort_values("year")
        missing = sub.loc[
            sub[["dom_eq", "bond", "bill", "inflation"]].isna().any(axis=1),
            "year"].to_numpy()
        if missing.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(missing) > 1)
        starts = np.concatenate([[0], breaks + 1])
        stops = np.concatenate([breaks, [missing.size - 1]])
        for a, b in zip(starts, stops):
            first, last = int(missing[a]), int(missing[b])
            which = sub[sub["year"].between(first, last)][
                ["dom_eq", "bond", "bill", "inflation"]]
            fields = [c for c in which.columns if which[c].isna().any()]
            rows.append({
                "iso": iso,
                "country": dl.ISO_TO_NAME.get(iso, iso),
                "first_year": first,
                "last_year": last,
                "n_years": last - first + 1,
                "missing_series": ", ".join(fields),
            })
    return pd.DataFrame.from_records(rows).sort_values(["iso", "first_year"])


def find_winsorised(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Country-years whose international leg the winsoriser actually clipped."""
    pct = float(cfg["data"].get("international_winsor_pct", 0.0))
    if pct <= 0:
        return pd.DataFrame(columns=["year", "iso", "raw_intl_eq",
                                     "winsorised_intl_eq"])
    raw_cfg = {k: (dict(v) if isinstance(v, dict) else v)
               for k, v in cfg.items()}
    raw_cfg["data"] = dict(cfg["data"])
    raw_cfg["data"]["international_winsor_pct"] = 0.0
    raw = dl.build_tier_a(raw_cfg)
    clipped = dl.build_tier_a(cfg)
    diff = ~np.isclose(np.nan_to_num(raw.intl_eq), np.nan_to_num(clipped.intl_eq),
                       rtol=0, atol=1e-12)
    diff &= raw.available
    rows = []
    for t, c in zip(*np.nonzero(diff)):
        rows.append({
            "year": int(raw.years[t]),
            "iso": raw.countries[c],
            "country": dl.ISO_TO_NAME.get(raw.countries[c], raw.countries[c]),
            "raw_intl_eq": float(raw.intl_eq[t, c]),
            "winsorised_intl_eq": float(clipped.intl_eq[t, c]),
        })
    return pd.DataFrame.from_records(rows).sort_values(["year", "iso"])


def step1_dataset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the panels and write docs/01."""
    LOGGER.info("=== STEP 1: country dataset ===")
    panels = dl.prepare_panels(cfg)
    headline_mode = cfg["bootstrap"]["panel"]
    panel = panels[headline_mode]

    summary = dl.summary_statistics(panel)
    coverage = dl.coverage_matrix(panel, decade=True)
    correlations = dl.correlation_matrices(panel)
    gaps = find_source_gaps(cfg)
    winsorised = find_winsorised(cfg)

    tables = cfg["run"]["table_dir"]
    _save_table(summary, tables, "panel_summary_statistics")
    _save_table(coverage.reset_index(names="iso"), tables, "panel_coverage")
    _save_table(correlations["cross_asset"].reset_index(names="series"),
                tables, "panel_cross_asset_correlation")
    _save_table(correlations["cross_country_equity"].reset_index(names="iso"),
                tables, "panel_cross_country_equity_correlation")
    _save_table(gaps, tables, "panel_structural_gaps")
    _save_table(winsorised, tables, "panel_winsorised_observations")

    monthly_shape = None
    if cfg["data"].get("emit_monthly", False):
        monthly = dl.to_monthly(panel, seed=int(cfg["data"]["monthly_seed"]))
        out = Path(cfg["run"]["processed_dir"]) / f"panel_monthly_{panel.name}.npz"
        np.savez_compressed(out, **monthly,
                            countries=np.array(panel.countries))
        monthly_shape = monthly["dom_eq"].shape
        LOGGER.info("wrote monthly array %s %s", out, monthly_shape)

    figures = [
        str(plots.plot_coverage(coverage, cfg["run"]["figure_dir"])),
        str(plots.plot_country_returns(summary, cfg["run"]["figure_dir"])),
    ]

    rp.write_doc_01(
        Path("docs") / "01_country_dataset_and_sources.md",
        cfg, panel, summary, coverage, correlations, gaps, winsorised,
        monthly_shape, figures,
    )
    LOGGER.info("docs/01 written")
    return {"panels": panels, "panel": panel, "summary": summary,
            "figures": figures}


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------
def step2_bootstrap(cfg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Run the bootstrap validation battery and write docs/02."""
    LOGGER.info("=== STEP 2: cross-country block bootstrap ===")
    panel = state["panel"]
    diag_cfg = cfg["bootstrap"]["diagnostics"]
    sampler = bs.from_config(panel, cfg)

    started = time.perf_counter()
    diagnostics = bs.diagnose(
        sampler,
        n_paths=int(diag_cfg["n_paths"]),
        chunk_size=int(cfg["bootstrap"]["chunk_size"]),
        percentiles=cfg["report"]["percentiles"],
    )
    LOGGER.info("diagnostics on %s paths in %.1fs",
                diag_cfg["n_paths"], time.perf_counter() - started)

    sensitivity = bs.block_length_sensitivity(
        panel, cfg, grid=diag_cfg["block_length_grid"],
        n_paths=int(diag_cfg["n_paths"]),
        chunk_size=int(cfg["bootstrap"]["chunk_size"]),
    )

    variants: Dict[str, pd.DataFrame] = {}
    for draw in cfg["report"].get("robustness_country_draw", []):
        alt = bs.from_config(panel, cfg, country_draw=draw)
        moments = bs.diagnose(alt, n_paths=int(diag_cfg["n_paths"]),
                              chunk_size=int(cfg["bootstrap"]["chunk_size"]))
        variants[f"country_draw={draw}"] = moments["moments"]
    for weighting in cfg["report"].get("robustness_country_weighting", []):
        alt = bs.from_config(panel, cfg, country_weighting=weighting)
        moments = bs.diagnose(alt, n_paths=int(diag_cfg["n_paths"]),
                              chunk_size=int(cfg["bootstrap"]["chunk_size"]))
        variants[f"country_weighting={weighting}"] = moments["moments"]

    tables = cfg["run"]["table_dir"]
    _save_table(diagnostics["moments"], tables, "bootstrap_moments")
    _save_table(diagnostics["autocorrelation"], tables,
                "bootstrap_autocorrelation")
    _save_table(diagnostics["correlation"].reset_index(names="series"),
                tables, "bootstrap_correlation")
    _save_table(diagnostics["terminal"], tables, "bootstrap_terminal")
    _save_table(diagnostics["countries"], tables, "bootstrap_countries")
    _save_table(diagnostics["blocks"], tables, "bootstrap_blocks")
    _save_table(sensitivity, tables, "bootstrap_block_sensitivity")

    figures = [
        str(plots.plot_bootstrap_diagnostics(diagnostics,
                                             cfg["run"]["figure_dir"])),
        str(plots.plot_block_sensitivity(sensitivity, cfg["run"]["figure_dir"])),
    ]

    rp.write_doc_02(
        Path("docs") / "02_multicountry_block_bootstrap.md",
        cfg, panel, diagnostics, sensitivity, variants, figures,
    )
    LOGGER.info("docs/02 written")
    state.update({"sampler": sampler, "diagnostics": diagnostics,
                  "sensitivity": sensitivity})
    return state


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def metrics_table(results: Mapping[str, lc.LifecycleOutcome],
                  cfg: Mapping[str, Any], spec: lc.LifecycleSpec
                  ) -> pd.DataFrame:
    """One row per strategy: preferences plus shortfall statistics."""
    replacement = float(cfg["report"].get("consumption_target_replacement", 0.70))
    rows: List[Dict[str, Any]] = []
    for key, outcome in results.items():
        bundle = ut.bundle_from_outcome(outcome, cfg, spec)
        # Strategy-invariant benchmark: a fixed replacement rate on the
        # investor's own career-average earnings.  Labour income is shared
        # across strategies, so this target is identical path-by-path.
        target = replacement * outcome.career_average_income
        row: Dict[str, Any] = {"strategy": key, "label": outcome.label}
        row.update(ut.evaluate_preferences(bundle, cfg))
        row.update(ut.shortfall_metrics(
            bundle, outcome.ruin, outcome.wealth_at_retirement,
            slice(0, bundle.horizon),
            percentiles=cfg["report"]["percentiles"],
            consumption_target=target,
        ))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def distribution_tables(results: Mapping[str, lc.LifecycleOutcome],
                        cfg: Mapping[str, Any], spec: lc.LifecycleSpec
                        ) -> Dict[str, pd.DataFrame]:
    """Percentile tables for wealth, bequests and retirement consumption."""
    percentiles = cfg["report"]["percentiles"]
    out: Dict[str, pd.DataFrame] = {}
    specs = {
        "wealth_at_retirement": lambda o: o.wealth_at_retirement,
        "bequest": lambda o: o.bequest,
        "retirement_consumption":
            lambda o: o.consumption[:, spec.retirement_slice].mean(axis=1),
    }
    for name, getter in specs.items():
        rows = []
        for key, outcome in results.items():
            values = getter(outcome)
            row: Dict[str, Any] = {"strategy": key, "label": outcome.label,
                                   "mean": float(values.mean())}
            for q in percentiles:
                row[f"p{q:g}"] = float(np.percentile(values, q))
            rows.append(row)
        out[name] = pd.DataFrame.from_records(rows)
    return out


def tier_breakdown(results: Mapping[str, lc.LifecycleOutcome],
                   sampler: bs.MultiCountryBlockBootstrap,
                   cfg: Mapping[str, Any], spec: lc.LifecycleSpec,
                   n_paths: int, chunk_size: int) -> pd.DataFrame:
    """Split headline metrics by the tier of the drawn domestic country."""
    tiers: List[np.ndarray] = []
    for chunk in sampler.chunks(n_paths, chunk_size):
        tiers.append(chunk.domestic_country[:, 0])
    country_of_path = np.concatenate(tiers)
    tier_of_country = np.array(sampler.panel.tier)
    path_tier = tier_of_country[country_of_path]

    rows: List[Dict[str, Any]] = []
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    for tier in ("A", "B", "C"):
        mask = path_tier == tier
        if mask.sum() < 100:
            continue
        for key, outcome in results.items():
            sub = lc.LifecycleOutcome(
                strategy=outcome.strategy, label=outcome.label,
                consumption=outcome.consumption[mask],
                wealth=outcome.wealth[mask],
                portfolio_return=outcome.portfolio_return[mask],
                wealth_at_retirement=outcome.wealth_at_retirement[mask],
                bequest=outcome.bequest[mask], ruin=outcome.ruin[mask],
                ruin_age=outcome.ruin_age[mask],
                social_security=outcome.social_security[mask],
                career_average_income=outcome.career_average_income[mask],
            )
            bundle = ut.bundle_from_outcome(sub, cfg, spec)
            rows.append({
                "tier": tier,
                "n_paths": int(mask.sum()),
                "strategy": key,
                "label": outcome.label,
                f"cec_crra_gamma{gamma:g}": ut.crra_certainty_equivalent(
                    bundle, gamma, float(cfg["utility"]["discount_factor"]),
                    float(cfg["utility"]["bequest_weight"]),
                    bool(cfg["utility"]["bequest_enabled"])),
                "prob_ruin": float(sub.ruin.mean()),
                "median_bequest": float(np.median(sub.bequest)),
            })
    return pd.DataFrame.from_records(rows)


def step3_lifecycle(cfg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate every strategy and write docs/03."""
    LOGGER.info("=== STEP 3: lifecycle simulation ===")
    spec = lc.spec_from_config(cfg)
    strategies = lc.build_strategies(cfg, spec)
    sampler = state.get("sampler") or bs.from_config(state["panel"], cfg)

    n_paths = int(cfg["bootstrap"]["n_paths"])
    chunk_size = int(cfg["bootstrap"]["chunk_size"])
    started = time.perf_counter()
    results = lc.run_chunked(sampler, strategies, spec, n_paths, chunk_size,
                             income_seed=int(cfg["run"]["seed"]))
    LOGGER.info("simulated %s lifetimes x %s strategies in %.1fs",
                f"{n_paths:,}", len(strategies), time.perf_counter() - started)

    glide = lc.glide_path_table(strategies, spec)
    profile = spec.deterministic_income()
    income_profile = pd.DataFrame({
        "age": spec.ages[:spec.n_working][::5],
        "real_income": profile[::5],
        "multiple_of_age25": profile[::5] / profile[0],
    })
    grid = np.linspace(0.2, 2.5, 12) * float(profile.mean())
    ss_schedule = pd.DataFrame({
        "career_average_earnings": grid,
        "annual_benefit": spec.social_security_benefit(grid),
        "replacement_rate": spec.social_security_benefit(grid) / grid,
    })

    tables = cfg["run"]["table_dir"]
    _save_table(glide.reset_index(), tables, "strategy_glide_paths")
    _save_table(income_profile, tables, "income_profile")
    _save_table(ss_schedule, tables, "social_security_schedule")

    figures = [str(plots.plot_glide_paths(
        glide, {k: s.label for k, s in strategies.items()},
        cfg["run"]["figure_dir"]))]

    rp.write_doc_03(
        Path("docs") / "03_lifecycle_utility_model.md",
        cfg, spec, strategies, glide, income_profile, ss_schedule, figures,
    )
    LOGGER.info("docs/03 written")
    state.update({"spec": spec, "strategies": strategies, "results": results,
                  "sampler": sampler})
    return state


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------
def _run_variant(cfg: Dict[str, Any], panel: dl.Panel,
                 spec: lc.LifecycleSpec,
                 strategies: Mapping[str, lc.Strategy],
                 n_paths: int, **overrides: Any) -> pd.DataFrame:
    """Re-run the simulation with one configuration change and summarise."""
    sampler = bs.from_config(panel, cfg, **overrides)
    results = lc.run_chunked(sampler, strategies, spec, n_paths,
                             int(cfg["bootstrap"]["chunk_size"]),
                             income_seed=int(cfg["run"]["seed"]))
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    psi = float(cfg["utility"]["baseline_ies"])
    rows = []
    for key, outcome in results.items():
        bundle = ut.bundle_from_outcome(outcome, cfg, spec)
        rows.append({
            "strategy": key,
            "label": outcome.label,
            f"cec_crra_gamma{gamma:g}": ut.crra_certainty_equivalent(
                bundle, gamma, float(cfg["utility"]["discount_factor"]),
                float(cfg["utility"]["bequest_weight"]),
                bool(cfg["utility"]["bequest_enabled"])),
            f"cec_ez_gamma{gamma:g}_psi{psi:g}":
                ut.epstein_zin_certainty_equivalent(
                    bundle, gamma, psi,
                    float(cfg["utility"]["discount_factor"]),
                    float(cfg["utility"]["bequest_weight"]),
                    bool(cfg["utility"]["bequest_enabled"])),
            "prob_ruin": float(outcome.ruin.mean()),
            "median_wealth_at_retirement":
                float(np.median(outcome.wealth_at_retirement)),
            "median_bequest": float(np.median(outcome.bequest)),
            "p5_retirement_consumption": float(np.percentile(
                outcome.consumption[:, spec.retirement_slice].mean(axis=1), 5)),
        })
    return pd.DataFrame.from_records(rows)


def step4_report(cfg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble tables, figures and docs/04."""
    LOGGER.info("=== STEP 4: results and report ===")
    panel = state["panel"]
    spec = state["spec"]
    strategies = state["strategies"]
    results = state["results"]
    sampler = state["sampler"]
    n_paths = int(cfg["bootstrap"]["n_paths"])
    chunk_size = int(cfg["bootstrap"]["chunk_size"])

    headline = metrics_table(results, cfg, spec)
    distributions = distribution_tables(results, cfg, spec)
    dominance = rp.dominance_check(
        headline, "balanced_all_equity",
        ["target_date_fund", "sixty_forty", "domestic_equity", "bills_only"])
    tier_split = tier_breakdown(results, sampler, cfg, spec, n_paths, chunk_size)

    robustness: Dict[str, pd.DataFrame] = {}
    for mode in cfg["report"].get("robustness_panels", []):
        alt_panel = state["panels"].get(mode) or dl.build_panel(cfg, mode)
        robustness[f"Panel `{mode}` (fully empirical countries only)"] = \
            _run_variant(cfg, alt_panel, spec, strategies, n_paths)
    for draw in cfg["report"].get("robustness_country_draw", []):
        robustness[f"Country draw `{draw}`"] = _run_variant(
            cfg, panel, spec, strategies, n_paths, country_draw=draw)
    for weighting in cfg["report"].get("robustness_country_weighting", []):
        robustness[f"Country weighting `{weighting}`"] = _run_variant(
            cfg, panel, spec, strategies, n_paths, country_weighting=weighting)

    tables = cfg["run"]["table_dir"]
    _save_table(headline, tables, "headline_lifecycle_metrics")
    for name, frame in distributions.items():
        _save_table(frame, tables, f"distribution_{name}")
    _save_table(dominance, tables, "dominance_check")
    if len(tier_split):
        _save_table(tier_split, tables, "results_by_country_tier")
    for name, frame in robustness.items():
        slug = (name.replace("`", "").replace(" ", "_").replace("/", "_")
                .replace("(", "").replace(")", "").replace("=", "_").lower())
        _save_table(frame, tables, f"robustness_{slug}")

    labels = {k: s.label for k, s in strategies.items()}
    figure_dir = cfg["run"]["figure_dir"]
    figures = [
        str(plots.plot_terminal_wealth_cdf(results, labels, figure_dir)),
        str(plots.plot_retirement_consumption(
            results, labels, spec.retirement_slice, figure_dir)),
        str(plots.plot_shortfall_curves(
            results, labels, spec.retirement_slice, figure_dir)),
        str(plots.plot_cec_by_risk_aversion(headline, figure_dir)),
        str(plots.plot_wealth_fan(results, labels,
                                  np.arange(spec.age_start, spec.age_death + 1),
                                  figure_dir)),
        str(plots.plot_ruin_probability(headline, figure_dir)),
    ]

    runtime_notes = {
        "n_countries": panel.n_countries,
        "n_years": panel.n_years,
        "n_tier_b": sum(1 for t in panel.tier if t != "A"),
        "n_tier_c": sum(1 for t in panel.tier if t == "C"),
        "n_partial": sum(1 for t in panel.tier if t == "B"),
        "fingerprint": panel.fingerprint(),
    }

    rp.write_doc_04(
        Path("docs") / "04_replicated_results_and_tables.md",
        cfg, headline, distributions, robustness, dominance, tier_split,
        figures + state.get("figures", []), runtime_notes,
    )
    LOGGER.info("docs/04 written")
    state.update({"headline": headline, "dominance": dominance,
                  "robustness": robustness})
    return state


# ---------------------------------------------------------------------------
# Step 5
# ---------------------------------------------------------------------------
def step5_sensitivity(cfg: Dict[str, Any], state: Dict[str, Any]
                      ) -> Dict[str, Any]:
    """Sweep every parameter that could move the conclusion; write docs/05."""
    sens = cfg.get("sensitivity", {})
    if not sens.get("enabled", False):
        LOGGER.info("sensitivity analysis disabled in config; skipping step 5")
        return state
    LOGGER.info("=== STEP 5: sensitivity analysis ===")
    started = time.perf_counter()

    panel = state["panel"]
    spec = state.get("spec") or lc.spec_from_config(cfg)
    grids = sens["grids"]
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    n_paths = int(sens["n_paths"])
    redraw_paths = int(sens["redraw_n_paths"])
    target_ruin = float(sens["safe_withdrawal_target_ruin"])

    ctx = sn.SweepContext.build(
        cfg, panel, n_paths=n_paths,
        max_horizon=int(sens["max_horizon_years"]),
        max_working=int(sens["max_working_years"]),
    )
    strategies = ctx.strategies_from_config()
    core = {k: v for k, v in strategies.items() if k in sn.CORE_STRATEGIES}

    sweeps: Dict[str, pd.DataFrame] = {}
    LOGGER.info("sweeping allocation dials")
    sweeps["domestic_share"] = sn.sweep_domestic_share(
        ctx, grids["domestic_share"], gammas)
    sweeps["equity_share"] = sn.sweep_equity_share(
        ctx, grids["equity_share"], gammas)

    LOGGER.info("sweeping preferences")
    sweeps["risk_aversion"] = sn.sweep_risk_aversion(
        ctx, core, grids["risk_aversion"])
    sweeps["ies"] = sn.sweep_ies(ctx, core, grids["ies"])
    sweeps["bequest_weight"] = sn.sweep_bequest_weight(
        ctx, core, grids["bequest_weight"])

    LOGGER.info("sweeping planning parameters")
    for field, key in (("age_death", "age_death"),
                       ("age_retire", "age_retire"),
                       ("savings_rate", "savings_rate"),
                       ("rule_rate", "withdrawal_rate")):
        sweeps[key] = sn.sweep_lifecycle_field(
            ctx, core, field, grids[key], gammas)
    sweeps["social_security"] = sn.sweep_social_security(ctx, core, gammas)

    LOGGER.info("sweeping sampling assumptions")
    sweeps["mean_block_years"] = sn.sweep_bootstrap_field(
        cfg, panel, core, spec, "mean_block_years",
        grids["mean_block_years"], redraw_paths, gammas)
    panels = {name: p for name, p in state.get("panels", {}).items()}
    if len(panels) > 1:
        sweeps["panel"] = sn.sweep_panels(cfg, panels, core, spec,
                                          redraw_paths, gammas)

    baseline_cec = f"cec_crra_gamma{float(cfg['utility']['baseline_risk_aversion']):g}"
    derived: Dict[str, pd.DataFrame] = {
        "domestic_optimum": sn.optimal_allocation(
            sweeps["domestic_share"], "domestic_share", gammas),
        "equity_optimum": sn.optimal_allocation(
            sweeps["equity_share"], "equity_share", gammas),
        "crossover": sn.crossover_risk_aversion(sweeps["risk_aversion"]),
        "safe_withdrawal_rates": sn.safe_withdrawal_rates(
            sweeps["withdrawal_rate"], target_ruin),
    }

    # Dimensions that carry a per-strategy CEC and so can enter the tornado.
    tornado_inputs = {
        "Longevity (age at death)": (sweeps["age_death"], "age_death"),
        "Retirement age": (sweeps["age_retire"], "age_retire"),
        "Savings rate": (sweeps["savings_rate"], "savings_rate"),
        "Withdrawal rate": (sweeps["withdrawal_rate"], "withdrawal_rate"),
        "Social security design": (sweeps["social_security"], "social_security"),
        "Bootstrap block length": (sweeps["mean_block_years"],
                                   "mean_block_years"),
    }
    if "panel" in sweeps:
        tornado_inputs["Return panel"] = (sweeps["panel"], "panel")
    tornado = sn.tornado(tornado_inputs, baseline_cec)

    # The preference sweeps report a bare `cec` column, so they are folded in
    # through the same machinery after a rename.
    for name, key in (("Risk aversion", "risk_aversion"),
                      ("Elasticity of substitution", "ies"),
                      ("Bequest weight", "bequest_weight")):
        frame = sweeps[key].rename(columns={"cec": baseline_cec})
        column = {"risk_aversion": "risk_aversion", "ies": "ies",
                  "bequest_weight": "bequest_weight"}[key]
        extra = sn.tornado({name: (frame, column)}, baseline_cec)
        tornado = pd.concat([tornado, extra], ignore_index=True)
    tornado = tornado.sort_values("range_pp", ascending=False)
    verdict = sn.overall_verdict(tornado)

    tables = cfg["run"]["table_dir"]
    for name, frame in sweeps.items():
        _save_table(frame, tables, f"sensitivity_{name}")
    for name, frame in derived.items():
        _save_table(frame, tables, f"sensitivity_{name}")
    _save_table(tornado, tables, "sensitivity_tornado")

    figure_dir = cfg["run"]["figure_dir"]
    figures = [
        str(plots.plot_allocation_frontier(
            sweeps["domestic_share"], sweeps["equity_share"], gammas,
            figure_dir)),
        str(plots.plot_risk_aversion_sweep(sweeps["risk_aversion"], figure_dir)),
        str(plots.plot_withdrawal_sensitivity(
            sweeps["withdrawal_rate"], derived["safe_withdrawal_rates"],
            target_ruin, figure_dir)),
        str(plots.plot_planning_sweeps({
            "Age at death": (sweeps["age_death"], "age_death"),
            "Retirement age": (sweeps["age_retire"], "age_retire"),
            "Savings rate": (sweeps["savings_rate"], "savings_rate"),
            "Withdrawal rate": (sweeps["withdrawal_rate"], "withdrawal_rate"),
            "Bootstrap block length": (sweeps["mean_block_years"],
                                       "mean_block_years"),
        }, figure_dir, metric=baseline_cec)),
        str(plots.plot_tornado(tornado, figure_dir)),
    ]

    elapsed = time.perf_counter() - started
    runtime_notes = {
        "n_simulations": int(
            len(grids["domestic_share"]) + len(grids["equity_share"])
            + len(core) * (len(grids["age_death"]) + len(grids["age_retire"])
                           + len(grids["savings_rate"])
                           + len(grids["withdrawal_rate"]) + 4
                           + len(grids["mean_block_years"]))),
        "elapsed_seconds": elapsed,
    }
    rp.write_doc_05(
        Path("docs") / "05_sensitivity_analysis.md",
        cfg, sweeps, derived, tornado, verdict, figures, runtime_notes,
    )
    LOGGER.info("docs/05 written (%.0fs, %s settings, %s reversals)",
                elapsed, verdict.get("n_settings"), verdict.get("n_lost"))
    state.update({"sweeps": sweeps, "sensitivity_derived": derived,
                  "tornado": tornado, "verdict": verdict,
                  "sweep_context": ctx})
    return state


# ---------------------------------------------------------------------------
# Step 6
# ---------------------------------------------------------------------------
def step6_spending(cfg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Compare retirement spending rules; write docs/06."""
    spend_cfg = cfg.get("spending", {})
    if not spend_cfg.get("enabled", False):
        LOGGER.info("spending-rule analysis disabled in config; skipping step 6")
        return state
    LOGGER.info("=== STEP 6: retirement spending rules ===")
    started = time.perf_counter()

    sens = cfg["sensitivity"]
    panel = state["panel"]
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    metric = f"cec_gamma{gamma:g}"
    n_paths = int(sens["n_paths"])

    ctx = state.get("sweep_context") or sn.SweepContext.build(
        cfg, panel, n_paths=n_paths,
        max_horizon=int(sens["max_horizon_years"]),
        max_working=int(sens["max_working_years"]),
    )

    rule_specs = spend_cfg["rules"]
    rate_grid = spend_cfg["rate_grid"]
    LOGGER.info("sweeping %s rule variants over %s rates",
                len(rule_specs), len(rate_grid))
    sweep = sn.sweep_spending_rules(ctx, rule_specs, rate_grid,
                                    strategy_key=str(spend_cfg["strategy"]),
                                    gammas=gammas)
    best = sn.best_spending_rules(sweep, metric)

    # One representative per rule *family*, not the top N overall: the top of
    # the ranking is crowded with high-spending horizon rules, and a shortlist
    # drawn from it could not show the family crossover in section 6.
    per_family = sn.best_spending_rules(sweep, metric, group="rule")
    shortlist = sn.rules_from_config(cfg, per_family)
    by_strategy = sn.spending_by_strategy(ctx, shortlist, gammas=gammas)
    bequest_pivot = sn.spending_bequest_sensitivity(
        ctx, shortlist, cfg["sensitivity"]["grids"]["bequest_weight"],
        strategy_key=str(spend_cfg["strategy"]))
    paths = sn.spending_paths(ctx, shortlist,
                              strategy_key=str(spend_cfg["strategy"]))

    catalogue = pd.DataFrame.from_records([
        {**spg.build(str(entry["key"]),
                     **dict(entry.get("params", {}) or {})).describe(),
         "variant_suffix": entry.get("suffix") or "-"}
        for entry in rule_specs
    ]).drop_duplicates("key")[["key", "label"]]
    catalogue["rate_parameterised"] = catalogue["key"].isin(
        spg.RATE_PARAMETERISED)
    catalogue["can_deplete_portfolio"] = ~catalogue["key"].isin(
        {"constant_percent", "life_expectancy", "gompertz"})
    rank_by_gamma = sn.spending_rank_by_risk_aversion(sweep, gammas)

    tables = cfg["run"]["table_dir"]
    _save_table(sweep, tables, "spending_rule_sweep")
    _save_table(best, tables, "spending_rule_best")
    _save_table(by_strategy, tables, "spending_rule_by_strategy")
    _save_table(bequest_pivot, tables, "spending_rule_bequest_pivot")
    _save_table(paths, tables, "spending_rule_paths")
    _save_table(catalogue, tables, "spending_rule_catalogue")
    _save_table(per_family, tables, "spending_rule_best_per_family")
    _save_table(rank_by_gamma, tables, "spending_rule_rank_by_risk_aversion")

    figure_dir = cfg["run"]["figure_dir"]
    figures = [
        str(plots.plot_spending_rate_curves(sweep, best, figure_dir, metric)),
        str(plots.plot_spending_paths(paths, per_family, figure_dir, metric)),
        str(plots.plot_spending_bequest_pivot(bequest_pivot, figure_dir)),
    ]

    elapsed = time.perf_counter() - started
    runtime_notes = {
        "n_paths": n_paths,
        "n_variants": len(rule_specs),
        "n_simulations": int(len(sweep) + len(by_strategy)
                             + 2 * len(shortlist)),
        "elapsed_seconds": elapsed,
    }
    rp.write_doc_06(
        Path("docs") / "06_retirement_spending_rules.md",
        cfg, sweep, best, by_strategy, bequest_pivot, catalogue,
        rank_by_gamma, figures, runtime_notes,
    )
    LOGGER.info("docs/06 written (%.0fs, best rule: %s)",
                elapsed, best.iloc[0]["variant"])
    state.update({"spending_sweep": sweep, "spending_best": best,
                  "sweep_context": ctx})
    return state


# ---------------------------------------------------------------------------
# Step 7
# ---------------------------------------------------------------------------
def step7_glide_path(cfg: Dict[str, Any], state: Dict[str, Any]
                     ) -> Dict[str, Any]:
    """Solve for the optimal age-by-asset schedule; write docs/07."""
    glide_cfg = cfg.get("glide_path", {})
    if not glide_cfg.get("enabled", False):
        LOGGER.info("glide-path search disabled in config; skipping step 7")
        return state
    LOGGER.info("=== STEP 7: solving for the optimal glide path ===")
    started = time.perf_counter()

    panel = state["panel"]
    spec = state.get("spec") or lc.spec_from_config(cfg)
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    baseline_gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(glide_cfg["n_paths"])
    bond_share = float(glide_cfg["bond_share"])
    equity_grid = glide_cfg["equity_grid"]
    domestic_grid = glide_cfg["domestic_grid"]

    sampler = bs.from_config(panel, cfg, horizon_years=spec.horizon)
    paths = sampler.sample(n_paths, chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    income = lc.simulate_income(
        spec, n_paths, shocks=lc.draw_income_shocks(n_paths, spec.n_working, rng))
    evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
    benchmarks = lc.build_strategies(cfg, spec)

    trace = gp.OptimisationTrace()
    schedules: List[pd.DataFrame] = []
    comparisons: List[pd.DataFrame] = []
    profiles: List[pd.DataFrame] = []
    n_evaluations = 0

    parameterisation = gp.GlideParameterisation(
        knot_ages=tuple(int(a) for a in glide_cfg["parametric_knot_ages"]),
        domestic_share=0.15, bond_share=bond_share)

    for gamma in gammas:
        LOGGER.info("solving free-form schedule for gamma=%.1f", gamma)
        equity, domestic, cec = gp.optimise_free_form_banded(
            evaluator, gamma, equity_grid, domestic_grid,
            bond_share=bond_share,
            domestic_band_years=int(glide_cfg["domestic_band_years"]),
            n_sweeps=int(glide_cfg["n_sweeps"]), trace=trace)
        n_evaluations += (spec.horizon * len(equity_grid)
                          + len(domestic_grid) * (spec.horizon
                                                  // int(glide_cfg["domestic_band_years"]) + 1)
                          ) * int(glide_cfg["n_sweeps"])
        schedules.append(gp.schedule_frame(equity, domestic, spec, gamma,
                                           "free_form"))

        knots, pcec = gp.optimise_parametric(
            evaluator, gamma, parameterisation, glide_cfg["parametric_grid"],
            n_sweeps=int(glide_cfg["parametric_sweeps"]))
        n_evaluations += (len(parameterisation.knot_ages)
                          * len(glide_cfg["parametric_grid"])
                          * int(glide_cfg["parametric_sweeps"]))
        p_weights = parameterisation.build(knots, spec)
        schedules.append(gp.schedule_frame(
            p_weights[:, 0] + p_weights[:, 1],
            np.divide(p_weights[:, 0], np.maximum(p_weights[:, 0] + p_weights[:, 1], 1e-12)),
            spec, gamma, "parametric"))

        profiles.append(gp.deviation_profile(
            evaluator, equity, domestic, gamma, bond_share=bond_share))
        comparisons.append(gp.compare_to_benchmarks(
            evaluator,
            {"free_form_optimal": gp.weights_from_shares(equity, domestic,
                                                         bond_share),
             "parametric_optimal": p_weights},
            benchmarks, gamma))

    schedule_frame = pd.concat(schedules, ignore_index=True)
    comparison = pd.concat(comparisons, ignore_index=True)
    deviation = pd.concat(profiles, ignore_index=True)

    # --- local-optimum check ---------------------------------------------
    restart_rows: List[Dict[str, Any]] = []
    restart_gamma = float(glide_cfg["restart_risk_aversion"])
    for start in glide_cfg["restart_equity_starts"]:
        eq, dom, cec = gp.optimise_free_form_banded(
            evaluator, restart_gamma, equity_grid, domestic_grid,
            start_equity=float(start), bond_share=bond_share,
            domestic_band_years=int(glide_cfg["domestic_band_years"]),
            n_sweeps=int(glide_cfg["n_sweeps"]))
        restart_rows.append({
            "start_equity_share": float(start),
            "solved_cec": cec,
            "mean_equity_share": float(eq.mean()),
            "share_of_ages_at_100pct": float((eq >= 0.999).mean()),
        })
    restarts = pd.DataFrame.from_records(restart_rows)
    if len(restarts):
        restarts["gap_to_best_pct"] = (
            restarts["solved_cec"] / restarts["solved_cec"].max() - 1.0) * 100.0

    # --- retirement-anchor check ------------------------------------------
    anchor_rows: List[pd.DataFrame] = []
    anchor_summary_rows: List[Dict[str, Any]] = []
    anchor_cfg = glide_cfg.get("anchor_check", {})
    if anchor_cfg.get("enabled", False):
        anchor_gamma = float(anchor_cfg["risk_aversion"])
        variants: List[Tuple[str, Any]] = [
            (f"4% rule (anchors on wealth at {spec.age_retire})", None)]
        for entry in anchor_cfg["rules"]:
            rule = spg.build(str(entry["key"]),
                             **dict(entry.get("params", {}) or {}))
            variants.append((rule.label, rule))
        for label, rule in variants:
            local = gp.BatchEvaluator(paths, spec, income, cfg, rule)
            eq, dom, cec = gp.optimise_free_form_banded(
                local, anchor_gamma, equity_grid, domestic_grid,
                bond_share=bond_share,
                domestic_band_years=int(glide_cfg["domestic_band_years"]),
                n_sweeps=int(glide_cfg["n_sweeps"]))
            frame = gp.schedule_frame(eq, dom, spec, anchor_gamma, "anchor")
            frame["rule"] = label
            anchor_rows.append(frame)
            window = (spec.ages >= spec.age_retire - 1) &                 (spec.ages <= spec.age_retire + 2)
            anchor_summary_rows.append({
                "rule": label,
                "min_equity_share_at_retirement": float(eq[window].min()),
                "mean_equity_share_elsewhere": float(eq[~window].mean()),
                "dip_size_pp": float((eq[~window].mean() - eq[window].min())
                                     * 100.0),
                "solved_cec": cec,
            })
    anchor = pd.concat(anchor_rows, ignore_index=True) if anchor_rows         else pd.DataFrame()
    anchor_summary = pd.DataFrame.from_records(anchor_summary_rows)

    tables = cfg["run"]["table_dir"]
    _save_table(schedule_frame, tables, "glide_solved_schedules")
    _save_table(comparison, tables, "glide_comparison")
    _save_table(trace.frame(), tables, "glide_convergence")
    _save_table(restarts, tables, "glide_restarts")
    _save_table(deviation, tables, "glide_deviation_profile")
    if len(anchor):
        _save_table(anchor, tables, "glide_retirement_anchor")
        _save_table(anchor_summary, tables, "glide_retirement_anchor_summary")

    figure_dir = cfg["run"]["figure_dir"]
    industry = lc.glide_path_table(benchmarks, spec)
    figures = [
        str(plots.plot_optimal_glide(schedule_frame, industry, deviation,
                                     figure_dir)),
        str(plots.plot_glide_comparison(comparison, trace.frame(), figure_dir)),
    ]
    if len(anchor):
        figures.append(str(plots.plot_retirement_anchor(
            anchor, spec.age_retire, figure_dir)))

    elapsed = time.perf_counter() - started
    rp.write_doc_07(
        Path("docs") / "07_optimal_glide_path.md",
        cfg, schedule_frame, comparison, trace.frame(), restarts, anchor,
        anchor_summary, deviation, figures,
        {"n_evaluations": n_evaluations, "elapsed_seconds": elapsed,
         "horizon": spec.horizon},
    )
    LOGGER.info("docs/07 written (%.0fs, %s evaluations)", elapsed,
                f"{n_evaluations:,}")
    state.update({"glide_schedules": schedule_frame,
                  "glide_comparison": comparison})
    return state


# ---------------------------------------------------------------------------
# Step 8
# ---------------------------------------------------------------------------
def step8_hedging(cfg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Sweep the currency-hedge ratio against its annual cost; write docs/08."""
    hedge_cfg = cfg.get("hedging", {})
    if not hedge_cfg.get("enabled", False):
        LOGGER.info("hedging analysis disabled in config; skipping step 8")
        return state
    LOGGER.info("=== STEP 8: currency hedging ===")
    started = time.perf_counter()

    spec = state.get("spec") or lc.spec_from_config(cfg)
    strategies = state.get("strategies") or lc.build_strategies(cfg, spec)
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    metric = f"cec_gamma{float(cfg['utility']['baseline_risk_aversion']):g}"

    sweep = hg.sweep_hedging(
        cfg, spec, strategies,
        ratios=[float(r) for r in hedge_cfg["ratios"]],
        costs=[float(c) for c in hedge_cfg["costs"]],
        n_paths=int(hedge_cfg["n_paths"]),
        gammas=gammas,
        strategy_keys=tuple(hedge_cfg["strategies"]),
    )
    break_even = hg.break_even_costs(sweep, metric)
    optimal = hg.optimal_ratio_by_cost(sweep, metric)

    tables = cfg["run"]["table_dir"]
    _save_table(sweep, tables, "hedging_sweep")
    _save_table(break_even, tables, "hedging_break_even")
    _save_table(optimal, tables, "hedging_optimal_ratio")

    figures = [str(plots.plot_hedging(sweep, break_even,
                                      cfg["run"]["figure_dir"], metric))]

    elapsed = time.perf_counter() - started
    rp.write_doc_08(
        Path("docs") / "08_currency_hedging.md",
        cfg, sweep, break_even, optimal, figures,
        {"elapsed_seconds": elapsed,
         "n_countries": state["panel"].n_countries},
    )
    LOGGER.info("docs/08 written (%.0fs)", elapsed)
    state.update({"hedging_sweep": sweep, "hedging_break_even": break_even})
    return state


# ---------------------------------------------------------------------------
# Step 9
# ---------------------------------------------------------------------------
def step9_retirement_timing(cfg: Dict[str, Any], state: Dict[str, Any]
                            ) -> Dict[str, Any]:
    """Make the retirement date a decision; write docs/09."""
    timing_cfg = cfg.get("retirement_timing", {})
    if not timing_cfg.get("enabled", False):
        LOGGER.info("retirement-timing analysis disabled; skipping step 9")
        return state
    LOGGER.info("=== STEP 9: endogenous retirement timing ===")
    started = time.perf_counter()

    panel = state["panel"]
    base_spec = state.get("spec") or lc.spec_from_config(cfg)
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    metric = f"cec_gamma{float(cfg['utility']['baseline_risk_aversion']):g}"
    n_paths = int(timing_cfg["n_paths"])
    floor = float(timing_cfg["working_income_floor"])

    sampler = bs.from_config(panel, cfg, horizon_years=base_spec.horizon)
    paths = sampler.sample(n_paths, chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    shocks = lc.draw_income_shocks(n_paths, base_spec.horizon, rng)
    strategy = lc.build_strategies(cfg, base_spec)[str(timing_cfg["strategy"])]

    rules = [(str(entry["label"]),
              rt.build(str(entry["key"]), **dict(entry.get("params", {}) or {})))
             for entry in timing_cfg["rules"]]

    floors = [floor] + ([0.0] if timing_cfg.get("compare_without_floor", False)
                        and floor != 0.0 else [])
    rows: List[Dict[str, Any]] = []
    ages: Dict[str, np.ndarray] = {}
    lottery = pd.DataFrame()
    lottery_stats: Dict[str, float] = {}
    bull: Dict[str, float] = {}

    for active_floor in floors:
        spec = dataclasses.replace(base_spec, working_income_floor=active_floor)
        income = rt.extended_income(spec, n_paths, shocks=shocks)
        for label, rule in rules:
            outcome = rt.simulate_flexible(paths, strategy, spec, income, rule)
            rows.append(rt.evaluate(
                outcome, cfg, spec, gammas,
                extra={"variant": label, "working_income_floor": active_floor}))
            if active_floor == floor:
                ages[label] = outcome.retire_age
                if label == str(timing_cfg["lottery_rule"]):
                    lottery, lottery_stats = rt.retirement_lottery(
                        outcome, spec,
                        before=int(timing_cfg["window_before"]),
                        after=int(timing_cfg["window_after"]),
                        n_buckets=int(timing_cfg["n_buckets"]))
                    bull = rt.bull_market_test(
                        outcome, spec, before=int(timing_cfg["window_before"]))
        LOGGER.info("retirement timing: floor %.2f done", active_floor)

    summary = pd.DataFrame.from_records(rows)
    conditioning = pd.concat(
        [rt.value_of_conditioning(summary[summary["working_income_floor"] == f],
                                  metric).assign(working_income_floor=f)
         for f in floors], ignore_index=True)

    tables = cfg["run"]["table_dir"]
    _save_table(summary, tables, "retirement_timing_summary")
    _save_table(conditioning, tables, "retirement_value_of_conditioning")
    _save_table(lottery, tables, "retirement_lottery_deciles")
    _save_table(pd.DataFrame([lottery_stats]), tables, "retirement_lottery_stats")
    _save_table(pd.DataFrame([bull]), tables, "retirement_bull_market_test")

    figures = [str(plots.plot_retirement_timing(
        summary, ages, lottery, cfg["run"]["figure_dir"], metric))]

    elapsed = time.perf_counter() - started
    rp.write_doc_09(
        Path("docs") / "09_retirement_timing.md",
        cfg, summary, conditioning, lottery, lottery_stats, bull, figures,
        {"elapsed_seconds": elapsed},
    )
    LOGGER.info("docs/09 written (%.0fs)", elapsed)
    state.update({"retirement_summary": summary, "retirement_lottery": lottery})
    return state


# ---------------------------------------------------------------------------
# Step 10
# ---------------------------------------------------------------------------
def step10_saving(cfg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Solve the savings profile and test conditioning; write docs/10."""
    save_cfg = cfg.get("saving", {})
    if not save_cfg.get("enabled", False):
        LOGGER.info("savings-rate analysis disabled; skipping step 10")
        return state
    LOGGER.info("=== STEP 10: conditioning the savings rate ===")
    started = time.perf_counter()

    panel = state["panel"]
    spec = dataclasses.replace(
        state.get("spec") or lc.spec_from_config(cfg),
        working_income_floor=float(save_cfg["working_income_floor"]))
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    baseline_gamma = float(cfg["utility"]["baseline_risk_aversion"])
    metric = f"cec_gamma{baseline_gamma:g}"
    n_paths = int(save_cfg["n_paths"])
    target_mean = float(save_cfg["target_mean_rate"])
    floor, cap = float(save_cfg["rate_floor"]), float(save_cfg["rate_cap"])

    sampler = bs.from_config(panel, cfg, horizon_years=spec.horizon)
    paths = sampler.sample(n_paths, chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    income = rt.extended_income(
        spec, n_paths, shocks=lc.draw_income_shocks(n_paths, spec.horizon, rng))
    strategy = lc.build_strategies(cfg, spec)[str(save_cfg["strategy"])]
    fixed_retirement = rt.FixedAgeRule(age=spec.age_retire)

    def simulate(saving_rule, retirement_rule=None):
        return rt.simulate_flexible(
            paths, strategy, spec, income,
            retirement_rule or fixed_retirement, saving=saving_rule)

    def cec_of(saving_rule, gamma, retirement_rule=None):
        outcome = simulate(saving_rule, retirement_rule)
        return rt.evaluate(outcome, cfg, spec, [gamma])[f"cec_gamma{gamma:g}"]

    # --- level: the constant-rate frontier --------------------------------
    frontier_rows: List[Dict[str, Any]] = []
    for rate in save_cfg["constant_rate_grid"]:
        outcome = simulate(sav.ConstantRateRule(rate_value=float(rate)))
        row = rt.evaluate(outcome, cfg, spec, gammas,
                          extra={"variant": f"Constant {float(rate):.0%}",
                                 "savings_rate": float(rate)})
        frontier_rows.append(row)
    frontier = pd.DataFrame.from_records(frontier_rows)

    flat = np.full(spec.horizon, target_mean)
    baseline_outcome = simulate(sav.AgeProfileRule(flat))
    target = sav.wealth_to_income_target(baseline_outcome, income, spec.horizon)

    # --- shape: solved at a pinned average rate ---------------------------
    profiles: List[pd.DataFrame] = []
    shape_rows: List[Dict[str, Any]] = []
    solved: Dict[float, np.ndarray] = {}
    for gamma in gammas:
        flat_cec = cec_of(sav.AgeProfileRule(flat), gamma)
        schedule, best = sav.optimise_shape_at_fixed_mean(
            lambda sched, g=gamma: cec_of(sav.AgeProfileRule(sched), g),
            spec.n_working, spec.horizon, target_mean,
            save_cfg["shape_multiplier_grid"], floor=floor, cap=cap,
            n_sweeps=int(save_cfg["shape_sweeps"]))
        solved[gamma] = schedule
        frame = sav.profile_frame(schedule, spec, f"solved γ={gamma:g}")
        frame["risk_aversion"] = gamma
        profiles.append(frame)
        shape_rows.append({
            "risk_aversion": gamma,
            "flat_cec": flat_cec,
            "solved_cec": best,
            "gain_vs_flat_pct": (best / flat_cec - 1.0) * 100.0,
            "realised_mean_rate": float(schedule[:spec.n_working].mean()),
            "peak_age": int(spec.ages[int(np.argmax(schedule[:spec.n_working]))]),
            "peak_rate": float(schedule[:spec.n_working].max()),
        })
        LOGGER.info("saving shape solved for gamma=%.1f: CEC %.5f (%+.2f%%)",
                    gamma, best, shape_rows[-1]["gain_vs_flat_pct"])
    profile_frame = pd.concat(profiles, ignore_index=True)
    shape_summary = pd.DataFrame.from_records(shape_rows)

    deviation = sav.deviation_profile(
        lambda sched: cec_of(sav.AgeProfileRule(sched), baseline_gamma),
        solved[baseline_gamma], spec.n_working, spec.ages)

    # --- conditioning, layered on the solved shape ------------------------
    base_schedule = solved[baseline_gamma]
    base_cec = cec_of(sav.AgeProfileRule(base_schedule), baseline_gamma)
    conditioning_rows: List[Dict[str, Any]] = []
    variants: List[Dict[str, Any]] = []
    for k in save_cfg["on_track_sensitivity"]:
        rule = sav.OnTrackRule(target=target, base=base_schedule,
                               sensitivity=float(k), floor=floor, cap=cap)
        outcome = simulate(rule)
        row = rt.evaluate(outcome, cfg, spec, gammas,
                          extra={"variant": f"On-track k={float(k):g}"})
        variants.append(row)
        conditioning_rows.append({
            "rule": "On-track (wealth vs age target)",
            "sensitivity": float(k), "cec": row[metric],
            "vs_base_pct": (row[metric] / base_cec - 1.0) * 100.0,
            "mean_savings_rate": row["mean_savings_rate"]})
    for k in save_cfg["return_sensitivity"]:
        rule = sav.ReturnResponsiveRule(base=base_schedule,
                                        sensitivity=float(k),
                                        floor=floor, cap=cap)
        outcome = simulate(rule)
        row = rt.evaluate(outcome, cfg, spec, gammas,
                          extra={"variant": f"Return-responsive k={float(k):g}"})
        variants.append(row)
        conditioning_rows.append({
            "rule": "Return-responsive (last year's return)",
            "sensitivity": float(k), "cec": row[metric],
            "vs_base_pct": (row[metric] / base_cec - 1.0) * 100.0,
            "mean_savings_rate": row["mean_savings_rate"]})
    conditioning = pd.DataFrame.from_records(conditioning_rows)

    shape_row = rt.evaluate(simulate(sav.AgeProfileRule(base_schedule)), cfg,
                            spec, gammas, extra={"variant": "Solved shape"})
    matched = sav.matched_rate_comparison(
        pd.concat([frontier, pd.DataFrame([shape_row] + variants)],
                  ignore_index=True), metric)

    # --- does it stack with retirement-side conditioning? -----------------
    combined = pd.DataFrame()
    if save_cfg.get("combine_with_retirement", False):
        entry = save_cfg["retirement_rule"]
        flexible_retirement = rt.build(str(entry["key"]),
                                       **dict(entry.get("params", {}) or {}))
        best_k = float(conditioning.loc[
            conditioning[conditioning["rule"].str.contains("track")]
            ["vs_base_pct"].idxmax(), "sensitivity"])
        best_saving = sav.OnTrackRule(target=target, base=base_schedule,
                                      sensitivity=best_k, floor=floor, cap=cap)
        rows = []
        for label, srule, rrule in (
            ("Neither", sav.ConstantRateRule(target_mean), fixed_retirement),
            ("Savings conditioning only", best_saving, fixed_retirement),
            ("Retirement conditioning only", sav.ConstantRateRule(target_mean),
             flexible_retirement),
            ("Both", best_saving, flexible_retirement),
        ):
            outcome = simulate(srule, rrule)
            rows.append(rt.evaluate(outcome, cfg, spec, gammas,
                                    extra={"variant": label}))
        combined = pd.DataFrame.from_records(rows)
        neither = float(combined.loc[combined.variant == "Neither", metric].iloc[0])
        combined["vs_neither_pct"] = (combined[metric] / neither - 1.0) * 100.0

    tables = cfg["run"]["table_dir"]
    _save_table(frontier, tables, "saving_constant_rate_frontier")
    _save_table(profile_frame, tables, "saving_solved_profiles")
    _save_table(shape_summary, tables, "saving_shape_summary")
    _save_table(conditioning, tables, "saving_conditioning")
    _save_table(deviation, tables, "saving_deviation_profile")
    if len(matched):
        _save_table(matched, tables, "saving_matched_rate")
    if len(combined):
        _save_table(combined, tables, "saving_combined_with_retirement")

    figures = [str(plots.plot_saving(profile_frame, frontier, conditioning,
                                     cfg["run"]["figure_dir"], target_mean))]

    elapsed = time.perf_counter() - started
    rp.write_doc_10(
        Path("docs") / "10_savings_rate.md",
        cfg, frontier, profile_frame, shape_summary, conditioning, matched,
        deviation, combined, figures, {"elapsed_seconds": elapsed},
    )
    LOGGER.info("docs/10 written (%.0fs)", elapsed)
    state.update({"saving_profiles": profile_frame,
                  "saving_conditioning": conditioning})
    return state


# ---------------------------------------------------------------------------
# Step 11
# ---------------------------------------------------------------------------
def _base_savings_schedule(cfg: Dict[str, Any], state: Dict[str, Any],
                           spec: lc.LifecycleSpec, target_mean: float,
                           gamma: float) -> Tuple[np.ndarray, str]:
    """The deterministic profile that conditioning is measured on top of.

    Step 10 already solved this, so it is reused when available -- from the
    live pipeline state, or failing that from the table it wrote. Running
    step 11 on its own falls back to a flat rate, which changes the base but
    not the question being asked, so the fallback is named in the document
    rather than hidden.
    """
    frame = state.get("saving_profiles")
    if frame is None:
        path = Path(cfg["run"]["table_dir"]) / "saving_solved_profiles.csv"
        if path.exists():
            frame = pd.read_csv(path)
    if frame is not None and len(frame):
        block = frame[np.isclose(frame["risk_aversion"], gamma)] \
            .sort_values("age")
        if len(block) >= spec.n_working:
            schedule = np.full(spec.horizon, target_mean)
            schedule[:spec.n_working] = \
                block["savings_rate"].to_numpy(dtype=float)[:spec.n_working]
            return schedule, "the solved age profile from docs/10"
    return (np.full(spec.horizon, target_mean),
            f"a flat {target_mean:.0%} rate (step 10 was not run)")


def step11_accumulation(cfg: Dict[str, Any],
                        state: Dict[str, Any]) -> Dict[str, Any]:
    """Take the accumulation signal apart; write docs/11."""
    acc_cfg = cfg.get("accumulation", {})
    if not acc_cfg.get("enabled", False):
        LOGGER.info("accumulation study disabled; skipping step 11")
        return state
    LOGGER.info("=== STEP 11: taking the accumulation signal apart ===")
    started = time.perf_counter()

    panel = state["panel"]
    spec = dataclasses.replace(
        state.get("spec") or lc.spec_from_config(cfg),
        working_income_floor=float(acc_cfg["working_income_floor"]))
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    baseline_gamma = float(cfg["utility"]["baseline_risk_aversion"])
    metric = f"cec_gamma{baseline_gamma:g}"
    n_paths = int(acc_cfg["n_paths"])
    target_mean = float(acc_cfg["target_mean_rate"])
    floor, cap = float(acc_cfg["rate_floor"]), float(acc_cfg["rate_cap"])
    tables = cfg["run"]["table_dir"]
    figure_dir = cfg["run"]["figure_dir"]

    sampler = bs.from_config(panel, cfg, horizon_years=spec.horizon)
    paths = sampler.sample(n_paths,
                           chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    income = rt.extended_income(
        spec, n_paths, shocks=lc.draw_income_shocks(n_paths, spec.horizon, rng))
    strategies = lc.build_strategies(cfg, spec)
    strategy = strategies[str(acc_cfg["strategy"])]
    fixed = rt.FixedAgeRule(age=spec.age_retire)

    def simulate(rule, *, strat=None, inc=None, sp=None):
        return rt.simulate_flexible(paths, strat or strategy, sp or spec,
                                    income if inc is None else inc, fixed,
                                    saving=rule)

    def make_run(strat=None, inc=None, sp=None):
        """A ``rule -> summary row`` callable for one arm of the study."""
        def run(rule):
            return rt.evaluate(simulate(rule, strat=strat, inc=inc, sp=sp),
                               cfg, sp or spec, gammas)
        return run

    def frontier_for(run) -> pd.DataFrame:
        rows = []
        for rate in acc_cfg["constant_rate_grid"]:
            row = dict(run(sav.ConstantRateRule(rate_value=float(rate))))
            row["savings_rate"] = float(rate)
            rows.append(row)
        return pd.DataFrame.from_records(rows)

    run = make_run()
    frontier = frontier_for(run)
    scorer = acc.MatchedScorer.from_frontier(frontier, metric)

    base, base_label = _base_savings_schedule(cfg, state, spec, target_mean,
                                              baseline_gamma)
    baseline_outcome = simulate(sav.AgeProfileRule(base))
    model_target = sav.wealth_to_income_target(baseline_outcome, income,
                                               spec.horizon)
    LOGGER.info("step 11 base profile: %s", base_label)

    # --- 1. functional form -----------------------------------------------
    forms = acc.sweep_response_forms(
        run, metric, scorer, base, model_target,
        {str(k): [float(v) for v in grid]
         for k, grid in acc_cfg["response_grids"].items()},
        floor=floor, cap=cap)
    best_form_row = forms.loc[forms["matched_value_pct"].idxmax()]
    best_form = str(best_form_row["form"])
    best_k = float(best_form_row["sensitivity"])
    LOGGER.info("best response form: %s, k=%g (%+.2f%%)", best_form, best_k,
                float(best_form_row["matched_value_pct"]))

    ratio_grid = np.linspace(0.2, 2.0, 19)
    reference_year = min(20, spec.n_working - 1)
    policy_rows: List[Dict[str, Any]] = []
    for form in forms["form"].unique():
        block = forms[forms["form"] == form]
        k = float(block.loc[block["matched_value_pct"].idxmax(), "sensitivity"])
        rule = acc.FundedRatioRule(target=model_target, base=base,
                                   k_behind=k, form=str(form),
                                   floor=floor, cap=cap)
        rates = acc.policy_curve(rule, reference_year,
                                 float(model_target[reference_year]),
                                 ratio_grid)
        policy_rows.extend({"form": str(form), "sensitivity": k,
                            "age": int(spec.ages[reference_year]),
                            "funded_ratio": float(r), "savings_rate": float(v)}
                           for r, v in zip(ratio_grid, rates))
    policy = pd.DataFrame.from_records(policy_rows)

    # --- 2. asymmetry ------------------------------------------------------
    asymmetry = acc.sweep_asymmetry(
        run, metric, scorer, base, model_target,
        [float(v) for v in acc_cfg["asymmetry"]["behind"]],
        [float(v) for v in acc_cfg["asymmetry"]["ahead"]],
        form=best_form, floor=floor, cap=cap)

    # --- 3. coarse guardrails ---------------------------------------------
    bands = acc.sweep_bands(run, metric, scorer, base, model_target,
                            [float(v) for v in acc_cfg["bands"]],
                            [float(v) for v in acc_cfg["band_steps"]],
                            floor=floor, cap=cap)

    # --- 4. which target ---------------------------------------------------
    target_set = {
        "model median path": model_target,
        "published ladder": acc.ladder_target(spec),
        f"flat {float(acc_cfg['flat_target_multiple']):g}x income":
            acc.flat_target(spec, float(acc_cfg["flat_target_multiple"])),
    }
    targets = acc.sweep_targets(run, metric, scorer, base, target_set,
                                [float(v) for v in acc_cfg["target_factors"]],
                                best_k, form=best_form, floor=floor, cap=cap)
    target_paths = pd.concat([
        pd.DataFrame({"target": name, "age": spec.ages[:spec.n_working],
                      "multiple": np.asarray(values)[:spec.n_working]})
        for name, values in target_set.items()], ignore_index=True)

    # --- 5. which signal ---------------------------------------------------
    income_profile = income.mean(axis=0)
    signals = {name: acc.make_signal(name, target=model_target,
                                     income_profile=income_profile,
                                     form=best_form)
               for name in acc_cfg["signals"]}
    race = acc.signal_race(run, metric, scorer, base, signals,
                           [float(v) for v in acc_cfg["signal_grid"]],
                           floor=floor, cap=cap)
    signal_best = acc.best_by(race, "signal")
    no_conditioning = float(race[race["signal"] == "none"]["matched_value_pct"]
                            .iloc[0]) if (race["signal"] == "none").any() else 0.0

    # The two leading signals, layered.  They may be reading the same state.
    ranked = signal_best.sort_values("matched_value_pct", ascending=False)
    pair = [str(k) for k in ranked["signal"] if k != "none"][:2]
    if len(pair) < 2:
        raise ValueError(
            "the combination sweep needs at least two live signals; "
            f"accumulation.signals gave {pair}")
    combo_grid = [float(v) for v in acc_cfg["combination_grid"]]
    combination = acc.sweep_combination(
        run, metric, scorer, base, signals[pair[0]], signals[pair[1]],
        combo_grid, combo_grid, first_name=pair[0], second_name=pair[1],
        floor=floor, cap=cap)

    # --- 6. feasibility ----------------------------------------------------
    feasibility = acc.feasibility_frontier(
        run, metric, scorer, base, model_target, best_k,
        [float(v) for v in acc_cfg["feasibility_widths"]], target_mean,
        form=best_form)

    best_rule = acc.FundedRatioRule(target=model_target, base=base,
                                    k_behind=best_k, form=best_form,
                                    floor=floor, cap=cap)
    best_outcome = simulate(best_rule)
    fan = acc.rate_fan(best_outcome, spec)
    activity = acc.activity_profile(baseline_outcome, best_outcome, spec)
    quantile_gain = acc.quantile_gain(baseline_outcome, best_outcome,
                                      [float(q) for q in acc_cfg["quantiles"]])

    # --- 7. when it matters ------------------------------------------------
    windows = acc.age_window_value(
        run, metric, scorer, base, model_target, best_k,
        [(int(lo), int(hi)) for lo, hi in acc_cfg["age_windows"]],
        form=best_form, floor=floor, cap=cap)

    # --- 8. who wants it ---------------------------------------------------
    gamma_rows: List[Dict[str, Any]] = []
    for gamma in gammas:
        gamma_scorer = acc.MatchedScorer.from_frontier(
            frontier, f"cec_gamma{gamma:g}")
        for k in acc_cfg["signal_grid"]:
            rule = acc.FundedRatioRule(target=model_target, base=base,
                                       k_behind=float(k), form=best_form,
                                       floor=floor, cap=cap)
            row = dict(run(rule))
            gamma_rows.append({
                "risk_aversion": gamma, "sensitivity": float(k),
                "cec": float(row[f"cec_gamma{gamma:g}"]),
                "mean_savings_rate": float(row["mean_savings_rate"]),
                "matched_value_pct": gamma_scorer.value_pct(
                    float(row[f"cec_gamma{gamma:g}"]),
                    float(row["mean_savings_rate"])),
            })
    by_gamma = pd.DataFrame.from_records(gamma_rows)

    # --- 9. what it interacts with ----------------------------------------
    strategy_rows: List[Dict[str, Any]] = []
    for key in acc_cfg["strategy_interaction"]:
        arm_run = make_run(strat=strategies[str(key)])
        arm_scorer = acc.MatchedScorer.from_frontier(
            frontier_for(arm_run), metric)
        rule = acc.FundedRatioRule(target=model_target, base=base,
                                   k_behind=best_k, form=best_form,
                                   floor=floor, cap=cap)
        strategy_rows.append(acc.score_rule(
            arm_run, rule, metric, arm_scorer,
            strategy=str(key).replace("_", " ")))
    by_strategy = pd.DataFrame.from_records(strategy_rows)

    income_rows: List[Dict[str, Any]] = []
    for factor in acc_cfg["income_volatility_factors"]:
        arm_spec = dataclasses.replace(
            spec,
            permanent_shock_sd=spec.permanent_shock_sd * float(factor),
            transitory_shock_sd=spec.transitory_shock_sd * float(factor))
        arm_rng = np.random.default_rng(int(cfg["run"]["seed"]))
        arm_income = rt.extended_income(
            arm_spec, n_paths,
            shocks=lc.draw_income_shocks(n_paths, spec.horizon, arm_rng))
        arm_run = make_run(inc=arm_income, sp=arm_spec)
        arm_scorer = acc.MatchedScorer.from_frontier(
            frontier_for(arm_run), metric)
        arm_target = sav.wealth_to_income_target(
            simulate(sav.AgeProfileRule(base), inc=arm_income, sp=arm_spec),
            arm_income, spec.horizon)
        rule = acc.FundedRatioRule(target=arm_target, base=base,
                                   k_behind=best_k, form=best_form,
                                   floor=floor, cap=cap)
        income_rows.append(acc.score_rule(arm_run, rule, metric, arm_scorer,
                                          volatility_factor=float(factor)))
    by_income = pd.DataFrame.from_records(income_rows)

    # Every sweep is scored against a matched constant rate, but each also sits
    # on top of an age profile that is itself worth something; the saved tables
    # carry both so a reader does not have to subtract by hand. `by_gamma` is
    # excluded because its baseline differs at each risk aversion.
    for frame, name, net_out in (
            (frontier, "acc_constant_rate_frontier", False),
            (forms, "acc_response_forms", True),
            (policy, "acc_policy_curves", False),
            (asymmetry, "acc_asymmetry", True),
            (bands, "acc_guardrail_bands", True),
            (targets, "acc_target_choice", True),
            (target_paths, "acc_target_paths", False),
            (race, "acc_signal_race", True),
            (signal_best, "acc_signal_best", True),
            (combination, "acc_signal_combination", True),
            (feasibility, "acc_feasibility", True),
            (fan, "acc_rate_fan", False),
            (activity, "acc_activity_profile", False),
            (quantile_gain, "acc_quantile_gain", False),
            (by_gamma, "acc_by_risk_aversion", False),
            (windows, "acc_age_windows", True),
            (by_strategy, "acc_by_strategy", True),
            (by_income, "acc_by_income_volatility", True)):
        _save_table(acc.increment_over(frame, no_conditioning) if net_out
                    else frame, tables, name)

    figures = [
        str(plots.plot_response_forms(forms, policy, figure_dir)),
        str(plots.plot_asymmetry(asymmetry, figure_dir)),
        str(plots.plot_signal_race(signal_best, race, combination,
                                   figure_dir)),
        str(plots.plot_feasibility(feasibility, fan, figure_dir, target_mean)),
        str(plots.plot_value_distribution(quantile_gain, by_gamma, figure_dir)),
        str(plots.plot_when_it_matters(windows, activity, figure_dir)),
        str(plots.plot_target_choice(targets, target_paths, figure_dir)),
        str(plots.plot_accumulation_interactions(by_strategy, by_income,
                                                 figure_dir)),
    ]

    elapsed = time.perf_counter() - started
    rp.write_doc_11(
        Path("docs") / "11_accumulation_signal.md", cfg,
        {"forms": forms, "policy": policy, "asymmetry": asymmetry,
         "bands": bands, "targets": targets, "race": race,
         "signal_best": signal_best, "combination": combination,
         "feasibility": feasibility,
         "fan": fan, "activity": activity, "quantile_gain": quantile_gain,
         "by_gamma": by_gamma, "windows": windows,
         "by_strategy": by_strategy, "by_income": by_income,
         "frontier": frontier},
        figures,
        {"elapsed_seconds": elapsed, "base_label": base_label,
         "best_form": best_form, "best_k": best_k,
         "n_paths": n_paths, "reference_age": int(spec.ages[reference_year]),
         "no_conditioning_pct": no_conditioning},
    )
    LOGGER.info("docs/11 written (%.0fs)", elapsed)
    state.update({"accumulation_forms": forms,
                  "accumulation_signals": signal_best})
    return state


# ---------------------------------------------------------------------------
# Step 12
# ---------------------------------------------------------------------------
def _solver_inputs(cfg: Dict[str, Any], state: Dict[str, Any],
                   n_paths: int) -> Tuple[Any, lc.LifecycleSpec, np.ndarray]:
    """Bootstrap paths, spec and income for the batched schedule solvers."""
    spec = state.get("spec") or lc.spec_from_config(cfg)
    sampler = bs.from_config(state["panel"], cfg, horizon_years=spec.horizon)
    paths = sampler.sample(n_paths,
                           chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    rng = np.random.default_rng(int(cfg["run"]["seed"]))
    income = lc.simulate_income(spec, n_paths, rng=rng)
    return paths, spec, income


def step12_allocation(cfg: Dict[str, Any],
                      state: Dict[str, Any]) -> Dict[str, Any]:
    """Solve the full four-asset weight simplex at every age; write docs/12."""
    alloc_cfg = cfg.get("allocation", {})
    if not alloc_cfg.get("enabled", False):
        LOGGER.info("full-allocation solve disabled; skipping step 12")
        return state
    LOGGER.info("=== STEP 12: solving the whole allocation ===")
    started = time.perf_counter()

    n_paths = int(alloc_cfg["n_paths"])
    paths, spec, income = _solver_inputs(cfg, state, n_paths)
    evaluator = gp.BatchEvaluator(paths, spec, income, cfg)
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    baseline_gamma = float(cfg["utility"]["baseline_risk_aversion"])
    search = dict(coarse_step=float(alloc_cfg["coarse_step"]),
                  fine_step=float(alloc_cfg["fine_step"]),
                  coarse_sweeps=int(alloc_cfg["coarse_sweeps"]),
                  fine_sweeps=int(alloc_cfg["fine_sweeps"]))

    solved: Dict[float, np.ndarray] = {}
    frames: List[pd.DataFrame] = []
    traces: List[pd.DataFrame] = []
    for gamma in gammas:
        schedule, cec, trace = al.optimise_full_simplex(
            evaluator, gamma, label=f"gamma={gamma:g} ", **search)
        solved[gamma] = schedule
        frames.append(al.schedule_frame(schedule, spec, gamma))
        trace["risk_aversion"] = gamma
        traces.append(trace)
        LOGGER.info("full simplex solved for gamma=%.1f: CEC=%.6f", gamma, cec)
    schedules = pd.concat(frames, ignore_index=True)
    convergence = pd.concat(traces, ignore_index=True)
    phases = al.phase_summary(schedules)

    deviation = pd.concat([
        al.deviation_profile(evaluator, solved[g], g, spec) for g in gammas],
        ignore_index=True)

    strategies = lc.build_strategies(cfg, spec)
    extra: Dict[str, np.ndarray] = {}
    glide = state.get("glide_schedules")
    if glide is not None and len(glide):
        for kind in glide["kind"].unique():
            block = glide[(glide["kind"] == kind)
                          & np.isclose(glide["risk_aversion"], baseline_gamma)]
            if len(block) == spec.horizon:
                block = block.sort_values("age")
                extra[f"docs07_{str(kind)}"] = gp.weights_from_shares(
                    block["equity_share"].to_numpy(dtype=float),
                    block["domestic_share_of_equity"].to_numpy(dtype=float),
                    float(cfg["glide_path"]["bond_share"]))
    comparison = al.compare_to_benchmarks(evaluator, solved, strategies,
                                          gammas, extra=extra)

    restarts, _, _ = al.restart_check(
        evaluator, float(alloc_cfg["restart_risk_aversion"]),
        [list(map(float, row)) for row in alloc_cfg["restart_starts"]],
        **search)

    tables = cfg["run"]["table_dir"]
    for frame, name in ((schedules, "allocation_solved_schedules"),
                        (phases, "allocation_phase_summary"),
                        (convergence, "allocation_convergence"),
                        (deviation, "allocation_deviation_profile"),
                        (comparison, "allocation_comparison"),
                        (restarts, "allocation_restarts")):
        _save_table(frame, tables, name)

    figures = [
        str(plots.plot_full_allocation(schedules, deviation,
                                       cfg["run"]["figure_dir"],
                                       retire_age=spec.age_retire)),
        str(plots.plot_allocation_comparison(comparison, phases,
                                             cfg["run"]["figure_dir"])),
    ]

    elapsed = time.perf_counter() - started
    rp.write_doc_12(
        Path("docs") / "12_full_allocation.md", cfg,
        {"schedules": schedules, "phases": phases,
         "convergence": convergence, "deviation": deviation,
         "comparison": comparison, "restarts": restarts},
        figures, {"elapsed_seconds": elapsed, "n_paths": n_paths,
                  "n_countries": state["panel"].n_countries})
    LOGGER.info("docs/12 written (%.0fs)", elapsed)
    state.update({"allocation_schedules": schedules,
                  "allocation_solved": solved})
    return state


# ---------------------------------------------------------------------------
# Step 13
# ---------------------------------------------------------------------------
def step13_leverage(cfg: Dict[str, Any],
                    state: Dict[str, Any]) -> Dict[str, Any]:
    """Optimal leverage and allocation by the price of credit; write docs/13."""
    lev_cfg = cfg.get("leverage", {})
    if not lev_cfg.get("enabled", False):
        LOGGER.info("leverage study disabled; skipping step 13")
        return state
    LOGGER.info("=== STEP 13: borrowing to invest ===")
    started = time.perf_counter()

    n_paths = int(lev_cfg["n_paths"])
    paths, spec, income = _solver_inputs(cfg, state, n_paths)
    gamma = float(lev_cfg["risk_aversion"])
    gammas = [float(g) for g in cfg["utility"]["risk_aversions"]]
    evaluator = lev.make_evaluator(paths, spec, income, cfg)
    allocations = al.simplex_lattice(float(lev_cfg["allocation_step"]))

    sweep = lev.sweep_cost_and_leverage(
        evaluator, gamma,
        [float(v) for v in lev_cfg["leverage_grid"]],
        [float(v) for v in lev_cfg["spread_grid"]],
        allocations)
    optimal = lev.optimal_by_cost(sweep)
    break_even = lev.break_even_spread(sweep)
    LOGGER.info("leverage break-even spread: %.4f", break_even)

    detail_spread = float(lev_cfg["detail_spread"])
    best_row = optimal[np.isclose(optimal["spread"], detail_spread)]
    best_weights = np.array(
        [float(best_row[a].iloc[0]) for a in lc.ASSETS]) if len(best_row) \
        else np.array([0.5, 0.5, 0.0, 0.0])
    detail_rows: List[Dict[str, Any]] = []
    for ratio in lev_cfg["detail_leverage"]:
        evaluator.spread = detail_spread
        evaluator.set_leverage(float(ratio))
        weights = np.repeat(best_weights[None, :], spec.horizon, axis=0)
        detail_rows.append(lev.outcome_detail(
            evaluator, weights, spec, cfg, gammas,
            extra={"leverage": float(ratio), "spread": detail_spread}))
    detail = pd.DataFrame.from_records(detail_rows)
    base_cec = float(detail[np.isclose(detail["leverage"], 1.0)]
                     [f"cec_gamma{gamma:g}"].iloc[0])
    detail["vs_unlevered_pct"] = (detail[f"cec_gamma{gamma:g}"]
                                  / base_cec - 1.0) * 100.0

    schedule_frames: List[pd.DataFrame] = []
    schedule_cfg = lev_cfg.get("schedule", {})
    if schedule_cfg.get("enabled", False):
        weights = np.repeat(best_weights[None, :], spec.horizon, axis=0)
        for spread in schedule_cfg["spreads"]:
            solved, cec, _ = lev.optimise_leverage_schedule(
                evaluator, gamma, weights,
                [float(v) for v in schedule_cfg["grid"]], float(spread),
                n_sweeps=int(schedule_cfg["sweeps"]))
            schedule_frames.append(pd.DataFrame({
                "spread": float(spread), "age": spec.ages[:spec.horizon],
                "leverage": solved, "solved_cec": cec,
                "phase": np.where(np.arange(spec.horizon) < spec.n_working,
                                  "working", "retired")}))
    schedule = pd.concat(schedule_frames, ignore_index=True) \
        if schedule_frames else pd.DataFrame()
    by_decade = lev.schedule_by_decade(schedule) if len(schedule) \
        else pd.DataFrame()

    # A 68-parameter schedule scored on its own solve paths will always look
    # good. This asks how much of the gain a one-parameter version keeps.
    two_level = lev.two_level_comparison(
        evaluator, gamma, np.repeat(best_weights[None, :], spec.horizon, axis=0),
        [float(v) for v in schedule_cfg.get("grid", lev_cfg["leverage_grid"])],
        [float(v) for v in schedule_cfg.get("spreads", [detail_spread])],
        int(spec.n_working), solved=schedule) \
        if schedule_cfg.get("enabled", False) else pd.DataFrame()

    tables = cfg["run"]["table_dir"]
    for frame, name in ((sweep, "leverage_sweep"),
                        (optimal, "leverage_optimal_by_cost"),
                        (detail, "leverage_outcome_detail"),
                        (schedule, "leverage_schedule"),
                        (by_decade, "leverage_schedule_by_decade"),
                        (two_level, "leverage_two_level")):
        if len(frame):
            _save_table(frame, tables, name)

    figures = [
        str(plots.plot_leverage_surface(sweep, optimal,
                                        cfg["run"]["figure_dir"]))]
    if len(schedule):
        figures.append(str(plots.plot_leverage_detail(
            detail, schedule, cfg["run"]["figure_dir"])))

    elapsed = time.perf_counter() - started
    rp.write_doc_13(
        Path("docs") / "13_leverage.md", cfg,
        {"sweep": sweep, "optimal": optimal, "detail": detail,
         "schedule": schedule, "by_decade": by_decade,
         "two_level": two_level},
        figures, {"elapsed_seconds": elapsed, "n_paths": n_paths,
                  "break_even_spread": break_even, "gamma": gamma})
    LOGGER.info("docs/13 written (%.0fs)", elapsed)
    state.update({"leverage_sweep": sweep, "leverage_optimal": optimal})
    return state


# ---------------------------------------------------------------------------
# Step 14
# ---------------------------------------------------------------------------
def step14_provenance(cfg: Dict[str, Any],
                      state: Dict[str, Any]) -> Dict[str, Any]:
    """Audit where every number in the panel came from; write docs/14."""
    audit_cfg = cfg.get("provenance", {})
    if not audit_cfg.get("enabled", False):
        LOGGER.info("provenance audit disabled; skipping step 14")
        return state
    LOGGER.info("=== STEP 14: data provenance audit ===")
    started = time.perf_counter()

    panel = state["panel"]
    raw = pvn.load_raw_workbook(cfg)
    digests = pvn.source_digests(cfg)
    coverage = pvn.coverage_by_series(raw, int(cfg["data"]["start_year"]),
                                      int(cfg["data"]["end_year"]))
    countries = pvn.country_provenance(panel)
    era = pvn.simulated_share_by_era(panel)
    contamination = pvn.international_leg_contamination(panel)
    unusable = pvn.unusable_observed_series(cfg, dl.load_jst(cfg))
    generated = pvn.generated_cells(panel)
    housing = pvn.housing_audit(raw, panel)
    wages = pvn.wage_audit(raw, panel, cfg)
    anchors = pvn.anchor_check(raw)
    identity = pvn.identity_check(raw)
    tail = pvn.tail_variance_test(
        raw, tail_start=int(audit_cfg["tail_start"]),
        reference_start=int(audit_cfg["reference_start"]),
        reference_end=int(audit_cfg["tail_start"]) - 1)
    summary = pvn.panel_summary(panel)
    verdict = pvn.tail_verdict(tail)

    if not bool(anchors["within_tolerance"].all()):
        failed = list(anchors[~anchors["within_tolerance"]]["what"])
        LOGGER.warning("provenance: anchor checks FAILED for %s", failed)

    if len(generated):
        LOGGER.error("provenance: %d panel cells are available but not "
                     "observed; a generated block has returned",
                     len(generated))

    # There is no second panel to compare against any more: the simulated one
    # was removed rather than reported alongside. What the headline needs from
    # this step is the advantage itself, on the only panel there is.
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    column = f"cec_crra_gamma{gamma:g}"
    # Read from state when step 4 ran in the same session, and from its table
    # otherwise, so the audit can be re-run on its own.
    tables_dir = Path(cfg["run"]["table_dir"])
    headline = state.get("headline")
    if headline is None:
        headline = pd.read_csv(tables_dir / "headline_lifecycle_metrics.csv")
    observed_cec = headline.set_index("strategy")[column]

    def advantage(series: pd.Series) -> float:
        return (float(series["balanced_all_equity"])
                / float(series["target_date_fund"]) - 1.0) * 100.0

    tables = cfg["run"]["table_dir"]
    for frame, name in ((digests, "provenance_source_files"),
                        (coverage, "provenance_source_coverage"),
                        (countries, "provenance_by_country"),
                        (era, "provenance_by_era"),
                        (contamination, "provenance_intl_contamination"),
                        (unusable, "provenance_unusable_series"),
                        (generated, "provenance_generated_cells"),
                        (housing, "provenance_housing_audit"),
                        (wages, "provenance_wage_audit"),
                        (anchors, "provenance_anchor_checks"),
                        (identity, "provenance_identity_check"),
                        (tail, "provenance_tail_variance")):
        _save_table(frame, tables, name)

    figures = [str(plots.plot_provenance(era, contamination, tail, countries,
                                         cfg["run"]["figure_dir"]))]
    if len(housing):
        figures.append(str(plots.plot_housing(housing,
                                              cfg["run"]["figure_dir"])))

    elapsed = time.perf_counter() - started
    rp.write_doc_14(
        Path("docs") / "14_data_provenance.md", cfg,
        {"digests": digests, "coverage": coverage, "countries": countries,
         "era": era, "contamination": contamination, "anchors": anchors,
         "unusable": unusable, "generated": generated,
         "housing": housing, "wages": wages,
         "identity": identity, "tail": tail},
        figures,
        {"elapsed_seconds": elapsed, "summary": summary,
         "tail_verdict": verdict, "housing": pvn.housing_summary(housing),
         "wages": pvn.wage_summary(wages, cfg),
         "advantage": advantage(observed_cec)})
    LOGGER.info("docs/14 written (%.0fs); %d of %d return cells generated "
                "across %d countries", elapsed,
                summary["return_cells_simulated"], summary["return_cells"],
                summary["n_countries"])
    state.update({"provenance_summary": summary,
                  "provenance_countries": countries})
    return state


def step15_valuation(cfg: Dict[str, Any],
                     state: Dict[str, Any]) -> Dict[str, Any]:
    """Condition the headline on the valuation a lifetime started at."""
    val_cfg = cfg.get("valuation", {})
    if not val_cfg.get("enabled", False):
        LOGGER.info("valuation study disabled; skipping step 15")
        return state
    LOGGER.info("=== STEP 15: starting valuation ===")
    started = time.perf_counter()

    panel = state["panel"]
    jst = dl.load_jst(cfg)
    spec = lc.spec_from_config(cfg)
    sampler = state.get("sampler") or bs.from_config(panel, cfg)
    n_paths = int(cfg["bootstrap"]["n_paths"])
    chunk_size = int(cfg["bootstrap"]["chunk_size"])

    domestic = vln.trailing_yield(jst, panel.countries, panel.years)
    international = vln.international_yield(domestic)
    blended = vln.blended_yield(domestic, international,
                                float(val_cfg.get("domestic_share", 0.5)))

    # The property the whole step rests on, checked rather than claimed. Probe
    # years are spread across the panel so a leak confined to one era would
    # still be caught.
    probes = [int(y) for y in np.linspace(int(panel.years[5]),
                                          int(panel.years[-2]), 6)]
    leak_free = all(vln.depends_only_on_past(jst, panel.countries, panel.years, y)
                    for y in probes)
    if not leak_free:
        raise RuntimeError(
            "the valuation state uses contemporaneous data; conditioning on it "
            "would build look-ahead into every result in this step")
    LOGGER.info("no-look-ahead check passed at %d probe years", len(probes))

    predictive = vln.predictive_power(
        domestic, panel.dom_eq,
        [int(h) for h in val_cfg.get("horizons", (1, 10, 20, 30))])

    results = state.get("results")
    if results is None:
        state = step3_lifecycle(cfg, state)
        results = state["results"]

    starts: List[np.ndarray] = []
    start_years: List[np.ndarray] = []
    for chunk in sampler.chunks(n_paths, chunk_size):
        starts.append(vln.path_starting_yield(chunk, blended))
        start_years.append(vln.path_start_cells(chunk)[0])
    starting = np.concatenate(starts)
    start_year = np.concatenate(start_years)

    labels = list(val_cfg.get("bucket_labels", vln.BUCKET_LABELS))
    edges = [float(e) for e in val_cfg.get("quantile_edges",
                                           vln.DEFAULT_EDGES)]

    # Two labellings of the same lifetimes. The pooled one takes its
    # boundaries from the whole panel at once, so a lifetime beginning in 1910
    # is called cheap or dear against a threshold that already knows about
    # 2020 -- the yield is look-ahead-free but the *classification* is not.
    # The expanding one takes its boundaries from country-years strictly
    # before each lifetime started, which is what its investor could have
    # known. The expanding one is used for every result; the pooled one is
    # kept only to measure what the look-ahead was worth.
    hindsight_index, hindsight_meta = vln.bucket_paths(starting, edges, labels)
    cuts, prior_counts = vln.expanding_cuts(
        blended, edges, int(val_cfg.get("min_history", vln.MIN_HISTORY)))
    index, meta = vln.expanding_bucket_paths(starting, start_year, cuts,
                                             labels)
    leak = vln.bucket_agreement(hindsight_index, index)
    first_usable = int(meta.get("first_usable_year_index", -1))
    LOGGER.info("implementable buckets: %s (%.1f%% of lifetimes classified; "
                "history begins %s)",
                dict(zip(labels, meta["counts"])), meta["classified_pct"],
                int(panel.years[first_usable]) if first_usable >= 0 else "never")
    LOGGER.info("pooled boundaries would have reclassified %d of %d "
                "lifetimes (%.1f%% agreement)", leak["reassigned"],
                leak["compared"], leak["agreement_pct"])
    if meta["classified"] < 1000:
        raise RuntimeError(
            "fewer than 1,000 lifetimes have enough prior history to be "
            "classified against boundaries their own investor could have "
            "known; the conditioning below would be estimated on noise")

    # The sleeve is a mean on construction grounds; measure what the median
    # would have done rather than leaving that as an argument.
    median_blended = vln.blended_yield(
        domestic, vln.international_yield_median(domestic),
        float(val_cfg.get("domestic_share", 0.5)))
    median_starts: List[np.ndarray] = []
    for chunk in sampler.chunks(n_paths, chunk_size):
        median_starts.append(vln.path_starting_yield(chunk, median_blended))
    sleeve, sleeve_notes = vln.sleeve_comparison(
        domestic, starting, np.concatenate(median_starts),
        [float(e) for e in val_cfg.get("quantile_edges", vln.DEFAULT_EDGES)],
        labels)
    LOGGER.info("sleeve check: mean and median agree on %.1f%% of lifetimes "
                "(correlation %.3f)", sleeve_notes["agreement_pct"],
                sleeve_notes["correlation"])

    # The bucket index is built from a re-drawn chunk stream; the outcomes came
    # from step 3's. Both use the same (seed, n_paths, chunk_size), which is
    # the sampler's reproducibility contract, but a silent mismatch here would
    # show up as a null result rather than an error -- so check the one thing
    # that is cheap to check.
    n_outcomes = int(next(iter(results.values())).ruin.shape[0])
    if index.size != n_outcomes:
        raise RuntimeError(
            f"valuation buckets cover {index.size} paths but the lifecycle "
            f"results hold {n_outcomes}; the two were drawn from different "
            "samplers and conditioning them on each other would be meaningless")

    buckets = vln.by_bucket(results, index, labels, cfg, spec)
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    column = f"cec_crra_gamma{gamma:g}"
    advantage = vln.advantage_by_bucket(
        buckets, "balanced_all_equity", "target_date_fund", column)
    position = vln.current_position(
        blended, domestic, panel.years, panel.countries,
        str(val_cfg.get("reference_country", "USA")))

    distribution = pd.DataFrame({
        "bucket": labels,
        "n_paths": meta["counts"],
        "yield_floor": [-np.inf] + list(hindsight_meta["cuts"]),
        "yield_ceiling": list(hindsight_meta["cuts"]) + [np.inf],
    })

    # The boundaries an investor would actually have faced, decade by decade.
    usable = np.isfinite(cuts).all(axis=1)
    boundaries = pd.DataFrame({
        "year": panel.years[usable],
        "prior_country_years": prior_counts[usable],
        "cut_expensive_middling": cuts[usable, 0],
        "cut_middling_cheap": cuts[usable, 1],
    })

    tables = cfg["run"]["table_dir"]
    for frame, name in ((predictive, "valuation_predictive_power"),
                        (buckets, "valuation_by_bucket"),
                        (advantage, "valuation_advantage"),
                        (sleeve, "valuation_sleeve_check"),
                        (boundaries, "valuation_expanding_boundaries"),
                        (distribution, "valuation_buckets")):
        _save_table(frame, tables, name)

    figures = [str(plots.plot_valuation(predictive, buckets, advantage,
                                        domestic, blended, position,
                                        cfg["run"]["figure_dir"],
                                        boundaries=boundaries))]

    elapsed = time.perf_counter() - started
    rp.write_doc_15(
        Path("docs") / "15_starting_valuation.md", cfg,
        {"predictive": predictive, "buckets": buckets,
         "advantage": advantage, "distribution": distribution,
         "sleeve": sleeve, "boundaries": boundaries},
        figures,
        {"elapsed_seconds": elapsed, "position": position, "meta": meta,
         "probe_years": probes, "leak_free": leak_free,
         "sleeve": sleeve_notes, "leak": leak,
         "hindsight_meta": hindsight_meta,
         "first_usable_year": int(panel.years[first_usable])
         if first_usable >= 0 else None})
    LOGGER.info("docs/15 written (%.0fs); %s sits at the %.0fth percentile",
                elapsed, position["iso"], position["blended_percentile"])
    state.update({"valuation_buckets": buckets,
                  "valuation_position": position})
    return state


def step16_housing(cfg: Dict[str, Any],
                   state: Dict[str, Any]) -> Dict[str, Any]:
    """Add housing to the investable set and price it by its holding cost."""
    house_cfg = cfg.get("housing", {})
    if not house_cfg.get("enabled", False):
        LOGGER.info("housing study disabled; skipping step 16")
        return state
    LOGGER.info("=== STEP 16: housing as a fifth asset ===")
    started = time.perf_counter()

    panel = state["panel"]
    jst = dl.load_jst(cfg)
    spec = lc.spec_from_config(cfg)
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(house_cfg.get("n_paths", 20000))
    coarse = float(house_cfg.get("coarse_step", 0.10))
    fine = float(house_cfg.get("fine_step", 0.025))

    desmoothed, audit = hsg.desmoothed_panel(jst, panel)
    raw = obs.housing_returns(jst, panel.countries, panel.years)

    # Housing is recorded for fewer country-years than equity is, so the study
    # runs on the intersection. The four-asset control is solved on the same
    # restricted panel, which is what stops the restriction from being read as
    # a housing effect.
    restricted = hsg.restrict_to_housing(panel, desmoothed)
    kept = int(restricted.available.sum())
    LOGGER.info("housing restricts the panel to %d of %d country-years (%.0f%%)",
                kept, int(panel.available.sum()),
                100.0 * kept / max(int(panel.available.sum()), 1))

    sampler = bs.from_config(restricted, cfg, horizon_years=spec.horizon)
    paths = sampler.sample(n_paths,
                           chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    income = lc.simulate_income(spec, paths.n_paths,
                                np.random.default_rng(12345))

    gross = hsg.gather(paths, desmoothed)
    costs = [float(c) for c in house_cfg.get("holding_costs", hsg.cost_grid(cfg))]
    sweep = hsg.solve_sweep(paths, spec, income, cfg, gross, costs, gamma,
                            coarse, fine, variant="de-smoothed")

    frames: Dict[str, pd.DataFrame] = {"audit": audit, "sweep": sweep}
    if bool(house_cfg.get("compare_raw_series", True)):
        raw_gross = hsg.gather(paths, np.where(np.isfinite(raw), raw, np.nan))
        if np.isfinite(raw_gross).all():
            frames["raw_sweep"] = hsg.solve_sweep(
                paths, spec, income, cfg, raw_gross, costs, gamma, coarse,
                fine, variant="raw (still smoothed)")
        else:
            LOGGER.warning("raw housing series has gaps on the drawn paths; "
                           "the smoothed comparison is skipped")

    five = sweep[sweep["investable_set"] == "five assets"]
    break_even = hsg.break_even_cost(five)
    moved = hsg.displacement(five)

    age_costs = [float(c) for c in house_cfg.get("age_varying_costs", [])]
    if age_costs:
        frames["age_varying"] = hsg.age_varying_check(
            paths, spec, income, cfg, gross, age_costs, gamma, five)

    tables = cfg["run"]["table_dir"]
    _save_table(audit, tables, "housing_desmoothing_audit")
    _save_table(sweep, tables, "housing_cost_sweep")
    _save_table(moved, tables, "housing_displacement")
    if "raw_sweep" in frames:
        _save_table(frames["raw_sweep"], tables, "housing_raw_sweep")
    if "age_varying" in frames:
        _save_table(frames["age_varying"], tables, "housing_age_varying")

    figures = [str(plots.plot_housing_sweep(
        sweep, audit, frames.get("raw_sweep"), break_even,
        cfg["run"]["figure_dir"]))]

    elapsed = time.perf_counter() - started
    notes = {
        "elapsed_seconds": elapsed,
        "break_even": break_even,
        "gamma": gamma,
        "n_paths": int(paths.n_paths),
        "kept_cells": kept,
        "total_cells": int(panel.available.sum()),
        "moments_desmoothed": hsg.moments(desmoothed),
        "moments_raw": hsg.moments(raw),
    }
    frames["displacement"] = moved
    rp.write_doc_16(Path("docs") / "16_housing.md", cfg, frames, figures, notes)
    LOGGER.info("docs/16 written (%.0fs); housing break-even holding cost %s",
                elapsed,
                f"{break_even:.2%}" if np.isfinite(break_even) else "not reached")
    state.update({"housing_sweep": sweep, "housing_break_even": break_even})
    return state


def _read_table_if_present(cfg: Mapping[str, Any], name: str):
    """A previously written table, so a step can reference an earlier one.

    Returns None when the table is absent, which is the normal case for a
    fresh checkout or a partial run -- the prose that would have used it is
    simply omitted rather than the step failing.
    """
    path = Path(cfg["run"]["table_dir"]) / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def step17_mortgage(cfg: Dict[str, Any],
                    state: Dict[str, Any]) -> Dict[str, Any]:
    """How much of the house should be borrowed, and at what age."""
    mg_cfg = cfg.get("mortgage", {})
    if not mg_cfg.get("enabled", False):
        LOGGER.info("mortgage study disabled; skipping step 17")
        return state
    LOGGER.info("=== STEP 17: a mortgage on the housing sleeve ===")
    started = time.perf_counter()

    panel = state["panel"]
    jst = dl.load_jst(cfg)
    spec = lc.spec_from_config(cfg)
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(mg_cfg.get("n_paths", 20000))
    grid = [float(v) for v in mg_cfg.get("lvr_grid", mgg.DEFAULT_LVR_GRID)]
    spreads = [float(v) for v in mg_cfg.get("spreads", (0.0, 0.02, 0.04))]
    holding_cost = float(mg_cfg.get("holding_cost", 0.02))
    rate_base = str(mg_cfg.get("rate_base", "bill"))
    coarse = float(mg_cfg.get("coarse_step", 0.10))
    fine = float(mg_cfg.get("fine_step", 0.025))

    desmoothed, _ = hsg.desmoothed_panel(jst, panel)
    restricted = hsg.restrict_to_housing(panel, desmoothed)
    sampler = bs.from_config(restricted, cfg, horizon_years=spec.horizon)
    paths = sampler.sample(n_paths,
                           chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    income = lc.simulate_income(spec, paths.n_paths,
                                np.random.default_rng(12345))
    gross = hsg.gather(paths, desmoothed)

    # A borrower pays the yield they contracted at, not the total return a
    # bondholder earns; the long-rate variant is built from the yield.
    base_rate = None
    if rate_base == "long_yield":
        yields = obs.real_long_yield(jst, panel.countries, panel.years)
        base_rate = hsg.gather(paths, yields)
        if not np.isfinite(base_rate).all():
            raise RuntimeError(
                "the long-yield series has gaps on the drawn paths; either "
                "restrict the panel further or price the loan off bills")

    sweep = mgg.sweep_spread(paths, spec, income, cfg, gross, spreads, gamma,
                             holding_cost, grid, rate_base, coarse, fine,
                             base_rate=base_rate)
    break_even = mgg.break_even_spread(sweep)

    # The age-by-age schedule at one named price of credit, reported in full.
    detail_spread = float(mg_cfg.get("detail_spread", spreads[len(spreads) // 2]))
    net = hsg.net_of_cost(gross, holding_cost)
    evaluator = mgg.MortgageEvaluator(
        paths, spec, income, cfg, extra={hsg.HOUSING: net},
        spread=detail_spread, rate_base=rate_base, base_rate=base_rate)
    solved = mgg.alternate(evaluator, gamma, grid,
                           rounds=int(mg_cfg.get("rounds", 3)),
                           coarse_step=coarse, fine_step=fine,
                           label=f"[detail @ {detail_spread:.1%}] ")
    schedule = mgg.schedule_frame(solved["lvr"], solved["weights"], spec,
                                  evaluator.assets)
    option = mgg.terminal_option_check(solved["lvr"], spec)
    # Never describe structure in a solved schedule before measuring what the
    # structure is worth: a coarse coordinate search produces a jagged line
    # whose jaggedness mostly sits on a flat part of the surface.
    profile = mgg.lvr_deviation_profile(
        evaluator, mgg._as_schedule(solved["weights"], spec.horizon),
        solved["lvr"], gamma, spec)
    profile_notes = mgg.profile_summary(profile)
    LOGGER.info("LVR schedule: %d of %d ages carry a material (>%.0fbp) "
                "loan-to-value decision",
                profile_notes["material_ages"], profile_notes["ages"],
                profile_notes["material_bp"])
    lvr_curve = mgg.best_constant_lvr(
        evaluator, mgg._as_schedule(solved["weights"], spec.horizon), gamma,
        grid)[2]
    LOGGER.info("solved mortgage: %.0f%% LVR while working, %.0f%% retired; "
                "terminal lift %.0f pp%s",
                100.0 * schedule[schedule["phase"] == "working"]["lvr"].mean(),
                100.0 * schedule[schedule["phase"] == "retired"]["lvr"].mean(),
                100.0 * option["terminal_lift"],
                " (looks like the limited-liability option)"
                if option["looks_like_the_option"] else "")

    # Why borrowing against housing behaves differently from borrowing
    # against the portfolio: the four numbers that settle it.
    comparison = hsg.investable_set_comparison(panel, desmoothed, holding_cost)
    comparison["equity_leg_correlation"] = comparison.attrs.get(
        "equity_leg_correlation", float("nan"))

    tables = cfg["run"]["table_dir"]
    for frame, name in ((comparison, "mortgage_asset_comparison"),
                        (sweep, "mortgage_spread_sweep"),
                        (schedule, "mortgage_lvr_schedule"),
                        (lvr_curve, "mortgage_constant_lvr_curve"),
                        (profile, "mortgage_lvr_deviation_profile"),
                        (solved["history"], "mortgage_alternation")):
        _save_table(frame, tables, name)

    figures = [str(plots.plot_mortgage(sweep, schedule, lvr_curve,
                                       break_even, cfg["run"]["figure_dir"],
                                       profile=profile))]

    elapsed = time.perf_counter() - started
    rp.write_doc_17(
        Path("docs") / "17_mortgage.md", cfg,
        {"sweep": sweep, "schedule": schedule, "curve": lvr_curve,
         "history": solved["history"], "profile": profile,
         "comparison": comparison,
         "leverage": state.get("leverage_sweep") if "leverage_sweep" in state
         else _read_table_if_present(cfg, "leverage_sweep")},
        figures,
        {"elapsed_seconds": elapsed, "break_even": break_even,
         "gamma": gamma, "n_paths": int(paths.n_paths),
         "holding_cost": holding_cost, "rate_base": rate_base,
         "detail_spread": detail_spread, "option": option,
         "lvr_cap": mgg.LVR_CAP, "profile": profile_notes,
         "solved_cec": float(solved["cec"]),
         "solved_mean_lvr": float(solved["lvr"].mean())})
    LOGGER.info("docs/17 written (%.0fs)", elapsed)
    state.update({"mortgage_sweep": sweep, "mortgage_schedule": schedule})
    return state




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Step 18
# ---------------------------------------------------------------------------
def step18_sleeve(cfg: Dict[str, Any],
                  state: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the international sleeve by GDP weight and re-run the headline."""
    sleeve_cfg = cfg.get("sleeve", {})
    if not sleeve_cfg.get("enabled", False):
        LOGGER.info("sleeve study disabled; skipping step 18")
        return state
    LOGGER.info("=== STEP 18: how the international sleeve is weighted ===")
    started = time.perf_counter()

    spec = state.get("spec") or lc.spec_from_config(cfg)
    strategies = state.get("strategies") or lc.build_strategies(cfg, spec)
    n_paths = int(sleeve_cfg.get("n_paths", cfg["bootstrap"]["n_paths"]))
    weightings = [str(w) for w in sleeve_cfg.get("weightings", slv.WEIGHTINGS)]

    panels = slv.build_panels(cfg, weightings)
    LOGGER.info("built %d panels on identical history: %s",
                len(panels), ", ".join(panels))

    # The headline summariser itself, not a copy of it: the whole claim of
    # this section is that only the weighting differs between the two runs.
    def summarise(panel: dl.Panel, paths: int) -> pd.DataFrame:
        return _run_variant(cfg, panel, spec, strategies, paths)

    comparison = slv.compare_headline(cfg, panels, summarise, n_paths)
    ranking = slv.ranking_shift(comparison)
    moments = slv.sleeve_moments(panels)
    conc = slv.concentration(cfg, panels[weightings[0]].countries,
                             panels[weightings[0]].years, weightings)
    spectrum = slv.concentration_vs_outcome(comparison, conc, moments)
    findings = slv.verdict(comparison)
    LOGGER.info("sleeve verdict: winner changes=%s, ordering changes=%s",
                findings["winner_changes"], findings["ordering_changes"])

    tables = cfg["run"]["table_dir"]
    for frame, name in ((comparison, "sleeve_comparison"),
                        (ranking, "sleeve_ranking"),
                        (moments, "sleeve_moments"),
                        (conc, "sleeve_concentration"),
                        (spectrum, "sleeve_spectrum")):
        if len(frame):
            _save_table(frame, tables, name)

    figures = [str(plots.plot_sleeve_weighting(
        conc, ranking, spectrum, cfg["run"]["figure_dir"]))]

    elapsed = time.perf_counter() - started
    rp.write_doc_18(
        Path("docs") / "18_sleeve_weighting.md", cfg,
        {"comparison": comparison, "ranking": ranking, "moments": moments,
         "concentration": conc, "spectrum": spectrum},
        figures, {"elapsed_seconds": elapsed, "n_paths": n_paths,
                  "gamma": float(cfg["utility"]["baseline_risk_aversion"]),
                  "verdict": findings})
    LOGGER.info("docs/18 written (%.0fs)", elapsed)
    state["sleeve_comparison"] = comparison
    return state


# ---------------------------------------------------------------------------
# Step 19
# ---------------------------------------------------------------------------
def step19_panel(cfg: Dict[str, Any],
                 state: Dict[str, Any]) -> Dict[str, Any]:
    """Delete one country at a time, and one era at a time; write docs/19."""
    pr_cfg = cfg.get("panel_robustness", {})
    if not pr_cfg.get("enabled", False):
        LOGGER.info("panel-robustness study disabled; skipping step 19")
        return state
    LOGGER.info("=== STEP 19: how much rests on any one country or era ===")
    started = time.perf_counter()

    spec = state.get("spec") or lc.spec_from_config(cfg)
    strategies = state.get("strategies") or lc.build_strategies(cfg, spec)
    n_paths = int(pr_cfg.get("n_paths", cfg["bootstrap"]["n_paths"]))

    # The headline summariser itself, so a delete-one run and the headline are
    # scored by the same code rather than by two copies of it.
    def summarise(panel: dl.Panel, paths: int,
                  override: Dict[str, Any] | None = None) -> pd.DataFrame:
        return _run_variant(override or cfg, panel, spec, strategies, paths)

    full_panel = dl.build_tier_a(cfg)
    full = summarise(full_panel, n_paths)

    floor = pr.noise_floor(cfg, summarise, n_paths,
                           [int(s) for s in pr_cfg.get("seeds", pr.DEFAULT_SEEDS)])
    floor_stats = pr.floor_summary(floor)
    LOGGER.info("noise floor: the headline gap spans %.3f points across "
                "%d seeds on an unchanged panel",
                floor_stats["range_pct"], floor_stats["seeds"])

    loco = pr.leave_one_out(cfg, summarise, n_paths, full_panel.countries)
    infl = pr.influence(loco, full)
    jack = pr.jackknife(infl)
    LOGGER.info("jackknife: gap %.2f%% +/- %.2f (1 se), CI [%.2f, %.2f]",
                jack["baseline_gap_pct"], jack["standard_error"],
                jack["ci_low"], jack["ci_high"])

    chan = pr.channels(loco, full)
    profile = pr.market_profile(cfg, full_panel)
    why = pr.explain(profile, infl, chan)
    LOGGER.info("channel split: %s matters most to everyone else's sleeve "
                "(%.2f pp of its compound return), %s most as its own home "
                "market", why["sleeve_pole"], why["sleeve_pole_delta"],
                why["home_pole"])

    periods = pr.subperiods(full_panel, summarise, n_paths,
                            [int(w) for w in pr_cfg.get("windows",
                                                        pr.DEFAULT_WINDOWS)])
    period = pr.period_summary(periods)
    findings = pr.verdict(infl, jack, floor_stats, period)
    LOGGER.info("panel verdict: survives every deletion=%s, all windows "
                "hold=%s", findings["survives_every_deletion"],
                findings.get("all_windows_hold"))

    tables = cfg["run"]["table_dir"]
    for frame, name in ((loco, "panel_leave_one_out"),
                        (infl, "panel_influence"),
                        (chan, "panel_channels"),
                        (profile, "panel_market_profile"),
                        (periods, "panel_subperiods"),
                        (period, "panel_period_summary"),
                        (floor, "panel_noise_floor")):
        if len(frame):
            _save_table(frame, tables, name)

    figures = [str(plots.plot_panel_robustness(
        infl, period, floor_stats, jack, cfg["run"]["figure_dir"],
        profile=profile))]

    elapsed = time.perf_counter() - started
    rp.write_doc_19(
        Path("docs") / "19_panel_robustness.md", cfg,
        {"influence": infl, "period": period, "loco": loco,
         "channels": chan, "profile": profile},
        figures, {"elapsed_seconds": elapsed, "n_paths": n_paths,
                  "gamma": float(cfg["utility"]["baseline_risk_aversion"]),
                  "verdict": findings, "jackknife": jack,
                  "floor": floor_stats, "why": why})
    LOGGER.info("docs/19 written (%.0fs)", elapsed)
    state["panel_influence"] = infl
    return state


# ---------------------------------------------------------------------------
# Step 20
# ---------------------------------------------------------------------------
def step20_fees(cfg: Dict[str, Any],
                state: Dict[str, Any]) -> Dict[str, Any]:
    """What costs do to the headline, and how small a cost undoes it."""
    fee_cfg = cfg.get("fees", {})
    if not fee_cfg.get("enabled", False):
        LOGGER.info("fee study disabled; skipping step 20")
        return state
    LOGGER.info("=== STEP 20: fees, and the differential that undoes it ===")
    started = time.perf_counter()

    spec = state.get("spec") or lc.spec_from_config(cfg)
    strategies = state.get("strategies") or lc.build_strategies(cfg, spec)
    n_paths = int(fee_cfg.get("n_paths", cfg["bootstrap"]["n_paths"]))
    pair = (str(fee_cfg.get("challenger", "international_equity")),
            str(fee_cfg.get("incumbent", "balanced_all_equity")))

    def summarise(panel: dl.Panel, paths: int) -> pd.DataFrame:
        return _run_variant(cfg, panel, spec, strategies, paths)

    panel = state.get("panel") or dl.build_panel(cfg)
    common_grid = [float(v) for v in fee_cfg["common_grid"]]
    diff_grid = [float(v) for v in fee_cfg["differential_grid"]]

    common = fee.sweep_common(panel, summarise, n_paths, common_grid)
    differential = fee.sweep_differential(panel, summarise, n_paths, diff_grid)
    common_curve = fee.gap_curve(common, "fee", pair)
    diff_curve = fee.gap_curve(differential, "differential", pair)
    anchors = fee.anchor_table(diff_curve, "differential")
    findings = fee.verdict(common, differential, pair, anchors)
    LOGGER.info("fee break-even: common %.0f bp, international differential "
                "%.1f bp", findings["break_even_common_bp"],
                findings["break_even_differential_bp"])

    tables = cfg["run"]["table_dir"]
    for frame, name in ((common, "fee_common"),
                        (differential, "fee_differential"),
                        (common_curve, "fee_common_curve"),
                        (diff_curve, "fee_differential_curve"),
                        (anchors, "fee_anchors")):
        if len(frame):
            _save_table(frame, tables, name)

    figures = [str(plots.plot_fees(common_curve, diff_curve, anchors,
                                   findings, cfg["run"]["figure_dir"]))]

    elapsed = time.perf_counter() - started
    rp.write_doc_20(
        Path("docs") / "20_fees.md", cfg,
        {"common": common_curve, "differential": diff_curve,
         "anchors": anchors,
         "ranking": fee.ranking_at(differential, "differential",
                                   diff_grid[0])},
        figures, {"elapsed_seconds": elapsed, "n_paths": n_paths,
                  "gamma": float(cfg["utility"]["baseline_risk_aversion"]),
                  "verdict": findings, "pair": pair})
    LOGGER.info("docs/20 written (%.0fs)", elapsed)
    state["fee_curve"] = diff_curve
    return state


STEPS = {1: step1_dataset, 2: step2_bootstrap, 3: step3_lifecycle,
         4: step4_report, 5: step5_sensitivity, 6: step6_spending,
         7: step7_glide_path, 8: step8_hedging, 9: step9_retirement_timing,
         10: step10_saving, 11: step11_accumulation,
         12: step12_allocation, 13: step13_leverage,
         14: step14_provenance, 15: step15_valuation,
         16: step16_housing, 17: step17_mortgage,
         18: step18_sleeve, 19: step19_panel,
         20: step20_fees}


def run(config_path: str = "config.yaml",
        steps: Sequence[int] = tuple(range(1, 21)),
        quick: bool = False) -> Dict[str, Any]:
    """Execute the pipeline and return the accumulated state."""
    cfg = dl.load_config(config_path)
    if quick:
        cfg = _apply_quick(cfg)
    np.random.seed(int(cfg["run"]["seed"]))
    for directory in (cfg["run"]["output_dir"], cfg["run"]["figure_dir"],
                      cfg["run"]["table_dir"], cfg["run"]["cache_dir"],
                      cfg["run"]["processed_dir"], "docs"):
        Path(directory).mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {}
    started = time.perf_counter()
    for step in sorted(steps):
        if step == 1:
            state = STEPS[step](cfg)
        else:
            if "panel" not in state:
                LOGGER.info("step %s needs the panel; building it now", step)
                state = step1_dataset(cfg)
            state = STEPS[step](cfg, state)
    LOGGER.info("pipeline finished in %.1fs", time.perf_counter() - started)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", nargs="+", type=int,
                        default=list(range(1, 21)),
                        choices=list(range(1, 21)))
    parser.add_argument("--quick", action="store_true",
                        help="small N for smoke tests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    run(config_path=args.config, steps=args.steps, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
