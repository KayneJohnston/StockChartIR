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

from src import bootstrap as bs
from src import data_loader as dl
from src import glidepath as gp
from src import hedging as hg
from src import lifecycle as lc
from src import plots
from src import report as rp
from src import retirement as rt
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
    for tier in ("A", "B"):
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
        "n_tier_b": sum(1 for t in panel.tier if t == "B"),
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
# Entry point
# ---------------------------------------------------------------------------
STEPS = {1: step1_dataset, 2: step2_bootstrap, 3: step3_lifecycle,
         4: step4_report, 5: step5_sensitivity, 6: step6_spending,
         7: step7_glide_path, 8: step8_hedging, 9: step9_retirement_timing}


def run(config_path: str = "config.yaml",
        steps: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
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
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--quick", action="store_true",
                        help="small N for smoke tests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    run(config_path=args.config, steps=args.steps, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
