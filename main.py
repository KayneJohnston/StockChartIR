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
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src import bootstrap as bs
from src import data_loader as dl
from src import lifecycle as lc
from src import plots
from src import report as rp
from src import sensitivity as sn
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
                  "tornado": tornado, "verdict": verdict})
    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
STEPS = {1: step1_dataset, 2: step2_bootstrap, 3: step3_lifecycle,
         4: step4_report, 5: step5_sensitivity}


def run(config_path: str = "config.yaml",
        steps: Sequence[int] = (1, 2, 3, 4, 5),
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
                        default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    parser.add_argument("--quick", action="store_true",
                        help="small N for smoke tests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    run(config_path=args.config, steps=args.steps, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
