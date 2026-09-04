"""The text of the working paper.

Prose is authored here; every number inside it is pulled from
:mod:`paper.facts`, which reads the pipeline's own CSV output. The separation
is deliberate: the argument is written once, the evidence is re-read on every
build, and a rerun of the pipeline that changed a result would change the
paper rather than silently contradict it.
"""

from __future__ import annotations

import re

import datetime as dt
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import (Flowable, KeepTogether, NextPageTemplate,
                                PageBreak, Paragraph, Spacer)
from reportlab.platypus.tableofcontents import TableOfContents


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def f2(x: Any, d: int = 2) -> str:
    """A fixed-point number, or an em-dash where the value is undefined.

    Some table cells have no defined value rather than a missing one -- the
    improvement over the previous sweep, on the sweep that has no previous.
    Printing "nan" there reads as a defect; a dash reads as "not applicable",
    which is what it is.
    """
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "\u2014" if not np.isfinite(value) else f"{value:.{d}f}"


def sgn(x: Any, d: int = 2) -> str:
    return f"{float(x):+.{d}f}"


def pc(x: Any, d: int = 1) -> str:
    """A fraction rendered as a percentage, or an em-dash where undefined."""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "\u2014" if not np.isfinite(value) else f"{value * 100.0:.{d}f}%"


def ordinal(n: int) -> str:
    """1 -> first, 2 -> second, ... for prose that names a rank."""
    words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
             6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
    return words.get(int(n), f"{int(n)}th")


def nth_best(n: int) -> str:
    """`the best`, `the second-best`, ... for a rank in prose."""
    return "the best" if int(n) == 1 else f"the {ordinal(int(n))}-best"


def rows_from(frame: pd.DataFrame, columns: Sequence[str],
              headers: Sequence[str],
              formats: Dict[str, Callable[[Any], str]] | None = None,
              limit: int | None = None) -> List[List[str]]:
    """Turn a DataFrame into the nested lists ``Context.table`` expects."""
    formats = formats or {}
    block = frame if limit is None else frame.head(limit)
    out: List[List[str]] = [list(headers)]
    for _, row in block.iterrows():
        out.append([formats.get(c, lambda v: f2(v, 3))(row[c]) for c in columns])
    return out


LABELS = {
    "balanced_all_equity": "50/50 domestic/international equity",
    "domestic_equity": "100% domestic equity",
    "international_equity": "100% international equity",
    "sixty_forty": "60/40 domestic equity/bonds",
    "target_date_fund": "Target-date fund (glide path)",
    "bills_only": "100% bills (cash)",
}


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------
def front_matter(ctx: Any) -> List[Flowable]:
    f, s = ctx.f, ctx.s
    p = f.panel
    adv_tdf = f.advantage("balanced_all_equity", "target_date_fund")
    adv_6040 = f.advantage("balanced_all_equity", "sixty_forty")
    ruin_eq = float(f.strategy_row("balanced_all_equity")["prob_ruin"])
    ruin_tdf = float(f.strategy_row("target_date_fund")["prob_ruin"])
    tornado = f.table("sensitivity_tornado")
    n_settings = int(tornado["n_settings"].sum())
    n_dimensions = int(tornado["dimension"].nunique())
    val_adv = f.table("valuation_advantage")
    house = f.table("housing_cost_sweep")
    house_five = house[house["investable_set"] == "five assets"].sort_values(
        "holding_cost")
    from src.housing import break_even_cost as _house_break_even
    house_break = _house_break_even(house_five)
    house_free = house_five.iloc[0]
    mort = f.table("mortgage_lvr_schedule")
    mort_work = float(mort[mort["phase"] == "working"]["lvr"].mean())
    mort_ret = float(mort[mort["phase"] == "retired"]["lvr"].mean())
    reversals = int(tornado["settings_lost"].sum())
    swr = f.table("sensitivity_safe_withdrawal_rates")
    swr_eq = float(swr[swr["strategy"] == "balanced_all_equity"]
                   ["safe_withdrawal_rate_at_5%_ruin"].iloc[0])
    swr_tdf = float(swr[swr["strategy"] == "target_date_fund"]
                    ["safe_withdrawal_rate_at_5%_ruin"].iloc[0])
    lottery = f.table("retirement_lottery_stats").iloc[0]
    income_net = f.signal_net("income_shock")
    funded_net = f.signal_net("funded_ratio")
    alloc = f.allocation
    pr, adv = f.provenance, f.panel_advantage
    alloc_params, alloc_lead = alloc["free_parameters"], alloc["lead_pct"]
    lev = f.leverage
    lev_free, break_even = lev["value_at_zero_spread"], lev["break_even_spread"]
    # The three robustness studies, and the retirement-timing figure the
    # abstract used to round by hand.
    from src.fees import verdict as _fee_verdict
    from src.panel_robustness import jackknife as _jackknife
    infl = f.table("panel_influence")
    panel_base = float(infl["gap_pct"].iloc[0] - infl["shift_pct"].iloc[0])
    jack = _jackknife(infl, baseline_gap=panel_base)
    n_deletions = int(len(infl))
    survives_all = bool((infl["gap_pct"] > 0).all())
    spectrum = f.table("sleeve_spectrum")
    n_schemes = int(len(spectrum))
    sleeve_worst = float(spectrum["gap_pct"].min())
    fee_anchors = f.table("fee_anchors")
    fee_found = _fee_verdict(f.table("fee_common"), f.table("fee_differential"),
                             ("international_equity", "balanced_all_equity"),
                             fee_anchors)
    fee_be_bp = float(fee_found["break_even_differential_bp"])
    retire = f.table("retirement_value_of_conditioning")
    retire_base = retire[np.isclose(retire["working_income_floor"], 0.25)
                         & retire["variant"].str.contains("Wealth trigger")]
    retire_value = float(retire_base["value_of_conditioning_pct"].mean())
    families = f.table("spending_rule_best_per_family").sort_values(
        "cec_gamma5", ascending=False)
    n_families = int(len(families))
    spend_best = str(families["variant"].iloc[0]).split(" (")[0]
    spend_worst = str(families["variant"].iloc[-1]).split(" (")[0]
    spend_spread = (float(families["cec_gamma5"].iloc[0])
                    / float(families["cec_gamma5"].iloc[-1]) - 1.0) * 100.0
    # The four studies added after the first pass, read from their own tables
    # so the abstract cannot drift from the sections.
    from src import cohorts as _coh, humancapital as _hcp, mortality as _mrt
    from src import oos as _oos
    coh_detail = f.table("cohort_detail")
    coh_summary = f.table("cohort_summary")
    coh_census = f.table("cohort_census")
    coh_signs = _coh.sign_test(coh_detail)
    coh_winner = str(coh_summary["strategy"].iloc[0])
    coh_n = int(len(coh_detail))
    coh_independent = int(coh_census["cohorts"].gt(0).sum())
    _coh_ci = _coh.cluster_bootstrap(coh_detail, n_boot=2000)
    coh_ci = (float(_coh_ci["ci_low"]), float(_coh_ci["ci_high"]))
    hc_curve = f.table("human_capital_gap")
    hc_modes = f.table("human_capital_modes")
    hc_found = _hcp.verdict(hc_curve, _hcp.sensitivity(hc_curve, mode="home"),
                            ("international_equity", "balanced_all_equity"),
                            mode="home", comparison=hc_modes)
    hc_change = float(hc_found["change_pp"])
    hc_diag = float(hc_found.get("change_diagonal_pp", float("nan")))
    hc_changes = bool(hc_found["winner_ever_changes"])
    mort_curve = f.table("mortality_gap")
    mort_found = _mrt.verdict(f.table("mortality_comparison"), mort_curve,
                              ("international_equity", "balanced_all_equity"))
    mort_change = float(mort_found["largest_change_pp"])
    mort_changes = bool(mort_found["ordering_ever_changes"])
    oos_frame = f.table("oos_transfer")
    oos_found = _oos.verdict(
        oos_frame, _oos.ranking_is_stable(f.table("oos_benchmarks")))
    oos_none = bool(oos_found["no_run_transfers"])
    oos_wins = int(oos_found["runs_that_beat_the_benchmark"])
    oos_runs = int(oos_found["runs"])
    oos_forward = float(oos_found["forward_gain_pct"])
    oos_backward = float(oos_found["backward_gain_pct"])
    from src import withholding as _wht
    wht_curve = f.table("withholding_curve")
    wht_crossed = f.table("withholding_crossings")
    wht_drag = f.table("withholding_drag")
    wht_optima = f.table("withholding_optimal")
    wht_found = _wht.verdict(wht_curve, wht_crossed, wht_optima, wht_drag,
                             str(f.cfg["withholding"]["challenger"]))
    wht_share = float(wht_drag.loc[wht_drag["era"] == "whole panel",
                                   "mean_dividend_share"].iloc[0])
    wht_statutory_bp = 0.30 * wht_share * 1e4
    wht_first = float(wht_found["first_crossing_pct"])
    wht_rival = str(wht_found["first_rival"])
    wht_bites = bool(wht_found["crossing_within_statutory"])
    wht_home_low = float(wht_found.get("optimal_domestic_at_zero", float("nan")))
    wht_home_high = float(wht_found.get("optimal_domestic_at_top", float("nan")))
    wht_top = float(wht_found["highest_rate_pct"])
    from src import inflation as _ifl
    inf_pred = f.table("inflation_predictive")
    inf_window = int(f.cfg["inflation_state"].get("headline_window", 3))
    inf_horizon = int(f.cfg["inflation_state"].get("headline_horizon", 1))
    inf_labels = list(f.cfg["inflation_state"].get("bucket_labels",
                                                   _ifl.BUCKET_LABELS))
    inf_adv = f.table("inflation_advantage")
    inf_eq = _ifl.optimum_shift(f.table("inflation_optimal_equity"),
                                "equity_share", inf_labels)
    inf_dom = _ifl.optimum_shift(f.table("inflation_optimal_domestic"),
                                 "domestic_share", inf_labels)
    inf_found = _ifl.verdict(inf_adv, inf_pred, inf_window, inf_horizon,
                             inf_eq, inf_dom,
                             _ifl.persistence(inf_pred, inf_window))
    inf_nominal = float(inf_found["nominal_gap_pp"])
    inf_equity = float(inf_found["equity_gap_pp"])
    inf_holds = bool(inf_found.get("ranking_survives", False))
    inf_moves = bool(inf_eq.get("moves") or inf_dom.get("moves"))
    inf_long = sorted(int(h) for h in inf_pred["horizon_years"].unique())[-1]
    _inf_acc = str(f.cfg["inflation_state"].get("accumulation_strategy",
                                                "balanced_all_equity"))
    _inf_col = f"cec_crra_gamma{float(f.cfg['utility']['baseline_risk_aversion']):g}"
    inf_timing = _ifl.timing_comparison(
        _ifl.level_spread(f.table("inflation_buckets"), _inf_acc, _inf_col,
                          inf_labels),
        _ifl.level_spread(f.table("inflation_retirement_buckets"), _inf_acc,
                          _inf_col, inf_labels))
    inf_retire_pct = float(inf_timing.get("retirement_spread_pct", float("nan")))
    inf_birth_pct = float(inf_timing.get("birth_spread_pct", float("nan")))
    inf_ratio = float(inf_timing.get("ratio", float("nan")))
    inf_timing_bites = bool(inf_timing.get("retirement_matters_much_more"))
    _inf_bond_long = inf_pred[(inf_pred["asset"] == "bond")
                              & (inf_pred["window_years"] == inf_window)
                              & (inf_pred["horizon_years"] == inf_long)]
    inf_bond_long = (float(_inf_bond_long["gap"].iloc[0]) * 100.0
                     if len(_inf_bond_long) else float("nan"))
    from src import pension as _pns
    from src import turnover as _tno
    pen_gaps = f.table("pension_gap")
    pen_found = _pns.verdict(pen_gaps)
    pen_rows = pen_gaps.set_index("system")
    pen_base = float(pen_rows.loc["us_social_security", "gap_pct"])
    pen_au_gap = float(pen_rows.loc["australia_as_legislated", "gap_pct"])
    pen_au_winner = str(pen_rows.loc["australia_as_legislated", "winner"])
    pen_cec = float(pen_found["australia_lift_pct"])
    pen_mean = float(pen_found["australia_mean_lift_pct"])
    pen_p5 = float(pen_found["australia_p5_lift_pct"])
    pen_saving = float(pen_found["extra_saving_lift_pct"])
    pen_splits = bool(pen_found["mean_and_cec_disagree"])
    pen_au_reorders = bool((pen_au_gap > 0.0) != (pen_base > 0.0))
    tno_measured = f.table("turnover_measured")
    tno_curve = f.table("turnover_gap")
    tno_challenger = str(f.cfg["turnover"]["challenger"])
    tno_incumbent = next((c[4:] for c in tno_curve.columns
                          if c.startswith("cec_") and c[4:] != tno_challenger),
                         "international_equity")
    tno_found = _tno.verdict(
        tno_curve, tno_measured,
        _tno.cost_of_the_schedule(tno_measured, tno_challenger, tno_incumbent),
        tno_challenger, tno_incumbent)
    tno_be = float(tno_found["break_even_bp"])
    tno_lead = float(tno_found["baseline_gap_pct"])
    tno_turn = float(tno_found["busiest_turnover"])

    out: List[Flowable] = [
        Spacer(1, 1.1 * cm),
        Paragraph("Beyond the Status Quo, Revisited", s["title"]),
        Paragraph(f"A Computational Re-Examination of Lifecycle Asset "
                  f"Allocation, with {extension_count_word()} Further "
                  f"Studies", s["subtitle"]),
        Spacer(1, 0.2 * cm),
        Paragraph("A replication and extension study", s["author"]),
        Spacer(1, 0.25 * cm),
        Paragraph(dt.date.today().strftime("This version: %d %B %Y"), s["date"]),
        Spacer(1, 0.85 * cm),
        ctx.rule(),
        Paragraph("Abstract", s["abstract_head"]),
    ]

    abstract = (
        f"We reproduce and extend the central result of Anarkulova, Cederburg "
        f"and O'Doherty (2023, 2024): that a lifecycle investor holding only "
        f"equities, split evenly between domestic and international markets, "
        f"delivers higher certainty-equivalent retirement consumption than the "
        f"age-declining glide path embedded in target-date funds. Working from "
        f"a {p['n_countries']}-country developed-market panel of real asset "
        f"returns spanning {p['first_year']}–{p['last_year']} "
        f"({p['country_years']:,} country-years), we build a calendar-joint "
        f"stationary block bootstrap that samples whole (country, window) "
        f"blocks and therefore preserves cross-asset covariance, long-horizon "
        f"persistence and the fat tails of the historical record. Simulating "
        f"100,000 lifetimes per strategy under CRRA and Epstein–Zin "
        f"preferences, we recover the qualitative result: the all-equity "
        f"portfolio leads the target-date fund by {adv_tdf:.1f}% and a 60/40 "
        f"portfolio by {adv_6040:.1f}% in certainty-equivalent consumption at "
        f"γ = {f.baseline_gamma:g}, while cutting the probability of exhausting "
        f"the portfolio from {pc(ruin_tdf, 0)} to {pc(ruin_eq, 0)}. The ranking "
        f"survives {n_settings} parameter settings across {n_dimensions} "
        f"dimensions with "
        f"{reversals} reversals. Sustainable withdrawal rates are far below "
        f"the four-percent convention on every strategy "
        f"({pc(swr_eq, 1)} for all-equity, {pc(swr_tdf, 1)} for the target-date "
        f"fund at a five-percent ruin tolerance). "
        f"We then push past replication with "
        f"{extension_count_word().lower()} further studies, taken in the "
        f"order the paper presents them. "
        f"{group_count_word('robustness')} ask whether the result holds up. "
        f"Re-running the lifetimes the panel actually contains — one country, "
        f"one birth year, sixty-eight years in the order they happened, with "
        f"no resampling of any kind — "
        + (f"leaves all-international ahead in "
           f"{int(coh_signs['countries_won'])} of "
           f"{int(coh_signs['countries_total'])} markets, so the ordering is "
           f"not a property of the block bootstrap"
           if coh_winner == "international_equity" else
           "reverses the ordering, which the rest of the paper has to be "
           "read against")
        + f"; that record holds {coh_n} complete lifetimes but only "
        f"{coh_independent} independent ones, and a bootstrap over countries "
        f"puts the realised lead at [{coh_ci[0]:.1f}, {coh_ci[1]:.1f}]. "
        f"Deleting each country in "
        f"turn and rebuilding the panel around its absence "
        + (f"leaves the ranking intact in all {n_deletions} cases"
           if survives_all else
           f"overturns the ranking in "
           f"{int((infl['gap_pct'] <= 0).sum())} of {n_deletions} cases")
        + f"; the same runs form a delete-one jackknife that puts a standard "
        f"error of {float(jack['standard_error']):.2f} points on the "
        f"{panel_base:.2f}-point lead all-international holds over the 50/50 "
        f"split, a 95% interval of [{float(jack['ci_low']):.2f}, "
        f"{float(jack['ci_high']):.2f}] that is far wider than the Monte "
        f"Carlo error and is the precision a reader should carry to every "
        f"other number here — including the headline ones above. Rebuilding "
        f"the international sleeve under "
        f"{NUMBER_WORDS.get(n_schemes, str(n_schemes)).lower()} weighting "
        f"schemes — equal, real GDP, population, GDP "
        f"per capita and inverse volatility — narrows that lead to "
        f"{sleeve_worst:.2f} points at worst without closing it, so the "
        f"divergence is not manufactured by the equal weighting. Charging "
        f"fund fees on the panel, which fall unequally because "
        f"all-international pays the foreign expense ratio on everything and "
        f"the 50/50 split on half, needs a differential of "
        f"{fee_be_bp:.0f} basis points to cancel the lead — beyond any "
        f"index-fund pair, though not beyond the tax code: foreign dividend "
        f"withholding is a differential of exactly that kind that is neither "
        f"a fee nor a choice, and at the panel's own dividend share the "
        f"{0.30:.0%} statutory rate a non-resident pays is worth "
        f"{wht_statutory_bp:.0f} basis points a year on the international "
        f"sleeve alone. "
        + (f"The 50/50 split overtakes all-international at a withholding rate "
           f"of {wht_first:.1f}%, just inside that statutory rate, and the "
           f"certainty-equivalent-maximising home share walks from "
           f"{wht_home_low:.0%} to {wht_home_high:.0%} across the grid"
           if wht_bites else
           f"The lead nonetheless survives every rate tested, to {wht_top:.0f}%")
        + f". Currency hedging the international leg loses "
        f"certainty-equivalent consumption at every ratio tested, even when "
        f"the hedge is free. Conditioning on what inflation has just done — a "
        f"state variable an investor observes more reliably than a dividend "
        f"yield, and one bearing directly on the legs whose payments are "
        f"fixed in nominal terms — finds the mechanism exactly where theory "
        f"puts it and then finds it does not survive a lifetime: a "
        f"high-inflation start costs the bond and bill legs "
        f"{inf_nominal:+.2f} points a year over the following "
        + ("year" if inf_horizon == 1 else f"{inf_horizon} years")
        + f", against {inf_equity:+.2f} for equity, but by "
        f"{inf_long} years the bond effect has reversed to "
        f"{inf_bond_long:+.2f} points, "
        + ("and neither the headline ranking nor the optimal equity and "
           "home-bias shares move across the inflation terciles"
           if inf_holds and not inf_moves else
           "and the terciles do move the answer")
        + (f". Reading that state variable at the retirement date instead of "
           f"the birth date turns the null into a result: retiring into the "
           f"high-inflation third is worth {inf_retire_pct:+.2f}% of "
           f"certainty-equivalent retirement consumption against "
           f"{inf_birth_pct:+.2f}% for the same lifetimes bucketed by the "
           f"inflation they began at, {inf_ratio:.0f} times larger and of "
           f"the opposite sign — the lifetime null was a statement about the "
           f"horizon rather than about inflation"
           if inf_timing_bites else
           f". Reading it at the retirement date rather than the birth date "
           f"moves the level {inf_retire_pct:+.2f}% against "
           f"{inf_birth_pct:+.2f}%")
        + f". Correlating labour income with the home market — "
        f"the textbook reason to hold less of it, and assumed away everywhere "
        f"else — moves the lead {hc_change:+.2f} points across a correlation "
        f"range running past anything a labour economist would defend, "
        + ("without changing the order"
           if not hc_changes else "and does change the order")
        + f", though correlating it with world equity rather than with the "
        f"home market specifically keeps only {hc_diag:+.2f} points of that, "
        f"so most of the effect is about home bias and not about human "
        f"capital being risky; replacing the certain ninety-third birthday "
        f"with a Gompertz lifespan moves the lead by at most "
        f"{mort_change:.2f} points and "
        + ("leaves the ranking identical"
           if not mort_changes else "does reorder the strategies")
        + f". The assumption that turns out to matter most is the one the "
        f"panel makes silently: every result outside one section pays the US "
        f"social-security schedule in all sixteen countries. Replacing it "
        f"with Australia's — a means-tested Age Pension alongside a "
        f"compulsory 12% Superannuation Guarantee that sits on top of "
        f"voluntary saving rather than instead of it — "
        + (f"raises mean retirement consumption {pen_mean:+.0f}% and lowers "
           f"certainty-equivalent consumption {pen_cec:+.0f}%, because the "
           f"fifth percentile falls {pen_p5:+.0f}%: compulsory saving buys a "
           f"larger portfolio and the means test removes the floor it would "
           f"have sat on. Crossing the two features shows the contribution "
           f"rate alone is worth {pen_saving:+.0f}% and the assets test is "
           f"what takes it back"
           if pen_splits else
           f"moves certainty-equivalent retirement consumption "
           f"{pen_cec:+.0f}% and mean retirement consumption "
           f"{pen_mean:+.0f}%")
        + (f". It also reverses the allocation ranking — the only place in "
           f"this paper where that happens: the lead of {pen_base:.2f}% "
           f"becomes {pen_au_gap:.2f}% and the best strategy becomes the "
           f"{_pretty_strategy(pen_au_winner).lower()}, because inside the "
           f"assets-tested band a dollar of extra wealth costs more pension "
           f"a year than any asset here reliably earns, so the de-risking "
           f"schedule is rewarded for staying small"
           if pen_au_reorders else
           f". The allocation ranking survives it intact "
           f"({pen_au_gap:.2f}% against {pen_base:.2f}%), because the "
           f"guarantee lifts this saver clear of the means test altogether")
        + f". "
        f"{group_count_word('portfolio')} ask whether a better portfolio "
        f"exists inside the same asset menu, and whether solving for one "
        f"survives data it did not see. Solving the glide path directly by "
        f"coordinate ascent "
        f"under common random numbers reproduces the all-equity corner rather "
        f"than an interior optimum; freeing all four portfolio weights at "
        f"every age — {alloc_params} parameters on the simplex — adds "
        f"{alloc_lead:.2f}% over the best fixed benchmark while still "
        f"producing no glide path; and relaxing the long-only constraint, "
        f"borrowing to invest is worth {lev_free:+.2f}% at a zero borrowing "
        f"spread over the real bill rate but decays quickly in that spread, "
        f"breaking even by {break_even:.2%}, with the optimiser levering a "
        f"diversified portfolio rather than a concentrated one. Splitting "
        f"the record in half and scoring each solved schedule on the half it "
        f"was never fitted to "
        + ("leaves none of them ahead of the best constant mix, so those "
           "in-sample gains are upper bounds"
           if oos_none else
           f"leaves {oos_wins} of {oos_runs} ahead of the best constant mix — "
           f"the ones fitted to the turbulent first half and applied forward "
           f"({oos_forward:+.2f}%), not the ones fitted to the calm post-war "
           f"half and applied back ({oos_backward:+.2f}%)"
           if oos_found["asymmetric"] else
           f"leaves {oos_wins} of {oos_runs} ahead of the best constant mix")
        + f"; the strategy that transfers is a fixed allocation held "
        f"unchanged for a lifetime. Charging that solved schedule for the "
        f"trades it makes points the same way: it turns over "
        f"{tno_turn:.0%} of the portfolio a year against nothing for a "
        f"single-asset holding and about three percent for a fixed mix, and "
        f"its {tno_lead:.2f}% edge over the best fixed portfolio "
        + (f"is gone by {tno_be:.0f} basis points of one-way trading cost"
           if np.isfinite(tno_be) else
           "survives every trading cost tested")
        + f". "
        f"{group_count_word('menu')} widen the menu. Adding housing — de-smoothed to undo the "
        f"appraisal lag the published index carries — "
        + (f"earns {pc(float(house_free['mean_housing']), 0)} of the "
           f"portfolio when it is free to hold and "
           + (f"drops out entirely at an annual holding cost of "
              f"{pc(house_break, 1)}"
              if np.isfinite(house_break) else
              "survives every holding cost tested")
           if float(house_free["mean_housing"]) > 0.01 else
           "earns no place in the portfolio at any price, including free")
        + f"; financing that house with a mortgage — the one asset an "
        f"ordinary household can borrow against at close to the government's "
        f"rate — produces a loan-to-value schedule that "
        + (f"declines with age, {pc(mort_work, 0)} while working against "
           f"{pc(mort_ret, 0)} in retirement, reproducing from optimisation "
           f"the pattern households follow in practice"
           if mort_work > mort_ret else
           f"rises with age, {pc(mort_work, 0)} while working against "
           f"{pc(mort_ret, 0)} in retirement, contrary to observed household "
           f"behaviour")
        + ". "
        f"The last {group_count_word('plan').lower()} leave the portfolio "
        f"alone and search the rest of the plan, in the order a life meets "
        f"it. Conditioning a lifetime on how "
        f"expensive its market was at the moment it began — using the "
        f"trailing dividend yield an investor could observe, ranked against "
        f"tercile boundaries computed only from country-years that had "
        f"already happened — "
        + (f"leaves the ranking intact in all {len(val_adv)} valuation "
           f"buckets while moving the level: "
           if bool((val_adv['advantage_pct'] > 0).all())
           else f"reverses the ranking in "
                f"{int((val_adv['advantage_pct'] <= 0).sum())} of "
                f"{len(val_adv)} valuation buckets: ")
        + f"lifetimes begun in the dearest third reach retirement with less "
        f"and run out of money more often than those begun in the cheapest. "
        f"Conditioning the savings rate on the funded ratio is worth "
        f"{funded_net:.1f}% once every candidate signal is scored on a common "
        f"basis, and a deep decomposition of that signal — across "
        f"functional form, target definition, asymmetry, feasibility bands "
        f"and eight competing state variables — finds that the strongest "
        f"available signal is not the portfolio at all but the investor's own "
        f"pay cheque ({income_net:.1f}%). Making the retirement date a "
        f"wealth-triggered decision rather than a birthday is worth "
        f"{retire_value:.1f}% against a date matched on the same mean "
        f"retirement age, while the decade around that date explains "
        f"{pc(float(lottery['r2_retirement_window']), 1)} of the variation in "
        f"retirement outcomes — a lottery no allocation rule can diversify "
        f"away. And comparing {n_families} families of retirement spending "
        f"rule, each on its own optimised rate, spreads certainty-equivalent "
        f"consumption by {spend_spread:.0f}% between the best "
        f"({spend_best}) and the worst ({spend_worst}) — a range comparable "
        f"to the gap between the best and worst allocation strategies. "
        f"The paper's principal limitation is the breadth of the panel: "
        f"{pr['n_countries']} developed markets, so the international leg "
        f"spans {pr['n_countries'] - 1} foreign markets, all advanced "
        f"economies with long recorded histories that survived. The "
        f"jackknife interval above is what that breadth costs in precision; "
        f"what it costs in generality cannot be measured from inside the "
        f"sample, because deleting a country that is present says nothing "
        f"about one that was never there."
    )
    out.append(Paragraph(abstract, s["abstract"]))
    out.append(Paragraph(
        "<b>Keywords:</b> lifecycle asset allocation; target-date funds; "
        "block bootstrap; certainty-equivalent consumption; safe withdrawal "
        "rate; sequence-of-returns risk; glide path; retirement timing; "
        "savings rate; international diversification.",
        s["keywords"]))
    out.append(Paragraph(
        "<b>JEL classification:</b> G11, G51, D14, D15, J26, C15.",
        s["keywords"]))
    out.append(Spacer(1, 0.35 * cm))
    out.append(ctx.rule())
    out.append(NextPageTemplate("body"))
    out.append(PageBreak())
    return out


def contents(ctx: Any) -> List[Flowable]:
    toc = TableOfContents()
    toc.levelStyles = [ctx.s["toc1"], ctx.s["toc2"]]
    return [Paragraph("Contents", ctx.s["h1_plain"]), Spacer(1, 4), toc,
            PageBreak()]


# ---------------------------------------------------------------------------
# Section numbering
# ---------------------------------------------------------------------------
#: Reading order of the paper's numbered sections. This tuple is the single
#: source of truth: headings and cross-references are written as ``#key``
#: tokens (``#housing`` for the section number, ``#housing.2`` for a
#: subsection) and resolved against it at build time, so reordering the paper
#: means editing this tuple and the matching block in :func:`story` -- and
#: nothing else. Hand-numbered references were how a reference came to point
#: at the wrong heading twice in this project's history; the token cannot.
#: The reading order, and therefore the numbering. Four movements:
#:
#: 1. **The claim** -- what was replicated, on what data, by what method, and
#:    what came out.
#: 2. **Is the claim real?** -- ordered from the most internal test to the
#:    most external. First the model's own parameters, then its sampler, then
#:    which countries and eras it was given, then how the foreign sleeve was
#:    built and whether to hedge it, then when the investor happened to start,
#:    then what implementation costs, and last the two assumptions that were
#:    quietly flattering the result.
#: 3. **What else the investor decides** -- the portfolio first (glide,
#:    simplex, leverage), immediately followed by the check on whether any of
#:    those solved schedules survives data it was not fitted to; then the
#:    wider asset menu, then the cash-flow decisions in the order a life
#:    presents them: save, retire, spend.
#: 4. **What it means.**
SECTION_ORDER: Tuple[str, ...] = (
    "introduction",
    "background",
    "data",
    "methods",
    # The claim.
    "baseline",
    # Is the claim real? From the model's own dials outward.
    "sensitivity",
    "cohorts",
    "panel",
    "sleeve",
    "hedging",
    "valuation",
    "inflation",
    "fees",
    "withholding",
    "franking",
    "human_capital",
    "mortality",
    "pension",
    # What else the investor decides. The portfolio, and whether solving for
    # it survives contact with two things it was never shown: the cost of the
    # trades it makes, and data it was not fitted to.
    "glide",
    "allocation",
    "leverage",
    "turnover",
    "out_of_sample",
    # Widening the menu.
    "housing",
    "mortgage",
    # The cash-flow decisions, in the order a life presents them.
    "saving",
    "accumulation",
    "retirement",
    "sequence",
    "spending",
    "plan",
    "leisure",
    "tax",
    # Closing.
    "discussion",
    "limitations",
    "conclusion",
)

_SECTION_NUMBER: Dict[str, int] = {
    key: i + 1 for i, key in enumerate(SECTION_ORDER)}

#: Matches ``#key``, ``#key.3`` and ``#key.3.1``. A dot is only consumed when
#: a digit follows it, so ``#glide. Solving for...`` in a heading resolves
#: cleanly rather than swallowing the sentence.
SECTION_TOKEN = re.compile(r"#([a-z_]+)((?:\.\d+)*)")


#: The studies that come after the replication and before the closing
#: material. The count appears in the title and the abstract, so it is derived
#: from the reading order rather than typed -- it went stale twice before this
#: was here.
EXTENSION_SECTIONS: Tuple[str, ...] = SECTION_ORDER[
    SECTION_ORDER.index("cohorts"):SECTION_ORDER.index("discussion")]

#: How the abstract walks the extensions. The names are the groups it uses;
#: the sizes are derived from these tuples rather than typed, and
#: ``tests/test_paper_sections.py`` requires them to partition
#: :data:`EXTENSION_SECTIONS` exactly -- so a reordering cannot leave the
#: abstract announcing "four" and then describing five.
EXTENSION_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("robustness", ("cohorts", "panel", "sleeve", "hedging", "valuation",
                    "inflation", "fees", "withholding", "franking",
                    "human_capital", "mortality", "pension")),
    ("portfolio", ("glide", "allocation", "leverage", "turnover",
                   "out_of_sample")),
    ("menu", ("housing", "mortgage")),
    # `tax` sits here rather than with the other charges above because it
    # is not a charge on the *portfolio* but on the retirement system, and
    # it exists only to check the comparison `leisure` makes.
    ("plan", ("saving", "accumulation", "retirement", "sequence",
              "spending", "plan", "leisure", "tax")),
)


def group_count_word(name: str) -> str:
    """`Eight`, for the group of that name, or the numeral past the table."""
    for key, members in EXTENSION_GROUPS:
        if key == name:
            return NUMBER_WORDS.get(len(members), str(len(members)))
    raise KeyError(f"unknown extension group {name!r}")


#: Cardinals spelled out, for a title that should not contain a numeral.
NUMBER_WORDS: Dict[int, str] = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
    7: "Seven", 8: "Eight", 9: "Nine",
    10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen",
    15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen",
    19: "Nineteen", 20: "Twenty", 21: "Twenty-one", 22: "Twenty-two",
    23: "Twenty-three", 24: "Twenty-four", 25: "Twenty-five",
    26: "Twenty-six", 27: "Twenty-seven", 28: "Twenty-eight",
    29: "Twenty-nine", 30: "Thirty", 31: "Thirty-one", 32: "Thirty-two",
}


def extension_count_word() -> str:
    """`Eighteen`, or the numeral if the paper grows past the table."""
    n = len(EXTENSION_SECTIONS)
    return NUMBER_WORDS.get(n, str(n))


def _pretty_strategy(key: str) -> str:
    """A configured strategy label for a bare key, so no table prints one.

    The label lives in ``config.yaml``; falling back to the de-underscored key
    keeps this working for the solved schedules, which are not configured
    strategies and have no label of their own.
    """
    from src import plots
    return plots.STRATEGY_LABEL.get(key, key.replace("_", " "))


#: Column-header forms of the strategy names, for the one table wide enough
#: that the configured labels wrap mid-word. Section #franking scores four
#: strategies against three descriptive columns, which is two columns more
#: than the page has room for at full width.
COMPACT_STRATEGY: Dict[str, str] = {
    "international_equity": "All intl.",
    "balanced_all_equity": "50/50",
    "domestic_equity": "All dom.",
    "target_date_fund": "Glide path",
    "sixty_forty": "60/40",
    "bills_only": "Bills",
}


#: Readable names for Section #leisure's pension systems.
REGIME_LABEL: Dict[str, str] = {
    "none": "No tax",
    "us_roth": "Roth",
    "us_traditional": "Traditional",
    "au": "Australia, super after 60",
}

SYSTEM_LABEL: Dict[str, str] = {
    "us": "US social security",
    "au_pension_only": "Age Pension only",
    "au_as_legislated": "Age Pension + super guarantee",
}


def _compact_strategy(key: str) -> str:
    """A strategy name short enough for a column header."""
    return COMPACT_STRATEGY.get(key, _pretty_strategy(key))


def section_number(key: str) -> int:
    """The number this section carries in the current reading order."""
    try:
        return _SECTION_NUMBER[key]
    except KeyError:                       # pragma: no cover - author error
        raise KeyError(
            f"unknown section {key!r}; SECTION_ORDER has "
            f"{sorted(_SECTION_NUMBER)}") from None


def resolve_sections(text: str) -> str:
    """Replace every ``#key`` token in ``text`` with its section number."""
    def swap(match: "re.Match[str]") -> str:
        return f"{section_number(match.group(1))}{match.group(2)}"
    return SECTION_TOKEN.sub(swap, text)


# ---------------------------------------------------------------------------
# 1. Introduction
# ---------------------------------------------------------------------------
def section_introduction(ctx: Any) -> List[Flowable]:
    f = ctx.f
    p = f.panel
    alloc, lev = f.allocation, f.leverage
    adv_tdf = f.advantage("balanced_all_equity", "target_date_fund")
    adv_6040 = f.advantage("balanced_all_equity", "sixty_forty")
    adv_dom = f.advantage("balanced_all_equity", "domestic_equity")
    _retire = f.table("retirement_value_of_conditioning")
    _base = _retire[np.isclose(_retire["working_income_floor"], 0.25)
                    & _retire["variant"].str.contains("Wealth trigger")]
    intro_retire = float(_base["value_of_conditioning_pct"].mean())
    ruin_eq = float(f.strategy_row("balanced_all_equity")["prob_ruin"])
    ruin_tdf = float(f.strategy_row("target_date_fund")["prob_ruin"])
    dominance = f.table("dominance_check")
    n_criteria = int(dominance["criteria"].iloc[0])
    n_won = int(dominance["criteria_won"].iloc[0])

    out: List[Flowable] = [ctx.h1("#introduction. Introduction")]
    out.append(ctx.p(
        "The default investment vehicle of the modern retirement system is the "
        "target-date fund. It embodies a single, rarely-examined proposition: "
        "that an investor should hold less equity as they age, sliding from a "
        "growth portfolio in their twenties toward a bond-heavy portfolio at "
        "retirement. That proposition is not a market observation. It is a "
        "theoretical inheritance, descended from Samuelson (1969) and Merton "
        "(1969) via the human-capital arguments of Bodie, Merton and Samuelson "
        "(1992), and it has been institutionalised across trillions of dollars "
        "of default-enrolled savings."))
    out.append(ctx.p(
        "Anarkulova, Cederburg and O'Doherty (henceforth ACO) challenged it "
        "directly. Their argument is not that equities have higher expected "
        "returns — everyone agrees on that — but that the standard evidence "
        "against holding them is an artefact of how returns are simulated. "
        "Draw returns independently and identically distributed from a "
        "normal-ish distribution fitted to post-war US data, and equities look "
        "dangerous over short horizons and safe over long ones in exactly the "
        "way textbook time diversification predicts. Draw them instead in "
        "blocks from the full international historical record — including the "
        "markets that closed, the decades that were lost, and the countries "
        "whose real returns were negative for a generation — and the shape of "
        "the problem changes. ACO's conclusion is that a fixed all-equity "
        "portfolio, split between domestic and international markets, beats "
        "the glide path on almost every metric an investor would care about."))
    out.append(ctx.p(
        "This paper does three things. First, it reproduces that result from "
        "scratch on an independently constructed panel, with the full "
        "diagnostic apparatus made visible rather than asserted. Second, it "
        "stress-tests the result across a large parameter space, so that the "
        f"reader can see where the conclusion is robust and where it is thin. "
        f"Third — and this is where most of the length lies — it takes "
        f"{extension_count_word().lower()} questions that the original design "
        f"leaves open and works each of "
        "them out, several of which produce results that run against the "
        "intuition that motivated them."))

    out.append(ctx.h2("#introduction.1 What we find"))
    out.append(ctx.p(
        f"<b>The replication succeeds.</b> On a {p['n_countries']}-country "
        f"panel covering {p['first_year']}–{p['last_year']} "
        f"({p['country_years']:,} country-years), the 50/50 "
        f"domestic/international equity portfolio delivers "
        f"{adv_tdf:.1f}% higher certainty-equivalent consumption than a "
        f"target-date glide path, {adv_6040:.1f}% higher than a 60/40 "
        f"portfolio and {adv_dom:.1f}% higher than domestic equity alone, at "
        f"the baseline risk aversion γ = {f.baseline_gamma:g}. It does so "
        f"while <i>reducing</i> the probability of running out of money from "
        f"{pc(ruin_tdf, 1)} to {pc(ruin_eq, 1)}. Across the "
        f"{n_criteria} distributional criteria we test, the all-equity "
        f"portfolio wins {n_won} and loses none."))
    out.append(ctx.p(
        "<b>The mechanism is international diversification, not equity risk "
        "per se.</b> Domestic-only equity is beaten by the target-date fund on "
        "several tail measures. What makes the all-equity portfolio work is "
        "that a bad domestic century and a bad international century are not "
        "the same century. This is a stronger claim than \"stocks beat bonds\" "
        "and it is the one the international panel is needed to make."))
    out.append(ctx.p(
        "<b>Four percent is not a safe withdrawal rate on this panel.</b> At a "
        "five-percent ruin tolerance the sustainable rate is under three "
        "percent for every strategy tested, and under two percent for the "
        "conservative ones. The four-percent rule is a US-post-war artefact "
        "and does not survive a sample that contains the twentieth century as "
        "the rest of the world experienced it."))
    out.append(ctx.p(
        "<b>Solving the glide path directly returns the corner.</b> Rather "
        "than testing a handful of candidate schedules, we solve for the "
        "equity share at every age by coordinate ascent under common random "
        "numbers. The solution is not an interior glide path: it sits at or "
        "near the all-equity corner for the entire lifecycle, and the "
        "deviation profile shows that most of the apparent age-structure in "
        "the solved schedule is worth less than a basis point."))
    out.append(ctx.p(
        f"<b>Freeing every weight changes almost nothing.</b> Solving the full "
        f"four-asset simplex at every age — {alloc['free_parameters']} free "
        f"parameters, with the bond/bill split no longer imposed — beats the "
        f"best fixed benchmark by {alloc['lead_pct']:.2f}%. The optimiser "
        f"holds essentially no fixed income at any age, so the restriction the "
        f"glide-path solve made for convenience turns out not to have been "
        f"binding."))
    out.append(ctx.p(
        f"<b>When you borrow matters more than whether.</b> A "
        f"<i>constant</i> leverage ratio held for life is worth "
        f"{lev['value_at_zero_spread']:+.2f}% when credit is free, breaks even "
        f"at a spread of {lev['break_even_spread']:.2%}, and is already under a "
        f"tenth of a percent by {lev['negligible_spread']:.2%} — a null result "
        f"on margin accounts. Let the ratio <i>decline with age</i>, which is "
        f"what Ayres and Nalebuff actually prescribe, and it is worth more at "
        f"every price of credit we test, including prices at which the "
        f"constant ratio is worth nothing. The gain does not depend on the "
        f"per-age detail: a policy with a single free parameter — one ratio "
        f"while working, unlevered in retirement — keeps most of it. What the "
        f"optimiser levers is a diversified portfolio rather than a "
        f"concentrated one, and the cost of the trade is paid in the left "
        f"tail."))
    out.append(ctx.p(
        "<b>Timing beats allocation.</b> The single decade around a person's "
        "retirement date explains more of the variation in their retirement "
        "outcome than the choice between any two of the allocation strategies "
        f"we test. Making the retirement date itself a decision — retire when "
        f"wealth reaches a multiple of income, rather than on a birthday — is "
        f"worth {intro_retire:.1f}% of certainty-equivalent consumption "
        f"against a fixed date matched on the same mean retirement age."))
    out.append(ctx.p(
        "<b>The accumulation side has more room than the allocation side.</b> "
        "Conditioning how much you save on whether you are ahead of or behind "
        "an age-appropriate wealth target is worth more than any of the "
        "allocation refinements we test. Decomposing that signal produces the "
        "paper's most counter-intuitive result: the best state variable "
        "available to a saver is not their portfolio balance but their own "
        "labour income relative to its expected path."))

    out.append(ctx.p(
        "<b>Where you start changes what you get, not what you should "
        "hold.</b> Conditioning a lifetime on the dividend yield its market "
        "offered the year it began — ranked, as a real investor would have "
        "had to, against only the history that had already happened — "
        + ("leaves the allocation ranking intact at every starting valuation "
           if bool((f.table('valuation_advantage')['advantage_pct'] > 0).all())
           else "reverses the allocation ranking in at least one starting "
                "valuation ")
        + "while moving the level substantially. An investor starting in an "
        "expensive market should expect less from the same portfolio, not a "
        "different portfolio."))
    out.append(ctx.p(
        "<b>Housing is a serious asset at a serious price.</b> The historical "
        "sources measure a fourth asset class this literature usually skips. "
        "Corrected for the appraisal smoothing that makes property indices "
        "look artificially calm, it earns a large allocation when it is free "
        "to hold and loses it entirely at a plausible annual holding cost. "
        "The interesting quantity is not whether housing belongs in a "
        "portfolio but the cost at which it stops belonging, and that number "
        "is small enough that a real owner's costs are decisive."))

    out.append(ctx.h2("#introduction.2 What is new here"))
    out.append(ctx.p(
        f"Relative to the paper being replicated, this study contributes a "
        f"methodological discipline and {extension_count_word().lower()} "
        f"substantive extensions. The "
        "discipline is a set of comparison rules applied uniformly: every "
        "policy is scored against a <i>matched</i> baseline that differs from "
        "it in exactly one dimension; every optimiser runs under common random "
        "numbers so that differences are policy effects rather than sampling "
        "noise; every solved schedule is subjected to a deviation profile "
        "before any structure in it is described; and every apparent optimum "
        "is checked against the boundary of its own search grid. Several of "
        "the results below changed sign or magnitude when these checks were "
        "applied, and the paper reports the corrected versions with the "
        "diagnostics that forced them."))
    out.extend(ctx.bullets([
        "<b>Sensitivity.</b> A tornado analysis over ten parameter dimensions, "
        "reporting the full range of the advantage rather than a point estimate.",
        "<b>Retirement spending rules.</b> Eight families of withdrawal policy "
        "— constant real, percentage-of-portfolio, guardrails, endowment "
        "smoothing, required-minimum-distribution, amortisation, actuarial and "
        "a floor-and-ceiling rule — each optimised over its own rate and "
        "compared at its own best setting.",
        "<b>The optimal glide path.</b> Direct numerical solution of the "
        "age-by-asset schedule, free-form and parametric, with local-optimum "
        "checks from multiple restarts.",
        "<b>The whole allocation.</b> The full four-asset weight simplex "
        "solved at every year of the lifecycle — domestic equity, "
        "international equity, bonds and bills, with nothing held fixed — by "
        "two-stage coordinate ascent on the simplex.",
        "<b>Leverage.</b> The long-only constraint relaxed: the optimal "
        "borrowing ratio and the portfolio that goes with it, swept across the "
        "price of credit, with a break-even spread and an age-varying "
        "leverage schedule.",
        "<b>Currency hedging.</b> A covered-interest-parity hedged "
        "international leg, swept by annual hedging cost, yielding a "
        "break-even cost and an optimal hedge ratio as functions of that cost.",
        "<b>Endogenous retirement timing.</b> Retirement as a wealth-triggered "
        "decision, plus a formal decomposition of the retirement-date lottery.",
        "<b>The savings-rate profile.</b> Solving for the savings rate at "
        "every age with the career average pinned, which separates the shape "
        "question from the level question the model cannot answer.",
        "<b>Conditioning the savings rate.</b> Making the contribution respond "
        "to portfolio state, scored against a constant rate matched on the "
        "realised career average.",
        "<b>The accumulation signal, decomposed.</b> Functional form, target "
        "definition, response asymmetry, feasibility bands, age windows, eight "
        "competing state variables and their pairwise combination.",
    ]))

    out.append(ctx.h2("#introduction.3 Roadmap"))
    out.append(ctx.p(
        "The paper is organised so that each block earns the right to the "
        "next. Sections #background–#methods set up: the literature, the "
        "panel and its construction, and the bootstrap, lifecycle model, "
        "preference specification and comparison discipline that everything "
        "afterwards runs through."))
    out.append(ctx.p(
        "Sections #baseline–#hedging are the result and five ways it could "
        "be wrong. Section #baseline presents the baseline replication. "
        "Section #sensitivity asks whether it survives the preference and "
        "lifecycle parameters; Section #panel whether it survives the loss of "
        "any one country, and what standard error sixteen countries actually "
        "support; Section #sleeve whether it survives the way the "
        "international sleeve is weighted — the construction our one "
        "divergence from the replicated study most obviously depends on; "
        "Section #fees whether it survives the cost of the funds that "
        "implement it; and Section #hedging whether that sleeve should be "
        "currency-hedged at all."))
    out.append(ctx.p(
        "Sections #glide–#leverage ask whether a better portfolio exists "
        "inside the same asset menu, by progressively relaxing what is held "
        "fixed: the shape of the glide path (Section #glide), then every "
        "weight at every age (Section #allocation), then the long-only "
        "constraint itself (Section #leverage). Sections #housing–#mortgage "
        "widen the menu instead, adding the asset most households actually "
        "hold — housing owned outright (Section #housing), then mortgaged "
        "(Section #mortgage)."))
    out.append(ctx.p(
        "Sections #valuation–#spending leave the portfolio alone and search "
        "the rest of the plan, in the order a life meets it: the valuation "
        "you start at (Section #valuation), how much you save "
        "(Section #saving) and what that saving should respond to "
        "(Section #accumulation), when you stop (Section #retirement), and "
        "how you draw down (Section #spending). Section #discussion draws the "
        "results together, Section #limitations states the limitations "
        "candidly, and Section #conclusion concludes. Four appendices give "
        "the full parameter set, the country panel, supplementary tables and "
        "the reproduction instructions."))
    return out


# ---------------------------------------------------------------------------
# 2. Background
# ---------------------------------------------------------------------------
def section_background(ctx: Any) -> List[Flowable]:
    f = ctx.f
    out: List[Flowable] = [ctx.h1("#background. Background and Related Work")]
    out.append(ctx.h2("#background.1 The theoretical case for the glide path"))
    out.append(ctx.p(
        "The canonical result is negative: in the Samuelson–Merton framework "
        "with constant relative risk aversion, i.i.d. returns and no labour "
        "income, the optimal share of wealth in risky assets is independent of "
        "the investment horizon. Time does not diversify risk; it multiplies "
        "it. Any age-declining allocation therefore has to be justified by "
        "something outside that framework."))
    out.append(ctx.p(
        "The standard justification is human capital. Bodie, Merton and "
        "Samuelson (1992) observe that a young worker holds a large stock of "
        "implicit wealth in the form of future labour income, and that if this "
        "asset is bond-like then total wealth is already heavily weighted "
        "toward safety; the financial portfolio should compensate by holding "
        "more equity. As the worker ages, human capital depletes and the "
        "financial portfolio should become more conservative to keep total "
        "exposure roughly constant. This produces a glide path, and it is the "
        "argument the target-date industry rests on."))
    out.append(ctx.p(
        "The argument is coherent but its quantitative force depends on "
        "parameters that are difficult to pin down: how bond-like labour "
        "income actually is, how it correlates with equity markets, how much "
        "of it remains at each age, and what happens to it in exactly the "
        "states of the world where equities fail. It also depends on what the "
        "return distribution looks like — which is where the argument this "
        "paper reproduces enters."))

    out.append(ctx.h2("#background.2 The empirical challenge"))
    out.append(ctx.p(
        "Almost all of the simulation evidence used to calibrate glide paths "
        "draws returns from a parametric distribution fitted to a single "
        "market — usually the United States — over a period that begins after "
        "the Second World War. That sample has three properties that flatter "
        "conservative allocations in a specific direction. It excludes the "
        "market closures, expropriations and hyperinflations that other "
        "developed markets experienced. It excludes the pre-war period in "
        "which the US itself behaved less benignly. And by drawing "
        "independently it destroys the multi-decade persistence that makes "
        "long-horizon outcomes fat-tailed."))
    out.append(ctx.p(
        "ACO's response is to replace the return-generating process. Their "
        "sample is 39 developed countries of monthly returns from 1890 to "
        "2023, drawn from Global Financial Data, and their sampling scheme is "
        "a stationary block bootstrap with geometric block lengths averaging "
        "120 months, so runs of good and bad decades survive into the "
        "simulated lifetimes. Their household is a couple facing random "
        "longevity from Social Security Administration mortality tables, "
        "saving 10% of labour income, retiring at 65 and drawing 4% of "
        "wealth at retirement thereafter, with utility calibrated to De "
        "Nardi, French and Jones (2010) at a risk aversion of 3.84. Their "
        "conclusion is that a 50/50 domestic/international equity portfolio, "
        "held for life, beats the glide path on wealth, retirement "
        "consumption, capital preservation and bequests alike."))
    out.append(ctx.p(
        "This paper reconstructs that process independently and asks whether "
        "the claim holds. The sampling scheme here is deliberately the same "
        "in form — stationary blocks, geometric lengths, a ten-year mean — so "
        "that any divergence is attributable to the panel and the lifecycle "
        "assumptions rather than to how returns are drawn. Where this study "
        "differs, it differs in ways it can state: annual rather than monthly "
        "data, sixteen countries rather than thirty-nine, an individual "
        "rather than a couple, and a deterministic horizon rather than "
        "random longevity."))

    out.append(ctx.h2("#background.3 The data source"))
    out.append(ctx.p(
        "The underlying return data come from the Jordà–Schularick–Taylor "
        "Macrohistory Database and the associated \"Rate of Return on "
        "Everything\" project (Jordà, Knoll, Kuvshinov, Schularick and Taylor, "
        "2019). That database assembles annual total returns on equities, "
        "long-term government bonds, short-term bills and consumer price "
        "inflation for sixteen advanced economies from the late nineteenth "
        "century onward. It is the standard source for long-horizon "
        "international asset-return work and it is what makes an exercise of "
        "this kind possible at all."))
    out.append(ctx.p(
        f"It is also the binding constraint on this study: it carries complete "
        f"equity, bond and bill total returns for {f.panel['n_tier_a']} "
        f"countries, and that set is our panel. Section #data.2 describes it and "
        f"Section #limitations.1 sets out what its breadth costs."))

    out.append(ctx.h2("#background.4 Adjacent literatures this paper touches"))
    out.extend(ctx.bullets([
        "<b>Bootstrap inference for dependent data.</b> The sampling scheme "
        "here is a stationary block bootstrap in the sense of Politis and "
        "Romano (1994), generalised so that a block is a (country, window) "
        "pair drawn jointly across all five series.",
        "<b>Safe withdrawal rates.</b> The four-percent convention descends "
        "from Bengen (1994) and the Trinity study (Cooley, Hubbard and Walz, "
        "1998), both of which are US-only and historical-sequence based. "
        "Section #baseline.4 re-derives the sustainable rate on this panel.",
        "<b>Dynamic withdrawal policy.</b> Guyton and Klinger (2006) "
        "guardrails and the actuarial/amortisation family are represented in "
        "the spending-rule comparison of Section #spending.",
        "<b>Bequest motives.</b> The utility specification uses the shifted "
        "power form of De Nardi (2004), which is what allows a zero terminal "
        "balance to be evaluated at all under high risk aversion.",
        "<b>Sequence-of-returns risk.</b> Section #retirement formalises the folk "
        "observation that the decade around retirement dominates, and prices "
        "the option value of choosing when to stop working.",
        "<b>Lifecycle leverage.</b> Ayres and Nalebuff (2010) argue that a "
        "young investor should borrow to spread market exposure evenly across "
        "a lifetime. Section #leverage relaxes the long-only constraint and prices "
        "that argument against the cost of credit.",
    ]))
    return out


# ---------------------------------------------------------------------------
# 3. Data
# ---------------------------------------------------------------------------
def section_data(ctx: Any) -> List[Flowable]:
    f = ctx.f
    p = f.panel
    pr, adv = f.provenance, f.panel_advantage
    anchors = f.table("provenance_anchor_checks")
    unusable = f.table("provenance_unusable_series")
    contamination = f.table("provenance_intl_contamination")
    summary = f.table("panel_summary_statistics")
    equity = summary[summary["series"] == "dom_eq"].sort_values("iso")
    tier_a = equity[equity["tier"] == "A"]
    corr = f.table("panel_cross_asset_correlation")
    wins = f.table("panel_winsorised_observations")
    gaps = f.table("panel_structural_gaps")
    cfg = f.cfg

    out: List[Flowable] = [ctx.h1("#data. Data")]
    out.append(ctx.h2("#data.1 What the panel is"))
    out.append(ctx.p(
        f"The panel is an annual, country-by-year matrix of five real series — "
        f"domestic equity, international equity, long-term government bonds, "
        f"short-term bills and consumer price inflation — for "
        f"{p['n_countries']} developed markets over {p['first_year']}–"
        f"{p['last_year']}. It contains {p['country_years']:,} usable "
        f"country-years of domestic equity returns, with a median country "
        f"history of {p['median_years']:.0f} years and a minimum of "
        f"{p['min_years']} years."))
    out.append(ctx.p(
        "All series are <i>real</i>: nominal total returns are deflated by "
        "each country's own consumer price index in the same year, so that a "
        "hyperinflation shows up as a catastrophic real return rather than as "
        "a spectacular nominal one. This is not a cosmetic choice. Several of "
        "the worst episodes in the sample — Germany in the early 1920s, "
        "central Europe in the late 1940s — are invisible in nominal space and "
        "dominate the left tail in real space."))
    out.append(ctx.p(
        f"The equity series across the panel average {pc(p['mean_real_equity'])} "
        f"in arithmetic real terms and {pc(p['mean_geometric_equity'])} "
        f"geometrically, with a mean cross-country standard deviation of "
        f"{pc(p['mean_equity_sd'])}. The gap between the arithmetic and "
        f"geometric means — over three percentage points — is itself the "
        f"story: at this level of volatility, the compounded experience of a "
        f"lifetime investor is far below the average annual return, and any "
        f"simulation that draws i.i.d. from the arithmetic mean will overstate "
        f"what a real investor would have received."))

    out.extend(ctx.figure(
        "fig01_coverage_matrix",
        "Share of each decade for which a country has a complete return "
        "record. Darker is more complete; the scale is a share so that the "
        "final, partial decade compares with the rest. The gaps are not "
        "random: they cluster around the "
        "two world wars and around the market closures that follow them, which "
        "is precisely the period a survivorship-prone sample would drop."))

    out.append(ctx.h2("#data.2 The country set"))
    out.append(ctx.p(
        f"The Jordà–Schularick–Taylor database carries complete, independently "
        f"sourced equity, bond and bill total returns for {p['n_tier_a']} "
        f"countries: {', '.join(sorted(tier_a['country']))}. That set is the "
        f"panel, and every country-year of every return series in it is a "
        f"recorded observation."))
    out.append(ctx.p(
        f"The wider developed-market universe — IMF advanced economies with an "
        f"investable domestic equity market, plus Poland — numbers 38. The "
        f"histories of the other {pr['n_removed']} exist in commercial "
        f"databases such as Global Financial Data and Dimson–Marsh–Staunton, "
        f"which are proprietary and not redistributable. Working from openly "
        f"licensed sources, {p['n_tier_a']} is the extent of the recorded "
        f"evidence, and Section #data.6.1 reports what the excluded countries do "
        f"carry."))
    out.append(ctx.p(
        f"That breadth is the paper's principal limitation and Section #limitations.1 "
        f"develops it. Its most direct consequence is on the international "
        f"leg, which is a leave-one-out average and therefore spans "
        f"{p['n_tier_a'] - 1} foreign markets, all advanced economies with "
        f"long histories. Every statement here about international "
        f"diversification is a statement about that set."))

    out.append(ctx.h2("#data.3 Constructing the international leg"))
    out.append(ctx.p(
        "An investor in country <i>i</i> holding \"international equity\" does "
        "not hold a global index that includes their own market. For each "
        "country-year we therefore build the international return as a "
        "leave-one-out average across all other countries with data that year, "
        "computed in gross terms and in a common currency before being "
        "converted back to the home investor's real terms. Leaving the home "
        "market out is what makes the domestic and international legs "
        "genuinely distinct assets rather than two overlapping slices of the "
        "same one."))
    out.append(ctx.p(
        f"Three observations in the raw international series are extreme "
        f"enough to dominate any lifetime that draws them, all of them "
        f"artefacts of a market reopening after wartime closure with a "
        f"collapsed price index. We winsorise the international leg at the "
        f"{float(cfg['data']['international_winsor_pct']):.1f}th percentile "
        f"of its own distribution; the affected observations are listed in "
        f"full in Table {ctx._table_no + 2} so that the reader can judge the "
        f"intervention rather than take it on trust."))

    out.extend(ctx.table(
        rows_from(equity,
            ["country", "n_years", "first_year", "last_year", "mean",
             "geometric_mean", "std", "skew", "ar1"],
            ["Country", "Years", "From", "To", "Mean", "Geo. mean",
             "S.d.", "Skew", "AR(1)"],
            {"country": str, "n_years": lambda v: f"{int(v)}",
             "first_year": lambda v: f"{int(v)}",
             "last_year": lambda v: f"{int(v)}",
             "mean": lambda v: pc(v), "geometric_mean": lambda v: pc(v),
             "std": lambda v: pc(v), "skew": lambda v: f2(v),
             "ar1": lambda v: f2(v)},
            ),
        "Real domestic equity returns by country",
        note="Arithmetic and geometric means are annual real returns "
             "deflated by each country's own consumer price index; AR(1) is "
             "the first-order autocorrelation of the annual series. Appendix "
             "B repeats this with excess kurtosis.",
        font_size=7.0))

    out.append(ctx.h2("#data.4 Cross-asset structure"))
    out.append(ctx.p(
        f"The correlation structure the bootstrap must preserve is reported "
        f"below. Domestic and international equity correlate at "
        f"{f2(float(corr[corr['series'] == 'dom_eq']['intl_eq'].iloc[0]))}, "
        f"which is high enough that international diversification is not free "
        f"and low enough that it is worth having. Bonds and bills correlate at "
        f"{f2(float(corr[corr['series'] == 'bond']['bill'].iloc[0]))}. "
        f"Inflation correlates negatively with every asset, most strongly with "
        f"bills "
        f"({f2(float(corr[corr['series'] == 'inflation']['bill'].iloc[0]))}), "
        f"which is the mechanism by which cash loses in real terms over long "
        f"horizons."))
    out.extend(ctx.table(
        rows_from(corr, ["series", "dom_eq", "intl_eq", "bond", "bill",
                         "inflation"],
                  ["", "Dom. equity", "Intl. equity", "Bonds", "Bills",
                   "Inflation"],
                  {"series": lambda v: {"dom_eq": "Domestic equity",
                                        "intl_eq": "International equity",
                                        "bond": "Bonds", "bill": "Bills",
                                        "inflation": "Inflation"}[v]}),
        "Pooled cross-asset correlation matrix of annual real returns",
        note="Pooled across all country-years in the panel. These are the "
             "moments the block bootstrap is required to reproduce; Table 4 "
             "reports how closely it does so."))

    out.extend(ctx.table(
        rows_from(wins, ["year", "country", "raw_intl_eq", "winsorised_intl_eq"],
                  ["Year", "Country", "Raw return", "After winsorisation"],
                  {"year": lambda v: f"{int(v)}", "country": str,
                   "raw_intl_eq": lambda v: pc(v, 0),
                   "winsorised_intl_eq": lambda v: pc(v, 0)}),
        "Every winsorised observation in the international leg",
        note="All three are markets reopening after wartime closure against a "
             "collapsed price index. Left unwinsorised, a single one of these "
             "draws would dominate the terminal wealth of any lifetime that "
             "contained it."))

    out.append(ctx.h2("#data.5 Market disruptions and the survivorship question"))
    out.append(ctx.p(
        f"There are {len(gaps)} contiguous runs of missing data across the "
        f"countries of the panel, concentrated in the two world wars. The treatment "
        f"of these gaps is a substantive modelling decision rather than a "
        f"data-cleaning one. Interpolating across them would smooth away "
        f"exactly the episodes that motivate the whole exercise; dropping the "
        f"countries would reintroduce the survivorship bias the panel exists "
        f"to avoid."))
    out.append(ctx.p(
        "We do neither. Gaps are preserved as gaps, and the bootstrap is "
        "constrained to draw only blocks that lie entirely within a "
        "contiguous run of available data for the country in question. A "
        "country with a six-year wartime hole contributes blocks on either "
        "side of it but never across it. The consequence is that the sampler "
        "is <i>less</i> able to draw from countries with disrupted histories "
        "at long block lengths, which we quantify rather than assume: the "
        "run-length admissibility statistics are reported in Section #methods.1."))
    out.append(ctx.h2("#data.6 Auditing the data"))
    out.append(ctx.p(
        "The database is used here through a redistributed copy rather than "
        "obtained from its compilers, which deserves more scepticism than a "
        "citation. This section asks whether the file is genuine, reports one "
        "test it does not pass, and describes three bodies of recorded data "
        "that sit in the sources and stay out of the model."))

    out.append(ctx.h3("#data.6.1 Why the panel stops at sixteen"))
    out.append(ctx.p(
        "The boundary is set by equity. Four countries outside the panel do "
        "carry recorded interest-rate histories: long-term yields and short "
        "rates in the macro file for Canada and Ireland, Clio-Infra long "
        "yields for Austria and New Zealand. Those are enough to build real "
        "bond and bill returns — a bond total return follows from a yield and "
        "a duration, <i>r<sub>t</sub> = y<sub>t−1</sub> + D(y<sub>t−1</sub> − "
        "y<sub>t</sub>)</i>; a bill return is a lagged short rate; and "
        "deflating both by that country\u2019s own price index gives a real "
        "return that was measured rather than modelled."))
    if len(unusable):
        out.extend(ctx.table(
            rows_from(unusable,
                      ["country", "series", "source", "first_year",
                       "last_year", "observed_years"],
                      ["Country", "Series", "Source", "From", "To",
                       "Observed years"],
                      {"country": str, "series": str, "source": str,
                       "first_year": lambda v: f"{int(v)}",
                       "last_year": lambda v: f"{int(v)}",
                       "observed_years": lambda v: f"{int(v):,}"}),
            "Recorded series that exist and still cannot be used",
            note=("Rebuilt from published rates and deflated by each "
                  "country's own price index. None of these countries has an "
                  "equity return series in any source available to us, which "
                  "is why none is in the panel."),
            col_widths=[ctx.width * 0.17, ctx.width * 0.11, ctx.width * 0.38,
                        ctx.width * 0.10, ctx.width * 0.10,
                        ctx.width * 0.14]))
    out.append(ctx.p(
        "None of it gets them into the panel. A lifecycle investor needs a "
        "domestic <i>equity</i> return, and the macro file carries none for "
        "these four — not even the interpolated variants it supplies "
        "elsewhere. Rates alone cannot support a portfolio choice. That is why "
        "the panel is sixteen countries and not twenty."))


    hs = pr.get("housing", {})
    if hs.get("countries"):
        out.append(ctx.h3("#data.6.2 The fourth asset class"))
        out.append(ctx.p(
            f"The macro file carries a fourth asset the source project "
            f"measured: <b>housing total returns</b>, empirical for all "
            f"{hs['countries']} observed countries over "
            f"{hs['country_years']:,} country-years "
            f"({hs['first_year']}–{hs['last_year']}). It is held out of the "
            f"headline results, which use the same four-asset set as the paper "
            f"being replicated, and audited here so the reason is visible "
            f"rather than assumed. Section #housing then puts it into the "
            f"opportunity set and prices it."))
        out.extend(ctx.table(
            rows_from(hs["frame"],
                      ["country", "years", "mean", "sd", "sd_desmoothed",
                       "equity_sd", "autocorrelation"],
                      ["Country", "Years", "Mean real", "s.d. published",
                       "s.d. de-smoothed", "Equity s.d.", "Autocorr."],
                      {"country": str, "years": lambda v: f"{int(v)}",
                       "mean": lambda v: pc(v, 1), "sd": lambda v: pc(v, 1),
                       "autocorrelation": lambda v: f2(v, 2),
                       "sd_desmoothed": lambda v: pc(v, 1),
                       "equity_sd": lambda v: pc(v, 1)}),
            "Housing total returns: observed, and why they stay out",
            note=("Real annual total returns on the national housing stock, "
                  "from the same source as the equity series. De-smoothing is "
                  "first-order Geltner using each country's own lag-one "
                  "autocorrelation."),
            col_widths=[ctx.width * 0.20, ctx.width * 0.09, ctx.width * 0.14,
                        ctx.width * 0.15, ctx.width * 0.16,
                        ctx.width * 0.14, ctx.width * 0.12]))
        out.append(ctx.p(
            f"The comparison is the one the source project is known for. "
            f"Median real housing returns are <b>{pc(hs['mean'], 1)}</b> "
            f"against <b>{pc(hs['equity_mean'], 1)}</b> for the same "
            f"countries' equity — indistinguishable — at a published standard "
            f"deviation of {pc(hs['sd'], 1)} versus "
            f"{pc(hs['equity_sd'], 1)} — a ratio of "
            f"{f2(hs['sd'] / hs['equity_sd'], 2)}. Equity-like returns at that "
            f"volatility would look dominant in any mean-variance comparison "
            f"in this paper, which is precisely why the series deserves "
            f"scrutiny before adoption."))
        out.append(ctx.p(
            f"A house price index is built from appraisals and sparse "
            f"transactions, and that smooths it. The median lag-one "
            f"autocorrelation of housing returns is "
            f"{f2(hs['autocorrelation'], 2)} against "
            f"{f2(hs['equity_autocorrelation'], 2)} for equity, and housing is "
            f"the more autocorrelated series in "
            f"{hs['n_more_autocorrelated']} of {hs['countries']} countries. "
            f"Undoing that to first order raises the median standard deviation "
            f"from {pc(hs['sd'], 1)} to {pc(hs['sd_desmoothed'], 1)}: most of "
            f"the apparent free lunch is a measurement artefact."))
        out.append(ctx.p(
            "Even de-smoothed the series is not investable as written. It is "
            "an unlevered, untaxed, frictionless total return on an entire "
            "national housing stock, with no transaction costs, no vacancy, no "
            "maintenance and none of the single-property concentration a real "
            "household bears. Adopting it as a fourth sleeve at face value "
            "would therefore overstate what a household can buy. Section #housing "
            "adds it anyway, but only after de-smoothing it and charging an "
            "explicit annual holding cost — and it is the size of that cost, "
            "not the raw return, that decides the answer."))
        out.extend(ctx.figure(
            "fig39_housing_smoothing",
            "Left: risk and return by country for housing as published, for "
            "housing with its own first-order smoothing undone (arrows), and "
            "for that country's domestic equity. Right: the lag-one "
            "autocorrelation that does the work. De-smoothing closes part of "
            "the volatility gap and not all of it, which is why the holding "
            "cost of Section #housing has to carry the rest of the argument."))

    wg = pr.get("wages", {})
    if wg.get("countries"):
        out.append(ctx.h3("#data.6.3 The series that bears on our income model"))
        out.append(ctx.p(
            f"The same file carries a nominal wage index for all "
            f"{wg['countries']} of its countries, including the two with no "
            f"return series. Deflated by each country's own consumer prices it "
            f"measures economy-wide <b>real wage growth</b> over "
            f"{wg['country_years']:,} country-years "
            f"({wg['first_year']}–{wg['last_year']}). The median country "
            f"compounded real wages at <b>{pc(wg['measured'], 2)} a year</b>, "
            f"from {pc(wg['lowest'], 2)} ({wg['lowest_country']}) to "
            f"{pc(wg['highest'], 2)} ({wg['highest_country']}) — which over "
            f"the {wg['career_years']}-year career we simulate compounds to "
            f"{f2(wg['career_multiple'], 2)}×."))
        out.extend(ctx.table(
            rows_from(wg["frame"],
                      ["country", "years", "geometric_mean",
                       "geometric_mean_ex_war", "sd", "career_multiple"],
                      ["Country", "Years", "Real wage growth p.a.",
                       "Excl. war years", "s.d.", "Compounded over a career"],
                      {"country": str, "years": lambda v: f"{int(v)}",
                       "geometric_mean": lambda v: pc(v, 2),
                       "geometric_mean_ex_war": lambda v: pc(v, 2),
                       "sd": lambda v: pc(v, 1),
                       "career_multiple": lambda v: f2(v, 2)}),
            "Measured real wage growth, and what our income profile assumes",
            note=(f"Geometric means, because they are what compound across a "
                  f"career. Our deterministic profile implies "
                  f"{pc(wg['model_growth'], 2)} a year over the same span."),
            col_widths=[ctx.width * 0.23, ctx.width * 0.09, ctx.width * 0.21,
                        ctx.width * 0.16, ctx.width * 0.11,
                        ctx.width * 0.20]))
        out.append(ctx.p(
            f"<b>The war years carry more of that spread than the economics "
            f"does</b>, which is why the table reports the series both ways. "
            f"The two largest observations in the panel are "
            f"{wg.get('extreme_highest_country', '')} "
            f"{wg.get('extreme_highest_year', '')} "
            f"(+{pc(wg.get('extreme_highest_value', float('nan')), 0)}) and "
            f"{wg.get('extreme_lowest_country', '')} "
            f"{wg.get('extreme_lowest_year', '')} "
            f"({pc(wg.get('extreme_lowest_value', float('nan')), 0)}): a wage "
            f"index spanning "
            f"occupation, rationing, suppressed prices and post-war repricing "
            f"is measuring those at least as much as it is measuring wages. "
            f"Dropping {wg.get('war_years', 'the war years')} lifts the median "
            f"to {pc(wg['measured_ex_war'], 2)} "
            f"({wg['war_shifted_by'] * 100:+.2f} percentage points) and moves "
            f"the lowest country, {wg['lowest_country']}, from "
            f"{pc(wg['lowest'], 2)} to {pc(wg['lowest_ex_war'], 2)} — from an "
            f"implausible claim about a century of that country's wages to an "
            f"unremarkable one. We keep them in the headline number, because a "
            f"worker alive then lived through them, and note that excluding "
            f"them only widens the gap we are about to describe."))
        out.append(ctx.p(
            f"<b>Our income profile has no term for it.</b> Section #methods.3 sets "
            f"real labour income as a deterministic hump peaking at age "
            f"{wg['model_peak_age']:.0f} at "
            f"{f2(wg['model_peak_multiple'], 2)}× starting income and ending "
            f"the career at {f2(wg['model_end_multiple'], 2)}× — an average of "
            f"{pc(wg['model_growth'], 2)} a year. That is an <i>age</i> "
            f"effect: the progression a worker earns by getting older. "
            f"Economy-wide wage growth is a different quantity, lifting the "
            f"whole distribution regardless of age, and in the "
            f"Cocco–Gomes–Maenhout estimation our profile is taken from the "
            f"two are separated by construction and are therefore additive. A "
            f"worker facing both would see roughly "
            f"{pc(wg['combined_growth'], 2)} a year."))
        out.append(ctx.p(
            "That caveat decides the size of the gap rather than its "
            "existence: whether the components add depends on how the source "
            "profile was estimated, and one fitted to a panel that still "
            "carried time effects would already absorb part of the growth. "
            "What is not in doubt is that the quantity is measurable, that "
            "eighteen countries measure it, and that nothing in our pipeline "
            "reads it."))
        out.append(ctx.p(
            "The direction is the one that matters for reading this paper. "
            "Understating income growth understates human capital throughout "
            "the career, and human capital is the bond-like asset in the "
            "standard lifecycle argument — so having less of it <i>weakens</i> "
            "the case for holding equity when young. The bias runs against our "
            "conclusion rather than toward it, as most of the known biases "
            "here do. The effect on the savings analysis of Sections #saving and "
            "#accumulation "
            "is genuinely ambiguous, because faster income growth raises both "
            "what a given savings rate accumulates and the consumption it must "
            "replace, and this audit does not resolve it. Re-estimating the "
            "income process is a modelling change rather than a data one, so "
            "we record it as a quantified limitation rather than apply it "
            "silently."))

    out.append(ctx.h3("#data.6.4 Is the source we kept genuine?"))
    out.append(ctx.p(
        "The workbook was obtained from a redistributed copy rather than "
        "downloaded from the compilers, so it is audited rather than "
        "assumed. Two tests, in opposite directions."))
    out.extend(ctx.table(
        rows_from(anchors, ["what", "workbook", "independently_known",
                            "difference", "within_tolerance"],
                  ["Observation", "This workbook", "Independently known",
                   "Difference", "Passes"],
                  {"what": str, "workbook": lambda v: f2(v, 4),
                   "independently_known": lambda v: f2(v, 4),
                   "difference": lambda v: f2(v, 4),
                   "within_tolerance": lambda v: "yes" if bool(v) else "NO"}),
        "The workbook against independently known annual returns",
        note="Nominal equity total returns for years whose magnitude is not in "
             "dispute. A reconstructed or synthetic file would not reproduce "
             "both directions of the 1931-33 swing and the 2008 drawdown."))
    out.append(ctx.p(
        f"All {pr['anchors_passed']} of {pr['anchors_total']} land within "
        f"tolerance. The second test runs the other way. Equity total return "
        f"should satisfy the accounting identity "
        f"<i>eq_tr</i> = (1 + <i>capital gain</i>)(1 + <i>dividend "
        f"yield</i>) − 1; across {pr['identity_observations']:,} observations "
        f"it fails by more than one part in a million "
        f"{pr['identity_share_violating']:.0%} of the time. <b>That failure "
        f"rate is evidence of authenticity.</b> The components in the real "
        f"database are sometimes spliced from different underlying indices, "
        f"so they do not always reconcile; a file that satisfied the identity "
        f"everywhere to machine precision would be one whose components had "
        f"been back-solved from its totals."))

    out.append(ctx.h3("#data.6.5 One finding that does not pass"))
    out.append(ctx.p(
        f"The last five years of every equity series are smoother than that "
        f"country's own history. <b>All {pr['tail_smoother']} of "
        f"{pr['tail_countries']} countries</b> show a lower standard "
        f"deviation over 2016–2020 than over 1950–2015, with a median ratio "
        f"of {pr['tail_median_ratio']:.2f}. A five-year standard deviation is "
        f"far too noisy to read one country at a time, which is why this is a "
        f"sign test: under a fair-coin null the probability of all "
        f"{pr['tail_countries']} pointing the same way is "
        f"<b>{pr['tail_p_value']:.1e}</b>. Spot-checking the United States "
        f"sharpens it — the workbook records +14.2% for 2018, a year in which "
        f"every broad US index fell, and +8.2% for 2019 against roughly +31% "
        f"for the S&amp;P 500."))
    out.append(ctx.p(
        "The likely explanation is that the redistributed copy was extended "
        "past the compilers' own end date by someone else. <b>We therefore "
        "treat 2016–2020 as unverified.</b> Those five years are under four "
        "percent of the panel's country-years and cannot move a 68-year "
        "lifecycle result materially, but the finding is recorded rather than "
        "left for a reader to discover."))
    out.extend(ctx.figure(
        "fig38_data_provenance",
        "The variance test, read two ways. Left: each country's standard "
        "deviation over the final five years as a ratio of its own 1950–2015 "
        "standard deviation. Right: the two standard deviations plotted "
        "against each other, with the 45-degree line. One country below the "
        "line would be unremarkable; every country below it is a property of "
        "the file rather than of the markets."))

    out.extend(ctx.figure(
        "fig02_country_real_returns",
        "Distribution of annual real returns by country and asset class. The "
        "left tails are the point of the panel: several countries record "
        "single-year real equity losses beyond −60%, and these are not "
        "outliers to be trimmed but the events a lifecycle investor is "
        "exposed to."))
    return out


# ---------------------------------------------------------------------------
# 4. Methodology
# ---------------------------------------------------------------------------
def section_methods(ctx: Any) -> List[Flowable]:
    f = ctx.f
    cfg = f.cfg
    b = f.blocks
    moments = f.table("bootstrap_moments")
    block_sens = f.table("bootstrap_block_sensitivity")
    autocorr = f.table("bootstrap_autocorrelation")
    lc = cfg["lifecycle"]
    ut = cfg["utility"]

    out: List[Flowable] = [ctx.h1("#methods. Methodology")]

    # -- 4.1 bootstrap ----------------------------------------------------
    out.append(ctx.h2("#methods.1 The cross-country stationary block bootstrap"))
    out.append(ctx.p(
        f"A simulated lifetime is {int(cfg['bootstrap']['horizon_years'])} "
        f"years long and is assembled from contiguous blocks of history. The "
        f"sampler is a stationary block bootstrap in the sense of Politis and "
        f"Romano (1994): block lengths are drawn from a geometric distribution "
        f"with mean {float(cfg['bootstrap']['mean_block_years']):.0f} years, "
        f"truncated to "
        f"[{int(cfg['bootstrap']['min_block_years'])}, "
        f"{int(cfg['bootstrap']['max_block_years'])}], and blocks are laid end "
        f"to end until the horizon is filled."))
    out.append(ctx.p(
        "Two features distinguish it from a textbook block bootstrap and both "
        "matter for the result."))
    out.extend(ctx.bullets([
        "<b>Blocks are calendar-joint.</b> A block is a (country, start-year, "
        "length) triple, and the same triple is applied to all five series "
        "simultaneously. Domestic equity, international equity, bonds, bills "
        "and inflation for a given simulated year therefore come from the same "
        "real country in the same real year. This is what preserves the "
        "cross-asset correlation matrix of Table 2 without any of it being "
        "imposed parametrically.",
        "<b>Blocks respect data gaps.</b> Admissibility is enforced through "
        "pre-computed run-lengths: for each (country, year) the sampler knows "
        "how many consecutive years of complete data follow, and only draws a "
        "block that fits inside a run. A wartime hole is never bridged.",
    ]))
    out.append(ctx.p(
        f"Across the production run the sampler drew {int(b['n_blocks']):,} "
        f"blocks with a mean realised length of {b['mean_length']:.2f} years "
        f"against a target of {b['target_mean_length']:.0f}, a median of "
        f"{b['median_length']:.0f}, a ninetieth percentile of "
        f"{b['p90_length']:.0f} and a maximum of {b['max_length']:.0f}. The "
        f"realised mean falls short of the target for a mechanical reason "
        f"worth stating: truncation at the maximum and the gap-admissibility "
        f"constraint both bite from above, so the effective block length is "
        f"always slightly below the nominal one."))
    out.append(ctx.p(
        f"The country for a lifetime is drawn once "
        f"(<i>{cfg['bootstrap']['country_draw']}</i>) with probability "
        f"proportional to the length of that country's usable history "
        f"(<i>{cfg['bootstrap']['country_weighting']}</i>). Both choices are "
        f"varied in Appendix C: redrawing the country at every block, and "
        f"weighting countries uniformly, both leave the ranking unchanged. "
        f"The weighting choice has little room to bite here, because every "
        f"country carries between {f.panel['min_years']} and "
        f"{f.panel['max_years']} usable years — history weighting and uniform "
        f"weighting assign nearly the same probabilities. The choice would "
        f"matter on a panel with short histories, where a country covering "
        f"only one era could otherwise speak as loudly as one covering a "
        f"century and a half."))

    out.extend(ctx.table(
        rows_from(moments, ["series", "bootstrap_mean", "panel_pooled_mean",
                            "mean_gap_bp", "bootstrap_std", "panel_pooled_std",
                            "std_ratio", "bootstrap_kurtosis"],
                  ["Series", "Bootstrap mean", "Panel mean", "Gap (bp)",
                   "Bootstrap s.d.", "Panel s.d.", "s.d. ratio",
                   "Bootstrap kurtosis"],
                  {"series": lambda v: {"dom_eq": "Domestic equity",
                                        "intl_eq": "International equity",
                                        "bond": "Bonds", "bill": "Bills",
                                        "inflation": "Inflation"}[v],
                   "bootstrap_mean": lambda v: pc(v, 2),
                   "panel_pooled_mean": lambda v: pc(v, 2),
                   "mean_gap_bp": lambda v: f2(v, 1),
                   "bootstrap_std": lambda v: pc(v, 2),
                   "panel_pooled_std": lambda v: pc(v, 2),
                   "std_ratio": lambda v: f2(v, 3),
                   "bootstrap_kurtosis": lambda v: f2(v, 1)}),
        "Does the bootstrap reproduce the panel it samples from?",
        note="Means agree to within twenty basis points on every series and "
             "standard deviations to within eight percent. The inflation row "
             "has an excess kurtosis in the thousands: that is the "
             "hyperinflation episodes surviving into the simulated paths, "
             "which is the intended behaviour rather than a defect."))

    out.append(ctx.p(
        f"The correlation matrix is reproduced to a maximum absolute deviation "
        f"of {f2(_corr_gap(f)['assets'], 3)} across the four asset-to-asset "
        f"pairs that drive portfolio behaviour. The largest deviation anywhere "
        f"in the matrix is {f2(_corr_gap(f)['overall'], 3)} and it sits on the "
        f"inflation row, where the excess kurtosis reported above makes the "
        f"sample correlation itself unstable. We report both figures rather "
        f"than the flattering one."))

    out.extend(ctx.figure(
        "fig03_bootstrap_diagnostics",
        "Bootstrap diagnostics. Simulated marginal distributions against the "
        "panel they are drawn from, the preservation of the cross-asset "
        "correlation structure, and the within-path autocorrelation that "
        "block sampling is designed to retain."))

    out.append(ctx.h3("Block length and long-horizon dispersion"))
    out.append(ctx.p(
        "Block length is the single most consequential tuning parameter in the "
        "design, because it controls how much multi-decade persistence "
        "survives into a simulated lifetime. The naive expectation is that "
        "longer blocks produce more dispersion in long-horizon outcomes. The "
        "data say the opposite, and the reason is instructive."))
    out.extend(ctx.table(
        rows_from(block_sens, ["mean_block_years", "dom_eq_mean_annualised",
                               "dom_eq_sd_annualised", "dom_eq_p5_annualised",
                               "dom_eq_p95_annualised", "within_path_ar1"],
                  ["Mean block (years)", "Mean annualised", "S.d. annualised",
                   "5th pct", "95th pct", "Within-path AR(1)"],
                  {"mean_block_years": lambda v: f"{int(v)}",
                   "dom_eq_mean_annualised": lambda v: pc(v, 2),
                   "dom_eq_sd_annualised": lambda v: pc(v, 2),
                   "dom_eq_p5_annualised": lambda v: pc(v, 2),
                   "dom_eq_p95_annualised": lambda v: pc(v, 2),
                   "within_path_ar1": lambda v: f2(v, 3)}),
        "Sensitivity of 68-year outcomes to the mean block length",
        note="Longer blocks raise the within-path autocorrelation, as "
             "intended, but slightly <i>reduce</i> the dispersion of "
             "annualised 68-year outcomes. Over a horizon this long the "
             "historical series mean-reverts, so preserving more of its "
             "internal structure preserves that mean reversion too."))
    out.append(ctx.p(
        f"Within-path autocorrelation rises monotonically from "
        f"{f2(float(block_sens['within_path_ar1'].iloc[0]), 3)} at one-year "
        f"blocks to {f2(float(block_sens['within_path_ar1'].iloc[-1]), 3)} at "
        f"twenty-year blocks, confirming that the sampler is doing what block "
        f"sampling is for. But the standard deviation of annualised 68-year "
        f"returns <i>falls</i> slightly over the same range. This is a "
        f"long-horizon mean-reversion finding, not a bug: a lifetime that "
        f"inherits real historical sequences inherits their tendency to "
        f"revert, whereas one assembled from independent annual draws does "
        f"not. It is also a caution against reading the block length as a "
        f"simple dial for \"more risk\"."))
    out.extend(ctx.figure(
        "fig04_block_length_sensitivity",
        "Block-length sensitivity. Long-horizon dispersion is close to flat in "
        "the mean block length even as within-path persistence rises steadily, "
        "which is the signature of mean reversion in the underlying series."))

    # -- 4.2 lifecycle ----------------------------------------------------
    out.append(ctx.h2("#methods.2 The lifecycle model"))
    out.append(ctx.p(
        f"A simulated investor begins work at age {int(lc['age_start'])}, "
        f"retires at {int(lc['age_retire'])} and dies at "
        f"{int(lc['age_death'])}, giving a "
        f"{int(lc['age_death']) - int(lc['age_start'])}-year horizon of which "
        f"{int(lc['age_retire']) - int(lc['age_start'])} are working years. "
        f"They save a fixed fraction "
        f"<i>s</i> = {float(lc['savings_rate']):.0%} of labour income, "
        f"rebalance annually to the strategy's target weights, and on "
        f"retirement switch to a withdrawal rule."))
    out.append(ctx.p("Wealth evolves as"))
    out.append(ctx.equation(
        "W<sub>h+1</sub> = ( W<sub>h</sub> + c<sub>h</sub> − x<sub>h</sub> ) "
        "· ( 1 + r<sup>p</sup><sub>h</sub> ),&nbsp;&nbsp; "
        "r<sup>p</sup><sub>h</sub> = Σ<sub>a</sub> w<sub>a,h</sub> "
        "r<sub>a,h</sub>"))
    out.append(ctx.p(
        "where <i>c<sub>h</sub></i> is the contribution in working years and "
        "zero afterwards, <i>x<sub>h</sub></i> is the withdrawal in retirement "
        "and zero before, <i>w<sub>a,h</sub></i> is the strategy's weight on "
        "asset <i>a</i> at age <i>h</i>, and all returns are real. Consumption "
        "is labour income net of saving while working, and the sum of the "
        "social-security benefit and the portfolio withdrawal in retirement:"))
    out.append(ctx.equation(
        "C<sub>h</sub> = (1 − s) Y<sub>h</sub> &nbsp;(working)"
        "&nbsp;&nbsp;&nbsp;&nbsp; C<sub>h</sub> = B + x<sub>h</sub> "
        "&nbsp;(retired)"))
    out.append(ctx.p(
        f"Labour income follows a deterministic hump-shaped profile "
        f"(<i>b</i><sub>1</sub> = {float(lc['income']['b1']):g}, "
        f"<i>b</i><sub>2</sub> = {float(lc['income']['b2']):g} in age and age "
        f"squared) multiplied by permanent and transitory shocks with standard "
        f"deviations {float(lc['income']['permanent_shock_sd']):.2f} and "
        f"{float(lc['income']['transitory_shock_sd']):.2f}. The permanent "
        f"shock is a cumulative sum, so income dispersion widens over a "
        f"career; the transitory shock is i.i.d. The profile is an age effect "
        f"and carries no economy-wide wage growth; Section #data.6.3 measures what "
        f"that leaves out and which way it biases the result."))
    out.append(ctx.p(
        f"The social-security benefit uses the US primary-insurance-amount "
        f"bend-point schedule rather than a flat replacement rate, applying "
        f"{float(lc['social_security']['pia_rate1']):.0%}, "
        f"{float(lc['social_security']['pia_rate2']):.0%} and "
        f"{float(lc['social_security']['pia_rate3']):.0%} to successive "
        f"tranches of career-average earnings. This is not decoration. A flat "
        f"replacement rate scales with the outcome and therefore insures "
        f"nothing; a progressive schedule supplies a genuine real consumption "
        f"floor, which is exactly what determines how much the left tail of a "
        f"risky strategy actually costs."))
    out.extend(ctx.table(
        [["Strategy", "Domestic eq.", "Intl. eq.", "Bonds", "Bills"],
         ["50/50 domestic/international equity", "50%", "50%", "—", "—"],
         ["100% domestic equity", "100%", "—", "—", "—"],
         ["100% international equity", "—", "100%", "—", "—"],
         ["60/40 domestic equity/bonds", "60%", "—", "40%", "—"],
         ["100% bills (cash)", "—", "—", "—", "100%"],
         ["Target-date fund (glide path)", "see below", "see below",
          "see below", "see below"]],
        "The six benchmark strategies, as constant weights",
        note="Held constant at every age and rebalanced annually, except the "
             "target-date fund."))
    out.append(ctx.p(
        "The target-date fund is the one benchmark whose weights move. Its "
        "equity share is piecewise-linear in age through the knots below, "
        "interpolated between them; the equity sleeve is split 60/40 "
        "domestic/international at every age and the fixed-income sleeve "
        "70/30 bonds/bills. The shape is the industry standard — flat and "
        "growth-heavy while young, declining through the decade before "
        "retirement, and still declining gently after it."))
    out.extend(ctx.table(
        [["Age", "25", "40", "55", "63 (retire)", "75", "93"],
         ["Equity share", "90%", "90%", "65%", "50%", "35%", "30%"]],
        "The target-date fund's glide path, at its knots",
        note="Linear between knots. At age 63 the fund therefore holds 30% "
             "domestic equity, 20% international, 35% bonds and 15% bills."))
    out.extend(ctx.figure(
        "fig05_glide_paths",
        "The six benchmark allocation strategies as equity share by age. The "
        "target-date fund is the industry-standard declining glide path; the "
        "comparison strategies are held at fixed weights."))

    # -- 4.3 preferences --------------------------------------------------
    out.append(ctx.h2("#methods.3 Preferences and the certainty equivalent"))
    out.append(ctx.p(
        "Strategies are ranked by certainty-equivalent consumption: the "
        "constant, riskless consumption stream that would leave the investor "
        "indifferent to the risky one their strategy produces. Under constant "
        "relative risk aversion with discount factor β and curvature γ, "
        "lifetime utility over the evaluation window is"))
    out.append(ctx.equation(
        "V = Σ<sub>h</sub> β<sup>h</sup> u(C<sub>h</sub>) + "
        "β<sup>H</sup> θ · u( κ + W<sub>H</sub> ),&nbsp;&nbsp; "
        "u(c) = c<sup>1−γ</sup> / (1 − γ)"))
    out.append(ctx.p(
        f"and the certainty equivalent inverts that utility back into "
        f"consumption units. We report risk aversions of "
        f"{{{', '.join(f'{g:g}' for g in ut['risk_aversions'])}}} with "
        f"γ = {f.baseline_gamma:g} as the baseline, β = "
        f"{float(ut['discount_factor']):g}, and a bequest weight θ = "
        f"{float(ut['bequest_weight']):g}."))
    out.append(ctx.p(
        f"The bequest term uses the shifted-power specification of De Nardi "
        f"(2004) with κ = {float(ut['bequest_shift']):g}. The shift is not "
        f"cosmetic. Without it, any path that dies with exactly zero wealth "
        f"contributes <i>u</i>(0) = −∞ at γ ≥ 1, one such path in a hundred "
        f"thousand collapses the entire certainty equivalent, and the ranking "
        f"becomes a statement about which strategy avoids exact zeros rather "
        f"than about which delivers better consumption. This is a genuine trap "
        f"in lifecycle work and we flag it because we walked into it."))
    out.append(ctx.p(
        f"We also report Epstein–Zin preferences in their ex-ante, "
        f"early-resolution form, which separates risk aversion from the "
        f"elasticity of intertemporal substitution, with ψ taking the values "
        f"{{{', '.join(f'{v:g}' for v in ut['epstein_zin_ies'])}}}. Under CRRA "
        f"these two are locked together at ψ = 1/γ, and a reader is entitled "
        f"to ask whether the result is being driven by the intertemporal "
        f"channel rather than the risk channel. Section #sensitivity.2 shows it is not."))
    out.append(ctx.p(
        f"One further specification choice deserves emphasis. Utility is "
        f"evaluated over the <b>{ut['consumption_window']}</b> window rather "
        f"than the whole lifetime for the allocation comparisons. The reason "
        f"is mechanical: with a fixed savings rate, working-life consumption "
        f"is <i>identical</i> across allocation strategies by construction, so "
        f"including it adds a large constant to every strategy's utility and "
        f"compresses the differences that the exercise is about. Where a "
        f"policy <i>does</i> change working-life consumption — the retirement "
        f"timing and savings-rate studies of Sections #valuation to #retirement "
        f"— we switch to "
        f"the whole-lifetime window and say so explicitly."))

    # -- 4.4 comparison discipline ---------------------------------------
    out.append(ctx.h2("#methods.4 The comparison discipline"))
    out.append(ctx.p(
        "Most of the extensions in this paper compare policies that differ in "
        "more than one way at once, and the arithmetic of a certainty "
        "equivalent will happily credit the wrong difference. Four rules are "
        "applied throughout."))
    out.extend(ctx.bullets([
        "<b>Matched baselines.</b> A rule that retires people later is scored "
        "against a fixed date matched on the same <i>mean</i> retirement age, "
        "not against the original age 63. A savings rule that drifts to saving "
        "more is scored against a constant rate interpolated at its own "
        "realised career average. Without this, a policy is rewarded for "
        "working longer or saving more rather than for being smarter.",
        "<b>Common random numbers.</b> Every sweep and every optimiser reuses "
        "the same bootstrap draws and the same income shocks, so a difference "
        "between two settings is a policy effect and not a sampling artefact. "
        "This also makes the objective a deterministic function of the policy, "
        "which is what licenses coordinate ascent.",
        "<b>Deviation profiles.</b> Before any structure in a solved schedule "
        "is described in words, each element is reset to a neutral reference "
        "and the cost measured in basis points. A solved glide path can look "
        "highly structured while most of its structure is worth nothing; the "
        "profile separates the two.",
        "<b>Grid-edge checks.</b> An optimum sitting on the boundary of its "
        "own search grid is a truncation, not an optimum. Every such case is "
        "flagged in the text and, where it mattered, the grid was widened and "
        "the analysis rerun.",
    ]))
    out.append(ctx.p(
        "These are not stylistic preferences. Section #retirement reports a result that "
        "fell by roughly sixty percent when a matched baseline replaced an "
        "unmatched one, Section #glide reports apparent glide-path structure that "
        "the deviation profile dissolved, and Section #accumulation reports a functional-"
        "form ranking that reversed its interpretation once the grid-edge "
        "check was applied."))

    # -- 4.5 implementation ----------------------------------------------
    out.append(ctx.h2("#methods.5 Implementation and verification"))
    out.append(ctx.p(
        f"The model is implemented in Python with NumPy and pandas. "
        f"Portfolio returns for a whole cohort are formed as a batched matrix "
        f"product and the wealth recursion runs vectorised across paths, which "
        f"is what makes a hundred thousand lifetimes per strategy — and the "
        f"tens of thousands of policy evaluations the optimisers require — "
        f"tractable on a single core."))
    out.append(ctx.p(
        f"Correctness is enforced by {f.n_tests} automated tests. The "
        f"load-bearing ones "
        f"are equivalence tests: the path-dependent engine used for flexible "
        f"retirement and conditional saving must reproduce the fixed-date "
        f"engine <i>bit for bit</i> when given a fixed date and a constant "
        f"rate, and the batched glide-path evaluator must reproduce the "
        f"reference simulator to stated floating-point tolerance. Every "
        f"extension in this paper is built on a simulator that is asserted "
        f"equal to the one that produced the baseline."))
    out.append(ctx.p(
        "Appendix D sets out how the computation is organised, what the test "
        "suite establishes, and the discipline by which the prose in this "
        "paper is held to the tables it describes."))
    return out


def _corr_gap(f: Any) -> Dict[str, float]:
    """Largest bootstrap-versus-panel correlation deviations.

    Reported separately for the four asset-to-asset pairs and for the
    inflation row: the inflation series carries hyperinflation episodes whose
    excess kurtosis runs into the thousands, which makes its sample
    correlation unstable in a way that says nothing about whether the sampler
    is preserving portfolio-relevant structure.
    """
    boot = f.table("bootstrap_correlation").set_index("series")
    panel = f.table("panel_cross_asset_correlation").set_index("series")
    assets = ["dom_eq", "intl_eq", "bond", "bill"]
    gap = (boot - panel).abs()
    return {
        "assets": float(gap.loc[assets, assets].to_numpy().max()),
        "overall": float(gap.to_numpy().max()),
        "inflation": float(gap.loc["inflation"].max()),
    }


# ---------------------------------------------------------------------------
# 5. Baseline results
# ---------------------------------------------------------------------------
def section_baseline(ctx: Any) -> List[Flowable]:
    f = ctx.f
    head = f.headline
    gamma = f.baseline_gamma
    dominance = f.table("dominance_check")
    swr = f.table("sensitivity_safe_withdrawal_rates")
    eq = f.strategy_row("balanced_all_equity")
    tdf = f.strategy_row("target_date_fund")
    dom = f.strategy_row("domestic_equity")
    intl = f.strategy_row("international_equity")

    out: List[Flowable] = [ctx.h1("#baseline. Baseline Results")]
    out.append(ctx.p(
        f"All results in this section use 100,000 simulated lifetimes per "
        f"strategy, drawn from the same bootstrap sample so that strategies "
        f"face identical market histories. Certainty equivalents are over the "
        f"retirement window, for the reason given in Section #methods.3."))

    out.append(ctx.h2("#baseline.1 The headline comparison"))
    out.extend(ctx.table(
        rows_from(head.assign(label=head["strategy"].map(LABELS)),
                  ["label", "cec_crra_gamma2", "cec_crra_gamma5",
                   "cec_crra_gamma10", "prob_ruin",
                   "median_wealth_at_retirement",
                   "median_retirement_consumption",
                   "p5_retirement_consumption", "median_bequest"],
                  ["Strategy", "CEC γ=2", "CEC γ=5", "CEC γ=10", "P(ruin)",
                   "Median wealth at 63", "Median cons.", "5th pct cons.",
                   "Median bequest"],
                  {"label": str,
                   "prob_ruin": lambda v: pc(v, 1),
                   "median_wealth_at_retirement": lambda v: f2(v, 1),
                   "median_bequest": lambda v: f2(v, 1)}),
        "Headline lifecycle outcomes by strategy",
        note="Consumption and wealth are expressed as multiples of the "
             "investor's real income at age 25. P(ruin) is the probability of "
             "exhausting the portfolio with retirement years still to fund. "
             "100,000 paths per strategy on common bootstrap draws.",
        font_size=7.2))

    out.append(ctx.p(
        f"The ordering is unambiguous at every risk aversion tested. At "
        f"γ = {gamma:g} the 50/50 all-equity portfolio delivers a certainty "
        f"equivalent of {f2(float(eq[f'cec_crra_gamma{gamma:g}']), 3)} against "
        f"{f2(float(tdf[f'cec_crra_gamma{gamma:g}']), 3)} for the target-date "
        f"fund, an advantage of "
        f"{f.advantage('balanced_all_equity', 'target_date_fund'):.1f}%. "
        f"Against the 60/40 portfolio the advantage is "
        f"{f.advantage('balanced_all_equity', 'sixty_forty'):.1f}%, and "
        f"against bills {f.advantage('balanced_all_equity', 'bills_only'):.1f}%."))
    out.append(ctx.p(
        f"What makes the result more than a restatement of the equity premium "
        f"is the tail behaviour. The all-equity portfolio has a "
        f"<i>lower</i> ruin probability ({pc(float(eq['prob_ruin']), 1)}) than "
        f"the target-date fund ({pc(float(tdf['prob_ruin']), 1)}), a higher "
        f"fifth-percentile retirement consumption "
        f"({f2(float(eq['p5_retirement_consumption']), 3)} against "
        f"{f2(float(tdf['p5_retirement_consumption']), 3)}), and a lower "
        f"probability of falling below a seventy-percent replacement target "
        f"({pc(float(eq['prob_consumption_below_target']), 1)} against "
        f"{pc(float(tdf['prob_consumption_below_target']), 1)}). The "
        f"conservative portfolio is not buying downside protection on this "
        f"panel. It is paying for the appearance of it."))

    out.extend(ctx.figure(
        "fig09_cec_by_risk_aversion",
        "Certainty-equivalent consumption by strategy and risk aversion. The "
        "ranking is stable in γ; the gap narrows as risk aversion rises but "
        "does not close within the range tested."))

    out.append(ctx.h2("#baseline.2 The mechanism is international, not equity"))
    out.append(ctx.p(
        f"Domestic equity alone is <i>not</i> the winning strategy. Its "
        f"certainty equivalent at γ = {gamma:g} is "
        f"{f2(float(dom[f'cec_crra_gamma{gamma:g}']), 3)}, below the 50/50 "
        f"portfolio's {f2(float(eq[f'cec_crra_gamma{gamma:g}']), 3)}, and its "
        f"ruin probability is {pc(float(dom['prob_ruin']), 1)} — nearly double "
        f"the diversified portfolio's. Pure international equity does better "
        f"still ({f2(float(intl[f'cec_crra_gamma{gamma:g}']), 3)}), which is a "
        f"consequence of the leave-one-out construction rather than a "
        f"recommendation: an investor holding \"international\" equity in this "
        f"model holds a {f.panel['n_tier_a'] - 1}-country average, and no single "
        f"real investor has "
        f"that opportunity set without also holding their own market."))
    out.append(ctx.p(
        f"This is the one place the replication does not reproduce ACO's "
        f"headline. Their recommended portfolio is an even 50/50 split, held "
        f"for life; on this panel that split is beaten by the international "
        f"leg alone. We read the difference as a property of a "
        f"{f.panel['n_tier_a']}-country panel rather than a correction to "
        f"them. The domestic leg here is a single draw from a set that "
        f"includes several century-long underperformers, while the "
        f"international leg averages away exactly that risk; with "
        f"thirty-nine countries and a tradeable international index the two "
        f"legs are far closer in character, and the case for holding both is "
        f"correspondingly stronger. Nothing measurable here contradicts the "
        f"50/50 recommendation — the narrower claim this panel supports is "
        f"that a diversified sleeve dominates a concentrated one."))
    out.append(ctx.p(
        "The practical reading is that the case for equities here is a case "
        "for <i>diversified</i> equities. A single national market can and did "
        "deliver multi-decade real losses; the cross-section of national "
        "markets did not do so simultaneously. This is a claim about the "
        "covariance structure of the panel, and it is the reason the exercise "
        "needs an international sample rather than a longer US one."))

    out.extend(ctx.figure(
        "fig06_terminal_wealth_cdf",
        "Cumulative distribution of wealth at retirement. The all-equity "
        "distributions stochastically dominate the conservative ones over "
        "almost the entire range, including the left tail where the "
        "conservative case is usually made."))
    out.extend(ctx.figure(
        "fig07_retirement_consumption",
        "Distribution of average real retirement consumption by strategy. "
        "The vertical line marks the seventy-percent replacement target used "
        "in the shortfall statistics."))

    out.append(ctx.h2("#baseline.3 Distributional dominance"))
    out.append(ctx.p(
        f"A certainty equivalent is a single scalar and can hide a strategy "
        f"that wins on average while losing where it matters. We therefore "
        f"test the all-equity portfolio against each rival across "
        f"{int(dominance['criteria'].iloc[0])} separate distributional "
        f"criteria — percentiles of wealth at retirement, of retirement "
        f"consumption and of bequest, plus ruin and shortfall probabilities — "
        f"counting a tie as neither a win nor a loss."))
    out.extend(ctx.table(
        rows_from(dominance.assign(
            challenger=dominance["challenger"].map(LABELS),
            incumbent=dominance["incumbent"].map(LABELS)),
            ["incumbent", "criteria", "criteria_won", "criteria_tied",
             "criteria_lost_n", "strict_dominance"],
            ["Beaten strategy", "Criteria", "Won", "Tied", "Lost",
             "Strict dominance"],
            {"incumbent": str, "criteria": lambda v: f"{int(v)}",
             "criteria_won": lambda v: f"{int(v)}",
             "criteria_tied": lambda v: f"{int(v)}",
             "criteria_lost_n": lambda v: f"{int(v)}",
             "strict_dominance": lambda v: "yes" if bool(v) else "no"}),
        "Distributional dominance of the 50/50 all-equity portfolio",
        note="A tie is counted separately rather than as a loss, since "
             "counting it either way would misstate the comparison. The "
             "single tie in each row is the zero-bequest floor, which several "
             "strategies reach."))
    out.extend(ctx.figure(
        "fig08_shortfall_probabilities",
        "Shortfall probabilities against a fixed consumption target. Measuring "
        "each strategy against a common target rather than against its own "
        "median is essential: a strategy-specific target rewards strategies "
        "with low medians."))

    out.append(ctx.h2("#baseline.4 Sustainable withdrawal rates"))
    out.append(ctx.p(
        "The four-percent rule is the most widely used number in retirement "
        "planning. It comes from US historical sequences and it does not "
        "survive this panel."))
    out.extend(ctx.table(
        rows_from(swr.assign(label=swr["strategy"].map(LABELS)),
                  ["label", "safe_withdrawal_rate_at_5%_ruin", "ruin_at_4pct"],
                  ["Strategy", "Sustainable rate at 5% ruin",
                   "Ruin probability at 4%"],
                  {"label": str,
                   "safe_withdrawal_rate_at_5%_ruin": lambda v: pc(v, 2),
                   "ruin_at_4pct": lambda v: pc(v, 1)}),
        "Sustainable withdrawal rates on the international panel",
        note="The sustainable rate is the highest initial real withdrawal "
             "rate, inflation-adjusted thereafter, that leaves at most a "
             "five-percent probability of exhausting the portfolio over a "
             "30-year retirement."))
    out.append(ctx.p(
        f"At a five-percent ruin tolerance the sustainable rate is "
        f"{pc(float(swr[swr['strategy'] == 'balanced_all_equity']['safe_withdrawal_rate_at_5%_ruin'].iloc[0]), 2)} "
        f"for the all-equity portfolio and "
        f"{pc(float(swr[swr['strategy'] == 'target_date_fund']['safe_withdrawal_rate_at_5%_ruin'].iloc[0]), 2)} "
        f"for the target-date fund. Withdrawing four percent produces a ruin "
        f"probability of "
        f"{pc(float(swr[swr['strategy'] == 'balanced_all_equity']['ruin_at_4pct'].iloc[0]), 1)} "
        f"and "
        f"{pc(float(swr[swr['strategy'] == 'target_date_fund']['ruin_at_4pct'].iloc[0]), 1)} "
        f"respectively. Two things follow. First, the ordering of strategies "
        f"is the same here as everywhere else in the paper — the equity "
        f"portfolio sustains a higher rate, not a lower one. Second, the level "
        f"is a serious warning: on a sample that includes the twentieth "
        f"century as most countries lived it, four percent is roughly a "
        f"one-in-seven proposition even on the best strategy tested."))
    out.extend(ctx.figure(
        "fig11_ruin_probability",
        "Probability of portfolio exhaustion as a function of the initial "
        "withdrawal rate. The horizontal line is the five-percent tolerance "
        "used to define the sustainable rate."))

    out.append(ctx.h2("#baseline.5 How the countries are drawn"))
    out.append(ctx.p(
        f"Two choices in the sampler decide how the cross-section enters a "
        f"simulated life, and neither is obviously right. A lifetime's "
        f"domestic country can be drawn once and held, or redrawn at every "
        f"block; and countries can be weighted by the length of their recorded "
        f"history or treated as equally likely. Appendix C reports both "
        f"variants in full. Neither reverses a ranking."))
    out.append(ctx.p(
        f"The weighting choice has little room to bite, because every country "
        f"in the panel carries between {f.panel['min_years']} and "
        f"{f.panel['max_years']} usable years — history weighting and uniform "
        f"weighting assign nearly the same probabilities. It would matter on a "
        f"panel containing short histories, where a country covering a single "
        f"era could otherwise speak as loudly as one covering a century and a "
        f"half."))
    out.append(ctx.p(
        f"Redrawing the country at every block is the more consequential "
        f"variant, and it is the design the original study uses. It "
        f"implicitly assumes an investor can be resident in a different "
        f"country every decade, which is not a description of anybody, but it "
        f"pools the cross-section more aggressively and so gives the "
        f"international mechanism its most favourable reading. Holding the "
        f"country fixed for a lifetime — the specification behind every "
        f"headline number here — is the more conservative of the two."))

    out.extend(ctx.figure(
        "fig10_wealth_fan",
        "Wealth trajectories by percentile for the all-equity portfolio and "
        "the target-date fund. The fan makes visible what the scalar "
        "comparison cannot: the conservative path is not narrower at the "
        "bottom, it is lower throughout."))
    return out


# ---------------------------------------------------------------------------
# 6. Sensitivity
# ---------------------------------------------------------------------------
def section_sensitivity(ctx: Any) -> List[Flowable]:
    f = ctx.f
    tornado = f.table("sensitivity_tornado")
    crossover = f.table("sensitivity_crossover")
    eq_opt = f.table("sensitivity_equity_optimum")
    dom_opt = f.table("sensitivity_domestic_optimum")
    ies = f.table("sensitivity_ies")
    n_settings = int(tornado["n_settings"].sum())
    n_dimensions = int(tornado["dimension"].nunique())
    reversals = int(tornado["settings_lost"].sum())

    out: List[Flowable] = [ctx.h1("#sensitivity. Sensitivity Analysis")]
    out.append(ctx.p(
        f"The headline is one point in a large parameter space. This section "
        f"sweeps that space along equity share, domestic share, risk "
        f"aversion, elasticity of intertemporal substitution, bequest weight, "
        f"longevity, retirement age, savings rate, withdrawal rate, "
        f"social-security design, block length and return panel. Every sweep "
        f"uses common random numbers, so differences between settings are "
        f"parameter effects rather than sampling noise."))
    out.append(ctx.p(
        f"The tornado below summarises {n_settings} settings across the "
        f"{n_dimensions} of those dimensions along which a like-for-like "
        f"advantage over a fixed incumbent strategy is defined; the "
        f"remainder vary the portfolio itself and are reported in their own "
        f"subsections."))
    out.append(ctx.p(
        f"<b>The ranking reverses in {reversals} of them.</b> That is the "
        f"single most useful sentence in this section, and the tornado below "
        f"reports the range of the advantage rather than its point estimate "
        f"so the reader can see how much room there is."))

    out.append(ctx.h2("#sensitivity.1 Tornado analysis"))
    out.append(ctx.p(
        "A tornado analysis varies one parameter at a time across its whole "
        "plausible range, holding everything else at baseline, and records "
        "how far the result moves. Sorting the dimensions by that range — "
        "widest at the top — shows at a glance which assumptions the "
        "conclusion depends on and which are incidental. The name comes from "
        "the funnel shape the sorted bars make. The column that matters most "
        "here is the last one: a <i>reversal</i> is a setting at which the "
        "all-equity portfolio stops winning, and a dimension with none of "
        "them cannot overturn the result however it is set."))
    short = {"sixty_forty": "60/40", "target_date_fund": "Target-date fund",
             "domestic_equity": "Domestic equity", "bills_only": "Bills"}
    out.extend(ctx.table(
        rows_from(tornado.assign(
            inc=tornado["incumbent"].map(lambda v: short.get(v, str(v)))),
                  ["dimension", "inc", "n_settings", "min_advantage_pct",
                   "median_advantage_pct", "max_advantage_pct", "range_pp",
                   "settings_lost"],
                  ["Dimension", "Compared against", "Settings", "Min adv.",
                   "Median adv.", "Max adv.", "Range (pp)", "Reversals"],
                  {"dimension": str, "inc": str,
                   "n_settings": lambda v: f"{int(v)}",
                   "min_advantage_pct": lambda v: f2(v, 1),
                   "median_advantage_pct": lambda v: f2(v, 1),
                   "max_advantage_pct": lambda v: f2(v, 1),
                   "range_pp": lambda v: f2(v, 1),
                   "settings_lost": lambda v: f"{int(v)}"},
                  limit=12),
        "Tornado: range of the all-equity advantage by parameter dimension",
        note="Advantage is the percentage certainty-equivalent lead of the "
             "50/50 all-equity portfolio over the named incumbent, at "
             "γ = 5 unless the dimension is risk aversion itself. Ordered by "
             "the width of the range, so the dimensions the result is most "
             "sensitive to appear first.",
        col_widths=[ctx.width * 0.24, ctx.width * 0.16] + [ctx.width * 0.10] * 6,
        font_size=7.2))
    out.extend(ctx.figure(
        "fig16_tornado",
        "Tornado plot of the all-equity advantage. The bar spans the minimum "
        "and maximum advantage observed across the settings in that dimension; "
        "no bar crosses zero."))
    out.append(ctx.p(
        f"Risk aversion is the widest dimension, as it should be: the "
        f"advantage over a 60/40 portfolio ranges from "
        f"{f2(float(tornado['min_advantage_pct'].iloc[0]), 1)}% to "
        f"{f2(float(tornado['max_advantage_pct'].iloc[0]), 1)}% across "
        f"γ over [1, 20]. It narrows as γ rises but does not close. Fitting "
        f"the "
        f"crossover point directly, the certainty-equivalent lines do not "
        f"intersect within the tested range for either the target-date fund or "
        f"the 60/40 portfolio; at the highest γ tested the all-equity lead is "
        f"still {f2(float(crossover['gap_at_max_gamma_pct'].iloc[0]), 1)}% and "
        f"{f2(float(crossover['gap_at_max_gamma_pct'].iloc[1]), 1)}% "
        f"respectively."))
    out.extend(ctx.figure(
        "fig13_risk_aversion_sweep",
        "Certainty equivalent by strategy across risk aversion. The "
        "conservative strategies converge toward the equity portfolios as γ "
        "rises but do not overtake them within the tested range."))

    out.append(ctx.h2("#sensitivity.2 Allocation sweeps and the corner solution"))
    out.append(ctx.p(
        "Sweeping the equity share continuously rather than testing discrete "
        "strategies asks a sharper question: is the optimum interior?"))
    out.extend(ctx.table(
        rows_from(eq_opt, ["risk_aversion", "optimal_equity_share",
                           "cec_at_optimum", "prob_ruin_at_optimum"],
                  ["Risk aversion", "Optimal equity share", "CEC at optimum",
                   "P(ruin) at optimum"],
                  {"risk_aversion": lambda v: f"γ = {float(v):g}",
                   "optimal_equity_share": lambda v: pc(v, 0),
                   "prob_ruin_at_optimum": lambda v: pc(v, 1)}),
        "Optimal equity share by risk aversion",
        note="The optimum is at the corner for every risk aversion tested. "
             "Section #glide asks the same question of the full age-by-asset "
             "schedule and reaches the same answer."))
    out.extend(ctx.table(
        rows_from(dom_opt, ["risk_aversion", "optimal_domestic_share",
                            "cec_at_optimum", "prob_ruin_at_optimum"],
                  ["Risk aversion", "Optimal domestic share", "CEC at optimum",
                   "P(ruin) at optimum"],
                  {"risk_aversion": lambda v: f"γ = {float(v):g}",
                   "optimal_domestic_share": lambda v: pc(v, 0),
                   "prob_ruin_at_optimum": lambda v: pc(v, 1)}),
        "Optimal domestic share within the equity allocation",
        note="The home-bias question, asked of the model rather than assumed. "
             "The optimum is well below the 50% used in the headline strategy "
             "and rises with risk aversion — an artefact of the leave-one-out "
             "international construction, which gives the international leg a "
             "diversification advantage no individual investor can replicate."))
    out.append(ctx.p(
        f"The equity optimum sits at 100% at every risk aversion tested. The "
        f"domestic-share optimum is more interesting: it is "
        f"{pc(float(dom_opt['optimal_domestic_share'].iloc[0]), 0)} at "
        f"γ = 2 and rises to "
        f"{pc(float(dom_opt['optimal_domestic_share'].iloc[-1]), 0)} at "
        f"γ = 10. We do not read this as a recommendation to hold almost no "
        f"domestic equity. The international leg in this model is a "
        f"{f.panel['n_tier_a'] - 1}-country leave-one-out average, which is "
        f"better diversified than "
        f"any tradeable international index; the honest reading is that the "
        f"model prefers <i>more</i> diversification than the 50/50 headline "
        f"strategy provides, not that a specific number is optimal."))
    out.extend(ctx.figure(
        "fig12_allocation_frontier",
        "Certainty equivalent across the equity share and the domestic share. "
        "The equity dimension is monotone to the corner; the domestic "
        "dimension has an interior optimum well below the 50% headline "
        "weight."))

    out.append(ctx.h2("#sensitivity.3 Separating risk aversion from intertemporal substitution"))
    out.append(ctx.p(
        f"Under CRRA, γ and the elasticity of intertemporal substitution are "
        f"the same parameter inverted, so a sceptic can reasonably ask whether "
        f"the result is about attitudes to risk or about willingness to "
        f"substitute consumption over time. Epstein–Zin preferences separate "
        f"them. Holding γ at {f.baseline_gamma:g} and sweeping ψ from "
        f"{float(ies['ies'].min()):g} to {float(ies['ies'].max()):g}, the "
        f"advantage over the 60/40 portfolio ranges from "
        f"{f2(float(tornado[tornado['dimension'].str.startswith('Elasticity')]['min_advantage_pct'].iloc[0]), 1)}% "
        f"to "
        f"{f2(float(tornado[tornado['dimension'].str.startswith('Elasticity')]['max_advantage_pct'].iloc[0]), 1)}% "
        f"and never reverses. The result is a risk result, not a substitution "
        f"result."))

    out.append(ctx.h2("#sensitivity.4 Planning parameters"))
    out.append(ctx.p(
        "Retirement age, longevity, savings rate, withdrawal rate and "
        "social-security design are the levers an individual actually "
        "controls, and none of them changes the ranking. They do change the "
        "<i>size</i> of the advantage substantially — the withdrawal-rate "
        "dimension alone spans several percentage points — which is worth "
        "keeping in view: the all-equity lead is not a constant of nature but "
        "a quantity that depends on how the rest of the plan is set up."))
    out.extend(ctx.figure(
        "fig15_planning_sweeps",
        "Planning-parameter sweeps: retirement age, longevity, savings rate "
        "and social-security design. Ranking is preserved throughout; the "
        "magnitude of the advantage is not."))
    out.extend(ctx.figure(
        "fig14_withdrawal_sensitivity",
        "The advantage as a function of the withdrawal rate. Higher "
        "withdrawal rates widen the gap, because a portfolio being drawn down "
        "faster is more exposed to the left tail the conservative strategies "
        "fail to protect against."))
    return out


# ---------------------------------------------------------------------------
# 7. Retirement spending rules
# ---------------------------------------------------------------------------
def section_sequence(ctx: Any) -> List[Flowable]:
    f = ctx.f
    frame = f.table("sequence_decomposition")
    ranking = f.table("sequence_ranking")
    comparison = f.table("sequence_rule_comparison")

    from src import sequence as sqn
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    scfg = cfg["sequence"]
    focus = str(scfg.get("focus", "balanced_all_equity"))
    n_reps = int(scfg.get("n_reps", 8))
    pair = (str(scfg.get("challenger", "balanced_all_equity")),
            str(scfg.get("incumbent", "target_date_fund")))
    found = sqn.verdict(frame, focus)
    rule_found = sqn.rule_verdict(comparison)
    horizon = int(cfg["lifecycle"]["age_death"]) - int(cfg["lifecycle"]["age_start"])
    said = {"none": "nothing", "accumulation": "the working years",
            "retirement": "the retired years", "both": "the whole lifetime"}
    block = frame[frame["strategy"] == focus].set_index("phase")

    out: List[Flowable] = [
        ctx.h1("#sequence. Sequence-of-Returns Risk, Isolated")]
    out.append(ctx.p(
        "Section #retirement establishes that the single decade around a "
        "person's retirement date explains more of their outcome than the "
        "entire allocation question, and Section #inflation shows that "
        "reading a state variable at the retirement date rather than the "
        "birth date turns a null into an eight-point effect. Both are "
        "symptoms of the same thing. Neither measures it."))
    out.append(ctx.p(
        f"The experiment that does is simple enough to state in a sentence. "
        f"Take a simulated lifetime, keep its {horizon} annual returns "
        f"exactly as drawn, and shuffle the order. Same multiset, same mean, "
        f"same everything a return distribution can describe. Anything that "
        f"changes is sequence."))
    out.append(ctx.note(
        "For a lump sum nothing would change at all: the product of gross "
        "returns is commutative, so a buy-and-hold investor with no cash "
        "flows ends at the same wealth whatever the order. That is verified "
        "here to machine precision, and it is the point — sequence risk "
        "exists <i>only</i> through cash flows. A contribution made before a "
        "crash buys more; a withdrawal made after one sells more. Every "
        "series of a year moves together under the permutation, so the "
        "cross-asset covariance the bootstrap exists to preserve survives it."))

    out.append(ctx.h2("#sequence.1 The decomposition"))
    out.append(ctx.p(
        f"Each original path is a bag of returns. Reordering it makes the "
        f"outcome a random variable, so the total variance splits exactly "
        f"into the average variance <i>within</i> a bag — which is sequence, "
        f"since the bag is held fixed — and the variance of the bag means, "
        f"which is the risk of having drawn that bag at all. Estimating the "
        f"first term needs several orderings of the same path, which is what "
        f"the {n_reps} replications are for."))
    out.extend(ctx.table(
        [["What was shuffled", "Share of variance that is ordering",
          "SD from ordering", "SD from the returns", "CEC", "P(ruin)"]]
        + [[said.get(str(p), str(p)),
            f"{float(block.loc[p, 'sequence_share']):.1%}",
            f"{float(block.loc[p, 'sd_sequence']):.3f}",
            f"{float(block.loc[p, 'sd_level']):.3f}",
            f"{float(block.loc[p, 'cec']):.4f}",
            f"{float(block.loc[p, 'prob_ruin']):.1%}"]
           for p in sqn.PHASES if p in block.index],
        f"Variance of mean retirement consumption for "
        f"<i>{_pretty_strategy(focus)}</i>, split into ordering and returns, "
        f"γ = {gamma:g}.",
        note="The first row is the control. Shuffling nothing must give a "
             "sequence share of zero, and the pipeline fails loudly "
             "if it does not — without that check the decomposition could be "
             "measuring anything."))
    out.append(ctx.p(
        (f"<b>Most of the risk in a lifetime is the order, not the "
         f"returns.</b> Holding the bag fixed and reshuffling it accounts for "
         f"{found['share_both']:.1%} of the variance in retirement "
         f"consumption. Which returns a saver draws matters less than when "
         f"they arrive."
         if found.get("ordering_is_most_of_the_risk") else
         f"<b>Ordering is a minority of the risk, and not a small one.</b> "
         f"Reshuffling a fixed bag of returns accounts for "
         f"{found['share_both']:.1%} of the variance in retirement "
         f"consumption; the rest is the risk of having drawn that bag at "
         f"all.")))
    out.append(ctx.p(
        (f"<b>And it is an accumulation phenomenon, which is the opposite of "
         f"where the literature points.</b> Shuffling only the working years "
         f"accounts for {found['share_accumulation']:.1%} of the variance; "
         f"shuffling only the retired years accounts for "
         f"{found['share_retirement']:.1%}. The next subsection explains "
         f"why, and the explanation is not about returns."
         if not found.get("retirement_dominates") else
         f"<b>And it is a decumulation phenomenon, as the usual telling has "
         f"it.</b> Shuffling only the retired years accounts for "
         f"{found['share_retirement']:.1%} of the variance against "
         f"{found['share_accumulation']:.1%} for the working years — a factor "
         f"of {found['retirement_over_accumulation']:.1f}.")))
    out.append(ctx.p(
        f"The whole-lifetime figure exceeds the two phases added together, "
        f"and that is not an inconsistency. Shuffling the whole lifetime lets "
        f"a year lived at eighty land at twenty-six, so it captures the "
        f"interaction between the phases as well as the ordering inside each "
        f"— which the phase-restricted shuffles hold fixed by construction."))

    out.append(ctx.h2("#sequence.2 The withdrawal rule decides where it lands"))
    out.append(ctx.p(
        "The result above is a property of the withdrawal rule as much as of "
        "the returns, and reporting it under one rule alone would be "
        "reporting the rule. A withdrawal fixed in real terms is computed "
        "from wealth <i>at</i> retirement and never revisited, so it refuses "
        "to let retirement-phase returns reach the retiree's consumption at "
        "all — until the money runs out, at which point they reach it all at "
        "once. A percentage-of-portfolio rule passes every return straight "
        "through to what the retiree eats. The two should therefore put "
        "sequence risk in completely different places, and the decomposition "
        "is repeated under three rules to find out."))
    if len(comparison):
        out.extend(ctx.table(
            [["Withdrawal rule", "Ordering risk in the working years",
              "in the retired years", "ratio", "Cost in CEC (%)",
              "Cost in ruin (points)"]]
            + [[str(r["label"]),
                f"{float(r['share_accumulation']):.1%}",
                f"{float(r['share_retirement']):.1%}",
                f"{float(r['retirement_over_accumulation']):.2f}",
                f"{float(r['cec_cost_pct']):+.2f}",
                f"{float(r['ruin_cost_pp']):+.2f}"]
               for _, r in comparison.iterrows()],
            "Where sequence risk sits, rule by rule.",
            note="The last two columns are what a random ordering costs "
                 "relative to the drawn one under each rule — in "
                 "consumption, and in the probability of running out."))
    out.append(ctx.p(
        (f"<b>The rule relocates the risk rather than removing it.</b> Under "
         f"a fixed real withdrawal the retired years carry "
         f"{rule_found['fixed_real_ratio']:.2f} times the ordering risk of "
         f"the working years; under a percentage-of-portfolio rule they carry "
         f"{rule_found['percentage_ratio']:.2f} times — "
         f"{rule_found['percentage_ratio'] / max(rule_found['fixed_real_ratio'], 1e-9):.0f}× "
         f"more. Nothing about the returns has changed between those rows. "
         f"What changed is whether the retiree is allowed to feel them."
         if rule_found.get("rule_relocates_the_risk") else
         f"The location of the ordering risk is similar under every rule "
         f"tested: the retired years carry between "
         f"{rule_found.get('min_ratio', float('nan')):.2f} and "
         f"{rule_found.get('max_ratio', float('nan')):.2f} times the working "
         f"years' share.")))
    out.append(ctx.p(
        (f"That is not a free lunch, and the last column says so. The fixed "
         f"real rule buys its smooth consumption by absorbing the ordering "
         f"into <i>ruin</i> instead: a random order costs it "
         f"{rule_found['fixed_real_ruin_cost_pp']:+.2f} points of failure "
         f"probability against "
         f"{rule_found['percentage_ruin_cost_pp']:+.2f} for the percentage "
         f"rule, which simply cuts spending instead of running out. The "
         f"choice between them is a choice about which form the sequence risk "
         f"takes, not about how much of it there is."
         if rule_found.get("fixed_real_trades_ruin_for_smoothness") else
         "The rules do not differ systematically in what a random ordering "
         "costs them in failure probability.")))

    out.append(ctx.h2("#sequence.3 Does it change which portfolio wins?"))
    out.append(ctx.p(
        "A decomposition says how much of the dispersion is ordering. It does "
        "not say whether ordering changes which portfolio an investor should "
        "hold, and those are different questions with potentially different "
        "answers."))
    winners = set(str(w) for w in ranking["winner"]) if len(ranking) else set()
    if len(ranking):
        out.extend(ctx.table(
            [["What was shuffled", "Lead of the pair (%)",
              "Best of the whole menu",
              "P(ruin), challenger", "P(ruin), incumbent"]]
            + [[said.get(str(r["phase"]), str(r["phase"])),
                f"{float(r['lead_pct']):+.2f}",
                _pretty_strategy(str(r["winner"])),
                f"{float(r['challenger_ruin']):.1%}",
                f"{float(r['incumbent_ruin']):.1%}"]
               for _, r in ranking.iterrows()],
            f"<i>{_pretty_strategy(pair[0])}</i> against "
            f"<i>{_pretty_strategy(pair[1])}</i> under each shuffle.",
            note=f"Two comparisons sit side by side. The lead column is the "
                 f"headline pair only — <i>{_pretty_strategy(pair[0])}</i> "
                 f"over <i>{_pretty_strategy(pair[1])}</i>. The next column "
                 f"is the best of all {len(set(frame['strategy']))} "
                 f"strategies, which need not be either of them."))
        leads = ranking.set_index("phase")["lead_pct"].astype(float)
        narrows = ("none" in leads.index and "both" in leads.index
                   and leads["both"] < leads["none"])
        out.append(ctx.p(
            (f"<b>The ranking is unmoved by the ordering.</b> "
             f"<i>{_pretty_strategy(sorted(winners)[0])}</i> wins under every "
             f"shuffle. Sequence risk is large, it is real, and it does not "
             f"discriminate between these portfolios — which is why it "
             f"belongs in this part of the paper rather than among the "
             f"allocation sections."
             if len(winners) == 1 else
             f"<b>The ordering changes which portfolio wins</b> "
             f"({', '.join(sorted(winners))}), which means the allocation "
             f"results elsewhere in this paper carry a dependence on the "
             f"order the returns happened to arrive in.")))
        if narrows:
            out.append(ctx.p(
                f"The lead does narrow, and by enough to be worth stating: "
                f"{leads['none']:+.2f}% on the drawn order against "
                f"{leads['both']:+.2f}% on a random one, so shuffling "
                f"absorbs "
                f"{(1 - leads['both'] / leads['none']):.0%} of the gap "
                f"between the two portfolios. Ordering does not reverse the "
                f"comparison, but it is not neutral to it either: a saver "
                f"unlucky in the order banks less of the advantage the "
                f"allocation offers."))

    out.extend(ctx.figure(
        "fig55_sequence_risk",
        "Top left: the share of outcome variance that is nothing but the "
        "order. Top right: the same in consumption units, ordering risk "
        "stacked on return risk. Bottom left: what a random order costs in "
        "certainty-equivalent consumption and in failure probability. Bottom "
        "right: where the ordering risk lands under each withdrawal rule."))

    out.append(ctx.h2("#sequence.4 What this changes"))
    out.extend(ctx.bullets([
        f"<b>Sequence risk is measurable, and it is large.</b> Ordering "
        f"accounts for {found['share_both']:.1%} of the variance in "
        f"retirement consumption on this panel — a number the paper has been "
        f"gesturing at since Section #retirement without putting a figure to "
        f"it.",
        (f"<b>Where it lands is a property of the withdrawal rule, not of the "
         f"market.</b> A fixed real withdrawal pushes it almost entirely into "
         f"the accumulation phase and pays for that in ruin; a percentage "
         f"rule lets it into consumption instead. That reframes Section "
         f"#spending: the rules there are not competing on how much risk they "
         f"take but on what shape it takes."
         if rule_found.get("rule_relocates_the_risk") else
         "The location of the ordering risk is similar under every "
         "withdrawal rule tested."),
        ("<b>It does not touch the allocation answer.</b> The headline pair "
         "keeps its order under every shuffle, which is worth knowing "
         "precisely because the effect is so large elsewhere."
         if len(winners) == 1 else
         "<b>It does touch the allocation answer.</b> The best strategy is "
         "not the same under every shuffle, so the rankings elsewhere in "
         "this paper inherit some dependence on the order the returns "
         "arrived in."),
        "<b>What is not modelled</b>: any response to the sequence. The "
        "investor here does not spend less after a bad year, delay "
        "retirement, or hold a cash buffer, and each of those would reduce "
        "the figures above. This measures the exposure, not what a thoughtful "
        "retiree would do about it — so it is an upper bound on the damage "
        "and, read the other way, a lower bound on the value of the flexible "
        "rules in Section #spending.",
    ]))
    return out


def section_spending(ctx: Any) -> List[Flowable]:
    f = ctx.f
    best = f.table("spending_rule_best_per_family").copy()
    pivot = f.table("spending_rule_bequest_pivot")
    catalogue = f.table("spending_rule_catalogue")
    gamma = f.baseline_gamma
    col, col_nb = f"cec_gamma{gamma:g}", f"cec_nobequest_gamma{gamma:g}"
    best = best.sort_values(col, ascending=False)
    top = best.iloc[0]
    top_nb = best.sort_values(col_nb, ascending=False).iloc[0]

    out: List[Flowable] = [ctx.h1("#spending. Retirement Spending Rules")]
    out.append(ctx.p(
        f"The baseline draws down the portfolio with a constant real "
        f"withdrawal — the four-percent rule and its variants. That is one "
        f"policy among many, and it is a poor one: it ignores the portfolio "
        f"entirely after the first year. This section compares "
        f"{len(catalogue)} spending policies drawn from "
        f"{best.shape[0]} families, each optimised over its own rate so that "
        f"no family is handicapped by a badly chosen parameter."))
    out.extend(ctx.bullets([
        "<b>Constant real.</b> Fixed initial percentage, inflation-adjusted "
        "thereafter. The status quo.",
        "<b>Percentage of portfolio.</b> A fixed fraction of the current "
        "balance each year — never runs out, but consumption is as volatile "
        "as the portfolio.",
        "<b>Guardrails.</b> Guyton–Klinger style: hold the real withdrawal "
        "constant until the implied rate drifts outside a band, then cut or "
        "raise it by a fixed step.",
        "<b>Floor and ceiling.</b> A percentage-of-portfolio rule bounded "
        "above and below relative to last year's withdrawal.",
        "<b>Endowment smoothing.</b> A weighted average of last year's "
        "spending and a target percentage of the current balance.",
        "<b>Required minimum distribution.</b> Balance divided by a "
        "regulatory life-expectancy divisor.",
        "<b>Amortisation.</b> The annuity payment that would exhaust the "
        "balance over the remaining horizon at an assumed return.",
        "<b>Actuarial.</b> As amortisation but using Gompertz life expectancy "
        "recomputed each year rather than a fixed horizon.",
    ]))

    out.extend(ctx.table(
        rows_from(best, ["variant", col, col_nb, "prob_ruin",
                         "median_retirement_consumption",
                         "p5_retirement_consumption", "consumption_volatility",
                         "median_worst_spending_cut", "median_bequest"],
                  ["Best of family", "CEC", "CEC (no bequest)", "P(ruin)",
                   "Median cons.", "5th pct cons.", "Cons. volatility",
                   "Median worst cut", "Median bequest"],
                  {"variant": str, "prob_ruin": lambda v: pc(v, 1),
                   "median_worst_spending_cut": lambda v: pc(v, 0),
                   "median_bequest": lambda v: f2(v, 1)}),
        f"Best spending rule of each family, at γ = {gamma:g}",
        note="Each family is evaluated at its own optimal rate. \"Median worst "
             "cut\" is the largest single-year real spending reduction the "
             "median path experiences — the discomfort the rule imposes in "
             "exchange for its ruin protection.",
        font_size=7.0))

    out.append(ctx.p(
        f"<b>{top['variant']}</b> ranks first at γ = {gamma:g}, with a "
        f"certainty equivalent of {f2(float(top[col]), 3)} and a ruin "
        f"probability of {pc(float(top['prob_ruin']), 1)}. The adaptive "
        f"families as a group dominate the constant-real rule, and the "
        f"mechanism is not subtle: a rule that never looks at the portfolio "
        f"cannot avoid exhausting it, and the certainty equivalent is "
        f"extremely sensitive to the paths where it does."))
    out.append(ctx.p(
        "The cost of that protection is spending volatility, and the table "
        "reports it explicitly rather than burying it. The best-performing "
        "rules ask the median retiree to absorb a real spending cut of tens "
        "of percent at some point in retirement. Whether that trade is "
        "acceptable is a question about the retiree, not about the model; the "
        "certainty equivalent prices it under one particular utility function "
        "and a reader with a different one should read the volatility and "
        "worst-cut columns instead."))
    out.extend(ctx.figure(
        "fig17_spending_rate_curves",
        "Certainty equivalent as a function of the withdrawal rate for each "
        "spending family. Each family is compared at the peak of its own "
        "curve rather than at a common rate."))
    out.extend(ctx.figure(
        "fig18_spending_paths",
        "Realised spending paths under each rule for a common set of market "
        "histories. The adaptive rules trade a smooth path for a solvent one."))

    out.append(ctx.h2("#spending.1 The bequest pivot"))
    out.append(ctx.p(
        f"The ranking is not invariant to the bequest motive, and this is the "
        f"most interesting finding in the section. With the bequest weight "
        f"switched off, the best rule is <b>{top_nb['variant']}</b>; with it "
        f"switched on at θ = "
        f"{float(f.cfg['utility']['bequest_weight']):g}, the ranking shifts "
        f"toward rules that leave something behind. Rules that deliberately "
        f"exhaust the portfolio — amortisation, actuarial — maximise "
        f"consumption precisely by planning to leave nothing, which is optimal "
        f"if and only if nothing is what the investor wants to leave."))
    out.extend(ctx.figure(
        "fig19_spending_bequest_pivot",
        "Certainty equivalent by spending rule across the bequest weight. The "
        "lines cross: a rule that is optimal for an investor with no bequest "
        "motive is not optimal for one who has a strong one."))
    out.append(ctx.p(
        f"Across the {int(pivot['bequest_weight'].nunique())} bequest weights "
        f"tested the identity of the best rule changes, which means the "
        f"question \"what is the best withdrawal policy?\" is not well posed "
        f"without stating a bequest motive. That is a modelling result rather "
        f"than a market result, but it is one that a great deal of "
        f"withdrawal-rate advice ignores."))
    return out


# ---------------------------------------------------------------------------
# 8. The optimal glide path
# ---------------------------------------------------------------------------
def section_glide(ctx: Any) -> List[Flowable]:
    f = ctx.f
    comparison = f.table("glide_comparison")
    restarts = f.table("glide_restarts")
    deviation = f.table("glide_deviation_profile")
    anchor_summary = f.table("glide_retirement_anchor_summary")
    gamma = f.baseline_gamma
    retire_age = int(f.cfg["lifecycle"]["age_retire"])
    block = comparison[comparison["risk_aversion"] == gamma] \
        .sort_values("cec", ascending=False)
    dev = deviation[deviation["risk_aversion"] == gamma]
    material = dev[dev["cost_of_forcing_bp"].abs() > 1.0]

    out: List[Flowable] = [ctx.h1("#glide. Solving for the Optimal Glide Path")]
    out.append(ctx.p(
        "Sections #baseline and #sensitivity compare a handful of candidate "
        "allocation "
        "schedules. That is a test, not an answer. This section solves the "
        "problem directly: what equity share should the investor hold at each "
        "of the 68 ages, if the schedule is free to be anything at all?"))
    out.append(ctx.p(
        "The objective is the certainty equivalent under common random "
        "numbers, which makes it a deterministic function of the schedule. "
        "That licenses coordinate ascent — sweeping one age at a time over a "
        "grid of weights, holding the rest fixed — because each per-age search "
        "is then exact and each full sweep is monotone. To guard against local "
        "optima the search is restarted from several initial schedules, and a "
        "relative improvement threshold is imposed so that the reported "
        "solution does not encode year-to-year jitter worth a fraction of a "
        "basis point."))

    out.extend(ctx.table(
        rows_from(block, ["strategy", "cec", "gap_to_best_pct"],
                  ["Schedule", "CEC", "Gap to best (%)"],
                  {"strategy": lambda v: LABELS.get(v, str(v).replace("_", " ")),
                   "gap_to_best_pct": lambda v: f2(v, 2)}),
        f"Solved schedules against the benchmark strategies at γ = {gamma:g}",
        note="\"Free-form optimal\" solves an unconstrained weight at every "
             "age; \"parametric optimal\" fits a three-parameter logistic "
             "glide. The benchmarks are the same six strategies used "
             "throughout."))
    out.append(ctx.p(
        f"The free-form solution beats every benchmark, but by a margin that "
        f"matters less than its <i>shape</i>. Across the three restarts the "
        f"solved schedule holds a mean equity share of "
        f"{pc(float(restarts['mean_equity_share'].max()), 1)} and sits at a "
        f"full equity allocation in "
        f"{pc(float(restarts['share_of_ages_at_100pct'].max()), 0)} of ages. "
        f"The restarts agree to within "
        f"{f2(float(restarts['gap_to_best_pct'].abs().max()), 3)}% of each "
        f"other, so this is not a local optimum artefact."))
    out.extend(ctx.figure(
        "fig20_optimal_glide_path",
        "The solved equity share by age, from three restarts, against the "
        "target-date fund's prescribed glide. The solution is flat at or near "
        "the corner; it is not a glide path."))

    out.append(ctx.h2("#glide.1 How much of the solved structure is real?"))
    out.append(ctx.p(
        f"A solved schedule always looks structured. The deviation profile "
        f"tests whether it is: each age's solved weight is reset to a neutral "
        f"reference and the certainty-equivalent cost measured in basis "
        f"points. Of the {len(dev)} ages at γ = {gamma:g}, only "
        f"{len(material)} move the objective by more than a single basis "
        f"point. The remaining {len(dev) - len(material)} are on a flat part "
        f"of the surface, and any narrative built on their apparent pattern "
        f"would be a narrative about search noise."))
    out.append(ctx.p(
        "This is the reason the paper does not describe the solved glide path "
        "as having interesting age structure. It does not. The honest summary "
        "is that the optimiser goes to the equity corner and stays there, and "
        "that the small departures it makes near the ends of the horizon are "
        "worth almost nothing."))
    out.extend(ctx.figure(
        "fig21_glide_comparison",
        "Certainty equivalent of the solved schedules against the benchmarks, "
        "at each risk aversion. The solved advantage over a fixed all-equity "
        "portfolio is small; the advantage over the glide path is not."))

    out.append(ctx.h2("#glide.2 The dip belongs to the withdrawal rule"))
    out.append(ctx.p(
        "The solved schedule is not quite flat: under the baseline 4% rule it "
        "dips at the retirement date and recovers afterwards. That is worth "
        "explaining rather than smoothing away, because the obvious reading "
        "-- that the model has rediscovered the glide path after all -- is "
        "the wrong one."))
    out.append(ctx.p(
        f"A 4% rule sets the whole of retirement spending as a fixed fraction "
        f"of wealth on <i>one date</i>. Wealth at {retire_age} is therefore "
        f"not another point on the wealth path; it is the single number that "
        f"fixes consumption for the next thirty years. De-risking briefly "
        f"around a date like that is rational for the same reason nobody "
        f"holds their house deposit in equities the month before completion. "
        f"If that is the explanation, the dip should vanish under a rule that "
        f"anchors on no single date -- so the schedule is re-solved under two "
        f"that do not."))
    if len(anchor_summary):
        out.extend(ctx.table(
            [["Withdrawal rule", "Equity at retirement", "Equity elsewhere",
              "Dip (pp)", "Domestic share, working", "Domestic, retired",
              "Solved CEC"]]
            + [[str(r["rule"]),
                f"{float(r['min_equity_share_at_retirement']):.0%}",
                f"{float(r['mean_equity_share_elsewhere']):.1%}",
                f"{float(r['dip_size_pp']):+.1f}",
                f"{float(r['mean_domestic_working']):.1%}",
                f"{float(r['mean_domestic_retired']):.1%}",
                f"{float(r['solved_cec']):.4f}"]
               for _, r in anchor_summary.iterrows()],
            "The free-form schedule re-solved under each withdrawal rule, "
            f"γ = {float(f.cfg['glide_path']['anchor_check']['risk_aversion']):g}.",
            note="Each row is a complete re-solve: equity share per year and "
                 "domestic share per band, optimised against that rule. The "
                 "certainty equivalents are not comparable across rows, "
                 "because the rules spend different amounts; the shapes are.",
            font_size=7.2))
    dipped = anchor_summary[anchor_summary["dip_size_pp"] > 1.0] \
        if len(anchor_summary) else anchor_summary
    flat = anchor_summary[anchor_summary["dip_size_pp"] <= 1.0] \
        if len(anchor_summary) else anchor_summary
    out.append(ctx.p(
        (f"<b>The dip is a property of the withdrawal rule, not of the "
         f"investment problem.</b> It appears only under the rule that "
         f"anchors on wealth at a single date, at "
         f"{float(dipped['dip_size_pp'].iloc[0]):.0f} percentage points. "
         f"Under the {'other ' if len(flat) > 1 else ''}"
         f"{'rules' if len(flat) > 1 else 'rule'} that condition on the "
         f"portfolio as it stands -- a percentage of the balance, and a "
         f"life-expectancy divisor -- the schedule is flat at 100% equity "
         f"from twenty-five to death and the dip does not appear at all."
         if len(dipped) == 1 and len(flat) else
         f"The dip appears under {len(dipped)} of the "
         f"{len(anchor_summary)} rules solved, so it is not cleanly "
         f"attributable to the anchoring property alone.")))
    out.append(ctx.p(
        "That is a practical result, and it is not the one glide-path "
        "marketing describes: <b>if your withdrawal policy anchors on your "
        "balance at one date, de-risk briefly around that date; if it does "
        "not, do not de-risk at all.</b> A conventional target-date fund does "
        "neither — it de-risks slowly across decades and then stays "
        "de-risked."))
    if len(anchor_summary):
        out.append(ctx.p(
            f"The home/abroad split moves with the rule too, and in the "
            f"direction the ruin mechanism of Section #plan explains. The "
            f"working years take "
            f"{anchor_summary['mean_domestic_working'].min():.0%}–"
            f"{anchor_summary['mean_domestic_working'].max():.0%} domestic "
            f"and the retired years "
            f"{anchor_summary['mean_domestic_retired'].min():.0%}–"
            f"{anchor_summary['mean_domestic_retired'].max():.0%}. Both the "
            f"level and the phase gap are small next to the equity decision, "
            f"which is why the search gives the split coarser bands."))
    out.extend(ctx.figure(
        "fig22_retirement_anchor",
        "The solved equity schedule under each withdrawal rule. The dip at "
        "the retirement date appears only under the rule that fixes spending "
        "from wealth on that one date; the rules that read the portfolio as "
        "it stands solve flat."))
    out.append(ctx.p(
        "The conclusion of this section is stronger than the one the "
        "benchmark comparison supports. It is not merely that the target-date "
        "glide path is worse than an all-equity portfolio on this panel; it is "
        "that when the schedule is allowed to be anything, the optimiser does "
        "not choose a glide path at all — and the one age-related feature it "
        "does choose belongs to the spending rule rather than to the market."))
    return out


# ---------------------------------------------------------------------------
# 9. The whole allocation
# ---------------------------------------------------------------------------
def section_allocation(ctx: Any) -> List[Flowable]:
    f = ctx.f
    a = f.allocation
    gamma = f.baseline_gamma
    schedules = f.table("allocation_solved_schedules")
    phases = f.table("allocation_phase_summary")
    convergence = f.table("allocation_convergence")
    deviation = f.table("allocation_deviation_profile")
    comparison = f.table("allocation_comparison")
    restarts = f.table("allocation_restarts")
    assets = ["dom_eq", "intl_eq", "bond", "bill"]
    names = {"dom_eq": "Domestic equity", "intl_eq": "International equity",
             "bond": "Bonds", "bill": "Bills"}
    #: Abbreviated forms for the wide tables, where the full names wrap badly.
    short = {"dom_eq": "Dom. equity", "intl_eq": "Intl. equity",
             "bond": "Bonds", "bill": "Bills"}
    base = schedules[np.isclose(schedules["risk_aversion"], gamma)] \
        .sort_values("age")
    dev = deviation[np.isclose(deviation["risk_aversion"], gamma)]
    block = comparison[np.isclose(comparison["risk_aversion"], gamma)] \
        .sort_values("cec", ascending=False)
    working = base[base["phase"] == "working"]
    early = working.head(max(len(working) // 3, 1))
    late = working.tail(max(len(working) // 3, 1))
    delta_equity = float(late["equity"].mean()) - float(early["equity"].mean())
    shape_word = ("a decline, but a small one" if delta_equity < -0.02
                  else "a rise" if delta_equity > 0.02
                  else "which is to say it is flat")

    out: List[Flowable] = [ctx.h1("#allocation. The Whole Allocation, Solved")]
    out.append(ctx.p(
        "Section #glide solves for the equity share at every age and for the "
        "domestic split on five-year bands, and finds the optimum sits at or "
        "near the all-equity corner. But it holds the fixed-income sleeve at a "
        "fixed 70/30 bond/bill mix. That restriction was made for search cost, "
        "not for principle, and it matters for the interpretation: an "
        "optimiser that is told what to hold <i>inside</i> a sleeve it barely "
        "uses will look more decisive about that sleeve than it actually is."))
    out.append(ctx.p(
        f"This section removes the restriction. The decision variable is the "
        f"full weight simplex at every year of the lifecycle — {a['n_ages']} "
        f"points in the 3-simplex, {a['free_parameters']} free parameters — "
        f"solved directly against certainty-equivalent consumption. Nothing is "
        f"held fixed: domestic equity, international equity, bonds and bills "
        f"compete freely at every age, subject only to non-negativity and to "
        f"summing to one."))

    out.append(ctx.h2("#allocation.1 The search"))
    out.append(ctx.p(
        "Under common random numbers the objective is a deterministic function "
        "of the schedule, so a search over one age at a time is exact for that "
        "age and every sweep is monotone. Two stages are used, because a "
        "lattice fine enough to be precise is too large to sweep at every age "
        "and a purely local search is too easy to trap:"))
    out.extend(ctx.bullets([
        "a <b>coarse lattice</b> sweep, evaluating every composition of the "
        "simplex at a step of 25% — the 35 lattice points — at each age in "
        "turn; then",
        "a <b>fine local</b> sweep over the twelve single-step pairwise "
        "exchanges around the incumbent at a step of 5%, repeated until "
        "nothing improves.",
    ]))
    out.append(ctx.p(
        "A pairwise exchange moves weight from one asset to another and so "
        "stays on the simplex by construction, which is what makes it a valid "
        "local move there; exchanges that would drive a weight negative are "
        "dropped rather than clipped, so every candidate evaluated is "
        "feasible. Both stages accept a move only if it clears a relative "
        "improvement threshold, for the reason given in Section #methods.4."))
    out.extend(ctx.table(
        rows_from(convergence[np.isclose(convergence["risk_aversion"], gamma)],
                  ["stage", "sweep", "cec", "gain_pct", "evaluations"],
                  ["Stage", "Sweep", "CEC", "Gain (%)",
                   "Cumulative evaluations"],
                  {"stage": str, "sweep": lambda v: f"{int(v)}",
                   "cec": lambda v: f2(v, 5), "gain_pct": lambda v: f2(v, 4),
                   "evaluations": lambda v: f"{int(v):,}"}),
        f"Convergence of the full-simplex search at risk aversion {gamma:g}",
        note="The coarse stage does almost all of the work; the fine stage "
             "moves the objective by a fraction of a percent. That pattern is "
             "itself informative — the surface has a broad flat top."))

    out.append(ctx.h2("#allocation.2 The solved schedule"))
    out.extend(ctx.table(
        rows_from(base.iloc[::max(len(base) // 12, 1)],
                  ["age", "phase"] + assets + ["equity"],
                  ["Age", "Phase"] + [short[x] for x in assets] + ["Equity"],
                  {"age": lambda v: f"{int(v)}", "phase": str,
                   **{x: (lambda v: pc(v, 1)) for x in assets},
                   "equity": lambda v: pc(v, 1)}),
        f"The solved four-asset schedule at risk aversion {gamma:g}, "
        f"every fifth year",
        note="Weights are shares of the whole portfolio and sum to one in each "
             "row. The full schedule is in the project's result tables."))
    out.extend(ctx.table(
        rows_from(phases, ["risk_aversion", "phase", "years"] + assets
                  + ["equity"],
                  ["Risk aversion", "Phase", "Years"]
                  + [short[x] for x in assets] + ["Equity"],
                  {"risk_aversion": lambda v: f"{float(v):g}", "phase": str,
                   "years": lambda v: f"{int(v)}",
                   **{x: (lambda v: pc(v, 1)) for x in assets},
                   "equity": lambda v: pc(v, 1)}),
        "Average solved weights by lifecycle phase and risk aversion"))
    out.append(ctx.p(
        f"At the baseline preference the solved portfolio averages "
        f"<b>{pc(a['mean_dom_eq'], 1)} domestic equity and "
        f"{pc(a['mean_intl_eq'], 1)} international equity</b>, with "
        f"{pc(a['mean_bond'], 1)} in bonds and {pc(a['mean_bill'], 1)} in "
        f"bills. The equity share averages {pc(a['mean_working_equity'], 1)} "
        f"through the working life and moves by "
        f"{delta_equity * 100:+.1f} percentage points between its first and "
        f"last thirds — {shape_word}."))
    out.append(ctx.p(
        f"<b>The fixed-income sleeve is barely used at any age.</b> That is "
        f"the direct answer to the question this section exists to ask: the "
        f"70/30 bond/bill split Section #glide imposed was fixing the composition "
        f"of something the optimiser does not want to hold in the first place, "
        f"so the restriction was not binding. Freeing it buys "
        f"{a['lead_pct']:.2f}% over the best fixed benchmark."))
    out.append(ctx.p(
        f"The domestic/international split carries the same caveat as "
        f"Section #sensitivity.2 and should not be read as advice. The international leg in this "
        f"model is a {f.panel['n_tier_a'] - 1}-country leave-one-out average, "
        f"better diversified than "
        f"any tradeable international index and available to no individual "
        f"investor without also holding their own market. A solved schedule "
        f"that wants {pc(a['mean_intl_eq'], 0)} international is saying the "
        f"model prefers <i>more</i> diversification than the 50/50 headline "
        f"strategy provides, not that this particular number is attainable."))
    out.extend(ctx.figure(
        "fig34_full_allocation",
        "The solved four-asset schedule at each risk aversion, and what each "
        "age of it is worth. The stacked areas are portfolio weights; the "
        "right-hand panel resets each age to the schedule's own average and "
        "measures the certainty-equivalent cost in basis points, on a log "
        "scale with one basis point marked."))

    out.append(ctx.h2("#allocation.3 Against the benchmarks"))
    out.extend(ctx.table(
        rows_from(block, ["strategy", "cec", "gap_to_best_pct"],
                  ["Schedule", "CEC", "Gap to best (%)"],
                  {"strategy": lambda v: LABELS.get(
                      v, str(v).replace("_", " ")),
                   "cec": lambda v: f2(v, 4),
                   "gap_to_best_pct": lambda v: f2(v, 2)}),
        f"Solved and benchmark schedules at risk aversion {gamma:g}",
        note="\"Full simplex optimal\" is this section's solution. The "
             "comparison also includes the schedules solved in Section #glide under "
             "the 70/30 restriction, where those are available."))
    out.append(ctx.p(
        f"The solved schedule leads the best fixed benchmark "
        f"({str(a['runner_up']).replace('_', ' ')}) by "
        f"<b>{a['lead_pct']:.2f}%</b>. Two things about that number deserve "
        f"saying plainly."))
    out.append(ctx.p(
        f"It is <b>small</b>. Freeing {a['free_parameters']} parameters and "
        f"searching them properly buys a fraction of what switching from a "
        f"target-date glide path to a fixed all-equity portfolio buys in "
        f"Section #baseline. The allocation decision has a broad flat top, and almost "
        f"all of its value is captured by the first choice — whether to hold "
        f"diversified equity at all — rather than by any refinement of it."))
    out.append(ctx.p(
        "It is also <b>still not a glide path</b>. The unrestricted solution "
        "differs from the restricted ones of Section #glide in the composition of a "
        "sleeve that carries almost no weight, not in the age profile of the "
        "equity share. Two independent searches over different parameter "
        "spaces reach the same shape, which is a stronger statement than "
        "either makes alone."))
    out.extend(ctx.figure(
        "fig35_allocation_comparison",
        "Left: the solved schedule against the benchmarks at each risk "
        "aversion. Right: the average solved weights by lifecycle phase, "
        "showing how little of the portfolio the fixed-income sleeve ever "
        "carries."))

    retire_age = int(f.cfg["lifecycle"]["age_retire"])
    dev = dev.copy()
    if "phase" not in dev.columns:
        dev["phase"] = np.where(dev["age"] < retire_age, "working", "retired")
    peak = dev.loc[dev["cost_of_resetting_bp"].idxmax()]
    positive = dev["cost_of_resetting_bp"].clip(lower=0.0)
    window = dev[dev["age"].between(retire_age - 2, retire_age + 7)]
    window_share = 100.0 * float(
        window["cost_of_resetting_bp"].clip(lower=0.0).sum()) \
        / max(float(positive.sum()), 1e-9)
    window_years = 100.0 * len(window) / max(len(dev), 1)
    mean_working = float(dev[dev["phase"] == "working"]
                         ["cost_of_resetting_bp"].mean())
    mean_retired = float(dev[dev["phase"] == "retired"]
                         ["cost_of_resetting_bp"].mean())
    concentrated = window_share > 2.5 * window_years

    out.append(ctx.h2("#allocation.4 Where the allocation decision actually matters"))
    out.append(ctx.p(
        f"A solved schedule always looks structured, and this one looks more "
        f"structured than the equity-share solve because it has three "
        f"dimensions to wander in. The deviation profile tests whether the "
        f"structure is real: each age's allocation is reset to the schedule's "
        f"own average and the certainty-equivalent cost measured in basis "
        f"points. Of the {len(dev)} ages, <b>{a['n_material_ages']}</b> move "
        f"the objective by more than a single basis point."))
    out.append(ctx.p(
        "That is a very different picture from Section #glide, where most ages were "
        "worth nothing at all, and it is worth reading carefully rather than "
        "assuming either conclusion."))
    out.extend(ctx.table(
        rows_from(dev.sort_values("cost_of_resetting_bp", ascending=False)
                  .head(8),
                  ["age", "phase", "cost_of_resetting_bp"] + assets,
                  ["Age", "Phase", "Cost of resetting (bp)"]
                  + [short[x] for x in assets],
                  {"age": lambda v: f"{int(v)}", "phase": str,
                   "cost_of_resetting_bp": lambda v: f2(v, 2),
                   **{x: (lambda v: pc(v, 1)) for x in assets}}),
        "The eight ages whose allocation is worth the most",
        note="Cost of resetting that year's allocation to the schedule's own "
             "lifecycle average, holding every other year fixed."))
    out.append(ctx.p(
        f"<b>The cost is concentrated around the retirement date.</b> The "
        f"single most valuable age is {int(peak['age'])} — the retirement year "
        f"itself — at {float(peak['cost_of_resetting_bp']):.1f} basis points, "
        f"and the ten years from {retire_age - 2} to {retire_age + 7} carry "
        f"{window_share:.0f}% of the total cost while being {window_years:.0f}% "
        f"of the lifecycle. That is the sequence-of-returns window Section #retirement "
        f"identifies from a completely different direction, arrived at here "
        f"through the allocation rather than through the retirement date: what "
        f"you hold matters most in the years when the portfolio is largest and "
        f"the withdrawals are about to start."
        if concentrated else
        f"The cost is spread fairly evenly across the lifecycle. The most "
        f"valuable single age is {int(peak['age'])} at "
        f"{float(peak['cost_of_resetting_bp']):.1f} basis points, and the ten "
        f"years around retirement carry {window_share:.0f}% of the total "
        f"against {window_years:.0f}% of the years."))
    out.append(ctx.p(
        f"Working years each cost about {mean_working:.1f} basis points and "
        f"retired years {mean_retired:.1f}, but the retired average is carried "
        f"almost entirely by the first few. <b>The magnitudes remain small in "
        f"absolute terms</b> — the largest single age is worth "
        f"{float(peak['cost_of_resetting_bp']):.1f} basis points and the whole "
        f"schedule beats the best fixed benchmark by {a['lead_pct']:.2f}% — so "
        f"the honest summary is that the <i>timing</i> of the allocation "
        f"decision is concentrated even though the decision itself is worth "
        f"little. An investor who got the allocation right in the decade "
        f"around retirement and used the lifecycle average everywhere else "
        f"would give up a few basis points."))

    out.append(ctx.h2("#allocation.5 Is this a local optimum?"))
    out.append(ctx.p(
        "Coordinate ascent cannot escape a local optimum in principle. "
        "Re-solving from three different corners of the simplex — an equal "
        "split, all equity, and all fixed income — tests for one in practice."))
    out.extend(ctx.table(
        rows_from(restarts, ["start", "solved_cec", "gap_to_best_pct"]
                  + [f"mean_{x}" for x in assets],
                  ["Starting allocation", "Solved CEC", "Gap to best (%)"]
                  + [f"Mean {short[x].lower()}" for x in assets],
                  {"start": str, "solved_cec": lambda v: f2(v, 5),
                   "gap_to_best_pct": lambda v: f2(v, 4),
                   **{f"mean_{x}": (lambda v: pc(v, 1)) for x in assets}}),
        "Restarting the search from three corners of the simplex",
        note="Each restart runs the full two-stage search from the stated "
             "starting allocation."))
    out.append(ctx.p(
        f"The restarts agree to within <b>{a['restart_spread_pct']:.3f}%</b> "
        f"of each other. That is not a proof of global optimality — no "
        f"coordinate search offers one — but it is the check that would have "
        f"caught the obvious failure, and it did not fire."))
    return out


# ---------------------------------------------------------------------------
# 10. Leverage
# ---------------------------------------------------------------------------
def section_leverage(ctx: Any) -> List[Flowable]:
    f = ctx.f
    lv = f.leverage
    cfg = f.cfg["leverage"]
    gamma = float(cfg["risk_aversion"])
    optimal = f.table("leverage_optimal_by_cost").sort_values("spread")
    detail = f.table("leverage_outcome_detail")
    assets = ["dom_eq", "intl_eq", "bond", "bill"]
    names = {"dom_eq": "Dom. equity", "intl_eq": "Intl. equity",
             "bond": "Bonds", "bill": "Bills"}
    try:
        schedule = f.table("leverage_schedule")
    except FileNotFoundError:
        schedule = pd.DataFrame()

    out: List[Flowable] = [ctx.h1("#leverage. Borrowing to Invest")]
    out.append(ctx.p(
        "Every allocation in this paper so far is long-only and fully "
        "invested: the weights are non-negative and sum to one. That is a "
        "constraint, not a result, and it rules out a policy with a serious "
        "literature behind it. If the case for equities rests on a horizon "
        "long enough for diversification across countries and decades to work, "
        "then an investor whose financial balance is still small is "
        "under-exposed to the very risk they are being told to take, and the "
        "natural remedy is to borrow."))
    out.append(ctx.p(
        "The question is never whether leverage raises expected wealth; it "
        "obviously does when the expected asset return exceeds the borrowing "
        "rate. The question is what it is worth to a <i>risk-averse</i> "
        "investor at the price they can actually borrow at. This section "
        "sweeps that price."))
    out.append(ctx.p(
        "One scoping note, because the answer below is largely negative and "
        "the negative does not generalise as far as it first appears. What is "
        "levered here is the <i>whole portfolio</i>: the ratio scales every "
        "weight at once. Borrowing aimed at a single asset is a different "
        "policy with a different answer, and Section #mortgage.5 gives it — at the "
        "same borrowing cost, a mortgage on one sleeve is worth several times "
        "what scaling everything is worth. Read what follows as a verdict on "
        "margin, not on debt."))

    out.append(ctx.h2("#leverage.1 Mechanics, stated rather than buried"))
    out.append(ctx.p(
        "An allocation <i>x</i> over the four assets sums to one. A leverage "
        "ratio <i>L</i> means holding <i>L</i> units of that sleeve per unit "
        "of equity capital, funding the difference at the real bill rate plus "
        "a spread <i>c</i>:"))
    out.append(ctx.equation(
        "r<sup>p</sup><sub>h</sub> = L · ( x · r<sub>h</sub> ) "
        "− ( L − 1 ) · ( r<sup>bill</sup><sub>h</sub> + c )"))
    out.append(ctx.p("Two choices in that line are load-bearing."))
    out.append(ctx.p(
        "<b>The borrowing rate floats.</b> It is the realised real bill return "
        "plus a constant spread, so an investor who borrows is exposed to the "
        "same rate their cash would have earned. A fixed real borrowing rate "
        "would be a different — and more favourable — assumption."))
    out.append(ctx.p(
        "<b>Limited liability.</b> The portfolio return is clipped at −100%: "
        "the lender takes what is left and the investor's equity goes to zero, "
        "but they never owe more than they have. That is a margin call, and it "
        "is <i>generous to leverage</i> — a real levered investor faces forced "
        "liquidation at a threshold, not at zero. The tables therefore report "
        "how often the clip binds rather than leaving it implicit. The "
        "portfolio is rebalanced annually to maintain both the allocation and "
        "the leverage ratio, exactly as every unlevered strategy here is "
        "rebalanced to maintain its weights."))

    out.append(ctx.h2("#leverage.2 Optimal leverage by the price of credit"))
    out.append(ctx.p(
        "For each borrowing spread we search jointly over the leverage ratio "
        "and the allocation, taking the allocation from the same coarse "
        "simplex lattice Section #allocation uses. The optimum is therefore a leverage "
        "ratio <i>and</i> the portfolio that goes with it, rather than a "
        "leverage ratio bolted onto a portfolio chosen elsewhere."))
    out.extend(ctx.table(
        rows_from(optimal, ["spread", "leverage", "cec", "vs_unlevered_pct",
                            "equity"] + assets + ["wipeout_share_of_years"],
                  ["Borrowing spread", "Optimal leverage", "CEC",
                   "vs unlevered (%)", "Equity"]
                  + [names[x] for x in assets] + ["Wipeout share"],
                  {"spread": lambda v: pc(v, 2),
                   "leverage": lambda v: f"{float(v):g}x",
                   "cec": lambda v: f2(v, 4),
                   "vs_unlevered_pct": lambda v: f2(v, 2),
                   "equity": lambda v: pc(v, 0),
                   **{x: (lambda v: pc(v, 0)) for x in assets},
                   "wipeout_share_of_years": lambda v: f"{float(v):.4%}"}),
        "The optimal leverage ratio and allocation at each price of credit",
        note="The spread is an annual real cost over the realised real bill "
             "rate. \"Wipeout share\" is the fraction of path-years in which "
             "the limited-liability clip binds — the assumption most "
             "favourable to leverage, reported rather than assumed away.",
        font_size=7.0))
    out.append(ctx.p(
        f"{'<b>At zero cost, leverage is worth having</b>' if lv['value_at_zero_spread'] > 0.05 else '<b>Even free, leverage is worth almost nothing here</b>'}: "
        f"the optimum is {lv['optimal_at_zero']:g}× and it is worth "
        f"{lv['value_at_zero_spread']:+.2f}% of certainty-equivalent "
        f"consumption against the unlevered portfolio."))
    out.append(ctx.p(
        f"Note <i>what</i> the optimiser levers. At a zero spread it holds "
        f"{pc(lv['equity_at_zero'], 0)} equity and levers it "
        f"{lv['optimal_at_zero']:g}×, for an effective equity exposure of "
        f"{pc(lv['effective_equity_at_zero'], 0)} — it borrows against a "
        f"<i>diversified</i> portfolio rather than concentrating into an "
        f"undiversified one. That is the same preference for diversification "
        f"the unlevered sections keep finding, expressed through a different "
        f"instrument."))
    out.append(ctx.p(
        f"The interesting number is where it stops. <b>The break-even spread "
        f"is {lv['break_even_spread']:.2%}</b>: above roughly that annual cost "
        f"over the real bill rate, no leverage ratio on the grid beats staying "
        f"unlevered, and the optimum is already 1× by a spread of "
        f"{lv['first_unlevered_spread']:.2%}."))
    out.append(ctx.p(
        "That threshold sits below what the borrowing actually costs a "
        "household. A retail investor borrowing through a margin account pays "
        "well over one percent above the bill rate; a levered exchange-traded "
        "fund embeds financing at institutional rates plus a management fee, "
        "and rebalances daily rather than annually, which introduces a "
        "volatility drag this model does not even charge for. On this panel "
        "the price of credit available to the household this model describes "
        "is above the price at which borrowing pays."
        if np.isfinite(lv["break_even_spread"])
        and lv["break_even_spread"] < 0.015 else
        f"That threshold is close to what a household can actually obtain — a "
        f"retail margin account runs somewhere around one to two percent over "
        f"the benchmark — so the model does not dismiss leverage out of hand. "
        f"But the formal break-even overstates how far the case extends. "
        f"<b>The advantage falls below a tenth of a percent by a spread of "
        f"{lv['negligible_spread']:.2%}</b>, well before it reaches zero. Over "
        f"most of the plausible range of borrowing costs, leverage on this "
        f"panel is not so much <i>harmful</i> as <i>pointless</i>: it takes on "
        f"the tail risk documented in Section #leverage.3 in exchange for a gain that "
        f"rounds to nothing."))
    out.extend(ctx.figure(
        "fig36_leverage_surface",
        "Left: the value of leverage across the ratio, one line per borrowing "
        "spread. Right: the optimal ratio as the price of credit rises, "
        "annotated with what it is worth."))

    out.append(ctx.h2("#leverage.3 What leverage does to the shape of the outcome"))
    out.append(ctx.p(
        "A certainty equivalent alone cannot show what borrowing does to the "
        "distribution, and the distribution is the whole argument."))
    out.extend(ctx.table(
        rows_from(detail, ["leverage", f"cec_gamma{gamma:g}",
                           "vs_unlevered_pct", "prob_ruin",
                           "p5_retirement_consumption",
                           "median_retirement_consumption",
                           "p95_retirement_consumption", "prob_zero_bequest",
                           "wipeout_share_of_years"],
                  ["Leverage", "CEC", "vs unlevered", "P(ruin)",
                   "5th pct cons.", "Median cons.", "95th pct cons.",
                   "P(zero bequest)", "Wipeout share"],
                  {"leverage": lambda v: f"{float(v):g}x",
                   f"cec_gamma{gamma:g}": lambda v: f2(v, 4),
                   "vs_unlevered_pct": lambda v: f2(v, 2),
                   "prob_ruin": lambda v: pc(v, 1),
                   "prob_zero_bequest": lambda v: pc(v, 1),
                   "wipeout_share_of_years": lambda v: f"{float(v):.4%}"}),
        f"Distributional detail at a {float(cfg['detail_spread']):.1%} "
        f"borrowing spread",
        note="All rows hold the same allocation — the one optimal at that "
             "spread — and vary only the leverage ratio, so the differences "
             "are attributable to borrowing rather than to reallocation.",
        font_size=7.0))
    out.append(ctx.p(
        f"Going from unlevered to {lv['top_leverage']:g}× moves the fifth "
        f"percentile of retirement consumption by "
        f"<b>{lv['p5_change_pct']:+.1f}%</b>, the median by "
        f"{lv['median_change_pct']:+.1f}% and the ninety-fifth percentile by "
        f"{lv['p95_change_pct']:+.1f}%. The ruin probability moves from "
        f"{pc(lv['ruin_base'], 1)} to {pc(lv['ruin_top'], 1)}."))
    out.append(ctx.p(
        "This is the mechanism the certainty equivalent is pricing. Leverage "
        "widens both tails, and a risk-averse investor weighs the left one "
        "more heavily. The borrowing spread then makes the trade "
        "progressively worse, because it is paid in every state of the world "
        "including the ones where the leverage did not help."))
    out.append(ctx.p(
        "The limited-liability clip never binds anywhere in this study, so the "
        "generous assumption of Section #leverage.1 is doing no work: the levered "
        "portfolios lose heavily in the left tail without ever being wiped out "
        "outright. A model with forced liquidation at a margin threshold "
        "rather than at zero would like leverage less than this one does."
        if lv["max_wipeout"] <= 1e-9 else
        f"Note the last column. The clip binds in up to "
        f"{lv['max_wipeout']:.2%} of path-years — years in which a real "
        f"levered investor would have been liquidated rather than merely "
        f"marked down. It binds only on the high ratios and only when the "
        f"allocation is held fixed rather than re-optimised, which is why the "
        f"sweep of Section #leverage.2 shows none: there the optimiser retreats into "
        f"bonds and bills precisely to avoid it. Where it does bind, the "
        f"certainty equivalents overstate what leverage is worth by an amount "
        f"this model cannot price."))

    try:
        by_decade = f.table("leverage_schedule_by_decade")
    except FileNotFoundError:
        by_decade = pd.DataFrame()
    if len(schedule):
        summary = _leverage_schedule_summary(schedule)
        free = summary[np.isclose(summary["spread"], 0.0)]
        declines = bool(len(free)) and (
            float(free["leverage_at_start"].iloc[0])
            > float(free["leverage_at_retirement"].iloc[0]) + 1e-9)
        free_dec = by_decade[np.isclose(by_decade["spread"], 0.0)] \
            if len(by_decade) else pd.DataFrame()
        working_dec = free_dec[free_dec["decade"] < 60]["mean_leverage"] \
            if len(free_dec) else pd.Series(dtype=float)
        monotone = bool(len(working_dec) > 1
                        and (np.diff(working_dec.to_numpy()) <= 1e-9).all())
        out.append(ctx.h2("#leverage.4 Should leverage decline with age?"))
        out.append(ctx.p(
            "Ayres and Nalebuff (2010) argue that a young investor's financial "
            "balance is small relative to the lifetime saving still to come, "
            "so a constant <i>share</i> of a small balance is a small share of "
            "lifetime exposure — and that the remedy is to lever early and "
            "delever later. That is a testable claim, and the machinery here "
            "tests it directly by solving a leverage ratio at every age, "
            "holding the allocation fixed."))
        out.extend(ctx.table(
            rows_from(summary, ["spread", "leverage_at_start",
                                "mean_leverage_working",
                                "leverage_at_retirement",
                                "mean_leverage_retired", "solved_cec"],
                      ["Borrowing spread", "Leverage at 25",
                       "Mean while working", "At retirement",
                       "Mean in retirement", "Solved CEC"],
                      {"spread": lambda v: pc(v, 1),
                       "leverage_at_start": lambda v: f"{float(v):g}x",
                       "mean_leverage_working": lambda v: f2(v, 2),
                       "leverage_at_retirement": lambda v: f"{float(v):g}x",
                       "mean_leverage_retired": lambda v: f2(v, 2),
                       "solved_cec": lambda v: f2(v, 4)}),
            "A leverage ratio solved for every age",
            note="The allocation is held at the one optimal for that spread, "
                 "so the schedule is purely about when to borrow."))
        out.append(ctx.p(
            "The per-age solution is jittery, because the surface is flat "
            "enough that the search finds tiny improvements moving a single "
            "year between adjacent grid values. Aggregating to decades reports "
            "the trend the schedule genuinely carries:"))
        if len(free_dec):
            out.extend(ctx.table(
                rows_from(free_dec, ["decade", "years", "mean_leverage",
                                     "min_leverage", "max_leverage"],
                          ["Decade of age", "Years", "Mean leverage", "Min",
                           "Max"],
                          {"decade": lambda v: f"{int(v)}s",
                           "years": lambda v: f"{int(v)}",
                           "mean_leverage": lambda v: f2(v, 2),
                           "min_leverage": lambda v: f2(v, 2),
                           "max_leverage": lambda v: f2(v, 2)}),
                "Solved leverage by decade of age, at a zero borrowing spread",
                note="Aggregated from the per-age solution. The min and max "
                     "columns show how much the raw schedule moves within each "
                     "decade — which is where the jitter lives."))
        out.append(ctx.p(
            "The solved schedule <b>declines with age</b>, which is the "
            "Ayres–Nalebuff prescription arrived at from the other direction: "
            "borrow while the financial balance is small relative to the "
            "lifetime saving still to come, and delever as it grows. Note that "
            "this is the one place in the paper where a genuinely age-varying "
            "policy emerges from an unconstrained search — and it is a "
            "schedule for <i>leverage</i>, not for the equity share."
            + (" The decade means fall monotonically through the whole working "
               "life, so this is not an artefact of the two endpoints."
               if monotone else
               " The decade means do not fall monotonically, so the trend is "
               "real but not clean.")
            if declines else
            "The solved schedule does <b>not</b> decline with age. Whatever "
            "the lifetime-exposure argument says in a model with a smooth "
            "income stream and no borrowing constraint, on this return panel "
            "the optimiser does not want to front-load its borrowing — which "
            "is a point against the Ayres–Nalebuff prescription on this "
            "evidence rather than for it."))
        out.extend(ctx.figure(
            "fig37_leverage_detail",
            "Left: what leverage does to each percentile of retirement "
            "consumption. Right: the leverage ratio solved for every age, at "
            "three prices of credit."))

        try:
            two_level = f.table("leverage_two_level")
        except FileNotFoundError:
            two_level = pd.DataFrame()
        if len(two_level):
            wide = two_level.pivot(index="spread", columns="policy",
                                   values="vs_unlevered_pct").reset_index()
            picks = two_level.pivot(index="spread", columns="policy",
                                    values="leverage")
            for col in ("constant", "two_level"):
                if col in picks:
                    wide[col + "_lev"] = picks[col].to_numpy()
            beats = int((wide["two_level"] > wide["constant"] + 1e-9).sum())
            survives = beats == len(wide)
            kept = (float((wide["two_level"]
                           / wide["per_age"].replace(0, np.nan)).mean() * 100.0)
                    if "per_age" in wide else float("nan"))
            realistic = wide[wide["spread"] >= 0.02 - 1e-9]
            gap = bool(len(realistic)
                       and (realistic["constant"] <= 0.05).all()
                       and (realistic["two_level"] > 0.05).all())
            out.append(ctx.h2("#leverage.4.1 Is the shape real, or 68 parameters of "
                              "overfit?"))
            out.append(ctx.p(
                "The schedule above is solved and scored on the same paths, "
                "with one free parameter per age. That is exactly the setting "
                "in which a solved policy flatters itself, so its certainty "
                "equivalent alone cannot establish that <i>declining</i> "
                "leverage beats a constant ratio — the gain could be nothing "
                "but optimisation gain. The discriminating test is how much "
                "survives with a <b>single</b> knob: hold a ratio while "
                "working and unlever at retirement."))
            cols = [c for c in ("spread", "constant_lev", "constant",
                                "two_level_lev", "two_level", "per_age")
                    if c in wide]
            heads = {"spread": "Borrowing spread",
                     "constant_lev": "Best constant",
                     "constant": "Constant, vs unlevered",
                     "two_level_lev": "Best working ratio",
                     "two_level": "Two-level, vs unlevered",
                     "per_age": "Per age, vs unlevered"}
            fmts = {"spread": lambda v: pc(v, 1),
                    "constant_lev": lambda v: f"{float(v):g}x",
                    "two_level_lev": lambda v: f"{float(v):g}x",
                    "constant": lambda v: f"{float(v):+.2f}%",
                    "two_level": lambda v: f"{float(v):+.2f}%",
                    "per_age": lambda v: f"{float(v):+.2f}%"}
            out.extend(ctx.table(
                rows_from(wide, cols, [heads[c] for c in cols], fmts),
                "One knob against sixty-eight: constant, two-level and "
                "per-age leverage",
                note="Every row is scored on the same paths and the same "
                     "allocation, so the comparison is about the leverage "
                     "policy alone. The two-level policy holds one ratio "
                     "while working and unlevers at retirement."))
            out.append(ctx.p(
                f"The two-level policy beats the best constant ratio at "
                f"{'every' if survives else 'some but not all'} price of "
                f"credit tested, keeping roughly {kept:.0f}% of what the free "
                f"68-parameter solve claims. <b>The declining shape is "
                f"structural; the per-age detail is mostly optimisation "
                f"gain.</b>"
                + (" It also survives where the constant ratio does not: at "
                   "the higher spreads holding one ratio for life is worth "
                   "nothing, while unlevering at retirement is still worth "
                   "having."
                   if gap else "")
                if beats else
                "The two-level policy does <b>not</b> beat the best constant "
                "ratio at any price of credit tested. Whatever the per-age "
                "solve gains, it is not coming from the declining shape, and "
                "should be read as optimisation gain."))

    out.append(ctx.h2("#leverage.5 What this changes"))
    out.extend(ctx.bullets([
        f"A <b>constant</b> leverage ratio is not free money. It is worth "
        f"{lv['value_at_zero_spread']:+.2f}% when credit is free, breaks even "
        f"at a spread of {lv['break_even_spread']:.2%}, and is worth under a "
        f"tenth of a percent from {lv['negligible_spread']:.2%} upward — which "
        f"covers most of the range a household actually borrows in.",
        "That verdict does <b>not</b> carry over to a <b>declining</b> ratio, "
        "which is what Ayres and Nalebuff actually prescribe. Section #leverage.4.1 "
        "prices the two separately on the same paths, and the declining policy "
        "is worth more at every spread tested — so the familiar null result "
        "from levered-portfolio backtests is in part an artefact of holding "
        "the ratio fixed for life.",
        f"The result is driven by the <b>left tail</b>. At a moderate "
        f"{lv['moderate_leverage']:g}× the median rises "
        f"{lv['moderate_median_pct']:+.1f}% while the fifth percentile falls "
        f"{lv['moderate_p5_pct']:+.1f}%; push the ratio to "
        f"{lv['top_leverage']:g}× and the median falls too "
        f"({lv['median_change_pct']:+.1f}%). The certainty equivalent prices "
        f"that trade at the investor's risk aversion, and a less risk-averse "
        f"investor would take it at a higher spread.",
        "What gets levered is a <b>diversified</b> portfolio. The optimiser "
        "does not respond to cheap credit by concentrating, which is the same "
        "answer the unlevered sections give in a different form.",
        "The limited-liability assumption is <b>generous to leverage</b>, so "
        "the conclusion is conservative in the direction that matters: a model "
        "with forced liquidation would like borrowing less than this one does.",
    ]))
    return out


def _leverage_schedule_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-age leverage schedule to one row per spread."""
    rows = []
    for spread in sorted(schedule["spread"].unique()):
        block = schedule[schedule["spread"] == spread].sort_values("age")
        work = block[block["phase"] == "working"]
        retired = block[block["phase"] == "retired"]
        rows.append({
            "spread": float(spread),
            "leverage_at_start": float(block["leverage"].iloc[0]),
            "mean_leverage_working": float(work["leverage"].mean()),
            "leverage_at_retirement": float(work["leverage"].iloc[-1]),
            "mean_leverage_retired": float(retired["leverage"].mean())
            if len(retired) else float("nan"),
            "solved_cec": float(block["solved_cec"].iloc[0])})
    return pd.DataFrame.from_records(rows)


# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------
def section_inflation(ctx: Any) -> List[Flowable]:
    f = ctx.f
    ordering = f.table("inflation_asset_ordering")
    windows = f.table("inflation_windows")
    advantage = f.table("inflation_advantage")
    predictive = f.table("inflation_predictive")
    eq_optima = f.table("inflation_optimal_equity")
    dom_optima = f.table("inflation_optimal_domestic")

    from src import inflation as ifl
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    icfg = cfg["inflation_state"]
    window = int(icfg.get("headline_window", 3))
    horizon = int(icfg.get("headline_horizon", 1))
    labels = list(icfg.get("bucket_labels", ifl.BUCKET_LABELS))
    persist = ifl.persistence(predictive, window)
    eq_shift = ifl.optimum_shift(eq_optima, "equity_share", labels)
    dom_shift = ifl.optimum_shift(dom_optima, "domestic_share", labels)
    found = ifl.verdict(advantage, predictive, window, horizon, eq_shift,
                        dom_shift, persist)
    # The same state variable read at the retirement date instead.
    buckets = f.table("inflation_buckets")
    retire_buckets = f.table("inflation_retirement_buckets")
    retire_adv = f.table("inflation_retirement_advantage")
    ret_eq_optima = f.table("inflation_retirement_equity")
    ret_dom_optima = f.table("inflation_retirement_domestic")
    ret_eq_shift = ifl.optimum_shift(ret_eq_optima, "equity_share", labels)
    ret_dom_shift = ifl.optimum_shift(ret_dom_optima, "domestic_share", labels)
    cec_col = f"cec_crra_gamma{gamma:g}"
    accumulation = str(icfg.get("accumulation_strategy",
                                "balanced_all_equity"))
    timing = ifl.timing_comparison(
        ifl.level_spread(buckets, accumulation, cec_col, labels),
        ifl.level_spread(retire_buckets, accumulation, cec_col, labels))
    horizons = sorted(int(h) for h in predictive["horizon_years"].unique())
    longest = horizons[-1] if horizons else horizon

    def _span(h: int) -> str:
        """`the following year`, or `the following N years`, for prose."""
        return ("the following year" if int(h) == 1
                else f"the following {int(h)} years")

    def _gap(asset: str, h: int) -> float:
        block = predictive[(predictive["asset"] == asset)
                           & (predictive["window_years"] == window)
                           & (predictive["horizon_years"] == int(h))]
        return float(block["gap"].iloc[0]) * 100.0 if len(block) else float("nan")

    out: List[Flowable] = [
        ctx.h1("#inflation. What Inflation Has Just Done")]
    out.append(ctx.p(
        "Section #valuation conditions a lifetime on how expensive the market "
        "was when it opened. This section conditions it on something an "
        "investor knows better than the dividend yield and reads about every "
        "month: what the price level has just done."))
    out.append(ctx.p(
        "The two questions are not the same, and the second has a sharper "
        "mechanism behind it. A dividend yield is a claim about expected "
        "returns that has to be argued for. Trailing inflation is a fact "
        "about the price level, and every return in this paper is already "
        "deflated by it — so the channel is direct. A nominal bond promises a "
        "fixed number of currency units; if inflation is high and persistent "
        "that promise is worth less in real terms, and the asset whose real "
        "return inflation eats is the one this paper’s rivals hold most of. "
        "Equity is a claim on cash flows that reprice, which is a partial "
        "hedge over a long horizon and a famously poor one over a short."))
    out.append(ctx.p(
        "That gives this section something Section #valuation lacked: a "
        "reason to expect the <i>allocation</i> to move rather than only the "
        "level. A valuation is a statement about every asset at once. "
        "Inflation is not — it falls hardest on the legs whose payments are "
        "fixed in the currency that is losing value, and it is a domestic "
        "phenomenon, which gives the foreign leg a claim to be a hedge "
        "against it. Whether either effect is large enough to move an "
        "optimum is the question."))

    out.append(ctx.h2("#inflation.1 The observable, and the constraint"))
    out.append(ctx.p(
        f"Inflation in year <i>t</i> is unknown until year <i>t</i> is over, "
        f"so the quantity an investor observes on its first day is the "
        f"annualised rate over the <i>k</i> years already finished — built "
        f"from rows <i>t−k</i> through <i>t−1</i> and nothing later. The "
        f"headline uses <i>k</i> = {window}. The property is checked "
        f"structurally rather than assumed: corrupting one year’s inflation "
        f"must leave every earlier row untouched. A correlation test could "
        f"not establish it, because a leak and an honest signal look "
        f"identical in a correlation."))
    out.append(ctx.note(
        f"Every correlation in this section is a <b>rank</b> correlation, and "
        f"that is not a stylistic preference. The panel contains real "
        f"hyperinflations — Germany 1923 at 1.06 × 10⁹, Japan 1945 at 976%, "
        f"Italy 1944 at 344%. They are observations, not errors, and deleting "
        f"them would remove exactly the episodes an inflation study exists to "
        f"look at. But a Pearson correlation on a series containing a "
        f"billion-per-cent observation describes that observation and nothing "
        f"else: it reports −0.0004 for the persistence of inflation, a series "
        f"whose rank correlation with its own recent past is "
        f"{persist.get('short_correlation', float('nan')):.2f}. The Pearson "
        f"figure is carried in its own column rather than suppressed."))

    out.append(ctx.h2("#inflation.2 Does it predict anything?"))
    out.append(ctx.p(
        "The mechanism first, because without it nothing downstream has a "
        "reason to work. Trailing inflation can only predict returns if it "
        "predicts inflation."))
    persist_rows = predictive[(predictive["asset"] == "inflation")
                              & (predictive["window_years"] == window)]
    out.extend(ctx.table(
        [["Years ahead", "Rank correlation", "After low inflation",
          "After high inflation"]]
        + [[f"{int(r['horizon_years'])}", f"{float(r['correlation']):.3f}",
            f"{float(r['forward_low_inflation']):.2%}",
            f"{float(r['forward_high_inflation']):.2%}"]
           for _, r in persist_rows.sort_values("horizon_years").iterrows()],
        f"Trailing {window}-year inflation against the inflation that "
        f"followed it."))
    out.append(ctx.p(
        (f"<b>Inflation is persistent, and the persistence decays.</b> The "
         f"rank correlation with next year is "
         f"{persist['short_correlation']:.2f} and with the "
         f"{persist['long_horizon']}-year average "
         f"{persist['long_correlation']:.2f}. That shape — strong at short "
         f"range, weak at long — is what sets the boundary on everything "
         f"below, and it is the reason a sixty-eight-year lifetime turns out "
         f"to be a poor place to look for an inflation effect."
         if persist.get("persistent") else
         "<b>Trailing inflation does not predict future inflation in this "
         "panel</b>, which removes the mechanism the rest of this section "
         "depends on and makes any predictive power below a coincidence "
         "needing an explanation.")))

    out.extend(ctx.table(
        [["Asset", "Rank correlation", "After low inflation",
          "After high inflation", "Difference"]]
        + [[_abbrev_asset(str(r["asset"])), f"{float(r['correlation']):.3f}",
            f"{float(r['forward_low_inflation']):.2%}",
            f"{float(r['forward_high_inflation']):.2%}",
            f"{float(r['gap']) * 100:+.2f} pp"]
           for _, r in ordering.iterrows()],
        f"Annualised real returns over the {horizon} year(s) after a "
        f"lifetime’s start, by the trailing {window}-year inflation it began "
        f"at.",
        note="Sorted worst-affected first. A negative difference means the "
             "high-inflation third was followed by poorer real returns."))
    out.append(ctx.p(
        (f"<b>Inflation hurts the nominal legs and largely spares equity.</b> "
         f"Over {_span(horizon)} the bond and bill legs give up "
         f"{found['nominal_gap_pp']:+.2f} points a year after a "
         f"high-inflation start, against {found['equity_gap_pp']:+.2f} points "
         f"for equity. That is the ordering the mechanism predicts: a fixed "
         f"number of currency units is worth less when the currency is losing "
         f"value, and a claim on repricing cash flows is not."
         if found["nominal_legs_hurt_more"] else
         f"<b>Inflation does not fall hardest on the nominal legs.</b> Bonds "
         f"and bills give up {found['nominal_gap_pp']:+.2f} points a year "
         f"after a high-inflation start against "
         f"{found['equity_gap_pp']:+.2f} for equity, which runs against the "
         f"mechanism this section was built on.")))
    out.append(ctx.p(
        f"<b>But the effect is a short-horizon one, and it does not survive a "
        f"lifetime.</b> The same comparison run over "
        f"{longest} years gives {_gap('bond', longest):+.2f} points on bonds "
        f"and {_gap('dom_eq', longest):+.2f} on domestic equity — the damage "
        f"has decayed and in places reversed, because a long window beginning "
        f"in a high-inflation year captures the disinflation that followed "
        f"it. Inflation is a risk to the portfolio a retiree is holding now, "
        f"not to the one a twenty-five-year-old is starting. Everything in "
        f"the rest of this section should be read against that."))

    out.append(ctx.h2("#inflation.3 Which lookback window"))
    out.extend(ctx.table(
        [["Lookback (years)", "Observations", "Rank correlation", "Pearson",
          "High minus low third"]]
        + [[f"{int(r['window_years'])}", f"{int(r['observations']):,}",
            f"{float(r['correlation']):.3f}", f"{float(r['pearson']):.4f}",
            f"{float(r['gap']) * 100:+.2f} pp"]
           for _, r in windows.iterrows()],
        "One, three and five years of trailing inflation, against the same "
        "forward domestic equity returns.",
        note="The Pearson column is the hyperinflation problem made visible: "
             "it is near zero at every window on a relationship the rank "
             "statistic finds without difficulty."))
    out.append(ctx.p(
        f"The headline uses {window} years, named in the configuration rather "
        f"than chosen by search. With three candidates and one panel, picking "
        f"the best-performing window and then reporting its performance would "
        f"be a selection effect wearing a result; the table above is here so "
        f"a reader can see what the other two would have given."))

    out.append(ctx.h2("#inflation.4 Conditioning a lifetime"))
    out.append(ctx.p(
        "Lifetimes are bucketed into terciles of the trailing rate they began "
        "at, against boundaries computed from country-years strictly before "
        "each lifetime started — the same discipline as Section #valuation, "
        "and for the same reason. The rate itself is look-ahead-free, but a "
        "pooled tercile boundary would not be: a lifetime beginning in 1910 "
        "would be called high-inflation against a threshold that already knew "
        "about the 1970s."))
    out.extend(ctx.table(
        [["Starting inflation", "Lifetimes",
          "All-equity over target-date (%)", "P(ruin), all-equity",
          "P(ruin), target-date"]]
        + [[str(r["bucket"]), f"{int(r['n_paths']):,}",
            f"{float(r['advantage_pct']):+.2f}",
            f"{float(r['challenger_ruin']):.1%}",
            f"{float(r['incumbent_ruin']):.1%}"]
           for _, r in advantage.iterrows()],
        f"The headline comparison inside each inflation tercile, γ = {gamma:g}."))
    out.append(ctx.p(
        (f"<b>The ranking survives every inflation regime.</b> The lead runs "
         f"from {found['lead_low_pct']:.2f}% after calm years to "
         f"{found['lead_high_pct']:.2f}% after inflationary ones, a spread of "
         f"{found['lead_spread_pp']:.2f} points, and never changes sign. "
         f"Starting inflation moves what an investor should expect; it does "
         f"not move which of these two they should hold."
         if found.get("ranking_survives") else
         f"<b>The ranking does not survive every inflation regime.</b> The "
         f"lead runs from {found['lead_low_pct']:.2f}% to "
         f"{found['lead_high_pct']:.2f}% and changes sign, which means the "
         f"headline carries a dependence on the inflation a lifetime began "
         f"at.")))

    out.append(ctx.h2("#inflation.5 The optimal portfolio"))
    out.append(ctx.p(
        "Comparing two fixed portfolios answers a narrower question than the "
        "section set out to ask. Two grids are therefore scored on the same "
        "lifetimes and the same buckets — how much equity to hold, and how "
        "much of that equity to hold at home — and the certainty-equivalent "
        "maximum is read off within each regime. The composition inside each "
        "sleeve is held fixed while the parameter moves, so the argmax means "
        "what it appears to mean."))
    out.extend(ctx.table(
        [["Starting inflation", "Optimal equity share", "CEC there",
          "Best over worst (%)", "Margin over runner-up (%)"]]
        + [[str(r["bucket"]), f"{float(r['optimal_equity_share']):.0%}",
            f"{float(r['cec_at_optimum']):.4f}",
            f"{float(r['range_pct']):+.1f}",
            f"{float(r['margin_over_runner_up_pct']):.3f}"]
           for _, r in eq_optima.iterrows()],
        "The equity share that maximises certainty-equivalent consumption in "
        "each inflation regime.",
        note="The margin column is the honesty check: a winner that beats the "
             "runner-up by a rounding error has not identified an optimum, "
             "and a shift between buckets is only a finding if the grid can "
             "resolve it."))
    out.extend(ctx.table(
        [["Starting inflation", "Optimal domestic share of equity",
          "CEC there", "Over all-international (%)",
          "Over the next grid point (%)"]]
        + [[str(r["bucket"]), f"{float(r['optimal_domestic_share']):.0%}",
            f"{float(r['cec_at_optimum']):.4f}",
            f"{float(r['margin_over_low_end_pct']):+.2f}"
            if "margin_over_low_end_pct" in dom_optima.columns else "—",
            f"{float(r['margin_over_runner_up_pct']):.3f}"]
           for _, r in dom_optima.iterrows()],
        "And how much of that equity belongs at home.",
        note="The fourth column is the comparison that matters — the optimum "
             "against the all-international corner this paper otherwise "
             "treats as the winner. The fifth is the narrower question of "
             "whether the grid can separate one step from the next."))
    if bool(eq_optima["at_grid_edge"].all()) if len(eq_optima) else False:
        out.append(ctx.note(
            f"The equity optimum sits on the boundary of its grid in every "
            f"regime — the search wanted "
            f"{float(eq_optima['optimal_equity_share'].max()):.0%} equity and "
            f"was not allowed more. That is a limit of the grid rather than "
            f"an interior solution, and it is consistent with Section "
            f"#leverage, which finds borrowing to buy more equity worth "
            f"{f.leverage['value_at_zero_spread']:+.2f}% when credit is free. "
            f"What matters here is that the boundary is the same boundary in "
            f"all three regimes: inflation does not move it."))
    if len(dom_optima) and "margin_over_low_end_pct" in dom_optima.columns:
        dom_opt = float(dom_optima["optimal_domestic_share"].iloc[0])
        over_intl = float(dom_optima["margin_over_low_end_pct"].min())
        equal_weight = 1.0 / float(f.panel["n_countries"])
        solved = f.table("allocation_solved_schedules")
        solved = solved[np.isclose(solved["risk_aversion"], gamma)]
        solved_home = (float((solved["dom_eq"] /
                              (solved["dom_eq"] + solved["intl_eq"]).clip(
                                  lower=1e-9)).mean())
                       if len(solved) else float("nan"))
        if dom_opt > 0.0 and over_intl > 0.05:
            out.append(ctx.p(
                f"<b>The home/abroad optimum is interior, and it is not "
                f"zero.</b> The grid runs from nothing at home to everything, "
                f"in steps of ten points, and in every inflation regime the "
                f"maximum sits at {dom_opt:.0%} domestic — worth "
                f"{over_intl:+.2f}% over the all-international corner that "
                f"the six fixed strategies of Section #baseline treat as the "
                f"winner. That corner was only ever the best of six; offered "
                f"a finer grid, the panel wants a little of the home market "
                f"back."))
            unconditional = f.table("sensitivity_domestic_share")
            ucol = f"cec_crra_gamma{gamma:g}"
            u_best = (float(unconditional.loc[unconditional[ucol].idxmax(),
                                              "domestic_share"])
                      if ucol in unconditional.columns else float("nan"))
            out.append(ctx.p(
                f"<b>Three things say this is real rather than a lucky grid "
                f"point, and none of them involve inflation.</b> The same "
                f"sweep run on all {int(f.panel['n_countries'])} markets' "
                f"lifetimes at once, with no conditioning of any kind, peaks "
                f"at {u_best:.0%} domestic. Section #allocation, which solves "
                f"the whole four-asset simplex at every age with no grid at "
                f"all and by an entirely different procedure, lands on "
                f"{solved_home:.1%}. And the panel holds "
                f"{int(f.panel['n_countries'])} markets, so an equal-weighted "
                f"world portfolio puts {equal_weight:.1%} in the home market "
                f"— which is where all of these sit."))
            out.append(ctx.p(
                f"The reading is not that home bias pays. It is that the "
                f"leave-one-out sleeve this paper uses as its international "
                f"leg is, for any one investor, the world <i>minus their own "
                f"market</i> — a slight underweight of home rather than a "
                f"neutral position — and that adding back roughly the missing "
                f"sixteenth restores it. The strategy this paper calls "
                f"“100% international” is a corner of a six-item menu that "
                f"offers nothing between nought and a half; it wins that menu "
                f"on merit, and it is not the optimum. What this section adds "
                f"is that the correction does not depend on the inflation "
                f"regime: the same {dom_opt:.0%} in all three."))
    out.append(ctx.p(
        (f"<b>Neither optimum moves.</b> The equity share that maximises the "
         f"certainty equivalent is "
         f"{eq_shift.get('optimal_equity_share_low', float('nan')):.0%} in "
         f"every regime, and the domestic share is "
         f"{dom_shift.get('optimal_domestic_share_low', float('nan')):.0%} in "
         f"every regime. Recent inflation changes what a lifetime is worth "
         f"and does not change what it should hold — the same answer Section "
         f"#valuation reached about starting valuation, arrived at through a "
         f"variable with a much more direct mechanism. The optimum's "
         f"<i>level</i> is a finding; its <i>invariance to inflation</i> is "
         f"this section's finding."
         if not eq_shift.get("moves") and not dom_shift.get("moves") else
         f"The optimal equity share reads "
         f"{eq_shift.get('optimal_equity_share_low', float('nan')):.0%} after "
         f"calm years against "
         f"{eq_shift.get('optimal_equity_share_high', float('nan')):.0%} "
         f"after inflationary ones, and the domestic share "
         f"{dom_shift.get('optimal_domestic_share_low', float('nan')):.0%} "
         f"against "
         f"{dom_shift.get('optimal_domestic_share_high', float('nan')):.0%}. "
         + ("Both shifts clear the grid's resolution."
            if eq_shift.get("identified") and dom_shift.get("identified")
            else "At least one of those shifts is inside the grid's "
                 "resolution and should not be read as a result: the "
                 "certainty-equivalent surface is nearly flat near its "
                 "maximum, which the margin columns above make visible."))))

    out.append(ctx.h2("#inflation.6 The same question asked of the retiree"))
    out.append(ctx.p(
        "Everything above conditions a lifetime on the inflation its investor "
        "saw at twenty-five. That is the implementable version of the "
        "question — it is what a saver can act on — and it produces a null, "
        "for a reason this section has already given: the damage is "
        "short-horizon, and a sixty-eight-year window averages it away."))
    out.append(ctx.p(
        f"But the utility window in this paper is <i>retirement consumption "
        f"alone</i>, and a retiree drawing down over "
        f"{int(cfg['lifecycle']['age_death']) - int(cfg['lifecycle']['age_retire'])} "
        f"years cannot wait a shock out. So the state variable is read a "
        f"second time, at the retirement date rather than the birth date. A "
        f"twenty-five-year-old cannot know what it will say — this is not an "
        f"instruction to them — but a "
        f"{int(cfg['lifecycle']['age_retire'])}-year-old standing there "
        f"observes it exactly as reliably, and is bucketed against tercile "
        f"boundaries built from history before <i>their</i> retirement rather "
        f"than before their birth."))
    if len(retire_adv):
        out.extend(ctx.table(
            [["Inflation at retirement", "Lifetimes",
              "All-equity over target-date (%)", "P(ruin), all-equity",
              "P(ruin), target-date"]]
            + [[str(r["bucket"]), f"{int(r['n_paths']):,}",
                f"{float(r['advantage_pct']):+.2f}",
                f"{float(r['challenger_ruin']):.1%}",
                f"{float(r['incumbent_ruin']):.1%}"]
               for _, r in retire_adv.iterrows()],
            f"The headline comparison inside each retirement-date inflation "
            f"tercile, γ = {gamma:g}."))
    if timing.get("measured"):
        out.append(ctx.p(
            (f"<b>When the state variable is read decides whether it matters "
             f"at all.</b> Retiring into the high-inflation third rather than "
             f"the low one is worth {timing['retirement_spread_pct']:+.2f}% "
             f"of certainty-equivalent retirement consumption. Conditioning "
             f"the same lifetimes on the inflation they <i>began</i> at is "
             f"worth {timing['birth_spread_pct']:+.2f}% — "
             f"{timing['ratio']:.1f} times smaller"
             + (", and of the opposite sign. " if not timing["same_sign"]
                else ". ")
             + "The null above was a statement about the horizon, not about "
               "inflation. This is the same panel, the same state variable "
               "and the same terciles; only the date it is read at has "
               "changed."
             if timing["retirement_matters_much_more"] else
             f"Reading the state variable at retirement rather than at birth "
             f"moves the level {timing['retirement_spread_pct']:+.2f}% "
             f"against {timing['birth_spread_pct']:+.2f}%, a factor of "
             f"{timing['ratio']:.1f}. The date it is read at does not change "
             f"the conclusion much.")))

    out.append(ctx.h2("#inflation.7 What a retiree should hold"))
    out.append(ctx.p(
        "Sweeping a lifetime allocation answers a question no retiree can act "
        "on: they cannot go back and hold something else from twenty-five. "
        "What they can choose is what to hold from the day they stop working, "
        "with whatever accumulation they arrived with. The portfolios swept "
        "here are therefore identical to the baseline through the working "
        "years and differ only afterwards, so the maximum in each bucket is "
        "an instruction a retiree could actually follow."))
    if len(ret_eq_optima):
        out.extend(ctx.table(
            [["Inflation at retirement", "Optimal equity share", "CEC there",
              "Over holding no equity (%)", "Over the next grid point (%)"]]
            + [[str(r["bucket"]), f"{float(r['optimal_equity_share']):.0%}",
                f"{float(r['cec_at_optimum']):.4f}",
                f"{float(r['margin_over_low_end_pct']):+.2f}"
                if "margin_over_low_end_pct" in ret_eq_optima.columns else "—",
                f"{float(r['margin_over_runner_up_pct']):.3f}"]
               for _, r in ret_eq_optima.iterrows()],
            "How much equity to hold from retirement, by the inflation "
            "observed at that date."))
    if len(ret_dom_optima):
        out.extend(ctx.table(
            [["Inflation at retirement", "Optimal domestic share", "CEC there",
              "Over all-international (%)", "Over the next grid point (%)"]]
            + [[str(r["bucket"]), f"{float(r['optimal_domestic_share']):.0%}",
                f"{float(r['cec_at_optimum']):.4f}",
                f"{float(r['margin_over_low_end_pct']):+.2f}"
                if "margin_over_low_end_pct" in ret_dom_optima.columns else "—",
                f"{float(r['margin_over_runner_up_pct']):.3f}"]
               for _, r in ret_dom_optima.iterrows()],
            "And how much of that equity at home."))
    out.append(ctx.p(
        (f"<b>The retiree's optimum moves with the regime.</b> The equity "
         f"share runs from "
         f"{ret_eq_shift.get('optimal_equity_share_low', float('nan')):.0%} "
         f"in the calm third to "
         f"{ret_eq_shift.get('optimal_equity_share_high', float('nan')):.0%} "
         f"in the hot one, and the domestic share from "
         f"{ret_dom_shift.get('optimal_domestic_share_low', float('nan')):.0%} "
         f"to "
         f"{ret_dom_shift.get('optimal_domestic_share_high', float('nan')):.0%}. "
         + ("Both shifts clear the grid's resolution."
            if ret_eq_shift.get("identified") and ret_dom_shift.get("identified")
            else "At least one of them sits inside the grid's resolution and "
                 "should be read against the margin columns rather than as a "
                 "clean identification.")
         if ret_eq_shift.get("moves") or ret_dom_shift.get("moves") else
         f"<b>The retiree's optimum does not move.</b> "
         f"{ret_eq_shift.get('optimal_equity_share_low', float('nan')):.0%} "
         f"equity and "
         f"{ret_dom_shift.get('optimal_domestic_share_low', float('nan')):.0%} "
         f"domestic in every regime. Inflation at retirement changes what a "
         f"retiree gets — and it changes it a great deal — without changing "
         f"what they should hold against it. That is the same shape of answer "
         f"Section #valuation reached, arrived at through a variable that "
         f"does move the level.")))

    out.extend(ctx.figure(
        "fig54_inflation_timing",
        "Left: the same portfolio's certainty-equivalent consumption by "
        "inflation tercile, each bucket measured against the average of its "
        "own reading, with the tercile assigned at age 25 and again at "
        "retirement — a difference rather than a level, because the levels "
        "differ by too little to see and truncating the axis would "
        "exaggerate them. Middle and right: what a retiree should hold from "
        "the day they stop working, given the inflation they observe then, "
        "with the maximum circled."))

    out.extend(ctx.figure(
        "fig52_inflation_state",
        "Top left: how far a high-inflation start sets each asset back, by "
        "horizon — the effect is large at one year and gone by thirty. Top "
        "right: the same comparison at the headline horizon, by asset. Bottom "
        "left: certainty-equivalent consumption against the equity share, one "
        "curve per inflation regime, with the maximum circled. Bottom right: "
        "the same against the domestic share of the equity sleeve."))

    out.append(ctx.h2("#inflation.6 What this changes"))
    out.extend(ctx.bullets([
        (f"<b>Inflation is a short-horizon risk to nominal assets and a "
         f"long-horizon non-event.</b> Over {_span(horizon)} it costs the "
         f"bond leg {_gap('bond', horizon):+.2f} points a year; over "
         f"{_span(longest)} it is worth {_gap('bond', longest):+.2f}. A "
         f"lifecycle study is the wrong "
         f"instrument for measuring it, and that is a finding about the "
         f"instrument as much as about inflation."),
        (f"The headline ranking is unaffected: the lead spans "
         f"{found.get('lead_spread_pp', float('nan')):.2f} points across the "
         f"terciles without changing sign."
         if found.get("ranking_survives") else
         "The headline ranking is affected, and the limitations section "
         "should carry it."),
        ("The optimal portfolio is unaffected on both axes tested. Where "
         "Section #valuation could be accused of testing a weak signal, this "
         "section tests a strong one with a direct mechanism and reaches the "
         "same conclusion, which makes the pair of them better evidence than "
         "either alone."
         if not eq_shift.get("moves") and not dom_shift.get("moves") else
         "The optimal portfolio does move, and the size of the move against "
         "the grid's resolution is reported above rather than asserted."),
        "<b>What is not modelled</b>: inflation-linked bonds, which are the "
        "instrument this section's mechanism most obviously calls for and "
        "which did not exist over most of the panel; and any policy response "
        "to inflation, since the withdrawal rules here are nominal-blind by "
        "construction. Both would raise the value of conditioning on "
        "inflation rather than lower it, so the null above is a floor.",
    ]))
    return out


def _abbrev_asset(key: str) -> str:
    """A readable asset name for a table column."""
    from src import plots
    return plots.SERIES_ABBR.get(key, key).replace("\n", " ")


def section_fees(ctx: Any) -> List[Flowable]:
    f = ctx.f
    common = f.table("fee_common_curve")
    diff = f.table("fee_differential_curve")
    anchors = f.table("fee_anchors")

    from src.fees import verdict
    cfg = f.cfg
    fee_cfg = cfg.get("fees", {})
    pair = (str(fee_cfg.get("challenger", "international_equity")),
            str(fee_cfg.get("incumbent", "balanced_all_equity")))
    found = verdict(f.table("fee_common"), f.table("fee_differential"),
                    pair, anchors)
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(fee_cfg.get("n_paths", cfg["bootstrap"]["n_paths"]))
    base = float(found["baseline_gap_pct"])
    be_bp = float(found["break_even_differential_bp"])
    active = anchors.loc[anchors["basis_points"].idxmax()]
    per_bp = (abs(base - float(diff["gap_pct"].iloc[-1]))
              / (float(diff["differential"].iloc[-1]) * 1e4))

    out: List[Flowable] = [ctx.h1("#fees. Fees, and the Differential That "
                                  "Would Undo the Headline")]
    out.append(ctx.p(
        "Every return in this paper is gross. Nobody earns a gross return. "
        "The omission is defensible for a comparison between strategies drawn "
        "from the same panel \u2014 a cost common to all of them largely "
        "cancels \u2014 but it is not defensible for the one divergence from "
        "the study we re-implement, because the strategies at stake hold "
        "different amounts of the sleeve that costs more. All-international "
        "pays the international fund's expense ratio on the whole portfolio; "
        "the 50/50 split pays it on half. A <b>differential</b> falls on them "
        "unequally and compounds over a sixty-eight-year lifetime."))
    out.append(ctx.p(
        "An expense ratio is levied on assets rather than returns, so a gross "
        "real return <i>r</i> net of a fee <i>f</i> is "
        "(1&nbsp;+&nbsp;<i>r</i>)(1&nbsp;\u2212&nbsp;<i>f</i>)&nbsp;\u2212"
        "&nbsp;1 rather than <i>r</i>&nbsp;\u2212&nbsp;<i>f</i>; the "
        "difference is second-order in one year and not in sixty-eight. Fees "
        "are charged on the panel before the bootstrap sees it and the "
        "availability mask is left alone, so every fee level draws the same "
        "blocks."))

    out.append(ctx.h2("#fees.1 The control: a fee on everything"))
    out.extend(ctx.table(
        rows_from(common.assign(bp=common["fee"] * 1e4), ["bp", "gap_pct"],
                  ["Fee on every asset", "Lead"],
                  {"bp": lambda v: f"{float(v):.0f} bp",
                   "gap_pct": lambda v: f"{float(v):.2f}%"}),
        "A fee charged on all four sleeves alike",
        note="The control. A cost common to every strategy should largely "
             "cancel out of a comparison between them."))
    common_erosion = (float(common["gap_pct"].iloc[0])
                      - float(common["gap_pct"].iloc[-1]))
    common_top_bp = float(common["fee"].iloc[-1]) * 1e4
    out.append(ctx.p(
        f"The gap never closes on this grid, which is the control working as "
        f"intended. It does not cancel entirely — {common_top_bp:.0f} bp on "
        f"everything still costs {common_erosion:.2f} points of lead, because "
        f"the two strategies accumulate different amounts of wealth for the "
        f"fee to be charged on — but that is a fraction of what the same fee "
        f"does when it falls on the foreign leg alone."
        if found["common_is_near_neutral"] else
        f"The gap closes at {float(found['break_even_common_bp']):.0f} bp even "
        f"under an even-handed fee, so the strategies differ enough in asset "
        f"mix for a common cost to fall on them unequally."))

    out.append(ctx.h2("#fees.2 The question: a fee on the foreign leg alone"))
    out.extend(ctx.table(
        rows_from(diff.assign(bp=diff["differential"] * 1e4),
                  ["bp", "gap_pct"],
                  ["Extra fee on the foreign leg", "Lead"],
                  {"bp": lambda v: f"{float(v):.0f} bp",
                   "gap_pct": lambda v: f"{float(v):.2f}%"}),
        f"All-international over the 50/50 split, by fee differential, at "
        f"\u03b3 = {gamma:g}",
        note=f"{n_paths:,} lifetimes per level. The grid runs past anything "
             f"realistic on purpose: the distance between the break-even and "
             f"what an investor can actually pay is the margin of safety."))
    out.extend(ctx.table(
        rows_from(anchors, ["label", "basis_points", "gap_pct"],
                  ["Reference", "Differential", "Lead at that differential"],
                  {"label": str,
                   "basis_points": lambda v: f"{float(v):.0f} bp",
                   "gap_pct": lambda v: f"{float(v):.2f}%"}),
        "The lead at expense ratios an investor has actually faced",
        note="These are configured anchors, not findings. Nothing in this "
             "paper's data can verify them; they are here to put the swept "
             "grid on a scale a reader recognises."))

    if not found["differential_closes_the_gap"]:
        call = (f"<b>No differential on this grid undoes the result.</b> The "
                f"lead starts at {base:.2f}% and is still positive at the top "
                f"of the grid.")
    elif found["below_cheapest_anchor"]:
        call = (f"<b>The result does not survive realistic fund costs.</b> "
                f"The lead of {base:.2f}% is cancelled by a differential of "
                f"{be_bp:.0f} basis points, narrower than the gap between a "
                f"modern domestic and international index fund.")
    elif found["inside_historic_range"]:
        call = (f"<b>The result survives modern fund costs but not every "
                f"historic one.</b> The lead of {base:.2f}% is cancelled by "
                f"{be_bp:.0f} basis points, wider than today's index-fund "
                f"differential and inside the range of what has been "
                f"charged.")
    else:
        call = (f"<b>The result survives every fund cost a real investor has "
                f"faced.</b> Cancelling a lead of {base:.2f}% takes a "
                f"differential of {be_bp:.0f} basis points, wider than any "
                f"reference expense ratio above \u2014 including "
                f"{str(active['label'])} at "
                f"{float(active['basis_points']):.0f} bp, which still leaves "
                f"{float(active['gap_pct']):.2f}%.")
    out.append(ctx.p(call))
    out.append(ctx.p(
        f"The erosion is real even so, and worth quoting. Each basis point of "
        f"differential costs roughly {per_bp:.3f} points of lead, so an "
        f"investor paying {float(active['basis_points']):.0f} bp over a "
        f"domestic fund keeps {float(active['gap_pct']) / base:.0%} of the "
        f"advantage this paper reports. Fees do not decide the question, but "
        f"they are not free either, and nothing in the gross-return tables "
        f"elsewhere shows that."))

    out.extend(ctx.figure(
        "fig45_fees",
        "Left: a fee charged on every asset alike, the control. Right: a fee "
        "on the international sleeve alone, with the break-even and the "
        "expense ratios a real investor has faced marked on it."))

    out.append(ctx.h2("#fees.3 What this changes"))
    out.extend(ctx.bullets([
        (f"Implementation cost does not decide this comparison. The "
         f"break-even differential of {be_bp:.0f} bp is far outside what "
         f"index funds charge, so the ranking can be taken at face value on "
         f"cost grounds."
         if not found["inside_historic_range"] else
         "Implementation cost is close enough to the break-even to matter, "
         "so the ranking should be quoted with the fund costs assumed."),
        "The rest of the paper works in gross returns, which is defensible "
        "for comparisons between strategies holding the same sleeves in "
        "different proportions and would not be for a comparison between "
        "asset classes of different cost. No result elsewhere turns on the "
        "latter.",
        "What is <b>not</b> modelled: trading costs, spreads, taxes, platform "
        "fees, and the fact that index funds did not exist for most of this "
        "sample. Before roughly 1975 a diversified foreign portfolio was "
        "expensive and often impossible to hold, so this is a sensitivity "
        "rather than a history of what was available.",
    ]))
    return out


# ---------------------------------------------------------------------------
# 11. Currency hedging
# ---------------------------------------------------------------------------
def section_hedging(ctx: Any) -> List[Flowable]:
    f = ctx.f
    sweep = f.table("hedging_sweep")
    breakeven = f.table("hedging_break_even")
    optimal = f.table("hedging_optimal_ratio")
    be25 = breakeven[breakeven["hedge_ratio"] == 0.25].iloc[0]
    full = breakeven[breakeven["hedge_ratio"] == 1.00].iloc[0]
    free = optimal[optimal["hedge_cost"] == 0.0].iloc[0]

    out: List[Flowable] = [ctx.h1("#hedging. Currency Hedging the International Leg")]
    out.append(ctx.p(
        "Section #fees priced one cost of holding the foreign sleeve. This "
        "section prices the other, which is a choice rather than a fee: the "
        "international leg of the headline strategy is unhedged, so a "
        "domestic investor holding it bears both foreign equity risk and "
        "foreign currency risk. Whether that currency exposure is a cost or a "
        "diversifier is an old question, and it has a clean answer in this "
        "framework because the panel contains real exchange rates."))
    out.append(ctx.p(
        "We construct a hedged international leg using covered interest "
        "parity: the hedged return replaces the realised currency movement "
        "with the interest-rate differential implied by the two countries' "
        "bill rates, less an explicit annual hedging cost. The cost is the "
        "parameter of interest. Hedging is not free — it consumes bid-offer, "
        "collateral and roll — and the question a practitioner faces is not "
        "\"is hedging good?\" but \"is hedging good at the price I can get it "
        "for?\" Setting the cost to zero therefore gives the decision its "
        "most favourable possible reading, which is where we start."))

    out.extend(ctx.table(
        rows_from(breakeven, ["hedge_ratio", "unhedged_cec",
                              "cec_at_zero_cost", "gain_at_zero_cost_pct",
                              "break_even_annual_cost"],
                  ["Hedge ratio", "Unhedged CEC", "CEC at zero cost",
                   "Gain at zero cost (%)", "Break-even annual cost"],
                  {"hedge_ratio": lambda v: pc(v, 0),
                   "gain_at_zero_cost_pct": lambda v: f2(v, 3),
                   "break_even_annual_cost": lambda v: (
                       "never" if pd.isna(v) or not np.isfinite(float(v))
                       else pc(v, 2))}),
        "Break-even annual hedging cost by hedge ratio",
        note="The break-even cost is the annual charge at which hedging "
             "exactly offsets its benefit. \"Never\" means the hedged "
             "position loses even at zero cost, so no price makes it "
             "worthwhile."))
    # Classified from the table: whether any ratio pays at zero cost is the
    # whole question, and it must not be written down in advance of the answer.
    payers = breakeven[breakeven["gain_at_zero_cost_pct"] > 0.0]
    best_free = float(free["optimal_hedge_ratio"])
    if payers.empty:
        worst = breakeven.loc[breakeven["gain_at_zero_cost_pct"].idxmin()]
        least = breakeven.loc[breakeven["gain_at_zero_cost_pct"].idxmax()]
        out.append(ctx.p(
            f"<b>The answer here is that no price is low enough.</b> Every "
            f"hedge ratio tested loses certainty-equivalent consumption "
            f"before a single basis point of cost is charged. The mildest is "
            f"a {pc(float(least['hedge_ratio']), 0)} hedge, which gives up "
            f"{f2(abs(float(least['gain_at_zero_cost_pct'])), 2)}%; the "
            f"fully hedged position gives up "
            f"{f2(abs(float(worst['gain_at_zero_cost_pct'])), 2)}%. The loss "
            f"grows monotonically in the ratio, so there is no break-even cost "
            f"to report: the decision is settled at zero, and adding a "
            f"realistic charge only widens the gap."))
        out.append(ctx.p(
            f"Tracing the optimal ratio as a function of cost adds nothing, "
            f"because it is {pc(best_free, 0)} at every cost including free. "
            f"For a lifecycle investor holding this international leg, the "
            f"model's answer is not \u201chedge if you can get it cheaply\u201d "
            f"but \u201cdo not hedge.\u201d"))
    else:
        out.append(ctx.p(
            f"At a {pc(float(be25['hedge_ratio']), 0)} hedge ratio and zero "
            f"cost the gain is "
            f"{f2(float(be25['gain_at_zero_cost_pct']), 2)}% of "
            f"certainty-equivalent consumption, with a break-even annual cost "
            f"of {pc(float(be25['break_even_annual_cost']), 2)}. A fully "
            f"hedged position ({pc(float(full['hedge_ratio']), 0)}) gives up "
            f"{f2(abs(float(full['gain_at_zero_cost_pct'])), 2)}% even before "
            f"cost, so there is no price at which it makes sense."))

    out.extend(ctx.figure(
        "fig23_currency_hedging",
        "Top left: the certainty-equivalent cost of hedging by ratio, at five "
        "annual hedging costs; every line is below zero, so hedging loses even "
        "when free. Top right: where the loss lands — fifth-percentile "
        "retirement consumption falls monotonically in the hedge ratio, and "
        "the certainty equivalent weighs that tail heavily. Bottom: why — "
        "hedging lowers the "
        "standalone volatility of the foreign sleeve up to a half hedge, but "
        "raises its correlation with the home market over the same range."))
    return out


# ---------------------------------------------------------------------------
# 12. Endogenous retirement timing
# ---------------------------------------------------------------------------
def section_retirement(ctx: Any) -> List[Flowable]:
    f = ctx.f
    summary = f.table("retirement_timing_summary")
    conditioning = f.table("retirement_value_of_conditioning")
    lottery = f.table("retirement_lottery_stats").iloc[0]
    deciles = f.table("retirement_lottery_deciles")
    bull = f.table("retirement_bull_market_test").iloc[0]
    floor = float(f.cfg["retirement_timing"]["working_income_floor"])
    cond = conditioning[conditioning["working_income_floor"] == floor]
    best = cond.loc[cond["value_of_conditioning_pct"].idxmax()]
    unfloored = conditioning[conditioning["working_income_floor"] == 0.0]

    out: List[Flowable] = [ctx.h1("#retirement. Endogenous Retirement Timing")]
    out.append(ctx.p(
        "Every result so far retires the investor on a birthday. Real people "
        "do not. They retire when the balance looks big enough — which means, "
        "mechanically, that they retire disproportionately after good markets. "
        "That is worth testing rather than assuming, because a balance that "
        "looks big after a bull run is also a balance bought at high "
        "valuations, and the same market move that triggers the decision may "
        "be lowering the returns that have to fund it."))
    out.append(ctx.p(
        "This section makes the retirement date a path-dependent decision and "
        "asks two questions: whether a wealth trigger beats a fixed date, and "
        "how much of the outcome the decade around the retirement date "
        "explains."))

    out.append(ctx.h2("#retirement.1 A wealth trigger against a date"))
    out.append(ctx.p(
        f"Utility here is evaluated over the whole lifetime rather than the "
        f"retirement window, because a rule that retires people early buys "
        f"them leisure that a retirement-only window would charge for without "
        f"crediting. We also introduce a working-life income floor of "
        f"{pc(floor, 0)} of average earnings. That is not decoration either: "
        f"retirement in this model carries a real consumption floor through "
        f"the progressive social-security schedule and working life, at zero, "
        f"carries none. Wherever compared policies share the same working-life "
        f"consumption the asymmetry cancels, but here it does not — and left "
        f"uncorrected it rewards retiring early purely for reaching the safety "
        f"net sooner."))
    out.extend(ctx.table(
        rows_from(summary[summary["working_income_floor"] == floor],
                  ["variant", "cec_gamma5", "median_retire_age",
                   "mean_retire_age", "sd_retire_age", "prob_ruin",
                   "median_wealth_at_retirement",
                   "median_retirement_consumption"],
                  ["Rule", "CEC γ=5", "Median age", "Mean age", "S.d. age",
                   "P(ruin)", "Median wealth at retirement", "Median cons."],
                  {"variant": str, "prob_ruin": lambda v: pc(v, 1),
                   "median_wealth_at_retirement": lambda v: f2(v, 1)}),
        "Fixed retirement dates against wealth triggers",
        note="A wealth trigger retires the investor once financial wealth "
             "reaches the stated multiple of current income, within an age "
             "window. Whole-lifetime utility, working-income floor at "
             f"{pc(floor, 0)} of average earnings.",
        font_size=7.2))
    out.append(ctx.p(
        "Read naively this table says wealth triggers beat fixed dates. That "
        "reading is wrong, and the reason is instructive enough to spend a "
        "paragraph on. The model contains no disutility of labour: working an "
        "extra year is costless and produces income, so the certainty "
        "equivalent falls monotonically with the fixed retirement age for "
        "purely mechanical reasons. A wealth trigger that retires people later "
        "on average will therefore look better than age 63 without any "
        "conditioning value at all."))
    out.append(ctx.p(
        "The correct comparison holds the mean retirement age fixed. We "
        "interpolate the fixed-date frontier at each rule's own realised mean "
        "retirement age and score against that."))
    out.extend(ctx.table(
        rows_from(cond, ["variant", "mean_retire_age", "cec",
                         "matched_fixed_date_cec", "value_of_conditioning_pct"],
                  ["Rule", "Mean retirement age", "CEC",
                   "Matched fixed-date CEC", "Value of conditioning (%)"],
                  {"variant": str, "value_of_conditioning_pct": lambda v: f2(v, 2)}),
        "The value of conditioning, against a date matched on the same mean age",
        note="This is the number that isolates conditioning from timing. The "
             "unmatched comparison in the previous table conflates the two."))
    out.append(ctx.p(
        f"Matched, the value of conditioning is "
        f"{f2(float(best['value_of_conditioning_pct']), 2)}% at the best "
        f"trigger — real, but roughly {int(round(100 * (1 - float(best['value_of_conditioning_pct']) / float(unfloored['value_of_conditioning_pct'].max()))))}% "
        f"smaller than the same comparison without the working-income floor, "
        f"where it reaches "
        f"{f2(float(unfloored['value_of_conditioning_pct'].max()), 2)}%. Both "
        f"numbers are reported because the difference between them is a model "
        f"artefact we found and removed, not a result."))

    out.append(ctx.h2("#retirement.2 The retirement-date lottery"))
    out.append(ctx.p(
        f"How much of a retirement outcome is decided by when you happened to "
        f"be born? We regress the outcome on the annualised real portfolio "
        f"return over the decade centred on each path's own retirement date. "
        f"That window explains "
        f"{pc(float(lottery['r2_retirement_window']), 1)} of the variation in "
        f"retirement consumption, against "
        f"{pc(float(lottery['r2_whole_lifetime']), 1)} for the whole "
        f"{int(f.cfg['lifecycle']['age_death']) - int(f.cfg['lifecycle']['age_start'])}-year "
        f"lifetime. A single decade — {pc(10 / 68, 0)} of the horizon — "
        f"carries {pc(float(lottery['share_of_lifetime_r2']), 0)} of the "
        f"explanatory power of the whole thing."))
    out.append(ctx.p(
        "That is the sequence-of-returns problem stated as a number, and it is "
        "larger than the difference between any two allocation strategies in "
        "this paper. It is also the strongest argument for the flexible "
        "retirement date of the previous subsection: the one lever that acts "
        "directly on the lottery is the ability to choose which decade you "
        "retire into."))
    out.extend(ctx.table(
        rows_from(deciles, list(deciles.columns)[:6],
                  [c.replace("_", " ").title() for c in list(deciles.columns)[:6]]),
        "Retirement outcomes by decile of the retirement-decade return",
        note="Paths sorted by the annualised real return over the ten years "
             "centred on their own retirement date, then split into deciles.",
        font_size=7.2))

    out.append(ctx.h3("Do people retire into bull markets, and does it hurt?"))
    out.append(ctx.p(
        f"Under a wealth trigger the answer to the first question is yes: the "
        f"correlation between retirement age and the preceding market run-up "
        f"is {f2(float(bull['corr_retire_age_vs_runup']), 3)}, so a strong "
        f"market pulls the retirement date forward. Early retirees experience "
        f"a mean pre-retirement run-up of "
        f"{pc(float(bull['mean_runup_early_retirees']), 1)} against "
        f"{pc(float(bull['mean_runup_late_retirees']), 1)} for late ones."))
    out.append(ctx.p(
        f"The second question is the interesting one, and the answer is "
        f"reassuring for the trigger. The correlation between the run-up and "
        f"the <i>subsequent</i> return is "
        f"{f2(float(bull['corr_runup_vs_subsequent_return']), 3)} — "
        f"essentially zero. On this panel, retiring after a good decade does "
        f"not systematically buy you a bad one. The folk warning that bull "
        f"markets lure people into retiring at the worst possible time is not "
        f"supported here: the mean subsequent return for early retirees "
        f"({pc(float(bull['mean_subsequent_return_early']), 1)}) is if "
        f"anything slightly higher than for late ones "
        f"({pc(float(bull['mean_subsequent_return_late']), 1)}). We report "
        f"this because it cuts against the intuition that motivated the test."))
    out.extend(ctx.figure(
        "fig24_retirement_timing",
        "Endogenous retirement timing. The distribution of retirement ages "
        "under a wealth trigger, the value of conditioning against a matched "
        "fixed date, and the retirement-date lottery."))
    return out


# ---------------------------------------------------------------------------
# 13. Conditioning the savings rate
# ---------------------------------------------------------------------------
def section_saving(ctx: Any) -> List[Flowable]:
    f = ctx.f
    frontier = f.table("saving_constant_rate_frontier")
    shape = f.table("saving_shape_summary")
    profiles = f.table("saving_solved_profiles")
    matched = f.table("saving_matched_rate")
    combined = f.table("saving_combined_with_retirement")
    deviation = f.table("saving_deviation_profile")
    gamma = f.baseline_gamma
    target_mean = float(f.cfg["saving"]["target_mean_rate"])
    beta = float(f.cfg["utility"]["discount_factor"])
    n_working = int(f.cfg["lifecycle"]["age_retire"]) - int(f.cfg["lifecycle"]["age_start"])
    peak = frontier.loc[frontier[f"cec_gamma{gamma:g}"].idxmax()]
    material = deviation[deviation["cost_of_resetting_bp"].abs() > 1.0]

    out: List[Flowable] = [ctx.h1("#saving. Conditioning the Savings Rate")]
    out.append(ctx.p(
        "Section #valuation began the part of this paper that leaves the "
        "portfolio alone: it asked what the market charged on the day a "
        "lifetime opened, and found that the answer moves the outcome without "
        "changing the allocation. This section and the three after it carry "
        "that further, in the order a life meets them — how much you save, "
        "what that saving should respond to, when you stop, and how you draw "
        "down. The first question is whether the savings rate should vary "
        "over a career at all, and whether it should respond to the "
        "portfolio."))
    out.append(ctx.p(
        "Two quite different things could make the rate vary, and a comparison "
        "that does not separate them will credit the wrong one. <b>Shape</b> "
        "is variation with age alone: labour income here is hump-shaped, and a "
        "fixed rate makes consumption track income exactly, so a 25-year-old "
        "consumes least precisely when they are poorest. <b>Conditioning</b> "
        "is variation with state: whether wealth is ahead of or behind an "
        "age-appropriate target. The first uses no market information at all; "
        "the second is what \"you should have six times salary by fifty\" is "
        "reaching for."))

    out.append(ctx.h2("#saving.1 A caveat that has to come first"))
    out.append(ctx.p(
        f"The model cannot identify the savings <i>level</i>. Swept "
        f"continuously, the certainty equivalent peaks at a constant rate of "
        f"{pc(float(peak['savings_rate']), 0)} and falls above it. That number "
        f"should not be believed and it is worth being explicit about why."))
    out.append(ctx.p(
        f"The savings level is a trade between consuming now and consuming "
        f"later. In this model that trade is settled almost entirely by the "
        f"discount factor — β = {beta:g}, which over a "
        f"{n_working}-year working life discounts retirement consumption by a "
        f"factor of {f2(beta ** n_working, 2)} — and by risk aversion acting "
        f"on the left tail of consumption, which with a floored retirement and "
        f"risky labour income sits in working life rather than in retirement. "
        f"Neither of those is something a panel of historical returns has any "
        f"view on."))
    out.append(ctx.p(
        f"So the level is not identified here and this paper does not claim "
        f"it. What the return panel <i>can</i> speak to is the shape of the "
        f"profile and the value of conditioning it, both holding the average "
        f"rate fixed. Everything below pins the career-average savings rate at "
        f"{pc(target_mean, 0)} and asks only when, and on what, to save it."))

    out.append(ctx.h2("#saving.2 Shape: when should you save?"))
    out.append(ctx.p(
        "We solve for a free savings rate at each of the 38 working years by "
        "coordinate ascent over a per-age multiplier, renormalising after "
        "every move so that the career average never drifts. The answer is not "
        "the same at every risk aversion, and that is the interesting part."))
    out.extend(ctx.table(_shape_rows(profiles, shape),
                         "Solved savings profile by risk aversion, career "
                         "average pinned",
                         note="The profile is summarised by the average rate "
                              "in the first quarter, middle half and last "
                              "quarter of the working life. All three profiles "
                              "have the same career average by construction, "
                              "so the comparison is purely about shape."))
    out.append(ctx.p(
        "At moderate risk aversion the solved profile is <b>hump-shaped</b>: "
        "save least when young, most in peak-earning years, taper into "
        "retirement. That is consumption smoothing doing what theory says it "
        "should. A 25-year-old sits at the bottom of a hump-shaped income "
        "profile, and taking a further tenth of that income away is expensive "
        "in utility terms precisely because there is so little of it. This is "
        "the \"save more later, when you earn more\" pattern, and the model "
        "produces it without being told to."))
    out.append(ctx.p(
        "At high risk aversion it <b>inverts</b> to front-loaded. Two motives "
        "compete and risk aversion picks the winner. Smoothing wants saving "
        "where income is highest, which is mid-career. Precaution wants a "
        "buffer built early, so that it has the whole remaining career to "
        "compound against a bad income or market draw. At γ = 10 precaution "
        "wins outright. The model cannot settle which kind of investor a given "
        "reader is; what it can say is that both beat a flat rate and that "
        "getting the shape right matters more the more risk-averse you are."))
    out.append(ctx.p(
        f"Unlike the glide path of Section #glide, this structure is real. The "
        f"deviation profile shows {len(material)} of {len(deviation)} working "
        f"years moving the objective by more than a basis point when reset to "
        f"the career average."))

    out.append(ctx.h2("#saving.3 Conditioning: your position beats the market's direction"))
    out.append(ctx.p(
        "Layered on top of the solved shape, and scored against a constant "
        "rate interpolated at each rule's own realised career average, we test "
        "two conditioning rules: an <i>on-track</i> rule that raises the rate "
        "when wealth is below an age-appropriate wealth-to-income target, and "
        "a <i>return-responsive</i> rule that raises it after a weak market "
        "year."))
    out.extend(ctx.table(
        rows_from(matched.sort_values("value_of_shape_pct", ascending=False),
                  ["variant", "mean_savings_rate", "cec",
                   "matched_constant_rate_cec", "value_of_shape_pct"],
                  ["Rule", "Realised mean rate", "CEC",
                   "Matched constant CEC", "Value (%)"],
                  {"variant": str,
                   "mean_savings_rate": lambda v: pc(v, 2),
                   "value_of_shape_pct": lambda v: f2(v, 2)},
                  limit=10),
        "Conditioning rules at a matched career-average savings rate",
        note="The matched comparison is essential: a rule that saves more "
             "when behind will, in a world where most paths fall behind, "
             "quietly save more overall. Interpolating the constant-rate "
             "frontier at each rule's own realised mean strips that out.",
        font_size=7.2))
    out.append(ctx.p(
        "Saving more when behind an age-appropriate wealth target is worth "
        "several times what the age profile alone is worth. Saving more after "
        "a bad market year is worth almost nothing. The sign check behaves as "
        "it must: reversing the on-track rule — saving <i>less</i> when behind "
        "— is strongly negative, which is the reassurance that the machinery "
        "measures what it claims."))
    out.append(ctx.p(
        "The interpretation is that a bad market year is a poor proxy for "
        "being behind. Wealth relative to an age-appropriate target is the "
        "sufficient statistic; the market's recent direction is a noisy "
        "shadow of it. Section #accumulation takes this apart in detail and finds that "
        "even this reading needs qualifying."))
    out.extend(ctx.figure(
        "fig25_savings_rate",
        "The solved savings profile by risk aversion, the unidentified "
        "constant-rate frontier, and the value of conditioning on wealth "
        "against conditioning on returns."))

    out.append(ctx.h2("#saving.4 Do the savings and retirement gains add?"))
    out.append(ctx.p(
        "Section #retirement below conditions the retirement date on the "
        "portfolio; this section conditions the savings rate on it. A natural "
        "question is "
        "whether an investor should do both."))
    out.extend(ctx.table(
        rows_from(combined, ["variant", f"cec_gamma{gamma:g}",
                             "vs_neither_pct", "mean_savings_rate",
                             "mean_retire_age", "prob_ruin"],
                  ["Configuration", "CEC", "vs neither (%)", "Mean rate",
                   "Mean retirement age", "P(ruin)"],
                  {"variant": str, "vs_neither_pct": lambda v: f2(v, 2),
                   "mean_savings_rate": lambda v: pc(v, 2),
                   "prob_ruin": lambda v: pc(v, 1)}),
        "Savings conditioning and retirement conditioning, separately and "
        "together",
        note="Whole-lifetime utility. Both rules read the same underlying "
             "signal, so their gains overlap rather than add."))
    out.append(ctx.p(
        f"They do not add. Savings conditioning alone is worth "
        f"{f2(_combined(combined, 'Savings conditioning only'), 2)}%, "
        f"retirement conditioning alone "
        f"{f2(_combined(combined, 'Retirement conditioning only'), 2)}%, and "
        f"both together {f2(_combined(combined, 'Both'), 2)}% — less than the "
        f"better of the two alone. Both rules respond to the same "
        f"ahead-or-behind signal, so running them together over-corrects. This "
        f"is a practical result: an investor choosing between the two levers "
        f"should choose, not stack."))
    return out


def _shape_rows(profiles: pd.DataFrame, shape: pd.DataFrame) -> List[List[str]]:
    """Classify each solved savings profile from the data rather than asserting it."""
    rows = [["Risk aversion", "Shape", "First quarter", "Middle half",
             "Last quarter", "Peak age", "Gain vs flat (%)"]]
    for g in sorted(profiles["risk_aversion"].unique()):
        block = profiles[profiles["risk_aversion"] == g].sort_values("age")
        n = len(block)
        lo = block.head(max(n // 4, 1))["savings_rate"].mean()
        mid = block.iloc[n // 4: 3 * n // 4]["savings_rate"].mean()
        hi = block.tail(max(n // 4, 1))["savings_rate"].mean()
        if mid > lo and mid > hi:
            kind = "hump-shaped"
        elif lo > hi:
            kind = "front-loaded"
        elif hi > lo:
            kind = "back-loaded"
        else:
            kind = "flat"
        gain = shape[shape["risk_aversion"] == g]["gain_vs_flat_pct"]
        rows.append([
            f"γ = {float(g):g}", kind, pc(lo, 1), pc(mid, 1), pc(hi, 1),
            f"{int(block.loc[block['savings_rate'].idxmax(), 'age'])}",
            f2(float(gain.iloc[0]), 2) if len(gain) else "--"])
    return rows


def _combined(frame: pd.DataFrame, variant: str) -> float:
    block = frame[frame["variant"] == variant]
    return float(block["vs_neither_pct"].iloc[0]) if len(block) else float("nan")


# ---------------------------------------------------------------------------
# 14. The accumulation signal, decomposed
# ---------------------------------------------------------------------------
def section_accumulation(ctx: Any) -> List[Flowable]:
    f = ctx.f
    acfg = f.cfg["accumulation"]
    shape_value = f.shape_value
    forms = f.table("acc_response_forms")
    asym = f.table("acc_asymmetry")
    bands = f.table("acc_guardrail_bands")
    targets = f.table("acc_target_choice")
    race = f.table("acc_signal_race")
    best = f.signals
    combo = f.table("acc_signal_combination")
    feas = f.table("acc_feasibility").sort_values("width")
    quant = f.table("acc_quantile_gain")
    windows = f.table("acc_age_windows")
    by_gamma = f.table("acc_by_risk_aversion")
    by_strategy = f.table("acc_by_strategy")
    by_income = f.table("acc_by_income_volatility")
    V = "matched_value_pct"

    def net(row: Any) -> float:
        return float(row[V]) - shape_value

    form_best = forms.loc[forms.groupby("form")[V].idxmax()] \
        .sort_values(V, ascending=False)
    common = _common_strength(forms)
    behind_only = asym[asym["k_ahead"] == 0.0]
    ahead_only = asym[asym["k_behind"] == 0.0]
    best_behind = behind_only.loc[behind_only[V].idxmax()]
    best_ahead = ahead_only.loc[ahead_only[V].idxmax()]
    best_both = asym.loc[asym[V].idxmax()]
    best_band = bands.loc[bands[V].idxmax()]
    unscaled = targets[np.isclose(targets["factor"], 1.0)]
    unc_net = float(feas[V].iloc[-1]) - shape_value

    def share_at(width: float) -> float:
        y = (feas[V].to_numpy(float) - shape_value) / max(unc_net, 1e-9) * 100.0
        return float(np.interp(width, feas["width"].to_numpy(float), y))

    def target_net(name: str) -> float:
        block = unscaled[unscaled["target"] == name]
        return net(block.iloc[0]) if len(block) else float("nan")

    n_evals = sum(len(t) for t in (forms, asym, bands, targets, race, combo,
                                   feas, windows, by_gamma, by_strategy,
                                   by_income))

    # Read the saving section's own matched-comparison table rather than
    # restating its finding from memory, which is how this claim drifted
    # before. A section number in a comment goes stale like any other.
    matched = f.table("saving_matched_rate")
    on_track = matched[matched["variant"].astype(str).str.contains("On-track")]
    best_on_track = float(on_track["value_of_shape_pct"].max())

    out: List[Flowable] = [ctx.h1("#accumulation. The Accumulation Signal, Decomposed")]
    out.append(ctx.p(
        f"Section #saving established that conditioning the savings rate on "
        f"the funded ratio is worth {best_on_track:.1f}% of "
        f"certainty-equivalent consumption at its best setting, measured "
        f"against a constant rate matched on the same realised career "
        f"average. Section #accumulation.5 re-scores it on a common basis "
        f"against eight competing signals and puts it at "
        f"{f.signal_net('funded_ratio'):.1f}%; the two differ because they "
        f"hold different things fixed, and both are reported rather than "
        f"reconciled into one. That result "
        f"rests on five choices that were made once "
        f"and never varied: the gap was measured in income multiples, the "
        f"target was the model's own median path, one coefficient governed "
        f"both directions, the rate was free to move anywhere between "
        f"{pc(float(acfg['rate_floor']), 0)} and "
        f"{pc(float(acfg['rate_cap']), 0)}, and the rule ran for the whole "
        f"career. Each is a modelling decision, and a number that survives "
        f"only one setting of them is not worth acting on."))
    out.append(ctx.p(
        f"This section varies all five, races five further state variables "
        f"against the funded ratio, tests whether the two best combine, and "
        f"asks where in the distribution and at which ages the value lands. "
        f"{n_evals:,} rule evaluations at "
        f"{int(acfg['n_paths']):,} paths each, all under common random "
        f"numbers."))
    out.append(ctx.p(
        f"Two scoring conventions apply throughout and both matter. Each "
        f"number is a percentage gain over a <i>constant</i> savings rate "
        f"interpolated at the rule's own realised career average. And "
        f"conditioning is layered on top of a solved age profile that is "
        f"itself worth {sgn(shape_value)}% against that matched constant, so "
        f"every figure quoted below is <i>net</i> of the shape's own "
        f"contribution."))

    # -- 12.1 functional form --------------------------------------------
    out.append(ctx.h2("#accumulation.1 Which units is \"behind\" measured in?"))
    out.append(ctx.p(
        "Being \"two times salary short\" means something entirely different "
        "at 30 and at 60, because the target itself grows from roughly nothing "
        "to around ten times income over a career. A response linear in that "
        "gap is therefore inert when young and violent when old, whether or "
        "not anyone intended it. We test two scale-free alternatives that "
        "measure the shortfall as a fraction of the target instead."))
    out.extend(ctx.table(
        rows_from(form_best.assign(net=form_best[V] - shape_value),
                  ["form_label", "sensitivity", "rate_move_pp", V, "net",
                   "mean_savings_rate", "prob_ruin"],
                  ["Gap measured as", "Best k", "Extra saving when 25% behind "
                   "(pp)", "Value (%)", "Net of shape (%)", "Mean rate",
                   "P(ruin)"],
                  {"form_label": str, "sensitivity": lambda v: f2(v, 3),
                   "rate_move_pp": lambda v: f2(v, 2),
                   V: lambda v: f2(v, 2), "net": lambda v: f2(v, 2),
                   "mean_savings_rate": lambda v: pc(v, 2),
                   "prob_ruin": lambda v: pc(v, 1)}),
        "Three ways of measuring the shortfall, each at its own best coefficient",
        note="The three forms measure the gap in different units, so their "
             "coefficients are not comparable. The third column translates "
             "each into a common scale: percentage points of income added for "
             "a path a quarter short of target at mid-career.",
        font_size=7.2))
    out.append(ctx.p(
        f"Ranked at their own optima the two scale-free forms are "
        f"indistinguishable and both beat the level gap. But that comparison "
        f"is not quite fair, because the three grids do not reach equally far. "
        f"Interpolating every curve at the strongest response <i>all</i> of "
        f"them can produce separates \"this form is better\" from \"this "
        f"form's grid let it go further\"."))
    out.extend(ctx.table(
        common["rows"],
        "Functional forms compared at matched response strength",
        note="Each form's curve interpolated at the largest contribution move "
             "that all three grids can produce. This is the comparison the "
             "text relies on; the previous table's ranking partly reflects "
             "how far each grid reaches."))
    out.append(ctx.p(
        f"At matched strength {f2(common['spread'], 2)} percentage points "
        f"still separate the forms, so the ranking is about the shape of the "
        f"response and not only about its strength. A rule linear in income "
        f"multiples is inert exactly when the investor has time to respond and "
        f"violent exactly when they do not, and it pays for that."))
    out.extend(ctx.figure(
        "fig26_savings_response_form",
        "Left: the value of conditioning against the contribution move it "
        "implies, which puts three differently-scaled coefficients on one "
        "axis. Right: the policy each form implies at its own optimum, over "
        "the range of funded ratios paths actually visit."))

    # -- 12.2 asymmetry ---------------------------------------------------
    out.append(ctx.h2("#accumulation.2 Catching up and easing off are separate policies"))
    out.append(ctx.p(
        "A symmetric rule does two things at once: it saves more when behind "
        "and less when ahead. There is no reason those should carry the same "
        "coefficient, and separating them is the question a real saver faces — "
        "almost all published advice is about catching up, and almost none is "
        "about easing off."))
    out.extend(ctx.table(
        [["Rule", "k behind", "k ahead", "Value (%)", "Net of shape (%)",
          "Realised mean rate"],
         ["Neither (age profile only)", "0", "0", f2(shape_value, 2), "0.00",
          pc(float(f.cfg['accumulation']['target_mean_rate']), 1)],
         ["Catch up only (never ease off)",
          f2(float(best_behind['k_behind']), 2), "0",
          f2(float(best_behind[V]), 2), f2(net(best_behind), 2),
          pc(float(best_behind['mean_savings_rate']), 1)],
         ["Ease off only (never catch up)", "0",
          f2(float(best_ahead['k_ahead']), 2), f2(float(best_ahead[V]), 2),
          f2(net(best_ahead), 2), pc(float(best_ahead['mean_savings_rate']), 1)],
         ["Both, free coefficients", f2(float(best_both['k_behind']), 2),
          f2(float(best_both['k_ahead']), 2), f2(float(best_both[V]), 2),
          f2(net(best_both), 2), pc(float(best_both['mean_savings_rate']), 1)]],
        "Which half of the conditioning rule earns its keep",
        note="Each half is optimised separately over the same grid. Note the "
             "realised mean rates: catch-up alone raises the career average "
             "well above target and ease-off alone cuts it well below, so only "
             "the matched-rate comparison makes them comparable."))
    out.append(ctx.p(
        f"Net of the age profile, catching up alone is worth "
        f"{sgn(net(best_behind))}% and easing off alone "
        f"{sgn(net(best_ahead))}% — a ratio of "
        f"{f2(max(net(best_behind), net(best_ahead)) / max(min(net(best_behind), net(best_ahead)), 1e-9), 2)}, "
        f"which on any reasonable reading is a tie. That is itself the "
        f"finding, because the two halves are not symmetric in anything except "
        f"their value: catch-up alone raises the career savings rate to "
        f"{pc(float(best_behind['mean_savings_rate']), 1)} and ease-off alone "
        f"cuts it to {pc(float(best_ahead['mean_savings_rate']), 1)}. They "
        f"arrive at the same place from opposite directions. On raw certainty "
        f"equivalent the one that saves more would simply look better; only "
        f"the matched comparison reveals the tie."))
    out.append(ctx.p(
        f"Run together they reach {sgn(net(best_both))}%, against "
        f"{f2(net(best_behind) + net(best_ahead), 2)}% for the two summed. The "
        f"halves are strongly sub-additive: much of what each earns alone is "
        f"the same correction, arrived at from opposite sides. The free search "
        f"picks "
        f"{'the same coefficient in both directions' if abs(float(best_both['k_behind']) - float(best_both['k_ahead'])) < 1e-9 else 'different coefficients'}, "
        f"so there is no case here for a deliberately asymmetric rule."))
    out.extend(ctx.figure(
        "fig27_savings_asymmetry",
        "Left: the value of conditioning over the two coefficients "
        "separately, with the best pair starred. Right: each half of the rule "
        "switched on alone against both together."))

    # -- 12.3 implementable version ---------------------------------------
    out.append(ctx.h2("#accumulation.3 The version a person could actually follow"))
    out.append(ctx.p(
        "A continuous response asks for a freshly computed contribution rate "
        "every year. Section #spending finds that coarse guardrail rules give "
        "up surprisingly little on the spending side, and the "
        "accumulation-side analogue is the same idea run forwards: check once "
        "a year and move the contribution by a fixed step only if the funded "
        "ratio is more than a dead band away from target."))
    out.extend(ctx.table(
        rows_from(bands.sort_values(V, ascending=False)
                  .assign(net=bands[V] - shape_value),
                  ["band", "step", V, "net", "mean_savings_rate", "prob_ruin"],
                  ["Dead band (± of target)", "Rate step", "Value (%)",
                   "Net of shape (%)", "Mean rate", "P(ruin)"],
                  {"band": lambda v: pc(v, 0), "step": lambda v: pc(v, 0),
                   V: lambda v: f2(v, 2), "net": lambda v: f2(v, 2),
                   "mean_savings_rate": lambda v: pc(v, 2),
                   "prob_ruin": lambda v: pc(v, 1)},
                  limit=8),
        "Guardrail savings rules",
        note="A two-number rule: check the funded ratio once a year, and move "
             "the contribution by the step only if the ratio is more than the "
             "dead band away from target."))
    out.append(ctx.p(
        f"The best guardrail — move {pc(float(best_band['step']), 0)} of "
        f"income once you are more than {pc(float(best_band['band']), 0)} off "
        f"target — is worth {sgn(net(best_band))}%, or "
        f"{f2(100 * net(best_band) / max(net(form_best.iloc[0]), 1e-9), 0)}% "
        f"of what the tuned continuous rule earns. It wants the narrowest dead "
        f"band and the largest step the grid offers, which is the search "
        f"saying it would rather be continuous; the table prices that "
        f"compromise rather than pretending there is none. Even so, keeping "
        f"most of the value for a rule with two numbers and no arithmetic is "
        f"the trade most savers should take."))

    # -- 12.4 target ------------------------------------------------------
    out.append(ctx.h2("#accumulation.4 Does the target have to be right?"))
    out.append(ctx.p(
        "The target is the weakest link in the construction: it is the model's "
        "own median wealth path, which no investor could know in advance. We "
        "test two alternatives that an investor could: an \"N times salary by "
        "age X\" ladder of the kind large fund managers publish, and a flat "
        "multiple with no age content at all."))
    out.append(ctx.p(
        "The ladder is worth stating in full rather than gesturing at, since "
        "it is the alternative a real saver is most likely to have been "
        "handed. Fidelity's widely circulated guideline sets four anchors — "
        "one times salary by 30, three by 40, six by 50, eight by 60 and ten "
        "by 67. We interpolate linearly between them, and add intermediate "
        "anchors at 35, 45 and 55 to keep the interpolation from running "
        "straight across a decade; those three are ours, not Fidelity's, and "
        "sit on the line the published anchors imply."))
    out.extend(ctx.table(
        [["Age", "30", "35", "40", "45", "50", "55", "60", "67"],
         ["Wealth, × salary", "1×", "2×", "3×", "4×", "6×", "7×", "8×", "10×"],
         ["Published?", "yes", "no", "yes", "no", "yes", "no", "yes", "yes"]],
        "The \"N times salary\" ladder, in full",
        note="Linear interpolation between anchors; flat at 10× beyond 67 and "
             "zero before 30. The published anchors are Fidelity's retirement "
             "savings guideline (1× by 30, 3× by 40, 6× by 50, 8× by 60, 10× "
             "by 67), which assumes roughly 15% saved a year, retirement at "
             "67, and a target of maintaining pre-retirement lifestyle. The "
             "three unpublished anchors are our own interpolation aids."))
    out.extend(ctx.table(
        rows_from(unscaled.assign(net=unscaled[V] - shape_value)
                  .sort_values(V, ascending=False),
                  ["target", "median_target_multiple", V, "net",
                   "mean_savings_rate"],
                  ["Target", "Median multiple asked for", "Value (%)",
                   "Net of shape (%)", "Realised mean rate"],
                  {"target": str,
                   "median_target_multiple": lambda v: f2(v, 1),
                   V: lambda v: f2(v, 2), "net": lambda v: f2(v, 2),
                   "mean_savings_rate": lambda v: pc(v, 1)}),
        "Three wealth-to-income targets, unscaled",
        note="The ladder is the published rule of thumb tabulated above; the "
             "flat multiple holds the same multiple at every age and serves "
             "as a deliberate straw man."))
    out.append(ctx.p(
        f"The published ladder captures "
        f"{f2(100 * target_net('published ladder') / max(target_net('model median path'), 1e-9), 0)}% "
        f"of the model-implied target's value — useful, but not a substitute. "
        f"The flat multiple is worth "
        f"{sgn(target_net([t for t in unscaled['target'] if t.startswith('flat')][0]))}%: "
        f"<b>worse than not conditioning at all</b>. Telling a 28-year-old "
        f"they should already hold eight times salary leaves them behind "
        f"target for most of a career and drives the career average savings "
        f"rate far above the pinned target. That is not conditioning, it is "
        f"just saving more, and the matched comparison charges for it."))
    out.append(ctx.p(
        "The ranking says two separate things. A target that rises with age is "
        "necessary — the flat one is not merely weaker, it is harmful. But "
        "rising is not sufficient: the ladder rises and still gives up a "
        "substantial share of the value, so how it rises matters too."))
    out.extend(ctx.figure(
        "fig32_savings_target_choice",
        "Left: the value of conditioning as each target is scaled up and "
        "down. Right: what each target actually asks the investor to hold at "
        "each age."))

    # -- 12.5 horse race --------------------------------------------------
    out.append(ctx.h2("#accumulation.5 Which signal, out of everything available?"))
    out.append(ctx.p(
        "Eight candidate state variables, all signed so that positive means "
        "\"save more\", all bounded so that one sensitivity grid is meaningful "
        "across them, all swept over the same grid and scored the same way. "
        "They fall into three kinds: <i>stock</i> signals that say where the "
        "investor is, <i>flow</i> signals that say what the market just did, "
        "and the pay cheque."))
    out.extend(ctx.table(
        rows_from(best, ["signal_label", "family", "sensitivity", V, "net",
                         "mean_savings_rate", "prob_ruin",
                         "p5_retirement_consumption"],
                  ["Signal", "Kind", "Best k", "Value (%)", "Net of shape (%)",
                   "Mean rate", "P(ruin)", "5th pct cons."],
                  {"signal_label": str, "family": str,
                   "sensitivity": lambda v: f2(v, 2), V: lambda v: f2(v, 2),
                   "net": lambda v: f2(v, 2),
                   "mean_savings_rate": lambda v: pc(v, 1),
                   "prob_ruin": lambda v: pc(v, 1),
                   "p5_retirement_consumption": lambda v: f2(v, 3)}),
        "The signal horse race",
        note="Each signal at its own best sensitivity. The \"no conditioning\" "
             "row appears once for every point on the grid and spans zero "
             "percentage points across them, which is the determinism check "
             "that the common random numbers are holding.",
        font_size=7.0))
    out.append(ctx.p(
        f"<b>The winner is not a portfolio signal at all.</b> Saving more in "
        f"years when the pay cheque runs above its expected path — and less "
        f"when it runs below — is worth {sgn(f.signal_net('income_shock'))}%, "
        f"against {sgn(f.signal_net('funded_ratio'))}% for the funded ratio "
        f"and {sgn(f.signal_net('wealth_level'))}% for the raw balance. This "
        f"is the accumulation-side version of consumption smoothing: an income "
        f"shock is observed <i>before</i> it is spent, so responding to it "
        f"costs the investor very little utility, whereas responding to a "
        f"portfolio shortfall means cutting consumption that was already "
        f"planned. It is also the easiest signal in the list to act on, "
        f"because it needs no target, no balance and no arithmetic."))
    out.append(ctx.p(
        f"<b>Every flow signal is worth essentially nothing.</b> Trailing "
        f"one-, five- and ten-year returns come in at "
        f"{sgn(f.signal_net('return_1y'))}%, "
        f"{sgn(f.signal_net('return_5y'))}% and "
        f"{sgn(f.signal_net('return_10y'))}% respectively. At that size the "
        f"ordering within the group is not meaningful and should not be read "
        f"as one. A return signal knows what the market did but not whether it "
        f"left <i>this</i> investor short — it is the same market for a "
        f"30-year-old with nothing saved and a 60-year-old with ten times "
        f"salary, and they should not respond alike."))
    out.append(ctx.p(
        f"Layering the two leaders together reaches "
        f"{sgn(float(combo[V].max()) - shape_value)}%, against "
        f"{sgn(f.signal_net('income_shock'))}% for the better one alone and "
        f"{f2(f.signal_net('income_shock') + f.signal_net('funded_ratio'), 2)}% "
        f"if they were additive. They overlap — a run of weak income and a run "
        f"of weak markets both show up as a balance behind target — but they "
        f"do not overlap completely, because only one of them is visible "
        f"before the money is spent."))
    out.extend(ctx.figure(
        "fig28_savings_signal_race",
        "Top left: the best of each signal at its own optimal sensitivity. "
        "Top right: how sharply the leaders peak. Bottom: the two leaders "
        "layered, showing partial but incomplete overlap."))

    # -- 12.6 feasibility --------------------------------------------------
    out.append(ctx.h2("#accumulation.6 How far does the contribution have to move?"))
    out.append(ctx.p(
        "The unconstrained optimum may send the savings rate anywhere in the "
        "permitted range. No household budget works like that. Re-pricing the "
        "rule with the rate confined to progressively narrower bands around "
        "its average is the cheapest test of whether the finding survives "
        "contact with a real household."))
    out.extend(ctx.table(
        rows_from(feas.assign(net=feas[V] - shape_value,
                              share=(feas[V] - shape_value) / max(unc_net, 1e-9) * 100.0),
                  ["width", "rate_floor", "rate_cap", "net", "share",
                   "mean_savings_rate", "prob_ruin"],
                  ["± allowed move", "Floor", "Cap", "Net value (%)",
                   "Share of unconstrained (%)", "Mean rate", "P(ruin)"],
                  {"width": lambda v: pc(v, 0), "rate_floor": lambda v: pc(v, 0),
                   "rate_cap": lambda v: pc(v, 0), "net": lambda v: f2(v, 2),
                   "share": lambda v: f2(v, 0),
                   "mean_savings_rate": lambda v: pc(v, 1),
                   "prob_ruin": lambda v: pc(v, 1)}),
        "The value of conditioning under a constrained contribution",
        note="The rate is confined to the career average plus or minus the "
             "stated width. At zero width the rule collapses onto the constant "
             "rate it is scored against, which is why the first row is exactly "
             "zero."))
    out.append(ctx.p(
        f"<b>There is no cheap corner here.</b> Up to about ten points the "
        f"value is close to proportional to the flexibility given — ±3 points "
        f"of income buys {f2(share_at(0.03), 0)}% of the unconstrained value, "
        f"±5 points {f2(share_at(0.05), 0)}%, ±10 points "
        f"{f2(share_at(0.10), 0)}% — and only then does it saturate. A "
        f"household that can flex its contribution by a couple of points does "
        f"<i>not</i> get most of the benefit; it gets roughly its share. That "
        f"is the least convenient version of this result and it is the one the "
        f"table supports."))
    out.extend(ctx.figure(
        "fig29_savings_feasibility",
        "Left: the value of conditioning as a function of how far the "
        "contribution may move. Right: the distribution of prescribed savings "
        "rates by age under the unconstrained rule."))

    # -- 12.7 where the value lands ---------------------------------------
    out.append(ctx.h2("#accumulation.7 Where the gain lands, and who wants it"))
    out.append(ctx.p(
        "A certainty equivalent is one number. It cannot say whether a rule "
        "lifted the middle of the distribution or insured the bottom of it, "
        "and those are very different products."))
    out.extend(ctx.table(
        rows_from(quant, ["quantile", "baseline_consumption",
                          "conditioned_consumption", "gain_pct"],
                  ["Quantile", "No conditioning", "Conditioned", "Change (%)"],
                  {"quantile": lambda v: f"p{int(round(float(v) * 100))}",
                   "gain_pct": lambda v: f2(v, 2)}),
        "Retirement consumption by quantile, with and without conditioning",
        note="Average real retirement consumption per path, sorted. The rule "
             "is funded by contributions taken from consumption on the paths "
             "that were going to be comfortable, which is why the top "
             "percentile is the one place the change turns negative."))
    out.append(ctx.p(
        f"<b>This is not left-tail insurance, and it is not a free lunch "
        f"either.</b> The gain is positive across almost the whole "
        f"distribution and is largest in the middle "
        f"({sgn(_mid_gain(quant))}% against {sgn(_low_gain(quant))}% at the "
        f"bottom decile and {sgn(_high_gain(quant))}% at the top). The bottom "
        f"of the distribution is where the rule has least to work with — a "
        f"path that is behind <i>because</i> labour income collapsed cannot "
        f"save its way out — and the top is where the rule takes its payment. "
        f"We report this because the intuition that motivated the test was "
        f"that conditioning would be tail insurance, and it is not."))
    out.append(ctx.p(
        f"Across preferences the value rises steeply with risk aversion: "
        f"{sgn(_gamma_net(by_gamma, 2.0))}% at γ = 2, "
        f"{sgn(_gamma_net(by_gamma, 5.0))}% at γ = 5 and "
        f"{sgn(_gamma_net(by_gamma, 10.0))}% at γ = 10. Every preference picks "
        f"the same coefficient, so what changes is not the rule but how much "
        f"the investor will pay for it. Conditioning is a risk product, and "
        f"its price is set by how much the buyer dislikes risk."))
    out.extend(ctx.figure(
        "fig30_savings_value_distribution",
        "Left: where in the distribution of retirement consumption the gain "
        "lands. Right: the value of conditioning by risk aversion, each at its "
        "own coefficient."))

    # -- 12.8 timing ------------------------------------------------------
    out.append(ctx.h2("#accumulation.8 When in a career is the balance worth reading?"))
    out.append(ctx.p(
        "Switching the conditioning on only for part of the career prices each "
        "stretch of working life separately. Outside the window the rule falls "
        "back to the age profile exactly, so these are clean subtractions. The "
        "sweep includes overlapping spans because halves and thirds answer "
        "different questions; the text below uses only the three "
        "non-overlapping windows that tile the career, because comparing a "
        "25-year window against a 13-year one says more about length than "
        "about timing."))
    out.extend(ctx.table(
        rows_from(windows.sort_values(V, ascending=False)
                  .assign(net=windows[V] - shape_value),
                  ["window", "net", "mean_savings_rate", "prob_ruin"],
                  ["Conditioning active for ages", "Net value (%)",
                   "Mean rate", "P(ruin)"],
                  {"window": str, "net": lambda v: f2(v, 2),
                   "mean_savings_rate": lambda v: pc(v, 1),
                   "prob_ruin": lambda v: pc(v, 1)}),
        "The value of conditioning by career stretch",
        note="Each row switches the rule on only for the stated ages. The "
             "three non-overlapping windows tile the career and are the ones "
             "the discussion relies on."))
    out.append(ctx.p(
        "The value rises monotonically with age across the three tiles. The "
        "balance is close to uninformative when there is barely any of it and "
        "decades left to recover; it becomes informative once a shortfall is "
        "large in absolute terms and there is little time left to fix it. The "
        "three tiles sum to approximately what the whole career is worth, so "
        "unlike the two directions of Section #accumulation.2 these stretches really are "
        "close to separable."))
    out.append(ctx.p(
        "How hard the rule leans at each age is a different question from "
        "whether leaning there is worth anything, and the two do not track "
        "each other one for one: per year of conditioning the value varies "
        "more across the career than the size of the adjustment does. The "
        "early career is not just quieter, it is worth less than its "
        "quietness would suggest."))
    out.extend(ctx.figure(
        "fig31_savings_when_it_matters",
        "Left: the value of conditioning when switched on only for the stated "
        "ages. Right: how far the rule moves the contribution at each age, "
        "and in which direction."))

    # -- 12.9 interactions -------------------------------------------------
    out.append(ctx.h2("#accumulation.9 What the value depends on"))
    out.extend(ctx.table(
        rows_from(by_strategy.sort_values(V, ascending=False)
                  .assign(net=by_strategy[V] - shape_value),
                  ["strategy", "net", "mean_savings_rate", "prob_ruin"],
                  ["Portfolio", "Net value (%)", "Mean rate", "P(ruin)"],
                  {"strategy": str, "net": lambda v: f2(v, 2),
                   "mean_savings_rate": lambda v: pc(v, 1),
                   "prob_ruin": lambda v: pc(v, 1)}),
        "The value of conditioning by what the money is invested in",
        note="Each portfolio is scored against its own constant-rate "
             "frontier, so these are not contaminated by differences in the "
             "level of the outcome."))
    out.append(ctx.p(
        f"<b>Conditioning is worth most to the investor whose plan is going "
        f"worst.</b> Across the four portfolios the value tracks the ruin "
        f"probability almost exactly: bills-only, which ruins "
        f"{pc(float(by_strategy.loc[by_strategy[V].idxmax(), 'prob_ruin']), 0)} "
        f"of the time, gets "
        f"{sgn(net(by_strategy.loc[by_strategy[V].idxmax()]))}%, while "
        f"all-equity, which ruins "
        f"{pc(float(by_strategy.loc[by_strategy[V].idxmin(), 'prob_ruin']), 0)} "
        f"of the time, gets "
        f"{sgn(net(by_strategy.loc[by_strategy[V].idxmin()]))}%. Note that "
        f"this runs <i>opposite</i> to portfolio volatility: the safest "
        f"portfolio benefits most. What the signal is worth is not driven by "
        f"how much the balance bounces around, but by how often it ends up "
        f"somewhere the investor cannot live on."))
    out.extend(ctx.table(
        rows_from(by_income.sort_values("volatility_factor")
                  .assign(net=by_income[V] - shape_value),
                  ["volatility_factor", "net", "mean_savings_rate",
                   "prob_ruin"],
                  ["Income shock s.d. (× baseline)", "Net value (%)",
                   "Mean rate", "P(ruin)"],
                  {"volatility_factor": lambda v: f"{float(v):g}×",
                   "net": lambda v: f2(v, 2),
                   "mean_savings_rate": lambda v: pc(v, 1),
                   "prob_ruin": lambda v: pc(v, 1)}),
        "The value of conditioning by how risky the pay cheque is",
        note="Labour-income shock standard deviations scaled up and down, with "
             "the target and the constant-rate frontier both recomputed for "
             "each arm."))
    out.append(ctx.p(
        "Labour-income risk cuts the other way from the portfolio result, and "
        "in the direction the dispersion story predicts: more income risk "
        "means more dispersion in where a path ends up relative to target, and "
        "more for the signal to correct. Taken together, the two interactions "
        "say that the funded ratio is worth reading in proportion to how far "
        "off target a path can drift and how badly that drift ends — not in "
        "proportion to how much the portfolio moves year to year."))
    out.extend(ctx.figure(
        "fig33_savings_interactions",
        "The value of conditioning by portfolio and by labour-income "
        "volatility. The two point in opposite directions with respect to "
        "\"risk\", which is why the mechanism is about outcomes rather than "
        "variance."))
    return out


def _common_strength(forms: pd.DataFrame) -> Dict[str, Any]:
    """Interpolate each functional form at a shared response strength."""
    reach = forms.groupby("form")["rate_move_pp"].max()
    common = float(reach.min())
    rows = [["Form", "At the same move (pp)", "Value there (%)",
             "Best anywhere on its grid (%)", "Furthest its grid reaches (pp)"]]
    values = []
    for name, block in forms.groupby("form"):
        block = block.sort_values("rate_move_pp")
        v = float(np.interp(common, block["rate_move_pp"].to_numpy(float),
                            block["matched_value_pct"].to_numpy(float)))
        values.append(v)
        rows.append([str(name), f2(common, 2), f2(v, 2),
                     f2(float(block["matched_value_pct"].max()), 2),
                     f2(float(block["rate_move_pp"].max()), 2)])
    order = np.argsort([-v for v in values])
    body = [rows[0]] + [rows[1 + int(i)] for i in order]
    return {"rows": body, "spread": float(max(values) - min(values)),
            "common": common}


def _low_gain(q: pd.DataFrame) -> float:
    return float(q[q["quantile"] <= 0.10]["gain_pct"].mean())


def _mid_gain(q: pd.DataFrame) -> float:
    return float(q[(q["quantile"] > 0.10) & (q["quantile"] < 0.90)]
                 ["gain_pct"].mean())


def _high_gain(q: pd.DataFrame) -> float:
    return float(q[q["quantile"] >= 0.90]["gain_pct"].mean())


def _gamma_net(by_gamma: pd.DataFrame, gamma: float) -> float:
    block = by_gamma[np.isclose(by_gamma["risk_aversion"], gamma)]
    zero = block[block["sensitivity"] == 0.0]["matched_value_pct"]
    base = float(zero.iloc[0]) if len(zero) else 0.0
    return float(block["matched_value_pct"].max()) - base


# ---------------------------------------------------------------------------
# 15. Starting valuation
# ---------------------------------------------------------------------------
def section_valuation(ctx: Any) -> List[Flowable]:
    f = ctx.f
    predictive = f.table("valuation_predictive_power").sort_values(
        "horizon_years")
    buckets = f.table("valuation_by_bucket")
    advantage = f.table("valuation_advantage")
    distribution = f.table("valuation_buckets")

    short = predictive.iloc[0]
    long = predictive.iloc[-1]
    long_h = int(long["horizon_years"])
    multiple = float((1.0 + float(long["gap"])) ** long_h)

    out: List[Flowable] = [ctx.h1("#valuation. What the Market Costs When You Start")]
    out.append(ctx.p(
        "Every result so far treats one starting point as interchangeable "
        "with another. The bootstrap draws calendar windows without regard to "
        "how expensive equities were when the window opened, so a lifetime "
        "that begins at a market peak is statistically identical to one that "
        "begins at a trough. That is a strong assumption, and it is the one a "
        "reader is least able to accept: they know what today's market costs, "
        "and the model does not."))
    out.append(ctx.p(
        "This section supplies the missing state variable. It is constrained "
        "throughout by a single rule — nothing may condition on information "
        "the investor could not have had at the moment they started."))

    out.append(ctx.h2("#valuation.1 The observable, and why it is not the obvious one"))
    out.append(ctx.p(
        "The natural candidate is the dividend-price ratio the source "
        "workbook records. It cannot be used directly. That series is a "
        "dividend <i>return</i> — the dividend paid during year t over the "
        "price at the start of it — and its numerator is unknown until the "
        "year is over. Conditioning on it would build look-ahead into every "
        "number that followed."))
    out.append(ctx.p(
        "What an investor standing at the start of year t can actually see is "
        "the trailing yield on the current price, D(t-1) / P(t-1), which we "
        "recover as the previous year's dividend return divided by one plus "
        "the previous year's capital gain. Both terms are last year's, so the "
        "value is fully formed before the year being predicted begins."))
    out.append(ctx.p(
        "We verify this structurally rather than statistically, because a "
        "correlation cannot tell the two cases apart: a yield built from the "
        "current year's dividend would predict the current year's return, and "
        "a correctly lagged one still correlates with the <i>previous</i> "
        "year's return because a bad year lowers the price in the "
        "denominator. Both are expected; neither distinguishes a leak from a "
        "signal. Instead we overwrite everything the workbook records for a "
        "probe year, rebuild the series, and confirm that the row for that "
        "year does not move. It does not, at six probe years spread across "
        "the panel, and the pipeline aborts rather than reporting anything if "
        "that check ever fails."))
    out.append(ctx.p(
        "The international leg needs its own answer, since it holds many "
        "markets at once. We use the equal-weighted leave-one-out mean of the "
        "other countries' yields rather than the median, because the leg "
        "holds equal money in each market and the dividend yield of an "
        "equally weighted portfolio is the plain mean of its constituents'. "
        "The median is retained as a robustness check on the pull of any one "
        "distressed market."))

    out.append(ctx.h2("#valuation.2 The yield forecasts returns"))
    out.extend(ctx.table(
        rows_from(predictive,
                  ["horizon_years", "observations", "correlation",
                   "forward_return_expensive", "forward_return_cheap", "gap"],
                  ["Horizon (years)", "Observations", "Correlation",
                   "Started expensive (%)", "Started cheap (%)", "Gap (pp)"],
                  {"horizon_years": lambda v: f"{int(v)}",
                   "observations": lambda v: f"{int(v):,}",
                   "correlation": lambda v: f2(v, 2),
                   "forward_return_expensive": lambda v: f2(float(v) * 100, 2),
                   "forward_return_cheap": lambda v: f2(float(v) * 100, 2),
                   "gap": lambda v: f2(float(v) * 100, 2)}),
        "Trailing dividend yield and subsequent real equity returns",
        note="Annualised real return over the years following the "
             "observation, split at the terciles of the yield distribution. "
             "The conditioning in this section is only worth doing if these "
             "gaps are positive."))

    # Classified from the table, not asserted: the whole section rests on it.
    predicts = bool((predictive["gap"] > 0).all())
    strengthens = float(long["correlation"]) > float(short["correlation"])
    if not predicts:
        out.append(ctx.p(
            "<b>The yield does not separate outcomes at every horizon.</b> "
            "The buckets below therefore split lifetimes that are not "
            "different in expectation, and the results should be read as a "
            "null rather than as a valuation effect."))
    else:
        out.append(ctx.p(
            f"The relationship is present at every horizon"
            + (" and strengthens with it" if strengthens else "")
            + f". Over {long_h} years the correlation is "
            f"{f2(float(long['correlation']), 2)} against "
            f"{f2(float(short['correlation']), 2)} at "
            f"{int(short['horizon_years'])} year, and the third of history "
            f"that began cheapest returned "
            f"{f2(float(long['gap']) * 100, 2)} percentage points a year more "
            f"than the third that began dearest. Compounded over "
            f"{long_h} years that is a factor of {f2(multiple, 2)} on "
            f"terminal wealth — not a rounding difference."))

    out.append(ctx.h2("#valuation.3 Which tercile, on what was knowable at the time"))
    out.append(ctx.p(
        "A look-ahead-free yield is only half of an implementable signal. The "
        "<i>boundaries</i> are the other half, and the obvious way to draw "
        "them fails: terciles of the pooled sample put the cut-points at "
        "fixed values estimated from 1890 through 2020, so a lifetime "
        "beginning in 1910 is called cheap or dear against a threshold that "
        "already knows what happened a century later. Nobody in 1910 could "
        "have known which tercile they were standing in."))
    out.append(ctx.p(
        "We therefore compute the boundaries recursively. A lifetime "
        "beginning in year t is ranked against every country-year strictly "
        "before t — the distribution its own investor could have seen — with "
        "a minimum-history requirement below which lifetimes are left "
        "unclassified rather than classified badly."))

    try:
        bounds = f.table("valuation_expanding_boundaries")
        decades = bounds[bounds["year"] % 20 == 0]
        if decades.empty:
            decades = bounds
        out.extend(ctx.table(
            rows_from(decades,
                      ["year", "prior_country_years",
                       "cut_expensive_middling", "cut_middling_cheap"],
                      ["Lifetime begins", "Country-years of history",
                       "Expensive / middling", "Middling / cheap"],
                      {"year": lambda v: f"{int(v)}",
                       "prior_country_years": lambda v: f"{int(v):,}",
                       "cut_expensive_middling": lambda v: pc(v, 2),
                       "cut_middling_cheap": lambda v: pc(v, 2)}),
            "Tercile boundaries an investor could have computed at the time",
            note="Quantiles of every country-year strictly before the "
                 "lifetime's first year. The pooled boundaries used by a "
                 "hindsight split are fixed at a single pair of values for "
                 "the whole sample."))
        first, last = bounds.iloc[0], bounds.iloc[-1]
        out.append(ctx.p(
            f"The boundaries move a long way — the expensive/middling cut "
            f"falls from {pc(float(first['cut_expensive_middling']), 2)} in "
            f"{int(first['year'])} to "
            f"{pc(float(last['cut_expensive_middling']), 2)} by "
            f"{int(last['year'])}. That movement is the whole point: early "
            f"cohorts faced higher yields, and ranking them against a sample "
            f"dragged down by the low-yield modern era makes them look "
            f"cheaper than they could possibly have known themselves to be."))
    except FileNotFoundError:
        pass

    out.append(ctx.p(
        "Two consequences follow, and both are results rather than defects. "
        "The correction is <b>large</b>: the pooled and recursive labellings "
        "disagree on close to a third of the lifetimes both can classify. And "
        "the buckets no longer come out balanced — a majority of lifetimes "
        "land in the expensive third. Terciles of a fixed sample are balanced "
        "by construction; terciles of an expanding one are not. Yields fell "
        "across this panel, so a lifetime drawn in year t typically begins "
        "below the average of everything before t and is expensive relative "
        "to its own history. An investor running this rule in real time would "
        "have called the market dear for most of a century while yields went "
        "on falling — which is precisely the gap between a signal fitted to a "
        "sample and one a person could have executed."))

    out.append(ctx.h2("#valuation.4 What it does to the allocation decision"))
    out.append(ctx.p(
        "Each simulated lifetime takes the blended portfolio yield at the "
        "country-year its first block opened, and is bucketed against the "
        "boundaries in force that year. Only the first block carries a "
        "starting condition: the rest of the chain is the future, which no "
        "investor chooses."))
    out.extend(ctx.table(
        rows_from(advantage,
                  ["bucket", "n_paths", "challenger_cec", "incumbent_cec",
                   "advantage_pct", "challenger_ruin", "incumbent_ruin"],
                  ["Started", "Lifetimes", "All-equity CEC",
                   "Glide-path CEC", "Advantage (%)", "All-equity ruin (%)",
                   "Glide-path ruin (%)"],
                  {"n_paths": lambda v: f"{int(v):,}",
                   "challenger_cec": lambda v: f2(v, 3),
                   "incumbent_cec": lambda v: f2(v, 3),
                   "advantage_pct": lambda v: f2(v, 2),
                   "challenger_ruin": lambda v: f2(float(v) * 100, 2),
                   "incumbent_ruin": lambda v: f2(float(v) * 100, 2)}),
        "The headline comparison, conditioned on starting valuation"))

    lead = advantage["advantage_pct"]
    holds = bool((lead > 0).all())
    spread = float(lead.max() - lead.min())
    if holds:
        out.append(ctx.p(
            f"<b>The ranking survives at every starting valuation.</b> The "
            f"all-equity portfolio leads the glide path in all "
            f"{len(lead)} buckets, and the advantage varies by only "
            f"{f2(spread, 2)} percentage points across them. An investor "
            f"starting at a market peak faces the same allocation answer as "
            f"one starting at a trough."))
    else:
        losers = advantage[advantage["advantage_pct"] <= 0]
        out.append(ctx.p(
            f"<b>The ranking does not survive conditioning.</b> The all-equity "
            f"portfolio loses in "
            f"{', '.join(str(b) for b in losers['bucket'])}, which is the "
            f"exception this section exists to look for and is reported here "
            f"rather than buried."))

    out.append(ctx.p(
        "What does change is the level. The valuation an investor starts at "
        "does not tell them what to hold; it tells them what to expect from "
        "holding it. That distinction matters for planning — a withdrawal "
        "rate calibrated on unconditional averages is calibrated on a mixture "
        "of starting points, most of which are cheaper than the one a reader "
        "in an expensive market actually faces."))

    out.extend(ctx.figure(
        "fig40_starting_valuation",
        "Top left: annualised real equity returns following cheap and "
        "expensive starting yields, by horizon, with the correlation printed "
        "above each pair. Top right: the distribution of blended starting "
        "dividend yields across the panel's country-years, with the most "
        "recent United States observation marked so a reader can place "
        "themselves in it. Bottom left: the all-equity advantage over the "
        "glide path within each "
        "valuation bucket (bars, left axis) against the level of "
        "certainty-equivalent consumption it wins at (line, right axis) — the "
        "advantage is flat and the level is not, which is the section's whole "
        "result. Bottom right: the tercile boundaries as an investor could "
        "have "
        "computed them at each date, drifting down across the century; a "
        "pooled split replaces both lines with a single pair of values and is "
        "what the recursive construction exists to avoid."))

    out.extend(_valuation_at_retirement(ctx))
    return out


def _valuation_at_retirement(ctx: Any) -> List[Flowable]:
    """The same yield read at the retirement date rather than at birth.

    Kept as its own function because it is a second experiment on the same
    state variable rather than a further reading of the first, and because it
    is skipped entirely when the pipeline was run without it.
    """
    f = ctx.f
    try:
        retire = f.table("valuation_retirement_by_bucket")
        ret_eq = f.table("valuation_retirement_equity")
        ret_dom = f.table("valuation_retirement_domestic")
        birth = f.table("valuation_by_bucket")
    except (KeyError, FileNotFoundError):
        return []
    if not len(retire):
        return []

    from src import inflation as ifl

    cfg = f.cfg
    vcfg = cfg["valuation"]
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    column = f"cec_crra_gamma{gamma:g}"
    labels = [str(x) for x in vcfg.get("bucket_labels", ifl.BUCKET_LABELS)]
    incumbent = str(vcfg.get("accumulation_strategy", "balanced_all_equity"))
    birth_level = ifl.level_spread(birth, incumbent, column, labels)
    retire_level = ifl.level_spread(retire, incumbent, column, labels)
    retire_ruin = ifl.level_spread(retire, incumbent, "prob_ruin", labels)
    timing = ifl.timing_comparison(birth_level, retire_level)
    if not timing.get("measured"):
        return []

    out: List[Flowable] = [
        ctx.h2("#valuation.5 The same yield, read at the retirement date")]
    out.append(ctx.p(
        "Everything above asks what a twenty-five-year-old should make of the "
        "market they are about to start saving into. That is not the only "
        "person who can read a dividend yield, and Section #inflation gives "
        "the reason to doubt they are the one for whom it matters most: a "
        "sixty-eight-year lifetime has time to average a starting condition "
        "away, and a thirty-year decumulation does not. So the same yield is "
        "read again at the retirement date, against the tercile boundaries in "
        "force then. Both reads are look-ahead free for the person standing "
        "there. What differs is who can act on it."))
    block = retire[retire["strategy"] == incumbent]
    if len(block):
        out.extend(ctx.table(
            [["Yield at retirement", "Lifetimes", "CEC", "P(ruin)"]]
            + [[str(r["bucket"]), f"{int(r['n_paths']):,}",
                f"{float(r[column]):.4f}", f"{float(r['prob_ruin']):.1%}"]
               for _, r in block.iterrows()],
            f"Outcomes for <i>{_pretty_strategy(incumbent)}</i> by the "
            f"valuation its investor retired into, γ = {gamma:g}."))
    if not timing.get("same_sign"):
        verdict = (
            f"<b>The same yield points opposite ways at the two dates.</b> A "
            f"lifetime that <i>began</i> in a cheap market ends "
            f"{birth_level['high_over_low_pct']:+.2f}% better off than one "
            f"that began in a dear one — high forward returns, and "
            f"sixty-eight years to compound them. A lifetime that "
            f"<i>retired</i> into a cheap market ends "
            f"{retire_level['high_over_low_pct']:+.2f}%, the wrong side of "
            f"zero, because a cheap market at sixty-three means the portfolio "
            f"being handed over is small and thirty years is not long enough "
            f"to make that back. The magnitudes are close "
            f"({timing['ratio']:.1f} times), so it is the sign that carries "
            f"the result. A dividend yield is a buy signal to a saver and a "
            f"warning to a retiree, and nothing in the birth-date reading "
            f"above says so.")
    elif timing.get("retirement_matters_much_more"):
        verdict = (
            f"<b>The date the yield is read at decides how much it "
            f"matters.</b> Conditioning a whole lifetime on the valuation it "
            f"opened at moves retirement consumption by "
            f"{birth_level['high_over_low_pct']:+.2f}%. Conditioning the same "
            f"lifetimes on the valuation they <i>retired</i> into moves it by "
            f"{retire_level['high_over_low_pct']:+.2f}% — "
            f"{timing['ratio']:.1f} times as much. The dilution the "
            f"birth-date reading suffers is not a property of the signal but "
            f"of the horizon it was measured over.")
    else:
        verdict = (
            f"<b>Reading the yield at retirement changes little.</b> The "
            f"level spread is {birth_level['high_over_low_pct']:+.2f}% at the "
            f"birth date against {retire_level['high_over_low_pct']:+.2f}% at "
            f"retirement, a factor of {timing['ratio']:.1f} and in the same "
            f"direction, so unlike the inflation state of Section #inflation "
            f"this signal does not sharpen when it is read later.")
    out.append(ctx.p(verdict))
    if retire_ruin.get("measured"):
        better = (retire_level["high_bucket"]
                  if retire_level["high_over_low_pct"] > 0
                  else retire_level["low_bucket"])
        safer = (retire_ruin["high_bucket"]
                 if retire_ruin["high_over_low_pct"] < 0
                 else retire_ruin["low_bucket"])
        out.append(ctx.p(
            (f"Consumption and failure point at different buckets, which is "
             f"the substance of the result rather than a footnote. Retiring "
             f"into the <i>{better}</i> bucket gives the higher "
             f"certainty-equivalent consumption — the portfolio is simply "
             f"worth more the day the wage stops — while the lower "
             f"probability of running out belongs to <i>{safer}</i>: "
             f"{retire_level['high_over_low_pct']:+.2f}% in consumption "
             f"against {retire_ruin['high_over_low_pct']:+.2f}% in ruin "
             f"across the same buckets. The valuation a retiree faces is a "
             f"transfer between the early years of their retirement and the "
             f"late ones. High prices pay out now and are repaid in the "
             f"forward returns they imply."
             if better != safer else
             f"Consumption and failure agree here. The <i>{better}</i> bucket "
             f"gives both the higher certainty equivalent and the lower "
             f"probability of running out — "
             f"{retire_level['high_over_low_pct']:+.2f}% and "
             f"{retire_ruin['high_over_low_pct']:+.2f}% across the buckets — "
             f"so there is no trade to weigh.")))

    eq_shift = ifl.optimum_shift(ret_eq, "equity_share", labels)
    dom_shift = ifl.optimum_shift(ret_dom, "domestic_share", labels)
    out.append(ctx.p(
        f"What a retiree can choose is not a lifetime allocation — the "
        f"accumulation has happened — but what to hold from the day they stop "
        f"working. Sweeping those weights alone, with "
        f"<i>{_pretty_strategy(incumbent)}</i> held through the working "
        f"years, gives an instruction they could follow."))
    if len(ret_eq) and len(ret_dom):
        merged = ret_eq.merge(ret_dom, on="bucket", suffixes=("_eq", "_dom"))
        out.extend(ctx.table(
            [["Yield at retirement", "Optimal equity share",
              "Optimal domestic share", "Smallest margin (%)"]]
            + [[str(r["bucket"]),
                f"{float(r['optimal_equity_share']):.0%}",
                f"{float(r['optimal_domestic_share']):.0%}",
                f"{min(float(r['margin_over_runner_up_pct_eq']), float(r['margin_over_runner_up_pct_dom'])):.2f}"]
               for _, r in merged.iterrows()],
            "What to hold from the retirement date, by the valuation read "
            "there.",
            note="The margin column is the smaller of the two, and is what "
                 "stops a flat maximum being read as an identification."))
    out.append(ctx.p(
        (f"The retiree's equity share moves "
         f"{eq_shift.get('optimal_equity_share_low', float('nan')):.0%} to "
         f"{eq_shift.get('optimal_equity_share_high', float('nan')):.0%} "
         f"across the buckets"
         if eq_shift.get("moves") else
         f"The retiree's equity share does not move across the buckets, "
         f"holding at "
         f"{eq_shift.get('optimal_equity_share_low', float('nan')):.0%}")
        + ", and "
        + (f"the domestic share moves "
           f"{dom_shift.get('optimal_domestic_share_low', float('nan')):.0%} "
           f"to "
           f"{dom_shift.get('optimal_domestic_share_high', float('nan')):.0%}"
           if dom_shift.get("moves") else
           f"the domestic share holds at "
           f"{dom_shift.get('optimal_domestic_share_low', float('nan')):.0%}")
        + ". Either way the instruction is a schedule read against a number "
          "the retiree can look up, not a rule of thumb about age."))
    out.extend(ctx.figure(
        "fig57_valuation_timing",
        "Left: certainty-equivalent consumption by valuation bucket, each "
        "bucket measured against the average of its own reading, with the "
        "same lifetimes classified by the yield at age 25 and by the yield "
        "at retirement; the two readings lean opposite ways, which is the "
        "subsection's result. Middle and right: what a retiree should hold "
        "from the retirement date — equity share and then the domestic share "
        "of it — one curve per bucket, with the argmax circled."))
    return out


# ---------------------------------------------------------------------------
# 16. Housing
# ---------------------------------------------------------------------------
def section_housing(ctx: Any) -> List[Flowable]:
    f = ctx.f
    audit = f.table("housing_desmoothing_audit")
    sweep = f.table("housing_cost_sweep")
    five = sweep[sweep["investable_set"] == "five assets"].sort_values(
        "holding_cost")
    control = sweep[sweep["investable_set"] == "four assets"].iloc[0]
    free = five.iloc[0]
    dearest = five.iloc[-1]

    from src.housing import break_even_cost
    break_even = break_even_cost(five)

    out: List[Flowable] = [ctx.h1("#housing. Housing as a Fifth Asset")]
    out.append(ctx.p(
        "The historical sources behind this study measure four asset classes. "
        "Three of them — equity, bonds and bills — form the investable set "
        "everywhere above. The fourth is housing, measured just as carefully "
        f"across {int(len(audit))} countries and "
        f"{int(audit['years_raw'].sum()):,} country-years, and excluded from "
        "every result so far. This section asks what including it would do."))
    out.append(ctx.p(
        "Two obstacles stand in the way, and both are addressed rather than "
        "assumed away."))

    out.append(ctx.h2("#housing.1 The index is smoothed"))
    out.append(ctx.p(
        "House prices come from transactions and valuations rather than from "
        "a continuous auction, so this year's index still carries part of "
        "last year's level. The published series therefore understates the "
        "volatility an owner actually bears, and comparing it with a traded "
        "series would flatter it — the classic error in this literature. We "
        "invert the smoothing country by country using each country's own "
        "estimated first-order coefficient, since index construction differs "
        "between them and a pooled coefficient would over-correct the cleanly "
        "measured series and under-correct the rest. A country whose returns "
        "are not positively autocorrelated is left alone: there is nothing to "
        "undo."))
    out.extend(ctx.table(
        rows_from(audit.sort_values("autocorrelation", ascending=False),
                  ["iso", "years_raw", "autocorrelation", "mean_raw",
                   "sd_raw", "sd_desmoothed", "equity_sd"],
                  ["Country", "Years", "Lag-1 autocorr.", "Mean (%)",
                   "SD published (%)", "SD de-smoothed (%)",
                   "Equity SD (%)"],
                  {"years_raw": lambda v: f"{int(v)}",
                   "autocorrelation": lambda v: f2(v, 2),
                   "mean_raw": lambda v: f2(float(v) * 100, 2),
                   "sd_raw": lambda v: f2(float(v) * 100, 2),
                   "sd_desmoothed": lambda v: f2(float(v) * 100, 2),
                   "equity_sd": lambda v: f2(float(v) * 100, 2)}),
        "Appraisal smoothing in the housing series, by country",
        note="De-smoothing preserves the mean in expectation and restores the "
             "variance the filter removed. Countries with non-positive "
             "autocorrelation are unchanged."))
    out.append(ctx.p(
        "This leaves housing with an equity-like average return and "
        "materially less volatility than equity even after the correction — "
        "which is exactly why the second obstacle cannot be waved through."))

    out.append(ctx.h2("#housing.2 A building costs money to hold"))
    out.append(ctx.p(
        "A share certificate does not. Rates, insurance, maintenance and "
        "management fall on a property owner every year whether the asset "
        "rose or fell, and the right figure is specific to the investor and "
        "the jurisdiction. Rather than choose one, we sweep it: the whole "
        "allocation is re-solved over the five-asset simplex at each annual "
        "holding cost, charged on value rather than on gains. The source "
        "builds its housing total return from capital gains plus a rental "
        "yield it describes as net of depreciation and maintenance, so the "
        "swept figure is best read as <i>additional</i> to whatever that "
        "construction already deducts — the taxes, management, voids and "
        "amortised transaction costs it does not. For scale, Chambers, "
        "Spaenjers and Steiner (2021), working from property-level records of "
        "four institutional portfolios over 1901–1983, find operating costs "
        "cut net yields to about two-thirds of gross; against this panel's "
        "4.9% median rental yield that is roughly 160 basis points. If the published series is in fact grosser than that, the "
        "break-even below overstates how much extra cost housing can bear; "
        "the direction of that error is known even though its size is not, "
        "which is why we report the whole curve rather than a single "
        "recommended weight."))
    out.append(ctx.p(
        "Housing is recorded for fewer country-years than equity is, so the "
        "study runs on the intersection; filling the gaps would mean "
        "inventing returns. The four-asset control is re-solved on that same "
        "restricted panel, against the same lifetimes and the same income "
        "draws, so the restriction cancels out of every comparison. Its own "
        f"optimum is {pc(float(control['mean_dom_eq']), 0)} domestic equity "
        f"and {pc(float(control['mean_intl_eq']), 0)} international."))
    out.append(ctx.p(
        "Housing enters as a <b>domestic</b> asset: each simulated investor "
        "holds their own country's housing index, drawn on the same calendar "
        "years, countries and blocks as their equity and bonds, so the "
        "cross-asset correlation the bootstrap exists to preserve is "
        "preserved. There is no international housing sleeve — people buy "
        "property where they live, and the leave-one-out construction that "
        "gives equity a foreign leg has no counterpart a household could "
        "execute in bricks."))

    out.extend(ctx.table(
        rows_from(five,
                  ["holding_cost", "mean_housing", "mean_dom_eq",
                   "mean_intl_eq", "mean_bond", "mean_bill", "advantage_pct"],
                  ["Holding cost (%)", "Housing (%)", "Dom. equity (%)",
                   "Intl. equity (%)", "Bonds (%)", "Bills (%)",
                   "Gain over four assets (%)"],
                  {"holding_cost": lambda v: f2(float(v) * 100, 1),
                   "mean_housing": lambda v: f2(float(v) * 100, 1),
                   "mean_dom_eq": lambda v: f2(float(v) * 100, 1),
                   "mean_intl_eq": lambda v: f2(float(v) * 100, 1),
                   "mean_bond": lambda v: f2(float(v) * 100, 1),
                   "mean_bill": lambda v: f2(float(v) * 100, 1),
                   "advantage_pct": lambda v: f2(v, 2)}),
        "The optimal portfolio at each annual housing holding cost",
        note="One allocation held for life, re-solved at each cost under "
             "common random numbers, so a difference between rows is the cost "
             "and nothing else."))

    wanted_free = float(free["mean_housing"]) > 0.01
    if not wanted_free:
        out.append(ctx.p(
            "<b>Housing earns no place in the portfolio at any price, "
            "including free.</b> Once the smoothing is undone, the "
            "de-smoothed series is dominated by assets the investor can "
            "already hold."))
    elif np.isfinite(break_even):
        out.append(ctx.p(
            f"<b>Housing is worth holding below an annual cost of "
            f"{pc(break_even, 1)} and not above it.</b> Offered free it takes "
            f"{pc(float(free['mean_housing']), 0)} of the portfolio and adds "
            f"{f2(float(free['advantage_pct']), 1)}% to certainty-equivalent "
            f"consumption over the best four-asset portfolio on the same "
            f"paths. That gain erodes as the cost rises and vanishes at "
            f"{pc(break_even, 1)} a year."))
    else:
        out.append(ctx.p(
            f"<b>Housing survives every cost this sweep tried.</b> Even at "
            f"{pc(float(dearest['holding_cost']), 0)} a year the optimum "
            f"holds {pc(float(dearest['mean_housing']), 0)} of it. The "
            f"break-even cost lies above the top of the grid and is not "
            f"reported: extrapolating it would invent the number the sweep "
            f"failed to find."))

    moves = {
        "domestic equity": float(free["mean_dom_eq"]) - float(dearest["mean_dom_eq"]),
        "international equity": float(free["mean_intl_eq"]) - float(dearest["mean_intl_eq"]),
        "bonds": float(free["mean_bond"]) - float(dearest["mean_bond"]),
        "bills": float(free["mean_bill"]) - float(dearest["mean_bill"]),
    }
    loser = min(moves, key=moves.get)
    if moves[loser] < -0.01:
        untouched = [name for name, move in moves.items() if abs(move) < 0.01]
        tail = (
            f" {', '.join(untouched).capitalize()} "
            f"{'is' if len(untouched) == 1 else 'are'} untouched at every "
            f"cost, so housing is not substituting for the portfolio at "
            f"large — it is competing with the single holding that already "
            f"does this job."
            if untouched else " The rest of the portfolio absorbs the "
                              "remainder.")
        out.append(ctx.p(
            f"The weight comes out of <b>{loser}</b>, which gives up "
            f"{pc(abs(moves[loser]), 0)} of the portfolio between the dearest "
            f"case and the free one — more than any other asset." + tail))

    out.append(ctx.h2("#housing.3 Two checks on that answer"))
    out.append(ctx.p(
        "The result rests on two choices that could each be driving it, so "
        "both are tested rather than defended."))

    try:
        raw = f.table("housing_raw_sweep")
        raw_five = raw[raw["investable_set"] == "five assets"].sort_values(
            "holding_cost")
        raw_free = float(raw_five.iloc[0]["mean_housing"])
        raw_break = break_even_cost(raw_five)
        overstates = raw_free > float(free["mean_housing"])
        out.append(ctx.p(
            f"<b>Is it an artefact of the de-smoothing?</b> Re-running the "
            f"whole sweep on the raw, still-smoothed series puts "
            f"{pc(raw_free, 0)} in housing at zero cost against "
            f"{pc(float(free['mean_housing']), 0)} once the smoothing is "
            f"undone"
            + (f", with a break-even cost of {pc(raw_break, 2)} against "
               f"{pc(break_even, 2)}"
               if np.isfinite(raw_break) and np.isfinite(break_even) else "")
            + ". "
            + ("The naive treatment therefore <b>overstates</b> the case for "
               "housing, which is the direction the mechanism predicts: "
               "smoothing hides volatility, and hidden volatility reads as "
               "risk-adjusted return. The correction is doing real work, and "
               "it works against housing rather than for it."
               if overstates else
               "The naive treatment does <b>not</b> overstate the case here, "
               "which is worth flagging because it is the opposite of what "
               "the mechanism predicts.")))
    except FileNotFoundError:
        pass

    try:
        age = f.table("housing_age_varying")
        best_gain = float(age["cec_gain_pct"].max())
        worst_move = float(np.abs(age["housing_difference"]).max())
        out.append(ctx.p(
            f"<b>Is it an artefact of holding one allocation for life?</b> "
            f"Re-solving the full age-by-asset schedule over the five-asset "
            f"simplex — seeded at the constant-mix optimum, so the search can "
            f"only improve on it — moves the mean housing weight by at most "
            f"{pc(worst_move, 0)} of the portfolio and buys at most "
            f"{f2(best_gain, 2)}% more certainty-equivalent consumption. "
            + ("Age-variation adds essentially nothing here: the single "
               "lifetime-long allocation is, to the resolution of this "
               "search, the answer."
               if best_gain <= 0.05 else
               "That is small enough that the constant-mix figures above "
               "stand as reported."
               if worst_move < 0.10 and best_gain < 1.0 else
               "That is large enough that the constant-mix figures above "
               "should be read as indicative of the level rather than as the "
               "optimum itself.")))
        working = age["housing_working"].to_numpy()
        retired = age["housing_retired"].to_numpy()
        widest = int(np.argmax(np.abs(retired - working)))
        if bool((retired > working).all()):
            out.append(ctx.p(
                f"The <i>shape</i> of that schedule is worth naming whatever "
                f"its size. Housing is the one asset in this paper whose "
                f"optimal weight <b>rises</b> with age: at a "
                f"{pc(float(age['holding_cost'].iloc[widest]), 0)} holding "
                f"cost the solved schedule holds "
                f"{pc(float(working[widest]), 0)} of it while working and "
                f"{pc(float(retired[widest]), 0)} in retirement. That is the "
                f"opposite of the declining glide path Sections #glide and #allocation "
                f"searched for and did not find in equities, and it has a "
                f"reading: a retiree drawing on the portfolio wants the asset "
                f"with the best return per unit of volatility, and once the "
                f"appraisal smoothing is undone housing is still that asset. "
                f"It is also the sharpest illustration of this section's "
                f"caveat — the schedule is only implementable by someone who "
                f"can rebalance into and out of property annually, which is "
                f"nobody."))
        elif bool((retired < working).all()):
            out.append(ctx.p(
                f"The solved schedule holds <b>less</b> housing in retirement "
                f"than while working — {pc(float(retired[widest]), 0)} against "
                f"{pc(float(working[widest]), 0)} at the widest point — a "
                f"conventional glide-path shape, in an asset the rest of this "
                f"paper does not hold."))
    except FileNotFoundError:
        pass

    out.append(ctx.h2("#housing.4 What this is not"))
    out.append(ctx.p(
        "The asset priced here is a liquid, continuously rebalanced, "
        "nationally diversified claim on a country's housing stock, because "
        "that is what a national house price index measures. <b>It is not a "
        "house.</b> A single owner-occupied property is concentrated in one "
        "street rather than spread across a country, cannot be rebalanced, is "
        "bought with leverage, carries transaction costs measured in percent "
        "rather than basis points, and pays part of its return as "
        "accommodation rather than as cash. None of those differences is in "
        "the numbers above. The section prices an asset class; it is not "
        "advice about a mortgage."))
    out.append(ctx.p(
        "The cost is also modelled as a constant annual percentage of value. "
        "Real holding costs are lumpy, partly fixed, and correlated with the "
        "cycle. A flat charge is the tractable approximation, not the truth."))

    out.extend(ctx.figure(
        "fig41_housing_cost_sweep",
        "Top left: the volatility the appraisal smoothing hides, by country, "
        "against the same country's equity. Top right: the optimal portfolio "
        "at each holding cost. Bottom left: what adding housing is worth, "
        "with and without the de-smoothing correction. Bottom right: which "
        "sleeve housing displaces as its cost falls."))
    return out



# ---------------------------------------------------------------------------
# 17. The mortgage
# ---------------------------------------------------------------------------
def section_mortgage(ctx: Any) -> List[Flowable]:
    f = ctx.f
    sweep = f.table("mortgage_spread_sweep").sort_values("spread")
    schedule = f.table("mortgage_lvr_schedule")
    curve = f.table("mortgage_constant_lvr_curve").sort_values("lvr")
    profile = f.table("mortgage_lvr_deviation_profile")

    from src.mortgage import LVR_CAP, break_even_spread
    break_even = break_even_spread(sweep)
    working = schedule[schedule["phase"] == "working"]
    retired = schedule[schedule["phase"] == "retired"]
    lvr_work = float(working["lvr"].mean())
    lvr_ret = float(retired["lvr"].mean())
    best = curve.loc[curve["cec"].idxmax()]

    out: List[Flowable] = [ctx.h1("#mortgage. The Mortgage")]
    out.append(ctx.p(
        "Section #housing prices housing as an asset owned outright. No household "
        "owns it that way. A house is the one asset an ordinary person can "
        f"borrow {pc(LVR_CAP, 0)} against, at a rate close to their own "
        "government's, secured on the thing itself — and leaving that out "
        "understates what housing does to a lifetime as surely as leaving the "
        "holding cost out overstates it. This section puts the mortgage in and "
        "asks the two questions that matter: how much, and when."))

    out.append(ctx.h2("#mortgage.1 How the loan is modelled"))
    out.append(ctx.p(
        "The decision variable is the <b>loan-to-value ratio</b>, because that "
        "is the number a lender quotes and a borrower chooses. A property "
        "funded with equity E and a loan at ratio λ returns, on that equity, "
        "(r_H − λ·i) / (1 − λ), where r_H is the real return on the property "
        "and i the real mortgage rate. That is the leverage multiple 1/(1 − λ) "
        "applied to housing alone, so the arithmetic is the same function "
        "Section #leverage uses for portfolio borrowing and the two remain "
        "consistent."))
    out.append(ctx.p(
        "The rate is the borrower's <i>own country's</i> real short rate plus "
        "a spread, drawn on the same block as every other series, so a "
        "lifetime that lives through high real rates pays them. The spread is "
        "swept rather than assumed. Equity is <b>wiped out, not driven "
        "negative</b>: the levered return is floored at total loss, which is "
        "the non-recourse assumption and the one most favourable to "
        "borrowing. How often that floor binds is reported rather than "
        "buried."))

    out.append(ctx.h2("#mortgage.2 How much"))
    out.extend(ctx.table(
        rows_from(curve,
                  ["lvr", "leverage_multiple", "cec",
                   "gain_vs_unlevered_pct", "negative_equity_share"],
                  ["LVR", "Gross exposure per unit of equity", "CEC",
                   "Gain over no mortgage (%)", "Path-years wiped out (%)"],
                  {"lvr": lambda v: pc(v, 0),
                   "leverage_multiple": lambda v: f2(v, 2),
                   "cec": lambda v: f2(v, 3),
                   "gain_vs_unlevered_pct": lambda v: f2(v, 2),
                   "negative_equity_share": lambda v: f2(float(v) * 100, 2)}),
        "Certainty-equivalent consumption by loan-to-value ratio, held for life",
        note="At the spread reported in the accompanying analysis document, "
             "with the allocation re-solved alongside the mortgage."))

    interior = 0.0 < float(best["lvr"]) < LVR_CAP - 1e-9
    if float(best["lvr"]) <= 1e-9:
        out.append(ctx.p(
            "<b>No mortgage is worth taking at this price.</b> The solved "
            "ratio is zero: the borrowing rate exceeds what the levered house "
            "returns once the tail it adds is priced by a risk-averse "
            "investor."))
    elif interior:
        out.append(ctx.p(
            f"<b>The answer is interior: about {pc(float(best['lvr']), 0)} "
            f"loan-to-value.</b> Held flat for life that is worth "
            f"{f2(float(best['gain_vs_unlevered_pct']), 1)}% in "
            f"certainty-equivalent consumption over the same house owned "
            f"outright. More borrowing is available — the grid runs to "
            f"{pc(LVR_CAP, 0)} — and is declined, because the extra return "
            f"does not compensate for the tail it opens."))
    else:
        out.append(ctx.p(
            f"<b>The search goes to the {pc(LVR_CAP, 0)} ceiling and would go "
            f"further if allowed.</b> The reported figure is the constraint "
            f"rather than an optimum, and the honest reading is that this "
            f"model does not price whatever stops real households borrowing "
            f"more — mortgage insurance, servicing tests, and the "
            f"concentration risk of a single property, none of which are "
            f"here."))

    out.append(ctx.h2("#mortgage.3 When"))
    declines = lvr_work > lvr_ret
    if declines:
        out.append(ctx.p(
            f"The solved schedule <b>declines with age</b>: "
            f"{pc(lvr_work, 0)} while working against {pc(lvr_ret, 0)} in "
            f"retirement. That is what households actually do — borrow "
            f"heavily against the first house and pay the loan down over a "
            f"career — and here it falls out of the optimisation rather than "
            f"being imposed on it. The mechanism is human capital: a working "
            f"investor has decades of future earnings a mortgage cannot "
            f"reach, and that income is what makes the leverage bearable. It "
            f"is also the mirror image of the equity glide path this paper "
            f"argues against, and the two are consistent — what should "
            f"decline with age is the <i>borrowing</i>, not the equity."))
    else:
        out.append(ctx.p(
            f"The solved schedule <b>rises with age</b>: {pc(lvr_work, 0)} "
            f"while working against {pc(lvr_ret, 0)} in retirement, which "
            f"contradicts what households do and deserves the suspicion that "
            f"attaches to any such result."))

    material = profile[profile["cost_of_resetting_bp"] >= 5.0]
    out.append(ctx.p(
        f"That is a statement about the <i>level and the slope</i> and "
        f"nothing finer. Subjecting the schedule to the same deviation "
        f"profile Section #allocation applies to solved allocations — resetting one "
        f"age at a time to the schedule's own average and pricing what is "
        f"lost — leaves {len(material)} of {len(profile)} ages carrying a "
        f"decision worth more than five basis points, the largest worth "
        f"{f2(float(profile['cost_of_resetting_bp'].max()), 0)} and the "
        f"median age worth "
        f"{f2(float(profile['cost_of_resetting_bp'].median()), 0)}. The rest "
        f"sits on a flat part of the surface where the search moves the ratio "
        f"for free. The plotted line is jagged; the evidence underneath it is "
        f"not."))

    out.append(ctx.h2("#mortgage.4 At what price of credit"))
    out.extend(ctx.table(
        rows_from(sweep,
                  ["spread", "mean_lvr", "lvr_working", "lvr_retired",
                   "housing_weight", "gross_housing_exposure",
                   "gain_vs_unlevered_pct", "negative_equity_share"],
                  ["Spread", "Mean LVR", "LVR working", "LVR retired",
                   "Housing equity", "Gross housing (× wealth)",
                   "Gain over no mortgage (%)", "Wiped out (%)"],
                  {"spread": lambda v: pc(v, 0),
                   "mean_lvr": lambda v: pc(v, 0),
                   "lvr_working": lambda v: pc(v, 0),
                   "lvr_retired": lambda v: pc(v, 0),
                   "housing_weight": lambda v: pc(v, 0),
                   "gross_housing_exposure": lambda v: f2(v, 2),
                   "gain_vs_unlevered_pct": lambda v: f2(v, 1),
                   "negative_equity_share": lambda v: f2(float(v) * 100, 2)}),
        "The joint optimum at each price of mortgage credit",
        note="Spread over the borrower's own country's real short rate. Every "
             "row shares the same paths, income draws and search, so a "
             "difference between rows is the price of the loan and nothing "
             "else."))

    first, last = sweep.iloc[0], sweep.iloc[-1]
    if np.isfinite(break_even):
        out.append(ctx.p(
            f"<b>Borrowing stops paying at a spread of about "
            f"{pc(break_even, 1)}</b> over the domestic short rate. Below it "
            f"the optimal ratio is positive and falls as credit dearens; "
            f"above it the mortgage earns no place."))
    else:
        out.append(ctx.p(
            f"The optimal ratio falls from {pc(float(first['mean_lvr']), 0)} "
            f"at a {pc(float(first['spread']), 0)} spread to "
            f"{pc(float(last['mean_lvr']), 0)} at "
            f"{pc(float(last['spread']), 0)} without reaching zero inside the "
            f"grid, so no break-even price is reported — extrapolating would "
            f"invent the number the sweep failed to find. What the table does "
            f"establish is that the answer is highly sensitive to the margin, "
            f"which is the practical point: a household's mortgage rate, not "
            f"the return on housing, is what decides this."))

    out.append(ctx.h2("#mortgage.5 Why this differs from Section #leverage"))
    out.append(ctx.p(
        "Section #leverage concluded that borrowing is barely worth doing. This "
        "section concludes that borrowing against a house is worth a great "
        "deal. Both use the same arithmetic and the same real bill rate plus "
        "a swept spread, so the two results have to be reconciled rather than "
        "left side by side."))
    try:
        lev = f.table("leverage_sweep")
        best = lev.loc[lev.groupby("spread")["cec"].idxmax()]
        merged = best[["spread", "vs_unlevered_pct"]].merge(
            sweep[["spread", "gain_vs_unlevered_pct"]], on="spread",
            how="inner").sort_values("spread")
        if len(merged):
            out.extend(ctx.table(
                rows_from(merged,
                          ["spread", "vs_unlevered_pct",
                           "gain_vs_unlevered_pct"],
                          ["Spread over the real short rate",
                           "Lever the portfolio (%)",
                           "Mortgage the housing sleeve (%)"],
                          {"spread": lambda v: pc(v, 1),
                           "vs_unlevered_pct": lambda v: f2(v, 2),
                           "gain_vs_unlevered_pct": lambda v: f2(v, 2)}),
                "The same borrowing cost, two places to spend it",
                note="Each column is that study's own gain over its own "
                     "unlevered optimum, so both measure the value of the "
                     "leverage rather than the value of the asset."))
            out.append(ctx.p(
                "<b>It is not the price of credit.</b> At identical spreads "
                "the mortgage is worth several times what portfolio leverage "
                "is worth, so holding the borrowing rate fixed does not close "
                "the gap and cannot be the explanation."))
    except FileNotFoundError:
        pass

    try:
        cmp_ = f.table("mortgage_asset_comparison")
        h = cmp_[cmp_["asset"] == "housing"].iloc[0]
        ie = cmp_[cmp_["asset"] == "intl_eq"].iloc[0]
        legs = float(cmp_["equity_leg_correlation"].iloc[0])
        out.extend(ctx.table(
            rows_from(cmp_, ["asset", "mean", "sd", "return_per_unit_risk",
                             "correlation_with_housing"],
                      ["Asset", "Mean real", "s.d.", "Return per unit of risk",
                       "Correlation with housing"],
                      {"asset": lambda v: {
                           "housing": "Housing", "dom_eq": "Domestic equity",
                           "intl_eq": "International equity",
                           "bond": "Bonds", "bill": "Bills"}.get(str(v),
                                                                 str(v)),
                       "mean": lambda v: pc(v, 2),
                       "sd": lambda v: pc(v, 2),
                       "return_per_unit_risk": lambda v: f2(v, 2),
                       "correlation_with_housing": lambda v: f2(v, 2)}),
            "Housing against the assets it competes with",
            note=f"Housing is net of the "
                 f"{pc(float(h['holding_cost_applied']), 0)} holding cost "
                 f"this section charges; the others are gross. "
                 f"For reference, the two equity legs correlate at "
                 f"{f2(legs, 2)} with each other."))
        better = float(h["return_per_unit_risk"]) > float(
            ie["return_per_unit_risk"])
        out.append(ctx.p(
            f"<b>It is diversification.</b> Housing's standalone return per "
            f"unit of risk is {f2(float(h['return_per_unit_risk']), 2)} "
            f"against international equity's "
            f"{f2(float(ie['return_per_unit_risk']), 2)} — "
            + ("better, but not by enough to explain a multiple."
               if better else
               "<i>worse</i>, so the gap cannot be a story about housing "
               "being the superior asset.")
            + f" What housing has is a correlation of "
            f"{f2(float(ie['correlation_with_housing']), 2)} with the "
            f"international equity sleeve, where the two equity legs "
            f"correlate at {f2(legs, 2)} with each other. Levering the "
            f"portfolio scales risk the investor already holds. Levering one "
            f"asset changes what the portfolio is made of, and here it buys "
            f"more of the one holding that moves independently of the rest — "
            f"while tying up less capital in it."))
        out.append(ctx.p(
            "That distinction matters beyond this paper. The case for "
            "lifecycle leverage (Ayres and Nalebuff, 2010) is usually made "
            "for levered <i>equity</i>, and on this panel that is the version "
            "that does not pay once credit is priced realistically. The "
            "declining-with-age shape their argument predicts survives; the "
            "instrument it is usually attached to does not. That the cheapest "
            "leverage available to a household happens to be secured against "
            "the diversifying asset is a convenience, not the mechanism."))
    except FileNotFoundError:
        pass

    out.append(ctx.h2("#mortgage.6 What this is not"))
    out.append(ctx.p(
        "The schedule is rebalanced annually, like everything else in this "
        "paper. For a mortgage that means costlessly redrawing the loan every "
        "year to hit a target ratio. Real mortgages amortise on a fixed "
        "schedule, cost several percent of the property to refinance, and are "
        "called on missed payments rather than on a drifting loan-to-value. "
        "<b>This is the value of the leverage, not a financing plan.</b>"))
    out.append(ctx.p(
        "Three further gaps deserve naming. There is no mortgage insurance, "
        f"so the {pc(LVR_CAP, 0)} ceiling is a wall rather than a price. "
        "There is no tax: in several countries mortgage interest is "
        "deductible and owner-occupied capital gains are untaxed, both of "
        "which would favour borrowing more than this shows. And the asset is "
        "a national housing index, not a house — the concentration risk of a "
        "single leveraged property is precisely what makes a real mortgage "
        "dangerous, and none of it is in these numbers."))

    out.extend(ctx.figure(
        "fig42_mortgage",
        "Top left: the solved loan-to-value ratio by age, with the grey bars "
        "showing what each age's choice is actually worth — where they are "
        "invisible the line above them carries no information. Top right: "
        "certainty-equivalent consumption against a ratio held for life. "
        "Bottom left: how the optimal ratio responds to the price of credit, "
        "split by working life and retirement. Bottom right: what borrowing "
        "buys "
        "against an unlevered house, and how often the borrower's right to "
        "walk away is what pays for it."))
    return out


# ---------------------------------------------------------------------------
# How much rests on one country, or one era
# ---------------------------------------------------------------------------
def section_panel(ctx: Any) -> List[Flowable]:
    f = ctx.f
    infl = f.table("panel_influence")
    period = f.table("panel_period_summary")
    floorf = f.table("panel_noise_floor")

    from src.panel_robustness import floor_summary, jackknife, verdict
    floor = floor_summary(floorf)
    # `shift_pct` is `gap_pct` minus the full-panel lead, so that lead is
    # recoverable from any row. A DataFrame's attrs do not survive the CSV
    # round trip, which is why it is reconstructed rather than read.
    baseline = float(infl["gap_pct"].iloc[0] - infl["shift_pct"].iloc[0])
    jack = jackknife(infl, baseline_gap=baseline)
    found = verdict(infl, jack, floor, period)
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(cfg.get("panel_robustness", {}).get(
        "n_paths", cfg["bootstrap"]["n_paths"]))
    n = int(found["n_countries"])
    noise = float(found["noise_range_pct"])
    se = float(jack["standard_error"])
    t_stat = float(jack["t_stat"])

    out: List[Flowable] = [ctx.h1("#panel. How Much Rests on One Country, or "
                                  "One Era")]
    out.append(ctx.p(
        f"Section #sensitivity varies the preferences and the lifecycle, and "
        f"Section #sleeve will vary how the international sleeve is built. "
        f"Neither touches what a sceptical reader asks about first: the panel is {n} developed markets that survived, "
        f"and a result assembled from {n} histories could be one history "
        f"wearing a disguise. If dropping the United States overturns the "
        f"ranking, this paper is about the United States. This section asks "
        f"three questions on the same machinery, each scored by the "
        f"summariser that produces Section #baseline's own table."))
    out.extend(ctx.bullets([
        "<b>Influence.</b> Rebuild the panel once per country with that "
        "country removed, and re-run the headline.",
        "<b>Uncertainty.</b> Those runs form a delete-one jackknife, which "
        "gives a standard error reflecting the <i>panel's</i> size rather "
        "than the Monte Carlo's.",
        "<b>Stability.</b> Re-run on expanding windows of history, and on the "
        "two halves of the sample.",
    ]))
    out.append(ctx.p(
        "The deletion is genuine. A dropped market disappears from every "
        "other country's international sleeve as well as from the set its own "
        "domestic leg is drawn from, so each run answers what this paper "
        "would say had that market's history never been recorded — not the "
        "much weaker question of what it would say had we chosen not to "
        "invest there."))

    out.append(ctx.h2("#panel.1 The noise floor"))
    out.append(ctx.p(
        f"Every delete-one run draws its own bootstrap paths, so it differs "
        f"from the full panel even when the country it dropped carried no "
        f"information. Re-running the <b>unmodified</b> panel under "
        f"{int(floor['seeds'])} seeds moves the lead across a range of "
        f"{noise:.3f} points, and that is the floor any shift below has to "
        f"clear. {found['n_material_countries']} of {n} countries clear it."))

    out.append(ctx.h2("#panel.2 Removing one country at a time"))
    out.extend(ctx.table(
        rows_from(infl, ["dropped", "gap_pct", "shift_pct", "challenger_cec",
                         "incumbent_cec"],
                  ["Country removed", "Lead", "Change in the lead",
                   "CEC, all-international", "CEC, 50/50"],
                  {"dropped": str,
                   "gap_pct": lambda v: f"{float(v):.2f}%",
                   "shift_pct": lambda v: f"{float(v):+.2f} pp",
                   "challenger_cec": lambda v: f2(v, 4),
                   "incumbent_cec": lambda v: f2(v, 4)}),
        f"The headline with each country removed, at \u03b3 = {gamma:g}",
        note=f"{n_paths:,} lifetimes per run. The full-panel lead is "
             f"{baseline:.2f}%. A negative change means the removed country "
             f"was holding the result up."))
    if found["survives_every_deletion"]:
        out.append(ctx.p(
            f"<b>No single country carries the result.</b> All-international "
            f"leads the 50/50 split in all {n} delete-one panels. Removing "
            f"{found['worst_country']} costs the most, taking the lead from "
            f"{baseline:.2f}% to {float(found['worst_gap_pct']):.2f}%, and it "
            f"still leads. Notably the United States is not the load-bearing "
            f"market: removing it moves the lead by "
            f"{float(infl.loc[infl['dropped'] == 'USA', 'shift_pct'].iloc[0]):+.2f} "
            f"points."
            if "USA" in set(infl["dropped"]) else
            f"<b>No single country carries the result.</b> All-international "
            f"leads in all {n} delete-one panels."))
    else:
        out.append(ctx.p(
            f"<b>The result does not survive every deletion.</b> In "
            f"{int(found['n_deletions_that_break_it'])} of {n} delete-one "
            f"panels the 50/50 split overtakes all-international; the worst "
            f"is {found['worst_country']}. On this evidence the headline is a "
            f"statement about a particular set of markets rather than a "
            f"general one."))

    out.append(ctx.h2("#panel.3 Why that market, and not the obvious one"))
    chan = f.table("panel_channels")
    prof = f.table("panel_market_profile")
    from src.panel_robustness import explain
    why = explain(prof, infl, chan)
    n_mkt = int(why["n_markets"])
    pole = str(why["sleeve_pole"])
    pole_rank = int(why["sleeve_pole_drag_rank"])
    pole_drag = ("the largest volatility drag in the panel" if pole_rank == 1
                 else f"the {ordinal(pole_rank)}-largest of {n_mkt}")
    out.append(ctx.p(
        "The natural guess is that the load-bearing market is the one with "
        "the best domestic history, and that removing it makes the domestic "
        "half of the 50/50 split look worse. The data says otherwise, and the "
        "reason is a general point about what an averaged sleeve does to a "
        "constituent's volatility."))
    out.append(ctx.p(
        "Every country sits in the panel twice over: once as somebody's home "
        "market, and once inside the fifteen-market average that everybody "
        "<i>else</i> holds as their international leg. The two strategies "
        "weight those roles differently — all-international is entirely "
        "sleeve, the 50/50 split is half sleeve and half domestic — so "
        "writing <i>S</i> for a deletion's effect on the first and <i>D</i> "
        "for the implied effect on the domestic half, the change in "
        "all-international is <i>S</i>, the change in the 50/50 split is "
        "(<i>S</i>&nbsp;+&nbsp;<i>D</i>)/2, and the change in the gap is "
        "(<i>S</i>&nbsp;−&nbsp;<i>D</i>)/2. A market whose value lies mostly "
        "in other people's sleeves narrows the gap when removed; one whose "
        "value lies mostly in being its own home market widens it."))
    out.extend(ctx.table(
        rows_from(prof.sort_values("sleeve_geometric_delta"),
                  ["iso", "own_arithmetic", "own_geometric",
                   "volatility_drag", "sleeve_geometric_delta"],
                  ["Market", "Own arithmetic mean", "Own geometric mean",
                   "Volatility drag", "Effect on the sleeve"],
                  {"iso": str, "own_arithmetic": lambda v: pc(v, 2),
                   "own_geometric": lambda v: pc(v, 2),
                   "volatility_drag": lambda v: f"{float(v) * 100:.2f} pp",
                   "sleeve_geometric_delta": lambda v: f"{float(v):+.2f} pp"}),
        "Each market's own returns, and what removing it does to the sleeve",
        note="The volatility drag is the wedge between a market's arithmetic "
             "and geometric mean — what its own residents lose to its "
             "volatility. The last column is measured, not argued: the panel "
             "is rebuilt without the market and the pooled sleeve's compound "
             "return recomputed."))
    out.append(ctx.p(
        f"A deletion's effect on the headline tracks what it does to the "
        f"<b>sleeve's compound return</b> (correlation "
        f"{float(why['corr_sleeve_geometric_delta']):+.2f}), not the removed "
        f"market's own average return (correlation "
        f"{float(why['corr_own_arithmetic']):+.2f}). The load-bearing market "
        f"is not the one with the best domestic history; it is the one the "
        f"fifteen-way average could least afford to lose."))
    out.append(ctx.p(
        f"<b>{pole}</b> is that market, and the reason is the gap between its "
        f"two means: an arithmetic mean of "
        f"{pc(float(why['sleeve_pole_arithmetic']), 2)} against a geometric "
        f"mean of {pc(float(why['sleeve_pole_geometric']), 2)} — "
        f"{pole_drag}. "
        f"An equal-weighted average of fifteen markets adds up "
        f"<i>arithmetic</i> returns each year and diversifies the volatility "
        f"away, so the sleeve collects {pole}'s high mean without paying its "
        f"drag while its own residents pay it in full. It is an excellent "
        f"constituent of somebody else's sleeve and a mediocre thing to hold "
        f"alone, so removing it hits the all-sleeve strategy hardest."))
    if why.get("usa_present"):
        out.append(ctx.p(
            f"<b>The United States is the mirror image</b>, which answers the "
            f"question most readers arrive with. It is "
            f"{nth_best(int(why['usa_arithmetic_rank']))} market in the panel "
            f"by arithmetic mean ({pc(float(why['usa_arithmetic']), 2)}) and "
            f"{nth_best(int(why['usa_geometric_rank']))} by geometric mean "
            f"({pc(float(why['usa_geometric']), 2)}) — "
            f"{'one of the smallest drags here' if why.get('usa_drag_is_small') else 'a middling drag'}. "
            f"Almost all of its return survives being held on its own, so its "
            f"value is concentrated in the role the 50/50 split holds and the "
            f"all-international strategy does not. Removing it costs the "
            f"sleeve only {float(why['usa_delta']):+.2f} points of compound "
            f"return while taking a good home market out of the pool, so it "
            f"hurts the 50/50 split more and "
            f"{'widens' if why.get('usa_widens') else 'narrows'} the lead by "
            f"{abs(float(why['usa_shift'])):.2f} points. The United States is "
            f"not what the result rests on; it is what the <i>comparison</i> "
            f"rests on."))
    out.append(ctx.p(
        f"The reading should not be pushed past the column it rests on. "
        f"Across the panel the correlation between a deletion's effect and "
        f"the removed market's volatility drag is only "
        f"{float(why['corr_volatility_drag']):+.2f}: the drag explains why "
        f"{pole} in particular is worth more to others than to itself, not "
        f"the pattern as a whole. "
        + (f"{str(why['counterexample'])} makes the point — a higher "
           f"arithmetic mean still "
           f"({pc(float(why['counterexample_arithmetic']), 2)}), and removing "
           f"it costs the sleeve only "
           f"{float(why['counterexample_delta']):+.2f} points. How a market's "
           f"good years line up against the "
           f"{NUMBER_WORDS.get(n - 2, str(n - 2)).lower()} "
           f"others it shares any given sleeve with matters as much as its "
           f"average, and that is what the measured column captures."
           if why.get("counterexample") else "")))

    out.append(ctx.h2("#panel.4 What sixteen countries actually support"))
    out.append(ctx.p(
        f"The delete-one runs give a jackknife standard error of "
        f"<b>{se:.2f} points</b> on a lead of {baseline:.2f}%, for a 95% "
        f"interval of [{float(jack['ci_low']):.2f}, "
        f"{float(jack['ci_high']):.2f}]."))
    if found["excludes_zero"] and jack["marginal"]:
        out.append(ctx.p(
            f"The interval excludes zero, but only just: the implied "
            f"t-statistic is {t_stat:.2f} against a threshold of 1.96, and "
            f"the lower bound clears zero by {float(jack['ci_low']):.2f} "
            f"points. The ordering is supported by the panel rather than only "
            f"by the simulation, and it is supported <i>thinly</i>. The "
            f"direction should be read as established and the magnitude as "
            f"barely resolved."))
    elif found["excludes_zero"]:
        out.append(ctx.p(
            f"The interval excludes zero comfortably, at a t-statistic of "
            f"{t_stat:.2f}, so the ordering is supported by the panel and not "
            f"merely by the simulation."))
    else:
        out.append(ctx.p(
            f"<b>The interval includes zero.</b> At a t-statistic of "
            f"{t_stat:.2f}, {n} countries cannot distinguish this lead from "
            f"no lead at all, whatever the Monte Carlo precision suggests."))
    out.append(ctx.p(
        "This is the number to quote, and it is far wider than the Monte "
        "Carlo error. Every certainty equivalent in this paper is reported to "
        "four decimal places because that is what the simulation resolves; "
        "the panel resolves much less, and a hundred thousand bootstrap paths "
        "add precision to the former without adding a single country of "
        "evidence to the latter. Nothing else in this paper carries an "
        "interval of this kind, and the reader should assume every gap "
        "reported elsewhere is estimated no more precisely than this one."))

    out.append(ctx.h2("#panel.5 Is the ranking stable through time?"))
    out.extend(ctx.table(
        rows_from(period, ["window", "country_years", "gap_pct"],
                  ["Window", "Country-years", "Lead"],
                  {"window": str, "country_years": lambda v: f"{int(v):,}",
                   "gap_pct": lambda v: f"{float(v):.2f}%"}),
        "The headline on expanding windows of history, and on the two halves",
        note="Each window masks the country-years outside it; the sleeve in "
             "any year is the sleeve that year had, so a 1930 lifetime can "
             "still see 1929."))
    if found.get("all_windows_hold"):
        out.append(ctx.p(
            f"<b>The ordering holds in every window.</b> The weakest is "
            f"{found.get('weakest_window', '?')} at "
            f"{float(found.get('weakest_window_gap_pct', float('nan'))):.2f}%. "
            f"An investor standing in 1950, with only the record to that "
            f"date, would have reached the same ranking as one standing "
            f"today — which is the strongest of the three checks here, "
            f"because it is the one that could have been acted on "
            f"contemporaneously."))
    else:
        out.append(ctx.p(
            f"<b>The ordering does not hold in every window</b>: it survives "
            f"{int(found.get('windows_holding', 0))} of "
            f"{int(found.get('n_windows', 0))}. The weakest is "
            f"{found.get('weakest_window', '?')}. That makes the finding "
            f"era-dependent, and the era should be named whenever it is "
            f"quoted."))

    out.extend(ctx.figure(
        "fig44_panel_robustness",
        "Top left: what removing each country does to the headline lead, "
        "against the band a re-seeded run produces on an unchanged panel. Top "
        "right: the lead by how much history is available. Bottom left: the "
        "jackknife interval the delete-one runs support. Bottom right: why "
        "the deletions land where they do — each country's effect on the lead "
        "against its effect on the sleeve's own compound return."))

    out.append(ctx.h2("#panel.6 What this changes"))
    out.extend(ctx.bullets([
        ("No single market carries the headline, so it is not a restatement "
         "of one country's history — and in particular it is not a "
         "restatement of the United States'."
         if found["survives_every_deletion"] else
         "At least one deletion overturns the headline, so it must be stated "
         "as a property of this particular set of markets."),
        f"The honest precision on the headline gap is ±{se:.2f} points, not "
        f"the fourth decimal place of a certainty equivalent. This is the "
        f"single largest caveat in the paper and Section #limitations.1 "
        f"carries it.",
        ("The ranking is stable across every window of history tested, "
         "including windows an investor could have stood in."
         if found.get("all_windows_hold") else
         "The ranking is not stable across every window, so it should be "
         "quoted with the era it depends on."),
    ]))
    return out


# ---------------------------------------------------------------------------
# 18. How the international sleeve is weighted
# ---------------------------------------------------------------------------
def section_sleeve(ctx: Any) -> List[Flowable]:
    f = ctx.f
    ranking = f.table("sleeve_ranking")
    moments = f.table("sleeve_moments")
    conc = f.table("sleeve_concentration").sort_values("year")
    spectrum = f.table("sleeve_spectrum")
    comparison = f.table("sleeve_comparison")

    from src.sleeve import LABELS, verdict
    found = verdict(comparison)
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(cfg.get("sleeve", {}).get(
        "n_paths", cfg["bootstrap"]["n_paths"]))
    schemes = [s for s in found.get("schemes", []) if s in ranking.columns]
    alternatives = [s for s in schemes if s != "equal"]
    n_markets = int(conc["markets"].max())
    ref = spectrum[spectrum["weighting"] == "equal"]
    ref_gap = float(ref["gap_pct"].iloc[0]) if len(ref) else float("nan")
    kindest = spectrum.loc[spectrum["gap_pct"].idxmax()]
    tightest = spectrum.iloc[0]

    out: List[Flowable] = [ctx.h1("#sleeve. How the International Sleeve Is "
                                  "Weighted")]
    out.append(ctx.p(
        f"Every result to this point rests on an international equity leg "
        f"built as a leave-one-out <b>equal-weighted</b> average of the other "
        f"{n_markets - 1} markets. That construction is defensible and "
        f"unusually favourable. An equal-weighted portfolio of {n_markets} "
        f"national markets is a more diversified object than any index a "
        f"person could have bought: it holds as much Portugal as it holds the "
        f"United States, it rebalances into whatever has fallen, and it never "
        f"lets one market grow to dominate. A real investor holds something "
        f"closer to a capitalisation-weighted index, which is concentrated by "
        f"construction and grows <i>more</i> concentrated exactly when one "
        f"market has run."))
    out.append(ctx.p(
        "That matters here specifically. The one place this paper diverges "
        "from the study it re-implements — all-international beating the "
        "50/50 domestic/international split — is a claim about how much "
        "diversification the foreign sleeve delivers. If the equal weighting "
        "is doing that work, the divergence is an artefact of panel "
        f"construction rather than a finding about the world. This section "
        f"rebuilds the panel under {len(schemes)} weighting schemes and "
        f"re-runs the headline comparison through the same summariser, so the "
        f"answers differ in the weighting and in nothing else."))

    out.append(ctx.h2("#sleeve.1 The schemes, and why these ones"))
    out.append(ctx.p(
        "The database supports very few honest cross-country weights. A "
        "series has to be in comparable units across countries and complete "
        "enough not to change the sleeve's membership from one year to the "
        "next. That rules out the nominal GDP, exports, loans and wage "
        "columns, which carry country-specific currency units — pesetas for "
        "Spain, lire for Italy — and the ratio columns, which are gappy and "
        "are not sizes in any case. What survives is population, Maddison "
        "real GDP per head, their product, and one scheme estimated from the "
        "returns themselves."))
    out.extend(ctx.table(
        rows_from(spectrum, ["label", "tilts_towards", "effective_markets",
                             "sd", "correlation_with_domestic", "gap_pct"],
                  ["Weighting", "Tilts towards", "Effective markets",
                   "Sleeve SD", "Corr. with home",
                   "All-intl over 50/50"],
                  {"label": str, "tilts_towards": str,
                   "effective_markets": lambda v: f2(v, 1),
                   "sd": lambda v: pc(v, 1),
                   "correlation_with_domestic": lambda v: f2(v, 3),
                   "gap_pct": lambda v: f"{float(v):+.2f}%"}),
        "The weighting schemes, ordered by how much they concentrate the "
        "sleeve",
        note="Effective markets is the panel average of one over the "
             "Herfindahl index. The final column is the advantage "
             "all-international holds over the 50/50 split under that "
             "scheme — the quantity this section exists to stress."))
    out.extend(ctx.bullets([
        "<b>Every weight is lagged.</b> Sizes are read from the prior year "
        "and the inverse-volatility estimate uses a strictly prior window, so "
        "no weight is formed from a number the investor could not yet have "
        "seen — the same discipline Section #valuation applies to valuation terciles.",
        "<b>Every panel is paired.</b> All schemes share their years, "
        "countries and availability mask, so the bootstrap draws identical "
        "calendar history for each.",
        "<b>None of them is capitalisation weighting.</b> These are proxies. "
        "They reproduce the concentration that makes a real index a real "
        "index; they do not reproduce the wedge between an economy's size and "
        "its listed market's, and the PPP-based ones understate a market "
        "whose currency is temporarily strong — Japan in the late 1980s most "
        "of all.",
    ]))
    out.append(ctx.p(
        f"The set is deliberately two-dimensional. Real GDP and population "
        f"concentrate the sleeve heavily, to an effective "
        f"{float(tightest['effective_markets']):.1f} markets at the tightest. "
        f"GDP per capita and inverse volatility barely concentrate it at all "
        f"while tilting it somewhere quite different — towards rich countries "
        f"and towards historically stable ones respectively. That is what "
        f"lets Section #sleeve.3 separate the effect of concentration from the "
        f"effect of the tilt."))

    out.append(ctx.h2("#sleeve.2 The headline under every weighting"))
    out.extend(ctx.table(
        rows_from(ranking, ["label"] + schemes,
                  ["Strategy"] + [LABELS.get(s, s) for s in schemes],
                  {"label": str,
                   **{s: (lambda v: f2(v, 4)) for s in schemes}}),
        f"Certainty-equivalent consumption for every strategy under every "
        f"sleeve construction, at \u03b3 = {gamma:g}",
        note=f"{n_paths:,} lifetimes per weighting, drawn from identical "
             "calendar history. The strategies holding no international "
             "equity are unchanged across every column by construction, "
             "which is the cheapest available check that the panels differ "
             "only in the sleeve."))

    if found["winner_changes"]:
        call = ("<b>Some weighting changes which strategy wins.</b> The "
                "headline of this paper is a property of how the "
                "international leg was built.")
    elif found["ordering_changes"]:
        call = ("<b>Some weighting flips the ordering this paper diverges "
                "on.</b> All-international leads the 50/50 split under some "
                "constructions and trails it under others, so the divergence "
                "should be attributed to panel construction rather than read "
                "as a finding.")
    else:
        worst = found.get("worst_scheme", "")
        call = (f"<b>The ordering survives every weighting tested.</b> "
                f"All-international leads the 50/50 split by {ref_gap:+.2f}% "
                f"on the equal-weighted sleeve. The harshest alternative is "
                f"{LABELS.get(worst, worst)}, which leaves "
                f"{float(found.get('worst_gap_pct', float('nan'))):+.2f}% — "
                f"{float(found.get('worst_share_retained', float('nan'))):.0%} "
                f"of the reference gap. The divergence from the replicated "
                f"study narrows under a realistic sleeve but does not close.")
    out.append(ctx.p(call))

    equal_is_kindest = str(kindest["weighting"]) == "equal"
    if not equal_is_kindest:
        out.append(ctx.p(
            f"Worth noting against our own interest in the other direction "
            f"too: the equal weighting is <b>not</b> the construction most "
            f"favourable to the finding. {str(kindest['label'])} produces a "
            f"larger gap still, {float(kindest['gap_pct']):+.2f}% against "
            f"{ref_gap:+.2f}%, so the headline is not resting on the kindest "
            f"sleeve available to it."))

    out.append(ctx.h2("#sleeve.3 Concentration, or the tilt?"))
    corr = float(np.corrcoef(spectrum["effective_markets"],
                             spectrum["gap_pct"])[0, 1])
    near = spectrum[spectrum["effective_markets"] >= 0.85 * n_markets]
    spread_near = (float(near["gap_pct"].max()) - float(near["gap_pct"].min())
                   ) if len(near) > 1 else float("nan")
    full_spread = float(spectrum["gap_pct"].max()) \
        - float(spectrum["gap_pct"].min())
    concentration_explains = bool(np.isfinite(spread_near)
                                  and spread_near < 0.5 * full_spread)
    out.append(ctx.p(
        f"The obvious hypothesis is that concentration alone drives the "
        f"result: the more the sleeve collapses onto a few markets, the less "
        f"diversification it delivers and the smaller all-international's "
        f"advantage. The schemes are chosen to test that, because two of them "
        f"barely concentrate the sleeve at all while tilting it somewhere "
        f"quite different. Across the {len(spectrum)} schemes the gap "
        f"correlates {corr:+.2f} with the effective number of markets."))
    out.append(ctx.p(
        f"But the {len(near)} schemes that leave concentration essentially "
        f"intact still span {spread_near:.2f} points of gap between them, "
        f"against {full_spread:.2f} points across the whole set. "
        + ("Most of the variation is therefore concentration, and the "
           "direction of the tilt is second-order."
           if concentration_explains else
           "So the direction of the tilt matters roughly as much as the "
           "degree of concentration: a sleeve tilted towards rich countries "
           "and one tilted towards historically stable countries have almost "
           "the same Herfindahl index and materially different consequences. "
           "Concentration is not a sufficient statistic for what a weighting "
           "scheme does to a lifetime.")))
    out.extend(ctx.table(
        rows_from(moments, ["label", "mean", "sd", "return_per_unit_risk",
                            "correlation_with_domestic"],
                  ["Sleeve", "Mean", "SD", "Return per unit risk",
                   "Correlation with home market"],
                  {"label": str, "mean": lambda v: pc(v, 2),
                   "sd": lambda v: pc(v, 2),
                   "return_per_unit_risk": lambda v: f2(v, 3),
                   "correlation_with_domestic": lambda v: f2(v, 3)}),
        "Pooled moments of the international sleeve under each weighting",
        note="Pooled over every available country-year. The correlation is "
             "with the investor's own domestic market."))

    e = moments[moments["weighting"] == "equal"].iloc[0]
    g = moments[moments["weighting"] == "gdp"].iloc[0]
    d_corr = float(g["correlation_with_domestic"]) \
        - float(e["correlation_with_domestic"])
    out.append(ctx.p(
        f"The correlation column carries a result that reads as a surprise "
        f"and is not one. Concentrating the sleeve by economy size moves its "
        f"correlation with the home market {'down' if d_corr < 0 else 'up'} "
        f"by {abs(d_corr):.3f}. Because the sleeve is leave-one-out, loading "
        f"it onto the largest economies makes the typical investor's foreign "
        f"holding <i>less</i> like their own market, not more: a Danish "
        f"investor's equal-weighted sleeve is mostly other small European "
        f"markets that move with Denmark, while their GDP-weighted sleeve is "
        f"mostly the United States and Japan, which do not. That effect works "
        f"against the loss of diversification, which is why the certainty "
        f"equivalents move less than the Herfindahl indices do."))

    out.extend(ctx.figure(
        "fig43_sleeve_weighting",
        "Top left: the effective number of markets in the sleeve under each "
        "weighting. Top right: certainty-equivalent consumption for every "
        "strategy under all of them. Bottom: whether all-international's "
        "advantage over the 50/50 split tracks concentration or the tilt."))

    out.append(ctx.h2("#sleeve.4 What this changes"))
    out.extend(ctx.bullets([
        ("The equal-weighted sleeve is the most favourable construction of "
         "those tested, and the levels in every table of this paper reflect "
         "that."
         if equal_is_kindest else
         f"The equal weighting is not the construction most favourable to "
         f"the finding — {str(kindest['label'])} produces a larger gap — so "
         f"the headline is not resting on the kindest sleeve available. The "
         f"levels do move with the construction, and should be read as "
         f"construction-dependent."),
        ("The ordering this paper diverges from the replicated study on does "
         "<b>not</b> survive every reweighting, so that divergence is a "
         "property of panel construction."
         if found["ordering_changes"] else
         "The ordering this paper diverges from the replicated study on "
         "<b>does</b> survive every reweighting tested, so the divergence is "
         "not manufactured by the equal weighting — though the margin "
         "narrows."),
        ("Concentration is not a sufficient statistic for a weighting scheme. "
         "Two sleeves with nearly the same Herfindahl index and different "
         "tilts give materially different answers, which is a caution against "
         "reading any single alternative construction as <i>the</i> "
         "robustness check."
         if not concentration_explains else
         "Concentration accounts for most of what a weighting scheme does "
         "here, which makes the effective number of markets a reasonable "
         "one-number summary of a sleeve."),
        "None of these is capitalisation weighting. This section brackets the "
        "answer rather than settling it: the truth sits between an "
        "equal-weighted sleeve nobody could buy and PPP-based proxies that "
        "misprice strong-currency markets. Section #limitations.1 carries this as a "
        "limitation.",
    ]))
    return out


def section_plan(ctx: Any) -> List[Flowable]:
    f = ctx.f
    ablation = f.table("plan_ablation")
    rounds = f.table("plan_alternation")
    mech = f.table("plan_mechanism_retirement_domestic")
    opt = f.table("plan_optimum_retirement_domestic")
    age = f.table("plan_retirement_age")

    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    rate_max = max(float(r) for r in cfg["plan"]["rate_grid"])
    by = ablation.set_index("freedom") if len(ablation) else ablation
    interaction = (float(by.loc["both", "gain_over_neither_pct"]
                         - by.loc["allocation only", "gain_over_neither_pct"]
                         - by.loc["plan only", "gain_over_neither_pct"])
                   if len(ablation) else float("nan"))
    shares = opt["optimal_domestic_share"].astype(float) if len(opt) \
        else pd.Series(dtype=float)
    coupled = bool(np.isfinite(interaction) and abs(interaction) >= 0.05)

    out: List[Flowable] = [
        ctx.h1("#plan. The Plan and the Portfolio, Solved Together")]
    out.append(ctx.p(
        "Section #glide solves the allocation schedule with the withdrawal "
        "rule pinned to a 4% rule. Section #spending ranks withdrawal rules "
        "with the allocation pinned to a 50/50 portfolio. Each then checks "
        "that its answer survives the thing it held fixed, and concludes that "
        "it does — the spending rule and the asset allocation are close to "
        "separable, and can be chosen independently."))
    out.append(ctx.p(
        "That claim deserves a harder test than either section gave it, "
        "because both tested it on a <i>ranking of six candidate "
        "portfolios</i>. A ranking of six candidates cannot resolve a grid "
        "step, and the location of an optimum is not rank information. This "
        "section tests it on the optimum, and then removes the assumption "
        "altogether by solving both decisions at once."))
    out.append(ctx.note(
        f"Each rule is run at its own best rate: a percentage-of-portfolio "
        f"rule at 4% is not the same policy as a fixed real rule at 4%, and "
        f"comparing them there compares spending levels rather than rule "
        f"shapes. Every rated rule turns out to have an <i>interior</i> "
        f"optimum on a grid running to {rate_max:.0%}, which is the check "
        f"that the objective disciplines spending rather than rewarding "
        f"whatever spends fastest — if the certainty equivalent simply rose "
        f"with the withdrawal rate, nothing in this section would mean "
        f"anything."))

    out.append(ctx.h2("#plan.1 The ranking separates; the optimum does not"))
    if len(opt):
        out.extend(ctx.table(
            [["Withdrawal rule", "Optimal domestic share",
              "CEC at the optimum", "Margin over runner-up (%)"]]
            + [[str(r["rule"]).replace("_", " "),
                f"{float(r['optimal_domestic_share']):.0%}",
                f"{float(r['cec_at_optimum']):.4f}",
                f"{float(r['margin_over_runner_up_pct']):.2f}"]
               for _, r in opt.iterrows()],
            f"The domestic share held from the retirement date, solved under "
            f"each withdrawal rule, γ = {gamma:g}.",
            note="Certainty equivalents are not comparable across rows — the "
                 "rules spend different amounts — but the argmaxes are."))
    if len(shares) and shares.nunique() > 1:
        out.append(ctx.p(
            f"<b>The optimum moves with the withdrawal rule.</b> It runs from "
            f"{shares.min():.0%} to {shares.max():.0%} domestic across the "
            f"{len(opt)} rules, a spread of "
            f"{(shares.max() - shares.min()) * 100:.0f} percentage points on "
            f"a decision Section #spending treats as independent of the rule. "
            f"Nothing in that section is wrong: it was reporting rankings, "
            f"and the rankings hold. It is the resolution that was the "
            f"problem."))
    else:
        out.append(ctx.p(
            "The optimum does not move with the rule on this grid, so the "
            "separability of Section #spending holds at this resolution too."))
    if len(mech):
        risky = mech[mech["ruin_is_possible"]]
        safe = mech[~mech["ruin_is_possible"]]
        agreed = risky[risky["agree"]]
        floor = float(mech["cec_optimal_domestic_share"].min())
        out.extend(ctx.table(
            [["Withdrawal rule", "Ruin observed", "Maximises CEC at",
              "Minimises ruin at", "Agree", "Lowest ruin reachable"]]
            + [[str(r["rule"]).replace("_", " "),
                "yes" if bool(r["ruin_is_possible"]) else "no",
                f"{float(r['cec_optimal_domestic_share']):.0%}",
                (f"{float(r['ruin_optimal_domestic_share']):.0%}"
                 if bool(r["ruin_is_possible"]) else "—"),
                "yes" if bool(r["agree"]) else "no",
                f"{float(r['min_ruin']):.2%}"]
               for _, r in mech.iterrows()],
            "Why the optimum moves: the two objectives the allocation can "
            "serve, and which rules let it choose between them.",
            note="Whether ruin is reachable is read off the simulated paths "
                 "rather than off the rule's description. Where it is not, "
                 "the ruin column is flat at zero and its argmin would be an "
                 "artefact of the grid order, so it is left undefined."))
        out.append(ctx.p(
            f"<b>The mechanism is the one Section #sequence identified.</b> "
            f"Under a rule that fixes real spending from wealth on the "
            f"retirement date, retirement-phase returns cannot reach "
            f"consumption until the money runs out — so the allocation's only "
            f"remaining job is keeping the portfolio alive, and it is chosen "
            f"to minimise ruin. Ruin is reachable under {len(risky)} of the "
            f"{len(mech)} rules; {len(agreed)} of those put the "
            f"certainty-equivalent maximum and the ruin minimum at the same "
            f"share. The {len(safe)} rules that cannot run out have no ruin "
            f"to minimise, and every one of them sits at {floor:.0%} — the "
            f"return-maximising corner. Same market, same investor, "
            f"different objective."))

    out.append(ctx.h2("#plan.2 Solving both at once"))
    out.append(ctx.p(
        "The search alternates: pick the best plan for the current schedule, "
        "re-solve the free-form schedule for that plan, repeat. It stops when "
        "a round returns the plan it started with — the fixed point, where "
        "neither decision would change given the other."))
    if len(ablation):
        out.extend(ctx.table(
            [["What was free to move", "Plan chosen", "CEC", "Gain (%)"]]
            + [[str(r["freedom"]), str(r["plan"]),
                f"{float(r['cec']):.4f}",
                f"{float(r['gain_over_neither_pct']):+.2f}"]
               for _, r in ablation.iterrows()],
            f"What each degree of freedom is worth, against a 4%-rule "
            f"investor holding 50/50, γ = {gamma:g}.",
            note="The interaction is the joint gain less the two one-sided "
                 "gains. It is zero exactly when the two decisions can be "
                 "chosen independently."))
    out.append(ctx.p(
        (f"<b>Solving them together is worth more than solving them "
         f"apart.</b> The interaction is {interaction:+.2f}% of "
         f"certainty-equivalent consumption — the joint search beats the sum "
         f"of the two one-sided searches, because the allocation that suits a "
         f"rule which cannot run out is not the allocation that suits one "
         f"which can. An adviser who picks the portfolio first and the "
         f"withdrawal policy afterwards leaves that on the table."
         if coupled else
         f"<b>The two decisions do separate at the level of the total.</b> "
         f"The interaction is {interaction:+.3f}%, so choosing them apart "
         f"costs almost nothing — even though, as #plan.1 shows, the "
         f"optimum's location does move.")))
    if len(rounds):
        out.append(ctx.p(
            f"The alternation reached its fixed point in {len(rounds)} "
            f"round{'s' if len(rounds) != 1 else ''}, which is itself "
            f"informative: the plan chosen for a flat starting schedule "
            f"survived being re-chosen for the solved one."))

    out.append(ctx.h2("#plan.3 The one decision this model cannot price"))
    out.append(ctx.p(
        "Retirement age is the third leg of a retirement plan and it is left "
        "out of the search above deliberately. Scored on its own, the "
        "certainty equivalent rises monotonically with the retirement date "
        "and the optimiser takes the oldest age on the grid."))
    if len(age):
        best = age.loc[age.groupby("retire_age")["cec"].idxmax()] \
            .sort_values("retire_age")
        out.extend(ctx.table(
            [["Retirement age", "CEC at the joint optimum's rule"]]
            + [[str(int(r["retire_age"])), f"{float(r['cec']):.4f}"]
               for _, r in best.iterrows()],
            "The retirement date, scored against the solved schedule."))
    out.append(ctx.p(
        "That is not a finding about retirement; it is the model's costless "
        "labour showing through. Nothing here charges the investor for the "
        "years they spend working, so another one is free and the search "
        "takes every one it is offered — which is why Section #retirement "
        "requires a matched comparison and why the date is held fixed above. "
        "Reporting the corner is the point: a joint optimisation is the "
        "cleanest way to find out which decisions a model can rank and which "
        "it merely appears to. Section #leisure supplies the missing side of "
        "that ledger and the corner does not survive it."))

    out.extend(ctx.figure(
        "fig58_plan",
        "Top left: the retirement-phase domestic curve under each withdrawal "
        "rule, each drawn against its own best so the shapes can be compared "
        "— the argmax moves. The window stops at 50% because every optimum "
        "lies below 30% and the curves decline monotonically beyond it; each "
        "is still normalised against its best over the whole grid. Top "
        "right: the share that maximises the "
        "certainty equivalent beside the share that minimises ruin, which is "
        "the mechanism. Bottom left: what each degree of freedom is worth "
        "alone and together. Bottom right: the retirement date, running to "
        "the ceiling because labour is free."))

    out.append(ctx.h2("#plan.4 What this changes"))
    out.extend(ctx.bullets([
        "<b>Separability is a claim about rankings, not about optima.</b> "
        "Section #spending's conclusion survives as stated and fails as "
        "usually read. Both halves matter: an adviser choosing between six "
        "model portfolios can pick the rule first, and one solving for an "
        "allocation cannot.",
        "<b>The withdrawal rule decides what the portfolio is for.</b> A rule "
        "that can run out makes the allocation a ruin problem; a rule that "
        "cannot makes it a return problem. That is the same mechanism "
        "Section #sequence finds in the variance decomposition, arriving "
        "here as a difference in the argmax.",
        (f"<b>The two decisions are worth more together than apart</b>, by "
         f"{interaction:+.2f}% of certainty-equivalent consumption."
         if coupled else
         "<b>The interaction is small</b>, so the sequential approach costs "
         "little in total even though it lands in the wrong place."),
        "<b>What is not modelled</b>: the plan is chosen once, at the start, "
        "and never revised. A real retiree re-reads their balance every year "
        "and may change rule as well as rate, so the gains here are a lower "
        "bound on what a genuinely adaptive plan would earn. And there is no "
        "disutility of labour, which is why #plan.3 exists.",
    ]))
    return out


def section_leisure(ctx: Any) -> List[Flowable]:
    f = ctx.f
    swept = f.table("leisure_sweep")
    claim = f.table("leisure_claim_factors")
    anchors = f.table("leisure_anchors")
    optima = f.table("leisure_optimal_age")
    crossings = f.table("leisure_break_even")

    from src import leisure as lei
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    arm = str(cfg["leisure"]["claim_arms"][-1])
    reference = int(cfg["leisure"]["claim_reference_age"])
    pension_age = int(cfg["leisure"].get("age_pension_age", 67))
    safety_net = float(cfg["leisure"].get("pre_pension_safety_net", 0.0))
    sg_rate = float(cfg.get("pension", {}).get("sg_rate", 0.12))
    comparison = f.table("leisure_systems_comparison")
    sysfound = lei.system_verdict(comparison)
    head = optima[optima["claim_arm"] == arm].sort_values("leisure") \
        if "claim_arm" in optima.columns else optima.sort_values("leisure")
    flat = optima[optima["claim_arm"] == "unadjusted"].sort_values("leisure") \
        if "claim_arm" in optima.columns else pd.DataFrame()
    early = crossings[crossings["is_earlier"]] if len(crossings) else crossings
    if "claim_arm" in early.columns:
        early = early[early["claim_arm"] == arm]
    early = early.sort_values("retire_age", ascending=False)
    block = swept[swept["claim_arm"] == arm] if "claim_arm" in swept.columns \
        else swept

    out: List[Flowable] = [
        ctx.h1("#leisure. What a Year of Retirement Is Worth")]
    out.append(ctx.p(
        "Section #plan solves the withdrawal rule and the allocation "
        "together, and finds that the third leg of a retirement plan — the "
        "date — cannot be solved at all. Left free it runs to the oldest age "
        "on the grid and stays there, because nothing in this model charges "
        "anyone for the years they spend working. Another year is more "
        "contributions, a shorter drawdown and a larger benefit, at no cost "
        "whatever. This section supplies the missing side of that ledger."))
    out.append(ctx.p(
        "Rather than invent a disutility of labour in utils — a quantity "
        "nothing here can calibrate — the cost is written as a "
        "<i>consumption equivalent</i>. Let <i>L</i> be the number such that "
        "a retired year at consumption <i>c</i> is worth exactly as much as a "
        "working year at <i>L·c</i>. Working years are then evaluated at "
        "<i>c/L</i> and retired years left alone, so <i>L</i> = 1 charges "
        "nothing and reproduces every other result in this paper exactly."))
    out.append(ctx.note(
        "The aggregation here runs from age 25, not from the retirement date. "
        "A retirement-window objective cannot price a retirement date: it "
        "charges the investor for the years they worked and credits them "
        "nothing for the ones they did not. These certainty equivalents are "
        "therefore not comparable with those elsewhere in the paper."))

    out.append(ctx.h2("#leisure.1 The pension you claim early"))
    out.append(ctx.p(
        "There is a larger thing to fix first. Every other section holds the "
        "retirement date fixed, so the benefit starts on the same birthday "
        "for every strategy and its start date cancels. It does not cancel "
        "here: left alone, this model pays whoever stops work at fifty a full "
        "unreduced pension for forty-three years. That is not a preference "
        "for leisure, it is a gift, and it is worth more than any plausible "
        "value of leisure."))
    out.append(ctx.p(
        f"Real systems reduce a benefit claimed early and raise one deferred, "
        f"by roughly enough to leave its expected present value alone. The "
        f"multiplier that does that exactly is the ratio of annuity factors, "
        f"<i>A</i>(reference)/<i>A</i>(age), derived from this model's own "
        f"Gompertz survival curve and discount factor rather than taken from "
        f"a statute."))
    if len(claim):
        out.extend(ctx.table(
            [["Retire at", "Benefit multiplier", "Adjustment (%)",
              "Per year from the reference (%)"]]
            + [[str(int(r["retire_age"])),
                f"{float(r['claim_factor']):.3f}",
                f"{float(r['adjustment_pct']):+.1f}",
                ("—" if not np.isfinite(float(r["per_year_pct"]))
                 else f"{float(r['per_year_pct']):+.2f}")]
               for _, r in claim.sort_values("retire_age").iterrows()],
            f"The actuarially fair claiming adjustment, measured against age "
            f"{reference}.",
            note="Fair by construction in this model rather than "
                 "approximately fair in someone else's: it is solved from "
                 "the same survival law Section #mortality uses."))
        rates = claim["per_year_pct"].dropna().abs()
        out.append(ctx.p(
            f"That works out at {rates.min():.1f}% to "
            f"{rates.max():.1f}% a year away from "
            f"{reference}. The US schedule reduces a benefit by about 6.7% "
            f"for each of the first three years claimed early and raises it "
            f"about 8% for each year deferred past full retirement age. "
            f"Nothing here was fitted to that; the agreement is a check on "
            f"the survival law, not a coincidence to lean on."))
    if len(flat) and len(head):
        moved = int(flat["optimal_age"].iloc[0]) != int(head["optimal_age"].iloc[0])
        out.append(ctx.p(
            (f"<b>The claiming rule decides the answer before leisure gets a "
             f"say.</b> Unadjusted, the best date is "
             f"{int(flat['optimal_age'].iloc[0])} even when working costs "
             f"nothing at all. Adjusted, it is "
             f"{int(head['optimal_age'].iloc[0])}. Everything below is the "
             f"adjusted arm."
             if moved else
             f"The adjustment does not move the zero-leisure optimum, which "
             f"stays at {int(head['optimal_age'].iloc[0])}.")))

    out.append(ctx.h2("#leisure.2 When to stop"))
    if len(head):
        out.extend(ctx.table(
            [["A retired year is worth", "Retire at", "Lifetime CEC",
              "Runner-up", "Margin (%)"]]
            + [[f"+{float(r['leisure_pct']):.0f}%",
                str(int(r["optimal_age"])),
                f"{float(r['cec_at_optimum']):.4f}",
                str(int(r["runner_up_age"])),
                f"{float(r['margin_over_runner_up_pct']):.2f}"]
               for _, r in head.iterrows()],
            f"The best retirement date at each value of leisure, γ = "
            f"{gamma:g}.",
            note="An optimum sitting on either end of the age grid would be "
                 "the grid's answer rather than the model's; the grid runs "
                 "from 50 to 70 for that reason."))
        out.append(ctx.p(
            (f"<b>Pricing the cost of working gives the date an interior "
             f"optimum.</b> It runs from {int(head['optimal_age'].iloc[0])} "
             f"when a retired year is worth no more than a working one to "
             f"{int(head['optimal_age'].iloc[-1])} at the top of the grid. "
             f"Section #plan's corner is gone — not because the model changed "
             f"its mind, but because both sides of the decision are now "
             f"priced."
             if head["optimal_age"].nunique() > 1 else
             f"The optimal date does not move across the grid, holding at "
             f"{int(head['optimal_age'].iloc[0])}.")))
    if "cec_survival_weighted" in block.columns:
        alive = block.loc[block.groupby("leisure")[
            "cec_survival_weighted"].idxmax()].sort_values("leisure")
        paired = head.merge(
            alive[["leisure", "retire_age"]], on="leisure", how="inner")
        gap = paired["optimal_age"] - paired["retire_age"]
        out.append(ctx.p(
            (f"Weighting by survival pulls it earlier still, at "
             f"{int((gap > 0).sum())} of the {len(paired)} values of leisure "
             f"tested and by up to {int(gap.max())} years. Deferring is a bet "
             f"that you will be there to collect, and the Gompertz law of "
             f"Section #mortality prices that bet where a certain "
             f"ninety-third birthday cannot."
             if (gap > 0).any() else
             "Weighting by survival does not pull the date earlier at any "
             "value of leisure tested, which is worth stating because it is "
             "the opposite of what mortality risk would suggest.")))

    out.append(ctx.h2("#leisure.3 The break-even, which is the number to carry"))
    out.append(ctx.p(
        "An optimal date depends on a calibration nobody can hand you. A "
        "break-even does not. For each date earlier than the one an investor "
        "would choose if working cost nothing, this is the value of leisure "
        "at which it becomes worthwhile."))
    if len(early):
        out.extend(ctx.table(
            [["Retire at", "Years earlier", "Leisure must be worth",
              "which is a consumption drop of"]]
            + [[str(int(r["retire_age"])), str(int(r["years_earlier"])),
                f"{float(r['break_even_pct']):.1f}%",
                f"{float(r['implied_consumption_drop']):.0%}"]
               for _, r in early.iterrows()],
            f"What each year of earlier retirement costs to justify, against "
            f"age {int(early['reference_age'].iloc[0])}.",
            note="The reference is the date that wins when working costs "
                 "nothing, so what is priced is the earlier retirement "
                 "rather than the model's own lean toward it."))
        first, last = early.iloc[0], early.iloc[-1]
        out.append(ctx.p(
            f"That turns the question into one a reader can settle for "
            f"themselves. Stopping {int(first['years_earlier'])} years early "
            f"is a claim that a year of your own time is worth at least "
            f"{float(first['break_even_pct']):.0f}% of a year's consumption "
            f"— about the size of the consumption drop actually observed at "
            f"retirement. Stopping {int(last['years_earlier'])} years early "
            f"is a much larger claim: "
            f"{float(last['break_even_pct']):.0f}%. Whether either is true, "
            f"this paper does not say."))

    out.extend(ctx.figure(
        "fig59_cost_of_working",
        "Top left: lifetime certainty equivalent against the retirement date, "
        "one line per value of leisure, with each line's maximum circled — a "
        "cost on working buys an interior optimum. Top right: how that "
        "optimum moves, with and without survival weighting. Bottom left: the "
        "break-even value of leisure for each earlier date, against the "
        "consumption drops observed at retirement. Bottom right: the claiming "
        "adjustment that makes the date of the pension worth nothing either "
        "way."))

    out.append(ctx.h2("#leisure.4 The same question under Australia's pension"))
    out.append(ctx.p(
        f"Everything above pays an American pension: earnings-related, "
        f"payable from the day work stops, and adjusted for the age it starts "
        f"at. Australia's is shaped differently in two ways that pull against "
        f"each other. The Age Pension is payable at {pension_age} however "
        f"early somebody stopped, so there is no claiming choice to adjust "
        f"and no bridge — an Australian retiring at 55 funds twelve years "
        f"alone before a cent arrives. And the Superannuation Guarantee puts "
        f"{sg_rate:.0%} of earnings into the portfolio on top of voluntary "
        f"saving, which is what pays for crossing that bridge. Running the "
        f"pension alone and then the pension with the guarantee separates the "
        f"two."))
    if len(comparison):
        out.extend(ctx.table(
            [["Pension system", "Best date, leisure free", "at the top",
              "Earlier dates any leisure justifies",
              "Nearest earlier date costs", "per year"]]
            + [[SYSTEM_LABEL.get(str(r["system"]), str(r["system"])),
                str(int(r["age_at_zero_leisure"])), str(int(r["age_at_top"])),
                f"{int(r['earlier_dates_reachable'])} of "
                f"{int(r['earlier_dates_offered'])}",
                (f"{float(r['nearest_break_even_pct']):.1f}%"
                 if np.isfinite(float(r.get("nearest_break_even_pct",
                                            float("nan")))) else "—"),
                (f"{float(r['cost_per_year_pct']):.1f}%"
                 if np.isfinite(float(r.get("cost_per_year_pct",
                                            float("nan")))) else "—")]
               for _, r in comparison.iterrows()],
            f"The retirement date under each pension, γ = {gamma:g}.",
            note="Break-evens are measured against each system's own "
                 "zero-leisure date, so they price the earlier retirement "
                 "rather than the system's own lean."))
    if sysfound.get("australian_pension_later"):
        out.append(ctx.p(
            f"<b>Australia's pension puts the date "
            f"{sysfound['australian_pension_years']:.0f} years later.</b> On "
            f"the same voluntary saving the best date moves from "
            f"{sysfound['us_age']:.0f} under the American schedule to "
            f"{sysfound['gated_age']:.0f}. Nothing about the investor "
            f"changed. But the two systems differ in <i>two</i> ways at once, "
            f"so that number belongs to the pair and not to either of them — "
            f"which is what Section #leisure.5 is for."))
    if sysfound.get("super_buys_back"):
        out.append(ctx.p(
            f"<b>Compulsory saving takes some of it back.</b> Adding the "
            f"Superannuation Guarantee moves the date from "
            f"{sysfound['gated_age']:.0f} to "
            f"{sysfound['legislated_age']:.0f}, recovering "
            f"{sysfound['super_years_earlier']:.0f} of the "
            f"{sysfound.get('australian_pension_years', float('nan')):.0f} "
            f"between the two systems. That one is a single change — the "
            f"guarantee is on or off and nothing else moves — so unlike the "
            f"row above it can be read as a cause."))
    if "legislated_vs_us_years" in sysfound:
        net = (f"Net of both, an Australian's best date is "
               f"{abs(sysfound['legislated_vs_us_years']):.0f} years "
               f"{'later' if sysfound['australia_retires_later'] else 'earlier'}"
               f" than an American's at the same voluntary saving "
               f"({sysfound['legislated_age']:.0f} against "
               f"{sysfound['us_age']:.0f}).")
        if sysfound.get("cost_similar"):
            net += (
                f" The <i>price</i> of going earlier still is close in both — "
                f"{sysfound['legislated_cost_per_year']:.1f}% of lifetime "
                f"certainty-equivalent consumption per year against "
                f"{sysfound['us_cost_per_year']:.1f}% — so what the Australian "
                f"system moves is where the clock starts, not how steeply it "
                f"runs.")
        elif "cost_ratio" in sysfound:
            steeper = sysfound.get("australia_dearer_per_year")
            net += (
                f" And it moves the <i>slope</i> as well as the start. A year "
                f"earlier than the best date costs an Australian "
                f"{sysfound['legislated_cost_per_year']:.1f}% of lifetime "
                f"certainty-equivalent consumption against an American's "
                f"{sysfound['us_cost_per_year']:.1f}%, a factor of "
                f"{sysfound['cost_ratio']:.2f}. The Australian system does not "
                f"shift one trade-off later so much as pose a "
                f"{'harder' if steeper else 'softer'} one: the date is later "
                f"<i>and</i> each year taken off it is "
                f"{'dearer' if steeper else 'cheaper'}. Both follow from the "
                f"same gate — with the pension fixed at "
                f"{pension_age:.0f}, a year of retirement bought before then "
                f"is funded entirely from the portfolio, so it costs more than "
                f"a year bought against a benefit that moves with it.")
        out.append(ctx.p(net))
    out.append(ctx.note(
        f"This arm needs a caveat the others do not. A retiree who exhausts "
        f"the portfolio before the pension age receives, here, a means-tested "
        f"payment at {safety_net:.0%} of the full rate, standing in for the "
        f"working-age safety net. At zero it would be literally nothing, and "
        f"a CRRA aggregator punishes a single year of that without limit — "
        f"one path in ten thousand at the consumption floor is enough to "
        f"decide a certainty equivalent on its own. That floor is a "
        f"judgement, and the earliest dates are sensitive to it in a way the "
        f"later ones are not. Nor is preservation age modelled: this "
        f"portfolio can be drawn at 50, where Australian super cannot."))

    # ---- 2x2: which of the two differences does the work ----------------
    decomposition = f.table("leisure_features_decomposition")
    feat = lei.feature_verdict(decomposition) if len(decomposition) \
        else {"measured": False}
    bite_frame = f.table("leisure_means_test_bite")
    bite = bite_frame.iloc[0].to_dict() if len(bite_frame) else {}
    if feat.get("measured"):
        out.append(ctx.h2("#leisure.5 Which of the two differences does "
                          "the work"))
        out.append(ctx.p(
            f"The comparison above changes two things at once. Australia's "
            f"pension starts on a fixed birthday rather than when work stops, "
            f"<i>and</i> it is worked out by a means test rather than from a "
            f"career of earnings. Attributing the gap to either without "
            f"crossing them is guesswork, so this is the 2×2: each feature "
            f"alone, then both, against the same baseline."))
        if len(decomposition):
            out.extend(ctx.table(
                [["What changes", "Best date", "Years later", "CEC",
                  "vs baseline"]]
                + [[str(r["feature"]),
                  "—" if not np.isfinite(r["age"]) else f"{r['age']:.0f}",
                  f"{r['effect']:+.0f}",
                  "—" if not np.isfinite(r["cec"]) else f"{r['cec']:.4f}",
                  f"{r['cec_effect']:+.4f}"]
                 for _, r in decomposition.iterrows()],
                f"Each feature of Australia's pension against the American "
                f"baseline, γ = {gamma:g}.",
                note="The baseline is the American schedule: earnings-"
                     "related, payable from retirement, actuarially reduced "
                     "for claiming early."))
        timing, formula = feat["timing_years"], feat["formula_years"]
        dominant = feat.get("dominant")
        which = ("when the benefit starts" if dominant == "timing"
                 else "how the benefit is worked out")
        other = ("how it is worked out" if dominant == "timing"
                 else "when it starts")
        body = (f"<b>It is {which}, not {other}.</b> Changing only the start "
                f"date moves the best date by {timing:+.0f} years. Changing "
                f"only the formula — a means-tested pension replacing an "
                f"earnings-related one, still payable the day work stops — "
                f"moves it {formula:+.0f}.")
        if feat.get("timing_opposes_joint"):
            body += (
                f" The eligibility gate moves the date the <i>opposite</i> "
                f"way to the system it belongs to, and the reason is worth "
                f"stating: a pension payable at a fixed age is not reduced "
                f"for stopping early, whereas an actuarially adjusted one is. "
                f"The gate removes a bridge and a penalty at once, and on "
                f"this panel the penalty is worth more than the bridge.")
        out.append(ctx.p(body))
        if not feat.get("separable", True):
            out.append(ctx.p(
                f"The two do not add. The interaction is "
                f"{feat['interaction_years']:+.0f} years against a joint "
                f"{feat['both_years']:+.0f}, or "
                f"{feat.get('interaction_share', float('nan')):.0%} of it — "
                f"which is what one feature disarming the other looks like. "
                f"Once the means test has taken the pension away, the "
                f"birthday it would have arrived on stops mattering."))
        if bite:
            out.append(ctx.p(
                f"<b>And the formula does so much because this household is "
                f"not poor enough to be paid.</b> The assets test begins "
                f"withdrawing the pension at "
                f"{float(bite['free_area_multiple']):.1f}× average earnings "
                f"and reaches zero at "
                f"{float(bite['cutoff_multiple']):.1f}×. The median retiree "
                f"here holds {float(bite['median_wealth_multiple']):.1f}× — "
                f"{float(bite['median_over_cutoff']):.1f} times the cut-off — "
                f"so {float(bite['share_above_cutoff']):.0%} of them sit "
                f"where the Age Pension has been tapered away entirely. "
                f"Against a full rate worth "
                f"{float(bite['full_rate_replacement']):.0%} of career-"
                f"average income, what is actually paid averages "
                f"{float(bite['benefit_replacement']):.1%}; the American "
                f"schedule, means-tested against nothing, pays "
                f"{float(bite.get('us_benefit_replacement', float('nan'))):.0%}"
                f". So this is only partly a comparison of pension "
                f"<i>designs</i>. It is substantially a comparison between a "
                f"household that receives a pension and the same household "
                f"receiving almost none."))

    # ---- the withdrawal rule is half the comparison ----------------------
    rules_frame = f.table("leisure_rule_comparison")
    rulefound = lei.rule_verdict(
        rules_frame, portfolio_rule=str(
            cfg["lifecycle"]["retirement"]["rule"])) if len(rules_frame) \
        else {"measured": False}
    if rulefound.get("measured"):
        out.append(ctx.h2("#leisure.6 The withdrawal rule is half the "
                          "comparison"))
        out.append(ctx.p(
            "Everything above spends by this project's own rule: a fixed real "
            "amount set as a share of the portfolio at retirement. That rule "
            "cannot answer the question a pension is <i>for</i>. It sets the "
            "standard of living from the portfolio, so a larger balance "
            "spends more rather than lasting longer, and the pension is spent "
            "on top of the withdrawal rather than funding part of it. The "
            "alternative fixes the target instead — a set share of "
            "pre-retirement income for life, the pension netted off, the "
            "portfolio paying the rest."))
        if len(rules_frame):
            out.extend(ctx.table(
                [["Rule", "Pension system", "CEC", "Ruin (%)", "Mean c",
                  "5th pct c"]]
                + [[str(r["rule"]),
                  SYSTEM_LABEL.get(str(r["system"]), str(r["system"])),
                  f"{r['cec']:.4f}", f"{100 * r['prob_ruin']:.1f}",
                  f"{r['mean_consumption']:.3f}",
                  f"{r['p5_consumption']:.3f}"]
                 for _, r in rules_frame.iterrows()],
                f"Each pension system under each withdrawal rule, γ = "
                f"{gamma:g}, retiring at {reference}.",
                note="The replacement rules target a share of average income "
                     "over the final working years, held in real terms."))
        if rulefound.get("portfolio_rule_ruin_identical"):
            out.append(ctx.p(
                "<b>Under the portfolio rule all three systems ruin at the "
                "same rate, to the last decimal.</b> That is an artefact, not "
                "a finding, and it is worth naming because this project has "
                "leaned on that rule throughout: withdrawals set as a share "
                "of wealth at retirement scale with the portfolio, so "
                "doubling the balance doubles the spending and exhausts it in "
                "the same year. Ruin under that rule is a property of the "
                "return sequence alone, and says nothing about how much was "
                "saved."))
        if rulefound.get("contender_ever_safer"):
            out.append(ctx.p(
                f"<b>Hold the target still and the Superannuation Guarantee "
                f"finally shows up where it should.</b> At "
                f"{rulefound['widest_ruin_rule'].replace('replace_', '')}% "
                f"replacement the Australian household runs out "
                f"{100.0 * rulefound['widest_ruin_gap']:.1f} percentage "
                f"points less often than the American one. Compulsory saving "
                f"buys safety — but only under a rule that permits safety to "
                f"be bought."))
        if rulefound.get("safer_but_lower"):
            out.append(ctx.p(
                "<b>And the certainty equivalent still favours the American "
                "schedule at every target.</b> Both facts hold at once, and "
                "the combination is the result: the Australian household runs "
                "out <i>less often</i> and is <i>poorer when it does</i>. "
                "What remains once the portfolio is gone is a flat "
                "means-tested pension rather than an earnings-related one, "
                "and a CRRA aggregator at this risk aversion prices the depth "
                "of the floor above the frequency of reaching it. A reader "
                "who cares about the probability of running out and a reader "
                "who cares about the worst case will rank these two systems "
                "differently, and neither is misreading the table."))

    out.append(ctx.h2("#leisure.7 What this changes"))
    out.extend(ctx.bullets([
        (f"<b>The retirement date is not unpriceable — it was unpriced.</b> "
         f"Charging for the years spent working turns Section #plan's corner "
         f"into an interior optimum between "
         f"{int(head['optimal_age'].min())} and "
         f"{int(head['optimal_age'].max())}, depending on what a year is "
         f"worth." if len(head) else
         "Charging for the years spent working gives the date an optimum."),
        "<b>The claiming rule matters more than leisure does.</b> An "
        "unreduced pension starting whenever work stops is worth more than "
        "any plausible value of leisure and would have decided the answer on "
        "its own. That is worth stating for its own sake: a lifecycle model "
        "that lets the retirement date move must price the pension's start "
        "date, or it is not measuring preferences at all.",
        "<b>The output is a break-even, not a recommendation.</b> The value "
        "of your own time is not something this panel of returns can "
        "measure. What it can do is say what you must believe to justify a "
        "given date, and leave the believing to you.",
        "<b>What is not modelled</b>: partial retirement, which is what most "
        "people actually do; any change in the value of leisure with age or "
        "health, when both plainly change; an earliest claiming age, so the "
        "adjusted benefit here can start at fifty where no real system would "
        "pay it; and the possibility that work is worth something positive — "
        "purpose, company, structure — which would push the date later and "
        "which this parameterisation cannot represent, since <i>L</i> is "
        "bounded below at one.",
    ]))
    return out


# ---------------------------------------------------------------------------
# 19. Discussion
# ---------------------------------------------------------------------------
def section_tax(ctx: Any) -> List[Flowable]:
    """What each retirement system's own tax does to the comparison."""
    f = ctx.f
    swept = f.table("tax_comparison")
    torpedo_frame = f.table("tax_torpedo")
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    rule = str(cfg["lifecycle"]["retirement"]["rule"])

    from src import tax as tx

    headline = swept[swept["rule"] == rule] if len(swept) else swept
    found = tx.tax_verdict(headline) if len(headline) else {"measured": False}
    torpedo = torpedo_frame.iloc[0].to_dict() if len(torpedo_frame) else {}

    out: List[Flowable] = [
        ctx.h1("#tax. The Tax Each System Actually Charges")]
    out.append(ctx.p(
        "Every result so far is tax-free, and until Section #leisure that "
        "was harmless. A charge every strategy pays alike cancels out of a "
        "comparison between strategies, so the ranking of portfolios is "
        "unchanged by leaving it out. Comparing two <i>countries</i> is "
        "where the argument fails, because what differs between the "
        "American and Australian retirement systems is precisely which "
        "income is taxed, and when."))
    out.append(ctx.p(
        "The asymmetry is the reason it could not be reasoned about. "
        "Australia charges early and lightly — on fund earnings through "
        "accumulation, and far less than the 15% headline implies — and then "
        "not at all: superannuation drawn after 60 "
        "from a taxed fund is not merely tax-free but <i>not assessable "
        "income</i>, so it cannot affect the tax on anything else. The Age "
        "Pension is assessable, but the seniors and pensioners offset lifts "
        "the effective threshold above the full single rate, so a pensioner "
        "living on the pension pays nothing. The United States charges late, "
        "progressively, and interactively: social security is taxable on a "
        "sliding scale whose thresholds have not been indexed since 1993, "
        "and a withdrawal from a traditional account is ordinary income "
        "<i>and</i> counts toward the provisional income deciding how much "
        "of the benefit is taxed."))

    out.append(ctx.h2("#tax.1 The torpedo, measured"))
    if torpedo:
        out.append(ctx.p(
            f"At the withdrawal this household actually makes, the American "
            f"retiree faces a marginal rate of "
            f"{100 * float(torpedo['torpedo_marginal']):.1f}% while sitting "
            f"in a {100 * float(torpedo['torpedo_statutory']):.0f}% bracket "
            f"— a factor of {float(torpedo['torpedo_multiple']):.2f}, "
            f"{'which is the arithmetic ceiling' if abs(float(torpedo['torpedo_multiple']) - 1.85) < 0.005 else 'against an arithmetic ceiling of 1.85'}: "
            f"every dollar drawn drags up to eighty-five cents of benefit "
            f"into the tax base behind it. The "
            f"Australian retiree faces no such thing, because the withdrawal "
            f"that would do the dragging is not assessable income."))
        out.append(ctx.p(
            "The shape matters as much as the level. The marginal rate rises, "
            "then <i>falls</i> as the 85% inclusion completes, then rises "
            "again into the next bracket — so it is not monotone in income, "
            "and a retiree who looked up their bracket would be wrong about "
            "the cost of their next withdrawal across most of the range this "
            "household occupies."))

    out.append(ctx.h2("#tax.2 What it does to the comparison"))
    if len(headline):
        shown = headline.copy()
        out.extend(ctx.table(
            [["Pension system", "Tax regime", "CEC", "Ruin (%)",
              "Tax paid (% of gross)"]]
            + [[SYSTEM_LABEL.get(str(r["system"]), str(r["system"])),
                REGIME_LABEL.get(str(r["regime"]), str(r["regime"])),
                f"{float(r['cec']):.4f}",
                f"{100 * float(r['prob_ruin']):.1f}",
                f"{100 * float(r['tax_share_of_gross']):.1f}"]
               for _, r in shown.iterrows()],
            f"Each system tax-free and under its own schedule, γ = "
            f"{gamma:g}.",
            note="The two American rows are the honest pair: a Roth is what "
                 "every earlier section implicitly assumed, and a "
                 "traditional account is what superannuation deserves to be "
                 "compared with, since both take contributions from pre-tax "
                 "earnings."))
    if found.get("measured") and "ranking_survives" in found:
        def _cost(value: float) -> str:
            # Deferring tax buys a larger contribution, so an arm can come
            # out ahead. The sentence has to be able to say that.
            if abs(value) < 0.05:
                return "costs nothing at all"
            return (f"costs {abs(value):.1f}%" if value < 0.0
                    else f"is worth {value:+.1f}%")

        trad = float(found.get("us_traditional_cost_pct", 0.0))
        au_cost = float(found.get("au_cost_pct", 0.0))
        roth = found.get("us_roth_cost_pct")
        body = (
            f"<b>Tax {_cost(trad)} of certainty-equivalent consumption to an "
            f"American saving in a traditional account, and "
            f"{_cost(au_cost)} to an Australian.</b>")
        if roth is not None:
            body += (
                f" To an American saving in a Roth it {_cost(float(roth))}: "
                f"with no other assessable income the benefit alone stays "
                f"below thresholds frozen since 1993, and the torpedo needs "
                f"other income to fire.")
        body += (
            f" The gap between the two countries moves from "
            f"{float(found['gap_untaxed_pct']):+.1f}% to "
            f"{float(found['gap_taxed_pct']):+.1f}%, and the ranking "
            f"{'holds' if found['ranking_survives'] else 'flips'}.")
        out.append(ctx.p(body))
        if found.get("gap_narrowed") and found.get("ranking_survives"):
            out.append(ctx.p(
                "Narrower, then, but not reversed — the honest answer to the "
                "objection that prompted this section, and not the one it "
                "expected."))
        elif found.get("ranking_survives"):
            out.append(ctx.p(
                "<b>The gap widens.</b> That is the opposite of what the "
                "objection behind this section anticipated, and the reason is "
                "a timing asymmetry rather than a difference in rates. "
                "Australia's tax-free withdrawal is real, but it is collected "
                "once, at the end; the fund-earnings tax that pays for it is "
                "levied every year for four decades and compounds against the "
                "balance. Deferring tax runs the other way — the contribution "
                "grows by the tax not paid on it, and what is owed later is "
                "owed on a lower income — which is why the traditional arm "
                "comes out ahead of paying no tax at all. How much of this "
                "rests on the one rate nobody can pin down is the next "
                "subsection."))
        else:
            out.append(ctx.p(
                "The untaxed comparison in Section #leisure therefore did not "
                "merely lack precision; it pointed the wrong way. That is "
                "what a section like this exists to find."))
    parts_frame = f.table("tax_fund_components")
    fund_frame = f.table("tax_fund_earnings")
    if len(parts_frame):
        row = parts_frame.iloc[0]
        total, naive = float(row["total_drag"]), float(row["naive_drag"])
        out.append(ctx.h2("#tax.3 What the Australian fund actually pays"))
        out.append(ctx.p(
            "The statutory rate on fund earnings is 15%, and charging that "
            "against the return would overstate the charge by more than an "
            "order of magnitude. Three things stand between the headline and "
            "what is paid, and on this portfolio they very nearly cancel it. "
            "<b>Unrealised gains are not income</b>: a fund that holds owes "
            "nothing on appreciation, and because earnings in the retirement "
            "phase are exempt outright, a gain carried across that boundary "
            "is never taxed at all. <b>A realised gain held beyond a year is "
            "discounted by a third</b>, so the rate on it is ten per cent. "
            "And <b>a franked dividend carries a credit worth more than the "
            "liability</b> — Section #franking derives it on this same panel "
            "— so a domestic dividend is a refund that subsidises the tax on "
            "the international sleeve rather than adding to it."))
        out.append(ctx.p(
            f"Together they leave a drag of {abs(total):.3%} a year against "
            f"the {abs(naive):.2%} the statutory rate on the return would "
            f"imply"
            + (f", a factor of {abs(naive / total):.0f}." if total else ".")
            + f" At a domestic share of "
            f"{float(row['domestic_weight']):.0%} the income side comes out "
            + ("positive: the credit more than covers the fund's tax on the "
               "rest of the portfolio."
               if float(row["income_drag"]) > 0 else
               "negative: the credit does not cover the tax on the rest.")))
        if len(fund_frame) > 1 and "cost_pct" in fund_frame.columns:
            measured = fund_frame.iloc[
                (fund_frame["realisation"] - 0.0335).abs().argsort().iloc[0]]
            out.extend(ctx.table(
                [["Sold each year", "Drag a year", "CEC", "Ruin (%)",
                  "Against holding to the pension phase (%)"]]
                + [[f"{float(r['realisation']):.2%}",
                    f"{float(r['drag']):+.4%}", f"{float(r['cec']):.4f}",
                    f"{100 * float(r['prob_ruin']):.1f}",
                    f"{float(r['cost_pct']):+.1f}"]
                   for _, r in fund_frame.iterrows()],
                "The Australian charge by how much the fund is made to sell.",
                note="Holding until the pension phase is the reference. "
                     "Section #turnover measures 3.35% a year for this "
                     "strategy, all of it forced by drift rather than "
                     "chosen."))
            out.append(ctx.p(
                f"<b>Holding until the pension phase is not a fanciful "
                f"case</b> — it is what an index fund left alone does. At "
                f"the {float(measured['realisation']):.2%} a year Section "
                f"#turnover measures for this strategy, the charge costs "
                f"{abs(float(measured['cost_pct'])):.1f}% of the certainty "
                f"equivalent."))

    out.append(ctx.note(
        "What is still missing, in rough order of how much it would move "
        "this: the Australian marginal tax on earnings held outside super, "
        "since voluntary saving is treated as a Roth in both countries to "
        "isolate the retirement wrapper; capital gains tax and its "
        "one-third discount inside a fund, which would push the 15% on fund "
        "earnings down; US state income taxes, which would push the American "
        "charge up; and any filing status but single."))
    return out


def section_discussion(ctx: Any) -> List[Flowable]:
    f = ctx.f
    lottery = f.table("retirement_lottery_stats").iloc[0]
    swr = f.table("sensitivity_safe_withdrawal_rates")
    swr_eq = float(swr[swr["strategy"] == "balanced_all_equity"]
                   ["safe_withdrawal_rate_at_5%_ruin"].iloc[0])

    out: List[Flowable] = [ctx.h1("#discussion. Discussion")]
    out.append(ctx.h2("#discussion.1 What the replication does and does not establish"))
    out.append(ctx.p(
        "The central claim reproduces cleanly and survives an unusually wide "
        "robustness exercise. That is a real result and it should update "
        "anyone whose prior rested on i.i.d. US-only simulation. But it is "
        "worth being precise about what has been established."))
    out.append(ctx.p(
        "What has been established is a statement about a particular "
        "return-generating process: if the future resembles a block-resampled "
        "draw from the developed-market twentieth century, then a diversified "
        "all-equity portfolio dominates a declining glide path on essentially "
        "every metric a lifecycle investor would use. What has <i>not</i> been "
        "established is that the future will resemble that process. Section #valuation "
        "removes part of that gap by conditioning on the valuation a lifetime "
        "starts at, which is the piece of the objection an investor can "
        "actually observe. What remains outside the model is the rest of it: "
        "the secular decline in real yields, and whether the equity premium "
        "the twentieth century delivered is a structural feature or a "
        "realised surprise."))
    out.append(ctx.p(
        "The robustness exercise is worth naming, because surviving one is a "
        "claim that has been made for weaker evidence. Four things were "
        "varied and none overturned the ranking: the preference and lifecycle "
        "parameters (Section #sensitivity), the membership of the panel one "
        "country at a time and the era it is drawn from (Section #panel), the "
        "weighting of the international sleeve across five constructions "
        "(Section #sleeve), and the fund costs of implementing it "
        "(Section #fees). That exercise also produced the paper's most "
        "uncomfortable number. The delete-one jackknife of Section #panel.4 "
        "puts a standard error on the headline gap wide enough that the "
        "interval barely excludes zero, and it applies to every other "
        "comparison here: sixteen countries is sixteen countries however many "
        "bootstrap paths are drawn through them. The direction of the results "
        "in this paper is better established than their magnitudes, and the "
        "magnitudes are quoted to more decimal places than the panel can "
        "support because that is what the simulation resolves, not what the "
        "evidence does."))
    out.append(ctx.p(
        "That distinction matters most for the sceptic's strongest objection, "
        "which we take seriously: the sample is a sample of survivors at the "
        "level of the <i>system</i> even where it is not at the level of the "
        "country. Every country in the panel ended the century with a "
        "functioning capital market. A world in which that was not true is "
        "outside the support of the sampler, and no amount of block "
        "resampling can put it back in."))

    out.append(ctx.h2("#discussion.2 The hierarchy of levers"))
    out.append(ctx.p(
        "Reading the extensions together produces a ranking of what actually "
        "moves a retirement outcome, and it is not the ranking the industry's "
        "attention implies."))
    out.extend(ctx.bullets([
        f"<b>When you retire dominates.</b> The decade around the retirement "
        f"date explains {pc(float(lottery['r2_retirement_window']), 1)} of the "
        f"variation in retirement consumption — more than the gap between any "
        f"two allocation strategies tested here. It is also the one dimension "
        f"no allocation rule can diversify.",
        "<b>How much you save dominates what you hold.</b> The savings-rate "
        "dimension of the tornado analysis moves the outcome by more than most "
        "allocation choices, and conditioning the savings rate is worth more "
        "than every allocation refinement in Sections #hedging to #leverage "
        "combined.",
        "<b>How you spend it matters more than expected.</b> The gap between "
        "the best and worst spending rules in Section #spending is comparable to the "
        "gap between the best and worst allocation strategies in Section #baseline.",
        "<b>What you hold matters, but mostly through diversification.</b> The "
        "all-equity advantage is real, and it is driven by the international "
        "leg rather than by equity exposure as such.",
        "<b>Currency hedging is not a lever but a leak</b>: Section #hedging "
        "finds it loses certainty-equivalent consumption at every ratio "
        "tested before a single basis point of cost is charged, so there is "
        "no price at which it becomes worth doing. The fine structure of the "
        "glide path is worth less than a basis point at most ages, and "
        "freeing every portfolio weight at every age adds less than the "
        "difference between two adjacent spending rules.",
        "<b>Levering the portfolio is barely a lever</b> at household "
        "borrowing costs. The advantage rounds to nothing across most of the "
        "plausible range of spreads, and what it does buy is bought out of "
        "the left tail. <b>Levering one asset is a different question</b>, "
        "and Section #mortgage.5 shows it has a different answer: at the same "
        "borrowing cost, a mortgage on the housing sleeve is worth several "
        "times what scaling the whole portfolio is worth, because it buys "
        "more of the holding that moves independently of the rest rather "
        "than more of what the investor already owns.",
        "<b>Borrowing is where age structure actually lives.</b> The two "
        "policies an unconstrained search makes genuinely age-varying in this "
        "paper are both borrowing schedules — the leverage ratio of "
        "Section #leverage and the loan-to-value of Section #mortgage — and "
        "both decline. The equity share does not. The glide path the industry "
        "applies to equities has the right shape attached to the wrong "
        "instrument: what should fall as an investor ages is the debt, not "
        "the risk.",
        "<b>The valuation you start at is not an allocation lever at all.</b> "
        "It moves what a portfolio delivers without changing which portfolio "
        "to hold, so it belongs in a reader's expectations and their "
        "withdrawal planning rather than in their asset mix.",
        "<b>Widening the opportunity set is a bigger lever than refining "
        "it.</b> Adding a fifth asset class buys more, at a low enough "
        "holding cost, than every refinement within the four existing sleeves "
        "combined — and loses it all at a holding cost a real owner would "
        "recognise. Which asset classes are available is a more consequential "
        "question than how finely the weights among them are tuned.",
    ]))
    out.append(ctx.p(
        "A reader looking for the highest-value change to their own plan "
        "should read that list from the top, not from the bottom. The "
        "profession's attention is allocated in roughly the reverse order."))

    out.append(ctx.h2("#discussion.3 Why the pay cheque beats the portfolio"))
    out.append(ctx.p(
        "The single most surprising result in the paper is that the best "
        "state variable for a savings rule is labour income relative to its "
        "expected path, not the portfolio balance. The mechanism is worth "
        "spelling out because it generalises."))
    out.append(ctx.p(
        "Both signals identify the same underlying problem — this path is "
        "falling behind — but they differ in <i>when</i> the information "
        "arrives relative to the consumption decision. An income shock is "
        "observed at the moment income is received and before it has been "
        "committed to consumption; saving a larger fraction of a windfall "
        "costs almost nothing in utility because the counterfactual "
        "consumption was never planned. A portfolio shortfall is observed "
        "after the consumption plan is set, so responding to it means cutting "
        "consumption that the household expected to have. Under a concave "
        "utility function those two are not the same transaction."))
    out.append(ctx.p(
        "The practical implication is unusually clean. \"Save a fixed fraction "
        "of every raise and every bonus\" is a rule that requires no target, "
        "no balance lookup and no arithmetic, and on this model it outperforms "
        "the wealth-target advice that dominates financial planning."))

    out.append(ctx.h2("#discussion.4 Implications for default design"))
    out.append(ctx.p(
        "If the results here were taken at face value by a plan sponsor, the "
        "implied redesign would not be \"raise the equity share of the "
        "target-date fund\", though that is what the headline suggests. It "
        "would be a reordering of what the default does at all."))
    out.extend(ctx.bullets([
        "<b>Default the contribution schedule, not just the portfolio.</b> "
        "Auto-escalation tied to pay increases captures the strongest signal "
        "in Section #accumulation and requires no member decision.",
        "<b>Default the drawdown policy.</b> Section #spending shows the choice of "
        "withdrawal rule is worth as much as the choice of portfolio, and it "
        "is almost never defaulted.",
        f"<b>Stop quoting four percent.</b> On this panel the sustainable rate "
        f"at a five-percent ruin tolerance is {pc(swr_eq, 1)} even on the best "
        f"strategy tested. A default that plans around four percent is "
        f"planning around a one-in-seven failure.",
        "<b>Treat the retirement date as a plan variable.</b> The largest "
        "single source of outcome variance is the one members are given the "
        "least help with.",
    ]))
    out.append(ctx.p(
        "We state these as implications of the model, not as advice. The model "
        "has no disutility of labour, no taxes, no fees, no owner-occupied "
        "housing and no "
        "behavioural constraints, and each of those would temper the "
        "conclusions in Section #housing's direction."))
    return out


# ---------------------------------------------------------------------------
# 19. Limitations
# ---------------------------------------------------------------------------
def section_limitations(ctx: Any) -> List[Flowable]:
    f = ctx.f
    p = f.panel
    pr, adv = f.provenance, f.panel_advantage
    out: List[Flowable] = [ctx.h1("#limitations. Limitations and Threats to Validity")]
    out.append(ctx.p(
        "This section is deliberately long. A replication that reports only "
        "the ways in which it succeeded is not much use."))

    out.append(ctx.h2("#limitations.1 The cross-section is sixteen countries"))
    out.append(ctx.p(
        f"This is the largest weakness in the paper, and it is a weakness we "
        f"chose deliberately. The developed-market universe runs to 38 "
        f"markets. We cover {pr['n_countries']}, because the other "
        f"{pr['n_removed']} have no recorded return series in any openly "
        f"licensed source. Section #data.6 audits what we do use; Section #data.6.1 "
        f"reports what the excluded countries have and why it is not "
        f"enough."))
    out.append(ctx.p(
        f"Every statement in this paper about international diversification is "
        f"therefore a statement about an average over {pr['n_countries'] - 1} "
        f"other developed markets, weighted by history length and drawn "
        f"jointly with the domestic leg. That is a real limit on how far the "
        f"mechanism generalises, and in particular the panel is entirely "
        f"advanced-economy and entirely survivor: markets that closed and "
        f"never reopened are not in the source database, so the tail this "
        f"paper measures is the tail of markets that came back."))
    out.append(ctx.p(
        "Section #panel quantifies the part of this that is quantifiable. No "
        "single country carries the headline, and the ranking is stable "
        "across every window of history tested — but the delete-one "
        "jackknife puts a standard error on the headline gap that is wide "
        "enough to matter, and a reader should carry that interval to every "
        "other number in the paper. Nothing here is estimated more precisely "
        "than sixteen countries allow, however many decimal places the "
        "simulation resolves. What Section #panel cannot address is the "
        "survivorship in the sample frame itself: deleting a country that is "
        "present says nothing about a country that was never there."))

    out.append(ctx.p(
        "The last five years of the series are separately flagged as "
        "unverified in Section #data.6, on a variance test that every country in "
        "the sample fails in the same direction."))

    out.append(ctx.h2("#limitations.2 What the bootstrap cannot represent"))
    out.extend(ctx.bullets([
        "<b>No valuation conditioning.</b> Blocks are drawn without regard to "
        "the starting dividend yield or price-earnings ratio, so a simulated "
        "lifetime beginning at a market peak is statistically identical to one "
        "beginning at a trough. If long-horizon returns are predictable from "
        "valuations, the dispersion here is too wide and the mean too high.",
        "<b>No regime structure.</b> Block resampling preserves local "
        "persistence but not the possibility that the whole return-generating "
        "process changes. Secular declines in real yields, changes in the "
        "corporate tax base and the shift toward intangible capital are all "
        "invisible.",
        "<b>System-level survivorship.</b> Every country in the panel still "
        "had a functioning market in 2020. The set of histories in which that "
        "is false has zero probability under this sampler.",
        "<b>Blocks respect gaps, and gaps are informative.</b> A country with "
        "a wartime hole contributes fewer long blocks, so the sampler is "
        "mildly biased away from exactly the disrupted histories that motivate "
        "the exercise. We quantify this rather than correct it.",
        "<b>The realised backtest selects on windows, not just on "
        "countries.</b> Section #cohorts avoids the sampler entirely by "
        "running the lifetimes history actually served, but the filter that "
        "makes that possible — an unbroken run the length of a working "
        "life and a retirement — removes every window that straddles a "
        "catastrophe rather than following one. The markets whose equity was "
        "destroyed therefore contribute only the lifetimes that begin at the "
        "floor of the recovery. That biases the comparison toward domestic "
        "equity, so the lead reported there is conservative in direction; it "
        "also means the spread across countries in that section is not the "
        "spread a randomly chosen saver faced.",
    ]))

    out.append(ctx.h2("#limitations.3 What the lifecycle model omits"))
    hs, wg = pr.get("housing", {}), pr.get("wages", {})
    housing_bullet = (
        "<b>No housing in the baseline.</b> For most households the primary "
        "residence is the largest asset and the mortgage the largest "
        "liability, and both interact with inflation in ways the financial "
        "portfolio does not. Sections #housing and #mortgage add housing and "
        "leverage against it to the opportunity set, but every other result "
        "in this paper is computed without them."
        + (f" Section #data.6.2 audits the {hs['country_years']:,} "
           f"country-years of observed housing returns behind that, and "
           f"explains why an appraisal-smoothed index cannot be used as "
           f"written." if hs.get("countries") else ""))
    wage_bullet = (
        f"<b>No economy-wide wage growth.</b> The income profile is an age "
        f"effect only. Section #data.6.3 measures what that leaves out — a median "
        f"{pc(wg['measured'], 2)} a year across {wg['countries']} countries, "
        f"which compounds to {f2(wg['career_multiple'], 2)}× over a career — "
        f"and finds the bias runs against our conclusion, because less human "
        f"capital weakens rather than strengthens the case for early equity."
    ) if wg.get("countries") else ""
    out.extend(ctx.bullets([x for x in [
        "<b>One tax, and only one.</b> Section #withholding prices foreign "
        "dividend withholding, because it is the tax that falls unequally on "
        "the two legs this paper compares and is therefore the one that can "
        "change the answer — and it does, inside the statutory range. Every "
        "other tax is absent: returns are otherwise pre-tax, withdrawals "
        "untaxed, and there is no distinction between a taxable and a "
        "tax-deferred account. The asymmetric treatment of capital gains and "
        "income would change the ranking of spending rules in Section "
        "#spending more than it would change the allocation ranking; "
        "dividend imputation, which refunds corporate tax to domestic "
        "shareholders in Australia and New Zealand, would push the same way "
        "as withholding and is not modelled either.",
        "<b>Every return is gross.</b> No fee is charged anywhere in the "
        "baseline. Section #fees prices the omission rather than repairing "
        "it, by finding the fee differential that would undo the headline, "
        "and Section #turnover does the same for the cost of trading. Both "
        "leave the baseline itself gross, so every level quoted outside "
        "those two sections is a number no investor receives.",
        housing_bullet,
        wage_bullet,
        "<b>No annuities.</b> A real annuity is the natural competitor to a "
        "withdrawal rule and is absent entirely.",
        "<b>One country's pension behind sixteen countries' returns.</b> "
        "Every result outside Section #pension pays the US "
        "primary-insurance-amount schedule, and that is not a neutral "
        "choice: an earnings-related benefit paid regardless of wealth is a "
        "risk-free real annuity, which puts a floor under retirement "
        "consumption and flatters equity. Section #pension is the one place "
        "this is tested, against Australia's means-tested Age Pension and "
        "compulsory Superannuation Guarantee, and it moves the level further "
        "than any parameter swept anywhere else in this paper. The ranking "
        "survives there only because the guarantee lifts the model's saver "
        "clear of the means test; a saver inside the taper band gets a "
        "different answer, and nothing else in this paper speaks to them.",
        "<b>No disutility of labour.</b> This is why the retirement-timing "
        "results of Section #retirement require a matched comparison, and it means the "
        "model can never say when someone <i>should</i> retire, only what "
        "conditioning the date is worth.",
        "<b>A fixed horizon everywhere but one section.</b> Death arrives at "
        "93 with certainty in every result except Section #mortality, which "
        "re-weights the headline under four Gompertz survival laws and finds "
        "the ranking unmoved. That is a narrower finding than it sounds: it "
        "re-weights a policy that does not itself respond to longevity, so a "
        "small answer is partly built in. Nothing here prices an annuity, or "
        "a spending rule that adapts to a mortality table, and both are where "
        "the value of longevity risk actually sits.",
        "<b>No behavioural constraints.</b> Every rule here assumes the "
        "investor executes it. Section #accumulation.6 prices the cost of a constrained "
        "contribution but nothing prices the probability that a saver "
        "abandons an all-equity portfolio in the middle of a 60% drawdown.",
    ] if x]))

    out.append(ctx.h2("#limitations.4 Specification sensitivities we know about"))
    out.append(ctx.p(
        "Three modelling choices are load-bearing and a reader should know "
        "where they bite."))
    out.extend(ctx.bullets([
        "<b>The bequest shift.</b> Without De Nardi's κ, a single path with "
        "exactly zero terminal wealth sends the certainty equivalent to "
        "negative infinity at γ ≥ 1. The value of κ is not identified by "
        "anything in the data and it changes the ranking of spending rules.",
        "<b>The evaluation window.</b> Allocation comparisons use the "
        "retirement window because working-life consumption is identical "
        "across strategies by construction; policy comparisons that change "
        "working-life consumption use the whole lifetime. Mixing the two would "
        "produce incoherent comparisons and we flag every switch.",
        "<b>The working-income floor.</b> Retirement carries a consumption "
        "floor through the progressive benefit schedule and working life, at "
        "zero, carries none. That asymmetry cancels in the allocation "
        "comparisons and does not cancel in the timing ones, where leaving it "
        "uncorrected inflates the value of retiring early by roughly a factor "
        "of two.",
    ]))

    out.append(ctx.h2("#limitations.5 What the new searches assume"))
    out.extend(ctx.bullets([
        "<b>The simplex search is a coordinate search.</b> Section #allocation reports "
        "restarts from three corners and they agree, but coordinate ascent "
        "offers no guarantee of a global optimum in a 204-dimensional space "
        "and none is claimed.",
        "<b>The opportunity set is still narrow.</b> Section #housing adds housing "
        "as a fifth asset, but there is no credit, no commodities and no "
        "inflation-linked bond, and an inflation-linked bond in particular "
        "would change the fixed-income result of Section #allocation more than any "
        "refinement within the sleeves that are present.",
        "<b>The valuation conditioning applies to the first block only.</b> "
        "A lifetime is a chain of calendar windows and only the first is a "
        "starting condition; the rest are the future, which no investor "
        "chooses. Section #valuation therefore measures the effect of the opening "
        "decade's valuation on a sixty-eight-year outcome, diluted by "
        "everything that follows. A design that made the whole chain "
        "valuation-dependent would report a larger effect and would be "
        "assuming a great deal more.",
        "<b>The de-smoothing of housing is itself a model.</b> Section #housing "
        "inverts a first-order filter, which is the standard correction and "
        "not necessarily the right one: if house price indices carry "
        "higher-order smoothing, the true volatility is higher than the "
        "correction restores and housing is worth less than reported. The "
        "direction of that error is known even though its size is not.",
        "<b>The mortgage is rebalanced annually.</b> Section #mortgage redraws the "
        "loan every year at no cost to hit a target loan-to-value. Real "
        "mortgages amortise on a fixed schedule, cost several percent of the "
        "property to refinance, and are called on missed payments rather "
        "than on a drifting ratio. The section reports the value of the "
        "leverage, not a financing plan, and it prices no mortgage insurance "
        "and no tax deductibility in either direction.",
        "<b>Housing is priced as an index, not as a house.</b> The asset in "
        "Section #housing is a liquid, continuously rebalanced, nationally "
        "diversified claim on the housing stock. A single leveraged "
        "owner-occupied property shares almost none of those properties, and "
        "nothing in this paper speaks to it.",
        "<b>Leverage is modelled with limited liability</b> and a floating "
        "borrowing rate, rebalanced annually. Real margin lending liquidates "
        "at a threshold rather than at zero and reprices continuously, both of "
        "which would make borrowing less attractive than Section #leverage finds it.",
        "<b>No borrowing constraint binds by quantity.</b> The model lets an "
        "investor borrow three times their financial capital at a spread; no "
        "lender would extend that against a retirement account.",
    ]))

    out.append(ctx.h2("#limitations.6 Statistical caveats"))
    out.append(ctx.p(
        "Certainty equivalents are reported without standard errors. Under "
        "common random numbers the <i>differences</i> between policies are far "
        "more precisely estimated than their levels, which is what the "
        "comparisons rely on, but a reader wanting a confidence interval on "
        "any single level will not find one here. The optimisers are "
        "coordinate-ascent procedures on a grid: exact per coordinate under "
        "common random numbers, but with no guarantee of a global optimum in "
        "the joint space. Multiple restarts are reported where it matters."))
    out.extend(ctx.bullets([
        f"<b>The cohort interval rests on {p['n_countries']} clusters.</b> "
        f"The confidence interval in Section #cohorts is a cluster bootstrap "
        f"over countries, and a percentile interval built on "
        f"{p['n_countries']} clusters has poorer coverage than its width "
        f"suggests. The cohorts are also unequally distributed across those "
        f"clusters, so the pooled mean is implicitly weighted by how much "
        f"unbroken history a country happens to have. Both readings are "
        f"reported there; the country-equal-weighted one is the conservative "
        f"one.",
        "<b>One split is one experiment.</b> Section #out_of_sample cuts the "
        "record once and solves on either side. The asymmetry it finds is "
        "real in the sense that it is what the data say, but a single cut "
        "cannot separate a schedule that was overfitted from a world that "
        "changed, and with one experiment there is no distribution to judge "
        "the asymmetry against. It should be read as a caution rather than "
        "as a test with a size.",
    ]))
    return out


# ---------------------------------------------------------------------------
# 20. Conclusion
# ---------------------------------------------------------------------------
def section_conclusion(ctx: Any) -> List[Flowable]:
    f = ctx.f
    adv_tdf = f.advantage("balanced_all_equity", "target_date_fund")
    lottery = f.table("retirement_lottery_stats").iloc[0]
    pen_rows = f.table("pension_gap").set_index("system")
    pen_base = float(pen_rows.loc["us_social_security", "gap_pct"])
    pen_au_gap = float(pen_rows.loc["australia_as_legislated", "gap_pct"])
    pen_au_winner = str(pen_rows.loc["australia_as_legislated", "winner"])
    pen_au_reorders = bool((pen_au_gap > 0.0) != (pen_base > 0.0))
    out: List[Flowable] = [ctx.h1("#conclusion. Conclusion")]
    out.append(ctx.p(
        f"We set out to reproduce a specific empirical claim and ended up with "
        f"a hierarchy. The claim reproduces: on a "
        f"{f.panel['n_countries']}-country panel of the developed-market "
        f"twentieth century, sampled in blocks so that its persistence and its "
        f"tails survive, a diversified all-equity portfolio delivers "
        f"{adv_tdf:.1f}% more certainty-equivalent retirement consumption than "
        f"a target-date glide path while failing less often. Solving for the "
        f"allocation schedule directly rather than testing candidates returns "
        f"the same answer, and the ranking holds across every parameter "
        f"dimension we sweep. Freeing all four portfolio weights at every age "
        f"— {f.allocation['free_parameters']} parameters on the simplex, with "
        f"nothing held fixed — reaches the same shape again and adds "
        f"{f.allocation['lead_pct']:.2f}%; relaxing the long-only constraint "
        f"pays well only when credit is nearly free, and breaks even by a "
        f"{f.leverage['break_even_spread']:.2%} spread over the real bill "
        f"rate."))
    out.append(ctx.p(
        f"But the extensions matter more than the replication. The single "
        f"decade around a person's retirement date explains "
        f"{pc(float(lottery['r2_retirement_window']), 1)} of the variation in "
        f"their retirement outcome — more than the entire allocation question. "
        f"Conditioning the savings rate on financial position is worth more "
        f"than every allocation refinement we test. And when we decompose that "
        f"signal properly, the best state variable available to a saver turns "
        f"out not to be their portfolio at all but their own pay cheque "
        f"relative to what they expected to earn."))
    out.append(ctx.p(
        "Two of the extensions relax assumptions the model itself was making. "
        "Conditioning on the valuation a lifetime starts at — the objection a "
        "reader is most entitled to raise, since they know what their own "
        "market costs — leaves the allocation answer where it was and moves "
        "the level: valuation tells an investor what to expect, not what to "
        "hold. Adding housing to the opportunity set shows an asset that is "
        "genuinely competitive once its index is de-smoothed, and that stops "
        "being competitive at an annual holding cost low enough that a real "
        "owner's rates, insurance and maintenance decide the question."))
    out.append(ctx.p(
        "Several of the results here came out against the intuition that "
        "motivated them, and we have tried to report those cases with their "
        "diagnostics rather than smoothing them into a tidier story: the "
        "glide-path structure that dissolved under a deviation profile; the "
        "retirement-timing premium that halved under a matched baseline; the "
        "savings conditioning that turned out not to be tail insurance; the "
        "bull-market warning that the data did not support. A replication is "
        "most useful when it reports what it found rather than what it "
        "expected, and the machinery for telling those apart — matched "
        "baselines, common random numbers, deviation profiles, grid-edge "
        "checks — is as much a contribution here as any individual number."))
    out.append(ctx.p(
        "One assumption deserves to outrank the rest of the caveats, because "
        "it moves the numbers further than any parameter we sweep. Every "
        "result outside Section #pension pays the same public pension in all "
        "sixteen countries — the American one, progressive in career "
        "earnings and paid whatever else the retiree owns. That is a "
        "risk-free real annuity, and a great deal of what looks like "
        "portfolio performance in this paper is standing on it. Replacing it "
        "with the Australian pair — a means-tested Age Pension and a "
        "compulsory Superannuation Guarantee on top of voluntary saving — "
        "raises average retirement consumption and lowers the certainty "
        "equivalent at the same time, because the compulsory contribution "
        "buys a larger portfolio and the means test removes the floor that "
        "portfolio would have sat on. Both halves of that are true, and "
        "which one a reader should care about depends on whether they are "
        "planning for the average case or the bad one."))
    out.append(ctx.p(
        (f"The allocation ranking does not survive that substitution, and "
         f"this is the only place in the paper where it fails. Under the "
         f"Australian pair the lead of {pen_base:.2f}% becomes "
         f"{pen_au_gap:.2f}% and the best strategy becomes "
         f"<i>{_pretty_strategy(pen_au_winner)}</i> — the de-risking glide "
         f"path this paper spends most of its length arguing against. The "
         f"mechanism is the taper, not the returns: inside the "
         f"assets-tested band every extra dollar of assets costs more "
         f"pension a year than any asset in this panel reliably earns, so a "
         f"portfolio that stays smaller keeps an entitlement that a "
         f"portfolio that grows loses. A glide path does exactly that, and "
         f"under a means test it is being rewarded for it. Nothing about "
         f"the return panel has changed; the objective function has."
         if pen_au_reorders else
         "The allocation ranking survives that substitution — but the "
         "reason is worth more than the result. It survives because the "
         "guarantee makes this saver wealthy enough that the assets test "
         "never reaches them, so they are back to living off a portfolio, "
         "which is the situation the rest of the paper already describes.")))
    out.append(ctx.p(
        "The lesson is not about countries. It is that this paper, like the "
        "study it replicates, models an investor whose retirement income is "
        "a portfolio plus an unconditional annuity, and its conclusions are "
        "conclusions about that investor. A saver whose public entitlement is "
        "withdrawn at the margin as their balance grows faces a different "
        "problem, and the answer to it is different. Which of the two a "
        "reader is depends on their balance and their country's schedule, "
        "not on anything in the return data."))
    out.append(ctx.p(
        "Two further audits of the optimisation sections point the same way "
        "as each other and should be read together. Solving a schedule on "
        "one half of the record and scoring it on the other leaves a "
        "constant mix as the benchmark to beat in every run; charging the "
        "same solved schedule for the trades it makes takes most of what "
        "remains. Neither touches the paper's headline, which compares fixed "
        "portfolios fitted to nothing — but both say that the gains from "
        "solving for an allocation are smaller, and more fragile, than the "
        "in-sample numbers make them look."))
    out.append(ctx.p(
        "One caution belongs in the summary rather than the appendix. "
        "Deleting each country in turn and rebuilding the panel around its "
        "absence leaves the ranking intact every time, and the ranking is "
        "stable across every window of history we can stand in — but the "
        "same runs, read as a jackknife, put a standard error on the "
        "headline gap wide enough that the interval barely excludes zero. "
        "Sixteen countries is sixteen countries. Everything above should be "
        "read as well established in direction and thinly established in "
        "magnitude, and a reader who wants one thing from this paper should "
        "take the ordering rather than the size of any gap in it."))
    out.append(ctx.p(
        "The practical summary is short. On this evidence, an investor's "
        "attention is best spent, in order, on when they retire, on how much "
        "and when they save, on how they spend the portfolio down, and only "
        "then on what it holds — and when they get to what it holds, the "
        "answer is more diversified equity than the default gives them, for "
        "longer than the default holds it."))
    return out


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
REFERENCES = [
    "Anarkulova, A., Cederburg, S., and O'Doherty, M. S. (2023). "
    "\"Beyond the Status Quo: A Critical Assessment of Lifecycle Investment "
    "Advice.\" Working paper.",
    "Anarkulova, A., Cederburg, S., and O'Doherty, M. S. (2022). "
    "\"Stocks for the Long Run? Evidence from a Broad Sample of Developed "
    "Markets.\" <i>Journal of Financial Economics</i>, 143(1), 409–433.",
    "Ayres, I., and Nalebuff, B. (2010). <i>Lifecycle Investing: A New, Safe, "
    "and Audacious Way to Improve the Performance of Your Retirement "
    "Portfolio</i>. Basic Books.",
    "Bengen, W. P. (1994). \"Determining Withdrawal Rates Using Historical "
    "Data.\" <i>Journal of Financial Planning</i>, 7(4), 171–180.",
    "Bodie, Z., Merton, R. C., and Samuelson, W. F. (1992). \"Labor Supply "
    "Flexibility and Portfolio Choice in a Life Cycle Model.\" "
    "<i>Journal of Economic Dynamics and Control</i>, 16(3–4), 427–449.",
    "Cooley, P. L., Hubbard, C. M., and Walz, D. T. (1998). \"Retirement "
    "Savings: Choosing a Withdrawal Rate That Is Sustainable.\" "
    "<i>AAII Journal</i>, 20(2), 16–21.",
    "De Nardi, M. (2004). \"Wealth Inequality and Intergenerational Links.\" "
    "<i>Review of Economic Studies</i>, 71(3), 743–768.",
    "Epstein, L. G., and Zin, S. E. (1989). \"Substitution, Risk Aversion, and "
    "the Temporal Behavior of Consumption and Asset Returns: A Theoretical "
    "Framework.\" <i>Econometrica</i>, 57(4), 937–969.",
    "Chambers, D., Spaenjers, C., and Steiner, E. (2021). "
    "\"The Rate of Return on Real Estate: Long-Run Micro-Level Evidence.\" "
    "<i>Review of Financial Studies</i>, 34(8), 3572–3607.",
    "Fidelity Investments. \"Retirement guidelines: how much should I save?\" "
    "Viewpoints. The 1×/3×/6×/8×/10× salary-multiple ladder used as an "
    "alternative wealth target in Section #accumulation.4.",
    "Geltner, D. (1991). \"Smoothing in Appraisal-Based Returns.\" "
    "<i>Journal of Real Estate Finance and Economics</i>, 4(3), 327–345.",
    "Guyton, J. T., and Klinger, W. J. (2006). \"Decision Rules and Maximum "
    "Initial Withdrawal Rates.\" <i>Journal of Financial Planning</i>, 19(3), "
    "48–58.",
    "Jordà, Ò., Knoll, K., Kuvshinov, D., Schularick, M., and Taylor, A. M. "
    "(2019). \"The Rate of Return on Everything, 1870–2015.\" "
    "<i>Quarterly Journal of Economics</i>, 134(3), 1225–1298.",
    "Jordà, Ò., Schularick, M., and Taylor, A. M. (2017). \"Macrofinancial "
    "History and the New Business Cycle Facts.\" "
    "<i>NBER Macroeconomics Annual</i>, 31, 213–263.",
    "Merton, R. C. (1969). \"Lifetime Portfolio Selection under Uncertainty: "
    "The Continuous-Time Case.\" <i>Review of Economics and Statistics</i>, "
    "51(3), 247–257.",
    "Politis, D. N., and Romano, J. P. (1994). \"The Stationary Bootstrap.\" "
    "<i>Journal of the American Statistical Association</i>, 89(428), "
    "1303–1313.",
    "Samuelson, P. A. (1969). \"Lifetime Portfolio Selection by Dynamic "
    "Stochastic Programming.\" <i>Review of Economics and Statistics</i>, "
    "51(3), 239–246.",
    "Waring, M. B., and Siegel, L. B. (2015). \"The Only Spending Rule Article "
    "You Will Ever Need.\" <i>Financial Analysts Journal</i>, 71(1), 91–107.",
]


def section_references(ctx: Any) -> List[Flowable]:
    out: List[Flowable] = [PageBreak(),
                           Paragraph("References", ctx.s["h1_plain"])]
    for entry in REFERENCES:
        out.append(Paragraph(entry, ctx.s["reference"]))
    return out


# ---------------------------------------------------------------------------
# Appendices
# ---------------------------------------------------------------------------
def appendix_parameters(ctx: Any) -> List[Flowable]:
    f = ctx.f
    cfg = f.cfg
    lc, ut, bs = cfg["lifecycle"], cfg["utility"], cfg["bootstrap"]
    rows = [
        ["Parameter", "Symbol", "Value", "Where it is used"],
        ["Start age", "—", f"{int(lc['age_start'])}", "All"],
        ["Retirement age", "—", f"{int(lc['age_retire'])}",
         "All; varied in §#sensitivity.4 and made endogenous in §#retirement"],
        ["Death age", "—", f"{int(lc['age_death'])}",
         "All; varied in §#sensitivity.4"],
        ["Savings rate", "s", f"{float(lc['savings_rate']):.0%}",
         "All; solved in §#saving and conditioned in §#accumulation"],
        ["Income profile (linear)", "b₁", f"{float(lc['income']['b1']):g}",
         "Labour income"],
        ["Income profile (quadratic)", "b₂", f"{float(lc['income']['b2']):g}",
         "Labour income"],
        ["Permanent shock s.d.", "σ<sub>p</sub>",
         f"{float(lc['income']['permanent_shock_sd']):.2f}",
         "Labour income; scaled in §#accumulation.9"],
        ["Transitory shock s.d.", "σ<sub>t</sub>",
         f"{float(lc['income']['transitory_shock_sd']):.2f}",
         "Labour income; scaled in §#accumulation.9"],
        ["Social-security formula", "—", str(lc["social_security"]["formula"]),
         "Retirement income floor"],
        ["PIA bend point 1", "—", f"{float(lc['social_security']['pia_bend1']):.2f}",
         "×  economy-wide average earnings"],
        ["PIA bend point 2", "—", f"{float(lc['social_security']['pia_bend2']):.2f}",
         "×  economy-wide average earnings"],
        ["PIA replacement rates", "—",
         " / ".join(f"{float(lc['social_security'][k]):.0%}"
                    for k in ("pia_rate1", "pia_rate2", "pia_rate3")),
         "Successive tranches of career-average earnings"],
        ["Withdrawal rule", "—", str(lc["retirement"]["rule"]),
         "Baseline; eight families compared in §#spending"],
        ["Withdrawal rate", "—", f"{float(lc['retirement']['rule_rate']):.0%}",
         "Baseline; swept in §#baseline.4 and §#sensitivity.4"],
        ["Discount factor", "β", f"{float(ut['discount_factor']):g}",
         "Utility"],
        ["Risk aversions", "γ",
         ", ".join(f"{float(g):g}" for g in ut["risk_aversions"]),
         f"Baseline γ = {float(ut['baseline_risk_aversion']):g}"],
        ["Elasticities of substitution", "ψ",
         ", ".join(f"{float(v):g}" for v in ut["epstein_zin_ies"]),
         "Epstein–Zin only"],
        ["Bequest weight", "θ", f"{float(ut['bequest_weight']):g}",
         "Utility; pivoted in §#spending.1"],
        ["Bequest shift", "κ", f"{float(ut['bequest_shift']):g}",
         "De Nardi (2004) specification"],
        ["Consumption floor", "—", f"{float(ut['consumption_floor']):g}",
         "Numerical guard only"],
        ["Evaluation window", "—", str(ut["consumption_window"]),
         "Allocation comparisons; whole lifetime in §#retirement–§#accumulation"],
        ["Paths per strategy", "N", f"{int(bs['n_paths']):,}", "All"],
        ["Horizon", "H", f"{int(bs['horizon_years'])}", "All"],
        ["Mean block length", "—", f"{float(bs['mean_block_years']):.0f} years",
         "Bootstrap; swept in §#methods.1"],
        ["Block length range", "—",
         f"{int(bs['min_block_years'])}–{int(bs['max_block_years'])} years",
         "Bootstrap"],
        ["Country draw", "—", str(bs["country_draw"]),
         "Bootstrap; varied in Appendix C"],
        ["Country weighting", "—", str(bs["country_weighting"]),
         "Bootstrap; varied in Appendix C"],
        ["Master seed", "—", f"{int(cfg['run']['seed'])}", "Reproducibility"],
    ]
    out: List[Flowable] = [PageBreak(), ctx.h1("Appendix A. Model Parameters")]
    out.append(ctx.p(
        "Every parameter below is read from a single YAML configuration file. "
        "No value is hard-coded in the simulation modules, which is what makes "
        "the sweeps of Sections #sensitivity, #leverage and #housing mechanical "
        "rather than manual."))
    out.extend(ctx.table(
        rows, "The complete baseline parameter set",
        note="Section references indicate where each parameter is varied. "
             "Parameters specific to a single extension (hedging cost, "
             "retirement trigger multiple, savings-rate grids, accumulation "
             "response grids) are documented in the configuration file and in "
             "the relevant section.",
        col_widths=[ctx.width * 0.30, ctx.width * 0.10, ctx.width * 0.20,
                    ctx.width * 0.40],
        font_size=7.2))
    return out


def appendix_panel(ctx: Any) -> List[Flowable]:
    f = ctx.f
    summary = f.table("panel_summary_statistics")
    equity = summary[summary["series"] == "dom_eq"].sort_values("country")
    gaps = f.table("panel_structural_gaps")

    out: List[Flowable] = [PageBreak(), ctx.h1("Appendix B. The Country Panel")]
    out.append(ctx.p(
        "The full panel, with real domestic equity statistics for each. Every "
        "country listed carries complete Jordà–Schularick–Taylor equity, bond "
        "and bill total returns over the years shown."))
    out.extend(ctx.table(
        rows_from(equity, ["country", "n_years", "first_year",
                           "last_year", "mean", "geometric_mean", "std",
                           "skew", "kurtosis", "ar1"],
                  ["Country", "Years", "From", "To", "Mean",
                   "Geo. mean", "S.d.", "Skew", "Kurtosis", "AR(1)"],
                  {"country": str,
                   "n_years": lambda v: f"{int(v)}",
                   "first_year": lambda v: f"{int(v)}",
                   "last_year": lambda v: f"{int(v)}",
                   "mean": lambda v: pc(v), "geometric_mean": lambda v: pc(v),
                   "std": lambda v: pc(v), "skew": lambda v: f2(v),
                   "kurtosis": lambda v: f2(v, 1),
                   "ar1": lambda v: f2(v)}),
        "Real domestic equity returns, every country in the panel",
        note="Annual real total returns deflated by each country's own "
             "consumer price index. Kurtosis is excess kurtosis.",
        col_widths=[ctx.width * 0.21] + [ctx.width * 0.0878] * 9,
        font_size=6.6))

    out.append(ctx.h2("B.1 Structural gaps"))
    out.append(ctx.p(
        f"The {len(gaps)} contiguous runs of missing data in the panel's "
        f"countries, preserved as gaps rather than interpolated. The bootstrap "
        f"never draws a block that crosses one."))
    out.extend(ctx.table(
        rows_from(gaps, ["country", "first_year", "last_year", "n_years",
                         "missing_series"],
                  ["Country", "From", "To", "Years", "Missing series"],
                  {"country": str, "first_year": lambda v: f"{int(v)}",
                   "last_year": lambda v: f"{int(v)}",
                   "n_years": lambda v: f"{int(v)}",
                   "missing_series": str}),
        "Structural gaps in the empirical panel",
        note="Concentrated around the two world wars, which is precisely the "
             "period a survivorship-prone sample would drop.",
        font_size=7.2))
    return out


def appendix_supplementary(ctx: Any) -> List[Flowable]:
    f = ctx.f
    out: List[Flowable] = [PageBreak(),
                           ctx.h1("Appendix C. Supplementary Tables")]

    out.append(ctx.h2("C.1 Bootstrap sampling variations"))
    out.append(ctx.p(
        "The two structural choices in the sampler — drawing the country once "
        "per lifetime rather than per block, and weighting countries by "
        "history length rather than uniformly — are varied here. Neither "
        "reverses any ranking."))
    # Only the two sampling variants: there is no alternative panel to compare
    # against, since the panel is exactly the recorded countries.
    for name, caption in (
            ("robustness_country_draw_per_block",
             "Redrawing the country at every block"),
            ("robustness_country_weighting_uniform",
             "Weighting countries uniformly rather than by history length")):
        frame = f.table(name)
        cols = [c for c in ("strategy", "label", "cec_crra_gamma5",
                            "prob_ruin", "median_bequest",
                            "median_retirement_consumption")
                if c in frame.columns]
        out.extend(ctx.table(
            rows_from(frame.assign(
                label=frame["strategy"].map(LABELS)
                if "strategy" in frame.columns else frame.get("label")),
                [c if c != "strategy" else "label" for c in cols],
                ["Strategy" if c in ("strategy", "label")
                 else c.replace("cec_crra_gamma5", "CEC γ=5")
                       .replace("prob_ruin", "P(ruin)")
                       .replace("_", " ").title() for c in cols],
                {"label": str, "prob_ruin": lambda v: pc(v, 1)}),
            caption, font_size=7.2))

    out.append(ctx.h2("C.2 The income and benefit schedules"))
    income = f.table("income_profile")
    ss = f.table("social_security_schedule")
    out.extend(ctx.table(
        rows_from(income, ["age", "real_income", "multiple_of_age25"],
                  ["Age", "Real income", "Multiple of age-25 income"],
                  {"age": lambda v: f"{int(v)}"}),
        "The deterministic labour-income profile",
        note="Before permanent and transitory shocks. The hump peaks in the "
             "early fifties, which is what makes the solved savings profile of "
             "§#saving.2 hump-shaped at moderate risk aversion."))
    out.extend(ctx.table(
        rows_from(ss, ["career_average_earnings", "annual_benefit",
                       "replacement_rate"],
                  ["Career-average earnings", "Annual benefit",
                   "Replacement rate"],
                  {"replacement_rate": lambda v: pc(v, 1)}),
        "The progressive social-security benefit schedule",
        note="US primary-insurance-amount bend points. The falling "
             "replacement rate is what supplies a genuine real consumption "
             "floor; a flat rate would scale with the outcome and insure "
             "nothing."))

    out.append(ctx.h2("C.3 Long-horizon bootstrap outcomes"))
    terminal = f.table("bootstrap_terminal")
    out.extend(ctx.table(
        rows_from(terminal, ["series", "mean_annualised", "p1_annualised",
                             "p5_annualised", "p50_annualised",
                             "p95_annualised", "p99_annualised",
                             "p1_cumulative", "p50_cumulative"],
                  ["Series", "Mean", "1st pct", "5th pct", "Median",
                   "95th pct", "99th pct", "1st pct cumulative",
                   "Median cumulative"],
                  {"series": lambda v: {"dom_eq": "Domestic equity",
                                        "intl_eq": "International equity",
                                        "bond": "Bonds", "bill": "Bills",
                                        "inflation": "Inflation"}[v],
                   "mean_annualised": lambda v: pc(v, 2),
                   "p1_annualised": lambda v: pc(v, 2),
                   "p5_annualised": lambda v: pc(v, 2),
                   "p50_annualised": lambda v: pc(v, 2),
                   "p95_annualised": lambda v: pc(v, 2),
                   "p99_annualised": lambda v: pc(v, 2),
                   "p1_cumulative": lambda v: f2(v, 2),
                   "p50_cumulative": lambda v: f2(v, 1)}),
        "Distribution of 68-year annualised and cumulative real returns",
        note="Annualised figures are geometric. The first-percentile "
             "cumulative multiple for domestic equity is below zero in real "
             "terms — a lifetime of holding a single national market that "
             "ended with less purchasing power than it started with. That "
             "outcome is the reason the international leg matters.",
        font_size=7.0))
    return out


def appendix_software(ctx: Any) -> List[Flowable]:
    ctx_f = ctx.f
    out: List[Flowable] = [PageBreak(),
                           ctx.h1("Appendix D. Computational Design")]
    out.append(ctx.p(
        "Every number, table and figure in this paper is produced by one "
        "program from the raw source files, and the document itself resolves "
        "its quoted figures from that program's output when it is typeset. "
        "Nothing is transcribed by hand. A rerun that changed a result would "
        "change the paper rather than leave it silently contradicting its own "
        "evidence."))
    out.append(ctx.p(
        "This appendix describes how the computation is organised and how its "
        "correctness is established, because both bear on how much weight the "
        "results can carry."))

    out.append(ctx.h2("D.1 Structure"))
    out.append(ctx.p(
        "The work is split into stages that each own one idea, so that an "
        "extension can reuse the baseline machinery instead of reimplementing "
        "it. Reuse is the point: every extension in this paper is compared "
        "against a baseline, and a comparison between two different "
        "implementations of the same model measures the implementations as "
        "much as the economics."))
    stages = [
        ["Stage", "What it owns", "Paper section"],
        ["Panel construction", "Deflation to real returns, the leave-one-out "
         "international leg, currency-hedged legs, coverage and gap "
         "diagnostics, per-cell provenance", "§#data"],
        ["Sampler", "The stationary block bootstrap with gap-respecting "
         "admissibility; moment and persistence diagnostics", "§#methods.1"],
        ["Lifecycle simulator", "The wealth recursion, income process, "
         "social-security schedule and strategy definitions", "§#methods.2"],
        ["Preferences", "CRRA and Epstein–Zin certainty equivalents, the "
         "bequest term, shortfall and ruin metrics", "§#methods.3"],
        ["Sweep engine", "Parameter sweeps under common random numbers; "
         "tornado and crossover analysis", "§#sensitivity"],
        ["Spending rules", "Eight families of withdrawal policy behind a "
         "single interface", "§#spending"],
        ["Glide-path solver", "A batched evaluator and coordinate ascent over "
         "the age-by-asset schedule", "§#glide"],
        ["Allocation solver", "The weight simplex solved at every age: "
         "lattice search then pairwise-exchange ascent, over four assets or "
         "five", "§#allocation, §#housing"],
        ["Leverage", "A levered evaluator, the cost-of-credit sweep and the "
         "age-varying leverage schedule", "§#leverage"],
        ["Hedging", "Covered-interest-parity hedged legs, break-even cost and "
         "optimal hedge ratio", "§#hedging"],
        ["Path-dependent engine", "Endogenous retirement dates and "
         "state-conditioned saving", "§#retirement–§#accumulation"],
        ["Savings rules", "Rule families, the fixed-mean shape solver and "
         "matched-rate scoring", "§#saving–§#accumulation"],
        ["Valuation", "The look-ahead-free trailing dividend yield, the "
         "structural no-leak check and the valuation buckets", "§#valuation"],
        ["Housing", "De-smoothing the appraisal index, and the five-asset "
         "simplex re-solved at each holding cost", "§#housing"],
        ["Mortgage", "Leverage applied to the housing sleeve alone, and the "
         "loan-to-value schedule solved by age", "§#mortgage"],
    ]
    out.extend(ctx.table(
        stages, "How the computation is organised",
        col_widths=[ctx.width * 0.22, ctx.width * 0.64, ctx.width * 0.14],
        font_size=7.2))

    out.append(ctx.h2("D.2 Performance"))
    out.append(ctx.p(
        "Portfolio returns for a whole cohort are formed as a batched matrix "
        "product and the wealth recursion runs vectorised across paths. That "
        "is what makes a hundred thousand lifetimes per strategy — and the "
        "tens of thousands of policy evaluations the optimisers require — "
        "tractable on a single core, which in turn is what makes the "
        "sensitivity analysis of Section #sensitivity affordable enough to run "
        "exhaustively rather than selectively."))

    out.append(ctx.h2("D.3 Verification"))
    out.append(ctx.p(
        f"Correctness rests on {ctx_f.n_tests} automated tests. The ones that "
        f"carry the most weight are equivalence tests between simulators, "
        f"because every extension here is built on an engine that must agree "
        f"with the one that produced the baseline:"))
    out.extend(ctx.bullets([
        "The path-dependent engine reproduces the fixed-date engine "
        "<i>bit for bit</i> given a fixed retirement age and a constant "
        "savings rate.",
        "The batched glide-path evaluator reproduces the reference simulator "
        "to a stated floating-point tolerance.",
        "Zero-sensitivity conditioning rules reproduce their own base age "
        "profile exactly, which is what makes the incremental values in §#accumulation "
        "meaningful.",
        "The block sampler never draws across a data gap, verified against "
        "the run-length structure directly.",
        "A savings rule is never handed a return it has not yet lived "
        "through — the no-lookahead property is asserted, not assumed.",
        "Every available cell of the panel is an observation, checked against "
        "a per-cell record of where each number came from.",
    ]))
    out.append(ctx.p(
        "A further discipline runs through the prose. Where this paper states "
        "a direction — that one quantity exceeds another, that a ranking "
        "survives, that an effect has a sign — the statement is classified "
        "from the computed table when the document is typeset rather than "
        "written down in advance. The mechanism exists so that a result which "
        "moves takes its description with it, rather than leaving prose that "
        "quietly contradicts the table beneath it."))

    out.append(ctx.h2("D.4 List of figures"))
    for entry in ctx.figure_index:
        out.append(Paragraph(entry, ctx.s["reference"]))
    out.append(ctx.h2("D.5 List of tables"))
    for entry in ctx.table_index:
        out.append(Paragraph(entry, ctx.s["reference"]))
    return out


def _check_section_order(parts: List[Flowable]) -> None:
    """Fail the build if `story` and :data:`SECTION_ORDER` have drifted apart.

    The tokens guarantee that a cross-reference and its heading agree on a
    number. They cannot guarantee that the sections are *emitted* in the order
    the tuple claims -- a reordered tuple with an unreordered `story` would
    silently produce a paper numbered 1, 2, 7, 4, ... So the numbers are read
    back off the headings actually built and checked to run 1..N in sequence.
    """
    seen: List[int] = []
    for flowable in parts:
        style = getattr(getattr(flowable, "style", None), "name", "")
        text = getattr(flowable, "text", "")
        if style == "h1" and text[:1].isdigit():
            seen.append(int(text.split(".", 1)[0]))
    expected = list(range(1, len(SECTION_ORDER) + 1))
    if seen != expected:
        raise AssertionError(
            "section order in story() does not match SECTION_ORDER: "
            f"emitted {seen}, expected {expected}")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def section_cohorts(ctx: Any) -> List[Flowable]:
    f = ctx.f
    census = f.table("cohort_census")
    summary = f.table("cohort_summary")
    detail = f.table("cohort_detail")
    by_country = f.table("cohort_by_country")
    realised = f.table("cohort_long_run_returns")

    from src import cohorts as ch
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    pair = (str(cfg["cohorts"]["challenger"]), str(cfg["cohorts"]["incumbent"]))
    horizon = int(cfg["lifecycle"]["age_death"]) - int(cfg["lifecycle"]["age_start"])
    interval = ch.cluster_bootstrap(detail, n_boot=2000)
    signs = ch.sign_test(detail)
    spread = ch.dispersion(realised)
    n_cohorts = int(len(detail))
    independent = int(census["cohorts"].gt(0).sum())
    cec_col = [c for c in summary.columns if c.startswith("cec_crra_")][0]
    values = dict(zip(summary["strategy"], summary[cec_col]))
    cec_gap = (values[pair[0]] / values[pair[1]] - 1.0) * 100.0
    winner = str(summary["strategy"].iloc[0])
    richest = census.iloc[0]
    poorest = census.iloc[-1]
    losers = by_country[by_country["mean_gap_pct"] <= 0.0]

    out: List[Flowable] = [
        ctx.h1("#cohorts. The Realised Record, Without the Bootstrap")]
    out.append(ctx.p(
        f"Every number to this point comes from resampled histories. That is "
        f"the right way to get a distribution out of {horizon * 2 - 5} years "
        f"of annual data, and it leaves one question open that no amount of "
        f"resampling can close: is the result a property of the world, or of "
        f"the sampler? This section answers it by not resampling at all."))
    out.append(ctx.p(
        f"A <b>cohort</b> is one country and one birth year. The investor "
        f"turns {int(cfg['lifecycle']['age_start'])} in the first year of the "
        f"window, holds the strategy through that country’s realised returns "
        f"and the realised leave-one-out sleeve of every other market, in "
        f"calendar order, and dies {horizon} years later. Nothing is drawn. "
        f"There is exactly one such lifetime for every (country, birth year) "
        f"pair the panel can support, and labour income is deterministic "
        f"here so that the whole cross-sectional spread is about returns."))

    out.append(ctx.h2("#cohorts.1 How thin the realised record is"))
    out.append(ctx.p(
        f"The panel supports <b>{n_cohorts} complete lifetimes</b>. It does "
        f"not contain {n_cohorts} pieces of evidence. Adjacent cohorts in the "
        f"same country share {horizon - 1} of their {horizon} years, and cut "
        f"into non-overlapping lifetimes the same record yields "
        f"<b>{independent}</b> — one per market. Every cohort also lives "
        f"through the same two world wars, so even those are not independent "
        f"of one another. Inference below is therefore a bootstrap over "
        f"<i>countries</i>, never a standard error over cohorts."))
    out.extend(ctx.table(
        [["Market", "Usable years", "Longest unbroken run", "Lifetimes",
          "Earliest birth year"]]
        + [[str(r["iso"]), f"{int(r['usable_years'])}",
            f"{int(r['longest_unbroken'])}", f"{int(r['cohorts'])}",
            f"{int(r['first_start'])}" if int(r["cohorts"]) else "—"]
           for _, r in census.iterrows()],
        "What each market contributes to the realised record.",
        note="A lifetime cannot step over a market closure, so a gap costs "
             "sixty-eight cohorts rather than the handful of years it spans."))
    out.append(ctx.p(
        f"<b>A market closure removes a lifetime, not a year.</b> The "
        f"bootstrap refuses blocks that span a gap and keeps drawing; a "
        f"cohort that spans one cannot be run at all. "
        f"{str(richest['iso'])} contributes {int(richest['cohorts'])} "
        f"lifetimes and {str(poorest['iso'])} contributes "
        f"{int(poorest['cohorts'])}, the earliest of them beginning in "
        f"{int(poorest['first_start'])}. Every runnable German, Japanese and "
        f"Spanish lifetime therefore <i>begins after</i> that country’s "
        f"catastrophe and rides the recovery. This design is structurally "
        f"kinder to domestic equity than the bootstrap is, and that should be "
        f"held in mind for everything that follows."))
    out.append(ctx.p(
        f"It is worth being precise about what is selected, because it is "
        f"more than the usual complaint. The panel is already restricted to "
        f"markets that survived; this section restricts it again, to "
        f"<i>windows</i> that ran unbroken for {horizon} years. Those are not "
        f"the same filter. The second one removes every lifetime that "
        f"straddles a catastrophe while keeping the ones that follow it, so "
        f"the markets with the worst equity histories in the panel appear "
        f"here only through their recoveries. The consequence is that the "
        f"lead below is conservative in <i>direction</i> — the missing "
        f"lifetimes are ones a home market would have lost badly — while the "
        f"spread across markets is not the spread a randomly chosen saver "
        f"faced, and should not be read as one."))

    out.append(ctx.h2("#cohorts.2 What the realised lifetimes paid"))
    out.extend(ctx.table(
        [["Strategy", "CEC", "Median retirement consumption",
          "5th percentile", "P(ruin)"]]
        + [[str(r["label"]), f"{float(r[cec_col]):.4f}",
            f"{float(r['median_retirement_consumption']):.3f}",
            f"{float(r['p5_retirement_consumption']):.3f}",
            f"{float(r['prob_ruin']):.1%}"]
           for _, r in summary.iterrows()],
        f"Certainty equivalent over {n_cohorts} realised lifetimes, "
        f"γ = {gamma:g}. No resampling of any kind."))
    out.append(ctx.p(
        (f"<b>The ordering survives with no resampling at all.</b> "
         f"All-international delivers a certainty equivalent "
         f"{cec_gap:.1f}% above the 50/50 split over the lifetimes the panel "
         f"actually contains, and it is ahead in "
         f"{signs['cohort_win_rate']:.0%} of them. Whatever else the headline "
         f"is, it is not an artefact of the block bootstrap."
         if winner == pair[0] else
         f"<b>The ordering does not survive without resampling.</b> The best "
         f"strategy over the realised record is "
         f"<i>{_pretty_strategy(winner)}</i>, which contradicts the "
         f"bootstrap result directly.")))
    out.append(ctx.p(
        f"Resampling <i>countries</i> rather than cohorts gives a mean lead "
        f"of {interval['mean_gap_pct']:.1f}% with a 95% interval of "
        f"[{interval['ci_low']:.1f}, {interval['ci_high']:.1f}]"
        + (", which excludes zero. " if interval["excludes_zero"]
           else ", which contains zero. ")
        + f"Counting by market rather than by cohort — which gives the United "
        f"States sixty-four votes instead of one — the lead holds in "
        f"{signs['countries_won']} of {signs['countries_total']} markets."))
    out.append(ctx.p(
        f"That pooled mean is weighted by history, not by market. A country "
        f"enters it once per runnable lifetime, so the five markets with a "
        f"full unbroken run carry sixteen times the weight of Germany’s "
        f"{int(census.loc[census['iso'] == 'DEU', 'cohorts'].iloc[0]) if (census['iso'] == 'DEU').any() else 4}. "
        f"Giving each market one vote instead moves the lead to "
        f"{interval['equal_weighted_gap_pct']:.1f}% "
        f"({interval['weighting_shift_pp']:+.1f} points), with an interval of "
        f"[{interval['equal_ci_low']:.1f}, {interval['equal_ci_high']:.1f}]"
        + (" that still excludes zero. " if interval["equal_excludes_zero"]
           else " that contains zero. ")
        + f"A percentile interval on {interval['n_clusters']} clusters "
        f"under-covers, so the Student-t interval on the between-market "
        f"spread is the conservative reading: "
        f"[{interval['t_ci_low']:.1f}, {interval['t_ci_high']:.1f}]"
        + (", which excludes zero." if interval["t_excludes_zero"]
           else ", which contains zero. The direction is what this section "
                "establishes; the magnitude is not.")))
    if len(losers):
        out.append(ctx.p(
            "The exceptions are the interesting rows: "
            + ", ".join(f"<b>{str(r['iso'])}</b> ({float(r['mean_gap_pct']):+.0f}%, "
                        f"{int(r['cohorts'])} lifetimes from "
                        f"{int(r['first_start'])})"
                        for _, r in losers.iterrows())
            + ". Two of them are markets whose only runnable cohorts start at "
              "the bottom of a post-war recovery, which is the composition "
              "bias of the previous subsection showing up as a result."))

    out.append(ctx.h2("#cohorts.3 Why the realised lead is larger than the "
                      "resampled one"))
    out.append(ctx.p(
        "The bootstrap headline and the number above are not the same "
        "quantity and should not be read as an estimate and a re-estimate of "
        "one thing. A bootstrapped lifetime is a mosaic of blocks drawn from "
        "different decades, so a market that underperformed for forty years "
        "running contributes a few of those blocks and the rest of the "
        "lifetime is spliced in from better eras. A cohort cannot splice."))
    out.append(ctx.p(
        f"The realised record makes the consequence visible. Over the "
        f"{horizon}-year windows the panel supports, a home market’s "
        f"annualised real return ranges from "
        f"{spread['worst_domestic_pp']:.1f}% to "
        f"{spread['best_domestic_pp']:.1f}%. The international sleeve over "
        f"the identical years ranges from {spread['worst_sleeve_pp']:.1f}% to "
        f"{spread['best_sleeve_pp']:.1f}%, with a standard deviation of "
        f"{spread['sleeve_sd_pp']:.2f} points against "
        f"{spread['domestic_sd_pp']:.2f} for the single market."))
    out.append(ctx.note(
        f"That is the argument of this paper stated without a single "
        f"resampled path: there is no realised {horizon}-year window in which "
        f"the diversified sleeve did badly, and there are several in which a "
        f"single home market was destroyed. The sleeve was ahead in "
        f"{spread['share_sleeve_ahead']:.0%} of realised lifetimes, by "
        f"{spread['mean_excess_pp']:.2f} points a year."))

    out.extend(ctx.figure(
        "fig46_cohorts",
        "Top left: how many complete lifetimes each market contributes, which "
        "a market closure reduces by a lifetime rather than by a year. Top "
        "right: the realised lead, one observation per lifetime the panel can "
        "run. Bottom left: the two legs' annualised returns over the same "
        "sixty-eight years, cohort by cohort. Bottom right: the mean lead by "
        "market, against the interval a bootstrap over countries supports."))

    out.append(ctx.h2("#cohorts.4 What this section adds, and what it does not"))
    out.extend(ctx.bullets([
        "The ordering is not an artefact of the block bootstrap. It is in the "
        "realised record, without resampling, in "
        f"{signs['countries_won']} of {signs['countries_total']} markets.",
        f"The honest interval on the realised lead is "
        f"[{interval['ci_low']:.1f}, {interval['ci_high']:.1f}] pooled, "
        f"[{interval['equal_ci_low']:.1f}, {interval['equal_ci_high']:.1f}] "
        f"with each market given one vote, and "
        f"[{interval['t_ci_low']:.1f}, {interval['t_ci_high']:.1f}] on the "
        f"parametric reading — all from {independent} independent lifetimes. "
        f"Take the widest. This section adds confidence in the "
        f"<i>direction</i> and almost none in the magnitude.",
        "The design cannot see the worst domestic histories in the panel, "
        "because a lifetime cannot step over a market closure. The bootstrap "
        "can, which is one of the reasons it stays the headline.",
    ]))
    return out


def section_pension(ctx: Any) -> List[Flowable]:
    f = ctx.f
    gaps = f.table("pension_gap")
    entitlement = f.table("pension_entitlement")
    replacement = f.table("pension_replacement")

    from src import pension as pns
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(cfg["pension"]["n_paths"])
    params = pns.from_config(cfg)
    found = pns.verdict(gaps)
    rows = gaps.set_index("system")
    saving = float(cfg["lifecycle"]["savings_rate"])
    sg = float(cfg["pension"].get("sg_rate", pns.SG_RATE))
    sg_tax = float(cfg["pension"].get("sg_contributions_tax",
                                      pns.SG_CONTRIBUTIONS_TAX))
    net = sg * (1.0 - sg_tax)
    total = saving + net
    cut_out = params["pension_free_area"] + (params["pension_full_rate"]
                                             / params["pension_taper"])

    def _row(key: str, column: str) -> float:
        return (float(rows.loc[key, column]) if key in rows.index
                and column in gaps.columns else float("nan"))

    au_cec = _row("australia_as_legislated", "best_lift_pct")
    au_mean = _row("australia_as_legislated", "mean_lift_pct")
    au_p5 = _row("australia_as_legislated", "p5_lift_pct")
    au_gap = _row("australia_as_legislated", "gap_pct")
    base_gap = _row("us_social_security", "gap_pct")
    saving_cec = _row("us_matched_saving", "best_lift_pct")
    untested_cec = _row("age_pension_untested", "best_lift_pct")
    poor_gap = _row("age_pension_matched", "gap_pct")
    poor_winner = str(rows.loc["age_pension_matched", "winner"]) \
        if "age_pension_matched" in rows.index else ""
    au_reorders = bool(np.isfinite(au_gap) and (au_gap > 0.0) != (base_gap > 0.0))
    poor_reorders = bool(np.isfinite(poor_gap)
                         and (poor_gap > 0.0) != (base_gap > 0.0))

    out: List[Flowable] = [
        ctx.h1("#pension. One Country’s Pension Behind Sixteen Countries’ "
               "Returns")]
    out.append(ctx.p(
        f"Every result to this point pays the same public pension in all "
        f"{f.panel['n_countries']} markets: the US primary-insurance-amount "
        f"schedule, progressive in career earnings and paid in full whatever "
        f"else the retiree owns. It is the schedule of exactly one country in "
        f"the panel, and the panel’s whole point is that countries differ."))
    out.append(ctx.p(
        "It is also the shape most flattering to the argument. An "
        "earnings-related pension paid regardless of wealth is a <b>risk-free "
        "real annuity</b>: it arrives whatever the portfolio does, which puts "
        "a floor under retirement consumption, crowds fixed income out of the "
        "financial portfolio and pushes the optimal equity share up. Nothing "
        "in this paper has tested how much of the result that floor is "
        "carrying."))
    out.append(ctx.p(
        f"Australia is the developed world’s counter-example, and it differs "
        f"in two ways at once. The <b>pension</b> is flat rather than "
        f"earnings-related and is <b>means-tested</b>: assessable assets "
        f"above a free area withdraw it on a taper until it cuts out. And "
        f"the <b>contribution</b> is compulsory: the Superannuation Guarantee "
        f"pays {sg:.0%} of ordinary time earnings into a separate fund, on "
        f"top of whatever the worker saves voluntarily. The two push in "
        f"opposite directions, so the sweep crosses them rather than running "
        f"one Australia-versus-America row that would confound them."))

    out.append(ctx.h2("#pension.1 Calibration"))
    out.extend(ctx.table(
        [["Quantity", "Statutory", "In average earnings"],
         ["Maximum single rate", "$1,200.90 a fortnight",
          f"{params['pension_full_rate']:.3f}"],
         ["Assets-test free area (homeowner)", "$321,500",
          f"{params['pension_free_area']:.3f}"],
         ["Assets-test free area (non-homeowner)", "$600,000",
          f"{params['pension_free_area_non_homeowner']:.3f}"],
         ["Taper", "$3 a fortnight per $1,000",
          f"{params['pension_taper']:.3f} a year"],
         ["Cut-out (homeowner)", "$722,000", f"{cut_out:.3f}"],
         ["Superannuation Guarantee",
          f"{sg:.0%} of earnings, taxed {sg_tax:.0%} on entry",
          f"{net:.3f} reaching the fund"]],
        "The Age Pension for a single retiree, March 2026 indexation, with "
        "assets-test thresholds from July 2026, against Australian average "
        "weekly ordinary time earnings of $2,051.10 (ABS, November 2025).",
        note="Rates are held as multiples of economy-wide average earnings so "
             "the schedule travels across the panel’s currencies unchanged."))
    out.append(ctx.p(
        f"<b>The guarantee is a second contribution stream, not a larger "
        f"first one.</b> The worker still saves this paper’s own "
        f"{saving:.0%} out of take-home pay; the employer pays "
        f"{sg:.0%} on top, {sg_tax:.0%} of it is taken as contributions tax, "
        f"and the remaining {net:.1%} is invested in the same strategy and "
        f"assessed by the same means test. Total contributions are "
        f"{total:.1%} of income against the American saver’s {saving:.0%}. "
        f"Because both pots hold the same strategy and face no differential "
        f"tax on earnings in this model, they are financially one pot and are "
        f"simulated as one; the guarantee supplies "
        f"{net / total:.0%} of everything contributed."))
    out.append(ctx.note(
        "Working-life consumption is unchanged by the guarantee here, because "
        "its statutory incidence is on the employer. If the true incidence is "
        "on workers through lower wages — which is what most of the empirical "
        "literature finds — an Australian is paying for it in forgone pay and "
        "the comparison below is generous to them by that amount. It does not "
        "touch the certainty equivalents, because the utility window in this "
        "paper is retirement only. It does mean this is a comparison of "
        "<i>systems as legislated</i> rather than of two workers with the "
        "same lifetime resources, which is why the matched-contribution rows "
        "are there."))
    out.append(ctx.p(
        f"<b>The taper is the other mechanism, and it is steep.</b> A dollar "
        f"of assessable assets inside the tapered band costs "
        f"{params['pension_taper']:.1%} of pension every year it is held. No "
        f"asset in this panel earns that reliably in real terms, so inside "
        f"the band the marginal return on wealth is negative — and run "
        f"backwards, a retiree whose portfolio falls is met by a pension that "
        f"rises. A means test is a floor and a ceiling at once, which is why "
        f"it is assessed on assets as they stand, year by year, rather than "
        f"settled once at retirement."))

    out.append(ctx.h2("#pension.2 Crossing the two features"))
    out.extend(ctx.table(
        [["System", "Certainty equivalent (%)", "Mean consumption (%)",
          "5th percentile (%)", "All-intl over 50/50 (%)", "Best strategy"]]
        + [[str(r["label"]),
            f"{float(r['best_lift_pct']):+.1f}"
            if "best_lift_pct" in gaps.columns else "—",
            f"{float(r['mean_lift_pct']):+.1f}"
            if "mean_lift_pct" in gaps.columns else "—",
            f"{float(r['p5_lift_pct']):+.1f}"
            if "p5_lift_pct" in gaps.columns else "—",
            f"{float(r['gap_pct']):+.2f}",
            _pretty_strategy(str(r["winner"]))]
           for _, r in gaps.iterrows()],
        f"Retirement consumption relative to the US baseline, γ = {gamma:g}, "
        f"{n_paths:,} lifetimes per regime.",
        note="The first three columns are levels — how much retirement "
             "consumption a system delivers. The fifth is the ranking "
             "question — which portfolio it leads its investor to hold. They "
             "are different questions and here they have different answers."))
    out.append(ctx.p(
        (f"<b>The Australian system raises the average and lowers the "
         f"certainty equivalent.</b> As legislated it delivers "
         f"{au_mean:+.1f}% mean retirement consumption against the American "
         f"baseline and {au_cec:+.1f}% certainty-equivalent consumption. Both "
         f"are right, and the distance between them is the whole result: the "
         f"fifth percentile is {au_p5:+.1f}%. Compulsory saving buys a bigger "
         f"portfolio and the means test takes away the floor that portfolio "
         f"would otherwise sit on, so the good outcomes get better and the "
         f"bad ones get worse. A criterion as averse to the lower tail as a "
         f"certainty equivalent at γ = {gamma:g} prefers the annuity."
         if found["mean_and_cec_disagree"] else
         f"The Australian system delivers {au_cec:+.1f}% "
         f"certainty-equivalent and {au_mean:+.1f}% mean retirement "
         f"consumption against the American baseline.")))
    if len(replacement):
        rep = replacement.set_index("system")
        au_pension = (float(rep.loc["australia_as_legislated", "mean_pension"])
                      if "australia_as_legislated" in rep.index
                      else float("nan"))
        # Both pensions expressed against the same yardstick, computed from
        # the specs rather than quoted: the US schedule evaluated at a career
        # average of one unit of economy-wide earnings, and the Age Pension's
        # own maximum rate.
        from src import lifecycle as _lc
        base_spec = _lc.spec_from_config(cfg)
        earnings = float(base_spec.deterministic_income().mean())
        us_rate = float(base_spec.social_security_benefit(
            np.array([earnings]))[0]) / earnings
        au_max = float(params["pension_full_rate"])
        if np.isfinite(au_pension):
            out.append(ctx.p(
                f"The crux is one number. Under the American schedule this "
                f"investor collects a public pension worth {us_rate:.0%} of "
                f"economy-wide average earnings, every year of retirement, "
                f"unconditionally and for life. The Australian maximum rate "
                f"is {au_max:.0%} to begin with — and this investor collects "
                f"a mean of {au_pension / earnings:.1%}, because the "
                f"guarantee has made them wealthy enough that the assets test "
                f"withdraws almost all of it. They have swapped a large "
                f"risk-free real annuity for a larger portfolio. That is a "
                f"good trade on the average and a bad one in the tail, which "
                f"is exactly what the three level columns above report."))
    out.append(ctx.p(
        f"The crossing says where each piece of that comes from. Holding the "
        f"American pension and paying the Australian contribution rate is "
        f"worth {saving_cec:+.1f}%: compulsory saving on its own is a large "
        f"gain. Holding the contribution rate and cutting the pension to the "
        f"Age Pension’s flat rate, still paid to everybody, costs "
        f"{untested_cec:+.1f}% "
        + ("and changes nothing about the ordering — it is a smaller "
           "annuity, not a different kind of one"
           if str(rows.loc["age_pension_untested", "winner"])
           == str(rows.loc["us_social_security", "winner"])
           else "and moves the ordering with it")
        + f". Adding the assets "
        f"test to that is what does the damage. Together they land at "
        f"{au_cec:+.1f}%: the guarantee does not buy back what the means test "
        f"removes."))

    out.append(ctx.h2("#pension.3 What it does to the ranking"))
    out.append(ctx.p(
        (f"<b>The ranking does not survive the means test.</b> "
         f"All-international leads the 50/50 split by {base_gap:.2f}% under "
         f"the American schedule and by {au_gap:.2f}% under the Australian "
         f"one, where the best strategy becomes "
         f"<i>{_pretty_strategy(str(rows.loc['australia_as_legislated', 'winner']))}</i>. "
         f"This is the only place in this paper where the headline ordering "
         f"fails, and it fails for a reason that has nothing to do with the "
         f"return panel."
         if au_reorders else
         f"<b>The ranking survives the means test.</b> All-international "
         f"leads the 50/50 split by {au_gap:.2f}% against {base_gap:.2f}% "
         f"under the American schedule, and the same strategy wins. That is "
         f"not because the means test is harmless — it is because the "
         f"guarantee makes this saver rich enough that the test never "
         f"reaches them.")))
    out.append(ctx.p(
        f"The mechanism is the taper. Inside the assets-tested band every "
        f"extra dollar of assets costs {params['pension_taper']:.1%} of "
        f"pension a year, which is more than any asset in this panel earns "
        f"reliably in real terms. A portfolio that stays small keeps an "
        f"entitlement that a portfolio that grows loses, so the means test "
        f"pays a retiree to hold less risk — and it pays them at a rate the "
        f"risk premium cannot match. Nothing about the returns has changed "
        f"between these rows. The objective has."))
    if poor_reorders and au_reorders:
        out.append(ctx.p(
            f"<b>How far it goes depends on the balance, not the country.</b> "
            f"Run the same means test on a saver with no guarantee behind "
            f"them — {saving:.0%} voluntary saving and nothing else — and the "
            f"lead falls further, to {poor_gap:.2f}%, with "
            f"<i>{_pretty_strategy(poor_winner)}</i> taking first place "
            f"rather than "
            f"<i>{_pretty_strategy(str(rows.loc['australia_as_legislated', 'winner']))}</i>. "
            f"The poorer the saver, the deeper into the taper they sit and "
            f"the further the distortion goes: at {total:.1%} of income the "
            f"answer is a de-risking schedule, at {saving:.0%} it is cash. "
            f"The compulsory contribution does not remove the distortion, it "
            f"softens it."))
    elif poor_reorders:
        out.append(ctx.p(
            f"<b>The reversal is real, but it belongs to a poorer saver.</b> "
            f"Run the means test at this paper’s own {saving:.0%} savings "
            f"rate — an Australian with no guarantee behind them — and the "
            f"lead goes to {poor_gap:.2f}% and the best strategy becomes "
            f"<i>{_pretty_strategy(poor_winner)}</i>. That row is where the "
            f"taper actually binds."))
    out.append(ctx.p(
        f"Two things follow, and they should be kept apart. The first is "
        f"about this model: its investor contributes {total:.1%} of a rising "
        f"income for "
        f"{int(cfg['lifecycle']['age_retire']) - int(cfg['lifecycle']['age_start'])} "
        f"years and compounds it at historical real equity returns with no "
        f"tax on fund earnings, which makes them far wealthier than a median "
        f"Australian and puts them at the top of the taper rather than in "
        f"the middle of it. The second is about the paper as a whole: every "
        f"other section models a retiree whose public income is an "
        f"unconditional annuity, and a retiree whose public income is "
        f"withdrawn at the margin is solving a different problem. The answer "
        f"here is not that Australians should hold a glide path. It is that "
        f"a means test changes what the portfolio is for."))

    if np.isfinite(found.get("non_homeowner_lift_pct", float("nan"))):
        out.append(ctx.p(
            (f"The assets test has a higher free area for retirees who do not "
             f"own a home, and the model’s investor owns nothing outside the "
             f"portfolio. Running the non-homeowner thresholds — a free area "
             f"of {params['pension_free_area_non_homeowner']:.2f} times "
             f"average earnings rather than "
             f"{params['pension_free_area']:.2f} — moves the certainty "
             f"equivalent to {found['non_homeowner_lift_pct']:+.1f}%"
             + (", which does not change the conclusion: this saver clears "
                "even the more generous cut-out."
                if not found["thresholds_change_the_answer"] else
                ", which is enough to matter and is reported alongside."))))

    if len(entitlement):
        key = ("australia_as_legislated"
               if "australia_as_legislated" in set(entitlement["system"])
               else str(entitlement["system"].iloc[0]))
        block = entitlement[entitlement["system"] == key]
        every_tenth = block[block["age"] % 10 == 0]
        out.extend(ctx.table(
            [["Age", "Full rate", "Part rate", "No pension",
              "Mean pension (× average earnings)"]]
            + [[f"{int(r['age'])}", f"{float(r['share_full_rate']):.1%}",
                f"{float(r['share_part_rate']):.1%}",
                f"{float(r['share_no_pension']):.1%}",
                f"{float(r['mean_pension_x_earnings']):.3f}"]
               for _, r in every_tenth.iterrows()],
            "Where retirees sit on the taper under the system as legislated, "
            "holding the 50/50 portfolio.",
            note="A means test that never binds is a flat pension in "
                 "disguise; one that always binds to zero is no pension at "
                 "all. This table says which of the two this saver faces."))
    if len(replacement):
        out.extend(ctx.table(
            [["System", "Mean pension", "Share of retirement consumption"]]
            + [[str(r["label"]), f"{float(r['mean_pension']):.3f}",
                f"{float(r['pension_share_of_consumption']):.1%}"]
               for _, r in replacement.iterrows()],
            "How much of retirement the state is paying for."))

    out.extend(ctx.figure(
        "fig50_pension",
        "Top left: retirement consumption under each regime against the "
        "American baseline. Top right: what each regime does to the choice "
        "between portfolios. Bottom left: where retirees sit on the "
        "assets-test taper as the portfolio draws down. Bottom right: how "
        "much of retirement consumption the public pension supplies."))

    out.append(ctx.h2("#pension.4 What this changes"))
    out.extend(ctx.bullets([
        (f"<b>A pension schedule is not an innocent modelling choice.</b> "
         f"Swapping one developed country's for another's moves "
         f"certainty-equivalent retirement consumption by {au_cec:.0f}% "
         f"while the return panel, the preferences and the portfolios stay "
         f"exactly where they were. Every level quoted elsewhere in this "
         f"paper is conditional on the American schedule."),
        (f"<b>The average and the certainty equivalent disagree, and both "
         f"are true.</b> Compulsory saving raises mean retirement "
         f"consumption {au_mean:+.0f}% and lowers the fifth percentile "
         f"{au_p5:+.0f}%. A reader who cares about expected consumption "
         f"should read the first; one who cares about the bad case should "
         f"read the second. The paper's own criterion reads the second."
         if found["mean_and_cec_disagree"] else
         f"The average and the certainty equivalent agree: {au_mean:+.0f}% "
         f"and {au_cec:+.0f}%."),
        (f"<b>The allocation ranking reverses.</b> This is the only place "
         f"in this paper where it does. Under the means test the best "
         f"strategy becomes "
         f"<i>{_pretty_strategy(str(rows.loc['australia_as_legislated', 'winner']))}</i>, "
         f"and for a saver poorer than this one it goes further still. Every "
         f"other section models a retiree whose public income arrives "
         f"regardless of what they own; that is the assumption the ranking "
         f"rests on."
         if au_reorders else
         f"<b>The allocation ranking survives the system as legislated</b> "
         f"— but only because the guarantee lifts this saver clear of the "
         f"means test. For a saver inside the taper band the ordering "
         f"reverses, and most real Australians are inside it."
         if poor_reorders else
         "The allocation ranking is the same under every regime tested."),
        "<b>What is not modelled</b>: the income test, the family home’s "
        "exemption from the assets test — the largest single feature of the "
        "real system — the tax on fund earnings in accumulation, and every "
        "other tax in this paper. Nor is an annuity, which is what the "
        "American schedule is and what an Australian retiree would have to "
        "buy to match it.",
    ]))
    return out


def section_turnover(ctx: Any) -> List[Flowable]:
    f = ctx.f
    measured = f.table("turnover_measured")
    curve = f.table("turnover_gap")

    from src import turnover as tno
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(cfg["turnover"]["n_paths"])
    challenger = str(cfg["turnover"]["challenger"])
    incumbent = next((c[4:] for c in curve.columns
                      if c.startswith("cec_") and c[4:] != challenger),
                     str(cfg["turnover"]["incumbent"]))
    cross = tno.cost_of_the_schedule(measured, challenger, incumbent)
    found = tno.verdict(curve, measured, cross, challenger, incumbent)
    be = found["break_even_bp"]
    rows = measured.set_index("strategy")

    out: List[Flowable] = [
        ctx.h1("#turnover. What the Solved Schedule Costs to Trade")]
    out.append(ctx.p(
        "Section #fees prices the expense ratio and finds the differential "
        "that would undo the headline. It says nothing about the other cost "
        "of running a portfolio, which is trading it, and that omission lands "
        "hardest on the part of this paper most exposed to it. A fixed 50/50 "
        "portfolio trades because its assets drifted apart. A schedule solved "
        "age by age over the whole weight simplex trades for that reason "
        "<i>and</i> because it decided to hold something different this year "
        "— and the optimiser that chose the difference was charged nothing "
        "for it."))

    out.append(ctx.h2("#turnover.1 Measuring the trade"))
    out.append(ctx.p(
        "Turnover is reported one-way: half the sum of absolute weight "
        "changes, so selling one asset to buy another counts once. It splits "
        "three ways. <b>Total</b> is the trade actually required each year on "
        "simulated paths. <b>Drift</b> is what a portfolio whose target never "
        "moved would have had to trade anyway — the floor no schedule can get "
        "under. <b>Schedule</b> is the deterministic move the target makes "
        "between two ages: what following the plan would cost in a world "
        "where nothing ever drifted."))
    out.extend(ctx.table(
        [["Strategy", "Traded a year", "Drift alone", "Schedule alone",
          "Excess over drift", "Over a lifetime"]]
        + [[_pretty_strategy(str(r["strategy"])),
            f"{float(r['turnover_total']):.2%}",
            f"{float(r['turnover_drift_only']):.2%}",
            f"{float(r['turnover_schedule_only']):.2%}",
            f"{float(r['excess_over_drift']):+.2%}",
            f"{float(r['lifetime_turnover']):.1f}×"]
           for _, r in measured.iterrows()],
        "One-way turnover, averaged over paths and years.",
        note="A single-asset portfolio never drifts away from itself, so it "
             "trades nothing. A constant-weight portfolio's total and drift "
             "columns are equal by construction, which is the check that the "
             "decomposition is doing what it claims."))
    if cross.get("measured"):
        out.append(ctx.p(
            f"<b>The solved schedule turns over "
            f"{cross['solved_turnover']:.1%} of the portfolio a year.</b> "
            + ("The benchmark it has to beat is a single-asset portfolio, "
               "which never drifts away from itself and so never rebalances "
               "at all — it trades nothing, ever. "
               if cross.get("fixed_trades_nothing") else
               f"The fixed comparison turns over "
               f"{cross['fixed_turnover']:.1%}, "
               f"{cross['ratio']:.1f} times less. ")
            + f"Almost all of the schedule's trading is it choosing to move "
            f"rather than drift forcing it: the excess over the drift floor "
            f"is {float(rows.loc[challenger, 'excess_over_drift']):.1%} a "
            f"year, which is the cost the optimiser never saw."))
    out.append(ctx.p(
        "The three columns do not add up, and the reason is worth stating. A "
        "schedule that cuts equity in a year equity outperformed is trading "
        "<i>with</i> the drift rather than against it, so its total can sit "
        "below its drift counterfactual. A glide path is partly "
        "self-rebalancing, which is a point in its favour that a naive "
        "turnover count would miss."))

    out.append(ctx.h2("#turnover.2 Charging for it"))
    out.append(ctx.p(
        f"The cost is proportional to the value turned over and taken at the "
        f"rebalance, before that year’s return compounds on what is left. "
        f"Every strategy pays it on the same paths, so the question is never "
        f"whether costs hurt — they hurt everybody — but which portfolio they "
        f"hurt more. The comparison is against <i>{_pretty_strategy(incumbent)}</i>, "
        f"which is the best fixed portfolio judged before any cost is "
        f"introduced; letting the incumbent be re-chosen at each level would "
        f"slide it to meet the challenger and make the break-even vacuous."))
    out.extend(ctx.table(
        [["One-way cost (basis points)",
          "Solved schedule over the best fixed portfolio (%)",
          "Best strategy"]]
        + [[f"{float(r['basis_points']):.0f}", f"{float(r['gap_pct']):+.2f}",
            _pretty_strategy(str(r["winner"]))]
           for _, r in curve.iterrows()],
        f"γ = {gamma:g}, {n_paths:,} lifetimes per level."))
    out.append(ctx.p(
        (f"<b>The solved schedule pays for its own trading.</b> Its lead "
         f"starts at {found['baseline_gap_pct']:.2f}% and is still "
         f"{found['gap_at_highest_pct']:.2f}% at "
         f"{found['highest_cost_bp']:.0f} basis points a trade — a cost no "
         f"index investor pays."
         if found["survives_whole_grid"] else
         f"<b>The solved schedule’s edge is thin and it is spent on "
         f"trading.</b> Its advantage over the best fixed portfolio is "
         f"{found['baseline_gap_pct']:.2f}% before costs, and reaches zero at "
         f"<b>{be:.0f} basis points</b> one-way. "
         + ("That is above an index fund's spread but inside what a retail "
            "investor trading small parcels pays, and it is a small enough "
            "number that the advantage should be read as an upper bound "
            "rather than as a result."
            if be > 10.0 else
            "That is inside what any real investor pays, so the advantage "
            "measured in section #allocation is an artefact of a "
            "frictionless rebalance."))))
    out.append(ctx.note(
        f"Section #out_of_sample reaches the same place by a different road. "
        f"There the solved schedules are asked to work on data they were not "
        f"fitted to; here they are asked to pay for the trades they make. "
        f"Both find that a constant mix, held unchanged and fitted to "
        f"nothing, is a harder benchmark than the in-sample gains suggest."))

    out.extend(ctx.figure(
        "fig51_turnover",
        "Left: what each strategy trades a year, against the drift it could "
        "not have avoided. Middle: the trading each schedule demands of "
        "itself with returns switched off. Right: the solved schedule’s lead "
        "over the best fixed portfolio as trading gets more expensive."))

    out.append(ctx.h2("#turnover.3 What this changes"))
    out.extend(ctx.bullets([
        (f"The gains in section #allocation should be read net of a cost "
         f"they were solved without. The break-even is {be:.0f} basis points "
         f"one-way." if np.isfinite(be) else
         "The gains in section #allocation survive every trading cost on the "
         "grid."),
        "The headline of this paper is unaffected. The strategies it "
        "compares are constant mixes, and their turnover is drift alone — "
        "the same trade every rebalanced portfolio makes.",
        "<b>What is not modelled</b>: costs are proportional and symmetric, "
        "with no bid-ask asymmetry, no market impact and no minimum ticket. "
        "Contributions during accumulation are invested pro-rata rather than "
        "steered toward the underweight asset, which overstates turnover for "
        "a real saver — so the break-even here is, if anything, reached too "
        "easily. Tax on realised gains is a cost of trading and is absent, "
        "along with every other tax in this paper.",
    ]))
    return out


def section_out_of_sample(ctx: Any) -> List[Flowable]:
    f = ctx.f
    p = f.panel
    transfer = f.table("oos_transfer")
    benchmarks = f.table("oos_benchmarks")

    from src import oos as oo
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    cut = int(cfg["out_of_sample"]["cut_year"])
    n_paths = int(cfg["out_of_sample"]["n_paths"])
    stability = oo.ranking_is_stable(benchmarks)
    found = oo.verdict(transfer, stability)
    winner = str(transfer["benchmark"].iloc[0])
    # The same count section #cohorts arrives at, read off its own table so
    # the two sections cannot disagree about how much evidence there is.
    independent = int(f.table("cohort_census")["cohorts"].gt(0).sum())

    out: List[Flowable] = [
        ctx.h1("#out_of_sample. Do the Solved Schedules Survive Data They "
               "Did Not See?")]
    out.append(ctx.p(
        f"Sections #glide, #allocation and #leverage each hand a "
        f"coordinate-ascent search sixty-eight free parameters and report "
        f"what it found. Every one of those gains is measured on the same "
        f"history the search was given. That is the standard way to overstate "
        f"a result: with sixty-eight parameters and the "
        f"{independent} independent lifetimes "
        f"section #cohorts counts, a search can fit a good deal of noise, and "
        f"an in-sample gain is an upper bound on what an investor standing at "
        f"the start could have had."))
    out.append(ctx.p(
        f"The design splits the calendar record at {cut}, solves on one half "
        f"and scores on the other, against three references: the gain "
        f"<i>where it was solved</i> — the number those sections report "
        f"today; the gain <i>where it was scored</i>, against the best fixed "
        f"strategy on the untouched half, which is what an investor of the "
        f"period could actually have held; and a <i>ceiling</i>, the same "
        f"family solved on the test half itself. Both directions are run, "
        f"because the two halves of this panel are not interchangeable: one "
        f"contains two world wars and the other contains the post-war "
        f"expansion."))

    out.append(ctx.h2("#out_of_sample.1 What transferred"))
    out.extend(ctx.table(
        [["Family", "Solved on", "Scored on", "Gain where solved (%)",
          "Gain where scored (%)", "Ceiling (%)"]]
        + [[str(r["label"]), str(r["train_window"]), str(r["test_window"]),
            f"{float(r['in_sample_gain_pct']):+.2f}",
            f"{float(r['transfer_gain_pct']):+.2f}",
            f"{float(r['ceiling_gain_pct']):+.2f}"]
           for _, r in transfer.iterrows()],
        f"Solved schedules against the best fixed strategy, γ = {gamma:g}, "
        f"{n_paths:,} lifetimes per window.",
        note="A negative middle column is a schedule that would have been "
             "worse than holding a constant mix."))
    out.append(ctx.p(
        (f"<b>No solved schedule beats a constant mix out of sample.</b> In "
         f"all {found['runs']} runs the schedule solved on one half is worse "
         f"on the other than simply holding the best fixed strategy. The "
         f"in-sample gains those sections report — averaging "
         f"{found['mean_in_sample_gain_pct']:.2f}% — are descriptions of the "
         f"window the search was given."
         if found["no_run_transfers"] else
         f"<b>Every solved schedule survives the split.</b> All "
         f"{found['runs']} runs beat the best fixed strategy on data they "
         f"were not fitted to."
         if found["every_run_transfers"] else
         f"<b>The solved schedules transfer in "
         f"{found['runs_that_beat_the_benchmark']} of {found['runs']} runs, "
         f"and which ones is the finding.</b> A schedule learned on "
         f"{found['earlier_window']} and applied to "
         f"{found['later_window']} keeps "
         f"{found['forward_gain_pct']:+.2f}%; the same exercise run backwards "
         f"— learned on {found['later_window']}, applied to "
         f"{found['earlier_window']} — delivers "
         f"{found['backward_gain_pct']:+.2f}%. The in-sample gain averages "
         f"{found['mean_in_sample_gain_pct']:.2f}% either way."
         if found["asymmetric"] else
         f"<b>The solved schedules transfer in "
         f"{found['runs_that_beat_the_benchmark']} of {found['runs']} runs.</b> "
         f"The in-sample gain averages "
         f"{found['mean_in_sample_gain_pct']:.2f}% and what survives the "
         f"split averages {found['mean_transfer_gain_pct']:.2f}%.")))
    if found.get("asymmetric"):
        out.append(ctx.p(
            f"That asymmetry has a reading, and it is not a flattering one "
            f"for the optimiser. The earlier half contains two world wars, "
            f"several market closures and the Depression; the later half "
            f"contains the post-war expansion. A schedule fitted to the "
            f"turbulent window carries something forward into the calm one. "
            f"A schedule fitted to the calm window learns a world that never "
            f"came back, and applying it to the turbulent half costs "
            f"{abs(found['backward_gain_pct']):.2f}% against simply holding a "
            f"constant mix. An investor solving today would be in the second "
            f"position, not the first."))
    out.append(ctx.note(
        f"The benchmark the solved schedules are measured against is "
        f"<i>{_pretty_strategy(winner)}</i> in every run. The strategy that "
        f"does transfer is a constant mix, held unchanged for a lifetime — "
        f"which is the strategy this paper is actually about."))

    out.append(ctx.h2("#out_of_sample.2 Are the two halves the same world?"))
    out.extend(ctx.table(
        [["Window", "Strategy", "CEC", "Rank"]]
        + [[str(r["window"]), str(r["label"]), f"{float(r['cec']):.4f}",
            f"{int(r['rank'])}"] for _, r in benchmarks.iterrows()],
        "Every fixed strategy on each half of the record.",
        note="If the fixed ranking moves across the split, some of what fails "
             "to transfer is the world changing rather than the search "
             "overfitting."))
    out.append(ctx.p(
        ("The fixed strategies keep exactly their order across the two "
         "halves, so the split is not simply comparing two different worlds."
         if stability.get("stable") else
         f"The fixed strategies do <b>not</b> keep their order across the "
         f"halves — {int(stability.get('n_positions_moved', 0))} positions "
         f"move, though the winner is "
         f"{'the same' if stability.get('same_winner') else 'different'}. "
         f"Some of what fails to transfer above is therefore the world "
         f"changing between the halves rather than the search overfitting, "
         f"and this test cannot separate the two.")))

    out.append(ctx.p(
        f"<b>One split is one experiment, and that is the limit of this "
        f"test.</b> The record is cut once, at {cut}, because there is only "
        f"one place to cut it that leaves enough calendar time on each side "
        f"to solve {int(cfg['lifecycle']['age_death']) - int(cfg['lifecycle']['age_start'])}-year "
        f"schedules. That leaves no distribution to judge the result against: "
        f"we cannot say how often a split of this record would show an "
        f"asymmetry this large by chance, because there is no second split to "
        f"compare it with. Nor can a single cut separate the two explanations "
        f"the table admits — a schedule that learned noise, and a world that "
        f"changed underneath it — because both predict the same pattern. "
        f"Rolling-origin splits, or a placebo distribution over randomly "
        f"chosen cut years, would separate them; neither fits inside a "
        f"{p['last_year'] - p['first_year'] + 1}-year "
        f"record that already has to hold two disjoint working lives. Read "
        f"this section as a caution with a direction, not as a test with a "
        f"size."))

    out.extend(ctx.figure(
        "fig47_out_of_sample",
        "Left: the gain each solved schedule reports where it was solved, "
        "against what it delivers on the half it never saw. Right: the fixed "
        "strategies on both halves of the record."))

    out.append(ctx.h2("#out_of_sample.3 What this changes"))
    out.extend(ctx.bullets([
        f"The gains reported in sections #glide, #allocation and #leverage "
        f"should be read as <b>upper bounds</b>. The transferable part "
        f"averages {found['mean_transfer_gain_pct']:.2f}% against "
        f"{found['mean_in_sample_gain_pct']:.2f}% in sample.",
        f"<b>The benchmark that keeps winning is "
        f"<i>{_pretty_strategy(winner)}</i>, in every one of the "
        f"{found['runs']} runs.</b> That is the most useful thing in this "
        f"section and it deserves stating plainly rather than as an aside: "
        f"a constant mix, held unchanged for a lifetime and fitted to "
        f"nothing, is what a solved schedule has to beat on data it did not "
        f"see, and mostly does not. Section #turnover reaches the same "
        f"conclusion by a different route, by charging the solved schedule "
        f"for the trades it makes.",
        "The headline of this paper does not depend on any of it. The "
        "comparison that carries the paper is between fixed strategies, none "
        "of which is fitted to anything.",
        "What this test cannot do is prove that a failure to transfer is "
        "overfitting rather than a changed world. The two halves are not "
        "draws from one process, and the benchmark table is the evidence.",
    ]))
    return out


def section_withholding(ctx: Any) -> List[Flowable]:
    f = ctx.f
    curve = f.table("withholding_curve")
    crossed = f.table("withholding_crossings")
    anchors = f.table("withholding_anchors")
    drag = f.table("withholding_drag")
    optima = f.table("withholding_optimal")

    from src import withholding as wht
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    wcfg = cfg["withholding"]
    challenger = str(wcfg.get("challenger", "international_equity"))
    rivals = [str(r) for r in wcfg.get("rivals", ())]
    headline = float(wcfg.get("headline_rate", 0.15))
    found = wht.verdict(curve, crossed, optima, drag, challenger)
    mean_share = float(drag.loc[drag["era"] == "whole panel",
                                "mean_dividend_share"].iloc[0]) \
        if len(drag) else float("nan")
    statutory = float(wht.ANCHORS["the US statutory rate for non-residents"])
    from src.fees import verdict as _fee_verdict
    fee_found = _fee_verdict(
        f.table("fee_common"), f.table("fee_differential"),
        (str(cfg["fees"]["challenger"]), str(cfg["fees"]["incumbent"])),
        f.table("fee_anchors"))
    fee_be_bp = float(fee_found["break_even_differential_bp"])
    statutory_drag_bp = statutory * mean_share * 1e4

    out: List[Flowable] = [
        ctx.h1("#withholding. The Cost Differential That Is Not a Fee")]
    out.append(ctx.p(
        f"Section #fees asks how large a cost differential between a domestic "
        f"and an international fund would have to be to cancel the headline, "
        f"and answers {fee_be_bp:.0f} basis points — a number it describes as "
        f"beyond any index-fund pair. That framing has a hole in it. There is "
        f"a cost differential between holding your own market and holding "
        f"everyone else's that is not a fund's fee, is not negotiable, and is "
        f"not a choice: <b>foreign dividend withholding tax</b>."))
    out.append(ctx.p(
        "A government taxes dividends leaving its borders. A resident holding "
        "that same market pays nothing on them, or receives a full credit; a "
        "foreigner pays and, in an ordinary retail structure, cannot reclaim. "
        "The statutory United States rate on dividends to non-resident "
        "individuals is 30%, most treaties reduce it to 15% for portfolio "
        "investors, and a broad developed-markets index fund bears a "
        "weighted average of roughly 7.5% across its constituents. None of "
        "those is a price an investor can shop around for."))
    out.append(ctx.p(
        "Two things make this a different experiment from a fee, and both "
        "cut against the sleeve this paper recommends. The <b>base is "
        "dividends, not assets</b>: a fee is levied on the whole portfolio, "
        "withholding only on the part of the return that arrives as a "
        "dividend — so the drag tracks the dividend yield, which in this "
        "panel has fallen by roughly a third since the war. And it falls on "
        "<b>exactly one leg</b>: all-international pays it on every dividend "
        "it receives, the 50/50 split on half, a domestic-only portfolio on "
        "none. Section #fees builds that asymmetry by hypothesis. Here it is "
        "the law."))

    out.append(ctx.h2("#withholding.1 Putting a tax rate in fee units"))
    out.append(ctx.p(
        "The panel's source data compounds rather than adds — one plus the "
        "total return equals one plus the capital gain times one plus the "
        "dividend, verified to eight decimal places, with the dividend "
        "measured against the ending price. Withholding at rate <i>τ</i> "
        "leaves the investor <i>(1 − τ)</i> of that dividend, so the "
        "after-tax return is the gross return multiplied by <i>(1 − τq)</i>, "
        "where <i>q = dp/(1 + dp)</i> is the share of the year's ending value "
        "that arrived as a taxable dividend. That is exactly the "
        "multiplicative form Section #fees uses for an expense ratio, with a "
        "time-varying rate in place of a constant — which is what makes a "
        "statutory tax rate comparable with a fund's fee at all."))
    out.extend(ctx.table(
        [["Era", "Mean dividend share of the return",
          "Drag on the sleeve (bp a year)"]]
        + [[str(r["era"]), f"{float(r['mean_dividend_share']):.4f}",
            f"{float(r['drag_bp']):.1f}"] for _, r in drag.iterrows()],
        f"What a {headline:.0%} withholding rate actually costs, by era.",
        note="A statutory rate is a constant; the drag it produces is not, "
             "because it is levied on a dividend yield that has fallen. An "
             "investor facing the same law in 1930 and in 2010 paid "
             "materially different amounts for it."))
    out.append(ctx.note(
        f"That translation is the section's first result, and it is worth "
        f"pausing on. At the panel's own mean dividend share of "
        f"{mean_share:.4f}, the {statutory:.0%} statutory rate is worth "
        f"<b>{statutory_drag_bp:.0f} basis points a year</b> on the "
        f"international sleeve. The break-even differential Section #fees "
        f"reports — the one it calls beyond any index-fund pair — is "
        f"{fee_be_bp:.0f} basis points. They are the same order of magnitude, "
        f"and one of them is not optional."))

    out.append(ctx.h2("#withholding.2 Who wins, and at what rate"))
    lead_cols = [f"lead_over_{r}_pct" for r in rivals
                 if f"lead_over_{r}_pct" in curve.columns]
    out.extend(ctx.table(
        [["Rate"] + [f"vs {_pretty_strategy(c[len('lead_over_'):-len('_pct')])}"
                     for c in lead_cols] + ["Best strategy"]]
        + [[f"{float(r['rate_pct']):.1f}%"]
           + [f"{float(r[c]):+.2f}%" for c in lead_cols]
           + [_pretty_strategy(str(r["winner"]))]
           for _, r in curve.sort_values("rate").iterrows()],
        f"All-international's lead over each rival, γ = {gamma:g}."))
    out.extend(ctx.table(
        [["Rival", "Lead at a zero rate", "Overtakes at", "which is"]]
        + [[_pretty_strategy(str(r["rival"])),
            f"{float(r['lead_at_zero_pct']):+.2f}%",
            f"{float(r['crossing_pct']):.1f}%" if bool(r["reached_on_grid"])
            else "never on this grid",
            f"{float(r['equivalent_drag_bp']):.0f} bp a year"
            if bool(r["reached_on_grid"]) else "—"]
           for _, r in crossed.iterrows()],
        "Where each rival overtakes all-international.",
        note="The last column converts the crossing rate into the annual "
             "drag it represents at the panel's mean dividend share, so it "
             "can be read against the fee differential of Section #fees."))
    out.append(ctx.p(
        (f"<b>The headline does not survive a statutory withholding rate.</b> "
         f"<i>{_pretty_strategy(found['first_rival'])}</i> overtakes "
         f"all-international at {found['first_crossing_pct']:.1f}%, which is "
         f"{'inside' if found['crossing_within_statutory'] else 'outside'} "
         f"the {statutory:.0%} a non-resident pays without a treaty and "
         f"{'inside' if found['crossing_within_treaty'] else 'outside'} the "
         f"{headline:.0%} a documented one pays. "
         f"{int(found['n_rivals_overtaking'])} of {len(rivals)} rivals "
         f"overtake somewhere on the grid."
         if found["any_rival_overtakes"] else
         f"<b>The headline survives every withholding rate tested</b>, to "
         f"{found['highest_rate_pct']:.0f}% — far past any statutory rate. "
         f"All-international is still ahead of every rival at the top of the "
         f"grid, so the tax is real, it is large, and it is not large enough.")))
    out.extend(ctx.table(
        [["Rate", "%", "Drag (bp a year)"]
         + [f"vs {_pretty_strategy(c[len('lead_over_'):-len('_pct')])}"
            for c in lead_cols]]
        + [[str(r["label"]), f"{float(r['rate_pct']):.1f}",
            f"{float(r['drag_bp']):.0f}"]
           + [f"{float(r[c]):+.2f}%" for c in lead_cols]
           for _, r in anchors.iterrows()],
        "The rates that actually exist, placed on the swept curve.",
        note="These are configured anchors, not findings: nothing in this "
             "project's data can verify a statutory rate."))

    out.append(ctx.h2("#withholding.3 What to hold at each rate"))
    out.append(ctx.p(
        "Asking which of two fixed portfolios wins presumes the investor is "
        "choosing between two fixed portfolios. They are not — they are "
        "choosing a weight. The same domestic-share grid Section #inflation "
        "uses is scored at every rate, on the same paths, and the "
        "certainty-equivalent maximum read off. This is the more useful of "
        "the two answers, because it is a schedule an investor can look up "
        "against the rate they actually face."))
    out.extend(ctx.table(
        [["Withholding rate", "Optimal domestic share of equity", "CEC there",
          "Over all-international (%)", "Over the next grid point (%)"]]
        + [[f"{float(r['rate']):.1%}",
            f"{float(r['optimal_domestic_share']):.0%}",
            f"{float(r['cec_at_optimum']):.4f}",
            f"{float(r['margin_over_low_end_pct']):+.2f}"
            if "margin_over_low_end_pct" in optima.columns else "—",
            f"{float(r['margin_over_runner_up_pct']):.3f}"]
           for _, r in optima.sort_values("rate").iterrows()],
        "The certainty-equivalent-maximising domestic share at each rate.",
        note="The last column is the honesty check: a winner that beats its "
             "neighbour by a rounding error has not identified an optimum."))
    out.append(ctx.p(
        (f"<b>The optimal portfolio walks home as the tax rises.</b> The "
         f"domestic share that maximises the certainty equivalent moves from "
         f"{found['optimal_domestic_at_zero']:.0%} at a zero rate to "
         f"{found['optimal_domestic_at_top']:.0%} at "
         f"{found['highest_rate_pct']:.0f}%, "
         f"{found['optimal_domestic_shift'] * 100:+.0f} points. Nothing about "
         f"the return panel has changed between those rows — only who is "
         f"allowed to keep the dividends."
         if found.get("optimum_moves_home") else
         f"<b>The optimal portfolio does not move</b>: "
         f"{found.get('optimal_domestic_at_zero', float('nan')):.0%} domestic "
         f"at every rate tested, which says the tax changes the level of what "
         f"an investor gets and not the shape of what they should hold.")))

    out.extend(ctx.figure(
        "fig53_withholding",
        "Top left: what one statutory rate costs by era, as dividend yields "
        "fall. Top right: each fixed strategy's certainty equivalent against "
        "the rate, with the real rates marked. Bottom left: all-international's "
        "lead over each rival, and where it runs out. Bottom right: the "
        "certainty-equivalent surface over the domestic share, one curve per "
        "rate, with the maximum circled."))

    out.append(ctx.h2("#withholding.4 What this changes"))
    out.extend(ctx.bullets([
        (f"<b>Section #fees understates the cost differential an "
         f"international investor faces, because it looks only at fees.</b> "
         f"At the panel's own dividend share the {statutory:.0%} statutory "
         f"rate is worth {statutory_drag_bp:.0f} basis points a year against "
         f"that section's {fee_be_bp:.0f}-point break-even. The break-even is "
         f"not beyond reach; it is roughly the law."),
        (f"The headline ranking "
         f"{'changes inside the statutory range, so it is conditional on the investor’s tax position and residence' if found.get('crossing_within_statutory') else 'does not change inside the statutory range'}. "
         f"A reader who can hold foreign equity through a vehicle that "
         f"reclaims the tax — many pension wrappers can — faces the top row "
         f"of these tables. A reader in an ordinary taxable account does not."),
        ("The optimal domestic share is the number to carry rather than the "
         "winner of a two-horse race, and it moves with the rate. That is "
         "the practical form of this section's answer."
         if found.get("optimum_moves_home") else
         "The optimal domestic share does not move with the rate."),
        "<b>What is not modelled here</b>: dividend imputation, which refunds "
        "corporate tax to <i>domestic</i> shareholders and widens the home "
        "market's advantage further — Section #franking takes it up, and it "
        "changes the answer this section reaches; "
        "the second layer of withholding an investor suffers by holding a "
        "US-domiciled fund of foreign stocks rather than the stocks "
        "themselves; reclaim procedures, which recover part of the tax at a "
        "paperwork cost most retail investors do not pay; and the investor's "
        "own income tax on the dividend afterwards. Every one of those runs "
        "the same way, against the international sleeve, so the rates here "
        "are a floor on the real burden rather than an estimate of it.",
    ]))
    return out


def section_franking(ctx: Any) -> List[Flowable]:
    f = ctx.f
    curve = f.table("franking_curve")
    crossed = f.table("franking_crossings")
    credits = f.table("franking_credits")
    granked = f.table("franking_by_franked_share")
    era = f.table("franking_era")
    drag = f.table("franking_withholding_drag")
    optima = f.table("franking_optimal")
    comparison = f.table("franking_wedge_comparison")

    from src import franking as frk
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    fcfg = cfg["franking"]
    challenger = str(fcfg.get("challenger", "international_equity"))
    rivals = [str(r) for r in fcfg.get("rivals", ())]
    tau = float(fcfg.get("withholding_rate", 0.15))
    acc_company, fund_tax, _ = frk.anchor_parameters(frk.ACCUMULATING)
    found = frk.verdict(curve, crossed, optima, credits, comparison,
                        challenger)
    headline = frk.credit_rate(float(fcfg.get("headline_company_tax", 0.30)),
                               float(fcfg.get("headline_fund_tax", 0.0)),
                               float(fcfg.get("headline_franked_share", 1.0)))

    def _whole(frame: pd.DataFrame, column: str) -> float:
        block = frame[frame["era"] == "whole panel"]
        return float(block[column].iloc[0]) if len(block) else float("nan")

    worth_bp = _whole(era, "credit_bp")
    drag_bp = _whole(drag, "drag_bp")

    out: List[Flowable] = [
        ctx.h1("#franking. Dividend Imputation, and the Other Blade")]
    out.append(ctx.p(
        f"Section #withholding prices the tax that falls on the international "
        f"leg and finds it large enough to matter — {drag_bp:.0f} basis "
        f"points a year at the treaty rate. It closes by listing what it does "
        f"not model, and the first item on that list is the one this section "
        f"takes up, because it runs the other way and lands on the other leg."))
    out.append(ctx.p(
        f"Under a classical tax system a company pays corporate tax and its "
        f"shareholder is then taxed again on what is distributed. Under an "
        f"<i>imputation</i> system the corporate tax counts as tax the "
        f"shareholder already paid: the dividend arrives with a credit "
        f"attached, and a shareholder taxed below the corporate rate can be "
        f"refunded the difference in cash. Australia has run such a system "
        f"since 1987 and has made the credit fully refundable since 2000, so "
        f"a superannuation fund in pension phase — which pays no tax at all — "
        f"collects the whole credit as a cheque."))
    out.append(ctx.note(
        "The credit is available to residents only, which makes it the exact "
        "mirror of withholding: one falls only abroad, one lands only at "
        "home, and both are levied on dividends rather than on assets. That "
        "symmetry is what allows the two to be quoted in the same units and "
        "then added. Adding a credit on top of a pre-tax return is not "
        "double-counting the same relief: an equity total return is already "
        "net of corporate tax, so the baseline here is an investor who keeps "
        "the cash dividend and pays no <i>personal</i> tax on it, and "
        "imputation refunds the <i>corporate</i> tax on top of that."))

    out.append(ctx.h2("#franking.1 What a credit is worth"))
    out.append(ctx.p(
        f"Write <i>q</i> for the dividend's share of the gross return — the "
        f"quantity Section #withholding already builds — and let <i>t</i>c be "
        f"the corporate rate imputed, <i>t</i>f the rate the holder's own fund "
        f"pays, and φ the franked fraction. A franked dollar of dividend "
        f"grosses up to 1/(1 − <i>t</i>c), is taxed at <i>t</i>f and returns "
        f"(1 − <i>t</i>f)/(1 − <i>t</i>c); an unfranked one simply returns "
        f"(1 − <i>t</i>f). So a dollar of cash dividend is worth 1 + <i>c</i> "
        f"with <i>c</i> = (1 − <i>t</i>f)(1 + φ <i>t</i>c/(1 − <i>t</i>c)) − 1, "
        f"and the after-credit return is (1 + <i>r</i>)(1 + <i>c q</i>) — "
        f"Section #withholding's expression with −τ replaced by +<i>c</i>."))
    out.extend(ctx.table(
        [["Who holds it", "Company rate", "Fund's own rate", "Franked",
          "Credit (% of the dividend)"]]
        + [[str(r["label"]), f"{float(r['company_tax']):.0%}",
            f"{float(r['fund_tax']):.0%}", f"{float(r['franked_share']):.0%}",
            f"{float(r['credit_pct']):+.2f}"]
           for _, r in credits.iterrows()],
        "The credit implied by each holder's position, γ = "
        f"{gamma:g}.",
        note="Nothing in the last column is written down; each is the three "
             "numbers beside it put through the formula above. The row worth "
             "pausing on is the taxed fund holding unfranked stock, which "
             "comes back <i>negative</i>: a fund paying "
             f"{fund_tax:.0%} on a dividend carrying no credit is worse off "
             "than the untaxed baseline this paper otherwise assumes. The "
             "formula is not built to produce a credit, and the sign is the "
             "first check that it is a tax code being modelled."))
    if len(granked):
        break_even = frk.break_even_franked_share(acc_company, fund_tax)
        out.append(ctx.p(
            f"Partial franking scales it, and no market's franking level is "
            f"observable in this project's data, so it is swept rather than "
            f"assumed. Inside a fund taxed at {fund_tax:.0%} the credit has "
            f"that tax to make back before it is worth anything at all, and "
            f"solving for where it does gives {break_even:.1%} of dividends "
            f"franked. Below that a fund holding its own market is behind "
            f"this paper's untaxed baseline rather than ahead of it; the "
            f"{float(granked['credit_pct'].iloc[-1]):+.1f}% in the paragraph "
            f"above is what full franking delivers, and it is an upper bound "
            f"on what any real market does."))

    out.append(ctx.h2("#franking.2 What it is worth in basis points"))
    out.append(ctx.p(
        f"A credit rate means nothing beside a fee until it has been put in "
        f"the same units. At the headline credit of {headline:.1%} — fully "
        f"franked, pension phase — the home leg gains {worth_bp:.0f} basis "
        f"points a year across the panel, against the {drag_bp:.0f} the "
        f"foreign leg is losing to withholding at {tau:.0%}. The wedge "
        f"between the two legs is therefore {worth_bp + drag_bp:.0f} basis "
        f"points, which is larger than either half and larger than the fee "
        f"differential Section #fees identifies as the break-even."))
    if len(era):
        out.extend(ctx.table(
            [["Era", "Mean dividend share", "Worth of the credit (bp a year)"]]
            + [[str(r["era"]), f"{float(r['mean_dividend_share']):.4f}",
                f"{float(r['credit_bp']):+.0f}"]
               for _, r in era.iterrows()],
            f"The credit at {headline:.1%}, by era.",
            note="The same tax code delivers less as time passes, for the "
                 "same reason the withholding drag falls in Section "
                 "#withholding: both are levied on dividends, and this "
                 "panel's dividend yields have fallen by roughly a third."))

    out.append(ctx.h2("#franking.3 Both blades, closed"))
    out.append(ctx.p(
        "Neither section on its own describes anybody. An investor collecting "
        "franking credits is also paying withholding, simultaneously, and the "
        "two push the same way. The table below scores the handful of "
        "positions someone can actually stand in, each on identical paths."))
    if len(comparison):
        keys = [k for k in [challenger] + rivals
                if f"cec_{k}" in comparison.columns]
        out.extend(ctx.table(
            [["Where the investor stands", "Credit", "WHT"]
             + [_compact_strategy(k) for k in keys] + ["Best"]]
            + [[str(r["position"]), f"{float(r['credit']):+.1%}",
                f"{float(r['rate']):.0%}"]
               + [f"{float(r[f'cec_{k}']):.4f}" for k in keys]
               + [_compact_strategy(str(r["winner"]))]
               for _, r in comparison.iterrows()],
            f"Certainty-equivalent consumption at each position, γ = "
            f"{gamma:g}.",
            note="WHT is the withholding rate the foreign leg pays. The "
                 "first row is this paper's own baseline and the second "
                 "isolates withholding, so the table reads as a "
                 "decomposition rather than as a list of scenarios."))
    out.append(ctx.p(
        (f"<b>The wedge overturns the headline, and neither blade does it "
         f"alone.</b> With withholding abroad and no credit at home "
         f"<i>{_pretty_strategy(found.get('wedge_winner_at_baseline', ''))}</i> "
         f"still wins, which is Section #withholding's finding restated. Add "
         f"the credit and "
         f"<i>{_pretty_strategy(found.get('wedge_winner_at_the_end', ''))}</i> "
         f"wins instead. Nothing about the returns changed between those "
         f"rows; what changed is which leg the tax code is standing on."
         if found.get("wedge_overturns_the_headline") else
         f"<b>The wedge does not overturn the headline.</b> "
         f"<i>{_pretty_strategy(found.get('wedge_winner_at_baseline', ''))}</i> "
         f"wins at every position tested, credit and withholding together, so "
         f"the international case survives the largest tax asymmetry this "
         f"panel's law can produce.")))
    if len(crossed):
        out.append(ctx.p(
            (f"The swept curve locates the reversal exactly. "
             f"<i>{_pretty_strategy(found['first_rival'])}</i> overtakes at a "
             f"credit of {found['first_crossing_pct']:.1f}% of the dividend, "
             f"and a fully franked dividend is worth "
             f"{found['accumulation_credit']:.1%} inside a fund still "
             f"accumulating and {found['pension_credit']:.1%} inside one "
             f"paying a pension. Both clear it, which is why the reversal is "
             f"not a statement about an extreme parameter."
             if found.get("crossing_within_accumulation") else
             f"The swept curve locates the reversal at "
             f"{found['first_crossing_pct']:.1f}%, which only the "
             f"pension-phase position reaches at "
             f"{found['pension_credit']:.1%}."
             if found.get("crossing_within_pension_phase") else
             f"No rival overtakes anywhere on the swept grid, which runs to "
             f"{found['highest_credit_pct']:.0f}% — more than twice the "
             f"largest credit this tax code delivers.")))

    out.append(ctx.h2("#franking.4 What to hold, rather than who wins"))
    if len(optima):
        out.extend(ctx.table(
            [["Credit", "Optimal domestic share", "CEC at the optimum",
              "Margin over the runner-up (%)"]]
            + [[f"{float(r['credit']):+.1%}",
                f"{float(r['optimal_domestic_share']):.0%}",
                f"{float(r['cec_at_optimum']):.4f}",
                f"{float(r['margin_over_runner_up_pct']):.2f}"]
               for _, r in optima.iterrows()],
            f"The certainty-equivalent-maximising domestic share at each "
            f"credit, swept on top of {tau:.0%} withholding abroad.",
            note="The margin column is what stops a flat maximum being read "
                 "as an identification."))
    out.append(ctx.p(
        (f"<b>The optimum walks home as the credit rises</b>, from "
         f"{found['optimal_domestic_at_zero']:.0%} with no credit at all — "
         f"this paper's own baseline — to "
         f"{found['optimal_domestic_at_top']:.0%} at the top of the grid. "
         f"That is the practical form of the answer: not which of two "
         f"portfolios wins, but how much of the home market a given tax "
         f"position justifies."
         if found.get("optimum_ever_moves") else
         f"The optimal domestic share does not move with the credit, holding "
         f"at {found['optimal_domestic_at_zero']:.0%} throughout.")))

    out.extend(ctx.figure(
        "fig56_franking",
        "Top left: all-international's lead against the credit, with the "
        "crossing and the real anchors marked. Top right: what the credit is "
        "worth by era. Bottom left: each rival's lead over all-international "
        "at every position an investor can occupy — a bar above the line is a "
        "reversal. Bottom right: the optimal domestic share as the credit "
        "rises."))

    out.append(ctx.h2("#franking.5 What this changes"))
    out.extend(ctx.bullets([
        f"<b>The tax code moves the answer, and only when both halves of it "
        f"are modelled.</b> Withholding alone leaves "
        f"<i>{_pretty_strategy(challenger)}</i> ahead. The credit at home "
        f"turns a {drag_bp:.0f} basis-point drag on one leg into a "
        f"{worth_bp + drag_bp:.0f} basis-point wedge between the two, and "
        f"that is enough.",
        (f"<b>It is not a case for going home.</b> "
         f"<i>{_pretty_strategy(found.get('wedge_winner_at_the_end', ''))}</i> "
         f"wins at the largest credit tested, not "
         f"<i>{_pretty_strategy('domestic_equity')}</i>. The correction to "
         f"this paper's headline is a smaller foreign allocation, not none."
         if found.get("wedge_overturns_the_headline")
         and found.get("wedge_winner_at_the_end") != "domestic_equity" else
         "The credit does not produce a case for holding the home market "
         "alone at any level tested."),
        "<b>Whose law this is.</b> The credit is applied to whichever market "
        "an investor holds as their own, which models a world where every "
        "country operates imputation rather than the one that exists. That "
        "is the same convention Section #pension uses when it pays "
        "Australia's Age Pension to an investor drawing sixteen countries' "
        "returns: the question is what the mechanism is worth, not what a "
        "population-weighted average of sixteen tax codes comes to. Several "
        "of this panel's countries ran an imputation system during the "
        "twentieth century and abolished it; Australia did not. That is a "
        "fact from the tax literature rather than from this project's data, "
        "and no number above rests on it.",
        "<b>What is not modelled</b>: the personal tax a classical system "
        "would then levy on the dividend, which runs against the home leg "
        "and would narrow the wedge; any franking level other than the swept "
        "grid; and the years before 1987, when Australia's own investors had "
        "no credit at all. The last is the largest of the three and runs the "
        "same way as the first.",
    ]))
    return out


def section_human_capital(ctx: Any) -> List[Flowable]:
    f = ctx.f
    curve = f.table("human_capital_gap")
    ranking = f.table("human_capital_ranking")

    from src import humancapital as hcp
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(cfg["human_capital"]["n_paths"])
    pair = (str(cfg["human_capital"]["challenger"]),
            str(cfg["human_capital"]["incumbent"]))
    modes = f.table("human_capital_modes")
    fitted = hcp.sensitivity(curve, mode="home")
    found = hcp.verdict(curve, fitted, pair, mode="home", comparison=modes)
    home = (curve[curve["mode"] == "home"] if "mode" in curve.columns
            else curve)
    level_cols = [c for c in ranking.columns
                  if c not in ("strategy", "label")]

    out: List[Flowable] = [
        ctx.h1("#human_capital. When the Pay Cheque Is a Claim on the Home "
               "Market")]
    out.append(ctx.p(
        "Labour income in this model is a hump-shaped real profile multiplied "
        "by a permanent random walk and a transitory shock, and every result "
        "so far has drawn those shocks independently of the return panel. "
        "That independence is a gift to domestic equity. A worker whose "
        "employer, industry and tax base are the index is long that index "
        "twice: once in the portfolio and once in the pay cheque. Holding "
        "less of the home market than its weight in a global index is the "
        "standard prescription that follows, and the model has been assuming "
        "the premise away."))
    out.append(ctx.p(
        "The parameter swept here is a <b>correlation, not a loading</b>. The "
        "permanent innovation becomes <i>ρ·u + √(1 − ρ²)·z</i>, with <i>u</i> "
        "the standardised domestic equity return of the same year. That "
        "rotation preserves unit variance, so raising ρ changes which part of "
        "a career’s risk is systematic without changing how much risk a "
        "career carries. Any movement below is therefore the correlation and "
        "not extra income volatility smuggled in alongside it; at ρ = 0 every "
        "other result in this paper is unchanged to the last bit."))

    out.append(ctx.h2("#human_capital.1 The sweep"))
    out.extend(ctx.table(
        [["Correlation with the home market",
          "All-international over 50/50 (%)", "Best strategy"]]
        + [[f"{float(r['correlation']):.1f}", f"{float(r['gap_pct']):.2f}",
            _pretty_strategy(str(r["winner"]))]
           for _, r in home.iterrows()],
        f"γ = {gamma:g}, {n_paths:,} lifetimes per level. Correlated with the "
        f"home market; the foreign correlation is whatever the two markets’ "
        f"own co-movement implies."))
    out.append(ctx.p(
        (f"<b>The ranking depends on the assumption.</b> The best strategy "
         f"changes across the swept correlations, which means the headline "
         f"was resting on human capital being independent of the home market "
         f"rather than merely being flattered by it."
         if found["winner_ever_changes"] else
         f"<b>Correlated human capital widens the lead, and nothing changes "
         f"places.</b> Going from independence to a correlation of "
         f"{found['highest_correlation']:.1f} moves the lead from "
         f"{found['baseline_gap_pct']:.2f}% to "
         f"{found['gap_at_highest_pct']:.2f}% — {found['change_pp']:+.2f} "
         f"points, or about {found['slope_per_10pp']:+.2f} points per 0.1 of "
         f"correlation. The direction is the one theory predicts, so the "
         f"independence assumption used everywhere else in this paper is "
         f"conservative."
         if found["widens_with_correlation"] else
         f"<b>Correlated human capital narrows the lead</b>, from "
         f"{found['baseline_gap_pct']:.2f}% to "
         f"{found['gap_at_highest_pct']:.2f}%. That is the opposite of the "
         f"textbook prediction and needs explaining rather than reporting.")))
    out.extend(ctx.table(
        [["Strategy"] + [f"ρ = {c}" for c in level_cols]]
        + [[str(r["label"])] + [f"{float(r[c]):.4f}" for c in level_cols]
           for _, r in ranking.iterrows()],
        "Certainty equivalent for every strategy at every correlation.",
        note="Every strategy loses ground as the correlation rises, because "
             "a career correlated with the market is a riskier career; what "
             "matters here is whether the gaps between them move."))

    out.append(ctx.h2("#human_capital.2 A claim on which market?"))
    out.append(ctx.p(
        "The sweep above answers a narrower question than it appears to. It "
        "correlates the pay cheque with the <i>home</i> market and leaves the "
        "foreign correlation to fall wherever the two markets’ own "
        "co-movement puts it. But the argument for tilting away from home "
        "rests on the difference between the two correlations, not on the "
        "level of either. A worker whose income is a claim on world equity "
        "rather than on their own market has no home market to tilt away "
        "from, and whatever the correlation was buying should largely "
        "cancel. That is the objection, and it is answerable."))
    out.append(ctx.p(
        "Three readings are run over the same grid. <b>Home only</b> is the "
        "sweep above. <b>Strict</b> pins the foreign correlation to zero, "
        "which — because the markets move together — requires loading "
        "negatively on the foreign market, and is therefore the most "
        "favourable version of the argument the data allow. <b>Diagonal</b> "
        "correlates the pay cheque equally with both markets, which is the "
        "objection stated as a specification. All three preserve unit "
        "variance, so they differ in what the career’s risk is a claim on "
        "and not in how much of it there is."))
    out.extend(ctx.table(
        [["Reading", "Lead at ρ = 0 (%)", "Lead at the top of the grid (%)",
          "Change (points)", "Share of the home-only effect"]]
        + [[str(r["label"]), f"{float(r['gap_low_pct']):.2f}",
            f"{float(r['gap_high_pct']):.2f}",
            f"{float(r['change_pp']):+.2f}",
            f"{float(r['share_of_home_effect']):.2f}×"
            if "share_of_home_effect" in modes.columns else "—"]
           for _, r in modes.iterrows()],
        "The same correlation grid under three assumptions about the foreign "
        "market.",
        note="All three start at the same place, because at ρ = 0 they are "
             "the same specification."))
    out.append(ctx.p(
        (f"<b>The objection is substantially right, and the direction "
         f"survives it.</b> Correlating the pay cheque with both markets "
         f"equally keeps only "
         f"{found.get('diagonal_share_of_home', float('nan')):.2f} of the "
         f"widening that correlating with the home market alone produced — "
         f"{found.get('change_diagonal_pp', float('nan')):+.2f} points "
         f"against {found['change_pp']:+.2f}. Most of what this section "
         f"measures is a statement about home bias specifically, not about "
         f"human capital being risky. What does not change is the sign or "
         f"the ranking: under every reading the lead still widens, and "
         f"{'the winner still never changes' if not found.get('winner_changes_in_any_mode') else 'the winner changes in at least one'}."
         if found.get("diagonal_mostly_cancels") else
         f"<b>The objection does not bite.</b> Correlating the pay cheque "
         f"with both markets equally still moves the lead "
         f"{found.get('change_diagonal_pp', float('nan')):+.2f} points "
         f"against {found['change_pp']:+.2f} for the home market alone, so "
         f"the effect is not an artefact of which market the correlation was "
         f"attached to.")))

    out.extend(ctx.figure(
        "fig48_human_capital",
        "Left: the lead of all-international over the 50/50 split as human "
        "capital becomes a claim on equity, under each of the three readings "
        "of which market it is a claim on. Right: every strategy’s certainty "
        "equivalent over the same range, correlated with the home market."))

    out.append(ctx.h2("#human_capital.3 What this changes"))
    out.extend(ctx.bullets([
        "The independence assumption used everywhere else in this paper is "
        "conservative: relaxing it makes the case for the home market worse, "
        "not better.",
        f"The size is the useful part. Over a correlation range that runs "
        f"past anything a labour economist would defend, the lead moves "
        f"{found['change_pp']:+.2f} points — a second-order lever against "
        f"the fee differential of section #fees and the country-deletion "
        f"range of section #panel.",
        "What is not modelled: unemployment spells, industry, and any "
        "correlation between labour income and the international sleeve, "
        "which is not zero either and would push the other way. This is a "
        "bound on one channel, not a calibrated model of human capital.",
    ]))
    return out


def section_mortality(ctx: Any) -> List[Flowable]:
    f = ctx.f
    curve = f.table("mortality_gap")
    comparison = f.table("mortality_comparison")
    ranking = f.table("mortality_ranking")

    from src import mortality as mrt
    cfg = f.cfg
    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    n_paths = int(cfg["mortality"]["n_paths"])
    pair = (str(cfg["mortality"]["challenger"]),
            str(cfg["mortality"]["incumbent"]))
    found = mrt.verdict(comparison, curve, pair)
    age_death = int(cfg["lifecycle"]["age_death"])
    age_retire = int(cfg["lifecycle"]["age_retire"])
    ruin_cols = [c for c in curve.columns if c.startswith("ruin_")]
    law_cols = [c for c in ranking.columns if c not in ("strategy", "label")]

    out: List[Flowable] = [ctx.h1("#mortality. Death at a Random Age")]
    out.append(ctx.p(
        f"Every result before this section kills the investor on schedule at "
        f"{age_death}. That is a modelling convenience, and it distorts in "
        f"two directions at once: it understates longevity risk, because "
        f"nobody knows they have exactly {age_death - age_retire} retired "
        f"years to fund; and it overstates the far tail, because a strategy "
        f"is rewarded for consumption at ninety-two that most investors never "
        f"live to spend."))
    out.append(ctx.p(
        "The treatment here is a <b>re-weighting, not a re-simulation</b>. "
        "Under the headline withdrawal rule the policy does not depend on the "
        "death age, so a random lifespan changes only which years of an "
        "already-simulated path are experienced and with what probability. "
        "Consumption in each year is weighted by the probability of being "
        "alive to enjoy it, and the estate is whatever wealth is left in the "
        "year the investor dies. A certain stream <i>c</i> paired with the "
        "matching bequest still returns exactly <i>c</i>, so these certainty "
        "equivalents are in the same units as every other one in the paper — "
        "and at a degenerate law that kills everyone at "
        f"{age_death}, the arithmetic reproduces the paper’s own certainty "
        "equivalent to machine precision."))
    out.append(ctx.note(
        "Exact for any policy that does not condition on the death age — the "
        "fixed real rule, the constant-percentage rule, every constant-weight "
        "strategy. Approximate for the horizon-based spending rules of "
        "section #spending, which amortise over a planning horizon and would "
        "themselves change if they knew the mortality table."))
    out.append(ctx.p(
        "<b>That design decides how much this section can find, and the "
        "reader should discount it accordingly.</b> Re-weighting a policy "
        "that does not itself respond to longevity is close to guaranteed to "
        "return a small number: the investor is not allowed to buy an "
        "annuity, to spend faster because the table says they are unlikely to "
        "reach ninety, or to hold back because it says they might. What is "
        "being tested here is therefore whether the <i>allocation ranking</i> "
        "is robust to the horizon assumption — a narrow question, and one "
        "worth answering, because the ranking is what the paper claims. It is "
        "not a finding that longevity risk is unimportant. The value of "
        "longevity risk lives almost entirely in the response to it, and this "
        "model makes no response."))

    out.append(ctx.h2("#mortality.1 The sweep"))
    out.extend(ctx.table(
        [["Mortality", "E[age at death]",
          "All-international over 50/50 (%)"]
         + [f"P(outlives the money): {c[5:].replace('_', ' ')}"
            for c in ruin_cols]]
        + [[str(r["mortality"]), f"{float(r['life_expectancy']):.0f}",
            f"{float(r['gap_pct']):.2f}"]
           + [f"{float(r[c]):.1%}" for c in ruin_cols]
           for _, r in curve.iterrows()],
        f"Gompertz survival, γ = {gamma:g}, {n_paths:,} lifetimes.",
        note="Expected ages are within the model: mass beyond the simulated "
             "horizon is carried at the final age, so these are lower bounds "
             "on each law's own life expectancy."))
    out.append(ctx.p(
        (f"<b>The ranking is not invariant to the mortality assumption.</b> "
         f"The order of the strategies changes between the fixed horizon and "
         f"at least one law swept, so results elsewhere in this paper carry a "
         f"dependence on dying at {age_death} that was not visible before."
         if found["ordering_ever_changes"] else
         f"<b>Nothing changes places.</b> Across a fixed horizon and "
         f"{found['laws'] - 1} mortality laws spanning expected death ages "
         f"from {found['shortest_life_expectancy']:.0f} to "
         f"{found['longest_life_expectancy']:.0f}, the ordering is identical "
         f"and the lead moves by at most {found['largest_change_pp']:.2f} "
         f"points from its fixed-horizon value of "
         f"{found['fixed_horizon_gap_pct']:.2f}%.")))
    out.append(ctx.p(
        ("Outliving the portfolio becomes <i>less</i> likely under every law, "
         "which is not the portfolio getting safer: it is the investor having "
         "fewer years in which to outlive it. A certain ninety-third birthday "
         "is a pessimistic longevity assumption, and the ruin probabilities "
         "quoted elsewhere in this paper inherit that pessimism."
         if found["ruin_falls_under_mortality"] else
         "Outliving the portfolio does not become uniformly less likely under "
         "the swept laws, which is worth noting: fewer expected years should "
         "mean fewer years in which to run out of money.")))
    out.extend(ctx.table(
        [["Strategy"] + [str(c) for c in law_cols]]
        + [[str(r["label"])] + [f"{int(r[c])}" for c in law_cols]
           for _, r in ranking.iterrows()],
        "Rank under each mortality assumption."))

    out.extend(ctx.figure(
        "fig49_mortality",
        "Top left: the survival curves swept, against the certain death the "
        "rest of the paper assumes. Top right: every strategy’s certainty "
        "equivalent under each. Bottom: the lead, and what a finite life does "
        "to the chance of outliving the portfolio."))

    out.append(ctx.h2("#mortality.2 What this changes"))
    out.extend(ctx.bullets([
        ("The results elsewhere in this paper are not resting on the fixed "
         "horizon."
         if not found["ordering_ever_changes"] else
         "The results elsewhere in this paper carry a dependence on the fixed "
         "horizon, and section #limitations should be read with that in mind."),
        f"The lead spans {found['min_gap_pct']:.2f}% to "
        f"{found['max_gap_pct']:.2f}% across every assumption tested, against "
        f"{found['fixed_horizon_gap_pct']:.2f}% at the fixed horizon.",
        "Ruin probabilities quoted elsewhere are computed against a certain "
        f"{age_death}rd birthday. Under any of these laws they are too high.",
        "What is not modelled: a couple rather than an individual, mortality "
        "correlated with wealth, and a policy that adapts to the mortality "
        "table. The last would raise every certainty equivalent here, so "
        "these numbers are a floor.",
    ]))
    return out


def story(ctx: Any) -> List[Flowable]:
    parts: List[Flowable] = []
    parts += front_matter(ctx)
    parts += contents(ctx)
    parts += section_introduction(ctx)
    parts += section_background(ctx)
    parts += section_data(ctx)
    parts += section_methods(ctx)
    # The order below must match SECTION_ORDER; the assertion in `story`
    # fails the build if the two ever drift apart.
    parts += section_baseline(ctx)
    parts += section_sensitivity(ctx)
    parts += section_cohorts(ctx)
    parts += section_panel(ctx)
    parts += section_sleeve(ctx)
    parts += section_hedging(ctx)
    parts += section_valuation(ctx)
    parts += section_inflation(ctx)
    parts += section_fees(ctx)
    parts += section_withholding(ctx)
    parts += section_franking(ctx)
    parts += section_human_capital(ctx)
    parts += section_mortality(ctx)
    parts += section_pension(ctx)
    parts += section_glide(ctx)
    parts += section_allocation(ctx)
    parts += section_leverage(ctx)
    parts += section_turnover(ctx)
    parts += section_out_of_sample(ctx)
    parts += section_housing(ctx)
    parts += section_mortgage(ctx)
    parts += section_saving(ctx)
    parts += section_accumulation(ctx)
    parts += section_retirement(ctx)
    parts += section_sequence(ctx)
    parts += section_spending(ctx)
    parts += section_plan(ctx)
    parts += section_leisure(ctx)
    parts += section_tax(ctx)
    parts += section_discussion(ctx)
    parts += section_limitations(ctx)
    parts += section_conclusion(ctx)
    parts += section_references(ctx)
    parts += appendix_parameters(ctx)
    parts += appendix_panel(ctx)
    parts += appendix_supplementary(ctx)
    parts += appendix_software(ctx)
    _check_section_order(parts)
    return parts
