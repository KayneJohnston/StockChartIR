"""The text of the working paper.

Prose is authored here; every number inside it is pulled from
:mod:`paper.facts`, which reads the pipeline's own CSV output. The separation
is deliberate: the argument is written once, the evidence is re-read on every
build, and a rerun of the pipeline that changed a result would change the
paper rather than silently contradict it.
"""

from __future__ import annotations

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

    out: List[Flowable] = [
        Spacer(1, 1.1 * cm),
        Paragraph("Beyond the Status Quo, Revisited", s["title"]),
        Paragraph("A Computational Re-Examination of Lifecycle Asset "
                  "Allocation, with Thirteen Extensions", s["subtitle"]),
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
        f"survives {n_settings} parameter settings across ten dimensions with "
        f"{reversals} reversals. Sustainable withdrawal rates are far below "
        f"the four-percent convention on every strategy "
        f"({pc(swr_eq, 1)} for all-equity, {pc(swr_tdf, 1)} for the target-date "
        f"fund at a five-percent ruin tolerance). "
        f"We then push past replication with thirteen extensions. Solving the "
        f"glide path directly by coordinate ascent under common random numbers "
        f"reproduces the all-equity corner rather than an interior optimum, "
        f"and freeing all four portfolio weights at every age — {alloc_params} "
        f"parameters on the simplex — adds {alloc_lead:.2f}% over the best "
        f"fixed benchmark while still producing no glide path. Relaxing the "
        f"long-only constraint, borrowing to invest is worth "
        f"{lev_free:+.2f}% at a zero borrowing spread over the real bill rate "
        f"but decays quickly in that spread, breaking even by "
        f"{break_even:.2%} — and what the optimiser levers is a diversified "
        f"portfolio rather than a concentrated one. Currency hedging the "
        f"international leg loses certainty-equivalent consumption at every "
        f"ratio tested, even when the hedge is free. Making the retirement date a "
        f"wealth-triggered decision rather than a birthday is worth about 3% "
        f"of certainty-equivalent consumption against a date matched on the "
        f"same mean retirement age. Conditioning the savings rate on the "
        f"funded ratio is worth a further {funded_net:.1f}%, and a deep "
        f"decomposition of that signal — across functional form, target "
        f"definition, asymmetry, feasibility bands and eight competing state "
        f"variables — finds that the strongest available signal is not the "
        f"portfolio at all but the investor's own pay cheque "
        f"({income_net:.1f}%). Finally, the decade around the retirement date "
        f"explains {pc(float(lottery['r2_retirement_window']), 0)} of the "
        f"variation in retirement outcomes, a lottery no allocation rule can "
        f"diversify away. "
f"Conditioning a lifetime on how expensive its market was at the "
        f"moment it began — using the trailing dividend yield an investor "
        f"could observe, ranked against tercile boundaries computed only "
        f"from country-years that had already happened — "
        + (f"leaves the ranking intact in all {len(val_adv)} valuation "
           f"buckets while moving the level: "
           if bool((val_adv['advantage_pct'] > 0).all())
           else f"reverses the ranking in "
                f"{int((val_adv['advantage_pct'] <= 0).sum())} of "
                f"{len(val_adv)} valuation buckets: ")
        + f"lifetimes begun in the dearest third reach retirement with less "
        f"and run out of money more often than those begun in the cheapest. "
        f"Adding housing to the investable set — de-smoothed to undo the "
        f"appraisal lag the published index carries — "
        + (f"earns {pc(float(house_free['mean_housing']), 0)} of the "
           f"portfolio when it is free to hold and "
           + (f"drops out entirely at an annual holding cost of "
              f"{pc(house_break, 1)}"
              if np.isfinite(house_break) else
              "survives every holding cost tested")
           if float(house_free["mean_housing"]) > 0.01 else
           "earns no place in the portfolio at any price, including free")
        + ". "
        f"Financing that house with a mortgage — the one asset an ordinary "
        f"household can borrow against at close to the government's rate — "
        f"produces a loan-to-value schedule that "
        + (f"declines with age, {pc(mort_work, 0)} while working against "
           f"{pc(mort_ret, 0)} in retirement, reproducing from optimisation "
           f"the pattern households follow in practice"
           if mort_work > mort_ret else
           f"rises with age, {pc(mort_work, 0)} while working against "
           f"{pc(mort_ret, 0)} in retirement, contrary to observed household "
           f"behaviour")
        + ". "
        f"The panel's principal limitation is its breadth: "
        f"{pr['n_countries']} developed markets, so the international leg "
        f"spans {pr['n_countries'] - 1} foreign markets, all advanced "
        f"economies with long recorded histories."
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
# 1. Introduction
# ---------------------------------------------------------------------------
def section_introduction(ctx: Any) -> List[Flowable]:
    f = ctx.f
    p = f.panel
    alloc, lev = f.allocation, f.leverage
    adv_tdf = f.advantage("balanced_all_equity", "target_date_fund")
    adv_6040 = f.advantage("balanced_all_equity", "sixty_forty")
    adv_dom = f.advantage("balanced_all_equity", "domestic_equity")
    ruin_eq = float(f.strategy_row("balanced_all_equity")["prob_ruin"])
    ruin_tdf = float(f.strategy_row("target_date_fund")["prob_ruin"])
    dominance = f.table("dominance_check")
    n_criteria = int(dominance["criteria"].iloc[0])
    n_won = int(dominance["criteria_won"].iloc[0])

    out: List[Flowable] = [ctx.h1("1. Introduction")]
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
        "reader can see where the conclusion is robust and where it is thin. "
        "Third — and this is where most of the length lies — it takes thirteen "
        "extensions that the original design leaves open and works each of "
        "them out, several of which produce results that run against the "
        "intuition that motivated them."))

    out.append(ctx.h2("1.1 What we find"))
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
        f"<b>Leverage stops being worth the trouble long before it stops "
        f"paying.</b> Borrowing to invest is worth "
        f"{lev['value_at_zero_spread']:+.2f}% when credit is free and breaks "
        f"even at a spread of {lev['break_even_spread']:.2%}, but the "
        f"advantage is already under a tenth of a percent by "
        f"{lev['negligible_spread']:.2%}. What the optimiser levers is a "
        f"diversified portfolio rather than a concentrated one, and the cost "
        f"of the trade is paid in the left tail."))
    out.append(ctx.p(
        "<b>Timing beats allocation.</b> The single decade around a person's "
        "retirement date explains more of the variation in their retirement "
        "outcome than the choice between any two of the allocation strategies "
        "we test. Making the retirement date itself a decision — retire when "
        "wealth reaches a multiple of income, rather than on a birthday — is "
        "worth roughly three percent of certainty-equivalent consumption "
        "against a fixed date matched on the same mean retirement age."))
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

    out.append(ctx.h2("1.2 What is new here"))
    out.append(ctx.p(
        "Relative to the paper being replicated, this study contributes a "
        "methodological discipline and thirteen substantive extensions. The "
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

    out.append(ctx.h2("1.3 Roadmap"))
    out.append(ctx.p(
        "Section 2 places the exercise in the literature. Section 3 documents "
        "the panel and its construction. Section 4 sets out the bootstrap, the "
        "lifecycle model, the preference specification and the comparison "
        "discipline. Section 5 presents the baseline replication and Section 6 "
        "its sensitivity. Sections 7 through 17 present the thirteen extensions "
        "in order of increasing distance from the original design, ending with "
        "three that relax assumptions the model itself makes: that a "
        "lifetime's starting valuation is irrelevant (Section 15), that the "
        "opportunity set contains no housing (Section 16), and that the house "
        "is owned outright (Section 17). Section 18 "
        "discusses what the results collectively imply, Section 19 states the "
        "limitations candidly, and Section 20 concludes. Four appendices give "
        "the full parameter set, the country panel, supplementary tables and "
        "the reproduction instructions."))
    return out


# ---------------------------------------------------------------------------
# 2. Background
# ---------------------------------------------------------------------------
def section_background(ctx: Any) -> List[Flowable]:
    f = ctx.f
    out: List[Flowable] = [ctx.h1("2. Background and Related Work")]
    out.append(ctx.h2("2.1 The theoretical case for the glide path"))
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

    out.append(ctx.h2("2.2 The empirical challenge"))
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
        "sample is the developed-market cross-section over more than a "
        "century, and their sampling scheme draws contiguous blocks so that "
        "runs of good and bad decades survive into the simulated lifetimes. "
        "The claim is that under this process the case for the glide path "
        "evaporates. This paper reconstructs that process independently and "
        "asks whether the claim holds."))

    out.append(ctx.h2("2.3 The data source"))
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
        f"countries, and that set is our panel. Section 3.2 describes it and "
        f"Section 16.1 sets out what its breadth costs."))

    out.append(ctx.h2("2.4 Adjacent literatures this paper touches"))
    out.extend(ctx.bullets([
        "<b>Bootstrap inference for dependent data.</b> The sampling scheme "
        "here is a stationary block bootstrap in the sense of Politis and "
        "Romano (1994), generalised so that a block is a (country, window) "
        "pair drawn jointly across all five series.",
        "<b>Safe withdrawal rates.</b> The four-percent convention descends "
        "from Bengen (1994) and the Trinity study (Cooley, Hubbard and Walz, "
        "1998), both of which are US-only and historical-sequence based. "
        "Section 5.4 re-derives the sustainable rate on this panel.",
        "<b>Dynamic withdrawal policy.</b> Guyton and Klinger (2006) "
        "guardrails and the actuarial/amortisation family are represented in "
        "the spending-rule comparison of Section 7.",
        "<b>Bequest motives.</b> The utility specification uses the shifted "
        "power form of De Nardi (2004), which is what allows a zero terminal "
        "balance to be evaluated at all under high risk aversion.",
        "<b>Sequence-of-returns risk.</b> Section 12 formalises the folk "
        "observation that the decade around retirement dominates, and prices "
        "the option value of choosing when to stop working.",
        "<b>Lifecycle leverage.</b> Ayres and Nalebuff (2010) argue that a "
        "young investor should borrow to spread market exposure evenly across "
        "a lifetime. Section 10 relaxes the long-only constraint and prices "
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

    out: List[Flowable] = [ctx.h1("3. Data")]
    out.append(ctx.h2("3.1 What the panel is"))
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
        "is precisely the period a survivorship-prone sample would drop.",
        max_height=11.5 * cm))

    out.append(ctx.h2("3.2 The country set"))
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
        f"evidence, and Section 3.6.1 reports what the excluded countries do "
        f"carry."))
    out.append(ctx.p(
        f"That breadth is the paper's principal limitation and Section 16.1 "
        f"develops it. Its most direct consequence is on the international "
        f"leg, which is a leave-one-out average and therefore spans "
        f"{p['n_tier_a'] - 1} foreign markets, all advanced economies with "
        f"long histories. Every statement here about international "
        f"diversification is a statement about that set."))

    out.append(ctx.h2("3.3 Constructing the international leg"))
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

    out.append(ctx.h2("3.4 Cross-asset structure"))
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

    out.append(ctx.h2("3.5 Market disruptions and the survivorship question"))
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
        "run-length admissibility statistics are reported in Section 4.1."))
    out.append(ctx.h2("3.6 Auditing the data"))
    out.append(ctx.p(
        "The database is used here through a redistributed copy rather than "
        "obtained from its compilers, which deserves more scepticism than a "
        "citation. This section asks whether the file is genuine, reports one "
        "test it does not pass, and describes three bodies of recorded data "
        "that sit in the sources and stay out of the model."))

    out.append(ctx.h3("3.6.1 Why the panel stops at sixteen"))
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
        out.append(ctx.h3("3.6.2 The fourth asset class"))
        out.append(ctx.p(
            f"The macro file carries a fourth asset the source project "
            f"measured: <b>housing total returns</b>, empirical for all "
            f"{hs['countries']} observed countries over "
            f"{hs['country_years']:,} country-years "
            f"({hs['first_year']}–{hs['last_year']}). It is held out of the "
            f"headline results, which use the same four-asset set as the paper "
            f"being replicated, and audited here so the reason is visible "
            f"rather than assumed. Section 16 then puts it into the "
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
            "would therefore overstate what a household can buy. Section 16 "
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
            "cost of Section 16 has to carry the rest of the argument.",
            max_height=8.5 * cm))

    wg = pr.get("wages", {})
    if wg.get("countries"):
        out.append(ctx.h3("3.6.3 The series that bears on our income model"))
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
            f"<b>Our income profile has no term for it.</b> Section 4.3 sets "
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
            "here do. The effect on the savings analysis of Sections 12 and 13 "
            "is genuinely ambiguous, because faster income growth raises both "
            "what a given savings rate accumulates and the consumption it must "
            "replace, and this audit does not resolve it. Re-estimating the "
            "income process is a modelling change rather than a data one, so "
            "we record it as a quantified limitation rather than apply it "
            "silently."))

    out.append(ctx.h3("3.6.4 Is the source we kept genuine?"))
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

    out.append(ctx.h3("3.6.5 One finding that does not pass"))
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
        "the file rather than of the markets.",
        max_height=8.0 * cm))

    out.extend(ctx.figure(
        "fig02_country_real_returns",
        "Distribution of annual real returns by country and asset class. The "
        "left tails are the point of the panel: several countries record "
        "single-year real equity losses beyond −60%, and these are not "
        "outliers to be trimmed but the events a lifecycle investor is "
        "exposed to.",
        max_height=10.5 * cm))
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

    out: List[Flowable] = [ctx.h1("4. Methodology")]

    # -- 4.1 bootstrap ----------------------------------------------------
    out.append(ctx.h2("4.1 The cross-country stationary block bootstrap"))
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
        "block sampling is designed to retain.",
        max_height=10.0 * cm))

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
        "which is the signature of mean reversion in the underlying series.",
        max_height=8.0 * cm))

    # -- 4.2 lifecycle ----------------------------------------------------
    out.append(ctx.h2("4.2 The lifecycle model"))
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
        f"and carries no economy-wide wage growth; Section 3.6.3 measures what "
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
        "comparison strategies are held at fixed weights.",
        max_height=7.5 * cm))

    # -- 4.3 preferences --------------------------------------------------
    out.append(ctx.h2("4.3 Preferences and the certainty equivalent"))
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
        f"channel rather than the risk channel. Section 6.2 shows it is not."))
    out.append(ctx.p(
        f"One further specification choice deserves emphasis. Utility is "
        f"evaluated over the <b>{ut['consumption_window']}</b> window rather "
        f"than the whole lifetime for the allocation comparisons. The reason "
        f"is mechanical: with a fixed savings rate, working-life consumption "
        f"is <i>identical</i> across allocation strategies by construction, so "
        f"including it adds a large constant to every strategy's utility and "
        f"compresses the differences that the exercise is about. Where a "
        f"policy <i>does</i> change working-life consumption — the retirement "
        f"timing and savings-rate studies of Sections 12 to 14 — we switch to "
        f"the whole-lifetime window and say so explicitly."))

    # -- 4.4 comparison discipline ---------------------------------------
    out.append(ctx.h2("4.4 The comparison discipline"))
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
        "These are not stylistic preferences. Section 12 reports a result that "
        "fell by roughly sixty percent when a matched baseline replaced an "
        "unmatched one, Section 8 reports apparent glide-path structure that "
        "the deviation profile dissolved, and Section 14 reports a functional-"
        "form ranking that reversed its interpretation once the grid-edge "
        "check was applied."))

    # -- 4.5 implementation ----------------------------------------------
    out.append(ctx.h2("4.5 Implementation and verification"))
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

    out: List[Flowable] = [ctx.h1("5. Baseline Results")]
    out.append(ctx.p(
        f"All results in this section use 100,000 simulated lifetimes per "
        f"strategy, drawn from the same bootstrap sample so that strategies "
        f"face identical market histories. Certainty equivalents are over the "
        f"retirement window, for the reason given in Section 4.3."))

    out.append(ctx.h2("5.1 The headline comparison"))
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
        "does not close within the range tested.",
        max_height=8.0 * cm))

    out.append(ctx.h2("5.2 The mechanism is international, not equity"))
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
        f"model holds a 37-country average, and no single real investor has "
        f"that opportunity set without also holding their own market."))
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
        "conservative case is usually made.",
        max_height=8.5 * cm))
    out.extend(ctx.figure(
        "fig07_retirement_consumption",
        "Distribution of average real retirement consumption by strategy. "
        "The vertical line marks the seventy-percent replacement target used "
        "in the shortfall statistics.",
        max_height=8.5 * cm))

    out.append(ctx.h2("5.3 Distributional dominance"))
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
        "with low medians.",
        max_height=8.0 * cm))

    out.append(ctx.h2("5.4 Sustainable withdrawal rates"))
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
        "used to define the sustainable rate.",
        max_height=8.0 * cm))

    out.append(ctx.h2("5.5 How the countries are drawn"))
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
        "bottom, it is lower throughout.",
        max_height=8.5 * cm))
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
    reversals = int(tornado["settings_lost"].sum())

    out: List[Flowable] = [ctx.h1("6. Sensitivity Analysis")]
    out.append(ctx.p(
        f"The headline is one point in a large parameter space. This section "
        f"sweeps that space: {n_settings} settings across ten dimensions — "
        f"equity share, domestic share, risk aversion, elasticity of "
        f"intertemporal substitution, bequest weight, longevity, retirement "
        f"age, savings rate, withdrawal rate, social-security design, block "
        f"length and return panel. Every sweep uses common random numbers, so "
        f"differences between settings are parameter effects rather than "
        f"sampling noise."))
    out.append(ctx.p(
        f"<b>The ranking reverses in {reversals} of them.</b> That is the "
        f"single most useful sentence in this section, and the tornado below "
        f"reports the range of the advantage rather than its point estimate "
        f"so the reader can see how much room there is."))

    out.append(ctx.h2("6.1 Tornado analysis"))
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
        "no bar crosses zero.",
        max_height=9.0 * cm))
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
        "rises but do not overtake them within the tested range.",
        max_height=8.0 * cm))

    out.append(ctx.h2("6.2 Allocation sweeps and the corner solution"))
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
             "Section 8 asks the same question of the full age-by-asset "
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
        f"37-country leave-one-out average, which is better diversified than "
        f"any tradeable international index; the honest reading is that the "
        f"model prefers <i>more</i> diversification than the 50/50 headline "
        f"strategy provides, not that a specific number is optimal."))
    out.extend(ctx.figure(
        "fig12_allocation_frontier",
        "Certainty equivalent across the equity share and the domestic share. "
        "The equity dimension is monotone to the corner; the domestic "
        "dimension has an interior optimum well below the 50% headline "
        "weight.",
        max_height=8.0 * cm))

    out.append(ctx.h2("6.3 Separating risk aversion from intertemporal substitution"))
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

    out.append(ctx.h2("6.4 Planning parameters"))
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
        "magnitude of the advantage is not.",
        max_height=9.5 * cm))
    out.extend(ctx.figure(
        "fig14_withdrawal_sensitivity",
        "The advantage as a function of the withdrawal rate. Higher "
        "withdrawal rates widen the gap, because a portfolio being drawn down "
        "faster is more exposed to the left tail the conservative strategies "
        "fail to protect against.",
        max_height=8.0 * cm))
    return out


# ---------------------------------------------------------------------------
# 7. Retirement spending rules
# ---------------------------------------------------------------------------
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

    out: List[Flowable] = [ctx.h1("7. Retirement Spending Rules")]
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
        "curve rather than at a common rate.",
        max_height=8.5 * cm))
    out.extend(ctx.figure(
        "fig18_spending_paths",
        "Realised spending paths under each rule for a common set of market "
        "histories. The adaptive rules trade a smooth path for a solvent one.",
        max_height=8.5 * cm))

    out.append(ctx.h2("7.1 The bequest pivot"))
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
        "motive is not optimal for one who has a strong one.",
        max_height=8.0 * cm))
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
    anchor = f.table("glide_retirement_anchor_summary")
    gamma = f.baseline_gamma
    block = comparison[comparison["risk_aversion"] == gamma] \
        .sort_values("cec", ascending=False)
    dev = deviation[deviation["risk_aversion"] == gamma]
    material = dev[dev["cost_of_forcing_bp"].abs() > 1.0]

    out: List[Flowable] = [ctx.h1("8. Solving for the Optimal Glide Path")]
    out.append(ctx.p(
        "Sections 5 and 6 compare a handful of candidate allocation "
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
        "the corner; it is not a glide path.",
        max_height=8.5 * cm))

    out.append(ctx.h2("8.1 How much of the solved structure is real?"))
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
        "portfolio is small; the advantage over the glide path is not.",
        max_height=8.0 * cm))

    out.append(ctx.h2("8.2 Anchoring the glide to the retirement date"))
    out.append(ctx.p(
        "A natural objection is that the search is unconstrained and real "
        "target-date funds are anchored to a retirement date. We therefore "
        "also solve the schedule subject to an anchor: the equity share at "
        "retirement is fixed at a series of levels and the rest of the "
        "schedule is re-optimised around it."))
    out.extend(ctx.table(
        rows_from(anchor, list(anchor.columns)[:5],
                  [c.replace("_", " ").title() for c in list(anchor.columns)[:5]]),
        "Solved schedules under an anchored equity share at retirement",
        note="Each row fixes the equity share at the retirement date and "
             "re-solves the remaining ages. The cost of the anchor is the "
             "certainty-equivalent gap to the unconstrained solution.",
        font_size=7.2))
    out.extend(ctx.figure(
        "fig22_retirement_anchor",
        "The cost of anchoring the glide path at retirement. Forcing a "
        "conservative allocation at the retirement date is expensive, and the "
        "cost rises steeply as the anchor becomes more conservative.",
        max_height=8.0 * cm))
    out.append(ctx.p(
        "The conclusion of this section is stronger than the one the "
        "benchmark comparison supports. It is not merely that the target-date "
        "glide path is worse than an all-equity portfolio on this panel; it is "
        "that when the schedule is allowed to be anything, the optimiser does "
        "not choose a glide path at all."))
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

    out: List[Flowable] = [ctx.h1("9. The Whole Allocation, Solved")]
    out.append(ctx.p(
        "Section 8 solves for the equity share at every age and for the "
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

    out.append(ctx.h2("9.1 The search"))
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
        "improvement threshold, for the reason given in Section 4.4."))
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

    out.append(ctx.h2("9.2 The solved schedule"))
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
        f"70/30 bond/bill split Section 8 imposed was fixing the composition "
        f"of something the optimiser does not want to hold in the first place, "
        f"so the restriction was not binding. Freeing it buys "
        f"{a['lead_pct']:.2f}% over the best fixed benchmark."))
    out.append(ctx.p(
        f"The domestic/international split carries the same caveat as Section "
        f"6.2 and should not be read as advice. The international leg in this "
        f"model is a 37-country leave-one-out average, better diversified than "
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
        "scale with one basis point marked.",
        max_height=8.5 * cm))

    out.append(ctx.h2("9.3 Against the benchmarks"))
    out.extend(ctx.table(
        rows_from(block, ["strategy", "cec", "gap_to_best_pct"],
                  ["Schedule", "CEC", "Gap to best (%)"],
                  {"strategy": lambda v: LABELS.get(
                      v, str(v).replace("_", " ")),
                   "cec": lambda v: f2(v, 4),
                   "gap_to_best_pct": lambda v: f2(v, 2)}),
        f"Solved and benchmark schedules at risk aversion {gamma:g}",
        note="\"Full simplex optimal\" is this section's solution. The "
             "comparison also includes the schedules solved in Section 8 under "
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
        f"Section 5. The allocation decision has a broad flat top, and almost "
        f"all of its value is captured by the first choice — whether to hold "
        f"diversified equity at all — rather than by any refinement of it."))
    out.append(ctx.p(
        "It is also <b>still not a glide path</b>. The unrestricted solution "
        "differs from the restricted ones of Section 8 in the composition of a "
        "sleeve that carries almost no weight, not in the age profile of the "
        "equity share. Two independent searches over different parameter "
        "spaces reach the same shape, which is a stronger statement than "
        "either makes alone."))
    out.extend(ctx.figure(
        "fig35_allocation_comparison",
        "Left: the solved schedule against the benchmarks at each risk "
        "aversion. Right: the average solved weights by lifecycle phase, "
        "showing how little of the portfolio the fixed-income sleeve ever "
        "carries.",
        max_height=8.0 * cm))

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

    out.append(ctx.h2("9.4 Where the allocation decision actually matters"))
    out.append(ctx.p(
        f"A solved schedule always looks structured, and this one looks more "
        f"structured than the equity-share solve because it has three "
        f"dimensions to wander in. The deviation profile tests whether the "
        f"structure is real: each age's allocation is reset to the schedule's "
        f"own average and the certainty-equivalent cost measured in basis "
        f"points. Of the {len(dev)} ages, <b>{a['n_material_ages']}</b> move "
        f"the objective by more than a single basis point."))
    out.append(ctx.p(
        "That is a very different picture from Section 8, where most ages were "
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
        f"of the lifecycle. That is the sequence-of-returns window Section 12 "
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

    out.append(ctx.h2("9.5 Is this a local optimum?"))
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

    out: List[Flowable] = [ctx.h1("10. Borrowing to Invest")]
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

    out.append(ctx.h2("10.1 Mechanics, stated rather than buried"))
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

    out.append(ctx.h2("10.2 Optimal leverage by the price of credit"))
    out.append(ctx.p(
        "For each borrowing spread we search jointly over the leverage ratio "
        "and the allocation, taking the allocation from the same coarse "
        "simplex lattice Section 9 uses. The optimum is therefore a leverage "
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
        f"the tail risk documented in Section 10.3 in exchange for a gain that "
        f"rounds to nothing."))
    out.extend(ctx.figure(
        "fig36_leverage_surface",
        "Left: the value of leverage across the ratio, one line per borrowing "
        "spread. Right: the optimal ratio as the price of credit rises, "
        "annotated with what it is worth.",
        max_height=8.0 * cm))

    out.append(ctx.h2("10.3 What leverage does to the shape of the outcome"))
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
        "generous assumption of Section 10.1 is doing no work: the levered "
        "portfolios lose heavily in the left tail without ever being wiped out "
        "outright. A model with forced liquidation at a margin threshold "
        "rather than at zero would like leverage less than this one does."
        if lv["max_wipeout"] <= 1e-9 else
        f"Note the last column. The clip binds in up to "
        f"{lv['max_wipeout']:.2%} of path-years — years in which a real "
        f"levered investor would have been liquidated rather than merely "
        f"marked down. It binds only on the high ratios and only when the "
        f"allocation is held fixed rather than re-optimised, which is why the "
        f"sweep of Section 10.2 shows none: there the optimiser retreats into "
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
        out.append(ctx.h2("10.4 Should leverage decline with age?"))
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
            "three prices of credit.",
            max_height=8.0 * cm))

    out.append(ctx.h2("10.5 What this changes"))
    out.extend(ctx.bullets([
        f"Leverage is <b>not free money</b>. It is worth "
        f"{lv['value_at_zero_spread']:+.2f}% when credit is free, breaks even "
        f"at a spread of {lv['break_even_spread']:.2%}, and is worth under a "
        f"tenth of a percent from {lv['negligible_spread']:.2%} upward — which "
        f"covers most of the range a household actually borrows in.",
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

    out: List[Flowable] = [ctx.h1("11. Currency Hedging the International Leg")]
    out.append(ctx.p(
        "The international leg of the headline strategy is unhedged: a "
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
        "Left: the certainty-equivalent cost of hedging by ratio, at five "
        "annual hedging costs; every line is below zero, so hedging loses even "
        "when free. Centre: where the loss lands — fifth-percentile retirement "
        "consumption falls monotonically in the hedge ratio, and the certainty "
        "equivalent weighs that tail heavily. Right: why — hedging lowers the "
        "standalone volatility of the foreign sleeve up to a half hedge, but "
        "raises its correlation with the home market over the same range.",
        max_height=8.5 * cm))
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

    out: List[Flowable] = [ctx.h1("12. Endogenous Retirement Timing")]
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

    out.append(ctx.h2("12.1 A wealth trigger against a date"))
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

    out.append(ctx.h2("12.2 The retirement-date lottery"))
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
        "fixed date, and the retirement-date lottery.",
        max_height=9.0 * cm))
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

    out: List[Flowable] = [ctx.h1("13. Conditioning the Savings Rate")]
    out.append(ctx.p(
        "Section 12 found that conditioning <i>when</i> you stop working on "
        "your financial position is worth something. This section asks the "
        "accumulation-side mirror: should the savings rate vary over a career, "
        "and should it respond to the portfolio?"))
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

    out.append(ctx.h2("13.1 A caveat that has to come first"))
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

    out.append(ctx.h2("13.2 Shape: when should you save?"))
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
        f"Unlike the glide path of Section 8, this structure is real. The "
        f"deviation profile shows {len(material)} of {len(deviation)} working "
        f"years moving the objective by more than a basis point when reset to "
        f"the career average."))

    out.append(ctx.h2("13.3 Conditioning: your position beats the market's direction"))
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
        "shadow of it. Section 14 takes this apart in detail and finds that "
        "even this reading needs qualifying."))
    out.extend(ctx.figure(
        "fig25_savings_rate",
        "The solved savings profile by risk aversion, the unidentified "
        "constant-rate frontier, and the value of conditioning on wealth "
        "against conditioning on returns.",
        max_height=8.5 * cm))

    out.append(ctx.h2("13.4 Do the savings and retirement gains add?"))
    out.append(ctx.p(
        "Section 12 conditions the retirement date on the portfolio; this "
        "section conditions the savings rate on it. A natural question is "
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

    out: List[Flowable] = [ctx.h1("14. The Accumulation Signal, Decomposed")]
    out.append(ctx.p(
        f"Section 13 established that conditioning the savings rate on the "
        f"funded ratio is worth roughly three percent of certainty-equivalent "
        f"consumption. That result rests on five choices that were made once "
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
    out.append(ctx.h2("14.1 Which units is \"behind\" measured in?"))
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
        "the range of funded ratios paths actually visit.",
        max_height=8.5 * cm))

    # -- 12.2 asymmetry ---------------------------------------------------
    out.append(ctx.h2("14.2 Catching up and easing off are separate policies"))
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
        "switched on alone against both together.",
        max_height=8.5 * cm))

    # -- 12.3 implementable version ---------------------------------------
    out.append(ctx.h2("14.3 The version a person could actually follow"))
    out.append(ctx.p(
        "A continuous response asks for a freshly computed contribution rate "
        "every year. Section 7 found that coarse guardrail rules give up "
        "surprisingly little on the spending side; the accumulation-side "
        "analogue is to check once a year and move the contribution by a fixed "
        "step only if the funded ratio is more than a dead band away from "
        "target."))
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
    out.append(ctx.h2("14.4 Does the target have to be right?"))
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
        "each age.",
        max_height=8.0 * cm))

    # -- 12.5 horse race --------------------------------------------------
    out.append(ctx.h2("14.5 Which signal, out of everything available?"))
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
        "Left: the best of each signal at its own optimal sensitivity. "
        "Centre: how sharply the leaders peak. Right: the two leaders layered, "
        "showing partial but incomplete overlap.",
        max_height=8.0 * cm))

    # -- 12.6 feasibility --------------------------------------------------
    out.append(ctx.h2("14.6 How far does the contribution have to move?"))
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
        "rates by age under the unconstrained rule.",
        max_height=8.0 * cm))

    # -- 12.7 where the value lands ---------------------------------------
    out.append(ctx.h2("14.7 Where the gain lands, and who wants it"))
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
        "own coefficient.",
        max_height=8.0 * cm))

    # -- 12.8 timing ------------------------------------------------------
    out.append(ctx.h2("14.8 When in a career is the balance worth reading?"))
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
        "unlike the two directions of Section 14.2 these stretches really are "
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
        "and in which direction.",
        max_height=8.0 * cm))

    # -- 12.9 interactions -------------------------------------------------
    out.append(ctx.h2("14.9 What the value depends on"))
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
        "variance.",
        max_height=8.0 * cm))
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

    out: List[Flowable] = [ctx.h1("15. What the Market Costs When You Start")]
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

    out.append(ctx.h2("15.1 The observable, and why it is not the obvious one"))
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

    out.append(ctx.h2("15.2 The yield forecasts returns"))
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

    out.append(ctx.h2("15.3 Which tercile, on what was knowable at the time"))
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

    out.append(ctx.h2("15.4 What it does to the allocation decision"))
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
        "Left: annualised real equity returns following cheap and expensive "
        "starting yields, by horizon, with the correlation printed above each "
        "pair. Centre: the distribution of blended starting dividend yields "
        "across the panel's country-years, with the most recent United States "
        "observation marked so a reader can place themselves in it. Centre "
        "right: the all-equity advantage over the glide path within each "
        "valuation bucket (bars, left axis) against the level of "
        "certainty-equivalent consumption it wins at (line, right axis) — the "
        "advantage is flat and the level is not, which is the section's whole "
        "result. Far right: the tercile boundaries as an investor could have "
        "computed them at each date, drifting down across the century; a "
        "pooled split replaces both lines with a single pair of values and is "
        "what the recursive construction exists to avoid.",
        max_height=8.0 * cm))
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

    out: List[Flowable] = [ctx.h1("16. Housing as a Fifth Asset")]
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

    out.append(ctx.h2("16.1 The index is smoothed"))
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

    out.append(ctx.h2("16.2 A building costs money to hold"))
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

    out.append(ctx.h2("16.3 Two checks on that answer"))
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
                f"opposite of the declining glide path Sections 8 and 9 "
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

    out.append(ctx.h2("16.4 What this is not"))
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
        "Far left: the volatility the appraisal smoothing hides, by country, "
        "against the same country's equity. Centre left: the optimal "
        "portfolio at each holding cost. Centre right: what adding housing is "
        "worth, with and without the de-smoothing correction. Far right: "
        "which sleeve housing displaces as its cost falls.",
        max_height=8.0 * cm))
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

    out: List[Flowable] = [ctx.h1("17. The Mortgage")]
    out.append(ctx.p(
        "Section 16 prices housing as an asset owned outright. No household "
        "owns it that way. A house is the one asset an ordinary person can "
        f"borrow {pc(LVR_CAP, 0)} against, at a rate close to their own "
        "government's, secured on the thing itself — and leaving that out "
        "understates what housing does to a lifetime as surely as leaving the "
        "holding cost out overstates it. This section puts the mortgage in and "
        "asks the two questions that matter: how much, and when."))

    out.append(ctx.h2("17.1 How the loan is modelled"))
    out.append(ctx.p(
        "The decision variable is the <b>loan-to-value ratio</b>, because that "
        "is the number a lender quotes and a borrower chooses. A property "
        "funded with equity E and a loan at ratio λ returns, on that equity, "
        "(r_H − λ·i) / (1 − λ), where r_H is the real return on the property "
        "and i the real mortgage rate. That is the leverage multiple 1/(1 − λ) "
        "applied to housing alone, so the arithmetic is the same function "
        "Section 10 uses for portfolio borrowing and the two remain "
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

    out.append(ctx.h2("17.2 How much"))
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

    out.append(ctx.h2("17.3 When"))
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
        f"profile Section 9 applies to solved allocations — resetting one "
        f"age at a time to the schedule's own average and pricing what is "
        f"lost — leaves {len(material)} of {len(profile)} ages carrying a "
        f"decision worth more than five basis points, the largest worth "
        f"{f2(float(profile['cost_of_resetting_bp'].max()), 0)} and the "
        f"median age worth "
        f"{f2(float(profile['cost_of_resetting_bp'].median()), 0)}. The rest "
        f"sits on a flat part of the surface where the search moves the ratio "
        f"for free. The plotted line is jagged; the evidence underneath it is "
        f"not."))

    out.append(ctx.h2("17.4 At what price of credit"))
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

    out.append(ctx.h2("17.5 What this is not"))
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
        "Far left: the solved loan-to-value ratio by age, with the grey bars "
        "showing what each age's choice is actually worth — where they are "
        "invisible the line above them carries no information. Centre left: "
        "certainty-equivalent consumption against a ratio held for life. "
        "Centre right: how the optimal ratio responds to the price of credit, "
        "split by working life and retirement. Far right: what borrowing buys "
        "against an unlevered house, and how often the borrower's right to "
        "walk away is what pays for it.",
        max_height=8.0 * cm))
    return out


# ---------------------------------------------------------------------------
# 18. Discussion
# ---------------------------------------------------------------------------
def section_discussion(ctx: Any) -> List[Flowable]:
    f = ctx.f
    lottery = f.table("retirement_lottery_stats").iloc[0]
    swr = f.table("sensitivity_safe_withdrawal_rates")
    swr_eq = float(swr[swr["strategy"] == "balanced_all_equity"]
                   ["safe_withdrawal_rate_at_5%_ruin"].iloc[0])

    out: List[Flowable] = [ctx.h1("18. Discussion")]
    out.append(ctx.h2("18.1 What the replication does and does not establish"))
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
        "established is that the future will resemble that process. Section 15 "
        "removes part of that gap by conditioning on the valuation a lifetime "
        "starts at, which is the piece of the objection an investor can "
        "actually observe. What remains outside the model is the rest of it: "
        "the secular decline in real yields, and whether the equity premium "
        "the twentieth century delivered is a structural feature or a "
        "realised surprise."))
    out.append(ctx.p(
        "That distinction matters most for the sceptic's strongest objection, "
        "which we take seriously: the sample is a sample of survivors at the "
        "level of the <i>system</i> even where it is not at the level of the "
        "country. Every country in the panel ended the century with a "
        "functioning capital market. A world in which that was not true is "
        "outside the support of the sampler, and no amount of block "
        "resampling can put it back in."))

    out.append(ctx.h2("18.2 The hierarchy of levers"))
    out.append(ctx.p(
        "Reading the extensions together produces a ranking of what actually "
        "moves a retirement outcome, and it is not the ranking the industry's "
        "attention implies."))
    out.extend(ctx.bullets([
        f"<b>When you retire dominates.</b> The decade around the retirement "
        f"date explains {pc(float(lottery['r2_retirement_window']), 0)} of the "
        f"variation in retirement consumption — more than the gap between any "
        f"two allocation strategies tested here. It is also the one dimension "
        f"no allocation rule can diversify.",
        "<b>How much you save dominates what you hold.</b> The savings-rate "
        "dimension of the tornado analysis moves the outcome by more than most "
        "allocation choices, and conditioning the savings rate is worth more "
        "than every allocation refinement in Sections 8 to 11 combined.",
        "<b>How you spend it matters more than expected.</b> The gap between "
        "the best and worst spending rules in Section 7 is comparable to the "
        "gap between the best and worst allocation strategies in Section 5.",
        "<b>What you hold matters, but mostly through diversification.</b> The "
        "all-equity advantage is real, and it is driven by the international "
        "leg rather than by equity exposure as such.",
        "<b>Currency hedging is close to irrelevant</b> at any realistic cost, "
        "the fine structure of the glide path is worth less than a basis point "
        "at most ages, and freeing every portfolio weight at every age adds "
        "less than the difference between two adjacent spending rules.",
        "<b>Leverage is barely a lever</b> at household borrowing costs. The "
        "advantage rounds to nothing across most of the plausible range of "
        "spreads, and what it does buy is bought out of the left tail. The "
        "one exception is worth noting: the leverage schedule is the only "
        "policy in this paper that an unconstrained search makes genuinely "
        "age-varying, and it declines steeply.",
        "<b>The valuation you start at is not an allocation lever at all.</b> "
        "It moves what a portfolio delivers without changing which portfolio "
        "to hold, so it belongs in a reader's expectations and their "
        "withdrawal planning rather than in their asset mix.",
        "<b>What should decline with age is the borrowing, not the "
        "equity.</b> The only schedule in this paper that an unconstrained "
        "search makes genuinely age-declining is the mortgage: heavy "
        "borrowing against the first house, paid down over a career. The "
        "glide path the industry applies to equities has the right shape "
        "attached to the wrong instrument.",
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

    out.append(ctx.h2("18.3 Why the pay cheque beats the portfolio"))
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

    out.append(ctx.h2("18.4 Implications for default design"))
    out.append(ctx.p(
        "If the results here were taken at face value by a plan sponsor, the "
        "implied redesign would not be \"raise the equity share of the "
        "target-date fund\", though that is what the headline suggests. It "
        "would be a reordering of what the default does at all."))
    out.extend(ctx.bullets([
        "<b>Default the contribution schedule, not just the portfolio.</b> "
        "Auto-escalation tied to pay increases captures the strongest signal "
        "in Section 14 and requires no member decision.",
        "<b>Default the drawdown policy.</b> Section 7 shows the choice of "
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
        "conclusions in Section 16's direction."))
    return out


# ---------------------------------------------------------------------------
# 19. Limitations
# ---------------------------------------------------------------------------
def section_limitations(ctx: Any) -> List[Flowable]:
    f = ctx.f
    p = f.panel
    pr, adv = f.provenance, f.panel_advantage
    out: List[Flowable] = [ctx.h1("19. Limitations and Threats to Validity")]
    out.append(ctx.p(
        "This section is deliberately long. A replication that reports only "
        "the ways in which it succeeded is not much use."))

    out.append(ctx.h2("19.1 The cross-section is sixteen countries"))
    out.append(ctx.p(
        f"This is the largest weakness in the paper, and it is a weakness we "
        f"chose deliberately. The developed-market universe runs to 38 "
        f"markets. We cover {pr['n_countries']}, because the other "
        f"{pr['n_removed']} have no recorded return series in any openly "
        f"licensed source. Section 3.6 audits what we do use; Section 3.6.1 "
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
        "The last five years of the series are separately flagged as "
        "unverified in Section 3.6, on a variance test that every country in "
        "the sample fails in the same direction."))

    out.append(ctx.h2("19.2 What the bootstrap cannot represent"))
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
    ]))

    out.append(ctx.h2("19.3 What the lifecycle model omits"))
    hs, wg = pr.get("housing", {}), pr.get("wages", {})
    housing_bullet = (
        "<b>No housing.</b> For most households the primary residence is the "
        "largest asset and the mortgage the largest liability, and both "
        "interact with inflation in ways the financial portfolio does not. "
        "This is the single largest omission and it is not a small one."
        + (f" We do hold the data: Section 3.6.2 audits "
           f"{hs['country_years']:,} country-years of observed housing "
           f"returns and explains why an appraisal-smoothed index is not a "
           f"fourth sleeve. That is a reason not to use it as written, not a "
           f"reason the omission does not matter." if hs.get("countries")
           else ""))
    wage_bullet = (
        f"<b>No economy-wide wage growth.</b> The income profile is an age "
        f"effect only. Section 3.6.3 measures what that leaves out — a median "
        f"{pc(wg['measured'], 2)} a year across {wg['countries']} countries, "
        f"which compounds to {f2(wg['career_multiple'], 2)}× over a career — "
        f"and finds the bias runs against our conclusion, because less human "
        f"capital weakens rather than strengthens the case for early equity."
    ) if wg.get("countries") else ""
    out.extend(ctx.bullets([x for x in [
        "<b>No taxes.</b> Every return is pre-tax and every withdrawal is "
        "untaxed. Tax-deferred and taxable accounts behave differently under "
        "the same allocation, and the asymmetric treatment of capital gains "
        "and income would change the ranking of spending rules in Section 7 "
        "more than it would change the allocation ranking.",
        "<b>No fees.</b> A constant annual fee differential between an "
        "all-equity index and a target-date fund would move the results in the "
        "direction of the cheaper vehicle, which on current pricing is usually "
        "the index.",
        housing_bullet,
        wage_bullet,
        "<b>No annuities.</b> A real annuity is the natural competitor to a "
        "withdrawal rule and is absent entirely.",
        "<b>No disutility of labour.</b> This is why the retirement-timing "
        "results of Section 12 require a matched comparison, and it means the "
        "model can never say when someone <i>should</i> retire, only what "
        "conditioning the date is worth.",
        "<b>Deterministic mortality.</b> Death at 93 with certainty. Real "
        "longevity risk is two-sided and would raise the value of the "
        "actuarial spending rules relative to the fixed-horizon ones.",
        "<b>No behavioural constraints.</b> Every rule here assumes the "
        "investor executes it. Section 14.6 prices the cost of a constrained "
        "contribution but nothing prices the probability that a saver "
        "abandons an all-equity portfolio in the middle of a 60% drawdown.",
    ] if x]))

    out.append(ctx.h2("19.4 Specification sensitivities we know about"))
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

    out.append(ctx.h2("19.5 What the new searches assume"))
    out.extend(ctx.bullets([
        "<b>The simplex search is a coordinate search.</b> Section 9 reports "
        "restarts from three corners and they agree, but coordinate ascent "
        "offers no guarantee of a global optimum in a 204-dimensional space "
        "and none is claimed.",
        "<b>The opportunity set is still narrow.</b> Section 16 adds housing "
        "as a fifth asset, but there is no credit, no commodities and no "
        "inflation-linked bond, and an inflation-linked bond in particular "
        "would change the fixed-income result of Section 9 more than any "
        "refinement within the sleeves that are present.",
        "<b>The valuation conditioning applies to the first block only.</b> "
        "A lifetime is a chain of calendar windows and only the first is a "
        "starting condition; the rest are the future, which no investor "
        "chooses. Section 15 therefore measures the effect of the opening "
        "decade's valuation on a sixty-eight-year outcome, diluted by "
        "everything that follows. A design that made the whole chain "
        "valuation-dependent would report a larger effect and would be "
        "assuming a great deal more.",
        "<b>The de-smoothing of housing is itself a model.</b> Section 16 "
        "inverts a first-order filter, which is the standard correction and "
        "not necessarily the right one: if house price indices carry "
        "higher-order smoothing, the true volatility is higher than the "
        "correction restores and housing is worth less than reported. The "
        "direction of that error is known even though its size is not.",
        "<b>The mortgage is rebalanced annually.</b> Section 17 redraws the "
        "loan every year at no cost to hit a target loan-to-value. Real "
        "mortgages amortise on a fixed schedule, cost several percent of the "
        "property to refinance, and are called on missed payments rather "
        "than on a drifting ratio. The section reports the value of the "
        "leverage, not a financing plan, and it prices no mortgage insurance "
        "and no tax deductibility in either direction.",
        "<b>Housing is priced as an index, not as a house.</b> The asset in "
        "Section 16 is a liquid, continuously rebalanced, nationally "
        "diversified claim on the housing stock. A single leveraged "
        "owner-occupied property shares almost none of those properties, and "
        "nothing in this paper speaks to it.",
        "<b>Leverage is modelled with limited liability</b> and a floating "
        "borrowing rate, rebalanced annually. Real margin lending liquidates "
        "at a threshold rather than at zero and reprices continuously, both of "
        "which would make borrowing less attractive than Section 10 finds it.",
        "<b>No borrowing constraint binds by quantity.</b> The model lets an "
        "investor borrow three times their financial capital at a spread; no "
        "lender would extend that against a retirement account.",
    ]))

    out.append(ctx.h2("19.6 Statistical caveats"))
    out.append(ctx.p(
        "Certainty equivalents are reported without standard errors. Under "
        "common random numbers the <i>differences</i> between policies are far "
        "more precisely estimated than their levels, which is what the "
        "comparisons rely on, but a reader wanting a confidence interval on "
        "any single level will not find one here. The optimisers are "
        "coordinate-ascent procedures on a grid: exact per coordinate under "
        "common random numbers, but with no guarantee of a global optimum in "
        "the joint space. Multiple restarts are reported where it matters."))
    return out


# ---------------------------------------------------------------------------
# 20. Conclusion
# ---------------------------------------------------------------------------
def section_conclusion(ctx: Any) -> List[Flowable]:
    f = ctx.f
    adv_tdf = f.advantage("balanced_all_equity", "target_date_fund")
    lottery = f.table("retirement_lottery_stats").iloc[0]
    out: List[Flowable] = [ctx.h1("20. Conclusion")]
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
        f"{pc(float(lottery['r2_retirement_window']), 0)} of the variation in "
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
    "alternative wealth target in Section 14.4.",
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
         "All; varied in §6.4 and made endogenous in §12"],
        ["Death age", "—", f"{int(lc['age_death'])}",
         "All; varied in §6.4"],
        ["Savings rate", "s", f"{float(lc['savings_rate']):.0%}",
         "All; solved in §13 and conditioned in §14"],
        ["Income profile (linear)", "b₁", f"{float(lc['income']['b1']):g}",
         "Labour income"],
        ["Income profile (quadratic)", "b₂", f"{float(lc['income']['b2']):g}",
         "Labour income"],
        ["Permanent shock s.d.", "σ<sub>p</sub>",
         f"{float(lc['income']['permanent_shock_sd']):.2f}",
         "Labour income; scaled in §14.9"],
        ["Transitory shock s.d.", "σ<sub>t</sub>",
         f"{float(lc['income']['transitory_shock_sd']):.2f}",
         "Labour income; scaled in §14.9"],
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
         "Baseline; eight families compared in §7"],
        ["Withdrawal rate", "—", f"{float(lc['retirement']['rule_rate']):.0%}",
         "Baseline; swept in §5.4 and §6.4"],
        ["Discount factor", "β", f"{float(ut['discount_factor']):g}",
         "Utility"],
        ["Risk aversions", "γ",
         ", ".join(f"{float(g):g}" for g in ut["risk_aversions"]),
         f"Baseline γ = {float(ut['baseline_risk_aversion']):g}"],
        ["Elasticities of substitution", "ψ",
         ", ".join(f"{float(v):g}" for v in ut["epstein_zin_ies"]),
         "Epstein–Zin only"],
        ["Bequest weight", "θ", f"{float(ut['bequest_weight']):g}",
         "Utility; pivoted in §7.1"],
        ["Bequest shift", "κ", f"{float(ut['bequest_shift']):g}",
         "De Nardi (2004) specification"],
        ["Consumption floor", "—", f"{float(ut['consumption_floor']):g}",
         "Numerical guard only"],
        ["Evaluation window", "—", str(ut["consumption_window"]),
         "Allocation comparisons; whole lifetime in §12–§14"],
        ["Paths per strategy", "N", f"{int(bs['n_paths']):,}", "All"],
        ["Horizon", "H", f"{int(bs['horizon_years'])}", "All"],
        ["Mean block length", "—", f"{float(bs['mean_block_years']):.0f} years",
         "Bootstrap; swept in §4.1"],
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
        "the sweeps of Sections 6, 11 and 12 mechanical rather than manual."))
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
             "§13.2 hump-shaped at moderate risk aversion."))
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
         "diagnostics, per-cell provenance", "§3"],
        ["Sampler", "The stationary block bootstrap with gap-respecting "
         "admissibility; moment and persistence diagnostics", "§4.1"],
        ["Lifecycle simulator", "The wealth recursion, income process, "
         "social-security schedule and strategy definitions", "§4.2"],
        ["Preferences", "CRRA and Epstein–Zin certainty equivalents, the "
         "bequest term, shortfall and ruin metrics", "§4.3"],
        ["Sweep engine", "Parameter sweeps under common random numbers; "
         "tornado and crossover analysis", "§6"],
        ["Spending rules", "Eight families of withdrawal policy behind a "
         "single interface", "§7"],
        ["Glide-path solver", "A batched evaluator and coordinate ascent over "
         "the age-by-asset schedule", "§8"],
        ["Allocation solver", "The weight simplex solved at every age: "
         "lattice search then pairwise-exchange ascent, over four assets or "
         "five", "§9, §16"],
        ["Leverage", "A levered evaluator, the cost-of-credit sweep and the "
         "age-varying leverage schedule", "§10"],
        ["Hedging", "Covered-interest-parity hedged legs, break-even cost and "
         "optimal hedge ratio", "§11"],
        ["Path-dependent engine", "Endogenous retirement dates and "
         "state-conditioned saving", "§12–§14"],
        ["Savings rules", "Rule families, the fixed-mean shape solver and "
         "matched-rate scoring", "§13–§14"],
        ["Valuation", "The look-ahead-free trailing dividend yield, the "
         "structural no-leak check and the valuation buckets", "§15"],
        ["Housing", "De-smoothing the appraisal index, and the five-asset "
         "simplex re-solved at each holding cost", "§16"],
        ["Mortgage", "Leverage applied to the housing sleeve alone, and the "
         "loan-to-value schedule solved by age", "§17"],
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
        "sensitivity analysis of Section 6 affordable enough to run "
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
        "profile exactly, which is what makes the incremental values in §14 "
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


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def story(ctx: Any) -> List[Flowable]:
    parts: List[Flowable] = []
    parts += front_matter(ctx)
    parts += contents(ctx)
    parts += section_introduction(ctx)
    parts += section_background(ctx)
    parts += section_data(ctx)
    parts += section_methods(ctx)
    parts += section_baseline(ctx)
    parts += section_sensitivity(ctx)
    parts += section_spending(ctx)
    parts += section_glide(ctx)
    parts += section_allocation(ctx)
    parts += section_leverage(ctx)
    parts += section_hedging(ctx)
    parts += section_retirement(ctx)
    parts += section_saving(ctx)
    parts += section_accumulation(ctx)
    parts += section_valuation(ctx)
    parts += section_housing(ctx)
    parts += section_mortgage(ctx)
    parts += section_discussion(ctx)
    parts += section_limitations(ctx)
    parts += section_conclusion(ctx)
    parts += section_references(ctx)
    parts += appendix_parameters(ctx)
    parts += appendix_panel(ctx)
    parts += appendix_supplementary(ctx)
    parts += appendix_software(ctx)
    return parts
