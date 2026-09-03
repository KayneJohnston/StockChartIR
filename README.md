# Lifecycle Asset Allocation: a replication of Anarkulova, Cederburg & O'Doherty

An end-to-end, reproducible replication of the empirical dataset,
multi-country block bootstrap and lifecycle allocation results in
**Anarkulova, Cederburg and O'Doherty (2023/2024), *Beyond the Status Quo: A
Critical Assessment of Lifecycle Asset Allocation***.

The paper's claim is that a fully diversified all-equity portfolio beats
age-based target-date glide paths and the classic 60/40 portfolio over a
lifetime -- not only on average, but on downside and capital-preservation
measures too -- once you stop resampling the single luckiest equity history in
the developed world and draw from a panel of developed markets instead.

This repository rebuilds that argument from openly licensed primary data and
reaches the same conclusion.

## Headline result

100,000 simulated 68-year lifetimes (ages 25-93, retirement at 63), drawn from
a 16-country developed-market panel of recorded returns, 1890-2020. Certainty
equivalent
consumption is in multiples of initial annual real income; ruin is exhausting
financial wealth before age 93 under a 4% real withdrawal rule.

| Strategy | CEC (γ=2) | CEC (γ=5) | CEC (γ=10) | P(ruin) | Median bequest |
| --- | --- | --- | --- | --- | --- |
| 100% International Equity | 1.521 | 1.108 | 0.782 | 9.0% | 64.2 |
| **50/50 Domestic/International Equity** | **1.406** | **1.048** | **0.756** | **11.7%** | **44.7** |
| Target-Date Fund (glide path) | 1.214 | 0.943 | 0.704 | 22.2% | 9.4 |
| 60/40 Domestic Equity / Domestic Bonds | 1.105 | 0.871 | 0.671 | 24.2% | 10.2 |
| 100% Domestic Equity | 1.183 | 0.882 | 0.668 | 25.4% | 16.7 |
| 100% Bills (cash) | 0.847 | 0.739 | 0.618 | 56.4% | 0.0 |

The 50/50 all-equity portfolio **strictly dominates** the target-date fund and
60/40 on all 21 reported criteria (20 wins, 1 tie, 0 losses), across every
risk-aversion level and both Epstein-Zin IES values tested. The ordering
survives under per-block country resampling and under uniform country
weighting. Every path behind these numbers was drawn from recorded returns --
see "How much of this data is real" below.

**How precisely is any of this known?** Less precisely than four decimal places
suggests. A delete-one-country jackknife puts a standard error of **2.92
points** on the 5.79-point lead all-international holds over the 50/50 split —
a 95% interval of [0.08, 11.51], a t-statistic of 1.99. The direction is
supported by the panel; the magnitude is barely resolved. Sixteen countries is
sixteen countries however long the computer runs, and a reader should carry
that interval to every other number in this repository. See
[`docs/19_panel_robustness.md`](docs/19_panel_robustness.md).

**And what is it conditional on?** One assumption more than any parameter we
sweep. Every table above pays the same public pension in all sixteen
countries -- the US primary-insurance-amount schedule, worth about 44% of
average earnings, paid for life regardless of what the retiree owns. That is
a risk-free real annuity, and a large part of what looks like portfolio
performance is standing on it. Swap it for Australia's means-tested Age
Pension plus its compulsory 12% Superannuation Guarantee and the ordering
above reverses: the de-risking glide path wins, because inside the
assets-test taper a dollar of extra wealth costs more pension a year than any
asset in this panel reliably earns. It is the only place in the project where
the headline ranking fails, and it fails on a change to the pension rather
than to the returns. See
[`docs/25_pension_system.md`](docs/25_pension_system.md).

See [`docs/04_replicated_results_and_tables.md`](docs/04_replicated_results_and_tables.md)
for the full analysis and the caveats.

## The working paper

A full academic writeup of the whole project — abstract, data, methodology,
the baseline replication, every extension, discussion, limitations and four
appendices — is at
[`paper/lifecycle_asset_allocation.pdf`](paper/lifecycle_asset_allocation.pdf).

Every number in it is read from the pipeline's own CSV output at build time,
so a rerun that changed a result changes the paper rather than silently
contradicting it. The counts of sections, figures and extensions are derived
the same way, which is why none of them are quoted here.

```bash
python paper/build_paper.py
```

Every number in its prose is resolved at build time from `results/tables/`, so
the paper cannot drift away from the pipeline that produced it. See
[`paper/README.md`](paper/README.md).

## Documentation

The `docs/` files are numbered by **pipeline step** — `docs/13` is what
`python main.py --steps 13` writes — which is the order the work was built in,
not the order it is best read in. The paper reorders it: the headline first,
then everything that tests whether the headline holds, then the searches for a
better portfolio, then a wider asset menu, then the levers outside the
portfolio in the order a life meets them. `paper/lifecycle_asset_allocation.pdf`
is the read-through; the table below is the build.


| Document | Contents |
| --- | --- |
| [`docs/01_country_dataset_and_sources.md`](docs/01_country_dataset_and_sources.md) | Data lineage, source mapping, real-return construction, per-country summary statistics, coverage matrices, market-disruption handling |
| [`docs/02_multicountry_block_bootstrap.md`](docs/02_multicountry_block_bootstrap.md) | Bootstrap algorithm, mean/covariance/persistence preservation, block-length sensitivity, tail diagnostics |
| [`docs/03_lifecycle_utility_model.md`](docs/03_lifecycle_utility_model.md) | State transitions, cash-flow rules, drawdown algorithm, CRRA and Epstein-Zin definitions |
| [`docs/04_replicated_results_and_tables.md`](docs/04_replicated_results_and_tables.md) | Headline tables, distributional evidence, robustness, comparison with the paper, limitations |
| [`docs/05_sensitivity_analysis.md`](docs/05_sensitivity_analysis.md) | Sweeps over allocation, preferences, planning and sampling assumptions; safe withdrawal rates; tornado analysis |
| [`docs/06_retirement_spending_rules.md`](docs/06_retirement_spending_rules.md) | Eight families of withdrawal policy compared at each rule's own optimal rate; the bequest-motive pivot |
| [`docs/07_optimal_glide_path.md`](docs/07_optimal_glide_path.md) | Solving for the age-by-asset schedule directly: free-form and parametric optimisation, local-optimum checks |
| [`docs/08_currency_hedging.md`](docs/08_currency_hedging.md) | Covered-interest-parity hedged international leg; break-even hedging cost and optimal hedge ratio |
| [`docs/09_retirement_timing.md`](docs/09_retirement_timing.md) | Retirement as a path-dependent decision; value of conditioning on markets; the retirement-date lottery |
| [`docs/10_savings_rate.md`](docs/10_savings_rate.md) | When to save over a career, and whether the rate should respond to the portfolio |
| [`docs/11_accumulation_signal.md`](docs/11_accumulation_signal.md) | The savings-rate signal taken apart: functional form, asymmetry, target choice, eight competing signals, feasibility bands, and where the value lands |
| [`docs/12_full_allocation.md`](docs/12_full_allocation.md) | The full four-asset weight simplex -- domestic equity, international equity, bonds and bills -- solved at every year of the lifecycle |
| [`docs/13_leverage.md`](docs/13_leverage.md) | Borrowing to invest: the optimal leverage ratio and allocation at each price of credit, the break-even spread, and whether the ratio should decline with age |
| [`docs/14_data_provenance.md`](docs/14_data_provenance.md) | **Which numbers were observed and which were generated** -- source fingerprints, authenticity checks on the primary file, and whether the headline survives on observed data alone |
| [`docs/15_starting_valuation.md`](docs/15_starting_valuation.md) | Conditioning the result on how expensive the market was when a lifetime began, using only the dividend yield an investor could actually observe |
| [`docs/16_housing.md`](docs/16_housing.md) | Housing as a fifth asset: undoing the appraisal smoothing in the index, then sweeping the annual holding cost to find the price at which it stops being worth holding |
| [`docs/17_mortgage.md`](docs/17_mortgage.md) | How much of the house to borrow and at what age: the loan-to-value ratio solved against the domestic short rate plus a swept mortgage spread, capped at 80% |
| [`docs/18_sleeve_weighting.md`](docs/18_sleeve_weighting.md) | Whether the headline needs an equal-weighted international sleeve, or survives five constructions: equal, real GDP, population, GDP per capita and inverse volatility |
| [`docs/19_panel_robustness.md`](docs/19_panel_robustness.md) | Delete-one-country influence, the jackknife standard error sixteen countries actually support, and whether the ranking is stable through time |
| [`docs/20_fees.md`](docs/20_fees.md) | What fund costs do to the headline, and how large a domestic-versus-international fee differential it takes to undo it |
| [`docs/21_realised_cohorts.md`](docs/21_realised_cohorts.md) | The lifetimes the panel actually contains -- one country, one birth year, sixty-eight years in calendar order -- run with no resampling of any kind |
| [`docs/22_out_of_sample.md`](docs/22_out_of_sample.md) | Solving each schedule on one half of the record and scoring it on the other, to separate the transferable part of an in-sample gain from the fitted part |
| [`docs/23_human_capital.md`](docs/23_human_capital.md) | Correlating labour income with the home market -- the textbook reason to hold less of it, and the assumption every other result quietly makes |
| [`docs/24_mortality.md`](docs/24_mortality.md) | Replacing the certain ninety-third birthday with a Gompertz lifespan, by re-weighting the utility aggregation rather than re-simulating |
| [`docs/25_pension_system.md`](docs/25_pension_system.md) | Swapping the US social-security schedule the whole panel silently assumes for Australia's means-tested Age Pension and Superannuation Guarantee -- with a control that separates the means test from the smaller cheque |
| [`docs/26_turnover.md`](docs/26_turnover.md) | What the solved schedules cost to trade: turnover split into the drift no portfolio can avoid and the move the schedule chose, then priced |
| [`docs/27_inflation_state.md`](docs/27_inflation_state.md) | Recent inflation as a state variable: what one, three and five years of it predict about forward real returns by asset class, and whether it moves the optimal portfolio -- read once at the birth date and again at the retirement date, which is where it turns out to matter |
| [`docs/28_withholding_tax.md`](docs/28_withholding_tax.md) | Foreign dividend withholding tax -- the cost differential between holding your own market and everyone else's that is not a fee, is not negotiable and is not a choice -- swept to the rate at which it undoes the headline |
| [`docs/29_sequence_risk.md`](docs/29_sequence_risk.md) | Sequence-of-returns risk, isolated by keeping each lifetime's returns and shuffling their order: how much of the outcome is the ordering, which phase it lands in, and why that turns out to be a property of the withdrawal rule |
| [`docs/30_franking_credits.md`](docs/30_franking_credits.md) | Dividend imputation -- the credit that lands only on the home leg, exactly mirroring the withholding tax that falls only on the foreign one -- and what the two of them together do to the case for going abroad |

All thirty are **generated** by `main.py` from live pipeline objects --
edit `src/report.py`, not the Markdown.

## How much of this data is real

All of it.

The panel is **16 developed markets, 2,010 country-years, 6,030 return cells,
every one of them a number somebody recorded**. `docs/14` proves that rather
than asserting it: provenance is tracked per cell, the tier labels are derived
from those records, and a separate check reports any cell that is available but
not observed. It is empty.

That is a change, and it is the most important one in the project. This
replication used to cover **38 developed markets** to match the paper's
cross-section. It could only do that by generating 22 of them: their equity,
bond and bill returns were draws from a single-factor model fitted to a
randomly assigned observed donor, plus Gaussian noise. The audit measured the
damage -- 39.6% of country-years, and **38% of the average investor's
international leg**, rising to 59% after 2000 -- and the first version of this
README argued the simulated data were tolerable because they *diluted* the
headline rather than creating it.

That was true and not a good enough reason. A reader cannot check a number that
came out of a random number generator, and a cross-section of thirty-eight
reads as stronger evidence than one of sixteen while being weaker. The
generated countries are gone.

**What it costs:** the international leg is now an average over 15 other
markets rather than 37. That is a real limit on how far the mechanism
generalises and `docs/14` states it plainly.

**What the sources still hold that this does not use**, each audited in
`docs/14` and each with its reason:

* **Interest rates for four removed countries** (s8). Austria, Canada, Ireland
  and New Zealand have real long yields and short rates; their bonds and bills
  are rebuilt from those and reported. They cannot re-enter the panel, because
  a lifecycle investor needs a domestic *equity* return and no reachable source
  carries one for them.
* **Housing total returns**, 1,805 country-years across all 16 panel countries
  (s3.1). Median 6.6% real against 6.9% for equity at 8.9% s.d. versus 21.0% --
  but median lag-one autocorrelation is +0.34 against +0.06, and undoing that
  appraisal smoothing takes the s.d. to 12.0%. Most of the apparent free lunch
  is a measurement artefact.
* **Real wage growth** for all 18 macro countries, 2,269 country-years (s3.2).
  The median country compounded real wages at 1.41% a year (1.77% excluding the
  war years). The lifecycle income profile has no term for it: its hump is an
  *age* effect implying 1.18% a year, and the two are separate and additive in
  the estimation it comes from. Recorded as a quantified limitation rather than
  applied, since re-estimating the income process is a modelling change. The
  bias runs *against* the conclusion -- less human capital weakens the case for
  early equity.

The 16 countries pass every authenticity check: five independently known annual
returns land within tolerance, and the internal accounting identity fails at
the rate a genuine spliced-source database should. One finding does not pass --
all 16 show a variance collapse over 2016-2020 (sign-test p = 3e-05) -- so
those five years are flagged as unverified.

No new external source could be added. Outbound access from the build
environment is denied to the macrohistory host and every other bulk macro-data
provider tested, and guessing at redistributed mirrors would have reproduced
exactly the unverifiable provenance this audit exists to warn about. Sixteen
countries is the ceiling on observed equity here, stated as a limit rather than
worked around.

## Sensitivity

The headline result is one point in a large parameter space, so the pipeline
sweeps that space. Across **136 parameter settings spanning 10 dimensions --
allocation, risk aversion, IES, bequest weight, longevity, retirement age,
savings rate, withdrawal rate, social-security design, block length and return
panel -- not one reverses the ranking.** The 50/50 all-equity advantage over
the target-date fund ranges from 3.3% to 26.6%.

Three findings from the sweeps are worth pulling out.

**Home bias is the expensive mistake, not the equity/bond split.** The optimal
domestic share of the equity sleeve is 0-20% depending on risk aversion.
Moving from that optimum to 100% domestic equity costs more certainty
equivalent consumption than the entire gap between the 50/50 portfolio and the
target-date fund.

**The optimal equity share is 100% at every risk aversion tested**, up to
γ = 20. Rising risk aversion lowers the level of the certainty equivalent
everywhere but never moves the argmax toward bonds, and no rival overtakes the
all-equity portfolio anywhere on the grid.

**The 4% rule is not safe for any strategy on a developed-market panel.** The
withdrawal rate that holds ruin probability to 5%:

| Strategy | Safe withdrawal rate | Ruin at 4% |
| --- | --- | --- |
| 100% International Equity | 2.89% | 11.6% |
| 50/50 Domestic/International Equity | 2.79% | 14.5% |
| Target-Date Fund (glide path) | 2.20% | 22.4% |
| 60/40 | 1.80% | 24.4% |
| 100% Domestic Equity | 1.45% | 28.1% |
| 100% Bills | 1.29% | 57.9% |

The 4% rule was calibrated on US history — exactly the hindsight the paper
removes, and removing it costs more than a percentage point of retirement
income.

## Does it rest on one country, or one era?

`docs/19` rebuilds the panel sixteen times, once per country removed, and
re-runs the headline. The deletion is genuine: a dropped market vanishes from
every other country's international sleeve as well as from its own domestic
column, so each run asks what this project would say had that market's history
never been recorded.

**The ordering survives all sixteen deletions.** Germany costs the most —
removing it takes the lead from 5.79% to 3.46% — and it still leads. The United
States is not the load-bearing market: removing it *raises* the lead by 0.39
points. Re-seeding the bootstrap on an unmodified panel moves the lead by 0.20
points, which is the floor a shift has to clear; eleven of sixteen countries
clear it.

The same runs form a delete-one jackknife, which is where the standard error in
the headline block above comes from. It is the number to quote.

**The ranking is also stable through time**, on expanding windows an investor
could actually have stood in:

| Data available through | Lead of all-international over 50/50 |
|---|---|
| 1950 | 4.41% |
| 1970 | 5.26% |
| 1990 | 5.82% |
| 2020 | 5.79% |

What none of this addresses is survivorship in the sample frame itself:
deleting a country that is present says nothing about a country that was never
there.

## Does the headline need an equal-weighted foreign sleeve?

The international leg everywhere in this project is a leave-one-out **equal
weighting** across the other fifteen markets. That is a more diversified object
than any index a person could have bought — it holds as much Portugal as it
holds the United States — and the one place this work diverges from the
replicated paper depends on exactly how much diversification that sleeve
delivers. So `docs/18` rebuilds the panel under five constructions and re-runs
the headline through the same summariser each time.

| Weighting | Tilts towards | Effective markets | All-intl over 50/50 |
|---|---|---|---|
| Real GDP | large economies | 4.6 | **+3.71%** |
| Population | populous countries | 6.3 | +4.36% |
| Inverse volatility | historically stable markets | 13.8 | +4.41% |
| GDP per capita | rich countries | 14.1 | **+6.75%** |
| Equal-weighted | nothing | 15.7 | +5.79% |

The ordering survives all five; the harshest, real-GDP weighting, leaves 64% of
the equal-weighted gap. Two things are worth noting against the easy reading.
Equal weighting is **not** the construction kindest to the finding — GDP per
capita gives a larger gap. And concentration is not a sufficient statistic: the
gap correlates +0.75 with the effective number of markets, but the two schemes
that barely concentrate the sleeve still span 2.34 of the 3.04 total points
between them, so *what* a scheme tilts towards matters about as much as how far
it concentrates. A single alternative weighting is not a robustness check.

All weights are lagged, so every sleeve is one an investor could have held.
None of them is capitalisation weighting; this brackets the answer rather than
settling it.

## Do fees undo it?

Every return in this project is gross. That is fine for comparing strategies
drawn from the same panel, and not fine for the one divergence from the
replicated paper — because all-international pays the international fund's
expense ratio on the whole portfolio while the 50/50 split pays it on half.

`docs/20` charges the fee on the panel before the bootstrap sees it, on assets
rather than returns — `(1+r)(1-f)-1`, which matters over sixty-eight years.

| Extra annual fee on the foreign leg | Lead of all-international over 50/50 |
|---|---|
| 0 bp | 5.79% |
| 5 bp (modern index funds) | 5.51% |
| 19 bp (index funds circa 2000) | 4.75% |
| 75 bp (an active international fund) | 1.85% |
| **114 bp** | **0 — break-even** |

**The result survives every fund cost a real investor has faced.** It takes a
114 basis-point differential to cancel the lead, well beyond any index-fund
pair. The erosion is real even so: each basis point costs about 0.047 points of
lead, so an investor paying an active fund's 75 bp keeps under a third of the
advantage.

Not modelled: trading costs, spreads, taxes, platform fees, and the fact that
index funds did not exist for most of this sample.

## Should you hedge the currency?

`docs/08` splits the international sleeve into its two exposures — the foreign
*asset* return and the foreign *currency* — by building a covered-interest-parity
hedged leg, and sweeps the hedge ratio against an annual holding cost. Every
ratio is evaluated on **literally the same simulated lives**: the hedge never
changes which country-years are usable, and the sweep reuses one set of drawn
(country, calendar) indices, re-reading only the international leg.

| Hedge ratio | CEC gain at zero cost | Break-even annual cost |
| --- | --- | --- |
| 25% | -0.13% | never worth it |
| 50% | -1.02% | never worth it |
| 75% | -2.43% | never worth it |
| 100% | -4.24% | never worth it |

**Hedging is not worth doing at any ratio, even free.** Every hedge ratio
tested loses certainty-equivalent consumption before a single basis point of
cost is charged, and the loss grows monotonically with the ratio. There is no
break-even cost to quote because there is no cost low enough: the decision is
settled at zero.

**Why, and it isn't the usual story.** Hedging does cut the standalone
volatility of the foreign sleeve — but not monotonically: volatility bottoms
out at a 50% hedge (18.48%, against 21.69% unhedged) and rises again toward a
full hedge (21.17%).
In *real* terms, foreign currency partly hedges domestic inflation, and
hedging removes that offset. Meanwhile hedging raises the correlation between
the foreign sleeve and the home market, from 0.43 unhedged to a peak of 0.52
at a half hedge: currency movement is part of what makes foreign equity a
*diversifier* rather than a second helping of the same risk. Correlation with
domestic inflation moves the other way, from -0.04 to -0.28, so hedging trades
away an inflation offset the unhedged sleeve provides for free.

The second effect wins. "Hedging reduces risk" is true of the sleeve in
isolation and false of the portfolio holding it — which is why every ratio
loses even when the hedge is free. (The conventional
advice to hedge foreign *bonds* survives untouched — this tests equities only.)

## The optimal glide path

`docs/07` stops testing glide paths and solves for one. Equity share gets a
free parameter at **every age** — 68 of them, no smoothness imposed — plus the
domestic split on five-year bands, optimised by coordinate ascent under common
random numbers.

**There is almost no glide.** The unconstrained optimum is 100% equity at
every age before retirement and ~96% after, at every risk aversion from γ=2
to γ=10. It is not a declining path, not a rising one, and not a U.

| Strategy | CEC (γ=5) | Gap to optimum |
| --- | --- | --- |
| Free-form optimal (68 free parameters) | 1.0528 | — |
| Parametric optimal (6 free knots) | 1.0490 | −0.35% |
| **Static 100% international equity** | **1.0415** | **−1.07%** |
| 50/50 all-equity | 0.9952 | −5.47% |
| Industry target-date glide path | 0.9278 | −11.87% |
| 60/40 | 0.8672 | −17.62% |

That third row is the finding. Solving a 68-dimensional allocation problem
beats the simplest possible portfolio by **one percent** of lifetime
consumption. Almost all the available value is in the *level* of equity
exposure and the international split; almost none is in its age profile.

**The one real feature is a dip at the retirement date.** Under the 4% rule
the solved schedule drops to 70% equity at exactly age 63 and recovers by 68 —
a 29pp dip. That is not the investment problem talking, it is the withdrawal
rule: the 4% rule sets thirty years of spending as a fraction of wealth on one
date, so wealth at 63 is an anchor worth protecting. Re-solving under a
percent-of-portfolio rule or a life-expectancy rule, neither of which anchors
on any single date, the dip vanishes entirely and the schedule is flat at
100%.

So the practical rule is the opposite of what target-date funds do: *if your
withdrawal policy anchors on your balance at one date, de-risk briefly around
that date; if it does not, do not de-risk at all.*

Two guards against over-reading the solved shape. Restarting the search from
flat schedules at 20%, 60% and 100% equity converges to the same certainty
equivalent within 0.013%. And forcing each age back to 100% equity one at a
time shows **54 of 68 ages cost less than a basis point** — several are
negative. Only the ages at and just after retirement clear one basis point, so
the rest of the plotted line is search noise on a flat objective, not
structure.

## When to save, and on what

`docs/10` is the accumulation-side mirror of the retirement-timing work. Two
things could make the savings rate vary, and conflating them credits the wrong
one, so they're measured separately.

**First, a caveat the document leads with: this model cannot identify the
savings *level*.** The certainty equivalent peaks at a constant 5% and falls
above it — but that's set by the discount factor (β=0.96 over a 38-year
working life discounts retirement consumption by 0.21) and by risk aversion
acting on the left tail of consumption, which with a floored retirement and
risky labour income sits in *working* life. None of that is something a panel
of historical returns has a view on. So everything below **pins the average
rate at 10%** and asks only *when*, and *on what*, to save it.

**Shape — and it flips with risk aversion.** Solving a free rate for each of
the 38 working years, with the average pinned:

| Risk aversion | Shape | First quarter | Middle half | Last quarter |
| --- | --- | --- | --- | --- |
| γ = 2 | hump-shaped | 3.1% | 13.0% | 10.1% |
| γ = 5 | hump-shaped | 5.8% | 13.8% | 6.1% |
| γ = 10 | **front-loaded** | 15.8% | 11.7% | 1.3% |

At moderate risk aversion it's a **hump** — save least when young, most in
peak-earning years, taper into retirement. That's consumption smoothing: a
25-year-old sits at the bottom of a hump-shaped income profile, so taking a
further tenth away is expensive precisely because there's so little of it.
This is the "save more later" pattern, and the model produces it unprompted.

At γ=10 it **inverts** to front-loaded. Two motives compete and risk aversion
picks the winner: smoothing wants saving where income is highest (mid-career),
precaution wants a buffer built early to insure the whole remaining career
against bad income draws. The model can't settle which kind of investor you
are — but both beat a flat rate, and getting the shape right is worth more the
more risk-averse you are.

**Conditioning — your own position beats the market's direction.** Layered on
top of the solved shape, and scored against a constant rate matched on the
realised career average:

| Rule | Signal | Value |
| --- | --- | --- |
| Solved shape | age only | +0.6% |
| **On-track** | wealth vs an age-appropriate target | **+3.0%** |
| Return-responsive | last year's market return | +0.9% |

Saving more when behind an age-appropriate wealth-to-income target is worth
~5× the shape. Saving more after a bad market year is worth well under a third as much —
and pushed harder, turns negative. A bad market year is a poor proxy for being
behind; wealth relative to an age target is the sufficient statistic. (Sign
check: saving *less* when behind costs −4.2%, which is the reassurance that
the machinery measures what it claims.)

**The savings and retirement gains don't add.** Conditioning the retirement
date is worth ~3.1%; conditioning the savings rate ~3.0%. Doing **both** is
worth less than savings conditioning alone — they read the same underlying
signal (am I ahead or behind), so stacking them over-corrects.

## Taking the savings signal apart

`docs/11` is the stress test of the section above. That +3.0% rests on five
choices that were made once and never varied — how the shortfall is measured,
which target it is measured against, whether the response is symmetric, how
far the contribution may move, and which years it runs in. 195 rule
evaluations at 20,000 paths, all scored against a constant rate matched on
each rule's own realised average, and all reported *net of* the 0.6% the
solved age profile already earns. (The funded-ratio rule scores +4.4% here
against +3.0% in `docs/10` because `docs/11` tunes the functional form and the
coefficient rather than fixing both in advance — the same rule, better set up.)

**The best signal is not the portfolio.** Eight candidates, one sensitivity
grid, same scoring:

| Signal | Kind | Value |
| --- | --- | --- |
| **Income vs its expected path** | pay cheque | **+6.4%** |
| Funded ratio (wealth vs age target) | stock | +4.4% |
| Raw balance (wealth ÷ income) | stock | +1.5% |
| Investment gain (balance vs contributions) | stock | +0.4% |
| Trailing 5- and 10-year return | flow | +0.03% |
| Last year's return | flow | +0.01% |

Saving more in years the pay cheque runs above its expected path beats every
balance rule tested. An income shock is observed *before* it is spent, so
acting on it costs almost no utility; acting on a portfolio shortfall means
cutting consumption that was already planned. Every **flow** signal is worth
essentially nothing — at 0.01–0.03% the ordering among them is not meaningful
and shouldn't be read as one.

Layered together, income and the funded ratio reach **+6.9%**, against +6.4%
for the better one alone and +10.9% if they were additive. They overlap
(both weak income and weak markets show up as a balance behind target) but not
completely.

**A target with no age content is worse than no rule at all.**

| Target | Value | Career average rate |
| --- | --- | --- |
| Model's own median path | +4.4% | 9.7% |
| Published "N× salary by age X" ladder | +1.8% | 10.3% |
| Flat 8× income at every age | **−7.6%** | 15.7% |

The ladder captures 41% — useful, not a substitute. The flat multiple leaves a
28-year-old permanently "behind", drives the career average to 15.7%, and the
matched comparison charges for the extra saving. Rising with age is necessary;
*how* it rises still matters.

**There is no cheap corner on feasibility.** Confining the contribution to
±3 points of income keeps 26% of the value, ±5 points 51%, ±10 points 92%.
Up to about ±10 the value is close to proportional to the flexibility given.
A household with a couple of points of slack gets roughly its share, not most
of the benefit. The implementable guardrail version — move 5% of income once
you are more than 10% off target — keeps 58% and needs no arithmetic.

**It is not left-tail insurance.** The gain is positive almost everywhere and
*largest in the middle* (+7.2% at the median against +6.0% at the 10th
percentile and +6.7% at the 90th, turning negative only at the 99th). The
bottom of the distribution is where the rule has least to work with: a path
behind because labour income collapsed cannot save its way out.

**It is worth most to the investor whose plan is failing**, not to the one
taking the most risk. Across four strategies the value tracks the ruin
probability exactly (rank correlation +1.00): bills-only, ruining 58% of the
time, gets +5.1%; all-equity, ruining 15%, gets +4.4%. That runs *opposite* to
portfolio volatility.

**And it is a risk product.** Every risk aversion picks the same coefficient;
what changes is the price — +2.0% at γ=2, +4.4% at γ=5, +5.4% at γ=10
(each netted against its own no-conditioning baseline). Two
further findings: catching up when behind (+2.5%) and easing off when ahead
(+2.6%) are worth the same and are strongly sub-additive (+4.4% together, not
+5.1%); and per year of career the value rises from 8.4bp in a saver's
thirties to ~14.5bp after 40, faster than the rule's activity does.

## When you retire matters more than what you hold

`docs/09` makes the retirement date a **decision** rather than a birthday:
retire once wealth reaches a multiple of income, inside an age window. That is
what people actually do, and it is testable.

**Conditioning the retirement date on the portfolio is worth ~3.1% of
certainty equivalent consumption** — measured against a fixed date *matched on
the same mean retirement age*, so it isolates the value of responding to
markets from the value of simply retiring earlier. It is stable across trigger
multiples (3.0–3.1%). For scale: currency hedging is worth **nothing at all**
on this panel — the optimal hedge ratio is zero at every cost tested — and
solving the full 68-dimensional glide path buys 6.9% over the balanced
all-equity portfolio. So conditioning the date is worth about half of the
hardest optimisation in this project, and unlike that one it costs nothing to
implement.

The rule isn't "retire later" or "retire earlier". Median retirement age is
63, but the 10th and 90th percentiles are 55 and 70: *retire when the market
lets you, keep working when it doesn't.*

**The retirement-date lottery.** Sorting paths by the real return over the
decade straddling their own retirement date:

| Decile of that decade | Median retirement consumption | Probability of ruin |
| --- | --- | --- |
| Worst | 1.14 | 37.9% |
| Median | 1.55 | 11.0% |
| Best | 1.83 | 2.3% |

Ten years out of sixty-eight — under 15% of the investing life — account for
about **37%** of the explanatory power of the entire return record
(R² 0.089 against 0.242). That is sequence-of-returns risk, measured.

**Do people retire into bad markets?** Half the folk claim holds. People on a
wealth trigger really do retire after good runs (correlation of retirement age
with the prior 5-year return: -0.35; early retirees saw 12.5%/yr run-ups
against 6.7% for late ones). But they were not punished: subsequent
returns were 6.0% against 5.3%, and the run-up/subsequent correlation
is 0.008. **This result is weaker than it looks** — a block bootstrap only
contains the mean reversion that fits inside a block, so it is close to unable
to produce the effect being tested. A valuation-conditional bootstrap is what
would settle it.

### A model artefact worth reading about

The first run said flexible retirement was worth +4.6%, and that retiring at
66 was *worse* than 63 despite higher retirement consumption, lower ruin and a
bigger bequest. That combination doesn't make sense, and chasing it down found
a real asymmetry: **retirement had a consumption floor (the progressive social
security schedule) and working life had none.** At γ=5 the certainty
equivalent is dominated by exactly that tail, so the model was rewarding early
retirement for reaching the safety net sooner. Adding a symmetric
working-income floor cut the apparent premium by ~60%. `docs/09` section 2
documents it in full.

## Spending rules

`docs/06` holds the portfolio fixed and compares eight families of withdrawal
policy, each **at its own optimal rate** — comparing them all at 4% would not
be a fair test, since a 4% constant-real withdrawal and a 4% share-of-portfolio
withdrawal are different amounts of money in every year but the first.

| Rule | CEC (γ=5) | Median spending | Worst spending cut | Median bequest |
| --- | --- | --- | --- | --- |
| Amortisation, 4% assumed return | 1.168 | 2.06 | 41% | 0.0 |
| Actuarial (Gompertz life expectancy) | 1.164 | 2.00 | 45% | 3.9 |
| Constant % of portfolio (7%) | 1.123 | 1.80 | 48% | 12.4 |
| Life expectancy / RMD | 1.120 | 2.10 | 32% | 0.0 |
| Guyton-Klinger guardrails (6.5%) | 1.107 | 1.83 | 31% | 12.1 |
| Vanguard dynamic (5.5%) | 1.070 | 1.67 | 19% | 21.2 |
| **Constant real, "4% rule" (4.5%)** | **1.003** | **1.51** | **0%** | **28.4** |

**The constant-real rule ranks last of every family tested** — a 16% CEC gap to
the winner, more than twice the entire all-equity-versus-glide-path gap. It
buys perfectly smooth spending by permanently forgoing consumption: at its own
optimal rate it dies with 28× initial annual income unspent.

It is also the worst rule in the **left tail**. Plotting the 10th-percentile
spending path by age, the constant-real rule is the lowest line from about age
78 onward — smooth right up to the point the portfolio is gone, then social
security for the rest of the plan. Rules that cut earlier and by less still
have a portfolio at 90.

**What this turns on:** how much you value the bequest. Horizon-based rules
spend the portfolio to zero by design; fixed-amount rules die with most of it.
`docs/06` sweeps the bequest weight from 0 to 10 and the winner does change —
from the amortisation rule to the actuarial rule — but the constant-real rule
loses at *every* weight tested, because the variable rules leave bequests too
while also spending more.

## Quick start

```bash
pip install numpy pandas scipy matplotlib pyyaml openpyxl pytest

python main.py --quick      # ~1 min smoke run at reduced N
python main.py              # ~1 h full run: N = 100,000 plus sweeps and searches
python -m pytest tests/ -q  # 730 tests
```

Selected steps and alternative configurations:

```bash
python main.py --steps 1 2          # panel + bootstrap only
python main.py --steps 1 5          # panel + sensitivity sweeps only
python main.py --steps 1 6          # panel + spending-rule comparison only
python main.py --steps 1 7          # panel + glide-path optimisation only
python main.py --steps 1 8          # panel + currency-hedging sweep only
python main.py --steps 1 9          # panel + retirement-timing analysis only
python main.py --steps 1 10         # panel + savings-rate analysis only
python main.py --steps 11           # the accumulation-signal deep dive
python main.py --steps 12 13        # the full allocation solve and leverage
python main.py --steps 14           # the data provenance audit
python main.py --steps 15           # the starting-valuation study
python main.py --steps 16           # housing as a fifth asset
python main.py --steps 17           # the mortgage on the housing sleeve
python main.py --steps 18           # five international-sleeve weighting schemes
python main.py --steps 19           # delete-one-country and sub-period robustness
python main.py --steps 20           # fees, and the differential that would undo it
python main.py --steps 21           # the realised cohorts, with no bootstrap
python main.py --steps 22           # solved schedules on data they did not see
python main.py --steps 23           # labour income correlated with the market
python main.py --steps 24           # death at a random age
python main.py --steps 25           # the Australian Age Pension instead of the US one
python main.py --steps 12 26        # what the solved schedule costs to trade
python main.py --steps 27           # recent inflation as a state variable
python main.py --steps 28           # foreign dividend withholding tax
python main.py --steps 29           # sequence-of-returns risk, isolated
python main.py --steps 30           # dividend imputation and the tax wedge
python main.py --config other.yaml  # a different parameterisation
```

## Layout

```
├── docs/                 # generated analysis documents (30 files)
├── data/
│   ├── raw/              # primary source files, unmodified
│   ├── processed/        # standardised real return panels (.csv and .npz)
│   └── cache/
├── src/
│   ├── data_loader.py    # panel ingestion, real-return conversion, diagnostics
│   ├── bootstrap.py      # cross-country joint block bootstrap + validation
│   ├── lifecycle.py      # accumulation/decumulation simulator
│   ├── utility.py        # CRRA, Epstein-Zin, shortfall metrics
│   ├── sensitivity.py    # common-random-number parameter sweeps
│   ├── spending.py       # pluggable retirement withdrawal rules
│   ├── glidepath.py      # batched evaluator + glide-path optimisers
│   ├── hedging.py        # currency-hedged leg, break-even cost
│   ├── retirement.py     # path-dependent retirement + saving engine
│   ├── saving.py         # savings-rate rules and shape optimiser
│   ├── accumulation.py   # response forms, signal horse race, feasibility bands
│   ├── allocation.py     # the four-asset simplex solved at every age
│   ├── leverage.py       # levered evaluator and the cost-of-credit sweep
│   ├── observed.py       # series recovered from published rates, not generated
│   ├── provenance.py     # what is observed, what is simulated, and does it matter
│   ├── valuation.py      # the starting dividend yield, with no look-ahead
│   ├── housing.py        # housing de-smoothed, and priced by its holding cost
│   ├── mortgage.py       # the loan-to-value decision, solved by age
│   ├── sleeve.py         # five international-sleeve weighting schemes
│   ├── panel_robustness.py  # delete-one-country, jackknife, sub-periods
│   ├── fees.py           # expense ratios, and the break-even differential
│   ├── plots.py          # publication-quality figures
│   └── report.py         # Markdown report generation
├── tests/                # 627 unit + integration tests
├── results/
│   ├── figures/          # 42 PNGs
│   └── tables/           # 110+ CSVs
├── config.yaml           # every tunable parameter
└── main.py               # entry point
```

`src/report.py`, `src/sensitivity.py`, `src/spending.py`, `src/glidepath.py`,
`src/hedging.py`, `src/retirement.py` and `src/saving.py` are additions to the structure specified in the brief:
keeping Markdown generation, the sweep engine, the withdrawal policies and the
optimiser out of `main.py` leaves the entry point readable as a seven-step
pipeline.

## Data

| Source | Used for | Licence |
| --- | --- | --- |
| [Jordà-Schularick-Taylor Macrohistory Database](https://www.macrohistory.net/database/) (release carrying the Jordà-Knoll-Kuvshinov-Schularick-Taylor *Rate of Return on Everything* series) | nominal equity/bond/bill total returns, CPI, USD exchange rates, 16 countries × 1870-2020 | CC BY-NC-SA 4.0 |
| [Clio Infra](https://clio-infra.eu/) | audit only: long bond yields and CPI for the non-panel countries reported in `docs/14` s8 | CC BY 4.0 |

Both were obtained from the openly licensed
[`unbalancedparentheses/forex-centuries`](https://github.com/unbalancedparentheses/forex-centuries)
mirror and are stored verbatim in `data/raw/`.

### An honest note on the panel

The paper uses a **proprietary**, hand-collected, monthly 38-country database
built from Global Financial Data and Dimson-Marsh-Staunton sources. It is not
redistributable and could not be used here.

This replication covers the **16 countries** for which openly licensed,
recorded equity, bond and bill total returns exist. It does not reach 38, and
it does not pretend to: the other 22 markets have no return series in any
source reachable from here, and filling them with model draws -- which an
earlier version did -- made the cross-section look twice as broad while adding
no evidence a reader could check.

Provenance is tracked per cell (`Panel.observed_mask`) and the tier label is
*derived* from those records (`data_loader.derive_tiers`), so it cannot drift
from the data it describes. Every country is Tier A. `provenance.generated_cells`
reports any cell that is available but not observed, and `main.py` logs an
error if it ever finds one; `build_panel` refuses the old `dev38` panel name
with an explanation rather than aliasing it, so a stale config cannot bring the
generated block back silently.

Section 7 of `docs/04` lists this and every other limitation.

## Design notes

**The bootstrap draws `(country, calendar window)` pairs, not independent
series.** Because the domestic and international legs are pre-joined in the
panel at the same calendar year, one draw carries the contemporaneous
cross-asset *and* cross-country covariance. Across the four asset return
series, the largest bootstrap-versus-panel correlation discrepancy is 0.03.
(Correlations involving the inflation series drift further — up to 0.12 —
because panel inflation has an excess kurtosis near 1,900 from a handful of
hyperinflation years. The engine works in real terms, so this does not touch
the lifecycle results; `docs/02` section 4.3 spells it out.)

**Blocks never span a data gap.** Market closures (Germany 1944-49, France
1915-21, Spain 1936-40) are holes, not interpolations, and a block is
admissible only if the domestic country has an unbroken run covering the whole
window.

**The domestic-country draw is weighted by usable history** in the headline
specification. Uniform weighting would give a 25-year synthetic history the
same weight as the UK's 131-year record, and would tilt the result toward
equities — the direction the finding runs — making it an artefact of the
weighting. Uniform weighting is reported as a robustness check.

**Utility is evaluated over retirement consumption plus the bequest.** With a
fixed savings rate, working-life consumption is identical across strategies by
construction; including it adds a large strategy-invariant term that flattens
every certainty equivalent toward a common value.

**Shortfall statistics use a strategy-invariant target** — a fixed replacement
rate on the investor's own career-average earnings. Measuring each strategy
against its own median makes every strategy look identical.

**Ruin means running out with years left to fund**, not "could not afford the
desired withdrawal". The latter misclassifies horizon-based spending rules,
which deliberately spend the last of the portfolio in the final year.

**The glide-path optimiser is coordinate ascent, deliberately.** Under common
random numbers the objective is a *deterministic* function of the weights, so
a grid search over one coordinate is exact for that coordinate and each sweep
is monotone. No gradients, no step size, no noise to average out. What it
cannot do is escape a local optimum, which is why the restart check exists.

**Sensitivity sweeps use common random numbers.** One set of bootstrap paths
and income shocks is drawn once and reused at every sweep point, so a
difference between two settings is the parameter's effect rather than Monte
Carlo noise. Preference sweeps re-evaluate cached consumption paths instead of
re-simulating, since γ, ψ and the bequest weight enter only the aggregator.

**Every figure is drawn at the width it is printed at** — 6.4 inches, the
paper's text column. A figure authored twice that wide arrives on the page
shrunk by half and takes its 8 pt tick labels down to 4 pt with it, which is
how the labels became unreadable in the first place. A row of three or four
panels therefore wraps into a grid rather than running off the column, and
`tests/test_plots.py` fails if any figure sets its own width.

---

### Repository note

An earlier, unrelated project on this repository -- image recognition on stock
charts -- remains on the `master` branch under `Up/`, `Down/`, `Validation/`
and `StockCharts.tgz`. None of it is used here, and none of it is present on
this branch.
