# A from-scratch re-implementation of Beyond the Status Quo — what held, what didn't, and six extensions

*Upfront: this was built with major help from Claude Code — it wrote most of the implementation under my direction, and the write-up with it. It has not been peer reviewed, submitted anywhere, or checked by anyone but me. Treat it as a carefully-built hobby project with open code, not as a paper. I'm posting it here precisely because this crowd will find the errors.*

I wanted to see whether the ACO result survives contact with a different dataset and an independent implementation. Everything below is reproducible from one config file and one command; the code, the 84-page write-up and every table are open. This is a re-implementation, **not** a replication of their code — I never saw it — so where I differ, the honest reading is usually "different sample" rather than "they got it wrong."

## Design differences worth knowing before you read the numbers

Checked against the 10 July 2025 version rather than my memory of it.

| | BTSQ (July 2025) | This build |
|---|---|---|
| Panel | **39 developed countries**, 1890–2023, ~2,600 country-years | **16 countries**, 1890–2020, 2,010 country-years |
| Source | GFDatabase (proprietary) | Jordà–Schularick–Taylor + *Rate of Return on Everything*, openly licensed |
| Frequency | Monthly | **Annual** |
| Household | **A couple** | A single investor |
| Mortality | Random, SSA conditional survival | **None** — fixed 25 → 63 → 93 |
| Retirement age | 65 | 63 |
| Bootstrap | Stationary block, geometric, mean 120 months | Stationary block, geometric, mean 10 years — *same* |
| Savings rate | 10% above a $15k income floor | 10% |
| Withdrawal | 4% then inflation-adjusted | Same |
| Social Security | SSA formulas incl. SSI | SSA PIA bend points |
| Preferences | CRRA, γ = 3.84, De Nardi–French–Jones (2010) bequest | CRRA (γ = 2/5/10) and Epstein–Zin, De Nardi (2004) bequest |

Two things worth noticing. The **sampler is essentially the same** — stationary block bootstrap, geometric block length, ten-year mean — which is reassuring, because it means differences below are about data and lifecycle assumptions rather than about how returns are drawn. And the **panel is less than half the size**: sixteen markets is what the openly licensed record supports with complete equity, bond and bill total returns. My international leg is a leave-one-out equal weighting across the other 15, all advanced economies that ended the century with functioning capital markets.

I also model an individual with a deterministic horizon where they model a couple with random longevity, and I stop at annual frequency where they work monthly.

The sampler is a calendar-joint stationary block bootstrap — whole (country, window) blocks, mean length 10 years — so cross-asset covariance, persistence and fat tails survive. Gaps (closures, war years) are respected rather than interpolated across, which matters because they cluster exactly where a survivorship-prone sample would quietly drop observations.

## What replicated

At γ = 5 over 100,000 lifetimes, certainty-equivalent consumption:

| Strategy | CEC | P(ruin) | SWR @ 5% ruin |
|---|---|---|---|
| 100% international equity | **1.108** | 9.0% | 3.5% |
| 50/50 domestic/international | 1.048 | 11.7% | 3.2% |
| Target-date glide path | 0.943 | 22.2% | 2.1% |
| 100% domestic equity | 0.882 | 25.4% | 1.6% |
| 60/40 | 0.871 | 24.2% | 1.7% |
| Bills | 0.739 | 56.4% | 1.1% |

All-equity beats the glide path by **11.2%** and 60/40 by **20.3%**, cutting ruin from 22% to 12%. It wins 20 of 21 dominance criteria, ties one, loses none, and the ranking survives 132 settings across the 9 dimensions where a like-for-like comparison against a fixed incumbent is defined — risk aversion, EIS, retirement age, savings rate, withdrawal rate, longevity, bequest weight, block length, social-security design — with **zero reversals**.

Two points I'd underline because they replicated hard:

**The mechanism is international, not equity.** 100% domestic equity *loses* to the glide path — 0.882 against 0.943. The all-equity result is a diversification result wearing an equity costume, and on this panel the domestic leg is doing none of the work.

Solving the glide path directly reproduces the all-equity corner. Freeing all four weights at every age — 204 parameters — adds 0.46% over the best fixed benchmark and still produces no glide. A deviation profile shows only **2 of 68 ages** move the objective by more than a basis point.

![Solved glide path](fig20_optimal_glide_path.png)

## One place I don't replicate them

Their headline is **50/50 domestic/international**, held for life. On my panel that is beaten by **100% international** — 1.108 against 1.048, with ruin at 9.0% against 11.7%. The all-equity conclusion survives; the *split* doesn't.

I'd read that as a panel artefact rather than a correction to them, and the mechanism is visible in the construction. My international leg is a leave-one-out equal weighting across the other fifteen markets, so it is a genuinely diversified sleeve. My domestic leg is a single draw from a sixteen-country set that includes some poor century-long performers — Portugal at 3.2% real, France at 3.6%. With 39 countries and a broader international index, the two legs are far closer in character than they are here, and the case for holding both is correspondingly stronger.

So: their 50/50 is not contradicted by anything I can measure. What my panel says is narrower — that on *these* sixteen markets, the diversified sleeve dominates the concentrated one badly enough that mixing in a domestic bet costs you. Whether that survives at 39 countries I can't test, and it is exactly the sort of thing the larger sample should settle.

**Sustainable withdrawal rates are far below 4%.** 3.2% for all-equity, 2.1% for the glide path, at a 5% ruin tolerance. This one is mine rather than a replication — they *assume* a 4% rule as the baseline withdrawal policy rather than solving for a sustainable rate.

## Where Ayres and Nalebuff turn out to be right — and about which instrument

Two of my results look contradictory until you notice what differs between them.

Levering the **portfolio** is barely worth doing: +5.1% at 1.75× with a *zero* borrowing spread, decaying fast in the spread. At real household margin rates it rounds to nothing, and what it buys comes out of the left tail. Read alone, that's a flat negative on *Lifecycle Investing*.

But put the leverage on **housing** instead and the optimiser produces the Ayres–Nalebuff shape without being asked to. Solving loan-to-value at every age, capped at 80% and priced off the borrower's own real short rate: **75% LVR while working, 49% in retirement**, declining monotonically at every spread I tested. That shape is consistent with their human-capital argument — a working investor has decades of future earnings no lender can reach — though I should be clear I haven't isolated that channel, only observed the schedule it would predict.

It isn't the borrowing rate. Both studies borrow at the same real bill rate plus the same swept spread, and at **identical spreads** the mortgage still wins by five to fifteen times:

| Spread over short rate | Lever the portfolio | Mortgage on housing |
|---|---|---|
| 0% | +5.1% | **+25.6%** |
| 1% | +2.0% | **+14.3%** |
| 2% | +0.4% | **+5.9%** |
| 3% | 0.0% | +1.0% |

It's diversification. Housing's *standalone* risk-adjusted return is slightly **worse** than international equity's — 4.8% real on 14.5% vol after a 2% holding cost, against 8.2% on 22.6%. What it has is a correlation of **+0.09** with the equity sleeve, where the two equity legs correlate at +0.40. Levering the portfolio just scales risk you already own; levering housing spends the borrowing on a near-uncorrelated asset.

So the Ayres–Nalebuff shape survives, but not for the reason their argument turns on. Levering *equity* — their instrument — is worth almost nothing here once credit is priced realistically. The gain appears when leverage buys a diversifier. That the cheapest household leverage happens to be secured against that diversifier is a convenience, not the mechanism.

Magnitudes are more modest than 2:1 — roughly 1.4:1 all-in at a 2% spread. And it's overwhelmingly sensitive to the margin: +25.6% at a zero spread, nothing by 4%. Your mortgage rate, not the return on housing, decides this.

It is also the mirror image of the glide path this whole literature argues about: **what should decline with age is the borrowing, not the equity.**

![Mortgage](fig42_mortgage.png)

## What else came out differently

**Currency hedging loses at every ratio, even free.** −0.14% at a 25% hedge, −4.24% fully hedged, before a basis point of cost. No break-even exists. Hedging lowers the foreign sleeve's standalone volatility up to a half hedge but raises its correlation with the home market over the same range.

![Currency hedging](fig23_currency_hedging.png)

**Timing beats allocation.** The decade around the retirement date accounts for ~35% of the explanatory power of the entire 68-year return path.

**The accumulation side has more room than the allocation side.** Conditioning the savings rate on being ahead of or behind an age-appropriate target beats every allocation refinement combined — and the best state variable turns out not to be the portfolio balance but **the investor's own pay cheque relative to its expected path** (+7.1% vs +5.1% for the funded ratio).

## The other extensions

**Starting valuation, with no look-ahead.** Conditioning on the trailing dividend yield an investor could actually observe. The subtlety is the *boundaries* — pooled terciles let a 1910 lifetime be ranked against a threshold that knows about 2020. Computing them recursively, from only the country-years before each start, reclassifies **32% of lifetimes**. The ranking survives all three buckets (11.3 / 11.5 / 11.7%) but the level moves: lifetimes begun expensive reach retirement with 9% less and run out 2.3pp more often. Valuation tells you what to expect, not what to hold.

![Starting valuation](fig40_starting_valuation.png)

**Housing as a fifth asset.** De-smoothed (Geltner), which lifts pooled volatility from 10.1% to 14.5% with the mean unchanged. At zero holding cost it takes 50% of the portfolio (+13.2%); it drops out entirely by **4.6% annual holding cost**. Chambers, Spaenjers & Steiner (RFS 2021) put real operating costs at about a third of gross yield, so the answer sits inside the range where the cost assumption decides it. Bonds and bills never take a cent — housing displaces equity.

![Housing](fig41_housing_cost_sweep.png)

Also: eight spending rules compared on their own optimised rates, and endogenous retirement timing (retire on wealth, not birthday — ~3% against a matched mean age).

## Caveats

Sixteen countries, all survivors at the system level. No taxes, fees, mortality risk or behavioural constraints. The mortgage rebalances annually, which no real mortgage does. Housing is a national index, not a house. Several results changed sign or magnitude once every policy was scored against a *matched* baseline differing in one dimension, every optimiser run under common random numbers, and every solved schedule put through a deviation profile before its shape was described.

Repo and full PDF: **github.com/KayneJohnston/StockChartIR**

Happy to be told where this is wrong.
