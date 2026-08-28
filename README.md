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
a 38-country developed-market panel, 1890-2020. Certainty equivalent
consumption is in multiples of initial annual real income; ruin is exhausting
financial wealth before age 93 under a 4% real withdrawal rule.

| Strategy | CEC (γ=2) | CEC (γ=5) | CEC (γ=10) | P(ruin) | Median bequest |
| --- | --- | --- | --- | --- | --- |
| 100% International Equity | 1.451 | 1.043 | 0.740 | 11.4% | 56.9 |
| **50/50 Domestic/International Equity** | **1.362** | **1.002** | **0.728** | **14.4%** | **37.5** |
| Target-Date Fund (glide path) | 1.212 | 0.934 | 0.700 | 22.3% | 9.5 |
| 60/40 Domestic Equity / Domestic Bonds | 1.120 | 0.873 | 0.673 | 24.3% | 10.0 |
| 100% Domestic Equity | 1.166 | 0.864 | 0.658 | 28.0% | 12.8 |
| 100% Bills (cash) | 0.844 | 0.736 | 0.616 | 57.7% | 0.0 |

The 50/50 all-equity portfolio **strictly dominates** the target-date fund and
60/40 on all 21 reported criteria (20 wins, 1 tie, 0 losses), across every
risk-aversion level and both Epstein-Zin IES values tested. The ordering
survives on the fully empirical 16-country panel, under per-block country
resampling, and under uniform country weighting.

See [`docs/04_replicated_results_and_tables.md`](docs/04_replicated_results_and_tables.md)
for the full analysis and the caveats.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/01_country_dataset_and_sources.md`](docs/01_country_dataset_and_sources.md) | Data lineage, source mapping, real-return construction, per-country summary statistics, coverage matrices, market-disruption handling |
| [`docs/02_multicountry_block_bootstrap.md`](docs/02_multicountry_block_bootstrap.md) | Bootstrap algorithm, mean/covariance/persistence preservation, block-length sensitivity, tail diagnostics |
| [`docs/03_lifecycle_utility_model.md`](docs/03_lifecycle_utility_model.md) | State transitions, cash-flow rules, drawdown algorithm, CRRA and Epstein-Zin definitions |
| [`docs/04_replicated_results_and_tables.md`](docs/04_replicated_results_and_tables.md) | Headline tables, distributional evidence, robustness, comparison with the paper, limitations |
| [`docs/05_sensitivity_analysis.md`](docs/05_sensitivity_analysis.md) | Sweeps over allocation, preferences, planning and sampling assumptions; safe withdrawal rates; tornado analysis |
| [`docs/06_retirement_spending_rules.md`](docs/06_retirement_spending_rules.md) | Eight families of withdrawal policy compared at each rule's own optimal rate; the bequest-motive pivot |
| [`docs/07_optimal_glide_path.md`](docs/07_optimal_glide_path.md) | Solving for the age-by-asset schedule directly: free-form and parametric optimisation, local-optimum checks |

All seven are **generated** by `main.py` from live pipeline objects -- edit
`src/report.py`, not the Markdown.

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

python main.py --quick      # ~20 s smoke run at reduced N
python main.py              # ~18 min full run: N = 100,000 plus sweeps and searches
python -m pytest tests/ -q  # 235 tests
```

Selected steps and alternative configurations:

```bash
python main.py --steps 1 2          # panel + bootstrap only
python main.py --steps 1 5          # panel + sensitivity sweeps only
python main.py --steps 1 6          # panel + spending-rule comparison only
python main.py --steps 1 7          # panel + glide-path optimisation only
python main.py --config other.yaml  # a different parameterisation
```

## Layout

```
├── docs/                 # generated analysis documents (7 files)
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
│   ├── plots.py          # publication-quality figures
│   └── report.py         # Markdown report generation
├── tests/                # 235 unit + integration tests
├── results/
│   ├── figures/          # 22 PNGs
│   └── tables/           # 40+ CSVs
├── config.yaml           # every tunable parameter
└── main.py               # entry point
```

`src/report.py`, `src/sensitivity.py`, `src/spending.py` and
`src/glidepath.py` are additions to the structure specified in the brief:
keeping Markdown generation, the sweep engine, the withdrawal policies and the
optimiser out of `main.py` leaves the entry point readable as a seven-step
pipeline.

## Data

| Source | Used for | Licence |
| --- | --- | --- |
| [Jordà-Schularick-Taylor Macrohistory Database](https://www.macrohistory.net/database/) (release carrying the Jordà-Knoll-Kuvshinov-Schularick-Taylor *Rate of Return on Everything* series) | nominal equity/bond/bill total returns, CPI, USD exchange rates, 16 countries × 1870-2020 | CC BY-NC-SA 4.0 |
| [Clio Infra](https://clio-infra.eu/) | CPI inflation and long bond yields for the extended country set | CC BY 4.0 |

Both were obtained from the openly licensed
[`unbalancedparentheses/forex-centuries`](https://github.com/unbalancedparentheses/forex-centuries)
mirror and are stored verbatim in `data/raw/`.

### An honest note on the panel

The paper uses a **proprietary**, hand-collected, monthly 38-country database
built from Global Financial Data and Dimson-Marsh-Staunton sources. It is not
redistributable and could not be used here. This replication therefore labels
every country by provenance:

* **Tier A (16 countries)** -- equity, bond, bill and inflation series are all
  empirical, from JST/JKKST. This is where the empirical content lives.
* **Tier B (22 countries)** -- inflation is empirical wherever a source exists;
  equity, bond and bill returns are **calibrated proxies** generated by a
  single-factor model fitted to the Tier-A cross-section, with donor-inherited
  residual covariance and documented market-inception dates.

The entire analysis is reported a second time on the Tier-A-only panel. The
ranking is unchanged and the all-equity advantage is in fact *larger* there, so
the conclusion is not an artefact of the calibrated extension. Section 7 of
`docs/04` lists this and every other limitation.

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

---

### Repository note

This project lives alongside `StockChartIR` (image recognition on stock
charts); the `Up/`, `Down/`, `Validation/` and `StockCharts.tgz` paths belong
to that unrelated work.
