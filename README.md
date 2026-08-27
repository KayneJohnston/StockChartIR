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
| 100% International Equity | 1.451 | 1.043 | 0.740 | 12.1% | 56.9 |
| **50/50 Domestic/International Equity** | **1.362** | **1.002** | **0.728** | **15.3%** | **37.5** |
| Target-Date Fund (glide path) | 1.212 | 0.934 | 0.700 | 24.0% | 9.5 |
| 60/40 Domestic Equity / Domestic Bonds | 1.120 | 0.873 | 0.673 | 25.7% | 10.0 |
| 100% Domestic Equity | 1.166 | 0.864 | 0.658 | 29.1% | 12.8 |
| 100% Bills (cash) | 0.844 | 0.736 | 0.616 | 61.7% | 0.0 |

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

All five are **generated** by `main.py` from live pipeline objects -- edit
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
| 100% International Equity | 2.82% | 12.2% |
| 50/50 Domestic/International Equity | 2.72% | 15.3% |
| Target-Date Fund (glide path) | 2.12% | 24.1% |
| 60/40 | 1.73% | 25.8% |
| 100% Domestic Equity | 1.38% | 29.2% |
| 100% Bills | 1.21% | 61.9% |

The 4% rule was calibrated on US history — exactly the hindsight the paper
removes, and removing it costs more than a percentage point of retirement
income.

## Quick start

```bash
pip install numpy pandas scipy matplotlib pyyaml openpyxl pytest

python main.py --quick      # ~20 s smoke run at reduced N
python main.py              # ~5 min full run: N = 100,000 plus sensitivity
python -m pytest tests/ -q  # 162 tests
```

Selected steps and alternative configurations:

```bash
python main.py --steps 1 2          # panel + bootstrap only
python main.py --steps 1 5          # panel + sensitivity sweeps only
python main.py --config other.yaml  # a different parameterisation
```

## Layout

```
├── docs/                 # generated analysis documents (5 files)
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
│   ├── plots.py          # publication-quality figures
│   └── report.py         # Markdown report generation
├── tests/                # 162 unit + integration tests
├── results/
│   ├── figures/          # 16 PNGs
│   └── tables/           # 40+ CSVs
├── config.yaml           # every tunable parameter
└── main.py               # entry point
```

`src/report.py` and `src/sensitivity.py` are additions to the structure
specified in the brief: keeping Markdown generation and the sweep engine out
of `main.py` leaves the entry point readable as a five-step pipeline.

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
