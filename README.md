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

All four are **generated** by `main.py` from live pipeline objects -- edit
`src/report.py`, not the Markdown.

## Quick start

```bash
pip install numpy pandas scipy matplotlib pyyaml openpyxl pytest

python main.py --quick      # ~15 s smoke run at N = 5,000
python main.py              # ~80 s full run at N = 100,000
python -m pytest tests/ -q  # 128 tests
```

Selected steps and alternative configurations:

```bash
python main.py --steps 1 2          # panel + bootstrap only
python main.py --config other.yaml  # a different parameterisation
```

## Layout

```
├── docs/                 # generated analysis documents (4 files)
├── data/
│   ├── raw/              # primary source files, unmodified
│   ├── processed/        # standardised real return panels (.csv and .npz)
│   └── cache/
├── src/
│   ├── data_loader.py    # panel ingestion, real-return conversion, diagnostics
│   ├── bootstrap.py      # cross-country joint block bootstrap + validation
│   ├── lifecycle.py      # accumulation/decumulation simulator
│   ├── utility.py        # CRRA, Epstein-Zin, shortfall metrics
│   ├── plots.py          # publication-quality figures
│   └── report.py         # Markdown report generation
├── tests/                # 128 unit + integration tests
├── results/
│   ├── figures/          # 11 PNGs
│   └── tables/           # 20+ CSVs
├── config.yaml           # every tunable parameter
└── main.py               # entry point
```

`src/report.py` is an addition to the structure specified in the brief: keeping
the ~700 lines of Markdown generation out of `main.py` leaves the entry point
readable as a four-step pipeline.

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

---

### Repository note

This project lives alongside `StockChartIR` (image recognition on stock
charts); the `Up/`, `Down/`, `Validation/` and `StockCharts.tgz` paths belong
to that unrelated work.
