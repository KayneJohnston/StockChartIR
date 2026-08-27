"""Cross-country joint block bootstrap.

The engine reproduces the sampling scheme that gives Anarkulova, Cederburg
and O'Doherty (2023/2024) their headline result: instead of resampling a
single country's history, blocks are drawn from a *panel* of developed
markets, so a simulated investor's lifetime can look like Japan's, or
Portugal's, or the United States'.

Two design choices matter and both are preserved here.

**Calendar-joint blocks.**  A block is a *(country, calendar-window)* pair.
Because the domestic and the international legs of the panel are indexed by
the same calendar year, drawing a window automatically carries the
contemporaneous cross-asset covariance (equity/bond/bill/inflation) *and*
the cross-country covariance embedded in the international leg.  Nothing is
re-drawn independently, so nothing decorrelates.

**Gap-respecting blocks.**  Market closures (Germany 1944-49, France
1915-21, Spain 1936-40, ...) leave holes in the panel.  A block is only
admissible if the domestic country has an *unbroken* run of observations
covering the whole window, which is the "continuous available sub-periods"
protocol.  Blocks never straddle a war-time closure.

Block lengths follow the stationary bootstrap of Politis and Romano (1994):
``L ~ Geometric(1 / mean_block)``, truncated to the configuration bounds
and to the longest run the drawn country can actually supply.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_loader import CORE_SERIES, Panel

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampling support structures
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class BlockIndex:
    """Pre-computed admissible block starts for every ``(country, length)``.

    ``starts_flat`` concatenates, for each country ``c`` and each length
    ``L``, the sorted calendar indices ``t`` such that
    ``available[t : t + L, c]`` is entirely true.  ``offset`` and ``count``
    address into that flat array, which lets the sampler pick a uniform
    admissible start for millions of ``(c, L)`` pairs with pure fancy
    indexing.
    """

    starts_flat: np.ndarray          # (M,) int32
    offset: np.ndarray               # (C, Lmax + 1) int64
    count: np.ndarray                # (C, Lmax + 1) int64
    max_run: np.ndarray              # (C,) int64  longest unbroken run
    max_length: int

    @property
    def n_countries(self) -> int:
        return int(self.offset.shape[0])


def run_lengths(available: np.ndarray) -> np.ndarray:
    """For each ``(t, c)``, the number of consecutive available years from t.

    ``run_lengths(...)[t, c] >= L`` is exactly the admissibility test for a
    block of length ``L`` starting at ``t`` in country ``c``.
    """
    n_t, n_c = available.shape
    runs = np.zeros((n_t, n_c), dtype=np.int64)
    for t in range(n_t - 1, -1, -1):
        nxt = runs[t + 1] if t + 1 < n_t else np.zeros(n_c, dtype=np.int64)
        runs[t] = np.where(available[t], nxt + 1, 0)
    return runs


def build_block_index(panel: Panel, max_length: int) -> BlockIndex:
    """Enumerate admissible block starts for every country and length."""
    runs = run_lengths(panel.available)
    n_c = panel.n_countries
    max_run = runs.max(axis=0)
    max_length = int(min(max_length, int(max_run.max()) if n_c else 1))
    max_length = max(max_length, 1)

    offset = np.zeros((n_c, max_length + 1), dtype=np.int64)
    count = np.zeros((n_c, max_length + 1), dtype=np.int64)
    chunks: List[np.ndarray] = []
    cursor = 0
    for c in range(n_c):
        for length in range(1, max_length + 1):
            starts = np.flatnonzero(runs[:, c] >= length).astype(np.int32)
            offset[c, length] = cursor
            count[c, length] = starts.size
            cursor += starts.size
            if starts.size:
                chunks.append(starts)
    starts_flat = (np.concatenate(chunks) if chunks
                   else np.zeros(0, dtype=np.int32))
    return BlockIndex(starts_flat=starts_flat, offset=offset, count=count,
                      max_run=max_run, max_length=max_length)


def country_probabilities(panel: Panel, weighting: str) -> np.ndarray:
    """Probability of drawing each country as the investor's domestic market."""
    if weighting == "uniform":
        probs = np.ones(panel.n_countries, dtype=float)
    elif weighting == "history":
        probs = panel.available.sum(axis=0).astype(float)
    else:
        raise ValueError(f"unknown country_weighting {weighting!r}")
    total = probs.sum()
    if total <= 0:  # pragma: no cover - empty panel
        raise ValueError("panel has no usable observations")
    return probs / total


# ---------------------------------------------------------------------------
# Simulated paths
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class BootstrapPaths:
    """One chunk of simulated lifetimes.

    Every return matrix is ``(n_paths, horizon)`` of *real* returns.
    ``domestic_country`` is ``(n_paths, horizon)`` so that the
    ``per_block`` scheme, in which the domestic market changes at block
    boundaries, is representable alongside ``per_lifetime``.
    """

    dom_eq: np.ndarray
    intl_eq: np.ndarray
    bond: np.ndarray
    bill: np.ndarray
    inflation: np.ndarray
    domestic_country: np.ndarray
    calendar_index: np.ndarray
    block_id: np.ndarray

    @property
    def n_paths(self) -> int:
        return int(self.dom_eq.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.dom_eq.shape[1])

    def series(self, key: str) -> np.ndarray:
        if key not in CORE_SERIES:
            raise KeyError(f"unknown series {key!r}")
        return getattr(self, key)

    def concat(self, other: "BootstrapPaths") -> "BootstrapPaths":
        return BootstrapPaths(
            **{
                field.name: np.concatenate(
                    [getattr(self, field.name), getattr(other, field.name)], axis=0
                )
                for field in dataclasses.fields(self)
            }
        )


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------
class MultiCountryBlockBootstrap:
    """Joint block bootstrap over a developed-market return panel.

    Parameters
    ----------
    panel:
        Calendar-aligned real return panel from :mod:`src.data_loader`.
    horizon:
        Lifetime length in years (68 = 38 accumulation + 30 decumulation).
    mean_block:
        Expected block length in years.  ACO use 120 months.
    block_length_distribution:
        ``"geometric"`` gives the stationary bootstrap of Politis-Romano;
        ``"fixed"`` gives a classical overlapping block bootstrap.
    country_draw:
        ``"per_lifetime"`` fixes the domestic market for a whole simulated
        life; ``"per_block"`` redraws it at every block boundary.
    country_weighting:
        ``"history"`` (probability proportional to usable country-years) or
        ``"uniform"``.
    """

    def __init__(
        self,
        panel: Panel,
        horizon: int = 68,
        mean_block: float = 10.0,
        block_length_distribution: str = "geometric",
        min_block: int = 1,
        max_block: int = 40,
        country_draw: str = "per_lifetime",
        country_weighting: str = "history",
        seed: int = 7,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if mean_block <= 0:
            raise ValueError("mean_block must be positive")
        if country_draw not in ("per_lifetime", "per_block"):
            raise ValueError(f"unknown country_draw {country_draw!r}")
        if block_length_distribution not in ("geometric", "fixed"):
            raise ValueError(
                f"unknown block_length_distribution {block_length_distribution!r}"
            )

        self.panel = panel
        self.horizon = int(horizon)
        self.mean_block = float(mean_block)
        self.block_length_distribution = block_length_distribution
        self.min_block = max(1, int(min_block))
        self.max_block = max(self.min_block, int(max_block))
        self.country_draw = country_draw
        self.country_weighting = country_weighting
        self.seed = int(seed)

        self.index = build_block_index(panel, self.max_block)
        self.country_probs = country_probabilities(panel, country_weighting)
        self._data = {key: np.ascontiguousarray(panel.series(key), dtype=np.float64)
                      for key in CORE_SERIES}

    # -- block length -------------------------------------------------------
    def _draw_lengths(self, rng: np.random.Generator, countries: np.ndarray
                      ) -> np.ndarray:
        n = countries.size
        if self.block_length_distribution == "fixed":
            lengths = np.full(n, int(round(self.mean_block)), dtype=np.int64)
        else:
            p = 1.0 / self.mean_block
            lengths = rng.geometric(p, size=n).astype(np.int64)
        lengths = np.clip(lengths, self.min_block, self.max_block)
        # A country cannot supply a block longer than its longest unbroken run.
        ceiling = np.minimum(self.index.max_run[countries], self.index.max_length)
        return np.minimum(lengths, np.maximum(ceiling, 1))

    def _draw_starts(self, rng: np.random.Generator, countries: np.ndarray,
                     lengths: np.ndarray) -> np.ndarray:
        counts = self.index.count[countries, lengths]
        if np.any(counts <= 0):  # pragma: no cover - guarded by _draw_lengths
            raise RuntimeError("drew a block length with no admissible start")
        picks = (rng.random(countries.size) * counts).astype(np.int64)
        picks = np.minimum(picks, counts - 1)
        return self.index.starts_flat[self.index.offset[countries, lengths] + picks]

    def _draw_countries(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(self.panel.n_countries, size=n, p=self.country_probs)

    # -- path construction --------------------------------------------------
    def sample_chunk(self, n_paths: int, rng: np.random.Generator
                     ) -> BootstrapPaths:
        """Draw ``n_paths`` independent lifetimes."""
        horizon = self.horizon
        cal = np.zeros((n_paths, horizon), dtype=np.int32)
        ctry = np.zeros((n_paths, horizon), dtype=np.int32)
        blk = np.zeros((n_paths, horizon), dtype=np.int32)

        lifetime_country = self._draw_countries(rng, n_paths)
        current_country = lifetime_country.copy()
        remaining = np.zeros(n_paths, dtype=np.int64)
        cursor = np.zeros(n_paths, dtype=np.int64)
        block_counter = np.zeros(n_paths, dtype=np.int32)

        for step in range(horizon):
            need = remaining <= 0
            if need.any():
                idx = np.flatnonzero(need)
                if self.country_draw == "per_block":
                    current_country[idx] = self._draw_countries(rng, idx.size)
                lengths = self._draw_lengths(rng, current_country[idx])
                starts = self._draw_starts(rng, current_country[idx], lengths)
                cursor[idx] = starts
                remaining[idx] = lengths
                block_counter[idx] += 1
            cal[:, step] = cursor
            ctry[:, step] = current_country
            blk[:, step] = block_counter
            cursor += 1
            remaining -= 1

        gathered = {key: arr[cal, ctry] for key, arr in self._data.items()}
        if not all(np.isfinite(v).all() for v in gathered.values()):  # pragma: no cover
            raise RuntimeError("bootstrap produced non-finite returns")

        return BootstrapPaths(
            dom_eq=gathered["dom_eq"],
            intl_eq=gathered["intl_eq"],
            bond=gathered["bond"],
            bill=gathered["bill"],
            inflation=gathered["inflation"],
            domestic_country=ctry,
            calendar_index=cal,
            block_id=blk,
        )

    def chunks(self, n_paths: int, chunk_size: int
               ) -> Iterator[BootstrapPaths]:
        """Yield ``BootstrapPaths`` chunks totalling ``n_paths`` lifetimes.

        Each chunk gets its own child generator spawned from the root seed.
        The reproducibility contract is therefore
        ``(seed, n_paths, chunk_size)``: rerunning with all three unchanged
        reproduces the draws exactly, while changing ``chunk_size`` re-cuts
        the random stream and produces a different (equally valid) sample.
        ``chunk_size`` is a configuration parameter for that reason, not a
        free performance knob.
        """
        root = np.random.SeedSequence(self.seed)
        n_chunks = int(np.ceil(n_paths / chunk_size))
        children = root.spawn(n_chunks)
        drawn = 0
        for i, child in enumerate(children):
            size = min(chunk_size, n_paths - drawn)
            if size <= 0:
                break
            yield self.sample_chunk(size, np.random.default_rng(child))
            drawn += size

    def sample(self, n_paths: int, chunk_size: int | None = None
               ) -> BootstrapPaths:
        """Materialise all ``n_paths`` lifetimes in one object."""
        chunk_size = chunk_size or n_paths
        out: BootstrapPaths | None = None
        for chunk in self.chunks(n_paths, chunk_size):
            out = chunk if out is None else out.concat(chunk)
        if out is None:  # pragma: no cover - n_paths <= 0
            raise ValueError("n_paths must be positive")
        return out


def from_config(panel: Panel, cfg: Mapping[str, Any],
                **overrides: Any) -> MultiCountryBlockBootstrap:
    """Build a sampler from the ``bootstrap`` section of the config."""
    boot = dict(cfg["bootstrap"])
    boot.update(overrides)
    return MultiCountryBlockBootstrap(
        panel=panel,
        horizon=int(boot["horizon_years"]),
        mean_block=float(boot["mean_block_years"]),
        block_length_distribution=str(boot["block_length_distribution"]),
        min_block=int(boot["min_block_years"]),
        max_block=int(boot["max_block_years"]),
        country_draw=str(boot["country_draw"]),
        country_weighting=str(boot.get("country_weighting", "history")),
        seed=int(boot["seed"]),
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _empirical_reference(panel: Panel, probs: np.ndarray) -> pd.DataFrame:
    """Population moments the bootstrap is supposed to reproduce."""
    records = []
    for key in CORE_SERIES:
        values = panel.series(key)
        mask = panel.available
        pooled = values[mask]
        per_country_mean = np.array([
            values[mask[:, c], c].mean() if mask[:, c].any() else np.nan
            for c in range(panel.n_countries)
        ])
        per_country_std = np.array([
            values[mask[:, c], c].std(ddof=1) if mask[:, c].sum() > 1 else np.nan
            for c in range(panel.n_countries)
        ])
        records.append({
            "series": key,
            "pooled_mean": float(pooled.mean()),
            "pooled_std": float(pooled.std(ddof=1)),
            "country_weighted_mean": float(np.nansum(probs * per_country_mean)),
            "country_weighted_std": float(np.nansum(probs * per_country_std)),
        })
    return pd.DataFrame.from_records(records)


def _autocorrelation(paths: np.ndarray, max_lag: int = 5) -> Dict[int, float]:
    """Average within-path autocorrelation across simulated lifetimes."""
    out: Dict[int, float] = {}
    demeaned = paths - paths.mean(axis=1, keepdims=True)
    denom = (demeaned * demeaned).sum(axis=1)
    for lag in range(1, max_lag + 1):
        num = (demeaned[:, :-lag] * demeaned[:, lag:]).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.where(denom > 0, num / denom, np.nan)
        out[lag] = float(np.nanmean(rho))
    return out


def _panel_autocorrelation(panel: Panel, key: str, max_lag: int = 5
                           ) -> Dict[int, float]:
    """Average within-country autocorrelation in the source panel."""
    values = panel.series(key)
    out: Dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        rhos = []
        for c in range(panel.n_countries):
            mask = panel.available[:, c]
            series = values[:, c]
            ok = mask[:-lag] & mask[lag:] & np.isfinite(series[:-lag]) \
                & np.isfinite(series[lag:])
            if ok.sum() < 10:
                continue
            a = series[:-lag][ok]
            b = series[lag:][ok]
            a = a - a.mean()
            b = b - b.mean()
            denom = np.sqrt((a * a).sum() * (b * b).sum())
            if denom > 0:
                rhos.append(float((a * b).sum() / denom))
        out[lag] = float(np.mean(rhos)) if rhos else float("nan")
    return out


def diagnose(
    sampler: MultiCountryBlockBootstrap,
    n_paths: int = 20000,
    chunk_size: int = 10000,
    percentiles: Sequence[float] = (1, 5, 10, 25, 50, 75, 90, 95, 99),
) -> Dict[str, pd.DataFrame]:
    """Run the validation battery reported in docs/02.

    Returns a dictionary of frames:

    ``moments``         bootstrap vs. panel means/standard deviations
    ``autocorrelation`` within-path vs. within-country persistence
    ``correlation``     bootstrap cross-asset correlation matrix
    ``correlation_gap`` bootstrap minus panel correlation
    ``terminal``        percentiles of cumulative real growth over the horizon
    ``countries``       realised vs. target domestic-country frequencies
    ``blocks``          realised block-length distribution
    """
    panel = sampler.panel
    accum: Dict[str, List[np.ndarray]] = {k: [] for k in CORE_SERIES}
    country_counts = np.zeros(panel.n_countries, dtype=np.int64)
    block_lengths: List[np.ndarray] = []
    n_seen = 0

    for chunk in sampler.chunks(n_paths, chunk_size):
        for key in CORE_SERIES:
            accum[key].append(chunk.series(key))
        counts = np.bincount(chunk.domestic_country[:, 0],
                             minlength=panel.n_countries)
        country_counts += counts
        # Realised block lengths: run-length encode the block id per path.
        ids = chunk.block_id
        changes = np.diff(ids, axis=1) != 0
        lengths = []
        for row in range(min(ids.shape[0], 2000)):
            edges = np.flatnonzero(changes[row]) + 1
            bounds = np.concatenate([[0], edges, [ids.shape[1]]])
            lengths.append(np.diff(bounds))
        if lengths:
            block_lengths.append(np.concatenate(lengths))
        n_seen += chunk.n_paths

    data = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    reference = _empirical_reference(panel, sampler.country_probs)

    moment_rows = []
    for key in CORE_SERIES:
        arr = data[key]
        ref = reference[reference["series"] == key].iloc[0]
        moment_rows.append({
            "series": key,
            "bootstrap_mean": float(arr.mean()),
            "panel_pooled_mean": float(ref["pooled_mean"]),
            "panel_country_weighted_mean": float(ref["country_weighted_mean"]),
            "mean_gap_bp": float((arr.mean() - ref["pooled_mean"]) * 1e4),
            "bootstrap_std": float(arr.std(ddof=1)),
            "panel_pooled_std": float(ref["pooled_std"]),
            "std_ratio": float(arr.std(ddof=1) / ref["pooled_std"]),
            "bootstrap_skew": float(pd.Series(arr.reshape(-1)).skew()),
            "bootstrap_kurtosis": float(pd.Series(arr.reshape(-1)).kurtosis()),
        })
    moments = pd.DataFrame.from_records(moment_rows)

    ac_rows = []
    for key in CORE_SERIES:
        boot_ac = _autocorrelation(data[key])
        panel_ac = _panel_autocorrelation(panel, key)
        for lag in sorted(boot_ac):
            ac_rows.append({
                "series": key, "lag": lag,
                "bootstrap": boot_ac[lag], "panel": panel_ac.get(lag, np.nan),
                "gap": boot_ac[lag] - panel_ac.get(lag, np.nan),
            })
    autocorrelation = pd.DataFrame.from_records(ac_rows)

    flat = np.column_stack([data[k].reshape(-1) for k in CORE_SERIES])
    boot_corr = pd.DataFrame(np.corrcoef(flat.T),
                             index=list(CORE_SERIES), columns=list(CORE_SERIES))
    panel_stack = panel.stacked().reshape(-1, len(CORE_SERIES))
    ok = np.all(np.isfinite(panel_stack), axis=1)
    panel_corr = pd.DataFrame(np.corrcoef(panel_stack[ok].T),
                              index=list(CORE_SERIES), columns=list(CORE_SERIES))

    terminal_rows = []
    for key in CORE_SERIES:
        cumulative = np.expm1(np.log1p(data[key]).sum(axis=1))
        annualised = np.expm1(np.log1p(data[key]).mean(axis=1))
        row: Dict[str, Any] = {"series": key,
                               "mean_annualised": float(annualised.mean())}
        for q in percentiles:
            row[f"p{q:g}_annualised"] = float(np.percentile(annualised, q))
        row["p1_cumulative"] = float(np.percentile(cumulative, 1))
        row["p50_cumulative"] = float(np.percentile(cumulative, 50))
        row["p99_cumulative"] = float(np.percentile(cumulative, 99))
        terminal_rows.append(row)
    terminal = pd.DataFrame.from_records(terminal_rows)

    countries = pd.DataFrame({
        "iso": list(panel.countries),
        "tier": list(panel.tier),
        "usable_years": panel.available.sum(axis=0),
        "target_probability": sampler.country_probs,
        "realised_frequency": country_counts / max(n_seen, 1),
    })

    all_lengths = (np.concatenate(block_lengths) if block_lengths
                   else np.zeros(0, dtype=int))
    blocks = pd.DataFrame({
        "statistic": ["n_blocks", "mean_length", "median_length",
                      "p90_length", "max_length", "target_mean_length"],
        "value": [
            float(all_lengths.size),
            float(all_lengths.mean()) if all_lengths.size else np.nan,
            float(np.median(all_lengths)) if all_lengths.size else np.nan,
            float(np.percentile(all_lengths, 90)) if all_lengths.size else np.nan,
            float(all_lengths.max()) if all_lengths.size else np.nan,
            float(sampler.mean_block),
        ],
    })

    return {
        "moments": moments,
        "autocorrelation": autocorrelation,
        "correlation": boot_corr,
        "panel_correlation": panel_corr,
        "correlation_gap": boot_corr - panel_corr,
        "terminal": terminal,
        "countries": countries,
        "blocks": blocks,
    }


def block_length_sensitivity(
    panel: Panel,
    cfg: Mapping[str, Any],
    grid: Sequence[float],
    n_paths: int = 20000,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """How the simulated horizon distribution responds to block length.

    Short blocks destroy persistence and shrink the dispersion of 68-year
    outcomes; long blocks preserve it but reduce the effective number of
    independent draws.  The table quantifies both ends.
    """
    rows = []
    for mean_block in grid:
        sampler = from_config(panel, cfg, mean_block_years=float(mean_block))
        eq_ann: List[np.ndarray] = []
        ac1: List[float] = []
        for chunk in sampler.chunks(n_paths, chunk_size):
            eq_ann.append(np.expm1(np.log1p(chunk.dom_eq).mean(axis=1)))
            ac1.append(_autocorrelation(chunk.dom_eq, max_lag=1)[1])
        annualised = np.concatenate(eq_ann)
        rows.append({
            "mean_block_years": float(mean_block),
            "dom_eq_mean_annualised": float(annualised.mean()),
            "dom_eq_sd_annualised": float(annualised.std(ddof=1)),
            "dom_eq_p5_annualised": float(np.percentile(annualised, 5)),
            "dom_eq_p95_annualised": float(np.percentile(annualised, 95)),
            "within_path_ar1": float(np.mean(ac1)),
        })
    return pd.DataFrame.from_records(rows)
