"""Do the solved schedules survive being solved on data they did not see?

Three sections of this paper hand a coordinate-ascent search sixty-eight free
parameters and report what it found. Section #glide solves an equity share at
every age; section #allocation solves the whole four-asset simplex at every
age; section #leverage solves a leverage ratio at every age. Each reports a
gain over the fixed benchmarks, and each of those gains is measured on the
same history the search was given.

That is the standard way to overstate a result. A search with sixty-eight
free parameters and one independent lifetime per market -- the count the
cohort census in section #cohorts arrives at -- can fit a good deal of noise,
and the in-sample gain is an upper bound on what an investor standing at the
start could have had. This module measures the part of it that is real.

**The design.** Split the calendar record at ``cut_year``. Solve on one half;
score the solved schedule on the *other* half, against three references:

``in_sample``
    the same family solved on the test half itself -- the ceiling, what the
    search would find if it had seen the answer;
``transferred``
    the schedule solved on the train half, scored on the test half -- what an
    investor of the period could actually have held;
``benchmark``
    the best of the fixed strategies on the test half -- what they could have
    held without solving anything at all.

The number that matters is ``transferred`` against ``benchmark``. If a solved
schedule cannot beat a constant mix on data it has not seen, its in-sample
gain was a description of the training window and not a rule.

Both directions are run -- solve on the first half and score on the second,
then the reverse -- because a single split is one draw and the two halves of
this panel are not interchangeable: the first contains two world wars and the
second contains the post-war expansion.

**A caveat this module cannot remove.** The two halves are not independent
samples of the same process. If the return-generating process changed between
them -- and the expanding-window evidence in section #panel suggests it
did -- then a schedule that fails to transfer may be reporting a change in the
world rather than an overfitted search. The test is still worth running: it
puts a floor under how much of the reported gain can be believed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import allocation as al
from . import bootstrap as bs
from . import data_loader as dl
from . import glidepath as gp
from . import lifecycle as lc
from . import panel_robustness as pr

LOGGER = logging.getLogger(__name__)

#: The families of solved schedule this module can transfer. Each is a
#: section of the paper that reports an in-sample gain.
FAMILIES: Tuple[str, ...] = ("glide", "simplex")

#: Human names, so a figure or a table never prints a bare key.
LABELS: Dict[str, str] = {
    "glide": "Glide path (equity share by age)",
    "simplex": "Full simplex (four assets by age)",
}


# ---------------------------------------------------------------------------
# The two halves
# ---------------------------------------------------------------------------
def split(panel: dl.Panel, cut_year: int) -> Dict[str, dl.Panel]:
    """The panel masked to each side of ``cut_year``.

    Masking rather than rebuilding, for the reason
    :func:`src.panel_robustness.restrict_years` gives: the sleeve in any year
    stays the sleeve that year had, so a lifetime beginning in 1930 can still
    see 1929 in the *other* markets even when 1929 is outside its own window.
    """
    first, last = int(panel.years[0]), int(panel.years[-1])
    cut = int(cut_year)
    if not first < cut < last:
        raise ValueError(f"cut year {cut} is outside {first}-{last}")
    return {
        f"{first}-{cut}": pr.restrict_years(panel, first, cut),
        f"{cut + 1}-{last}": pr.restrict_years(panel, cut + 1, last),
    }


def _evaluator(cfg: Mapping[str, Any], panel: dl.Panel, spec: lc.LifecycleSpec,
               n_paths: int, seed: int) -> gp.BatchEvaluator:
    """A batched evaluator over one window of history."""
    sampler = bs.from_config(panel, cfg, horizon_years=spec.horizon, seed=seed)
    paths = sampler.sample(n_paths, chunk_size=int(cfg["bootstrap"]["chunk_size"]))
    income = lc.simulate_income(spec, n_paths,
                                rng=np.random.default_rng(int(cfg["run"]["seed"])))
    return gp.BatchEvaluator(paths, spec, income, cfg)


# ---------------------------------------------------------------------------
# Solving one family on one window
# ---------------------------------------------------------------------------
def solve(family: str, evaluator: gp.BatchEvaluator, gamma: float,
          cfg: Mapping[str, Any], label: str = "") -> np.ndarray:
    """The solved ``(H, 4)`` weight schedule for one family on one window."""
    if family == "glide":
        glide_cfg = cfg["glide_path"]
        equity, domestic, _ = gp.optimise_free_form_banded(
            evaluator, gamma,
            [float(v) for v in glide_cfg["equity_grid"]],
            [float(v) for v in glide_cfg["domestic_grid"]],
            bond_share=float(glide_cfg["bond_share"]),
            domestic_band_years=int(glide_cfg["domestic_band_years"]),
            n_sweeps=int(glide_cfg["n_sweeps"]))
        return gp.weights_from_shares(equity, domestic,
                                      float(glide_cfg["bond_share"]))
    if family == "simplex":
        alloc_cfg = cfg["allocation"]
        schedule, _, _ = al.optimise_full_simplex(
            evaluator, gamma,
            coarse_step=float(alloc_cfg["coarse_step"]),
            fine_step=float(alloc_cfg["fine_step"]),
            coarse_sweeps=int(alloc_cfg["coarse_sweeps"]),
            fine_sweeps=int(alloc_cfg["fine_sweeps"]),
            label=label)
        return schedule
    raise ValueError(f"unknown schedule family {family!r}")


def score(evaluator: gp.BatchEvaluator, weights: np.ndarray,
          gamma: float) -> float:
    """Certainty equivalent of one schedule on one window."""
    return float(evaluator.cec(np.asarray(weights)[None, ...], gamma)[0])


def benchmark_scores(evaluator: gp.BatchEvaluator,
                     strategies: Mapping[str, lc.Strategy],
                     gamma: float) -> pd.DataFrame:
    """Every fixed strategy scored on one window, best first."""
    keys = list(strategies)
    tensor = np.stack([strategies[k].weights for k in keys])
    values = evaluator.cec(tensor, gamma)
    return pd.DataFrame({
        "strategy": keys,
        "label": [strategies[k].label for k in keys],
        "cec": [float(v) for v in values],
    }).sort_values("cec", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
def transfer(cfg: Mapping[str, Any], panel: dl.Panel, spec: lc.LifecycleSpec,
             strategies: Mapping[str, lc.Strategy], gamma: float,
             cut_year: int, n_paths: int,
             families: Sequence[str] = FAMILIES,
             seed: int = 4242) -> pd.DataFrame:
    """Solve on each half, score on the other, and record all three references.

    Every schedule is scored on the *same* paths for a given test window, so
    the comparison between a transferred schedule, an in-sample schedule and a
    fixed benchmark is a comparison of schedules and not of draws.
    """
    windows = split(panel, cut_year)
    names = list(windows)
    evaluators = {name: _evaluator(cfg, windows[name], spec, n_paths,
                                   seed + 17 * i)
                  for i, name in enumerate(names)}
    benchmarks = {name: benchmark_scores(evaluators[name], strategies, gamma)
                  for name in names}

    solved: Dict[Tuple[str, str], np.ndarray] = {}
    for family in families:
        for name in names:
            LOGGER.info("solving %s on %s", family, name)
            solved[(family, name)] = solve(family, evaluators[name], gamma,
                                           cfg, label=f"{family} {name} ")

    rows: List[Dict[str, Any]] = []
    for family in families:
        for train, test in ((names[0], names[1]), (names[1], names[0])):
            evaluator = evaluators[test]
            best = benchmarks[test].iloc[0]
            transferred = score(evaluator, solved[(family, train)], gamma)
            in_sample = score(evaluator, solved[(family, test)], gamma)
            trained_home = score(evaluators[train], solved[(family, train)],
                                 gamma)
            home_best = float(benchmarks[train].iloc[0]["cec"])
            rows.append({
                "family": family,
                "label": LABELS.get(family, family),
                "train_window": train,
                "test_window": test,
                "cec_transferred": transferred,
                "cec_in_sample": in_sample,
                "cec_benchmark": float(best["cec"]),
                "benchmark": str(best["strategy"]),
                # What the paper reports today: the gain measured where the
                # search was run.
                "in_sample_gain_pct": (trained_home / home_best - 1.0) * 100.0,
                # What an investor of the period would have got.
                "transfer_gain_pct": (transferred / float(best["cec"]) - 1.0)
                * 100.0,
                # The ceiling: solving with the answer in hand.
                "ceiling_gain_pct": (in_sample / float(best["cec"]) - 1.0)
                * 100.0,
            })
    frame = pd.DataFrame.from_records(rows)
    frame["retained_share"] = np.where(
        frame["in_sample_gain_pct"] > 0.0,
        frame["transfer_gain_pct"] / frame["in_sample_gain_pct"],
        np.nan)
    return frame


def benchmark_table(cfg: Mapping[str, Any], panel: dl.Panel,
                    spec: lc.LifecycleSpec,
                    strategies: Mapping[str, lc.Strategy], gamma: float,
                    cut_year: int, n_paths: int, seed: int = 4242
                    ) -> pd.DataFrame:
    """Every fixed strategy on both halves, so the split itself is visible.

    The solved schedules are not the only thing that can fail to transfer. If
    the *ranking of the fixed strategies* also changes across the split, the
    two windows are not drawn from one world and the transfer test is
    measuring that as much as it is measuring overfitting.
    """
    windows = split(panel, cut_year)
    frames = []
    for i, (name, window) in enumerate(windows.items()):
        evaluator = _evaluator(cfg, window, spec, n_paths, seed + 17 * i)
        block = benchmark_scores(evaluator, strategies, gamma)
        block.insert(0, "window", name)
        block["rank"] = np.arange(1, len(block) + 1)
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def ranking_is_stable(benchmarks: pd.DataFrame) -> Dict[str, Any]:
    """Whether the fixed strategies keep their order across the two halves."""
    windows = list(dict.fromkeys(benchmarks["window"]))
    if len(windows) != 2:
        return {"stable": False, "windows": windows}
    order = {w: list(benchmarks[benchmarks["window"] == w]
                     .sort_values("rank")["strategy"]) for w in windows}
    first, second = (order[w] for w in windows)
    return {
        "stable": bool(first == second),
        "winner_first": first[0] if first else "",
        "winner_second": second[0] if second else "",
        "same_winner": bool(first[:1] == second[:1]),
        "n_positions_moved": int(sum(a != b for a, b in zip(first, second))),
    }


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
def verdict(frame: pd.DataFrame, stability: Mapping[str, Any]
            ) -> Dict[str, Any]:
    """What transferred and what did not, classified rather than asserted."""
    if not len(frame):
        return {"families": 0}
    survives = frame["transfer_gain_pct"] > 0.0
    by_family = frame.groupby("family")["transfer_gain_pct"].mean()
    kept = frame["retained_share"].replace([np.inf, -np.inf], np.nan)
    # Which *direction* transfers is the interesting part when only some do.
    # A schedule learned from a turbulent window and applied to a calm one is
    # a different experiment from the reverse, and if the two disagree that
    # asymmetry says more than the pooled average does.
    by_train = frame.groupby("train_window")["transfer_gain_pct"].mean()
    windows = sorted(by_train.index, key=lambda w: str(w))
    earlier, later = (windows + windows[:1])[:2] if windows else ("", "")
    forward = float(by_train.get(earlier, float("nan")))
    backward = float(by_train.get(later, float("nan")))
    return {
        "earlier_window": str(earlier),
        "later_window": str(later),
        "forward_gain_pct": forward,
        "backward_gain_pct": backward,
        "transfers_forward": bool(forward > 0.0),
        "transfers_backward": bool(backward > 0.0),
        "asymmetric": bool(np.isfinite(forward) and np.isfinite(backward)
                           and (forward > 0.0) != (backward > 0.0)),
        "families": int(frame["family"].nunique()),
        "runs": int(len(frame)),
        "runs_that_beat_the_benchmark": int(survives.sum()),
        "every_run_transfers": bool(survives.all()),
        "no_run_transfers": bool((~survives).all()),
        "families_that_transfer": [str(k) for k, v in by_family.items()
                                   if v > 0.0],
        "families_that_do_not": [str(k) for k, v in by_family.items()
                                 if v <= 0.0],
        "mean_in_sample_gain_pct": float(frame["in_sample_gain_pct"].mean()),
        "mean_transfer_gain_pct": float(frame["transfer_gain_pct"].mean()),
        "mean_ceiling_gain_pct": float(frame["ceiling_gain_pct"].mean()),
        "median_retained_share": float(kept.median()) if kept.notna().any()
        else float("nan"),
        "best_family": str(by_family.idxmax()),
        "best_family_gain_pct": float(by_family.max()),
        "worst_family": str(by_family.idxmin()),
        "worst_family_gain_pct": float(by_family.min()),
        "benchmark_ranking_stable": bool(stability.get("stable", False)),
        "benchmark_same_winner": bool(stability.get("same_winner", False)),
    }
