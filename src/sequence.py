"""How much of a lifetime's outcome is the order the returns arrived in?

Section #retirement establishes that the decade around a person's retirement
date explains more of their outcome than the allocation question does, and
Section #inflation shows that reading a state variable at the retirement date
rather than the birth date turns a null into an eight-point effect. Both are
symptoms of the same thing, and neither measures it: **sequence-of-returns
risk** -- the fact that when a return arrives matters, not only what it was.

This module measures it directly, by the only experiment that isolates it.
Take a simulated lifetime, keep its sixty-eight annual returns exactly as they
are, and shuffle the order. Same multiset, same mean, same everything a
return distribution can describe. Anything that changes is sequence.

**Why anything changes at all.** For a lump sum it would not: the product of
gross returns is commutative, so a buy-and-hold investor with no cash flows
ends at the same wealth whatever the order. Sequence risk exists *only*
because of flows. A contribution made before a crash buys more; a withdrawal
made after one sells more. That is why the effect is expected to concentrate
in decumulation, where the flows run outward and cannot be paused, and it is
why :func:`permutation` can restrict the shuffle to one phase at a time.

**The decomposition.** Each original path is a bag of returns. Reordering it
makes the outcome a random variable, so the total variance of outcomes splits
exactly:

    Var(outcome) = E[ Var over orderings | bag ] + Var[ E over orderings | bag ]
                 = sequence risk            + return-level risk

The first term is what a saver could in principle diversify away by not having
cash flows; the second is the risk of having drawn a bad bag in the first
place. Estimating it needs several orderings of the *same* path, which is what
``n_reps`` is for. The ``none`` phase, which shuffles nothing, must reproduce
every lifetime bit for bit and so must return a sequence share of zero -- and
does, which is the check that the machinery is measuring what it claims. It is
tested at a tolerance rather than at equality only because the variance of
identical floats is not exactly zero.

**What is permuted.** Every series of a year moves together: returns,
inflation and the calendar metadata share one permutation per path. Shuffling
the assets independently would break the cross-asset covariance the block
bootstrap exists to preserve, and would measure something nobody has.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Which years may change places.
#:
#: ``none``
#:     Nothing moves. The control, and the check: every lifetime must come
#:     back bit for bit, so its sequence share must be zero.
#: ``accumulation``
#:     The working years only. Isolates the risk of contributing at the wrong
#:     prices, which shows up as dispersion in wealth at retirement.
#: ``retirement``
#:     The retired years only. Isolates the risk of withdrawing at the wrong
#:     prices -- the one a retiree cannot wait out.
#: ``both``
#:     The whole lifetime, unrestricted, so a year lived at eighty may land at
#:     twenty-six. The complete ordering effect, and a superset of the two
#:     phase-restricted shuffles.
PHASES: Tuple[str, ...] = ("none", "accumulation", "retirement", "both")


def phase_bounds(phase: str, spec: Any) -> Tuple[int, int]:
    """The half-open range of years a phase is allowed to shuffle within."""
    if phase == "none":
        return (0, 0)
    if phase == "accumulation":
        return (0, int(spec.n_working))
    if phase == "retirement":
        return (int(spec.n_working), int(spec.horizon))
    if phase == "both":
        return (0, int(spec.horizon))
    raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")


def permutation(n_paths: int, horizon: int, phase: str, spec: Any,
                rng: np.random.Generator) -> np.ndarray:
    """``(n_paths, horizon)`` column indices, one independent shuffle per path.

    Every path gets its own ordering, because the decomposition needs the
    variation to be *within* a bag of returns rather than across bags.
    """
    order = np.tile(np.arange(int(horizon)), (int(n_paths), 1))
    lo, hi = phase_bounds(phase, spec)
    if hi > lo:
        block = order[:, lo:hi]
        order[:, lo:hi] = rng.permuted(block, axis=1)
    return order


def permute(paths: Any, order: np.ndarray) -> Any:
    """One chunk of paths with every series reordered by the same permutation.

    All eight fields move together. Permuting them independently would
    dismantle the joint draw a block bootstrap is built to preserve: the year
    equity fell would no longer be the year inflation rose.
    """
    order = np.asarray(order)
    rows = np.arange(order.shape[0])[:, None]
    return dataclasses.replace(paths, **{
        field.name: np.asarray(getattr(paths, field.name))[rows, order]
        for field in dataclasses.fields(paths)})


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------
def run(sampler: Any, strategies: Mapping[str, Any], spec: Any,
        cfg: Mapping[str, Any], n_paths: int, chunk_size: int,
        phase: str, n_reps: int, seed: int = 20260904,
        income_seed: int = 12345,
        spending: Any = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-path outcomes under ``n_reps`` orderings of the same lifetimes.

    Four scalars are kept per path per replication rather than the consumption
    matrix behind them: the certainty equivalent is a function of the *mean*
    of per-path lifetime utility, so carrying the utility carries everything
    the comparison needs at a fraction of the memory.

    The sampler and the income draw are re-run from the same seeds on every
    replication, so the bag of returns and the labour income a path faces are
    identical across orderings and the only thing that differs is the order.

    ``spending`` selects the withdrawal rule, and it is not a detail. A rule
    that fixes consumption in real terms insulates the retiree from
    retirement-phase returns entirely -- converting sequence risk into *ruin*
    risk rather than consumption risk -- while a percentage-of-portfolio rule
    passes every return straight through to what they eat. The split between
    accumulation and decumulation ordering therefore depends on the rule, and
    reporting it under one rule alone would be reporting the rule.
    """
    from . import lifecycle as lc
    from . import utility as ut

    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    beta = float(cfg["utility"]["discount_factor"])
    bequest_weight = float(cfg["utility"]["bequest_weight"])
    bequest_on = bool(cfg["utility"]["bequest_enabled"])

    out: Dict[str, Dict[str, List[np.ndarray]]] = {
        key: {"utility": [], "consumption": [], "ruin": [], "wealth": []}
        for key in strategies}
    for rep in range(int(n_reps)):
        rng = np.random.default_rng(int(seed) + rep)
        income_root = np.random.SeedSequence(int(income_seed))
        n_chunks = int(np.ceil(n_paths / chunk_size))
        income_children = iter(income_root.spawn(n_chunks))
        per_rep: Dict[str, Dict[str, List[np.ndarray]]] = {
            key: {"utility": [], "consumption": [], "ruin": [], "wealth": []}
            for key in strategies}
        for chunk in sampler.chunks(n_paths, chunk_size):
            income = lc.simulate_income(
                spec, chunk.n_paths, np.random.default_rng(next(income_children)),
                dom_eq=chunk.dom_eq, intl_eq=chunk.intl_eq)
            shuffled = permute(chunk, permutation(chunk.n_paths, spec.horizon,
                                                  phase, spec, rng))
            for key, outcome in lc.simulate_all(shuffled, strategies, spec,
                                                income, spending).items():
                bundle = ut.bundle_from_outcome(outcome, cfg, spec)
                per_rep[key]["utility"].append(ut.crra_lifetime_utility(
                    bundle, gamma, beta, bequest_weight, bequest_on))
                per_rep[key]["consumption"].append(
                    outcome.consumption[:, spec.retirement_slice].mean(axis=1))
                per_rep[key]["ruin"].append(outcome.ruin.astype(float))
                per_rep[key]["wealth"].append(outcome.wealth_at_retirement)
        for key in strategies:
            for name, blocks in per_rep[key].items():
                out[key][name].append(np.concatenate(blocks))
    return {key: {name: np.stack(cols, axis=1) for name, cols in fields.items()}
            for key, fields in out.items()}


def certainty_equivalent(utility: np.ndarray, spec: Any,
                         cfg: Mapping[str, Any]) -> float:
    """The certainty equivalent implied by pooled per-path lifetime utility.

    The certainty equivalent is the inverse felicity of mean utility per unit
    of discount weight, so pooling every ordering of every path and taking one
    mean gives the certainty equivalent an investor faces when the order is
    itself uncertain -- which is the quantity this section is about.
    """
    from . import utility as ut

    util_cfg = cfg["utility"]
    gamma = float(util_cfg["baseline_risk_aversion"])
    beta = float(util_cfg["discount_factor"])
    window = str(util_cfg.get("consumption_window", "retirement"))
    horizon = spec.n_retired if window == "retirement" else spec.horizon
    weights = ut.discount_weights(
        int(horizon), beta, float(util_cfg["bequest_weight"]),
        bool(util_cfg["bequest_enabled"]))
    return float(ut._inverse_felicity(
        float(np.asarray(utility).mean()) / weights.sum(), gamma))


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------
def decompose(values: np.ndarray) -> Dict[str, Any]:
    """Split outcome variance into ordering and level.

    ``values`` is ``(n_paths, n_reps)``: one row per bag of returns, one
    column per ordering of it. The law of total variance splits the pooled
    variance exactly into the average variance *within* a bag -- which is
    sequence risk, since the bag is held fixed -- and the variance of the bag
    means, which is the risk of having drawn that bag at all.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("values must be (n_paths, n_reps)")
    reps = int(values.shape[1])
    per_path_mean = values.mean(axis=1)
    within = (float(values.var(axis=1, ddof=1).mean()) if reps > 1 else 0.0)
    between = float(per_path_mean.var(ddof=1))
    total = within + between
    return {
        "n_paths": int(values.shape[0]),
        "n_reps": reps,
        "mean": float(values.mean()),
        "sd_total": float(np.sqrt(total)),
        "sd_sequence": float(np.sqrt(within)),
        "sd_level": float(np.sqrt(between)),
        "variance_sequence": within,
        "variance_level": between,
        # The headline: the share of outcome variance that is nothing but the
        # order the same returns arrived in.
        "sequence_share": (within / total) if total > 0 else 0.0,
    }


def summarise(results: Mapping[str, Mapping[str, np.ndarray]], phase: str,
              spec: Any, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """One row per strategy: the decomposition and what it costs."""
    rows: List[Dict[str, Any]] = []
    for key, fields in results.items():
        consumption = decompose(fields["consumption"])
        wealth = decompose(fields["wealth"])
        rows.append({
            "phase": phase,
            "strategy": key,
            "cec": certainty_equivalent(fields["utility"], spec, cfg),
            "mean_retirement_consumption": consumption["mean"],
            "sd_total": consumption["sd_total"],
            "sd_sequence": consumption["sd_sequence"],
            "sd_level": consumption["sd_level"],
            "sequence_share": consumption["sequence_share"],
            "wealth_sequence_share": wealth["sequence_share"],
            "prob_ruin": float(np.asarray(fields["ruin"]).mean()),
            "n_paths": consumption["n_paths"],
            "n_reps": consumption["n_reps"],
        })
    return pd.DataFrame.from_records(rows)


def verdict(frame: pd.DataFrame, strategy: str,
            phases: Sequence[str] = PHASES) -> Dict[str, Any]:
    """What the ordering is worth, classified from the decomposition."""
    block = frame[frame["strategy"] == strategy].set_index("phase")
    present = [p for p in phases if p in block.index]
    if not present:
        return {"measured": False}

    def _get(phase: str, column: str) -> float:
        return (float(block.loc[phase, column]) if phase in block.index
                else float("nan"))

    control = _get("none", "sequence_share")
    found: Dict[str, Any] = {
        "measured": True,
        "strategy": strategy,
        "phases": present,
        # The control has to come back at zero -- to the tolerance a
        # variance of identical floats can reach -- or the machinery is
        # measuring something other than ordering.
        "control_share": control,
        "control_is_clean": bool(np.isfinite(control) and abs(control) < 1e-9),
        "share_accumulation": _get("accumulation", "sequence_share"),
        "share_retirement": _get("retirement", "sequence_share"),
        "share_both": _get("both", "sequence_share"),
        "cec_none": _get("none", "cec"),
        "cec_both": _get("both", "cec"),
        "ruin_none": _get("none", "prob_ruin"),
        "ruin_both": _get("both", "prob_ruin"),
    }
    a, r = found["share_accumulation"], found["share_retirement"]
    found.update({
        "retirement_dominates": bool(np.isfinite(a) and np.isfinite(r)
                                     and r > a),
        "retirement_over_accumulation": (r / a if np.isfinite(a) and a > 0
                                         else float("inf")),
        "cec_cost_of_ordering_pct": (
            (found["cec_both"] / found["cec_none"] - 1.0) * 100.0
            if np.isfinite(found["cec_none"]) and found["cec_none"] else
            float("nan")),
        "ordering_is_most_of_the_risk": bool(
            np.isfinite(found["share_both"]) and found["share_both"] > 0.5),
    })
    return found


def ranking_holds(frame: pd.DataFrame, challenger: str, incumbent: str,
                  ) -> pd.DataFrame:
    """The headline pair's lead under each phase, so ordering can be ruled in.

    A decomposition says how much of the dispersion is ordering. It does not
    say whether ordering changes *which portfolio wins*, and those are
    different questions with potentially different answers.
    """
    rows: List[Dict[str, Any]] = []
    for phase, block in frame.groupby("phase", sort=False):
        indexed = block.set_index("strategy")
        if challenger not in indexed.index or incumbent not in indexed.index:
            continue
        a = float(indexed.loc[challenger, "cec"])
        b = float(indexed.loc[incumbent, "cec"])
        ordered = indexed["cec"].sort_values(ascending=False)
        rows.append({
            "phase": phase,
            "lead_pct": (a / b - 1.0) * 100.0 if b else float("nan"),
            "winner": str(ordered.index[0]),
            "challenger_ruin": float(indexed.loc[challenger, "prob_ruin"]),
            "incumbent_ruin": float(indexed.loc[incumbent, "prob_ruin"]),
        })
    return pd.DataFrame.from_records(rows)


#: Withdrawal rules the decomposition is repeated under, from fully insulated
#: to fully pass-through. The point is not which is best -- `docs/06` settles
#: that -- but that the *location* of sequence risk moves with the rule.
DEFAULT_RULES: Tuple[Tuple[str, str], ...] = (
    ("constant_real", "Fixed real withdrawal"),
    ("vanguard_dynamic", "Dynamic, with guardrails"),
    ("constant_percent", "Fixed percentage of the portfolio"),
)


def rule_comparison(frame: pd.DataFrame, focus: str) -> pd.DataFrame:
    """Where sequence risk sits, rule by rule.

    One row per withdrawal rule: how much of the ordering risk lands in the
    working years and how much in the retired ones. A fixed real rule should
    push almost all of it into accumulation, because it refuses to let
    retirement returns touch consumption until the money runs out; a
    percentage rule should do the opposite.
    """
    rows: List[Dict[str, Any]] = []
    for rule, block in frame[frame["strategy"] == focus].groupby(
            "rule", sort=False):
        indexed = block.set_index("phase")

        def _get(phase: str, column: str) -> float:
            return (float(indexed.loc[phase, column])
                    if phase in indexed.index else float("nan"))

        acc = _get("accumulation", "sequence_share")
        ret = _get("retirement", "sequence_share")
        rows.append({
            "rule": rule,
            "label": str(block["rule_label"].iloc[0]),
            "share_accumulation": acc,
            "share_retirement": ret,
            "share_both": _get("both", "sequence_share"),
            "retirement_over_accumulation": (ret / acc if acc > 0
                                             else float("inf")),
            "cec_none": _get("none", "cec"),
            "cec_both": _get("both", "cec"),
            "cec_cost_pct": ((_get("both", "cec") / _get("none", "cec") - 1.0)
                             * 100.0 if _get("none", "cec") else float("nan")),
            "ruin_none": _get("none", "prob_ruin"),
            "ruin_both": _get("both", "prob_ruin"),
            "ruin_cost_pp": (_get("both", "prob_ruin")
                             - _get("none", "prob_ruin")) * 100.0,
            "consumption_sd_both": _get("both", "sd_total"),
        })
    return pd.DataFrame.from_records(rows)


def rule_verdict(comparison: pd.DataFrame) -> Dict[str, Any]:
    """Whether the location of sequence risk is a property of the rule."""
    if not len(comparison):
        return {"measured": False}
    indexed = comparison.set_index("rule")
    ratios = comparison["retirement_over_accumulation"]
    finite = ratios[np.isfinite(ratios)]

    def _get(rule: str, column: str) -> float:
        return (float(indexed.loc[rule, column]) if rule in indexed.index
                else float("nan"))

    real_ratio = _get("constant_real", "retirement_over_accumulation")
    pct_ratio = _get("constant_percent", "retirement_over_accumulation")
    return {
        "measured": True,
        "rules": int(len(comparison)),
        "min_ratio": float(finite.min()) if len(finite) else float("nan"),
        "max_ratio": float(finite.max()) if len(finite) else float("nan"),
        "fixed_real_ratio": real_ratio,
        "percentage_ratio": pct_ratio,
        # The finding, if it holds: a fixed real rule refuses to let
        # retirement returns touch consumption, so the ordering risk it cannot
        # absorb comes out as ruin instead.
        "rule_relocates_the_risk": bool(
            np.isfinite(real_ratio) and np.isfinite(pct_ratio)
            and pct_ratio > 2.0 * real_ratio),
        "fixed_real_ruin_cost_pp": _get("constant_real", "ruin_cost_pp"),
        "percentage_ruin_cost_pp": _get("constant_percent", "ruin_cost_pp"),
        "fixed_real_trades_ruin_for_smoothness": bool(
            np.isfinite(_get("constant_real", "ruin_cost_pp"))
            and np.isfinite(_get("constant_percent", "ruin_cost_pp"))
            and _get("constant_real", "ruin_cost_pp")
            > _get("constant_percent", "ruin_cost_pp")),
    }
