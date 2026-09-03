"""The retirement plan, solved with the portfolio rather than before it.

Every other section of this paper fixes one of these two decisions and solves
the other. Section #glide solves the allocation schedule with the withdrawal
rule pinned to a 4% rule; Section #spending ranks withdrawal rules with the
allocation pinned to a 50/50 portfolio. Each concludes, reasonably, that its
own answer is robust to the thing it held fixed -- and `docs/06` states the
consequence outright: "the spending rule and the asset allocation are close to
separable ... they can be chosen independently".

That claim is true at the resolution it was tested at and false at a finer
one, which is what this section is for.

**Where separability holds.** The *ranking* of six candidate portfolios is
identical under every withdrawal rule, and the ranking of rules is identical
on every portfolio. Nothing in Section #spending is overturned.

**Where it fails.** The *location of the optimum* is not rank information, and
it does move. Under a rule that fixes real spending from wealth on the
retirement date, retirement-phase returns cannot reach consumption until the
money runs out -- the finding of Section #sequence -- so the allocation's only
remaining job is to keep the portfolio alive, and it is chosen to minimise
ruin. Under a rule that spends a percentage of the balance, ruin is impossible
by construction and the allocation is chosen to maximise the balance instead.
Those are different objectives and they have different argmaxes.

**What is solved here.** Two things at once, by alternating maximisation:

``allocation``
    The equity share at every age and the domestic share of it on multi-year
    bands, exactly the free-form schedule of Section #glide.
``plan``
    Which withdrawal rule, at what rate, retiring at what age.

Alternating between them until neither moves is what makes the answer a joint
optimum rather than two separate ones. The gain over the better of the two
one-sided searches is the interaction, and the interaction is the answer to
whether the decisions separate.

**A warning the search itself produces.** One degree of freedom here is not
priced by this model at all. Labour is costless in it -- there is no
disutility of work -- so an extra year of employment is pure gain and the
optimiser will retire as late as the grid permits. That corner is reported
rather than hidden, because a joint search is the cleanest way to find out
which decisions a model can rank and which it merely appears to.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Rules crossed with the allocation grid, from fully insulated to fully
#: pass-through. The order is the mechanism's, not the ranking's: how much of
#: a retirement-phase return the rule allows to reach consumption.
DEFAULT_RULES: Tuple[Tuple[str, str], ...] = (
    ("constant_real", "Fixed real withdrawal"),
    ("vanguard_dynamic", "Dynamic, with guardrails"),
    ("guyton_klinger", "Guyton-Klinger guardrails"),
    ("endowment", "Endowment smoothing"),
    ("constant_percent", "Fixed percentage of the portfolio"),
    ("life_expectancy", "Life expectancy / RMD"),
)

#: Whether a rule can exhaust the portfolio, as a *prior* rather than a
#: finding: a rule that spends a fraction of whatever is left cannot reach
#: zero in finite time, so its ruin probability is zero by arithmetic rather
#: than by prudence, and any comparison that forgets this reads its safety as
#: skill. :func:`ruin_minimum_by_rule` re-derives the same fact from the
#: simulated paths and reports where the two disagree, because the prior is
#: easy to get wrong: a *smoothed* percentage rule spends a fraction of a
#: lagged balance rather than the current one, so it can overshoot a bad year
#: and does deplete, which is why ``endowment`` sits on the True side here.
CAN_DEPLETE: Mapping[str, bool] = {
    "constant_real": True, "vanguard_dynamic": True, "guyton_klinger": True,
    "endowment": True, "constant_percent": False, "life_expectancy": False,
    "gompertz": False, "amortisation": False,
}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Plan:
    """One retirement plan: a rule, the rate it spends at, and when it starts."""

    rule: str
    rate: float | None = None
    retire_age: int = 63

    def build(self) -> Any:
        """The :class:`src.spending.SpendingRule` this plan describes."""
        from . import spending as spg

        if self.rate is None or self.rule not in spg.RATE_PARAMETERISED:
            return spg.build(self.rule)
        return spg.build(self.rule, rate=float(self.rate))

    @property
    def rated(self) -> bool:
        """Whether this rule's spending level is set by a rate at all."""
        from . import spending as spg

        return self.rule in spg.RATE_PARAMETERISED

    def label(self) -> str:
        rate = f" at {self.rate:.1%}" if self.rated and self.rate is not None \
            else ""
        return f"{self.rule}{rate}, retire at {self.retire_age}"

    def key(self) -> str:
        rate = f"_{int(round((self.rate or 0) * 1000)):03d}" if self.rated \
            else "_xxx"
        return f"{self.rule}{rate}_a{self.retire_age}"


def plan_grid(rules: Sequence[str], rates: Sequence[float],
              ages: Sequence[int]) -> List[Plan]:
    """Every plan in the search space, with the rateless rules listed once.

    Sweeping a rate over a rule that derives its spending from the planning
    horizon would score the same plan several times and bias any "how many
    plans did this rule win" count toward the rules that happen to take a
    parameter.
    """
    from . import spending as spg

    out: List[Plan] = []
    for rule in rules:
        options = [float(r) for r in rates] if rule in spg.RATE_PARAMETERISED \
            else [None]
        for rate in options:
            for age in ages:
                out.append(Plan(rule=rule, rate=rate, retire_age=int(age)))
    return out


def spec_for(spec: Any, retire_age: int) -> Any:
    """The same investor retiring on a different birthday."""
    return dataclasses.replace(spec, age_retire=int(retire_age))


# ---------------------------------------------------------------------------
# Part 1: the allocation grid crossed with the rule
# ---------------------------------------------------------------------------
def score_strategies(sampler: Any, strategies: Mapping[str, Any], spec: Any,
                     cfg: Mapping[str, Any], n_paths: int, chunk_size: int,
                     rule: Any, income_seed: int = 12345) -> pd.DataFrame:
    """Every strategy scored against one withdrawal rule, on shared paths."""
    from . import lifecycle as lc
    from . import utility as ut

    gamma = float(cfg["utility"]["baseline_risk_aversion"])
    beta = float(cfg["utility"]["discount_factor"])
    bequest_weight = float(cfg["utility"]["bequest_weight"])
    bequest_on = bool(cfg["utility"]["bequest_enabled"])
    root = np.random.SeedSequence(int(income_seed))
    children = iter(root.spawn(int(np.ceil(n_paths / chunk_size))))
    parts: Dict[str, Dict[str, List[np.ndarray]]] = {
        key: {"utility": [], "ruin": [], "consumption": []}
        for key in strategies}
    for chunk in sampler.chunks(n_paths, chunk_size):
        income = lc.simulate_income(
            spec, chunk.n_paths, np.random.default_rng(next(children)),
            dom_eq=chunk.dom_eq, intl_eq=chunk.intl_eq)
        for key, outcome in lc.simulate_all(chunk, strategies, spec, income,
                                            rule).items():
            bundle = ut.bundle_from_outcome(outcome, cfg, spec)
            parts[key]["utility"].append(ut.crra_lifetime_utility(
                bundle, gamma, beta, bequest_weight, bequest_on))
            parts[key]["ruin"].append(outcome.ruin.astype(float))
            parts[key]["consumption"].append(
                outcome.consumption[:, spec.retirement_slice].mean(axis=1))

    window = str(cfg["utility"].get("consumption_window", "retirement"))
    horizon = spec.n_retired if window == "retirement" else spec.horizon
    weights = ut.discount_weights(int(horizon), beta, bequest_weight,
                                  bequest_on)
    rows: List[Dict[str, Any]] = []
    for key, fields in parts.items():
        utility = np.concatenate(fields["utility"])
        rows.append({
            "strategy": key,
            "cec": float(ut._inverse_felicity(
                float(utility.mean()) / weights.sum(), gamma)),
            "prob_ruin": float(np.concatenate(fields["ruin"]).mean()),
            "mean_retirement_consumption": float(
                np.concatenate(fields["consumption"]).mean()),
            "n_paths": int(utility.size),
        })
    return pd.DataFrame.from_records(rows)


def sweep_by_rule(sampler_factory: Callable[[], Any],
                  strategies: Mapping[str, Any], spec: Any,
                  cfg: Mapping[str, Any], n_paths: int, chunk_size: int,
                  plans: Sequence[Plan], income_seed: int = 12345,
                  ) -> pd.DataFrame:
    """The whole allocation grid, re-scored under each plan.

    ``sampler_factory`` is called once per plan so every rule sees the same
    lifetimes: the comparison is paired, and the differences are the rule's
    effect rather than the draw's.
    """
    frames: List[pd.DataFrame] = []
    for plan in plans:
        LOGGER.info("scoring %d portfolios under %s", len(strategies),
                    plan.label())
        block = score_strategies(sampler_factory(), strategies, spec, cfg,
                                 n_paths, chunk_size, plan.build(),
                                 income_seed)
        block.insert(0, "rule", plan.rule)
        block.insert(1, "rate", np.nan if plan.rate is None else plan.rate)
        block.insert(2, "can_deplete", CAN_DEPLETE.get(plan.rule, True))
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def optimum_by_rule(frame: pd.DataFrame, parameter: Mapping[str, float],
                    name: str = "domestic_share") -> pd.DataFrame:
    """The certainty-equivalent-maximising share under each rule."""
    from . import inflation as ifl

    return ifl.optimum_by_bucket(frame.rename(columns={"cec": "cec_plan"}),
                                 parameter, "cec_plan", name, group="rule")


def ruin_minimum_by_rule(frame: pd.DataFrame,
                         parameter: Mapping[str, float],
                         name: str = "domestic_share") -> pd.DataFrame:
    """The ruin-minimising share under each rule, beside the CEC-maximising one.

    The mechanism check. If the retiree's tilt toward the home market is
    bought with ruin rather than with return, these two columns should agree
    wherever ruin is possible and part company wherever it is not.
    """
    rows: List[Dict[str, Any]] = []
    for rule, block in frame.groupby("rule", sort=False):
        block = block[block["strategy"].isin(parameter)].copy()
        if not len(block):
            continue
        block["share"] = [float(parameter[k]) for k in block["strategy"]]
        by_cec = block.loc[block["cec"].idxmax()]
        by_ruin = block.loc[block["prob_ruin"].idxmin()]
        # Whether ruin is on the table is read off the simulated paths, not
        # off the prior in CAN_DEPLETE. Where a rule cannot run out the ruin
        # column is flat at zero and its argmin is an artefact of the grid
        # order, so it is reported as undefined rather than as a number.
        possible = bool(block["prob_ruin"].max() > 0.0)
        prior = bool(block["can_deplete"].iloc[0])
        rows.append({
            "rule": str(rule),
            "can_deplete": prior,
            "ruin_is_possible": possible,
            "prior_matches_observed": bool(prior == possible),
            f"cec_optimal_{name}": float(by_cec["share"]),
            f"ruin_optimal_{name}": (float(by_ruin["share"]) if possible
                                     else float("nan")),
            "agree": bool(possible
                          and np.isclose(by_cec["share"], by_ruin["share"])),
            "min_ruin": float(block["prob_ruin"].min()),
            "max_ruin": float(block["prob_ruin"].max()),
            "cec_at_optimum": float(by_cec["cec"]),
        })
    return pd.DataFrame.from_records(rows)


def separability(optima: pd.DataFrame, frame: pd.DataFrame,
                 parameter: Mapping[str, float],
                 name: str = "domestic_share") -> Dict[str, Any]:
    """Does the allocation answer depend on the withdrawal rule, and how?

    Two questions with different answers, which is the point. The *ranking*
    of portfolios can be identical under every rule while the *optimum* moves,
    because a ranking of six candidates cannot resolve a grid step.
    """
    column = f"optimal_{name}"
    if not len(optima) or column not in optima.columns:
        return {"measured": False}
    shares = optima[column].astype(float)
    rankings = {}
    for rule, block in frame.groupby("rule", sort=False):
        block = block[block["strategy"].isin(parameter)]
        rankings[str(rule)] = tuple(
            block.sort_values("cec", ascending=False)["strategy"])
    orders = set(rankings.values())
    return {
        "measured": True,
        "rules": int(len(optima)),
        "optimum_low": float(shares.min()),
        "optimum_high": float(shares.max()),
        "optimum_spread_pp": float(shares.max() - shares.min()) * 100.0,
        "optimum_moves_with_the_rule": bool(shares.nunique() > 1),
        "rankings_identical": bool(len(orders) == 1),
        "distinct_rankings": int(len(orders)),
        # The precise form of the finding, when it holds: the coarse claim in
        # `docs/06` survives and the fine one does not.
        "ranking_separable_but_optimum_is_not": bool(
            len(orders) == 1 and shares.nunique() > 1),
    }


# ---------------------------------------------------------------------------
# Part 2: solving the plan and the portfolio together
# ---------------------------------------------------------------------------
class PlanBench:
    """Scores any ``(plan, allocation)`` pair against one set of lifetimes.

    One evaluator is built per *retirement age* and shared across every rule
    that retires on that birthday, because the expensive state -- the return
    stack -- does not depend on the withdrawal policy. A plan sweep is then
    one cheap certainty-equivalent evaluation per plan rather than a rebuild.
    """

    def __init__(self, paths: Any, spec: Any, cfg: Mapping[str, Any],
                 income_seed: int = 12345) -> None:
        from . import lifecycle as lc

        self.paths = paths
        self.spec = spec
        self.cfg = cfg
        # Drawn once at the longest career on the grid and sliced per age, so
        # a shorter working life is a prefix of a longer one rather than an
        # independent draw. Without that, changing the retirement age would
        # change the income shocks too and the comparison would not be paired.
        self._shocks = lc.draw_income_shocks(
            paths.n_paths, spec.horizon, np.random.default_rng(income_seed))
        self._by_age: Dict[int, Any] = {}

    def evaluator(self, retire_age: int) -> Any:
        from . import glidepath as gp
        from . import lifecycle as lc

        age = int(retire_age)
        if age not in self._by_age:
            spec = spec_for(self.spec, age)
            income = lc.simulate_income(
                spec, self.paths.n_paths, shocks=self._shocks,
                dom_eq=self.paths.dom_eq, intl_eq=self.paths.intl_eq)
            self._by_age[age] = gp.BatchEvaluator(self.paths, spec, income,
                                                  self.cfg)
        return self._by_age[age]

    def for_plan(self, plan: Plan) -> Any:
        """An evaluator set up for this plan's rule and retirement age."""
        return self.evaluator(plan.retire_age).with_rule(plan.build())

    def score(self, plan: Plan, equity: np.ndarray, domestic: np.ndarray,
              gamma: float, bond_share: float = 0.7) -> float:
        from . import glidepath as gp

        weights = gp.weights_from_shares(equity, domestic, bond_share)
        return float(self.for_plan(plan).cec(weights[None], gamma)[0])


def score_plans(bench: PlanBench, plans: Sequence[Plan],
                equity: np.ndarray, domestic: np.ndarray, gamma: float,
                bond_share: float = 0.7) -> pd.DataFrame:
    """Every plan scored against one fixed allocation schedule."""
    rows: List[Dict[str, Any]] = []
    for plan in plans:
        rows.append({
            "rule": plan.rule,
            "rate": np.nan if plan.rate is None else float(plan.rate),
            "retire_age": int(plan.retire_age),
            "can_deplete": bool(CAN_DEPLETE.get(plan.rule, True)),
            "label": plan.label(),
            "cec": bench.score(plan, equity, domestic, gamma, bond_share),
        })
    frame = pd.DataFrame.from_records(rows)
    return frame.sort_values("cec", ascending=False).reset_index(drop=True)


def solve_allocation(bench: PlanBench, plan: Plan, gamma: float,
                     equity_grid: Sequence[float],
                     domestic_grid: Sequence[float],
                     start_equity: float = 1.0, start_domestic: float = 0.1,
                     bond_share: float = 0.7, domestic_band_years: int = 5,
                     n_sweeps: int = 2,
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
    """The free-form schedule of Section #glide, solved under one plan."""
    from . import glidepath as gp

    return gp.optimise_free_form_banded(
        bench.for_plan(plan), gamma, equity_grid, domestic_grid,
        start_equity=start_equity, start_domestic=start_domestic,
        bond_share=bond_share, domestic_band_years=domestic_band_years,
        n_sweeps=n_sweeps)


def alternate(bench: PlanBench, plans: Sequence[Plan], gamma: float,
              equity_grid: Sequence[float], domestic_grid: Sequence[float],
              start_equity: float = 1.0, start_domestic: float = 0.1,
              bond_share: float = 0.7, domestic_band_years: int = 5,
              n_sweeps: int = 2, max_rounds: int = 4,
              ) -> Dict[str, Any]:
    """Alternate between choosing the plan and solving the allocation.

    Each round picks the best plan for the current schedule, then re-solves
    the schedule for that plan. The search stops when a round returns the plan
    it started with, which is the fixed point: neither decision would change
    given the other. Reporting how many rounds that took is the honest way to
    say whether the two decisions interact at all -- a search that converges
    after one round has found that they do not.
    """
    horizon = bench.spec.horizon
    equity = np.full(horizon, float(start_equity))
    domestic = np.full(horizon, float(start_domestic))
    history: List[Dict[str, Any]] = []
    chosen: Plan | None = None
    best = float("-inf")

    for rnd in range(int(max_rounds)):
        ranked = score_plans(bench, plans, equity, domestic, gamma, bond_share)
        top = ranked.iloc[0]
        plan = next(p for p in plans if p.label() == top["label"])
        LOGGER.info("round %d: best plan at the current schedule is %s "
                    "(CEC=%.6f)", rnd, plan.label(), float(top["cec"]))
        settled = chosen is not None and plan == chosen
        chosen = plan
        equity, domestic, best = solve_allocation(
            bench, plan, gamma, equity_grid, domestic_grid,
            float(equity.mean()), float(domestic.mean()), bond_share,
            domestic_band_years, n_sweeps)
        history.append({
            "round": rnd, "plan": plan.label(), "rule": plan.rule,
            "rate": np.nan if plan.rate is None else float(plan.rate),
            "retire_age": int(plan.retire_age),
            "cec_at_incoming_schedule": float(top["cec"]),
            "cec_after_resolving": best,
            "mean_equity": float(equity.mean()),
            "mean_domestic": float(domestic.mean()),
            "plan_unchanged": bool(settled),
        })
        if settled:
            break

    return {
        "plan": chosen, "equity": equity, "domestic": domestic, "cec": best,
        "rounds": pd.DataFrame.from_records(history),
        "converged": bool(history and history[-1]["plan_unchanged"]),
    }


def ablation(bench: PlanBench, plans: Sequence[Plan], baseline: Plan,
             gamma: float, equity_grid: Sequence[float],
             domestic_grid: Sequence[float], base_equity: np.ndarray,
             base_domestic: np.ndarray, joint: Mapping[str, Any],
             bond_share: float = 0.7, domestic_band_years: int = 5,
             n_sweeps: int = 2) -> pd.DataFrame:
    """What each degree of freedom is worth alone, and what they are worth
    together.

    Four rows. The interaction -- the joint gain less the two one-sided gains
    -- is the quantity this whole section exists to produce: it is zero
    exactly when the plan and the portfolio can be chosen independently, which
    is what `docs/06` assumes.
    """
    rows: List[Dict[str, Any]] = []
    base = bench.score(baseline, base_equity, base_domestic, gamma, bond_share)
    rows.append({"freedom": "neither", "plan": baseline.label(),
                 "cec": base, "mean_equity": float(np.mean(base_equity)),
                 "mean_domestic": float(np.mean(base_domestic))})

    eq, dom, alloc_cec = solve_allocation(
        bench, baseline, gamma, equity_grid, domestic_grid,
        float(np.mean(base_equity)), float(np.mean(base_domestic)),
        bond_share, domestic_band_years, n_sweeps)
    rows.append({"freedom": "allocation only", "plan": baseline.label(),
                 "cec": alloc_cec, "mean_equity": float(eq.mean()),
                 "mean_domestic": float(dom.mean())})

    ranked = score_plans(bench, plans, base_equity, base_domestic, gamma,
                         bond_share)
    rows.append({"freedom": "plan only", "plan": str(ranked["label"].iloc[0]),
                 "cec": float(ranked["cec"].iloc[0]),
                 "mean_equity": float(np.mean(base_equity)),
                 "mean_domestic": float(np.mean(base_domestic))})

    rows.append({"freedom": "both", "plan": joint["plan"].label(),
                 "cec": float(joint["cec"]),
                 "mean_equity": float(np.mean(joint["equity"])),
                 "mean_domestic": float(np.mean(joint["domestic"]))})

    frame = pd.DataFrame.from_records(rows)
    frame["gain_over_neither_pct"] = (frame["cec"] / base - 1.0) * 100.0
    by = frame.set_index("freedom")["gain_over_neither_pct"]
    frame["interaction_pct"] = np.where(
        frame["freedom"] == "both",
        by.get("both", np.nan) - by.get("allocation only", np.nan)
        - by.get("plan only", np.nan), np.nan)
    return frame


def verdict(joint: Mapping[str, Any], ablation_frame: pd.DataFrame,
            plans: Sequence[Plan], baseline: Plan,
            spec: Any) -> Dict[str, Any]:
    """What the joint search found, classified rather than asserted."""
    if not len(ablation_frame) or joint.get("plan") is None:
        return {"measured": False}
    by = ablation_frame.set_index("freedom")
    plan = joint["plan"]
    ages = sorted({int(p.retire_age) for p in plans})
    interaction = float(
        by.loc["both", "gain_over_neither_pct"]
        - by.loc["allocation only", "gain_over_neither_pct"]
        - by.loc["plan only", "gain_over_neither_pct"])
    equity = np.asarray(joint["equity"], dtype=float)
    domestic = np.asarray(joint["domestic"], dtype=float)
    working = spec.ages < plan.retire_age
    rated = [p for p in plans if p.rule == plan.rule and p.rate is not None]
    rates = sorted({float(p.rate) for p in rated})
    found: Dict[str, Any] = {
        "measured": True,
        "plan": plan.label(),
        "rule": plan.rule,
        "rate": np.nan if plan.rate is None else float(plan.rate),
        "retire_age": int(plan.retire_age),
        "rule_can_deplete": bool(CAN_DEPLETE.get(plan.rule, True)),
        "cec": float(joint["cec"]),
        "rounds": int(len(joint["rounds"])),
        "converged": bool(joint.get("converged", False)),
        # The interaction is the finding. Zero means the two decisions
        # separate; anything else means solving them apart leaves value on
        # the table.
        "interaction_pct": interaction,
        "decisions_separate": bool(abs(interaction) < 0.05),
        "gain_allocation_pct": float(
            by.loc["allocation only", "gain_over_neither_pct"]),
        "gain_plan_pct": float(by.loc["plan only", "gain_over_neither_pct"]),
        "gain_joint_pct": float(by.loc["both", "gain_over_neither_pct"]),
        "plan_beats_allocation": bool(
            by.loc["plan only", "gain_over_neither_pct"]
            > by.loc["allocation only", "gain_over_neither_pct"]),
        "joint_plan_differs_from_plan_only": bool(
            str(by.loc["both", "plan"]) != str(by.loc["plan only", "plan"])),
        "mean_equity_working": float(equity[working].mean()),
        "mean_equity_retired": float(equity[~working].mean()),
        "mean_domestic_working": float(domestic[working].mean()),
        "mean_domestic_retired": float(domestic[~working].mean()),
        "domestic_rises_after_retirement": bool(
            domestic[~working].mean() > domestic[working].mean()),
    }
    if len(ages) > 1:
        found.update({
            "ages_searched": ages,
            "retire_age_at_ceiling": bool(int(plan.retire_age) == max(ages)),
            "retire_age_at_floor": bool(int(plan.retire_age) == min(ages)),
        })
    if rates:
        found.update({
            "rates_searched": rates,
            "rate_at_ceiling": bool(plan.rate is not None
                                    and float(plan.rate) >= max(rates)),
            "rate_at_floor": bool(plan.rate is not None
                                  and float(plan.rate) <= min(rates)),
        })
    return found
