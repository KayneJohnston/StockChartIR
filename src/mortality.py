"""Death at a random age, rather than on schedule at ninety-three.

Every result before this section kills the investor on their ninety-third
birthday with certainty. That is a modelling convenience and `docs/04`
already lists it as one of the differences from the study this project
re-implements, which draws a lifespan from a mortality table. It matters for
two reasons that pull in opposite directions:

* a fixed horizon **understates** longevity risk, because nobody knows they
  have exactly thirty retired years to fund; and
* a fixed horizon **overstates** the value of the far tail, because a
  strategy is rewarded for consumption at ninety-two that most investors
  never live to spend.

**How it is done here.** Not by re-simulating. Under the headline withdrawal
rule -- a fixed real fraction of wealth at retirement -- the policy does not
depend on the death age at all, so a random lifespan changes only *which*
years of an already-simulated path are experienced, and with what
probability. That makes the exact treatment a re-weighting of the utility
aggregation rather than a new set of paths:

    U = sum_h  beta^h S(h) u(c_h)
      + b * sum_h beta^(h+1) (S(h) - S(h+1)) u(kappa + W_(h+1))
      + b * beta^H S(H) u(kappa + W_H)

Consumption in year ``h`` is enjoyed only if the investor is alive, which
happens with probability ``S(h)``; the estate is whatever wealth is left in
the year they die. A certain stream ``c`` paired with bequest ``c - kappa``
still returns ``CEC = c`` exactly, so the units are the same units as every
other certainty equivalent in the paper and the two are directly comparable.

**Where it is exact and where it is not.** Exact for any policy that does not
condition on the death age -- the fixed real rule, the constant percentage
rule, every constant-weight strategy. *Approximate* for the horizon-based
spending rules of section #spending, which amortise over a planning horizon:
those policies would themselves change if they knew the mortality table, and
re-weighting their outcomes measures the effect of random death on a plan
built for a fixed one. The Gompertz spending rule is the exception -- it
already plans to an actuarial life expectancy -- and it is the one to watch
in the table for that reason.

**The mortality model.** A Gompertz law, ``S(t | x) = exp(exp((x - m) / b)
(1 - exp(t / b)))``, the same functional form the actuarial spending rule in
:mod:`src.spending` uses as its planning divisor. It is a model rather than a
life table lifted from data, and the calibration is swept rather than
asserted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from . import lifecycle as lc
from . import utility as ut

LOGGER = logging.getLogger(__name__)

#: Calibrations swept by default: ``(label, modal age, dispersion)``. The
#: middle one is the Milevsky calibration used elsewhere in the project; the
#: others move the mode five years either way and tighten the spread, which
#: is a wider range than the disagreement between real life tables.
DEFAULT_CALIBRATIONS: Tuple[Tuple[str, float, float], ...] = (
    ("shorter lives (m=83)", 83.0, 10.0),
    ("baseline (m=88)", 88.0, 10.0),
    ("longer lives (m=93)", 93.0, 10.0),
    ("less dispersed (m=88, b=7)", 88.0, 7.0),
)


def survival(spec: lc.LifecycleSpec, modal_age: float = 88.0,
             dispersion: float = 10.0) -> np.ndarray:
    """``S(h)`` for ``h = 0 .. H``: alive at the start of simulated year ``h``.

    Conditioned on being alive at ``age_start``, so ``S(0) == 1`` by
    construction and the investor cannot die before the simulation begins.
    """
    horizon = spec.horizon
    ages = spec.age_start + np.arange(horizon + 1, dtype=float)
    start = float(spec.age_start)
    raw = np.exp(np.exp((start - modal_age) / dispersion)
                 * (1.0 - np.exp((ages - start) / dispersion)))
    out = raw / raw[0]
    return np.clip(out, 0.0, 1.0)


def death_probabilities(survive: np.ndarray) -> np.ndarray:
    """``Pr(die during year h)`` for ``h = 0 .. H-1``, plus the surviving mass.

    The last entry carries ``S(H)``: everyone still alive at the end of the
    simulated horizon dies then, because the model has no year ``H + 1``. It
    is a real mass -- around a fifth under the baseline calibration -- and
    dropping it would both understate life expectancy and lose probability.
    """
    step = survive[:-1] - survive[1:]
    step = np.append(step, survive[-1])
    total = float(step.sum())
    return step / total if total > 0 else step


def death_ages(spec: lc.LifecycleSpec, survive: np.ndarray) -> np.ndarray:
    """Age at death for each entry of :func:`death_probabilities`.

    Dying *during* year ``h`` is death at age ``age_start + h + 1``; the
    trailing survivor mass is carried at the end of the horizon, which is the
    same age as the final step, so the last two entries share an age.
    """
    horizon = survive.size - 1
    ages = spec.age_start + np.arange(1, horizon + 1, dtype=float)
    return np.append(ages, float(spec.age_start + horizon))


def life_expectancy(spec: lc.LifecycleSpec, survive: np.ndarray) -> float:
    """Expected age at death under ``survive``, in years.

    The expectation is *within the model*: the Gompertz law puts mass beyond
    the simulated horizon and the simulation has nowhere to put it, so that
    mass is carried at the final age. The number is therefore a lower bound
    on the law's own life expectancy, and it is the one the results are
    actually conditioned on.
    """
    return float(death_ages(spec, survive) @ death_probabilities(survive))


def window(spec: lc.LifecycleSpec, cfg: Mapping[str, Any]) -> Tuple[int, int]:
    """``(offset, length)`` of the dates the aggregator sees.

    The project's utility is defined over *retirement* consumption plus the
    bequest, because with a fixed savings rate working-life consumption is
    identical on every strategy and including it only dilutes the comparison.
    The same choice has to be made here, or the survival weights would be
    applied to years that carry no information about the allocation.
    """
    name = str(cfg["utility"].get("consumption_window", "retirement"))
    if name == "retirement":
        return spec.n_working, spec.n_retired
    if name == "full":
        return 0, spec.horizon
    raise ValueError(f"unknown consumption_window {name!r}")


def conditional_survival(survive: np.ndarray, offset: int,
                         length: int) -> np.ndarray:
    """``S`` restricted to the window and renormalised to start at one.

    On the retirement window this conditions on reaching retirement. Someone
    who dies at fifty never draws a pension and leaves an estate the model
    does not score, so their probability mass belongs outside a comparison of
    retirement portfolios -- and because the chance of reaching retirement is
    the same on every strategy, conditioning on it cannot bias the ranking.
    """
    block = np.asarray(survive, dtype=float)[offset:offset + length + 1]
    base = float(block[0])
    return block / base if base > 0 else block


def weights(spec: lc.LifecycleSpec, survive: np.ndarray, beta: float,
            bequest_weight: float, offset: int = 0,
            length: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Consumption weights ``beta^j s(j)`` and bequest weights, one per year.

    The bequest weight on year ``j`` is the discounted probability of dying
    with that year's wealth still in the estate, so the two arrays together
    are the full aggregation and their sum is the divisor the certainty
    equivalent inverts with. Dates are indexed from the start of the window,
    which is what keeps these certainty equivalents in the same units as
    :func:`src.utility.crra_certainty_equivalent` on the same window.
    """
    n = int(spec.horizon if length is None else length)
    s = conditional_survival(survive, offset, n)
    discount = beta ** np.arange(n, dtype=float)
    consume = discount * s[:n]
    died = s[:n] - s[1:n + 1]
    beq = bequest_weight * (beta ** np.arange(1, n + 1, dtype=float)) * died
    # Whoever is still alive at the end of the window leaves the balance.
    beq[-1] += bequest_weight * beta ** n * s[n]
    return consume, beq


def certainty_equivalent(outcome: lc.LifecycleOutcome, spec: lc.LifecycleSpec,
                         cfg: Mapping[str, Any], gamma: float,
                         survive: np.ndarray) -> float:
    """Survival-weighted CRRA certainty equivalent, in the usual units."""
    beta = float(cfg["utility"]["discount_factor"])
    bequest_weight = float(cfg["utility"]["bequest_weight"])
    shift = float(cfg["utility"].get("bequest_shift", 1.0))
    floor = float(cfg["utility"].get("consumption_floor", ut.DEFAULT_FLOOR))
    offset, length = window(spec, cfg)

    consume_w, beq_w = weights(spec, survive, beta, bequest_weight,
                               offset, length)
    consumption = np.maximum(
        outcome.consumption[:, offset:offset + length], floor)
    # Wealth at the *end* of year h is the estate of someone who dies in it.
    estate = np.maximum(
        shift + outcome.wealth[:, offset + 1:offset + length + 1], floor)

    utility = (ut._felicity(consumption, gamma) @ consume_w
               + ut._felicity(estate, gamma) @ beq_w)
    total = float(consume_w.sum() + beq_w.sum())
    return float(ut._inverse_felicity(float(utility.mean()) / total, gamma))


def probability_of_ruin(outcome: lc.LifecycleOutcome, spec: lc.LifecycleSpec,
                        survive: np.ndarray,
                        cfg: Mapping[str, Any] | None = None) -> float:
    """Chance of outliving the portfolio, integrated over the death age.

    Ruin under a fixed horizon is "the money ran out before ninety-three".
    Under a random one it is "the money ran out before *you* did", which is
    strictly kinder: a path that depletes at ninety-one is not a ruined
    retirement for an investor who dies at eighty-four. Conditioned on
    reaching retirement, for the reason :func:`conditional_survival` gives.
    """
    offset = spec.n_working if cfg is None else window(spec, cfg)[0]
    length = spec.horizon - offset
    s = conditional_survival(survive, offset, length)
    death_prob = death_probabilities(s)
    ages = spec.age_start + offset + np.arange(1, length + 1, dtype=float)
    ages = np.append(ages, float(spec.age_start + offset + length))
    ruin_age = outcome.ruin_age.astype(float)[:, None]
    ruined_before_death = (ruin_age < ages[None, :]) & outcome.ruin[:, None]
    return float((ruined_before_death @ death_prob).mean())


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def compare(outcomes: Mapping[str, lc.LifecycleOutcome],
            spec: lc.LifecycleSpec, cfg: Mapping[str, Any], gamma: float,
            calibrations: Sequence[Tuple[str, float, float]]
            = DEFAULT_CALIBRATIONS) -> pd.DataFrame:
    """Every strategy re-scored under a fixed horizon and each mortality law.

    The fixed-horizon row is the paper's existing number, recomputed through
    the same code path as the others so that the comparison is a comparison of
    mortality assumptions and not of two implementations.
    """
    horizon = spec.horizon
    certain = np.ones(horizon + 1)
    certain[-1] = 0.0            # dies at age_death with probability one
    rows: List[Dict[str, Any]] = []
    laws: List[Tuple[str, np.ndarray]] = [("fixed horizon", certain)]
    laws += [(label, survival(spec, modal, disp))
             for label, modal, disp in calibrations]
    for label, survive in laws:
        expectancy = life_expectancy(spec, survive)
        for key, outcome in outcomes.items():
            rows.append({
                "mortality": label,
                "life_expectancy": expectancy,
                "strategy": key,
                "label": outcome.label,
                "cec": certainty_equivalent(outcome, spec, cfg, gamma, survive),
                "prob_ruin": probability_of_ruin(outcome, spec, survive, cfg),
            })
    frame = pd.DataFrame.from_records(rows)
    frame["rank"] = frame.groupby("mortality")["cec"].rank(
        ascending=False, method="min").astype(int)
    return frame


def ranking_shift(frame: pd.DataFrame) -> pd.DataFrame:
    """Each strategy's rank under every mortality assumption, wide."""
    return frame.pivot_table(index=["strategy", "label"], columns="mortality",
                             values="rank").reset_index()


def gap_curve(frame: pd.DataFrame, pair: Tuple[str, str]) -> pd.DataFrame:
    """The lead of one strategy over another under each assumption."""
    rows: List[Dict[str, Any]] = []
    for label in dict.fromkeys(frame["mortality"]):
        block = frame[frame["mortality"] == label]
        values = {r["strategy"]: float(r["cec"]) for _, r in block.iterrows()}
        ruin = {r["strategy"]: float(r["prob_ruin"]) for _, r in block.iterrows()}
        rows.append({
            "mortality": label,
            "life_expectancy": float(block["life_expectancy"].iloc[0]),
            "gap_pct": (values[pair[0]] / values[pair[1]] - 1.0) * 100.0
            if pair[0] in values and pair[1] in values else float("nan"),
            f"cec_{pair[0]}": values.get(pair[0], float("nan")),
            f"cec_{pair[1]}": values.get(pair[1], float("nan")),
            f"ruin_{pair[0]}": ruin.get(pair[0], float("nan")),
            f"ruin_{pair[1]}": ruin.get(pair[1], float("nan")),
            "winner": max(values, key=values.get) if values else "",
        })
    return pd.DataFrame.from_records(rows)


def verdict(frame: pd.DataFrame, curve: pd.DataFrame,
            pair: Tuple[str, str]) -> Dict[str, Any]:
    """What random mortality does to the ranking, classified from the sweep."""
    if not len(curve):
        return {"laws": 0}
    fixed = curve[curve["mortality"] == "fixed horizon"]
    others = curve[curve["mortality"] != "fixed horizon"]
    base_gap = float(fixed["gap_pct"].iloc[0]) if len(fixed) else float("nan")
    winners = set(str(w) for w in curve["winner"])
    orders = {label: tuple(frame[frame["mortality"] == label]
                           .sort_values("rank")["strategy"])
              for label in dict.fromkeys(frame["mortality"])}
    reference = orders.get("fixed horizon")
    return {
        "laws": int(len(curve)),
        "fixed_horizon_gap_pct": base_gap,
        "min_gap_pct": float(curve["gap_pct"].min()),
        "max_gap_pct": float(curve["gap_pct"].max()),
        "largest_change_pp": float((others["gap_pct"] - base_gap).abs().max())
        if len(others) else float("nan"),
        "winner_ever_changes": bool(len(winners) > 1),
        "winners_seen": sorted(winners),
        "winner_is_expected_throughout": bool(winners == {pair[0]}),
        "ordering_ever_changes": bool(
            any(order != reference for order in orders.values())),
        "shortest_life_expectancy": float(curve["life_expectancy"].min()),
        "longest_life_expectancy": float(curve["life_expectancy"].max()),
        # Random death cannot make a portfolio safer, but it can make the
        # *chance of outliving it* smaller, because the investor has fewer
        # years to outlive it in.
        "ruin_falls_under_mortality": bool(
            len(others) and float(others[f"ruin_{pair[1]}"].max())
            < float(fixed[f"ruin_{pair[1]}"].iloc[0])),
    }
