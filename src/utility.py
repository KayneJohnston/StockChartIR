"""Preference aggregation: CRRA, Epstein-Zin and shortfall statistics.

Every metric here maps a cross-section of simulated *real consumption
paths* (and terminal bequests) into a single scalar that can be compared
across candidate portfolios.  The workhorse is the **certainty equivalent
consumption** (CEC): the constant real consumption stream that would leave
the investor exactly as well off as the risky strategy.  Because it is
denominated in consumption units it is directly comparable across
preference parameters, which raw expected utility is not.

Notation used throughout:

``C[n, h]``  real consumption of path ``n`` at horizon step ``h``
``B[n]``     real bequest of path ``n``, received one step after the last
             consumption date
``beta``     annual subjective discount factor
``gamma``    coefficient of relative risk aversion
``psi``      elasticity of intertemporal substitution (Epstein-Zin only)
``b``        weight on the bequest inside the consumption aggregator
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Sequence

import numpy as np

#: Below this, consumption is treated as the floor value.  Only binds when
#: social security is switched off and a path is fully ruined.
DEFAULT_FLOOR = 1.0e-4


# ---------------------------------------------------------------------------
# Discount weights
# ---------------------------------------------------------------------------
def discount_weights(horizon: int, beta: float, bequest_weight: float = 0.0,
                     include_bequest: bool = False) -> np.ndarray:
    """Weights ``[beta^0, ..., beta^(H-1)]`` plus ``b * beta^H`` if requested."""
    weights = beta ** np.arange(horizon, dtype=float)
    if include_bequest:
        weights = np.concatenate([weights, [bequest_weight * beta ** horizon]])
    return weights


def _felicity(consumption: np.ndarray, gamma: float) -> np.ndarray:
    """``u(c) = c^(1-gamma)/(1-gamma)``, with the log limit at ``gamma == 1``."""
    if np.isclose(gamma, 1.0):
        return np.log(consumption)
    return np.power(consumption, 1.0 - gamma) / (1.0 - gamma)


def _inverse_felicity(mean_utility: np.ndarray, gamma: float) -> np.ndarray:
    """Invert :func:`_felicity` to recover a consumption level."""
    if np.isclose(gamma, 1.0):
        return np.exp(mean_utility)
    return np.power((1.0 - gamma) * mean_utility, 1.0 / (1.0 - gamma))


def _power_mean(values: np.ndarray, weights: np.ndarray, exponent: float,
                axis: int = -1) -> np.ndarray:
    """Weighted power mean ``(sum w x^e / sum w)^(1/e)`` with the log limit."""
    total = weights.sum()
    if np.isclose(exponent, 0.0):
        return np.exp(np.tensordot(weights, np.log(values),
                                   axes=([0], [axis])) / total)
    powered = np.power(values, exponent)
    aggregate = np.tensordot(weights, powered, axes=([0], [axis])) / total
    return np.power(aggregate, 1.0 / exponent)


# ---------------------------------------------------------------------------
# Consumption bundles
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ConsumptionBundle:
    """Simulated real consumption paths plus optional terminal bequests.

    ``bequest_shift`` is the ``kappa`` of the De Nardi (2004) "bequests are a
    luxury good" specification: the bequest enters the aggregator as
    ``kappa + B`` rather than ``B``.  Without it a zero bequest carries
    ``u(0) = -inf`` for every ``gamma >= 1``, and a single ruined path drives
    the certainty equivalent of *every* strategy to zero -- the metric stops
    discriminating.  With ``kappa > 0`` a zero bequest is merely bad.  A
    certain stream ``c`` paired with bequest ``B = c - kappa`` returns
    ``CEC = c`` exactly, which keeps the units interpretable.
    """

    consumption: np.ndarray            # (N, H)
    bequest: np.ndarray | None = None  # (N,)
    floor: float = DEFAULT_FLOOR
    bequest_shift: float = 1.0

    def __post_init__(self) -> None:
        if self.consumption.ndim != 2:
            raise ValueError("consumption must be (n_paths, horizon)")
        if self.bequest is not None and self.bequest.ndim != 1:
            raise ValueError("bequest must be (n_paths,)")
        if self.bequest is not None and \
                self.bequest.shape[0] != self.consumption.shape[0]:
            raise ValueError("bequest and consumption disagree on n_paths")

    @property
    def n_paths(self) -> int:
        return int(self.consumption.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.consumption.shape[1])

    def matrix(self, include_bequest: bool) -> np.ndarray:
        """Floored ``(N, H)`` or ``(N, H+1)`` consumption-equivalent matrix."""
        block = np.maximum(self.consumption, self.floor)
        if include_bequest and self.bequest is not None:
            beq = np.maximum(self.bequest_shift + self.bequest, self.floor)
            block = np.concatenate([block, beq[:, None]], axis=1)
        return block


# ---------------------------------------------------------------------------
# CRRA
# ---------------------------------------------------------------------------
def crra_lifetime_utility(bundle: ConsumptionBundle, gamma: float, beta: float,
                          bequest_weight: float = 0.0,
                          include_bequest: bool = False) -> np.ndarray:
    """Per-path discounted lifetime CRRA utility, shape ``(N,)``."""
    block = bundle.matrix(include_bequest)
    weights = discount_weights(bundle.horizon, beta, bequest_weight,
                               include_bequest and bundle.bequest is not None)
    return _felicity(block, gamma) @ weights


def crra_certainty_equivalent(bundle: ConsumptionBundle, gamma: float,
                              beta: float, bequest_weight: float = 0.0,
                              include_bequest: bool = False) -> float:
    """Certainty equivalent consumption under time-additive CRRA utility.

    The constant real stream ``c*`` such that
    ``sum_h beta^h u(c*) == E[sum_h beta^h u(C_h)]``.  A perfectly certain
    constant stream returns itself, which is the property the unit tests
    pin down.
    """
    utility = crra_lifetime_utility(bundle, gamma, beta, bequest_weight,
                                    include_bequest)
    weights = discount_weights(bundle.horizon, beta, bequest_weight,
                               include_bequest and bundle.bequest is not None)
    return float(_inverse_felicity(utility.mean() / weights.sum(), gamma))


# ---------------------------------------------------------------------------
# Epstein-Zin
# ---------------------------------------------------------------------------
def epstein_zin_certainty_equivalent(
    bundle: ConsumptionBundle,
    gamma: float,
    psi: float,
    beta: float,
    bequest_weight: float = 0.0,
    include_bequest: bool = False,
) -> float:
    """Epstein-Zin certainty equivalent under early resolution of uncertainty.

    Recursive Epstein-Zin-Weil preferences separate the two roles that CRRA
    conflates: ``psi`` governs willingness to substitute consumption across
    *time*, ``gamma`` governs aversion to consumption risk across *states*.
    Evaluating the full recursion path-by-path would require conditional
    continuation values, which a state-free bootstrap cannot identify.  This
    function therefore uses the *ex-ante* (early-resolution) form, in which
    all uncertainty is resolved at date 0:

    1. Aggregate each realised path over time with a CES index in ``psi``::

           Chat_n = [ sum_h beta^h C[n,h]^(1-1/psi) / sum_h beta^h ]^(1/(1-1/psi))

    2. Aggregate the resulting cross-section over states with a CRRA
       certainty equivalent in ``gamma``::

           CE = ( E[ Chat^(1-gamma) ] )^(1/(1-gamma))

    The specification nests time-additive CRRA exactly when ``psi == 1/gamma``,
    which is asserted in ``tests/test_utility.py``.  Section 4 of
    ``docs/03_lifecycle_utility_model.md`` states the assumption and its
    limitation explicitly.
    """
    block = bundle.matrix(include_bequest)
    weights = discount_weights(bundle.horizon, beta, bequest_weight,
                               include_bequest and bundle.bequest is not None)
    time_index = _power_mean(block, weights, 1.0 - 1.0 / psi, axis=1)
    risk_weights = np.ones(bundle.n_paths)
    return float(_power_mean(time_index, risk_weights, 1.0 - gamma, axis=0))


# ---------------------------------------------------------------------------
# Shortfall / distributional metrics
# ---------------------------------------------------------------------------
def shortfall_metrics(
    bundle: ConsumptionBundle,
    ruin: np.ndarray,
    wealth_at_retirement: np.ndarray,
    retirement_slice: slice,
    percentiles: Sequence[float] = (1, 5, 10, 25, 50, 75, 90, 95, 99),
    consumption_target: np.ndarray | float | None = None,
) -> Dict[str, float]:
    """Ruin, bequest and retirement-consumption statistics for one strategy.

    ``consumption_target`` is the benchmark the shortfall statistics are
    measured against.  It **must** be strategy-invariant -- a per-path
    replacement-rate target computed from labour income, for instance -- or
    the shortfall numbers are not comparable across strategies.  Measuring
    each strategy against its own median (the tempting shortcut) makes every
    strategy look identical by construction.
    """
    retirement_consumption = bundle.consumption[:, retirement_slice]
    bequest = (bundle.bequest if bundle.bequest is not None
               else np.zeros(bundle.n_paths))
    out: Dict[str, float] = {
        "prob_ruin": float(ruin.mean()),
        "median_wealth_at_retirement": float(np.median(wealth_at_retirement)),
        "mean_wealth_at_retirement": float(wealth_at_retirement.mean()),
        "median_bequest": float(np.median(bequest)),
        "mean_bequest": float(bequest.mean()),
        "p5_bequest": float(np.percentile(bequest, 5)),
        "prob_zero_bequest": float((bequest <= 0).mean()),
        "median_retirement_consumption":
            float(np.median(retirement_consumption)),
        "mean_retirement_consumption": float(retirement_consumption.mean()),
    }
    for q in percentiles:
        out[f"p{q:g}_wealth_at_retirement"] = float(
            np.percentile(wealth_at_retirement, q))
        out[f"p{q:g}_bequest"] = float(np.percentile(bequest, q))
        out[f"p{q:g}_retirement_consumption"] = float(
            np.percentile(retirement_consumption, q))
    # Expected shortfall of retirement consumption against a common,
    # strategy-invariant target.
    per_path_mean = retirement_consumption.mean(axis=1)
    if consumption_target is None:
        consumption_target = float(np.median(per_path_mean))
    target = np.asarray(consumption_target, dtype=float)
    shortfall = np.maximum(target - per_path_mean, 0.0)
    out["mean_consumption_target"] = float(np.mean(target))
    out["mean_consumption_shortfall"] = float(shortfall.mean())
    out["prob_consumption_below_target"] = float((per_path_mean < target).mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(target > 0, per_path_mean / target, np.nan)
    out["median_consumption_replacement_ratio"] = float(np.nanmedian(ratio))
    out["p5_consumption_replacement_ratio"] = float(
        np.nanpercentile(ratio, 5))
    return out


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------
def evaluate_preferences(
    bundle: ConsumptionBundle,
    cfg: Mapping[str, Any],
) -> Dict[str, float]:
    """Every CEC the report needs, keyed by preference specification."""
    util_cfg = cfg["utility"]
    beta = float(util_cfg["discount_factor"])
    bequest_weight = float(util_cfg["bequest_weight"])
    include_bequest = bool(util_cfg["bequest_enabled"])
    out: Dict[str, float] = {}
    if bundle.bequest_shift <= 0 and include_bequest:
        raise ValueError("bequest_shift must be positive when bequests are on")
    for gamma in util_cfg["risk_aversions"]:
        out[f"cec_crra_gamma{float(gamma):g}"] = crra_certainty_equivalent(
            bundle, float(gamma), beta, bequest_weight, include_bequest)
    for gamma in util_cfg["risk_aversions"]:
        for psi in util_cfg["epstein_zin_ies"]:
            key = f"cec_ez_gamma{float(gamma):g}_psi{float(psi):g}"
            out[key] = epstein_zin_certainty_equivalent(
                bundle, float(gamma), float(psi), beta, bequest_weight,
                include_bequest)
    return out


def bundle_from_outcome(outcome: Any, cfg: Mapping[str, Any],
                        spec: Any | None = None) -> ConsumptionBundle:
    """Wrap a :class:`~src.lifecycle.LifecycleOutcome` using config settings.

    ``utility.consumption_window`` selects the dates that enter the
    aggregator.  With a fixed savings rate, working-life consumption is
    ``(1 - s) * Y`` on every strategy, so the ``"retirement"`` window is what
    actually discriminates between portfolios; ``"full"`` is available for
    sensitivity work.  Re-indexing the retirement window to start at ``h = 0``
    rescales both ``E[U]`` and the weight total by ``beta^n_working`` and so
    leaves the certainty equivalent unchanged.
    """
    util_cfg = cfg["utility"]
    window = str(util_cfg.get("consumption_window", "retirement"))
    consumption = outcome.consumption
    if window == "retirement":
        if spec is None:
            raise ValueError("spec is required for the retirement window")
        consumption = consumption[:, spec.retirement_slice]
    elif window != "full":
        raise ValueError(f"unknown consumption_window {window!r}")
    return ConsumptionBundle(
        consumption=consumption,
        bequest=outcome.bequest,
        floor=float(util_cfg.get("consumption_floor", DEFAULT_FLOOR)),
        bequest_shift=float(util_cfg.get("bequest_shift", 1.0)),
    )
