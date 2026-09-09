"""Confidence-gated cascade: routing, cost accounting, and threshold selection.

Thresholds are always selected on the validation split as the cheapest
configuration whose accuracy falls within a stated tolerance of a reference
system. The test split is used once, for the selected configuration.
"""
import numpy as np

from .metrics import confidence

DEFAULT_GRID = np.round(np.arange(0.50, 1.0001, 0.0025), 4)
COARSE_GRID = np.round(np.arange(0.50, 1.0001, 0.005), 4)


def product_ensemble(probs_a, probs_b):
    joint = probs_a * probs_b
    return joint / joint.sum(axis=1, keepdims=True)


def route_two_tier(gate_probs, upper_probs, tau):
    """Gate emits above tau, escalates otherwise. Returns predictions and the mask."""
    escalated = confidence(gate_probs) <= tau
    predictions = np.where(escalated, upper_probs.argmax(1), gate_probs.argmax(1))
    return predictions, escalated


def cost_two_tier(escalated, cost_gate, cost_upper):
    return cost_gate + escalated.mean() * cost_upper


def route_three_tier(gate_probs, mid_probs, top_probs, tau_gate, tau_mid):
    exit_gate = confidence(gate_probs) > tau_gate
    exit_mid = (~exit_gate) & (confidence(mid_probs) > tau_mid)
    exit_top = (~exit_gate) & (~exit_mid)
    predictions = np.where(
        exit_gate, gate_probs.argmax(1),
        np.where(exit_mid, mid_probs.argmax(1), top_probs.argmax(1)))
    return predictions, (exit_gate, exit_mid, exit_top)


def cost_three_tier(exits, cost_gate, cost_mid, cost_top):
    exit_gate, _, exit_top = exits
    return cost_gate + (~exit_gate).mean() * cost_mid + exit_top.mean() * cost_top


def select_tau_two_tier(gate_val, upper_val, y_val, reference_accuracy,
                        tolerance_pp, cost_gate, cost_upper, grid=DEFAULT_GRID):
    """Cheapest threshold on validation within `tolerance_pp` of the reference."""
    best = None
    for tau in grid:
        predictions, escalated = route_two_tier(gate_val, upper_val, tau)
        acc = (predictions == y_val).mean() * 100.0
        if acc >= reference_accuracy - tolerance_pp:
            cost = cost_two_tier(escalated, cost_gate, cost_upper)
            if best is None or cost < best[1]:
                best = (float(tau), cost)
    return best[0] if best else None


def select_tau_three_tier(gate_val, mid_val, top_val, y_val, reference_accuracy,
                          tolerance_pp, costs, grid=COARSE_GRID):
    """Grid search over both thresholds. The coarser grid keeps this tractable."""
    cost_gate, cost_mid, cost_top = costs
    best = None
    for tau_gate in grid:
        for tau_mid in grid:
            predictions, exits = route_three_tier(
                gate_val, mid_val, top_val, tau_gate, tau_mid)
            acc = (predictions == y_val).mean() * 100.0
            if acc >= reference_accuracy - tolerance_pp:
                cost = cost_three_tier(exits, cost_gate, cost_mid, cost_top)
                if best is None or cost < best[1]:
                    best = ((float(tau_gate), float(tau_mid)), cost)
    return best[0] if best else None


def select_tau_per_language(gate_val, upper_val, y_val, lang_val, tolerance_pp,
                            grid=DEFAULT_GRID):
    """One threshold per language, each selected against that language's own
    reference accuracy. Group-conditioned recalibration is not a single monotone
    map, so unlike a global temperature it can change the routing decisions."""
    thresholds = {}
    for language in sorted(set(lang_val)):
        mask = lang_val == language
        reference = (upper_val.argmax(1)[mask] == y_val[mask]).mean() * 100.0
        best = None
        for tau in grid:
            predictions, escalated = route_two_tier(gate_val, upper_val, tau)
            acc = (predictions[mask] == y_val[mask]).mean() * 100.0
            if acc >= reference - tolerance_pp:
                rate = escalated[mask].mean()
                if best is None or rate < best[1]:
                    best = (float(tau), rate)
        thresholds[language] = best[0] if best else 1.0
    return thresholds


def apply_per_language(gate_probs, upper_probs, lang, thresholds):
    tau = np.array([thresholds[language] for language in lang])
    escalated = confidence(gate_probs) <= tau
    predictions = np.where(escalated, upper_probs.argmax(1), gate_probs.argmax(1))
    return predictions, escalated
