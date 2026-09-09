"""Confidence metrics, recalibration, and paired significance tests.

All rank statistics are computed on the logit scale. Confidence saturates as a
function of the logit, so a rank statistic computed on probabilities is
contaminated by floating-point ties at high confidence: on one of the ensembles
used here the difficulty inversion score reads 77.36 on the probability scale
and 95.28 on the logit scale, because 16,825 of 17,341 confidences round to
exactly 1.0.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

EPS = 1e-45


def logit(probs):
    """Recover the binary logit from a two-column softmax output."""
    return np.log(np.clip(probs[:, 1], EPS, 1.0) / np.clip(probs[:, 0], EPS, 1.0))


def probs_from_logit(z):
    positive = 1.0 / (1.0 + np.exp(-z))
    return np.stack([1.0 - positive, positive], axis=1)


def temperature_scale(probs, temperature):
    return probs_from_logit(logit(probs) / temperature)


def fit_temperature(probs_val, y_val, bounds=(0.05, 10.0)):
    """Single-parameter temperature fitted by negative log likelihood."""
    z = logit(probs_val)

    def nll(temperature):
        if temperature <= 0:
            return 1e9
        positive = np.clip(1.0 / (1.0 + np.exp(-z / temperature)), EPS, 1 - EPS)
        return -np.mean(y_val * np.log(positive) + (1 - y_val) * np.log(1 - positive))

    return float(minimize_scalar(nll, bounds=bounds, method='bounded').x)


def confidence(probs):
    """Maximum class probability."""
    return probs.max(axis=1)


def top_two_margin(probs):
    """Gap between the two largest softmax values.

    For two classes this equals 2 * confidence - 1, so thresholding it is
    equivalent to thresholding the maximum class probability.
    """
    ordered = np.sort(probs, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def expected_calibration_error(probs, y, n_bins=10):
    conf = confidence(probs)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (conf > low) & (conf <= high)
        if in_bin.sum():
            error += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(error)


def difficulty_inversion_score(probs, y):
    """Proportion of (difficult, easy) instance pairs ranked correctly by confidence.

    Equivalent to the area under the ROC curve of confidence as a predictor of
    correctness, and therefore a rank statistic: invariant under any strictly
    increasing recalibration of the confidence score. Computed on |logit| to
    avoid the saturation described in the module docstring.
    """
    score = np.abs(logit(probs))
    correct = (probs.argmax(1) == y).astype(int)
    if correct.min() == correct.max():
        return float('nan')
    return float(roc_auc_score(correct, score) * 100.0)


def accuracy(probs, y):
    return float((probs.argmax(1) == y).mean() * 100.0)


def mcnemar(pred_a, pred_b, y):
    """Exact paired test. Returns the two discordant counts and the p-value."""
    a_only = int(((pred_a == y) & (pred_b != y)).sum())
    b_only = int(((pred_a != y) & (pred_b == y)).sum())
    if a_only + b_only == 0:
        return a_only, b_only, 1.0
    p = binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue
    return a_only, b_only, float(p)


def escalation_sets_identical(probs, temperature, rates=(0.80, 0.85, 0.90, 0.95, 0.99)):
    """Check that recalibration leaves the routing decisions unchanged.

    For each retention rate, the set escalated under the recalibrated score at
    the matching quantile is compared to the set escalated under the raw score.
    Reported as set equality rather than as a rank correlation, which is
    contaminated by ties at high confidence.
    """
    raw = 1.0 / (1.0 + np.exp(-np.abs(logit(probs))))
    scaled = 1.0 / (1.0 + np.exp(-np.abs(logit(probs)) / temperature))
    return all(
        np.array_equal(scaled <= np.quantile(scaled, 1 - rate),
                       raw <= np.quantile(raw, 1 - rate))
        for rate in rates)
