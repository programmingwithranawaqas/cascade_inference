"""Distribution-free threshold calibration.

Risk is the one-sided damage rate

    R(tau) = P(cascade wrong AND reference right),

the probability that routing away from the reference costs a correct answer. It
is bounded in [0, 1], as the concentration bounds require, and upper-bounds the
signed excess error, since excess error is damage minus repair. A signed excess
loss lives in {-1, 0, 1}; clipping its empirical mean at zero does not make a
[0, 1] bound apply to it.

Thresholds are tested in a fixed sequence from most conservative to least,
stopping at the first failure, which controls the family-wise error at delta
without a multiplicity correction. Testing a whole grid and then taking the most
permissive threshold that passed does not.

At n = 8,670 and delta = 0.05 a Hoeffding bound carries an additive slack of
1.31 percentage points whatever the data, so certificates below that are
unreachable with it. The KL and Bentkus bounds agree to within one grid step.
"""
import numpy as np
from scipy.stats import binom

from .metrics import confidence


def hoeffding_p(risk_hat, n, alpha):
    if risk_hat >= alpha:
        return 1.0
    return float(np.exp(-2.0 * n * (alpha - risk_hat) ** 2))


def kl_p(risk_hat, n, alpha):
    """Chernoff bound via the binary relative entropy. Tighter than Hoeffding."""
    if risk_hat >= alpha:
        return 1.0
    if risk_hat <= 0:
        divergence = -np.log(1.0 - alpha)
    else:
        divergence = (risk_hat * np.log(risk_hat / alpha)
                      + (1 - risk_hat) * np.log((1 - risk_hat) / (1 - alpha)))
    return float(np.exp(-n * divergence))


def bentkus_p(risk_hat, n, alpha):
    """Tightest of the three when the empirical risk is near zero."""
    return float(np.e * binom.cdf(int(np.ceil(n * risk_hat)), n, alpha))


BOUNDS = {'hoeffding': hoeffding_p, 'kl': kl_p, 'bentkus': bentkus_p}


def damage_rate(gate_probs, reference_probs, y, tau):
    escalated = confidence(gate_probs) <= tau
    predictions = np.where(escalated, reference_probs.argmax(1), gate_probs.argmax(1))
    reference_right = reference_probs.argmax(1) == y
    return float(np.mean((predictions != y) & reference_right)), escalated


def calibrate(gate_val, reference_val, y_val, alpha, delta, grid, bound='kl'):
    """Fixed-sequence search. Returns the last threshold that passed, or None."""
    p_value = BOUNDS[bound]
    n = len(y_val)
    selected = None
    for tau in sorted(grid, reverse=True):
        risk_hat, _ = damage_rate(gate_val, reference_val, y_val, tau)
        if p_value(risk_hat, n, alpha) <= delta:
            selected = (float(tau), risk_hat)
        else:
            break
    return selected


def smallest_certifiable_alpha(risk_hat, n, delta, bound='kl', iterations=200):
    """Bisection for the tightest tolerance a given empirical risk supports."""
    p_value = BOUNDS[bound]
    low, high = risk_hat, 1.0
    for _ in range(iterations):
        mid = (low + high) / 2.0
        if p_value(risk_hat, n, mid) <= delta:
            high = mid
        else:
            low = mid
    return high
