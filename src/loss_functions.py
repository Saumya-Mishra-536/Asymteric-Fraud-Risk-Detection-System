"""
Asymmetric Loss Functions & Prospect Theory Utilities
======================================================

Core Idea (Kahneman & Tversky, 1979):
    People feel losses more acutely than equivalent gains.
    Value function: v(x) = x^α  if x >= 0
                          = -λ(-x)^β  if x < 0
    where λ > 1 is the loss aversion coefficient.

Applied to Fraud Detection:
    - False Negative (missed fraud) = LOSS → penalized by λ
    - False Positive (wrongly blocked legit) = GAIN → smaller penalty
    - Asymmetric cost: C_FN / C_FP = λ (typically 4x to 20x)

Mathematical Formulation:
    Expected Cost = C_FP * FP + C_FN * FN
    Standard cross-entropy treats all errors equally.
    Asymmetric cross-entropy weights positive class by scale_pos_weight.
    For XGBoost, we provide a custom objective function.
"""

import numpy as np
from typing import Tuple


# --- Cost Constants (Banking Defaults) ---
# C_FP: Cost of blocking a legitimate transaction (churn risk, support cost)
# C_FN: Cost of missing a fraud (chargeback + investigation + reputational damage)
DEFAULT_C_FP = 5.0    # $5 equivalent (annoyance, support ticket)
DEFAULT_C_FN = 20.0   # $20 equivalent (chargeback median ~$20 after fees)

# Prospect Theory parameters
ALPHA = 0.88    # Gain curvature (Tversky & Kahneman 1992)
BETA = 0.88     # Loss curvature
LAMBDA = 2.25   # Loss aversion coefficient (canonical value)


def asymmetric_cost(y_true: np.ndarray, y_pred_binary: np.ndarray,
                    c_fp: float = DEFAULT_C_FP,
                    c_fn: float = DEFAULT_C_FN) -> float:
    """
    Compute total asymmetric cost of a prediction set.

    Cost(y, ŷ) = C_FP * Σ[FP] + C_FN * Σ[FN]

    Args:
        y_true: Ground truth labels (0/1)
        y_pred_binary: Predicted labels at chosen threshold
        c_fp: Cost per false positive
        c_fn: Cost per false negative

    Returns:
        Total cost (float)
    """
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    return c_fp * fp + c_fn * fn


def expected_loss(y_true: np.ndarray, y_prob: np.ndarray,
                  c_fp: float = DEFAULT_C_FP,
                  c_fn: float = DEFAULT_C_FN) -> float:
    """
    Expected cost using probabilistic predictions (soft version).

    E[Cost] = C_FP * Σ[p_i * (1-y_i)] + C_FN * Σ[(1-p_i) * y_i]

    This is differentiable and used during threshold selection.

    Args:
        y_true: Ground truth labels (0/1)
        y_prob: Predicted fraud probabilities [0, 1]
        c_fp: Cost per false positive
        c_fn: Cost per false negative

    Returns:
        Expected total cost
    """
    fp_cost = c_fp * np.sum(y_prob * (1 - y_true))
    fn_cost = c_fn * np.sum((1 - y_prob) * y_true)
    return fp_cost + fn_cost


def xgboost_asymmetric_objective(c_fp: float = DEFAULT_C_FP,
                                  c_fn: float = DEFAULT_C_FN):
    """
    Factory: returns custom XGBoost objective function with asymmetric costs.

    XGBoost requires: objective(y_pred, dtrain) → (grad, hess)
    We derive grad/hess from the weighted log-loss:

        L = -[C_FN * y * log(σ(p)) + C_FP * (1-y) * log(1-σ(p))]

    where σ(p) is sigmoid of raw score p.

    Gradient:  dL/dp = C_FN*(σ-1)*y + C_FP*σ*(1-y)  [simplified]
    Hessian:   d²L/dp² = σ*(1-σ) * (C_FN*y + C_FP*(1-y))

    Args:
        c_fp: Cost weight for false positives
        c_fn: Cost weight for false negatives

    Returns:
        Callable objective function for XGBoost
    """
    def objective(y_pred: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dtrain.get_label()
        # Sigmoid of raw predictions
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # Asymmetric gradient
        grad = -c_fn * y_true * (1 - p) + c_fp * (1 - y_true) * p
        # Hessian (second derivative) — must be positive
        hess = p * (1 - p) * (c_fn * y_true + c_fp * (1 - y_true))
        return grad, hess

    return objective


def prospect_theory_utility(x: np.ndarray,
                             alpha: float = ALPHA,
                             beta: float = BETA,
                             lam: float = LAMBDA) -> np.ndarray:
    """
    Prospect Theory value function (Kahneman & Tversky, 1979).

    v(x) =  x^α              for x >= 0  (gains, concave)
           -λ * (-x)^β       for x < 0   (losses, convex, steeper)

    Key properties:
    - Gains and losses measured relative to reference point (x=0)
    - Loss aversion: v(-x) > |v(x)| by factor λ ≈ 2.25
    - Diminishing sensitivity: marginal impact decreases away from reference

    Args:
        x: Outcome values (positive = gains, negative = losses)
        alpha: Gain curvature parameter [0, 1]
        beta: Loss curvature parameter [0, 1]
        lam: Loss aversion coefficient (>1)

    Returns:
        Subjective utility values
    """
    x = np.asarray(x, dtype=float)
    v = np.where(
        x >= 0,
        np.power(np.abs(x), alpha),
        -lam * np.power(np.abs(x), beta)
    )
    return v


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                       c_fp: float = DEFAULT_C_FP,
                       c_fn: float = DEFAULT_C_FN,
                       n_thresholds: int = 1000) -> Tuple[float, float]:
    """
    Find decision threshold that minimizes total asymmetric cost.

    Standard approach: threshold = 0.5 (symmetric assumption)
    Asymmetric approach: threshold = C_FP / (C_FP + C_FN)
                                   = 5 / (5 + 20) = 0.20

    This means: flag as fraud if P(fraud) > 0.20 (lower bar due to high FN cost)

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        c_fp: Cost per false positive
        c_fn: Cost per false negative
        n_thresholds: Number of threshold candidates to evaluate

    Returns:
        (optimal_threshold, minimum_cost) tuple
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        costs.append(asymmetric_cost(y_true, y_pred, c_fp, c_fn))

    optimal_t = thresholds[np.argmin(costs)]
    min_cost = min(costs)

    # Theoretical optimal: c_fp / (c_fp + c_fn)
    theoretical_t = c_fp / (c_fp + c_fn)

    print(f"[Threshold] Theoretical optimal: {theoretical_t:.3f}")
    print(f"[Threshold] Empirical optimal:   {optimal_t:.3f} (cost={min_cost:,.1f})")

    return optimal_t, min_cost, thresholds, costs


def scale_pos_weight(class_counts: dict,
                      c_fp: float = DEFAULT_C_FP,
                      c_fn: float = DEFAULT_C_FN) -> float:
    """
    Compute XGBoost scale_pos_weight incorporating both class imbalance
    and asymmetric costs.

    Standard scale_pos_weight = n_negative / n_positive
    Asymmetric: multiply by (c_fn / c_fp) to amplify fraud class further

    Args:
        class_counts: {0: n_legit, 1: n_fraud}
        c_fp: Cost per false positive
        c_fn: Cost per false negative

    Returns:
        Adjusted scale_pos_weight for XGBoost
    """
    base = class_counts[0] / class_counts[1]
    adjusted = base * (c_fn / c_fp)
    print(f"[Loss] Base scale_pos_weight (imbalance): {base:.2f}")
    print(f"[Loss] Cost-adjusted scale_pos_weight:    {adjusted:.2f}")
    return adjusted