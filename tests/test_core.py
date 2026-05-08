"""
Test Suite — Kaggle Credit Card Fraud Dataset Schema
=====================================================
Tests for loss functions, preprocessing pipeline, and API logic.
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loss_functions import (
    asymmetric_cost, expected_loss, optimal_threshold,
    prospect_theory_utility, scale_pos_weight,
    xgboost_asymmetric_objective, DEFAULT_C_FP, DEFAULT_C_FN,
)
from src.preprocessing import engineer_features, FEATURE_COLS


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_raw_row(**overrides) -> pd.DataFrame:
    """Build a minimal valid raw transaction row (Kaggle schema)."""
    row = {f"V{i}": 0.0 for i in range(1, 29)}
    row.update({"Amount": 100.0, "Time": 36000.0})  # 10:00 AM
    row.update(overrides)
    return pd.DataFrame([row])


# ─── Loss Function Tests ───────────────────────────────────────────────────────

class TestAsymmetricCost:

    def test_perfect_predictions_zero_cost(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert asymmetric_cost(y_true, y_pred) == 0.0

    def test_fn_costs_more_than_fp(self):
        y_true = np.array([1, 0])
        fn_cost = asymmetric_cost(y_true, np.array([0, 0]))  # 1 FN
        fp_cost = asymmetric_cost(y_true, np.array([1, 1]))  # 1 FP
        assert fn_cost > fp_cost, "FN should cost more than FP"

    def test_fn_fp_ratio_matches_costs(self):
        """1 FN should cost exactly C_FN/C_FP times more than 1 FP."""
        y_true = np.array([1, 0])
        fn_cost = asymmetric_cost(y_true, np.array([0, 0]), DEFAULT_C_FP, DEFAULT_C_FN)
        fp_cost = asymmetric_cost(y_true, np.array([1, 1]), DEFAULT_C_FP, DEFAULT_C_FN)
        assert fn_cost / fp_cost == pytest.approx(DEFAULT_C_FN / DEFAULT_C_FP)

    def test_cost_scales_linearly(self):
        y_true = np.array([1, 1, 0, 0])
        cost_2fn = asymmetric_cost(y_true, np.array([0, 0, 0, 0]))
        cost_1fn = asymmetric_cost(y_true, np.array([1, 0, 0, 0]))
        assert cost_2fn == pytest.approx(2 * cost_1fn)

    def test_custom_costs(self):
        y_true = np.array([1, 0])
        cost = asymmetric_cost(y_true, np.array([0, 0]), c_fp=5, c_fn=100)
        assert cost == pytest.approx(100.0)


class TestExpectedLoss:

    def test_perfect_model_zero_loss(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.0, 1.0])
        assert expected_loss(y_true, y_prob) == pytest.approx(0.0, abs=1e-6)

    def test_worse_model_higher_loss(self):
        y_true = np.array([0, 1, 0, 1])
        good = expected_loss(y_true, np.array([0.05, 0.95, 0.05, 0.95]))
        bad  = expected_loss(y_true, np.array([0.5,  0.5,  0.5,  0.5]))
        assert good < bad

    def test_fn_weighted_more_than_fp(self):
        """Missing fraud (FN) should dominate expected loss."""
        # 1 certain fraud missed vs 1 certain legit wrongly flagged
        y_true = np.array([1, 0])
        fn_loss = expected_loss(y_true, np.array([0.0, 0.0]))  # fraud prob=0 → FN
        fp_loss = expected_loss(y_true, np.array([1.0, 1.0]))  # fraud prob=1 → FP
        assert fn_loss > fp_loss


class TestProspectTheory:

    def test_loss_aversion_lambda(self):
        """v(-x) must be steeper than v(x): |v(-x)| > |v(x)|."""
        gain = prospect_theory_utility(np.array([10.0]))[0]
        loss = prospect_theory_utility(np.array([-10.0]))[0]
        assert abs(loss) > abs(gain)

    def test_gain_is_positive(self):
        assert prospect_theory_utility(np.array([5.0]))[0] > 0

    def test_loss_is_negative(self):
        assert prospect_theory_utility(np.array([-5.0]))[0] < 0

    def test_reference_point_is_zero(self):
        assert prospect_theory_utility(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-9)

    def test_lambda_approximately_2_25(self):
        """Loss aversion coefficient should produce ~2.25x asymmetry."""
        gain = prospect_theory_utility(np.array([10.0]))[0]
        loss = prospect_theory_utility(np.array([-10.0]))[0]
        # ratio |loss|/|gain| ≈ λ = 2.25 (exact when α=β=1; with curvature ≈ 2.25)
        assert abs(loss) / abs(gain) == pytest.approx(2.25, rel=0.01)


class TestOptimalThreshold:

    def test_theoretical_threshold(self):
        """τ* = C_FP / (C_FP + C_FN) = 5/25 = 0.20."""
        tau = DEFAULT_C_FP / (DEFAULT_C_FP + DEFAULT_C_FN)
        assert tau == pytest.approx(0.20, abs=1e-6)

    def test_lower_than_0_5_for_asymmetric_costs(self):
        """Empirical optimal should be below 0.5 with well-separated predictions."""
        np.random.seed(42)
        n = 5000
        y_true = np.zeros(n, dtype=int)
        # ~2% fraud
        fraud_idx = np.random.choice(n, size=100, replace=False)
        y_true[fraud_idx] = 1
        # Probabilities: fraud → [0.5, 0.95], legit → [0.0, 0.25]
        y_prob = np.where(y_true == 1,
                          np.random.uniform(0.5, 0.95, n),
                          np.random.uniform(0.0, 0.25, n))
        opt_t, _, _, _ = optimal_threshold(y_true, y_prob, c_fp=5, c_fn=20)
        assert opt_t < 0.5, f"Expected threshold < 0.5, got {opt_t:.3f}"


class TestScalePosWeight:

    def test_asymmetric_weight_larger_than_baseline(self):
        counts = {0: 284315, 1: 492}  # Real Kaggle class distribution
        baseline = counts[0] / counts[1]
        adjusted = scale_pos_weight(counts, c_fp=5, c_fn=20)
        assert adjusted > baseline

    def test_ratio_correct(self):
        counts = {0: 1000, 1: 100}
        adjusted = scale_pos_weight(counts, c_fp=5, c_fn=20)
        expected = (1000 / 100) * (20 / 5)
        assert adjusted == pytest.approx(expected)


# ─── Preprocessing Tests ───────────────────────────────────────────────────────

class TestPreprocessing:

    def test_amount_log_transform(self):
        df = make_raw_row(Amount=100.0)
        result = engineer_features(df)
        assert "Amount_log" in result.columns
        assert "Amount" not in result.columns
        assert result["Amount_log"].iloc[0] == pytest.approx(np.log1p(100.0))

    def test_amount_zero_safe(self):
        """log1p(0) = 0, should not error."""
        df = make_raw_row(Amount=0.0)
        result = engineer_features(df)
        assert result["Amount_log"].iloc[0] == pytest.approx(0.0)

    def test_cyclic_time_encoding_present(self):
        df = make_raw_row(Time=0.0)
        result = engineer_features(df)
        assert "Time_sin" in result.columns
        assert "Time_cos" in result.columns
        assert "Time" not in result.columns

    def test_cyclic_time_midnight(self):
        """Time=0 → hour=0 → sin=0, cos=1."""
        df = make_raw_row(Time=0.0)
        result = engineer_features(df)
        assert result["Time_sin"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["Time_cos"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_cyclic_time_6am(self):
        """Time = 6h*3600 = 21600s → hour=6 → sin=1, cos=0."""
        df = make_raw_row(Time=21600.0)
        result = engineer_features(df)
        assert result["Time_sin"].iloc[0] == pytest.approx(1.0, abs=1e-6)
        assert result["Time_cos"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_cyclic_wraps_correctly(self):
        """Time spanning >24h should wrap: 25h == 1h."""
        df_1h  = make_raw_row(Time=1 * 3600.0)
        df_25h = make_raw_row(Time=25 * 3600.0)
        r1  = engineer_features(df_1h)
        r25 = engineer_features(df_25h)
        assert r1["Time_sin"].iloc[0] == pytest.approx(r25["Time_sin"].iloc[0], abs=1e-6)
        assert r1["Time_cos"].iloc[0] == pytest.approx(r25["Time_cos"].iloc[0], abs=1e-6)

    def test_all_feature_columns_present(self):
        df = make_raw_row()
        result = engineer_features(df)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing feature: {col}"

    def test_feature_count(self):
        """Should have exactly 31 features: V1–V28 + Amount_log + Time_sin + Time_cos."""
        assert len(FEATURE_COLS) == 31

    def test_v_columns_unchanged(self):
        """V1–V28 values must pass through without modification."""
        values = {f"V{i}": float(i) for i in range(1, 29)}
        df = make_raw_row(**values)
        result = engineer_features(df)
        for i in range(1, 29):
            assert result[f"V{i}"].iloc[0] == pytest.approx(float(i))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])