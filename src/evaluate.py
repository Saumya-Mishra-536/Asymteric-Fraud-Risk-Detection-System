"""
Evaluation & Visualization
===========================
Comprehensive model evaluation including:
  - Standard classification metrics (precision, recall, F1, AUC)
  - Cost-based evaluation (CRITICAL for asymmetric systems)
  - Confusion matrix comparison
  - Threshold sensitivity analysis
  - SHAP feature importance
  - Scenario simulation (FP and FN cases)
  - Prospect Theory visualization
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
from src.loss_functions import (
    asymmetric_cost, expected_loss, optimal_threshold,
    prospect_theory_utility, DEFAULT_C_FP, DEFAULT_C_FN
)

# Output directory for plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "plots")


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   c_fp: float = DEFAULT_C_FP, c_fn: float = DEFAULT_C_FN,
                   label: str = "Model") -> dict:
    """
    Full evaluation of a fraud detection model.

    Returns dict with all metrics including cost-based measures.
    """
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    total_cost = asymmetric_cost(y_test, y_pred, c_fp, c_fn)
    exp_loss = expected_loss(y_test, y_prob, c_fp, c_fn)

    # Per-transaction expected cost
    n = len(y_test)

    metrics = {
        "label": label,
        "threshold": model.threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"],
        "Accuracy": report["accuracy"],
        "ROC_AUC": auc,
        "Avg_Precision": ap,
        "Total_Asymmetric_Cost": total_cost,
        "Expected_Loss": exp_loss,
        "Cost_Per_Transaction": total_cost / n,
        "Fraud_Caught_Rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "False_Alarm_Rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
    }

    print(f"\n{'='*55}")
    print(f"EVALUATION: {label} (threshold={model.threshold:.3f})")
    print(f"{'='*55}")
    print(f"  Confusion Matrix:")
    print(f"    TP={tp:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TN={tn:,}")
    print(f"  Precision:          {metrics['Precision']:.4f}")
    print(f"  Recall:             {metrics['Recall']:.4f}")
    print(f"  F1 Score:           {metrics['F1']:.4f}")
    print(f"  ROC-AUC:            {metrics['ROC_AUC']:.4f}")
    print(f"  Avg Precision (PR): {metrics['Avg_Precision']:.4f}")
    print(f"  ── Cost-Based ─────────────────────────────")
    print(f"  Total Asymmetric Cost: ${total_cost:,.1f}")
    print(f"  Expected Loss:         ${exp_loss:,.1f}")
    print(f"  Cost per Transaction:  ${metrics['Cost_Per_Transaction']:.4f}")
    print(f"  Fraud Caught Rate:     {metrics['Fraud_Caught_Rate']:.2%}")
    print(f"  False Alarm Rate:      {metrics['False_Alarm_Rate']:.4f}")

    return metrics


def compare_models(sym_metrics: dict, asym_metrics: dict):
    """Print side-by-side comparison of symmetric vs asymmetric model."""
    print(f"\n{'='*65}")
    print(f"{'MODEL COMPARISON':^65}")
    print(f"{'='*65}")
    print(f"{'Metric':<30} {'Symmetric':>15} {'Asymmetric':>15}")
    print(f"{'-'*65}")

    keys = ["Recall", "Precision", "F1", "ROC_AUC",
            "Total_Asymmetric_Cost", "Expected_Loss", "Fraud_Caught_Rate"]
    for k in keys:
        s = sym_metrics[k]
        a = asym_metrics[k]
        flag = " ◄" if (k in ["Total_Asymmetric_Cost", "Expected_Loss"] and a < s) \
               else (" ◄" if k not in ["Total_Asymmetric_Cost", "Expected_Loss"] and a > s else "")
        if isinstance(s, float):
            print(f"  {k:<28} {s:>14.4f} {a:>14.4f}{flag}")
        else:
            print(f"  {k:<28} {s:>15} {a:>15}{flag}")

    cost_reduction = (sym_metrics["Total_Asymmetric_Cost"] - asym_metrics["Total_Asymmetric_Cost"])
    pct = cost_reduction / sym_metrics["Total_Asymmetric_Cost"] * 100
    print(f"\n  Cost Reduction: ${cost_reduction:,.1f} ({pct:.1f}%)")
    print(f"  Threshold shift: {sym_metrics['threshold']:.2f} → {asym_metrics['threshold']:.2f}")


def plot_confusion_matrices(sym_metrics: dict, asym_metrics: dict, save: bool = True):
    """Side-by-side confusion matrix comparison."""
    ensure_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix: Symmetric vs Asymmetric", fontsize=14, fontweight="bold")

    for ax, m, title in zip(axes,
                             [sym_metrics, asym_metrics],
                             ["Symmetric Model\n(threshold=0.50)",
                              f"Asymmetric Model\n(threshold={asym_metrics['threshold']:.2f})"]):
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Predicted: Legit", "Predicted: Fraud"],
                    yticklabels=["Actual: Legit", "Actual: Fraud"])
        ax.set_title(title, fontsize=11)
        # Annotate cost impact
        fp_cost = m["FP"] * DEFAULT_C_FP
        fn_cost = m["FN"] * DEFAULT_C_FN
        ax.set_xlabel(f"FP Cost: ${fp_cost:,.0f}  |  FN Cost: ${fn_cost:,.0f}\n"
                      f"Total: ${fp_cost + fn_cost:,.0f}", fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "confusion_matrix_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {path}")
    return fig


def plot_threshold_tuning(y_val: np.ndarray, sym_proba: np.ndarray,
                           asym_proba: np.ndarray, save: bool = True):
    """Plot cost vs threshold for both models."""
    ensure_plots_dir()
    _, _, thresholds, sym_costs = optimal_threshold(y_val, sym_proba, DEFAULT_C_FP, DEFAULT_C_FN)
    _, _, _, asym_costs = optimal_threshold(y_val, asym_proba, DEFAULT_C_FP, DEFAULT_C_FN)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Threshold Tuning: Cost vs Decision Threshold", fontsize=13, fontweight="bold")

    theoretical_t = DEFAULT_C_FP / (DEFAULT_C_FP + DEFAULT_C_FN)

    for ax, costs, proba, title, color in zip(
        axes,
        [sym_costs, asym_costs],
        [sym_proba, asym_proba],
        ["Symmetric Model", "Asymmetric Model"],
        ["steelblue", "coral"]
    ):
        ax.plot(thresholds, costs, color=color, linewidth=2)
        opt_t = thresholds[np.argmin(costs)]
        ax.axvline(opt_t, color="green", linestyle="--", label=f"Empirical opt: {opt_t:.2f}")
        ax.axvline(0.5, color="gray", linestyle=":", label="Default (0.5)")
        ax.axvline(theoretical_t, color="orange", linestyle="-.",
                   label=f"Theoretical opt: {theoretical_t:.2f}")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Total Asymmetric Cost ($)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "threshold_tuning.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {path}")
    return fig


def plot_prospect_theory_curve(save: bool = True):
    """Visualize the Prospect Theory value function."""
    ensure_plots_dir()
    x = np.linspace(-100, 100, 1000)
    pt_value = prospect_theory_utility(x)
    linear_value = x  # Symmetric reference

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: PT vs Linear ---
    ax = axes[0]
    ax.plot(x, linear_value, "--", color="gray", label="Symmetric (Linear)", linewidth=1.5)
    ax.plot(x[x >= 0], pt_value[x >= 0], color="green", linewidth=2.5, label="PT: Gains (concave)")
    ax.plot(x[x < 0], pt_value[x < 0], color="red", linewidth=2.5, label="PT: Losses (convex, steep)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.fill_between(x[x < 0], pt_value[x < 0], linear_value[x < 0],
                    alpha=0.1, color="red", label="Loss aversion gap")
    ax.set_xlabel("Outcome (relative to reference point)", fontsize=11)
    ax.set_ylabel("Subjective Value / Utility", fontsize=11)
    ax.set_title("Prospect Theory Value Function\n(Kahneman & Tversky, 1979)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.text(-80, 30, "λ = 2.25\n(Loss Aversion)", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- Right: FP vs FN cost impact ---
    ax2 = axes[1]
    scenarios = ["Miss $100 fraud\n(FN, 1 case)", "Block $100 legit\n(FP, 1 case)"]
    sym_impact = [-100, 100]  # Symmetric: equal magnitude
    asym_impact = [-100 * 2.25, 100]  # Asymmetric: loss weighted by λ
    colors = ["red", "steelblue"]

    x_pos = np.arange(len(scenarios))
    width = 0.3
    ax2.bar(x_pos - width/2, sym_impact, width, label="Symmetric model impact", color="gray", alpha=0.7)
    ax2.bar(x_pos + width/2, asym_impact, width, label="Asymmetric model impact", color=colors, alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Subjective Cost Impact", fontsize=11)
    ax2.set_title("Symmetric vs Asymmetric Cost Perception\nApplied to Fraud Detection", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "prospect_theory_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {path}")
    return fig


def plot_roc_pr_curves(sym_model, asym_model, X_test: np.ndarray,
                        y_test: np.ndarray, save: bool = True):
    """ROC and Precision-Recall curves for both models."""
    ensure_plots_dir()
    sym_proba = sym_model.predict_proba(X_test)
    asym_proba = asym_model.predict_proba(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    ax = axes[0]
    for proba, label, color in [
        (sym_proba, f"Symmetric (AUC={roc_auc_score(y_test, sym_proba):.3f})", "steelblue"),
        (asym_proba, f"Asymmetric (AUC={roc_auc_score(y_test, asym_proba):.3f})", "coral"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=label, linewidth=2, color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    # PR
    ax2 = axes[1]
    for proba, label, color in [
        (sym_proba, f"Symmetric (AP={average_precision_score(y_test, sym_proba):.3f})", "steelblue"),
        (asym_proba, f"Asymmetric (AP={average_precision_score(y_test, asym_proba):.3f})", "coral"),
    ]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax2.plot(rec, prec, label=label, linewidth=2, color=color)
    baseline = y_test.mean()
    ax2.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve Comparison")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "roc_pr_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {path}")
    return fig


def shap_analysis(asym_model, X_test: np.ndarray, feature_names: list,
                   n_samples: int = 500, save: bool = True):
    """
    SHAP analysis for asymmetric model explainability.
    Uses TreeExplainer (exact, fast for XGBoost trees).
    """
    ensure_plots_dir()
    print("[SHAP] Computing SHAP values (this may take ~30s)...")

    # Sample for speed
    idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[idx]

    import xgboost as xgb
    explainer = shap.TreeExplainer(asym_model.model)
    dmat = xgb.DMatrix(X_sample, feature_names=feature_names)
    shap_values = explainer.shap_values(dmat)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, plot_type="dot")
    plt.title("SHAP Feature Importance — Asymmetric Model\n"
              "(Red = pushes toward fraud prediction, Blue = toward legit)", fontsize=12)
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {path}")
    plt.close()

    # Bar plot of mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values("Mean |SHAP|", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bars = ax2.barh(importance_df["Feature"], importance_df["Mean |SHAP|"], color="coral")
    ax2.set_xlabel("Mean |SHAP Value|")
    ax2.set_title("Feature Importance by Mean |SHAP|\nAsymmetric Fraud Model", fontsize=12)
    ax2.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "shap_bar.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved: {path}")
    plt.close()

    return shap_values, importance_df


def simulate_scenarios(sym_model, asym_model, X_test: np.ndarray,
                        y_test: np.ndarray):
    """
    Scenario Simulation:
    1. Small anomaly → legitimate transaction that looks slightly unusual
       → Symmetric model may block it (FP). Asymmetric may pass it.
    2. Large fraud → clearly fraudulent transaction
       → Symmetric model may miss it. Asymmetric catches it.

    Shows the real-world impact under both models.
    """
    print(f"\n{'='*65}")
    print(f"{'SCENARIO SIMULATION':^65}")
    print(f"{'='*65}")

    sym_proba = sym_model.predict_proba(X_test)
    asym_proba = asym_model.predict_proba(X_test)

    # --- Scenario A: FP Risk (legit transactions blocked) ---
    legit_idx = np.where(y_test == 0)[0]
    # Find legit transactions with borderline fraud probability (near 0.5)
    borderline = legit_idx[np.argsort(np.abs(sym_proba[legit_idx] - 0.5))[:5]]

    print("\n📌 SCENARIO A: Borderline Legitimate Transactions")
    print("   (Risk: Symmetric model blocks them → bad UX)")
    print(f"   {'TxID':<8} {'True Label':<12} {'Sym Prob':>10} {'Sym Decision':>14} "
          f"{'Asym Prob':>10} {'Asym Decision':>14}")
    print("   " + "-"*70)
    for i in borderline:
        sym_d = "BLOCK 🚫" if sym_proba[i] >= sym_model.threshold else "ALLOW ✅"
        asym_d = "BLOCK 🚫" if asym_proba[i] >= asym_model.threshold else "ALLOW ✅"
        print(f"   {i:<8} {'LEGIT':<12} {sym_proba[i]:>10.3f} {sym_d:>14} "
              f"{asym_proba[i]:>10.3f} {asym_d:>14}")

    # --- Scenario B: FN Risk (fraud missed) ---
    fraud_idx = np.where(y_test == 1)[0]
    # Find fraud transactions with low predicted probability (model uncertain)
    sneaky = fraud_idx[np.argsort(sym_proba[fraud_idx])[:5]]

    print("\n🚨 SCENARIO B: Low-Probability Fraud Transactions")
    print("   (Risk: Symmetric model misses them → financial loss)")
    print(f"   {'TxID':<8} {'True Label':<12} {'Sym Prob':>10} {'Sym Decision':>14} "
          f"{'Asym Prob':>10} {'Asym Decision':>14}")
    print("   " + "-"*70)
    for i in sneaky:
        sym_d = "BLOCK 🚫" if sym_proba[i] >= sym_model.threshold else "MISS ❌"
        asym_d = "BLOCK 🚫" if asym_proba[i] >= asym_model.threshold else "MISS ❌"
        print(f"   {i:<8} {'FRAUD':<12} {sym_proba[i]:>10.3f} {sym_d:>14} "
              f"{asym_proba[i]:>10.3f} {asym_d:>14}")

    print(f"\n  Key Insight:")
    print(f"  Asymmetric threshold ({asym_model.threshold:.2f}) vs Symmetric ({sym_model.threshold:.2f})")
    print(f"  Lower threshold → catches more fraud at acceptable FP increase")
    print(f"  Because: C_FN ($20) >> C_FP ($5) → worth accepting more FPs to reduce FNs")


def run_full_evaluation(sym_model, asym_model, data: dict,
                         c_fp=DEFAULT_C_FP, c_fn=DEFAULT_C_FN):
    """
    Master evaluation function. Runs all metrics and generates all plots.
    """
    X_test = data["X_test"]
    X_val = data["X_val"]
    y_test = data["y_test"]
    y_val = data["y_val"]
    feature_names = data["feature_names"]

    # --- Metrics ---
    sym_metrics = evaluate_model(sym_model, X_test, y_test, c_fp, c_fn, "Symmetric")
    asym_metrics = evaluate_model(asym_model, X_test, y_test, c_fp, c_fn, "Asymmetric")
    compare_models(sym_metrics, asym_metrics)

    # --- Scenarios ---
    simulate_scenarios(sym_model, asym_model, X_test, y_test)

    # --- Plots ---
    print("\n[Evaluation] Generating plots...")
    plot_confusion_matrices(sym_metrics, asym_metrics)
    plot_threshold_tuning(y_val, sym_model.predict_proba(X_val),
                           asym_model.predict_proba(X_val))
    plot_prospect_theory_curve()
    plot_roc_pr_curves(sym_model, asym_model, X_test, y_test)
    shap_analysis(asym_model, X_test, feature_names)

    print(f"\n[Evaluation] All plots saved to: {PLOTS_DIR}")
    return sym_metrics, asym_metrics


if __name__ == "__main__":
    from src.train import train
    result = train()
    run_full_evaluation(
        result["symmetric_model"],
        result["asymmetric_model"],
        result["data"]
    )