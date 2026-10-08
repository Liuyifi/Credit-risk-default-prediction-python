"""Credit-risk metrics, rank-based operating views, and portfolio figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, score) -> float:
    frame = pd.DataFrame({"target": np.asarray(y_true), "score": np.asarray(score)})
    grouped = frame.groupby("score", sort=True, observed=True)["target"].agg(["sum", "count"])
    bad_total = grouped["sum"].sum()
    good_total = grouped["count"].sum() - bad_total
    if bad_total == 0 or good_total == 0:
        return np.nan
    cum_bad = grouped["sum"].cumsum() / bad_total
    cum_good = (grouped["count"] - grouped["sum"]).cumsum() / good_total
    return float(np.max(np.abs(cum_bad - cum_good)))


def calibration_slope_intercept(y_true, score) -> tuple[float, float]:
    p = np.clip(np.asarray(score), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logit, np.asarray(y_true))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def metric_row(y_true, score, model: str, sample: str, calibrated: bool) -> dict:
    slope, intercept = calibration_slope_intercept(y_true, score)
    auc = roc_auc_score(y_true, score)
    return {
        "model": model,
        "sample": sample,
        "calibrated": bool(calibrated),
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "average_precision": float(average_precision_score(y_true, score)),
        "ks": ks_statistic(y_true, score),
        "brier": float(brier_score_loss(y_true, score)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "event_rate": float(np.mean(y_true)),
        "rows": int(len(y_true)),
    }


def assign_risk_deciles(score, n_bins: int = 10) -> np.ndarray:
    """Assign decile 1 to highest predicted risk, without consulting outcomes."""
    p = pd.Series(np.asarray(score)).reset_index(drop=True)
    # first resolves ties deterministically from score and original row order only.
    ranks = p.rank(method="first", ascending=False)
    return pd.qcut(ranks, q=n_bins, labels=np.arange(1, n_bins + 1)).astype(int).to_numpy()


def risk_decile_table(y_true, score, model: str) -> pd.DataFrame:
    frame = pd.DataFrame({"target": np.asarray(y_true), "score": np.asarray(score)})
    frame["risk_decile"] = assign_risk_deciles(frame["score"])
    out = frame.groupby("risk_decile", observed=True).agg(
        borrowers=("target", "size"),
        bads=("target", "sum"),
        observed_bad_rate=("target", "mean"),
        mean_predicted_risk=("score", "mean"),
        min_predicted_risk=("score", "min"),
        max_predicted_risk=("score", "max"),
    ).reset_index()
    out["population_share"] = out["borrowers"] / out["borrowers"].sum()
    out["bad_capture_share"] = out["bads"] / out["bads"].sum()
    out["cumulative_bad_capture"] = out["bad_capture_share"].cumsum()
    out["lift_vs_portfolio"] = out["observed_bad_rate"] / frame["target"].mean()
    out.insert(0, "model", model)
    return out


def capacity_table(y_true, score, model: str, capacities=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Prespecified review capacities; selection is based only on predicted score."""
    frame = pd.DataFrame({"target": np.asarray(y_true), "score": np.asarray(score)})
    frame = frame.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    total_bads = frame["target"].sum()
    base_rate = frame["target"].mean()
    rows = []
    for capacity in capacities:
        reviewed_n = int(np.ceil(len(frame) * capacity))
        reviewed = frame.iloc[:reviewed_n]
        bads = int(reviewed["target"].sum())
        rows.append({
            "model": model,
            "capacity_share": capacity,
            "reviewed_borrowers": reviewed_n,
            "population_share": reviewed_n / len(frame),
            "captured_bads": bads,
            "observed_event_count": bads,
            "bad_capture_share": bads / total_bads,
            "event_capture": bads / total_bads,
            "cumulative_event_capture": bads / total_bads,
            "reviewed_bad_rate": reviewed["target"].mean(),
            "observed_event_rate": reviewed["target"].mean(),
            "lift_vs_portfolio": reviewed["target"].mean() / base_rate,
            "non_events_reviewed": reviewed_n - bads,
            "reviews_per_captured_event": reviewed_n / bads if bads else np.nan,
        })
    return pd.DataFrame(rows)


def coefficient_stability(coefficient_frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(coefficient_frames, ignore_index=True)
    out = combined.groupby("feature", observed=True)["coefficient"].agg(
        coefficient_mean="mean",
        coefficient_std="std",
        coefficient_min="min",
        coefficient_max="max",
        folds="count",
    ).reset_index()
    out["sign_stability"] = combined.groupby("feature")["coefficient"].apply(
        lambda x: max((x >= 0).mean(), (x <= 0).mean())
    ).to_numpy()
    out["abs_coefficient_mean"] = out["coefficient_mean"].abs()
    return out.sort_values("abs_coefficient_mean", ascending=False)


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def plot_target_distribution(target, output_path: Union[Path, str]) -> None:
    _style()
    counts = pd.Series(target).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=["No payment difficulty", "Payment difficulty"], y=counts.values, ax=ax, color="#31688e")
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}\n({value / counts.sum():.1%})", ha="center", va="bottom")
    ax.set(xlabel="Observed outcome", ylabel="Applications", title="Application target distribution")
    ax.set_ylim(0, counts.max() * 1.16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_roc_pr(y_true, predictions: Mapping[str, np.ndarray], output_path: Union[Path, str]) -> None:
    from sklearn.metrics import precision_recall_curve
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    prevalence = np.mean(y_true)
    for name, score in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        precision, recall, _ = precision_recall_curve(y_true, score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, score):.3f})")
        axes[1].plot(recall, precision, label=f"{name} (AP {average_precision_score(y_true, score):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[1].axhline(prevalence, linestyle="--", color="grey", linewidth=1, label=f"Baseline {prevalence:.3f}")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibration(y_true, predictions: Mapping[str, np.ndarray], output_path: Union[Path, str]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7, 6))
    highest = 0.0
    for name, score in predictions.items():
        observed, predicted = calibration_curve(y_true, score, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
        highest = max(highest, float(observed.max()), float(predicted.max()))
    limit = min(1.0, max(0.15, highest * 1.18))
    ax.plot([0, limit], [0, limit], "--", color="grey", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed bad rate", title="Final-test calibration by score decile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bad_rate_by_decile(deciles: pd.DataFrame, output_path: Union[Path, str]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=deciles, x="risk_decile", y="observed_bad_rate", hue="model", ax=ax)
    ax.set(xlabel="Risk decile (1 = highest predicted risk)", ylabel="Observed bad rate", title="Rank ordering on the locked final test")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cumulative_capture(deciles: pd.DataFrame, output_path: Union[Path, str]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, frame in deciles.groupby("model", sort=False):
        ax.plot(frame["population_share"].cumsum(), frame["cumulative_bad_capture"], marker="o", label=model)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random selection")
    ax.set(xlabel="Cumulative population reviewed", ylabel="Cumulative bads captured", title="Cumulative bad capture")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_capacity_tradeoff(capacity: pd.DataFrame, output_path: Union[Path, str]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, frame in capacity.groupby("model", sort=False):
        ax.plot(frame["capacity_share"], frame["bad_capture_share"], marker="o", linewidth=2, label=model)
    ax.plot([0, 0.20], [0, 0.20], "--", color="grey", linewidth=1, label="Random selection")
    ax.set(xlabel="Manual-review capacity", ylabel="Observed bads captured", title="Prespecified review-capacity trade-off")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
