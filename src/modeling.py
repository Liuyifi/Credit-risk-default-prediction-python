"""Fold-safe development selection and locked final fitting for two models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .validation import capacity_table, coefficient_stability, ks_statistic


RANDOM_STATE = 42
C_GRID = (0.01, 0.1, 1.0, 10.0)
CAPACITIES = (0.05, 0.10, 0.20)

LIGHTGBM_CANDIDATES = (
    {"num_leaves": 15, "max_depth": 5, "min_child_samples": 100, "learning_rate": 0.04,
     "colsample_bytree": 0.8, "subsample": 0.8, "reg_alpha": 0.0, "reg_lambda": 3.0},
    {"num_leaves": 31, "max_depth": 8, "min_child_samples": 100, "learning_rate": 0.03,
     "colsample_bytree": 0.8, "subsample": 0.8, "reg_alpha": 0.2, "reg_lambda": 5.0},
    {"num_leaves": 31, "max_depth": -1, "min_child_samples": 200, "learning_rate": 0.03,
     "colsample_bytree": 0.7, "subsample": 0.9, "reg_alpha": 0.5, "reg_lambda": 8.0},
    {"num_leaves": 15, "max_depth": 8, "min_child_samples": 50, "learning_rate": 0.05,
     "colsample_bytree": 1.0, "subsample": 0.8, "reg_alpha": 0.2, "reg_lambda": 5.0},
)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns using training-fold quantiles only."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.lower_bounds_))]
        return np.asarray(input_features, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Pool categories with support below a training-fold threshold."""

    def __init__(self, min_count=200):
        self.min_count = min_count

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).fillna("__MISSING__").astype(str)
        self.frequent_ = [set(frame[col].value_counts()[lambda s: s >= self.min_count].index) for col in frame]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).fillna("__MISSING__").astype(str)
        for idx, col in enumerate(frame):
            frame[col] = frame[col].where(frame[col].isin(self.frequent_[idx]), "__RARE__")
        return frame.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.frequent_))]
        return np.asarray(input_features, dtype=object)


@dataclass
class DevelopmentResult:
    split_ids: Dict[str, np.ndarray]
    selected_c: float
    selected_lgbm_params: dict
    selected_lgbm_iterations: int
    calibration: Dict[str, dict]
    oof_predictions: Dict[str, np.ndarray]
    cv_summary: pd.DataFrame
    logistic_stability: pd.DataFrame


def stratified_development_split(ids, target, test_size=0.20, random_state=RANDOM_STATE):
    dev_ids, test_ids = train_test_split(
        np.asarray(ids), test_size=test_size, stratify=np.asarray(target), random_state=random_state
    )
    return {"development": np.sort(dev_ids), "final_test": np.sort(test_ids)}


def predictor_columns(frame: pd.DataFrame, application_only=False) -> List[str]:
    cols = [c for c in frame.columns if c not in {"SK_ID_CURR", "TARGET"}]
    if application_only:
        cols = [c for c in cols if c.startswith("APP_")]
    return cols


def column_types(frame: pd.DataFrame, columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    categorical = [c for c in columns if frame[c].dtype == "object" or str(frame[c].dtype).startswith("category")]
    numeric = [c for c in columns if c not in categorical]
    return numeric, categorical


def make_preprocessor(frame: pd.DataFrame, columns: Sequence[str], model_kind: str) -> ColumnTransformer:
    numeric, categorical = column_types(frame, columns)
    numeric_steps = [
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if model_kind == "logistic":
        numeric_steps.append(("scale", StandardScaler()))
        cat_encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True)
    else:
        # Nominal labels must not acquire an arbitrary numeric order in tree splits.
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("rare", RareCategoryGrouper(min_count=200)),
        ("encode", cat_encoder),
    ])
    return ColumnTransformer(
        [("numeric", Pipeline(numeric_steps), numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.3 if model_kind == "logistic" else 0.0,
        verbose_feature_names_out=True,
    )


def _fold_metric_rows(y, score, model, candidate, fold) -> List[dict]:
    auc = roc_auc_score(y, score)
    rows = [{
        "model": model,
        "candidate": candidate,
        "fold": fold,
        "metric": "roc_auc",
        "value": auc,
    }, {
        "model": model,
        "candidate": candidate,
        "fold": fold,
        "metric": "gini",
        "value": 2 * auc - 1,
    }, {
        "model": model,
        "candidate": candidate,
        "fold": fold,
        "metric": "average_precision",
        "value": average_precision_score(y, score),
    }, {
        "model": model,
        "candidate": candidate,
        "fold": fold,
        "metric": "ks",
        "value": ks_statistic(y, score),
    }, {
        "model": model,
        "candidate": candidate,
        "fold": fold,
        "metric": "brier",
        "value": brier_score_loss(y, score),
    }]
    cap = capacity_table(y, score, model)
    for row in cap.itertuples(index=False):
        rows.append({
            "model": model,
            "candidate": candidate,
            "fold": fold,
            "metric": f"capture_at_{int(row.capacity_share * 100)}pct",
            "value": row.bad_capture_share,
        })
    return rows


def _summary(metric_rows: list) -> pd.DataFrame:
    detail = pd.DataFrame(metric_rows)
    return detail.groupby(["model", "candidate", "metric"], observed=True)["value"].agg(
        mean="mean", std="std", min="min", max="max", folds="count"
    ).reset_index()


def cross_validate_logistic(
    frame: pd.DataFrame,
    columns: Sequence[str],
    folds: StratifiedKFold,
    c_grid=C_GRID,
    model_name="Logistic",
):
    y = frame["TARGET"].to_numpy()
    predictions = {c: np.zeros(len(frame)) for c in c_grid}
    metric_rows = []
    coefficients = {c: [] for c in c_grid}
    for fold, (train_idx, valid_idx) in enumerate(folds.split(frame, y), 1):
        train = frame.iloc[train_idx]
        valid = frame.iloc[valid_idx]
        preprocessor = make_preprocessor(train, columns, "logistic")
        x_train = preprocessor.fit_transform(train[list(columns)])
        x_valid = preprocessor.transform(valid[list(columns)])
        names = preprocessor.get_feature_names_out()
        for c in c_grid:
            model = LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=1000)
            model.fit(x_train, train["TARGET"])
            score = model.predict_proba(x_valid)[:, 1]
            predictions[c][valid_idx] = score
            metric_rows.extend(_fold_metric_rows(valid["TARGET"], score, model_name, f"C={c:g}", fold))
            coefficients[c].append(pd.DataFrame({
                "feature": names,
                "coefficient": model.coef_[0],
                "fold": fold,
            }))
    summary = _summary(metric_rows)
    aucs = summary[summary["metric"].eq("roc_auc")].sort_values(
        ["mean", "std"], ascending=[False, True]
    )
    chosen_label = aucs.iloc[0]["candidate"]
    chosen_c = float(str(chosen_label).split("=")[1])
    stability = coefficient_stability(coefficients[chosen_c])
    stability["odds_ratio"] = np.exp(stability["coefficient_mean"])
    stability["source_family"] = stability["feature"].str.extract(r"(?:numeric|categorical)__(APP|BUREAU|BB|PREV|INST)_", expand=False).fillna("Encoded")
    return chosen_c, predictions[chosen_c], summary, stability


def _lgbm_model(params: dict, random_state=RANDOM_STATE, n_estimators=800):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        subsample_freq=1,
        **params,
    )


def cross_validate_lightgbm(frame, columns, folds, candidates=LIGHTGBM_CANDIDATES):
    y = frame["TARGET"].to_numpy()
    predictions = {i: np.zeros(len(frame)) for i in range(len(candidates))}
    metric_rows, best_iterations = [], {i: [] for i in range(len(candidates))}
    for fold, (train_idx, valid_idx) in enumerate(folds.split(frame, y), 1):
        train = frame.iloc[train_idx]
        valid = frame.iloc[valid_idx]
        preprocessor = make_preprocessor(train, columns, "lightgbm")
        x_train = preprocessor.fit_transform(train[list(columns)])
        x_valid = preprocessor.transform(valid[list(columns)])
        for idx, params in enumerate(candidates):
            model = _lgbm_model(params)
            model.fit(
                x_train,
                train["TARGET"],
                eval_set=[(x_valid, valid["TARGET"])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            score = model.predict_proba(x_valid, num_iteration=model.best_iteration_)[:, 1]
            predictions[idx][valid_idx] = score
            best_iterations[idx].append(int(model.best_iteration_))
            metric_rows.extend(_fold_metric_rows(valid["TARGET"], score, "LightGBM", f"candidate_{idx + 1}", fold))
    summary = _summary(metric_rows)
    aucs = summary[summary["metric"].eq("roc_auc")].sort_values(
        ["mean", "std"], ascending=[False, True]
    )
    chosen_label = aucs.iloc[0]["candidate"]
    chosen_idx = int(str(chosen_label).split("_")[1]) - 1
    selected_iterations = int(np.median(best_iterations[chosen_idx]))
    return chosen_idx, predictions[chosen_idx], selected_iterations, summary


def cross_fit_sigmoid(y, raw_score, folds: StratifiedKFold, minimum_brier_gain=1e-4):
    y = np.asarray(y)
    raw = np.clip(np.asarray(raw_score), 1e-6, 1 - 1e-6)
    calibrated = np.zeros(len(y))
    x = logit(raw).reshape(-1, 1)
    for train_idx, valid_idx in folds.split(x, y):
        calibrator = LogisticRegression(C=1e6, solver="lbfgs")
        calibrator.fit(x[train_idx], y[train_idx])
        calibrated[valid_idx] = calibrator.predict_proba(x[valid_idx])[:, 1]
    raw_brier = brier_score_loss(y, raw)
    calibrated_brier = brier_score_loss(y, calibrated)
    use_calibration = raw_brier - calibrated_brier > minimum_brier_gain
    final_calibrator = LogisticRegression(C=1e6, solver="lbfgs").fit(x, y)
    return {
        "use_calibration": bool(use_calibration),
        "raw_brier": float(raw_brier),
        "calibrated_brier": float(calibrated_brier),
        "brier_gain": float(raw_brier - calibrated_brier),
        "coefficient": float(final_calibrator.coef_[0, 0]),
        "intercept": float(final_calibrator.intercept_[0]),
        "cross_fitted_score": calibrated if use_calibration else raw,
    }


def apply_sigmoid(score, calibration: dict):
    p = np.clip(np.asarray(score), 1e-6, 1 - 1e-6)
    if not calibration["use_calibration"]:
        return p
    return expit(calibration["intercept"] + calibration["coefficient"] * logit(p))


def run_development(frame: pd.DataFrame) -> DevelopmentResult:
    split_ids = stratified_development_split(frame["SK_ID_CURR"], frame["TARGET"])
    dev = frame[frame["SK_ID_CURR"].isin(split_ids["development"])].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_columns = predictor_columns(dev)
    app_columns = predictor_columns(dev, application_only=True)

    app_c, app_oof, app_summary, _ = cross_validate_logistic(
        dev, app_columns, folds, model_name="Application-only Logistic"
    )
    selected_c, logistic_oof, logistic_summary, stability = cross_validate_logistic(
        dev, all_columns, folds, model_name="Logistic"
    )
    selected_lgbm, lgbm_oof, iterations, lgbm_summary = cross_validate_lightgbm(
        dev, all_columns, folds
    )
    calibration = {
        "Logistic": cross_fit_sigmoid(dev["TARGET"], logistic_oof, folds),
        "LightGBM": cross_fit_sigmoid(dev["TARGET"], lgbm_oof, folds),
    }
    oof_predictions = {
        "Application-only Logistic": app_oof,
        "Logistic": calibration["Logistic"]["cross_fitted_score"],
        "LightGBM": calibration["LightGBM"]["cross_fitted_score"],
    }
    cv_summary = pd.concat([app_summary, logistic_summary, lgbm_summary], ignore_index=True)
    cv_summary["selected"] = (
        ((cv_summary["model"] == "Application-only Logistic") & (cv_summary["candidate"] == f"C={app_c:g}"))
        | ((cv_summary["model"] == "Logistic") & (cv_summary["candidate"] == f"C={selected_c:g}"))
        | ((cv_summary["model"] == "LightGBM") & (cv_summary["candidate"] == f"candidate_{selected_lgbm + 1}"))
    )
    return DevelopmentResult(
        split_ids=split_ids,
        selected_c=selected_c,
        selected_lgbm_params=dict(LIGHTGBM_CANDIDATES[selected_lgbm]),
        selected_lgbm_iterations=iterations,
        calibration=calibration,
        oof_predictions=oof_predictions,
        cv_summary=cv_summary,
        logistic_stability=stability,
    )


def fit_locked_models(frame: pd.DataFrame, result: DevelopmentResult):
    dev = frame[frame["SK_ID_CURR"].isin(result.split_ids["development"])].reset_index(drop=True)
    final = frame[frame["SK_ID_CURR"].isin(result.split_ids["final_test"])].reset_index(drop=True)
    columns = predictor_columns(dev)

    logistic_pre = make_preprocessor(dev, columns, "logistic")
    x_dev_log = logistic_pre.fit_transform(dev[columns])
    x_final_log = logistic_pre.transform(final[columns])
    logistic = LogisticRegression(C=result.selected_c, penalty="l2", solver="liblinear", max_iter=1000)
    logistic.fit(x_dev_log, dev["TARGET"])
    logistic_raw = logistic.predict_proba(x_final_log)[:, 1]

    lgbm_pre = make_preprocessor(dev, columns, "lightgbm")
    x_dev_lgbm = lgbm_pre.fit_transform(dev[columns])
    x_final_lgbm = lgbm_pre.transform(final[columns])
    challenger = _lgbm_model(result.selected_lgbm_params, n_estimators=result.selected_lgbm_iterations)
    challenger.fit(x_dev_lgbm, dev["TARGET"])
    lgbm_raw = challenger.predict_proba(x_final_lgbm)[:, 1]

    predictions = {
        "Logistic": apply_sigmoid(logistic_raw, result.calibration["Logistic"]),
        "LightGBM": apply_sigmoid(lgbm_raw, result.calibration["LightGBM"]),
    }
    importance = pd.DataFrame({
        "feature": lgbm_pre.get_feature_names_out(),
        "gain_importance": challenger.booster_.feature_importance(importance_type="gain"),
        "split_importance": challenger.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain_importance", ascending=False)
    final_coefficients = pd.DataFrame({
        "feature": logistic_pre.get_feature_names_out(),
        "coefficient": logistic.coef_[0],
    })
    final_coefficients["odds_ratio"] = np.exp(final_coefficients["coefficient"])
    return {
        "development": dev,
        "final_test": final,
        "predictions": predictions,
        "models": {"Logistic": logistic, "LightGBM": challenger},
        "preprocessors": {"Logistic": logistic_pre, "LightGBM": lgbm_pre},
        "logistic_coefficients": final_coefficients,
        "lightgbm_importance": importance,
    }
