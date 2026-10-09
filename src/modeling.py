"""Nested outer-CV development and descriptive legacy-holdout fitting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .validation import calibration_slope_intercept, capacity_table, ks_statistic


RANDOM_STATE = 42
C_GRID = (0.01, 0.1, 1.0, 10.0)
OUTER_FOLDS = 5
INNER_FOLDS = 3
LGBM_MAX_ESTIMATORS = 1200
LGBM_EARLY_STOPPING_ROUNDS = 50

REFERENCE_CATEGORIES = {
    "APP_CONTRACT_TYPE": "Cash loans",
    "APP_OWNS_CAR": "N",
    "APP_OWNS_REALTY": "Y",
    "APP_INCOME_TYPE": "Working",
    "APP_EDUCATION_TYPE": "Secondary / secondary special",
    "APP_FAMILY_STATUS": "Married",
    "APP_HOUSING_TYPE": "House / apartment",
    "APP_OCCUPATION_TYPE": "Laborers",
}

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
    """Winsorize numeric columns using training-partition quantiles only."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper, axis=0)
        raw_min = np.nanmin(array, axis=0)
        raw_max = np.nanmax(array, axis=0)
        collapsed = (self.upper_bounds_ <= self.lower_bounds_) & (raw_max > raw_min)
        self.lower_bounds_[collapsed] = raw_min[collapsed]
        self.upper_bounds_[collapsed] = raw_max[collapsed]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.lower_bounds_))]
        return np.asarray(input_features, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Pool categories with support below a training-partition threshold."""

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
    selected_lgbm_candidate: int
    selected_lgbm_iterations: int
    outer_selections: pd.DataFrame
    selection_audit: pd.DataFrame
    calibration: Dict[str, dict]
    oof_predictions: Dict[str, np.ndarray]
    outer_fold_metrics: pd.DataFrame
    outer_cv_summary: pd.DataFrame
    logistic_stability: pd.DataFrame
    primary_model: str


def stratified_legacy_split(ids, target, test_size=0.20, random_state=RANDOM_STATE):
    development_ids, legacy_ids = train_test_split(
        np.asarray(ids), test_size=test_size, stratify=np.asarray(target), random_state=random_state
    )
    return {"development": np.sort(development_ids), "legacy_holdout": np.sort(legacy_ids)}


stratified_development_split = stratified_legacy_split


def predictor_columns(frame: pd.DataFrame, application_only=False) -> List[str]:
    columns = [c for c in frame.columns if c not in {"SK_ID_CURR", "TARGET"}]
    return [c for c in columns if c.startswith("APP_")] if application_only else columns


def column_types(frame: pd.DataFrame, columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    categorical = [c for c in columns if frame[c].dtype == "object" or str(frame[c].dtype).startswith("category")]
    return [c for c in columns if c not in categorical], categorical


def _category_levels(frame: pd.DataFrame, categorical: Sequence[str], min_count=200):
    levels = []
    for column in categorical:
        values = frame[column].fillna("__MISSING__").astype(str)
        frequent = set(values.value_counts()[lambda counts: counts >= min_count].index)
        reference = REFERENCE_CATEGORIES.get(column)
        ordered = [reference] if reference is not None else []
        ordered.extend(sorted(frequent - set(ordered)))
        # Missing is already present when frequent and otherwise maps to rare.
        # Rare is always reserved so validation-only categories remain known.
        if "__RARE__" not in ordered:
            ordered.append("__RARE__")
        levels.append(np.asarray(ordered, dtype=object))
    return levels


def make_preprocessor(frame: pd.DataFrame, columns: Sequence[str], model_kind: str) -> ColumnTransformer:
    numeric, categorical = column_types(frame, columns)
    numeric_steps = [
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
    ]
    categories = _category_levels(frame, categorical)
    if model_kind == "logistic":
        numeric_steps.append(("scale", StandardScaler()))
        cat_encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            drop=[REFERENCE_CATEGORIES[column] for column in categorical],
            sparse_output=True,
        )
    else:
        cat_encoder = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)
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


def transformed_matrix_diagnostics(matrix) -> dict:
    array = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    variances = np.var(array, axis=0)
    return {
        "rows": int(array.shape[0]),
        "columns": int(array.shape[1]),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "largest_absolute": float(np.max(np.abs(array))),
        "non_finite": int((~np.isfinite(array)).sum()),
        "zero_variance_columns": int(np.sum(variances == 0)),
    }


def _metric_values(y, score) -> dict:
    auc = roc_auc_score(y, score)
    values = {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "average_precision": float(average_precision_score(y, score)),
        "ks": float(ks_statistic(y, score)),
        "brier": float(brier_score_loss(y, score)),
    }
    for row in capacity_table(y, score, "model").itertuples(index=False):
        values[f"capture_at_{int(row.capacity_share * 100)}pct"] = float(row.bad_capture_share)
    return values


def _metric_rows(y, score, model, fold) -> List[dict]:
    return [
        {"model": model, "outer_fold": fold, "metric": metric, "value": value}
        for metric, value in _metric_values(y, score).items()
    ]


def _outer_summary(detail: pd.DataFrame) -> pd.DataFrame:
    grouped = detail.groupby(["model", "metric"], observed=True)["value"].agg(["mean", "std"])
    rows = []
    for model in detail["model"].drop_duplicates():
        row = {"model": model}
        for metric in detail["metric"].unique():
            row[f"{metric}_mean"] = grouped.loc[(model, metric), "mean"]
            row[f"{metric}_std"] = grouped.loc[(model, metric), "std"]
        rows.append(row)
    return pd.DataFrame(rows)


def _new_logistic(c):
    return LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=1000)


def _logistic_score(model, matrix):
    """Score without the platform BLAS matmul warnings seen in sklearn's path."""
    if hasattr(matrix, "multiply"):
        linear = np.asarray(matrix.multiply(model.coef_[0]).sum(axis=1)).ravel()
    else:
        linear = np.einsum("ij,j->i", np.asarray(matrix), model.coef_[0], optimize=False)
    return expit(linear + model.intercept_[0])


def _select_logistic_c(outer_train, columns, seed):
    y = outer_train["TARGET"].to_numpy()
    folds = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    aucs = {c: [] for c in C_GRID}
    for inner_train_idx, inner_valid_idx in folds.split(outer_train, y):
        inner_train = outer_train.iloc[inner_train_idx]
        inner_valid = outer_train.iloc[inner_valid_idx]
        preprocessor = make_preprocessor(inner_train, columns, "logistic")
        x_train = preprocessor.fit_transform(inner_train[list(columns)])
        x_valid = preprocessor.transform(inner_valid[list(columns)])
        for c in C_GRID:
            model = _new_logistic(c).fit(x_train, inner_train["TARGET"])
            score = _logistic_score(model, x_valid)
            aucs[c].append(roc_auc_score(inner_valid["TARGET"], score))
    chosen = sorted(C_GRID, key=lambda c: (-np.mean(aucs[c]), c))[0]
    return float(chosen), {str(c): float(np.mean(values)) for c, values in aucs.items()}


def _lgbm_model(params: dict, random_state=RANDOM_STATE, n_estimators=LGBM_MAX_ESTIMATORS):
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=n_estimators, random_state=random_state,
        n_jobs=-1, verbosity=-1, subsample_freq=1, **params,
    )


def _select_lightgbm(outer_train, columns, seed):
    indices = np.arange(len(outer_train))
    inner_train_idx, inner_valid_idx = train_test_split(
        indices, test_size=0.20, stratify=outer_train["TARGET"], random_state=seed
    )
    inner_train = outer_train.iloc[inner_train_idx]
    inner_valid = outer_train.iloc[inner_valid_idx]
    preprocessor = make_preprocessor(inner_train, columns, "lightgbm")
    x_train = preprocessor.fit_transform(inner_train[list(columns)])
    x_valid = preprocessor.transform(inner_valid[list(columns)])
    candidates = []
    for index, params in enumerate(LIGHTGBM_CANDIDATES):
        model = _lgbm_model(params, random_state=seed)
        model.fit(
            x_train, inner_train["TARGET"],
            eval_set=[(x_valid, inner_valid["TARGET"])], eval_metric="auc",
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        score = model.booster_.predict(x_valid, num_iteration=model.best_iteration_)
        candidates.append({
            "candidate": index,
            "inner_auc": float(roc_auc_score(inner_valid["TARGET"], score)),
            "best_iteration": int(model.best_iteration_),
        })
    selected = sorted(candidates, key=lambda row: (-row["inner_auc"], row["candidate"]))[0]
    return selected, inner_train_idx, inner_valid_idx, candidates


def _feature_metadata(preprocessor) -> pd.DataFrame:
    numeric = list(preprocessor.transformers_[0][2])
    categorical = list(preprocessor.transformers_[1][2])
    rows = [
        {"transformed_feature": f"numeric__{feature}", "source_feature": feature,
         "category": "", "reference_category": ""}
        for feature in numeric
    ]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encode"]
    encoded_names = list(encoder.get_feature_names_out(categorical))
    position = 0
    for index, feature in enumerate(categorical):
        drop_index = None if encoder.drop_idx_ is None else encoder.drop_idx_[index]
        for category_index, category in enumerate(encoder.categories_[index]):
            if drop_index is not None and category_index == drop_index:
                continue
            rows.append({
                "transformed_feature": f"categorical__{encoded_names[position]}",
                "source_feature": feature,
                "category": str(category),
                "reference_category": REFERENCE_CATEGORIES[feature],
            })
            position += 1
    return pd.DataFrame(rows)


def _coefficient_stability(coefficient_frames):
    combined = pd.concat(coefficient_frames, ignore_index=True)
    keys = ["transformed_feature", "source_feature", "category", "reference_category"]
    grouped = combined.groupby(keys, observed=True, dropna=False)["coefficient"]
    out = grouped.agg(fold_mean="mean", fold_std="std", coefficient="mean", folds="count").reset_index()
    signs = grouped.apply(lambda values: max((values >= 0).mean(), (values <= 0).mean())).rename("sign_stability")
    out = out.merge(signs.reset_index(), on=keys)
    out["odds_ratio"] = np.exp(out["coefficient"])
    out["source_family"] = out["source_feature"].str.extract(r"^(APP|BUREAU|BB|PREV|INST)_", expand=False)
    columns = keys + ["coefficient", "odds_ratio", "fold_mean", "fold_std", "sign_stability", "source_family", "folds"]
    return out[columns].sort_values("coefficient", key=lambda values: values.abs(), ascending=False)


def _stable_mode(values):
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def nested_split_plan(frame: pd.DataFrame) -> pd.DataFrame:
    """Return auditable global-index relationships for the nested split design."""
    y = frame["TARGET"].to_numpy()
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (outer_train_idx, outer_valid_idx) in enumerate(outer.split(frame, y), 1):
        inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE + fold)
        for inner_fold, (relative_train, relative_valid) in enumerate(inner.split(outer_train_idx, y[outer_train_idx]), 1):
            inner_train = outer_train_idx[relative_train]
            inner_valid = outer_train_idx[relative_valid]
            rows.append({
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "inner_train_outer_valid_overlap": len(np.intersect1d(inner_train, outer_valid_idx)),
                "inner_valid_outer_valid_overlap": len(np.intersect1d(inner_valid, outer_valid_idx)),
                "outer_train_rows": len(outer_train_idx),
                "outer_valid_rows": len(outer_valid_idx),
            })
    return pd.DataFrame(rows)


def run_development(frame: pd.DataFrame) -> DevelopmentResult:
    """Generate primary evidence with fully separated outer validation folds."""
    split_ids = stratified_legacy_split(frame["SK_ID_CURR"], frame["TARGET"])
    all_columns = predictor_columns(frame)
    app_columns = predictor_columns(frame, application_only=True)
    y = frame["TARGET"].to_numpy()
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    predictions = {name: np.zeros(len(frame)) for name in (
        "Application-only Logistic", "Logistic", "LightGBM"
    )}
    metric_rows, selection_rows, audit_rows, coefficient_frames = [], [], [], []

    for fold, (outer_train_idx, outer_valid_idx) in enumerate(outer.split(frame, y), 1):
        outer_train = frame.iloc[outer_train_idx].reset_index(drop=True)
        outer_valid = frame.iloc[outer_valid_idx]
        outer_valid_ids = set(outer_valid["SK_ID_CURR"])

        for name, columns in (("Application-only Logistic", app_columns), ("Logistic", all_columns)):
            selected_c, inner_scores = _select_logistic_c(outer_train, columns, RANDOM_STATE + fold)
            preprocessor = make_preprocessor(outer_train, columns, "logistic")
            x_train = preprocessor.fit_transform(outer_train[list(columns)])
            x_valid = preprocessor.transform(outer_valid[list(columns)])
            model = _new_logistic(selected_c).fit(x_train, outer_train["TARGET"])
            score = _logistic_score(model, x_valid)
            predictions[name][outer_valid_idx] = score
            metric_rows.extend(_metric_rows(outer_valid["TARGET"], score, name, fold))
            selection_rows.append({
                "outer_fold": fold, "model": name, "selected_c": selected_c,
                "selected_lgbm_candidate": np.nan, "selected_lgbm_iterations": np.nan,
                "inner_selection_score": inner_scores[str(selected_c)],
            })
            if name == "Logistic":
                metadata = _feature_metadata(preprocessor)
                metadata["coefficient"] = model.coef_[0]
                metadata["outer_fold"] = fold
                coefficient_frames.append(metadata)

        selected, inner_train_idx, inner_valid_idx, candidate_results = _select_lightgbm(
            outer_train, all_columns, RANDOM_STATE + 100 + fold
        )
        preprocessor = make_preprocessor(outer_train, all_columns, "lightgbm")
        x_train = preprocessor.fit_transform(outer_train[all_columns])
        x_valid = preprocessor.transform(outer_valid[all_columns])
        challenger = _lgbm_model(
            LIGHTGBM_CANDIDATES[selected["candidate"]],
            random_state=RANDOM_STATE + fold,
            n_estimators=selected["best_iteration"],
        ).fit(x_train, outer_train["TARGET"])
        score = challenger.booster_.predict(x_valid)
        predictions["LightGBM"][outer_valid_idx] = score
        metric_rows.extend(_metric_rows(outer_valid["TARGET"], score, "LightGBM", fold))
        selection_rows.append({
            "outer_fold": fold, "model": "LightGBM", "selected_c": np.nan,
            "selected_lgbm_candidate": selected["candidate"] + 1,
            "selected_lgbm_iterations": selected["best_iteration"],
            "inner_selection_score": selected["inner_auc"],
        })
        inner_train_ids = set(outer_train.iloc[inner_train_idx]["SK_ID_CURR"])
        inner_valid_ids = set(outer_train.iloc[inner_valid_idx]["SK_ID_CURR"])
        audit_rows.append({
            "outer_fold": fold,
            "outer_train_rows": len(outer_train_idx),
            "outer_valid_rows": len(outer_valid_idx),
            "lgbm_inner_train_rows": len(inner_train_idx),
            "lgbm_inner_valid_rows": len(inner_valid_idx),
            "inner_train_outer_valid_overlap": len(inner_train_ids & outer_valid_ids),
            "inner_valid_outer_valid_overlap": len(inner_valid_ids & outer_valid_ids),
            "outer_validation_used_for_early_stopping": False,
            "candidate_results": candidate_results,
        })

    detail = pd.DataFrame(metric_rows)
    summary = _outer_summary(detail)
    selections = pd.DataFrame(selection_rows)
    selected_c = float(_stable_mode(selections.loc[selections["model"].eq("Logistic"), "selected_c"]))
    selected_candidate = int(_stable_mode(
        selections.loc[selections["model"].eq("LightGBM"), "selected_lgbm_candidate"].astype(int)
    ))
    matching_iterations = selections.loc[
        selections["model"].eq("LightGBM") & selections["selected_lgbm_candidate"].eq(selected_candidate),
        "selected_lgbm_iterations",
    ]
    selected_iterations = int(np.median(matching_iterations))
    calibration = {}
    for model, score in predictions.items():
        slope, intercept = calibration_slope_intercept(y, score)
        calibration[model] = {
            "applied": False, "raw_brier": float(brier_score_loss(y, score)),
            "slope": slope, "intercept": intercept,
        }
    log_auc = float(summary.loc[summary["model"].eq("Logistic"), "roc_auc_mean"].iloc[0])
    lgb_auc = float(summary.loc[summary["model"].eq("LightGBM"), "roc_auc_mean"].iloc[0])
    primary = "LightGBM" if lgb_auc - log_auc >= 0.005 else "Logistic"
    return DevelopmentResult(
        split_ids=split_ids,
        selected_c=selected_c,
        selected_lgbm_params=dict(LIGHTGBM_CANDIDATES[selected_candidate - 1]),
        selected_lgbm_candidate=selected_candidate,
        selected_lgbm_iterations=selected_iterations,
        outer_selections=selections,
        selection_audit=pd.DataFrame(audit_rows),
        calibration=calibration,
        oof_predictions=predictions,
        outer_fold_metrics=detail,
        outer_cv_summary=summary,
        logistic_stability=_coefficient_stability(coefficient_frames),
        primary_model=primary,
    )


def fit_legacy_holdout_models(frame: pd.DataFrame, result: DevelopmentResult):
    """Fit outer-evidence specifications and score the historical split once."""
    development = frame[frame["SK_ID_CURR"].isin(result.split_ids["development"])].reset_index(drop=True)
    legacy = frame[frame["SK_ID_CURR"].isin(result.split_ids["legacy_holdout"])].reset_index(drop=True)
    columns = predictor_columns(development)

    logistic_pre = make_preprocessor(development, columns, "logistic")
    x_development = logistic_pre.fit_transform(development[columns])
    x_legacy = logistic_pre.transform(legacy[columns])
    logistic = _new_logistic(result.selected_c).fit(x_development, development["TARGET"])
    logistic_score = _logistic_score(logistic, x_legacy)

    lgbm_pre = make_preprocessor(development, columns, "lightgbm")
    x_development_lgbm = lgbm_pre.fit_transform(development[columns])
    x_legacy_lgbm = lgbm_pre.transform(legacy[columns])
    challenger = _lgbm_model(
        result.selected_lgbm_params, n_estimators=result.selected_lgbm_iterations
    ).fit(x_development_lgbm, development["TARGET"])
    lgbm_score = challenger.booster_.predict(x_legacy_lgbm)

    importance = pd.DataFrame({
        "feature": lgbm_pre.get_feature_names_out(),
        "gain_importance": challenger.booster_.feature_importance(importance_type="gain"),
        "split_importance": challenger.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain_importance", ascending=False)
    coefficients = _feature_metadata(logistic_pre)
    coefficients["coefficient"] = logistic.coef_[0]
    coefficients["odds_ratio"] = np.exp(coefficients["coefficient"])
    return {
        "development": development,
        "legacy_holdout": legacy,
        "predictions": {"Logistic": logistic_score, "LightGBM": lgbm_score},
        "models": {"Logistic": logistic, "LightGBM": challenger},
        "preprocessors": {"Logistic": logistic_pre, "LightGBM": lgbm_pre},
        "logistic_coefficients": coefficients,
        "lightgbm_importance": importance,
    }


fit_locked_models = fit_legacy_holdout_models
