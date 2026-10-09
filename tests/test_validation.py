import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.modeling import (
    REFERENCE_CATEGORIES,
    make_preprocessor,
    nested_split_plan,
    transformed_matrix_diagnostics,
)
from src.validation import assign_risk_deciles, capacity_table, ks_statistic, risk_decile_table


def test_ks_matches_two_sample_reference_with_ties():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.2, 0.4, 0.6, 0.7, 0.7, 0.9])
    reference = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert np.isclose(ks_statistic(y, score), reference)


def test_deciles_and_capacity_reconcile_without_outcome_ranking():
    rng = np.random.default_rng(7)
    score = rng.uniform(size=1_000)
    y = rng.binomial(1, score)
    deciles_before = assign_risk_deciles(score)
    deciles_after = assign_risk_deciles(score)
    assert np.array_equal(deciles_before, deciles_after)

    table = risk_decile_table(y, score, "test")
    assert table["borrowers"].sum() == len(y)
    assert table["bads"].sum() == y.sum()
    assert np.isclose(table["cumulative_bad_capture"].iloc[-1], 1.0)

    capacity = capacity_table(y, score, "test")
    assert capacity["reviewed_borrowers"].tolist() == [50, 100, 200]
    assert (capacity["captured_bads"] <= y.sum()).all()
    assert (capacity["non_events_reviewed"] + capacity["captured_bads"] == capacity["reviewed_borrowers"]).all()


def _synthetic_model_frame(rows=500):
    rng = np.random.default_rng(9)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(rows),
        "TARGET": np.tile([0] * 9 + [1], rows // 10),
        "APP_INCOME": rng.lognormal(10, 1, rows),
        "APP_EMPLOYMENT_YEARS": np.where(np.arange(rows) % 7, rng.uniform(0, 40, rows), np.nan),
        "APP_EMPLOYMENT_MISSING": (np.arange(rows) % 7 == 0).astype(int),
        "APP_CONTRACT_TYPE": np.where(np.arange(rows) % 4, "Cash loans", "Revolving loans"),
        "APP_OWNS_CAR": np.where(np.arange(rows) % 3, "N", "Y"),
        "APP_OWNS_REALTY": np.where(np.arange(rows) % 3, "Y", "N"),
        "APP_INCOME_TYPE": "Working",
        "APP_EDUCATION_TYPE": "Secondary / secondary special",
        "APP_FAMILY_STATUS": "Married",
        "APP_HOUSING_TYPE": "House / apartment",
        "APP_OCCUPATION_TYPE": "Laborers",
    })


def test_nested_validation_indices_are_isolated():
    plan = nested_split_plan(_synthetic_model_frame())
    assert len(plan) == 15
    assert plan["inner_train_outer_valid_overlap"].eq(0).all()
    assert plan["inner_valid_outer_valid_overlap"].eq(0).all()


def test_transformed_matrix_is_finite_without_duplicate_auto_indicators():
    frame = _synthetic_model_frame()
    columns = [c for c in frame if c not in {"SK_ID_CURR", "TARGET"}]
    preprocessor = make_preprocessor(frame, columns, "logistic")
    matrix = preprocessor.fit_transform(frame[columns])
    diagnostics = transformed_matrix_diagnostics(matrix)
    assert diagnostics["non_finite"] == 0
    names = preprocessor.get_feature_names_out()
    assert not any("missingindicator_" in name for name in names)
    assert sum(name.endswith("APP_EMPLOYMENT_MISSING") for name in names) == 1


def test_logistic_reference_categories_are_explicit_and_deterministic():
    frame = _synthetic_model_frame()
    columns = [c for c in frame if c not in {"SK_ID_CURR", "TARGET"}]
    preprocessor = make_preprocessor(frame, columns, "logistic").fit(frame[columns])
    categorical = preprocessor.named_transformers_["categorical"].named_steps["encode"]
    categorical_columns = preprocessor.transformers_[1][2]
    dropped = {
        column: str(categorical.categories_[index][categorical.drop_idx_[index]])
        for index, column in enumerate(categorical_columns)
    }
    assert dropped == REFERENCE_CATEGORIES
