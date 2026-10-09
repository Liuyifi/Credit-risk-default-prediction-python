from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    load_application,
    temporal_filter_bureau,
    temporal_filter_installments,
    validate_raw_files,
)
from src.features import (
    bureau_balance_status_columns,
    consolidate_installment_rows,
    safe_ratio,
    validate_feature_table,
)
from src.modeling import REFERENCE_CATEGORIES


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def test_required_raw_files_and_application_grain():
    inventory = validate_raw_files(RAW)
    assert len(inventory) == 6
    application = load_application(RAW)
    assert len(application) == 307_511
    assert application["SK_ID_CURR"].is_unique
    assert application["TARGET"].sum() == 24_825
    bureau = pd.read_csv(RAW / "bureau.csv", usecols=["SK_ID_BUREAU", "DAYS_CREDIT", "DAYS_CREDIT_UPDATE"])
    assert bureau["SK_ID_BUREAU"].notna().all()
    assert bureau["SK_ID_BUREAU"].is_unique
    eligible, audit = temporal_filter_bureau(bureau)
    assert audit["excluded_rows"] == 17
    assert eligible["DAYS_CREDIT"].le(0).all()
    assert eligible["DAYS_CREDIT_UPDATE"].le(0).all()


def test_temporal_filters_and_borrower_contract():
    bureau = pd.DataFrame({
        "DAYS_CREDIT": [-10, -4, 1],
        "DAYS_CREDIT_UPDATE": [-2, 3, -1],
    })
    kept_bureau, audit = temporal_filter_bureau(bureau)
    assert len(kept_bureau) == 1
    assert audit["excluded_rows"] == 2

    installments = pd.DataFrame({
        "DAYS_INSTALMENT": [-20, -3, -2, -1],
        "DAYS_ENTRY_PAYMENT": [-10, np.nan, 2, -1],
    })
    kept_installments, audit = temporal_filter_installments(installments)
    assert len(kept_installments) == 2
    assert audit["excluded_rows"] == 2

    columns = {"SK_ID_CURR": [1, 2], "TARGET": [0, 1]}
    for idx in range(95):
        columns[f"APP_TEST_{idx}"] = [idx, idx + 1]
    table = pd.DataFrame(columns)
    validate_feature_table(table, expected_rows=2)


def test_bureau_balance_numeric_status_denominator_semantics():
    months = pd.DataFrame({"STATUS": ["0", "1", "2", "C", "X"]})
    flagged = bureau_balance_status_columns(months)
    assert flagged["BB_STATUS_OBSERVED"].sum() == 3
    assert flagged["BB_DELINQUENT"].sum() == 2
    assert flagged["BB_31_PLUS"].sum() == 1
    delinquent_share = safe_ratio(
        pd.Series([flagged["BB_DELINQUENT"].sum()]),
        pd.Series([flagged["BB_STATUS_OBSERVED"].sum()]),
    ).iloc[0]
    assert np.isclose(delinquent_share, 2 / 3)


def test_partial_installments_use_last_payment_day_and_summed_amount():
    raw = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_PREV": [10, 10],
        "NUM_INSTALMENT_VERSION": [1, 1],
        "NUM_INSTALMENT_NUMBER": [2, 2],
        "DAYS_INSTALMENT": [-20, -20],
        "DAYS_ENTRY_PAYMENT": [-22, -15],
        "AMT_INSTALMENT": [100.0, 100.0],
        "AMT_PAYMENT": [40.0, 60.0],
    })
    consolidated = consolidate_installment_rows(raw).iloc[0]
    assert consolidated["SCHEDULED_DAY"] == -20
    assert consolidated["PAYMENT_DAY"] == -15
    assert consolidated["SCHEDULED_AMOUNT"] == 100.0
    assert consolidated["PAYMENT_AMOUNT"] == 100.0


def test_cached_feature_table_contract_when_available():
    cache = ROOT / "data" / "processed" / "borrower_features.pkl"
    if not cache.exists():
        return
    frame = pd.read_pickle(cache)
    validate_feature_table(frame, expected_rows=307_511)
    for feature, reference in REFERENCE_CATEGORIES.items():
        assert frame[feature].fillna("__MISSING__").astype(str).eq(reference).sum() >= 200
