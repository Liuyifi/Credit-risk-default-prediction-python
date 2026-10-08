"""Deliberate borrower-level features from application and historical tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd

from .data import (
    iter_csv,
    load_application,
    load_bureau,
    load_previous,
    temporal_filter_bureau,
    temporal_filter_bureau_balance,
    temporal_filter_installments,
    temporal_filter_previous,
)


ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
MAX_FEATURES = 150


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan)


def application_features(application: pd.DataFrame) -> pd.DataFrame:
    """Small, interpretable set known at application time."""
    out = pd.DataFrame({ID_COL: application[ID_COL], TARGET_COL: application[TARGET_COL]})
    numeric = {
        "AMT_INCOME_TOTAL": "APP_INCOME",
        "AMT_CREDIT": "APP_CREDIT",
        "AMT_ANNUITY": "APP_ANNUITY",
        "CNT_CHILDREN": "APP_CHILDREN",
        "CNT_FAM_MEMBERS": "APP_FAMILY_MEMBERS",
        "OWN_CAR_AGE": "APP_CAR_AGE",
        "REGION_POPULATION_RELATIVE": "APP_REGION_POPULATION_RELATIVE",
    }
    categorical = {
        "NAME_CONTRACT_TYPE": "APP_CONTRACT_TYPE",
        "FLAG_OWN_CAR": "APP_OWNS_CAR",
        "FLAG_OWN_REALTY": "APP_OWNS_REALTY",
        "NAME_INCOME_TYPE": "APP_INCOME_TYPE",
        "NAME_EDUCATION_TYPE": "APP_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS": "APP_FAMILY_STATUS",
        "NAME_HOUSING_TYPE": "APP_HOUSING_TYPE",
        "OCCUPATION_TYPE": "APP_OCCUPATION_TYPE",
    }
    for source, destination in {**numeric, **categorical}.items():
        out[destination] = application[source]

    employed = application["DAYS_EMPLOYED"].replace(365243, np.nan)
    out["APP_AGE_YEARS"] = -application["DAYS_BIRTH"] / 365.25
    out["APP_EMPLOYMENT_YEARS"] = -employed / 365.25
    out["APP_EMPLOYMENT_MISSING"] = employed.isna().astype("int8")
    out["APP_CREDIT_TO_INCOME"] = safe_ratio(application["AMT_CREDIT"], application["AMT_INCOME_TOTAL"])
    out["APP_ANNUITY_TO_INCOME"] = safe_ratio(application["AMT_ANNUITY"], application["AMT_INCOME_TOTAL"])
    out["APP_INCOME_PER_FAMILY_MEMBER"] = safe_ratio(application["AMT_INCOME_TOTAL"], application["CNT_FAM_MEMBERS"])
    out["APP_CREDIT_TO_ANNUITY"] = safe_ratio(application["AMT_CREDIT"], application["AMT_ANNUITY"])
    out["APP_CREDIT_TO_GOODS"] = safe_ratio(application["AMT_CREDIT"], application["AMT_GOODS_PRICE"])
    return out


def bureau_features(bureau: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    bureau, audit = temporal_filter_bureau(bureau)
    b = bureau.copy()
    b["IS_ACTIVE"] = b["CREDIT_ACTIVE"].eq("Active").astype("int8")
    b["IS_CLOSED"] = b["CREDIT_ACTIVE"].eq("Closed").astype("int8")
    b["IS_OVERDUE"] = b["AMT_CREDIT_SUM_OVERDUE"].fillna(0).gt(0).astype("int8")
    b["ACTIVE_CREDIT"] = b["AMT_CREDIT_SUM"].where(b["IS_ACTIVE"].eq(1), 0)
    b["ACTIVE_DEBT"] = b["AMT_CREDIT_SUM_DEBT"].where(b["IS_ACTIVE"].eq(1), 0)

    grouped = b.groupby(ID_COL, observed=True)
    out = grouped.agg(
        BUREAU_RECORD_COUNT=("SK_ID_BUREAU", "size"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_CLOSED_COUNT=("IS_CLOSED", "sum"),
        BUREAU_CREDIT_TYPE_COUNT=("CREDIT_TYPE", "nunique"),
        BUREAU_DAYS_SINCE_MOST_RECENT=("DAYS_CREDIT", lambda x: -x.max()),
        BUREAU_HISTORY_DAYS=("DAYS_CREDIT", lambda x: -x.min()),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_OVERDUE_ACCOUNT_COUNT=("IS_OVERDUE", "sum"),
        BUREAU_PROLONG_COUNT=("CNT_CREDIT_PROLONG", "sum"),
        BUREAU_ANNUITY_SUM=("AMT_ANNUITY", "sum"),
        BUREAU_ACTIVE_CREDIT_SUM=("ACTIVE_CREDIT", "sum"),
        BUREAU_ACTIVE_DEBT_SUM=("ACTIVE_DEBT", "sum"),
    ).reset_index()
    out["BUREAU_ACTIVE_SHARE"] = safe_ratio(out["BUREAU_ACTIVE_COUNT"], out["BUREAU_RECORD_COUNT"])
    out["BUREAU_DEBT_TO_CREDIT"] = safe_ratio(out["BUREAU_DEBT_SUM"], out["BUREAU_CREDIT_SUM"])
    out["BUREAU_ACTIVE_DEBT_TO_CREDIT"] = safe_ratio(out["BUREAU_ACTIVE_DEBT_SUM"], out["BUREAU_ACTIVE_CREDIT_SUM"])
    return out, audit


def bureau_balance_features(
    raw_dir: Union[Path, str],
    bureau_map: pd.DataFrame,
    chunksize: int = 1_000_000,
) -> Tuple[pd.DataFrame, dict]:
    """Aggregate monthly bureau status first to credit, then to borrower."""
    bureau_map = bureau_map[["SK_ID_BUREAU", ID_COL]].drop_duplicates("SK_ID_BUREAU")
    valid_ids = set(bureau_map["SK_ID_BUREAU"])
    partials = []
    input_rows = retained_rows = temporal_excluded = unmapped_rows = 0
    for chunk in iter_csv(raw_dir, "bureau_balance.csv", chunksize):
        input_rows += len(chunk)
        chunk, audit = temporal_filter_bureau_balance(chunk)
        temporal_excluded += audit["excluded_rows"]
        mapped = chunk["SK_ID_BUREAU"].isin(valid_ids)
        unmapped_rows += int((~mapped).sum())
        chunk = chunk.loc[mapped].copy()
        retained_rows += len(chunk)
        severity = pd.to_numeric(chunk["STATUS"], errors="coerce")
        chunk["BB_DELINQUENT"] = severity.between(1, 5).astype("int8")
        chunk["BB_31_PLUS"] = severity.between(2, 5).astype("int8")
        chunk["BB_61_PLUS"] = severity.between(3, 5).astype("int8")
        chunk["BB_91_PLUS"] = severity.between(4, 5).astype("int8")
        chunk["BB_CLEAN"] = chunk["STATUS"].eq("0").astype("int8")
        chunk["BB_SEVERITY"] = severity.fillna(0)
        chunk["BB_RECENT_DELINQUENT"] = (
            chunk["MONTHS_BALANCE"].ge(-5) & chunk["BB_DELINQUENT"].eq(1)
        ).astype("int8")
        partials.append(
            chunk.groupby("SK_ID_BUREAU", observed=True).agg(
                BB_MONTH_COUNT=("MONTHS_BALANCE", "size"),
                BB_DELINQUENT_MONTHS=("BB_DELINQUENT", "sum"),
                BB_31_PLUS_MONTHS=("BB_31_PLUS", "sum"),
                BB_61_PLUS_MONTHS=("BB_61_PLUS", "sum"),
                BB_91_PLUS_MONTHS=("BB_91_PLUS", "sum"),
                BB_CLEAN_MONTHS=("BB_CLEAN", "sum"),
                BB_WORST_SEVERITY=("BB_SEVERITY", "max"),
                BB_RECENT_DELINQUENT=("BB_RECENT_DELINQUENT", "max"),
            ).reset_index()
        )

    credit = pd.concat(partials, ignore_index=True)
    sum_cols = [c for c in credit if c.endswith("_MONTHS") or c == "BB_MONTH_COUNT"]
    credit = credit.groupby("SK_ID_BUREAU", observed=True).agg(
        **{c: (c, "sum") for c in sum_cols},
        BB_WORST_SEVERITY=("BB_WORST_SEVERITY", "max"),
        BB_RECENT_DELINQUENT=("BB_RECENT_DELINQUENT", "max"),
    ).reset_index()
    credit["BB_CREDIT_EVER_DELINQUENT"] = credit["BB_DELINQUENT_MONTHS"].gt(0).astype("int8")
    credit = credit.merge(bureau_map, on="SK_ID_BUREAU", how="left", validate="one_to_one")
    borrower = credit.groupby(ID_COL, observed=True).agg(
        BB_MONTH_COUNT=("BB_MONTH_COUNT", "sum"),
        BB_DELINQUENT_MONTHS=("BB_DELINQUENT_MONTHS", "sum"),
        BB_31_PLUS_MONTHS=("BB_31_PLUS_MONTHS", "sum"),
        BB_61_PLUS_MONTHS=("BB_61_PLUS_MONTHS", "sum"),
        BB_91_PLUS_MONTHS=("BB_91_PLUS_MONTHS", "sum"),
        BB_CLEAN_MONTHS=("BB_CLEAN_MONTHS", "sum"),
        BB_WORST_SEVERITY=("BB_WORST_SEVERITY", "max"),
        BB_RECENT_DELINQUENT=("BB_RECENT_DELINQUENT", "max"),
        BB_CREDITS_EVER_DELINQUENT=("BB_CREDIT_EVER_DELINQUENT", "sum"),
    ).reset_index()
    borrower["BB_DELINQUENT_MONTH_SHARE"] = safe_ratio(borrower["BB_DELINQUENT_MONTHS"], borrower["BB_MONTH_COUNT"])
    borrower["BB_31_PLUS_MONTH_SHARE"] = safe_ratio(borrower["BB_31_PLUS_MONTHS"], borrower["BB_MONTH_COUNT"])
    audit = {
        "source": "bureau_balance",
        "input_rows": int(input_rows),
        "retained_rows": int(retained_rows),
        "excluded_rows": int(temporal_excluded + unmapped_rows),
        "temporal_excluded_rows": int(temporal_excluded),
        "unmapped_bureau_rows": int(unmapped_rows),
        "rule": "MONTHS_BALANCE <= 0 and SK_ID_BUREAU maps to an eligible bureau record",
    }
    return borrower, audit


def previous_features(previous: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    previous, audit = temporal_filter_previous(previous)
    p = previous.copy()
    status = p["NAME_CONTRACT_STATUS"]
    for label in ("Approved", "Refused", "Canceled", "Unused offer"):
        p[label.upper().replace(" ", "_")] = status.eq(label).astype("int8")
    contract = p["NAME_CONTRACT_TYPE"]
    p["IS_CASH_LOAN"] = contract.eq("Cash loans").astype("int8")
    p["IS_CONSUMER_LOAN"] = contract.eq("Consumer loans").astype("int8")
    p["REQUEST_TO_GRANTED"] = safe_ratio(p["AMT_CREDIT"], p["AMT_APPLICATION"])
    p["APPROVED_CREDIT"] = p["AMT_CREDIT"].where(status.eq("Approved"), 0)
    grouped = p.groupby(ID_COL, observed=True)
    out = grouped.agg(
        PREV_APPLICATION_COUNT=("SK_ID_PREV", "size"),
        PREV_APPROVED_COUNT=("APPROVED", "sum"),
        PREV_REFUSED_COUNT=("REFUSED", "sum"),
        PREV_CANCELED_COUNT=("CANCELED", "sum"),
        PREV_UNUSED_OFFER_COUNT=("UNUSED_OFFER", "sum"),
        PREV_DAYS_SINCE_MOST_RECENT=("DAYS_DECISION", lambda x: -x.max()),
        PREV_HISTORY_DAYS=("DAYS_DECISION", lambda x: -x.min()),
        PREV_APPLICATION_AMOUNT_MEAN=("AMT_APPLICATION", "mean"),
        PREV_CREDIT_AMOUNT_MEAN=("AMT_CREDIT", "mean"),
        PREV_CREDIT_AMOUNT_SUM=("AMT_CREDIT", "sum"),
        PREV_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_REQUEST_TO_GRANTED_MEAN=("REQUEST_TO_GRANTED", "mean"),
        PREV_APPROVED_CREDIT_SUM=("APPROVED_CREDIT", "sum"),
        PREV_CASH_LOAN_COUNT=("IS_CASH_LOAN", "sum"),
        PREV_CONSUMER_LOAN_COUNT=("IS_CONSUMER_LOAN", "sum"),
    ).reset_index()
    for stem in ("APPROVED", "REFUSED", "CANCELED", "UNUSED_OFFER", "CASH_LOAN", "CONSUMER_LOAN"):
        out[f"PREV_{stem}_SHARE"] = safe_ratio(out[f"PREV_{stem}_COUNT"], out["PREV_APPLICATION_COUNT"])
    return out, audit


def installment_features(
    raw_dir: Union[Path, str],
    borrower_ids: Iterable[int],
    chunksize: int = 1_000_000,
) -> Tuple[pd.DataFrame, dict]:
    """Consolidate payment rows to a scheduled installment before borrower roll-up."""
    valid_ids = set(borrower_ids)
    partials = []
    input_rows = retained_rows = excluded_rows = outside_population = 0
    keys = [ID_COL, "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]
    for chunk in iter_csv(raw_dir, "installments_payments.csv", chunksize):
        input_rows += len(chunk)
        chunk, audit = temporal_filter_installments(chunk)
        excluded_rows += audit["excluded_rows"]
        in_population = chunk[ID_COL].isin(valid_ids)
        outside_population += int((~in_population).sum())
        chunk = chunk.loc[in_population]
        retained_rows += len(chunk)
        partials.append(
            chunk.groupby(keys, observed=True, dropna=False).agg(
                SCHEDULED_DAY=("DAYS_INSTALMENT", "min"),
                PAYMENT_DAY=("DAYS_ENTRY_PAYMENT", "max"),
                SCHEDULED_AMOUNT=("AMT_INSTALMENT", "max"),
                PAYMENT_AMOUNT=("AMT_PAYMENT", "sum"),
            ).reset_index()
        )
    schedule = pd.concat(partials, ignore_index=True)
    schedule = schedule.groupby(keys, observed=True, dropna=False).agg(
        SCHEDULED_DAY=("SCHEDULED_DAY", "min"),
        PAYMENT_DAY=("PAYMENT_DAY", "max"),
        SCHEDULED_AMOUNT=("SCHEDULED_AMOUNT", "max"),
        PAYMENT_AMOUNT=("PAYMENT_AMOUNT", "sum"),
    ).reset_index()
    schedule["DAYS_LATE"] = (schedule["PAYMENT_DAY"] - schedule["SCHEDULED_DAY"]).clip(lower=0)
    schedule["PAYMENT_RATIO"] = safe_ratio(schedule["PAYMENT_AMOUNT"], schedule["SCHEDULED_AMOUNT"])
    schedule["IS_LATE"] = schedule["DAYS_LATE"].gt(0).astype("int8")
    schedule["IS_30_PLUS_LATE"] = schedule["DAYS_LATE"].gt(30).astype("int8")
    schedule["IS_UNDERPAID"] = schedule["PAYMENT_RATIO"].lt(0.95).astype("int8")
    schedule["IS_SEVERELY_UNDERPAID"] = schedule["PAYMENT_RATIO"].lt(0.50).astype("int8")
    schedule["IS_RECENT"] = schedule["SCHEDULED_DAY"].ge(-365).astype("int8")
    schedule["RECENT_LATE"] = (schedule["IS_RECENT"].eq(1) & schedule["IS_LATE"].eq(1)).astype("int8")
    schedule["RECENT_UNDERPAID"] = (schedule["IS_RECENT"].eq(1) & schedule["IS_UNDERPAID"].eq(1)).astype("int8")

    out = schedule.groupby(ID_COL, observed=True).agg(
        INST_SCHEDULE_COUNT=("NUM_INSTALMENT_NUMBER", "size"),
        INST_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
        INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INST_DAYS_LATE_STD=("DAYS_LATE", "std"),
        INST_LATE_COUNT=("IS_LATE", "sum"),
        INST_30_PLUS_LATE_COUNT=("IS_30_PLUS_LATE", "sum"),
        INST_UNDERPAID_COUNT=("IS_UNDERPAID", "sum"),
        INST_SEVERELY_UNDERPAID_COUNT=("IS_SEVERELY_UNDERPAID", "sum"),
        INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INST_PAYMENT_RATIO_MIN=("PAYMENT_RATIO", "min"),
        INST_DAYS_SINCE_LAST_SCHEDULED=("SCHEDULED_DAY", lambda x: -x.max()),
        INST_RECENT_SCHEDULE_COUNT=("IS_RECENT", "sum"),
        INST_RECENT_LATE_COUNT=("RECENT_LATE", "sum"),
        INST_RECENT_UNDERPAID_COUNT=("RECENT_UNDERPAID", "sum"),
    ).reset_index()
    for stem in ("LATE", "30_PLUS_LATE", "UNDERPAID", "SEVERELY_UNDERPAID"):
        out[f"INST_{stem}_RATE"] = safe_ratio(out[f"INST_{stem}_COUNT"], out["INST_SCHEDULE_COUNT"])
    out["INST_RECENT_LATE_RATE"] = safe_ratio(out["INST_RECENT_LATE_COUNT"], out["INST_RECENT_SCHEDULE_COUNT"])
    out["INST_RECENT_UNDERPAID_RATE"] = safe_ratio(out["INST_RECENT_UNDERPAID_COUNT"], out["INST_RECENT_SCHEDULE_COUNT"])
    audit = {
        "source": "installments_payments",
        "input_rows": int(input_rows),
        "retained_rows": int(retained_rows),
        "excluded_rows": int(excluded_rows + outside_population),
        "temporal_or_missing_date_excluded_rows": int(excluded_rows),
        "outside_application_population_rows": int(outside_population),
        "rule": "borrower in application_train; both scheduled and payment dates non-null and <= 0",
    }
    return out, audit


def build_feature_table(raw_dir: Union[Path, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build and validate the complete one-row-per-borrower feature table."""
    application = load_application(raw_dir)
    borrower_ids = application[ID_COL]
    app = application_features(application)

    bureau_raw = load_bureau(raw_dir, borrower_ids)
    bureau, bureau_audit = bureau_features(bureau_raw)
    eligible_bureau, _ = temporal_filter_bureau(bureau_raw)
    balance, balance_audit = bureau_balance_features(raw_dir, eligible_bureau[["SK_ID_BUREAU", ID_COL]])

    previous_raw = load_previous(raw_dir, borrower_ids)
    previous, previous_audit = previous_features(previous_raw)
    installments, installment_audit = installment_features(raw_dir, borrower_ids)

    table = app
    for family in (bureau, balance, previous, installments):
        table = table.merge(family, on=ID_COL, how="left", validate="one_to_one")

    table["BUREAU_NO_HISTORY"] = table["BUREAU_RECORD_COUNT"].isna().astype("int8")
    table["BB_NO_HISTORY"] = table["BB_MONTH_COUNT"].isna().astype("int8")
    table["PREV_NO_HISTORY"] = table["PREV_APPLICATION_COUNT"].isna().astype("int8")
    table["INST_NO_HISTORY"] = table["INST_SCHEDULE_COUNT"].isna().astype("int8")
    table = table.replace([np.inf, -np.inf], np.nan)
    validate_feature_table(table, len(application))
    audit = pd.DataFrame([bureau_audit, balance_audit, previous_audit, installment_audit])
    return table, audit


def validate_feature_table(table: pd.DataFrame, expected_rows: int) -> None:
    if len(table) != expected_rows:
        raise AssertionError(f"Feature row count {len(table):,} != application rows {expected_rows:,}")
    if table[ID_COL].isna().any() or not table[ID_COL].is_unique:
        raise AssertionError("Feature table grain is not one row per SK_ID_CURR")
    if table[TARGET_COL].isna().any():
        raise AssertionError("TARGET alignment failed")
    leaked_ids = [c for c in table.columns if c.startswith("SK_ID_") and c != ID_COL]
    if leaked_ids:
        raise AssertionError(f"Relationship identifiers entered the predictor table: {leaked_ids}")
    feature_count = len(table.columns) - 2
    if not 60 <= feature_count <= MAX_FEATURES:
        raise AssertionError(f"Feature count {feature_count} is outside the approved 60-{MAX_FEATURES} range")
    numeric = table.select_dtypes(include=np.number)
    if np.isinf(numeric.to_numpy()).any():
        raise AssertionError("Feature table contains infinite values")


def feature_family_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = {
        "APP_": "Application",
        "BUREAU_": "Bureau",
        "BB_": "Bureau balance",
        "PREV_": "Previous applications",
        "INST_": "Installments",
    }
    for prefix, family in labels.items():
        cols = [c for c in table if c.startswith(prefix)]
        rows.append({"feature_family": family, "prefix": prefix, "feature_count": len(cols)})
    return pd.DataFrame(rows)
