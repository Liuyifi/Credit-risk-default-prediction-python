"""Raw-data contracts and pre-application temporal filters.

The project deliberately keeps source loading explicit.  Every relational row used
for features must be observable no later than the current application date (day 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Union

import pandas as pd


RAW_FILES = (
    "application_train.csv",
    "HomeCredit_columns_description.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
)


REQUIRED_COLUMNS: Mapping[str, set[str]] = {
    "application_train.csv": {
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_CONTRACT_TYPE",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED", "OWN_CAR_AGE",
        "REGION_POPULATION_RELATIVE",
    },
    "bureau.csv": {
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT",
        "DAYS_CREDIT_UPDATE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG",
        "AMT_ANNUITY",
    },
    "bureau_balance.csv": {"SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"},
    "previous_application.csv": {
        "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS",
        "DAYS_DECISION", "AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY",
    },
    "installments_payments.csv": {
        "SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT", "AMT_PAYMENT",
    },
}


USECOLS = {name: sorted(cols) for name, cols in REQUIRED_COLUMNS.items()}


def raw_paths(raw_dir: Union[Path, str]) -> Dict[str, Path]:
    root = Path(raw_dir)
    return {name: root / name for name in RAW_FILES}


def validate_raw_files(raw_dir: Union[Path, str]) -> pd.DataFrame:
    """Validate file presence and required schemas without loading full tables."""
    rows = []
    for name, path in raw_paths(raw_dir).items():
        if not path.exists():
            raise FileNotFoundError(f"Required raw file is missing: {path}")
        encoding = "latin1" if name == "HomeCredit_columns_description.csv" else "utf-8"
        header = pd.read_csv(path, nrows=0, encoding=encoding)
        required = REQUIRED_COLUMNS.get(name, set())
        missing = sorted(required.difference(header.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
        rows.append({"file": name, "bytes": path.stat().st_size, "columns": len(header.columns)})
    return pd.DataFrame(rows)


def load_application(raw_dir: Union[Path, str]) -> pd.DataFrame:
    path = Path(raw_dir) / "application_train.csv"
    frame = pd.read_csv(path, usecols=USECOLS[path.name])
    validate_application_contract(frame)
    return frame


def load_bureau(raw_dir: Union[Path, str], borrower_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    frame = pd.read_csv(Path(raw_dir) / "bureau.csv", usecols=USECOLS["bureau.csv"])
    if borrower_ids is not None:
        frame = frame[frame["SK_ID_CURR"].isin(set(borrower_ids))].copy()
    return frame


def load_previous(raw_dir: Union[Path, str], borrower_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
    frame = pd.read_csv(
        Path(raw_dir) / "previous_application.csv",
        usecols=USECOLS["previous_application.csv"],
    )
    if borrower_ids is not None:
        frame = frame[frame["SK_ID_CURR"].isin(set(borrower_ids))].copy()
    return frame


def iter_csv(
    raw_dir: Union[Path, str],
    filename: str,
    chunksize: int = 1_000_000,
) -> Iterable[pd.DataFrame]:
    """Stream a large relationship table using only approved columns."""
    return pd.read_csv(
        Path(raw_dir) / filename,
        usecols=USECOLS[filename],
        chunksize=chunksize,
    )


def validate_application_contract(frame: pd.DataFrame) -> None:
    if frame["SK_ID_CURR"].isna().any() or not frame["SK_ID_CURR"].is_unique:
        raise ValueError("application_train must have one non-null row per SK_ID_CURR")
    if set(frame["TARGET"].dropna().unique()) != {0, 1}:
        raise ValueError("TARGET must be binary and contain both classes")


def temporal_filter_bureau(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = frame["DAYS_CREDIT"].le(0) & frame["DAYS_CREDIT_UPDATE"].le(0)
    kept = frame.loc[mask].copy()
    return kept, {
        "source": "bureau",
        "input_rows": int(len(frame)),
        "retained_rows": int(len(kept)),
        "excluded_rows": int((~mask).sum()),
        "rule": "DAYS_CREDIT <= 0 and DAYS_CREDIT_UPDATE <= 0",
    }


def temporal_filter_previous(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = frame["DAYS_DECISION"].lt(0)
    kept = frame.loc[mask].copy()
    return kept, {
        "source": "previous_application",
        "input_rows": int(len(frame)),
        "retained_rows": int(len(kept)),
        "excluded_rows": int((~mask).sum()),
        "rule": "DAYS_DECISION < 0",
    }


def temporal_filter_bureau_balance(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = frame["MONTHS_BALANCE"].le(0)
    kept = frame.loc[mask].copy()
    return kept, {
        "source": "bureau_balance",
        "input_rows": int(len(frame)),
        "retained_rows": int(len(kept)),
        "excluded_rows": int((~mask).sum()),
        "rule": "MONTHS_BALANCE <= 0",
    }


def temporal_filter_installments(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = (
        frame["DAYS_INSTALMENT"].notna()
        & frame["DAYS_ENTRY_PAYMENT"].notna()
        & frame["DAYS_INSTALMENT"].le(0)
        & frame["DAYS_ENTRY_PAYMENT"].le(0)
    )
    kept = frame.loc[mask].copy()
    return kept, {
        "source": "installments_payments",
        "input_rows": int(len(frame)),
        "retained_rows": int(len(kept)),
        "excluded_rows": int((~mask).sum()),
        "rule": "non-null DAYS_INSTALMENT and DAYS_ENTRY_PAYMENT, both <= 0",
    }


def assert_temporal_contracts(
    bureau: pd.DataFrame,
    previous: pd.DataFrame,
    installments: Optional[pd.DataFrame] = None,
    bureau_balance: Optional[pd.DataFrame] = None,
) -> None:
    if not (bureau["DAYS_CREDIT"].le(0) & bureau["DAYS_CREDIT_UPDATE"].le(0)).all():
        raise AssertionError("Post-application bureau rows remain")
    if not previous["DAYS_DECISION"].lt(0).all():
        raise AssertionError("Non-historical previous applications remain")
    if installments is not None and not (
        installments["DAYS_INSTALMENT"].le(0) & installments["DAYS_ENTRY_PAYMENT"].le(0)
    ).all():
        raise AssertionError("Post-application installment rows remain")
    if bureau_balance is not None and not bureau_balance["MONTHS_BALANCE"].le(0).all():
        raise AssertionError("Post-application bureau-balance rows remain")
