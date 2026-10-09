
"""Adapter for the historical LendingClub accepted-loan dataset.

Source-specific fields are mapped to a documented loan-level schema.
Outcome and repayment fields are retained for analysis, but should not
be used as predictors for an origination-time PD model.
"""

from pathlib import Path
import re

import pandas as pd


DEFAULT_PATH = "data/raw/lendingclub/accepted_2007_to_2018Q4.csv"

SOURCE_COLUMNS = [
    "id",
    "issue_d",
    "loan_status",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "out_prncp",
    "total_pymnt",
    "total_rec_prncp",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
]

# Explicit label policy for the first historical binary baseline.
DEFAULT_LABELS = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
}


def _transform(chunk: pd.DataFrame) -> pd.DataFrame:
    """Map one source-data chunk to the Creditos LendingClub schema."""
    missing = set(SOURCE_COLUMNS) - set(chunk.columns)
    if missing:
        raise ValueError(
            f"LendingClub dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    df = chunk.copy()

    # Preserve source values so the mapping remains auditable.
    df["source_loan_status"] = df["loan_status"].astype("string").str.strip()
    df["target_default"] = df["source_loan_status"].map(DEFAULT_LABELS).astype("Int64")
    df["label_eligible"] = df["target_default"].notna()

    term = df["term"].astype("string").str.extract(r"(\d+)")[0]
    interest_rate = (
        df["int_rate"]
        .astype("string")
        .str.replace("%", "", regex=False)
    )

    result = pd.DataFrame(index=df.index)
    result["loan_id"] = df["id"].astype("string")
    result["origination_date"] = pd.to_datetime(
        df["issue_d"], format="%b-%Y", errors="coerce"
    )
    result["source_loan_status"] = df["source_loan_status"]
    result["target_default"] = df["target_default"]
    result["label_eligible"] = df["label_eligible"]

    # Origination/application fields: candidate PD predictors.
    result["loan_amount"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    result["funded_amount"] = pd.to_numeric(df["funded_amnt"], errors="coerce")
    result["term_months"] = pd.to_numeric(term, errors="coerce").astype("Int64")
    result["interest_rate_pct"] = pd.to_numeric(
        interest_rate, errors="coerce"
    )
    result["annual_income"] = pd.to_numeric(df["annual_inc"], errors="coerce")
    result["debt_to_income"] = pd.to_numeric(df["dti"], errors="coerce")
    result["fico_low"] = pd.to_numeric(df["fico_range_low"], errors="coerce")
    result["fico_high"] = pd.to_numeric(df["fico_range_high"], errors="coerce")

    # Post-origination outcomes: for analysis/target construction only.
    # Do not include these in an origination-time PD feature matrix.
    result["outstanding_principal"] = pd.to_numeric(
        df["out_prncp"], errors="coerce"
    )
    result["total_payments"] = pd.to_numeric(df["total_pymnt"], errors="coerce")
    result["principal_recovered"] = pd.to_numeric(
        df["total_rec_prncp"], errors="coerce"
    )
    result["recoveries"] = pd.to_numeric(df["recoveries"], errors="coerce")
    result["collection_recovery_fee"] = pd.to_numeric(
        df["collection_recovery_fee"], errors="coerce"
    )
    result["last_payment_date"] = pd.to_datetime(
        df["last_pymnt_d"], format="%b-%Y", errors="coerce"
    )

    return result.reset_index(drop=True)


def iter_lendingclub(
    path: str | Path = DEFAULT_PATH,
    chunksize: int = 100_000,
):
    """Yield transformed DataFrame chunks to limit memory use."""
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LendingClub CSV not found: {path}")

    for chunk in pd.read_csv(
        path,
        usecols=SOURCE_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    ):
        yield _transform(chunk)


def load_lendingclub(
    path: str | Path = DEFAULT_PATH,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a whole dataset or a small sample.

    For the full historical dataset, prefer iter_lendingclub() to avoid
    holding the entire transformed dataset in memory.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LendingClub CSV not found: {path}")

    chunk = pd.read_csv(
        path,
        usecols=SOURCE_COLUMNS,
        nrows=nrows,
        low_memory=False,
    )
    return _transform(chunk)
