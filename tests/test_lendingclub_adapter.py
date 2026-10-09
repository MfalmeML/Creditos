
import pandas as pd
import pytest

from src.adapters.lendingclub import (
    SOURCE_COLUMNS,
    iter_lendingclub,
    load_lendingclub,
)


def sample_data():
    return pd.DataFrame([
        {
            "id": 101,
            "issue_d": "Jan-2018",
            "loan_status": "Fully Paid",
            "loan_amnt": 1000,
            "funded_amnt": 1000,
            "term": "36 months",
            "int_rate": "10.5%",
            "annual_inc": 30000,
            "dti": 12.0,
            "fico_range_low": 680,
            "fico_range_high": 684,
            "out_prncp": 0,
            "total_pymnt": 1100,
            "total_rec_prncp": 1000,
            "recoveries": 0,
            "collection_recovery_fee": 0,
            "last_pymnt_d": "Jan-2021",
        },
        {
            "id": 102,
            "issue_d": "Feb-2018",
            "loan_status": "Charged Off",
            "loan_amnt": 2000,
            "funded_amnt": 2000,
            "term": "60 months",
            "int_rate": "15.0%",
            "annual_inc": 40000,
            "dti": 20.0,
            "fico_range_low": 620,
            "fico_range_high": 624,
            "out_prncp": 0,
            "total_pymnt": 500,
            "total_rec_prncp": 300,
            "recoveries": 100,
            "collection_recovery_fee": 10,
            "last_pymnt_d": "Jun-2019",
        },
        {
            "id": 103,
            "issue_d": "Mar-2018",
            "loan_status": "Current",
            "loan_amnt": 1500,
            "funded_amnt": 1500,
            "term": "36 months",
            "int_rate": "12.0%",
            "annual_inc": 35000,
            "dti": 15.0,
            "fico_range_low": 660,
            "fico_range_high": 664,
            "out_prncp": 500,
            "total_pymnt": 700,
            "total_rec_prncp": 500,
            "recoveries": 0,
            "collection_recovery_fee": 0,
            "last_pymnt_d": "Aug-2020",
        },
    ])


def write_fixture(tmp_path):
    path = tmp_path / "lendingclub_sample.csv"
    sample_data().to_csv(path, index=False)
    return path


def test_load_maps_statuses_and_fields(tmp_path):
    path = write_fixture(tmp_path)
    result = load_lendingclub(path)

    assert len(result) == 3
    assert result["loan_id"].tolist() == ["101", "102", "103"]
    assert result["target_default"].iloc[0] == 0
    assert result["target_default"].iloc[1] == 1
    assert pd.isna(result["target_default"].iloc[2])
    assert result["label_eligible"].tolist() == [True, True, False]
    assert result["term_months"].tolist() == [36, 60, 36]
    assert result["interest_rate_pct"].tolist() == [10.5, 15.0, 12.0]


def test_chunked_reader_preserves_rows(tmp_path):
    path = write_fixture(tmp_path)

    chunks = list(iter_lendingclub(path, chunksize=2))

    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert sum(len(chunk) for chunk in chunks) == 3


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_lendingclub("this_file_does_not_exist.csv")


def test_missing_required_column_raises():
    incomplete = sample_data().drop(columns=["loan_status"])

    with pytest.raises(ValueError, match="missing required columns"):
        from src.adapters.lendingclub import _transform
        _transform(incomplete)
