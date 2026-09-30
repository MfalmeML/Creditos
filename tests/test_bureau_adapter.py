from src.adapters.bureau import load_bureau

def test_load_bureau_shape():
    df = load_bureau()
    expected = {"age", "credit_amount", "duration", "target", "CODE_GENDER",
                "EXT_SOURCE_1_missing", "bureau_credit_count"}
    missing = expected - set(df.columns)
    assert not missing, f"load_bureau() missing expected columns: {missing}"
