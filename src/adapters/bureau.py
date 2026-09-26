import numpy as np
import pandas as pd


def load_bureau(path):
    df = pd.read_csv(path)

    if 'TARGET' not in df.columns:
        raise ValueError(f"Expected a TARGET column in the Home Credit dataset: {path}")

    df['age'] = (-df['DAYS_BIRTH'] / 365).round(1)
    df['credit_amount'] = df['AMT_CREDIT']
    df['duration'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']).replace([np.inf, -np.inf], np.nan)

    employed = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['employment_years'] = (-employed / 365).round(1)
    df['employment_type'] = df['NAME_INCOME_TYPE']
    df['EXT_SOURCE_1_missing'] = df['EXT_SOURCE_1'].isna().astype(int)

    df['target'] = df['TARGET']

    keep = [
        'age', 'credit_amount', 'duration', 'employment_years',
        'employment_type', 'AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CNT_CHILDREN',
        'EXT_SOURCE_1', 'EXT_SOURCE_1_missing', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'target'
    ]
    return df[keep].copy()
