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

    bureau = pd.read_csv('data/bureau.csv')
    agg = bureau.groupby('SK_ID_CURR').agg(
        bureau_credit_count=('SK_ID_BUREAU', 'count'),
        bureau_days_overdue_max=('CREDIT_DAY_OVERDUE', 'max'),
        bureau_credit_sum=('AMT_CREDIT_SUM', 'sum'),
    ).reset_index()
    df = df.merge(agg, on='SK_ID_CURR', how='left')

    keep = [
        'age', 'credit_amount', 'duration', 'employment_years',
        'employment_type', 'AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'CNT_CHILDREN',
        'EXT_SOURCE_1', 'EXT_SOURCE_1_missing', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'bureau_credit_count', 'bureau_days_overdue_max', 'bureau_credit_sum',
        'CODE_GENDER', 'target'
    ]
    return df[keep].copy()
