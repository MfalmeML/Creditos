import pandas as pd

ID = 'SK_ID_CURR'
BUREAU_ID = 'SK_ID_BUREAU'

def load_bureau(path='data/raw/bureau.csv'):
    return pd.read_csv(path)

def aggregate_bureau(df):
    g = df.groupby(ID)
    feats = pd.DataFrame({
        'bureau_n': g.size(),
        'bureau_active_n': g['CREDIT_ACTIVE'].apply(lambda s: (s == 'Active').sum()),
        'bureau_closed_n': g['CREDIT_ACTIVE'].apply(lambda s: (s == 'Closed').sum()),
        'bureau_days_credit_mean': g['DAYS_CREDIT'].mean(),
        'bureau_days_credit_min': g['DAYS_CREDIT'].min(),
        'bureau_days_enddate_mean': g['DAYS_CREDIT_ENDDATE'].mean(),
        'bureau_credit_sum': g['AMT_CREDIT_SUM'].sum(),
        'bureau_credit_sum_mean': g['AMT_CREDIT_SUM'].mean(),
        'bureau_debt_sum': g['AMT_CREDIT_SUM_DEBT'].sum(),
        'bureau_overdue_sum': g['AMT_CREDIT_SUM_OVERDUE'].sum(),
        'bureau_overdue_max': g['AMT_CREDIT_SUM_OVERDUE'].max(),
        'bureau_prolong_max': g['CNT_CREDIT_PROLONG'].max(),
    })
    return feats.reset_index()

if __name__ == '__main__':
    b = load_bureau()
    agg = aggregate_bureau(b)
    print(agg.shape)
    print(agg.head())
