import pandas as pd
from src.adapters.application import load_application, ID, TARGET
from src.adapters.bureau import load_bureau, aggregate_bureau

def build_dataset():
    app = load_application()
    bur = aggregate_bureau(load_bureau())
    df = app.merge(bur, on=ID, how='left')
    return df

if __name__ == '__main__':
    df = build_dataset()
    print(df.shape)
    print('missing bureau (thin-file proxy):', df['bureau_n'].isna().mean())
    df.to_parquet('data/processed_dataset.parquet')
