import pandas as pd

ID = 'SK_ID_CURR'
TARGET = 'TARGET'


def load_application(path='data/application_train.csv'):
    """Lightweight raw-data loader for sanity checks only.
    Use src.adapters.bureau.load_bureau() for the actual PD training
    pipeline — that function applies the real feature mapping and bureau
    merge this one does not.
    """
    df = pd.read_csv(path)
    assert ID in df.columns and TARGET in df.columns
    return df

if __name__ == '__main__':
    df = load_application()
    print(df.shape)
    print(df[TARGET].mean())
