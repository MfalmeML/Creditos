import pandas as pd

ID = 'SK_ID_CURR'
TARGET = 'TARGET'

def load_application(path='data/raw/application_train.csv'):
    df = pd.read_csv(path)
    assert ID in df.columns and TARGET in df.columns
    return df

if __name__ == '__main__':
    df = load_application()
    print(df.shape)
    print(df[TARGET].mean())
