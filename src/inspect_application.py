import pandas as pd
from pathlib import Path

path = next(Path('data').rglob('application_train.csv'))
df = pd.read_csv(path)
print('path:', path)
print('shape:', df.shape)
print('TARGET rate:', df['TARGET'].mean())
print('columns:', len(df.columns))
print('SK_ID_CURR unique:', df['SK_ID_CURR'].nunique())
