import pandas as pd
from sklearn.datasets import fetch_openml

data = fetch_openml(name='credit-g', version=1, as_frame=True)
df = data.frame
print(df.shape)
print(df.head())
print(df['class'].value_counts())
