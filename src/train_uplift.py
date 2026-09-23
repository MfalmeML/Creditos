import pandas as pd
import joblib
from src.uplift import TLearner

df = pd.read_csv('data/intervention.csv')
feats = ['age', 'credit_amount']
model = TLearner().fit(df[feats], df['treatment'], df['default'])
df['uplift'] = model.predict_uplift(df[feats])
print(df.groupby('treatment')['uplift'].mean())
joblib.dump(model, 'models/uplift_tlearner.joblib')
