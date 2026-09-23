import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

class TLearner:
    def __init__(self):
        self.m0 = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        self.m1 = GradientBoostingClassifier(n_estimators=100, max_depth=3)

    def fit(self, X, treatment, y):
        X = pd.DataFrame(X)
        treatment = pd.Series(treatment).values
        y = pd.Series(y).values
        self.m0.fit(X[treatment == 0], y[treatment == 0])
        self.m1.fit(X[treatment == 1], y[treatment == 1])
        return self

    def predict_uplift(self, X):
        X = pd.DataFrame(X)
        p0 = self.m0.predict_proba(X)[:, 1]
        p1 = self.m1.predict_proba(X)[:, 1]
        return p1 - p0  # negative = intervention reduces default

if __name__ == '__main__':
    df = pd.read_csv('data/intervention.csv')
    feats = ['age', 'credit_amount']
    model = TLearner().fit(df[feats], df['treatment'], df['default'])
    df['uplift'] = model.predict_uplift(df[feats])
    print(df.groupby('treatment')['uplift'].mean())
    joblib.dump(model, 'models/uplift_tlearner.joblib')
