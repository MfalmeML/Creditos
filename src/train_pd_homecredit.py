import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.adapters.application import ID, TARGET

DROP = [ID, TARGET]

def load():
    return pd.read_parquet('data/processed_dataset.parquet')

def build_pipeline(X):
    cat = X.select_dtypes(include=['object']).columns.tolist()
    num = X.select_dtypes(exclude=['object']).columns.tolist()
    pre = ColumnTransformer([
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat),
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num),
    ])
    return Pipeline([('pre', pre),
                     ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))])

def main():
    df = load()
    y = df[TARGET].astype(int)
    X = df.drop(columns=DROP)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)
    model = build_pipeline(X_tr)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    print('AUC:', round(roc_auc_score(y_te, p), 4))
    print('Brier:', round(brier_score_loss(y_te, p), 4))
    joblib.dump({'model': model, 'columns': list(X.columns)}, 'models/pd_homecredit.joblib')

if __name__ == '__main__':
    main()
