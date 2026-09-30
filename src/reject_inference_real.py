import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.adapters.bureau import load_bureau


def build_pipeline(X):
    cat_cols = X.select_dtypes(include=['category', 'object']).columns
    num_cols = X.select_dtypes(exclude=['category', 'object']).columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocess = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
    ])
    return Pipeline([
        ('prep', preprocess),
        ('clf', LogisticRegression(max_iter=1000)),
    ])


def main():
    df = load_bureau('data/application_train.csv')
    X = df.drop(columns=['target', 'CODE_GENDER'])
    y = df['target'].astype(int)

    # Same split as train_pd.py
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline arm: same pipeline, same split, approved population only
    baseline = build_pipeline(X_tr)
    baseline.fit(X_tr, y_tr)
    p_base_te = baseline.predict_proba(X_te)[:, 1]
    auc_base = roc_auc_score(y_te, p_base_te)
    print(f'baseline AUC (approved holdout): {auc_base:.4f}')

    # Hypothetical decline boundary from the baseline model's own scores.
    # Home Credit contains only originated loans; no real declined records exist.
    p_all = baseline.predict_proba(X)[:, 1]
    threshold = 0.15
    decline_mask = p_all > threshold
    approve_mask = ~decline_mask

    print(f'boundary threshold: {threshold}')
    print(f'approved share: {approve_mask.mean():.4f}')
    print(f'declined share: {decline_mask.mean():.4f}')
    print(f'TARGET among would-decline: {y[decline_mask].mean():.4f}')
    print(f'TARGET among would-approve: {y[approve_mask].mean():.4f}')

    # Parcelling: assign pseudo-labels to the declined region.
    # NOTE: this is a sensitivity study, not a correction of survivorship bias.
    # Real correction requires actual declined records, which this dataset lacks.
    k = 10
    p_declined = p_all[decline_mask]
    order = np.argsort(p_declined)
    declined_idx = np.where(decline_mask)[0][order]
    bins = np.array_split(declined_idx, k)

    y_pseudo = y.copy().values
    for i, b in enumerate(bins):
        rate = min(0.05 + 0.05 * (i + 1), 0.9)
        y_pseudo[b] = 0
        n_bad = int(round(rate * len(b)))
        y_pseudo[b[:n_bad]] = 1

    # Reject-aware arm: same pipeline, same split, trained on all applicants
    # with pseudo-labels in the declined region.
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X, pd.Series(y_pseudo, index=X.index),
        test_size=0.2, random_state=42, stratify=y
    )
    reject_aware = build_pipeline(X_tr2)
    reject_aware.fit(X_tr2, y_tr2)

    # Evaluate both arms on the same approved holdout only,
    # because real labels exist only for approved applicants.
    approved_holdout = approve_mask[X_te.index]
    auc_reject = roc_auc_score(
        y_te[approved_holdout],
        reject_aware.predict_proba(X_te[approved_holdout])[:, 1],
    )
    auc_base_holdout = roc_auc_score(
        y_te[approved_holdout],
        baseline.predict_proba(X_te[approved_holdout])[:, 1],
    )

    print(f'baseline AUC on approved holdout:        {auc_base_holdout:.4f}')
    print(f'reject-aware AUC on approved holdout:    {auc_reject:.4f}')
    print(f'delta:                                   {auc_reject - auc_base_holdout:+.4f}')

    # Do not save artifact until these numbers are reviewed.
    # joblib.dump({'model': reject_aware, 'columns': list(X.columns)},
    #             'models/pd_reject_aware_homecredit.joblib')


if __name__ == '__main__':
    main()
