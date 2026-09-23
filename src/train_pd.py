from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

data = fetch_openml(name='credit-g', version=1, as_frame=True)
df = data.frame
X = df.drop(columns=['class'])
y = (df['class'] == 'bad').astype(int)

cat_cols = X.select_dtypes(include=['category', 'object']).columns
num_cols = X.select_dtypes(exclude=['category', 'object']).columns

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

model = Pipeline([
    ('prep', preprocess),
    ('clf', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f'AUC: {auc:.4f}')
joblib.dump(model, 'models/pd_baseline.joblib')
