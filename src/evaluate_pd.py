import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

from src.adapters.bureau import load_bureau
from src.fairness import approval_rate_parity, error_rate_parity


df = load_bureau('data/application_train.csv')
gender = df['CODE_GENDER']
X = df.drop(columns=['target', 'CODE_GENDER'])
y = df['target'].astype(int)

X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X, y, gender, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load('models/pd_baseline.joblib')
probs = model.predict_proba(X_test)[:, 1]

mean_pred, frac_pos = calibration_curve(y_test, probs, n_bins=10, strategy='quantile')
print('Calibration (predicted PD vs actual default rate, by decile):')
for p, f in zip(mean_pred, frac_pos):
    print(f'  predicted={p:.3f}  actual={f:.3f}')

plt.figure()
plt.plot(mean_pred, frac_pos, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel('Mean predicted PD')
plt.ylabel('Observed default rate')
plt.title('PD Calibration - Home Credit real data')
plt.legend()
plt.savefig('docs/calibration_curve.png', dpi=150, bbox_inches='tight')
print('Saved docs/calibration_curve.png')

threshold = float(y_test.mean())
print()
print('Actual default rate by gender:')
print(pd.DataFrame({
    'gender': gender_test.values,
    'y': y_test.values,
}).loc[lambda d: d['gender'].isin(['M', 'F'])].groupby('gender')['y'].mean().rename('base_default_rate').to_string())
print()
print(f'Fairness evaluation using threshold={threshold:.4f} (observed default rate):')

eval_df = pd.DataFrame({
    'gender': gender_test.values,
    'y': y_test.values,
    'yhat': (probs >= threshold).astype(int),
    'approved': (probs < threshold).astype(int),
})
eval_df = eval_df[eval_df['gender'].isin(['M', 'F'])]

print()
print('Predicted PD distribution by gender:')
print(pd.DataFrame({'gender': gender_test.values, 'pred_pd': probs}).groupby('gender')['pred_pd'].describe().to_string())
print()
print('Approval rate by gender:')
print(approval_rate_parity(eval_df, 'gender').to_string())
print()
print('Error rate by gender:')
print(error_rate_parity(eval_df, 'gender').to_string())
