import pandas as pd
import joblib

def recommend_actions(df, model, pd_col='pd_horizon',
                      alert=0.15, watch=0.08, effect_threshold=-0.05):
    rows = []
    for _, r in df.iterrows():
        te = model.m1.predict_proba(pd.DataFrame([r[['age','credit_amount']]]))[:, 1][0] - \
             model.m0.predict_proba(pd.DataFrame([r[['age','credit_amount']]]))[:, 1][0]
        pd_h = r.get(pd_col, 0.0)
        if pd_h >= alert and te <= effect_threshold:
            action = 'immediate_intervention'
        elif pd_h >= watch and te <= effect_threshold:
            action = 'restructure'
        elif pd_h >= alert:
            action = 'monitor_no_action'
        else:
            action = 'no_action'
        rows.append({'account_id': r.get('account_id'), 'pd_horizon': pd_h,
                     'treatment_effect': float(te), 'action': action})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    model = joblib.load('models/uplift_tlearner.joblib')
    d = pd.DataFrame({
        'account_id': ['A1','A2','A3','A4'],
        'age': [30, 55, 25, 60],
        'credit_amount': [12000, 2000, 15000, 3000],
        'pd_horizon': [0.22, 0.18, 0.10, 0.02],
    })
    print(recommend_actions(d, model).to_string())
