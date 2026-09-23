import pandas as pd
from src.early_warning import flag_alerts

def test_alert_labels():
    d = pd.DataFrame({'account_id': ['A'], 'pd_horizon': [0.20]})
    out = flag_alerts(d)
    assert out['status'].iloc[0] == 'alert'
    print('ok')

if __name__ == '__main__':
    test_alert_labels()
