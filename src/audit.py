from src.governance import collect_governance
import json

def audit():
    rep = collect_governance()
    alerts = []
    dd = rep.get('model_risk', {}).get('data_drift_flagged', 0)
    if dd > 3:
        alerts.append(f'data drift on {dd} features')
    pd_ = rep.get('model_risk', {}).get('prediction_drift', {})
    if pd_.get('drift'):
        alerts.append('prediction drift detected')
    return {'report': rep, 'alerts': alerts}

if __name__ == '__main__':
    out = audit()
    print(json.dumps(out['alerts'], indent=2))
