import pandas as pd

def flag_alerts(scores_df, threshold=0.15, watch_threshold=0.08):
    def label(p):
        if p >= threshold:
            return 'alert'
        if p >= watch_threshold:
            return 'watch'
        return 'healthy'
    out = scores_df.copy()
    out['status'] = out['pd_horizon'].apply(label)
    return out

if __name__ == '__main__':
    d = pd.DataFrame({'account_id': ['A1','A2','A3'], 'pd_horizon': [0.05, 0.10, 0.22]})
    print(flag_alerts(d).to_string())
