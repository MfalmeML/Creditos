import pandas as pd


def approval_rate_parity(df, group_col, decision_col='approved'):
    return df.groupby(group_col)[decision_col].mean().rename('approval_rate')


def error_rate_parity(df, group_col, y_true='y', y_pred='yhat'):
    df = df.copy()
    df['err'] = (df[y_true] != df[y_pred]).astype(int)
    return df.groupby(group_col)['err'].mean().rename('error_rate')


if __name__ == '__main__':
    d = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'A', 'B'],
        'approved': [1, 0, 1, 0, 1, 0],
        'y': [0, 1, 0, 1, 0, 1],
        'yhat': [0, 1, 1, 1, 0, 0],
    })
    print(approval_rate_parity(d, 'group').to_string())
    print(error_rate_parity(d, 'group').to_string())
