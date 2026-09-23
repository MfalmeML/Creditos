from src.governance import collect_governance

def test_governance_runs():
    r = collect_governance()
    assert 'model_risk' in r
    print('ok')

if __name__ == '__main__':
    test_governance_runs()
