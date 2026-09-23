from src.fraud_credit_unified import generate_unified_data, train_router

def test_router():
    df = generate_unified_data(n=800)
    m, feats = train_router(df)
    pred = m.predict(df[feats])
    assert len(pred) == len(df)
    print('ok')

if __name__ == '__main__':
    test_router()
