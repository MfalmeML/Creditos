from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200

def test_score():
    payload = {
        'age': 35,
        'credit_amount': 8000,
        'duration': 24,
        'employment': '1<=X<4',
        'purpose': 'radio/tv',
        'checking_status': '0<=X<200',
        'savings_status': '<100',
        'credit_history': 'existing paid',
        'personal_status': 'male single',
        'other_parties': 'none',
        'residence_since': 2,
        'property_magnitude': 'car',
        'other_payment_plans': 'none',
        'housing': 'own',
        'existing_credits': 1,
        'job': 'skilled',
        'num_dependents': 1,
        'own_telephone': 'yes',
        'foreign_worker': 'yes',
        'installment_commitment': 3,
    }
    r = client.post('/score', json=payload)
    print(r.status_code, r.text)
    assert r.status_code == 200
    body = r.json()
    assert 'pd' in body and 'ecl' in body
    print('api ok')

if __name__ == '__main__':
    test_health()
    test_score()
