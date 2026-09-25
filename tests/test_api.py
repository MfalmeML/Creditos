from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200

def test_score():
    payload = {
        'age': 35, 'credit_amount': 8000, 'duration': 24,
        'employment': '1<=X<4', 'purpose': 'radio/tv',
        'checking_status': '0<=X<200', 'savings_status': '<100',
    }
    r = client.post('/score', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 'pd' in body and 'ecl' in body
    print('api ok')

if __name__ == '__main__':
    test_health()
    test_score()
