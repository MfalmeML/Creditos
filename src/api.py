from fastapi import FastAPI
from src.api_schemas import Applicant, Decision
from src.predictor import score

app = FastAPI(title='CreditOS Decision API')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/score', response_model=Decision)
def score_endpoint(applicant: Applicant):
    result = score(applicant.dict())
    approved = result['limit'] is not None
    return Decision(approved=approved, **result)
