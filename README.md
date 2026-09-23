# CreditOS

End-to-end ML credit decision and loss-prevention platform.

## What it does
- Builds three feature profiles: credit, cash-flow, behavior
- Scores PD, LGD, EAD and computes ECL = PD x LGD x EAD
- Simulates candidate offers (limit, rate, term) and selects the constrained-optimal one
- Emits a decision record with top drivers, counterfactual, and fallback flag
- Monitors data drift, prediction drift, calibration, and fairness parity
- Serves a FastAPI endpoint, supports shadow mode and staged rollout

## Layout
- src/        modules (data, models, decisions, monitoring, API)
- tests/      unit and integration tests
- configs/    runtime settings and sample applicant
- docs/       decision record schema
- data/       generated artifacts (gitignored)

## Run
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src\train_pd.py
    python src\run_full_decisions.py
    pytest -q
    uvicorn src.api:app --port 8000

## Docker
    docker build -t creditos:latest .
    docker run -p 8000:8000 creditos:latest

## Scope
v1: PD + LGD + EAD + ECL, thin-file fallback, decision optimizer, explanation, reject inference, fairness metrics, drift monitoring, API, shadow mode.
Deferred: competing-risk (default/prepayment), continuous early warning, causal intervention, fraud unification, portfolio stress testing.
