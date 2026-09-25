# CreditOS

AI credit risk, loss prevention and decision intelligence platform.

An end-to-end ML system that continuously assesses borrower risk,
predicts expected credit loss, optimizes lending decisions and
exposure, detects financial distress before default, selects
loss-prevention interventions, and stress-tests the portfolio
under changing economic conditions.

## Capabilities

| Layer | Module | Question answered |
|---|---|---|
| Feature intelligence | src/predictor.py | What do we know about this applicant? |
| Risk core | src/lgd.py, src/ead.py, src/ecl.py | PD x LGD x EAD = ECL |
| Thin-file fallback | src/thin_file.py | Score when bureau history is insufficient |
| Decision optimizer | src/select_offer.py, src/constraints.py | Which offer maximizes expected profit? |
| Explanation | src/explain.py, src/counterfactual.py | Why this decision, and what would change it? |
| Reject inference | src/reject_inference_v2.py | What is the risk in the population we already decline? |
| Competing risks | src/competing_risks.py | Default vs prepayment, jointly |
| Continuous monitoring | src/monitor_accounts.py | Has risk changed since origination? |
| Early warning | src/early_warning.py | Who is about to become delinquent? |
| Causal intervention | src/uplift.py, src/intervention.py | Who responds to intervention, and how? |
| Fraud + credit | src/fraud_credit_unified.py | Distress or abuse? |
| Graph risk | src/graph_risk.py | Who shares devices, accounts, merchants? |
| Fairness | src/fairness.py | Are errors equal across groups? |
| Stress testing | src/portfolio_stress.py | What happens to the book under stress? |
| Governance | src/governance.py, src/audit.py | Is the model still trustworthy? |

## Central objective

Maximize Expected Decision Value subject to:

- Expected Loss <= Risk appetite
- Fairness metrics >= Minimum standard
- Decision latency <= Operational constraint
- Model calibration >= Required level
- Capital exposure <= Constraint

## Quickstart

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

    python src\cli.py train-pd
    python src\cli.py ecl
    python src\cli.py decide
    python src\cli.py survival
    python src\cli.py uplift
    python src\cli.py stress
    python src\cli.py audit

## Serve

    python src\cli.py serve

Endpoints:

- GET  /health
- POST /score    body = configs/sample_applicant.json

## Docker

    docker build -t creditos:latest .
    docker run -p 8000:8000 creditos:latest

## Test

    pytest -q

## Layout

    src/        modules (data, models, decisions, monitoring, API)
    tests/      unit, integration, end-to-end
    configs/    runtime settings
    docs/       decision record schema, governance, causal assumptions
    data/       generated artifacts (gitignored)
    models/     fitted model artifacts

## Scope and phases

- v0.1 Foundation: PD + LGD + EAD + ECL, thin-file, decision optimizer, explanation
- v0.2 Lifecycle: competing-risk survival, continuous monitoring, early warning
- v0.3 Loss prevention: causal/uplift intervention optimization
- v0.4 Governance: reject inference v2, fraud/credit unification, graph risk
- v0.5 Enterprise risk: portfolio stress testing, model-risk governance

## Non-negotiable constraints

- Point-in-time feature discipline: no feature available after the decision point
  may leak into training.
- Reject inference applied to every retrain, not just the first build.
- Every decision produces a reproducible explanation against the model version
  that produced it.
- No model fails open: if a risk component is unavailable, decline to the
  conservative pre-approved policy tier.
