# Full rebuild from scratch
python src\train_pd.py
python src\pipeline_ecl.py
python src\run_full_decisions.py
python src\fit_hazards.py
python src\uplift.py
python src\fraud_credit_unified.py
python src\stress_report.py
python src\governance.py
python src\audit.py
pytest -q
