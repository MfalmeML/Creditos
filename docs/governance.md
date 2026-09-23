# Model Risk Governance

## Tracked signals (all monitored separately)
- Data drift: KS test per feature, alert if p < alpha
- Prediction drift: KS test on score distribution
- Performance drift: AUC / Gini over time windows
- Calibration drift: predicted PD vs observed default rate per bin
- Fairness drift: approval-rate parity and error-rate parity per group
- Economic regime drift: macro scenario re-runs, updated quarterly

## Alert thresholds
- Data drift: > 3 features flagged in one window
- Prediction drift: KS p < 0.05
- Calibration: |predicted - observed| > 0.03 in any bin
- Fairness: parity ratio < 0.8 or > 1.25

## Response protocol
1. Alert fires -> log to governance_report.json
2. Investigate root cause (data, code, or regime)
3. If regime: update stress scenarios
4. If data or code: retrain with reject inference re-applied
5. Never auto-retrain on drifted data without understanding the drift

## Artifacts
- data/governance_report.json (latest)
- data/stress_report.csv (scenario ECL)
- models/*.joblib (versioned by git tag)
