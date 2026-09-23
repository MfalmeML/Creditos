# Intervention assignment: causal identification assumptions

v1 uses a T-learner trained on data with randomized treatment assignment
(50/50 in the synthetic generator). In production, intervention decisions
were historically not randomized.

Required before deploying on real data:
- Either an A/B test with randomized intervention assignment
- Or a validated causal-identification approach (IPW, doubly-robust)
  with explicit confounder set, checked for balance and overlap

The treatment effect estimate is only as credible as the identification
strategy behind it. Do not treat uplift scores as causal without one.
