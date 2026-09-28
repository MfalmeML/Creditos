# Model Evaluation: Real Data (Home Credit Default Risk)

## Summary

This model was originally built and validated against the synthetic/proxy
German Credit dataset (via sklearn's fetch_openml). This document reports
the results of re-validating the same pipeline against real data
(Kaggle's Home Credit Default Risk dataset) and states plainly where the
real-data results fall short of the synthetic baseline and why.

## 1. Headline performance

| Dataset            | AUC     |
|---------------------|---------|
| Synthetic (German Credit) | 0.8083 |
| Real (Home Credit)        | 0.7340 |
| Tolerance floor (baseline - 0.05) | 0.7583 |

The real-data AUC falls outside the tolerance floor set for this project.
Two isolated interventions were tested to close the gap:

- Added an explicit missingness flag for `EXT_SOURCE_1` (56% null in the
  raw data): AUC 0.7318 -> 0.7338 (+0.0020)
- Added aggregated bureau-history features (credit count, max days
  overdue, total credit sum) via a merge against `bureau.csv`:
  AUC 0.7338 -> 0.7340 (+0.0002)

Both interventions were verified to have executed correctly (checked for
null rates and value spread on the new columns) before being credited or
discredited. Neither closed a meaningful share of the gap. This is treated
as evidence that the gap is due to a thinner real feature set (no direct
equivalents for `purpose`, `checking_status`, `savings_status` exist in
this dataset) rather than an engineering defect in the pipeline.

**Conclusion: real AUC of 0.7340 is the honest number for this project,
not 0.8083.**

## 2. Calibration

Calibration was checked on the held-out test split using 10 quantile bins
(predicted PD vs. observed default rate per bin). The model is
well-calibrated on real data: predicted deciles tracked observed default
rates closely across the range (e.g. predicted 0.141 vs. observed 0.140;
predicted 0.250 vs. observed 0.245). See `docs/calibration_curve.png`.

## 3. Fairness (gender)

`CODE_GENDER` was excluded from model training features throughout.
Rows coded `XNA` (unknown/undisclosed) were excluded from the fairness
comparison itself.

### 3.1 Naive comparison (does not hold up)

A first pass compared raw error rates by gender at threshold=0.5:
F 0.070 vs. M 0.101. This threshold approves ~99.8% of applicants
regardless of group, so this comparison carries almost no information
about the model's actual decisions and was discarded.

### 3.2 Base rates and predicted risk

| Gender | Actual default rate | Mean predicted PD |
|--------|---------------------|--------------------|
| F      | 6.99%               | 7.47%              |
| M      | 10.17%              | 9.18%              |

The model's predicted-risk gap between groups (1.7 points) is smaller
than the true outcome gap (3.2 points). The model does not over-separate
by gender relative to real risk in this data.

### 3.3 Approval-rate disparity under a shared threshold

At a threshold matched to the overall population default rate (0.0807):

| Gender | Approval rate |
|--------|----------------|
| F      | 68.8%          |
| M      | 58.2%          |

A 10.6-point gap — much larger than the 3.2-point true risk difference
would predict under fair, risk-based decisioning.

### 3.4 Isolating the mechanism: threshold policy, not model bias

Approving each group against its *own* base rate instead of one shared
cutoff:

| Gender | Approval rate (own base rate) |
|--------|-------------------------------|
| F      | 62.5%                         |
| M      | 68.7%                         |

The gap reverses direction entirely under group-specific thresholds. This
confirms the 10.6-point disparity under 3.2 is a **shared-threshold
artifact** driven by the two groups' score distributions differing in
both mean and spread — not evidence that the underlying model
overstates risk for either group.

### 3.5 Conclusion

This is a **disparate-impact finding attributable to threshold policy**,
not a model-fairness defect. The two would call for different remedies:
threshold/policy review (e.g. group-conditional or business-calibrated
cutoffs) vs. retraining or feature auditing. Given the model's own risk
assessment tracks true outcomes closely (3.2's section), retraining is
not indicated by this evidence; a policy-level review of the shared
cutoff is.

## 4. Scope and limitations

- This is a single-dataset, single-model artifact (logistic regression
  PD only). It does not cover EAD, LGD, or downstream decisioning on
  real data — those remain built on synthetic/simulated logic
  (`estimate_ead`, `estimate_lgd`) and have not been validated against
  real recovery or exposure data.
- Transaction-stream data was never available for this project and is
  explicitly out of scope (see `docs/data_inventory.md`).
- The fairness analysis covers gender only; other protected attributes
  (age, family status effects) were not examined here.
- Per project scoping (step 250, Path B): this is the deliberately
  scoped artifact. Further capability work (uplift, survival models,
  monitoring) was deprioritized in favor of validating this one model on
  real data honestly.
