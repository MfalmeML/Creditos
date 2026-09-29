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

## 5. Path decision and final project status

Per the project checklist's step 250, two paths were available: Path A
(acquire real data for every stage within two weeks) or Path B (ship one
honest, scoped artifact and stop).

Before committing to either, every file in `data/` was inventoried and
traced to its producer script, not judged by filename alone. Three files
whose names suggested they might contain real loan-performance or
experimental data were checked directly:

- `loan_events.csv` — traced to `synthetic_events.py`. 5,000 seeded,
  simulated rows. Has duration/event fields but no loan or customer
  identifier, no calendar timestamps, no balances, no recovery amounts.
- `intervention.csv` — traced to `synthetic_intervention.py`. 8,000
  seeded, simulated rows. Not a real randomized treatment experiment.
- `decisions_full.csv` — traced to `run_full_decisions.py`. Built from
  OpenML German Credit rows plus this project's own simulated ECL
  outputs. Not a real decision log.

None of these unblock steps 243/245/246. This confirms, rather than
assumes, that the project is on **Path B**.

### What each synthetic file is still useful for

A distinction worth keeping separate: these files can still exercise
code paths even though they cannot validate outcomes against reality.

- `loan_events.csv` lets `fit_hazards.py` be run and its output shape
  checked (code-path verified) but says nothing about real default/
  prepayment timing (not outcome-validated).
- `intervention.csv` lets the `TLearner` uplift model be trained and
  scored correctly (code-path verified, including the pickling fix
  applied earlier in this project) but has never been checked against
  a real randomized intervention (not outcome-validated).
- `decisions_full.csv` lets `governance.py`'s reporting logic run
  end-to-end (code-path verified) but its `n_decisions`/`approval_rate`
  figures do not describe real decisions (not outcome-validated).

Conflating "the code runs without error on this file" with "this result
is validated against reality" was the exact failure mode this project's
earlier synthetic-only phase was built on. Naming the two separately here
is meant to prevent repeating it.

### Final status

- **Code complete:** yes. Modules, adapters, tests, API, and governance
  reporting all exist and run.
- **Project complete, per this project's own definition:** no. A
  measured before/after reduction in credit loss on real borrowers has
  not been produced and cannot be, absent real LGD/EAD/performance/
  experiment data.
- **What this artifact actually is:** a real-data PD model (Home Credit,
  AUC 0.7340), calibrated and fairness-audited on real outcomes, with
  every other stage (LGD, EAD, survival modeling, uplift) explicitly
  documented as running on synthetic data and unvalidated against
  reality.
- **Next real step, if one is taken:** acquiring a genuine loan
  performance table with dated repay/default/prepay outcomes and
  recovery amounts — not writing more code against data that does not
  exist. Until that data exists, further capability work on LGD, EAD,
  hazards, or uplift is decoration, consistent with the project
  checklist's own instruction.
