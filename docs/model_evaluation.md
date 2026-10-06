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

## Reject-inference sensitivity study (Home Credit)

Baseline AUC on general approved test split: 0.7340
Boundary threshold: 0.15 (illustrative decline boundary for this study only)
Approved share: 87.10%
Declined share: 12.90%
TARGET among would-decline: 22.55%
TARGET among would-approve: 5.93%

Baseline AUC on approved-only holdout:     0.6785
Reject-aware AUC on approved-only holdout: 0.6787
Delta:                                     +0.0001

Interpretation:
- The 0.7340 figure measures discrimination across the full applicant
  population as observed in this dataset. It remains the documented
  baseline for that population.
- The 0.6785 figure measures discrimination within the subset of
  applicants the baseline model itself would approve at an illustrative
  threshold (PD > 0.15). This threshold was chosen for this sensitivity
  study only and has not been validated against any real underwriting
  policy. A different threshold would produce a different number.
  Neither figure is more correct than the other; they answer different
  questions about different populations.
- Parcelling reject inference changed the approved-holdout AUC by
  +0.0001. This is a demonstration of the mechanism, not a correction
  of survivorship bias.
- Home Credit contains only originated loans. No technique can recover
  the true declined-applicant population from this dataset. Real
  correction requires declined records with their decision-time features
  and eventual outcomes.

This study is a bounded sensitivity result, not evidence that the model
is validated for the population a real policy would decline. Both AUC
figures are documented with the population each was measured on.

## Point-in-time leakage check (Home Credit)

Check run: `src/leakage_check.py` against `load_bureau('data/application_train.csv')`.

Result:
- 18 feature columns checked; label `target` present.
- No feature name matches a forbidden post-outcome pattern.
- No bureau-derived column outside the three allowed aggregates
  (`bureau_credit_count`, `bureau_days_overdue_max`, `bureau_credit_sum`)
  survived the merge.

Review-required (timing unverified, not failures):
- `bureau_days_overdue_max`: safe only if the bureau pull occurred at or
  before the decision point. Home Credit does not disclose pull timing.
- `bureau_credit_sum`: safe only if aggregated as of the application date.
  Relative day offsets (`DAYS_CREDIT`, `DAYS_CREDIT_ENDDATE`) suggest
  as-of-or-before application, but this is inferred from dataset
  documentation, not verified against an upstream data contract.
- `bureau_credit_count`: same caveat as `bureau_credit_sum`.

What this check proves and does not prove:
- Proves: no common post-outcome name pattern and no contract violation
  in the merge produced by `src/adapters/bureau.py`.
- Does NOT prove: absence of leakage. A feature with a benign name can
  still be computed post-decision if the upstream aggregation window is
  wrong. The name check catches common patterns; it is not a substitute
  for a documented as-of timestamp on every upstream source.

Action: the three bureau aggregates are trained on as-is, with the timing
caveat recorded. If any upstream contract becomes available, revisit.

## Decision optimizer verification (item 8)

Traced the full served decision path: predictor.py -> select_offer.py
(pick_optimal) -> optimizer.py (simulate_offers) -> constraints.py
(filter_offers). No bypass: the chain is real and is what api.py's
/score endpoint actually calls.

Findings:

1. Candidate offers: three fixed offers (limit/rate/term), identical
   for every applicant. Not applicant-derived — a fixed menu, not a
   decision space.
2. ECL is constant across all three candidates (`ecl = pd_ * lgd * ead`
   inside the offer loop never references the candidate). The "risk"
   side of the optimizer does not vary by offer; only revenue
   (limit * rate * term) does. The optimizer selects highest revenue
   from a fixed menu, not the risk-adjusted-optimal offer. This is the
   root cause of the $20,000 limit returned earlier today for an
   $8,000 request: candidate 3 has the highest revenue product of the
   three fixed offers for every applicant, regardless of risk.
3. constraints.py filters on ECL and profit only. Fairness and
   calibration constraints, both required by the build plan's central
   optimization problem, are absent. Because ECL is constant per
   applicant, the ECL filter is all-or-nothing across the three
   offers, not a differentiating per-offer filter.
4. Selection mechanics (max-profit after filtering, explicit None on
   empty) are correctly implemented, but operate on the flawed
   candidate set above.
5. predictor.py findings: LGD is a fresh random Beta(1) draw on every
   call, unrelated to the applicant and non-deterministic between
   identical requests. EAD is set to the requested credit amount with
   no CCF applied, defensible for term loans, incorrect for revolving
   products per the build plan's own EAD requirement.

Status: the decision-optimizer code path is real (no bypass) but does
not perform risk-adjusted offer optimization. It currently returns the
same fixed-menu highest-revenue offer regardless of applicant risk.
Both real fixes (per-candidate ECL, requiring per-offer EAD/exposure
structure; fairness/calibration constraints, requiring population-
level metrics wired into per-applicant filtering) are non-trivial and
were deliberately not attempted this session, consistent with this
project's decision not to layer synthetic fixes on top of data-blocked
components. Documented as a known defect, not fixed.

## Optimizer fix — target behavior (Fix A, pre-change spec)

Before any code change, the following is the stated target for
per-candidate ECL:

1. EAD per candidate: EAD for a candidate offer of limit L is assumed
   equal to L (full-draw-at-origination). This is a defensible
   simplification for term loans, which this project currently handles
   exclusively. It is explicitly NOT valid for revolving credit, where
   EAD is not the current balance (per the build plan, section 4). If
   revolving products are ever added, this assumption must be revisited.

2. LGD per call: a single LGD value is drawn once per applicant per
   scoring call and reused across all three candidates in that call, so
   identical requests return identical numbers. This is a determinism
   fix, not a realism fix — LGD remains a synthetic Beta-distribution
   draw, unvalidated against real recovery data, exactly as documented
   elsewhere in this file. Determinism and realism are separate
   properties; this change addresses only the former.

3. Risk-adjusted selection means: select the candidate maximizing
   expected profit, where profit = revenue - ECL, and ECL = PD * LGD *
   EAD(candidate), varying per candidate via EAD(candidate) = candidate
   limit.

4. Per-candidate failure: a candidate with profit < min_profit or
   ECL > max_ecl is filtered out individually by the existing
   filter_offers logic — no change needed there. Call-level failure:
   None is returned only when all three candidates are filtered out,
   not on any single candidate's failure.

5. Rollback condition: if this change produces implausible or worse
   behavior when checked against the same sample applicants used in
   today's audit, revert optimizer.py to the fixed-menu version,
   document why, and stop — do not patch further in the same session.

6. Success check, to run once the change lands, before this section is
   marked done: score two applicants with the same requested amount but
   materially different PD (e.g. a high-PD and low-PD sample from
   configs/sample_applicant.json or similar). Confirm they no longer
   both receive the $20,000/36-month candidate by default, and confirm
   the selected offer differs in a way that tracks their PD difference.
   If both still receive the same offer, the fix did not address the
   root cause and needs re-diagnosis before being marked fixed.

## Optimizer Fix A — applied and verified

Change: `src/optimizer.py` — `simulate_offers` now computes
EAD(candidate) = candidate limit, inside the offer loop, so ECL varies
per candidate. Previously ECL was computed once per applicant and was
constant across the three fixed offers, making the selector a pure
revenue maximizer over a fixed menu.

Contract change: `simulate_offers(pd_, lgd, ead, offers=OFFERS)`
becomes `simulate_offers(pd_, lgd, offers=OFFERS)`. Corresponding
mechanical drop of the `ead` argument in `src/select_offer.py`'s
`pick_optimal`. `src/predictor.py` call updated to
`pick_optimal(pd_hat, lgd)` and its sample config switched to
`configs/sample_applicant_homecredit.json`.

Item-6 success check (two applicants, same requested amount,
EXT_SOURCE_1/2/3 varied to force materially different PD):

- low_pd: pd=0.0045, limit=20000, term=36, profit=8977.91
- high_pd: pd=0.5611, limit=10000, term=24, profit=1031.13
- Condition 1 (offers differ): True
- Condition 2 (direction, low_pd limit >= high_pd limit): True

Fix A is verified against the pre-change spec's success condition.

Still unresolved after Fix A, on record:

- LGD in the served path: superseded by "LGD placeholder in served
  path — fixed" below. The served path now uses a stated constant,
  LGD_PLACEHOLDER = 2/7. LGD remains synthetic and is not learned
  from real recovery data.
- `src/run_decisions.py` and `src/run_full_decisions.py` signature
  drift: superseded by commit `fea6a8a`, which corrected both call
  sites. `run_full_decisions.py` remains stale for a different reason
  (German Credit inputs against a Home Credit model) and is documented
  in "CLI decide command removed" below. `run_decisions.py` runs but
  produces meaningless output for the same input mismatch.
- `high_pd` at PD 0.5611 still receives a 10000/24 offer because
  `constraints.py` caps ECL, not PD. Whether a PD cap belongs in the
  constrained objective is a policy question, not a code defect.
  Left as-is.

## LGD placeholder in served path — fixed

Change: `src/predictor.py` no longer calls `estimate_lgd(1)` inside
`score()`. It uses `LGD_PLACEHOLDER = 2 / 7`, a stated constant equal
to the mean of the synthetic Beta(2, 5) generator used elsewhere in
the project.

Why: `estimate_lgd(1, seed=42)` was already deterministic per call —
verified, not assumed. The defect was not nondeterminism; it was that
the served path used one arbitrary draw from that distribution
(0.24395464376443093) and applied it identically to every applicant,
presenting a stochastic-looking value that was neither the
distribution mean nor applicant-derived nor learned.

What this fixes: the served path now uses a value that is honest about
what it is — a stated placeholder, not a per-applicant estimate.

What this does not fix: LGD remains synthetic. It is not learned from
real recovery or collections data, which this dataset does not contain.
Replace `LGD_PLACEHOLDER` with a real LGD model when recovery data
becomes available.

Verification:
- lgd on two identical `score()` calls: 0.2857142857142857 (both)
- 2/7 = 0.2857142857142857
- offers identical across the two calls: True

`estimate_lgd` remains in `src/lgd.py` for the non-served pipelines
(event simulation, stress scenarios) and now carries a docstring
stating it is not to be used in the served path.

## CLI decide command removed — stale pipeline

Change: `src/cli.py`'s `COMMANDS` dict no longer includes the `decide`
entry, which invoked `python -m src.run_full_decisions`.

Reason: `run_full_decisions.py` has two independent defects and has
never run correctly since the Home Credit swap:

1. It loads German Credit (`fetch_openml('credit-g')`) at line 9.
2. It loads the Home Credit `models/pd_baseline.joblib` at line 11.
3. At line 18 it passes German Credit rows to the Home Credit model's
   `top_drivers`, which raises ValueError because the Home Credit
   `ColumnTransformer` expects Home Credit columns.

An earlier commit (`fea6a8a`) fixed the `pick_optimal` signature in this
file and in `run_decisions.py`, but the input/model mismatch was never
addressed and is not a one-line fix.

Not fixed, documented as stale:
- `src/run_full_decisions.py` remains in the tree but is no longer
  reachable via the CLI.
- `src/run_decisions.py` runs but reads German Credit inputs
  (`data/ecl_output.csv`) and produces meaningless Home Credit decisions.
  It is not currently reachable via any CLI or API entry point.

Restoring `decide` requires replacing the German Credit input path with
Home Credit inputs (`load_bureau`) and rebuilding the record loop. That
is a spec-and-change cycle, not a mechanical fix, and is deferred.

Correction (same session): the CLI change described above was not
included in commit ed9e7cd. That commit contained only this doc section.
The `decide` entry was removed from `src/cli.py` in the following
commit, after this note was appended. The change is real; the commit
record is split across two commits rather than one.
