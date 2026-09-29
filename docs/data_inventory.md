# Data inventory

- **Bureau pulls:** Partial yes. Source: Kaggle "Home Credit Default Risk"
	(`bureau.csv`, `bureau_balance.csv` available but only `bureau.csv` used).
	Format: CSV, one row per prior credit per applicant (not a raw bureau
	extract format — pre-aggregated by Kaggle from an unspecified bureau
	source). Date range: not disclosed by the dataset publisher; contains
	relative day offsets (`DAYS_CREDIT`, `DAYS_CREDIT_ENDDATE`) from each
	application date, not absolute calendar dates. Coverage: 1,716,428 prior
	credit records across ~305,811 of the ~307,511 applicants (some
	applicants have zero bureau history, which is itself real information,
	not missing data).
- **Transaction streams:** No. Not acquired. No per-customer cashflow or
	transaction-level data exists anywhere in this project. This is a real,
	unaddressed gap — the `transactions.py` adapter (step 242) was never
	written because there is nothing to feed it. Any product decision
	requiring cashflow-based underwriting is unsupported by current data.
- **Loan performance:** No, not in the form step 243 requires. What
	exists is `application_train.csv`'s `TARGET` column: a single static
	binary flag (defaulted vs. not, as of an unspecified observation
	window) per application. This is a label, not a performance history —
	there are no dated observations of repayment status over time, no
	distinction between "repaid," "defaulted," and "prepaid," and no
	`balance_at_default` or recovery amounts. The `performance.py` adapter
	(step 243) has not been written because the required fields do not
	exist in the data acquired so far. Direct consequence: LGD (step 245)
	and survival/hazard modeling (step 246) remain on synthetic/simulated
	logic (`estimate_lgd`'s Beta-distribution draw, `fit_hazards.py`
	untouched) and have not been, and currently cannot be, validated
	against real outcomes.
- **Application events:** Yes. `application_train.csv` — one row per
	application with applicant-level fields at time of application
	(income, employment, credit amount requested, family status, etc.).
	No decision timestamp is present; there's no field indicating when
	or whether an application was actually approved by whatever process
	originated this data, only the eventual default outcome for the loans
	that were originated.
- **Outcome labels:** Yes, but limited. `TARGET` (binary: 1 = client had
	payment difficulties, 0 = did not) exists for all rows in
	`application_train.csv`. No timestamps on the label itself. Survivorship
	bias is present and specifically identifiable: this file only contains
	applications where a loan was actually originated (i.e. previously
	approved) — applicants who were rejected and never received a loan do
	not appear here, so the population is conditioned on a prior,
	unobserved approval decision. Any PD model trained on this data
	reflects "risk given approval under whatever original underwriting
	policy produced this dataset," not "risk of the general applicant
	population," and this caveat has not been corrected for (no reject
	inference was applied here).

## Summary

Real data supports PD model training and validation only (steps 240,
241, 244, 247 — completed and documented in `model_evaluation.md`).
It does not support real LGD, real EAD, real survival/hazard modeling,
or transaction-based underwriting. Per step 250, this project is
effectively on **Path B**: a single honest artifact (real-data PD model,
calibration curve, fairness check) rather than a full real-data platform.
Steps 242, 243, 245, and 246 remain open and are blocked on data that has
not been acquired, not on unfinished code.
