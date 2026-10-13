# Assignment 1 Report

- **Name**: Tomás Roschge
- **Student ID**: <!-- TODO -->
- **Email**: <!-- TODO -->
- **Group**: <!-- TODO: BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A -->

> ✍️ Sections marked **TODO (own words)** must be written by me, the author, per the AI policy.
> Everything else is a factual description of what the pipeline does and the numbers it produced.

## Dataset

**Predict Students' Dropout and Academic Success**, UCI Machine Learning Repository (dataset #697,
Realinho et al., 2022), committed as `data/students.csv`. One row = one undergraduate student of a
Portuguese higher-education institution: 4,424 rows × 37 columns (demographics, admission data,
socio-economic background, 1st- and 2nd-semester curricular units, three macro-economic indicators)
and a 3-class `Target`: Graduate 2,209 / Dropout 1,421 / Enrolled 794. No missing values, no duplicates.

Data finding: the three macro columns (unemployment, inflation, GDP growth) take only **10 distinct
combinations**, and they match Portugal's yearly figures for 2008–2017 (e.g. unemployment 16.2 % in 2013,
GDP −4.06 % in 2012). They therefore identify the **enrollment cohort**, which `pipeline.py` reconstructs
as a `cohort` column. Students per cohort range from 362 to 571.

**TODO (own words):** why I picked it.

## Business / real-life framing

Decision supported: **at the end of each student's 1st semester, the university decides who is offered a
mentoring programme** to prevent dropout. Costs (assumed): a missed dropout (FN) loses ≈ **€3,000** of
remaining tuition; mentoring a student who would have graduated anyway (FP) costs ≈ **€300**.

How this framing drives the pipeline:

| Framing implication | Pipeline decision |
|---|---|
| The decision is made at the end of semester 1 | All 2nd-semester columns are dropped (they don't exist yet at decision time, so using them would be leakage) |
| The model will be applied to *future* cohorts | Split by cohort, not randomly: train 2008–2013 (2,311 rows), validation 2014–2015 (681), test 2016–2017 (638) |
| "Enrolled" = still studying at the end of the normal course duration, final outcome unknown | Excluded; target = 1 if Dropout, 0 if Graduate |
| Errors have asymmetric costs (10:1) | Threshold chosen by minimising €-cost on the validation cohorts, then applied unchanged to test; C chosen by validation log loss |

**TODO (own words):** explain the scenario and why each of these decisions follows from it.

## Data preparation & feature engineering

All features are built in `pipeline.engineer_features`. The preprocessor is fitted on the train cohorts only
(`pipeline.build_preprocessor`).

- **1st-semester performance:** `s1_approval_rate` = approved / enrolled units, `s1_failed_units`,
  `s1_evals_per_unit`, `s1_no_evaluations`, `s1_units_without_eval`, `s1_has_credits` (credited units = transfer students).
- **Grade = 0 is not a grade:** in the raw data a 1st-semester grade of 0 means "nothing approved". `s1_no_grade`
  flags it, the grade itself is set to missing and imputed with the train median. Without this, a linear model
  would treat 0/20 as a real, extreme grade.
- **Financial:** `scholarship` only (tuition and debtor flags excluded, see *Dropped*).
- **Background:** `log_age_at_enrollment` (age is right-skewed: median 20, max 70), `mature_student` (≥ 23),
  `application_order`, `parents_higher_ed` (0/1/2, from 29 qualification codes mapped to higher-ed or not),
  `non_portuguese` (97.5 % are Portuguese, so 21 nationality codes collapse to one flag), `not_single`, `evening`,
  `male`, `displaced`, `special_needs`, plus admission and previous-qualification grades.
- **High-cardinality codes** (`course` 17 levels, `application_mode` 18, `prev_qualification` 17): one-hot,
  with levels that have < 30 train rows grouped into "infrequent" (also catches unseen levels at test time).
- **Dropped:** all 2nd-semester columns (leakage), the macro columns (cohort identifiers with only 10 values:
  they would learn cohort effects that can't extrapolate to new cohorts), parents' occupations (32 noisy codes each),
  `International` (duplicate of nationality), and **`Tuition fees up to date` + `Debtor`**: the data dictionary
  doesn't say when they were recorded, and 457 of the 486 students not up to date (94 %) are dropouts, so the flag
  most likely records the dropout (a student who left stops paying) instead of preceding it. With both features the
  test ROC AUC was 0.909 (tuition was the largest coefficient, +1.57); without them it is 0.895.
- **Scaling:** numeric features are standardised (needed for gradient descent to converge at one learning rate,
  and makes L2 penalise all features on the same scale). Binary flags are left as 0/1.

Final design matrix: 54 columns.

**TODO (own words):** what didn't work / what I'd tried, and why each choice matters.

## Modeling: three implementations, one model

Model: L2-regularised **logistic regression** (binary target). All three versions minimise the same objective
on the same standardised train matrix:

`mean BCE + λ/2 · ||w||²`, with `λ = 1 / (C · n_train)`, bias not penalised.

This matches sklearn's `0.5·||w||² + C·Σ logloss` divided by `C·n`.

- **scikit-learn:** `LogisticRegression(C=0.03)`, lbfgs. C picked from {0.01 … 10} by validation log loss
  (0.01 → 0.362, 0.03 → **0.353**, 0.1 → 0.353, 10 → 0.372).
- **Manual PyTorch:** raw tensors `w`, `b`, explicit sigmoid + BCE + L2, `loss.backward()`, and a manual
  `w -= lr·w.grad` step inside `torch.no_grad()`. Full batch, 4,000 epochs.
- **Standard PyTorch:** `nn.Module` with one `nn.Linear`, `BCEWithLogitsLoss`, and `torch.optim.SGD` with
  `weight_decay=λ` on the weights only. `weight_decay` adds `λ·w` to the gradient, which is the gradient of `λ/2·||w||²`.
- **Learning rate** (both torch versions): picked from {0.05, 0.1, 0.5, 1.0} by the final training objective,
  since it is an optimisation setting, not a model setting → 1.0 (all four reach 0.33885 within 1e-5 after
  4,000 epochs).

### Results on the test cohorts (2016–2017, n = 638, dropout rate 34.3 %)

Threshold chosen on the validation cohorts.

| Model | ROC AUC | PR AUC | Log loss | Threshold | Recall | Precision | Flagged | Cost (€) | €/student |
|---|---|---|---|---|---|---|---|---|---|
| Naive baseline (train rate 0.423) | 0.500 | 0.343 | 0.656 | 0.01 | 1.000 | 0.343 | 100 % | 125,700 | 197 |
| scikit-learn | 0.895 | 0.879 | 0.346 | 0.11 | 0.904 | 0.538 | 57.7 % | 114,000 | 179 |
| PyTorch manual loop | 0.895 | 0.879 | 0.346 | 0.13 | 0.890 | 0.580 | 52.7 % | 114,300 | 179 |
| PyTorch nn.Module + optim | 0.895 | 0.879 | 0.346 | 0.13 | 0.890 | 0.580 | 52.7 % | 114,300 | 179 |
| Policy: mentor nobody | – | – | – | – | 0.000 | – | 0 % | 657,000 | 1,030 |
| Policy: mentor everybody | – | – | – | – | 1.000 | 0.343 | 100 % | 125,700 | 197 |

Agreement with sklearn on the test set:
- **Manual loop and nn.Module:** max |Δ probability| = 0.0009 and max |Δ coefficient| = 0.0018 vs. sklearn. The two
  torch versions differ from each other by at most 1.2e-7 in probability (float32 precision).
- **Intercepts:** sklearn −0.185 vs. torch −0.188.
- **Same decisions as sklearn:** 95.0 % of test students when each model uses its own validation-chosen threshold
  (0.11 vs 0.13). **100 %** when they use the same threshold (checked at 0.11 and 0.13).
- **Why the thresholds differ:** on the validation cohorts the cost curve is jagged. At 0.11, sklearn costs €120,900 and
  torch €123,900. The difference is one FN: exactly one validation student (an actual dropout) gets 0.11013 from
  sklearn and 0.1098 from torch, so they land on opposite sides of 0.11. At 0.13 the costs are €122,100 and €121,800.
  That flips the argmin. Logged for all three models in `artifacts/metrics.json`
  (`agreement.val_cost_by_threshold_near_optimum`) and shown in the dashboard's agreement panel.

Largest coefficients (log-odds, same sign and size in all three):
- **Increase dropout risk:** `s1_failed_units` +1.06, course 9853 +0.57, `mature_student` +0.30, `log_age_at_enrollment` +0.24.
- **Decrease dropout risk:** `s1_approval_rate` −0.90, `scholarship` −0.50, course 9500 −0.38, `s1_grade` −0.36.

**TODO (own words):** do the three agree? Explain the 95 % vs 100 % decision agreement and the small intercept gap.
Also: the best model is only €11,700 (9 %) cheaper than mentoring everybody. What that means for the business case.

## Limitations & next steps

**TODO (own words).**

## Generative AI use disclosure

**TODO (own words; edit to be accurate).** Claude Code (Anthropic, Claude Opus) was used to:
- propose candidate datasets and cost figures; I chose the dataset, framing and costs;
- write the code in `pipeline.py`, `train.py` and `app.py`;
- write the factual/descriptive parts of this report (dataset facts, pipeline description, results tables).

The framing justification, interpretation of results and limitations are my own.
