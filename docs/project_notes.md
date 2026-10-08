# Project notes

## Analytical question

How much incremental risk information is added by historical credit and repayment behaviour beyond application-only characteristics, and how does an interpretable Logistic benchmark compare with a nonlinear challenger in discrimination, calibration, and operational risk concentration?

`TARGET=1` is Home Credit's payment-difficulty outcome. It is not treated as a regulatory default definition or a production probability of default.

## Main design decisions

The source build uses `application_train`, `bureau`, `bureau_balance`, `previous_application`, and `installments_payments`. POS cash and credit-card tables are omitted to keep the relational architecture reviewable. All historical records must be observable no later than the current application date.

The feature table contains 98 predictors and one row per borrower. It avoids automatic aggregation spam. Bureau balance is aggregated from month to credit to borrower. Installment payment rows are first consolidated to a scheduled-installment grain, because partial payments make the raw schedule key non-unique. Missing relationship history is represented with explicit flags instead of being treated as an ordinary missing value.

`CODE_GENDER` is excluded. Age and family status remain in the model with subgroup diagnostics. Raw `AMT_GOODS_PRICE` is excluded because it is nearly collinear with application credit; the credit-to-goods ratio is retained. Opaque `EXT_SOURCE` scores are omitted so the relational increment remains interpretable. Future-realization fields and sentinel dates do not enter the feature table.

## Validation lock

The 80/20 stratified development/final-test split is saved locally by borrower ID. Five-fold CV, hyperparameter selection, calibration choice, and the primary-model rule use development data only. All learned preprocessing is fitted inside each fold. Historical aggregation occurs before splitting only because it is deterministic, target-free, and contains no distributional fitting.

The Logistic grid is limited to four `C` values. LightGBM uses four manual candidates, restrained tree settings, and early stopping. Development OOF calibration checks did not justify sigmoid calibration. Review capacities were fixed at 5%, 10%, and 20% before final evaluation.

The primary-model rule selects LightGBM only when its development CV AUC exceeds Logistic by at least 0.005. The observed gap was 0.0286, so LightGBM was locked as primary before the final test was scored.

## Interpretation

Application-only Logistic reached 0.6576 mean CV AUC. Adding historical families increased full-Logistic CV AUC to 0.7225, showing that external credit and repayment history contains material incremental ranking information. LightGBM reached 0.7511 mean CV AUC.

On the untouched final test, LightGBM achieved 0.7553 AUC, 0.2377 PR-AUC, 0.3807 KS, and 0.06805 Brier. Logistic achieved 0.7248 AUC, 0.2025 PR-AUC, 0.3392 KS, and 0.06976 Brier. At 10% review capacity, LightGBM captured 33.76% of observed events versus 29.83% for Logistic.

LightGBM is therefore the stronger ranking candidate for this dataset. Logistic remains useful where transparent directionality and coefficient stability matter more than the incremental discrimination. Neither coefficient signs nor tree importance are causal explanations.

## Boundaries

The final split is random, so it is not evidence of temporal stability. No approval policy, reject inference, decision cost, external validation, or monitoring population is available. The demographic and socioeconomic subgroup checks are diagnostic rather than a claim of fairness or compliance. The code and generated tables are the analytical source of truth; local Word/PDF material is outside this rebuild.
