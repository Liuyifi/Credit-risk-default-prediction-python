# Project notes

## Analytical question

How much incremental risk information is added by historical credit and repayment behaviour beyond application-only characteristics, and how does an interpretable Logistic benchmark compare with a nonlinear challenger in discrimination, calibration, and operational risk concentration?

`TARGET=1` is Home Credit's payment-difficulty outcome. It is not treated as a regulatory default definition or a production probability of default.

## Main design decisions

The source build uses `application_train`, `bureau`, `bureau_balance`, `previous_application`, and `installments_payments`. POS cash and credit-card tables are omitted to keep the relational architecture reviewable. All historical records must be observable no later than the current application date.

The feature table contains 99 predictors and one row per borrower. It avoids automatic aggregation spam. Bureau balance is aggregated from month to credit to borrower; numeric delinquency shares use status-observed months (statuses 0–5) as their denominator. Installment payment rows are first consolidated on borrower, prior contract, version, and installment number: payment amount is summed and the last payment day is retained so late full settlement is represented correctly. Missing relationship history is represented with explicit flags instead of duplicated automatic indicators.

`CODE_GENDER` is excluded. Age and family status remain in the model with subgroup diagnostics. Raw `AMT_GOODS_PRICE` is excluded because it is nearly collinear with application credit; the credit-to-goods ratio is retained. Opaque `EXT_SOURCE` scores are omitted so the relational increment remains interpretable. Future-realization fields and sentinel dates do not enter the feature table.

## Validation design

The original 80/20 stratified split is saved locally by borrower ID as a legacy holdout. It is exactly the same population evaluated in the earlier project version, so rotating to a different seed would not create independent evidence. The split is retained only for descriptive comparability.

Primary evidence instead comes from five-fold outer stratified CV across the full labelled population. Each outer validation fold is excluded from preprocessing, Logistic regularization selection, LightGBM candidate selection, best-iteration selection, and calibration diagnostics until scoring. Historical aggregation occurs before splitting only because it is deterministic, target-free, and contains no distributional fitting.

The Logistic grid is limited to four `C` values and selected with three inner folds. LightGBM uses four manual candidates, restrained tree settings, and early stopping on an inner validation split with a 1,200-tree ceiling. Raw outer-OOF calibration is diagnostic only; no transform is fitted. Review capacities remain fixed at 5%, 10%, and 20%.

The pre-existing primary-model rule is retained: select LightGBM only when its outer-CV AUC exceeds Logistic by at least 0.005; otherwise prefer Logistic transparency. The separately reported legacy-holdout metrics cannot change this choice.

## Interpretation

Application-only Logistic reached 0.6579 mean outer-CV AUC. Adding historical families increased full-Logistic outer-CV AUC to 0.7232, showing that external credit and repayment history contains material incremental ranking information. LightGBM reached 0.7534 mean outer-CV AUC and remains primary under the pre-existing 0.005 improvement rule.

On the previously used legacy holdout, the rebuilt model results are reported only as secondary comparability evidence. Because outer CV uses all labelled rows, those borrowers necessarily participate in primary development folds; however, hyperparameters, iteration count, model family, calibration treatment, and capacity levels are not changed after consulting the separately reported 80/20 results.

LightGBM is therefore the stronger ranking candidate for this dataset. Logistic remains useful where transparent directionality and coefficient stability matter more than the incremental discrimination. Neither coefficient signs nor tree importance are causal explanations.

## Boundaries

Outer CV does not establish temporal or external validity, and the legacy split is explicitly historical. No approval policy, reject inference, decision cost, external validation, or monitoring population is available. The demographic and socioeconomic subgroup checks are diagnostic rather than a claim of fairness or compliance. The code and generated tables are the analytical source of truth; local Word/PDF material is outside this rebuild.
