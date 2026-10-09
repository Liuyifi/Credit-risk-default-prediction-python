# Credit Risk Default Prediction

This project tests how much pre-application credit and repayment history improves prediction of Home Credit's payment-difficulty outcome beyond application characteristics alone. The outcome is the competition label, not a Basel, legal, lifetime-default, or production PD definition.

## Data scope

The analysis uses 307,511 applications (24,825 events; 8.1%) and four relational sources: bureau records, monthly bureau status, previous applications, and installment payments. Raw CSVs stay local. The feature build enforces a day/month-0 cutoff and produces one row per `SK_ID_CURR`.

Temporal checks excluded 17 bureau rows with a post-application credit update and 2,905 installment rows whose actual payment date was missing. Previous applications all occurred before day 0; bureau-balance months were all non-positive. Multiple payment rows for one scheduled installment are consolidated before borrower aggregation.

## Feature architecture

The final candidate set has 99 predictors:

| Family | Features |
|---|---:|
| Application and affordability | 23 |
| Bureau credit history | 19 |
| Bureau monthly status | 13 |
| Previous applications | 22 |
| Installment behaviour | 22 |

The set is intentionally selective rather than a generic aggregation grid. It includes explicit no-history flags. `CODE_GENDER`, opaque `EXT_SOURCE` scores, raw `AMT_GOODS_PRICE`, relationship IDs, post-application lifecycle fields, and sentinel dates are excluded. Age and family status remain candidate predictors and receive subgroup checks; several retained fields can act as socioeconomic proxies, so this public-data exercise makes no fairness or legal-compliance claim.

## Models and validation

Primary performance evidence comes from five-fold outer stratified cross-validation over all labelled applicants. Every outer fold keeps its validation rows outside preprocessing, tuning, and early stopping. The original stratified 80/20 split (`random_state=42`) is retained as a legacy holdout for continuity with the earlier project version; because that same population was evaluated previously, it is not treated as a pristine independent test.

- Logistic Regression: L2 penalty and a four-value `C` grid (`0.01`, `0.1`, `1`, `10`) selected by three inner folds. Numeric clipping, imputation, scaling, rare-category pooling, and encoding are refit within training partitions. Reference categories are explicit.
- LightGBM: four restrained candidates. Candidate and best-iteration selection use an inner validation split inside each outer-training fold; the outer validation fold is scored only after refitting with the locked iteration. The internal ceiling is 1,200 trees. No Optuna, SHAP, or additional challenger families are used.
- Calibration: outer out-of-fold raw scores are assessed with Brier score and calibration slope/intercept. No calibration transform is selected or applied.
- Operations: risk deciles and fixed top-5%, top-10%, and top-20% review capacities are assigned by model score only, with deterministic tie handling.

The application-only Logistic comparator is supporting evidence for the value of relational history, not a third production candidate.

## Results

Nested outer-CV means (standard deviation in parentheses):

| Model | ROC-AUC | PR-AUC | KS | Brier | Top-10% capture |
|---|---:|---:|---:|---:|---:|
| Application-only Logistic | 0.6579 (0.0038) | 0.1455 (0.0028) | 0.2354 (0.0087) | 0.07227 (0.00010) | 22.18% (0.43 pp) |
| Full Logistic | 0.7232 (0.0012) | 0.2004 (0.0037) | 0.3319 (0.0023) | 0.06985 (0.00013) | 29.21% (0.46 pp) |
| LightGBM | 0.7534 (0.0030) | 0.2346 (0.0030) | 0.3786 (0.0061) | 0.06824 (0.00014) | 32.70% (0.15 pp) |

Legacy-holdout results on 61,503 applicants (secondary comparability evidence):

| Model | ROC-AUC | Gini | PR-AUC | KS | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.7248 | 0.4495 | 0.2029 | 0.3392 | 0.06974 |
| LightGBM | 0.7552 | 0.5104 | 0.2371 | 0.3798 | 0.06809 |

Historical features add 0.0652 outer-CV AUC to the Logistic benchmark relative to application-only inputs. LightGBM adds a further 0.0303 and remains the selected primary ranking model under the pre-existing 0.005 rule. Logistic remains the transparency-first alternative: its standardized coefficients are reported with five-fold sign stability, while the challenger uses gain importance.

## Review capacity

For the selected LightGBM model:

| Review capacity | Applicants reviewed | Events captured | Capture | Observed event rate | Lift |
|---|---:|---:|---:|---:|---:|
| 5% | 3,076 | 1,006 | 20.26% | 32.70% | 4.05× |
| 10% | 6,151 | 1,683 | 33.90% | 27.36% | 3.39× |
| 20% | 12,301 | 2,602 | 52.41% | 21.15% | 2.62× |

These are prespecified workload scenarios, not optimized lending cutoffs. No costs or approval economics are assumed.

![ROC and precision-recall comparison](outputs/figures/roc_pr_model_comparison.png)

![Bad rate by risk decile](outputs/figures/bad_rate_by_decile.png)

## Limitations

- The legacy holdout was used by an earlier project iteration and is neither independent nor out-of-time evidence.
- The public competition outcome and source population do not establish production lending suitability.
- Age, family status, education, occupation, housing, and asset variables may encode sensitive or socioeconomic effects; subgroup diagnostics do not prove fairness.
- LightGBM importance and penalized Logistic coefficients are associative, not causal.
- Review-capacity results omit decision costs, policy constraints, reject inference, and portfolio drift.

## Repository structure

```text
data/raw/                   local source CSVs and data instructions
data/processed/             local, ignored feature and split artifacts
notebooks/01_data_audit.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_model_development.ipynb
notebooks/04_model_validation.ipynb
src/data.py                 schemas and temporal filters
src/features.py             five-family borrower aggregation
src/modeling.py             nested outer CV and legacy-holdout fitting
src/validation.py           risk metrics, deciles, capacity, figures
tests/                      data-contract and validation tests
outputs/tables/             generated validation tables
outputs/figures/            generated figures
```

## Reproduction

Use the repository root as the working directory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_data_audit.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/02_feature_engineering.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/03_model_development.ipynb
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/04_model_validation.ipynb
```

Place the six files listed in [`data/raw/README.md`](data/raw/README.md) before running. On macOS, LightGBM may also require `brew install libomp`. The processed borrower table, split IDs, fitted artifacts, and raw CSVs remain gitignored; all committed tables and figures are regenerated by the notebooks.

Source tables: [`dataset_summary.csv`](outputs/tables/dataset_summary.csv), [`feature_family_summary.csv`](outputs/tables/feature_family_summary.csv), [`outer_cv_model_comparison.csv`](outputs/tables/outer_cv_model_comparison.csv), [`legacy_holdout_model_metrics.csv`](outputs/tables/legacy_holdout_model_metrics.csv), [`risk_decile_summary.csv`](outputs/tables/risk_decile_summary.csv), and [`capacity_analysis.csv`](outputs/tables/capacity_analysis.csv).
