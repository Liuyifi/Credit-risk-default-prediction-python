# Credit Risk Default Prediction

This project tests how much pre-application credit and repayment history improves prediction of Home Credit's payment-difficulty outcome beyond application characteristics alone. The outcome is the competition label, not a Basel, legal, lifetime-default, or production PD definition.

## Data scope

The analysis uses 307,511 applications (24,825 events; 8.1%) and four relational sources: bureau records, monthly bureau status, previous applications, and installment payments. Raw CSVs stay local. The feature build enforces a day/month-0 cutoff and produces one row per `SK_ID_CURR`.

Temporal checks excluded 17 bureau rows with a post-application credit update and 2,905 installment rows whose actual payment date was missing. Previous applications all occurred before day 0; bureau-balance months were all non-positive. Multiple payment rows for one scheduled installment are consolidated before borrower aggregation.

## Feature architecture

The final candidate set has 98 predictors:

| Family | Features |
|---|---:|
| Application and affordability | 23 |
| Bureau credit history | 19 |
| Bureau monthly status | 12 |
| Previous applications | 22 |
| Installment behaviour | 22 |

The set is intentionally selective rather than a generic aggregation grid. It includes explicit no-history flags. `CODE_GENDER`, opaque `EXT_SOURCE` scores, raw `AMT_GOODS_PRICE`, relationship IDs, post-application lifecycle fields, and sentinel dates are excluded. Age and family status remain candidate predictors and receive subgroup checks; several retained fields can act as socioeconomic proxies, so this public-data exercise makes no fairness or legal-compliance claim.

## Models and validation

Applicants are split once into 80% development and 20% final test with stratification and `random_state=42`. Model design uses five-fold stratified CV inside development only.

- Logistic Regression: L2 penalty and a four-value `C` grid (`0.01`, `0.1`, `1`, `10`). Numeric clipping, imputation, scaling, rare-category pooling, and encoding are refit within each fold.
- LightGBM: four restrained candidates and early stopping. No Optuna, SHAP, or additional challenger families are used.
- Calibration: a cross-fitted sigmoid check uses development OOF scores. It was not retained because it did not improve Brier score by the prespecified `0.0001` minimum for either model.
- Operations: risk deciles and fixed top-5%, top-10%, and top-20% review capacities are assigned by model score only, with deterministic tie handling.

The application-only Logistic comparator is supporting evidence for the value of relational history, not a third production candidate.

## Results

Development CV means (standard deviation in parentheses):

| Model | ROC-AUC | PR-AUC | KS | Brier | Top-10% capture |
|---|---:|---:|---:|---:|---:|
| Application-only Logistic | 0.6576 (0.0051) | 0.1449 (0.0027) | 0.2349 (0.0074) | 0.07229 (0.00012) | 22.18% (0.30 pp) |
| Full Logistic | 0.7225 (0.0044) | 0.2002 (0.0030) | 0.3287 (0.0073) | 0.06988 (0.00016) | 29.03% (0.53 pp) |
| LightGBM | 0.7511 (0.0031) | 0.2323 (0.0030) | 0.3737 (0.0067) | 0.06836 (0.00016) | 32.10% (0.34 pp) |

Locked final-test results on 61,503 applicants:

| Model | ROC-AUC | Gini | PR-AUC | KS | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.7248 | 0.4496 | 0.2025 | 0.3392 | 0.06976 |
| LightGBM | 0.7553 | 0.5106 | 0.2377 | 0.3807 | 0.06805 |

Historical features add 0.0648 CV AUC to the Logistic benchmark relative to application-only inputs. LightGBM adds a further 0.0286 CV AUC and is the selected primary ranking model. Logistic remains the transparency-first alternative: its standardized coefficients are reported with five-fold sign stability, while the challenger uses gain importance.

## Review capacity

For the selected LightGBM model:

| Review capacity | Applicants reviewed | Events captured | Capture | Observed event rate | Lift |
|---|---:|---:|---:|---:|---:|
| 5% | 3,076 | 1,017 | 20.48% | 33.06% | 4.10× |
| 10% | 6,151 | 1,676 | 33.76% | 27.25% | 3.38× |
| 20% | 12,301 | 2,593 | 52.23% | 21.08% | 2.61× |

These are prespecified workload scenarios, not optimized lending cutoffs. No costs or approval economics are assumed.

![ROC and precision-recall comparison](outputs/figures/roc_pr_model_comparison.png)

![Bad rate by risk decile](outputs/figures/bad_rate_by_decile.png)

## Limitations

- The final test is a random holdout, not an out-of-time or external validation sample.
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
src/modeling.py             fold-safe CV and locked fitting
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

Source tables: [`dataset_summary.csv`](outputs/tables/dataset_summary.csv), [`feature_family_summary.csv`](outputs/tables/feature_family_summary.csv), [`cv_model_comparison.csv`](outputs/tables/cv_model_comparison.csv), [`final_test_model_metrics.csv`](outputs/tables/final_test_model_metrics.csv), [`risk_decile_summary.csv`](outputs/tables/risk_decile_summary.csv), and [`capacity_analysis.csv`](outputs/tables/capacity_analysis.csv).
