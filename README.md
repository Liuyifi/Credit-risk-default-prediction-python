# Credit Risk Default Prediction

This project uses Home Credit's `application_train.csv` to build an interpretable Logistic Regression baseline for payment-difficulty risk ranking. It covers data checks, a small set of engineered features, an `80/20` train/holdout split, and holdout evaluation.

The holdout ROC-AUC is about `0.652`, so the model is only a modest baseline. At an illustrative threshold of `0.15`, it identifies a small High Risk group with an observed payment-difficulty rate of about `18.6%`, compared with the `8.1%` holdout baseline.

[View the analysis notebook](notebooks/01_credit_risk_default_prediction.ipynb) · [Read the project notes](docs/project_notes.md)

## Key results

| Holdout result | Value |
| --- | ---: |
| Observed payment-difficulty rate | 8.1% |
| ROC-AUC | 0.652 |
| Implied Gini (`2 × ROC-AUC - 1`) | 0.304 |
| Illustrative threshold | 0.15 |
| High Risk applicants | 4,209 of 61,503 (6.8%) |
| High Risk observed payment-difficulty rate / precision | 18.6% |
| Recall | 15.8% |
| F1-score | 17.1% |
| High Risk lift versus holdout baseline | 2.3× |

The model is only a modest baseline. The `0.15` cutoff is an illustrative threshold, not an optimized lending cutoff.

## Data and features

The data comes from the [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk/data). The source application table has `307,511` rows and `122` columns. This project uses selected application-level fields rather than the other Home Credit tables.

Home Credit defines `TARGET = 1` as an applicant with payment difficulties: late payment of more than X days on at least one of the first Y instalments of the sample loan. `TARGET = 0` means that outcome was not observed under the dataset definition. The project title uses “default” as shorthand; `TARGET` is not a general legal or lender-specific definition of default.

The selected fields cover application ID, target, income, credit and annuity amounts, contract type, asset ownership, age, employment, education, family status, housing, and occupation. Four features are engineered:

- `AGE_YEARS = -DAYS_BIRTH / 365.25`
- `YEARS_EMPLOYED = -DAYS_EMPLOYED / 365.25`
- `CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL`
- `ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL`

Positive `DAYS_EMPLOYED` placeholder values are treated as missing before employment years are calculated. `SK_ID_CURR` is used for data checks but not as a predictor. `CODE_GENDER` was inspected during data checks but excluded from the model. I kept the model focused on a limited set of applicant and loan characteristics.

Raw CSV files are not committed to Git. Their expected location is described in [data/raw/README.md](data/raw/README.md).

## Analysis workflow

1. Load the selected fields and check application IDs, duplicate rows, missing targets, and target values.
2. Handle the `DAYS_EMPLOYED` placeholder and create the four derived features.
3. Make a stratified `80/20` train/holdout split with `random_state=42`.
4. Calculate `0.1%` and `99.9%` clipping bounds from the training set and apply the same bounds to both splits.
5. Fit median imputation and standardization for numeric features, plus `Unknown` imputation and one-hot encoding for categorical features, on the training data.
6. Fit Logistic Regression with L2 regularization, `C=1.0`, `solver="liblinear"`, and no class weighting.
7. Evaluate the holdout scores, compare thresholds, and create Low, Medium, and High Risk segments.

Imputation, scaling, encoding, and clipping do not use the holdout distribution.

## Model results and threshold trade-off

ROC-AUC measures ranking across all thresholds. Precision, recall, and F1 depend on the selected cutoff. At `0.15`, the model flags about `6.8%` of the holdout set; precision is about `18.6%`, recall about `15.8%`, and F1 about `17.1%`. The flag is targeted, but its coverage is low because it misses most observed payment-difficulty cases.

![Threshold trade-off curve](outputs/figures/threshold_tradeoff_curve.png)

![ROC curve](outputs/figures/roc_curve.png)

Supporting files: [model metrics](outputs/model_metrics.csv), [threshold comparison](outputs/threshold_comparison.csv), and [Logistic Regression coefficients](outputs/logistic_regression_coefficients.csv).

## Risk segmentation

Holdout scores are divided into three illustrative groups:

| Segment | Rule | Applicants | Observed payment-difficulty rate | Lift vs holdout baseline |
| --- | --- | ---: | ---: | ---: |
| Low Risk | score < 0.05 | 14,679 | 3.9% | 0.5× |
| Medium Risk | 0.05 ≤ score < 0.15 | 42,615 | 8.5% | 1.0× |
| High Risk | score ≥ 0.15 | 4,209 | 18.6% | 2.3× |

![Observed payment-difficulty rate by risk segment](outputs/figures/payment_difficulty_rate_by_risk_segment.png)

The High Risk segment has a clearly higher observed rate than the holdout baseline, but it contains only a small share of applicants and captures `15.8%` of the observed payment-difficulty cases. The segments describe score ranges; they are not credit decision rules.

[View the risk segment summary](outputs/risk_segment_summary.csv).

## Limitations

- The dataset is public competition data rather than a current lender sample.
- The model uses selected fields from the application table only; it does not include bureau, repayment, or cash-flow history.
- Ranking performance is modest (`ROC-AUC = 0.652`).
- Evaluation uses one random holdout split, with no independent or out-of-time test.
- Scores are not calibrated, and the `0.15` threshold has not been optimized with lending costs or review capacity.

## What I would try next

- Add cross-validation and reserve a separate final test set.
- Check calibration with a calibration curve and Brier score.
- Compare thresholds using explicit false-positive and false-negative costs.
- Test one simple tree-based challenger against the Logistic Regression baseline.

## Project structure

```text
credit-risk-default-prediction-python/
├── data/
│   └── raw/
│       └── README.md
├── docs/
│   └── project_notes.md
├── notebooks/
│   └── 01_credit_risk_default_prediction.ipynb
├── outputs/
│   ├── figures/
│   ├── logistic_regression_coefficients.csv
│   ├── model_metrics.csv
│   ├── risk_segment_summary.csv
│   └── threshold_comparison.csv
├── .gitignore
├── README.md
└── requirements.txt
```

## How to run

1. Download the Home Credit competition data.
2. Place `application_train.csv` in `data/raw/`. The optional field description file can go in the same folder.
3. Install the dependencies and open the notebook from the project root:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_credit_risk_default_prediction.ipynb
```

To execute it non-interactively:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_credit_risk_default_prediction.ipynb
```

Running the notebook writes the documented CSV and figure outputs under `outputs/`.
