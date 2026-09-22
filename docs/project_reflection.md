# Project Reflection

## Why I chose this project

I chose a credit default prediction project because my target roles are in credit risk analytics, risk data analysis, and financial data analysis. I wanted a portfolio project that connects my finance and econometrics background with Python-based data analysis and machine learning in a practical lending scenario.

Credit risk is also a good fit for the way I want to present my analytical work. The problem is not just to build a model; it is to understand repayment difficulty, handle class imbalance, evaluate risk ranking, and communicate results in a way that could support underwriting or application-risk reporting.

## What I focused on

For the first version, I focused on a complete and explainable project workflow rather than the highest possible model score.

The scope was intentionally controlled:

- use only `application_train.csv`
- clean and explain a focused set of applicant and loan features
- engineer interpretable financial risk ratios
- build a Logistic Regression baseline model
- evaluate ROC-AUC, precision, recall, F1-score, and threshold trade-offs
- translate model-generated payment-difficulty scores into Low, Medium, and High Risk segments
- connect the results back to credit-risk interpretation and manual-review prioritization

This makes the project easier to explain in an interview and closer to an analyst workflow than a pure Kaggle optimization exercise.

## What I learned

This project reinforced several practical lessons:

- Accuracy is not reliable by itself when the observed payment-difficulty rate is low. With a baseline observed payment-difficulty rate around 8.1%, a model can look accurate while missing many higher-risk applications.
- Classification thresholds matter. Lower thresholds capture more observed payment-difficulty cases but create more false positives, while higher thresholds improve selectivity but miss more higher-risk applications.
- Precision and recall need to be interpreted together. In credit risk, review capacity, applicant experience, and loss prevention all matter.
- Risk segmentation is often more useful for business communication than a single binary prediction. At the illustrative `0.15` threshold, the High Risk segment contained 4,209 holdout applications with an observed payment-difficulty rate of about 18.6%, roughly 2.3× the 8.1% holdout baseline.
- Distribution-based preprocessing boundaries should be estimated from the training set and then applied to the holdout set to prevent information leakage.
- Excluding `CODE_GENDER` from the formal model reflects awareness of sensitive-variable governance, fairness, and compliance considerations.
- Reference categories matter for interpretation. I explicitly used common, higher-volume groups such as `Secondary / secondary special`, `Married`, `House / apartment`, and `Laborers` rather than allowing rare alphabetically first categories to become the Logistic Regression baselines.
- Model metrics need business context. The holdout ROC-AUC of about 0.652 and the threshold comparison are useful for discussing risk ranking and manual-review trade-offs, but they do not establish a production credit policy.

## What I would improve next

In a next version, I would improve the project in several directions:

- add a Random Forest comparison as a non-linear benchmark
- review feature importance and compare it with Logistic Regression coefficients
- add SQL-based EDA to show database-style analytical workflow
- select thresholds using business cost assumptions for false positives and false negatives
- calibrate model scores before interpreting them as probabilities or using them in policy decisions
- include fairness and compliance review considerations
- use more Home Credit tables after the first-version workflow is stable

I would still keep interpretability and business communication as priorities, because credit risk analysis needs more than a model score.
