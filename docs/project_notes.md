# Project Notes

## Why I used Logistic Regression

I wanted to start with a model that I could understand from beginning to end. Logistic Regression was a good fit because it gives a clear baseline for a binary outcome and makes it possible to inspect the direction of each coefficient. It also kept my attention on the data, feature definitions, preprocessing, and evaluation instead of on tuning a more complicated algorithm.

This project is not an attempt to maximize a Kaggle score. My first goal was to understand whether a small set of application-level variables contained useful information for ranking payment-difficulty risk. I also wanted to see how a continuous model score changes when it is turned into a flag or a risk segment. That made the threshold analysis as important as the ROC-AUC result.

The model uses median imputation and standardization for numeric variables, plus one-hot encoding for categorical variables. These steps are fitted on the training set. The clipping limits are also calculated from the training set and then applied unchanged to the holdout set. Keeping these steps separate from the holdout data was one of the most important parts of the analysis.

## What surprised me

The holdout ROC-AUC is about `0.652`, so the model is not strong. I expected a limited feature set to have limits, but the result made it clear how much information is missing when only the main application table is used. Application details alone cannot describe a person's full credit history or later repayment behavior.

At the same time, the score is not completely uninformative. Using `0.15` as an illustrative threshold creates a High Risk group of `4,209` applicants, or about `6.8%` of the holdout set. This group has an observed payment-difficulty rate of about `18.6%`, compared with the holdout baseline of about `8.1%`. That is roughly `2.3x` the baseline.

The trade-off is coverage. At the same threshold, recall is only about `15.8%`. Most holdout applications with observed payment difficulty are not included in the High Risk group. Precision is about `18.6%` and F1 is about `17.1%`. I read these results as evidence that the model has some ranking value, but only modest discrimination and limited coverage at this cutoff.

## Decisions I made

I used only `application_train.csv`. Home Credit provides several related tables, but bringing them in would have expanded the project before I had a clear baseline. The selected variables cover basic applicant details, loan amounts, housing, education, family status, and occupation.

I added four simple features: age in years, employment length in years, credit-to-income ratio, and annuity-to-income ratio. I kept both the feature calculations and their interpretation straightforward. The unusual positive value in `DAYS_EMPLOYED` is treated as missing before employment years are calculated.

I used a stratified `80/20` train/holdout split with a fixed random state. The model was evaluated on the holdout set, while preprocessing rules that depend on the data were learned from the training set only. `SK_ID_CURR` is used for checks but not as a predictor. `CODE_GENDER` was inspected during data checks but excluded from the model. I kept the model focused on a limited set of applicant and loan characteristics.

The Low, Medium, and High Risk segments are based on score boundaries of `0.05` and `0.15`. These boundaries make the threshold trade-off easy to inspect; they are not claimed to be optimal. In particular, `0.15` is an illustrative threshold, not an optimized lending cutoff. I did not keep adding models just to improve the headline metric because the purpose of this version was to establish and explain a simple baseline.

## What I would try next

I would extend the work in four focused steps:

1. Use cross-validation and a separate final test design to check how stable the result is across samples.
2. Examine score calibration, including a calibration curve and Brier score, before interpreting scores as probabilities.
3. Choose thresholds with explicit assumptions about the relative cost of missed payment difficulty and unnecessary review.
4. Compare the baseline with one simple tree-based challenger model to see whether non-linear relationships add useful ranking power.

These checks would help show whether the current result is stable and whether the extra complexity produces a meaningful improvement.
