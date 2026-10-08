# Raw data

Raw Home Credit files are stored locally and are not committed to Git.

Download these files from the [Home Credit Default Risk data page](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place them directly in this directory:

- `application_train.csv`
- `HomeCredit_columns_description.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`

The rebuild intentionally omits `POS_CASH_balance.csv` and `credit_card_balance.csv`. Four relational sources are enough to demonstrate external credit history, prior applications, and repayment behaviour without creating an uncontrolled feature dump.

The notebooks expect the filenames above and validate their schemas before feature construction. Do not rename the files.
