# Processed data

This directory is intentionally gitignored except for this file.

Running the notebooks creates rebuildable local artifacts here:

- `borrower_features.pkl`: one row per `SK_ID_CURR`, including `TARGET` for local modelling.
- `development_ids.csv` and `final_test_ids.csv`: the reproducible stratified split.

Development-only model choices and out-of-fold scores are written to the ignored `models/` directory. These artifacts contain applicant-level information or rebuildable fitted state and must not be committed. Rebuild them by executing notebooks `01` through `03` in order.
