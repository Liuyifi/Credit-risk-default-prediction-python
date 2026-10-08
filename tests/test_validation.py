import numpy as np
from scipy.stats import ks_2samp

from src.validation import assign_risk_deciles, capacity_table, ks_statistic, risk_decile_table


def test_ks_matches_two_sample_reference_with_ties():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.2, 0.4, 0.6, 0.7, 0.7, 0.9])
    reference = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert np.isclose(ks_statistic(y, score), reference)


def test_deciles_and_capacity_reconcile_without_outcome_ranking():
    rng = np.random.default_rng(7)
    score = rng.uniform(size=1_000)
    y = rng.binomial(1, score)
    deciles_before = assign_risk_deciles(score)
    deciles_after = assign_risk_deciles(score)
    assert np.array_equal(deciles_before, deciles_after)

    table = risk_decile_table(y, score, "test")
    assert table["borrowers"].sum() == len(y)
    assert table["bads"].sum() == y.sum()
    assert np.isclose(table["cumulative_bad_capture"].iloc[-1], 1.0)

    capacity = capacity_table(y, score, "test")
    assert capacity["reviewed_borrowers"].tolist() == [50, 100, 200]
    assert (capacity["captured_bads"] <= y.sum()).all()
    assert (capacity["non_events_reviewed"] + capacity["captured_bads"] == capacity["reviewed_borrowers"]).all()
