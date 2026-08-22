from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import bayes_threshold, cost, robust_threshold


def test_bayes_threshold_ratio_5():
    assert np.isclose(bayes_threshold(5), 1 / 6)


def test_cost_one_fp_one_fn():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.2, 0.9])
    assert cost(y, p, 0.5, 1, 5) == 6


def test_robust_threshold_valid():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.2, 0.4, 0.45, 0.8, 0.95])
    t, meta = robust_threshold(y, p, [2, 5, 10], 0.25)
    assert 0 < t < 1
    assert np.isfinite(meta["objective"])
