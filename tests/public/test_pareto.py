import numpy as np
import pytest

from common.pareto import pareto_mask_array

pytestmark = pytest.mark.d1


def test_min_max_directions():
    # columns: latency (min), accuracy (max)
    v = np.array([[1, 0.5], [2, 0.7], [3, 0.8], [4, 0.79], [2.5, 0.6]])
    assert pareto_mask_array(v, ["min", "max"]).tolist() == [True, True, True, False, False]


def test_both_minimised():
    v = np.array([[1, 3], [2, 2], [3, 1], [3, 3]])
    assert pareto_mask_array(v, ["min", "min"]).tolist() == [True, True, True, False]


def test_three_objectives():
    v = np.array([[1, 1, 5], [1, 1, 4], [2, 0, 5]])
    assert pareto_mask_array(v, ["min", "min", "min"]).tolist() == [False, True, True]
