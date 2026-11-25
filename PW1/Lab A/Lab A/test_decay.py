"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


def test_rejects_negative_rate():
    with pytest.raises(ValueError):
        simulate(N0=1000, lam=-0.4)


def test_matches_law():
    N0, lam = 1000, 0.4
    avg=np.mean([simulate(N0,lam) for _ in range(200)], axis=0)
    t = np.arange(len(avg))
    expected = N0 * np.exp(-lam * t)
    assert avg == pytest.approx(expected, rel=0.05)
#   Check that the simulation's AVERAGE over many seeds is close to the
#   physical law  N0 * exp(-lam * t).
#   Which pytest tool compares floating-point values with a tolerance?