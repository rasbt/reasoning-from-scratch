# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from reasoning_from_scratch.appendix_f import elo_ratings
import math


def test_elo_single_match():
    r = elo_ratings([("A", "B")], k_factor=32, initial_rating=1000)
    assert math.isclose(r["A"], 1016)
    assert math.isclose(r["B"], 984)


def test_elo_total_points_constant():
    votes = [("A", "B"), ("B", "C"), ("A", "C")]
    r = elo_ratings(votes)
    assert math.isclose(sum(r.values()), 3000)