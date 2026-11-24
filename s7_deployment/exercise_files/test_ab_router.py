"""Tests for the sticky bucketing in ab_router.py and the statistics in ab_analysis.py."""

# ruff: noqa: S101
import sys
from pathlib import Path

# This folder contains a file called fastapi.py that would shadow the real fastapi package, so import the real one
# first and only then make the exercise files importable (at the end of the path).
HERE = str(Path(__file__).parent)
if HERE in sys.path:
    sys.path.remove(HERE)
import fastapi  # noqa: E402, F401

sys.path.append(HERE)

from ab_analysis import required_sample_size, sample_ratio_mismatch, two_proportion_ztest  # noqa: E402
from ab_router import assign_variant  # noqa: E402

USERS = [f"user-{i:04d}" for i in range(10000)]


def test_assignment_is_deterministic():
    """The same user always gets the same variant."""
    first = [assign_variant(u, "exp", {"a": 50, "b": 50}) for u in USERS[:200]]
    second = [assign_variant(u, "exp", {"a": 50, "b": 50}) for u in USERS[:200]]
    assert first == second


def test_split_matches_weights():
    """About 80% of 10000 users end up in variant a."""
    share_a = sum(assign_variant(u, "exp", {"a": 80, "b": 20}) == "a" for u in USERS) / len(USERS)
    assert abs(share_a - 0.8) < 0.02


def test_experiments_are_independent():
    """Being in B in one experiment says nothing about being in B in another one."""
    w = {"a": 50, "b": 50}
    in_b_1 = [assign_variant(u, "exp1", w) == "b" for u in USERS]
    in_b_2 = [assign_variant(u, "exp2", w) == "b" for u in USERS]
    both = sum(x and y for x, y in zip(in_b_1, in_b_2)) / len(USERS)
    assert abs(both - 0.25) < 0.02  # independent: 0.5 * 0.5


def test_full_weight_goes_to_one_variant():
    """With weights 100/0 everybody is in a."""
    assert all(assign_variant(u, "exp", {"a": 100, "b": 0}) == "a" for u in USERS[:1000])


def test_ztest_textbook_example():
    """92% vs 94% with 10000 samples each is clearly significant and the CI excludes zero."""
    z, p, (low, high) = two_proportion_ztest(9200, 10000, 9400, 10000)
    assert z > 0
    assert p < 0.001
    assert 0 < low < 0.02 < high


def test_required_sample_size():
    """Detecting +1pp at 92% accuracy needs roughly 10-11 thousand samples per group."""
    assert 10000 < required_sample_size(0.92, 0.01) < 12000


def test_sample_ratio_mismatch():
    """An even split is fine, a 5000/5300 split is suspicious."""
    assert sample_ratio_mismatch(5000, 5000) > 0.99
    assert sample_ratio_mismatch(5000, 5300) < 0.01
