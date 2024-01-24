"""
Example of pytest functionality

Goes over some basic assert examples
"""
import pytest


def test_example_assert_equal():
    assert 0 == 0


def test_example_assert_no_equal():
    assert 0 != 1


def test_example_assert_almost_equal():
    assert 1.0 == pytest.approx(1.01, .1)


"""
To run this suite of tests, run 'pytest' in the main directory
"""