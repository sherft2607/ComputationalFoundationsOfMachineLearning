"""
10607 Fall 2023 Homework 4 Tests

Test file for hw4.py

Make sure you have Python and numpy installed on your system.

To run the tests, ensure hw4.py is in the same directory and run
    python3 hw4_tests.py

These tests are only a subset of the tests on Gradescope,
so passing all these local tests does not mean you got full credit.

By Jocelyn Tseng
2023
"""

from numpy.testing import assert_allclose
from numpy.testing import assert_equal

import hw4

# Question 1: Sequence
assert_equal(int(hw4.sequence1(10)), 77)
assert_equal(hw4.sequence1(10), hw4.sequence2(10))
assert_equal(hw4.sequence1(10), hw4.sequence3(10))

# Question 2.1
assert_allclose(hw4.recursive_solution(20), 0.9523809523809522, atol=1e-8)
assert_allclose(hw4.recursive_solution(20), hw4.static_solution(20), atol=1e-8)

# Question 2.2
string = 'computation for machine learning!'
assert_equal(hw4.string_length(string), len(string))

# Question 2.3
assert_allclose(hw4.geometric_sum(2, 3, 10), 59048)
assert_allclose(hw4.geometric_sum(2, 3, 10), int(hw4.geometric_sum_definition(2, 3, 10)))