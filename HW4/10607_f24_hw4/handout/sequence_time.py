"""
10607 Fall 2023 Homework 4 Recursive Sequence Supplement

Code for you to observe the speed of the techniques you have implemented for the recursive sequence.
Take a minute to observe and reason about the relative amount of time each technique uses.

Make sure you have Python and numpy installed on your system.

To run this script, ensure hw4.py is in the same directory and run
    python3 sequence_time.py

By Jocelyn Tseng
2023
"""

import hw4
import time

start = time.time()
recursive_result = hw4.sequence1(30)
one = time.time()
iterative_result = hw4.sequence2(30)
two = time.time()
memoized_result = hw4.sequence3(30)
three = time.time()

def seconds_to_ms(t):
    return round(float(t), 5)

print(f'Recursive Sequence Experiment:')
print(f'Recursive: {seconds_to_ms(one - start)} ms')
print(f'Iterative: {seconds_to_ms(two - one)} ms')
print(f'Memoized: {seconds_to_ms(three - two)} ms')