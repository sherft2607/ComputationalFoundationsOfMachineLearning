"""
10607 Fall 2023 Homework 4

Instructions: Fill in the functions according to the instructions in the writeup.

Make sure you have Python and numpy installed on your system.

By Jocelyn Tseng
2023
"""

# Question 1: For the sequence functions, n is an integer >= 0
def sequence1(n):
    # YOUR CODE HERE
    if n <= 3:
        return n
    return sequence1(n - 1) + sequence1(n - 2) - (sequence1(n-4) / sequence1(n-3))


def sequence2(n):
    # YOUR CODE HERE
    if n <= 3:
        return n
    values = [0, 1, 2, 3]
    for _ in range(4, n + 1):
        nextVal = values[-1] + values[-2] - (values[-4] / values[-3])
        values.append(nextVal)
    return values[-1]

table = {}
def sequence3(n):
    # YOUR CODE HERE
    if n in table:
        return table[n]
    if n <= 3:
        result = n
    else:
        result = sequence3(n-1) + sequence3(n-2) - (sequence3(n-4) / sequence3(n-3))
    table[n] = result
    return result

# Question 2.1: n is an integer >= 1
def recursive_solution(n):
    # YOUR CODE HERE
    if n == 1:
        return 1 / (1 * 2)
    return 1 / (n * (n + 1)) + recursive_solution(n - 1)

def static_solution(n):
    # YOUR CODE HERE
    return n / (n + 1)

# Question 2.2
def string_length(string):
    # YOUR CODE HERE
    if string == "":
        return 0
    return 1 + string_length(string[1:])

# Question 2.3
def geometric_sum(a,r,n):
    # YOUR CODE HERE
    if n == 1:
        return a
    return a * r ** (n - 1) + geometric_sum(a, r, n - 1)

def geometric_sum_definition(a,r,n):
    # YOUR CODE HERE
    sum = 0
    for i in range(n):
        sum += a * r ** i
    return sum