#!/usr/bin/env python3
"""
FizzBuzz implementation
"""

def get_fizzbuzz_value(i: int) -> str:
    """
    Returns the FizzBuzz value for a single number.

    Args:
        i: The number to evaluate.

    Returns:
        "FizzBuzz", "Fizz", "Buzz", or the number as a string.
    """
    if i % 15 == 0:
        return "FizzBuzz"
    if i % 3 == 0:
        return "Fizz"
    if i % 5 == 0:
        return "Buzz"
    return str(i)


def fizzbuzz(n: int = 100) -> None:
    """
    Print FizzBuzz sequence from 1 to n.

    Args:
        n: Upper limit of the sequence (default: 100)
    """
    for i in range(1, n + 1):
        print(get_fizzbuzz_value(i))


if __name__ == "__main__":
    print("=== Standard FizzBuzz (1-100) ===")
    fizzbuzz(100)
