# /// script
# requires-python = ">=3.12"
# ///

"""
A quick demo of arithmetic coding.

PS: here are some useful references to understand floating-point arithmetic
https://stackoverflow.com/questions/21895756/why-are-floating-point-numbers-inaccurate
https://stackoverflow.com/questions/588004/is-floating-point-math-broken
"""

from collections import OrderedDict
from decimal import Decimal
from itertools import accumulate
from typing import List, Tuple

def box_print(text: str, padding: int = 4):
    """Prints text inside a box for better visibility in console output."""
    length = len(text) + 2 * (padding+1)
    print("\n" + "=" * length)
    print(f"|{' ' * padding}{text}{' ' * padding}|")
    print("=" * length + "\n")

def compute_cdf(probs: List[float]) -> List[float]:
    """Calculates the cumulative distribution function (CDF)."""
    return list(accumulate(probs))

def compute_ranges(dist: OrderedDict) -> dict:
    """Calculates decimal ranges for a given CDF."""
    probs = [Decimal(str(prob)) for prob in dist.values()]
    cdf = [Decimal("0.0")] + compute_cdf(probs)
    return {ch: (cdf[i], cdf[i+1]) for i, ch in enumerate(dist.keys())}

def arithmetic_encode(dist: OrderedDict, text: str) -> Tuple[Decimal]:
    """Turns a string into a decimal using arithmetic encoding."""
    ranges = compute_ranges(dist)
    scaled_ranges = ranges
    low, high = Decimal("0.0"), Decimal("1.0")
    print(f"{"CHAR":5s} {"LOW":10s} {"HIGH":10s}")
    for ch in text:
        low, high = scaled_ranges[ch]
        delta = high - low
        scaled_ranges = {ch: (rng[0]*delta + low, rng[1]*delta + low) for ch, rng in ranges.items()}
        print(f"{ch.upper():5s} {str(low):10s} {str(high):10s}")
    return low, high

def arithmetic_decode(dist: OrderedDict, encoded_value: Decimal, length: int) -> str:
    """Decodes a decimal value back to the original string using arithmetic decoding."""
    ranges = compute_ranges(dist)
    decoded = []
    value = encoded_value

    for _ in range(length):
        # Find which character range the current value falls into
        for ch, (low, high) in ranges.items():
            if low <= value < high:
                decoded.append(ch)
                # Scale the value for the next iteration
                delta = high - low
                value = (value - low) / delta
                break

    return ''.join(decoded)

def decimal_to_binary(value: Decimal, precision: int = 32) -> str:
    """Converts a decimal value to its binary representation with specified precision."""
    if value < 0 or value >= 1:
        raise ValueError("Value must be between 0 and 1")

    binary = "0."
    current = value

    for _ in range(precision):
        current *= 2
        if current >= 1:
            binary += "1"
            current -= 1
        else:
            binary += "0"

    return binary

def binary_to_decimal(binary_str: str) -> Decimal:
    """Converts a binary fractional string back to decimal."""
    if not binary_str.startswith("0."):
        raise ValueError("Binary string must start with '0.'")

    fractional_part = binary_str[2:]  # Remove "0."
    result = Decimal("0")

    for i, bit in enumerate(fractional_part):
        if bit == "1":
            result += Decimal("2") ** -(i + 1)

    return result

def main():
    """Main entrypoint"""

    # Example:
    # https://go-compression.github.io/algorithms/arithmetic
    dist = OrderedDict({'h': 0.2, 'e': 0.2, 'l': 0.4, 'o': 0.2})
    original_text = "hello"

    # Original text
    box_print("Original")
    print(f"Original text: {original_text}")
    print(f"Distribution: {dict(dist)}")

    # Encoded text
    box_print("Encoded")
    low, high = arithmetic_encode(dist, original_text)

    print(f"\nFinal range: [{low}, {high})")
    print(f"Range width: {high - low}")

    # Choose a value within the range for encoding
    encoded_value = (low + high) / 2  # Use midpoint
    print(f"Encoded value (midpoint): {encoded_value}")

    # Convert to binary representation
    binary_repr = decimal_to_binary(encoded_value, precision=20)
    print(f"Binary representation: {binary_repr}")

    # Decode back to original text
    box_print("Decoded")
    decoded_text = arithmetic_decode(dist, encoded_value, len(original_text))
    print(f"Decoded text: {decoded_text}")

    # Verify the round-trip
    print(f"\nRound-trip successful: {original_text == decoded_text}")

    # Demonstrate binary conversion round-trip
    print("\nBinary conversion round-trip:")
    decimal_from_binary = binary_to_decimal(binary_repr)
    epsilon = Decimal(f'1e-{len(original_text)+1}')
    print(f"Original decimal: {encoded_value}")
    print(f"From binary:      {decimal_from_binary}")
    print(f"Precision:        {epsilon}")
    print(f"Conversion accurate: {abs(encoded_value - decimal_from_binary) < epsilon}")


if __name__ == '__main__':
    main()
