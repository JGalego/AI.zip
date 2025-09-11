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
    
    def visualize_range(low_val, high_val, width=50):
        """Creates a visual representation of the current range."""
        # Calculate positions for the visual bar
        low_float = float(low_val)
        high_float = float(high_val)
        
        # Create the visual bar
        low_pos = int(low_float * width)
        high_pos = int(high_float * width)
        
        bar = [' '] * width
        for i in range(low_pos, min(high_pos + 1, width)):
            bar[i] = '█'
        
        return ''.join(bar)
    
    print(f"{'STEP':4s} {'CHAR':4s} {'LOW':12s} {'HIGH':12s} {'WIDTH':12s} {'RANGE VISUALIZATION':s}")
    print("-" * 85)
    
    # Show initial state
    range_viz = visualize_range(low, high)
    print(f"{'0':4s} {'':4s} {str(low):12s} {str(high):12s} {str(high-low):12s} |{range_viz}|")
    
    for i, ch in enumerate(text, 1):
        low, high = scaled_ranges[ch]
        delta = high - low
        scaled_ranges = {ch: (rng[0]*delta + low, rng[1]*delta + low) for ch, rng in ranges.items()}
        
        range_viz = visualize_range(low, high)
        print(f"{str(i):4s} {ch.upper():4s} {str(low):12s} {str(high):12s} {str(delta):12s} |{range_viz}|")
    
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

def range_to_binary(low: Decimal, high: Decimal, max_bits: int = 32) -> str:
    """
    Computes the binary sequence by iteratively splitting the interval [0, 1) in half
    until the [low, high) range is contained within it.
    
    This function implements a direct approach to arithmetic coding by finding the
    shortest binary representation that uniquely identifies the given range.
    
    Algorithm:
    1. Start with current interval [0, 1)
    2. For each bit position:
       - Split current interval at midpoint
       - If target range fits entirely in lower half: append '0'  
       - If target range fits entirely in upper half: append '1'
       - If target range spans both halves: choose the half with more overlap
    3. Stop when current interval is contained within target range
    
    Args:
        low: Lower bound of the range (inclusive)
        high: Upper bound of the range (exclusive)
        max_bits: Maximum number of binary digits to generate
    
    Returns:
        Binary string representation (e.g., "0.1011")
        
    Example:
        >>> range_to_binary(Decimal("0.5"), Decimal("1.0"))
        "0.1"
        >>> range_to_binary(Decimal("0.25"), Decimal("0.5")) 
        "0.01"
    """
    if low < 0 or high > 1 or low >= high:
        raise ValueError("Invalid range: must have 0 <= low < high <= 1")
    
    binary = "0."
    current_low = Decimal("0")
    current_high = Decimal("1")
    
    for _ in range(max_bits):
        # Split current interval in half
        mid = (current_low + current_high) / 2
        
        if high <= mid:
            # Range is entirely in the lower half [current_low, mid)
            binary += "0"
            current_high = mid
        elif low >= mid:
            # Range is entirely in the upper half [mid, current_high)
            binary += "1"
            current_low = mid
        else:
            # Range spans both halves - choose the half with more overlap
            lower_overlap = min(high, mid) - max(low, current_low)
            upper_overlap = min(high, current_high) - max(low, mid)
            
            if lower_overlap >= upper_overlap:
                binary += "0"
                current_high = mid
            else:
                binary += "1"
                current_low = mid
        
        # Check if we've narrowed down enough that our range is contained
        # This is an optimization - we could continue but don't need to
        if current_low >= low and current_high <= high:
            break
    
    return binary

def arithmetic_coding(text: str, dist: OrderedDict):
    """Demonstrates arithmetic coding with the given text and probability distribution."""

    # Original text
    box_print("Original")
    print(f"Original text: {text}")
    print(f"Distribution: {dict(dist)}")

    # Encoded text
    box_print("Encoded")
    low, high = arithmetic_encode(dist, text)

    print(f"\nFinal range: [{low}, {high})")
    print(f"Range width: {high - low}")

    # Choose a value within the range for encoding
    encoded_value = (low + high) / 2  # Use midpoint
    print(f"Encoded value (midpoint): {encoded_value}")

    # Convert to binary representation
    binary_repr = decimal_to_binary(encoded_value, precision=20)
    print(f"Binary representation (midpoint): {binary_repr}")
    
    # Convert range directly to binary
    range_binary = range_to_binary(low, high, max_bits=20)
    print(f"Binary representation (range):    {range_binary}")
    
    # Compare the two approaches
    range_decimal = binary_to_decimal(range_binary)
    print(f"Range binary as decimal: {range_decimal}")
    print(f"Range contains encoded value: {low <= range_decimal < high}")

    # Decode back to original text
    box_print("Decoded")
    decoded_text = arithmetic_decode(dist, encoded_value, len(text))
    print(f"Decoded text: {decoded_text}")

    # Verify the round-trip
    print(f"\nRound-trip successful: {text == decoded_text}")

    # Demonstrate binary conversion round-trip
    print("\nBinary conversion round-trip (midpoint approach):")
    decimal_from_binary = binary_to_decimal(binary_repr)
    epsilon = Decimal(f'1e-{len(text)+1}')
    print(f"Original decimal: {encoded_value}")
    print(f"From binary:      {decimal_from_binary}")
    print(f"Precision:        {epsilon}")
    print(f"Conversion accurate: {abs(encoded_value - decimal_from_binary) < epsilon}")
    
    print("\nBinary conversion round-trip (range approach):")
    print(f"Range binary:     {range_binary}")
    print(f"From binary:      {range_decimal}")
    print(f"In valid range:   {low <= range_decimal < high}")
    
    # Test decoding with the range-based binary
    decoded_from_range = arithmetic_decode(dist, range_decimal, len(text))
    print(f"Decoded from range binary: {decoded_from_range}")
    print(f"Range decode successful: {text == decoded_from_range}")


if __name__ == '__main__':
    # Example:
    # https://go-compression.github.io/algorithms/arithmetic
    dist = OrderedDict({'h': 0.2, 'e': 0.2, 'l': 0.4, 'o': 0.2})
    text = "hello"
    arithmetic_coding(text, dist)

