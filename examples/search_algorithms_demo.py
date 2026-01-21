"""
Example: Function Search with Binary Search Algorithms

This example demonstrates finding the input x for which f(x) = target_value
using the binary search algorithm for functions.

Demonstrates:
- Square root calculation
- Cube root calculation
- Temperature conversion
- Exponential function inverse
"""

import math
from binary_search_algorithms import BinarySearch


def main():
    print("=" * 60)
    print("Example: Function Search")
    print("=" * 60)
    
    # Example 1: Square root
    print("\n" + "=" * 60)
    print("Example 1: Square Root (find x where x² = 25)")
    print("=" * 60)
    
    def square(x):
        return x ** 2
    
    target = 25
    x = BinarySearch.search_for_function(target, square, tolerance=1e-6)
    print(f"Target: {target}")
    print(f"Found x: {x:.6f}")
    print(f"Verification: {x}² = {square(x):.6f}")
    print(f"Error: {abs(square(x) - target):.9f}")
    
    # Example 2: Cube root
    print("\n" + "=" * 60)
    print("Example 2: Cube Root (find x where x³ = 8)")
    print("=" * 60)
    
    def cube(x):
        return x ** 3
    
    target = 8
    x = BinarySearch.search_for_function(target, cube, tolerance=1e-6)
    print(f"Target: {target}")
    print(f"Found x: {x:.6f}")
    print(f"Verification: {x}³ = {cube(x):.6f}")
    print(f"Expected: 2.0 (cube root of 8)")
    
    # Example 3: Temperature conversion
    print("\n" + "=" * 60)
    print("Example 3: Temperature Conversion")
    print("Find Fahrenheit for 37°C (body temperature)")
    print("=" * 60)
    
    def celsius_to_fahrenheit(c):
        return (9/5) * c + 32
    
    target_f = celsius_to_fahrenheit(37)  # Expected: 98.6°F
    
    def fahrenheit_to_celsius(f):
        return (5/9) * (f - 32)
    
    # Find what Celsius gives us 98.6°F
    c = BinarySearch.search_for_function(target_f, celsius_to_fahrenheit, tolerance=0.01)
    print(f"Target Fahrenheit: {target_f:.1f}°F")
    print(f"Found Celsius: {c:.2f}°C")
    print(f"Expected: 37.0°C")
    print(f"Verification: {c:.2f}°C = {celsius_to_fahrenheit(c):.1f}°F")
    
    # Example 4: Exponential (demonstrates bug fix)
    print("\n" + "=" * 60)
    print("Example 4: Natural Log (find x where e^x = 10)")
    print("Demonstrates overflow protection (BUG FIX v1.0)")
    print("=" * 60)
    
    def exp(x):
        return math.exp(x)
    
    target = 10
    x = BinarySearch.search_for_function(target, exp, tolerance=0.01)
    print(f"Target: {target}")
    print(f"Found x: {x:.6f}")
    print(f"Verification: e^{x:.6f} = {exp(x):.6f}")
    print(f"Expected: ln(10) ≈ 2.302585")
    print(f"Actual ln(10): {math.log(10):.6f}")
    print("\n✅ No overflow error (bug fixed in v1.0)")
    
    # Example 5: Array search
    print("\n" + "=" * 60)
    print("Example 5: Array Search with Tolerance")
    print("=" * 60)
    
    searcher = BinarySearch()
    
    # Price list (sorted)
    prices = [9.99, 19.99, 29.99, 49.99, 99.99, 199.99, 499.99]
    budget = 45.00
    
    print(f"Available prices: {prices}")
    print(f"Budget: ${budget:.2f}")
    
    index, is_max, is_min = searcher.search_for_array(budget, prices, tolerance=0.01)
    
    print(f"\nClosest price: ${prices[index]:.2f} (index {index})")
    if prices[index] <= budget:
        print(f"✅ Within budget!")
    else:
        print(f"⚠️  Slightly over budget")
    
    # Example 6: Dictionary search
    print("\n" + "=" * 60)
    print("Example 6: Dictionary Search")
    print("Find product by approximate price")
    print("=" * 60)
    
    products = {
        'Basic': 29.99,
        'Standard': 49.99,
        'Premium': 79.99,
        'Enterprise': 149.99
    }
    
    target_price = 60.00
    print(f"Products: {products}")
    print(f"Looking for price near: ${target_price:.2f}")
    
    value, key = searcher.binary_search_to_find_miniterm_from_dict(target_price, products)
    print(f"\nBest match: {key} at ${value:.2f}")
    print(f"Difference: ${abs(value - target_price):.2f}")
    
    print("\n" + "=" * 60)
    print("✅ All examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
