####################################################################
# Plotting functions for UrbanSurge
####################################################################

# Library imports.
import matplotlib.pyplot as plt
import numpy as np

def linear_spaced_array(value1, value2, N=5, D=1):
    """
    Generate a linearly spaced array of N numbers between two values with numbers 
    rounded to D decimal places.

    Parameters:
        value1 (float): The starting value of the array.
        value2 (float): The ending value of the array.
        N (int): Number of values in the array.
        D (int): Maximum number of decimal places.

    Returns:
        list: A list of N linearly spaced numbers rounded to D decimal places.
    """
    # Generate a linearly spaced array with N numbers
    linear_array = np.linspace(value1, value2, N)

    # Round the numbers to the specified number of decimal places
    rounded_array = np.round(linear_array, D)

    # Ensure unique values by rounding edge cases
    rounded_array = np.unique(rounded_array)

    # Check if we achieved the desired N numbers; if not, regenerate
    while len(rounded_array) < N:
        step = (value2 - value1) / (N - 1)
        linear_array = np.array([value1 + i * step for i in range(N)])
        rounded_array = np.round(linear_array, D)
        rounded_array = np.unique(rounded_array)

    return rounded_array.tolist()

# Example usage
result = linear_spaced_array(0.11, 0.36, N=4, D=1)
print(result)  # Output: [0.1, 0.2, 0.3, 0.4]
