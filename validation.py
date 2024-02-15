"""
   Name: validation.py
   Purpose: Validation functions for the project
   Created on: 2/13/2024
   Created by: Heng Sun
   Additional Notes: 
"""
import numpy as np


def validate_equal(a,b,Name_a = "First Item", Name_b = "Second Item", atol = 1e-8, rtol = 1e-5, equal_nan = False):
    """
    Validate if two items are equal in absolute value
    """
    if np.allclose(abs(a), abs(b), atol = atol, rtol = rtol, equal_nan = equal_nan):
        print(f"{Name_a} and {Name_b} are equal in absolute value")
    else:
        print(f"{Name_a} and {Name_b} are not equal in absolute value")