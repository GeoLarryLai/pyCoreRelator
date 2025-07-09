"""
General utility and helper functions for pyCoreRelator.

Included Functions:
- find_nearest_index: Find the index in depth_array that has the closest depth value to the given depth_value.

This module provides essential utility functions for depth-based operations and data
manipulation commonly used throughout the core correlation analysis workflow.
"""

import numpy as np


def find_nearest_index(depth_array, depth_value):
    """
    Find the index in depth_array that has the closest depth value to the given depth_value.
    
    This function is commonly used when converting between depth values and array indices
    in core log data analysis, particularly for finding segment boundaries and correlation points.
    
    Parameters
    ----------
    depth_array : array-like
        Array of depth values, typically measured depths from core logs
    depth_value : float
        Target depth value to find the nearest match for
        
    Returns
    -------
    int
        Index in depth_array with the closest value to depth_value
        
    Example
    -------
    >>> import numpy as np
    >>> depths = np.array([10.5, 11.2, 12.1, 13.0, 14.5])
    >>> target_depth = 12.5
    >>> idx = find_nearest_index(depths, target_depth)
    >>> print(f"Nearest index: {idx}, depth: {depths[idx]}")
    Nearest index: 2, depth: 12.1
    """
    return np.abs(np.array(depth_array) - depth_value).argmin()


def cohens_d(x, y):
    """Calculate Cohen's d for effect size between two samples"""
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    # Cohen's d
    d = (np.mean(x) - np.mean(y)) / s_pooled
    return d
