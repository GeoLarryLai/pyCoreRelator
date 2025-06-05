"""
General utility and helper functions
"""

import numpy as np


# Helper function to find the nearest index in a depth array to a given depth value
def find_nearest_index(depth_array, depth_value):
    """
    Find the index in depth_array that has the closest depth value to the given depth_value.
    
    Args:
        depth_array: Array of depth values
        depth_value: Target depth value to find
        
    Returns:
        Index in depth_array with the closest value to depth_value
    """
    return np.abs(np.array(depth_array) - depth_value).argmin()
