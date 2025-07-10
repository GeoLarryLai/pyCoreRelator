"""
Log processing functions for pyCoreRelator.

This module contains functions for processing various types of log data
including RGB image analysis.
"""

from .rgb_image2log import (
    trim_image,
    extract_rgb_profile, 
    plot_rgb_profile,
    stitch_core_sections
)

__all__ = [
    'trim_image',
    'extract_rgb_profile',
    'plot_rgb_profile', 
    'stitch_core_sections'
] 