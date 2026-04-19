"""
General utility and helper functions for pyCoreRelator.

Included Functions:
- find_nearest_index: Find the index in depth_array that has the closest depth value to the given depth_value.
- validate_fig_formats: Validate and normalize figure format list.
- save_figure_formats: Save a matplotlib figure in multiple formats based on fig_format logic.

This module provides essential utility functions for depth-based operations and data
manipulation commonly used throughout the core correlation analysis workflow.
"""

import numpy as np
import os


_VALID_FIG_FORMATS = {'png', 'jpeg', 'jpg', 'svg', 'pdf'}


def validate_fig_formats(fig_format):
    """Validate and return a normalized list of figure formats.

    Accepted formats: 'png', 'jpeg', 'jpg', 'svg', 'pdf'.
    Falls back to ['png'] when *fig_format* is ``None``, empty, or
    contains only invalid entries.
    """
    if fig_format is None:
        return ['png']
    if isinstance(fig_format, str):
        fig_format = [fig_format]
    validated = [f.lower() for f in fig_format if f.lower() in _VALID_FIG_FORMATS]
    return validated if validated else ['png']


def save_figure_formats(fig, output_path, fig_format=None, dpi=150):
    """Save a figure in the requested formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    output_path : str
        Target file path (may contain an image extension).
    fig_format : list of str or None
        Desired output formats (e.g. ``['png', 'svg']``).
    dpi : int
        Resolution for raster formats.

    Returns
    -------
    str
        The primary saved file path (the original *output_path* when it
        has a recognised extension, otherwise ``base.{first_format}``).

    Saving rules
    ------------
    The function always saves one file per format in the **union** of
    *fig_format* and the extension of *output_path* (when recognised).
    This means every format requested via *fig_format* is produced, and
    any explicit extension on *output_path* is also honoured even if it
    is not listed in *fig_format*.
    """
    if not output_path:
        return None

    fig_format = validate_fig_formats(fig_format)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base, ext = os.path.splitext(output_path)
    ext_lower = ext.lstrip('.').lower()

    def _norm(e):
        return 'jpeg' if e == 'jpg' else e

    recognised = {'png', 'jpg', 'jpeg', 'svg', 'pdf'}

    # Build the union of all formats to save
    all_fmts = list(dict.fromkeys(fig_format))  # preserve order, deduplicate
    if ext_lower in recognised and _norm(ext_lower) not in {_norm(f) for f in all_fmts}:
        all_fmts.append(ext_lower)

    if ext_lower in recognised:
        primary_path = output_path
    else:
        base = output_path
        primary_path = f"{base}.{all_fmts[0]}"

    saved = set()
    for fmt in all_fmts:
        fmt_path = f"{base}.{fmt}"
        norm_key = os.path.abspath(fmt_path)
        if norm_key in saved:
            continue
        saved.add(norm_key)
        if fmt == 'jpeg':
            fig.savefig(fmt_path, dpi=dpi, bbox_inches='tight', format='jpg')
        else:
            fig.savefig(fmt_path, dpi=dpi, bbox_inches='tight')

    return primary_path


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
