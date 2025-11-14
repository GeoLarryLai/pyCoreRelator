"""
RGB image plotting functions for pyCoreRelator.

Included Functions:
- plot_rgb_profile: Trim specified number of pixels from top and bottom of image array
 along the y-axis of an image file
- plot_rgb_profile: Create visualization plots of RGB analysis results
 together by processing RGB profiles
  (supports multiple empty segments with automatic numbering)

This module provides comprehensive tools for processing RGB images of geological cores,
extracting color profiles, and creating visualizations for core analysis.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_rgb_profile(depths, r, g, b, r_std, g_std, b_std, lum, lum_std, img, 
                    core_name=None, save_figs=False, output_dir=None):
    """
    Create visualization plots of RGB analysis results.
    
    This function generates a comprehensive three-panel visualization showing the
    original core image, RGB color profiles with standard deviation bands, and
    standard deviation plots. The visualization is optimized for geological core
    analysis and correlation studies.
    
    Parameters
    ----------
    depths : numpy.ndarray
        Depth positions in pixels for the RGB data
    r : numpy.ndarray
        Red color intensity values
    g : numpy.ndarray
        Green color intensity values
    b : numpy.ndarray
        Blue color intensity values
    r_std : numpy.ndarray
        Standard deviations of red values
    g_std : numpy.ndarray
        Standard deviations of green values
    b_std : numpy.ndarray
        Standard deviations of blue values
    lum : numpy.ndarray
        Relative luminance values
    lum_std : numpy.ndarray
        Standard deviations of luminance values
    img : numpy.ndarray
        Core image array to display
    core_name : str, optional
        Name of the core for plot title and file naming
    save_figs : bool, default=False
        Whether to save the plots as image files
    output_dir : str, optional
        Directory to save output files (required if save_figs is True)
        
    Returns
    -------
    None
        Displays the plot and optionally saves files
        
    Raises
    ------
    ValueError
        If output_dir is not provided when save_figs is True
        
    Example
    -------
    >>> plot_rgb_profile(depths, r, g, b, r_std, g_std, b_std, lum, lum_std, img,
    ...                  core_name='Core_A_RGB', save_figs=True, output_dir='./output')
    """
    # Create figure with three subplots side by side
    height_to_width_ratio = 2 * img.shape[0] / img.shape[1]
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7, height_to_width_ratio),
        gridspec_kw={'width_ratios': [1, 0.7, 0.3], 'wspace': 0.22}
    )

    # Plot the image on the left subplot
    img_normalized = np.clip(img / 255.0, 0, 1)  # Normalize and clip image data to [0,1] range
    ax1.imshow(img_normalized)
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_ylabel('Depth (pixels)')

    # Plot RGB profiles with standard deviation bands in middle subplot
    ax2.fill_betweenx(depths, lum - lum_std, lum + lum_std, color='black', alpha=0.2, linewidth=0)
    ax2.fill_betweenx(depths, r - r_std, r + r_std, color='red', alpha=0.2, linewidth=0)
    ax2.fill_betweenx(depths, g - g_std, g + g_std, color='green', alpha=0.2, linewidth=0)
    ax2.fill_betweenx(depths, b - b_std, b + b_std, color='blue', alpha=0.1, linewidth=0)

    ax2.plot(r, depths, 'r-', label='Red', linewidth=0.4)
    ax2.plot(g, depths, 'g-', label='Green', linewidth=0.4)
    ax2.plot(b, depths, 'b-', label='Blue', linewidth=0.4)
    ax2.plot(lum, depths, 'k--', label='Relative\\nLuminance', linewidth=1)

    # Set x-axis limits based on RGB value range only
    rgb_values = np.concatenate([r[~np.isnan(r)], g[~np.isnan(g)], b[~np.isnan(b)]])
    
    # Handle case where all values are NaN or no valid data
    if len(rgb_values) == 0:
        # Set default limits when no valid data is available
        ax2.set_xlim(0, 255)
    else:
        rgb_min = np.min(rgb_values)
        rgb_max = np.max(rgb_values)
        if rgb_min == rgb_max:
            # Handle case where all valid values are the same
            ax2.set_xlim(rgb_min - 10, rgb_max + 10)
        else:
            padding = (rgb_max - rgb_min) * 0.15  # Add 15% padding
            ax2.set_xlim(rgb_min - padding, rgb_max + padding)

    ax2.invert_yaxis()  # Invert y-axis to match image orientation
    ax2.set_ylim(max(depths), min(depths))  # Invert y-axis limits
    ax2.set_yticklabels([])
    ax2.set_xlabel('Color Intensity')
    ax2.legend(fontsize='x-small', loc='upper left')
    ax2.grid(True)

    # Plot standard deviations in right subplot
    ax3.plot(r_std, depths, 'r-', label='Red', linewidth=0.4)
    ax3.plot(g_std, depths, 'g-', label='Green', linewidth=0.4)
    ax3.plot(b_std, depths, 'b-', label='Blue', linewidth=0.4)
    ax3.plot(lum_std, depths, 'k--', label='Relative\\nLuminance', linewidth=1)

    ax3.set_xlabel('σ (STDEV)')
    ax3.set_ylim(ax2.get_ylim())
    ax3.grid(True)
    ax3.set_yticklabels([])
    ax3.set_xlim(left=0)  # Set x-axis to start at 0

    # Add text annotation if core name is provided
    if core_name:
        fig.text(.5, 0.89, core_name, fontweight='bold', ha='center', va='top')

    plt.show()

    # Save figures if requested
    if save_figs:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save_figs is True")
            
        # Save full figure as PNG and SVG
        output_file = os.path.join(output_dir, f"{core_name}.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"RGB profile results saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
        
        output_file = os.path.join(output_dir, f"{core_name}.svg")
        fig.savefig(output_file, bbox_inches='tight')
        print(f"RGB profile image saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
        
        # Save composite TIFF image with compression using PIL
        output_file = os.path.join(output_dir, f"{core_name}.tiff")
        # Clip values to valid range [0,1] and convert to uint8
        img_to_save = np.clip(img_normalized * 255, 0, 255).astype(np.uint8)
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img_to_save)
        # Save with ZIP compression
        pil_img.save(output_file, format='TIFF', compression='tiff_deflate')
        print(f"Core composite image saved to: ~/{'/'.join(output_file.split('/')[-2:])}")


