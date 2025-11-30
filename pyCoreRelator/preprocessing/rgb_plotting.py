"""
RGB image plotting functions for pyCoreRelator.

Included Functions:
- plot_rgbimg_curves: Create visualization plots of RGB analysis results showing
  core image alongside RGB color profiles and standard deviation plots
  (supports multiple empty segments with automatic numbering)

This module provides comprehensive tools for plotting RGB images of geological cores
and creating visualizations for core analysis.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_rgbimg_curves(depths=None, r=None, g=None, b=None, r_std=None, g_std=None, b_std=None, lum=None, lum_std=None, img=None,
                    rgb_metadata=None, core_name=None, save_figs=False, output_dir=None, fig_format=['png', 'tiff'], dpi=150):
    """
    Create visualization plots of RGB analysis results.
    
    This function generates a comprehensive three-panel visualization showing the
    original core image, RGB color profiles with standard deviation bands, and
    standard deviation plots. The visualization is optimized for geological core
    analysis and correlation studies.
    
    Parameters
    ----------
    depths : numpy.ndarray, optional
        Depth positions in pixels for the RGB data (not required if rgb_metadata is provided)
    r : numpy.ndarray, optional
        Red color intensity values (not required if rgb_metadata is provided)
    g : numpy.ndarray, optional
        Green color intensity values (not required if rgb_metadata is provided)
    b : numpy.ndarray, optional
        Blue color intensity values (not required if rgb_metadata is provided)
    r_std : numpy.ndarray, optional
        Standard deviations of red values (not required if rgb_metadata is provided)
    g_std : numpy.ndarray, optional
        Standard deviations of green values (not required if rgb_metadata is provided)
    b_std : numpy.ndarray, optional
        Standard deviations of blue values (not required if rgb_metadata is provided)
    lum : numpy.ndarray, optional
        Relative luminance values (not required if rgb_metadata is provided)
    lum_std : numpy.ndarray, optional
        Standard deviations of luminance values (not required if rgb_metadata is provided)
    img : numpy.ndarray, optional
        Core image array to display (not required if rgb_metadata is provided)
    rgb_metadata : dict, optional
        Dictionary containing all RGB data from rgb_process_and_stitch(). Expected keys:
        'depths', 'r', 'g', 'b', 'r_std', 'g_std', 'b_std', 'lum', 'lum_std', 'image'.
        If provided, individual parameters are ignored.
    core_name : str, optional
        Name of the core for plot title and file naming
    save_figs : bool, default=False
        Whether to save the plots as image files
    output_dir : str, optional
        Directory to save output files (required if save_figs is True)
    fig_format : list, default=['png', 'tiff']
        List of file formats to save. Acceptable formats: 'png', 'jpg'/'jpeg', 
        'svg', 'tiff', 'pdf'
    dpi : int, default=150
        Resolution in dots per inch for saved figures. Applies to all formats 
        in fig_format
        
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
    >>> plot_rgbimg_curves(depths, r, g, b, r_std, g_std, b_std, lum, lum_std, img,
    ...                     core_name='Core_A_RGB', save_figs=True, output_dir='./output',
    ...                     fig_format=['png', 'svg'])
    >>> # Or using metadata from rgb_process_and_stitch:
    >>> plot_rgbimg_curves(rgb_metadata=rgb_metadata, core_name='Core_A_RGB', save_figs=True, output_dir='./output')
    """
    # Extract data from rgb_metadata if provided
    if rgb_metadata is not None:
        depths = rgb_metadata['depths']
        r = rgb_metadata['r']
        g = rgb_metadata['g']
        b = rgb_metadata['b']
        r_std = rgb_metadata['r_std']
        g_std = rgb_metadata['g_std']
        b_std = rgb_metadata['b_std']
        lum = rgb_metadata['lum']
        lum_std = rgb_metadata['lum_std']
        img = rgb_metadata['image']
    
    # Normalize format names (handle jpg/jpeg)
    fig_format = [fmt.lower() for fmt in fig_format]
    fig_format = ['jpeg' if fmt == 'jpg' else fmt for fmt in fig_format]
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
    ax2.plot(lum, depths, 'k--', label='Relative\nLuminance', linewidth=1)

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
    ax3.plot(lum_std, depths, 'k--', label='Relative\nLuminance', linewidth=1)

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
            
        # Save composite figure in requested formats
        for fmt in fig_format:
            if fmt in ['png', 'jpeg', 'svg', 'pdf']:
                output_file = os.path.join(output_dir, f"{core_name}.{fmt}")
                if fmt == 'jpeg':
                    fig.savefig(output_file, dpi=dpi, bbox_inches='tight', format='jpg')
                else:
                    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
                print(f"RGB profile results saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
        
        # Save RGB image only as TIFF with Deflate compression if tiff is requested
        if 'tiff' in fig_format:
            # Calculate aspect ratio of data
            data_height, data_width = img.shape[0], img.shape[1]
            data_aspect = data_height / data_width
            
            # Set figure size to maintain data aspect ratio
            fig_width = 2
            fig_height = fig_width * data_aspect
            
            fig_img = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            
            # Create axes that fills entire figure
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            
            # Display the normalized image
            ax.imshow(img_normalized, aspect='auto')
            
            # Ensure no padding
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            
            # Convert matplotlib figure to RGB array
            fig_img.canvas.draw()
            img_array = np.array(fig_img.canvas.renderer.buffer_rgba())[:,:,:3]
            
            # Save as Deflate-compressed TIFF using PIL with specified DPI
            output_file = os.path.join(output_dir, f"{core_name}.tiff")
            pil_img = Image.fromarray(img_array)
            pil_img.save(output_file, format='TIFF', compression='tiff_deflate', dpi=(dpi, dpi))
            print(f"Core composite image saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
            plt.close(fig_img)


