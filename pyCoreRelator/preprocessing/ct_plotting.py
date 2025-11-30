"""
CT image plotting functions for pyCoreRelator.

This module provides plotting functions for CT image processing,
including slice visualization and stitched curve plotting.

Functions:
- display_slice: Display a slice with optional physical dimensions
- plot_ctimg_curves: Display a core slice with corresponding brightness trace and standard deviation plots
- plot_stitched_curves: Visualize stitched curves showing overlap regions and final result
"""

import os
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def display_slice(slice_data: np.ndarray, 
                 pixel_spacing: Optional[Tuple[float, float]] = None,
                 title: str = "Core Slice") -> None:
    """
    Display a slice with optional physical dimensions.
    
    This function creates a matplotlib visualization of a CT slice with proper
    scaling and labeling. If pixel spacing is provided, the axes will show
    physical distances in millimeters.
    
    Parameters
    ----------
    slice_data : numpy.ndarray
        2D numpy array of slice data to display
    pixel_spacing : tuple of float, optional
        Tuple of (x, y) pixel spacing in mm/pixel for physical scaling
    title : str, default="Core Slice"
        Title to display above the plot
        
    Returns
    -------
    None
        Displays the plot using matplotlib
        
    Example
    -------
    >>> slice_data = np.random.rand(100, 80) * 255
    >>> display_slice(slice_data, pixel_spacing=(0.5, 0.5), title="CT Slice")
    """
    plt.figure(figsize=(10, 6))
    
    if pixel_spacing is not None:
        extent = [0, slice_data.shape[1] * pixel_spacing[0],
                 0, slice_data.shape[0] * pixel_spacing[1]]
        plt.imshow(slice_data, extent=extent)
        plt.xlabel("Distance (mm)")
        plt.ylabel("Distance (mm)")
    else:
        plt.imshow(slice_data)
        plt.xlabel("Pixels")
        plt.ylabel("Pixels")
    
    plt.colorbar(label="Intensity")
    plt.title(title)
    plt.show()



def plot_ctimg_curves(slice_data: Optional[np.ndarray] = None,
                        brightness: Optional[np.ndarray] = None,
                        stddev: Optional[np.ndarray] = None,
                        pixel_spacing: Optional[Tuple[float, float]] = None,
                        ct_metadata: Optional[dict] = None,
                        core_name: str = "",
                        save_figs: bool = False,
                        output_dir: Optional[str] = None,
                        vmin = 400,
                        vmax = 2400,
                        fig_format: list = ['png', 'tiff'],
                        dpi: int = 150) -> None:
    """
    Display a core slice and corresponding brightness trace and standard deviation.
    
    This function creates a comprehensive visualization showing the CT slice
    alongside its brightness profile and standard deviation plots. The layout
    includes three panels: the CT image, brightness trace with standard deviation
    shading, and standard deviation profile.
    
    Parameters
    ----------
    slice_data : numpy.ndarray, optional
        2D numpy array of slice data to display (not required if ct_metadata is provided)
    brightness : numpy.ndarray, optional
        1D array of mean brightness values along depth (not required if ct_metadata is provided)
    stddev : numpy.ndarray, optional
        1D array of standard deviation values along depth (not required if ct_metadata is provided)
    pixel_spacing : tuple of float, optional
        Tuple of (x, y) pixel spacing in mm/pixel for physical scaling (not required if ct_metadata is provided)
    ct_metadata : dict, optional
        Dictionary from `ct_process_and_stitch()` containing all CT data. Expected keys:
        'slice', 'brightness', 'stddev', 'px_spacing_x', 'px_spacing_y'.
        If provided, individual parameters (slice_data, brightness, stddev, pixel_spacing) are ignored
    core_name : str, default=""
        Name of the core to display in title and filenames
    save_figs : bool, default=False
        Whether to save figures to files
    output_dir : str, optional
        Directory to save figures if save_figs is True
    vmin : float, optional
        Minimum value for colormap scaling. Defaults is 400
    vmax : float, optional
        Maximum value for colormap scaling. Defaults is 2400
    fig_format : list, default=['png', 'tiff']
        List of file formats to save. Acceptable formats: 'png', 'jpg'/'jpeg', 
        'svg', 'tiff', 'pdf'
    dpi : int, default=150
        Resolution in dots per inch for saved figures. Applies to all formats 
        in fig_format
        
    Returns
    -------
    None
        Displays the plot using matplotlib and optionally saves files
        
    Raises
    ------
    ValueError
        If save_figs is True but output_dir is not provided
        
    Example
    -------
    >>> # Using individual parameters
    >>> slice_data = np.random.rand(100, 80) * 255
    >>> brightness = np.random.rand(100) * 1000 + 400
    >>> stddev = np.random.rand(100) * 100 + 50
    >>> plot_ctimg_curves(slice_data, brightness, stddev, core_name="Test Core")
    
    >>> # Using metadata from ct_process_and_stitch
    >>> ct_metadata = ct_process_and_stitch(...)
    >>> plot_ctimg_curves(ct_metadata=ct_metadata, core_name="Test Core")
    """
    
    # Extract data from ct_metadata if provided
    if ct_metadata is not None:
        slice_data = ct_metadata['slice']
        brightness = ct_metadata['brightness']
        stddev = ct_metadata['stddev']
        pixel_spacing = (ct_metadata['px_spacing_x'], ct_metadata['px_spacing_y'])
    
    # Normalize format names (handle jpg/jpeg)
    fig_format = [fmt.lower() for fmt in fig_format]
    fig_format = ['jpeg' if fmt == 'jpg' else fmt for fmt in fig_format]
    
    # Create figure with 3 subplots with specific width ratios and smaller space between subplots
    # Calculate height based on data dimensions while keeping width fixed at 8
    height = 2 * (slice_data.shape[0] / slice_data.shape[1])
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, 
        figsize=(7, height),
        sharey=True,
        gridspec_kw={'width_ratios': [1.7, 0.6, 0.3], 'wspace': 0.22}
    )
    # Plot the slice
    slice_dim = slice_data.shape
    if pixel_spacing is not None:
        extent = [
            0, slice_dim[1] * pixel_spacing[0],
            slice_dim[0] * pixel_spacing[1], 0
        ]  # Note: y-axis inverted
        im = ax1.imshow(slice_data, extent=extent, cmap='jet', vmin=vmin, vmax=vmax)
        ax1.set_xlabel("Width (mm)", fontsize='small')
        ax1.set_ylabel("Depth (mm)", fontsize='small')
    else:
        im = ax1.imshow(slice_data, cmap='jet', vmin=vmin, vmax=vmax)
        ax1.set_xlabel("Width (pixels)", fontsize='small')
        ax1.set_ylabel("Depth (pixels)", fontsize='small')
    
    # Add colorbar with small tick labels
    cbar = plt.colorbar(im, ax=ax1)
    cbar.ax.tick_params(labelsize='x-small')
    
    # Calculate y coordinates for brightness and stddev plots
    y_coords = np.arange(len(brightness))
    if pixel_spacing is not None:
        y_coords = y_coords * pixel_spacing[1]
    
    # Plot brightness trace with standard deviation shading
    ax2.plot(brightness, y_coords, 'b-', label='Mean')
    ax2.fill_betweenx(
        y_coords, 
        brightness - stddev, 
        brightness + stddev, 
        alpha=0.1, color='b', label='±1σ', linewidth=0
    )
    ax2.set_xlabel("CT# ±1σ", fontsize='small')
    ax2.grid(True)
    ax2.set_xlim(left=400)  # Set x-axis to start at 400
    
    # Plot standard deviation
    ax3.plot(stddev, y_coords)
    ax3.set_xlabel("σ (STDEV)", fontsize='small')
    ax3.grid(True)
    
    # Add text annotation if core name is provided
    if core_name:
        fig.text(.5, 0.92, core_name, fontweight='bold', ha='center', va='top')
    
    # Make tick labels smaller for all axes
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize='x-small')
    
    # show plot
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
                print(f"Composite CT results saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
        
        # Create and save colormap image if tiff is requested
        if 'tiff' in fig_format:
            # Calculate aspect ratio of data
            data_height, data_width = slice_data.shape
            data_aspect = data_height / data_width
            
            # Set figure size to maintain data aspect ratio while filling width
            fig_width = 2  # Keep original width
            fig_height = fig_width * data_aspect
            
            fig_img = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            
            # Create axes that fills entire figure
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            
            if pixel_spacing is not None:
                im_img = plt.imshow(slice_data, extent=extent, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
            else:
                im_img = plt.imshow(slice_data, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
                
            # Ensure no padding
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            
            # Convert matplotlib figure to RGB array
            fig_img.canvas.draw()
            img_array = np.array(fig_img.canvas.renderer.buffer_rgba())[:,:,:3]
            
            # Save as Deflate-compressed TIFF using PIL with specified DPI
            output_file = os.path.join(output_dir, f"{core_name}.tiff")
            img = Image.fromarray(img_array)
            img.save(output_file, format='TIFF', compression='tiff_deflate', dpi=(dpi, dpi))
            print(f"CT image saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
            plt.close(fig_img)



def plot_stitched_curves(stitched_depth, stitched_brightness, stitched_stddev, 
                        brightness_1, brightness_2, stddev_1, stddev_2,
                        brightness_2_shifted, stddev_2_shifted,
                        final_overlap, overlap_depth_1, overlap_depth_2, px_spacing_y_1, px_spacing_y_2):
    """
    Visualize stitched curves showing overlap regions and final result.
    
    This function creates a comprehensive plot showing the original curves,
    their overlap region, and the final stitched result. It displays both
    brightness and standard deviation data with clear visual indicators
    of the stitching process.
    
    Parameters
    ----------
    stitched_depth : numpy.ndarray
        Depth coordinates for stitched curve
    stitched_brightness : numpy.ndarray
        Stitched brightness values
    stitched_stddev : numpy.ndarray
        Stitched standard deviation values
    brightness_1 : numpy.ndarray
        Original brightness values from first curve
    brightness_2 : numpy.ndarray
        Original brightness values from second curve
    stddev_1 : numpy.ndarray
        Original standard deviation values from first curve
    stddev_2 : numpy.ndarray
        Original standard deviation values from second curve
    brightness_2_shifted : numpy.ndarray
        Adjusted brightness values for second curve
    stddev_2_shifted : numpy.ndarray
        Adjusted standard deviation values for second curve
    final_overlap : int
        Length of overlap region in pixels
    overlap_depth_1 : numpy.ndarray
        Depth coordinates for overlap region from first curve
    overlap_depth_2 : numpy.ndarray
        Depth coordinates for overlap region from second curve
    px_spacing_y_1 : float
        Pixel spacing for first curve
    px_spacing_y_2 : float
        Pixel spacing for second curve
        
    Returns
    -------
    None
        Displays the plot using matplotlib
        
    Example
    -------
    >>> # After running stitch_curves()
    >>> plot_stitched_curves(stitched_depth, stitched_brightness, stitched_stddev,
    ...                      brightness_1, brightness_2, stddev_1, stddev_2,
    ...                      brightness_2_shifted, stddev_2_shifted,
    ...                      final_overlap, overlap_depth_1, overlap_depth_2, 
    ...                      px_spacing_y_1, px_spacing_y_2)
    """
    # Plot the stitched results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    
    # Calculate overlap region correctly using overlap_depth_1
    overlap_start = overlap_depth_1[0]
    overlap_end = overlap_depth_1[-1]
    
    # Add shaded overlap region for brightness
    ax1.axvspan(overlap_start, overlap_end, 
                color='gray', alpha=0.2, label='Overlap Region')
    
    # Plot overlapping sections
    ax1.plot(overlap_depth_1, brightness_1[-final_overlap:], 'g:', label='Overlapped Curve 1')
    ax1.plot(overlap_depth_1, brightness_2_shifted[:final_overlap], 'r:', label='Adjusted Curve 2')
    
    # Plot brightness curves
    ax1.plot(stitched_depth, stitched_brightness, 'b-', label='Stitched Curves')
    
    # Create depth arrays for original and shifted curves using respective spacings
    depth_2 = np.arange(len(brightness_2)) * px_spacing_y_2
    depth_2_shifted = depth_2 + overlap_start  # Adjust to use overlap_start
    
    ax1.set_xlabel('Depth (mm)')
    ax1.set_ylabel('Adjusted CT#')
    ax1.grid(True)
    ax1.legend(loc='upper left', fontsize='x-small')
    
    # Add shaded overlap region for stddev
    ax2.axvspan(overlap_start, overlap_end, 
                color='gray', alpha=0.2, label='Overlap Region')
    
    # Plot overlapping sections
    ax2.plot(overlap_depth_1, stddev_1[-final_overlap:], 'g:', label='Overlapped Curve 1')
    ax2.plot(overlap_depth_1, stddev_2_shifted[:final_overlap], 'r:', label='Adjusted Curve 2')
    
    # Plot standard deviation curves
    ax2.plot(stitched_depth, stitched_stddev, 'b-', label='Stitched Curves')
    ax2.set_xlabel('Depth (mm)')
    ax2.set_ylabel('Adjusted σ (STDEV)')
    ax2.grid(True)
    ax2.legend(loc='upper left', fontsize='x-small')
    
    plt.tight_layout()
    plt.show()


