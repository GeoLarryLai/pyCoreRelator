"""
CT image processing functions for pyCoreRelator.

Included Functions:
- load_dicom_files: Load DICOM files from a directory and return 3D volume data
- get_slice: Extract a 2D slice from the 3D volume along specified axis
- trim_slice: Trim empty space from a slice based on intensity threshold
- get_brightness_trace: Calculate brightness trace along specified axis
- get_brightness_stats: Calculate brightness mean and standard deviation along an axis
- display_slice: Display a slice with optional physical dimensions
- display_slice_bt_std: Display a core slice with corresponding brightness trace and standard deviation plots
- process_brightness_data: Process CT scan data to get brightness and standard deviation statistics

This module provides comprehensive tools for processing CT images of geological cores,
extracting brightness profiles, and creating visualizations for core analysis.
"""

import os
from typing import Tuple, List, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.signal import correlate
import pydicom
import cv2
from PIL import Image


def load_dicom_files(dir_path: str, force: bool = True) -> Tuple[np.ndarray, float, float, float]:
    """
    Load DICOM files from a directory and return the 3D volume data.
    
    This function loads all DICOM files from a specified directory, sorts them by
    z-coordinate position, and creates a 3D volume array. It also extracts pixel
    spacing and slice thickness information from the DICOM headers.
    
    Parameters
    ----------
    dir_path : str
        Path to directory containing DICOM files
    force : bool, default=True
        If True, ignore files that produce errors and continue processing
        
    Returns
    -------
    tuple
        Contains the following elements:
        - volume_data (numpy.ndarray): 3D array with shape (height, width, slices)
        - pixel_spacing_x (float): Pixel spacing in x direction (mm/pixel)
        - pixel_spacing_y (float): Pixel spacing in y direction (mm/pixel)
        - slice_thickness (float): Thickness of each slice (mm)
        
    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist
    ValueError
        If no valid DICOM files are found in the directory
        
    Example
    -------
    >>> volume, px_x, px_y, thickness = load_dicom_files('/path/to/dicom/files')
    >>> print(f"Volume shape: {volume.shape}")
    >>> print(f"Pixel spacing: {px_x:.3f} x {px_y:.3f} mm")
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Get list of files in directory
    files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    
    if not files:
        raise ValueError(f"No files found in directory: {dir_path}")
    
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dir_path, f))
            slices.append(ds)
        except Exception as e:
            if not force:
                raise ValueError(f"Error reading DICOM file '{f}': {str(e)}")
            continue
    
    if not slices:
        raise ValueError(f"No valid DICOM files found in directory: {dir_path}")
    
    # Sort slices by ImagePositionPatient z coordinate
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Get image dimensions and spacing info
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = float(slices[0].SliceThickness)
    
    # Create 3D numpy array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    volume_data = np.zeros(img_shape)
    
    # Fill 3D array with the images from the files
    for i, s in enumerate(slices):
        volume_data[:,:,i] = s.pixel_array
        
    return volume_data, pixel_spacing[0], pixel_spacing[1], slice_thickness


def get_slice(volume: np.ndarray, index: int, axis: int = 0) -> np.ndarray:
    """
    Extract a 2D slice from the 3D volume along specified axis.
    
    This function extracts a single 2D slice from a 3D volume array at the
    specified index along the given axis. Useful for viewing different
    orientations of the CT volume data.
    
    Parameters
    ----------
    volume : numpy.ndarray
        3D numpy array of CT data with shape (height, width, depth)
    index : int
        Index of slice to extract along the specified axis
    axis : int, default=0
        Axis along which to take slice:
        - 0: Extract slice perpendicular to height (sagittal view)
        - 1: Extract slice perpendicular to width (coronal view)  
        - 2: Extract slice perpendicular to depth (axial view)
        
    Returns
    -------
    numpy.ndarray
        2D numpy array representing the extracted slice
        
    Raises
    ------
    IndexError
        If index is out of bounds for the specified axis
    ValueError
        If axis is not 0, 1, or 2
        
    Example
    -------
    >>> volume = np.random.rand(100, 80, 60)
    >>> slice_data = get_slice(volume, index=50, axis=2)
    >>> slice_data.shape
    (100, 80)
    """
    if axis not in [0, 1, 2]:
        raise ValueError("Axis must be 0, 1, or 2")
        
    if axis == 0:
        if index >= volume.shape[0]:
            raise IndexError(f"Index {index} out of bounds for axis 0 (max: {volume.shape[0]-1})")
        return volume[index, :, :]
    elif axis == 1:
        if index >= volume.shape[1]:
            raise IndexError(f"Index {index} out of bounds for axis 1 (max: {volume.shape[1]-1})")
        return volume[:, index, :]
    else:  # axis == 2
        if index >= volume.shape[2]:
            raise IndexError(f"Index {index} out of bounds for axis 2 (max: {volume.shape[2]-1})")
        return volume[:, :, index]


def trim_slice(slice_data: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Trim empty space from a slice based on intensity threshold.
    
    This function removes empty or low-intensity regions from the edges of a
    CT slice by finding regions where pixel values exceed the threshold and
    cropping to the bounding box of these regions.
    
    Parameters
    ----------
    slice_data : numpy.ndarray
        2D numpy array of slice data
    threshold : float, default=0.05
        Intensity value below which pixels are considered empty or background
        
    Returns
    -------
    numpy.ndarray
        Trimmed 2D numpy array with reduced dimensions
        
    Raises
    ------
    ValueError
        If no pixels exceed the threshold (entire image would be trimmed)
        
    Example
    -------
    >>> slice_data = np.random.rand(100, 100) * 255
    >>> slice_data[:10, :] = 0  # Add empty top border
    >>> trimmed = trim_slice(slice_data, threshold=1.0)
    >>> trimmed.shape[0] < slice_data.shape[0]
    True
    """
    # Find non-empty rows and columns
    row_mask = np.any(slice_data > threshold, axis=1)
    col_mask = np.any(slice_data > threshold, axis=0)
    
    # Get indices of non-empty regions
    row_indices = np.where(row_mask)[0]
    col_indices = np.where(col_mask)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        raise ValueError(f"No pixels exceed threshold {threshold}. Cannot trim slice.")
    
    # Trim the slice
    return slice_data[row_indices[0]:row_indices[-1]+1, 
                     col_indices[0]:col_indices[-1]+1]


def get_brightness_trace(slice_data: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Calculate brightness trace along specified axis.
    
    This function computes the mean brightness values along the specified axis
    of a CT slice, creating a 1D profile that represents the average intensity
    variation across the slice.
    
    Parameters
    ----------
    slice_data : numpy.ndarray
        2D numpy array of slice data
    axis : int, default=1
        Axis along which to calculate mean:
        - 0: Calculate mean along vertical direction (horizontal profile)
        - 1: Calculate mean along horizontal direction (vertical profile)
        
    Returns
    -------
    numpy.ndarray
        1D numpy array of mean brightness values
        
    Raises
    ------
    ValueError
        If axis is not 0 or 1
        
    Example
    -------
    >>> slice_data = np.random.rand(100, 80) * 255
    >>> trace = get_brightness_trace(slice_data, axis=1)
    >>> len(trace) == slice_data.shape[0]
    True
    """
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1")
        
    return np.mean(slice_data, axis=axis)


def get_brightness_stats(slice_data: np.ndarray, axis: int = 1, width_start_pct: float = 0.25, 
                        width_end_pct: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate brightness mean and standard deviation along specified axis.
    
    This function computes statistical measures of brightness along the specified
    axis within a central strip of the slice. The strip is defined by percentage
    positions along the width, allowing analysis of the core region while
    excluding edge artifacts.
    
    Parameters
    ----------
    slice_data : numpy.ndarray
        2D numpy array of slice data
    axis : int, default=1
        Axis along which to calculate stats:
        - 0: Calculate stats along vertical direction
        - 1: Calculate stats along horizontal direction
    width_start_pct : float, default=0.25
        Starting percentage of width for the analysis strip (0.0 to 1.0)
    width_end_pct : float, default=0.75
        Ending percentage of width for the analysis strip (0.0 to 1.0)
        
    Returns
    -------
    tuple
        Contains the following arrays:
        - mean_values (numpy.ndarray): Mean brightness values along axis
        - std_values (numpy.ndarray): Standard deviation values along axis
        
    Raises
    ------
    ValueError
        If percentages are invalid or axis is not 0 or 1
        
    Example
    -------
    >>> slice_data = np.random.rand(100, 80) * 255
    >>> means, stds = get_brightness_stats(slice_data, axis=1)
    >>> len(means) == slice_data.shape[0]
    True
    """
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1")
        
    if not (0.0 <= width_start_pct <= 1.0) or not (0.0 <= width_end_pct <= 1.0):
        raise ValueError("Width percentages must be between 0.0 and 1.0")
        
    if width_start_pct >= width_end_pct:
        raise ValueError("width_start_pct must be less than width_end_pct")
    
    # Calculate the start and end indices based on provided percentages
    width = slice_data.shape[1]
    start_idx = int(width * width_start_pct)
    end_idx = int(width * width_end_pct)
    
    # Use only the specified strip for calculations
    center_slice = slice_data[:, start_idx:end_idx]
    
    return np.mean(center_slice, axis=axis), np.std(center_slice, axis=axis)


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


def display_slice_bt_std(slice_data: np.ndarray,
                        brightness: np.ndarray,
                        stddev: np.ndarray,
                        pixel_spacing: Optional[Tuple[float, float]] = None,
                        core_name: str = "",
                        save_figs: bool = False,
                        output_dir: Optional[str] = None,
                        vmin = None,
                        vmax = None) -> None:
    """
    Display a core slice and corresponding brightness trace and standard deviation.
    
    This function creates a comprehensive visualization showing the CT slice
    alongside its brightness profile and standard deviation plots. The layout
    includes three panels: the CT image, brightness trace with standard deviation
    shading, and standard deviation profile.
    
    Parameters
    ----------
    slice_data : numpy.ndarray
        2D numpy array of slice data to display
    brightness : numpy.ndarray
        1D array of mean brightness values along depth
    stddev : numpy.ndarray
        1D array of standard deviation values along depth
    pixel_spacing : tuple of float, optional
        Tuple of (x, y) pixel spacing in mm/pixel for physical scaling
    core_name : str, default=""
        Name of the core to display in title and filenames
    save_figs : bool, default=False
        Whether to save figures to files
    output_dir : str, optional
        Directory to save figures if save_figs is True
    vmin : float, optional
        Minimum value for colormap scaling. If None, defaults to 400
    vmax : float, optional
        Maximum value for colormap scaling. If None, defaults to 1700
        
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
    >>> slice_data = np.random.rand(100, 80) * 255
    >>> brightness = np.random.rand(100) * 1000 + 400
    >>> stddev = np.random.rand(100) * 100 + 50
    >>> display_slice_bt_std(slice_data, brightness, stddev, core_name="Test Core")
    """
    # Set default colormap values if not provided
    if vmin is None:
        vmin = 400
    if vmax is None:
        vmax = 1700
    
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
            
        # Save full figure as PNG and SVG
        output_file = os.path.join(output_dir, f"{core_name}.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Composite CT results saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
        
        output_file = os.path.join(output_dir, f"{core_name}.svg") 
        fig.savefig(output_file, bbox_inches='tight')
        print(f"Composite CT image saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
        
        # Create and save colormap image as compressed TIFF
        # Calculate aspect ratio of data
        data_height, data_width = slice_data.shape
        data_aspect = data_height / data_width
        
        # Set figure size to maintain data aspect ratio while filling width
        fig_width = 2  # Keep original width
        fig_height = fig_width * data_aspect
        
        fig_img = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        
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
        
        # Save as compressed TIFF using PIL
        output_file = os.path.join(output_dir, f"{core_name}.tiff")
        img = Image.fromarray(img_array)
        img.save(output_file, format='TIFF', compression='tiff_deflate')
        print(f"CT image saved to: ~/{'/'.join(output_file.split('/')[-2:])}")
        plt.close(fig_img)


def process_brightness_data(slice_data, px_spacing_y, trim_top, trim_bottom, min_brightness=400, buffer=5, 
                           width_start_pct=0.25, width_end_pct=0.75):
    """
    Process CT scan slice data to get brightness and standard deviation statistics with masking.
    
    This function processes a CT slice to extract brightness statistics while applying
    trimming, filtering, and masking operations. It trims the slice based on specified
    parameters, calculates brightness statistics within a central strip, and masks
    regions below a minimum brightness threshold with buffer zones.
    
    Parameters
    ----------
    slice_data : numpy.ndarray
        2D numpy array of CT scan slice data
    px_spacing_y : float
        Pixel spacing in y direction (mm/pixel) for buffer calculations
    trim_top : int
        Number of pixels to trim from top of slice
    trim_bottom : int
        Number of pixels to trim from bottom of slice
    min_brightness : float, default=400
        Minimum brightness threshold below which values are masked
    buffer : float, default=5
        Buffer size in mm around masked values
    width_start_pct : float, default=0.25
        Starting percentage of width to use for brightness calculation (0.0 to 1.0)
    width_end_pct : float, default=0.75
        Ending percentage of width to use for brightness calculation (0.0 to 1.0)
        
    Returns
    -------
    tuple
        Contains the following arrays:
        - brightness (numpy.ndarray): Masked brightness values with NaN for excluded regions
        - stddev (numpy.ndarray): Masked standard deviation values with NaN for excluded regions  
        - trimmed_slice (numpy.ndarray): Processed and trimmed slice data
        
    Raises
    ------
    ValueError
        If trim parameters exceed slice dimensions or percentages are invalid
        
    Example
    -------
    >>> slice_data = np.random.rand(1000, 500) * 1000 + 300
    >>> brightness, stddev, trimmed = process_brightness_data(
    ...     slice_data, px_spacing_y=0.1, trim_top=10, trim_bottom=10
    ... )
    >>> np.any(np.isnan(brightness))  # Should have some NaN values due to masking
    True
    """
    if trim_top < 0 or trim_bottom < 0:
        raise ValueError("Trim values must be non-negative")
        
    if trim_top + trim_bottom >= slice_data.shape[0]:
        raise ValueError("Total trim amount exceeds slice height")
    
    # Trim the slice based on trim values
    if trim_top == 0 and trim_bottom == 0:
        trimmed_slice = trim_slice(slice_data)
    elif trim_top == 0:
        trimmed_slice = trim_slice(slice_data[:-trim_bottom])
    elif trim_bottom == 0:
        trimmed_slice = trim_slice(slice_data[trim_top:])
    else:
        trimmed_slice = trim_slice(slice_data[trim_top:-trim_bottom])
    
    # Get brightness stats with width percentage parameters
    brightness, stddev = get_brightness_stats(trimmed_slice, width_start_pct=width_start_pct, width_end_pct=width_end_pct)
    
    # Convert buffer from mm to pixels
    if px_spacing_y <= 0:
        raise ValueError("Pixel spacing must be positive")
        
    px_per = 1/px_spacing_y
    buffer_px = int(buffer * px_per)
    
    # Find indices below minimum brightness
    neg_indices = np.where(brightness < min_brightness)[0]
    
    # Create buffer zones around values below threshold
    buffer_zones = []
    for idx in neg_indices:
        start = max(0, idx - buffer_px)
        end = min(len(brightness), idx + buffer_px + 1)
        buffer_zones.extend(range(start, end))
        
    # Create and apply mask
    mask = np.ones(len(brightness), dtype=bool)
    mask[buffer_zones] = False
    brightness[~mask] = np.nan
    stddev[~mask] = np.nan
    
    return brightness, stddev, trimmed_slice


def find_best_overlap(curve1, curve2, min_overlap=20, max_overlap=450):
    """
    Find optimal overlap between two curves by maximizing correlation and peak matching scores.
    
    This function determines the best overlap position between two brightness curves
    by combining Pearson correlation coefficients with peak/valley matching scores.
    It searches through different overlap lengths and evaluates alignment quality
    based on pattern similarity.
    
    Parameters
    ----------
    curve1 : numpy.ndarray
        First brightness curve (end section will be matched)
    curve2 : numpy.ndarray
        Second brightness curve (beginning section will be matched)
    min_overlap : int, default=20
        Minimum overlap length in pixels to consider
    max_overlap : int, default=450
        Maximum overlap length in pixels to consider
        
    Returns
    -------
    tuple
        Contains the following elements:
        - best_overlap (int): Optimal overlap length in pixels
        - max_score (float): Combined correlation and peak matching score
        
    Raises
    ------
    ValueError
        If curves are too short for minimum overlap or parameters are invalid
        
    Example
    -------
    >>> curve1 = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.1, 200)
    >>> curve2 = np.sin(np.linspace(3*np.pi, 7*np.pi, 200)) + np.random.normal(0, 0.1, 200)
    >>> overlap, score = find_best_overlap(curve1, curve2)
    >>> print(f"Best overlap: {overlap} pixels, Score: {score:.3f}")
    """
    if min_overlap <= 0 or max_overlap <= min_overlap:
        raise ValueError("Invalid overlap parameters: min_overlap must be positive and less than max_overlap")
        
    if len(curve1) < min_overlap or len(curve2) < min_overlap:
        raise ValueError("Curves are too short for minimum overlap requirement")
    
    max_corr = -np.inf
    best_overlap = 0
    
    # Remove NaN values for correlation calculation
    curve1_clean = pd.Series(curve1).fillna(0)
    curve2_clean = pd.Series(curve2).fillna(0)
    
    # Find peaks with prominence to identify major peaks
    curve1_peaks, properties1 = scipy.signal.find_peaks(curve1_clean, prominence=50)
    curve2_peaks, properties2 = scipy.signal.find_peaks(curve2_clean, prominence=50)
    
    # Find valleys (negative peaks)
    curve1_valleys, properties1_valleys = scipy.signal.find_peaks(-curve1_clean, prominence=50)
    curve2_valleys, properties2_valleys = scipy.signal.find_peaks(-curve2_clean, prominence=50)
    
    # Sort peaks by prominence
    curve1_peak_heights = properties1['prominences']
    curve2_peak_heights = properties2['prominences']
    curve1_valley_depths = properties1_valleys['prominences']
    curve2_valley_depths = properties2_valleys['prominences']
    
    # Get first peak/valley and top peaks/valleys
    curve1_first_peak = curve1_peaks[0] if len(curve1_peaks) > 0 else None
    curve2_first_peak = curve2_peaks[0] if len(curve2_peaks) > 0 else None
    curve1_first_valley = curve1_valleys[0] if len(curve1_valleys) > 0 else None
    curve2_first_valley = curve2_valleys[0] if len(curve2_valleys) > 0 else None
    curve1_major_peaks = curve1_peaks[np.argsort(curve1_peak_heights)[-10:]] # Top 10 peaks
    curve2_major_peaks = curve2_peaks[np.argsort(curve2_peak_heights)[-10:]]
    curve1_major_valleys = curve1_valleys[np.argsort(curve1_valley_depths)[-5:]] # Top 5 valleys
    curve2_major_valleys = curve2_valleys[np.argsort(curve2_valley_depths)[-5:]]
    
    # Try different overlap positions
    for overlap in range(min_overlap, min(len(curve1), len(curve2), max_overlap)):
        # Get overlapping sections
        overlap1 = curve1_clean[-overlap:]
        overlap2 = curve2_clean[:overlap]
        
        if len(overlap1) == len(overlap2):
            # Calculate regular correlation
            corr = np.corrcoef(overlap1, overlap2)[0,1]
            
            # Handle NaN correlation
            if np.isnan(corr):
                corr = 0
            
            # Calculate peak matching score with weighted priorities
            peak_score = 0
            
            # Check first peaks in overlap region
            if curve1_first_peak is not None and curve2_first_peak is not None:
                if curve1_first_peak >= len(curve1)-overlap:
                    p1_adjusted = curve1_first_peak - (len(curve1) - overlap)
                    if abs(p1_adjusted - curve2_first_peak) < 5:  # Small tolerance
                        peak_score += 2.0  # Highest weight for first peak match
            
            # Check first valleys in overlap region
            if curve1_first_valley is not None and curve2_first_valley is not None:
                if curve1_first_valley >= len(curve1)-overlap:
                    v1_adjusted = curve1_first_valley - (len(curve1) - overlap)
                    if abs(v1_adjusted - curve2_first_valley) < 5:  # Small tolerance
                        peak_score += 1.2  # High weight for first valley match
            
            # Check major peaks in overlap region
            overlap1_peaks = [p for p in curve1_major_peaks if p >= len(curve1)-overlap]
            overlap2_peaks = [p for p in curve2_major_peaks if p < overlap]
            
            # Add bonus for matching major peaks
            if len(overlap1_peaks) > 0 and len(overlap2_peaks) > 0:
                for p1 in overlap1_peaks:
                    for p2 in overlap2_peaks:
                        p1_adjusted = p1 - (len(curve1) - overlap)
                        if abs(p1_adjusted - p2) < 5:  # Small tolerance
                            peak_score += 0.5  # Weight for major peak matches
                            
            # Check major valleys in overlap region
            overlap1_valleys = [v for v in curve1_major_valleys if v >= len(curve1)-overlap]
            overlap2_valleys = [v for v in curve2_major_valleys if v < overlap]
            
            # Add bonus for matching major valleys
            if len(overlap1_valleys) > 0 and len(overlap2_valleys) > 0:
                for v1 in overlap1_valleys:
                    for v2 in overlap2_valleys:
                        v1_adjusted = v1 - (len(curve1) - overlap)
                        if abs(v1_adjusted - v2) < 3:  # Small tolerance
                            peak_score += 0.3  # Weight for major valley matches
            
            # Combine scores with emphasis on peaks first, then correlation
            total_score = peak_score + corr
            
            if total_score > max_corr:
                max_corr = total_score
                best_overlap = overlap
                
    return best_overlap, max_corr


def stitch_curves(brightness_1, brightness_2, stddev_1, stddev_2, px_spacing_y_1, px_spacing_y_2, 
                 min_overlap=20, max_overlap=450):
    """
    Stitch two overlapping brightness curves together with optimal alignment.
    
    This function combines two brightness curves by finding the optimal overlap
    region, calculating brightness shifts to align the curves, and creating
    averaged values in the overlap region. The result is a seamless stitched
    curve with continuous depth coordinates.
    
    Parameters
    ----------
    brightness_1 : numpy.ndarray
        First brightness curve values
    brightness_2 : numpy.ndarray
        Second brightness curve values  
    stddev_1 : numpy.ndarray
        Standard deviation values for first curve
    stddev_2 : numpy.ndarray
        Standard deviation values for second curve
    px_spacing_y_1 : float
        Pixel spacing in y direction for first curve (mm/pixel)
    px_spacing_y_2 : float
        Pixel spacing in y direction for second curve (mm/pixel)
    min_overlap : int, default=20
        Minimum overlap length in pixels
    max_overlap : int, default=450
        Maximum overlap length in pixels
        
    Returns
    -------
    tuple
        Contains the following elements:
        - final_overlap (int): Actual overlap length used
        - overlap_depth_1 (numpy.ndarray): Depth coordinates for overlap region from curve 1
        - overlap_depth_2 (numpy.ndarray): Depth coordinates for overlap region from curve 2
        - stitched_brightness (numpy.ndarray): Combined brightness values
        - stitched_stddev (numpy.ndarray): Combined standard deviation values
        - stitched_depth (numpy.ndarray): Continuous depth coordinates for stitched curve
        - brightness_2_shifted (numpy.ndarray): Adjusted brightness values for curve 2
        - stddev_2_shifted (numpy.ndarray): Adjusted standard deviation values for curve 2
        
    Example
    -------
    >>> curve1_bright = np.random.rand(200) * 500 + 800
    >>> curve2_bright = np.random.rand(180) * 500 + 750  # Slightly different baseline
    >>> curve1_std = np.random.rand(200) * 50 + 100
    >>> curve2_std = np.random.rand(180) * 50 + 95
    >>> result = stitch_curves(curve1_bright, curve2_bright, curve1_std, curve2_std, 0.1, 0.1)
    >>> overlap, _, _, stitched_bright, _, _, _, _ = result
    >>> print(f"Stitched curve length: {len(stitched_bright)}")
    """
    # Convert to true depth values for overlap finding
    depth_1 = np.arange(len(brightness_1)) * px_spacing_y_1
    depth_2 = np.arange(len(brightness_2)) * px_spacing_y_2
    
    # Find best overlap position using only brightness
    final_overlap, brightness_corr = find_best_overlap(brightness_1, brightness_2, min_overlap=min_overlap, max_overlap=max_overlap)
    
    print(f"Brightness overlap: {final_overlap} pixels (correlation: {brightness_corr:.3f})")
    
    # Get overlapping sections in true depth
    overlap_depth_1 = np.arange(len(brightness_1)-final_overlap, len(brightness_1)) * px_spacing_y_1
    overlap_depth_2 = np.arange(final_overlap) * px_spacing_y_2  # Use curve 2's spacing
    
    # Calculate shift values based on the overlap region
    brightness_shift = np.nanmean(brightness_1[-final_overlap:] - brightness_2[:final_overlap])  #avoid NaN
    stddev_shift = np.nanmean(stddev_1[-final_overlap:] - stddev_2[:final_overlap])              #avoid NaN
    
    # Shift the entire second dataset
    brightness_2_shifted = brightness_2 + brightness_shift
    stddev_2_shifted = stddev_2 + stddev_shift
    
    print(f"Applied brightness shift: {brightness_shift:.3f}")
    print(f"Applied stddev shift: {stddev_shift:.3f}")
    
    # Create averaged values in overlap region using shifted brightness and stddev
    # Create masks for invalid values (NaN or zero)
    mask1_brightness = ~(np.isnan(brightness_1[-final_overlap:]) | (brightness_1[-final_overlap:] == 0))
    mask2_brightness = ~(np.isnan(brightness_2_shifted[:final_overlap]) | (brightness_2_shifted[:final_overlap] == 0))
    mask1_stddev = ~(np.isnan(stddev_1[-final_overlap:]) | (stddev_1[-final_overlap:] == 0))
    mask2_stddev = ~(np.isnan(stddev_2_shifted[:final_overlap]) | (stddev_2_shifted[:final_overlap] == 0))
    
    # Only average where both values are valid
    overlap_brightness = np.full(final_overlap, np.nan)
    overlap_stddev = np.full(final_overlap, np.nan)
    valid_mask_brightness = mask1_brightness & mask2_brightness
    valid_mask_stddev = mask1_stddev & mask2_stddev
    
    overlap_brightness[valid_mask_brightness] = (brightness_1[-final_overlap:][valid_mask_brightness] + 
                                               brightness_2_shifted[:final_overlap][valid_mask_brightness]) / 2
    overlap_stddev[valid_mask_stddev] = (stddev_1[-final_overlap:][valid_mask_stddev] + 
                                        stddev_2_shifted[:final_overlap][valid_mask_stddev]) / 2
    
    # Stitch the curves using the final overlap position with averaged overlap region
    stitched_brightness = np.concatenate([
        brightness_1[:-final_overlap],  # Non-overlapped part of curve 1
        overlap_brightness,             # Averaged overlap region
        brightness_2_shifted[final_overlap:]  # Non-overlapped part of curve 2
    ])
    stitched_stddev = np.concatenate([
        stddev_1[:-final_overlap],
        overlap_stddev,
        stddev_2_shifted[final_overlap:]
    ])
    
    # Create depth array for the stitched data using appropriate spacing for each section
    depth_before_overlap = np.arange(len(brightness_1)-final_overlap) * px_spacing_y_1
    depth_overlap = overlap_depth_1  # Use spacing from curve 1 for overlap region
    depth_after_overlap = (np.arange(len(brightness_2)-final_overlap) * px_spacing_y_2) + overlap_depth_1[-1]
    stitched_depth = np.concatenate([depth_before_overlap, depth_overlap, depth_after_overlap])
    
    return (final_overlap, overlap_depth_1, overlap_depth_2, 
            stitched_brightness, stitched_stddev, stitched_depth, 
            brightness_2_shifted, stddev_2_shifted)


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


def create_stitched_slice(trimmed_slice_1, trimmed_slice_2, final_overlap, px_spacing_x_1, px_spacing_y_1, 
                         px_spacing_x_2, px_spacing_y_2):
    """
    Create a stitched CT slice from two overlapping slices.
    
    This function combines two trimmed CT slices by vertically stacking them
    with appropriate overlap handling. It resizes slices to have matching
    widths while maintaining aspect ratios and calculates averaged pixel
    spacing for the combined slice.
    
    Parameters
    ----------
    trimmed_slice_1 : numpy.ndarray
        First CT slice to stitch
    trimmed_slice_2 : numpy.ndarray 
        Second CT slice to stitch
    final_overlap : int
        Number of overlapping pixels between slices
    px_spacing_x_1 : float
        Pixel spacing in x direction for first slice (mm/pixel)
    px_spacing_y_1 : float
        Pixel spacing in y direction for first slice (mm/pixel)
    px_spacing_x_2 : float
        Pixel spacing in x direction for second slice (mm/pixel)
    px_spacing_y_2 : float
        Pixel spacing in y direction for second slice (mm/pixel)
        
    Returns
    -------
    tuple
        Contains the following elements:
        - stitched_slice (numpy.ndarray): The combined CT slice
        - pixel_spacing (tuple): (x_spacing, y_spacing) for the stitched slice
        
    Raises
    ------
    ValueError
        If slices cannot be resized or overlap exceeds slice dimensions
        
    Example
    -------
    >>> slice1 = np.random.rand(500, 200) * 1000 + 400
    >>> slice2 = np.random.rand(450, 180) * 1000 + 400  
    >>> stitched, spacing = create_stitched_slice(slice1, slice2, 50, 0.1, 0.1, 0.1, 0.1)
    >>> print(f"Stitched slice shape: {stitched.shape}")
    >>> print(f"Pixel spacing: {spacing}")
    """
    # Get dimensions of both slices
    height1, width1 = trimmed_slice_1.shape
    height2, width2 = trimmed_slice_2.shape

    if final_overlap >= min(height1, height2):
        raise ValueError("Overlap exceeds minimum slice height")

    # Use the smaller width for both slices
    min_width = min(width1, width2)

    # Calculate new heights maintaining aspect ratio
    if width1 > width2:
        new_height1 = int(height1 * (min_width / width1))
        trimmed_slice_1 = cv2.resize(trimmed_slice_1, (min_width, new_height1))
    elif width2 > width1:
        new_height2 = int(height2 * (min_width / width2))
        trimmed_slice_2 = cv2.resize(trimmed_slice_2, (min_width, new_height2))

    stitched_slice = np.vstack([
        trimmed_slice_1,
        trimmed_slice_2[final_overlap:]
    ])

    # Calculate pixel spacing for stitched image - use average of both slices
    stitched_px_spacing = ((px_spacing_x_1 + px_spacing_x_2)/2, 
                          (px_spacing_y_1 + px_spacing_y_2)/2)
    
    return stitched_slice, stitched_px_spacing


def process_single_scan(data_dir, params, segment, scan_name, width_start_pct=0.25, width_end_pct=0.75, 
                       max_value_side_trim=1200, vmin=None, vmax=None):
    """
    Process a single CT scan and return the processed data.
    
    This function loads DICOM files from a directory, extracts a central slice,
    trims the slice to remove non-core regions, and processes the brightness
    data according to the specified parameters. It also displays the results
    using the standard visualization format.
    
    Parameters
    ----------
    data_dir : str
        Directory path containing DICOM files for the scan
    params : dict
        Dictionary containing processing parameters with keys:
        - 'trim_top': pixels to trim from top
        - 'trim_bottom': pixels to trim from bottom  
        - 'min_brightness': minimum brightness threshold
        - 'buffer': buffer size in mm around masked regions
    segment : str
        Name identifier for the core segment
    scan_name : str
        Name identifier for the specific scan
    width_start_pct : float, default=0.25
        Starting percentage of width for brightness analysis (0.0 to 1.0)
    width_end_pct : float, default=0.75
        Ending percentage of width for brightness analysis (0.0 to 1.0)
    max_value_side_trim : float, default=1200
        Threshold value for automatic side trimming of non-core regions
    vmin : float, optional
        Minimum value for display colormap. If None, uses default scaling
    vmax : float, optional
        Maximum value for display colormap. If None, uses default scaling
        
    Returns
    -------
    tuple
        Contains the following elements:
        - brightness (numpy.ndarray): Processed brightness values
        - stddev (numpy.ndarray): Standard deviation values
        - trimmed_slice (numpy.ndarray): Trimmed slice data
        - px_spacing_x (float): Pixel spacing in x direction
        - px_spacing_y (float): Pixel spacing in y direction
        
    Example
    -------
    >>> params = {'trim_top': 50, 'trim_bottom': 20, 'min_brightness': 400, 'buffer': 5}
    >>> brightness, stddev, slice_data, px_x, px_y = process_single_scan(
    ...     '/path/to/dicom', params, 'Core-1', 'SE000000'
    ... )
    """
    volume_data, px_spacing_x, px_spacing_y, slice_thickness = load_dicom_files(data_dir)
    slice_data = get_slice(volume_data, index=int(len(volume_data[0,0,:])*1/2), axis=2)
    
    # Trim left and right columns where all values are < max_value_side_trim (outside of the cores)
    left_idx = 0
    right_idx = slice_data.shape[1]
    
    for col in range(slice_data.shape[1]):
        if (slice_data[:,col] >= max_value_side_trim).any():
            left_idx = col
            break
            
    for col in range(slice_data.shape[1]-1, -1, -1):
        if (slice_data[:,col] >= max_value_side_trim).any():
            right_idx = col + 1
            break
    
    slice_data = slice_data[:, left_idx:right_idx]
    
    brightness, stddev, trimmed_slice = process_brightness_data(slice_data,
                                          px_spacing_y,
                                          trim_top=params['trim_top'],
                                          trim_bottom=params['trim_bottom'],
                                          min_brightness=params['min_brightness'],
                                          buffer=params['buffer'],
                                          width_start_pct=width_start_pct,
                                          width_end_pct=width_end_pct)
    
    display_slice_bt_std(trimmed_slice, brightness, stddev,
                        pixel_spacing=(px_spacing_x, px_spacing_y),
                        core_name=f"{segment} ({scan_name})",
                        vmin=vmin, vmax=vmax)
    
    return brightness, stddev, trimmed_slice, px_spacing_x, px_spacing_y


def process_two_scans(segment_data, segment, mother_dir, width_start_pct=0.25, width_end_pct=0.75, 
                     max_value_side_trim=1200, min_overlap=20, max_overlap=450, vmin=None, vmax=None):
    """
    Process and stitch two CT scans together.
    
    This function processes two separate CT scans for a single core segment,
    finds the optimal overlap between them, and creates a stitched result.
    It handles the complete workflow from individual scan processing through
    curve stitching and final visualization.
    
    Parameters
    ----------
    segment_data : dict
        Dictionary containing segment information with keys:
        - 'scans': list of scan names
        - 'params': dictionary of parameters for each scan
    segment : str
        Name identifier for the core segment
    mother_dir : str
        Base directory path containing all scan subdirectories
    width_start_pct : float, default=0.25
        Starting percentage of width for brightness analysis
    width_end_pct : float, default=0.75
        Ending percentage of width for brightness analysis
    max_value_side_trim : float, default=1200
        Threshold for automatic side trimming
    min_overlap : int, default=20
        Minimum overlap length for stitching
    max_overlap : int, default=450
        Maximum overlap length for stitching
    vmin : float, optional
        Minimum value for display colormap. If None, uses default scaling
    vmax : float, optional
        Maximum value for display colormap. If None, uses default scaling
        
    Returns
    -------
    tuple
        Contains the following elements:
        - st_bright_re (numpy.ndarray): Recalculated stitched brightness values
        - st_std_re (numpy.ndarray): Recalculated stitched standard deviation values
        - st_depth_re (numpy.ndarray): Depth coordinates for stitched data
        - st_slice (numpy.ndarray): Stitched slice data
        - pixel_spacing (tuple): Pixel spacing for stitched data
        
    Example
    -------
    >>> segment_data = {
    ...     'scans': ['SE000000', 'SE000002'],
    ...     'params': {
    ...         'SE000000': {'trim_top': 50, 'trim_bottom': 10, 'min_brightness': 400, 'buffer': 5},
    ...         'SE000002': {'trim_top': 20, 'trim_bottom': 50, 'min_brightness': 400, 'buffer': 5}
    ...     }
    ... }
    >>> result = process_two_scans(segment_data, 'Core-1', '/path/to/data')
    """
    scans = segment_data['scans']
    data_dir_1 = f"{mother_dir}/{segment}/{scans[0]}"
    data_dir_2 = f"{mother_dir}/{segment}/{scans[1]}"
    
    # Process first scan
    bright1, std1, trimmed_slice1, px_spacing_x1, px_spacing_y1 = process_single_scan(
        data_dir_1, 
        segment_data['params'][scans[0]], 
        segment, 
        scans[0],
        width_start_pct=width_start_pct,
        width_end_pct=width_end_pct,
        max_value_side_trim=max_value_side_trim,
        vmin=vmin, vmax=vmax
    )
    
    # Process second scan
    bright2, std2, trimmed_slice2, px_spacing_x2, px_spacing_y2 = process_single_scan(
        data_dir_2,
        segment_data['params'][scans[1]],
        segment,
        scans[1],
        width_start_pct=width_start_pct,
        width_end_pct=width_end_pct,
        max_value_side_trim=max_value_side_trim,
        vmin=vmin, vmax=vmax
    )
    
    # Stitch scans
    final_overlap, od1, od2, st_bright, st_std, st_depth, bright2_shifted, std2_shifted = stitch_curves(
        bright1, bright2, std1, std2, px_spacing_y1, px_spacing_y2, min_overlap=min_overlap, max_overlap=max_overlap
    )
    
    # Plot and display results
    plot_stitched_curves(st_depth, st_bright, st_std,
                        bright1, bright2, std1, std2,
                        bright2_shifted, std2_shifted,
                        final_overlap, od1, od2, px_spacing_y1, px_spacing_y2)

    # Create stitched slice
    st_slice, pixel_spacing = create_stitched_slice(
        trimmed_slice1, trimmed_slice2, final_overlap,
        px_spacing_x1, px_spacing_y1, px_spacing_x2, px_spacing_y2
    )

    # Recalculate brightness and std dev on stitched slice without trimming
    st_bright_re, st_std_re, st_slice = process_brightness_data(st_slice, 
                                                               pixel_spacing[1],
                                                               trim_top=0,
                                                               trim_bottom=0,
                                                               min_brightness=400,
                                                               buffer=5,
                                                               width_start_pct=width_start_pct,
                                                               width_end_pct=width_end_pct)
    
    # Get depth array from length of recalculated brightness
    st_depth_re = np.arange(len(st_bright_re))
    
    # Display stitched slice with recalculated brightness/std
    display_slice_bt_std(st_slice, st_bright_re, st_std_re,
                        pixel_spacing=pixel_spacing,
                        core_name=f"{segment} (stitched)",
                        vmin=vmin, vmax=vmax)
    
    return st_bright_re, st_std_re, st_depth_re, st_slice, pixel_spacing


def process_and_stitch_segments(core_structure, mother_dir, width_start_pct=0.25, width_end_pct=0.75, 
                               max_value_side_trim=1200, min_overlap=20, max_overlap=450, vmin=None, vmax=None):
    """
    Process and stitch all segments of a core according to the specified structure.
    
    This function orchestrates the complete processing workflow for a multi-segment
    core. It handles single scans, two-scan segments, and multi-section segments
    (with A, B, C suffixes), then combines all segments into a final stitched core.
    Each segment is rescaled to match RGB image dimensions and can be inverted if needed.
    
    Multiple 'empty' segments are automatically numbered as 'empty_1', 'empty_2', etc.,
    while preserving their order in the core structure.
    
    Parameters
    ----------
    core_structure : dict or list
        Core structure definition in one of two formats:
        1. Dictionary format: {segment_name: segment_data, ...}
        2. List format: [(segment_name, segment_data), ...]
        
        Each segment_data contains:
        - 'scans': list of scan names
        - 'params': processing parameters for each scan
        - 'rgb_pxlength': target pixel length for rescaling
        - 'rgb_pxwidth': target pixel width for rescaling  
        - 'upside_down': boolean indicating if segment should be inverted
        - 'suffixes': optional list of section suffixes (A, B, C)
        
        Note: Multiple 'empty' segments will be automatically numbered as 'empty_1', 'empty_2', etc.
        For dictionaries, use unique keys like 'empty_1', 'empty_2' or use list format for duplicates.
    mother_dir : str
        Base directory path containing all segment subdirectories
    width_start_pct : float, default=0.25
        Starting percentage of width for brightness analysis
    width_end_pct : float, default=0.75
        Ending percentage of width for brightness analysis
    max_value_side_trim : float, default=1200
        Threshold for automatic side trimming
    min_overlap : int, default=20
        Minimum overlap length for stitching
    max_overlap : int, default=450
        Maximum overlap length for stitching
    vmin : float, optional
        Minimum value for display colormap. If None, uses default scaling
    vmax : float, optional
        Maximum value for display colormap. If None, uses default scaling
        
    Returns
    -------
    tuple
        Contains the following elements:
        - final_stitched_slice (numpy.ndarray): Complete stitched CT slice
        - final_stitched_brightness (numpy.ndarray): Complete brightness profile
        - final_stitched_stddev (numpy.ndarray): Complete standard deviation profile
        - final_stitched_depth (numpy.ndarray): Depth coordinates in pixels
        - px_spacing_x (float): Final pixel spacing in x direction (always 1.0)
        - px_spacing_y (float): Final pixel spacing in y direction (always 1.0)
        
    Example
    -------
    >>> core_structure = {
    ...     'Core-1': {
    ...         'scans': ['SE000000'],
    ...         'params': {'SE000000': {'trim_top': 50, 'trim_bottom': 20, 'min_brightness': 400, 'buffer': 5}},
    ...         'rgb_pxlength': 1000, 'rgb_pxwidth': 200, 'upside_down': False
    ...     }
    ... }
    >>> result = process_and_stitch_segments(core_structure, '/path/to/data')
    """
    # Process core structure in order and handle multiple 'empty' segments
    # Handle both dictionary and list formats
    if isinstance(core_structure, dict):
        # Dictionary format - convert to list of tuples
        processed_segments = []
        empty_counter = 1
        
        for segment, segment_data in core_structure.items():
            if segment == 'empty':
                # Automatically number empty segments
                if empty_counter == 1:
                    processed_segment_name = 'empty_1'
                else:
                    processed_segment_name = f'empty_{empty_counter}'
                empty_counter += 1
                processed_segments.append((processed_segment_name, segment_data))
            else:
                processed_segments.append((segment, segment_data))
    else:
        # List format - already in correct format, just number empty segments
        processed_segments = []
        empty_counter = 1
        
        for segment, segment_data in core_structure:
            if segment == 'empty':
                # Automatically number empty segments
                if empty_counter == 1:
                    processed_segment_name = 'empty_1'
                else:
                    processed_segment_name = f'empty_{empty_counter}'
                empty_counter += 1
                processed_segments.append((processed_segment_name, segment_data))
            else:
                processed_segments.append((segment, segment_data))
    
    # Dictionary to store stitched results for each core segment
    stitched_segments = {}
    
    # Process each segment
    for segment, segment_data in processed_segments:
        print(f"Processing {segment}")
        
        # Check if this is an empty segment
        if segment_data.get('scans') is None and 'rgb_pxlength' in segment_data and 'rgb_pxwidth' in segment_data:
            print(f"Creating empty segment for {segment}")
            
            # Get target dimensions
            target_height = segment_data['rgb_pxlength']
            target_width = segment_data['rgb_pxwidth']
            
            # Create empty slice (filled with zeros or a background value)
            empty_slice = np.zeros((target_height, target_width), dtype=np.float64)
            
            # Create empty brightness and stddev curves with NaN values
            empty_brightness = np.full(target_height, np.nan)
            empty_stddev = np.full(target_height, np.nan)
            
            stitched_segments[segment] = {
                'brightness': empty_brightness,
                'stddev': empty_stddev,
                'depth': np.arange(target_height),  # Use pixel units
                'slice': empty_slice,
                'px_spacing': (1, 1)  # Use pixel units
            }
            
            # Display empty slice
            display_slice_bt_std(empty_slice, empty_brightness, empty_stddev,
                               pixel_spacing=(1, 1),
                               core_name=f"{segment} (empty segment - {target_height} x {target_width}px)",
                               vmin=vmin, vmax=vmax)
            
            continue
        
        if 'suffixes' not in segment_data:  # Regular segments with one or two scans
            if len(segment_data['scans']) == 1:  # Single scan
                data_dir = f"{mother_dir}/{segment}/{segment_data['scans'][0]}"
                brightness, stddev, trimmed_slice, px_spacing_x, px_spacing_y = process_single_scan(
                    data_dir,
                    segment_data['params'][segment_data['scans'][0]],
                    segment,
                    segment_data['scans'][0],
                    width_start_pct=width_start_pct,
                    width_end_pct=width_end_pct,
                    max_value_side_trim=max_value_side_trim,
                    vmin=vmin, vmax=vmax
                )
                
                # Rescale slice to match rgb dimensions
                target_height = segment_data['rgb_pxlength']
                target_width = segment_data['rgb_pxwidth']
                
                rescaled_slice = cv2.resize(trimmed_slice, (target_width, target_height))
                
                # Interpolate brightness and stddev to match new height
                new_bright = np.interp(np.linspace(0, len(brightness)-1, target_height),
                                     np.arange(len(brightness)), brightness)
                new_std = np.interp(np.linspace(0, len(stddev)-1, target_height),
                                  np.arange(len(stddev)), stddev)
                
                stitched_segments[segment] = {
                    'brightness': new_bright,
                    'stddev': new_std,
                    'depth': np.arange(target_height),  # Use pixel units
                    'slice': rescaled_slice,
                    'px_spacing': (1, 1)  # Use pixel units
                }
                
                # If upside_down is True, rotate slice and reverse data arrays
                if segment_data.get('upside_down', False):
                    stitched_segments[segment]['slice'] = cv2.rotate(rescaled_slice, cv2.ROTATE_180)
                    stitched_segments[segment]['brightness'] = np.flip(new_bright)
                    stitched_segments[segment]['stddev'] = np.flip(new_std)
                
                # Display rescaled slice
                core_name = f"[UPSIDE DOWN] {segment} (rescaled to {target_height} x {target_width}px)" if segment_data.get('upside_down', False) else f"{segment} (rescaled to {target_height} x {target_width}px)"
                display_slice_bt_std(stitched_segments[segment]['slice'], stitched_segments[segment]['brightness'], stitched_segments[segment]['stddev'],
                                   pixel_spacing=(1, 1),
                                   core_name=core_name,
                                   vmin=vmin, vmax=vmax)
                
            else:  # Two scans
                st_bright, st_std, st_depth, st_slice, pixel_spacing = process_two_scans(
                    segment_data, segment, mother_dir, width_start_pct=width_start_pct, width_end_pct=width_end_pct, max_value_side_trim=max_value_side_trim, min_overlap=min_overlap, max_overlap=max_overlap, vmin=vmin, vmax=vmax
                )
                
                # Rescale slice to match rgb dimensions
                target_height = segment_data['rgb_pxlength']
                target_width = segment_data['rgb_pxwidth']
                
                rescaled_slice = cv2.resize(st_slice, (target_width, target_height))
                
                # Interpolate brightness and stddev to match new height
                new_bright = np.interp(np.linspace(0, len(st_bright)-1, target_height), 
                                     np.arange(len(st_bright)), st_bright)
                new_std = np.interp(np.linspace(0, len(st_std)-1, target_height),
                                  np.arange(len(st_std)), st_std)
                
                stitched_segments[segment] = {
                    'brightness': new_bright,
                    'stddev': new_std,
                    'depth': np.arange(target_height),  # Use pixel units
                    'slice': rescaled_slice,
                    'px_spacing': (1, 1)  # Use pixel units
                }
                
                # If upside_down is True, rotate slice and reverse data arrays
                if segment_data.get('upside_down', False):
                    stitched_segments[segment]['slice'] = cv2.rotate(rescaled_slice, cv2.ROTATE_180)
                    stitched_segments[segment]['brightness'] = np.flip(new_bright)
                    stitched_segments[segment]['stddev'] = np.flip(new_std)
                
                # Display rescaled slice
                core_name = f"[UPSIDE DOWN] {segment} (rescaled to {target_height} x {target_width}px)" if segment_data.get('upside_down', False) else f"{segment} (rescaled to {target_height} x {target_width}px)"
                display_slice_bt_std(stitched_segments[segment]['slice'], stitched_segments[segment]['brightness'], stitched_segments[segment]['stddev'],
                                   pixel_spacing=(1, 1),
                                   core_name=core_name,
                                   vmin=vmin, vmax=vmax)
                
        else:  # Segments with suffixes (A, B, C or A, B)
            print(f"Processing {segment}: Stitching sections {', '.join(segment_data['suffixes'])}")
            brightness_list = []
            stddev_list = []
            slice_list = []
            
            # Process each section
            for suffix in segment_data['suffixes']:
                folder_name = f"{segment}{suffix}/{segment_data['scans'][0]}"
                data_dir = f"{mother_dir}/{folder_name}"
                param_key = f"{suffix}/{segment_data['scans'][0]}"
                
                brightness, stddev, trimmed_slice, px_spacing_x, px_spacing_y = process_single_scan(
                    data_dir,
                    segment_data['params'][param_key],
                    f"{segment}{suffix}",
                    segment_data['scans'][0],
                    width_start_pct=width_start_pct,
                    width_end_pct=width_end_pct,
                    max_value_side_trim=max_value_side_trim,
                    vmin=vmin, vmax=vmax
                )
                
                brightness_list.append(brightness)
                stddev_list.append(stddev)
                slice_list.append(trimmed_slice)
            
            if len(segment_data['suffixes']) == 1:  # Only A
                # Use data from single section without stitching
                st_bright = brightness_list[0]
                st_std = stddev_list[0]
                st_slice = slice_list[0]
                
                # Rescale slice to match rgb dimensions
                target_height = segment_data['rgb_pxlength']
                target_width = segment_data['rgb_pxwidth']
                
                rescaled_slice = cv2.resize(st_slice, (target_width, target_height))
                
                # Interpolate brightness and stddev to match new height
                new_bright = np.interp(np.linspace(0, len(st_bright)-1, target_height),
                                     np.arange(len(st_bright)), st_bright)
                new_std = np.interp(np.linspace(0, len(st_std)-1, target_height),
                                  np.arange(len(st_std)), st_std)
                
                stitched_segments[segment] = {
                    'brightness': new_bright,
                    'stddev': new_std,
                    'depth': np.arange(target_height),  # Use pixel units
                    'slice': rescaled_slice,
                    'px_spacing': (1, 1)  # Use pixel units
                }
                
                # If upside_down is True, rotate slice and reverse data arrays
                if segment_data.get('upside_down', False):
                    stitched_segments[segment]['slice'] = cv2.rotate(rescaled_slice, cv2.ROTATE_180)
                    stitched_segments[segment]['brightness'] = np.flip(new_bright)
                    stitched_segments[segment]['stddev'] = np.flip(new_std)
                
                # Display rescaled slice
                core_name = f"[UPSIDE DOWN] {segment} (rescaled to {target_height} x {target_width}px)" if segment_data.get('upside_down', False) else f"{segment} (rescaled to {target_height} x {target_width}px)"
                display_slice_bt_std(stitched_segments[segment]['slice'], stitched_segments[segment]['brightness'], stitched_segments[segment]['stddev'],
                                   pixel_spacing=(1, 1),
                                   core_name=core_name,
                                   vmin=vmin, vmax=vmax)
                
            elif len(segment_data['suffixes']) == 2:  # A and B
                # Stitch A and B
                final_overlap_ab, od1_ab, od2_ab, st_bright_ab, st_std_ab, st_depth_ab, bright2_shifted_ab, std2_shifted_ab = stitch_curves(
                    brightness_list[0], brightness_list[1], 
                    stddev_list[0], stddev_list[1],
                    px_spacing_y, px_spacing_y,  # Add second px_spacing_y parameter
                    min_overlap=min_overlap, max_overlap=max_overlap
                )
                
                st_slice_ab, pixel_spacing_ab = create_stitched_slice(
                    slice_list[0], slice_list[1],
                    final_overlap_ab, px_spacing_x, px_spacing_y, px_spacing_x, px_spacing_y
                )
                plot_stitched_curves(st_depth_ab, st_bright_ab, st_std_ab,
                                   brightness_list[0], brightness_list[1],
                                   stddev_list[0], stddev_list[1],
                                   bright2_shifted_ab, std2_shifted_ab,
                                   final_overlap_ab, od1_ab, od2_ab, px_spacing_y, px_spacing_y)
                
                # Rescale slice to match rgb dimensions
                target_height = segment_data['rgb_pxlength']
                target_width = segment_data['rgb_pxwidth']
                
                rescaled_slice = cv2.resize(st_slice_ab, (target_width, target_height))
                
                # Interpolate brightness and stddev to match new height
                new_bright = np.interp(np.linspace(0, len(st_bright_ab)-1, target_height),
                                     np.arange(len(st_bright_ab)), st_bright_ab)
                new_std = np.interp(np.linspace(0, len(st_std_ab)-1, target_height),
                                  np.arange(len(st_std_ab)), st_std_ab)
                
                stitched_segments[segment] = {
                    'brightness': new_bright,
                    'stddev': new_std,
                    'depth': np.arange(target_height),  # Use pixel units
                    'slice': rescaled_slice,
                    'px_spacing': (1, 1)  # Use pixel units
                }
                
                # If upside_down is True, rotate slice and reverse data arrays
                if segment_data.get('upside_down', False):
                    stitched_segments[segment]['slice'] = cv2.rotate(rescaled_slice, cv2.ROTATE_180)
                    stitched_segments[segment]['brightness'] = np.flip(new_bright)
                    stitched_segments[segment]['stddev'] = np.flip(new_std)
                
                # Display rescaled slice
                core_name = f"[UPSIDE DOWN] {segment} (rescaled to {target_height} x {target_width}px)" if segment_data.get('upside_down', False) else f"{segment} (rescaled to {target_height} x {target_width}px)"
                display_slice_bt_std(stitched_segments[segment]['slice'], stitched_segments[segment]['brightness'], stitched_segments[segment]['stddev'],
                                   pixel_spacing=(1, 1),
                                   core_name=core_name,
                                   vmin=vmin, vmax=vmax)
                
            else:  # A, B and C
                # Stitch A and B first
                final_overlap_ab, od1_ab, od2_ab, st_bright_ab, st_std_ab, st_depth_ab, bright2_shifted_ab, std2_shifted_ab = stitch_curves(
                    brightness_list[0], brightness_list[1], 
                    stddev_list[0], stddev_list[1],
                    px_spacing_y, px_spacing_y,  # Add second px_spacing_y parameter
                    min_overlap=min_overlap, max_overlap=max_overlap
                )
                
                st_slice_ab, pixel_spacing_ab = create_stitched_slice(
                    slice_list[0], slice_list[1],
                    final_overlap_ab, px_spacing_x, px_spacing_y, px_spacing_x, px_spacing_y
                )
                plot_stitched_curves(st_depth_ab, st_bright_ab, st_std_ab,
                                   brightness_list[0], brightness_list[1],
                                   stddev_list[0], stddev_list[1],
                                   bright2_shifted_ab, std2_shifted_ab,
                                   final_overlap_ab, od1_ab, od2_ab, px_spacing_y, px_spacing_y)
                
                # Stitch AB with C
                final_overlap_abc, od1_abc, od2_abc, st_bright, st_std, st_depth, bright2_shifted_abc, std2_shifted_abc = stitch_curves(
                    st_bright_ab, brightness_list[2],
                    st_std_ab, stddev_list[2],
                    px_spacing_y, px_spacing_y,  # Add second px_spacing_y parameter
                    min_overlap=min_overlap, max_overlap=max_overlap
                )
                
                st_slice, pixel_spacing = create_stitched_slice(
                    st_slice_ab, slice_list[2],
                    final_overlap_abc, px_spacing_x, px_spacing_y, px_spacing_x, px_spacing_y
                )
                
                # Rescale slice to match rgb dimensions
                target_height = segment_data['rgb_pxlength']
                target_width = segment_data['rgb_pxwidth']
                
                rescaled_slice = cv2.resize(st_slice, (target_width, target_height))
                
                # Interpolate brightness and stddev to match new height
                new_bright = np.interp(np.linspace(0, len(st_bright)-1, target_height),
                                     np.arange(len(st_bright)), st_bright)
                new_std = np.interp(np.linspace(0, len(st_std)-1, target_height),
                                  np.arange(len(st_std)), st_std)
                
                stitched_segments[segment] = {
                    'brightness': new_bright,
                    'stddev': new_std,
                    'depth': np.arange(target_height),  # Use pixel units
                    'slice': rescaled_slice,
                    'px_spacing': (1, 1)  # Use pixel units
                }
                
                plot_stitched_curves(st_depth, st_bright, st_std,
                                   st_bright_ab, brightness_list[2],
                                   st_std_ab, stddev_list[2],
                                   bright2_shifted_abc, std2_shifted_abc,
                                   final_overlap_abc, od1_abc, od2_abc, px_spacing_y, px_spacing_y)
                
                # If upside_down is True, rotate slice and reverse data arrays
                if segment_data.get('upside_down', False):
                    stitched_segments[segment]['slice'] = cv2.rotate(rescaled_slice, cv2.ROTATE_180)
                    stitched_segments[segment]['brightness'] = np.flip(new_bright)
                    stitched_segments[segment]['stddev'] = np.flip(new_std)
                
                # Display rescaled slice
                core_name = f"[UPSIDE DOWN] {segment} (rescaled to {target_height} x {target_width}px)" if segment_data.get('upside_down', False) else f"{segment} (rescaled to {target_height} x {target_width}px)"
                display_slice_bt_std(stitched_segments[segment]['slice'], stitched_segments[segment]['brightness'], stitched_segments[segment]['stddev'],
                                   pixel_spacing=(1, 1),
                                   core_name=core_name,
                                   vmin=vmin, vmax=vmax)
    
    # Stack all segments together (from top to bottom)
    segment_order = [segment for segment, _ in processed_segments]
    print(f"Stitching all core segments together: {', '.join(segment_order)}")
    final_slices = []
    final_brightness = []
    final_stddev = []
    cumulative_depth = 0
    
    # Find minimum width among all segments
    min_width = min([stitched_segments[seg]['slice'].shape[1] for seg in segment_order])

    for segment in segment_order:
        data = stitched_segments[segment]
        slice_data = data['slice']
        
        # Calculate scale factors
        current_width = slice_data.shape[1]
        width_scale = min_width / current_width
        
        # Rescale slice while preserving aspect ratio
        new_height = int(slice_data.shape[0] * width_scale)
        slice_data = cv2.resize(slice_data, (min_width, new_height))
        final_slices.append(slice_data)
        
        # Interpolate brightness and stddev to match new height
        old_indices = np.arange(len(data['brightness']))
        new_indices = np.linspace(0, len(data['brightness'])-1, new_height)
        
        new_brightness = np.interp(new_indices, old_indices, data['brightness'])
        new_stddev = np.interp(new_indices, old_indices, data['stddev'])
        
        # Adjust depths to be continuous
        depths = np.arange(new_height) + cumulative_depth
        cumulative_depth = depths[-1]
        
        final_brightness.extend(new_brightness)
        final_stddev.extend(new_stddev)

    # Create final stitched arrays
    final_stitched_slice = np.vstack(final_slices)
    final_stitched_brightness = np.array(final_brightness)
    final_stitched_stddev = np.array(final_stddev)
    final_stitched_depth = np.arange(len(final_stitched_brightness))  # Use pixel units
    
    return final_stitched_slice, final_stitched_brightness, final_stitched_stddev, final_stitched_depth, 1, 1  # Use pixel units 