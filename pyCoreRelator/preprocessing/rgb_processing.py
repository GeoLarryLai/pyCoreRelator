"""
RGB image processing functions for pyCoreRelator.

Included Functions:
- trim_image: Trim specified number of pixels from top and bottom of image array
- extract_rgb_profile: Extract RGB color profiles along the y-axis of an image file
- rgb_process_and_stitch: Stitch multiple core sections together by processing RGB profiles
  (supports multiple empty segments with automatic numbering) with optional CSV export

This module provides comprehensive tools for processing RGB images of geological cores,
extracting color profiles, and creating visualizations for core analysis.
"""

from PIL import Image
from .rgb_plotting import plot_rgbimg_curves
import numpy as np
# matplotlib not needed for computation functions
import os
import pandas as pd


def trim_image(img_array, top_trim=0, bottom_trim=0):
    """
    Trim specified number of pixels from top and bottom of image array.
    
    This function removes a specified number of pixels from the top and bottom edges
    of an image array, which is useful for removing unwanted borders or artifacts
    from core images.
    
    Parameters
    ----------
    img_array : numpy.ndarray
        Input image array with shape (height, width, channels)
    top_trim : int, default=0
        Number of pixels to trim from the top of the image
    bottom_trim : int, default=0
        Number of pixels to trim from the bottom of the image
        
    Returns
    -------
    numpy.ndarray
        Trimmed image array with reduced height
        
    Raises
    ------
    ValueError
        If total trim amount exceeds image height
        
    Example
    -------
    >>> img = np.random.rand(100, 50, 3)
    >>> trimmed = trim_image(img, top_trim=10, bottom_trim=5)
    >>> trimmed.shape
    (85, 50, 3)
    """
    if top_trim + bottom_trim >= img_array.shape[0]:
        raise ValueError("Total trim amount exceeds image height")
        
    return img_array[top_trim:img_array.shape[0]-bottom_trim]



def extract_rgb_profile(image_path, upper_rgb_threshold=100, lower_rgb_threshold=0, buffer=20, 
                       top_trim=0, bottom_trim=0, target_luminance=130, bin_size=10, 
                       width_start_pct=0.25, width_end_pct=0.75):
    """
    Extract RGB color profiles along the y-axis of an image file.
    
    This function processes a core image to extract RGB color values along the depth
    (y-axis). It analyzes the center strip of the image, filters out extreme values,
    calculates statistics for binned data, and normalizes the results to a target
    luminance value.
    
    Parameters
    ----------
    image_path : str
        Path to the image file (supported formats: BMP, JPEG, PNG, TIFF)
    upper_rgb_threshold : float, default=100
        Upper RGB threshold value for data filtering to exclude bright artifacts
    lower_rgb_threshold : float, default=0
        Lower RGB threshold value to exclude extremely dark regions
    buffer : int, default=20
        Number of buffer pixels above and below filtered regions
    top_trim : int, default=0
        Number of pixels to trim from top of image
    bottom_trim : int, default=0
        Number of pixels to trim from bottom of image
    target_luminance : float, default=130
        Target mean luminance value to scale RGB values to
    bin_size : int, default=10
        Size of bins in pixels for averaging RGB values along depth
    width_start_pct : float, default=0.25
        Starting percentage of width for analysis strip (0.0 to 1.0)
    width_end_pct : float, default=0.75
        Ending percentage of width for analysis strip (0.0 to 1.0)
        
    Returns
    -------
    tuple
        Contains the following arrays:
        - depths_pixels (numpy.ndarray): Depth positions in pixels
        - widths_pixels (numpy.ndarray): Width positions in pixels
        - r_means (numpy.ndarray): Mean red values for each depth bin
        - g_means (numpy.ndarray): Mean green values for each depth bin
        - b_means (numpy.ndarray): Mean blue values for each depth bin
        - r_stds (numpy.ndarray): Standard deviations of red values
        - g_stds (numpy.ndarray): Standard deviations of green values
        - b_stds (numpy.ndarray): Standard deviations of blue values
        - lum_means (numpy.ndarray): Mean luminance values for each depth bin
        - lum_stds (numpy.ndarray): Standard deviations of luminance values
        - img_array (numpy.ndarray): Processed and scaled image array
        
    Raises
    ------
    ValueError
        If image file format is not supported or file cannot be opened
    FileNotFoundError
        If image file is not found
        
    Example
    -------
    >>> depths, widths, r, g, b, r_std, g_std, b_std, lum, lum_std, img = extract_rgb_profile(
    ...     'core_section.bmp', upper_rgb_threshold=120, buffer=30
    ... )
    """
    # Validate file format
    supported_formats = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported file format '{file_ext}'. "
                        f"Supported formats are: {', '.join(sorted(supported_formats))}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Open image and convert to numpy array
    try:
        img = Image.open(image_path)
        img_array = np.array(img, dtype=float)
    except Exception as e:
        raise ValueError(f"Error opening image file '{image_path}': {str(e)}")
    
    # Trim image
    img_array = trim_image(img_array, top_trim, bottom_trim)
    
    # Get image dimensions
    height = img_array.shape[0]
    width = img_array.shape[1]
    
    # Get strip based on specified width percentages
    center_start = int(width * width_start_pct)
    center_end = int(width * width_end_pct)
    center_strip = img_array[:, center_start:center_end]
    
    # Calculate number of bins
    num_bins = height // bin_size
    if height % bin_size != 0:
        num_bins += 1
    
    # Initialize arrays for binned values
    r_means = np.zeros(num_bins)
    g_means = np.zeros(num_bins)
    b_means = np.zeros(num_bins)
    r_stds = np.zeros(num_bins)
    g_stds = np.zeros(num_bins)
    b_stds = np.zeros(num_bins)
    lum_means = np.zeros(num_bins)
    lum_stds = np.zeros(num_bins)
    
    # Process each bin
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, height)
        
        # Calculate mean RGB values for this bin
        r_means[i] = np.nanmean(center_strip[start_idx:end_idx, :, 0])
        g_means[i] = np.nanmean(center_strip[start_idx:end_idx, :, 1])
        b_means[i] = np.nanmean(center_strip[start_idx:end_idx, :, 2])
        
        # Calculate standard deviations for this bin
        r_stds[i] = np.nanstd(center_strip[start_idx:end_idx, :, 0])
        g_stds[i] = np.nanstd(center_strip[start_idx:end_idx, :, 1])
        b_stds[i] = np.nanstd(center_strip[start_idx:end_idx, :, 2])
        
        # Calculate luminance for this bin
        r_values = center_strip[start_idx:end_idx, :, 0]
        g_values = center_strip[start_idx:end_idx, :, 1]
        b_values = center_strip[start_idx:end_idx, :, 2]
        luminance = 0.2126 * r_values + 0.7152 * g_values + 0.0722 * b_values
        
        lum_means[i] = np.nanmean(luminance)
        lum_stds[i] = np.nanstd(luminance)
    
    # Create mask where 2 or more RGB values exceed upper threshold or are below lower threshold
    over_threshold = np.array([r_means > upper_rgb_threshold, g_means > upper_rgb_threshold, b_means > upper_rgb_threshold])
    under_threshold = np.array([r_means < lower_rgb_threshold, g_means < lower_rgb_threshold, b_means < lower_rgb_threshold])
    
    mask_over = np.sum(over_threshold, axis=0) >= 2
    mask_under = np.sum(under_threshold, axis=0) >= 2
    mask = np.logical_or(mask_over, mask_under)
    
    # Add buffer zones above and below masked regions
    buffer_size = buffer // bin_size  # Adjust buffer size for binned data
    buffered_mask = np.copy(mask)
    for i in range(len(mask)):
        if mask[i]:
            start = max(0, i - buffer_size)
            end = min(len(mask), i + buffer_size + 1)
            buffered_mask[start:end] = True
    
    # Add nan values to top and bottom quarter of buffer size
    quarter_buffer = buffer_size //2
    buffered_mask[:quarter_buffer] = True  # Top
    buffered_mask[-quarter_buffer:] = True  # Bottom
    
    # Apply buffered mask to means and standard deviations
    r_means[buffered_mask] = np.nan
    g_means[buffered_mask] = np.nan
    b_means[buffered_mask] = np.nan
    r_stds[buffered_mask] = np.nan
    g_stds[buffered_mask] = np.nan
    b_stds[buffered_mask] = np.nan
    lum_means[buffered_mask] = np.nan
    lum_stds[buffered_mask] = np.nan
    
    # Rescale RGB values to target luminance
    scale_factor = target_luminance / np.nanmean(lum_means)
    r_means *= scale_factor
    g_means *= scale_factor
    b_means *= scale_factor
    lum_means *= scale_factor
    
    # Scale the image array accordingly
    img_array = img_array * scale_factor
    
    # Create depths array (center points of bins)
    depths_pixels = np.arange(bin_size//2, height, bin_size)
    if len(depths_pixels) < num_bins:
        depths_pixels = np.append(depths_pixels, height - bin_size//2)
        
    # Create widths array (center points of bins)
    widths_pixels = np.arange(bin_size//2, img_array.shape[1], bin_size)
    if len(widths_pixels) < num_bins:
        widths_pixels = np.append(widths_pixels, img_array.shape[1] - bin_size//2)
    
    return depths_pixels, widths_pixels, r_means, g_means, b_means, r_stds, g_stds, b_stds, lum_means, lum_stds, img_array



def rgb_process_and_stitch(data_reading_structure, rgb_data_dir, stitchbuffer=10, 
                        width_start_pct=0.25, width_end_pct=0.75,
                        save_csv=True, output_csv=None, total_length_cm=None):
    """
    Stitch multiple core sections together by processing RGB profiles and combining images.
    
    This function processes multiple core section images in sequence, extracts RGB profiles
    from each section using section-specific parameters, adjusts depths to create a
    continuous profile, and combines the images and data arrays. Buffer zones at section
    boundaries are handled to avoid artifacts at stitching points.
    Optionally exports results to CSV file.
    
    Multiple 'empty' segments are automatically numbered as 'empty_1', 'empty_2', etc.,
    while preserving their order in the core structure.
    
    Parameters
    ----------
    data_reading_structure : dict or list
        Core structure definition in one of two formats:
        1. Dictionary format: {section_name: section_params, ...}
        2. List format: [(section_name, section_params), ...]
        
        Each section_params contains processing parameters:
        - upper_rgb_threshold: Upper RGB threshold for filtering
        - lower_rgb_threshold: Lower RGB threshold for filtering
        - buffer: Buffer size for filtering
        - top_trim: Pixels to trim from top
        - bottom_trim: Pixels to trim from bottom
        - target_luminance: Target luminance for normalization
        
        For empty segments, use:
        - rgb_pxlength: Height in pixels
        - rgb_pxwidth: Width in pixels
        
        Note: Multiple 'empty' segments will be automatically numbered as 'empty_1', 'empty_2', etc.
        For dictionaries, use unique keys like 'empty_1', 'empty_2' or use list format for duplicates.
    rgb_data_dir : str
        Base directory path containing the core section image files
    stitchbuffer : int, default=10
        Number of bin rows to remove at stitching edges to avoid artifacts
    width_start_pct : float, default=0.25
        Starting percentage of width for analysis strip (0.0 to 1.0)
    width_end_pct : float, default=0.75
        Ending percentage of width for analysis strip (0.0 to 1.0)
    save_csv : bool, default=True
        Whether to save results to CSV file
    output_csv : str, optional
        Full path for output CSV file. Required if save_csv=True
    total_length_cm : float, optional
        Total core length in centimeters for depth conversion. Required if save_csv=True
        
    Returns
    -------
    dict
        stitched_rgb_metadata : Dictionary containing all stitched RGB data with keys:
        - 'depths' (numpy.ndarray): Continuous depth array in pixels
        - 'r' (numpy.ndarray): Red color values for complete core
        - 'g' (numpy.ndarray): Green color values for complete core
        - 'b' (numpy.ndarray): Blue color values for complete core
        - 'r_std' (numpy.ndarray): Red standard deviations for complete core
        - 'g_std' (numpy.ndarray): Green standard deviations for complete core
        - 'b_std' (numpy.ndarray): Blue standard deviations for complete core
        - 'lum' (numpy.ndarray): Luminance values for complete core
        - 'lum_std' (numpy.ndarray): Luminance standard deviations for complete core
        - 'image' (numpy.ndarray): Combined image array for complete core
        
    Raises
    ------
    ValueError
        If save_csv is True but output_csv or total_length_cm is not specified
        
    Example
    -------
    >>> data_reading_structure = {
    ...     'section1.bmp': {'upper_rgb_threshold': 120, 'buffer': 30, 'top_trim': 50, ...},
    ...     'empty_1': {'rgb_pxlength': 1000, 'rgb_pxwidth': 500},
    ...     'section2.bmp': {'upper_rgb_threshold': 110, 'buffer': 40, 'top_trim': 60, ...}
    ... }
    >>> rgb_metadata = rgb_process_and_stitch(
    ...     data_reading_structure, '/path/to/images/', stitchbuffer=15,
    ...     save_csv=True, output_csv='output.csv', total_length_cm=100
    ... )
    >>> # Access individual components
    >>> depths = rgb_metadata['depths']
    >>> stitched_image = rgb_metadata['image']
    """
    # Validate CSV export parameters
    if save_csv:
        if output_csv is None:
            raise ValueError("output_csv must be specified when save_csv=True")
        if total_length_cm is None:
            raise ValueError("total_length_cm must be specified when save_csv=True")
    # Process data reading structure in order and handle multiple 'empty' segments
    # Handle both dictionary and list formats
    if isinstance(data_reading_structure, dict):
        # Dictionary format - convert to list of tuples
        processed_segments = []
        empty_counter = 1
        
        for section_name, section_params in data_reading_structure.items():
            if section_name == 'empty':
                # Automatically number empty segments
                if empty_counter == 1:
                    processed_section_name = 'empty_1'
                else:
                    processed_section_name = f'empty_{empty_counter}'
                empty_counter += 1
                processed_segments.append((processed_section_name, section_params))
            else:
                processed_segments.append((section_name, section_params))
    else:
        # List format - already in correct format, just number empty segments
        processed_segments = []
        empty_counter = 1
        
        for section_name, section_params in data_reading_structure:
            if section_name == 'empty':
                # Automatically number empty segments
                if empty_counter == 1:
                    processed_section_name = 'empty_1'
                else:
                    processed_section_name = f'empty_{empty_counter}'
                empty_counter += 1
                processed_segments.append((processed_section_name, section_params))
            else:
                processed_segments.append((section_name, section_params))

    # Initialize lists to store data for stitching
    all_depths = []
    all_r = []
    all_g = []
    all_b = []
    all_r_std = []
    all_g_std = []
    all_b_std = []
    all_lum = []
    all_lum_std = []
    all_images = []
    current_depth = 0
    max_width = 0  # Track the maximum width of all images

    # Process each section
    for file_name, params in processed_segments:
        print(f"\nProcessing {file_name}...")
        
        # Check if this is an empty segment
        if params.get('scans') is None and 'rgb_pxlength' in params and 'rgb_pxwidth' in params:
            print(f"Creating empty RGB segment for {file_name}")
            
            # Get target dimensions
            target_height = params['rgb_pxlength']
            target_width = params['rgb_pxwidth']
            
            # Create empty image (RGB with 3 channels, filled with white)
            empty_image = np.full((target_height, target_width, 3), 255.0, dtype=np.float64)
            
            # Create depth array with bin_size=10 (default from extract_rgb_profile)
            bin_size = 10
            num_bins = target_height // bin_size
            if target_height % bin_size != 0:
                num_bins += 1
                
            # Create empty RGB curves with NaN values
            depths = np.arange(bin_size//2, target_height, bin_size)
            if len(depths) < num_bins:
                depths = np.append(depths, target_height - bin_size//2)
                
            r = np.full(num_bins, np.nan)
            g = np.full(num_bins, np.nan)
            b = np.full(num_bins, np.nan)
            r_std = np.full(num_bins, np.nan)
            g_std = np.full(num_bins, np.nan)
            b_std = np.full(num_bins, np.nan)
            lum = np.full(num_bins, np.nan)
            lum_std = np.full(num_bins, np.nan)
            
            # Plot empty segment
            core_name = f"{file_name} (empty)"
            plot_rgbimg_curves(depths, r, g, b, r_std, g_std, b_std, lum, lum_std, empty_image, core_name=core_name)
            
            # Adjust depths to continue from previous section
            adjusted_depths = depths + current_depth
            current_depth = adjusted_depths[-1]
            print(f"\nEmpty segment length {target_height} (pixels), width {target_width} (pixels)")
            
            # Keep track of the maximum width
            if target_width > max_width:
                max_width = target_width
            
            # Handle stitching for empty segments
            if len(all_depths) > 0:  # Not the first section
                # Set last buffer rows of previous section to nan
                for lst in [all_r, all_g, all_b, all_r_std, all_g_std, all_b_std, all_lum, all_lum_std]:
                    lst[-stitchbuffer:] = [np.nan] * stitchbuffer
                
                # Skip first buffer rows of current section
                all_depths.extend(adjusted_depths[stitchbuffer:])
                all_r.extend(r[stitchbuffer:])
                all_g.extend(g[stitchbuffer:])
                all_b.extend(b[stitchbuffer:])
                all_r_std.extend(r_std[stitchbuffer:])
                all_g_std.extend(g_std[stitchbuffer:])
                all_b_std.extend(b_std[stitchbuffer:])
                all_lum.extend(lum[stitchbuffer:])
                all_lum_std.extend(lum_std[stitchbuffer:])
            else:  # First section
                all_depths.extend(adjusted_depths)
                all_r.extend(r)
                all_g.extend(g)
                all_b.extend(b)
                all_r_std.extend(r_std)
                all_g_std.extend(g_std)
                all_b_std.extend(b_std)
                all_lum.extend(lum)
                all_lum_std.extend(lum_std)
            
            all_images.append(empty_image)
            continue
        
        image_path = f"{rgb_data_dir}/{file_name}"
        core_name = file_name.split('-image')[0].upper()
        
        # Extract RGB profile with file-specific parameters
        depths, width, r, g, b, r_std, g_std, b_std, lum, lum_std, img = extract_rgb_profile(
            image_path,
            upper_rgb_threshold=params["upper_rgb_threshold"],
            lower_rgb_threshold=params.get("lower_rgb_threshold", 0),
            buffer=params["buffer"],
            top_trim=params["top_trim"],
            bottom_trim=params["bottom_trim"],
            target_luminance=params["target_luminance"],
            width_start_pct=width_start_pct, 
            width_end_pct=width_end_pct
        )
        
        # Plot individual section
        section_metadata = {
            'depths': depths, 'r': r, 'g': g, 'b': b,
            'r_std': r_std, 'g_std': g_std, 'b_std': b_std,
            'lum': lum, 'lum_std': lum_std, 'image': img
        }
        plot_rgbimg_curves(rgb_metadata=section_metadata, core_name=core_name)
        
        # Adjust depths to continue from previous section
        adjusted_depths = depths + current_depth
        current_depth = adjusted_depths[-1]
        print(f"\nTrimmed core length {max(depths)} (pixels), width {max(width)} (pixels)")
        
        # Keep track of the maximum width
        if img.shape[1] > max_width:
            max_width = img.shape[1]
        
        # Append data to lists, removing buffer rows at stitching edges except for first and last sections
        if len(all_depths) > 0:  # Not the first section
            # Set last buffer rows of previous section to nan
            for lst in [all_r, all_g, all_b, all_r_std, all_g_std, all_b_std, all_lum, all_lum_std]:
                lst[-stitchbuffer:] = [np.nan] * stitchbuffer
            
            # Skip first buffer rows of current section
            all_depths.extend(adjusted_depths[stitchbuffer:])
            all_r.extend(r[stitchbuffer:])
            all_g.extend(g[stitchbuffer:])
            all_b.extend(b[stitchbuffer:])
            all_r_std.extend(r_std[stitchbuffer:])
            all_g_std.extend(g_std[stitchbuffer:])
            all_b_std.extend(b_std[stitchbuffer:])
            all_lum.extend(lum[stitchbuffer:])
            all_lum_std.extend(lum_std[stitchbuffer:])
        else:  # First section
            all_depths.extend(adjusted_depths)
            all_r.extend(r)
            all_g.extend(g)
            all_b.extend(b)
            all_r_std.extend(r_std)
            all_g_std.extend(g_std)
            all_b_std.extend(b_std)
            all_lum.extend(lum)
            all_lum_std.extend(lum_std)
        
        all_images.append(img)

    # Convert lists to numpy arrays
    all_depths = np.array(all_depths)
    all_r = np.array(all_r)
    all_g = np.array(all_g)
    all_b = np.array(all_b)
    all_r_std = np.array(all_r_std)
    all_g_std = np.array(all_g_std)
    all_b_std = np.array(all_b_std)
    all_lum = np.array(all_lum)
    all_lum_std = np.array(all_lum_std)

    # Resize all images to have the same width before stacking
    resized_images = []
    for img in all_images:
        if img.shape[1] < max_width:
            # Pad the image to match the maximum width
            pad_width = max_width - img.shape[1]
            padded_img = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            resized_images.append(padded_img)
        else:
            resized_images.append(img)
    
    # Stitch images vertically
    stitched_image = np.vstack(resized_images)
    
    # Export to CSV if requested
    if save_csv:
        # Convert depth from pixels to cm
        depth_cm = all_depths * (total_length_cm / all_depths[-1])
        
        # Create DataFrame with all RGB and luminance data
        df = pd.DataFrame({
            'SB_DEPTH_pxl': all_depths,
            'SB_DEPTH_cm': depth_cm,
            'R': all_r,
            'R_std': all_r_std,
            'G': all_g,
            'G_std': all_g_std,
            'B': all_b,
            'B_std': all_b_std,
            'Lumin': all_lum,
            'Lumin_std': all_lum_std
        })
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save RGB data to CSV
        df.to_csv(output_csv, index=False)
        print(f"RGB data saved to: {output_csv}")
    
    # Create metadata dictionary containing all results
    stitched_rgb_metadata = {
        'depths': all_depths,
        'r': all_r,
        'g': all_g,
        'b': all_b,
        'r_std': all_r_std,
        'g_std': all_g_std,
        'b_std': all_b_std,
        'lum': all_lum,
        'lum_std': all_lum_std,
        'image': stitched_image
    }
    
    return stitched_rgb_metadata 