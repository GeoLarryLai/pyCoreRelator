"""
RGB image processing functions for pyCoreRelator.

Included Functions:
- trim_image: Trim specified number of pixels from top and bottom of image array
- extract_rgb_profile: Extract RGB color profiles along the y-axis of an image file
- plot_rgb_profile: Create visualization plots of RGB analysis results
- stitch_core_sections: Stitch multiple core sections together by processing RGB profiles

This module provides comprehensive tools for processing RGB images of geological cores,
extracting color profiles, and creating visualizations for core analysis.
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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


def stitch_core_sections(core_structure, mother_dir, stitchbuffer=10, 
                        width_start_pct=0.25, width_end_pct=0.75):
    """
    Stitch multiple core sections together by processing RGB profiles and combining images.
    
    This function processes multiple core section images in sequence, extracts RGB profiles
    from each section using section-specific parameters, adjusts depths to create a
    continuous profile, and combines the images and data arrays. Buffer zones at section
    boundaries are handled to avoid artifacts at stitching points.
    
    Parameters
    ----------
    core_structure : dict
        Dictionary containing parameters for each core section file.
        Keys are filenames, values are dictionaries with processing parameters:
        - upper_rgb_threshold: Upper RGB threshold for filtering
        - lower_rgb_threshold: Lower RGB threshold for filtering
        - buffer: Buffer size for filtering
        - top_trim: Pixels to trim from top
        - bottom_trim: Pixels to trim from bottom
        - target_luminance: Target luminance for normalization
    mother_dir : str
        Base directory path containing the core section image files
    stitchbuffer : int, default=10
        Number of bin rows to remove at stitching edges to avoid artifacts
    width_start_pct : float, default=0.25
        Starting percentage of width for analysis strip (0.0 to 1.0)
    width_end_pct : float, default=0.75
        Ending percentage of width for analysis strip (0.0 to 1.0)
        
    Returns
    -------
    tuple
        Contains the following stitched arrays:
        - all_depths (numpy.ndarray): Continuous depth array in pixels
        - all_r (numpy.ndarray): Red color values for complete core
        - all_g (numpy.ndarray): Green color values for complete core
        - all_b (numpy.ndarray): Blue color values for complete core
        - all_r_std (numpy.ndarray): Red standard deviations for complete core
        - all_g_std (numpy.ndarray): Green standard deviations for complete core
        - all_b_std (numpy.ndarray): Blue standard deviations for complete core
        - all_lum (numpy.ndarray): Luminance values for complete core
        - all_lum_std (numpy.ndarray): Luminance standard deviations for complete core
        - stitched_image (numpy.ndarray): Combined image array for complete core
        
    Example
    -------
    >>> core_structure = {
    ...     'section1.bmp': {'upper_rgb_threshold': 120, 'buffer': 30, 'top_trim': 50, ...},
    ...     'section2.bmp': {'upper_rgb_threshold': 110, 'buffer': 40, 'top_trim': 60, ...}
    ... }
    >>> depths, r, g, b, r_std, g_std, b_std, lum, lum_std, img = stitch_core_sections(
    ...     core_structure, '/path/to/images/', stitchbuffer=15
    ... )
    """
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

    # Process each file
    for file_name, params in core_structure.items():
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
            plot_rgb_profile(depths, r, g, b, r_std, g_std, b_std, lum, lum_std, empty_image, core_name=core_name)
            
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
        
        image_path = f"{mother_dir}/{file_name}"
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
        plot_rgb_profile(depths, r, g, b, r_std, g_std, b_std, lum, lum_std, img, core_name=core_name)
        
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
    
    return (all_depths, all_r, all_g, all_b, all_r_std, all_g_std, 
            all_b_std, all_lum, all_lum_std, stitched_image) 