"""
Data loading and core plotting functions for pyCoreRelator.

Included Functions:
- load_log_data: Load log data from CSV files and resample to common depth scale
- resample_datasets: Resample multiple datasets to a common depth scale
- plot_core_data: Plot core data with optional RGB and CT images

This module provides utilities for loading core log data from CSV files and 
plotting core data with optional RGB and CT images. It handles multiple log types,
data normalization, and resampling to common depth scales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_log_data(log_paths, img_paths, log_columns, depth_column='SB_DEPTH_cm', normalize=True, column_alternatives=None):
    """
    Load log data and images for a core.
    
    This function loads multiple log datasets from CSV files, resamples them to a common
    depth scale, and optionally loads RGB and CT images. It supports alternative column
    names and automatic data normalization.
    
    Parameters
    ----------
    log_paths : dict
        Dictionary mapping log names to file paths
    img_paths : dict
        Dictionary mapping image types ('rgb', 'ct') to file paths
    log_columns : list of str
        List of log column names to load from the CSV files
    depth_column : str, default='SB_DEPTH_cm'
        Name of the depth column in the CSV files
    normalize : bool, default=True
        Whether to normalize each log to the range [0, 1]
    column_alternatives : dict, optional
        Dictionary mapping log column names to lists of alternative column names
        to try if the primary column name is not found
    
    Returns
    -------
    log : numpy.ndarray
        Log data with shape (n_samples, n_logs) for multiple logs or (n_samples,) for single log
    md : numpy.ndarray
        Measured depths array
    available_columns : list of str
        Names of the logs that were successfully loaded
    rgb_img : numpy.ndarray or None
        RGB image array if available, None otherwise
    ct_img : numpy.ndarray or None
        CT image array if available, None otherwise
    
    Example
    -------
    >>> log_paths = {'MS': 'data/core1_ms.csv', 'Lumin': 'data/core1_lumin.csv'}
    >>> img_paths = {'rgb': 'data/core1_rgb.jpg', 'ct': 'data/core1_ct.jpg'}
    >>> log_columns = ['MS', 'Lumin']
    >>> log, md, cols, rgb_img, ct_img = load_log_data(log_paths, img_paths, log_columns)
    """
    # Initialize lists to store data
    datasets = []
    available_columns = []
    
    # Try to load images
    rgb_img = None
    ct_img = None
    
    try:
        rgb_img = plt.imread(img_paths.get('rgb', ''))
        print(f"Loaded RGB image")
    except Exception as e:
        print(f"Could not load RGB image: {e}")
    
    try:
        ct_img = plt.imread(img_paths.get('ct', ''))
        print(f"Loaded CT image")
    except Exception as e:
        print(f"Could not load CT image: {e}")
    
    # Process each log column
    for log_column in log_columns:
        if log_column not in log_paths:
            print(f"Warning: No path defined for {log_column}. Skipping.")
            continue
            
        log_path = log_paths[log_column]
        
        # Try to load the data
        try:
            df = pd.read_csv(log_path)
            
            # Check if the required depth column exists
            if depth_column not in df.columns:
                print(f"Warning: Depth column {depth_column} not found in {log_path}. Skipping.")
                continue
                
            # Find the effective column name
            effective_column = log_column
            if log_column not in df.columns:
                # Try alternative column names if provided
                found = False
                if column_alternatives and log_column in column_alternatives:
                    for alt in column_alternatives[log_column]:
                        if alt in df.columns:
                            effective_column = alt
                            found = True
                            print(f"Using alternative column {alt} for {log_column}")
                            break
                
                if not found:
                    print(f"Warning: Log column {log_column} not found in {log_path}. Available columns: {list(df.columns)}. Skipping.")
                    continue
            
            # Extract data
            data = {}
            data['depth'] = np.array(df[depth_column])
            data[log_column] = np.array(df[effective_column])
            
            # Normalize the log data if requested
            if normalize:
                min_val = np.min(data[log_column])
                max_val = np.max(data[log_column])
                if max_val > min_val:  # Avoid division by zero
                    data[log_column] = (data[log_column] - min_val) / (max_val - min_val)
                else:
                    data[log_column] = np.zeros_like(data[log_column])
            
            # Add to datasets and available columns
            datasets.append(data)
            available_columns.append(log_column)
            
        except Exception as e:
            print(f"Error loading {log_path}: {e}")
    
    # If no datasets were loaded, return empty arrays
    if not datasets:
        print(f"No log datasets were loaded")
        return np.array([]), np.array([]), [], rgb_img, ct_img
    
    # Resample all datasets to a common depth scale
    resampled_data = resample_datasets(datasets)
    
    # Stack the selected columns
    log = np.column_stack([resampled_data[col] for col in available_columns])
    md = resampled_data['depth']
    
    # If only one log column was loaded, return as 1D array for backward compatibility
    if len(available_columns) == 1:
        log = log.flatten()
    
    return log, md, available_columns, rgb_img, ct_img

def resample_datasets(datasets, target_resolution_factor=2):
    """
    Resample multiple datasets to a common depth scale.
    
    This function takes multiple datasets with potentially different depth sampling
    and resamples them all to a common high-resolution depth scale using linear
    interpolation.
    
    Parameters
    ----------
    datasets : list of dict
        List of dictionaries, each containing 'depth' array and data arrays
    target_resolution_factor : float, default=2
        Factor to divide the lowest resolution by to create target resolution
    
    Returns
    -------
    dict
        Dictionary with resampled data arrays and common 'depth' scale
    
    Example
    -------
    >>> datasets = [
    ...     {'depth': np.array([0, 10, 20]), 'MS': np.array([0.1, 0.5, 0.9])},
    ...     {'depth': np.array([0, 5, 15, 20]), 'Lumin': np.array([0.2, 0.4, 0.6, 0.8])}
    ... ]
    >>> resampled = resample_datasets(datasets)
    """
    if not datasets:
        return {'depth': np.array([])}
        
    # Find depth ranges across all datasets
    min_depths = [np.min(dataset['depth']) for dataset in datasets]
    max_depths = [np.max(dataset['depth']) for dataset in datasets]
    start_depth = min(min_depths)
    end_depth = max(max_depths)
    
    # Calculate resolutions for each dataset
    def calculate_resolution(depth_array):
        return (depth_array[-1] - depth_array[0]) / len(depth_array)
    
    resolutions = [calculate_resolution(dataset['depth']) for dataset in datasets]
    lowest_resolution = max(resolutions)  # Largest value = lowest resolution
    
    # Create target depth array with improved resolution
    target_resolution = lowest_resolution / target_resolution_factor
    num_points = int((end_depth - start_depth) / target_resolution) + 1
    target_depth = np.linspace(start_depth, end_depth, num_points)
    
    # Resample all data to the target depth using linear interpolation
    resampled_data = {'depth': target_depth}
    
    for dataset in datasets:
        for key, values in dataset.items():
            if key != 'depth':  # Skip depth as we already have the target depth
                resampled_data[key] = np.interp(target_depth, dataset['depth'], values)
    
    return resampled_data


def plot_core_data(md, log, title, rgb_img=None, ct_img=None, boundaries=None, figsize=(20, 4), 
                  label_name=None, available_columns=None, is_multilog=False):
    """
    Plot core data with optional RGB and CT images and support for multiple log types.
    
    This function creates a comprehensive plot of core data including log curves and
    optional RGB/CT images. It automatically adjusts the layout based on available
    images and supports both single and multiple log plotting with custom styling.
    
    Parameters
    ----------
    md : array_like
        Array of depth values
    log : array_like
        Array of log values, either 1D for single log or 2D for multiple logs
    title : str
        Title for the plot
    rgb_img : array_like, optional
        RGB image array to display above the log curves
    ct_img : array_like, optional
        CT image array to display above the log curves
    boundaries : array_like, optional
        Array of depth points for marking boundaries with vertical lines
    figsize : tuple, default=(20, 4)
        Figure size tuple (width, height)
    label_name : str, optional
        Name for the log curve label (used for single log)
    available_columns : list of str, optional
        Names of the log columns for multidimensional logs
    is_multilog : bool, default=False
        Whether log contains multiple logs (multidimensional)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    plot_ax : matplotlib.axes.Axes
        The main plotting axis containing the log curves
    
    Example
    -------
    >>> import numpy as np
    >>> md = np.linspace(0, 100, 200)
    >>> log = np.random.random((200, 2))
    >>> fig, ax = plot_core_data(md, log, "Core 1", available_columns=['MS', 'Lumin'], is_multilog=True)
    """
    if is_multilog and log.ndim > 1 and log.shape[1] > 1:
        # Define color and style mapping for each column type
        color_style_map = {
            'R': {'color': 'red', 'linestyle': '--'},
            'G': {'color': 'green', 'linestyle': '--'},
            'B': {'color': 'blue', 'linestyle': '--'},
            'Lumin': {'color': 'darkgray', 'linestyle': '--'},
            'hiresMS': {'color': 'black', 'linestyle': '-'},
            'MS': {'color': 'gray', 'linestyle': '-'},
            'Den_gm/cc': {'color': 'orange', 'linestyle': '-'},
            'CT': {'color': 'purple', 'linestyle': '-'}
        }
        
        # Use names from available_columns or create generic names
        column_names = available_columns if available_columns else [f"Log {i+1}" for i in range(log.shape[1])]
    
    # Setup figure layout based on available images
    if rgb_img is not None and ct_img is not None:
        # Create figure with three subplots for both images
        fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1, 2]})
        
        # RGB image display with transposed axes for proper orientation
        axs[0].imshow(rgb_img.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
        axs[0].set_ylabel('RGB')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        # CT image display with conditional transposition for grayscale/color
        axs[1].imshow(ct_img.transpose(1, 0, 2) if len(ct_img.shape) == 3 else 
                      ct_img.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
        axs[1].set_ylabel('CT')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        # Log curves plotting
        if is_multilog and log.ndim > 1 and log.shape[1] > 1:
            for i, col_name in enumerate(column_names):
                # Get predefined color and style or use defaults
                if col_name in color_style_map:
                    color = color_style_map[col_name]['color']
                    linestyle = color_style_map[col_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                
                axs[2].plot(md, log[:, i], label=col_name, color=color, 
                          linestyle=linestyle, linewidth=0.7)
        else:
            axs[2].plot(md, log, label=label_name, linewidth=0.7)
        
        axs[2].set_ylim(0, 1)
        axs[2].set_xlim(md[0], md[-1])
        axs[2].set_xlabel('depth (cm)')
        axs[2].set_ylabel('Normalized Intensity')
        
        if is_multilog or label_name is not None:
            axs[2].legend(loc='upper left')
        
        # Add boundary markers if provided
        if boundaries is not None:
            count = 1
            for i in range(len(boundaries)):
                axs[2].axvline(x=boundaries[i], color='k', linestyle='--')
                axs[2].text(boundaries[i], 0.9, str(count), fontsize=12, fontweight='bold')
                count += 1
        
        plot_ax = axs[2]
    
    elif rgb_img is not None:
        # Create figure with two subplots for RGB image only
        fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})
        
        axs[0].imshow(rgb_img.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
        axs[0].set_ylabel('RGB')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        # Log curves plotting
        if is_multilog and log.ndim > 1 and log.shape[1] > 1:
            for i, col_name in enumerate(column_names):
                if col_name in color_style_map:
                    color = color_style_map[col_name]['color']
                    linestyle = color_style_map[col_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                
                axs[1].plot(md, log[:, i], label=col_name, color=color, 
                          linestyle=linestyle, linewidth=0.7)
        else:
            axs[1].plot(md, log, label=label_name, linewidth=0.7)
        
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(md[0], md[-1])
        axs[1].set_xlabel('depth (cm)')
        axs[1].set_ylabel('Normalized Intensity')
        
        if is_multilog or label_name is not None:
            axs[1].legend(loc='upper left')
        
        if boundaries is not None:
            count = 1
            for i in range(len(boundaries)):
                axs[1].axvline(x=boundaries[i], color='k', linestyle='--')
                axs[1].text(boundaries[i], 0.9, str(count), fontsize=12, fontweight='bold')
                count += 1
        
        plot_ax = axs[1]
    
    elif ct_img is not None:
        # Create figure with two subplots for CT image only
        fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})
        
        axs[0].imshow(ct_img.transpose(1, 0, 2) if len(ct_img.shape) == 3 else 
                      ct_img.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
        axs[0].set_ylabel('CT')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        # Log curves plotting
        if is_multilog and log.ndim > 1 and log.shape[1] > 1:
            for i, col_name in enumerate(column_names):
                if col_name in color_style_map:
                    color = color_style_map[col_name]['color']
                    linestyle = color_style_map[col_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                
                axs[1].plot(md, log[:, i], label=col_name, color=color, 
                          linestyle=linestyle, linewidth=0.7)
        else:
            axs[1].plot(md, log, label=label_name, linewidth=0.7)
        
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(md[0], md[-1])
        axs[1].set_xlabel('depth (cm)')
        axs[1].set_ylabel('Normalized Intensity')
        
        if is_multilog or label_name is not None:
            axs[1].legend(loc='upper left')
        
        if boundaries is not None:
            count = 1
            for i in range(len(boundaries)):
                axs[1].axvline(x=boundaries[i], color='k', linestyle='--')
                axs[1].text(boundaries[i], 0.9, str(count), fontsize=12, fontweight='bold')
                count += 1
        
        plot_ax = axs[1]
    
    else:
        # Create figure with single subplot for log curves only
        fig, ax = plt.subplots(figsize=figsize)
        
        # Log curves plotting
        if is_multilog and log.ndim > 1 and log.shape[1] > 1:
            for i, col_name in enumerate(column_names):
                if col_name in color_style_map:
                    color = color_style_map[col_name]['color']
                    linestyle = color_style_map[col_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                
                ax.plot(md, log[:, i], label=col_name, color=color, 
                      linestyle=linestyle, linewidth=0.7)
        else:
            ax.plot(md, log, label=label_name, linewidth=0.7)
        
        ax.set_ylim(0, 1)
        ax.set_xlim(md[0], md[-1])
        ax.set_xlabel('depth (cm)')
        ax.set_ylabel('Normalized Intensity')
        
        if is_multilog or label_name is not None:
            ax.legend(loc='upper left')
        
        if boundaries is not None:
            count = 1
            for i in range(len(boundaries)):
                ax.axvline(x=boundaries[i], color='r')
                ax.text(boundaries[i], 0.2, str(count))
                count += 1
        
        plot_ax = ax
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    return fig, plot_ax