"""
Data loading and core plotting functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_log_data(log_paths, img_paths, log_columns, depth_column='SB_DEPTH_cm', normalize=True, column_alternatives=None):
    """
    Load log data and images for a core.
    
    Parameters:
    -----------
    log_paths : dict
        Dictionary mapping log names to file paths
    img_paths : dict
        Dictionary mapping image types to file paths
    log_columns : list of str
        List of log column names to load
    depth_column : str, default='SB_DEPTH_cm'
        Name of the depth column
    normalize : bool, default=True
        Whether to normalize each log to the range [0, 1]
    column_alternatives : dict, optional
        Dictionary mapping log column names to lists of alternative column names.
        If None, no alternative column names will be tried.
    
    Returns:
    --------
    log : numpy.ndarray
        Log data with shape (n_samples, n_logs) or (n_samples,)
    md : numpy.ndarray
        Measured depths
    available_columns : list of str
        Names of the logs that were successfully loaded
    rgb_img : numpy.ndarray or None
        RGB image if available, None otherwise
    ct_img : numpy.ndarray or None
        CT image if available, None otherwise
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
            
            # Check if the required columns exist
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
    
    Parameters:
    -----------
    datasets : list of dict
        List of dictionaries containing depth and data arrays
    target_resolution_factor : float, default=2
        Factor to divide the lowest resolution by
    
    Returns:
    --------
    dict
        Dictionary with resampled data arrays and common depth scale
    """
    if not datasets:
        return {'depth': np.array([])}
        
    # Find depth ranges
    min_depths = [np.min(dataset['depth']) for dataset in datasets]
    max_depths = [np.max(dataset['depth']) for dataset in datasets]
    start_depth = min(min_depths)
    end_depth = max(max_depths)
    
    # Calculate resolutions
    def calculate_resolution(depth_array):
        return (depth_array[-1] - depth_array[0]) / len(depth_array)
    
    resolutions = [calculate_resolution(dataset['depth']) for dataset in datasets]
    lowest_resolution = max(resolutions)  # Largest value = lowest resolution
    
    # Create target depth array with improved resolution
    target_resolution = lowest_resolution / target_resolution_factor
    num_points = int((end_depth - start_depth) / target_resolution) + 1
    target_depth = np.linspace(start_depth, end_depth, num_points)
    
    # Resample all data to the target depth
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
    
    Parameters:
    - md: array of depth values
    - log: array of log values (single log or multidimensional logs)
    - title: title for the plot
    - rgb_img: optional RGB image array
    - ct_img: optional CT image array
    - boundaries: optional array of depth points for marking boundaries
    - figsize: figure size tuple (width, height)
    - label_name: optional name for the log curve label (for single log)
    - available_columns: list of column names for multidimensional logs
    - is_multilog: whether log contains multiple logs (multidimensional)
    
    Returns:
    - fig: the created figure
    """
    if is_multilog and log.ndim > 1 and log.shape[1] > 1:
        # Define color and style mapping for each column
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
    
    # Setup figure based on available images
    if rgb_img is not None and ct_img is not None:
        # Create figure with three subplots if both RGB and CT images are provided
        fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1, 2]})
        
        # RGB image - flipping x & y axes
        axs[0].imshow(rgb_img.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
        axs[0].set_ylabel('RGB')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plots
        axs[0].set_yticks([])  # Hide y-axis ticks for image
        
        # CT image - flipping x & y axes
        axs[1].imshow(ct_img.transpose(1, 0, 2) if len(ct_img.shape) == 3 else 
                      ct_img.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
        axs[1].set_ylabel('CT')
        axs[1].set_xticks([])  # Hide x-axis ticks for middle plot
        axs[1].set_yticks([])  # Hide y-axis ticks for image
        
        # Log curves
        if is_multilog and log.ndim > 1 and log.shape[1] > 1:
            for i, col_name in enumerate(column_names):
                # Get color and style
                if col_name in color_style_map:
                    color = color_style_map[col_name]['color']
                    linestyle = color_style_map[col_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                
                # Plot the log
                axs[2].plot(md, log[:, i], label=col_name, color=color, 
                          linestyle=linestyle, linewidth=0.7)
        else:
            # Single log case
            axs[2].plot(md, log, label=label_name, linewidth=0.7)
        
        axs[2].set_ylim(0, 1)
        axs[2].set_xlim(md[0], md[-1])
        axs[2].set_xlabel('depth (cm)')
        axs[2].set_ylabel('Normalized Intensity')
        
        # Add legend for log curves
        if is_multilog or label_name is not None:
            axs[2].legend(loc='upper left')
        
        # Add boundaries if provided
        if boundaries is not None:
            count = 1
            for i in range(len(boundaries)):
                axs[2].axvline(x=boundaries[i], color='k', linestyle='--')
                axs[2].text(boundaries[i], 0.9, str(count), fontsize=12, fontweight='bold')
                count += 1
        
        plot_ax = axs[2]  # Store reference to main plotting axis
    
    elif rgb_img is not None:
        # Create figure with two subplots if only RGB image is provided
        fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})
        
        # RGB image - flipping x & y axes
        axs[0].imshow(rgb_img.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
        axs[0].set_ylabel('RGB')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plots
        axs[0].set_yticks([])  # Hide y-axis ticks for image
        
        # Log curves
        if is_multilog and log.ndim > 1 and log.shape[1] > 1:
            for i, col_name in enumerate(column_names):
                # Get color and style
                if col_name in color_style_map:
                    color = color_style_map[col_name]['color']
                    linestyle = color_style_map[col_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                
                # Plot the log
                axs[1].plot(md, log[:, i], label=col_name, color=color, 
                          linestyle=linestyle, linewidth=0.7)
        else:
            # Single log case
            axs[1].plot(md, log, label=label_name, linewidth=0.7)
        
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(md[0], md[-1])
        axs[1].set_xlabel('depth (cm)')
        axs[1].set_ylabel('Normalized Intensity')
        
        # Add legend for log curves
        if is_multilog or label_name is not None:
            axs[1].legend(loc='upper left')
        
        # Add boundaries if provided
        if boundaries is not None:
            count = 1
            for i in range(len(boundaries)):
                axs[1].axvline(x=boundaries[i], color='k', linestyle='--')
                axs[1].text(boundaries[i], 0.9, str(count), fontsize=12, fontweight='bold')
                count += 1
        
        plot_ax = axs[1]  # Store reference to main plotting axis
    
    elif ct_img is not None:
        # Create figure with two subplots if only CT image is provided
        fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 2]})
        
        # CT image - flipping x & y axes
        axs[0].imshow(ct_img.transpose(1, 0, 2) if len(ct_img.shape) == 3 else 
                      ct_img.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
        axs[0].set_ylabel('CT')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plots
        axs[0].set_yticks([])  # Hide y-axis ticks for image
        
        # Log curves
        if is_multilog and log.ndim > 1 and log.shape[1] > 1:
            for i, col_name in enumerate(column_names):
                # Get color and style
                if col_name in color_style_map:
                    color = color_style_map[col_name]['color']
                    linestyle = color_style_map[col_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                
                # Plot the log
                axs[1].plot(md, log[:, i], label=col_name, color=color, 
                          linestyle=linestyle, linewidth=0.7)
        else:
            # Single log case
            axs[1].plot(md, log, label=label_name, linewidth=0.7)
        
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(md[0], md[-1])
        axs[1].set_xlabel('depth (cm)')
        axs[1].set_ylabel('Normalized Intensity')
        
        # Add legend for log curves
        if is_multilog or label_name is not None:
            axs[1].legend(loc='upper left')
        
        # Add boundaries if provided
        if boundaries is not None:
            count = 1
            for i in range(len(boundaries)):
                axs[1].axvline(x=boundaries[i], color='k', linestyle='--')
                axs[1].text(boundaries[i], 0.9, str(count), fontsize=12, fontweight='bold')
                count += 1
        
        plot_ax = axs[1]  # Store reference to main plotting axis
    
    else:
        # Create figure with single subplot if no images
        fig, ax = plt.subplots(figsize=figsize)
        
        # Log curves
        if is_multilog and log.ndim > 1 and log.shape[1] > 1:
            for i, col_name in enumerate(column_names):
                # Get color and style
                if col_name in color_style_map:
                    color = color_style_map[col_name]['color']
                    linestyle = color_style_map[col_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                
                # Plot the log
                ax.plot(md, log[:, i], label=col_name, color=color, 
                      linestyle=linestyle, linewidth=0.7)
        else:
            # Single log case
            ax.plot(md, log, label=label_name, linewidth=0.7)
        
        ax.set_ylim(0, 1)
        ax.set_xlim(md[0], md[-1])
        ax.set_xlabel('depth (cm)')
        ax.set_ylabel('Normalized Intensity')
        
        # Add legend for log curves
        if is_multilog or label_name is not None:
            ax.legend(loc='upper left')
        
        # Add boundaries if provided
        if boundaries is not None:
            count = 1
            for i in range(len(boundaries)):
                ax.axvline(x=boundaries[i], color='r')
                ax.text(boundaries[i], 0.2, str(count))
                count += 1
        
        plot_ax = ax  # Store reference to main plotting axis
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    return fig, plot_ax  # Return both the figure and the main plotting axis