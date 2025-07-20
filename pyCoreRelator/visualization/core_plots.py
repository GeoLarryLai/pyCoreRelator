"""
Basic core plotting functions for core data visualization.

Included Functions:
- plot_core_data: Plot core data with optional RGB and CT images

This module provides basic plotting functionality for geological core data,
including log curves and optional RGB/CT images with category-based depth marking.
Compatible with both original and ML-processed core data.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_core_data(md, log, title, core_img_1=None, core_img_2=None, figsize=(20, 4), 
                  label_name=None, available_columns=None, is_multilog=False,
                  picked_depths=None, picked_categories=None, picked_uncertainties=None,
                  show_category=None, show_bed_number=False, cluster_data=None):
    """
    Plot core data with optional core images and support for multiple log types with category visualization.
    
    This function creates a comprehensive plot of core data including log curves and
    optional core images. It automatically adjusts the layout based on available
    images and supports both single and multiple log plotting with custom styling.
    Enhanced with category-based depth marking functionality.
    
    Parameters
    ----------
    md : array_like
        Array of depth values
    log : array_like
        Array of log values, either 1D for single log or 2D for multiple logs
    title : str
        Title for the plot
    core_img_1 : array_like, optional
        First core image array to display above the log curves
    core_img_2 : array_like, optional
        Second core image array to display above the log curves
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
    picked_depths : list, optional
        List of picked depths for category visualization
    picked_categories : list, optional
        List of categories corresponding to picked_depths
    picked_uncertainties : list, optional
        List of uncertainties for each picked depth
    show_category : list or None, default=None
        List of specific categories to show. If None, shows all available categories.
        If specified, only depths belonging to these categories will be plotted.
    show_bed_number : bool, default=False
        If True, displays bed numbers next to category depth lines in black bold text.
        Numbers are assigned based on depth order of the smallest category at each depth.
    cluster_data : dict, optional
        Dictionary containing cluster data with keys:
            'depth_vals': depth_values_array,
            'labels_vals': cluster_labels_array,
            'k': number_of_clusters
    
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
    >>> picked_depths = [25.5, 50.0, 75.2]
    >>> picked_categories = [1, 2, 1]
    >>> picked_uncertainties = [1.0, 1.5, 1.0]
    >>> fig, ax = plot_core_data(md, log, "Core 1", 
    ...                         available_columns=['MS', 'Lumin'], 
    ...                         is_multilog=True,
    ...                         picked_depths=picked_depths,
    ...                         picked_categories=picked_categories,
    ...                         picked_uncertainties=picked_uncertainties,
    ...                         show_category=[1],
    ...                         show_bed_number=True)
    """
    
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
    
    # Define colors for different categories
    category_colors = {
        1: 'red',
        2: 'blue',
        3: 'green',
        4: 'purple',
        5: 'orange',
        6: 'cyan',
        7: 'magenta',
        8: 'yellow',
        9: 'black'
    }
    
    # Define cluster colors if cluster data is provided
    if cluster_data is not None:
        cluster_colors = {
            0: (0.8, 0.2, 0.2),  # Red
            1: (0.2, 0.4, 0.8),  # Blue  
            2: (0.2, 0.7, 0.3),  # Green
            3: (0.8, 0.6, 0.2),  # Orange
            4: (0.6, 0.2, 0.8),  # Purple
            5: (0.8, 0.8, 0.2),  # Yellow
            6: (0.3, 0.8, 0.8),  # Cyan
            7: (0.8, 0.4, 0.6),  # Pink
            8: (0.5, 0.3, 0.1),  # Brown
            9: (0.7, 0.7, 0.7),  # Gray
        }
    
    # Use names from available_columns or create generic names
    if is_multilog and log.ndim > 1 and log.shape[1] > 1:
        column_names = available_columns if available_columns else [f"Log {i+1}" for i in range(log.shape[1])]
    
    # Calculate number of plots needed
    num_data_plots = 1  # Always have at least one plot for the log data
    total_plots = num_data_plots
    if cluster_data is not None:
        total_plots += 1  # Add one plot for cluster data
    
    # Setup figure layout based on available images
    if core_img_1 is not None and core_img_2 is not None:
        # Create figure with core images and data plots
        height_ratios = [1, 1] + [2]  # Two image rows, one data row
        fig, axs = plt.subplots(3, total_plots, figsize=figsize, 
                               gridspec_kw={'height_ratios': height_ratios, 'width_ratios': [1]*total_plots})
        
        # Handle axes indexing for multiple columns
        if total_plots == 1:
            axs = axs.reshape(-1, 1)  # Make it 2D for consistent indexing
        
        # First core image display - span all columns
        for col in range(total_plots):
            axs[0, col].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
            axs[0, col].set_xticks([])
            axs[0, col].set_yticks([])
            if col == 0:
                axs[0, col].set_ylabel('Core\nImage')
        
        # Second core image display - span all columns
        for col in range(total_plots):
            axs[1, col].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else 
                              core_img_2.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
            axs[1, col].set_xticks([])
            axs[1, col].set_yticks([])
            if col == 0:
                axs[1, col].set_ylabel('Core\nImage')
        
        # Set up data plot axes
        if cluster_data is not None:
            cluster_ax = axs[2, 0]
            plot_ax = axs[2, 1]
        else:
            plot_ax = axs[2, 0]
    
    elif core_img_1 is not None:
        # Create figure with one core image and data plots
        height_ratios = [1, 2]  # One image row, one data row
        fig, axs = plt.subplots(2, total_plots, figsize=figsize, 
                               gridspec_kw={'height_ratios': height_ratios, 'width_ratios': [1]*total_plots})
        
        # Handle axes indexing for multiple columns
        if total_plots == 1:
            axs = axs.reshape(-1, 1)  # Make it 2D for consistent indexing
        
        # Core image display - span all columns
        for col in range(total_plots):
            axs[0, col].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
            axs[0, col].set_xticks([])
            axs[0, col].set_yticks([])
            if col == 0:
                axs[0, col].set_ylabel('Core\nImage')
        
        # Set up data plot axes
        if cluster_data is not None:
            cluster_ax = axs[1, 0]
            plot_ax = axs[1, 1]
        else:
            plot_ax = axs[1, 0]
    
    elif core_img_2 is not None:
        # Create figure with second core image and data plots
        height_ratios = [1, 2]  # One image row, one data row
        fig, axs = plt.subplots(2, total_plots, figsize=figsize, 
                               gridspec_kw={'height_ratios': height_ratios, 'width_ratios': [1]*total_plots})
        
        # Handle axes indexing for multiple columns
        if total_plots == 1:
            axs = axs.reshape(-1, 1)  # Make it 2D for consistent indexing
        
        # Core image display - span all columns
        for col in range(total_plots):
            axs[0, col].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else 
                              core_img_2.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
            axs[0, col].set_xticks([])
            axs[0, col].set_yticks([])
            if col == 0:
                axs[0, col].set_ylabel('Core\nImage')
        
        # Set up data plot axes
        if cluster_data is not None:
            cluster_ax = axs[1, 0]
            plot_ax = axs[1, 1]
        else:
            plot_ax = axs[1, 0]
    
    else:
        # Create figure with data plots only
        if total_plots == 1:
            fig, plot_ax = plt.subplots(figsize=figsize)
        else:
            fig, axs = plt.subplots(1, total_plots, figsize=figsize, 
                                   gridspec_kw={'width_ratios': [1]*total_plots})
            cluster_ax = axs[0]
            plot_ax = axs[1]
    
    # Plot cluster data if provided
    if cluster_data is not None:
        depth_vals = cluster_data['depth_vals']
        labels_vals = cluster_data['labels_vals']
        k = cluster_data['k']
        
        # Use cluster colors
        colors = [cluster_colors[i] for i in range(k)]
        
        for i in range(len(depth_vals)-1):
            top = depth_vals[i]
            bottom = depth_vals[i+1]
            cluster_label = labels_vals[i]
            color = colors[cluster_label]
            
            # Create rectangle for cluster segment
            cluster_ax.fill_between([0, 1], [top, top], [bottom, bottom], 
                                   color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        cluster_ax.set_xlim(0, 1)
        cluster_ax.set_ylim(md[0], md[-1])
        cluster_ax.set_xlabel('Cluster\nLabels', fontweight='bold', fontsize='small')
        cluster_ax.set_xticks([])
        cluster_ax.invert_yaxis()
    
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
            
            plot_ax.plot(md, log[:, i], label=col_name, color=color, 
                      linestyle=linestyle, linewidth=0.7)
    else:
        plot_ax.plot(md, log, label=label_name, linewidth=0.7)
    
    plot_ax.set_ylim(0, 1)
    plot_ax.set_xlim(md[0], md[-1])
    plot_ax.set_xlabel('Depth')
    plot_ax.set_ylabel('Normalized Values')
    
    # Add category-based depth marking if data is provided
    if picked_depths is not None and picked_categories is not None:
        # Ensure uncertainties list exists
        if picked_uncertainties is None:
            picked_uncertainties = [1.0] * len(picked_depths)
        
        # Validate show_category parameter and filter data
        if show_category is not None:
            # Check if requested categories exist in the data
            available_categories = set(picked_categories)
            requested_categories = set(show_category)
            missing_categories = requested_categories - available_categories
            
            if missing_categories:
                print(f"Warning: Requested categories {list(missing_categories)} not found in data. "
                      f"Available categories: {list(available_categories)}")
            
            # Filter to show only requested categories that exist
            valid_categories = requested_categories & available_categories
            if not valid_categories:
                print("Error: None of the requested categories exist in the data. No category markers will be shown.")
            else:
                # Filter the data to only include requested categories
                filtered_data = [(depth, cat, unc) for depth, cat, unc in 
                               zip(picked_depths, picked_categories, picked_uncertainties) 
                               if cat in valid_categories]
                picked_depths_filtered, picked_categories_filtered, picked_uncertainties_filtered = zip(*filtered_data) if filtered_data else ([], [], [])
        else:
            # Show all categories
            picked_depths_filtered = picked_depths
            picked_categories_filtered = picked_categories
            picked_uncertainties_filtered = picked_uncertainties
        
        # Add colored uncertainty shading and boundaries
        for depth, category, uncertainty in zip(picked_depths_filtered, picked_categories_filtered, picked_uncertainties_filtered):
            color = category_colors.get(category, 'red')
            plot_ax.axvspan(depth - uncertainty, depth + uncertainty, color=color, alpha=0.1)
            plot_ax.axvline(x=depth, color=color, linestyle='--', linewidth=1.2,
                          label=f'#{category}' if f'#{category}' not in plot_ax.get_legend_handles_labels()[1] else "")
        
        # Add bed numbers if requested - only on smallest category per bed
        if show_bed_number and picked_depths_filtered:
            # Group by unique depths to identify beds
            unique_depths = sorted(set(picked_depths_filtered))
            
            # For each bed (unique depth), find the minimum category shown
            bed_numbers_to_show = {}
            for bed_number, depth in enumerate(unique_depths, 1):
                # Find all categories at this depth that are being shown
                categories_at_depth = [cat for d, cat in zip(picked_depths_filtered, picked_categories_filtered) if d == depth]
                if categories_at_depth:
                    min_category_at_depth = min(categories_at_depth)
                    bed_numbers_to_show[depth] = (bed_number, min_category_at_depth)
            
            # Display bed numbers only at depths with the minimum category
            for depth, category in zip(picked_depths_filtered, picked_categories_filtered):
                if depth in bed_numbers_to_show:
                    bed_number, min_category = bed_numbers_to_show[depth]
                    if category == min_category:
                        plot_ax.text(depth, 0.9, str(bed_number), fontsize=12, fontweight='bold', 
                                   color='black', ha='center', va='center')
                        # Remove from dict to avoid duplicate numbering
                        del bed_numbers_to_show[depth]
        
        # Separate legends for data curves and categories
        handles, labels = plot_ax.get_legend_handles_labels()
        
        # Split handles and labels into data curves and categories
        data_handles = []
        data_labels = []
        category_handles = []
        category_labels = []
        
        for handle, label in zip(handles, labels):
            if label.startswith('#'):
                category_handles.append(handle)
                category_labels.append(label)
            else:
                data_handles.append(handle)
                data_labels.append(label)
        
        # Create two separate legends
        if data_handles:
            # Place data curves legend in upper left
            leg1 = plot_ax.legend(data_handles, data_labels, loc='upper left', title="Data Curves", ncol=len(data_handles))
            plot_ax.add_artist(leg1)  # Make sure this legend is added to the plot
        if category_handles:
            # Place category legend in upper right
            plot_ax.legend(category_handles, category_labels, loc='upper right', title="Categories", ncol=len(category_handles))
            
        # Update title to include number of picked boundaries
        if show_category is not None:
            title += f" with {len(picked_depths_filtered)} Picked Boundaries (Categories: {list(valid_categories)})"
        else:
            title += f" with {len(picked_depths)} Picked Boundaries"
    else:
        # Original legend behavior for non-category plots
        if is_multilog or label_name is not None:
            plot_ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    return fig, plot_ax 