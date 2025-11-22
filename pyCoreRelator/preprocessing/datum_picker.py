"""
Core datum picking functions for pyCoreRelator.

Included Functions:
- onclick_boundary: Handle mouse click events for interactive boundary picking
- get_category_color: Return color based on category for visualization
- onkey_boundary: Handle keyboard events for interactive boundary picking
- create_interactive_figure: Create interactive plot with core images and log curves
- pick_stratigraphic_levels: Create interactive plot for picking stratigraphic levels
- interpret_bed_names: Interactive widget for naming picked stratigraphic beds

This module provides interactive tools for manually picking stratigraphic boundaries
and datum levels in geological core data, with support for multiple categories and
real-time visualization.
"""

# Data manipulation and analysis
import numpy as np
import pandas as pd
import os

# Visualization
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import LinearSegmentedColormap

# Image handling
from PIL import Image

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Global variables used by the interactive functions
fig = None
selection_complete = [False]

def onclick_boundary(event, xs, lines, ax, toolbar, categories, current_category, status_text, 
                     action_history, is_new_pick, loaded_count):
    """
    Handle mouse click events: left-click to add x value and vertical line.
    
    This function processes left mouse clicks to add depth values and corresponding
    vertical lines to the interactive plot. Only active when toolbar is not being used
    and selection is not complete.
    
    Parameters
    ----------
    event : matplotlib event object
        Mouse click event containing position and button information
    xs : list
        List to store x-coordinate values of clicked points
    lines : list
        List to store matplotlib line objects for visualization
    ax : matplotlib.axes.Axes
        The axes object where the clicking occurs
    toolbar : matplotlib toolbar object
        Navigation toolbar to check if any tools are active
    categories : list
        List to store category values for each clicked point
    current_category : list
        Single-element list containing the current category value
    status_text : matplotlib.text.Text
        Text object for displaying status messages
    action_history : list
        List to store history of user actions
    is_new_pick : list
        List to track which picks are new (True) vs loaded (False)
    loaded_count : int
        Number of data points loaded from existing file
        
    Returns
    -------
    None
        Modifies input lists and plot in place
        
    Example
    -------
    >>> # Used internally within pick_stratigraphic_levels function
    >>> # Users typically don't call this directly
    """
    global fig, selection_complete
    if event.inaxes == ax and event.name == 'button_press_event':
        # Check if any toolbar buttons are active
        if toolbar.mode == '' and not selection_complete[0]:  # No buttons pressed and selection not complete
            if event.button == 1:  # Left mouse button
                x1 = event.xdata
                xs.append(x1)
                categories.append(current_category[0])
                is_new_pick.append(True)
                
                # Add a vertical red dashed line at the clicked x position
                # Use different colors based on category
                color = get_category_color(current_category[0])
                line = ax.axvline(x=x1, color=color, linestyle='--')
                lines.append(line)
                
                # Record action in history
                action_history.append(f'ADD: depth={x1:.2f} cm, category={current_category[0]}')
                
                # Update status text
                status_text.set_text(r'$\mathbf{STATUS:}$ ' + f'Added x={x1:.2f}, Category={current_category[0]}')
                
                # Force immediate update
                ax.figure.canvas.draw()
                ax.figure.canvas.flush_events() 


def get_category_color(category):
    """
    Return a color based on the category number.
    
    This function maps category identifiers to specific colors for consistent
    visualization of different stratigraphic units or boundary types.
    
    Parameters
    ----------
    category : str or int
        Category identifier (can be string or numeric)
        
    Returns
    -------
    str
        Color string compatible with matplotlib (e.g., 'r', 'g', 'b')
        
    Example
    -------
    >>> get_category_color('1')
    'r'
    >>> get_category_color(2)
    'g'
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # Convert category to a hash value if it's a string
    if isinstance(category, str):
        category_hash = sum(ord(c) for c in category)
        return colors[category_hash % len(colors)]
    else:
        return colors[category % len(colors)]


def print_table_summary(xs, categories, loaded_count):
    """
    Print a formatted table summary of picked depths.
    
    Parameters
    ----------
    xs : list
        List of picked depth values
    categories : list
        List of category values
    loaded_count : int
        Number of originally loaded data points
    """
    if not xs:
        print("No picked depths.")
    else:
        # Create sorted data
        sorted_data = sorted(zip(xs, categories), key=lambda x: (x[1], x[0]))  # Sort by category, then depth
        
        print(f"Total: {len(xs)} picks")
        print("╔════╦══════════╦═══════════╗")
        print("║ #  ║ Category ║   Depth   ║")
        print("╠════╬══════════╬═══════════╣")
        
        prev_cat = None
        for i, (depth, cat) in enumerate(sorted_data):
            # Add separator between different categories
            if prev_cat is not None and cat != prev_cat:
                print("╠════╬══════════╬═══════════╣")
            
            print(f"║ {i+1:2d} ║    {cat:>2s}    ║  {depth:7.2f}  ║")
            prev_cat = cat
        
        print("╚════╩══════════╩═══════════╝") 


def onkey_boundary(event, xs, lines, ax, cid, toolbar, categories, current_category, picked_datum_csv, 
                   status_text, sort_csv, action_history, is_new_pick, loaded_count):
    """
    Handle keyboard events: delete to remove last point, enter to finish, esc to exit without saving, numbers 0-9 to change category.
    
    This function processes keyboard inputs for interactive boundary picking,
    including category changes, point removal, and completion of selection.
    
    Parameters
    ----------
    event : matplotlib event object
        Keyboard event containing key information
    xs : list
        List storing x-coordinate values of clicked points
    lines : list
        List storing matplotlib line objects for visualization
    ax : matplotlib.axes.Axes
        The axes object where the interaction occurs
    cid : list
        List containing connection IDs for event handlers
    toolbar : matplotlib toolbar object
        Navigation toolbar reference
    categories : list
        List storing category values for each clicked point
    current_category : list
        Single-element list containing the current category value
    picked_datum_csv : str
        Full path/filename for the output CSV file
    status_text : matplotlib.text.Text
        Text object for displaying status messages
    sort_csv : bool
        Whether to sort the CSV data by category then by picked_depths_cm
        when saving the results.
    action_history : list
        List to store history of user actions
    is_new_pick : list
        List to track which picks are new (True) vs loaded (False)
    loaded_count : int
        Number of data points loaded from existing file
        
    Returns
    -------
    None
        Modifies input lists and saves data when 'enter' is pressed
        
    Example
    -------
    >>> # Used internally within pick_stratigraphic_levels function
    >>> # Users typically don't call this directly
    """
    global fig, selection_complete
    if event.key and event.key in '0123456789':  # Check if key is a digit between 0-9
        # Change the current category
        current_category[0] = event.key
        status_text.set_text(r'$\mathbf{STATUS:}$ ' + f'Changed to Category {current_category[0]}')
        ax.figure.canvas.draw_idle()
    elif event.key in ('delete', 'backspace'):
        if xs and not selection_complete[0]:
            removed_x = xs.pop()
            removed_category = categories.pop()
            removed_line = lines.pop()
            removed_is_new = is_new_pick.pop()
            removed_line.remove()  # Remove the line from the plot
            
            # Record action in history
            action_history.append(f'REMOVE: depth={removed_x:.2f} cm, category={removed_category}')
            
            status_text.set_text(r'$\mathbf{STATUS:}$ ' + f'Removed x={removed_x:.2f}, Category={removed_category}')
            
            # Force immediate update
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
        else:
            status_text.set_text(r'$\mathbf{STATUS:}$ No points to remove.')
    elif event.key == 'escape':
        # Exit without saving
        fig.canvas.mpl_disconnect(cid[0])
        fig.canvas.mpl_disconnect(cid[1])
        selection_complete[0] = True
        ax.set_title("Selection Cancelled - No Changes Saved")
        status_text.set_text(r'$\mathbf{STATUS:}$ Exited without saving. No changes were made.')
        
        print("\n" + "="*60)
        print("CANCELLED: Exited without saving any changes.")
        print("="*60)
        
        # Clear the lists to indicate no changes should be returned
        xs.clear()
        categories.clear()
        is_new_pick.clear()
        
        ax.figure.canvas.draw_idle()
    elif event.key == 'enter':
        # Sort the picked depths and categories based on depth values (smallest to highest)
        if xs and categories:
            sorted_triplets = sorted(zip(xs, categories, is_new_pick), key=lambda t: (t[1], t[0]))
            xs[:] = [t[0] for t in sorted_triplets]
            categories[:] = [t[1] for t in sorted_triplets]
            is_new_pick[:] = [t[2] for t in sorted_triplets]
        
        # Disconnect the event handlers
        fig.canvas.mpl_disconnect(cid[0])
        fig.canvas.mpl_disconnect(cid[1])
        selection_complete[0] = True
        ax.set_title("Selection Completed")
        status_text.set_text(r'$\mathbf{STATUS:}$ Finished selecting points. Selection is now locked.')
        
        print("\n" + "="*60)
        print("FINISHED: Selection completed and saved.")
        print(f"Total actions performed: {len(action_history)}")
        print(f"Total picks: {len(xs)} ({len([p for p in is_new_pick if p])} new, {loaded_count} loaded)")
        if picked_datum_csv:
            print(f"Saved to: {picked_datum_csv}")
        print("="*60)
        
        # Print final table summary
        print("\nFinal picked depths:")
        print_table_summary(xs, categories, loaded_count)
        
        # Export to CSV if filename is provided and we have data
        if picked_datum_csv and xs:
            # Check if CSV file already exists and has other columns
            if os.path.exists(picked_datum_csv):
                try:
                    # Load existing CSV to preserve other columns
                    existing_df = pd.read_csv(picked_datum_csv)
                    
                    # Create new DataFrame with current picked data
                    new_df = pd.DataFrame({
                        'picked_depths_cm': xs,
                        'category': categories
                    })
                    
                    # Create DataFrame with new picked data, properly matching other columns by depth value
                    other_columns = [col for col in existing_df.columns if col not in ['picked_depths_cm', 'category']]
                    
                    if other_columns:
                        # Create a new DataFrame with the same column structure as existing file
                        df = pd.DataFrame(columns=existing_df.columns)
                        
                        # For each new picked depth, try to find matching data in existing CSV
                        for depth, cat in zip(xs, categories):
                            new_row = {}
                            new_row['picked_depths_cm'] = depth
                            new_row['category'] = cat
                            
                            # Try to find this exact depth in existing data to preserve other columns
                            matching_row = existing_df[existing_df['picked_depths_cm'] == depth]
                            
                            if not matching_row.empty:
                                # Found exact match - preserve other column data
                                for col in other_columns:
                                    new_row[col] = matching_row.iloc[0][col]
                            else:
                                # No exact match - set other columns to None
                                for col in other_columns:
                                    new_row[col] = None
                            
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        # No other columns, just use new data
                        df = new_df
                        
                except Exception as e:
                    print(f"Warning: Could not read existing CSV {picked_datum_csv}: {e}. Creating new file.")
                    df = pd.DataFrame({
                        'picked_depths_cm': xs,
                        'category': categories
                    })
            else:
                # Create new DataFrame for new file with interpreted_bed column
                df = pd.DataFrame({
                    'picked_depths_cm': xs,
                    'category': categories,
                    'interpreted_bed': [''] * len(xs)  # Empty column for future use
                })
            
            # Ensure interpreted_bed column exists (add if missing)
            if 'interpreted_bed' not in df.columns:
                df['interpreted_bed'] = ''
            
            # Sort the data if sort_csv is True (sort all columns together)
            if sort_csv:
                # Convert sorting columns to numeric types to ensure correct sorting
                df_sort = df.copy()
                df_sort['category'] = pd.to_numeric(df_sort['category'], errors='coerce')
                df_sort['picked_depths_cm'] = pd.to_numeric(df_sort['picked_depths_cm'], errors='coerce')
                # Drop rows with conversion issues
                valid_rows = df_sort.dropna(subset=['category', 'picked_depths_cm']).index
                df = df.iloc[valid_rows]
                df_sort = df_sort.iloc[valid_rows]
                # Sort all columns together based on category, then picked_depths_cm
                sort_order = df_sort.sort_values(by=['category', 'picked_depths_cm']).index
                df = df.iloc[sort_order].reset_index(drop=True)
                
            # Save to CSV
            df.to_csv(picked_datum_csv, index=False)
            sort_msg = " (sorted)" if sort_csv else ""
            status_text.set_text(r'$\mathbf{STATUS:}$ ' + f"Saved {len(df)} picked depths to {picked_datum_csv}{sort_msg}") 


def create_interactive_figure(md, log, core_img_1=None, core_img_2=None, miny=0, maxy=1, available_logs=None):
    """
    Create an interactive plot with core images and log curve.
    
    This function creates a matplotlib figure with subplots for core images and
    log data visualization, optimized for interactive boundary picking.
    
    Parameters
    ----------
    md : array-like
        Depth values for x-axis data
    log : array-like
        Log data for y-axis data (either 1D for single log or 2D for multiple logs)
    core_img_1 : numpy.ndarray, optional
        First core image data (e.g., RGB image)
    core_img_2 : numpy.ndarray, optional
        Second core image data (e.g., CT image)
    miny : float, default=0
        Minimum y-axis limit for log plot
    maxy : float, default=1
        Maximum y-axis limit for log plot
    available_logs : list, optional
        List of log names corresponding to columns in log data (for multi-log plotting)
        
    Returns
    -------
    tuple
        (figure, axes) - Matplotlib figure and the interactive axes object
        
    Example
    -------
    >>> fig, ax = create_interactive_figure(depth_data, log_data, rgb_img, ct_img)
    >>> plt.show()
    """
    global fig
    
    # Define color and style mapping for each column type (same as plot_core_data)
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
    
    if core_img_1 is not None and core_img_2 is not None and not isinstance(core_img_2, str):
        # Create figure with three subplots if both core images are provided
        # Log curve on top, images below
        fig, axs = plt.subplots(3, 1, figsize=(20, 5.5), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Log curve on top - handle both single and multi-log cases
        if log.ndim > 1 and log.shape[1] > 1:
            # Multi-log case
            for i in range(log.shape[1]):
                log_name = available_logs[i] if available_logs and i < len(available_logs) else f'Log {i}'
                if log_name in color_style_map:
                    color = color_style_map[log_name]['color']
                    linestyle = color_style_map[log_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                axs[0].plot(md, log[:, i], linestyle=linestyle, linewidth=0.7, color=color, label=log_name)
            axs[0].legend(loc='upper left', fontsize='small')
        else:
            # Single log case
            axs[0].plot(md, log, linestyle='-', linewidth=0.7, color='black')
        
        axs[0].set_ylim(miny, maxy)
        axs[0].set_xlim(md[0], md[-1])
        axs[0].set_ylabel('Normalized Values')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plot
        
        # First core image - flipping x & y axes
        axs[1].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
        axs[1].set_ylabel('Core\nImage 1')
        axs[1].set_xticks([])  # Hide x-axis ticks
        axs[1].set_yticks([])  # Hide y-axis ticks for image
        
        # Second core image - flipping x & y axes
        axs[2].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else 
                      core_img_2.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
        axs[2].set_ylabel('Core\nImage 2')
        axs[2].set_xlabel('Depth')
        axs[2].set_yticks([])  # Hide y-axis ticks for image
        
        plt.tight_layout()
        # Adjust top to make room for title
        plt.subplots_adjust(top=0.92)
        return fig, axs[0]  # Return the log plot axis for interaction
    
    elif core_img_1 is not None:
        # Create figure with two subplots if only first core image is provided
        # Log curve on top, image below
        fig, axs = plt.subplots(2, 1, figsize=(20, 4), gridspec_kw={'height_ratios': [2, 1]})
        
        # Log curve on top - handle both single and multi-log cases
        if log.ndim > 1 and log.shape[1] > 1:
            # Multi-log case
            for i in range(log.shape[1]):
                log_name = available_logs[i] if available_logs and i < len(available_logs) else f'Log {i}'
                if log_name in color_style_map:
                    color = color_style_map[log_name]['color']
                    linestyle = color_style_map[log_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                axs[0].plot(md, log[:, i], linestyle=linestyle, linewidth=0.7, color=color, label=log_name)
            axs[0].legend(loc='upper left', fontsize='small')
        else:
            # Single log case
            axs[0].plot(md, log, linestyle='-', linewidth=0.7, color='black')
        
        axs[0].set_ylim(miny, maxy)
        axs[0].set_xlim(md[0], md[-1])
        axs[0].set_ylabel('Normalized Values')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plot
        
        # First core image - flipping x & y axes
        axs[1].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
        axs[1].set_ylabel('Core\nImage 1')
        axs[1].set_xlabel('Depth')
        axs[1].set_yticks([])  # Hide y-axis ticks for image
        
        plt.tight_layout()
        # Adjust top to make room for title
        plt.subplots_adjust(top=0.92)
        return fig, axs[0]  # Return the log plot axis for interaction
    
    elif core_img_2 is not None and not isinstance(core_img_2, str):
        # Create figure with two subplots if only second core image is provided
        # Log curve on top, image below
        fig, axs = plt.subplots(2, 1, figsize=(20, 4), gridspec_kw={'height_ratios': [2, 1]})
        
        # Log curve on top - handle both single and multi-log cases
        if log.ndim > 1 and log.shape[1] > 1:
            # Multi-log case
            for i in range(log.shape[1]):
                log_name = available_logs[i] if available_logs and i < len(available_logs) else f'Log {i}'
                if log_name in color_style_map:
                    color = color_style_map[log_name]['color']
                    linestyle = color_style_map[log_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                axs[0].plot(md, log[:, i], linestyle=linestyle, linewidth=0.7, color=color, label=log_name)
            axs[0].legend(loc='upper left', fontsize='small')
        else:
            # Single log case
            axs[0].plot(md, log, linestyle='-', linewidth=0.7, color='black')
        
        axs[0].set_ylim(miny, maxy)
        axs[0].set_xlim(md[0], md[-1])
        axs[0].set_ylabel('Normalized Values')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plot
        
        # Second core image - flipping x & y axes
        axs[1].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else 
                      core_img_2.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
        axs[1].set_ylabel('Core\nImage 2')
        axs[1].set_xlabel('Depth')
        axs[1].set_yticks([])  # Hide y-axis ticks for image
        
        plt.tight_layout()
        # Adjust top to make room for title
        plt.subplots_adjust(top=0.92)
        return fig, axs[0]  # Return the log plot axis for interaction
    
    else:
        # Create figure with single subplot if no images or if core_img_2 is a string
        fig, ax = plt.subplots(figsize=(20, 2.5))
        
        # Log curve - handle both single and multi-log cases
        if log.ndim > 1 and log.shape[1] > 1:
            # Multi-log case
            for i in range(log.shape[1]):
                log_name = available_logs[i] if available_logs and i < len(available_logs) else f'Log {i}'
                if log_name in color_style_map:
                    color = color_style_map[log_name]['color']
                    linestyle = color_style_map[log_name]['linestyle']
                else:
                    color = f'C{i}'
                    linestyle = '-'
                ax.plot(md, log[:, i], linestyle=linestyle, linewidth=0.7, color=color, label=log_name)
            ax.legend(loc='upper left', fontsize='small')
        else:
            # Single log case
            ax.plot(md, log, linestyle='-', linewidth=0.7, color='black')
        
        ax.set_ylim(miny, maxy)
        ax.set_xlim(md[0], md[-1])
        ax.set_xlabel('depth')
        ax.set_ylabel('Normalized Intensity')
        
        plt.tight_layout()
        # Adjust top to make room for title
        plt.subplots_adjust(top=0.85)
        return fig, ax  # Return the log plot axis for interaction


def pick_stratigraphic_levels(md=None, log=None, core_img_1=None, core_img_2=None, core_name="", 
                             picked_datum_csv=None, sort_csv=True, 
                             log_data_dir=None, log_columns=None, depth_column='SB_DEPTH_cm'):
    """
    Create an interactive plot for picking stratigraphic levels.
    
    This is the main function that sets up an interactive matplotlib environment
    for manually picking stratigraphic boundaries and datum levels. Users can
    click to select points, categorize them, and save the results to CSV.
    If a CSV file already exists, it loads the existing data and allows users
    to continue picking from where they left off.
    
    The function can work in two modes:
    1. Direct mode: Pass md, log, and images directly (backward compatible)
    2. File mode: Pass file paths and let function load and process data
    
    Parameters
    ----------
    md : array-like, optional
        Depth values for x-axis data. If None, will load from log_data_dir.
    log : array-like, optional
        Log data for y-axis data (typically normalized 0-1). If None, will load from log_data_dir.
    core_img_1 : numpy.ndarray or str, optional
        First core image data as numpy array, or path to image file to load.
    core_img_2 : numpy.ndarray or str, optional
        Second core image data as numpy array, or path to image file to load.
    core_name : str, default=""
        Name of the core for display in plot title
    picked_datum_csv : str, optional
        Full path/filename for the output CSV file. If file exists, 
        data will be loaded from it.
    sort_csv : bool, default=True
        Whether to sort the CSV data by category then by picked_depths_cm
        when saving the results.
    log_data_dir : dict, optional
        Dictionary mapping log names to their file paths.
        Example: {'hiresMS': 'path/to/hiresMS.csv', 'CT': 'path/to/CT.csv'}
    log_columns : list, optional
        List of log column names to load from log_data_dir.
        Example: ['hiresMS', 'CT', 'Lumin']
    depth_column : str, default='SB_DEPTH_cm'
        Name of the depth column in the log files.
        
    Returns
    -------
    tuple
        (picked_depths, categories) - Lists of picked depth values and their categories
        
    Notes
    -----
    Interactive Controls:
    - Left-click: Add depth point
    - Number keys (0-9): Change current category
    - Delete/Backspace: Remove last point
    - Enter: Finish selection and save
    - Esc: Exit without saving any changes
    - Pan/Zoom tools: Temporarily disable point selection
    
    Features:
    - Displays a live-updating table below the plot showing all picked datums
    - New picks are marked with '*' in the table (not saved to CSV)
    - Prints action history for all user interactions (add, remove, category change)
    - Loads existing data from CSV if file exists
    - Sorts data by category then depth when saving
    
    Examples
    --------
    Direct mode (backward compatible):
    >>> depths, categories = pick_stratigraphic_levels(
    ...     measured_depth, log_data, 'path/to/img1.tiff', 'path/to/img2.tiff', 
    ...     core_name="Sample-01", picked_datum_csv="picked_depths.csv"
    ... )
    
    File mode (new):
    >>> depths, categories = pick_stratigraphic_levels(
    ...     core_name="M9907-23PC",
    ...     log_data_dir={'hiresMS': 'data/hiresMS.csv', 'CT': 'data/CT.csv'},
    ...     log_columns=['hiresMS', 'CT'],
    ...     core_img_1='data/RGB.tiff',
    ...     core_img_2='data/CT.tiff',
    ...     picked_datum_csv="pickeddepth/M9907-23PC_pickeddepth.csv"
    ... )
    """
    global fig, selection_complete
    
    # Set default picked_datum_csv if not provided
    if picked_datum_csv is None:
        if core_name:
            picked_datum_csv = f"{core_name}_pickeddepth.csv"
        else:
            picked_datum_csv = "pickeddepth.csv"
    
    # File mode: Load and process data from files
    if md is None or log is None:
        if log_data_dir is None or log_columns is None:
            raise ValueError("Either provide (md, log) directly or provide (log_data_dir, log_columns) to load from files")
        
        print(f"Loading data from files...")
        
        # Load log data from separate files
        dfs = {}
        for log_name in log_columns:
            if log_name not in log_data_dir:
                print(f"Warning: {log_name} not found in log_data_dir, skipping")
                continue
            try:
                df = pd.read_csv(log_data_dir[log_name])
                if log_name not in df.columns:
                    print(f"Skipping {log_name}: {log_name} column not found in file")
                    continue
                if depth_column not in df.columns:
                    print(f"Skipping {log_name}: {depth_column} column not found")
                    continue
                dfs[log_name] = df[[depth_column, log_name]]
            except Exception as e:
                print(f"Error loading {log_name}: {e}")
        
        if not dfs:
            raise ValueError("No data could be loaded from the specified log files")
        
        # Get list of successfully loaded logs
        available_logs = list(dfs.keys())
        print(f"Successfully loaded logs: {available_logs}")
        
        # Merge dataframes on depth column
        merged_df = dfs[available_logs[0]]
        for log_name in available_logs[1:]:
            merged_df = pd.merge(merged_df, dfs[log_name], on=depth_column, how='outer')
        
        # Sort by depth and handle missing values
        merged_df = merged_df.sort_values(by=depth_column).fillna(method='ffill').fillna(method='bfill')
        
        # Extract and normalize log data
        log_data_raw = np.array(merged_df[available_logs])
        log = (log_data_raw - np.min(log_data_raw, axis=0)) / (np.max(log_data_raw, axis=0) - np.min(log_data_raw, axis=0))
        md = np.array(merged_df[depth_column])
        
        print(f"Processed {len(md)} depth points across {len(available_logs)} logs")
        
        # Store available_logs for plotting
        log_names = available_logs
    else:
        # Direct mode - no log names available
        log_names = None
    
    # Load images if they are provided as file paths (strings)
    if isinstance(core_img_1, str):
        try:
            img_path = core_img_1
            core_img_1 = plt.imread(core_img_1)
            print(f"Loaded core image 1 from {img_path}")
        except Exception as e:
            print(f"Warning: Could not load core image 1: {e}")
            core_img_1 = None
    
    if isinstance(core_img_2, str):
        try:
            img_path = core_img_2
            core_img_2 = plt.imread(core_img_2)
            print(f"Loaded core image 2 from {img_path}")
        except Exception as e:
            print(f"Warning: Could not load core image 2: {e}")
            core_img_2 = None
    
    # Create figure and axes
    fig, ax = create_interactive_figure(md, log, core_img_1, core_img_2, 0, 1, available_logs=log_names)
    
    # Add title to the axes with bold keywords
    if core_name:
        ax.set_title(f'Pick datums for {core_name}: '
                 r'$\mathbf{ADD}$: Left-Click | '
                 r'$\mathbf{CHANGE\ CATEGORY}$: 0-9 | '
                 r'$\mathbf{REMOVE}$: Delete/Backspace | '
                 r'$\mathbf{FINISH}$: Enter/Return | '
                 r'$\mathbf{CANCEL}$: Esc', 
                 fontsize=11, pad=10)
    else:
        ax.set_title('Pick datums: '
                 r'$\mathbf{ADD}$: Left-Click | '
                 r'$\mathbf{CHANGE\ CATEGORY}$: 0-9 | '
                 r'$\mathbf{REMOVE}$: Delete/Backspace | '
                 r'$\mathbf{FINISH}$: Enter/Return | '
                 r'$\mathbf{CANCEL}$: Esc', 
                 fontsize=11, pad=10)
    
    # Determine status box position based on number of images
    if core_img_1 is not None and core_img_2 is not None and not isinstance(core_img_2, str):
        # Both images available
        status_y = 0.05
    elif core_img_1 is not None or (core_img_2 is not None and not isinstance(core_img_2, str)):
        # Only one image available
        status_y = 0.07
    else:
        # No images available
        status_y = 0.095
    
    # Add status text box below the curve using figure coordinates
    # Position it between the curve and images
    status_text = fig.text(0.035, status_y, r'$\mathbf{STATUS:}$ Ready for selection...', 
                          fontsize=10,
                          verticalalignment='top', horizontalalignment='left',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Lists to store picked values, lines, and categories
    xs = []
    lines = []
    categories = []
    is_new_pick = []  # Track which picks are new
    action_history = []  # Track user actions
    current_category = ['1']  # Default category is '1'
    selection_complete = [False]
    loaded_count = 0  # Track how many points were loaded from file
    
    # Load existing data from CSV if file exists
    if picked_datum_csv and os.path.exists(picked_datum_csv):
        try:
            existing_df = pd.read_csv(picked_datum_csv)
            if 'picked_depths_cm' in existing_df.columns and 'category' in existing_df.columns:
                # Load existing data
                loaded_depths = existing_df['picked_depths_cm'].tolist()
                loaded_categories = existing_df['category'].tolist()
                
                # Add loaded data to lists
                xs.extend(loaded_depths)
                categories.extend([str(cat) for cat in loaded_categories])  # Ensure categories are strings
                is_new_pick.extend([False] * len(loaded_depths))  # Mark as loaded (not new)
                loaded_count = len(loaded_depths)
                
                # Display loaded data as lines on the plot
                for depth, category in zip(loaded_depths, loaded_categories):
                    color = get_category_color(str(category))
                    line = ax.axvline(x=depth, color=color, linestyle='--')
                    lines.append(line)
                
                # Update status text
                status_text.set_text(r'$\mathbf{STATUS:}$ ' + f'Loaded {len(loaded_depths)} existing points from {os.path.basename(picked_datum_csv)}')
                print(f"\nLOADED: {len(loaded_depths)} existing data points from {picked_datum_csv}")
                
                # Show loaded data in table format
                print_table_summary(xs, categories, loaded_count)
            else:
                print(f"Warning: CSV file {picked_datum_csv} exists but doesn't contain required columns 'picked_depths_cm' and 'category'")
        except Exception as e:
            print(f"Warning: Could not load existing data from {picked_datum_csv}: {e}")
            status_text.set_text(r'$\mathbf{STATUS:}$ ' + f'Could not load existing data: {e}')
    elif picked_datum_csv:
        print(f"\nCSV file {picked_datum_csv} does not exist yet. Starting fresh.")
        status_text.set_text(r'$\mathbf{STATUS:}$ Starting fresh selection...')
    
    # Display instructions BEFORE showing the figure
    print("\n" + "="*60)
    print("INTERACTIVE SELECTION INSTRUCTIONS:")
    print("="*60)
    print(" - Press number keys (0-9) to change category (default: 1)")
    print(" - Left-click on the plot to add a depth point")
    print(" - Press Delete/Backspace to remove the last point")
    print(" - Press Enter to finish and save all changes")
    print(" - Press Esc to exit without saving any changes")
    print(" - Pan and Zoom tools will temporarily disable point selection")
    if picked_datum_csv:
        if os.path.exists(picked_datum_csv):
            print(f" - Changes will be saved to: {picked_datum_csv}")
        else:
            print(f" - New file will be created: {picked_datum_csv}")
    print("="*60 + "\n")
    
    # Get the toolbar instance
    toolbar = fig.canvas.toolbar
    
    # Show the plot (no figure title)
    plt.show()
    
    # Connect both click and keyboard events to their handlers
    cid = [
        fig.canvas.mpl_connect('button_press_event', 
                              lambda event: onclick_boundary(event, xs, lines, ax, toolbar, categories, current_category, 
                                                            status_text, action_history, is_new_pick, loaded_count)),
        fig.canvas.mpl_connect('key_press_event', 
                              lambda event: onkey_boundary(event, xs, lines, ax, cid, toolbar, categories, current_category, 
                                                          picked_datum_csv, status_text, sort_csv, action_history, is_new_pick, 
                                                          loaded_count))
    ]
    
    # Return the picked (and sorted) values
    return xs, categories


def interpret_bed_names(picked_datum_csv, core_name="", 
                        log_data_dir=None, log_columns=None, depth_column='SB_DEPTH_cm',
                        core_img_1=None, core_img_2=None):
    """
    Interactive widget for naming picked stratigraphic beds.
    
    This function provides an interactive Jupyter widget interface for assigning
    names to previously picked stratigraphic datums. It loads the picked depths
    from a CSV file, displays the core data with marked boundaries, and allows
    users to interactively name each bed.
    
    Parameters
    ----------
    picked_datum_csv : str
        Path to the CSV file containing picked depths (required)
    core_name : str, default=""
        Name of the core for display
    log_data_dir : dict, optional
        Dictionary mapping log names to their file paths.
        Example: {'hiresMS': 'path/to/hiresMS.csv', 'CT': 'path/to/CT.csv'}
    log_columns : list, optional
        List of log column names to load from log_data_dir.
        Example: ['hiresMS', 'CT', 'Lumin']
    depth_column : str, default='SB_DEPTH_cm'
        Name of the depth column in the log files.
    core_img_1 : numpy.ndarray or str, optional
        First core image data as numpy array, or path to image file to load.
    core_img_2 : numpy.ndarray or str, optional
        Second core image data as numpy array, or path to image file to load.
        
    Returns
    -------
    None
        Updates the CSV file in place with interpreted bed names
        
    Notes
    -----
    Requires Jupyter environment with ipywidgets installed.
    The function creates interactive widgets for:
    - Selecting rows by depth and category
    - Entering bed names
    - Updating individual rows
    - Saving all changes
    
    Examples
    --------
    >>> interpret_bed_names(
    ...     picked_datum_csv='pickeddepth/M9907-23PC_pickeddepth.csv',
    ...     core_name="M9907-23PC",
    ...     log_data_dir={'hiresMS': 'data/hiresMS.csv', 'CT': 'data/CT.csv'},
    ...     log_columns=['hiresMS', 'CT'],
    ...     core_img_1='data/RGB.tiff',
    ...     core_img_2='data/CT.tiff'
    ... )
    """
    # Import required modules
    try:
        from ipywidgets import interact, widgets, VBox, HBox, Button, Output
        from IPython.display import display, clear_output
    except ImportError:
        print("Error: This function requires ipywidgets. Install with: pip install ipywidgets")
        return
    
    # Check if CSV file exists (picked_datum_csv is now required)
    if not os.path.exists(picked_datum_csv):
        print(f"Error: File '{picked_datum_csv}' not found.")
        print(f"Please ensure you have:")
        print(f"  1. Run pick_stratigraphic_levels() to create the picked depths CSV")
        print(f"  2. Provided the correct path to the CSV file")
        return
    
    # Load picked depths from CSV
    df = pd.read_csv(picked_datum_csv)
    
    # Add 'interpreted_bed' column if it doesn't exist
    if 'interpreted_bed' not in df.columns:
        df['interpreted_bed'] = ''
    
    # Handle NaN values by converting them to empty strings
    df['interpreted_bed'] = df['interpreted_bed'].fillna('')
    
    print(f"Interactive datum naming for {core_name if core_name else 'core'}")
    print("Current data:")
    display(df[['category', 'picked_depths_cm', 'interpreted_bed']])
    
    # Load log data from files
    if log_data_dir is None or log_columns is None:
        print("Error: log_data_dir and log_columns are required")
        return
    
    print(f"\nLoading data from files...")
    
    # Load log data from separate files
    dfs = {}
    for log_name in log_columns:
        if log_name not in log_data_dir:
            print(f"Warning: {log_name} not found in log_data_dir, skipping")
            continue
        try:
            log_df = pd.read_csv(log_data_dir[log_name])
            if log_name not in log_df.columns:
                print(f"Skipping {log_name}: {log_name} column not found in file")
                continue
            if depth_column not in log_df.columns:
                print(f"Skipping {log_name}: {depth_column} column not found")
                continue
            dfs[log_name] = log_df[[depth_column, log_name]]
        except Exception as e:
            print(f"Error loading {log_name}: {e}")
    
    if not dfs:
        print("Error: No data could be loaded from the specified log files")
        return
    
    # Get list of successfully loaded logs
    available_logs = list(dfs.keys())
    print(f"Successfully loaded logs: {available_logs}")
    
    # Merge dataframes on depth column
    merged_df = dfs[available_logs[0]]
    for log_name in available_logs[1:]:
        merged_df = pd.merge(merged_df, dfs[log_name], on=depth_column, how='outer')
    
    # Sort by depth and handle missing values
    merged_df = merged_df.sort_values(by=depth_column).fillna(method='ffill').fillna(method='bfill')
    
    # Extract and normalize log data
    log_data_raw = np.array(merged_df[available_logs])
    log_data = (log_data_raw - np.min(log_data_raw, axis=0)) / (np.max(log_data_raw, axis=0) - np.min(log_data_raw, axis=0))
    measured_depth = np.array(merged_df[depth_column])
    
    print(f"Processed {len(measured_depth)} depth points across {len(available_logs)} logs")
    
    # Load images if they are provided as file paths (strings)
    if isinstance(core_img_1, str):
        try:
            core_img_1 = plt.imread(core_img_1)
            print(f"Loaded core image 1")
        except Exception as e:
            print(f"Warning: Could not load core image 1: {e}")
            core_img_1 = None
    
    if isinstance(core_img_2, str):
        try:
            core_img_2 = plt.imread(core_img_2)
            print(f"Loaded core image 2")
        except Exception as e:
            print(f"Warning: Could not load core image 2: {e}")
            core_img_2 = None
    
    # Prepare picked points data
    picked_points = list(zip(df['picked_depths_cm'].values.tolist(), 
                            df['category'].values.tolist()))
    picked_uncertainty = [1] * len(picked_points)
    
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
    
    # Define color and style mapping for each column type (same as plot_core_data)
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
    
    # Create figure following pick_stratigraphic_levels style (curve on top, images below)
    if core_img_1 is not None and core_img_2 is not None:
        # Both images available - 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(20, 5.5), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Log curve on top with color/style mapping
        for i in range(log_data.shape[1]):
            log_name = available_logs[i] if i < len(available_logs) else f'Log {i}'
            if log_name in color_style_map:
                color = color_style_map[log_name]['color']
                linestyle = color_style_map[log_name]['linestyle']
            else:
                color = f'C{i}'
                linestyle = '-'
            axs[0].plot(measured_depth, log_data[:, i], linestyle=linestyle, linewidth=0.7, 
                       color=color, label=log_name)
        axs[0].set_ylabel('Normalized Values')
        axs[0].set_xlim(measured_depth[0], measured_depth[-1])
        axs[0].set_xticks([])
        axs[0].legend(loc='upper left', fontsize='small')
        
        # Core image 1
        axs[1].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[measured_depth[0], measured_depth[-1], 0, 1])
        axs[1].set_ylabel('Core\nImage 1')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        # Core image 2
        axs[2].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else core_img_2.transpose(), 
                     aspect='auto', extent=[measured_depth[0], measured_depth[-1], 0, 1], cmap='gray')
        axs[2].set_ylabel('Core\nImage 2')
        axs[2].set_xlabel('Depth')
        axs[2].set_yticks([])
        
        plot_ax = axs[0]
        
    elif core_img_1 is not None:
        # Only first image - 2 subplots
        fig, axs = plt.subplots(2, 1, figsize=(20, 4), gridspec_kw={'height_ratios': [2, 1]})
        
        # Log curve on top with color/style mapping
        for i in range(log_data.shape[1]):
            log_name = available_logs[i] if i < len(available_logs) else f'Log {i}'
            if log_name in color_style_map:
                color = color_style_map[log_name]['color']
                linestyle = color_style_map[log_name]['linestyle']
            else:
                color = f'C{i}'
                linestyle = '-'
            axs[0].plot(measured_depth, log_data[:, i], linestyle=linestyle, linewidth=0.7,
                       color=color, label=log_name)
        axs[0].set_ylabel('Normalized Values')
        axs[0].set_xlim(measured_depth[0], measured_depth[-1])
        axs[0].set_xticks([])
        axs[0].legend(loc='upper left', fontsize='small')
        
        # Core image 1
        axs[1].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[measured_depth[0], measured_depth[-1], 0, 1])
        axs[1].set_ylabel('Core\nImage 1')
        axs[1].set_xlabel('Depth')
        axs[1].set_yticks([])
        
        plot_ax = axs[0]
        
    elif core_img_2 is not None:
        # Only second image - 2 subplots
        fig, axs = plt.subplots(2, 1, figsize=(20, 4), gridspec_kw={'height_ratios': [2, 1]})
        
        # Log curve on top with color/style mapping
        for i in range(log_data.shape[1]):
            log_name = available_logs[i] if i < len(available_logs) else f'Log {i}'
            if log_name in color_style_map:
                color = color_style_map[log_name]['color']
                linestyle = color_style_map[log_name]['linestyle']
            else:
                color = f'C{i}'
                linestyle = '-'
            axs[0].plot(measured_depth, log_data[:, i], linestyle=linestyle, linewidth=0.7,
                       color=color, label=log_name)
        axs[0].set_ylabel('Normalized Values')
        axs[0].set_xlim(measured_depth[0], measured_depth[-1])
        axs[0].set_xticks([])
        axs[0].legend(loc='upper left', fontsize='small')
        
        # Core image 2
        axs[1].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else core_img_2.transpose(),
                     aspect='auto', extent=[measured_depth[0], measured_depth[-1], 0, 1], cmap='gray')
        axs[1].set_ylabel('Core\nImage 2')
        axs[1].set_xlabel('Depth')
        axs[1].set_yticks([])
        
        plot_ax = axs[0]
        
    else:
        # No images - single subplot
        fig, ax = plt.subplots(figsize=(20, 2.5))
        
        # Log curve with color/style mapping
        for i in range(log_data.shape[1]):
            log_name = available_logs[i] if i < len(available_logs) else f'Log {i}'
            if log_name in color_style_map:
                color = color_style_map[log_name]['color']
                linestyle = color_style_map[log_name]['linestyle']
            else:
                color = f'C{i}'
                linestyle = '-'
            ax.plot(measured_depth, log_data[:, i], linestyle=linestyle, linewidth=0.7,
                   color=color, label=log_name)
        ax.set_ylabel('Normalized Values')
        ax.set_xlabel('Depth')
        ax.set_xlim(measured_depth[0], measured_depth[-1])
        ax.legend(loc='upper left', fontsize='small')
        
        plot_ax = ax
    
    # Add colored uncertainty shading and boundaries to the plot axis
    for (depth, category), uncertainty in zip(picked_points, picked_uncertainty):
        color = category_colors.get(int(category), 'red')
        # Add transparent shading covering the uncertainty interval
        plot_ax.axvspan(depth - uncertainty, 
                       depth + uncertainty, 
                       color=color, 
                       alpha=0.1)
        # Add the picked depth line on top (thinner linewidth for consistency)
        plot_ax.axvline(x=depth, 
                       color=color, 
                       linestyle='--', 
                       linewidth=0.8, 
                       label=f'#{int(category)}' if f'#{int(category)}' not in plot_ax.get_legend_handles_labels()[1] else "")
    
    # Update legend with unique category entries
    handles, labels = plot_ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plot_ax.legend(by_label.values(), 
                  by_label.keys(), 
                  loc='upper left', 
                  ncol=len(by_label))
    
    # Add title
    title_text = f"{core_name} with {len(picked_points)} Picked Boundaries" if core_name else f"Core with {len(picked_points)} Picked Boundaries"
    plot_ax.set_title(title_text)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Create interactive widgets for datum naming
    row_selector = widgets.Dropdown(
        options=[(f"Row {i}: Depth {df.loc[i, 'picked_depths_cm']:.2f} cm, Cat {df.loc[i, 'category']}", i) 
                for i in range(len(df))],
        description='Select Row:',
        style={'description_width': 'initial'}
    )
    
    name_input = widgets.Text(
        description='Bed Name:',
        placeholder='Enter bed name (e.g., Bed_A)'
    )
    
    update_button = widgets.Button(
        description='Update Name',
        button_style='info'
    )
    
    save_button = widgets.Button(
        description='Save All Changes',
        button_style='success'
    )
    
    output_area = widgets.Output()
    
    # Event handlers
    def on_row_change(change):
        if change['new'] is not None:
            current_name = df.loc[change['new'], 'interpreted_bed']
            name_input.value = str(current_name) if pd.notna(current_name) else ''
    
    def on_update_click(b):
        row_idx = row_selector.value
        bed_name = name_input.value.strip()
        
        if row_idx is not None:
            df.loc[row_idx, 'interpreted_bed'] = bed_name if bed_name else ''
            with output_area:
                clear_output()
                print(f"Updated row {row_idx} with name '{bed_name}'")
                print(f"Depth: {df.loc[row_idx, 'picked_depths_cm']:.2f} cm, Category: {df.loc[row_idx, 'category']}")
        else:
            with output_area:
                clear_output()
                print("Please select a row")
    
    def on_save_click(b):
        # Save data to CSV
        df_to_save = df.copy()
        df_to_save.to_csv(picked_datum_csv, index=False)
        
        with output_area:
            clear_output()
            print(f"Saved updated data to {picked_datum_csv}")
            
            # Reload the data from the saved CSV file
            updated_df = pd.read_csv(picked_datum_csv)
            # Keep empty cells as empty strings instead of NaN
            updated_df['interpreted_bed'] = updated_df['interpreted_bed'].fillna('')
            print("Final data:")
            display(updated_df[['category', 'picked_depths_cm', 'interpreted_bed']])
            
            # Create figure with same style as initial plot
            if core_img_1 is not None and core_img_2 is not None:
                # Both images available - 3 subplots
                fig, axs = plt.subplots(3, 1, figsize=(20, 5.5), gridspec_kw={'height_ratios': [2, 1, 1]})
                
                # Log curve on top
                for i in range(log_data.shape[1]):
                    axs[0].plot(measured_depth, log_data[:, i], linestyle='-', linewidth=0.7,
                               label=available_logs[i] if i < len(available_logs) else f'Log {i}')
                axs[0].set_ylabel('Normalized Values')
                axs[0].set_xlim(measured_depth[0], measured_depth[-1])
                axs[0].set_xticks([])
                axs[0].legend(loc='upper left')
                
                # Core image 1
                axs[1].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[measured_depth[0], measured_depth[-1], 0, 1])
                axs[1].set_ylabel('Core\nImage 1')
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                
                # Core image 2
                axs[2].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else core_img_2.transpose(),
                             aspect='auto', extent=[measured_depth[0], measured_depth[-1], 0, 1], cmap='gray')
                axs[2].set_ylabel('Core\nImage 2')
                axs[2].set_xlabel('Depth')
                axs[2].set_yticks([])
                
                final_plot_ax = axs[0]
                
            elif core_img_1 is not None:
                # Only first image - 2 subplots
                fig, axs = plt.subplots(2, 1, figsize=(20, 4), gridspec_kw={'height_ratios': [2, 1]})
                
                # Log curve on top
                for i in range(log_data.shape[1]):
                    axs[0].plot(measured_depth, log_data[:, i], linestyle='-', linewidth=0.7,
                               label=available_logs[i] if i < len(available_logs) else f'Log {i}')
                axs[0].set_ylabel('Normalized Values')
                axs[0].set_xlim(measured_depth[0], measured_depth[-1])
                axs[0].set_xticks([])
                axs[0].legend(loc='upper left')
                
                # Core image 1
                axs[1].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[measured_depth[0], measured_depth[-1], 0, 1])
                axs[1].set_ylabel('Core\nImage 1')
                axs[1].set_xlabel('Depth')
                axs[1].set_yticks([])
                
                final_plot_ax = axs[0]
                
            elif core_img_2 is not None:
                # Only second image - 2 subplots
                fig, axs = plt.subplots(2, 1, figsize=(20, 4), gridspec_kw={'height_ratios': [2, 1]})
                
                # Log curve on top
                for i in range(log_data.shape[1]):
                    axs[0].plot(measured_depth, log_data[:, i], linestyle='-', linewidth=0.7,
                               label=available_logs[i] if i < len(available_logs) else f'Log {i}')
                axs[0].set_ylabel('Normalized Values')
                axs[0].set_xlim(measured_depth[0], measured_depth[-1])
                axs[0].set_xticks([])
                axs[0].legend(loc='upper left')
                
                # Core image 2
                axs[1].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else core_img_2.transpose(),
                             aspect='auto', extent=[measured_depth[0], measured_depth[-1], 0, 1], cmap='gray')
                axs[1].set_ylabel('Core\nImage 2')
                axs[1].set_xlabel('Depth')
                axs[1].set_yticks([])
                
                final_plot_ax = axs[0]
                
            else:
                # No images - single subplot
                fig, ax = plt.subplots(figsize=(20, 2.5))
                
                # Log curve
                for i in range(log_data.shape[1]):
                    ax.plot(measured_depth, log_data[:, i], linestyle='-', linewidth=0.7,
                           label=available_logs[i] if i < len(available_logs) else f'Log {i}')
                ax.set_ylabel('Normalized Values')
                ax.set_xlabel('Depth')
                ax.set_xlim(measured_depth[0], measured_depth[-1])
                ax.legend(loc='upper left')
                
                final_plot_ax = ax
            
            # Add colored uncertainty shading and boundaries with bed names
            for idx, row in updated_df.iterrows():
                depth = row['picked_depths_cm']
                category = int(row['category'])
                bed_name = row['interpreted_bed']
                uncertainty = 1
                
                color = category_colors.get(category, 'red')
                
                # Add transparent shading
                final_plot_ax.axvspan(depth - uncertainty,
                                     depth + uncertainty,
                                     color=color,
                                     alpha=0.1)
                
                # Add the picked depth line
                final_plot_ax.axvline(x=depth,
                                     color=color,
                                     linestyle='--',
                                     linewidth=1.2,
                                     label=f'#{category}' if f'#{category}' not in final_plot_ax.get_legend_handles_labels()[1] else "")
                
                # Add bed name as text if provided
                if bed_name and str(bed_name).strip():
                    final_plot_ax.text(depth, final_plot_ax.get_ylim()[1] * 0.95,
                                      str(bed_name),
                                      rotation=90,
                                      verticalalignment='top',
                                      horizontalalignment='right',
                                      fontsize=9,
                                      fontweight='bold',
                                      color='black')
            
            # Update legend
            handles, labels = final_plot_ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            final_plot_ax.legend(by_label.values(),
                                by_label.keys(),
                                loc='upper left',
                                ncol=len(by_label))
            
            # Add title
            title_text = f"{core_name} with Named Boundaries" if core_name else "Core with Named Boundaries"
            final_plot_ax.set_title(title_text)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.show()
    
    # Connect event handlers
    row_selector.observe(on_row_change, names='value')
    update_button.on_click(on_update_click)
    save_button.on_click(on_save_click)
    
    # Display widgets
    controls = VBox([
        row_selector,
        name_input,
        HBox([update_button, save_button]),
        output_area
    ])
    
    display(controls) 