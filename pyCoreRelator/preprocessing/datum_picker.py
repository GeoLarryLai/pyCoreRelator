"""
Core datum picking functions for pyCoreRelator.

Included Functions:
- onclick_boundary: Handle mouse click events for interactive boundary picking
- get_category_color: Return color based on category for visualization
- onkey_boundary: Handle keyboard events for interactive boundary picking
- create_interactive_figure: Create interactive plot with core images and log curves
- pick_stratigraphic_levels: Create interactive plot for picking stratigraphic levels

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

def onclick_boundary(event, xs, lines, ax, toolbar, categories, current_category, status_text=None):
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
    status_text : matplotlib.text.Text, optional
        Text object for displaying status messages
        
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
                # Add a vertical red dashed line at the clicked x position
                # Use different colors based on category
                color = get_category_color(current_category[0])
                line = ax.axvline(x=x1, color=color, linestyle='--')
                lines.append(line)
                
                # Update status text instead of print
                if status_text:
                    status_text.set_text(f'Added x={x1:.2f}, Category={current_category[0]}')
                ax.figure.canvas.draw_idle() 


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


def onkey_boundary(event, xs, lines, ax, cid, toolbar, categories, current_category, csv_filename=None, status_text=None, sort_csv=True):
    """
    Handle keyboard events: delete to remove last point, enter to finish, numbers 0-9 to change category.
    
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
    csv_filename : str, optional
        Full path/filename for the output CSV file
    status_text : matplotlib.text.Text, optional
        Text object for displaying status messages
    sort_csv : bool, default=True
        Whether to sort the CSV data by category then by picked_depths_cm
        when saving the results.
        
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
        if status_text:
            status_text.set_text(f'Changed to Category {current_category[0]}')
        ax.figure.canvas.draw_idle()
    elif event.key in ('delete', 'backspace'):
        if xs and not selection_complete[0]:
            removed_x = xs.pop()
            removed_category = categories.pop()
            removed_line = lines.pop()
            removed_line.remove()  # Remove the line from the plot
            if status_text:
                status_text.set_text(f'Removed x={removed_x:.2f}, Category={removed_category}')
            ax.figure.canvas.draw_idle()
        else:
            if status_text:
                status_text.set_text('No points to remove.')
    elif event.key == 'enter':
        # Sort the picked depths and categories based on depth values (smallest to highest)
        if xs and categories:
            sorted_pairs = sorted(zip(xs, categories), key=lambda pair: (pair[1], pair[0]))
            xs[:] = [pair[0] for pair in sorted_pairs]
            categories[:] = [pair[1] for pair in sorted_pairs]
        # Disconnect the event handlers
        fig.canvas.mpl_disconnect(cid[0])
        fig.canvas.mpl_disconnect(cid[1])
        selection_complete[0] = True
        ax.set_title("Selection Completed")
        if status_text:
            status_text.set_text('Finished selecting points. Selection is now locked.')
        
        # Export to CSV if filename is provided and we have data
        if csv_filename and xs:
            # Check if CSV file already exists and has other columns
            if os.path.exists(csv_filename):
                try:
                    # Load existing CSV to preserve other columns
                    existing_df = pd.read_csv(csv_filename)
                    
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
                    print(f"Warning: Could not read existing CSV {csv_filename}: {e}. Creating new file.")
                    df = pd.DataFrame({
                        'picked_depths_cm': xs,
                        'category': categories
                    })
            else:
                # Create new DataFrame for new file
                df = pd.DataFrame({
                    'picked_depths_cm': xs,
                    'category': categories
                })
            
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
            df.to_csv(csv_filename, index=False)
            if status_text:
                sort_msg = " (sorted)" if sort_csv else ""
                status_text.set_text(f"Saved {len(df)} picked depths to {csv_filename}{sort_msg}") 


def create_interactive_figure(md, log, core_img_1=None, core_img_2=None, miny=0, maxy=1):
    """
    Create an interactive plot with core images and log curve.
    
    This function creates a matplotlib figure with subplots for core images and
    log data visualization, optimized for interactive boundary picking.
    
    Parameters
    ----------
    md : array-like
        Depth values for x-axis data
    log : array-like
        Log data for y-axis data
    core_img_1 : numpy.ndarray, optional
        First core image data (e.g., RGB image)
    core_img_2 : numpy.ndarray, optional
        Second core image data (e.g., CT image)
    miny : float, default=0
        Minimum y-axis limit for log plot
    maxy : float, default=1
        Maximum y-axis limit for log plot
        
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
    
    if core_img_1 is not None and core_img_2 is not None and not isinstance(core_img_2, str):
        # Create figure with three subplots if both core images are provided
        fig, axs = plt.subplots(3, 1, figsize=(20, 5.5), gridspec_kw={'height_ratios': [1, 1, 2]})
        
        # First core image - flipping x & y axes
        axs[0].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
        axs[0].set_ylabel('Core\nImage')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plots
        axs[0].set_yticks([])  # Hide y-axis ticks for image
        
        # Second core image - flipping x & y axes
        axs[1].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else 
                      core_img_2.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
        axs[1].set_ylabel('Core\nImage')
        axs[1].set_xticks([])  # Hide x-axis ticks for middle plot
        axs[1].set_yticks([])  # Hide y-axis ticks for image
        
        # Log curve
        axs[2].plot(md, log, linestyle='-', linewidth=0.7)  # Using thinner line width of 0.7
        axs[2].set_ylim(miny, maxy)
        axs[2].set_xlim(md[0], md[-1])
        axs[2].set_xlabel('depth')
        axs[2].set_ylabel('Normalized Values')
        axs[2].set_title('Interactive Selection: `Left-Click` to Add, `Delete/Backspace` to Remove, `Enter/Return` to Finish')
        
        plt.tight_layout()
        return fig, axs[2]  # Return the log plot axis for interaction
    
    elif core_img_1 is not None:
        # Create figure with two subplots if only first core image is provided
        fig, axs = plt.subplots(2, 1, figsize=(20, 4), gridspec_kw={'height_ratios': [1, 2]})
        
        # First core image - flipping x & y axes
        axs[0].imshow(core_img_1.transpose(1, 0, 2), aspect='auto', extent=[md[0], md[-1], 0, 1])
        axs[0].set_ylabel('Core\nImage')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plots
        axs[0].set_yticks([])  # Hide y-axis ticks for image
        
        # Log curve
        axs[1].plot(md, log, linestyle='-', linewidth=0.7)  # Using thinner line width of 0.7
        axs[1].set_ylim(miny, maxy)
        axs[1].set_xlim(md[0], md[-1])
        axs[1].set_xlabel('depth (cm)')
        axs[1].set_ylabel('Normalized Values')
        axs[1].set_title('Interactive Selection: `Left-Click` to Add, `Delete/Backspace` to Remove, `Enter/Return` to Finish')
        
        plt.tight_layout()
        return fig, axs[1]  # Return the log plot axis for interaction
    
    elif core_img_2 is not None and not isinstance(core_img_2, str):
        # Create figure with two subplots if only second core image is provided
        fig, axs = plt.subplots(2, 1, figsize=(20, 4), gridspec_kw={'height_ratios': [1, 2]})
        
        # Second core image - flipping x & y axes
        axs[0].imshow(core_img_2.transpose(1, 0, 2) if len(core_img_2.shape) == 3 else 
                      core_img_2.transpose(), aspect='auto', extent=[md[0], md[-1], 0, 1], cmap='gray')
        axs[0].set_ylabel('Core\nImage')
        axs[0].set_xticks([])  # Hide x-axis ticks for top plots
        axs[0].set_yticks([])  # Hide y-axis ticks for image
        
        # Log curve
        axs[1].plot(md, log, linestyle='-', linewidth=0.7)  # Using thinner line width of 0.7
        axs[1].set_ylim(miny, maxy)
        axs[1].set_xlim(md[0], md[-1])
        axs[1].set_xlabel('depth')
        axs[1].set_ylabel('Normalized Values')
        axs[1].set_title('Interactive Selection: `Left-Click` to Add, `Delete/Backspace` to Remove, `Enter/Return` to Finish')
        
        plt.tight_layout()
        return fig, axs[1]  # Return the log plot axis for interaction
    
    else:
        # Create figure with single subplot if no images or if core_img_2 is a string
        fig, ax = plt.subplots(figsize=(20, 2.5))
        
        # Log curve
        ax.plot(md, log, linestyle='-', linewidth=0.7)  # Using thinner line width of 0.7
        ax.set_ylim(miny, maxy)
        ax.set_xlim(md[0], md[-1])
        ax.set_xlabel('depth')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('Interactive Selection: `Left-Click` to Add, `Delete/Backspace` to Remove, `Enter/Return` to Finish')
        
        plt.tight_layout()
        return fig, ax  # Return the log plot axis for interaction


def pick_stratigraphic_levels(md, log, core_img_1=None, core_img_2=None, core_name="", csv_filename=None, sort_csv=True):
    """
    Create an interactive plot for picking stratigraphic levels.
    
    This is the main function that sets up an interactive matplotlib environment
    for manually picking stratigraphic boundaries and datum levels. Users can
    click to select points, categorize them, and save the results to CSV.
    If a CSV file already exists, it loads the existing data and allows users
    to continue picking from where they left off.
    
    Parameters
    ----------
    md : array-like
        Depth values for x-axis data
    log : array-like
        Log data for y-axis data (typically normalized 0-1)
    core_img_1 : numpy.ndarray, optional
        First core image data (e.g., RGB image)
    core_img_2 : numpy.ndarray, optional
        Second core image data (e.g., CT image)
    core_name : str, default=""
        Name of the core for display in plot title
    csv_filename : str, optional
        Full path/filename for the output CSV file. If file exists, 
        data will be loaded from it.
    sort_csv : bool, default=True
        Whether to sort the CSV data by category then by picked_depths_cm
        when saving the results.
        
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
    - Pan/Zoom tools: Temporarily disable point selection
    
    Example
    -------
    >>> depths, categories = pick_stratigraphic_levels(
    ...     measured_depth, log_data, rgb_img, ct_img, 
    ...     core_name="Sample-01", csv_filename="picked_depths.csv"
    ... )
    """
    global fig, selection_complete
    
    # Create figure and axes
    fig, ax = create_interactive_figure(md, log, core_img_1, core_img_2, 0, 1)
    
    # Add status text box
    status_text = ax.text(0.02, 0.98, 'Ready for selection...', 
                         transform=ax.transAxes, fontsize=10,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Lists to store picked values, lines, and categories
    xs = []
    lines = []
    categories = []
    current_category = ['1']  # Default category is '1'
    selection_complete = [False]
    
    # Load existing data from CSV if file exists
    if csv_filename and os.path.exists(csv_filename):
        try:
            existing_df = pd.read_csv(csv_filename)
            if 'picked_depths_cm' in existing_df.columns and 'category' in existing_df.columns:
                # Load existing data
                loaded_depths = existing_df['picked_depths_cm'].tolist()
                loaded_categories = existing_df['category'].tolist()
                
                # Add loaded data to lists
                xs.extend(loaded_depths)
                categories.extend([str(cat) for cat in loaded_categories])  # Ensure categories are strings
                
                # Display loaded data as lines on the plot
                for depth, category in zip(loaded_depths, loaded_categories):
                    color = get_category_color(str(category))
                    line = ax.axvline(x=depth, color=color, linestyle='--')
                    lines.append(line)
                
                # Update status text
                status_text.set_text(f'Loaded {len(loaded_depths)} existing points from {os.path.basename(csv_filename)}')
                print(f"Loaded {len(loaded_depths)} existing data points from {csv_filename}")
            else:
                print(f"Warning: CSV file {csv_filename} exists but doesn't contain required columns 'picked_depths_cm' and 'category'")
        except Exception as e:
            print(f"Warning: Could not load existing data from {csv_filename}: {e}")
            status_text.set_text(f'Could not load existing data: {e}')
    elif csv_filename:
        print(f"CSV file {csv_filename} does not exist yet. Starting fresh.")
        status_text.set_text('Starting fresh selection...')
    
    # Get the toolbar instance
    toolbar = fig.canvas.toolbar
    
    # Connect both click and keyboard events to their handlers
    cid = [
        fig.canvas.mpl_connect('button_press_event', 
                              lambda event: onclick_boundary(event, xs, lines, ax, toolbar, categories, current_category, status_text)),
        fig.canvas.mpl_connect('key_press_event', 
                              lambda event: onkey_boundary(event, xs, lines, ax, cid, toolbar, categories, current_category, csv_filename, status_text, sort_csv))
    ]
    
    # Display instructions (these should show up)
    print("Instructions:")
    print(" - Press any number key (0-9) to change the category of subsequent selections. The default category is set to '1'")
    print(" - Left-click on the plot to select and save an x-value.")
    print(" - Press Delete/Backspace to undo the last selection.")
    print(" - Press Enter when finished selecting points.")
    print(" - Pan and Zoom tools will temporarily disable point selection.")
    if csv_filename:
        if os.path.exists(csv_filename):
            print(f" - Picked depths will be updated and saved to: {csv_filename}")
        else:
            print(f" - Picked depths will be saved to: {csv_filename}")
    
    fig.suptitle(f'{core_name}', fontsize=16, y=1.02)
    plt.show()
    
    # Return the picked (and sorted) values
    return xs, categories 