"""
Machine Learning log data gap filling plotting functions for pyCoreRelator.

Included Functions:
 by cleaning and scaling depth values using configurable parameters
- plot_core_logs: Plot core logs using fully configurable parameters from data_config
- plot_filled_data: Plot original and ML-filled data for a given log using configurable parameters
 for ML training using configurable parameters
 using configurable parameters from data_config
 for gap rows to blend with linear interpolation

 using specified ML method
 using ML methods with fully configurable parameters

This module provides comprehensive machine learning tools for filling data gaps in
geological core log data, with support for multiple ML algorithms and configurable
preprocessing parameters.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll  # For PolyCollection
import numpy as np
import os
import warnings
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# Create interaction features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from joblib import Parallel, delayed

def plot_core_logs(data_config, file_type='clean', title=None, pickeddepth_csv=None, 
                   save_fig=False, output_dir=None, fig_format=['png'], dpi=None):
    """
    Plot core logs using fully configurable parameters from data_config.
    
    This function creates subplot panels for different types of core data (images and logs)
    based on the configuration provided. All plotting decisions are driven by column_configs content.
    
    Parameters
    ----------
    data_config : dict
        Configuration dictionary containing:
        - depth_column: Primary depth column name
        - column_configs: Dictionary of data type configurations
        - mother_dir: Base directory path
        - clean_output_folder or filled_output_folder: Data folder path
        - clean_file_paths or filled_file_paths: Dictionary of file paths by data type
        - core_length: Core length for y-axis limits
        - core_name: Core name for title
    file_type : str, default='clean'
        Type of data files to plot ('clean' or 'filled')
    title : str, optional
        Custom title for the plot. If None, generates default title
    pickeddepth_csv : str, optional
        Path to CSV file containing picked depths. If None, no picked depths
        column will be displayed.
    save_fig : bool, default=False
        Whether to save the figure to disk
    output_dir : str, optional
        Directory path where the figure will be saved. Required if save_fig=True.
    fig_format : list of str, default=['png']
        List of file formats to save the figure. Supported formats: 'png', 'jpg', 
        'jpeg', 'svg', 'pdf'. Multiple formats can be specified.
    dpi : int or None, default=None
        Resolution in dots per inch for saved figures. If None, uses matplotlib's 
        default dpi. If specified (e.g., 300), applies to all formats in fig_format.
        
    Returns
    -------
    tuple
        (fig, axes) - matplotlib figure and axes objects
        
    Notes
    -----
    The function automatically determines plot structure based on column_configs and
    available data files. Supports both image and data plotting panels.
    For file_type='filled', plots both cleaned data (solid lines on top) and filled 
    data (dashed lines) for comparison.
    If pickeddepth_csv is provided, adds a thin leftmost column showing horizontal
    lines at picked depths with category-based styling.
    """
    # Get primary depth column from config
    depth_col = data_config['depth_column']
    
    # Get file paths based on type
    if file_type == 'clean':
        data_paths = data_config.get('clean_file_paths', {})
        output_folder = data_config['clean_output_folder']
        clean_data = None
    else:
        data_paths = data_config.get('filled_file_paths', {})
        output_folder = data_config['filled_output_folder']
        
        # Load cleaned data for comparison when plotting filled data
        clean_paths = data_config.get('clean_file_paths', {})
        clean_output_folder = data_config['clean_output_folder']
        clean_data = {}
        for data_type in clean_paths.keys():
            clean_full_path = data_config['mother_dir'] + clean_output_folder + clean_paths[data_type]
            if os.path.exists(clean_full_path):
                loaded_clean_data = pd.read_csv(clean_full_path)
                if depth_col in loaded_clean_data.columns:
                    clean_data[data_type] = loaded_clean_data
    
    # Get available column configs
    available_columns = data_config.get('column_configs', {})
    
    # Only process data types that have both file path and column config
    valid_data_types = set(data_paths.keys()) & set(available_columns.keys())
    
    # Build full file paths and load data
    data = {}
    for data_type in valid_data_types:
        full_path = data_config['mother_dir'] + output_folder + data_paths[data_type]
        if os.path.exists(full_path):
            loaded_data = pd.read_csv(full_path)
            if depth_col in loaded_data.columns:
                data[data_type] = loaded_data
    
    if not data:
        raise ValueError("No valid data files found to plot")
    
    # Load Core Length and Name
    core_length = data_config['core_length']
    core_name = data_config['core_name']
    
    # Load picked depths data only if CSV path is explicitly provided
    picked_depths_data = None
    if pickeddepth_csv is not None:
        # Use the provided path directly without combining with mother_dir
        if os.path.exists(pickeddepth_csv):
            try:
                picked_depths_data = pd.read_csv(pickeddepth_csv)
                print(f"Successfully loaded picked depths CSV with {len(picked_depths_data)} rows")
                # Validate required columns
                if 'picked_depths_cm' not in picked_depths_data.columns or 'category' not in picked_depths_data.columns:
                    print(f"Warning: Required columns 'picked_depths_cm' or 'category' not found in {pickeddepth_csv}")
                    picked_depths_data = None
            except Exception as e:
                print(f"Warning: Could not load picked depths CSV {pickeddepth_csv}: {e}")
                picked_depths_data = None
        else:
            print(f"Warning: Picked depths CSV file not found: {pickeddepth_csv}")
    
    if title is None:
        file_type_title = 'Cleaned' if file_type == 'clean' else 'ML-Filled'
        title = f'{core_name} {file_type_title} Logs'
    
    # Determine plot structure based on column_configs
    plot_panels = []
    
    # Process each data type according to its configuration in the order defined in column_configs
    for data_type in available_columns.keys():
        if data_type not in valid_data_types or data_type not in data:
            continue
            
        type_config = available_columns[data_type]
        
        # Check for image configuration
        if 'image_path' in type_config:
            image_path = data_config['mother_dir'] + type_config['image_path']
            if os.path.exists(image_path):
                # Add image panel
                plot_panels.append({
                    'type': 'image',
                    'data_type': data_type,
                    'image_path': image_path,
                    'colormap': type_config.get('image_colormap', 'gray')
                })
        
        # Handle different config structures for data plotting
        if 'data_col' in type_config:
            # Single column data type
            data_col = type_config['data_col']
            if data_col in data[data_type].columns and not data[data_type][data_col].isna().all():
                plot_panels.append({
                    'type': 'data',
                    'data_type': data_type,
                    'columns': [data_col],
                    'config': type_config
                })
                
        elif 'data_cols' in type_config:
            # Multi-column data type
            data_cols = type_config['data_cols']
            available_cols = [col for col in data_cols if col in data[data_type].columns 
                            and not data[data_type][col].isna().all()]
            
            if available_cols:
                # Check for subplot grouping control
                if type_config.get('group_in_subplot', True):
                    # Plot all columns in one subplot
                    plot_panels.append({
                        'type': 'data',
                        'data_type': data_type,
                        'columns': available_cols,
                        'config': type_config
                    })
                else:
                    # Plot each column separately
                    for col in available_cols:
                        col_config = type_config.copy()
                        col_config['data_cols'] = [col]
                        plot_panels.append({
                            'type': 'data',
                            'data_type': data_type,
                            'columns': [col],
                            'config': col_config
                        })
                        
        else:
            # Nested configuration - process in the order defined in config
            for item_name in type_config.keys():
                item_config = type_config[item_name]
                if (isinstance(item_config, dict) and 
                    'data_col' in item_config):
                    data_col = item_config['data_col']
                    if data_col in data[data_type].columns and not data[data_type][data_col].isna().all():
                        plot_panels.append({
                            'type': 'data',
                            'data_type': data_type,
                            'columns': [data_col],
                            'config': item_config,
                            'item_name': item_name
                        })
    
    if not plot_panels:
        raise ValueError("No data available to plot")
    
    # Create subplot - add extra column for picked depths if data is available
    n_plots = len(plot_panels)
    has_picked_depths = picked_depths_data is not None
    total_plots = n_plots + (1 if has_picked_depths else 0)
    
    # Calculate figure width to maintain consistent column widths
    # Use a standard width per column (1.2) regardless of content type
    standard_width_per_column = 1.2
    fig_width = standard_width_per_column * total_plots
    
    fig, axes = plt.subplots(1, total_plots, figsize=(fig_width, 12))
    if total_plots == 1:
        axes = [axes]
    
    # Adjust subplot widths if picked depths column exists
    if has_picked_depths:
        # Make all columns the same width as they would be without picked depths
        width_ratios = [1.0] * total_plots  # All columns same width
        gs = fig.add_gridspec(1, total_plots, width_ratios=width_ratios)
        fig.clear()
        axes = [fig.add_subplot(gs[0, i]) for i in range(total_plots)]
    
    fig.suptitle(title, fontweight='bold', fontsize=14)
    
    # Plot picked depths column first if available
    plot_start_index = 0
    if has_picked_depths:
        picked_ax = axes[0]
        _plot_picked_depths(picked_ax, picked_depths_data, core_length, core_name)
        # Apply the same y-axis properties as other columns
        picked_ax.invert_yaxis()
        picked_ax.set_ylim(core_length, 0)
        picked_ax.set_ylabel('Depth', fontweight='bold')
        plot_start_index = 1
    
    # Plot each panel
    for i, panel in enumerate(plot_panels):
        ax_index = i + plot_start_index
        ax = axes[ax_index]
        
        # Only set y-label for the leftmost data subplot if no picked depths column exists
        if ax_index == plot_start_index and not has_picked_depths:
            ax.set_ylabel('Depth', fontweight='bold')
        
        if panel['type'] == 'image':
            # Plot image
            img = plt.imread(panel['image_path'])
            ax.imshow(img, aspect='auto', extent=[0, 1, core_length, 0], cmap=panel['colormap'])
            ax.set_xticks([])
            ax.set_xlabel(f'{panel["data_type"].upper()}\nImage', fontweight='bold', fontsize='small')
            
        elif panel['type'] == 'data':
            # Plot data
            _plot_data_panel(ax, panel, data, depth_col, core_length, clean_data, file_type)
        
        # Set common y-axis properties
        ax.invert_yaxis()
        ax.set_ylim(core_length, 0)
        # Hide y-axis labels for all data columns if picked depths column exists
        if has_picked_depths or ax_index > plot_start_index:
            ax.tick_params(axis='y', labelleft=False)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save_fig=True")
        
        # Validate formats
        supported_formats = ['png', 'jpg', 'jpeg', 'svg', 'pdf']
        for fmt in fig_format:
            if fmt.lower() not in supported_formats:
                raise ValueError(f"Unsupported format '{fmt}'. Supported formats: {supported_formats}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on file_type
        core_name = data_config['core_name']
        if file_type == 'clean':
            filename_base = f'{core_name}_ML-clean'
        else:
            filename_base = f'{core_name}_ML-filled'
        
        # Save in each requested format
        for fmt in fig_format:
            output_path = os.path.join(output_dir, f'{filename_base}.{fmt.lower()}')
            if dpi is not None:
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(output_path, bbox_inches='tight')
            print(f"Figure saved to: {output_path}")
    
    return fig, axes



def plot_filled_data(target_log, original_data, filled_data, data_config, ML_type='ML'):
    """
    Plot original and ML-filled data for a given log using configurable parameters.
    
    This function creates a horizontal plot showing the original data overlaid with
    ML-filled gaps, including uncertainty shading if available. All plotting parameters
    are driven by data_config content.
    
    Parameters
    ----------
    target_log : str
        Name of the log to plot
    original_data : pandas.DataFrame
        Original data containing the log with gaps
    filled_data : pandas.DataFrame
        Data with ML-filled gaps
    data_config : dict
        Configuration containing all parameters including depth column, plot labels, etc.
    ML_type : str, default='ML'
        Type of ML method used for title
        
    Returns
    -------
    None
        Displays the plot directly
        
    Notes
    -----
    The function searches through column_configs to find the target log configuration
    and uses appropriate plot labels and styling. Handles standard deviation shading
    if configured.
    """
    # Get parameters from config
    depth_col = data_config['depth_column']
    core_length = data_config['core_length']
    core_name = data_config['core_name']
    
    # Find the configuration for this target log
    target_config = None
    target_data_type = None
    
    # Search through column_configs to find the target log
    for data_type, type_config in data_config['column_configs'].items():
        if isinstance(type_config, dict):
            # Check for single column data types
            if 'data_col' in type_config and type_config['data_col'] == target_log:
                target_config = type_config
                target_data_type = data_type
                break
            # Check for multi-column data types
            elif 'data_cols' in type_config and target_log in type_config['data_cols']:
                target_config = type_config
                target_data_type = data_type
                break
            # Check for nested configurations
            else:
                for sub_key, sub_config in type_config.items():
                    if isinstance(sub_config, dict) and 'data_col' in sub_config and sub_config['data_col'] == target_log:
                        target_config = sub_config
                        target_data_type = data_type
                        break
                if target_config:
                    break
    
    # Get plot label from config or use default
    if target_config and 'plot_label' in target_config:
        plot_label = target_config['plot_label']
    else:
        plot_label = f'{target_log}\nBrightness'
    
    # Check if there are any gaps
    has_gaps = original_data[target_log].isna().any()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 3))
    title_suffix = f'Use {ML_type} for Data Gap Filling' if has_gaps else "(No Data Gap to be filled by ML)"
    fig.suptitle(f'{core_name} {target_log} Values {title_suffix}', fontweight='bold')

    # Plot data with ML-predicted gaps only if gaps exist
    if has_gaps:
        ax.plot(filled_data[depth_col], filled_data[target_log], 
                color='red', label=f'ML Predicted {target_log}', linewidth=0.7, alpha=0.7)

    # Plot original data
    ax.plot(original_data[depth_col], original_data[target_log], 
            color='black', label=f'Original {target_log}', linewidth=0.7)

    # Add uncertainty shade if std column exists - get std column name from config
    std_col = None
    if target_config:
        # For single column configs
        if 'std_col' in target_config:
            std_col = target_config['std_col']
        # For multi-column configs, find the corresponding std column
        elif 'std_cols' in target_config and 'data_cols' in target_config:
            data_cols = target_config['data_cols']
            std_cols = target_config['std_cols']
            if target_log in data_cols and len(std_cols) > data_cols.index(target_log):
                std_col = std_cols[data_cols.index(target_log)]
    
    if std_col and std_col in original_data.columns:
        ax.fill_between(original_data[depth_col],
                       original_data[target_log] - original_data[std_col],
                       original_data[target_log] + original_data[std_col],
                       color='black', alpha=0.2, linewidth=0)

    # Customize plot
    ax.set_ylabel(plot_label, fontweight='bold', fontsize='small')
    ax.set_xlabel('Depth')
    ax.grid(True)
    ax.invert_xaxis()
    ax.set_xlim(0, core_length)
    ax.tick_params(axis='y', labelsize='x-small')
    ax.legend()

    plt.tight_layout()
    plt.show()


# ==================== Helper Functions ====================

def _plot_picked_depths(ax, picked_depths_data, core_length, core_name):
    """
    Helper function to plot picked depths column.
    
    This function plots horizontal lines at picked depths with category-based styling,
    adds interpreted bed names for category 1 entries, and fills intervals with colors
    based on category transitions.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    picked_depths_data : pandas.DataFrame
        DataFrame containing picked depths data with columns:
        - picked_depths_cm: Depth values
        - category: Category values (1, 2, 3)
        - interpreted_bed: Bed names (optional, used for category 1)
    core_length : float
        Total core length for axis limits
    core_name : str
        Core name for the column title
        
    Returns
    -------
    None
        Modifies the axes object in place
    """
    # Set only the specific properties for picked depths column
    ax.set_xlim(0, 1)
    ax.set_xlabel('Datums', fontweight='bold', fontsize='small')
    ax.set_xticks([])
    
    # Define styling for each category
    category_styles = {
        1: {'color': 'black', 'linestyle': '-', 'linewidth': 1.0},
        2: {'color': 'brown', 'linestyle': '--', 'linewidth': 0.75},
        3: {'color': 'darkgrey', 'linestyle': '--', 'linewidth': 0.75}
    }
    
    # Define fill colors and patterns
    fill_styles = {
        'Ta-d': {'color': '#d9c355', 'alpha': 0.7},
        'Te': {'color': '#94724b', 'alpha': 0.7},
        'Pelagic': {'color': '#a3a2a2', 'alpha': 0.7}
    }
    
    # Sort data by depth for interval processing
    sorted_data = picked_depths_data.sort_values('picked_depths_cm').reset_index(drop=True)
    
    # Create intervals for filling
    intervals = []
    
    # Add top boundary (depth 0)
    boundaries = [(0, None)]
    
    # Add all picked depths
    for _, row in sorted_data.iterrows():
        if row['category'] in [1, 2, 3]:
            boundaries.append((row['picked_depths_cm'], row['category']))
    
    # Add bottom boundary (core_length)
    boundaries.append((core_length, None))
    
    # Determine fill type for each interval
    for i in range(len(boundaries) - 1):
        top_depth, top_cat = boundaries[i]
        bottom_depth, bottom_cat = boundaries[i + 1]
        
        # Determine fill type based on category transitions
        fill_type = None
        
        # Rule 1: Ta-d
        if ((top_cat == 2 and bottom_cat == 1) or 
            (top_cat is None and bottom_cat == 1) or 
            (top_cat == 2 and bottom_cat is None)):
            fill_type = 'Ta-d'
        
        # Rule 2: Te  
        elif ((top_cat == 3 and bottom_cat == 2) or 
              (top_cat is None and bottom_cat == 2) or 
              (top_cat == 3 and bottom_cat is None)):
            fill_type = 'Te'
        
        # Rule 3: Pelagic (same conditions as Te, but different interpretation)
        # Note: This appears to be a duplicate rule in the original specification
        # Keeping as Pelagic for areas that don't match Ta-d or Te specifically
        elif ((top_cat == 1 and bottom_cat == 3) or
              (top_cat == 2 and bottom_cat == 3) or
              (top_cat == 1 and bottom_cat == 2) or
              (top_cat is None and bottom_cat == 3)):
            fill_type = 'Pelagic'
        
        # Store interval information
        intervals.append({
            'top': top_depth,
            'bottom': bottom_depth,
            'fill_type': fill_type
        })
    
    # Plot fill areas
    legend_elements = []
    used_fills = set()
    
    for interval in intervals:
        top = interval['top']
        bottom = interval['bottom']
        fill_type = interval['fill_type']
        
        if fill_type in fill_styles:
            style = fill_styles[fill_type]
            ax.axvspan(0, 1, ymin=(core_length - bottom) / core_length, 
                      ymax=(core_length - top) / core_length,
                      color=style['color'], alpha=style['alpha'], zorder=1)
            
            if fill_type not in used_fills:
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                     facecolor=style['color'], 
                                     alpha=style['alpha'], 
                                     label=fill_type))
                used_fills.add(fill_type)
        else:
            # Fill with crossing pattern for "Others"
            ax.axvspan(0, 1, ymin=(core_length - bottom) / core_length, 
                      ymax=(core_length - top) / core_length,
                      color='lightgray', alpha=0.3, zorder=1,
                      hatch='///')
            
            if 'Others' not in used_fills:
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                     facecolor='lightgray', 
                                     alpha=0.3, hatch='///',
                                     label='Others'))
                used_fills.add('Others')
    
    # Plot horizontal lines for each picked depth
    for _, row in sorted_data.iterrows():
        depth = row['picked_depths_cm']
        category = row['category']
        
        # Skip if category is not in our defined styles
        if category not in category_styles:
            continue
            
        # Get style for this category
        style = category_styles[category]
        
        # Plot horizontal line across the width of the subplot
        ax.axhline(y=depth, xmin=0, xmax=1, 
                  color=style['color'], 
                  linestyle=style['linestyle'], 
                  linewidth=style['linewidth'],
                  zorder=10)
    
    # Add interpreted bed names for category 1
    text_positions = []
    if 'interpreted_bed' in sorted_data.columns:
        for _, row in sorted_data.iterrows():
            if (row['category'] == 1 and 
                pd.notna(row['interpreted_bed']) and 
                str(row['interpreted_bed']).strip() != ''):
                
                bed_name = str(row['interpreted_bed']).strip()
                depth = row['picked_depths_cm']
                # Place text slightly to the right and above the line
                ax.text(0.05, depth - 0.5, bed_name, 
                       fontsize='small', fontweight='bold', 
                       verticalalignment='bottom',
                       horizontalalignment='left',
                       color='black', zorder=15)
                text_positions.append(depth - 0.5)
    
    # Add legend in best position
    if legend_elements:
        # Find best position for legend to avoid text
        legend_y = core_length * 0.85  # Default position
        
        # Adjust if there are text labels
        if text_positions:
            # Find a gap in text positions
            text_positions.sort()
            best_gap = core_length * 0.85
            
            for i in range(len(text_positions) - 1):
                gap_start = text_positions[i] + 2
                gap_end = text_positions[i + 1] - 2
                gap_size = gap_end - gap_start
                
                if gap_size > 15:  # Minimum gap size needed for legend
                    best_gap = gap_start + gap_size / 2
                    break
            
            legend_y = best_gap
        
        # Ensure legend is within bounds
        legend_y = max(10, min(legend_y, core_length - 10))
        
        legend = ax.legend(handles=legend_elements, loc='center', 
                          bbox_to_anchor=(0.5, (core_length - legend_y) / core_length),
                          fontsize='x-small', frameon=True, fancybox=True, 
                          shadow=True, framealpha=0.9)
        legend.set_zorder(20)


def _plot_data_panel(ax, panel, data, depth_col, core_length, clean_data=None, file_type='clean'):
    """
    Helper function to plot a single data panel.
    
    This function plots data columns with optional standard deviation shading and
    colormap visualization based on the panel configuration. For file_type='filled',
    plots both cleaned (solid) and filled (dashed) data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    panel : dict
        Panel configuration containing data type, columns, and styling config
    data : dict
        Dictionary of loaded data by data type
    depth_col : str
        Name of the depth column
    core_length : float
        Total core length for axis limits
    clean_data : dict, optional
        Dictionary of cleaned data by data type (for comparison when file_type='filled')
    file_type : str, default='clean'
        Type of plot being created ('clean' or 'filled')
        
    Returns
    -------
    None
        Modifies the axes object in place
    """
    data_type = panel['data_type']
    columns = panel['columns']
    config = panel['config']
    
    df = data[data_type]
    depth_values = df[depth_col].astype(np.float32)
    
    # Get plot styling from config
    if 'plot_colors' in config:
        plot_colors = config['plot_colors']
    elif 'plot_color' in config:
        plot_colors = [config['plot_color']]
    else:
        plot_colors = ['black'] * len(columns)
    
    if len(plot_colors) < len(columns):
        plot_colors.extend(['black'] * (len(columns) - len(plot_colors)))
    
    # Plot each column
    for j, col in enumerate(columns):
        if col not in df.columns:
            continue
            
        values = df[col].astype(np.float32)
        color = plot_colors[j]
        
        # For filled plots, first plot the filled data as dashed lines
        if file_type == 'filled':
            ax.plot(values, depth_values, color=color, linewidth=0.7, linestyle='--', alpha=0.7, zorder=10)
            
            # Then plot cleaned data as solid lines on top if available
            if clean_data and data_type in clean_data:
                clean_df = clean_data[data_type]
                if col in clean_df.columns:
                    clean_values = clean_df[col].astype(np.float32)
                    clean_depth_values = clean_df[depth_col].astype(np.float32)
                    ax.plot(clean_values, clean_depth_values, color=color, linewidth=0.7, linestyle='-', zorder=20)
        else:
            # Plot main line for clean data
            ax.plot(values, depth_values, color=color, linewidth=0.7)
        
        # Add standard deviation if available (only for the main data)
        std_col = None
        if 'std_col' in config:
            std_col = config['std_col']
        elif 'std_cols' in config and j < len(config['std_cols']):
            std_col = config['std_cols'][j]
        
        if std_col and std_col in df.columns:
            std_values = df[std_col].astype(np.float32)
            ax.fill_betweenx(depth_values,
                           values - std_values,
                           values + std_values,
                           color=color, alpha=0.2, linewidth=0, zorder=1)
        
        # Add colormap visualization if configured (only for the main data)
        show_colormap = config.get('show_colormap', False)
        if show_colormap:
            colormap_name = config.get('colormap', 'viridis')
            _add_colormap_visualization(ax, values, depth_values, colormap_name, zorder=0)
        elif 'colormap_cols' in config and col in config['colormap_cols']:
            colormap_name = config.get('colormap', 'viridis')
            _add_colormap_visualization(ax, values, depth_values, colormap_name, zorder=0)
    
    # Set axis labels and styling
    plot_label = config.get('plot_label', columns[0] if len(columns) == 1 else 'Data')
    ax.set_xlabel(plot_label, fontweight='bold', fontsize='small')
    ax.grid(True)
    ax.tick_params(axis='x', labelsize='x-small')


def _add_colormap_visualization(ax, values, depths, colormap_name, zorder=0):
    """
    Helper function to add colormap visualization using PolyCollection.
    
    This function creates a colored background visualization where colors represent
    data values along the depth profile using the specified colormap.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add colormap to
    values : pandas.Series or numpy.array
        Data values for color mapping
    depths : pandas.Series or numpy.array
        Corresponding depth values
    colormap_name : str
        Name of the matplotlib colormap to use
    zorder : int, default=0
        Z-order for the colormap visualization
        
    Returns
    -------
    None
        Adds PolyCollection to the axes object
        
    Notes
    -----
    Creates polygons between consecutive data points and colors them based on
    the average value. Ignores NaN values in the data.
    """
    # Compute normalization range ignoring NaNs
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return
        
    vmin, vmax = valid_values.min(), valid_values.max()
    if np.isclose(vmin, vmax):
        return
        
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.colormaps[colormap_name]
    
    polys = []
    facecolors = []
    
    for i in range(len(depths) - 1):
        if not (np.isnan(values.iloc[i]) or np.isnan(values.iloc[i+1])):
            poly = [
                (0, depths.iloc[i]),
                (values.iloc[i], depths.iloc[i]),
                (values.iloc[i+1], depths.iloc[i+1]),
                (0, depths.iloc[i+1])
            ]
            polys.append(poly)
            avg_val = (values.iloc[i] + values.iloc[i+1]) / 2
            facecolors.append(cmap(norm(avg_val)))
    
    if polys:
        pc = mcoll.PolyCollection(polys, facecolors=facecolors, edgecolors='none', alpha=0.95, zorder=zorder)
        ax.add_collection(pc) 