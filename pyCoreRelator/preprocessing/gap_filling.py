"""
Machine Learning log data gap filling functions (computation) for pyCoreRelator.

Included Functions:
- preprocess_core_data: Preprocess core data by cleaning and scaling depth values using configurable parameters
 using fully configurable parameters from data_config
 for a given log using configurable parameters
- prepare_feature_data: Prepare merged feature data for ML training using configurable parameters
- apply_feature_weights: Apply feature weights using configurable parameters from data_config
- adjust_gap_predictions: Adjust ML predictions for gap rows to blend with linear interpolation
- train_model: Helper function for parallel model training
- fill_gaps_with_ml: Fill gaps in target data using specified ML method
- process_and_fill_logs: Process and fill gaps in log data using ML methods with fully configurable parameters

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
from .gap_filling_plots import plot_core_logs, plot_filled_data


def preprocess_core_data(data_config, resample_resolution=1):
    """
    Preprocess core data by cleaning and scaling depth values using configurable parameters.
    
    This function processes core data by applying data-type specific thresholds to remove
    artifacts and noises, then scales depth values according to the specified core length.
    All processing actions are driven by the data_config content.
    
    Parameters
    ----------
    data_config : dict
        Configuration dictionary containing:
        - column_configs: Dictionary of data type configurations with thresholds and depth_col
        - input_file_paths: Dictionary of input file directories by data type
        - clean_file_paths: Dictionary of output file directories by data type
        - core_length: Target core length for scaling
    resample_resolution : float, default=1
        Target depth resolution for resampling (spacing between depth values).
        If set to 1, data will be resampled to uniform depth spacing of 1 unit.
        
    Returns
    -------
    None
        Saves cleaned data files to the specified output directory
        
    Notes
    -----
    The function validates threshold conditions and processes different data types
    based on their configuration structure (single column, multi-column, or nested).
    Data will be resampled to uniform depth grid after cleaning and scaling operations
    based on the resample_resolution parameter.
    """
    # Get primary depth column from first available config
    depth_col = _get_depth_column(data_config)
    
    # Validate threshold conditions from column configs
    valid_conditions = ['>', '<', '<=', '>=']
    
    # Validate thresholds for all data types that have them
    for data_type, type_config in data_config['column_configs'].items():
        if isinstance(type_config, dict):
            # Check for any threshold that uses condition operators
            for key, value in type_config.items():
                if key == 'threshold' and isinstance(value, list) and len(value) >= 1:
                    if value[0] not in valid_conditions:
                        raise ValueError(f"Invalid condition '{value[0]}' for {data_type}.{key}.")
                # Check nested configs for thresholds
                elif isinstance(value, dict):
                    for sub_key, sub_config in value.items():
                        if isinstance(sub_config, dict) and 'threshold' in sub_config:
                            threshold = sub_config['threshold']
                            if isinstance(threshold, list) and len(threshold) >= 1 and threshold[0] not in valid_conditions:
                                raise ValueError(f"Invalid condition '{threshold[0]}' for {data_type}.{sub_key}.")

    # Process data types that exist in both input_file_paths and column_configs
    input_paths = data_config.get('input_file_paths', {})
    clean_paths = data_config.get('clean_file_paths', {})
    available_columns = data_config.get('column_configs', {})
    
    # Only process data types that have all necessary configurations
    valid_data_types = set(input_paths.keys()) & set(clean_paths.keys()) & set(available_columns.keys())
    
    # Create output directories for clean files
    for data_type in valid_data_types:
        output_path = clean_paths[data_type]
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Process each valid data type
    for data_type in valid_data_types:
        # Get data input path from config
        data_path = input_paths[data_type]
        
        if os.path.exists(data_path):
            print(f"Processing {data_type} data...")
            data = pd.read_csv(data_path).astype('float32')
            
            if data is not None:
                # Get type-specific configuration
                type_config = available_columns[data_type]
                
                # Apply data-type specific processing based on config structure
                if isinstance(type_config, dict):
                    # Handle different config structures
                    if 'data_cols' in type_config:
                        # data with multiple columns
                        data_columns = type_config['data_cols']
                        
                        # Apply multi-column threshold processing if configured
                        for threshold_key, threshold_config in type_config.items():
                            if threshold_key.endswith('_threshold') and isinstance(threshold_config, list) and len(threshold_config) >= 3:
                                min_val, max_val, buffer_size = threshold_config
                                buffer_indices = []
                                for col in data_columns:
                                    if col in data.columns:
                                        extreme_values = (data[col] <= min_val) | (data[col] >= max_val)
                                        for i in range(len(data)):
                                            if extreme_values[i]:
                                                buffer_indices.extend(range(max(0, i-buffer_size), min(len(data), i+buffer_size+1)))
                             
                                if buffer_indices:
                                    std_columns = type_config.get('std_cols', [])
                                    data.loc[buffer_indices, data_columns + std_columns] = np.nan
                    
                    elif 'data_col' in type_config:
                        # Single column data
                        data_col = type_config['data_col']
                        
                        # Apply threshold if defined
                        if 'threshold' in type_config:
                            condition, threshold_value, buffer_size = type_config['threshold']
                            extreme_values = eval(f"data['{data_col}'] {condition} {threshold_value}")
                            
                            extreme_indices = []
                            for i in range(len(data)):
                                if extreme_values[i]:
                                    extreme_indices.extend(range(max(0, i-buffer_size), min(len(data), i+buffer_size+1)))
                            
                            if extreme_indices:
                                data.loc[extreme_indices, data_col] = np.nan
                    
                    else:
                        # Nested configuration
                        # Map original column names to config keys for threshold lookup
                        column_to_config_map = {}
                        special_extreme_indices = []
                        
                        for log_type, config in type_config.items():
                            if isinstance(config, dict) and 'data_col' in config:
                                column_to_config_map[config['data_col']] = log_type
                                
                                # Apply threshold for any sub-type that has special processing flag
                                if 'threshold' in config and config.get('apply_to_all_columns', False):
                                    data_col = config['data_col']
                                    if data_col in data.columns:
                                        condition, threshold_value, buffer_size = config['threshold']
                                        extreme_values = eval(f"data['{data_col}'] {condition} {threshold_value}")
                                        for i in range(len(data)):
                                            if extreme_values[i]:
                                                special_extreme_indices.extend(range(max(0, i-buffer_size), min(len(data), i+buffer_size+1)))

                        # Process each column using config-based thresholds
                        for column in data.columns:
                            if column in column_to_config_map:
                                config_key = column_to_config_map[column]
                                if 'threshold' in type_config[config_key]:
                                    condition, threshold_value, buffer_size = type_config[config_key]['threshold']
                                    extreme_values = eval(f"data['{column}'] {condition} {threshold_value}")
                                    
                                    extreme_indices = []
                                    for i in range(len(data)):
                                        if extreme_values[i]:
                                            extreme_indices.extend(range(max(0, i-buffer_size), min(len(data), i+buffer_size+1)))
                                    
                                    # Combine with special extreme indices if applicable
                                    all_extreme_indices = list(set(extreme_indices + special_extreme_indices))
                                    if all_extreme_indices:
                                        data.loc[all_extreme_indices, column] = np.nan

                # Scale depth using configurable depth column
                depth_scale = data_config['core_length'] / data[depth_col].max()
                data[depth_col] = data[depth_col] * depth_scale
                
                # Apply resampling based on resample_resolution parameter
                if isinstance(resample_resolution, (int, float)) and resample_resolution > 0:
                    print(f"Resampling {data_type} data to {resample_resolution} depth resolution...")
                    data = _resample_to_target_resolution(data, depth_col, resample_resolution)
                
                # Use direct file path from config
                output_path = clean_paths[data_type]
                data.to_csv(output_path, index=False)
                print(f"Saved cleaned {data_type} data to: {output_path}")
        else:
            print(f"Warning: Raw file not found for {data_type}: {data_path}")
    
    print("Data preprocessing completed.") 



def prepare_feature_data(target_log, All_logs, merge_tolerance, data_config):
    """
    Prepare merged feature data for ML training using configurable parameters.
    
    This function merges data from multiple log sources to create a comprehensive
    feature dataset for machine learning. It handles depth alignment with tolerance
    and converts data types appropriately.
    
    Parameters
    ----------
    target_log : str
        Name of the target column for prediction
    All_logs : dict
        Dictionary of (dataframe, columns) pairs containing feature data
    merge_tolerance : float
        Maximum allowed difference in depth for merging rows
    data_config : dict
        Configuration containing column configs with depth_col and other parameters
        
    Returns
    -------
    tuple
        Contains the following elements:
        - target_data (pandas.DataFrame): Original target data
        - merged_data (pandas.DataFrame): Merged feature data for ML
        - features (list): List of feature column names
        
    Notes
    -----
    Uses merge_asof with tolerance for data alignment. Warns about unmatched rows
    due to tolerance constraints. Renames feature columns to avoid conflicts.
    """
    # Get primary depth column from first available config
    depth_col = _get_depth_column(data_config)
    
    # Get target data from All_logs
    target_data = None
    for df, cols in All_logs.values():
        if target_log in cols:
            target_data = df.copy()
            break
    
    if target_data is None:
        raise ValueError(f"Target log '{target_log}' not found in any dataset")

    # Convert depth column to float32 in target data
    target_data[depth_col] = target_data[depth_col].astype('float32')
    
    # Prepare training data by merging all available logs
    merged_data = target_data[[depth_col, target_log]].copy()
    features = []
    
    # Merge feature dataframes one by one, using their own depth column
    for df_name, (df, cols) in All_logs.items():
        if target_log not in cols:  # Skip the target dataframe
            df = df.copy()
            df[depth_col] = df[depth_col].astype('float32')
            # Rename depth column temporarily to avoid conflicts during merging
            temp_depth_col = f'{depth_col}_{df_name}'
            df = df.rename(columns={depth_col: temp_depth_col})
            # Convert all numeric columns to float32
            for col in cols:
                if col != depth_col and df[col].dtype.kind in 'biufc':
                    df[col] = df[col].astype('float32')
            # Rename feature columns for merging
            df_renamed = df.rename(columns={col: f'{df_name}_{col}' for col in cols if col != depth_col})
            df_renamed = df_renamed.sort_values(temp_depth_col)
            
            # Perform merge_asof with tolerance for data alignment
            merged_data = pd.merge_asof(
                merged_data.sort_values(depth_col),
                df_renamed,
                left_on=depth_col,
                right_on=temp_depth_col,
                direction='nearest',
                tolerance=merge_tolerance
            )
            
            # Check for unmatched rows due to the tolerance constraint
            unmatched = merged_data[temp_depth_col].isna().sum()
            if unmatched > 0:
                warnings.warn(f"{unmatched} rows did not have a matching depth within tolerance for log '{df_name}'.")
            
            # Add renamed feature columns to features list
            features.extend([f'{df_name}_{col}' for col in cols if col != depth_col])
            # Drop the temporary depth column used for merging
            merged_data = merged_data.drop(columns=[temp_depth_col])
    
    # Add depth column as a feature
    features.append(depth_col)
    
    return target_data, merged_data, features



def apply_feature_weights(X, data_config):
    """
    Apply feature weights using configurable parameters from data_config.
    
    This function multiplies feature columns by their configured weights to
    adjust their relative importance in machine learning models. Weights for
    empty or non-existent columns are ignored.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature dataframe to apply weights to
    data_config : dict
        Configuration containing column_configs with feature weights
        
    Returns
    -------
    pandas.DataFrame
        Weighted feature dataframe
        
    Notes
    -----
    Processes different configuration structures (single column, multi-column, nested)
    and applies weights to matching columns in the feature dataframe. Ignores weights
    for columns that don't exist or are empty in the feature dataframe.
    """
    X_weighted = X.copy()
    column_configs = data_config['column_configs']
    
    def _has_valid_data(col_name):
        """Check if column exists and has valid (non-empty, non-all-NaN) data."""
        matching_cols = [x_col for x_col in X_weighted.columns if col_name in x_col]
        if not matching_cols:
            return False
        
        for x_col in matching_cols:
            # Check if column has any non-NaN values
            if not X_weighted[x_col].isna().all() and len(X_weighted[x_col].dropna()) > 0:
                return True
        return False
    
    # Process each data type in column_configs
    for data_type, type_config in column_configs.items():
        if isinstance(type_config, dict):
            # Handle multi-column data types with feature_weights
            if 'data_cols' in type_config and 'feature_weights' in type_config:
                data_cols = type_config['data_cols']
                weights = type_config['feature_weights']
                
                # Apply weights to each column that exists and has valid data
                for col, weight in zip(data_cols, weights):
                    if _has_valid_data(col):
                        matching_cols = [x_col for x_col in X_weighted.columns if col in x_col]
                        for x_col in matching_cols:
                            # Additional check for the specific column
                            if not X_weighted[x_col].isna().all() and len(X_weighted[x_col].dropna()) > 0:
                                X_weighted[x_col] = (X_weighted[x_col] * weight).astype('float32')
            
            # Handle single column data types with feature_weight
            elif 'data_col' in type_config and 'feature_weight' in type_config:
                data_col = type_config['data_col']
                weight = type_config['feature_weight']
                
                # Apply weight only if column exists and has valid data
                if _has_valid_data(data_col):
                    matching_cols = [x_col for x_col in X_weighted.columns if data_col in x_col]
                    for x_col in matching_cols:
                        # Additional check for the specific column
                        if not X_weighted[x_col].isna().all() and len(X_weighted[x_col].dropna()) > 0:
                            X_weighted[x_col] = (X_weighted[x_col] * weight).astype('float32')
            
            # Handle nested configurations (with multiple sub-types)
            else:
                for sub_type, sub_config in type_config.items():
                    if isinstance(sub_config, dict) and 'data_col' in sub_config and 'feature_weight' in sub_config:
                        data_col = sub_config['data_col']
                        weight = sub_config['feature_weight']
                        
                        # Apply weight only if column exists and has valid data
                        if _has_valid_data(data_col):
                            matching_cols = [x_col for x_col in X_weighted.columns if data_col in x_col]
                            for x_col in matching_cols:
                                # Additional check for the specific column
                                if not X_weighted[x_col].isna().all() and len(X_weighted[x_col].dropna()) > 0:
                                    X_weighted[x_col] = (X_weighted[x_col] * weight).astype('float32')
    
    return X_weighted



def adjust_gap_predictions(df, gap_mask, ml_preds, target_log, data_config):
    """
    Adjust ML predictions for gap rows to blend with linear interpolation between boundaries.
    
    This function adjusts ML predictions for contiguous gap segments by blending them
    with linear interpolation between the boundary values, ensuring smoother transitions.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Original dataframe containing depth and target data
    gap_mask : pandas.Series
        Boolean mask indicating gap positions
    ml_preds : numpy.array
        ML predictions for gap positions
    target_log : str
        Name of the target column
    data_config : dict
        Configuration containing column configs with depth_col
        
    Returns
    -------
    numpy.array
        Adjusted predictions with blended interpolation
        
    Notes
    -----
    For each contiguous gap segment with both left and right boundaries available,
    the predictions are blended with linear interpolation. The blending weight varies
    from the boundaries (more interpolation) to the middle (more ML prediction).
    """
    # Get primary depth column from first available config
    depth_col = _get_depth_column(data_config)
    
    # Get the integer positions (row numbers) of missing values
    gap_positions = np.where(gap_mask.values)[0]
    # Create a Series for easier handling; index = positions in df
    preds_series = pd.Series(ml_preds, index=gap_positions)
    
    # Identify contiguous segments in the gap positions
    segments = np.split(gap_positions, np.where(np.diff(gap_positions) != 1)[0] + 1)
    
    adjusted = preds_series.copy()
    for seg in segments:
        # seg is an array of row positions (in df) for a contiguous gap segment.
        start_pos = seg[0]
        end_pos = seg[-1]
        
        # Enforce trend constraints only if both boundaries exist.
        if start_pos == 0 or end_pos == len(df) - 1:
            continue  # Skip segments at the very beginning or end.
        
        # Retrieve boundary (observed) values and depths using configurable depth column
        left_value = df.iloc[start_pos - 1][target_log]
        right_value = df.iloc[end_pos + 1][target_log]
        # Skip if boundaries are missing (should not happen if gap_mask is correct)
        if pd.isna(left_value) or pd.isna(right_value):
            continue
        left_depth = df.iloc[start_pos - 1][depth_col]
        right_depth = df.iloc[end_pos + 1][depth_col]
        
        # For each gap row in the segment, blend the ML prediction with linear interpolation
        for pos in seg:
            current_depth = df.iloc[pos][depth_col]
            # Normalize the depth position (x in [0, 1])
            if right_depth == left_depth:
                x = 0.5
            else:
                x = (current_depth - left_depth) / (right_depth - left_depth)
            # Compute the linear interpolation value at this depth
            interp_val = left_value + (right_value - left_value) * x
            # Define a weight that is 0 at the boundaries and 1 at the middle.
            # Here we use: weight = 1 - 2*|x - 0.5|
            weight = 1 - 2 * abs(x - 0.5)
            weight = max(0, min(weight, 1))  # Ensure weight is between 0 and 1
            # Blend: final = interpolation + weight*(ML_prediction - interpolation)
            adjusted[pos] = interp_val + weight * (preds_series[pos] - interp_val)
    
    return adjusted.values



def train_model(model):
    """
    Helper function for parallel model training.
    
    This function creates a wrapper for training a single model in parallel processing
    contexts. It fits the model on training data and returns predictions.
    
    Parameters
    ----------
    model : sklearn-like model
        Machine learning model with fit and predict methods
        
    Returns
    -------
    function
        Wrapper function that trains the model and returns predictions
        
    Notes
    -----
    Used in conjunction with joblib.Parallel for training multiple models simultaneously.
    """
    def train_wrapper(X_train, y_train, X_pred):
        model.fit(X_train, y_train)
        return model.predict(X_pred)
    return train_wrapper 



def fill_gaps_with_ml(target_log, All_logs, data_config, output_csv=True, 
                      merge_tolerance=3.0, ml_method='xgblgbm'):
    """
    Fill gaps in target data using specified ML method.
    
    This function prepares feature data, applies the specified machine learning method,
    and fills gaps in the target log data. All parameters and file paths are driven
    by data_config content.
    
    Parameters
    ----------
    target_log : str
        Name of the target column to fill gaps in
    All_logs : dict
        Dictionary of dataframes containing feature data and target data
    data_config : dict
        Configuration containing all parameters including file paths, core info, etc.
    output_csv : bool, default=True
        Whether to output filled data to CSV file
    merge_tolerance : float, default=3.0
        Maximum allowed difference in depth for merging rows
    ml_method : str, default='xgblgbm'
        ML method to use - 'rf', 'rftc', 'xgb', 'xgblgbm'
        
    Returns
    -------
    tuple
        (target_data_filled, gap_mask) containing filled data and gap locations
        
    Raises
    ------
    ValueError
        If required parameters are missing or ml_method is invalid
        
    Notes
    -----
    Supports four ML methods: Random Forest ('rf'), Random Forest with Trend
    Constraints ('rftc'), XGBoost ('xgb'), and XGBoost+LightGBM ('xgblgbm').
    """
    # Input validation
    if target_log is None or All_logs is None or data_config is None:
        raise ValueError("target_log, All_logs, and data_config must be provided")
    
    if ml_method not in ['rf', 'rftc', 'xgb', 'xgblgbm']:
        raise ValueError("ml_method must be one of: 'rf', 'rftc', 'xgb', 'xgblgbm'")
    
    # Prepare feature data
    target_data, merged_data, features = prepare_feature_data(target_log, All_logs, merge_tolerance, data_config)
    
    # Create a copy of the original data to hold the interpolated results
    target_data_filled = target_data.copy()

    # Identify gaps in target data
    gap_mask = target_data[target_log].isna()
    
    # Check if target column has any valid data for training
    valid_target_data = target_data[target_log].dropna()
    if len(valid_target_data) == 0:
        print(f"Skipping {target_log} - no valid training data available (0 samples)")
        if output_csv:
            # Generate output path based on target_log and data_config
            output_path = _generate_output_filepath(target_log, data_config)
            target_data_filled.to_csv(output_path, index=False)
        return target_data_filled, gap_mask
    
    # If no gaps exist, save to CSV if requested and return original data
    if not gap_mask.any():
        if output_csv:
            # Generate output path based on target_log and data_config
            output_path = _generate_output_filepath(target_log, data_config)
            target_data_filled.to_csv(output_path, index=False)
        return target_data_filled, gap_mask

    # Prepare features and target for ML
    X = merged_data[features].copy()
    y = merged_data[target_log].copy()

    # Convert all features to float32
    for col in X.columns:
        if X[col].dtype.kind in 'biufc':
            X[col] = X[col].astype('float32')
    y = y.astype('float32')

    # Split into training (non-gap) and prediction (gap) sets
    X_train = X[~gap_mask]
    y_train = y[~gap_mask]
    X_pred = X[gap_mask]
    
    # Final check: ensure we have sufficient training data after feature preparation
    if len(y_train.dropna()) == 0:
        print(f"Skipping {target_log} - no valid training samples after feature preparation")
        if output_csv:
            # Generate output path based on target_log and data_config
            output_path = _generate_output_filepath(target_log, data_config)
            target_data_filled.to_csv(output_path, index=False)
        return target_data_filled, gap_mask

    # Apply specific ML method
    if ml_method == 'rf':
        predictions = _apply_random_forest(X_train, y_train, X_pred, data_config)
    elif ml_method == 'rftc':
        predictions = _apply_random_forest_with_trend_constraints(X_train, y_train, X_pred, merged_data, gap_mask, target_log, data_config)
    elif ml_method == 'xgb':
        predictions = _apply_xgboost(X_train, y_train, X_pred, data_config)
    elif ml_method == 'xgblgbm':
        predictions = _apply_xgboost_lightgbm(X_train, y_train, X_pred, data_config)

    # Fill gaps with predictions
    target_data_filled.loc[gap_mask, target_log] = predictions
    
    # Save to CSV if requested
    if output_csv:
        # Generate output path based on target_log and data_config
        output_path = _generate_output_filepath(target_log, data_config)
        target_data_filled.to_csv(output_path, index=False)

    return target_data_filled, gap_mask 



def process_and_fill_logs(data_config, ml_method='xgblgbm', n_jobs=-1, show_plots=True):
    """
    Process and fill gaps in log data using ML methods with fully configurable parameters.
    
    This function orchestrates the complete ML-based gap filling process for all configured
    log data types. It loads cleaned data, processes each target log, applies ML methods,
    and consolidates results into final output files. Supports parallel processing of
    multiple target logs.
    
    Parameters
    ----------
    data_config : dict
        Configuration containing all parameters including:
        - clean_file_paths: Dictionary of input file directories by data type
        - filled_file_paths: Dictionary of output file directories by data type
        - column_configs: Dictionary of data type configurations with depth_col
    ml_method : str, default='xgblgbm'
        ML method to use - 'rf', 'rftc', 'xgb', 'xgblgbm'
    n_jobs : int, default=-1
        Number of parallel jobs for processing multiple target logs.
        -1 means using all available CPU cores.
        1 means sequential processing (no parallelization).
    show_plots : bool, default=True
        Whether to generate and display plots during processing.
        Works in both sequential and parallel modes using appropriate matplotlib backend.
        
    Returns
    -------
    None
        Saves filled data files and displays progress information
        
    Notes
    -----
    The function handles different data type configurations (single column, multi-column,
    nested) and creates both individual and consolidated output files as appropriate.
    Removes temporary individual files for multi-column data types after consolidation.
    
    Parallel processing is applied at the target log level, allowing multiple logs to be
    processed simultaneously. Each parallel job will use its own memory space.
    Plots are generated using the Agg backend in parallel mode to work properly across
    worker processes.
    """
    # Get configurable parameters
    depth_col = _get_depth_column(data_config)
    
    clean_paths = data_config.get('clean_file_paths', {})
    filled_paths = data_config.get('filled_file_paths', {})
    available_columns = data_config.get('column_configs', {})
    valid_data_types = set(clean_paths.keys()) & set(available_columns.keys())
    
    # Create output directories for filled files
    for data_type in valid_data_types:
        if data_type in filled_paths:
            output_path = filled_paths[data_type]
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
    
    # Load data using full file paths
    data_dict = {}
    for data_type in valid_data_types:
        full_path = clean_paths[data_type]
        if os.path.exists(full_path):
            data = pd.read_csv(full_path)
            if not data.empty:
                data_dict[data_type] = data
        else:
            print(f"Warning: Clean file not found for {data_type}: {full_path}")

    if not data_dict:
        print("No valid data files found for processing")
        return

    # Create feature data dictionary using configurable column names
    feature_data = {}
    
    for data_type in valid_data_types:
        if data_type in data_dict:
            type_config = available_columns[data_type]
            
            # Handle single column data types
            if 'data_col' in type_config:
                data_col = type_config['data_col']
                if data_col in data_dict[data_type].columns:
                    # Check if column has valid data for features
                    col_data = data_dict[data_type][data_col]
                    if not col_data.isna().all() and len(col_data.dropna()) > 0:
                        feature_data[data_type] = (data_dict[data_type], [depth_col, data_col])
            
            # Handle multi-column data types
            elif 'data_cols' in type_config:
                valid_cols = [depth_col]
                for col in type_config['data_cols']:
                    if col in data_dict[data_type].columns:
                        # Check if column has valid data for features
                        col_data = data_dict[data_type][col]
                        if not col_data.isna().all() and len(col_data.dropna()) > 0:
                            valid_cols.append(col)
                if len(valid_cols) > 1:
                    feature_data[data_type] = (data_dict[data_type], valid_cols)
            
            # Handle nested configurations
            else:
                data_cols = [depth_col]
                for sub_key, sub_config in type_config.items():
                    if isinstance(sub_config, dict) and 'data_col' in sub_config:
                        data_col = sub_config['data_col']
                        if data_col in data_dict[data_type].columns:
                            # Check if column has valid data for features
                            col_data = data_dict[data_type][data_col]
                            if not col_data.isna().all() and len(col_data.dropna()) > 0:
                                data_cols.append(data_col)
                if len(data_cols) > 1:
                    feature_data[data_type] = (data_dict[data_type], data_cols)

    if not feature_data:
        print("No valid feature data found for ML processing")
        return

    # ML method names for plotting
    ml_names = {
        'rf': 'Random Forest', 
        'rftc': 'Random Forest with Trend Constraints',
        'xgb': 'XGBoost', 
        'xgblgbm': 'XGBoost + LightGBM'
    }

    # Helper function to get additional feature support for multi-column data types
    def get_additional_features_for_type(target_data_type):
        """Get additional feature columns for a data type based on configuration."""
        type_config = available_columns.get(target_data_type, {})
        
        # Check if this data type has additional feature support configured
        if 'additional_feature_source' in type_config:
            source_type = type_config['additional_feature_source']
            source_columns = type_config.get('additional_feature_columns', [])
            
            if source_type in feature_data and source_columns:
                df, available_cols = feature_data[source_type]
                valid_columns = [col for col in source_columns if col in available_cols and col in df.columns]
                if valid_columns:
                    return source_type, [depth_col] + valid_columns
        
        return None, None

    # Collect target logs dynamically from column configurations
    # Process in the order defined in column_configs to maintain consistency
    target_logs = []
    
    # Process data types in the order they appear in column_configs
    for data_type in available_columns.keys():
        if data_type in valid_data_types and data_type in feature_data:
            type_config = available_columns[data_type]
            
            # Handle multi-column data types
            if 'data_cols' in type_config:
                # Process columns in the order defined in data_cols
                for col in type_config['data_cols']:
                    if col in data_dict[data_type].columns:
                        # Check if column has valid non-empty data for ML training
                        col_data = data_dict[data_type][col]
                        if not col_data.isna().all() and len(col_data.dropna()) > 0:
                            target_logs.append((col, data_type))
                        else:
                            print(f"Skipping {col} - column is empty (0 samples)")
            
            # Handle single column data types
            elif 'data_col' in type_config:
                col = type_config['data_col']
                if col in data_dict[data_type].columns:
                    # Check if column has valid non-empty data for ML training
                    col_data = data_dict[data_type][col]
                    if not col_data.isna().all() and len(col_data.dropna()) > 0:
                        target_logs.append((col, data_type))
                    else:
                        print(f"Skipping {col} - column is empty (0 samples)")
            
            # Handle nested configurations
            else:
                # Process sub-keys in the order they appear in the config
                for sub_key in type_config.keys():
                    sub_config = type_config[sub_key]
                    if isinstance(sub_config, dict) and 'data_col' in sub_config:
                        col = sub_config['data_col']
                        if col in data_dict[data_type].columns:
                            # Check if column has valid non-empty data for ML training
                            col_data = data_dict[data_type][col]
                            if not col_data.isna().all() and len(col_data.dropna()) > 0:
                                target_logs.append((col, data_type))
                            else:
                                print(f"Skipping {col} - column is empty (0 samples)")

    # Helper function to process a single target log
    def process_single_log(target_log, data_type, enable_plotting=True, log_idx=None, total_logs=None, is_parallel=False):
        """Process a single target log with ML gap filling."""
        # Print progress for sequential mode
        if log_idx is not None and total_logs is not None:
            print(f"[{log_idx}/{total_logs}] Processing {target_log}...")
        
        # Get source data
        data = data_dict[data_type]
        plot_name = target_log
        
        # Create filtered feature data based on configuration
        type_config = available_columns[data_type]
        
        # For multi-column data types, create filtered features
        if 'data_cols' in type_config:
            # Use all other feature data types except the current one
            filtered_features = {k: v for k, v in feature_data.items() if k != data_type}
            filtered_features[data_type] = (data, [depth_col, target_log])
            
            # Check if this data type has additional feature support configured
            additional_source_type, additional_columns = get_additional_features_for_type(data_type)
            if additional_source_type and additional_columns:
                df, _ = feature_data[additional_source_type]
                filtered_features[additional_source_type] = (df, additional_columns)
                    
            filled_data, gap_mask = fill_gaps_with_ml(
                target_log=target_log,
                All_logs=filtered_features,
                data_config=data_config,
                output_csv=True,
                ml_method=ml_method
            )
            if enable_plotting and not is_parallel:
                # Only show plots in sequential mode
                plot_filled_data(plot_name, data, filled_data, data_config, ML_type=ml_names[ml_method])
            elif enable_plotting and is_parallel:
                # In parallel mode, return plot data for later display
                # Make sure filled_data includes depth column
                filled_data_with_depth = filled_data.copy()
                if depth_col not in filled_data_with_depth:
                    filled_data_with_depth[depth_col] = data[depth_col]
                return ('plot_data', target_log, data, filled_data_with_depth, ml_names[ml_method])
            return None  # Multi-column types write directly to files
        else:
            # For single column or nested data types, don't create individual files
            filled_data, gap_mask = fill_gaps_with_ml(
                target_log=target_log,
                All_logs=feature_data,
                data_config=data_config,
                output_csv=False,
                ml_method=ml_method
            )
            
            # Plot filled data for each column
            if enable_plotting and not is_parallel:
                # Only show plots in sequential mode
                plot_filled_data(plot_name, data, filled_data, data_config, ML_type=ml_names[ml_method])
                # Return the filled results for consolidation
                return (data_type, target_log, filled_data[target_log])
            elif enable_plotting and is_parallel:
                # In parallel mode, return both plot data and filled results
                # Make sure filled_data includes depth column for plotting
                filled_data_with_depth = filled_data.copy()
                if depth_col not in filled_data_with_depth:
                    filled_data_with_depth[depth_col] = data[depth_col]
                return ('plot_and_data', data_type, target_log, filled_data[target_log], data, filled_data_with_depth, ml_names[ml_method])
            else:
                # No plotting, just return filled results
                return (data_type, target_log, filled_data[target_log])
    
    # Process each target log (with optional parallelization)
    data_type_results = {}  # Store results by data type for consolidation
    
    # Use show_plots parameter directly
    enable_plotting = show_plots
    
    total_logs = len(target_logs)
    
    if n_jobs == 1:
        # Sequential processing with simple progress messages
        print(f"Processing {total_logs} logs sequentially...")
        for idx, (target_log, data_type) in enumerate(target_logs, 1):
            result = process_single_log(target_log, data_type, enable_plotting, idx, total_logs, is_parallel=False)
            if result is not None:
                dt, tl, filled_values = result
                if dt not in data_type_results:
                    data_type_results[dt] = {}
                data_type_results[dt][tl] = filled_values
        print(f"Completed processing {total_logs} logs.")
    else:
        # Parallel processing
        print(f"Processing {total_logs} logs in parallel using {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_log)(target_log, data_type, enable_plotting, None, None, is_parallel=True) 
            for target_log, data_type in target_logs
        )
        
        print(f"Completed processing {total_logs} logs.")
        
        # Collect results from parallel execution and plot data
        plot_queue = []  # Store plot data for later display
        for result in results:
            if result is not None:
                if result[0] == 'plot_data':
                    # Multi-column type with plot data
                    _, target_log, data, filled_data_with_depth, ml_type = result
                    plot_queue.append((target_log, data, filled_data_with_depth, ml_type))
                elif result[0] == 'plot_and_data':
                    # Single column type with both plot data and filled results
                    _, dt, tl, filled_values, data, filled_data_with_depth, ml_type = result
                    if dt not in data_type_results:
                        data_type_results[dt] = {}
                    data_type_results[dt][tl] = filled_values
                    plot_queue.append((tl, data, filled_data_with_depth, ml_type))
                else:
                    # No plotting, just filled results
                    dt, tl, filled_values = result
                    if dt not in data_type_results:
                        data_type_results[dt] = {}
                    data_type_results[dt][tl] = filled_values
        
        # Display all plots after parallel processing completes
        if enable_plotting and plot_queue:
            print(f"Displaying {len(plot_queue)} plots...")
            for plot_name, data, filled_data, ml_type in plot_queue:
                plot_filled_data(plot_name, data, filled_data, data_config, ML_type=ml_type)
            print("All plots displayed.")

    # Create consolidated files for each data type using configured paths
    for data_type, filled_columns in data_type_results.items():
        if data_type in data_dict and data_type in filled_paths:
            data_copy = data_dict[data_type].copy()
            updated_columns = []
            
            for col, filled_values in filled_columns.items():
                data_copy[col] = filled_values
                updated_columns.append(col)
            
            # Save consolidated file using filled_file_paths from config
            output_path = filled_paths[data_type]
            data_copy.to_csv(output_path, index=False)
            print(f"Saved [{', '.join(updated_columns)}] to {os.path.basename(output_path)}")

    # Consolidate multi-column data types - remove individual files
    multi_column_types = [dt for dt, config in available_columns.items() 
                         if 'data_cols' in config and dt in data_dict]
    
    for data_type in multi_column_types:
        if data_type in filled_paths:
            data_copy = data_dict[data_type].copy()
            type_config = available_columns[data_type]
            data_columns = type_config['data_cols']
            updated_columns = []
            
            for col in data_columns:
                if col in data_copy.columns:
                    # Generate individual filepath using the same logic as fill_gaps_with_ml
                    individual_filepath = _generate_output_filepath(col, data_config)
                    if os.path.exists(individual_filepath):
                        filled_data = pd.read_csv(individual_filepath)
                        if col in filled_data.columns:
                            data_copy[col] = filled_data[col]
                            updated_columns.append(col)
            
            if updated_columns:
                # Save consolidated file using filled_file_paths from config
                output_path = filled_paths[data_type]
                data_copy.to_csv(output_path, index=False)
                print(f"Saved [{', '.join(updated_columns)}] to {os.path.basename(output_path)}")
                
                # Remove individual files
                for col in data_columns:
                    individual_filepath = _generate_output_filepath(col, data_config)
                    if os.path.exists(individual_filepath):
                        os.remove(individual_filepath)

    print("ML-based gap filling completed for all target logs.")


# ==================== Helper Functions ====================

def _get_depth_column(data_config):
    """
    Extract depth column name from column_configs.
    
    Parameters
    ----------
    data_config : dict
        Configuration dictionary containing column_configs
        
    Returns
    -------
    str
        Name of the depth column
        
    Raises
    ------
    ValueError
        If no depth_col found in any configuration
    """
    column_configs = data_config.get('column_configs', {})
    
    for data_type, type_config in column_configs.items():
        if isinstance(type_config, dict):
            # Check for depth_col at top level of config
            if 'depth_col' in type_config:
                return type_config['depth_col']
            
            # Check nested configurations
            for sub_key, sub_config in type_config.items():
                if isinstance(sub_config, dict) and 'depth_col' in sub_config:
                    return sub_config['depth_col']
    
    raise ValueError("No depth_col found in column_configs")


def _resample_to_target_resolution(data, depth_col, target_resolution):
    """
    Resample data to a uniform target depth resolution using linear interpolation.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data with depth column and data columns
    depth_col : str
        Name of the depth column
    target_resolution : float
        Target depth resolution (spacing between depth values)
        
    Returns
    -------
    pandas.DataFrame
        Resampled data with uniform depth spacing
    """
    # Create new uniform depth grid
    min_depth = data[depth_col].min()
    max_depth = data[depth_col].max()
    new_depth = np.arange(min_depth, max_depth + target_resolution, target_resolution)
    
    # Create new dataframe with uniform depth
    resampled_data = pd.DataFrame({depth_col: new_depth})
    
    # Interpolate each data column
    for col in data.columns:
        if col != depth_col:
            resampled_data[col] = np.interp(new_depth, data[depth_col], data[col], 
                                            left=np.nan, right=np.nan)
    
    return resampled_data.astype('float32')


def _generate_output_filepath(target_log, data_config):
    """
    Generate output filepath for ML-filled data based on target log and data configuration.
    
    Parameters
    ----------
    target_log : str
        Name of the target column
    data_config : dict
        Configuration containing filled_file_paths and column_configs
        
    Returns
    -------
    str
        Generated full filepath for the output file
    """
    column_configs = data_config['column_configs']
    filled_file_paths = data_config.get('filled_file_paths', {})
    
    # Find which data type this target log belongs to
    target_data_type = None
    for data_type, type_config in column_configs.items():
        if isinstance(type_config, dict):
            # Check for single column data types
            if 'data_col' in type_config and type_config['data_col'] == target_log:
                target_data_type = data_type
                break
            # Check for multi-column data types
            elif 'data_cols' in type_config and target_log in type_config['data_cols']:
                target_data_type = data_type
                break
            # Check for nested configurations
            else:
                for sub_key, sub_config in type_config.items():
                    if isinstance(sub_config, dict) and 'data_col' in sub_config and sub_config['data_col'] == target_log:
                        target_data_type = data_type
                        break
                if target_data_type:
                    break
    
    # Use existing filled_file_paths from data_config
    if target_data_type and target_data_type in filled_file_paths:
        type_config = column_configs[target_data_type]
        base_filepath = filled_file_paths[target_data_type]
        
        # For multi-column data types, create individual file variation
        if 'data_cols' in type_config:
            # Extract directory and filename from the base filepath
            base_dir = os.path.dirname(base_filepath)
            base_filename = os.path.basename(base_filepath)
            base_name, ext = os.path.splitext(base_filename)
            
            # Replace the data type part with the specific column name
            # Find data type identifier in base filename and replace with target_log
            if f'_{target_data_type.upper()}_' in base_name:
                individual_filename = base_name.replace(f'_{target_data_type.upper()}_', f'_{target_log}_') + ext
            elif f'_{target_data_type}_' in base_name:
                individual_filename = base_name.replace(f'_{target_data_type}_', f'_{target_log}_') + ext
            else:
                # Fallback: insert target_log before the last part
                parts = base_name.split('_')
                if len(parts) >= 2:
                    parts.insert(-1, target_log)
                    individual_filename = '_'.join(parts) + ext
                else:
                    individual_filename = f"{base_name}_{target_log}{ext}"
            
            # Reconstruct full path
            return os.path.join(base_dir, individual_filename)
        else:
            # For single-column or nested types, use the base filepath directly
            return base_filepath
    
    # If no configuration found, raise an error rather than hardcoding
    raise ValueError(f"No filled_file_paths configuration found for target_log '{target_log}' in data_type '{target_data_type}'")


def _apply_random_forest(X_train, y_train, X_pred, data_config):
    """Apply Random Forest method."""
    # Handle outliers using IQR method
    quantile_cutoff = 0.025
    Q1 = y_train.quantile(quantile_cutoff)
    Q3 = y_train.quantile(1 - quantile_cutoff)
    IQR = Q3 - Q1
    outlier_mask = (y_train >= Q1 - 1.5 * IQR) & (y_train <= Q3 + 1.5 * IQR)
    X_train = X_train[outlier_mask]
    y_train = y_train[outlier_mask]

    def train_model_wrapper(model):
        model.fit(X_train, y_train)
        return model.predict(X_pred)

    # Initialize two ensemble models
    models = [
        RandomForestRegressor(n_estimators=1000,
                              max_depth=30,
                              min_samples_split=5,
                              min_samples_leaf=5,
                              max_features='sqrt',
                              bootstrap=True,
                              random_state=42,
                              n_jobs=-1),
        HistGradientBoostingRegressor(max_iter=800,
                                      learning_rate=0.05,
                                      max_depth=5,
                                      min_samples_leaf=50,
                                      l2_regularization=1.0,
                                      random_state=42,
                                      verbose=0)
    ]

    # Train models in parallel
    predictions = Parallel(n_jobs=-1)(delayed(train_model_wrapper)(model) for model in models)

    # Ensemble predictions by averaging
    ensemble_predictions = np.mean(predictions, axis=0)
    
    return ensemble_predictions


def _apply_random_forest_with_trend_constraints(X_train, y_train, X_pred, merged_data, gap_mask, target_log, data_config):
    """Apply Random Forest with trend constraints method."""
    # Handle outliers using IQR method
    quantile_cutoff = 0.15
    Q1 = y_train.quantile(quantile_cutoff)
    Q3 = y_train.quantile(1 - quantile_cutoff)
    IQR = Q3 - Q1
    outlier_mask = (y_train >= Q1 - 1.5 * IQR) & (y_train <= Q3 + 1.5 * IQR)
    X_train = X_train[outlier_mask]
    y_train = y_train[outlier_mask]
    
    def train_model_wrapper(model):
        model.fit(X_train, y_train)
        return model.predict(X_pred)
    
    # Initialize two ensemble models
    models = [
        RandomForestRegressor(n_estimators=1000,
                              max_depth=30,
                              min_samples_split=5,
                              min_samples_leaf=5,
                              max_features='sqrt',
                              bootstrap=True,
                              random_state=42,
                              n_jobs=-1),
        HistGradientBoostingRegressor(max_iter=800,
                                      learning_rate=0.05,
                                      max_depth=5,
                                      min_samples_leaf=50,
                                      l2_regularization=1.0,
                                      random_state=42,
                                      verbose=-1)
    ]
    
    # Train models in parallel and average their predictions
    predictions = Parallel(n_jobs=-1)(delayed(train_model_wrapper)(model) for model in models)
    ensemble_predictions = np.mean(predictions, axis=0)
    
    # Apply trend constraints using the helper function
    adjusted_predictions = adjust_gap_predictions(merged_data, gap_mask, ensemble_predictions, target_log, data_config)
    
    return adjusted_predictions


def _apply_xgboost(X_train, y_train, X_pred, data_config):
    """Apply XGBoost method with configurable feature weights."""
    # Apply feature weights BEFORE processing
    X_train_weighted = apply_feature_weights(X_train, data_config)
    X_pred_weighted = apply_feature_weights(X_pred, data_config)
    
    # Handle outliers using IQR method
    quantile_cutoff = 0.025
    Q1 = y_train.quantile(quantile_cutoff)
    Q3 = y_train.quantile(1 - quantile_cutoff)
    IQR = Q3 - Q1
    outlier_mask = (y_train >= Q1 - 1.5 * IQR) & (y_train <= Q3 + 1.5 * IQR)
    X_train_weighted = X_train_weighted[outlier_mask]
    y_train = y_train[outlier_mask]

    # Create feature pipeline
    feature_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)),
        ('selector', SelectKBest(score_func=f_regression, k='all'))
    ])

    # Process features
    X_train_processed = feature_pipeline.fit_transform(X_train_weighted, y_train)
    X_pred_processed = feature_pipeline.transform(X_pred_weighted)

    # Convert processed arrays to float32
    X_train_processed = X_train_processed.astype('float32')
    X_pred_processed = X_pred_processed.astype('float32')
    y_train = y_train.astype('float32')

    # Initialize and train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.003,
        max_depth=10,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train_processed, y_train)
    predictions = model.predict(X_pred_processed).astype('float32')
    
    return predictions


def _apply_xgboost_lightgbm(X_train, y_train, X_pred, data_config):
    """Apply XGBoost + LightGBM ensemble method with configurable feature weights."""
    # Apply feature weights BEFORE processing
    X_train_weighted = apply_feature_weights(X_train, data_config)
    X_pred_weighted = apply_feature_weights(X_pred, data_config)
    
    # Create feature pipeline
    feature_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=True))
    ])

    # Process features without selector first to get actual feature count
    X_train_processed = feature_pipeline.fit_transform(X_train_weighted, y_train)
    
    # Now add selector with correct feature count
    max_features = min(50, X_train.shape[0]//10, X_train_processed.shape[1])
    selector = SelectKBest(score_func=f_regression, k=max_features)
    X_train_processed = selector.fit_transform(X_train_processed, y_train)
    X_pred_processed = feature_pipeline.transform(X_pred_weighted)
    X_pred_processed = selector.transform(X_pred_processed)

    # Convert processed arrays to float32
    X_train_processed = X_train_processed.astype('float32')
    X_pred_processed = X_pred_processed.astype('float32')
    y_train = y_train.astype('float32')

    # Initialize models
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.003,
        max_depth=10,
        min_child_weight=5,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
    )

    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.003,
        max_depth=6,
        num_leaves=20,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.3,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1
    )

    # Train both models with warnings suppressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_model.fit(X_train_processed, y_train)
        lgb_model.fit(X_train_processed, y_train, feature_name='auto')

    # Make predictions with both models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_predictions = xgb_model.predict(X_pred_processed).astype('float32')
        lgb_predictions = lgb_model.predict(X_pred_processed).astype('float32')

    # Ensemble predictions (simple average)
    predictions = (xgb_predictions + lgb_predictions) / 2

    return predictions 