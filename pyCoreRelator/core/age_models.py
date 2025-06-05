"""
Age modeling and age constraint functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from joblib import Parallel, delayed
import itertools


def calculate_interpolated_ages(picked_depths, age_constraints_depths, age_constraints_ages, 
                                age_constraints_pos_errors, age_constraints_neg_errors,
                                age_constraint_source_core=None,
                                age_constraints_in_sequence_flags=None, top_bottom=True, top_age=0, top_age_pos_error=0, 
                                top_age_neg_error=0, top_depth=0.0, bottom_depth=None, 
                                uncertainty_method='MonteCarlo', n_monte_carlo=10000,
                                show_plot=False, core_name=None, export_csv=True):
    """
    Calculate interpolated/extrapolated ages for picked depths based on age constraints.
    This function handles both in-sequence and out-of-sequence constraints internally.
    
    Parameters:
    -----------
    picked_depths : list
        List of picked depths in cm
    age_constraints_depths : list or pd.Series
        List of mean depths for all age constraints (calculated from mindepth_cm and maxdepth_cm)
    age_constraints_ages : list
        List of calibrated ages for all age constraints
    age_constraints_pos_errors : list
        List of positive error values for all age constraints
    age_constraints_neg_errors : list
        List of negative error values for all age constraints
    age_constraints_in_sequence_flags : list or None, optional
        List indicating which age constraints are in sequence (True/False or 1/0).
        If None, all constraints are treated as in-sequence.
    age_constraint_source_core : list or None, default=None
        List of source core names for each age constraint. If provided, constraints from the same core
        will be colored differently in the plot than those from adjacent cores.
    top_age : float, default=0
        Age at the top of the core (depth = 0) in years BP
    top_age_pos_error : float, default=0
        Positive uncertainty of the top age in years
    top_age_neg_error : float, default=0
        Negative uncertainty of the top age in years
    top_depth : float, default=0.0
        Depth at the top of the core in cm
    bottom_depth : float, default=None
        Depth at the bottom of the core in cm. If None, uses the last in-sequence constraint depth
    uncertainty_method : str, default='Linear'
        Method for calculating uncertainties. Options: 'Linear', 'MonteCarlo', 'Gaussian'
        - 'Linear': Current worst-case scenario approach using all combinations
        - 'MonteCarlo': Monte Carlo simulation with sampling from asymmetric distributions (sampling from the ±2σ of the age uncertainties)
        - 'Gaussian': Gaussian error propagation assuming normal distributions
    n_monte_carlo : int, default=10000
        Number of Monte Carlo iterations (only used when uncertainty_method='MonteCarlo')
    show_plot : bool, default=False
        If True, plot the age-depth model and constraints.
    core_name : str, optional
        Name of the core for plot title (only used if show_plot is True).
    export_csv : bool, default=True
        If True, export the results to a CSV file with columns for depths, ages, and uncertainties.
    top_bottom : bool, default=True
        If True, include top and bottom depths/ages in the results. If False, only include picked depths.
        
    Returns:
    --------
    dict
        Dictionary containing the interpolated/extrapolated ages and uncertainties for each picked depth,
        as well as the top and bottom ages and their uncertainties if top_bottom is True
    """
    
    # Check if age constraints are empty
    if len(age_constraints_depths) == 0:
        raise ValueError("No age constraints provided. Cannot calculate interpolated ages.")
    
    # Validate uncertainty_method parameter
    valid_methods = ['Linear', 'MonteCarlo', 'Gaussian']
    if uncertainty_method not in valid_methods:
        raise ValueError(f"uncertainty_method must be one of {valid_methods}, got {uncertainty_method}")
    
    # Convert all inputs to numpy arrays for consistent handling
    if isinstance(age_constraints_depths, pd.Series):
        age_constraints_depths = age_constraints_depths.values
    else:
        age_constraints_depths = np.array(age_constraints_depths)
    
    age_constraints_ages = np.array(age_constraints_ages)
    age_constraints_pos_errors = np.array(age_constraints_pos_errors)
    age_constraints_neg_errors = np.array(age_constraints_neg_errors)
    
    # Create the in-sequence mask
    if age_constraints_in_sequence_flags is None:
        # If no in-sequence information provided, treat all as in-sequence
        in_seq_mask = np.ones(len(age_constraints_depths), dtype=bool)
    else:
        # Convert the in-sequence indicators to a boolean mask
        in_seq_flags = np.array(age_constraints_in_sequence_flags)
        if in_seq_flags.dtype == bool:
            in_seq_mask = in_seq_flags
        else:
            # Handle various formats case-insensitively
            in_seq_mask = np.zeros(len(in_seq_flags), dtype=bool)
            for i, flag in enumerate(in_seq_flags):
                if isinstance(flag, (int, float)):
                    # Numeric values where 1 means in-sequence
                    in_seq_mask[i] = (flag == 1)
                elif isinstance(flag, str):
                    # String values like 'true', 'True', 'T', etc.
                    flag_lower = flag.lower()
                    in_seq_mask[i] = (flag_lower in ['true', 't', '1', 'yes', 'y'])
                else:
                    # Any other type, treat as False
                    in_seq_mask[i] = False
    
    # Filter to get only in-sequence constraints for interpolation/extrapolation
    constraint_depths = age_constraints_depths[in_seq_mask]
    constraint_ages = age_constraints_ages[in_seq_mask]
    constraint_pos_errors = age_constraints_pos_errors[in_seq_mask]
    constraint_neg_errors = age_constraints_neg_errors[in_seq_mask]
    
    # Ensure we have at least one constraint (excluding the top)
    if len(constraint_depths) == 0:
        print("Warning: No in-sequence constraints available. Using top age only.")
        if bottom_depth is None:
            bottom_depth = np.max(age_constraints_depths) if len(age_constraints_depths) > 0 else 100.0
    
    # Set bottom_depth to the last in-sequence constraint depth if not provided
    if bottom_depth is None:
        bottom_depth = constraint_depths[-1] if len(constraint_depths) > 0 else 100.0
    
    # Add the top of the core as a constraint point
    all_constraint_depths_with_top = np.insert(constraint_depths, 0, top_depth)
    all_constraint_ages_with_top = np.insert(constraint_ages, 0, top_age)
    all_constraint_pos_errors_with_top = np.insert(constraint_pos_errors, 0, top_age_pos_error)
    all_constraint_neg_errors_with_top = np.insert(constraint_neg_errors, 0, top_age_neg_error)
    
    # Helper function for asymmetric sampling in Monte Carlo
    def sample_asymmetric_age(mean_age, pos_error, neg_error, size=1):
        """Sample from an asymmetric distribution representing age uncertainty"""
        if pos_error == 0 and neg_error == 0:
            return np.full(size, mean_age)
        
        # Use skewed normal distribution to approximate asymmetric uncertainty
        # If errors are equal, use normal distribution
        if abs(pos_error - neg_error) < 0.01 * max(pos_error, neg_error):
            return np.random.normal(mean_age, (pos_error + neg_error) / 2, size)
        
        # For asymmetric case, use a simple approach:
        # Sample from uniform, then map to asymmetric distribution
        u = np.random.uniform(0, 1, size)
        samples = np.zeros(size)
        
        # Split the distribution: lower half maps to negative errors, upper half to positive
        lower_mask = u < 0.5
        upper_mask = ~lower_mask
        
        # Map lower half (0 to 0.5) to (mean_age - neg_error, mean_age)
        if np.any(lower_mask):
            u_lower = u[lower_mask] * 2  # Scale to 0-1
            samples[lower_mask] = mean_age - neg_error * (1 - u_lower)
        
        # Map upper half (0.5 to 1) to (mean_age, mean_age + pos_error)
        if np.any(upper_mask):
            u_upper = (u[upper_mask] - 0.5) * 2  # Scale to 0-1
            samples[upper_mask] = mean_age + pos_error * u_upper
        
        return samples
    
    # Helper function for single Monte Carlo iteration
    def monte_carlo_iteration(picked_depths, all_constraint_depths_with_top, 
                            all_constraint_ages_with_top, all_constraint_pos_errors_with_top,
                            all_constraint_neg_errors_with_top, bottom_depth, top_bottom):
        # Sample ages for all constraints
        sampled_ages = np.array([
            sample_asymmetric_age(age, pos_err, neg_err)[0]
            for age, pos_err, neg_err in zip(all_constraint_ages_with_top, 
                                           all_constraint_pos_errors_with_top,
                                           all_constraint_neg_errors_with_top)
        ])
        
        # Calculate interpolated ages for this iteration
        iteration_ages = []
        
        for depth in picked_depths:
            # Find bracketing indices (same logic as original)
            if len(all_constraint_depths_with_top) <= 1:
                age = sampled_ages[0]
            elif depth <= all_constraint_depths_with_top[1]:
                start_idx = 0
                end_idx = 1
                depth_fraction = (depth - all_constraint_depths_with_top[start_idx]) / (all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx])
                age = sampled_ages[start_idx] + depth_fraction * (sampled_ages[end_idx] - sampled_ages[start_idx])
            elif depth > all_constraint_depths_with_top[-1]:
                if len(all_constraint_depths_with_top) <= 2:
                    start_idx = 0
                    end_idx = 1
                else:
                    start_idx = len(all_constraint_depths_with_top) - 2
                    end_idx = len(all_constraint_depths_with_top) - 1
                depth_fraction = (depth - all_constraint_depths_with_top[start_idx]) / (all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx])
                age = sampled_ages[start_idx] + depth_fraction * (sampled_ages[end_idx] - sampled_ages[start_idx])
            else:
                for i in range(len(all_constraint_depths_with_top) - 1):
                    if all_constraint_depths_with_top[i] <= depth <= all_constraint_depths_with_top[i+1]:
                        start_idx = i
                        end_idx = i + 1
                        break
                depth_fraction = (depth - all_constraint_depths_with_top[start_idx]) / (all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx])
                age = sampled_ages[start_idx] + depth_fraction * (sampled_ages[end_idx] - sampled_ages[start_idx])
            
            iteration_ages.append(age)
        
        # Add bottom age if requested
        if top_bottom:
            if len(all_constraint_depths_with_top) <= 2:
                start_idx = 0
                end_idx = 1
            else:
                start_idx = len(all_constraint_depths_with_top) - 2
                end_idx = len(all_constraint_depths_with_top) - 1
            
            depth_fraction = (bottom_depth - all_constraint_depths_with_top[start_idx]) / (all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx])
            bottom_age = sampled_ages[start_idx] + depth_fraction * (sampled_ages[end_idx] - sampled_ages[start_idx])
            iteration_ages.append(bottom_age)
        
        return iteration_ages
    
    # Calculate central values (same as before)
    estimated_ages = []
    
    # Process each picked depth for central values
    for depth in picked_depths:
        # If we have no in-sequence constraints or only the top, use linear extrapolation from top
        if len(constraint_depths) == 0:
            estimated_age = top_age + (depth - top_depth) * 100  
            estimated_ages.append(estimated_age)
            continue
        
        # Determine the bracketing depth indices
        if depth <= all_constraint_depths_with_top[1]:
            start_idx = 0
            end_idx = 1
        elif depth > all_constraint_depths_with_top[-1]:
            if len(all_constraint_depths_with_top) <= 2:
                start_idx = 0
                end_idx = 1
            else:
                start_idx = len(all_constraint_depths_with_top) - 2
                end_idx = len(all_constraint_depths_with_top) - 1
        else:
            for i in range(len(all_constraint_depths_with_top) - 1):
                if all_constraint_depths_with_top[i] <= depth <= all_constraint_depths_with_top[i+1]:
                    start_idx = i
                    end_idx = i + 1
                    break
        
        # Calculate interpolation/extrapolation based on central values
        depth_fraction = (depth - all_constraint_depths_with_top[start_idx]) / (all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx])
        estimated_age = all_constraint_ages_with_top[start_idx] + depth_fraction * (all_constraint_ages_with_top[end_idx] - all_constraint_ages_with_top[start_idx])
        estimated_ages.append(estimated_age)
    
    # Calculate bottom age if requested
    if top_bottom:
        if len(all_constraint_depths_with_top) <= 2:
            start_idx = 0
            end_idx = 1
        else:
            start_idx = len(all_constraint_depths_with_top) - 2
            end_idx = len(all_constraint_depths_with_top) - 1
        
        depth_fraction = (bottom_depth - all_constraint_depths_with_top[start_idx]) / (all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx])
        bottom_age = all_constraint_ages_with_top[start_idx] + depth_fraction * (all_constraint_ages_with_top[end_idx] - all_constraint_ages_with_top[start_idx])
    
    # Calculate uncertainties based on method
    if uncertainty_method == 'Linear':
        # Original method using itertools.product for worst-case scenarios
        estimated_pos_uncertainties = []
        estimated_neg_uncertainties = []
        
        # Calculate max and min possible ages for each constraint
        max_constraint_ages = all_constraint_ages_with_top + all_constraint_pos_errors_with_top
        min_constraint_ages = all_constraint_ages_with_top - all_constraint_neg_errors_with_top
        
        for i, depth in enumerate(picked_depths):
            # Same bracketing logic as for central values
            if len(constraint_depths) == 0:
                estimated_pos_uncertainties.append(top_age_pos_error + (depth - top_depth) * 50)
                estimated_neg_uncertainties.append(top_age_neg_error + (depth - top_depth) * 50)
                continue
            
            # Find bracketing indices
            if depth <= all_constraint_depths_with_top[1]:
                start_idx = 0
                end_idx = 1
            elif depth > all_constraint_depths_with_top[-1]:
                if len(all_constraint_depths_with_top) <= 2:
                    start_idx = 0
                    end_idx = 1
                else:
                    start_idx = len(all_constraint_depths_with_top) - 2
                    end_idx = len(all_constraint_depths_with_top) - 1
            else:
                for j in range(len(all_constraint_depths_with_top) - 1):
                    if all_constraint_depths_with_top[j] <= depth <= all_constraint_depths_with_top[j+1]:
                        start_idx = j
                        end_idx = j + 1
                        break
            
            depth_fraction = (depth - all_constraint_depths_with_top[start_idx]) / (all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx])
            
            # Generate all possible combinations of min/max ages for the bracketing points
            possible_ages = []
            start_ages = [min_constraint_ages[start_idx], max_constraint_ages[start_idx]]
            end_ages = [min_constraint_ages[end_idx], max_constraint_ages[end_idx]]
            
            for start_age, end_age in itertools.product(start_ages, end_ages):
                possible_age = start_age + depth_fraction * (end_age - start_age)
                possible_ages.append(possible_age)
            
            max_possible_age = np.max(possible_ages)
            min_possible_age = np.min(possible_ages)
            
            estimated_pos_uncertainties.append(max(0, max_possible_age - estimated_ages[i]))
            estimated_neg_uncertainties.append(max(0, estimated_ages[i] - min_possible_age))
        
        # Calculate bottom uncertainties for Linear method
        if top_bottom:
            if len(all_constraint_depths_with_top) <= 2:
                start_idx = 0
                end_idx = 1
            else:
                start_idx = len(all_constraint_depths_with_top) - 2
                end_idx = len(all_constraint_depths_with_top) - 1
            
            depth_fraction = (bottom_depth - all_constraint_depths_with_top[start_idx]) / (all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx])
            
            possible_bottom_ages = []
            start_ages = [min_constraint_ages[start_idx], max_constraint_ages[start_idx]]
            end_ages = [min_constraint_ages[end_idx], max_constraint_ages[end_idx]]
            
            for start_age, end_age in itertools.product(start_ages, end_ages):
                possible_age = start_age + depth_fraction * (end_age - start_age)
                possible_bottom_ages.append(possible_age)
            
            max_bottom_age = np.max(possible_bottom_ages)
            min_bottom_age = np.min(possible_bottom_ages)
            
            bottom_pos_uncertainty = max(0, max_bottom_age - bottom_age)
            bottom_neg_uncertainty = max(0, bottom_age - min_bottom_age)
    
    elif uncertainty_method == 'MonteCarlo':
        # Monte Carlo simulation with parallel processing
        print(f"Running Monte Carlo simulation with {n_monte_carlo} iterations...")
        
        # Run Monte Carlo iterations in parallel
        results = Parallel(n_jobs=-1)(
            delayed(monte_carlo_iteration)(
                picked_depths, all_constraint_depths_with_top, 
                all_constraint_ages_with_top, all_constraint_pos_errors_with_top,
                all_constraint_neg_errors_with_top, bottom_depth, top_bottom
            ) for _ in range(n_monte_carlo)
        )
        
        # Convert results to numpy array for easier processing
        results = np.array(results)  # Shape: (n_monte_carlo, n_depths)
        
        # Calculate percentiles for uncertainties (2.5th and 97.5th percentiles ≈ ±2σ)
        estimated_pos_uncertainties = []
        estimated_neg_uncertainties = []
        
        n_picked = len(picked_depths)
        for i in range(n_picked):
            percentile_max = np.percentile(results[:, i], 97.5)
            percentile_min = np.percentile(results[:, i], 2.5)
            
            estimated_pos_uncertainties.append(max(0, percentile_max - estimated_ages[i]))
            estimated_neg_uncertainties.append(max(0, estimated_ages[i] - percentile_min))
        
        # Calculate bottom uncertainties
        if top_bottom:
            percentile_max_bottom = np.percentile(results[:, -1], 97.5)
            percentile_min_bottom = np.percentile(results[:, -1], 2.5)
            
            bottom_pos_uncertainty = max(0, percentile_max_bottom - bottom_age)
            bottom_neg_uncertainty = max(0, bottom_age - percentile_min_bottom)
    
    elif uncertainty_method == 'Gaussian':
        # Gaussian error propagation
        estimated_pos_uncertainties = []
        estimated_neg_uncertainties = []
        
        for i, depth in enumerate(picked_depths):
            if len(constraint_depths) == 0:
                estimated_pos_uncertainties.append(top_age_pos_error + (depth - top_depth) * 50)
                estimated_neg_uncertainties.append(top_age_neg_error + (depth - top_depth) * 50)
                continue
            
            # Find bracketing indices
            if depth <= all_constraint_depths_with_top[1]:
                start_idx = 0
                end_idx = 1
            elif depth > all_constraint_depths_with_top[-1]:
                if len(all_constraint_depths_with_top) <= 2:
                    start_idx = 0
                    end_idx = 1
                else:
                    start_idx = len(all_constraint_depths_with_top) - 2
                    end_idx = len(all_constraint_depths_with_top) - 1
            else:
                for j in range(len(all_constraint_depths_with_top) - 1):
                    if all_constraint_depths_with_top[j] <= depth <= all_constraint_depths_with_top[j+1]:
                        start_idx = j
                        end_idx = j + 1
                        break
            
            # Calculate interpolation weights
            depth_diff = all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx]
            weight_start = (all_constraint_depths_with_top[end_idx] - depth) / depth_diff
            weight_end = (depth - all_constraint_depths_with_top[start_idx]) / depth_diff
            
            # Use average of pos and neg errors as standard deviation
            sigma_start = (all_constraint_pos_errors_with_top[start_idx] + all_constraint_neg_errors_with_top[start_idx]) / 2
            sigma_end = (all_constraint_pos_errors_with_top[end_idx] + all_constraint_neg_errors_with_top[end_idx]) / 2
            
            # Error propagation for weighted sum
            combined_sigma = np.sqrt((weight_start * sigma_start)**2 + (weight_end * sigma_end)**2)
            
            # For asymmetric uncertainties, scale the combined sigma
            avg_pos_error = weight_start * all_constraint_pos_errors_with_top[start_idx] + weight_end * all_constraint_pos_errors_with_top[end_idx]
            avg_neg_error = weight_start * all_constraint_neg_errors_with_top[start_idx] + weight_end * all_constraint_neg_errors_with_top[end_idx]
            
            estimated_pos_uncertainties.append(max(combined_sigma, avg_pos_error))
            estimated_neg_uncertainties.append(max(combined_sigma, avg_neg_error))
        
        # Calculate bottom uncertainties for Gaussian method
        if top_bottom:
            if len(all_constraint_depths_with_top) <= 2:
                start_idx = 0
                end_idx = 1
            else:
                start_idx = len(all_constraint_depths_with_top) - 2
                end_idx = len(all_constraint_depths_with_top) - 1
            
            depth_diff = all_constraint_depths_with_top[end_idx] - all_constraint_depths_with_top[start_idx]
            weight_start = (all_constraint_depths_with_top[end_idx] - bottom_depth) / depth_diff
            weight_end = (bottom_depth - all_constraint_depths_with_top[start_idx]) / depth_diff
            
            sigma_start = (all_constraint_pos_errors_with_top[start_idx] + all_constraint_neg_errors_with_top[start_idx]) / 2
            sigma_end = (all_constraint_pos_errors_with_top[end_idx] + all_constraint_neg_errors_with_top[end_idx]) / 2
            
            combined_sigma = np.sqrt((weight_start * sigma_start)**2 + (weight_end * sigma_end)**2)
            
            avg_pos_error = weight_start * all_constraint_pos_errors_with_top[start_idx] + weight_end * all_constraint_pos_errors_with_top[end_idx]
            avg_neg_error = weight_start * all_constraint_neg_errors_with_top[start_idx] + weight_end * all_constraint_neg_errors_with_top[end_idx]
            
            bottom_pos_uncertainty = max(combined_sigma, avg_pos_error)
            bottom_neg_uncertainty = max(combined_sigma, avg_neg_error)
    
    # Prepare final results
    if top_bottom:
        all_depths = [top_depth] + list(picked_depths) + [bottom_depth]
        all_ages = [top_age] + estimated_ages + [bottom_age]
        all_pos_uncertainties = [top_age_pos_error] + estimated_pos_uncertainties + [bottom_pos_uncertainty]
        all_neg_uncertainties = [top_age_neg_error] + estimated_neg_uncertainties + [bottom_neg_uncertainty]
    else:
        all_depths = list(picked_depths)
        all_ages = estimated_ages
        all_pos_uncertainties = estimated_pos_uncertainties
        all_neg_uncertainties = estimated_neg_uncertainties

    # Create a dictionary to return the results
    result = {
        'depths': all_depths,
        'ages': all_ages,
        'pos_uncertainties': all_pos_uncertainties,
        'neg_uncertainties': all_neg_uncertainties,
        'uncertainty_method': uncertainty_method
    }
    
    # Export results to CSV if requested
    if export_csv:
        export_df = pd.DataFrame({
            'picked_depths_cm': all_depths,
            'est_age': all_ages,
            'est_age_poserr': all_pos_uncertainties,
            'est_age_negerr': all_neg_uncertainties
        })
        
        filename = f"{core_name}_pickeddepth_age_{uncertainty_method}.csv" if core_name else f"pickeddepth_age_{uncertainty_method}.csv"
        export_df.to_csv(filename, index=False)
        print(f"Exported interpolated/extrapolated ages ({uncertainty_method} method) to {filename}")

    # Plot if requested
    if show_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        if core_name is None:
            core_name = "Core"
        ax.set_title(f'Core: {core_name} {"(including top/bottom)" if top_bottom else "(picked depths only)"}')
        ax.set_xlabel('Age (years BP)')
        ax.set_ylabel('Depth (cm)')

        # Calculate sedimentation rates for each segment
        sed_rates = []
        sed_rate_uncertainties = []
        
        for i in range(len(all_constraint_depths_with_top) - 1):
            depth_diff = all_constraint_depths_with_top[i+1] - all_constraint_depths_with_top[i]
            age_diff = all_constraint_ages_with_top[i+1] - all_constraint_ages_with_top[i]
            
            # Calculate sedimentation rate in mm/year
            if age_diff > 0:
                # Convert from cm/year to mm/year (multiply by 10)
                rate = (depth_diff / age_diff) * 10
                
                # Calculate uncertainties for sedimentation rate
                # Consider all possible combinations of min/max ages
                rates = []
                start_ages = [all_constraint_ages_with_top[i] - all_constraint_neg_errors_with_top[i], 
                              all_constraint_ages_with_top[i] + all_constraint_pos_errors_with_top[i]]
                end_ages = [all_constraint_ages_with_top[i+1] - all_constraint_neg_errors_with_top[i+1], 
                            all_constraint_ages_with_top[i+1] + all_constraint_pos_errors_with_top[i+1]]
                
                for start_age, end_age in itertools.product(start_ages, end_ages):
                    if end_age > start_age:  # Avoid division by zero or negative rates
                        rates.append((depth_diff / (end_age - start_age)) * 10)  # Convert to mm/year
                
                if rates:
                    max_rate = max(rates)
                    min_rate = min(rates)
                    rate_uncertainty = max(max_rate - rate, rate - min_rate)
                else:
                    rate_uncertainty = 0
            else:
                rate = 0
                rate_uncertainty = 0
                
            sed_rates.append(rate)
            sed_rate_uncertainties.append(rate_uncertainty)

        # Plot the actual interpolation/extrapolation function (piecewise linear) based on in-sequence constraints
        ax.plot(all_ages, all_depths, 'k-', linewidth=1.0, alpha=0.7, label='Age-Depth Model')

        # Add sedimentation rate annotations to each line segment
        for i in range(len(all_constraint_depths_with_top) - 1):
            # Find a point on the line segment for placing the text
            # Use a position 30% along the line segment from the start
            pos_fraction = 0.3
            text_age = all_constraint_ages_with_top[i] + pos_fraction * (all_constraint_ages_with_top[i+1] - all_constraint_ages_with_top[i])
            text_depth = all_constraint_depths_with_top[i] + pos_fraction * (all_constraint_depths_with_top[i+1] - all_constraint_depths_with_top[i])
            
            # Format the sedimentation rate text (1 digit after decimal point)
            if sed_rates[i] > 0:
                rate_text = f"{sed_rates[i]:.1f} ± {sed_rate_uncertainties[i]:.1f} mm/yr"
            else:
                rate_text = "Undefined rate"
            
            # Calculate angle of the line segment to determine text offset direction
            angle = np.arctan2(
                all_constraint_depths_with_top[i+1] - all_constraint_depths_with_top[i],
                all_constraint_ages_with_top[i+1] - all_constraint_ages_with_top[i]
            ) * 180 / np.pi
            
            # Adjust text position to not overlap with the line
            # If angle is steep, place text to the right
            if abs(angle) > 45:
                x_offset = 0.05 * (max(all_ages) - min(all_ages))
                y_offset = 0
                ha = 'left'
            else:
                # For flatter angles, place text slightly above the line
                x_offset = 0.02 * (max(all_ages) - min(all_ages))
                y_offset = -0.02 * (max(all_depths) - min(all_depths))
                ha = 'left'
            
            # Place the text with appropriate offset
            ax.text(text_age + x_offset, text_depth + y_offset, rate_text, fontsize=8, 
                    # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2),
                    ha=ha, va='center')

        # Plot estimated ages at the picked depths
        ax.errorbar(
            all_ages, 
            all_depths, 
            xerr=[np.abs(all_neg_uncertainties), np.abs(all_pos_uncertainties)], 
            fmt='bo', 
            label='Estimated Ages at Selected Depths', 
            capsize=4,
            # markersize=2,
            zorder=3
        )

        # Create masks for different constraint categories
        if np.any(in_seq_mask):
            if age_constraint_source_core is not None and core_name is not None:
                # Create masks for same-core and different-core constraints
                same_core_mask = np.zeros_like(in_seq_mask, dtype=bool)
                for i in range(len(in_seq_mask)):
                    if in_seq_mask[i] and age_constraint_source_core[i] is not None:
                        # Check if the constraint's source core name is contained in the core_name
                        same_core_mask[i] = age_constraint_source_core[i] in core_name
                
                # Different core but in-sequence mask
                diff_core_mask = in_seq_mask & ~same_core_mask
                
                # Plot in-sequence constraints from the same core
                if np.any(same_core_mask):
                    ax.errorbar(
                        age_constraints_ages[same_core_mask], 
                        age_constraints_depths[same_core_mask], 
                        xerr=[np.abs(age_constraints_neg_errors[same_core_mask]), np.abs(age_constraints_pos_errors[same_core_mask])], 
                        fmt='^', color='r', 
                        label='Age Constraints (In Sequence)', 
                        capsize=4,
                        alpha=0.5,
                        markerfacecolor='w',
                        zorder=4
                    )
                    # Add horizontal lines from y-axis to in-sequence constraints from same core
                    for i in range(len(age_constraints_depths)):
                        if same_core_mask[i]:
                            ax.hlines(y=age_constraints_depths[i], xmin=0, xmax=age_constraints_ages[i], 
                                     color='r', linestyle='-', alpha=0.3)
                
                # Plot in-sequence constraints from different cores
                if np.any(diff_core_mask):
                    ax.errorbar(
                        age_constraints_ages[diff_core_mask], 
                        age_constraints_depths[diff_core_mask], 
                        xerr=[np.abs(age_constraints_neg_errors[diff_core_mask]), np.abs(age_constraints_pos_errors[diff_core_mask])], 
                        fmt='^', color='indigo',
                        label='Age Constraints (In Sequence; Adjacent Core)', 
                        capsize=4,
                        alpha=0.5,
                        markerfacecolor='w',
                        zorder=4
                    )
                    # Add horizontal lines from y-axis to in-sequence constraints from different cores
                    for i in range(len(age_constraints_depths)):
                        if diff_core_mask[i]:
                            ax.hlines(y=age_constraints_depths[i], xmin=0, xmax=age_constraints_ages[i], 
                                     color='indigo', linestyle='-', alpha=0.3)
            else:
                # Original behavior: Plot all in-sequence constraints the same way
                ax.errorbar(
                    age_constraints_ages[in_seq_mask], 
                    age_constraints_depths[in_seq_mask], 
                    xerr=[np.abs(age_constraints_neg_errors[in_seq_mask]), np.abs(age_constraints_pos_errors[in_seq_mask])], 
                    fmt='o', color='r', 
                    label='Age Constraints (In Sequence)', 
                    capsize=4,
                    alpha=0.5,
                    markerfacecolor='w',
                    zorder=4
                )
                # Add horizontal lines from y-axis to in-sequence constraints
                for i in range(len(age_constraints_depths)):
                    if in_seq_mask[i]:
                        ax.hlines(y=age_constraints_depths[i], xmin=0, xmax=age_constraints_ages[i], 
                                 color='r', linestyle='-', alpha=0.3)

        # Out-of-sequence constraints
        if age_constraints_in_sequence_flags is not None and np.any(~in_seq_mask):
            ax.errorbar(
                age_constraints_ages[~in_seq_mask], 
                age_constraints_depths[~in_seq_mask], 
                xerr=[np.abs(age_constraints_neg_errors[~in_seq_mask]), np.abs(age_constraints_pos_errors[~in_seq_mask])], 
                fmt='x', color='g', 
                label='Age Constraints (Out of Sequence)', 
                capsize=4,
                alpha=0.5,
                zorder=4
            )

        # Add dashed horizontal lines from y-axis to estimated ages
        for i in range(len(all_depths)):
            ax.hlines(y=all_depths[i], xmin=0, xmax=all_ages[i], color='b', linestyle=':', alpha=0.5)

        ax.invert_yaxis()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    return result


def check_age_constraint_compatibility(a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound,
                                      constraint_ages_a, constraint_ages_b,
                                      constraint_pos_errors_a, constraint_pos_errors_b, 
                                      constraint_neg_errors_a, constraint_neg_errors_b,
                                      ages_a=None, ages_b=None):
    """
    Check if segments in each core are compatible with age constraints by:
    1. Finding bracketing constraints for each segment in its own core
    2. Checking if the paired segment overlaps with those bracketing constraints
    """
    
    # Create extended constraints including top and bottom ages
    if ages_a and ages_b and len(ages_a['ages']) > 0 and len(ages_b['ages']) > 0:
        # Get top ages (youngest)
        top_age_a = ages_a['ages'][0]
        top_age_pos_error_a = ages_a['pos_uncertainties'][0]
        top_age_neg_error_a = ages_a['neg_uncertainties'][0]
        
        top_age_b = ages_b['ages'][0]
        top_age_pos_error_b = ages_b['pos_uncertainties'][0]
        top_age_neg_error_b = ages_b['neg_uncertainties'][0]
        
        # Get bottom ages (oldest)
        bottom_age_a = ages_a['ages'][-1]
        bottom_age_pos_error_a = ages_a['pos_uncertainties'][-1]
        bottom_age_neg_error_a = ages_a['neg_uncertainties'][-1]
        
        bottom_age_b = ages_b['ages'][-1]
        bottom_age_pos_error_b = ages_b['pos_uncertainties'][-1]
        bottom_age_neg_error_b = ages_b['neg_uncertainties'][-1]
        
        # Create extended constraints arrays including top and bottom
        ext_ages_a = np.concatenate(([top_age_a], constraint_ages_a, [bottom_age_a]))
        ext_pos_errors_a = np.concatenate(([top_age_pos_error_a], constraint_pos_errors_a, [bottom_age_pos_error_a]))
        ext_neg_errors_a = np.concatenate(([top_age_neg_error_a], constraint_neg_errors_a, [bottom_age_neg_error_a]))
        
        ext_ages_b = np.concatenate(([top_age_b], constraint_ages_b, [bottom_age_b]))
        ext_pos_errors_b = np.concatenate(([top_age_pos_error_b], constraint_pos_errors_b, [bottom_age_pos_error_b]))
        ext_neg_errors_b = np.concatenate(([top_age_neg_error_b], constraint_neg_errors_b, [bottom_age_neg_error_b]))
    else:
        # If ages_a or ages_b not provided, use original constraints
        ext_ages_a = constraint_ages_a
        ext_pos_errors_a = constraint_pos_errors_a
        ext_neg_errors_a = constraint_neg_errors_a
        
        ext_ages_b = constraint_ages_b
        ext_pos_errors_b = constraint_pos_errors_b
        ext_neg_errors_b = constraint_neg_errors_b
    
    # 1. For segment A: Find bracketing constraints in core A
    younger_constraints_a = ext_ages_a <= a_lower_bound
    older_constraints_a = ext_ages_a >= a_upper_bound
    
    # If we have both younger and older constraints, find the closest ones
    if np.any(younger_constraints_a) and np.any(older_constraints_a):
        # Find nearest younger constraint
        nearest_younger_a_idx = np.where(younger_constraints_a)[0][np.argmax(ext_ages_a[younger_constraints_a])]
        # Find nearest older constraint
        nearest_older_a_idx = np.where(older_constraints_a)[0][np.argmin(ext_ages_a[older_constraints_a])]
        
        # Get age ranges of these bracketing constraints with uncertainties
        bracket_a_lower = ext_ages_a[nearest_younger_a_idx] - ext_neg_errors_a[nearest_younger_a_idx]
        bracket_a_upper = ext_ages_a[nearest_older_a_idx] + ext_pos_errors_a[nearest_older_a_idx]
    
    # If only younger constraints exist, use youngest constraint and bottom age
    elif np.any(younger_constraints_a):
        nearest_younger_a_idx = np.where(younger_constraints_a)[0][np.argmax(ext_ages_a[younger_constraints_a])]
        bracket_a_lower = ext_ages_a[nearest_younger_a_idx] - ext_neg_errors_a[nearest_younger_a_idx]
        bracket_a_upper = ext_ages_a[-1] + ext_pos_errors_a[-1]  # Bottom age + uncertainty
    
    # If only older constraints exist, use oldest constraint and top age
    elif np.any(older_constraints_a):
        nearest_older_a_idx = np.where(older_constraints_a)[0][np.argmin(ext_ages_a[older_constraints_a])]
        bracket_a_lower = ext_ages_a[0] - ext_neg_errors_a[0]  # Top age - uncertainty
        bracket_a_upper = ext_ages_a[nearest_older_a_idx] + ext_pos_errors_a[nearest_older_a_idx]
    
    # If no bracketing constraints (segment covers all constraints), use full range
    else:
        bracket_a_lower = ext_ages_a[0] - ext_neg_errors_a[0]  # Top age - uncertainty
        bracket_a_upper = ext_ages_a[-1] + ext_pos_errors_a[-1]  # Bottom age + uncertainty
    
    # 2. For segment B: Find bracketing constraints in core B
    younger_constraints_b = ext_ages_b <= b_lower_bound
    older_constraints_b = ext_ages_b >= b_upper_bound
    
    # If we have both younger and older constraints, find the closest ones
    if np.any(younger_constraints_b) and np.any(older_constraints_b):
        # Find nearest younger constraint
        nearest_younger_b_idx = np.where(younger_constraints_b)[0][np.argmax(ext_ages_b[younger_constraints_b])]
        # Find nearest older constraint
        nearest_older_b_idx = np.where(older_constraints_b)[0][np.argmin(ext_ages_b[older_constraints_b])]
        
        # Get age ranges of these bracketing constraints with uncertainties
        bracket_b_lower = ext_ages_b[nearest_younger_b_idx] - ext_neg_errors_b[nearest_younger_b_idx]
        bracket_b_upper = ext_ages_b[nearest_older_b_idx] + ext_pos_errors_b[nearest_older_b_idx]
    
    # If only younger constraints exist, use youngest constraint and bottom age
    elif np.any(younger_constraints_b):
        nearest_younger_b_idx = np.where(younger_constraints_b)[0][np.argmax(ext_ages_b[younger_constraints_b])]
        bracket_b_lower = ext_ages_b[nearest_younger_b_idx] - ext_neg_errors_b[nearest_younger_b_idx]
        bracket_b_upper = ext_ages_b[-1] + ext_pos_errors_b[-1]  # Bottom age + uncertainty
    
    # If only older constraints exist, use oldest constraint and top age
    elif np.any(older_constraints_b):
        nearest_older_b_idx = np.where(older_constraints_b)[0][np.argmin(ext_ages_b[older_constraints_b])]
        bracket_b_lower = ext_ages_b[0] - ext_neg_errors_b[0]  # Top age - uncertainty
        bracket_b_upper = ext_ages_b[nearest_older_b_idx] + ext_pos_errors_b[nearest_older_b_idx]
    
    # If no bracketing constraints (segment covers all constraints), use full range
    else:
        bracket_b_lower = ext_ages_b[0] - ext_neg_errors_b[0]  # Top age - uncertainty
        bracket_b_upper = ext_ages_b[-1] + ext_pos_errors_b[-1]  # Bottom age + uncertainty
    
    # 3. Check if segment B overlaps with segment A's bracketing constraints
    b_overlaps_a_bracket = (b_lower_bound <= bracket_a_upper and b_upper_bound >= bracket_a_lower)
    
    # 4. Check if segment A overlaps with segment B's bracketing constraints
    a_overlaps_b_bracket = (a_lower_bound <= bracket_b_upper and a_upper_bound >= bracket_b_lower)
    
    # Both need to overlap to be compatible
    return b_overlaps_a_bracket and a_overlaps_b_bracket