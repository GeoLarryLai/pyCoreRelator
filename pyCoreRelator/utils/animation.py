"""
Animation and GIF creation functions for DTW visualization.

Included Functions:
- create_gif: Create a GIF animation from PNG frames in a specified folder
- create_segment_dtw_animation: Create an optimized animation of DTW correlations between individual segment pairs

This module provides functionality for creating animated visualizations of DTW 
(Dynamic Time Warping) correlations between core segments, including GIF generation
from frame sequences and comprehensive animation creation with age constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import os
import gc
import shutil
from joblib import Parallel, delayed
from tqdm import tqdm
import csv
import random
from .plotting import visualize_combined_segments
from .plotting import plot_segment_pair_correlation


def create_gif(frame_folder, output_filename, duration=300):
    """
    Create a GIF animation from PNG frames in a specified folder.
    
    Args:
        frame_folder (str): Path to folder containing PNG frame files
        output_filename (str): Output GIF filename with path
        duration (int, optional): Duration between frames in milliseconds. Defaults to 300.
    
    Returns:
        str: Status message indicating success or failure
        
    Example:
        >>> result = create_gif('outputs/frames', 'outputs/animation.gif', duration=500)
        >>> print(result)
        Created GIF with 25 frames at outputs/animation.gif
    """
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])
    
    if not frame_files:
        return f"No frames found for GIF creation in {frame_folder}"
    
    # Open first image to get dimensions and mode
    first_img = PILImage.open(os.path.join(frame_folder, frame_files[0]))
    
    # Create GIF with append_images to avoid loading all frames at once
    frames_iterator = (PILImage.open(os.path.join(frame_folder, f)) for f in tqdm(frame_files[1:], 
                        desc=f"Processing frames for {output_filename}"))
    
    first_img.save(
        output_filename,
        format='GIF',
        append_images=frames_iterator,
        save_all=True,
        duration=duration,
        loop=0,
        optimize=False  # Faster processing
    )
    
    return f"Created GIF with {len(frame_files)} frames at {output_filename}"

def create_segment_dtw_animation(log_a, log_b, md_a, md_b, dtw_results, valid_dtw_pairs, 
                              segments_a, segments_b, depth_boundaries_a, depth_boundaries_b,
                              color_interval_size=None, 
                              keep_frames=True, output_filename='SegmentPair_DTW_animation.gif',
                              max_frames=100, parallel=True, debug=False,
                              age_consideration=False, ages_a=None, ages_b=None,
                              restricted_age_correlation=True,
                              all_constraint_depths_a=None, all_constraint_depths_b=None,
                              all_constraint_ages_a=None, all_constraint_ages_b=None,
                              all_constraint_pos_errors_a=None, all_constraint_pos_errors_b=None,
                              all_constraint_neg_errors_a=None, all_constraint_neg_errors_b=None):
    """
    Create an optimized animation of DTW correlations between individual segment pairs.
    
    This function generates frame-by-frame visualizations of DTW segment correlations
    with optional age constraints and creates a comprehensive GIF animation showing
    the correlation patterns across different segment pairs.
    
    Args:
        log_a, log_b (array): Normalized log data for cores A and B (0-1 range)
        md_a, md_b (array): Measured depth arrays for cores A and B
        dtw_results (dict): Dictionary containing DTW paths and quality indicators
        valid_dtw_pairs (set): Set of valid segment pair indices for correlation
        segments_a, segments_b (list): List of segment boundary indices
        depth_boundaries_a, depth_boundaries_b (array): Depth boundary arrays
        color_interval_size (int, optional): Step size for warping path visualization
        keep_frames (bool, optional): Whether to keep individual PNG frames. Defaults to True.
        output_filename (str, optional): Output GIF filename. Defaults to 'SegmentPair_DTW_animation.gif'.
        max_frames (int, optional): Maximum number of frames to generate. Defaults to 100.
        parallel (bool, optional): Whether to use parallel processing. Defaults to True.
        debug (bool, optional): Enable debug output. Defaults to False.
        age_consideration (bool, optional): Include age information in visualization. Defaults to False.
        ages_a, ages_b (dict, optional): Age data dictionaries with 'depths', 'ages', 'pos_uncertainties', 'neg_uncertainties'
        restricted_age_correlation (bool, optional): Use restricted age correlation mode. Defaults to True.
        all_constraint_depths_a, all_constraint_depths_b (array, optional): Constraint depth arrays
        all_constraint_ages_a, all_constraint_ages_b (array, optional): Constraint age arrays
        all_constraint_pos_errors_a, all_constraint_pos_errors_b (array, optional): Positive error arrays
        all_constraint_neg_errors_a, all_constraint_neg_errors_b (array, optional): Negative error arrays
    
    Returns:
        str: Path to the created GIF animation file
        
    Example:
        >>> output_path = create_segment_dtw_animation(
        ...     log_a, log_b, md_a, md_b, dtw_results, valid_pairs,
        ...     segments_a, segments_b, depths_a, depths_b,
        ...     max_frames=50, age_consideration=True,
        ...     ages_a=age_data_a, ages_b=age_data_b
        ... )
        >>> print(f"Animation saved to: {output_path}")
    """

    def create_global_colormap(log_a, log_b):
        """
        Create a global colormap function based on the full normalized log data.
        
        Parameters:
            log_a (array): First normalized log data (0-1 range)
            log_b (array): Second normalized log data (0-1 range)
            
        Returns:
            function: A function that maps log values to consistent colors
        """
        def global_color_function(log_value):
            """
            Generate a color in the yellow-brown spectrum based on log value using global mapping.
            
            Parameters:
                log_value (float): Value between 0-1 to determine color
                
            Returns:
                array: RGB color values in range 0-1
            """
            log_value = 1 - log_value

            color = np.array([1-0.4*log_value, 1-0.7*log_value, 0.6-0.6*log_value])
            color[color > 1] = 1
            color[color < 0] = 0
            return color
        
        return global_color_function

    print("=== Starting Segment DTW Animation Creation ===")
    
    # Create fresh frame directory
    frames_dir = 'SegmentPair_DTW_frames'
    
    # Remove existing frame directory if it exists, then create a fresh one
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        print(f"Removed existing frame directory: {frames_dir}")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Created fresh frame directory: {frames_dir}")

    # Process a subset of pairs if there are too many
    if valid_dtw_pairs and len(valid_dtw_pairs) > 0:
        valid_pairs_list = list(valid_dtw_pairs)
        total_pairs = len(valid_pairs_list)
        
        # Sample pairs if there are too many
        if total_pairs > max_frames:
            print(f"Using first {max_frames} frames from {total_pairs} total pairs")
            valid_pairs_list = valid_pairs_list[:max_frames]
        
        print(f"Processing {len(valid_pairs_list)} segment pairs for animation...")
        
        def create_frame(pair_idx, pair):
            """Create a single animation frame for a segment pair."""
            a_idx, b_idx = pair
            frame_filename = os.path.join(frames_dir, f"SegmentPair_{a_idx+1:04d}_{b_idx+1:04d}.png")

            # Skip if file already exists
            if os.path.exists(frame_filename):
                return frame_filename
            
            # Get paths and quality indicators
            paths, _, quality_indicators = dtw_results.get((a_idx, b_idx), ([], [], []))
            
            if not paths or len(paths) == 0:
                return None
            
            # Get segment boundaries
            a_start = depth_boundaries_a[segments_a[a_idx][0]]
            a_end = depth_boundaries_a[segments_a[a_idx][1]]
            b_start = depth_boundaries_b[segments_b[b_idx][0]]
            b_end = depth_boundaries_b[segments_b[b_idx][1]]
            
            # Check for single-point segments
            segment_a_len = a_end - a_start + 1
            segment_b_len = b_end - b_start + 1
            
            # Get warping path
            wp = paths[0]
            qi = quality_indicators[0] if quality_indicators else None
            
            try:
                # Use smaller step size for efficiency
                if color_interval_size is None:
                    step_size = max(1, min(5, len(wp) // 5))
                else:
                    step_size = int(color_interval_size)
                
                # Use non-interactive backend for better memory management
                plt.switch_backend('Agg')
                
                # Filter picked depths to only include those near the segment
                picked_depths_a = [idx for idx in range(len(depth_boundaries_a)) 
                                 if idx > 0 and idx < len(depth_boundaries_a) - 1 and 
                                 a_start - 20 <= depth_boundaries_a[idx] <= a_end + 20]
                
                picked_depths_b = [idx for idx in range(len(depth_boundaries_b)) 
                                 if idx > 0 and idx < len(depth_boundaries_b) - 1 and
                                 b_start - 20 <= depth_boundaries_b[idx] <= b_end + 20]
                
                # Create the global colormap once at the beginning
                global_colormap = create_global_colormap(log_a, log_b)
                
                # Create correlation plot using single segment mode with age information
                fig = plot_segment_pair_correlation(
                    log_a, log_b, md_a, md_b, 
                    single_segment_mode=True,
                    # Single segment mode parameters
                    wp=wp, a_start=a_start, a_end=a_end, b_start=b_start, b_end=b_end,
                    # Common parameters
                    step=step_size,
                    picked_depths_a=depth_boundaries_a,
                    picked_depths_b=depth_boundaries_b,
                    quality_indicators=qi,
                    color_function=global_colormap,
                    visualize_pairs=False,
                    visualize_segment_labels=False,
                    # Age-related parameters
                    age_consideration=age_consideration,
                    ages_a=ages_a,
                    ages_b=ages_b,
                    mark_ages=age_consideration, 
                    all_constraint_depths_a=all_constraint_depths_a,
                    all_constraint_depths_b=all_constraint_depths_b,
                    all_constraint_ages_a=all_constraint_ages_a,
                    all_constraint_ages_b=all_constraint_ages_b,
                    all_constraint_pos_errors_a=all_constraint_pos_errors_a,
                    all_constraint_pos_errors_b=all_constraint_pos_errors_b,
                    all_constraint_neg_errors_a=all_constraint_neg_errors_a,
                    all_constraint_neg_errors_b=all_constraint_neg_errors_b
                )

                # Add title with age information if available
                title = f"Segment Pair ({a_idx+1},{b_idx+1}): A[{a_start}:{a_end}] x B[{b_start}:{b_end}]\n" + \
                       f"A: {segment_a_len} points, B: {segment_b_len} points"
                
                # Add age information if enabled
                if age_consideration and ages_a and ages_b:
                    # Get boundary depths
                    a_boundary_depth = md_a[depth_boundaries_a[segments_a[a_idx][1]]]
                    b_boundary_depth = md_b[depth_boundaries_b[segments_b[b_idx][1]]]
                    
                    # Find index of nearest depth in ages_a and ages_b
                    a_age_idx = np.argmin(np.abs(np.array(ages_a['depths']) - a_boundary_depth))
                    b_age_idx = np.argmin(np.abs(np.array(ages_b['depths']) - b_boundary_depth))
                    
                    age_a = ages_a['ages'][a_age_idx]
                    age_b = ages_b['ages'][b_age_idx]
                    age_a_pos_err = ages_a['pos_uncertainties'][a_age_idx]
                    age_a_neg_err = ages_a['neg_uncertainties'][a_age_idx]
                    age_b_pos_err = ages_b['pos_uncertainties'][b_age_idx]
                    age_b_neg_err = ages_b['neg_uncertainties'][b_age_idx]
                    
                    # Add correlation mode information
                    if restricted_age_correlation:
                        title += f"\nMode: Restricted Age Correlation"
                    else:
                        title += f"\nMode: Flexible Age Correlation"
                
                fig.suptitle(title, fontsize=12, y=1.02)
                
                # Save with full resolution
                plt.savefig(frame_filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Force garbage collection
                gc.collect()
                
                return frame_filename
                
            except Exception as e:
                print(f"Error creating frame for pair ({a_idx+1},{b_idx+1}): {e}")
                return None
        
        # Process frames in parallel or sequentially
        if parallel:
            print("Generating frames in parallel...")
            n_jobs = -1
            verbose_level = 10 if debug else 0
            
            png_files = Parallel(n_jobs=n_jobs, verbose=verbose_level)(
                delayed(create_frame)(i, pair) for i, pair in enumerate(valid_pairs_list)
            )
        else:
            print("Generating frames sequentially...")
            png_files = []
            for i, pair in enumerate(tqdm(valid_pairs_list, desc="Creating frames")):
                frame = create_frame(i, pair)
                if frame:
                    png_files.append(frame)
                # More aggressive garbage collection in sequential mode
                if i % 5 == 0:  
                    gc.collect()
        
        # Filter out None values
        png_files = [f for f in png_files if f]
        
        if png_files:
            # Sort the PNG files
            png_files.sort()
            
            # Create GIF with full quality
            try:
                frames = []
                for png in tqdm(png_files, desc=f"Loading frames for creating GIF animation..."):
                    try:
                        img = PILImage.open(png)
                        frames.append(img)
                    except Exception as e:
                        print(f"Error opening {png}: {e}")
                    
                    # Free memory periodically while loading frames
                    if len(frames) % 10 == 0:
                        gc.collect()
                
                if frames:
                    # Save all frames in a single GIF
                    output_filepath = output_filename
                    
                    output_dir = os.path.dirname(output_filepath)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True) 
                    frames[0].save(output_filepath, format='GIF', append_images=frames[1:], 
                                save_all=True, duration=500, loop=0)
                    
                    print(f"Created GIF animation: {output_filepath}")
            except Exception as e:
                print(f"Error creating GIF: {e}")
                
            # Clean up PNG files and frame directory if not keeping them
            if not keep_frames:
                print("Cleaning up frame folder as keep_frames=False...")
                for png in png_files:
                    try:
                        os.remove(png)
                    except:
                        pass
                # Remove the frame directory itself
                try:
                    os.rmdir(frames_dir)
                    print(f"Removed frame directory: {frames_dir}")
                except:
                    pass
            else:
                print("Preserving frame folder as keep_frames=True...")
        else:
            print("No frames were successfully created.")
    else:
        print("No valid segment pairs found for animation")
    
    # Force garbage collection
    gc.collect()
    return output_filepath



def visualize_dtw_results_from_csv(csv_path, log_a, log_b, md_a, md_b, 
                                  dtw_results, valid_dtw_pairs, 
                                  segments_a, segments_b, 
                                  depth_boundaries_a, depth_boundaries_b,
                                  dtw_distance_matrix_full,
                                  color_interval_size=None,
                                  max_frames=150,
                                  debug=False,
                                  creategif=True,
                                  keep_frames=True,
                                  correlation_gif_output_filename="CombinedDTW_correlation_mappings.gif",
                                  matrix_gif_output_filename="CombinedDTW_matrix_mappings.gif",
                                  visualize_pairs=False,
                                  visualize_segment_labels=False,
                                  mark_depths=True, mark_ages=True,
                                  ages_a=None, ages_b=None,
                                  all_constraint_depths_a=None, all_constraint_depths_b=None,
                                  all_constraint_ages_a=None, all_constraint_ages_b=None,
                                  all_constraint_pos_errors_a=None, all_constraint_pos_errors_b=None,
                                  all_constraint_neg_errors_a=None, all_constraint_neg_errors_b=None,
                                  age_constraint_a_source_cores=None,
                                  age_constraint_b_source_cores=None,
                                  core_a_name=None,
                                  core_b_name=None,
                                  interpreted_bed_a=None,
                                  interpreted_bed_b=None):
    """
    Create comprehensive DTW visualization animations from CSV mapping results.
    
    This function processes CSV files containing DTW mapping results and generates
    two types of animated visualizations: correlation plots and distance matrix plots.
    It handles both correlation mappings and matrix visualizations with age constraints.
    
    Args:
        csv_path (str): Path to CSV file containing DTW mapping results
        log_a, log_b (array): Normalized log data for cores A and B
        md_a, md_b (array): Measured depth arrays for cores A and B
        dtw_results (dict): Dictionary containing DTW paths and quality indicators
        valid_dtw_pairs (set): Set of valid segment pair indices
        segments_a, segments_b (list): Lists of segment boundary indices
        depth_boundaries_a, depth_boundaries_b (array): Depth boundary arrays
        dtw_distance_matrix_full (array): Full DTW distance matrix
        color_interval_size (int, optional): Step size for warping path visualization
        max_frames (int, optional): Maximum number of frames to generate. Defaults to 150.
        debug (bool, optional): Enable debug output. Defaults to False.
        creategif (bool, optional): Whether to create GIF files. Defaults to True.
        keep_frames (bool, optional): Whether to keep individual PNG frames. Defaults to True.
        correlation_gif_output_filename (str, optional): Output filename for correlation GIF
        matrix_gif_output_filename (str, optional): Output filename for matrix GIF
        visualize_pairs (bool, optional): Whether to visualize segment pairs. Defaults to False.
        visualize_segment_labels (bool, optional): Whether to show segment labels. Defaults to False.
        mark_depths (bool, optional): Whether to mark depth boundaries. Defaults to True.
        mark_ages (bool, optional): Whether to mark age constraints. Defaults to True.
        ages_a, ages_b (dict, optional): Age data dictionaries
        all_constraint_depths_a, all_constraint_depths_b (array, optional): Constraint depth arrays
        all_constraint_ages_a, all_constraint_ages_b (array, optional): Constraint age arrays
        all_constraint_pos_errors_a, all_constraint_pos_errors_b (array, optional): Positive error arrays
        all_constraint_neg_errors_a, all_constraint_neg_errors_b (array, optional): Negative error arrays
        age_constraint_a_source_cores (list, optional): Source core names for age constraints in core A
        age_constraint_b_source_cores (list, optional): Source core names for age constraints in core B
        core_a_name, core_b_name (str, optional): Core names for constraint visualization
        interpreted_bed_a, interpreted_bed_b (array, optional): Interpreted bed names for core A and core B
    
    Returns:
        None: Creates GIF files and frame directories
        
    Example:
        >>> visualize_dtw_results_from_csv(
        ...     'outputs/dtw_mappings.csv',
        ...     log_a, log_b, md_a, md_b,
        ...     dtw_results, valid_pairs,
        ...     segments_a, segments_b,
        ...     depths_a, depths_b,
        ...     distance_matrix,
        ...     max_frames=100,
        ...     mark_ages=True,
        ...     ages_a=age_data_a,
        ...     ages_b=age_data_b
        ... )
    """
    import csv
    import os
    import gc
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    from joblib import Parallel, delayed
    import math
    import random

    def parse_compact_path(compact_path_str):
        """Parse compact path format "2,3;4,5;6,7" back to list of tuples"""
        if not compact_path_str or compact_path_str == "":
            return []
        return [tuple(map(int, pair.split(','))) for pair in compact_path_str.split(';')]

    # Use provided paths directly
    corr_gif_path = correlation_gif_output_filename
    matrix_gif_path = matrix_gif_output_filename
    
    # Only create frame directories if we're creating a GIF
    if creategif:
        # Create frame directories at the same level as the user-specified GIF names
        corr_user_dir = os.path.dirname(correlation_gif_output_filename) if os.path.dirname(correlation_gif_output_filename) else '.'
        matrix_user_dir = os.path.dirname(matrix_gif_output_filename) if os.path.dirname(matrix_gif_output_filename) else '.'
        
        correlation_frames_dir = os.path.join(corr_user_dir, 'CombinedDTW_correlation_frames')
        matrix_frames_dir = os.path.join(matrix_user_dir, 'CombinedDTW_matrix_frames')

        for frame_dir in [correlation_frames_dir, matrix_frames_dir]:
            # Remove existing frame directory if it exists, then create a fresh one
            if os.path.exists(frame_dir):
                shutil.rmtree(frame_dir)
                print(f"Removed existing frame directory: {frame_dir}")
            os.makedirs(frame_dir, exist_ok=True)
            print(f"Created fresh frame directory: {frame_dir}")
    else:
        # If not creating GIF, skip frame creation entirely
        print("Skipping frame creation as creategif=False")
        return
    
    # Count the total number of mappings without loading them all
    total_mappings = 0
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for _ in reader:
            total_mappings += 1
    
    print(f"Found {total_mappings} mappings in CSV")
    
    # Determine which mappings to visualize
    if total_mappings > max_frames:
        # Select evenly spaced indices
        step = total_mappings / max_frames
        selected_indices = set(int(i * step) for i in range(max_frames))
        print(f"Selected {len(selected_indices)} representative mappings for visualization")
    else:
        selected_indices = set(range(total_mappings))
    
    # Determine number of digits needed for frame numbering
    num_digits = len(str(len(selected_indices)))
    
    # Extract the selected mappings with a single pass through the file
    selected_mappings = []
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for idx, row in enumerate(reader):
            if idx in selected_indices:
                try:
                    mapping_id = int(row[0])
                    segment_pairs = parse_compact_path(row[1])
                    selected_mappings.append((idx, mapping_id, segment_pairs))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing row {idx}: {e}")
    
    def process_mapping_visualization(frame_idx, mapping_id, mapping):
        """Process a single mapping for visualization."""
        # Format the frame number with leading zeros
        frame_num = str(frame_idx + 1).zfill(num_digits)
        
        # Convert the 1-based mapping pairs back to 0-based for processing
        pairs_to_combine = [(a-1, b-1) for a, b in mapping]
        
        try:        
            # Generate visualizations for this mapping
            _, _, correlation_fig, matrix_fig = visualize_combined_segments(
                log_a, log_b, md_a, md_b, 
                dtw_results, valid_dtw_pairs, 
                segments_a, segments_b, 
                depth_boundaries_a, depth_boundaries_b,
                dtw_distance_matrix_full,
                pairs_to_combine,
                color_interval_size=color_interval_size,
                visualize_pairs=visualize_pairs,
                visualize_segment_labels=visualize_segment_labels,
                correlation_save_path=os.path.join(correlation_frames_dir, f"CombinedDTW_correlation_mappings_{frame_num}.png"),
                matrix_save_path=os.path.join(matrix_frames_dir, f"CombinedDTW_matrix_mappings_{frame_num}.png"),
                mark_depths=mark_depths,
                mark_ages=mark_ages,
                ages_a=ages_a,
                ages_b=ages_b,
                all_constraint_ages_a=all_constraint_ages_a,
                all_constraint_ages_b=all_constraint_ages_b,
                all_constraint_depths_a=all_constraint_depths_a,
                all_constraint_depths_b=all_constraint_depths_b,
                all_constraint_pos_errors_a=all_constraint_pos_errors_a,
                all_constraint_pos_errors_b=all_constraint_pos_errors_b,
                all_constraint_neg_errors_a=all_constraint_neg_errors_a,
                all_constraint_neg_errors_b=all_constraint_neg_errors_b,
                # Age constraint visualization parameters
                age_constraint_a_source_cores=age_constraint_a_source_cores,
                age_constraint_b_source_cores=age_constraint_b_source_cores,
                core_a_name=core_a_name,
                core_b_name=core_b_name,
                interpreted_bed_a=interpreted_bed_a,
                interpreted_bed_b=interpreted_bed_b
            )
            
            # Close figures to free memory
            plt.close(correlation_fig)
            plt.close(matrix_fig)
            
            # Force garbage collection
            gc.collect()
            
            return True, frame_num, mapping_id
        except Exception as e:
            print(f"Error processing mapping {mapping_id}: {e}")
            return False, frame_num, mapping_id
    
    # Process mappings in batches to manage memory
    batch_size = min(20, len(selected_mappings))
    batches = [selected_mappings[i:i+batch_size] for i in range(0, len(selected_mappings), batch_size)]
    
    # Set verbose level based on debug parameter
    verbose_level = 10 if debug else 0
    
    # Process all batches with parallel execution
    all_results = []
    print("Starting parallel visualization processing...")
    
    # Use fewer jobs to prevent memory issues with large datasets
    n_jobs = os.cpu_count() or 4
    
    with tqdm(total=len(batches), desc="Processing batches") as pbar:
        for batch_idx, batch in enumerate(batches):
            # Process each batch in parallel
            batch_results = Parallel(n_jobs=n_jobs, verbose=verbose_level)(
                delayed(process_mapping_visualization)(frame_idx, mapping_id, mapping) 
                for frame_idx, mapping_id, mapping in batch
            )
            all_results.extend(batch_results)
            pbar.update(1)
            
            # Periodic garbage collection
            if batch_idx % 5 == 0:
                gc.collect()
    
    # Count successful visualizations
    successful_mappings = sum(1 for result, _, _ in all_results if result)
    print(f"Successfully visualized {successful_mappings} out of {len(selected_mappings)} mappings")
    
    # Create GIFs from frames if requested
    if creategif:
        try:            
            print("\nCreating GIFs from frames...")
            # Create directories for both GIF outputs (paths already computed above)
            os.makedirs(os.path.dirname(corr_gif_path), exist_ok=True)
            os.makedirs(os.path.dirname(matrix_gif_path), exist_ok=True)
            
            corr_gif_result = create_gif(correlation_frames_dir, corr_gif_path)
            matrix_gif_result = create_gif(matrix_frames_dir, matrix_gif_path)
            print(corr_gif_result)
            print(matrix_gif_result)
            
            # Frame cleanup logic: 
            # if keep_frames=False, remove frame folders and files after GIF creation
            # if keep_frames=True, preserve frame folders and files after GIF creation
            if not keep_frames:
                print("Cleaning up frame folders as keep_frames=False...")
                for frame_dir in [correlation_frames_dir, matrix_frames_dir]:
                    if os.path.exists(frame_dir):
                        for file in os.listdir(frame_dir):
                            if file.endswith(".png"):
                                os.remove(os.path.join(frame_dir, file))
                        # Remove the directory itself
                        os.rmdir(frame_dir)
                        print(f"Removed frame directory: {frame_dir}")
            else:
                print("Preserving frame folders as keep_frames=True...")
        except Exception as e:
            print(f"Error creating GIFs: {e}")
            import traceback
            traceback.print_exc()