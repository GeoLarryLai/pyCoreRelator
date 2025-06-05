"""
Animation and GIF creation functions for DTW visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import os
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
import csv
import random
from ..visualization.plotting import visualize_combined_segments


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
    Create an optimized animation of DTW correlations with full resolution.
    Optionally include age information when age_consideration is enabled.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    import os
    import gc
    from joblib import Parallel, delayed
    from tqdm import tqdm

    # Helper funcition: Create a global colormap function that uses the full log data
    def create_global_colormap(log_a, log_b):
        """
        Create a global colormap function based on the full normalized log data.
        
        Parameters:
            log_a (array): First normalized log data (0-1 range)
            log_b (array): Second normalized log data (0-1 range)
            
        Returns:
            function: A function that maps log values to consistent colors
        """
        # Since logs are already normalized to 0-1 range, we don't need to recalculate
        # the range but can directly create a function that maps to the yellow-brown spectrum
        # with consistent coloring across all segments
        
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
    
    #########################################################

    print("=== Starting Segment DTW Animation Creation ===")
    
    # Create output directory if it doesn't exist
    if not os.path.exists('SegmentPair_DTW_frames'):
        os.makedirs('SegmentPair_DTW_frames')
    
    # Process a subset of pairs if there are too many
    if valid_dtw_pairs and len(valid_dtw_pairs) > 0:
        valid_pairs_list = list(valid_dtw_pairs)
        total_pairs = len(valid_pairs_list)
        
        # Sample pairs if there are too many
        if total_pairs > max_frames:
            print(f"Using first {max_frames} frames from {total_pairs} total pairs")
            valid_pairs_list = valid_pairs_list[:max_frames]
        
        print(f"Processing {len(valid_pairs_list)} segment pairs for animation...")
        
        # Function to create a single frame
        def create_frame(pair_idx, pair):
            a_idx, b_idx = pair
            frame_filename = f"SegmentPair_DTW_frames/SegmentPair_{a_idx+1:04d}_{b_idx+1:04d}.png"
            
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
                # This reduces computational load without affecting resolution
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
                    single_segment_mode=True,  # Explicitly set to single segment mode
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
                    
                    # title += f"\nAge A: {age_a:.1f} yrs (+{age_a_pos_err:.1f}/-{age_a_neg_err:.1f}), " + \
                    #         f"Age B: {age_b:.1f} yrs (+{age_b_pos_err:.1f}/-{age_b_neg_err:.1f})"
                    
                    # Add correlation mode information
                    if restricted_age_correlation:
                        title += f"\nMode: Restricted Age Correlation"
                    else:
                        title += f"\nMode: Flexible Age Correlation"
                
                fig.suptitle(title, fontsize=12, y=1.02)
                
                # Save with full resolution (using original DPI of 150)
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
            # Use moderate number of jobs to prevent memory issues while maintaining performance
            n_jobs = -1  # Lower number of parallel jobs to prevent memory issues
            
            # Set verbose level based on debug parameter
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
                    # print(f"Creating GIF with {len(frames)} frames...")
                    # Save all frames in a single GIF regardless of the number
                    frames[0].save(output_filename, format='GIF', append_images=frames[1:], 
                                save_all=True, duration=500, loop=0)
                    
                    print(f"Created GIF animation: {output_filename}")
            except Exception as e:
                print(f"Error creating GIF: {e}")
                
            # Clean up PNG files if not keeping them
            if not keep_frames:
                for png in png_files:
                    try:
                        os.remove(png)
                    except:
                        pass
        else:
            print("No frames were successfully created.")
    else:
        print("No valid segment pairs found for animation")
    
    # Force garbage collection
    gc.collect()
    return output_filename



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
                                  # Age constraint visualization parameters (default None)
                                  age_constraint_a_source_cores=None,
                                  age_constraint_b_source_cores=None,
                                  core_a_name=None,
                                  core_b_name=None):
    """
    UPDATED: Handle new compact CSV format with fewer columns.
    age_constraint_a_source_cores : list or None, default=None
        List of source core names for each age constraint in core A.
        When provided, vertical constraint lines will be drawn in DTW matrix frames.
    age_constraint_b_source_cores : list or None, default=None
        List of source core names for each age constraint in core B.
        When provided, horizontal constraint lines will be drawn in DTW matrix frames.
    core_a_name, core_b_name : str or None, default=None
        Core names for determining same vs adjacent core coloring.
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

    # Create output directories
    os.makedirs("CombinedDTW_correlation_frames", exist_ok=True)
    os.makedirs("CombinedDTW_matrix_frames", exist_ok=True)
    
    # Clean existing files
    for frame_dir in ["CombinedDTW_correlation_frames", "CombinedDTW_matrix_frames"]:
        for file in os.listdir(frame_dir):
            if file.endswith(".png"):
                os.remove(os.path.join(frame_dir, file))
    
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
                    # UPDATED: Parse compact format instead of ast.literal_eval
                    segment_pairs = parse_compact_path(row[1])
                    selected_mappings.append((idx, mapping_id, segment_pairs))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing row {idx}: {e}")
    
    # Define processing function for joblib (rest remains the same)
    def process_mapping_visualization(frame_idx, mapping_id, mapping):
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
                correlation_save_path=f"CombinedDTW_correlation_frames/CombinedDTW_correlation_mappings_{frame_num}.png",
                matrix_save_path=f"CombinedDTW_matrix_frames/CombinedDTW_matrix_mappings_{frame_num}.png",
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
                # Age constraint visualization parameters (passed through)
                age_constraint_a_source_cores=age_constraint_a_source_cores,
                age_constraint_b_source_cores=age_constraint_b_source_cores,
                core_a_name=core_a_name,
                core_b_name=core_b_name
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
    n_jobs = min(os.cpu_count() or 4, 8)  # Use at most 8 cores
    
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
            # Make sure these are proper strings without trailing commas
            corr_gif_result = create_gif("CombinedDTW_correlation_frames", correlation_gif_output_filename)
            matrix_gif_result = create_gif("CombinedDTW_matrix_frames", matrix_gif_output_filename)
            print(corr_gif_result)
            print(matrix_gif_result)
            
            # Clean up PNG files if keepframes is False
            if not keep_frames:
                print("Cleaning up PNG files...")
                for frame_dir in ["CombinedDTW_correlation_frames", "CombinedDTW_matrix_frames"]:
                    for file in os.listdir(frame_dir):
                        if file.endswith(".png"):
                            os.remove(os.path.join(frame_dir, file))
                    print(f"Cleaned {frame_dir}")
        except Exception as e:
            print(f"Error creating GIFs: {e}")
            import traceback
            traceback.print_exc()  # Print detailed traceback for debugging