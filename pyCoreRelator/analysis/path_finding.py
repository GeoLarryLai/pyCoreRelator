"""
Path finding algorithms for core correlation analysis.

Included Functions:
- compute_total_complete_paths: Compute the total number of complete paths using dynamic programming.
- find_complete_core_paths: Find complete core paths through the segment network.

This module provides comprehensive path finding functionality for geological core 
correlation workflows, using dynamic programming and database optimization for 
efficient enumeration of correlation paths through large segment networks.
"""

import numpy as np
import pandas as pd
import os
import csv
import tempfile
import random
import heapq
from collections import deque, defaultdict
import gc
from tqdm import tqdm
from joblib import Parallel, delayed
import hashlib
import sys
import psutil
import threading
import sqlite3
import json
import random
import math
from .segments import identify_special_segments


def compute_total_complete_paths(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, mute_mode=False):
    """
    Compute the total number of complete paths using dynamic programming.
    
    A complete path goes from a top segment (starts at depth 0) to a bottom segment 
    (ends at maximum depth) through connected segments. Uses memoization to efficiently
    count paths without enumerating them.
    
    Parameters:
        valid_dtw_pairs (set): Valid segment pairs
        detailed_pairs (dict): Segment depth details
        max_depth_a (float): Maximum depth for core A  
        max_depth_b (float): Maximum depth for core B
        mute_mode (bool): If True, suppress all print output
        
    Returns:
        dict: Path computation results including:
            - total_complete_paths: Total number of complete paths
            - viable_segments: Segments excluding dead ends and orphans
            - viable_tops/bottoms: Lists of viable top/bottom segments
            - paths_from_tops: Path counts from each top segment
    
    Example:
        >>> results = compute_total_complete_paths(valid_pairs, details, 1000, 1200)
        >>> print(f"Total complete paths: {results['total_complete_paths']}")
        >>> for top, count in results['paths_from_tops'].items():
        ...     print(f"From top {top}: {count} paths")
    """
    
    # Get segment classifications
    top_segments, bottom_segments, dead_ends, orphans, successors, predecessors = identify_special_segments(
        valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b
    )
    
    # Filter viable segments (exclude problematic ones)
    viable_segments = set(valid_dtw_pairs) - set(dead_ends) - set(orphans)
    viable_tops = [seg for seg in top_segments if seg in viable_segments]
    viable_bottoms = [seg for seg in bottom_segments if seg in viable_segments]
    
    if not mute_mode:
        print(f"Viable segments (excluding dead ends and orphans): {len(viable_segments)}")
        print(f"Viable top segments: {len(viable_tops)}")
        print(f"Viable bottom segments: {len(viable_bottoms)}")
    
    if not viable_tops or not viable_bottoms:
        if not mute_mode:
            print("No viable complete paths possible")
        return {
            'total_complete_paths': 0,
            'viable_segments': viable_segments,
            'viable_tops': viable_tops,
            'viable_bottoms': viable_bottoms,
            'paths_from_tops': {}
        }
    
    # Dynamic programming to count paths
    path_count = {}
    
    # Initialize bottom segments with 1 path each
    for bottom_seg in viable_bottoms:
        if bottom_seg in viable_segments:
            path_count[bottom_seg] = 1
    
    def count_paths_from(segment, visited=None):
        """Recursively count paths from segment to any bottom with cycle detection."""
        if visited is None:
            visited = set()
        
        if segment in visited:  # Cycle detection
            return 0
        
        if segment in path_count:
            return path_count[segment]
        
        visited.add(segment)
        
        # Sum paths through all successors
        total_paths = 0
        for successor in successors.get(segment, []):
            if successor in viable_segments:
                total_paths += count_paths_from(successor, visited.copy())
        
        path_count[segment] = total_paths
        return total_paths
    
    # Calculate total paths from all viable top segments
    total_complete_paths = 0
    paths_from_tops = {}
    
    for top_seg in viable_tops:
        if top_seg in viable_segments:
            paths_from_top = count_paths_from(top_seg)
            paths_from_tops[top_seg] = paths_from_top
            total_complete_paths += paths_from_top
            if not mute_mode:
                print(f"  Paths from top segment ({top_seg[0]+1},{top_seg[1]+1}): {paths_from_top}")
    
    if not mute_mode:
        print(f"Total complete paths: {total_complete_paths}")
    
    return {
        'total_complete_paths': total_complete_paths,
        'viable_segments': viable_segments,
        'viable_tops': viable_tops,
        'viable_bottoms': viable_bottoms,
        'paths_from_tops': paths_from_tops
    }


def find_complete_core_paths(
    dtw_result,
    log_a,
    log_b,
    output_csv="complete_core_paths.csv",
    debug=False,
    start_from_top_only=True,
    batch_size=1000,
    n_jobs=-1,
    shortest_path_search=True,
    shortest_path_level=2,
    max_search_path=100000,
    output_metric_only=False,
    mute_mode=False,
    pca_for_dependent_dtw=False,
    metrics_to_compute=None,
    max_paths_for_metrics=None,
    return_dataframe=False
):
    """
    Find and enumerate all complete core-to-core correlation paths with advanced optimization features.
    
    Searches for paths that span from the top to bottom of both cores through connected segments.
    Includes memory management, duplicate removal, and performance optimizations for large datasets.
    
    Parameters:
        dtw_result (dict): Dictionary containing DTW analysis results from run_comprehensive_dtw_analysis().
            Expected keys: 'valid_dtw_pairs', 'segments_a', 'segments_b', 'depth_boundaries_a',
            'depth_boundaries_b', 'dtw_correlation', 'dtw_distance_matrix_full'
        log_a, log_b (array): Core log data for metric computation
        output_csv (str): Output CSV filename. Ignored if return_dataframe=True.
        debug (bool): Enable detailed progress reporting
        start_from_top_only (bool): Only start paths from top segments
        batch_size (int): Processing batch size
        n_jobs (int): Number of parallel jobs (-1 for all cores)
        shortest_path_search (bool): Keep only shortest path lengths during search
        shortest_path_level (int): Number of shortest unique lengths to keep
        max_search_path (int): Maximum complete paths to find before stopping. Default 100000.
        output_metric_only (bool): Only output quality metrics in the output CSV, no paths info
        mute_mode (bool): If True, suppress all print output
        pca_for_dependent_dtw (bool): If True, perform PCA for dependent DTW
        metrics_to_compute (list or str, optional): List of metrics to compute. 
            If None, computes default metrics: ['norm_dtw', 'corr_coef', 'norm_dtw_sect', 'corr_coef_sect'].
            If 'ALL', computes all available metrics.
            Available options:
            - 'norm_dtw': Normalized DTW distance (lower is better)
            - 'corr_coef': Correlation coefficient (higher is better)
            - 'norm_dtw_sect': Sectional normalized DTW (excludes pinch-outs)
            - 'corr_coef_sect': Sectional correlation coefficient (excludes pinch-outs)
            - 'dtw_ratio': DTW warping ratio (lower is better)
            - 'perc_diag': Path diagonality percentage (higher is better)
            - 'dtw_warp_eff': DTW warping efficiency
            - 'perc_age_overlap': Age overlap percentage (higher is better)
            Invalid metrics in the list are silently skipped.
        max_paths_for_metrics (int, optional): Maximum paths to compute metrics for.
            If total paths exceed this, a random sample is used. 
            If None, uses max_search_path value (no additional sampling).
        return_dataframe (bool): If True, return metrics as DataFrame instead of writing to file.
            This eliminates file I/O overhead for in-memory processing.
        
    Returns:
        dict: Comprehensive results including:
            - total_complete_paths_theoretical: Theoretical path count
            - total_complete_paths_found: Actually enumerated paths
            - viable_segments: Set of viable segments
            - output_csv: Path to generated CSV file (or None if return_dataframe=True)
            - duplicates_removed: Number of duplicates removed
            - search_limit_reached: Whether search limit was hit
            - metrics_dataframe: DataFrame with metrics (only if return_dataframe=True)
    
    Example:
        >>> dtw_result = run_comprehensive_dtw_analysis(...)
        >>> results = find_complete_core_paths(
        ...     dtw_result, log_a, log_b,
        ...     max_search_path=10000, debug=True
        ... )
        >>> print(f"Found {results['total_complete_paths_found']} complete paths")
        >>> print(f"Results saved to: {results['output_csv']}")
    """
    
    # Extract variables from unified dictionary
    valid_dtw_pairs = dtw_result['valid_dtw_pairs']
    segments_a = dtw_result['segments_a']
    segments_b = dtw_result['segments_b']
    depth_boundaries_a = dtw_result['depth_boundaries_a']
    depth_boundaries_b = dtw_result['depth_boundaries_b']
    dtw_results = dtw_result['dtw_correlation']
    dtw_distance_matrix_full = dtw_result['dtw_distance_matrix_full']

    # Import helper functions from path_helpers module
    from .path_helpers import (
        check_memory, calculate_diagonality, compress_path, decompress_path,
        remove_duplicates_from_db, filter_shortest_paths, compute_path_metrics_lazy,
        setup_database, insert_compressed_path, prune_shared_database_if_needed
    )

    # Create directory structure if needed
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Performance warning for unlimited search
    if max_search_path is None and not mute_mode:
        print("⚠️  WARNING: max_search_path=None can be very time consuming and require high memory usage!")
        print("   Consider setting max_search_path to a reasonable limit (e.g., 50000) for better performance.")

    # Setup boundary constraints and segment classification
    if not mute_mode:
        print("Setting up boundary constraints...")
    
    max_depth_a = max(depth_boundaries_a)
    max_depth_b = max(depth_boundaries_b)
    
    detailed_pairs = {}
    true_bottom_segments = set()
    true_top_segments = set()
    
    # Create detailed segment information
    for a_idx, b_idx in valid_dtw_pairs:
        a_start = depth_boundaries_a[segments_a[a_idx][0]]
        a_end = depth_boundaries_a[segments_a[a_idx][1]]
        b_start = depth_boundaries_b[segments_b[b_idx][0]]
        b_end = depth_boundaries_b[segments_b[b_idx][1]]
        
        detailed_pairs[(a_idx, b_idx)] = {
            'a_start': a_start,
            'a_end': a_end,
            'b_start': b_start,
            'b_end': b_end
        }
        
        # Identify true top and bottom segments
        if abs(a_end - max_depth_a) < 1e-6 and abs(b_end - max_depth_b) < 1e-6:
            true_bottom_segments.add((a_idx, b_idx))
        
        if abs(a_start) < 1e-6 and abs(b_start) < 1e-6:
            true_top_segments.add((a_idx, b_idx))
    
    # Filter for segments with valid DTW
    valid_top_segments = true_top_segments.intersection(valid_dtw_pairs)
    valid_bottom_segments = true_bottom_segments.intersection(valid_dtw_pairs)
    
    if not mute_mode:
        print(f"Identified {len(valid_top_segments)} valid segments at the top of both cores")
        print(f"Valid top segments (1-based indices): {[(a_idx+1, b_idx+1) for a_idx, b_idx in valid_top_segments]}")
        print(f"Identified {len(valid_bottom_segments)} valid segments at the bottom of both cores")
        print(f"Valid bottom segments (1-based indices): {[(a_idx+1, b_idx+1) for a_idx, b_idx in valid_bottom_segments]}")

    # Early exit if no complete paths possible
    if not true_bottom_segments:
        if not mute_mode:
            print("No segments found that contain the bottom of both cores. Cannot find complete paths.")
        return {
            'total_complete_paths_theoretical': 0,
            'total_complete_paths_found': 0,
            'viable_segments': set(),
            'viable_tops': [],
            'viable_bottoms': [],
            'output_csv': output_csv,
            'duplicates_removed': 0
        }
        
    if not true_top_segments:
        if not mute_mode:
            print("No segments found that contain the top of both cores. Cannot find complete paths.")
        return {
            'total_complete_paths_theoretical': 0,
            'total_complete_paths_found': 0,
            'viable_segments': set(),
            'viable_tops': [],
            'viable_bottoms': [],
            'output_csv': output_csv,
            'duplicates_removed': 0
        }

    # Compute theoretical path count
    if not mute_mode:
        print(f"\n=== COMPLETE PATH COMPUTATION ===")
    path_computation_results = compute_total_complete_paths(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, mute_mode=mute_mode)

    # Build segment relationships
    if not mute_mode:
        print("\nBuilding segment relationships...")
    predecessor_lookup = defaultdict(list)
    successor_lookup = defaultdict(list)
    
    for a_idx, b_idx in valid_dtw_pairs:
        pair_details = detailed_pairs[(a_idx, b_idx)]
        a_end = pair_details['a_end']
        b_end = pair_details['b_end']
        
        for next_a_idx, next_b_idx in valid_dtw_pairs:
            if (a_idx, b_idx) != (next_a_idx, next_b_idx):
                next_details = detailed_pairs[(next_a_idx, next_b_idx)]
                next_a_start = next_details['a_start']
                next_b_start = next_details['b_start']
                
                # Check exact depth matching for connectivity
                if (abs(next_a_start - a_end) < 1e-6 and 
                    abs(next_b_start - b_end) < 1e-6):
                    successor_lookup[(a_idx, b_idx)].append((next_a_idx, next_b_idx))
                    predecessor_lookup[(next_a_idx, next_b_idx)].append((a_idx, b_idx))
    
    # Filter starting segments if requested
    final_top_segments = true_top_segments
    if start_from_top_only:
        allowed_top_pairs = [(1,0), (1,1), (0,1)]
        final_top_segments = {seg for seg in true_top_segments if seg in allowed_top_pairs}
        
    if not mute_mode:
        print(f"Using {len(final_top_segments)} valid top segments for path starting points")
    
    # Topological ordering for processing
    def topological_sort():
        """Create topological ordering of segments for efficient processing."""
        visited = set()
        temp_visited = set()
        order = []
        
        def dfs(segment):
            if segment in temp_visited:
                return False  # Cycle detected
            
            if segment in visited:
                return True
                
            temp_visited.add(segment)
            
            # Visit successors first
            for next_segment in successor_lookup[segment]:
                if not dfs(next_segment):
                    return False
            
            temp_visited.remove(segment)
            visited.add(segment)
            order.append(segment)
            return True
        
        # Start from top segments
        for segment in final_top_segments:
            if segment not in visited:
                if not dfs(segment):
                    if not mute_mode:
                        print("Warning: Cycle detected in segment relationships. Using BFS ordering instead.")
                    return None
        
        # Process remaining segments
        for segment in valid_dtw_pairs:
            if segment not in visited:
                if not dfs(segment):
                    if not mute_mode:
                        print("Warning: Cycle detected in segment relationships. Using BFS ordering instead.")
                    return None
        
        return list(reversed(order))  # Reverse for top-to-bottom order
    
    # Get processing order
    topo_order = topological_sort()
    
    if topo_order is None:
        # Fall back to level-based ordering
        if not mute_mode:
            print("Using level-based ordering instead of topological sort...")
        
        levels = {}
        queue = deque([(seg, 0) for seg in final_top_segments])
        
        while queue:
            segment, level = queue.popleft()
            
            if segment in levels:
                continue
                
            levels[segment] = level
            
            for next_segment in successor_lookup[segment]:
                if next_segment not in levels:
                    queue.append((next_segment, level + 1))
        
        topo_order = sorted(valid_dtw_pairs, key=lambda seg: levels.get(seg, float('inf')))
    
    if not mute_mode:
        print(f"Identified {len(topo_order)} segments in processing order")
    
    # Database setup
    temp_dir = tempfile.mkdtemp()
    if not mute_mode:
        print(f"Created temporary directory for databases: {temp_dir}")
    
    shared_read_db_path = os.path.join(temp_dir, "shared_read.db")
    shared_read_conn = setup_database(shared_read_db_path, read_only=False)
    
    # Initialize with top segments
    if not mute_mode:
        print("Initializing shared database with top segments...")
    for segment in final_top_segments:
        compressed_path = compress_path([segment])
        insert_compressed_path(shared_read_conn, segment, segment, compressed_path, 1, False)
    shared_read_conn.commit()
    
    # Create processing groups (always 1 segment per group for complete enumeration)
    segment_groups = []
    current_group = []
    
    for segment in topo_order:
        current_group.append(segment)
        if len(current_group) >= 1:  
            segment_groups.append(current_group)
            current_group = []
    
    if current_group:
        segment_groups.append(current_group)
    
    if not mute_mode:
        print(f"Processing {len(topo_order)} segments in {len(segment_groups)} groups (1 segment per group for complete enumeration)")
    
    # Initialize path tracking
    complete_paths_found = 0
    search_limit_reached = False
    
    def process_segment_group_with_database_and_dedup(group_idx, segment_group, shared_read_conn):
        """Process a group of segments with optimized database operations and path pruning."""
        nonlocal complete_paths_found, search_limit_reached
        
        # Use in-memory database for temporary storage
        group_write_conn = sqlite3.connect(":memory:")
        
        # Performance optimizations for temporary database
        group_write_conn.execute("PRAGMA synchronous = OFF")
        group_write_conn.execute("PRAGMA journal_mode = MEMORY")
        group_write_conn.execute("PRAGMA cache_size = 50000")
        group_write_conn.execute("PRAGMA temp_store = MEMORY")
        
        # Create table structure
        group_write_conn.execute("""
            CREATE TABLE compressed_paths (
                start_segment TEXT NOT NULL,
                last_segment TEXT NOT NULL,
                compressed_path TEXT NOT NULL,
                length INTEGER NOT NULL,
                is_complete BOOLEAN DEFAULT 0
            )
        """)
        
        group_write_conn.execute("CREATE INDEX idx_compressed_path ON compressed_paths(compressed_path)")
        
        batch_inserts = []
        complete_paths_count = 0
        

        def prune_paths_if_needed():
            """Prune intermediate paths when they exceed max_search_path limit using random sampling."""
            nonlocal complete_paths_found, search_limit_reached
            
            if max_search_path is None:
                return 0  # No pruning needed if no limit set
            
            # Count only intermediate paths (is_complete = 0)
            cursor = group_write_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 0")
            current_intermediate_count = cursor.fetchone()[0]
            
            if current_intermediate_count <= max_search_path:
                return 0  # No pruning needed
            
            paths_to_remove = current_intermediate_count - max_search_path
            
            if debug and not mute_mode:
                print(f"    Path pruning: {current_intermediate_count} intermediate paths exceed limit of {max_search_path}")
                print(f"    Randomly excluding {paths_to_remove} intermediate paths")
            
            # Get only intermediate paths with their rowids for random selection
            cursor = group_write_conn.execute("""
                SELECT rowid, start_segment 
                FROM compressed_paths
                WHERE is_complete = 0
            """)
            intermediate_paths = cursor.fetchall()
            
            # Randomly select intermediate paths to remove
            if len(intermediate_paths) > max_search_path:
                selected_for_removal = random.sample(intermediate_paths, paths_to_remove)
                rowids_to_remove = [rowid for rowid, _ in selected_for_removal]
                
                # Remove selected intermediate paths from database
                if rowids_to_remove:
                    placeholders = ",".join("?" * len(rowids_to_remove))
                    group_write_conn.execute(f"""
                        DELETE FROM compressed_paths 
                        WHERE rowid IN ({placeholders})
                    """, rowids_to_remove)
                    
                    group_write_conn.commit()
                    
                    if debug and not mute_mode:
                        print(f"    Removed {len(rowids_to_remove)} intermediate paths")
                        # Verify final count
                        cursor = group_write_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 0")
                        final_count = cursor.fetchone()[0]
                        print(f"    Final intermediate path count after pruning: {final_count}")
                
                return len(rowids_to_remove)
            
            return 0
        
        # Process each segment in the group
        for segment in segment_group:
            # Get predecessor paths
            direct_predecessors = predecessor_lookup[segment]
            predecessor_paths = []
            
            if direct_predecessors:
                # Batch read predecessor paths
                placeholders = ",".join("?" * len(direct_predecessors))
                pred_strings = [f"{a},{b}" for a, b in direct_predecessors]
                
                cursor = shared_read_conn.execute(f"""
                    SELECT compressed_path FROM compressed_paths 
                    WHERE last_segment IN ({placeholders})
                """, pred_strings)
                
                predecessor_paths = [row[0] for row in cursor.fetchall()]
            
            # For top segments, start with singleton paths
            if not predecessor_paths and segment in final_top_segments:
                compressed_path = compress_path([segment])
                predecessor_paths = [compressed_path]
            
            if not predecessor_paths:
                continue
            
            # STEP 1: Generate ALL possible intermediate paths for current segment
            new_paths_data = []
            
            for compressed_pred_path in predecessor_paths:
                pred_path = decompress_path(compressed_pred_path)
                
                if not pred_path or pred_path[-1] != segment:
                    extended_path = pred_path + [segment]
                else:
                    extended_path = pred_path
                
                compressed_extended_path = compress_path(extended_path)
                is_complete = segment in true_bottom_segments
                
                new_paths_data.append((compressed_extended_path, len(extended_path), is_complete))
                
                if is_complete:
                    complete_paths_count += 1
            
            # STEP 2: Apply shortest path filtering if enabled
            if shortest_path_search:
                new_paths_data = filter_shortest_paths(new_paths_data, shortest_path_level, debug, mute_mode)
            
            # STEP 3: Apply random pruning to intermediate paths if exceeding limit
            if max_search_path is not None:
                # Separate complete and intermediate paths
                complete_paths = [(path, length, is_complete) for path, length, is_complete in new_paths_data if is_complete]
                intermediate_paths = [(path, length, is_complete) for path, length, is_complete in new_paths_data if not is_complete]
                
                # If intermediate paths exceed limit, randomly exclude excess
                if len(intermediate_paths) > max_search_path:
                    if debug and not mute_mode:
                        print(f"  Segment ({segment[0]+1},{segment[1]+1}): {len(intermediate_paths)} intermediate paths exceed limit of {max_search_path}")
                        print(f"  Randomly excluding {len(intermediate_paths) - max_search_path} intermediate paths")
                    
                    # Keep all complete paths + randomly sampled intermediate paths
                    sampled_intermediate = random.sample(intermediate_paths, max_search_path)
                    new_paths_data = complete_paths + sampled_intermediate
                else:
                    # Keep all paths
                    new_paths_data = complete_paths + intermediate_paths
            
            # STEP 4: Convert to batch inserts and store
            for compressed_extended_path, length, is_complete in new_paths_data:
                extended_path = decompress_path(compressed_extended_path)
                
                batch_inserts.append((
                    f"{extended_path[0][0]},{extended_path[0][1]}",
                    f"{extended_path[-1][0]},{extended_path[-1][1]}", 
                    compressed_extended_path,
                    length,
                    is_complete
                ))
                
                # Batch insert when batch gets large
                if len(batch_inserts) >= 5000:
                    group_write_conn.executemany("""
                        INSERT INTO compressed_paths (start_segment, last_segment, compressed_path, length, is_complete)
                        VALUES (?, ?, ?, ?, ?)
                    """, batch_inserts)
                    batch_inserts = []
        
        # Insert any remaining batch
        if batch_inserts:
            group_write_conn.executemany("""
                INSERT INTO compressed_paths (start_segment, last_segment, compressed_path, length, is_complete)
                VALUES (?, ?, ?, ?, ?)
            """, batch_inserts)
        
        # Remove duplicates
        duplicates_removed = remove_duplicates_from_db(group_write_conn, debug, mute_mode)
        
        # Recalculate complete paths after deduplication
        cursor = group_write_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 1")
        complete_paths_count_after_dedup = cursor.fetchone()[0]
        
        return group_write_conn, complete_paths_count_after_dedup, duplicates_removed
    
    # Process all groups with optimization
    total_complete_paths = 0
    total_duplicates_removed = 0
    
    # Determine sync frequency
    sync_every_n_groups = 1
    if debug and not mute_mode:
        processing_msg = "syncing after every segment with incremental duplicate removal for maximum reliability"
        
        optimization_msgs = []
        if shortest_path_search:
            optimization_msgs.append(f"\n- shortest path search (keeping {shortest_path_level} shortest lengths)")
        if max_search_path is not None:
            optimization_msgs.append(f"\n- intermediate path limit ({max_search_path} intermediate paths per step)")
        
        if optimization_msgs:
            processing_msg += f" with {' and '.join(optimization_msgs)}"
        
        print(f"Processing mode: {processing_msg}")
    
    # Main processing loop
    if not mute_mode:
        pbar = tqdm(total=len(segment_groups), desc="Processing segment groups")
    group_results = []
    
    for group_idx, segment_group in enumerate(segment_groups):
        
        if search_limit_reached:
            if debug and not mute_mode:
                print(f"Stopping processing due to search limit reached")
            break
        
        # Process group
        group_write_conn, group_complete_paths, group_duplicates = process_segment_group_with_database_and_dedup(
            group_idx, segment_group, shared_read_conn
        )
        
        group_results.append((group_write_conn, group_complete_paths, group_duplicates))
        total_complete_paths += group_complete_paths
        total_duplicates_removed += group_duplicates
        
        # Determine if should sync
        should_sync = (
            (group_idx + 1) % sync_every_n_groups == 0 or
            group_idx == len(segment_groups) - 1 or
            search_limit_reached
        )

        if should_sync:
            if len(group_results) > 1:
                if debug and not mute_mode:
                    print(f"Syncing {len(group_results)} groups to shared database...")
            
            # Bulk transfer from group databases
            for group_conn, _, _ in group_results:
                cursor = group_conn.execute("""
                    SELECT start_segment, last_segment, compressed_path, length, is_complete 
                    FROM compressed_paths
                """)
                
                all_rows = cursor.fetchall()
                
                if all_rows:
                    shared_read_conn.executemany("""
                        INSERT INTO compressed_paths (start_segment, last_segment, compressed_path, length, is_complete)
                        VALUES (?, ?, ?, ?, ?)
                    """, all_rows)
                
                group_conn.close()

            shared_read_conn.commit()

            # **NEW: Apply pruning to shared database after sync**
            if max_search_path is not None:
                shared_pruned = prune_shared_database_if_needed(shared_read_conn, max_search_path, debug, mute_mode)
                if shared_pruned > 0 and debug and not mute_mode:
                    print(f"  Pruned {shared_pruned} paths from shared database after sync")

            # Remove duplicates after sync
            if len(group_results) > 1:
                shared_duplicates = remove_duplicates_from_db(shared_read_conn, debug, mute_mode)
                total_duplicates_removed += shared_duplicates
            
            # Clear the results batch
            group_results = []
            
            # Garbage collection (every 10 segments)
            if group_idx % 10 == 0:
                gc.collect()
        
        # Update progress
        if not mute_mode:
            pbar.update(1)

        # Get current counts from shared database
        cursor = shared_read_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 0")
        current_intermediate_paths = cursor.fetchone()[0]

        cursor = shared_read_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 1")
        current_complete_paths = cursor.fetchone()[0]

        # Update progress bar with current statistics
        if not mute_mode:
            postfix_dict = {
                "segment": f"{group_idx + 1}/{len(segment_groups)}",
                "intermediate_paths": f"{current_intermediate_paths}/{max_search_path}" if max_search_path is not None else f"{current_intermediate_paths}",
                "complete_paths_found": current_complete_paths,
                "duplicates_removed": total_duplicates_removed
            }
            pbar.set_postfix(postfix_dict)
    
    if not mute_mode:
        pbar.close()
    
    # Final deduplication on shared database
    if not mute_mode:
        print("Performing final deduplication on complete database...")
    final_duplicates = remove_duplicates_from_db(shared_read_conn, debug, mute_mode)
    total_duplicates_removed += final_duplicates
    
    # Get final count after all deduplication
    cursor = shared_read_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 1")
    final_complete_paths = cursor.fetchone()[0]
    
    # Print completion message with search limit information
    if not mute_mode:
        completion_msg = f"Processing complete. Found {final_complete_paths} unique complete paths after removing {total_duplicates_removed} duplicates."
        if search_limit_reached:
            completion_msg += f" (Search stopped at limit of {max_search_path} complete paths)"
        print(completion_msg)
    
    # Direct output generation from deduplicated database
    if not mute_mode:
        print("\n=== Computing Metrics and Generating CSV Output ===")
    
    # Process metrics_to_compute parameter
    all_available_metrics = ['norm_dtw', 'dtw_ratio', 'perc_diag', 'dtw_warp_eff', 'corr_coef', 'perc_age_overlap', 'norm_dtw_sect', 'corr_coef_sect']
    default_metrics = ['norm_dtw', 'corr_coef', 'norm_dtw_sect', 'corr_coef_sect']
    
    if metrics_to_compute is None:
        # Default to the 4 primary metrics
        metrics_to_compute_final = default_metrics
    elif isinstance(metrics_to_compute, str) and metrics_to_compute.upper() == 'ALL':
        # Compute all available metrics
        metrics_to_compute_final = all_available_metrics
    else:
        # Filter to only valid metrics, skip invalid ones silently
        metrics_to_compute_final = [m for m in metrics_to_compute if m in all_available_metrics]
        if not metrics_to_compute_final:
            # If no valid metrics, fall back to default
            metrics_to_compute_final = default_metrics
    
    # Process max_paths_for_metrics parameter
    # If None, use max_search_path (no additional sampling beyond path enumeration limit)
    max_paths_for_metrics_final = max_paths_for_metrics if max_paths_for_metrics is not None else max_search_path
    
    # Create output CSV with batch processing for memory efficiency
    def generate_output_csv():
        """Generate final output directly from deduplicated database using parallel processing."""
        
        # Determine output mode
        use_pickle = not return_dataframe and output_csv.lower().endswith('.pkl')
        use_dataframe = return_dataframe
        
        # Define column names
        if output_metric_only:
            columns = ['mapping_id', 'length', 'norm_dtw', 'dtw_ratio', 'perc_diag', 'dtw_warp_eff', 'corr_coef', 'perc_age_overlap', 'norm_dtw_sect', 'corr_coef_sect']
        else:
            columns = ['mapping_id', 'path', 'length', 'combined_wp', 'norm_dtw', 'dtw_ratio', 'perc_diag', 'dtw_warp_eff', 'corr_coef', 'perc_age_overlap', 'norm_dtw_sect', 'corr_coef_sect']
        
        # Initialize storage for results (for pickle, dataframe, or in-memory modes)
        all_results = []
        
        # For CSV file output, create file with header
        if not use_pickle and not use_dataframe:
            with open(output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
        
        # Get total number of complete paths for progress reporting
        cursor = shared_read_conn.execute("""
            SELECT COUNT(*) FROM compressed_paths 
            WHERE is_complete = 1
        """)
        total_paths = cursor.fetchone()[0]
        
        # Process complete paths in larger batches for parallel processing
        cursor = shared_read_conn.execute("""
            SELECT compressed_path, length, path_id FROM compressed_paths 
            WHERE is_complete = 1 
            ORDER BY length, start_segment
        """)
        
        # Get all paths to process
        all_paths_from_db = cursor.fetchall()
        
        # Solution 1: Sample paths if total exceeds max_paths_for_metrics_final
        paths_to_process = all_paths_from_db
        if len(all_paths_from_db) > max_paths_for_metrics_final:
            if not mute_mode:
                print(f"Sampling {max_paths_for_metrics_final} paths from {len(all_paths_from_db)} total for metric computation")
            paths_to_process = random.sample(all_paths_from_db, max_paths_for_metrics_final)
        
        # Use the function's parameters for batch size and number of jobs
        n_jobs_to_use = os.cpu_count() if n_jobs == -1 else n_jobs
        batch_size_to_use = batch_size if batch_size > 0 else 500
        
        # Create batches of paths
        batches = [paths_to_process[i:i + batch_size_to_use] for i in range(0, len(paths_to_process), batch_size_to_use)]
        
        # Process a batch of paths (designed to be parallelizable)
        def process_batch(batch, start_id):
            batch_results = []
            mapping_id = start_id
            
            for compressed_path, length, _ in batch:
                # Decompress and format path
                full_path = decompress_path(compressed_path)
                
                # Convert to 1-based and use semicolon separator for compactness
                formatted_path_compact = ";".join(f"{a+1},{b+1}" for a, b in full_path)
                
                # Compute metrics and warping path (with selective metrics - Solution 5)
                combined_wp, metrics = compute_path_metrics_lazy(
                    compressed_path, log_a, log_b, dtw_results, dtw_distance_matrix_full, 
                    pca_for_dependent_dtw=pca_for_dependent_dtw,
                    segments_a=segments_a, segments_b=segments_b,
                    depth_boundaries_a=depth_boundaries_a, depth_boundaries_b=depth_boundaries_b,
                    metrics_to_compute=metrics_to_compute_final
                )
                
                # Format warping path compactly
                if combined_wp is not None and len(combined_wp) > 0:
                    combined_wp_compact = ";".join(f"{int(wp[0])},{int(wp[1])}" for wp in combined_wp)
                else:
                    combined_wp_compact = ""
                
                # Add result
                if output_metric_only:
                    batch_results.append([
                        mapping_id, 
                        length,
                        round(metrics['norm_dtw'], 6),
                        round(metrics['dtw_ratio'], 6),
                        round(metrics['perc_diag'], 2),
                        round(metrics['dtw_warp_eff'], 6),
                        round(metrics['corr_coef'], 6),
                        round(metrics['perc_age_overlap'], 2),
                        round(metrics['norm_dtw_sect'], 6),
                        round(metrics['corr_coef_sect'], 6)
                    ])
                else:
                    batch_results.append([
                        mapping_id, 
                        formatted_path_compact,
                        length,
                        combined_wp_compact,
                        round(metrics['norm_dtw'], 6),
                        round(metrics['dtw_ratio'], 6),
                        round(metrics['perc_diag'], 2),
                        round(metrics['dtw_warp_eff'], 6),
                        round(metrics['corr_coef'], 6),
                        round(metrics['perc_age_overlap'], 2),
                        round(metrics['norm_dtw_sect'], 6),
                        round(metrics['corr_coef_sect'], 6)
                    ])
                
                mapping_id += 1
                
            return batch_results
        
        paths_being_processed = len(paths_to_process)
        if not mute_mode:
            print(f"Processing {paths_being_processed} paths in {len(batches)} batches using {n_jobs_to_use} parallel jobs")
        
        # Solution 4: Parallel batch processing
        if n_jobs_to_use > 1 and len(batches) > 1:
            # Parallel processing using joblib
            batch_args = [(batch, batch_idx * batch_size_to_use + 1) for batch_idx, batch in enumerate(batches)]
            
            if not mute_mode:
                parallel_results = Parallel(n_jobs=n_jobs_to_use, verbose=0)(
                    delayed(process_batch)(batch, start_id) 
                    for batch, start_id in tqdm(batch_args, desc="Processing batches (parallel)")
                )
            else:
                parallel_results = Parallel(n_jobs=n_jobs_to_use, verbose=0)(
                    delayed(process_batch)(batch, start_id) 
                    for batch, start_id in batch_args
                )
            
            # Collect all results
            for batch_results in parallel_results:
                if use_pickle or use_dataframe:
                    all_results.extend(batch_results)
                else:
                    # For CSV: write batch results immediately
                    with open(output_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(batch_results)
        else:
            # Sequential processing (original behavior for n_jobs=1 or single batch)
            if not mute_mode:
                pbar = tqdm(total=len(batches), desc="Processing batches")
            
            for batch_idx, batch in enumerate(batches):
                start_id = batch_idx * batch_size_to_use + 1
                batch_results = process_batch(batch, start_id)
                
                if use_pickle or use_dataframe:
                    all_results.extend(batch_results)
                else:
                    with open(output_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(batch_results)
                
                if not mute_mode:
                    pbar.update(1)
                    
                if batch_idx % 5 == 0:
                    gc.collect()
            
            if not mute_mode:
                pbar.close()
        
        # Create DataFrame from results
        result_df = pd.DataFrame(all_results, columns=columns) if all_results else pd.DataFrame(columns=columns)
        
        # Solution 2: Return DataFrame or save to file
        if use_dataframe:
            # Return DataFrame directly (no file I/O)
            pass  # result_df will be returned
        elif use_pickle:
            result_df.to_pickle(output_csv)
        
        # Clean up
        if not use_dataframe:
            del all_results
            gc.collect()
        
        return paths_being_processed, result_df if use_dataframe else None
    
    # Generate the output
    total_paths_written, metrics_df = generate_output_csv()
    
    # Close shared database
    shared_read_conn.close()
    
    # Print final statistics
    if not mute_mode:
        print(f"\nFinal Results:")
        print(f"  Total unique complete paths processed: {total_paths_written}")
        print(f"  Total duplicates removed during processing: {total_duplicates_removed}")
        print(f"  Deduplication efficiency: {(total_duplicates_removed/(total_paths_written + total_duplicates_removed)*100) if (total_paths_written + total_duplicates_removed) > 0 else 0:.2f}%")
        
        # Add search limit information to final results
        if search_limit_reached:
            print(f"  Search was limited to {max_search_path} complete paths for performance")
        
        # Add sampling information if applicable
        if max_paths_for_metrics is not None:
            print(f"  Metrics computed for sampled subset: {total_paths_written} paths")
    
    # Cleanup - remove all temporary files
    try:
        if not mute_mode:
            print("Cleaning up temporary databases...")
        import shutil
        shutil.rmtree(temp_dir)
        if not mute_mode:
            print("Cleanup complete.")
    except Exception as e:
        if not mute_mode:
            print(f"Could not clean temporary directory: {e}")
    
    if not mute_mode and not return_dataframe:
        print(f"All complete core-to-core paths saved to {output_csv}")
    
    # Return comprehensive results dictionary
    result_dict = {
        'total_complete_paths_theoretical': path_computation_results['total_complete_paths'],
        'total_complete_paths_found': total_paths_written,
        'viable_segments': path_computation_results['viable_segments'],
        'viable_tops': path_computation_results['viable_tops'],
        'viable_bottoms': path_computation_results['viable_bottoms'],
        'paths_from_tops': path_computation_results['paths_from_tops'],
        'output_csv': output_csv if not return_dataframe else None,
        'duplicates_removed': total_duplicates_removed,
        'search_limit_reached': search_limit_reached,
        'metrics_dataframe': metrics_df
    }
    
    return result_dict 