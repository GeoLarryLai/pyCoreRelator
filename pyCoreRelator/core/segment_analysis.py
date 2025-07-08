"""
Segment analysis and path finding functions for core correlation.

Included Functions:
- find_all_segments: Find segments in two logs using depth boundaries.
- build_connectivity_graph: Build predecessor and successor relationships between valid segment pairs.
- identify_special_segments: Identify special types of segments: tops, bottoms, dead ends, and orphans.
- filter_dead_end_pairs: Remove dead end and orphan segment pairs from the valid set.
- compute_total_complete_paths: Compute the total number of complete paths using dynamic programming.
- find_complete_core_paths: Find complete core paths through the segment network.
- diagnose_chain_breaks: Diagnose chain breaks in the segment network.

This module provides comprehensive segment analysis and path finding functionality for 
geological core correlation workflows. It enables the decomposition of well log data into 
analyzable segments using depth boundaries, constructs connectivity graphs to model 
predecessor-successor relationships between valid segment pairs, identifies special 
segment types including starting points (tops), ending points (bottoms), dead ends, 
and isolated segments (orphans), filters out problematic segment pairs that cannot 
contribute to complete correlation paths, and employs dynamic programming algorithms 
to efficiently compute the total number of viable complete correlation paths through 
the segment network.
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


def find_all_segments(log_a, log_b, md_a, md_b, picked_depths_a=None, picked_depths_b=None, top_bottom=True, top_depth=0.0, mute_mode=False):
    """
    Find segments in two logs using depth boundaries to create consecutive and single-point segments.
    
    Converts user-picked depth values to indices in the log arrays and generates all possible
    segment combinations for DTW analysis.
    
    Args:
        log_a (array): Log data for core A
        log_b (array): Log data for core B
        md_a (array): Measured depth values corresponding to log_a
        md_b (array): Measured depth values corresponding to log_b
        picked_depths_a (list, optional): User-selected depth values for core A boundaries
        picked_depths_b (list, optional): User-selected depth values for core B boundaries
        top_bottom (bool): Whether to add top and bottom boundaries automatically
        top_depth (float): Depth value to use for top boundary
        mute_mode (bool, default=False): If True, suppress all print output
        
    Returns:
        tuple: (segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, depth_values_a, depth_values_b)
            - segments_a/b: List of (start_idx, end_idx) tuples for each segment
            - depth_boundaries_a/b: List of indices corresponding to depth values
            - depth_values_a/b: List of actual depth values used
    
    Example:
        >>> segments_a, segments_b, bounds_a, bounds_b, depths_a, depths_b = find_all_segments(
        ...     log_a, log_b, md_a, md_b, 
        ...     picked_depths_a=[0, 100, 200], 
        ...     picked_depths_b=[0, 150, 300]
        ... )
        >>> print(f"Core A has {len(segments_a)} segments")
        >>> print(f"First segment A spans indices {segments_a[0]}")
    """
    
    # Initialize depth lists
    if picked_depths_a is None:
        picked_depths_a = []
    if picked_depths_b is None:
        picked_depths_b = []
    
    # Ensure picked_depths are Python lists
    if isinstance(picked_depths_a, np.ndarray):
        depth_values_a = picked_depths_a.tolist()
    else:
        depth_values_a = list(picked_depths_a)
        
    if isinstance(picked_depths_b, np.ndarray):
        depth_values_b = picked_depths_b.tolist()
    else:
        depth_values_b = list(picked_depths_b)
    
    # Add top and bottom boundaries if requested
    if top_bottom:
        if top_depth not in depth_values_a:
            depth_values_a.append(top_depth)
        if md_a[-1] not in depth_values_a:
            depth_values_a.append(md_a[-1])
            
        if top_depth not in depth_values_b:
            depth_values_b.append(top_depth)
        if md_b[-1] not in depth_values_b:
            depth_values_b.append(md_b[-1])
    
    # Sort and remove duplicates
    depth_values_a = sorted(list(set(depth_values_a)))
    depth_values_b = sorted(list(set(depth_values_b)))
    
    # Create default segments if no depths specified
    if len(depth_values_a) == 0:
        if not mute_mode:
            print("Warning: No depth boundaries specified for log A. Using evenly spaced boundaries.")
        depth_values_a = [top_depth, md_a[len(log_a) // 3], md_a[2 * len(log_a) // 3], md_a[-1]]
    
    if len(depth_values_b) == 0:
        if not mute_mode:
            print("Warning: No depth boundaries specified for log B. Using evenly spaced boundaries.")
        depth_values_b = [top_depth, md_b[len(log_b) // 3], md_b[2 * len(log_b) // 3], md_b[-1]]
    
    def find_nearest_index(depth_array, depth_value):
        """Find the index in depth_array closest to the given depth_value."""
        return np.abs(np.array(depth_array) - depth_value).argmin()
    
    # Convert depth values to array indices
    depth_boundaries_a = [find_nearest_index(md_a, depth) for depth in depth_values_a]
    depth_boundaries_b = [find_nearest_index(md_b, depth) for depth in depth_values_b]
    
    # Generate consecutive and single-point segments
    segments_a = []
    for i in range(len(depth_boundaries_a)):
        segments_a.append((i, i))  # Single point segment
        if i < len(depth_boundaries_a) - 1:
            segments_a.append((i, i+1))  # Consecutive segment
    
    segments_b = []
    for i in range(len(depth_boundaries_b)):
        segments_b.append((i, i))  # Single point segment
        if i < len(depth_boundaries_b) - 1:
            segments_b.append((i, i+1))  # Consecutive segment

    # Print summary information
    if not mute_mode:
        print(f"\nLog A depth values: {[float(d) for d in depth_values_a]}")
        print(f"Log A depth boundaries: {[int(i) for i in depth_boundaries_a]}")
        print(f"\nLog B depth values: {[float(d) for d in depth_values_b]}")
        print(f"Log B depth boundaries: {[int(i) for i in depth_boundaries_b]}")
        print(f"Generated {len(segments_a)} possible segments for log A")
        print(f"Generated {len(segments_b)} possible segments for log B")
    
    return segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, depth_values_a, depth_values_b


def build_connectivity_graph(valid_dtw_pairs, detailed_pairs):
    """
    Build predecessor and successor relationships between valid segment pairs.
    
    Two segments are connected if the end depth of one segment matches the start depth
    of another segment for both cores A and B.
    
    Parameters:
        valid_dtw_pairs (set): Valid segment pairs from DTW analysis
        detailed_pairs (dict): Dictionary mapping segment pairs to their depth details
        
    Returns:
        tuple: (successors, predecessors) dictionaries mapping segments to connected segments
    
    Example:
        >>> successors, predecessors = build_connectivity_graph(valid_pairs, details)
        >>> # Check what follows segment (1,2)
        >>> next_segments = successors.get((1,2), [])
        >>> print(f"Segment (1,2) connects to: {next_segments}")
    """
    
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    
    # Build connectivity by comparing end/start depths
    for a_idx, b_idx in valid_dtw_pairs:
        pair_details = detailed_pairs[(a_idx, b_idx)]
        a_end = pair_details['a_end']
        b_end = pair_details['b_end']
        
        for next_a_idx, next_b_idx in valid_dtw_pairs:
            if (a_idx, b_idx) != (next_a_idx, next_b_idx):
                next_details = detailed_pairs[(next_a_idx, next_b_idx)]
                next_a_start = next_details['a_start']
                next_b_start = next_details['b_start']
                
                # Check if segments connect exactly
                if (abs(next_a_start - a_end) < 1e-6 and 
                    abs(next_b_start - b_end) < 1e-6):
                    successors[(a_idx, b_idx)].append((next_a_idx, next_b_idx))
                    predecessors[(next_a_idx, next_b_idx)].append((a_idx, b_idx))
    
    return dict(successors), dict(predecessors)

def identify_special_segments(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b):
    """
    Identify special types of segments: tops, bottoms, dead ends, and orphans.
    
    - Top segments: Start at depth 0 for both cores
    - Bottom segments: End at maximum depth for both cores  
    - Dead ends: Have no successors (but aren't bottom segments)
    - Orphans: Have no predecessors (but aren't top segments)
    
    Parameters:
        valid_dtw_pairs (set): Valid segment pairs
        detailed_pairs (dict): Segment depth details
        max_depth_a (float): Maximum depth for core A
        max_depth_b (float): Maximum depth for core B
        
    Returns:
        tuple: (top_segments, bottom_segments, dead_ends, orphans, successors, predecessors)
    
    Example:
        >>> tops, bottoms, dead, orphans, succ, pred = identify_special_segments(
        ...     valid_pairs, details, 1000.0, 1200.0
        ... )
        >>> print(f"Found {len(tops)} top segments and {len(bottoms)} bottom segments")
        >>> print(f"Warning: {len(dead)} dead ends and {len(orphans)} orphans detected")
    """
    
    # Build connectivity graph first
    successors, predecessors = build_connectivity_graph(valid_dtw_pairs, detailed_pairs)
    
    # Classify segments based on position and connectivity
    top_segments = []
    bottom_segments = []
    dead_ends = []
    orphans = []
    
    for a_idx, b_idx in valid_dtw_pairs:
        details = detailed_pairs[(a_idx, b_idx)]
        
        # Top segments start at depth 0 for both cores
        if abs(details['a_start']) < 1e-6 and abs(details['b_start']) < 1e-6:
            top_segments.append((a_idx, b_idx))
        
        # Bottom segments end at maximum depth for both cores
        if (abs(details['a_end'] - max_depth_a) < 1e-6 and 
            abs(details['b_end'] - max_depth_b) < 1e-6):
            bottom_segments.append((a_idx, b_idx))
        
        # Dead ends have no successors but aren't bottom segments
        if len(successors.get((a_idx, b_idx), [])) == 0 and (a_idx, b_idx) not in bottom_segments:
            dead_ends.append((a_idx, b_idx))
        
        # Orphans have no predecessors but aren't top segments
        if len(predecessors.get((a_idx, b_idx), [])) == 0 and (a_idx, b_idx) not in top_segments:
            orphans.append((a_idx, b_idx))
    
    return top_segments, bottom_segments, dead_ends, orphans, successors, predecessors


def filter_dead_end_pairs(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, debug=False):
    """
    Remove dead end and orphan segment pairs from the valid set.
    
    This filtering improves path finding by removing segments that cannot be part
    of complete paths from top to bottom.
    
    Parameters:
        valid_dtw_pairs (set): Valid segment pairs to filter
        detailed_pairs (dict): Segment depth details  
        max_depth_a (float): Maximum depth for core A
        max_depth_b (float): Maximum depth for core B
        debug (bool): Whether to print filtering statistics
        
    Returns:
        set: Filtered segment pairs without dead ends and orphans
    
    Example:
        >>> filtered_pairs = filter_dead_end_pairs(valid_pairs, details, 1000, 1200, debug=True)
        >>> print(f"Retained {len(filtered_pairs)} viable segments")
    """
    
    # Get special segment classifications
    top_segments, bottom_segments, dead_ends, orphans, successors, predecessors = identify_special_segments(
        valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b
    )
    
    # Combine problematic segments
    dead_end_pairs = set(dead_ends + orphans)
    
    # Filter out problematic segments
    filtered_pairs = set(valid_dtw_pairs) - dead_end_pairs
    
    if debug:
        print(f"Dead end filtering: {len(filtered_pairs)}/{len(valid_dtw_pairs)} segments retained")
        print(f"Removed {len(dead_end_pairs)} dead-end pairs:")
        print(f"  - {len(dead_ends)} dead ends (no successors)")
        print(f"  - {len(orphans)} orphans (no predecessors)")
    
    return filtered_pairs


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
    valid_dtw_pairs,
    segments_a,
    segments_b,
    log_a,
    log_b,
    depth_boundaries_a,
    depth_boundaries_b,
    dtw_results,  
    output_csv="complete_core_paths.csv",
    debug=False,
    start_from_top_only=True,
    batch_size=1000,
    n_jobs=-1,
    shortest_path_search=True,
    shortest_path_level=2,
    max_search_path=5000,
    output_metric_only=False,  # Add this new parameter
    mute_mode=False  # Add this new parameter
):
    """
    Find and enumerate all complete core-to-core correlation paths with advanced optimization features.
    
    Searches for paths that span from the top to bottom of both cores through connected segments.
    Includes memory management, duplicate removal, and performance optimizations for large datasets.
    
    Parameters:
        valid_dtw_pairs (set): Valid segment pairs from DTW analysis
        segments_a, segments_b (list): Segment definitions for both cores
        log_a, log_b (array): Core log data for metric computation
        depth_boundaries_a, depth_boundaries_b (list): Depth boundary indices
        dtw_results (dict): DTW results for quality metrics
        output_csv (str): Output CSV filename
        debug (bool): Enable detailed progress reporting
        start_from_top_only (bool): Only start paths from top segments
        batch_size (int): Processing batch size
        n_jobs (int): Number of parallel jobs (-1 for all cores)
        shortest_path_search (bool): Keep only shortest path lengths during search
        shortest_path_level (int): Number of shortest unique lengths to keep
        max_search_path (int): Maximum complete paths to find before stopping
        output_metric_only (bool): Only output quality metrics in the output CSV, no paths info
        mute_mode (bool): If True, suppress all print output
        
    Returns:
        dict: Comprehensive results including:
            - total_complete_paths_theoretical: Theoretical path count
            - total_complete_paths_found: Actually enumerated paths
            - viable_segments: Set of viable segments
            - output_csv: Path to generated CSV file
            - duplicates_removed: Number of duplicates removed
            - search_limit_reached: Whether search limit was hit
    
    Example:
        >>> results = find_complete_core_paths(
        ...     valid_pairs, segs_a, segs_b, log_a, log_b, 
        ...     bounds_a, bounds_b, dtw_results,
        ...     max_search_path=10000, debug=True
        ... )
        >>> print(f"Found {results['total_complete_paths_found']} complete paths")
        >>> print(f"Results saved to: {results['output_csv']}")
    """

    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    output_csv_filename = os.path.basename(output_csv)
    output_csv = os.path.join('outputs', output_csv_filename)

    # Performance warning for unlimited search
    if max_search_path is None and not mute_mode:
        print("⚠️  WARNING: max_search_path=None can be very time consuming and require high memory usage!")
        print("   Consider setting max_search_path to a reasonable limit (e.g., 50000) for better performance.")

    def check_memory(threshold_percent=85):
        """Check if memory usage is high and force cleanup if needed."""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > threshold_percent:
            if not mute_mode:
                print(f"⚠️ Memory usage high ({memory_percent}%)! Forcing cleanup...")
            gc.collect()
            return True
        return False

    def calculate_diagonality(wp):
        """Calculate how diagonal/linear the DTW path is (0-1, higher is better)."""
        if len(wp) < 2:
            return 1.0
            
        # Measure deviation from perfect diagonal
        a_indices = wp[:, 0]
        b_indices = wp[:, 1]
        
        a_range = np.max(a_indices) - np.min(a_indices)
        b_range = np.max(b_indices) - np.min(b_indices)
        
        if a_range == 0 or b_range == 0:
            return 0.0  # Perfectly horizontal or vertical
        
        # Normalize and calculate distance from diagonal
        a_norm = (a_indices - np.min(a_indices)) / a_range
        b_norm = (b_indices - np.min(b_indices)) / b_range
        distances = np.abs(a_norm - b_norm)
        avg_distance = np.mean(distances)
        
        return float(1.0 - avg_distance)
    
    # Path compression for memory efficiency
    def compress_path(path_segment_pairs):
        """Compress path to save memory: [(1,2), (2,4)] -> "1,2|2,4" """
        if not path_segment_pairs:
            return ""
        return "|".join(f"{a},{b}" for a, b in path_segment_pairs)
    
    def decompress_path(compressed_path):
        """Decompress path: "1,2|2,4" -> [(1,2), (2,4)]"""
        if not compressed_path:
            return []
        return [tuple(map(int, segment.split(','))) for segment in compressed_path.split('|')]
    
    def remove_duplicates_from_db(conn, debug_info=""):
        """Remove duplicate paths from database and return count of removed duplicates."""
        if debug and not mute_mode:
            print(f"Removing duplicates from database... {debug_info}")
        
        # Create temporary table for unique paths
        conn.execute("""
            CREATE TEMPORARY TABLE temp_unique_paths AS
            SELECT MIN(rowid) as keep_rowid, compressed_path, COUNT(*) as duplicate_count
            FROM compressed_paths 
            GROUP BY compressed_path
        """)
        
        # Count duplicates
        cursor = conn.execute("""
            SELECT SUM(duplicate_count - 1) FROM temp_unique_paths 
            WHERE duplicate_count > 1
        """)
        total_duplicates = cursor.fetchone()[0] or 0
        
        if total_duplicates > 0:
            # Delete duplicates, keep only first occurrence
            conn.execute("""
                DELETE FROM compressed_paths 
                WHERE rowid NOT IN (SELECT keep_rowid FROM temp_unique_paths)
            """)
            
            if debug and not mute_mode:
                print(f"  Removed {total_duplicates} duplicate paths")
        
        conn.execute("DROP TABLE temp_unique_paths")
        conn.commit()
        
        return total_duplicates
    
    def filter_shortest_paths(paths_data, shortest_path_level):
        """Filter paths to keep only the shortest path lengths."""
        if not paths_data:
            return paths_data
        
        # Get unique lengths and keep shortest ones
        lengths = [length for _, length, _ in paths_data]
        unique_lengths = sorted(set(lengths))
        keep_lengths = set(unique_lengths[:shortest_path_level])
        
        # Filter paths
        filtered_paths = [(path, length, is_complete) for path, length, is_complete in paths_data 
                         if length in keep_lengths]
        
        if debug and not mute_mode and len(filtered_paths) < len(paths_data):
            print(f"  Shortest path filtering: kept {len(filtered_paths)}/{len(paths_data)} paths with lengths {sorted(keep_lengths)}")
        
        return filtered_paths
    
    def compute_path_metrics_lazy(compressed_path, log_a, log_b):
        """Compute quality metrics lazily only when needed for final output."""
        path_segment_pairs = decompress_path(compressed_path)
        
        # Collect DTW results for path segments
        all_quality_indicators = []
        age_overlap_values = []
        all_wps = []
        
        for a_idx, b_idx in path_segment_pairs:
            if (a_idx, b_idx) in dtw_results:
                paths, _, quality_indicators = dtw_results[(a_idx, b_idx)]
                
                if not paths or len(paths) == 0:
                    continue
                    
                all_wps.append(paths[0])
                
                if quality_indicators and len(quality_indicators) > 0:
                    qi = quality_indicators[0]
                    all_quality_indicators.append(qi)
                    
                    if 'perc_age_overlap' in qi:
                        age_overlap_values.append(float(qi['perc_age_overlap']))
        
        # Combine warping paths
        if all_wps:
            combined_wp = np.vstack(all_wps)
            combined_wp = np.unique(combined_wp, axis=0)
            combined_wp = combined_wp[combined_wp[:, 0].argsort()]
        else:
            combined_wp = np.array([])
        
        # Compute combined metrics
        from ..utils.path_processing import compute_combined_path_metrics
        metrics = compute_combined_path_metrics(combined_wp, log_a, log_b, all_quality_indicators, age_overlap_values)
        
        return combined_wp, metrics

    # Database setup and operations
    def setup_database(db_path, read_only=False):
        """Setup SQLite database with performance optimizations."""
        conn = sqlite3.connect(db_path)
        
        # SQLite performance optimizations
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        if not read_only:
            # Create tables and indexes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compressed_paths (
                    path_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_segment TEXT NOT NULL,
                    last_segment TEXT NOT NULL,
                    compressed_path TEXT NOT NULL,
                    length INTEGER NOT NULL,
                    is_complete BOOLEAN DEFAULT 0
                )
            """)
            
            # Performance indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_segment ON compressed_paths(last_segment)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_start_segment ON compressed_paths(start_segment)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_is_complete ON compressed_paths(is_complete)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_compressed_path ON compressed_paths(compressed_path)")
        
        conn.commit()
        return conn

    def insert_compressed_path(conn, start_segment, last_segment, compressed_path, length, is_complete=False):
        """Insert a compressed path into database."""
        conn.execute("""
            INSERT INTO compressed_paths (start_segment, last_segment, compressed_path, length, is_complete)
            VALUES (?, ?, ?, ?, ?)
        """, (f"{start_segment[0]},{start_segment[1]}", 
              f"{last_segment[0]},{last_segment[1]}", 
              compressed_path, length, is_complete))
        
    def prune_shared_database_if_needed(shared_conn, max_paths, debug=False):
        """Prune intermediate paths in shared database when they exceed the maximum limit."""
        if max_paths is None:
            return 0
        
        # Count only intermediate paths (is_complete = 0)
        cursor = shared_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 0")
        current_intermediate_count = cursor.fetchone()[0]
        
        if current_intermediate_count <= max_paths:
            return 0  # No pruning needed
        
        paths_to_remove = current_intermediate_count - max_paths
        
        if debug and not mute_mode:
            print(f"  Shared DB pruning: {current_intermediate_count} intermediate paths exceed limit of {max_paths}")
            print(f"  Randomly excluding {paths_to_remove} intermediate paths from shared database")
        
        # Get only intermediate paths with their rowids for random selection
        cursor = shared_conn.execute("""
            SELECT rowid, start_segment 
            FROM compressed_paths
            WHERE is_complete = 0
        """)
        intermediate_paths = cursor.fetchall()
        
        # Randomly select intermediate paths to remove
        if len(intermediate_paths) > max_paths:
            selected_for_removal = random.sample(intermediate_paths, paths_to_remove)
            rowids_to_remove = [rowid for rowid, _ in selected_for_removal]
            
            # Remove selected intermediate paths from database
            if rowids_to_remove:
                placeholders = ",".join("?" * len(rowids_to_remove))
                shared_conn.execute(f"""
                    DELETE FROM compressed_paths 
                    WHERE rowid IN ({placeholders})
                """, rowids_to_remove)
                
                shared_conn.commit()
                
                if debug and not mute_mode:
                    print(f"  Removed {len(rowids_to_remove)} intermediate paths")
                    # Verify final count
                    cursor = shared_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 0")
                    final_count = cursor.fetchone()[0]
                    print(f"  Final intermediate path count after pruning: {final_count}")
            
            return len(rowids_to_remove)
        
        return 0

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
                new_paths_data = filter_shortest_paths(new_paths_data, shortest_path_level)
            
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
        duplicates_removed = remove_duplicates_from_db(group_write_conn, f"Group {group_idx+1}")
        
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
                shared_pruned = prune_shared_database_if_needed(shared_read_conn, max_search_path, debug)
                if shared_pruned > 0 and debug and not mute_mode:
                    print(f"  Pruned {shared_pruned} paths from shared database after sync")

            # Remove duplicates after sync
            if len(group_results) > 1:
                shared_duplicates = remove_duplicates_from_db(shared_read_conn, f"Shared DB after sync")
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
    final_duplicates = remove_duplicates_from_db(shared_read_conn, "Final cleanup")
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
    
    # Create output CSV with batch processing for memory efficiency
    def generate_output_csv():
        """Generate final CSV output directly from deduplicated database using parallel processing."""
        
        # Create output CSV with header
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            if output_metric_only:
                writer.writerow(['mapping_id', 'length', 
                            'norm_dtw', 'dtw_ratio', 'perc_diag', 'dtw_warp_eff', 'corr_coef', 'perc_age_overlap'])
            else:
                writer.writerow(['mapping_id', 'path', 'length', 'combined_wp', 
                            'norm_dtw', 'dtw_ratio', 'perc_diag', 'dtw_warp_eff', 'corr_coef', 'perc_age_overlap'])
        
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
        all_paths = cursor.fetchall()
        
        # Use the function's parameters for batch size and number of jobs
        # If n_jobs is -1, use all cores; otherwise use the specified number
        n_jobs_to_use = os.cpu_count() if n_jobs == -1 else n_jobs
        # Use batch_size parameter directly, with a reasonable fallback
        batch_size_to_use = batch_size if batch_size > 0 else 500
        # Create batches of paths
        batches = [all_paths[i:i + batch_size] for i in range(0, len(all_paths), batch_size)]
        
        # Process a batch of paths
        def process_batch(batch, start_id):
            batch_results = []
            mapping_id = start_id
            
            for compressed_path, length, _ in batch:
                # Decompress and format path
                full_path = decompress_path(compressed_path)
                
                # Convert to 1-based and use semicolon separator for compactness
                formatted_path_compact = ";".join(f"{a+1},{b+1}" for a, b in full_path)
                
                # Compute metrics and warping path
                combined_wp, metrics = compute_path_metrics_lazy(compressed_path, log_a, log_b)
                
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
                        round(metrics['corr_coef'], 6),
                        round(metrics['perc_age_overlap'], 2)
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
                        round(metrics['perc_age_overlap'], 2)
                    ])
                
                mapping_id += 1
                
            return batch_results
        
        if not mute_mode:
            print(f"Processing {total_paths} paths in {len(batches)} batches")
        
        # Process batches in parallel
        if not mute_mode:
            pbar = tqdm(total=len(batches), desc="Processing batches")
        
        for batch_idx, batch in enumerate(batches):
            # Calculate starting ID for this batch
            start_id = batch_idx * batch_size + 1
            
            # Process this batch
            batch_results = process_batch(batch, start_id)
            
            # Write batch results
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(batch_results)
            
            if not mute_mode:
                pbar.update(1)
                
            # Periodic garbage collection
            if batch_idx % 5 == 0:
                gc.collect()
        
        if not mute_mode:
            pbar.close()
        
        return total_paths
    
    # Generate the output
    total_paths_written = generate_output_csv()
    
    # Close shared database
    shared_read_conn.close()
    
    # Print final statistics
    if not mute_mode:
        print(f"\nFinal Results:")
        print(f"  Total unique complete paths written: {total_paths_written}")
        print(f"  Total duplicates removed during processing: {total_duplicates_removed}")
        print(f"  Deduplication efficiency: {(total_duplicates_removed/(total_paths_written + total_duplicates_removed)*100) if (total_paths_written + total_duplicates_removed) > 0 else 0:.2f}%")
        
        # Add search limit information to final results
        if search_limit_reached:
            print(f"  Search was limited to {max_search_path} complete paths for performance")
    
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
    
    if not mute_mode:
        print(f"All complete core-to-core paths saved to {output_csv}")
    
    # Return comprehensive results dictionary
    return {
        'total_complete_paths_theoretical': path_computation_results['total_complete_paths'],
        'total_complete_paths_found': total_paths_written,
        'viable_segments': path_computation_results['viable_segments'],
        'viable_tops': path_computation_results['viable_tops'],
        'viable_bottoms': path_computation_results['viable_bottoms'],
        'paths_from_tops': path_computation_results['paths_from_tops'],
        'output_csv': output_csv,
        'duplicates_removed': total_duplicates_removed,
        'search_limit_reached': search_limit_reached
    }


def diagnose_chain_breaks(valid_dtw_pairs, segments_a, segments_b, 
                          depth_boundaries_a, depth_boundaries_b):
    """
    Comprehensive diagnostic to find exactly where segment chains break.
    
    This function will trace all possible paths and identify missing connections.
    Additionally computes total complete paths and finds the "far most" bounding complete paths.
    
    Parameters:
    -----------
    valid_dtw_pairs : set or list
        Valid segment pairs
    segments_a, segments_b : list
        Segments in log_a and log_b
    depth_boundaries_a, depth_boundaries_b : list
        Depth boundaries for log_a and log_b
    
    Returns:
    --------
    dict: Enhanced results including complete path counts and bounding paths
    """
    
    print("=== CHAIN BREAK DIAGNOSTIC ===")
    
    # Get max depths
    max_depth_a = max(depth_boundaries_a)
    max_depth_b = max(depth_boundaries_b)
    
    # Create detailed segment info
    detailed_pairs = {}
    for a_idx, b_idx in valid_dtw_pairs:
        a_start = depth_boundaries_a[segments_a[a_idx][0]]
        a_end = depth_boundaries_a[segments_a[a_idx][1]]
        b_start = depth_boundaries_b[segments_b[b_idx][0]]
        b_end = depth_boundaries_b[segments_b[b_idx][1]]
        
        detailed_pairs[(a_idx, b_idx)] = {
            'a_start': a_start, 'a_end': a_end,
            'b_start': b_start, 'b_end': b_end,
            'a_len': a_end - a_start + 1,
            'b_len': b_end - b_start + 1
        }
    
    # Use the standalone helper functions
    top_segments, bottom_segments, dead_ends, orphans, successors, predecessors = identify_special_segments(
        valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b
    )
    
    print(f"\n=== SEGMENT INVENTORY & CONNECTIVITY ===")
    print(f"Core A max depth: {max_depth_a}, Core B max depth: {max_depth_b}")
    print(f"Total valid segment pairs: {len(valid_dtw_pairs)}")
    print(f"Top segments (start at 0,0): {[(a+1, b+1) for a, b in top_segments]}")
    print(f"Bottom segments (end at max): {[(a+1, b+1) for a, b in bottom_segments]}")
    print(f"Dead ends (no successors): {len(dead_ends)} - {[(a+1, b+1) for a, b in dead_ends[:5]]}{'...' if len(dead_ends) > 5 else ''}")
    print(f"Orphans (no predecessors): {len(orphans)} - {[(a+1, b+1) for a, b in orphans[:5]]}{'...' if len(orphans) > 5 else ''}")
    
    if not top_segments:
        print("❌ FATAL: No top segments found!")
        return None
        
    if not bottom_segments:
        print("❌ FATAL: No bottom segments found!")
        return None
    
    # Trace reachability from each top segment
    print(f"\n=== DETAILED SEGMENT ANALYSIS ===")
    
    # Print all segments with their connectivity and types
    for i, (a_idx, b_idx) in enumerate(sorted(valid_dtw_pairs)):
        details = detailed_pairs[(a_idx, b_idx)]
        pred_count = len(predecessors.get((a_idx, b_idx), []))
        succ_count = len(successors.get((a_idx, b_idx), []))
        
        # Determine segment type
        segment_types = []
        if (a_idx, b_idx) in top_segments:
            segment_types.append("TOP")
        if (a_idx, b_idx) in bottom_segments:
            segment_types.append("BOTTOM")
        if (a_idx, b_idx) in dead_ends:
            segment_types.append("DEAD_END")
        if (a_idx, b_idx) in orphans:
            segment_types.append("ORPHAN")
        if not segment_types:
            segment_types.append("MIDDLE")
        
        print(f"{i+1:3d}. Segment ({a_idx+1:2d},{b_idx+1:2d}): "
              f"A[{details['a_start']:6.1f}:{details['a_end']:6.1f}] "
              f"B[{details['b_start']:6.1f}:{details['b_end']:6.1f}] "
              f"(A_len={details['a_len']:3.0f}, B_len={details['b_len']:3.0f}) "
              f"pred:{pred_count} succ:{succ_count} {'/'.join(segment_types)}")
    
    print(f"\n=== REACHABILITY ANALYSIS ===")
    
    def trace_reachable_segments(start_segment):
        """Trace all segments reachable from a starting segment"""
        visited = set()
        queue = deque([(start_segment, 0, [start_segment])])  # (segment, depth, path)
        all_paths = []
        max_depth_reached = 0
        
        while queue:
            current, depth, path = queue.popleft()
            
            if current in visited:
                continue
                
            visited.add(current)
            max_depth_reached = max(max_depth_reached, depth)
            
            # Check if this is a bottom segment
            if current in bottom_segments:
                all_paths.append(path)
                continue
            
            # Add successors to queue
            for successor in successors.get(current, []):
                if successor not in visited:
                    new_path = path + [successor]
                    queue.append((successor, depth + 1, new_path))
        
        return visited, all_paths, max_depth_reached
    
    # Analyze each top segment
    all_complete_paths = []
    
    for i, top_seg in enumerate(top_segments):
        print(f"\nTop Segment ({top_seg[0]+1},{top_seg[1]+1}):")
        
        reachable, complete_paths, max_depth = trace_reachable_segments(top_seg)
        
        print(f"  Reachable segments: {len(reachable)}, Complete paths: {len(complete_paths)}, Max chain depth: {max_depth}")
        
        if len(complete_paths) == 0:
            print(f"  ❌ NO COMPLETE PATHS - Chain breaks detected")
            
            # Find the deepest reachable segments
            deepest_segments = []
            for seg in reachable:
                seg_details = detailed_pairs[seg]
                deepest_segments.append((seg, seg_details['a_end'], seg_details['b_end']))
            
            # Sort by depth and show the deepest reachable segments
            deepest_segments.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            print(f"  🔍 Deepest reachable segments:")
            for j, (seg, a_depth, b_depth) in enumerate(deepest_segments[:3]):
                print(f"    {j+1}. ({seg[0]+1},{seg[1]+1}): A ends at {a_depth}, B ends at {b_depth}")
                
                # Check what's missing to continue
                missing_connections = []
                for next_seg in valid_dtw_pairs:
                    next_details = detailed_pairs[next_seg]
                    if (abs(next_details['a_start'] - a_depth) < 1e-6 and 
                        abs(next_details['b_start'] - b_depth) < 1e-6):
                        if next_seg not in reachable:
                            missing_connections.append(next_seg)
                
                if missing_connections:
                    print(f"       💡 Could connect to: {[(a+1,b+1) for a,b in missing_connections]} (but not reachable)")
                else:
                    print(f"       ⛔ No valid next segments available")
        else:
            print(f"  ✅ Complete paths exist")
            all_complete_paths.extend(complete_paths)
            # Show first complete path as example
            if complete_paths:
                example_path = complete_paths[0]
                path_str = " → ".join([f"({seg[0]+1},{seg[1]+1})" for seg in example_path])
                print(f"  Example path: {path_str}")

    
    # ===== COMPLETE PATH ANALYSIS =====
    # Execute new functionality using the standalone functions
    total_paths_results = compute_total_complete_paths(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b)
    
    return {
        'top_segments': top_segments,
        'bottom_segments': bottom_segments,
        'complete_paths': all_complete_paths,
        'successors': successors,
        'predecessors': predecessors,
        'dead_ends': dead_ends,
        'orphans': orphans,
        'detailed_pairs': detailed_pairs,
        # New additions from standalone functions
        'total_complete_paths': total_paths_results['total_complete_paths'],
        'viable_segments': total_paths_results['viable_segments'],
        'viable_tops': total_paths_results['viable_tops'],
        'viable_bottoms': total_paths_results['viable_bottoms'],
        'paths_from_tops': total_paths_results['paths_from_tops']
    }