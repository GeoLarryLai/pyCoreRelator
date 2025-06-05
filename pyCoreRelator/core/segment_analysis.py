"""
Segment analysis and path finding functions
"""

import numpy as np
import pandas as pd
import os
import csv
import tempfile
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


def find_all_segments(log_a, log_b, md_a, md_b, picked_depths_a=None, picked_depths_b=None, top_bottom=True, top_depth=0.0):
    """
    Find segments in two logs, including only consecutive boundaries and single point segments.
    Uses picked depths to find the nearest corresponding indices in the logs.
    
    Args:
        log_a, log_b: Log data arrays
        md_a, md_b: Measured depth arrays corresponding to log_a and log_b
        picked_depths_a, picked_depths_b: User-picked depth values (not indices)
        top_bottom: If True, add top and bottom depth values to depth boundaries. If False, use only picked depths.
        top_depth: The depth value to use for the top boundary (default is 0.0)
        
    Returns:
        segments_a, segments_b, depth_boundaries_a, depth_boundaries_b
    """
    
    # Create depth boundaries from picked depths, add top and bottom if specified
    if picked_depths_a is None:
        picked_depths_a = []
    
    if picked_depths_b is None:
        picked_depths_b = []
    
    # Ensure picked_depths are Python lists, not NumPy arrays
    if isinstance(picked_depths_a, np.ndarray):
        depth_values_a = picked_depths_a.tolist()
    else:
        depth_values_a = list(picked_depths_a)
        
    if isinstance(picked_depths_b, np.ndarray):
        depth_values_b = picked_depths_b.tolist()
    else:
        depth_values_b = list(picked_depths_b)
    
    if top_bottom:
        # Add top and bottom actual depth values
        if top_depth not in depth_values_a:
            depth_values_a.append(top_depth)  # Top depth value
        if md_a[-1] not in depth_values_a:
            depth_values_a.append(md_a[-1])  # Bottom depth value
            
        if top_depth not in depth_values_b:
            depth_values_b.append(top_depth)  # Top depth value
        if md_b[-1] not in depth_values_b:
            depth_values_b.append(md_b[-1])  # Bottom depth value
    
    # Sort and remove duplicates
    depth_values_a = sorted(list(set(depth_values_a)))
    depth_values_b = sorted(list(set(depth_values_b)))
    
    # If no depths specified, create default segments
    if len(depth_values_a) == 0:
        # If no picked depths, create at least a few segments
        print("Warning: No depth boundaries specified for log A. Using evenly spaced boundaries.")
        depth_values_a = [top_depth, md_a[len(log_a) // 3], md_a[2 * len(log_a) // 3], md_a[-1]]
    
    if len(depth_values_b) == 0:
        # If no picked depths, create at least a few segments
        print("Warning: No depth boundaries specified for log B. Using evenly spaced boundaries.")
        depth_values_b = [top_depth, md_b[len(log_b) // 3], md_b[2 * len(log_b) // 3], md_b[-1]]
    
    # Helper function to find nearest index
    def find_nearest_index(depth_array, depth_value):
        """Find the index in depth_array that has the closest depth value to the given depth_value."""
        return np.abs(np.array(depth_array) - depth_value).argmin()
    
    # Convert depth values to indices using find_nearest_index
    depth_boundaries_a = [find_nearest_index(md_a, depth) for depth in depth_values_a]
    depth_boundaries_b = [find_nearest_index(md_b, depth) for depth in depth_values_b]
    
    # Generate only consecutive boundary segments and single point segments for log A
    segments_a = []
    for i in range(len(depth_boundaries_a)):
        # Add single point segments (same boundary)
        segments_a.append((i, i))
        # Add consecutive boundary segments
        if i < len(depth_boundaries_a) - 1:
            segments_a.append((i, i+1))
    
    # Generate only consecutive boundary segments and single point segments for log B
    segments_b = []
    for i in range(len(depth_boundaries_b)):
        # Add single point segments (same boundary)
        segments_b.append((i, i))
        # Add consecutive boundary segments
        if i < len(depth_boundaries_b) - 1:
            segments_b.append((i, i+1))

    # Always print depth values regardless of top_bottom setting
    print(f"\nLog A depth values: {[float(d) for d in depth_values_a]}")
    print(f"Log A depth boundaries: {[int(i) for i in depth_boundaries_a]}")
    
    print(f"\nLog B depth values: {[float(d) for d in depth_values_b]}")
    print(f"Log B depth boundaries: {[int(i) for i in depth_boundaries_b]}")
    
    print(f"Generated {len(segments_a)} possible segments for log A")
    print(f"Generated {len(segments_b)} possible segments for log B")
    
    return segments_a, segments_b, depth_boundaries_a, depth_boundaries_b, depth_values_a, depth_values_b


def build_connectivity_graph(valid_dtw_pairs, detailed_pairs):
    """
    Build predecessor and successor graphs for segments.
    
    Parameters:
    -----------
    valid_dtw_pairs : set or list
        Valid segment pairs
    detailed_pairs : dict
        Dictionary containing segment details with start/end depths
    
    Returns:
    --------
    tuple : (successors, predecessors)
        Dictionaries mapping segments to their successors and predecessors
    """
    
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    
    for a_idx, b_idx in valid_dtw_pairs:
        pair_details = detailed_pairs[(a_idx, b_idx)]
        a_end = pair_details['a_end']
        b_end = pair_details['b_end']
        
        for next_a_idx, next_b_idx in valid_dtw_pairs:
            if (a_idx, b_idx) != (next_a_idx, next_b_idx):
                next_details = detailed_pairs[(next_a_idx, next_b_idx)]
                next_a_start = next_details['a_start']
                next_b_start = next_details['b_start']
                
                # Check if segments connect (end of current = start of next)
                if (abs(next_a_start - a_end) < 1e-6 and 
                    abs(next_b_start - b_end) < 1e-6):
                    successors[(a_idx, b_idx)].append((next_a_idx, next_b_idx))
                    predecessors[(next_a_idx, next_b_idx)].append((a_idx, b_idx))
    
    return dict(successors), dict(predecessors)


def identify_special_segments(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b):
    """
    Identify top segments, bottom segments, dead ends, and orphans.
    
    Parameters:
    -----------
    valid_dtw_pairs : set or list
        Valid segment pairs
    detailed_pairs : dict
        Dictionary containing segment details
    max_depth_a, max_depth_b : float
        Maximum depths for cores A and B
    
    Returns:
    --------
    tuple : (top_segments, bottom_segments, dead_ends, orphans, successors, predecessors)
    """
    
    # Build connectivity graph
    successors, predecessors = build_connectivity_graph(valid_dtw_pairs, detailed_pairs)
    
    # Find top and bottom segments
    top_segments = []
    bottom_segments = []
    dead_ends = []
    orphans = []
    
    for a_idx, b_idx in valid_dtw_pairs:
        details = detailed_pairs[(a_idx, b_idx)]
        
        # Top segments (both start at 0)
        if abs(details['a_start']) < 1e-6 and abs(details['b_start']) < 1e-6:
            top_segments.append((a_idx, b_idx))
        
        # Bottom segments (both end at max depth)
        if (abs(details['a_end'] - max_depth_a) < 1e-6 and 
            abs(details['b_end'] - max_depth_b) < 1e-6):
            bottom_segments.append((a_idx, b_idx))
        
        # Dead ends (no successors, not bottom segments)
        if len(successors.get((a_idx, b_idx), [])) == 0 and (a_idx, b_idx) not in bottom_segments:
            dead_ends.append((a_idx, b_idx))
        
        # Orphans (no predecessors, not top segments)
        if len(predecessors.get((a_idx, b_idx), [])) == 0 and (a_idx, b_idx) not in top_segments:
            orphans.append((a_idx, b_idx))
    
    return top_segments, bottom_segments, dead_ends, orphans, successors, predecessors


def filter_dead_end_pairs(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b, debug=False):
    """
    Filter out dead end and orphan segment pairs.
    
    Parameters:
    -----------
    valid_dtw_pairs : set or list
        Valid segment pairs to filter
    detailed_pairs : dict
        Dictionary containing segment details
    max_depth_a, max_depth_b : float
        Maximum depths for cores A and B
    debug : bool
        Whether to print debug information
    
    Returns:
    --------
    set : Filtered segment pairs without dead ends and orphans
    """
    # Identify special segments
    top_segments, bottom_segments, dead_ends, orphans, successors, predecessors = identify_special_segments(
        valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b
    )
    
    # Combine dead ends and orphans
    dead_end_pairs = set(dead_ends + orphans)
    
    # Filter out dead end pairs
    filtered_pairs = set(valid_dtw_pairs) - dead_end_pairs
    
    if debug:
        print(f"Dead end filtering: {len(filtered_pairs)}/{len(valid_dtw_pairs)} segments retained")
        print(f"Removed {len(dead_end_pairs)} dead-end pairs:")
        print(f"  - {len(dead_ends)} dead ends (no successors)")
        print(f"  - {len(orphans)} orphans (no predecessors)")
    
    return filtered_pairs


def compute_total_complete_paths(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b):
    """
    Compute total number of complete paths using dynamic programming.
    
    Parameters:
    -----------
    valid_dtw_pairs : set or list
        Valid segment pairs
    detailed_pairs : dict
        Dictionary containing segment details
    max_depth_a, max_depth_b : float
        Maximum depths for cores A and B
    
    Returns:
    --------
    dict : Dictionary containing path computation results
        - total_complete_paths: Total number of complete paths
        - viable_segments: Set of viable segments (excluding dead ends and orphans)
        - viable_tops: List of viable top segments
        - viable_bottoms: List of viable bottom segments
        - paths_from_tops: Dictionary mapping top segments to their path counts
    """
    
    # Get special segments using helper function
    top_segments, bottom_segments, dead_ends, orphans, successors, predecessors = identify_special_segments(
        valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b
    )
    
    # Filter out dead ends and orphans for viable segments
    viable_segments = set(valid_dtw_pairs) - set(dead_ends) - set(orphans)
    viable_tops = [seg for seg in top_segments if seg in viable_segments]
    viable_bottoms = [seg for seg in bottom_segments if seg in viable_segments]
    
    print(f"Viable segments (excluding dead ends and orphans): {len(viable_segments)}")
    print(f"Viable top segments: {len(viable_tops)}")
    print(f"Viable bottom segments: {len(viable_bottoms)}")
    
    if not viable_tops or not viable_bottoms:
        print("No viable complete paths possible")
        return {
            'total_complete_paths': 0,
            'viable_segments': viable_segments,
            'viable_tops': viable_tops,
            'viable_bottoms': viable_bottoms,
            'paths_from_tops': {}
        }
    
    # Use dynamic programming to count paths
    # path_count[segment] = number of paths from segment to any bottom
    path_count = {}
    
    # Initialize bottom segments
    for bottom_seg in viable_bottoms:
        if bottom_seg in viable_segments:
            path_count[bottom_seg] = 1
    
    # Process segments in reverse topological order
    def count_paths_from(segment, visited=None):
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
            print(f"  Paths from top segment ({top_seg[0]+1},{top_seg[1]+1}): {paths_from_top}")
    
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
    depth_boundaries_a,
    depth_boundaries_b,
    dtw_results,  
    output_csv="complete_core_paths.csv",
    debug=False,
    start_from_top_only=True,
    batch_size=1000,
    n_jobs=-1,
    batch_grouping=False,
    n_groups=5,
    shortest_path_search=True,
    shortest_path_level=2,
    max_search_path=5000
):
    """
    Enhanced version with complete path computation and far most bounding paths analysis integrated.
    
    Key changes:
    - Removed pre_filter_unreachable_pairs parameter and related processing
    - Returns comprehensive results dictionary instead of just CSV filename
    - Added shortest path search optimization and path limit controls
    
    Parameters:
    -----------
    valid_dtw_pairs : set or list
        Valid segment pairs
    segments_a, segments_b : list
        Segments in log_a and log_b
    depth_boundaries_a, depth_boundaries_b : list
        Depth boundaries for log_a and log_b
    dtw_results : dict
        DTW results for each segment pair
    output_csv : str, default="complete_core_paths.csv"
        Output CSV filename
    debug : bool, default=False
        Whether to print debug information
    start_from_top_only : bool, default=True
        Whether to start only from top segments
    batch_size : int, default=500
        Batch size for processing
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores)
    batch_grouping : bool, default=False
        Whether to use batch grouping for processing
    n_groups : int, default=5
        Number of groups for batch processing
    shortest_path_search : bool, default=True
        Whether to only keep shortest path lengths during search (only active when batch_grouping=False)
    shortest_path_level : int, default=2
        Number of shortest unique path lengths to keep (only active when shortest_path_search=True and batch_grouping=False)
    max_search_path : int or None, default=5000
        Maximum number of complete paths to search before stopping (only active when batch_grouping=False). 
        If None, search all paths but show warning about performance
    
    Returns:
    --------
    dict : Comprehensive results containing:
        - total_complete_paths_theoretical: Total paths computed by dynamic programming
        - total_complete_paths_found: Actual paths found and written to CSV
        - viable_segments: Set of viable segments
        - viable_tops: List of viable top segments
        - viable_bottoms: List of viable bottom segments
        - output_csv: Path to the generated CSV file
        - duplicates_removed: Number of duplicates removed during processing
    """

    # Add warning for unlimited search when batch_grouping=False
    if not batch_grouping and max_search_path is None:
        print("⚠️  WARNING: max_search_path=None with batch_grouping=False can be very time consuming and require high memory usage!")
        print("   Consider setting max_search_path to a reasonable limit (e.g., 50000) for better performance.")

    # Function to check memory usage and force cleanup if needed
    def check_memory(threshold_percent=85):
        """Check if memory usage is high and force cleanup if needed."""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > threshold_percent:
            print(f"⚠️ Memory usage high ({memory_percent}%)! Forcing cleanup...")
            gc.collect()
            return True
        return False

    # Function to calculate diagonality from warping path
    def calculate_diagonality(wp):
        """Calculate how diagonal/linear the path is (0-1, higher is better)."""
        if len(wp) < 2:
            return 1.0
            
        # Measure deviation from perfect diagonal
        a_indices = wp[:, 0]
        b_indices = wp[:, 1]
        
        a_range = np.max(a_indices) - np.min(a_indices)
        b_range = np.max(b_indices) - np.min(b_indices)
        
        if a_range == 0 or b_range == 0:
            return 0.0  # Perfectly horizontal or vertical (not diagonal)
        
        # Normalize path lengths
        a_norm = (a_indices - np.min(a_indices)) / a_range
        b_norm = (b_indices - np.min(b_indices)) / b_range
        
        # Calculate average distance from diagonal
        distances = np.abs(a_norm - b_norm)
        avg_distance = np.mean(distances)
        
        # Convert to 0-1 score (0 = far from diagonal, 1 = perfect diagonal)
        return float(1.0 - avg_distance)
    
    # Path compression functions
    def compress_path(path_segment_pairs):
        """Compress path to save memory: [(1,2), (2,4), (4,6)] -> "1,2|2,4|4,6" """
        if not path_segment_pairs:
            return ""
        return "|".join(f"{a},{b}" for a, b in path_segment_pairs)
    
    def decompress_path(compressed_path):
        """Decompress path: "1,2|2,4|4,6" -> [(1,2), (2,4), (4,6)]"""
        if not compressed_path:
            return []
        return [tuple(map(int, segment.split(','))) for segment in compressed_path.split('|')]
    
    # Duplicate detection functions    
    def remove_duplicates_from_db(conn, debug_info=""):
        """Remove duplicate paths from database and return count of removed duplicates."""
        if debug:
            print(f"Removing duplicates from database... {debug_info}")
        
        # Create a temporary table to hold unique paths
        conn.execute("""
            CREATE TEMPORARY TABLE temp_unique_paths AS
            SELECT MIN(rowid) as keep_rowid, compressed_path, COUNT(*) as duplicate_count
            FROM compressed_paths 
            GROUP BY compressed_path
        """)
        
        # Count total duplicates
        cursor = conn.execute("""
            SELECT SUM(duplicate_count - 1) FROM temp_unique_paths 
            WHERE duplicate_count > 1
        """)
        total_duplicates = cursor.fetchone()[0] or 0
        
        if total_duplicates > 0:
            # Delete duplicates - keep only the first occurrence of each unique path
            conn.execute("""
                DELETE FROM compressed_paths 
                WHERE rowid NOT IN (SELECT keep_rowid FROM temp_unique_paths)
            """)
            
            if debug:
                print(f"  Removed {total_duplicates} duplicate paths")
        
        # Drop temporary table
        conn.execute("DROP TABLE temp_unique_paths")
        conn.commit()
        
        return total_duplicates
    
    # New function for shortest path filtering (only active when batch_grouping=False)
    def filter_shortest_paths(paths_data, shortest_path_level):
        """
        Filter paths to keep only the shortest path lengths.
        
        Parameters:
        -----------
        paths_data : list of tuples
            List of (compressed_path, length, is_complete) tuples
        shortest_path_level : int
            Number of shortest unique lengths to keep
        
        Returns:
        --------
        list : Filtered paths_data keeping only shortest path lengths
        """
        if not paths_data:
            return paths_data
        
        # Get unique lengths and sort them
        lengths = [length for _, length, _ in paths_data]
        unique_lengths = sorted(set(lengths))
        
        # Keep only the shortest_path_level unique lengths
        keep_lengths = set(unique_lengths[:shortest_path_level])
        
        # Filter paths
        filtered_paths = [(path, length, is_complete) for path, length, is_complete in paths_data 
                         if length in keep_lengths]
        
        if debug and len(filtered_paths) < len(paths_data):
            print(f"  Shortest path filtering: kept {len(filtered_paths)}/{len(paths_data)} paths with lengths {sorted(keep_lengths)}")
        
        return filtered_paths
    
    # Lazy computation of metrics and warping paths
    def compute_path_metrics_lazy(compressed_path):
        """Compute quality metrics lazily only when needed for final output."""
        path_segment_pairs = decompress_path(compressed_path)
        
        # Initialize metrics
        metrics = {
            'norm_dtw': 0.0,
            'dtw_ratio': 0.0,
            'variance_deviation': 0.0,
            'perc_diag': 0.0,
            'corr_coef': 0.0,
            'match_min': 0.0,
            'match_mean': 0.0
        }
        
        # Lists to store metric values for averaging
        metric_values = {metric: [] for metric in metrics}
        
        # NEW: Add age overlap percentage collection
        age_overlap_values = []
        
        # Collect all warping paths to combine
        all_wps = []
        
        for a_idx, b_idx in path_segment_pairs:
            if (a_idx, b_idx) in dtw_results:
                paths, _, quality_indicators = dtw_results[(a_idx, b_idx)]
                
                # Skip pairs with no valid path
                if not paths or len(paths) == 0:
                    continue
                    
                # Add warping path to our collection
                wp = paths[0]  # Use the first (best) path
                all_wps.append(wp)
                
                # Add quality metrics
                if quality_indicators and len(quality_indicators) > 0:
                    qi = quality_indicators[0]
                    for metric in metrics:
                        if metric in qi:
                            metric_values[metric].append(float(qi[metric]))
                    
                    # NEW: Collect age overlap percentage if available
                    if 'perc_age_overlap' in qi:
                        age_overlap_values.append(float(qi['perc_age_overlap']))
        
        # Combine warping paths
        if all_wps:
            # Stack warping paths
            combined_wp = np.vstack(all_wps)
            
            # Remove duplicate points
            combined_wp = np.unique(combined_wp, axis=0)
            
            # Sort by first coordinate
            combined_wp = combined_wp[combined_wp[:, 0].argsort()]
        else:
            combined_wp = np.array([])
        
        # Calculate diagonality directly from the combined warping path
        if len(combined_wp) > 1:
            calculated_diagonality = calculate_diagonality(combined_wp)
            metrics['perc_diag'] = float(calculated_diagonality * 100)  # Convert to percentage
        
        # For other metrics, use average of segment metrics
        for metric in metrics:
            if metric != 'perc_diag':  # Skip perc_diag as we calculated it directly
                values = metric_values[metric]
                if values:
                    if metric in ['norm_dtw', 'match_min', 'match_mean']:
                        # For distance metrics, use sum
                        metrics[metric] = float(sum(values))
                    else:
                        # For other metrics, use average
                        metrics[metric] = float(sum(values) / len(values))
        
        # NEW: Calculate average age overlap percentage
        if age_overlap_values:
            avg_age_overlap = float(sum(age_overlap_values) / len(age_overlap_values))
        else:
            avg_age_overlap = 0.0  # Default when no age data available
        
        # NEW: Add to metrics dictionary
        metrics['perc_age_overlap'] = avg_age_overlap
        
        return combined_wp, metrics

    # Database setup functions
    def setup_database(db_path, read_only=False):
        """Setup SQLite database with optimizations."""
        conn = sqlite3.connect(db_path)
        
        # SQLite optimizations
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        if not read_only:
            # Create tables for compressed paths
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
            
            # Create indexes for fast querying
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

    # Setup: Create segment details
    print("Setting up boundary constraints...")
    
    # Get exact max depths for both cores
    max_depth_a = max(depth_boundaries_a)
    max_depth_b = max(depth_boundaries_b)
    
    detailed_pairs = {}
    
    # Identify segments that precisely contain the bottom of BOTH cores
    true_bottom_segments = set()
    
    # Identify segments that precisely contain the top of BOTH cores
    true_top_segments = set()
    
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
        
        # Check if this segment contains the bottom of BOTH cores
        if abs(a_end - max_depth_a) < 1e-6 and abs(b_end - max_depth_b) < 1e-6:
            true_bottom_segments.add((a_idx, b_idx))
        
        # Check if this segment contains the top of BOTH cores
        if abs(a_start) < 1e-6 and abs(b_start) < 1e-6:
            true_top_segments.add((a_idx, b_idx))
    
    # Only print segments with valid DTW
    valid_top_segments = true_top_segments.intersection(valid_dtw_pairs)
    valid_bottom_segments = true_bottom_segments.intersection(valid_dtw_pairs)
    
    print(f"Identified {len(valid_top_segments)} valid segments at the top of both cores")
    print(f"Valid top segments (1-based indices): {[(a_idx+1, b_idx+1) for a_idx, b_idx in valid_top_segments]}")
    print(f"Identified {len(valid_bottom_segments)} valid segments at the bottom of both cores")
    print(f"Valid bottom segments (1-based indices): {[(a_idx+1, b_idx+1) for a_idx, b_idx in valid_bottom_segments]}")

    # If no true bottom or top segments, exit early
    if not true_bottom_segments:
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

    # ===== INTEGRATED COMPLETE PATH COMPUTATION =====
    print(f"\n=== COMPLETE PATH COMPUTATION ===")
    path_computation_results = compute_total_complete_paths(valid_dtw_pairs, detailed_pairs, max_depth_a, max_depth_b)

    # Build predecessor and successor relationships
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
                
                if (abs(next_a_start - a_end) < 1e-6 and 
                    abs(next_b_start - b_end) < 1e-6):
                    # (a_idx, b_idx) precedes (next_a_idx, next_b_idx)
                    successor_lookup[(a_idx, b_idx)].append((next_a_idx, next_b_idx))
                    predecessor_lookup[(next_a_idx, next_b_idx)].append((a_idx, b_idx))
    
    # Filter top segments if necessary
    final_top_segments = true_top_segments
    if start_from_top_only:
        # FILTER: Only allow specific top pairs that are also true top segments
        allowed_top_pairs = [(1,0), (1,1), (0,1)]
        final_top_segments = {seg for seg in true_top_segments if seg in allowed_top_pairs}
        
    print(f"Using {len(final_top_segments)} valid top segments for path starting points")
    
    # Topological ordering
    def topological_sort():
        visited = set()
        temp_visited = set()
        order = []
        
        def dfs(segment):
            if segment in temp_visited:
                return False
            
            if segment in visited:
                return True
                
            temp_visited.add(segment)
            
            # Visit all successors first
            for next_segment in successor_lookup[segment]:
                if not dfs(next_segment):
                    return False
            
            # Add current segment to order
            temp_visited.remove(segment)
            visited.add(segment)
            order.append(segment)
            return True
        
        # Start DFS from all top segments
        for segment in final_top_segments:
            if segment not in visited:
                if not dfs(segment):
                    print("Warning: Cycle detected in segment relationships. Using BFS ordering instead.")
                    return None
        
        # Add any remaining segments (should be rare)
        for segment in valid_dtw_pairs:
            if segment not in visited:
                if not dfs(segment):
                    print("Warning: Cycle detected in segment relationships. Using BFS ordering instead.")
                    return None
        
        return list(reversed(order))  # Reverse to get top-to-bottom order
    
    # Get topological ordering or fall back to BFS-based level ordering
    topo_order = topological_sort()
    
    if topo_order is None:
        # Fall back to level-based ordering if cycles exist
        print("Using level-based ordering instead of topological sort...")
        
        # Assign levels to each segment (shortest distance from any top segment)
        levels = {}
        queue = deque([(seg, 0) for seg in final_top_segments])
        
        while queue:
            segment, level = queue.popleft()
            
            if segment in levels:
                # Already processed with a shorter path
                continue
                
            levels[segment] = level
            
            # Process successors
            for next_segment in successor_lookup[segment]:
                if next_segment not in levels:
                    queue.append((next_segment, level + 1))
        
        # Sort segments by level
        topo_order = sorted(valid_dtw_pairs, key=lambda seg: levels.get(seg, float('inf')))
    
    print(f"Identified {len(topo_order)} segments in processing order")
    
    # Create database directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for databases: {temp_dir}")
    
    # Setup shared read database
    shared_read_db_path = os.path.join(temp_dir, "shared_read.db")
    shared_read_conn = setup_database(shared_read_db_path, read_only=False)
    
    # Initialize with top segment paths
    print("Initializing shared database with top segments...")
    for segment in final_top_segments:
        compressed_path = compress_path([segment])
        insert_compressed_path(shared_read_conn, segment, segment, compressed_path, 1, False)
    shared_read_conn.commit()
    
    # Always use batch size = 1 for complete path enumeration
    segment_groups = []
    current_group = []
    
    for segment in topo_order:
        current_group.append(segment)
        # ALWAYS process 1 segment at a time for maximum reliability
        if len(current_group) >= 1:  
            segment_groups.append(current_group)
            current_group = []
    
    if current_group:
        segment_groups.append(current_group)
    
    print(f"Processing {len(topo_order)} segments in {len(segment_groups)} groups (1 segment per group for complete enumeration)")
    
    # Initialize complete path counter for max_search_path limit (only when batch_grouping=False)
    complete_paths_found = 0
    search_limit_reached = False
    
    # Enhanced processing with incremental duplicate removal and new optimizations
    def process_segment_group_with_database_and_dedup(group_idx, segment_group, shared_read_conn):
        """Process a group of segments with optimized database operations and incremental duplicate removal."""
        nonlocal complete_paths_found, search_limit_reached
        
        # OPTIMIZATION 1: Use in-memory database for temporary storage
        group_write_conn = sqlite3.connect(":memory:")
        
        # OPTIMIZATION 2: Configure for speed over durability (safe for temp data)
        group_write_conn.execute("PRAGMA synchronous = OFF")
        group_write_conn.execute("PRAGMA journal_mode = MEMORY")
        group_write_conn.execute("PRAGMA cache_size = 50000")  # Larger cache
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
        
        # Add index for duplicate detection in temporary database
        group_write_conn.execute("CREATE INDEX idx_compressed_path ON compressed_paths(compressed_path)")
        
        # Batch inserts for better performance
        batch_inserts = []
        complete_paths_count = 0
        
        # Process each segment in the group
        for segment in segment_group:
            # Check if search limit reached (only when batch_grouping=False and max_search_path is not None)
            if not batch_grouping and max_search_path is not None and complete_paths_found >= max_search_path:
                search_limit_reached = True
                if debug:
                    print(f"  Search limit reached ({max_search_path} complete paths). Stopping search.")
                break
            
            # Get all predecessors
            direct_predecessors = predecessor_lookup[segment]
            
            # Batch read predecessor paths
            predecessor_paths = []
            
            if direct_predecessors:
                # Read predecessor paths in a single query
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
            
            # Skip if no paths to process
            if not predecessor_paths:
                continue
            
            # Collect all new paths for this segment
            new_paths_data = []
            
            # Process paths in batches
            for compressed_pred_path in predecessor_paths:
                pred_path = decompress_path(compressed_pred_path)
                
                # Extend path with current segment
                if not pred_path or pred_path[-1] != segment:
                    extended_path = pred_path + [segment]
                else:
                    extended_path = pred_path
                
                compressed_extended_path = compress_path(extended_path)
                is_complete = segment in true_bottom_segments
                
                # Store path data for filtering
                new_paths_data.append((compressed_extended_path, len(extended_path), is_complete))
                
                if is_complete:
                    complete_paths_count += 1
            
            # Apply shortest path filtering if enabled (only when batch_grouping=False)
            if not batch_grouping and shortest_path_search:
                new_paths_data = filter_shortest_paths(new_paths_data, shortest_path_level)
            
            # Convert filtered paths to batch inserts
            for compressed_extended_path, length, is_complete in new_paths_data:
                extended_path = decompress_path(compressed_extended_path)
                
                # Add to batch instead of immediate insert
                batch_inserts.append((
                    f"{extended_path[0][0]},{extended_path[0][1]}",
                    f"{extended_path[-1][0]},{extended_path[-1][1]}", 
                    compressed_extended_path,
                    length,
                    is_complete
                ))
                
                # Batch insert when batch gets large
                if len(batch_inserts) >= 5000:  # Larger batch size
                    group_write_conn.executemany("""
                        INSERT INTO compressed_paths (start_segment, last_segment, compressed_path, length, is_complete)
                        VALUES (?, ?, ?, ?, ?)
                    """, batch_inserts)
                    batch_inserts = []
                
                # Check for random sampling if we're approaching the limit (only when batch_grouping=False)
                if not batch_grouping and max_search_path is not None:
                    if is_complete:
                        complete_paths_found += 1
                        if complete_paths_found >= max_search_path:
                            search_limit_reached = True
                            if debug:
                                print(f"  Search limit reached ({max_search_path} complete paths). Stopping search.")
                            break
            
            if search_limit_reached:
                break
        
        # Insert remaining batch
        if batch_inserts:
            group_write_conn.executemany("""
                INSERT INTO compressed_paths (start_segment, last_segment, compressed_path, length, is_complete)
                VALUES (?, ?, ?, ?, ?)
            """, batch_inserts)
        
        # Remove duplicates from this group's database before returning
        duplicates_removed = remove_duplicates_from_db(group_write_conn, f"Group {group_idx+1}")
        
        # Recalculate complete paths count after deduplication
        cursor = group_write_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 1")
        complete_paths_count_after_dedup = cursor.fetchone()[0]
        
        # Return in-memory database connection for bulk transfer
        return group_write_conn, complete_paths_count_after_dedup, duplicates_removed
    
    # Process all groups with bulk operations and incremental deduplication
    total_complete_paths = 0
    total_duplicates_removed = 0
    
    # Only batch sync if batch_grouping=True
    if batch_grouping:
        # Batch processing: sync every few groups for performance
        if n_groups is not None:
            sync_every_n_groups = max(1, min(n_groups, len(segment_groups) // 10))
        else:
            sync_every_n_groups = max(1, min(5, len(segment_groups) // 10))
        print(f"Batch grouping enabled: will sync every {sync_every_n_groups} groups with incremental duplicate removal")
    else:
        # No grouping: sync after every single group for immediate availability
        sync_every_n_groups = 1
        batch_grouping_msg = "syncing after every segment with incremental duplicate removal for maximum reliability"
        
        # Add information about new optimizations when batch_grouping=False
        optimization_msgs = []
        if shortest_path_search:
            optimization_msgs.append(f"shortest path search (keeping {shortest_path_level} shortest lengths)")
        if max_search_path is not None:
            optimization_msgs.append(f"path limit ({max_search_path} complete paths)")
        
        if optimization_msgs:
            batch_grouping_msg += f" with {' and '.join(optimization_msgs)}"
        
        print(f"Batch grouping disabled: {batch_grouping_msg}")
    
    with tqdm(total=len(segment_groups), desc="Processing segment groups") as pbar:
        group_results = []  # Store results for batch sync
        
        for group_idx, segment_group in enumerate(segment_groups):
            
            # Check if search should stop due to limit (only when batch_grouping=False)
            if not batch_grouping and search_limit_reached:
                if debug:
                    print(f"Stopping processing due to search limit reached")
                break
            
            # Process group with optimized database operations and incremental deduplication
            group_write_conn, group_complete_paths, group_duplicates = process_segment_group_with_database_and_dedup(
                group_idx, segment_group, shared_read_conn
            )
            
            group_results.append((group_write_conn, group_complete_paths, group_duplicates))
            total_complete_paths += group_complete_paths
            total_duplicates_removed += group_duplicates
            
            # Sync behavior depends on batch_grouping parameter
            should_sync = (
                (group_idx + 1) % sync_every_n_groups == 0 or  # Every N groups (N=1 if batch_grouping=False)
                group_idx == len(segment_groups) - 1 or  # Last group
                search_limit_reached  # Search limit reached
            )
            
            if should_sync:
                # Only show batch sync message if actually batching multiple groups
                if batch_grouping and len(group_results) > 1:
                    if debug:
                        print(f"Syncing {len(group_results)} groups to shared database...")
                
                # Bulk transfer from all accumulated group databases
                for group_conn, _, _ in group_results:
                    # Bulk transfer using executemany
                    cursor = group_conn.execute("""
                        SELECT start_segment, last_segment, compressed_path, length, is_complete 
                        FROM compressed_paths
                    """)
                    
                    # Read all rows into memory (safe since we're using small batches)
                    all_rows = cursor.fetchall()
                    
                    if all_rows:
                        # Single bulk insert operation
                        shared_read_conn.executemany("""
                            INSERT INTO compressed_paths (start_segment, last_segment, compressed_path, length, is_complete)
                            VALUES (?, ?, ?, ?, ?)
                        """, all_rows)
                    
                    # Close the in-memory database
                    group_conn.close()
                
                # Single commit for all groups
                shared_read_conn.commit()
                
                # Remove duplicates from shared database after sync (when batch_grouping=True)
                if batch_grouping and len(group_results) > 1:
                    shared_duplicates = remove_duplicates_from_db(shared_read_conn, f"Shared DB after batch sync")
                    total_duplicates_removed += shared_duplicates
                
                # Clear the results batch
                group_results = []
                
                # Garbage collection frequency based on grouping
                if batch_grouping:
                    # Less frequent GC when batching
                    if group_idx % (sync_every_n_groups * 2) == 0:
                        gc.collect()
                else:
                    # More frequent GC when not batching (every 10 segments)
                    if group_idx % 10 == 0:
                        gc.collect()
            
            # Update progress
            pbar.update(1)
            if batch_grouping:
                pbar.set_postfix({
                    "batch": f"{(group_idx // sync_every_n_groups) + 1}",
                    "complete_paths": total_complete_paths,
                    "duplicates_removed": total_duplicates_removed
                })
            else:
                postfix_dict = {
                    "segment": f"{group_idx + 1}/{len(segment_groups)}",
                    "complete_paths": total_complete_paths,
                    "duplicates_removed": total_duplicates_removed
                }
                if max_search_path is not None:
                    postfix_dict["limit"] = f"{complete_paths_found}/{max_search_path}"
                pbar.set_postfix(postfix_dict)
            
            # Break if search limit reached
            if search_limit_reached:
                break
    
    # Final deduplication on shared database
    print("Performing final deduplication on complete database...")
    final_duplicates = remove_duplicates_from_db(shared_read_conn, "Final cleanup")
    total_duplicates_removed += final_duplicates
    
    # Get final count after all deduplication
    cursor = shared_read_conn.execute("SELECT COUNT(*) FROM compressed_paths WHERE is_complete = 1")
    final_complete_paths = cursor.fetchone()[0]
    
    # Print completion message with search limit information
    completion_msg = f"Processing complete. Found {final_complete_paths} unique complete paths after removing {total_duplicates_removed} duplicates."
    if not batch_grouping and search_limit_reached:
        completion_msg += f" (Search stopped at limit of {max_search_path} complete paths)"
    print(completion_msg)
    
    # Direct output generation from deduplicated database
    print("\n=== Computing Metrics and Generating CSV Output ===")
    
    # Create output CSV with batch processing for memory efficiency
    def generate_output_csv():
        """Generate final CSV output directly from deduplicated database using parallel processing."""
        
        # Create output CSV with header - ADD perc_age_overlap column
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mapping_id', 'path', 'length', 'combined_wp', 
                        'norm_dtw', 'dtw_ratio', 'variance_deviation', 
                        'perc_diag', 'corr_coef', 'match_min', 'match_mean', 'perc_age_overlap'])
        
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
                combined_wp, metrics = compute_path_metrics_lazy(compressed_path)
                
                # Format warping path compactly
                if combined_wp is not None and len(combined_wp) > 0:
                    combined_wp_compact = ";".join(f"{int(wp[0])},{int(wp[1])}" for wp in combined_wp)
                else:
                    combined_wp_compact = ""
                
                # Add result - INCLUDE perc_age_overlap in the output
                batch_results.append([
                    mapping_id, 
                    formatted_path_compact,
                    length,
                    combined_wp_compact,
                    round(metrics['norm_dtw'], 6),
                    round(metrics['dtw_ratio'], 6),
                    round(metrics['variance_deviation'], 6),
                    round(metrics['perc_diag'], 2),
                    round(metrics['corr_coef'], 6),
                    round(metrics['match_min'], 6),
                    round(metrics['match_mean'], 6),
                    round(metrics['perc_age_overlap'], 2)  # NEW: Add age overlap percentage
                ])
                
                mapping_id += 1
                
            return batch_results
        
        print(f"Processing {total_paths} paths in {len(batches)} batches")
        
        # Process batches in parallel
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for batch_idx, batch in enumerate(batches):
                # Calculate starting ID for this batch
                start_id = batch_idx * batch_size + 1
                
                # Process this batch
                batch_results = process_batch(batch, start_id)
                
                # Write batch results
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_results)
                
                pbar.update(1)
                
                # Periodic garbage collection
                if batch_idx % 5 == 0:
                    gc.collect()
        
        return total_paths
    
    # Generate the output
    total_paths_written = generate_output_csv()
    
    # Close shared database
    shared_read_conn.close()
    
    # Print final statistics
    print(f"\nFinal Results:")
    print(f"  Total unique complete paths written: {total_paths_written}")
    print(f"  Total duplicates removed during processing: {total_duplicates_removed}")
    print(f"  Deduplication efficiency: {(total_duplicates_removed/(total_paths_written + total_duplicates_removed)*100) if (total_paths_written + total_duplicates_removed) > 0 else 0:.2f}%")
    
    # Add search limit information to final results
    if not batch_grouping and search_limit_reached:
        print(f"  Search was limited to {max_search_path} complete paths for performance")
    
    # Cleanup - remove all temporary files
    try:
        print("Cleaning up temporary databases...")
        import shutil
        shutil.rmtree(temp_dir)
        print("Cleanup complete.")
    except Exception as e:
        print(f"Could not clean temporary directory: {e}")
    
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
        'search_limit_reached': search_limit_reached if not batch_grouping else False
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