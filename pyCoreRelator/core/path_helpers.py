"""
Helper functions for path finding operations.

This module contains utility functions that support the find_complete_core_paths function,
extracted to improve code organization while maintaining exact functionality.
"""

import numpy as np
import gc
import psutil
import sqlite3


def check_memory(threshold_percent=85, mute_mode=False):
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


def remove_duplicates_from_db(conn, debug=False, mute_mode=False):
    """Remove duplicate paths from database and return count of removed duplicates."""
    if debug and not mute_mode:
        print(f"Removing duplicates from database...")
    
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


def filter_shortest_paths(paths_data, shortest_path_level, debug=False, mute_mode=False):
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


def compute_path_metrics_lazy(compressed_path, log_a, log_b, dtw_results):
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


def prune_shared_database_if_needed(shared_conn, max_paths, debug=False, mute_mode=False):
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
        SELECT rowid FROM compressed_paths 
        WHERE is_complete = 0 
        ORDER BY RANDOM() 
        LIMIT ?
    """, (paths_to_remove,))
    
    rowids_to_remove = [row[0] for row in cursor.fetchall()]
    
    if rowids_to_remove:
        placeholders = ','.join('?' * len(rowids_to_remove))
        shared_conn.execute(f"DELETE FROM compressed_paths WHERE rowid IN ({placeholders})", rowids_to_remove)
        shared_conn.commit()
    
    return paths_to_remove 