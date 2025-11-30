"""
Diagnostic functions for core correlation analysis.

Included Functions:
- diagnose_chain_breaks: Diagnose chain breaks in the segment network.

This module provides diagnostic functionality for geological core correlation workflows,
enabling detailed analysis of segment connectivity, path completeness, and identification
of breaks in correlation chains.
"""

from collections import deque
from .segments import identify_special_segments
from .path_finding import compute_total_complete_paths


def diagnose_chain_breaks(dtw_result):
    """
    Comprehensive diagnostic to find exactly where segment chains break.
    
    This function will trace all possible paths and identify missing connections.
    Additionally computes total complete paths and finds the "far most" bounding complete paths.
    
    Parameters:
    -----------
    dtw_result : dict
        Dictionary containing DTW analysis results from run_comprehensive_dtw_analysis().
        Expected keys: 'valid_dtw_pairs', 'segments_a', 'segments_b', 
        'depth_boundaries_a', 'depth_boundaries_b'
    
    Returns:
    --------
    dict: Enhanced results including complete path counts and bounding paths
    """
    
    # Extract variables from unified dictionary
    valid_dtw_pairs = dtw_result['valid_dtw_pairs']
    segments_a = dtw_result['segments_a']
    segments_b = dtw_result['segments_b']
    depth_boundaries_a = dtw_result['depth_boundaries_a']
    depth_boundaries_b = dtw_result['depth_boundaries_b']
    
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