"""
Synthetic stratigraphy functions for pyCoreRelator

This module provides functions for generating synthetic core data and running
null hypothesis tests for correlation analysis. It includes segment pool management,
synthetic log generation, and visualization tools.

Functions:
- load_segment_pool: Load segment pool data from turbidite database
- modify_segment_pool: Remove unwanted segments from the pool data
- create_synthetic_log: Create synthetic log using turbidite database approach with picked depths at turbidite bases
- create_synthetic_core_pair: Generate synthetic core pair and optionally plot the results
- plot_segment_pool: Plot all segments from the pool in a grid layout
- plot_synthetic_log: Plot a single synthetic log with turbidite boundaries
- synthetic_correlation_quality: Generate DTW correlation quality analysis for synthetic core pairs with multiple iterations
- plot_synthetic_correlation_quality: Plot synthetic correlation quality distributions from saved CSV files
- generate_constraint_subsets: Generate all possible subsets of constraints (2^n combinations)
- run_multi_parameter_analysis: Run comprehensive multi-parameter analysis for core correlation
"""

# Data manipulation and analysis
import os
import gc
import random
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import from other pyCoreRelator modules
from ..utils.data_loader import load_log_data
from .dtw_core import run_comprehensive_dtw_analysis
from .path_finding import find_complete_core_paths
from .age_models import calculate_interpolated_ages
# Note: plot_correlation_distribution is imported inside functions to avoid circular imports

# Scikit-learn imports for Markov Chain clustering (imported here to check availability)
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KneeLocator = None
    KNEED_AVAILABLE = False


# =============================================================================
# MARKOV CHAIN HELPER FUNCTIONS
# =============================================================================

def find_optimal_k(X_scaled, k_range=range(2, 16)):
    """
    Find optimal number of clusters using Kneedle/elbow method.
    
    Parameters
    ----------
    X_scaled : array-like of shape (n_samples, n_features)
        Scaled feature array (should already be scaled before passing)
    k_range : range or list, default=range(2, 11)
        Range of k values to test
    
    Returns
    -------
    optimal_k : int
        Optimal number of clusters
    elbow_info : dict
        Dictionary with 'inertias', 'distances', 'k_list' for plotting
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for Markov Chain clustering. Install with: pip install scikit-learn")
    
    k_list = list(k_range)
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Normalize for distance calculation
    k_norm = (np.array(k_list) - k_list[0]) / (k_list[-1] - k_list[0])
    inertia_norm = (np.array(inertias) - inertias[-1]) / (inertias[0] - inertias[-1])
    
    # Perpendicular distance from line y = 1 - x (Kneedle algorithm)
    distances = np.abs(k_norm + inertia_norm - 1) / np.sqrt(2)
    optimal_k = k_list[np.argmax(distances)]
    
    elbow_info = {
        'inertias': inertias,
        'distances': distances.tolist(),
        'k_list': k_list
    }
    
    return optimal_k, elbow_info


def build_transition_matrix(cluster_labels, unit_sequence_per_core, n_clusters):
    """
    Build transition probability matrix from cluster sequences.
    
    Parameters
    ----------
    cluster_labels : array-like of shape (n_units,)
        Cluster assignment for each unit
    unit_sequence_per_core : dict
        Dictionary mapping core_name -> list of unit indices (deepest to shallowest).
        Indices refer to positions in cluster_labels array.
    n_clusters : int
        Number of clusters
    
    Returns
    -------
    transition_matrix : ndarray of shape (n_clusters, n_clusters)
        Transition probability matrix where [i,j] is P(cluster j | cluster i)
    stationary_dist : ndarray of shape (n_clusters,)
        Stationary distribution of the Markov chain
    """
    cluster_labels = np.asarray(cluster_labels)
    
    # Build transition count matrix from observed sequences
    transition_counts = np.zeros((n_clusters, n_clusters), dtype=int)
    
    for core_name, unit_indices in unit_sequence_per_core.items():
        # Get cluster sequence for this core (deepest to shallowest)
        cluster_seq = [cluster_labels[idx] for idx in unit_indices]
        
        # Count transitions
        for i in range(len(cluster_seq) - 1):
            from_cluster = cluster_seq[i]
            to_cluster = cluster_seq[i + 1]
            transition_counts[from_cluster, to_cluster] += 1
    
    # Convert to probability matrix
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero temporarily
    transition_matrix = transition_counts / row_sums
    
    # For clusters with no observed transitions, use uniform distribution
    # (required for valid probability distribution that sums to 1)
    for i in range(n_clusters):
        if zero_rows[i]:
            transition_matrix[i, :] = 1.0 / n_clusters
    
    # Compute stationary distribution via eigenvalue decomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1))
        stationary_dist = np.real(eigenvectors[:, stationary_idx])
        stationary_dist = np.abs(stationary_dist) / np.abs(stationary_dist).sum()
    except:
        # Fallback: use uniform distribution
        stationary_dist = np.ones(n_clusters) / n_clusters
    
    return transition_matrix, stationary_dist


def build_higher_order_transitions(cluster_labels, unit_sequence_per_core, n_clusters, order):
    """
    Build higher-order transition probability dictionary.
    
    For an n-th order Markov chain, the next state depends on the previous n states.
    
    Parameters
    ----------
    cluster_labels : array-like of shape (n_units,)
        Cluster assignment for each unit
    unit_sequence_per_core : dict
        Dictionary mapping core_name -> list of unit indices (deepest to shallowest).
    n_clusters : int
        Number of clusters
    order : int
        Order of the Markov chain (number of previous states to consider)
    
    Returns
    -------
    transition_dict : dict
        Dictionary mapping tuples of previous states to probability arrays.
        Key: tuple of length `order` representing previous states
        Value: 1D array of shape (n_clusters,) with transition probabilities
    """
    from collections import defaultdict
    
    cluster_labels = np.asarray(cluster_labels)
    
    # Count transitions: (prev_n_states) -> next_state
    transition_counts = defaultdict(lambda: np.zeros(n_clusters))
    
    for core_name, unit_indices in unit_sequence_per_core.items():
        # Get cluster sequence for this core (deepest to shallowest)
        cluster_seq = [cluster_labels[idx] for idx in unit_indices]
        
        if len(cluster_seq) <= order:
            continue
            
        for i in range(len(cluster_seq) - order):
            # Previous n states as tuple
            prev_states = tuple(cluster_seq[i:i + order])
            next_state = cluster_seq[i + order]
            transition_counts[prev_states][next_state] += 1
    
    # Convert counts to probabilities
    transition_dict = {}
    for prev_states, counts in transition_counts.items():
        total = counts.sum()
        if total > 0:
            transition_dict[prev_states] = counts / total
        else:
            transition_dict[prev_states] = np.ones(n_clusters) / n_clusters
    
    return transition_dict


def _get_higher_order_transition_probs(cluster_history, transition_dict, transition_matrix, 
                                        stationary_dist, mc_order):
    """
    Get transition probabilities for higher-order Markov chain with fallback.
    
    Parameters
    ----------
    cluster_history : list
        History of cluster states (most recent at end)
    transition_dict : dict
        Higher-order transition dictionary mapping state tuples to probabilities
    transition_matrix : ndarray
        1st-order transition matrix for fallback
    stationary_dist : ndarray
        Stationary distribution for ultimate fallback
    mc_order : int
        Order of the Markov chain
    
    Returns
    -------
    transition_probs : ndarray
        Transition probability array
    """
    # Get the last mc_order states as tuple
    if len(cluster_history) >= mc_order:
        prev_tuple = tuple(cluster_history[-mc_order:])
    else:
        prev_tuple = tuple(cluster_history)
    
    # Try exact match first
    if prev_tuple in transition_dict:
        return transition_dict[prev_tuple]
    
    # Fallback: try progressively shorter history
    for length in range(len(prev_tuple) - 1, 0, -1):
        shorter_tuple = prev_tuple[-length:]
        if shorter_tuple in transition_dict:
            return transition_dict[shorter_tuple]
    
    # Fallback to 1st-order transition matrix
    if len(cluster_history) > 0:
        last_cluster = cluster_history[-1]
        return transition_matrix[last_cluster]
    
    # Ultimate fallback: stationary distribution
    return stationary_dist


def build_vom_transitions(cluster_labels, unit_sequence_per_core, n_clusters, max_order):
    """
    Build Variable-Order Markov (VOM) transition dictionaries for all orders.
    
    VOM uses the longest observed context (up to max_order) for predictions,
    naturally handling unseen n-grams by falling back to shorter contexts.
    
    Parameters
    ----------
    cluster_labels : array-like
        Cluster assignment for each unit
    unit_sequence_per_core : dict
        Dictionary mapping core_name -> list of unit indices
    n_clusters : int
        Number of clusters
    max_order : int
        Maximum order to build transitions for
    
    Returns
    -------
    vom_transition_dicts : dict
        Dictionary mapping order -> transition_dict for orders 1 to max_order
    """
    cluster_labels = np.asarray(cluster_labels)
    vom_transition_dicts = {}
    
    for order in range(1, max_order + 1):
        transition_counts = {}
        
        for core_name, unit_indices in unit_sequence_per_core.items():
            cluster_seq = [cluster_labels[idx] for idx in unit_indices]
            
            if len(cluster_seq) <= order:
                continue
            
            for i in range(len(cluster_seq) - order):
                prev_state = tuple(cluster_seq[i:i + order])
                next_state = cluster_seq[i + order]
                
                if prev_state not in transition_counts:
                    transition_counts[prev_state] = {}
                if next_state not in transition_counts[prev_state]:
                    transition_counts[prev_state][next_state] = 0
                transition_counts[prev_state][next_state] += 1
        
        # Normalize to probabilities (store as dict of dicts for VOM)
        transition_dict = {}
        for prev_state, next_counts in transition_counts.items():
            total = sum(next_counts.values())
            transition_dict[prev_state] = {s: c/total for s, c in next_counts.items()}
        
        vom_transition_dicts[order] = transition_dict
    
    return vom_transition_dicts


def build_fp_transitions(cluster_labels, unit_sequence_per_core, n_clusters, order):
    """
    Build Forbidden Path (FP) transition dictionary and observed starting sequences.
    
    FP only allows transitions from n-grams that were observed in training data.
    Initializes sequences from observed starting patterns to avoid phantom histories.
    
    Parameters
    ----------
    cluster_labels : array-like
        Cluster assignment for each unit
    unit_sequence_per_core : dict
        Dictionary mapping core_name -> list of unit indices
    n_clusters : int
        Number of clusters
    order : int
        Order of the Markov chain
    
    Returns
    -------
    transition_dict : dict
        Dictionary mapping observed n-grams to next-state probability dicts
    observed_starts : list
        List of observed starting n-gram sequences
    """
    cluster_labels = np.asarray(cluster_labels)
    transition_counts = {}
    observed_starts = set()
    
    for core_name, unit_indices in unit_sequence_per_core.items():
        cluster_seq = [cluster_labels[idx] for idx in unit_indices]
        
        if len(cluster_seq) <= order:
            continue
        
        # Record the starting sequence (first 'order' clusters)
        if len(cluster_seq) >= order:
            observed_starts.add(tuple(cluster_seq[:order]))
        
        # Build transition counts
        for i in range(len(cluster_seq) - order):
            prev_state = tuple(cluster_seq[i:i + order])
            next_state = cluster_seq[i + order]
            
            if prev_state not in transition_counts:
                transition_counts[prev_state] = {}
            if next_state not in transition_counts[prev_state]:
                transition_counts[prev_state][next_state] = 0
            transition_counts[prev_state][next_state] += 1
    
    # Normalize to probabilities
    transition_dict = {}
    for prev_state, next_counts in transition_counts.items():
        total = sum(next_counts.values())
        transition_dict[prev_state] = {s: c/total for s, c in next_counts.items()}
    
    return transition_dict, list(observed_starts)


def _get_vom_transition_probs(cluster_history, vom_transition_dicts, stationary_dist, n_clusters, max_order):
    """
    Get transition probabilities using Variable-Order Markov (longest matching context).
    
    Returns
    -------
    probs : ndarray
        Transition probability array
    order_used : int
        The order that was actually used (0 if stationary fallback)
    """
    # Try from max_order down to 1
    for order in range(min(len(cluster_history), max_order), 0, -1):
        context = tuple(cluster_history[-order:])
        if order in vom_transition_dicts and context in vom_transition_dicts[order]:
            probs = np.zeros(n_clusters)
            for next_state, prob in vom_transition_dicts[order][context].items():
                probs[next_state] = prob
            return probs, order
    
    # Fallback to stationary distribution
    return stationary_dist, 0


def _get_fp_transition_probs(cluster_history, fp_transition_dict, stationary_dist, n_clusters, order):
    """
    Get transition probabilities for Forbidden Path model.
    
    Returns None if the current n-gram was never observed (forbidden path).
    
    Returns
    -------
    probs : ndarray or None
        Transition probability array, or None if forbidden
    is_valid : bool
        True if the n-gram was observed, False if forbidden
    """
    if len(cluster_history) < order:
        return stationary_dist, False
    
    context = tuple(cluster_history[-order:])
    
    if context in fp_transition_dict:
        probs = np.zeros(n_clusters)
        for next_state, prob in fp_transition_dict[context].items():
            probs[next_state] = prob
        return probs, True
    
    return None, False  # Forbidden path


def _plot_higher_order_transition_matrix(transition_dict, n_clusters, order, show_plots, save_figure_func,
                                          observed_only=False):
    """
    Plot a higher-order transition matrix.
    
    For n-th order MC, rows represent history sequences of length n,
    and columns represent the single next state.
    
    Parameters
    ----------
    transition_dict : dict
        Dictionary mapping state tuples to probability arrays
    n_clusters : int
        Number of clusters
    order : int
        Order of the Markov chain
    show_plots : bool
        Whether to display the plot
    save_figure_func : callable
        Function to save the figure
    observed_only : bool, default=False
        If True, only show rows for observed histories (for VOM/FP modes).
        If False, show all possible histories including phantom ones (for basic mode).
    """
    from itertools import product
    
    # Generate all possible history sequences of length `order`
    all_histories = list(product(range(n_clusters), repeat=order))
    n_total_histories = len(all_histories)  # n_clusters^order
    
    # Count observed histories
    n_observed = sum(1 for h in all_histories if h in transition_dict)
    
    if observed_only:
        # Only include observed histories (for VOM/FP modes)
        observed_histories = [h for h in all_histories if h in transition_dict]
        if len(observed_histories) == 0:
            print(f"  No observed {order}-grams to plot")
            return
        histories_to_plot = observed_histories
    else:
        # Include all possible histories (for basic mode)
        histories_to_plot = all_histories
    
    n_rows = len(histories_to_plot)
    
    # Build the transition matrix: rows = histories, columns = next state
    trans_matrix = np.zeros((n_rows, n_clusters))
    
    for row_idx, history in enumerate(histories_to_plot):
        if history in transition_dict:
            trans_matrix[row_idx, :] = transition_dict[history]
        else:
            # No observed transitions for this history - leave as zeros
            trans_matrix[row_idx, :] = 0
    
    # Create row labels: e.g., "C1,C2" for 2nd order, "C1,C2,C3" for 3rd order
    row_labels = [','.join([f'C{c+1}' for c in h]) for h in histories_to_plot]
    col_labels = [f'C{i+1}' for i in range(n_clusters)]
    
    # Determine figure size based on matrix dimensions
    fig_height = max(4, min(12, 0.3 * n_rows + 1))
    fig_width = max(4, min(8, 0.5 * n_clusters + 2))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(trans_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    
    # Add text annotations if matrix is not too large
    if n_rows <= 27 and n_clusters <= 9:  # Up to 3^3 histories
        for i in range(n_rows):
            for j in range(n_clusters):
                val = trans_matrix[i, j]
                if val > 0:  # Only show non-zero values
                    text_color = 'white' if val > 0.5 else 'black'
                    fontsize = 7 if n_rows > 9 else 9
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           fontsize=fontsize, color=text_color)
    
    # Set axis labels
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7 if n_rows > 9 else 9)
    
    ax.set_xlabel('Next State', fontsize=10)
    ax.set_ylabel(f'History (previous {order} states)', fontsize=10)
    
    # Add ordinal suffix
    ordinal = {1: 'st', 2: 'nd', 3: 'rd'}.get(order, 'th')
    if observed_only:
        ax.set_title(f'{order}{ordinal}-Order Transition Matrix\n({n_observed}/{n_total_histories} histories observed, showing observed only)', 
                     fontsize=10)
    else:
        ax.set_title(f'{order}{ordinal}-Order Transition Matrix\n({n_observed}/{n_total_histories} histories observed)', 
                     fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=9)
    plt.tight_layout()
    save_figure_func(fig, f'markov_transition_matrix_order{order}')
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def train_markov_model(features, unit_sequence_per_core, n_clusters=None, k_range=range(2, 11),
                       mc_order=1, mc_model='VOM', feature_names=None,
                       show_plots=True, savefig=False, save_path=None, save_format='png',
                       save_prefix='', save_dpi=150):
    """
    Train Markov model from unit features and stacking sequences.
    
    This function performs K-means clustering on the provided features and builds
    a transition probability matrix based on observed cluster sequences in cores.
    Supports 1st-order, higher-order, VOM, and FP Markov chains.
    
    Parameters
    ----------
    features : array-like of shape (n_units, n_features)
        Feature array for all units. Can be any features computed by the user
        (thickness, mud_cap_fraction, log ratios, etc.)
    unit_sequence_per_core : dict
        Dictionary mapping core_name -> list of unit indices (deepest to shallowest).
        Indices refer to positions in the features array.
        Used to build transition matrix from observed cluster sequences.
    n_clusters : int or None, default=None
        Number of clusters. If None, auto-detect using elbow method.
    k_range : range or list, default=range(2, 11)
        Range of k values for elbow method (used if n_clusters is None)
    mc_order : int, default=1
        Order of the Markov chain. 
        - mc_order=1: Standard 1st-order MC where next state depends only on current state.
        - mc_order>1: Higher-order MC where next state depends on previous `mc_order` states.
    mc_model : str, default='VOM'
        Markov chain model type for higher-order chains (mc_order > 1):
        - 'basic': Simple higher-order MC without phantom history handling.
        - 'VOM': Variable-Order Markov Model - uses longest observed context up to mc_order.
          Naturally handles unseen n-grams by falling back to shorter contexts.
        - 'FP': Forbidden Path Model - only allows transitions from observed n-grams.
          Initializes sequences from observed starting patterns.
    feature_names : list of str or None, default=None
        Names for each feature column, used for axis labels in scatter plots.
        If None, uses generic names like "Feature 1", "Feature 2", etc.
    show_plots : bool, default=True
        If True, display elbow plot and cluster scatter plots.
    savefig : bool, default=False
        If True, save figures to disk.
    save_path : str or None, default=None
        Directory path for saving figures. Required if savefig=True.
    save_format : str or list, default='png'
        Format(s) for saved figures. Can be single format ('png') or list (['png', 'svg']).
        Supported: 'png', 'jpg', 'svg', 'pdf'
    save_prefix : str, default=''
        Optional prefix for saved filenames (e.g. 'combo0_' for combo0_markov_elbow_plot.png).
    save_dpi : int, default=150
        DPI for saved figures.
    
    Returns
    -------
    markov_params : dict
        Dictionary containing trained model parameters:
        - 'kmeans': trained KMeans model
        - 'scaler': trained RobustScaler
        - 'n_clusters': int
        - 'mc_order': int, the Markov chain order
        - 'mc_model': str, the model type ('basic', 'VOM', or 'FP')
        - 'cluster_labels': array of cluster assignments for training units
        - 'transition_matrix': 2D array of 1st-order transition probabilities
        - 'transition_dict': highest-order transition dict (for basic mc_order>1, else None)
        - 'transition_dicts': dict mapping order -> transition_dict (for basic mode)
        - 'vom_transition_dicts': dict for VOM model (order -> {n-gram: {next: prob}})
        - 'fp_transition_dict': dict for FP model ({n-gram: {next: prob}})
        - 'observed_starts': list of observed starting n-grams (for FP model)
        - 'stationary_dist': 1D array of stationary distribution
        - 'cluster_centers': 2D array of cluster centers (in original scale)
        - 'elbow_info': dict (if n_clusters was auto-detected, else None)
    
    Example
    -------
    >>> # 1st-order Markov Chain
    >>> markov_params = train_markov_model(features, unit_seq)
    >>> 
    >>> # 3rd-order VOM (Variable-Order Markov, default for higher-order)
    >>> markov_params = train_markov_model(features, unit_seq, mc_order=3, mc_model='VOM')
    >>> 
    >>> # 3rd-order FP (Forbidden Path)
    >>> markov_params = train_markov_model(features, unit_seq, mc_order=3, mc_model='FP')
    >>> 
    >>> # 3rd-order basic (no phantom history handling)
    >>> markov_params = train_markov_model(features, unit_seq, mc_order=3, mc_model='basic')
    """
    # Validate mc_model
    mc_model = mc_model.upper()
    if mc_model not in ['BASIC', 'VOM', 'FP']:
        raise ValueError(f"mc_model must be 'basic', 'VOM', or 'FP', got '{mc_model}'")
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for Markov Chain clustering. Install with: pip install scikit-learn")
    
    # Helper function to save figures
    def _save_figure(fig, filename_base):
        if not savefig:
            return
        if save_path is None:
            raise ValueError("save_path is required when savefig=True")
        os.makedirs(save_path, exist_ok=True)
        
        name = (save_prefix or '') + filename_base
        formats = save_format if isinstance(save_format, list) else [save_format]
        for fmt in formats:
            filepath = os.path.join(save_path, f"{name}.{fmt}")
            fig.savefig(filepath, dpi=save_dpi, bbox_inches='tight')
            print(f"  Saved: {filepath}")
    
    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    
    n_features = features.shape[1]
    
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    elif len(feature_names) < n_features:
        # Extend with generic names if not enough provided
        feature_names = list(feature_names) + [f'Feature {i+1}' for i in range(len(feature_names), n_features)]
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Find optimal k if not specified
    elbow_info = None
    if n_clusters is None:
        n_clusters, elbow_info = find_optimal_k(X_scaled, k_range)
        print(f"Auto-detected optimal k = {n_clusters}")
        
        # Plot elbow curve
        if show_plots or savefig:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(elbow_info['k_list'], elbow_info['inertias'], 'bo-', markersize=6)
            ax.axvline(x=n_clusters, color='r', linestyle='--', linewidth=1.5, 
                       label=f'Optimal k = {n_clusters}')
            ax.set_xlabel('Number of Clusters (k)', fontsize=10)
            ax.set_ylabel('Inertia (Within-cluster SSE)', fontsize=10)
            ax.set_title('Elbow Method for Optimal k', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            _save_figure(fig, 'markov_elbow_plot')
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Plot cluster scatter plots with cluster names
    # When more than 2 features, show Feature 1 vs each other feature
    if show_plots or savefig:
        from matplotlib.colors import BoundaryNorm
        
        # Count units per cluster
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        
        # Create discrete colormap for clusters
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        bounds = np.arange(-0.5, n_clusters + 0.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        
        # Determine number of scatter plots needed
        n_scatter_plots = max(1, n_features - 1)  # Feature 1 vs all others
        
        for plot_idx in range(n_scatter_plots):
            feat_x_idx = 0  # Always Feature 1 on x-axis
            feat_y_idx = plot_idx + 1 if n_features > 1 else 0
            
            fig, ax = plt.subplots(figsize=(5, 4))
            scatter = ax.scatter(features[:, feat_x_idx], features[:, feat_y_idx], 
                                c=cluster_labels, cmap=cmap, norm=norm, alpha=0.7, s=40, 
                                edgecolors='k', linewidth=0.5)
            ax.scatter(cluster_centers[:, feat_x_idx], cluster_centers[:, feat_y_idx], 
                      c='red', marker='X', s=150, edgecolors='k', linewidth=1.5, 
                      label='Centroids')
            
            # Annotate cluster centers with cluster name (C1, C2, ...)
            for i in range(n_clusters):
                cx, cy = cluster_centers[i, feat_x_idx], cluster_centers[i, feat_y_idx]
                ax.annotate(f'C{i+1}', (cx, cy), textcoords='offset points', 
                           xytext=(8, 8), fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            ax.set_xlabel(f'({feature_names[feat_x_idx]})', fontsize=10)
            ax.set_ylabel(f'({feature_names[feat_y_idx]})', fontsize=10)
            ax.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_clusters))
            cbar.ax.set_yticklabels([f'C{i+1}' for i in range(n_clusters)])
            cbar.set_label('Cluster', fontsize=9)
            plt.tight_layout()
            
            suffix = f'_f1_vs_f{feat_y_idx+1}' if n_features > 2 else ''
            _save_figure(fig, f'markov_cluster_scatter{suffix}')
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
    
    # Build transition matrix (always build 1st-order for plotting and fallback)
    transition_matrix, stationary_dist = build_transition_matrix(
        cluster_labels, unit_sequence_per_core, n_clusters
    )
    
    # Initialize model-specific parameters
    transition_dicts = {}  # For basic mode
    vom_transition_dicts = None  # For VOM mode
    fp_transition_dict = None  # For FP mode
    fp_transition_dicts_for_plot = {}  # For FP mode plotting (all orders)
    observed_starts = None  # For FP mode
    
    # Build higher-order transitions based on mc_model
    if mc_order > 1:
        if mc_model == 'BASIC':
            # Basic higher-order MC (no phantom history handling)
            for order in range(2, mc_order + 1):
                transition_dicts[order] = build_higher_order_transitions(
                    cluster_labels, unit_sequence_per_core, n_clusters, order
                )
                print(f"Built {order}-order basic MC with {len(transition_dicts[order])} state transitions")
        
        elif mc_model == 'VOM':
            # Variable-Order Markov Model
            vom_transition_dicts = build_vom_transitions(
                cluster_labels, unit_sequence_per_core, n_clusters, mc_order
            )
            print(f"Built VOM with max_order={mc_order}")
            for order in range(1, mc_order + 1):
                n_ngrams = len(vom_transition_dicts.get(order, {}))
                print(f"  Order {order}: {n_ngrams} observed {order}-grams")
        
        elif mc_model == 'FP':
            # Forbidden Path Model - build for all orders (for plotting)
            # but only the highest order is used for generation
            fp_transition_dict, observed_starts = build_fp_transitions(
                cluster_labels, unit_sequence_per_core, n_clusters, mc_order
            )
            # Also build for intermediate orders (for plotting purposes)
            fp_transition_dicts_for_plot = {}
            for order in range(2, mc_order + 1):
                fp_dict_order, _ = build_fp_transitions(
                    cluster_labels, unit_sequence_per_core, n_clusters, order
                )
                fp_transition_dicts_for_plot[order] = fp_dict_order
            print(f"Built Forbidden Path with order={mc_order}")
            for order in range(2, mc_order + 1):
                n_ngrams = len(fp_transition_dicts_for_plot.get(order, {}))
                print(f"  Order {order}: {n_ngrams} observed {order}-grams")
            print(f"  Observed starting sequences: {len(observed_starts)}")
    
    # Plot transition matrices for all orders
    if show_plots or savefig:
        # Plot 1st-order transition matrix
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(n_clusters):
            for j in range(n_clusters):
                val = transition_matrix[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=9, color=text_color)
        
        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        ax.set_xticklabels([f'C{i+1}' for i in range(n_clusters)])
        ax.set_yticklabels([f'C{i+1}' for i in range(n_clusters)])
        ax.set_xlabel('To Cluster', fontsize=10)
        ax.set_ylabel('From Cluster', fontsize=10)
        ax.set_title('1st-Order Transition Matrix', fontsize=11)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability', fontsize=9)
        plt.tight_layout()
        _save_figure(fig, 'markov_transition_matrix_order1')
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        # Plot higher-order transition matrices
        if mc_order > 1:
            for order in range(2, mc_order + 1):
                # Get the appropriate transition dict based on model type
                if mc_model == 'BASIC' and order in transition_dicts:
                    # Basic mode: show all possible histories (including phantom)
                    _plot_higher_order_transition_matrix(
                        transition_dicts[order], n_clusters, order,
                        show_plots, _save_figure, observed_only=False
                    )
                elif mc_model == 'VOM' and vom_transition_dicts and order in vom_transition_dicts:
                    # VOM mode: show only observed histories (no phantom rows)
                    vom_dict_for_plot = {}
                    for ngram, next_dict in vom_transition_dicts[order].items():
                        probs = np.zeros(n_clusters)
                        for next_state, prob in next_dict.items():
                            probs[next_state] = prob
                        vom_dict_for_plot[ngram] = probs
                    _plot_higher_order_transition_matrix(
                        vom_dict_for_plot, n_clusters, order,
                        show_plots, _save_figure, observed_only=True
                    )
                elif mc_model == 'FP' and order in fp_transition_dicts_for_plot:
                    # FP mode: show only observed histories (no phantom rows)
                    fp_dict_for_plot = {}
                    for ngram, next_dict in fp_transition_dicts_for_plot[order].items():
                        probs = np.zeros(n_clusters)
                        for next_state, prob in next_dict.items():
                            probs[next_state] = prob
                        fp_dict_for_plot[ngram] = probs
                    _plot_higher_order_transition_matrix(
                        fp_dict_for_plot, n_clusters, order,
                        show_plots, _save_figure, observed_only=True
                    )
    
    # For backward compatibility, transition_dict is the highest order dict (or None)
    transition_dict = transition_dicts.get(mc_order, None) if mc_model == 'BASIC' and mc_order > 1 else None
    
    markov_params = {
        'kmeans': kmeans,
        'scaler': scaler,
        'n_clusters': n_clusters,
        'mc_order': mc_order,
        'mc_model': mc_model,
        'cluster_labels': cluster_labels,
        'transition_matrix': transition_matrix,
        'transition_dict': transition_dict,  # For basic mode backward compat
        'transition_dicts': transition_dicts if mc_model == 'BASIC' and mc_order > 1 else None,
        'vom_transition_dicts': vom_transition_dicts,  # For VOM mode
        'fp_transition_dict': fp_transition_dict,  # For FP mode
        'observed_starts': observed_starts,  # For FP mode
        'stationary_dist': stationary_dist,
        'cluster_centers': cluster_centers,
        'elbow_info': elbow_info
    }
    
    return markov_params


# =============================================================================
# SEGMENT POOL FUNCTIONS
# =============================================================================

def load_segment_pool(core_names, log_data_csv, log_data_type, picked_datum, 
                     depth_column, alternative_column_names=None, boundary_category=None, 
                     neglect_topbottom=True, require_valid_sequence=True):
    """
    Load segment pool data from turbidite database.
    
    Parameters:
    - core_names: list of core names to process
    - log_data_csv: dict mapping core names to log file paths
    - log_data_type: list of log column names to load
    - picked_datum: dict mapping core names to picked depth file paths
    - depth_column: name of depth column
    - alternative_column_names: dict of alternative column names (optional)
    - boundary_category: category number for turbidite boundaries (default: None). 
                        If None, uses category 1 if available, otherwise uses the lowest available category
    - neglect_topbottom: if True, skip the first and last segments of each core (default: True)
    - require_valid_sequence: if True, only include segments with valid 1-2-3 category sequence (default: True)
    
    Returns:
    - seg_pool_metadata: dict containing loaded core data
    - seg_logs: list of turbidite log segments
    - seg_depths: list of turbidite depth segments
    """
    
    seg_pool_metadata = {}
    seg_logs = []
    seg_depths = []
    
    print("Loading segment pool from available cores...")
    
    for core_name in core_names:
        print(f"Processing {core_name}...")
        
        try:
            # First, verify all required log files exist before loading
            missing_logs = []
            for log_col in log_data_type:
                if log_col not in log_data_csv[core_name]:
                    missing_logs.append(log_col)
                else:
                    log_path = log_data_csv[core_name][log_col]
                    if not os.path.exists(log_path):
                        missing_logs.append(log_col)
            
            # Skip core if any required log type is missing
            if missing_logs:
                print(f"  Skipping {core_name}: Missing log types: {missing_logs}")
                continue
            
            # Load data for segment pool
            log_data, md_data = load_log_data(
                log_data_csv[core_name],
                log_columns=log_data_type,
                depth_column=depth_column,
                normalize=True
            )
            
            # Check if data was successfully loaded
            if len(md_data) == 0:
                print(f"  Skipping {core_name}: Failed to load log data")
                continue
            
            # Verify the loaded data has correct dimensions
            expected_dims = len(log_data_type)
            actual_dims = log_data.shape[1] if log_data.ndim > 1 else 1
            if actual_dims != expected_dims:
                print(f"  Skipping {core_name}: Expected {expected_dims} log columns, got {actual_dims}")
                continue
            
            # Store core data
            seg_pool_metadata[core_name] = {
                'log_data': log_data,
                'md_data': md_data,
                'available_columns': log_data_type
            }
            
            # Load turbidite boundaries for this core
            picked_file = picked_datum[core_name]
            try:
                picked_df = pd.read_csv(picked_file)
                
                # Determine boundary_category to use
                effective_category = boundary_category
                if effective_category is None:
                    # Get all available categories
                    available_categories = sorted(picked_df['category'].unique())
                    if len(available_categories) == 0:
                        raise ValueError(f"No categories found in picked datum file for {core_name}")
                    
                    # Use category 1 if available, otherwise use the lowest available category
                    if 1 in available_categories:
                        effective_category = 1
                    else:
                        effective_category = available_categories[0]
                    print(f"  Using boundary_category={effective_category} (auto-detected)")
                
                # Filter for specified category boundaries only
                category_depths = picked_df[picked_df['category'] == effective_category]['picked_depths_cm'].values
                
                if len(category_depths) == 0:
                    raise ValueError(f"No boundaries found for category {effective_category} in {core_name}")
                
                category_depths = np.sort(category_depths)  # Ensure sorted order
                
                # Build set of valid cat1 depths (those with valid 1-2-3 sequence) if required
                valid_cat1_depths = set()
                if require_valid_sequence:
                    # Get all depths and categories sorted by depth
                    all_depths = picked_df['picked_depths_cm'].values
                    all_categories = picked_df['category'].values
                    sorted_indices = np.argsort(all_depths)
                    sorted_depths = all_depths[sorted_indices]
                    sorted_categories = all_categories[sorted_indices]
                    
                    # Find valid 1-2-3 sequences
                    cat1_indices = np.where(sorted_categories == 1)[0]
                    used_cat2 = set()
                    used_cat3 = set()
                    
                    for cat1_idx in cat1_indices:
                        cat1_depth = sorted_depths[cat1_idx]
                        
                        # Find closest cat2 shallower than cat1 (smaller depth value)
                        cat2_candidates = np.where((sorted_categories == 2) & (sorted_depths < cat1_depth))[0]
                        cat2_candidates = np.array([idx for idx in cat2_candidates if idx not in used_cat2])
                        
                        if len(cat2_candidates) == 0:
                            continue
                        
                        # Get the closest cat2 (largest depth that's still < cat1_depth)
                        closest_cat2_idx = cat2_candidates[np.argmax(sorted_depths[cat2_candidates])]
                        cat2_depth = sorted_depths[closest_cat2_idx]
                        
                        # Find closest cat3 shallower than cat2
                        cat3_candidates = np.where((sorted_categories == 3) & (sorted_depths < cat2_depth))[0]
                        cat3_candidates = np.array([idx for idx in cat3_candidates if idx not in used_cat3])
                        
                        if len(cat3_candidates) == 0:
                            continue
                        
                        # Valid 1-2-3 sequence found
                        used_cat2.add(closest_cat2_idx)
                        used_cat3.add(cat3_candidates[np.argmax(sorted_depths[cat3_candidates])])
                        valid_cat1_depths.add(cat1_depth)
                    
                    print(f"  Found {len(valid_cat1_depths)} segments with valid 1-2-3 sequences")
                
                # Create turbidite segments (from boundary to boundary)
                # Determine range based on neglect_topbottom parameter
                if neglect_topbottom and len(category_depths) > 2:
                    # Skip first and last segments
                    start_range = 1
                    end_range = len(category_depths) - 2
                else:
                    # Include all segments
                    start_range = 0
                    end_range = len(category_depths) - 1
                
                segments_added = 0
                segments_skipped = 0
                
                for i in range(start_range, end_range):
                    start_depth = category_depths[i]
                    end_depth = category_depths[i + 1]
                    
                    # Check if this segment has valid 1-2-3 sequence (if required)
                    if require_valid_sequence:
                        # The segment's base is at end_depth (deeper), check if it's in valid set
                        if end_depth not in valid_cat1_depths:
                            segments_skipped += 1
                            continue
                    
                    # Find indices corresponding to these depths
                    start_idx = np.argmin(np.abs(md_data - start_depth))
                    end_idx = np.argmin(np.abs(md_data - end_depth))
                    
                    if end_idx > start_idx:
                        # Extract turbidite segment
                        turb_segment = log_data[start_idx:end_idx]
                        turb_depth = md_data[start_idx:end_idx] - md_data[start_idx]  # Relative depths
                        
                        seg_logs.append(turb_segment)
                        seg_depths.append(turb_depth)
                        segments_added += 1
                
                if require_valid_sequence:
                    print(f"  Added {segments_added} segments, skipped {segments_skipped} (no valid 1-2-3 sequence)")
                
            except Exception as e:
                print(f"Warning: Could not load turbidite boundaries for {core_name}: {e}")
            
            print(f"  Loaded: {len(md_data)} points, columns: {log_data_type}")
            
        except Exception as e:
            print(f"Error loading {core_name}: {e}")
    
    # Set target dimensions based on segment pool
    target_dimensions = seg_logs[0].shape[1] if len(seg_logs) > 0 and seg_logs[0].ndim > 1 else 1
    
    print(f"Segment pool created with {len(seg_logs)} turbidites")
    print(f"Total cores processed: {len(seg_pool_metadata)}")
    print(f"Target dimensions: {target_dimensions}")
    
    return seg_logs, seg_depths, seg_pool_metadata



def modify_segment_pool(segment_logs, segment_depths, remove_list=None):
    """
    Remove unwanted segments from the pool data and return the modified pool.
    
    Parameters:
    - segment_logs: list of log data arrays (segments)
    - segment_depths: list of depth arrays corresponding to each segment
    - remove_list: list of 1-based segment numbers to remove (optional)
                  If None or empty, no segments are removed
    
    Returns:
    - modified_segment_logs: list of remaining log data arrays
    - modified_segment_depths: list of remaining depth arrays
    """
    
    # If remove_list is None or empty, return original data
    if not remove_list:
        print("No segments to remove. Returning original pool data.")
        return segment_logs.copy(), segment_depths.copy()
    
    # Convert remove_list to 0-based indices and validate
    remove_indices = []
    for item in remove_list:
        try:
            # Convert to int (handle both string and int inputs)
            segment_num = int(item)
            if 1 <= segment_num <= len(segment_logs):
                remove_indices.append(segment_num - 1)  # Convert to 0-based
            else:
                print(f"Warning: Segment number {segment_num} is out of range (1-{len(segment_logs)}). Skipping.")
        except (ValueError, TypeError):
            print(f"Warning: Invalid segment number '{item}'. Skipping.")
    
    # Remove duplicates and sort
    remove_indices = sorted(set(remove_indices))
    
    if not remove_indices:
        print("No valid segments to remove. Returning original pool data.")
        return segment_logs.copy(), segment_depths.copy()
    
    # Create modified lists by excluding specified indices
    modified_segment_logs = []
    modified_segment_depths = []
    
    for i, (segment_log, segment_depth) in enumerate(zip(segment_logs, segment_depths)):
        if i not in remove_indices:
            modified_segment_logs.append(segment_log)
            modified_segment_depths.append(segment_depth)
    
    # Print summary of changes
    removed_segments_1based = [idx + 1 for idx in remove_indices]
    print(f"Removed segments: {removed_segments_1based}")
    print(f"Original pool size: {len(segment_logs)} segments")
    print(f"Modified pool size: {len(modified_segment_logs)} segments")
    
    return modified_segment_logs, modified_segment_depths

def create_synthetic_log(segment_logs, segment_depths, max_thickness=None, max_num_units=None,
                         exclude_inds=None, repetition=False, method='random', markov_params=None, 
                         segment_features=None, random_seed=None):
    """Create synthetic log using turbidite database approach with picked depths at turbidite bases.
    
    Parameters
    ----------
    segment_logs : list
        List of turbidite log segments (numpy arrays)
    segment_depths : list
        List of corresponding depth arrays
    max_thickness : float or None, default=None
        Maximum thickness for the synthetic log. If None and max_num_units is provided,
        thickness is determined by stacking units until max_num_units is reached.
        If both are None, defaults to max_num_units=10.
    max_num_units : int or None, default=None
        Maximum number of units to stack. If None and max_thickness is provided,
        stacking continues until max_thickness is reached.
        If both are None, defaults to 10.
    exclude_inds : list or None, default=None
        Indices to exclude from selection
    repetition : bool, default=False
        If True, allow reusing turbidite segments; if False, each segment can only be used once
    method : str, default='random'
        Segment selection method:
        - 'random': Random selection (original behavior)
        - 'MarkovChain': Markov Chain-based selection using trained cluster transitions.
          Supports both 1st-order and higher-order Markov chains based on markov_params.
    markov_params : dict or None, default=None
        Dictionary from train_markov_model() containing trained model parameters.
        Required when method='MarkovChain'. Contains:
        - 'kmeans', 'scaler', 'n_clusters': clustering model
        - 'transition_matrix', 'stationary_dist': 1st-order MC parameters
        - 'mc_order': Markov chain order (1 for 1st-order, >1 for higher-order)
        - 'transition_dict': higher-order transition dictionary (for mc_order > 1)
    segment_features : array-like or None, default=None
        Feature array of shape (n_segments, n_features) for cluster assignment.
        Required when method='MarkovChain'. Should contain the same feature types used for training.
    random_seed : int or None, default=None
        Random seed for reproducibility. If None, results vary between runs.
    
    Returns
    -------
    log : ndarray
        Synthetic log data array
    d : ndarray
        Depth values array
    valid_picked_depths : list
        List of boundary depth values (just depths, no category info)
    inds : list
        List of indices of segments used from the pool
    
    Example
    -------
    >>> # Random method with max_thickness (default, backward compatible)
    >>> syn_log, syn_md, syn_depths, inds = create_synthetic_log(
    ...     segment_logs=seg_logs,
    ...     segment_depths=seg_depths,
    ...     max_thickness=400
    ... )
    >>> 
    >>> # Random method with max_num_units only
    >>> syn_log, syn_md, syn_depths, inds = create_synthetic_log(
    ...     segment_logs=seg_logs,
    ...     segment_depths=seg_depths,
    ...     max_num_units=15
    ... )
    >>> 
    >>> # MarkovChain method with both constraints
    >>> markov_params = train_markov_model(features, unit_seq)
    >>> syn_log, syn_md, syn_depths, inds = create_synthetic_log(
    ...     segment_logs=seg_logs,
    ...     segment_depths=seg_depths,
    ...     max_thickness=400,
    ...     max_num_units=10,
    ...     method='MarkovChain',
    ...     markov_params=markov_params,
    ...     segment_features=segment_features
    ... )
    """
    # Handle default case: if both are None, default to max_num_units=10
    if max_thickness is None and max_num_units is None:
        max_num_units = 10
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Validate method
    method = method.lower()
    if method not in ['random', 'markovchain']:
        raise ValueError(f"method must be 'random' or 'MarkovChain', got '{method}'")
    
    # Validate MarkovChain requirements
    if method == 'markovchain':
        if markov_params is None:
            raise ValueError("markov_params is required when method='MarkovChain'. Use train_markov_model() first.")
        if segment_features is None:
            raise ValueError("segment_features is required when method='MarkovChain'. Provide feature array for segments.")
        
        # Assign clusters to segments using trained model
        segment_features = np.asarray(segment_features)
        if segment_features.ndim == 1:
            segment_features = segment_features.reshape(-1, 1)
        
        scaler = markov_params['scaler']
        kmeans = markov_params['kmeans']
        segment_features_scaled = scaler.transform(segment_features)
        segment_clusters = kmeans.predict(segment_features_scaled)
        
        # Get transition parameters
        transition_matrix = markov_params['transition_matrix']
        stationary_dist = markov_params['stationary_dist']
        n_clusters = markov_params['n_clusters']
        
        # Get MC model parameters
        mc_order = markov_params.get('mc_order', 1)
        mc_model = markov_params.get('mc_model', 'BASIC').upper()
        
        # Get model-specific transition data
        transition_dict = markov_params.get('transition_dict', None)  # For basic mode
        vom_transition_dicts = markov_params.get('vom_transition_dicts', None)  # For VOM
        fp_transition_dict = markov_params.get('fp_transition_dict', None)  # For FP
        observed_starts = markov_params.get('observed_starts', None)  # For FP
        
        # Build available segments by cluster
        available_by_cluster = {i: [] for i in range(n_clusters)}
        for seg_idx, cluster in enumerate(segment_clusters):
            if exclude_inds is None or seg_idx not in exclude_inds:
                available_by_cluster[cluster].append(seg_idx)
        
        # Track used segments for MarkovChain
        used_segments = set()
        
        # Track cluster history for higher-order MC
        cluster_history = []
        
        # Sample initial cluster(s) based on model type
        if mc_order > 1 and mc_model == 'FP' and observed_starts and len(observed_starts) > 0:
            # FP: Initialize with an observed starting sequence
            start_seq = list(observed_starts[np.random.randint(len(observed_starts))])
            cluster_history = start_seq.copy()
            current_cluster = cluster_history[-1]
        elif mc_order > 1:
            # VOM or basic: Initialize with stationary distribution
            for _ in range(mc_order):
                cluster_history.append(np.random.choice(n_clusters, p=stationary_dist))
            current_cluster = cluster_history[-1]
        else:
            current_cluster = np.random.choice(n_clusters, p=stationary_dist)
    
    # Determine target dimensions from the first available segment
    target_dimensions = segment_logs[0].shape[1] if len(segment_logs) > 0 and segment_logs[0].ndim > 1 else 1
    
    fake_log = np.array([]).reshape(0, target_dimensions) if target_dimensions > 1 else np.array([])
    md_log = np.array([])
    max_depth = 0
    inds = []
    picked_depths = []
    
    # Initialize available indices for random selection
    if method == 'random':
        if repetition:
            available_inds = list(range(len(segment_logs)))
        else:
            available_inds = list(range(len(segment_logs)))
            if exclude_inds is not None:
                available_inds = [ind for ind in available_inds if ind not in exclude_inds]
    
    # Add initial boundary
    picked_depths.append((0, 1))
    
    # Track number of units added
    num_units = 0
    
    # Determine loop continuation condition
    def should_continue():
        # Check unit limit if specified
        if max_num_units is not None and num_units >= max_num_units:
            return False
        # Check thickness limit if specified
        if max_thickness is not None and max_depth > max_thickness:
            return False
        # If only max_num_units specified (no thickness limit), continue until unit limit
        if max_thickness is None:
            return True
        # Otherwise continue until thickness is reached
        return max_depth <= max_thickness
    
    while should_continue():
        if method == 'random':
            # ===== RANDOM SELECTION =====
            if not repetition and len(available_inds) == 0:
                print("Warning: No more unique turbidite segments available. Stopping log generation.")
                break
                
            if repetition:
                potential_inds = [ind for ind in range(len(segment_logs)) if exclude_inds is None or ind not in exclude_inds]
                if not potential_inds:
                    print("Warning: No available turbidite segments after exclusions. Stopping log generation.")
                    break
                ind = random.choices(potential_inds, k=1)[0]
            else:
                ind = random.choices(available_inds, k=1)[0]
                available_inds.remove(ind)
        
        else:
            # ===== MARKOV CHAIN SELECTION =====
            # Find available segments in current cluster
            available = [s for s in available_by_cluster[current_cluster] 
                         if repetition or s not in used_segments]
            
            # If no segments in current cluster, try other clusters
            if len(available) == 0:
                found = False
                for alt_cluster in range(n_clusters):
                    alt_available = [s for s in available_by_cluster[alt_cluster] 
                                     if repetition or s not in used_segments]
                    if len(alt_available) > 0:
                        current_cluster = alt_cluster
                        available = alt_available
                        found = True
                        break
                if not found:
                    print("Warning: No more segments available for MarkovChain. Stopping log generation.")
                    break
            
            # Select random segment from current cluster
            ind = random.choice(available)
            used_segments.add(ind)
            
            # Track cluster in history
            cluster_history.append(current_cluster)
            
            # Transition to next cluster based on MC model and order
            if mc_order > 1:
                if mc_model == 'VOM' and vom_transition_dicts is not None:
                    # VOM: Use longest matching context
                    transition_probs, _ = _get_vom_transition_probs(
                        cluster_history, vom_transition_dicts, stationary_dist, n_clusters, mc_order
                    )
                elif mc_model == 'FP' and fp_transition_dict is not None:
                    # FP: Use forbidden path transitions
                    transition_probs, is_valid = _get_fp_transition_probs(
                        cluster_history, fp_transition_dict, stationary_dist, n_clusters, mc_order
                    )
                    if not is_valid:
                        # Forbidden path - fall back to stationary or try to find valid path
                        transition_probs = stationary_dist
                elif mc_model == 'BASIC' and transition_dict is not None:
                    # Basic higher-order MC
                    transition_probs = _get_higher_order_transition_probs(
                        cluster_history, transition_dict, transition_matrix, stationary_dist, mc_order
                    )
                else:
                    # Fallback to 1st-order
                    transition_probs = transition_matrix[current_cluster]
            else:
                # 1st-order MC: use transition matrix
                transition_probs = transition_matrix[current_cluster]
            
            current_cluster = np.random.choice(n_clusters, p=transition_probs)
            
        inds.append(ind)
        
        # Get turbidite segment from database
        turb_segment = segment_logs[ind]
        turb_depths = segment_depths[ind]
        
        # Ensure turbidite has proper dimensions
        if turb_segment.ndim == 1:
            turb_segment = turb_segment.reshape(-1, 1)
        
        # Ensure proper dimensions match target
        if turb_segment.shape[1] < target_dimensions:
            # Pad with noise if needed
            padding = np.random.normal(0, 0.1, (len(turb_segment), target_dimensions - turb_segment.shape[1]))
            turb_segment = np.hstack([turb_segment, padding])
        elif turb_segment.shape[1] > target_dimensions:
            # Truncate if needed
            turb_segment = turb_segment[:, :target_dimensions]
        
        # Append log data
        if target_dimensions > 1:
            if len(fake_log) == 0:
                fake_log = turb_segment.copy()
            else:
                fake_log = np.vstack((fake_log, turb_segment))
        else:
            fake_log = np.hstack((fake_log, turb_segment.flatten()))
        
        # Append depth data
        if len(md_log) == 0:
            md_log = np.hstack((md_log, 1 + turb_depths))
        else:
            md_log = np.hstack((md_log, 1 + md_log[-1] + turb_depths))
            
        max_depth = md_log[-1]
        num_units += 1
        
        # Add picked depth at the base of this turbidite (current max_depth)
        # Only add if within thickness limit (if specified) or always add if no thickness limit
        if max_thickness is None or max_depth <= max_thickness:
            picked_depths.append((max_depth, 1))
    
    # Truncate to target thickness if specified
    if max_thickness is not None:
        valid_indices = md_log <= max_thickness
        if target_dimensions > 1:
            log = fake_log[valid_indices]
        else:
            log = fake_log[valid_indices]
        d = md_log[valid_indices]
        
        # Filter picked depths to only include those within the valid range
        valid_picked_depths = [depth for depth, category in picked_depths if depth <= max_thickness]
    else:
        # No thickness limit - use all data
        log = fake_log
        d = md_log
        valid_picked_depths = [depth for depth, category in picked_depths]
    
    # Ensure we have an end boundary
    if len(valid_picked_depths) == 0 or (len(d) > 0 and valid_picked_depths[-1] != d[-1]):
        if len(d) > 0:
            valid_picked_depths.append(d[-1])
    
    return log, d, valid_picked_depths, inds



def generate_constraint_subsets(n_constraints):
    """Generate all possible subsets of constraints (2^n combinations)"""
    all_subsets = []
    for r in range(n_constraints + 1):  # 0 to n_constraints
        for subset in combinations(range(n_constraints), r):
            all_subsets.append(list(subset))
    return all_subsets


def _process_single_parameter_combination(
    idx, params, 
    log_a, log_b, md_a, md_b,
    picked_datum_a, picked_datum_b,
    datum_ages_a, datum_ages_b,
    core_a_age_data, core_b_age_data,
    target_quality_indices,
    output_csv_filenames,
    synthetic_csv_filenames,
    pca_for_dependent_dtw,
    test_age_constraint_removal,
    output_metric_only=True
):
    """Process a single parameter combination with parallel computation inside."""
    
    # Import here to avoid circular imports in workers
    from .dtw_core import run_comprehensive_dtw_analysis
    from .path_finding import find_complete_core_paths
    from ..utils.plotting import plot_correlation_distribution
    
    # Generate a random suffix for temporary files in this iteration
    random_suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=8))

    # Initialize temp_mapping_file in Downloads folder to avoid cloud sync overhead
    temp_dir = os.path.expanduser('~/Downloads')
    temp_mapping_file = os.path.join(temp_dir, f'temp_mappings_{random_suffix}.pkl')

    # Extract parameters
    age_consideration = params['age_consideration']
    restricted_age_correlation = params['restricted_age_correlation']
    shortest_path_search = params['shortest_path_search']
    
    # Generate parameter labels
    if age_consideration:
        if restricted_age_correlation:
            age_label = 'restricted_age'
        else:
            age_label = 'loose_age'
    else:
        age_label = 'no_age'
    
    search_label = 'optimal' if shortest_path_search else 'random'
    combo_id = f"{age_label}_{search_label}"
    
    # Cache for synthetic CSV DataFrames to avoid redundant reads
    synthetic_df_cache = {}
    
    # Check if this scenario exists in synthetic CSV files (if provided)
    if synthetic_csv_filenames:
        scenario_exists = False
        for quality_index in target_quality_indices:
            if quality_index in synthetic_csv_filenames:
                synthetic_csv_file = synthetic_csv_filenames[quality_index]
                if os.path.exists(synthetic_csv_file):
                    try:
                        synthetic_df = pd.read_csv(synthetic_csv_file)
                        # Cache the DataFrame for later use
                        synthetic_df_cache[quality_index] = synthetic_df
                        # Check if this combination_id exists in the synthetic CSV
                        if 'combination_id' in synthetic_df.columns:
                            if combo_id in synthetic_df['combination_id'].values:
                                scenario_exists = True
                                break
                    except Exception:
                        pass
        
        # If synthetic CSV exists but this scenario is not in it, skip processing
        if not scenario_exists:
            return True, combo_id, {}  # Return success with empty results to skip without error
    
    try:
        # Validate input age data when age consideration is enabled
        if age_consideration:
            # Check datum_ages_a and datum_ages_b
            if datum_ages_a is None or datum_ages_b is None:
                return False, combo_id, "datum_ages_a or datum_ages_b is None when age_consideration=True"
            
            # Check required keys in datum_ages
            required_age_keys = ['depths', 'ages', 'pos_uncertainties', 'neg_uncertainties']
            for key in required_age_keys:
                if key not in datum_ages_a or not datum_ages_a[key]:
                    return False, combo_id, f"Missing or empty key '{key}' in datum_ages_a"
                if key not in datum_ages_b or not datum_ages_b[key]:
                    return False, combo_id, f"Missing or empty key '{key}' in datum_ages_b"
            
            # Check age_data constraints
            required_constraint_keys = ['in_sequence_ages', 'in_sequence_depths', 'in_sequence_pos_errors', 'in_sequence_neg_errors']
            for key in required_constraint_keys:
                if key not in core_a_age_data:
                    return False, combo_id, f"Missing key '{key}' in core_a_age_data"
                if key not in core_b_age_data:
                    return False, combo_id, f"Missing key '{key}' in core_b_age_data"
                    
                # Check for all NaN values in constraint data
                try:
                    if core_a_age_data[key] and len(core_a_age_data[key]) > 0:
                        valid_values_a = [val for val in core_a_age_data[key] if not (np.isnan(val) if isinstance(val, (int, float)) else False)]
                        if len(valid_values_a) == 0:
                            return False, combo_id, f"All values are NaN in core_a_age_data['{key}']"
                    
                    if core_b_age_data[key] and len(core_b_age_data[key]) > 0:
                        valid_values_b = [val for val in core_b_age_data[key] if not (np.isnan(val) if isinstance(val, (int, float)) else False)]
                        if len(valid_values_b) == 0:
                            return False, combo_id, f"All values are NaN in core_b_age_data['{key}']"
                except Exception as e:
                    return False, combo_id, f"Error validating age_data['{key}']: {str(e)}"
        
        # Run comprehensive DTW analysis with original constraints
        dtw_result = run_comprehensive_dtw_analysis(
            log_a, log_b, md_a, md_b,
            picked_datum_a=picked_datum_a,
            picked_datum_b=picked_datum_b,
            independent_dtw=False,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            top_bottom=True,
            top_depth=0.0,
            exclude_deadend=True,
            mute_mode=True,
            age_consideration=age_consideration,
            datum_ages_a=datum_ages_a if age_consideration else None,
            datum_ages_b=datum_ages_b if age_consideration else None,
            restricted_age_correlation=restricted_age_correlation if age_consideration else False,
            core_a_age_data=core_a_age_data if age_consideration else None,
            core_b_age_data=core_b_age_data if age_consideration else None,
            n_jobs=-1  # Use all available cores
        )
        
        # Check if DTW analysis returned None
        if dtw_result is None:
            return False, combo_id, "run_comprehensive_dtw_analysis returned None"
        
        # Validate DTW results before proceeding (check required keys exist)
        required_keys = ['dtw_correlation', 'valid_dtw_pairs', 'segments_a', 'segments_b', 
                        'depth_boundaries_a', 'depth_boundaries_b', 'dtw_distance_matrix_full']
        if not all(key in dtw_result for key in required_keys):
            return False, combo_id, "DTW analysis returned incomplete dictionary"
        
        if any(dtw_result[key] is None for key in required_keys):
            return False, combo_id, "DTW analysis returned None values"
        
        # Find complete core paths
        if shortest_path_search:
            _ = find_complete_core_paths(
                dtw_result, log_a, log_b,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=-1,  # Use all available cores
                shortest_path_search=True, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw,
                output_metric_only=output_metric_only
            )
        else:
            _ = find_complete_core_paths(
                dtw_result, log_a, log_b,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=-1,  # Use all available cores
                shortest_path_search=False, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw,
                output_metric_only=output_metric_only
            )
        
        # Process quality indices and collect results
        results = {}
        for quality_index in target_quality_indices:
            
            # Extract bin size information from cached synthetic CSV if available
            targeted_binsize = None
            if quality_index in synthetic_df_cache:
                try:
                    synthetic_df = synthetic_df_cache[quality_index]
                    if not synthetic_df.empty and 'bins' in synthetic_df.columns:
                        # Parse the first row's bins to get the bin structure
                        bins_str = synthetic_df.iloc[0]['bins']
                        if pd.notna(bins_str):
                            synthetic_bins = np.fromstring(bins_str.strip('[]'), sep=' ')
                            bin_width_synthetic = np.mean(np.diff(synthetic_bins))
                            targeted_binsize = (synthetic_bins, bin_width_synthetic)
                except Exception:
                    pass  # Use default binning if extraction fails
            
            fit_params = plot_correlation_distribution(
                mapping_csv=f'{temp_mapping_file}',
                quality_index=quality_index,
                save_png=False, pdf_method='normal',
                kde_bandwidth=0.05, mute_mode=True, targeted_binsize=targeted_binsize
            )
            
            if fit_params is not None:
                fit_params_copy = fit_params.copy()
                
                # Remove kde_object as it can't be serialized to CSV
                fit_params_copy.pop('kde_object', None)
                
                # Add metadata fields
                fit_params_copy['combination_id'] = combo_id
                fit_params_copy['age_consideration'] = age_consideration
                fit_params_copy['restricted_age_correlation'] = restricted_age_correlation
                fit_params_copy['shortest_path_search'] = shortest_path_search
                
                # Add constraint tracking columns
                fit_params_copy['core_a_constraints_count'] = len(core_a_age_data['in_sequence_ages']) if age_consideration else 0
                fit_params_copy['core_b_constraints_count'] = len(core_b_age_data['in_sequence_ages']) if age_consideration else 0
                fit_params_copy['constraint_scenario_description'] = 'all_original_constraints_remained' if age_consideration else 'no_age_constraints_used'
                
                results[quality_index] = fit_params_copy
        
        # Clean up
        if os.path.exists(f'{temp_mapping_file}'):
            os.remove(f'{temp_mapping_file}')
        
        del dtw_result
        gc.collect()
        
        return True, combo_id, results
        
    except Exception as e:
        if os.path.exists(f'{temp_mapping_file}'):
            os.remove(f'{temp_mapping_file}')
        gc.collect()
        return False, combo_id, str(e)


def _process_single_constraint_scenario(
    param_idx, params, constraint_subset, in_sequence_indices,
    log_a, log_b, md_a, md_b,
    picked_datum_a, picked_datum_b,
    datum_ages_a, datum_ages_b,
    core_a_age_data, core_b_age_data,
    uncertainty_method,
    target_quality_indices,
    cached_binsizes,
    pca_for_dependent_dtw,
    output_metric_only=True
):
    """Process a single constraint scenario (exact copy of original loop body)"""
    
    # Import here to avoid circular imports in workers
    from .dtw_core import run_comprehensive_dtw_analysis
    from .path_finding import find_complete_core_paths
    from ..utils.plotting import plot_correlation_distribution
    from .age_models import calculate_interpolated_ages
    
    # Generate a process-safe random suffix for temporary files using numpy
    # This avoids conflicts between parallel workers
    import os
    process_id = os.getpid()
    np.random.seed(None)  # Use current time as seed (non-deterministic)
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    random_suffix = ''.join(np.random.choice(list(chars), size=8))
    
    # Initialize temp_mapping_file in Downloads folder to avoid cloud sync overhead
    temp_dir = os.path.expanduser('~/Downloads')
    temp_mapping_file = os.path.join(temp_dir, f'temp_mappings_{random_suffix}.pkl')
    
    # Reset and reload parameters correctly for each iteration
    age_consideration = params['age_consideration']
    restricted_age_correlation = params['restricted_age_correlation']
    shortest_path_search = params['shortest_path_search']
    
    # Generate parameter labels
    if restricted_age_correlation:
        age_label = 'restricted_age'
    else:
        age_label = 'loose_age'
    
    search_label = 'optimal' if shortest_path_search else 'random'
    combo_id = f"{age_label}_{search_label}"
    
    try:
        # Map subset indices to original constraint indices (in-sequence only)
        # No NaN filtering - user confirmed the data_columns fields are valid
        original_indices = []
        for i in constraint_subset:
            idx = in_sequence_indices[i]
            original_indices.append(idx)
        
        # Check if we have any constraints (should always be true)
        if len(original_indices) == 0:
            return False, f"{combo_id}_no_constraints", "No constraints in subset"
        
        # Create modified core_b_age_data using original indices
        # Convert pandas Series to lists and ensure proper data types
        age_data_b_current = {
            'depths': [float(core_b_age_data['depths'].iloc[i]) if hasattr(core_b_age_data['depths'], 'iloc') else float(core_b_age_data['depths'][i]) for i in original_indices],
            'ages': [float(core_b_age_data['ages'][i]) for i in original_indices],
            'pos_errors': [float(core_b_age_data['pos_errors'][i]) for i in original_indices],
            'neg_errors': [float(core_b_age_data['neg_errors'][i]) for i in original_indices],
            'in_sequence_flags': [core_b_age_data['in_sequence_flags'][i] for i in original_indices],
            'in_sequence_depths': [float(core_b_age_data['depths'].iloc[i]) if hasattr(core_b_age_data['depths'], 'iloc') else float(core_b_age_data['depths'][i]) for i in original_indices],
            'in_sequence_ages': [float(core_b_age_data['ages'][i]) for i in original_indices],
            'in_sequence_pos_errors': [float(core_b_age_data['pos_errors'][i]) for i in original_indices],
            'in_sequence_neg_errors': [float(core_b_age_data['neg_errors'][i]) for i in original_indices],
            'core': [core_b_age_data['core'][i] for i in original_indices]
        }
        
        # Recalculate interpolated ages for core B with reduced constraints
        datum_ages_b_current = calculate_interpolated_ages(
            picked_datum=picked_datum_b,
            age_constraints_depths=np.array(age_data_b_current['depths']),
            age_constraints_ages=np.array(age_data_b_current['ages']),
            age_constraints_pos_errors=np.array(age_data_b_current['pos_errors']),
            age_constraints_neg_errors=np.array(age_data_b_current['neg_errors']),
            age_constraints_in_sequence_flags=age_data_b_current['in_sequence_flags'],
            age_constraint_source_core=age_data_b_current['core'],
            top_bottom=True,
            top_depth=0.0,
            bottom_depth=md_b[-1],
            top_age=0,
            top_age_pos_error=75,
            top_age_neg_error=75,
            uncertainty_method=uncertainty_method,
            n_monte_carlo=10000,
            show_plot=False,
            export_csv=False,
            mute_mode=True
        )
        
        # Validate the interpolated ages result
        if datum_ages_b_current is None:
            return False, f"{combo_id}_invalid_ages", "calculate_interpolated_ages returned None"
        
        # Check if interpolated ages contain valid values
        required_keys = ['depths', 'ages', 'pos_uncertainties', 'neg_uncertainties']
        for key in required_keys:
            if key not in datum_ages_b_current or not datum_ages_b_current[key]:
                return False, f"{combo_id}_missing_age_key", f"Missing or empty key '{key}' in interpolated ages"
            
            # Check for all NaN values
            if all(np.isnan(val) for val in datum_ages_b_current[key]):
                return False, f"{combo_id}_all_nan_ages", f"All values are NaN in interpolated ages key '{key}'"
        
        # Run DTW analysis with reduced constraints
        dtw_result = run_comprehensive_dtw_analysis(
            log_a, log_b, md_a, md_b,
            picked_datum_a=picked_datum_a,
            picked_datum_b=picked_datum_b,
            independent_dtw=False,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            top_bottom=True,
            top_depth=0.0,
            exclude_deadend=True,
            mute_mode=True,
            age_consideration=age_consideration,
            datum_ages_a=datum_ages_a,  # Use original ages for core A
            datum_ages_b=datum_ages_b_current,  # Use modified ages for core B
            restricted_age_correlation=restricted_age_correlation,
            core_a_age_data=core_a_age_data,  # Original age constraint data for core A
            core_b_age_data=age_data_b_current,  # Modified age constraint data for core B
            n_jobs=-1  # Use all available cores
        )
        
        # Check if DTW analysis returned None
        if dtw_result is None:
            return False, f"{combo_id}_dtw_none", "run_comprehensive_dtw_analysis returned None"
        
        # Validate DTW results before proceeding (check required keys exist)
        required_keys = ['dtw_correlation', 'valid_dtw_pairs', 'segments_a', 'segments_b', 
                        'depth_boundaries_a', 'depth_boundaries_b', 'dtw_distance_matrix_full']
        if not all(key in dtw_result for key in required_keys):
            return False, f"{combo_id}_dtw_incomplete", "DTW analysis returned incomplete dictionary"
        
        if any(dtw_result[key] is None for key in required_keys):
            return False, f"{combo_id}_dtw_none", "DTW analysis returned None values"
        
        # Find paths with correct parameters
        if shortest_path_search:
            _ = find_complete_core_paths(
                dtw_result, log_a, log_b,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=-1,  # Use all available cores
                shortest_path_search=True, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw,
                output_metric_only=output_metric_only
            )
        else:
            _ = find_complete_core_paths(
                dtw_result, log_a, log_b,
                output_csv=temp_mapping_file,
                start_from_top_only=True, batch_size=1000, n_jobs=-1,  # Use all available cores
                shortest_path_search=False, shortest_path_level=2,
                max_search_path=100000, mute_mode=True, pca_for_dependent_dtw=pca_for_dependent_dtw,
                output_metric_only=output_metric_only
            )
        
        # Process quality indices
        results = {}
        for quality_index in target_quality_indices:
            
            # Use pre-cached bin size information if available
            targeted_binsize = cached_binsizes.get(quality_index, None)
            
            fit_params = plot_correlation_distribution(
                mapping_csv=f'{temp_mapping_file}',
                quality_index=quality_index,
                save_png=False, pdf_method='normal',
                kde_bandwidth=0.05, mute_mode=True, targeted_binsize=targeted_binsize
            )
            
            if fit_params is not None:
                fit_params_copy = fit_params.copy()
                
                # Remove kde_object as it can't be serialized to CSV
                fit_params_copy.pop('kde_object', None)
                
                # Convert 0-based indices to 1-based for constraint description
                remaining_indices_1based = [i + 1 for i in sorted(constraint_subset)]
                
                # Add metadata fields
                fit_params_copy['combination_id'] = combo_id
                fit_params_copy['age_consideration'] = age_consideration
                fit_params_copy['restricted_age_correlation'] = restricted_age_correlation
                fit_params_copy['shortest_path_search'] = shortest_path_search
                
                # Add constraint tracking with correct counts
                fit_params_copy['core_a_constraints_count'] = len(core_a_age_data['in_sequence_ages'])  # Original count for core A
                fit_params_copy['core_b_constraints_count'] = len(constraint_subset)  # Modified count for core B
                fit_params_copy['constraint_scenario_description'] = f'constraints_{remaining_indices_1based}_remained'
                
                results[quality_index] = fit_params_copy
        
        # Clean up temporary files
        if os.path.exists(f'{temp_mapping_file}'):
            os.remove(f'{temp_mapping_file}')
        
        del dtw_result
        del age_data_b_current, datum_ages_b_current
        gc.collect()
        
        scenario_id = f"{combo_id}_subset_{len(constraint_subset)}"
        return True, scenario_id, results
        
    except Exception as e:
        if os.path.exists(f'{temp_mapping_file}'):
            os.remove(f'{temp_mapping_file}')
        # Clean up variables in case of error
        if 'age_data_b_current' in locals():
            del age_data_b_current
        if 'datum_ages_b_current' in locals():
            del datum_ages_b_current
        gc.collect()
        return False, f"{combo_id}_subset_error", str(e)


def run_multi_parameter_analysis(
    # Core data inputs
    log_a, log_b, md_a, md_b,
    picked_datum_a, picked_datum_b,
    datum_ages_a, datum_ages_b,
    core_a_age_data, core_b_age_data,
    uncertainty_method,
    
    # Core identifiers
    core_a_name, 
    core_b_name,
    
    # Output configuration
    output_csv_directory,  # Directory path for output CSV files
    
    # Analysis parameters
    parameter_combinations,
    target_quality_indices=['corr_coef', 'norm_dtw'],
    log_columns=None,  # List of log column names (e.g., ['hiresMS', 'CT', 'Lumin'])
    test_age_constraint_removal = True,

    # Optional parameters
    synthetic_csv_filenames=None,  # Dict with quality_index as key and synthetic CSV filename as value
    pca_for_dependent_dtw=False,
    max_search_per_layer=None,  # Max scenarios per constraint removal layer
    output_metric_only=True  # Only output quality metrics, skip full path info for faster processing
):
    """
    Run comprehensive multi-parameter analysis for core correlation.
    
    Brief summary: This function performs DTW analysis across multiple parameter combinations
    and optionally tests age constraint removal scenarios, generating distribution fit parameters
    for various quality indices.
    
    Parameters:
    -----------
    log_a, log_b : array-like
        Log data for cores A and B
    md_a, md_b : array-like
        Measured depth arrays for cores A and B
    picked_datum_a, picked_datum_b : array-like
        Picked depths of category 1 for cores A and B
    datum_ages_a, datum_ages_b : dict
        Age interpolation results for picked depths
    core_a_age_data, core_b_age_data : dict
        Age constraint data for cores A and B
    uncertainty_method : str
        Method for uncertainty calculation
    core_a_name, core_b_name : str
        Names of cores A and B
    output_csv_directory : str
        Directory path where output CSV files will be saved
    parameter_combinations : list of dict
        List of parameter combinations to test
    target_quality_indices : list, default=['corr_coef', 'norm_dtw']
        Quality indices to analyze (e.g., ['corr_coef', 'norm_dtw', 'perc_diag'])
    log_columns : list or None, default=None
        List of log column names (e.g., ['hiresMS', 'CT', 'Lumin']).
        If provided, will be used in the output directory structure.
    test_age_constraint_removal : bool (default=True)
        Whether to test age constraint removal scenarios
    synthetic_csv_filenames : dict or None, default=None
        Dictionary mapping quality_index to synthetic CSV filename for consistent bin sizing
    pca_for_dependent_dtw : bool
        Whether to use PCA for dependent DTW
    max_search_per_layer : int or None, default=None
        Maximum number of scenarios to process per constraint removal layer.
        If None, processes all scenarios. A layer represents combinations with
        the same number of remaining age constraints.
    output_metric_only : bool, default=True
        If True, only output quality metrics in path finding results, skip storing
        full path information (path sequences and warping paths). This significantly
        reduces memory usage and speeds up processing. Set to False only if you need
        the detailed path information for downstream analysis.
    
    Returns:
    --------
    None
        Results are saved to CSV files in output_csv_directory
    """
    
    # Generate output CSV filenames based on directory and log_columns
    if log_columns is not None and len(log_columns) > 0:
        # If log_columns provided, use subdirectory structure
        log_cols_str = "_".join(log_columns)
        full_output_dir = os.path.join(output_csv_directory, log_cols_str)
    else:
        # If no log_columns, use the directory as is
        full_output_dir = output_csv_directory
    
    # Create output CSV filenames dictionary
    output_csv_filenames = {}
    for quality_index in target_quality_indices:
        output_csv_filenames[quality_index] = os.path.join(
            full_output_dir, 
            f'{quality_index}_fit_params.csv'
        )
    
    # Create directories for output files if needed
    os.makedirs(full_output_dir, exist_ok=True)
    
    # VALIDATE AGE DATA BEFORE PROCESSING
    print("Validating age data for age-based analysis...")
    
    # Check if age data contains valid age values
    def validate_age_data(age_data, core_name):
        """Validate age data for a given core"""
        if age_data is None:
            return False, f"{core_name}: age_data is None"
        
        if 'ages' not in age_data or not age_data['ages']:
            return False, f"{core_name}: no 'ages' found in age_data"
        
        # Check if all ages are NaN or invalid
        valid_ages = [age for age in age_data['ages'] if isinstance(age, (int, float)) and not np.isnan(age)]
        if len(valid_ages) == 0:
            return False, f"{core_name}: all age values in age_data['ages'] are NaN or invalid"
        
        return True, f"{core_name}: age_data validation passed"
    
    def validate_pickeddepth_ages(pickeddepth_ages, core_name):
        """Validate pickeddepth ages for a given core"""
        if pickeddepth_ages is None:
            return False, f"{core_name}: pickeddepth_ages is None"
        
        if 'ages' not in pickeddepth_ages or not pickeddepth_ages['ages']:
            return False, f"{core_name}: no 'ages' found in pickeddepth_ages"
        
        # Check if the last age value is NaN
        ages = pickeddepth_ages['ages']
        if len(ages) > 0 and np.isnan(ages[-1]):
            return False, f"{core_name}: last age value in pickeddepth_ages['ages'] is NaN"
        
        # Check if all ages are NaN
        valid_ages = [age for age in ages if not np.isnan(age)]
        if len(valid_ages) == 0:
            return False, f"{core_name}: all age values in pickeddepth_ages['ages'] are NaN"
        
        return True, f"{core_name}: pickeddepth_ages validation passed"
    
    # Validate age data for both cores
    age_data_a_valid, age_data_a_msg = validate_age_data(core_a_age_data, core_a_name)
    age_data_b_valid, age_data_b_msg = validate_age_data(core_b_age_data, core_b_name)
    pickeddepth_ages_a_valid, pickeddepth_ages_a_msg = validate_pickeddepth_ages(datum_ages_a, core_a_name)
    pickeddepth_ages_b_valid, pickeddepth_ages_b_msg = validate_pickeddepth_ages(datum_ages_b, core_b_name)
    
    # Determine if age-based analysis can be performed
    age_analysis_possible = (age_data_a_valid and age_data_b_valid and 
                           pickeddepth_ages_a_valid and pickeddepth_ages_b_valid)
    
    # Determine which cores have invalid age constraints for specific warning
    core_a_age_valid = age_data_a_valid and pickeddepth_ages_a_valid
    core_b_age_valid = age_data_b_valid and pickeddepth_ages_b_valid
    
    # Print validation results and specific warning
    if not age_analysis_possible:
        # Determine which cores have invalid age data
        invalid_cores = []
        if not core_a_age_valid:
            invalid_cores.append(core_a_name)
        if not core_b_age_valid:
            invalid_cores.append(core_b_name)
        
        if len(invalid_cores) == 2:
            core_warning = f"{core_a_name} & {core_b_name}"
        else:
            core_warning = invalid_cores[0]
        
        print(f"⚠️  WARNING: No valid age constraints in CORE {core_warning}. Only compute results without age consideration.")
    else:
        print("✓ Age data validation passed - age-based analysis is possible")
    
    # Filter parameter combinations based on age data validity
    original_param_count = len(parameter_combinations)
    if age_analysis_possible:
        # Keep all parameter combinations if age analysis is possible
        filtered_parameter_combinations = parameter_combinations
    else:
        # Remove parameter combinations with age_consideration=True if age data is invalid
        filtered_parameter_combinations = [
            params for params in parameter_combinations 
            if not params.get('age_consideration', False)
        ]
    
    filtered_param_count = len(filtered_parameter_combinations)
    
    if filtered_param_count < original_param_count:
        skipped_count = original_param_count - filtered_param_count
        print(f"   Filtered out {skipped_count} parameter combinations with age_consideration=True")
        print(f"   Proceeding with {filtered_param_count} parameter combinations")
    
    # Disable age constraint removal testing if age data is invalid
    effective_test_age_constraint_removal = test_age_constraint_removal and age_analysis_possible
    
    if test_age_constraint_removal and not effective_test_age_constraint_removal:
        print("   Age constraint removal testing has been disabled due to invalid age data")
    
    # Update parameter_combinations for the rest of the function
    parameter_combinations = filtered_parameter_combinations
    test_age_constraint_removal = effective_test_age_constraint_removal
    
    # Loop through all quality indices
    print(f"Running {len(parameter_combinations)} parameter combinations for {len(target_quality_indices)} quality indices...")
    print(f"Processing scenarios sequentially with parallel computation inside each scenario")

    # Reset variables at the beginning
    n_constraints_b = 0
    age_enabled_params = []
    total_additional_scenarios = 0

    if test_age_constraint_removal:
        n_constraints_b = len(core_b_age_data['in_sequence_ages'])
        age_enabled_params = [p for p in parameter_combinations if p['age_consideration']]
        constraint_scenarios_per_param = (2 ** n_constraints_b) - 1  # Exclude empty set
        total_additional_scenarios = len(age_enabled_params) * (constraint_scenarios_per_param - 1)  # Exclude original scenario
        
        print(f"Age constraint removal testing enabled:")
        print(f"- Core B has {n_constraints_b} age constraints")
        
        if max_search_per_layer is None:
            print(f"- Additional scenarios to process: {total_additional_scenarios}")
            # Warning for large number of scenarios
            if total_additional_scenarios > 5000:
                print(f"⚠️  WARNING: Processing {total_additional_scenarios} scenarios will take very long time and use large memory.")
                print(f"   Recommend setting max_search_per_layer to about 200-300 (but >= the number of constraints in core B: {n_constraints_b})")
                print(f"   to reduce the number of scenarios per constraint removal layer to process.")
                print(f"   When max_search_per_layer is set, scenarios are randomly sampled from each layer.")
                print(f"   This provides statistical approximation while maintaining computational feasibility.")
        else:
            # Calculate how many scenarios will actually be processed
            # This is an approximation since it depends on the distribution across layers
            print(f"- Additional {total_additional_scenarios} scenarios exist")
            print(f"- As max_search_per_layer is defined: {max_search_per_layer} scenarios are randomly sampled from each constraint removal layer")
            print(f"- ⚠️  WARNING: Due to random sampling, not every run will yield identical results")
            print(f"-     This just provides statistical approximation while maintaining computational feasibility")
            # Check if max_search_per_layer is too small
            if max_search_per_layer < n_constraints_b:
                print(f"- ⚠️  WARNING: max_search_per_layer ({max_search_per_layer}) is less than the number of constraints ({n_constraints_b})")
                print(f"-     Recommend setting max_search_per_layer >= {n_constraints_b} to ensure all age constraints are evaluated")

    # PHASE 1: Run original parameter combinations
    if test_age_constraint_removal:
        print("\n=== PHASE 1: Running original parameter combinations ===")
    else:
        print("\nRunning parameter combinations...")
    
    if not age_analysis_possible:
        print("Note: Only processing parameter combinations with age_consideration=False due to invalid age data")

    # Parallelism strategy: sequential scenarios with full parallelism inside each
    n_combinations = len(parameter_combinations)

    # Prepare data for processing
    phase1_args = [
        (idx, params, 
         log_a, log_b, md_a, md_b,
         picked_datum_a, picked_datum_b,
         datum_ages_a, datum_ages_b,
         core_a_age_data, core_b_age_data,
         target_quality_indices,
         output_csv_filenames,
         synthetic_csv_filenames,
         pca_for_dependent_dtw,
         test_age_constraint_removal,
         output_metric_only) 
        for idx, params in enumerate(parameter_combinations)
    ]
    
    # Run Phase 1 sequentially with progress bar showing elapsed time
    desc_phase1 = "Phase 1: Parameter combinations" if test_age_constraint_removal else "Processing parameter combinations"
    phase1_results = []
    with tqdm(total=n_combinations, desc=desc_phase1, unit="combo", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for args in phase1_args:
            result = _process_single_parameter_combination(*args)
            phase1_results.append(result)
            pbar.update(1)
    
    # Process Phase 1 results and write to CSV
    # Track which quality indices have had their header written
    header_written = {quality_index: False for quality_index in target_quality_indices}
    
    for idx, (success, combo_id, results) in enumerate(phase1_results):
        if success:
            # Check if results is empty (scenario was skipped)
            if not results:
                print(f"⊘ Skipped {combo_id}: scenario not found in synthetic CSV")
                continue
            
            # Write results to CSV files
            for quality_index in target_quality_indices:
                if quality_index in results:
                    fit_params = results[quality_index]
                    master_csv_filename = output_csv_filenames[quality_index]
                    
                    df_single = pd.DataFrame([fit_params])
                    # Write header only for the first result for each quality index
                    if not header_written[quality_index]:
                        df_single.to_csv(master_csv_filename, mode='w', index=False, header=True)
                        header_written[quality_index] = True
                    else:
                        df_single.to_csv(master_csv_filename, mode='a', index=False, header=False)
                    del df_single
        else:
            print(f"✗ Error in {combo_id}: {results}")

    print("✓ All original parameter combinations processed" if test_age_constraint_removal else "✓ All parameter combinations processed")

    # PHASE 2: Run age constraint removal scenarios (if enabled)
    # Warning for large number of scenarios
    if test_age_constraint_removal:
        print("\n=== PHASE 2: Running age constraint removal scenarios ===")
        
        # Calculate additional scenarios
        n_constraints_b = len(core_b_age_data['in_sequence_ages'])
        age_enabled_params = [p for p in parameter_combinations if p['age_consideration']]
        
        # Check if there are any valid age constraints and age-enabled parameters
        if n_constraints_b == 0 or len(age_enabled_params) == 0:
            print("✓ Phase 2 completed: No age constraint removal scenarios to process")
        else:
            constraint_scenarios_per_param = (2 ** n_constraints_b) - 1  # Exclude empty set
            total_additional_scenarios = len(age_enabled_params) * (constraint_scenarios_per_param - 1)  # Exclude original scenario
           
            print(f"- Core B has {n_constraints_b} age constraints")
            print(f"- Processing {total_additional_scenarios} additional constraint removal scenarios")
            
            # Get indices of only in-sequence constraints from the original data
            in_sequence_indices = []
            for i, flag in enumerate(core_b_age_data['in_sequence_flags']):
                # Handle various flag formats (string, boolean, numeric)
                is_in_sequence = False
                if isinstance(flag, str):
                    is_in_sequence = flag.upper() == 'TRUE'
                elif isinstance(flag, (bool, np.bool_)):
                    is_in_sequence = flag
                else:
                    is_in_sequence = flag == 1
                
                if is_in_sequence:
                    in_sequence_indices.append(i)
            
            n_constraints_b = len(in_sequence_indices)  # Count of in-sequence constraints only

            # Generate subsets from in-sequence constraint indices only
            all_subsets = generate_constraint_subsets(n_constraints_b)
            constraint_subsets = [subset for subset in all_subsets if 0 < len(subset) < n_constraints_b]
            
            # Apply max_search_per_layer limitation if specified
            if max_search_per_layer is not None:
                # Group constraint subsets by layer (number of remaining constraints)
                print(f"max_search_per_layer is defined: randomly sampling up to {max_search_per_layer} scenarios per layer of search")
                layers = {}
                for subset in constraint_subsets:
                    layer_size = len(subset)
                    if layer_size not in layers:
                        layers[layer_size] = []
                    layers[layer_size].append(subset)
                
                # Sample from each layer if it exceeds max_search_per_layer
                limited_constraint_subsets = []
                for layer_size in sorted(layers.keys()):
                    layer_subsets = layers[layer_size]
                    if len(layer_subsets) > max_search_per_layer:
                        # Use numpy random to avoid conflicts and ensure no repeats within layer
                        layer_indices = np.arange(len(layer_subsets))
                        sampled_indices = np.random.choice(layer_indices, size=max_search_per_layer, replace=False)
                        sampled_subsets = [layer_subsets[i] for i in sampled_indices]
                        
                        limited_constraint_subsets.extend(sampled_subsets)
                        print(f"- Layer {layer_size} constraints: {len(layer_subsets)} scenarios → {max_search_per_layer} sampled")
                    else:
                        limited_constraint_subsets.extend(layer_subsets)
                        print(f"- Layer {layer_size} constraints: {len(layer_subsets)} scenarios (all processed)")
                
                constraint_subsets = limited_constraint_subsets
            
            # Pre-cache synthetic CSV bin information to avoid redundant reads in parallel workers
            cached_binsizes = {}
            if synthetic_csv_filenames:
                for quality_index in target_quality_indices:
                    if quality_index in synthetic_csv_filenames:
                        synthetic_csv_file = synthetic_csv_filenames[quality_index]
                        if os.path.exists(synthetic_csv_file):
                            try:
                                synthetic_df = pd.read_csv(synthetic_csv_file)
                                if not synthetic_df.empty and 'bins' in synthetic_df.columns:
                                    bins_str = synthetic_df.iloc[0]['bins']
                                    if pd.notna(bins_str):
                                        synthetic_bins = np.fromstring(bins_str.strip('[]'), sep=' ')
                                        bin_width_synthetic = np.mean(np.diff(synthetic_bins))
                                        cached_binsizes[quality_index] = (synthetic_bins, bin_width_synthetic)
                            except Exception:
                                pass
            
            # Prepare data for Phase 2 parallel processing
            phase2_args = []
            for param_idx, params in enumerate(age_enabled_params):
                for constraint_subset in constraint_subsets:
                    phase2_args.append((
                        param_idx, params, constraint_subset, in_sequence_indices,
                        log_a, log_b, md_a, md_b,
                        picked_datum_a, picked_datum_b,
                        datum_ages_a, datum_ages_b,
                        core_a_age_data, core_b_age_data,
                        uncertainty_method,
                        target_quality_indices,
                        cached_binsizes,
                        pca_for_dependent_dtw,
                        output_metric_only
                    ))
            
            # Run Phase 2 sequentially with progress bar showing elapsed time
            print(f"Processing {len(phase2_args)} scenarios sequentially (parallel inside each scenario)")
            
            phase2_results = []
            with tqdm(total=len(phase2_args), desc="Phase 2: Age constraint removal", unit="scenario",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for args in phase2_args:
                    result = _process_single_constraint_scenario(*args)
                    phase2_results.append(result)
                    pbar.update(1)
            
            # Process Phase 2 results and append to CSV
            for success, scenario_id, results in phase2_results:
                if success:
                    # Append results to CSV files
                    for quality_index in target_quality_indices:
                        if quality_index in results:
                            fit_params = results[quality_index]
                            master_csv_filename = output_csv_filenames[quality_index]
                            
                            df_single = pd.DataFrame([fit_params])
                            df_single.to_csv(master_csv_filename, mode='a', index=False, header=False)
                            del df_single
                else:
                    print(f"✗ Error in {scenario_id}: {results}")

            print("✓ Phase 2 completed: All age constraint removal scenarios processed")

    # Final summary
    print(f"\n✓ All processing completed")
    
    for quality_index in target_quality_indices:
        filename = output_csv_filenames[quality_index]
        print(f"✓ {quality_index} fit_params saved to: {filename}")


def create_synthetic_core_pair(core_a_length, core_b_length, seg_logs, seg_depths, 
                                       log_columns, repetition=False, plot_results=True, save_plot=False, plot_filename=None):
    """
    Generate synthetic core pair (computation only).
    
    Parameters:
    - core_a_length: target length for core A
    - core_b_length: target length for core B
    - seg_logs: list of turbidite log segments
    - seg_depths: list of corresponding depth arrays
    - log_columns: list of log column names for labeling
    - repetition: if True, allow reusing turbidite segments; if False, each segment can only be used once (default: False)
    - plot_results: whether to display the plot
    - save_plot: whether to save the plot to file
    - plot_filename: filename for saving plot (if save_plot=True)
    
    Returns:
    - tuple: (synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a,
              synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b)
    """
    
    # Generate synthetic logs for cores A and B
    print("Generating synthetic core pair...")

    synthetic_log_a, synthetic_md_a, synthetic_picked_a, inds_a = create_synthetic_log(
        core_a_length, seg_logs, seg_depths, exclude_inds=None, repetition=repetition
    )
    synthetic_log_b, synthetic_md_b, synthetic_picked_b, inds_b = create_synthetic_log(
        core_b_length, seg_logs, seg_depths, exclude_inds=None, repetition=repetition
    )

    # Plot synthetic core pair if requested
    if plot_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 9))
        
        # Define colors for different log types
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        line_styles = ['-', '-', '-', '-.', '-.', '-.', ':', ':', ':']

        # Plot synthetic core A
        if synthetic_log_a.ndim > 1:
            n_log_types = synthetic_log_a.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_columns[col_idx] if col_idx < len(log_columns) else f'Log_{col_idx}'
                
                ax1.plot(synthetic_log_a[:, col_idx], synthetic_md_a, 
                        color=color, linestyle=line_style, linewidth=1, 
                        label=col_name, alpha=0.8)
            
            # Add legend if multiple log types
            if n_log_types > 1:
                ax1.legend(fontsize=8, loc='upper right')
                
            # Set xlabel to show all log types
            if len(log_columns) > 1:
                ax1.set_xlabel(f'Multiple Logs (normalized)')
            else:
                ax1.set_xlabel(f'{log_columns[0]} (normalized)')
        else:
            ax1.plot(synthetic_log_a, synthetic_md_a, 'b-', linewidth=1)
            ax1.set_xlabel(f'{log_columns[0]} (normalized)')

        # Add picked depths as horizontal lines
        for depth in synthetic_picked_a:
            ax1.axhline(y=depth, color='black', linestyle='--', alpha=0.7, linewidth=1)

        ax1.set_ylabel('Depth (cm)')
        ax1.set_title(f'Synthetic Core A\n({len(inds_a)} turbidites)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()

        # Plot synthetic core B
        if synthetic_log_b.ndim > 1:
            n_log_types = synthetic_log_b.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_columns[col_idx] if col_idx < len(log_columns) else f'Log_{col_idx}'
                
                ax2.plot(synthetic_log_b[:, col_idx], synthetic_md_b, 
                        color=color, linestyle=line_style, linewidth=1, 
                        label=col_name, alpha=0.8)
            
            # Add legend if multiple log types
            if n_log_types > 1:
                ax2.legend(fontsize=8, loc='upper right')
                
            # Set xlabel to show all log types
            if len(log_columns) > 1:
                ax2.set_xlabel(f'Multiple Logs (normalized)')
            else:
                ax2.set_xlabel(f'{log_columns[0]} (normalized)')
        else:
            ax2.plot(synthetic_log_b, synthetic_md_b, 'g-', linewidth=1)
            ax2.set_xlabel(f'{log_columns[0]} (normalized)')

        # Add picked depths as horizontal lines
        for depth in synthetic_picked_b:
            ax2.axhline(y=depth, color='black', linestyle='--', alpha=0.7, linewidth=1)

        ax2.set_ylabel('Depth (cm)')
        ax2.set_title(f'Synthetic Core B\n({len(inds_b)} turbidites)')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_plot and plot_filename:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {plot_filename}")
        
        plt.show()

    print(f"Synthetic Core A: {len(synthetic_log_a)} points, {len(inds_a)} turbidites, {len(synthetic_picked_a)} boundaries")
    print(f"Synthetic Core B: {len(synthetic_log_b)} points, {len(inds_b)} turbidites, {len(synthetic_picked_b)} boundaries")
    print(f"Turbidite indices used in A: {[int(x) for x in inds_a[:10]]}..." if len(inds_a) > 10 else f"Turbidite indices used in A: {[int(x) for x in inds_a]}")
    print(f"Turbidite indices used in B: {[int(x) for x in inds_b[:10]]}..." if len(inds_b) > 10 else f"Turbidite indices used in B: {[int(x) for x in inds_b]}")
    
    return (synthetic_log_a, synthetic_md_a, inds_a, synthetic_picked_a,
            synthetic_log_b, synthetic_md_b, inds_b, synthetic_picked_b)


def plot_segment_pool(segment_logs, segment_depths, log_data_type, n_cols=8, figsize_per_row=4, 
                     plot_segments=True, save_plot=False, plot_filename=None):
    """
    Plot all segments from the pool in a grid layout.
    
    Parameters:
    - segment_logs: list of log data arrays (segments)
    - segment_depths: list of depth arrays corresponding to each segment
    - log_data_type: list of column names for labeling
    - n_cols: number of columns in the subplot grid
    - figsize_per_row: height per row in the figure
    - plot_segments: whether to plot the segments (default True)
    - save_plot: whether to save the plot to file (default False)
    - plot_filename: filename for saving plot (optional)
    
    Returns:
    - None
    """
    print(f"Plotting {len(segment_logs)} segments from the pool...")
    
    if not plot_segments:
        return
    
    # Create subplot grid
    n_segments = len(segment_logs)
    n_rows = int(np.ceil(n_segments / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, figsize_per_row * n_rows))
    axes = axes.flatten() if n_segments > 1 else [axes]
    
    # Define colors for different log types
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    line_styles = ['-', '-', '-', '-.', '-.', '-.', ':', ':', ':']
    
    for i, (segment, depth) in enumerate(zip(segment_logs, segment_depths)):
        ax = axes[i]
        
        # Plot segment
        if segment.ndim > 1:
            # Multi-dimensional data - plot all columns
            n_log_types = segment.shape[1]
            
            for col_idx in range(n_log_types):
                color = colors[col_idx % len(colors)]
                line_style = line_styles[col_idx % len(line_styles)]
                
                # Get column name for label
                col_name = log_data_type[col_idx] if col_idx < len(log_data_type) else f'Log_{col_idx}'
                
                ax.plot(segment[:, col_idx], depth, 
                       color=color, linestyle=line_style, linewidth=1, 
                       label=col_name, alpha=0.8)
            
            # Set xlabel to show all log types
            if len(log_data_type) > 1:
                ax.set_xlabel(f'Multiple Logs: {", ".join(log_data_type[:n_log_types])} (normalized)')
            else:
                ax.set_xlabel(f'{log_data_type[0]} (normalized)')
                
            # Add legend if multiple log types
            if n_log_types > 1:
                ax.legend(fontsize=8, loc='best')
                
        else:
            # 1D data
            ax.plot(segment, depth, 'b-', linewidth=1)
            ax.set_xlabel(f'{log_data_type[0]} (normalized)')
        
        ax.set_ylabel('Relative Depth (cm)')
        ax.set_title(f'Segment {i+1}\n({len(segment)} pts, {depth[-1]:.1f} cm)')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Depth increases downward
    
    # Hide unused subplots
    for i in range(n_segments, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Update title to reflect multiple log types if present
    if len(log_data_type) > 1:
        plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments, {len(log_data_type)} log types)', 
                     y=1.02, fontsize=16)
    else:
        plt.suptitle(f'Turbidite Segment Pool ({len(segment_logs)} segments)', y=1.02, fontsize=16)
    
    if save_plot and plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    plt.show()


def plot_synthetic_log(synthetic_log, synthetic_md, synthetic_picked_datum, log_data_type, 
                      title="Synthetic Log", save_plot=False, plot_filename=None):
    """
    Plot a single synthetic log with turbidite boundaries.

    Parameters:
    - synthetic_log: numpy array of log values (can be 1D or 2D for multiple log types)
    - synthetic_md: numpy array of depth values
    - synthetic_picked_datum: list of turbidite boundary depths
    - log_data_type: name(s) of the log column(s) for labeling (string or list)
    - title: title for the plot (default: "Synthetic Log")
    - save_plot: whether to save the plot to file (default: False)
    - plot_filename: filename for saving plot (if save_plot=True)

    Returns:
    - fig, ax: matplotlib figure and axis objects
    """

    # Convert log_data_type to list if it's a string
    if isinstance(log_data_type, str):
        log_data_type = [log_data_type]

    # Increase width for legend outside plot
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 8))

    # Define colors and line styles for different log types (matching old code)
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    line_styles = ['-', ':', '--', '-.', '-', ':', '--', '-.']

    # Plot the synthetic log
    legend_handle = None
    if synthetic_log.ndim > 1:
        # Multi-dimensional data - plot all columns
        n_log_types = synthetic_log.shape[1]

        for col_idx in range(n_log_types):
            color = colors[col_idx % len(colors)]
            line_style = line_styles[col_idx % len(line_styles)]

            # Get column name for label
            col_name = log_data_type[col_idx] if col_idx < len(log_data_type) else f'Log_{col_idx}'

            ax.plot(synthetic_log[:, col_idx], synthetic_md,
                    color=color, linestyle=line_style, linewidth=1,
                    label=col_name, alpha=0.8)

        # Add legend if multiple log types, outside the plot at best position
        if n_log_types > 1:
            # Shrink current axis by 20% on the right to fit the legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend_handle = ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Set xlabel to show all log types
        if len(log_data_type) > 1:
            ax.set_xlabel(f'Multiple Logs\n(Normalized)')
        else:
            ax.set_xlabel(f'{log_data_type[0]}\n(normalized)')
    else:
        # 1D data
        ax.plot(synthetic_log, synthetic_md, 'b-', linewidth=1)
        ax.set_xlabel(f'{log_data_type[0]} (normalized)')

    # Add picked depths as horizontal lines
    for depth in synthetic_picked_datum:
        ax.axhline(y=depth, color='black', linestyle='--', alpha=0.7, linewidth=1)

    ax.set_ylabel('Depth (cm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_plot and plot_filename:
        # If legend exists, pass it to savefig so it is not cut off
        if legend_handle is not None:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', bbox_extra_artists=(legend_handle,))
        else:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")

    plt.show()

    return fig, ax

def plot_synthetic_correlation_quality(
    input_csv, 
    quality_indices=['corr_coef', 'norm_dtw'],
    bin_width=None,
    plot_individual_pdf=False,
    save_plot=False,
    plot_filename=None,
    invert_norm_dtw=True
):
    """
    Plot synthetic correlation quality distributions from saved CSV files.
    
    This function reads fit parameters from CSV files generated by synthetic_correlation_quality
    and creates visualization plots showing either individual PDFs from each iteration or a 
    combined distribution.
    
    Parameters:
    -----------
    input_csv : str
        Path to the CSV file containing fit parameters. Can include {quality_index} placeholder
        which will be replaced for each quality index.
        Example: 'outputs/synthetic_PDFs_hiresMS_CT_Lumin_{quality_index}.csv'
    quality_indices : list, default=['corr_coef', 'norm_dtw']
        List of quality indices to plot
    bin_width : float or None, default=None
        Bin width for histogram. If None, uses quality-specific defaults:
        - corr_coef: 0.025
        - norm_dtw: 0.0025
        - others: auto-determined using Freedman-Diaconis rule
    plot_individual_pdf : bool, default=False
        If True, plots all individual iteration PDFs overlaid (following Cell 9 approach)
        If False, plots combined distribution across all iterations (following Cell 10 approach)
    save_plot : bool, default=False
        Whether to save the plot to file
    plot_filename : str or None, default=None
        Filename for saving plot. Can include {quality_index} placeholder.
        If save_plot=True but plot_filename=None, uses default naming based on plot type
    invert_norm_dtw : bool, default=True
        If True, for norm_dtw and norm_dtw_sect metrics, plot (1 - norm_dtw) values
        so that higher values indicate better quality (consistent with corr_coef).
        If False, plot original norm_dtw values (lower = better).
    
    Returns:
    --------
    None
        Displays and optionally saves plots
    """
    
    from scipy import stats
    
    for targeted_quality_index in quality_indices:
        print(f"\nPlotting distribution for {targeted_quality_index}...")
        
        # Replace placeholder in input_csv with actual quality index
        input_csv_filename = input_csv.replace('{quality_index}', targeted_quality_index).replace('{targeted_quality_index}', targeted_quality_index)
        
        # Check if file exists
        if not os.path.exists(input_csv_filename):
            print(f"Error: File {input_csv_filename} does not exist. Skipping {targeted_quality_index}.")
            continue
        
        # Load fit params from CSV
        df_fit_params = pd.read_csv(input_csv_filename)
        
        # Reconstruct raw data for each iteration
        all_fit_params = []
        for _, row in df_fit_params.iterrows():
            # Extract binned data for reconstruction
            bins = np.fromstring(row['bins'].strip('[]'), sep=' ') if 'bins' in row and pd.notna(row['bins']) else None
            hist_percentages = np.fromstring(row['hist'].strip('[]'), sep=' ') if 'hist' in row and pd.notna(row['hist']) else None
            n_points = row['n_points'] if 'n_points' in row and pd.notna(row['n_points']) else None
            
            # Reconstruct raw data from histogram
            raw_data = []
            if bins is not None and hist_percentages is not None and n_points is not None:
                # Normalize histogram to get proper proportions
                hist_sum = np.sum(hist_percentages)
                if hist_sum > 0:
                    raw_counts = (hist_percentages / hist_sum) * n_points
                else:
                    raw_counts = np.zeros_like(hist_percentages)
                
                # Reconstruct data points by sampling from each bin
                for i, count in enumerate(raw_counts):
                    if count > 0:
                        n_samples = int(round(count))
                        if n_samples > 0:
                            bin_samples = np.random.uniform(bins[i], bins[i+1], n_samples)
                            raw_data.extend(bin_samples)
            
            # Store fit params with reconstructed raw data
            raw_data_array = np.array(raw_data)
            
            # Transform norm_dtw values if invert_norm_dtw is True
            if invert_norm_dtw and targeted_quality_index in ['norm_dtw', 'norm_dtw_sect']:
                raw_data_array = 1 - raw_data_array
            
            fit_params = {
                'raw_data': raw_data_array,
                'bins': bins,
                'mean': row['mean'] if 'mean' in row else None,
                'std': row['std'] if 'std' in row else None
            }
            all_fit_params.append(fit_params)
        
        # Determine bin width
        if bin_width is None:
            if targeted_quality_index == 'corr_coef':
                current_bin_width = 0.025
            elif targeted_quality_index == 'norm_dtw':
                current_bin_width = 0.0025
            else:
                # Use bin_width from first iteration as fallback
                if len(all_fit_params) > 0 and all_fit_params[0]['bins'] is not None:
                    first_bins = all_fit_params[0]['bins']
                    current_bin_width = first_bins[1] - first_bins[0]
                else:
                    current_bin_width = 0.025  # Default fallback
        else:
            current_bin_width = bin_width
        
        if plot_individual_pdf:
            # ===== CELL 9 APPROACH: Plot individual PDFs =====
            
            # Find data range across ALL iterations to create consistent bins
            all_min = min([fp['raw_data'].min() for fp in all_fit_params if len(fp['raw_data']) > 0])
            all_max = max([fp['raw_data'].max() for fp in all_fit_params if len(fp['raw_data']) > 0])
            
            # Create explicit bin edges with consistent bin width
            bin_start = np.floor(all_min / current_bin_width) * current_bin_width
            bin_end = np.ceil(all_max / current_bin_width) * current_bin_width
            consistent_bins = np.arange(bin_start, bin_end + current_bin_width, current_bin_width)
            
            # Plot all distribution curves and histogram bars
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Plot histogram bars for each iteration
            for fit_params in all_fit_params:
                raw_data = fit_params.get('raw_data')
                if raw_data is not None and len(raw_data) > 0:
                    ax.hist(raw_data, bins=consistent_bins, alpha=0.1, color='gray', 
                            density=False, edgecolor='none',
                            weights=np.ones(len(raw_data)) * 100 / len(raw_data))
            
            # Plot all PDF curves as transparent red lines
            for fit_params in all_fit_params:
                raw_data = fit_params.get('raw_data')
                
                if raw_data is not None and len(raw_data) > 0:
                    # Compute mean and std from the (potentially inverted) raw data
                    mean_val = np.mean(raw_data)
                    std_val = np.std(raw_data)
                    
                    if std_val > 0:
                        # Generate PDF curve with proper scaling
                        x_min = raw_data.min()
                        x_max = raw_data.max()
                        x = np.linspace(x_min, x_max, 1000)
                        # Scale PDF by bin_width * 100 to match histogram percentage scale
                        y = stats.norm.pdf(x, mean_val, std_val) * current_bin_width * 100
                        ax.plot(x, y, 'r-', linewidth=.7, alpha=0.3)
            
            # Formatting based on quality index
            # Sectional metrics use the same x-axis range as their non-sectional counterparts
            if targeted_quality_index in ['corr_coef', 'corr_coef_sect']:
                ax.set_xlabel("Pearson's r\n(Correlation Coefficient)")
                ax.set_xlim(0, 1.0)
            elif targeted_quality_index in ['norm_dtw', 'norm_dtw_sect']:
                if invert_norm_dtw:
                    ax.set_xlabel("1 - Normalized DTW Cost\n(Higher = Better)")
                    ax.set_xlim(0.6, 1.0)
                else:
                    ax.set_xlabel("Normalized DTW Cost")
            elif targeted_quality_index == 'dtw_ratio':
                ax.set_xlabel("DTW Ratio")
            elif targeted_quality_index == 'perc_diag':
                ax.set_xlabel("Percentage Diagonal (%)")
            
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Synthetic Core {targeted_quality_index.replace("_", " ").title()}: {len(all_fit_params)} Iterations\n[Optimal (shortest path) search; no age consideration)]')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plot:
                if plot_filename:
                    output_filename = plot_filename.replace('{quality_index}', targeted_quality_index).replace('{targeted_quality_index}', targeted_quality_index)
                else:
                    output_filename = f'synthetic_iterations_{targeted_quality_index}.png'
                plt.savefig(output_filename, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {output_filename}")
            
            plt.show()
            
        else:
            # ===== CELL 10 APPROACH: Combined distribution =====
            
            # Initialize lists to collect all raw data points
            all_raw_data = []
            
            # Process each iteration to reconstruct raw data from binned data
            for fit_params in all_fit_params:
                raw_data = fit_params.get('raw_data')
                if raw_data is not None and len(raw_data) > 0:
                    all_raw_data.extend(raw_data)
            
            # Convert to numpy array
            combined_data = np.array(all_raw_data)
            
            print(f"Combined {len(combined_data)} data points from {len(df_fit_params)} iterations")
            
            # Calculate combined statistics
            combined_mean = np.mean(combined_data)
            combined_std = np.std(combined_data)
            combined_median = np.median(combined_data)
            
            # Determine bin width for combined data
            if bin_width is None:
                if targeted_quality_index == 'corr_coef':
                    final_bin_width = 0.025
                elif targeted_quality_index == 'norm_dtw':
                    final_bin_width = 0.0025
                else:
                    # Automatically determine bin width using Freedman-Diaconis rule
                    data_range = combined_data.max() - combined_data.min()
                    if data_range > 0:
                        if len(combined_data) > 1:
                            q75, q25 = np.percentile(combined_data, [75, 25])
                            iqr = q75 - q25
                            if iqr > 0:
                                final_bin_width = 2 * iqr / (len(combined_data) ** (1/3))
                            else:
                                # Fallback to simple rule if IQR is 0
                                final_bin_width = data_range / max(10, min(int(np.sqrt(len(combined_data))), 100))
                        else:
                            final_bin_width = 0.1  # Default for single value
                    else:
                        final_bin_width = 0.1  # Default for zero range
            else:
                final_bin_width = bin_width
            
            # Calculate number of bins based on bin_width
            data_range = combined_data.max() - combined_data.min()
            if data_range > 0 and final_bin_width > 0:
                n_bins = max(1, int(np.ceil(data_range / final_bin_width)))
                # Constrain to reasonable range
                n_bins = max(10, min(n_bins, 200))
            else:
                n_bins = 10  # Default fallback
            
            # Create new histogram from combined data as PERCENTAGES
            hist_combined, bins_combined = np.histogram(combined_data, bins=n_bins, density=False)
            # Convert counts to percentages
            hist_percentages = (hist_combined / len(combined_data)) * 100
            actual_bin_width = bins_combined[1] - bins_combined[0]
            
            # Fit normal distribution to combined data
            fitted_mean, fitted_std = stats.norm.fit(combined_data)
            
            # Generate fitted curve and scale to percentage
            x_fitted = np.linspace(combined_data.min(), combined_data.max(), 1000)
            # PDF must be scaled by: actual_bin_width * 100 (to convert density to percentage per bin)
            y_fitted = stats.norm.pdf(x_fitted, fitted_mean, fitted_std) * actual_bin_width * 100
            
            # Verify histogram normalization
            total_percentage = np.sum(hist_percentages)
            print(f"Histogram total percentage: {total_percentage:.2f}% (should be 100%)")
            print(f"Number of bins used: {n_bins}, Bin width: {actual_bin_width:.6f}")
            
            # Verify PDF curve integration
            dx = x_fitted[1] - x_fitted[0]
            pdf_area = np.trapz(y_fitted, dx=dx)
            print(f"PDF curve area: {pdf_area:.2f}% (should be ~100%)")
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Plot combined histogram in gray bars as PERCENTAGES
            ax.hist(combined_data, bins=n_bins, alpha=0.5, color='gray', density=False, 
                    weights=np.ones(len(combined_data)) * 100 / len(combined_data), 
                    label=f'Combined Data (n = {len(combined_data):,})')
            
            # Plot fitted normal curve as red line
            ax.plot(x_fitted, y_fitted, 'r-', linewidth=2, alpha=0.8,
                    label=f'Normal Fit (μ={fitted_mean:.3f}, σ={fitted_std:.3f})')
            
            # Formatting based on quality index
            # Sectional metrics use the same x-axis range as their non-sectional counterparts
            if targeted_quality_index in ['corr_coef', 'corr_coef_sect']:
                ax.set_xlabel("Pearson's r\n(Correlation Coefficient)")
                ax.set_xlim(0, 1.0)
            elif targeted_quality_index in ['norm_dtw', 'norm_dtw_sect']:
                if invert_norm_dtw:
                    ax.set_xlabel("1 - Normalized DTW Cost\n(Higher = Better)")
                    ax.set_xlim(0.6, 1.0)
                else:
                    ax.set_xlabel("Normalized DTW Cost")
            elif targeted_quality_index == 'dtw_ratio':
                ax.set_xlabel("DTW Ratio")
            elif targeted_quality_index == 'perc_diag':
                ax.set_xlabel("Percentage Diagonal (%)")
            
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Combined {targeted_quality_index.replace("_", " ").title()} Distribution from All {len(df_fit_params)} Iterations\n[Synthetic Core Analysis - Null Hypothesis]')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_plot:
                if plot_filename:
                    output_filename = plot_filename.replace('{quality_index}', targeted_quality_index).replace('{targeted_quality_index}', targeted_quality_index)
                else:
                    output_filename = f'combined_synthetic_distribution_{targeted_quality_index}.png'
                plt.savefig(output_filename, dpi=150, bbox_inches='tight')
                print(f"Saved plot to {output_filename}")
            
            plt.show()
            
            # Print comprehensive summary
            print(f"\nCombined Distribution Summary for {targeted_quality_index}:")
            print(f"{'='*50}")
            print(f"Total data points: {len(combined_data):,}")
            print(f"Number of iterations combined: {len(df_fit_params)}")
            print(f"Combined Mean: {combined_mean:.4f}")
            print(f"Combined Median: {combined_median:.4f}")
            print(f"Combined Std Dev: {combined_std:.4f}")
            print(f"Data Range: {combined_data.min():.4f} to {combined_data.max():.4f}")
            print(f"\nFitted Normal Distribution:")
            print(f"Fitted Mean (μ): {fitted_mean:.4f}")
            print(f"Fitted Std Dev (σ): {fitted_std:.4f}")
            
            # Calculate percentiles
            percentiles = [5, 25, 50, 75, 95]
            pct_values = np.percentile(combined_data, percentiles)
            print(f"\nPercentiles:")
            for pct, val in zip(percentiles, pct_values):
                print(f"{pct}th percentile: {val:.4f}")


def synthetic_correlation_quality(
    segment_logs, 
    segment_depths, 
    log_data_type, 
    quality_indices=['corr_coef', 'norm_dtw'], 
    number_of_iterations=20, 
    max_core_a_thickness=None, 
    max_core_b_thickness=None,
    max_num_units_core_a=None,
    max_num_units_core_b=None,
    repetition=False, 
    pca_for_dependent_dtw=False, 
    output_csv_dir=None,
    max_search_path=10000,
    mute_mode=True,
    append_mode=False,
    combination_id=None,
    max_paths_for_metrics=None,
    n_jobs=-1,
    method='random',
    markov_params=None,
    segment_features=None
):
    """
    Run DTW correlation quality measurement analysis for synthetic core pairs over multiple iterations.

    This function generates synthetic core pairs, computes selected correlation quality metrics (such as DTW and correlation coefficient) for each pair, and saves the resulting quality distributions for later plotting and assessment. Distribution parameters for each quality metric are saved across all iterations.

    Parameters
    ----------
    segment_logs : list
        List of turbidite log segments (numpy arrays or compatible).
    segment_depths : list
        List of turbidite depth segments (numpy arrays or compatible).
    log_data_type : list of str
        List of log column names to consider in the analysis.
    quality_indices : list of str, default ['corr_coef', 'norm_dtw']
        List of quality indices (metrics) to compute for each synthetic pair.
    number_of_iterations : int, default 20
        Number of synthetic core pairs to generate/analyze.
    max_core_a_thickness : float or None, default None
        Maximum thickness for synthetic core A. If None and max_num_units_core_a is provided,
        thickness is determined by stacking units. If both are None, defaults to max_num_units=10.
    max_core_b_thickness : float or None, default None
        Maximum thickness for synthetic core B. If None and max_num_units_core_b is provided,
        thickness is determined by stacking units. If both are None, defaults to max_num_units=10.
    max_num_units_core_a : int or None, default None
        Maximum number of units to stack for core A. If None and max_core_a_thickness is provided,
        stacking continues until thickness is reached.
    max_num_units_core_b : int or None, default None
        Maximum number of units to stack for core B. If None and max_core_b_thickness is provided,
        stacking continues until thickness is reached.
    repetition : bool, default False
        Whether to allow resampling (reuse) of turbidite segments when assembling synthetic cores.
    pca_for_dependent_dtw : bool, default False
        If True, applies PCA transformation to logs for dependent DTW analysis.
    output_csv_dir : str or None, default None
        Output directory path for saving quality metric CSV files.
        If None, files are saved in the current directory.
        If provided, directory is created if it does not exist.
        Output files will be named 'synthetic_PDFs_{log_columns}_{quality_index}.csv'.
    max_search_path : int, default 10000
        Maximum allowable search path for the DTW algorithm.
    mute_mode : bool, default True
        If True, suppresses detailed informational output.
    append_mode : bool, default False
        If True, appends results to existing CSV files instead of overwriting.
        Useful when running multiple combinations of core lengths.
    combination_id : int or None, default None
        Optional identifier for the current combination of core lengths.
        If provided, will be saved in the output CSV.
    max_paths_for_metrics : int or None, default None
        Maximum paths to compute metrics for. If total paths exceed this, a random
        sample is used. Useful for distribution fitting where full enumeration is
        unnecessary. If None, uses max_search_path value (no additional sampling).
    n_jobs : int, default -1
        Number of parallel jobs for metric computation. -1 uses all available cores.
    method : str, default 'random'
        Segment selection method for create_synthetic_log:
        - 'random': Random selection (original behavior)
        - 'MarkovChain': Markov Chain-based selection using trained cluster transitions
    markov_params : dict or None, default None
        Dictionary from train_markov_model() containing trained model parameters.
        Required when method='MarkovChain'.
    segment_features : array-like or None, default None
        Feature array of shape (n_segments, n_features) for cluster assignment.
        Required when method='MarkovChain'.

    Returns
    -------
    dict
        Mapping from each quality index to its corresponding output CSV filename.
    """
    
    # Create output directory if specified
    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)
    
    # Prepare output filenames
    output_files = {}
    for targeted_quality_index in quality_indices:
        # Construct filename
        filename = f'synthetic_PDFs_{"_".join(log_data_type)}_{targeted_quality_index}.csv'
        if output_csv_dir:
            output_files[targeted_quality_index] = os.path.join(output_csv_dir, filename)
        else:
            output_files[targeted_quality_index] = filename
    
    # Define temp CSV path
    temp_filename = f'temp_synthetic_{"_".join(log_data_type)}_core_pair_metrics.csv'
    if output_csv_dir:
        temp_csv = os.path.join(output_csv_dir, temp_filename)
    else:
        temp_csv = temp_filename
    
    print(f"Starting synthetic correlation analysis with {number_of_iterations} iterations...")
    print(f"Quality indices: {quality_indices}")
    # Build constraint description
    core_a_desc = []
    if max_core_a_thickness is not None:
        core_a_desc.append(f"max_thickness={max_core_a_thickness}")
    if max_num_units_core_a is not None:
        core_a_desc.append(f"max_units={max_num_units_core_a}")
    if not core_a_desc:
        core_a_desc.append("max_units=10 (default)")
    
    core_b_desc = []
    if max_core_b_thickness is not None:
        core_b_desc.append(f"max_thickness={max_core_b_thickness}")
    if max_num_units_core_b is not None:
        core_b_desc.append(f"max_units={max_num_units_core_b}")
    if not core_b_desc:
        core_b_desc.append("max_units=10 (default)")
    
    print(f"Core A constraints: {', '.join(core_a_desc)}")
    print(f"Core B constraints: {', '.join(core_b_desc)}")
    
    # Run iterations with progress bar
    for iteration in tqdm(range(number_of_iterations), desc="Running synthetic analysis"):
        
        # Generate synthetic core pair
        syn_log_a, syn_md_a, syn_picked_a, inds_a = create_synthetic_log(
            segment_logs=segment_logs, 
            segment_depths=segment_depths,
            max_thickness=max_core_a_thickness,
            max_num_units=max_num_units_core_a,
            exclude_inds=None, 
            repetition=repetition,
            method=method,
            markov_params=markov_params,
            segment_features=segment_features
        )
        syn_log_b, syn_md_b, syn_picked_b, inds_b = create_synthetic_log(
            segment_logs=segment_logs, 
            segment_depths=segment_depths,
            max_thickness=max_core_b_thickness,
            max_num_units=max_num_units_core_b,
            exclude_inds=None, 
            repetition=repetition,
            method=method,
            markov_params=markov_params,
            segment_features=segment_features
        )
        
        # Run DTW analysis (syn_picked_a and syn_picked_b are already just depth values)
        dtw_result = run_comprehensive_dtw_analysis(
            syn_log_a, syn_log_b, syn_md_a, syn_md_b,
            picked_datum_a=syn_picked_a,
            picked_datum_b=syn_picked_b,
            independent_dtw=False,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            top_bottom=False,
            mute_mode=mute_mode
        )
        
        # Find complete core paths with optimizations:
        # - metrics_to_compute: Only compute the quality indices we need
        # - max_paths_for_metrics: Sample paths if too many (for distribution fitting)
        # - return_dataframe: Get results in memory (no file I/O)
        # - n_jobs: Parallel metric computation
        path_result = find_complete_core_paths(
            dtw_result,
            syn_log_a, 
            syn_log_b,
            output_csv=temp_csv,
            output_metric_only=False,
            shortest_path_search=True,
            shortest_path_level=2,
            max_search_path=max_search_path,
            mute_mode=mute_mode,
            pca_for_dependent_dtw=pca_for_dependent_dtw,
            metrics_to_compute=quality_indices,
            max_paths_for_metrics=max_paths_for_metrics,
            return_dataframe=True,
            n_jobs=n_jobs
        )
        
        # Get metrics DataFrame directly from result (Solution 2: no file I/O)
        metrics_df = path_result.get('metrics_dataframe')
        
        # Iterate through each quality index to extract fit_params
        for targeted_quality_index in quality_indices:
            
            output_csv_filename = output_files[targeted_quality_index]

            # Compute distribution params directly from DataFrame (no file read)
            if metrics_df is not None and targeted_quality_index in metrics_df.columns:
                quality_values = metrics_df[targeted_quality_index].dropna().values
                
                if len(quality_values) > 0:
                    # Compute histogram for reconstruction in plot function
                    data_min = float(np.min(quality_values))
                    data_max = float(np.max(quality_values))
                    mean_val = float(np.mean(quality_values))
                    std_val = float(np.std(quality_values))
                    median_val = float(np.median(quality_values))
                    n_points = len(quality_values)
                    
                    # Determine bin width based on quality index
                    if 'corr_coef' in targeted_quality_index:
                        bin_width = 0.025
                    elif 'norm_dtw' in targeted_quality_index:
                        bin_width = 0.0025
                    else:
                        # Use Freedman-Diaconis rule
                        iqr = np.percentile(quality_values, 75) - np.percentile(quality_values, 25)
                        bin_width = 2 * iqr / (n_points ** (1/3)) if iqr > 0 else 0.025
                    
                    # Create histogram bins
                    bin_start = np.floor(data_min / bin_width) * bin_width
                    bin_end = np.ceil(data_max / bin_width) * bin_width
                    bins = np.arange(bin_start, bin_end + bin_width, bin_width)
                    
                    # Compute histogram (as percentages)
                    hist_counts, _ = np.histogram(quality_values, bins=bins)
                    hist_percentages = (hist_counts / n_points) * 100
                    
                    # Fit normal distribution for x_range and y_values
                    x_range = np.linspace(data_min, data_max, 200)
                    from scipy import stats
                    y_values = stats.norm.pdf(x_range, mean_val, std_val) * n_points * bin_width
                    
                    fit_params = {
                        'data_min': data_min,
                        'data_max': data_max,
                        'median': median_val,
                        'std': std_val,
                        'n_points': n_points,
                        'hist_area': float(np.sum(hist_percentages)),
                        'bins': np.array2string(bins, separator=' ', max_line_width=np.inf),
                        'hist': np.array2string(hist_percentages, separator=' ', max_line_width=np.inf),
                        'bin_width': bin_width,
                        'n_bins': len(bins) - 1,
                        'method': 'normal',
                        'mean': mean_val,
                        'x_range': np.array2string(x_range, separator=' ', max_line_width=np.inf),
                        'y_values': np.array2string(y_values, separator=' ', max_line_width=np.inf)
                    }
                else:
                    fit_params = None
            else:
                fit_params = None
            
            # Store fit_params with iteration number and incrementally save to CSV
            if fit_params is not None:
                fit_params_copy = fit_params.copy()
                fit_params_copy['iteration'] = iteration
                fit_params_copy['core_a_length'] = max_core_a_thickness
                fit_params_copy['core_b_length'] = max_core_b_thickness
                if combination_id is not None:
                    fit_params_copy['combination_id'] = combination_id
                
                # Incrementally save to CSV
                df_single = pd.DataFrame([fit_params_copy])
                
                # Determine write mode based on append_mode and iteration
                if append_mode:
                    # In append mode, check if file exists to decide on header
                    file_exists = os.path.exists(output_csv_filename)
                    df_single.to_csv(output_csv_filename, mode='a', index=False, header=not file_exists)
                else:
                    if iteration == 0:
                        # Write header for first iteration
                        df_single.to_csv(output_csv_filename, mode='w', index=False, header=True)
                    else:
                        # Append subsequent iterations without header
                        df_single.to_csv(output_csv_filename, mode='a', index=False, header=False)
                
                del df_single, fit_params_copy
            
            del fit_params
        
        # Clear memory after each iteration
        del syn_log_a, syn_md_a, inds_a, syn_picked_a
        del syn_log_b, syn_md_b, inds_b, syn_picked_b
        del dtw_result, path_result, metrics_df
        
        gc.collect()

    # Remove temporary CSV file if it exists (for backward compatibility)
    # Note: With return_dataframe=True, no temp file is created
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    print(f"\nCompleted {number_of_iterations} iterations for all quality indices: {quality_indices}")
    
    for targeted_quality_index in quality_indices:
        print(f"Distribution parameters for {targeted_quality_index} saved to: {output_files[targeted_quality_index]}")
