# pyCoreRelator

A Python package for performing semi-automatic, quantitative stratigraphic correlation between geological drill cores. The package employs Dynamic Time Warping (DTW) algorithms to align well log data while incorporating age model constraints and lithological information to ensure geologically meaningful correlations.

## Overview

pyCoreRelator provides comprehensive tools for correlating well log data between geological cores using advanced DTW algorithms. The package features segment-based correlation analysis, age constraint compatibility checking, quality metrics computation, and rich visualization capabilities for geological correlation workflows.

## Key Features

- **Segment-based DTW Analysis**: Divide cores into analyzable segments using user-picked depth boundaries
- **Age Constraint Integration**: Apply age model constraints to filter correlations based on chronostratigraphic compatibility
- **Quality Metrics**: Comprehensive quality indicators including correlation coefficients, diagonality measures, and DTW distance metrics
- **Complete Path Finding**: Identify correlation paths spanning entire cores from top to bottom
- **Rich Visualizations**: DTW matrices, segment correlation plots, and animated correlation sequences
- **Multidimensional Support**: Handle multiple log types simultaneously with independent or joint processing
- **Robust Edge Case Handling**: Custom DTW implementation with special handling for single-point segments

## Core Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting and visualization
- `scipy`: Statistical functions
- `librosa`: DTW algorithms
- `tqdm`: Progress bars
- `joblib`: Parallel processing

## Package Structure

```
pyCoreRelator/
├── core/
│   ├── dtw_analysis.py      # DTW computation and analysis
│   ├── segment_analysis.py  # Segment identification and path finding
│   ├── quality_metrics.py  # Quality indicators computation
│   └── age_models.py        # Age constraint handling
├── utils/
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── path_processing.py   # Path analysis utilities
│   └── helpers.py           # General utility functions
└── visualization/
    ├── plotting.py          # Core plotting functions
    ├── matrix_plots.py      # DTW matrix visualizations
    └── animation.py         # Animated correlation displays
```

## Core Functions

### Data Loading
- `load_log_data()`: Load and preprocess log data from CSV files with optional image support
- `resample_datasets()`: Resample multiple datasets to common depth scales

### DTW Analysis
- `run_comprehensive_dtw_analysis()`: Main function for segment-based DTW correlation with age constraints
- `custom_dtw()`: Robust DTW implementation with edge case handling
- `compute_quality_indicators()`: Calculate correlation quality metrics

### Segment Analysis
- `find_all_segments()`: Identify correlation segments from picked depths
- `find_complete_core_paths()`: Find complete correlation paths spanning entire cores
- `diagnose_chain_breaks()`: Identify and diagnose connectivity issues

### Age Models
- `calculate_interpolated_ages()`: Interpolate ages for picked depths using age constraints
- `check_age_constraint_compatibility()`: Validate age compatibility between segments

### Visualization
- `plot_dtw_matrix_with_paths()`: Visualize DTW distance matrices with optimal paths
- `visualize_combined_segments()`: Display segment correlations on log plots
- `create_segment_dtw_animation()`: Generate animated correlation sequences

## Output Files

- **DTW Matrix Plot**: `SegmentPair_DTW_matrix.png` - Heatmap showing DTW distances between all segment pairs
- **Animation**: `SegmentPair_DTW_animation.gif` - Animated sequence of segment correlations
- **Complete Paths**: `complete_core_paths.csv` - Detailed results of complete correlation paths
- **Quality Metrics**: Embedded in results with correlation coefficients, diagonality measures, and DTW statistics

## Quality Metrics

The package computes comprehensive quality indicators for each correlation:

- Normalized DTW Distance: Cost per alignment step
  - Lower values indicate better alignment quality
  - Calculated as total DTW cost divided by path length
- DTW Ratio: DTW distance relative to Euclidean distance
  - DTW ratio < 1.0: Better DTW alignment than linear mapping
  - DTW ratio ≈ 1.0: Similar to linear alignment (already well-aligned sequences)
  - DTW ratio > 1.0: Linear alignment performs better than DTW match
- Correlation Coefficient (Pearson's r value): Linear correlation between aligned sequences
  - Values range from -1 to 1 (1 = perfect positive correlation, 0 = no correlation, -1 = perfect negative correlation)
  - Calculated using linear regression on DTW-aligned data points
- Diagonality Percentage: Measure of path straightness (higher = better)
  - 100% indicates perfect diagonal path (minimal warping)
  - Lower values suggest more complex warping patterns
- Variance Deviation: Warping path deviation from diagonal
  - Measures how much the DTW path deviates from a straight diagonal
  - Higher values indicate more complex temporal alignments
- Age Overlap Percentage: Chronostratigraphic compatibility (when age constraints applied)
  - Percentage of overlap between age intervals of correlated segments
  - 100% indicates perfect chronological agreement, 0% means no temporal overlap

## License

**pyCoreRelator** is licensed under the Apache License 2.0.

## Requirements

The following Python packages are required to run pyCoreRelator:

- `numpy>=1.20.0` - Numerical computations and array operations
- `pandas>=1.3.0` - Data manipulation and analysis
- `matplotlib>=3.5.0` - Plotting and visualization
- `scipy>=1.7.0` - Scientific computing and statistical functions
- `librosa>=0.9.0` - Audio and signal processing (used for DTW algorithms)
- `tqdm>=4.60.0` - Progress bars for long-running operations
- `joblib>=1.1.0` - Parallel processing and caching
- `Pillow>=8.3.0` - Image processing capabilities
- `IPython>=7.25.0` - Interactive Python environment features
- `psutil>=5.8.0` - System and process utilities

### Installation

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

Or install individual packages:

```bash
pip install numpy>=1.20.0 pandas>=1.3.0 matplotlib>=3.5.0 scipy>=1.7.0 librosa>=0.9.0 tqdm>=4.60.0 joblib>=1.1.0 Pillow>=8.3.0 IPython>=7.25.0 psutil>=5.8.0
```
