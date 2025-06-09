# pyCoreRelator

A Python package for semi-automatic quantitative stratigraphic correlation between geological cores using Dynamic Time Warping (DTW) with age constraint integration.

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

- **Normalized DTW Distance**: Cost per alignment step
- **DTW Ratio**: DTW distance relative to Euclidean distance
- **Correlation Coefficient**: Linear correlation between aligned sequences
- **Diagonality Percentage**: Measure of path straightness (higher = better)
- **Variance Deviation**: Warping path deviation from diagonal
- **Age Overlap Percentage**: Chronostratigraphic compatibility (when age constraints applied)

## License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.
