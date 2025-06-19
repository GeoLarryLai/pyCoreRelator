# pyCoreRelator

A Python package for quantitative stratigraphic correlation between geological drill cores using Dynamic Time Warping (DTW) algorithms. The package performs segment-based correlation analysis with age model constraints and comprehensive quality assessment for geological core correlation workflows.

## Overview

pyCoreRelator enables automated correlation of well log data between geological cores using advanced DTW algorithms. It provides segment-based analysis, age constraint integration, quality metrics computation, null hypothesis testing, and rich visualization capabilities designed specifically for geological correlation studies.

## Key Features

- **Segment-Based DTW Correlation**: Divide cores into analyzable segments using user-picked depth boundaries
- **Age Model Integration**: Apply chronostratigraphic constraints to filter correlations based on age compatibility
- **Quality Assessment**: Comprehensive quality metrics including correlation coefficients, diagonality measures, and DTW statistics
- **Complete Path Finding**: Identify correlation paths spanning entire cores from top to bottom
- **Null Hypothesis Testing**: Generate synthetic cores and test correlation significance
- **Multidimensional Log Support**: Handle multiple log types (MS, CT, RGB, density) simultaneously
- **Rich Visualizations**: DTW matrices, segment correlations, animated sequences, and diagnostic plots
- **Edge Case Handling**: Robust DTW implementation for single-point segments and irregular data

## Installation

### Requirements
Python 3.7+ with the following packages:

```bash
pip install -r requirements.txt
```

### Package Structure

```
pyCoreRelator/
├── core/                    # Core analysis modules
│   ├── dtw_analysis.py      # DTW computation and comprehensive analysis
│   ├── segment_analysis.py  # Segment identification and correlation
│   ├── segment_operations.py # Segment manipulation utilities
│   ├── path_finding.py      # Complete path discovery algorithms
│   ├── quality_metrics.py   # Quality indicators computation
│   ├── age_models.py        # Age constraint handling and interpolation
│   ├── diagnostics.py       # Chain break analysis and debugging
│   ├── null_hypothesis.py   # Synthetic data generation and testing
│   └── path_helpers.py      # Path processing utilities
├── utils/                   # Data handling utilities
│   ├── data_loader.py       # Multi-format data loading with image support
│   ├── path_processing.py   # Path analysis and manipulation
│   └── helpers.py           # General utility functions
└── visualization/           # Plotting and visualization
    ├── plotting.py          # Core plotting functions and segment visualization
    ├── matrix_plots.py      # DTW matrix heatmaps and path overlays
    ├── animation.py         # Animated correlation sequences
    └── core_plots.py        # Basic core data visualization
```

## Quick Start

### Basic Workflow

```python
import pyCoreRelator as pcr

# 1. Load well log data
log_a, md_a, columns_a, rgb_img_a, ct_img_a = pcr.load_log_data(
    core_a_paths, image_paths_a, ['hiresMS', 'CT'], normalize=True
)
log_b, md_b, columns_b, rgb_img_b, ct_img_b = pcr.load_log_data(
    core_b_paths, image_paths_b, ['hiresMS', 'CT'], normalize=True
)

# 2. Run comprehensive DTW analysis
dtw_results, valid_pairs, segments_a, segments_b, boundaries_a, boundaries_b, dtw_matrix = pcr.run_comprehensive_dtw_analysis(
    log_a, log_b, md_a, md_b,
    picked_depths_a=picked_depths_a,
    picked_depths_b=picked_depths_b,
    age_consideration=True,
    ages_a=age_constraints_a,
    ages_b=age_constraints_b
)

# 3. Find complete correlation paths
complete_paths_csv = pcr.find_complete_core_paths(
    valid_pairs, segments_a, segments_b, log_a, log_b,
    boundaries_a, boundaries_b, dtw_results
)

# 4. Visualize results
pcr.visualize_combined_segments(log_a, log_b, md_a, md_b, dtw_results, valid_pairs)
pcr.plot_dtw_matrix_with_paths(dtw_matrix, valid_pairs, segments_a, segments_b)
```

### Null Hypothesis Testing

```python
# Generate synthetic core pairs for significance testing
segment_pool = pcr.load_segment_pool(log_a, log_b, md_a, md_b, picked_depths_a, picked_depths_b)
synthetic_log, synthetic_md = pcr.create_synthetic_log_with_depths(segment_pool, target_length=500)
pcr.create_and_plot_synthetic_core_pair(segment_pool, show_images=True)
```

## Core Functions

### Data Loading and Preprocessing
- **`load_log_data()`**: Load multi-column log data with optional image support and normalization
- **`plot_core_data()`**: Visualize core data with multiple log curves and images

### DTW Analysis
- **`run_comprehensive_dtw_analysis()`**: Main function for segment-based DTW with age constraints
- **`compute_quality_indicators()`**: Calculate correlation quality metrics and statistics
- **`calculate_interpolated_ages()`**: Interpolate ages for depth boundaries using age models

### Segment Operations
- **`find_all_segments()`**: Identify correlation segments from user-picked depths
- **`find_complete_core_paths()`**: Discover complete correlation paths spanning entire cores
- **`diagnose_chain_breaks()`**: Identify and analyze connectivity gaps in correlation chains

### Null Hypothesis Testing
- **`load_segment_pool()`**: Create pools of real segments for synthetic core generation
- **`create_synthetic_log_with_depths()`**: Generate synthetic cores from segment pools
- **`create_and_plot_synthetic_core_pair()`**: Create and visualize synthetic core pairs

### Visualization
- **`plot_dtw_matrix_with_paths()`**: Visualize DTW distance matrices with correlation paths
- **`visualize_combined_segments()`**: Display segment correlations overlaid on log plots
- **`visualize_dtw_results_from_csv()`**: Generate animated correlation sequences from results
- **`plot_correlation_distribution()`**: Analyze and plot correlation quality distributions

## Quality Metrics

The package computes comprehensive quality indicators for each correlation:

- **Normalized DTW Distance**: Cost per alignment step (lower = better alignment)
- **DTW Ratio**: DTW distance relative to Euclidean distance (<1.0 = DTW improves alignment)
- **Correlation Coefficient**: Pearson's r between aligned sequences (-1 to 1)
- **Diagonality Percentage**: Path straightness measure (100% = perfect diagonal)
- **Variance Deviation**: Warping path deviation from diagonal
- **Age Overlap Percentage**: Chronostratigraphic compatibility when age constraints applied

## Output Files

- **DTW Matrix**: `SegmentPair_DTW_matrix.png` - Heatmap of segment-to-segment distances
- **Animations**: `SegmentPair_DTW_animation.gif` - Step-by-step correlation sequences  
- **Complete Paths**: `complete_core_paths.csv` - Detailed results of end-to-end correlations
- **Diagnostic Plots**: Various PNG files for quality assessment and troubleshooting

## Use Cases

- **Turbidite Correlation**: Correlate event beds between marine sediment cores
- **Paleoclimate Studies**: Align proxy records between drilling sites
- **Hydrocarbon Exploration**: Correlate well logs across petroleum reservoirs
- **Stratigraphic Analysis**: Quantitative correlation with age model constraints
- **Method Development**: Test correlation algorithms using synthetic data

## Example Applications

The package includes several Jupyter notebooks demonstrating real-world applications:

- `pyCoreRelator_test.ipynb`: Comprehensive workflow with Cascadia margin turbidite cores
- `LSHL_Cascadia_turbidite_correlation.ipynb`: Large-scale multi-core correlation project
- `pyCoreRelator_null_hypothesis.ipynb`: Synthetic data generation and significance testing
- `Core_depthboundary_picker.ipynb`: Interactive depth boundary selection tool

## License

pyCoreRelator is licensed under the Apache License 2.0.

## Documentation

Detailed function documentation is available in `FUNCTION_DOCUMENTATION.md`.
