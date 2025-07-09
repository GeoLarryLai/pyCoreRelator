# pyCoreRelator

<div align="center">
  <img src="pyCoreRelator_logo.png" alt="pyCoreRelator Logo" width="200"/>
</div>

**pyCoreRelator** is a Python package for quantitative stratigraphic correlation between geological core data, leveraging Dynamic Time Warping (DTW) algorithms for automatic signal alignment and considering stratigraphic principles (e.g., superposition, age succession, pinchouts, etc.). The package performs segment-based correlation analysis with age model constraints and comprehensive quality assessment for geological core correlation workflows.

## Key Features

- **Segment-Based DTW Correlation**: Divide cores into analyzable segments using user-picked depth boundaries, enabling controls on the stratigraphic pinchouts or forced correlation datums.
- **Age Constraints Integration**: Apply chronostratigraphic constraints to filter correlations based on age compatibility
- **Quality Assessment**: Comprehensive quality metrics including correlation coefficients, diagonality measures, and DTW statistics
- **Complete Path Finding**: Identify correlation paths spanning entire cores from top to bottom with advanced optimization
- **Null Hypothesis Testing**: Generate synthetic cores and test correlation significance with multi-parameter analysis
- **Multidimensional Log Support**: Handle multiple log types (MS, CT, RGB, density) simultaneously with independent or dependent DTW
- **Rich Visualizations**: DTW matrices, segment correlations, animated sequences, and diagnostic plots with statistical analysis
- **Performance Optimization**: Memory-efficient path finding with database storage, parallel processing, and path pruning

## Requirements

Python 3.7+ with the following packages:

**Core Dependencies:**
- `numpy>=1.20.0` - Numerical computing and array operations
- `pandas>=1.3.0` - Data manipulation and analysis
- `scipy>=1.7.0` - Scientific computing and optimization
- `matplotlib>=3.5.0` - Plotting and visualization
- `Pillow>=8.3.0` - Image processing
- `librosa>=0.9.0` - Audio/signal processing for DTW algorithms
- `tqdm>=4.60.0` - Progress bars
- `joblib>=1.1.0` - Parallel processing
- `IPython>=7.25.0` - Interactive environment support
- `psutil>=5.8.0` - System utilities and memory monitoring

**Optional Dependencies for Notebooks:**
- `opencv-python>=4.5.0` - Computer vision tasks (optional)
- `scikit-image>=0.18.0` - Advanced image processing (optional)

**Installation:**
```bash
pip install -r requirements.txt
```

## Package Structure

```
pyCoreRelator/
├── core/                    # Core analysis modules
│   ├── dtw_analysis.py      # DTW computation and comprehensive analysis
│   ├── segment_operations.py # Segment identification and manipulation
│   ├── segment_analysis.py  # Legacy segment analysis functions
│   ├── path_finding.py      # Complete path discovery algorithms
│   ├── quality_metrics.py   # Quality indicators computation
│   ├── age_models.py        # Age constraint handling and interpolation
│   ├── diagnostics.py       # Chain break analysis and debugging
│   ├── null_hypothesis.py   # Synthetic data generation and multi-parameter testing
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

## Core Functions

Detailed function documentation is available in [FUNCTION_DOCUMENTATION.md](FUNCTION_DOCUMENTATION.md).

### Data Loading and Preprocessing
- **`load_log_data()`**: Load multi-column log data with optional image support and normalization
- **`load_core_age_constraints()`**: Load age constraint data from CSV files with support for adjacent cores
- **`plot_core_data()`**: Visualize core data with multiple log curves and images

### DTW Analysis and Path Finding
- **`run_comprehensive_dtw_analysis()`**: Main function for segment-based DTW with age constraints and visualization
- **`find_complete_core_paths()`**: Advanced complete path discovery with memory optimization and path pruning
- **`compute_quality_indicators()`**: Calculate correlation quality metrics and statistics
- **`calculate_interpolated_ages()`**: Interpolate ages for depth boundaries using age models with uncertainty propagation

### Segment Operations and Diagnostics
- **`find_all_segments()`**: Identify correlation segments from user-picked depths with enhanced connectivity
- **`build_connectivity_graph()`**: Build segment relationship graphs for path analysis
- **`identify_special_segments()`**: Classify segments as top, bottom, dead-end, or orphaned
- **`diagnose_chain_breaks()`**: Identify and analyze connectivity gaps in correlation chains

### Multi-Parameter and Null Hypothesis Testing
- **`run_multi_parameter_analysis()`**: Comprehensive analysis across parameter combinations with statistical testing
- **`load_segment_pool()`**: Create pools of real segments for synthetic core generation
- **`create_synthetic_log_with_depths()`**: Generate synthetic cores from segment pools
- **`create_and_plot_synthetic_core_pair()`**: Create and visualize synthetic core pairs

### Advanced Visualization and Analysis
- **`plot_dtw_matrix_with_paths()`**: Visualize DTW distance matrices with correlation paths and color-coded metrics
- **`visualize_combined_segments()`**: Display segment correlations overlaid on log plots
- **`visualize_dtw_results_from_csv()`**: Generate animated correlation sequences from results
- **`plot_correlation_distribution()`**: Analyze and plot correlation quality distributions with statistical fitting
- **`plot_quality_distributions()`**: Compare real data distributions against null hypothesis with statistical tests
- **`find_best_mappings()`**: Identify optimal correlation mappings using weighted quality metrics

## Correlation Quality Assessment

The package computes comprehensive quality indicators for each correlation with enhanced statistical analysis:

### Core Quality Metrics
- **Normalized DTW Distance**: Cost per alignment step normalized by path length (lower = better alignment)
- **DTW Warping Ratio**: DTW distance relative to Euclidean distance (<1.0 indicates DTW improves alignment)
- **DTW Warping Efficiency**: Efficiency measure combining path length and alignment quality (higher = better)
- **Correlation Coefficient**: Pearson's r between aligned sequences (-1 to 1, higher = better)
- **Diagonality Percentage**: Path straightness measure (100% = perfect diagonal alignment)
- **Variance Deviation**: Warping path deviation from diagonal (lower = more consistent alignment)
- **Age Overlap Percentage**: Chronostratigraphic compatibility when age constraints applied (higher = better)

### Advanced Quality Analysis
- **Statistical Distribution Fitting**: Normal, skew-normal, and KDE fitting of quality metric distributions
- **Null Hypothesis Testing**: Comparison against synthetic data with t-statistics and effect sizes
- **Multi-Parameter Analysis**: Systematic evaluation across age constraint combinations
- **Quality Metric Weighting**: Customizable weighted scoring for optimal correlation identification
- **Percentile Ranking**: Statistical ranking of correlations within distribution context

### Quality Metric Interpretation
- **Normalized DTW Distance**: Values < 0.1 indicate excellent alignment, > 0.5 suggest poor correlation
- **Correlation Coefficient**: Values > 0.7 indicate strong correlation, < 0.3 suggest weak relationships
- **Diagonality**: Values > 80% indicate good stratigraphic consistency, < 60% suggest complex warping
- **Age Overlap**: Values > 70% indicate good chronostratigraphic compatibility

## Output Files and Results

### Analysis Results
- **Complete Paths CSV**: `complete_core_paths.csv` - Detailed results of end-to-end correlations with quality metrics
- **DTW Results Database**: Temporary SQLite databases for memory-efficient path storage during analysis
- **Quality Distribution Data**: Statistical parameters and histogram data for null hypothesis testing

### Visualization Outputs
- **DTW Matrix Plots**: `SegmentPair_DTW_matrix.png` - Heatmap of segment-to-segment distances with color-coded paths
- **Correlation Animations**: `SegmentPair_DTW_animation.gif` - Step-by-step correlation sequences
- **Combined Correlation Plots**: Static visualization of optimal correlation mappings
- **Statistical Distribution Plots**: Quality metric distributions with fitted curves and significance tests
- **Diagnostic Plots**: Various PNG files for quality assessment and troubleshooting

## Example Applications

The package includes several Jupyter notebooks demonstrating real-world applications:

- **`pyCoreRelator_test.ipynb`**: Comprehensive workflow with Cascadia margin turbidite cores showing full analysis pipeline
- **`pyCoreRelator_test_comparedtoNull.ipynb`**: Advanced comparison against null hypothesis with multi-parameter analysis
- **`pyCoreRelator_null_hypothesis.ipynb`**: Synthetic data generation and significance testing examples
- **`pyCoreRelator_depthboundary_picker.ipynb`**: Interactive depth boundary selection tool

## Performance and Optimization

The package includes several performance enhancements for large-scale analysis:

- **Memory Management**: SQLite-based path storage with automatic cleanup and garbage collection
- **Path Pruning**: Intelligent pruning of intermediate paths to prevent memory overflow
- **Parallel Processing**: Multi-core support for batch processing and quality metric computation
- **Shortest Path Search**: Optimized algorithms that focus on most promising correlation paths
- **Database Optimization**: Indexed database operations with WAL mode for efficient read/write operations

## License

**pyCoreRelator** is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
