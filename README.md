# pyCoreRelator

<div align="center">
  <img src="pyCoreRelator_logo.png" alt="pyCoreRelator Logo" width="200"/>
</div>

**pyCoreRelator** is a Python package for quantitative stratigraphic correlation between geological core data, leveraging Dynamic Time Warping (DTW) algorithms for automatic signal alignment and considering stratigraphic principles (e.g., superposition, age succession, pinchouts, etc.). The package performs segment-based correlation analysis with age model constraints and comprehensive quality assessment for geological core correlation workflows.

## Key Features

- **Segment-Based DTW Correlation**: Divide cores into analyzable segments using user-picked or machine-learning based (future feature) depth boundaries, enabling controls on the stratigraphic pinchouts or forced correlation datums
- **Interactive Core Datum Picking**: Manual stratigraphic boundary picking with real-time visualization, category-based classification, and CSV export for quality control
- **Age Constraints Integration**: Apply chronostratigraphic constraints to search the optimal correlation solutions
- **Quality Assessment**: Compute metrics for the quality of correlation and optimal solution search.
- **Complete DTW Path Finding**: Identify correlation DTW paths spanning entire cores from top to bottom
- **Null Hypothesis Testing**: Generate synthetic cores and test correlation significance with multi-parameter analysis
- **Log Data Cleaning & Processing**: Convert core images (CT scans, RGB photos) to digital log data with capabilities of automated brightness/color profile extraction, image alignment & stitching
- **Machine Learning Data Imputation**: Advanced ML-based gap filling for core log data using ensemble methods (Random Forest, XGBoost, LightGBM) with configurable feature weighting and trend constraints
- **Multi-dimensional Log Support**: Handle multiple log types (MS, CT, RGB, density) simultaneously with dependent or independent multi-dimentiaonl DTW approach
- **Visualizations**: DTW cost matrix and paths, segment-wise core correlations, animated sequences, and statistical analysis for the correlation solutions

## Requirements

Python 3.9+ with the following packages:

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
- `pydicom>=2.3.0` - Image processing for CT scan DICOM files
- `opencv-python>=4.5.0` - Computer vision and image processing

**Machine Learning Dependencies:**
- `scikit-learn>=1.0.0` - Machine learning algorithms and preprocessing
- `xgboost>=1.6.0` - XGBoost gradient boosting framework
- `lightgbm>=3.3.0` - LightGBM gradient boosting framework

**Optional Dependencies:**
- `ipympl>=0.9.0` - Interactive matplotlib widgets for depth picking functions (for Jupyter notebooks)
- `scikit-image>=0.18.0` - Advanced image processing features

**Installation:**
```bash
pip install -r requirements.txt
```

## Package Structure

```
pyCoreRelator/
├── core/                          
│   ├── dtw_analysis.py            # DTW computation & comprehensive analysis
│   ├── segment_operations.py      # Segment identification & manipulation
│   ├── segment_analysis.py        # Legacy segment analysis functions
│   ├── path_finding.py            # Complete DTW path discovery algorithms
│   ├── quality_metrics.py         # Quality indicators computation
│   ├── age_models.py              # Age constraint handling & interpolation
│   ├── diagnostics.py             # Chain break analysis & debugging
│   ├── null_hypothesis.py         # Synthetic data generation & multi-parameter testing
│   └── path_helpers.py            # DTW path processing utilities
├── log/                           
│   ├── rgb_image2log.py           # RGB image processing & color profile extraction
│   ├── ct_image2log.py            # CT image processing & brightness analysis
│   ├── core_datum_picker.py       # Interactive core boundary picking & depth selection
│   └── ml_log_data_imputation.py  # ML-based data gap filling
├── utils/                         
│   ├── data_loader.py             # Multi-format data loading with image support
│   ├── path_processing.py         # DTW path analysis & manipulation
│   └── helpers.py                 # General utility functions
└── visualization/                 
    ├── plotting.py                # Core plotting functions & segment visualization
    ├── matrix_plots.py            # DTW matrix & path overlays
    ├── animation.py               # Animated correlation sequences
    └── core_plots.py              # Basic core data visualization
```



## Correlation Quality Assessment

The package computes comprehensive quality indicators for each correlation with enhanced statistical analysis:

### Core Quality Metrics
- **Normalized DTW Distance**: Normalized DTW cost per alignment
- **DTW Warping Ratio**: DTW distance relative to Euclidean distance
- **DTW Warping Efficiency**: Efficiency measure combining DTW path length and alignment quality
- **Correlation Coefficient**: Pearson's r between DTW aligned sequences
- **Diagonality Percentage**: 100% = perfect diagonal alignment in the DTW matrix
- **Age Overlap Percentage**: Chronostratigraphic compatibility when age constraints applied

## Example Applications

The package includes several Jupyter notebooks demonstrating real-world applications:

### Correlation analysis
- **`pyCoreRelator_core_pair_analysis.ipynb`**: Comprehensive workflow with Cascadia margin turbidite cores showing full analysis pipeline
- **`pyCoreRelator_null_hypothesis.ipynb`**: Synthetic data generation and significance testing examples
- **`pyCoreRelator_compare2Null.ipynb`**: Advanced comparison against null hypothesis with multi-parameter analysis

### Log data processing
- **`pyCoreRelator_datum_picker.ipynb`**: Interactive stratigraphic boundary picking with real-time visualization and category-based classification
- **`pyCoreRelator_data_gap_fill.ipynb`**: Machine learning-based data processing and gap filling for core log data
- **`pyCoreRelator_RGB_image2log.ipynb`**: Processing, stitching, and converting RGB core images into RGB color logs
- **`pyCoreRelator_CT_image2log.ipynb`**: Processing, stiching, and converting CT scan images into CT intensity (brightness) logs

## Core Functions

Detailed function documentation is available in [FUNCTION_DOCUMENTATION.md](FUNCTION_DOCUMENTATION.md).

### Main Analysis Functions
- **`run_comprehensive_dtw_analysis()`**: Main function for segment-based DTW with age constraints and visualization
- **`find_complete_core_paths()`**: Advanced complete DTW path discovery with memory optimization
- **`calculate_interpolated_ages()`**: Interpolate ages for depth boundaries using age models with uncertainty propagation
- **`diagnose_chain_breaks()`**: Identify and analyze connectivity gaps in correlation chains
- **`run_multi_parameter_analysis()`**: Comprehensive analysis across parameter combinations with statistical testing
- **`find_best_mappings()`**: Identify optimal correlation mappings using weighted quality metrics

### Data Loading and Visualization
- **`load_log_data()`**: Load multi-column log data with optional image support and normalization
- **`load_core_age_constraints()`**: Load age constraint data from CSV files with support for adjacent cores
- **`plot_core_data()`**: Visualize core data with multiple log curves and images

### Machine Learning Data Imputation Functions
- **`preprocess_core_data()`**: Clean and preprocess core data with configurable thresholds and scaling
- **`plot_core_logs()`**: Visualize core logs with configurable parameters and multiple data types
- **`process_and_fill_logs()`**: Complete ML-based gap filling workflow for all configured log types

### Null Hypothesis Testing Functions
- **`load_segment_pool()`**: Create pools of real segments for synthetic core generation
- **`plot_segment_pool()`**: Visualize all segments from the turbidite database pool
- **`print_segment_pool_summary()`**: Print summary statistics for the segment pool
- **`create_and_plot_synthetic_core_pair()`**: Create and visualize synthetic core pairs
- **`create_synthetic_log_with_depths()`**: Generate synthetic cores from segment pools

### Visualization Functions
- **`visualize_combined_segments()`**: Display segment correlations overlaid on log plots
- **`visualize_dtw_results_from_csv()`**: Generate animated correlation sequences from results
- **`plot_dtw_matrix_with_paths()`**: Visualize DTW cost matrices with correlation paths
- **`plot_correlation_distribution()`**: Visualize and statistically analyze the distributions of the correlation quality metrics
- **`calculate_quality_comparison_t_statistics()`**: Calculate t-statistics for quality metric comparisons
- **`plot_quality_comparison_t_statistics()`**: Plot quality metric comparison results with statistical analysis

### Interactive Core Analysis Functions
- **`pick_stratigraphic_levels()`**: Interactive manual stratigraphic boundary picking with real-time visualization

### Image Processing Functions
- **`plot_rgb_profile()`**: Create comprehensive RGB analysis visualizations
- **`stitch_core_sections()`**: Stitch multiple core section images and RGB profiles
- **`display_slice_bt_std()`**: Display CT slices with brightness traces and standard deviation plots
- **`process_and_stitch_segments()`**: Complete workflow for multi-segment CT processing

## License

**pyCoreRelator** is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
