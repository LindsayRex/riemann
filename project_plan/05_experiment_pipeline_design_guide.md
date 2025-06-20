# Experiment Pipeline Design Guide

## Overview

Software architecture and implementation patterns for mathematical experiments in the Riemann Hypothesis research pipeline. This guide documents the modular design, data management, and analysis patterns developed across three experiments.

## Core Architecture

### Directory Structure
```
experiment_X/
├── experiment_X_batch.sage          # Entry point and orchestrator
├── experiment_X_math.sage           # Mathematical computations
├── experiment_X_stats.sage          # Statistical analysis
├── experiment_X_viz.sage            # Visualization generation
├── experiment_X_config*.json        # Configuration files
├── generate_*_config.py             # Configuration generators
├── data/                            # HDF5 data storage
│   └── experiment_X_*.h5           # Numerical results
└── results/                         # Analysis outputs
    ├── experiment_X_summary_*.png   # Key visualizations
    └── experiment_X_summary_*.txt   # Statistical summaries
```

### Processing Pipeline
1. **Configuration**: JSON-based parameter specification
2. **Mathematics**: Core energy functional computations
3. **Statistics**: Regression analysis and hypothesis testing  
4. **Visualization**: Summary plots and statistical dashboards
5. **Storage**: Structured HDF5 data with metadata

### File Naming Convention
The batch orchestrator relies on specific naming patterns to locate and execute pipeline components:

```sage
# Required naming pattern for batch system
experiment_X_batch.sage    # Loads experiment_X_math.sage
experiment_X_math.sage     # Loads experiment_X_stats.sage  
experiment_X_stats.sage    # Loads experiment_X_viz.sage
experiment_X_viz.sage      # Final visualization step
```

**Critical**: The `experiment_X_batch.sage` file uses string interpolation to construct module names:
- `load("experiment_X_math.sage")` where X matches the experiment number
- Each component assumes the next component follows this naming pattern
- Configuration files must also follow `experiment_X_config*.json` pattern

Breaking this naming convention will cause the batch system to fail during module loading.

## Data Storage

### HDF5 Structure
```
experiment_X_analysis.h5
├── config_1_gamma1_X_gamma2_Y/
│   ├── metadata/                    # Experiment parameters
│   ├── perturbation_analysis/       # Primary results
│   │   ├── delta                   # Perturbation values
│   │   ├── delta_E                 # Energy changes
│   │   ├── polyfit_coeffs          # Regression coefficients
│   │   ├── bootstrap_CI            # Confidence intervals
│   │   └── attributes              # R², p-values, etc.
│   └── interference_analysis/       # Cross-coupling effects
├── config_2.../
└── ...
```

### Storage Practices
- Single HDF5 file per experiment
- Hierarchical organization by configuration
- Comprehensive metadata as attributes
- Append mode for incremental processing

## Configuration Management

### JSON Schema
```json
{
  "gamma1": 14.13,
  "gamma2": 21.02,
  "delta_range": 0.05,
  "delta_steps": 41,
  "test_function_type": "gaussian",
  "num_test_functions": 20,
  "confidence_level": 0.95,
  "bootstrap_samples": 1000,
  "output_file": "experiment_X.h5",
  "batch_configs": [
    {"gamma1": X, "gamma2": Y},
    // ... additional configurations
  ]
}
```

### Configuration Patterns
- Base configuration with batch parameter overrides
- Specialized configs for different experiment scales
- Python generators for systematic parameter spaces
- Parameter validation and constraint checking

## Implementation Patterns

### Mathematical Core Structure
```sage
class ExperimentXMath:
    def __init__(self, config_file):
        # Load configuration and initialize parameters
        
    def energy_functional(self, parameters):
        # Core mathematical computation
        
    def perturbation_analysis(self, delta_values):
        # Systematic perturbation study
        
    def run_analysis(self):
        # Execute full mathematical pipeline
        
    def write_to_hdf5(self, results):
        # Store results with metadata
```

### Statistical Analysis Framework
- Quadratic fitting: `delta_E = C₁·δ + C₂·δ²`
- Bootstrap confidence intervals (1000+ samples)
- Hypothesis testing for stability (C₁ > 0)
- R² fit quality assessment
- Cross-coupling analysis when applicable

### Visualization Standards
- Maximum 2 subplots per figure
- Consistent color schemes
- Professional typography and formatting
- Error bars with confidence intervals
- 300 DPI for publication quality

## Report Generation

### Analysis Report Structure
- Configuration summary and parameter space
- Statistical results with confidence intervals
- Stability assessment across configurations
- Key findings and mathematical significance
- Tabulated results for all configurations

### Cross-Experiment Analysis
Located in `analysis/` directory:
- `generate_markdown.py` - Comprehensive report generation
- `generate_pdf.py` - PDF with rendered mathematics
- Image selection from experiment results
- Unified analysis across all experiments

## Error Handling

### Common Issues
- Matrix singularities in regression fitting
- Division by zero in interference calculations
- Memory constraints with large parameter sweeps
- HDF5 file corruption during processing

### Mitigation Strategies
```sage
# Robust regression with fallbacks
try:
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
except np.linalg.LinAlgError:
    coeffs = np.zeros(X.shape[1])

# Safe division
ratio = numerator / (denominator + 1e-10)

# Memory management for large batches
if len(batch_configs) > threshold:
    # Process in chunks
```

## Extension Guidelines

### Adding New Experiments
1. Copy existing experiment structure
2. Modify mathematical core for new computations
3. Update statistical analysis for experiment-specific metrics
4. Adapt visualizations for new data types
5. Create appropriate configuration templates

### Scaling Considerations
- Configuration generators for large parameter spaces
- Chunked processing for memory constraints
- Result aggregation across multiple runs
- Distributed execution strategies

This design provides a reusable framework for mathematical experiments requiring systematic parameter exploration, statistical analysis, and comprehensive reporting.
