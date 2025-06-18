# Riemann Experiment Pipeline: Software Design Guide

## Overview

This guide captures the refined software architecture and best practices developed through Experiment 2 for building robust, scalable mathematical experiments in the Riemann Hypothesis research pipeline. The design emphasizes modularity, reproducibility, statistical rigor, and high-quality visualization.

## Core Architecture Principles

### 1. Modular Pipeline Design
```
experiment_X/
├── experiment_X_batch.sage          # Orchestrator/Entry Point
├── experiment_X_math.sage           # Mathematical Core
├── experiment_X_stats.sage          # Statistical Analysis
├── experiment_X_viz.sage            # Visualization Engine
├── experiment_X_config.json         # Base Configuration
├── experiment_X_config_*.json       # Specialized Configs
├── generate_*_config.py             # Config Generators
├── data/                            # Data Storage
│   └── experiment_X_*.h5           # HDF5 Data Files
└── results/                         # Outputs
    ├── experiment_X_summary_*.png   # Visualizations
    └── experiment_X_summary_report.txt # Text Summary
```

### 2. Four-Layer Architecture

#### **Layer 1: Batch Orchestrator (`experiment_X_batch.sage`)**
- Single entry point for entire experiment
- Configuration management and parameter sweeps
- Progress tracking and error handling
- Coordinates execution of Math → Stats → Viz pipeline

#### **Layer 2: Mathematical Core (`experiment_X_math.sage`)**
- Contains all mathematical computations
- Implements energy functionals and perturbation analysis
- Outputs structured data to HDF5 format
- **Class-based design** for reusability

#### **Layer 3: Statistical Analysis (`experiment_X_stats.sage`)**
- Polynomial fitting and regression analysis
- Bootstrap confidence intervals
- Hypothesis testing (t-tests, p-values)
- Statistical significance assessment

#### **Layer 4: Visualization Engine (`experiment_X_viz.sage`)**
- High-quality publication-ready plots
- **Maximum 2 plots per figure** for readability
- Focused dataset summaries (exactly 5 summary images)
- Professional color schemes and typography

## Data Management Standards

### HDF5 Storage Structure
```
experiment_X_analysis.h5
├── config_1_gamma1_X_gamma2_Y/
│   ├── metadata/                    # Experiment parameters
│   ├── scheme_i/                   # Primary analysis
│   │   ├── delta                   # Perturbation values
│   │   ├── delta_E                 # Energy changes
│   │   ├── polyfit_coeffs          # [C₁, C₂] coefficients
│   │   ├── bootstrap_CI            # Confidence intervals
│   │   └── attributes              # r_squared, p_values, etc.
│   ├── scheme_ii/                  # Secondary analysis
│   ├── scheme_both/                # Combined analysis
│   └── interference_analysis/       # Cross-coupling effects
├── config_2.../
└── ...
```

### Key HDF5 Practices
- **Single file per experiment** for batch processing
- **Hierarchical group structure** by configuration
- **Comprehensive metadata** stored as attributes
- **Append mode** for incremental data addition
- **Consistent dataset naming** across configurations

## Configuration Management

### JSON Configuration Schema
```json
{
  "gamma1": 14.13,                   // Primary zero height
  "gamma2": 21.02,                   // Secondary zero height  
  "delta_range": 0.05,               // Perturbation range ±δ
  "delta_steps": 41,                 // Sampling resolution
  "test_function_type": "gaussian",   // Basis function type
  "num_test_functions": 20,          // Basis size
  "confidence_level": 0.95,          // Statistical confidence
  "bootstrap_samples": 1000,         // Bootstrap iterations
  "output_file": "experiment_X.h5",  // Data output file
  "verbose": false,                  // Logging level
  "batch_configs": [                 // Parameter sweep
    {"gamma1": X, "gamma2": Y},
    // ... more configurations
  ]
}
```

### Configuration Best Practices
- **Base configuration** + **batch overrides** pattern
- **Specialized config files** for different scales (small/large/first_100)
- **Python generators** for systematic parameter spaces
- **Validation** of parameter ranges and constraints

## Mathematical Implementation Standards

### Class Structure Template
```sage
class ExperimentXMath:
    def __init__(self, config_file):
        # Load and validate configuration
        # Initialize mathematical parameters
        
    def energy_functional(self, parameters):
        # Core mathematical computation
        # Return numerical results
        
    def perturbation_analysis(self, delta_values):
        # Systematic perturbation study
        # Return energy changes
        
    def run_analysis(self):
        # Orchestrate full mathematical pipeline
        # Return structured results dictionary
        
    def write_to_hdf5(self, results):
        # Structured data storage with metadata
        # Group-based organization
```

### Numerical Precision Requirements
- **Float64 precision** for all computations
- **Gradient computations** using `np.gradient()`
- **Error handling** for numerical instabilities
- **Baseline subtraction** for relative energy changes
- **Convergence validation** for iterative methods

## Statistical Analysis Framework

### Core Statistical Methods
1. **Quadratic Fitting**: `delta_E = C₁·δ + C₂·δ²`
2. **Bootstrap Confidence Intervals** (1000 samples minimum)
3. **Hypothesis Testing**: One-sided t-tests for stability (C₁ > 0)
4. **R² Fit Quality Assessment**
5. **Cross-coupling Analysis**: C₁₂ = C₁(both) - C₁(γ₁) - C₁(γ₂)

### Statistical Output Requirements
- **Coefficients with standard errors**
- **95% confidence intervals**
- **P-values for all hypotheses**
- **Effect size measurements**
- **Summary statistics** across configurations

## Visualization Design Standards

### Figure Organization
```sage
# RULE: Maximum 2 subplots per figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Professional styling
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Consistent palette
fig.suptitle('Clear, Descriptive Title', fontsize=18, fontweight='bold')
ax.set_xlabel('Parameter', fontsize=14)
ax.grid(True, alpha=0.3)
```

### Five Summary Images Standard
1. **Image 1**: Stability Analysis (C₁ coefficients + confidence intervals)
2. **Image 2**: Fit Quality Assessment (R² values + distribution)
3. **Image 3**: Interference Patterns (Cross-coupling effects)
4. **Image 4**: Cross-Coupling Analysis (C₁₂ vs separation)
5. **Image 5**: Parameter Space Coverage (γ₁, γ₂ distribution)

### Visualization Best Practices
- **DPI 300** for publication quality
- **Error bars** using bootstrap confidence intervals
- **Professional color schemes** (avoid rainbow)
- **Clear legends** and **descriptive titles**
- **Grid lines** with `alpha=0.3` for readability
- **Consistent marker sizes** (6-8 points for visibility)

## Summary Report Generation

### Text Report Structure
```
EXPERIMENT X: [TITLE]
======================================================================

Analysis Timestamp: YYYY-MM-DD HH:MM:SS
Dataset: N configurations
Parameter Space: γ₁ ∈ [min, max], γ₂ ∈ [min, max]

STABILITY ANALYSIS SUMMARY:
----------------------------------------
Total Configurations: N
Stable Coefficients (C₁ > 0): N (XX.X%)
Mean C₁ Coefficient: X.XXXe+XX
Mean R² (Fit Quality): X.XXXXXX
Significant Stability (p < 0.05): N (XX.X%)

[ADDITIONAL ANALYSIS SECTIONS...]

DETAILED CONFIGURATION RESULTS:
----------------------------------------
Config                         γ₁           γ₂           C₁              [metrics...]
------------------------------ ------------ ------------ --------------- ------------
[tabular data...]

STATISTICAL SUMMARY:
----------------------------------------
Overall Assessment: [STABLE/UNSTABLE/MIXED]
Riemann Hypothesis Support: [interpretation]
Mathematical Significance: [key findings]

EXPERIMENTAL DETAILS:
----------------------------------------
Energy Functional: [mathematical description]
Test Function Basis: [basis description]
Statistical Methods: [methods used]
```

## Error Handling and Robustness

### Common Numerical Issues
1. **Matrix singularities** in least squares fitting
2. **Divide by zero** in interference calculations
3. **Memory overflow** with large parameter sweeps
4. **HDF5 file corruption** during batch processing

### Mitigation Strategies
```sage
# Robust least squares with regularization
try:
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
except np.linalg.LinAlgError:
    coeffs = np.zeros(X.shape[1])  # Fallback values

# Safe division with epsilon
ratio = numerator / (denominator + 1e-10)

# Memory-conscious batch processing
if len(batch_configs) > 1000:
    # Process in chunks
    
# HDF5 file validation
with h5py.File(filename, 'a') as f:
    if group_name in f:
        del f[group_name]  # Clean restart
```

## Performance Optimization

### Computational Efficiency
- **Vectorized operations** using NumPy
- **Minimal file I/O** (single HDF5 file)
- **Progress indicators** for long-running batches
- **Memory management** for large datasets

### Parallelization Considerations
- **SageMath thread safety** limitations
- **HDF5 concurrent write** restrictions
- **Batch chunking** for distributed processing
- **Configuration independence** for parallel execution

## Quality Assurance Checklist

### Before Running Experiments
- [ ] Configuration files validated
- [ ] Output directories exist
- [ ] HDF5 file permissions correct
- [ ] Memory/disk space sufficient
- [ ] Backup previous results

### During Execution
- [ ] Progress monitoring active
- [ ] Error logs captured
- [ ] Intermediate results validated
- [ ] Resource usage tracked

### After Completion
- [ ] Statistical significance verified
- [ ] Visualization quality checked
- [ ] Summary report generated
- [ ] Data integrity validated
- [ ] Results archived

## Common Anti-Patterns to Avoid

### ❌ Poor Practices
- **Rainbow color schemes** (hard to distinguish)
- **More than 2 plots per figure** (cluttered)
- **Hardcoded file paths** (not portable)
- **Missing error handling** (fragile pipeline)
- **Inconsistent data formats** (analysis difficulties)
- **No statistical validation** (unreliable results)

### ✅ Best Practices
- **Professional color palettes**
- **Maximum 2 subplots per figure**
- **Relative path references**
- **Comprehensive exception handling**
- **Standardized HDF5 schema**
- **Statistical significance testing**

## Extension Guidelines

### Adding New Experiment Types
1. **Copy template structure** from Experiment 2
2. **Modify mathematical core** in `_math.sage`
3. **Update statistical analysis** in `_stats.sage`
4. **Adapt visualizations** in `_viz.sage`
5. **Create specialized configs** as needed

### Scaling Considerations
- **Config generation scripts** for large parameter spaces
- **Chunked processing** for memory constraints
- **Distributed execution** strategies
- **Result aggregation** across runs

## Documentation Requirements

### Code Documentation
- **Docstrings** for all public methods
- **Inline comments** for complex mathematics
- **Configuration examples** with explanations
- **Error condition descriptions**

### Experiment Documentation
- **Mathematical background** and motivation
- **Parameter space justification**
- **Statistical methodology** explanation
- **Results interpretation** guidelines

This design guide represents the accumulated knowledge and best practices from developing a robust, scalable mathematical experiment pipeline. Following these patterns will ensure consistency, reliability, and maintainability across future experiments in the Riemann Hypothesis research program.
