# Universal Critical Restoration Conjecture

A computational investigation of the Riemann Hypothesis through energy minimization and variational methods.

## Overview

This repository contains experimental mathematics code testing the **Universal Critical Restoration conjecture** - a novel reformulation of the Riemann Hypothesis as an energy minimization problem. Instead of proving where zeros must lie, we treat zero configurations as physical systems and measure the energy changes when zeros are perturbed from the critical line.

## Project Documentation

- **[Research Background](reserach_background.md)** - Complete mathematical background and motivation for the energy-based approach
- **[Project Plan](project_plan/)** - Design documents and technical specifications
  - [L-Function Zero Energy Functional](project_plan/L_Function_Zero_Energy_Functional.md)
  - [Experiment Pipeline Design Guide](project_plan/Riemann_Experiment_Pipeline_Design_Guide.md)

## Experiments

**Important:** Ensure the conda environment is activated before running experiments:
```bash
conda activate riemann-experiments
```

The project consists of three complementary experiments testing different aspects of the Universal Critical Restoration conjecture:

### Experiment 1: Single-Zero Perturbation
Tests local stability by perturbing individual zeros and measuring quadratic energy behavior.

```bash
# Run with default configuration
sage experiment1/experiment1_batch.sage experiment1/experiment1_config.json

# Run individual components
sage experiment1/experiment1_math.sage
sage experiment1/experiment1_stats.sage
sage experiment1/experiment1_viz.sage
```

### Experiment 2: Two-Zero Interaction
Analyzes interference effects and additivity properties in two-zero systems.

```bash
# Run batch analysis
sage experiment2/experiment2_batch.sage experiment2/experiment2_config.json

# Individual components
sage experiment2/experiment2_math.sage
sage experiment2/experiment2_stats.sage  
sage experiment2/experiment2_viz.sage
```

### Experiment 3: Multi-Zero Scaling
Tests scaling laws and universal stability across large multi-zero configurations.

```bash
# Small-scale test (6 configs)
sage experiment3/experiment3_batch.sage experiment3/experiment3_config.json

# Medium-scale test (10 configs) 
sage experiment3/experiment3_batch.sage experiment3/experiment3_config_phase3.json

# Large-scale test (486 configs)
sage experiment3/experiment3_batch.sage experiment3/experiment3_config_phase3_full.json
```

## Requirements

- **Conda/Miniconda** for environment management
- **SageMath 10.6+** with HDF5 support (installed via conda)
- **Python 3.12+** with numpy, matplotlib, scipy, h5py, pandas
- **System:** Linux/macOS recommended, 16+ GB RAM for large experiments

The provided `environment.yml` file will install all necessary dependencies.

## Quick Start

1. Clone the repository
2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate riemann-experiments
   ```
3. Run a small experiment to test setup:
   ```bash
   cd experiment1
   sage experiment1_batch.sage experiment1_config.json
   ```
4. Check results in `experiment1/results/`

**Note:** All experiments must be run with the `riemann-experiments` conda environment activated, as SageMath is only available in this environment.

## Results and Analysis

- **Experiment Results:** Each experiment generates summary reports and visualizations in its `results/` directory
- **Comprehensive Analysis:** See `analysis/` folder for combined analysis across all experiments
- **Report Generation:** Use `analysis/generate_universal_critical_restoration_report.py` to generate full research report

## Key Findings

The experiments provide computational evidence that:
- The critical line ℜ(s) = 1/2 exhibits stable equilibrium behavior
- Energy changes follow perfect quadratic scaling ΔE(δ) ≈ C₁δ² with C₁ > 0
- Restoring forces scale linearly with system size: C₁⁽ᴺ⁾ ≈ 0.889N
- Universal stability holds across zero heights γ = 14 to γ = 909

## File Structure

```
riemann/
├── experiment1/          # Single-zero perturbation analysis
├── experiment2/          # Two-zero interaction analysis  
├── experiment3/          # Multi-zero scaling analysis
├── analysis/             # Combined analysis and report generation
├── project_plan/         # Design documents and specifications
├── backup/               # Backup files
└── README.md            # This file
```

## Contributing

This is a research codebase. For questions about the mathematical background or computational methods, please refer to the documentation in `project_plan/` and `reserach_background.md`.

## License

Research code - see individual files for specific terms.
