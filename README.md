# Universal Critical Restoration Conjecture

A computational investigation of the Riemann Hypothesis through energy minimization and variational methods.

## Overview

<img src="20250620_2006_Holographic Gradient Field.png" align="left" width="300" style="margin-right: 20px; margin-bottom: 10px;">

This repository contains experimental mathematics code testing the **Universal Critical Restoration conjecture** - a novel reformulation of the Riemann Hypothesis as an energy minimization problem. Instead of proving where zeros must lie, we treat zero configurations as physical systems and measure the energy changes when zeros are perturbed from the critical line.

<br clear="left">

## Project Documentation

### Research Foundation
- **[01. Research Background](research/01_Reserach_background.md)** - Complete mathematical background and motivation for the energy-based approach
- **[02. Heuristic Framework for Evaluating Mathematical Proof Strategies](research/02_Heuristic%20Framework%20for%20Evaluating%20Mathematical%20Proof%20Strategies.md)** - Methodology for proof strategy evaluation
- **[03. Generalized Riemann Hypothesis](research/03_Generalized_Riemann_Hypothesis_v01.md)** - Extensions to L-functions and broader contexts

### Project Plan & Implementation
- **[04. L-Function Zero Energy Functional](project_plan/04_L_Function_Zero_Energy_Functional.md)** - Mathematical framework and energy functional definition
- **[05. Experiment Pipeline Design Guide](project_plan/05_experiment_pipeline_design_guide.md)** - Technical specifications and experimental methodology
- **[06. Report Architecture](project_plan/06_report_architecture.md)** - Documentation structure and report generation specifications

### Results & Analysis
- **[Universal Critical Restoration Conjecture Analysis](results/universal_critical_restoration_conjecture_analysis.md)** - Comprehensive research report with experimental evidence
- **[PDF Report](results/universal_critical_restoration_conjecture_analysis.pdf)** - Publication-quality PDF with typeset mathematics

## Experiments

**Important:** Ensure the conda environment is activated before running experiments:
```bash
conda activate sagemath
```

The project consists of three complementary experiments testing different aspects of the Universal Critical Restoration conjecture:

### Experiment 1: Single-Zero Perturbation
Tests local stability by perturbing individual zeros and measuring quadratic energy behavior.

**Status: ✅ FULLY VALIDATED AND PRODUCTION-READY**

The Experiment 1 pipeline has been extensively refactored, debugged, and validated with:
- Robust modular architecture with unique output file naming
- Support for multiple configurations (zero heights, test function types, precision levels)
- Comprehensive statistical analysis and scientific visualization
- Professional summary reports with rigorous hypothesis testing

```bash
# Available configurations (all fully tested):
sage experiment1/experiment1_batch.sage experiment1/experiment1_config.json              # Multi-config batch (gamma 14, 21, 25)
sage experiment1/experiment1_batch.sage experiment1/experiment1_config_high_precision.json  # Ultra-high precision (gamma 14)
sage experiment1/experiment1_batch.sage experiment1/experiment1_config_gamma2.json          # Medium precision (gamma 21)  
sage experiment1/experiment1_batch.sage experiment1/experiment1_config_gamma3_fourier.json  # Fourier basis (gamma 25)

# Or run individual components (legacy):
sage experiment1/experiment1_math.sage
sage experiment1/experiment1_stats.sage
sage experiment1/experiment1_viz.sage
```

**Key Features:**
- ✅ Unique HDF5 and result file naming per configuration (no overwrites)
- ✅ Comprehensive statistical analysis with bootstrap confidence intervals
- ✅ Professional scientific visualizations (9-13 plots per run)
- ✅ Rigorous hypothesis testing with p-value significance assessment
- ✅ Cross-configuration analysis and parameter sensitivity studies
- ✅ All outputs stored in proper subdirectories (`data/`, `results/`)

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

### Core Requirements
- **Conda/Miniconda** for environment management
- **SageMath 10.6+** with HDF5 support (installed via conda)
- **Python 3.12+** with numpy, matplotlib, scipy, h5py, pandas
- **Pandoc** for document generation (included in environment)

### System Requirements
- **OS:** Linux/macOS recommended
- **RAM:** 16+ GB for large experiments
- **Browser:** Chrome/Chromium for PDF generation (must be installed separately)

### PDF Generation Requirements
For generating research reports with perfect math rendering:
- **Chrome or Chromium browser** (install via system package manager)
- **Pandoc** (included in conda environment)

The PDF generation uses: **Markdown → Pandoc → HTML+MathJax → Chrome headless → PDF**

### Installation

The provided `environment.yml` file installs all conda dependencies. Chrome must be installed separately:

```bash
# Ubuntu/Debian
sudo apt install google-chrome-stable
# or
sudo apt install chromium-browser

# macOS
brew install --cask google-chrome
# or
brew install chromium
```

## Quick Start

1. Clone the repository
2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate sagemath
   ```
3. Run a small experiment to test setup:
   ```bash
   cd experiment1
   sage experiment1_batch.sage experiment1_config.json
   ```
4. Check results in `experiment1/results/`

**Note:** All experiments must be run with the `sagemath` conda environment activated, as SageMath is only available in this environment.

## Results and Analysis

- **Experiment Results:** Each experiment generates summary reports and visualizations in its `results/` directory
- **Comprehensive Analysis:** See `analysis/` folder for combined analysis across all experiments

### Report Generation

The project includes advanced report generation with perfect math rendering:

```bash
# Generate markdown report with cherry-picked images
python analysis/generate_markdown.py

# Generate beautiful PDF with rendered math equations
python analysis/generate_pdf.py
```

**Output Location:** All reports are generated in the `results/` folder:
- `universal_critical_restoration_conjecture_analysis.md` - Comprehensive markdown report
- `universal_critical_restoration_conjecture_analysis.pdf` - Professional PDF with typeset math

**Features:**
- ✅ Perfect LaTeX math rendering (equations display as beautiful mathematical formulas)
- ✅ Cherry-picked key images from all experiments  
- ✅ Professional CSS styling and formatting
- ✅ Table of contents and cross-references
- ✅ Publication-quality output suitable for research submission


## Repository Structure

```
riemann/
├── research/                    # Research documentation and background
│   ├── 01_Reserach_background.md          # Complete mathematical background
│   ├── 02_Heuristic Framework...md        # Proof strategy evaluation methodology  
│   └── 03_Generalized_Riemann_Hypothesis_v01.md  # L-function extensions
├── project_plan/               # Technical specifications and design
│   ├── 04_L_Function_Zero_Energy_Functional.md    # Mathematical framework
│   ├── 05_experiment_pipeline_design_guide.md     # Experimental methodology
│   └── 06_report_architecture.md                  # Documentation structure
├── experiment1/                # Single-zero perturbation analysis
│   ├── experiment1_batch.sage             # Main batch processor
│   ├── experiment1_config*.json           # Configuration files
│   ├── experiment1_math.sage              # Core mathematical computations
│   ├── experiment1_stats.sage             # Statistical analysis
│   ├── experiment1_viz.sage               # Visualization generation
│   ├── data/                              # HDF5 computation results
│   └── results/                           # Summary reports and visualizations
├── experiment2/                # Two-zero interaction analysis  
│   ├── experiment2_batch.sage             # Main batch processor
│   ├── experiment2_config*.json           # Configuration files
│   ├── experiment2_math.sage              # Core mathematical computations
│   ├── experiment2_stats.sage             # Statistical analysis
│   ├── experiment2_viz.sage               # Visualization generation
│   ├── data/                              # HDF5 computation results
│   └── results/                           # Summary reports and visualizations
├── experiment3/                # Multi-zero scaling analysis
│   ├── experiment3_batch.sage             # Main batch processor
│   ├── experiment3_config*.json           # Configuration files
│   ├── experiment3_math.sage              # Core mathematical computations
│   ├── experiment3_stats.sage             # Statistical analysis
│   ├── experiment3_viz.sage               # Visualization generation
│   ├── data/                              # HDF5 computation results
│   └── results/                           # Summary reports and visualizations
├── analysis/                   # Report generation and cross-experiment analysis
│   ├── generate_markdown.py               # Markdown report generator
│   ├── generate_pdf.py                    # PDF report generator with math rendering
│   └── custom_template.html               # HTML template for PDF generation
├── results/                    # Final publication-quality reports
│   ├── images/                            # 13 cherry-picked key visualizations
│   ├── universal_critical_restoration_conjecture_analysis.md  # Comprehensive report
│   └── universal_critical_restoration_conjecture_analysis.pdf # Publication PDF
├── environment.yml             # Conda environment specification
├── riemann.code-workspace      # VS Code workspace configuration
└── README.md                   # This documentation
```

## Contributing

This is a research codebase. For questions about the mathematical background or computational methods, please refer to the documentation in `project_plan/` and `reserach_background.md`.

## License

Research code - see individual files for specific terms.
