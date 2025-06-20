# Universal Critical Restoration Conjecture

A computational investigation of the Riemann Hypothesis through energy minimization and variational methods. This program is not just about getting a “yes or no” to the Riemann Hypothesis. It’s a research path to build new mathematics, step by step: Discovery → Conjecture → Proof. 

## Overview

<img src="20250620_2006_Holographic Gradient Field.png" align="left" width="300" style="margin-right: 20px; margin-bottom: 10px;">

This repository contains experimental mathematics code testing the Universal Critical Restoration conjecture - a novel reformulation of the Riemann Hypothesis as an energy minimization problem. Instead of proving where zeros must lie, we treat zero configurations as physical systems and measure the energy changes when zeros are perturbed from the critical line.

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
- **[Broader Implications of Research](results/Broader%20Implications%20of%20Research.md)** - Summary of conceptual and technical advances

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
│   ├── generate_*_config.py               # Configuration generators
│   ├── data/                              # HDF5 computation results
│   └── results/                           # Summary reports and visualizations
├── experiment3/                # Multi-zero scaling analysis
│   ├── experiment3_batch.sage             # Main batch processor
│   ├── experiment3_config*.json           # Configuration files
│   ├── experiment3_math.sage              # Core mathematical computations
│   ├── experiment3_stats.sage             # Statistical analysis
│   ├── experiment3_viz.sage               # Visualization generation
│   ├── generate_*_config.py               # Configuration generators
│   ├── data/                              # HDF5 computation results
│   └── results/                           # Summary reports and visualizations
├── analysis/                   # Report generation and cross-experiment analysis
│   ├── generate_markdown.py               # Markdown report generator
│   ├── generate_pdf.py                    # PDF report generator with math rendering
│   └── custom_template.html               # HTML template for PDF generation
├── results/                    # Final publication-quality reports
│   ├── images/                            # Key visualizations from all experiments
│   ├── universal_critical_restoration_conjecture_analysis.md  # Comprehensive report
│   └── universal_critical_restoration_conjecture_analysis.pdf # Publication PDF
├── environment.yml             # Conda environment specification
├── riemann.code-workspace      # VS Code workspace configuration
└── README.md                   # This documentation
```

## Experiments

**Important:** Ensure the conda environment is activated before running experiments:
```bash
conda activate sagemath
```

The project consists of three complementary experiments testing different aspects of the Universal Critical Restoration conjecture:

### Experiment 1: Single-Zero Perturbation
Tests local stability by perturbing individual zeros and measuring quadratic energy behavior.

```bash
# Run with any configuration file
sage experiment1/experiment1_batch.sage experiment1/experiment1_config.json
sage experiment1/experiment1_batch.sage experiment1/experiment1_config_high_precision.json
sage experiment1/experiment1_batch.sage experiment1/experiment1_config_gamma2.json
sage experiment1/experiment1_batch.sage experiment1/experiment1_config_gamma3_fourier.json

# Or run individual components
sage experiment1/experiment1_math.sage
sage experiment1/experiment1_stats.sage
sage experiment1/experiment1_viz.sage
```

### Experiment 2: Two-Zero Interaction
Analyzes interference effects and additivity properties in two-zero systems.

```bash
# Run with configuration file
sage experiment2/experiment2_batch.sage experiment2/experiment2_config.json

# Generate custom configurations
python experiment2/generate_large_scale_config.py
python experiment2/generate_first_100_zeros_config.py

# Or run individual components
sage experiment2/experiment2_math.sage
sage experiment2/experiment2_stats.sage  
sage experiment2/experiment2_viz.sage
```

### Experiment 3: Multi-Zero Scaling
Tests scaling laws and universal stability across large multi-zero configurations.

```bash
# Run with configuration file (small to large scale)
sage experiment3/experiment3_batch.sage experiment3/experiment3_config.json
sage experiment3/experiment3_batch.sage experiment3/experiment3_config_phase3.json
sage experiment3/experiment3_batch.sage experiment3/experiment3_config_phase3_full.json

# Generate custom configurations
python experiment3/generate_large_scale_config.py
python experiment3/generate_phase3_config.py
```

**Output:** Each experiment generates HDF5 data files, statistical summaries, and visualizations in its respective `data/` and `results/` directories.

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

### Report Generation

Generate comprehensive analysis reports combining results from all experiments:

```bash
# Generate markdown report with key images from all experiments
python analysis/generate_markdown.py

# Generate PDF with rendered math equations
python analysis/generate_pdf.py
```

**Output Location:** Reports are generated in the `results/` folder:
- `universal_critical_restoration_conjecture_analysis.md` - Comprehensive markdown report
- `universal_critical_restoration_conjecture_analysis.pdf` - Publication-quality PDF with typeset math

**Features:**
- LaTeX math rendering (equations display as mathematical formulas)
- Key visualizations from all experiments  
- Professional formatting and styling
- Table of contents and cross-references
- Publication-quality output


## Contributing

This is a research codebase. For questions about the mathematical background or computational methods, please refer to the documentation in `project_plan/` and `reserach_background.md`.

## License

Research code - see individual files for specific terms.
