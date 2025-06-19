I just # Experiment 1 Refactor Project Plan: HDF5 Migration & Architecture Alignment

## Project Overview

**Objective:** Refactor Experiment 1 to align with the proven architecture from Experiments 2 & 3, implementing HDF5 unified data storage, consolidated statistical analysis, and standardized visualization pipeline.

**Current Problems:**
- Multiple CSV files per configuration variant (10+ separate files)
- Individual image generation per variant (fragmented visualization)
- No consolidated statistical analysis across all configurations
- Architecture inconsistent with Experiments 2 & 3
- Missing HDF5 data storage and metadata preservation

**Target State:**
- Single HDF5 file containing all configuration data
- Unified statistical analysis across entire dataset
- Exactly 5 summary visualization images
- Four-layer architecture matching Design Guide
- Professional publication-ready outputs

## Current State Analysis

### Existing File Structure
```
experiment1/
├── experiment1_batch_runner.sage         # Current orchestrator (needs restructuring)
├── experiment1_math.sage                 # Mathematical core (has good foundation)
├── experiment1_stats.sage                # Stats module (needs HDF5 integration)
├── experiment1_viz.sage                  # Viz module (needs consolidation)
├── experiment1_config*.json              # Multiple config files (good)
├── data/                                 # CSV files (convert to HDF5)
│   ├── experiment1_*_math_results.csv   # 5 separate math CSV files
│   └── experiment1_*_stats_results.csv  # 5 separate stats CSV files
└── results/                              # Individual outputs (consolidate)
    ├── experiment1_*_comprehensive_analysis.png  # 6 separate analysis images
    ├── experiment1_*_publication_figure.png      # 6 separate publication images
    └── experiment1_*_summary_report.txt          # 6 separate text reports
```

### Target File Structure (Aligned with Design Guide)
```
experiment1/
├── experiment1_batch.sage               # ✅ Orchestrator (rename + restructure)
├── experiment1_math.sage                # ✅ Mathematical Core (adapt to HDF5)
├── experiment1_stats.sage               # ✅ Statistical Analysis (HDF5 processing)
├── experiment1_viz.sage                 # ✅ Visualization Engine (5 summary images)
├── experiment1_config.json              # ✅ Base Configuration
├── experiment1_config_*.json            # ✅ Specialized Configs
├── data/                                # ✅ HDF5 Data Storage
│   └── experiment1_analysis.h5         # Single consolidated HDF5 file
└── results/                             # ✅ Standardized Outputs
    ├── experiment1_summary_1.png        # 5 standardized summary images
    ├── experiment1_summary_2.png
    ├── experiment1_summary_3.png
    ├── experiment1_summary_4.png
    ├── experiment1_summary_5.png
    └── experiment1_summary_report.txt    # Single comprehensive report
```

### HDF5 Schema Design (Based on Design Guide)

**Target HDF5 Structure:**
```
experiment1_analysis.h5
├── config_1_gamma_14.13/               # Base configuration
│   ├── metadata/                       # Experiment parameters
│   │   ├── gamma                       # Zero height: 14.13
│   │   ├── delta_range                 # Perturbation range
│   │   ├── test_function_type          # gaussian/fourier
│   │   └── timestamp                   # Analysis timestamp
│   ├── perturbation_analysis/          # Primary mathematical results
│   │   ├── delta                       # Perturbation values array
│   │   ├── delta_E                     # Energy changes array
│   │   ├── polyfit_coeffs              # [C₁, C₂] coefficients
│   │   ├── bootstrap_CI                # Confidence intervals
│   │   └── attributes                  # r_squared, p_values, etc.
│   └── numerical_results/              # Additional computational data
├── config_2_gamma_21.02/               # Gamma2 configuration
├── config_3_gamma_25.01_fourier/       # Gamma3 Fourier configuration
├── config_4_gamma_14.13_high_precision/ # High precision variant
└── config_5_gamma_14.13_test/          # Test configuration
```

## Implementation Tasks

### Phase 1: Infrastructure Setup ✅

- [x] **Task 1.1:** Create new branch `experiment1-refactor-hdf5`
- [x] **Task 1.2:** Fix conda environment naming issues
- [x] **Task 1.3:** Ensure terminal functionality works

### Phase 2: File Structure Reorganization

- [ ] **Task 2.1:** Rename `experiment1_batch_runner.sage` → `experiment1_batch.sage`
- [ ] **Task 2.2:** Create `experiment1/data/` and `experiment1/results/` directories
- [ ] **Task 2.3:** Backup existing CSV and image files to `experiment1/archive/`
- [ ] **Task 2.4:** Update VS Code tasks to reference new batch orchestrator

### Phase 3: Mathematical Core Adaptation (experiment1_math.sage)

- [ ] **Task 3.1:** Modify `Experiment1Math` class to output HDF5 instead of CSV
- [ ] **Task 3.2:** Implement HDF5 group creation for each configuration
- [ ] **Task 3.3:** Add comprehensive metadata storage (timestamps, parameters)
- [ ] **Task 3.4:** Preserve existing mathematical computations (good foundation)
- [ ] **Task 3.5:** Add progress tracking for batch configurations

**Key Changes Needed:**
```sage
class Experiment1Math:
    def write_to_hdf5(self, results, config_name):
        # Write structured data to experiment1_analysis.h5
        # Create groups: metadata/, perturbation_analysis/, numerical_results/
        # Store arrays: delta, delta_E, coefficients, etc.
```

### Phase 4: Batch Orchestrator Restructuring (experiment1_batch.sage)

- [ ] **Task 4.1:** Implement batch configuration processing loop
- [ ] **Task 4.2:** Coordinate Math → Stats → Viz pipeline execution
- [ ] **Task 4.3:** Add configuration validation and error handling
- [ ] **Task 4.4:** Implement progress tracking across all configurations

**Key Architecture Pattern:**
```sage
# Load all modules
load('experiment1_math.sage')
load('experiment1_stats.sage') 
load('experiment1_viz.sage')

# Process batch configurations
for config in batch_configs:
    # Math: Generate data → HDF5
    # Stats: Process HDF5 data → Statistical results
    # Viz: Aggregate analysis → Summary images
```

### Phase 5: Statistical Analysis Integration (experiment1_stats.sage)

- [ ] **Task 5.1:** Modify to read from HDF5 instead of CSV files
- [ ] **Task 5.2:** Implement cross-configuration statistical analysis
- [ ] **Task 5.3:** Add bootstrap confidence intervals (1000 samples minimum)
- [ ] **Task 5.4:** Implement hypothesis testing for stability analysis
- [ ] **Task 5.5:** Generate consolidated statistical summary

**Key Statistical Methods:**
```sage
class Experiment1Stats:
    def process_hdf5_data(self, hdf5_file):
        # Read all configurations from single HDF5 file
        # Perform cross-configuration analysis
        # Generate consolidated statistics
        
    def stability_analysis(self):
        # C₁ > 0 hypothesis testing across all configs
        # Bootstrap confidence intervals
        # Statistical significance assessment
```

### Phase 6: Visualization Consolidation (experiment1_viz.sage)

- [ ] **Task 6.1:** Replace individual images with 5 standardized summary images
- [ ] **Task 6.2:** Implement cross-configuration visualization patterns
- [ ] **Task 6.3:** Apply Design Guide visualization standards (max 2 subplots/figure)
- [ ] **Task 6.4:** Use professional color schemes and typography
- [ ] **Task 6.5:** Generate publication-quality outputs (DPI 300)

**Target Visualization Set:**
1. **Image 1:** Stability Analysis - C₁ coefficients across all configurations
2. **Image 2:** Fit Quality Assessment - R² values and distributions  
3. **Image 3:** Energy Perturbation Patterns - δE vs δ for key configurations
4. **Image 4:** Configuration Comparison - Gamma values vs stability metrics
5. **Image 5:** Parameter Space Coverage - Test function types and precision levels

### Phase 7: Summary Report Generation

- [ ] **Task 7.1:** Create unified summary report generator
- [ ] **Task 7.2:** Implement tabular configuration comparison
- [ ] **Task 7.3:** Add statistical significance interpretation
- [ ] **Task 7.4:** Include mathematical methodology description

**Report Structure (per Design Guide):**
```
EXPERIMENT 1: Single-Zero Perturbation Analysis
======================================================================

Analysis Timestamp: 2025-06-19 XX:XX:XX
Dataset: 5 configurations
Parameter Space: γ ∈ [14.13, 25.01], Test Functions: [Gaussian, Fourier]

STABILITY ANALYSIS SUMMARY:
----------------------------------------
Total Configurations: 5
Stable Coefficients (C₁ > 0): X (XX.X%)
Mean C₁ Coefficient: X.XXXe+XX
Mean R² (Fit Quality): X.XXXXXX
Significant Stability (p < 0.05): X (XX.X%)

[Additional analysis sections...]
```

### Phase 8: Configuration Management

- [ ] **Task 8.1:** Validate all existing configuration files
- [ ] **Task 8.2:** Ensure consistent parameter naming
- [ ] **Task 8.3:** Add batch_configs arrays to base configuration
- [ ] **Task 8.4:** Test configuration loading and validation

### Phase 9: Data Migration & Validation

- [ ] **Task 9.1:** Create migration script: CSV → HDF5
- [ ] **Task 9.2:** Validate mathematical results consistency
- [ ] **Task 9.3:** Compare old vs new statistical outputs
- [ ] **Task 9.4:** Verify visualization accuracy

### Phase 10: Integration Testing

- [ ] **Task 10.1:** Test full pipeline with small configuration set
- [ ] **Task 10.2:** Validate HDF5 file structure and metadata
- [ ] **Task 10.3:** Check statistical analysis accuracy
- [ ] **Task 10.4:** Verify 5 summary images generation
- [ ] **Task 10.5:** Test summary report generation

### Phase 11: Documentation & Cleanup

- [ ] **Task 11.1:** Update experiment1 README documentation
- [ ] **Task 11.2:** Add docstrings to refactored classes
- [ ] **Task 11.3:** Clean up archived files
- [ ] **Task 11.4:** Update VS Code task definitions

## Risk Assessment & Mitigation

### High Risk Areas:
1. **Mathematical Accuracy:** Ensure refactored math produces identical results
   - *Mitigation:* Comprehensive validation against existing CSV data
   
2. **HDF5 Data Corruption:** Large batch processing could corrupt files
   - *Mitigation:* Incremental saves, validation checks, backup strategy
   
3. **Memory Usage:** Single HDF5 file might consume excessive memory
   - *Mitigation:* Chunked processing, memory monitoring
   
4. **Statistical Method Changes:** Bootstrap CI might change interpretation
   - *Mitigation:* Parallel validation, confidence level consistency

### Medium Risk Areas:
1. **Configuration Compatibility:** Existing configs might need modifications
2. **Visualization Quality:** Consolidated images might lose detail
3. **Performance Regression:** HDF5 processing might be slower than CSV

## Success Criteria

### Functional Requirements:
- [x] Single HDF5 file contains all configuration data
- [ ] Exactly 5 summary visualization images generated
- [ ] One comprehensive summary report produced
- [ ] All existing mathematical results preserved
- [ ] Statistical analysis covers entire dataset

### Quality Requirements:
- [ ] Professional publication-ready visualizations
- [ ] Comprehensive error handling and validation
- [ ] Consistent architecture with Experiments 2 & 3
- [ ] Complete documentation and comments

### Performance Requirements:
- [ ] Full pipeline completes within reasonable time (< 2 hours)
- [ ] Memory usage remains manageable (< 8GB)
- [ ] HDF5 file size reasonable (< 1GB)

## Dependencies & Prerequisites

### Technical Dependencies:
- [x] SageMath environment (`sagemath` conda env)
- [x] HDF5 Python library (`h5py`)
- [x] NumPy, SciPy, Matplotlib
- [x] Git branch: `experiment1-refactor-hdf5`

### Knowledge Dependencies:
- [x] Design Guide HDF5 schema understanding
- [x] Experiment 2 & 3 architecture patterns
- [x] Bootstrap statistical methods
- [x] Professional visualization standards

## Timeline Estimate

**Total Estimated Time:** 12-16 hours across 2-3 days

- **Phase 2-4:** Infrastructure & Core (4-5 hours)
- **Phase 5-6:** Stats & Visualization (4-5 hours)  
- **Phase 7-8:** Reports & Configuration (2-3 hours)
- **Phase 9-11:** Testing & Documentation (2-3 hours)

## Next Actions

**Immediate Next Steps:**
1. ✅ Review and approve this project plan
2. Begin Phase 2: File structure reorganization
3. Start with Task 2.1: Rename batch orchestrator file
4. Create data/ and results/ directories
5. Begin mathematical core HDF5 adaptation

**Ready to proceed with Phase 2?** 🚀
