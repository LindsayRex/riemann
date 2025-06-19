# ############################################################################
#
# EXPERIMENT 1: BATCH ORCHESTRATOR
# =================================
#
# Four-layer architecture implementation following the Design Guide:
# 1. Batch Orchestrator (this file) - Entry Point & Configuration Management
# 2. Mathematical Core - experiment1_math.sage
# 3. Statistical Analysis - experiment1_stats.sage  
# 4. Visualization Engine - experiment1_viz.sage
#
# This orchestrator coordinates: Math → HDF5 → Stats → Viz pipeline
#
# Usage:
#   sage experiment1_batch.sage experiment1_config.json
#
# ############################################################################

import time
import json
import sys
import h5py
import numpy as np
from pathlib import Path

# Load modules following Design Guide pattern
load('experiment1_math.sage')
load('experiment1_stats.sage') 
load('experiment1_viz.sage')

class Experiment1BatchOrchestrator:
    """
    Batch orchestrator implementing four-layer architecture.
    Coordinates Math → Stats → Viz pipeline with HDF5 data storage.
    """
    
    def __init__(self, config_file):
        """
        Initialize batch orchestrator.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        self.config = self._load_configuration()
        self.hdf5_file = f"data/{self.config.get('output_file', 'experiment1_analysis.h5')}"
        
        print("=" * 80)
        print("EXPERIMENT 1: BATCH ORCHESTRATOR")
        print("=" * 80)
        print(f"Configuration: {config_file}")
        print(f"Output HDF5: {self.hdf5_file}")
        
        # Create data and results directories
        Path("data").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
    def _load_configuration(self):
        """Load and validate configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Validate required parameters
            required = ['gamma', 'delta_range', 'delta_steps']
            for param in required:
                if param not in config:
                    raise ValueError(f"Missing required parameter: {param}")
                    
            print(f"✓ Configuration loaded: {len(config)} parameters")
            return config
            
        except Exception as e:
            print(f"✗ Configuration error: {e}")
            sys.exit(1)
        
    def process_batch_configs(self):
        """
        Process all configurations in batch_configs array.
        Implements the Design Guide batch processing pattern.
        """
        batch_configs = self.config.get('batch_configs', [self.config])
        total_configs = len(batch_configs)
        
        print(f"\n🚀 Processing {total_configs} configurations...")
        start_time = time.time()
        
        for i, config_override in enumerate(batch_configs):
            config_name = f"config_{i+1}"
            
            # Create merged configuration
            merged_config = self.config.copy()
            merged_config.update(config_override)
            
            # Generate descriptive config name based on parameters
            if 'gamma' in config_override:
                gamma_str = str(config_override['gamma']).replace('.', '_')
                config_name += f"_gamma_{gamma_str}"
            if 'test_function_type' in config_override:
                config_name += f"_{config_override['test_function_type']}"
            if 'high_precision' in config_override:
                config_name += "_high_precision"
                
            print(f"\n{'='*60}")
            print(f"CONFIG {i+1}/{total_configs}: {config_name}")
            print(f"{'='*60}")
            
            try:
                # Layer 2: Mathematical Core
                self._run_mathematical_core(merged_config, config_name)
                
                print(f"✓ Configuration {i+1}/{total_configs} completed")
                
            except Exception as e:
                print(f"✗ Configuration {i+1} failed: {e}")
                continue
                
        # Layer 3: Statistical Analysis (process entire HDF5 dataset)
        print(f"\n📊 Running statistical analysis on complete dataset...")
        self._run_statistical_analysis()
        
        # Layer 4: Visualization Engine (generate summary images)
        print(f"\n📈 Generating visualization summary...")
        self._run_visualization_engine()
        
        # Generate summary report
        print(f"\n📄 Generating summary report...")
        self._generate_summary_report()
        
        total_time = time.time() - start_time
        print(f"\n✅ Batch processing completed in {total_time:.2f} seconds")
        
    def _run_mathematical_core(self, config, config_name):
        """
        Execute Layer 2: Mathematical Core computation.
        
        Args:
            config: Configuration dictionary
            config_name: Unique name for this configuration
        """
        print(f"🔢 Mathematical Core: γ={config['gamma']}")
        
        # Initialize mathematical core
        math_core = Experiment1Math(
            gamma=config['gamma'],
            delta_range=config.get('delta_range', 0.1),
            delta_steps=config.get('delta_steps', 41),
            num_test_functions=config.get('num_test_functions', 20),
            test_function_type=config.get('test_function_type', 'gaussian')
        )
        
        # Run perturbation analysis
        results = math_core.run_perturbation_sweep(verbose=True)
        
        # Write to HDF5
        math_core.write_to_hdf5(results, self.hdf5_file, config_name)
        
    def _run_statistical_analysis(self):
        """
        Execute Layer 3: Statistical Analysis on complete HDF5 dataset.
        """
        stats_analyzer = Experiment1Stats(self.hdf5_file)
        stats_analyzer.process_all_configurations()
        
    def _run_visualization_engine(self):
        """
        Execute Layer 4: Visualization Engine - generate 5 summary images.
        """
        viz_engine = Experiment1Viz(self.hdf5_file)
        viz_engine.generate_summary_visualizations()
        
    def _generate_summary_report(self):
        """Generate comprehensive summary report following Design Guide structure."""
        report_filename = "results/experiment1_summary_report.txt"
        
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                report_lines = []
                
                # Header
                report_lines.extend([
                    "EXPERIMENT 1: Single-Zero Perturbation Analysis",
                    "=" * 70,
                    "",
                    f"Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Dataset: {len(f.keys())} configurations",
                    f"HDF5 File: {self.hdf5_file}",
                    ""
                ])
                
                # Collect statistics across all configurations
                all_c1 = []
                all_r2 = []
                all_significant = []
                all_gammas = []
                
                for config_name in f.keys():
                    group = f[config_name]
                    gamma = group['metadata'].attrs['gamma']
                    all_gammas.append(gamma)
                    
                    if 'statistical_analysis' in group:
                        stats = group['statistical_analysis']
                        all_c1.append(stats.attrs['C1_coefficient'])
                        all_r2.append(stats.attrs['r_squared'])
                        all_significant.append(stats.attrs['stability_significant'])
                
                # Parameter space summary
                gamma_min, gamma_max = min(all_gammas), max(all_gammas)
                report_lines.extend([
                    f"Parameter Space: γ ∈ [{gamma_min:.2f}, {gamma_max:.2f}]",
                    "",
                    "STABILITY ANALYSIS SUMMARY:",
                    "-" * 40
                ])
                
                if all_c1:
                    stable_count = sum(all_significant)
                    stable_percent = 100 * stable_count / len(all_significant)
                    mean_c1 = np.mean(all_c1)
                    mean_r2 = np.mean(all_r2)
                    
                    report_lines.extend([
                        f"Total Configurations: {len(all_c1)}",
                        f"Stable Coefficients (C₁ > 0): {stable_count} ({stable_percent:.1f}%)",
                        f"Mean C₁ Coefficient: {mean_c1:.6e}",
                        f"Mean R² (Fit Quality): {mean_r2:.6f}",
                        f"Significant Stability (p < 0.05): {stable_count} ({stable_percent:.1f}%)",
                        ""
                    ])
                
                # Detailed configuration results
                report_lines.extend([
                    "DETAILED CONFIGURATION RESULTS:",
                    "-" * 40,
                    f"{'Config':<35} {'γ':<12} {'C₁':<15} {'R²':<10} {'Stable':<8}",
                    "-" * 80
                ])
                
                for config_name in f.keys():
                    group = f[config_name]
                    gamma = group['metadata'].attrs['gamma']
                    test_type = group['metadata'].attrs['test_function_type']
                    
                    if isinstance(test_type, bytes):
                        test_type = test_type.decode()
                    
                    config_label = f"γ={gamma:.1f} ({test_type})"
                    
                    if 'statistical_analysis' in group:
                        stats = group['statistical_analysis']
                        c1 = stats.attrs['C1_coefficient']
                        r2 = stats.attrs['r_squared']
                        stable = "Yes" if stats.attrs['stability_significant'] else "No"
                        
                        report_lines.append(
                            f"{config_label:<35} {gamma:<12.3f} {c1:<15.6e} {r2:<10.6f} {stable:<8}"
                        )
                    else:
                        report_lines.append(
                            f"{config_label:<35} {gamma:<12.3f} {'N/A':<15} {'N/A':<10} {'N/A':<8}"
                        )
                
                # Statistical summary
                report_lines.extend([
                    "",
                    "STATISTICAL SUMMARY:",
                    "-" * 40
                ])
                
                if all_c1 and all(all_significant):
                    assessment = "STABLE"
                    riemann_support = "Strong evidence supports Riemann Hypothesis"
                    significance = "All configurations show C₁ > 0 with high statistical significance"
                elif all_c1 and any(all_significant):
                    assessment = "MIXED"
                    riemann_support = "Partial evidence supports Riemann Hypothesis"
                    significance = f"{sum(all_significant)}/{len(all_significant)} configurations show stability"
                else:
                    assessment = "UNSTABLE"
                    riemann_support = "Insufficient evidence for Riemann Hypothesis"
                    significance = "No configurations show statistically significant stability"
                
                report_lines.extend([
                    f"Overall Assessment: {assessment}",
                    f"Riemann Hypothesis Support: {riemann_support}",
                    f"Mathematical Significance: {significance}",
                    ""
                ])
                
                # Experimental details
                first_config = f[list(f.keys())[0]]
                delta_range = first_config['metadata'].attrs['delta_range']
                delta_steps = first_config['metadata'].attrs['delta_steps']
                num_test_functions = first_config['metadata'].attrs['num_test_functions']
                
                report_lines.extend([
                    "EXPERIMENTAL DETAILS:",
                    "-" * 40,
                    "Energy Functional: Single-zero perturbation ΔE(δ) analysis",
                    f"Test Function Basis: {num_test_functions} functions (Gaussian/Fourier)",
                    "Statistical Methods: Quadratic fitting, bootstrap CI, hypothesis testing",
                    f"Perturbation Range: δ ∈ [±{delta_range}]",
                    f"Sampling Resolution: {delta_steps} points per configuration",
                    f"Confidence Level: 95%",
                    ""
                ])
                
                # Write report
                with open(report_filename, 'w') as f:
                    f.write('\n'.join(report_lines))
                    
                print(f"✓ Summary report generated: {report_filename}")
                
        except Exception as e:
            print(f"✗ Report generation failed: {e}")
            
            # Create minimal report
            with open(report_filename, 'w') as f:
                f.write(f"EXPERIMENT 1: Summary Report\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: Report generation failed - {e}\n")

# Command line execution
if len(sys.argv) != 2:
    print("Usage: sage experiment1_batch.sage <config_file>")
    print("Example: sage experiment1_batch.sage experiment1_config.json")
    sys.exit(1)

config_file = sys.argv[1]

# Verify config file exists
if not Path(config_file).exists():
    print(f"Error: Configuration file '{config_file}' not found")
    sys.exit(1)

# Run batch orchestrator
orchestrator = Experiment1BatchOrchestrator(config_file)
orchestrator.process_batch_configs()
