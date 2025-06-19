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
load('experiment1/experiment1_math.sage')
load('experiment1/experiment1_stats.sage') 
load('experiment1/experiment1_viz.sage')

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
        
        # Create unique output file names based on config
        if 'output_file' in self.config:
            # Use explicit output_file if provided
            self.hdf5_file = f"experiment1/data/{self.config['output_file']}"
            self.output_prefix = self.config['output_file'].replace('.h5', '')
        elif 'output_prefix' in self.config:
            # Use output_prefix to generate unique names
            prefix = self.config['output_prefix']
            self.hdf5_file = f"experiment1/data/{prefix}.h5"
            self.output_prefix = prefix
        else:
            # Default fallback
            self.hdf5_file = f"experiment1/data/experiment1_analysis.h5"
            self.output_prefix = "experiment1_analysis"
        
        print("=" * 80)
        print("EXPERIMENT 1: BATCH ORCHESTRATOR")
        print("=" * 80)
        print(f"Configuration: {config_file}")
        print(f"Output HDF5: {self.hdf5_file}")
        
        # Create data and results directories
        Path("experiment1/data").mkdir(exist_ok=True)
        Path("experiment1/results").mkdir(exist_ok=True)
        
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
        Execute Layer 3: Enhanced Statistical Analysis on complete HDF5 dataset.
        """
        print("🔬 Running comprehensive statistical analysis...")
        stats_analyzer = Experiment1Statistics(self.hdf5_file)
        stats_analyzer.analyze_all_configurations()
        
        # Generate comprehensive summary report
        summary_report_path = f"experiment1/results/{self.output_prefix}_comprehensive_statistical_summary.txt"
        stats_analyzer.generate_summary_report(summary_report_path)
        
    def _run_visualization_engine(self):
        """
        Execute Layer 4: Enhanced Visualization Engine - generate comprehensive analysis.
        """
        print("📊 Running comprehensive visualization engine...")
        viz_engine = Experiment1Visualization(self.hdf5_file, output_dir="experiment1/results", output_prefix=self.output_prefix)
        generated_files = viz_engine.generate_all_visualizations()
        
        print(f"✅ Generated {len(generated_files)} visualization files")
        
    def _generate_summary_report(self):
        """
        Generate unified summary report following existing format.
        Creates experiment1_summary_report.txt in results directory.
        """
        report_path = f"experiment1/results/{self.output_prefix}_summary_report.txt"
        
        # Read all configuration data from HDF5
        config_data = []
        
        with h5py.File(self.hdf5_file, 'r') as f:
            for config_name in f.keys():
                group = f[config_name]
                
                # Extract metadata
                metadata = group['metadata']
                
                # Extract statistical results if available
                stats_data = {}
                if 'statistics' in group:
                    stats = group['statistics']
                    # Extract quadratic fitting results (C1, R²)
                    if 'fitting_results' in stats and 'quadratic' in stats['fitting_results']:
                        quad = stats['fitting_results']['quadratic']
                        stats_data['C1_coefficient'] = quad['C1'][()]
                        stats_data['r_squared'] = quad['r_squared'][()]
                    
                    # Extract hypothesis testing p-value
                    if 'hypothesis_testing' in stats and 'local_stability' in stats['hypothesis_testing']:
                        local_stab = stats['hypothesis_testing']['local_stability']
                        stats_data['p_value_stability'] = local_stab['p_value'][()]
                
                config_data.append({
                    'name': config_name,
                    'gamma': metadata.attrs['gamma'],
                    'test_function_type': str(metadata.attrs['test_function_type']),
                    'delta_range': metadata.attrs['delta_range'],
                    'num_test_functions': metadata.attrs['num_test_functions'],
                    'delta_steps': metadata.attrs['delta_steps'],
                    'computation_time': metadata.attrs['computation_time'],
                    'timestamp': str(metadata.attrs['timestamp']),
                    'stats': stats_data
                })
        
        # Generate report content
        with open(report_path, 'w') as f:
            f.write("EXPERIMENT 1: SINGLE-ZERO PERTURBATION ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {len(config_data)} configurations\n")
            if config_data:
                gamma_values = [cfg['gamma'] for cfg in config_data]
                f.write(f"Parameter Space: γ ∈ [{min(gamma_values):.3f}, {max(gamma_values):.3f}]\n\n")
            
            # Stability Analysis Summary
            f.write("STABILITY ANALYSIS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            stable_count = 0
            c1_values = []
            r_squared_values = []
            p_values = []
            
            for cfg in config_data:
                if 'C1_coefficient' in cfg['stats']:
                    c1_values.append(cfg['stats']['C1_coefficient'])
                if 'r_squared' in cfg['stats']:
                    r_squared_values.append(cfg['stats']['r_squared'])
                if 'p_value_stability' in cfg['stats']:
                    p_values.append(cfg['stats']['p_value_stability'])
                    if cfg['stats']['p_value_stability'] < 0.05:
                        stable_count += 1
            
            f.write(f"Total Configurations: {len(config_data)}\n")
            if c1_values:
                positive_c1 = sum(1 for c1 in c1_values if c1 > 0)
                percentage = float(100*positive_c1)/float(len(c1_values))
                f.write(f"Stable Coefficients (C₁ > 0): {positive_c1} ({percentage:.1f}%)\n")
                f.write(f"Mean C₁ Coefficient: {float(np.mean(c1_values)):.3e}\n")
            if r_squared_values:
                f.write(f"Mean R² (Fit Quality): {float(np.mean(r_squared_values)):.6f}\n")
            if p_values:
                significant = sum(1 for p in p_values if p < 0.05)
                sig_percentage = float(100*significant)/float(len(p_values))
                f.write(f"Significant Stability (p < 0.05): {significant} ({sig_percentage:.1f}%)\n\n")
            
            # Detailed Configuration Results
            f.write("DETAILED CONFIGURATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Config':<30} {'γ':<12} {'Type':<10} {'C₁':<15} {'R²':<10} {'p-value':<10}\n")
            f.write("-" * 90 + "\n")
            
            for cfg in config_data:
                config_short = cfg['name'].replace('config_', '').replace('_gamma_', 'γ')[:28]
                gamma = f"{cfg['gamma']:.3f}"
                func_type = cfg['test_function_type'][:8]
                
                c1_str = f"{cfg['stats'].get('C1_coefficient', 0):.3e}" if 'C1_coefficient' in cfg['stats'] else "N/A"
                r2_str = f"{cfg['stats'].get('r_squared', 0):.6f}" if 'r_squared' in cfg['stats'] else "N/A"
                p_str = f"{cfg['stats'].get('p_value_stability', 1):.3e}" if 'p_value_stability' in cfg['stats'] else "N/A"
                
                f.write(f"{config_short:<30} {gamma:<12} {func_type:<10} {c1_str:<15} {r2_str:<10} {p_str:<10}\n")
            
            # Statistical Summary
            f.write("\nSTATISTICAL SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            if c1_values and all(c1 > 0 for c1 in c1_values):
                f.write("Overall Assessment: STABLE\n")
                f.write("Riemann Hypothesis Support: Energy functional exhibits local stability\n")
                f.write("Mathematical Significance: C₁ > 0 confirmed across all configurations\n\n")
            elif c1_values:
                f.write("Overall Assessment: MIXED\n")
                f.write("Riemann Hypothesis Support: Partial stability observed\n\n")
            else:
                f.write("Overall Assessment: ANALYSIS INCOMPLETE\n\n")
            
            # Experimental Details
            f.write("EXPERIMENTAL DETAILS:\n")
            f.write("-" * 40 + "\n")
            f.write("Energy Functional: Single-zero perturbation ΔE(δ) = C₁δ² + C₂δ³ + ...\n")
            
            test_functions = set(cfg['test_function_type'] for cfg in config_data)
            f.write(f"Test Function Basis: {', '.join(test_functions)}\n")
            f.write("Statistical Methods: Polynomial fitting, bootstrap confidence intervals, hypothesis testing\n")
            
            # Generated Files
            f.write("\nGENERATED FILES:\n")
            f.write("-" * 20 + "\n")
            f.write(f"  HDF5 data: {self.hdf5_file}\n")
            f.write("  Summary images: experiment1_summary_1.png through experiment1_summary_5.png\n")
            f.write(f"  This report: {report_path}\n\n")
        
        print(f"✓ Summary report generated: '{report_path}'")
        return report_path

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
