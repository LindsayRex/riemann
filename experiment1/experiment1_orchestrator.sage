# ############################################################################
#
# EXPERIMENT 1: SINGLE-ZERO PERTURBATION - MAIN ORCHESTRATOR
# ===========================================================
#
# This script orchestrates the complete Experiment 1 analysis for testing
# the quadratic behavior ΔE(δ) ≈ C₁δ² of energy differences when a single
# zero is perturbed from the critical line.
#
# The analysis pipeline includes:
# 1. Mathematical Core: Energy functional computation and perturbation sweep
# 2. Statistical Analysis: Polynomial fitting, hypothesis testing, bootstrap
# 3. Visualization: Comprehensive plots with error bars and p-values
#
# Usage:
#   sage experiment1_orchestrator.sage [--config config.json]
#
# ############################################################################

import sys
import json
import time
from pathlib import Path

# Import our modular components
load('experiment1_math.sage')
load('experiment1_stats.sage')
load('experiment1_viz.sage')

class Experiment1Orchestrator:
    """Main orchestrator for the single-zero perturbation analysis pipeline."""
    
    def __init__(self, config=None):
        """
        Initialize the orchestrator with configuration parameters.
        
        Args:
            config: Dictionary with configuration parameters or None for defaults
        """
        # Default configuration
        self.default_config = {
            'gamma': 14.13,                    # Height of the single zero
            'delta_range': 0.1,                # Range for δ ∈ [-range, range]
            'delta_steps': 41,                 # Number of δ values to test
            'num_test_functions': 20,          # Number of test functions in basis
            'test_function_type': 'gaussian',  # Type of test functions
            'confidence_level': 0.95,          # Statistical confidence level
            'bootstrap_samples': 10000,        # Bootstrap resamples
            'output_prefix': 'experiment1',    # Output file prefix
            'verbose': True                    # Verbose output
        }
        
        # Use provided config or defaults
        if config:
            self.config = {**self.default_config, **config}
        else:
            self.config = self.default_config
        
        self.verbose = self.config['verbose']
        
        if self.verbose:
            print("=" * 80)
            print("EXPERIMENT 1: SINGLE-ZERO PERTURBATION ANALYSIS")
            print("Testing ΔE(δ) ≈ C₁δ² + C₂δ³ for local stability")
            print("=" * 80)
            print(f"Configuration: {self.config}")
    
    def run_complete_analysis(self):
        """
        Run the complete Experiment 1 analysis pipeline.
        
        Returns:
            dict: Complete results including mathematical, statistical, and visual outputs
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("PHASE 1: MATHEMATICAL COMPUTATION")
            print("=" * 60)
        
        # 1. Mathematical Core - Energy functional computation
        exp1_math = create_experiment1_math(
            gamma=self.config['gamma'],
            delta_range=self.config['delta_range'],
            delta_steps=self.config['delta_steps'],
            num_test_functions=self.config['num_test_functions'],
            test_function_type=self.config['test_function_type']
        )
        
        # Run perturbation sweep
        math_results = exp1_math.run_perturbation_sweep(verbose=self.verbose)
        
        # Compute numerical derivatives
        derivative_analysis = exp1_math.compute_numerical_derivatives(math_results)
        math_results['derivative_analysis'] = derivative_analysis
        
        # Export mathematical results
        math_csv_file = exp1_math.export_results_csv(
            math_results, 
            f"{self.config['output_prefix']}_math_results.csv"
        )
        
        if self.verbose:
            print(f"✓ Mathematical computation completed in {math_results['computation_time']:.2f} seconds")
            print(f"✓ Computed ΔE for {len(math_results['delta_values'])} δ values")
            print(f"✓ Zero height γ = {math_results['gamma']}")
            print(f"✓ C₁ estimate from derivatives: {derivative_analysis['C1_estimate']:.2e}")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("PHASE 2: STATISTICAL ANALYSIS")
            print("=" * 60)
        
        # 2. Statistical Analysis
        exp1_stats = create_experiment1_statistics(
            confidence_level=self.config['confidence_level'],
            bootstrap_samples=self.config['bootstrap_samples']
        )
        
        stats_results = exp1_stats.comprehensive_analysis(
            math_results['delta_values'],
            math_results['delta_E_values']
        )
        
        # Print detailed statistical report
        if self.verbose:
            exp1_stats.print_detailed_report(stats_results)
        
        # Export statistical results
        stats_csv_file = exp1_stats.export_statistical_results_csv(
            stats_results,
            f"{self.config['output_prefix']}_stats_results.csv"
        )
        
        if self.verbose:
            print(f"✓ Statistical analysis completed in {stats_results['analysis_time']:.2f} seconds")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("PHASE 3: VISUALIZATION")
            print("=" * 60)
        
        # 3. Visualization
        exp1_plotter = create_experiment1_visualization(figsize=(20, 12), dpi=300)
        
        # Generate comprehensive analysis plot
        comprehensive_plot = exp1_plotter.create_comprehensive_plot(
            math_results['delta_values'],
            math_results['delta_E_values'],
            stats_results['fitting_results'],
            stats_results,
            derivative_analysis,
            f"{self.config['output_prefix']}_comprehensive_analysis.png"
        )
        
        # Generate publication-ready figure
        publication_plot = exp1_plotter.create_publication_figure(
            math_results['delta_values'],
            math_results['delta_E_values'],
            stats_results['fitting_results'],
            stats_results,
            f"{self.config['output_prefix']}_publication_figure.png"
        )
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"✓ Visualization completed")
            print(f"✓ Total analysis time: {total_time:.2f} seconds")
        
        # Compile complete results
        complete_results = {
            'configuration': self.config,
            'mathematical_results': math_results,
            'statistical_results': stats_results,
            'files': {
                'math_csv': math_csv_file,
                'stats_csv': stats_csv_file,
                'comprehensive_plot': comprehensive_plot,
                'publication_plot': publication_plot
            },
            'analysis_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Export complete summary
        self.export_summary_report(complete_results)
        
        return complete_results
    
    def export_summary_report(self, complete_results):
        """
        Export a comprehensive summary report of the entire experiment.
        
        Args:
            complete_results: Complete analysis results
        """
        summary_filename = f"{self.config['output_prefix']}_summary_report.txt"
        
        with open(summary_filename, 'w') as f:
            f.write("EXPERIMENT 1: SINGLE-ZERO PERTURBATION ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Analysis Timestamp: {complete_results['timestamp']}\n")
            f.write(f"Total Analysis Time: {complete_results['analysis_time']:.2f} seconds\n\n")
            
            # Configuration
            f.write("CONFIGURATION PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            for key, value in complete_results['configuration'].items():
                f.write(f"{key:20}: {value}\n")
            f.write("\n")
            
            # Mathematical Results Summary
            math_results = complete_results['mathematical_results']
            f.write("MATHEMATICAL COMPUTATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Zero height (γ): {math_results['gamma']}\n")
            f.write(f"Perturbation range: ±{math_results['delta_range']}\n")
            f.write(f"Number of δ points: {len(math_results['delta_values'])}\n")
            f.write(f"Computation time: {math_results['computation_time']:.2f} seconds\n")
            
            # Derivative analysis
            if 'derivative_analysis' in math_results:
                deriv = math_results['derivative_analysis']
                f.write(f"C₁ estimate (numerical): {deriv['C1_estimate']:.6e}\n")
                f.write(f"C₂ estimate (numerical): {deriv['C2_estimate']:.6e}\n")
                f.write(f"Gradient at δ=0: {deriv['gradient_at_zero']:.6e}\n")
            f.write("\n")
            
            # Statistical Results Summary
            stats_results = complete_results['statistical_results']
            f.write("STATISTICAL ANALYSIS RESULTS:\n")
            f.write("-" * 35 + "\n")
            
            # Polynomial fitting
            fitting = stats_results['fitting_results']
            for model_name, fit_result in fitting.items():
                if fit_result is None:
                    continue
                f.write(f"{model_name.upper()} MODEL:\n")
                f.write(f"  R² = {fit_result['r_squared']:.6f}\n")
                f.write(f"  AIC = {fit_result['aic']:.2f}\n")
                if 'C1' in fit_result:
                    f.write(f"  C₁ = {fit_result['C1']:.6e} ± {fit_result['C1_stderr']:.2e}\n")
                if 'C2' in fit_result:
                    f.write(f"  C₂ = {fit_result['C2']:.6e} ± {fit_result['C2_stderr']:.2e}\n")
                f.write("\n")
            
            # Hypothesis testing
            if 'hypothesis_testing' in stats_results:
                hyp_tests = stats_results['hypothesis_testing']
                f.write("HYPOTHESIS TESTING:\n")
                f.write("-" * 20 + "\n")
                
                if 'local_stability' in hyp_tests:
                    stability = hyp_tests['local_stability']
                    f.write(f"Local Stability Test (C₁ > 0):\n")
                    f.write(f"  t-statistic: {stability['test_statistic']:.4f}\n")
                    f.write(f"  p-value: {stability['p_value']:.6f}\n")
                    f.write(f"  Result: {'STABLE' if stability['significant'] else 'INCONCLUSIVE'}\n\n")
                
                if 'cubic_significance' in hyp_tests:
                    cubic = hyp_tests['cubic_significance']
                    f.write(f"Cubic Term Significance (C₂ ≠ 0):\n")
                    f.write(f"  t-statistic: {cubic['test_statistic']:.4f}\n")
                    f.write(f"  p-value: {cubic['p_value']:.6f}\n")
                    f.write(f"  Result: {'SIGNIFICANT' if cubic['significant'] else 'NOT SIGNIFICANT'}\n\n")
            
            # Bootstrap results
            if 'bootstrap_analysis' in stats_results and stats_results['bootstrap_analysis'] is not None:
                bootstrap = stats_results['bootstrap_analysis']
                f.write("BOOTSTRAP ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Successful samples: {bootstrap['successful_samples']}\n")
                if 'C1' in bootstrap:
                    c1_stats = bootstrap['C1']
                    f.write(f"C₁ bootstrap mean: {c1_stats['mean']:.6e}\n")
                    f.write(f"C₁ 95% CI: [{c1_stats['ci_lower']:.2e}, {c1_stats['ci_upper']:.2e}]\n")
                f.write("\n")
            
            # Best model
            if 'best_model' in stats_results and stats_results['best_model'] is not None:
                best = stats_results['best_model']
                f.write("BEST MODEL SELECTION:\n")
                f.write("-" * 22 + "\n")
                f.write(f"Selected model: {best['model_name'].upper()}\n")
                f.write(f"AIC: {best['aic']:.2f}\n")
                f.write(f"R²: {best['r_squared']:.6f}\n\n")
            
            # Generated Files
            f.write("GENERATED FILES:\n")
            f.write("-" * 16 + "\n")
            for file_type, filename in complete_results['files'].items():
                f.write(f"  {file_type}: {filename}\n")
            f.write("\n")
            
            # Conclusions
            f.write("EXPERIMENTAL CONCLUSIONS:\n")
            f.write("-" * 25 + "\n")
            
            # Determine overall conclusion from statistical tests
            if 'hypothesis_testing' in stats_results:
                hyp_tests = stats_results['hypothesis_testing']
                
                if 'local_stability' in hyp_tests:
                    stability_result = hyp_tests['local_stability']['significant']
                    if stability_result:
                        f.write("✓ LOCAL STABILITY CONFIRMED: C₁ > 0 is statistically significant\n")
                        f.write("  The critical line (δ=0) is a local minimum of the energy functional\n")
                    else:
                        f.write("? LOCAL STABILITY INCONCLUSIVE: C₁ > 0 not statistically significant\n")
                        f.write("  More data or different parameters may be needed\n")
                
                if 'cubic_significance' in hyp_tests:
                    cubic_result = hyp_tests['cubic_significance']['significant']
                    if cubic_result:
                        f.write("⚠ CUBIC TERM DETECTED: C₂ ≠ 0 is statistically significant\n")
                        f.write("  Higher-order effects may be important for larger perturbations\n")
                    else:
                        f.write("✓ QUADRATIC DOMINANCE: C₂ ≈ 0, quadratic behavior confirmed\n")
                        f.write("  Energy functional is well-approximated by ΔE ≈ C₁δ²\n")
        
        if self.verbose:
            print(f"✓ Summary report saved: '{summary_filename}'")
    
    def load_config_from_file(self, config_file):
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            dict: Configuration parameters
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ Configuration loaded from: '{config_file}'")
            return config
        except Exception as e:
            print(f"Error loading config file '{config_file}': {e}")
            print("Using default configuration")
            return {}

def create_example_config():
    """Create an example configuration file for Experiment 1."""
    example_config = {
        "gamma": 14.13,
        "delta_range": 0.1,
        "delta_steps": 41,
        "num_test_functions": 20,
        "test_function_type": "gaussian",
        "confidence_level": 0.95,
        "bootstrap_samples": 10000,
        "output_prefix": "experiment1_example",
        "verbose": True,
        "comments": {
            "gamma": "Height of single zero (first nontrivial zeta zero)",
            "delta_range": "Perturbation range δ ∈ [-0.1, 0.1] from critical line",
            "delta_steps": "41 points gives good resolution with δ=0 in center",
            "num_test_functions": "20 Gaussian test functions for energy functional",
            "test_function_type": "Use 'gaussian' or 'fourier' test functions",
            "bootstrap_samples": "10000 bootstrap samples for robust statistics"
        }
    }
    
    with open('experiment1_config_example.json', 'w') as f:
        json.dump(example_config, f, indent=4)
    
    print("✓ Example configuration saved: 'experiment1_config_example.json'")

def main():
    """Main entry point for Experiment 1 analysis."""
    
    # Parse command line arguments
    config = {}
    if len(sys.argv) > 1:
        if sys.argv[1] == '--create-config':
            create_example_config()
            return
        elif sys.argv[1] == '--config' and len(sys.argv) > 2:
            config_file = sys.argv[2]
            # Load configuration (simplified for SageMath)
            try:
                with open(config_file, 'r') as f:
                    import json
                    config = json.load(f)
                print(f"✓ Configuration loaded from: '{config_file}'")
            except Exception as e:
                print(f"Error loading config: {e}")
                config = {}
    
    # Create orchestrator and run analysis
    orchestrator = Experiment1Orchestrator(config)
    results = orchestrator.run_complete_analysis()
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 ANALYSIS COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    
    # Extract key results for summary
    math_results = results['mathematical_results']
    stats_results = results['statistical_results']
    
    print(f"Mathematical Computation: ✓ Completed ({math_results['computation_time']:.1f}s)")
    print(f"Statistical Analysis: ✓ Completed ({stats_results['analysis_time']:.1f}s)")
    print(f"Visualization: ✓ Completed")
    print(f"")
    
    # Mathematical summary
    if 'derivative_analysis' in math_results:
        deriv = math_results['derivative_analysis']
        print(f"Numerical Analysis:")
        print(f"  C₁ estimate: {deriv['C1_estimate']:.6e}")
        print(f"  C₂ estimate: {deriv['C2_estimate']:.6e}")
    
    # Statistical summary
    if 'fitting_results' in stats_results and 'cubic' in stats_results['fitting_results']:
        cubic_fit = stats_results['fitting_results']['cubic']
        if cubic_fit is not None:
            print(f"Polynomial Fitting (Cubic Model):")
            print(f"  C₁ = {cubic_fit['C1']:.6e} ± {cubic_fit['C1_stderr']:.2e}")
            print(f"  C₂ = {cubic_fit['C2']:.6e} ± {cubic_fit['C2_stderr']:.2e}")
            print(f"  R² = {cubic_fit['r_squared']:.6f}")
    
    # Hypothesis testing summary
    if 'hypothesis_testing' in stats_results:
        hyp_tests = stats_results['hypothesis_testing']
        
        if 'local_stability' in hyp_tests:
            stability = hyp_tests['local_stability']
            print(f"Local Stability Test:")
            print(f"  p-value: {stability['p_value']:.6f}")
            print(f"  Result: {'STABLE (C₁ > 0)' if stability['significant'] else 'INCONCLUSIVE'}")
        
        if 'cubic_significance' in hyp_tests:
            cubic = hyp_tests['cubic_significance']
            print(f"Cubic Term Test:")
            print(f"  p-value: {cubic['p_value']:.6f}")
            print(f"  Result: {'SIGNIFICANT' if cubic['significant'] else 'NEGLIGIBLE'}")
    
    print(f"")
    print(f"Generated Files:")
    for file_type, filename in results['files'].items():
        print(f"  • {filename}")
    
    print(f"\n🎉 Experiment 1 analysis pipeline completed successfully!")
    print(f"📊 Check the generated plots and CSV files for detailed results")

if __name__ == "__main__":
    main()
