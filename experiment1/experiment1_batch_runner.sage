# ############################################################################
#
# EXPERIMENT 1: BATCH ANALYSIS RUNNER
# ====================================
#
# This script runs multiple configurations of Experiment 1 to test:
# 1. Different zero heights (γ values)
# 2. Different test function types (Gaussian vs Fourier)
# 3. Different precision levels
# 4. Robustness of C₁ > 0 stability result
#
# Usage:
#   sage experiment1_batch_runner.sage
#
# ############################################################################

import time
import json
from pathlib import Path

# Import the orchestrator
load('experiment1_orchestrator.sage')

class Experiment1BatchRunner:
    """Batch runner for multiple Experiment 1 configurations."""
    
    def __init__(self):
        """Initialize the batch runner."""
        self.results_summary = []
        self.config_files = [
            'experiment1_config.json',                    # Original
            'experiment1_config_gamma2.json',             # Second zero
            'experiment1_config_gamma3_fourier.json',     # Third zero + Fourier
            'experiment1_config_high_precision.json'      # High precision
        ]
        
        print("=" * 80)
        print("EXPERIMENT 1: BATCH ANALYSIS RUNNER")
        print("=" * 80)
        print(f"Configurations to run: {len(self.config_files)}")
        
    def run_batch_analysis(self):
        """Run all configurations and collect results."""
        start_time = time.time()
        
        for i, config_file in enumerate(self.config_files):
            print(f"\n{'='*60}")
            print(f"BATCH RUN {i+1}/{len(self.config_files)}: {config_file}")
            print(f"{'='*60}")
            
            try:
                # Load configuration
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Run experiment
                orchestrator = Experiment1Orchestrator(config)
                results = orchestrator.run_complete_analysis()
                
                # Extract key results for summary
                math_results = results['mathematical_results']
                stats_results = results['statistical_results']
                
                summary = {
                    'config_file': config_file,
                    'gamma': config['gamma'],
                    'test_function_type': config['test_function_type'],
                    'delta_range': config['delta_range'],
                    'num_test_functions': config['num_test_functions'],
                    'output_prefix': config['output_prefix'],
                    'C1_numerical': math_results['derivative_analysis']['C1_estimate'],
                    'C2_numerical': math_results['derivative_analysis']['C2_estimate'],
                    'max_delta_E': math_results['derivative_analysis']['max_delta_E'],
                    'analysis_time': results['analysis_time']
                }
                
                # Add statistical results if available
                if 'fitting_results' in stats_results:
                    fitting = stats_results['fitting_results']
                    
                    if 'cubic' in fitting and fitting['cubic'] is not None:
                        cubic_fit = fitting['cubic']
                        summary.update({
                            'C1_fitted': cubic_fit['C1'],
                            'C1_stderr': cubic_fit['C1_stderr'],
                            'C2_fitted': cubic_fit['C2'],
                            'C2_stderr': cubic_fit['C2_stderr'],
                            'r_squared': cubic_fit['r_squared']
                        })
                    
                    if 'best_model' in stats_results and stats_results['best_model']:
                        summary['best_model'] = stats_results['best_model']['model_name']
                
                # Add hypothesis test results
                if 'hypothesis_testing' in stats_results:
                    hyp_tests = stats_results['hypothesis_testing']
                    
                    if 'local_stability' in hyp_tests:
                        stability = hyp_tests['local_stability']
                        summary.update({
                            'stability_p_value': stability['p_value'],
                            'stability_significant': stability['significant']
                        })
                    
                    if 'cubic_significance' in hyp_tests:
                        cubic_test = hyp_tests['cubic_significance']
                        summary.update({
                            'cubic_p_value': cubic_test['p_value'],
                            'cubic_significant': cubic_test['significant']
                        })
                
                self.results_summary.append(summary)
                print(f"✓ Configuration {config_file} completed successfully")
                
            except Exception as e:
                print(f"✗ Configuration {config_file} failed: {e}")
                error_summary = {
                    'config_file': config_file,
                    'error': str(e),
                    'status': 'FAILED'
                }
                self.results_summary.append(error_summary)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("BATCH ANALYSIS COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Successful runs: {len([r for r in self.results_summary if 'error' not in r])}")
        print(f"Failed runs: {len([r for r in self.results_summary if 'error' in r])}")
        
        # Generate comparative report
        self.generate_comparative_report()
        
        return self.results_summary
    
    def generate_comparative_report(self):
        """Generate a comparative analysis report across all runs."""
        report_filename = "experiment1_batch_comparison_report.txt"
        
        with open(report_filename, 'w') as f:
            f.write("EXPERIMENT 1: BATCH ANALYSIS COMPARATIVE REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Configurations Tested: {len(self.config_files)}\n\n")
            
            # Configuration comparison table
            f.write("CONFIGURATION COMPARISON:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Config':<25} {'γ':<12} {'Type':<10} {'δ_range':<10} {'Test_Funcs':<12}\n")
            f.write("-" * 70 + "\n")
            
            for summary in self.results_summary:
                if 'error' in summary:
                    continue
                config_name = summary['config_file'].replace('experiment1_config', '').replace('.json', '')
                if not config_name:
                    config_name = 'original'
                
                f.write(f"{config_name:<25} {summary['gamma']:<12.3f} {summary['test_function_type']:<10} "
                       f"{summary['delta_range']:<10.3f} {summary['num_test_functions']:<12}\n")
            
            f.write("\n")
            
            # Results comparison
            f.write("STABILITY COEFFICIENT (C₁) COMPARISON:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Config':<25} {'C₁ (fitted)':<15} {'Std Error':<12} {'p-value':<12} {'Stable?':<8}\n")
            f.write("-" * 75 + "\n")
            
            successful_runs = [r for r in self.results_summary if 'error' not in r and 'C1_fitted' in r]
            
            for summary in successful_runs:
                config_name = summary['config_file'].replace('experiment1_config', '').replace('.json', '')
                if not config_name:
                    config_name = 'original'
                
                c1_fitted = summary.get('C1_fitted', 'N/A')
                c1_stderr = summary.get('C1_stderr', 'N/A')
                p_value = summary.get('stability_p_value', 'N/A')
                stable = 'YES' if summary.get('stability_significant', False) else 'NO'
                
                if isinstance(c1_fitted, (int, float)):
                    c1_str = f"{c1_fitted:.6e}"
                else:
                    c1_str = str(c1_fitted)
                    
                if isinstance(c1_stderr, (int, float)):
                    stderr_str = f"{c1_stderr:.2e}"
                else:
                    stderr_str = str(c1_stderr)
                    
                if isinstance(p_value, (int, float)):
                    p_str = f"{p_value:.2e}" if p_value > 0 else "< 1e-16"
                else:
                    p_str = str(p_value)
                
                f.write(f"{config_name:<25} {c1_str:<15} {stderr_str:<12} {p_str:<12} {stable:<8}\n")
            
            f.write("\n")
            
            # Statistical summary
            if successful_runs:
                c1_values = [r['C1_fitted'] for r in successful_runs if 'C1_fitted' in r]
                if c1_values:
                    import numpy as np
                    f.write("CROSS-CONFIGURATION STATISTICAL SUMMARY:\n")
                    f.write("-" * 45 + "\n")
                    f.write(f"C₁ values across configurations:\n")
                    f.write(f"  Mean: {np.mean(c1_values):.6e}\n")
                    f.write(f"  Std:  {np.std(c1_values):.6e}\n")
                    f.write(f"  Min:  {np.min(c1_values):.6e}\n")
                    f.write(f"  Max:  {np.max(c1_values):.6e}\n")
                    f.write(f"  Range: {np.max(c1_values) - np.min(c1_values):.6e}\n\n")
                    
                    all_stable = all(r.get('stability_significant', False) for r in successful_runs)
                    f.write(f"Stability consistency: {'ALL CONFIGURATIONS STABLE' if all_stable else 'MIXED RESULTS'}\n")
            
            # Conclusions
            f.write("\nCONCLUSIONS:\n")
            f.write("-" * 12 + "\n")
            
            if successful_runs:
                all_stable = all(r.get('stability_significant', False) for r in successful_runs)
                if all_stable:
                    f.write("✓ LOCAL STABILITY (C₁ > 0) CONFIRMED ACROSS ALL CONFIGURATIONS\n")
                    f.write("  - Result is robust to different zero heights (γ values)\n")
                    f.write("  - Result is robust to different test function types\n")
                    f.write("  - Result holds across different precision levels\n")
                else:
                    f.write("⚠ MIXED STABILITY RESULTS ACROSS CONFIGURATIONS\n")
                    f.write("  - Further investigation needed for inconsistent cases\n")
                    
                all_quadratic = all(not r.get('cubic_significant', True) for r in successful_runs if 'cubic_significant' in r)
                if all_quadratic:
                    f.write("✓ QUADRATIC DOMINANCE CONFIRMED ACROSS ALL CONFIGURATIONS\n")
                    f.write("  - Cubic terms (C₂) are negligible in all cases\n")
                    f.write("  - Energy functional follows ΔE ≈ C₁δ² behavior\n")
            else:
                f.write("✗ NO SUCCESSFUL RUNS - UNABLE TO DRAW CONCLUSIONS\n")
        
        print(f"✓ Comparative report saved: '{report_filename}'")
    
    def print_quick_summary(self):
        """Print a quick summary of batch results."""
        print("\n" + "="*60)
        print("QUICK BATCH SUMMARY")
        print("="*60)
        
        successful_runs = [r for r in self.results_summary if 'error' not in r]
        failed_runs = [r for r in self.results_summary if 'error' in r]
        
        print(f"Successful runs: {len(successful_runs)}/{len(self.config_files)}")
        
        if successful_runs:
            stable_runs = [r for r in successful_runs if r.get('stability_significant', False)]
            print(f"Stable configurations (C₁ > 0): {len(stable_runs)}/{len(successful_runs)}")
            
            if 'C1_fitted' in successful_runs[0]:
                c1_values = [r['C1_fitted'] for r in successful_runs]
                print(f"C₁ range: [{min(c1_values):.2e}, {max(c1_values):.2e}]")
        
        if failed_runs:
            print(f"Failed configurations:")
            for r in failed_runs:
                print(f"  - {r['config_file']}: {r['error']}")

def main():
    """Main entry point for batch analysis."""
    runner = Experiment1BatchRunner()
    results = runner.run_batch_analysis()
    runner.print_quick_summary()
    
    print(f"\n🎉 Batch analysis completed!")
    print(f"📊 Check experiment1_batch_comparison_report.txt for detailed comparison")

if __name__ == "__main__":
    main()
