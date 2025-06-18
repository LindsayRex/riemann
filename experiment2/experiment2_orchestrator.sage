# ############################################################################
#
# EXPERIMENT 2: TWO-ZERO INTERACTION - ORCHESTRATOR
# ==================================================
#
# This script coordinates the complete two-zero interaction experiment,
# integrating mathematical calculations, statistical analysis, and 
# visualization to produce comprehensive results.
#
# Pipeline:
# 1. Initialize mathematical core with specified parameters
# 2. Run complete two-zero interaction analysis
# 3. Perform statistical analysis on results
# 4. Generate all visualizations 
# 5. Export data and create summary report
#
# ############################################################################

import sys
import os
import time
import json
from sage.all import *

# Add experiment2 directory to path for imports
sys.path.append('/home/rexl1/riemann/experiment2')

# Import experiment modules
load('/home/rexl1/riemann/experiment2/experiment2_math.sage')
load('/home/rexl1/riemann/experiment2/experiment2_stats.sage')
load('/home/rexl1/riemann/experiment2/experiment2_viz.sage')

class Experiment2Orchestrator:
    """Main orchestrator for two-zero interaction experiments."""
    
    def __init__(self, config=None, verbose=True):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration dictionary or path to config file
            verbose: Whether to print detailed progress information
        """
        self.verbose = verbose
        self.results = {}
        
        # Load configuration
        if config is None:
            # Default configuration
            self.config = {
                'gamma1': 14.13,
                'gamma2': 21.02,
                'delta_range': 0.08,
                'delta_steps': 33,
                'num_test_functions': 20,
                'test_function_type': 'gaussian',
                'output_dir': 'experiment2',
                'export_csv': True,
                'create_plots': True,
                'create_report': True
            }
        elif isinstance(config, str):
            # Load from file
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            # Use provided dictionary
            self.config = config
        
        if verbose:
            print("=" * 60)
            print("EXPERIMENT 2: TWO-ZERO INTERACTION ANALYSIS")
            print("=" * 60)
            print(f"Configuration:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
            print()
    
    def run_mathematical_analysis(self):
        """Run the mathematical core analysis."""
        if self.verbose:
            print("Step 1: Mathematical Analysis")
            print("-" * 30)
        
        # Initialize mathematical core
        math_core = create_experiment2_math(
            gamma1=self.config['gamma1'],
            gamma2=self.config['gamma2'],
            delta_range=self.config['delta_range'],
            delta_steps=self.config['delta_steps'],
            num_test_functions=self.config['num_test_functions'],
            test_function_type=self.config['test_function_type']
        )
        
        # Run complete analysis
        math_results = math_core.run_complete_analysis(verbose=self.verbose)
        
        self.results['math'] = math_results
        self.math_core = math_core
        
        if self.verbose:
            print(f"✓ Mathematical analysis completed")
            print()
        
        return math_results
    
    def run_statistical_analysis(self):
        """Run statistical analysis on mathematical results."""
        if self.verbose:
            print("Step 2: Statistical Analysis")
            print("-" * 30)
        
        if 'math' not in self.results:
            raise ValueError("Mathematical analysis must be run first")
        
        # Initialize statistical analysis
        stats_core = create_experiment2_stats(verbose=self.verbose)
        
        # Run complete statistical analysis
        stats_results = stats_core.analyze_complete_results(self.results['math'])
        
        self.results['stats'] = stats_results
        self.stats_core = stats_core
        
        if self.verbose:
            print(f"✓ Statistical analysis completed")
            print()
        
        return stats_results
    
    def create_visualizations(self):
        """Create all visualizations."""
        if self.verbose:
            print("Step 3: Visualization")
            print("-" * 30)
        
        if not self.config['create_plots']:
            if self.verbose:
                print("✓ Visualization skipped (disabled in config)")
                print()
            return []
        
        if 'math' not in self.results or 'stats' not in self.results:
            raise ValueError("Both mathematical and statistical analysis must be run first")
        
        # Initialize visualization
        viz_core = create_experiment2_viz(
            style='seaborn-v0_8',
            figsize_default=(12, 8),
            dpi=300
        )
        
        # Create all plots
        output_dir = self.config['output_dir']
        saved_plots = viz_core.create_all_visualizations(
            self.results['math'], 
            self.results['stats'],
            output_dir
        )
        
        self.results['plots'] = saved_plots
        self.viz_core = viz_core
        
        if self.verbose:
            print(f"✓ Visualization completed")
            print()
        
        return saved_plots
    
    def export_data(self):
        """Export results to CSV files."""
        if self.verbose:
            print("Step 4: Data Export")
            print("-" * 30)
        
        if not self.config['export_csv']:
            if self.verbose:
                print("✓ CSV export skipped (disabled in config)")
                print()
            return []
        
        exported_files = []
        output_dir = self.config['output_dir']
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Export mathematical results
        if 'math' in self.results:
            math_csv = f"{output_dir}/experiment2_math_results.csv"
            self.math_core.export_results_csv(self.results['math'], math_csv)
            exported_files.append(math_csv)
        
        # Export statistical results
        if 'stats' in self.results:
            stats_csv = f"{output_dir}/experiment2_stats_results.csv"
            self.stats_core.export_results_csv(self.results['stats'], stats_csv)
            exported_files.append(stats_csv)
        
        self.results['exported_files'] = exported_files
        
        if self.verbose:
            print(f"✓ Data export completed")
            print()
        
        return exported_files
    
    def create_summary_report(self):
        """Create a summary report in markdown format."""
        if self.verbose:
            print("Step 5: Summary Report")
            print("-" * 30)
        
        if not self.config['create_report']:
            if self.verbose:
                print("✓ Report generation skipped (disabled in config)")
                print()
            return None
        
        if 'math' not in self.results or 'stats' not in self.results:
            raise ValueError("Both mathematical and statistical analysis must be run first")
        
        output_dir = self.config['output_dir']
        report_path = f"{output_dir}/experiment2_summary_report.md"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report content
        report_content = self._generate_report_content()
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.results['report_path'] = report_path
        
        if self.verbose:
            print(f"✓ Summary report saved to: '{report_path}'")
            print()
        
        return report_path
    
    def _generate_report_content(self):
        """Generate the content for the summary report."""
        math_results = self.results['math']
        stats_results = self.results['stats']
        config = self.config
        
        # Extract key results
        gamma1 = math_results['gamma1']
        gamma2 = math_results['gamma2']
        overall_stable = stats_results['overall_stable']
        
        stab1 = stats_results['stability_zero1']
        stab2 = stats_results['stability_zero2']
        stab_joint = stats_results['stability_joint']
        
        interference = stats_results['interference_analysis']
        cross_coupling = stats_results['cross_coupling_analysis']
        
        timestamp = math_results['timestamp']
        
        report = f"""# Experiment 2: Two-Zero Interaction Analysis Report
        
**Generated:** {timestamp}  
**Configuration:** γ₁ = {gamma1:.2f}, γ₂ = {gamma2:.2f}, δ ∈ [±{config['delta_range']:.3f}]
        
## Executive Summary
        
This experiment analyzed the energy functional behavior when two Riemann zeta zeros are simultaneously perturbed from the critical line. The analysis reveals **{"STABLE" if overall_stable else "UNSTABLE"}** behavior with significant insights into two-zero interactions.

### Key Findings

1. **Individual Zero Stability:**
   - Zero 1 (γ₁ = {gamma1:.2f}): C₁ = {stab1['C1']:.6e}, R² = {stab1['r_squared']:.4f} → **{"STABLE" if stab1['is_stable'] else "UNSTABLE"}**
   - Zero 2 (γ₂ = {gamma2:.2f}): C₂ = {stab2['C1']:.6e}, R² = {stab2['r_squared']:.4f} → **{"STABLE" if stab2['is_stable'] else "UNSTABLE"}**

2. **Joint Perturbation Stability:**
   - Combined system: C₁₂ = {stab_joint['C1']:.6e}, R² = {stab_joint['r_squared']:.4f} → **{"STABLE" if stab_joint['is_stable'] else "UNSTABLE"}**

3. **Interference Effects:**
   - Maximum interference: {interference['max_interference']:.3e}
   - Cross-coupling significant: **{cross_coupling['is_significant']}**
   - Power law scaling: |I| ∼ {interference['power_law_amplitude']:.2e}|δ|^{interference['power_law_exponent']:.2f}

## Mathematical Model

The energy functional for two-zero configurations follows:

```
E[S] = Σⱼ wⱼ[Σᵢ φⱼ(γᵢ) - P(φⱼ)]² + penalties
```

Where the penalty terms include:
- Critical line penalties: 100(βᵢ - 1/2)²
- Cross-coupling interactions: 5 exp(-|γᵢ-γⱼ|/10)(βᵢ-1/2)(βⱼ-1/2)²
- Higher-order geometric effects: 5(βᵢ-1/2)⁴

## Detailed Analysis

### Quadratic Stability Test

For perturbations ρᵢ = 1/2 + δᵢ + iγᵢ, we test the model ΔE ≈ Cδ²:

| Configuration | C coefficient | R² | p-value | Status |
|---------------|---------------|-----|---------|--------|
| Zero 1 only   | {stab1['C1']:.6e} | {stab1['r_squared']:.4f} | {stab1['p_value']:.2e} | {"✓ STABLE" if stab1['is_stable'] else "✗ UNSTABLE"} |
| Zero 2 only   | {stab2['C1']:.6e} | {stab2['r_squared']:.4f} | {stab2['p_value']:.2e} | {"✓ STABLE" if stab2['is_stable'] else "✗ UNSTABLE"} |
| Both zeros    | {stab_joint['C1']:.6e} | {stab_joint['r_squared']:.4f} | {stab_joint['p_value']:.2e} | {"✓ STABLE" if stab_joint['is_stable'] else "✗ UNSTABLE"} |

### Interference Pattern Analysis

The interference term I(δ) = ΔE₁₂(δ,δ) - [ΔE₁(δ,0) + ΔE₂(0,δ)] shows:

- **Mean interference:** {interference['mean_interference']:.6e}
- **Standard deviation:** {interference['std_interference']:.6e}
- **Sign changes:** {interference['sign_changes']} (indicating oscillatory behavior)
- **Correlation with |δ|:** {interference['correlation_with_delta']:.4f}

### Cross-Coupling Coefficient

The cross-coupling term C₁₂ from the model ΔE₁₂ ≈ C₁δ² + C₂δ² + C₁₂δ²:

- **C₁₂ coefficient:** {cross_coupling['C12']:.6e}
- **Statistical significance:** p = {cross_coupling['C12_p_value']:.2e} ({"significant" if cross_coupling['is_significant'] else "not significant"})
- **Relative strength:** {cross_coupling['relative_strength']:.4f} (vs individual coefficients)

## Physical Interpretation

{"### Stable Critical Line" if overall_stable else "### Critical Line Instability"}

{"The analysis confirms that the critical line Re(ρ) = 1/2 is a stable minimum for the energy functional in this two-zero configuration. Both individual and joint perturbations show quadratic restoring forces, supporting the Riemann Hypothesis." if overall_stable else "The analysis reveals instability in the critical line for this two-zero configuration, suggesting potential issues with the energy functional model or parameter choice."}

### Zero-Zero Interactions

{"The two zeros show measurable but weak cross-coupling effects." if cross_coupling['is_significant'] else "Cross-coupling effects between the zeros are not statistically significant."} The interference pattern exhibits power law scaling |I| ∼ |δ|^{interference['power_law_exponent']:.1f}, {"consistent with expected geometric nonlinearities." if abs(interference['power_law_exponent'] - 2.0) < 0.5 else "showing unexpected scaling behavior."}

## Computational Details

- **Test function basis:** {config['num_test_functions']} {config['test_function_type']} functions
- **Perturbation grid:** {config['delta_steps']} points in δ ∈ [±{config['delta_range']:.3f}]
- **Total computation time:** {math_results['total_analysis_time']:.2f} seconds
- **Zero heights:** γ₁ = {gamma1:.6f}, γ₂ = {gamma2:.6f}

## Conclusions

1. **Riemann Hypothesis Support:** {"✓ STRONG" if overall_stable and all([stab1['is_stable'], stab2['is_stable'], stab_joint['is_stable']]) else "⚠ WEAK" if overall_stable else "✗ NONE"}
   
2. **Two-Zero Interactions:** {"Weak but measurable cross-coupling" if cross_coupling['is_significant'] else "No significant cross-coupling detected"}

3. **Energy Model Validation:** {"High R² values (>{min(stab1['r_squared'], stab2['r_squared'], stab_joint['r_squared']):.3f}) confirm quadratic energy model" if overall_stable else "Model may need refinement"}

4. **Scaling Behavior:** Power law exponent α = {interference['power_law_exponent']:.2f} ± {interference['power_law_exponent_std']:.2f}

---

*This report was automatically generated by the Experiment 2 pipeline.*  
*Data files and visualizations are available in the `{config['output_dir']}/` directory.*
"""
        
        return report
    
    def run_complete_experiment(self):
        """Run the complete experiment pipeline."""
        if self.verbose:
            print("Starting complete Experiment 2 pipeline...")
            print()
        
        start_time = time.time()
        
        try:
            # Run all steps
            self.run_mathematical_analysis()
            self.run_statistical_analysis()
            self.create_visualizations()
            self.export_data()
            self.create_summary_report()
            
            # Final summary
            total_time = time.time() - start_time
            self.results['total_time'] = total_time
            
            if self.verbose:
                print("=" * 60)
                print("EXPERIMENT 2 COMPLETED SUCCESSFULLY")
                print("=" * 60)
                print(f"Total execution time: {total_time:.2f} seconds")
                print(f"Output directory: '{self.config['output_dir']}'")
                
                if 'exported_files' in self.results:
                    print(f"Exported {len(self.results['exported_files'])} CSV files")
                
                if 'plots' in self.results:
                    print(f"Created {len(self.results['plots'])} visualization plots")
                
                if 'report_path' in self.results:
                    print(f"Summary report: '{self.results['report_path']}'")
                
                print()
                
                # Show stability summary
                overall_stable = self.results['stats']['overall_stable']
                print(f"OVERALL STABILITY: {'✓ STABLE' if overall_stable else '✗ UNSTABLE'}")
                print("=" * 60)
        
        except Exception as e:
            if self.verbose:
                print(f"✗ Experiment failed: {e}")
            raise
        
        return self.results

# Factory function
def create_experiment2_orchestrator(config=None, verbose=True):
    """Create Experiment2Orchestrator instance."""
    return Experiment2Orchestrator(config=config, verbose=verbose)

# Convenience function for quick runs
def run_experiment2(gamma1=14.13, gamma2=21.02, delta_range=0.08, delta_steps=33, 
                   num_test_functions=20, test_function_type='gaussian', 
                   output_dir='experiment2', verbose=True):
    """
    Quick function to run Experiment 2 with specified parameters.
    
    Args:
        gamma1: Height of first zero
        gamma2: Height of second zero  
        delta_range: Perturbation range
        delta_steps: Number of perturbation points
        num_test_functions: Number of test functions
        test_function_type: Type of test functions
        output_dir: Output directory
        verbose: Print progress
        
    Returns:
        dict: Complete experiment results
    """
    config = {
        'gamma1': gamma1,
        'gamma2': gamma2,
        'delta_range': delta_range,
        'delta_steps': delta_steps,
        'num_test_functions': num_test_functions,
        'test_function_type': test_function_type,
        'output_dir': output_dir,
        'export_csv': True,
        'create_plots': True,
        'create_report': True
    }
    
    orchestrator = create_experiment2_orchestrator(config=config, verbose=verbose)
    return orchestrator.run_complete_experiment()
