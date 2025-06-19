# ############################################################################
#
# EXPERIMENT 1: SINGLE-ZERO PERTURBATION - VISUALIZATION MODULE
# ==============================================================
#
# This module provides comprehensive plotting and visualization for Experiment 1
# results including:
# - ΔE vs δ plots with polynomial fits and confidence bands
# - ΔE vs δ² plots to verify quadratic behavior
# - Residual plots for model validation
# - Gradient and curvature analysis plots
# - Bootstrap distribution visualizations
# - Publication-quality figures with error bars and p-values
#
# ############################################################################

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import stats as scipy_stats
import time
import h5py

class Experiment1Visualization:
    """Visualization module for single-zero perturbation experiment."""
    
    def __init__(self, figsize=(20, 12), dpi=300):
        """
        Initialize visualization module.
        
        Args:
            figsize: Figure size for plots
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set matplotlib style for publication quality
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'lines.markersize': 4
        })
        
        # Color scheme
        self.colors = {
            'data': '#1f77b4',
            'quadratic': '#ff7f0e', 
            'cubic': '#2ca02c',
            'quartic': '#d62728',
            'confidence': '#1f77b4',
            'residuals': '#8c564b',
            'critical': '#e377c2'
        }
    
    def plot_energy_vs_delta(self, delta_values, delta_E_values, fitting_results=None, 
                           stats_results=None, ax=None):
        """
        Plot ΔE vs δ with polynomial fits and confidence bands.
        
        Args:
            delta_values: Array of δ perturbation values
            delta_E_values: Array of ΔE values
            fitting_results: Polynomial fitting results
            stats_results: Statistical analysis results
            ax: Matplotlib axis (optional)
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw data
        ax.scatter(delta_values, delta_E_values, alpha=0.7, color=self.colors['data'],
                  s=30, label='ΔE data', zorder=3)
        
        # Plot polynomial fits if available
        if fitting_results is not None:
            delta_fine = np.linspace(np.min(delta_values), np.max(delta_values), 200)
            
            # Quadratic fit
            if 'quadratic' in fitting_results and fitting_results['quadratic'] is not None:
                quad_fit = fitting_results['quadratic']
                C1 = quad_fit['C1']
                quad_pred = C1 * delta_fine**2
                
                ax.plot(delta_fine, quad_pred, color=self.colors['quadratic'], 
                       linestyle='--', linewidth=2,
                       label=f'Quadratic: C₁δ² (C₁={C1:.2e})')
            
            # Cubic fit
            if 'cubic' in fitting_results and fitting_results['cubic'] is not None:
                cubic_fit = fitting_results['cubic']
                C1, C2 = cubic_fit['C1'], cubic_fit['C2']
                cubic_pred = C1 * delta_fine**2 + C2 * delta_fine**3
                
                ax.plot(delta_fine, cubic_pred, color=self.colors['cubic'],
                       linewidth=2,
                       label=f'Cubic: C₁δ² + C₂δ³ (C₁={C1:.2e}, C₂={C2:.2e})')
        
        # Add confidence bands if bootstrap results available
        if (stats_results is not None and 'bootstrap_analysis' in stats_results and 
            stats_results['bootstrap_analysis'] is not None):
            self._add_confidence_bands(ax, delta_fine, stats_results['bootstrap_analysis'])
        
        # Reference line at ΔE = 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2, 
                  label='Critical line (δ=0)')
        
        # Formatting
        ax.set_xlabel('Perturbation δ from critical line')
        ax.set_ylabel('Energy difference ΔE(δ)')
        ax.set_title('Single-Zero Perturbation: Energy vs Displacement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_energy_vs_delta_squared(self, delta_values, delta_E_values, 
                                   fitting_results=None, ax=None):
        """
        Plot ΔE vs δ² to verify quadratic behavior.
        
        Args:
            delta_values: Array of δ values
            delta_E_values: Array of ΔE values
            fitting_results: Polynomial fitting results
            ax: Matplotlib axis (optional)
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        delta_squared = delta_values**2
        
        # Plot data
        ax.scatter(delta_squared, delta_E_values, alpha=0.7, color=self.colors['data'],
                  s=30, label='ΔE vs δ²')
        
        # Linear fit to verify quadratic behavior
        # Remove δ=0 point for fitting
        mask = np.abs(delta_values) > 1e-12
        delta_sq_fit = delta_squared[mask]
        delta_E_fit = delta_E_values[mask]
        
        if len(delta_sq_fit) > 1:
            # Linear regression
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                delta_sq_fit, delta_E_fit
            )
            
            # Plot fit line
            delta_sq_range = np.linspace(0, np.max(delta_squared), 100)
            fit_line = slope * delta_sq_range + intercept
            
            ax.plot(delta_sq_range, fit_line, color=self.colors['quadratic'],
                   linewidth=2, 
                   label=f'Linear fit: slope={slope:.2e} (R²={r_value**2:.4f})')
        
        # Add quadratic coefficient from polynomial fit if available
        if (fitting_results is not None and 'quadratic' in fitting_results and 
            fitting_results['quadratic'] is not None):
            C1 = fitting_results['quadratic']['C1']
            theoretical_line = C1 * delta_sq_range
            ax.plot(delta_sq_range, theoretical_line, color=self.colors['cubic'],
                   linestyle='--', linewidth=2,
                   label=f'Theoretical: C₁δ² (C₁={C1:.2e})')
        
        # Formatting
        ax.set_xlabel('δ² (squared perturbation)')
        ax.set_ylabel('Energy difference ΔE(δ)')
        ax.set_title('Quadratic Behavior Verification: ΔE vs δ²')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_residual_analysis(self, fitting_results, delta_values, ax=None):
        """
        Plot residual analysis for model validation.
        
        Args:
            fitting_results: Polynomial fitting results
            delta_values: Original δ values
            ax: Matplotlib axis (optional)
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Remove δ=0 point
        mask = np.abs(delta_values) > 1e-12
        delta_fit = delta_values[mask]
        
        # Plot residuals for each model
        colors = [self.colors['quadratic'], self.colors['cubic'], self.colors['quartic']]
        
        for i, (model_name, fit_result) in enumerate(fitting_results.items()):
            if fit_result is None:
                continue
                
            residuals = fit_result['residuals']
            color = colors[i % len(colors)]
            
            ax.scatter(delta_fit, residuals, alpha=0.7, color=color, s=25,
                      label=f'{model_name.capitalize()} residuals')
        
        # Reference line at residual = 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Formatting
        ax.set_xlabel('Perturbation δ')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Analysis: Model Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_gradient_analysis(self, delta_values, delta_E_values, 
                             derivative_analysis=None, ax=None):
        """
        Plot gradient and curvature analysis.
        
        Args:
            delta_values: Array of δ values
            delta_E_values: Array of ΔE values
            derivative_analysis: Numerical derivative results
            ax: Matplotlib axis (optional)
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if derivative_analysis is not None:
            gradient = derivative_analysis['gradient']
            
            # Plot gradient
            ax.plot(delta_values, gradient, color=self.colors['cubic'], linewidth=2,
                   marker='o', markersize=3, label='dΔE/dδ (numerical)')
            
            # Highlight gradient at δ=0
            zero_idx = derivative_analysis['zero_index']
            gradient_at_zero = derivative_analysis['gradient_at_zero']
            
            ax.scatter([delta_values[zero_idx]], [gradient_at_zero], 
                      color='red', s=100, zorder=5,
                      label=f'Gradient at δ=0: {gradient_at_zero:.2e}')
        
        # Reference line at gradient = 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        # Formatting
        ax.set_xlabel('Perturbation δ')
        ax.set_ylabel('dΔE/dδ')
        ax.set_title('Gradient Analysis: Local Stability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_bootstrap_distributions(self, bootstrap_stats, ax=None):
        """
        Plot bootstrap distributions of coefficients.
        
        Args:
            bootstrap_stats: Bootstrap analysis results
            ax: Matplotlib axis (optional)
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if bootstrap_stats is None or 'C1' not in bootstrap_stats:
            ax.text(0.5, 0.5, 'No bootstrap data available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            return ax
        
        C1_values = bootstrap_stats['C1']['values']
        
        # Histogram of C1 values
        ax.hist(C1_values, bins=50, alpha=0.7, density=True, 
               color=self.colors['confidence'], edgecolor='black', linewidth=0.5,
               label=f'C₁ distribution (n={len(C1_values)})')
        
        # Add statistics
        C1_mean = bootstrap_stats['C1']['mean']
        C1_ci_lower = bootstrap_stats['C1']['ci_lower']
        C1_ci_upper = bootstrap_stats['C1']['ci_upper']
        
        # Vertical lines for mean and CI
        ax.axvline(C1_mean, color='red', linewidth=2, 
                  label=f'Mean: {C1_mean:.2e}')
        ax.axvline(C1_ci_lower, color='orange', linestyle='--', linewidth=2,
                  label=f'95% CI: [{C1_ci_lower:.2e}, {C1_ci_upper:.2e}]')
        ax.axvline(C1_ci_upper, color='orange', linestyle='--', linewidth=2)
        
        # Vertical line at C1 = 0 (stability threshold)
        ax.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=2,
                  label='Stability threshold (C₁=0)')
        
        # Shade region where C1 > 0 (stable)
        x_range = ax.get_xlim()
        ax.axvspan(0, x_range[1], alpha=0.2, color='green', 
                  label='Stable region (C₁>0)')
        
        # Formatting
        ax.set_xlabel('C₁ coefficient value')
        ax.set_ylabel('Probability density')
        ax.set_title('Bootstrap Distribution: Quadratic Coefficient C₁')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_hypothesis_testing_summary(self, hypothesis_tests, ax=None):
        """
        Create a visual summary of hypothesis testing results.
        
        Args:
            hypothesis_tests: Hypothesis testing results
            ax: Matplotlib axis (optional)
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.axis('off')
        
        # Create text summary
        y_pos = 0.9
        line_height = 0.08
        
        ax.text(0.05, y_pos, 'HYPOTHESIS TESTING SUMMARY', 
               transform=ax.transAxes, fontsize=16, fontweight='bold')
        y_pos -= line_height * 1.5
        
        # Local stability test
        if 'local_stability' in hypothesis_tests:
            stability = hypothesis_tests['local_stability']
            result_color = 'green' if stability['significant'] else 'orange'
            
            ax.text(0.05, y_pos, '1. LOCAL STABILITY TEST (C₁ > 0):', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold')
            y_pos -= line_height
            
            ax.text(0.1, y_pos, f"H₀: C₁ ≤ 0 (unstable) vs H₁: C₁ > 0 (stable)",
                   transform=ax.transAxes, fontsize=10)
            y_pos -= line_height
            
            ax.text(0.1, y_pos, f"t-statistic: {stability['test_statistic']:.4f}",
                   transform=ax.transAxes, fontsize=10)
            y_pos -= line_height
            
            ax.text(0.1, y_pos, f"p-value: {stability['p_value']:.6f}",
                   transform=ax.transAxes, fontsize=10)
            y_pos -= line_height
            
            result_text = 'STABLE' if stability['significant'] else 'INCONCLUSIVE'
            ax.text(0.1, y_pos, f"Result: {result_text}",
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   color=result_color)
            y_pos -= line_height * 1.5
        
        # Cubic term significance
        if 'cubic_significance' in hypothesis_tests:
            cubic = hypothesis_tests['cubic_significance']
            
            ax.text(0.05, y_pos, '2. CUBIC TERM SIGNIFICANCE (C₂ ≠ 0):', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold')
            y_pos -= line_height
            
            ax.text(0.1, y_pos, f"H₀: C₂ = 0 vs H₁: C₂ ≠ 0",
                   transform=ax.transAxes, fontsize=10)
            y_pos -= line_height
            
            ax.text(0.1, y_pos, f"t-statistic: {cubic['test_statistic']:.4f}",
                   transform=ax.transAxes, fontsize=10)
            y_pos -= line_height
            
            ax.text(0.1, y_pos, f"p-value: {cubic['p_value']:.6f}",
                   transform=ax.transAxes, fontsize=10)
            y_pos -= line_height
            
            result_text = 'SIGNIFICANT' if cubic['significant'] else 'NOT SIGNIFICANT'
            result_color = 'red' if cubic['significant'] else 'green'
            ax.text(0.1, y_pos, f"Result: {result_text}",
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   color=result_color)
            y_pos -= line_height * 1.5
        
        # Bootstrap stability test
        if 'bootstrap_stability' in hypothesis_tests:
            bootstrap = hypothesis_tests['bootstrap_stability']
            
            ax.text(0.05, y_pos, '3. BOOTSTRAP STABILITY CONFIRMATION:', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold')
            y_pos -= line_height
            
            ax.text(0.1, y_pos, f"Fraction of bootstrap samples with C₁ > 0: {bootstrap['fraction_positive']:.4f}",
                   transform=ax.transAxes, fontsize=10)
            y_pos -= line_height
            
            result_text = 'CONFIRMED' if bootstrap['significant'] else 'UNCERTAIN'
            result_color = 'green' if bootstrap['significant'] else 'orange'
            ax.text(0.1, y_pos, f"Bootstrap stability: {result_text}",
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   color=result_color)
        
        return ax
    
    def _add_confidence_bands(self, ax, delta_range, bootstrap_stats):
        """
        Add confidence bands to energy plot using bootstrap results.
        
        Args:
            ax: Matplotlib axis
            delta_range: Range of δ values for prediction
            bootstrap_stats: Bootstrap analysis results
        """
        if 'C1' not in bootstrap_stats:
            return
        
        # Get bootstrap coefficient samples
        C1_samples = bootstrap_stats['C1']['values']
        
        if 'C2' in bootstrap_stats:
            C2_samples = bootstrap_stats['C2']['values']
            n_samples = min(len(C1_samples), len(C2_samples))
            
            # Compute prediction bands
            predictions = []
            for i in range(n_samples):
                pred = C1_samples[i] * delta_range**2 + C2_samples[i] * delta_range**3
                predictions.append(pred)
        else:
            # Quadratic only
            predictions = []
            for C1 in C1_samples:
                pred = C1 * delta_range**2
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute confidence bands
        lower_band = np.percentile(predictions, 2.5, axis=0)
        upper_band = np.percentile(predictions, 97.5, axis=0)
        
        # Plot confidence bands
        ax.fill_between(delta_range, lower_band, upper_band, 
                       alpha=0.2, color=self.colors['confidence'],
                       label='95% confidence band')
    
    def create_comprehensive_plot(self, delta_values, delta_E_values, 
                                fitting_results, stats_results, 
                                derivative_analysis=None,
                                filename="experiment1_comprehensive_analysis.png"):
        """
        Create comprehensive 6-panel analysis plot.
        
        Args:
            delta_values: Array of δ values
            delta_E_values: Array of ΔE values
            fitting_results: Polynomial fitting results
            stats_results: Statistical analysis results
            derivative_analysis: Numerical derivative analysis
            filename: Output filename
            
        Returns:
            str: Saved filename
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('Experiment 1: Single-Zero Perturbation Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: ΔE vs δ with fits
        self.plot_energy_vs_delta(delta_values, delta_E_values, 
                                 fitting_results, stats_results, axes[0, 0])
        
        # Plot 2: ΔE vs δ² (quadratic verification)
        self.plot_energy_vs_delta_squared(delta_values, delta_E_values, 
                                         fitting_results, axes[0, 1])
        
        # Plot 3: Gradient analysis
        self.plot_gradient_analysis(delta_values, delta_E_values, 
                                   derivative_analysis, axes[0, 2])
        
        # Plot 4: Residual analysis
        self.plot_residual_analysis(fitting_results, delta_values, axes[1, 0])
        
        # Plot 5: Bootstrap distributions
        bootstrap_stats = stats_results.get('bootstrap_analysis')
        self.plot_bootstrap_distributions(bootstrap_stats, axes[1, 1])
        
        # Plot 6: Hypothesis testing summary
        hypothesis_tests = stats_results.get('hypothesis_testing', {})
        self.plot_hypothesis_testing_summary(hypothesis_tests, axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        print(f"📊 Comprehensive analysis plot saved: '{filename}'")
        plt.close()
        
        return filename
    
    def create_publication_figure(self, delta_values, delta_E_values, 
                                fitting_results, stats_results,
                                filename="experiment1_publication_figure.png"):
        """
        Create publication-ready figure with key results.
        
        Args:
            delta_values: Array of δ values
            delta_E_values: Array of ΔE values
            fitting_results: Polynomial fitting results
            stats_results: Statistical analysis results
            filename: Output filename
            
        Returns:
            str: Saved filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Single-Zero Energy Perturbation: Critical Line Stability', 
                    fontsize=14, fontweight='bold')
        
        # Main energy plot with fits and confidence bands
        self.plot_energy_vs_delta(delta_values, delta_E_values, 
                                 fitting_results, stats_results, ax1)
        
        # Bootstrap distribution with hypothesis test results
        bootstrap_stats = stats_results.get('bootstrap_analysis')
        self.plot_bootstrap_distributions(bootstrap_stats, ax2)
        
        # Add statistical results as text
        if 'hypothesis_testing' in stats_results:
            hyp_tests = stats_results['hypothesis_testing']
            
            if 'local_stability' in hyp_tests:
                stability = hyp_tests['local_stability']
                result_text = f"Stability Test: p = {stability['p_value']:.2e}\n"
                result_text += f"Result: {'STABLE' if stability['significant'] else 'INCONCLUSIVE'}"
                
                ax1.text(0.05, 0.95, result_text, transform=ax1.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        print(f"📊 Publication figure saved: '{filename}'")
        plt.close()
        
        return filename

# Factory function for easy usage
def create_experiment1_visualization(figsize=(20, 12), dpi=300):
    """
    Factory function to create Experiment1Visualization instance.
    
    Args:
        figsize: Figure size for plots
        dpi: Resolution for saved figures
        
    Returns:
        Experiment1Visualization: Configured visualization instance
    """
    return Experiment1Visualization(figsize=figsize, dpi=dpi)

# ############################################################################
# HDF5 INTEGRATION FOR BATCH ORCHESTRATOR  
# ############################################################################

class Experiment1Viz:
    """
    HDF5-compatible wrapper for visualization engine.
    Implements the interface expected by the batch orchestrator.
    """
    
    def __init__(self, hdf5_file):
        """
        Initialize visualization engine for HDF5 data.
        
        Args:
            hdf5_file: Path to HDF5 file containing experiment data
        """
        self.hdf5_file = hdf5_file
        self.viz_engine = Experiment1Visualization(figsize=(18, 10), dpi=300)
        
        print(f"📈 Experiment1Viz initialized for: {hdf5_file}")
        
    def generate_summary_visualizations(self):
        """
        Generate exactly 5 summary visualization images following Design Guide.
        """
        print("🎨 Generating 5 summary visualization images...")
        
        # Read all data from HDF5
        all_configs_data = self._load_all_configurations()
        
        if not all_configs_data:
            print("✗ No data found in HDF5 file")
            return
            
        # Generate the 5 standardized summary images
        self._generate_image_1_stability_analysis(all_configs_data)
        self._generate_image_2_fit_quality(all_configs_data) 
        self._generate_image_3_energy_patterns(all_configs_data)
        self._generate_image_4_configuration_comparison(all_configs_data)
        self._generate_image_5_parameter_space(all_configs_data)
        
        print("✅ All 5 summary visualizations generated")
        
    def _load_all_configurations(self):
        """Load data from all configurations in HDF5 file."""
        all_data = {}
        
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                for config_name in f.keys():
                    config_group = f[config_name]
                    
                    # Load perturbation analysis data
                    analysis_data = config_group['perturbation_analysis']
                    delta = analysis_data['delta'][:]
                    delta_E = analysis_data['delta_E'][:]
                    
                    # Load metadata
                    metadata = config_group['metadata']
                    gamma = metadata.attrs['gamma']
                    test_function_type = metadata.attrs['test_function_type'].decode() if isinstance(metadata.attrs['test_function_type'], bytes) else metadata.attrs['test_function_type']
                    
                    # Load statistical results if available
                    stats_data = {}
                    if 'statistical_analysis' in config_group:
                        stats_group = config_group['statistical_analysis']
                        stats_data = {key: stats_group.attrs[key] for key in stats_group.attrs.keys()}
                    
                    all_data[config_name] = {
                        'delta': delta,
                        'delta_E': delta_E,
                        'gamma': gamma,
                        'test_function_type': test_function_type,
                        'stats': stats_data
                    }
                    
        except Exception as e:
            print(f"✗ Error loading HDF5 data: {e}")
            return {}
            
        print(f"✓ Loaded data from {len(all_data)} configurations")
        return all_data
        
    def _generate_image_1_stability_analysis(self, all_data):
        """Image 1: Stability Analysis - C₁ coefficients + confidence intervals"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle('Image 1: Stability Analysis - C₁ Coefficients Across Configurations', 
                    fontsize=18, fontweight='bold')
        
        # Extract C1 coefficients and confidence intervals
        config_names = []
        c1_values = []
        c1_errors = []
        gamma_values = []
        
        for config_name, data in all_data.items():
            if 'C1_coefficient' in data['stats']:
                config_names.append(config_name.replace('config_', '').replace('_', ' '))
                c1_values.append(data['stats']['C1_coefficient'])
                c1_errors.append(data['stats']['C1_stderr'])
                gamma_values.append(data['gamma'])
        
        if c1_values:
            # Left plot: C1 coefficients with error bars
            x_pos = range(len(config_names))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            bars = ax1.bar(x_pos, c1_values, yerr=c1_errors, capsize=10, 
                          color=[colors[i % len(colors)] for i in range(len(c1_values))],
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label='Stability threshold (C₁=0)')
            ax1.set_xlabel('Configuration', fontsize=14)
            ax1.set_ylabel('C₁ Coefficient', fontsize=14)
            ax1.set_title('C₁ Coefficients with Standard Errors', fontsize=16)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(config_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Right plot: C1 vs gamma values
            ax2.scatter(gamma_values, c1_values, s=100, alpha=0.8, 
                       c=[colors[i % len(colors)] for i in range(len(c1_values))],
                       edgecolors='black', linewidth=1)
            
            for i, (gamma, c1, config) in enumerate(zip(gamma_values, c1_values, config_names)):
                ax2.annotate(f'γ={gamma:.1f}', (gamma, c1), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax2.set_xlabel('Zero Height γ', fontsize=14)
            ax2.set_ylabel('C₁ Coefficient', fontsize=14)
            ax2.set_title('C₁ vs Zero Height', fontsize=16)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/experiment1_summary_1.png', dpi=300, bbox_inches='tight')
        print("✓ Image 1: Stability Analysis saved")
        plt.close()
        
    def _generate_image_2_fit_quality(self, all_data):
        """Image 2: Fit Quality Assessment - R² values + distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle('Image 2: Fit Quality Assessment - R² Values and Distributions', 
                    fontsize=18, fontweight='bold')
        
        r_squared_values = []
        config_labels = []
        
        for config_name, data in all_data.items():
            if 'r_squared' in data['stats']:
                r_squared_values.append(data['stats']['r_squared'])
                config_labels.append(config_name.replace('config_', '').replace('_', ' '))
        
        if r_squared_values:
            # Left plot: R² bar chart
            x_pos = range(len(config_labels))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            bars = ax1.bar(x_pos, r_squared_values,
                          color=[colors[i % len(colors)] for i in range(len(r_squared_values))],
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            ax1.axhline(y=0.95, color='green', linestyle='--', linewidth=2, alpha=0.7,
                       label='Excellent fit (R²>0.95)')
            ax1.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                       label='Good fit (R²>0.90)')
            ax1.set_xlabel('Configuration', fontsize=14)
            ax1.set_ylabel('R² Value', fontsize=14)
            ax1.set_title('Fit Quality by Configuration', fontsize=16)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(config_labels, rotation=45, ha='right')
            ax1.set_ylim(0, 1.02)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Right plot: R² distribution histogram
            ax2.hist(r_squared_values, bins=10, alpha=0.7, color='skyblue', 
                    edgecolor='black', linewidth=1)
            ax2.axvline(np.mean(r_squared_values), color='red', linestyle='-', linewidth=2,
                       label=f'Mean R² = {np.mean(r_squared_values):.4f}')
            ax2.set_xlabel('R² Value', fontsize=14)
            ax2.set_ylabel('Frequency', fontsize=14)
            ax2.set_title('R² Distribution', fontsize=16)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig('results/experiment1_summary_2.png', dpi=300, bbox_inches='tight')
        print("✓ Image 2: Fit Quality saved")
        plt.close()
        
    def _generate_image_3_energy_patterns(self, all_data):
        """Image 3: Energy Perturbation Patterns - δE vs δ for key configurations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle('Image 3: Energy Perturbation Patterns - ΔE vs δ Comparison', 
                    fontsize=18, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Left plot: All configurations overlay
        for i, (config_name, data) in enumerate(all_data.items()):
            color = colors[i % len(colors)]
            label = f"γ={data['gamma']:.1f} ({data['test_function_type']})"
            
            ax1.plot(data['delta'], data['delta_E'], 'o-', color=color, 
                    alpha=0.8, linewidth=2, markersize=4, label=label)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax1.set_xlabel('Perturbation δ', fontsize=14)
        ax1.set_ylabel('Energy Change ΔE', fontsize=14)
        ax1.set_title('Energy Patterns Across Configurations', fontsize=16)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Quadratic fits comparison
        delta_fine = np.linspace(-0.08, 0.08, 100)
        
        for i, (config_name, data) in enumerate(all_data.items()):
            if 'C1_coefficient' in data['stats']:
                color = colors[i % len(colors)]
                c1 = data['stats']['C1_coefficient']
                label = f"γ={data['gamma']:.1f}: C₁={c1:.2e}"
                
                quadratic_fit = c1 * delta_fine**2
                ax2.plot(delta_fine, quadratic_fit, '--', color=color, 
                        linewidth=2, alpha=0.8, label=label)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax2.set_xlabel('Perturbation δ', fontsize=14)
        ax2.set_ylabel('Quadratic Fit ΔE = C₁δ²', fontsize=14)
        ax2.set_title('Quadratic Behavior Comparison', fontsize=16)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/experiment1_summary_3.png', dpi=300, bbox_inches='tight')
        print("✓ Image 3: Energy Patterns saved")
        plt.close()
        
    def _generate_image_4_configuration_comparison(self, all_data):
        """Image 4: Configuration Comparison - Gamma values vs stability metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle('Image 4: Configuration Comparison - Parameters vs Stability', 
                    fontsize=18, fontweight='bold')
        
        gamma_values = []
        c1_values = []
        p_values = []
        test_types = []
        
        for config_name, data in all_data.items():
            gamma_values.append(data['gamma'])
            test_types.append(data['test_function_type'])
            
            if 'C1_coefficient' in data['stats']:
                c1_values.append(data['stats']['C1_coefficient'])
            else:
                c1_values.append(0)
                
            if 'stability_p_value' in data['stats']:
                p_values.append(data['stats']['stability_p_value'])
            else:
                p_values.append(1.0)
        
        # Left plot: C1 vs gamma with test function type
        gaussian_mask = [t == 'gaussian' for t in test_types]
        fourier_mask = [t == 'fourier' for t in test_types]
        
        if any(gaussian_mask):
            gamma_gauss = [g for g, m in zip(gamma_values, gaussian_mask) if m]
            c1_gauss = [c for c, m in zip(c1_values, gaussian_mask) if m]
            ax1.scatter(gamma_gauss, c1_gauss, s=100, alpha=0.8, 
                       c='blue', marker='o', label='Gaussian', edgecolors='black')
                       
        if any(fourier_mask):
            gamma_four = [g for g, m in zip(gamma_values, fourier_mask) if m]
            c1_four = [c for c, m in zip(c1_values, fourier_mask) if m]
            ax1.scatter(gamma_four, c1_four, s=100, alpha=0.8, 
                       c='red', marker='s', label='Fourier', edgecolors='black')
        
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Zero Height γ', fontsize=14)
        ax1.set_ylabel('C₁ Coefficient', fontsize=14)
        ax1.set_title('Stability vs Zero Height by Test Function', fontsize=16)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: P-values vs gamma
        significance_mask = [p < 0.05 for p in p_values]
        
        ax2.scatter(gamma_values, p_values, s=100, alpha=0.8,
                   c=['green' if sig else 'red' for sig in significance_mask],
                   edgecolors='black', linewidth=1)
        
        ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='Significance threshold (p=0.05)')
        ax2.set_xlabel('Zero Height γ', fontsize=14)
        ax2.set_ylabel('Stability p-value', fontsize=14)
        ax2.set_title('Statistical Significance vs Zero Height', fontsize=16)
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/experiment1_summary_4.png', dpi=300, bbox_inches='tight')
        print("✓ Image 4: Configuration Comparison saved")
        plt.close()
        
    def _generate_image_5_parameter_space(self, all_data):
        """Image 5: Parameter Space Coverage - Test function types and precision levels"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle('Image 5: Parameter Space Coverage - Function Types and Computational Details', 
                    fontsize=18, fontweight='bold')
        
        # Left plot: Test function type distribution
        test_types = [data['test_function_type'] for data in all_data.values()]
        type_counts = {}
        for t in test_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
               startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Test Function Type Distribution', fontsize=16)
        
        # Right plot: Data points and computation time summary
        config_names = []
        data_points = []
        gammas = []
        
        for config_name, data in all_data.items():
            config_names.append(config_name.replace('config_', '').replace('_', ' '))
            data_points.append(len(data['delta']))
            gammas.append(data['gamma'])
        
        x_pos = range(len(config_names))
        bars = ax2.bar(x_pos, data_points, alpha=0.8, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(config_names)],
                      edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Configuration', fontsize=14)
        ax2.set_ylabel('Number of Data Points', fontsize=14)
        ax2.set_title('Data Points per Configuration', fontsize=16)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add text summary
        total_configs = len(all_data)
        total_points = sum(data_points)
        avg_points = total_points / total_configs if total_configs > 0 else 0
        
        summary_text = f"Total Configurations: {total_configs}\n"
        summary_text += f"Total Data Points: {total_points}\n"
        summary_text += f"Average Points/Config: {avg_points:.1f}\n"
        summary_text += f"Gamma Range: {min(gammas):.1f} - {max(gammas):.1f}"
        
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('results/experiment1_summary_5.png', dpi=300, bbox_inches='tight')
        print("✓ Image 5: Parameter Space saved")
        plt.close()
