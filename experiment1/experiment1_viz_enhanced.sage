# ############################################################################
#
# EXPERIMENT 1: SINGLE-ZERO PERTURBATION - ENHANCED VISUALIZATION MODULE
# ========================================================================
#
# This module provides comprehensive plotting and visualization for Experiment 1
# results including:
# - ΔE vs δ plots with polynomial fits and confidence bands
# - ΔE vs δ² plots to verify quadratic behavior
# - Residual plots for model validation
# - Gradient and curvature analysis plots
# - Bootstrap distribution visualizations
# - Publication-quality figures with error bars and p-values
# - HDF5 data integration with modern naming conventions
# - Max 2 panels per figure as per Design Guide
#
# ############################################################################

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import time
import os

class Experiment1Visualization:
    """Enhanced visualization module for single-zero perturbation experiment with HDF5 support."""
    
    def __init__(self, hdf5_file, output_dir="results", figsize=(12, 8), dpi=300):
        """
        Initialize visualization module.
        
        Args:
            hdf5_file: Path to HDF5 data file
            output_dir: Directory for output files
            figsize: Figure size for plots
            dpi: Resolution for saved figures
        """
        self.hdf5_file = hdf5_file
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        print(f"Experiment 1 Visualization initialized:")
        print(f"  HDF5 file: {hdf5_file}")
        print(f"  Output directory: {output_dir}")
    
    def plot_energy_vs_delta(self, delta_values, delta_E_values, fitting_results=None, 
                           stats_results=None, ax=None, title_suffix=""):
        """
        Plot ΔE vs δ with polynomial fits and confidence bands.
        
        Args:
            delta_values: Array of δ perturbation values
            delta_E_values: Array of ΔE values
            fitting_results: Polynomial fitting results
            stats_results: Statistical analysis results
            ax: Matplotlib axis (optional)
            title_suffix: Additional text for title
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
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
        ax.set_title(f'Energy vs Displacement{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_energy_vs_delta_squared(self, delta_values, delta_E_values, 
                                   fitting_results=None, ax=None, title_suffix=""):
        """
        Plot ΔE vs δ² to verify quadratic behavior.
        
        Args:
            delta_values: Array of δ values
            delta_E_values: Array of ΔE values
            fitting_results: Polynomial fitting results
            ax: Matplotlib axis (optional)
            title_suffix: Additional text for title
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
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
        ax.set_title(f'Quadratic Behavior Verification{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_residual_analysis(self, fitting_results, delta_values, ax=None, title_suffix=""):
        """
        Plot residual analysis for model validation.
        
        Args:
            fitting_results: Polynomial fitting results
            delta_values: Original δ values
            ax: Matplotlib axis (optional)
            title_suffix: Additional text for title
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
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
        ax.set_title(f'Residual Analysis{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_bootstrap_distributions(self, bootstrap_stats, ax=None, title_suffix=""):
        """
        Plot bootstrap distributions of coefficients.
        
        Args:
            bootstrap_stats: Bootstrap analysis results
            ax: Matplotlib axis (optional)
            title_suffix: Additional text for title
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
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
        ax.set_title(f'Bootstrap Distribution: C₁ Coefficient{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_hypothesis_testing_summary(self, hypothesis_tests, ax=None, title_suffix=""):
        """
        Create a visual summary of hypothesis testing results.
        
        Args:
            hypothesis_tests: Hypothesis testing results
            ax: Matplotlib axis (optional)
            title_suffix: Additional text for title
            
        Returns:
            matplotlib axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.axis('off')
        
        # Create text summary
        y_pos = 0.9
        line_height = 0.08
        
        ax.text(0.05, y_pos, f'HYPOTHESIS TESTING SUMMARY{title_suffix}', 
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
    
    def generate_energy_behavior_plot(self, config_name):
        """Generate exp1_energy_behavior.png for a configuration."""
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                if config_name not in f:
                    print(f"Configuration {config_name} not found")
                    return None
                
                config_group = f[config_name]
                delta_values = config_group['delta_values'][:]
                delta_E_values = config_group['delta_E_values'][:]
                
                # Load fitting results if available
                fitting_results = None
                stats_results = None
                if 'statistics' in config_group:
                    stats_group = config_group['statistics']
                    fitting_results = self._load_fitting_results_from_hdf5(stats_group)
                    stats_results = self._load_stats_results_from_hdf5(stats_group)
                
                # Create 2-panel figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
                fig.suptitle(f'Energy Behavior Analysis: {config_name}', fontsize=14, fontweight='bold')
                
                # Panel 1: Energy vs displacement
                self.plot_energy_vs_delta(delta_values, delta_E_values, fitting_results, 
                                         stats_results, ax1)
                
                # Panel 2: Quadratic behavior verification
                self.plot_energy_vs_delta_squared(delta_values, delta_E_values, 
                                                 fitting_results, ax2)
                
                plt.tight_layout()
                filename = os.path.join(self.output_dir, f"exp1_energy_behavior_{config_name}.png")
                plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                print(f"📊 Energy behavior plot saved: {filename}")
                return filename
                
        except Exception as e:
            print(f"Error generating energy behavior plot for {config_name}: {e}")
            return None
    
    def generate_statistical_analysis_plot(self, config_name):
        """Generate exp1_statistical_analysis.png for a configuration."""
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                if config_name not in f:
                    print(f"Configuration {config_name} not found")
                    return None
                
                config_group = f[config_name]
                delta_values = config_group['delta_values'][:]
                
                if 'statistics' not in config_group:
                    print(f"No statistics available for {config_name}")
                    return None
                
                stats_group = config_group['statistics']
                fitting_results = self._load_fitting_results_from_hdf5(stats_group)
                stats_results = self._load_stats_results_from_hdf5(stats_group)
                
                # Create 2-panel figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
                fig.suptitle(f'Statistical Analysis: {config_name}', fontsize=14, fontweight='bold')
                
                # Panel 1: Residual analysis
                self.plot_residual_analysis(fitting_results, delta_values, ax1)
                
                # Panel 2: Bootstrap distributions
                bootstrap_stats = stats_results.get('bootstrap_analysis')
                self.plot_bootstrap_distributions(bootstrap_stats, ax2)
                
                plt.tight_layout()
                filename = os.path.join(self.output_dir, f"exp1_statistical_analysis_{config_name}.png")
                plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                print(f"📊 Statistical analysis plot saved: {filename}")
                return filename
                
        except Exception as e:
            print(f"Error generating statistical analysis plot for {config_name}: {e}")
            return None
    
    def generate_hypothesis_testing_plot(self, config_name):
        """Generate exp1_hypothesis_testing.png for a configuration."""
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                if config_name not in f:
                    print(f"Configuration {config_name} not found")
                    return None
                
                config_group = f[config_name]
                
                if 'statistics' not in config_group:
                    print(f"No statistics available for {config_name}")
                    return None
                
                stats_group = config_group['statistics']
                stats_results = self._load_stats_results_from_hdf5(stats_group)
                
                # Create single-panel figure for hypothesis testing summary
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                hypothesis_tests = stats_results.get('hypothesis_testing', {})
                self.plot_hypothesis_testing_summary(hypothesis_tests, ax, f" - {config_name}")
                
                plt.tight_layout()
                filename = os.path.join(self.output_dir, f"exp1_hypothesis_testing_{config_name}.png")
                plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                print(f"📊 Hypothesis testing plot saved: {filename}")
                return filename
                
        except Exception as e:
            print(f"Error generating hypothesis testing plot for {config_name}: {e}")
            return None
    
    def generate_cross_configuration_analysis(self):
        """Generate cross-configuration comparison plots."""
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                config_names = [name for name in f.keys() if 'statistics' in f[name]]
                
                if len(config_names) == 0:
                    print("No configurations with statistics found")
                    return []
                
                # Extract key metrics for all configurations
                C1_values = []
                C2_values = []
                R2_values = []
                stability_pvalues = []
                config_labels = []
                
                for config_name in config_names:
                    stats_group = f[config_name]['statistics']
                    
                    if 'fitting_results' in stats_group and 'cubic' in stats_group['fitting_results']:
                        cubic_group = stats_group['fitting_results']['cubic']
                        C1_values.append(float(cubic_group['C1'][()]))
                        C2_values.append(float(cubic_group['C2'][()]))
                        R2_values.append(float(cubic_group['r_squared'][()]))
                        config_labels.append(config_name)
                        
                        # Get stability p-value
                        if 'hypothesis_testing' in stats_group and 'local_stability' in stats_group['hypothesis_testing']:
                            stability_group = stats_group['hypothesis_testing']['local_stability']
                            stability_pvalues.append(float(stability_group['p_value'][()]))
                        else:
                            stability_pvalues.append(1.0)  # No test = inconclusive
                
                if len(config_labels) == 0:
                    print("No valid configurations found for cross-analysis")
                    return []
                
                # Create cross-configuration comparison plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
                fig.suptitle('Cross-Configuration Analysis', fontsize=14, fontweight='bold')
                
                # Panel 1: C1 vs C2 coefficients with stability coloring
                colors = ['green' if p < 0.05 else 'red' for p in stability_pvalues]
                scatter = ax1.scatter(np.array(C1_values)*1e6, np.array(C2_values)*1e9, 
                                    c=colors, s=100, alpha=0.7, edgecolors='black')
                ax1.set_xlabel('C₁ coefficient (×10⁻⁶)')
                ax1.set_ylabel('C₂ coefficient (×10⁻⁹)')
                ax1.set_title('Polynomial Coefficients by Configuration')
                ax1.grid(True, alpha=0.3)
                ax1.axvline(0, color='red', linestyle='--', alpha=0.5, label='Stability threshold')
                
                # Add legend for stability
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='green', label='Stable (p < 0.05)'),
                                 Patch(facecolor='red', label='Unstable (p ≥ 0.05)')]
                ax1.legend(handles=legend_elements, loc='best')
                
                # Panel 2: R² vs C1 with configuration labels
                ax2.scatter(np.array(C1_values)*1e6, R2_values, c=colors, s=100, alpha=0.7, edgecolors='black')
                ax2.set_xlabel('C₁ coefficient (×10⁻⁶)')
                ax2.set_ylabel('R² (goodness of fit)')
                ax2.set_title('Model Fit Quality vs Stability')
                ax2.grid(True, alpha=0.3)
                ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
                
                # Add configuration labels if not too many
                if len(config_labels) <= 10:
                    for i, label in enumerate(config_labels):
                        ax2.annotate(label, (C1_values[i]*1e6, R2_values[i]), 
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=8, alpha=0.8)
                
                plt.tight_layout()
                filename = os.path.join(self.output_dir, "exp1_cross_configuration_analysis.png")
                plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                print(f"📊 Cross-configuration analysis plot saved: {filename}")
                
                # Generate summary table plot
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.axis('off')
                
                # Create summary table
                table_data = []
                headers = ['Configuration', 'C₁ (×10⁻⁶)', 'C₂ (×10⁻⁹)', 'R²', 'Stability p-value', 'Status']
                
                for i, config_name in enumerate(config_labels):
                    status = 'Stable' if stability_pvalues[i] < 0.05 else 'Unstable'
                    table_data.append([
                        config_name,
                        f"{C1_values[i]*1e6:.3f}",
                        f"{C2_values[i]*1e9:.3f}",
                        f"{R2_values[i]:.6f}",
                        f"{stability_pvalues[i]:.2e}",
                        status
                    ])
                
                table = ax.table(cellText=table_data, colLabels=headers, 
                               cellLoc='center', loc='center', fontsize=10)
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                
                # Color code rows by stability
                for i, p_val in enumerate(stability_pvalues):
                    color = 'lightgreen' if p_val < 0.05 else 'lightcoral'
                    for j in range(len(headers)):
                        table[(i+1, j)].set_facecolor(color)
                
                ax.set_title('Configuration Summary Table', fontsize=16, fontweight='bold', pad=20)
                
                plt.tight_layout()
                filename_table = os.path.join(self.output_dir, "exp1_configuration_summary_table.png")
                plt.savefig(filename_table, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                print(f"📊 Configuration summary table saved: {filename_table}")
                
                return [filename, filename_table]
                
        except Exception as e:
            print(f"Error generating cross-configuration analysis: {e}")
            return []
    
    def _load_fitting_results_from_hdf5(self, stats_group):
        """Load fitting results from HDF5 statistics group."""
        fitting_results = {}
        
        if 'fitting_results' not in stats_group:
            return fitting_results
        
        fitting_group = stats_group['fitting_results']
        
        for model_name in ['quadratic', 'cubic', 'quartic']:
            if model_name in fitting_group:
                model_group = fitting_group[model_name]
                fitting_results[model_name] = {}
                
                for key in model_group.keys():
                    fitting_results[model_name][key] = model_group[key][...]
        
        return fitting_results
    
    def _load_stats_results_from_hdf5(self, stats_group):
        """Load comprehensive statistics from HDF5 group."""
        stats_results = {}
        
        # Load hypothesis testing
        if 'hypothesis_testing' in stats_group:
            stats_results['hypothesis_testing'] = {}
            hyp_group = stats_group['hypothesis_testing']
            
            for test_name in hyp_group.keys():
                test_group = hyp_group[test_name]
                stats_results['hypothesis_testing'][test_name] = {}
                
                for key in test_group.keys():
                    value = test_group[key][...]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    stats_results['hypothesis_testing'][test_name][key] = value
        
        # Load bootstrap analysis
        if 'bootstrap_analysis' in stats_group:
            stats_results['bootstrap_analysis'] = {}
            bootstrap_group = stats_group['bootstrap_analysis']
            
            for param_name in ['C1', 'C2', 'model_type', 'successful_samples']:
                if param_name in bootstrap_group:
                    if param_name in ['model_type']:
                        stats_results['bootstrap_analysis'][param_name] = bootstrap_group[param_name][...]
                    elif param_name in ['successful_samples']:
                        stats_results['bootstrap_analysis'][param_name] = int(bootstrap_group[param_name][()])
                    else:
                        # Parameter statistics
                        param_group = bootstrap_group[param_name]
                        stats_results['bootstrap_analysis'][param_name] = {}
                        
                        for key in param_group.keys():
                            stats_results['bootstrap_analysis'][param_name][key] = param_group[key][...]
        
        return stats_results
    
    def generate_all_visualizations(self):
        """Generate all visualizations for all configurations."""
        try:
            with h5py.File(self.hdf5_file, 'r') as f:
                config_names = list(f.keys())
            
            print(f"Generating visualizations for {len(config_names)} configurations...")
            
            generated_files = []
            
            # Generate per-configuration plots
            for config_name in config_names:
                print(f"\nGenerating plots for {config_name}...")
                
                # Energy behavior plot
                filename = self.generate_energy_behavior_plot(config_name)
                if filename:
                    generated_files.append(filename)
                
                # Statistical analysis plot
                filename = self.generate_statistical_analysis_plot(config_name)
                if filename:
                    generated_files.append(filename)
                
                # Hypothesis testing plot
                filename = self.generate_hypothesis_testing_plot(config_name)
                if filename:
                    generated_files.append(filename)
            
            # Generate cross-configuration analysis
            print(f"\nGenerating cross-configuration analysis...")
            cross_files = self.generate_cross_configuration_analysis()
            generated_files.extend(cross_files)
            
            print(f"\n✅ Generated {len(generated_files)} visualization files:")
            for filename in generated_files:
                print(f"  📊 {filename}")
            
            return generated_files
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            return []
