# ############################################################################
#
# EXPERIMENT 2: TWO-ZERO INTERACTION - VISUALIZATION
# ===================================================
#
# This module creates visualizations for two-zero interaction analysis,
# including energy landscapes, interference patterns, and stability plots.
#
# Key visualizations:
# - Individual vs joint perturbation energy curves
# - Interference pattern plots
# - Cross-coupling coefficient visualization
# - Stability analysis summary plots
# - Comparative energy landscape plots
#
# ############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from sage.all import *

class Experiment2Viz:
    """Visualization module for two-zero interaction experiments."""
    
    def __init__(self, style='seaborn-v0_8', figsize_default=(12, 8), dpi=300):
        """
        Initialize visualization module.
        
        Args:
            style: Matplotlib style
            figsize_default: Default figure size
            dpi: Resolution for saved figures
        """
        self.style = style
        self.figsize_default = figsize_default
        self.dpi = dpi
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color scheme for consistency
        self.colors = {
            'zero1': '#1f77b4',  # Blue
            'zero2': '#ff7f0e',  # Orange  
            'joint': '#2ca02c',  # Green
            'interference': '#d62728',  # Red
            'fit': '#9467bd',    # Purple
            'critical': '#8c564b'  # Brown
        }
        
        print(f"Experiment 2 Visualization Module initialized")
        print(f"  Style: {style}, DPI: {dpi}")
    
    def plot_energy_perturbations(self, math_results, stats_results, save_path="experiment2_energy_perturbations.png"):
        """
        Create comprehensive energy perturbation plot.
        
        Args:
            math_results: Results from experiment2_math
            stats_results: Results from experiment2_stats
            save_path: Path to save the plot
            
        Returns:
            str: Path to saved plot
        """
        print("Creating energy perturbation visualization...")
        
        # Extract data
        individual = math_results['individual']
        joint = math_results['joint']
        
        delta_values = individual['delta_values']
        delta_E1_values = individual['delta_E1_values']
        delta_E2_values = individual['delta_E2_values']
        delta_E12_values = joint['delta_E12_values']
        interference_values = joint['interference_values']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 1.5], hspace=0.3, wspace=0.25)
        
        # Subplot 1: Individual energy perturbations
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(delta_values, delta_E1_values, 'o-', color=self.colors['zero1'], 
                label=f'Zero 1 (γ₁={math_results["gamma1"]:.2f})', markersize=4, linewidth=2)
        ax1.plot(delta_values, delta_E2_values, 's-', color=self.colors['zero2'], 
                label=f'Zero 2 (γ₂={math_results["gamma2"]:.2f})', markersize=4, linewidth=2)
        
        # Add quadratic fits
        delta_squared = delta_values**2
        C1 = stats_results['stability_zero1']['C1']
        C2 = stats_results['stability_zero2']['C1']
        fit1 = C1 * delta_squared
        fit2 = C2 * delta_squared
        
        ax1.plot(delta_values, fit1, '--', color=self.colors['zero1'], alpha=0.7, 
                label=f'Fit: C₁={C1:.2e}δ²')
        ax1.plot(delta_values, fit2, '--', color=self.colors['zero2'], alpha=0.7, 
                label=f'Fit: C₂={C2:.2e}δ²')
        
        ax1.set_xlabel('Perturbation δ')
        ax1.set_ylabel('Energy Change ΔE')
        ax1.set_title('Individual Zero Perturbations')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Subplot 2: Joint vs sum comparison
        ax2 = fig.add_subplot(gs[0, 1])
        expected_sum = delta_E1_values + delta_E2_values
        
        ax2.plot(delta_values, delta_E12_values, 'o-', color=self.colors['joint'], 
                label='Joint ΔE₁₂', markersize=4, linewidth=2)
        ax2.plot(delta_values, expected_sum, 's-', color=self.colors['fit'], 
                label='Sum ΔE₁ + ΔE₂', markersize=4, linewidth=2, alpha=0.7)
        
        # Add joint quadratic fit
        C12_total = stats_results['stability_joint']['C1']
        fit12 = C12_total * delta_squared
        ax2.plot(delta_values, fit12, '--', color=self.colors['joint'], alpha=0.7,
                label=f'Fit: C₁₂={C12_total:.2e}δ²')
        
        ax2.set_xlabel('Perturbation δ')
        ax2.set_ylabel('Energy Change ΔE')
        ax2.set_title('Joint Perturbation vs Sum')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Subplot 3: Interference pattern
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(delta_values, interference_values, 'o-', color=self.colors['interference'], 
                label='Interference: ΔE₁₂ - (ΔE₁ + ΔE₂)', markersize=4, linewidth=2)
        
        # Add cross-coupling fit if significant
        cross_coupling = stats_results['cross_coupling_analysis']
        if cross_coupling['is_significant']:
            C12_cross = cross_coupling['C12']
            fit_cross = C12_cross * delta_squared
            ax3.plot(delta_values, fit_cross, '--', color=self.colors['interference'], alpha=0.7,
                    label=f'Cross-coupling: C₁₂={C12_cross:.2e}δ²')
        
        ax3.set_xlabel('Perturbation δ')
        ax3.set_ylabel('Interference Energy')
        ax3.set_title('Two-Zero Interference Pattern')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Subplot 4: Stability summary
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Bar chart of coefficients
        coefficients = [
            stats_results['stability_zero1']['C1'],
            stats_results['stability_zero2']['C1'],
            stats_results['stability_joint']['C1']
        ]
        labels = ['C₁ (Zero 1)', 'C₂ (Zero 2)', 'C₁₂ (Joint)']
        colors = [self.colors['zero1'], self.colors['zero2'], self.colors['joint']]
        
        bars = ax4.bar(labels, coefficients, color=colors, alpha=0.7)
        ax4.set_ylabel('Stability Coefficient')
        ax4.set_title('Quadratic Stability Coefficients')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, coeff in zip(bars, coefficients):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{coeff:.2e}', ha='center', va='bottom', fontsize=8)
        
        # Subplot 5: Statistical summary
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # Create summary text
        gamma1 = math_results['gamma1']
        gamma2 = math_results['gamma2']
        overall_stable = stats_results['overall_stable']
        
        summary_text = f"""Statistical Summary:
        
Zero Heights: γ₁ = {gamma1:.2f}, γ₂ = {gamma2:.2f}
        
Stability Analysis:
• Zero 1: R² = {stats_results['stability_zero1']['r_squared']:.4f}
• Zero 2: R² = {stats_results['stability_zero2']['r_squared']:.4f}  
• Joint: R² = {stats_results['stability_joint']['r_squared']:.4f}

Overall Stability: {"STABLE" if overall_stable else "UNSTABLE"}

Interference:
• Max |interference|: {stats_results['interference_analysis']['max_interference']:.2e}
• Cross-coupling significant: {stats_results['cross_coupling_analysis']['is_significant']}"""
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Main title
        fig.suptitle(f'Experiment 2: Two-Zero Interaction Analysis\nγ₁={gamma1:.2f}, γ₂={gamma2:.2f}', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Energy perturbation plot saved to: '{save_path}'")
        return save_path
    
    def plot_interference_analysis(self, math_results, stats_results, save_path="experiment2_interference_analysis.png"):
        """
        Create detailed interference analysis plot.
        
        Args:
            math_results: Results from experiment2_math
            stats_results: Results from experiment2_stats  
            save_path: Path to save the plot
            
        Returns:
            str: Path to saved plot
        """
        print("Creating interference analysis visualization...")
        
        # Extract data
        individual = math_results['individual']
        joint = math_results['joint']
        interference_analysis = stats_results['interference_analysis']
        
        delta_values = individual['delta_values']
        interference_values = joint['interference_values']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Two-Zero Interference Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Interference vs perturbation
        ax1.plot(delta_values, interference_values, 'o-', color=self.colors['interference'], 
                markersize=4, linewidth=2, label='Interference')
        
        # Add power law fit if successful
        if interference_analysis['power_law_fit_success']:
            A = interference_analysis['power_law_amplitude']
            alpha = interference_analysis['power_law_exponent']
            delta_abs = np.abs(delta_values)
            power_fit = A * delta_abs**alpha
            
            # Apply sign
            power_fit_signed = np.where(delta_values >= 0, power_fit, -power_fit)
            ax1.plot(delta_values, power_fit_signed, '--', color=self.colors['fit'], 
                    alpha=0.7, label=f'Power law: {A:.2e}|δ|^{alpha:.2f}')
        
        ax1.set_xlabel('Perturbation δ')
        ax1.set_ylabel('Interference Energy')
        ax1.set_title('Interference vs Perturbation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Absolute interference vs |δ|
        delta_abs = np.abs(delta_values)
        interference_abs = np.abs(interference_values)
        
        ax2.loglog(delta_abs[delta_abs > 0], interference_abs[delta_abs > 0], 'o', 
                  color=self.colors['interference'], markersize=4, label='|Interference|')
        
        if interference_analysis['power_law_fit_success']:
            A = interference_analysis['power_law_amplitude']
            alpha = interference_analysis['power_law_exponent']
            delta_fit = delta_abs[delta_abs > 0]
            power_fit_abs = A * delta_fit**alpha
            ax2.loglog(delta_fit, power_fit_abs, '--', color=self.colors['fit'], 
                      alpha=0.7, label=f'|I| = {A:.2e}|δ|^{alpha:.2f}')
        
        ax2.set_xlabel('|δ|')
        ax2.set_ylabel('|Interference|')
        ax2.set_title('Log-Log: Interference Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Interference distribution histogram
        ax3.hist(interference_values, bins=15, color=self.colors['interference'], 
                alpha=0.7, edgecolor='black', linewidth=1)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Zero')
        
        mean_int = interference_analysis['mean_interference']
        ax3.axvline(x=mean_int, color='red', linestyle='-', alpha=0.7, 
                   label=f'Mean: {mean_int:.2e}')
        
        ax3.set_xlabel('Interference Energy')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Interference Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistical summary
        ax4.axis('off')
        
        # Create summary text
        stats_text = f"""Interference Statistics:

Mean: {interference_analysis['mean_interference']:.3e}
Std: {interference_analysis['std_interference']:.3e}
Max |I|: {interference_analysis['max_interference']:.3e}

Sign Changes: {interference_analysis['sign_changes']}
Correlation with |δ|: {interference_analysis['correlation_with_delta']:.4f}

Power Law Fit:
• Amplitude A: {interference_analysis['power_law_amplitude']:.3e}
• Exponent α: {interference_analysis['power_law_exponent']:.2f} ± {interference_analysis['power_law_exponent_std']:.2f}
• Success: {interference_analysis['power_law_fit_success']}

Bias Test (vs zero):
• t-statistic: {interference_analysis['bias_t_statistic']:.3f}
• p-value: {interference_analysis['bias_p_value']:.3e}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Interference analysis plot saved to: '{save_path}'")
        return save_path
    
    def plot_stability_comparison(self, stats_results, save_path="experiment2_stability_comparison.png"):
        """
        Create stability comparison visualization.
        
        Args:
            stats_results: Results from experiment2_stats
            save_path: Path to save the plot
            
        Returns:
            str: Path to saved plot
        """
        print("Creating stability comparison visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quadratic Stability Analysis Comparison', fontsize=16, fontweight='bold')
        
        # Extract stability data
        stab1 = stats_results['stability_zero1']
        stab2 = stats_results['stability_zero2']
        stab_joint = stats_results['stability_joint']
        
        # Plot 1: Stability coefficients
        labels = ['Zero 1', 'Zero 2', 'Joint']
        coeffs = [stab1['C1'], stab2['C1'], stab_joint['C1']]
        colors = [self.colors['zero1'], self.colors['zero2'], self.colors['joint']]
        
        bars = ax1.bar(labels, coeffs, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Stability Coefficient C₁')
        ax1.set_title('Quadratic Stability Coefficients')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, coeff in zip(bars, coeffs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{coeff:.2e}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: R-squared values
        r_squared = [stab1['r_squared'], stab2['r_squared'], stab_joint['r_squared']]
        
        bars = ax2.bar(labels, r_squared, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('R² (Goodness of Fit)')
        ax2.set_title('Quadratic Model Fit Quality')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, r2 in zip(bars, r_squared):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{r2:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: P-values (log scale)
        p_values = [stab1['p_value'], stab2['p_value'], stab_joint['p_value']]
        
        bars = ax3.bar(labels, [-np.log10(max(p, 1e-16)) for p in p_values], 
                      color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('-log₁₀(p-value)')
        ax3.set_title('Statistical Significance')
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, 
                   label='p = 0.05 threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary table
        ax4.axis('off')
        
        # Create comparison table
        table_data = [
            ['Metric', 'Zero 1', 'Zero 2', 'Joint'],
            ['C₁ coefficient', f'{stab1["C1"]:.2e}', f'{stab2["C1"]:.2e}', f'{stab_joint["C1"]:.2e}'],
            ['R²', f'{stab1["r_squared"]:.4f}', f'{stab2["r_squared"]:.4f}', f'{stab_joint["r_squared"]:.4f}'],
            ['p-value', f'{stab1["p_value"]:.2e}', f'{stab2["p_value"]:.2e}', f'{stab_joint["p_value"]:.2e}'],
            ['RMSE', f'{stab1["rmse"]:.2e}', f'{stab2["rmse"]:.2e}', f'{stab_joint["rmse"]:.2e}'],
            ['Stable?', str(stab1["is_stable"]), str(stab2["is_stable"]), str(stab_joint["is_stable"])]
        ]
        
        # Create table
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code stability results
        for i in range(1, 4):
            stable = table_data[5][i] == 'True'
            color = 'lightgreen' if stable else 'lightcoral'
            table[(5, i)].set_facecolor(color)
        
        ax4.set_title('Stability Analysis Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Stability comparison plot saved to: '{save_path}'")
        return save_path
    
    def create_all_visualizations(self, math_results, stats_results, output_dir="experiment2"):
        """
        Create all visualization plots for the experiment.
        
        Args:
            math_results: Results from experiment2_math
            stats_results: Results from experiment2_stats
            output_dir: Directory to save plots
            
        Returns:
            list: List of saved plot paths
        """
        print("Creating all visualizations for Experiment 2...")
        
        start_time = time.time()
        saved_plots = []
        
        # Ensure output directory exists
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create all plots
        plot1 = self.plot_energy_perturbations(
            math_results, stats_results, 
            f"{output_dir}/experiment2_energy_perturbations.png"
        )
        saved_plots.append(plot1)
        
        plot2 = self.plot_interference_analysis(
            math_results, stats_results,
            f"{output_dir}/experiment2_interference_analysis.png"
        )
        saved_plots.append(plot2)
        
        plot3 = self.plot_stability_comparison(
            stats_results,
            f"{output_dir}/experiment2_stability_comparison.png"
        )
        saved_plots.append(plot3)
        
        total_time = time.time() - start_time
        
        print(f"✓ All visualizations completed in {total_time:.2f} seconds")
        print(f"✓ Saved {len(saved_plots)} plots to '{output_dir}/'")
        
        return saved_plots

# Factory function
def create_experiment2_viz(style='seaborn-v0_8', figsize_default=(12, 8), dpi=300):
    """Create Experiment2Viz instance."""
    return Experiment2Viz(style=style, figsize_default=figsize_default, dpi=dpi)
