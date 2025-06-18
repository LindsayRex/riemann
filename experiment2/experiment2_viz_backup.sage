#!/usr/bin/env sage

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class Experiment2Viz:
    def __init__(self, hdf5_file="experiment2_two_zero_interaction.h5", batch_mode=False):
        self.hdf5_file = hdf5_file
        self.batch_mode = batch_mode

    def load_metadata(self, f):
        meta = f['metadata']
        return {
            'gamma1': meta.attrs['gamma_1'],
            'gamma2': meta.attrs['gamma_2'],
            'description': meta.attrs['description'],
            'basis': meta.attrs['test_function_basis']
        }
        filename4 = "results/experiment2_summary_4_cross_coupling.png"ivity Testing ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle(f'Image 4/5: Cross-Coupling Analysis - Quantitative Additivity Assessment', fontsize=16, fontweight='bold', y=0.92)t(0.02, 0.08, context_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.18, 1, 0.88])
        filename3 = "results/experiment2_summary_3_interference.png"tivity ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle(f'Image 3/5: Interference Analysis - Zero-Zero Interaction Effects', fontsize=16, fontweight='bold', y=0.92)t(0.02, 0.08, context_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.18, 1, 0.88])
        filename2 = "results/experiment2_summary_2_fit_quality.png"ical Robustness ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle(f'Image 2/5: Statistical Robustness - Fit Quality and Model Validation', fontsize=16, fontweight='bold', y=0.92)t(0.02, 0.08, context_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.18, 1, 0.88])
        filename1 = "results/experiment2_summary_1_stability.png"ig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.suptitle(f'Image 1/5: Stability Analysis - C₁ Coefficients and Bootstrap Confidence Intervals', fontsize=16, fontweight='bold', y=0.92)         'gamma1': meta.attrs['gamma_1'],
            'gamma2': meta.attrs['gamma_2'],
            'description': meta.attrs['description'],
            'basis': meta.attrs['test_function_basis']
        }

    def create_summary_plot(self, f, metadata):
        """Create single summary plot for this configuration"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Two-Zero Interaction: γ₁={metadata["gamma1"]:.2f}, γ₂={metadata["gamma2"]:.2f}', fontsize=12)
        
        # Energy comparison plot
        schemes = ['scheme_i', 'scheme_ii', 'scheme_both']
        colors = ['blue', 'green', 'red']
        labels = ['γ₁ only', 'γ₂ only', 'Both']
        
        for scheme, color, label in zip(schemes, colors, labels):
            data = f[scheme]
            delta = data['delta'][:]
            delta_E = data['delta_E'][:]
            axes[0].plot(delta, delta_E, 'o-', color=color, label=label, markersize=3)
        
        axes[0].set_title('Energy Changes')
        axes[0].set_xlabel('δ')
        axes[0].set_ylabel('ΔE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Interference plot
        interference = f['interference_analysis']
        delta = interference['delta'][:]
        ratio = interference['interference_ratio'][:]
        axes[1].plot(delta, ratio, 'mo-', label='Interference', markersize=3)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_title('Interference Ratio')
        axes[1].set_xlabel('δ')
        axes[1].set_ylabel('I(δ)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.batch_mode:
            return fig  # Return figure for batch processing
        else:
            # Save individual plot
            filename = f"experiment2_gamma1_{metadata['gamma1']:.2f}_gamma2_{metadata['gamma2']:.2f}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            return filename
        
        # Scheme Both
        scheme_both = f['scheme_both']
        delta = scheme_both['delta'][:]
        delta_E = scheme_both['delta_E'][:]
        coeffs = scheme_both['polyfit_coeffs'][:]
        r_squared = scheme_both.attrs['r_squared']
        
        axes[1,0].plot(delta, delta_E, 'mo-', label='Data')
        fit_curve = coeffs[0] * delta + coeffs[1] * delta**2
        axes[1,0].plot(delta, fit_curve, 'r-', label=f'Fit: R²={r_squared:.3f}')
        axes[1,0].set_title('Scheme Both: Shift both zeros')
        axes[1,0].set_xlabel('δ')
        axes[1,0].set_ylabel('ΔE')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Interference analysis
        interference = f['interference_analysis']
        delta = interference['delta'][:]
        int_ratio = interference['interference_ratio'][:]
        
        axes[1,1].plot(delta, int_ratio, 'co-', label='Interference ratio')
        axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1,1].set_title('Interference Analysis')
        axes[1,1].set_xlabel('δ')
        axes[1,1].set_ylabel('I(δ)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        return fig

    def plot_stability_summary(self, f, metadata):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        schemes = ['scheme_i', 'scheme_ii', 'scheme_both']
        labels = ['Scheme I (γ₁)', 'Scheme II (γ₂)', 'Scheme Both']
        colors = ['blue', 'green', 'magenta']
        
        c1_values = []
        c2_values = []
        stabilities = []
        
        for scheme_name in schemes:
            scheme = f[scheme_name]
            coeffs = scheme['polyfit_coeffs'][:]
            c1_values.append(coeffs[0])
            c2_values.append(-coeffs[1])  # Convert back from stored -C2
            stabilities.append(scheme.attrs['stability'])
        
        x_pos = np.arange(len(schemes))
        
        bars1 = ax.bar(x_pos - 0.2, c1_values, 0.4, label='C₁ coefficient', color=colors, alpha=0.7)
        bars2 = ax.bar(x_pos + 0.2, c2_values, 0.4, label='C₂ coefficient', color=colors, alpha=0.3)
        
        ax.set_xlabel('Perturbation Scheme')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'Stability Coefficients\nγ₁={metadata["gamma1"]:.2f}, γ₂={metadata["gamma2"]:.2f}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Add stability annotations
        for i, stability in enumerate(stabilities):
            color = 'green' if stability == 'stable' else 'red'
            ax.text(i, max(max(c1_values), max(c2_values)) * 0.8, stability.upper(), 
                   ha='center', va='bottom', color=color, fontweight='bold')
        
        plt.tight_layout()
        return fig

    def plot_energy_curves(self, f, metadata):
        """Create ΔE(δ) vs δ plots showing parabolic fits"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Energy Curves: γ₁={metadata["gamma1"]:.2f}, γ₂={metadata["gamma2"]:.2f}', fontsize=14)
        
        schemes = ['scheme_i', 'scheme_ii', 'scheme_both']
        titles = ['Scheme I: γ₁ perturbation only', 'Scheme II: γ₂ perturbation only', 'Scheme Both: Both zeros perturbed']
        colors = ['blue', 'green', 'red']
        
        for i, (scheme, title, color) in enumerate(zip(schemes, titles, colors)):
            row, col = i // 2, i % 2
            data = f[scheme]
            delta = data['delta'][:]
            delta_E = data['delta_E'][:]
            coeffs = data['polyfit_coeffs'][:]
            r_squared = data.attrs['r_squared']
            c1_p_value = data.attrs['c1_p_value']
            
            # Plot data points
            axes[row, col].plot(delta, delta_E, 'o', color=color, markersize=4, alpha=0.7, label='Data')
            
            # Plot parabolic fit
            delta_fine = np.linspace(delta[0], delta[-1], 100)
            fit_curve = coeffs[0] * delta_fine + coeffs[1] * delta_fine**2
            axes[row, col].plot(delta_fine, fit_curve, '-', color=color, linewidth=2, 
                              label=f'C₁={coeffs[0]:.2f}, C₂={coeffs[1]:.2f}')
            
            axes[row, col].set_title(f'{title}\nR²={r_squared:.4f}, p={c1_p_value:.2e}')
            axes[row, col].set_xlabel('δ')
            axes[row, col].set_ylabel('ΔE')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[row, col].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Interference plot in the remaining subplot
        interference = f['interference_analysis']
        delta = interference['delta'][:]
        ratio = interference['interference_ratio'][:]
        p_values = interference['p_values'][:]
        
        axes[1, 1].plot(delta, ratio, 'mo-', markersize=4, linewidth=2, label='I(δ) ratio')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Additive (I=0)')
        axes[1, 1].fill_between(delta, -0.01, 0.01, alpha=0.2, color='green', label='±1% tolerance')
        
        # Highlight significant deviations
        significant = np.abs(ratio) > 0.01
        if np.any(significant):
            axes[1, 1].scatter(delta[significant], ratio[significant], 
                             s=50, c='red', marker='x', label='|I(δ)| > 1%')
        
        max_interference = np.max(np.abs(ratio))
        mean_p_value = np.mean(p_values)
        axes[1, 1].set_title(f'Interference Analysis\nMax |I(δ)|={max_interference:.3f}, p̄={mean_p_value:.3f}')
        axes[1, 1].set_xlabel('δ')
        axes[1, 1].set_ylabel('I(δ)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_additivity_analysis(self, f, metadata):
        """Create detailed additivity analysis plot"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Additivity Analysis: γ₁={metadata["gamma1"]:.2f}, γ₂={metadata["gamma2"]:.2f}', fontsize=14)
        
        # Extract coefficients and confidence intervals
        schemes = ['scheme_i', 'scheme_ii', 'scheme_both']
        c1_values = []
        c1_cis = []
        
        for scheme in schemes:
            data = f[scheme]
            coeffs = data['polyfit_coeffs'][:]
            ci = data['bootstrap_CI'][:]
            c1_values.append(coeffs[0])
            c1_cis.append(ci[0])  # CI for C1
        
        # Plot 1: C₁ coefficients with confidence intervals
        x_pos = np.arange(len(schemes))
        labels = ['C₁(γ₁)', 'C₁(γ₂)', 'C₁⁽¹²⁾']
        colors = ['blue', 'green', 'red']
        
        for i, (c1, ci, label, color) in enumerate(zip(c1_values, c1_cis, labels, colors)):
            axes[0].errorbar(i, c1, yerr=[[c1-ci[0]], [ci[1]-c1]], 
                           fmt='o', color=color, markersize=8, capsize=5, label=label)
        
        # Show additivity line
        additive_sum = c1_values[0] + c1_values[1]
        axes[0].axhline(y=additive_sum, color='orange', linestyle='--', 
                       label=f'Additive: {additive_sum:.2f}')
        
        axes[0].set_title('C₁ Coefficients')
        axes[0].set_ylabel('C₁ value')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Additivity test
        interference_analysis = f['interference_analysis']
        diff = interference_analysis.attrs['additivity_difference']
        se_diff = interference_analysis.attrs['additivity_se']
        p_val = interference_analysis.attrs['additivity_p_value']
        
        axes[1].bar(['Difference'], [diff], yerr=[se_diff], 
                   color='purple', alpha=0.7, capsize=5)
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        axes[1].set_title(f'Additivity Test\np = {p_val:.2e}')
        axes[1].set_ylabel('C₁⁽¹²⁾ - (C₁(γ₁) + C₁(γ₂))')
        axes[1].grid(True, alpha=0.3)
        
        # Add significance annotation
        significance = 'Significant' if p_val < 0.05 else 'Not significant'
        color = 'red' if p_val < 0.05 else 'green'
        axes[1].text(0, diff + se_diff, significance, ha='center', va='bottom', 
                    color=color, fontweight='bold')
        
        # Plot 3: Cross-coupling coefficient
        cross_coupling = interference_analysis.attrs['cross_coupling_coeff']
        axes[2].bar(['C₁₂'], [cross_coupling], color='orange', alpha=0.7)
        axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        axes[2].set_title('Cross-Coupling Coefficient')
        axes[2].set_ylabel('C₁₂')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def create_plots(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            metadata = self.load_metadata(f)
            base_name = self.hdf5_file.replace('.h5', '')
            
            # 1. Energy curves plot (ΔE vs δ with parabolic fits)
            fig1 = self.plot_energy_curves(f, metadata)
            plot1_name = f"{base_name}_energy_curves.png"
            fig1.savefig(plot1_name, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # 2. Additivity analysis plot
            fig2 = self.plot_additivity_analysis(f, metadata)
            plot2_name = f"{base_name}_additivity.png"
            fig2.savefig(plot2_name, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            return [plot1_name, plot2_name]

def run_experiment2_viz(hdf5_file="experiment2_complete_analysis.h5"):
    """Generate focused summary visualizations for complete dataset"""
    with h5py.File(hdf5_file, 'r') as f:
        print(f"Creating complete dataset visualizations from {hdf5_file}")
        
        # Find all configuration groups
        config_groups = [key for key in f.keys() if key.startswith('config_')]
        print(f"Found {len(config_groups)} configurations")
        
        # Create 5 focused summary images
        summary_files = create_focused_dataset_summaries(hdf5_file)
        
        print(f"✓ Complete dataset visualizations created:")
        for i, summary_file in enumerate(summary_files, 1):
            print(f"  Image {i}: {summary_file}")
        print(f"  Analyzed {len(config_groups)} configurations")
        
        return summary_files

def create_focused_dataset_summaries(hdf5_file):
    """Create 5 focused statistical summary images for complete dataset"""
    with h5py.File(hdf5_file, 'r') as f:
        config_groups = [key for key in f.keys() if key.startswith('config_')]
        
        # Collect statistical data from all configurations
        gamma_pairs = []
        c1_values = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
        c1_errors = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
        c1_cis = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
        r_squared = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
        p_values = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
        interference_stats = []
        cross_coupling_coeffs = []
        additivity_results = []
        
        for config_name in config_groups:
            config_group = f[config_name]
            meta = config_group['metadata']
            gamma1, gamma2 = float(meta.attrs['gamma_1']), float(meta.attrs['gamma_2'])
            gamma_pairs.append((gamma1, gamma2))
            
            for scheme in ['scheme_i', 'scheme_ii', 'scheme_both']:
                data = config_group[scheme]
                coeffs = data['polyfit_coeffs'][:]
                bootstrap_ci = data['bootstrap_CI'][:]
                c1_values[scheme].append(float(coeffs[0]))
                c1_errors[scheme].append(float(data.attrs.get('c1_std_error', 0)))
                c1_cis[scheme].append(bootstrap_ci[0])  # C1 confidence interval
                r_squared[scheme].append(float(data.attrs.get('r_squared', 0)))
                p_values[scheme].append(float(data.attrs.get('c1_p_value', 1)))
            
            # Interference analysis data
            interference = config_group['interference_analysis']
            max_interference = float(np.max(np.abs(interference['interference_ratio'][:])))
            mean_interference = float(np.mean(np.abs(interference['interference_ratio'][:])))
            interference_p_value = float(interference.attrs.get('interference_p_value', 1))
            interference_stats.append((max_interference, mean_interference, interference_p_value))
            
            # Cross-coupling and additivity
            cross_coupling = float(config_group['interference_analysis'].attrs.get('cross_coupling_coeff', 0))
            cross_coupling_coeffs.append(cross_coupling)
            
            additivity_diff = float(config_group['interference_analysis'].attrs.get('additivity_difference', 0))
            additivity_p = float(config_group['interference_analysis'].attrs.get('additivity_p_value', 1))
            additivity_se = float(config_group['interference_analysis'].attrs.get('additivity_se', 0))
            additivity_results.append((additivity_diff, additivity_p, additivity_se))

        # Calculate summary statistics for context
        x = range(len(gamma_pairs))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional color scheme
        labels = ['γ₁ only', 'γ₂ only', 'Both zeros']
        
        stable_count = sum(1 for scheme in ['scheme_i', 'scheme_ii', 'scheme_both'] 
                          for c1 in c1_values[scheme] if c1 > 0)
        total_count = len(config_groups) * 3
        significant_interference = sum(1 for stats in interference_stats if stats[2] < 0.05)
        significant_additivity = sum(1 for res in additivity_results if res[1] < 0.05)
        mean_max_interference = float(np.mean([stats[0] for stats in interference_stats]))
        mean_cross_coupling = float(np.mean(np.abs(cross_coupling_coeffs)))
        
        # Create 5 focused images
        summary_files = []
        
        # === IMAGE 1: Stability Analysis ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Image 1/5: Stability Analysis - C₁ Coefficients and Bootstrap Confidence Intervals', fontsize=16, fontweight='bold', y=0.92)
        
        # Plot 1: C₁ coefficients with bootstrap confidence intervals
        for i, (scheme, color, label) in enumerate(zip(['scheme_i', 'scheme_ii', 'scheme_both'], colors, labels)):
            c1_vals = np.array(c1_values[scheme])
            c1_cis_array = np.array(c1_cis[scheme])
            ci_lower = c1_cis_array[:, 0]
            ci_upper = c1_cis_array[:, 1]
            
            # Error bars using bootstrap CIs
            yerr_lower = c1_vals - ci_lower
            yerr_upper = ci_upper - c1_vals
            
            ax1.errorbar([xi + i*0.25 for xi in x], c1_vals, 
                        yerr=[yerr_lower, yerr_upper], fmt='o', color=color, 
                        label=label, markersize=6, capsize=4, capthick=2, alpha=0.8, linewidth=2)
        
        ax1.set_title('C₁ Coefficients with Bootstrap 95% CIs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Configuration Index', fontsize=12)
        ax1.set_ylabel('C₁ Value', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.6, linewidth=2)
        
        # Plot 2: P-values with significance testing  
        for i, (scheme, color, label) in enumerate(zip(['scheme_i', 'scheme_ii', 'scheme_both'], colors, labels)):
            p_vals_safe = [max(p, 1e-16) for p in p_values[scheme]]
            ax2.scatter([xi + i*0.25 for xi in x], p_vals_safe, 
                       c=color, label=label, s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax2.set_title('P-values for C₁ > 0 (Log Scale)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Configuration Index', fontsize=12)
        ax2.set_ylabel('P-value', fontsize=12)
        ax2.set_yscale('log')
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, linewidth=3, label='α=0.05')
        ax2.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.8, linewidth=3, label='α=0.01')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Context summary
        context_text = f"""STABILITY ANALYSIS CONTEXT:
• Dataset: {len(config_groups)} zero-pair configurations analyzed
• Overall stability: {stable_count}/{total_count} ({float(stable_count/total_count*100):.1f}%) coefficients show C₁ > 0
• Statistical significance: Bootstrap 95% confidence intervals and hypothesis tests for C₁ > 0
• Interpretation: Positive C₁ indicates local energy minimum (stable zero configuration)"""
        
        fig.text(0.02, 0.08, context_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.18)
        filename1 = "results/experiment2_summary_1_stability.png"
        plt.savefig(filename1, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        summary_files.append(filename1)
        
        # === IMAGE 2: Fit Quality and Statistical Robustness ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Image 2/5: Statistical Robustness - Fit Quality and Model Validation', fontsize=16, fontweight='bold', y=0.92)
        
        # Plot 1: R² fit quality
        for i, (scheme, color, label) in enumerate(zip(['scheme_i', 'scheme_ii', 'scheme_both'], colors, labels)):
            ax1.scatter([xi + i*0.25 for xi in x], r_squared[scheme], 
                       c=color, label=label, s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax1.set_title('Parabolic Fit Quality (R²)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Configuration Index', fontsize=12)
        ax1.set_ylabel('R²', fontsize=12)
        ax1.axhline(y=0.99, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Excellent (0.99)')
        ax1.axhline(y=0.95, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Good (0.95)')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.9, 1.001])
        
        # Plot 2: Distribution of R² values
        all_r2 = []
        for scheme in ['scheme_i', 'scheme_ii', 'scheme_both']:
            all_r2.extend(r_squared[scheme])
        
        ax2.hist(all_r2, bins=20, alpha=0.7, color='green', edgecolor='black', linewidth=1)
        ax2.axvline(x=np.mean(all_r2), color='red', linestyle='-', linewidth=3, label=f'Mean: {np.mean(all_r2):.4f}')
        ax2.axvline(x=0.99, color='orange', linestyle='--', linewidth=2, label='Excellent threshold')
        ax2.set_title('R² Distribution (All Schemes)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('R² Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Context summary
        context_text = f"""FIT QUALITY ANALYSIS CONTEXT:
• Model: Parabolic energy functional ΔE(δ) = C₁δ + C₂δ² fitted to perturbation data
• Quality metrics: R² coefficient of determination measures goodness of fit
• Performance: Mean R² = {np.mean(all_r2):.4f}, indicating excellent parabolic approximation
• Validation: High R² values confirm energy functional's mathematical validity"""
        
        fig.text(0.02, 0.08, context_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.18)
        filename2 = "results/experiment2_summary_2_fit_quality.png"
        plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        summary_files.append(filename2)
        
        # === IMAGE 3: Interference and Non-Additivity ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Image 3/5: Interference Analysis - Zero-Zero Interaction Effects', fontsize=16, fontweight='bold', y=0.92)
        
        # Plot 1: Maximum interference with statistical significance
        max_interf = [stats[0] for stats in interference_stats]
        interf_p_vals = [stats[2] for stats in interference_stats]
        
        # Color-code by significance
        colors_significance = ['red' if p < 0.05 else 'blue' for p in interf_p_vals]
        scatter = ax1.scatter(x, max_interf, c=colors_significance, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax1.set_title('Maximum |I(δ)| by Statistical Significance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Configuration Index', fontsize=12)
        ax1.set_ylabel('Max |I(δ)|', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add legend for significance
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label=f'p < 0.05 (Significant, n={sum(1 for p in interf_p_vals if p < 0.05)})'),
                          Patch(facecolor='blue', label=f'p ≥ 0.05 (Not significant, n={sum(1 for p in interf_p_vals if p >= 0.05)})')]
        ax1.legend(handles=legend_elements, fontsize=11)
        
        # Plot 2: Interference P-values distribution
        interf_p_vals_safe = [max(p, 1e-16) for p in interf_p_vals]
        ax2.scatter(x, interf_p_vals_safe, c='purple', s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax2.set_title('Interference P-values (H₀: I(δ)=0)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Configuration Index', fontsize=12)
        ax2.set_ylabel('P-value', fontsize=12)
        ax2.set_yscale('log')
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, linewidth=3, label='α=0.05')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Context summary
        context_text = f"""INTERFERENCE ANALYSIS CONTEXT:
• Interference ratio: I(δ) = [E(γ₁+δ,γ₂+δ) - E(γ₁,γ₂)] / [E(γ₁+δ,γ₂) + E(γ₂+δ,γ₁) - 2E(γ₁,γ₂)]
• Non-additivity test: Statistical significance of zero-zero interaction effects  
• Results: {significant_interference}/{len(config_groups)} ({float(significant_interference/len(config_groups)*100):.1f}%) configurations show significant interference
• Interpretation: I(δ) ≠ 0 indicates non-additive zero interaction beyond linear superposition"""
        
        fig.text(0.02, 0.08, context_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.18)
        filename3 = "results/experiment2_summary_3_interference.png"
        plt.savefig(filename3, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        summary_files.append(filename3)
        
        # === IMAGE 4: Cross-Coupling and Additivity Testing ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Image 4/5: Cross-Coupling Analysis - Quantitative Additivity Assessment', fontsize=16, fontweight='bold', y=0.92)
        
        # Plot 1: Cross-coupling coefficients with error bars
        additivity_diffs = [res[0] for res in additivity_results]
        additivity_ses = [res[2] for res in additivity_results]
        
        ax1.errorbar(x, cross_coupling_coeffs, yerr=additivity_ses, 
                    fmt='o', color='orange', markersize=8, capsize=4, capthick=2, 
                    alpha=0.8, linewidth=2, elinewidth=2)
        ax1.set_title('Cross-Coupling C₁₂ with Standard Errors', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Configuration Index', fontsize=12)
        ax1.set_ylabel('C₁₂', fontsize=12)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.6, linewidth=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Additivity test results
        additivity_p_vals = [res[1] for res in additivity_results]
        additivity_colors = ['red' if p < 0.05 else 'green' for p in additivity_p_vals]
        
        ax2.scatter(x, additivity_diffs, c=additivity_colors, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax2.set_title('Additivity Test: C₁⁽¹²⁾ - (C₁¹ + C₁²)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Configuration Index', fontsize=12)
        ax2.set_ylabel('Difference', fontsize=12)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.6, linewidth=2)
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [Patch(facecolor='red', label=f'Non-additive (p < 0.05, n={sum(1 for p in additivity_p_vals if p < 0.05)})'),
                          Patch(facecolor='green', label=f'Additive (p ≥ 0.05, n={sum(1 for p in additivity_p_vals if p >= 0.05)})')]
        ax2.legend(handles=legend_elements, fontsize=11)
        
        # Context summary
        context_text = f"""CROSS-COUPLING ANALYSIS CONTEXT:
• Cross-coupling: C₁₂ quantifies deviation from perfect additivity in stability coefficients
• Additivity test: Statistical comparison C₁⁽¹²⁾ vs (C₁¹ + C₁²) with standard error estimation
• Results: {significant_additivity}/{len(config_groups)} ({float(significant_additivity/len(config_groups)*100):.1f}%) configurations show non-additive behavior
• Physical meaning: C₁₂ ≠ 0 indicates coupling between zero positions beyond linear approximation"""
        
        fig.text(0.02, 0.08, context_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.18)
        filename4 = "results/experiment2_summary_4_cross_coupling.png"
        plt.savefig(filename4, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        summary_files.append(filename4)
        
        # === IMAGE 5: Parameter Space and Overall Summary ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Image 5/5: Parameter Space Coverage and Dataset Summary', fontsize=16, fontweight='bold', y=0.92)
        
        # Plot 1: Parameter space coverage with interference coloring
        g1_vals = [g[0] for g in gamma_pairs]
        g2_vals = [g[1] for g in gamma_pairs]
        scatter = ax1.scatter(g1_vals, g2_vals, c=max_interf, s=120, alpha=0.8, 
                             cmap='viridis', edgecolors='black', linewidth=0.5)
        ax1.set_title('Zero-Pair Parameter Space Coverage', fontsize=14, fontweight='bold')
        ax1.set_xlabel('γ₁ (First Zero)', fontsize=12)
        ax1.set_ylabel('γ₂ (Second Zero)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Max |I(δ)|', fontsize=12)
        
        # Plot 2: Overall summary statistics
        categories = ['Stable\nCoefficients\n(C₁ > 0)', 'Significant\nInterference\n(p < 0.05)', 'Non-Additive\nConfigs\n(p < 0.05)']
        values = [float(stable_count/total_count*100), float(significant_interference/len(config_groups)*100), 
                 float(significant_additivity/len(config_groups)*100)]
        colors_bars = ['green', 'red', 'orange']
        
        bars = ax2.bar(categories, values, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Dataset Summary Statistics (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage', fontsize=12)
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Context summary
        min_gamma1 = float(min(g[0] for g in gamma_pairs))
        max_gamma1 = float(max(g[0] for g in gamma_pairs))
        min_gamma2 = float(min(g[1] for g in gamma_pairs))
        max_gamma2 = float(max(g[1] for g in gamma_pairs))
        
        context_text = f"""DATASET SUMMARY CONTEXT:
• Complete analysis: {len(config_groups)} zero-pair configurations in RH critical strip
• Parameter coverage: γ₁ ∈ [{min_gamma1:.1f}, {max_gamma1:.1f}], γ₂ ∈ [{min_gamma2:.1f}, {max_gamma2:.1f}]  
• Statistical robustness: Bootstrap confidence intervals, hypothesis testing, model validation
• Key findings: Mean max interference = {mean_max_interference:.2e}, Mean |cross-coupling| = {mean_cross_coupling:.2e}"""
        
        fig.text(0.02, 0.08, context_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.18)
        filename5 = "results/experiment2_summary_5_parameter_space.png"
        plt.savefig(filename5, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        summary_files.append(filename5)
        
        # Print summary statistics
        print(f"✓ Created 5 focused summary images:")
        print(f"  Overall stability: {float(100*stable_count/total_count):.1f}%")
        print(f"  Significant interference: {significant_interference}/{len(config_groups)} configurations")
        print(f"  Non-additive behavior: {significant_additivity}/{len(config_groups)} configurations")
        print(f"  Mean max interference: {mean_max_interference:.2e}")
        print(f"  Mean |cross-coupling|: {mean_cross_coupling:.2e}")
        
        return summary_files
def run_batch_visualization(data_dir="data"):
    """Process all HDF5 files in data directory and create batch summary"""
    import glob
    
    # Find all HDF5 files in data directory
    hdf5_pattern = os.path.join(data_dir, "*.h5")
    hdf5_files = glob.glob(hdf5_pattern)
    
    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}/")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files in {data_dir}/")
    
    # Process the complete dataset
    for hdf5_file in hdf5_files:
        print(f"Processing {hdf5_file}...")
        result = run_experiment2_viz(hdf5_file)
        print(f"✓ Complete visualization: {result}")

if __name__ == "__main__":
    # Don't run automatically to avoid errors when loading module
    pass
