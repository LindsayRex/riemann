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

def run_experiment2_viz(hdf5_file="experiment2_two_zero_interaction.h5"):
    """Generate individual plots for this configuration"""
    viz = Experiment2Viz(hdf5_file)
    plot_files = viz.create_plots()
    
    with h5py.File(hdf5_file, 'r') as f:
        meta = f['metadata']
        gamma1, gamma2 = meta.attrs['gamma_1'], meta.attrs['gamma_2']
        
        # Verify statistical data is present
        schemes = ['scheme_i', 'scheme_ii', 'scheme_both']
        stats_present = all(
            'polyfit_coeffs' in f[scheme] and 'r_squared' in f[scheme].attrs 
            for scheme in schemes
        )
        
        print(f"✓ Plots created for γ₁={gamma1:.2f}, γ₂={gamma2:.2f}")
        print(f"  Energy curves: {plot_files[0]}")
        print(f"  Additivity analysis: {plot_files[1]}")
        print(f"  Statistical data: {'Present' if stats_present else 'Missing'}")
        
        return plot_files

def create_statistical_summary(hdf5_files):
    """Create comprehensive statistical summary plots for batch analysis"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Experiment 2: Complete Statistical Analysis', fontsize=16)
    
    # Collect statistical data
    gamma_pairs = []
    c1_values = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
    c1_errors = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
    r_squared = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
    p_values = {'scheme_i': [], 'scheme_ii': [], 'scheme_both': []}
    interference_stats = []
    cross_coupling_coeffs = []
    
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, 'r') as f:
            meta = f['metadata']
            gamma1, gamma2 = float(meta.attrs['gamma_1']), float(meta.attrs['gamma_2'])
            gamma_pairs.append((gamma1, gamma2))
            
            for scheme in ['scheme_i', 'scheme_ii', 'scheme_both']:
                data = f[scheme]
                coeffs = data['polyfit_coeffs'][:]
                c1_values[scheme].append(float(coeffs[0]))
                c1_errors[scheme].append(float(data.attrs.get('c1_std_error', 0)))
                r_squared[scheme].append(float(data.attrs.get('r_squared', 0)))
                p_values[scheme].append(float(data.attrs.get('c1_p_value', 1)))
            
            # Interference analysis data
            interference = f['interference_analysis']
            max_interference = float(np.max(np.abs(interference['interference_ratio'][:])))
            mean_interference = float(np.mean(np.abs(interference['interference_ratio'][:])))
            interference_p_value = float(interference.attrs.get('interference_p_value', 1))
            interference_stats.append((max_interference, mean_interference, interference_p_value))
            
            # Cross-coupling coefficient
            cross_coupling = float(f['interference_analysis'].attrs.get('cross_coupling_coeff', 0))
            cross_coupling_coeffs.append(cross_coupling)
    
    colors = ['blue', 'green', 'red']
    labels = ['γ₁ only', 'γ₂ only', 'Both']
    x = range(len(gamma_pairs))
    
    # Row 1: Stability Analysis
    # Plot 1: C₁ coefficients with error bars
    for i, (scheme, color, label) in enumerate(zip(['scheme_i', 'scheme_ii', 'scheme_both'], colors, labels)):
        axes[0,0].errorbar([xi + i*0.25 for xi in x], c1_values[scheme], 
                          yerr=c1_errors[scheme], fmt='o', color=color, label=label, markersize=4)
    
    axes[0,0].set_title('C₁ Coefficients (Stability)')
    axes[0,0].set_xlabel('Configuration')
    axes[0,0].set_ylabel('C₁')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: P-values for C₁ > 0
    for i, (scheme, color, label) in enumerate(zip(['scheme_i', 'scheme_ii', 'scheme_both'], colors, labels)):
        # Add small epsilon to avoid log(0) warnings
        p_vals_safe = [max(p, 1e-16) for p in p_values[scheme]]
        axes[0,1].scatter([xi + i*0.25 for xi in x], p_vals_safe, 
                         c=color, label=label, s=30, alpha=0.7)
    
    axes[0,1].set_title('P-values for C₁ > 0')
    axes[0,1].set_xlabel('Configuration')
    axes[0,1].set_ylabel('P-value')
    axes[0,1].set_yscale('log')
    axes[0,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α=0.05')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: R² values
    for i, (scheme, color, label) in enumerate(zip(['scheme_i', 'scheme_ii', 'scheme_both'], colors, labels)):
        axes[0,2].scatter([xi + i*0.25 for xi in x], r_squared[scheme], 
                         c=color, label=label, s=30, alpha=0.7)
    
    axes[0,2].set_title('Fit Quality (R²)')
    axes[0,2].set_xlabel('Configuration')
    axes[0,2].set_ylabel('R²')
    axes[0,2].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Good fit')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Row 2: Interference Analysis
    # Plot 4: Maximum interference ratios
    max_interf = [stats[0] for stats in interference_stats]
    axes[1,0].bar(x, max_interf, alpha=0.7, color='purple')
    axes[1,0].set_title('Maximum Interference |I(δ)|')
    axes[1,0].set_xlabel('Configuration')
    axes[1,0].set_ylabel('Max |I(δ)|')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Interference P-values
    interf_p_vals = [max(stats[2], 1e-16) for stats in interference_stats]
    axes[1,1].scatter(x, interf_p_vals, c='purple', s=50, alpha=0.7)
    axes[1,1].set_title('Interference Significance')
    axes[1,1].set_xlabel('Configuration')
    axes[1,1].set_ylabel('P-value for I(δ)=0')
    axes[1,1].set_yscale('log')
    axes[1,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='α=0.05')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Cross-coupling coefficients
    axes[1,2].bar(x, cross_coupling_coeffs, alpha=0.7, color='orange')
    axes[1,2].set_title('Cross-Coupling C₁₂')
    axes[1,2].set_xlabel('Configuration')
    axes[1,2].set_ylabel('C₁₂')
    axes[1,2].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[1,2].grid(True, alpha=0.3)
    
    # Row 3: Summary Analysis
    # Plot 7: Overall stability
    stable_count = sum(1 for scheme in ['scheme_i', 'scheme_ii', 'scheme_both'] 
                      for c1 in c1_values[scheme] if c1 > 0)
    total_count = len(hdf5_files) * 3
    
    axes[2,0].bar(['Stable', 'Unstable'], 
                  [stable_count, total_count - stable_count],
                  color=['green', 'red'], alpha=0.7)
    axes[2,0].set_title(f'Overall Stability: {stable_count}/{total_count} ({float(100*stable_count/total_count):.1f}%)')
    axes[2,0].set_ylabel('Count')
    
    # Plot 8: Interference vs Stability
    stable_markers = [c1_values['scheme_both'][i] > 0 for i in range(len(gamma_pairs))]
    stable_interf = [max_interf[i] for i in range(len(max_interf)) if stable_markers[i]]
    unstable_interf = [max_interf[i] for i in range(len(max_interf)) if not stable_markers[i]]
    
    if stable_interf:
        axes[2,1].hist(stable_interf, bins=10, alpha=0.7, color='green', label='Stable')
    if unstable_interf:
        axes[2,1].hist(unstable_interf, bins=10, alpha=0.7, color='red', label='Unstable')
    
    axes[2,1].set_title('Interference Distribution')
    axes[2,1].set_xlabel('Max |I(δ)|')
    axes[2,1].set_ylabel('Count')
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    # Plot 9: Parameter space coverage
    g1_vals = [g[0] for g in gamma_pairs]
    g2_vals = [g[1] for g in gamma_pairs]
    scatter = axes[2,2].scatter(g1_vals, g2_vals, c=max_interf, s=100, alpha=0.7, cmap='viridis')
    axes[2,2].set_title('Zero Pair Coverage')
    axes[2,2].set_xlabel('γ₁')
    axes[2,2].set_ylabel('γ₂')
    axes[2,2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2,2], label='Max |I(δ)|')
    
    plt.tight_layout()
    filename = "experiment2_statistical_summary.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    significant_interference = sum(1 for stats in interference_stats if stats[2] < 0.05)
    mean_max_interference = float(np.mean([stats[0] for stats in interference_stats]))
    
    print(f"✓ Complete statistical summary: {filename}")
    print(f"  Analyzed {len(hdf5_files)} configurations")
    print(f"  Overall stability: {float(100*stable_count/total_count):.1f}%")
    print(f"  Significant interference: {significant_interference}/{len(hdf5_files)} configurations")
    print(f"  Mean max interference: {mean_max_interference:.2e}")
    
    return filename

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
    
    # Process individual files
    validated_files = []
    for hdf5_file in hdf5_files:
        print(f"Processing {hdf5_file}...")
        result = run_experiment2_viz(hdf5_file)
        validated_files.append(result)
    
    # Create batch statistical summary
    print(f"\nGenerating batch statistical summary...")
    summary_file = create_statistical_summary(hdf5_files)
    
    print(f"\n✓ Batch visualization complete!")
    print(f"  Summary plot: {summary_file}")
    print(f"  Processed {len(hdf5_files)} configurations")

if __name__ == "__main__":
    run_batch_visualization()
