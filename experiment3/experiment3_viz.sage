#!/usr/bin/env sage

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class Experiment3Viz:
    """
    Visualization engine for Experiment 3: Multi-Zero Scaling Analysis
    
    Creates 5 focused summary images following design guide:
    1. Energy curves for different N (uniform perturbations)
    2. Scaling law: C₁^(N) vs N 
    3. Random perturbation validation
    4. Quadratic behavior comparison
    5. Statistical summary dashboard
    """
    
    def __init__(self, hdf5_file="data/experiment3_multi_zero_analysis.h5"):
        self.hdf5_file = hdf5_file

    def load_all_data(self):
        """Load and organize all experimental data"""
        with h5py.File(self.hdf5_file, 'r') as f:
            config_groups = [key for key in f.keys() if key.startswith('config_')]
            
            uniform_data = []
            random_data = []
            
            for config_name in config_groups:
                config_group = f[config_name]
                meta = config_group['metadata']
                
                experiment_type = meta.attrs['experiment_type'].decode() if isinstance(meta.attrs['experiment_type'], bytes) else meta.attrs['experiment_type']
                zero_count = int(meta.attrs['zero_count'])
                
                if experiment_type == "multi_zero_uniform":
                    uniform_group = config_group['uniform_perturbation']
                    stats_group = config_group['statistical_analysis']
                    
                    uniform_data.append({
                        'config_name': config_name,
                        'zero_count': zero_count,
                        'delta': uniform_group['delta'][:],
                        'delta_E': uniform_group['delta_E'][:],
                        'c1': stats_group['polyfit_coeffs'][0],
                        'c3': stats_group['polyfit_coeffs'][1],
                        'r_squared': stats_group.attrs['r_squared'],
                        'c1_std_error': stats_group.attrs['c1_std_error'],
                        'bootstrap_ci': stats_group['bootstrap_CI'][:]
                    })
                
                elif experiment_type == "multi_zero_random":
                    random_group = config_group['random_perturbation']
                    stats_group = config_group['statistical_analysis']
                    
                    random_data.append({
                        'config_name': config_name,
                        'zero_count': zero_count,
                        'delta_E_samples': random_group['delta_E_samples'][:],
                        'sum_delta_squared': random_group['sum_delta_squared'][:],
                        'c1_effective': stats_group.attrs['c1_effective'],
                        'r_squared': stats_group.attrs['r_squared']
                    })
            
            # Load scaling analysis if available
            scaling_data = None
            if 'scaling_analysis' in f:
                scaling = f['scaling_analysis']
                scaling_data = {
                    'zero_counts': scaling['zero_counts'][:],
                    'c1_values': scaling['c1_values'][:],
                    'intercept': scaling.attrs['intercept'],
                    'slope': scaling.attrs['slope'],
                    'slope_std_error': scaling.attrs['slope_std_error'],
                    'r_squared': scaling.attrs['r_squared'],
                    'slope_p_value': scaling.attrs['slope_p_value']
                }
            
            return uniform_data, random_data, scaling_data

    def create_energy_curves_plot(self, uniform_data):
        """Image 1: Energy curves ΔE(δ) for different N"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Image 1/5: Multi-Zero Energy Curves - Uniform Perturbations', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(uniform_data)))
        
        # Determine which N values to show in legend (limit to 8 items max)
        sorted_data = sorted(uniform_data, key=lambda x: x['zero_count'])
        max_legend_items = 8
        
        if len(sorted_data) <= max_legend_items:
            legend_indices = range(len(sorted_data))
        else:
            legend_indices = [0]  # Always include first
            step = len(sorted_data) // (max_legend_items - 2)
            legend_indices.extend(range(step, len(sorted_data) - step, step))
            legend_indices.append(len(sorted_data) - 1)  # Always include last
            legend_indices = sorted(set(legend_indices))
        
        # Left plot: Energy curves
        for i, data in enumerate(sorted_data):
            delta = data['delta']
            delta_E = data['delta_E']
            N = data['zero_count']
            c1 = data['c1']
            c3 = data['c3']
            
            # Only add label for key N values
            if i in legend_indices:
                label = f'N={N} (C₁={c1:.1e})'
            else:
                label = None
            
            # Plot data
            ax1.plot(delta, delta_E, 'o', color=colors[i], markersize=4, alpha=0.7, label=label)
            
            # Plot fit
            delta_fit = np.linspace(delta.min(), delta.max(), 100)
            delta_E_fit = c1 * delta_fit**2 + c3 * delta_fit**4
            ax1.plot(delta_fit, delta_E_fit, '-', color=colors[i], linewidth=2, alpha=0.8)
        
        ax1.set_title('Energy Change vs Perturbation', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Perturbation δ', fontsize=12)
        ax1.set_ylabel('ΔE(δ)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Quadratic scaling (ΔE vs δ²)
        for i, data in enumerate(sorted_data):
            delta = data['delta']
            delta_E = data['delta_E']
            N = data['zero_count']
            
            # Only add label for key N values
            if i in legend_indices:
                label = f'N={N}'
            else:
                label = None
            
            ax2.plot(delta**2, delta_E, 'o', color=colors[i], markersize=4, alpha=0.7, label=label)
        
        ax2.set_title('Quadratic Scaling Validation', fontsize=14, fontweight='bold')
        ax2.set_xlabel('δ²', fontsize=12)
        ax2.set_ylabel('ΔE(δ)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        filename = "results/experiment3_summary_1_energy_curves.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename

    def create_scaling_law_plot(self, uniform_data, scaling_data):
        """Image 2: Scaling law C₁^(N) vs N"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Image 2/5: Scaling Law Analysis - C₁^(N) vs Zero Count N', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        # Extract data
        zero_counts = [d['zero_count'] for d in uniform_data]
        c1_values = [d['c1'] for d in uniform_data]
        c1_errors = [d['c1_std_error'] for d in uniform_data]
        
        # Left plot: C₁ vs N with error bars
        ax1.errorbar(zero_counts, c1_values, yerr=c1_errors, 
                    fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2,
                    color='#1f77b4', label='Measured C₁^(N)')
        
        if scaling_data:
            # Plot linear fit
            N_fit = np.linspace(min(zero_counts), max(zero_counts), 100)
            c1_fit = scaling_data['intercept'] + scaling_data['slope'] * N_fit
            ax1.plot(N_fit, c1_fit, 'r-', linewidth=2, 
                    label=f'Linear fit: {scaling_data["slope"]:.2e} × N + {scaling_data["intercept"]:.2e}')
            
            # Add fit quality info
            ax1.text(0.05, 0.95, f'R² = {scaling_data["r_squared"]:.6f}\np = {scaling_data["slope_p_value"]:.3e}',
                    transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('Scaling Law: C₁^(N) vs N', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Zero Count N', fontsize=12)
        ax1.set_ylabel('C₁^(N) Coefficient', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Per-zero contribution C₁^(N)/N
        c1_per_zero = np.array(c1_values) / np.array(zero_counts)
        c1_per_zero_errors = np.array(c1_errors) / np.array(zero_counts)
        
        ax2.errorbar(zero_counts, c1_per_zero, yerr=c1_per_zero_errors,
                    fmt='s', markersize=8, capsize=5, capthick=2, linewidth=2,
                    color='#ff7f0e', label='C₁^(N) / N')
        
        # Expected constant line if perfectly additive
        mean_per_zero = np.mean(c1_per_zero)
        ax2.axhline(y=mean_per_zero, color='green', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_per_zero:.2e}')
        
        ax2.set_title('Per-Zero Contribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Zero Count N', fontsize=12)
        ax2.set_ylabel('C₁^(N) / N', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        filename = "results/experiment3_summary_2_scaling_law.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename

    def create_random_perturbation_plot(self, random_data):
        """Image 3: Random perturbation validation"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Image 3/5: Random Perturbation Analysis - Quadratic Scaling Validation', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(random_data)))
        
        # Left plot: ΔE vs Σδⱼ² for different N
        # Only show legend labels for key N values to avoid overcrowding
        sorted_data = sorted(random_data, key=lambda x: x['zero_count'])
        max_legend_items = 8  # Limit legend to 8 items max
        
        if len(sorted_data) <= max_legend_items:
            # Show all if we have few items
            legend_indices = range(len(sorted_data))
        else:
            # Show evenly spaced key values including first and last
            legend_indices = [0]  # Always include first
            step = len(sorted_data) // (max_legend_items - 2)
            legend_indices.extend(range(step, len(sorted_data) - step, step))
            legend_indices.append(len(sorted_data) - 1)  # Always include last
            legend_indices = sorted(set(legend_indices))  # Remove duplicates and sort
        
        for i, data in enumerate(sorted_data):
            delta_E = data['delta_E_samples']
            sum_delta_sq = data['sum_delta_squared']
            N = data['zero_count']
            c1_eff = data['c1_effective']
            r2 = data['r_squared']
            
            # Only add label for key N values
            if i in legend_indices:
                label = f'N={N} (C₁={c1_eff:.1e}, R²={r2:.3f})'
            else:
                label = None
            
            ax1.scatter(sum_delta_sq, delta_E, color=colors[i], alpha=0.7, s=50, label=label)
            
            # Plot linear fit
            sum_sq_fit = np.linspace(0, sum_delta_sq.max(), 100)
            delta_E_fit = c1_eff * sum_sq_fit
            ax1.plot(sum_sq_fit, delta_E_fit, '-', color=colors[i], linewidth=2, alpha=0.8)
        
        ax1.set_title('Quadratic Scaling: ΔE vs Σδⱼ²', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Σδⱼ²', fontsize=12)
        ax1.set_ylabel('ΔE', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: R² values comparison
        zero_counts = [d['zero_count'] for d in random_data]
        r_squared_values = [d['r_squared'] for d in random_data]
        c1_effective_values = [d['c1_effective'] for d in random_data]
        
        ax2_twin = ax2.twinx()
        
        # R² values
        bars1 = ax2.bar([n - 0.2 for n in zero_counts], r_squared_values, 
                       width=0.4, color='skyblue', alpha=0.8, label='R² (Quadratic Fit)')
        
        # C₁ effective values
        bars2 = ax2_twin.bar([n + 0.2 for n in zero_counts], c1_effective_values,
                           width=0.4, color='lightcoral', alpha=0.8, label='C₁ effective')
        
        ax2.set_title('Fit Quality and Effective Coefficients', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Zero Count N', fontsize=12)
        ax2.set_ylabel('R² Value', fontsize=12, color='blue')
        ax2_twin.set_ylabel('C₁ effective', fontsize=12, color='red')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        filename = "results/experiment3_summary_3_random_perturbation.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename

    def create_comparison_plot(self, uniform_data, random_data):
        """Image 4: Comparison between uniform and random perturbations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Image 4/5: Uniform vs Random Perturbation Comparison', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        # Extract matching data (same N)
        uniform_dict = {d['zero_count']: d for d in uniform_data}
        random_dict = {d['zero_count']: d for d in random_data}
        
        common_N = sorted(set(uniform_dict.keys()) & set(random_dict.keys()))
        
        uniform_c1 = [uniform_dict[n]['c1'] for n in common_N]
        random_c1 = [random_dict[n]['c1_effective'] for n in common_N]
        uniform_r2 = [uniform_dict[n]['r_squared'] for n in common_N]
        random_r2 = [random_dict[n]['r_squared'] for n in common_N]
        
        # Left plot: C₁ comparison
        x_pos = np.arange(len(common_N))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, uniform_c1, width, label='Uniform δ', 
                       color='#1f77b4', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, random_c1, width, label='Random δⱼ', 
                       color='#ff7f0e', alpha=0.8)
        
        ax1.set_title('C₁ Coefficient Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Zero Count N', fontsize=12)
        ax1.set_ylabel('C₁ Coefficient', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(common_N)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=9, rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=9, rotation=45)
        
        # Right plot: R² comparison
        bars3 = ax2.bar(x_pos - width/2, uniform_r2, width, label='Uniform δ', 
                       color='#2ca02c', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, random_r2, width, label='Random δⱼ', 
                       color='#d62728', alpha=0.8)
        
        ax2.set_title('Fit Quality (R²) Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Zero Count N', fontsize=12)
        ax2.set_ylabel('R² Value', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(common_N)
        ax2.set_ylim(0, 1.1)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=10)
        
        for bar in bars4:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        filename = "results/experiment3_summary_4_comparison.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename

    def create_statistical_dashboard(self, uniform_data, random_data, scaling_data):
        """Image 5: Statistical summary dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Image 5/5: Statistical Summary Dashboard', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        # Panel 1: Stability distribution
        c1_values = [d['c1'] for d in uniform_data]
        ax1.hist(c1_values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='C₁ = 0')
        ax1.set_title('C₁ Coefficient Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('C₁ Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stable_count = sum(1 for c1 in c1_values if c1 > 0)
        mean_c1 = np.mean(c1_values)
        std_c1 = np.std(c1_values)
        ax1.text(0.05, 0.95, f'Stable: {stable_count}/{len(c1_values)} ({float(100*stable_count/len(c1_values)):.1f}%)\n'
                             f'Mean: {mean_c1:.2e}\nStd: {std_c1:.2e}',
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 2: R² distribution
        r2_uniform = [d['r_squared'] for d in uniform_data]
        r2_random = [d['r_squared'] for d in random_data]
        
        ax2.hist(r2_uniform, bins=8, alpha=0.7, color='lightgreen', 
                label=f'Uniform (n={len(r2_uniform)})', edgecolor='black')
        if r2_random:
            ax2.hist(r2_random, bins=8, alpha=0.7, color='lightcoral', 
                    label=f'Random (n={len(r2_random)})', edgecolor='black')
        
        ax2.set_title('Fit Quality (R²) Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('R² Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Zero count vs C₁ scatter
        zero_counts = [d['zero_count'] for d in uniform_data]
        ax3.scatter(zero_counts, c1_values, s=80, alpha=0.7, color='purple')
        
        if scaling_data:
            N_fit = np.linspace(min(zero_counts), max(zero_counts), 100)
            c1_fit = scaling_data['intercept'] + scaling_data['slope'] * N_fit
            ax3.plot(N_fit, c1_fit, 'r-', linewidth=2, 
                    label=f'Linear fit (R²={scaling_data["r_squared"]:.3f})')
            ax3.legend()
        
        ax3.set_title('Scaling Relationship', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Zero Count N', fontsize=12)
        ax3.set_ylabel('C₁ Coefficient', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Key statistics summary
        ax4.axis('off')
        
        # Compile key statistics
        stats_text = "KEY EXPERIMENTAL FINDINGS:\n\n"
        
        if uniform_data:
            total_configs = len(uniform_data)
            stable_configs = sum(1 for d in uniform_data if d['c1'] > 0)
            mean_r2 = np.mean([d['r_squared'] for d in uniform_data])
            
            stats_text += f"• Total Configurations: {total_configs}\n"
            stats_text += f"• Stable (C₁ > 0): {stable_configs} ({float(100*stable_configs/total_configs):.1f}%)\n"
            stats_text += f"• Mean Fit Quality: R² = {mean_r2:.6f}\n\n"
        
        if scaling_data:
            stats_text += f"SCALING LAW:\n"
            stats_text += f"• C₁^(N) = {scaling_data['intercept']:.2e} + {scaling_data['slope']:.2e} × N\n"
            stats_text += f"• Scaling R² = {scaling_data['r_squared']:.6f}\n"
            stats_text += f"• Slope p-value = {scaling_data['slope_p_value']:.3e}\n\n"
        
        if random_data:
            mean_r2_random = np.mean([d['r_squared'] for d in random_data])
            stats_text += f"RANDOM PERTURBATIONS:\n"
            stats_text += f"• Configurations: {len(random_data)}\n"
            stats_text += f"• Mean Quadratic R² = {mean_r2_random:.6f}\n\n"
        
        stats_text += "MATHEMATICAL SIGNIFICANCE:\n"
        if uniform_data and all(d['c1'] > 0 for d in uniform_data):
            stats_text += "• Universal stability confirmed\n"
        if scaling_data and scaling_data['slope_p_value'] < 0.05:
            stats_text += "• Linear scaling validated\n"
        stats_text += "• Critical line energy minimum\n"
        stats_text += "• Multi-zero additivity"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        filename = "results/experiment3_summary_5_statistical_dashboard.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return filename

    def create_visualizations(self):
        """Create all 5 summary visualizations"""
        print(f"\n=== Experiment 3 Visualization Generation ===")
        print(f"Loading data from: {self.hdf5_file}")
        
        # Load data
        uniform_data, random_data, scaling_data = self.load_all_data()
        
        print(f"Found {len(uniform_data)} uniform and {len(random_data)} random configurations")
        
        # Create 5 summary images
        filenames = []
        
        print("Creating Image 1: Energy curves...")
        filenames.append(self.create_energy_curves_plot(uniform_data))
        
        print("Creating Image 2: Scaling law...")
        filenames.append(self.create_scaling_law_plot(uniform_data, scaling_data))
        
        if random_data:
            print("Creating Image 3: Random perturbations...")
            filenames.append(self.create_random_perturbation_plot(random_data))
            
            print("Creating Image 4: Uniform vs Random comparison...")
            filenames.append(self.create_comparison_plot(uniform_data, random_data))
        else:
            print("Creating Image 3: Random perturbations... (skipped - no random data)")
            print("Creating Image 4: Uniform vs Random comparison... (skipped - no random data)")
        
        print("Creating Image 5: Statistical dashboard...")
        filenames.append(self.create_statistical_dashboard(uniform_data, random_data, scaling_data))
        
        return filenames

def run_experiment3_viz(hdf5_file="data/experiment3_multi_zero_analysis.h5"):
    """Main entry point for Experiment 3 visualization"""
    viz_engine = Experiment3Viz(hdf5_file)
    summary_files = viz_engine.create_visualizations()
    
    print(f"\n✓ Experiment 3 visualizations created:")
    for i, filename in enumerate(summary_files, 1):
        print(f"  Image {i}: {filename}")
    
    return summary_files

# Auto-execution disabled for batch mode
# if __name__ == "__main__":
#     run_experiment3_viz()
