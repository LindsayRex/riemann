#!/usr/bin/env sage

import h5py
import numpy as np
import json
from sage.all import *

try:
    from scipy import stats as scipy_stats
    scipy_available = True
except ImportError:
    scipy_available = False

class Experiment3Stats:
    """
    Statistical analysis for Experiment 3: Multi-Zero Scaling
    
    Tests:
    1. Scaling law: ΔE_N(δ) ≈ C₁^(N) δ² where C₁^(N) ≈ Σ C₁(γⱼ)
    2. Quadratic behavior validation
    3. Higher-order term significance
    4. Random perturbation vs uniform perturbation consistency
    """
    
    def __init__(self, hdf5_file="data/experiment3_multi_zero_analysis.h5"):
        self.hdf5_file = hdf5_file

    def fit_quadratic(self, delta, delta_E):
        """Fit: delta_E = C1*delta^2 + C2*delta^3 + C3*delta^4"""
        # Use only even powers due to symmetry
        X = np.column_stack([delta**2, delta**4])
        try:
            coeffs = np.linalg.lstsq(X, delta_E, rcond=None)[0]
            C1, C3 = coeffs[0], coeffs[1]
            
            # Calculate R^2
            y_pred = C1 * delta**2 + C3 * delta**4
            ss_res = np.sum((delta_E - y_pred)**2)
            ss_tot = np.sum((delta_E - np.mean(delta_E))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Standard errors for coefficients
            residuals = delta_E - y_pred
            mse = np.sum(residuals**2) / (len(delta) - 2)
            try:
                cov_matrix = mse * np.linalg.inv(X.T @ X)
                std_errors = np.sqrt(np.diag(cov_matrix))
            except:
                std_errors = [0.0, 0.0]
            
            return [C1, C3], r_squared, std_errors
        except:
            return [0.0, 0.0], 0.0, [0.0, 0.0]

    def bootstrap_ci(self, delta, delta_E, n_bootstrap=100):
        """Bootstrap confidence intervals for coefficients"""
        n = len(delta)
        coeffs_list = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            boot_delta = delta[indices]
            boot_E = delta_E[indices]
            coeffs, _, _ = self.fit_quadratic(boot_delta, boot_E)
            coeffs_list.append(coeffs)
        
        coeffs_array = np.array(coeffs_list)
        ci_lower = np.percentile(coeffs_array, 2.5, axis=0)
        ci_upper = np.percentile(coeffs_array, 97.5, axis=0)
        
        return np.column_stack([ci_lower, ci_upper])

    def compute_c1_p_value(self, c1_val, std_error, n_points):
        """One-sided t-test for C1 > 0"""
        if std_error > 0:
            t_stat = c1_val / std_error
            if scipy_available:
                p_value = 1 - scipy_stats.t.cdf(t_stat, df=n_points-2)
            else:
                # Rough approximation
                p_value = max(0.001, 1 - min(0.999, t_stat / 3))
        else:
            p_value = 0.5
        return p_value

    def analyze_scaling_law(self, c1_values, zero_counts):
        """
        Test scaling law: C₁^(N) vs N
        Expected: C₁^(N) ≈ linear in N (additive behavior)
        """
        # Linear regression: C₁^(N) = a + b*N
        X = np.column_stack([np.ones(len(zero_counts)), zero_counts])
        try:
            coeffs = np.linalg.lstsq(X, c1_values, rcond=None)[0]
            a, b = coeffs[0], coeffs[1]
            
            # R^2 for linearity
            y_pred = a + b * np.array(zero_counts)
            ss_res = np.sum((c1_values - y_pred)**2)
            ss_tot = np.sum((c1_values - np.mean(c1_values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Test if slope b > 0 (positive scaling)
            residuals = c1_values - y_pred
            mse = np.sum(residuals**2) / (len(zero_counts) - 2)
            cov_matrix = mse * np.linalg.inv(X.T @ X)
            b_std_error = np.sqrt(cov_matrix[1, 1])
            
            # P-value for b > 0
            if b_std_error > 0:
                t_stat = b / b_std_error
                if scipy_available:
                    p_value = 1 - scipy_stats.t.cdf(t_stat, df=len(zero_counts)-2)
                else:
                    p_value = max(0.001, 1 - min(0.999, t_stat / 3))
            else:
                p_value = 0.5
            
            return {
                'intercept': a,
                'slope': b,
                'slope_std_error': b_std_error,
                'r_squared': r_squared,
                'slope_p_value': p_value
            }
        except:
            return {
                'intercept': 0,
                'slope': 0,
                'slope_std_error': 0,
                'r_squared': 0,
                'slope_p_value': 1
            }

    def analyze_random_perturbation_scaling(self, delta_E_samples, sum_delta_squared):
        """
        Test quadratic scaling in random perturbations:
        ΔE ≈ C₁ Σδⱼ² (should be linear relationship)
        """
        # Linear regression: ΔE = C₁ * Σδⱼ²
        X = sum_delta_squared.reshape(-1, 1)
        try:
            coeffs = np.linalg.lstsq(X, delta_E_samples, rcond=None)[0]
            c1_effective = coeffs[0]
            
            # R^2
            y_pred = c1_effective * sum_delta_squared
            ss_res = np.sum((delta_E_samples - y_pred)**2)
            ss_tot = np.sum((delta_E_samples - np.mean(delta_E_samples))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Standard error
            residuals = delta_E_samples - y_pred
            mse = np.sum(residuals**2) / (len(delta_E_samples) - 1)
            c1_std_error = np.sqrt(mse / np.sum(sum_delta_squared**2))
            
            return {
                'c1_effective': c1_effective,
                'c1_std_error': c1_std_error,
                'r_squared': r_squared
            }
        except:
            return {
                'c1_effective': 0,
                'c1_std_error': 0,
                'r_squared': 0
            }

    def run_statistical_analysis(self):
        """Run complete statistical analysis on all configurations"""
        
        print(f"\n=== Experiment 3 Statistical Analysis ===")
        print(f"Analyzing data from: {self.hdf5_file}")
        
        with h5py.File(self.hdf5_file, 'a') as f:
            config_groups = [key for key in f.keys() if key.startswith('config_')]
            print(f"Found {len(config_groups)} configurations")
            
            # Collect data for scaling analysis
            uniform_results = []
            random_results = []
            
            for config_name in config_groups:
                config_group = f[config_name]
                meta = config_group['metadata']
                
                experiment_type = meta.attrs['experiment_type'].decode() if isinstance(meta.attrs['experiment_type'], bytes) else meta.attrs['experiment_type']
                zero_count = int(meta.attrs['zero_count'])
                
                print(f"\nAnalyzing {config_name}...")
                
                if experiment_type == "multi_zero_uniform":
                    # Analyze uniform perturbation
                    uniform_data = config_group['uniform_perturbation']
                    delta = uniform_data['delta'][:]
                    delta_E = uniform_data['delta_E'][:]
                    
                    # Fit quadratic model
                    coeffs, r_squared, std_errors = self.fit_quadratic(delta, delta_E)
                    c1, c3 = coeffs[0], coeffs[1]
                    
                    # Bootstrap confidence intervals
                    bootstrap_ci = self.bootstrap_ci(delta, delta_E)
                    
                    # P-value for C1 > 0
                    c1_p_value = self.compute_c1_p_value(c1, std_errors[0], len(delta))
                    
                    # Store statistical results
                    if 'statistical_analysis' in config_group:
                        del config_group['statistical_analysis']  # Remove if exists
                    stats_group = config_group.create_group('statistical_analysis')
                    stats_group.create_dataset('polyfit_coeffs', data=[c1, c3])
                    stats_group.create_dataset('bootstrap_CI', data=bootstrap_ci)
                    stats_group.attrs['r_squared'] = r_squared
                    stats_group.attrs['c1_std_error'] = std_errors[0]
                    stats_group.attrs['c3_std_error'] = std_errors[1]
                    stats_group.attrs['c1_p_value'] = c1_p_value
                    
                    # Collect for scaling analysis
                    uniform_results.append({
                        'zero_count': zero_count,
                        'c1': c1,
                        'c1_std_error': std_errors[0],
                        'r_squared': r_squared
                    })
                    
                    print(f"  Uniform: C₁={c1:.2e}, R²={r_squared:.6f}, p={c1_p_value:.3e}")
                
                elif experiment_type == "multi_zero_random":
                    # Analyze random perturbation
                    random_data = config_group['random_perturbation']
                    delta_E_samples = random_data['delta_E_samples'][:]
                    sum_delta_squared = random_data['sum_delta_squared'][:]
                    
                    # Analyze scaling
                    scaling_results = self.analyze_random_perturbation_scaling(delta_E_samples, sum_delta_squared)
                    
                    # Store results
                    if 'statistical_analysis' in config_group:
                        del config_group['statistical_analysis']  # Remove if exists
                    stats_group = config_group.create_group('statistical_analysis')
                    stats_group.attrs['c1_effective'] = scaling_results['c1_effective']
                    stats_group.attrs['c1_std_error'] = scaling_results['c1_std_error']
                    stats_group.attrs['r_squared'] = scaling_results['r_squared']
                    
                    # Collect for comparison
                    random_results.append({
                        'zero_count': zero_count,
                        'c1_effective': scaling_results['c1_effective'],
                        'c1_std_error': scaling_results['c1_std_error'],
                        'r_squared': scaling_results['r_squared']
                    })
                    
                    print(f"  Random: C₁_eff={scaling_results['c1_effective']:.2e}, R²={scaling_results['r_squared']:.6f}")
            
            # Analyze scaling law across different N
            if uniform_results:
                print(f"\n--- Scaling Law Analysis ---")
                zero_counts = [r['zero_count'] for r in uniform_results]
                c1_values = [r['c1'] for r in uniform_results]
                
                scaling_analysis = self.analyze_scaling_law(c1_values, zero_counts)
                
                # Store scaling analysis
                if 'scaling_analysis' in f:
                    del f['scaling_analysis']  # Remove if exists
                scaling_group = f.create_group('scaling_analysis')
                scaling_group.attrs['intercept'] = scaling_analysis['intercept']
                scaling_group.attrs['slope'] = scaling_analysis['slope']
                scaling_group.attrs['slope_std_error'] = scaling_analysis['slope_std_error']
                scaling_group.attrs['r_squared'] = scaling_analysis['r_squared']
                scaling_group.attrs['slope_p_value'] = scaling_analysis['slope_p_value']
                
                scaling_group.create_dataset('zero_counts', data=zero_counts)
                scaling_group.create_dataset('c1_values', data=c1_values)
                
                print(f"Scaling: C₁^(N) = {scaling_analysis['intercept']:.2e} + {scaling_analysis['slope']:.2e} × N")
                print(f"R²={scaling_analysis['r_squared']:.6f}, slope p-value={scaling_analysis['slope_p_value']:.3e}")
            
            # Generate summary report
            self.generate_summary_report(f, uniform_results, random_results)

    def generate_summary_report(self, hdf5_file_handle, uniform_results, random_results):
        """Generate comprehensive summary report"""
        
        report_lines = []
        report_lines.append("EXPERIMENT 3: MULTI-ZERO SCALING ANALYSIS")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Timestamp and dataset info
        import time
        report_lines.append(f"Analysis Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Dataset: {len(uniform_results)} uniform + {len(random_results)} random configurations")
        
        if uniform_results:
            zero_counts = [r['zero_count'] for r in uniform_results]
            report_lines.append(f"Zero Count Range: N ∈ [{min(zero_counts)}, {max(zero_counts)}]")
        report_lines.append("")
        
        # Uniform perturbation summary
        if uniform_results:
            report_lines.append("UNIFORM PERTURBATION ANALYSIS:")
            report_lines.append("-" * 40)
            
            stable_count = sum(1 for r in uniform_results if r['c1'] > 0)
            mean_c1 = float(np.mean([r['c1'] for r in uniform_results]))
            mean_r2 = float(np.mean([r['r_squared'] for r in uniform_results]))
            
            report_lines.append(f"Total Configurations: {len(uniform_results)}")
            report_lines.append(f"Stable Coefficients (C₁ > 0): {stable_count} ({float(100*stable_count/len(uniform_results)):.1f}%)")
            report_lines.append(f"Mean C₁ Coefficient: {mean_c1:.6e}")
            report_lines.append(f"Mean R² (Fit Quality): {mean_r2:.6f}")
            
            # Scaling law
            if 'scaling_analysis' in hdf5_file_handle:
                scaling = hdf5_file_handle['scaling_analysis']
                slope = float(scaling.attrs['slope'])
                slope_p = float(scaling.attrs['slope_p_value'])
                scaling_r2 = float(scaling.attrs['r_squared'])
                
                report_lines.append(f"Scaling Law: C₁^(N) ∝ N with slope {slope:.2e}")
                report_lines.append(f"Scaling R²: {scaling_r2:.6f}, p-value: {slope_p:.3e}")
            report_lines.append("")
        
        # Random perturbation summary
        if random_results:
            report_lines.append("RANDOM PERTURBATION ANALYSIS:")
            report_lines.append("-" * 40)
            
            mean_c1_eff = float(np.mean([r['c1_effective'] for r in random_results]))
            mean_r2_random = float(np.mean([r['r_squared'] for r in random_results]))
            
            report_lines.append(f"Total Random Configurations: {len(random_results)}")
            report_lines.append(f"Mean Effective C₁: {mean_c1_eff:.6e}")
            report_lines.append(f"Mean R² (Quadratic Scaling): {mean_r2_random:.6f}")
            report_lines.append("")
        
        # Detailed results table
        report_lines.append("DETAILED CONFIGURATION RESULTS:")
        report_lines.append("-" * 40)
        
        # Uniform results
        if uniform_results:
            report_lines.append("Config                           N    C₁              R²        p-value")
            report_lines.append("-" * 70)
            
            for i, result in enumerate(uniform_results):
                config_name = f"uniform_{result['zero_count']}zeros"
                c1_val = float(result['c1'])
                r2_val = float(result['r_squared'])
                p_val = float(result.get('p_value', 0))
                report_lines.append(f"{config_name:30} {result['zero_count']:3d} "
                                  f"{c1_val:12.6e} {r2_val:8.6f} {p_val:.3e}")
            report_lines.append("")
        
        # Mathematical significance
        report_lines.append("MATHEMATICAL SIGNIFICANCE:")
        report_lines.append("-" * 40)
        
        if uniform_results and all(r['c1'] > 0 for r in uniform_results):
            report_lines.append("Universal Stability: All tested multi-zero configurations show C₁ > 0")
            report_lines.append("Energy Functional: ΔE(δ) ≈ C₁ δ² confirms critical line stability")
        
        if 'scaling_analysis' in hdf5_file_handle:
            scaling = hdf5_file_handle['scaling_analysis']
            slope_p_val = float(scaling.attrs['slope_p_value'])
            if slope_p_val < 0.05:
                report_lines.append("Scaling Law: Linear scaling C₁^(N) ∝ N confirmed (p < 0.05)")
                report_lines.append("Additivity: Multi-zero interactions are approximately additive")
        
        report_lines.append("")
        report_lines.append("EXPERIMENTAL DETAILS:")
        report_lines.append("-" * 40)
        report_lines.append("Energy Functional: E[S] = Σ w_j (D_S(φ_j))²")
        report_lines.append("Test Function Basis: Gaussian functions φ_j")
        report_lines.append("Perturbation Analysis: Uniform δ and random δ_j ~ N(0,σ²)")
        report_lines.append("Statistical Methods: Quadratic fitting, bootstrap CI, scaling analysis")
        report_lines.append("")
        report_lines.append("Generated by Experiment 3 Pipeline - Multi-Zero Scaling Analysis")
        
        # Write report to file
        report_text = "\n".join(report_lines)
        with open("results/experiment3_summary_report.txt", "w") as f:
            f.write(report_text)
        
        print(f"\n✓ Summary report written to: results/experiment3_summary_report.txt")

def run_experiment3_stats(hdf5_file="data/experiment3_multi_zero_analysis.h5"):
    """Main entry point for Experiment 3 statistical analysis"""
    stats_engine = Experiment3Stats(hdf5_file)
    stats_engine.run_statistical_analysis()
    return hdf5_file

# Auto-execution disabled for batch mode
# if __name__ == "__main__":
#     run_experiment3_stats()
