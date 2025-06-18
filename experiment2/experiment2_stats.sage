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

class Experiment2Stats:
    def __init__(self, hdf5_file="experiment2_two_zero_interaction.h5"):
        self.hdf5_file = hdf5_file

    def fit_quadratic(self, delta, delta_E):
        # Fit: delta_E = C1*delta + C2*delta^2
        X = np.column_stack([delta, delta**2])
        try:
            coeffs = np.linalg.lstsq(X, delta_E, rcond=None)[0]
            C1, C2 = coeffs[0], coeffs[1]
            
            # Calculate R^2
            y_pred = C1 * delta + C2 * delta**2
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
            
            return [C1, -C2], r_squared, std_errors  # Store as [C1, -C2] per HDF5 spec
        except:
            return [0.0, 0.0], 0.0, [0.0, 0.0]

    def bootstrap_ci(self, delta, delta_E, n_bootstrap=100):
        # Bootstrap confidence intervals for coefficients
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
        # One-sided t-test for C1 > 0
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

    def compute_interference_stats(self, interference):
        # Comprehensive interference analysis
        mean_interference = np.mean(interference)
        std_interference = np.std(interference, ddof=1)
        max_interference = np.max(np.abs(interference))
        n = len(interference)
        
        # P-value for H0: mean interference = 0
        if std_interference > 0:
            t_stat = mean_interference / (std_interference / np.sqrt(n))
            if scipy_available:
                p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n-1))
            else:
                p_value = 2 * max(0.001, 1 - min(0.999, abs(t_stat) / 3))
        else:
            p_value = 1.0
        
        return {
            'mean_interference': mean_interference,
            'std_interference': std_interference,
            'max_interference': max_interference,
            'p_value': p_value
        }

    def compute_cross_coupling(self, c1_i, c1_ii, c1_both):
        # Cross-coupling coefficient: C12 = C1_both - (C1_i + C1_ii)
        c12 = c1_both - (c1_i + c1_ii)
        return c12

    def additivity_test(self, c1_i, c1_ii, c1_both, ci_i, ci_ii, ci_both):
        # Test if C1_both ≈ C1_i + C1_ii (additivity hypothesis)
        predicted_sum = c1_i + c1_ii
        observed = c1_both
        difference = observed - predicted_sum
        
        # Approximate standard error for the difference
        # SE(C1_both - C1_i - C1_ii) ≈ sqrt(SE_both² + SE_i² + SE_ii²)
        se_i = (ci_i[0, 1] - ci_i[0, 0]) / 3.92  # 95% CI ≈ ±1.96*SE
        se_ii = (ci_ii[0, 1] - ci_ii[0, 0]) / 3.92
        se_both = (ci_both[0, 1] - ci_both[0, 0]) / 3.92
        se_diff = np.sqrt(se_both**2 + se_i**2 + se_ii**2)
        
        # T-test for additivity
        if se_diff > 0:
            t_stat = difference / se_diff
            # Two-sided test
            if scipy_available:
                p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=18))  # Conservative df
            else:
                p_value = 2 * max(0.001, 1 - min(0.999, abs(t_stat) / 2.5))
        else:
            p_value = 1.0
        
        return {
            'difference': difference,
            'se_difference': se_diff,
            'p_value_additivity': p_value,
            'additivity_violated': p_value < 0.05
        }

    def process_statistics(self):
        with h5py.File(self.hdf5_file, 'r+') as f:
            print(f"Processing statistics for all configurations in {self.hdf5_file}")
            
            # Find all configuration groups
            config_groups = [key for key in f.keys() if key.startswith('config_')]
            print(f"Found {len(config_groups)} configurations")
            
            for config_name in config_groups:
                print(f"  Processing {config_name}...")
                config_group = f[config_name]
                
                # Store coefficients and CIs for cross-coupling analysis
                scheme_results = {}
                
                # Process each scheme
                for scheme_name in ['scheme_i', 'scheme_ii', 'scheme_both']:
                    scheme = config_group[scheme_name]
                    delta = scheme['delta'][:]
                    delta_E = scheme['delta_E'][:]
                    n_points = len(delta)
                    
                    # Enhanced polynomial fit with standard errors
                    polyfit_coeffs, r_squared, std_errors = self.fit_quadratic(delta, delta_E)
                    
                    # Bootstrap confidence intervals
                    bootstrap_CI = self.bootstrap_ci(delta, delta_E)
                    
                    # P-value for C1 > 0 test
                    c1_p_value = self.compute_c1_p_value(polyfit_coeffs[0], std_errors[0], n_points)
                    
                    # Store results for cross-coupling analysis
                    scheme_results[scheme_name] = {
                        'c1': polyfit_coeffs[0],
                        'ci': bootstrap_CI,
                        'std_error': std_errors[0]
                    }
                    
                    # Write results back to HDF5
                    if 'polyfit_coeffs' in scheme:
                        del scheme['polyfit_coeffs']
                    if 'bootstrap_CI' in scheme:
                        del scheme['bootstrap_CI']
                    
                    scheme.create_dataset('polyfit_coeffs', data=polyfit_coeffs)
                    scheme.create_dataset('bootstrap_CI', data=bootstrap_CI)
                    scheme.attrs['r_squared'] = float(r_squared)
                    scheme.attrs['c1_std_error'] = float(std_errors[0])
                    scheme.attrs['c1_p_value'] = float(c1_p_value)
                    scheme.attrs['stability'] = 'stable' if polyfit_coeffs[0] > 0 else 'unstable'
                
                # Enhanced interference analysis
                interference_group = config_group['interference_analysis']
                interference = interference_group['interference_ratio'][:]
                
                # Comprehensive interference statistics
                interference_stats = self.compute_interference_stats(interference)
                
                # Write interference results
                if 'p_values' in interference_group:
                    del interference_group['p_values']
                interference_group.create_dataset('p_values', data=np.full(len(interference), interference_stats['p_value']))
                
                # Write interference attributes
                interference_group.attrs['mean_interference'] = float(interference_stats['mean_interference'])
                interference_group.attrs['std_interference'] = float(interference_stats['std_interference'])
                interference_group.attrs['max_interference'] = float(interference_stats['max_interference'])
                interference_group.attrs['interference_p_value'] = float(interference_stats['p_value'])
                
                # Cross-coupling analysis
                c1_i = scheme_results['scheme_i']['c1']
                c1_ii = scheme_results['scheme_ii']['c1']
                c1_both = scheme_results['scheme_both']['c1']
                
                cross_coupling_coeff = self.compute_cross_coupling(c1_i, c1_ii, c1_both)
                interference_group.attrs['cross_coupling_coeff'] = float(cross_coupling_coeff)
                
                # Additivity test
                ci_i = scheme_results['scheme_i']['ci']
                ci_ii = scheme_results['scheme_ii']['ci']
                ci_both = scheme_results['scheme_both']['ci']
                
                additivity_results = self.additivity_test(c1_i, c1_ii, c1_both, ci_i, ci_ii, ci_both)
                
                interference_group.attrs['additivity_difference'] = float(additivity_results['difference'])
                interference_group.attrs['additivity_se'] = float(additivity_results['se_difference'])
                interference_group.attrs['additivity_p_value'] = float(additivity_results['p_value_additivity'])
                interference_group.attrs['additivity_violated'] = bool(additivity_results['additivity_violated'])
                
                # Summary notes
                notes = (f"Interference: mean={interference_stats['mean_interference']:.3e}, "
                        f"max={interference_stats['max_interference']:.3e}, "
                        f"p={interference_stats['p_value']:.3e}; "
                        f"Cross-coupling C12={cross_coupling_coeff:.3e}; "
                        f"Additivity: diff={additivity_results['difference']:.3e}, "
                    f"p={additivity_results['p_value_additivity']:.3e}")
            interference_group.attrs['notes'] = notes

def run_experiment2_stats(hdf5_file="experiment2_two_zero_interaction.h5"):
    print(f"Starting stats processing for {hdf5_file}")
    stats_engine = Experiment2Stats(hdf5_file)
    stats_engine.process_statistics()
    print(f"Statistics computed and written to {hdf5_file}")

if __name__ == "__main__":
    print("Running experiment2_stats.sage")
    run_experiment2_stats()
