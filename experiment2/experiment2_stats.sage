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
            
            return [C1, -C2], r_squared  # Store as [C1, -C2] per HDF5 spec
        except:
            return [0.0, 0.0], 0.0

    def bootstrap_ci(self, delta, delta_E, n_bootstrap=100):
        # Simple bootstrap confidence intervals
        n = len(delta)
        coeffs_list = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            boot_delta = delta[indices]
            boot_E = delta_E[indices]
            coeffs, _ = self.fit_quadratic(boot_delta, boot_E)
            coeffs_list.append(coeffs)
        
        coeffs_array = np.array(coeffs_list)
        ci_lower = np.percentile(coeffs_array, 2.5, axis=0)
        ci_upper = np.percentile(coeffs_array, 97.5, axis=0)
        
        return np.column_stack([ci_lower, ci_upper])

    def compute_p_values(self, interference):
        # Simple one-sample t-test against zero
        if scipy_available:
            t_stat, p_value = scipy_stats.ttest_1samp(interference, 0)
            return np.full(len(interference), p_value)
        else:
            # Manual t-test
            mean_int = np.mean(interference)
            std_int = np.std(interference, ddof=1)
            n = len(interference)
            t_stat = mean_int / (std_int / np.sqrt(n)) if std_int > 0 else 0
            # Rough p-value approximation
            p_value = 2 * (1 - min(0.999, abs(t_stat) / 3))
            return np.full(len(interference), p_value)

    def process_statistics(self):
        with h5py.File(self.hdf5_file, 'r+') as f:
            # Process each scheme
            for scheme_name in ['scheme_i', 'scheme_ii', 'scheme_both']:
                scheme = f[scheme_name]
                delta = scheme['delta'][:]
                delta_E = scheme['delta_E'][:]
                
                # Polynomial fit
                polyfit_coeffs, r_squared = self.fit_quadratic(delta, delta_E)
                
                # Bootstrap confidence intervals
                bootstrap_CI = self.bootstrap_ci(delta, delta_E)
                
                # Write results back to HDF5
                if 'polyfit_coeffs' in scheme:
                    del scheme['polyfit_coeffs']
                if 'bootstrap_CI' in scheme:
                    del scheme['bootstrap_CI']
                
                scheme.create_dataset('polyfit_coeffs', data=polyfit_coeffs)
                scheme.create_dataset('bootstrap_CI', data=bootstrap_CI)
                scheme.attrs['r_squared'] = r_squared
                scheme.attrs['stability'] = 'stable' if polyfit_coeffs[1] < 0 else 'unstable'  # C2 > 0 means stable (stored as -C2)
            
            # Process interference analysis
            interference_group = f['interference_analysis']
            interference = interference_group['interference_ratio'][:]
            
            # Compute p-values
            p_values = self.compute_p_values(interference)
            
            # Write p-values
            if 'p_values' in interference_group:
                del interference_group['p_values']
            interference_group.create_dataset('p_values', data=p_values)
            
            # Add analysis notes
            mean_interference = np.mean(np.abs(interference))
            max_interference = np.max(np.abs(interference))
            notes = f"Mean |interference|: {mean_interference:.3e}, Max |interference|: {max_interference:.3e}"
            interference_group.attrs['notes'] = notes

def run_experiment2_stats(hdf5_file="experiment2_two_zero_interaction.h5"):
    stats_engine = Experiment2Stats(hdf5_file)
    stats_engine.process_statistics()
    print(f"Statistics computed and written to {hdf5_file}")

if __name__ == "__main__":
    run_experiment2_stats()
