#!/usr/bin/env sage

import h5py
import numpy as np
import json
import time
from sage.all import *

class Experiment2Math:
    def __init__(self, config_file="experiment2_config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.gamma1 = float(self.config['gamma1'])
        self.gamma2 = float(self.config['gamma2'])
        self.delta_range = float(self.config['delta_range'])
        self.delta_steps = int(self.config['delta_steps'])
        self.test_function_type = self.config['test_function_type']
        self.num_test_functions = int(self.config['num_test_functions'])
        self.output_file = self.config['output_file']
        
        self.delta_values = np.linspace(-self.delta_range, self.delta_range, self.delta_steps)

    def gaussian_test_function(self, x, center, width):
        return exp(-((x - center)**2) / (2 * width**2))

    def compute_energy_change(self, delta, zero_height, perturb_zero):
        total_energy = 0
        
        for i in range(self.num_test_functions):
            center = float(i) / self.num_test_functions - 0.5
            width = 0.1
            
            # Energy with perturbation
            if perturb_zero == 1:
                shifted_height = zero_height + delta
                energy = (shifted_height - 0.5)**2 * self.gaussian_test_function(center, 0, width)
            elif perturb_zero == 2:
                shifted_height = zero_height + delta  
                energy = (shifted_height - 0.5)**2 * self.gaussian_test_function(center, 0.3, width)
            else:  # both
                shifted_height1 = self.gamma1 + delta
                shifted_height2 = self.gamma2 + delta
                energy = ((shifted_height1 - 0.5)**2 + (shifted_height2 - 0.5)**2) * self.gaussian_test_function(center, 0, width)
            
            total_energy += float(energy)
        
        # Baseline energy (no perturbation)
        baseline = 0
        for i in range(self.num_test_functions):
            center = float(i) / self.num_test_functions - 0.5
            width = 0.1
            if perturb_zero == 1:
                baseline += (self.gamma1 - 0.5)**2 * self.gaussian_test_function(center, 0, width)
            elif perturb_zero == 2:
                baseline += (self.gamma2 - 0.5)**2 * self.gaussian_test_function(center, 0.3, width)
            else:
                baseline += ((self.gamma1 - 0.5)**2 + (self.gamma2 - 0.5)**2) * self.gaussian_test_function(center, 0, width)
        
        return total_energy - float(baseline)

    def run_analysis(self):
        # Compute energy changes for all three schemes
        delta_E1 = []
        delta_E2 = []
        delta_E12 = []
        
        for delta in self.delta_values:
            delta_E1.append(self.compute_energy_change(delta, self.gamma1, 1))
            delta_E2.append(self.compute_energy_change(delta, self.gamma2, 2))
            delta_E12.append(self.compute_energy_change(delta, 0, 3))  # both zeros
        
        # Convert to numpy arrays
        delta_E1 = np.array(delta_E1)
        delta_E2 = np.array(delta_E2)
        delta_E12 = np.array(delta_E12)
        
        # Compute interference ratio
        interference = (delta_E12 - delta_E1 - delta_E2) / (np.abs(delta_E1 + delta_E2) + 1e-10)
        
        return {
            'delta': self.delta_values,
            'delta_E1': delta_E1,
            'delta_E2': delta_E2,
            'delta_E12': delta_E12,
            'interference': interference
        }

    def write_to_hdf5(self, results):
        group_name = self.config.get('group_name', 'config_1')
        
        # Open in append mode, create if doesn't exist
        with h5py.File(self.output_file, 'a') as f:
            # Create group for this configuration
            if group_name in f:
                del f[group_name]  # Remove if exists
            group = f.create_group(group_name)
            
            # Metadata for this configuration
            meta = group.create_group('metadata')
            meta.attrs['description'] = 'Experiment 2: Two-zero interaction energy functional analysis'
            meta.attrs['gamma_1'] = self.gamma1
            meta.attrs['gamma_2'] = self.gamma2
            meta.attrs['delta_range'] = [-self.delta_range, self.delta_range, self.delta_range*2/(self.delta_steps-1)]
            meta.attrs['n_steps'] = self.delta_steps
            meta.attrs['test_function_basis'] = self.test_function_type
            
            # Scheme I: shift only gamma1
            scheme_i = group.create_group('scheme_i')
            scheme_i.create_dataset('delta', data=results['delta'])
            scheme_i.create_dataset('delta_E', data=results['delta_E1'])
            scheme_i.create_dataset('dE_d_delta', data=np.gradient(results['delta_E1'], results['delta']))
            
            # Scheme II: shift only gamma2
            scheme_ii = group.create_group('scheme_ii')
            scheme_ii.create_dataset('delta', data=results['delta'])
            scheme_ii.create_dataset('delta_E', data=results['delta_E2'])
            scheme_ii.create_dataset('dE_d_delta', data=np.gradient(results['delta_E2'], results['delta']))
            
            # Scheme Both: shift both zeros
            scheme_both = group.create_group('scheme_both')
            scheme_both.create_dataset('delta', data=results['delta'])
            scheme_both.create_dataset('delta_E', data=results['delta_E12'])
            scheme_both.create_dataset('dE_d_delta', data=np.gradient(results['delta_E12'], results['delta']))
            
            # Interference analysis
            interference = group.create_group('interference_analysis')
            interference.create_dataset('delta', data=results['delta'])
            interference.create_dataset('interference_ratio', data=results['interference'])

def run_experiment2_math(config_file="experiment2_config.json"):
    math_engine = Experiment2Math(config_file)
    results = math_engine.run_analysis()
    math_engine.write_to_hdf5(results)
    print(f"Math results written to {math_engine.output_file}")
    return math_engine.output_file

if __name__ == "__main__":
    run_experiment2_math()
