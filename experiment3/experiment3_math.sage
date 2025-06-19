#!/usr/bin/env sage

import h5py
import numpy as np
import json
import time
from sage.all import *

class Experiment3Math:
    """
    Experiment 3: Multi-Zero Scaling Analysis
    
    Implements energy functional E[S] for multi-zero perturbations.
    Tests scaling behavior: ΔE_N(δ) vs. Σ C₁(γⱼ) δ² as N increases.
    """
    
    def __init__(self, config_file="experiment3_config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Extract configuration parameters
        self.delta_range = float(self.config['delta_range'])
        self.delta_steps = int(self.config['delta_steps'])
        self.test_function_type = self.config['test_function_type']
        self.num_test_functions = int(self.config['num_test_functions'])
        self.output_file = self.config['output_file']
        self.verbose = self.config.get('verbose', False)
        
        # Generate perturbation values
        self.delta_values = np.linspace(-self.delta_range, self.delta_range, self.delta_steps)
        
        if self.verbose:
            print(f"Experiment 3 Math Engine initialized")
            print(f"Delta range: ±{self.delta_range}, {self.delta_steps} steps")
            print(f"Test functions: {self.num_test_functions} {self.test_function_type}")

    def gaussian_test_function(self, x, center, width):
        """Gaussian test function for energy computation"""
        return exp(-((x - center)**2) / (2 * width**2))

    def compute_discrepancy_field(self, gamma_values, delta_values, perturbation_mode="uniform"):
        """
        Compute D_S(φⱼ) for multi-zero configuration using quadratic energy model
        
        Energy measures deviation from critical line: E = Σⱼ wⱼ (δⱼ)² φⱼ(γⱼ)
        where δⱼ is the real part deviation from 1/2
        
        Args:
            gamma_values: List of zero heights [γ₁, γ₂, ..., γₙ]
            delta_values: Either scalar (uniform) or array (random) perturbations
            perturbation_mode: "uniform" or "random"
        
        Returns:
            Array of discrepancy values for each test function
        """
        N = len(gamma_values)
        discrepancies = np.zeros(self.num_test_functions)
        
        # Get perturbation deltas
        if perturbation_mode == "uniform":
            # All zeros shifted by same δ
            deltas = [delta_values for _ in gamma_values]
        elif perturbation_mode == "random":
            # Each zero shifted by δⱼ
            if isinstance(delta_values, (int, float)):
                # Generate random perturbations
                np.random.seed(42)  # Reproducible
                deltas = np.random.normal(0, abs(delta_values), N)
            else:
                deltas = delta_values
        
        # Compute energy contribution for each test function
        # Test functions span the gamma range to capture interactions
        gamma_min, gamma_max = min(gamma_values), max(gamma_values)
        gamma_span = gamma_max - gamma_min + 20  # Add padding
        
        for j in range(self.num_test_functions):
            # Test function parameters - spread across gamma range
            center = gamma_min - 10 + (float(j) / (self.num_test_functions - 1)) * gamma_span
            width = gamma_span / (3 * self.num_test_functions)  # Overlapping support
            
            # Compute discrepancy: sum of (deviation²) weighted by test function
            total_discrepancy = 0
            
            for k, (gamma, delta) in enumerate(zip(gamma_values, deltas)):
                # Test function evaluation at this zero height
                test_weight = self.gaussian_test_function(gamma, center, width)
                
                # Energy contribution: δ² × test_function_weight
                # This gives the characteristic C₁ δ² behavior
                energy_contribution = (delta**2) * test_weight
                
                total_discrepancy += energy_contribution
            
            # Store the total discrepancy for this test function
            discrepancies[j] = float(total_discrepancy)
        
        return discrepancies

    def compute_energy_change(self, gamma_values, delta, perturbation_mode="uniform", random_deltas=None):
        """
        Compute energy change ΔE(δ) = E[S_c(δ)] - E[S_c]
        
        Energy functional: E[S] = Σⱼ (Σₖ δₖ² φⱼ(γₖ))
        For uniform: E = δ² Σⱼ (Σₖ φⱼ(γₖ)) = δ² × C₁^(N)
        Expected: C₁^(N) ≈ Σₖ C₁(γₖ) (additivity)
        
        Args:
            gamma_values: Zero heights
            delta: Perturbation parameter
            perturbation_mode: "uniform" or "random"
            random_deltas: Specific random perturbations (if mode="random")
        
        Returns:
            Energy change value
        """
        # For this model, baseline energy is always 0 (δ = 0 everywhere)
        # So ΔE = E[perturbed] - E[baseline] = E[perturbed] - 0 = E[perturbed]
        
        if perturbation_mode == "random" and random_deltas is not None:
            discrepancies = self.compute_discrepancy_field(gamma_values, random_deltas, "random")
        else:
            discrepancies = self.compute_discrepancy_field(gamma_values, delta, perturbation_mode)
        
        # Total energy: sum of all discrepancy contributions
        total_energy = np.sum(discrepancies)
        
        return float(total_energy)

    def run_uniform_perturbation_analysis(self, gamma_values, zero_count):
        """
        Analyze uniform perturbation: all zeros shifted by same δ
        Test scaling: ΔE_N(δ) ≈ (Σ C₁(γⱼ)) δ²
        """
        delta_E_values = []
        
        if self.verbose:
            print(f"  Uniform perturbation analysis: {zero_count} zeros")
        
        for delta in self.delta_values:
            delta_E = self.compute_energy_change(gamma_values, delta, "uniform")
            delta_E_values.append(delta_E)
        
        return {
            'delta': self.delta_values,
            'delta_E': np.array(delta_E_values),
            'dE_d_delta': np.gradient(delta_E_values, self.delta_values),
            'zero_count': zero_count,
            'gamma_values': gamma_values
        }

    def run_random_perturbation_analysis(self, gamma_values, zero_count, random_scale=0.01, n_samples=20):
        """
        Analyze random perturbations: δⱼ ~ N(0, σ²)
        Test quadratic scaling: ΔE ≈ Σ C₁(γⱼ) δⱼ²
        """
        delta_E_samples = []
        sum_delta_squared_samples = []
        
        if self.verbose:
            print(f"  Random perturbation analysis: {zero_count} zeros, {n_samples} samples")
        
        np.random.seed(42)  # Reproducible
        
        for _ in range(n_samples):
            # Generate random perturbations
            random_deltas = np.random.normal(0, random_scale, zero_count)
            
            # Compute energy change
            delta_E = self.compute_energy_change(gamma_values, 0, "random", random_deltas)
            
            # Record data
            delta_E_samples.append(delta_E)
            sum_delta_squared_samples.append(np.sum(random_deltas**2))
        
        return {
            'delta_E_samples': np.array(delta_E_samples),
            'sum_delta_squared': np.array(sum_delta_squared_samples),
            'random_scale': random_scale,
            'n_samples': n_samples,
            'zero_count': zero_count,
            'gamma_values': gamma_values
        }

    def run_analysis(self, batch_config):
        """Run complete analysis for one batch configuration"""
        experiment_type = batch_config['experiment_type']
        zero_count = batch_config['zero_count']
        gamma_values = batch_config['gamma_values']
        perturbation_mode = batch_config['perturbation_mode']
        
        if self.verbose:
            print(f"Running {experiment_type} with {zero_count} zeros")
            print(f"Gamma values: {gamma_values[:3]}... (showing first 3)")
        
        if experiment_type == "multi_zero_uniform":
            return self.run_uniform_perturbation_analysis(gamma_values, zero_count)
        
        elif experiment_type == "multi_zero_random":
            random_scale = batch_config.get('random_scale', 0.01)
            return self.run_random_perturbation_analysis(gamma_values, zero_count, random_scale)
        
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

    def write_to_hdf5(self, results, batch_config, config_name):
        """Write results to HDF5 following design guide structure"""
        
        # Open in append mode, create if doesn't exist
        with h5py.File(self.output_file, 'a') as f:
            # Create group for this configuration
            if config_name in f:
                del f[config_name]  # Remove if exists
            group = f.create_group(config_name)
            
            # Metadata for this configuration
            meta = group.create_group('metadata')
            meta.attrs['description'] = 'Experiment 3: Multi-zero scaling energy functional analysis'
            meta.attrs['experiment_type'] = batch_config['experiment_type']
            meta.attrs['zero_count'] = batch_config['zero_count']
            meta.attrs['perturbation_mode'] = batch_config['perturbation_mode']
            meta.attrs['gamma_values'] = batch_config['gamma_values']
            meta.attrs['delta_range'] = [-self.delta_range, self.delta_range, self.delta_range*2/(self.delta_steps-1)]
            meta.attrs['n_steps'] = self.delta_steps
            meta.attrs['test_function_basis'] = self.test_function_type
            meta.attrs['num_test_functions'] = self.num_test_functions
            
            # Store results based on experiment type
            if batch_config['experiment_type'] == "multi_zero_uniform":
                # Uniform perturbation data
                uniform_group = group.create_group('uniform_perturbation')
                uniform_group.create_dataset('delta', data=results['delta'])
                uniform_group.create_dataset('delta_E', data=results['delta_E'])
                uniform_group.create_dataset('dE_d_delta', data=results['dE_d_delta'])
                uniform_group.attrs['zero_count'] = results['zero_count']
                
            elif batch_config['experiment_type'] == "multi_zero_random":
                # Random perturbation data
                random_group = group.create_group('random_perturbation')
                random_group.create_dataset('delta_E_samples', data=results['delta_E_samples'])
                random_group.create_dataset('sum_delta_squared', data=results['sum_delta_squared'])
                random_group.attrs['random_scale'] = results['random_scale']
                random_group.attrs['n_samples'] = results['n_samples']
                random_group.attrs['zero_count'] = results['zero_count']

def run_experiment3_math(config_file="experiment3_config.json"):
    """Main entry point for Experiment 3 mathematical analysis"""
    math_engine = Experiment3Math(config_file)
    
    # Load individual configuration  
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract configuration details
    experiment_type = config['experiment_type']
    zero_count = config['zero_count']
    perturbation_mode = config['perturbation_mode']
    group_name = config.get('group_name', f"{experiment_type}_{zero_count}zeros_{perturbation_mode}")
    
    print(f"Running {experiment_type} with {zero_count} zeros")
    print(f"Gamma values: {config['gamma_values'][:3]}... (showing first 3)")
    
    # Run analysis
    start_time = time.time()
    results = math_engine.run_analysis(config)
    elapsed = time.time() - start_time
    
    # Save results
    math_engine.write_to_hdf5(results, config, group_name)
    
    print(f"✓ Completed {group_name} in {elapsed:.2f}s")
    
    return math_engine.output_file

# Run directly if called as script
# if __name__ == "__main__":
#     import sys
#     config_file = sys.argv[1] if len(sys.argv) > 1 else 'experiment3_config.json'
#     run_experiment3_math(config_file)
