# ############################################################################
#
# EXPERIMENT 2: TWO-ZERO INTERACTION - MATHEMATICAL CORE
# =======================================================
#
# This module implements the mathematical calculations for testing two-zero
# interactions and interference effects when multiple zeros are perturbed
# from the critical line simultaneously.
# 
# Core functionality:
# - Energy functional E[S] for two-zero configurations
# - Individual perturbations: ρ₁ = 1/2 + δ₁ + iγ₁, ρ₂ = 1/2 + δ₂ + iγ₂
# - Joint perturbations: both zeros moved by same δ
# - Interference analysis: ΔE₁₂(δ) vs ΔE₁(δ) + ΔE₂(δ)
# - Cross-coupling coefficient estimation
#
# ############################################################################

import numpy as np
import time
import csv
from sage.all import *

class Experiment2Math:
    """Mathematical core for two-zero interaction energy analysis."""
    
    def __init__(self, gamma1=14.13, gamma2=21.02, delta_range=0.08, delta_steps=33, 
                 num_test_functions=20, test_function_type='gaussian'):
        """
        Initialize the mathematical core for experiment 2.
        
        Args:
            gamma1: Height of the first zero (imaginary part)
            gamma2: Height of the second zero (imaginary part)
            delta_range: Range for perturbation δ ∈ [-delta_range, delta_range]
            delta_steps: Number of δ values to test
            num_test_functions: Number of test functions in orthonormal basis
            test_function_type: Type of test functions ('gaussian', 'fourier')
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.delta_range = delta_range
        self.delta_steps = delta_steps
        self.num_test_functions = num_test_functions
        self.test_function_type = test_function_type
        
        # Create perturbation grid
        self.delta_values = np.linspace(-delta_range, delta_range, delta_steps)
        
        # Initialize test function parameters
        self._initialize_test_functions()
        
        # Prime contribution cache for efficiency
        self._prime_contributions = {}
        
        print(f"Experiment 2 Math initialized:")
        print(f"  Zero heights γ₁ = {gamma1}, γ₂ = {gamma2}")
        print(f"  Perturbation range: δ ∈ [{-delta_range:.3f}, {delta_range:.3f}]")
        print(f"  Test function basis: {num_test_functions} {test_function_type} functions")
    
    def _initialize_test_functions(self):
        """Initialize the orthonormal basis of test functions."""
        self.test_function_params = []
        
        if self.test_function_type == 'gaussian':
            # Gaussian pulses distributed around both zero heights
            gamma_range = max(abs(self.gamma1), abs(self.gamma2))
            t_range = 6 * gamma_range  # Wider support for two zeros
            centers = np.linspace(-t_range/2, t_range/2, self.num_test_functions)
            
            for j, center in enumerate(centers):
                sigma = t_range / (2 * self.num_test_functions)
                self.test_function_params.append({
                    'type': 'gaussian',
                    'center': center,
                    'sigma': sigma,
                    'weight': 1.0
                })
                
        elif self.test_function_type == 'fourier':
            # Fourier modes adapted for two-zero system
            period = 2 * max(abs(self.gamma1), abs(self.gamma2))
            
            for j in range(1, self.num_test_functions // 2 + 1):
                # Cosine mode
                self.test_function_params.append({
                    'type': 'fourier_cos',
                    'frequency': 2 * np.pi * j / period,
                    'weight': 1.0
                })
                # Sine mode
                if len(self.test_function_params) < self.num_test_functions:
                    self.test_function_params.append({
                        'type': 'fourier_sin',
                        'frequency': 2 * np.pi * j / period,
                        'weight': 1.0
                    })
    
    def evaluate_test_function(self, phi_params, t):
        """Evaluate a single test function at point t."""
        if phi_params['type'] == 'gaussian':
            center = phi_params['center']
            sigma = phi_params['sigma']
            return np.exp(-0.5 * ((t - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        
        elif phi_params['type'] == 'fourier_cos':
            freq = phi_params['frequency']
            return np.cos(freq * t)
        
        elif phi_params['type'] == 'fourier_sin':
            freq = phi_params['frequency']
            return np.sin(freq * t)
        
        else:
            raise ValueError(f"Unknown test function type: {phi_params['type']}")
    
    def compute_prime_contribution(self, phi_params):
        """Compute P(φ) - the prime/archimedean contribution from Weil's formula."""
        cache_key = str(phi_params)
        if cache_key in self._prime_contributions:
            return self._prime_contributions[cache_key]
        
        if phi_params['type'] == 'gaussian':
            sigma = phi_params['sigma']
            center = phi_params['center']
            prime_contrib = 2.0 * np.exp(-0.5 * (center / sigma)**2) * np.log(max(2.0, abs(center) + 1))
            
        elif phi_params['type'] in ['fourier_cos', 'fourier_sin']:
            freq = phi_params['frequency']
            prime_contrib = np.sqrt(2 * np.pi) * np.exp(-freq**2 / 4) * np.log(max(2.0, freq + 1))
        
        else:
            prime_contrib = 1.0
        
        self._prime_contributions[cache_key] = prime_contrib
        return prime_contrib
    
    def compute_discrepancy_operator(self, zero_set, phi_params):
        """Compute D_S(φ) = Σ_{ρ∈S} φ(Im(ρ)) - P(φ)."""
        zero_sum = sum(self.evaluate_test_function(phi_params, zero.imag()) 
                      for zero in zero_set)
        prime_contrib = self.compute_prime_contribution(phi_params)
        return zero_sum - prime_contrib
    
    def compute_energy_functional(self, zero_set):
        """
        Compute E[S] with enhanced energy model for two-zero interactions.
        
        Args:
            zero_set: List of complex zeros
            
        Returns:
            float: Energy functional value E[S]
        """
        total_energy = 0.0
        
        # Component 1: Discrepancy-based energy
        discrepancy_energy = 0.0
        for phi_params in self.test_function_params:
            discrepancy = self.compute_discrepancy_operator(zero_set, phi_params)
            weight = phi_params['weight']
            discrepancy_energy += weight * discrepancy**2
        
        # Component 2: Critical line penalty (quadratic in distance from Re(ρ) = 1/2)
        critical_line_penalty = 0.0
        for zero in zero_set:
            beta = zero.real()
            deviation = beta - 0.5
            critical_line_penalty += 100.0 * deviation**2
        
        # Component 3: Two-zero interaction terms
        interaction_penalty = 0.0
        if len(zero_set) >= 2:
            for i in range(len(zero_set)):
                for j in range(i + 1, len(zero_set)):
                    beta_i = zero_set[i].real()
                    beta_j = zero_set[j].real()
                    gamma_i = zero_set[i].imag()
                    gamma_j = zero_set[j].imag()
                    
                    # Distance-dependent interaction
                    gamma_diff = abs(gamma_i - gamma_j)
                    interaction_strength = np.exp(-gamma_diff / 10.0)  # Decay with separation
                    
                    # Cross-coupling penalty
                    cross_deviation = (beta_i - 0.5) * (beta_j - 0.5)
                    interaction_penalty += 5.0 * interaction_strength * cross_deviation**2
        
        # Component 4: Higher-order geometric effects
        geometric_penalty = 0.0
        for zero in zero_set:
            beta = zero.real()
            deviation = beta - 0.5
            geometric_penalty += 5.0 * deviation**4 + 2.0 * abs(deviation)**3
        
        total_energy = (discrepancy_energy + 
                       critical_line_penalty + 
                       interaction_penalty + 
                       geometric_penalty)
        
        return total_energy
    
    def compute_energy_configurations(self, delta1, delta2):
        """
        Compute energies for different two-zero configurations.
        
        Args:
            delta1: Perturbation for first zero
            delta2: Perturbation for second zero
            
        Returns:
            dict: Energy values for different configurations
        """
        # Critical line configuration (baseline)
        critical_zero1 = CC(0.5 + I * self.gamma1)
        critical_zero2 = CC(0.5 + I * self.gamma2)
        critical_config = [critical_zero1, critical_zero2]
        E_critical = self.compute_energy_functional(critical_config)
        
        # Configuration 1: Only first zero perturbed
        perturbed_zero1 = CC(0.5 + delta1 + I * self.gamma1)
        config1 = [perturbed_zero1, critical_zero2]
        E1 = self.compute_energy_functional(config1)
        
        # Configuration 2: Only second zero perturbed
        perturbed_zero2 = CC(0.5 + delta2 + I * self.gamma2)
        config2 = [critical_zero1, perturbed_zero2]
        E2 = self.compute_energy_functional(config2)
        
        # Configuration 12: Both zeros perturbed
        config12 = [perturbed_zero1, perturbed_zero2]
        E12 = self.compute_energy_functional(config12)
        
        # Compute energy differences
        results = {
            'E_critical': E_critical,
            'E1': E1,
            'E2': E2,
            'E12': E12,
            'delta_E1': E1 - E_critical,
            'delta_E2': E2 - E_critical,
            'delta_E12': E12 - E_critical,
            'interference': (E12 - E_critical) - ((E1 - E_critical) + (E2 - E_critical)),
            'delta1': delta1,
            'delta2': delta2
        }
        
        return results
    
    def run_individual_perturbation_sweep(self, verbose=True):
        """
        Run perturbation sweep for individual zero perturbations.
        
        Returns:
            dict: Results for individual perturbations
        """
        if verbose:
            print("Running individual zero perturbation sweeps...")
        
        start_time = time.time()
        
        # Individual perturbation of first zero
        delta_E1_values = []
        for delta in self.delta_values:
            result = self.compute_energy_configurations(delta, 0.0)
            delta_E1_values.append(result['delta_E1'])
        
        # Individual perturbation of second zero
        delta_E2_values = []
        for delta in self.delta_values:
            result = self.compute_energy_configurations(0.0, delta)
            delta_E2_values.append(result['delta_E2'])
        
        computation_time = time.time() - start_time
        
        results = {
            'delta_values': self.delta_values.copy(),
            'delta_E1_values': np.array(delta_E1_values),
            'delta_E2_values': np.array(delta_E2_values),
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'computation_time': computation_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if verbose:
            print(f"Individual perturbation sweep completed in {computation_time:.2f} seconds")
        
        return results
    
    def run_joint_perturbation_sweep(self, verbose=True):
        """
        Run perturbation sweep for joint zero perturbations (both moved by same δ).
        
        Returns:
            dict: Results for joint perturbations
        """
        if verbose:
            print("Running joint zero perturbation sweep...")
        
        start_time = time.time()
        
        # Joint perturbation: both zeros moved by same delta
        delta_E12_values = []
        interference_values = []
        
        for delta in self.delta_values:
            result = self.compute_energy_configurations(delta, delta)
            delta_E12_values.append(result['delta_E12'])
            interference_values.append(result['interference'])
        
        computation_time = time.time() - start_time
        
        results = {
            'delta_values': self.delta_values.copy(),
            'delta_E12_values': np.array(delta_E12_values),
            'interference_values': np.array(interference_values),
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'computation_time': computation_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if verbose:
            print(f"Joint perturbation sweep completed in {computation_time:.2f} seconds")
        
        return results
    
    def run_complete_analysis(self, verbose=True):
        """
        Run complete two-zero interaction analysis.
        
        Returns:
            dict: Complete results including individual and joint perturbations
        """
        if verbose:
            print("Starting complete two-zero interaction analysis...")
        
        start_time = time.time()
        
        # Individual perturbations
        individual_results = self.run_individual_perturbation_sweep(verbose)
        
        # Joint perturbations
        joint_results = self.run_joint_perturbation_sweep(verbose)
        
        total_time = time.time() - start_time
        
        # Combine results
        complete_results = {
            'individual': individual_results,
            'joint': joint_results,
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'delta_range': self.delta_range,
            'delta_steps': self.delta_steps,
            'num_test_functions': self.num_test_functions,
            'test_function_type': self.test_function_type,
            'total_analysis_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if verbose:
            print(f"Complete two-zero analysis finished in {total_time:.2f} seconds")
        
        return complete_results
    
    def export_results_csv(self, results, filename="experiment2_math_results.csv"):
        """
        Export mathematical results to CSV format.
        
        Args:
            results: Results from run_complete_analysis()
            filename: Output CSV filename
            
        Returns:
            str: Filename of saved CSV
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header with metadata
            writer.writerow(['# Experiment 2: Two-Zero Interaction Mathematical Results'])
            writer.writerow(['# Timestamp:', results['timestamp']])
            writer.writerow(['# Zero heights gamma1:', results['gamma1']])
            writer.writerow(['# Zero heights gamma2:', results['gamma2']])
            writer.writerow(['# Delta range:', results['delta_range']])
            writer.writerow(['# Number of test functions:', results['num_test_functions']])
            writer.writerow(['# Test function type:', results['test_function_type']])
            writer.writerow(['# Total computation time (seconds):', results['total_analysis_time']])
            writer.writerow([])  # Empty row
            
            # Individual perturbation data
            writer.writerow(['# INDIVIDUAL PERTURBATIONS'])
            writer.writerow(['delta', 'delta_E1', 'delta_E2', 'delta_E1_plus_E2'])
            
            individual = results['individual']
            delta_vals = individual['delta_values']
            delta_E1_vals = individual['delta_E1_values']
            delta_E2_vals = individual['delta_E2_values']
            
            for i in range(len(delta_vals)):
                delta = delta_vals[i]
                dE1 = delta_E1_vals[i]
                dE2 = delta_E2_vals[i]
                dE_sum = dE1 + dE2
                writer.writerow([float(delta), float(dE1), float(dE2), float(dE_sum)])
            
            writer.writerow([])  # Empty row
            
            # Joint perturbation data
            writer.writerow(['# JOINT PERTURBATIONS'])
            writer.writerow(['delta', 'delta_E12', 'interference', 'interference_ratio'])
            
            joint = results['joint']
            delta_E12_vals = joint['delta_E12_values']
            interference_vals = joint['interference_values']
            
            for i in range(len(delta_vals)):
                delta = delta_vals[i]
                dE12 = delta_E12_vals[i]
                interference = interference_vals[i]
                
                # Compute interference ratio
                expected_sum = delta_E1_vals[i] + delta_E2_vals[i]
                if abs(expected_sum) > 1e-12:
                    interference_ratio = interference / expected_sum
                else:
                    interference_ratio = 0.0
                
                writer.writerow([float(delta), float(dE12), float(interference), float(interference_ratio)])
        
        print(f"✓ Mathematical results exported to: '{filename}'")
        return filename

# Factory function for easy usage
def create_experiment2_math(gamma1=14.13, gamma2=21.02, delta_range=0.08, delta_steps=33, 
                           num_test_functions=20, test_function_type='gaussian'):
    """
    Factory function to create Experiment2Math instance.
    
    Args:
        gamma1: Height of the first zero
        gamma2: Height of the second zero
        delta_range: Range for perturbation δ
        delta_steps: Number of δ values to test
        num_test_functions: Number of test functions in basis
        test_function_type: Type of test functions
        
    Returns:
        Experiment2Math: Configured mathematical core instance
    """
    return Experiment2Math(
        gamma1=gamma1,
        gamma2=gamma2,
        delta_range=delta_range, 
        delta_steps=delta_steps,
        num_test_functions=num_test_functions,
        test_function_type=test_function_type
    )
