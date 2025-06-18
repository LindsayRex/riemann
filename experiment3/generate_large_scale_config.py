#!/usr/bin/env python3
"""
Generate large-scale Experiment 3 configurations for multi-zero scaling analysis
Focus: Systematic study of scaling law C₁^(N) vs N across many zero counts
"""

import json
import math

def riemann_zeros_first_n(n):
    """Return first n Riemann zero heights (approximate)"""
    # First 20 accurate zeros
    accurate_zeros = [
        14.134725, 21.022040, 25.010857, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        52.970321, 56.446247, 59.347044, 60.831778, 65.112544,
        67.079810, 69.546401, 72.067158, 75.704690, 77.144840
    ]
    
    if n <= 20:
        return accurate_zeros[:n]
    
    # Extend with approximation for n > 20
    zeros = accurate_zeros.copy()
    
    # Approximate spacing around ~2.5 for higher zeros
    for k in range(21, n + 1):
        # Li's approximation with corrections
        gamma_k = 2 * math.pi * k / math.log(k + 10) + 0.5 * math.log(k + 10) - 1
        zeros.append(gamma_k)
    
    return zeros[:n]

def generate_scaling_configs():
    """Generate systematic scaling analysis configurations"""
    configs = []
    
    print("Generating Experiment 3 large-scale configurations...")
    
    # Get first 50 zero heights
    zeros = riemann_zeros_first_n(50)
    print(f"Zero range: γ ∈ [{zeros[0]:.3f}, {zeros[-1]:.3f}]")
    
    # ===== UNIFORM PERTURBATION SCALING =====
    print("Adding uniform perturbation configurations...")
    
    # Test scaling law: N = 2, 3, 4, 5, 6, 8, 10, 12, 15, 20
    zero_counts = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    
    for N in zero_counts:
        if N <= len(zeros):
            configs.append({
                "experiment_type": "multi_zero_uniform",
                "zero_count": N,
                "gamma_values": zeros[:N],
                "perturbation_mode": "uniform"
            })
    
    # ===== RANDOM PERTURBATION VALIDATION =====
    print("Adding random perturbation configurations...")
    
    # Test random perturbations for key N values
    random_counts = [3, 5, 8, 10, 15]
    
    for N in random_counts:
        if N <= len(zeros):
            configs.append({
                "experiment_type": "multi_zero_random",
                "zero_count": N,
                "gamma_values": zeros[:N],
                "perturbation_mode": "random",
                "random_seed": 42,
                "random_scale": 0.008  # Smaller for larger N
            })
    
    # ===== DIFFERENT ZERO SELECTIONS =====
    print("Adding different zero selection patterns...")
    
    # Every 2nd zero (test spacing effects)
    configs.append({
        "experiment_type": "multi_zero_uniform",
        "zero_count": 10,
        "gamma_values": [zeros[2*i] for i in range(10)],  # 0, 2, 4, 6, ...
        "perturbation_mode": "uniform"
    })
    
    # High zeros only
    configs.append({
        "experiment_type": "multi_zero_uniform", 
        "zero_count": 10,
        "gamma_values": zeros[20:30],  # Zeros 21-30
        "perturbation_mode": "uniform"
    })
    
    # Mixed low and high
    mixed_indices = [0, 1, 2, 10, 11, 12, 20, 21, 22, 30]
    configs.append({
        "experiment_type": "multi_zero_uniform",
        "zero_count": 10,
        "gamma_values": [zeros[i] for i in mixed_indices if i < len(zeros)],
        "perturbation_mode": "uniform"
    })
    
    print(f"Generated {len(configs)} configurations")
    return configs

def create_large_scale_config():
    """Create large-scale configuration file"""
    
    # Generate configurations
    batch_configs = generate_scaling_configs()
    
    # Base configuration
    config = {
        "description": "Experiment 3: Large-Scale Multi-Zero Scaling Analysis",
        "delta_range": 0.04,
        "delta_steps": 33,
        "test_function_type": "gaussian",
        "num_test_functions": 20,
        "confidence_level": 0.95,
        "bootstrap_samples": 1000,
        "output_file": "data/experiment3_large_scale_analysis.h5",
        "verbose": True,
        "batch_configs": batch_configs
    }
    
    # Write to file
    output_file = "experiment3_config_large_scale.json"
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nLarge-scale configuration saved to: {output_file}")
    print(f"Total configurations: {len(batch_configs)}")
    
    # Print summary
    uniform_configs = [c for c in batch_configs if c['experiment_type'] == 'multi_zero_uniform']
    random_configs = [c for c in batch_configs if c['experiment_type'] == 'multi_zero_random']
    
    print(f"Uniform perturbations: {len(uniform_configs)}")
    print(f"Random perturbations: {len(random_configs)}")
    
    if uniform_configs:
        zero_counts = [c['zero_count'] for c in uniform_configs]
        print(f"Zero count range: N ∈ [{min(zero_counts)}, {max(zero_counts)}]")
    
    print(f"\nConfiguration parameters:")
    print(f"  - Delta range: ±{config['delta_range']}")
    print(f"  - Delta steps: {config['delta_steps']}")
    print(f"  - Test functions: {config['num_test_functions']}")
    print(f"  - Bootstrap samples: {config['bootstrap_samples']}")
    
    return output_file

if __name__ == "__main__":
    create_large_scale_config()
