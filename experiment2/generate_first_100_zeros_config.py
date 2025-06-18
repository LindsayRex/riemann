#!/usr/bin/env python3
"""
Generate configuration for first 100 Riemann zeros analysis
Creates all adjacent pairs γₙ, γₙ₊₁ for comprehensive statistical power
"""

import json
import math

def riemann_zero_approximation(n):
    """
    Approximation for the nth non-trivial zero of Riemann zeta function
    Using the asymptotic formula: γₙ ≈ 2πn/log(n/2πe) for large n
    For small n, use known exact values
    """
    # First 20 zeros (exact values)
    first_20_zeros = [
        14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
        37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
        52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
        67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069
    ]
    
    if n <= 20:
        return first_20_zeros[n-1]
    
    # Asymptotic approximation for n > 20
    # More accurate formula: γₙ ≈ 2π * n / log(n/(2πe)) + corrections
    x = n / (2 * math.pi * math.e)
    log_term = math.log(n / (2 * math.pi * math.e))
    
    # Main term
    gamma_n = 2 * math.pi * n / log_term
    
    # First-order correction
    correction = 2 * math.pi * math.log(log_term) / (log_term**2)
    
    return gamma_n - correction

def generate_first_100_zeros_config():
    """Generate configuration for all adjacent pairs from first 100 zeros"""
    
    # Generate first 100 zeros
    zeros = [riemann_zero_approximation(n) for n in range(1, 101)]
    
    # Create all adjacent pairs: (γₙ, γₙ₊₁) for n = 1..99
    batch_configs = []
    for i in range(99):  # 0 to 98, giving us pairs (γ₁,γ₂), (γ₂,γ₃), ..., (γ₉₉,γ₁₀₀)
        gamma1 = round(zeros[i], 3)
        gamma2 = round(zeros[i+1], 3)
        batch_configs.append({
            "gamma1": gamma1,
            "gamma2": gamma2
        })
    
    # Base configuration
    config = {
        "description": "First 100 Riemann zeros - adjacent pairs analysis for maximum statistical power",
        "N": 65536,
        "zeros_to_test": 50,
        "delta_values": [-0.1, -0.05, 0.0, 0.05, 0.1],
        "test_function_basis": "daubechies",
        "k": 6,
        "cores": 16,
        "bootstrap_samples": 1000,
        "confidence_level": 0.95,
        "batch_configs": batch_configs
    }
    
    return config

if __name__ == "__main__":
    config = generate_first_100_zeros_config()
    
    # Save configuration
    with open('experiment2_config_first_100_zeros.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Generated configuration with {len(config['batch_configs'])} adjacent zero pairs")
    print(f"✓ Coverage: γ₁ = {config['batch_configs'][0]['gamma1']:.3f} to γ₁₀₀ = {config['batch_configs'][-1]['gamma2']:.3f}")
    print(f"✓ Total configurations: {len(config['batch_configs'])}")
    print(f"✓ Saved to: experiment2_config_first_100_zeros.json")
    
    # Show first few and last few pairs
    print(f"\nFirst 5 pairs:")
    for i in range(5):
        cfg = config['batch_configs'][i]
        print(f"  γ_{i+1}, γ_{i+2} = ({cfg['gamma1']:.3f}, {cfg['gamma2']:.3f})")
    
    print(f"\nLast 5 pairs:")
    for i in range(-5, 0):
        cfg = config['batch_configs'][i]
        n = len(config['batch_configs']) + i + 1
        print(f"  γ_{n}, γ_{n+1} = ({cfg['gamma1']:.3f}, {cfg['gamma2']:.3f})")
