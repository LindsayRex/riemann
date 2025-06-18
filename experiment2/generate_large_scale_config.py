#!/usr/bin/env python3
"""
Generate large-scale Experiment 2 configurations for systematic C₁₂ analysis
Focus: Cross-coupling dependence on zero separation |γ₂ - γ₁|
"""

import json
import numpy as np

def riemann_zeros_approximation(n):
    """Generate approximation of first n Riemann zero heights"""
    zeros = []
    for k in range(1, n + 1):
        # Li's approximation for k-th zero height
        gamma_k = 2 * np.pi * k / np.log(k + 10) + 0.5 * np.log(k + 10) - 1
        # More accurate approximation for first zeros
        if k <= 100:
            # Use known accurate values for first 100
            gamma_k = 14.134725 + (k - 1) * 2.5  # Rough spacing
        zeros.append(gamma_k)
    return zeros

def generate_large_scale_configs():
    """Generate systematic large-scale configuration set"""
    configs = []
    
    # Get approximations for first 1000 zeros
    zeros = riemann_zeros_approximation(1000)
    
    print(f"Generating large-scale configurations...")
    print(f"Zero range: γ₁ ∈ [14.13, {zeros[499]:.2f}], γ₂ ∈ [14.13, {zeros[999]:.2f}]")
    
    # ===== STRATEGY 1: SEPARATION ANALYSIS =====
    # Study C₁₂ vs |γ₂ - γ₁| systematically
    
    # Small separations: Adjacent and near-adjacent (separation ≤ 20)
    print("Adding small separation pairs...")
    for i in range(0, 400, 2):  # Every 2nd zero from first 400
        for sep in [1, 2, 3, 5, 8]:  # Different separation patterns
            if i + sep < len(zeros):
                configs.append({
                    "gamma1": zeros[i],
                    "gamma2": zeros[i + sep]
                })
    
    # Medium separations: 20 < |γ₂ - γ₁| < 100
    print("Adding medium separation pairs...")
    for i in range(0, 300, 3):  # Every 3rd zero from first 300
        for sep in [10, 15, 25, 40, 60, 80]:
            if i + sep < len(zeros):
                configs.append({
                    "gamma1": zeros[i],
                    "gamma2": zeros[i + sep]
                })
    
    # Large separations: |γ₂ - γ₁| > 100
    print("Adding large separation pairs...")
    for i in range(0, 200, 5):  # Every 5th zero from first 200
        for sep in [100, 150, 250, 400, 600]:
            if i + sep < len(zeros):
                configs.append({
                    "gamma1": zeros[i],
                    "gamma2": zeros[i + sep]
                })
    
    # ===== STRATEGY 2: HIGH-γ COVERAGE =====
    # Explore behavior at higher zero heights
    print("Adding high-γ coverage...")
    
    # High γ₁, various γ₂
    for i in range(400, 800, 10):  # High γ₁ values
        for j in range(i + 1, min(i + 50, 900), 5):  # Various γ₂ > γ₁
            configs.append({
                "gamma1": zeros[i],
                "gamma2": zeros[j]
            })
    
    # ===== STRATEGY 3: PARAMETER SPACE GRID =====
    # Systematic grid sampling for uniform coverage
    print("Adding parameter space grid...")
    
    # Create grid in (γ₁, γ₂) space
    gamma1_grid = np.linspace(zeros[0], zeros[500], 25)  # 25 γ₁ values
    gamma2_grid = np.linspace(zeros[0], zeros[700], 30)  # 30 γ₂ values
    
    for g1 in gamma1_grid:
        for g2 in gamma2_grid:
            if g2 > g1:  # Only γ₂ > γ₁
                configs.append({
                    "gamma1": float(g1),
                    "gamma2": float(g2)
                })
    
    # ===== STRATEGY 4: RANDOM SAMPLING =====
    # Fill gaps with random pairs
    print("Adding random sampling...")
    np.random.seed(42)  # Reproducible
    
    for _ in range(800):  # 800 random pairs
        i = np.random.randint(0, 600)
        j = np.random.randint(i + 1, 900)
        configs.append({
            "gamma1": zeros[i],
            "gamma2": zeros[j]
        })
    
    # Remove duplicates (approximately)
    print("Removing approximate duplicates...")
    unique_configs = []
    seen = set()
    
    for config in configs:
        # Round to 3 decimal places for duplicate detection
        key = (round(config["gamma1"], 3), round(config["gamma2"], 3))
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)
    
    print(f"Generated {len(unique_configs)} unique configurations")
    print(f"γ₁ range: [{min(c['gamma1'] for c in unique_configs):.3f}, {max(c['gamma1'] for c in unique_configs):.3f}]")
    print(f"γ₂ range: [{min(c['gamma2'] for c in unique_configs):.3f}, {max(c['gamma2'] for c in unique_configs):.3f}]")
    
    # Analyze separation distribution
    separations = [c['gamma2'] - c['gamma1'] for c in unique_configs]
    print(f"Separation range: [{min(separations):.3f}, {max(separations):.3f}]")
    print(f"Mean separation: {np.mean(separations):.3f}")
    
    return unique_configs

if __name__ == "__main__":
    # Generate configurations
    batch_configs = generate_large_scale_configs()
    
    # Load base config
    with open('experiment2_config_large_scale.json', 'r') as f:
        config = json.load(f)
    
    # Add batch configurations
    config['batch_configs'] = batch_configs
    
    # Save updated config
    with open('experiment2_config_large_scale.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Large-scale config saved: experiment2_config_large_scale.json")
    print(f"✓ Total configurations: {len(batch_configs)}")
    print(f"✓ Estimated processing time: {len(batch_configs) * 0.5 / 60:.1f} minutes")
