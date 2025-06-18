#!/usr/bin/env sage

import json
import h5py

# Load modules
load('experiment2_math.sage')
load('experiment2_stats.sage') 
load('experiment2_viz.sage')

def run_batch_experiment(config_file="experiment2_config.json"):
    """Run all configurations in a single HDF5 file"""
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    batch_configs = config['batch_configs']
    base_config = {k: v for k, v in config.items() if k != 'batch_configs'}
    
    # Single HDF5 file for all results
    output_file = "data/experiment2_batch_results.h5"
    
    print(f"\n=== Experiment 2 Batch Processing ===")
    print(f"Running {len(batch_configs)} configurations")
    print(f"Output file: {output_file}")
    
    # Process all configurations into one file
    for i, batch_config in enumerate(batch_configs):
        print(f"\n=== Configuration {i+1}/{len(batch_configs)} ===")
        gamma1, gamma2 = batch_config['gamma1'], batch_config['gamma2']
        print(f"γ₁={gamma1}, γ₂={gamma2}")
        
        # Create current config
        current_config = base_config.copy()
        current_config.update(batch_config)
        
        # Run math for this configuration
        group_name = f"config_{gamma1:.2f}_{gamma2:.2f}"
        print(f"  Running math → {group_name}")
        run_experiment2_math(current_config, output_file, group_name)
    
    print(f"\n=== Running Statistics on Complete Dataset ===")
    print(f"Processing {output_file}")
    run_experiment2_batch_stats(output_file)
    
    print(f"\n=== Creating Summary Visualization ===")
    run_batch_visualization_single_file(output_file)
    
    print(f"\n✅ Batch Complete!")
    print(f"  Results: {output_file}")
    print(f"  Processed {len(batch_configs)} configurations")

if __name__ == "__main__":
    run_batch_experiment()
