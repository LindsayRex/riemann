#!/usr/bin/env sage

import json
import h5py
import numpy as np

# Load modules
load('experiment2_math.sage')
load('experiment2_stats.sage') 
load('experiment2_viz.sage')

def run_single_hdf5_batch():
    """Run all configurations and store in ONE HDF5 file"""
    
    # Load configuration
    with open('experiment2_config.json', 'r') as f:
        config = json.load(f)

    batch_configs = config['batch_configs']
    base_config = {k: v for k, v in config.items() if k != 'batch_configs'}
    
    # Single HDF5 file for everything
    hdf5_file = "experiment2_complete_dataset.h5"
    
    print(f"\n=== Experiment 2 Complete Dataset ===")
    print(f"Running {len(batch_configs)} configurations")
    print(f"Output: {hdf5_file}")
    print()

    # Process all configurations into single file
    for i, batch_config in enumerate(batch_configs):
        gamma1, gamma2 = batch_config['gamma1'], batch_config['gamma2']
        print(f"=== Configuration {i+1}/{len(batch_configs)}: γ₁={gamma1}, γ₂={gamma2} ===")
        
        # Create current config
        current_config = base_config.copy()
        current_config.update(batch_config)
        current_config['output_file'] = hdf5_file
        current_config['config_group'] = f"config_{i+1:02d}_gamma1_{gamma1}_gamma2_{gamma2}"
        
        # Run math computation for this config
        run_experiment2_math(current_config)
        print(f"✓ Math results added to {hdf5_file}")

    # Run statistics on complete dataset
    print(f"\n=== Complete Dataset Statistics ===")
    run_experiment2_stats(hdf5_file)
    print(f"✓ Statistics computed for complete dataset")
    
    # Generate visualizations from complete dataset
    print(f"\n=== Complete Dataset Visualization ===")
    run_experiment2_viz(hdf5_file)
    print(f"✓ Visualizations generated from complete dataset")
    
    print(f"\n✅ COMPLETE: All data in {hdf5_file}")
    
    # Summary
    with h5py.File(hdf5_file, 'r') as f:
        config_groups = [k for k in f.keys() if k.startswith('config_')]
        print(f"   {len(config_groups)} configurations")
        print(f"   Complete statistical analysis")
        print(f"   Summary visualizations")

if __name__ == "__main__":
    run_single_hdf5_batch()
