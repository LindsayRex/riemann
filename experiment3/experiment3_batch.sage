#!/usr/bin/env sage

import json
import sys

# Load modules
load('experiment3_math.sage')
load('experiment3_stats.sage') 
load('experiment3_viz.sage')

# Get config file from command line argument or use default
config_file = sys.argv[1] if len(sys.argv) > 1 else 'experiment3_config.json'

# Load configuration
with open(config_file, 'r') as f:
    config = json.load(f)

batch_configs = config['batch_configs']
base_config = {k: v for k, v in config.items() if k != 'batch_configs'}

print(f"\n=== Experiment 3 Batch Processing ===")
print(f"Config file: {config_file}")
print(f"Running {len(batch_configs)} configurations")
print(f"Output: {config['output_file']}")

# Single HDF5 file for all results
batch_filename = config['output_file']

for i, batch_config in enumerate(batch_configs):
    print(f"\n=== Running configuration {i+1}/{len(batch_configs)} ===")
    
    # Extract config details for display
    exp_type = batch_config['experiment_type']
    zero_count = batch_config['zero_count']
    pert_mode = batch_config['perturbation_mode']
    
    print(f"Type: {exp_type}, N={zero_count}, Mode: {pert_mode}")
    
    # Create current config
    current_config = base_config.copy()
    current_config.update(batch_config)
    config_name = f"config_{i+1}_{exp_type}_{zero_count}zeros_{pert_mode}"
    current_config['output_file'] = batch_filename
    current_config['group_name'] = config_name
    
    # Run pipeline: Math
    with open('temp_config.json', 'w') as f:
        json.dump(current_config, f, indent=2)
    
    run_experiment3_math('temp_config.json')
    print(f"✓ Math completed for {config_name}")

print(f"\n=== Running Statistics on Complete Dataset ===")
run_experiment3_stats(batch_filename)

print(f"\n=== Running Visualization on Complete Dataset ===")
run_experiment3_viz(batch_filename)

print(f"\n=== Batch Complete ===")
print(f"Single output file: {batch_filename}")
print(f"Analyzed {len(batch_configs)} configurations")
