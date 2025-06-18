#!/usr/bin/env sage

import json
import sys

# Load modules
load('experiment2_math.sage')
load('experiment2_stats.sage') 
load('experiment2_viz.sage')

# Get config file from command line argument or use default
config_file = sys.argv[1] if len(sys.argv) > 1 else 'experiment2_config.json'

# Load configuration
with open(config_file, 'r') as f:
    config = json.load(f)

batch_configs = config['batch_configs']
base_config = {k: v for k, v in config.items() if k != 'batch_configs'}

print(f"\n=== Experiment 2 Batch Processing ===")
print(f"Config file: {config_file}")
print(f"Running {len(batch_configs)} configurations")
print(f"Output: experiment2_complete_analysis.h5")

# Single HDF5 file for all results
batch_filename = "data/experiment2_complete_analysis.h5"

for i, batch_config in enumerate(batch_configs):
    print(f"\n=== Running configuration {i+1}/{len(batch_configs)} ===")
    gamma1, gamma2 = batch_config['gamma1'], batch_config['gamma2']
    print(f"γ₁={gamma1}, γ₂={gamma2}")
    
    # Create current config
    current_config = base_config.copy()
    current_config.update(batch_config)
    config_name = f"config_{i+1}_gamma1_{gamma1}_gamma2_{gamma2}"
    current_config['output_file'] = batch_filename
    current_config['group_name'] = config_name
    
    # Run pipeline: Math → Stats
    with open('temp_config.json', 'w') as f:
        json.dump(current_config, f, indent=2)
    
    run_experiment2_math('temp_config.json')
    print(f"✓ Math completed for {config_name}")

print(f"\n=== Running Statistics on Complete Dataset ===")
run_experiment2_stats(batch_filename)

print(f"\n=== Running Visualization on Complete Dataset ===")
run_experiment2_viz(batch_filename)

print(f"\n=== Generating Summary Report ===")
create_experiment2_summary_report(batch_filename)

print(f"\n=== Batch Complete ===")
print(f"Single output file: {batch_filename}")
print(f"Analyzed {len(batch_configs)} configurations")
