#!/usr/bin/env sage

import json

# Load modules
load('experiment2_math.sage')
load('experiment2_stats.sage') 
load('experiment2_viz.sage')

# Load configuration
with open('experiment2_config.json', 'r') as f:
    config = json.load(f)

batch_configs = config['batch_configs']
base_config = {k: v for k, v in config.items() if k != 'batch_configs'}

print(f"\n=== Experiment 2 Batch Processing ===")
print(f"Running {len(batch_configs)} configurations")

results = []

for i, batch_config in enumerate(batch_configs):
    print(f"\n=== Running configuration {i+1}/{len(batch_configs)} ===")
    gamma1, gamma2 = batch_config['gamma1'], batch_config['gamma2']
    print(f"γ₁={gamma1}, γ₂={gamma2}")
    
    # Create current config
    current_config = base_config.copy()
    current_config.update(batch_config)
    filename = f"data/experiment2_gamma1_{gamma1}_gamma2_{gamma2}.h5"
    current_config['output_file'] = filename
    
    # Run pipeline: Math → Stats → Viz
    with open('temp_config.json', 'w') as f:
        json.dump(current_config, f, indent=2)
    
    run_experiment2_math('temp_config.json')
    run_experiment2_stats(filename)
    run_experiment2_viz(filename)
    
    results.append(filename)
    print(f"✓ Completed: {filename}")

print(f"\n=== Batch Complete ===")
print(f"Generated {len(results)} HDF5 files:")
for result in results:
    print(f"  - {result}")
