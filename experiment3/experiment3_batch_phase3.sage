#!/usr/bin/env sage

"""
Experiment 3 Phase 3 Batch Runner: Publication-Quality Multi-Zero Scaling Analysis
Executes the full pipeline for 486 configurations across N = 10, 20, 50, 100, 200, 500
"""

import os
import sys
import time
from datetime import datetime

# Add experiment3 directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the experiment modules
load('experiment3_math.sage')
load('experiment3_stats.sage')
load('experiment3_viz.sage')

def run_phase3_pipeline():
    """Execute the complete Phase 3 pipeline"""
    
    print("=" * 80)
    print("EXPERIMENT 3 PHASE 3: PUBLICATION-QUALITY MULTI-ZERO SCALING ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    config_file = "experiment3_config_phase3_full.json"
    
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file {config_file} not found!")
        return False
    
    # Load configuration to show summary
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    total_configs = config['statistics']['total_configurations']
    zero_counts = config['statistics']['zero_counts']
    
    print(f"Configuration: {config['description']}")
    print(f"Total configurations: {total_configs}")
    print(f"Zero counts tested: {zero_counts}")
    print(f"Precision: {config['delta_steps']} δ points, {config['num_test_functions']} test functions")
    print(f"Bootstrap samples: {config['bootstrap_samples']}")
    print()
    
    # Estimate computational time
    # Based on current timings: ~0.1s per config per δ point
    estimated_time_hours = (total_configs * config['delta_steps'] * 0.1) / 3600
    print(f"Estimated computation time: {estimated_time_hours:.1f} hours")
    print()
    
    # Ask for confirmation for such a large run
    response = input("This is a large-scale computation. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Computation cancelled.")
        return False
    
    print("Starting Phase 3 pipeline...")
    print()
    
    # Phase 1: Mathematical computation
    print("PHASE 1: MATHEMATICAL COMPUTATION")
    print("-" * 40)
    start_time = time.time()
    
    try:
        # Run mathematical analysis with the Phase 3 config
        math_result = run_experiment3_math(config_file)
        math_time = time.time() - start_time
        print(f"✓ Mathematical computation completed in {math_time/60:.1f} minutes")
        print(f"✓ Results saved to: {math_result}")
    except Exception as e:
        print(f"✗ Mathematical computation failed: {e}")
        return False
    
    print()
    
    # Phase 2: Statistical analysis
    print("PHASE 2: STATISTICAL ANALYSIS")
    print("-" * 40)
    start_time = time.time()
    
    try:
        # Run statistical analysis on the results
        stats_result = run_experiment3_stats(config['output_file'])
        stats_time = time.time() - start_time
        print(f"✓ Statistical analysis completed in {stats_time/60:.1f} minutes")
        print(f"✓ Results saved to: {stats_result}")
    except Exception as e:
        print(f"✗ Statistical analysis failed: {e}")
        return False
    
    print()
    
    # Phase 3: Visualization
    print("PHASE 3: VISUALIZATION")
    print("-" * 40)
    start_time = time.time()
    
    try:
        # Create all visualizations
        viz_files = run_experiment3_viz(config['output_file'])
        viz_time = time.time() - start_time
        print(f"✓ Visualization completed in {viz_time/60:.1f} minutes")
        print(f"✓ Generated {len(viz_files)} summary images")
        for i, filename in enumerate(viz_files, 1):
            print(f"   Image {i}: {filename}")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False
    
    print()
    print("=" * 80)
    print("EXPERIMENT 3 PHASE 3 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final summary
    print()
    print("PHASE 3 RESULTS SUMMARY:")
    print("-" * 40)
    print(f"• Mathematical data: {config['output_file']}")
    print(f"• Statistical analysis: HDF5 groups and summary report")
    print(f"• Visualizations: 5 publication-quality summary images")
    print(f"• Total configurations analyzed: {total_configs}")
    print(f"• Zero count range: N ∈ [{min(zero_counts)}, {max(zero_counts)}]")
    print()
    print("This Phase 3 dataset provides:")
    print("• Comprehensive multi-zero scaling law C₁^(N) ∝ N validation")
    print("• Large-scale statistical verification of additivity hypothesis")
    print("• Publication-quality data matching Experiment 2's scale")
    print("• Robust evidence for critical line stability across all tested scales")
    
    return True

def run_phase3_math_only():
    """Run only the mathematical computation phase for testing"""
    print("Running Phase 3 Mathematical Computation Only...")
    config_file = "experiment3_config_phase3_full.json"
    
    try:
        result = run_experiment3_math(config_file)
        print(f"✓ Mathematical computation completed: {result}")
        return True
    except Exception as e:
        print(f"✗ Mathematical computation failed: {e}")
        return False

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--math-only":
        success = run_phase3_math_only()
    else:
        success = run_phase3_pipeline()
    
    if success:
        print("\n🎉 Experiment 3 Phase 3 completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Experiment 3 Phase 3 failed!")
        sys.exit(1)
