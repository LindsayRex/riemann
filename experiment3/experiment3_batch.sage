#!/usr/bin/env sage

import json
import sys
import time

# Load modules
load('experiment3_math.sage')
load('experiment3_stats.sage') 
load('experiment3_viz.sage')

def run_experiment3_batch(config_file="experiment3_config.json"):
    """
    Experiment 3 Batch Orchestrator
    
    Pipeline: Math → Stats → Viz
    Tests multi-zero scaling behavior and additivity
    """
    
    print(f"\n" + "="*70)
    print(f"EXPERIMENT 3: MULTI-ZERO SCALING ANALYSIS")
    print(f"="*70)
    
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found!")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return False
    
    batch_configs = config['batch_configs']
    
    print(f"Configuration: {config_file}")
    print(f"Batch size: {len(batch_configs)} configurations")
    print(f"Output file: {config['output_file']}")
    print(f"Description: {config.get('description', 'Multi-Zero Scaling Analysis')}")
    
    # Display experimental scope
    uniform_configs = [c for c in batch_configs if c['experiment_type'] == 'multi_zero_uniform']
    random_configs = [c for c in batch_configs if c['experiment_type'] == 'multi_zero_random']
    
    if uniform_configs:
        zero_counts_uniform = [c['zero_count'] for c in uniform_configs]
        print(f"Uniform perturbations: N ∈ {sorted(set(zero_counts_uniform))}")
    
    if random_configs:
        zero_counts_random = [c['zero_count'] for c in random_configs]
        print(f"Random perturbations: N ∈ {sorted(set(zero_counts_random))}")
    
    print(f"Delta range: ±{config['delta_range']} ({config['delta_steps']} steps)")
    print(f"Test functions: {config['num_test_functions']} {config['test_function_type']}")
    
    # ===== MATHEMATICAL ANALYSIS =====
    print(f"\n" + "-"*50)
    print(f"PHASE 1: MATHEMATICAL ANALYSIS")
    print(f"-"*50)
    
    start_time = time.time()
    
    try:
        output_file = run_experiment3_math(config_file)
        math_time = time.time() - start_time
        print(f"✓ Mathematical analysis completed in {math_time:.1f}s")
        print(f"  Data saved to: {output_file}")
    except Exception as e:
        print(f"✗ Mathematical analysis failed: {e}")
        return False
    
    # ===== STATISTICAL ANALYSIS =====
    print(f"\n" + "-"*50)
    print(f"PHASE 2: STATISTICAL ANALYSIS")
    print(f"-"*50)
    
    start_time = time.time()
    
    try:
        run_experiment3_stats(output_file)
        stats_time = time.time() - start_time
        print(f"✓ Statistical analysis completed in {stats_time:.1f}s")
    except Exception as e:
        print(f"✗ Statistical analysis failed: {e}")
        return False
    
    # ===== VISUALIZATION =====
    print(f"\n" + "-"*50)
    print(f"PHASE 3: VISUALIZATION")
    print(f"-"*50)
    
    start_time = time.time()
    
    try:
        summary_files = run_experiment3_viz(output_file)
        viz_time = time.time() - start_time
        print(f"✓ Visualizations completed in {viz_time:.1f}s")
        
        print(f"\nGenerated visualizations:")
        for i, filename in enumerate(summary_files, 1):
            print(f"  {i}. {filename}")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False
    
    # ===== COMPLETION SUMMARY =====
    total_time = math_time + stats_time + viz_time
    
    print(f"\n" + "="*70)
    print(f"EXPERIMENT 3 COMPLETED SUCCESSFULLY")
    print(f"="*70)
    print(f"Total runtime: {total_time:.1f}s")
    print(f"  - Mathematical analysis: {math_time:.1f}s")
    print(f"  - Statistical analysis: {stats_time:.1f}s") 
    print(f"  - Visualization: {viz_time:.1f}s")
    print(f"\nOutputs:")
    print(f"  - Data: {output_file}")
    print(f"  - Report: results/experiment3_summary_report.txt")
    print(f"  - Images: {len(summary_files)} summary visualizations")
    
    # Mathematical insights summary
    print(f"\nKey Findings:")
    try:
        # Read some key results for quick summary
        import h5py
        with h5py.File(output_file, 'r') as f:
            if 'scaling_analysis' in f:
                scaling = f['scaling_analysis']
                slope = scaling.attrs['slope']
                slope_p = scaling.attrs['slope_p_value']
                print(f"  - Scaling law: C₁^(N) ∝ N with slope {slope:.2e}")
                print(f"  - Statistical significance: p = {slope_p:.3e}")
            
            # Count stable configurations
            config_groups = [key for key in f.keys() if key.startswith('config_')]
            uniform_groups = [g for g in config_groups if 'uniform' in g]
            
            if uniform_groups:
                stable_count = 0
                total_count = 0
                for group_name in uniform_groups:
                    if 'statistical_analysis' in f[group_name]:
                        stats = f[group_name]['statistical_analysis']
                        c1 = stats['polyfit_coeffs'][0]
                        if c1 > 0:
                            stable_count += 1
                        total_count += 1
                
                if total_count > 0:
                    print(f"  - Stability: {stable_count}/{total_count} configurations have C₁ > 0")
                    print(f"  - Success rate: {100*stable_count/total_count:.1f}%")
    except:
        print(f"  - Analysis complete (see detailed report)")
    
    print(f"\nExperiment 3 pipeline executed successfully!")
    return True

if __name__ == "__main__":
    # Get config file from command line argument or use default
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'experiment3_config.json'
    
    success = run_experiment3_batch(config_file)
    
    if success:
        print(f"\n🎉 Experiment 3 completed successfully!")
        print(f"Ready to analyze multi-zero scaling behavior.")
    else:
        print(f"\n❌ Experiment 3 failed. Check error messages above.")
        sys.exit(1)
