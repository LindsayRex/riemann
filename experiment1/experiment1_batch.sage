# ############################################################################
#
# EXPERIMENT 1: BATCH ORCHESTRATOR
# =================================
#
# Four-layer architecture implementation following the Design Guide:
# 1. Batch Orchestrator (this file) - Entry Point & Configuration Management
# 2. Mathematical Core - experiment1_math.sage
# 3. Statistical Analysis - experiment1_stats.sage  
# 4. Visualization Engine - experiment1_viz.sage
#
# This orchestrator coordinates: Math → HDF5 → Stats → Viz pipeline
#
# Usage:
#   sage experiment1_batch.sage experiment1_config.json
#
# ############################################################################

import time
import json
import sys
from pathlib import Path

# Load modules following Design Guide pattern
load('experiment1_math.sage')
load('experiment1_stats.sage') 
load('experiment1_viz.sage')

class Experiment1BatchOrchestrator:
    """
    Batch orchestrator implementing four-layer architecture.
    Coordinates Math → Stats → Viz pipeline with HDF5 data storage.
    """
    
    def __init__(self, config_file):
        """
        Initialize batch orchestrator.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        self.config = self._load_configuration()
        self.hdf5_file = f"data/{self.config.get('output_file', 'experiment1_analysis.h5')}"
        
        print("=" * 80)
        print("EXPERIMENT 1: BATCH ORCHESTRATOR")
        print("=" * 80)
        print(f"Configuration: {config_file}")
        print(f"Output HDF5: {self.hdf5_file}")
        
        # Create data and results directories
        Path("data").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
    def _load_configuration(self):
        """Load and validate configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Validate required parameters
            required = ['gamma', 'delta_range', 'delta_steps']
            for param in required:
                if param not in config:
                    raise ValueError(f"Missing required parameter: {param}")
                    
            print(f"✓ Configuration loaded: {len(config)} parameters")
            return config
            
        except Exception as e:
            print(f"✗ Configuration error: {e}")
            sys.exit(1)
        
    def process_batch_configs(self):
        """
        Process all configurations in batch_configs array.
        Implements the Design Guide batch processing pattern.
        """
        batch_configs = self.config.get('batch_configs', [self.config])
        total_configs = len(batch_configs)
        
        print(f"\n🚀 Processing {total_configs} configurations...")
        start_time = time.time()
        
        for i, config_override in enumerate(batch_configs):
            config_name = f"config_{i+1}"
            
            # Create merged configuration
            merged_config = self.config.copy()
            merged_config.update(config_override)
            
            # Generate descriptive config name based on parameters
            if 'gamma' in config_override:
                gamma_str = str(config_override['gamma']).replace('.', '_')
                config_name += f"_gamma_{gamma_str}"
            if 'test_function_type' in config_override:
                config_name += f"_{config_override['test_function_type']}"
            if 'high_precision' in config_override:
                config_name += "_high_precision"
                
            print(f"\n{'='*60}")
            print(f"CONFIG {i+1}/{total_configs}: {config_name}")
            print(f"{'='*60}")
            
            try:
                # Layer 2: Mathematical Core
                self._run_mathematical_core(merged_config, config_name)
                
                print(f"✓ Configuration {i+1}/{total_configs} completed")
                
            except Exception as e:
                print(f"✗ Configuration {i+1} failed: {e}")
                continue
                
        # Layer 3: Statistical Analysis (process entire HDF5 dataset)
        print(f"\n📊 Running statistical analysis on complete dataset...")
        self._run_statistical_analysis()
        
        # Layer 4: Visualization Engine (generate summary images)
        print(f"\n📈 Generating visualization summary...")
        self._run_visualization_engine()
        
        total_time = time.time() - start_time
        print(f"\n✅ Batch processing completed in {total_time:.2f} seconds")
        
    def _run_mathematical_core(self, config, config_name):
        """
        Execute Layer 2: Mathematical Core computation.
        
        Args:
            config: Configuration dictionary
            config_name: Unique name for this configuration
        """
        print(f"🔢 Mathematical Core: γ={config['gamma']}")
        
        # Initialize mathematical core
        math_core = Experiment1Math(
            gamma=config['gamma'],
            delta_range=config.get('delta_range', 0.1),
            delta_steps=config.get('delta_steps', 41),
            num_test_functions=config.get('num_test_functions', 20),
            test_function_type=config.get('test_function_type', 'gaussian')
        )
        
        # Run perturbation analysis
        results = math_core.run_perturbation_sweep(verbose=True)
        
        # Write to HDF5
        math_core.write_to_hdf5(results, self.hdf5_file, config_name)
        
    def _run_statistical_analysis(self):
        """
        Execute Layer 3: Statistical Analysis on complete HDF5 dataset.
        """
        stats_analyzer = Experiment1Stats(self.hdf5_file)
        stats_analyzer.process_all_configurations()
        
    def _run_visualization_engine(self):
        """
        Execute Layer 4: Visualization Engine - generate 5 summary images.
        """
        viz_engine = Experiment1Viz(self.hdf5_file)
        viz_engine.generate_summary_visualizations()

# Command line execution
if len(sys.argv) != 2:
    print("Usage: sage experiment1_batch.sage <config_file>")
    print("Example: sage experiment1_batch.sage experiment1_config.json")
    sys.exit(1)

config_file = sys.argv[1]

# Verify config file exists
if not Path(config_file).exists():
    print(f"Error: Configuration file '{config_file}' not found")
    sys.exit(1)

# Run batch orchestrator
orchestrator = Experiment1BatchOrchestrator(config_file)
orchestrator.process_batch_configs()
