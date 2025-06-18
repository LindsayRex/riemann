#!/usr/bin/env sage

print("Testing Experiment 2 Batch Integration...")
print("=" * 50)

try:
    # Load the batch orchestrator (which should load chunked stats too)
    load('experiment2_batch.sage')
    print("✓ Batch orchestrator loaded successfully")
    
    # Test factory function
    orchestrator = create_experiment2_batch_orchestrator(verbose=False)
    print("✓ Batch orchestrator created successfully")
    print(f"✓ Configurations generated: {len(orchestrator.configurations)}")
    print(f"✓ Chunked stats result initialized: {orchestrator.chunked_stats_result}")
    
    # Test that chunked stats function is available
    from experiment2_chunked_stats import process_hdf5_statistics
    print("✓ Chunked statistics function imported successfully")
    
    print()
    print("INTEGRATION TEST PASSED!")
    print("✓ Option B implementation successful")
    print("✓ Chunked statistics integrated into batch workflow")
    
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
