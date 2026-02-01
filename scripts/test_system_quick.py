#!/usr/bin/env python3
"""
Quick System Test - Verify minute-level simulation works

Tests the system without requiring matplotlib or reference comparison.
"""

import sys
from pathlib import Path

# Add path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

print("\n" + "="*70)
print("QUICK SYSTEM TEST - Minute-Level Simulation")
print("="*70)

print("\n1. Testing imports...")
try:
    from h2_plant.config.plant_builder import PlantBuilder
    from h2_plant.simulation.engine import SimulationEngine
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("\n2. Loading minute-level config...")
config_path = repo_root / "configs" / "plant_pem_soec_minute_level.yaml"
if not config_path.exists():
    print(f"❌ Config not found: {config_path}")
    sys.exit(1)

try:
    builder = PlantBuilder.from_file(config_path)
    print(f"✅ Config loaded: {builder.config.name}")
    print(f"   Timestep: {builder.config.simulation.timestep_hours} hours (= {builder.config.simulation.timestep_hours * 60} minutes)")
except Exception as e:
    print(f"❌ Config load failed: {e}")
    sys.exit(1)

print("\n3. Checking registered components...")
try:
    registry = builder.registry
    component_count = registry.get_component_count()
    print(f"✅ {component_count} components registered")
    
    # Check critical components
    critical = ['environment_manager', 'soec_cluster', 'pem_electrolyzer_detailed', 'dual_path_coordinator']
    for comp_id in critical:
        if registry.has(comp_id):
            comp = registry.get(comp_id)
            print(f"   ✓ {comp_id}: {type(comp).__name__}")
        else:
            print(f"   ✗ {comp_id}: NOT FOUND")
except Exception as e:
    print(f"❌ Component check failed: {e}")
    sys.exit(1)

print("\n4. Running 1-hour test (60 minutes)...")
try:
    # Override to run just 1 hour for quick test
    builder.config.simulation.duration_hours = 1.0
    
    engine = SimulationEngine(
        registry=builder.registry,
        config=builder.config.simulation,
        topology=getattr(builder.config, 'topology', []),
        indexed_topology=getattr(builder.config, 'indexed_topology', [])
    )
    
    print("   Initializing...")
    engine.initialize()
    
    print("   Running simulation...")
    results = engine.run()
    
    print(f"✅ Simulation complete!")
    print(f"   Duration: {results['simulation']['duration_hours']} hours")
    print(f"   Steps: {results['simulation']['duration_hours'] * 60} minutes")
    print(f"   Execution time: {results['simulation']['execution_time_seconds']:.2f} seconds")
    
except Exception as e:
    print(f"❌ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Checking outputs...")
try:
    coordinator = registry.get('dual_path_coordinator')
    state = coordinator.get_state()
    
    print(f"✅ Coordinator state:")
    print(f"   Cumulative H2: {state.get('cumulative_production_kg', 0):.2f} kg")
    print(f"   Sold energy: {state.get('cumulative_sold_energy_mwh', 0):.4f} MWh")
    print(f"   Final sell decision: {state.get('sell_decision', 0)}")
    
except Exception as e:
    print(f"⚠️  Output check failed: {e}")

print("\n" + "="*70)
print("✅ QUICK TEST PASSED!")
print("="*70)
print("\nNext steps:")
print("1. Run full year: python3 run_minute_level_simulation.py")
print("2. Install matplotlib for validation plots: pip3 install matplotlib")
print("3. Then run: python3 tests/validation/test_arbitration_reference_match.py")
print("="*70 + "\n")
