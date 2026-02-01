#!/usr/bin/env python3
"""
Test script for complete h2plant_detailed.yaml with native SOEC.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.config.plant_config import SimulationConfig

def main():
    print("="*70)
    print("DETAILED PLANT TEST - Native SOEC (1-minute timestep)")
    print("="*70 + "\n")
    
    config_path = "configs/h2plant_detailed.yaml"
    
    print(f"[1/3] Loading configuration: {config_path}")
    try:
        builder = PlantBuilder.from_file(config_path)
        print(f"  ✓ Plant built: {builder.registry.get_component_count()} components")
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n[2/3] Initializing simulation...")
    dt = 1/60.0  # 1 minute in hours
    try:
        builder.registry.initialize_all(dt=dt)
        print(f"  ✓ All components initialized (dt={dt*60:.1f} min)")
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n[3/3] Running 8-hour simulation...")
    try:
        # Create simulation config
        sim_config = SimulationConfig(
            timestep_hours=dt,
            duration_hours=8.0,
            start_hour=0
        )
        
        engine = SimulationEngine(
            registry=builder.registry,
            config=sim_config
        )
        duration_hours = 8.0  # Set in config above
        
        history = engine.run()
        
        print(f"  ✓ Simulation complete!")
        print(f"    - Simulated: {duration_hours} hours ({int(duration_hours*60)} steps)")
        print(f"    - History entries: {len(history)}")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Get final state from monitoring or results
    # The engine returns a dict with 'history', 'metadata', etc.
    if isinstance(history, dict):
        # New format: get from monitoring data
        final_state = {}
        
        # Try to get final component states
        for comp_id in ['soec_cluster', 'pem_electrolyzer_detailed', 'lp_tanks', 'hp_tanks', 'dual_path_coordinator']:
            try:
                comp = builder.registry.get(comp_id)
                final_state[comp_id] = comp.get_state()
            except:
                final_state[comp_id] = {}
    else:
        # Old format: list of states
        final_state = history[-1] if history else {}
    
    # SOEC metrics
    soec_state = final_state.get('soec_cluster', {})
    print(f"\nSOEC Cluster:")
    print(f"  Total H2 produced: {soec_state.get('total_h2_produced_kg', 0):.2f} kg")
    print(f"  Total steam consumed: {soec_state.get('total_steam_consumed_kg', 0):.2f} kg")
    print(f"  Final power: {soec_state.get('P_actual_mw', 0):.2f} MW")
    print(f"  Active modules: {soec_state.get('active_modules', 0)}")
    
    # PEM metrics
    pem_state = final_state.get('pem_electrolyzer_detailed', {})
    print(f"\nPEM Electrolyzer:")
    print(f"  Total H2 produced: {pem_state.get('cumulative_h2_kg', 0):.2f} kg")
    print(f"  Final power: {pem_state.get('power_consumption_mw', 0):.2f} MW")
    
    # Storage
    lp_state = final_state.get('lp_tanks', {})
    hp_state = final_state.get('hp_tanks', {})
    print(f"\nStorage:")
    print(f"  LP tanks: {lp_state.get('current_mass_kg', 0):.2f} kg @ {lp_state.get('current_pressure_bar', 0):.1f} bar")
    print(f"  HP tanks: {hp_state.get('current_mass_kg', 0):.2f} kg @ {hp_state.get('current_pressure_bar', 0):.1f} bar")
    
    # Coordinator
    coord_state = final_state.get('dual_path_coordinator', {})
    print(f"\nCoordinator:")
    print(f"  Total H2 produced: {coord_state.get('cumulative_production_kg', 0):.2f} kg")
    print(f"  Energy sold: {coord_state.get('cumulative_sold_energy_mwh', 0):.4f} MWh")
    
    print("\n✅ TEST COMPLETE\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
