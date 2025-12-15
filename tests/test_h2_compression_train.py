#!/usr/bin/env python3
"""
H2 Compression Train Test Script

Tests the flow of hydrogen through: H2Source → LP Compressor → LP Tank → HP Compressor → HP Tank

This script manually propagates streams between components (simulating FlowNetwork)
to display all H2 stream characteristics at each stage.
"""

import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# Copy test topology
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_topo = os.path.join(BASE_DIR, 'scenarios/topologies/topology_h2_compression_test.yaml')
dst_topo = os.path.join(BASE_DIR, 'scenarios/plant_topology.yaml')
shutil.copy(src_topo, dst_topo)


def print_stream(label: str, stream: Stream):
    """Print detailed stream properties."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    
    if stream is None:
        print("  [No stream data]")
        return
    
    print(f"  {'Mass Flow:':<20} {stream.mass_flow_kg_h:>12.2f} kg/h")
    print(f"  {'Temperature:':<20} {stream.temperature_k:>12.2f} K  ({stream.temperature_k - 273.15:>6.1f}°C)")
    print(f"  {'Pressure:':<20} {stream.pressure_pa:>12.0f} Pa ({stream.pressure_pa / 1e5:>6.2f} bar)")
    
    print(f"  Composition:")
    for species, frac in stream.composition.items():
        if frac > 0:
            print(f"    {species:>5}: {frac*100:>10.4f} %")
    
    print(f"  {'Phase:':<20} {stream.phase}")


def print_tank_state(label: str, tank):
    """Print tank state."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    
    state = tank.get_state()
    total_mass = state.get('total_mass_kg', 0)
    available = state.get('available_capacity_kg', 0)
    n_tanks = state.get('n_tanks', 1)
    
    print(f"  {'Total Mass:':<20} {total_mass:>12.4f} kg")
    print(f"  {'Available Capacity:':<20} {available:>12.4f} kg")
    print(f"  {'Number of Tanks:':<20} {n_tanks:>12}")
    
    # Per-tank info if available
    if 'pressures' in state:
        pressures = state['pressures']
        print(f"  Tank Pressures (bar): {[f'{p:.1f}' for p in pressures]}")


def propagate_stream(source_comp, source_port: str, target_comp, target_port: str):
    """Propagate stream from source to target (simulates FlowNetwork)."""
    try:
        stream = source_comp.get_output(source_port)
        if stream and hasattr(target_comp, 'receive_input'):
            target_comp.receive_input(target_port, stream, 'hydrogen')
            return stream
    except (ValueError, NotImplementedError) as e:
        print(f"  [!] Propagation error: {e}")
    return None


def main():
    print("\n" + "="*70)
    print("  H2 COMPRESSION TRAIN TEST - TankArray Version")
    print("="*70)
    
    # Load and build
    print("\n[1] Loading configuration and building components...")
    loader = ConfigLoader(os.path.join(BASE_DIR, 'scenarios'))
    context = loader.load_context()
    
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # Create registry and initialize
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)
    
    dt = 1/60  # 1-minute timestep
    registry.initialize_all(dt)
    
    print(f"\n  Built {len(components)} components:")
    for cid, comp in components.items():
        print(f"    - {cid}: {comp.__class__.__name__}")
    
    # Get components
    h2_source = components.get('H2_Source')
    lp_comp = components.get('LP_Compressor')
    lp_tank = components.get('LP_Tank')
    hp_comp = components.get('HP_Compressor')
    hp_tank = components.get('HP_Tank')
    
    # Run simulation with manual stream propagation
    num_steps = 10
    print(f"\n[2] Running simulation ({num_steps} timesteps with stream propagation)...")
    
    for step in range(num_steps):
        t = step * dt
        
        # 1. H2 Source produces H2
        h2_source.step(t)
        
        # 2. Propagate H2Source → LP Compressor
        propagate_stream(h2_source, 'h2_out', lp_comp, 'inlet')
        lp_comp.step(t)
        
        # 3. Propagate LP Compressor → LP Tank
        # TankArray uses receive_input with 'h2_in' port
        propagate_stream(lp_comp, 'outlet', lp_tank, 'h2_in')
        lp_tank.step(t)
        
        # 4. LP Tank → HP Compressor (demand-driven discharge)
        # TankArray uses discharge() method
        discharge_amount = 100.0 * dt  # kg per timestep
        actual_discharge = lp_tank.discharge(discharge_amount)
        
        if actual_discharge > 0:
            # Create stream from tank output
            lp_state = lp_tank.get_state()
            discharge_stream = Stream(
                mass_flow_kg_h=actual_discharge / dt,
                temperature_k=298.15,
                pressure_pa=lp_state.get('pressures', [200])[0] * 1e5,
                composition={'H2': 1.0},
                phase='gas'
            )
            hp_comp.receive_input('inlet', discharge_stream, 'hydrogen')
        
        hp_comp.step(t)
        
        # 5. HP Compressor → HP Tank
        propagate_stream(hp_comp, 'outlet', hp_tank, 'h2_in')
        hp_tank.step(t)
    
    # Display results
    print(f"\n[3] Results after {num_steps} timesteps:")
    
    # H2 Source
    print_stream("H2_Source - OUTPUT", h2_source.get_output('h2_out'))
    h2_state = h2_source.get_state()
    print(f"  Cumulative H2: {h2_state.get('cumulative_h2_kg', 0):.2f} kg")
    
    # LP Compressor
    print_stream("LP_Compressor - OUTPUT", lp_comp.get_output('outlet'))
    lp_comp_state = lp_comp.get_state()
    print(f"  Energy Used: {lp_comp_state.get('cumulative_energy_kwh', 0):.2f} kWh")
    
    # LP Tank
    print_tank_state("LP_Tank (TankArray)", lp_tank)
    
    # HP Compressor
    print_stream("HP_Compressor - OUTPUT", hp_comp.get_output('outlet'))
    hp_comp_state = hp_comp.get_state()
    print(f"  Energy Used: {hp_comp_state.get('cumulative_energy_kwh', 0):.2f} kWh")
    
    # HP Tank
    print_tank_state("HP_Tank (TankArray)", hp_tank)
    
    # Summary
    print("\n" + "="*70)
    print("  MASS BALANCE SUMMARY")
    print("="*70)
    
    h2_delivered = h2_state.get('cumulative_h2_kg', 0)
    lp_stored = lp_tank.get_state().get('total_mass_kg', 0)
    hp_stored = hp_tank.get_state().get('total_mass_kg', 0)
    
    print(f"  H2 Source Delivered:  {h2_delivered:>10.4f} kg")
    print(f"  LP Tank Stored:       {lp_stored:>10.4f} kg")
    print(f"  HP Tank Stored:       {hp_stored:>10.4f} kg")
    print(f"  Total Stored:         {lp_stored + hp_stored:>10.4f} kg")
    print(f"  Balance (In - Out):   {h2_delivered - lp_stored - hp_stored:>10.4f} kg")
    
    print("\n  Test completed!")


if __name__ == "__main__":
    main()
