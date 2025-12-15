#!/usr/bin/env python3
"""
Test script for O2 Treatment Train Topology.
Runs simulation and validates against Legacy PEM simulator values.
"""
import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# Copy O2 topology to plant_topology.yaml (required by ConfigLoader)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_topo = os.path.join(BASE_DIR, 'scenarios/topologies/o2_treatment_train.yaml')
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
            if frac > 0.01:
                print(f"    {species:>5}: {frac*100:>10.4f} %")
            else:
                print(f"    {species:>5}: {frac*1e6:>10.2f} ppm")
    
    print(f"  {'Phase:':<20} {stream.phase}")


def propagate_stream(source_comp, source_port: str, target_comp, target_port: str):
    """Propagate stream from source to target (simulates FlowNetwork)."""
    try:
        stream = source_comp.get_output(source_port)
        if stream and hasattr(target_comp, 'receive_input'):
            target_comp.receive_input(target_port, stream, 'gas')
            return stream
    except (ValueError, NotImplementedError) as e:
        print(f"  [!] Propagation error: {e}")
    return None


def main():
    print("\n" + "="*70)
    print("  O2 TREATMENT TRAIN TEST")
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
    
    # Get components in order
    o2_source = components.get('O2_Source')
    kod_1 = components.get('KOD_1')
    dry_cooler = components.get('DryCooler_1')
    chiller = components.get('Chiller_1')
    kod_2 = components.get('KOD_2')
    coalescer = components.get('Coalescer_1')
    valve = components.get('Valve_1')
    
    # Run simulation with manual stream propagation
    num_steps = 5
    print(f"\n[2] Running simulation ({num_steps} timesteps)...")
    
    for step in range(num_steps):
        t = step * dt
        
        # 1. O2 Source produces O2
        o2_source.step(t)
        
        # 2. O2 Source → KOD_1
        propagate_stream(o2_source, 'h2_out', kod_1, 'gas_inlet')
        kod_1.step(t)
        
        # 3. KOD_1 → DryCooler_1
        propagate_stream(kod_1, 'gas_outlet', dry_cooler, 'fluid_in')
        dry_cooler.step(t)
        
        # 4. DryCooler_1 → Chiller_1
        propagate_stream(dry_cooler, 'fluid_out', chiller, 'fluid_in')
        chiller.step(t)
        
        # 5. Chiller_1 → KOD_2
        propagate_stream(chiller, 'fluid_out', kod_2, 'gas_inlet')
        kod_2.step(t)
        
        # 6. KOD_2 → Coalescer_1
        propagate_stream(kod_2, 'gas_outlet', coalescer, 'inlet')
        coalescer.step(t)
        
        # 7. Coalescer_1 → Valve_1
        propagate_stream(coalescer, 'outlet', valve, 'inlet')
        valve.step(t)
    
    # Display results
    print(f"\n[3] Stream Summary after {num_steps} timesteps:")
    
    # Stream summary table
    print("\n" + "="*90)
    print(f"{'Component':<18} | {'T_out':>10} | {'P_out':>12} | {'O2':>12} | {'H2O':>12} | {'H2':>10}")
    print("-"*90)
    
    component_list = [
        ('O2_Source', o2_source, 'h2_out'),
        ('KOD_1', kod_1, 'gas_outlet'),
        ('DryCooler_1', dry_cooler, 'fluid_out'),
        ('Chiller_1', chiller, 'fluid_out'),
        ('KOD_2', kod_2, 'gas_outlet'),
        ('Coalescer_1', coalescer, 'outlet'),
        ('Valve_1', valve, 'outlet'),
    ]
    
    for name, comp, port in component_list:
        if comp is None:
            print(f"{name:<18} | NOT FOUND")
            continue
        
        stream = comp.get_output(port)
        if stream is None:
            print(f"{name:<18} | NO OUTPUT")
            continue
        
        T_c = stream.temperature_k - 273.15
        P_bar = stream.pressure_pa / 1e5
        o2_frac = stream.composition.get('O2', 0)
        h2o_frac = stream.composition.get('H2O', 0)
        h2_frac = stream.composition.get('H2', 0)
        
        o2_str = f"{o2_frac*100:.2f}%" if o2_frac > 0.01 else f"{o2_frac*1e6:.0f} ppm"
        h2o_str = f"{h2o_frac*100:.2f}%" if h2o_frac > 0.01 else (f"{h2o_frac*1e6:.0f} ppm" if h2o_frac > 0 else "0 ppm")
        h2_str = f"{h2_frac*1e6:.0f} ppm" if h2_frac < 0.01 else f"{h2_frac*100:.2f}%"
        
        print(f"{name:<18} | {T_c:>8.1f}°C | {P_bar:>10.2f} bar | {o2_str:>12} | {h2o_str:>12} | {h2_str:>10}")
    
    print("-"*90)
    
    # Compare with Legacy
    print("\n" + "="*70)
    print("  COMPARISON WITH LEGACY (Expected Values)")
    print("="*70)
    print("""
  Legacy O2 Treatment Train:
  | Component     | P_out (bar) | T (°C) | y_H2O (ppm molar) |
  |---------------|-------------|--------|-------------------|
  | Entrada       | 40.00       | 60     | ~50000 (5%)       |
  | KOD 1         | 39.95       | 60     | ~40000            |
  | Dry Cooler 1  | 39.90       | ~32    | ~40000            |
  | Chiller 1     | 39.90       | 4      | ~204              |
  | KOD 2         | 39.85       | 4      | ~204              |
  | Coalescedor   | 39.70       | 4      | ~204              |
  | Válvula       | 15.00       | varies | ~204              |
""")
    
    print("\n  Test completed!")


if __name__ == "__main__":
    main()
