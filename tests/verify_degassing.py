
import sys
import os
from pathlib import Path

# Add project root to path
# Add project root to path (current directory)
sys.path.append(os.path.dirname(__file__))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.stream import Stream

def main():
    base_path = Path("/home/stuart/Documentos/Planta Hidrogenio")
    print("Loading configuration...")
    loader = ConfigLoader(str(base_path / "scenarios"))
    context = loader.load_context(
        topology_file="plant_topology.yaml"
    )
    
    print("Building plant graph...")
    builder = PlantGraphBuilder(context)
    components = builder.build()
    
    # Populate registry manually if needed, or use a proper Registry object
    registry = ComponentRegistry()
    for comp_id, comp in components.items():
        registry.register(comp_id, comp)
        
    registry.initialize_all(dt=1.0/60.0) # Initialize components
    
    # Manual Test for H2 Path
    print("\n--- Testing H2 Drain Path (Path 1) ---")
    
    valve1 = registry.get("PEM_Water_Return_Valve_1")
    degasser1 = registry.get("PEM_Degasser_1")
    combiner = registry.get("PEM_Drains_Combiner")
    
    if not (valve1 and degasser1 and combiner):
        print("ERROR: Could not find H2 Path components!")
        return

    # Simulate H2-saturated drain water (e.g. from KOD)
    # 5% H2 gas by mass (extreme case for visibility), mostly water
    h2_drain_input = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=350.0, # Hot drain
        pressure_pa=1000000.0, # 10 bar
        composition={'H2O': 0.95, 'H2': 0.05},
        phase='mixed'
    )
    
    # 1. Step Valve 1
    print(f"Injecting {h2_drain_input.mass_flow_kg_h} kg/h (5% H2) into Valve 1")
    valve1.receive_input("inlet", h2_drain_input, resource_type="water")
    valve1.step(0.0)
    
    valve_out = valve1.get_output("outlet")
    print(f"Valve 1 Outlet Pressure: {valve_out.pressure_pa/1e5:.2f} bar")
    
    # 2. Step Degasser 1
    degasser1.receive_input("mixture_in", valve_out, resource_type="water")
    degasser1.step(0.0)
    
    gas_out = degasser1.get_output("gas_out")
    liquid_out = degasser1.get_output("liquid_out")
    
    print(f"Degasser 1 Gas Vent: {gas_out.mass_flow_kg_h:.2f} kg/h (Should be ~5.0)")
    print(f"Degasser 1 Liquid Recovery: {liquid_out.mass_flow_kg_h:.2f} kg/h (Should be ~95.0)")
    print(f"Degasser 1 Liquid Composition: {liquid_out.composition}")
    
    if abs(gas_out.mass_flow_kg_h - 5.0) < 0.1:
        print("✅ H2 Gas separation successful")
    else:
        print("❌ H2 Gas separation FAILED")

    # Manual Test for O2 Path
    print("\n--- Testing O2 Drain Path (Path 2) ---")
    
    valve2 = registry.get("PEM_Water_Return_Valve_2")
    degasser2 = registry.get("PEM_Degasser_2")
    
    if not (valve2 and degasser2):
        print("ERROR: Could not find O2 Path components!")
        return
        
    # Simulate O2-saturated drain water
    o2_drain_input = Stream(
        mass_flow_kg_h=50.0,
        temperature_k=350.0,
        pressure_pa=1000000.0, # 10 bar
        composition={'H2O': 0.95, 'O2': 0.05},
        phase='mixed'
    )
    
    # 1. Step Valve 2
    valve2.receive_input("inlet", o2_drain_input, resource_type="water")
    valve2.step(0.0)
    
    # 2. Step Degasser 2
    degasser2.receive_input("mixture_in", valve2.get_output("outlet"), resource_type="water")
    degasser2.step(0.0)
    
    gas_out_2 = degasser2.get_output("gas_out")
    liquid_out_2 = degasser2.get_output("liquid_out")
    
    print(f"Degasser 2 Gas Vent: {gas_out_2.mass_flow_kg_h:.2f} kg/h (Should be ~2.5)")
    print(f"Degasser 2 Liquid Recovery: {liquid_out_2.mass_flow_kg_h:.2f} kg/h (Should be ~47.5)")
    
    if abs(gas_out_2.mass_flow_kg_h - 2.5) < 0.1:
        print("✅ O2 Gas separation successful")
    else:
        print("❌ O2 Gas separation FAILED")
        
    # 3. Step Combiner
    print("\n--- Testing Combiner ---")
    combiner.receive_input("inlet_1", liquid_out, resource_type="water") # From H2 path
    combiner.receive_input("inlet_2", liquid_out_2, resource_type="water") # From O2 path
    combiner.step(0.0)
    
    combined = combiner.get_output("outlet")
    print(f"Combiner Output: {combined.mass_flow_kg_h:.2f} kg/h (Should be ~142.5)")
    print(f"Combiner Composition: {combined.composition}")
    
    if abs(combined.mass_flow_kg_h - 142.5) < 0.5:
        print("✅ Combiner merging successful")
    else:
        print("❌ Combiner merging FAILED")

if __name__ == "__main__":
    main()
