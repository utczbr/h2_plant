
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
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
    
    registry = ComponentRegistry()
    for comp_id, comp in components.items():
        registry.register(comp_id, comp)
        
    registry.initialize_all(dt=1.0/60.0)
    
    print("\n--- Testing SOEC H2 Boiler ---")
    boiler = registry.get("SOEC_H2_Boiler")
    
    if not boiler:
        print("❌ SOEC_H2_Boiler NOT found in registry!")
        return

    print(f"Boiler Found: {boiler.component_id}")
    print(f"Target Temp: {boiler.target_temp_k - 273.15:.2f} C")
    
    # Simulate Cold H2 Input (e.g. 5C from Cyclone after KOD cooling?)
    # Assuming Cyclone output might be low temp if upstream KOD cooled it.
    input_stream = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=278.15, # 5.0 C
        pressure_pa=3000000.0, # 30 bar
        composition={'H2': 1.0},
        phase='gas'
    )
    
    print(f"Injecting {input_stream.mass_flow_kg_h} kg/h H2 at {input_stream.temperature_k - 273.15:.2f} C")
    print(f"Stream Enthalpy (h_in): {input_stream.specific_enthalpy_j_kg} J/kg")
    
    if boiler.lut:
        h_target_check = boiler.lut.lookup('H2', 'H', input_stream.pressure_pa, boiler.target_temp_k)
        print(f"Expected Target Enthalpy (H2 at 15C): {h_target_check} J/kg")
    else:
        print("Boiler has no LUT!")

    boiler.receive_input("fluid_in", input_stream)
    boiler.step(0.0)
    
    out_stream = boiler.get_output("fluid_out")
    
    if out_stream:
        print(f"Outlet Temp: {out_stream.temperature_k - 273.15:.2f} C")
        print(f"Power Used: {boiler.power_kw:.2f} kW")
        
        if abs(out_stream.temperature_k - 288.15) < 0.5:
             print("✅ Boiler successfully heated stream to ~15 C")
        else:
             print("❌ Boiler output temperature incorrect")
    else:
        print("❌ No output stream generated")

if __name__ == "__main__":
    main()
