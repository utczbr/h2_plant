import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.getcwd())

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.core.constants import DryCoolerIndirectConstants as DCC

logging.basicConfig(level=logging.INFO)

def main():
    base_path = Path("/home/stuart/Documentos/Planta Hidrogenio")
    topology_file = base_path / "scenarios/plant_topology.yaml"
    
    loader = ConfigLoader(str(base_path / "scenarios"))
    context = loader.load_context(
        topology_file="plant_topology.yaml"
    )
    
    builder = PlantGraphBuilder(context)
    registry = builder.build()
    
    print(f"Context type: {type(context)}")
    print(f"Context.simulation type: {type(context.simulation)}")
    print(f"Context.simulation vars: {vars(context.simulation) if hasattr(context.simulation, '__dict__') else 'No __dict__'}")
    
    # Manual Test
    dc = registry.get("SOEC_H2_Intercooler_1")
    if dc:
        print(f"SOEC_H2_Intercooler_1 uses central utility? {dc.use_central_utility}")
        print(f"Design Capacity: {dc.design_capacity_kw} kW")
        
        # Manually initialize
        dt = 1.0/60.0
        dc.initialize(dt, registry)
        
        # Create dummy hot inlet stream (typical compressor outlet)
        # Mass flow ~836 kg/h (from previous summary context), Temp ~150 C, Pressure ~30 bar (guess)
        from h2_plant.core.stream import Stream
        hot_stream = Stream(
            mass_flow_kg_h=836.5,
            temperature_k=150.0 + 273.15,
            pressure_pa=30.0 * 1e5,
            composition={'H2': 1.0},
            phase='gas'
        )
        
        # Inject input
        dc.receive_input("fluid_in", hot_stream)
        
        # Step
        dc.step(0.0)
        
        # Check State
        state = dc.get_state()
        print(f"\nSOEC_H2_Intercooler_1 State (Manual Step):")
        print(f"  Inlet Temp: {hot_stream.temperature_k - 273.15:.2f} C")
        print(f"  Outlet Temp: {state.get('outlet_temp_c', 'N/A')} C")
        print(f"  Target Temp: {dc.target_outlet_temp_c} C")
        print(f"  TQC Duty: {state.get('tqc_duty_kw', 'N/A')} kW")
        print(f"  Glycol Hot: {state.get('glycol_hot_c', 'N/A')} C")
        print(f"  Glycol Cold: {state.get('glycol_cold_c', 'N/A')} C")
        
        # Check TQC params
        print(f"  TQC Area: {dc.tqc_area_m2:.2f} m2")
        print(f"  TQC UA: {dc.tqc_u_value * dc.tqc_area_m2 / 1000.0:.2f} kW/K")
        
        # Check C_min
        print(f"  Glycol Flow: {dc.glycol_flow_kg_s:.2f} kg/s")
    else:
        print("SOEC_H2_Intercooler_1 not found!")

if __name__ == "__main__":
    main()
