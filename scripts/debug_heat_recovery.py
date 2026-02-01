
import logging
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.config.loader import ConfigLoader
from h2_plant.core.stream import Stream

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Topology
loader = ConfigLoader("/home/stuart/Documentos/Planta Hidrogenio/scenarios")
context = loader.load_context()
builder = PlantGraphBuilder(context)
components = builder.build()

if isinstance(components, dict):
    component_map = components
else:
    component_map = {c.component_id: c for c in components}

# Chain components
chain_ids = [
    "SOEC_Makeup_Mixer",
    "SOEC_Water_Splitter",
    "ATR_Syngas_Cooler",
    "ATR_H2O_Boiler",
    "ATR_H2O_Compressor_1",
    "ATR_H2O_DryCooler",
    "ATR_H2O_Compressor_2",
    "SOEC_Feed_Attemperator"
]

print("--- Checking Chain Integrity ---")
for cid in chain_ids:
    if cid not in component_map:
        print(f"ERROR: Component {cid} not found!")
    else:
        print(f"FOUND: {cid} ({type(component_map[cid]).__name__})")

from h2_plant.core.component_registry import ComponentRegistry
registry = ComponentRegistry()

# Initialize components
print("--- Initializing Components ---")
for c in component_map.values():
    c.initialize(1.0, registry)

# Manually push flow
print("\n--- Simulating Step ---")
dt = 1.0 # 1 hour
makeup_mixer = component_map["SOEC_Makeup_Mixer"]

# Mock Inputs for Makeup Mixer
makeup_mixer.receive_input("consumption_in", 3600.0, "signal")
makeup_mixer.step(0.0)
out_makeup = makeup_mixer.get_output("water_out")
print(f"Makeup Output: {out_makeup.mass_flow_kg_h:.2f} kg/h")

# Propagate manually
current_stream = out_makeup
current_sender = "SOEC_Makeup_Mixer"

next_steps = [
    ("SOEC_Water_Splitter", "inlet"),
    ("ATR_Syngas_Cooler", "water_in"), # Splitter outlet_1
    ("ATR_H2O_Boiler", "inlet"),
    ("ATR_H2O_Compressor_1", "h2_in"), # Assuming CompressorStorage uses h2_in/inlet
    ("ATR_H2O_DryCooler", "fluid_in"),
    ("ATR_H2O_Compressor_2", "h2_in"),
    ("SOEC_Feed_Attemperator", "inlet_1")
]

# Splitter Special Handling
splitter = component_map["SOEC_Water_Splitter"]
splitter.receive_input("inlet", current_stream, "stream")
splitter.step(0.0)
out_split_1 = splitter.get_output("outlet_1") # 91.54%
out_split_2 = splitter.get_output("outlet_2") # 8.46%
print(f"Splitter Out 1: {out_split_1.mass_flow_kg_h:.2f} kg/h")
print(f"Splitter Out 2: {out_split_2.mass_flow_kg_h:.2f} kg/h")

# Follow Main Loop (Out 1)
current_stream = out_split_1

loop_chain = [
    ("ATR_Syngas_Cooler", "water_in", "water_out"),
    ("ATR_H2O_Boiler", "fluid_in", "fluid_out"),
    ("ATR_H2O_Compressor_1", "inlet", "outlet"),
    ("ATR_H2O_DryCooler", "fluid_in", "fluid_out"),
    ("ATR_H2O_Compressor_2", "inlet", "outlet"),
    ("SOEC_Feed_Attemperator", "inlet_1", "outlet")
]

for comp_id, in_port, out_port in loop_chain:
    comp = component_map[comp_id]
    print(f"\nProcessing {comp_id}...")
    
    # Send Input
    accepted = comp.receive_input(in_port, current_stream, "stream")
    print(f"  Sent {current_stream.mass_flow_kg_h:.2f} kg/h to {in_port}, Accepted: {accepted:.2f}")
    
    # Step
    comp.step(0.0)
    
    # Get Output
    output = comp.get_output(out_port)
    if output:
        print(f"  Output {out_port}: {output.mass_flow_kg_h:.2f} kg/h, T={output.temperature_k:.1f}K, Phase={output.phase}")
        if output.composition:
            print(f"  Composition: {output.composition}")
        current_stream = output
    else:
        print(f"  ERROR: No output from {out_port}")
        break
