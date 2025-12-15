#!/usr/bin/env python3
"""Debug script to trace dispatch strategy component detection."""

import sys
import os
sys.path.append(os.path.abspath('.'))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy

print("=== Debug: Component Detection ===\n")

# 1. Load config
scenarios_dir = "scenarios"
loader = ConfigLoader(scenarios_dir)
context = loader.load_context()

# 2. Build graph
builder = PlantGraphBuilder(context)
components = builder.build()

# 3. List components
print("Components built by PlantGraphBuilder:")
for cid, comp in components.items():
    print(f"  - {cid}: {comp.__class__.__name__}")
    if hasattr(comp, 'V_cell'):
        print(f"      (has V_cell attribute)")
    if hasattr(comp, 'soec_state'):
        print(f"      (has soec_state attribute)")

# 4. Create registry
registry = ComponentRegistry()
for cid, comp in components.items():
    registry.register(cid, comp)

# 5. Test dispatch strategy detection
strategy = HybridArbitrageEngineStrategy()
soec = strategy._find_soec(registry)
pem = strategy._find_pem(registry)

print(f"\nDispatch Strategy Detection:")
print(f"  SOEC found: {soec is not None} -> {soec.__class__.__name__ if soec else 'None'}")
print(f"  PEM found: {pem is not None} -> {pem.__class__.__name__ if pem else 'None'}")

# 6. Check if stepping works
if pem:
    dt = context.simulation.timestep_hours
    registry.initialize_all(dt)
    
    # Supply water
    from h2_plant.core.stream import Stream
    water = Stream(mass_flow_kg_h=10000.0, temperature_k=298.15, pressure_pa=5e5, composition={'H2O': 1.0}, phase='liquid')
    pem.receive_input('water_in', water, 'water')
    
    # Set power
    pem.set_power_input_mw(2.5)
    pem.step(0.0)
    
    print(f"\n  PEM step test:")
    print(f"    h2_output_kg: {pem.h2_output_kg}")
    print(f"    water_buffer_kg: {pem.water_buffer_kg}")
    print(f"    P_consumed_W: {pem.P_consumed_W}")
