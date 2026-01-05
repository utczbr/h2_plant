
import sys
import os
import logging
from typing import Dict, Any

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.core.stream import Stream
from h2_plant.components.separation.psa_syngas import SyngasPSA
from h2_plant.reporting.stream_table import print_stream_summary_table

# Setup minimal components dict
components = {}
cid = "ATR_PSA_1"

# Instantiate SyngasPSA
psa = SyngasPSA(component_id=cid)
psa.initialize(dt=1.0, registry=None)

# Create a dummy output stream
psa.product_outlet = Stream(
    mass_flow_kg_h=100.0,
    temperature_k=300.0,
    pressure_pa=2000000.0,
    composition={'H2': 0.9999, 'CH4': 0.0001}
)

components[cid] = psa
topo_order = [cid]

print("--- DEBUGGING ATR_PSA_1 TABLE OUTPUT ---")
print(f"Component Class: {type(psa).__name__}")
print(f"Component ID: {cid}")

# Test manual get_output
print(f"Manual get_output('purified_gas_out'): {psa.get_output('purified_gas_out')}")

print("\nRunning print_stream_summary_table:")
try:
    print_stream_summary_table(components, topo_order)
except Exception as e:
    print(f"CRASHED: {e}")
    import traceback
    traceback.print_exc()
