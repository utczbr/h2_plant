
import sys
import os
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append('/home/stuart/Documentos/Planta Hidrogenio')

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.component_ids import ComponentID

# Import Migrated Components
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.control.valve import ThrottlingValve
from h2_plant.components.separation.psa import PSA

from h2_plant.core.component import Component

# Mock LUT Manager (Optional, minimal implementation if needed)
class MockLUTManager(Component):
    def initialize(self, dt: float, registry: Any) -> None: pass
    def step(self, t: float) -> None: pass
    def get_state(self) -> Dict[str, Any]: return {}
    def get_output(self, port: str) -> Any: return None
    
    def lookup(self, fluid, prop, P, T):
        # Return dummy values to prevent crashes if real LUT fails or is missing data
        if prop == 'H':
            return 1000.0 * T # Fake Cp=1000
        return 0.0

def run_test():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 1. Setup Registry
    registry = ComponentRegistry()
    registry.register(ComponentID.LUT_MANAGER.value, MockLUTManager()) # Register mock if real one is complex to set up here

    # 2. Instantiate Components
    coalescer = Coalescer(component_id='C-01', d_shell=0.32, l_elem=1.0, gas_type='H2')
    deoxo = DeoxoReactor(component_id='R-01')
    chiller = Chiller(component_id='HX-01', cooling_capacity_kw=50, target_temp_k=303.15, cop=4.0)
    valve = ThrottlingValve({'component_id': 'V-01', 'P_out_pa': 30e5, 'fluid': 'H2'}) # Drop to 30 bar
    psa = PSA(component_id='D-01', recovery_rate=0.90)

    # 3. Initialize
    dt = 1.0/60.0 # 1 minute
    components = [coalescer, deoxo, chiller, valve, psa]
    for c in components:
        c.initialize(dt, registry)

    # 4. Create Input Stream (Wet H2 with O2 contamination)
    # 60C, 40 bar, 100 kg/h
    # 98% H2, 1% O2, 1% H2O (mass frac approx)
    input_stream = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=333.15, # 60 C
        pressure_pa=40e5,     # 40 bar
        composition={'H2': 0.98, 'O2': 0.01, 'H2O': 0.09, 'H2O_liq': 0.01} # Some liquid water
    )

    print("\n--- SIMULATION START ---")
    print(f"Input: {input_stream.mass_flow_kg_h} kg/h @ {input_stream.temperature_k:.1f}K, {input_stream.pressure_pa/1e5:.1f} bar")

    # 5. Execute Chain (Manual Push)
    
    # Step 1: Coalescer
    print(f"\n[1] Coalescer Input")
    coalescer.receive_input('inlet', input_stream, 'gas')
    coalescer.step(0.0)
    s1 = coalescer.get_output('outlet')
    drain = coalescer.get_output('drain')
    print(f"-> Gas Out: {s1.mass_flow_kg_h:.4f} kg/h")
    print(f"-> Liquid Removed: {drain.mass_flow_kg_h:.4f} kg/h")
    
    # Step 2: Deoxo
    print(f"\n[2] Deoxo Input")
    deoxo.receive_input('inlet', s1, 'gas')
    deoxo.step(0.0)
    s2 = deoxo.get_output('outlet')
    state_deoxo = deoxo.get_state()
    print(f"-> Gas Out: {s2.mass_flow_kg_h:.4f} kg/h @ {s2.temperature_k:.1f}K (Peak: {state_deoxo.get('peak_temperature_c',0)+273.15:.1f}K)")
    print(f"-> Conversion: {state_deoxo.get('conversion_o2_percent',0):.2f}%")
    
    # Step 3: Chiller
    print(f"\n[3] Chiller Input")
    chiller.receive_input('fluid_in', s2, 'gas')
    chiller.step(0.0)
    s3 = chiller.get_output('fluid_out')
    state_chiller = chiller.get_state()
    print(f"-> Gas Out: {s3.mass_flow_kg_h:.4f} kg/h @ {s3.temperature_k:.1f}K")
    print(f"-> Cooling Load: {state_chiller.get('cooling_load_kw',0):.2f} kW")
    
    # Step 4: Valve
    print(f"\n[4] Valve Input (Throttle to 30 bar)")
    valve.receive_input('inlet', s3, 'gas')
    valve.step(0.0)
    s4 = valve.get_output('outlet')
    print(f"-> Gas Out: {s4.mass_flow_kg_h:.4f} kg/h @ {s4.pressure_pa/1e5:.1f} bar")
    
    # Step 5: PSA
    print(f"\n[5] PSA Input")
    psa.receive_input('gas_in', s4, 'gas')
    psa.step(0.0)
    prod = psa.get_output('purified_gas_out')
    tail = psa.get_output('tail_gas_out')
    print(f"-> Product H2: {prod.mass_flow_kg_h:.4f} kg/h (Purity H2={prod.composition.get('H2',0):.4f})")
    print(f"-> Tail Gas: {tail.mass_flow_kg_h:.4f} kg/h")
    
    print("\n--- TEST COMPLETE ---")

if __name__ == "__main__":
    run_test()
