
import sys
import os

# Add project root to path
sys.path.append('/home/stuart/Documentos/Planta Hidrogenio')

from bs4 import BeautifulSoup # Just to ensure environment is sane
import numpy as np
from h2_plant.core.stream import Stream
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.components.cooling.dry_cooler import DryCooler

def analyze_stream(name, s: Stream):
    print(f"\n--- Analysis of {name} ---")
    print(f"Mass Flow: {s.mass_flow_kg_h:.4f} kg/h")
    print(f"Temperature: {s.temperature_k - 273.15:.2f} C")
    print(f"Pressure: {s.pressure_pa/1e5:.2f} bar")
    print(f"Phase: {s.phase}")
    print("Composition (Mass Fraction):")
    for k, v in s.composition.items():
        if v > 1e-6:
            print(f"  {k}: {v:.6f}")
    
    if s.extra:
        print("Extra Attributes:")
        for k, v in s.extra.items():
            print(f"  {k}: {v}")

    # Calculate Species Mass
    h2_mass = s.mass_flow_kg_h * s.composition.get('H2', 0.0)
    h2o_vap_mass = s.mass_flow_kg_h * s.composition.get('H2O', 0.0)
    h2o_liq_mass = s.mass_flow_kg_h * s.composition.get('H2O_liq', 0.0)
    # Extra liquid water?
    extra_liq = s.entrained_liq_kg_s * 3600.0
    
    print(f"Species Mass Flow (kg/h):")
    print(f"  H2: {h2_mass:.4f}")
    print(f"  H2O (Vap): {h2o_vap_mass:.4f}")
    print(f"  H2O (Liq): {h2o_liq_mass:.4f}")
    if extra_liq > 0:
        print(f"  H2O (Extra Liq): {extra_liq:.4f}")
    
    total_water = h2o_vap_mass + h2o_liq_mass + extra_liq
    print(f"  Total H2O: {total_water:.4f}")
    print(f"  Total Check: {h2_mass + total_water + (s.mass_flow_kg_h * s.composition.get('O2', 0.0)):.4f}")

# 1. Setup Input Stream (SOEC Output)
# Based on User Data:
# H2: 193.38 kg/h
# H2O: 1840.12 kg/h
# O2: 0.61 kg/h
# Total: 2034.12 kg/h
# T = 152 C
# P = 1.0 bar (100,000 Pa)

total_mass = 2034.12
comp = {
    'H2': 193.38 / total_mass,
    'H2O': 1840.12 / total_mass,
    'O2': 0.61 / total_mass
}
# Normalize comp to be safe (though it sums to ~1)
s_in = Stream(
    mass_flow_kg_h=total_mass,
    temperature_k=152 + 273.15,
    pressure_pa=100000.0,
    composition=comp,
    phase='gas',
    extra={'m_dot_H2O_liq_accomp_kg_s': 5.0 / 3600.0} # 5 kg/h entrained
)

analyze_stream("SOEC_Cluster Output (Simulated)", s_in)

# 2. Setup Interchanger
# Needs a cold stream to function
interchanger = Interchanger("Test_Interchanger")
s_cold_in = Stream(mass_flow_kg_h=5000, temperature_k=20+273.15) # Dummy cold stream
interchanger.receive_input("hot_in", s_in)
interchanger.receive_input("cold_in", s_cold_in)

# Initialize registry mock if needed?
# logic uses LUT manager if available, else fallback
# We will rely on fallback for now (Antoine etc)

interchanger.initialize(dt=1.0, registry=None)
interchanger.step(t=0.0)

s_inter_out = interchanger.get_output("hot_out")
analyze_stream("Interchanger Output", s_inter_out)

if abs(s_inter_out.mass_flow_kg_h - s_in.mass_flow_kg_h) > 0.1:
    print(f"WARNING: Mass non-conservation in Interchanger! Delta: {s_inter_out.mass_flow_kg_h - s_in.mass_flow_kg_h:.4f} kg/h")

# Verify H2 Mass Conservation
h2_in_mass = s_in.mass_flow_kg_h * s_in.composition.get('H2', 0.0)
h2_out_mass = s_inter_out.mass_flow_kg_h * s_inter_out.composition.get('H2', 0.0)
if abs(h2_in_mass - h2_out_mass) > 1e-4:
     print(f"CRITICAL WARNING: H2 mass created/destroyed in Interchanger! Delta: {h2_out_mass - h2_in_mass:.6f} kg/h")
else:
     print(f"SUCCESS: H2 mass strictly conserved. Delta: {h2_out_mass - h2_in_mass:.6f} kg/h")

# 3. Setup DryCooler
dc = DryCooler("Test_DryCooler", use_central_utility=False, target_outlet_temp_c=40.0)
dc.initialize(dt=1.0, registry=None)
dc.receive_input("fluid_in", s_inter_out)
dc.step(t=0.0)

s_dc_out = dc.get_output("fluid_out")
analyze_stream("DryCooler Output", s_dc_out)

if abs(s_dc_out.mass_flow_kg_h - s_inter_out.mass_flow_kg_h) > 0.1:
    print(f"WARNING: Mass non-conservation in DryCooler! Delta: {s_dc_out.mass_flow_kg_h - s_inter_out.mass_flow_kg_h:.4f} kg/h")

