import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append("/home/stuart/Documentos/Planta Hidrogenio")

from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.core.stream import Stream

def run_test():
    print("### Mass Gain Reproduction Test ###")
    
    # 1. SOEC with tiny entrainment (simulated mist) to match user observation
    # 1.0 kg diff on 1110 kg flow => ~0.0009 fraction
    soec_config = {
        'num_modules': 6,
        'max_power_nominal_mw': 2.4,
        'optimal_limit': 0.80,
        'steam_input_ratio_kg_per_kg_h2': 10.3,
        'entrained_water_fraction': 0.0009, # Hypothesized cause
        'out_pressure_pa': 101325.0
    }
    soec = SOECOperator(soec_config)
    class MockRegistry:
        def has(self, x): return False
        def get(self, x): return None
    soec.initialize(dt=1.0, registry=MockRegistry())
    
    # 2. Interchanger
    interchanger = Interchanger("Interchanger")
    interchanger.initialize(dt=1.0, registry=MockRegistry())
    
    # 3. Dry Cooler
    dry_cooler = DryCooler("DryCooler")
    dry_cooler.initialize(dt=1.0, registry=MockRegistry())

    # Step SOEC (Low power to match ~1110 kg/h total flow)
    # Rated flow ~4900 kg/h at 100% capacity (14MW).
    # Need ~1110 kg/h -> ~22% capacity (~3MW)
    soec.receive_input('power_in', 3.0) 
    soec.receive_input('water_in', Stream(mass_flow_kg_h=2000.0, composition={'H2O': 1.0}))
    soec.step(t=0.0)
    
    h2_out = soec.get_output('h2_out')
    print(f"\n[SOEC Out] Mass: {h2_out.mass_flow_kg_h:.4f} kg/h")
    print(f"           Entrained: {h2_out.extra.get('m_dot_H2O_liq_accomp_kg_s', 0)*3600:.4f} kg/h")
    
    # Step Interchanger
    cold_in = Stream(2000.0, 298.15, 101325.0, {'H2O': 1.0}, 'liquid')
    interchanger.receive_input('hot_in', h2_out)
    interchanger.receive_input('cold_in', cold_in)
    interchanger.step(t=0.0)
    
    ic_out = interchanger.get_output('hot_out')
    print(f"\n[Intercharger Out] Mass: {ic_out.mass_flow_kg_h:.4f} kg/h")
    print(f"                   Extra Preserved? {ic_out.extra.get('m_dot_H2O_liq_accomp_kg_s', 0)*3600:.4f} kg/h")
    
    # Step Dry Cooler
    dry_cooler.receive_input('fluid_in', ic_out)
    dry_cooler.step(t=0.0)
    
    dc_out = dry_cooler.get_output('fluid_out')
    print(f"\n[DryCooler Out] Mass: {dc_out.mass_flow_kg_h:.4f} kg/h")
    
    print(f"\n### Discrepancy Analysis ###")
    delta = dc_out.mass_flow_kg_h - ic_out.mass_flow_kg_h
    print(f"Intercharger -> DryCooler Mass Delta: {delta:.4f} kg/h")
    
    internal_liq = ic_out.extra.get('m_dot_H2O_liq_accomp_kg_s', 0.0) * 3600.0
    print(f"Entrained Liquid in Stream: {internal_liq:.4f} kg/h")
    
    if abs(delta - internal_liq) < 0.001:
        print("CONFIRMED: Discrepancy equals Entrained Liquid mass being merged into bulk.")
    else:
        print("FAILED: Discrepancy does not match entrainment.")

if __name__ == "__main__":
    run_test()
