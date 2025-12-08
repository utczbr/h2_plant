
import pytest
import sys
import os
sys.path.append(os.getcwd())

from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.core.stream import Stream
from h2_plant.core.constants import DryCoolerConstants as DCC

def test_dry_cooler_h2_sizing():
    """Verify Dry Cooler auto-sizing and fan power for H2 service."""
    dc = DryCooler("dry_cooler_h2")
    dc.initialize(1.0, None)
    
    # Create H2-rich stream
    # Mass flow doesn't affect fan power (fixed design) but needed for step()
    h2_stream = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=353.15, # 80C
        pressure_pa=30e5,
        composition={'H2': 0.9, 'H2O': 0.1}
    )
    
    dc.receive_input("fluid_in", h2_stream)
    dc.step(0.0)
    
    # Check Geometry & Airflow
    assert dc.fluid_type == "H2"
    assert dc.area_m2 == DCC.AREA_H2_M2
    assert dc.design_air_flow_kg_s == DCC.MDOT_AIR_DESIGN_H2_KG_S
    
    # Check Fan Power
    # Exp: 872 W = 0.872 kW
    # Allow small float tolerance
    print(f"H2 Fan Power: {dc.fan_power_kw:.4f} kW")
    assert abs(dc.fan_power_kw - 0.872) < 0.0022

def test_dry_cooler_o2_sizing():
    """Verify Dry Cooler auto-sizing and fan power for O2 service."""
    dc = DryCooler("dry_cooler_o2")
    dc.initialize(1.0, None)
    
    # Create O2-rich stream
    o2_stream = Stream(
        mass_flow_kg_h=800.0,
        temperature_k=353.15, # 80C
        pressure_pa=30e5,
        composition={'O2': 0.9, 'H2O': 0.1}
    )
    
    dc.receive_input("fluid_in", o2_stream)
    dc.step(0.0)
    
    # Check Geometry & Airflow
    assert dc.fluid_type == "O2"
    assert dc.area_m2 == DCC.AREA_O2_M2
    assert dc.design_air_flow_kg_s == DCC.MDOT_AIR_DESIGN_O2_KG_S
    
    # Check Fan Power
    # Exp: 96.6 kW
    print(f"O2 Fan Power: {dc.fan_power_kw:.4f} kW")
    assert abs(dc.fan_power_kw - 96.6) < 0.1

if __name__ == "__main__":
    test_dry_cooler_h2_sizing()
    test_dry_cooler_o2_sizing()
    print("All Dry Cooler verification tests passed.")
