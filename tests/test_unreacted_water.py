import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.components.production.pem_electrolyzer_detailed import DetailedPEMElectrolyzer
from h2_plant.components.production.soec_electrolyzer_detailed import DetailedSOECElectrolyzer
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.constants_physics import PEMConstants, SOECConstants

def test_unreacted_water():
    print("Testing Unreacted Water Logic...")
    
    # 1. Test PEM
    print("\n--- PEM Test ---")
    pem = DetailedPEMElectrolyzer(max_power_mw=1.0)
    registry = ComponentRegistry()
    
    from h2_plant.core.component import Component
    
    # Mock lut_manager
    class MockLUT(Component):
        def initialize(self, dt, registry): pass
        def step(self, t): pass
        def get_state(self): return {}
        def get_degradation_voltage(self, t): return 0.0
    registry.register("lut_manager", MockLUT())
    
    pem.initialize(dt=1.0, registry=registry)
    
    # Mock coordinator
    class MockCoordinator(Component):
        pem_setpoint_mw = 1.0
        def initialize(self, dt, registry): pass
        def step(self, t): pass
        def get_state(self): return {}
    registry.register("dual_path_coordinator", MockCoordinator())
    
    pem.step(0)
    output = pem.get_output("h2_out")
    
    print(f"PEM Output Mass Flow: {output.mass_flow_kg_h:.4f} kg/h")
    print(f"PEM Composition: {output.composition}")
    
    h2_frac = output.composition.get('H2', 0)
    h2o_frac = output.composition.get('H2O', 0)
    
    expected_water_frac = PEMConstants().unreacted_water_fraction
    print(f"Expected Water Fraction: {expected_water_frac}")
    
    if abs(h2o_frac - expected_water_frac) < 1e-4:
        print("✅ PEM Water Fraction Correct")
    else:
        print(f"❌ PEM Water Fraction Incorrect: {h2o_frac}")

    # 2. Test SOEC
    print("\n--- SOEC Test ---")
    registry_soec = ComponentRegistry() # New registry
    soec = DetailedSOECElectrolyzer(max_power_kw=1000.0)
    soec.initialize(dt=1.0, registry=registry_soec)
    
    # Mock coordinator
    class MockCoordinatorSOEC(Component):
        soec_setpoint_mw = 1.0
        def initialize(self, dt, registry): pass
        def step(self, t): pass
        def get_state(self): return {}
    registry_soec.register("dual_path_coordinator", MockCoordinatorSOEC()) # Use different ID or overwrite
    # Actually SOEC looks for "dual_path_coordinator" too.
    # I should overwrite it.
    # registry.register("dual_path_coordinator", MockCoordinatorSOEC()) # Overwrite logic? Registry raises DuplicateComponentError.
    # I need to create a new registry or handle this.
    # Let's create a new registry for SOEC test.
    
    # Mock steam input
    soec.receive_input("steam_in", type('obj', (object,), {'mass_flow_kg_h': 1000.0}), "water")
    
    soec.step(0)
    output = soec.get_output("h2_out")
    
    print(f"SOEC Output Mass Flow: {output.mass_flow_kg_h:.4f} kg/h")
    print(f"SOEC Composition: {output.composition}")
    
    h2_frac = output.composition.get('H2', 0)
    h2o_frac = output.composition.get('H2O', 0)
    
    expected_water_frac = SOECConstants().UNREACTED_WATER_FRACTION
    print(f"Expected Water Fraction: {expected_water_frac}")
    
    if abs(h2o_frac - expected_water_frac) < 1e-4:
        print("✅ SOEC Water Fraction Correct")
    else:
        print(f"❌ SOEC Water Fraction Incorrect: {h2o_frac}")

if __name__ == "__main__":
    test_unreacted_water()
