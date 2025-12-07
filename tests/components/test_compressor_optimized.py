
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.components.compression.compressor import CompressorStorage
from h2_plant.core.component_registry import ComponentRegistry

def test_lut_isentropic_lookup():
    """Verify LUTManager can perform H(P,S) lookup."""
    lut = LUTManager()
    # Mock initialization and table generation for speed
    lut._initialized = True
    lut.config.properties = ('H',)
    
    # Create fake grids
    lut._pressure_grid = np.array([1e5, 2e5, 3e5])
    lut._entropy_grid = np.array([20000.0, 30000.0, 40000.0])
    
    # Create fake H table [P, S]
    # Simple function H = P + S for testing interpolation
    h_table = np.zeros((3, 3))
    for i, p in enumerate(lut._pressure_grid):
        for j, s in enumerate(lut._entropy_grid):
            h_table[i, j] = p + s
            
    lut._luts = {'TEST': {'H_from_PS': h_table}}
    
    # Test valid lookup
    # P=1.5e5 (midpoint), S=25000 (midpoint)
    # Expected H = 1.5e5 + 25000 = 175000
    val = lut.lookup_isentropic_enthalpy('TEST', 1.5e5, 25000.0)
    assert abs(val - 175000.0) < 1.0

def test_compressor_optimization_integration():
    """Verify Compressor uses LUTManager for isentropic step."""
    registry = ComponentRegistry()
    lut = MagicMock(spec=LUTManager)
    
    # Mock lookups
    # P_in=30bar, T_in=298K
    lut.lookup.return_value = 1000.0 # Standard property return
    lut.lookup_isentropic_enthalpy.return_value = 2000.0 # H2s
    
    registry.register('lut_manager', lut)
    
    comp = CompressorStorage(max_flow_kg_h=100.0, inlet_pressure_bar=30.0, outlet_pressure_bar=60.0)
    comp.initialize(1.0, registry)
    
    # One stage approx since 30->60 bar
    
    comp.transfer_mass_kg = 10.0
    comp.step(0.0)
    
    # Verify lookup_isentropic_enthalpy was called
    # Compressor logic: 
    # 1. lookup S_in (lut.lookup)
    # 2. lookup H2s (lut.lookup_isentropic_enthalpy)
    assert lut.lookup_isentropic_enthalpy.call_count >= 1
    
    # Verify calculation happened
    assert comp.energy_consumed_kwh > 0.0

if __name__ == "__main__":
    pytest.main([__file__])
