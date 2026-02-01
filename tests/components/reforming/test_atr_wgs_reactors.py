
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, ANY
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from h2_plant.components.reforming.atr_reactor import ATRReactor
from h2_plant.components.atr.wgs_reactor import WGSReactor
from h2_plant.core.stream import Stream

# Mock LUT Manager
class MockLUTManager:
    def lookup(self, species, prop, P, T):
        # Return dummy values
        if prop == 'H': # Enthalpy J/kg
            # Simple linear model for testing: Cp * T
            if species == 'H2': return 14300.0 * T 
            if species == 'CO': return 1040.0 * T
            if species == 'CO2': return 846.0 * T
            if species == 'H2O': return 1890.0 * T
            return 1000.0 * T
        return 0.0

@pytest.fixture
def mock_registry():
    registry = MagicMock()
    lut = MockLUTManager()
    registry.has.return_value = True
    registry.get.return_value = lut
    return registry

@pytest.fixture
def mock_atr_model():
    # Helper to create a dummy scipy-like object if needed, 
    # but we mock the OptimizedATRModel class or the pickle load directly.
    return {
        'F_H2_func': MagicMock(),
        'H01_Q_func': MagicMock(), 
        'H02_Q_func': MagicMock(), 
        'H04_Q_func': MagicMock(),
        'F_bio_func': MagicMock(), 
        'F_steam_func': MagicMock(), 
        'F_water_func': MagicMock()
    }

class TestWGSReactor:
    def test_initialization(self, mock_registry):
        wgs = WGSReactor('WGS_01')
        wgs.initialize(dt=1.0, registry=mock_registry)
        assert wgs.lut_manager is not None
        assert wgs._output_stream is not None

    def test_stoichiometry_and_equilibrium(self, mock_registry):
        wgs = WGSReactor('WGS_01')
        wgs.initialize(dt=1.0, registry=mock_registry)

        # Feed: 100 kmol CO, 150 kmol H2O (Excess Steam)
        # Mass flows
        flow_CO = 100 * 28.01
        flow_H2O = 150 * 18.015
        
        # Streams
        s1 = Stream(mass_flow_kg_h=flow_CO, composition={'CO': 1.0}, temperature_k=600, pressure_pa=3e5)
        s2 = Stream(mass_flow_kg_h=flow_H2O, composition={'H2O': 1.0}, temperature_k=600, pressure_pa=3e5)

        wgs.receive_input('syngas_in', s1, 'syngas')
        wgs.receive_input('in', s2, 'steam')

        # Run step
        wgs.step(0.0)

        # Check conversion
        # Equilibrium at 600K should be high for WGS (Exothermic)
        # Keq approx 10^(2073/600 - 2.029) = 10^(3.455 - 2.029) = 10^1.426 ~ 26
        # Reaction should proceed significantly
        assert wgs.CO_conversion > 0.5
        
        # Check mass balance
        # Mass in = Mass out
        mass_in = flow_CO + flow_H2O
        mock_out = wgs.get_output('out')
        mass_out = mock_out.mass_flow_kg_h
        
        assert abs(mass_in - mass_out) < 1.0 # Relaxed for floating point MW discrepancies
        
        # Check Exothermicity
        # Output temp should be higher than input (600K)
        assert wgs.outlet_temp_k > 600.0
        
    def test_no_flow(self, mock_registry):
        wgs = WGSReactor('WGS_01')
        wgs.initialize(dt=1.0, registry=mock_registry)
        wgs.step(0.0)
        assert wgs.outlet_flow_kg_h == 0.0
        assert wgs.get_output('out').mass_flow_kg_h == 0.0

def wgs_mw(species):
    mws = {'CO': 28.01, 'H2O': 18.015, 'CO2': 44.01, 'H2': 2.016, 'CH4': 16.04, 'N2': 28.014}
    return mws.get(species, 28.0)


class TestATRReactor:
    
    @patch('h2_plant.components.reforming.atr_reactor.OptimizedATRModel')
    @patch('builtins.open')
    @patch('pickle.load')
    def test_atr_logic(self, mock_pickle, mock_open, MockModelClass, mock_registry):
        # Mock the model instance
        mock_model_instance = MockModelClass.return_value
        
        # Define side_effect for get_outputs to simulate model behavior
        def get_outputs_side_effect(F_O2):
            # Simple linear model: 
            # 1 kmol O2 -> 2 kmol H2
            # 1 kmol O2 -> 1 kmol Biogas required
            # 1 kmol O2 -> 1 kmol Steam required
            return {
                'h2_production': F_O2 * 2.0,
                'total_heat_duty': -50.0,
                'biogas_required': F_O2 * 1.0,
                'steam_required': F_O2 * 1.0,
                'water_required': 0.0
            }
        mock_model_instance.get_outputs.side_effect = get_outputs_side_effect
        
        atr = ATRReactor('ATR_01', 1000.0)
        atr.initialize(dt=1.0, registry=mock_registry)
        
        # Ensure model was "loaded" (mocked)
        assert atr.model is not None
        
        # Test Step with sufficient feeds
        # Required: 50 O2 -> 50 Bio, 50 Steam
        atr.receive_input('o2_in', 50.0, 'oxygen') # 50 kmol/h
        atr.receive_input('biogas_in', 60.0, 'methane') # 60 kmol/h
        atr.receive_input('steam_in', 60.0, 'steam') # 60 kmol/h
        
        atr.step(0.0)
        
        # Should execute at full target rate (50 kmol O2)
        # H2 output = 50 * 2 = 100
        assert abs(atr.h2_production_kmol_h - 100.0) < 1e-2
        
        # Verify buffers decremented
        # Biogas: required 50. Start 60, End 10
        assert abs(atr.buffer_biogas_kmol - 10.0) < 1e-2
        
        # Test Stoichiometric Limiting
        # Reset
        atr.buffer_oxygen_kmol = 50.0
        atr.buffer_biogas_kmol = 25.0 # Limit = 0.5
        atr.buffer_steam_kmol = 100.0
        
        atr.step(0.0)
        
        # Actual O2 = 25. H2 = 50.
        assert abs(atr.h2_production_kmol_h - 50.0) < 1e-2
        
        # Biogas should be fully consumed
        assert abs(atr.buffer_biogas_kmol) < 1e-5

    def test_get_output_thermo(self, mock_registry):
        # Without mocking pickle, model load fails -> model is None.
        # Component should safely handle this (no production)
        # But get_output should still return a Stream object if there was buffered output
        
        atr = ATRReactor('ATR_01', 1000.0)
        atr.initialize(dt=1.0, registry=mock_registry)
        atr._h2_output_buffer_kmol = 10.0
        
        out = atr.get_output('h2_out')
        assert isinstance(out, Stream)
        assert out.temperature_k == 900.0
        assert out.phase == 'gas'
        # With MockLUT, enthalpy implies success
        
