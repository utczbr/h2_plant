import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from h2_plant.components.reforming.integrated_atr_plant import IntegratedATRPlant
from h2_plant.config.constants_physics import LHV_CONSTANTS

class TestATREfficiency:
    
    @pytest.fixture
    def atr_plant(self):
        """Fixture to provide an initialized ATR plant."""
        config = {'ATR': {}} # Minimal config
        plant = IntegratedATRPlant('ATR_Test', config)
        
        # Mocking the DataManager to avoid needing real CSVs for physical properties
        plant.dm = MagicMock()
        # Mock heat duty lookups to return predictable values
        # Default to 0 values
        plant.dm.lookup.return_value = 0.0
        
        # Assign mock to data_manager property expected by ATRBaseComponent/IntegratedATRPlant
        plant.data_manager = plant.dm
        
        # Mock Rate Limiter to bypass dynamics
        plant.rate_limiter = MagicMock()
        plant.rate_limiter.update.side_effect = lambda rate, dt: rate
        
        # Manually set MW for calculation
        # Simplified MW values
        global MW
        # Usually MW is imported in the unit, but for testing context:
        MW = {'CH4': 16.04, 'H2': 2.016} 
        
        return plant

    def test_useful_heat_extraction(self, atr_plant):
        """Verify only negative (exothermic/cooling) duties from H05 and H09 count as useful heat."""
        
        # Ensure O2 flow is sufficient to trigger logic (avoid early return)
        atr_plant._buffer_oxygen_kmol = 10.0
        
        # Scenario 1: Both H05 and H09 are cooling (negative) -> Useful Heat
        # H05 = -500 kW, H09 = -200 kW
        def lookup_side_effect(func_name, val):
            if func_name == 'H05_Q_func': return -500.0
            if func_name == 'H09_Q_func': return -200.0
            return 10.0 # Other heaters
            
        atr_plant.dm.lookup.side_effect = lookup_side_effect
        
        # Run step
        atr_plant.dt = 1.0
        atr_plant.step(0.0)
        
        # Check q_useful
        # Should be |-500| + |-200| = 700
        assert atr_plant.q_useful_kw == 700.0
        
        # Check total heat duty (sum of all)
        # H01, H02, H04, H08 = 10.0 each (4 * 10 = 40)
        # H05 = -500, H09 = -200
        # Total = 40 - 700 = -660
        assert atr_plant.heat_duty_kw == pytest.approx(-660.0)

    def test_chemical_efficiency_calculation(self, atr_plant):
        """Verify chemical efficiency: Energy_Out_H2 / Energy_In_Biogas."""
        
        # Setup inputs
        mass_ch4_kg = 100.0
        moles_ch4 = mass_ch4_kg / 16.04
        atr_plant._buffer_biogas_kmol = moles_ch4 
        atr_plant._buffer_oxygen_kmol = 10.0
        atr_plant.dt = 1.0 
        
        # Target Output H2: 25 kg/h -> ~12.40 kmol/h
        moles_h2 = 25.0 / 2.016
        
        # Mock lookup to return H2 production when queried
        def lookup_side_effect(func_name, val):
            if func_name == 'F_H2_func': return moles_h2
            return 0.0
        atr_plant.dm.lookup.side_effect = lookup_side_effect
        
        # Run step
        atr_plant.step(0.0)
        
        # Energy calculations
        input_kw = (moles_ch4 * 16.04 * 50000.0) / 3600.0
        output_kw = (moles_h2 * 2.016 * 120000.0) / 3600.0
        
        expected_eff = output_kw / input_kw
        
        assert atr_plant.atr_efficiency_chemical == pytest.approx(expected_eff, rel=1e-3)
        
    def test_global_efficiency_calculation(self, atr_plant):
        """Verify global efficiency: (Energy_H2 + Q_useful) / (Energy_Biogas + P_el)."""
        
        mass_ch4_kg = 100.0
        moles_ch4 = mass_ch4_kg / 16.04
        atr_plant._buffer_biogas_kmol = moles_ch4
        atr_plant._buffer_oxygen_kmol = 10.0
        atr_plant.dt = 1.0
        
        moles_h2 = 25.0 / 2.016
        atr_plant.aux_power_kw = 100.0
        
        def lookup_side_effect(func_name, val):
            if func_name == 'H05_Q_func': return -500.0 # Useful
            if func_name == 'F_H2_func': return moles_h2 # H2 Prod
            return 0.0
        atr_plant.dm.lookup.side_effect = lookup_side_effect
        
        atr_plant.step(0.0)
        
        input_kw = (moles_ch4 * 16.04 * 50000.0) / 3600.0 # 1388.88
        output_kw = (moles_h2 * 2.016 * 120000.0) / 3600.0 # 833.33
        
        expected_global = (output_kw + 500.0) / (input_kw + 100.0)
        
        assert atr_plant.atr_efficiency_global == pytest.approx(expected_global, rel=1e-3)

