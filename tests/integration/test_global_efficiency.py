import pytest
import numpy as np
import logging
from unittest.mock import MagicMock, Mock
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy, IntegratedDispatchState
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import SimulationConfig
from h2_plant.config.models import SimulationContext
from h2_plant.components.reforming.integrated_atr_plant import IntegratedATRPlant

# Validates compliance with Eq 5.42:
# eta_global = (m_H2_total * LHV_H2) / (P_el_total + Energy_Biogas)

class TestGlobalEfficiency:
    
    @pytest.fixture
    def strategy(self):
        """Initialize the strategy with mocked dependencies."""
        strat = HybridArbitrageEngineStrategy()
        
        # Mock Registry
        strat._registry = ComponentRegistry()
        
        # Mock Context
        context = MagicMock(spec=SimulationContext)
        context.simulation = MagicMock(spec=SimulationConfig)
        context.simulation.timestep_hours = 1.0 # 1 hour timestep for simple arithmetic
        strat._context = context
        
        # Mock History Structure
        strat._history = {
            'integrated_global_efficiency': np.zeros(10),
            'H2_soec_kg': np.zeros(10),
            'H2_pem_kg': np.zeros(10),
            'H2_atr_kg': np.zeros(10),
            'P_soec_actual': np.zeros(10), # MW
            'P_pem': np.zeros(10), # MW
            'P_bop_mw': np.zeros(10)
        }
        strat._total_steps = 10
        strat._state = IntegratedDispatchState()
        
        return strat

    def test_baseline_electrolyzer_efficiency(self, strategy):
        """
        Scenario 1: Electrolyzer Only (No Biogas, No ATR).
        Checks if Efficiency = (H2 Energy) / (Elec Energy).
        """
        step = 0
        
        # INPUTS
        # Produce 10 kg H2 total
        strategy._history['H2_soec_kg'][step] = 10.0
        
        # LHV H2 = 33.33 kWh/kg
        # Energy Out = 10 * 33.33 = 333.3 kW
        
        # Power Consumption
        # 0.4 MW = 400 kW
        strategy._history['P_soec_actual'][step] = 0.4 
        
        # Expected Efficiency = 333.3 / 400 = 0.833
        
        # RUN
        strategy._calculate_integrated_efficiency(step)
        
        # ASSERT
        result = strategy._history['integrated_global_efficiency'][step]
        expected = (10.0 * 33.33) / 400.0
        
        assert result == pytest.approx(expected, rel=1e-3)

    def test_hybrid_atr_integration(self, strategy):
        """
        Scenario 2: Hybrid (PEM + ATR).
        Verifies aggregation of H2 from multiple sources and Energy Inputs (Elec + Biogas).
        """
        step = 0
        
        # 1. H2 Production (PEM + ATR)
        strategy._history['H2_pem_kg'][step] = 5.0
        strategy._history['H2_atr_kg'][step] = 15.0 # Total 20 kg
        
        # Energy Out = 20 * 33.33 = 666.6 kW
        
        # 2. Electricity Input (PEM + BOP)
        strategy._history['P_pem'][step] = 0.25 # 250 kW
        strategy._history['P_bop_mw'][step] = 0.05 # 50 kW
        # Total Elec = 300 kW
        
        # 3. Biogas Input (Dynamic LHV)
        # Mock ATR component in registry
        atr_mock = Mock()
        atr_mock.get_state.return_value = {'biogas_energy_input_kw': 500.0}
        strategy._atr = atr_mock # Inject reference
        
        # Total Energy In = 300 (Elec) + 500 (Biogas) = 800 kW
        
        # Expected Efficiency = 666.6 / 800 = 0.833
        
        strategy._calculate_integrated_efficiency(step)
        
        result = strategy._history['integrated_global_efficiency'][step]
        expected = (20.0 * 33.33) / 800.0
        
        assert result == pytest.approx(expected, rel=1e-3)

    def test_auxiliary_power_aggregation(self, strategy):
        """
        Scenario 3: Validation of Auxiliary Power Aggregation (Principle 2).
        Ensures P_el_total sums tracked auxiliaries (Compressors, Cooling, etc).
        """
        step = 0
        
        # H2 Prod
        strategy._history['H2_pem_kg'][step] = 10.0 # 333.3 kW out
        
        # Elec Cons
        strategy._history['P_pem'][step] = 0.3 # 300 kW
        
        # Mock Auxiliaries in Registry
        # 1. Compressor (power_kw)
        comp_mock = Mock()
        comp_mock.get_state.return_value = {'power_kw': 10.0}
        
        # 2. Dry Cooler (fan_power_kw)
        fan_mock = Mock()
        fan_mock.get_state.return_value = {'fan_power_kw': 5.0}
        
        # 3. Chiller (electrical_power_kw)
        chiller_mock = Mock()
        chiller_mock.get_state.return_value = {'electrical_power_kw': 15.0}

        # 4. Storage Tank (No power)
        tank_mock = Mock()
        tank_mock.get_state.return_value = {'inventory_kg': 100}
        
        strategy._registry.list_components = MagicMock(return_value=[
            ('Comp', comp_mock), 
            ('Fan', fan_mock), 
            ('Chiller', chiller_mock),
            ('Tank', tank_mock)
        ])
        
        # Total Elec = 300 (Stack) + 10 + 5 + 15 = 330 kW
        
        strategy._calculate_integrated_efficiency(step)
        
        result = strategy._history['integrated_global_efficiency'][step]
        expected = (10.0 * 33.33) / 330.0
        
        assert result == pytest.approx(expected, rel=1e-3)

    def test_dynamic_biogas_lhv(self, strategy):
        """
        Scenario 4: Variable Biogas Energy.
        Verifies that changes in biogas_energy_input_kw directly affect the denominator.
        """
        step = 0
        strategy._history['H2_atr_kg'][step] = 10.0 # 333.3 kW
        
        # Case A: Low Biogas Energy (Lean Biogas)
        atr_mock = Mock()
        atr_mock.get_state.return_value = {'biogas_energy_input_kw': 400.0}
        strategy._atr = atr_mock
        
        strategy._calculate_integrated_efficiency(step)
        eff_A = strategy._history['integrated_global_efficiency'][step] # 333.3 / 400 = 0.833
        
        # Case B: High Biogas Energy (Rich Biogas) -> Should lower efficiency for same H2
        atr_mock.get_state.return_value = {'biogas_energy_input_kw': 600.0}
        strategy._calculate_integrated_efficiency(step)
        eff_B = strategy._history['integrated_global_efficiency'][step] # 333.3 / 600 = 0.555
        
        assert eff_A > eff_B
