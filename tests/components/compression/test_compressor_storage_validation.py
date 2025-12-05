"""
Validation tests for CompressorStorage refactoring.

Compares the new implementation against the legacy reference code
to ensure exact preservation of physics and calculations.

Run with: pytest tests/components/compression/test_compressor_storage_validation.py -v -s
"""

import pytest
import numpy as np

# Try to import CoolProp
try:
    import CoolProp.CoolProp as CP
    PropsSI = CP.PropsSI
    USING_REAL_COOLPROP = True
except ImportError:
    CP = None
    USING_REAL_COOLPROP = False

from h2_plant.components.compression.compressor_storage import CompressorStorage
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.optimization.lut_manager import LUTManager, LUTConfig


# --- LEGACY CODE (EXACT COPY FROM Compressor Armazenamento.py) ---

T_IN_C = 10.0
T_IN_K = T_IN_C + 273.15 
ETA_C = 0.65
T_MAX_C = 85.0
T_MAX_K = T_MAX_C + 273.15
COP = 3.0
P_TO_PA = 1e5
J_PER_KG_TO_KWH_PER_KG = 2.7778e-7
FLUIDO = 'H2'

def legacy_calculate_compression_energy(P_in_bar, P_out_bar):
    """
    Legacy calculation from Compressor Armazenamento.py.
    Returns (energy_kwh_per_kg, num_stages).
    """
    if not USING_REAL_COOLPROP:
        pytest.skip("CoolProp not available - cannot validate")
    
    P_in_Pa = P_in_bar * P_TO_PA
    P_out_Pa = P_out_bar * P_TO_PA
    
    try:
        h1_val = PropsSI('H', 'P', P_in_Pa, 'T', T_IN_K, FLUIDO)
        s1_val = PropsSI('S', 'P', P_in_Pa, 'T', T_IN_K, FLUIDO)

        P_out_1s_max_T = PropsSI('P', 'S', s1_val, 'T', T_MAX_K, FLUIDO)
        r_stage_max_isentropic = P_out_1s_max_T / P_in_Pa
        r_stage_max_isentropic = max(2.0, r_stage_max_isentropic) 
        r_total = P_out_Pa / P_in_Pa
        N_stages = int(np.ceil(np.log(r_total) / np.log(r_stage_max_isentropic)))
        N_stages = max(1, N_stages)
        
        W_compression_total = 0.0
        Q_removed_total = 0.0
        P_current = P_in_Pa
        r_stage = r_total**(1/N_stages)
        
        for i in range(N_stages):
            P_out_stage = P_current * r_stage
            if i == N_stages - 1: 
                P_out_stage = P_out_Pa

            h2s = PropsSI('H', 'P', P_out_stage, 'S', s1_val, FLUIDO)
            Ws = h2s - h1_val
            Wa = Ws / ETA_C
            h2a = h1_val + Wa
            W_compression_total += Wa
            
            if i < N_stages - 1:
                h_cooled = PropsSI('H', 'P', P_out_stage, 'T', T_IN_K, FLUIDO)
                Q_removed = h2a - h_cooled
                Q_removed_total += Q_removed
                P_current = P_out_stage
        
        W_chilling_total = Q_removed_total / COP
        W_total_J_per_kg = W_compression_total + W_chilling_total
        W_total_kWh_per_kg = W_total_J_per_kg * J_PER_KG_TO_KWH_PER_KG
        
        return W_total_kWh_per_kg, N_stages
        
    except Exception as e:
        pytest.fail(f"Legacy calculation failed: {e}")


# --- FIXTURES ---

@pytest.fixture(scope="module")
def registry_with_lut():
    """Create registry with initialized LUT manager."""
    registry = ComponentRegistry()
    
    # Initialize LUT manager (will cache tables)
    lut_config = LUTConfig(
        pressure_min=1e5,
        pressure_max=900e5,
        temperature_min=273.15,
        temperature_max=400.0
    )
    lut_manager = LUTManager(config=lut_config)
    lut_manager.set_component_id('lut_manager')
    registry.register(ComponentID.LUT_MANAGER, lut_manager)
    lut_manager.initialize(dt=1.0, registry=registry)
    
    return registry


# --- VALIDATION TESTS ---

@pytest.mark.skipif(not USING_REAL_COOLPROP, reason="CoolProp required")
class TestCompressorStorageValidation:
    """Validation against legacy implementation."""
    
    def test_charging_scenario(self, registry_with_lut):
        """
        Test charging scenario: 40 → 140 bar.
        
        Expected from legacy: ~1.2556 kWh/kg, 2 stages
        """
        p_in_bar = 40.0
        p_out_bar = 140.0
        
        print(f"\n{'='*70}")
        print(f"Testing: Charging (Enchimento)")
        print(f"Pressure: {p_in_bar:.0f} → {p_out_bar:.0f} bar")
        print(f"{'='*70}")
        
        # Legacy
        legacy_energy, legacy_stages = legacy_calculate_compression_energy(p_in_bar, p_out_bar)
        print(f"Legacy: {legacy_energy:.6f} kWh/kg | {legacy_stages} stages")
        
        # New implementation
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=p_in_bar,
            outlet_pressure_bar=p_out_bar,
            inlet_temperature_c=T_IN_C,
            max_temperature_c=T_MAX_C,
            isentropic_efficiency=ETA_C,
            chiller_cop=COP
        )
        compressor.set_component_id('compressor_charging')
        compressor.initialize(dt=1.0, registry=registry_with_lut)
        
        # Process 1 kg
        compressor.transfer_mass_kg = 1.0
        compressor.step(t=0.0)
        
        new_energy = compressor.specific_energy_kwh_kg
        new_stages = compressor.num_stages
        
        print(f"New:    {new_energy:.6f} kWh/kg | {new_stages} stages")
        
        # Validation
        energy_diff_pct = abs(legacy_energy - new_energy) / legacy_energy * 100
        print(f"Difference: {energy_diff_pct:.4f}%")
        
        assert new_stages == legacy_stages, f"Stage mismatch: {new_stages} vs {legacy_stages}"
        assert energy_diff_pct < 1.0, f"Energy difference too large: {energy_diff_pct:.2f}%"
        
        print("✅ VALIDATION PASSED")
    
    def test_discharging_scenario(self, registry_with_lut):
        """
        Test discharging scenario: 50 → 500 bar.
        
        Expected from legacy: ~3.7287 kWh/kg, 3 stages
        """
        p_in_bar = 50.0
        p_out_bar = 500.0
        
        print(f"\n{'='*70}")
        print(f"Testing: Discharging (Esvaziamento)")
        print(f"Pressure: {p_in_bar:.0f} → {p_out_bar:.0f} bar")
        print(f"{'='*70}")
        
        # Legacy
        legacy_energy, legacy_stages = legacy_calculate_compression_energy(p_in_bar, p_out_bar)
        print(f"Legacy: {legacy_energy:.6f} kWh/kg | {legacy_stages} stages")
        
        # New implementation
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=p_in_bar,
            outlet_pressure_bar=p_out_bar,
            inlet_temperature_c=T_IN_C,
            max_temperature_c=T_MAX_C,
            isentropic_efficiency=ETA_C,
            chiller_cop=COP
        )
        compressor.set_component_id('compressor_discharging')
        compressor.initialize(dt=1.0, registry=registry_with_lut)
        
        # Process 1 kg
        compressor.transfer_mass_kg = 1.0
        compressor.step(t=0.0)
        
        new_energy = compressor.specific_energy_kwh_kg
        new_stages = compressor.num_stages
        
        print(f"New:    {new_energy:.6f} kWh/kg | {new_stages} stages")
        
        # Detailed breakdown
        state = compressor.get_state()
        print(f"\nDetailed Breakdown:")
        print(f"  Compression Work:  {state['compression_work_kwh']:.6f} kWh")
        print(f"  Chilling Work:     {state['chilling_work_kwh']:.6f} kWh")
        print(f"  Heat Removed:      {state['heat_removed_kwh']:.6f} kWh")
        print(f"  Stage Ratio:       {state['stage_pressure_ratio']:.3f}")
        
        # Validation
        energy_diff_pct = abs(legacy_energy - new_energy) / legacy_energy * 100
        print(f"Difference: {energy_diff_pct:.4f}%")
        
        assert new_stages == legacy_stages, f"Stage mismatch: {new_stages} vs {legacy_stages}"
        assert energy_diff_pct < 1.0, f"Energy difference too large: {energy_diff_pct:.2f}%"
        
        print("✅ VALIDATION PASSED")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

