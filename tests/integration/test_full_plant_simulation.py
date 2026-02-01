
import pytest
import numpy as np
from unittest.mock import MagicMock
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.components.compression.compressor import CompressorStorage
from h2_plant.config.constants_physics import PEMConstants

def test_full_green_hydrogen_chain():
    """
    Test the complete Green Hydrogen production chain:
    Water Sources -> Mixer -> Pump -> PEM Electrolyzer -> Compressor -> Storage
    
    Verifies:
    1. Mass Balance across the chain.
    2. Pressure propagation.
    3. Correct transformation of streams (Water -> H2).
    4. Component interaction via 'receive_input' and 'get_output'.
    """
    
    # 1. Setup Registry and Shared Resources
    registry = ComponentRegistry()
    
    # LUT Manager (Essential for performance)
    lut = LUTManager()
    lut.initialize(dt=1.0, registry=registry)
    registry.register('lut_manager', lut)
    
    # 2. Initialize Components
    
    # A. Water Mixer (Fresh Water + Recycled Water)
    mixer = WaterMixer(outlet_pressure_kpa=101.325) # 1 atm output
    mixer.set_component_id("mixer_01")
    mixer.initialize(1.0, registry)
    registry.register(mixer.component_id, mixer)
    
    # B. Water Pump (Boost to 40 bar for PEM)
    pump = WaterPumpThermodynamic(
        pump_id="feed_pump_01",
        target_pressure_pa=40e5, # 40 bar in Pa
        eta_is=0.75
    )
    pump.set_component_id("feed_pump_01")
    pump.initialize(1.0, registry)
    registry.register(pump.component_id, pump)
    
    # C. PEM Electrolyzer (Consumers Water, Produces H2)
    pem_config = {
        'max_power_mw': 5.0,
        'base_efficiency': 0.65,
        'water_excess_factor': 0.0, # Exact consumption for easier mass balance check
        'use_polynomials': False    # Use solver for accuracy
    }
    pem = DetailedPEMElectrolyzer(pem_config)
    pem.set_component_id("pem_stack_01")
    pem.initialize(1.0, registry)
    registry.register(pem.component_id, pem)
    
    # D. Compressor (Compresses H2 from 30 bar to 500 bar)
    # Note: PEM outputs at 30 bar (via back-pressure regulator in get_output)
    compressor = CompressorStorage(
        max_flow_kg_h=200.0, # Limit
        inlet_pressure_bar=30.0,
        outlet_pressure_bar=500.0,
        inlet_temperature_c=80.0, # From PEM
        max_temperature_c=90.0
    )
    compressor.set_component_id("compressor_01")
    compressor.initialize(1.0, registry)
    registry.register(compressor.component_id, compressor)
    
    # 3. Simulate One Timestep
    
    # --- Step 1: Supply Water to Mixer ---
    stream_fresh = Stream(
        mass_flow_kg_h=800.0, 
        temperature_k=293.15, # 20C
        pressure_pa=101325.0, # 1 atm
        composition={'H2O': 1.0}
    )
    stream_recycled = Stream(
        mass_flow_kg_h=200.0, 
        temperature_k=303.15, # 30C
        pressure_pa=101325.0,
        composition={'H2O': 1.0}
    )
    
    mixer.receive_input('in_1', stream_fresh, 'water')
    mixer.receive_input('in_2', stream_recycled, 'water')
    
    mixer.step(0.0)
    
    # Verify Mixer Output
    mixed_water = mixer.get_output('outlet')
    assert mixed_water.mass_flow_kg_h == 1000.0
    # Mixing Temp: (800*20 + 200*30)/1000 = 22C (approx, Cp is const-ish)
    assert 295.0 < mixed_water.temperature_k < 296.0 
    
    # --- Step 2: Pump Mixed Water ---
    pump.receive_input('water_in', mixed_water, 'water')
    pump.step(0.0)
    
    pumped_water = pump.get_output('water_out')
    
    # Verify Pump Output
    assert abs(pumped_water.pressure_pa - 40e5) < 1e3 # 40 bar
    assert pumped_water.temperature_k > mixed_water.temperature_k # slight heating
    assert pumped_water.mass_flow_kg_h == 1000.0
    
    # --- Step 3: Electrolysis ---
    # Set Power for PEM
    pem.set_power_input_mw(4.0) # 80% load
    
    # Feed Water
    pem.receive_input('water_in', pumped_water, 'water')
    pem.step(0.0)
    
    # Verify PEM Output
    h2_out = pem.get_output('h2_out')
    o2_out = pem.get_output('oxygen_out')
    
    # Approx 4MW -> ~70-80 kg/h H2 depending on efficiency
    # 1 kg H2 = ~50 kWh (LHV) / efficiency
    print(f"H2 Produced: {h2_out.mass_flow_kg_h:.2f} kg/h")
    assert h2_out.mass_flow_kg_h > 50.0 
    
    # Mass Balance Check (Water -> H2 + O2)
    # H2O molar mass 18, H2 2, O 16. Ratio H2 is 2/18 = 1/9.
    # So 1 kg H2O -> 1/9 kg H2 + 8/9 kg O2
    # In PEM, water is consumed.
    consumed_water_kg = pem.water_consumption_kg # This is per timestep (1h) -> kg
    produced_h2_kg = pem.h2_output_kg
    produced_o2_kg = pem.o2_output_kg
    
    # Water Consumed should roughly match stoichiometric req for produced H2
    # produced_h2 * 9 = water_consumed (approx)
    stoich_water = produced_h2_kg * (18.015/2.016)
    assert abs(consumed_water_kg - stoich_water) < 1.0 # Tolerance
    
    # Verify H2 Stream Properties
    assert h2_out.phase == 'gas'
    assert h2_out.pressure_pa == 30e5 # 30 bar reg output
    # Expected purity ~97% (3% water carryover defined in PEMConstants)
    assert h2_out.composition['H2'] > 0.96 
    
    # --- Step 4: Compression ---
    # Feed PEM H2 to Compressor
    compressor.receive_input('h2_in', h2_out, 'hydrogen')
    
    # Compressor needs Mass to be transferred explicitly in its current implementation?
    # Checking compressor.py: receive_input updates self.transfer_mass_kg
    # and mixes temperature.
    
    compressor.step(0.0)
    
    # Verify Compressor Output
    compressed_h2 = compressor.get_output('h2_out')
    
    # Pressure should be 500 bar
    assert abs(compressed_h2.pressure_pa - 500e5) < 1e4
    
    # Mass should be conserved 
    # (CompressorStorage might accumulate if demand is lower? 
    # But usually it passes through if below max_flow)
    assert abs(compressed_h2.mass_flow_kg_h - h2_out.mass_flow_kg_h) < 1e-3
    
    # Energy check
    energy_kwh_kg = compressor.specific_energy_kwh_kg
    # Should be around 2.5-2.7 for 30->500 bar @ 80C inlet
    # Physics Note: High inlet T (80C) increases specific work substantially vs 10C.
    print(f"Compressor Specific Energy: {energy_kwh_kg:.4f} kWh/kg")
    assert 2.0 < energy_kwh_kg < 3.0
    
    print("\nIntegration Test PASSED: Full Green H2 Chain Validated.")

if __name__ == "__main__":
    test_full_green_hydrogen_chain()
