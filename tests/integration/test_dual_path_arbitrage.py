
import pytest
import os
import pandas as pd
import numpy as np
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.components.mixing.water_mixer import WaterMixer
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.compression.compressor import CompressorStorage
from h2_plant.data.price_loader import EnergyPriceLoader

# Arbitrage Logic Constants
PRICE_THRESHOLD_PEM = 50.0   # EUR/MWh - Cheap power
PRICE_THRESHOLD_SOEC = 80.0  # EUR/MWh - Moderate power (higher eff)
WIND_THRESHOLD = 5.0         # MW

def test_dual_path_arbitrage_simulation():
    """
    Simulates a dual-path plant:
    Line 1: Water -> Mixer0 -> Pump0 -> PEM
    Line 2: Water -> Mixer2 -> Pump1 -> SOEC
    Merge:  PEM_H2 + SOEC_H2 -> Mixer1 -> Compressor0
    
    Includes simple arbitrage logic based on mocked price/wind data.
    """
    print("\nStarting Dual Path Simulation (24h Real Data)...")
    
    # 1. Setup Registry
    registry = ComponentRegistry()
    lut = LUTManager()
    lut.initialize(dt=1.0, registry=registry)
    registry.register('lut_manager', lut)
    
    # 2. Instantiate Components
    
    # --- Line 1 (PEM - 5MW) ---
    mixer_0 = WaterMixer(outlet_pressure_kpa=101.325, fluid_type='Water')
    mixer_0.set_component_id("mixer_0")
    mixer_0.initialize(1.0, registry)
    registry.register("mixer_0", mixer_0)
    
    pump_0 = WaterPumpThermodynamic(pump_id="pump_0", target_pressure_pa=40e5, eta_is=0.75)
    pump_0.set_component_id("pump_0")
    pump_0.initialize(1.0, registry)
    registry.register("pump_0", pump_0)
    
    pem_config = {
        'max_power_mw': 5.0,
        'base_efficiency': 0.65
    }
    pem = DetailedPEMElectrolyzer(pem_config)
    pem.set_component_id("pem")
    pem.initialize(1.0, registry)
    registry.register("pem", pem)
    
    # --- Line 2 (SOEC - 14.4MW) ---
    # 6 Modules * 2.4 MW = 14.4 MW
    mixer_2 = WaterMixer(outlet_pressure_kpa=101.325, fluid_type='Water')
    mixer_2.set_component_id("mixer_2")
    mixer_2.initialize(1.0, registry)
    registry.register("mixer_2", mixer_2)
    
    pump_1 = WaterPumpThermodynamic(pump_id="pump_1", target_pressure_pa=101325.0, eta_is=0.75) 
    pump_1.set_component_id("pump_1")
    pump_1.initialize(1.0, registry)
    registry.register("pump_1", pump_1)
    
    soec_config = {
        'max_power_nominal_mw': 2.4, # Per module
        'num_modules': 6,
        'power_first_step_mw': 0.1,
        'ramp_step_mw': 0.2
    }
    soec = SOECOperator(soec_config)
    soec.set_component_id("soec")
    soec.initialize(1.0, registry)
    registry.register("soec", soec)
    
    # --- Common (Merge & Compress) ---
    mixer_1 = WaterMixer(outlet_pressure_kpa=3000.0, fluid_type='H2') # 30 bar output
    mixer_1.set_component_id("mixer_1")
    mixer_1.initialize(1.0, registry)
    registry.register("mixer_1", mixer_1)
    
    compressor_0 = CompressorStorage(
        max_flow_kg_h=2000.0, # Increased capacity for ~14MW SOEC + 5MW PEM
        inlet_pressure_bar=30.0,
        outlet_pressure_bar=500.0,
        inlet_temperature_c=80.0
    )
    compressor_0.set_component_id("compressor_0")
    compressor_0.initialize(1.0, registry)
    registry.register("compressor_0", compressor_0)
    
    # 3. Load Real 24h Data
    # Direct load of environment data which has both Price and Time
    try:
        csv_path = 'h2_plant/data/environment_data_2024.csv'
        df = pd.read_csv(csv_path)
        # Check columns
        # Expecting: time, price_eur_mwh, ...
        # If headers differ, we adjust. 
        # From head: time,wind_power_coefficient,wind_speed,air_density,price_eur_mwh,...
        
        prices = df['price_eur_mwh'].values[:24]
        # Ensure we have 24 hours
        if len(prices) < 24:
            prices = np.pad(prices, (0, 24-len(prices)), 'edge')
            
    except Exception as e:
        print(f"Warning: Failed to load real data ({e}), using mock.")
        prices = np.array([40]*6 + [90]*4 + [30]*4 + [100]*4 + [40]*6)
    
    print(f"{'Hour':<5} | {'Price':<6} | {'Mode':<10} | {'PEM (MW)':<8} | {'SOEC (MW)':<9} | {'H2 (kg)':<8}")
    print("-" * 65)
    
    total_h2 = 0.0
    pem_energy_mwh = 0.0
    soec_energy_mwh = 0.0
    
    pem_h2_total = 0.0
    soec_h2_total = 0.0
    
    # 4. Run Simulation
    print(f"\n--- Simulation Start (Standard 'Green' Arbitrage Mode) ---")
    print(f"Policy: Max Prod < 50, SOEC < 80, Standby > 80 EUR/MWh")
    print(f"Note: To match Legacy Output (~9000kg), assume H2 Value = 9.6 EUR/kg (Threshold ~306 EUR/MWh).")
    
    total_h2_kg = 0.0
    
    for t in range(24):
        price = prices[t]
        
        # Simple Arbitrage Logic
        mode = "STANDBY"
        pem_setpoint = 0.0
        soec_setpoint = 0.0
        
        if price < 50.0:
            # Very cheap: Run everything max
            mode = "MAX_PROD"
            pem_setpoint = 5.0
            soec_setpoint = 14.4
        elif price < 80.0:
            # Moderate: Run high efficiency SOEC only
            mode = "SOEC_ONLY"
            pem_setpoint = 0.0
            soec_setpoint = 14.4
        else:
            # Expensive: Shut down
            mode = "STANDBY"
            pem_setpoint = 0.0
            soec_setpoint = 0.0
            
        # Execute Components
        # 1. Sources (Unlimited Water)
        # Original code used a single water_source stream for both lines.
        # The provided snippet had a generic 'mixer' and 'Stream' constructor.
        # Reverting to original component names and Stream constructor.
        water_source = Stream(mass_flow_kg_h=20000.0, temperature_k=293.15, pressure_pa=101325.0, composition={'H2O': 1.0})
        
        # 2. Line 1 (PEM)
        # Mixer -> Pump -> PEM
        # If setpoint > 0, we flow.
        if pem_setpoint > 0:
             mixer_0.receive_input('src', water_source, 'water')
        else:
             mixer_0.receive_input('src', Stream(mass_flow_kg_h=0.0, temperature_k=293.15, pressure_pa=101325.0, composition={'H2O': 1.0}), 'water')
             
        mixer_0.step(t)
        mix0_out = mixer_0.get_output('outlet')
        
        pump_0.receive_input('water_in', mix0_out, 'water')
        pump_0.step(t)
        pump0_out = pump_0.get_output('water_out')
        
        pem.set_power_input_mw(pem_setpoint)
        pem.receive_input('water_in', pump0_out, 'water')
        pem.step(t)
        pem_h2 = pem.get_output('h2_out')
        
        pem_energy_mwh += (pem.P_consumed_W / 1e6) * 1.0 # 1h timestep
        pem_h2_total += pem_h2.mass_flow_kg_h # kg/h * 1h = kg
        
        # 3. Line 2 (SOEC)
        mixer_2.receive_input('src', water_source, 'water')
        mixer_2.step(t)
        mix2_out = mixer_2.get_output('outlet')
        
        pump_1.receive_input('water_in', mix2_out, 'water')
        pump_1.step(t)
        pump1_out = pump_1.get_output('water_out')
        
        # SOEC Step (Accepts Water now!)
        soec.receive_input('steam_in', pump1_out, 'water')
        soec.step(reference_power_mw=soec_setpoint, t=t)
        soec_h2 = soec.get_output('h2_out')
        
        # SOEC tracks its own energy/production internally in step()
        # But we can grab it from get_state or calculate
        # step() returns (power_mw, h2_kg, steam_kg)
        # Note: step() was called with return values, but we didn't capture them above.
        # We can use the outputs.
        # Total power from SOEC object is available via get_status()
        status = soec.get_status()
        soec_power = status['total_power_mw']
        soec_energy_mwh += soec_power * 1.0
        
        soec_prod = soec.last_step_h2_kg # or similar, checking SOEC impl
        soec_h2_total += soec.total_h2_produced - (soec_h2_total) # Delta
        # Actually simplest is just to use final cumulative
        
        # 4. Merge H2
        if soec_h2.mass_flow_kg_h > 0:
             soec_h2.pressure_pa = 30e5 # Boost pressure
        
        mixer_1.receive_input('in_pem', pem_h2, 'hydrogen')
        mixer_1.receive_input('in_soec', soec_h2, 'hydrogen')
        mixer_1.step(t)
        merged_h2 = mixer_1.get_output('outlet')
        
        # 5. Compress
        compressor_0.receive_input('h2_in', merged_h2, 'hydrogen')
        compressor_0.step(t)
        final_h2 = compressor_0.get_output('h2_out')
        
        if final_h2:
            total_h2 += final_h2.mass_flow_kg_h
            
        print(f"{t:<5} | {price:<6.2f} | {mode:<10} | {pem_setpoint:<8.1f} | {soec_setpoint:<9.1f} | {final_h2.mass_flow_kg_h if final_h2 else 0.0:<8.1f}")
        
    print("-" * 65)
    print("## Simulation Summary (New System)")
    print(f"* Total Offered Energy: {pem_energy_mwh + soec_energy_mwh:.2f} MWh")
    print(f"* Energy Supplied to SOEC: {soec_energy_mwh:.2f} MWh")
    print(f"* Energy Supplied to PEM: {pem_energy_mwh:.2f} MWh")
    print(f"* Total System Hydrogen Production (Compressor Out): {total_h2:.2f} kg")
    # Using cumulative counters from components for exact comparison
    print(f"  * SOEC Output (Component): {soec.total_h2_produced:.2f} kg")
    print(f"  * PEM Output (Component): {registry.get('pem').cumulative_h2_kg:.2f} kg")
    
if __name__ == "__main__":
    test_dual_path_arbitrage_simulation()
