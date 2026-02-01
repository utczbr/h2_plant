
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add project path
project_path = "/home/stuart/Documentos/Planta Hidrogenio"
sys.path.insert(0, project_path)

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.core.component_ids import ComponentID
from h2_plant.components.environment.environment_manager import EnvironmentManager
from h2_plant.gui.utils.report_generator import generate_reports

# --- CONFIGURATION ---
CONFIG_FILE = Path(project_path) / "scenarios" / "simulation_config.yaml"

# Constants for Arbitrage (Still hardcoded or extracted from config if possible)
# We will extract what we can from the loaded plant config
H2_PRICE_EUR_KG = 9.6
PPA_PRICE_EUR_MWH = 50.0

# Water Consumption Factors
EXTRA_WATER_CONSUMPTION_SOEC = 0.10
EXTRA_WATER_CONSUMPTION_PEM = 0.02

# --- SIMULATION ---
def run_simulation(config=None):
    if config is None:
        print(f"Loading configuration from {CONFIG_FILE}...")
        try:
            builder = PlantBuilder.from_file(CONFIG_FILE)
            registry = builder.registry
            config = builder.config
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error building plant: {e}")
            return
    else:
        print("Using provided configuration object...")
        # If config is passed, we assume it's a PlantConfig object
        # We need to build the registry from it
        try:
            builder = PlantBuilder(config)
            builder.build()
            registry = builder.registry
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error building plant from provided config: {e}")
            return

    # 2. Get Components
    try:
        env_manager: EnvironmentManager = registry.get(ComponentID.ENVIRONMENT_MANAGER)
        
        try:
            pem = registry.get(ComponentID.PEM_ELECTROLYZER_DETAILED)
        except:
            pem = None
            
        try:
            soec = registry.get(ComponentID.SOEC_CLUSTER)
        except:
            soec = None
            
    except Exception as e:
        print(f"Error retrieving components: {e}")
        return

    # 3. Initialize Components
    # EnvironmentManager loads data during initialization
    # We use a small dt for initialization, but the step loop will handle time
    registry.initialize_all(dt=1.0/60.0)
    
    # 4. Simulation Parameters
    sim_duration_hours = config.simulation.duration_hours
    sim_minutes = int(sim_duration_hours * 60)
    
    # Extract limits from config/components
    # Use V2.0 detailed config sections if available
    if config.pem_system:
        MAX_PEM_POWER_MW = config.pem_system.max_power_mw
    else:
        MAX_PEM_POWER_MW = getattr(config.production.electrolyzer, 'max_power_mw', 5.0)

    if config.soec_cluster:
        SOEC_MAX_CAPACITY_MW = config.soec_cluster.num_modules * config.soec_cluster.max_power_nominal_mw * getattr(config.soec_cluster, 'efficient_threshold', 0.80)
    else:
        # Fallback to production.soec if it exists (legacy)
        soec_cfg = config.production.soec
        if soec_cfg:
             SOEC_MAX_CAPACITY_MW = getattr(soec_cfg, 'num_modules', 6) * getattr(soec_cfg, 'max_power_nominal_mw', 2.4) * getattr(soec_cfg, 'optimal_limit', 0.80)
        else:
             SOEC_MAX_CAPACITY_MW = 11.52 # Default fallback
    
    print(f"Simulation Duration: {sim_duration_hours} hours ({sim_minutes} minutes)")
    print(f"PEM Max Power: {MAX_PEM_POWER_MW} MW")
    print(f"SOEC Max Capacity: {SOEC_MAX_CAPACITY_MW:.2f} MW")

    # Get Efficiency Constants from SOEC (via wrapper -> operator)
    # We need these for arbitrage calculation
    # Accessing private/internal logic for consistency
    import sys
    sys.path.insert(0, str(Path(project_path) / "h2_plant/legacy/pem_soec_reference/ALL_Reference"))
    import soec_operator
    SOEC_H2_KWH_KG = soec_operator.BASE_H2_CONSUMPTION_KWH_PER_KG
    
    # Arbitrage Thresholds
    h2_eq_price = (1000.0 / SOEC_H2_KWH_KG) * H2_PRICE_EUR_KG
    arbitrage_limit = PPA_PRICE_EUR_MWH + h2_eq_price
    
    print(f"Arbitrage Limit: {arbitrage_limit:.2f} EUR/MWh")
    
    # State Variables
    force_sell_flag = False
    previous_soec_power = 0.0
    
    # History
    history = {
        'minute': [],
        'P_offer': [], 'P_soec': [], 'P_pem': [], 'P_sold': [], 'Spot': [],
        'H2_soec': [], 'H2_pem': [], 'Steam_soec': [], 'H2O_pem': [],
        'sell_decision': []
    }
    
    print("\nStarting Simulation...")
    
    # Pre-check data availability
    if env_manager.wind_data is None or env_manager.price_data is None:
        print("Error: Environmental data not loaded correctly.")
        return

    for t_min in range(sim_minutes):
        t_hours = t_min / 60.0
        minute_of_hour = t_min % 60
        
        # Update Environment
        env_manager.step(t_hours)
        
        # Get Current Conditions
        P_offer = env_manager.current_wind_power_mw
        spot_price = env_manager.current_energy_price_eur_mwh
        
        # Get Future Conditions (for ramp down)
        # EnvironmentManager has get_future_power(minutes_ahead)
        P_future_offer = env_manager.get_future_power(minutes_ahead=60)
        
        # --- ARBITRAGE LOGIC (Match manager.py) ---
        
        # 1. Check Ramp Up Opportunity (Minute 0)
        if minute_of_hour == 0:
            diff = P_offer - previous_soec_power
            if diff > 0:
                # Profit check
                sale_profit = diff * 0.25 * (spot_price - PPA_PRICE_EUR_MWH)
                h2_profit = (diff * 0.25 * 1000.0 / SOEC_H2_KWH_KG) * H2_PRICE_EUR_KG
                
                if sale_profit > h2_profit:
                    force_sell_flag = True
                else:
                    force_sell_flag = False
        
        # 2. Continuous Checks
        if force_sell_flag and spot_price <= arbitrage_limit:
            force_sell_flag = False
            
        # 3. Ramp Down Anticipation (Minute 45)
        if minute_of_hour == 45:
            P_soec_fut = min(P_future_offer, SOEC_MAX_CAPACITY_MW)
            if previous_soec_power > P_soec_fut:
                force_sell_flag = False
                
        # --- DISPATCH ---
        sell_decision = force_sell_flag
        
        P_soec_set = 0.0
        P_pem_set = 0.0
        P_sold = 0.0
        
        if sell_decision:
            # Sell everything above previous SOEC level (Bypass)
            P_soec_set = previous_soec_power
            # PEM is 0
            # Sold is calculated after actuals
            
        else:
            # Normal Operation
            P_soec_set = min(P_offer, SOEC_MAX_CAPACITY_MW)
            
            # Future constraint (Minute 45-59)
            if 45 <= minute_of_hour < 60:
                if P_future_offer < P_offer:
                    P_soec_fut = min(P_future_offer, SOEC_MAX_CAPACITY_MW)
                    P_soec_set = min(P_soec_set, P_soec_fut)
            
            # PEM Logic
            # Will be calculated after SOEC actual
            
        # --- EXECUTE COMPONENTS ---
        
        # 1. SOEC
        if soec:
            soec.set_power_setpoint(P_soec_set)
            soec.step(t_hours) 
            P_soec_actual = soec.get_current_power()
            h2_soec = soec.get_h2_production_rate()
            steam_soec = soec.get_steam_consumption_rate()
        else:
            P_soec_actual = 0.0
            h2_soec = 0.0
            steam_soec = 0.0
        
        # 2. PEM & Sold Calculation
        surplus = P_offer - P_soec_actual
        
        if sell_decision:
            P_sold = surplus
            P_pem_set = 0.0
        else:
            # Check for local arbitrage (PEM vs Sell)
            if surplus > 0 and spot_price > arbitrage_limit:
                P_sold = surplus
                P_pem_set = 0.0
            elif surplus > 0:
                P_pem_set = min(surplus, MAX_PEM_POWER_MW)
                P_sold = surplus - P_pem_set
            else:
                P_pem_set = 0.0
                P_sold = 0.0
                
        # Execute PEM
        if pem:
            pem.set_power_input_mw(P_pem_set)
            pem.step(t_hours)
            P_pem_actual = pem.P_consumed_W / 1e6
            h2_pem = pem.m_H2_kg_s * 60.0 # kg/min
            h2o_pem = pem.m_H2O_kg_s * 60.0 # kg/min
        else:
            P_pem_actual = 0.0
            h2_pem = 0.0
            h2o_pem = 0.0
        
        # Update State
        previous_soec_power = P_soec_actual
        
        # Log
        history['minute'].append(t_min)
        history['P_offer'].append(P_offer)
        history['P_soec'].append(P_soec_actual)
        history['P_pem'].append(P_pem_actual)
        history['P_sold'].append(P_sold)
        history['Spot'].append(spot_price)
        history['H2_soec'].append(h2_soec)
        history['H2_pem'].append(h2_pem)
        history['Steam_soec'].append(steam_soec)
        history['H2O_pem'].append(h2o_pem)
        history['sell_decision'].append(1 if sell_decision else 0)
        
        if t_min % (24*60) == 0:
            print(f"Day {t_min//(24*60) + 1}/{int(sim_duration_hours/24)} completed...")

    # --- SUMMARY ---
    print("\n--- Simulation Summary ---")
    total_offer = sum(history['P_offer']) / 60.0
    total_soec = sum(history['P_soec']) / 60.0
    total_pem = sum(history['P_pem']) / 60.0
    total_sold = sum(history['P_sold']) / 60.0
    
    total_h2_soec = sum(history['H2_soec'])
    total_h2_pem = sum(history['H2_pem'])
    total_h2 = total_h2_soec + total_h2_pem
    
    total_steam_soec = sum(history['Steam_soec'])
    total_water_soec = total_steam_soec * (1 + EXTRA_WATER_CONSUMPTION_SOEC)
    
    total_h2o_pem = sum(history['H2O_pem'])
    total_water_pem = total_h2o_pem * (1 + EXTRA_WATER_CONSUMPTION_PEM)
    
    print(f"Total Offered Energy: {total_offer:.2f} MWh")
    print(f"SOEC Consumption: {total_soec:.2f} MWh")
    print(f"PEM Consumption: {total_pem:.2f} MWh")
    print(f"Sold Energy: {total_sold:.2f} MWh")
    print(f"Total H2 Production: {total_h2:.2f} kg")
    print(f"  SOEC: {total_h2_soec:.2f} kg")
    print(f"  PEM: {total_h2_pem:.2f} kg")
    print(f"Total Water Consumption: {total_water_soec + total_water_pem:.2f} kg")
    
    # Save results
    df_res = pd.DataFrame(history)
    df_res.to_csv("simulation_results_30days.csv")
    print("Results saved to simulation_results_30days.csv")
    
    # Generate Reports
    generate_reports(history, output_dir=project_path)

if __name__ == "__main__":
    run_simulation()
