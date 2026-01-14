
import pandas as pd
import numpy as np
import os

def create_audit_test_data():
    minutes = 60 * 24  # 1 day
    df = pd.DataFrame({'minute': range(minutes)})
    
    # Random data helper
    def rand_col(scale=100):
        return np.random.uniform(0, scale, minutes)
    
    # 1. WATER FIX VERIFICATION
    # Old columns (Proxy)
    df['Steam_soec'] = rand_col(10)
    df['H2O_pem'] = rand_col(10)
    
    # New columns (Actual Source)
    df['Water_Source_outlet_mass_flow_kg_h'] = rand_col(50) # significantly different scale to verify plot uses this
    
    # 2. OXYGEN FIX VERIFICATION
    # Old columns (Proxy Base)
    df['H2_soec'] = rand_col(5)
    df['H2_pem'] = rand_col(5)
    
    # New columns (Sensor)
    df['PEM_Stack_1_o2_production_kg_h'] = rand_col(40) # 8x H2 roughly
    df['SOEC_Stack_1_o2_production_kg_h'] = rand_col(40)
    
    # 3. THERMAL FIX VERIFICATION
    # Old columns (DryCooler)
    df['DryCooler_1_tqc_duty_kw'] = rand_col(500)
    df['Chiller_1_cooling_load_kw'] = rand_col(200)
    
    # New columns (Intercooler)
    df['SOEC_H2_Intercooler_1_tqc_duty_kw'] = rand_col(300) 
    df['SOEC_H2_Intercooler_2_heat_rejected_kw'] = rand_col(300)
    
    # 4. DRY COOLER FIX VERIFICATION
    df['DryCooler_1_heat_rejected_kw'] = rand_col(500)
    df['DryCooler_1_outlet_temp_k'] = 300 + rand_col(20)
    df['SOEC_H2_Intercooler_1_outlet_temp_k'] = 310 + rand_col(20)
    
    # Add Core columns to avoid errors
    df['P_offer'] = rand_col(100)
    df['P_soec_actual'] = rand_col(100)
    df['P_pem'] = rand_col(100)
    df['P_sold'] = rand_col(100)
    df['spot_price'] = rand_col(100)
    df['h2_kg'] = rand_col(10)
    df['time'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(df['minute'], unit='min')
    df['hour'] = df['minute'] // 60
    
    # Save to root
    df.to_csv('simulation_history.csv', index=False)
    print(f"Created simulation_history.csv with {len(df)} rows and {len(df.columns)} columns.")
    print("Columns includes: Water_Source, O2 production, Intercooler data.")

if __name__ == "__main__":
    create_audit_test_data()
