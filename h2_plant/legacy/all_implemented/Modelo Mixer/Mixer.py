import CoolProp.CoolProp as CP
import numpy as np

# =========================================================================
# DATA INPUT FUNCTIONS
# =========================================================================

def get_example_data():
    """Returns a predefined set of data for the example."""
    # m_dot (kg/s), T (°C), P (kPa)
    print("\nUsing the Default Example Data:")
    streams = [
        {"m_dot": 0.5, "T": 15.0, "P": 200.0},  # Stream 1: Cold water
        {"m_dot": 0.3, "T": 80.0, "P": 220.0},  # Stream 2: Hot water
        {"m_dot": 0.2, "T": 50.0, "P": 210.0}   # Stream 3: Warm water
    ]
    P_out = 200.0 # kPa
    return streams, P_out

def get_user_data():
    """Captures input data interactively from the user."""
    print("\n--- User Data Entry ---")
    streams = []
    
    for i in range(1, 4):
        print(f"\nData for Input Stream {i}:")
        try:
            m_dot = float(input(f"  Mass Flow Rate (m_dot, kg/s): "))
            T = float(input(f"  Temperature (T, °C): "))
            P = float(input(f"  Pressure (P, kPa): "))
            streams.append({"m_dot": m_dot, "T": T, "P": P})
        except ValueError:
            print("Invalid input. Please enter a number.")
            return None, None

    try:
        P_out = float(input("\nOutput Pressure (P_4, kPa): "))
    except ValueError:
        print("Invalid Output Pressure input.")
        return None, None

    return streams, P_out

# =========================================================================
# MAIN MIXER MODEL FUNCTION
# =========================================================================

def mixer_model(input_streams, P_out_kPa):
    """
    Calculates the output properties of a water mixer (Mass and Energy Balance).
    """
    
    fluid = 'Water'

    total_energy_in = 0.0 # In kJ/s (or kW)
    total_mass_in = 0.0   # In kg/s
    
    detailed_input_data = [] # To store m_dot, T, P, and calculated h for inputs
    
    print("-" * 50)
    print(f"**Initial Thermodynamic Calculations ({fluid})**")
    print("-" * 50)
    
    for i, stream in enumerate(input_streams):
        m_dot_i = stream['m_dot'] # kg/s
        T_i_C = stream['T']       # °C
        P_i_kPa = stream['P']     # kPa
        
        # Unit conversion for CoolProp standard (K and Pa)
        T_i_K = T_i_C + 273.15 
        P_i_Pa = P_i_kPa * 1000 
        
        try:
            # Find Enthalpy (H) in J/kg and convert to kJ/kg
            h_i_kJ_kg = CP.PropsSI('H', 'T', T_i_K, 'P', P_i_Pa, fluid) / 1000 
            
            # Mass Balance: Sum of mass flow rates
            total_mass_in += m_dot_i
            
            # Energy Balance: Sum of (m_dot_i * h_i)
            energy_in_i = m_dot_i * h_i_kJ_kg
            total_energy_in += energy_in_i
            
            # Store detailed input data for output analysis
            detailed_input_data.append({
                'Stream': i + 1,
                'm_dot (kg/s)': m_dot_i,
                'T (°C)': T_i_C,
                'P (kPa)': P_i_kPa,
                'h (kJ/kg)': h_i_kJ_kg
            })
            
            print(f"Stream {i+1}: T={T_i_C:.2f}°C, P={P_i_kPa:.2f} kPa -> h={h_i_kJ_kg:.3f} kJ/kg")
            
        except ValueError as e:
            print(f"\nERROR in Stream {i+1}: CoolProp failed to calculate Enthalpy with T={T_i_C}°C and P={P_i_kPa} kPa.")
            print(f"Details: {e}")
            return None, None

    # -----------------------------------------------------------
    # FINAL CALCULATIONS (OUTPUT PROPERTIES)
    # -----------------------------------------------------------

    # Mass Balance: Output Mass Flow Rate (m_dot_4)
    m_dot_4 = total_mass_in
    if m_dot_4 <= 0:
        print("\nERROR: Total input mass flow rate is zero.")
        return detailed_input_data, None
        
    # Energy Balance: Output Enthalpy (h_4)
    # h_4 = Total_Energy_In / m_dot_4
    h_4_kJ_kg = total_energy_in / m_dot_4
    
    # Find Output Temperature (T_4) using h_4 and P_4
    h_4_J_kg = h_4_kJ_kg * 1000 
    P_4_Pa = P_out_kPa * 1000   
    
    try:
        # Get Temperature ('T') from Enthalpy ('H') and Pressure ('P')
        T_4_K = CP.PropsSI('T', 'H', h_4_J_kg, 'P', P_4_Pa, fluid)
        T_4_C = T_4_K - 273.15 
    except ValueError as e:
        T_4_C = "ERROR (Invalid properties)"
        print(f"\nERROR calculating T_4: {e}")
        
    output_results = {
        'Output Mass Flow Rate (kg/s)': m_dot_4,
        'Output Specific Enthalpy (kJ/kg)': h_4_kJ_kg,
        'Output Pressure (kPa)': P_out_kPa,
        'Output Temperature (°C)': T_4_C
    }
    
    return detailed_input_data, output_results

# =========================================================================
# USER INTERFACE AND RESULT PRINTING
# =========================================================================

def main():
    """Main function for program execution."""
    
    print("===============================================")
    print("      3-STREAM WATER MIXER MODEL")
    print("===============================================")
    
    choice = input("Do you want to use the Example (E) or Enter Data (D)? [E/D]: ").upper()
    
    input_streams = None
    P_out = None

    if choice == 'E':
        input_streams, P_out = get_example_data()
    elif choice == 'D':
        input_streams, P_out = get_user_data()
    else:
        print("Invalid option. Exiting.")
        return

    if input_streams and P_out is not None:
        detailed_input_data, output_results = mixer_model(input_streams, P_out)
        
        print("\n\n" + "#" * 50)
        print("            CALCULATION SUMMARY")
        print("#" * 50)

        if detailed_input_data:
            ## 📊 Input Values and Calculated Enthalpies
            print("\n## 📊 Input Values and Calculated Enthalpies (CoolProp)")
            print("-" * 50)
            print(f"{'Stream':<6} | {'m_dot (kg/s)':<15} | {'T (°C)':<10} | {'P (kPa)':<10} | {'h (kJ/kg)':<15}")
            print("-" * 50)
            for d in detailed_input_data:
                print(f"{d['Stream']:<6} | {d['m_dot (kg/s)']:<15.3f} | {d['T (°C)']:<10.2f} | {d['P (kPa)']:<10.1f} | {d['h (kJ/kg)']:<15.3f}")
            print("-" * 50)

        if output_results:
            ## 🚀 Final Output Results (Balance Equations)
            print("\n## 🚀 Final Output Results (Balance Equations)")
            print("-" * 50)
            for key, value in output_results.items():
                if isinstance(value, float):
                    print(f"{key:<40}: {value:.3f}")
                else:
                    print(f"{key:<40}: {value}")
            print("-" * 50)

if __name__ == "__main__":
    main()