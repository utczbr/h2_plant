import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append("/home/stuart/Documentos/Planta Hidrogenio")

from h2_plant.visualization.plotly_graphs import plot_process_train_profile

def run_test():
    print("### Testing Graph Verification Prints ###")
    
    # Mock DataFrame with required columns for SOEC Cluster -> Interchanger
    # Scenario: 1110.3 Bulk, 1.0 Entrained -> 1111.3 Total. 984857.7 PPM.
    data = {
        'SOEC_Cluster_outlet_mass_flow_kg_h': [1110.3],
        'SOEC_Cluster_outlet_entrained_mass_kg_h': [1.0],
        'SOEC_Cluster_outlet_H2O_molf': [0.9848577],
        
        'SOEC_H2_Interchanger_1_outlet_mass_flow_kg_h': [1110.3],
        'SOEC_H2_Interchanger_1_outlet_entrained_mass_kg_h': [1.0],
        'SOEC_H2_Interchanger_1_outlet_H2O_molf': [0.9848577],
        
        'SOEC_H2_DryCooler_1_outlet_mass_flow_kg_h': [1111.3],
        'SOEC_H2_DryCooler_1_outlet_entrained_mass_kg_h': [0.0],
        'SOEC_H2_DryCooler_1_outlet_H2O_molf': [0.9848577],
    }
    
    df = pd.DataFrame(data)
    
    print("\n[Calling plot_process_train_profile]...")
    try:
        # This function should now PRINT the values to console
        fig = plot_process_train_profile(df, title="Test Graph")
        print("\n[Graph Generation Complete]")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_test()
