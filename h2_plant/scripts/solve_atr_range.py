import sys
import os
import pickle
import numpy as np

# Add project root to python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from h2_plant.components.reforming.atr_reactor import linear_interp_scalar, cubic_interp_scalar

def main():
    # Path to the model pickle file
    model_path = os.path.join(project_root, 'h2_plant/data/ATR_model_functions.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        with open(model_path, 'rb') as f:
            raw_model = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    # Extract all available functions
    func_names = sorted(raw_model.keys())
    print(f"Found {len(func_names)} functions in model.")
    
    # Build interpolation data for all functions
    interp_data = {}
    max_o2_flow = 0.0
    
    for name in func_names:
        f_obj = raw_model[name]
        x_data = f_obj.x.astype(np.float64)
        y_data = f_obj.y.astype(np.float64)
        kind = f_obj.kind if hasattr(f_obj, 'kind') else 'linear'
        
        interp_data[name] = (x_data, y_data, kind)
        
        # Determine max O2 flow from first function
        if max_o2_flow == 0.0 and len(x_data) > 0:
            max_o2_flow = x_data[-1]

    if max_o2_flow == 0.0:
        print("Error: Could not determine maximum Oxygen flow from model data.")
        return

    print(f"Nominal (Max) Oxygen Flow: {max_o2_flow:.4f} kmol/h")
    
    # Define output file
    output_file = os.path.join(os.path.dirname(__file__), 'atr_range_results.csv')
    
    print(f"Generating CSV with 10% step to {output_file}...")
    
    # Range from 30% to 100% in 10% steps
    percentages = np.arange(30, 101, 10)
    
    with open(output_file, 'w') as f_out:
        # Header
        header = ["Charge_%", "O2_In_kmol_h"] + func_names
        f_out.write(",".join(header) + "\n")
        
        for p in percentages:
            o2_flow = max_o2_flow * (p / 100.0)
            
            row = [str(p), f"{o2_flow:.4f}"]
            
            for name in func_names:
                x_data, y_data, kind = interp_data[name]
                if 'cubic' in str(kind) or kind == 3:
                    val = cubic_interp_scalar(o2_flow, x_data, y_data)
                else:
                    val = linear_interp_scalar(o2_flow, x_data, y_data)
                row.append(f"{val:.4f}")
            
            f_out.write(",".join(row) + "\n")

    print("Done.")
    
    # =========================================================================
    # Save all interpolation functions as Numba-compatible arrays
    # =========================================================================
    # This exports the extracted interpolation data (x_data, y_data, kind)
    # from all functions in the original pickle, making them available for
    # fast JIT-compiled interpolation in other components.
    output_pkl = os.path.join(project_root, 'h2_plant/data/ATR_interp_data.pkl')
    
    print(f"Saving all interpolation functions to {output_pkl}...")
    
    with open(output_pkl, 'wb') as f_pkl:
        pickle.dump(interp_data, f_pkl)
    
    print(f"Saved {len(interp_data)} interpolation functions:")
    for name in sorted(interp_data.keys()):
        x_data, y_data, kind = interp_data[name]
        print(f"  - {name}: {len(x_data)} points, kind={kind}")
    
    print("All functions saved successfully.")

if __name__ == "__main__":
    main()
