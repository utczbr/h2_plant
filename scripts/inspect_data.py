
import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def inspect_polynomials():
    path = Path("h2_plant/data/degradation_polynomials.pkl")
    if not path.exists():
        print(f"❌ {path} not found")
        return

    try:
        with open(path, 'rb') as f:
            polys = pickle.load(f)
        print(f"✅ Loaded {len(polys)} polynomials")
        
        # Test the first polynomial with a dummy value
        # Assuming it expects Watts. Let's try 3.5 MW (3.5e6 W)
        p0 = polys[0]
        val_watts = p0(3.5e6)
        val_kw = p0(3500)
        print(f"Poly[0](3.5e6) [Watts] = {val_watts}")
        print(f"Poly[0](3500) [kW] = {val_kw}")
        
    except Exception as e:
        print(f"❌ Error loading polynomials: {e}")

def inspect_lut():
    path = Path("h2_plant/data/lut_pem_vcell.npz")
    if not path.exists():
        print(f"❌ {path} not found")
        return

    try:
        data = np.load(path)
        print(f"✅ Loaded LUT data: {list(data.keys())}")
        v_cell = data['v_cell']
        j_op = data['j_op']
        t_op_h = data['t_op_h']
        
        print(f"v_cell shape: {v_cell.shape}")
        print(f"j_op range: {j_op.min()} to {j_op.max()}")
        print(f"t_op_h range: {t_op_h.min()} to {t_op_h.max()}")
        
        # Sample some values
        print(f"Sample V_cell at min j, min t: {v_cell[0,0]}")
        print(f"Sample V_cell at max j, min t: {v_cell[-1,0]}")
        print(f"Sample V_cell at mid j, mid t: {v_cell[len(j_op)//2, len(t_op_h)//2]}")
        
    except Exception as e:
        print(f"❌ Error loading LUT: {e}")

if __name__ == "__main__":
    print("--- Inspecting Polynomials ---")
    inspect_polynomials()
    print("\n--- Inspecting LUT ---")
    inspect_lut()
