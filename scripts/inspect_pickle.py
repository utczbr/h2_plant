import pickle
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def inspect_pickle():
    path = Path("h2_plant/data/degradation_polynomials.pkl")
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, 'rb') as f:
        polys = pickle.load(f)

    print(f"Loaded {len(polys)} polynomials.")
    
    if not polys:
        return

    poly0 = polys[0]
    print(f"Polynomial 0: {poly0}")
    
    # Test values
    powers_W = [1e5, 1e6, 5e6]
    powers_kW = [100, 1000, 5000]
    powers_MW = [0.1, 1.0, 5.0]
    
    print("\nTesting Watts:")
    for p in powers_W:
        print(f"P={p:.1e} W -> j={poly0(p):.4f}")
        
    print("\nTesting kW:")
    for p in powers_kW:
        print(f"P={p:.1f} kW -> j={poly0(p):.4f}")

    print("\nTesting MW:")
    for p in powers_MW:
        print(f"P={p:.1f} MW -> j={poly0(p):.4f}")

if __name__ == "__main__":
    inspect_pickle()
