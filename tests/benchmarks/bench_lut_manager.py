import time
import numpy as np
from h2_plant.optimization.lut_manager import LUTManager

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False

def benchmark_lut_vs_coolprop(num_lookups: int = 1000):
    """Compare LUT vs CoolProp performance."""
    
    lut = LUTManager()
    lut.initialize()
    
    # Random test points
    pressures = np.random.uniform(30e5, 900e5, num_lookups)
    temperatures = np.random.uniform(250, 350, num_lookups)
    
    # Benchmark LUT
    start = time.perf_counter()
    for p, t in zip(pressures, temperatures):
        density = lut.lookup('H2', 'D', p, t)
    lut_time = time.perf_counter() - start
    
    # Benchmark CoolProp (if available)
    if COOLPROP_AVAILABLE:
        start = time.perf_counter()
        for p, t in zip(pressures, temperatures):
            density = CP.PropsSI('D', 'P', p, 'T', t, 'H2')
        coolprop_time = time.perf_counter() - start
        
        speedup = coolprop_time / lut_time
        
        print(f"Benchmark Results ({num_lookups} lookups):")
        print(f"  LUT:      {lut_time*1000:.2f} ms ({lut_time/num_lookups*1000:.4f} ms/lookup)")
        print(f"  CoolProp: {coolprop_time*1000:.2f} ms ({coolprop_time/num_lookups*1000:.4f} ms/lookup)")
        print(f"  Speedup:  {speedup:.1f}x")
    else:
        print(f"LUT: {lut_time*1000:.2f} ms ({lut_time/num_lookups*1000:.4f} ms/lookup)")

if __name__ == '__main__':
    benchmark_lut_vs_coolprop()
