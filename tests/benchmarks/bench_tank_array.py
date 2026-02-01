import time
import numpy as np
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.core.component_registry import ComponentRegistry

def benchmark_fill_operations(n_tanks: int = 100, n_operations: int = 1000):
    """Benchmark tank filling operations."""
    
    tanks = TankArray(n_tanks=n_tanks, capacity_kg=200.0, pressure_bar=350)
    tanks.initialize(dt=1.0, registry=ComponentRegistry())
    
    # Random fill amounts
    fill_amounts = np.random.uniform(10, 500, n_operations)
    
    start = time.perf_counter()
    for mass in fill_amounts:
        tanks.fill(mass)
    elapsed = time.perf_counter() - start
    
    print(f"Tank Fill Benchmark ({n_tanks} tanks, {n_operations} operations):")
    print(f"  Total time: {elapsed*1000:.2f} ms")
    print(f"  Per operation: {elapsed/n_operations*1000:.4f} ms")
    print(f"  Operations/sec: {n_operations/elapsed:.0f}")

if __name__ == '__main__':
    benchmark_fill_operations()
