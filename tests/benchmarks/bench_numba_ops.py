import time
import numpy as np
from h2_plant.optimization.numba_ops import (
    find_available_tank,
    batch_pressure_update,
    calculate_compression_work
)
from h2_plant.core.enums import TankState

def benchmark_find_available_tank():
    """Compare Numba JIT vs Python loop."""
    
    n_tanks = 100
    states = np.random.randint(0, 4, n_tanks, dtype=np.int32)
    masses = np.random.uniform(0, 200, n_tanks)
    capacities = np.full(n_tanks, 200.0)
    
    # Warm up JIT
    find_available_tank(states, masses, capacities)
    
    # Benchmark
    num_trials = 10000
    start = time.perf_counter()
    for _ in range(num_trials):
        idx = find_available_tank(states, masses, capacities)
    elapsed = time.perf_counter() - start
    
    print(f"find_available_tank (Numba JIT):")
    print(f"  {elapsed*1000:.2f} ms for {num_trials} calls")
    print(f"  {elapsed/num_trials*1e6:.2f} μs per call")

def benchmark_batch_pressure_update():
    """Benchmark vectorized pressure calculation."""
    
    n_tanks = 1000
    masses = np.random.uniform(50, 200, n_tanks)
    volumes = np.random.uniform(0.8, 1.2, n_tanks)
    
    # Warm up
    batch_pressure_update(masses, volumes, 298.15)
    
    # Benchmark
    num_trials = 1000
    start = time.perf_counter()
    for _ in range(num_trials):
        pressures = batch_pressure_update(masses, volumes, 298.15)
    elapsed = time.perf_counter() - start
    
    print(f"\nbatch_pressure_update (Numba JIT):")
    print(f"  {elapsed*1000:.2f} ms for {num_trials} calls ({n_tanks} tanks each)")
    print(f"  {elapsed/num_trials*1000:.4f} ms per call")

if __name__ == '__main__':
    benchmark_find_available_tank()
    benchmark_batch_pressure_update()
