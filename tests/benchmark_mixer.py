
import time
import numpy as np
from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer
from h2_plant.core.stream import Stream

def benchmark_mixer():
    print("Benchmarking MultiComponentMixer (Before Optimization)...")
    
    class DummyRegistry:
        def has(self, x): return False
        def get(self, x): return None

    # Setup
    mixer = MultiComponentMixer(volume_m3=10.0)
    mixer.initialize(dt=0.1, registry=DummyRegistry())
    
    # Warmup
    for _ in range(10):
        s = Stream(mass_flow_kg_h=100.0, temperature_k=300.0, pressure_pa=30e5, composition={'H2': 1.0}, phase='gas')
        mixer.receive_input('inlet', s)
        mixer.step(0.0)
        
    # Benchmark
    start_time = time.time()
    n_steps = 1000
    
    for i in range(n_steps):
        # Vary input to force non-trivial flash
        T_in = 300.0 + (i % 50)
        s = Stream(mass_flow_kg_h=100.0, temperature_k=T_in, pressure_pa=30e5, composition={'H2': 0.8, 'H2O': 0.2}, phase='gas')
        mixer.receive_input('inlet', s)
        mixer.step(0.0)
        
    end_time = time.time()
    duration = end_time - start_time
    avg_time = (duration / n_steps) * 1000.0 # ms
    
    print(f"Total time: {duration:.4f} s")
    print(f"Avg time per step: {avg_time:.4f} ms")
    return avg_time

if __name__ == "__main__":
    benchmark_mixer()
