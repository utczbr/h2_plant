"""
Performance Profiling Script for Mixture Thermodynamics Bottleneck Analysis.
"""
import sys
import os
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')

import cProfile
import pstats
from io import StringIO
import time

def run_profiled_simulation():
    """Run a short simulation with profiling enabled."""
    from h2_plant.run_integrated_simulation import main
    
    # Override to run shorter simulation
    import h2_plant.run_integrated_simulation as sim_module
    
    # Run simulation
    main()

if __name__ == '__main__':
    print("=" * 60)
    print("PERFORMANCE PROFILING - 60 step benchmark")
    print("=" * 60)
    
    # Time the imports first
    t0 = time.perf_counter()
    from h2_plant.simulation.engine import SimulationEngine
    from h2_plant.core.plant_graph_builder import PlantGraphBuilder
    import_time = time.perf_counter() - t0
    print(f"Import time: {import_time:.2f}s")
    
    # Load topology
    t0 = time.perf_counter()
    builder = PlantGraphBuilder('scenarios/plant_topology.yaml')
    plant = builder.build()
    build_time = time.perf_counter() - t0
    print(f"Build time: {build_time:.2f}s")
    
    # Initialize engine
    t0 = time.perf_counter()
    engine = SimulationEngine(plant)
    init_time = time.perf_counter() - t0
    print(f"Engine init time: {init_time:.2f}s")
    
    # Profile the step execution
    print("\nProfiling 60 simulation steps...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    t0 = time.perf_counter()
    engine.run(steps=60)
    run_time = time.perf_counter() - t0
    
    profiler.disable()
    
    steps_per_sec = 60 / run_time
    print(f"\nRun time: {run_time:.2f}s ({steps_per_sec:.1f} steps/sec)")
    
    # Get stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(40)
    
    print("\n" + "=" * 60)
    print("TOP 40 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 60)
    print(s.getvalue())
    
    # Check for mixture_thermodynamics specifically
    s2 = StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.sort_stats('cumulative')
    ps2.print_stats('mixture')
    
    print("\n" + "=" * 60)
    print("MIXTURE_THERMODYNAMICS CALLS")
    print("=" * 60)
    print(s2.getvalue())
    
    # Check for LUT lookups
    s3 = StringIO()
    ps3 = pstats.Stats(profiler, stream=s3)
    ps3.sort_stats('tottime')
    ps3.print_stats('lookup')
    
    print("\n" + "=" * 60)
    print("LUT LOOKUP CALLS")
    print("=" * 60)
    print(s3.getvalue())
