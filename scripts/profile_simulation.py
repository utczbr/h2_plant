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
from h2_plant.run_integrated_simulation import run_with_dispatch_strategy

if __name__ == '__main__':
    print("=" * 60)
    print("PERFORMANCE PROFILING - 1 Hour Simulation Benchmark")
    print("=" * 60)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    t0 = time.perf_counter()
    
    # Run using the standard entry point to ensure correct setup
    run_with_dispatch_strategy(
        scenarios_dir='scenarios',
        hours=1  # Short duration for benchmarking
    )
    
    run_time = time.perf_counter() - t0
    
    profiler.disable()
    
    print(f"\nTotal Run time: {run_time:.2f}s")
    
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
