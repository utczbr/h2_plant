"""
Performance benchmark comparing legacy Mixer.py with new WaterMixer component.

Measures execution time for thermodynamic calculations to ensure
the new architecture doesn't introduce significant overhead.
"""

import sys
import time
import statistics
from typing import List, Tuple

from Mixer import mixer_model, get_example_data
from Refactored_Mixer import Mixer
from Framework import Stream, ComponentRegistry


def benchmark_legacy(iterations: int = 1000) -> Tuple[float, float, float]:
    """
    Benchmark legacy Mixer.py implementation.
    
    Args:
        iterations: Number of iterations to run
        
    Returns:
        Tuple of (mean_time, min_time, max_time) in milliseconds
    """
    legacy_streams, legacy_p_out_kpa = get_example_data()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mixer_model(legacy_streams, legacy_p_out_kpa)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return (
        statistics.mean(times),
        min(times),
        max(times)
    )


def benchmark_new_architecture(iterations: int = 1000) -> Tuple[float, float, float]:
    """
    Benchmark new WaterMixer component.
    
    Args:
        iterations: Number of iterations to run
        
    Returns:
        Tuple of (mean_time, min_time, max_time) in milliseconds
    """
    # Setup (done once, not timed)
    legacy_streams, legacy_p_out_kpa = get_example_data()
    
    times = []
    for _ in range(iterations):
        # Create fresh mixer for each iteration
        new_mixer = Mixer(
            mixer_id="BenchmarkMixer",
            fluid_type="Water",
            outlet_pressure_kpa=legacy_p_out_kpa
        )
        registry = ComponentRegistry()
        new_mixer.initialize(dt=1.0, registry=registry)
        
        # Convert legacy data to Stream objects
        streams = []
        for i, s in enumerate(legacy_streams):
            stream_obj = Stream(
                mass_flow_kg_h=s['m_dot'] * 3600.0,
                temperature_k=s['T'] + 273.15,
                pressure_pa=s['P'] * 1000.0,
                composition={'H2O': 1.0},
                phase='liquid'
            )
            streams.append(stream_obj)
        
        # Time the actual mixing calculation
        start = time.perf_counter()
        for i, stream in enumerate(streams):
            new_mixer.receive_input(f"stream_{i+1}", stream, "water")
        new_mixer.step(t=0.0)
        _ = new_mixer.get_state()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    return (
        statistics.mean(times),
        min(times),
        max(times)
    )


def run_benchmark():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("          MIXER PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()
    
    # Warmup runs
    print("Warming up...")
    benchmark_legacy(iterations=10)
    benchmark_new_architecture(iterations=10)
    print("Warmup complete.\n")
    
    # Actual benchmark
    iterations = 1000
    print(f"Running benchmark with {iterations} iterations per implementation...")
    print()
    
    # Legacy benchmark
    print(">>> Benchmarking Legacy Mixer.py...")
    legacy_mean, legacy_min, legacy_max = benchmark_legacy(iterations)
    
    # New architecture benchmark
    print(">>> Benchmarking New WaterMixer...")
    new_mean, new_min, new_max = benchmark_new_architecture(iterations)
    
    # Results
    print()
    print("=" * 70)
    print("                         RESULTS")
    print("=" * 70)
    print()
    print(f"{'Implementation':<25} | {'Mean (ms)':<12} | {'Min (ms)':<12} | {'Max (ms)':<12}")
    print("-" * 70)
    print(f"{'Legacy Mixer.py':<25} | {legacy_mean:<12.4f} | {legacy_min:<12.4f} | {legacy_max:<12.4f}")
    print(f"{'New WaterMixer':<25} | {new_mean:<12.4f} | {new_min:<12.4f} | {new_max:<12.4f}")
    print("-" * 70)
    
    # Performance comparison
    print()
    print("=" * 70)
    print("                      PERFORMANCE ANALYSIS")
    print("=" * 70)
    print()
    
    overhead_ms = new_mean - legacy_mean
    overhead_percent = (overhead_ms / legacy_mean) * 100
    
    print(f"Overhead (absolute):  {overhead_ms:+.4f} ms")
    print(f"Overhead (relative):  {overhead_percent:+.2f}%")
    print()
    
    # Performance rating
    if abs(overhead_percent) < 5:
        rating = "✅ EXCELLENT"
        msg = "New implementation has negligible overhead"
    elif abs(overhead_percent) < 15:
        rating = "✅ GOOD"
        msg = "New implementation has acceptable overhead"
    elif abs(overhead_percent) < 30:
        rating = "⚠️  ACCEPTABLE"
        msg = "New implementation has moderate overhead"
    else:
        rating = "❌ NEEDS OPTIMIZATION"
        msg = "New implementation has significant overhead"
    
    print(f"Performance Rating: {rating}")
    print(f"Assessment: {msg}")
    print()
    
    # Throughput analysis
    print("=" * 70)
    print("                    THROUGHPUT ANALYSIS")
    print("=" * 70)
    print()
    
    legacy_throughput = 1000 / legacy_mean  # calculations per second
    new_throughput = 1000 / new_mean
    
    print(f"Legacy Mixer.py:      {legacy_throughput:,.0f} calculations/second")
    print(f"New WaterMixer:       {new_throughput:,.0f} calculations/second")
    print()
    
    # Statistical analysis
    print("=" * 70)
    print("                   STATISTICAL SUMMARY")
    print("=" * 70)
    print()
    
    legacy_range = legacy_max - legacy_min
    new_range = new_max - new_min
    
    print(f"{'Metric':<30} | {'Legacy':<15} | {'New':<15}")
    print("-" * 70)
    print(f"{'Time Range (ms)':<30} | {legacy_range:<15.4f} | {new_range:<15.4f}")
    print(f"{'Coefficient of Variation':<30} | {'N/A':<15} | {'N/A':<15}")
    print()
    
    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
