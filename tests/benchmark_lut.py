"""
Benchmark: LUT Manager vs CoolProp Performance

This test benchmarks the performance improvement from using pre-computed
lookup tables instead of direct CoolProp.PropsSI() calls.

Test Strategy:
    1. Time 1000 single lookups with both methods.
    2. Time batch lookup (vectorized).
    3. Compare saturation pressure lookup (1D LUT).
    4. Report speedup factors.
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    print("WARNING: CoolProp not available - limited benchmark")

from h2_plant.optimization.lut_manager import LUTManager, LUTConfig
from h2_plant.optimization.coolprop_lut import CoolPropLUT


def benchmark_property_lookup():
    """Benchmark single property lookups."""
    print("\n" + "=" * 60)
    print("BENCHMARK: LUT vs CoolProp Property Lookup")
    print("=" * 60)
    
    # Create minimal config for faster initialization
    config = LUTConfig(
        pressure_points=100,  # Reduced for benchmark
        temperature_points=100,
        fluids=('H2',)
    )
    
    lut = LUTManager(config)
    print("\nInitializing LUT Manager...")
    lut.initialize()
    
    # Test parameters
    N_SAMPLES = 1000
    pressures = np.random.uniform(10e5, 100e5, N_SAMPLES)
    temperatures = np.random.uniform(280, 400, N_SAMPLES)
    
    # === CoolProp Direct ===
    if COOLPROP_AVAILABLE:
        print(f"\nBenchmarking {N_SAMPLES} CoolProp.PropsSI() calls...")
        t0 = time.perf_counter()
        coolprop_results = [
            CP.PropsSI('D', 'P', p, 'T', t, 'H2')
            for p, t in zip(pressures, temperatures)
        ]
        t_coolprop = time.perf_counter() - t0
        print(f"  CoolProp time: {t_coolprop*1000:.2f} ms ({t_coolprop/N_SAMPLES*1e6:.1f} μs/call)")
    else:
        t_coolprop = None
    
    # === CoolPropLUT (Cached) ===
    print(f"\nBenchmarking {N_SAMPLES} CoolPropLUT.PropsSI() calls (cached)...")
    CoolPropLUT.clear_cache()
    t0 = time.perf_counter()
    cached_results = [
        CoolPropLUT.PropsSI('D', 'P', p, 'T', t, 'H2')
        for p, t in zip(pressures, temperatures)
    ]
    t_cached = time.perf_counter() - t0
    print(f"  Cached time: {t_cached*1000:.2f} ms ({t_cached/N_SAMPLES*1e6:.1f} μs/call)")
    
    # === LUT Manager (Single) ===
    print(f"\nBenchmarking {N_SAMPLES} LUTManager.lookup() calls...")
    t0 = time.perf_counter()
    lut_results = [
        lut.lookup('H2', 'D', p, t)
        for p, t in zip(pressures, temperatures)
    ]
    t_lut_single = time.perf_counter() - t0
    print(f"  LUT single time: {t_lut_single*1000:.2f} ms ({t_lut_single/N_SAMPLES*1e6:.1f} μs/call)")
    
    # === Summary ===
    print("\n" + "-" * 60)
    print("SUMMARY: Speedup Factors")
    print("-" * 60)
    if t_coolprop:
        print(f"  LUT vs CoolProp: {t_coolprop/t_lut_single:.1f}x faster")
        print(f"  Cached vs CoolProp: {t_coolprop/t_cached:.1f}x faster")
    print(f"  LUT vs Cached: {t_cached/t_lut_single:.1f}x faster")


def benchmark_saturation_lookup():
    """Benchmark saturation pressure lookup."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Saturation Pressure Lookup (T → P_sat)")
    print("=" * 60)
    
    lut = LUTManager()
    print("\nInitializing LUT Manager with saturation table...")
    lut.initialize()
    
    N_SAMPLES = 1000
    temperatures = np.random.uniform(280, 400, N_SAMPLES)
    
    # === CoolProp Direct ===
    if COOLPROP_AVAILABLE:
        print(f"\nBenchmarking {N_SAMPLES} CoolProp saturation calls...")
        t0 = time.perf_counter()
        cp_results = [
            CP.PropsSI('P', 'T', t, 'Q', 0, 'Water')
            for t in temperatures
        ]
        t_coolprop = time.perf_counter() - t0
        print(f"  CoolProp time: {t_coolprop*1000:.2f} ms ({t_coolprop/N_SAMPLES*1e6:.1f} μs/call)")
    else:
        t_coolprop = None
        cp_results = None
    
    # === LUT Saturation ===
    print(f"\nBenchmarking {N_SAMPLES} LUT saturation calls...")
    t0 = time.perf_counter()
    lut_results = [
        lut.lookup_saturation_pressure(t)
        for t in temperatures
    ]
    t_lut = time.perf_counter() - t0
    print(f"  LUT time: {t_lut*1000:.2f} ms ({t_lut/N_SAMPLES*1e6:.1f} μs/call)")
    
    # === Accuracy Check ===
    if cp_results:
        errors = np.abs(np.array(lut_results) - np.array(cp_results)) / np.array(cp_results) * 100
        print(f"\n  Mean error: {np.mean(errors):.4f}%")
        print(f"  Max error: {np.max(errors):.4f}%")
    
    # === Summary ===
    print("\n" + "-" * 60)
    print("SUMMARY: Saturation Lookup Speedup")
    print("-" * 60)
    if t_coolprop:
        print(f"  LUT vs CoolProp: {t_coolprop/t_lut:.1f}x faster")


def benchmark_full_step():
    """Simulate performance impact on full simulation step."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Simulated 1000 Timesteps (KOD Component)")
    print("=" * 60)
    
    from h2_plant.core.stream import Stream
    from h2_plant.core.component_registry import ComponentRegistry
    from h2_plant.components.separation.knock_out_drum import KnockOutDrum
    
    kod = KnockOutDrum(diameter_m=0.5, gas_species='H2')
    registry = ComponentRegistry()
    kod.initialize(dt=1/60, registry=registry)
    
    N_STEPS = 1000
    
    inlet = Stream(
        mass_flow_kg_h=100.0,
        temperature_k=333.15,
        pressure_pa=40e5,
        composition={'H2': 0.98, 'H2O': 0.02},
        phase='gas'
    )
    
    print(f"\nBenchmarking {N_STEPS} KnockOutDrum steps...")
    t0 = time.perf_counter()
    for i in range(N_STEPS):
        kod.receive_input('gas_inlet', inlet)
        kod.step(t=i/60)
    t_total = time.perf_counter() - t0
    
    print(f"  Total time: {t_total*1000:.2f} ms")
    print(f"  Time per step: {t_total/N_STEPS*1e6:.1f} μs")
    print(f"  Throughput: {N_STEPS/t_total:.0f} steps/second")
    
    # Estimate for 1-year simulation at 1-min timestep
    steps_per_year = 525600  # 60 * 24 * 365
    estimated_time_s = t_total / N_STEPS * steps_per_year
    print(f"\n  Estimated 1-year simulation: {estimated_time_s:.1f} seconds")


if __name__ == '__main__':
    benchmark_property_lookup()
    benchmark_saturation_lookup()
    benchmark_full_step()
    
    print("\n" + "=" * 60)
    print("✅ ALL BENCHMARKS COMPLETE")
    print("=" * 60)
