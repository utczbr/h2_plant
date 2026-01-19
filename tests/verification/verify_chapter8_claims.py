#!/usr/bin/env python3
"""
Chapter 8 Claims Verification Script (Uses existing LUT cache)

Verifies:
1. LUT Performance: Speedup factor vs. direct CoolProp calls
2. LUT Accuracy: Error deviation vs. CoolProp ground truth  
3. Time-Stepping Architecture: Whether dt is a variable parameter
"""

import sys
import time
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

def benchmark_lut_vs_coolprop(n_iterations: int = 10000):
    """
    Benchmark LUT lookups against direct CoolProp calls.
    
    CRITICAL: Tests both scalar and vectorized performance to measure
    the ACTUAL speedup in simulation context.
    """
    print("\n" + "=" * 70)
    print("PHASE 1.1: LUT vs. CoolProp PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    try:
        import CoolProp.CoolProp as CP
    except ImportError:
        print("ERROR: CoolProp not installed")
        return {"error": "CoolProp unavailable"}
    
    from h2_plant.optimization.lut_manager import LUTManager
    from h2_plant.optimization.numba_ops import bilinear_interp_jit
    
    # Use default config (matches existing cache)
    lut = LUTManager()
    lut.initialize()
    
    # Random test points within LUT bounds
    np.random.seed(42)
    pressures = np.random.uniform(1e5, 100e5, n_iterations)
    temperatures = np.random.uniform(300, 800, n_iterations)
    
    # =========================================================================
    # WARM-UP: JIT compile all functions before timing
    # =========================================================================
    print("\nWarming up JIT functions...")
    _ = lut.lookup('H2', 'H', pressures[0], temperatures[0])
    _ = bilinear_interp_jit(lut._pressure_grid, lut._temperature_grid, 
                            lut._luts['H2']['H'], pressures[0], temperatures[0])
    print("  JIT warm-up complete.")
    
    # =========================================================================
    # TEST 1: Scalar lookups (Python overhead included)
    # =========================================================================
    print(f"\n--- TEST 1: SCALAR LOOKUPS ({n_iterations} calls) ---")
    
    # CoolProp scalar
    start = time.perf_counter()
    for i in range(n_iterations):
        CP.PropsSI('H', 'P', pressures[i], 'T', temperatures[i], 'H2')
    coolprop_scalar_time = time.perf_counter() - start
    print(f"  CoolProp scalar: {coolprop_scalar_time:.3f}s ({n_iterations/coolprop_scalar_time:.0f}/sec)")
    
    # LUT scalar (via Python wrapper)
    start = time.perf_counter()
    for i in range(n_iterations):
        lut.lookup('H2', 'H', pressures[i], temperatures[i])
    lut_scalar_time = time.perf_counter() - start
    print(f"  LUT scalar:      {lut_scalar_time:.3f}s ({n_iterations/lut_scalar_time:.0f}/sec)")
    
    scalar_speedup = coolprop_scalar_time / lut_scalar_time
    print(f"  Scalar speedup: {scalar_speedup:.1f}x")
    
    # =========================================================================
    # TEST 2: Direct JIT kernel (no Python wrapper overhead)
    # =========================================================================
    print(f"\n--- TEST 2: DIRECT JIT KERNEL ({n_iterations} calls) ---")
    
    p_grid = lut._pressure_grid
    t_grid = lut._temperature_grid
    h_lut = lut._luts['H2']['H']
    
    # Direct JIT calls
    start = time.perf_counter()
    for i in range(n_iterations):
        bilinear_interp_jit(p_grid, t_grid, h_lut, pressures[i], temperatures[i])
    jit_direct_time = time.perf_counter() - start
    print(f"  JIT direct:  {jit_direct_time:.3f}s ({n_iterations/jit_direct_time:.0f}/sec)")
    
    jit_speedup = coolprop_scalar_time / jit_direct_time
    print(f"  JIT speedup: {jit_speedup:.1f}x")
    
    # =========================================================================
    # TEST 3: Vectorized batch (the TRUE simulation use case)
    # =========================================================================
    print(f"\n--- TEST 3: VECTORIZED BATCH ({n_iterations} points) ---")
    
    try:
        from h2_plant.optimization.numba_ops import batch_bilinear_interp_jit
        
        # Warm up batch function
        _ = batch_bilinear_interp_jit(p_grid, t_grid, h_lut, 
                                       pressures[:10], temperatures[:10])
        
        # CoolProp vectorized (still uses loop internally)
        start = time.perf_counter()
        cp_results = np.array([CP.PropsSI('H', 'P', p, 'T', t, 'H2') 
                               for p, t in zip(pressures, temperatures)])
        coolprop_batch_time = time.perf_counter() - start
        print(f"  CoolProp batch: {coolprop_batch_time:.3f}s")
        
        # LUT vectorized
        start = time.perf_counter()
        lut_results = batch_bilinear_interp_jit(p_grid, t_grid, h_lut,
                                                 np.ascontiguousarray(pressures),
                                                 np.ascontiguousarray(temperatures))
        lut_batch_time = time.perf_counter() - start
        print(f"  LUT batch:      {lut_batch_time:.3f}s")
        
        batch_speedup = coolprop_batch_time / lut_batch_time
        print(f"  Batch speedup: {batch_speedup:.1f}x")
    except ImportError:
        batch_speedup = None
        print("  (batch_bilinear_interp_jit not available)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*50}")
    print(f"SPEEDUP RESULTS:")
    print(f"  Scalar (Python wrapper): {scalar_speedup:.1f}x")
    print(f"  Direct JIT kernel:       {jit_speedup:.1f}x")
    if batch_speedup:
        print(f"  Vectorized batch:        {batch_speedup:.1f}x")
    print(f"{'='*50}")
    
    # Use the most representative speedup (JIT direct or batch)
    representative_speedup = batch_speedup if batch_speedup else jit_speedup
    
    if 50 <= representative_speedup <= 200:
        print(f"✅ VERIFIED: {representative_speedup:.0f}x within claimed [50-200x]")
    elif representative_speedup > 200:
        print(f"✅ EXCEEDS CLAIM: {representative_speedup:.0f}x above maximum")
    elif representative_speedup > 10:
        print(f"⚠️ PARTIAL: {representative_speedup:.0f}x significant but below 50x claim")
    else:
        print(f"❌ BELOW CLAIM: {representative_speedup:.0f}x")
    
    return {
        "scalar_speedup": scalar_speedup,
        "jit_speedup": jit_speedup,
        "batch_speedup": batch_speedup,
        "representative": representative_speedup
    }


def benchmark_lut_accuracy(n_samples: int = 500):
    """Compare LUT values against CoolProp ground truth."""
    print("\n" + "=" * 70)
    print("PHASE 1.2: LUT ACCURACY VERIFICATION")
    print("=" * 70)
    
    try:
        import CoolProp.CoolProp as CP
    except ImportError:
        return {"error": "CoolProp unavailable"}
    
    from h2_plant.optimization.lut_manager import LUTManager
    
    lut = LUTManager()
    lut.initialize()
    
    np.random.seed(123)
    fluids = ['H2', 'O2', 'N2']
    all_errors = []
    
    for fluid in fluids:
        print(f"\nTesting {fluid}...")
        pressures = np.random.uniform(1e5, 100e5, n_samples)
        temperatures = np.random.uniform(300, 800, n_samples)
        
        errors = []
        for i in range(n_samples):
            try:
                lut_val = lut.lookup(fluid, 'H', pressures[i], temperatures[i])
                cp_val = CP.PropsSI('H', 'P', pressures[i], 'T', temperatures[i], fluid)
                if abs(cp_val) > 1e-10:
                    errors.append(abs(lut_val - cp_val) / abs(cp_val) * 100)
            except:
                continue
        
        if errors:
            print(f"  Mean: {np.mean(errors):.4f}%, Max: {np.max(errors):.4f}%, P95: {np.percentile(errors, 95):.4f}%")
            all_errors.extend(errors)
    
    p95 = np.percentile(all_errors, 95)
    print(f"\n{'='*50}")
    print(f"OVERALL P95 ERROR: {p95:.4f}%")
    
    if p95 < 0.1:
        print(f"✅ Chapter 8 claim (<0.1%) VERIFIED")
    elif p95 < 0.5:
        print(f"⚠️ LUT docstring claim (<0.5%) verified, but Chapter 8 (<0.1%) FAILS")
    else:
        print(f"❌ Both claims FAIL")
    
    return {"p95_error": p95, "mean_error": np.mean(all_errors)}


def verify_timestep_architecture():
    """Verify if architecture supports variable time-stepping."""
    print("\n" + "=" * 70)
    print("PHASE 1.3: TIME-STEPPING ARCHITECTURE CHECK")
    print("=" * 70)
    
    from h2_plant.core.component import Component
    import inspect
    
    # Check Component.initialize signature
    sig = inspect.signature(Component.initialize)
    params = list(sig.parameters.keys())
    
    print(f"\n1. Component.initialize() signature: {sig}")
    has_dt = 'dt' in params
    print(f"   'dt' parameter present: {'✅ YES' if has_dt else '❌ NO'}")
    
    # Check if dt is stored
    class TestComp(Component):
        def initialize(self, dt, registry):
            super().initialize(dt, registry)
        def step(self, t): pass
        def get_state(self): return {}
    
    class MockReg:
        def has(self, x): return False
    
    tc = TestComp()
    tc.initialize(0.5, MockReg())  # 30-min timestep
    stored = hasattr(tc, 'dt') and tc.dt == 0.5
    print(f"\n2. dt stored in component: {'✅ YES' if stored else '❌ NO'} (dt={tc.dt})")
    
    # Check engine source
    engine_file = project_root / "h2_plant" / "simulation" / "engine.py"
    src = engine_file.read_text()
    fixed_default = "1.0 / 60.0" in src or "1/60" in src
    
    print(f"\n3. Engine default timestep hardcoded: {'YES (1 minute)' if fixed_default else 'NO'}")
    
    print(f"\n{'='*50}")
    if has_dt and stored:
        print("✅ ARCHITECTURE SUPPORTS VARIABLE dt")
        if fixed_default:
            print("   (Current implementation uses FIXED 1-minute default)")
    else:
        print("❌ Architecture does NOT support variable dt")
    
    return {"supports_variable_dt": has_dt and stored, "fixed_default": fixed_default}


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHAPTER 8 CLAIMS VERIFICATION SUITE")
    print("=" * 70)
    
    r1 = benchmark_lut_vs_coolprop(5000)
    r2 = benchmark_lut_accuracy(500)
    r3 = verify_timestep_architecture()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("="*70)
    if 'representative' in r1:
        print(f"1. Representative Speedup: {r1['representative']:.1f}x")
        print(f"   (Scalar: {r1['scalar_speedup']:.1f}x, JIT: {r1['jit_speedup']:.1f}x, Batch: {r1.get('batch_speedup', 'N/A')}x)")
    else:
        print(f"1. Error: {r1}")
    print(f"2. P95 Error: {r2.get('p95_error', 'N/A'):.4f}%" if 'p95_error' in r2 else f"2. Error: {r2}")
    print(f"3. Variable dt: {'YES' if r3.get('supports_variable_dt') else 'NO'}")
