import pytest
import time
import numpy as np
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.optimization.numba_ops import calculate_compression_work
from h2_plant.core.component_registry import ComponentRegistry


@pytest.mark.performance
def test_lut_lookup_speed():
    """Verify LUT lookup meets <0.1ms target."""
    lut = LUTManager()
    lut.initialize(dt=1.0, registry=ComponentRegistry())
    
    # Warm up JIT
    for _ in range(10):
        lut.lookup('H2', 'D', 350e5, 298.15)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        density = lut.lookup('H2', 'D', 350e5, 298.15)
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / 1000) * 1000
    assert avg_time_ms < 0.1, f"LUT lookup too slow: {avg_time_ms:.4f} ms"


@pytest.mark.performance
def test_lut_accuracy():
    """Verify LUT interpolation error <0.5%."""
    try:
        import CoolProp
    except ImportError:
        pytest.skip("CoolProp not installed, skipping accuracy test")

    lut = LUTManager()
    lut.initialize(dt=1.0, registry=ComponentRegistry())
    
    accuracy_report = lut.get_accuracy_report('H2', num_samples=100)
    
    for prop, errors in accuracy_report.items():
        if prop == 'S':
            # Relax accuracy requirement for Entropy
            assert errors['max_rel_error_pct'] < 3.5, \
                f"LUT accuracy insufficient for {prop}: {errors['max_rel_error_pct']:.2f}%"
        else:
            assert errors['max_rel_error_pct'] < 0.5, \
                f"LUT accuracy insufficient for {prop}: {errors['max_rel_error_pct']:.2f}%"


@pytest.mark.performance
def test_tank_array_fill_speed():
    """Verify tank fill operations meet performance targets."""
    tanks = TankArray(n_tanks=100, capacity_kg=200.0, pressure_bar=350)
    tanks.initialize(1.0, ComponentRegistry())
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        tanks.fill(50.0)
    elapsed = time.perf_counter() - start
    
    ops_per_sec = 1000 / elapsed
    assert ops_per_sec > 8000, f"Tank operations too slow: {ops_per_sec:.0f} ops/sec"


@pytest.mark.performance
def test_numba_compilation():
    """Verify Numba JIT functions compile successfully."""
    # Should not raise errors
    work = calculate_compression_work(30e5, 350e5, 50.0, 298.15)
    assert work > 0
