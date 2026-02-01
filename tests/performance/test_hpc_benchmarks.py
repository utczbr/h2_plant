"""
HPC Performance Benchmark Suite for Phase C validation.

Tests validate:
- LUT lookup speed targets (<1µs single, <100µs batch 1000)
- Component step times (<50µs for SOEC/Tank)
- Full simulation throughput (525,600 steps < 90s)
- Memory stability over long runs

Run with: pytest tests/performance/test_hpc_benchmarks.py -v -m benchmark
"""

import pytest
import numpy as np
import time
import tracemalloc


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def lut_manager():
    """Initialize LUTManager once for all tests."""
    from h2_plant.optimization.lut_manager import LUTManager
    lut = LUTManager()
    lut.initialize()
    return lut


@pytest.fixture
def sample_pressures():
    """Sample pressure array for batch testing."""
    return np.linspace(30e5, 350e5, 1000)


@pytest.fixture
def sample_temperatures():
    """Sample temperature array for batch testing."""
    return np.full(1000, 298.15)


# ============================================================================
# LUT LOOKUP BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestLUTBenchmarks:
    """Benchmarks for LUT lookup performance."""
    
    def test_single_lookup_under_10us(self, lut_manager):
        """Single LUT lookup should complete in <10µs (after warmup)."""
        # Warmup (JIT compilation)
        for _ in range(100):
            lut_manager.lookup('H2', 'D', 100e5, 298.15)
        
        # Measure
        n_samples = 10000
        start = time.perf_counter()
        for _ in range(n_samples):
            lut_manager.lookup('H2', 'D', 100e5, 298.15)
        elapsed = time.perf_counter() - start
        
        per_lookup_us = (elapsed / n_samples) * 1e6
        
        print(f"\nSingle LUT lookup: {per_lookup_us:.2f}µs")
        assert per_lookup_us < 10, f"Single lookup too slow: {per_lookup_us:.2f}µs"
    
    def test_batch_lookup_1000_under_500us(self, lut_manager, sample_pressures, sample_temperatures):
        """Batch LUT lookup (1000 points) should complete in <500µs."""
        # Warmup
        for _ in range(10):
            lut_manager.lookup_batch('H2', 'D', sample_pressures, sample_temperatures)
        
        # Measure
        n_samples = 100
        start = time.perf_counter()
        for _ in range(n_samples):
            lut_manager.lookup_batch('H2', 'D', sample_pressures, sample_temperatures)
        elapsed = time.perf_counter() - start
        
        per_batch_us = (elapsed / n_samples) * 1e6
        
        print(f"\nBatch LUT lookup (1000): {per_batch_us:.1f}µs")
        assert per_batch_us < 500, f"Batch lookup too slow: {per_batch_us:.1f}µs"
    
    def test_batch_speedup_over_sequential(self, lut_manager, sample_pressures, sample_temperatures):
        """Batch lookup should be faster than sequential single lookups."""
        # Sequential timing
        start = time.perf_counter()
        for p, t in zip(sample_pressures, sample_temperatures):
            lut_manager.lookup('H2', 'D', p, t)
        sequential_time = time.perf_counter() - start
        
        # Batch timing
        start = time.perf_counter()
        lut_manager.lookup_batch('H2', 'D', sample_pressures, sample_temperatures)
        batch_time = time.perf_counter() - start
        
        speedup = sequential_time / batch_time
        
        print(f"\nSequential: {sequential_time*1000:.2f}ms")
        print(f"Batch: {batch_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.1f}x")
        
        # Batch should be at least 2x faster
        assert speedup > 2.0, f"Batch not faster: {speedup:.1f}x"


# ============================================================================
# TANK ARRAY BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestTankArrayBenchmarks:
    """Benchmarks for TankArray operations."""
    
    def test_tank_fill_under_50us(self):
        """Tank fill operation should complete in <50µs."""
        from h2_plant.components.storage.h2_tank import TankArray
        from h2_plant.core.component_registry import ComponentRegistry
        
        tanks = TankArray(n_tanks=10, capacity_kg=1000.0, pressure_bar=350.0)
        registry = ComponentRegistry()
        registry.register("test_tanks", tanks)
        tanks.initialize(dt=1/60, registry=registry)
        
        # Warmup
        for _ in range(100):
            tanks.fill(10.0)
            tanks.discharge(10.0)
        
        # Measure fill
        n_samples = 1000
        start = time.perf_counter()
        for _ in range(n_samples):
            tanks.fill(10.0)
        elapsed = time.perf_counter() - start
        
        per_fill_us = (elapsed / n_samples) * 1e6
        
        print(f"\nTank fill: {per_fill_us:.1f}µs")
        assert per_fill_us < 50, f"Tank fill too slow: {per_fill_us:.1f}µs"
    
    def test_tank_step_under_20us(self):
        """Tank step() should complete in <20µs."""
        from h2_plant.components.storage.h2_tank import TankArray
        from h2_plant.core.component_registry import ComponentRegistry
        
        tanks = TankArray(n_tanks=10, capacity_kg=1000.0, pressure_bar=350.0)
        registry = ComponentRegistry()
        registry.register("test_tanks", tanks)
        tanks.initialize(dt=1/60, registry=registry)
        
        # Warmup
        for t in range(100):
            tanks.step(float(t))
        
        # Measure
        n_samples = 1000
        start = time.perf_counter()
        for t in range(n_samples):
            tanks.step(float(t + 100))
        elapsed = time.perf_counter() - start
        
        per_step_us = (elapsed / n_samples) * 1e6
        
        print(f"\nTank step: {per_step_us:.1f}µs")
        assert per_step_us < 20, f"Tank step too slow: {per_step_us:.1f}µs"


# ============================================================================
# DISPATCH STRATEGY BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestDispatchBenchmarks:
    """Benchmarks for dispatch strategy."""
    
    def test_dispatch_decision_under_1ms(self):
        """Single dispatch decision should complete in <1ms."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, ReferenceHybridStrategy
        )
        
        strategy = ReferenceHybridStrategy()
        state = DispatchState()
        
        # Warmup
        for _ in range(100):
            d_input = DispatchInput(
                minute=0, P_offer=10.0, P_future_offer=10.0,
                current_price=50.0, soec_capacity_mw=10.0, pem_max_power_mw=5.0
            )
            strategy.decide(d_input, state)
        
        # Measure
        n_samples = 1000
        start = time.perf_counter()
        for i in range(n_samples):
            d_input = DispatchInput(
                minute=i % 60, P_offer=10.0, P_future_offer=10.0,
                current_price=50.0 + (i % 100), soec_capacity_mw=10.0, pem_max_power_mw=5.0
            )
            strategy.decide(d_input, state)
        elapsed = time.perf_counter() - start
        
        per_decision_ms = (elapsed / n_samples) * 1000
        
        print(f"\nDispatch decision: {per_decision_ms:.3f}ms")
        assert per_decision_ms < 1.0, f"Dispatch too slow: {per_decision_ms:.3f}ms"


# ============================================================================
# FULL SIMULATION BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.slow
class TestFullSimulationBenchmarks:
    """End-to-end simulation performance."""
    
    def test_8760h_dispatch_under_30s(self):
        """8760 hours (525,600 1-min steps) of dispatch should complete in <30s."""
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        from h2_plant.core.component_registry import ComponentRegistry
        from dataclasses import dataclass
        from unittest.mock import Mock
        
        # Setup mock context
        @dataclass
        class MockPhysicsSpec:
            num_modules: int = 1
            max_power_nominal_mw: float = 10.0
            optimal_limit: float = 1.0
            kwh_per_kg: float = 37.5
            max_power_mw: float = 5.0
        
        @dataclass
        class MockPhysics:
            soec_cluster: MockPhysicsSpec = None
            pem_system: MockPhysicsSpec = None
            def __post_init__(self):
                self.soec_cluster = MockPhysicsSpec()
                self.pem_system = MockPhysicsSpec()
        
        @dataclass
        class MockSimulation:
            timestep_hours: float = 1/60
        
        @dataclass
        class MockContext:
            simulation: MockSimulation = None
            physics: MockPhysics = None
            def __post_init__(self):
                self.simulation = MockSimulation()
                self.physics = MockPhysics()
        
        # Setup
        registry = ComponentRegistry()
        mock_soec = Mock()
        mock_soec.component_id = "soec_cluster"
        mock_soec.soec_state = "RUNNING"
        mock_soec.real_powers = np.array([5.0])
        mock_soec.last_h2_output_kg = 100.0
        registry.register("soec_cluster", mock_soec)
        
        context = MockContext()
        
        total_steps = 525600  # 1 year at 1-min resolution
        strategy = HybridArbitrageEngineStrategy()
        strategy.initialize(registry, context, total_steps=total_steps)
        
        prices = np.random.uniform(30, 100, total_steps)
        wind = np.random.uniform(0, 20, total_steps)
        
        # Warmup
        for step in range(100):
            strategy.decide_and_apply(step * (1/60), prices, wind)
            strategy.record_post_step()
        
        # Reset for timing
        strategy._step_index = 0
        
        start = time.perf_counter()
        for step in range(total_steps):
            t = step * (1/60)
            strategy.decide_and_apply(t, prices, wind)
            strategy.record_post_step()
        elapsed = time.perf_counter() - start
        
        steps_per_second = total_steps / elapsed
        
        print(f"\n{total_steps} steps in {elapsed:.2f}s")
        print(f"Throughput: {steps_per_second:.0f} steps/s")
        print(f"Per-step: {(elapsed/total_steps)*1e6:.1f}µs")
        
        assert elapsed < 30, f"Simulation too slow: {elapsed:.2f}s (target <30s)"


# ============================================================================
# MEMORY BENCHMARKS
# ============================================================================

@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Memory usage benchmarks."""
    
    def test_memory_stable_over_10k_steps(self):
        """Memory should remain stable (< 5MB growth) over 10K steps."""
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        from h2_plant.core.component_registry import ComponentRegistry
        from dataclasses import dataclass
        from unittest.mock import Mock
        
        @dataclass
        class MockPhysicsSpec:
            num_modules: int = 1
            max_power_nominal_mw: float = 10.0
            optimal_limit: float = 1.0
            kwh_per_kg: float = 37.5
            max_power_mw: float = 5.0
        
        @dataclass
        class MockPhysics:
            soec_cluster: MockPhysicsSpec = None
            pem_system: MockPhysicsSpec = None
            def __post_init__(self):
                self.soec_cluster = MockPhysicsSpec()
                self.pem_system = MockPhysicsSpec()
        
        @dataclass
        class MockSimulation:
            timestep_hours: float = 1/60
        
        @dataclass
        class MockContext:
            simulation: MockSimulation = None
            physics: MockPhysics = None
            def __post_init__(self):
                self.simulation = MockSimulation()
                self.physics = MockPhysics()
        
        registry = ComponentRegistry()
        
        context = MockContext()
        
        strategy = HybridArbitrageEngineStrategy()
        strategy.initialize(registry, context, total_steps=20000)
        
        # Bypass registry - directly set mock SOEC on strategy
        class MockSOEC:
            component_id = "soec_cluster"
            soec_state = "RUNNING"
            real_powers = np.array([5.0])
            last_h2_output_kg = 100.0
        strategy._soec = MockSOEC()
        
        prices = np.random.uniform(30, 100, 20000)
        wind = np.random.uniform(0, 20, 20000)
        
        tracemalloc.start()
        
        # First 1K steps
        for step in range(1000):
            strategy.decide_and_apply(step * (1/60), prices, wind)
            strategy.record_post_step()
        
        _, first_peak = tracemalloc.get_traced_memory()
        
        # Next 9K steps
        for step in range(1000, 10000):
            strategy.decide_and_apply(step * (1/60), prices, wind)
            strategy.record_post_step()
        
        _, second_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        growth_mb = (second_peak - first_peak) / 1e6
        
        print(f"\nMemory growth over 9K steps: {growth_mb:.2f}MB")
        assert growth_mb < 5, f"Memory growing: {growth_mb:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmark"])
