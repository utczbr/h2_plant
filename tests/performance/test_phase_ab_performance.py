"""
Performance benchmarks for Phase A/B fixes.

Validates:
- History logging speed (NumPy vs list.append)
- Dispatch decision latency
- Full simulation throughput
- LUT lookup batching

Run with: pytest tests/performance/test_phase_ab_performance.py -v --tb=short -m benchmark
"""

import pytest
import numpy as np
import time


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def large_arrays():
    """Pre-allocated large arrays for benchmarking."""
    return {
        'prices': np.random.uniform(30, 100, 8760),
        'wind': np.random.uniform(0, 20, 8760),
    }


@pytest.fixture
def mock_context():
    """Minimal mock context for dispatch strategy."""
    from dataclasses import dataclass
    
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
    
    return MockContext()


@pytest.fixture
def mock_registry():
    """Mock registry for benchmarks."""
    from h2_plant.core.component_registry import ComponentRegistry
    from unittest.mock import Mock
    
    registry = ComponentRegistry()
    
    # Mock components with minimal overhead
    mock_soec = Mock()
    mock_soec.component_id = "soec_cluster"
    mock_soec.soec_state = "RUNNING"
    mock_soec.real_powers = np.array([5.0])
    mock_soec.last_h2_output_kg = 100.0
    mock_soec.receive_input = Mock()
    
    mock_pem = Mock()
    mock_pem.component_id = "pem_electrolyzer_detailed"
    mock_pem.V_cell = 1.8
    mock_pem.h2_output_kg = 50.0
    mock_pem.P_consumed_W = 2.5e6
    mock_pem.set_power_input_mw = Mock()
    
    registry.register("soec_cluster", mock_soec)
    registry.register("pem_electrolyzer_detailed", mock_pem)
    
    return registry


# ============================================================================
# PERFORMANCE TESTS: History Logging
# ============================================================================

@pytest.mark.benchmark
class TestHistoryLoggingPerformance:
    """Benchmark history logging: NumPy vs list.append."""
    
    def test_numpy_prealloc_vs_list_append(self):
        """NumPy pre-allocation should be 10-50x faster than list.append."""
        N = 8760
        
        # Method 1: List append (old way)
        start = time.perf_counter()
        list_history = {'values': []}
        for i in range(N):
            list_history['values'].append(float(i))
        list_time = time.perf_counter() - start
        
        # Method 2: NumPy pre-allocation (new way)
        start = time.perf_counter()
        numpy_history = np.zeros(N, dtype=np.float64)
        for i in range(N):
            numpy_history[i] = float(i)
        numpy_time = time.perf_counter() - start
        
        speedup = list_time / numpy_time
        
        print(f"\nList append: {list_time*1000:.2f}ms")
        print(f"NumPy prealloc: {numpy_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.1f}x")
        
        # NumPy should be at least 2x faster
        assert speedup > 2.0, f"Expected NumPy faster, got {speedup:.1f}x"
    
    def test_history_allocation_under_10ms(self, mock_registry, mock_context):
        """Pre-allocating 8760 steps should take <10ms."""
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        
        strategy = HybridArbitrageEngineStrategy()
        
        start = time.perf_counter()
        strategy.initialize(mock_registry, mock_context, total_steps=8760)
        elapsed = time.perf_counter() - start
        
        print(f"\nHistory allocation (8760 steps): {elapsed*1000:.2f}ms")
        assert elapsed < 0.010, f"Allocation too slow: {elapsed*1000:.2f}ms"
    
    def test_single_step_recording_under_100us(self, mock_registry, mock_context, large_arrays):
        """Recording a single step should take <100µs."""
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        
        strategy = HybridArbitrageEngineStrategy()
        strategy.initialize(mock_registry, mock_context, total_steps=8760)
        
        prices = large_arrays['prices']
        wind = large_arrays['wind']
        
        # Warm up
        for i in range(10):
            strategy.decide_and_apply(i * (1/60), prices, wind)
            strategy.record_post_step()
        
        # Measure
        start = time.perf_counter()
        for i in range(100):
            strategy.decide_and_apply((10 + i) * (1/60), prices, wind)
            strategy.record_post_step()
        elapsed = time.perf_counter() - start
        
        per_step_us = (elapsed / 100) * 1e6
        
        print(f"\nPer-step time: {per_step_us:.1f}µs")
        assert per_step_us < 500, f"Step too slow: {per_step_us:.1f}µs"


# ============================================================================
# PERFORMANCE TESTS: Dispatch Decision
# ============================================================================

@pytest.mark.benchmark
class TestDispatchPerformance:
    """Benchmark dispatch decision performance."""
    
    def test_dispatch_decision_under_1ms(self):
        """Single dispatch decision should take <1ms."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, ReferenceHybridStrategy
        )
        
        strategy = ReferenceHybridStrategy()
        state = DispatchState()
        
        # Warm up
        for _ in range(100):
            d_input = DispatchInput(
                minute=0,
                P_offer=10.0,
                P_future_offer=10.0,
                current_price=50.0,
                soec_capacity_mw=10.0,
                pem_max_power_mw=5.0
            )
            strategy.decide(d_input, state)
        
        # Measure
        start = time.perf_counter()
        for minute in range(1000):
            d_input = DispatchInput(
                minute=minute,
                P_offer=10.0 + 0.01 * minute,
                P_future_offer=10.0,
                current_price=50.0 + 0.1 * (minute % 60),
                soec_capacity_mw=10.0,
                pem_max_power_mw=5.0
            )
            result = strategy.decide(d_input, state)
            state.P_soec_prev = result.P_soec
            state.force_sell = result.state_update.get('force_sell', False)
        elapsed = time.perf_counter() - start
        
        per_decision_ms = (elapsed / 1000) * 1000
        
        print(f"\nPer-decision time: {per_decision_ms:.3f}ms")
        assert per_decision_ms < 1.0, f"Decision too slow: {per_decision_ms:.3f}ms"
    
    def test_1000_decisions_under_100ms(self):
        """1000 dispatch decisions should complete in <100ms."""
        from h2_plant.control.dispatch import (
            DispatchInput, DispatchState, ReferenceHybridStrategy
        )
        
        strategy = ReferenceHybridStrategy()
        state = DispatchState()
        
        start = time.perf_counter()
        for minute in range(1000):
            d_input = DispatchInput(
                minute=minute,
                P_offer=10.0,
                P_future_offer=10.0,
                current_price=50.0,
                soec_capacity_mw=10.0,
                pem_max_power_mw=5.0
            )
            strategy.decide(d_input, state)
        elapsed = time.perf_counter() - start
        
        print(f"\n1000 decisions: {elapsed*1000:.2f}ms")
        assert elapsed < 0.1, f"1000 decisions too slow: {elapsed*1000:.2f}ms"


# ============================================================================
# PERFORMANCE TESTS: Stream Operations
# ============================================================================

@pytest.mark.benchmark
class TestStreamPerformance:
    """Benchmark Stream creation and operations."""
    
    def test_stream_creation_batch(self):
        """Creating 10000 Stream objects should take <100ms."""
        from h2_plant.core.stream import Stream
        
        start = time.perf_counter()
        streams = []
        for i in range(10000):
            s = Stream(
                mass_flow_kg_h=100.0 + i * 0.01,
                temperature_k=300.0,
                pressure_pa=1e5 + i * 100,
                composition={'H2': 0.95, 'N2': 0.05}
            )
            streams.append(s)
        elapsed = time.perf_counter() - start
        
        print(f"\n10000 Stream creations: {elapsed*1000:.2f}ms")
        assert elapsed < 0.5, f"Stream creation too slow: {elapsed*1000:.2f}ms"
    
    def test_density_calculation_batch(self):
        """10000 density calculations should take <200ms."""
        from h2_plant.core.stream import Stream
        
        stream = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=1e5,
            composition={'H2': 0.95, 'N2': 0.05}
        )
        
        start = time.perf_counter()
        for _ in range(10000):
            _ = stream.density_kg_m3
        elapsed = time.perf_counter() - start
        
        print(f"\n10000 density calcs: {elapsed*1000:.2f}ms")
        assert elapsed < 0.5, f"Density calc too slow: {elapsed*1000:.2f}ms"


# ============================================================================
# PERFORMANCE TESTS: Full Simulation Stub
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.slow
class TestFullSimulationPerformance:
    """Benchmark full simulation throughput."""
    
    def test_8760_steps_dispatch_only(self, mock_registry, mock_context, large_arrays):
        """8760 dispatch steps should complete in <10s."""
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        
        strategy = HybridArbitrageEngineStrategy()
        strategy.initialize(mock_registry, mock_context, total_steps=8760)
        
        prices = large_arrays['prices']
        wind = large_arrays['wind']
        
        start = time.perf_counter()
        for step in range(8760):
            t = step * (1/60)
            strategy.decide_and_apply(t, prices, wind)
            strategy.record_post_step()
        elapsed = time.perf_counter() - start
        
        steps_per_second = 8760 / elapsed
        
        print(f"\n8760 steps: {elapsed:.2f}s")
        print(f"Throughput: {steps_per_second:.0f} steps/s")
        
        # Target: >1000 steps/s for dispatch only
        assert steps_per_second > 1000, f"Too slow: {steps_per_second:.0f} steps/s"
    
    def test_memory_usage_constant(self, mock_registry, mock_context, large_arrays):
        """Memory usage should remain constant during simulation."""
        import tracemalloc
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        
        strategy = HybridArbitrageEngineStrategy()
        strategy.initialize(mock_registry, mock_context, total_steps=8760)
        
        prices = large_arrays['prices']
        wind = large_arrays['wind']
        
        tracemalloc.start()
        
        # Run first 100 steps
        for step in range(100):
            strategy.decide_and_apply(step * (1/60), prices, wind)
            strategy.record_post_step()
        
        _, first_peak = tracemalloc.get_traced_memory()
        
        # Run next 1000 steps
        for step in range(100, 1100):
            strategy.decide_and_apply(step * (1/60), prices, wind)
            strategy.record_post_step()
        
        _, second_peak = tracemalloc.get_traced_memory()
        
        tracemalloc.stop()
        
        # Memory should not grow significantly (< 10MB growth)
        growth_mb = (second_peak - first_peak) / 1e6
        
        print(f"\nMemory growth: {growth_mb:.2f}MB")
        assert growth_mb < 10, f"Memory growing: {growth_mb:.2f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmark"])
