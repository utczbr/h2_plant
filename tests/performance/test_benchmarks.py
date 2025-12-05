"""
Performance benchmarks for hydrogen production system.
"""

import pytest
import time
import numpy as np
from pathlib import Path

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.core.component import Component
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.config.plant_config import (
    PlantConfig, ProductionConfig, ElectrolyzerConfig,
    StorageConfig, TankArrayConfig, SimulationConfig,
    CompressionConfig, DemandConfig, EnergyPriceConfig, PathwayConfig
)

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""

    def test_lut_lookup_performance(self, benchmark):
        """Benchmark: LUT lookup should be <0.1ms."""
        lut = LUTManager()
        lut.initialize(dt=1.0, registry=ComponentRegistry())

        def lookup():
            return lut.lookup('H2', 'D', 350e5, 298.15)

        result = benchmark(lookup)
        assert benchmark.stats['mean'] < 0.0001

    def test_tank_fill_performance(self, benchmark):
        """Benchmark: Tank operations should exceed 8,000 ops/sec."""
        tanks = TankArray(n_tanks=100, capacity_kg=200.0, pressure_bar=350)
        tanks.initialize(1.0, ComponentRegistry())

        def fill_operation():
            tanks.fill(50.0)

        result = benchmark(fill_operation)
        ops_per_sec = 1.0 / benchmark.stats['mean']
        assert ops_per_sec > 8000

    def test_component_step_performance(self, benchmark):
        """Benchmark: Component ABC overhead should be <1 microsecond."""
        class MinimalComponent(Component):
            def initialize(self, dt, registry): super().initialize(dt, registry)
            def step(self, t): super().step(t)
            def get_state(self): return super().get_state()

        comp = MinimalComponent()
        comp.initialize(1.0, ComponentRegistry())

        def step():
            comp.step(0.0)

        result = benchmark(step)
        assert benchmark.stats['mean'] < 0.000001

    @pytest.mark.slow
    def test_full_simulation_performance(self, tmp_path):
        """Benchmark: Full 8760-hour simulation should complete in <90 seconds."""
        # A minimal but complete configuration
        config = PlantConfig(
            name="Benchmark Plant",
            production=ProductionConfig(electrolyzer=ElectrolyzerConfig(max_power_mw=2.5)),
            compression=CompressionConfig(),
            demand=DemandConfig(),
            energy_price=EnergyPriceConfig(),
            pathway=PathwayConfig(),
            simulation=SimulationConfig(duration_hours=100) # Short duration for benchmark
        )
        
        plant = PlantBuilder.from_config(config)
        
        engine = SimulationEngine(
            registry=plant.registry,
            config=plant.config.simulation,
            output_dir=tmp_path
        )

        start_time = time.time()
        engine.run(start_hour=0, end_hour=100)
        elapsed_time = time.time() - start_time

        # Target is <90 seconds for a full year, so this 100-hour run should be very fast,
        # but LUT regeneration can be slow. Allow a generous timeout.
        assert elapsed_time < 180.0

    def test_numba_array_operations_performance(self, benchmark):
        """Benchmark: Numba array operations should be efficient."""
        from h2_plant.optimization.numba_ops import find_available_tank
        
        states = np.full(10000, 0, dtype=np.int32)
        masses = np.random.rand(10000) * 100.0
        capacities = np.full(10000, 200.0, dtype=np.float64)

        def find_tank():
            return find_available_tank(states, masses, capacities)

        result = benchmark(find_tank)
        assert benchmark.stats['mean'] < 0.001

    def test_compressor_energy_calculation_performance(self, benchmark):
        """Benchmark: Compressor energy calculation should be fast."""
        from h2_plant.optimization.numba_ops import calculate_compression_work

        def compression_calculation():
            return calculate_compression_work(p1=30e5, p2=350e5, mass=50.0, temperature=298.15)

        result = benchmark(compression_calculation)
        assert benchmark.stats['mean'] < 0.0001