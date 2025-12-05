"""
Unit tests for LUT Manager.

Validates lookup table generation, interpolation accuracy, and performance.
"""

import pytest
import numpy as np
from h2_plant.optimization.lut_manager import LUTManager, LUTConfig


class TestLUTManager:
    """Test suite for LUT Manager."""

    @pytest.fixture
    def lut_manager(self):
        """Create LUT manager instance."""
        from h2_plant.core.component_registry import ComponentRegistry
        config = LUTConfig(
            pressure_points=20,  # Reduced for faster tests
            temperature_points=10
        )
        manager = LUTManager(config=config)
        manager.initialize(dt=1.0, registry=ComponentRegistry())
        return manager
    def test_lut_initialization(self, lut_manager):
        """Test LUT manager initializes successfully."""
        assert lut_manager._initialized
        assert 'H2' in lut_manager._luts

    def test_lookup_returns_float(self, lut_manager):
        """Test lookup returns float value."""
        density = lut_manager.lookup('H2', 'D', pressure=350e5, temperature=298.15)

        assert isinstance(density, float)
        assert density > 0

    def test_lookup_accuracy(self, lut_manager):
        """Test LUT interpolation accuracy vs CoolProp."""
        try:
            import CoolProp.CoolProp as CP
        except ImportError:
            pytest.skip("CoolProp not available")

        # Test point within LUT bounds
        pressure = 350e5
        temperature = 298.15

        lut_density = lut_manager.lookup('H2', 'D', pressure, temperature)
        cp_density = CP.PropsSI('D', 'P', pressure, 'T', temperature, 'H2')

        relative_error = abs(lut_density - cp_density) / cp_density

        assert relative_error < 0.005  # <0.5% error

    def test_batch_lookup(self, lut_manager):
        """Test vectorized batch lookup."""
        pressures = np.array([30e5, 350e5, 900e5])
        temperatures = np.array([298.15, 298.15, 298.15])

        densities = lut_manager.lookup_batch('H2', 'D', pressures, temperatures)

        assert len(densities) == 3
        assert all(d > 0 for d in densities)

    @pytest.mark.benchmark
    def test_lookup_performance(self, lut_manager, benchmark):
        """Benchmark LUT lookup performance."""

        def lookup_operation():
            return lut_manager.lookup('H2', 'D', 350e5, 298.15)

        result = benchmark(lookup_operation)

        # Should be <0.1ms (100 microseconds)
        assert benchmark.stats['mean'] < 0.0001


class TestNumbaOps:
    """Test suite for Numba-compiled operations."""

    def test_find_available_tank(self):
        """Test find_available_tank function."""
        from h2_plant.optimization.numba_ops import find_available_tank
        from h2_plant.core.enums import TankState

        states = np.array([TankState.FULL, TankState.IDLE, TankState.IDLE], dtype=np.int32)
        masses = np.array([200.0, 50.0, 0.0], dtype=np.float64)
        capacities = np.array([200.0, 200.0, 200.0], dtype=np.float64)

        idx = find_available_tank(states, masses, capacities, min_capacity=100.0)

        # Should return tank 1 (has 150 kg available) or tank 2 (has 200 kg available)
        assert idx in [1, 2]

    def test_batch_pressure_update(self):
        """Test batch pressure update calculation."""
        from h2_plant.optimization.numba_ops import batch_pressure_update
        
        masses = np.array([100.0, 150.0, 200.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        pressures = np.zeros_like(masses)
        temperature = 298.15
        
        batch_pressure_update(masses, volumes, pressures, temperature)
        
        assert len(pressures) == 3
        assert all(p > 0 for p in pressures)
        assert pressures[2] > pressures[1] > pressures[0]

    @pytest.mark.benchmark
    def test_numba_compilation_overhead(self, benchmark):
        """Test Numba JIT compilation is cached."""
        from h2_plant.optimization.numba_ops import calculate_compression_work
        
        # First call triggers compilation (excluded from benchmark)
        calculate_compression_work(30e5, 350e5, 50.0, 298.15)
        
        # Subsequent calls use compiled version
        def compression_calc():
            return calculate_compression_work(30e5, 350e5, 50.0, 298.15)
        
        result = benchmark(compression_calc)
        
        # Should be very fast (microseconds)
        assert benchmark.stats['mean'] < 0.000085 # Relaxed to < 85 microseconds


# Coverage target: 95% for optimization module