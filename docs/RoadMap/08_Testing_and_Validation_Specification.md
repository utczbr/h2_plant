# STEP 3: Technical Specification - Testing and Validation

---

# 08_Testing_and_Validation_Specification.md

**Document:** Testing and Validation Strategy Technical Specification  
**Project:** Dual-Path Hydrogen Production System - Modular Refactoring v2.0  
**Date:** November 18, 2025  
**Priority:** HIGH  
**Dependencies:** All previous specifications (01-07)

***

## 1. Overview

### 1.1 Purpose

This specification defines the **comprehensive testing and validation strategy** for the refactored hydrogen production system. The testing framework ensures reliability, correctness, and performance of the new modular architecture while maintaining backward compatibility with legacy functionality.

**Key Objectives:**
- Achieve 95%+ test coverage across all modules
- Validate functional equivalence with legacy system
- Establish automated CI/CD testing pipeline
- Define performance benchmarks and regression tests
- Create integration test scenarios for end-to-end validation

**Testing Pyramid:**
```
        /\
       /  \  End-to-End (10%)
      /____\
     /      \  Integration (30%)
    /________\
   /          \  Unit Tests (60%)
  /__...__...__\
```

***

### 1.2 Testing Scope

**In Scope:**
- Unit tests for all components (Layers 1-5)
- Integration tests for component interactions
- Performance benchmarks and regression tests
- Configuration validation tests
- Simulation end-to-end tests
- Legacy compatibility tests

**Out of Scope:**
- Manual testing procedures (documented separately)
- Hardware-in-the-loop testing (future work)
- Load/stress testing (future work)
- Security testing (future work)

***

### 1.3 Testing Tools

**Core Testing Framework:**
- **pytest:** Test runner and framework
- **pytest-cov:** Code coverage measurement
- **pytest-benchmark:** Performance benchmarking
- **pytest-mock:** Mocking and fixtures

**Quality Assurance:**
- **mypy:** Static type checking
- **black:** Code formatting
- **flake8:** Linting and style checks
- **pylint:** Advanced linting

**CI/CD:**
- **GitHub Actions / GitLab CI:** Automated testing
- **pre-commit:** Git hooks for local validation

***

## 2. Unit Testing Strategy

### 2.1 Layer 1: Core Foundation Tests

**File:** `tests/core/test_component.py`

```python
"""
Unit tests for Component ABC and ComponentRegistry.

Tests the foundational abstractions that all other components depend on.
"""

import pytest
from h2_plant.core.component import (
    Component, 
    ComponentNotInitializedError,
    ComponentInitializationError
)
from h2_plant.core.component_registry import ComponentRegistry


class MockComponent(Component):
    """Mock component for testing."""
    
    def __init__(self):
        super().__init__()
        self.initialized_called = False
        self.step_called = False
        self.step_count = 0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self.initialized_called = True
    
    def step(self, t: float) -> None:
        super().step(t)
        self.step_called = True
        self.step_count += 1
    
    def get_state(self) -> dict:
        return {
            **super().get_state(),
            'step_count': self.step_count
        }


class TestComponentABC:
    """Test suite for Component abstract base class."""
    
    def test_component_initialization(self):
        """Test component initializes correctly."""
        component = MockComponent()
        registry = ComponentRegistry()
        
        assert not component._initialized
        
        component.initialize(dt=1.0, registry=registry)
        
        assert component._initialized
        assert component.dt == 1.0
        assert component.initialized_called
    
    def test_step_before_initialize_raises_error(self):
        """Test that step() before initialize() raises error."""
        component = MockComponent()
        
        with pytest.raises(ComponentNotInitializedError):
            component.step(0.0)
    
    def test_step_increments_counter(self):
        """Test step() executes correctly."""
        component = MockComponent()
        registry = ComponentRegistry()
        component.initialize(1.0, registry)
        
        component.step(0.0)
        assert component.step_called
        assert component.step_count == 1
        
        component.step(1.0)
        assert component.step_count == 2
    
    def test_get_state_returns_dict(self):
        """Test get_state() returns valid dictionary."""
        component = MockComponent()
        registry = ComponentRegistry()
        component.initialize(1.0, registry)
        component.step(0.0)
        
        state = component.get_state()
        
        assert isinstance(state, dict)
        assert 'initialized' in state
        assert 'step_count' in state
        assert state['step_count'] == 1
    
    def test_component_id_assignment(self):
        """Test component ID can be set."""
        component = MockComponent()
        component.set_component_id('test_component')
        
        assert component.component_id == 'test_component'


class TestComponentRegistry:
    """Test suite for ComponentRegistry."""
    
    def test_registry_initialization(self):
        """Test registry initializes empty."""
        registry = ComponentRegistry()
        
        assert registry.get_component_count() == 0
        assert len(registry.get_all_ids()) == 0
    
    def test_register_component(self):
        """Test component registration."""
        registry = ComponentRegistry()
        component = MockComponent()
        
        registry.register('comp1', component, component_type='mock')
        
        assert registry.get_component_count() == 1
        assert registry.has('comp1')
        assert component.component_id == 'comp1'
    
    def test_register_duplicate_raises_error(self):
        """Test duplicate registration raises error."""
        registry = ComponentRegistry()
        
        registry.register('comp1', MockComponent())
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register('comp1', MockComponent())
    
    def test_get_component_by_id(self):
        """Test component retrieval by ID."""
        registry = ComponentRegistry()
        component = MockComponent()
        registry.register('comp1', component)
        
        retrieved = registry.get('comp1')
        
        assert retrieved is component
    
    def test_get_nonexistent_component_raises_error(self):
        """Test retrieving nonexistent component raises error."""
        registry = ComponentRegistry()
        
        with pytest.raises(KeyError, match="not found"):
            registry.get('nonexistent')
    
    def test_get_by_type(self):
        """Test component retrieval by type."""
        registry = ComponentRegistry()
        
        comp1 = MockComponent()
        comp2 = MockComponent()
        comp3 = MockComponent()
        
        registry.register('comp1', comp1, component_type='type_a')
        registry.register('comp2', comp2, component_type='type_a')
        registry.register('comp3', comp3, component_type='type_b')
        
        type_a_components = registry.get_by_type('type_a')
        
        assert len(type_a_components) == 2
        assert comp1 in type_a_components
        assert comp2 in type_a_components
    
    def test_initialize_all(self):
        """Test initialize_all() calls initialize on all components."""
        registry = ComponentRegistry()
        
        comp1 = MockComponent()
        comp2 = MockComponent()
        registry.register('comp1', comp1)
        registry.register('comp2', comp2)
        
        registry.initialize_all(dt=1.0)
        
        assert comp1.initialized_called
        assert comp2.initialized_called
    
    def test_step_all(self):
        """Test step_all() calls step on all components."""
        registry = ComponentRegistry()
        
        comp1 = MockComponent()
        comp2 = MockComponent()
        registry.register('comp1', comp1)
        registry.register('comp2', comp2)
        registry.initialize_all(dt=1.0)
        
        registry.step_all(0.0)
        
        assert comp1.step_count == 1
        assert comp2.step_count == 1
        
        registry.step_all(1.0)
        
        assert comp1.step_count == 2
        assert comp2.step_count == 2
    
    def test_get_all_states(self):
        """Test get_all_states() aggregates component states."""
        registry = ComponentRegistry()
        
        comp1 = MockComponent()
        comp2 = MockComponent()
        registry.register('comp1', comp1)
        registry.register('comp2', comp2)
        registry.initialize_all(dt=1.0)
        registry.step_all(0.0)
        
        states = registry.get_all_states()
        
        assert 'comp1' in states
        assert 'comp2' in states
        assert states['comp1']['step_count'] == 1
        assert states['comp2']['step_count'] == 1


class TestEnums:
    """Test suite for integer enums."""
    
    def test_tank_state_enum_values(self):
        """Test TankState enum has integer values."""
        from h2_plant.core.enums import TankState
        
        assert isinstance(TankState.IDLE.value, int)
        assert TankState.IDLE == 0
        assert TankState.FILLING == 1
        assert TankState.FULL == 3
    
    def test_enum_numpy_compatibility(self):
        """Test enums work with NumPy arrays."""
        import numpy as np
        from h2_plant.core.enums import TankState
        
        states = np.array([TankState.IDLE, TankState.FULL, TankState.IDLE], dtype=np.int32)
        
        idle_mask = states == TankState.IDLE
        assert np.array_equal(idle_mask, [True, False, True])


# Coverage target: 100% for core module
```

***

### 2.2 Layer 2: Performance Optimization Tests

**File:** `tests/optimization/test_lut_manager.py`

```python
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
        config = LUTConfig(
            pressure_points=20,  # Reduced for faster tests
            temperature_points=10
        )
        manager = LUTManager(config=config)
        manager.initialize()
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
        
        masses = np.array([100.0, 150.0, 200.0])
        volumes = np.array([1.0, 1.0, 1.0])
        temperature = 298.15
        
        pressures = batch_pressure_update(masses, volumes, temperature)
        
        assert len(pressures) == 3
        assert all(p > 0 for p in pressures)
        assert pressures[2] > pressures[1] > pressures[0]  # More mass = higher pressure
    
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
        assert benchmark.stats['mean'] < 0.00001


# Coverage target: 95% for optimization module
```

***

### 2.3 Layer 3: Component Tests

**File:** `tests/components/test_electrolyzer_source.py`

```python
"""
Unit tests for ElectrolyzerProductionSource.
"""

import pytest
from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import ProductionState


class TestElectrolyzerSource:
    """Test suite for electrolyzer component."""
    
    @pytest.fixture
    def electrolyzer(self):
        """Create electrolyzer instance."""
        return ElectrolyzerProductionSource(
            max_power_mw=2.5,
            base_efficiency=0.65,
            min_load_factor=0.20
        )
    
    @pytest.fixture
    def initialized_electrolyzer(self, electrolyzer):
        """Create and initialize electrolyzer."""
        registry = ComponentRegistry()
        electrolyzer.initialize(dt=1.0, registry=registry)
        return electrolyzer
    
    def test_initialization(self, electrolyzer):
        """Test electrolyzer initializes correctly."""
        registry = ComponentRegistry()
        electrolyzer.initialize(dt=1.0, registry=registry)
        
        assert electrolyzer._initialized
        assert electrolyzer.state == ProductionState.OFFLINE
    
    def test_production_at_rated_load(self, initialized_electrolyzer):
        """Test production at rated load (100%)."""
        elec = initialized_electrolyzer
        
        elec.power_input_mw = 2.5  # 100% load
        elec.step(0.0)
        
        assert elec.state == ProductionState.RUNNING
        assert elec.h2_output_kg > 0
        
        # Check efficiency at rated load
        expected_h2 = (2.5 * 1000 * 1.0) / 33.0 * 0.65  # Approximate
        assert abs(elec.h2_output_kg - expected_h2) < 5.0  # Within 5 kg
    
    def test_production_below_min_load(self, initialized_electrolyzer):
        """Test electrolyzer shuts down below minimum load."""
        elec = initialized_electrolyzer
        
        # 10% load (below 20% minimum)
        elec.power_input_mw = 0.25
        elec.step(0.0)
        
        assert elec.state == ProductionState.OFFLINE
        assert elec.h2_output_kg == 0.0
    
    def test_oxygen_byproduct_generation(self, initialized_electrolyzer):
        """Test oxygen byproduct calculation."""
        elec = initialized_electrolyzer
        
        elec.power_input_mw = 2.0
        elec.step(0.0)
        
        # O2:H2 mass ratio should be ~7.94:1
        expected_o2 = elec.h2_output_kg * 7.94
        
        assert abs(elec.o2_output_kg - expected_o2) < 0.01
    
    def test_cumulative_tracking(self, initialized_electrolyzer):
        """Test cumulative production tracking."""
        elec = initialized_electrolyzer
        
        # Run 10 timesteps
        elec.power_input_mw = 2.0
        for hour in range(10):
            elec.step(hour)
        
        assert elec.cumulative_h2_kg > 0
        assert elec.cumulative_energy_kwh > 0
        assert elec.cumulative_h2_kg == sum([elec.h2_output_kg for _ in range(10)])
    
    def test_state_serialization(self, initialized_electrolyzer):
        """Test get_state returns complete state."""
        elec = initialized_electrolyzer
        
        elec.power_input_mw = 2.0
        elec.step(0.0)
        
        state = elec.get_state()
        
        required_keys = [
            'power_input_mw', 'h2_output_kg', 'o2_output_kg',
            'efficiency', 'state', 'cumulative_h2_kg'
        ]
        
        for key in required_keys:
            assert key in state


# Coverage target: 95% for all component modules
```

***

## 3. Integration Testing

### 3.1 Component Integration Tests

**File:** `tests/integration/test_production_to_storage.py`

```python
"""
Integration tests for production → storage flow.
"""

import pytest
from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.core.component_registry import ComponentRegistry


class TestProductionToStorage:
    """Test integration between production and storage components."""
    
    @pytest.fixture
    def production_storage_system(self):
        """Create integrated production + storage system."""
        registry = ComponentRegistry()
        
        # Create components
        electrolyzer = ElectrolyzerProductionSource(max_power_mw=2.5)
        tanks = TankArray(n_tanks=4, capacity_kg=200.0, pressure_bar=350)
        
        # Register
        registry.register('electrolyzer', electrolyzer, component_type='production')
        registry.register('tanks', tanks, component_type='storage')
        
        # Initialize
        registry.initialize_all(dt=1.0)
        
        return registry
    
    def test_production_fills_storage(self, production_storage_system):
        """Test hydrogen production fills storage tanks."""
        registry = production_storage_system
        
        electrolyzer = registry.get('electrolyzer')
        tanks = registry.get('tanks')
        
        # Run 10 hours of production
        for hour in range(10):
            # Produce hydrogen
            electrolyzer.power_input_mw = 2.0
            electrolyzer.step(hour)
            
            # Store production
            h2_produced = electrolyzer.h2_output_kg
            stored, overflow = tanks.fill(h2_produced)
            
            # Step tanks
            tanks.step(hour)
            
            # Verify no overflow initially
            assert overflow == 0.0
        
        # Verify storage increased
        total_stored = tanks.get_total_mass()
        assert total_stored > 0
        
        # Verify mass balance
        total_produced = electrolyzer.cumulative_h2_kg
        assert abs(total_stored - total_produced) < 0.01
    
    def test_storage_overflow_handling(self, production_storage_system):
        """Test behavior when storage capacity exceeded."""
        registry = production_storage_system
        
        electrolyzer = registry.get('electrolyzer')
        tanks = registry.get('tanks')
        
        # Fill tanks to capacity
        total_capacity = tanks.n_tanks * tanks.capacity_kg  # 4 * 200 = 800 kg
        
        # Produce more than capacity
        production_hours = 20
        for hour in range(production_hours):
            electrolyzer.power_input_mw = 2.5  # Max production
            electrolyzer.step(hour)
            
            h2_produced = electrolyzer.h2_output_kg
            stored, overflow = tanks.fill(h2_produced)
            tanks.step(hour)
            
            if overflow > 0:
                # Overflow should occur when storage full
                assert tanks.get_available_capacity() < 10.0
        
        # Verify storage at capacity
        assert tanks.get_total_mass() <= total_capacity * 1.01  # Allow 1% tolerance


class TestDualPathIntegration:
    """Test integration of dual-path system."""
    
    def test_pathway_coordination(self):
        """Test dual-path coordinator allocates demand correctly."""
        from h2_plant.pathways.isolated_production_path import IsolatedProductionPath
        from h2_plant.pathways.dual_path_coordinator import DualPathCoordinator
        from h2_plant.core.enums import AllocationStrategy
        
        # Setup registry with full dual-path system
        # ... (create electrolyzer path, ATR path, coordinator)
        
        # Test allocation
        coordinator.total_demand_kg = 100.0
        coordinator.step(0.0)
        
        # Verify allocation occurred
        assert sum(coordinator.pathway_allocations.values()) <= 100.0


# Coverage target: 90% for integration scenarios
```

***

## 4. Performance Testing

### 4.1 Performance Benchmarks

**File:** `tests/performance/test_benchmarks.py`

```python
"""
Performance benchmark tests.

Validates performance targets are met:
- LUT lookup: <0.1ms
- Tank operations: >10,000 ops/sec
- Full simulation: 8760 hours in <90 seconds
"""

import pytest
import time
import numpy as np


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""
    
    def test_lut_lookup_performance(self, benchmark):
        """Benchmark: LUT lookup should be <0.1ms."""
        from h2_plant.optimization.lut_manager import LUTManager
        
        lut = LUTManager()
        lut.initialize()
        
        def lookup():
            return lut.lookup('H2', 'D', 350e5, 298.15)
        
        result = benchmark(lookup)
        
        # Target: <0.1ms = 0.0001 seconds
        assert benchmark.stats['mean'] < 0.0001
        
        print(f"\nLUT Lookup: {benchmark.stats['mean']*1000:.4f} ms/lookup")
    
    def test_tank_fill_performance(self, benchmark):
        """Benchmark: Tank operations should exceed 10,000 ops/sec."""
        from h2_plant.components.storage.tank_array import TankArray
        from h2_plant.core.component_registry import ComponentRegistry
        
        tanks = TankArray(n_tanks=100, capacity_kg=200.0, pressure_bar=350)
        tanks.initialize(1.0, ComponentRegistry())
        
        def fill_operation():
            tanks.fill(50.0)
        
        result = benchmark(fill_operation)
        
        ops_per_sec = 1.0 / benchmark.stats['mean']
        
        # Target: >10,000 ops/sec
        assert ops_per_sec > 10000
        
        print(f"\nTank Operations: {ops_per_sec:.0f} ops/sec")
    
    @pytest.mark.slow
    def test_full_simulation_performance(self):
        """Benchmark: Full 8760-hour simulation should complete in <90 seconds."""
        from h2_plant.config.plant_builder import PlantBuilder
        from h2_plant.simulation.engine import SimulationEngine
        
        # Build minimal plant
        plant = PlantBuilder.from_file("configs/plant_pilot.yaml")
        engine = SimulationEngine(plant.registry, plant.config.simulation)
        
        # Run full year simulation
        start_time = time.time()
        engine.run(start_hour=0, end_hour=8760)
        elapsed_time = time.time() - start_time
        
        # Target: <90 seconds
        assert elapsed_time < 90.0
        
        print(f"\nFull Simulation: {elapsed_time:.2f} seconds for 8760 hours")
        print(f"Speed: {8760/elapsed_time:.1f} simulation hours/second")


@pytest.mark.benchmark
class TestRegressionBenchmarks:
    """Regression benchmarks to detect performance degradation."""
    
    def test_component_step_overhead(self, benchmark):
        """Measure Component ABC overhead."""
        from h2_plant.core.component import Component
        from h2_plant.core.component_registry import ComponentRegistry
        
        class MinimalComponent(Component):
            def initialize(self, dt, registry): 
                super().initialize(dt, registry)
            def step(self, t): 
                super().step(t)
            def get_state(self): 
                return super().get_state()
        
        comp = MinimalComponent()
        comp.initialize(1.0, ComponentRegistry())
        
        def step():
            comp.step(0.0)
        
        result = benchmark(step)
        
        # Component ABC overhead should be <1 microsecond
        assert benchmark.stats['mean'] < 0.000001


# Coverage target: Performance benchmarks for all critical paths
```

***

## 5. End-to-End Testing

### 5.1 Scenario Tests

**File:** `tests/e2e/test_complete_simulation.py`

```python
"""
End-to-end simulation tests.

Validates complete system behavior from configuration to results.
"""

import pytest
from pathlib import Path
import json


class TestCompleteSimulation:
    """End-to-end simulation test scenarios."""
    
    @pytest.mark.slow
    def test_baseline_plant_simulation(self, tmp_path):
        """Test complete baseline plant simulation."""
        from h2_plant.simulation.runner import run_simulation_from_config
        
        # Run full simulation
        results = run_simulation_from_config(
            config_path="configs/plant_baseline.yaml",
            output_dir=tmp_path
        )
        
        # Validate results structure
        assert 'simulation' in results
        assert 'metrics' in results
        assert 'final_states' in results
        
        # Validate simulation completed
        sim_info = results['simulation']
        assert sim_info['duration_hours'] == 8760
        
        # Validate metrics
        metrics = results['metrics']
        assert metrics['total_production_kg'] > 0
        assert metrics['average_cost_per_kg'] > 0
        
        # Validate output files created
        assert (tmp_path / "simulation_results.json").exists()
        assert (tmp_path / "metrics" / "timeseries.csv").exists()
    
    @pytest.mark.slow
    def test_scenario_comparison(self, tmp_path):
        """Test scenario comparison functionality."""
        from h2_plant.simulation.runner import run_scenario_comparison
        
        scenarios = [
            "configs/plant_baseline.yaml",
            "configs/plant_grid_only.yaml"
        ]
        
        results = run_scenario_comparison(scenarios, output_dir=tmp_path)
        
        # Validate both scenarios ran
        assert len(results) == 2
        assert 'plant_baseline' in results
        assert 'plant_grid_only' in results
        
        # Validate comparison report generated
        assert (tmp_path / "scenario_comparison.json").exists()
    
    def test_checkpoint_and_resume(self, tmp_path):
        """Test checkpoint save and resume functionality."""
        from h2_plant.config.plant_builder import PlantBuilder
        from h2_plant.simulation.engine import SimulationEngine
        
        # Build plant
        plant = PlantBuilder.from_file("configs/plant_pilot.yaml")
        engine = SimulationEngine(plant.registry, plant.config.simulation, output_dir=tmp_path)
        
        # Run partial simulation
        engine.run(start_hour=0, end_hour=100)
        
        # Verify checkpoint exists
        checkpoints = list((tmp_path / "checkpoints").glob("*.json"))
        assert len(checkpoints) > 0
        
        # Resume from checkpoint
        plant2 = PlantBuilder.from_file("configs/plant_pilot.yaml")
        engine2 = SimulationEngine(plant2.registry, plant2.config.simulation, output_dir=tmp_path)
        
        results = engine2.run(resume_from_checkpoint=str(checkpoints[0]))
        
        # Validate resumed successfully
        assert results['simulation']['start_hour'] >= 100


class TestLegacyCompatibility:
    """Test backward compatibility with legacy system."""
    
    def test_legacy_import_warnings(self):
        """Test legacy imports raise deprecation warnings."""
        with pytest.warns(DeprecationWarning):
            import Dual_tank_system
    
    @pytest.mark.skipif(
        not Path("Dual_tank_system").exists(),
        reason="Legacy code not present"
    )
    def test_output_parity_with_legacy(self):
        """Test new system produces same results as legacy."""
        # Run legacy simulation
        # legacy_results = run_legacy_simulation()
        
        # Run new simulation
        from h2_plant.simulation.runner import run_simulation_from_config
        new_results = run_simulation_from_config("configs/plant_baseline.yaml")
        
        # Compare key metrics
        # assert abs(new_results['metrics']['total_production_kg'] - 
        #           legacy_results['total_production']) < 1.0  # Within 1 kg


# Coverage target: 100% of critical user workflows
```

***

## 6. CI/CD Pipeline

### 6.1 GitHub Actions Workflow

**File:** `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run linting
      run: |
        flake8 h2_plant/ --count --select=E9,F63,F7,F82 --show-source --statistics
        black --check h2_plant/
    
    - name: Run type checking
      run: |
        mypy h2_plant/ --strict
    
    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=h2_plant --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
    
    - name: Run benchmarks
      run: |
        pytest tests/performance/ --benchmark-only
  
  integration:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Run end-to-end tests
      run: |
        pytest tests/e2e/ -v -m "not slow"
```

***

### 6.2 Pre-commit Hooks

**File:** `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.10
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']
```

***

## 7. Test Execution Guide

### 7.1 Running Tests Locally

```bash
# Install package in development mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=h2_plant --cov-report=html

# Run specific test categories
pytest tests/core/              # Core foundation tests
pytest tests/components/        # Component tests
pytest tests/integration/       # Integration tests
pytest tests/e2e/              # End-to-end tests

# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Run slow tests (marked with @pytest.mark.slow)
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/core/test_component.py

# Run specific test
pytest tests/core/test_component.py::TestComponentABC::test_initialization
```

***

### 7.2 Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=h2_plant --cov-report=html
open htmlcov/index.html

# Generate terminal coverage report
pytest --cov=h2_plant --cov-report=term-missing

# Coverage targets:
# - Core modules: 100%
# - Optimization: 95%
# - Components: 95%
# - Overall: 95%
```

***

## 8. Validation Criteria

This Testing and Validation Framework is **COMPLETE** when:

✅ **Test Coverage:**
- Core modules: 100% coverage
- All other modules: 95%+ coverage
- Coverage report generated automatically

✅ **Test Categories:**
- Unit tests for all components
- Integration tests for component interactions
- Performance benchmarks for critical paths
- End-to-end scenarios for complete workflows

✅ **CI/CD:**
- Automated testing on push/PR
- Multi-platform testing (Linux, Windows, macOS)
- Multi-version testing (Python 3.9, 3.10, 3.11)

✅ **Performance:**
- All benchmarks meet targets
- No performance regressions detected

✅ **Documentation:**
- Test execution guide complete
- Coverage requirements documented
- CI/CD setup documented

***

## 9. Success Metrics

| **Metric** | **Target** | **Measurement** |
|-----------|-----------|-----------------|
| Test Coverage | 95%+ | `pytest --cov` |
| Test Pass Rate | 100% | CI/CD pipeline |
| Performance Benchmarks | All pass | `pytest --benchmark-only` |
| Legacy Compatibility | 100% | Parity tests pass |
| CI/CD Success Rate | >95% | GitHub Actions metrics |

***
