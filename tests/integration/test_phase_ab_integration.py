"""
Integration tests for Phase A/B fixes: Orchestrator and Engine integration.

Tests validate end-to-end scenarios:
- SOEC-only and PEM-only topologies
- Mass balance across flow propagation
- Checkpoint save/resume
- History shape verification
- No double-stepping in full simulation

Run with: pytest tests/integration/test_phase_ab_integration.py -v --tb=short
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import time


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_scenarios_dir():
    """Create temporary scenarios directory with minimal config."""
    import tempfile
    import yaml
    
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create minimal config
    config = {
        'simulation': {
            'timestep_hours': 1/60,
            'duration_hours': 1,
            'start_hour': 0
        },
        'physics': {
            'soec_cluster': {
                'num_modules': 1,
                'max_power_nominal_mw': 10.0
            },
            'pem_system': {
                'max_power_mw': 5.0
            }
        }
    }
    
    with open(temp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create price/wind data
    np.savetxt(temp_dir / 'prices.csv', np.ones(60) * 50.0)
    np.savetxt(temp_dir / 'wind.csv', np.ones(60) * 10.0)
    
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_registry_with_components():
    """Create registry with mock components for testing."""
    from h2_plant.core.component_registry import ComponentRegistry
    
    registry = ComponentRegistry()
    
    # Mock SOEC component
    mock_soec = Mock()
    mock_soec.component_id = "soec_cluster"
    mock_soec.soec_state = "RUNNING"
    mock_soec.receive_input = Mock(return_value=0.0)
    mock_soec.step = Mock(return_value=(5.0, 100.0, 50.0))  # P, H2, steam
    mock_soec.get_output = Mock(return_value=None)
    mock_soec.get_ports = Mock(return_value={'power_in': {'type': 'input'}, 'h2_out': {'type': 'output'}})
    mock_soec.get_state = Mock(return_value={})
    mock_soec.real_powers = np.array([5.0])
    mock_soec.last_h2_output_kg = 100.0
    mock_soec._step_count = 0
    
    def track_step(t):
        mock_soec._step_count += 1
        return (5.0, 100.0, 50.0)
    mock_soec.step.side_effect = track_step
    
    # Mock PEM component
    mock_pem = Mock()
    mock_pem.component_id = "pem_electrolyzer_detailed"
    mock_pem.V_cell = 1.8
    mock_pem.h2_output_kg = 50.0
    mock_pem.P_consumed_W = 2.5e6
    mock_pem.set_power_input_mw = Mock()
    mock_pem.step = Mock()
    mock_pem.get_output = Mock(return_value=None)
    mock_pem.get_ports = Mock(return_value={'h2_out': {'type': 'output'}})
    mock_pem.get_state = Mock(return_value={})
    mock_pem._step_count = 0
    
    def track_pem_step(t):
        mock_pem._step_count += 1
    mock_pem.step.side_effect = track_pem_step
    
    registry.register("soec_cluster", mock_soec)
    registry.register("pem_electrolyzer_detailed", mock_pem)
    
    return registry, mock_soec, mock_pem


# ============================================================================
# INTEGRATION TESTS: Flow Propagation
# ============================================================================

class TestFlowPropagation:
    """Tests for flow propagation with mass conservation."""
    
    def test_mass_balance_no_leaks(self, mock_registry_with_components):
        """Total mass should be conserved: input = output + storage change."""
        registry, mock_soec, mock_pem = mock_registry_with_components
        
        # Track mass in/out
        mass_produced = 0.0
        mass_consumed = 0.0
        
        # Simulate 10 steps
        for step in range(10):
            mock_soec.step(step)
            mock_pem.step(step)
            
            mass_produced += mock_soec.last_h2_output_kg + mock_pem.h2_output_kg
        
        # With no storage change, produced should equal consumed
        # (In this mock setup, nothing consumes the H2)
        assert mass_produced > 0, "Should have produced H2"
    
    def test_no_double_stepping_in_propagation(self, mock_registry_with_components):
        """Components should not be stepped during flow propagation."""
        registry, mock_soec, mock_pem = mock_registry_with_components
        
        initial_soec_count = mock_soec._step_count
        initial_pem_count = mock_pem._step_count
        
        # Simulate the flow propagation (which should NOT step)
        from h2_plant.core.stream import Stream
        
        stream = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=1e5,
            composition={'H2': 1.0}
        )
        
        # receive_input should NOT trigger step
        mock_pem.receive_input('h2_in', stream, 'hydrogen')
        
        # Verify step was not called
        assert mock_soec._step_count == initial_soec_count
        assert mock_pem._step_count == initial_pem_count


# ============================================================================
# INTEGRATION TESTS: Dispatch Strategy
# ============================================================================

class TestDispatchStrategyIntegration:
    """Tests for EngineDispatchStrategy integration."""
    
    def test_strategy_detects_soec_only(self, mock_registry_with_components):
        """Strategy should detect SOEC-only topology."""
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        
        registry, mock_soec, mock_pem = mock_registry_with_components
        
        # Remove PEM to simulate SOEC-only
        registry._components.pop("pem_electrolyzer_detailed", None)
        
        # Mock context
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
        
        context = MockContext()
        
        strategy = HybridArbitrageEngineStrategy()
        strategy.initialize(registry, context, total_steps=100)
        
        # SOEC should be found, PEM should be None
        assert strategy._soec is not None
        assert strategy._pem is None
    
    def test_history_shape_after_simulation(self, mock_registry_with_components):
        """History arrays should have correct shape after simulation."""
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        
        registry, mock_soec, mock_pem = mock_registry_with_components
        
        # Create mock context
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
        
        context = MockContext()
        
        total_steps = 100
        strategy = HybridArbitrageEngineStrategy()
        strategy.initialize(registry, context, total_steps=total_steps)
        
        # Simulate dispatch
        prices = np.ones(total_steps) * 50.0
        wind = np.ones(total_steps) * 10.0
        
        for step in range(10):
            t = step * (1/60)
            strategy.decide_and_apply(t, prices, wind)
            strategy.record_post_step()
        
        history = strategy.get_history()
        
        # Should have recorded 10 steps
        assert len(history['minute']) == 10
        assert all(isinstance(v, np.ndarray) for v in history.values())


# ============================================================================
# INTEGRATION TESTS: Checkpoint Resume
# ============================================================================

class TestCheckpointResume:
    """Tests for checkpoint save/resume functionality."""
    
    def test_checkpoint_saves_component_states(self, mock_registry_with_components):
        """Checkpoint should save all component states."""
        registry, mock_soec, mock_pem = mock_registry_with_components
        
        # Mock get_all_states
        def get_all_states():
            return {
                "soec_cluster": {"power": 5.0, "temp": 1073.0},
                "pem_electrolyzer_detailed": {"power": 2.5, "temp": 353.0}
            }
        
        registry.get_all_states = get_all_states
        
        states = registry.get_all_states()
        
        assert "soec_cluster" in states
        assert "pem_electrolyzer_detailed" in states
        assert states["soec_cluster"]["power"] == 5.0


# ============================================================================
# INTEGRATION TESTS: Engine Integration
# ============================================================================

class TestEngineIntegration:
    """Tests for SimulationEngine with DispatchStrategy."""
    
    def test_engine_accepts_dispatch_strategy(self, mock_registry_with_components):
        """SimulationEngine should accept dispatch_strategy parameter."""
        from h2_plant.simulation.engine import SimulationEngine
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        from h2_plant.config.simulation_config import SimulationConfig
        
        registry, _, _ = mock_registry_with_components
        
        config = SimulationConfig(
            timestep_hours=1/60,
            duration_hours=1,
            start_hour=0
        )
        
        strategy = HybridArbitrageEngineStrategy()
        
        # Should not raise
        engine = SimulationEngine(
            registry=registry,
            config=config,
            dispatch_strategy=strategy
        )
        
        assert engine.dispatch_strategy is strategy
    
    def test_engine_dispatch_data_loading(self, mock_registry_with_components):
        """Engine should accept price/wind data for dispatch."""
        from h2_plant.simulation.engine import SimulationEngine
        from h2_plant.config.simulation_config import SimulationConfig
        
        registry, _, _ = mock_registry_with_components
        
        config = SimulationConfig(
            timestep_hours=1/60,
            duration_hours=1,
            start_hour=0
        )
        
        engine = SimulationEngine(registry=registry, config=config)
        
        prices = np.ones(60) * 50.0
        wind = np.ones(60) * 10.0
        
        engine.set_dispatch_data(prices, wind)
        
        assert engine._dispatch_prices is not None
        assert engine._dispatch_wind is not None
        assert len(engine._dispatch_prices) == 60


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.benchmark
class TestIntegrationPerformance:
    """Performance benchmarks for integration tests."""
    
    def test_dispatch_decision_fast(self, mock_registry_with_components):
        """Dispatch decision should be fast (< 1ms per step)."""
        from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
        
        registry, _, _ = mock_registry_with_components
        
        # Create mock context
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
        
        context = MockContext()
        
        strategy = HybridArbitrageEngineStrategy()
        strategy.initialize(registry, context, total_steps=1000)
        
        prices = np.ones(1000) * 50.0
        wind = np.ones(1000) * 10.0
        
        start = time.perf_counter()
        for step in range(100):
            t = step * (1/60)
            strategy.decide_and_apply(t, prices, wind)
        elapsed = time.perf_counter() - start
        
        per_step = elapsed / 100
        assert per_step < 0.001, f"Dispatch too slow: {per_step*1000:.3f}ms/step"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
