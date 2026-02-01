import pytest
from pathlib import Path
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.config.plant_config import SimulationConfig
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component import Component

# Mock component for engine tests
class StepCounter(Component):
    def __init__(self):
        super().__init__()
        self.step_count = 0
    def initialize(self, dt, registry):
        super().initialize(dt, registry)
    def step(self, t):
        super().step(t)
        self.step_count += 1
    def get_state(self):
        return {**super().get_state(), 'step_count': self.step_count}

@pytest.fixture
def basic_engine(tmp_path):
    """Fixture for a basic SimulationEngine with a simple component."""
    registry = ComponentRegistry()
    registry.register("counter", StepCounter())
    
    config = SimulationConfig(duration_hours=10)
    
    engine = SimulationEngine(registry, config, output_dir=tmp_path)
    return engine

def test_engine_initialization(basic_engine):
    """Test the simulation engine initializes correctly."""
    assert not basic_engine.is_initialized
    engine = basic_engine
    engine.initialize()
    
    assert engine.is_initialized
    counter = engine.registry.get("counter")
    assert counter._initialized

def test_engine_run_loop(basic_engine):
    """Test that the run loop executes for the correct number of steps."""
    engine = basic_engine
    results = engine.run()
    
    # Check that the simulation ran for the configured duration
    assert engine.current_hour == 9 # loop is range(0, 10)
    assert results['simulation']['duration_hours'] == 10
    
    # Check that the component was stepped
    final_state = results['final_states']['counter']
    assert final_state['step_count'] == 10

def test_engine_checkpointing(tmp_path):
    """Test that the engine creates checkpoint files."""
    registry = ComponentRegistry()
    registry.register("counter", StepCounter())
    config = SimulationConfig(duration_hours=10, checkpoint_interval_hours=5)
    engine = SimulationEngine(registry, config, output_dir=tmp_path)
    
    engine.run()
    
    # A checkpoint should have been created at hour 5
    checkpoint_file = tmp_path / "checkpoints" / "checkpoint_hour_5.json"
    assert checkpoint_file.exists()

def test_engine_event_processing(basic_engine, mocker):
    """Test that the engine processes scheduled events."""
    engine = basic_engine
    handler = mocker.Mock()

    from h2_plant.simulation.event_scheduler import Event
    event = Event(hour=3, event_type="test", handler=handler)
    engine.schedule_event(event)

    engine.run()

    # The handler should have been called once
    handler.assert_called_once()
