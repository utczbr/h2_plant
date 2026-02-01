import pytest
from h2_plant.core.component import Component, ComponentNotInitializedError
from h2_plant.core.component_registry import ComponentRegistry

class MockComponent(Component):
    """Mock component for testing."""
    
    def __init__(self):
        super().__init__()
        self.step_count = 0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        super().step(t)
        self.step_count += 1
    
    def get_state(self) -> dict:
        return {**super().get_state(), "step_count": self.step_count}


def test_component_initialization():
    """Test component initialization flow."""
    component = MockComponent()
    registry = ComponentRegistry()
    
    # Not initialized yet
    assert not component._initialized
    
    # Initialize
    component.initialize(dt=1.0, registry=registry)
    assert component._initialized
    assert component.dt == 1.0

def test_step_before_initialize_raises_error():
    """Test that step() before initialize() raises error."""
    component = MockComponent()
    
    with pytest.raises(ComponentNotInitializedError):
        component.step(0.0)

def test_component_state_serialization():
    """Test get_state() returns valid dictionary."""
    component = MockComponent()
    registry = ComponentRegistry()
    component.initialize(1.0, registry)
    
    state = component.get_state()
    assert isinstance(state, dict)
    assert "initialized" in state
    assert state["initialized"] is True
