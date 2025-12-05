import pytest
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry, DuplicateComponentError

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


def test_component_registration():
    """Test basic component registration and lookup."""
    registry = ComponentRegistry()
    component = MockComponent()
    
    registry.register("test_comp", component, component_type="mock")
    
    # Lookup by ID
    retrieved = registry.get("test_comp")
    assert retrieved is component
    assert retrieved.component_id == "test_comp"
    
    # Lookup by type
    by_type = registry.get_by_type("mock")
    assert len(by_type) == 1
    assert by_type[0] is component

def test_duplicate_registration_raises_error():
    """Test that duplicate IDs raise DuplicateComponentError."""
    registry = ComponentRegistry()
    
    registry.register("comp1", MockComponent())
    
    with pytest.raises(DuplicateComponentError, match="already registered"):
        registry.register("comp1", MockComponent())

def test_initialize_all():
    """Test initialize_all() calls initialize on all components."""
    registry = ComponentRegistry()
    
    comp1 = MockComponent()
    comp2 = MockComponent()
    registry.register("comp1", comp1)
    registry.register("comp2", comp2)
    
    registry.initialize_all(dt=1.0)
    
    assert comp1._initialized
    assert comp2._initialized
    assert comp1.dt == 1.0
    assert comp2.dt == 1.0

def test_step_all():
    """Test step_all() calls step on all components."""
    registry = ComponentRegistry()
    
    comp1 = MockComponent()
    comp2 = MockComponent()
    registry.register("comp1", comp1)
    registry.register("comp2", comp2)
    registry.initialize_all(dt=1.0)
    
    registry.step_all(0.0)
    
    assert comp1.step_count == 1
    assert comp2.step_count == 1
