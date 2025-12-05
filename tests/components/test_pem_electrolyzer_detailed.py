import pytest
from h2_plant.components.production.pem_electrolyzer_detailed import DetailedPEMElectrolyzer
from h2_plant.core.component_registry import ComponentRegistry

def test_detailed_pem_instantiation():
    """Test that the detailed PEM electrolyzer and all its subsystems can be instantiated."""
    try:
        component = DetailedPEMElectrolyzer(max_power_kw=2500.0)
    except Exception as e:
        pytest.fail(f"Failed to instantiate DetailedPEMElectrolyzer: {e}")
    
    # Check that subsystems were created
    assert component.feedwater_inlet is not None
    assert component.pem_stacks is not None
    assert len(component.pem_stacks.stacks) == 10
    assert component.rectifier is not None

def test_detailed_pem_initialization_and_step():
    """Test that the detailed PEM electrolyzer can be initialized and can run a step."""
    registry = ComponentRegistry()
    component = DetailedPEMElectrolyzer(max_power_kw=2500.0)
    
    # Register the main component and its subsystems if they need to be found by others
    # For this test, we are testing it as a self-contained unit, so we only need to register itself
    # if other components were to depend on it.
    
    try:
        # The CompositeComponent's initialize will call initialize on all subsystems
        component.initialize(dt=1.0, registry=registry)
    except Exception as e:
        pytest.fail(f"DetailedPEMElectrolyzer failed to initialize: {e}")

    try:
        # Set some inputs to see if the system reacts
        # (Note: The internal orchestration is simplified, a real one would be more complex)
        component.rectifier.ac_input_kw = 2000.0
        for stack in component.pem_stacks.stacks:
            stack.current_a = 4000.0 # Set current for each stack

        component.step(t=0)
    except Exception as e:
        pytest.fail(f"DetailedPEMElectrolyzer failed during step: {e}")

    # Check for some output
    state = component.get_state()
    assert state['summary']['total_power_input_kw'] > 0
    assert state['summary']['h2_product_kg_h'] > 0
