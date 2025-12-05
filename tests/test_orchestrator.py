import pytest
import os
from h2_plant.orchestrator import Orchestrator

def test_orchestrator_init():
    # Path to scenarios
    scenarios_dir = os.path.abspath("scenarios")
    
    # Initialize Orchestrator
    orchestrator = Orchestrator(scenarios_dir)
    
    # Check Configs Loaded
    assert "economics_parameters" in orchestrator.configs
    assert "h2plant_detailed" in orchestrator.configs
    assert "physics_parameters" in orchestrator.configs
    
    # Check Specific Config Value
    econ = orchestrator.get_config("economics_parameters")
    assert econ["economics"]["h2_price_eur_kg"] == 9.6
    
    # Initialize Components
    orchestrator.initialize_components()
    
    # Check SOEC Component Initialized
    assert "soec" in orchestrator.components
    soec = orchestrator.components["soec"]
    
    # Verify SOEC State
    status = soec.get_status()
    assert status["active_modules"] == 0 # Should start with 0 active or standby depending on logic
    # Actually logic sets them to Hot Standby (State 1), but power is 0.0 (standby power)
    # active_modules counts power > 0.01. Standby power is 0.0. So 0 active is correct.
    
    print("Orchestrator Test Passed!")

if __name__ == "__main__":
    test_orchestrator_init()
