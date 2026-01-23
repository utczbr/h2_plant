
import pandas as pd
import numpy as np
import os
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.electrolysis.soec_operator import SOECOperator

# Mock setup to see what keys are generated
def check_history_keys():
    registry = ComponentRegistry()
    
    # Create SOEC component
    soec_config = {
        'num_modules': 1,
        'max_power_nominal_mw': 1.0, 
        'component_id': 'SOEC_Cluster'
    }
    # Mocking SOECOperator to just be a component
    soec = SOECOperator(soec_config)
    soec.component_id = 'SOEC_Cluster'
    registry.register(soec)
    
    strategy = HybridArbitrageEngineStrategy()
    
    # Initialize strategy (this triggers history allocation)
    # We need a dummy context
    class MockContext:
        class physics:
            class soec_cluster:
                num_modules = 1
                max_power_nominal_mw = 1.0
                optimal_limit = 0.8
            class pem_system:
                max_power_mw = 1.0
        class economics:
            pass
        class simulation:
            storage_control_mode = 'SCHMITT_TRIGGER'
            
    strategy.initialize(registry, MockContext(), total_steps=10)
    
    # Print all keys related to SOEC
    print("SOEC History Keys:")
    soec_keys = [k for k in strategy._history.keys() if 'SOEC' in k]
    for k in sorted(soec_keys):
        print(f" - {k}")

if __name__ == "__main__":
    check_history_keys()
