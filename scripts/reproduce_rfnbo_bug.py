
import sys
import os
from pathlib import Path
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from h2_plant.simulation.monitoring import MonitoringSystem
from h2_plant.core.component_registry import ComponentRegistry

# Mock classes to simulate Engine and DispatchStrategy behavior
class MockDispatchStrategy:
    def get_history(self):
        # Simulate history with non-zero non-RFNBO hydrogen
        steps = 10
        return {
            'h2_rfnbo_kg': np.array([1.0] * steps),
            'h2_non_rfnbo_kg': np.array([0.5] * steps)  # Total 5.0 kg
        }

class MockEngine:
    def __init__(self):
        self.dispatch_strategy = MockDispatchStrategy()

def reproduce():
    output_dir = Path("reproduce_output")
    output_dir.mkdir(exist_ok=True)
    
    monitoring = MonitoringSystem(output_dir=output_dir)
    mock_engine = MockEngine()
    
    print("--- Reproducing RFNBO Reporting Bug ---")
    
    # 1. Call export_dashboard_data which uses get_summary()
    dashboard_path = monitoring.export_dashboard_data(engine=mock_engine, filename="reproduce_dashboard.json")
    
    # 2. Check results
    with open(dashboard_path, 'r') as f:
        data = json.load(f)
    
    kpis = data.get('kpis', {})
    h2_non_rfnbo = kpis.get('h2_non_rfnbo_kg', 0.0)
    
    print(f"Reported h2_non_rfnbo_kg: {h2_non_rfnbo}")
    
    if h2_non_rfnbo == 0.0:
        print("FAIL: Reported 0.0 kg but mock data had 5.0 kg.")
        print("Bug Reproduced: MonitoringSystem ignored dispatch history sums.")
    else:
        print(f"SUCCESS: Correctly reported {h2_non_rfnbo} kg.")

    # Cleanup
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    reproduce()
