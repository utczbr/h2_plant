"""
Test SimulationWorker in isolation.
"""
import sys
import time
from pathlib import Path
from PySide6.QtCore import QCoreApplication

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from h2_plant.gui.core.worker import SimulationWorker

def test_worker():
    print("Testing SimulationWorker...")
    
    # Minimal config for a fast simulation
    config_dict = {
        "name": "Test Plant",
        "version": "1.0",
        "production": {
            "electrolyzer": {"enabled": True, "max_power_mw": 5.0, "base_efficiency": 0.7}
        },
        "storage": {
            "lp_tanks": {"count": 2, "capacity_kg": 50.0, "pressure_bar": 30.0},
            "hp_tanks": {"count": 2, "capacity_kg": 100.0, "pressure_bar": 350.0},
            "source_isolated": False
        },
        "compression": {},
        "demand": {"pattern": "constant", "base_demand_kg_h": 10.0},
        "energy_price": {"source": "constant", "constant_price_per_mwh": 50.0},
        "pathway": {"allocation_strategy": "COST_OPTIMAL"},
        "simulation": {
            "timestep_hours": 1.0,
            "duration_hours": 24, # Short duration
            "checkpoint_interval_hours": 24
        }
    }
    
    app = QCoreApplication(sys.argv)
    
    worker = SimulationWorker(config_dict)
    
    # Track signals
    signals_received = {"progress": False, "finished": False, "error": False}
    
    def on_progress(val):
        print(f"Progress: {val}%")
        signals_received["progress"] = True
        
    def on_finished(results):
        print("Finished!")
        print("Results keys:", results.keys())
        signals_received["finished"] = True
        app.quit()
        
    def on_error(msg):
        print(f"Error: {msg}")
        signals_received["error"] = True
        app.quit()
        
    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    
    print("Starting worker...")
    worker.start()
    
    # Run event loop
    app.exec()
    
    # Verify
    if signals_received["error"]:
        print("TEST FAILED: Worker emitted error")
        sys.exit(1)
        
    if not signals_received["finished"]:
        print("TEST FAILED: Worker did not finish")
        sys.exit(1)
        
    print("TEST PASSED: Worker completed successfully")

if __name__ == "__main__":
    test_worker()
