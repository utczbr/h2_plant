"""
Background worker for running simulations.
"""
import traceback
import logging
from PySide6.QtCore import QThread, Signal

# from h2_plant.config.plant_builder import PlantBuilder

logger = logging.getLogger(__name__)

class SimulationWorker(QThread):
    """
    Worker thread for running the simulation without freezing the UI.
    """
    progress = Signal(int)
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, context):
        super().__init__()
        self.context = context
        self._is_running = True
        
    def run(self):
        try:
            # 1. Convert config dict to SimulationContext
            # Note: We assume self.config_dict is actually the adapter instance or we need to refactor how worker is called.
            # Ideally, the worker should receive the Context object directly.
            # But for minimal changes, let's assume we pass the adapter or context.
            
            # Wait, the previous code passed `config` which was a dict from adapter.to_config_dict().
            # Now we want to use adapter.to_simulation_context().
            # So we should update the caller (MainWindow) to pass the context, OR pass the adapter.
            # Let's assume the worker now receives the context object directly in __init__.
            
            if not hasattr(self, 'context'):
                raise ValueError("SimulationContext not provided to worker")

            # 2. Initialize Orchestrator with Context
            from h2_plant.orchestrator import Orchestrator
            
            print(f"DEBUG WORKER: Context type: {type(self.context)}")
            if hasattr(self.context, 'topology'):
                print(f"DEBUG WORKER: Topology type: {type(self.context.topology)}")
                if hasattr(self.context.topology, 'nodes'):
                    print(f"DEBUG WORKER: Topology nodes type: {type(self.context.topology.nodes)}")
                else:
                    print(f"DEBUG WORKER: Topology has no 'nodes' attribute: {self.context.topology}")
            
            # We need a dummy path for scenarios_dir if we are using context injection
            orchestrator = Orchestrator(scenarios_dir=".", context=self.context)
            
            # 3. Run Simulation
            # The orchestrator handles initialization and stepping
            history = orchestrator.run_simulation()
            
            # 4. Emit Results
            self.finished.emit(history)
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            logger.error(traceback.format_exc())
            self.error.emit(str(e))
            
    def stop(self):
        self._is_running = False

