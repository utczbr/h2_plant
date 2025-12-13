"""
Background worker for running simulations using unified SimulationEngine.

This module implements the Phase B1 architecture:
- SimulationEngine handles component lifecycle
- HybridArbitrageEngineStrategy handles dispatch decisions
- NumPy pre-allocated history arrays for HPC performance
"""
import traceback
import logging
import numpy as np
from pathlib import Path
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class SimulationWorker(QThread):
    """
    Worker thread for running the simulation without freezing the UI.
    
    Uses unified SimulationEngine architecture (Phase B1):
    - SimulationEngine: Component lifecycle, stepping, flow propagation
    - HybridArbitrageEngineStrategy: Dispatch decisions, pre-allocated history
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
            if not hasattr(self, 'context') or self.context is None:
                raise ValueError("SimulationContext not provided to worker")

            logger.info("Starting unified SimulationEngine...")
            
            # === 1. Build Component Graph ===
            from h2_plant.core.component_registry import ComponentRegistry
            from h2_plant.core.graph_builder import PlantGraphBuilder
            
            builder = PlantGraphBuilder(self.context)
            components = builder.build()
            
            # Create registry and register components
            registry = ComponentRegistry()
            for comp_id, comp in components.items():
                registry.register(comp_id, comp)
            
            # === 2. Create Dispatch Strategy ===
            from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
            dispatch_strategy = HybridArbitrageEngineStrategy()
            
            # === 3. Create SimulationEngine ===
            from h2_plant.simulation.engine import SimulationEngine
            from h2_plant.config.plant_config import SimulationConfig
            
            # Convert context.simulation to SimulationConfig if needed
            sim_config = SimulationConfig(
                timestep_hours=self.context.simulation.timestep_hours,
                duration_hours=self.context.simulation.duration_hours,
                start_hour=0
            )
            
            output_dir = Path("simulation_output")
            engine = SimulationEngine(
                registry=registry,
                config=sim_config,
                output_dir=output_dir,
                dispatch_strategy=dispatch_strategy
            )
            
            # === 4. Load Price/Wind Data ===
            from h2_plant.data.price_loader import EnergyPriceLoader
            
            data_dir = Path(self.context.simulation.energy_price_file).parent
            loader = EnergyPriceLoader(str(data_dir))
            
            total_steps = int(self.context.simulation.duration_hours * 60)
            prices, wind = loader.load_data(
                self.context.simulation.energy_price_file,
                self.context.simulation.wind_data_file,
                duration_hours=self.context.simulation.duration_hours,
                timestep_hours=self.context.simulation.timestep_hours
            )
            
            # Ensure arrays are correct length
            if len(prices) < total_steps:
                prices = np.tile(prices, int(np.ceil(total_steps / len(prices))))[:total_steps]
            if len(wind) < total_steps:
                wind = np.tile(wind, int(np.ceil(total_steps / len(wind))))[:total_steps]
            
            # === 5. Initialize Engine with Dispatch Strategy ===
            engine.set_dispatch_data(prices, wind)
            engine.initialize()
            engine.initialize_dispatch_strategy(self.context, total_steps)
            
            # === 6. Run Simulation ===
            logger.info(f"Running simulation for {self.context.simulation.duration_hours} hours...")
            _ = engine.run(
                start_hour=0,
                end_hour=self.context.simulation.duration_hours
            )
            
            # === 7. Get Dispatch History for Plotting ===
            # The plotter expects a flat dict of arrays, not the nested engine results
            dispatch_history = engine.get_dispatch_history()
            
            if dispatch_history:
                # Convert numpy arrays to lists for plotter compatibility
                results = {}
                for key, arr in dispatch_history.items():
                    if isinstance(arr, np.ndarray):
                        results[key] = arr.tolist()
                    else:
                        results[key] = arr
            else:
                logger.warning("No dispatch history available, using empty results")
                results = {}
            
            # === 8. Add required keys for plotter ===
            # Map dispatch history keys to expected plotter keys
            results = self._normalize_results(results, total_steps)
            
            # Emit Results
            logger.info("Simulation completed successfully!")
            self.finished.emit(results)
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            logger.error(traceback.format_exc())
            self.error.emit(str(e))
    
    def _normalize_results(self, results: dict, total_steps: int) -> dict:
        """
        Normalize result keys for plotter compatibility.
        
        Maps SimulationEngine/DispatchStrategy keys to plotter expected keys.
        """
        # Ensure minute index exists
        if 'minute' not in results:
            results['minute'] = list(range(total_steps))
        
        # Map power keys
        key_mappings = {
            'P_soec_actual': 'P_soec',
            'spot_price': 'Spot',
            'H2_soec_kg': 'H2_soec',
            'H2_pem_kg': 'H2_pem',
            'steam_soec_kg': 'Steam_soec',
            'H2O_pem_kg': 'H2O_pem',
        }
        
        for src_key, dst_key in key_mappings.items():
            if src_key in results and dst_key not in results:
                results[dst_key] = results[src_key]
        
        # Ensure required keys exist with defaults
        required_defaults = {
            'P_soec': 0.0,
            'P_pem': 0.0,
            'P_sold': 0.0,
            'P_offer': 0.0,
            'Spot': 0.0,
            'H2_soec': 0.0,
            'H2_pem': 0.0,
            'Steam_soec': 0.0,
            'H2O_pem': 0.0,
        }
        
        for key, default in required_defaults.items():
            if key not in results:
                results[key] = [default] * total_steps
        
        return results
            
    def stop(self):
        self._is_running = False

