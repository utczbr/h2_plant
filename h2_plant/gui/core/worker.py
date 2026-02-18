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
    
    Supports two modes:
    - Graph mode: context built from GraphToConfigAdapter
    - Scenario mode: context loaded from ConfigLoader (scenarios_dir provided)
    """
    progress = Signal(int)
    finished = Signal(dict, object)
    error = Signal(str)
    
    def __init__(self, context, strategy_override: str = None,
                 scenarios_dir: str = None):
        super().__init__()
        self.context = context
        self.strategy_override = strategy_override
        self.scenarios_dir = scenarios_dir
        self._is_running = True
        
    def run(self):
        try:
            if not hasattr(self, 'context') or self.context is None:
                raise ValueError("SimulationContext not provided to worker")

            logger.info("Starting unified SimulationEngine...")

            from h2_plant.run_integrated_simulation import run_with_dispatch_context

            # Resolve base directory for data files (energy prices, wind data).
            if self.scenarios_dir:
                data_dir = self.scenarios_dir
                output_dir = Path(self.scenarios_dir) / "simulation_output"
            else:
                data_dir = str(Path(self.context.simulation.energy_price_file).parent) or "."
                output_dir = Path("simulation_output")

            if not self._is_running:
                logger.info("Simulation cancelled before start")
                return

            # Graph mode (no scenarios_dir) uses fallback; scenario mode is strict.
            dispatch_history, registry = run_with_dispatch_context(
                self.context,
                data_dir=data_dir,
                output_dir=output_dir,
                strategy=self.strategy_override,
                allow_graph_dispatch_fallback=not bool(self.scenarios_dir),
                return_registry=True,
            )

            if not self._is_running:
                logger.info("Simulation cancelled after engine run")
                return

            total_steps = int(self.context.simulation.duration_hours * 60)
            results = dict(dispatch_history or {})
            results = self._normalize_results(results, total_steps)

            logger.info("Simulation completed successfully!")
            self.finished.emit(results, registry)

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            logger.error(traceback.format_exc())
            self.error.emit(str(e))
    
    def _normalize_results(self, results: dict, total_steps: int) -> dict:
        """
        Normalize result keys for plotter compatibility.
        
        Maps SimulationEngine/DispatchStrategy keys to plotter expected keys.
        Values remain as NumPy arrays where possible; list fallbacks are
        created only for missing keys (constant fills).
        """
        # Ensure minute index exists
        if 'minute' not in results:
            results['minute'] = np.arange(total_steps)
        
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
        
        # Ensure required keys exist with zero-filled arrays
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
                results[key] = np.full(total_steps, default)
        
        return results
            
    def stop(self):
        """Request cooperative cancellation of the simulation."""
        self._is_running = False
