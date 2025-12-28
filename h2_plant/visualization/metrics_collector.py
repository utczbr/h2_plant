"""
MetricsCollector: Centralized time-series data collection during simulation.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesBuffer:
    """Buffer for storing time-series data for a specific metric."""
    timestamps: List[float] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    
    def append(self, timestamp: float, value: Any) -> None:
        """Append a new data point."""
        self.timestamps.append(timestamp)
        self.values.append(value)
    
    def to_numpy(self) -> tuple:
        """Convert to numpy arrays."""
        return np.array(self.timestamps), np.array(self.values)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.timestamps.clear()
        self.values.clear()


class MetricsCollector:
    """
    Centralized collector for simulation metrics.
    
    Subscribes to component state updates and maintains time-series buffers
    for all relevant metrics needed for visualization.
    """
    
    def __init__(self):
        """Initialize the metrics collector with empty buffers.
        
        .. deprecated::
            MetricsCollector is not integrated with the main simulation loop.
            Use the history DataFrame from `run_integrated_simulation.py` instead.
            Graph generation should use `GraphOrchestrator` with `visualization_config.yaml`.
        """
        import warnings
        warnings.warn(
            "MetricsCollector is deprecated and unused. "
            "Use GraphOrchestrator with visualization_config.yaml instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.timeseries = {
            'timestamps': [],
            'pem': {
                'h2_production_kg_h': [],
                'voltage': [],
                'efficiency': [],
                'power_mw': [],
                'cumulative_h2_kg': [],
                'cumulative_energy_kwh': []
            },
            'soec': {
                'h2_production_kg_h': [],
                'module_states': [],  # Will be 2D array [timestep][module_id]
                'active_modules': [],
                'power_mw': [],
                'cumulative_h2_kg': [],
                'cumulative_energy_kwh': [],
                'ramp_rates': []  # New: MW/min
            },
            'tanks': {
                'lp_masses': [],  # Will be 2D array [timestep][tank_id]
                'lp_states': [],
                'hp_masses': [],
                'hp_states': [],
                'hp_pressures': [], # New: bar [timestep][tank_id]
                'total_stored': []
            },
            'pricing': {
                'energy_price_eur_kwh': [],
                'wind_coefficient': [],
                'air_density': [],
                'grid_exchange_mw': [] # New: +Import/-Export
            },
            'economics': { # New category
                'capex_cost': [],
                'opex_cost': [],
                'energy_cost': [],
                'water_cost': [],
                'lcoh_cumulative': []
            },
            'coordinator': {
                'pem_setpoint_mw': [],
                'soec_setpoint_mw': [],
                'sell_power_mw': []
            },
            'demand': {
                'current_demand_kg_h': [],
                'cumulative_demand_kg': []
            }
        }
        
        self._collection_enabled = True
        self._step_count = 0
        
        logger.info("MetricsCollector initialized")
    
    def collect_step(self, t: float, component_states: Dict[str, Dict[str, Any]]) -> None:
        """
        Collect metrics from all components at a given timestep.
        
        Args:
            t: Current simulation time (hours)
            component_states: Dictionary of component states from registry
        """
        if not self._collection_enabled:
            return
        
        self.timeseries['timestamps'].append(t)
        self._step_count += 1
        
        # PEM Electrolyzer
        if 'pem_electrolyzer_detailed' in component_states:
            pem_state = component_states['pem_electrolyzer_detailed']
            self.timeseries['pem']['h2_production_kg_h'].append(pem_state.get('h2_production_kg_h', 0.0))
            self.timeseries['pem']['voltage'].append(pem_state.get('cell_voltage_v', 0.0))
            self.timeseries['pem']['efficiency'].append(pem_state.get('system_efficiency_percent', 0.0))
            self.timeseries['pem']['power_mw'].append(pem_state.get('power_consumption_mw', 0.0))
            self.timeseries['pem']['cumulative_h2_kg'].append(pem_state.get('cumulative_h2_kg', 0.0))
            self.timeseries['pem']['cumulative_energy_kwh'].append(pem_state.get('cumulative_energy_kwh', 0.0))
        else:
            self.timeseries['pem']['h2_production_kg_h'].append(0.0)
            self.timeseries['pem']['voltage'].append(0.0)
            self.timeseries['pem']['efficiency'].append(0.0)
            self.timeseries['pem']['power_mw'].append(0.0)
            # For cumulative, repeat last value or 0.0
            self.timeseries['pem']['cumulative_h2_kg'].append(self.timeseries['pem']['cumulative_h2_kg'][-1] if self.timeseries['pem']['cumulative_h2_kg'] else 0.0)
            self.timeseries['pem']['cumulative_energy_kwh'].append(self.timeseries['pem']['cumulative_energy_kwh'][-1] if self.timeseries['pem']['cumulative_energy_kwh'] else 0.0)
        
        # SOEC Cluster
        if 'soec_cluster' in component_states:
            soec_state = component_states['soec_cluster']
            self.timeseries['soec']['h2_production_kg_h'].append(soec_state.get('h2_production_kg_h', 0.0))
            self.timeseries['soec']['active_modules'].append(soec_state.get('active_modules', 0))
            self.timeseries['soec']['power_mw'].append(soec_state.get('power_consumption_mw', 0.0))
            self.timeseries['soec']['cumulative_h2_kg'].append(soec_state.get('cumulative_h2_kg', 0.0))
            self.timeseries['soec']['cumulative_energy_kwh'].append(soec_state.get('cumulative_energy_kwh', 0.0))
            # Calculate ramp rate
            if len(self.timeseries['soec']['power_mw']) > 1:
                ramp = (self.timeseries['soec']['power_mw'][-1] - self.timeseries['soec']['power_mw'][-2]) / 60.0
                self.timeseries['soec']['ramp_rates'].append(ramp)
            else:
                self.timeseries['soec']['ramp_rates'].append(0.0)
        else:
            self.timeseries['soec']['h2_production_kg_h'].append(0.0)
            self.timeseries['soec']['active_modules'].append(0)
            self.timeseries['soec']['power_mw'].append(0.0)
            self.timeseries['soec']['cumulative_h2_kg'].append(self.timeseries['soec']['cumulative_h2_kg'][-1] if self.timeseries['soec']['cumulative_h2_kg'] else 0.0)
            self.timeseries['soec']['cumulative_energy_kwh'].append(self.timeseries['soec']['cumulative_energy_kwh'][-1] if self.timeseries['soec']['cumulative_energy_kwh'] else 0.0)
            self.timeseries['soec']['ramp_rates'].append(0.0)

        
        # Storage Tanks
        if 'lp_tanks' in component_states:
            lp_state = component_states['lp_tanks']
            self.timeseries['tanks']['lp_masses'].append(lp_state.get('masses', []))
            self.timeseries['tanks']['lp_states'].append(lp_state.get('states', []))
        else:
            self.timeseries['tanks']['lp_masses'].append([])
            self.timeseries['tanks']['lp_states'].append([])
        
        if 'hp_tanks' in component_states:
            hp_state = component_states['hp_tanks']
            self.timeseries['tanks']['hp_masses'].append(hp_state.get('masses', []))
            self.timeseries['tanks']['hp_states'].append(hp_state.get('states', []))
            self.timeseries['tanks']['hp_pressures'].append(hp_state.get('pressures', []))
            
            total = sum(hp_state.get('masses', [])) + sum(
                self.timeseries['tanks']['lp_masses'][-1] if self.timeseries['tanks']['lp_masses'] else []
            )
            self.timeseries['tanks']['total_stored'].append(total)
        else:
            self.timeseries['tanks']['hp_masses'].append([])
            self.timeseries['tanks']['hp_states'].append([])
            self.timeseries['tanks']['hp_pressures'].append([])
            self.timeseries['tanks']['total_stored'].append(0.0)
        
        # Environment
        if 'environment_manager' in component_states:
            env_state = component_states['environment_manager']
            self.timeseries['pricing']['energy_price_eur_kwh'].append(env_state.get('energy_price_eur_kwh', 0.0))
            self.timeseries['pricing']['wind_coefficient'].append(env_state.get('wind_power_coefficient', 0.0))
            self.timeseries['pricing']['air_density'].append(env_state.get('air_density_kg_m3', 1.225))
        else:
            self.timeseries['pricing']['energy_price_eur_kwh'].append(0.0)
            self.timeseries['pricing']['wind_coefficient'].append(0.0)
            self.timeseries['pricing']['air_density'].append(1.225)
        
        # Coordinator
        if 'dual_path_coordinator' in component_states:
            coord_state = component_states['dual_path_coordinator']
            self.timeseries['coordinator']['pem_setpoint_mw'].append(coord_state.get('pem_setpoint_mw', 0.0))
            self.timeseries['coordinator']['soec_setpoint_mw'].append(coord_state.get('soec_setpoint_mw', 0.0))
            sell_mw = coord_state.get('sell_power_mw', 0.0)
            self.timeseries['coordinator']['sell_power_mw'].append(sell_mw)
            self.timeseries['pricing']['grid_exchange_mw'].append(-sell_mw)
        else:
            self.timeseries['coordinator']['pem_setpoint_mw'].append(0.0)
            self.timeseries['coordinator']['soec_setpoint_mw'].append(0.0)
            self.timeseries['coordinator']['sell_power_mw'].append(0.0)
            self.timeseries['pricing']['grid_exchange_mw'].append(0.0)

        
        # Demand
        if 'demand_scheduler' in component_states:
            demand_state = component_states['demand_scheduler']
            self.timeseries['demand']['current_demand_kg_h'].append(demand_state.get('current_demand_kg_h', 0.0))
            self.timeseries['demand']['cumulative_demand_kg'].append(demand_state.get('cumulative_demand_kg', 0.0))
        else:
            self.timeseries['demand']['current_demand_kg_h'].append(0.0)
            self.timeseries['demand']['cumulative_demand_kg'].append(self.timeseries['demand']['cumulative_demand_kg'][-1] if self.timeseries['demand']['cumulative_demand_kg'] else 0.0)
            
    def downsample(self, target_points: int = 5000) -> None:
        """
        Downsample time-series data to reduce size for visualization.
        Uses simple striding for now (LTTB is complex to implement without extra deps).
        
        Args:
            target_points: Maximum number of points to keep
        """
        current_points = len(self.timeseries['timestamps'])
        if current_points <= target_points:
            return
            
        step = current_points // target_points
        
        logger.info(f"Downsampling metrics from {current_points} to ~{target_points} points (step={step})")
        
        # Recursive function to downsample nested dictionaries
        def _downsample_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    _downsample_dict(value)
                elif isinstance(value, list):
                    # Keep every step-th element
                    d[key] = value[::step]
                    
        _downsample_dict(self.timeseries)
        self._step_count = len(self.timeseries['timestamps'])
    
    def get_dataframe(self, category: Optional[str] = None):
        """
        Convert collected metrics to pandas DataFrame.
        
        Args:
            category: Optional category filter ('pem', 'soec', 'tanks', etc.)
        
        Returns:
            pandas.DataFrame with time-series data
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed, cannot create DataFrame")
            return None
        
        if not self.timeseries['timestamps']:
            logger.warning("No data collected yet")
            return pd.DataFrame()
        
        # Base DataFrame with timestamps
        df = pd.DataFrame({'time_h': self.timeseries['timestamps']})
        
        # Add requested category or all
        if category:
            if category in self.timeseries:
                for key, values in self.timeseries[category].items():
                    if isinstance(values, list) and len(values) > 0:
                        # Handle 2D arrays (e.g., tank masses)
                        if isinstance(values[0], (list, np.ndarray)):
                            # Flatten or select specific indices
                            pass  # TODO: Implement multi-column expansion
                        else:
                            df[f'{category}_{key}'] = values
        else:
            # Add all categories
            for cat_name, cat_data in self.timeseries.items():
                if cat_name == 'timestamps':
                    continue
                for key, values in cat_data.items():
                    if isinstance(values, list) and len(values) > 0:
                        if not isinstance(values[0], (list, np.ndarray)):
                            df[f'{cat_name}_{key}'] = values
        
        return df
    
    def enable_collection(self) -> None:
        """Enable metrics collection."""
        self._collection_enabled = True
        logger.info("Metrics collection enabled")
    
    def disable_collection(self) -> None:
        """Disable metrics collection."""
        self._collection_enabled = False
        logger.info("Metrics collection disabled")
    
    def clear(self) -> None:
        """Clear all collected metrics."""
        for category in self.timeseries.values():
            if isinstance(category, dict):
                for buffer in category.values():
                    if isinstance(buffer, list):
                        buffer.clear()
            elif isinstance(category, list):
                category.clear()
        
        self._step_count = 0
        logger.info("Metrics collector cleared")
    
    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of collected data."""
        return {
            'total_steps': self._step_count,
            'time_range': (
                self.timeseries['timestamps'][0] if self.timeseries['timestamps'] else 0,
                self.timeseries['timestamps'][-1] if self.timeseries['timestamps'] else 0
            ),
            'collection_enabled': self._collection_enabled,
            'categories': list(self.timeseries.keys())
        }
