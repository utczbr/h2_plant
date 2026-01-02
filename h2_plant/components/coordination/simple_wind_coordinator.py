"""
Simple Wind Coordinator with Grid Firming (Guaranteed Power)

Dispatches power to electrolyzers with a guaranteed minimum availability
from the grid connection. When wind power falls below the guaranteed floor,
the grid imports the difference to maintain continuous operation.
"""
from h2_plant.core.component import Component
import logging

logger = logging.getLogger(__name__)


class SimpleWindCoordinator(Component):
    """
    Coordinator that feeds wind power to electrolyzers,
    supplemented by a guaranteed grid connection (Baseload).
    
    Grid Firming Logic:
        - If Wind >= Guaranteed: Use Wind (Grid Import = 0)
        - If Wind < Guaranteed: Total = Guaranteed (Grid Import = Guaranteed - Wind)
    
    This ensures the SOEC system can maintain continuous operation,
    preventing thermal stress from frequent cycling.
    """
    
    def __init__(self, environment_manager, electrolyzer, soec=None, guaranteed_power_mw: float = 10.0):
        """
        Initialize the wind coordinator with grid firming.
        
        Args:
            environment_manager: Source of wind data (EnvironmentManager).
            electrolyzer: PEM electrolyzer stack.
            soec: SOEC electrolyzer stack (optional).
            guaranteed_power_mw: Minimum power availability (Grid Firming).
                Default 10.0 MW - represents the minimum power the utility
                company guarantees to provide.
        """
        super().__init__()
        self.env = environment_manager
        self.electrolyzer = electrolyzer
        self.soec = soec
        
        # Grid firming parameter
        self.guaranteed_power_mw = guaranteed_power_mw
        
        # Tracking variables for state reporting
        self._current_total_power_mw = 0.0
        self._grid_import_mw = 0.0
        
        logger.info(f"SimpleWindCoordinator initialized with guaranteed_power={guaranteed_power_mw} MW")

    def initialize(self, dt: float, registry) -> None:
        """Initialize coordinator."""
        super().initialize(dt, registry)
        
    def get_state(self) -> dict:
        """Return coordinator state including grid import data."""
        return {
            'wind_power_mw': self.env.current_wind_power_mw if hasattr(self.env, 'current_wind_power_mw') else 0.0,
            'guaranteed_power_mw': self.guaranteed_power_mw,
            'total_available_power_mw': self._current_total_power_mw,
            'grid_import_mw': self._grid_import_mw,  # Useful for OPEX calculation
            'pem_setpoint_mw': self.electrolyzer._target_power_mw if hasattr(self.electrolyzer, '_target_power_mw') else 0.0,
            'soec_setpoint_mw': self.soec._target_power_mw if self.soec and hasattr(self.soec, '_target_power_mw') else 0.0
        }
        
    def step(self, t: float) -> None:
        """Dispatch power with guaranteed floor logic."""
        super().step(t)
        
        # 1. Get Variable Renewable Energy (VRE)
        wind_power_mw = self.env.current_wind_power_mw
        
        # 2. Apply Guaranteed Power Logic (Grid Firming)
        # Logic: If Wind < Guaranteed, we import the difference to meet the floor.
        # If Wind >= Guaranteed, we use Wind (Grid Import = 0).
        if wind_power_mw < self.guaranteed_power_mw:
            total_power_available = self.guaranteed_power_mw
            self._grid_import_mw = self.guaranteed_power_mw - wind_power_mw
        else:
            total_power_available = wind_power_mw
            self._grid_import_mw = 0.0
            
        self._current_total_power_mw = total_power_available

        # 3. Dispatch Logic (Prioritizing SOEC as Baseload)
        # SOEC Reference Logic:
        SOEC_OPTIMAL_LIMIT = 0.80
        SOEC_NOMINAL_MW = 14.4  # 6 × 2.4 MW
        SOEC_MAX_CAPACITY = SOEC_NOMINAL_MW * SOEC_OPTIMAL_LIMIT  # ~11.52 MW
        
        soec_setpoint = 0.0
        if self.soec:
            # SOEC takes priority (Base Load)
            soec_setpoint = min(total_power_available, SOEC_MAX_CAPACITY)
            
        # PEM takes the Surplus (Peak Shaving)
        pem_setpoint = 0.0
        remaining_power = max(0.0, total_power_available - soec_setpoint)
        
        if self.electrolyzer:
            # Get PEM max power from electrolyzer attributes
            pem_max = getattr(self.electrolyzer, 'max_power_mw', 5.0)
            pem_setpoint = min(remaining_power, pem_max)
            
        # 4. Actuate Targets
        if self.electrolyzer:
            self.electrolyzer._target_power_mw = pem_setpoint
        if self.soec:
            self.soec._target_power_mw = soec_setpoint
        
        # Debug print (first few hours only)
        hour_index = int(t)
        if hour_index < 2:
            if self._grid_import_mw > 0:
                print(f"Coordinator: t={t:.2f}h, wind={wind_power_mw:.2f}MW, grid_import={self._grid_import_mw:.2f}MW, total={total_power_available:.2f}MW")
            if self.soec:
                print(f"  Dispatch: SOEC={soec_setpoint:.2f}MW, PEM={pem_setpoint:.2f}MW")
