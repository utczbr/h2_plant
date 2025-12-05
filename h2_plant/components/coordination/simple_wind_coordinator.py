"""
Simple Wind Coordinator

Minimal coordinator to feed wind power directly to electrolyzer.
For testing/validation purposes.
"""
from h2_plant.core.component import Component

class SimpleWindCoordinator(Component):
    """Simple coordinator that feeds wind power to electrolyzer."""
    
    def __init__(self, environment_manager, electrolyzer, soec=None):
        super().__init__()
        self.env = environment_manager
        self.electrolyzer = electrolyzer
        self.soec = soec
    
    def initialize(self, dt: float, registry) -> None:
        """Initialize coordinator."""
        super().initialize(dt, registry)
        
    def get_state(self) -> dict:
        """Return coordinator state."""
        return {
            'wind_power_mw': 0.0,
            'pem_setpoint_mw': self.electrolyzer._target_power_mw if hasattr(self.electrolyzer, '_target_power_mw') else 0.0,
            'soec_setpoint_mw': self.soec._target_power_mw if self.soec and hasattr(self.soec, '_target_power_mw') else 0.0
        }
        
    def step(self, t: float) -> None:
        """Read wind power and dispatch between PEM and SOEC."""
        super().step(t)
        
        # Get wind power from environment (property from EnvironmentManager)
        wind_power_mw = self.env.current_wind_power_mw
        
        # Reference Logic (manager.py):
        # 1. SOEC has PRIORITY (Base Load)
        # 2. SOEC is capped at 80% efficient limit (11.52 MW)
        # 3. PEM takes the surplus (Peak Shaving)
        
        # Constants from reference
        SOEC_OPTIMAL_LIMIT = 0.80
        SOEC_NOMINAL_MW = 14.4 # 6 * 2.4
        SOEC_MAX_CAPACITY = SOEC_NOMINAL_MW * SOEC_OPTIMAL_LIMIT # 11.52 MW
        
        # 1. Dispatch to SOEC first
        soec_setpoint = 0.0
        if self.soec:
            # SOEC takes up to its efficient limit
            soec_setpoint = min(wind_power_mw, SOEC_MAX_CAPACITY)
            
        # 2. Dispatch to PEM (Surplus)
        pem_setpoint = 0.0
        remaining_power = max(0.0, wind_power_mw - soec_setpoint)
        
        if self.electrolyzer:
            pem_max = self.electrolyzer.max_power_mw
            pem_setpoint = min(remaining_power, pem_max)
            
        # Set the targets
        self.electrolyzer._target_power_mw = pem_setpoint
        if self.soec:
            self.soec._target_power_mw = soec_setpoint
        
        # Debug print (first 10 hours only)
        hour_index = int(t)
        if hour_index < 10:
            if self.soec:
                print(f"Coordinator: t={t}, wind={wind_power_mw:.2f}MW, SOEC={soec_setpoint:.2f}MW, PEM={pem_setpoint:.2f}MW")
            else:
                print(f"Coordinator: t={t}, wind={wind_power_mw:.2f}MW, PEM={pem_setpoint:.2f}MW")
