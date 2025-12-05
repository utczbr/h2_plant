"""
Biogas Source component for ATR feedstock.

Provides biogas input for autothermal reforming.
"""

from typing import Dict, Any
from h2_plant.core.component import Component


class BiogasSource(Component):
    """
    Biogas source for ATR feedstock supply.
    
    Provides methane-rich biogas for autothermal reforming process.
    """
    
    def __init__(
        self,
        component_id: str = "biogas_source",
        max_flow_rate_kg_h: float = 1000.0,
        methane_content: float = 0.60,  # 60% CH4 content
        pressure_bar: float = 5.0
    ):
        """
        Initialize BiogasSource.
        
        Args:
            component_id: Unique identifier
            max_flow_rate_kg_h: Maximum biogas flow rate in kg/h
            methane_content: Methane fraction (0-1)
            pressure_bar: Supply pressure in bar
        """
        super().__init__()
        self.component_id = component_id
        self.max_flow_rate_kg_h = max_flow_rate_kg_h
        self.methane_content = methane_content
        self.pressure_bar = pressure_bar
        
        # State variables
        self.biogas_output_kg_h = 0.0
        self.cumulative_biogas_kg = 0.0
        
    def initialize(self, dt: float, registry: Any) -> None:
        """Initialize component."""
        super().initialize(dt, registry)
        self._initialized = True
        
    def step(self, t: float) -> None:
        """
        Execute one timestep of biogas supply.
        
        Args:
            t: Current simulation time in hours
        """
        super().step(t)
        
        # Simple constant supply (stub - could be demand-driven later)
        self.biogas_output_kg_h = self.max_flow_rate_kg_h * 0.5  # 50% utilization
        self.cumulative_biogas_kg += self.biogas_output_kg_h * self.dt
        
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            "biogas_output_kg_h": self.biogas_output_kg_h,
            "cumulative_biogas_kg": self.cumulative_biogas_kg,
            "methane_content": self.methane_content,
            "pressure_bar": self.pressure_bar
        }

    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name in ['biogas_out', 'out']:
            from h2_plant.core.stream import Stream
            return Stream(
                mass_flow_kg_h=self.biogas_output_kg_h,
                temperature_k=300.0,
                pressure_pa=self.pressure_bar * 1e5,
                composition={'CH4': self.methane_content, 'CO2': 1.0 - self.methane_content},
                phase='gas'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")
