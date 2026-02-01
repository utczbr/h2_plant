"""
Steam Generator Component.

This module implements a steam generator for converting pressurized water
to high-temperature steam. Essential for SOEC electrolysis feed and ATR
pre-heating applications.

Thermodynamic Model:
    Steam generation requires three energy components:
    1. **Sensible Heat (Liquid)**: Q₁ = ṁ × Cp_water × (T_boil - T_in)
    2. **Latent Heat (Vaporization)**: Q₂ = ṁ × h_fg
    3. **Sensible Heat (Vapor)**: Q₃ = ṁ × Cp_steam × (T_out - T_boil)

    Total: **Q_total = Q₁ + Q₂ + Q₃**

    Using typical values:
    - Cp_water = 4.18 kJ/(kg·K)
    - Cp_steam = 2.0 kJ/(kg·K)
    - h_fg = 2,260 kJ/kg (at 100°C)

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Calculates steam output and heat requirement.
    - `get_state()`: Returns flows, temperatures, and heat input.

Process Flow Integration:
    - HX-4: SOEC steam generation (800°C steam)
    - HX-7: ATR steam generation (reforming feed)
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class SteamGenerator(Component):
    """
    Steam generator for producing high-temperature steam.

    Converts pressurized water to steam by applying sensible and latent
    heat loads. Tracks heat input requirement for energy balance.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Calculates steam output and required heat input.
        - `get_state()`: Returns flows, temperatures, and heat duty.

    Energy Balance:
        Q_total = Q_sensible_liquid + Q_latent + Q_sensible_vapor

    Attributes:
        max_capacity_kg_h (float): Maximum steam generation rate (kg/h).
        efficiency (float): Heat transfer efficiency (0-1).
        steam_temp_k (float): Steam outlet temperature (K).
        heat_input_kw (float): Required heat input (kW).

    Example:
        >>> steam_gen = SteamGenerator(
        ...     component_id='HX-4',
        ...     max_capacity_kg_h=100.0,
        ...     steam_temp_k=1073.15  # 800°C for SOEC
        ... )
        >>> steam_gen.initialize(dt=1/60, registry=registry)
        >>> steam_gen.receive_input('water_in', water_stream, 'water')
        >>> steam_gen.step(t=0.0)
    """

    def __init__(
        self,
        component_id: str = "steam_generator",
        max_capacity_kg_h: float = 100.0,
        efficiency: float = 0.90,
        steam_temp_k: float = 423.15
    ):
        """
        Initialize the steam generator.

        Args:
            component_id (str): Unique identifier. Default: 'steam_generator'.
            max_capacity_kg_h (float): Maximum steam generation capacity in kg/h.
                Default: 100.0.
            efficiency (float): Heat transfer efficiency (0-1). Default: 0.90.
            steam_temp_k (float): Steam outlet temperature in K.
                Default: 423.15 (150°C).
        """
        super().__init__()
        self.component_id = component_id
        self.max_capacity_kg_h = max_capacity_kg_h
        self.efficiency = efficiency
        self.steam_temp_k = steam_temp_k

        # Flow tracking
        self.water_inlet_kg_h = 0.0
        self.steam_outlet_kg_h = 0.0
        self.heat_input_kw = 0.0
        self.current_inlet_temp_k = 293.15

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self._initialized = True

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Calculates steam generation and required heat input:
        1. Sensible heat to raise water to boiling point (100°C).
        2. Latent heat of vaporization.
        3. Sensible heat to superheat steam to target temperature.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # Pass-through: steam out equals water in
        self.steam_outlet_kg_h = self.water_inlet_kg_h

        # Thermodynamic constants
        CP_WATER_KJ_KG_K = 4.18
        CP_STEAM_KJ_KG_K = 2.0
        LATENT_HEAT_KJ_KG = 2260.0
        T_BOILING_K = 373.15

        t_in = self.current_inlet_temp_k
        m_dot_kg_s = self.water_inlet_kg_h / 3600.0

        # Sensible heat: water → boiling point
        q_sensible_water = m_dot_kg_s * CP_WATER_KJ_KG_K * max(0, T_BOILING_K - t_in)

        # Latent heat: liquid → vapor
        q_latent = m_dot_kg_s * LATENT_HEAT_KJ_KG

        # Sensible heat: saturated steam → superheated
        q_sensible_steam = m_dot_kg_s * CP_STEAM_KJ_KG_K * max(0, self.steam_temp_k - T_BOILING_K)

        # Total heat rate in kW (converting from kJ/s)
        self.heat_input_kw = q_sensible_water + q_latent + q_sensible_steam

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - water_inlet_kg_h (float): Water inlet flow (kg/h).
                - steam_outlet_kg_h (float): Steam outlet flow (kg/h).
                - heat_input_kw (float): Required heat input (kW).
                - steam_temp_k (float): Steam outlet temperature (K).
        """
        return {
            **super().get_state(),
            "water_inlet_kg_h": self.water_inlet_kg_h,
            "steam_outlet_kg_h": self.steam_outlet_kg_h,
            "heat_input_kw": self.heat_input_kw,
            "steam_temp_k": self.steam_temp_k
        }

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('steam_out' or 'out').

        Returns:
            Stream: Steam output stream.

        Raises:
            ValueError: If port_name is not recognized.
        """
        if port_name in ['steam_out', 'out']:
            return Stream(
                mass_flow_kg_h=self.steam_outlet_kg_h,
                temperature_k=self.steam_temp_k,
                pressure_pa=30e5,
                composition={'H2O': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept input at specified port.

        Args:
            port_name (str): Target port ('water_in' or 'in').
            value (Any): Input stream or flow value.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h).
        """
        if port_name in ['water_in', 'in']:
            if isinstance(value, Stream):
                self.water_inlet_kg_h = value.mass_flow_kg_h
                self.current_inlet_temp_k = value.temperature_k
            else:
                self.water_inlet_kg_h = float(value)
                self.current_inlet_temp_k = 293.15
            return self.water_inlet_kg_h
        return 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'steam_out': {'type': 'output', 'resource_type': 'steam', 'units': 'kg/h'}
        }
