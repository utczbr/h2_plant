"""
Pressure Swing Adsorption (PSA) Component.

This module implements a Pressure Swing Adsorption unit for high-purity
gas separation. PSA exploits the pressure-dependent adsorption behavior
of molecular sieves to separate gas mixtures.

Operating Principle:
    PSA operates in cyclic fashion with multiple beds:
    - **Adsorption Phase**: At high pressure, impurities (H₂O, CO₂, N₂)
      preferentially adsorb onto molecular sieves while product gas passes.
    - **Regeneration Phase**: At low pressure, impurities desorb and are
      purged as tail gas.
    - **Cycle Switching**: Beds alternate between phases, providing
      continuous product flow.

Mass Balance:
    - Product: m_product = m_feed × recovery_rate
    - Tail gas: m_tail = m_feed × (1 - recovery_rate)
    - Composition: Product enriched in target species (99.99% typical),
      tail gas enriched in impurities.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Prepares component for simulation.
    - `step()`: Advances cycle position, calculates separation.
    - `get_state()`: Returns flows, cycle position, and power consumption.

Process Flow Integration:
    - D-1, D-2, D-3, D-4: Various PSA units for H₂, O₂, and syngas.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class PSA(Component):
    """
    Pressure Swing Adsorption unit for gas purification.

    Multi-bed cyclic system that produces high-purity product gas through
    pressure-driven adsorption/desorption. Tracks cycle position and
    applies mass balance with configurable recovery rate.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard initialization, marks component ready.
        - `step()`: Advances cycle and performs separation calculation.
        - `get_state()`: Returns product/tail flows and cycle metrics.

    The separation model:
    1. Identify dominant species in feed (target for purification).
    2. Apply recovery rate to determine product flow.
    3. Calculate product composition at target purity.
    4. Compute tail gas composition via strict mass balance.

    Attributes:
        num_beds (int): Number of adsorption beds.
        cycle_time_min (float): Total cycle duration (minutes).
        purity_target (float): Product purity mole fraction.
        recovery_rate (float): Fraction of feed recovered as product.
        cycle_position (float): Current cycle phase (0-1).

    Example:
        >>> psa = PSA(component_id='PSA-H2', num_beds=4, recovery_rate=0.90)
        >>> psa.initialize(dt=1/60, registry=registry)
        >>> psa.receive_input('gas_in', h2_stream, 'gas')
        >>> psa.step(t=0.0)
        >>> product = psa.get_output('purified_gas_out')
    """

    def __init__(
        self,
        component_id: str = "psa",
        num_beds: int = 2,
        cycle_time_min: float = 5.0,
        purity_target: float = 0.9999,
        recovery_rate: float = 0.90,
        power_consumption_kw: float = 10.0
    ):
        """
        Initialize the PSA unit.

        Args:
            component_id (str): Unique identifier. Default: 'psa'.
            num_beds (int): Number of adsorption beds. More beds enable
                smoother operation but increase cost. Default: 2.
            cycle_time_min (float): Total cycle time in minutes
                (adsorption + regeneration). Default: 5.0.
            purity_target (float): Target product purity as mole fraction.
                Typical values: 0.999-0.9999 for fuel cell grade. Default: 0.9999.
            recovery_rate (float): Fraction of feed recovered as product.
                Higher values reduce hydrogen losses but may lower purity.
                Default: 0.90.
            power_consumption_kw (float): Electrical power for valves and
                controls in kW. Default: 10.0.
        """
        super().__init__()
        self.component_id = component_id
        self.num_beds = num_beds
        self.cycle_time_min = cycle_time_min
        self.purity_target = purity_target
        self.recovery_rate = recovery_rate
        self.power_consumption_kw = power_consumption_kw

        # Stream state
        self.inlet_stream: Stream = Stream(0.0)
        self.product_outlet: Stream = Stream(0.0)
        self.tail_gas_outlet: Stream = Stream(0.0)

        # Cycle tracking
        self.cycle_position: float = 0.0
        self.active_beds: int = num_beds // 2

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        self.initialized = True

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Advances cycle position and performs separation calculation:
        1. Advance cycle based on timestep duration.
        2. Identify target species (dominant in feed).
        3. Calculate product and tail gas flows via recovery rate.
        4. Determine compositions with strict mass balance.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        if self.inlet_stream.mass_flow_kg_h <= 0:
            self.product_outlet = Stream(0.0)
            self.tail_gas_outlet = Stream(0.0)
            return

        # Advance cycle position
        dt_hours = self.dt
        cycle_fraction = (dt_hours * 60) / self.cycle_time_min
        self.cycle_position = (self.cycle_position + cycle_fraction) % 1.0

        # Identify target species (dominant in feed)
        composition = self.inlet_stream.composition
        target_species = max(composition, key=composition.get) if composition else 'H2'

        # Product and tail gas flows
        inlet_flow = self.inlet_stream.mass_flow_kg_h
        product_flow = inlet_flow * self.recovery_rate
        tail_gas_flow = inlet_flow - product_flow

        # Product composition (purified)
        product_composition = {target_species: self.purity_target}
        for species in composition:
            if species != target_species:
                product_composition[species] = (1 - self.purity_target) / max(1, len(composition) - 1)

        # Tail gas composition via strict mass balance
        tail_gas_composition = {}
        total_tail_mol_frac = 0.0

        for species in composition:
            inlet_mass_i = composition[species] * inlet_flow
            product_mass_i = product_composition.get(species, 0.0) * product_flow
            tail_mass_i = max(0.0, inlet_mass_i - product_mass_i)

            if tail_gas_flow > 1e-9:
                tail_gas_composition[species] = tail_mass_i / tail_gas_flow
            else:
                tail_gas_composition[species] = 0.0

            total_tail_mol_frac += tail_gas_composition[species]

        # Normalize tail gas composition
        if total_tail_mol_frac > 0:
            for s in tail_gas_composition:
                tail_gas_composition[s] /= total_tail_mol_frac

        # Create outlet streams
        self.product_outlet = Stream(
            mass_flow_kg_h=product_flow,
            temperature_k=self.inlet_stream.temperature_k,
            pressure_pa=self.inlet_stream.pressure_pa,
            composition=product_composition
        )

        self.tail_gas_outlet = Stream(
            mass_flow_kg_h=tail_gas_flow,
            temperature_k=self.inlet_stream.temperature_k,
            pressure_pa=101325.0,
            composition=tail_gas_composition
        )

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('purified_gas_out' or 'tail_gas_out').

        Returns:
            Stream: Requested output stream.
        """
        if port_name == "purified_gas_out":
            return self.product_outlet
        elif port_name == "tail_gas_out":
            return self.tail_gas_outlet
        return Stream(0.0)

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input at specified port.

        Args:
            port_name (str): Target port ('gas_in' or 'electricity_in').
            value (Any): Stream object or power value.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Amount accepted (flow rate or power).
        """
        if port_name == "gas_in" and isinstance(value, Stream):
            self.inlet_stream = value
            return value.mass_flow_kg_h
        elif port_name == "electricity_in" and isinstance(value, (int, float)):
            return min(value, self.power_consumption_kw)
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str = None) -> None:
        """
        Acknowledge extraction of output (no-op for continuous process).

        Args:
            port_name (str): Port from which output was extracted.
            amount (float): Amount extracted.
            resource_type (str, optional): Resource classification hint.
        """
        pass

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'gas_in': {'type': 'input', 'resource_type': 'gas'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity'},
            'purified_gas_out': {'type': 'output', 'resource_type': 'gas'},
            'tail_gas_out': {'type': 'output', 'resource_type': 'gas'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - product_flow_kg_h (float): Product gas rate (kg/h).
                - tail_gas_flow_kg_h (float): Tail gas rate (kg/h).
                - cycle_position (float): Current cycle phase (0-1).
                - power_consumption_kw (float): Electrical power (kW).
        """
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'product_flow_kg_h': self.product_outlet.mass_flow_kg_h,
            'tail_gas_flow_kg_h': self.tail_gas_outlet.mass_flow_kg_h,
            'cycle_position': self.cycle_position,
            'power_consumption_kw': self.power_consumption_kw
        }
