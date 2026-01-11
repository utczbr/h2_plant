"""
Signal-Based Proportional Makeup Mixer for ATR.

This component implements the "Demand Signal" pattern for the Auto-Thermal
Reforming (ATR) section. It calculates water demand proportional to a
reference stream (typically Oxygen from O2_Backup_Supply) but does NOT
generate water internally.

Architecture:
    1. Read Reference Flow (e.g., O2 output from reference component).
    2. Calculate Target Water = Reference Flow × Ratio.
    3. Calculate Deficit = Target Water - Recirculated Drain.
    4. Send 'demand_signal' (Signal Stream) to upstream Tank.
    5. Receive physical 'makeup_water_in' from Tank/Pump.
    6. Mix recirculated + makeup using enthalpy balance.

Professional Engineering Standard:
    NO FALLBACK WATER GENERATION. If makeup water is unavailable,
    the outlet flow will be reduced, correctly modeling system starvation.
"""

from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.optimization.numba_ops import solve_water_T_from_H_jit

logger = logging.getLogger(__name__)


class ProportionalMakeupMixer(Component):
    """
    Signal-based proportional makeup mixer for ATR water feed.

    The target flow rate is scaled dynamically based on a reference
    component's output flow (e.g., the Oxygen supply node). Unlike
    legacy mixers, this component does NOT generate makeup water.

    Attributes:
        max_flow_rate_kg_h (float): Design maximum flow rate.
        reference_ratio (float): Scale factor (H2O/O2 ratio).
        current_demand_kg_h (float): Deficit signal to send upstream.
        current_target_kg_h (float): Dynamically calculated target.

    Example:
        >>> mixer = ProportionalMakeupMixer(
        ...     component_id='ATR_Makeup_Mixer',
        ...     reference_component_id='O2_Backup_Supply',
        ...     reference_ratio=3.0,
        ...     max_flow_rate_kg_h=2332.0
        ... )
    """

    def __init__(
        self,
        component_id: str,
        max_flow_rate_kg_h: float = 2331.95,
        min_flow_rate_kg_h: float = 0.0,
        makeup_temp_c: float = 20.0,
        makeup_pressure_bar: float = 1.0,
        reference_component_id: str = None,
        reference_ratio: float = None,
        reference_max_flow_kg_h: float = None,
        reference_min_flow_kg_h: float = None
    ):
        """
        Initialize the proportional makeup mixer.

        Args:
            component_id (str): Unique identifier.
            max_flow_rate_kg_h (float): Maximum output flow rate (kg/h).
            min_flow_rate_kg_h (float): Minimum output flow rate (kg/h).
            makeup_temp_c (float): Default makeup temperature (°C).
            makeup_pressure_bar (float): Outlet pressure (bar).
            reference_component_id (str): ID of reference component to track.
            reference_ratio (float): Direct ratio (H2O per unit reference).
            reference_max_flow_kg_h (float): Max reference flow for auto-ratio.
            reference_min_flow_kg_h (float): Min reference flow for scaling.
        """
        super().__init__()
        self.component_id = component_id
        self.max_flow_rate_kg_h = max_flow_rate_kg_h
        self.min_flow_rate_kg_h = min_flow_rate_kg_h
        self.default_makeup_temp_k = makeup_temp_c + 273.15
        self.outlet_pressure_pa = makeup_pressure_bar * 1e5

        # Proportional control settings
        self.reference_component_id = reference_component_id
        self._reference_component = None

        # Calculate reference ratio
        if reference_ratio is not None:
            self.reference_ratio = reference_ratio
        elif reference_max_flow_kg_h is not None and reference_max_flow_kg_h > 0:
            self.reference_ratio = max_flow_rate_kg_h / reference_max_flow_kg_h
        else:
            self.reference_ratio = 1.0

        # Input streams
        self.drain_stream: Optional[Stream] = None
        self.makeup_stream: Optional[Stream] = None

        # LUT Manager
        self._lut_manager = None

        # State
        self.current_target_kg_h = 0.0
        self.current_demand_kg_h = 0.0
        self.actual_drain_kg_h = 0.0
        self.actual_makeup_kg_h = 0.0

        # Output
        self.outlet_stream: Optional[Stream] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

        # Resolve reference component
        if self.reference_component_id and registry:
            self._reference_component = registry.get(self.reference_component_id)
            if not self._reference_component:
                logger.warning(
                    f"ATRMakeupMixer {self.component_id}: "
                    f"Reference {self.reference_component_id} not found."
                )

        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Process:
        1. Calculate target flow from reference component.
        2. Measure recirculated drain flow.
        3. Calculate deficit (demand signal).
        4. Mix drain + actual makeup received.
        5. Output mixed stream.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        # 1. Determine Target Flow from Reference Component
        ref_flow = 0.0
        if self._reference_component and self.reference_ratio:
            # Try to get flow from reference component
            try:
                ref_state = self._reference_component.get_state()
                ref_flow = ref_state.get('mass_flow_kg_h', 0.0)
                if ref_flow == 0.0:
                    ref_flow = ref_state.get('outlet_flow_kg_h', 0.0)
                if ref_flow == 0.0:
                    ref_flow = ref_state.get('current_flow_kg_h', 0.0)
            except Exception:
                ref_flow = 0.0

        self.current_target_kg_h = min(
            ref_flow * self.reference_ratio,
            self.max_flow_rate_kg_h
        )

        # 2. Measure Drain (Recirculated) Input
        drain_flow = 0.0
        drain_temp_k = self.default_makeup_temp_k
        drain_pressure_pa = self.outlet_pressure_pa

        if self.drain_stream:
            drain_flow = self.drain_stream.mass_flow_kg_h
            drain_temp_k = self.drain_stream.temperature_k
            drain_pressure_pa = self.drain_stream.pressure_pa

        self.actual_drain_kg_h = drain_flow

        # 3. Calculate Deficit (Demand Signal) for NEXT timestep
        deficit = self.current_target_kg_h - drain_flow
        self.current_demand_kg_h = max(0.0, deficit)

        # 4. Measure Makeup Water Actually Received THIS step
        makeup_flow = 0.0
        makeup_temp_k = self.default_makeup_temp_k
        makeup_pressure_pa = self.outlet_pressure_pa

        if self.makeup_stream:
            makeup_flow = self.makeup_stream.mass_flow_kg_h
            makeup_temp_k = self.makeup_stream.temperature_k
            makeup_pressure_pa = self.makeup_stream.pressure_pa

        self.actual_makeup_kg_h = makeup_flow

        # 5. NO MAGIC SOURCE - Total flow is ONLY physical inputs
        total_flow = drain_flow + makeup_flow

        if total_flow <= 0:
            self.outlet_stream = Stream(0.0)
            self._clear_inputs()
            return

        # 6. Rigorous Enthalpy Mixing
        H_drain = 0.0
        H_makeup = 0.0

        if self._lut_manager:
            try:
                H_drain = self._lut_manager.lookup(
                    'H2O', 'H', drain_pressure_pa, drain_temp_k
                )
                H_makeup = self._lut_manager.lookup(
                    'H2O', 'H', makeup_pressure_pa, makeup_temp_k
                )
            except Exception:
                H_drain = 4184.0 * (drain_temp_k - 273.15)
                H_makeup = 4184.0 * (makeup_temp_k - 273.15)
        else:
            H_drain = 4184.0 * (drain_temp_k - 273.15)
            H_makeup = 4184.0 * (makeup_temp_k - 273.15)

        # Mixed enthalpy
        H_mix = ((drain_flow * H_drain) + (makeup_flow * H_makeup)) / total_flow

        # Solve for temperature
        T_guess = (drain_flow * drain_temp_k + makeup_flow * makeup_temp_k) / total_flow
        T_mix = solve_water_T_from_H_jit(H_mix, self.outlet_pressure_pa, T_guess)

        # 7. Create Outlet Stream
        self.outlet_stream = Stream(
            mass_flow_kg_h=total_flow,
            temperature_k=T_mix,
            pressure_pa=self.outlet_pressure_pa,
            composition={'H2O': 1.0},
            phase='liquid'
        )

        # 8. Clear Input Buffers
        self._clear_inputs()

    def _clear_inputs(self) -> None:
        """Clear input stream buffers for next timestep."""
        self.drain_stream = None
        self.makeup_stream = None

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input streams at specified ports.

        Ports:
            drain_in: Recirculated water from loop
            makeup_water_in: Fresh water from tank/pump

        Args:
            port_name (str): Target port.
            value (Any): Input stream.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h).
        """
        if port_name == 'drain_in' and isinstance(value, Stream):
            self.drain_stream = value
            return value.mass_flow_kg_h

        elif port_name == 'makeup_water_in' and isinstance(value, Stream):
            self.makeup_stream = value
            return value.mass_flow_kg_h

        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output streams from specified ports.

        Ports:
            water_out: Mixed water stream to downstream
            demand_signal: Signal Stream encoding deficit (kg/h)

        Args:
            port_name (str): Port identifier.

        Returns:
            Stream: Output stream or signal.
        """
        if port_name == 'water_out':
            return self.outlet_stream

        elif port_name == 'demand_signal':
            return Stream(
                mass_flow_kg_h=self.current_demand_kg_h,
                temperature_k=298.15,
                pressure_pa=101325.0,
                composition={'Signal': 1.0},
                phase='signal'
            )

        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'drain_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'makeup_water_in': {'type': 'input', 'resource_type': 'water', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'},
            'demand_signal': {'type': 'output', 'resource_type': 'signal', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: State dictionary with demand and flow metrics.
        """
        return {
            **super().get_state(),
            'max_flow_rate_kg_h': self.max_flow_rate_kg_h,
            'current_target_kg_h': self.current_target_kg_h,
            'current_demand_kg_h': self.current_demand_kg_h,
            'actual_drain_kg_h': self.actual_drain_kg_h,
            'actual_makeup_kg_h': self.actual_makeup_kg_h,
            'outlet_flow_kg_h': self.outlet_stream.mass_flow_kg_h if self.outlet_stream else 0.0,
            'reference_ratio': self.reference_ratio
        }
