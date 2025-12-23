"""
Rigorous Thermodynamic Water Mixer.

This component performs exact mass and energy balancing for liquid water streams,
accounting for non-linear property variations (e.g., Cp(T)) using high-fidelity
state equations.

Thermodynamic Principle (Adiabatic Isobaric Mixing):
    1. **Mass Balance**: ṁ_out = Σ ṁ_i
    2. **Energy Balance**: H_out = (Σ ṁ_i * h_i(T_in, P_in)) / ṁ_out
    3. **State Resolution**: T_out = T(P_out, H_out)

Computational Strategy:
    - **Tier 1 (Speed)**: `LUTManager` (Bilinear Interpolation)
    - **Tier 2 (Accuracy)**: `CoolProp` (Helmholtz Energy EOS)
"""

from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    CP = None
    COOLPROP_AVAILABLE = False

try:
    from h2_plant.optimization.coolprop_lut import CoolPropLUT
except ImportError:
    CoolPropLUT = None

logger = logging.getLogger(__name__)


class WaterMixer(Component):
    """
    Multi-port thermodynamic mixer for water streams.

    Aggregates N inlet streams into a single outlet stream, resolving final
    temperature via enthalpy conservation. This is crucial for accurate loop
    temperature tracking where simple T_mix = Avg(T_in) is insufficient due to 
    Cp variations.

    Attributes:
        outlet_pressure_kpa (float): Regulated downstream pressure (kPa).
        fluid_type (str): Fluid identifier for EOS model (default: 'Water').
        max_inlet_streams (int): Connection limit.
    """

    def __init__(
        self,
        outlet_pressure_kpa: float = 200.0,
        fluid_type: str = 'Water',
        max_inlet_streams: int = 10
    ):
        """
        Initialize the water mixer.

        Args:
            outlet_pressure_kpa (float): Fixed outlet pressure in kPa.
                Default: 200.0.
            fluid_type (str): CoolProp fluid identifier for property lookups.
                Default: 'Water'.
            max_inlet_streams (int): Maximum number of simultaneous inlet
                connections. Default: 10.
        """
        super().__init__()

        if not COOLPROP_AVAILABLE:
            logger.warning(
                "CoolProp not available. WaterMixer requires CoolProp for "
                "thermodynamic calculations. Install with: pip install CoolProp"
            )

        self.outlet_pressure_kpa = outlet_pressure_kpa
        self.outlet_pressure_pa = outlet_pressure_kpa * 1000.0
        self.fluid_type = fluid_type
        self.max_inlet_streams = max_inlet_streams

        # Dynamic inlet stream dictionary
        self.inlet_streams: Dict[str, Optional[Stream]] = {}

        # Output stream
        self.outlet_stream: Optional[Stream] = None

        # State tracking for monitoring
        self.last_mass_flow_kg_h = 0.0
        self.last_temperature_k = 0.0
        self.last_enthalpy_j_kg = 0.0

        self._lut_manager = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Executes initialization phase of Component Lifecycle.

        Connects to `LUTManager` for optimized property lookups ('H', 'T') to avoid
        runtime penalties of EOS solving where possible.

        Args:
            dt (float): Simulation timestep (hours).
            registry (ComponentRegistry): Central service registry.

        Raises:
            RuntimeError: If primary physics engine (CoolProp) is missing.
        """
        super().initialize(dt, registry)

        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')

        logger.info(
            f"WaterMixer {self.component_id} initialized. "
            f"P_out={self.outlet_pressure_kpa} kPa"
        )

        if not COOLPROP_AVAILABLE:
            raise RuntimeError(
                f"WaterMixer {self.component_id} cannot initialize: "
                "CoolProp library not available"
            )

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept water stream on a specified port.

        Validates temperature and pressure bounds before accepting.

        Args:
            port_name (str): Target inlet port name.
            value (Any): Stream object containing water.
            resource_type (str): Resource classification hint.

        Returns:
            float: Mass flow accepted (kg/h), or 0.0 if rejected.
        """
        if not isinstance(value, Stream):
            logger.warning(
                f"{self.component_id}: Expected Stream object, got {type(value)}"
            )
            return 0.0

        # Physical bounds validation
        if self.fluid_type == 'Water' and (value.temperature_k < 273.15 or value.temperature_k > 647.0):
            logger.warning(f"WaterMixer: Invalid temperature {value.temperature_k}K rejected for Water")
            return 0.0
        elif value.temperature_k < 0:
            return 0.0
        if value.pressure_pa <= 0:
            logger.warning(f"WaterMixer: Invalid pressure {value.pressure_pa}Pa rejected")
            return 0.0

        if port_name in self.inlet_streams or len(self.inlet_streams) < self.max_inlet_streams:
            self.inlet_streams[port_name] = value
            return value.mass_flow_kg_h
        else:
            logger.warning(
                f"{self.component_id}: Max inlet streams ({self.max_inlet_streams}) "
                f"reached. Rejecting stream on port '{port_name}'"
            )
            return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve mixed output stream from specified port.

        Args:
            port_name (str): Output port ('outlet', 'mixed_out', or 'output').

        Returns:
            Stream: Mixed output stream, or None if no valid output.
        """
        if port_name in ('outlet', 'mixed_out', 'output'):
            return self.outlet_stream

        logger.warning(f"{self.component_id}: Unknown output port '{port_name}'")
        return None

    def step(self, t: float) -> None:
        """
        Executes the mixing physics step.

        Process Logic:
        1. **Filter**: Identifies active streams (mass_flow > 0).
        2. **Enthalpy Calculation**: Retrieves h_i for each stream using LUTs/CoolProp.
        3. **Conservation**: Sums Mass (kg/s) and Energy (kW).
        4. **State Equation**: Solves T_out = f(H_mix_avg, P_out).
        5. **Update**: Publishes new `outlet_stream`.

        Args:
            t (float): Current simulation time (hours).
        """
        super().step(t)

        # Remove stale (zero-flow) streams
        self.inlet_streams = {
            k: v for k, v in self.inlet_streams.items()
            if v is not None and v.mass_flow_kg_h > 0
        }

        active_streams = list(self.inlet_streams.values())

        if not active_streams:
            self.outlet_stream = None
            self.last_mass_flow_kg_h = 0.0
            self.last_temperature_k = 0.0
            self.last_enthalpy_j_kg = 0.0
            return

        # Energy and mass accumulators
        total_energy_in = 0.0  # kJ/s
        total_mass_in = 0.0    # kg/s

        for stream in active_streams:
            m_dot_i = stream.mass_flow_kg_h / 3600.0  # kg/s
            T_i_C = stream.temperature_k - 273.15     # °C
            P_i_kPa = stream.pressure_pa / 1000.0     # kPa

            T_i_K = T_i_C + 273.15
            P_i_Pa = P_i_kPa * 1000.0

            try:
                # Enthalpy lookup with fallback hierarchy
                h_i_kJ_kg = 0.0

                if self._lut_manager:
                    try:
                        h_i_J_kg = self._lut_manager.lookup(self.fluid_type, 'H', P_i_Pa, T_i_K)
                        h_i_kJ_kg = h_i_J_kg / 1000.0
                    except (ValueError, RuntimeError):
                        if CoolPropLUT:
                            h_i_kJ_kg = CoolPropLUT.PropsSI('H', 'T', T_i_K, 'P', P_i_Pa, self.fluid_type) / 1000.0
                        elif CP:
                            h_i_kJ_kg = CP.PropsSI('H', 'T', T_i_K, 'P', P_i_Pa, self.fluid_type) / 1000.0
                        else:
                            raise RuntimeError("No CoolProp backend available")
                else:
                    if CoolPropLUT:
                        h_i_kJ_kg = CoolPropLUT.PropsSI('H', 'T', T_i_K, 'P', P_i_Pa, self.fluid_type) / 1000.0
                    elif CP:
                        h_i_kJ_kg = CP.PropsSI('H', 'T', T_i_K, 'P', P_i_Pa, self.fluid_type) / 1000.0
                    else:
                        raise RuntimeError("No CoolProp backend available")

                if abs(h_i_kJ_kg) < 1e-6:
                    logger.warning(f"WaterMixer: Enthalpy calculation returned 0.0 for T={T_i_K}K, P={P_i_Pa}Pa")

                total_mass_in += m_dot_i
                energy_in_i = m_dot_i * h_i_kJ_kg
                total_energy_in += energy_in_i

            except Exception as e:
                logger.error(
                    f"{self.component_id}: CoolProp error for stream at "
                    f"T={T_i_C:.2f}°C, P={P_i_kPa:.2f}kPa: {e}"
                )
                self.outlet_stream = Stream(
                    mass_flow_kg_h=0.0,
                    temperature_k=298.15,
                    pressure_pa=self.outlet_pressure_pa,
                    composition={'H2O': 1.0},
                    phase='liquid'
                )
                return

        if total_mass_in <= 0:
            self.outlet_stream = None
            return

        # Mixed enthalpy
        h_out_kJ_kg = total_energy_in / total_mass_in

        # Composition Mixing (Mass Weighted)
        mixed_composition = {}
        for stream in active_streams:
            m_stream = stream.mass_flow_kg_h
            for species, frac in stream.composition.items():
                mixed_composition[species] = mixed_composition.get(species, 0.0) + (frac * m_stream)
        
        # Normalize by total mass
        total_mass_h = total_mass_in * 3600.0
        if total_mass_h > 0:
             mixed_composition = {k: v / total_mass_h for k, v in mixed_composition.items()}
        else:
             mixed_composition = {'H2O': 1.0}

        # Inverse lookup: T from (H, P)
        h_out_J_kg = h_out_kJ_kg * 1000.0
        P_out_Pa = self.outlet_pressure_pa

        try:
            if CoolPropLUT:
                T_out_K = CoolPropLUT.PropsSI('T', 'H', h_out_J_kg, 'P', P_out_Pa, self.fluid_type)
            elif CP:
                T_out_K = CP.PropsSI('T', 'H', h_out_J_kg, 'P', P_out_Pa, self.fluid_type)
            else:
                raise RuntimeError("No CoolProp backend available")

        except Exception as e:
            logger.error(f"{self.component_id}: CoolProp error: {e}")
            self.outlet_stream = Stream(
                mass_flow_kg_h=0.0,
                temperature_k=298.15,
                pressure_pa=self.outlet_pressure_pa,
                composition={'H2O': 1.0},
                phase='liquid'
            )
            return

        # Store state for monitoring
        self.last_mass_flow_kg_h = total_mass_in * 3600.0
        self.last_temperature_k = T_out_K
        self.last_enthalpy_j_kg = h_out_J_kg

        # Create output stream
        self.outlet_stream = Stream(
            mass_flow_kg_h=total_mass_in * 3600.0,
            temperature_k=T_out_K,
            pressure_pa=P_out_Pa,
            composition=mixed_composition,
            phase='liquid'
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieves component operational telemetry.

        Fulfills Layer 1 Contract for GUI and Data Logging.

        Returns:
            Dict[str, Any]: Mixing results including N_active_inlets and PPM purity.
        """
        num_inlets = len([s for s in self.inlet_streams.values() if s is not None])
        
        # Calculate PPM
        ppm = 0.0
        if self.outlet_stream:
            # Sum all non-water species
            non_water_mass_frac = sum(
                frac for sp, frac in self.outlet_stream.composition.items() 
                if sp not in ('H2O', 'H2O_liq')
            )
            ppm = non_water_mass_frac * 1e6

        return {
            **super().get_state(),
            'outlet_pressure_kpa': float(self.outlet_pressure_kpa),
            'fluid_type': self.fluid_type,
            'num_active_inlets': num_inlets,
            'max_inlets': self.max_inlet_streams,
            'outlet_mass_flow_kg_h': float(self.last_mass_flow_kg_h),
            'outlet_temperature_k': float(self.last_temperature_k),
            'outlet_temperature_c': float(self.last_temperature_k - 273.15),
            'outlet_enthalpy_j_kg': float(self.last_enthalpy_j_kg),
            'outlet_enthalpy_kj_kg': float(self.last_enthalpy_j_kg / 1000.0),
            'dissolved_gas_ppm': float(ppm)
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions including dynamic
                inlet ports (inlet_0 through inlet_{max-1}).
        """
        ports = {
            'outlet': {
                'type': 'output',
                'resource_type': 'water',
                'units': 'kg/h',
                'phase': 'liquid'
            }
        }

        for i in range(self.max_inlet_streams):
            port_name = f'inlet_{i}'
            ports[port_name] = {
                'type': 'input',
                'resource_type': 'water',
                'units': 'kg/h',
                'phase': 'liquid'
            }

        return ports
