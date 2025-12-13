"""
Thermal Swing Adsorption (TSA) Unit Component.

This module implements a Thermal Swing Adsorption unit for removing water
vapor from hydrogen streams. TSA uses temperature-driven regeneration
rather than pressure swing, making it suitable for trace contaminant removal.

Operating Principle:
    - **Adsorption Phase**: At low temperature, molecular sieves (3A, 4A)
      selectively adsorb water vapor from the hydrogen stream.
    - **Regeneration Phase**: Hot purge gas (dry H₂ at ~250°C) desorbs
      accumulated water, regenerating the bed capacity.
    - **Dual-Bed Operation**: Two beds alternate between adsorption and
      regeneration, ensuring continuous drying.

Physical Model:
    - **Adsorption Capacity**: Working capacity (kg H₂O/kg adsorbent)
      determines the amount of water that can be removed per cycle.
    - **Pressure Drop**: Ergun equation for packed bed flow resistance,
      accounting for both viscous and kinetic contributions.
    - **Regeneration Energy**: Sensible heat (bed mass), desorption heat
      (water latent heat), and purge gas heating.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Standard initialization.
    - `step()`: Calculates adsorption, pressure drop, and regeneration energy.
    - `get_state()`: Returns pressure drop, power, and water removal metrics.

References:
    - Ruthven, D.M. (1984). Principles of Adsorption and Adsorption Processes.
    - Perry's Chemical Engineers' Handbook, 8th Ed., Section 16.
"""

import numpy as np
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# Default bed geometry from legacy model
DEFAULT_BED_DIAMETER_M = 0.320
DEFAULT_BED_LENGTH_M = 0.800
DEFAULT_PARTICLE_DIAMETER_M = 0.0025
DEFAULT_BED_POROSITY = 0.40
DEFAULT_ADSORBENT_DENSITY_KG_M3 = 700.0
DEFAULT_CYCLE_TIME_HOURS = 6.0
DEFAULT_REGEN_TEMP_K = 250.0 + 273.15  # 523.15 K

# Material properties
MM_H2 = 2.016e-3    # kg/mol
MM_H2O = 18.015e-3  # kg/mol
R_UNIVERSAL = 8.314  # J/(mol·K)
CP_ADSORBENT_J_KG_K = 900.0
HEAT_ADSORPTION_J_KG = 2000.0 * 1000.0  # 2000 kJ/kg → J/kg


class TSAUnit(Component):
    """
    Thermal Swing Adsorption unit for hydrogen drying.

    Removes water vapor from hydrogen streams using molecular sieve
    adsorbents with thermal regeneration. Models adsorption dynamics,
    Ergun pressure drop, and regeneration energy requirements.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Standard component initialization.
        - `step()`: Calculates water removal, pressure drop, and regen power.
        - `get_state()`: Returns operational metrics and energy consumption.

    The drying process:
    1. Wet H₂ enters active adsorption bed.
    2. Water vapor adsorbs onto molecular sieves (100% efficiency until saturation).
    3. Pressure drop calculated via Ergun equation.
    4. Inactive bed regenerates with hot purge gas.
    5. Beds switch when cycle time expires.

    Attributes:
        bed_diameter (float): Adsorption bed diameter (m).
        bed_length (float): Bed length (m).
        num_beds (int): Number of beds (typically 2 for continuous operation).
        cycle_time_hours (float): Total cycle duration (hours).
        working_capacity (float): Adsorbent capacity (kg H₂O/kg adsorbent).

    Example:
        >>> tsa = TSAUnit(component_id='TSA-1', cycle_time_hours=6.0)
        >>> tsa.initialize(dt=1/60, registry=registry)
        >>> tsa.receive_input('wet_h2_in', wet_stream, 'stream')
        >>> tsa.step(t=0.0)
        >>> dry_h2 = tsa.get_output('dry_h2_out')
    """

    def __init__(
        self,
        component_id: str,
        bed_diameter_m: float = DEFAULT_BED_DIAMETER_M,
        bed_length_m: float = DEFAULT_BED_LENGTH_M,
        num_beds: int = 2,
        particle_diameter_m: float = DEFAULT_PARTICLE_DIAMETER_M,
        porosity: float = DEFAULT_BED_POROSITY,
        regen_temp_k: float = DEFAULT_REGEN_TEMP_K,
        cycle_time_hours: float = DEFAULT_CYCLE_TIME_HOURS,
        working_capacity: float = 0.05,
        purge_fraction: float = 0.02
    ):
        """
        Initialize the TSA unit.

        Args:
            component_id (str): Unique identifier for this component.
            bed_diameter_m (float): Adsorption bed diameter in m. Default: 0.32.
            bed_length_m (float): Bed length in m. Default: 0.80.
            num_beds (int): Number of beds for alternating operation. Default: 2.
            particle_diameter_m (float): Adsorbent particle diameter in m.
                Affects pressure drop. Default: 0.0025.
            porosity (float): Bed void fraction (0-1). Default: 0.40.
            regen_temp_k (float): Regeneration temperature in K. Default: 523.15.
            cycle_time_hours (float): Duration of one complete cycle in hours.
                Default: 6.0.
            working_capacity (float): Adsorbent capacity in kg H₂O/kg adsorbent.
                Determines bed sizing. Default: 0.05.
            purge_fraction (float): Fraction of feed used for purge gas.
                Default: 0.02.
        """
        super().__init__()
        self.component_id = component_id
        self.bed_diameter = bed_diameter_m
        self.bed_length = bed_length_m
        self.num_beds = num_beds
        self.particle_diameter = particle_diameter_m
        self.porosity = porosity
        self.regen_temp_k = regen_temp_k
        self.cycle_time_hours = cycle_time_hours
        self.working_capacity = working_capacity
        self.purge_fraction = purge_fraction

        # Calculated geometry
        self.bed_area = np.pi * (self.bed_diameter ** 2) / 4.0
        self.adsorbent_mass_per_bed = (self.bed_area * self.bed_length *
                                       DEFAULT_ADSORBENT_DENSITY_KG_M3)
        self.max_water_loading_kg = self.adsorbent_mass_per_bed * self.working_capacity

        # Operating state
        self.active_bed_idx = 0
        self.water_loading_kg = 0.0
        self.cycle_timer_h = 0.0
        self.total_h2_purified_kg = 0.0
        self.total_water_removed_kg = 0.0
        self.is_saturated = False

        # Performance metrics
        self.last_pressure_drop_bar = 0.0
        self.last_regen_energy_kw = 0.0
        self.electrical_power_kw = 0.0

        # Input stream
        self.inlet_stream: Optional[Stream] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation execution.

        Fulfills the Component Lifecycle Contract initialization phase.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Performs complete TSA calculation sequence:
        1. Calculate gas properties (density, viscosity, molar flows).
        2. Adsorb water vapor (100% efficiency until bed saturation).
        3. Compute Ergun pressure drop through packed bed.
        4. Calculate regeneration energy (sensible + desorption + purge).
        5. Advance cycle timer and switch beds when cycle completes.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        if self.inlet_stream is None:
            return

        stream = self.inlet_stream

        # Gas composition
        y_h2o = stream.composition.get('H2O', 0.0)
        y_h2 = stream.composition.get('H2', 1.0 - y_h2o)

        # Mixture molecular weight
        mm_mix = y_h2o * MM_H2O + (1.0 - y_h2o) * MM_H2

        # Ideal gas density: ρ = PM/(RT)
        rho_gas = (stream.pressure_pa * mm_mix) / (R_UNIVERSAL * stream.temperature_k)

        # Dynamic viscosity (Sutherland power law for H₂)
        mu_h2 = 1.05e-5 * (stream.temperature_k / 277.15) ** 0.7

        # Molar and mass flow rates
        q_molar_total = (stream.mass_flow_kg_h / 3600.0) / mm_mix
        q_mass_h2o_s = q_molar_total * y_h2o * MM_H2O
        q_mass_h2_s = (stream.mass_flow_kg_h / 3600.0) - q_mass_h2o_s

        # Adsorption physics (active bed)
        if self.water_loading_kg >= self.max_water_loading_kg:
            self.is_saturated = True
            water_removed_this_step_kg = 0.0
        else:
            self.is_saturated = False
            water_removed_this_step_kg = q_mass_h2o_s * (self.dt * 3600.0)

        self.water_loading_kg += water_removed_this_step_kg
        self.total_water_removed_kg += water_removed_this_step_kg
        self.total_h2_purified_kg += q_mass_h2_s * (self.dt * 3600.0)

        # Ergun pressure drop: ΔP/L = 150μU(1-ε)²/(Dp²ε³) + 1.75ρU²(1-ε)/(Dpε³)
        G = (stream.mass_flow_kg_h / 3600.0) / self.bed_area
        u_s = G / rho_gas

        term_viscous = (150.0 * ((1.0 - self.porosity)**2 / (self.porosity**3)) *
                       (mu_h2 * u_s / (self.particle_diameter**2)))

        term_kinetic = (1.75 * ((1.0 - self.porosity) / (self.porosity**3)) *
                       (rho_gas * (u_s**2) / self.particle_diameter))

        delta_p_pa = self.bed_length * (term_viscous + term_kinetic)
        self.last_pressure_drop_bar = delta_p_pa / 1e5

        # Regeneration energy calculation (inactive bed)
        eta = 0.85  # Heater efficiency

        # Sensible heat: Q = m × Cp × ΔT
        q_sensible_cycle_j = (self.adsorbent_mass_per_bed * CP_ADSORBENT_J_KG_K *
                             (self.regen_temp_k - stream.temperature_k))

        # Desorption heat: Q = m_water × ΔH_ads
        cycle_water_kg = q_mass_h2o_s * (self.cycle_time_hours * 3600.0)
        q_desorption_cycle_j = cycle_water_kg * HEAT_ADSORPTION_J_KG

        # Purge gas heating: Q = m_purge × Cp × ΔT
        cp_h2 = 14300.0  # J/(kg·K)
        mass_purge_cycle_kg = (stream.mass_flow_kg_h * self.cycle_time_hours) * self.purge_fraction
        q_purge_cycle_j = mass_purge_cycle_kg * cp_h2 * (self.regen_temp_k - stream.temperature_k)

        total_energy_cycle_j = (q_sensible_cycle_j + q_desorption_cycle_j + q_purge_cycle_j) / eta

        # Average regeneration power
        avg_power_w = total_energy_cycle_j / (self.cycle_time_hours * 3600.0)
        self.last_regen_energy_kw = avg_power_w / 1000.0
        self.electrical_power_kw = self.last_regen_energy_kw

        # Cycle timing and bed switching
        self.cycle_timer_h += self.dt
        if self.cycle_timer_h >= self.cycle_time_hours:
            self.active_bed_idx = 1 - self.active_bed_idx
            self.cycle_timer_h = 0.0
            self.water_loading_kg = 0.0
            self.is_saturated = False

    def receive_input(self, port_name: str, value: Any, resource_type: str = 'stream') -> None:
        """
        Accept input stream at specified port.

        Args:
            port_name (str): Target port ('wet_h2_in').
            value (Any): Stream object containing wet hydrogen.
            resource_type (str): Resource classification hint. Default: 'stream'.
        """
        if port_name == 'wet_h2_in' and isinstance(value, Stream):
            self.inlet_stream = value

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Port identifier ('dry_h2_out' or 'water_out').

        Returns:
            Stream: Requested output stream, or None if unavailable.
        """
        if self.inlet_stream is None:
            return None

        if port_name == 'dry_h2_out':
            # Calculate dry H₂ mass flow
            y_h2o = self.inlet_stream.composition.get('H2O', 0.0)
            mm_mix = y_h2o * MM_H2O + (1.0 - y_h2o) * MM_H2
            molar_flow = (self.inlet_stream.mass_flow_kg_h / 3600.0) / mm_mix
            h2_molar_flow = molar_flow * (1.0 - y_h2o)
            h2_mass_flow = h2_molar_flow * MM_H2 * 3600.0

            return Stream(
                mass_flow_kg_h=h2_mass_flow,
                temperature_k=self.inlet_stream.temperature_k,
                pressure_pa=self.inlet_stream.pressure_pa - (self.last_pressure_drop_bar * 1e5),
                composition={'H2': 1.0, 'H2O': 0.0},
                phase='gas'
            )

        elif port_name == 'water_out':
            y_h2o = self.inlet_stream.composition.get('H2O', 0.0)
            if y_h2o == 0:
                return Stream(0, 300, 1e5, {'H2O': 1}, 'liquid')

            dry_h2_flow = self.get_output('dry_h2_out').mass_flow_kg_h
            return Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h - dry_h2_flow,
                temperature_k=self.inlet_stream.temperature_k,
                pressure_pa=1e5,
                composition={'H2O': 1.0},
                phase='liquid'
            )

        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.

        Returns:
            Dict[str, Dict[str, str]]: Port definitions.
        """
        return {
            'wet_h2_in': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'dry_h2_out': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'},
            'water_out': {'type': 'output', 'resource_type': 'water', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Fulfills the Component Lifecycle Contract state access.

        Returns:
            Dict[str, Any]: State dictionary containing:
                - bed_diameter_m (float): Bed diameter (m).
                - pressure_drop_bar (float): Current ΔP (bar).
                - regen_power_kw (float): Regeneration power (kW).
                - water_removed_total_kg (float): Cumulative water (kg).
                - current_water_loading_kg (float): Current bed loading (kg).
        """
        return {
            **super().get_state(),
            'bed_diameter_m': self.bed_diameter,
            'pressure_drop_bar': self.last_pressure_drop_bar,
            'regen_power_kw': self.last_regen_energy_kw,
            'water_removed_total_kg': self.total_water_removed_kg,
            'current_water_loading_kg': self.water_loading_kg
        }
