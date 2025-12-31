"""
Pressure Swing Adsorption (PSA) Component.

This module implements a Pressure Swing Adsorption unit for high-purity
gas separation. PSA exploits the pressure-dependent adsorption behavior
of molecular sieves to separate gas mixtures.

Physical Model:
    - **Adsorption/Regeneration**: Cyclic process controlled by pressure swing.
      High pressure favors adsorption of impurities (H₂O, CO₂, N₂), while
      low pressure triggers desorption (regeneration).
    - **Mass Balance**: Strict conservation of mass.
      m_feed = m_product + m_tail
      Product purity determines separation split based on feed composition.
    - **Pressure Drop**: Ergun equation for packed beds:
      ΔP = 150(1-ε)²μU/ (ε³dp²) + 1.75(1-ε)ρU² / (ε³dp)
    - **Power Consumption**: Compression work for purge gas recovery or vacuum
      pump operation during regeneration, calculated via isentropic efficiency.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Sets up cycle tracking variables.
    - `step()`: Performs discrete-time update of cycle phase and mass separation.
    - `get_state()`: Exposes product flows and power metrics.

Process Flow:
    Unit receives 'gas_in', separates it into 'purified_gas_out' (product) and
    'tail_gas_out' (waste/purge). Includes implicit electrical consumption for
    valve actuation and vacuum/purge pumps.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream


class PSA(Component):
    """
    Pressure Swing Adsorption unit for high-purity gas refinement.

    Models a multi-bed cyclic adsorption system. Instead of simulating individual
    bed dynamics in real-time (which requires PDE solvers), this component
    models the cycle-averaged performance using mass balance and recovery efficiency
    parameters, while still capturing physical costs like pressure drop and power.

    This component fulfills the Component Lifecycle Contract (Layer 1):
        - `initialize()`: Validates configuration and prepares streams.
        - `step()`: Advances cycle time and computes instantaneous separation.
        - `get_state()`: Reports detailed flow and cycle status.

    Attributes:
        num_beds (int): Number of adsorption beds (affects cycle smoothness).
        cycle_time_min (float): Duration of full adsorption-regeneration cycle.
        purity_target (float): Target mole fraction of primary species in product.
        recovery_rate (float): Efficiency of product recovery (0.0-1.0).
        cycle_position (float): Progress through current cycle (0.0 to 1.0).

    Example:
        >>> psa = PSA(component_id='PSA-01', recovery_rate=0.85)
        >>> psa.step(t=0.1)
        >>> h2_pure = psa.get_output('purified_gas_out')
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
        Initialize the PSA unit configuration.

        Args:
            component_id (str): Unique system identifier.
            num_beds (int): Number of beds. Used for informative state, implies
                continuity of flow.
            cycle_time_min (float): Full cycle duration in minutes.
            purity_target (float): molar purity target (e.g., 0.9999 for 99.99%).
            recovery_rate (float): Hydrogen recovery ratio (Yield).
                Yield = (Moles H2 Product) / (Moles H2 Feed).
            power_consumption_kw (float): Base electrical load for control system.
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
        
        # H2 Recovery Tracking
        self._last_h2_in_kg_h: float = 0.0

    @property
    def power_kw(self) -> float:
        """Expose power consumption in kW for dispatch tracking."""
        return self.power_consumption_kw

    def initialize(self, dt: float, registry: Any) -> None:
        """
        Prepare component for simulation.

        Fulfills Layer 1 Lifecycle Contract requirements.

        Args:
            dt (float): Timestep in hours.
            registry (ComponentRegistry): Interface to other system components.
        """
        super().initialize(dt, registry)
        self.initialized = True

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Core logic flow:
        1. Cycle Management: updates phase progress based on timestep.
        2. Mass Balance: Calculates split between Product and Tail Gas based on Recovery Rate.
        3. Physics: Calculates Pressure Drop (Ergun) and Compression Work.
        4. Stream Update: Refreshes output objects.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        if self.inlet_stream.mass_flow_kg_h <= 0:
            self.product_outlet = Stream(0.0)
            self.tail_gas_outlet = Stream(0.0)
            self.power_consumption_kw = 0.0
            return

        # Advance cycle position
        dt_hours = self.dt
        cycle_fraction = (dt_hours * 60) / self.cycle_time_min
        self.cycle_position = (self.cycle_position + cycle_fraction) % 1.0

        # Calculate Inputs
        inlet_flow_kg_h = self.inlet_stream.mass_flow_kg_h
        T_in = self.inlet_stream.temperature_k
        P_in = self.inlet_stream.pressure_pa
        rho_in = self.inlet_stream.density_kg_m3
        
        # 1. Pressure Drop Calculation (Ergun Equation)
        # Justification: Calculates loss through packed adsorbent beds
        flow_m3_s = (inlet_flow_kg_h / 3600.0) / rho_in if rho_in > 0 else 0.0
        delta_p_pa = self._calculate_delta_p_ergun(flow_m3_s, T_in, P_in)
        P_out_pa = max(101325.0, P_in - delta_p_pa)

        # 2. Separation Logic (Cycle-Averaged Mass Balance)
        composition = self.inlet_stream.composition
        # Identify dominant species to purify
        target_species = max(composition, key=composition.get) if composition else 'H2'
        
        product_flow = inlet_flow_kg_h * self.recovery_rate
        tail_gas_flow = inlet_flow_kg_h - product_flow

        # 3. Dynamic Power Consumption
        # Justification: Energy required for purge/vacuum regeneration steps
        purge_kg_s = tail_gas_flow / 3600.0
        # Assume regeneration at 1 bar, adsorption at P_in
        self.power_consumption_kw = self._calculate_purge_power(
            purge_kg_s, T_in, P_in, 101325.0
        ) + 1.0 # +1kW control overhead base load

        # 4. Composition Update
        # Product: Enriched to target purity
        product_composition = {target_species: self.purity_target}
        for species in composition:
            if species != target_species:
                product_composition[species] = (1 - self.purity_target) / max(1, len(composition) - 1)

        # Tail Gas: Contains rejected impurities via conservation of mass
        tail_gas_composition = {}
        total_tail_mol_frac = 0.0

        for species in composition:
             inlet_mass_i = composition[species] * inlet_flow_kg_h
             product_mass_i = product_composition.get(species, 0.0) * product_flow
             tail_mass_i = max(0.0, inlet_mass_i - product_mass_i)

             if tail_gas_flow > 1e-9:
                 tail_gas_composition[species] = tail_mass_i / tail_gas_flow
             else:
                 tail_gas_composition[species] = 0.0

             total_tail_mol_frac += tail_gas_composition[species]

        if total_tail_mol_frac > 0:
            for s in tail_gas_composition:
                tail_gas_composition[s] /= total_tail_mol_frac

        # Create outlet streams
        self.product_outlet = Stream(
            mass_flow_kg_h=product_flow,
            temperature_k=T_in, # Isothermal assumption for PSA product
            pressure_pa=P_out_pa,
            composition=product_composition
        )

        self.tail_gas_outlet = Stream(
            mass_flow_kg_h=tail_gas_flow,
            temperature_k=T_in,
            pressure_pa=101325.0,
            composition=tail_gas_composition
        )


        
    def _calculate_delta_p_ergun(
        self,
        flow_m3_s: float,
        temperature_k: float,
        pressure_pa: float,
        density_kg_m3: float = None
    ) -> float:
        """
        Calculate pressure drop using Ergun equation.
        
        Args:
            flow_m3_s (float): Volumetric flow rate (m³/s).
            temperature_k (float): Gas temperature (K).
            pressure_pa (float): Gas pressure (Pa).
            density_kg_m3 (float, optional): Gas density (kg/m³). If None, calculated for pure H2 (Ideal).
            
        Returns:
            float: Pressure drop (Pa).
        """
        if flow_m3_s <= 0:
            return 0.0
            
        # Bed geometry (Assumed typical values matching legacy model)
        D_bed = 0.35  # m
        L_bed = 1.0   # m
        epsilon = 0.40 # Void fraction
        dp = 0.003     # Particle diameter (m)
        
        A_c = 3.14159 * (D_bed / 2)**2
        u = flow_m3_s / A_c # Superficial velocity
        
        # Fluid properties (Approximate for H2 dominance)
        # Viscosity H2 at 300K ~ 9e-6 Pa.s
        mu = 9.0e-6 
        
        # Density 
        if density_kg_m3 is not None:
            rho = density_kg_m3
        else:
            # Fallback: Density (Ideal Gas H2)
            R_spec = 4124.0 # J/kgK for H2
            rho = pressure_pa / (R_spec * temperature_k)
        
        # Ergun Equation
        # Term 1 (Viscous/Laminar): 150 * (1-eps)^2 * mu * u / (eps^3 * dp^2)
        term1 = (150 * (1 - epsilon)**2 * mu * u) / (epsilon**3 * dp**2)
        
        # Term 2 (Inertial/Turbulent): 1.75 * (1-eps) * rho * u^2 / (eps^3 * dp)
        term2 = (1.75 * (1 - epsilon) * rho * u**2) / (epsilon**3 * dp)
        
        delta_p_per_m = term1 + term2
        return delta_p_per_m * L_bed

    def _calculate_purge_power(
        self,
        purge_flow_kg_s: float,
        temperature_k: float,
        p_high_pa: float,
        p_low_pa: float
    ) -> float:
        """Estimate isentropic power for purge gas handling (compression/vacuum)."""
        if purge_flow_kg_s <= 0:
            return 0.0
            
        # Isentropic compression: W = m * Cp * T * [(P2/P1)^((k-1)/k) - 1] / eta
        # H2 properties
        Cp = 14300.0 # J/kgK
        gamma = 1.4
        eta = 0.75
        
        exponent = (gamma - 1) / gamma
        # Work to re-compress from low P (regeneration) to high P (or just vacuum work)
        # Legacy assumed work proportional to pressure ratio
        ratio = p_high_pa / p_low_pa
        
        work_j_s = (purge_flow_kg_s * Cp * temperature_k / eta) * (ratio**exponent - 1.0)
        return work_j_s / 1000.0 # kW

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Advances cycle position and performs separation calculation:
        1. Advance cycle based on timestep duration.
        2. Identify target species (dominant in feed).
        3. Calculate product and tail gas flows via recovery rate.
        4. Calculate Pressure Drop (Ergun).
        5. Calculate Power Consumption (Purge/Valve work).
        6. Determine compositions with strict mass balance.

        Args:
            t (float): Current simulation time in hours.
        """
        super().step(t)

        if self.inlet_stream.mass_flow_kg_h <= 0:
            self.product_outlet = Stream(0.0)
            self.tail_gas_outlet = Stream(0.0)
            self.power_consumption_kw = 0.0
            return

        # Advance cycle position
        dt_hours = self.dt
        cycle_fraction = (dt_hours * 60) / self.cycle_time_min
        self.cycle_position = (self.cycle_position + cycle_fraction) % 1.0

        # Calculate Inputs
        inlet_flow_kg_h = self.inlet_stream.mass_flow_kg_h
        T_in = self.inlet_stream.temperature_k
        P_in = self.inlet_stream.pressure_pa
        rho_in = self.inlet_stream.density_kg_m3
        
        # 1. Pressure Drop (Ergun)
        flow_m3_s = (inlet_flow_kg_h / 3600.0) / rho_in if rho_in > 0 else 0.0
        delta_p_pa = self._calculate_delta_p_ergun(flow_m3_s, T_in, P_in, density_kg_m3=rho_in)
        P_out_pa = max(101325.0, P_in - delta_p_pa)

        # 2. Separation Logic (Species-Specific Recovery)
        # Identify target species (Dominant in feed, default H2)
        composition = self.inlet_stream.composition
        target_species = max(composition, key=composition.get) if composition else 'H2'
        
        # Calculate mass flows
        target_mass_in = composition.get(target_species, 0.0) * inlet_flow_kg_h
        impurities_mass_in = inlet_flow_kg_h - target_mass_in
        
        # Apply Recovery Rate to TARGET SPECIES only (Legacy Logic: ETA_REC applies to H2)
        target_mass_out = target_mass_in * self.recovery_rate
        
        # Calculate allowed impurities in product based on Purity Target
        # mass_frac_target = purity_target (Assuming purity is defined mass-basis for simplicity? 
        # Docstring says "molar purity". 
        # If Purity is Molar, converting to Mass depends on MW.
        # For simplicity and robustness, and since Stream uses Mass, we interpret purity_target as Mass Fraction 
        # OR we perform conversion. 
        # Legacy: "Y_H2O_OUT_PPM" (Mole/Vol).
        # PSA Purity Target: 0.9999.
        # If 99.99% Mole H2, Mass H2 is even higher (lightest gas).
        # Let's assume Purity Target is MASS FRACTION for `Stream` consistency unless configured otherwise.
        # If user provides 0.9999, Mass Fraction is safe enough.
        
        # product_total = target_mass_out / purity_target
        # This implies we PULL impurities from the feed to dilute the product?
        # No, physically, some impurities slip.
        # If we enforce purity, we define impurity_mass_out = target_mass_out * (1/purity - 1).
        # We must ensure we don't request more impurities than exist (unlikely for high purity).
        
        impurity_mass_out_req = target_mass_out * (1.0/self.purity_target - 1.0)
        impurity_mass_out = min(impurity_mass_out_req, impurities_mass_in) # Cap at available
        
        product_flow = target_mass_out + impurity_mass_out
        tail_gas_flow = inlet_flow_kg_h - product_flow

        # 3. Power Consumption (Dynamic)
        # Legacy: Includes vacuum/compression work for purge
        # Purge mass = Tail gas mass
        purge_kg_s = tail_gas_flow / 3600.0
        # Assume regeneration at 1 bar, adsorption at P_in
        self.power_consumption_kw = self._calculate_purge_power(
            purge_kg_s, T_in, P_in, 101325.0
        ) + 1.0 # +1kW control overhead

        # 4. Composition Update
        # Product: Reconstruct composition
        product_composition = {target_species: target_mass_out / product_flow}
        
        # Distribute remaining impurity mass proportional to inlet impurity ratios
        if impurity_mass_out > 0:
            total_impurity_in = impurities_mass_in
            if total_impurity_in > 0:
                for species, frac in composition.items():
                    if species == target_species:
                        continue
                    # Mass of species i in outlet
                    mass_i_in = frac * inlet_flow_kg_h
                    ratio_i = mass_i_in / total_impurity_in
                    mass_i_out = impurity_mass_out * ratio_i
                    product_composition[species] = mass_i_out / product_flow
            else:
                # No impurities but purity requirement forced? (Impossible if min cap worked)
                pass
        
        # Tail Gas: Conservation of Mass
        tail_gas_composition = {}
        if tail_gas_flow > 1e-9:
            for species in composition:
                mass_i_in = composition[species] * inlet_flow_kg_h
                mass_i_prod = product_composition.get(species, 0.0) * product_flow
                mass_i_tail = max(0.0, mass_i_in - mass_i_prod)
                tail_gas_composition[species] = mass_i_tail / tail_gas_flow
        else:
            tail_gas_composition = {target_species: 1.0} # Fallback

        # Check normalization
        # Streams auto-normalize, but good to be precise.

        # Create outlet streams
        self.product_outlet = Stream(
            mass_flow_kg_h=product_flow,
            temperature_k=T_in, # Isothermal 
            pressure_pa=P_out_pa,
            composition=product_composition
        )

        self.tail_gas_outlet = Stream(
            mass_flow_kg_h=tail_gas_flow,
            temperature_k=T_in,
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
        if port_name == "purified_gas_out" or port_name == "outlet":
            return self.product_outlet
        elif port_name == "tail_gas_out":
            return self.tail_gas_outlet
        
        raise ValueError(f"Unknown output port '{port_name}'")

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Accept input at specified port.
        Also tracks H2 mass flow entering for recovery balance.

        Args:
            port_name (str): Target port ('gas_in' or 'electricity_in').
            value (Any): Stream object or power value.
            resource_type (str, optional): Resource classification hint.

        Returns:
            float: Amount accepted (flow rate or power).
        """
        if port_name == "gas_in" and isinstance(value, Stream):
            self.inlet_stream = value
            # Track inlet H2 mass for recovery balance
            y_h2 = value.composition.get('H2', 0.0)
            self._last_h2_in_kg_h = value.mass_flow_kg_h * y_h2
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
                - h2_in_kg_h (float): H2 mass entering (kg/h).
                - h2_product_kg_h (float): H2 mass in product (kg/h).
                - h2_tail_loss_kg_h (float): H2 mass lost to tail gas (kg/h).
                - h2_recovery_actual (float): Actual H2 recovery fraction (0-1).
        """
        # Calculate H2 loss to tail gas
        h2_loss = 0.0
        h2_product = 0.0
        if self.tail_gas_outlet:
            h2_loss = self.tail_gas_outlet.mass_flow_kg_h * self.tail_gas_outlet.composition.get('H2', 0.0)
        if self.product_outlet:
            h2_product = self.product_outlet.mass_flow_kg_h * self.product_outlet.composition.get('H2', 0.0)
        
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'product_flow_kg_h': self.product_outlet.mass_flow_kg_h,
            'tail_gas_flow_kg_h': self.tail_gas_outlet.mass_flow_kg_h,
            'cycle_position': self.cycle_position,
            'power_consumption_kw': self.power_consumption_kw,
            'outlet_o2_ppm_mol': (self.product_outlet.get_total_mole_frac('O2') * 1e6) if self.product_outlet else 0.0,
            
            # H2 Recovery Tracking
            'h2_in_kg_h': self._last_h2_in_kg_h,
            'h2_product_kg_h': h2_product,
            'h2_tail_loss_kg_h': h2_loss,
            'h2_recovery_actual': (1.0 - h2_loss / self._last_h2_in_kg_h) if self._last_h2_in_kg_h > 0 else 0.0
        }
