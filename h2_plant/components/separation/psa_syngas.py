"""
Syngas Pressure Swing Adsorption (PSA) Component.

This module implements an enhanced PSA unit optimized for upgrading syngas 
(mixtures of H2, CO, CO2, CH4, H2O) into high-purity hydrogen.

Capabilities:
    - **Multi-Species Removal**: Specifically targets CO, CO2, CH4, H2O.
    - **Mass Balance**: H2_out = H2_in * Recovery_Rate.
    - **Purity Control**: Limits total impurities in product to (1 - purity_target).
    - **Tail Gas Generation**: Concentrates rejected species for downstream utilization.

Physical Model:
    - Ergun Equation for packed bed pressure drop.
    - Isentropic compression power for purge/vacuum regeneration.
    - Cycle-averaged separation logic.

Engineering Notes:
    - Stream.composition uses MOLE FRACTIONS. Conversion to mass fractions is
      performed internally for mass balance calculations.
    - Regeneration power uses empirical 50 kJ/kg_tailgas (typical for Zeolite
      adsorbent at ~0.3 bar vacuum depth). See Industrial PSA Handbook, Ch. 7.
"""

import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)

# Molar Masses (kg/kmol)
MW = {
    'H2': 2.016,
    'CO': 28.01,
    'CO2': 44.01,
    'CH4': 16.04,
    'H2O': 18.015,
    'N2': 28.014,
    'O2': 31.998
}

# Species to be adsorbed (heavy components)
SYNGAS_IMPURITIES = ['H2O', 'CO2', 'CO', 'CH4', 'N2']


class SyngasPSA(Component):
    """
    High-fidelity PSA for Syngas purification.

    Splits an incoming feed stream into:
    1. Product: High purity H2 (e.g., 99.999%).
    2. Tail Gas: Concentrated impurities + unrecovered H2 (Low pressure).

    Attributes:
        recovery_rate (float): Hydrogen recovery efficiency (Target: 0.85-0.90).
        purity_target (float): Product purity requirement (Target: 0.9999+).
        cycle_time_min (float): Cycle duration for pressure drop dynamics.
        power_consumption_kw (float): Current electrical load.
    """

    def __init__(
        self,
        component_id: str = "PSA_Syngas",
        num_beds: int = 4,
        cycle_time_min: float = 10.0,
        purity_target: float = 0.9999,
        recovery_rate: float = 0.90,
        power_consumption_kw: float = 15.0,
        adsorbent_density_kg_m3: float = 750.0,
        bed_length_m: float = 2.5,
        bed_diameter_m: float = 1.0,
        **kwargs
    ):
        """
        Initialize Syngas PSA.

        Args:
            component_id: Unique ID.
            num_beds: Number of beds (affects footprint/smoothing).
            purity_target: Target mass fraction of H2 in product.
            recovery_rate: Fraction of H2 in feed captured in product.
            power_consumption_kw: Base control/valve power.
            bed_length_m: Adsorbent bed length (for pressure drop).
            bed_diameter_m: Adsorbent bed diameter.
        """
        super().__init__(**kwargs)
        self.component_id = component_id
        self.num_beds = num_beds
        self.cycle_time_min = cycle_time_min
        self.purity_target = purity_target
        self.recovery_rate = recovery_rate
        self.base_power_kw = power_consumption_kw
        self.adsorbent_rho = adsorbent_density_kg_m3
        self.bed_length_m = bed_length_m
        self.bed_diameter_m = bed_diameter_m

        # Input buffer
        self._inlet_stream: Optional[Stream] = None

        # Output streams
        self.product_outlet: Stream = Stream(0.0)
        self.tail_gas_outlet: Stream = Stream(0.0)

        # Operational Metrics
        self.cycle_position: float = 0.0
        self.power_consumption_kw: float = 0.0
        self._last_h2_in_kg_h: float = 0.0

    def initialize(self, dt: float, registry: Any) -> None:
        """Lifecycle Phase 1: Initialize."""
        super().initialize(dt, registry)

    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Receive input stream from upstream component.

        Args:
            port_name: Port identifier ('gas_in').
            value: Stream object or scalar.
            resource_type: Resource type hint.

        Returns:
            Accepted flow rate (kg/h).
        """
        if port_name == 'gas_in' and isinstance(value, Stream):
            self._inlet_stream = value
            return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        """
        Lifecycle Phase 2: Execute Timestep.

        Performs separation of Syngas components based on recovery limits
        and calculates energy penalties.
        """
        super().step(t)

        inlet = self._inlet_stream

        # 0. Handle Zero Flow
        if inlet is None or inlet.mass_flow_kg_h <= 1e-6:
            self._set_zero_flow_state()
            return

        # 1. Update Cycle Dynamics
        dt_hours = self.dt
        cycle_fraction = (dt_hours * 60) / self.cycle_time_min
        self.cycle_position = (self.cycle_position + cycle_fraction) % 1.0

        # 2. Retrieve Inlet Properties
        in_flow_kg_h = inlet.mass_flow_kg_h
        P_in = inlet.pressure_pa
        T_in = inlet.temperature_k
        rho_in = inlet.density_kg_m3 if inlet.density_kg_m3 > 0 else 1.0

        # 3. Retrieve Composition
        # CRITICAL FIX: Stream.composition is in MOLE FRACTIONS.
        # We must convert to MASS FRACTIONS for the mass balance logic.
        comp_mole = inlet.composition or {}
        comp_mass = self._mole_to_mass_fractions(comp_mole)

        # 4. Physics: Calculate Pressure Drop (Ergun)
        flow_m3_s = (in_flow_kg_h / 3600.0) / rho_in if rho_in > 0 else 0.0
        delta_p = self._calculate_delta_p_ergun(flow_m3_s, T_in, P_in, rho_in)
        P_out_pa = max(101325.0, P_in - delta_p)

        # 5. Separation Logic: Mass Balance
        target_species = 'H2'

        # Calculate Input Masses using MASS fractions
        w_target = comp_mass.get(target_species, 0.0)
        m_target_in = w_target * in_flow_kg_h
        m_impurities_in = in_flow_kg_h - m_target_in

        # Target Output (Based on Recovery Rate)
        # H2_out = H2_in * Recovery%
        m_target_out = m_target_in * self.recovery_rate

        # Impurity Output (Based on Purity Target)
        # If Product is 99.99% H2, then Impurities = H2_out * (1/0.9999 - 1)
        if self.purity_target < 1.0:
            allowed_impurity_mass = m_target_out * (1.0 / self.purity_target - 1.0)
        else:
            allowed_impurity_mass = 0.0

        m_impurity_out = min(allowed_impurity_mass, m_impurities_in)

        # Total Product Flow
        m_product_total = m_target_out + m_impurity_out

        # 6. Tail Gas Mass Balance (Conservation)
        m_tail_total = in_flow_kg_h - m_product_total

        # 7. Compose Output Streams (in MASS fractions, then convert back to mole)
        # Product Composition (mass fractions)
        prod_comp_mass = {}
        if m_product_total > 0:
            prod_comp_mass[target_species] = m_target_out / m_product_total

            # Distribute "slip" impurities proportional to feed ratio
            if m_impurity_out > 0 and m_impurities_in > 0:
                for s, w_frac in comp_mass.items():
                    if s != target_species and w_frac > 0:
                        # Ratio of this impurity in total feed impurities
                        rel_frac = (w_frac * in_flow_kg_h) / m_impurities_in
                        prod_comp_mass[s] = (m_impurity_out * rel_frac) / m_product_total
        else:
            prod_comp_mass = {target_species: 1.0}

        # Tail Gas Composition (mass fractions)
        tail_comp_mass = {}
        if m_tail_total > 1e-6:
            # Unrecovered H2
            m_h2_tail = m_target_in - m_target_out
            tail_comp_mass[target_species] = m_h2_tail / m_tail_total

            # Rejected Impurities
            for s, w_frac in comp_mass.items():
                if s != target_species:
                    m_s_in = w_frac * in_flow_kg_h
                    m_s_prod = prod_comp_mass.get(s, 0.0) * m_product_total
                    m_s_tail = max(0.0, m_s_in - m_s_prod)
                    tail_comp_mass[s] = m_s_tail / m_tail_total
        else:
            tail_comp_mass = {'CO2': 1.0}  # Fallback

        # 8. Convert mass fractions back to mole fractions for Stream
        prod_comp_mol = self._mass_to_mole_fractions(prod_comp_mass)
        tail_comp_mol = self._mass_to_mole_fractions(tail_comp_mass)

        # 9. Power Calculation (Regeneration / Vacuum)
        # Empirical: ~50 kJ/kg tailgas for Zeolite regen at typical vacuum depth
        purge_kg_s = m_tail_total / 3600.0
        regen_power = purge_kg_s * 50.0  # kW
        self.power_consumption_kw = self.base_power_kw + regen_power

        # 10. Update Streams
        self.product_outlet = Stream(
            mass_flow_kg_h=m_product_total,
            temperature_k=T_in,  # Isothermal approximation (adsorption heat ignored)
            pressure_pa=P_out_pa,
            composition=prod_comp_mol,
            phase='gas'
        )

        # Tail gas released at lower pressure (typical: 1.1-1.5 bar)
        self.tail_gas_outlet = Stream(
            mass_flow_kg_h=m_tail_total,
            temperature_k=T_in,
            pressure_pa=130000.0,
            composition=tail_comp_mol,
            phase='gas'
        )

        self._last_h2_in_kg_h = m_target_in

        # Log for debugging
        if m_product_total > 0:
            actual_purity = prod_comp_mass.get('H2', 0.0)
            logger.debug(f"{self.component_id}: H2 purity={actual_purity:.4f}, "
                         f"recovery={self.recovery_rate:.2f}, tail={m_tail_total:.1f} kg/h")
            
            # DEBUG: Trace input composition (once per hour)
            if int((self.cycle_position * self.cycle_time_min)) % 10 == 0:
                logger.info(f"PSA [{self.component_id}] INPUT: {in_flow_kg_h:.1f} kg/h, "
                           f"H2(mol)={comp_mole.get('H2',0)*100:.1f}%, H2(mass)={comp_mass.get('H2',0)*100:.1f}%")
                logger.info(f"PSA [{self.component_id}] OUTPUT: {m_product_total:.1f} kg/h H2 product, "
                           f"tail={m_tail_total:.1f} kg/h")

    def _mole_to_mass_fractions(self, y_mol: Dict[str, float]) -> Dict[str, float]:
        """
        Convert mole fractions to mass fractions.

        Args:
            y_mol: Dictionary of mole fractions.

        Returns:
            Dictionary of mass fractions.
        """
        if not y_mol:
            return {}

        # Calculate average molecular weight
        total_mw = sum(y_mol.get(s, 0) * MW.get(s, 28.0) for s in y_mol)

        if total_mw <= 0:
            return y_mol  # Fallback to input

        w_mass = {}
        for s, y in y_mol.items():
            mw_s = MW.get(s, 28.0)
            w_mass[s] = (y * mw_s) / total_mw

        return w_mass

    def _mass_to_mole_fractions(self, w_mass: Dict[str, float]) -> Dict[str, float]:
        """
        Convert mass fractions to mole fractions.

        Args:
            w_mass: Dictionary of mass fractions.

        Returns:
            Dictionary of mole fractions.
        """
        if not w_mass:
            return {}

        # Calculate total (w_i / MW_i)
        total = sum(w / MW.get(s, 28.0) for s, w in w_mass.items() if w > 0)

        if total <= 0:
            return w_mass  # Fallback

        y_mol = {}
        for s, w in w_mass.items():
            if w > 0:
                mw_s = MW.get(s, 28.0)
                y_mol[s] = (w / mw_s) / total

        return y_mol

    def _calculate_delta_p_ergun(self, flow_m3_s: float, T: float, P: float, rho: float) -> float:
        """
        Calculate pressure drop across adsorbent bed using Ergun Equation.

        Args:
            flow_m3_s: Volumetric flow rate (m³/s).
            T: Temperature (K).
            P: Pressure (Pa).
            rho: Density (kg/m³).

        Returns:
            Pressure drop (Pa).
        """
        if flow_m3_s <= 0:
            return 0.0

        # Bed Geometry
        epsilon = 0.35  # Void fraction (packed bed)
        dp = 0.003  # Pellet diameter 3mm
        Area = 3.14159 * (self.bed_diameter_m / 2) ** 2

        u = flow_m3_s / Area  # Superficial velocity (m/s)
        mu = 1.8e-5  # Gas viscosity approx (Pa·s)

        # Ergun Equation:
        # ΔP/L = 150(1-ε)²μu/(ε³dp²) + 1.75(1-ε)ρu²/(ε³dp)
        term1 = (150 * (1 - epsilon) ** 2 * mu * u) / (epsilon ** 3 * dp ** 2)
        term2 = (1.75 * (1 - epsilon) * rho * u ** 2) / (epsilon ** 3 * dp)

        return (term1 + term2) * self.bed_length_m

    def _set_zero_flow_state(self) -> None:
        """Reset outputs during idle."""
        self.product_outlet = Stream(0.0)
        self.tail_gas_outlet = Stream(0.0)
        self.power_consumption_kw = 0.0

    def get_output(self, port_name: str) -> Any:
        """Retrieve output stream from specified port."""
        if port_name in ["purified_gas_out", "outlet", "h2_out"]:
            return self.product_outlet
        elif port_name == "tail_gas_out":
            return self.tail_gas_outlet
        return None

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Define the physical connection ports for this component."""
        return {
            'gas_in': {'type': 'input', 'resource_type': 'gas', 'units': 'kg/h'},
            'electricity_in': {'type': 'input', 'resource_type': 'electricity', 'units': 'kW'},
            'purified_gas_out': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'},
            'h2_out': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'},
            'tail_gas_out': {'type': 'output', 'resource_type': 'gas', 'units': 'kg/h'}
        }

    def get_state(self) -> Dict[str, Any]:
        """Retrieve the component's current operational state."""
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'h2_recovery_efficiency': self.recovery_rate,
            'purity_target': self.purity_target,
            'product_flow_kg_h': self.product_outlet.mass_flow_kg_h,
            'tail_gas_flow_kg_h': self.tail_gas_outlet.mass_flow_kg_h,
            'product_purity_h2': self.product_outlet.composition.get('H2', 0.0),
            'tail_gas_ch4_frac': self.tail_gas_outlet.composition.get('CH4', 0.0),
            'power_consumption_kw': self.power_consumption_kw,
            'cycle_position': self.cycle_position
        }
