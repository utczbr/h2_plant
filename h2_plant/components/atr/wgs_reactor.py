"""
Water-Gas Shift (WGS) Reactor Component.

Physically rigorous implementation solving Chemical Equilibrium and 
Adiabatic Energy Balance using shared thermodynamic LUTs.

Reaction: CO + H2O <-> CO2 + H2 (Exothermic, ΔH ≈ -41 kJ/mol)

This component implements an adiabatic equilibrium reactor that solves
for both the extent of reaction and outlet temperature simultaneously.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# Import thermodynamic helpers
try:
    import h2_plant.optimization.mixture_thermodynamics as mix_thermo
except ImportError:
    mix_thermo = None

logger = logging.getLogger(__name__)

# Molar Masses (kg/kmol)
MW = {'CO': 28.01, 'H2O': 18.015, 'CO2': 44.01, 'H2': 2.016, 'CH4': 16.04, 'N2': 28.014}


class WGSReactor(Component):
    """
    Adiabatic Equilibrium WGS Reactor.
    
    Calculates conversion of CO to H2 based on temperature-dependent equilibrium
    constant Keq(T) and enforces adiabatic energy balance (H_in = H_out).

    The reactor solves for the reaction extent (ξ) and outlet temperature (T_out)
    iteratively:
    1. Compute Keq(T) using empirical correlation: log10(K) = 2073/T - 2.029
    2. Solve quadratic for extent of reaction at equilibrium
    3. Apply energy balance to find new T
    4. Iterate until convergence

    Attributes:
        lut_manager: Injected LUT manager for thermodynamic lookups.
        outlet_temp_k (float): Calculated outlet temperature (K).
        outlet_composition (dict): Mole fractions at outlet.
        CO_conversion (float): Fractional conversion of CO (0-1).
    """

    def __init__(self, component_id: str, conversion_rate: float = 0.0):
        """
        Initialize the WGS reactor.

        Args:
            component_id (str): Unique component identifier.
            conversion_rate (float): DEPRECATED - kept for signature compatibility.
                Actual conversion is now calculated from equilibrium.
        """
        super().__init__()
        self.component_id = component_id
        self.lut_manager = None
        
        # Input Buffer (accumulates moles over timestep)
        self.inlet_buffer: Dict[str, float] = {k: 0.0 for k in MW}  # kmol
        self.inlet_enthalpy_accumulated = 0.0  # J
        self.inlet_mass_accumulated = 0.0  # kg
        self.inlet_temp_k = 900.0  # Default inlet temperature
        self.inlet_pressure_pa = 3.0e5  # Default inlet pressure
        
        # Output State
        self.outlet_temp_k = 0.0
        self.outlet_pressure_pa = 3.0e5
        self.outlet_composition: Dict[str, float] = {}
        self.outlet_flow_kg_h = 0.0
        self.CO_conversion = 0.0
        
        # Cached output stream
        self._output_stream: Optional[Stream] = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Prepare the component for simulation.

        Args:
            dt (float): Simulation timestep in hours.
            registry (ComponentRegistry): Central registry for component access.
        """
        super().initialize(dt, registry)
        if registry and hasattr(registry, 'has') and registry.has('lut_manager'):
            self.lut_manager = registry.get('lut_manager')
        else:
            logger.warning(f"WGS {self.component_id}: LUTManager missing. "
                           "Simulation will use simplified thermodynamics.")
        
        # Pre-allocate output stream
        self._output_stream = Stream(
            mass_flow_kg_h=0.0,
            temperature_k=900.0,
            pressure_pa=3.0e5,
            composition={'H2': 1.0},
            phase='gas'
        )

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep.

        Solves the adiabatic equilibrium problem for the WGS reaction.
        """
        super().step(t)
        
        # 1. Check if we have flow
        total_moles = sum(self.inlet_buffer.values())
        if total_moles < 1e-9:
            # === DEBUG: Log early exit ===
            if int(t * 60) % 60 == 0:
                logger.warning(f"WGS [{self.component_id}] [t={t:.2f}h]: NO FLOW - inlet_buffer empty!")
            self.outlet_flow_kg_h = 0.0
            self.CO_conversion = 0.0
            self._output_stream.mass_flow_kg_h = 0.0
            return

        # 2. Get inlet species moles
        CO_in = self.inlet_buffer.get('CO', 0.0)
        H2O_in = self.inlet_buffer.get('H2O', 0.0)
        CO2_in = self.inlet_buffer.get('CO2', 0.0)
        H2_in = self.inlet_buffer.get('H2', 0.0)
        CH4_in = self.inlet_buffer.get('CH4', 0.0)
        N2_in = self.inlet_buffer.get('N2', 0.0)
        
        # 3. Initial guess for outlet temperature
        T_calc = self.inlet_temp_k if self.inlet_temp_k > 0 else 900.0
        
        # Maximum extent of reaction (limited by stoichiometry)
        max_xi = min(CO_in, H2O_in) if (CO_in > 0 and H2O_in > 0) else 0.0
        
        xi = 0.0
        
        # 4. Iterative solver for adiabatic equilibrium
        for iteration in range(15):
            # A. Calculate Keq(T) using empirical correlation
            # log10(K) = 2073/T - 2.029 (valid for HT-WGS, approx 200-800°C)
            Keq = 10.0**(2073.0/T_calc - 2.029)
            
            # B. Solve for extent of reaction xi
            # K = [(CO2 + xi)(H2 + xi)] / [(CO - xi)(H2O - xi)]
            # Rearranging to quadratic: a*xi^2 + b*xi + c = 0
            
            if max_xi < 1e-9:
                xi_new = 0.0
            else:
                # Use bisection for robustness
                low, high = 0.0, max_xi * 0.999
                
                for _ in range(20):  # Bisection iterations
                    mid = (low + high) / 2.0
                    
                    # Calculate reaction quotient Q at extent = mid
                    num = (CO2_in + mid) * (H2_in + mid)
                    den = (CO_in - mid) * (H2O_in - mid)
                    
                    if den < 1e-12:
                        Q = 1e10  # Effectively infinity
                    else:
                        Q = num / den
                    
                    if Q > Keq:
                        # Q > K means reaction needs to go backward (less xi)
                        high = mid
                    else:
                        # Q < K means reaction needs to go forward (more xi)
                        low = mid
                
                xi_new = (low + high) / 2.0
            
            # C. Calculate outlet composition (in kmol)
            moles_out = {
                'CO': max(0.0, CO_in - xi_new),
                'H2O': max(0.0, H2O_in - xi_new),
                'CO2': CO2_in + xi_new,
                'H2': H2_in + xi_new,
                'CH4': CH4_in,
                'N2': N2_in
            }
            
            total_moles_out = sum(moles_out.values())
            
            # D. Energy Balance: Find T that satisfies H_out = H_in (adiabatic)
            # Heat of reaction at standard conditions: ΔH_rxn ≈ -41 kJ/mol
            dH_rxn_per_mol = -41000.0  # J/mol = J/kmol * 1e-3
            
            # Heat released by reaction
            Q_rxn = dH_rxn_per_mol * xi_new * 1000.0  # Convert kmol to mol, J
            
            # Calculate total mass of outlet
            total_mass_out = sum(n * MW.get(s, 28.0) for s, n in moles_out.items())
            
            if total_mass_out < 1e-9:
                break
            
            # Estimate Cp of mixture (simplified - roughly 2 kJ/kg·K for syngas)
            Cp_mix = 2000.0  # J/kg·K
            
            # Use rigorous enthalpy calculation if LUT manager available
            if self.lut_manager and mix_thermo and self.inlet_mass_accumulated > 0:
                # Calculate mass fractions
                w_out = {s: (n * MW.get(s, 28.0)) / total_mass_out 
                         for s, n in moles_out.items() if n > 0}
                
                try:
                    # Get enthalpy at current T guess
                    h_at_T = mix_thermo.get_mixture_enthalpy(
                        w_out, self.inlet_pressure_pa, T_calc, self.lut_manager
                    )
                    
                    # Get enthalpy at T+1 for numerical Cp
                    h_at_T_plus = mix_thermo.get_mixture_enthalpy(
                        w_out, self.inlet_pressure_pa, T_calc + 1.0, self.lut_manager
                    )
                    Cp_mix = h_at_T_plus - h_at_T
                    if Cp_mix < 100:
                        Cp_mix = 2000.0  # Fallback
                except Exception:
                    pass
            
            # Temperature rise from exothermic reaction
            dT = -Q_rxn / (total_mass_out * Cp_mix)
            T_new = self.inlet_temp_k + dT
            
            # Check convergence
            if abs(T_new - T_calc) < 0.5 and abs(xi_new - xi) < 1e-6:
                T_calc = T_new
                xi = xi_new
                break
            
            # Update for next iteration (damped)
            T_calc = 0.7 * T_new + 0.3 * T_calc
            xi = xi_new

        # 5. Update final state
        self.outlet_temp_k = max(300.0, min(1500.0, T_calc))  # Clamp to reasonable range
        self.CO_conversion = xi / CO_in if CO_in > 1e-9 else 0.0
        
        # Final outlet composition (mole fractions)
        total_mol_final = sum(moles_out.values())
        if total_mol_final > 0:
            self.outlet_composition = {s: n / total_mol_final for s, n in moles_out.items()}
        else:
            self.outlet_composition = {}
        
        # Calculate outlet mass flow
        total_mass_final = sum(n * MW.get(s, 28.0) for s, n in moles_out.items())
        self.outlet_flow_kg_h = total_mass_final / self.dt if self.dt > 0 else 0.0
        
        # Update cached stream
        self._output_stream.mass_flow_kg_h = self.outlet_flow_kg_h
        self._output_stream.temperature_k = self.outlet_temp_k
        self._output_stream.pressure_pa = self.outlet_pressure_pa
        self._output_stream.composition = self.outlet_composition.copy()
        
        # === DEBUG LOGGING ===
        if int(t * 60) % 60 == 0:  # Log once per hour
            co_in_pct = (CO_in / total_moles * 100) if total_moles > 0 else 0
            co_out_pct = self.outlet_composition.get('CO', 0) * 100
            logger.info(f"WGS DEBUG [{self.component_id}] [t={t:.2f}h]:")
            logger.info(f"  Inlet: T={self.inlet_temp_k:.1f}K, CO_in={CO_in:.2f} kmol")
            logger.info(f"  Reaction: Keq={Keq:.2f}, xi={xi:.4f} kmol, conversion={self.CO_conversion*100:.1f}%")
            logger.info(f"  Outlet: T={self.outlet_temp_k:.1f}K, CO={co_out_pct:.2f}%")
        
        # 6. Clear input buffers
        self.inlet_buffer = {k: 0.0 for k in MW}
        self.inlet_enthalpy_accumulated = 0.0
        self.inlet_mass_accumulated = 0.0

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept syngas input and accumulate mass/enthalpy for the step.

        Args:
            port_name (str): Input port name.
            value (Any): Stream object or scalar flow.
            resource_type (str): Resource type hint.

        Returns:
            float: Accepted flow rate (kg/h).
        """
        if port_name in ['syngas_in', 'in', 'cooled_gas_in']:
            if isinstance(value, Stream):
                mass_in = value.mass_flow_kg_h * self.dt
                if mass_in <= 0:
                    return 0.0
                
                # Track inlet conditions
                self.inlet_temp_k = value.temperature_k
                self.inlet_pressure_pa = value.pressure_pa
                
                # Accumulate enthalpy
                h_spec = getattr(value, 'specific_enthalpy_j_kg', 0.0) or 0.0
                if h_spec == 0.0 and self.lut_manager and mix_thermo:
                    try:
                        # Calculate from composition
                        comp = value.composition or {}
                        total_mass = sum(comp.get(s, 0) * MW.get(s, 28.0) for s in comp)
                        if total_mass > 0:
                            w_mass = {s: (y * MW.get(s, 28.0)) / total_mass 
                                      for s, y in comp.items()}
                            h_spec = mix_thermo.get_mixture_enthalpy(
                                w_mass, value.pressure_pa, value.temperature_k, 
                                self.lut_manager
                            )
                    except Exception:
                        pass
                
                self.inlet_enthalpy_accumulated += h_spec * mass_in
                self.inlet_mass_accumulated += mass_in
                
                # Accumulate moles by species
                comp = value.composition or {}
                # Composition is typically mole fractions from upstream
                total_moles_in = 0.0
                for species, mole_frac in comp.items():
                    if species in MW and mole_frac > 0:
                        # Convert mass to moles: n = m * y / MW
                        # where y is mole fraction and we assume ideal mixing
                        # Actually: mass_i = mass_total * w_i = mass_total * (y_i * MW_i / MW_avg)
                        pass
                
                # Simpler approach: use mole fractions directly with total moles estimate
                # MW_avg = sum(y * MW) 
                MW_avg = sum(comp.get(s, 0) * MW.get(s, 28.0) for s in MW)
                if MW_avg > 0:
                    total_moles_in = mass_in / MW_avg
                    for species in MW:
                        y_i = comp.get(species, 0.0)
                        self.inlet_buffer[species] += y_i * total_moles_in
                
                return value.mass_flow_kg_h
            
            elif isinstance(value, (int, float)):
                # Legacy scalar support
                return float(value)
        
        return 0.0

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.

        Args:
            port_name (str): Output port name.

        Returns:
            Stream: Shifted gas output stream.
        """
        if port_name in ['syngas_out', 'shifted_gas_out', 'out']:
            return self._output_stream
        elif port_name == 'heat_out':
            return 0.0  # Adiabatic reactor - no external heat exchange
        else:
            raise ValueError(f"Unknown output port '{port_name}' on {self.component_id}")

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the component's current operational state.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'syngas_output_kg_h': self.outlet_flow_kg_h,
            'outlet_temp_k': self.outlet_temp_k,
            'CO_conversion': self.CO_conversion,
            'outlet_composition': self.outlet_composition
        }

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """
        Define the physical connection ports for this component.
        """
        return {
            'syngas_in': {'type': 'input', 'resource_type': 'syngas', 'units': 'kg/h'},
            'cooled_gas_in': {'type': 'input', 'resource_type': 'syngas', 'units': 'kg/h'},
            'in': {'type': 'input', 'resource_type': 'syngas', 'units': 'kg/h'},
            'shifted_gas_out': {'type': 'output', 'resource_type': 'syngas', 'units': 'kg/h'},
            'syngas_out': {'type': 'output', 'resource_type': 'syngas', 'units': 'kg/h'},
            'out': {'type': 'output', 'resource_type': 'syngas', 'units': 'kg/h'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }
