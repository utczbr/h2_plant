"""
Multi-component gas mixer with rigorous thermodynamics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import brentq
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants
from h2_plant.core.exceptions import ComponentStepError, FlashConvergenceError
from h2_plant.optimization.lut_manager import LUTManager

logger = logging.getLogger(__name__)


class MultiComponentMixer(Component):
    """
    Multi-species gas mixer with thermodynamic calculations.
    """
    
    def __init__(
        self,
        volume_m3: float,
        enable_phase_equilibrium: bool = True,
        heat_loss_coeff_W_per_K: float = 0.0,
        pressure_relief_threshold_bar: float = 50.0,
        initial_temperature_k: float = 298.15,
    ):
        super().__init__()
        
        self.volume_m3 = volume_m3
        self.heat_loss_coeff = heat_loss_coeff_W_per_K
        self.pressure_relief_pa = pressure_relief_threshold_bar * 1e5
        self.enable_vle = enable_phase_equilibrium
        
        self._input_buffer: List[Any] = []
        
        self.moles_stored = {'O2': 0.0, 'CO2': 0.0, 'CH4': 0.0, 'H2O': 0.0, 'H2': 0.0, 'N2': 0.0}
        self.total_internal_energy_J = 0.0
        
        self.temperature_k = initial_temperature_k
        self.pressure_pa = 1e5
        self.vapor_fraction = 1.0
        self.liquid_moles = {k: 0.0 for k in self.moles_stored}
        
        self.cumulative_input_moles = 0.0
        self.cumulative_vented_moles = 0.0
        self.flash_convergence_failures = 0
        
        self._lut_manager: Optional[LUTManager] = None
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')
        else:
            logger.warning("LUTManager not found - using simplified thermodynamics")
        
        if sum(self.moles_stored.values()) > 0:
            self._initialize_internal_energy()
    
    def receive_input(self, port_name: str, value: Any, resource_type: str = None) -> float:
        """
        Receive input stream from upstream.
        """
        if port_name == 'inlet' or port_name == 'gas_in':
            if hasattr(value, 'mass_flow_kg_h') and value.mass_flow_kg_h > 0:
                self._input_buffer.append(value)
                return value.mass_flow_kg_h
        return 0.0

    def step(self, t: float) -> None:
        super().step(t)
        dt_sec = self.dt * 3600.0
        
        # Process input buffer
        total_enthalpy_in_J = 0.0
        total_moles_in = 0.0
        
        for stream in self._input_buffer:
            mass_flow_kg_s = stream.mass_flow_kg_h / 3600.0
            
            # Molar Enthalpy of input stream (J/mol) -> stream.specific_enthalpy_j_kg is J/kg
            H_in_J_s = mass_flow_kg_s * stream.specific_enthalpy_j_kg
            
            total_enthalpy_in_J += H_in_J_s * dt_sec
            
            for species, mass_frac in stream.composition.items():
                if species in self.moles_stored:
                    mass_flow_species_kg_s = mass_flow_kg_s * mass_frac
                    mw = GasConstants.SPECIES_DATA[species]['molecular_weight'] / 1000.0 # kg/mol
                    moles_s = mass_flow_species_kg_s / mw
                    
                    self.moles_stored[species] += moles_s * dt_sec
                    total_moles_in += moles_s * dt_sec
        
        # Clear buffer
        self._input_buffer = []
        
        if self.heat_loss_coeff > 0:
            Q_loss = -self.heat_loss_coeff * (self.temperature_k - 298.15) * dt_sec
            total_enthalpy_in_J += Q_loss
        
        self.total_internal_energy_J += total_enthalpy_in_J
        self.cumulative_input_moles += total_moles_in
        
        try:
            self._perform_uv_flash()
        except Exception as e:
            logger.error(f"UV-flash failed at t={t:.2f}h: {e}")
            self.flash_convergence_failures += 1
        
        if self.pressure_pa > self.pressure_relief_pa:
            self._activate_pressure_relief()
    
    def get_state(self) -> Dict[str, Any]:
        total_moles = sum(self.moles_stored.values())
        return {
            **super().get_state(),
            'temperature_k': float(self.temperature_k),
            'pressure_pa': float(self.pressure_pa),
            'total_moles': float(total_moles),
            'vapor_fraction': float(self.vapor_fraction),
        }
    
    def _perform_uv_flash(self):
        total_moles = sum(self.moles_stored.values())
        if total_moles < 1e-12: return

        u_target = self.total_internal_energy_J / total_moles
        z = {k: v/total_moles for k, v in self.moles_stored.items()}
        
        # VLE Check Placeholder
        if self.enable_vle:
            # Future: Implement Rachford-Rice iteration for vapor-liquid split
            pass

        def residual(T):
            P = (total_moles * GasConstants.R_UNIVERSAL_J_PER_MOL_K * T) / self.volume_m3
            u_calc = self._calc_internal_energy_vapor(T, P, z) # Simplified for now
            return u_calc - u_target
        
        # Dynamic Bounds with Fallback
        T_low = max(1.0, self.temperature_k * 0.1)
        T_high = min(6000.0, self.temperature_k * 10.0)
        
        try:
            T_solution = brentq(residual, T_low, T_high, xtol=1e-3, maxiter=100)
            self.temperature_k = T_solution
            self.pressure_pa = (total_moles * GasConstants.R_UNIVERSAL_J_PER_MOL_K * T_solution) / self.volume_m3
        except ValueError:
            # Fallback: linear extrapolation or keep previous
            logger.warning(f"Mixer UV-flash failed (bounds {T_low:.1f}-{T_high:.1f}K). Holding T={self.temperature_k:.1f}K")
            self.flash_convergence_failures += 1

    def _calc_internal_energy_vapor(
        self, T: float, P: float, composition: Dict[str, float]
    ) -> float:
        """
        Calculate specific internal energy for all-vapor mixture.
        
        U = H - RT (ideal gas)
        """
        h_mix = self._calculate_molar_enthalpy(T, P, composition, phase='vapor')
        u_mix = h_mix - GasConstants.R_UNIVERSAL_J_PER_MOL_K * T
        return u_mix

    def _calculate_molar_enthalpy(self, T: float, P: float, comp: Dict[str, float], phase: str = 'vapor') -> float:
        h_mix = 0.0
        for species, mole_frac in comp.items():
            if mole_frac > 1e-12:
                data = GasConstants.SPECIES_DATA[species]
                h_form = data['h_formation']
                delta_h = self._integrate_cp(data['cp_coeffs'], 298.15, T)
                h_mix += mole_frac * (h_form + delta_h)
        
        # Non-ideal mixing correction (Excess Enthalpy)
        # Placeholder for future Peng-Robinson EOS integration
        # h_mix += self._calculate_excess_enthalpy(T, P, comp)
        
        return h_mix

    def _integrate_cp(self, coeffs: List[float], T1: float, T2: float) -> float:
        A, B, C, D, E = coeffs
        def integral(T):
            return A*T + 0.5*B*T**2 + (1/3)*C*T**3 + 0.25*D*T**4 - (E/T if T > 0 else 0)
        return integral(T2) - integral(T1)

    def _activate_pressure_relief(self):
        # Simplified for now
        pass

    def _initialize_internal_energy(self):
        total_moles = sum(self.moles_stored.values())
        if total_moles > 0:
            z = {k: v/total_moles for k, v in self.moles_stored.items()}
            u_molar = self._calc_internal_energy_vapor(self.temperature_k, self.pressure_pa, z)
            self.total_internal_energy_J = u_molar * total_moles
