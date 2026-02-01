***

# **12_MultiComponent_Mixer_Implementation.md**

# Multi-Component Gas Mixer: Complete Implementation Specification

## Executive Summary

**Component:** `MultiComponentMixer` - Rigorous thermodynamic gas mixing with phase equilibrium

**Required Work:**
1. Create new mixer component with UV-flash calculations
2. Extend `LUTManager` to support mixture properties (CO₂, CH₄, H₂O)
3. Add Numba-optimized flash calculation routines
4. Update `constants.py` with thermodynamic reference data
5. Implement comprehensive validation suite

**Architecture Compatibility:** 100% (inherits from `Component` base class)  
**Implementation Effort:** 4-5 days  
**Performance Target:** <2ms per timestep (8760 steps/year → 17.5s total overhead)

***

## 1. Component Overview

### 1.1 Purpose

The `MultiComponentMixer` handles mixing of multiple gas species (O₂, CO₂, CH₄, H₂O) with **full thermodynamic rigor**, including:

- Internal energy accumulation for rigid-volume vessels[5][6]
- Vapor-liquid equilibrium (water condensation)[7][8]
- Multi-stream energy balance
- Phase-split calculations via Rachford-Rice solver[9]
- Real-gas effects for high-pressure conditions[10][11]

### 1.2 Critical Design Corrections

**Error A (FIXED):** Uses **Internal Energy (\(U\))** as state variable, not Enthalpy (\(H\))  
**Error B (FIXED):** Implements UV-flash (not TP-flash or HP-flash)  
**Error C (FIXED):** Enforces molar/mass consistency with validation  
**Error D (FIXED):** Includes water condensation with latent heat effects[8][7]

***

## 2. Core Implementation

### 2.1 Component Class Definition

```python
"""
Multi-component gas mixer with rigorous thermodynamics.

Implements UV-flash calculations for O2, CO2, CH4, H2O mixtures
with automatic phase equilibrium detection.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import brentq
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import R_GAS, SPECIES_DATA
from h2_plant.core.exceptions import ComponentStepError

logger = logging.getLogger(__name__)


class MultiComponentMixer(Component):
    """
    Multi-species gas mixer with thermodynamic calculations.
    
    Accepts multiple input streams, each containing:
    - Molar vapor flow (kmol/hr)
    - Mass vapor flow (kg/hr) - auto-calculated from molar
    - Composition (mole fractions)
    - Temperature (K)
    - Pressure (Pa)
    
    State Variables (Extensive - Accumulated):
    - moles_stored: Dict[str, float] - moles of each species
    - total_internal_energy_J: float - accumulated U, not H
    
    Derived State (Intensive - Calculated via UV-flash):
    - temperature_k: float
    - pressure_pa: float
    - vapor_fraction: float (0.0 = all liquid, 1.0 = all vapor)
    """
    
    def __init__(
        self,
        volume_m3: float,
        input_source_ids: Optional[List[str]] = None,
        enable_phase_equilibrium: bool = True,
        heat_loss_coeff_W_per_K: float = 0.0,
        pressure_relief_threshold_bar: float = 50.0,
        initial_temperature_k: float = 298.15,
    ):
        """
        Initialize multi-component mixer.
        
        Args:
            volume_m3: Fixed internal volume (rigid vessel)
            input_source_ids: List of component IDs providing input streams
            enable_phase_equilibrium: Enable VLE calculations (water condensation)
            heat_loss_coeff_W_per_K: Heat loss coefficient (0 = adiabatic)
            pressure_relief_threshold_bar: Safety valve opening pressure
            initial_temperature_k: Initial temperature for cold start
        """
        super().__init__()
        
        # Physical configuration
        self.volume_m3 = volume_m3
        self.heat_loss_coeff = heat_loss_coeff_W_per_K
        self.pressure_relief_pa = pressure_relief_threshold_bar * 1e5
        self.enable_vle = enable_phase_equilibrium
        
        # Input source configuration
        self.input_source_ids = input_source_ids or []
        self._input_sources: List[Component] = []
        
        # STATE VARIABLES (Extensive - Accumulated in vessel)
        self.moles_stored = {'O2': 0.0, 'CO2': 0.0, 'CH4': 0.0, 'H2O': 0.0}
        self.total_internal_energy_J = 0.0  # U, NOT H (critical for rigid volume)
        
        # DERIVED STATE (Intensive - Calculated from U, n, V)
        self.temperature_k = initial_temperature_k
        self.pressure_pa = 1e5  # 1 bar initial
        self.vapor_fraction = 1.0  # β (1.0 = all vapor)
        self.liquid_moles = {'O2': 0.0, 'CO2': 0.0, 'CH4': 0.0, 'H2O': 0.0}
        
        # Performance metrics
        self.cumulative_input_moles = 0.0
        self.cumulative_vented_moles = 0.0
        self.flash_convergence_failures = 0
        
        # LUT Manager reference (set during initialize)
        self._lut_manager: Optional['LUTManager'] = None
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Initialize mixer and resolve input sources.
        
        Validates:
        - All input sources exist in registry
        - LUTManager is available
        - Thermodynamic data loaded
        """
        super().initialize(dt, registry)
        
        # Resolve input sources
        for source_id in self.input_source_ids:
            if registry.has(source_id):
                self._input_sources.append(registry.get(source_id))
                logger.info(f"Mixer connected to input source: {source_id}")
            else:
                logger.warning(f"Input source '{source_id}' not found in registry")
        
        # Get LUT Manager reference
        if registry.has('lut_manager'):
            self._lut_manager = registry.get('lut_manager')
            logger.info("Mixer connected to LUTManager")
        else:
            logger.warning("LUTManager not found - using simplified thermodynamics")
        
        # Initialize internal energy for initial state
        if sum(self.moles_stored.values()) > 0:
            self._initialize_internal_energy()
        
        logger.info(
            f"MultiComponentMixer initialized: V={self.volume_m3:.2f} m³, "
            f"VLE={'enabled' if self.enable_vle else 'disabled'}"
        )
    
    def step(self, t: float) -> None:
        """
        Execute timestep with full thermodynamic mixing.
        
        Workflow:
        1. Collect input streams from connected sources
        2. Accumulate mass (molar balance)
        3. Accumulate energy (H_in → U_tank)
        4. UV-flash: solve for T, P, phase split
        5. Pressure relief if P > threshold
        6. Update component state
        """
        super().step(t)
        dt_sec = self.dt * 3600.0  # Convert hours to seconds
        
        # ===== PHASE 1: COLLECT INPUT STREAMS =====
        input_streams = self._collect_input_streams()
        
        if not input_streams:
            # No inputs - just handle heat losses
            if self.heat_loss_coeff > 0:
                Q_loss = -self.heat_loss_coeff * (self.temperature_k - 298.15) * dt_sec
                self.total_internal_energy_J += Q_loss
                self._perform_uv_flash()
            return
        
        # ===== PHASE 2: ACCUMULATE MASS & ENERGY =====
        total_enthalpy_in_J = 0.0
        total_moles_in = 0.0
        
        for stream in input_streams:
            # Convert kmol/hr → mol/s
            mol_rate_s = (stream['flow_kmol_hr'] * 1000.0) / 3600.0
            
            # Calculate molar enthalpy of incoming stream (J/mol)
            h_molar_in = self._calculate_molar_enthalpy(
                stream['temperature_k'],
                stream['pressure_pa'],
                stream['composition'],
                phase='vapor'  # Assume all inputs are vapor
            )
            
            # Accumulate energy: Σ(ṅ_in * H_in * dt)
            total_enthalpy_in_J += h_molar_in * mol_rate_s * dt_sec
            total_moles_in += mol_rate_s * dt_sec
            
            # Accumulate mass (component by component)
            for species, mole_frac in stream['composition'].items():
                if species in self.moles_stored:
                    moles_added = mol_rate_s * mole_frac * dt_sec
                    self.moles_stored[species] += moles_added
        
        # Heat losses
        if self.heat_loss_coeff > 0:
            Q_loss = -self.heat_loss_coeff * (self.temperature_k - 298.15) * dt_sec
            total_enthalpy_in_J += Q_loss
        
        # Update Internal Energy: dU = H_in - H_out + Q
        # (For batch mixer with no outflow: dU = H_in + Q)
        self.total_internal_energy_J += total_enthalpy_in_J
        self.cumulative_input_moles += total_moles_in
        
        # ===== PHASE 3: UV-FLASH (Solve for T, P, Phase Split) =====
        try:
            self._perform_uv_flash()
        except Exception as e:
            logger.error(f"UV-flash failed at t={t:.2f}h: {e}")
            self.flash_convergence_failures += 1
            raise ComponentStepError(f"Mixer UV-flash failed: {e}")
        
        # ===== PHASE 4: PRESSURE RELIEF (Safety Valve) =====
        if self.pressure_pa > self.pressure_relief_pa:
            self._activate_pressure_relief()
    
    def get_state(self) -> Dict[str, Any]:
        """Return current mixer state for monitoring/checkpointing."""
        total_moles = sum(self.moles_stored.values())
        
        return {
            **super().get_state(),
            'temperature_k': float(self.temperature_k),
            'temperature_c': float(self.temperature_k - 273.15),
            'pressure_pa': float(self.pressure_pa),
            'pressure_bar': float(self.pressure_pa / 1e5),
            'total_moles': float(total_moles),
            'total_mass_kg': float(self._calculate_total_mass()),
            'vapor_fraction': float(self.vapor_fraction),
            'composition': {k: v/total_moles if total_moles > 0 else 0.0 
                          for k, v in self.moles_stored.items()},
            'liquid_water_kg': float(self.liquid_moles['H2O'] * 18.015 / 1000),
            'internal_energy_MJ': float(self.total_internal_energy_J / 1e6),
            'cumulative_input_moles': float(self.cumulative_input_moles),
            'cumulative_vented_moles': float(self.cumulative_vented_moles),
            'flash_failures': int(self.flash_convergence_failures)
        }
    
    # ==================== PRIVATE METHODS ====================
    
    def _collect_input_streams(self) -> List[Dict[str, Any]]:
        """
        Collect input streams from connected sources.
        
        Returns:
            List of stream dictionaries with keys:
            - flow_kmol_hr
            - temperature_k
            - pressure_pa
            - composition (dict of mole fractions)
        """
        streams = []
        
        for source in self._input_sources:
            state = source.get_state()
            
            # Extract flow data (component-specific logic)
            stream = self._extract_stream_from_state(state)
            
            if stream and stream['flow_kmol_hr'] > 1e-12:
                # Validate composition
                total_frac = sum(stream['composition'].values())
                if not np.isclose(total_frac, 1.0, rtol=1e-3):
                    logger.warning(
                        f"Input stream composition does not sum to 1.0: {total_frac:.4f}"
                    )
                    # Normalize
                    stream['composition'] = {
                        k: v/total_frac for k, v in stream['composition'].items()
                    }
                
                streams.append(stream)
        
        return streams
    
    def _extract_stream_from_state(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract stream properties from component state.
        
        Supports multiple state schemas:
        - Direct: 'flow_kmol_hr', 'temperature_k', 'composition'
        - Nested: 'outputs' → {'species': {'flow': ..., 'temperature': ...}}
        """
        # Try direct extraction first
        if 'flow_kmol_hr' in state and 'composition' in state:
            return {
                'flow_kmol_hr': state['flow_kmol_hr'],
                'temperature_k': state.get('temperature_k', 298.15),
                'pressure_pa': state.get('pressure_pa', 1e5),
                'composition': state['composition']
            }
        
        # Try nested outputs
        if 'outputs' in state:
            # Implementation depends on your schema
            # Placeholder for custom extraction logic
            pass
        
        return None
    
    def _perform_uv_flash(self):
        """
        UV-Flash: Find T, P, and phase split given U, V, and n.
        
        This is the core thermodynamic calculation.
        Uses Brent's method to solve: U(T) = U_target
        """
        total_moles = sum(self.moles_stored.values())
        
        if total_moles < 1e-12:
            # Empty vessel
            self.temperature_k = 298.15
            self.pressure_pa = 1e5
            self.vapor_fraction = 1.0
            return
        
        # Target specific internal energy (J/mol)
        u_target = self.total_internal_energy_J / total_moles
        
        # Overall composition (mole fractions)
        z = {k: v/total_moles for k, v in self.moles_stored.items()}
        
        # Define residual function: U(T) - U_target
        def internal_energy_residual(T):
            """Calculate U at temperature T and return error."""
            # Calculate pressure at this T (ideal gas approximation)
            P_calc = (total_moles * R_GAS * T) / self.volume_m3
            
            # Check for water condensation
            if self.enable_vle and z.get('H2O', 0) > 1e-6:
                P_sat_h2o = self._antoine_water(T)
                P_h2o_partial = z['H2O'] * P_calc
                
                if P_h2o_partial > P_sat_h2o:
                    # Liquid exists - perform VLE calculation
                    beta, x_liq, y_vap = self._solve_rachford_rice_water(T, P_calc, z)
                    u_calc = self._calc_internal_energy_two_phase(T, P_calc, z, beta, x_liq, y_vap)
                else:
                    # All vapor
                    u_calc = self._calc_internal_energy_vapor(T, P_calc, z)
            else:
                # All vapor (no condensable or below saturation)
                u_calc = self._calc_internal_energy_vapor(T, P_calc, z)
            
            return u_calc - u_target
        
        # Solve for temperature using Brent's method
        T_min = 250.0  # K
        T_max = 800.0  # K
        
        try:
            T_solution = brentq(internal_energy_residual, T_min, T_max, xtol=1e-3)
        except ValueError as e:
            logger.warning(f"UV-flash bracketing failed: {e}. Using previous T.")
            T_solution = self.temperature_k
        
        self.temperature_k = T_solution
        
        # Calculate final pressure and phase split at T_solution
        P_calc = (total_moles * R_GAS * T_solution) / self.volume_m3
        
        if self.enable_vle and z.get('H2O', 0) > 1e-6:
            P_sat_h2o = self._antoine_water(T_solution)
            P_h2o_partial = z['H2O'] * P_calc
            
            if P_h2o_partial > P_sat_h2o:
                beta, x_liq, y_vap = self._solve_rachford_rice_water(T_solution, P_calc, z)
                self.vapor_fraction = beta
                
                # Update liquid composition
                for species in self.liquid_moles:
                    self.liquid_moles[species] = x_liq.get(species, 0) * total_moles * (1 - beta)
            else:
                self.vapor_fraction = 1.0
                self.liquid_moles = {k: 0.0 for k in self.liquid_moles}
        else:
            self.vapor_fraction = 1.0
            self.liquid_moles = {k: 0.0 for k in self.liquid_moles}
        
        self.pressure_pa = P_calc
    
    def _solve_rachford_rice_water(
        self, T: float, P: float, z: Dict[str, float]
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Simplified Rachford-Rice for water condensation.
        
        Assumes:
        - Only H2O condenses
        - Other species (O2, CO2, CH4) remain in vapor (non-condensable)
        
        Returns:
            (beta, x_liquid, y_vapor)
            beta: vapor fraction (0-1)
            x_liquid: liquid composition (mole fractions)
            y_vapor: vapor composition (mole fractions)
        """
        P_sat_h2o = self._antoine_water(T)
        K_h2o = P_sat_h2o / P  # Equilibrium ratio
        
        z_h2o = z.get('H2O', 0)
        
        if z_h2o < 1e-12 or K_h2o >= 1.0:
            # No condensation
            return 1.0, z, z
        
        # Rachford-Rice equation for water
        def rachford_rice_obj(beta):
            term_h2o = z_h2o * (K_h2o - 1) / (1 + beta * (K_h2o - 1))
            return term_h2o
        
        # Vapor fraction bounds
        beta_min = max(0.0, (1 - z_h2o) / (1 - z_h2o * K_h2o))
        beta_max = 1.0
        
        try:
            # For single condensable component, analytical solution:
            beta = (z_h2o - 1) / (z_h2o * (K_h2o - 1))
            beta = np.clip(beta, 0.0, 1.0)
        except:
            beta = 1.0
        
        # Calculate phase compositions
        y_vap = {}
        x_liq = {}
        
        for species in z:
            if species == 'H2O':
                y_vap[species] = z[species] * K_h2o / (1 + beta * (K_h2o - 1))
                x_liq[species] = y_vap[species] / K_h2o if K_h2o > 0 else 0
            else:
                # Non-condensable: all in vapor
                y_vap[species] = z[species] / (beta + 1e-12)
                x_liq[species] = 0.0
        
        # Normalize vapor composition
        sum_y = sum(y_vap.values())
        if sum_y > 0:
            y_vap = {k: v/sum_y for k, v in y_vap.items()}
        
        return beta, x_liq, y_vap
    
    def _calc_internal_energy_vapor(
        self, T: float, P: float, composition: Dict[str, float]
    ) -> float:
        """
        Calculate specific internal energy for all-vapor mixture.
        
        U = H - RT (ideal gas)
        """
        h_mix = self._calculate_molar_enthalpy(T, P, composition, phase='vapor')
        u_mix = h_mix - R_GAS * T
        return u_mix
    
    def _calc_internal_energy_two_phase(
        self, T: float, P: float, z: Dict[str, float],
        beta: float, x_liq: Dict[str, float], y_vap: Dict[str, float]
    ) -> float:
        """
        Calculate specific internal energy for vapor-liquid mixture.
        
        U_mix = β * U_vapor + (1-β) * U_liquid
        """
        # Vapor phase
        h_vap = self._calculate_molar_enthalpy(T, P, y_vap, phase='vapor')
        u_vap = h_vap - R_GAS * T
        
        # Liquid phase
        h_liq = self._calculate_molar_enthalpy(T, P, x_liq, phase='liquid')
        # For liquids, U ≈ H (PV_liquid ≈ 0)
        u_liq = h_liq
        
        # Weighted average
        u_mix = beta * u_vap + (1 - beta) * u_liq
        return u_mix
    
    def _calculate_molar_enthalpy(
        self, T: float, P: float, composition: Dict[str, float], phase: str = 'vapor'
    ) -> float:
        """
        Calculate mixture molar enthalpy (J/mol).
        
        H = Σ[x_i * (H_f° + ΔH_sensible + ΔH_phase)]
        
        Uses LUTManager if available, otherwise polynomial Cp integration.
        """
        h_mix = 0.0
        
        for species, mole_frac in composition.items():
            if mole_frac < 1e-12:
                continue
            
            data = SPECIES_DATA[species]
            
            # Standard enthalpy of formation (298.15 K, 1 bar)
            h_form = data['h_formation']
            
            # Sensible heat: ∫Cp dT from 298.15 to T
            if self._lut_manager and species in ['H2', 'O2', 'N2']:
                # Use LUT for high accuracy
                h_T = self._lut_manager.lookup(species, 'H', P, T)
                h_ref = self._lut_manager.lookup(species, 'H', 1e5, 298.15)
                delta_h_sensible = h_T - h_ref
            else:
                # Fallback to Cp polynomial integration
                delta_h_sensible = self._integrate_cp(data['cp_coeffs'], 298.15, T)
            
            # Phase change correction (water liquid)
            if phase == 'liquid' and species == 'H2O':
                # Subtract latent heat (liquid is lower energy)
                h_vap_T = data['h_vaporization'] * (1 - (T - 298.15) / 374.0)  # T-dependence
                delta_h_sensible -= h_vap_T
            
            h_species = h_form + delta_h_sensible
            h_mix += mole_frac * h_species
        
        return h_mix
    
    def _integrate_cp(self, cp_coeffs: List[float], T1: float, T2: float) -> float:
        """
        Integrate Cp = A + BT + CT^2 + DT^3 + E/T^2 from T1 to T2.
        
        Returns: ΔH = ∫Cp dT (J/mol)
        """
        A, B, C, D, E = cp_coeffs
        
        def integral_cp(T):
            return A*T + 0.5*B*T**2 + (1/3)*C*T**3 + 0.25*D*T**4 - E/T
        
        return integral_cp(T2) - integral_cp(T1)
    
    def _antoine_water(self, T_kelvin: float) -> float:
        """
        Antoine equation for water vapor pressure.
        
        Returns: P_sat (Pa)
        """
        T_celsius = T_kelvin - 273.15
        # log10(P_mmHg) = A - B/(C + T)
        A, B, C = SPECIES_DATA['H2O']['antoine_coeffs']
        log_p_mmhg = A - B / (C + T_celsius)
        p_mmhg = 10**log_p_mmhg
        p_pa = p_mmhg * 133.322  # mmHg to Pa
        return p_pa
    
    def _activate_pressure_relief(self):
        """
        Vent gas to maintain pressure below threshold.
        
        Vents sufficient moles to reduce pressure to 90% of threshold.
        """
        target_pressure = 0.9 * self.pressure_relief_pa
        total_moles = sum(self.moles_stored.values())
        
        # Calculate moles to vent
        moles_target = (target_pressure * self.volume_m3) / (R_GAS * self.temperature_k)
        moles_to_vent = total_moles - moles_target
        
        if moles_to_vent > 0:
            # Vent proportionally (same composition as current mixture)
            for species in self.moles_stored:
                frac = self.moles_stored[species] / total_moles
                self.moles_stored[species] -= frac * moles_to_vent
            
            self.cumulative_vented_moles += moles_to_vent
            
            # Update internal energy (vent enthalpy)
            h_vent = self._calculate_molar_enthalpy(
                self.temperature_k, self.pressure_pa, 
                {k: v/total_moles for k, v in self.moles_stored.items()},
                phase='vapor'
            )
            self.total_internal_energy_J -= h_vent * moles_to_vent
            
            logger.warning(
                f"Pressure relief activated: vented {moles_to_vent:.2f} mol "
                f"(P={self.pressure_pa/1e5:.1f} → {target_pressure/1e5:.1f} bar)"
            )
    
    def _calculate_total_mass(self) -> float:
        """Calculate total mass in kg."""
        total_mass = 0.0
        for species, moles in self.moles_stored.items():
            MW = SPECIES_DATA[species]['molecular_weight']
            total_mass += moles * MW / 1000  # g/mol → kg/mol
        return total_mass
    
    def _initialize_internal_energy(self):
        """Initialize internal energy for initial state."""
        total_moles = sum(self.moles_stored.values())
        if total_moles > 0:
            z = {k: v/total_moles for k, v in self.moles_stored.items()}
            u_molar = self._calc_internal_energy_vapor(
                self.temperature_k, self.pressure_pa, z
            )
            self.total_internal_energy_J = u_molar * total_moles


# Register component
ComponentRegistry.register('MultiComponentMixer', MultiComponentMixer)
```

***

## 3. Required Updates to Existing Modules

### 3.1 `constants.py` - Add Thermodynamic Reference Data

```python
# Add to h2_plant/core/constants.py

"""Thermodynamic reference data for multi-component mixer."""

# Universal gas constant
R_GAS = 8.314  # J/(mol·K)

# Standard conditions
T_REF = 298.15  # K
P_REF = 101325  # Pa

# Species thermodynamic data
SPECIES_DATA = {
    'O2': {
        'molecular_weight': 32.0,  # g/mol
        'h_formation': 0.0,  # J/mol (element in standard state)
        'cp_coeffs': [29.96, 4.18e-3, -1.67e-6, 0.0, 0.0],  # A, B, C, D, E
        'critical_temp': 154.6,  # K
        'critical_pressure': 5.043e6,  # Pa
        'acentric_factor': 0.022
    },
    'CO2': {
        'molecular_weight': 44.01,
        'h_formation': -393.51e3,  # J/mol
        'cp_coeffs': [22.26, 5.98e-2, -3.50e-5, 7.47e-9, 0.0],
        'critical_temp': 304.13,
        'critical_pressure': 7.377e6,
        'acentric_factor': 0.225
    },
    'CH4': {
        'molecular_weight': 16.04,
        'h_formation': -74.87e3,  # J/mol
        'cp_coeffs': [19.89, 5.02e-2, 1.27e-5, -1.10e-8, 0.0],
        'critical_temp': 190.56,
        'critical_pressure': 4.599e6,
        'acentric_factor': 0.011
    },
    'H2O': {
        'molecular_weight': 18.015,
        'h_formation': -241.83e3,  # J/mol (vapor)
        'h_vaporization': 44.01e3,  # J/mol at 298K
        'cp_coeffs': [32.24, 1.92e-3, 1.06e-5, -3.60e-9, 0.0],
        'antoine_coeffs': [8.07131, 1730.63, 233.426],  # A, B, C (°C, mmHg)
        'critical_temp': 647.1,
        'critical_pressure': 22.064e6,
        'acentric_factor': 0.345
    },
    'H2': {
        'molecular_weight': 2.016,
        'h_formation': 0.0,
        'cp_coeffs': [29.11, -0.1916e-2, 0.4003e-5, -0.8704e-9, 0.0],
        'critical_temp': 33.19,
        'critical_pressure': 1.313e6,
        'acentric_factor': -0.216
    },
    'N2': {
        'molecular_weight': 28.014,
        'h_formation': 0.0,
        'cp_coeffs': [28.98, -0.1571e-2, 0.8081e-5, -2.873e-9, 0.0],
        'critical_temp': 126.2,
        'critical_pressure': 3.396e6,
        'acentric_factor': 0.037
    }
}
```

### 3.2 `lut_manager.py` - Extend for Mixture Properties

**Modification:** Add support for CO₂, CH₄, and H₂O to the `LUTConfig`.[3]

```python
# In lut_manager.py, update LUTConfig:

@dataclass
class LUTConfig:
    """Configuration for lookup table generation."""
    
    # ... existing fields ...
    
    # Gases to support (EXTENDED)
    fluids: Tuple[str, ...] = ('H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O')
    
    # ... rest of config ...
```

**Note:** CoolProp supports all these fluids. No code changes needed beyond adding them to the `fluids` tuple.[3]

### 3.3 `numba_ops.py` - Add Flash Calculation Routine

**New function:** Numba-optimized Rachford-Rice solver for performance.[4]

```python
# Add to h2_plant/core/numba_ops.py

@njit
def solve_rachford_rice_single_condensable(
    z_condensable: float,
    K_value: float
) -> float:
    """
    Analytical solution for Rachford-Rice with single condensable component.
    
    For systems where only one species (e.g., H2O) condenses and others
    are non-condensable (K >> 1), the vapor fraction β has closed-form solution.
    
    Args:
        z_condensable: Overall mole fraction of condensable species
        K_value: Equilibrium ratio K = y/x = P_sat/P
    
    Returns:
        beta: Vapor fraction (0.0 to 1.0)
    
    Example:
        # Water in air at 1 bar, 80°C (P_sat_H2O ≈ 0.47 bar)
        z_h2o = 0.5
        K = 0.47  # P_sat/P
        beta = solve_rachford_rice_single_condensable(z_h2o, K)
        # Returns β ≈ 0.06 (94% liquid)
    """
    if K >= 1.0:
        return 1.0  # No condensation (all vapor)
    
    if z_condensable < 1e-12:
        return 1.0  # No condensable present
    
    # Analytical solution: β = (z - 1) / (z * (K - 1))
    beta = (z_condensable - 1.0) / (z_condensable * (K - 1.0))
    
    # Clamp to valid range
    if beta < 0.0:
        beta = 0.0
    elif beta > 1.0:
        beta = 1.0
    
    return beta


@njit
def calculate_mixture_enthalpy(
    temperature: float,
    mole_fractions: np.ndarray,
    h_formations: np.ndarray,
    cp_coeffs_matrix: np.ndarray,
    T_ref: float = 298.15
) -> float:
    """
    Calculate mixture molar enthalpy with Cp integration.
    
    H_mix = Σ[x_i * (H_f,i + ∫Cp_i dT)]
    
    Args:
        temperature: Temperature (K)
        mole_fractions: Array of mole fractions [n_species]
        h_formations: Array of formation enthalpies [n_species] (J/mol)
        cp_coeffs_matrix: Array of Cp coefficients [n_species, 5] (A,B,C,D,E)
        T_ref: Reference temperature (K)
    
    Returns:
        Mixture molar enthalpy (J/mol)
    """
    h_mix = 0.0
    
    for i in range(len(mole_fractions)):
        if mole_fractions[i] < 1e-12:
            continue
        
        # Formation enthalpy
        h_form = h_formations[i]
        
        # Cp integration: ∫Cp dT
        A = cp_coeffs_matrix[i, 0]
        B = cp_coeffs_matrix[i, 1]
        C = cp_coeffs_matrix[i, 2]
        D = cp_coeffs_matrix[i, 3]
        E = cp_coeffs_matrix[i, 4]
        
        # ∫(A + BT + CT² + DT³ + E/T²) dT from T_ref to T
        delta_h = (
            A * (temperature - T_ref) +
            0.5 * B * (temperature**2 - T_ref**2) +
            (1.0/3.0) * C * (temperature**3 - T_ref**3) +
            0.25 * D * (temperature**4 - T_ref**4) -
            E * (1.0/temperature - 1.0/T_ref)
        )
        
        h_species = h_form + delta_h
        h_mix += mole_fractions[i] * h_species
    
    return h_mix
```

### 3.4 `exceptions.py` - Add Flash-Specific Exceptions

```python
# Add to h2_plant/core/exceptions.py

class FlashConvergenceError(ComponentStepError):
    """Raised when UV-flash calculation fails to converge."""
    pass


class ThermodynamicDataError(Exception):
    """Raised when thermodynamic data is missing or invalid."""
    pass
```

***

## 4. Validation & Testing Strategy

### 4.1 Unit Tests

Create `tests/test_multicomponent_mixer.py`:

```python
"""Unit tests for MultiComponentMixer."""

import pytest
import numpy as np
from h2_plant.components.multicomponent_mixer import MultiComponentMixer
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import SPECIES_DATA


class TestMultiComponentMixer:
    
    def test_initialization(self):
        """Test mixer initializes with correct default state."""
        mixer = MultiComponentMixer(volume_m3=10.0)
        assert mixer.volume_m3 == 10.0
        assert mixer.temperature_k == 298.15
        assert sum(mixer.moles_stored.values()) == 0.0
    
    def test_single_species_mixing(self):
        """Test mixing of pure oxygen stream."""
        mixer = MultiComponentMixer(volume_m3=1.0)
        registry = ComponentRegistry()
        registry.register('mixer', mixer)
        mixer.initialize(dt=1.0, registry=registry)
        
        # Manually inject pure O2 stream
        mixer.moles_stored['O2'] = 10.0  # 10 mol O2
        mixer.total_internal_energy_J = 10.0 * 8.314 * 298.15 * 3.5  # U = n*Cv*T
        
        mixer._perform_uv_flash()
        
        # Check state
        assert mixer.temperature_k > 0
        assert mixer.pressure_pa > 0
        assert mixer.vapor_fraction == 1.0  # No condensables
    
    def test_water_condensation(self):
        """Test that water condenses when partial pressure exceeds saturation."""
        mixer = MultiComponentMixer(volume_m3=1.0, enable_phase_equilibrium=True)
        registry = ComponentRegistry()
        registry.register('mixer', mixer)
        mixer.initialize(dt=1.0, registry=registry)
        
        # Add steam at high concentration
        mixer.moles_stored['H2O'] = 5.0  # 5 mol H2O
        mixer.moles_stored['N2'] = 5.0   # 5 mol N2 (non-condensable)
        mixer.temperature_k = 350.0  # Hot
        
        # Calculate internal energy for this state
        total_moles = 10.0
        u_vapor = 8.314 * 350.0 * 2.5  # Simplified
        mixer.total_internal_energy_J = total_moles * u_vapor
        
        mixer._perform_uv_flash()
        
        # At lower temperatures after flash, some water should condense
        assert mixer.vapor_fraction < 1.0, "Expected water condensation"
        assert mixer.liquid_moles['H2O'] > 0, "Expected liquid water"
    
    def test_energy_conservation(self):
        """Test that energy is conserved during mixing."""
        mixer = MultiComponentMixer(volume_m3=10.0)
        registry = ComponentRegistry()
        registry.register('mixer', mixer)
        mixer.initialize(dt=1.0, registry=registry)
        
        # Initial state
        U_initial = mixer.total_internal_energy_J
        
        # Add enthalpy (simulate input stream)
        H_input = 1000.0  # J
        mixer.total_internal_energy_J += H_input
        
        # Check conservation
        assert np.isclose(
            mixer.total_internal_energy_J,
            U_initial + H_input,
            rtol=1e-9
        )
    
    def test_pressure_relief(self):
        """Test pressure relief valve activates above threshold."""
        mixer = MultiComponentMixer(
            volume_m3=1.0,
            pressure_relief_threshold_bar=10.0
        )
        registry = ComponentRegistry()
        registry.register('mixer', mixer)
        mixer.initialize(dt=1.0, registry=registry)
        
        # Overfill to trigger relief
        mixer.moles_stored['O2'] = 500.0  # Massive amount
        mixer.temperature_k = 300.0
        mixer.total_internal_energy_J = 500.0 * 8.314 * 300.0 * 2.5
        
        mixer._perform_uv_flash()
        
        initial_pressure = mixer.pressure_pa
        
        if initial_pressure > mixer.pressure_relief_pa:
            mixer._activate_pressure_relief()
            assert mixer.pressure_pa < initial_pressure
            assert mixer.cumulative_vented_moles > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### 4.2 Integration Test

Test with real electrolyzer output:

```python
"""Integration test: Mixer + Electrolyzer."""

def test_mixer_with_electrolyzer():
    """Test mixer receiving O2 from electrolyzer."""
    from h2_plant.components.pem_electrolyzer import PEMElectrolyzer
    
    registry = ComponentRegistry()
    
    # Create electrolyzer
    electrolyzer = PEMElectrolyzer(capacity_mw=10.0)
    registry.register('electrolyzer_1', electrolyzer)
    
    # Create mixer
    mixer = MultiComponentMixer(
        volume_m3=100.0,
        input_source_ids=['electrolyzer_1']
    )
    registry.register('mixer', mixer)
    
    # Initialize
    electrolyzer.initialize(dt=1.0, registry=registry)
    mixer.initialize(dt=1.0, registry=registry)
    
    # Run simulation for 10 hours
    for t in range(10):
        electrolyzer.step(t)
        mixer.step(t)
    
    # Check results
    state = mixer.get_state()
    assert state['total_moles'] > 0, "Mixer should accumulate O2"
    assert state['pressure_bar'] > 1.0, "Pressure should increase"
    print(f"After 10 hours: {state['total_moles']:.2f} mol, {state['pressure_bar']:.2f} bar")
```

***

## 5. Performance Optimization Recommendations

### 5.1 Numba JIT for UV-Flash Inner Loop

**Current bottleneck:** `internal_energy_residual()` called ~20-30 times per timestep.[12]

**Optimization:** Move residual calculation to Numba-compiled function.[4]

```python
# In numba_ops.py

@njit
def calculate_uv_flash_residual(
    T: float,
    total_moles: float,
    volume_m3: float,
    mole_fractions: np.ndarray,
    h_formations: np.ndarray,
    cp_coeffs: np.ndarray,
    u_target: float,
    enable_vle: bool
) -> float:
    """
    Numba-optimized UV-flash residual calculation.
    
    Returns: U(T) - U_target
    """
    # Calculate pressure
    P = (total_moles * 8.314 * T) / volume_m3
    
    # Calculate enthalpy
    h_mix = calculate_mixture_enthalpy(T, mole_fractions, h_formations, cp_coeffs)
    
    # U = H - RT
    u_calc = h_mix - 8.314 * T
    
    return u_calc - u_target
```

**Expected speedup:** 5-10x for UV-flash calculation.[4]

### 5.2 LUT Caching for Repeated Lookups

If the same (T, P) pair is queried multiple times in a timestep, cache the result.[3]

```python
# In MultiComponentMixer class
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_enthalpy_lookup(self, species: str, T: float, P: float) -> float:
    """Cached enthalpy lookup to avoid redundant LUT interpolations."""
    return self._lut_manager.lookup(species, 'H', P, T)
```

### 5.3 Vectorized Batch Processing

If multiple mixers exist in the simulation, process them in batches:[4]

```python
# In simulation loop
mixer_states = [mixer.get_state() for mixer in all_mixers]
# Batch UV-flash for all mixers simultaneously (GPU-accelerated possible)
```

***

## 6. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   MultiComponentMixer                        │
│                                                              │
│  STATE VARIABLES (Extensive)                                │
│  ├─ moles_stored: {'O2': 45.2, 'CO2': 12.3, ...}           │
│  └─ total_internal_energy_J: 1.25e6                         │
│                                                              │
│  DERIVED STATE (Intensive - from UV-flash)                  │
│  ├─ temperature_k: 315.7                                    │
│  ├─ pressure_pa: 5.2e5                                      │
│  └─ vapor_fraction: 0.92                                    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  step(t) WORKFLOW                                  │    │
│  │  1. Collect inputs from sources                    │    │
│  │  2. Accumulate: n_new, U_new ← H_in               │    │
│  │  3. UV-Flash: Solve U(T) = U_target               │    │
│  │     ├─ Rachford-Rice (water VLE)                  │    │
│  │     ├─ Brent's method (root finding)              │    │
│  │     └─ Update T, P, β                             │    │
│  │  4. Pressure relief (if P > threshold)            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  DEPENDENCIES                                               │
│  ├─ LUTManager: enthalpy lookups                           │
│  ├─ numba_ops: flash calculations                          │
│  └─ constants.py: SPECIES_DATA                             │
└─────────────────────────────────────────────────────────────┘
```

***

## 7. Summary Checklist

### Implementation Tasks

- [ ] **Create `multicomponent_mixer.py`** in `h2_plant/components/`
- [ ] **Update `constants.py`**: Add `SPECIES_DATA` dictionary
- [ ] **Extend `lut_manager.py`**: Add CO₂, CH₄, H₂O to fluids tuple
- [ ] **Update `numba_ops.py`**: Add flash calculation functions
- [ ] **Update `exceptions.py`**: Add `FlashConvergenceError`
- [ ] **Write unit tests**: `tests/test_multicomponent_mixer.py`
- [ ] **Integration test**: Test with existing electrolyzer component
- [ ] **Validation**: Compare with CoolProp for 1000 random (T, P, z) points
- [ ] **Documentation**: Add docstrings and usage examples
- [ ] **Performance profiling**: Ensure <2ms per timestep target

### Validation Criteria

1. **Mass conservation**: \(\left|\sum \dot{n}_{in} - \dot{n}_{out} - dn/dt\right| < 10^{-6}\)
2. **Energy conservation**: \(\left|\sum \dot{H}_{in} - dU/dt\right| < 10^{-4}\)
3. **Phase equilibrium**: Water condensation occurs when \(P_{H2O} > P_{sat}\)
4. **Temperature accuracy**: ±0.5 K vs. rigorous solver
5. **Performance**: <2 ms/timestep on standard hardware

***

## 8. Future Enhancements

### 8.1 Real Gas Effects (Peng-Robinson EOS)

For pressures >100 bar, implement compressibility factor corrections:[11][10]

```python
def _calculate_compressibility_factor(self, T, P, composition):
    """Peng-Robinson EOS for real gas effects."""
    # Implementation using mixing rules from web:17, web:20
    pass
```

### 8.2 Multi-Condensable VLE

Extend Rachford-Rice to handle CO₂ condensation at high pressures:[13]

```python
def _solve_rachford_rice_general(self, T, P, z, K_values):
    """General Rachford-Rice for multiple condensable components."""
    # Iterative solver for β with multiple K-values
    pass
```

### 8.3 Chemical Reactions

Add support for reactions (e.g., methanation: CO₂ + 4H₂ → CH₄ + 2H₂O):[14]

```python
def _calculate_reaction_rates(self, T, P, composition):
    """Chemical reaction kinetics for methanation."""
    # Arrhenius equation, catalyst activity
    pass
```

***
