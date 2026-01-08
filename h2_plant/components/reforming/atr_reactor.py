"""
Autothermal Reforming (ATR) Reactor Component.

Updated to use ATRDataManager for surrogate model lookups and Layer 1 Thermodynamics.
Implements Dynamic Stoichiometry to correctly process variable Biogas compositions.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.components.reforming.atr_data_manager import ATRBaseComponent, KW_TO_W

# Import thermodynamic helpers
try:
    import h2_plant.optimization.mixture_thermodynamics as mix_thermo
except ImportError:
    mix_thermo = None

logger = logging.getLogger(__name__)

class ATRReactor(ATRBaseComponent):
    """
    Autothermal Reforming reactor for hydrogen production from biogas.
    
    Integrated with ATRDataManager for process model via surrogate functions,
    and LUTManager for rigorous output stream enthalpy.

    Attributes:
        max_flow_kg_h (float): Maximum design flow capacity (kg/h).
        lut_manager: Injected LUT manager for thermodynamic lookups.
        h2_production_kmol_h (float): Current H₂ production rate.
        heat_duty_kw (float): Current net heat duty.
    """

    def __init__(
        self,
        component_id: str,
        max_flow_kg_h: float,
    ):
        super().__init__()
        self.component_id = component_id
        self.max_flow_kg_h = max_flow_kg_h
        self.lut_manager = None

        # State Variables
        self.oxygen_flow_kmol_h = 0.0
        self.h2_production_kmol_h = 0.0
        self.heat_duty_kw = 0.0
        self._current_syngas_flow_kg_h = 0.0
        self.biogas_input_kmol_h = 0.0
        self.steam_input_kmol_h = 0.0
        self.water_input_kmol_h = 0.0
        
        # Input Buffers
        self.buffer_oxygen_kmol = 0.0
        self.buffer_biogas_kmol = 0.0
        self.buffer_steam_kmol = 0.0
        
        # Track total input for mass balance
        self._total_input_mass_kg = 0.0
        self._inlet_pressure_pa = 3.0e5  # Default, updated by receive_input
        
        # Dynamic Biogas Composition Tracking
        # Default: 60% CH4, 40% CO2 (will be updated from actual stream)
        self._biogas_ch4_fraction = 0.60

        # Output Buffers
        self._h2_output_buffer_kmol = 0.0
        self._offgas_output_buffer_kmol = 0.0
        self._heat_output_buffer_kw = 0.0
        
        # Dynamic Composition State
        self.current_offgas_comp = {'CO2': 0.7, 'H2': 0.3}

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """
        Initialize reactor, load data manager, and inject thermo dependencies.
        """
        # Call ATRBaseComponent.initialize -> sets up self.data_manager
        super().initialize(dt, registry)
        
        # Inject LUT Manager from Registry
        if registry and hasattr(registry, 'has') and registry.has('lut_manager'):
            self.lut_manager = registry.get('lut_manager')
        else:
            logger.warning(f"ATR {self.component_id}: LUTManager not found in registry. "
                           "Thermodynamics will be approximate.")

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep using STRICT ATOMIC MASS BALANCE.

        Algorithm:
        1. Clamp O2 rate to design limit (23.75 kmol/h) BEFORE any lookups.
        2. Count input atoms using ACTUAL dynamic biogas composition.
        3. Determine H2 production from regression, capped at stoichiometric limit.
        4. Solve C/O balance algebraically to conserve all atoms.
        5. Update buffers with actual consumption.
        """
        
        # ============================================================
        # STEP 1: STRICT O2 RATE CLAMPING (The "10x Fix")
        # ============================================================
        MAX_O2_RATE_KMOL_H = 23.75
        
        # Calculate potential O2 rate from accumulated buffer
        potential_o2_rate = self.buffer_oxygen_kmol / self.dt if self.dt > 0 else 0.0
        
        # CRITICAL: Clamp O2 rate to design limit BEFORE all lookups
        clamped_o2_rate = min(potential_o2_rate, MAX_O2_RATE_KMOL_H)
        
        if clamped_o2_rate < 1e-6:
            # Idle State
            self.h2_production_kmol_h = 0.0
            self.heat_duty_kw = 0.0
            self.oxygen_flow_kmol_h = 0.0
            self._syngas_output_buffer_kg = 0.0
            self._current_syngas_flow_kg_h = 0.0
            self.current_offgas_comp = {'H2': 0.35, 'CO': 0.10, 'CO2': 0.15, 'H2O': 0.39, 'CH4': 0.01}
            return
        
        # ============================================================
        # STEP 2: LOOKUP STOICHIOMETRIC REQUIREMENTS (Using CLAMPED O2)
        # ============================================================
        # All lookups use clamped_o2_rate to stay within regression domain
        req_biogas_rate = self.data_manager.lookup('F_bio_func', clamped_o2_rate)
        req_steam_rate = self.data_manager.lookup('F_steam_func', clamped_o2_rate)
        h2_target = self.data_manager.lookup('F_H2_func', clamped_o2_rate)
        
        # ============================================================
        # STEP 3: DETERMINE ACTUAL CONSUMPTION (Limited by Buffers)
        # ============================================================
        # Available rates from buffers (kmol/h)
        avail_biogas_rate = self.buffer_biogas_kmol / self.dt if self.dt > 0 else 0.0
        avail_steam_rate = self.buffer_steam_kmol / self.dt if self.dt > 0 else 0.0
        avail_o2_rate = self.buffer_oxygen_kmol / self.dt if self.dt > 0 else 0.0
        
        # Actual consumption: min(required, available)
        n_biogas = min(req_biogas_rate, avail_biogas_rate)
        n_steam = min(req_steam_rate, avail_steam_rate)
        n_oxygen = min(clamped_o2_rate, avail_o2_rate)
        
        # ============================================================
        # STEP 4: COUNT INPUT ATOMS (Dynamic Biogas Stoichiometry)
        # ============================================================
        # Uses actual CH4 fraction from input stream (tracked in receive_input)
        x_ch4 = self._biogas_ch4_fraction  # Actual CH4 mole fraction in biogas
        x_co2 = 1.0 - x_ch4                # Remaining is assumed CO2/Inert C
        
        # CH4 contributes: 1 C, 4 H per mole
        # CO2 contributes: 1 C, 2 O per mole
        n_C_in = n_biogas * 1.0  # x_ch4*1 + x_co2*1 = 1 C per mole biogas (always)
        n_H_in = (n_biogas * (x_ch4 * 4.0)) + (n_steam * 2.0)  # Dynamic H from actual CH4
        n_O_in = (n_oxygen * 2.0) + (n_steam * 1.0) + (n_biogas * (x_co2 * 2.0))  # O from CO2
        
        # ============================================================
        # STEP 5: SOLVE FOR OUTPUTS (Atomic Mass Balance)
        # ============================================================
        
        # A. Methane Slip: Fixed 0.5% of biogas CH4 content unreacted
        n_CH4_out = n_biogas * x_ch4 * 0.005
        
        # B. Hydrogen Production (Driver)
        # Stoichiometric limit: Cap H2 at 95% of available H atoms (ensures water remains)
        h2_stoich_limit = (n_H_in / 2.0) * 0.95
        n_H2_out = min(h2_target * (x_ch4 / 0.6), h2_stoich_limit) # Scale target by CH4 richness
        # Note: We scale the regression target by (actual_ch4 / 0.6) because the regression
        # was trained on 60% CH4. If we have 80%, we should expect ~33% more yield potential.
        
        n_H2_out = max(0.0, n_H2_out)
        
        # C. Water Output (Hydrogen Balance)
        # Remaining H goes to water: H_remaining = H_in - 2*H2 - 4*CH4_slip
        n_H2O_out = (n_H_in - (2.0 * n_H2_out) - (4.0 * n_CH4_out)) / 2.0
        n_H2O_out = max(0.0, n_H2O_out)
        
        # D. Carbon/Oxygen Balance (The "Missing Carbon" Fix)
        # Remaining oxygen available for carbon species
        n_O_available = n_O_in - n_H2O_out
        n_C_available = n_C_in - n_CH4_out
        
        # System of equations:
        # (A) n_CO + n_CO2 = n_C_available
        # (B) n_CO + 2*n_CO2 = n_O_available
        # Solution:
        n_CO2_out = n_O_available - n_C_available
        n_CO_out = n_C_available - n_CO2_out
        
        # Sanity Handling
        n_O2_excess = 0.0
        if n_CO_out < 0:
            # Excess Oxygen -> Combustion mode (all C goes to CO2)
            n_O2_excess = -n_CO_out / 2.0
            n_CO_out = 0.0
            n_CO2_out = n_C_available
        
        if n_CO2_out < 0:
            # Oxygen Deficit -> Maximize CO formation
            n_CO2_out = 0.0
            n_CO_out = min(n_C_available, n_O_available)
        
        # ============================================================
        # STEP 6: COMPUTE COMPOSITION AND MASS FLOW
        # ============================================================
        total_moles_out = n_H2_out + n_CH4_out + n_H2O_out + n_CO_out + n_CO2_out + n_O2_excess
        
        if total_moles_out > 1e-9:
            self.current_offgas_comp = {
                'H2': n_H2_out / total_moles_out,
                'CH4': n_CH4_out / total_moles_out,
                'H2O': n_H2O_out / total_moles_out,
                'CO': n_CO_out / total_moles_out,
                'CO2': n_CO2_out / total_moles_out,
                'O2': n_O2_excess / total_moles_out
            }
        else:
            self.current_offgas_comp = {'H2': 0.35, 'CO': 0.10, 'CO2': 0.15, 'H2O': 0.39, 'CH4': 0.01}
        
        # Mass output (kg/h)
        MW = {'H2': 2.016, 'CH4': 16.04, 'H2O': 18.015, 'CO': 28.01, 'CO2': 44.01, 'O2': 31.998}
        total_mass_out = (n_H2_out * MW['H2'] + 
                          n_CH4_out * MW['CH4'] + 
                          n_H2O_out * MW['H2O'] + 
                          n_CO_out * MW['CO'] + 
                          n_CO2_out * MW['CO2'] + 
                          n_O2_excess * MW['O2'])
        
        # ============================================================
        # STEP 7: UPDATE STATE VARIABLES
        # ============================================================
        self.h2_production_kmol_h = n_H2_out
        self.oxygen_flow_kmol_h = n_oxygen
        self.biogas_input_kmol_h = n_biogas
        self.steam_input_kmol_h = n_steam
        self._current_syngas_flow_kg_h = total_mass_out
        self._syngas_output_buffer_kg = total_mass_out * self.dt
        
        # Heat Duty
        h01 = self.data_manager.lookup('H01_Q_func', clamped_o2_rate)
        h02 = self.data_manager.lookup('H02_Q_func', clamped_o2_rate)
        h04 = self.data_manager.lookup('H04_Q_func', clamped_o2_rate)
        self.heat_duty_kw = h01 + h02 + h04
        self._heat_output_buffer_kw = self.heat_duty_kw * self.dt
        
        # ============================================================
        # STEP 8: CONSUME BUFFERS
        # ============================================================
        self.buffer_biogas_kmol -= n_biogas * self.dt
        self.buffer_steam_kmol -= n_steam * self.dt
        self.buffer_oxygen_kmol -= n_oxygen * self.dt
        
        # Clamp to non-negative
        self.buffer_biogas_kmol = max(0.0, self.buffer_biogas_kmol)
        self.buffer_steam_kmol = max(0.0, self.buffer_steam_kmol)
        self.buffer_oxygen_kmol = max(0.0, self.buffer_oxygen_kmol)
        
        # ============================================================
        # DEBUG LOGGING
        # ============================================================
        if int(t * 60) % 60 == 0:  # Log once per hour
            h2_mass = n_H2_out * MW['H2']
            logger.info(f"ATR STRICT BALANCE [t={t:.2f}h]:")
            logger.info(f"  Biogas Quality: {x_ch4*100:.1f}% CH4 (Stoich Factor: {x_ch4/0.6:.2f}x)")
            logger.info(f"  Inputs: Biogas={n_biogas:.2f}, Steam={n_steam:.2f}, O2={n_oxygen:.2f} kmol/h")
            logger.info(f"  Atoms IN: C={n_C_in:.2f}, H={n_H_in:.2f}, O={n_O_in:.2f}")
            logger.info(f"  H2: target={h2_target:.2f}, actual={n_H2_out:.2f} kmol/h ({h2_mass:.1f} kg/h)")
            logger.info(f"  Mass Out: {total_mass_out:.1f} kg/h")

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'h2_production_kmol_h': self.h2_production_kmol_h,
            'heat_duty_kw': self.heat_duty_kw,
            'biogas_input_kmol_h': self.biogas_input_kmol_h,
            'oxygen_flow_kmol_h': self.oxygen_flow_kmol_h
        }

    def get_output(self, port_name: str) -> Any:
        T_out_nominal = 950.0 + 273.15  # 1223.15 K
        P_out_nominal = 3.0e5

        if port_name == 'syngas_out':
            mass_flow = self._current_syngas_flow_kg_h
            P_out = max(self._inlet_pressure_pa - 0.5e5, 1.0e5)
            
            syngas_comp = self.current_offgas_comp.copy()
            total = sum(syngas_comp.values())
            if total > 1e-9:
                syngas_comp = {k: v/total for k, v in syngas_comp.items()}
            else:
                syngas_comp = {'H2': 0.35, 'CO': 0.10, 'CO2': 0.15, 'H2O': 0.39, 'CH4': 0.01}
            
            return Stream(
                mass_flow_kg_h=mass_flow,
                temperature_k=T_out_nominal,
                pressure_pa=P_out,
                composition=syngas_comp,
                phase='gas'
            )
        
        elif port_name == 'heat_out':
            return self._heat_output_buffer_kw / self.dt if self.dt > 0 else 0.0
        else:
             return None

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        if isinstance(value, Stream):
            mass_kg = value.mass_flow_kg_h * self.dt

            if port_name == 'inlet':
                comp = value.composition
                self._total_input_mass_kg += mass_kg
                self._inlet_pressure_pa = value.pressure_pa
                
                if 'CH4' in comp:
                    self.buffer_biogas_kmol += (mass_kg * comp['CH4']) / 16.034
                if 'CO2' in comp:
                    # DYNAMIC STOICHIOMETRY TRACKING
                    ch4_frac = comp.get('CH4', 0.0)
                    co2_frac = comp.get('CO2', 0.0)
                    if ch4_frac + co2_frac > 0.01:
                        # Calculate mole fraction of CH4 in the Biogas portion
                        # (ignoring inert N2 for the reaction stoichiometry calc)
                        self._biogas_ch4_fraction = ch4_frac / (ch4_frac + co2_frac)
                        
                if 'O2' in comp:
                    self.buffer_oxygen_kmol += (mass_kg * comp['O2']) / 31.998
                if 'H2O' in comp:
                    self.buffer_steam_kmol += (mass_kg * comp['H2O']) / 18.015
                return value.mass_flow_kg_h

            elif port_name == 'o2_in':
                self.buffer_oxygen_kmol += mass_kg / 32.0
                return value.mass_flow_kg_h
            elif port_name == 'biogas_in':
                self.buffer_biogas_kmol += mass_kg / 16.0
                return value.mass_flow_kg_h
            elif port_name == 'steam_in':
                self.buffer_steam_kmol += mass_kg / 18.0
                return value.mass_flow_kg_h

        elif isinstance(value, (int, float)):
            amount_kmol = value * self.dt
            if port_name == 'o2_in': self.buffer_oxygen_kmol += amount_kmol
            elif port_name == 'biogas_in': self.buffer_biogas_kmol += amount_kmol
            elif port_name == 'steam_in': self.buffer_steam_kmol += amount_kmol
            return value

        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        if port_name == 'syngas_out':
            self._syngas_output_buffer_kg = 0.0
        elif port_name == 'heat_out':
            self._heat_output_buffer_kw = 0.0

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        return {
            'inlet': {'type': 'input', 'resource_type': 'stream', 'units': 'kg/h'},
            'biogas_in': {'type': 'input', 'resource_type': 'methane', 'units': 'kmol/h'},
            'steam_in': {'type': 'input', 'resource_type': 'steam', 'units': 'kmol/h'},
            'o2_in': {'type': 'input', 'resource_type': 'oxygen', 'units': 'kmol/h'},
            'syngas_out': {'type': 'output', 'resource_type': 'syngas', 'units': 'kg/h'},
            'heat_out': {'type': 'output', 'resource_type': 'heat', 'units': 'kW'}
        }
