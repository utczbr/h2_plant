
"""
Autothermal Reforming (ATR) Reactor Component.

Updated to use ATRDataManager for surrogate model lookups and Layer 1 Thermodynamics.

This module implements an autothermal reformer for converting methane (biogas)
to synthesis gas (syngas) containing hydrogen. ATR combines partial oxidation
with steam reforming in a single vessel, achieving thermal self-sufficiency.

Architecture:
    Implements the Component Lifecycle Contract (Layer 1):
    - `initialize()`: Initializes ATRDataManager and injects LUTManager.
    - `step()`: Calculates stoichiometry and production based on oxygen feed rate using Surrogate Model.
    - `get_state()`: Exposes production rates, heat duty, and feed requirements.

Model Approach:
    Uses `ATRDataManager` (CSV-based Linear Regression/Interpolation) to determine
    stoichiometric requirements and outputs based on Oxygen flow.
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
        # model_path arg removed as DataManager handles loading via singleton
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
        Execute one simulation timestep.

        1. Determine target O2 rate from buffer.
        2. Lookup required Biogas/Steam using ATRDataManager.
        3. Determine limiting reagent.
        4. Calculate actual outputs using ATRDataManager.
        5. Update buffers.
        """
        # Note: ATRBaseComponent.step() is not called because we override logic completely
        # But we need self.dt from Component.initialize (called via super)
        
        # Limiting Reagent Determination
        available_o2_kmol = self.buffer_oxygen_kmol
        target_o2_rate_kmol_h = available_o2_kmol / self.dt if self.dt > 0 else 0.0

        if target_o2_rate_kmol_h > 1e-6:
            # Query constraints based on Pilot Scale Model (F_O2)
            # Safe O2 range is ~7-24 kmol/h per module, but we scale linearly here
            
            # 1. Get Requirements
            req_biogas_rate = self.data_manager.lookup('F_bio_func', target_o2_rate_kmol_h)
            req_steam_rate = self.data_manager.lookup('F_steam_func', target_o2_rate_kmol_h)
            
            req_biogas_total = req_biogas_rate * self.dt
            req_steam_total = req_steam_rate * self.dt
            
            # 2. Compute limiting factor
            limit_factor = 1.0
            
            if req_biogas_total > 1e-9:
                if self.buffer_biogas_kmol < req_biogas_total:
                    limit_factor = min(limit_factor, self.buffer_biogas_kmol / req_biogas_total)
            
            if req_steam_total > 1e-9:
                if self.buffer_steam_kmol < req_steam_total:
                    limit_factor = min(limit_factor, self.buffer_steam_kmol / req_steam_total)
            
            # 3. Actual Consumption/Production Rates
            actual_o2_rate = target_o2_rate_kmol_h * limit_factor
            
            # Lookup outputs at actual operating point
            req_biogas_actual = self.data_manager.lookup('F_bio_func', actual_o2_rate)
            req_steam_actual = self.data_manager.lookup('F_steam_func', actual_o2_rate)
            
            h2_prod = self.data_manager.lookup('F_H2_func', actual_o2_rate)
            water_req = self.data_manager.lookup('F_water_func', actual_o2_rate)
            
            # Heat Duty Aggregation
            h01 = self.data_manager.lookup('H01_Q_func', actual_o2_rate)
            h02 = self.data_manager.lookup('H02_Q_func', actual_o2_rate)
            h04 = self.data_manager.lookup('H04_Q_func', actual_o2_rate)
            total_heat = h01 + h02 + h04
            
            # Offgas Composition
            x_co2 = self.data_manager.lookup('xCO2_offgas_func', actual_o2_rate)
            x_h2_off = self.data_manager.lookup('xH2_offgas_func', actual_o2_rate)
            x_ch4 = self.data_manager.lookup('xCH4_offgas_func', actual_o2_rate)

            # 4. Update Buffers (Consume Inputs)
            self.buffer_biogas_kmol -= req_biogas_actual * self.dt
            self.buffer_steam_kmol -= req_steam_actual * self.dt
            self.buffer_oxygen_kmol -= actual_o2_rate * self.dt
            
            # Clamp
            self.buffer_biogas_kmol = max(0.0, self.buffer_biogas_kmol)
            self.buffer_steam_kmol = max(0.0, self.buffer_steam_kmol)
            self.buffer_oxygen_kmol = max(0.0, self.buffer_oxygen_kmol)
            
            # 5. Update State
            self.h2_production_kmol_h = h2_prod
            self.heat_duty_kw = total_heat
            
            self.oxygen_flow_kmol_h = actual_o2_rate
            self.biogas_input_kmol_h = req_biogas_actual
            self.steam_input_kmol_h = req_steam_actual
            self.water_input_kmol_h = water_req
            
            # Accumulate Total Mass for Pass-through (MASS CONSERVATION)
            # Use actual input mass, not model requirements
            # The reactor should not create or destroy mass
            self._syngas_output_buffer_kg = self._total_input_mass_kg
            self._current_syngas_flow_kg_h = self._total_input_mass_kg / self.dt if self.dt > 0 else 0.0
            
            # Reset input mass tracking for next step
            self._total_input_mass_kg = 0.0
            
            total_x = x_co2 + x_h2_off + x_ch4
            if total_x > 0:
                self.current_offgas_comp = {'CO2': x_co2, 'H2': x_h2_off, 'CH4': x_ch4}
            
            # 6. Accumulate Outputs (Instantaneous for stream reporting)
            self._h2_output_buffer_kmol = self.h2_production_kmol_h * self.dt
            self._heat_output_buffer_kw = self.heat_duty_kw * self.dt
            # Offgas approximation (or use F_offgas if available in CSV)
            # Assuming 10% or derived from mass balance if critical
            self._offgas_output_buffer_kmol = 0.1 * self.h2_production_kmol_h * self.dt

        else:
            # Idle
            self.h2_production_kmol_h = 0.0
            self.heat_duty_kw = 0.0
            self.oxygen_flow_kmol_h = 0.0
            self._syngas_output_buffer_kg = 0.0
            self._current_syngas_flow_kg_h = 0.0

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(), # Note: ATRBaseComponent doesn't override get_state, uses Component's
            'component_id': self.component_id,
            'h2_production_kmol_h': self.h2_production_kmol_h,
            'heat_duty_kw': self.heat_duty_kw,
            'biogas_input_kmol_h': self.biogas_input_kmol_h,
            'oxygen_flow_kmol_h': self.oxygen_flow_kmol_h
        }

    def get_output(self, port_name: str) -> Any:
        # Nominal outlet conditions from ATR process design
        T_out_nominal = 900.0  # K (Reactor Exit is very hot) is 1300 kenvin in "legacy"
        P_out_nominal = 3.0e5  # Pa 

        if port_name == 'syngas_out':
            # Use cached flow rate for reporting because buffer might be cleared by extract_output
            mass_flow = self._current_syngas_flow_kg_h
            
            # Use inlet pressure with nominal reactor pressure drop (~0.5 bar)
            P_out = max(self._inlet_pressure_pa - 0.5e5, 1.0e5)  # At least 1 bar
            
            # Syngas composition from surrogate model
            # Use raw fractions from CSV as requested
            x_h2 = self.current_offgas_comp.get('H2', 0.15)
            x_co2 = self.current_offgas_comp.get('CO2', 0.78)
            x_ch4 = self.current_offgas_comp.get('CH4', 0.035)
            
            # Fixed CO fraction (small amount to fit within balance)
            x_co = 0.01 
            
            # Calculate H2O as balance
            # Clamp to 0 if sum exceeds 1.0 (will be handled by normalization)
            x_h2o = max(0.0, 1.0 - (x_h2 + x_co2 + x_ch4 + x_co))
            
            syngas_comp = {
                'H2': x_h2,
                'CO2': x_co2,
                'CH4': x_ch4,
                'CO': x_co,
                'H2O': x_h2o,
                'N2': 0.0 
            }
            
            # Normalize to ensure sum is exactly 1.0
            total = sum(syngas_comp.values())
            if total > 0:
                syngas_comp = {k: v/total for k, v in syngas_comp.items()}
            
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
        # Same accumulation logic as before
        if isinstance(value, Stream):
            mass_kg = value.mass_flow_kg_h * self.dt

            if port_name == 'inlet':
                comp = value.composition
                # Track total input mass for mass balance
                self._total_input_mass_kg += mass_kg
                # Capture inlet pressure for outlet calculation
                self._inlet_pressure_pa = value.pressure_pa
                
                if 'CH4' in comp:
                    self.buffer_biogas_kmol += (mass_kg * comp['CH4']) / 16.034
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
