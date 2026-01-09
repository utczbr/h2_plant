"""
Integrated ATR Plant Block - Black Box Component.

Replaces the three-component chain (ATR_Reactor → ATR_HTWGS_Block → ATR_LTWGS_Block)
with a single data-driven component that maps inputs directly to final post-WGS syngas
output using the ATR_linear_regressions.csv surrogate model.

The regression data provides FINAL offgas composition after all shift reactions are complete.
This eliminates inter-component "drift" and ensures mass/energy balance matches the
rigorous Aspen/HYSYS model that generated the CSV.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.components.reforming.atr_data_manager import (
    ATRBaseComponent, 
    ATRDataManager,
    C_TO_K, 
    KW_TO_W
)

logger = logging.getLogger(__name__)

# Molar Masses (kg/kmol)
MW = {
    'H2': 2.016,
    'CH4': 16.04,
    'H2O': 18.015,
    'CO': 28.01,
    'CO2': 44.01,
    'O2': 31.998,
    'N2': 28.014
}


class IntegratedATRPlant(ATRBaseComponent):
    """
    Integrated ATR Plant Block - Data-Driven Black Box.
    
    Combines ATR Reactor + HTWGS + LTWGS into a single component that uses
    regression table lookups to determine outputs directly from oxygen input.
    
    Inputs:
        inlet: Mixed feed stream (biogas + steam + oxygen from ATR_Feed_Mixer)
        
    Outputs:
        syngas_out: Fully-shifted wet syngas (H₂, CO₂, H₂O, trace CH₄)
        heat_duty: Total thermal load across all integrated units (kW)
        
    Key Features:
        - Uses O₂ rate as independent variable for all lookups
        - Retrieves FINAL offgas composition (xH2, xCO2, xCH4) from regression
        - Calculates H₂O by difference to ensure unity
        - Sums heat duties H01-H09 for energy tracking
        - Applies lumped pressure drop across entire plant
    """
    
    def __init__(
        self,
        component_id: str,
        max_flow_kg_h: float = 20000.0,
        pressure_drop_bar: float = 2.5,
        **kwargs
    ):
        super().__init__(component_id=component_id)
        self.component_id = component_id
        self.max_flow_kg_h = max_flow_kg_h
        self.pressure_drop_bar = pressure_drop_bar
        self.lut_manager = None
        
        # Valid O2 range from regression data (kmol/h)
        # Match exact bounds from ATR_linear_regressions.csv to avoid extrapolation
        self.MIN_O2_RATE = 7.194  # First row: 7.19391
        self.MAX_O2_RATE = 23.75  # Last row: 23.7509
        
        # State Variables
        self.oxygen_flow_kmol_h = 0.0
        self.h2_production_kmol_h = 0.0
        self.heat_duty_kw = 0.0
        self.syngas_flow_kg_h = 0.0
        self.syngas_flow_kmol_h = 0.0
        
        # Input Buffers (accumulated per timestep)
        self._buffer_total_mass_kg = 0.0
        self._buffer_oxygen_kmol = 0.0
        self._buffer_biogas_kmol = 0.0
        self._buffer_steam_kmol = 0.0
        self._inlet_pressure_pa = 3.0e5  # Default inlet pressure
        self._inlet_temp_k = 700.0       # Default pre-reactor temp
        
        # Output State
        self._syngas_output_buffer_kg = 0.0
        self.current_syngas_comp = {
            'H2': 0.15,
            'CO2': 0.78,
            'H2O': 0.05,
            'CH4': 0.02
        }
        
        # Hourly logging tracker
        self._last_log_hour = -1

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize component, load data manager."""
        super().initialize(dt, registry)
        
        # Inject LUT Manager from Registry (optional, for advanced thermo)
        if registry and hasattr(registry, 'has') and registry.has('lut_manager'):
            self.lut_manager = registry.get('lut_manager')
        
        logger.info(f"IntegratedATRPlant '{self.component_id}' initialized. "
                   f"O₂ range: [{self.MIN_O2_RATE:.2f}, {self.MAX_O2_RATE:.2f}] kmol/h, "
                   f"ΔP: {self.pressure_drop_bar:.1f} bar")

    def step(self, t: float) -> None:
        """
        Execute one simulation timestep using regression table lookups.
        
        Algorithm (TOTAL REACTOR EFFLUENT - Mixed Phase):
        1. Calculate O₂ rate from accumulated buffer
        2. Clamp O₂ to valid regression domain
        3. Lookup gas (Fm_offgas, F_offgas) AND liquid (Fm_water, F_water) streams
        4. Calculate TOTAL mass flow (gas + liquid)
        5. Re-normalize compositions against TOTAL molar flow
        6. Generate mixed-phase output stream
        7. Sum heat duties for energy tracking
        """
        
        # ===================================================================
        # STEP 1: DETERMINE O2 RATE FROM BUFFER
        # ===================================================================
        o2_rate_kmol_h = self._buffer_oxygen_kmol / self.dt if self.dt > 0 else 0.0
        
        # Handle idle state
        if o2_rate_kmol_h < 0.1:
            self.h2_production_kmol_h = 0.0
            self.heat_duty_kw = 0.0
            self.oxygen_flow_kmol_h = 0.0
            self.syngas_flow_kg_h = 0.0
            self.syngas_flow_kmol_h = 0.0
            self._syngas_output_buffer_kg = 0.0
            self.current_syngas_comp = {'H2': 0.0, 'CO2': 0.0, 'H2O': 1.0, 'CH4': 0.0}
            self._clear_buffers()
            return
        
        # ===================================================================
        # STEP 2: CLAMP O2 TO REGRESSION DOMAIN
        # ===================================================================
        clamped_o2 = np.clip(o2_rate_kmol_h, self.MIN_O2_RATE, self.MAX_O2_RATE)
        self.oxygen_flow_kmol_h = clamped_o2
        
        # ===================================================================
        # STEP 3: LOOKUP ALL THREE OUTPUT STREAMS FROM REGRESSION
        # ===================================================================
        # The CSV represents post-PSA separation:
        #   - F_H2_func / Fm_H2_func: Pure H2 product from PSA
        #   - F_offgas_func / Fm_offgas_func: Tail gas (CO2, CH4, unrecovered H2)
        #   - F_water_func / Fm_water_func: Liquid condensate water
        #
        # To get TRUE REACTOR EFFLUENT, we must RECOMBINE all three!
        
        dm = self.data_manager
        
        # --- Pure H2 Product Stream (from PSA) ---
        Fm_H2 = dm.lookup('Fm_H2_func', clamped_o2)    # kg/h (pure H2 mass)
        F_H2 = dm.lookup('F_H2_func', clamped_o2)      # kmol/h (pure H2 moles)
        
        # --- Tail Gas Stream (PSA reject) ---
        Fm_offgas = dm.lookup('Fm_offgas_func', clamped_o2)  # kg/h (tail gas mass)
        F_offgas = dm.lookup('F_offgas_func', clamped_o2)    # kmol/h (tail gas moles)
        
        # --- Liquid Condensate Water ---
        Fm_water = dm.lookup('Fm_water_func', clamped_o2)    # kg/h (liquid water mass)
        F_water = dm.lookup('F_water_func', clamped_o2)      # kmol/h (liquid water moles)
        
        # --- Tail Gas Composition (mole fractions of offgas stream) ---
        x_H2_tail = dm.lookup('xH2_offgas_func', clamped_o2)
        x_CO2_tail = dm.lookup('xCO2_offgas_func', clamped_o2)
        x_CH4_tail = dm.lookup('xCH4_offgas_func', clamped_o2)
        
        # ===================================================================
        # STEP 4: CALCULATE TOTAL FLOW (H2 + OFFGAS + WATER = REACTOR EFFLUENT)
        # ===================================================================
        # Total Mass Flow: Fm_total = Fm_H2 + Fm_offgas + Fm_water
        Fm_total = Fm_H2 + Fm_offgas + Fm_water
        
        # Total Molar Flow: F_total = F_H2 + F_offgas + F_water
        F_total = F_H2 + F_offgas + F_water
        
        # ===================================================================
        # STEP 5: CALCULATE INDIVIDUAL SPECIES MOLES (RECOMBINATION)
        # ===================================================================
        # H2: Pure product + slip in tail gas
        moles_H2 = F_H2 + (F_offgas * x_H2_tail)
        
        # CO2: Only in tail gas
        moles_CO2 = F_offgas * x_CO2_tail
        
        # CH4: Only in tail gas
        moles_CH4 = F_offgas * x_CH4_tail
        
        # H2O: Liquid condensate + Vapor in tail gas (the ~2.5% gap)
        # NOTE: The regression "gap" is water vapor, NOT CO.
        x_gap_vapor = 1.0 - (x_H2_tail + x_CO2_tail + x_CH4_tail)
        moles_H2O = F_water + (F_offgas * x_gap_vapor)

        # ===================================================================
        # STEP 6: CALCULATE NEW MOLE FRACTIONS WITH CLOSURE
        # ===================================================================
        if F_total > 1e-6:
            # Calculate preliminary fractions
            y_H2  = moles_H2 / F_total
            y_CO2 = moles_CO2 / F_total
            y_CH4 = moles_CH4 / F_total
            y_H2O = moles_H2O / F_total
            
            # Closure: Assign any remaining mathematical residual to CO
            # (This fulfills the user requirement: co_mol_frac = 1 - sum)
            y_CO = 1.0 - (y_H2 + y_CO2 + y_CH4 + y_H2O)
            
            # Safety clamp for floating point noise
            y_CO = max(0.0, y_CO)
        else:
            y_H2, y_CO2, y_CH4, y_CO, y_H2O = 0.0, 0.0, 0.0, 0.0, 1.0
        
        self.current_syngas_mol_frac = {
            'H2': y_H2,
            'CO2': y_CO2,
            'H2O': y_H2O,
            'CH4': y_CH4,
            'CO': y_CO
        }
        
        # Convert mole fractions to mass fractions (Stream expects mass fractions)
        mass_H2 = y_H2 * MW['H2']
        mass_CO2 = y_CO2 * MW['CO2']
        mass_H2O = y_H2O * MW['H2O']
        mass_CH4 = y_CH4 * MW['CH4']
        mass_CO = y_CO * MW['CO']
        total_mass_mw = mass_H2 + mass_CO2 + mass_H2O + mass_CH4 + mass_CO
        
        if total_mass_mw > 0:
            self.current_syngas_comp = {
                'H2': mass_H2 / total_mass_mw,
                'CO2': mass_CO2 / total_mass_mw,
                'H2O': mass_H2O / total_mass_mw,
                'CH4': mass_CH4 / total_mass_mw,
                'CO': mass_CO / total_mass_mw
            }
        else:
            self.current_syngas_comp = {'H2': 0.0, 'CO2': 0.0, 'H2O': 1.0, 'CH4': 0.0, 'CO': 0.0}
        
        # ===================================================================
        # STEP 6: USE TOTAL MASS FLOW AS OUTPUT (Gas + Liquid)
        # ===================================================================
        self.syngas_flow_kg_h = Fm_total  # Total reactor effluent
        self.syngas_flow_kmol_h = F_total
        self.h2_production_kmol_h = F_H2
        
        # Buffer for extract_output
        self._syngas_output_buffer_kg = Fm_total * self.dt
        
        # Track liquid water for downstream separation
        self.liquid_water_flow_kg_h = Fm_water
        self.liquid_water_flow_kmol_h = F_water
        
        # ===================================================================
        # STEP 7: SUM HEAT DUTIES (H01 through H09)
        # ===================================================================
        # Heating duties (positive = heat added to system)
        H01 = dm.lookup('H01_Q_func', clamped_o2)  # Biogas preheater
        H02 = dm.lookup('H02_Q_func', clamped_o2)  # Oxygen preheater
        H04 = dm.lookup('H04_Q_func', clamped_o2)  # Steam/feed heater
        H05 = dm.lookup('H05_Q_func', clamped_o2)  # Syngas cooler (usually negative)
        H08 = dm.lookup('H08_Q_func', clamped_o2)  # HTWGS heat (usually negative, exothermic)
        H09 = dm.lookup('H09_Q_func', clamped_o2)  # LTWGS heat (usually negative, exothermic)
        
        # Total heat duty (positive = net heating required, negative = cooling required)
        self.heat_duty_kw = H01 + H02 + H04 + H05 + H08 + H09
        
        # ===================================================================
        # STEP 8: CLEAR INPUT BUFFERS
        # ===================================================================
        self._clear_buffers()
        
        # ===================================================================
        # STEP 9: DEBUG LOGGING (Hourly)
        # ===================================================================
        current_hour = int(t)
        if current_hour > self._last_log_hour:
            self._last_log_hour = current_hour
            h2_mass_kg_h = self.h2_production_kmol_h * MW['H2']
            
            logger.debug(f"IntegratedATRPlant [t={t:.2f}h] (TRUE REACTOR EFFLUENT):")
            logger.debug(f"  O₂ Rate: {clamped_o2:.2f} kmol/h")
            logger.debug(f"  Total Out: {self.syngas_flow_kg_h:.1f} kg/h (H2:{Fm_H2:.1f} + Tail:{Fm_offgas:.1f} + Liq:{Fm_water:.1f})")
            logger.debug(f"  H₂ in Mix: {moles_H2:.2f} kmol/h (Pure:{F_H2:.2f} + Slip:{F_offgas * x_H2_tail:.2f})")
            logger.debug(f"  Total Comp: H2={y_H2*100:.1f}%, CO2={y_CO2*100:.1f}%, H2O={y_H2O*100:.1f}%, CH4={y_CH4*100:.2f}%")
            logger.debug(f"  Heat Duty: {self.heat_duty_kw:.1f} kW")

    def _clear_buffers(self) -> None:
        """Clear all input buffers after processing."""
        self._buffer_total_mass_kg = 0.0
        self._buffer_oxygen_kmol = 0.0
        self._buffer_biogas_kmol = 0.0
        self._buffer_steam_kmol = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Return current component state for reporting."""
        return {
            **super().get_state(),
            'component_id': self.component_id,
            'oxygen_flow_kmol_h': self.oxygen_flow_kmol_h,
            'h2_production_kmol_h': self.h2_production_kmol_h,
            'syngas_flow_kg_h': self.syngas_flow_kg_h,
            'syngas_flow_kmol_h': self.syngas_flow_kmol_h,
            'liquid_water_flow_kg_h': getattr(self, 'liquid_water_flow_kg_h', 0.0),
            'heat_duty_kw': self.heat_duty_kw,
            'syngas_composition': self.current_syngas_comp.copy()
        }

    def get_output(self, port_name: str) -> Any:
        """
        Retrieve output stream from specified port.
        
        Args:
            port_name: 'syngas_out' or 'heat_duty'
            
        Returns:
            Stream object for syngas, float for heat duty
        """
        if port_name == 'syngas_out':
            # Outlet temperature: Use Tin_H05 from regression (Syngas Cooler INLET)
            # This represents the hot reactor effluent BEFORE cooling
            T_out_c = self.data_manager.lookup('Tin_H05_func', self.oxygen_flow_kmol_h)
            if T_out_c < 50 or T_out_c > 500:
                T_out_c = 200.0  # Fallback to reasonable post-LTWGS temperature
            T_out_k = T_out_c + C_TO_K
            
            # Apply lumped pressure drop
            P_out_pa = max(self._inlet_pressure_pa - (self.pressure_drop_bar * 1e5), 1.0e5)
            
            # Construct output stream
            return Stream(
                mass_flow_kg_h=self.syngas_flow_kg_h,
                temperature_k=T_out_k,
                pressure_pa=P_out_pa,
                composition=self.current_syngas_comp.copy(),
                phase='gas'
            )
        
        elif port_name == 'heat_duty' or port_name == 'heat_out':
            return self.heat_duty_kw
        
        elif port_name == 'o2_signal':
            # Signal port: broadcast current O2 operating point (kmol/h)
            return self.oxygen_flow_kmol_h
        
        else:
            return None

    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """
        Accept input streams and accumulate mass/moles for the timestep.
        
        Args:
            port_name: 'inlet' for mixed feed from ATR_Feed_Mixer
            value: Stream object
            resource_type: Resource hint (ignored, we parse composition)
            
        Returns:
            Accepted flow rate (kg/h)
        """
        if isinstance(value, Stream):
            mass_kg = value.mass_flow_kg_h * self.dt
            self._buffer_total_mass_kg += mass_kg
            self._inlet_pressure_pa = value.pressure_pa
            self._inlet_temp_k = value.temperature_k
            
            comp = value.composition
            
            # Extract O2 (the primary driver for the regression model)
            if 'O2' in comp:
                o2_mass_fraction = comp['O2']
                o2_mass_kg = mass_kg * o2_mass_fraction
                self._buffer_oxygen_kmol += o2_mass_kg / MW['O2']
            
            # Track biogas components (for logging/debugging)
            if 'CH4' in comp:
                ch4_mass_kg = mass_kg * comp['CH4']
                self._buffer_biogas_kmol += ch4_mass_kg / MW['CH4']
            
            # Track steam
            if 'H2O' in comp:
                h2o_mass_kg = mass_kg * comp['H2O']
                self._buffer_steam_kmol += h2o_mass_kg / MW['H2O']
            
            return value.mass_flow_kg_h
        
        elif isinstance(value, (int, float)):
            # Scalar input (assumed to be flow rate in kmol/h)
            amount_kmol = value * self.dt
            if port_name in ('o2_in', 'oxygen_in'):
                self._buffer_oxygen_kmol += amount_kmol
            elif port_name in ('biogas_in', 'methane_in'):
                self._buffer_biogas_kmol += amount_kmol
            elif port_name in ('steam_in', 'water_in'):
                self._buffer_steam_kmol += amount_kmol
            return value
        
        return 0.0

    def extract_output(self, port_name: str, amount: float, resource_type: str) -> None:
        """Mark output as extracted (buffer clearing)."""
        if port_name == 'syngas_out':
            self._syngas_output_buffer_kg = max(0.0, self._syngas_output_buffer_kg - amount * self.dt)

    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Define physical connection ports for this component."""
        return {
            'inlet': {
                'type': 'input',
                'resource_type': 'stream',
                'units': 'kg/h',
                'description': 'Mixed feed from ATR_Feed_Mixer (biogas + steam + O2)'
            },
            'syngas_out': {
                'type': 'output',
                'resource_type': 'syngas',
                'units': 'kg/h',
                'description': 'Fully-shifted wet syngas (H2, CO2, H2O, trace CH4)'
            },
            'heat_duty': {
                'type': 'output',
                'resource_type': 'heat',
                'units': 'kW',
                'description': 'Net heat duty (positive = heating, negative = cooling)'
            },
            'o2_signal': {
                'type': 'output',
                'resource_type': 'signal',
                'units': 'kmol/h',
                'description': 'Current O2 operating point for syncing other components'
            }
        }
