"""
Compressor for hydrogen storage applications.

Refactored from legacy 'Compressor Armazenamento.py' to match h2_plant architecture.
Strictly preserves thermodynamic calculations and values from the reference implementation.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_ids import ComponentID
from h2_plant.core.enums import CompressorMode
from h2_plant.core.constants import ConversionFactors
from h2_plant.core.stream import Stream

# Import CoolProp for inverse lookup (P from S,T) which LUT doesn't support yet
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


class CompressorStorage(Component):
    """
    Multi-stage compressor for hydrogen storage operations.
    
    Implements multi-stage compression with inter-cooling to minimize work.
    Automatically calculates optimal number of stages based on maximum 
    allowable discharge temperature per stage.
    
    Physics preserved from legacy 'Compressor Armazenamento.py':
    - Multi-stage polytropic compression
    - Inter-stage cooling to inlet temperature
    - Chilling energy calculated from COP
    - Temperature-limited stage pressure ratio
    
    Example:
        # Charging scenario: 40 bar → 140 bar
        compressor = CompressorStorage(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=40.0,
            outlet_pressure_bar=140.0
        )
        
        compressor.initialize(dt=1.0, registry)
        
        # Process mass
        compressor.transfer_mass_kg = 50.0
        compressor.step(t=0.0)
        
        # Read results
        print(f"Energy: {compressor.energy_consumed_kwh:.2f} kWh")
        print(f"Stages: {compressor.num_stages}")
        print(f"Specific: {compressor.specific_energy_kwh_kg:.4f} kWh/kg")
    """
    
    def __init__(
        self,
        max_flow_kg_h: float,
        inlet_pressure_bar: float,
        outlet_pressure_bar: float,
        # Default values from legacy 'Compressor Armazenamento.py'
        inlet_temperature_c: float = 10.0,
        max_temperature_c: float = 85.0,
        isentropic_efficiency: float = 0.65,
        chiller_cop: float = 3.0
    ):
        """
        Initialize compressor storage.
        
        Args:
            max_flow_kg_h: Maximum mass flow rate (kg/h)
            inlet_pressure_bar: Inlet pressure (bar)
            outlet_pressure_bar: Outlet pressure (bar)
            inlet_temperature_c: Inlet temperature and inter-stage cooling target (°C)
            max_temperature_c: Maximum allowable discharge temperature (°C)
            isentropic_efficiency: Isentropic efficiency (0-1)
            chiller_cop: Chiller coefficient of performance for inter-cooling
        """
        super().__init__()
        
        # Configuration
        self.max_flow_kg_h = max_flow_kg_h
        self.inlet_pressure_bar = inlet_pressure_bar
        self.outlet_pressure_bar = outlet_pressure_bar
        
        # Physics Constants (from legacy source)
        self.inlet_temperature_c = inlet_temperature_c
        self.inlet_temperature_k = inlet_temperature_c + 273.15
        self.max_temperature_c = max_temperature_c
        self.max_temperature_k = max_temperature_c + 273.15
        self.isentropic_efficiency = isentropic_efficiency
        self.chiller_cop = chiller_cop
        
        # Unit Conversions (legacy source uses 1e5 for bar->Pa)
        self.BAR_TO_PA = 1e5
        # Legacy uses exact value 2.7778e-7 for J to kWh
        self.J_TO_KWH = 2.7778e-7
        
        # Internal State
        self.num_stages = 0
        self.stage_pressure_ratio = 1.0
        
        # Input variables (set by flow network or control logic)
        self.transfer_mass_kg = 0.0
        
        # Output variables (calculated in step)
        self.actual_mass_transferred_kg = 0.0
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.energy_consumed_kwh = 0.0
        self.heat_removed_kwh = 0.0
        self.energy_consumed_kwh = 0.0
        self.heat_removed_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0
        
        # Power attribute for Orchestrator monitoring
        self.power_kw = 0.0
        
        # State variables
        self.mode = CompressorMode.IDLE
        self.cumulative_energy_kwh = 0.0
        self.cumulative_mass_kg = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize component and calculate stage configuration."""
        super().initialize(dt, registry)
        
        # Calculate optimal number of stages
        self._calculate_stage_configuration()
        
        logger.info(
            f"CompressorStorage '{self.component_id}': "
            f"{self.inlet_pressure_bar:.0f} → {self.outlet_pressure_bar:.0f} bar, "
            f"{self.num_stages} stages, ratio={self.stage_pressure_ratio:.2f}, "
            f"T_in={self.inlet_temperature_c:.0f}°C, T_max={self.max_temperature_c:.0f}°C"
        )

    def step(self, t: float) -> None:
        """Execute timestep logic."""
        super().step(t)
        
        # Reset step variables
        self.actual_mass_transferred_kg = 0.0
        self.energy_consumed_kwh = 0.0
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.heat_removed_kwh = 0.0
        
        if self.transfer_mass_kg > 0:
            self.mode = CompressorMode.LP_TO_HP

            
            # 1. Determine actual mass to transfer (limited by max flow rate)
            max_transfer = self.max_flow_kg_h * self.dt
            self.actual_mass_transferred_kg = min(self.transfer_mass_kg, max_transfer)
            
            # 2. Check for trivial/no compression
            if self.outlet_pressure_bar <= self.inlet_pressure_bar:
                self._calculate_trivial_pass_through()
            else:
                # 3. Calculate compression physics
                self._calculate_compression_physics()
            
            # 3. Total energy for this step
            self.energy_consumed_kwh = (
                self.compression_work_kwh + self.chilling_work_kwh
            )
            
            # 4. Update cumulative statistics
            self.cumulative_energy_kwh += self.energy_consumed_kwh
            self.cumulative_energy_kwh += self.energy_consumed_kwh
            self.cumulative_mass_kg += self.actual_mass_transferred_kg
            
            # Calculate instantaneous power (average over step)
            if self.dt > 0:
                self.power_kw = self.energy_consumed_kwh / self.dt  # kWh / h = kW
            else:
                self.power_kw = 0.0
            
            # Reset input for next step
            self.transfer_mass_kg = 0.0
        else:
            self.mode = CompressorMode.IDLE
            self.power_kw = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Return current state for checkpointing/monitoring."""
        cumulative_specific = 0.0
        if self.cumulative_mass_kg > 0:
            cumulative_specific = self.cumulative_energy_kwh / self.cumulative_mass_kg
        
        return {
            **super().get_state(),
            'mode': int(self.mode),
            'num_stages': int(self.num_stages),
            'stage_pressure_ratio': float(self.stage_pressure_ratio),
            'transfer_mass_kg': float(self.transfer_mass_kg),
            'actual_mass_transferred_kg': float(self.actual_mass_transferred_kg),
            'compression_work_kwh': float(self.compression_work_kwh),
            'chilling_work_kwh': float(self.chilling_work_kwh),
            'energy_consumed_kwh': float(self.energy_consumed_kwh),
            'heat_removed_kwh': float(self.heat_removed_kwh),
            'specific_energy_kwh_kg': float(self.specific_energy_kwh_kg),
            'cumulative_energy_kwh': float(self.cumulative_energy_kwh),
            'cumulative_mass_kg': float(self.cumulative_mass_kg),
            'cumulative_specific_kwh_kg': float(cumulative_specific),
            'inlet_pressure_bar': float(self.inlet_pressure_bar),
            'outlet_pressure_bar': float(self.outlet_pressure_bar),
            'inlet_temperature_c': float(self.inlet_temperature_c),
            'max_temperature_c': float(self.max_temperature_c)
        }

    def _calculate_stage_configuration(self) -> None:
        """
        Determine optimal number of compression stages.
        
        Logic strictly copied from 'calculate_compression_energy' in legacy code.
        Based on maximum allowable discharge temperature per stage.
        
        Note: Uses direct CoolProp call for inverse lookup (P from S,T)
        since LUTManager doesn't support this yet. This is a one-time
        calculation during initialization, so performance impact is negligible.
        """
        if not COOLPROP_AVAILABLE:
            logger.warning(
                "CoolProp not available. Using fallback stage calculation "
                "(may not match legacy behavior exactly)"
            )
            # Fallback: use typical pressure ratio
            self._calculate_stages_fallback()
            return
        
        lut = self.get_registry_safe(ComponentID.LUT_MANAGER)
        
        # If LUT not available, use fallback
        if lut is None:
            logger.warning(
                "LUT Manager not available. Using fallback stage calculation."
            )
            self._calculate_stages_fallback()
            return
        
        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
        
        # Get inlet entropy (using LUT)
        s1 = lut.lookup('H2', 'S', p_in_pa, self.inlet_temperature_k)
        
        # Find pressure that reaches T_max isentropically
        # This requires inverse lookup: given (S, T_max), find P
        # LUT doesn't support this, so use CoolProp directly (one-time calc)
        try:
            if CoolPropLUT:
                p_out_1s_max_t = CoolPropLUT.PropsSI(
                    'P', 'S', s1, 'T', self.max_temperature_k, 'H2'
                )
            elif COOLPROP_AVAILABLE:
                p_out_1s_max_t = CP.PropsSI(
                    'P', 'S', s1, 'T', self.max_temperature_k, 'H2'
                )
            else:
                raise RuntimeError("CoolProp unavailable")
        except Exception as e:
            logger.warning(f"CoolProp inverse lookup failed: {e}. Using fallback.")
            self._calculate_stages_fallback()
            return
        
        # Calculate maximum stage pressure ratio (from legacy)
        r_stage_max_isentropic = p_out_1s_max_t / p_in_pa
        r_stage_max_isentropic = max(2.0, r_stage_max_isentropic)  # Safety floor
        
        # Calculate total ratio and number of stages
        r_total = p_out_pa / p_in_pa
        
        # Exact logic from legacy code
        n_stages = int(np.ceil(np.log(r_total) / np.log(r_stage_max_isentropic)))
        self.num_stages = max(1, n_stages)
        
        # Actual stage pressure ratio for equal distribution
        self.stage_pressure_ratio = r_total ** (1.0 / self.num_stages)
    
    def _calculate_trivial_pass_through(self) -> None:
        """Handle case where outlet pressure <= inlet pressure (no compression)."""
        self.compression_work_kwh = 0.0
        self.chilling_work_kwh = 0.0
        self.specific_energy_kwh_kg = 0.0
        self.heat_removed_kwh = 0.0

    def _calculate_stages_fallback(self) -> None:
        """Fallback stage calculation if CoolProp unavailable."""
        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
        r_total = p_out_pa / p_in_pa
        
        # Use typical max ratio of 4 (conservative)
        r_stage_max = 4.0
        n_stages = int(np.ceil(np.log(r_total) / np.log(r_stage_max)))
        self.num_stages = max(1, n_stages)
        self.stage_pressure_ratio = r_total ** (1.0 / self.num_stages)

    def _calculate_compression_fallback(self) -> None:
        """
        Simplified compression energy calculation when LUT is unavailable.
        Uses ideal gas approximation for hydrogen.
        """
        # Hydrogen properties (ideal gas approximation)
        gamma = 1.41  # Cp/Cv for H2
        cp = 14300.0  # J/(kg·K) - specific heat at constant pressure
        
        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
        r_total = p_out_pa / p_in_pa
        
        # Simplified adiabatic work per stage with isentropic efficiency
        # W = cp * T1 * [(P2/P1)^((gamma-1)/gamma) - 1] / eta_is
        exponent = (gamma - 1) / gamma
        
        w_compression_total = 0.0
        q_removed_total = 0.0
        t_current = self.inlet_temperature_k
        
        for i in range(self.num_stages):
            # Temperature rise for isentropic compression
            t_out_isentropic = t_current * (self.stage_pressure_ratio ** exponent)
            
            # Actual temperature rise (considering efficiency)
            delta_t_ideal = t_out_isentropic - t_current
            delta_t_actual = delta_t_ideal / self.isentropic_efficiency
            t_out_actual = t_current + delta_t_actual
            
            # Work for this stage
            w_stage = cp * delta_t_actual  # J/kg
            w_compression_total += w_stage
            
            # Cooling work (cool back to inlet temperature)
            if i < self.num_stages - 1:
                q_stage = cp * (t_out_actual - self.inlet_temperature_k)
                q_removed_total += q_stage
                t_current = self.inlet_temperature_k  # Reset for next stage
            else:
                t_current = t_out_actual  # Final stage temperature
        
        # Convert to kWh/kg
        compress_kwh_kg = w_compression_total * self.J_TO_KWH
        
        # Chilling energy
        q_chiller_j_kg = q_removed_total / self.chiller_cop
        self.chilling_energy_kwh_kg = q_chiller_j_kg * self.J_TO_KWH
        
        # Specific energy includes both (matching main physics method)
        self.specific_energy_kwh_kg = compress_kwh_kg + self.chilling_energy_kwh_kg
        
        # Calculate actual energy for this step using actual mass
        self.compression_work_kwh = compress_kwh_kg * self.actual_mass_transferred_kg
        self.chilling_work_kwh = self.chilling_energy_kwh_kg * self.actual_mass_transferred_kg
        
        # Note: energy_consumed_kwh is calculated in step() from work components

    def _calculate_compression_physics(self) -> None:
        """
        Calculate compression energy and work distribution.
        
        Logic strictly copied from 'calculate_compression_energy' loop in legacy code.
        Uses LUT for most lookups, CoolProp for inverse lookup (H from P,S).
        """
        lut = self.get_registry_safe(ComponentID.LUT_MANAGER)
        
        # If LUT not available, use simplified energy calculation
        if lut is None:
            self._calculate_compression_fallback()
            return
        
        p_in_pa = self.inlet_pressure_bar * self.BAR_TO_PA
        p_out_pa = self.outlet_pressure_bar * self.BAR_TO_PA
        
        # Inlet properties
        h1 = lut.lookup('H2', 'H', p_in_pa, self.inlet_temperature_k)
        s1 = lut.lookup('H2', 'S', p_in_pa, self.inlet_temperature_k)
        
        # Stage pressure ratio
        r_total = p_out_pa / p_in_pa
        r_stage = r_total ** (1.0 / self.num_stages)
        
        # Accumulators
        w_compression_total = 0.0  # J/kg
        q_removed_total = 0.0       # J/kg
        p_current = p_in_pa
        
        # Multi-stage compression loop (from legacy)
        # We model each stage independently:
        # Inlet -> Isentropic Compression -> Outlet -> Cooley -> Next Inlet
        
        # Current stage inlet temperature (initially global inlet temp)
        t_stage_in = self.inlet_temperature_k
        
        for i in range(self.num_stages):
            # 1. Determine local stage inlet conditions
            # If i > 0, gas was cooled to self.inlet_temperature_k (t_stage_in) at P_current
            # We must recalculate Entropy at this new state for the next compression step
            # This fixes the "constant entropy across all stages" inaccuracy
            
            s_stage_in = lut.lookup('H2', 'S', p_current, t_stage_in)
            h_stage_in = lut.lookup('H2', 'H', p_current, t_stage_in)
            
            # 2. Determine stage outlet pressure
            p_out_stage = p_current * r_stage
            if i == self.num_stages - 1:
                p_out_stage = p_out_pa  # Exact final pressure
            
            # 3. Isentropic compression
            # H2s = H(P_out, S_in)
            try:
                h2s = lut.lookup_isentropic_enthalpy('H2', p_out_stage, s_stage_in)
            except Exception:
                # Fallback if LUT fails
                if CoolPropLUT:
                    h2s = CoolPropLUT.PropsSI('H', 'P', p_out_stage, 'S', s_stage_in, 'H2')
                elif COOLPROP_AVAILABLE:
                    h2s = CP.PropsSI('H', 'P', p_out_stage, 'S', s_stage_in, 'H2')
                else:
                    # Rough fallback
                    h2s = h_stage_in * (p_out_stage/p_current)**0.28 
            
            # 4. Actual work accounting for efficiency
            ws = h2s - h_stage_in
            wa = ws / self.isentropic_efficiency
            h2a = h_stage_in + wa
            w_compression_total += wa
            
            # 5. Inter-cooling (if not last stage)
            if i < self.num_stages - 1:
                # Cool gas back to inlet temperature at elevated pressure
                h_cooled_next = lut.lookup('H2', 'H', p_out_stage, self.inlet_temperature_k)
                q_removed = h2a - h_cooled_next
                q_removed_total += q_removed
                
                # Update for next stage
                p_current = p_out_stage
                t_stage_in = self.inlet_temperature_k # Cooled back to target
        
        # Calculate chilling work (heat removed / COP)
        w_chilling_total = q_removed_total / self.chiller_cop
        
        # Total specific work (J/kg)
        w_total_j_kg = w_compression_total + w_chilling_total
        
        # Convert to kWh/kg and kWh for this mass
        self.specific_energy_kwh_kg = w_total_j_kg * self.J_TO_KWH
        
        # Calculate totals for this step
        self.compression_work_kwh = (w_compression_total * self.J_TO_KWH * 
                                     self.actual_mass_transferred_kg)
        self.chilling_work_kwh = (w_chilling_total * self.J_TO_KWH * 
                                  self.actual_mass_transferred_kg)
        self.heat_removed_kwh = (q_removed_total * self.J_TO_KWH * 
                                 self.actual_mass_transferred_kg)
    
    # ========== Port Interface Methods ==========
    
    def get_output(self, port_name: str) -> Any:
        """Get output from specific port."""
        if port_name == 'h2_out':
            # Return compressed hydrogen stream
            return Stream(
                mass_flow_kg_h=(self.actual_mass_transferred_kg / self.dt 
                               if self.dt > 0 else 0.0),
                temperature_k=self.inlet_temperature_k,  # Cooled to inlet temp
                pressure_pa=self.outlet_pressure_bar * self.BAR_TO_PA,
                composition={'H2': 1.0},
                phase='gas'
            )
        else:
            raise ValueError(
                f"Unknown output port '{port_name}' on {self.component_id}"
            )
    
    def receive_input(self, port_name: str, value: Any, resource_type: str) -> float:
        """Receive input into port."""
        if port_name == 'h2_in':
            if isinstance(value, Stream):
                available_mass = value.mass_flow_kg_h * self.dt
                max_capacity = self.max_flow_kg_h * self.dt
                accepted_mass = min(available_mass, max_capacity)
                
                self.transfer_mass_kg = accepted_mass
                return accepted_mass
        
        elif port_name == 'electricity_in':
            # Assume grid provides sufficient power
            return value if isinstance(value, (int, float)) else 0.0
        
        return 0.0
    
    def get_ports(self) -> Dict[str, Dict[str, str]]:
        """Return port metadata."""
        return {
            'h2_in': {
                'type': 'input',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            },
            'electricity_in': {
                'type': 'input',
                'resource_type': 'electricity',
                'units': 'MW'
            },
            'h2_out': {
                'type': 'output',
                'resource_type': 'hydrogen',
                'units': 'kg/h'
            }
        }
