import numpy as np
from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream

# default constants from legacy model
DEFAULT_BED_DIAMETER_M = 0.320
DEFAULT_BED_LENGTH_M = 0.800
DEFAULT_PARTICLE_DIAMETER_M = 0.0025
DEFAULT_BED_POROSITY = 0.40
DEFAULT_ADSORBENT_DENSITY_KG_M3 = 700.0
DEFAULT_CYCLE_TIME_HOURS = 6.0
DEFAULT_REGEN_TEMP_K = 250.0 + 273.15  # 523.15 K

# Material Properties
MM_H2 = 2.016e-3  # kg/mol
MM_H2O = 18.015e-3  # kg/mol
R_UNIVERSAL = 8.314
CP_ADSORBENT_J_KG_K = 900.0
HEAT_ADSORPTION_J_KG = 2000.0 * 1000.0  # 2000 kJ/kg -> J/kg

class TSAUnit(Component):
    """
    Thermal Swing Adsorption (TSA) Unit for Hydrogen Purification.
    Removes water vapor using molecular sieve adsorbents.
    
    Model based on simplified 1D static simulation (legacy/TSA/modelo/modelo tsa.py),
    enhanced with dynamic property lookups and purge gas energy accounting.
    """
    
    def __init__(self, 
                 component_id: str,
                 bed_diameter_m: float = DEFAULT_BED_DIAMETER_M,
                 bed_length_m: float = DEFAULT_BED_LENGTH_M,
                 num_beds: int = 2,
                 particle_diameter_m: float = DEFAULT_PARTICLE_DIAMETER_M,
                 porosity: float = DEFAULT_BED_POROSITY,
                 regen_temp_k: float = DEFAULT_REGEN_TEMP_K,
                 cycle_time_hours: float = DEFAULT_CYCLE_TIME_HOURS,
                 working_capacity: float = 0.05,  # kg_H2O/kg_ads
                 purge_fraction: float = 0.02):   # Fraction of feed used for purge (tuned for ~20% extra energy)
        
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
        
        # Calculated Geometry
        self.bed_area = np.pi * (self.bed_diameter ** 2) / 4.0
        self.adsorbent_mass_per_bed = (self.bed_area * self.bed_length * 
                                      DEFAULT_ADSORBENT_DENSITY_KG_M3)
        self.max_water_loading_kg = self.adsorbent_mass_per_bed * self.working_capacity
        
        # State
        self.active_bed_idx = 0  # 0 or 1
        self.water_loading_kg = 0.0
        self.cycle_timer_h = 0.0
        self.total_h2_purified_kg = 0.0
        self.total_water_removed_kg = 0.0
        self.is_saturated = False
        
        # Performance Metrics (Last Step)
        self.last_pressure_drop_bar = 0.0
        self.last_regen_energy_kw = 0.0
        
        # Inputs
        self.inlet_stream: Optional[Stream] = None
        
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        
    def step(self, t: float) -> None:
        super().step(t)
        
        if self.inlet_stream is None:
            return

        # 1. Properties Calculation (Dynamic)
        stream = self.inlet_stream
        
        y_h2o = stream.composition.get('H2O', 0.0)
        y_h2 = stream.composition.get('H2', 1.0 - y_h2o)
        
        mm_mix = y_h2o * MM_H2O + (1.0 - y_h2o) * MM_H2
        
        # Ideal gas density
        rho_gas = (stream.pressure_pa * mm_mix) / (R_UNIVERSAL * stream.temperature_k)
        
        # Dynamic Viscosity Approx (Power Law)
        mu_h2 = 1.05e-5 * (stream.temperature_k / 277.15) ** 0.7 
        
        # 2. Flow Rates
        # Molar flow for composition handling
        q_molar_total = (stream.mass_flow_kg_h / 3600.0) / mm_mix
        q_mass_h2o_s = q_molar_total * y_h2o * MM_H2O
        q_mass_h2_s = (stream.mass_flow_kg_h / 3600.0) - q_mass_h2o_s
        
        # 3. Adsorption Physics (Active Bed)
        # Check saturation
        if self.water_loading_kg >= self.max_water_loading_kg:
            self.is_saturated = True
            water_removed_this_step_kg = 0.0
        else:
            self.is_saturated = False
            # Assume 100% removal efficiency until saturation (Simplified)
            water_removed_this_step_kg = q_mass_h2o_s * (self.dt * 3600.0)
            
        self.water_loading_kg += water_removed_this_step_kg
        self.total_water_removed_kg += water_removed_this_step_kg
        self.total_h2_purified_kg += q_mass_h2_s * (self.dt * 3600.0)
        
        # Ergun Pressure Drop
        # Superficial Velocity
        G = (stream.mass_flow_kg_h / 3600.0) / self.bed_area
        u_s = G / rho_gas
        
        # Ergun Terms
        term_viscous = 150.0 * ((1.0 - self.porosity)**2 / (self.porosity**3)) * \
                      (mu_h2 * u_s / (self.particle_diameter**2))
                      
        term_kinetic = 1.75 * ((1.0 - self.porosity) / (self.porosity**3)) * \
                      (rho_gas * (u_s**2) / self.particle_diameter)
                      
        delta_p_pa = self.bed_length * (term_viscous + term_kinetic)
        self.last_pressure_drop_bar = delta_p_pa / 1e5
        
        # 4. Regeneration Energy (Inactive Bed)
        # Energy = Sensible + Desorption + Purge Heating
        eta = 0.85
        
        # A. Sensible Heat (Bed Mass)
        # Distributed over cycle time to get avg power
        q_sensible_cycle_j = self.adsorbent_mass_per_bed * CP_ADSORBENT_J_KG_K * \
                            (self.regen_temp_k - stream.temperature_k)
                            
        # B. Desorption Heat (Water Mass - using Capacity or Actual Loading?)
        # Use Actual Loading from previous cycle (Approximated by current removal rate * cycle)
        # This keeps power proportional to load.
        cycle_water_kg = q_mass_h2o_s * (self.cycle_time_hours * 3600.0)
        q_desorption_cycle_j = cycle_water_kg * HEAT_ADSORPTION_J_KG
        
        # C. Purge Gas Heating (Fraction of Feed)
        # Q = m_purge * Cp * dT
        cp_h2 = 14300.0 # J/kgK
        mass_surge_cycle_kg = (stream.mass_flow_kg_h * self.cycle_time_hours) * self.purge_fraction
        q_purge_cycle_j = mass_surge_cycle_kg * cp_h2 * (self.regen_temp_k - stream.temperature_k)
                         
        total_energy_cycle_j = (q_sensible_cycle_j + q_desorption_cycle_j + q_purge_cycle_j) / eta
        
        # Average Power
        avg_power_w = total_energy_cycle_j / (self.cycle_time_hours * 3600.0)
        self.last_regen_energy_kw = avg_power_w / 1000.0
        
        # 5. Cycle Logic
        self.cycle_timer_h += self.dt
        if self.cycle_timer_h >= self.cycle_time_hours:
            # Switch beds
            # Reset active bed to 0 loading (it was regenerating)
            # The currently active bed becomes the regenerating one
            self.active_bed_idx = 1 - self.active_bed_idx
            self.cycle_timer_h = 0.0
            self.water_loading_kg = 0.0 
            self.is_saturated = False
            
    def receive_input(self, port_name: str, value: Any, resource_type: str = 'stream') -> None:
        if port_name == 'wet_h2_in' and isinstance(value, Stream):
            self.inlet_stream = value
            
    def get_output(self, port_name: str) -> Any:
        if self.inlet_stream is None:
            return None
            
        if port_name == 'dry_h2_out':
            # Clone inlet but remove water and reduce pressure
            out = Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h * (1.0 - self.inlet_stream.composition.get('H2O', 0.0)), # Approx
                temperature_k=self.inlet_stream.temperature_k, # Assume roughly isothermal for product
                pressure_pa=self.inlet_stream.pressure_pa - (self.last_pressure_drop_bar * 1e5),
                composition={'H2': 1.0, 'H2O': 0.0},
                phase='gas'
            )
            # Correct mass flow strictly
            y_h2o = self.inlet_stream.composition.get('H2O', 0.0)
            mm_mix = y_h2o * MM_H2O + (1.0 - y_h2o) * MM_H2
            molar_flow = (self.inlet_stream.mass_flow_kg_h/3600.0) / mm_mix
            h2_molar_flow = molar_flow * (1.0 - y_h2o)
            h2_mass_flow = h2_molar_flow * MM_H2 * 3600.0
            out.mass_flow_kg_h = h2_mass_flow
            return out
            
        elif port_name == 'water_out':
            # Calculate removed water stream
            # (In reality this leaves during regeneration, but we output it for mass balance)
            y_h2o = self.inlet_stream.composition.get('H2O', 0.0)
            if y_h2o == 0: return Stream(0, 300, 1e5, {'H2O':1}, 'liquid')
            
            # Logic similar to dry_h2_out
            return Stream(
                mass_flow_kg_h=self.inlet_stream.mass_flow_kg_h - self.get_output('dry_h2_out').mass_flow_kg_h,
                temperature_k=self.inlet_stream.temperature_k,
                pressure_pa=1e5, # Vented?
                composition={'H2O': 1.0},
                phase='liquid'
            )
            
        return None
        
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            'bed_diameter_m': self.bed_diameter,
            'pressure_drop_bar': self.last_pressure_drop_bar,
            'regen_power_kw': self.last_regen_energy_kw,
            'water_removed_total_kg': self.total_water_removed_kg,
            'current_water_loading_kg': self.water_loading_kg
        }
