"""
Physics constants configuration loader.

This module loads physical parameters for the PEM electrolyzer and balance of plant (BoP)
from external configuration files, falling back to safe defaults if necessary.
It centralizes the source of truth for simulation parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any
from h2_plant.config.physics_loader import load_physics_parameters

# Load configuration once at module level
_CONFIG = load_physics_parameters()

def _get_val(section: str, key: str, default: Any) -> Any:
    """
    Retrieve value from configuration dictionary with type safety mechanisms.

    Args:
        section (str): Configuration section (e.g., 'pem', 'physical_constants').
        key (str): Parameter name.
        default (Any): Fallback value if key is missing or invalid.

    Returns:
        Any: Type-casted configuration value or default.
    """
    try:
        value = _CONFIG.get(section, {}).get(key, default)
        # If value exists in config but is wrong type, convert to match default
        if value != default and default is not None:
            default_type = type(default)
            if not isinstance(value, default_type):
                try:
                    # Try to convert to default type
                    return default_type(value)
                except (ValueError, TypeError):
                    # If conversion fails, use default
                    return default
        return value
    except (AttributeError, TypeError):
        return default

@dataclass(frozen=True)
class PEMConstants:
    """
    Physical constants for PEM electrolyzer model.
    
    Encapsulates electrochemical parameters, stack geometry, and degradation
    curves required for the polarization curve model involved in V-I calculations.
    """
    # Physical Constants
    F: float = _get_val('physical_constants', 'F', 96485.33)      # Faraday Constant (C/mol)
    R: float = _get_val('physical_constants', 'R', 8.314)         # Ideal Gas Constant (J/(mol·K))
    P_ref: float = _get_val('physical_constants', 'P_ref', 1.0e5) # Reference Pressure (Pa)
    z: int = _get_val('physical_constants', 'z', 2)               # Electrons per H2 molecule (2 for H2)
    MH2: float = _get_val('physical_constants', 'MH2', 2.016e-3)  # Molar Mass H2 (kg/mol)
    MO2: float = _get_val('physical_constants', 'MO2', 31.998e-3) # Molar Mass O2 (kg/mol)
    MH2O: float = _get_val('physical_constants', 'MH2O', 18.015e-3) # Molar Mass H2O (kg/mol)
    LHVH2_kWh_kg: float = _get_val('physical_constants', 'LHVH2_kWh_kg', 33.33) # Lower Heating Value
    
    # Geometry
    N_stacks: int = _get_val('pem', 'N_stacks', 35)               # Total number of stacks
    # Updated to 85 cells per stack per user specification
    N_cell_per_stack: int = _get_val('pem', 'N_cell_per_stack', 85)
    A_cell_cm2: float = _get_val('pem', 'A_cell_cm2', 300.0)      # Active area per cell
    
    @property
    def Area_Total(self) -> float:
        """
        Total active area in cm².
        
        Formula: Area_Total = N_stacks * N_cells_per_stack * Area_cell
        WARNING: Unit is cm², consistent with electrochemical literature.
        """
        return self.N_stacks * self.N_cell_per_stack * self.A_cell_cm2
    
    # Electrochemistry
    # Note: j parameters commonly defined in A/cm² in empirical models
    delta_mem: float = _get_val('pem', 'delta_mem', 100 * 1e-4)  # Membrane thickness (cm)
    sigma_base: float = _get_val('pem', 'sigma_base', 0.1)       # Proton conductivity (S/cm)
    j0: float = _get_val('pem', 'j0', 1.0e-6)                    # Exchange current density (A/cm²)
    alpha: float = _get_val('pem', 'alpha', 0.5)                 # Charge transfer coefficient (symmetric)
    j_lim: float = _get_val('pem', 'j_lim', 4.0)                 # Limiting current density (A/cm²)
    j_nom: float = _get_val('pem', 'j_nom', 2.91)                # Nominal current density (A/cm²)
    
    # Operating Conditions
    T_default: float = _get_val('pem_system', 'T_default', 333.15)      # Default temperature (K)
    P_op_default: float = _get_val('pem_system', 'P_op_default', 40.0e5)  # Default pressure (Pa)
    
    # Balance of Plant (BoP) and System Power
    floss: float = _get_val('pem_system', 'floss', 0.02)                          # Fluid loss factor
    k_bop_var: float = _get_val('pem_system', 'k_bop_var', 0.04)                  # Variable BoP power fraction
    water_excess_factor: float = _get_val('pem_system', 'water_excess_factor', 0.02)
    P_nominal_sistema_kW: float = _get_val('pem_system', 'P_nominal_sistema_kW', 5000.0)
    
    # === Output Stream Conditions (from physics_parameters.yaml) ===
    # Temperature and pressure at PEM outlet
    output_temperature_c: float = _get_val('pem_system', 'output_temperature_c', 60.0)
    output_pressure_bar: float = _get_val('pem_system', 'output_pressure_bar', 40.0)
    
    @property
    def output_temperature_k(self) -> float:
        """Output temperature in Kelvin."""
        return self.output_temperature_c + 273.15
    
    @property
    def output_pressure_pa(self) -> float:
        """Output pressure in Pascals."""
        return self.output_pressure_bar * 1e5
    
    # === Gas Phase Composition (MOLAR PPM / fractions) ===
    h2_purity_molar: float = _get_val('pem_system', 'h2_purity_molar', 0.994800) # Updated to ~99.48% accounting for sat. water
    o2_crossover_ppm_molar: float = _get_val('pem_system', 'o2_crossover_ppm_molar', 200.0) # 200 ppm
    anode_h2_crossover_ppm_molar: float = _get_val('pem_system', 'anode_h2_crossover_ppm_molar', 4000.0) # 4000 ppm
    
    # Vapor saturation at 60C/40bar is approx 0.5% (5000 ppm)
    # We keep this constant for reference, but code should calculate dynamically if possible
    h2o_vapor_ppm_molar: float = _get_val('pem_system', 'h2o_vapor_ppm_molar', 5000.0) 
    
    # === Entrained Liquid Water ===
    # Cathode: Net electro-osmotic drag after back-diffusion.
    # Raw drag is ~2.5 H2O/H+ (~45 kg H2O per kg H2), but most returns via back-diffusion.
    # Net effective drag is typically 5-15% of stoichiometric water consumption.
    # Using 0.10 (10% of consumption) for realistic mass balance.
    cathode_liquid_water_factor: float = _get_val('pem_system', 'cathode_liquid_water_factor', 0.10)
    
    # Anode: Carries the bulk cooling flow. 
    # Assumed Delta T for cooling loop sizing if not specified dynamically
    cooling_delta_t_k: float = _get_val('pem_system', 'cooling_delta_t_k', 5.0)

    entrained_water_fraction: float = _get_val('pem_system', 'entrained_water_fraction', 0.15) # Legacy, keeping for compat
    water_reuse_fraction: float = _get_val('pem_system', 'water_reuse_fraction', 0.95)  # 95% reused
    demister_limit_mg_nm3: float = _get_val('pem_system', 'demister_limit_mg_nm3', 20.0)
    
    # Legacy aliases for backward compatibility
    @property
    def unreacted_water_fraction(self) -> float:
        """Alias for entrained_water_fraction (deprecated name)."""
        return self.entrained_water_fraction
    
    @property
    def o2_crossover_molar_ppm(self) -> float:
        """Alias for o2_crossover_ppm_molar."""
        return self.o2_crossover_ppm_molar
    
    
    @property
    def P_nominal_sistema_W(self) -> float:
        """Nominal system power capacity in Watts."""
        return self.P_nominal_sistema_kW * 1000.0
    
    @property
    def P_bop_fixo(self) -> float:
        """
        Fixed BoP power consumption.
        
        Assumed constant overhead (e.g., control systems, lighting).
        Currently estimated as 2.5% of nominal system power.
        """
        return 0.025 * self.P_nominal_sistema_W
    
    # Degradation Tables (Reference Data)
    # Maps operational years to expected stack voltage rise due to degradation.
    DEGRADATION_TABLE_YEARS: tuple = (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0)
    DEGRADATION_TABLE_V_STACK: tuple = (171, 172, 176, 178, 178, 180, 181, 183, 184, 187, 190, 193, 197)
    
    @property
    def DEGRADATION_TABLE_V_CELL(self) -> tuple:
        """
        Degradation table normalized to V/cell.
        
        Used for scalable performance calculations independent of stack count.
        """
        return tuple(v / self.N_cell_per_stack for v in self.DEGRADATION_TABLE_V_STACK)
    
    # Backward compatibility aliases
    @property
    def DEGRADATION_YEARS(self) -> tuple:
        return self.DEGRADATION_TABLE_YEARS
    
    @property
    def DEGRADATION_V_STACK(self) -> tuple:
        return self.DEGRADATION_TABLE_V_STACK
    
    # Simulation Time Parameters
    H_MES: float = 730.0       # Average operational hours per month
    H_SIM_YEARS: float = 10.0  # Simulation horizon
    
    @property
    def H_SIM_TOTAL_PRECALC(self) -> int:
        """Total simulation duration in hours."""
        return int(self.H_SIM_YEARS * 8760)

@dataclass(frozen=True)
class SOECConstants:
    """
    Constants for Solid Oxide Electrolyzer Cell (SOEC) operations.
    
    Parameters defining the high-temperature electrolysis performance.
    """
    # Performance
    SPECIFIC_ENERGY_KWH_KG: float = _get_val('soec', 'specific_energy_kwh_kg', 37.5) # Efficiency metric
    STEAM_CONSUMPTION_KG_PER_MWH: float = _get_val('soec', 'steam_consumption_kg_per_mwh', 280.0)
    
    # Operational Limits and Grid Integration
    NUM_MODULES: int = 6
    MAX_POWER_NOMINAL_MW: float = 2.4
    LIMIT_OPTIMAL_RATIO: float = _get_val('soec', 'limit_optimal_ratio', 0.80) # Optimal efficiency point
    POWER_FIRST_STEP_MW: float = _get_val('soec', 'power_first_step_mw', 0.12) # Minimum stable load
    POWER_STANDBY_MW: float = 0.0                                              # Hot standby power
    RAMP_STEP_MW: float = _get_val('soec', 'ramp_step_mw', 0.24)               # Max ramp rate per step
    UNREACTED_WATER_FRACTION: float = _get_val('soec', 'unreacted_water_fraction', 0.10)
    ROTATION_PERIOD_MIN: int = 10
    V_CUTOFF_EOL: float = 2.2 # End-of-Life Voltage limit

@dataclass(frozen=True)
class HybridConstants:
    """Economic parameters for hybrid system dispatch optimization."""
    PEM_MAX_MW: float = 5.0
    PH2_EQUIVALENT_EUR_MWh: float = 288.29   # H2 value equivalent
    PPPA_EUR_MWh: float = 50.0               # Power Purchase Agreement price
    ARBITRAGE_LIMIT_EUR_MWh: float = 338.29  # Grid price limit for arbitrage

@dataclass(frozen=True)
class WaterConstants:
    """
    Constants for Water Treatment System.
    
    Defines parameters for Reverse Osmosis (RO) and ultrapure water storage.
    """
    WATER_AMBIENT_T_K: float = 293.15  # 20°C
    WATER_ATM_P_PA: float = 101325.0   # 1 atm
    
    # Reverse Osmosis / Purification
    WATER_RO_RECOVERY_RATIO: float = _get_val('water', 'ro_recovery_ratio', 0.75)
    WATER_RO_SPEC_ENERGY_KWH_KG: float = _get_val('water', 'ro_spec_energy_kwh_kg', 0.004) # 4 kWh/m3
    WATER_PURIFIER_MAX_FLOW_KGH: float = _get_val('water', 'purifier_max_flow_kgh', 5000.0)
    WATER_PURIFIER_PRESSURE_DROP_PA: float = 5e4  # 0.5 bar
    
    # Storage Tank
    ULTRAPURE_TANK_CAPACITY_KG: float = _get_val('water', 'tank_capacity_kg', 10000.0)
    ULTRAPURE_TANK_OUTLET_MAX_KGH: float = 2000.0
    ULTRAPURE_TANK_LOW_FILL_RATIO: float = 0.2  # Trigger pump start


