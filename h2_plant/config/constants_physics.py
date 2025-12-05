from dataclasses import dataclass
from typing import Dict, Any
from h2_plant.config.physics_loader import load_physics_parameters

# Load configuration once at module level
_CONFIG = load_physics_parameters()

def _get_val(section: str, key: str, default: Any) -> Any:
    """Helper to get value from config or return default with type safety."""
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
    """Physical constants for PEM electrolyzer model."""
    # Physical Constants
    F: float = _get_val('physical_constants', 'F', 96485.33)
    R: float = _get_val('physical_constants', 'R', 8.314)
    P_ref: float = _get_val('physical_constants', 'P_ref', 1.0e5)
    z: int = _get_val('physical_constants', 'z', 2)
    MH2: float = _get_val('physical_constants', 'MH2', 2.016e-3)
    MO2: float = _get_val('physical_constants', 'MO2', 31.998e-3)
    MH2O: float = _get_val('physical_constants', 'MH2O', 18.015e-3)
    LHVH2_kWh_kg: float = _get_val('physical_constants', 'LHVH2_kWh_kg', 33.33)
    
    # Geometry
    N_stacks: int = _get_val('pem', 'N_stacks', 35)
    # Updated to 85 cells per stack per user specification
    # With degradation table: 1290V / 85 cells = 15.18V/cell at BOL+1yr
    N_cell_per_stack: int = _get_val('pem', 'N_cell_per_stack', 85)
    A_cell_cm2: float = _get_val('pem', 'A_cell_cm2', 300.0)
    
    @property
    def Area_Total(self) -> float:
        """Total active area in cm² (N_stacks × N_cell_per_stack × A_cell_cm2).
        
        WARNING: This is in cm², NOT m². For SI units, divide by 10000.
        """
        return self.N_stacks * self.N_cell_per_stack * self.A_cell_cm2
    
    # Electrochemistry (Note: j parameters in A/cm² - empirical, non-SI)
    delta_mem: float = _get_val('pem', 'delta_mem', 100 * 1e-4)  # Membrane thickness in cm
    sigma_base: float = _get_val('pem', 'sigma_base', 0.1)       # Conductivity in S/cm
    j0: float = _get_val('pem', 'j0', 1.0e-6)                    # Exchange current density (A/cm²)
    alpha: float = _get_val('pem', 'alpha', 0.5)                 # Charge transfer coefficient
    j_lim: float = _get_val('pem', 'j_lim', 4.0)                 # Limiting current density (A/cm²)
    j_nom: float = _get_val('pem', 'j_nom', 2.91)                # Nominal current density (A/cm²)
    
    # Operating Conditions
    T_default: float = _get_val('pem', 'T_default', 333.15)      # Default temperature (K) = 60°C
    P_op_default: float = _get_val('pem', 'P_op_default', 40.0e5)  # Default pressure (Pa) = 40 bar
    
    # BoP and System Power
    floss: float = _get_val('pem', 'floss', 0.02)
    k_bop_var: float = _get_val('pem', 'k_bop_var', 0.04)
    unreacted_water_fraction: float = _get_val('pem', 'unreacted_water_fraction', 0.03)
    P_nominal_sistema_kW: float = _get_val('pem', 'P_nominal_sistema_kW', 5000.0)  # 5 MW nominal
    
    @property
    def P_nominal_sistema_W(self) -> float:
        """Nominal system power in Watts"""
        return self.P_nominal_sistema_kW * 1000.0
    
    @property
    def P_bop_fixo(self) -> float:
        """Fixed BoP power (25% of nominal)"""
        return 0.025 * self.P_nominal_sistema_W
    
    # Degradation Tables (Years, V_stack) - UPDATED TO MATCH ALL_Reference
    # These are the TOTAL stack voltages at different operating years
    DEGRADATION_TABLE_YEARS: tuple = (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0)
    # FIXED: Using ALL_Reference values (171-197V) instead of old values (1290-1490V)
    DEGRADATION_TABLE_V_STACK: tuple = (171, 172, 176, 178, 178, 180, 181, 183, 184, 187, 190, 193, 197)
    
    @property
    def DEGRADATION_TABLE_V_CELL(self) -> tuple:
        """Degradation table in V/cell for backward compatibility"""
        return tuple(v / self.N_cell_per_stack for v in self.DEGRADATION_TABLE_V_STACK)
    
    # Backward compatibility aliases
    @property
    def DEGRADATION_YEARS(self) -> tuple:
        return self.DEGRADATION_TABLE_YEARS
    
    @property
    def DEGRADATION_V_STACK(self) -> tuple:
        return self.DEGRADATION_TABLE_V_STACK
    
    # Simulation Time Parameters
    H_MES: float = 730.0  # Hours per operational month for degradation model
    H_SIM_YEARS: float = 10.0  # Simulation years for pre-calculator
    
    @property
    def H_SIM_TOTAL_PRECALC(self) -> int:
        """Total hours for pre-calculation"""
        return int(self.H_SIM_YEARS * 8760)

@dataclass(frozen=True)
class SOECConstants:
    """Constants for SOEC cluster operation."""
    # Performance
    SPECIFIC_ENERGY_KWH_KG: float = _get_val('soec', 'specific_energy_kwh_kg', 37.5)
    STEAM_CONSUMPTION_KG_PER_MWH: float = _get_val('soec', 'steam_consumption_kg_per_mwh', 280.0)
    
    # Operational
    NUM_MODULES: int = 6
    MAX_POWER_NOMINAL_MW: float = 2.4
    LIMIT_OPTIMAL_RATIO: float = _get_val('soec', 'limit_optimal_ratio', 0.80)
    POWER_FIRST_STEP_MW: float = _get_val('soec', 'power_first_step_mw', 0.12)
    POWER_STANDBY_MW: float = 0.0
    RAMP_STEP_MW: float = _get_val('soec', 'ramp_step_mw', 0.24)
    UNREACTED_WATER_FRACTION: float = _get_val('soec', 'unreacted_water_fraction', 0.10)
    ROTATION_PERIOD_MIN: int = 10
    V_CUTOFF_EOL: float = 2.2

@dataclass(frozen=True)
class HybridConstants:
    """Economic and hybrid dispatch constants."""
    PEM_MAX_MW: float = 5.0
    PH2_EQUIVALENT_EUR_MWh: float = 288.29
    PPPA_EUR_MWh: float = 50.0
    ARBITRAGE_LIMIT_EUR_MWh: float = 338.29
