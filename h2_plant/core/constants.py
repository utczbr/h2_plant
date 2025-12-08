"""
Physical constants and operational parameters for hydrogen production system.
"""

from typing import Final

class GasConstants:
    """Universal gas constant and species-specific data."""
    R_UNIVERSAL_J_PER_MOL_K: Final[float] = 8.314
    R_H2: Final[float] = 4124.0  # Specific gas constant for H2, J/(kg.K)
    GAMMA_H2: Final[float] = 1.41 # Specific heat ratio for H2
    # Average specific heat for gas phase (J/kg·K) - used in chiller fallback
    CP_H2_AVG: Final[float] = 14300.0  # Hydrogen
    CP_O2_AVG: Final[float] = 918.0    # Oxygen
    SPECIES_DATA: Final[dict] = {
        'O2': {
            'molecular_weight': 32.0,
            'h_formation': 0.0,
            'cp_coeffs': [29.96, 4.18e-3, -1.67e-6, 0.0, 0.0],
            'critical_temp': 154.6,
            'critical_pressure': 5.043e6,
            'acentric_factor': 0.022
        },
        'CO2': {
            'molecular_weight': 44.01,
            'h_formation': -393.51e3,
            'cp_coeffs': [22.26, 5.98e-2, -3.50e-5, 7.47e-9, 0.0],
            'critical_temp': 304.13,
            'critical_pressure': 7.377e6,
            'acentric_factor': 0.225
        },
        'CH4': {
            'molecular_weight': 16.04,
            'h_formation': -74.87e3,
            'cp_coeffs': [19.89, 5.02e-2, 1.27e-5, -1.10e-8, 0.0],
            'critical_temp': 190.56,
            'critical_pressure': 4.599e6,
            'acentric_factor': 0.011
        },
        'H2O': {
            'molecular_weight': 18.015,
            'h_formation': -241.83e3,
            'h_vaporization': 44.01e3,
            'cp_coeffs': [32.24, 1.92e-3, 1.06e-5, -3.60e-9, 0.0],
            'antoine_coeffs': [8.07131, 1730.63, 233.426],
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

# Standard conditions
T_REF: Final[float] = 298.15  # K
P_REF: Final[float] = 101325  # Pa

class StandardConditions:
    TEMPERATURE_K: Final[float] = 298.15
    TEMPERATURE_C: Final[float] = 25.0
    PRESSURE_PA: Final[float] = 101325.0
    PRESSURE_BAR: Final[float] = 1.01325

class ConversionFactors:
    PA_TO_BAR: Final[float] = 1e-5
    BAR_TO_PA: Final[float] = 1e5
    PSI_TO_PA: Final[float] = 6894.76
    KWH_TO_J: Final[float] = 3.6e6
    J_TO_KWH: Final[float] = 1 / 3.6e6
    MWH_TO_KWH: Final[float] = 1000.0
    KG_TO_G: Final[float] = 1000.0
    KG_TO_LB: Final[float] = 2.20462
    MW_TO_KW: Final[float] = 1000.0
    KW_TO_W: Final[float] = 1000.0

class ProductionConstants:
    H2_ENERGY_CONTENT_LHV_KWH_PER_KG: Final[float] = 33.0
    H2_ENERGY_CONTENT_HHV_KWH_PER_KG: Final[float] = 39.4
    ELECTROLYSIS_THEORETICAL_ENERGY_KWH_PER_KG: Final[float] = 39.4
    ELECTROLYSIS_TYPICAL_EFFICIENCY: Final[float] = 0.65
    ATR_TYPICAL_EFFICIENCY: Final[float] = 0.75
    ATR_STARTUP_TIME_HOURS: Final[float] = 1.0
    ATR_COOLDOWN_TIME_HOURS: Final[float] = 0.5
    O2_TO_H2_MASS_RATIO: Final[float] = 7.94

class StorageConstants:
    LOW_PRESSURE_PA: Final[float] = 30e5
    HIGH_PRESSURE_PA: Final[float] = 350e5
    DELIVERY_PRESSURE_PA: Final[float] = 900e5
    TYPICAL_LP_CAPACITY_KG: Final[float] = 50.0
    TYPICAL_HP_CAPACITY_KG: Final[float] = 200.0
    TANK_FULL_THRESHOLD: Final[float] = 0.99
    TANK_EMPTY_THRESHOLD: Final[float] = 0.01

class CompressionConstants:
    ISENTROPIC_EFFICIENCY: Final[float] = 0.75
    MECHANICAL_EFFICIENCY: Final[float] = 0.95
    TYPICAL_STAGE_PRESSURE_RATIO: Final[float] = 3.5
    MAX_STAGES: Final[int] = 4

class EconomicConstants:
    ENERGY_PRICE_MIN: Final[float] = 20.0
    ENERGY_PRICE_MAX: Final[float] = 200.0
    ENERGY_PRICE_AVERAGE: Final[float] = 60.0
    H2_SELLING_PRICE: Final[float] = 5.0
    NG_PRICE_TYPICAL: Final[float] = 3.5

class SimulationDefaults:
    TIMESTEP_HOURS: Final[float] = 1.0
    ANNUAL_HOURS: Final[int] = 8760
    CHECKPOINT_INTERVAL_HOURS: Final[int] = 168
    MASS_TOLERANCE_KG: Final[float] = 1e-6
    PRESSURE_TOLERANCE_PA: Final[float] = 1e3
    TEMPERATURE_TOLERANCE_K: Final[float] = 0.01

class CoalescerConstants:
    """
    Coalescer cartridge filter performance constants.
    Source: CoalescerModel.py / coalescedor-1.pdf
    """
    # Geometry defaults
    D_SHELL_DEFAULT_M: Final[float] = 0.32  # Vessel diameter (32 cm)
    D_ELEM_DEFAULT_M: Final[float] = 0.20   # Element diameter (20 cm)
    L_ELEM_DEFAULT_M: Final[float] = 1.00   # Element length (100 cm)
    N_ELEM_DEFAULT: Final[int] = 1          # Elements per vessel

    # Physics / Performance (CoalescerModel.py lines 41-45)
    K_PERDA: Final[float] = 0.5e6           # Loss factor (empirical)
    ETA_LIQUID_REMOVAL: Final[float] = 0.9999  # 99.99% efficiency

    # Sutherland viscosity reference (CoalescerModel.py lines 41-42)
    MU_REF_H2_PA_S: Final[float] = 9.0e-6   # H2 viscosity at T_ref
    MU_REF_O2_PA_S: Final[float] = 2.1e-5   # O2 viscosity at T_ref (approx)
    T_REF_K: Final[float] = 303.15          # 30°C reference temperature

    # Worst case sizing (for reference)
    C_LIQ_IN_WORST_CASE_MG_M3: Final[float] = 100.0

class DeoxoConstants:
    """
    Constants for the Catalytic Deoxidizer (Deoxo) PFR.
    Sources: dimensionando_deoxer.pdf, deoxo-dim.py, modleo_do_deoxo.pdf
    """
    # Reactor Geometry
    L_REACTOR_M: Final[float] = 1.294
    D_REACTOR_M: Final[float] = 0.324
    AREA_REACTOR_M2: Final[float] = 0.0824  # pi * D^2 / 4
    CATALYST_POROSITY: Final[float] = 0.4
    PELLET_DIAMETER_M: Final[float] = 0.003
    
    # Kinetics (Arrhenius)
    # k_eff = k0_vol * exp(-Ea / RT)
    K0_VOL_S1: Final[float] = 1.0e10
    EA_J_MOL: Final[float] = 55000.0
    
    # Thermodynamics
    DELTA_H_RXN_J_MOL_O2: Final[float] = -242000.0  # Exothermic
    CP_MIX_AVG_J_MOL_K: Final[float] = 29.0       # Simplified approximation
    
    # Thermal Control
    U_A_W_M3_K: Final[float] = 5000.0  # Volumetric heat transfer coefficient
    T_JACKET_K: Final[float] = 120.0 + 273.15  # 120°C
    
    # Operational Limits
    MAX_ALLOWED_O2_OUT_MOLE_FRAC: Final[float] = 5.0e-6 # 5 ppm
    CRITICAL_INLET_T_K: Final[float] = 4.0 + 273.15     # Worst case

class DryCoolerConstants:
    """
    Physical parameters for Dry Cooler (Air-Cooled Heat Exchanger).
    Designed for cooling PEM electrolysis outlet streams (H2 and O2).
    Source: dry_cooler-1.pdf, drydim.py
    """
    # Areas (m²)
    AREA_H2_M2: Final[float] = 219.0
    AREA_O2_M2: Final[float] = 24251.05
    
    # Heat Transfer & Flow
    U_W_M2_K: Final[float] = 35.0  # Overall Heat Transfer Coefficient
    # Note: Text mentions 500 Pa, but Power/Flow relationship implies ~124 Pa
    # (872 W * 0.6) / (5.175 / 1.225) = 123.9 Pa
    DP_AIR_PA: Final[float] = 124.0  
    DP_FLUID_BAR: Final[float] = 0.05 # Process-side pressure drop
    
    # Air Properties (Design Basis)
    CP_AIR_J_KG_K: Final[float] = 1005.0
    RHO_AIR_KG_M3: Final[float] = 1.225
    T_A_IN_DESIGN_C: Final[float] = 20.0
    
    # Equipment efficiencies / Factors
    ETA_FAN: Final[float] = 0.60
    F_LMTD: Final[float] = 0.85
    
    # Reference Mass Flows (for validation/defaults)
    MDOT_AIR_DESIGN_H2_KG_S: Final[float] = 5.175
    MDOT_AIR_DESIGN_O2_KG_S: Final[float] = 573.0
