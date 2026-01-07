"""
Physical constants and operational parameters for hydrogen production system.

This module serves as the central repository for physical constants, conversion factors,
and equipment-specific parameters. It ensures consistency across thermodynamic calculations
and simulation sub-models.
"""

from typing import Final

class GasConstants:
    """
    Universal gas constant and species-specific thermodynamic data.
    
    Data Source: NIST REFPROP and standard engineering handbooks.
    """
    R_UNIVERSAL_J_PER_MOL_K: Final[float] = 8.314
    
    # Specific Gas Constants (R_specific = R_univ / MolarMass)
    R_H2: Final[float] = 4124.0  # Hydrogen (J/(kg·K))
    
    # Specific Heat Ratios (Cp/Cv) for Adiabatic Calculations
    GAMMA_H2: Final[float] = 1.41 # Diatomic ideal gas approx
    
    # Specific heat at constant volume (J/(kg·K))
    # Cp = 14300, Gamma = 1.41 => Cv = Cp/Gamma = 14300/1.41 = ~10142
    # Using slightly more precise value from literature for ambient H2
    CV_H2: Final[float] = 10183.0
    
    # Average Specific Heat Capacities (Isobaric)
    # Used for simplified thermal models (e.g., fallback in Chiller)
    CP_H2_AVG: Final[float] = 14300.0  # Hydrogen (J/(kg·K))
    CP_O2_AVG: Final[float] = 918.0    # Oxygen (J/(kg·K))
    
    # Species Thermodynamic Properties Library
    # Contains critical properties for Equations of State (EOS)
    SPECIES_DATA: Final[dict] = {
        'O2': {
            'molecular_weight': 32.0,
            'h_formation': 0.0,
            'cp_coeffs': [29.96, 4.18e-3, -1.67e-6, 0.0, 0.0], # Shomate Coeffs
            'critical_temp': 154.6,     # K
            'critical_pressure': 5.043e6, # Pa
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
            'antoine_coeffs': [8.07131, 1730.63, 233.426], # For vapor pressure
            'critical_temp': 647.1,
            'critical_pressure': 22.064e6,
            'acentric_factor': 0.345
        },
        'H2O_liq': {
            'molecular_weight': 18.015,
            'h_formation': -285.83e3,  # Liquid formation enthalpy
            'h_vaporization': 0.0,
            'cp_coeffs': [75.3, 0.0, 0.0, 0.0, 0.0], # Cp ~ 75.3 J/molK (constant)
            'antoine_coeffs': [0, 0, 0], 
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

# Standard conditions (STP/NTP definitions)
T_REF: Final[float] = 298.15  # K (25°C)
P_REF: Final[float] = 101325  # Pa (1 atm)

class StandardConditions:
    """Explicit definitions for Standard Temperature and Pressure."""
    TEMPERATURE_K: Final[float] = 298.15
    TEMPERATURE_C: Final[float] = 25.0
    PRESSURE_PA: Final[float] = 101325.0
    PRESSURE_BAR: Final[float] = 1.01325
    
    # Canonical order for array-based mixture thermodynamics (JIT)
    # MUST match the stacking order in LUTManager and Stream arrays
    CANONICAL_FLUID_ORDER: Final[tuple[str, ...]] = ('H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O')

class ConversionFactors:
    """
    Common engineering unit conversion factors.
    
    Used to normalize inputs/outputs to SI units (kg, m, s, Pa, K, J, W).
    """
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
    """Parameters related to hydrogen production efficiency and stoichiometry."""
    H2_ENERGY_CONTENT_LHV_KWH_PER_KG: Final[float] = 33.0
    H2_ENERGY_CONTENT_HHV_KWH_PER_KG: Final[float] = 39.4
    
    # Electrolysis Benchmarks
    ELECTROLYSIS_THEORETICAL_ENERGY_KWH_PER_KG: Final[float] = 39.4
    ELECTROLYSIS_TYPICAL_EFFICIENCY: Final[float] = 0.65
    
    # ATR (Auto-Thermal Reforming) Parameters
    ATR_TYPICAL_EFFICIENCY: Final[float] = 0.75
    ATR_STARTUP_TIME_HOURS: Final[float] = 1.0
    ATR_COOLDOWN_TIME_HOURS: Final[float] = 0.5
    
    # Stoichiometry (Mass Basis)
    O2_TO_H2_MASS_RATIO: Final[float] = 7.94 # 16/2.016

class StorageConstants:
    """Pressure tiers and tank sizing norms."""
    LOW_PRESSURE_PA: Final[float] = 30e5    # 30 bar
    HIGH_PRESSURE_PA: Final[float] = 350e5  # 350 bar
    DELIVERY_PRESSURE_PA: Final[float] = 900e5 # 900 bar
    
    TYPICAL_LP_CAPACITY_KG: Final[float] = 50.0
    TYPICAL_HP_CAPACITY_KG: Final[float] = 200.0
    
    TANK_FULL_THRESHOLD: Final[float] = 0.99
    TANK_EMPTY_THRESHOLD: Final[float] = 0.01

class CompressionConstants:
    """Compressor performance assumptions."""
    ISENTROPIC_EFFICIENCY: Final[float] = 0.75
    MECHANICAL_EFFICIENCY: Final[float] = 0.95
    TYPICAL_STAGE_PRESSURE_RATIO: Final[float] = 3.5
    MAX_STAGES: Final[int] = 4

class EconomicConstants:
    """Market parameters for TEA (Techno-Economic Analysis)."""
    ENERGY_PRICE_MIN: Final[float] = 20.0
    ENERGY_PRICE_MAX: Final[float] = 200.0
    ENERGY_PRICE_AVERAGE: Final[float] = 60.0 # EUR/MWh
    H2_SELLING_PRICE: Final[float] = 5.0      # EUR/kg
    NG_PRICE_TYPICAL: Final[float] = 3.5      # EUR/MMBtu or equivalent unit
    
    # Economic Spot Dispatch Parameters
    H2_NON_RFNBO_PRICE: Final[float] = 2.0    # EUR/kg non-certified H2
    P_GRID_MAX_MW: Final[float] = 30.0        # Maximum grid connection capacity (MW)

class SimulationDefaults:
    """Global simulation constraints and tolerances."""
    TIMESTEP_HOURS: Final[float] = 1.0        # Default (overridden by 1/60)
    ANNUAL_HOURS: Final[int] = 8760
    CHECKPOINT_INTERVAL_HOURS: Final[int] = 168
    MASS_TOLERANCE_KG: Final[float] = 1e-6
    PRESSURE_TOLERANCE_PA: Final[float] = 1e3
    TEMPERATURE_TOLERANCE_K: Final[float] = 0.01

class CoalescerConstants:
    """
    Coalescer cartridge filter performance constants.
    
    Sources:
    - Legacy Model: CoalescerModel.py
    - Technical Reference: coalescedor-1.pdf
    """
    # Geometry defaults
    D_SHELL_DEFAULT_M: Final[float] = 0.32  # Vessel diameter
    D_ELEM_DEFAULT_M: Final[float] = 0.20   # Element diameter
    L_ELEM_DEFAULT_M: Final[float] = 1.00   # Element length
    N_ELEM_DEFAULT: Final[int] = 1          # Elements per vessel

    # Physics / Performance coefficients
    # Recalibrated to match legacy 0.1500 bar drop at nominal flow (previously 0.5e6)
    K_PERDA: Final[float] = 1.996271e+10    # Permeability resistance factor
    ETA_LIQUID_REMOVAL: Final[float] = 0.98  # separation efficiency (98%)

    # Molar masses (kg/mol) - from modelo_coalescedor.py
    
    M_H2O: Final[float] = 0.018015
    M_H2: Final[float] = 0.002016
    M_O2: Final[float] = 0.031998

    # Reference properties for Sutherland viscosity model
    MU_REF_H2_PA_S: Final[float] = 9.0e-6
    MU_REF_O2_PA_S: Final[float] = 2.1e-5
    T_REF_K: Final[float] = 303.15

    # Worst case sizing (for reference)
    C_LIQ_IN_WORST_CASE_MG_M3: Final[float] = 100.0

class DeoxoConstants:
    """
    Constants for the Catalytic Deoxidizer (Deoxo) PFR.
    
    Sources:
    - Legacy Model: modleo_do_deoxo.pdf / deoxo-dim.py
    """
    # Reactor Geometry
    L_REACTOR_M: Final[float] = 1.333    # Derived from V=0.11, D=0.324
    D_REACTOR_M: Final[float] = 0.324
    AREA_REACTOR_M2: Final[float] = 0.0824  # pi * D^2 / 4
    CATALYST_POROSITY: Final[float] = 0.4
    PELLET_DIAMETER_M: Final[float] = 0.003
    
    # Kinetics (Arrhenius Parameters)
    # k_eff = K0 * exp(-Ea / RT)
    K0_VOL_S1: Final[float] = 1.0e10
    EA_J_MOL: Final[float] = 55000.0
    
    # Thermodynamics
    DELTA_H_RXN_J_MOL_O2: Final[float] = -242000.0  # Exothermic reaction enthalpy
    CP_MIX_AVG_J_MOL_K: Final[float] = 29.5       # Legacy Mixture Cp (Reverted)
    
    # Thermal Control
    U_A_W_M3_K: Final[float] = 0.0        # Adiabatic Mode (Legacy)
    T_JACKET_K: Final[float] = 323.15     # 50°C (Legacy ref)
    
    # Pressure Drop Design Point (Calibrated to 0.05 bar at nominal)
    DESIGN_SURF_VEL_M_S: Final[float] = 0.095855
    DESIGN_DP_BAR: Final[float] = 0.05
    
    # Operational Limits/Safety
    MAX_ALLOWED_O2_OUT_MOLE_FRAC: Final[float] = 0 # 5 ppm
    CRITICAL_INLET_T_K: Final[float] = 277.15           # 4°C Min Inlet
    T_CAT_MAX_K: Final[float] = 423.15                  # Catalyst limit (~150°C for Pd/Al₂O₃)
    DELTA_T_AD_MAX_K: Final[float] = 150.0              # Adiabatic ΔT warning threshold
    
    # Zoned PFR Parameters (Froment/Bischoff Literature Alignment)
    # Zone fractions: Hotspot (front), Cooling (mid), Polish (rear)
    L_ZONE_FRAC: Final[tuple[float, ...]] = (0.2, 0.6, 0.2)
    # Heat transfer: Low→high (lit: 50-500 W/m³K for packed-bed wall-cooled)
    U_A_ZONE_W_M3_K: Final[tuple[float, ...]] = (50.0, 250.0, 300.0)
    # Effectiveness: η=1,1,0.8 for Zone 3 diffusion limit
    K0_ZONE_MULT: Final[tuple[float, ...]] = (1.0, 1.0, 0.8)

class DryCoolerConstants:
    """
    Physical parameters for Dry Cooler (Air-Cooled Heat Exchanger).
    
    Designed for cooling PEM electrolysis outlet streams.
    Source: dry_cooler-1.pdf
    """
    # Active Heat Transfer Areas (m²)
    AREA_H2_M2: Final[float] = 219.0
    AREA_O2_M2: Final[float] = 24251.05
    
    # Heat Transfer Coefficients
    U_W_M2_K: Final[float] = 35.0  # Overall U value
    
    # Design Pressure Drops
    DP_AIR_PA: Final[float] = 124.0   # Air-side
    DP_FLUID_BAR: Final[float] = 0.05 # Process-side
    
    # Air Properties (Design Basis: 20°C, 1 atm)
    CP_AIR_J_KG_K: Final[float] = 1005.0
    RHO_AIR_KG_M3: Final[float] = 1.225
    T_A_IN_DESIGN_C: Final[float] = 20.0
    
    # Efficiencies
    ETA_FAN: Final[float] = 0.60
    F_LMTD: Final[float] = 0.85 # LMTD correction factor for crossflow

    # Reference Mass Flows
    MDOT_AIR_DESIGN_H2_KG_S: Final[float] = 5.175
    MDOT_AIR_DESIGN_O2_KG_S: Final[float] = 573.0

class DryCoolerIndirectConstants:
    """
    Physical parameters for Indirect Dry Cooler (Gas -> Glycol -> Air).
    
    Replicates the legacy model logic (modelo_dry_cooler.py).
    """
    # Intermediate Loop (Glycol/Water)
    GLYCOL_FRACTION: Final[float] = 0.40  # Mass fraction
    T_REF_IN_TQC_DEFAULT: Final[float] = 30.0 # Standard coolant return temp (C)
    
    # Reference Coolant Flows (kg/s)
    M_DOT_REF_H2: Final[float] = 5.0
    M_DOT_REF_O2: Final[float] = 1.0
    
    # --- Stage 1: TQC (Gas -> Glycol) ---
    # Heat Transfer Areas (m²)
    AREA_H2_TQC_M2: Final[float] = 10.0
    AREA_O2_TQC_M2: Final[float] = 5.0  
    
    # Design Values
    U_VALUE_TQC_W_M2_K: Final[float] = 1000.0
    DP_LIQ_TQC_BAR: Final[float] = 0.05 
    
    # --- Stage 2: DC (Glycol -> Air) ---
    # Heat Transfer Areas (m²)
    AREA_H2_DC_M2: Final[float] = 453.62
    AREA_O2_DC_M2: Final[float] = 92.95  
    
    # Design Values
    U_VALUE_DC_W_M2_K: Final[float] = 35.0
    DP_LIQ_DC_BAR: Final[float] = 0.5
    DP_AIR_DESIGN_PA: Final[float] = 500.0
    
    # Air Side Design Parameters
    T_AIR_DESIGN_C: Final[float] = 25.0         
    MDOT_AIR_DESIGN_H2_KG_S: Final[float] = 25.887
    MDOT_AIR_DESIGN_O2_KG_S: Final[float] = 3.552
    
    # Air Properties & Equipment
    CP_AIR_J_KG_K: Final[float] = 1005.0
    RHO_AIR_KG_M3: Final[float] = 1.225
    ETA_FAN: Final[float] = 0.60

class HenryConstants:
    """
    Henry's Law constants for gas solubility in water.
    
    Formula: H(T) = H_298 * exp(C * (1/T - 1/298.15))
    Where C = -Delta_H_sol / R.
    
    Source: Legacy aux_coolprop.py
    """
    # Hydrogen
    H2_H_298_L_ATM_MOL: Final[float] = 1300.0
    H2_DELTA_H_R_K: Final[float] = 500.0
    H2_MOLAR_MASS_KG_MOL: Final[float] = 0.002016

    # Oxygen
    O2_H_298_L_ATM_MOL: Final[float] = 770.0
    O2_DELTA_H_R_K: Final[float] = 1700.0
    O2_MOLAR_MASS_KG_MOL: Final[float] = 0.031998

