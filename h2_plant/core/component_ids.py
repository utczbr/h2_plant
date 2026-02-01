"""
Component IDs for the H2 Plant simulation.

This module defines a central Enum for component identifiers to eliminate
magic strings and ensure type safety in component lookups.
"""

from enum import Enum

class ComponentID(Enum):
    """
    Enumeration of standard component identifiers.
    """
    # Core Managers
    LUT_MANAGER = "lut_manager"
    ENVIRONMENT_MANAGER = "environment_manager"
    DUAL_PATH_COORDINATOR = "dual_path_coordinator"
    MONITOR = "monitor"
    THERMAL_MANAGER = "thermal_manager"
    WATER_BALANCE_TRACKER = "water_balance_tracker"
    
    # Production
    ELECTROLYZER = "electrolyzer"  # Generic/Simple
    PEM_ELECTROLYZER_DETAILED = "pem_electrolyzer_detailed"
    SOEC_CLUSTER = "soec_cluster"
    ATR = "atr"
    
    # Storage
    LP_TANKS = "lp_tanks"
    HP_TANKS = "hp_tanks"
    HP_STORAGE_MANAGER = "hp_storage_manager"
    OXYGEN_BUFFER = "oxygen_buffer"
    BATTERY = "battery"
    
    # Compression
    FILLING_COMPRESSOR = "filling_compressor"
    OUTGOING_COMPRESSOR = "outgoing_compressor"
    
    # Utilities
    DEMAND_SCHEDULER = "demand_scheduler"
    ENERGY_PRICE_TRACKER = "energy_price_tracker"
    
    # External Inputs
    EXTERNAL_OXYGEN_SOURCE = "external_oxygen_source"
    EXTERNAL_HEAT_SOURCE = "external_heat_source"
    
    # Oxygen Management
    OXYGEN_MIXER = "oxygen_mixer"
    
    # Water System
    WATER_QUALITY_TEST = "water_quality_test"
    WATER_TREATMENT_BLOCK = "water_treatment_block"
    WATER_PURIFIER = "water_purifier"
    WATER_INTAKE_SOURCE = "water_intake_source"
    ULTRAPURE_WATER_STORAGE = "ultrapure_water_storage"
    WATER_PUMP_A = "water_pump_a"
    WATER_PUMP_B = "water_pump_b"

    # Separation / Purification
    COALESCER = "coalescer"

    # Control
    VALVE = "valve"

    def __str__(self) -> str:
        return self.value
