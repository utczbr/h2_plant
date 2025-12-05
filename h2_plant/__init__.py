"""
Hydrogen Production Plant - Main Package

This package contains a complete modular hydrogen production system with:
- Component-based architecture
- Performance optimization layers
- Dual-path production system
- Configuration-driven deployment
- Simulation engine

For more details, see:
- 01_Core_Foundation_Specification.md
- 02_Performance_Optimization_Specification.md
- 03_Component_Standardization_Specification.md
- 04_Configuration_System_Specification.md
- 05_Pathway_Integration_Specification.md
- 06_Simulation_Engine_Specification.md
- 07_Code_Consolidation_Guide.md
"""

__version__ = "2.0.0"
__author__ = "Hydrogen Production Team"

# Import main components for easy access
from .core import *
from .optimization import *
from .components import *
from .pathways import *
from .config import *
from .simulation import *

__all__ = [
    # Core
    'Component',
    'ComponentRegistry',
    'ComponentNotInitializedError',
    'ComponentInitializationError',
    'ComponentStepError',
    
    # Enums
    'TankState',
    'ProductionState',
    'CompressorState',
    'AllocationStrategy',
    'PathType',
    'StorageType',
    'DemandType',
    'ResourceType',
    'SimulationEvent',
    
    # Constants
    'R_H2',
    'R_O2',
    'R_STEAM',
    'R_AIR',
    'MOLAR_MASS_H2',
    'MOLAR_MASS_O2',
    'MOLAR_MASS_H2O',
    'MOLAR_MASS_CO2',
    'MOLAR_MASS_CH4',
    'STANDARD_TEMPERATURE',
    'STANDARD_PRESSURE',
    'STANDARD_CONDITIONS',
    'H2_CRITICAL_TEMPERATURE',
    'H2_CRITICAL_PRESSURE',
    'AMBIENT_TEMPERATURE',
    'AMBIENT_PRESSURE',
    'H2_HEATING_VALUE_HHV',
    'H2_HEATING_VALUE_LHV',
    'H2_ENERGY_DENSITY',
    'H2_PER_KWH_ELECTROLYSIS',
    'KW_PER_KG_H2_ELECTROLYSIS',
    'STOICH_RATIO_H2O_TO_H2',
    'STOICH_RATIO_O2_TO_H2',
    'H2_PER_KG_BIOGAS_ATR',
    'CO2_PER_KG_BIOGAS_ATR',
    'SIMULATION_HOURS',
    'DEFAULT_TIMESTEP',
    'TANK_FILL_THRESHOLD',
    'TANK_EMPTY_THRESHOLD',
    'TANK_OPERATIONAL_THRESHOLD',
    'MIN_OPERATIONAL_PRESSURE',
    'MAX_OPERATIONAL_PRESSURE',
    'MAX_STORAGE_PRESSURE',
    'COMPRESSOR_START_PRESSURE',
    'TARGET_DELIVERY_PRESSURE',
    'MIN_OPERATIONAL_TEMPERATURE',
    'MAX_OPERATIONAL_TEMPERATURE',
    'ELECTROLYZER_EFFICIENCY',
    'ELECTROLYZER_STARTUP_TIME',
    'ATR_EFFICIENCY',
    'CARBON_TAX_PER_KG_CO2',
    'ELECTRICITY_PRICE_PEAK',
    'ELECTRICITY_PRICE_OFFPEAK',
    'H2_SALES_PRICE',
    'MAX_RAMP_RATE',
    'EMERGENCY_SHUTDOWN_PRESSURE',
    'COMPRESSOR_EFFICIENCY',
    'COMPRESSOR_POWER_FACTOR',
    
    # Types
    'FloatArray',
    'IntArray',
    'BoolArray',
    'Timestamp',
    'StateVector',
    'StateHistory',
    'MassKg',
    'PressureBar',
    'TemperatureK',
    'VolumeM3',
    'EnergyKWh',
    'PowerMW',
    'CostUSD',
    'PricePerKgUSD',
    'ComponentProtocol',
    'ProductionComponentProtocol',
    'StorageComponentProtocol',
    'CompressionComponentProtocol',
    'PathwayComponentProtocol',
    'SimulationEngineProtocol',
    'ProductionFunction',
    'CompressionFunction',
    'AllocationFunction',
    'StateUpdateFunction',
    'HydrogenQualitySpec',
    'BatchRecord',
    
    # Production Components
    'ElectrolyzerProductionSource',
    'ATRProductionSource',
    
    # Storage Components
    'TankArray',
    'SourceIsolatedTanks',
    'SourceTag',
    'OxygenBuffer',
    
    # Compression Components
    'FillingCompressor',
    'OutgoingCompressor',
    
    # Utility Components
    'DemandScheduler',
    'EnergyPriceTracker',
    
    # Pathway Components
    'IsolatedProductionPath',
    'DualPathCoordinator',
    
    # Simulation Engine
    'SimulationEngine',
]