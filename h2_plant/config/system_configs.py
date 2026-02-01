# System Configuration Dataclasses (to be added to plant_config.py)

from dataclasses import dataclass
from typing import List, Optional

# ============================================================================
# DETAILED SYSTEM CONFIGURATIONS (v2.0)
# ============================================================================

@dataclass
class ComponentConfig:
    """Base class for individual components."""
    component_id: str
    enabled: bool = True

# --- PEM System Components ---

@dataclass 
class PEMStackConfig(ComponentConfig):
    """PEM Stack configuration."""
    max_power_kw: float = 2500.0
    cells_per_stack: int = 85
    parallel_stacks: int = 36
    active_area_m2: float = 0.03

@dataclass
class RectifierConfig(ComponentConfig):
    """Rectifier/Transformer configuration."""
    max_power_kw: float = 2500.0
    efficiency: float = 0.98

@dataclass
class HeatExchangerConfig(ComponentConfig):
    """Heat Exchanger configuration."""
    max_heat_removal_kw: float = 500.0
    target_outlet_temp_c: float = 25.0

@dataclass
class PumpConfig(ComponentConfig):
    """Pump configuration."""
    max_flow_kg_h: float = 1000.0
    pressure_bar: float = 5.0

@dataclass  
class SeparationTankConfig(ComponentConfig):
    """Separation Tank configuration."""
    gas_type: str = "H2"

@dataclass
class PSAConfig(ComponentConfig):
    """PSA Unit configuration."""
    gas_type: str = "H2"

@dataclass
class PEMSystemConfig:
    """Complete PEM electrolysis system."""
    stacks: List[PEMStackConfig] = None
    rectifiers: List[RectifierConfig] = None
    heat_exchangers: List[HeatExchangerConfig] = None
    pumps: List[PumpConfig] = None
    separation_tanks: List[SeparationTankConfig] = None
    psa_units: List[PSAConfig] = None

# --- SOEC System Components ---

@dataclass
class SOECStackConfig(ComponentConfig):
    """SOEC Stack configuration."""
    max_power_kw: float = 1000.0

@dataclass
class SteamGeneratorConfig(ComponentConfig):
    """Steam Generator configuration."""
    max_flow_kg_h: float = 500.0

@dataclass
class CompressorConfig(ComponentConfig):
    """Process Compressor configuration."""
    max_flow_kg_h: float = 500.0
    pressure_ratio: float = 2.0

@dataclass
class SOECSystemConfig:
    """Complete SOEC electrolysis system."""
    stacks: List[SOECStackConfig] = None
    rectifiers: List[RectifierConfig] = None
    steam_generators: List[SteamGeneratorConfig] = None
    heat_exchangers: List[HeatExchangerConfig] = None
    pumps: List[PumpConfig] = None
    compressors: List[CompressorConfig] = None
    separation_tanks: List[SeparationTankConfig] = None
    psa_units: List[PSAConfig] = None

# --- ATR System Components ---

@dataclass
class ATRReactorConfig(ComponentConfig):
    """ATR Reactor configuration."""
    max_flow_kg_h: float = 1500.0
    model_path: str = "ATR_model_functions.pkl"

@dataclass
class WGSReactorConfig(ComponentConfig):
    """WGS Reactor configuration."""
    conversion_rate: float = 0.7

@dataclass
class ATRSystemConfig:
    """Complete ATR reforming system."""
    reactors: List[ATRReactorConfig] = None
    wgs_reactors: List[WGSReactorConfig] = None
    steam_generators: List[SteamGeneratorConfig] = None
    heat_exchangers: List[HeatExchangerConfig] = None
    compressors: List[CompressorConfig] = None
    separation_tanks: List[SeparationTankConfig] = None
    psa_units: List[PSAConfig] = None

# --- Logistics ---

@dataclass
class ConsumerConfig(ComponentConfig):
    """Consumer/Refueling Station configuration."""
    num_bays: int = 4
    filling_rate_kg_h: float = 50.0

@dataclass
class LogisticsConfig:
    """Logistics configuration."""
    consumers: List[ConsumerConfig] = None
