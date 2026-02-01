# STEP 3: Technical Specification - Configuration System

***

# 04_Configuration_System_Specification.md

**Document:** Configuration System Technical Specification  
**Project:** Dual-Path Hydrogen Production System - Modular Refactoring v2.0  
**Date:** November 18, 2025  
**Layer:** Layer 4 - Configuration & Orchestration  
**Priority:** MEDIUM  
**Dependencies:** Layers 1-3 (Core Foundation, Performance Optimization, Component Standardization)

***

## 1. Overview

### 1.1 Purpose

This specification defines the **configuration-driven plant assembly system** that enables zero-code reconfiguration of the hydrogen production plant. The configuration system addresses a critical gap identified in the critique: the absence of declarative plant design and the prevalence of hardcoded parameters.

**Key Objectives:**
- Enable YAML/JSON-based plant configuration
- Implement `PlantBuilder` factory for configuration-driven assembly
- Replace hardcoded parameters in `system_setup.py`
- Support configuration versioning and validation
- Enable multiple plant designs without code changes

**Critique Remediation:**
- **FAIL → PASS:** "No YAML/JSON-based configuration loading" (Section 2-3)
- **PARTIAL → PASS:** "system_setup.py has hardcoded parameters" (Section 4)

***

### 1.2 Configuration Philosophy

**Before (Hardcoded):**
```python
# system_setup.py - requires code changes for different plants
hp_tanks = [SourceTaggedTank(200.0, 350e5, "electrolyzer") for _ in range(8)]
electrolyzer = HydrogenProductionSource(2.5e6, 0.65)
compressor = FillingCompressor(100.0, 30e5, 350e5)
```

**After (Configuration-Driven):**
```yaml
# configs/plant_baseline.yaml - no code changes needed
production:
  electrolyzer:
    max_power_mw: 2.5
    efficiency: 0.65
storage:
  hp_tanks:
    count: 8
    capacity_kg: 200.0
    pressure_bar: 350
compression:
  filling_compressor:
    max_flow_kg_h: 100.0
```

```python
# Code - same for all plant designs
config = load_plant_config("configs/plant_baseline.yaml")
plant = PlantBuilder.from_config(config)
```

***

### 1.3 Scope

**In Scope:**
- `config/plant_config.py`: Dataclass-based configuration structures
- `config/schemas/`: JSON Schema definitions for validation
- `config/plant_builder.py`: Factory for configuration-driven assembly
- `config/loaders.py`: YAML/JSON loading with validation
- Example configurations for common scenarios

**Out of Scope:**
- GUI configuration editors (future enhancement)
- Dynamic runtime reconfiguration (requires simulation restart)
- Database-backed configuration (future enhancement)

***

### 1.4 Design Principles

1. **Declarative Over Imperative:** Describe what the plant should be, not how to build it
2. **Type Safety:** Leverage Python dataclasses with validation
3. **Schema Validation:** JSON Schema ensures configuration correctness
4. **Version Control Friendly:** YAML/JSON files in source control
5. **Sensible Defaults:** Minimize required configuration, maximize optional parameters

***

## 2. Configuration Data Structures

### 2.1 Dataclass Hierarchy

**File:** `h2_plant/config/plant_config.py`

```python
"""
Configuration dataclasses for hydrogen production plant.

Provides type-safe, validated configuration structures using Python
dataclasses with JSON Schema validation support.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
import numpy as np

from h2_plant.core.enums import AllocationStrategy


@dataclass
class ProductionSourceConfig:
    """Base configuration for production sources."""
    enabled: bool = True
    max_capacity: float = 0.0  # MW for electrolyzer, kg/h for ATR


@dataclass
class ElectrolyzerConfig(ProductionSourceConfig):
    """Electrolyzer configuration."""
    max_power_mw: float = 2.5
    base_efficiency: float = 0.65
    min_load_factor: float = 0.20
    startup_time_hours: float = 0.1
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.base_efficiency <= 1.0:
            raise ValueError(f"Efficiency must be in (0, 1], got {self.base_efficiency}")
        if not 0 <= self.min_load_factor < 1.0:
            raise ValueError(f"Min load factor must be in [0, 1), got {self.min_load_factor}")
        if self.max_power_mw <= 0:
            raise ValueError(f"Max power must be positive, got {self.max_power_mw}")


@dataclass
class ATRConfig(ProductionSourceConfig):
    """ATR (Auto-Thermal Reforming) configuration."""
    max_ng_flow_kg_h: float = 100.0
    efficiency: float = 0.75
    reactor_temperature_k: float = 1200.0
    reactor_pressure_bar: float = 25.0
    startup_time_hours: float = 1.0
    cooldown_time_hours: float = 0.5
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.efficiency <= 1.0:
            raise ValueError(f"Efficiency must be in (0, 1], got {self.efficiency}")
        if self.max_ng_flow_kg_h <= 0:
            raise ValueError(f"Max NG flow must be positive, got {self.max_ng_flow_kg_h}")


@dataclass
class ProductionConfig:
    """Production system configuration."""
    electrolyzer: Optional[ElectrolyzerConfig] = None
    atr: Optional[ATRConfig] = None
    
    def validate(self) -> None:
        """Validate production configuration."""
        if self.electrolyzer is None and self.atr is None:
            raise ValueError("At least one production source must be configured")
        
        if self.electrolyzer:
            self.electrolyzer.validate()
        if self.atr:
            self.atr.validate()


@dataclass
class TankArrayConfig:
    """Tank array configuration."""
    count: int = 4
    capacity_kg: float = 200.0
    pressure_bar: float = 350.0
    temperature_k: float = 298.15
    
    def validate(self) -> None:
        """Validate tank array configuration."""
        if self.count <= 0:
            raise ValueError(f"Tank count must be positive, got {self.count}")
        if self.capacity_kg <= 0:
            raise ValueError(f"Capacity must be positive, got {self.capacity_kg}")
        if self.pressure_bar <= 0:
            raise ValueError(f"Pressure must be positive, got {self.pressure_bar}")


@dataclass
class SourceIsolatedStorageConfig:
    """Source-isolated storage configuration."""
    electrolyzer_tanks: TankArrayConfig = field(default_factory=TankArrayConfig)
    atr_tanks: Optional[TankArrayConfig] = None
    oxygen_buffer_capacity_kg: float = 500.0
    
    def validate(self) -> None:
        """Validate storage configuration."""
        self.electrolyzer_tanks.validate()
        if self.atr_tanks:
            self.atr_tanks.validate()


@dataclass
class StorageConfig:
    """Storage system configuration."""
    lp_tanks: TankArrayConfig = field(default_factory=lambda: TankArrayConfig(
        count=4, capacity_kg=50.0, pressure_bar=30.0
    ))
    hp_tanks: TankArrayConfig = field(default_factory=lambda: TankArrayConfig(
        count=8, capacity_kg=200.0, pressure_bar=350.0
    ))
    source_isolated: bool = False  # If True, use SourceIsolatedStorageConfig
    isolated_config: Optional[SourceIsolatedStorageConfig] = None
    
    def validate(self) -> None:
        """Validate storage configuration."""
        self.lp_tanks.validate()
        self.hp_tanks.validate()
        
        if self.source_isolated:
            if self.isolated_config is None:
                raise ValueError("isolated_config required when source_isolated=True")
            self.isolated_config.validate()


@dataclass
class CompressorConfig:
    """Compressor configuration."""
    max_flow_kg_h: float = 100.0
    inlet_pressure_bar: float = 30.0
    outlet_pressure_bar: float = 350.0
    num_stages: int = 3
    efficiency: float = 0.75
    
    def validate(self) -> None:
        """Validate compressor configuration."""
        if self.max_flow_kg_h <= 0:
            raise ValueError(f"Max flow must be positive, got {self.max_flow_kg_h}")
        if self.outlet_pressure_bar <= self.inlet_pressure_bar:
            raise ValueError("Outlet pressure must exceed inlet pressure")
        if not 0 < self.efficiency <= 1.0:
            raise ValueError(f"Efficiency must be in (0, 1], got {self.efficiency}")


@dataclass
class CompressionConfig:
    """Compression system configuration."""
    filling_compressor: CompressorConfig = field(default_factory=lambda: CompressorConfig(
        inlet_pressure_bar=30.0, outlet_pressure_bar=350.0
    ))
    outgoing_compressor: CompressorConfig = field(default_factory=lambda: CompressorConfig(
        inlet_pressure_bar=350.0, outlet_pressure_bar=900.0
    ))
    
    def validate(self) -> None:
        """Validate compression configuration."""
        self.filling_compressor.validate()
        self.outgoing_compressor.validate()


@dataclass
class DemandConfig:
    """Demand profile configuration."""
    pattern: Literal['constant', 'day_night', 'weekly', 'custom'] = 'constant'
    base_demand_kg_h: float = 50.0
    
    # Day/night pattern parameters
    day_demand_kg_h: Optional[float] = None
    night_demand_kg_h: Optional[float] = None
    day_start_hour: int = 6
    night_start_hour: int = 22
    
    # Custom pattern
    custom_profile_file: Optional[str] = None  # Path to CSV/NPY file
    
    def validate(self) -> None:
        """Validate demand configuration."""
        if self.base_demand_kg_h < 0:
            raise ValueError(f"Demand must be non-negative, got {self.base_demand_kg_h}")
        
        if self.pattern == 'day_night':
            if self.day_demand_kg_h is None or self.night_demand_kg_h is None:
                raise ValueError("day_demand_kg_h and night_demand_kg_h required for day_night pattern")
        
        if self.pattern == 'custom' and self.custom_profile_file is None:
            raise ValueError("custom_profile_file required for custom pattern")


@dataclass
class EnergyPriceConfig:
    """Energy price configuration."""
    source: Literal['constant', 'file', 'api'] = 'constant'
    constant_price_per_mwh: float = 60.0
    price_file: Optional[str] = None  # Path to CSV/NPY file with hourly prices
    
    def validate(self) -> None:
        """Validate energy price configuration."""
        if self.source == 'file' and self.price_file is None:
            raise ValueError("price_file required when source='file'")


@dataclass
class SimulationConfig:
    """Simulation parameters configuration."""
    timestep_hours: float = 1.0
    duration_hours: int = 8760  # Full year by default
    start_hour: int = 0
    checkpoint_interval_hours: int = 168  # Weekly
    
    def validate(self) -> None:
        """Validate simulation configuration."""
        if self.timestep_hours <= 0:
            raise ValueError(f"Timestep must be positive, got {self.timestep_hours}")
        if self.duration_hours <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration_hours}")


@dataclass
class PathwayConfig:
    """Dual-path coordination configuration."""
    allocation_strategy: AllocationStrategy = AllocationStrategy.COST_OPTIMAL
    priority_source: Optional[Literal['electrolyzer', 'atr']] = None
    
    def validate(self) -> None:
        """Validate pathway configuration."""
        # AllocationStrategy enum validates itself
        pass


@dataclass
class PlantConfig:
    """Complete hydrogen production plant configuration."""
    
    # Plant metadata
    name: str = "Hydrogen Production Plant"
    version: str = "1.0"
    description: str = ""
    
    # System configurations
    production: ProductionConfig = field(default_factory=ProductionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    energy_price: EnergyPriceConfig = field(default_factory=EnergyPriceConfig)
    pathway: PathwayConfig = field(default_factory=PathwayConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    def validate(self) -> None:
        """Validate entire plant configuration."""
        self.production.validate()
        self.storage.validate()
        self.compression.validate()
        self.demand.validate()
        self.energy_price.validate()
        self.pathway.validate()
        self.simulation.validate()
        
        # Cross-validation checks
        if self.storage.source_isolated:
            # Ensure isolated storage matches production sources
            if self.production.electrolyzer and not self.storage.isolated_config.electrolyzer_tanks:
                raise ValueError("Electrolyzer configured but no electrolyzer tanks in isolated storage")
            if self.production.atr and not self.storage.isolated_config.atr_tanks:
                raise ValueError("ATR configured but no ATR tanks in isolated storage")
```

***

## 3. YAML Schema and Examples

### 3.1 JSON Schema Definition

**File:** `h2_plant/config/schemas/plant_schema_v1.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Hydrogen Production Plant Configuration",
  "description": "Configuration schema for dual-path H2 production plant",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Plant name"
    },
    "version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+$"
    },
    "production": {
      "type": "object",
      "properties": {
        "electrolyzer": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "max_power_mw": {"type": "number", "minimum": 0},
            "base_efficiency": {"type": "number", "minimum": 0, "maximum": 1},
            "min_load_factor": {"type": "number", "minimum": 0, "maximum": 1}
          }
        },
        "atr": {
          "type": "object",
          "properties": {
            "enabled": {"type": "boolean"},
            "max_ng_flow_kg_h": {"type": "number", "minimum": 0},
            "efficiency": {"type": "number", "minimum": 0, "maximum": 1}
          }
        }
      },
      "minProperties": 1
    },
    "storage": {
      "type": "object",
      "properties": {
        "lp_tanks": {
          "$ref": "#/definitions/tank_array"
        },
        "hp_tanks": {
          "$ref": "#/definitions/tank_array"
        },
        "source_isolated": {"type": "boolean"}
      },
      "required": ["lp_tanks", "hp_tanks"]
    },
    "compression": {
      "type": "object",
      "properties": {
        "filling_compressor": {
          "$ref": "#/definitions/compressor"
        },
        "outgoing_compressor": {
          "$ref": "#/definitions/compressor"
        }
      }
    },
    "demand": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "enum": ["constant", "day_night", "weekly", "custom"]
        },
        "base_demand_kg_h": {"type": "number", "minimum": 0}
      }
    },
    "simulation": {
      "type": "object",
      "properties": {
        "timestep_hours": {"type": "number", "minimum": 0},
        "duration_hours": {"type": "integer", "minimum": 1}
      }
    }
  },
  "required": ["production", "storage"],
  
  "definitions": {
    "tank_array": {
      "type": "object",
      "properties": {
        "count": {"type": "integer", "minimum": 1},
        "capacity_kg": {"type": "number", "minimum": 0},
        "pressure_bar": {"type": "number", "minimum": 0}
      },
      "required": ["count", "capacity_kg", "pressure_bar"]
    },
    "compressor": {
      "type": "object",
      "properties": {
        "max_flow_kg_h": {"type": "number", "minimum": 0},
        "inlet_pressure_bar": {"type": "number", "minimum": 0},
        "outlet_pressure_bar": {"type": "number", "minimum": 0},
        "efficiency": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  }
}
```

***

### 3.2 Example Configuration - Baseline Plant

**File:** `configs/plant_baseline.yaml`

```yaml
# Baseline Hydrogen Production Plant Configuration
# Version: 1.0
# Description: Standard dual-path configuration with electrolyzer + ATR

name: "Baseline Dual-Path H2 Plant"
version: "1.0"
description: >
  Standard configuration with 2.5 MW electrolyzer and 100 kg/h ATR.
  Uses source-isolated storage for emissions tracking.

# Production Sources
production:
  electrolyzer:
    enabled: true
    max_power_mw: 2.5
    base_efficiency: 0.65
    min_load_factor: 0.20
    startup_time_hours: 0.1
  
  atr:
    enabled: true
    max_ng_flow_kg_h: 100.0
    efficiency: 0.75
    reactor_temperature_k: 1200.0
    reactor_pressure_bar: 25.0
    startup_time_hours: 1.0
    cooldown_time_hours: 0.5

# Storage System
storage:
  source_isolated: true
  
  # Low-pressure buffer storage
  lp_tanks:
    count: 4
    capacity_kg: 50.0
    pressure_bar: 30.0
    temperature_k: 298.15
  
  # High-pressure delivery storage
  hp_tanks:
    count: 8
    capacity_kg: 200.0
    pressure_bar: 350.0
    temperature_k: 298.15
  
  # Source-isolated configuration
  isolated_config:
    electrolyzer_tanks:
      count: 4
      capacity_kg: 200.0
      pressure_bar: 350.0
    
    atr_tanks:
      count: 4
      capacity_kg: 200.0
      pressure_bar: 350.0
    
    oxygen_buffer_capacity_kg: 500.0

# Compression System
compression:
  filling_compressor:
    max_flow_kg_h: 100.0
    inlet_pressure_bar: 30.0
    outlet_pressure_bar: 350.0
    num_stages: 3
    efficiency: 0.75
  
  outgoing_compressor:
    max_flow_kg_h: 200.0
    inlet_pressure_bar: 350.0
    outlet_pressure_bar: 900.0
    num_stages: 2
    efficiency: 0.75

# Demand Profile
demand:
  pattern: "day_night"
  day_demand_kg_h: 80.0
  night_demand_kg_h: 20.0
  day_start_hour: 6
  night_start_hour: 22

# Energy Pricing
energy_price:
  source: "file"
  price_file: "data/energy_prices_2025.csv"
  constant_price_per_mwh: 60.0  # Fallback if file unavailable

# Pathway Coordination
pathway:
  allocation_strategy: "COST_OPTIMAL"  # Minimize production cost
  priority_source: null  # Let optimizer decide

# Simulation Parameters
simulation:
  timestep_hours: 1.0
  duration_hours: 8760  # Full year
  start_hour: 0
  checkpoint_interval_hours: 168  # Weekly checkpoints
```

***

### 3.3 Example Configuration - Grid-Only Plant

**File:** `configs/plant_grid_only.yaml`

```yaml
# Grid-Only Hydrogen Production Plant
# Electrolyzer-only configuration (no ATR)

name: "Grid-Only H2 Plant"
version: "1.0"
description: "Pure renewable hydrogen from grid-powered electrolysis"

production:
  electrolyzer:
    enabled: true
    max_power_mw: 5.0  # Larger electrolyzer
    base_efficiency: 0.68  # More efficient
    min_load_factor: 0.15
  
  atr: null  # No natural gas reforming

storage:
  source_isolated: false  # Simple storage (only one source)
  
  lp_tanks:
    count: 6
    capacity_kg: 75.0
    pressure_bar: 30.0
  
  hp_tanks:
    count: 12
    capacity_kg: 250.0
    pressure_bar: 350.0

compression:
  filling_compressor:
    max_flow_kg_h: 150.0
    inlet_pressure_bar: 30.0
    outlet_pressure_bar: 350.0
    num_stages: 3
    efficiency: 0.78
  
  outgoing_compressor:
    max_flow_kg_h: 250.0
    inlet_pressure_bar: 350.0
    outlet_pressure_bar: 900.0
    num_stages: 2
    efficiency: 0.78

demand:
  pattern: "constant"
  base_demand_kg_h: 100.0

energy_price:
  source: "file"
  price_file: "data/renewable_prices.csv"

pathway:
  allocation_strategy: "PRIORITY_GRID"  # Always use electrolyzer

simulation:
  timestep_hours: 1.0
  duration_hours: 8760
```

***

### 3.4 Example Configuration - Small-Scale Pilot

**File:** `configs/plant_pilot.yaml`

```yaml
# Small-Scale Pilot Plant
# Minimal configuration for testing/development

name: "Pilot H2 Plant"
version: "1.0"
description: "Small-scale pilot for testing and validation"

production:
  electrolyzer:
    enabled: true
    max_power_mw: 0.5  # 500 kW
    base_efficiency: 0.60
    min_load_factor: 0.25

storage:
  source_isolated: false
  
  lp_tanks:
    count: 2
    capacity_kg: 20.0
    pressure_bar: 30.0
  
  hp_tanks:
    count: 3
    capacity_kg: 50.0
    pressure_bar: 350.0

compression:
  filling_compressor:
    max_flow_kg_h: 20.0
    inlet_pressure_bar: 30.0
    outlet_pressure_bar: 350.0
    num_stages: 2
    efficiency: 0.70

demand:
  pattern: "constant"
  base_demand_kg_h: 10.0

energy_price:
  source: "constant"
  constant_price_per_mwh: 50.0

simulation:
  timestep_hours: 1.0
  duration_hours: 168  # One week
  checkpoint_interval_hours: 24  # Daily checkpoints
```

***

## 4. Configuration Loading and Validation

### 4.1 YAML/JSON Loaders

**File:** `h2_plant/config/loaders.py`

```python
"""
Configuration loading with validation.

Supports YAML and JSON formats with JSON Schema validation.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
import jsonschema
import logging

from h2_plant.config.plant_config import PlantConfig
from h2_plant.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader with schema validation.
    
    Example:
        loader = ConfigLoader()
        config = loader.load_yaml("configs/plant_baseline.yaml")
        config.validate()
    """
    
    def __init__(self, schema_path: Path = None):
        """
        Initialize configuration loader.
        
        Args:
            schema_path: Path to JSON schema file (uses default if None)
        """
        if schema_path is None:
            schema_path = Path(__file__).parent / "schemas" / "plant_schema_v1.json"
        
        self.schema_path = schema_path
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema from file."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load schema from {self.schema_path}: {e}")
            return {}
    
    def load_yaml(self, config_path: Path | str) -> PlantConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            PlantConfig instance
            
        Raises:
            ConfigurationError: If file not found or validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse YAML: {e}")
        
        return self._dict_to_config(config_dict)
    
    def load_json(self, config_path: Path | str) -> PlantConfig:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            PlantConfig instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse JSON: {e}")
        
        return self._dict_to_config(config_dict)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> PlantConfig:
        """
        Convert dictionary to PlantConfig with validation.
        
        Args:
            config_dict: Configuration dictionary from YAML/JSON
            
        Returns:
            PlantConfig instance
            
        Raises:
            ConfigurationError: If validation fails
        """
        # JSON Schema validation
        if self.schema:
            try:
                jsonschema.validate(instance=config_dict, schema=self.schema)
                logger.debug("JSON schema validation passed")
            except jsonschema.ValidationError as e:
                raise ConfigurationError(f"Schema validation failed: {e.message}")
        
        # Convert to PlantConfig using dacite or manual construction
        try:
            config = self._build_plant_config(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to build PlantConfig: {e}")
        
        # Dataclass validation
        try:
            config.validate()
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")
        
        logger.info(f"Loaded configuration: {config.name} v{config.version}")
        return config
    
    def _build_plant_config(self, d: Dict[str, Any]) -> PlantConfig:
        """Build PlantConfig from dictionary (manual construction)."""
        from h2_plant.config.plant_config import (
            ElectrolyzerConfig, ATRConfig, ProductionConfig,
            TankArrayConfig, StorageConfig, SourceIsolatedStorageConfig,
            CompressorConfig, CompressionConfig,
            DemandConfig, EnergyPriceConfig, PathwayConfig, SimulationConfig
        )
        from h2_plant.core.enums import AllocationStrategy
        
        # Production
        elec_cfg = None
        if 'production' in d and 'electrolyzer' in d['production']:
            elec_cfg = ElectrolyzerConfig(**d['production']['electrolyzer'])
        
        atr_cfg = None
        if 'production' in d and 'atr' in d['production'] and d['production']['atr']:
            atr_cfg = ATRConfig(**d['production']['atr'])
        
        production = ProductionConfig(electrolyzer=elec_cfg, atr=atr_cfg)
        
        # Storage
        lp_tanks = TankArrayConfig(**d['storage']['lp_tanks'])
        hp_tanks = TankArrayConfig(**d['storage']['hp_tanks'])
        
        isolated_cfg = None
        if d['storage'].get('source_isolated'):
            iso_dict = d['storage'].get('isolated_config', {})
            isolated_cfg = SourceIsolatedStorageConfig(
                electrolyzer_tanks=TankArrayConfig(**iso_dict['electrolyzer_tanks']),
                atr_tanks=TankArrayConfig(**iso_dict['atr_tanks']) if 'atr_tanks' in iso_dict else None,
                oxygen_buffer_capacity_kg=iso_dict.get('oxygen_buffer_capacity_kg', 500.0)
            )
        
        storage = StorageConfig(
            lp_tanks=lp_tanks,
            hp_tanks=hp_tanks,
            source_isolated=d['storage'].get('source_isolated', False),
            isolated_config=isolated_cfg
        )
        
        # Compression
        fill_comp = CompressorConfig(**d['compression']['filling_compressor'])
        out_comp = CompressorConfig(**d['compression']['outgoing_compressor'])
        compression = CompressionConfig(filling_compressor=fill_comp, outgoing_compressor=out_comp)
        
        # Demand
        demand = DemandConfig(**d.get('demand', {}))
        
        # Energy price
        energy_price = EnergyPriceConfig(**d.get('energy_price', {}))
        
        # Pathway
        pathway_dict = d.get('pathway', {})
        if 'allocation_strategy' in pathway_dict and isinstance(pathway_dict['allocation_strategy'], str):
            pathway_dict['allocation_strategy'] = AllocationStrategy[pathway_dict['allocation_strategy']]
        pathway = PathwayConfig(**pathway_dict)
        
        # Simulation
        simulation = SimulationConfig(**d.get('simulation', {}))
        
        return PlantConfig(
            name=d.get('name', 'Hydrogen Plant'),
            version=d.get('version', '1.0'),
            description=d.get('description', ''),
            production=production,
            storage=storage,
            compression=compression,
            demand=demand,
            energy_price=energy_price,
            pathway=pathway,
            simulation=simulation
        )


def load_plant_config(config_path: Path | str) -> PlantConfig:
    """
    Convenience function to load plant configuration.
    
    Automatically detects YAML or JSON based on file extension.
    
    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)
        
    Returns:
        Validated PlantConfig instance
        
    Example:
        config = load_plant_config("configs/plant_baseline.yaml")
    """
    loader = ConfigLoader()
    config_path = Path(config_path)
    
    if config_path.suffix in ['.yaml', '.yml']:
        return loader.load_yaml(config_path)
    elif config_path.suffix == '.json':
        return loader.load_json(config_path)
    else:
        raise ConfigurationError(f"Unsupported file format: {config_path.suffix}")
```

***

## 5. PlantBuilder Factory

### 5.1 Configuration-Driven Assembly

**File:** `h2_plant/config/plant_builder.py`

```python
"""
PlantBuilder: Factory for configuration-driven plant assembly.

Constructs complete hydrogen production system from PlantConfig,
registering all components and wiring dependencies.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import logging

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import PlantConfig
from h2_plant.config.loaders import load_plant_config
from h2_plant.core.exceptions import ConfigurationError

# Component imports
from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource
from h2_plant.components.production.atr_source import ATRProductionSource
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.components.storage.source_isolated_tanks import (
    SourceIsolatedTanks, SourceTag
)
from h2_plant.components.storage.oxygen_buffer import OxygenBuffer
from h2_plant.components.compression.filling_compressor import FillingCompressor
from h2_plant.components.compression.outgoing_compressor import OutgoingCompressor
from h2_plant.components.utility.demand_scheduler import DemandScheduler
from h2_plant.components.utility.energy_price_tracker import EnergyPriceTracker
from h2_plant.optimization.lut_manager import LUTManager

logger = logging.getLogger(__name__)


class PlantBuilder:
    """
    Factory for building hydrogen production plants from configuration.
    
    Example:
        # From configuration file
        plant = PlantBuilder.from_file("configs/plant_baseline.yaml")
        registry = plant.registry
        registry.initialize_all(dt=1.0)
        
        # From PlantConfig object
        config = PlantConfig(...)
        plant = PlantBuilder.from_config(config)
    """
    
    def __init__(self, config: PlantConfig):
        """
        Initialize PlantBuilder.
        
        Args:
            config: Validated PlantConfig instance
        """
        self.config = config
        self.registry = ComponentRegistry()
    
    @classmethod
    def from_file(cls, config_path: Path | str) -> 'PlantBuilder':
        """
        Build plant from configuration file.
        
        Args:
            config_path: Path to YAML/JSON configuration file
            
        Returns:
            PlantBuilder instance with populated registry
        """
        config = load_plant_config(config_path)
        builder = cls(config)
        builder.build()
        return builder
    
    @classmethod
    def from_config(cls, config: PlantConfig) -> 'PlantBuilder':
        """
        Build plant from PlantConfig object.
        
        Args:
            config: PlantConfig instance
            
        Returns:
            PlantBuilder instance with populated registry
        """
        config.validate()  # Ensure valid
        builder = cls(config)
        builder.build()
        return builder
    
    def build(self) -> None:
        """Build complete plant system and populate registry."""
        logger.info(f"Building plant: {self.config.name}")
        
        # 1. Build performance infrastructure
        self._build_lut_manager()
        
        # 2. Build production components
        self._build_production()
        
        # 3. Build storage components
        self._build_storage()
        
        # 4. Build compression components
        self._build_compression()
        
        # 5. Build utility components
        self._build_utilities()
        
        logger.info(f"Plant built successfully: {self.registry.get_component_count()} components registered")
    
    def _build_lut_manager(self) -> None:
        """Build and register LUT Manager."""
        lut = LUTManager()
        self.registry.register('lut_manager', lut, component_type='performance')
        logger.debug("Registered LUT Manager")
    
    def _build_production(self) -> None:
        """Build production components from configuration."""
        prod_cfg = self.config.production
        
        # Electrolyzer
        if prod_cfg.electrolyzer and prod_cfg.electrolyzer.enabled:
            electrolyzer = ElectrolyzerProductionSource(
                max_power_mw=prod_cfg.electrolyzer.max_power_mw,
                base_efficiency=prod_cfg.electrolyzer.base_efficiency,
                min_load_factor=prod_cfg.electrolyzer.min_load_factor,
                startup_time_hours=prod_cfg.electrolyzer.startup_time_hours
            )
            self.registry.register('electrolyzer', electrolyzer, component_type='production')
            logger.debug(f"Registered electrolyzer: {prod_cfg.electrolyzer.max_power_mw} MW")
        
        # ATR
        if prod_cfg.atr and prod_cfg.atr.enabled:
            atr = ATRProductionSource(
                max_ng_flow_kg_h=prod_cfg.atr.max_ng_flow_kg_h,
                efficiency=prod_cfg.atr.efficiency,
                reactor_temperature_k=prod_cfg.atr.reactor_temperature_k,
                reactor_pressure_bar=prod_cfg.atr.reactor_pressure_bar,
                startup_time_hours=prod_cfg.atr.startup_time_hours,
                cooldown_time_hours=prod_cfg.atr.cooldown_time_hours
            )
            self.registry.register('atr', atr, component_type='production')
            logger.debug(f"Registered ATR: {prod_cfg.atr.max_ng_flow_kg_h} kg/h")
    
    def _build_storage(self) -> None:
        """Build storage components from configuration."""
        stor_cfg = self.config.storage
        
        if stor_cfg.source_isolated:
            # Source-isolated storage
            sources = {}
            
            if self.config.production.electrolyzer:
                sources['electrolyzer'] = SourceTag(
                    source_id='electrolyzer',
                    source_type='electrolyzer',
                    emissions_factor=0.0  # Green hydrogen
                )
            
            if self.config.production.atr:
                sources['atr'] = SourceTag(
                    source_id='atr',
                    source_type='atr',
                    emissions_factor=10.5  # kg CO2 per kg H2 (typical)
                )
            
            iso_cfg = stor_cfg.isolated_config
            hp_storage = SourceIsolatedTanks(
                sources=sources,
                tanks_per_source=iso_cfg.electrolyzer_tanks.count,
                capacity_kg=iso_cfg.electrolyzer_tanks.capacity_kg,
                pressure_bar=iso_cfg.electrolyzer_tanks.pressure_bar
            )
            self.registry.register('hp_storage', hp_storage, component_type='storage')
            
            # Oxygen buffer
            o2_buffer = OxygenBuffer(capacity_kg=iso_cfg.oxygen_buffer_capacity_kg)
            self.registry.register('oxygen_buffer', o2_buffer, component_type='storage')
        
        else:
            # Standard tank arrays
            lp_tanks = TankArray(
                n_tanks=stor_cfg.lp_tanks.count,
                capacity_kg=stor_cfg.lp_tanks.capacity_kg,
                pressure_bar=stor_cfg.lp_tanks.pressure_bar
            )
            self.registry.register('lp_tanks', lp_tanks, component_type='storage')
            
            hp_tanks = TankArray(
                n_tanks=stor_cfg.hp_tanks.count,
                capacity_kg=stor_cfg.hp_tanks.capacity_kg,
                pressure_bar=stor_cfg.hp_tanks.pressure_bar
            )
            self.registry.register('hp_tanks', hp_tanks, component_type='storage')
        
        logger.debug("Registered storage components")
    
    def _build_compression(self) -> None:
        """Build compression components from configuration."""
        comp_cfg = self.config.compression
        
        # Filling compressor
        filling_comp = FillingCompressor(
            max_flow_kg_h=comp_cfg.filling_compressor.max_flow_kg_h,
            inlet_pressure_bar=comp_cfg.filling_compressor.inlet_pressure_bar,
            outlet_pressure_bar=comp_cfg.filling_compressor.outlet_pressure_bar,
            num_stages=comp_cfg.filling_compressor.num_stages,
            efficiency=comp_cfg.filling_compressor.efficiency
        )
        self.registry.register('filling_compressor', filling_comp, component_type='compression')
        
        # Outgoing compressor
        outgoing_comp = OutgoingCompressor(
            max_flow_kg_h=comp_cfg.outgoing_compressor.max_flow_kg_h,
            inlet_pressure_bar=comp_cfg.outgoing_compressor.inlet_pressure_bar,
            outlet_pressure_bar=comp_cfg.outgoing_compressor.outlet_pressure_bar,
            efficiency=comp_cfg.outgoing_compressor.efficiency
        )
        self.registry.register('outgoing_compressor', outgoing_comp, component_type='compression')
        
        logger.debug("Registered compression components")
    
    def _build_utilities(self) -> None:
        """Build utility components from configuration."""
        
        # Demand scheduler
        demand_cfg = self.config.demand
        
        custom_profile = None
        if demand_cfg.pattern == 'custom' and demand_cfg.custom_profile_file:
            custom_profile = self._load_profile_file(demand_cfg.custom_profile_file)
        
        demand_scheduler = DemandScheduler(
            pattern=demand_cfg.pattern,
            base_demand_kg_h=demand_cfg.base_demand_kg_h,
            day_demand_kg_h=demand_cfg.day_demand_kg_h,
            night_demand_kg_h=demand_cfg.night_demand_kg_h,
            day_start_hour=demand_cfg.day_start_hour,
            night_start_hour=demand_cfg.night_start_hour,
            custom_profile=custom_profile
        )
        self.registry.register('demand_scheduler', demand_scheduler, component_type='utility')
        
        # Energy price tracker
        price_cfg = self.config.energy_price
        
        if price_cfg.source == 'file' and price_cfg.price_file:
            prices = self._load_profile_file(price_cfg.price_file)
        else:
            prices = np.full(8760, price_cfg.constant_price_per_mwh)
        
        energy_tracker = EnergyPriceTracker(
            prices_per_mwh=prices,
            default_price_per_mwh=price_cfg.constant_price_per_mwh
        )
        self.registry.register('energy_price_tracker', energy_tracker, component_type='utility')
        
        logger.debug("Registered utility components")
    
    def _load_profile_file(self, file_path: str) -> np.ndarray:
        """Load time-series profile from CSV or NPY file."""
        path = Path(file_path)
        
        if not path.exists():
            raise ConfigurationError(f"Profile file not found: {file_path}")
        
        if path.suffix == '.npy':
            return np.load(path)
        elif path.suffix == '.csv':
            return np.loadtxt(path, delimiter=',')
        else:
            raise ConfigurationError(f"Unsupported profile format: {path.suffix}")
```

***

## 6. Usage Examples

### 6.1 Basic Configuration Loading

```python
from h2_plant.config.plant_builder import PlantBuilder

# Load plant from configuration file
plant = PlantBuilder.from_file("configs/plant_baseline.yaml")

# Access registry
registry = plant.registry

# Initialize all components
registry.initialize_all(dt=1.0)

# Run simulation
for hour in range(8760):
    registry.step_all(hour)
    
    if hour % 168 == 0:  # Weekly checkpoint
        state = registry.get_all_states()
        print(f"Hour {hour}: Total H2 = {state['hp_storage']['total_mass_kg']:.1f} kg")
```

***

### 6.2 Multiple Plant Configurations

```python
# Run different scenarios without code changes

scenarios = [
    "configs/plant_baseline.yaml",
    "configs/plant_grid_only.yaml",
    "configs/plant_pilot.yaml"
]

for scenario_path in scenarios:
    print(f"\n=== Running {scenario_path} ===")
    
    plant = PlantBuilder.from_file(scenario_path)
    registry = plant.registry
    registry.initialize_all(dt=1.0)
    
    # Run simulation
    for hour in range(plant.config.simulation.duration_hours):
        registry.step_all(hour)
    
    # Report results
    final_state = registry.get_all_states()
    print(f"Final H2 storage: {final_state['hp_storage']['total_mass_kg']:.1f} kg")
```

***

### 6.3 Programmatic Configuration

```python
from h2_plant.config.plant_config import (
    PlantConfig, ProductionConfig, ElectrolyzerConfig,
    StorageConfig, TankArrayConfig
)

# Build configuration programmatically
config = PlantConfig(
    name="Custom Plant",
    production=ProductionConfig(
        electrolyzer=ElectrolyzerConfig(max_power_mw=3.0, base_efficiency=0.70)
    ),
    storage=StorageConfig(
        hp_tanks=TankArrayConfig(count=10, capacity_kg=300.0, pressure_bar=350)
    )
)

# Validate
config.validate()

# Build plant
plant = PlantBuilder.from_config(config)
```

***

## 7. Testing Strategy

### 7.1 Configuration Validation Tests

**File:** `tests/config/test_plant_config.py`

```python
import pytest
from h2_plant.config.plant_config import PlantConfig, ElectrolyzerConfig, ProductionConfig


def test_electrolyzer_config_validation():
    """Test electrolyzer configuration validation."""
    
    # Valid configuration
    config = ElectrolyzerConfig(max_power_mw=2.5, base_efficiency=0.65)
    config.validate()  # Should not raise
    
    # Invalid efficiency
    invalid_config = ElectrolyzerConfig(max_power_mw=2.5, base_efficiency=1.5)
    with pytest.raises(ValueError, match="Efficiency"):
        invalid_config.validate()


def test_plant_config_cross_validation():
    """Test cross-validation between subsystems."""
    
    # Source-isolated storage without ATR tanks should fail
    config = PlantConfig()
    config.storage.source_isolated = True
    config.production.atr = ATRConfig(max_ng_flow_kg_h=100.0)
    config.storage.isolated_config = SourceIsolatedStorageConfig(
        electrolyzer_tanks=TankArrayConfig()
        # Missing atr_tanks!
    )
    
    with pytest.raises(ValueError, match="ATR configured but no ATR tanks"):
        config.validate()
```

***

### 7.2 Configuration Loading Tests

**File:** `tests/config/test_loaders.py`

```python
from pathlib import Path
from h2_plant.config.loaders import load_plant_config


def test_load_yaml_configuration(tmp_path):
    """Test loading YAML configuration."""
    
    yaml_content = """
name: "Test Plant"
version: "1.0"
production:
  electrolyzer:
    max_power_mw: 2.5
    base_efficiency: 0.65
storage:
  lp_tanks:
    count: 4
    capacity_kg: 50.0
    pressure_bar: 30.0
  hp_tanks:
    count: 8
    capacity_kg: 200.0
    pressure_bar: 350.0
"""
    
    # Write temporary config file
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)
    
    # Load configuration
    config = load_plant_config(config_file)
    
    assert config.name == "Test Plant"
    assert config.production.electrolyzer.max_power_mw == 2.5
    assert config.storage.hp_tanks.count == 8
```

***

### 7.3 PlantBuilder Tests

**File:** `tests/config/test_plant_builder.py`

```python
from h2_plant.config.plant_builder import PlantBuilder


def test_plant_builder_from_config():
    """Test PlantBuilder constructs plant correctly."""
    
    config = PlantConfig(
        production=ProductionConfig(
            electrolyzer=ElectrolyzerConfig(max_power_mw=2.5)
        )
    )
    
    plant = PlantBuilder.from_config(config)
    
    # Verify components registered
    assert plant.registry.has('electrolyzer')
    assert plant.registry.has('lut_manager')
    assert plant.registry.get_component_count() > 0


def test_plant_builder_initializes_components():
    """Test PlantBuilder components can be initialized."""
    
    plant = PlantBuilder.from_file("configs/plant_pilot.yaml")
    
    # Should initialize without errors
    plant.registry.initialize_all(dt=1.0)
    
    # Verify initialization
    electrolyzer = plant.registry.get('electrolyzer')
    assert electrolyzer._initialized
```

***

## 8. Validation Criteria

This Configuration System is **COMPLETE** when:

 **Dataclass Structures:**
- All configuration dataclasses defined
- Validation methods implemented
- Type hints complete

 **YAML/JSON Support:**
- ConfigLoader implemented
- JSON Schema validation working
- Example configurations provided

 **PlantBuilder:**
- From_file() and from_config() working
- All component types supported
- Registry population correct

 **Testing:**
- Configuration validation tests pass
- Loading tests pass
- PlantBuilder tests pass
- 95%+ test coverage

 **Documentation:**
- Example configurations documented
- Migration guide from hardcoded setup

---

## 9. Success Metrics

| **Metric** | **Target** | **Validation** |
|-----------|-----------|----------------|
| Configuration Coverage | 100% | All component types configurable |
| Validation Completeness | 100% | All invalid configs rejected |
| Example Configurations | 3+ | Baseline, grid-only, pilot provided |
| Test Coverage | 95%+ | `pytest --cov=h2_plant.config` |
| Zero-Code Reconfiguration | Yes | Multiple plants without code changes |

***
