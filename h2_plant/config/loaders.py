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
            CompressorConfig, OutgoingCompressorConfig, CompressionConfig,
            DemandConfig, EnergyPriceConfig, PathwayConfig, SimulationConfig,
            BatteryConfig, WaterTreatmentConfig, WaterQualityTestConfig,
            WaterTreatmentBlockConfig, UltrapureWaterStorageConfig, WaterPumpsConfig, PumpConfig,
            ExternalInputsConfig, OxygenSourceConfig, HeatSourceConfig,
            WaterTreatmentBlockConfig, UltrapureWaterStorageConfig, WaterPumpsConfig, PumpConfig,
            ExternalInputsConfig, OxygenSourceConfig, HeatSourceConfig, BiogasSourceConfig,
            OxygenManagementConfig, MixerConfig,
            ThermalComponentsConfig, SeparationComponentsConfig, FluidComponentsConfig, PowerComponentsConfig,
            IndexedConnectionConfig, ConnectionConfig,
            PEMConfig, SOECConfig
        )
        from h2_plant.core.enums import AllocationStrategy
        
        # Production
        elec_cfg = None
        if 'production' in d and 'electrolyzer' in d['production'] and d['production']['electrolyzer']:
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
            electrolyzer_tanks_cfg = TankArrayConfig(**iso_dict['electrolyzer_tanks']) if 'electrolyzer_tanks' in iso_dict else TankArrayConfig()
            atr_tanks_cfg = TankArrayConfig(**iso_dict['atr_tanks']) if 'atr_tanks' in iso_dict and iso_dict['atr_tanks'] else None

            isolated_cfg = SourceIsolatedStorageConfig(
                electrolyzer_tanks=electrolyzer_tanks_cfg,
                atr_tanks=atr_tanks_cfg,
                oxygen_buffer_capacity_kg=iso_dict.get('oxygen_buffer_capacity_kg', 500.0)
            )
        
        storage = StorageConfig(
            lp_tanks=lp_tanks,
            hp_tanks=hp_tanks,
            source_isolated=d['storage'].get('source_isolated', False),
            isolated_config=isolated_cfg
        )
        
        # Compression
        compression_dict = d.get('compression', {})
        fill_comp = CompressorConfig(**compression_dict.get('filling_compressor', {}))
        out_comp = OutgoingCompressorConfig(**compression_dict.get('outgoing_compressor', {}))
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
        
        # Battery
        battery = None
        if 'battery' in d:
            battery = BatteryConfig(**d['battery'])

        # Water Treatment
        water_treatment = None
        if 'water_treatment' in d:
            wt_d = d['water_treatment']
            water_treatment = WaterTreatmentConfig(
                quality_test=WaterQualityTestConfig(**wt_d.get('quality_test', {})),
                treatment_block=WaterTreatmentBlockConfig(**wt_d.get('treatment_block', {})),
                ultrapure_storage=UltrapureWaterStorageConfig(**wt_d.get('ultrapure_storage', {})),
                pumps=WaterPumpsConfig(
                    pump_a=PumpConfig(**wt_d.get('pumps', {}).get('pump_a', {})),
                    pump_b=PumpConfig(**wt_d.get('pumps', {}).get('pump_b', {}))
                )
            )

        # External Inputs
        external_inputs = None
        if 'external_inputs' in d:
            ext_d = d['external_inputs']
            oxy = OxygenSourceConfig(**ext_d['oxygen_source']) if 'oxygen_source' in ext_d else None
            heat = HeatSourceConfig(**ext_d['heat_source']) if 'heat_source' in ext_d else None
            biogas = BiogasSourceConfig(**ext_d['biogas_source']) if 'biogas_source' in ext_d else None
            external_inputs = ExternalInputsConfig(oxygen_source=oxy, heat_source=heat, biogas_source=biogas)

        # Oxygen Management
        oxygen_management = None
        if 'oxygen_management' in d:
            om_d = d['oxygen_management']
            mixer = MixerConfig(**om_d['mixer']) if 'mixer' in om_d else None
            oxygen_management = OxygenManagementConfig(use_mixer=om_d.get('use_mixer', False), mixer=mixer)

        # V3.0 Indexed Component Arrays
        thermal_components = None
        if 'thermal_components' in d:
            thermal_components = ThermalComponentsConfig(**d['thermal_components'])
            
        separation_components = None
        if 'separation_components' in d:
            separation_components = SeparationComponentsConfig(**d['separation_components'])
            
        fluid_components = None
        if 'fluid_components' in d:
            fluid_components = FluidComponentsConfig(**d['fluid_components'])
            
        power_components = None
        if 'power_components' in d:
            power_components = PowerComponentsConfig(**d['power_components'])
            
        # Topology
        topology = []
        if 'topology' in d:
            topology = [ConnectionConfig(**c) for c in d['topology']]
            
        indexed_topology = []
        if 'indexed_topology' in d:
            indexed_topology = [IndexedConnectionConfig(**c) for c in d['indexed_topology']]


        # PEM System
        pem_system = None
        if 'pem_system' in d:
            pem_system = PEMConfig(**d['pem_system'])

        # SOEC Cluster
        soec_cluster = None
        if 'soec_cluster' in d:
            soec_cluster = SOECConfig(**d['soec_cluster'])

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
            simulation=simulation,
            battery=battery,
            water_treatment=water_treatment,
            external_inputs=external_inputs,
            oxygen_management=oxygen_management,
            pem_system=pem_system,
            soec_cluster=soec_cluster,
            atr_system=d.get('atr_system'),
            logistics=d.get('logistics'),
            thermal_components=thermal_components,
            separation_components=separation_components,
            fluid_components=fluid_components,
            power_components=power_components,
            topology=topology,
            indexed_topology=indexed_topology
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
