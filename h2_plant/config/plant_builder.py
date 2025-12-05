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
from h2_plant.core.component_ids import ComponentID
from h2_plant.config.plant_config import PlantConfig
from h2_plant.config.loaders import load_plant_config
from h2_plant.core.exceptions import ConfigurationError

# Component imports
# from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource
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
from h2_plant.components.external.oxygen_source import ExternalOxygenSource
from h2_plant.components.external.heat_source import ExternalHeatSource
from h2_plant.components.mixing.oxygen_mixer import OxygenMixer
from h2_plant.components.storage.battery_storage import BatteryStorage
from h2_plant.components.storage.h2_storage_enhanced import H2StorageTankEnhanced
from h2_plant.components.water.quality_test import WaterQualityTestBlock
from h2_plant.components.water.treatment import WaterTreatmentBlock
from h2_plant.components.water.storage import UltrapureWaterStorageTank
from h2_plant.components.water.pump import WaterPump

# PEM/SOEC component imports
from h2_plant.components.production.pem_electrolyzer_detailed import DetailedPEMElectrolyzer
from h2_plant.components.electrolysis.soec_cluster_wrapper import SOECClusterWrapper
# from h2_plant.components.pathways.dual_path_coordinator import DualPathCoordinator

# V3.0 Indexed component imports
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.separation.psa import PSA
from h2_plant.components.power.rectifier import Rectifier

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

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PlantBuilder':
        """
        Build plant from configuration dictionary.
        
        Args:
            config_dict: Dictionary containing plant configuration
            
        Returns:
            PlantBuilder instance with populated registry
        """
        from h2_plant.config.loaders import ConfigLoader
        loader = ConfigLoader()
        # Use the loader's internal method to convert dict to config
        # This handles schema validation and dataclass construction
        config = loader._dict_to_config(config_dict)
        
        builder = cls(config)
        builder.build()
        return builder
    
    def build(self) -> None:
        """Build complete plant system and populate registry."""
        logger.info(f"Building plant: {self.config.name}")
        
        # Build in logical order
        self._build_lut_manager()
        self._build_environment_manager()  # Add environment manager early
        self._build_production()
        self._build_storage()
        self._build_compression()
        self._build_utilities()
        
        # Build simple coordinator if we have electrolyzer and wind data
        if self.config.production.electrolyzer and self.config.production.electrolyzer.enabled:
            if self.config.energy_price.source == 'file' and self.config.energy_price.wind_data_file:
                self._build_simple_coordinator()
        self._build_external_inputs()
        self._build_oxygen_management()
        self._build_battery()
        self._build_water_treatment()
        
        # V2.0 Detailed system builds
        self._build_dual_path_coordinator() # Register FIRST to ensure it runs before pathways
        self._build_pem_system()
        self._build_soec_system()
        self._build_atr_system()
        self._build_logistics()
        # self._build_dual_path_coordinator() # Moved up
        self._build_thermal_manager()
        self._build_water_balance_tracker()
        
        # V3.0 Indexed component arrays
        self._build_thermal_components()
        self._build_separation_components()
        self._build_power_components()
        self._build_fluid_system()
        self._build_h2_distribution()
        self._build_co2_system()
        
        logger.info(f"Plant built successfully: {self.registry.get_component_count()} components registered")
    
    def _build_lut_manager(self) -> None:
        """Build and register LUT Manager."""
        lut = LUTManager()
        self.registry.register(ComponentID.LUT_MANAGER, lut, component_type='performance')
        logger.debug("Registered LUT Manager")
    
    def _build_production(self) -> None:
        """Build production components from configuration."""
        prod_cfg = self.config.production
        
        if prod_cfg.electrolyzer and prod_cfg.electrolyzer.enabled:
            pem_config_dict = {
                'max_power_mw': prod_cfg.electrolyzer.max_power_mw,
                'base_efficiency': prod_cfg.electrolyzer.base_efficiency,
                'use_polynomials': getattr(prod_cfg.electrolyzer, 'use_polynomials', False)
            }
            electrolyzer = DetailedPEMElectrolyzer(pem_config_dict)
            self.registry.register(ComponentID.ELECTROLYZER, electrolyzer, component_type='production')
            logger.debug(f"Registered PEM electrolyzer: {prod_cfg.electrolyzer.max_power_mw} MW")
        
        # SOEC support
        if hasattr(prod_cfg, 'soec') and prod_cfg.soec and prod_cfg.soec.enabled:
            print(f"DEBUG: Building SOEC Cluster with {getattr(prod_cfg.soec, 'num_modules', 6)} modules")
            soec_cluster = SOECClusterWrapper(
                max_power_mw=prod_cfg.soec.max_power_mw,
                num_modules=getattr(prod_cfg.soec, 'num_modules', 6),
                t_op_h_initial=0.0
            )
            self.registry.register(ComponentID.SOEC_CLUSTER, soec_cluster, component_type='production')
            logger.debug(f"Registered SOEC cluster: {prod_cfg.soec.max_power_mw} MW ({getattr(prod_cfg.soec, 'num_modules', 6)} modules)")
        else:
            print(f"DEBUG: SOEC config missing or disabled: {getattr(prod_cfg, 'soec', 'Missing')}")
        
        if prod_cfg.atr and prod_cfg.atr.enabled:
            atr = ATRProductionSource(
                max_ng_flow_kg_h=prod_cfg.atr.max_ng_flow_kg_h,
                efficiency=prod_cfg.atr.efficiency,
                reactor_temperature_k=prod_cfg.atr.reactor_temperature_k,
                reactor_pressure_bar=prod_cfg.atr.reactor_pressure_bar,
                startup_time_hours=prod_cfg.atr.startup_time_hours,
                cooldown_time_hours=prod_cfg.atr.cooldown_time_hours
            )
            self.registry.register(ComponentID.ATR, atr, component_type='production')
            logger.debug(f"Registered ATR: {prod_cfg.atr.max_ng_flow_kg_h} kg/h")
    
    def _build_simple_coordinator(self) -> None:
        """Build simple wind coordinator for testing."""
        from h2_plant.components.coordination.simple_wind_coordinator import SimpleWindCoordinator
        
        electrolyzer = self.registry.get(ComponentID.ELECTROLYZER)
        environment = self.registry.get(ComponentID.ENVIRONMENT_MANAGER)
        
        # Check if SOEC exists
        soec = None
        try:
            soec = self.registry.get(ComponentID.SOEC_CLUSTER)
            print(f"DEBUG: Found SOEC in registry: {soec}")
        except Exception as e:
            print(f"DEBUG: SOEC not found in registry: {e}")
            # List all registered components for debugging
            print(f"DEBUG: All registered components: {self.registry._components.keys()}")
        
        coordinator = SimpleWindCoordinator(environment, electrolyzer, soec)
        self.registry.register('wind_coordinator', coordinator, component_type='coordination')
        logger.debug(f"Registered SimpleWindCoordinator (PEM + {('SOEC' if soec else 'none')})")
    
    def _build_storage(self) -> None:
        """Build storage components from configuration."""
        stor_cfg = self.config.storage
        
        # Helper to calculate volume from capacity (Ideal Gas Law)
        # V = (m * R_specific * T) / P
        # R_specific H2 approx 4124 J/(kg*K)
        R_H2 = 4124.0
        T_STD = 298.15
        
        # LP Tanks (Buffer)
        # Calculate total volume for the array
        lp_total_mass = stor_cfg.lp_tanks.count * stor_cfg.lp_tanks.capacity_kg
        lp_pressure_pa = stor_cfg.lp_tanks.pressure_bar * 1e5
        lp_volume = (lp_total_mass * R_H2 * T_STD) / lp_pressure_pa
        
        lp_tanks = H2StorageTankEnhanced(
            tank_id="lp_tanks",
            volume_m3=lp_volume,
            initial_pressure_bar=1.0, # Start empty/atmospheric
            max_pressure_bar=stor_cfg.lp_tanks.pressure_bar
        )
        self.registry.register(ComponentID.LP_TANKS, lp_tanks, component_type='storage')

        # HP Storage Logic
        if stor_cfg.source_isolated:
            # ... (Keep existing logic for source isolated for now, or update it too?)
            # The user asked to integrate into "main plant_config.yaml topology".
            # h2plant_detailed.yaml does NOT use source_isolated (it's false by default/omission in the yaml I saw, wait let me check).
            # In h2plant_detailed.yaml:
            # storage:
            #   lp_tanks: ...
            #   hp_tanks: ...
            # No source_isolated: true.
            
            # So I will focus on the standard HP tanks branch first.
            # If source_isolated IS used, I should probably update it too, but let's stick to the main path.
            
            sources = {}
            
            # Check for Electrolysis sources (PEM OR SOEC)
            has_pem = (self.config.production.electrolyzer and 
                      self.config.production.electrolyzer.enabled)
            
            # Check SOEC config (handles both dict and object access)
            has_soec = False
            if hasattr(self.config, 'soec_cluster'):
                if isinstance(self.config.soec_cluster, dict):
                    has_soec = self.config.soec_cluster.get('enabled', False)
                else:
                    has_soec = getattr(self.config.soec_cluster, 'enabled', False)

            # Register 'electrolyzer' source if either green pathway exists
            if has_pem or has_soec:
                # Tag 0.0 emissions for green H2
                sources['electrolyzer'] = SourceTag('electrolyzer', 'electrolyzer', 0.0)

            # Check ATR source
            if self.config.production.atr and self.config.production.atr.enabled:
                # Tag 10.5 emissions (approx) for blue H2
                sources['atr'] = SourceTag('atr', 'atr', 10.5)

            iso_cfg = stor_cfg.isolated_config
            if iso_cfg:
                hp_storage_manager = SourceIsolatedTanks(
                    sources=sources,
                    tanks_per_source=iso_cfg.electrolyzer_tanks.count,
                    capacity_kg=iso_cfg.electrolyzer_tanks.capacity_kg,
                    pressure_bar=iso_cfg.electrolyzer_tanks.pressure_bar
                )
                self.registry.register(ComponentID.HP_STORAGE_MANAGER, hp_storage_manager, component_type='storage_manager')
                
                # Register individual arrays for visibility
                for source_name, tank_array in hp_storage_manager._tank_arrays.items():
                    # Keep dynamic IDs as strings for now, or use a convention
                    self.registry.register(f'{source_name}_hp_tanks', tank_array, component_type='storage')
                
                # Oxygen buffer if needed
                o2_buffer = OxygenBuffer(capacity_kg=iso_cfg.oxygen_buffer_capacity_kg)
                self.registry.register(ComponentID.OXYGEN_BUFFER, o2_buffer, component_type='storage')
            else:
                raise ConfigurationError("isolated_config is missing for source_isolated storage.")
        else:
            # Standard mixed storage
            # Calculate total volume
            hp_total_mass = stor_cfg.hp_tanks.count * stor_cfg.hp_tanks.capacity_kg
            hp_pressure_pa = stor_cfg.hp_tanks.pressure_bar * 1e5
            hp_volume = (hp_total_mass * R_H2 * T_STD) / hp_pressure_pa
            
            hp_tanks = H2StorageTankEnhanced(
                tank_id="hp_tanks",
                volume_m3=hp_volume,
                initial_pressure_bar=1.0, # Start empty
                max_pressure_bar=stor_cfg.hp_tanks.pressure_bar
            )
            self.registry.register(ComponentID.HP_TANKS, hp_tanks, component_type='storage')
            
        logger.debug("Registered storage components (Enhanced)")
    
    def _build_compression(self) -> None:
        """Build compression components from configuration."""
        comp_cfg = self.config.compression
        
        filling_comp = FillingCompressor(**comp_cfg.filling_compressor.__dict__)
        self.registry.register(ComponentID.FILLING_COMPRESSOR, filling_comp, component_type='compression')
        
        outgoing_comp = OutgoingCompressor(**comp_cfg.outgoing_compressor.__dict__)
        self.registry.register(ComponentID.OUTGOING_COMPRESSOR, outgoing_comp, component_type='compression')
        
        logger.debug("Registered compression components")
    
    def _build_utilities(self) -> None:
        """Build utility components from configuration."""
        demand_cfg = self.config.demand
        custom_profile = None
        if demand_cfg.pattern == 'custom' and demand_cfg.custom_profile_file:
            custom_profile = self._load_profile_file(demand_cfg.custom_profile_file)
        
        demand_scheduler = DemandScheduler(**{k:v for k,v in demand_cfg.__dict__.items() if k != 'custom_profile_file'}, custom_profile=custom_profile)
        self.registry.register(ComponentID.DEMAND_SCHEDULER, demand_scheduler, component_type='utility')
        self.registry.register("demand_scheduler_0", demand_scheduler, component_type='utility')
        
        price_cfg = self.config.energy_price
        if price_cfg.source == 'file' and price_cfg.price_file:
            prices = self._load_profile_file(price_cfg.price_file)
        else:
            prices = np.full(self.config.simulation.duration_hours, price_cfg.constant_price_per_mwh)
        
        energy_tracker = EnergyPriceTracker(
            prices_per_mwh=prices, 
            default_price_per_mwh=price_cfg.constant_price_per_mwh,
            data_resolution_minutes=price_cfg.data_resolution_minutes
        )
        self.registry.register(ComponentID.ENERGY_PRICE_TRACKER, energy_tracker, component_type='utility')
        
        logger.debug("Registered utility components")

    def _build_external_inputs(self) -> None:
        """Build external input components."""
        print(f"DEBUG: _build_external_inputs called. Config has external_inputs: {bool(self.config.external_inputs)}")
        if not self.config.external_inputs:
            return
        
        ext_inputs = self.config.external_inputs
        print(f"DEBUG: ext_inputs keys/attrs: {ext_inputs.__dict__.keys() if hasattr(ext_inputs, '__dict__') else ext_inputs}")
        
        if ext_inputs.oxygen_source and ext_inputs.oxygen_source.enabled:
            params = {k: v for k, v in ext_inputs.oxygen_source.__dict__.items() if k != 'enabled'}
            o2_source = ExternalOxygenSource(**params)
            self.registry.register(ComponentID.EXTERNAL_OXYGEN_SOURCE, o2_source, component_type='external_input')
            self.registry.register("oxygen_source_0", o2_source, component_type='external_input')
        
        if ext_inputs.heat_source and ext_inputs.heat_source.enabled:
            params = {k: v for k, v in ext_inputs.heat_source.__dict__.items() if k != 'enabled'}
            heat_source = ExternalHeatSource(**params)
            self.registry.register(ComponentID.EXTERNAL_HEAT_SOURCE, heat_source, component_type='external_input')
            self.registry.register("heat_source_0", heat_source, component_type='external_input')
            
        if ext_inputs.biogas_source:
            print(f"DEBUG: biogas_source type: {type(ext_inputs.biogas_source)}")
            print(f"DEBUG: biogas_source value: {ext_inputs.biogas_source}")
            
            # Check enabled status safely
            is_enabled = False
            if isinstance(ext_inputs.biogas_source, dict):
                is_enabled = ext_inputs.biogas_source.get('enabled', False)
            else:
                is_enabled = getattr(ext_inputs.biogas_source, 'enabled', False)
            
            print(f"DEBUG: biogas_source enabled: {is_enabled}")
            
            if is_enabled:
                from h2_plant.components.external.biogas_source import BiogasSource
                # Handle dict vs object just in case
                if isinstance(ext_inputs.biogas_source, dict):
                    params = ext_inputs.biogas_source.copy()
                    if 'enabled' in params: del params['enabled']
                else:
                    params = {k: v for k, v in ext_inputs.biogas_source.__dict__.items() if k != 'enabled'}
                    
                biogas = BiogasSource(component_id="biogas_source", **params)
                self.registry.register("biogas_source", biogas, component_type='external_input')
                self.registry.register("biogas_source_0", biogas, component_type='external_input')
                print("DEBUG: Registered biogas_source")
            
        logger.debug("Registered external input components")

    def _build_oxygen_management(self) -> None:
        """Build oxygen management (mixer or buffer)."""
        if not self.config.oxygen_management:
            # If not defined, an oxygen buffer might have been created in _build_storage
            if not self.registry.has(ComponentID.OXYGEN_BUFFER.value):
                 logger.debug("No oxygen management configured.")
            return

        o2_mgmt = self.config.oxygen_management
        if o2_mgmt.use_mixer and o2_mgmt.mixer:
            mixer = OxygenMixer(**o2_mgmt.mixer.__dict__)
            self.registry.register(ComponentID.OXYGEN_MIXER, mixer, component_type='oxygen_management')
            logger.debug("Registered OxygenMixer")
        elif not self.registry.has(ComponentID.OXYGEN_BUFFER.value):
             # Default to simple buffer if not using mixer and one wasn't created in isolated storage
             pass

    def _build_battery(self) -> None:
        """Build battery storage (if enabled)."""
        if not self.config.battery or not self.config.battery.enabled:
            return
        
        battery_params = {k: v for k, v in self.config.battery.__dict__.items() if k != 'enabled'}
        battery = BatteryStorage(**battery_params)
        self.registry.register(ComponentID.BATTERY, battery, component_type='energy_storage')
        logger.debug("Registered BatteryStorage")
    
    def _build_water_treatment(self) -> None:
        """Build water treatment components."""
        if not self.config.water_treatment:
            return

        wt_cfg = self.config.water_treatment
        
        if wt_cfg.quality_test.enabled:
            qt_params = {k: v for k, v in wt_cfg.quality_test.__dict__.items() if k != 'enabled'}
            quality_test = WaterQualityTestBlock(**qt_params)
            self.registry.register(ComponentID.WATER_QUALITY_TEST, quality_test, component_type='water')

        if wt_cfg.treatment_block.enabled:
            tb_params = {k: v for k, v in wt_cfg.treatment_block.__dict__.items() if k != 'enabled'}
            treatment_block = WaterTreatmentBlock(**tb_params)
            self.registry.register(ComponentID.WATER_TREATMENT_BLOCK, treatment_block, component_type='water')

        if wt_cfg.ultrapure_storage:
            # Storage config might not have enabled, but safer to filter if it does
            st_params = {k: v for k, v in wt_cfg.ultrapure_storage.__dict__.items() if k != 'enabled'}
            storage = UltrapureWaterStorageTank(**st_params)
            self.registry.register(ComponentID.ULTRAPURE_WATER_STORAGE, storage, component_type='water')
            self.registry.register("ultrapure_water_storage_0", storage, component_type='water')
            
        if wt_cfg.pumps.pump_a.enabled:
            pa_params = {k: v for k, v in wt_cfg.pumps.pump_a.__dict__.items() if k != 'enabled'}
            pump_a = WaterPump(pump_id='pump_a', **pa_params)
            self.registry.register(ComponentID.WATER_PUMP_A, pump_a, component_type='water')

        if wt_cfg.pumps.pump_b.enabled:
            pb_params = {k: v for k, v in wt_cfg.pumps.pump_b.__dict__.items() if k != 'enabled'}
            pump_b = WaterPump(pump_id='pump_b', **pb_params)
            self.registry.register(ComponentID.WATER_PUMP_B, pump_b, component_type='water')
        
        logger.debug("Registered water treatment components")

    def _load_profile_file(self, file_path: str) -> np.ndarray:
        """Load time-series profile from CSV or NPY file."""
        path = Path(file_path)
        
        if not path.exists():
            raise ConfigurationError(f"Profile file not found: {file_path}")
        
        if path.suffix == '.npy':
            return np.load(path)
        elif path.suffix == '.csv':
            try:
                return np.loadtxt(path, delimiter=',', skiprows=1)
            except ValueError:
                 # Try reading with pandas if numpy fails (e.g. mixed types or complex header)
                 import pandas as pd
                 df = pd.read_csv(path)
                 # Assume the data is in the second column if multiple, or first if single
                 if len(df.columns) > 1:
                     return df.iloc[:, 1].values
                 return df.iloc[:, 0].values
        else:
            raise ConfigurationError(f"Unsupported profile format: {path.suffix}")


    def _build_pem_system(self) -> None:
        """Build PEM electrolysis system from detailed component configuration."""
        # Check for PEM config
        pem_cfg = self.config.pem_system
        
        # Handle dict vs object
        if isinstance(pem_cfg, dict):
            enabled = pem_cfg.get('enabled', False)
            max_power = pem_cfg.get('max_power_mw', 5.0)
            base_eff = pem_cfg.get('base_efficiency', 0.65)
            use_poly = pem_cfg.get('use_polynomials', False)
        elif pem_cfg:
            enabled = pem_cfg.enabled
            max_power = pem_cfg.max_power_mw
            base_eff = pem_cfg.base_efficiency
            use_poly = getattr(pem_cfg, 'use_polynomials', False)
        else:
            enabled = False
            
        if not enabled:
            logger.debug("PEM electrolyzer disabled")
            return
        
        logger.info("Building detailed PEM electrolyzer")
        
        # Create detailed PEM electrolyzer
        pem_config_dict = {
            'max_power_mw': max_power,
            'base_efficiency': base_eff,
            'use_polynomials': use_poly
        }
        pem = DetailedPEMElectrolyzer(pem_config_dict)
        
        self.registry.register(ComponentID.PEM_ELECTROLYZER_DETAILED, pem, component_type='pem_production')
        # Alias for indexed topology
        self.registry.register("pem_electrolyzer_detailed_0", pem, component_type='pem_production')
        logger.debug(f"Registered DetailedPEMElectrolyzer: {max_power} MW")

    def _build_soec_system(self) -> None:
        """Build SOEC multi-module cluster using reference implementation wrapper."""
        # Check for SOEC config
        soec_cfg = self.config.soec_cluster
        
        # Handle dict vs object
        if isinstance(soec_cfg, dict):
            enabled = soec_cfg.get('enabled', False)
            num_modules = soec_cfg.get('num_modules', 6)
            max_power_nom = soec_cfg.get('max_power_nominal_mw', 2.4)
            optimal_limit = soec_cfg.get('optimal_limit', 0.80)
            rotation = soec_cfg.get('rotation_enabled', False)
            off_modules = soec_cfg.get('real_off_modules', [])
        elif soec_cfg:
            enabled = soec_cfg.enabled
            num_modules = soec_cfg.num_modules
            max_power_nom = soec_cfg.max_power_nominal_mw
            optimal_limit = getattr(soec_cfg, 'optimal_limit', 0.80)
            rotation = getattr(soec_cfg, 'rotation_enabled', False)
            off_modules = getattr(soec_cfg, 'real_off_modules', [])
        else:
            enabled = False
            
        if not enabled:
            logger.debug("SOEC cluster disabled")
            return
        
        # Create SOEC cluster
        if getattr(soec_cfg, 'use_native', False) or (isinstance(soec_cfg, dict) and soec_cfg.get('use_native', False)):
            logger.info("Building SOEC cluster using NATIVE implementation")
            from h2_plant.components.production.soec_cluster import SOECMultiModuleCluster
            soec = SOECMultiModuleCluster(
                num_modules=num_modules,
                max_power_nominal_mw=max_power_nom,
                rotation_enabled=rotation,
                efficient_threshold=optimal_limit
            )
        else:
            logger.info("Building SOEC cluster using reference implementation wrapper")
            soec = SOECClusterWrapper(
                num_modules=num_modules,
                max_nominal_power_mw=max_power_nom,
                optimal_limit=optimal_limit,
                rotation_enabled=rotation,
                real_off_modules=off_modules
            )
        
        self.registry.register(ComponentID.SOEC_CLUSTER, soec, component_type='soec_production')
        # Alias for indexed topology
        self.registry.register("soec_cluster_0", soec, component_type='soec_production')
        logger.debug(f"Registered SOECClusterWrapper: {num_modules} modules, {num_modules * max_power_nom * optimal_limit:.2f} MW capacity")

    def _build_atr_system(self) -> None:
        """Build ATR reforming system from detailed component configuration."""
        if not self.config.atr_system:
            return
        
        atr = self.config.atr_system
        logger.info("Building ATR system from detailed components")
        
        from h2_plant.components.reforming.atr_reactor import ATRReactor
        from h2_plant.components.reforming.wgs_reactor import WGSReactor
        
        # Build ATR reactors
        for reactor_cfg in atr.get('reactors', []):
            reactor = ATRReactor(
                component_id=reactor_cfg.get('component_id', 'ATR-1'),
                max_flow_kg_h=reactor_cfg.get('max_flow_kg_h', 1500.0),
                model_path=reactor_cfg.get('model_path', 'to_integrate/ATR_model_functions.pkl')
            )
            self.registry.register(reactor.component_id, reactor, component_type='atr_production')
        
        # Build WGS reactors
        for wgs_cfg in atr.get('wgs_reactors', []):
            wgs = WGSReactor(
                component_id=wgs_cfg.get('component_id', 'WGS-HT'),
                conversion_rate=wgs_cfg.get('conversion_rate', 0.7)
            )
            self.registry.register(wgs.component_id, wgs, component_type='atr_conversion')
            # Aliases for indexed topology (assuming order: HT=0, LT=1)
            if wgs.component_id == 'WGS-HT':
                self.registry.register("wgs_reactor_0", wgs, component_type='atr_conversion')
            elif wgs.component_id == 'WGS-LT':
                self.registry.register("wgs_reactor_1", wgs, component_type='atr_conversion')

        # Alias for ATR-1 -> atr_reactor_0
        if self.registry.has('ATR-1'):
            self.registry.register("atr_reactor_0", self.registry.get('ATR-1'), component_type='atr_production')

    def _build_logistics(self) -> None:
        """Build logistics components (consumers/refueling stations)."""
        if not self.config.logistics:
            return
        
        logistics = self.config.logistics
        logger.info("Building logistics components")
        
        from h2_plant.components.logistics.consumer import Consumer
        
        for consumer_cfg in logistics.get('consumers', []):
            consumer = Consumer(
                num_bays=consumer_cfg.get('num_bays', 4),
                filling_rate_kg_h=consumer_cfg.get('filling_rate_kg_h', 50.0)
            )
            component_id = consumer_cfg.get('component_id', 'consumer_1')
            self.registry.register(component_id, consumer, component_type='logistics')
    
    def _build_environment_manager(self) -> None:
        """Build and register Environment Manager for time-series data."""
        from h2_plant.components.environment.environment_manager import EnvironmentManager
        
        # Check if config specifies custom paths
        wind_path = None
        price_path = None
        
        # Check for wind_turbines config (power input file)
        if hasattr(self.config, 'wind_turbines') and self.config.wind_turbines:
            if hasattr(self.config.wind_turbines, 'data_file'):
                wind_path = self.config.wind_turbines.data_file
                print(f"DEBUG PlantBuilder: Setting wind_path = {wind_path}")
        else:
            print(f"DEBUG PlantBuilder: No wind_turbines config found")
                
        # Check for energy_price config
        if hasattr(self.config, 'energy_price') and self.config.energy_price:
            if hasattr(self.config.energy_price, 'price_file'):
                price_path = self.config.energy_price.price_file
                print(f"DEBUG PlantBuilder: Setting price_path = {price_path}")
            # WORKAROUND: Also check for wind_data_file in energy_price section
            if hasattr(self.config.energy_price, 'wind_data_file') and wind_path is None:
                wind_path = self.config.energy_price.wind_data_file
                print(f"DEBUG PlantBuilder: Setting wind_path from energy_price = {wind_path}")
        
        print(f"DEBUG PlantBuilder: Creating EnvironmentManager with wind_path={wind_path}, price_path={price_path}")
        
        # Environment manager always uses default paths unless specified
        env_manager = EnvironmentManager(
            wind_data_path=wind_path,
            price_data_path=price_path,
            use_default_data=(wind_path is None and price_path is None)
        )
        
        self.registry.register(ComponentID.ENVIRONMENT_MANAGER, env_manager, component_type='environment')
        logger.debug("Registered Environment Manager")
    
    def _build_dual_path_coordinator(self) -> None:
        """Build dual-path coordinator for hybrid PEM/SOEC dispatch."""
        # Check if both PEM and SOEC are enabled
        # Check for PEM
        pem_cfg = getattr(self.config, 'pem_system', None)
        if isinstance(pem_cfg, dict):
            has_pem = pem_cfg.get('enabled', False)
        elif pem_cfg:
            has_pem = pem_cfg.enabled
        else:
            # Fallback to legacy production.electrolyzer
            has_pem = (self.config.production and 
                       self.config.production.electrolyzer and 
                       self.config.production.electrolyzer.enabled)

        # Check for SOEC
        soec_cfg = getattr(self.config, 'soec_cluster', None)
        if isinstance(soec_cfg, dict):
            has_soec = soec_cfg.get('enabled', False)
        elif soec_cfg:
            has_soec = soec_cfg.enabled
        else:
            has_soec = False
        
        if not (has_pem or has_soec):
            logger.debug("Dual-path coordinator not needed (no PEM/SOEC)")
            return
        
        logger.info("Building dual-path coordinator")
        
        # Get config if available
        coord_cfg = getattr(self.config, 'dual_path_coordinator', {})
        
        # Import from correct location
        from h2_plant.pathways.dual_path_coordinator import DualPathCoordinator
        
        # Determine pathway IDs
        pathway_ids = []
        if has_soec:
            pathway_ids.append(ComponentID.SOEC_CLUSTER.value)
        if has_pem:
            pathway_ids.append(ComponentID.PEM_ELECTROLYZER_DETAILED.value)
            
        # Create coordinator
        coordinator = DualPathCoordinator(
            pathway_ids=pathway_ids,
            allocation_strategy=self.config.pathway.allocation_strategy
        )
        
        # Set arbitrage threshold if available (it's not in init but used in logic)
        # Actually, the logic in step() uses constants or hardcoded values currently.
        # We should probably pass it or set it.
        # My refactored DualPathCoordinator uses hardcoded constants in _execute_dispatch_logic for now.
        # I should update it to use the config value later.
        
        self.registry.register(ComponentID.DUAL_PATH_COORDINATOR, coordinator, component_type='coordination')
        logger.debug("Registered DualPathCoordinator")

    def _build_thermal_manager(self) -> None:
        """Build and register Thermal Manager."""
        from h2_plant.components.thermal.thermal_manager import ThermalManager
        
        manager = ThermalManager()
        self.registry.register(ComponentID.THERMAL_MANAGER, manager, component_type='thermal_management')
        logger.debug("Registered ThermalManager")

    def _build_water_balance_tracker(self) -> None:
        """Build and register Water Balance Tracker."""
        from h2_plant.components.water.water_balance_tracker import WaterBalanceTracker
        
        tracker = WaterBalanceTracker()
        self.registry.register(ComponentID.WATER_BALANCE_TRACKER, tracker, component_type='water_management')
        logger.debug("Registered WaterBalanceTracker")

    def _build_thermal_components(self) -> None:
        """Build thermal management components (chillers, HX)."""
        if not self.config.thermal_components:
            return
        
        cfg = self.config.thermal_components
        
        # Build chillers (HX-1 through HX-11)
        # Build chillers (HX-1 through HX-11)
        for i in range(cfg.chillers):
            chiller = Chiller(
                component_id=f"chiller_{i}",
                cooling_capacity_kw=500.0,  # Default, can be customized later
                efficiency=0.95
            )
            self.registry.register(f"chiller_{i}", chiller, component_type='thermal')
            
        # Build Steam Generators (HX-4, HX-7)
        from h2_plant.components.thermal.steam_generator import SteamGenerator
        for i in range(cfg.steam_generators):
            sg = SteamGenerator(
                component_id=f"steam_generator_{i}",
                max_capacity_kg_h=500.0,
                efficiency=0.90
            )
            self.registry.register(f"steam_generator_{i}", sg, component_type='thermal')
            
        # Build Heat Exchangers (Generic)
        from h2_plant.components.thermal.heat_exchanger import HeatExchanger
        for i in range(cfg.heat_exchangers):
            hx = HeatExchanger(
                component_id=f"heat_exchanger_{i}",
                max_heat_removal_kw=500.0
            )
            self.registry.register(f"heat_exchanger_{i}", hx, component_type='thermal')
        
        logger.debug(f"Registered {cfg.chillers} chillers, {cfg.steam_generators} steam generators, {cfg.heat_exchangers} heat exchangers")

    def _build_separation_components(self) -> None:
        """Build separation units (tanks, PSA)."""
        if not self.config.separation_components:
            return
            
        cfg = self.config.separation_components
        
        from h2_plant.components.separation.separation_tank import SeparationTank
        
        # Separation tanks (ST-1 through ST-4)
        for i in range(cfg.separation_tanks):
            tank = SeparationTank(
                component_id=f"separation_tank_{i}",
                volume_m3=5.0
            )
            self.registry.register(f"separation_tank_{i}", tank, component_type='separation')
        
        # PSA units (D-1 through D-4)
        for i in range(cfg.psa_units):
            psa = PSA(
                component_id=f"psa_{i}",
                num_beds=2
            )
            self.registry.register(f"psa_{i}", psa, component_type='separation')
            
        logger.debug(f"Registered separation components: {cfg.separation_tanks} tanks, {cfg.psa_units} PSAs")

    def _build_power_components(self) -> None:
        """Build power conditioning components."""
        if not self.config.power_components:
            return
            
        cfg = self.config.power_components
        
        # Rectifiers (RT-1, RT-2)
        for i in range(cfg.rectifiers):
            rectifier = Rectifier(
                component_id=f"rectifier_{i}",
                rated_power_mw=10.0
            )
            self.registry.register(f"rectifier_{i}", rectifier, component_type='power')
            
        logger.debug(f"Registered {cfg.rectifiers} rectifiers")

    def _build_fluid_system(self) -> None:
        """Build fluid handling components (pumps, compressors)."""
        if not self.config.fluid_components:
            return
            
        cfg = self.config.fluid_components
        
        from h2_plant.components.water.pump import WaterPump
        from h2_plant.components.compression.filling_compressor import FillingCompressor
        
        # Pumps (P-1, P-2, P-3)
        for i in range(cfg.pumps):
            pump = WaterPump(
                pump_id=f"pump_{i}",
                power_kw=5.0,
                power_source="grid",
                outlet_pressure_bar=10.0
            )
            self.registry.register(f"pump_{i}", pump, component_type='fluid')
            
        # Compressors (C-1 through C-7)
        # Using FillingCompressor as generic compressor for now
        for i in range(cfg.compressors):
            comp = FillingCompressor(
                max_flow_kg_h=100.0,
                inlet_pressure_bar=1.0,  # Placeholder
                outlet_pressure_bar=30.0 # Placeholder
            )
            self.registry.register(f"compressor_{i}", comp, component_type='compression')
            
        logger.debug(f"Registered fluid components: {cfg.pumps} pumps, {cfg.compressors} compressors")

    def _build_h2_distribution(self) -> None:
        """Build H2 Distribution Line."""
        from h2_plant.components.mixing.h2_distribution import H2Distribution
        
        # Create distribution line
        # Config might not exist, use defaults
        dist_line = H2Distribution(
            component_id="h2_distribution",
            num_inputs=3
        )
        self.registry.register("h2_distribution", dist_line, component_type='distribution')
        self.registry.register("h2_distribution_0", dist_line, component_type='distribution')
        logger.debug("Registered H2Distribution")

    def _build_co2_system(self) -> None:
        """Build CO2 Capture and Storage."""
        from h2_plant.components.carbon.co2_capture_detailed import CO2CaptureUnit
        from h2_plant.components.carbon.co2_storage import CO2Storage
        
        # CO2 Capture
        capture = CO2CaptureUnit(
            capture_id="co2_capture",
            max_flow_kg_h=500.0
        )
        self.registry.register("co2_capture", capture, component_type='separation')
        self.registry.register("co2_capture_0", capture, component_type='separation')
        
        # CO2 Storage
        storage = CO2Storage(
            component_id="co2_storage",
            capacity_kg=10000.0
        )
        self.registry.register("co2_storage", storage, component_type='storage')
        self.registry.register("co2_storage_0", storage, component_type='storage')
        
        logger.debug("Registered CO2 system")

