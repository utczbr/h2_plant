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
class ConnectionConfig:
    """Configuration for a connection between two components."""
    source_id: str
    source_port: str
    target_id: str
    target_port: str
    resource_type: str  # e.g., 'hydrogen', 'water', 'electricity'

    def validate(self) -> None:
        """Validate connection configuration."""
        if not self.source_id or not self.target_id:
            raise ValueError("Source and target IDs must be specified")
        if not self.source_port or not self.target_port:
            raise ValueError("Source and target ports must be specified")
        if not self.resource_type:
            raise ValueError("Resource type must be specified")


@dataclass
class IndexedConnectionConfig:
    """Configuration for indexed component connections."""
    source_name: str       # Component type name (e.g., 'pump', 'chiller')
    source_index: int      # Instance index (e.g., 0 for pump_0, 1 for pump_1)
    source_port: str
    target_name: str
    target_index: int
    target_port: str
    resource_type: str

    def validate(self) -> None:
        """Validate indexed connection configuration."""
        if not self.source_name or not self.target_name:
            raise ValueError("Source and target names must be specified")
        if self.source_index < 0 or self.target_index < 0:
            raise ValueError("Component indices must be non-negative")
        if not self.source_port or not self.target_port:
            raise ValueError("Source and target ports must be specified")
        if not self.resource_type:
            raise ValueError("Resource type must be specified")


@dataclass
class ProductionSourceConfig:
    """Base configuration for production sources."""
    enabled: bool = True
    max_capacity: float = 0.0  # MW for electrolyzer, kg/h for ATR


@dataclass
class ElectrolyzerConfig(ProductionSourceConfig):
    """Electrolyzer configuration."""
    max_power_mw: float = 15.0
    base_efficiency: float = 0.80
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
    degradation_rate_percent_1000h: float = 0.15 # %/1000h
    use_polynomials: bool = False # Optimization flag.
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
class PEMConfig:
    """Detailed PEM Electrolyzer configuration."""
    enabled: bool = True
    max_power_mw: float = 5.0
    base_efficiency: float = 0.65
    kwh_per_kg: float = 56.16  # User specified reference
    use_polynomials: bool = False # Optimization flag
    
    def validate(self) -> None:
        if self.max_power_mw <= 0:
            raise ValueError("PEM max power must be positive")
        if not 0 < self.base_efficiency <= 1.0:
            raise ValueError("PEM efficiency must be in (0, 1]")

@dataclass
class SOECConfig:
    """Detailed SOEC Cluster configuration."""
    enabled: bool = True
    num_modules: int = 6
    max_power_nominal_mw: float = 2.4  
    ramp_rate_mw_per_min: float = 0.05
    rotation_enabled: bool = True
    efficient_threshold: float = 0.80
    kwh_per_kg: float = 37.5  # User specified reference
    
    def validate(self) -> None:
        if self.num_modules <= 0:
            raise ValueError("SOEC modules must be positive")

@dataclass
class ProductionConfig:
    """Production sources configuration."""
    electrolyzer: Optional[ElectrolyzerConfig] = None
    soec: Optional[ElectrolyzerConfig] = None  # Use same config structure for simplicity
    atr: Optional[ATRConfig] = None
    
    def validate(self) -> None:
        """Validate production configuration."""
        if self.electrolyzer:
            self.electrolyzer.validate()
        if self.soec:
            self.soec.validate()
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
class OutgoingCompressorConfig:
    """Outgoing compressor configuration."""
    max_flow_kg_h: float = 100.0
    inlet_pressure_bar: float = 350.0
    outlet_pressure_bar: float = 900.0
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
    outgoing_compressor: OutgoingCompressorConfig = field(default_factory=lambda: OutgoingCompressorConfig(
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
    data_resolution_minutes: int = 60  # Resolution of the price data (e.g., 60 for hourly, 15 for 15-min)
    wind_data_file: Optional[str] = None  # Workaround for wind data file path
    
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
    arbitrage_threshold_eur_mwh: float = 338.29
    priority_order: List[str] = field(default_factory=lambda: ["arbitrage", "production"])
    
    def validate(self) -> None:
        """Validate pathway configuration."""
        # AllocationStrategy enum validates itself
        if self.arbitrage_threshold_eur_mwh < 0:
            raise ValueError("Arbitrage threshold must be non-negative")
        pass

@dataclass
class OxygenSourceConfig:
    enabled: bool = True
    mode: str = "fixed_flow"
    flow_rate_kg_h: float = 0.0
    pressure_bar: float = 5.0
    cost_per_kg: float = 0.15
    max_capacity_kg_h: float = 100.0

@dataclass
class HeatSourceConfig:
    enabled: bool = True
    thermal_power_kw: float = 500.0
    temperature_c: float = 150.0
    availability_factor: float = 1.0
    cost_per_kwh: float = 0.0
    min_output_fraction: float = 0.2

@dataclass
class BiogasSourceConfig:
    enabled: bool = True
    max_flow_rate_kg_h: float = 1000.0
    methane_content: float = 0.60
    pressure_bar: float = 5.0

@dataclass
class ExternalInputsConfig:
    oxygen_source: Optional[OxygenSourceConfig] = None
    heat_source: Optional[HeatSourceConfig] = None
    biogas_source: Optional[BiogasSourceConfig] = None

@dataclass
class MixerConfig:
    capacity_kg: float = 1000.0
    target_pressure_bar: float = 5.0
    target_temperature_c: float = 25.0
    input_sources: List[str] = field(default_factory=list)

@dataclass
class OxygenManagementConfig:
    use_mixer: bool = False
    mixer: Optional[MixerConfig] = None

@dataclass
class BatteryConfig:
    enabled: bool = False
    capacity_kwh: float = 1000.0
    max_charge_power_kw: float = 500.0
    max_discharge_power_kw: float = 500.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    min_soc: float = 0.20
    max_soc: float = 0.95
    initial_soc: float = 0.50

@dataclass
class WaterQualityTestConfig:
    enabled: bool = True
    sample_interval_hours: float = 1.0

@dataclass
class WaterTreatmentBlockConfig:
    enabled: bool = True
    max_flow_m3h: float = 10.0
    power_consumption_kw: float = 20.0

@dataclass
class UltrapureWaterStorageConfig:
    capacity_l: float = 5000.0
    initial_fill_ratio: float = 0.5

@dataclass
class PumpConfig:
    enabled: bool = True
    power_kw: float = 0.75
    power_source: str = "grid_or_battery"
    outlet_pressure_bar: float = 5.0

@dataclass
class WaterPumpsConfig:
    pump_a: PumpConfig = field(default_factory=PumpConfig)
    pump_b: PumpConfig = field(default_factory=lambda: PumpConfig(power_kw=1.5, power_source="grid", outlet_pressure_bar=8.0))

@dataclass
class WaterTreatmentConfig:
    quality_test: WaterQualityTestConfig = field(default_factory=WaterQualityTestConfig)
    treatment_block: WaterTreatmentBlockConfig = field(default_factory=WaterTreatmentBlockConfig)
    ultrapure_storage: UltrapureWaterStorageConfig = field(default_factory=UltrapureWaterStorageConfig)
    pumps: WaterPumpsConfig = field(default_factory=WaterPumpsConfig)


# ==================== Indexed Component Array Configurations ====================

@dataclass
class ThermalComponentsConfig:
    """Configuration for thermal management component arrays."""
    chillers: int = 11  # HX-1 through HX-11 in Process Flow
    heat_exchangers: int = 0
    steam_generators: int = 2  # HX-4, HX-7


@dataclass
class SeparationComponentsConfig:
    """Configuration for separation component arrays."""
    separation_tanks: int = 4  # ST-1 through ST-4
    psa_units: int = 4         # D-1 through D-4


@dataclass
class FluidComponentsConfig:
    """Configuration for fluid handling component arrays."""
    pumps: int = 3              # P-1, P-2, P-3
    compressors: int = 7        # C-1 through C-7


@dataclass
class PowerComponentsConfig:
    """Configuration for power conditioning component arrays."""
    rectifiers: int = 2         # RT-1 (PEM), RT-2 (SOEC)

@dataclass
class PlantConfig:
    """Complete hydrogen production plant configuration."""
    
    name: str = "Hydrogen Production Plant"
    version: str = "1.0"
    description: str = ""
    
    production: ProductionConfig = field(default_factory=ProductionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    energy_price: EnergyPriceConfig = field(default_factory=EnergyPriceConfig)
    pathway: PathwayConfig = field(default_factory=PathwayConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    external_inputs: Optional[ExternalInputsConfig] = None
    oxygen_management: Optional[OxygenManagementConfig] = None
    battery: Optional[BatteryConfig] = None
    water_treatment: Optional[WaterTreatmentConfig] = None
    
    # V2.0 Detailed System Configurations
    pem_system: Optional[PEMConfig] = None
    soec_cluster: Optional[SOECConfig] = None
    atr_system: Optional[Dict] = None
    logistics: Optional[Dict] = None
    
    # Topology (supports both legacy and indexed connections)
    topology: List[ConnectionConfig] = field(default_factory=list)
    indexed_topology: List[IndexedConnectionConfig] = field(default_factory=list)
    
    # V3.0 Indexed Component Arrays
    thermal_components: Optional[ThermalComponentsConfig] = None
    separation_components: Optional[SeparationComponentsConfig] = None
    fluid_components: Optional[FluidComponentsConfig] = None
    power_components: Optional[PowerComponentsConfig] = None

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
            if self.storage.isolated_config is None:
                raise ValueError("isolated_config required when source_isolated=True")
            
            # Ensure isolated storage matches production sources
            if self.production.electrolyzer and not (self.storage.isolated_config.electrolyzer_tanks.count > 0):
                raise ValueError("Electrolyzer configured but no electrolyzer tanks in isolated storage")
            if self.production.atr and not (self.storage.isolated_config.atr_tanks and self.storage.isolated_config.atr_tanks.count > 0):
                raise ValueError("ATR configured but no ATR tanks in isolated storage")
