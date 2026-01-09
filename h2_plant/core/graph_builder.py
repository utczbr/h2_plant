from typing import Dict, List, Any
import logging
from h2_plant.config.models import SimulationContext, ComponentNode
from h2_plant.core.component import Component

# Import Components
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.storage.h2_tank import TankArray
from h2_plant.components.storage.h2_storage_enhanced import H2StorageTankEnhanced
from h2_plant.components.storage.detailed_tank import DetailedTankArray
from h2_plant.components.delivery.discharge_station import DischargeStation
from h2_plant.components.compression.compressor import CompressorStorage as Compressor
from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.components.balance_of_plant.pump import Pump
from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer as Mixer
from h2_plant.components.control.valve import ThrottlingValve as Valve
from h2_plant.components.external.biogas_source import BiogasSource
from h2_plant.components.water.drain_recorder_mixer import DrainRecorderMixer
from h2_plant.components.water.makeup_mixer import MakeupMixer
from h2_plant.components.atr.atr_makeup_mixer import ProportionalMakeupMixer
from h2_plant.components.external.oxygen_makeup import OxygenMakeupNode
from h2_plant.components.water.water_balance_tracker import WaterBalanceTracker
from h2_plant.components.thermal.interchanger import Interchanger
from h2_plant.components.thermal.attemperator import Attemperator
from h2_plant.components.water.water_pump import WaterPumpThermodynamic
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.core.component_ids import ComponentID

# ATR Components
from h2_plant.components.reforming.atr_thermal_components import Boiler as ATR_Boiler
from h2_plant.components.reforming.atr_system_components import ATRSystemCompressor, ATRProductSeparator
from h2_plant.components.reforming.atr_recovery import ATRSyngasCooler


# Passive Components (Placeholder implementations for now)
class PassiveComponent(Component):
    def initialize(self, dt, registry): super().initialize(dt, registry)
    def step(self, t): pass
    def get_state(self): return {"id": self.component_id}

logger = logging.getLogger(__name__)

class PlantGraphBuilder:
    """
    Constructs the simulation graph from the SimulationContext.
    Instantiates components and injects physics parameters.
    """
    def __init__(self, context: SimulationContext):
        self.context = context
        self.components: Dict[str, Component] = {}

    def build(self) -> Dict[str, Component]:
        """
        Builds and returns the dictionary of components.
        """
        logger.info("Building plant graph...")
        
        # Create and register LUTManager FIRST so other components can use it
        lut_manager = LUTManager()
        lut_manager.set_component_id(ComponentID.LUT_MANAGER.value)
        self.components[ComponentID.LUT_MANAGER.value] = lut_manager
        logger.info(f"Registered LUTManager for component optimization")
        
        for node in self.context.topology.nodes:
            component = self._create_component(node)
            if component:
                component.set_component_id(node.id)
                
                # Inject graph grouping metadata from topology params
                if node.params:
                    component.system_group = node.params.get('system_group')
                    component.process_step = int(node.params.get('process_step', 0))
                
                self.components[node.id] = component
                logger.info(f"Created component: {node.id} ({node.type})")
                
        return self.components

    def _create_component(self, node: ComponentNode) -> Component:
        # Wrapper with logging
        comp = self._create_component_internal(node)
        if comp:
             logger.info(f"Initialized {node.id} ({node.type}) as {type(comp).__name__}")
        else:
             logger.warning(f"Failed to initialize {node.id} ({node.type})")
        return comp

    def _create_component_internal(self, node: ComponentNode) -> Component:
        """Factory method to create component based on type."""
        
        # Strip metadata keys used for graph grouping (injected later in build())
        # This prevents components that don't accept **kwargs from failing
        METADATA_KEYS = {'system_group', 'process_step'}
        if node.params:
            # Create a filtered copy of params
            filtered_params = {k: v for k, v in node.params.items() if k not in METADATA_KEYS}
            # Temporarily replace node.params for component construction
            original_params = node.params
            node.params = filtered_params
        else:
            original_params = None
        
        try:
            component = self._create_component_by_type(node)
        finally:
            # Restore original params to avoid mutation
            if original_params is not None:
                node.params = original_params
        
        return component
    
    def _create_component_by_type(self, node: ComponentNode) -> Component:
        """Internal factory - creates component by type after params are filtered."""
        if node.type == "PEM":
            # Inject PEM Physics Spec AND merge with node.params for overrides
            # Convert physics_spec Pydantic model to dict, then merge node.params
            if hasattr(self.context.physics, 'pem_system'):
                physics_dict = self.context.physics.pem_system.model_dump() if hasattr(self.context.physics.pem_system, 'model_dump') else dict(self.context.physics.pem_system)
            else:
                physics_dict = {}
            
            # Merge node.params to allow YAML overrides (e.g., out_pressure_pa)
            if node.params:
                physics_dict.update(node.params)
            
            # Add component_id from node.id
            physics_dict['component_id'] = node.id
            
            return DetailedPEMElectrolyzer(physics_dict)
            
        elif node.type == "SOEC":
            # Start with physics defaults if available
            soec_config = {}
            if hasattr(self.context.physics, 'soec_cluster'):
                 soec_config = self.context.physics.soec_cluster.model_dump() if hasattr(self.context.physics.soec_cluster, 'model_dump') else dict(self.context.physics.soec_cluster)
            
            # Merge/Override with node params
            if node.params:
                soec_config.update(node.params)
                
            # Ensure component_id is set
            soec_config['component_id'] = node.id
            
            return SOECOperator(soec_config)
            
        elif node.type == "Compressor":
            # Inject defaults if params are missing
            params = node.params.copy() if node.params else {}
            if 'max_flow_kg_h' not in params:
                params['max_flow_kg_h'] = 500.0  # Default value
            if 'inlet_pressure_bar' not in params:
                params['inlet_pressure_bar'] = 30.0
            if 'outlet_pressure_bar' not in params:
                params['outlet_pressure_bar'] = 200.0
            return Compressor(**params)
            
        elif node.type == "CompressorSingle":
            # Single-stage adiabatic compressor (no intercooling)
            params = node.params.copy() if node.params else {}
            if 'max_flow_kg_h' not in params:
                params['max_flow_kg_h'] = 500.0
            if 'inlet_pressure_bar' not in params:
                params['inlet_pressure_bar'] = 1.0
            if 'outlet_pressure_bar' not in params:
                params['outlet_pressure_bar'] = 30.0
            return CompressorSingle(**params)
            
        elif node.type == "Tank":
            # Simple vectorized tank array
            n_tanks = int(node.params.get('n_tanks', 1))
            capacity = float(node.params.get('capacity_kg', 1000.0))
            pressure = float(node.params.get('max_pressure_bar', 200.0))
            temp = float(node.params.get('temperature_k', 298.15))
            max_out_flow = float(node.params.get('max_output_flow_kg_h', 5000.0))
            return TankArray(
                n_tanks=n_tanks, 
                capacity_kg=capacity, 
                pressure_bar=pressure, 
                temperature_k=temp,
                max_output_flow_kg_h=max_out_flow
            )
            
        elif node.type == "Tank_Enhanced":
            # Enhanced single tank with PVT dynamics
            tank_id = node.params.get('tank_id', node.id)
            volume = float(node.params.get('volume_m3', 10.0))
            init_p = float(node.params.get('initial_pressure_bar', 40.0))
            max_p = float(node.params.get('max_pressure_bar', 350.0))
            return H2StorageTankEnhanced(tank_id=tank_id, volume_m3=volume, initial_pressure_bar=init_p, max_pressure_bar=max_p)
            
        elif node.type == "Pump":
            # Extract explicit params to avoid kwargs issues (Pump doesn't accept **kwargs)
            # Default mapping: design_flow_kg_h -> capacity_kg_h
            capacity = float(node.params.get('capacity_kg_h', 1000.0))
            target_p = float(node.params.get('target_pressure_bar', 30.0))
            eta_is = float(node.params.get('eta_is', 0.82))
            eta_m = float(node.params.get('eta_m', 0.96))
            return Pump(target_pressure_bar=target_p, eta_is=eta_is, eta_m=eta_m, capacity_kg_h=capacity)
            
        elif node.type == "Mixer":
            vol = float(node.params.get('volume_m3', 10.0))
            continuous_flow = bool(node.params.get('continuous_flow', True))
            return Mixer(volume_m3=vol, continuous_flow=continuous_flow)

        elif node.type == "WaterMixer":
            from h2_plant.components.mixing.water_mixer import WaterMixer
            p_out = float(node.params.get('outlet_pressure_kpa', 200.0))
            max_in = int(node.params.get('max_inlet_streams', 10))
            return WaterMixer(outlet_pressure_kpa=p_out, max_inlet_streams=max_in)

        elif node.type == "Chiller":
            from h2_plant.components.thermal.chiller import Chiller
            return Chiller(**node.params)

        elif node.type == "Coalescer":
            from h2_plant.components.separation.coalescer import Coalescer
            return Coalescer(**node.params)

        elif node.type == "KnockOutDrum":
            from h2_plant.components.separation.knock_out_drum import KnockOutDrum
            d_m = float(node.params.get('diameter_m', 1.0))
            dp_bar = float(node.params.get('delta_p_bar', 0.05))
            species = node.params.get('gas_species', 'H2')
            return KnockOutDrum(diameter_m=d_m, delta_p_bar=dp_bar, gas_species=species)

        elif node.type == "HydrogenMultiCyclone":
            from h2_plant.components.separation.hydrogen_cyclone import HydrogenMultiCyclone
            element_d_mm = float(node.params.get('element_diameter_mm', 50.0))
            vane_angle = float(node.params.get('vane_angle_deg', 45.0))
            target_vel = float(node.params.get('target_velocity_ms', 20.0))
            gas_species = node.params.get('gas_species', 'H2')
            return HydrogenMultiCyclone(
                element_diameter_mm=element_d_mm,
                vane_angle_deg=vane_angle,
                target_velocity_ms=target_vel,
                gas_species=gas_species
            )
            
        elif node.type == "DeoxoReactor":
            from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
            return DeoxoReactor(node.id)


        elif node.type == "PSA Unit":
            from h2_plant.components.separation.psa import PSA
            return PSA(
                component_id=node.id,
                **node.params
            )

        elif node.type == "SyngasPSA":
            from h2_plant.components.separation.psa_syngas import SyngasPSA
            return SyngasPSA(
                component_id=node.id,
                **node.params
            )

        elif node.type == "TSA Unit":
            from h2_plant.components.separation.tsa_unit import TSAUnit
            # Map params safely
            return TSAUnit(
                component_id=node.id,
                bed_diameter_m=float(node.params.get('bed_diameter_m', 0.32)),
                bed_length_m=float(node.params.get('bed_length_m', 0.8)),
                cycle_time_hours=float(node.params.get('cycle_time_hours', 6.0)),
                regen_temp_k=float(node.params.get('regen_temp_k', 523.15))
                # Add other optional params if needed
            )

        elif node.type == "Valve":
            return Valve(node.params)
            
        elif node.type == "ATR":
            from h2_plant.components.reforming.atr_reactor import ATRReactor
            # ATRReactor requires max_flow_kg_h
            max_flow = float(node.params.get('max_flow_kg_h', 1000.0))
            return ATRReactor(node.id, max_flow_kg_h=max_flow)

        elif node.type == "ATR_Boiler":
            params = node.params or {}
            lookup_id = params.get('lookup_id', params.get('component_id', None))
            return ATR_Boiler(node.id, lookup_id=lookup_id)

        # ATR_HeatExchanger, ATR_HTWGS, ATR_LTWGS removed - replaced by IntegratedATRPlant

        elif node.type == "IntegratedATRPlant":
            from h2_plant.components.reforming.integrated_atr_plant import IntegratedATRPlant
            max_flow = float(node.params.get('max_flow_kg_h', 20000.0))
            pressure_drop = float(node.params.get('pressure_drop_bar', 2.5))
            return IntegratedATRPlant(
                component_id=node.id,
                max_flow_kg_h=max_flow,
                pressure_drop_bar=pressure_drop
            )

        elif node.type == "ATR_SystemCompressor":
            return ATRSystemCompressor(node.id)

        elif node.type == "ATR_ProductSeparator":
            return ATRProductSeparator(node.id)

        elif node.type == "ATRSyngasCooler":
            lookup_id = node.params.get('lookup_id', "Tin_H05")
            return ATRSyngasCooler(node.id, lookup_id=lookup_id)
            
        elif node.type == "WaterPurifier":
            from h2_plant.components.water.water_purifier import WaterPurifier
            max_flow = float(node.params.get('max_flow_kg_h', 2000.0))
            return WaterPurifier(node.id, max_flow_kg_h=max_flow)
            
        elif node.type == "UltraPureWaterTank":
            from h2_plant.components.water.ultrapure_water_tank import UltraPureWaterTank
            # Convert volume_m3 (from GUI) to capacity_kg (approx 1000 kg/m3)
            vol = float(node.params.get('volume_m3', 5.0))
            cap_kg = vol * 1000.0
            return UltraPureWaterTank(node.id, capacity_kg=cap_kg)
            
        elif node.type == "Battery":
            from h2_plant.components.storage.battery_storage import BatteryStorage
            # Map params or use defaults
            capacity_kwh = float(node.params.get('capacity_kwh', 1000.0))
            max_power_kw = float(node.params.get('max_power_kw', 500.0))
            return BatteryStorage(node.id, capacity_kwh=capacity_kwh, max_power_kw=max_power_kw)
            
        elif node.type == "DryCooler":
            from h2_plant.components.cooling.dry_cooler import DryCooler
            comp_id = node.params.get('component_id', node.id)
            target_temp_c = node.params.get('target_outlet_temp_c', node.params.get('target_temp_c', None))
            if target_temp_c is not None:
                target_temp_c = float(target_temp_c)
            return DryCooler(component_id=comp_id, target_outlet_temp_c=target_temp_c)

        elif node.type == "DryCoolerSimplified":
            from h2_plant.components.cooling.dry_cooler_simplified import DryCoolerSimplified
            return DryCoolerSimplified(
                component_id=node.id,
                target_temp_k=float(node.params.get('target_temp_k', 313.15)),
                pressure_drop_bar=float(node.params.get('pressure_drop_bar', 0.05)),
                fan_specific_power_kw_per_mw=float(node.params.get('fan_specific_power_kw_per_mw', 15.0))
            )
            
        elif node.type == "Chiller":
            from h2_plant.components.thermal.chiller import Chiller
            return Chiller(
                component_id=node.params.get('component_id', node.id),
                cooling_capacity_kw=float(node.params.get('cooling_capacity_kw', 100.0)),
                target_temp_k=float(node.params.get('target_temp_k', 278.15)),
                cop=float(node.params.get('cop', 4.0))
            )
            
        elif node.type == "HeatExchanger":
            from h2_plant.components.thermal.heat_exchanger import HeatExchanger
            return HeatExchanger(
                component_id=node.params.get('component_id', node.id),
                max_heat_removal_kw=float(node.params.get('max_heat_removal_kw', 50.0)),
                target_outlet_temp_c=float(node.params.get('target_outlet_temp_c', 25.0))
            )
            
        elif node.type == "ElectricBoiler":
            from h2_plant.components.thermal.electric_boiler import ElectricBoiler
            config = {
                'max_power_kw': float(node.params.get('max_power_kw', 1000.0)),
                'efficiency': float(node.params.get('efficiency', 0.99)),
                'design_pressure_bar': float(node.params.get('design_pressure_bar', 10.0))
            }
            if 'target_temp_c' in node.params:
                config['target_temp_c'] = float(node.params['target_temp_c'])
            return ElectricBoiler(config=config)
            
        elif node.type == "Coalescer":
            from h2_plant.components.separation.coalescer import Coalescer
            return Coalescer(component_id=node.params.get('component_id', node.id))
            
        elif node.type == "Consumer":
             from h2_plant.components.logistics.consumer import Consumer
             return Consumer(node.id, node.params.get('daily_demand_kg', 0.0))
             
        elif node.type == "GridConnection":
             # Placeholder for Grid Connection
             return PassiveComponent()
             
        elif node.type == "WaterSupply":
            from h2_plant.components.external.water_source import ExternalWaterSource
            return ExternalWaterSource(node.params)

        elif node.type == "H2Source":
            from h2_plant.components.external.h2_source import ExternalH2Source
            return ExternalH2Source(config=node.params)

        elif node.type == "ExternalOxygenSource":
            from h2_plant.components.external.oxygen_source import ExternalOxygenSource
            return ExternalOxygenSource(config=node.params)

        elif node.type == "WaterSource" or node.type == "ExternalWaterSource":
            from h2_plant.components.external.water_source import ExternalWaterSource
            return ExternalWaterSource(node.params)

        elif node.type == "DrainRecorderMixer":
            # Pass explicit source_ids if provided
            sources = node.params.get("source_ids", None)
            return DrainRecorderMixer(source_ids=sources)

        elif node.type == "MakeupMixer":
            return MakeupMixer(
                component_id=node.id,
                target_flow_kg_h=float(node.params.get("target_flow_kg_h", 100.0)),
                makeup_temp_c=float(node.params.get("makeup_temp_c", 5.0)),
                makeup_pressure_bar=float(node.params.get("makeup_pressure_bar", 1.0))
            )

        elif node.type == "ProportionalMakeupMixer":
            return ProportionalMakeupMixer(
                component_id=node.id,
                max_flow_rate_kg_h=float(node.params.get("max_flow_rate_kg_h", 2331.95)),
                makeup_temp_c=float(node.params.get("makeup_temp_c", 20.0)),
                makeup_pressure_bar=float(node.params.get("makeup_pressure_bar", 15.0)),
                reference_component_id=node.params.get("reference_component_id", None),
                reference_ratio=node.params.get("reference_ratio", None),
                reference_max_flow_kg_h=node.params.get("reference_max_flow_kg_h", None)
            )

        elif node.type == "Interchanger":
            return Interchanger(
                component_id=node.id,
                min_approach_temp_k=float(node.params.get("min_approach_temp_k", 10.0)),
                target_cold_out_temp_c=float(node.params.get("target_cold_out_temp_c", 95.0)),
                efficiency=float(node.params.get("efficiency", 0.95))
            )
        
        elif node.type == "WaterPumpThermodynamic":
             capacity = float(node.params.get('capacity_kg_h', 1000.0))
             target_p = float(node.params.get('target_pressure_pa', 500000.0))
             eta_is = float(node.params.get('eta_is', 0.80))
             eta_m = float(node.params.get('eta_m', 0.95))
             pump_id = node.params.get('pump_id', node.id)
             return WaterPumpThermodynamic(
                 pump_id=pump_id,
                 target_pressure_pa=target_p,
                 eta_is=eta_is,
                 eta_m=eta_m
             )

        elif node.type == "BiogasSource":
            return BiogasSource(
                component_id=node.id,
                pressure_bar=float(node.params.get('pressure_bar', 5.0)),
                temperature_c=float(node.params.get('temperature_c', 25.0)),
                max_flow_rate_kg_h=float(node.params.get('max_flow_rate_kg_h', 1000.0)),
                methane_content=float(node.params.get('methane_content', 0.60)),
                # Proportional control params
                reference_component_id=node.params.get('reference_component_id', None),
                reference_ratio=node.params.get('reference_ratio', None),
                reference_max_flow_kg_h=node.params.get('reference_max_flow_kg_h', None)
            )

        elif node.type == "OxygenMakeupNode":
            return OxygenMakeupNode(
                component_id=node.id,
                target_flow_kg_h=node.params.get('target_flow_kg_h', None),
                min_target_flow_kg_h=node.params.get('min_target_flow_kg_h', None),
                max_limit_flow_kg_h=node.params.get('max_limit_flow_kg_h', None),
                supply_pressure_bar=float(node.params.get('supply_pressure_bar', 15.0)),
                supply_temperature_c=float(node.params.get('supply_temperature_c', 25.0)),
                supply_purity=float(node.params.get('supply_purity', 0.995))
            )

        elif node.type == "Attemperator":
            return Attemperator(
                component_id=node.id,
                target_temp_k=float(node.params.get('target_temp_k', 623.15)),
                max_water_flow_kg_h=float(node.params.get('max_water_flow_kg_h', 1000.0)),
                pressure_drop_bar=float(node.params.get('pressure_drop_bar', 0.5)),
                min_superheat_delta_k=float(node.params.get('min_superheat_delta_k', 5.0))
            )

        elif node.type == "StreamSplitter":
            from h2_plant.components.mixing.stream_splitter import StreamSplitter
            return StreamSplitter(node.params, component_id=node.id)

        elif node.type == "WaterBalanceTracker":
            return WaterBalanceTracker()

        elif node.type == "DetailedTank":
            # High-fidelity tank array with state machines
            return DetailedTankArray(
                n_tanks=int(node.params.get('n_tanks', 10)),
                volume_per_tank_m3=float(node.params.get('volume_per_tank_m3', 50.0)),
                max_pressure_bar=float(node.params.get('max_pressure_bar', 500.0)),
                initial_pressure_bar=float(node.params.get('initial_pressure_bar', 1.0)),
                ambient_temp_k=float(node.params.get('ambient_temp_k', 293.15))
            )

        elif node.type == "DischargeStation":
            # Truck loading station with compression energy
            return DischargeStation(
                station_id=int(node.params.get('station_id', 1)),
                truck_capacity_kg=float(node.params.get('truck_capacity_kg', 1000.0)),
                delivery_pressure_bar=float(node.params.get('delivery_pressure_bar', 500.0)),
                max_fill_rate_kg_min=float(node.params.get('max_fill_rate_kg_min', 60.0)),
                isen_efficiency=float(node.params.get('isen_efficiency', 0.75)),
                mech_efficiency=float(node.params.get('mech_efficiency', 0.95)),
                cooldown_minutes=float(node.params.get('cooldown_minutes', 150.0)),
                arrival_probability=float(node.params.get('arrival_probability', 0.3))
            )

        else:
            logger.warning(f"Unknown component type: {node.type} (ID: {node.id}) -> Instantiating PassiveComponent")
            return PassiveComponent()
