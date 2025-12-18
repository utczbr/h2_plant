from typing import Dict, List, Any
import logging
from h2_plant.config.models import SimulationContext, ComponentNode
from h2_plant.core.component import Component

# Import Components
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.storage.h2_tank import TankArray
from h2_plant.components.storage.h2_storage_enhanced import H2StorageTankEnhanced
from h2_plant.components.compression.compressor import CompressorStorage as Compressor
from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.components.balance_of_plant.pump import Pump
from h2_plant.components.mixing.multicomponent_mixer import MultiComponentMixer as Mixer
from h2_plant.components.control.valve import ThrottlingValve as Valve

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
        
        for node in self.context.topology.nodes:
            component = self._create_component(node)
            if component:
                component.set_component_id(node.id)
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
            # Inject SOEC Physics Spec directly
            physics_spec = self.context.physics.soec_cluster
            # SOECOperator now accepts the spec as the first argument
            return SOECOperator(physics_spec)
            
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
            return TankArray(n_tanks=n_tanks, capacity_kg=capacity, pressure_bar=pressure, temperature_k=temp)
            
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
            return Mixer(volume_m3=vol)

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
            
        elif node.type == "DeoxoReactor":
            from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
            return DeoxoReactor(node.id)

        elif node.type == "PSA Unit":
            from h2_plant.components.separation.psa import PSA
            return PSA(
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
            return DryCooler(component_id=comp_id)
            
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
            return ElectricBoiler(config={
                'max_power_kw': float(node.params.get('max_power_kw', 1000.0)),
                'efficiency': float(node.params.get('efficiency', 0.99)),
                'design_pressure_bar': float(node.params.get('design_pressure_bar', 10.0))
            })
            
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

        else:
            logger.warning(f"Unknown component type: {node.type} (ID: {node.id}) -> Instantiating PassiveComponent")
            return PassiveComponent()
