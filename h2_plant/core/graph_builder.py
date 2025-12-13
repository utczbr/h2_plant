from typing import Dict, List, Any
import logging
from h2_plant.config.models import SimulationContext, ComponentNode
from h2_plant.core.component import Component

# Import Components
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.balance_of_plant.tank import Tank
from h2_plant.components.compression.compressor import CompressorStorage as Compressor
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
        """Factory method to create component based on type."""
        
        if node.type == "PEM":
            # Inject PEM Physics Spec directly
            physics_spec = self.context.physics.pem_system
            return DetailedPEMElectrolyzer(physics_spec)
            
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
            
        elif node.type == "Tank":
            return Tank(node.params)
            
        elif node.type == "Pump":
            # Extract explicit params to avoid kwargs issues (Pump doesn't accept **kwargs)
            # Default mapping: design_flow_kg_h -> capacity_kg_h
            capacity = float(node.params.get('capacity_kg_h', 1000.0))
            target_p = float(node.params.get('target_pressure_bar', 30.0))
            eta_is = float(node.params.get('eta_is', 0.82))
            eta_m = float(node.params.get('eta_m', 0.96))
            return Pump(target_pressure_bar=target_p, eta_is=eta_is, eta_m=eta_m, capacity_kg_h=capacity)
            
        elif node.type == "Mixer":
            # Extract volume explicitly
            vol = float(node.params.get('volume_m3', 10.0))
            return Mixer(volume_m3=vol)

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
            from h2_plant.components.separation.psa_unit import PSAUnit
            # Extract specific params or use defaults
            gas_type = node.params.get('gas_type', 'H2')
            return PSAUnit(node.id, gas_type=gas_type)

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
            return DryCooler(**node.params)
            
        elif node.type == "Consumer":
             from h2_plant.components.logistics.consumer import Consumer
             return Consumer(node.id, node.params.get('daily_demand_kg', 0.0))
             
        elif node.type == "GridConnection":
             # Placeholder for Grid Connection
             return PassiveComponent()
             
        elif node.type == "WaterSupply":
             # Placeholder for Water Supply
             return PassiveComponent()

        else:
            logger.warning(f"Unknown component type: {node.type}")
            return PassiveComponent()
