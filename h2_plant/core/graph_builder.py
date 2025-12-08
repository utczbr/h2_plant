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
            return Compressor(**node.params)
            
        elif node.type == "Tank":
            return Tank(node.params)
            
        elif node.type == "Pump":
            return Pump(**node.params)
            
        elif node.type == "Mixer":
            return Mixer(**node.params)

        elif node.type == "Chiller":
            from h2_plant.components.thermal.chiller import Chiller
            return Chiller(**node.params)

        elif node.type == "Coalescer":
            from h2_plant.components.separation.coalescer import Coalescer
            return Coalescer(**node.params)

        elif node.type == "KnockOutDrum":
            from h2_plant.components.separation.knock_out_drum import KnockOutDrum
            return KnockOutDrum(**node.params)
            
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

        else:
            logger.warning(f"Unknown component type: {node.type}")
            return PassiveComponent()
