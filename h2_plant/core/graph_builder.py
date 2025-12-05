from typing import Dict, List, Any
import logging
from h2_plant.config.models import SimulationContext, ComponentNode
from h2_plant.core.component import Component

# Import Components
from h2_plant.components.electrolysis.pem_electrolyzer import DetailedPEMElectrolyzer
from h2_plant.components.electrolysis.soec_operator import SOECOperator
from h2_plant.components.balance_of_plant.compressor import Compressor
from h2_plant.components.balance_of_plant.pump import Pump
from h2_plant.components.balance_of_plant.tank import Tank

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
            config = node.dict()
            config.update(node.params)
            return Compressor(config)
            
        elif node.type == "Tank":
            config = node.dict()
            config.update(node.params)
            return Tank(config)
            
        elif node.type == "Pump":
            config = node.dict()
            config.update(node.params)
            return Pump(config)
            
        else:
            logger.warning(f"Unknown component type: {node.type}")
            return PassiveComponent()
