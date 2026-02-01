"""
Logistics component nodes (Consumer/Refueling Station).
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class ConsumerNode(ConfigurableNode):
    __identifier__ = 'nodes.Consumers'
    NODE_NAME = 'Refueling Station'
    
    def _init_ports(self):
        self.add_input('h2_in', flow_type='hydrogen', color=(0, 255, 255))

    def _init_properties(self):
        self.add_text_input('component_id', default='Refueling-Station-1')
        self.add_integer_input('num_bays', default=4, min_val=1)
        self.add_float_input('filling_rate_kg_h', default=50.0, min_val=1.0)
