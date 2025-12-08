
from h2_plant.gui.nodes.base_node import ConfigurableNode

class ValveNode(ConfigurableNode):
    """
    Throttling Valve node for pressure reduction.
    """
    __identifier__ = 'h2_plant.control.valve'
    NODE_NAME = 'Valve'

    def __init__(self):
        super(ValveNode, self).__init__()
        self.add_input('fluid_in', flow_type='gas')
        self.add_output('fluid_out', flow_type='gas')
        
    def _init_properties(self):
        self.add_text_property('component_id', default='V-1', tab='Properties')
        self.add_float_property('target_pressure_bar', default=20.0, min_val=0.1, max_val=1000.0, tab='Properties')
        self.add_float_property('pressure_drop_bar', default=0.0, tab='Properties') # Optional, if using delta p mode
