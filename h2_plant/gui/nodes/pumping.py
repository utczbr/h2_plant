
"""
Pumping component nodes (Water Pump).
"""
from h2_plant.gui.nodes.base_node import ConfigurableNode

class PumpNode(ConfigurableNode):
    """
    Water Pump Node with rigorous pressure/efficiency inputs.
    """
    __identifier__ = 'h2_plant.fluid.pump'
    NODE_NAME = 'Pump'

    def __init__(self):
        super(PumpNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='water', color=(100, 150, 255))
        self.add_output('outlet', flow_type='water', color=(100, 150, 255))

    def _init_properties(self):
        self.add_text_property('component_id', default='P-1', tab='Properties')

        # Pump Tab
        self.add_float_property(
            'target_pressure_bar', default=30.0, min_val=1.0, max_val=1000.0,
            unit='bar', tab='Pump'
        )
        self.add_percentage_property(
            'isentropic_efficiency', default=75.0, tab='Pump'
        )
        self.add_percentage_property(
            'mechanical_efficiency', default=95.0, tab='Pump'
        )
        
        # Power calc display (optional placeholder property)
        self.add_float_property(
            'design_flow_kg_h', default=1000.0, min_val=0.0,
            unit='kg/h', tab='Pump'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(100, 149, 237), tab='Custom') # Cornflower Blue
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)
