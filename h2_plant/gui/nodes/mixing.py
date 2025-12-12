"""
Mixing and flow control nodes.
"""
from h2_plant.gui.nodes.base_node import ConfigurableNode


class MixerNode(ConfigurableNode):
    """
    Gas/Fluid Mixer node.
    """
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Mixer (Generic)'

    def __init__(self):
        super(MixerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        # Multiple inputs converge to single output
        self.add_input('in_1', flow_type='default', multi_input=True)
        self.add_input('in_2', flow_type='default', multi_input=True)
        self.add_output('out', flow_type='default')

    def _init_properties(self):
        self.add_text_property('component_id', default='MIX-1', tab='Properties')

        # Mixer Tab
        self.add_float_property(
            'volume_m3', default=1.0, min_val=0.01,
            unit='m³', tab='Mixer'
        )
        self.add_enum_property(
            'fluid_type',
            options=['Water', 'Hydrogen', 'Oxygen', 'Natural Gas', 'Generic'],
            default_index=0,
            tab='Mixer'
        )
        self.add_float_property(
            'target_pressure_bar', default=5.0, min_val=0.0,
            unit='bar', tab='Mixer'
        )
        
        # Custom Tab
        self.add_color_property('node_color', default=(180, 180, 180), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)

class WaterMixerNode(ConfigurableNode):
    """
    Specialized Water Mixer Node.
    """
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Water Mixer'

    def __init__(self):
        super(WaterMixerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('in_1', flow_type='water')
        self.add_input('in_2', flow_type='water')
        self.add_input('in_3', flow_type='water')
        self.add_output('out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='W-MIX-1', tab='Properties')
        self.add_float_property('capacity_kg_h', default=2000.0, unit='kg/h', tab='Properties')
        self.add_float_property('volume_m3', default=0.5, min_val=0.01, unit='m³', tab='Properties')
        self.add_color_property('node_color', default=(0, 191, 255), tab='Custom') # Deep Sky Blue
