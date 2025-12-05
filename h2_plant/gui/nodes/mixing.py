"""
Mixing and flow control nodes.
"""
from h2_plant.gui.nodes.base_node import ConfigurableNode


class MixerNode(ConfigurableNode):
    """
    Mixer node for combining multiple flow inputs into a single output.
    Useful for combining H2 streams from multiple electrolyzers.
    """
    __identifier__ = 'h2_plant.flow.mixer'
    NODE_NAME = 'Mixer'

    def __init__(self):
        super(MixerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        # Multiple inputs converge to single output
        self.add_input('input_1', flow_type='hydrogen', multi_input=True)
        self.add_input('input_2', flow_type='hydrogen', multi_input=True)
        self.add_input('input_3', flow_type='hydrogen', multi_input=True)
        self.add_output('mixed_out', flow_type='hydrogen')

    def _init_properties(self):
        self.add_text_property('component_id', default='MIX-1', tab='Properties')

        # Mixer Tab
        self.add_enum_property(
            'flow_type',
            options=['hydrogen', 'water', 'oxygen', 'gas'],
            default_index=0,
            tab='Mixer'
        )
        self.add_float_property(
            'max_throughput_kg_h', default=500.0, min_val=1.0,
            unit='kg/h', tab='Mixer'
        )
        self.add_float_property(
            'pressure_drop_bar', default=0.5, min_val=0.0,
            unit='bar', tab='Mixer'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(180, 180, 180), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
