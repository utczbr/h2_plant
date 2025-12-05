
"""
Compression component nodes for storage subsystem.
Refactored to follow the new collapsible pattern.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class FillingCompressorNode(ConfigurableNode):
    __identifier__ = 'h2_plant.compression.filling'
    NODE_NAME = 'Compressor Filling'

    def __init__(self):
        super(FillingCompressorNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('h2_in', flow_type='hydrogen')
        self.add_output('h2_out', flow_type='compressed_h2')

    def _init_properties(self):
        self.add_text_property('component_id', default='FC-1', tab='Properties')

        # Filling Compressor Tab
        self.add_float_property(
            'max_flow_kg_h', default=100.0, min_val=1.0, unit='kg/h', tab='Filling Compressor'
        )
        self.add_float_property(
            'inlet_pressure_bar', default=30.0, min_val=1.0, max_val=100.0, unit='bar', tab='Filling Compressor'
        )
        self.add_float_property(
            'outlet_pressure_bar', default=350.0, min_val=100.0, max_val=900.0, unit='bar', tab='Filling Compressor'
        )
        self.add_float_property(
            'num_stages', default=3.0, min_val=1.0, max_val=5.0, unit='stages', tab='Filling Compressor'
        )
        self.add_percentage_property(
            'efficiency', default=75.0, tab='Filling Compressor'
        )
        self.add_float_property(
            'power_consumption_kw', default=50.0, min_val=0.0, unit='kW', tab='Filling Compressor'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(100, 200, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)

class OutgoingCompressorNode(ConfigurableNode):
    __identifier__ = 'h2_plant.compression.outgoing'
    NODE_NAME = 'Compressor Outgoing'

    def __init__(self):
        super(OutgoingCompressorNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('h2_in', flow_type='compressed_h2')
        self.add_output('h2_out', flow_type='compressed_h2')

    def _init_properties(self):
        self.add_text_property('component_id', default='OC-1', tab='Properties')

        # Outgoing Compressor Tab
        self.add_float_property(
            'max_flow_kg_h', default=100.0, min_val=1.0, unit='kg/h', tab='Outgoing Compressor'
        )
        self.add_float_property(
            'inlet_pressure_bar', default=350.0, min_val=100.0, max_val=500.0, unit='bar', tab='Outgoing Compressor'
        )
        self.add_float_property(
            'outlet_pressure_bar', default=900.0, min_val=500.0, max_val=1000.0, unit='bar', tab='Outgoing Compressor'
        )
        self.add_percentage_property(
            'efficiency', default=75.0, tab='Outgoing Compressor'
        )
        self.add_float_property(
            'power_consumption_kw', default=75.0, min_val=0.0, unit='kW', tab='Outgoing Compressor'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(150, 200, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)
