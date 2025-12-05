
"""
Separation component nodes (PSA, Separation Tank).
Refactored to follow the new collapsible pattern.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class PSAUnitNode(ConfigurableNode):
    __identifier__ = 'h2_plant.separation.psa'
    NODE_NAME = 'PSA Unit'

    def __init__(self):
        super(PSAUnitNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('feed_in', flow_type='gas')
        self.add_output('product_out', flow_type='gas')
        self.add_output('waste_out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='D-1', tab='Properties')

        # PSA Unit Tab
        self.add_percentage_property('efficiency', default=85.0, tab='PSA Unit')
        self.add_percentage_property('recovery_rate', default=90.0, tab='PSA Unit')
        self.add_float_property(
            'operating_pressure_bar', default=30.0, min_val=1.0, unit='bar', tab='PSA Unit'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(200, 200, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)

class SeparationTankNode(ConfigurableNode):
    __identifier__ = 'h2_plant.separation.tank'
    NODE_NAME = 'Separation Tank'

    def __init__(self):
        super(SeparationTankNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('mixed_in', flow_type='gas')
        self.add_output('gas_out', flow_type='gas')
        self.add_output('liquid_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='ST-1', tab='Properties')

        # Separation Tank Tab
        self.add_float_property(
            'volume_m3', default=5.0, min_val=0.1, unit='m³', tab='Separation Tank'
        )
        self.add_float_property(
            'operating_pressure_bar', default=30.0, min_val=1.0, unit='bar', tab='Separation Tank'
        )

        self.add_color_property('node_color', default=(150, 150, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
