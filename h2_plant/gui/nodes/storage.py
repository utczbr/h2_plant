
"""
Storage component nodes.
Refactored to follow the new collapsible pattern.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class LPTankNode(ConfigurableNode):
    __identifier__ = 'h2_plant.storage.lp'
    NODE_NAME = 'LP Tank'

    def __init__(self):
        super(LPTankNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('h2_in', flow_type='hydrogen')
        self.add_output('h2_out', flow_type='hydrogen')

    def _init_properties(self):
        self.add_text_property('component_id', default='LP-Array-1', tab='Properties')

        # LP Tank Array Tab
        self.add_float_property(
            'tank_count', default=4.0, min_val=1.0, unit='tanks', tab='LP Tank Array'
        )
        self.add_float_property(
            'capacity_per_tank_kg', default=50.0, min_val=1.0, unit='kg', tab='LP Tank Array'
        )
        self.add_float_property(
            'operating_pressure_bar', default=30.0, min_val=1.0, max_val=100.0, unit='bar', tab='LP Tank Array'
        )
        self.add_percentage_property(
            'min_fill_level', default=5.0, tab='LP Tank Array'
        )
        self.add_percentage_property(
            'max_fill_level', default=95.0, tab='LP Tank Array'
        )
        self.add_float_property(
            'ambient_temp_c', default=20.0, min_val=-40.0, max_val=60.0, unit='°C', tab='LP Tank Array'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(0, 255, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)

class HPTankNode(ConfigurableNode):
    __identifier__ = 'h2_plant.storage.hp'
    NODE_NAME = 'HP Tank'

    def __init__(self):
        super(HPTankNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('h2_in', flow_type='compressed_h2')
        self.add_output('h2_out', flow_type='compressed_h2')

    def _init_properties(self):
        self.add_text_property('component_id', default='HP-Array-1', tab='Properties')

        # HP Tank Array Tab
        self.add_float_property(
            'tank_count', default=8.0, min_val=1.0, unit='tanks', tab='HP Tank Array'
        )
        self.add_float_property(
            'capacity_per_tank_kg', default=200.0, min_val=1.0, unit='kg', tab='HP Tank Array'
        )
        self.add_float_property(
            'operating_pressure_bar', default=350.0, min_val=100.0, max_val=900.0, unit='bar', tab='HP Tank Array'
        )
        self.add_percentage_property(
            'min_fill_level', default=5.0, tab='HP Tank Array'
        )
        self.add_percentage_property(
            'max_fill_level', default=95.0, tab='HP Tank Array'
        )
        self.add_float_property(
            'ambient_temp_c', default=20.0, min_val=-40.0, max_val=60.0, unit='°C', tab='HP Tank Array'
        )
        self.add_text_property(
            'material_type', default='Type IV Composite', tab='HP Tank Array'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(0, 200, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)

class OxygenBufferNode(ConfigurableNode):
    __identifier__ = 'h2_plant.storage.o2'
    NODE_NAME = 'Oxygen Buffer'

    def __init__(self):
        super(OxygenBufferNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('o2_in', flow_type='oxygen')
        self.add_output('o2_out', flow_type='oxygen')

    def _init_properties(self):
        self.add_text_property('component_id', default='O2-Buffer-1', tab='Properties')

        # Oxygen Buffer Tab
        self.add_float_property(
            'capacity_kg', default=500.0, min_val=1.0, unit='kg', tab='Oxygen Buffer'
        )
        self.add_float_property(
            'operating_pressure_bar', default=10.0, min_val=1.0, max_val=50.0, unit='bar', tab='Oxygen Buffer'
        )
        self.add_percentage_property(
            'min_fill_level', default=10.0, tab='Oxygen Buffer'
        )
        self.add_percentage_property(
            'max_fill_level', default=90.0, tab='Oxygen Buffer'
        )
        self.add_float_property(
            'ambient_temp_c', default=20.0, min_val=-40.0, max_val=60.0, unit='°C', tab='Oxygen Buffer'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(255, 200, 0), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=70)
