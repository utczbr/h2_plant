
"""
Reforming component nodes (ATR, WGS, Steam Generator).
Refactored to follow the new collapsible pattern.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class ATRReactorNode(ConfigurableNode):
    __identifier__ = 'h2_plant.reforming.atr'
    NODE_NAME = 'ATR Reactor'

    def __init__(self):
        super(ATRReactorNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('oxygen_in', flow_type='oxygen')
        self.add_input('biogas_in', flow_type='gas')
        self.add_input('steam_in', flow_type='water')

        self.add_output('h2_out', flow_type='hydrogen')
        self.add_output('heat_out', flow_type='heat')

    def _init_properties(self):
        self.add_text_property('component_id', default='ATR-Reactor', tab='Properties')

        # ATR Reactor Tab
        self.add_float_property(
            'max_flow_kg_h', default=1500.0, min_val=1.0, unit='kg/h', tab='ATR Reactor'
        )
        self.add_text_property(
            'model_path', default='h2_plant/data/ATR_model_functions.pkl', tab='ATR Reactor'
        )
        self.add_float_property(
            'operating_temp_c', default=900.0, min_val=500.0, max_val=1200.0, unit='°C', tab='ATR Reactor'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(150, 100, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)

class WGSReactorNode(ConfigurableNode):
    __identifier__ = 'h2_plant.reforming.wgs'
    NODE_NAME = 'WGS Reactor'

    def __init__(self):
        super(WGSReactorNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('syngas_in', flow_type='gas')
        self.add_output('syngas_out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='WGS-HT', tab='Properties')

        # WGS Reactor Tab
        self.add_percentage_property('conversion_rate', default=70.0, tab='WGS Reactor')
        self.add_float_property(
            'operating_temp_c', default=350.0, min_val=200.0, max_val=500.0, unit='°C', tab='WGS Reactor'
        )

        self.add_color_property('node_color', default=(150, 150, 250), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)

class SteamGeneratorNode(ConfigurableNode):
    __identifier__ = 'h2_plant.reforming.steam_gen'
    NODE_NAME = 'Steam Generator'

    def __init__(self):
        super(SteamGeneratorNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('water_in', flow_type='water')
        self.add_input('heat_in', flow_type='heat')
        self.add_output('steam_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='HX-4', tab='Properties')

        # Steam Generator Tab
        self.add_float_property(
            'max_flow_kg_h', default=500.0, min_val=1.0, unit='kg/h', tab='Steam Generator'
        )
        self.add_float_property(
            'target_temp_c', default=150.0, min_val=100.0, max_val=300.0, unit='°C', tab='Steam Generator'
        )
        self.add_percentage_property('efficiency', default=90.0, tab='Steam Generator')

        self.add_color_property('node_color', default=(255, 150, 150), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
