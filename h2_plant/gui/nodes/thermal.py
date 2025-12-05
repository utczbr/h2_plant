
"""
Thermal component nodes (Heat Exchanger / Chiller).
Refactored to follow the new collapsible pattern.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class HeatExchangerNode(ConfigurableNode):
    __identifier__ = 'h2_plant.thermal.hx'
    NODE_NAME = 'Heat Exchanger'

    def __init__(self):
        super(HeatExchangerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('hot_in', flow_type='default')
        self.add_input('cold_in', flow_type='default')
        self.add_output('hot_out', flow_type='default')
        self.add_output('cold_out', flow_type='default')

    def _init_properties(self):
        self.add_text_property('component_id', default='HX-1', tab='Properties')

        # Heat Exchanger Tab
        self.add_float_property(
            'cooling_capacity_kw', default=500.0, min_val=1.0, unit='kW', tab='Heat Exchanger'
        )
        self.add_float_property(
            'outlet_temp_setpoint_c', default=25.0, min_val=-50.0, max_val=300.0, unit='°C', tab='Heat Exchanger'
        )

        self.add_color_property('node_color', default=(255, 100, 100), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
