
"""
Electrolysis component nodes (PEM, SOEC, Rectifier).
Refactored to follow the new collapsible pattern.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class PEMStackNode(ConfigurableNode):
    __identifier__ = 'h2_plant.electrolysis.pem'
    NODE_NAME = 'PEM'

    def __init__(self):
        super(PEMStackNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('power_in', flow_type='electricity')
        self.add_input('water_in', flow_type='water')

        self.add_output('h2_out', flow_type='hydrogen')
        self.add_output('o2_out', flow_type='oxygen')
        self.add_output('heat_out', flow_type='heat')

    def _init_properties(self):
        # Identity
        self.add_text_property('component_id', default='PEM-Stack-1', tab='Properties')

        # PEM Stack Tab
        self.add_float_property(
            'rated_power_kw', default=2500.0, min_val=1.0, unit='kW', tab='PEM Stack'
        )
        self.add_percentage_property(
            'efficiency_rated', default=65.0, tab='PEM Stack'
        )
        self.add_float_property( # Using float for int/count if int prop missing, or use default text
             'number_of_cells', default=85.0, min_val=1.0, unit='cells', tab='PEM Stack'
        )
        self.add_float_property(
             'active_area_m2', default=0.03, min_val=0.001, unit='m²', tab='PEM Stack'
        )
        self.add_float_property(
            'operating_temp_c', default=60.0, min_val=20.0, max_val=90.0, unit='°C', tab='PEM Stack'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(0, 255, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)

class RectifierNode(ConfigurableNode):
    __identifier__ = 'h2_plant.electrolysis.rectifier'
    NODE_NAME = 'Rectifier'

    def __init__(self):
        super(RectifierNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('ac_power_in', flow_type='electricity')
        self.add_output('dc_power_out', flow_type='electricity')

    def _init_properties(self):
        self.add_text_property('component_id', default='RT-1', tab='Properties')

        # Rectifier Tab
        self.add_float_property(
            'max_power_kw', default=2500.0, min_val=1.0, unit='kW', tab='Rectifier'
        )
        self.add_percentage_property(
            'conversion_efficiency', default=98.0, tab='Rectifier'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(255, 255, 0), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)

class SOECStackNode(ConfigurableNode):
    __identifier__ = 'h2_plant.electrolysis.soec'
    NODE_NAME = 'SOEC'

    def __init__(self):
        super(SOECStackNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('power_in', flow_type='electricity')
        self.add_input('steam_in', flow_type='water')
        self.add_output('h2_out', flow_type='hydrogen')
        self.add_output('o2_out', flow_type='oxygen') # Added Oxygen Output
        self.add_output('heat_out', flow_type='heat')

    def _init_properties(self):
        self.add_text_property('component_id', default='SOEC-Stack-1', tab='Properties')

        self.add_float_property(
            'rated_power_kw', default=1000.0, min_val=1.0, unit='kW', tab='SOEC Stack'
        )
        self.add_float_property(
            'operating_temp_c', default=800.0, min_val=600.0, max_val=1000.0, unit='°C', tab='SOEC Stack'
        )

        self.add_color_property('node_color', default=(255, 200, 100), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)
