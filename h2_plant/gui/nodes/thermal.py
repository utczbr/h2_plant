
"""
Thermal component nodes.
Covers: Chiller, DryCooler, Interchanger, ElectricBoiler, Attemperator, CoolingManager.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode


class ChillerNode(ConfigurableNode):
    """Chiller/cooler node for thermal management of gas streams."""
    __identifier__ = 'nodes.Thermal'
    NODE_NAME = 'Chiller'

    def __init__(self):
        super(ChillerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('fluid_in', flow_type='gas')
        self.add_output('fluid_out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='HX-1', tab='Properties')

        # Chiller Tab
        self.add_float_property(
            'cooling_capacity_kw', default=100.0, min_val=1.0, unit='kW', tab='Chiller'
        )
        self.add_float_property(
            'target_temp_c', default=25.0, min_val=-50.0, max_val=300.0, unit='°C', tab='Chiller'
        )
        self.add_float_property(
            'cop', default=4.0, min_val=1.0, max_val=10.0, unit='', tab='Chiller'
        )
        self.add_float_property(
            'pressure_drop_bar', default=0.2, min_val=0.0, max_val=5.0, unit='bar', tab='Chiller'
        )

        self.add_color_property('node_color', default=(100, 200, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class DryCoolerNode(ConfigurableNode):
    """Dry Cooler node for ambient air cooling."""
    __identifier__ = 'nodes.Thermal'
    NODE_NAME = 'Dry Cooler'

    def __init__(self):
        super(DryCoolerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('fluid_in', flow_type='gas')
        self.add_output('fluid_out', flow_type='gas')
        self.add_input('electricity_in', flow_type='electricity')

    def _init_properties(self):
        self.add_text_property('component_id', default='DC-1', tab='Properties')

        # Dry Cooler Tab
        self.add_float_property(
            'fan_power_kw', default=10.0, min_val=0.1, unit='kW', tab='Dry Cooler'
        )
        self.add_float_property(
            'delta_t_air_c', default=5.0, min_val=1.0, max_val=20.0, unit='°C', tab='Dry Cooler'
        )
        self.add_float_property(
            'approach_temp_c', default=5.0, min_val=1.0, max_val=20.0, unit='°C', tab='Dry Cooler'
        )
        self.add_float_property(
            'pressure_drop_bar', default=0.05, min_val=0.0, max_val=1.0, unit='bar', tab='Dry Cooler'
        )

        self.add_color_property('node_color', default=(255, 165, 0), tab='Custom') # Orange
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class InterchangerNode(ConfigurableNode):
    """Counter-flow heat exchanger (interchanger) for hot/cold stream exchange."""
    __identifier__ = 'nodes.Thermal'
    NODE_NAME = 'Interchanger'

    def __init__(self):
        super(InterchangerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('hot_in', flow_type='gas')
        self.add_input('cold_in', flow_type='water')
        self.add_output('hot_out', flow_type='gas')
        self.add_output('cold_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='IX-1', tab='Properties')

        self.add_float_property(
            'min_approach_temp_k', default=10.0, min_val=1.0, max_val=50.0,
            unit='K', tab='Interchanger'
        )
        self.add_float_property(
            'target_cold_out_temp_c', default=80.0, min_val=0.0, max_val=900.0,
            unit='°C', tab='Interchanger'
        )
        self.add_percentage_property(
            'efficiency', default=95.0, tab='Interchanger'
        )

        self.add_color_property('node_color', default=(255, 180, 100), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class ElectricBoilerNode(ConfigurableNode):
    """Electric boiler / heater for fluid stream heating."""
    __identifier__ = 'nodes.Thermal'
    NODE_NAME = 'Electric Boiler'

    def __init__(self):
        super(ElectricBoilerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('fluid_in', flow_type='gas')
        self.add_output('fluid_out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='EB-1', tab='Properties')

        self.add_float_property(
            'target_outlet_temp_c', default=80.0, min_val=0.0, max_val=1000.0,
            unit='°C', tab='Electric Boiler'
        )
        self.add_float_property(
            'max_power_kw', default=500.0, min_val=1.0, unit='kW',
            tab='Electric Boiler'
        )
        self.add_percentage_property(
            'efficiency', default=98.0, tab='Electric Boiler'
        )

        self.add_color_property('node_color', default=(255, 120, 80), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class AttemperatorNode(ConfigurableNode):
    """Attemperator for steam temperature control via water injection."""
    __identifier__ = 'nodes.Thermal'
    NODE_NAME = 'Attemperator'

    def __init__(self):
        super(AttemperatorNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('steam_in', flow_type='water')
        self.add_input('water_in', flow_type='water')
        self.add_output('steam_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='ATT-1', tab='Properties')

        self.add_float_property(
            'target_temp_k', default=430.0, min_val=300.0, max_val=900.0,
            unit='K', tab='Attemperator'
        )
        self.add_float_property(
            'min_superheat_delta_k', default=5.0, min_val=0.0, max_val=50.0,
            unit='K', tab='Attemperator'
        )
        self.add_float_property(
            'max_water_flow_kg_h', default=500.0, min_val=0.0,
            unit='kg/h', tab='Attemperator'
        )
        self.add_float_property(
            'pressure_drop_bar', default=0.5, min_val=0.0, max_val=5.0,
            unit='bar', tab='Attemperator'
        )

        self.add_color_property('node_color', default=(200, 150, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class CoolingManagerNode(ConfigurableNode):
    """Global cooling manager — configuration-only node with no ports."""
    __identifier__ = 'nodes.Thermal'
    NODE_NAME = 'Cooling Manager'

    def __init__(self):
        super(CoolingManagerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        pass  # CoolingManager is a global utility with no port connections

    def _init_properties(self):
        self.add_text_property('component_id', default='CM-1', tab='Properties')

        self.add_float_property(
            'dc_total_area_m2', default=5000.0, min_val=1.0,
            unit='m²', tab='Cooling Manager'
        )
        self.add_float_property(
            'dc_air_flow_kg_s', default=1000.0, min_val=1.0,
            unit='kg/s', tab='Cooling Manager'
        )
        self.add_float_property(
            'dc_u_value', default=35.0, min_val=1.0,
            unit='W/(m²·K)', tab='Cooling Manager'
        )
        self.add_float_property(
            'glycol_inventory_kg', default=50000.0, min_val=1.0,
            unit='kg', tab='Cooling Manager'
        )
        self.add_float_property(
            'tower_design_load_kw', default=300.0, min_val=1.0,
            unit='kW', tab='Cooling Manager'
        )
        self.add_float_property(
            't_dry_bulb_c', default=25.0, min_val=-30.0, max_val=50.0,
            unit='°C', tab='Cooling Manager'
        )
        self.add_float_property(
            't_wet_bulb_c', default=18.0, min_val=-30.0, max_val=40.0,
            unit='°C', tab='Cooling Manager'
        )

        self.add_color_property('node_color', default=(100, 180, 220), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
