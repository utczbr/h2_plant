
"""
External resource nodes (grid, water supply, ambient heat, natural gas).
Refactored to collapsible pattern with supply mode selector.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

SUPPLY_MODES = ['on_demand', 'scaled', 'constant']

class GridConnectionNode(ConfigurableNode):
    __identifier__ = 'nodes.Sources'
    NODE_NAME = 'Grid Connection'

    def __init__(self):
        super(GridConnectionNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        # Power exported from grid into plant
        self.add_output('power_out', flow_type='electricity')

    def _init_properties(self):
        # Identification
        self.add_text_property('component_id', default='Grid-1', tab='Properties')

        # Grid Connection Tab
        self.add_enum_property(
            'supply_mode', options=SUPPLY_MODES, default_index=0, tab='Grid Connection'
        )
        # For 'constant' mode this is a fixed available power; for 'scaled' it can be a factor.
        self.add_float_property(
            'mode_value', default=10000.0, min_val=0.0, unit='kW', tab='Grid Connection'
        )

        # Custom visuals
        self.add_color_property('node_color', default=(255, 255, 0), tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class WaterSupplyNode(ConfigurableNode):
    __identifier__ = 'nodes.Sources'
    NODE_NAME = 'Water Supply'

    def __init__(self):
        super(WaterSupplyNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_output('water_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='Water-Supply-1', tab='Properties')

        # Water Supply Tab
        self.add_enum_property(
            'supply_mode', options=SUPPLY_MODES, default_index=0, tab='Water Supply'
        )
        # For 'constant' this is a fixed max flow; for 'scaled' it's a factor.
        self.add_float_property(
            'mode_value', default=100.0, min_val=0.0, unit='m³/h', tab='Water Supply'
        )

        self.add_color_property('node_color', default=(100, 150, 255), tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class AmbientHeatNode(ConfigurableNode):
    __identifier__ = 'nodes.Sources'
    NODE_NAME = 'Ambient Heat Source'

    def __init__(self):
        super(AmbientHeatNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_output('heat_out', flow_type='heat')

    def _init_properties(self):
        self.add_text_property('component_id', default='Ambient-Heat-1', tab='Properties')

        # Ambient Heat Tab
        self.add_enum_property(
            'supply_mode', options=SUPPLY_MODES, default_index=0, tab='Ambient Heat'
        )
        self.add_float_property(
            'mode_value', default=1000.0, min_val=0.0, unit='kW', tab='Ambient Heat'
        )

        self.add_color_property('node_color', default=(255, 100, 100), tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class NaturalGasSupplyNode(ConfigurableNode):
    __identifier__ = 'nodes.Sources'
    NODE_NAME = 'Natural Gas Supply'

    def __init__(self):
        super(NaturalGasSupplyNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_output('gas_out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='NG-Supply-1', tab='Properties')

        # Natural Gas Supply Tab
        self.add_enum_property(
            'supply_mode', options=SUPPLY_MODES, default_index=0, tab='Natural Gas Supply'
        )
        self.add_float_property(
            'mode_value', default=500.0, min_val=0.0, unit='kg/h', tab='Natural Gas Supply'
        )

        self.add_color_property('node_color', default=(200, 200, 200), tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
