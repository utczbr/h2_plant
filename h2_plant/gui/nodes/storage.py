"""
Storage and delivery nodes.
Covers: DetailedTank, DischargeStation, CompressorSingle.
"""
from h2_plant.gui.nodes.base_node import ConfigurableNode


class DetailedTankNode(ConfigurableNode):
    """High-pressure hydrogen storage tank bank."""
    __identifier__ = 'nodes.Storage'
    NODE_NAME = 'Detailed Tank'

    def __init__(self):
        super(DetailedTankNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('h2_in', flow_type='hydrogen')
        self.add_input('demand_signal', flow_type='default')
        self.add_output('h2_out', flow_type='hydrogen')

    def _init_properties(self):
        self.add_text_property('component_id', default='DT-1', tab='Properties')

        self.add_integer_property(
            'n_tanks', default=30, min_val=1, max_val=200,
            unit='', tab='Detailed Tank'
        )
        self.add_float_property(
            'volume_per_tank_m3', default=103.0, min_val=1.0,
            unit='m³', tab='Detailed Tank'
        )
        self.add_float_property(
            'max_pressure_bar', default=70.0, min_val=1.0,
            unit='bar', tab='Detailed Tank'
        )
        self.add_float_property(
            'initial_pressure_bar', default=1.0, min_val=0.0,
            unit='bar', tab='Detailed Tank'
        )
        self.add_float_property(
            'min_discharge_pressure_bar', default=40.0, min_val=0.0,
            unit='bar', tab='Detailed Tank'
        )
        self.add_float_property(
            'ambient_temp_k', default=298.15, min_val=200.0, max_val=350.0,
            unit='K', tab='Detailed Tank'
        )

        self.add_color_property('node_color', default=(120, 180, 120), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class DischargeStationNode(ConfigurableNode):
    """Truck discharge / filling station for hydrogen delivery."""
    __identifier__ = 'nodes.Storage'
    NODE_NAME = 'Discharge Station'

    def __init__(self):
        super(DischargeStationNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('h2_in', flow_type='hydrogen')
        self.add_output('demand_signal', flow_type='default')

    def _init_properties(self):
        self.add_text_property('component_id', default='DS-1', tab='Properties')

        self.add_integer_property(
            'n_stations', default=5, min_val=1, max_val=50,
            unit='', tab='Discharge Station'
        )
        self.add_float_property(
            'truck_capacity_kg', default=280.0, min_val=10.0,
            unit='kg', tab='Discharge Station'
        )
        self.add_float_property(
            'delivery_pressure_bar', default=500.0, min_val=1.0,
            unit='bar', tab='Discharge Station'
        )
        self.add_float_property(
            'max_fill_rate_kg_min', default=1.5, min_val=0.1,
            unit='kg/min', tab='Discharge Station'
        )

        self.add_color_property('node_color', default=(200, 160, 100), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class CompressorSingleNode(ConfigurableNode):
    """Single-stage compressor for gas pressurisation."""
    __identifier__ = 'nodes.Storage'
    NODE_NAME = 'Single Compressor'

    def __init__(self):
        super(CompressorSingleNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='gas')
        self.add_output('outlet', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='COMP-1', tab='Properties')

        self.add_float_property(
            'max_flow_kg_h', default=1000.0, min_val=1.0,
            unit='kg/h', tab='Compressor'
        )
        self.add_float_property(
            'outlet_pressure_bar', default=30.0, min_val=1.0,
            unit='bar', tab='Compressor'
        )
        self.add_float_property(
            'max_temp_c', default=180.0, min_val=50.0, max_val=500.0,
            unit='°C', tab='Compressor'
        )

        self.add_color_property('node_color', default=(200, 180, 140), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
