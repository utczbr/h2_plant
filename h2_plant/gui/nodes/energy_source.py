"""
Energy source nodes for power input to the plant.
"""
from h2_plant.gui.nodes.base_node import ConfigurableNode


class WindEnergySourceNode(ConfigurableNode):
    """
    Wind energy source node for power input.
    Supports constant power or CSV time series input.
    """
    __identifier__ = 'nodes.Sources'
    NODE_NAME = 'Wind Energy Source'

    def __init__(self):
        super(WindEnergySourceNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        # Output only - this is a source node
        self.add_output('power_out', flow_type='electricity')

    def _init_properties(self):
        self.add_text_property('component_id', default='WIND-1', tab='Properties')

        # Production Mode Tab
        self.add_enum_property(
            'production_mode',
            options=['Constant', 'CSV Time Series'],
            default_index=0,
            tab='Production'
        )

        # Constant mode parameters
        self.add_float_property(
            'constant_power_kw', default=5000.0, min_val=0.0,
            unit='kW', tab='Production'
        )

        # CSV mode parameters
        self.add_text_property(
            'csv_file_path',
            default='producao_horaria_2_turbinas.csv',
            tab='Production'
        )

        self.add_text_property(
            'csv_format_info',
            default='Format: timestamp,power_kw (hourly)',
            tab='Production'
        )

        # Turbine Info Tab
        self.add_float_property(
            'num_turbines', default=2.0, min_val=1.0,
            unit='turbines', tab='Turbine Info'
        )
        self.add_float_property(
            'rated_power_per_turbine_kw', default=2500.0, min_val=0.0,
            unit='kW', tab='Turbine Info'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(100, 255, 100), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=80)
