"""
Economics and arbitrage nodes for plant optimization.
"""
from h2_plant.gui.nodes.base_node import ConfigurableNode


class ArbitrageNode(ConfigurableNode):
    """
    Arbitrage decision node for economic dispatch optimization.
    Determines when to produce H2 vs sell electricity based on prices.
    """
    __identifier__ = 'h2_plant.economics.arbitrage'
    NODE_NAME = 'Arbitrage'

    def __init__(self):
        super(ArbitrageNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        # Inputs: receives energy availability
        self.add_input('power_available', flow_type='electricity')
        
        # Outputs: dispatch to components
        self.add_output('primary_power', flow_type='electricity')
        self.add_output('secondary_power', flow_type='electricity')

    def _init_properties(self):
        self.add_text_property('component_id', default='ARB-1', tab='Properties')

        # Price Parameters Tab (maps to economics_parameters.yaml)
        self.add_float_property(
            'electricity_buy_price', default=0.08, min_val=0.0,
            unit='€/kWh', tab='Price Parameters'
        )
        self.add_float_property(
            'electricity_sell_price', default=0.05, min_val=0.0,
            unit='€/kWh', tab='Price Parameters'
        )
        self.add_float_property(
            'h2_sell_price', default=5.0, min_val=0.0,
            unit='€/kg', tab='Price Parameters'
        )
        self.add_float_property(
            'h2_production_cost', default=3.5, min_val=0.0,
            unit='€/kg', tab='Price Parameters'
        )

        # Arbitrage Strategy Tab
        self.add_enum_property(
            'strategy_mode',
            options=['Simple Threshold', 'Price Forecast', 'Storage Optimization'],
            default_index=0,
            tab='Strategy'
        )
        self.add_float_property(
            'min_storage_threshold', default=20.0, min_val=0.0, max_val=100.0,
            unit='%', tab='Strategy'
        )
        self.add_float_property(
            'sell_trigger_price', default=6.0, min_val=0.0,
            unit='€/kg', tab='Strategy'
        )
        self.add_float_property(
            'buy_trigger_price', default=0.05, min_val=0.0,
            unit='€/kWh', tab='Strategy'
        )

        # Time Parameters Tab
        self.add_float_property(
            'forecast_horizon_h', default=24.0, min_val=1.0,
            unit='hours', tab='Time Parameters'
        )
        self.add_float_property(
            'decision_interval_min', default=15.0, min_val=1.0,
            unit='minutes', tab='Time Parameters'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(255, 215, 0), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=100)
