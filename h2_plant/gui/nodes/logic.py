"""
Logic and control nodes (Demand, Price, etc.).
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class DemandSchedulerNode(ConfigurableNode):
    __identifier__ = 'nodes.Logic'
    NODE_NAME = 'Demand Scheduler'
    
    def _init_ports(self):
        self.add_output('demand_signal', flow_type='default', color=(255, 255, 255))

    def _init_properties(self):
        self.add_enum_input('pattern', ['constant', 'day_night', 'weekly', 'custom'], 0)
        self.add_float_input('base_demand_kg_h', default=50.0, min_val=0.0)
        
        # Day/Night params
        self.add_float_input('day_demand_kg_h', default=60.0, min_val=0.0)
        self.add_float_input('night_demand_kg_h', default=40.0, min_val=0.0)
        self.add_integer_input('day_start_hour', default=6, min_val=0, max_val=23)
        self.add_integer_input('night_start_hour', default=22, min_val=0, max_val=23)

class EnergyPriceNode(ConfigurableNode):
    __identifier__ = 'nodes.Logic'
    NODE_NAME = 'Energy Price'
    
    def _init_ports(self):
        self.add_output('price_signal', flow_type='electricity', color=(255, 255, 0))

    def _init_properties(self):
        self.add_enum_input('source', ['constant', 'file', 'api'], 0)
        self.add_float_input('constant_price_per_mwh', default=60.0, min_val=0.0)
