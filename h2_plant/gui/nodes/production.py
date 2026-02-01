"""
Production component nodes.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class ElectrolyzerNode(ConfigurableNode):
    __identifier__ = 'h2_plant.production'
    NODE_NAME = 'Electrolyzer'
    
    def _init_ports(self):
        self.add_input('power_in', flow_type='electricity', color=(255, 255, 0))
        self.add_input('water_in', flow_type='water', color=(100, 150, 255))
        
        self.add_output('h2_out', flow_type='hydrogen', color=(0, 255, 255))
        self.add_output('o2_out', flow_type='oxygen', color=(255, 200, 0))
        self.add_output('heat_out', flow_type='heat', color=(255, 100, 100))

    def _init_properties(self):
        self.add_float_input('max_power_mw', default=2.5, min_val=0.1)
        self.add_float_input('base_efficiency', default=0.65, min_val=0.1, max_val=1.0)
        self.add_float_input('min_load_factor', default=0.20, min_val=0.0, max_val=1.0)
        self.add_float_input('startup_time_hours', default=0.1, min_val=0.0)

class ATRNode(ConfigurableNode):
    __identifier__ = 'h2_plant.production'
    NODE_NAME = 'ATR Source'
    
    def _init_ports(self):
        self.add_input('ng_in', flow_type='gas', color=(200, 200, 200))
        self.add_input('power_in', flow_type='electricity', color=(255, 255, 0))
        self.add_input('water_in', flow_type='water', color=(100, 150, 255))
        
        self.add_output('h2_out', flow_type='hydrogen', color=(0, 255, 255))
        self.add_output('co2_out', flow_type='gas', color=(100, 100, 100))

    def _init_properties(self):
        self.add_float_input('max_ng_flow_kg_h', default=100.0, min_val=1.0)
        self.add_float_input('efficiency', default=0.75, min_val=0.1, max_val=1.0)
        self.add_float_input('reactor_temperature_k', default=1200.0, min_val=273.15)
        self.add_float_input('reactor_pressure_bar', default=25.0, min_val=1.0)
        self.add_float_input('startup_time_hours', default=1.0, min_val=0.0)
