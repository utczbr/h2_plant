"""
Utility component nodes (Water, Battery, External Inputs).
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class WaterTreatmentNode(ConfigurableNode):
    __identifier__ = 'h2_plant.utilities'
    NODE_NAME = 'Water Treatment'
    
    def _init_ports(self):
        self.add_input('water_in', flow_type='water', color=(100, 150, 255))
        self.add_output('water_out', flow_type='water', color=(100, 150, 255))

    def _init_properties(self):
        # Note: The config structure for water treatment is nested
        # We might need to flatten it or handle nested properties in the adapter
        # For now, exposing key top-level params
        self.add_float_input('max_flow_m3h', default=10.0, min_val=0.1)
        self.add_float_input('power_consumption_kw', default=20.0, min_val=0.0)

class BatteryNode(ConfigurableNode):
    __identifier__ = 'h2_plant.utilities'
    NODE_NAME = 'Battery Storage'
    
    def _init_ports(self):
        self.add_input('power_in', flow_type='electricity', color=(255, 255, 0))
        self.add_output('power_out', flow_type='electricity', color=(255, 255, 0))

    def _init_properties(self):
        self.add_float_input('capacity_kwh', default=1000.0, min_val=1.0)
        self.add_float_input('max_charge_power_kw', default=500.0, min_val=1.0)
        self.add_float_input('max_discharge_power_kw', default=500.0, min_val=1.0)
        self.add_float_input('min_soc', default=0.20, min_val=0.0, max_val=1.0)
        self.add_float_input('max_soc', default=0.95, min_val=0.0, max_val=1.0)

class OxygenSourceNode(ConfigurableNode):
    __identifier__ = 'h2_plant.utilities'
    NODE_NAME = 'External Oxygen'
    
    def _init_ports(self):
        self.add_output('o2_out', flow_type='oxygen', color=(255, 200, 0))

    def _init_properties(self):
        self.add_float_input('flow_rate_kg_h', default=0.0, min_val=0.0)
        self.add_float_input('pressure_bar', default=5.0, min_val=1.0)
        self.add_float_input('cost_per_kg', default=0.15, min_val=0.0)

class HeatSourceNode(ConfigurableNode):
    __identifier__ = 'h2_plant.utilities'
    NODE_NAME = 'External Heat'
    
    def _init_ports(self):
        self.add_output('heat_out', flow_type='heat', color=(255, 100, 100))

    def _init_properties(self):
        self.add_float_input('thermal_power_kw', default=500.0, min_val=0.0)
        self.add_float_input('temperature_c', default=150.0, min_val=0.0)
        self.add_float_input('cost_per_kwh', default=0.0, min_val=0.0)

class MixerNode(ConfigurableNode):
    __identifier__ = 'h2_plant.utilities'
    NODE_NAME = 'Gas Mixer'
    
    def _init_ports(self):
        self.add_input('in_1', flow_type='gas', color=(200, 200, 200))
        self.add_input('in_2', flow_type='gas', color=(200, 200, 200))
        self.add_output('out', flow_type='gas', color=(200, 200, 200))

    def _init_properties(self):
        self.add_float_input('capacity_kg', default=1000.0, min_val=1.0)
        self.add_float_input('target_pressure_bar', default=5.0, min_val=1.0)
