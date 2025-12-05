"""
Fluid handling component nodes (Compressor, Pump).
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class ProcessCompressorNode(ConfigurableNode):
    __identifier__ = 'h2_plant.fluid'
    NODE_NAME = 'Process Compressor'
    
    def _init_ports(self):
        self.add_input('gas_in', flow_type='gas', color=(200, 200, 200))
        self.add_output('gas_out', flow_type='gas', color=(255, 200, 200))

    def _init_properties(self):
        self.add_text_input('component_id', default='C-1')
        self.add_enum_input('system', ['PEM', 'SOEC', 'ATR'], 2)  # Default to ATR
        self.add_float_input('max_flow_kg_h', default=500.0, min_val=1.0)
        self.add_float_input('pressure_ratio', default=2.0, min_val=1.0)

class RecirculationPumpNode(ConfigurableNode):
    __identifier__ = 'h2_plant.fluid'
    NODE_NAME = 'Recirculation Pump'
    
    def _init_ports(self):
        self.add_input('fluid_in', flow_type='water', color=(100, 150, 255))
        self.add_output('fluid_out', flow_type='water', color=(150, 200, 255))

    def _init_properties(self):
        self.add_text_input('component_id', default='P-1')
        self.add_enum_input('system', ['PEM', 'SOEC', 'ATR'], 0)  # Default to PEM
        self.add_float_input('max_flow_kg_h', default=1000.0, min_val=1.0)
        self.add_float_input('pressure_bar', default=5.0, min_val=1.0)
