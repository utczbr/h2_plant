import sys
import os

# Mock classes to avoid full instantiation
class MockStream:
    def __init__(self):
        self.mass_flow_kg_h = 1000.0
        self.temperature_k = 300.0
        self.pressure_pa = 100000.0
        self.composition = {'H2O': 1.0}
        self.phase = 'liquid'

class MockComponent:
    def __init__(self, output_port='outlet'):
        self.output_port = output_port
        self.output_stream = MockStream()
    
    def get_output(self, port_name):
        if port_name == self.output_port:
            return self.output_stream
        return None

# Add project root to path
sys.path.append(os.getcwd())

from h2_plant.reporting.stream_table import print_stream_summary_table

# Setup test components
components = {
    "ATR_H2O_Boiler": MockComponent(output_port='fluid_out'),
    "ATR_H2O_Compressor_1": MockComponent(output_port='outlet'),
    "ATR_H2O_DryCooler": MockComponent(output_port='fluid_out'),
    "SOEC_Water_Splitter": MockComponent(output_port='outlet_1'), # Should find outlet_1
    "ATR_Syngas_Cooler": MockComponent(output_port='syngas_out') # Should go to Sec 8
}

topo_order = [
    "SOEC_Water_Splitter",
    "ATR_H2O_Boiler",
    "ATR_H2O_Compressor_1",
    "ATR_H2O_DryCooler",
    "ATR_Syngas_Cooler"
]

print("Running Stream Table Section Verification...")
print_stream_summary_table(components, topo_order)
