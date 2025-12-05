"""
Water-related GUI nodes for the hydrogen plant.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class WaterPurifierNode(ConfigurableNode):
    __identifier__ = 'h2_plant.water'
    NODE_NAME = 'Water Purifier'
    
    def __init__(self):
        super(WaterPurifierNode, self).__init__()
        self.enable_collapse()
    
    def _init_ports(self):
        # Inputs
        self.add_input('water_in', flow_type='water')
        self.add_input('electricity_in', flow_type='electricity')
        
        # Outputs
        self.add_output('ultrapure_out', flow_type='water')
    
    def _init_properties(self):
        # Properties Tab - Component identification
        self.add_text_property('component_id', default='WP-1', tab='Properties')
        
        # Water Purifier Tab - Operational parameters
        self.add_float_property(
            'power_cost_kwh',
            default=0.005,
            min_val=0.0,
            max_val=1.0,
            unit='kW/h',
            tab='Water Purifier'
        )
        
        self.add_float_property(
            'output_flow_kgh',
            default=1000.0,
            min_val=1.0,
            max_val=10000.0,
            unit='kg/h',
            tab='Water Purifier'
        )
        
        self.add_float_property(
            'output_temperature_c',
            default=25.0,
            min_val=0.0,
            max_val=80.0,
            unit='°C',
            tab='Water Purifier'
        )
        
        self.add_float_property(
            'output_pressure_bar',
            default=5.0,
            min_val=1.0,
            max_val=20.0,
            unit='bar',
            tab='Water Purifier'
        )
        
        self.add_percentage_property(
            'purification_efficiency',
            default=99.5,
            min_val=90.0,
            max_val=100.0,
            tab='Water Purifier'
        )
        
        self.add_float_property(
            'maintenance_interval_hours',
            default=720.0,
            min_val=1.0,
            max_val=8760.0,
            unit='hours',
            tab='Water Purifier'
        )
        
        # Custom Tab - Visual customization
        self.add_color_property('node_color', default=(100, 200, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')


class UltraPureWaterTankNode(ConfigurableNode):
    __identifier__ = 'h2_plant.water_tank'
    NODE_NAME = 'Ultrapure Water Tank'
    
    def __init__(self):
        super(UltraPureWaterTankNode, self).__init__()
        self.enable_collapse()
    
    def _init_ports(self):
        # Inputs
        self.add_input('water_in', flow_type='water')
        
        # Outputs (multiple distribution points)
        self.add_output('water_out_pem', flow_type='water')
        self.add_output('water_out_soec', flow_type='water')
        self.add_output('water_out_atr', flow_type='water')
    
    def _init_properties(self):
        # Properties Tab - Component identification
        self.add_text_property('component_id', default='UPT-1', tab='Properties')
        
        # Ultrapure Water Tank Tab - Storage parameters
        self.add_float_property(
            'capacity_m3',
            default=50.0,
            min_val=1.0,
            max_val=500.0,
            unit='m³',
            tab='Ultrapure Water Tank'
        )
        
        self.add_float_property(
            'water_temperature_c',
            default=20.0,
            min_val=0.0,
            max_val=80.0,
            unit='°C',
            tab='Ultrapure Water Tank'
        )
        
        self.add_float_property(
            'tank_pressure_bar',
            default=1.0,
            min_val=0.5,
            max_val=10.0,
            unit='bar',
            tab='Ultrapure Water Tank'
        )
        
        self.add_percentage_property(
            'min_safe_level',
            default=10.0,
            min_val=0.0,
            max_val=50.0,
            tab='Ultrapure Water Tank'
        )
        
        self.add_percentage_property(
            'max_safe_level',
            default=95.0,
            min_val=50.0,
            max_val=100.0,
            tab='Ultrapure Water Tank'
        )
        
        # Custom Tab - Visual customization
        self.add_color_property('node_color', default=(100, 200, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        
        # Spacer for collapsed mode to avoid arrow overlapping ports
        self.add_spacer('collapse_spacer', height=75)

