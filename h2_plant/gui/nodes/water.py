"""
Water-related GUI nodes for the hydrogen plant.
Covers: WaterPurifier, UltraPureWaterTank, ExternalWaterSource, WaterPumpThermodynamic.
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
        self.add_input('raw_water_in', flow_type='water')
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
        self.add_input('ultrapure_in', flow_type='water')
        
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


class ExternalWaterSourceNode(ConfigurableNode):
    """External water source providing raw water to the plant."""
    __identifier__ = 'h2_plant.water'
    NODE_NAME = 'External Water Source'

    def __init__(self):
        super(ExternalWaterSourceNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('control_signal', flow_type='default')
        self.add_output('water_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='EWS-1', tab='Properties')

        self.add_float_property(
            'flow_rate_kg_h', default=12000.0, min_val=0.0,
            unit='kg/h', tab='External Water Source'
        )
        self.add_float_property(
            'pressure_bar', default=5.0, min_val=0.1,
            unit='bar', tab='External Water Source'
        )
        self.add_float_property(
            'temperature_c', default=25.0, min_val=0.0, max_val=80.0,
            unit='°C', tab='External Water Source'
        )
        self.add_enum_property(
            'mode', options=['external_control', 'constant_flow'],
            default_index=0, tab='External Water Source'
        )

        self.add_color_property('node_color', default=(80, 180, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class WaterPumpThermodynamicNode(ConfigurableNode):
    """Thermodynamic water pump for pressurised water delivery."""
    __identifier__ = 'h2_plant.water'
    NODE_NAME = 'Water Pump'

    def __init__(self):
        super(WaterPumpThermodynamicNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('water_in', flow_type='water')
        self.add_output('water_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='WPP-1', tab='Properties')

        self.add_float_property(
            'capacity_kg_h', default=2000.0, min_val=1.0,
            unit='kg/h', tab='Water Pump'
        )
        self.add_float_property(
            'target_pressure_pa', default=500000.0, min_val=100000.0,
            unit='Pa', tab='Water Pump'
        )

        self.add_color_property('node_color', default=(100, 160, 255), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)

