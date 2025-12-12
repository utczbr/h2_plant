from h2_plant.gui.nodes.base_node import ConfigurableNode

class ValveNode(ConfigurableNode):
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Throttling Valve'

    def __init__(self):
        super(ValveNode, self).__init__()
        self.add_output('outlet')
        self.add_input('inlet')
        self.enable_collapse()

    def _init_properties(self):
        # Identification
        self.add_text_property('component_id', default='Valve-1', tab='Properties')
        
        # Configuration
        self.add_float_property(
            'outlet_pressure_bar', 
            default=5.0, 
            min_val=0.1, 
            unit='bar', 
            tab='Control'
        )
        
        self.add_enum_property(
            'fluid_type',
            options=['H2', 'N2', 'O2', 'CO2'],
            default_index=0,
            tab='Control'
        )
        
        # Visuals
        self.add_color_property('node_color', default=(150, 150, 150), tab='Custom')
