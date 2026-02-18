"""
Reforming and biogas source nodes.
Covers: IntegratedATRPlant, ATR_Boiler, BiogasSource.
"""
from h2_plant.gui.nodes.base_node import ConfigurableNode


class IntegratedATRPlantNode(ConfigurableNode):
    """Integrated autothermal reformer plant."""
    __identifier__ = 'nodes.Reforming'
    NODE_NAME = 'Integrated ATR Plant'

    def __init__(self):
        super(IntegratedATRPlantNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='gas')
        self.add_output('syngas_out', flow_type='gas')
        self.add_output('o2_signal', flow_type='default')

    def _init_properties(self):
        self.add_text_property('component_id', default='ATR-1', tab='Properties')

        self.add_float_property(
            'max_flow_kg_h', default=20000.0, min_val=1.0,
            unit='kg/h', tab='ATR Plant'
        )
        self.add_float_property(
            'pressure_drop_bar', default=2.5, min_val=0.0, max_val=10.0,
            unit='bar', tab='ATR Plant'
        )

        self.add_color_property('node_color', default=(220, 180, 120), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class ATRBoilerNode(ConfigurableNode):
    """ATR waste-heat boiler (heat recovery from syngas)."""
    __identifier__ = 'nodes.Reforming'
    NODE_NAME = 'ATR Boiler'

    def __init__(self):
        super(ATRBoilerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='gas')
        self.add_input('o2_signal_in', flow_type='default')
        self.add_output('outlet', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='ATRB-1', tab='Properties')

        self.add_color_property('node_color', default=(220, 160, 100), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class BiogasSourceNode(ConfigurableNode):
    """Biogas feedstock source for the ATR subsystem."""
    __identifier__ = 'nodes.Reforming'
    NODE_NAME = 'Biogas Source'

    def __init__(self):
        super(BiogasSourceNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_output('out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='BGS-1', tab='Properties')

        self.add_float_property(
            'pressure_bar', default=3.0, min_val=0.1,
            unit='bar', tab='Biogas Source'
        )
        self.add_float_property(
            'temperature_c', default=25.0, min_val=0.0, max_val=100.0,
            unit='°C', tab='Biogas Source'
        )
        self.add_float_property(
            'max_flow_rate_kg_h', default=1960.0, min_val=0.0,
            unit='kg/h', tab='Biogas Source'
        )
        self.add_float_property(
            'min_flow_rate_kg_h', default=594.0, min_val=0.0,
            unit='kg/h', tab='Biogas Source'
        )
        self.add_percentage_property(
            'methane_content', default=60.0, tab='Biogas Source'
        )

        self.add_color_property('node_color', default=(180, 220, 120), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
