
"""
Separation component nodes (PSA, Separation Tank).
Refactored to follow the new collapsible pattern.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class PSAUnitNode(ConfigurableNode):
    __identifier__ = 'h2_plant.separation.psa'
    NODE_NAME = 'PSA Unit'

    def __init__(self):
        super(PSAUnitNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('feed_in', flow_type='gas')
        self.add_output('product_out', flow_type='gas')
        self.add_output('waste_out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='D-1', tab='Properties')

        # PSA Unit Tab
        self.add_percentage_property('efficiency', default=85.0, tab='PSA Unit')
        self.add_percentage_property('recovery_rate', default=90.0, tab='PSA Unit')
        self.add_float_property(
            'operating_pressure_bar', default=30.0, min_val=1.0, unit='bar', tab='PSA Unit'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(200, 200, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)

class SeparationTankNode(ConfigurableNode):
    __identifier__ = 'h2_plant.separation.tank'
    NODE_NAME = 'Separation Tank'

    def __init__(self):
        super(SeparationTankNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('mixed_in', flow_type='gas')
        self.add_output('gas_out', flow_type='gas')
        self.add_output('liquid_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='ST-1', tab='Properties')

        # Separation Tank Tab
        self.add_float_property(
            'volume_m3', default=5.0, min_val=0.1, unit='m³', tab='Separation Tank'
        )
        self.add_float_property(
            'operating_pressure_bar', default=30.0, min_val=1.0, unit='bar', tab='Separation Tank'
        )

        self.add_color_property('node_color', default=(150, 150, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class CoalescerNode(ConfigurableNode):
    """Coalescer node for aerosol/liquid removal from gas streams."""
    __identifier__ = 'h2_plant.separation.coalescer'
    NODE_NAME = 'Coalescer'

    def __init__(self):
        super(CoalescerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='gas')
        self.add_output('outlet', flow_type='gas')
        self.add_output('drain', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='C-1', tab='Properties')

        # Coalescer Tab
        self.add_float_property(
            'd_shell_m', default=0.32, min_val=0.05, max_val=2.0, unit='m', tab='Coalescer'
        )
        self.add_float_property(
            'l_elem_m', default=1.0, min_val=0.1, max_val=5.0, unit='m', tab='Coalescer'
        )
        self.add_enum_property(
            'gas_type', options=['H2', 'O2'], default_index=0, tab='Coalescer'
        )

        self.add_color_property('node_color', default=(150, 200, 150), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class KnockOutDrumNode(ConfigurableNode):
    """
    Knock-Out Drum node for liquid water removal from gas streams.
    
    A vertical separator vessel that removes liquid water droplets
    from H2 or O2 streams using gravity separation.
    """
    __identifier__ = 'h2_plant.separation.knock_out_drum'
    NODE_NAME = 'Knock-Out Drum'

    def __init__(self):
        super(KnockOutDrumNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('gas_inlet', flow_type='gas')
        self.add_output('gas_outlet', flow_type='gas')
        self.add_output('liquid_drain', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='KOD-1', tab='Properties')

        # Knock-Out Drum Tab
        self.add_float_property(
            'diameter_m', default=1.0, min_val=0.1, max_val=5.0, unit='m', tab='KOD'
        )
        self.add_float_property(
            'delta_p_bar', default=0.05, min_val=0.0, max_val=1.0, unit='bar', tab='KOD'
        )
        self.add_enum_property(
            'gas_species', options=['H2', 'O2'], default_index=0, tab='KOD'
        )

        self.add_color_property('node_color', default=(100, 150, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)

