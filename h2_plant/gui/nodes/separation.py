
"""
Separation component nodes.
Covers: PSAUnit, Coalescer, KnockOutDrum, DeoxoReactor,
        HydrogenMultiCyclone, SeparationTank, SyngasPSA.
"""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class PSAUnitNode(ConfigurableNode):
    __identifier__ = 'nodes.Separation'
    NODE_NAME = 'PSA Unit'

    def __init__(self):
        super(PSAUnitNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('gas_in', flow_type='gas')
        self.add_output('purified_gas_out', flow_type='gas')
        self.add_output('tail_gas_out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='D-1', tab='Properties')

        # PSA Unit Tab
        self.add_percentage_property('efficiency', default=85.0, tab='PSA Unit')
        self.add_percentage_property('recovery_rate', default=90.0, tab='PSA Unit')
        self.add_float_property(
            'operating_pressure_bar', default=30.0, min_val=1.0, unit='bar', tab='PSA Unit'
        )
        self.add_enum_property(
            'gas_type', options=['H2', 'O2'], default_index=0, tab='PSA Unit'
        )

        # Custom Tab
        self.add_color_property('node_color', default=(200, 200, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class CoalescerNode(ConfigurableNode):
    """Coalescer node for aerosol/liquid removal from gas streams."""
    __identifier__ = 'nodes.Separation'
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
            'gas_type', options=['H2', 'O2', 'Syngas'], default_index=0, tab='Coalescer'
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
    __identifier__ = 'nodes.Separation'
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
            'gas_species', options=['H2', 'O2', 'Syngas'], default_index=0, tab='KOD'
        )

        self.add_color_property('node_color', default=(100, 150, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class DeoxoReactorNode(ConfigurableNode):
    """Catalytic Deoxidizer for removing O2 from H2 streams."""
    __identifier__ = 'nodes.Separation'
    NODE_NAME = 'Deoxo Reactor'

    def __init__(self):
        super(DeoxoReactorNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='gas')
        self.add_output('outlet', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='Deoxo-1', tab='Properties')
        self.add_color_property('node_color', default=(255, 100, 150), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class HydrogenMultiCycloneNode(ConfigurableNode):
    """Multi-cyclone separator for water droplet removal from gas streams."""
    __identifier__ = 'nodes.Separation'
    NODE_NAME = 'Hydrogen Multi-Cyclone'

    def __init__(self):
        super(HydrogenMultiCycloneNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='gas')
        self.add_output('outlet', flow_type='gas')
        self.add_output('drain', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='CYC-1', tab='Properties')

        self.add_float_property(
            'element_diameter_mm', default=50.0, min_val=10.0, max_val=200.0,
            unit='mm', tab='Multi-Cyclone'
        )
        self.add_float_property(
            'vane_angle_deg', default=45.0, min_val=15.0, max_val=75.0,
            unit='°', tab='Multi-Cyclone'
        )
        self.add_float_property(
            'target_velocity_ms', default=20.0, min_val=5.0, max_val=40.0,
            unit='m/s', tab='Multi-Cyclone'
        )
        self.add_enum_property(
            'gas_species', options=['H2', 'O2', 'Syngas'], default_index=0,
            tab='Multi-Cyclone'
        )

        self.add_color_property('node_color', default=(170, 200, 170), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class SeparationTankNode(ConfigurableNode):
    """Gravity separation tank (degasser) for liquid/gas separation."""
    __identifier__ = 'nodes.Separation'
    NODE_NAME = 'Separation Tank'

    def __init__(self):
        super(SeparationTankNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('mixture_in', flow_type='water')
        self.add_output('liquid_out', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='ST-1', tab='Properties')

        self.add_float_property(
            'volume_m3', default=2.0, min_val=0.1, max_val=50.0,
            unit='m³', tab='Separation Tank'
        )
        self.add_percentage_property(
            'efficiency', default=100.0, tab='Separation Tank'
        )

        self.add_color_property('node_color', default=(150, 180, 220), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class SyngasPSANode(ConfigurableNode):
    """Pressure Swing Adsorption unit for syngas H2 purification."""
    __identifier__ = 'nodes.Separation'
    NODE_NAME = 'Syngas PSA'

    def __init__(self):
        super(SyngasPSANode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('gas_in', flow_type='gas')
        self.add_output('purified_gas_out', flow_type='gas')

    def _init_properties(self):
        self.add_text_property('component_id', default='SPSA-1', tab='Properties')

        self.add_float_property(
            'cycle_time_min', default=10.0, min_val=1.0, max_val=60.0,
            unit='min', tab='Syngas PSA'
        )
        self.add_percentage_property(
            'recovery_rate', default=90.0, tab='Syngas PSA'
        )
        self.add_float_property(
            'purity_target', default=0.999, min_val=0.9, max_val=1.0,
            unit='', tab='Syngas PSA'
        )
        self.add_float_property(
            'power_consumption_kw', default=25.0, min_val=0.0,
            unit='kW', tab='Syngas PSA'
        )
        self.add_float_property(
            'bed_length_m', default=2.5, min_val=0.5, max_val=10.0,
            unit='m', tab='Syngas PSA'
        )
        self.add_float_property(
            'bed_diameter_m', default=1.0, min_val=0.1, max_val=5.0,
            unit='m', tab='Syngas PSA'
        )

        self.add_color_property('node_color', default=(220, 200, 180), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
