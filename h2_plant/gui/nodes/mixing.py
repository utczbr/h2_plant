"""
Mixing and flow control nodes.
Covers: Mixer, StreamSplitter, DrainRecorderMixer,
        SignalMakeupMixer, ProportionalMakeupMixer, OxygenMakeupNode.
"""
from h2_plant.gui.nodes.base_node import ConfigurableNode


class MixerNode(ConfigurableNode):
    """
    Gas/Fluid Mixer node.
    """
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Mixer (Generic)'

    def __init__(self):
        super(MixerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        # Multiple inputs converge to single output
        self.add_input('inlet_1', flow_type='default', multi_input=True)
        self.add_input('inlet_2', flow_type='default', multi_input=True)
        self.add_output('outlet', flow_type='default')

    def _init_properties(self):
        self.add_text_property('component_id', default='MIX-1', tab='Properties')

        # Mixer Tab
        self.add_float_property(
            'volume_m3', default=1.0, min_val=0.01,
            unit='m³', tab='Mixer'
        )
        self.add_enum_property(
            'fluid_type',
            options=['Water', 'Hydrogen', 'Oxygen', 'Natural Gas', 'Generic'],
            default_index=0,
            tab='Mixer'
        )
        self.add_float_property(
            'target_pressure_bar', default=5.0, min_val=0.0,
            unit='bar', tab='Mixer'
        )
        
        # Custom Tab
        self.add_color_property('node_color', default=(180, 180, 180), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class StreamSplitterNode(ConfigurableNode):
    """Stream splitter that divides a single input into two outputs by ratio."""
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Stream Splitter'

    def __init__(self):
        super(StreamSplitterNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='default')
        self.add_output('outlet_1', flow_type='default')
        self.add_output('outlet_2', flow_type='default')

    def _init_properties(self):
        self.add_text_property('component_id', default='SPL-1', tab='Properties')

        self.add_float_property(
            'split_ratio', default=0.5, min_val=0.0, max_val=1.0,
            unit='', tab='Stream Splitter'
        )

        self.add_color_property('node_color', default=(180, 200, 180), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class DrainRecorderMixerNode(ConfigurableNode):
    """Drain recorder mixer — collects multiple drain streams into one outlet."""
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Drain Recorder Mixer'

    def __init__(self):
        super(DrainRecorderMixerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet_1', flow_type='water', multi_input=True)
        self.add_input('inlet_2', flow_type='water', multi_input=True)
        self.add_input('inlet_3', flow_type='water', multi_input=True)
        self.add_input('inlet_4', flow_type='water', multi_input=True)
        self.add_output('outlet', flow_type='water')

    def _init_properties(self):
        self.add_text_property('component_id', default='DRM-1', tab='Properties')

        self.add_text_property(
            'source_ids', default='', tab='Drain Recorder Mixer'
        )

        self.add_color_property('node_color', default=(160, 180, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class SignalMakeupMixerNode(ConfigurableNode):
    """Makeup mixer with demand signal output for water supply control."""
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Signal Makeup Mixer'

    def __init__(self):
        super(SignalMakeupMixerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('makeup_water_in', flow_type='water')
        self.add_input('drain_in', flow_type='water')
        self.add_output('mixture_out', flow_type='water')
        self.add_output('demand_signal', flow_type='default')

    def _init_properties(self):
        self.add_text_property('component_id', default='SMM-1', tab='Properties')

        self.add_float_property(
            'target_flow_kg_h', default=3600.0, min_val=0.0,
            unit='kg/h', tab='Signal Makeup Mixer'
        )
        self.add_float_property(
            'makeup_temp_c', default=25.0, min_val=0.0, max_val=100.0,
            unit='°C', tab='Signal Makeup Mixer'
        )

        self.add_color_property('node_color', default=(140, 200, 180), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class ProportionalMakeupMixerNode(ConfigurableNode):
    """Proportional makeup mixer with flow scaling from a reference component."""
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Proportional Makeup Mixer'

    def __init__(self):
        super(ProportionalMakeupMixerNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('makeup_water_in', flow_type='water')
        self.add_input('drain_in', flow_type='water')
        self.add_output('water_out', flow_type='water')
        self.add_output('demand_signal', flow_type='default')

    def _init_properties(self):
        self.add_text_property('component_id', default='PMM-1', tab='Properties')

        self.add_float_property(
            'max_flow_rate_kg_h', default=2500.0, min_val=0.0,
            unit='kg/h', tab='Proportional Makeup Mixer'
        )
        self.add_float_property(
            'min_flow_rate_kg_h', default=700.0, min_val=0.0,
            unit='kg/h', tab='Proportional Makeup Mixer'
        )
        self.add_float_property(
            'makeup_pressure_bar', default=15.0, min_val=1.0,
            unit='bar', tab='Proportional Makeup Mixer'
        )
        self.add_float_property(
            'makeup_temp_c', default=25.0, min_val=0.0, max_val=100.0,
            unit='°C', tab='Proportional Makeup Mixer'
        )

        self.add_color_property('node_color', default=(140, 180, 200), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)


class OxygenMakeupNode(ConfigurableNode):
    """Oxygen backup/makeup supply node for ATR oxygen requirements."""
    __identifier__ = 'nodes.Flow'
    NODE_NAME = 'Oxygen Makeup'

    def __init__(self):
        super(OxygenMakeupNode, self).__init__()
        self.enable_collapse()

    def _init_ports(self):
        self.add_input('inlet', flow_type='oxygen')
        self.add_output('outlet', flow_type='oxygen')

    def _init_properties(self):
        self.add_text_property('component_id', default='O2M-1', tab='Properties')

        self.add_float_property(
            'min_target_flow_kg_h', default=230.0, min_val=0.0,
            unit='kg/h', tab='Oxygen Makeup'
        )
        self.add_float_property(
            'max_limit_flow_kg_h', default=760.0, min_val=0.0,
            unit='kg/h', tab='Oxygen Makeup'
        )
        self.add_float_property(
            'supply_pressure_bar', default=1.05, min_val=0.1,
            unit='bar', tab='Oxygen Makeup'
        )
        self.add_float_property(
            'supply_temperature_c', default=25.0, min_val=0.0, max_val=100.0,
            unit='°C', tab='Oxygen Makeup'
        )

        self.add_color_property('node_color', default=(255, 200, 100), tab='Custom')
        self.add_text_property('custom_label', default='', tab='Custom')
        self.add_spacer('collapse_spacer', height=60)
