"""
Updated GraphToConfigAdapter with component aggregation.
"""

from typing import Dict, List, Any
from collections import defaultdict

def aggregate_components_to_systems(nodes: List) -> Dict[str, Any]:
    """
    Aggregate sanitized GUI node set into a compact system-level config.
    """
    systems = {
        "production": defaultdict(list),
        "thermal": defaultdict(list),
        "separation": defaultdict(list),
        "utilities": defaultdict(list),
        "storage": defaultdict(list),
        "reforming": defaultdict(list),
    }

    _TYPE_MAP = {
        # Production
        "PEMStackNode": ("production", "pem_stacks"),
        "SOECStackNode": ("production", "soec_stacks"),
        "RectifierNode": ("production", "rectifiers"),
        # Thermal
        "ChillerNode": ("thermal", "chillers"),
        "DryCoolerNode": ("thermal", "dry_coolers"),
        "InterchangerNode": ("thermal", "interchangers"),
        "ElectricBoilerNode": ("thermal", "electric_boilers"),
        "AttemperatorNode": ("thermal", "attemperators"),
        "CoolingManagerNode": ("thermal", "cooling_managers"),
        # Separation
        "PSAUnitNode": ("separation", "psa_units"),
        "CoalescerNode": ("separation", "coalescers"),
        "KnockOutDrumNode": ("separation", "knockout_drums"),
        "DeoxoReactorNode": ("separation", "deoxo_reactors"),
        "HydrogenMultiCycloneNode": ("separation", "cyclones"),
        "SeparationTankNode": ("separation", "separation_tanks"),
        "SyngasPSANode": ("separation", "syngas_psa_units"),
        # Utilities / Flow
        "MixerNode": ("utilities", "mixers"),
        "ValveNode": ("utilities", "valves"),
        "StreamSplitterNode": ("utilities", "stream_splitters"),
        "DrainRecorderMixerNode": ("utilities", "drain_mixers"),
        "SignalMakeupMixerNode": ("utilities", "signal_makeup_mixers"),
        "ProportionalMakeupMixerNode": ("utilities", "proportional_makeup_mixers"),
        "OxygenMakeupNode": ("utilities", "oxygen_makeups"),
        "WaterPurifierNode": ("utilities", "water_purifiers"),
        "UltraPureWaterTankNode": ("utilities", "ultrapure_tanks"),
        "ExternalWaterSourceNode": ("utilities", "external_water_sources"),
        "WaterPumpThermodynamicNode": ("utilities", "water_pumps"),
        # Storage / Delivery
        "DetailedTankNode": ("storage", "detailed_tanks"),
        "DischargeStationNode": ("storage", "discharge_stations"),
        "CompressorSingleNode": ("storage", "compressors"),
        # Reforming
        "IntegratedATRPlantNode": ("reforming", "atr_plants"),
        "ATRBoilerNode": ("reforming", "atr_boilers"),
        "BiogasSourceNode": ("reforming", "biogas_sources"),
    }

    for node in nodes:
        if hasattr(node, 'type'):
            node_type = node.type
        else:
            node_type = node.__class__.__name__

        if hasattr(node, 'properties'):
            props = dict(node.properties())
        elif hasattr(node, 'get_properties'):
            props = node.get_properties()
        else:
            props = getattr(node, 'properties', {})
            if not isinstance(props, dict):
                props = {}
        props = dict(props)

        mapping = _TYPE_MAP.get(node_type)
        if mapping:
            system_key, component_key = mapping
            systems[system_key][component_key].append(props)

    config = {
        "name": "GUI-Generated Plant",
        "version": "2.0",
        "simulation": {
            "timestep_hours": 1.0 / 60.0,
            "duration_hours": 8760,
            "checkpoint_interval_hours": 168,
        },
    }

    for system_name, system_data in systems.items():
        if any(system_data.values()):
            config[system_name] = dict(system_data)

    return config
