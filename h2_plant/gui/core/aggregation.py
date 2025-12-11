"""
Updated GraphToConfigAdapter with component aggregation.
"""

from typing import Dict, List, Any
from collections import defaultdict

def aggregate_components_to_systems(nodes: List) -> Dict[str, Any]:
    """
    Aggregate component-level nodes into system-level configuration.
    Uses 'system' property from nodes to group components.
    """
    # Initialize system structures
    systems = {
        'pem_system': defaultdict(list),
        'soec_system': defaultdict(list),
        'atr_system': defaultdict(list),
    }
    
    # Direct mappings (non-system components)
    storage_lp = []
    storage_hp = []
    compression = {}
    demand = {}
    energy_price = {}
    battery = {}
    logistics = defaultdict(list)
    
    for node in nodes:
        # Handle different node object types (GraphNode vs NodeGraphQt Node)
        if hasattr(node, 'type'):
            node_type = node.type
        else:
            node_type = node.__class__.__name__
            
        # Handle properties access
        if hasattr(node, 'properties'):
            # NodeGraphQt node
            props = dict(node.properties())
        elif hasattr(node, 'get_properties'):
            # Custom node with get_properties method
            props = node.get_properties()
        else:
            # GraphNode object or dict
            props = getattr(node, 'properties', {})
            if not isinstance(props, dict):
                props = {}
        
        # Ensure props is a copy
        props = dict(props)
        
        # Get system assignment (if applicable)
        system_name = props.pop('system', None)
        if system_name is not None:
            # Map enum index to system name
            system_map = {0: 'pem_system', 1: 'soec_system', 2: 'atr_system'}
            system_key = system_map.get(system_name) if isinstance(system_name, int) else f"{system_name.lower()}_system"
        else:
            system_key = None
        
        # Aggregate by node type
        if node_type == 'PEMStackNode':
            systems['pem_system']['stacks'].append(props)
        elif node_type == 'SOECStackNode':
            systems['soec_system']['stacks'].append(props)
        elif node_type == 'ATRReactorNode':
            systems['atr_system']['reactors'].append(props)
        elif node_type == 'WGSReactorNode':
            systems['atr_system']['wgs_reactors'].append(props)
        
        # Context-dependent components (use system property)
        elif node_type == 'RectifierNode' and system_key:
            systems[system_key]['rectifiers'].append(props)
        elif node_type == 'HeatExchangerNode' and system_key:
            systems[system_key]['heat_exchangers'].append(props)
        elif node_type == 'SteamGeneratorNode' and system_key:
            systems[system_key]['steam_generators'].append(props)
        elif node_type == 'PSAUnitNode' and system_key:
            systems[system_key]['psa_units'].append(props)
        elif node_type == 'SeparationTankNode' and system_key:
            systems[system_key]['separation_tanks'].append(props)
        elif node_type == 'ProcessCompressorNode' and system_key:
            systems[system_key]['compressors'].append(props)
        elif node_type == 'RecirculationPumpNode' and system_key:
            systems[system_key]['pumps'].append(props)
        
        # Storage
        elif node_type == 'LPTankNode':
            if not storage_lp:
                storage_lp = props
        elif node_type == 'HPTankNode':
            if not storage_hp:
                storage_hp = props
        
        # Compression
        elif node_type == 'FillingCompressorNode':
            compression['filling_compressor'] = props
        elif node_type == 'OutgoingCompressorNode':
            compression['outgoing_compressor'] = props
        
        # Logic & Utilities
        elif node_type == 'DemandSchedulerNode':
            demand.update(props)
        elif node_type == 'EnergyPriceNode':
            energy_price.update(props)
        elif node_type == 'BatteryNode':
            battery.update(props)
            battery['enabled'] = True
        elif node_type == 'ConsumerNode':
            logistics['consumers'].append(props)
    
    # Build final config
    config = {
        'name': 'GUI-Generated Plant',
        'version': '2.0',
        'simulation': {
            'timestep_hours': 1.0/60.0,
            'duration_hours': 8760,
            'checkpoint_interval_hours': 168
        },
        'pathway': {
            'allocation_strategy': 'COST_OPTIMAL'
        }
    }
    
    # Add non-empty systems
    for system_name, system_data in systems.items():
        if any(system_data.values()):
            config[system_name] = dict(system_data)
    
    # Add storage
    if storage_lp or storage_hp:
        config['storage'] = {
            'source_isolated': False
        }
        if storage_lp:
            config['storage']['lp_tanks'] = storage_lp
        if storage_hp:
            config['storage']['hp_tanks'] = storage_hp
    
    # Add other systems
    if compression:
        config['compression'] = compression
    if demand:
        config['demand'] = demand
    if energy_price:
        config['energy_price'] = energy_price
    if battery:
        config['battery'] = battery
    if logistics['consumers']:
        config['logistics'] = dict(logistics)
    
    return config
