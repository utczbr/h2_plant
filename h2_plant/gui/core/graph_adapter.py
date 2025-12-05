"""
Graph-to-Config adapter: Converts visual node graph to PlantConfig dictionary.

This is the bridge between GUI (visual) and backend (configuration).
All validation happens here before PlantBuilder is called.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class FlowType(str, Enum):
    """Port connection types (prevent invalid connections)."""
    HYDROGEN = "hydrogen"
    OXYGEN = "oxygen"
    ELECTRICITY = "electricity"
    HEAT = "heat"
    WATER = "water"
    COMPRESSED_H2 = "compressed_h2"
    GAS = "gas" # Added for ATR

@dataclass
class Port:
    """Represents an input or output port on a node."""
    name: str  # e.g., "h2_output", "power_input"
    flow_type: FlowType
    direction: str  # "input" or "output"
    description: str = ""
    unit: str = ""

@dataclass
class GraphNode:
    """Represents a visual node in the editor."""
    id: str  # Unique identifier (UUID or auto-generated)
    type: str  # e.g., "ElectrolyzerNode", "TankNode"
    display_name: str  # User-visible name
    x: float  # Canvas position (not used for config, but saved for UX)
    y: float
    properties: Dict[str, Any]  # {"max_power_mw": 5.0, "efficiency": 0.68, ...}
    ports: List[Port]  # Input and output ports

@dataclass
class GraphEdge:
    """Represents a connection between two nodes."""
    source_node_id: str
    source_port: str
    target_node_id: str
    target_port: str
    flow_type: FlowType
    
    def validate(self, nodes: Dict[str, GraphNode]) -> None:
        """Ensure both endpoints exist and flow types match."""
        if self.source_node_id not in nodes:
            raise ValueError(f"Source node {self.source_node_id} not found")
        if self.target_node_id not in nodes:
            raise ValueError(f"Target node {self.target_node_id} not found")
        
        src_node = nodes[self.source_node_id]
        tgt_node = nodes[self.target_node_id]
        
        # In a real implementation, we'd check if ports exist on the node definition
        # For now, we assume the edge creation logic in GUI ensures ports exist
        pass

class GraphToConfigAdapter:
    """Main conversion engine."""
    
    # Mapping from visual node types to config sections
    NODE_TYPE_MAPPING = {
        # Production
        "ElectrolyzerNode": ("production", "electrolyzer"),
        "PEMStackNode": ("production", "electrolyzer"),  # Map PEM to electrolyzer config
        "SOECStackNode": ("production", "soec"),  # Map SOEC to separate config
        "ATRSourceNode": ("production", "atr"),
        
        # Storage
        "LPTankNode": ("storage", "lp_tanks"),
        "HPTankNode": ("storage", "hp_tanks"),
        "OxygenBufferNode": ("storage", "oxygen_buffer"), # Note: This might need special handling if it's part of isolated config
        
        # Compression
        "FillingCompressorNode": ("compression", "filling_compressor"),
        "OutgoingCompressorNode": ("compression", "outgoing_compressor"),
        
        # Logic
        "DemandSchedulerNode": ("demand",),
        "EnergyPriceNode": ("energy_price",),
        
        # Utilities
        "BatteryNode": ("battery",),
        "WaterTreatmentNode": ("water_treatment",),
        "OxygenSourceNode": ("external_inputs", "oxygen_source"),
        "HeatSourceNode": ("external_inputs", "heat_source"),
        "MixerNode": ("oxygen_management", "mixer"),
    }
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.errors: List[str] = []
    
    def add_node(self, node: GraphNode) -> None:
        """Register a visual node."""
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node ID: {node.id}")
        self.nodes[node.id] = node
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Register a connection."""
        edge.validate(self.nodes)
        self.edges.append(edge)
    
    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert the graph to a PlantConfig-compatible dictionary.
        
        Returns a dict suitable for PlantBuilder.from_dict().
        """
        config = {
            "name": self._infer_plant_name(),
            "version": "1.0",
            "production": {},
            "storage": {
                "lp_tanks": {
                    "count": 4,
                    "capacity_kg": 50.0,
                    "pressure_bar": 30.0
                },
                "hp_tanks": {
                    "count": 8,
                    "capacity_kg": 200.0,
                    "pressure_bar": 350.0
                }
            },
            "compression": {},
            "demand": {
                "pattern": "constant",
                "base_demand_kg_h": 50.0
            },
            "energy_price": {
                "source": "file",
                "price_file": "/home/stuart/Documentos/Planta Hidrogenio/h2_plant/data/NL_Prices_2024_15min.csv",
                "wind_data_file": "/home/stuart/Documentos/Planta Hidrogenio/h2_plant/data/producao_horaria_2_turbinas.csv",
                "data_resolution_minutes": 15
            },
            "pathway": {
                "allocation_strategy": "COST_OPTIMAL"
            },
            "simulation": {
                "timestep_hours": 1.0,
                "duration_hours": 8760,
                "checkpoint_interval_hours": 168,
            }
        }
        
        # Process each node and extract its configuration
        for node_id, node in self.nodes.items():
            # Handle full class paths (e.g. h2_plant.nodes.PEMStackNode -> PEMStackNode)
            node_type_short = node.type.split('.')[-1]
            
            # Also handle mapping for specific node types that might be aliased
            # e.g. PEMStackNode -> ElectrolyzerNode if needed, but for now we rely on direct mapping
            # or update the mapping keys.
            
            # Let's try to find the key in mapping that matches the short name
            # Note: The mapping keys are like "ElectrolyzerNode", "LPTankNode"
            
            # Special case mapping if needed (e.g. PEMStackNode -> ElectrolyzerNode)
            # But wait, the mapping has "ElectrolyzerNode". 
            # The node type in the graph is "PEMStackNode" (from class name).
            # So we need to ensure the mapping covers "PEMStackNode".
            
            # Let's update the mapping lookup to be more robust or update the mapping itself.
            # For now, let's assume the mapping keys match the class names.
            
            # If the key is not found, try to map known subclasses
            lookup_type = node_type_short
            if lookup_type == "PEMStackNode": lookup_type = "ElectrolyzerNode"
            if lookup_type == "SOECStackNode": lookup_type = "ElectrolyzerNode" # Or separate config?
            if lookup_type == "RectifierNode": lookup_type = "PowerElectronicsNode" # Not in mapping yet
            
            section, *subsection = self.NODE_TYPE_MAPPING.get(lookup_type, (None,))
            
            if section is None:
                # Fallback: try exact match
                section, *subsection = self.NODE_TYPE_MAPPING.get(node.type, (None,))
            
            if section is None:
                self.errors.append(f"Unknown node type: {node.type} (short: {node_type_short})")
                continue
            
            if section not in config:
                config[section] = {}
            
            node_config = self._extract_node_config(node)
            
            if subsection:
                config[section][subsection[0]] = node_config
            else:
                config[section].update(node_config)
        
        # Infer complex topology settings (e.g., isolated storage)
        self._infer_topology_settings(config)
        
        return config

    def _infer_topology_settings(self, config: Dict[str, Any]) -> None:
        """
        Analyze graph connections to set backend configuration flags.
        
        The backend uses 'source_isolated' flag to determine if storage is shared or split.
        The GUI must infer this from the visual connections.
        """
        # 1. Check if we have multiple production sources
        # Check using short type names to handle full module paths
        prod_nodes = [n for n in self.nodes.values() 
                      if n.type.split('.')[-1] in ["PEMStackNode", "SOECStackNode", "ATRReactorNode", "ATRSourceNode"]]
        
        # 2. Check connections from sources to tanks
        source_tank_map = {}
        for edge in self.edges:
            src = self.nodes[edge.source_node_id]
            tgt = self.nodes[edge.target_node_id]
            
            if src in prod_nodes and "Tank" in tgt.type:
                if src.id not in source_tank_map:
                    source_tank_map[src.id] = set()
                source_tank_map[src.id].add(tgt.id)
        
        # 3. Logic: If sources connect to DIFFERENT tanks, it's source_isolated
        #    If all sources connect to the SAME tank(s), it's shared storage
        
        # Simplified detection logic:
        # If we have >1 source and they don't share any tank connections -> Isolated
        if len(prod_nodes) > 1:
            tanks_sets = list(source_tank_map.values())
            if len(tanks_sets) > 1 and tanks_sets[0].isdisjoint(tanks_sets[1]):
                 config["storage"]["source_isolated"] = True
                 
                 # Populate isolated_config
                 # This assumes specific node types map to specific isolated config fields
                 # For prototype, we'll do a best-effort mapping
                 isolated_config = {}

                 # Find electrolyzer tanks
                 elec_nodes = [n for n in prod_nodes if n.type.split('.')[-1] in ["PEMStackNode", "SOECStackNode"]]
                 if elec_nodes:
                     elec_id = elec_nodes[0].id
                     if elec_id in source_tank_map:
                         tank_ids = source_tank_map[elec_id]
                         # Assume first tank found is the representative for config
                         if tank_ids:
                             tank_node = self.nodes[list(tank_ids)[0]]
                             isolated_config["electrolyzer_tanks"] = {
                                 "count": 1, # TODO: Count actual tanks
                                 "capacity_kg": tank_node.properties.get("capacity_kg", 50.0),
                                 "pressure_bar": tank_node.properties.get("pressure_bar", 30.0)
                             }

                 # Find ATR tanks
                 atr_nodes = [n for n in prod_nodes if n.type.split('.')[-1] in ["ATRReactorNode", "ATRSourceNode"]]
                 if atr_nodes:
                     atr_id = atr_nodes[0].id
                     if atr_id in source_tank_map:
                         tank_ids = source_tank_map[atr_id]
                         if tank_ids:
                             tank_node = self.nodes[list(tank_ids)[0]]
                             isolated_config["atr_tanks"] = {
                                 "count": 1,
                                 "capacity_kg": tank_node.properties.get("capacity_kg", 50.0),
                                 "pressure_bar": tank_node.properties.get("pressure_bar", 30.0)
                             }
                 
                 config["storage"]["isolated_config"] = isolated_config

            else:
                 config["storage"]["source_isolated"] = False
        else:
             config["storage"]["source_isolated"] = False

    def _extract_node_config(self, node: GraphNode) -> Dict[str, Any]:
        """Extract properties from a node, potentially restructuring them."""
        
        # Get short type name
        node_type_short = node.type.split('.')[-1]
        
        # Property name mappings: GUI name -> Backend schema name
        PROPERTY_MAPPINGS = {
            'LPTankNode': {
                'tank_count': 'count',
                'capacity_per_tank_kg': 'capacity_kg',
                'operating_pressure_bar': 'pressure_bar'
            },
            'HPTankNode': {
                'tank_count': 'count',
                'capacity_per_tank_kg': 'capacity_kg',
                'operating_pressure_bar': 'pressure_bar'
            },
            'FillingCompressorNode': {
                # No renaming needed - backend uses same names
            },
            'OutgoingCompressorNode': {
                # No renaming needed - backend uses same names
            },
            'PEMStackNode': {
                'rated_power_kw': 'max_power_mw',  # Convert kW to MW
                'efficiency_rated': 'base_efficiency'
            },
            'SOECStackNode': {
                'rated_power_kw': 'max_power_mw',  # Convert kW to MW
            },
            'ConsumerNode': {
                'num_bays': 'num_dispensers',
                'filling_rate_kg_h': 'demand_rate_kg_h'
            }
        }
        
        # Property whitelists: Only these properties are passed to backend
        PROPERTY_WHITELISTS = {
            'PEMStackNode': ['rated_power_kw', 'efficiency_rated', 'component_id'],
            'SOECStackNode': ['rated_power_kw', 'operating_temp_c', 'component_id'],
            'SOECStackNode': ['rated_power_kw', 'efficiency_rated', 'component_id'],
            'LPTankNode': ['tank_count', 'capacity_per_tank_kg', 'operating_pressure_bar', 'component_id'],
            'HPTankNode': ['tank_count', 'capacity_per_tank_kg', 'operating_pressure_bar', 'component_id'],
            'FillingCompressorNode': ['max_flow_kg_h', 'inlet_pressure_bar', 'outlet_pressure_bar', 'efficiency', 'component_id'],
            'OutgoingCompressorNode': ['max_flow_kg_h', 'inlet_pressure_bar', 'outlet_pressure_bar', 'efficiency', 'component_id'],
            'ConsumerNode': ['num_bays', 'filling_rate_kg_h', 'component_id']
        }
        
        # Get mapping and whitelist for this node type
        mapping = PROPERTY_MAPPINGS.get(node_type_short, {})
        whitelist = PROPERTY_WHITELISTS.get(node_type_short, None)
        
        # Apply mapping and filter GUI properties
        config = {}
        for k, v in node.properties.items():
            # Skip GUI-only properties
            if k in ['node_color', 'custom_label']:
                continue
            
            # Apply whitelist if defined for this node type
            if whitelist is not None and k not in whitelist:
                continue
                
            # Map property name if needed
            backend_key = mapping.get(k, k)
            
            # Convert string numbers to actual numbers
            if isinstance(v, str):
                try:
                    # Try int first
                    if '.' not in v:
                        v = int(v)
                    else:
                        v = float(v)
                except (ValueError, TypeError):
                    pass  # Keep as string
            
            # Special conversions
            if backend_key == 'max_power_mw' and k == 'rated_power_kw':
                # Convert kW to MW
                v = v / 1000.0 if isinstance(v, (int, float)) else float(v) / 1000.0
            elif backend_key == 'base_efficiency' and k == 'efficiency_rated':
                # Convert percentage to decimal if needed
                if isinstance(v, (int, float)) and v > 1:
                    v = v / 100.0
            elif backend_key == 'efficiency':
                # Convert percentage to decimal if needed (for compressors, etc.)
                if isinstance(v, (int, float)) and v > 1:
                    v = v / 100.0
                    
            config[backend_key] = v
        
        # Custom handlers
        lookup_type = node_type_short
        if lookup_type == "PEMStackNode": lookup_type = "ElectrolyzerNode"
        if lookup_type == "SOECStackNode": lookup_type = "ElectrolyzerNode"
        
        if lookup_type in ["ElectrolyzerNode", "ATRSourceNode", "BatteryNode", "OxygenSourceNode", "HeatSourceNode"]:
            config["enabled"] = True
            
        if node.type == "WaterTreatmentNode":
            # Remap flat properties to nested structure
            return {
                "treatment_block": {
                    "enabled": True,
                    "max_flow_m3h": config.pop("max_flow_m3h", 10.0),
                    "power_consumption_kw": config.pop("power_consumption_kw", 20.0)
                },
                "quality_test": {"enabled": True},
                "pumps": {
                    "pump_a": {"enabled": True},
                    "pump_b": {"enabled": True}
                }
            }
            
        return config
    
    def _infer_plant_name(self) -> str:
        """Generate a name based on node count and types."""
        # Use short type names to handle full module paths
        prod_count = sum(1 for n in self.nodes.values() 
                         if n.type.split('.')[-1] in ["PEMStackNode", "SOECStackNode", "ATRReactorNode", "ATRSourceNode"])
        stor_count = sum(1 for n in self.nodes.values() 
                         if "Tank" in n.type.split('.')[-1])
        return f"Custom Plant ({prod_count} producers, {stor_count} tanks)"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the entire graph for logical consistency.
        
        Returns (is_valid, error_messages)
        """
        self.errors = []
        
        # Check: At least one production source
        has_production = any(
            n.type in ["ElectrolyzerNode", "ATRSourceNode"] 
            for n in self.nodes.values()
        )
        if not has_production:
            self.errors.append("Plant must have at least one production source")
        
        # Check: At least one storage tank
        has_storage = any(
            "Tank" in n.type 
            for n in self.nodes.values()
        )
        if not has_storage:
            self.errors.append("Plant must have at least one storage tank")
        
        # Check: All connections are valid
        for edge in self.edges:
            try:
                edge.validate(self.nodes)
            except ValueError as e:
                self.errors.append(str(e))
        
        # Check: All node properties are valid (schema validation happens later)
        for node in self.nodes.values():
            if not node.properties:
                self.errors.append(
                    f"Node '{node.display_name}' has no properties configured"
                )
        
        return len(self.errors) == 0, self.errors

    def to_simulation_context(self) -> 'SimulationContext':
        """
        Convert the graph to a validated SimulationContext object.
        """
        from h2_plant.config.models import (
            SimulationContext, PhysicsConfig, TopologyConfig, 
            SimulationConfig, EconomicsConfig, ComponentNode, 
            NodeConnection, PEMPhysicsSpec, SOECPhysicsSpec
        )
        
        # 1. Build TopologyConfig
        nodes = []
        for node_id, node in self.nodes.items():
            # Extract connections
            connections = []
            for edge in self.edges:
                if edge.source_node_id == node_id:
                    # Find target node name/ID
                    target_node = self.nodes.get(edge.target_node_id)
                    target_name = target_node.display_name if target_node else edge.target_node_id
                    
                    connections.append(NodeConnection(
                        source_port=edge.source_port,
                        target_name=edge.target_node_id, # Use ID for robustness
                        target_port=edge.target_port,
                        resource_type=edge.flow_type.value
                    ))
            
            # Map type (simple mapping for now)
            # e.g. h2_plant.nodes.PEMStackNode -> PEM
            node_type_short = node.type.split('.')[-1]
            backend_type = self._map_node_type(node_type_short)
            
            nodes.append(ComponentNode(
                id=node.id,
                type=backend_type,
                connections=connections,
                params=self._extract_node_config(node)
            ))
            
        topology = TopologyConfig(nodes=nodes)
        
        # 2. Build PhysicsConfig (Defaults for now)
        # In a real app, we might extract these from a global settings node or dialog
        physics = PhysicsConfig(
            pem_system=PEMPhysicsSpec(
                max_power_mw=5.0,
                base_efficiency=0.65,
                kwh_per_kg=50.0
            ),
            soec_cluster=SOECPhysicsSpec(
                max_power_nominal_mw=2.4,
                optimal_limit=0.8
            )
        )
        
        # 3. Build SimulationConfig
        # Extract from specific nodes if present (e.g. Logic/Settings node)
        # For now, defaults
        simulation = SimulationConfig(
            timestep_hours=1.0/60.0,
            duration_hours=8760,
            energy_price_file="/home/stuart/Documentos/Planta Hidrogenio/h2_plant/data/NL_Prices_2024_15min.csv",
            wind_data_file="/home/stuart/Documentos/Planta Hidrogenio/h2_plant/data/producao_horaria_2_turbinas.csv"
        )
        
        # 4. Build EconomicsConfig
        economics = EconomicsConfig(
            h2_price_eur_kg=5.0,
            ppa_price_eur_mwh=50.0
        )
        
        return SimulationContext(
            physics=physics,
            topology=topology,
            simulation=simulation,
            economics=economics
        )

    def _map_node_type(self, gui_type: str) -> str:
        """Map GUI node class name to Backend component type string."""
        MAPPING = {
            "PEMStackNode": "PEM",
            "SOECStackNode": "SOEC",
            "FillingCompressorNode": "Compressor",
            "OutgoingCompressorNode": "Compressor",
            "LPTankNode": "Tank",
            "HPTankNode": "Tank",
            "RecirculationPumpNode": "Pump",
            # Add more as needed
        }
        return MAPPING.get(gui_type, "PassiveComponent")
