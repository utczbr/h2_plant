"""
Topology Inference Engine - Auto-detect system assignment from connections

This module analyzes the graph structure to automatically determine how nodes
are organized into systems and detect the topology configuration without explicit
user markup. It infers:

1. Production system assignment (which production node feeds which storage)
2. Storage isolation patterns (shared vs. isolated storage)
3. Compression chains (filling, outgoing, recirculation paths)
4. Demand chains (how demand flows through the system)
5. System boundaries and data flow patterns
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class SystemType(str, Enum):
    """Recognized system types in the plant."""
    PRODUCTION = "production"
    STORAGE = "storage"
    COMPRESSION = "compression"
    THERMAL = "thermal"
    SEPARATION = "separation"
    LOGIC = "logic"
    UTILITIES = "utilities"
    EXTERNAL = "external"


class TopologyPattern(str, Enum):
    """Detected topology patterns."""
    SINGLE_SOURCE = "single_source"  # One producer to shared storage
    ISOLATED_SOURCES = "isolated_sources"  # Multiple producers with dedicated storage
    MULTI_PRODUCER_SHARED = "multi_producer_shared"  # Multiple producers to shared storage
    CASCADED_STORAGE = "cascaded_storage"  # Storage chains (LP -> HP)
    BRANCHED_COMPRESSION = "branched_compression"  # Multiple compression paths
    RECIRCULATION_LOOP = "recirculation_loop"  # Closed loop detected
    COMPLEX_NETWORK = "complex_network"  # Multiple interconnected systems


@dataclass
class NodeMetadata:
    """Extracted metadata about a node."""
    node_id: str
    node_type: str
    system_type: SystemType
    display_name: str
    properties: Dict[str, Any]
    inbound_ports: List[str] = field(default_factory=list)
    outbound_ports: List[str] = field(default_factory=list)


@dataclass
class EdgeMetadata:
    """Extracted metadata about an edge."""
    source_id: str
    target_id: str
    source_port: str
    target_port: str
    flow_type: str


@dataclass
class SystemAssignment:
    """Assignment of a node to a system."""
    node_id: str
    system_id: str
    system_type: SystemType
    role: str  # 'producer', 'storage', 'consumer', 'transformer', etc.
    confidence: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class TopoAnalysis:
    """Complete topology analysis result."""
    detected_pattern: TopologyPattern
    system_assignments: Dict[str, SystemAssignment]
    production_systems: Dict[str, List[str]]  # {producer_id: [tank_ids]}
    storage_chains: Dict[str, List[str]]  # {storage_id: [downstream_storage_ids]}
    compression_paths: List[List[str]]  # [[path_nodes], ...]
    has_recirculation: bool
    flow_networks: Dict[str, Set[str]]  # {system_id: {connected_node_ids}}
    confidence_score: float
    warnings: List[str]
    metadata: Dict[str, Any]


class TopologyInferenceEngine:
    """
    Analyzes graph structure to auto-detect topology and system assignments.

    Usage:
        engine = TopologyInferenceEngine()
        nodes, edges = extract_from_graph(node_graph)  # Your graph extraction
        analysis = engine.infer(nodes, edges)
        
        print(f"Pattern: {analysis.detected_pattern}")
        for node_id, assignment in analysis.system_assignments.items():
            print(f"  {node_id} -> {assignment.system_id} ({assignment.role})")
    """

    # Mapping from node types to system types
    NODE_TYPE_TO_SYSTEM = {
        # Production
        "ElectrolyzerNode": SystemType.PRODUCTION,
        "PEMStackNode": SystemType.PRODUCTION,
        "SOECStackNode": SystemType.PRODUCTION,
        "ATRSourceNode": SystemType.PRODUCTION,
        "ATRReactorNode": SystemType.PRODUCTION,
        "OxygenSourceNode": SystemType.EXTERNAL,
        "NaturalGasSupplyNode": SystemType.EXTERNAL,
        
        # Storage
        "LPTankNode": SystemType.STORAGE,
        "HPTankNode": SystemType.STORAGE,
        "OxygenBufferNode": SystemType.STORAGE,
        "SeparationTankNode": SystemType.STORAGE,
        
        # Compression
        "FillingCompressorNode": SystemType.COMPRESSION,
        "OutgoingCompressorNode": SystemType.COMPRESSION,
        "ProcessCompressorNode": SystemType.COMPRESSION,
        
        # Thermal
        "HeatExchangerNode": SystemType.THERMAL,
        "SteamGeneratorNode": SystemType.THERMAL,
        "ChillerNode": SystemType.THERMAL,
        "DryCoolerNode": SystemType.THERMAL,
        
        # Separation
        "PSAUnitNode": SystemType.SEPARATION,
        "WGSReactorNode": SystemType.SEPARATION,
        "CoalescerNode": SystemType.SEPARATION,
        "KnockOutDrumNode": SystemType.SEPARATION,
        "DeoxoReactorNode": SystemType.SEPARATION,
        "TSAUnitNode": SystemType.SEPARATION,
        
        # Logic & Control
        "DemandSchedulerNode": SystemType.LOGIC,
        "EnergyPriceNode": SystemType.LOGIC,
        "ArbitrageNode": SystemType.LOGIC,
        
        # Utilities
        "BatteryNode": SystemType.UTILITIES,
        "GridConnectionNode": SystemType.EXTERNAL,
        "WaterSupplyNode": SystemType.EXTERNAL,
        "MixerNode": SystemType.UTILITIES,
        "WaterMixerNode": SystemType.UTILITIES,
        
        # Fluid
        "RecirculationPumpNode": SystemType.COMPRESSION,
        "PumpNode": SystemType.COMPRESSION,
        
        # Logistics
        "ConsumerNode": SystemType.LOGIC,
    }

    def __init__(self):
        """Initialize the inference engine."""
        self.nodes: Dict[str, NodeMetadata] = {}
        self.edges: List[EdgeMetadata] = []
        self.graph: Dict[str, List[str]] = defaultdict(list)  # Adjacency list
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)
        self.visited: Set[str] = set()

    def infer(self, nodes: List[Dict[str, Any]], 
              edges: List[Dict[str, Any]]) -> TopoAnalysis:
        """
        Analyze graph topology and infer system assignments.

        Args:
            nodes: List of node dicts with keys: id, type, display_name, properties
            edges: List of edge dicts with keys: source_id, target_id, source_port, 
                   target_port, flow_type

        Returns:
            TopoAnalysis with detected pattern and assignments
        """
        self._load_graph(nodes, edges)
        
        # Analyze structure
        system_assignments = self._infer_assignments()
        production_map = self._map_production_to_storage()
        storage_chains = self._detect_storage_chains()
        compression_paths = self._detect_compression_paths()
        has_recirculation = self._detect_recirculation()
        flow_networks = self._detect_flow_networks()
        
        # Detect overall pattern
        pattern = self._classify_topology(
            production_map, storage_chains, compression_paths, has_recirculation
        )
        
        confidence = self._calculate_confidence(system_assignments)
        warnings = self._validate_topology(system_assignments)
        
        return TopoAnalysis(
            detected_pattern=pattern,
            system_assignments=system_assignments,
            production_systems=production_map,
            storage_chains=storage_chains,
            compression_paths=compression_paths,
            has_recirculation=has_recirculation,
            flow_networks=flow_networks,
            confidence_score=confidence,
            warnings=warnings,
            metadata=self._collect_metadata()
        )

    def _load_graph(self, nodes: List[Dict], edges: List[Dict]) -> None:
        """Load and normalize node/edge data."""
        # Load nodes
        for node in nodes:
            node_id = node["id"]
            node_type = node["type"]
            system_type = self.NODE_TYPE_TO_SYSTEM.get(
                node_type, SystemType.UTILITIES
            )
            
            metadata = NodeMetadata(
                node_id=node_id,
                node_type=node_type,
                system_type=system_type,
                display_name=node.get("display_name", node_id),
                properties=node.get("properties", {}),
                inbound_ports=node.get("inbound_ports", []),
                outbound_ports=node.get("outbound_ports", [])
            )
            self.nodes[node_id] = metadata
        
        # Load edges
        for edge in edges:
            edge_meta = EdgeMetadata(
                source_id=edge["source_id"],
                target_id=edge["target_id"],
                source_port=edge.get("source_port", ""),
                target_port=edge.get("target_port", ""),
                flow_type=edge.get("flow_type", "default")
            )
            self.edges.append(edge_meta)
            
            # Build adjacency lists
            self.graph[edge["source_id"]].append(edge["target_id"])
            self.reverse_graph[edge["target_id"]].append(edge["source_id"])

    def _infer_assignments(self) -> Dict[str, SystemAssignment]:
        """
        Infer which system each node belongs to based on:
        1. Node type (primary)
        2. Connection patterns (producers -> storage -> consumers)
        3. Role inference from graph position
        """
        assignments = {}

        for node_id, metadata in self.nodes.items():
            system_type = metadata.system_type
            role = self._infer_role(node_id, metadata)
            system_id = self._infer_system_id(node_id, metadata, role)
            
            confidence = self._calculate_node_confidence(node_id, metadata, role)
            reasoning = self._generate_reasoning(node_id, metadata, role)
            
            assignments[node_id] = SystemAssignment(
                node_id=node_id,
                system_id=system_id,
                system_type=system_type,
                role=role,
                confidence=confidence,
                reasoning=reasoning
            )
        
        return assignments

    def _infer_role(self, node_id: str, metadata: NodeMetadata) -> str:
        """
        Infer node role based on position in graph:
        - 'producer': Generates hydrogen/resources
        - 'storage': Holds/buffers material
        - 'consumer': Uses material/resources
        - 'transformer': Changes form (compression, separation, etc.)
        - 'source': External input
        - 'sink': External output
        """
        if metadata.system_type == SystemType.PRODUCTION:
            return "producer"
        elif metadata.system_type == SystemType.STORAGE:
            return "storage"
        elif metadata.system_type == SystemType.COMPRESSION:
            return "transformer"
        elif metadata.system_type == SystemType.THERMAL:
            return "transformer"
        elif metadata.system_type == SystemType.SEPARATION:
            return "transformer"
        elif metadata.system_type == SystemType.LOGIC:
            return "controller"
        elif metadata.system_type == SystemType.EXTERNAL:
            has_inbound = len(self.reverse_graph[node_id]) > 0
            has_outbound = len(self.graph[node_id]) > 0
            
            if has_inbound and not has_outbound:
                return "sink"
            elif has_outbound and not has_inbound:
                return "source"
            else:
                return "transceiver"
        else:
            return "utility"

    def _infer_system_id(self, node_id: str, metadata: NodeMetadata, 
                         role: str) -> str:
        """
        Infer which system this node belongs to.
        Returns a system identifier like "production_1", "storage_1", etc.
        """
        system_type = metadata.system_type
        
        # For producers, they define their own system
        if role == "producer":
            return f"production_{node_id}"
        
        # For storage, check what producers feed it
        if role == "storage":
            producers = self._find_upstream_producers(node_id)
            if len(producers) == 1:
                return f"production_{producers[0]}"
            elif len(producers) > 1:
                return "production_shared"
            else:
                return "storage_orphaned"
        
        # For transformers/consumers, find what system they're part of
        if role in ["transformer", "consumer", "sink"]:
            producers = self._find_upstream_producers(node_id)
            if len(producers) == 1:
                return f"production_{producers[0]}"
            elif len(producers) > 1:
                return "production_shared"
            else:
                return "system_external"
        
        # For sources, create external system
        if role == "source":
            return f"external_{node_id}"
        
        # For controllers, map to controlled system
        if role == "controller":
            return "system_control"
        
        return f"{system_type}_{node_id}"

    def _find_upstream_producers(self, node_id: str) -> List[str]:
        """Recursively find all producer nodes upstream of this node."""
        producers = []
        visited = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for upstream in self.reverse_graph[current]:
                metadata = self.nodes.get(upstream)
                if metadata and metadata.system_type == SystemType.PRODUCTION:
                    producers.append(upstream)
                else:
                    queue.append(upstream)
        
        return list(set(producers))

    def _map_production_to_storage(self) -> Dict[str, List[str]]:
        """Map each producer to its downstream storage nodes."""
        mapping = defaultdict(list)
        
        for node_id, metadata in self.nodes.items():
            if metadata.system_type == SystemType.PRODUCTION:
                # Find all storage nodes reachable from this producer
                storage_nodes = self._find_downstream_storage(node_id)
                mapping[node_id] = storage_nodes
        
        return dict(mapping)

    def _find_downstream_storage(self, node_id: str) -> List[str]:
        """Find all storage nodes reachable from this node."""
        storage = []
        visited = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for downstream in self.graph[current]:
                metadata = self.nodes.get(downstream)
                if metadata:
                    if metadata.system_type == SystemType.STORAGE:
                        storage.append(downstream)
                    # Continue searching (in case storage is not directly connected)
                    if metadata.system_type != SystemType.LOGIC:
                        queue.append(downstream)
        
        return list(set(storage))

    def _detect_storage_chains(self) -> Dict[str, List[str]]:
        """Detect chains of storage (e.g., LP -> HP compression)."""
        chains = {}
        
        storage_nodes = {
            node_id: metadata for node_id, metadata in self.nodes.items()
            if metadata.system_type == SystemType.STORAGE
        }
        
        for storage_id in storage_nodes:
            downstream = [
                target for target in self.graph[storage_id]
                if self.nodes[target].system_type == SystemType.STORAGE
            ]
            if downstream:
                chains[storage_id] = downstream
        
        return chains

    def _detect_compression_paths(self) -> List[List[str]]:
        """Find linear compression paths (sequences of compressors)."""
        paths = []
        visited_in_path = set()
        
        for node_id, metadata in self.nodes.items():
            if metadata.system_type == SystemType.COMPRESSION and node_id not in visited_in_path:
                path = self._trace_compression_path(node_id)
                if len(path) > 1:
                    paths.append(path)
                    visited_in_path.update(path)
        
        return paths

    def _trace_compression_path(self, start_id: str) -> List[str]:
        """Trace a linear compression path starting from a compressor."""
        path = [start_id]
        current = start_id
        
        while True:
            # Find next compressor in chain
            next_compressor = None
            for downstream in self.graph[current]:
                metadata = self.nodes.get(downstream)
                if metadata and metadata.system_type == SystemType.COMPRESSION:
                    next_compressor = downstream
                    break
            
            if next_compressor and next_compressor not in path:
                path.append(next_compressor)
                current = next_compressor
            else:
                break
        
        return path

    def _detect_recirculation(self) -> bool:
        """Detect if there are cycles in the graph (recirculation)."""
        for start_node in self.nodes:
            if self._has_cycle_from(start_node):
                return True
        return False

    def _has_cycle_from(self, start_id: str) -> bool:
        """Check if there's a cycle starting from a node using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in self.graph[node_id]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        return dfs(start_id)

    def _detect_flow_networks(self) -> Dict[str, Set[str]]:
        """Partition nodes into connected flow networks."""
        networks = {}
        visited = set()
        network_count = 0
        
        for node_id in self.nodes:
            if node_id not in visited:
                component = self._get_connected_component(node_id)
                network_id = f"network_{network_count}"
                networks[network_id] = component
                visited.update(component)
                network_count += 1
        
        return networks

    def _get_connected_component(self, start_id: str) -> Set[str]:
        """Get all nodes connected to start_id (forward and backward)."""
        component = set()
        queue = deque([start_id])
        
        while queue:
            node_id = queue.popleft()
            if node_id in component:
                continue
            component.add(node_id)
            
            # Add all neighbors (both directions)
            for neighbor in self.graph[node_id]:
                if neighbor not in component:
                    queue.append(neighbor)
            
            for neighbor in self.reverse_graph[node_id]:
                if neighbor not in component:
                    queue.append(neighbor)
        
        return component

    def _classify_topology(self, production_map: Dict[str, List[str]],
                          storage_chains: Dict[str, List[str]],
                          compression_paths: List[List[str]],
                          has_recirculation: bool) -> TopologyPattern:
        """Classify overall topology based on detected features."""
        num_producers = len(production_map)
        num_storage = sum(len(tanks) for tanks in production_map.values())
        
        # Single producer pattern
        if num_producers == 1:
            if has_recirculation:
                return TopologyPattern.RECIRCULATION_LOOP
            elif num_storage == 1:
                return TopologyPattern.SINGLE_SOURCE
            elif storage_chains:
                return TopologyPattern.CASCADED_STORAGE
            else:
                return TopologyPattern.SINGLE_SOURCE
        
        # Multiple producers
        if num_producers > 1:
            # Check if producers have isolated storage
            all_storage_sets = list(production_map.values())
            has_isolated = any(
                not any(set(s1) & set(s2) for s2 in all_storage_sets if s2 != s1)
                for s1 in all_storage_sets
            )
            
            if has_isolated:
                return TopologyPattern.ISOLATED_SOURCES
            else:
                return TopologyPattern.MULTI_PRODUCER_SHARED
        
        # Complex detection
        if compression_paths and len(compression_paths) > 1:
            return TopologyPattern.BRANCHED_COMPRESSION
        
        if has_recirculation:
            return TopologyPattern.RECIRCULATION_LOOP
        
        return TopologyPattern.COMPLEX_NETWORK

    def _calculate_node_confidence(self, node_id: str, 
                                   metadata: NodeMetadata,
                                   role: str) -> float:
        """Calculate confidence in role/system assignment for a node."""
        confidence = 0.9  # Start high
        
        # Reduce confidence if role is ambiguous
        if role in ["transceiver", "utility"]:
            confidence -= 0.2
        
        # Reduce if node is isolated
        if (node_id not in self.graph or len(self.graph[node_id]) == 0) and \
           (node_id not in self.reverse_graph or len(self.reverse_graph[node_id]) == 0):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))

    def _calculate_confidence(self, assignments: Dict[str, SystemAssignment]) -> float:
        """Calculate overall confidence in the analysis."""
        if not assignments:
            return 0.0
        
        avg_confidence = sum(a.confidence for a in assignments.values()) / len(assignments)
        
        # Reduce if many high-level system assignments
        orphaned = sum(1 for a in assignments.values() if "orphaned" in a.system_id)
        orphaned_penalty = orphaned / len(assignments) * 0.2
        
        return max(0.0, avg_confidence - orphaned_penalty)

    def _generate_reasoning(self, node_id: str, metadata: NodeMetadata, 
                            role: str) -> str:
        """Generate human-readable reasoning for an assignment."""
        reasons = [
            f"Node type '{metadata.node_type}' -> {metadata.system_type.value}"
        ]
        
        inbound = len(self.reverse_graph[node_id])
        outbound = len(self.graph[node_id])
        
        if inbound > 0:
            reasons.append(f"Receives input from {inbound} node(s)")
        if outbound > 0:
            reasons.append(f"Feeds output to {outbound} node(s)")
        
        if role != "utility":
            reasons.append(f"Inferred role: {role}")
        
        return "; ".join(reasons)

    def _validate_topology(self, assignments: Dict[str, SystemAssignment]) -> List[str]:
        """Generate warnings about potential topology issues."""
        warnings = []
        
        # Check for orphaned nodes
        orphaned = [
            a for a in assignments.values() if "orphaned" in a.system_id
        ]
        if orphaned:
            warnings.append(
                f"Found {len(orphaned)} orphaned node(s) with no clear system assignment"
            )
        
        # Check for missing production
        has_production = any(
            a.role == "producer" for a in assignments.values()
        )
        if not has_production:
            warnings.append("No production nodes detected")
        
        # Check for missing storage
        has_storage = any(
            a.role == "storage" for a in assignments.values()
        )
        if not has_storage:
            warnings.append("No storage nodes detected")
        
        return warnings

    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect additional metadata about the analysis."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": set(m.node_type for m in self.nodes.values()),
            "system_types": [st.value for st in set(
                m.system_type for m in self.nodes.values()
            )],
        }

    def detect_arbitrage_topology(self) -> str:
        """
        Detect the connected topology for the Arbitrage Node.
        Returns:
            'SOEC_ONLY': If Arbitrage is connected to SOEC but not PEM.
            'SOEC_PEM': If Arbitrage is connected to both SOEC and PEM.
            'UNKNOWN': If Arbitrage node is not found or connections are unclear.
        """
        arbitrage_node_id = None
        for node_id, metadata in self.nodes.items():
            if metadata.node_type == "ArbitrageNode":
                arbitrage_node_id = node_id
                break
        
        if not arbitrage_node_id:
            return "UNKNOWN"
            
        # Check downstream connections
        connected_types = set()
        if arbitrage_node_id in self.graph:
            for target_id in self.graph[arbitrage_node_id]:
                target_metadata = self.nodes.get(target_id)
                if target_metadata:
                    connected_types.add(target_metadata.node_type)
        
        has_soec = "SOECStackNode" in connected_types or "ElectrolyzerNode" in connected_types # Assuming ElectrolyzerNode is SOEC base
        has_pem = "PEMStackNode" in connected_types
        
        # Also check for 'ElectrolyzerNode' which might be generic, check properties if needed
        # But for now, rely on type names.
        
        if has_soec and has_pem:
            return "SOEC_PEM"
        elif has_soec:
            return "SOEC_ONLY"
        else:
            return "UNKNOWN"


# Convenience functions
def extract_nodes_edges_from_graph(node_graph) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract nodes and edges from NodeGraphQt graph object.
    Returns (nodes, edges) suitable for TopologyInferenceEngine.infer()
    """
    nodes = []
    edges = []
    
    # Extract nodes
    for node in node_graph.all_nodes():
        # Handle ports safely
        in_ports = node.input_ports() if callable(node.input_ports) else (node.input_ports if hasattr(node, 'input_ports') else [])
        out_ports = node.output_ports() if callable(node.output_ports) else (node.output_ports if hasattr(node, 'output_ports') else [])
        
        node_dict = {
            "id": node.id,
            "type": node.__class__.__name__,
            "display_name": node.name() if callable(getattr(node, 'name', None)) else getattr(node, 'name', node.id),
            "properties": node.get_properties() if hasattr(node, 'get_properties') else {},
            "inbound_ports": [p.name() if callable(p.name) else p.name for p in in_ports],
            "outbound_ports": [p.name() if callable(p.name) else p.name for p in out_ports],
        }
        nodes.append(node_dict)
    
    # Extract edges
    # Extract edges
    for node in node_graph.all_nodes():
        for output_port in node.output_ports():
            for target_port in output_port.connected_ports():
                target_node = target_port.node()
                
                # Handle port name extraction safely
                source_port_name = output_port.name() if callable(output_port.name) else output_port.name
                target_port_name = target_port.name() if callable(target_port.name) else target_port.name
                
                edge_dict = {
                    "source_id": node.id,
                    "target_id": target_node.id,
                    "source_port": source_port_name,
                    "target_port": target_port_name,
                    "flow_type": "default",
                }
                edges.append(edge_dict)
    
    return nodes, edges
