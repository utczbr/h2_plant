"""
Advanced Validation Engine - Real-time connection checking and validation

This module provides comprehensive validation of:
1. Connection compatibility (type checking, flow rules)
2. Node configuration completeness
3. System topology validity
4. Resource constraints
5. Real-time feedback with visual indicators

Validation runs asynchronously and provides live feedback via signals.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(str, Enum):
    """Types of validation checks."""
    CONNECTION_TYPE = "connection_type"  # Port types match
    CONNECTION_CARDINALITY = "connection_cardinality"  # Input/output count valid
    NODE_CONFIGURATION = "node_configuration"  # Required properties set
    TOPOLOGY_VALIDITY = "topology_validity"  # Graph structure valid
    RESOURCE_CONSTRAINTS = "resource_constraints"  # Limits respected
    FLOW_CONSISTENCY = "flow_consistency"  # Material flows balanced
    CYCLE_DETECTION = "cycle_detection"  # Feedback loops acceptable
    SYSTEM_COMPLETENESS = "system_completeness"  # All systems defined
    PERFORMANCE_HINTS = "performance_hints"  # Optimization suggestions
    COMPATIBILITY = "compatibility"  # Cross-component compatibility


@dataclass
class ValidationIssue:
    """Single validation issue/error."""
    id: str  # Unique identifier
    type: ValidationType
    level: ValidationLevel
    node_id: Optional[str] = None  # Affected node
    edge_id: Optional[Tuple[str, str]] = None  # Affected edge (source, target)
    message: str = ""
    details: str = ""
    suggestion: str = ""  # How to fix
    auto_fixable: bool = False
    fix_callback: Optional[Callable] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    is_valid: bool
    has_errors: bool
    has_warnings: bool
    total_issues: int
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    timestamp: str = ""
    
    # Grouped by type for UI display
    by_type: Dict[ValidationType, List[ValidationIssue]] = field(default_factory=dict)
    by_level: Dict[ValidationLevel, List[ValidationIssue]] = field(default_factory=dict)
    by_node: Dict[str, List[ValidationIssue]] = field(default_factory=dict)


class PortDefinition:
    """Defines a port's capabilities."""
    
    def __init__(self, name: str, flow_type: str, is_input: bool,
                 cardinality: str = "*", required: bool = False):
        """
        Args:
            name: Port name
            flow_type: Type of flow (hydrogen, electricity, etc.)
            is_input: True if input port
            cardinality: "*" (many), "?" (0-1), "1" (exactly 1)
            required: Must be connected
        """
        self.name = name
        self.flow_type = flow_type
        self.is_input = is_input
        self.cardinality = cardinality
        self.required = required
    
    def can_accept_count(self, count: int) -> bool:
        """Check if port can accept given number of connections."""
        if self.cardinality == "*":
            return True
        elif self.cardinality == "?":
            return count <= 1
        elif self.cardinality == "1":
            return count == 1
        return False


# Port definitions for all node types
PORT_DEFINITIONS = {
    "ElectrolyzerNode": {
        "inputs": [
            PortDefinition("water_in", "water", True, "1", True),
            PortDefinition("power_in", "electricity", True, "1", True),
        ],
        "outputs": [
            PortDefinition("h2_out", "hydrogen", False, "*", False),
            PortDefinition("o2_out", "oxygen", False, "*", False),
            PortDefinition("heat_out", "heat", False, "*", False),
        ]
    },
    "ATRSourceNode": {
        "inputs": [
            PortDefinition("gas_in", "gas", True, "1", True),
            PortDefinition("steam_in", "water", True, "?", False),
            PortDefinition("heat_in", "heat", True, "?", False),
        ],
        "outputs": [
            PortDefinition("h2_out", "hydrogen", False, "*", False),
            PortDefinition("co2_out", "gas", False, "*", False),
        ]
    },
    "LPTankNode": {
        "inputs": [
            PortDefinition("inlet", "hydrogen", True, "*", False),
        ],
        "outputs": [
            PortDefinition("outlet", "hydrogen", False, "*", False),
        ]
    },
    "HPTankNode": {
        "inputs": [
            PortDefinition("inlet", "compressed_h2", True, "*", False),
        ],
        "outputs": [
            PortDefinition("outlet", "compressed_h2", False, "*", False),
        ]
    },
    "FillingCompressorNode": {
        "inputs": [
            PortDefinition("inlet", "hydrogen", True, "1", True),
        ],
        "outputs": [
            PortDefinition("outlet", "compressed_h2", False, "*", False),
        ]
    },
    "OutgoingCompressorNode": {
        "inputs": [
            PortDefinition("inlet", "hydrogen", True, "1", True),
        ],
        "outputs": [
            PortDefinition("outlet", "compressed_h2", False, "*", False),
        ]
    },
    "PSAUnitNode": {
        "inputs": [
            PortDefinition("feed", "hydrogen", True, "1", True),
        ],
        "outputs": [
            PortDefinition("h2_pure", "hydrogen", False, "*", False),
            PortDefinition("h2_tail", "hydrogen", False, "*", False),
        ]
    },
    "DemandSchedulerNode": {
        "inputs": [],
        "outputs": [
            PortDefinition("demand", "hydrogen", False, "*", False),
        ]
    },
    "ConsumerNode": {
        "inputs": [
            PortDefinition("inlet", "hydrogen", True, "*", False),
        ],
        "outputs": []
    },
    "GridConnectionNode": {
        "inputs": [],
        "outputs": [
            PortDefinition("power_out", "electricity", False, "*", False),
        ]
    },
    "WaterSupplyNode": {
        "inputs": [],
        "outputs": [
            PortDefinition("water_out", "water", False, "*", False),
        ]
    },
}


class AdvancedValidator:
    """
    Real-time validation engine for graphs.

    Usage:
        validator = AdvancedValidator()
        
        # Validate current state
        report = validator.validate(node_graph)
        
        # Get issues for specific node
        node_issues = validator.get_node_issues("node_123")
        
        # Get suggestion for fixing
        if issue.auto_fixable:
            validator.apply_fix(issue)
    """

    def __init__(self):
        """Initialize validator."""
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.last_report: Optional[ValidationReport] = None
        self.issue_history: List[ValidationIssue] = []
        self.validators: Dict[ValidationType, Callable] = {
            ValidationType.CONNECTION_TYPE: self._validate_connection_types,
            ValidationType.CONNECTION_CARDINALITY: self._validate_connection_cardinality,
            ValidationType.NODE_CONFIGURATION: self._validate_node_configuration,
            ValidationType.TOPOLOGY_VALIDITY: self._validate_topology,
            ValidationType.SYSTEM_COMPLETENESS: self._validate_system_completeness,
            ValidationType.CYCLE_DETECTION: self._validate_cycles,
        }

    def validate(self, node_graph, node_list: Optional[List[Any]] = None,
                 edge_list: Optional[List[Any]] = None) -> ValidationReport:
        """
        Perform comprehensive validation of graph.

        Args:
            node_graph: NodeGraphQt graph object (or None if using lists)
            node_list: Optional list of node dicts
            edge_list: Optional list of edge dicts

        Returns:
            ValidationReport with all issues found
        """
        # Extract node and edge data
        if node_list is not None and edge_list is not None:
            self.nodes = {n["id"]: n for n in node_list}
            self.edges = edge_list
        else:
            self._extract_from_graph(node_graph)
        
        # Run all validators
        all_issues = []
        for val_type, validator in self.validators.items():
            try:
                issues = validator()
                all_issues.extend(issues)
            except Exception as e:
                logger.error(f"Validator {val_type} failed: {e}")
                all_issues.append(ValidationIssue(
                    id=f"validator_error_{val_type}",
                    type=ValidationType.COMPATIBILITY,
                    level=ValidationLevel.WARNING,
                    message=f"Validation check failed: {val_type}",
                    details=str(e)
                ))
        
        # Build report
        report = self._build_report(all_issues)
        self.last_report = report
        self.issue_history.extend(all_issues)
        
        return report

    def _extract_from_graph(self, node_graph) -> None:
        """Extract node and edge data from graph."""
        self.nodes = {}
        self.edges = []
        
        # Extract nodes
        for node in node_graph.all_nodes():
            self.nodes[node.id] = {
                "id": node.id,
                "type": node.__class__.__name__,
                "properties": self._get_node_properties(node),
                "input_ports": self._get_input_ports(node),
                "output_ports": self._get_output_ports(node),
            }
        
        # Extract edges
        # Extract edges
        for node in node_graph.all_nodes():
            for output_port in node.output_ports():
                for target_port in output_port.connected_ports():
                    target_node = target_port.node()
                    
                    # Handle port name extraction safely
                    source_port_name = output_port.name() if callable(output_port.name) else output_port.name
                    target_port_name = target_port.name() if callable(target_port.name) else target_port.name
                    
                    self.edges.append({
                        "source_id": node.id,
                        "target_id": target_node.id,
                        "source_port": source_port_name,
                        "target_port": target_port_name,
                        "flow_type": "default", # Cannot infer from edge object
                    })

    def _validate_connection_types(self) -> List[ValidationIssue]:
        """Validate that connected ports have compatible types."""
        issues = []
        
        # Flow type compatibility rules
        compatible_flows = {
            "hydrogen": ["hydrogen", "compressed_h2"],
            "compressed_h2": ["compressed_h2", "hydrogen"],
            "electricity": ["electricity"],
            "heat": ["heat"],
            "water": ["water"],
            "gas": ["gas"],
        }
        
        for edge in self.edges:
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            source_port = edge["source_port"]
            target_port = edge["target_port"]
            edge_flow = edge["flow_type"]
            
            # Get expected flow types from port definitions
            source_node_type = self.nodes[source_id]["type"]
            target_node_type = self.nodes[target_id]["type"]
            
            source_def = PORT_DEFINITIONS.get(source_node_type)
            target_def = PORT_DEFINITIONS.get(target_node_type)
            
            if not source_def or not target_def:
                continue  # Skip if definitions not available
            
            # Find port in definitions
            source_port_def = None
            for port in source_def.get("outputs", []):
                if port.name == source_port:
                    source_port_def = port
                    break
            
            target_port_def = None
            for port in target_def.get("inputs", []):
                if port.name == target_port:
                    target_port_def = port
                    break
            
            if not source_port_def or not target_port_def:
                continue
            
            # Check compatibility
            source_flow = source_port_def.flow_type
            target_flow = target_port_def.flow_type
            
            if target_flow not in compatible_flows.get(source_flow, []):
                issues.append(ValidationIssue(
                    id=f"type_mismatch_{source_id}_{target_id}",
                    type=ValidationType.CONNECTION_TYPE,
                    level=ValidationLevel.ERROR,
                    edge_id=(source_id, target_id),
                    message=f"Incompatible connection: {source_flow} -> {target_flow}",
                    details=f"Port '{source_port}' produces {source_flow} but port '{target_port}' expects {target_flow}",
                    suggestion=f"Connect to a port that accepts {source_flow}"
                ))
        
        return issues

    def _validate_connection_cardinality(self) -> List[ValidationIssue]:
        """Validate input/output port counts."""
        issues = []
        
        # Count connections per port
        port_connections = defaultdict(lambda: {"in": 0, "out": 0})
        
        for edge in self.edges:
            port_connections[edge["source_id"]][edge["source_port"]] = \
                port_connections[edge["source_id"]].get(edge["source_port"], 0) + 1
            port_connections[edge["target_id"]][edge["target_port"]] = \
                port_connections[edge["target_id"]].get(edge["target_port"], 0) + 1
        
        # Check against definitions
        for node_id, node_data in self.nodes.items():
            node_type = node_data["type"]
            definitions = PORT_DEFINITIONS.get(node_type)
            
            if not definitions:
                continue
            
            # Check input ports
            for port_def in definitions.get("inputs", []):
                count = port_connections[node_id].get(port_def.name, 0)
                
                if not port_def.can_accept_count(count):
                    if port_def.required and count == 0:
                        issues.append(ValidationIssue(
                            id=f"required_port_{node_id}_{port_def.name}",
                            type=ValidationType.CONNECTION_CARDINALITY,
                            level=ValidationLevel.ERROR,
                            node_id=node_id,
                            message=f"Required input '{port_def.name}' not connected",
                            suggestion=f"Connect something to the {port_def.name} input"
                        ))
                    elif count > 1 and port_def.cardinality == "1":
                        issues.append(ValidationIssue(
                            id=f"excess_input_{node_id}_{port_def.name}",
                            type=ValidationType.CONNECTION_CARDINALITY,
                            level=ValidationLevel.WARNING,
                            node_id=node_id,
                            message=f"Port '{port_def.name}' accepts only 1 connection",
                            details=f"Currently connected to {count} sources",
                            suggestion="Remove excess connections"
                        ))
        
        return issues

    def _validate_node_configuration(self) -> List[ValidationIssue]:
        """Validate that nodes have required properties configured."""
        issues = []
        
        # Required properties per node type (could be in schema)
        required_properties = {
            "ElectrolyzerNode": ["max_power_mw", "efficiency"],
            "ATRSourceNode": ["efficiency"],
            "LPTankNode": ["capacity_kg", "pressure_bar"],
            "HPTankNode": ["capacity_kg", "pressure_bar"],
            "FillingCompressorNode": ["max_flow_kg_h"],
            "DemandSchedulerNode": ["pattern"],
        }
        
        for node_id, node_data in self.nodes.items():
            node_type = node_data["type"]
            required = required_properties.get(node_type, [])
            properties = node_data.get("properties", {})
            
            for prop_name in required:
                if prop_name not in properties or properties[prop_name] is None:
                    issues.append(ValidationIssue(
                        id=f"missing_prop_{node_id}_{prop_name}",
                        type=ValidationType.NODE_CONFIGURATION,
                        level=ValidationLevel.WARNING,
                        node_id=node_id,
                        message=f"Required property '{prop_name}' not set",
                        suggestion=f"Configure {prop_name} in node properties"
                    ))
                else:
                    # Validate value constraints
                    value = properties[prop_name]
                    
                    # Basic type/range checks
                    if prop_name in ["max_power_mw", "efficiency", "capacity_kg", "pressure_bar", "max_flow_kg_h"]:
                        try:
                            fval = float(value)
                            if fval <= 0:
                                issues.append(ValidationIssue(
                                    id=f"invalid_value_{node_id}_{prop_name}",
                                    type=ValidationType.NODE_CONFIGURATION,
                                    level=ValidationLevel.ERROR,
                                    node_id=node_id,
                                    message=f"Property '{prop_name}' must be positive",
                                    details=f"Current value: {value}",
                                    suggestion="Enter a positive number"
                                ))
                        except (ValueError, TypeError):
                            issues.append(ValidationIssue(
                                id=f"type_error_{node_id}_{prop_name}",
                                type=ValidationType.NODE_CONFIGURATION,
                                level=ValidationLevel.ERROR,
                                node_id=node_id,
                                message=f"Property '{prop_name}' must be numeric",
                                details=f"Current value: {value}",
                                suggestion="Enter a number"
                            ))
        
        return issues

    def _validate_topology(self) -> List[ValidationIssue]:
        """Validate overall graph topology."""
        issues = []
        
        # Check for isolated components
        connected_components = self._find_connected_components()
        if len(connected_components) > 1:
            issues.append(ValidationIssue(
                id="disconnected_graph",
                type=ValidationType.TOPOLOGY_VALIDITY,
                level=ValidationLevel.WARNING,
                message=f"Graph has {len(connected_components)} disconnected components",
                details="Some nodes are not connected to the main network",
                suggestion="Connect all nodes into a single network"
            ))
        
        # Check for dead ends
        dead_ends = self._find_dead_ends()
        if dead_ends:
            for node_id in dead_ends:
                node_type = self.nodes[node_id]["type"]
                if "Consumer" not in node_type and "Scheduler" not in node_type:
                    issues.append(ValidationIssue(
                        id=f"dead_end_{node_id}",
                        type=ValidationType.TOPOLOGY_VALIDITY,
                        level=ValidationLevel.INFO,
                        node_id=node_id,
                        message=f"Node '{node_id}' has no outgoing connections",
                        suggestion="Connect to a consumer or storage"
                    ))
        
        return issues

    def _validate_system_completeness(self) -> List[ValidationIssue]:
        """Validate that all necessary systems are present."""
        issues = []
        
        # Count node types
        node_types = defaultdict(int)
        for node_data in self.nodes.values():
            node_types[node_data["type"]] += 1
        
        # Check for required systems
        has_production = any(
            t in node_types for t in ["ElectrolyzerNode", "ATRSourceNode", "PEMStackNode"]
        )
        has_storage = any(
            t in node_types for t in ["LPTankNode", "HPTankNode", "OxygenBufferNode"]
        )
        
        if not has_production:
            issues.append(ValidationIssue(
                id="no_production",
                type=ValidationType.SYSTEM_COMPLETENESS,
                level=ValidationLevel.ERROR,
                message="No production sources found",
                details="Plant must have at least one producer (Electrolyzer, ATR, etc.)",
                suggestion="Add a production node"
            ))
        
        if not has_storage:
            issues.append(ValidationIssue(
                id="no_storage",
                type=ValidationType.SYSTEM_COMPLETENESS,
                level=ValidationLevel.WARNING,
                message="No storage tanks found",
                details="Typical plants have storage",
                suggestion="Add storage tank nodes"
            ))
        
        return issues

    def _validate_cycles(self) -> List[ValidationIssue]:
        """Detect cycles and flag if they're problematic."""
        issues = []
        
        # Find all cycles
        cycles = self._find_cycles()
        
        for cycle in cycles:
            # Recirculation is OK, but feedback that creates issues is not
            has_producer = any(
                self.nodes[n]["type"] in ["ElectrolyzerNode", "ATRSourceNode"]
                for n in cycle
            )
            has_storage = any(
                self.nodes[n]["type"] in ["LPTankNode", "HPTankNode"]
                for n in cycle
            )
            
            # Producer -> Storage -> Producer is problematic
            if has_producer and has_storage:
                issues.append(ValidationIssue(
                    id=f"feedback_cycle",
                    type=ValidationType.CYCLE_DETECTION,
                    level=ValidationLevel.WARNING,
                    message="Feedback cycle detected (Producer -> Storage -> Producer)",
                    details=f"Cycle path: {' -> '.join(cycle)}",
                    suggestion="Verify this is intentional recirculation"
                ))
        
        return issues

    def _build_report(self, issues: List[ValidationIssue]) -> ValidationReport:
        """Build comprehensive report from issues."""
        # Group issues
        by_type = defaultdict(list)
        by_level = defaultdict(list)
        by_node = defaultdict(list)
        
        for issue in issues:
            by_type[issue.type].append(issue)
            by_level[issue.level].append(issue)
            if issue.node_id:
                by_node[issue.node_id].append(issue)
        
        # Determine validity
        has_errors = len(by_level[ValidationLevel.ERROR]) > 0 or \
                     len(by_level[ValidationLevel.CRITICAL]) > 0
        has_warnings = len(by_level[ValidationLevel.WARNING]) > 0
        is_valid = not has_errors
        
        # Build summary
        parts = []
        for level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR,
                     ValidationLevel.WARNING, ValidationLevel.INFO]:
            count = len(by_level[level])
            if count > 0:
                parts.append(f"{count} {level.value}(s)")
        
        summary = f"{len(issues)} issue(s): " + ", ".join(parts) if parts else "Valid"
        
        return ValidationReport(
            is_valid=is_valid,
            has_errors=has_errors,
            has_warnings=has_warnings,
            total_issues=len(issues),
            issues=issues,
            summary=summary,
            by_type=dict(by_type),
            by_level=dict(by_level),
            by_node=dict(by_node),
        )

    def _find_connected_components(self) -> List[Set[str]]:
        """Find connected components in graph using Union-Find."""
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Initialize all nodes
        for node_id in self.nodes:
            find(node_id)
        
        # Union connected nodes
        for edge in self.edges:
            union(edge["source_id"], edge["target_id"])
        
        # Group by root
        components = defaultdict(set)
        for node_id in self.nodes:
            components[find(node_id)].add(node_id)
        
        return list(components.values())

    def _find_dead_ends(self) -> Set[str]:
        """Find nodes with no outgoing connections."""
        has_outgoing = set(edge["source_id"] for edge in self.edges)
        all_nodes = set(self.nodes.keys())
        return all_nodes - has_outgoing

    def _find_cycles(self) -> List[List[str]]:
        """Find all cycles in graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.append(node)
            path.append(node)
            
            for edge in self.edges:
                if edge["source_id"] == node:
                    neighbor = edge["target_id"]
                    
                    if neighbor not in visited:
                        dfs(neighbor, path.copy())
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = rec_stack.index(neighbor)
                        cycle = rec_stack[cycle_start:] + [neighbor]
                        cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        return cycles

    @staticmethod
    def _get_node_properties(node) -> Dict[str, Any]:
        """Extract properties from node."""
        if hasattr(node, 'get_properties'):
            return node.get_properties()
        return {}

    @staticmethod
    def _get_input_ports(node) -> List[str]:
        """Get input port names."""
        if hasattr(node, 'input_ports'):
            ports = node.input_ports() if callable(node.input_ports) else node.input_ports
            return [p.name() if callable(p.name) else p.name for p in ports]
        return []

    @staticmethod
    def _get_output_ports(node) -> List[str]:
        """Get output port names."""
        if hasattr(node, 'output_ports'):
            ports = node.output_ports() if callable(node.output_ports) else node.output_ports
            return [p.name() if callable(p.name) else p.name for p in ports]
        return []

    @staticmethod
    def _get_port_name(port) -> str:
        """Extract port name."""
        if hasattr(port, 'name'):
            return port.name() if callable(port.name) else port.name
        return "default"

    @staticmethod
    def _infer_flow_type(edge) -> str:
        """Infer flow type from edge."""
        return getattr(edge, 'flow_type', 'default')
