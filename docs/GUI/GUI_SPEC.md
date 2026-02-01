# H₂ Plant GUI: Technical Implementation Specification

**Version:** 1.0 Prototype  
**Target Release:** 4-week iteration  
**Status:** Ready for Development

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Pattern](#architecture-pattern)
3. [Module Specifications](#module-specifications)
4. [Backend Modifications](#backend-modifications)
5. [Node System Design](#node-system-design)
6. [Data Flow Diagrams](#data-flow-diagrams)
7. [API Contracts](#api-contracts)
8. [Testing Strategy](#testing-strategy)

---

## Overview

### Scope

This GUI provides a **visual configuration editor** for hydrogen plant simulations. It does **NOT**:
- Execute simulations directly
- Modify simulation internals
- Store proprietary data formats
- Include real-time monitoring (that's Phase 2)

It **DOES**:
- Let users graphically compose plant configurations
- Export valid YAML that your PlantBuilder understands
- Validate configurations against your JSON schema
- Provide async simulation runner (separate from GUI thread)
- Display results in charts

### Design Principles

1. **Non-Invasive**: Zero changes to existing simulation core
2. **Schema-Driven**: All validation rules come from `plant_schema_v1.json`
3. **Async-First**: Simulations run in background threads
4. **User-Friendly**: Minimalist aesthetic (Antigravity/VS Code style)
5. **Maintainable**: Every node type follows identical pattern

---

## Architecture Pattern

### The "Graph-to-Config" Pipeline

```python
# User creates visual graph in GUI
graph = NodeGraph()
graph.create_node("ElectrolyzerNode", "Electrolyzer_1")
graph.create_node("TankNode", "Tank_LP_1")
graph.add_connection("Electrolyzer_1", "h2_out", "Tank_LP_1", "h2_in", flow_type="hydrogen")

# GUI exports to adapter
from h2_plant.gui.core.graph_adapter import graph_to_config_dict
config_dict = graph_to_config_dict(
    nodes=graph.all_nodes(),
    edges=graph.all_connections()
)

# Adapter produces PlantConfig-compatible dict
config_dict = {
    "name": "GUI Generated Plant",
    "production": {
        "electrolyzer": {
            "enabled": True,
            "max_power_mw": 5.0,
            "base_efficiency": 0.68
        }
    },
    "storage": {
        "lp_tanks": {
            "count": 1,
            "capacity_kg": 50.0,
            "pressure_bar": 30.0
        }
    },
    "simulation": {...}
}

# PlantBuilder consumes dict (NEW from_dict method)
from h2_plant.config.plant_builder import PlantBuilder
plant = PlantBuilder.from_dict(config_dict)

# Simulation engine receives fully instantiated registry
from h2_plant.simulation.runner import run_simulation_from_registry
results = run_simulation_from_registry(plant.registry, config_dict)
```

---

## Module Specifications

### 1. Core Adapter Layer

**File:** `h2_plant/gui/core/graph_adapter.py`

```python
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
        if source_node_id not in nodes:
            raise ValueError(f"Source node {source_node_id} not found")
        if target_node_id not in nodes:
            raise ValueError(f"Target node {target_node_id} not found")
        
        src_node = nodes[source_node_id]
        tgt_node = nodes[target_node_id]
        
        src_port = next((p for p in src_node.ports if p.name == source_port), None)
        tgt_port = next((p for p in tgt_node.ports if p.name == target_port), None)
        
        if not src_port or not tgt_port:
            raise ValueError(f"Port not found")
        
        if src_port.flow_type != tgt_port.flow_type:
            raise ValueError(
                f"Flow type mismatch: {src_port.flow_type} → {tgt_port.flow_type}"
            )

class GraphToConfigAdapter:
    """Main conversion engine."""
    
    # Mapping from visual node types to config sections
    NODE_TYPE_MAPPING = {
        "ElectrolyzerNode": ("production", "electrolyzer"),
        "ATRSourceNode": ("production", "atr"),
        "LPTankNode": ("storage", "lp_tanks"),
        "HPTankNode": ("storage", "hp_tanks"),
        "OxygenBufferNode": ("storage", "oxygen_buffer"),
        "FillingCompressorNode": ("compression", "filling_compressor"),
        "OutgoingCompressorNode": ("compression", "outgoing_compressor"),
        "DemandSchedulerNode": ("demand",),
        "EnergyPriceNode": ("energy_price",),
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
            "storage": {},
            "compression": {},
            "simulation": {
                "timestep_hours": 1.0,
                "duration_hours": 8760,
                "checkpoint_interval_hours": 168,
            }
        }
        
        # Process each node and extract its configuration
        for node_id, node in self.nodes.items():
            section, *subsection = self.NODE_TYPE_MAPPING.get(node.type, (None,))
            
            if section is None:
                self.errors.append(f"Unknown node type: {node.type}")
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
        prod_nodes = [n for n in self.nodes.values() if n.type in ["ElectrolyzerNode", "ATRSourceNode"]]
        
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
                 # Note: You would also need to populate 'isolated_config' structure here
                 # This is where the "Compiler" complexity lives
            else:
                 config["storage"]["source_isolated"] = False
        else:
             config["storage"]["source_isolated"] = False

    def _extract_node_config(self, node: GraphNode) -> Dict[str, Any]:
        """Extract properties from a node, adding metadata."""
        return {
            "enabled": True,
            **node.properties
        }
    
    def _infer_plant_name(self) -> str:
        """Generate a name based on node count and types."""
        prod_count = sum(1 for n in self.nodes.values() 
                         if n.type in ["ElectrolyzerNode", "ATRSourceNode"])
        stor_count = sum(1 for n in self.nodes.values() 
                         if "Tank" in n.type)
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
```

---

### 1.1 Project Persistence (Saving the GUI State)

The `to_config_dict()` method exports a simulation configuration (YAML), but this **loses visual information** (node positions, colors, comments). We need a separate "Project Save" format.

**File:** `h2_plant/gui/core/project_serializer.py`

```python
"""
Handles saving/loading of the GUI project state (.h2gui files).
This preserves visual layout, which the simulation config does not.
"""

import json
from typing import Dict, Any

class ProjectSerializer:
    @staticmethod
    def serialize(graph_adapter: 'GraphToConfigAdapter') -> str:
        """Save full project state to JSON string."""
        project = {
            "version": "1.0",
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "x": n.x,
                    "y": n.y,
                    "properties": n.properties,
                    "custom_name": n.display_name
                }
                for n in graph_adapter.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_node_id,
                    "src_port": e.source_port,
                    "target": e.target_node_id,
                    "tgt_port": e.target_port
                }
                for e in graph_adapter.edges
            ]
        }
        return json.dumps(project, indent=2)

    @staticmethod
    def deserialize(json_str: str, graph_adapter: 'GraphToConfigAdapter') -> None:
        """Load project state from JSON string."""
        data = json.loads(json_str)
        # Logic to clear current graph and recreate nodes/edges...
```

---

### 2. Schema Inspector

**File:** `h2_plant/gui/core/schema_inspector.py`

```python
"""
Dynamic schema inspection for widget generation.

Reads plant_schema_v1.json and exposes validation rules to GUI.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

class SchemaInspector:
    """
    Provides validation constraints from JSON schema.
    
    The GUI uses this to generate widgets with proper constraints
    (min/max, enums, type conversions, etc.).
    """
    
    def __init__(self, schema_path: Optional[Path] = None):
        if schema_path is None:
            # Default to h2_plant's bundled schema
            schema_path = (
                Path(__file__).parent.parent.parent / 
                "config" / "schemas" / "plant_schema_v1.json"
            )
        
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
    
    def get_node_schema(self, node_type: str) -> Dict[str, Any]:
        """
        Get schema for a specific node type.
        
        Example:
            schema = inspector.get_node_schema("ElectrolyzerNode")
            # Returns schema for production.electrolyzer properties
        """
        mapping = {
            "ElectrolyzerNode": ["production", "properties", "electrolyzer"],
            "ATRSourceNode": ["production", "properties", "atr"],
            "LPTankNode": ["storage", "properties", "lp_tanks"],
            "HPTankNode": ["storage", "properties", "hp_tanks"],
        }
        
        path = mapping.get(node_type)
        if not path:
            return {}
        
        current = self.schema
        for part in path:
            current = current.get(part, {})
        
        return current
    
    def get_property_validator(self, node_type: str, property_name: str) -> Dict[str, Any]:
        """
        Get validation rules for a specific property.
        
        Returns a dict with:
            - type: "number", "integer", "string", "boolean"
            - minimum, maximum
            - enum (if applicable)
            - pattern (regex for strings)
            - description
        """
        node_schema = self.get_node_schema(node_type)
        properties = node_schema.get("properties", {})
        return properties.get(property_name, {})
    
    def list_required_properties(self, node_type: str) -> List[str]:
        """Get list of required properties for a node type."""
        node_schema = self.get_node_schema(node_type)
        return node_schema.get("required", [])
    
    def get_enum_values(self, node_type: str, property_name: str) -> Tuple[str, ...]:
        """Get possible values for an enum property."""
        validator = self.get_property_validator(node_type, property_name)
        return tuple(validator.get("enum", []))
    
    def validate_property(self, 
                         node_type: str, 
                         property_name: str, 
                         value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a property value against schema.
        
        Returns (is_valid, error_message)
        """
        validator = self.get_property_validator(node_type, property_name)
        
        if not validator:
            return False, f"Unknown property: {property_name}"
        
        prop_type = validator.get("type")
        
        # Type validation
        if prop_type == "number":
            if not isinstance(value, (int, float)):
                return False, f"Expected number, got {type(value).__name__}"
            
            minimum = validator.get("minimum")
            if minimum is not None and value < minimum:
                return False, f"Value {value} < minimum {minimum}"
            
            exclusive_min = validator.get("exclusiveMinimum")
            if exclusive_min and value <= exclusive_min:
                return False, f"Value {value} ≤ {exclusive_min}"
            
            maximum = validator.get("maximum")
            if maximum is not None and value > maximum:
                return False, f"Value {value} > maximum {maximum}"
        
        elif prop_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Expected integer, got {type(value).__name__}"
            
            minimum = validator.get("minimum")
            if minimum is not None and value < minimum:
                return False, f"Value {value} < minimum {minimum}"
        
        elif prop_type == "string":
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"
        
        # Enum validation
        enum_values = validator.get("enum")
        if enum_values and value not in enum_values:
            return False, f"Value '{value}' not in {enum_values}"
        
        return True, None
```

---

### 3. Node Base Class

**File:** `h2_plant/gui/nodes/base_node.py`

```python
"""
Abstract base class for all node types.

Every specific node (ElectrolyzerNode, TankNode, etc.) inherits from this.
"""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from nodegraphqt import BaseNode as QtBaseNode

class ConfigurableNode(QtBaseNode, ABC):
    """
    Base class for all plant component nodes.
    
    Subclasses must define:
    - __identifier__: Unique identifier (e.g., 'h2_plant.production.electrolyzer')
    - NODE_NAME: Display name (e.g., 'Electrolyzer')
    - _init_ports(): Define input/output ports
    - _init_properties(): Define parameter fields
    """
    
    # Subclasses must override these
    __identifier__ = 'h2_plant.base'
    NODE_NAME = 'BaseNode'
    ICON_PATH = None  # Optional icon
    
    PORT_COLORS = {
        "hydrogen": (0, 255, 255),      # Cyan
        "oxygen": (255, 200, 0),        # Orange
        "electricity": (255, 255, 0),   # Yellow
        "heat": (255, 100, 100),        # Red
        "water": (100, 150, 255),       # Blue
        "compressed_h2": (0, 200, 255), # Light cyan
    }
    
    def __init__(self):
        super().__init__()
        self.set_name(self.NODE_NAME)
        
        # Track property changes
        self._property_validators: Dict[str, Callable] = {}
        self._on_property_changed_callbacks: List[Callable] = []
        
        # Initialize ports and properties
        self._init_ports()
        self._init_properties()
    
    @abstractmethod
    def _init_ports(self) -> None:
        """
        Define input/output ports.
        
        Example (in ElectrolyzerNode):
            self.add_input('power_mw', 'Electricity')
            self.add_output('h2_kg_h', 'Hydrogen')
        """
        pass
    
    @abstractmethod
    def _init_properties(self) -> None:
        """
        Define configurable properties.
        
        Example:
            self.add_text_input('name', 'Electrolyzer_1')
            self.add_float_input('max_power_mw', 5.0)
            self.add_float_input('efficiency', 0.68)
        """
        pass
    
    def add_input(self, 
                  port_name: str, 
                  flow_type: str,
                  multi_connection: bool = False) -> None:
        """Add an input port."""
        color = self.PORT_COLORS.get(flow_type, (128, 128, 128))
        self.add_input(
            name=port_name,
            multi_connection=multi_connection,
            color=color
        )
        # Store metadata
        port = self.inputs[port_name]
        port.flow_type = flow_type
    
    def add_output(self, 
                   port_name: str, 
                   flow_type: str,
                   multi_connection: bool = True) -> None:
        """Add an output port."""
        color = self.PORT_COLORS.get(flow_type, (128, 128, 128))
        self.add_output(
            name=port_name,
            multi_connection=multi_connection,
            color=color
        )
        # Store metadata
        port = self.outputs[port_name]
        port.flow_type = flow_type
    
    def add_float_input(self, 
                        name: str, 
                        default: float = 0.0,
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None) -> None:
        """Add a float property with optional constraints."""
        # NodeGraphQt syntax
        self.add_input(name=name, value=default)
        
        # Store constraints for validation
        self._property_validators[name] = (
            lambda v: self._validate_float(v, min_val, max_val)
        )
    
    def add_integer_input(self, 
                         name: str, 
                         default: int = 0,
                         min_val: Optional[int] = None,
                         max_val: Optional[int] = None) -> None:
        """Add an integer property."""
        self.add_input(name=name, value=default)
        self._property_validators[name] = (
            lambda v: self._validate_integer(v, min_val, max_val)
        )
    
    def add_enum_input(self, 
                      name: str, 
                      options: List[str],
                      default_index: int = 0) -> None:
        """Add an enum (dropdown) property."""
        self.add_input(name=name, value=options[default_index])
        # TODO: Use QComboBox in property panel
    
    def get_properties(self) -> Dict[str, Any]:
        """Export node properties as a dict."""
        properties = {}
        for input_port in self.inputs.values():
            if not input_port.connected_ports:  # Only export unconnected
                properties[input_port.name] = input_port.value
        return properties
    
    def set_properties(self, properties: Dict[str, Any]) -> None:
        """Import node properties from dict."""
        for name, value in properties.items():
            if name in self.inputs:
                # Validate
                if name in self._property_validators:
                    is_valid, error = self._property_validators[name](value)
                    if not is_valid:
                        raise ValueError(f"{name}: {error}")
                
                self.inputs[name].set_value(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dict (for save/export)."""
        return {
            "id": self.id,
            "type": self.__identifier__,
            "display_name": self.name,
            "x": self.x_pos,
            "y": self.y_pos,
            "properties": self.get_properties(),
        }
    
    @staticmethod
    def _validate_float(value: Any, 
                       min_val: Optional[float], 
                       max_val: Optional[float]) -> Tuple[bool, Optional[str]]:
        try:
            fval = float(value)
            if min_val is not None and fval < min_val:
                return False, f"Must be ≥ {min_val}"
            if max_val is not None and fval > max_val:
                return False, f"Must be ≤ {max_val}"
            return True, None
        except (ValueError, TypeError):
            return False, "Must be a number"
    
    @staticmethod
    def _validate_integer(value: Any, 
                         min_val: Optional[int], 
                         max_val: Optional[int]) -> Tuple[bool, Optional[str]]:
        try:
            ival = int(value)
            if min_val is not None and ival < min_val:
                return False, f"Must be ≥ {min_val}"
            if max_val is not None and ival > max_val:
                return False, f"Must be ≤ {max_val}"
            return True, None
        except (ValueError, TypeError):
            return False, "Must be an integer"
```

---

### 4. Concrete Node Implementations

**File:** `h2_plant/gui/nodes/production_nodes.py`

```python
"""Concrete node implementations for production sources."""

from h2_plant.gui.nodes.base_node import ConfigurableNode

class ElectrolyzerNode(ConfigurableNode):
    __identifier__ = 'h2_plant.production.electrolyzer'
    NODE_NAME = 'Electrolyzer'
    
    def _init_ports(self) -> None:
        """Electrolyzer: Input power, output H2."""
        self.add_input('power_mw', 'electricity')
        self.add_output('h2_kg_h', 'hydrogen')
        self.add_output('o2_kg_h', 'oxygen')
        self.add_output('heat_kw', 'heat')
    
    def _init_properties(self) -> None:
        """Electrolyzer configuration."""
        self.add_text_input('name', 'Electrolyzer_1')
        self.add_float_input('max_power_mw', 5.0, min_val=0.1, max_val=1000.0)
        self.add_float_input('efficiency', 0.68, min_val=0.001, max_val=0.999)
        self.add_float_input('min_load_factor', 0.15, min_val=0.0, max_val=1.0)

class ATRSourceNode(ConfigurableNode):
    __identifier__ = 'h2_plant.production.atr'
    NODE_NAME = 'ATR'
    
    def _init_ports(self) -> None:
        self.add_input('ng_kg_h', 'gas')
        self.add_input('heat_kw', 'heat')
        self.add_output('h2_kg_h', 'hydrogen')
        self.add_output('co2_kg_h', 'gas')
        self.add_output('power_mw', 'electricity')
    
    def _init_properties(self) -> None:
        self.add_text_input('name', 'ATR_1')
        self.add_float_input('max_ng_flow_kg_h', 100.0, min_val=1.0, max_val=10000.0)
        self.add_float_input('efficiency', 0.75, min_val=0.001, max_val=0.999)
```

---

### 5. Simulation Worker (Async Execution)

**File:** `h2_plant/gui/core/simulation_worker.py`

```python
"""
Background simulation runner using Python threading.

Prevents the GUI from freezing during 30-90 second simulations.
"""

import threading
import traceback
from typing import Optional, Callable, Dict, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SimulationProgress:
    """Progress update from simulation."""
    current_hour: float
    total_hours: float
    percentage: float
    status: str  # "initializing", "running", "finalizing", "complete"
    message: str = ""

@dataclass
class SimulationResult:
    """Results from completed simulation."""
    success: bool
    results_path: Path
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class SimulationWorker(threading.Thread):
    """
    Run simulation in background thread.
    
    Usage:
        worker = SimulationWorker(config_dict)
        worker.on_progress(lambda p: update_progress_bar(p.percentage))
        worker.on_complete(lambda r: show_results(r.results_path))
        worker.start()
    """
    
    def __init__(self, config_dict: Dict[str, Any], output_dir: Path = None):
        super().__init__(daemon=False)
        self.config_dict = config_dict
        self.output_dir = output_dir or Path("./simulation_output")
        
        # Callbacks
        self._progress_callbacks = []
        self._complete_callbacks = []
        self._error_callbacks = []
        
        # State
        self._result: Optional[SimulationResult] = None
        self._stop_event = threading.Event()
    
    def on_progress(self, callback: Callable[[SimulationProgress], None]) -> None:
        """Register progress callback."""
        self._progress_callbacks.append(callback)
    
    def on_complete(self, callback: Callable[[SimulationResult], None]) -> None:
        """Register completion callback."""
        self._complete_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[str], None]) -> None:
        """Register error callback."""
        self._error_callbacks.append(callback)
    
    def run(self) -> None:
        """Main thread execution."""
        try:
            # Step 1: Build plant from config
            self._emit_progress(0, 0, "initializing", "Building plant...")
            from h2_plant.config.plant_builder import PlantBuilder
            
            plant = PlantBuilder.from_dict(self.config_dict)
            
            # Step 2: Create and run simulation
            self._emit_progress(0, self.config_dict['simulation']['duration_hours'], 
                               "running", "Starting simulation...")
            
            from h2_plant.simulation.engine import SimulationEngine
            engine = SimulationEngine(plant.registry, self.config_dict)
            
            # Hook into engine progress (requires adding progress signals to SimulationEngine)
            # For now, we'll simulate progress updates
            engine.run()
            
            # Step 3: Collect results
            self._emit_progress(
                self.config_dict['simulation']['duration_hours'],
                self.config_dict['simulation']['duration_hours'],
                "finalizing", "Saving results..."
            )
            
            # Read results from output
            results_file = self.output_dir / "simulation_results.json"
            
            self._result = SimulationResult(
                success=True,
                results_path=self.output_dir,
                metrics={
                    "h2_produced": 0.0,  # Parse from results_file
                    "energy_consumed": 0.0,
                }
            )
            
            self._emit_progress(
                self.config_dict['simulation']['duration_hours'],
                self.config_dict['simulation']['duration_hours'],
                "complete", "Simulation complete"
            )
            
            # Emit completion
            for callback in self._complete_callbacks:
                callback(self._result)
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._result = SimulationResult(success=False, error=error_msg, results_path=None)
            
            for callback in self._error_callbacks:
                callback(error_msg)
    
    def _emit_progress(self, current_hour: float, total_hours: float,
                      status: str, message: str) -> None:
        """Send progress update to all listeners."""
        progress = SimulationProgress(
            current_hour=current_hour,
            total_hours=total_hours,
            percentage=100.0 * current_hour / total_hours if total_hours > 0 else 0.0,
            status=status,
            message=message,
        )
        
        for callback in self._progress_callbacks:
            callback(progress)
    
    def stop(self) -> None:
        """Gracefully stop the simulation."""
        self._stop_event.set()
    
    def get_result(self) -> Optional[SimulationResult]:
        """Retrieve result after completion."""
        return self._result
```

---

## Backend Modifications

### Modification 1: Add `PlantBuilder.from_dict()`

**File:** `h2_plant/config/plant_builder.py`

```python
class PlantBuilder:
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PlantBuilder':
        """
        Build plant from Python dictionary.
        
        This is called by the GUI after converting the visual graph.
        """
        import dataclasses
        
        # Convert dict to dataclass
        config = cls._dict_to_config(config_dict)
        config.validate()
        
        # Same as from_file()
        builder = cls(config)
        builder.build()
        return builder
    
    @staticmethod
    def _dict_to_config(data: Dict[str, Any]) -> 'PlantConfig':
        """Convert dict to PlantConfig dataclass recursively."""
        # Implementation depends on your dataclass structure
        # Typically uses dataclasses.asdict() in reverse
        return PlantConfig(**{
            'name': data.get('name', 'Unnamed Plant'),
            'production': data.get('production', {}),
            'storage': data.get('storage', {}),
            'compression': data.get('compression', {}),
            'demand': data.get('demand', {}),
            'energy_price': data.get('energy_price', {}),
            'simulation': data.get('simulation', {}),
            # ... other fields
        })
```

---

### Modification 2: Export Utilities

**File:** `h2_plant/config/serializers.py` (NEW or ADD TO EXISTING)

```python
"""Configuration serialization utilities."""

import json
import yaml
from typing import Dict, Any

def dict_to_yaml(config_dict: Dict[str, Any]) -> str:
    """Convert config dict to YAML string."""
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

def dict_to_json(config_dict: Dict[str, Any]) -> str:
    """Convert config dict to JSON string."""
    return json.dumps(config_dict, indent=2)
```

---

## Node System Design

### Port System

Each node has typed input/output ports:

```python
class ElectrolyzerNode(ConfigurableNode):
    def _init_ports(self):
        # Input: Electricity (can be connected from grid or battery)
        self.add_input('power_mw', 'electricity')
        
        # Outputs: Multiple products
        self.add_output('h2_kg_h', 'hydrogen')     # Main product
        self.add_output('o2_kg_h', 'oxygen')       # Byproduct
        self.add_output('heat_kw', 'heat')         # Waste heat
```

**Port Color Coding:**
- 🔵 Cyan: Hydrogen (primary output)
- 🟠 Orange: Oxygen (byproduct)
- 🟡 Yellow: Electricity (power)
- 🔴 Red: Heat (thermal energy)
- 🔵 Light Blue: Water

---

## Data Flow Diagrams

### Complete GUI → Simulation Flow

```
┌─────────────────────────────────────────┐
│ 1. User creates nodes visually          │
│    - Drag "Electrolyzer" from palette   │
│    - Drag "Tank" from palette           │
│    - Set properties (5 MW, 0.68 eff.)   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. User connects nodes                  │
│    - Draw line: Electrolyzer → Tank     │
│    - Port types validated (cyan → cyan) │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. GUI validates configuration          │
│    - graph_adapter.validate()           │
│    - Check: ≥1 production, ≥1 storage   │
│    - Check: All properties set          │
└──────────────┬──────────────────────────┘
               │
               ▼ (Export → Run)
┌─────────────────────────────────────────┐
│ 4. GUI exports to dict                  │
│    - graph_to_config_dict()             │
│    - Returns PlantConfig-compatible     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 5. SimulationWorker spawned (thread)    │
│    - PlantBuilder.from_dict() called    │
│    - Components instantiated            │
│    - SimulationEngine.run() starts      │
└──────────────┬──────────────────────────┘
               │
      (background, async)
               │
               ▼
┌─────────────────────────────────────────┐
│ 6. Progress updates emitted             │
│    - Hour 0/8760 (0%)                   │
│    - Hour 1000/8760 (11%)               │
│    - ...                                │
│    - Hour 8760/8760 (100%)              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 7. Results saved to disk                │
│    - simulation_output/                 │
│    - Results JSON loaded into GUI       │
│    - Charts rendered (matplotlib)       │
└─────────────────────────────────────────┘
```

---

## API Contracts

### Graph Adapter API

```python
# Input: Visual graph structure
adapter = GraphToConfigAdapter()
adapter.add_node(GraphNode(
    id="elec_1",
    type="ElectrolyzerNode",
    display_name="Electrolyzer 1",
    x=100, y=200,
    properties={"max_power_mw": 5.0, "efficiency": 0.68},
    ports=[...]
))

adapter.add_edge(GraphEdge(
    source_node_id="elec_1",
    source_port="h2_kg_h",
    target_node_id="tank_1",
    target_port="h2_in",
    flow_type=FlowType.HYDROGEN
))

# Output: Valid configuration
is_valid, errors = adapter.validate()
if is_valid:
    config_dict = adapter.to_config_dict()
    plant = PlantBuilder.from_dict(config_dict)
else:
    print(f"Configuration errors: {errors}")
```

---

## Testing Strategy

### Unit Tests

```python
# tests/gui/test_graph_adapter.py
def test_electrolyzer_node_export():
    node = GraphNode(
        id="e1",
        type="ElectrolyzerNode",
        properties={"max_power_mw": 5.0, "efficiency": 0.68}
    )
    adapter = GraphToConfigAdapter()
    adapter.add_node(node)
    
    config = adapter.to_config_dict()
    assert config["production"]["electrolyzer"]["max_power_mw"] == 5.0

def test_invalid_port_connection():
    # Hydrogen output to electricity input
    edge = GraphEdge(
        source_node_id="e1",
        source_port="h2_out",
        target_node_id="c1",
        target_port="power_in",
        flow_type=FlowType.HYDROGEN
    )
    
    with pytest.raises(ValueError):
        edge.validate(nodes)
```

### Integration Tests

```python
# tests/gui/test_end_to_end.py
def test_gui_to_simulation():
    # Create graph
    adapter = GraphToConfigAdapter()
    # ... add nodes and edges ...
    
    # Export and build
    config = adapter.to_config_dict()
    plant = PlantBuilder.from_dict(config)
    
    # Simulate
    registry = plant.registry
    assert registry.has("electrolyzer")
    assert registry.has("lp_tank_array")
```

---

**Status:** Ready for implementation  
**Next Step:** Begin Day 1 canvas setup with PySide6
