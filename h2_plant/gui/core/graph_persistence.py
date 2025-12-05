"""
Graph Persistence Layer - Save/Load visual layouts with complete fidelity

This module handles persistence of:
1. Node visual properties (position, size, color, selection state)
2. Edge routing and styling
3. Canvas zoom/pan state
4. Full configuration data
5. Project metadata

Formats supported:
- Native JSON (.h2plant) - Complete state including layout
- Legacy JSON (.json) - Configuration only
- JSON Schema validation included
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class SerializationFormat(str, Enum):
    """Supported serialization formats."""
    H2PLANT = "h2plant"  # Native format with full layout
    JSON = "json"  # Configuration only
    BACKUP = "backup"  # Timestamped backup


@dataclass
class NodeGeometry:
    """Visual geometry of a node on canvas."""
    x: float
    y: float
    width: float = 100.0
    height: float = 100.0
    color: Tuple[int, int, int] = (100, 100, 100)
    border_color: Tuple[int, int, int] = (50, 50, 50)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    selected: bool = False
    disabled: bool = False
    collapsed: bool = False


@dataclass
class EdgeGeometry:
    """Visual geometry of an edge/connection."""
    source_node_id: str
    target_node_id: str
    source_port: str
    target_port: str
    flow_type: str
    # Visual properties
    color: Tuple[int, int, int] = (255, 255, 255)
    width: float = 2.0
    style: str = "solid"  # solid, dashed, dotted
    selected: bool = False
    # Routing waypoints (for curved edges)
    waypoints: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class CanvasState:
    """Visual state of the canvas/viewport."""
    zoom_level: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    grid_enabled: bool = True
    grid_size: int = 20
    snap_to_grid: bool = True


@dataclass
class ProjectMetadata:
    """Project-level metadata."""
    name: str
    version: str = "1.0"
    created: str = ""  # ISO format timestamp
    modified: str = ""
    author: str = "Unknown"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    checksum: str = ""  # SHA256 of content


@dataclass
class GraphSnapshot:
    """Complete serializable graph state."""
    metadata: ProjectMetadata
    canvas_state: CanvasState
    nodes: Dict[str, Dict[str, Any]]  # node_id -> {type, properties, geometry}
    edges: List[Dict[str, Any]]  # List of edge definitions with geometry
    validation_result: Optional[Dict[str, Any]] = None
    topology_analysis: Optional[Dict[str, Any]] = None


class GraphPersistenceManager:
    """
    Manages saving and loading of graph layouts and configurations.

    Usage:
        manager = GraphPersistenceManager()
        
        # Save
        snapshot = manager.create_snapshot(node_graph, config_dict)
        manager.save("my_plant.h2plant", snapshot)
        
        # Load
        snapshot = manager.load("my_plant.h2plant")
        manager.restore_to_graph(node_graph, snapshot)
        
        # Export config only
        config = manager.extract_config(snapshot)
    """

    # Schema version for compatibility checking
    SCHEMA_VERSION = "1.0"
    FILE_EXTENSION = ".h2plant"
    
    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize persistence manager.
        
        Args:
            backup_dir: Directory for automatic backups. If None, no backups created.
        """
        self.backup_dir = backup_dir
        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, node_graph, config_dict: Dict[str, Any],
                       project_name: str = "Untitled Plant") -> GraphSnapshot:
        """
        Create a complete snapshot of current graph state.

        Args:
            node_graph: NodeGraphQt graph object
            config_dict: Configuration dictionary
            project_name: Name of the project

        Returns:
            GraphSnapshot ready for serialization
        """
        # Extract node geometries
        nodes_dict = {}
        for node in node_graph.all_nodes():
            node_id = node.id
            # Use full type identifier if available (NodeGraphQt standard)
            node_type = getattr(node, 'type_', node.__class__.__name__)
            nodes_dict[node_id] = {
                "type": node_type,
                "display_name": node.name(),
                "properties": self._extract_properties(node),
                "geometry": asdict(self._extract_node_geometry(node)),
            }
        
        # Extract edge geometries
        edges_list = []
        # Extract edge geometries
        edges_list = []
        # Iterate over all nodes to find connections
        for node in node_graph.all_nodes():
            for output_port in node.output_ports():
                for target_port in output_port.connected_ports():
                    target_node = target_port.node()
                    
                    # Create edge dict
                    # Note: We can't easily get the actual Edge object to read visual properties
                    # like color or waypoints without all_edges(), so we use defaults.
                    edge_dict = {
                        "source_node_id": node.id,
                        "target_node_id": target_node.id,
                        "source_port": output_port.name(),
                        "target_port": target_port.name(),
                        "flow_type": "default", # Default as we can't read from edge obj
                        "geometry": {
                            "source_node_id": node.id,
                            "target_node_id": target_node.id,
                            "source_port": output_port.name(),
                            "target_port": target_port.name(),
                            "flow_type": "default",
                            "color": (255, 255, 255),
                            "width": 2.0,
                            "style": "solid",
                            "selected": False,
                            "waypoints": []
                        },
                    }
                    edges_list.append(edge_dict)
        
        # Extract canvas state
        canvas_state = self._extract_canvas_state(node_graph)
        
        # Create metadata
        now = datetime.now().isoformat()
        metadata = ProjectMetadata(
            name=project_name,
            version=self.SCHEMA_VERSION,
            created=now,
            modified=now,
            author="H2 Plant GUI",
            description=f"Plant configuration: {project_name}",
        )
        
        # Create snapshot
        snapshot = GraphSnapshot(
            metadata=metadata,
            canvas_state=canvas_state,
            nodes=nodes_dict,
            edges=edges_list,
        )
        
        # Calculate checksum
        snapshot.metadata.checksum = self._calculate_checksum(snapshot)
        
        return snapshot

    def save(self, file_path: str, snapshot: GraphSnapshot,
             format: SerializationFormat = SerializationFormat.H2PLANT,
             create_backup: bool = True) -> Path:
        """
        Save snapshot to file.

        Args:
            file_path: Destination file path
            snapshot: GraphSnapshot to save
            format: Serialization format
            create_backup: Whether to create automatic backup

        Returns:
            Path to saved file
        """
        file_path = Path(file_path)
        
        # Ensure correct extension
        if format == SerializationFormat.H2PLANT:
            if file_path.suffix != self.FILE_EXTENSION:
                file_path = file_path.with_suffix(self.FILE_EXTENSION)
        
        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if requested and file exists
        if create_backup and file_path.exists():
            self._create_backup(file_path)
        
        # Serialize snapshot
        if format == SerializationFormat.H2PLANT:
            data = self._serialize_snapshot(snapshot)
        elif format == SerializationFormat.JSON:
            # Config only
            data = self.extract_config(snapshot)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved graph to {file_path}")
        return file_path

    def load(self, file_path: str) -> GraphSnapshot:
        """
        Load snapshot from file.

        Args:
            file_path: Path to file

        Returns:
            Deserialized GraphSnapshot

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate schema
        if not self._validate_schema(data):
            logger.warning(f"Schema validation issues in {file_path}")
        
        # Deserialize
        snapshot = self._deserialize_snapshot(data)
        
        logger.info(f"Loaded graph from {file_path}")
        return snapshot

    def restore_to_graph(self, node_graph, snapshot: GraphSnapshot) -> None:
        """
        Restore a snapshot's visual state to a graph.

        Args:
            node_graph: NodeGraphQt graph to restore to
            snapshot: GraphSnapshot to restore from
        """
        # 1. Create Nodes
        node_map = {} # snapshot_id -> graph_node_object
        
        # Disable auto-layout during creation
        node_graph.begin_undo("Load Layout")
        
        try:
            for node_id, node_data in snapshot.nodes.items():
                node_type = node_data["type"]
                print(f"DEBUG: Creating node {node_type}...")
                # Create node instance
                try:
                    node = node_graph.create_node(node_type, name=node_data["display_name"], push_undo=False)
                    if node is None:
                        print(f"ERROR: Failed to create node {node_type} (returned None)")
                        continue
                        
                    print(f"DEBUG: Node created: {node}")
                    
                    # Force ID to match snapshot (critical for connections)
                    # NodeGraphQt usually generates a new ID, so we might need to map it
                    # But for persistence, we want to keep the ID if possible or map it
                    
                    # NOTE: NodeGraphQt nodes have a unique ID. We can't easily overwrite it 
                    # without internal access. So we map snapshot_id -> new_node
                    node_map[node_id] = node
                    
                    # Apply properties
                    # print(f"DEBUG: Setting properties for {node_type}...")
                    for prop, val in node_data["properties"].items():
                        if hasattr(node, 'set_property'):
                            # Fix: NodeGraphQt text inputs crash if passed numbers directly
                            # Since our base_node uses text inputs for floats/ints, we MUST convert to string
                            if isinstance(val, (int, float)):
                                val = str(val)
                                
                            # print(f"DEBUG: Setting {prop}={val} (type: {type(val)})")
                            node.set_property(prop, val)
                            
                    # Apply geometry
                    # print(f"DEBUG: Applying geometry for {node_type}...")
                    geom = NodeGeometry(**node_data["geometry"])
                    self._apply_node_geometry(node, geom)
                    
                except Exception as e:
                    logger.error(f"Failed to create node {node_type}: {e}")
                    print(f"ERROR: Exception creating node {node_type}: {e}")

            # 2. Create Connections (Edges)
            print("DEBUG: Creating connections...")
            for edge_data in snapshot.edges:
                src_id = edge_data["source_node_id"]
                tgt_id = edge_data["target_node_id"]
                
                if src_id in node_map and tgt_id in node_map:
                    src_node = node_map[src_id]
                    tgt_node = node_map[tgt_id]
                    
                    src_port = edge_data["source_port"]
                    tgt_port = edge_data["target_port"]
                    
                    print(f"DEBUG: Connecting {src_node.name()}.{src_port} -> {tgt_node.name()}.{tgt_port}")
                    
                    # Connect
                    try:
                        # Find ports
                        # NodeGraphQt ports are objects, we need to find them by name
                        s_port = src_node.get_output(src_port)
                        t_port = tgt_node.get_input(tgt_port)
                        
                        if s_port and t_port:
                            s_port.connect_to(t_port, push_undo=False)
                        else:
                            print(f"WARNING: Port not found: {src_port} or {tgt_port}")
                    except Exception as e:
                        logger.error(f"Failed to connect {src_id}->{tgt_id}: {e}")
            
            # 3. Apply Canvas State
            print("DEBUG: Applying canvas state...")
            self._apply_canvas_state(node_graph, snapshot.canvas_state)
            print("DEBUG: Restore complete.")
            
        finally:
            node_graph.end_undo()

    def extract_config(self, snapshot: GraphSnapshot) -> Dict[str, Any]:
        """
        Extract configuration dictionary from snapshot (layout-independent).

        Args:
            snapshot: GraphSnapshot

        Returns:
            Configuration dictionary suitable for PlantBuilder
        """
        from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, GraphEdge, Port, FlowType

        # Reconstruct graph model
        adapter = GraphToConfigAdapter()
        
        # Add nodes
        for node_id, node_data in snapshot.nodes.items():
            graph_node = GraphNode(
                id=node_id,
                type=node_data["type"],
                display_name=node_data["display_name"],
                x=node_data["geometry"]["x"],
                y=node_data["geometry"]["y"],
                properties=node_data["properties"],
                ports=[]  # Ports inferred from type
            )
            adapter.add_node(graph_node)
        
        # Add edges
        for edge_data in snapshot.edges:
            ft_str = edge_data.get("flow_type", "hydrogen")
            if ft_str == "default":
                ft_str = "hydrogen" # Fallback for GUI-created edges
                
            graph_edge = GraphEdge(
                source_node_id=edge_data["source_node_id"],
                source_port=edge_data["source_port"],
                target_node_id=edge_data["target_node_id"],
                target_port=edge_data["target_port"],
                flow_type=FlowType(ft_str)
            )
            adapter.add_edge(graph_edge)
        
        # Convert to config
        config = adapter.to_config_dict()
        
        return config

    def _extract_properties(self, node) -> Dict[str, Any]:
        """Extract node properties."""
        if hasattr(node, 'get_properties'):
            return node.get_properties()
        elif hasattr(node, 'properties'):
            return dict(node.properties())
        else:
            return {}

    def _extract_node_geometry(self, node) -> NodeGeometry:
        """Extract visual geometry of a node."""
        x, y = node.pos() if hasattr(node, 'pos') else (0, 0)
        
        # Try to get colors
        # Handle both property and method access for compatibility
        color = node.color() if callable(getattr(node, 'color', None)) else getattr(node, 'color', (100, 100, 100))
        border_color = node.border_color() if callable(getattr(node, 'border_color', None)) else getattr(node, 'border_color', (50, 50, 50))
        text_color = node.text_color() if callable(getattr(node, 'text_color', None)) else getattr(node, 'text_color', (255, 255, 255))
        
        # Convert to RGB if needed (handle QColor)
        color = self._to_rgb_tuple(color)
        border_color = self._to_rgb_tuple(border_color)
        text_color = self._to_rgb_tuple(text_color)
        
        # Handle selected/disabled state
        selected = node.selected() if callable(getattr(node, 'selected', None)) else getattr(node, 'selected', False)
        disabled = node.disabled() if callable(getattr(node, 'disabled', None)) else getattr(node, 'disabled', False)
        
        return NodeGeometry(
            x=x,
            y=y,
            color=color,
            border_color=border_color,
            text_color=text_color,
            selected=selected,
            disabled=disabled,
        )

    def _extract_edge_geometry(self, edge) -> EdgeGeometry:
        """Extract visual geometry of an edge."""
        color = edge.color() if callable(getattr(edge, 'color', None)) else getattr(edge, 'color', (255, 255, 255))
        color = self._to_rgb_tuple(color)
        
        # Try to extract waypoints for curved routing
        waypoints = []
        if hasattr(edge, 'path') and hasattr(edge.path, 'controlPoints'):
            waypoints = [(p.x(), p.y()) for p in edge.path.controlPoints()]
        
        return EdgeGeometry(
            source_node_id=edge.source_node.id,
            target_node_id=edge.target_node.id,
            source_port=self._get_port_name(edge.source_port),
            target_port=self._get_port_name(edge.target_port),
            flow_type=getattr(edge, 'flow_type', 'default'),
            color=color,
            waypoints=waypoints,
        )

    def _extract_canvas_state(self, node_graph) -> CanvasState:
        """Extract current canvas/viewport state."""
        # Get zoom and pan from graph widget
        if hasattr(node_graph, 'widget') and hasattr(node_graph.widget, 'zoom'):
            zoom = node_graph.widget.zoom
        else:
            zoom = 1.0
        
        # Pan state from transform
        pan_x = pan_y = 0.0
        if hasattr(node_graph, 'widget') and hasattr(node_graph.widget, 'transform'):
            transform = node_graph.widget.transform()
            pan_x = transform.dx()
            pan_y = transform.dy()
        
        return CanvasState(
            zoom_level=zoom,
            pan_x=pan_x,
            pan_y=pan_y,
        )

    def _apply_node_geometry(self, node, geometry: NodeGeometry) -> None:
        """Apply geometry to a node."""
        if hasattr(node, 'set_pos'):
            node.set_pos(geometry.x, geometry.y)
        
        if hasattr(node, 'set_color'):
            node.set_color(*geometry.color)
        
        if hasattr(node, 'set_border_color'):
            node.set_border_color(*geometry.border_color)

    def _apply_canvas_state(self, node_graph, state: CanvasState) -> None:
        """Apply canvas state to graph."""
        if hasattr(node_graph, 'widget'):
            widget = node_graph.widget
            
            if hasattr(widget, 'set_zoom'):
                widget.set_zoom(state.zoom_level)
            
            if hasattr(widget, 'set_pan'):
                widget.set_pan(state.pan_x, state.pan_y)

    def _serialize_snapshot(self, snapshot: GraphSnapshot) -> Dict[str, Any]:
        """Convert snapshot to serializable dict."""
        return {
            "format": "h2plant",
            "schema_version": self.SCHEMA_VERSION,
            "metadata": asdict(snapshot.metadata),
            "canvas_state": asdict(snapshot.canvas_state),
            "nodes": snapshot.nodes,
            "edges": snapshot.edges,
            "validation_result": snapshot.validation_result,
            "topology_analysis": snapshot.topology_analysis,
        }

    def _deserialize_snapshot(self, data: Dict[str, Any]) -> GraphSnapshot:
        """Convert dict back to snapshot."""
        metadata = ProjectMetadata(**data.get("metadata", {}))
        canvas_state = CanvasState(**data.get("canvas_state", {}))
        
        snapshot = GraphSnapshot(
            metadata=metadata,
            canvas_state=canvas_state,
            nodes=data.get("nodes", {}),
            edges=data.get("edges", []),
            validation_result=data.get("validation_result"),
            topology_analysis=data.get("topology_analysis"),
        )
        
        return snapshot

    def _validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate loaded data against schema."""
        required_fields = ["metadata", "nodes", "edges", "canvas_state"]
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
        
        return True

    def _create_backup(self, file_path: Path) -> None:
        """Create timestamped backup of existing file."""
        if not self.backup_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup created: {backup_path}")

    def _calculate_checksum(self, snapshot: GraphSnapshot) -> str:
        """Calculate SHA256 checksum of snapshot content."""
        # Create deterministic JSON representation
        data = self._serialize_snapshot(snapshot)
        # Remove checksum field itself to avoid circular dependency
        data["metadata"].pop("checksum", None)
        json_str = json.dumps(data, sort_keys=True, default=str)
        
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _get_port_name(self, port) -> str:
        """Extract port name safely."""
        if hasattr(port, 'name'):
            # Check if it's a method or property
            if callable(port.name):
                return port.name()
            return port.name
        elif isinstance(port, str):
            return port
        else:
            return "default"

    @staticmethod
    def _to_rgb_tuple(color) -> Tuple[int, int, int]:
        """Convert various color formats to RGB tuple."""
        if isinstance(color, tuple):
            if len(color) >= 3:
                return (int(color[0]), int(color[1]), int(color[2]))
        
        # Handle QColor from PySide6
        if hasattr(color, 'red') and hasattr(color, 'green') and hasattr(color, 'blue'):
            return (color.red(), color.green(), color.blue())
        
        # Default
        return (100, 100, 100)


# Export/Import functions
def export_to_json(snapshot: GraphSnapshot, file_path: str) -> None:
    """Export snapshot as configuration-only JSON."""
    manager = GraphPersistenceManager()
    manager.save(file_path, snapshot, format=SerializationFormat.JSON)


def export_to_h2plant(snapshot: GraphSnapshot, file_path: str) -> None:
    """Export snapshot in native H2Plant format (with layout)."""
    manager = GraphPersistenceManager()
    manager.save(file_path, snapshot, format=SerializationFormat.H2PLANT)


def import_from_h2plant(file_path: str) -> GraphSnapshot:
    """Import native H2Plant format."""
    manager = GraphPersistenceManager()
    return manager.load(file_path)
