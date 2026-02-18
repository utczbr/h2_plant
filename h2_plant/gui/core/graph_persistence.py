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

from h2_plant.gui.core.scenario_param_mapper import backend_to_gui_props

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
    VISUAL_FIDELITY_SCHEMA_VERSION = 1
    FILE_EXTENSION = ".h2plant"
    IGNORED_RESTORE_KEYS: Set[str] = {
        # NodeGraphQt/UI session keys that are not functional process configuration.
        "type_",
        "icon",
        "visible",
        "width",
        "height",
        "pos",
        "layout_direction",
        "port_deletion_allowed",
        "subgraph_session",
        "inputs",
        "outputs",
        "custom",
    }
    
    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize persistence manager.
        
        Args:
            backup_dir: Directory for automatic backups. If None, no backups created.
        """
        self.backup_dir = backup_dir
        self._last_visual_fidelity_schema_version = 0
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
        
        # Extract edge geometries from edge objects when available.
        # Fallback to legacy port traversal if the graph API doesn't expose edges.
        edges_list = self._extract_edges_from_graph(node_graph)
        
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
        visual_restore_stats = {
            "node_applied": 0,
            "node_skipped": 0,
            "edge_applied": 0,
            "edge_skipped": 0,
            "canvas_applied": 0,
            "canvas_skipped": 0,
        }
        
        # Disable auto-layout during creation
        node_graph.begin_undo("Load Layout")
        
        try:
            for node_id, node_data in snapshot.nodes.items():
                try:
                    node_type = node_data["type"]
                    node_properties = node_data.get("properties", {})
                    logger.debug(f"Creating node {node_type}...")
                    # Create node instance
                    try:
                        node = node_graph.create_node(node_type, name=node_data["display_name"], push_undo=False)
                    except Exception as create_exc:
                        logger.warning(
                            "Failed to create node type '%s' (id=%s). Will try ScenarioComponent fallback: %s",
                            node_type,
                            node_id,
                            create_exc,
                        )
                        node = None

                    if node is None:
                        fallback_type = "nodes.Scenario.ScenarioComponentNode"
                        fallback_properties = self._build_fallback_scenario_properties(
                            node_id=node_id,
                            node_type=node_type,
                            node_properties=node_properties,
                        )
                        try:
                            node = node_graph.create_node(
                                fallback_type,
                                name=node_data["display_name"],
                                push_undo=False,
                            )
                        except Exception as fallback_exc:
                            logger.error(
                                "Failed to create fallback node for '%s' (id=%s): %s",
                                node_type,
                                node_id,
                                fallback_exc,
                            )
                            continue

                        node_type = fallback_type
                        node_properties = fallback_properties
                        logger.warning(
                            "Applied fallback restore: original_type='%s' id='%s' -> ScenarioComponentNode (backend_type='%s')",
                            node_data["type"],
                            node_id,
                            fallback_properties.get("__scenario_backend_type", "PassiveComponent"),
                        )

                    logger.debug(f"Node created: {node}")
                    
                    # Force ID to match snapshot (critical for connections)
                    # NodeGraphQt usually generates a new ID, so we might need to map it
                    # But for persistence, we want to keep the ID if possible or map it
                    
                    # NOTE: NodeGraphQt nodes have a unique ID. We can't easily overwrite it 
                    # without internal access. So we map snapshot_id -> new_node
                    node_map[node_id] = node
                    default_props = set(node.properties().keys()) if hasattr(node, "properties") else set()

                    # Rebuild dynamic scenario ports before property assignment and
                    # edge restoration so exact imported port names remain connectable.
                    self._restore_dynamic_ports(node, node_properties)

                    # Apply properties using permissive restore:
                    # unknown properties are auto-created as hidden metadata.
                    self._apply_node_properties_safe(
                        node=node,
                        node_id=node_id,
                        node_type=node_type,
                        node_properties=node_properties,
                    )
                    self._hydrate_typed_scenario_properties(
                        node=node,
                        node_type=node_type,
                        node_properties=node_properties,
                        default_props=default_props,
                    )
                            
                    # Apply geometry
                    geom = self._coerce_node_geometry(node_data.get("geometry", {}))
                    self._apply_node_geometry(node, geom, visual_restore_stats)
                except Exception as e:
                    logger.error(f"Failed to create node {node_type}: {e}")

            # 2.5 Restore dynamic ports from edges (legacy fallback)
            # Some legacy snapshots do not carry __scenario_inputs/__scenario_outputs.
            self._restore_ports_from_edges(snapshot=snapshot, node_map=node_map)

            # 2. Create Connections (Edges)
            logger.debug("Creating connections...")
            for edge_data in snapshot.edges:
                src_id = edge_data["source_node_id"]
                tgt_id = edge_data["target_node_id"]
                
                if src_id in node_map and tgt_id in node_map:
                    src_node = node_map[src_id]
                    tgt_node = node_map[tgt_id]
                    
                    src_port = edge_data["source_port"]
                    tgt_port = edge_data["target_port"]
                    
                    logger.debug(f"Connecting {src_node.name()}.{src_port} -> {tgt_node.name()}.{tgt_port}")
                    
                    # Connect
                    try:
                        # Find ports
                        # NodeGraphQt ports are objects, we need to find them by name
                        s_port = src_node.get_output(src_port)
                        t_port = tgt_node.get_input(tgt_port)
                        
                        if s_port and t_port:
                            created_edge = s_port.connect_to(t_port, push_undo=False)
                            if created_edge is not None and isinstance(edge_data.get("geometry"), dict):
                                edge_geom = self._coerce_edge_geometry(edge_data=edge_data, geometry_data=edge_data["geometry"])
                                self._apply_edge_geometry(created_edge, edge_geom, visual_restore_stats)
                        else:
                            logger.warning(f"Port not found: {src_port} or {tgt_port}")
                    except Exception as e:
                        logger.error(f"Failed to connect {src_id}->{tgt_id}: {e}")
            
            # 3. Apply Canvas State
            logger.debug("Applying canvas state...")
            self._apply_canvas_state(node_graph, snapshot.canvas_state, visual_restore_stats)
            logger.debug("Restore complete.")
            
        finally:
            node_graph.end_undo()
            logger.info(
                "Visual restore summary: node(applied=%d skipped=%d) edge(applied=%d skipped=%d) canvas(applied=%d skipped=%d)",
                visual_restore_stats["node_applied"],
                visual_restore_stats["node_skipped"],
                visual_restore_stats["edge_applied"],
                visual_restore_stats["edge_skipped"],
                visual_restore_stats["canvas_applied"],
                visual_restore_stats["canvas_skipped"],
            )

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
                # Infer from port name instead of blindly coercing to hydrogen
                ft_str = self._infer_flow_type_from_port(edge_data.get("source_port", ""))
                
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

    def _extract_edges_from_graph(self, node_graph) -> List[Dict[str, Any]]:
        """Extract edge records from graph edge objects or legacy port traversal."""
        edges = self._extract_edges_from_objects(node_graph)
        if edges:
            return edges
        return self._extract_edges_from_ports(node_graph)

    def _extract_edges_from_objects(self, node_graph) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str, str]] = set()
        for edge in self._list_graph_edge_objects(node_graph):
            endpoint = self._resolve_edge_endpoints(edge)
            if endpoint is None:
                continue
            source_node_id, target_node_id, source_port_name, target_port_name = endpoint
            edge_key = (source_node_id, source_port_name, target_node_id, target_port_name)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            geometry = self._extract_edge_geometry(
                edge=edge,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                source_port_name=source_port_name,
                target_port_name=target_port_name,
            )
            edges.append(
                {
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "source_port": source_port_name,
                    "target_port": target_port_name,
                    "flow_type": geometry.flow_type,
                    "geometry": asdict(geometry),
                }
            )
        return edges

    def _extract_edges_from_ports(self, node_graph) -> List[Dict[str, Any]]:
        """Legacy edge extraction path based on connected ports."""
        edges: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str, str]] = set()
        for node in node_graph.all_nodes():
            for output_port in node.output_ports():
                for target_port in output_port.connected_ports():
                    target_node = target_port.node()
                    source_port_name = self._get_port_name(output_port)
                    target_port_name = self._get_port_name(target_port)
                    edge_key = (node.id, source_port_name, target_node.id, target_port_name)
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)

                    inferred_ft = self._infer_flow_type_from_port(source_port_name)
                    geometry = EdgeGeometry(
                        source_node_id=node.id,
                        target_node_id=target_node.id,
                        source_port=source_port_name,
                        target_port=target_port_name,
                        flow_type=inferred_ft,
                    )
                    edges.append(
                        {
                            "source_node_id": node.id,
                            "target_node_id": target_node.id,
                            "source_port": source_port_name,
                            "target_port": target_port_name,
                            "flow_type": inferred_ft,
                            "geometry": asdict(geometry),
                        }
                    )
        return edges

    def _list_graph_edge_objects(self, node_graph) -> List[Any]:
        """Best-effort discovery of edge objects from graph APIs."""
        candidates: List[Any] = []
        for attr_name in ("all_edges", "all_pipes", "edges", "pipes"):
            ref = getattr(node_graph, attr_name, None)
            if ref is None:
                continue
            try:
                values = ref() if callable(ref) else ref
            except Exception:
                continue
            if values is None:
                continue
            try:
                candidates.extend(list(values))
            except Exception:
                continue
            if candidates:
                break
        return candidates

    def _resolve_edge_endpoints(self, edge) -> Optional[Tuple[str, str, str, str]]:
        """Resolve edge endpoint identity across different edge APIs."""
        src_port_obj = getattr(edge, "source_port", None) or getattr(edge, "output_port", None)
        tgt_port_obj = getattr(edge, "target_port", None) or getattr(edge, "input_port", None)
        src_node_obj = getattr(edge, "source_node", None)
        tgt_node_obj = getattr(edge, "target_node", None)
        if src_node_obj is None and src_port_obj is not None and hasattr(src_port_obj, "node"):
            try:
                src_node_obj = src_port_obj.node()
            except Exception:
                src_node_obj = None
        if tgt_node_obj is None and tgt_port_obj is not None and hasattr(tgt_port_obj, "node"):
            try:
                tgt_node_obj = tgt_port_obj.node()
            except Exception:
                tgt_node_obj = None

        source_node_id = getattr(src_node_obj, "id", None)
        target_node_id = getattr(tgt_node_obj, "id", None)
        source_port_name = self._get_port_name(src_port_obj)
        target_port_name = self._get_port_name(tgt_port_obj)

        if not source_node_id or not target_node_id:
            return None
        if source_port_name == "unknown" or target_port_name == "unknown":
            return None
        return str(source_node_id), str(target_node_id), str(source_port_name), str(target_port_name)

    def _extract_node_pos(self, node) -> Tuple[float, float]:
        raw_pos = self._read_attr_or_call(node, "pos", (0.0, 0.0))
        if hasattr(raw_pos, "x") and hasattr(raw_pos, "y"):
            try:
                return float(raw_pos.x()), float(raw_pos.y())
            except Exception:
                pass
        if isinstance(raw_pos, (tuple, list)) and len(raw_pos) >= 2:
            return self._safe_float(raw_pos[0], 0.0), self._safe_float(raw_pos[1], 0.0)
        return 0.0, 0.0

    def _extract_node_dimensions(self, node) -> Tuple[float, float]:
        defaults = (100.0, 100.0)
        width = self._safe_float(self._read_attr_or_call(node, "width", defaults[0]), defaults[0])
        height = self._safe_float(self._read_attr_or_call(node, "height", defaults[1]), defaults[1])

        view = getattr(node, "view", None)
        if view is not None:
            width = self._safe_float(self._read_attr_or_call(view, "width", width), width)
            height = self._safe_float(self._read_attr_or_call(view, "height", height), height)
            rect = self._read_attr_or_call(view, "boundingRect")
            if rect is not None and hasattr(rect, "width") and hasattr(rect, "height"):
                try:
                    width = self._safe_float(rect.width(), width)
                    height = self._safe_float(rect.height(), height)
                except Exception:
                    pass
        return width, height

    def _extract_node_collapsed(self, node) -> bool:
        for obj in (node, getattr(node, "view", None)):
            if obj is None:
                continue
            for key in ("collapsed", "is_collapsed"):
                value = self._read_attr_or_call(obj, key, None)
                if value is not None:
                    return bool(value)
        return False

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _read_attr_or_call(obj: Any, attr_name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if not hasattr(obj, attr_name):
            return default
        try:
            value = getattr(obj, attr_name)
            return value() if callable(value) else value
        except Exception:
            return default

    def _extract_node_geometry(self, node) -> NodeGeometry:
        """Extract visual geometry of a node."""
        x, y = self._extract_node_pos(node)
        width, height = self._extract_node_dimensions(node)

        # Handle both property and method access for compatibility.
        color = self._to_rgb_tuple(self._read_attr_or_call(node, "color", (100, 100, 100)))
        border_color = self._to_rgb_tuple(self._read_attr_or_call(node, "border_color", (50, 50, 50)))
        text_color = self._to_rgb_tuple(self._read_attr_or_call(node, "text_color", (255, 255, 255)))

        selected = bool(self._read_attr_or_call(node, "selected", False))
        disabled = bool(self._read_attr_or_call(node, "disabled", False))
        collapsed = self._extract_node_collapsed(node)

        return NodeGeometry(
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            border_color=border_color,
            text_color=text_color,
            selected=selected,
            disabled=disabled,
            collapsed=collapsed,
        )

    def _extract_edge_geometry(
        self,
        edge,
        source_node_id: str,
        target_node_id: str,
        source_port_name: str,
        target_port_name: str,
    ) -> EdgeGeometry:
        """Extract visual geometry of an edge with robust API fallbacks."""
        color = self._to_rgb_tuple(self._read_attr_or_call(edge, "color", (255, 255, 255)))
        width = float(self._read_attr_or_call(edge, "width", 2.0))
        style = str(self._read_attr_or_call(edge, "style", "solid"))
        selected = bool(self._read_attr_or_call(edge, "selected", False))

        flow_type = self._read_attr_or_call(edge, "flow_type", "default")
        if hasattr(flow_type, "value"):
            flow_type = flow_type.value
        flow_type = str(flow_type)
        if not flow_type or flow_type == "default":
            flow_type = self._infer_flow_type_from_port(source_port_name)

        waypoints = []
        edge_path = getattr(edge, "path", None)
        if callable(edge_path):
            try:
                edge_path = edge_path()
            except Exception:
                edge_path = None
        if edge_path is not None and hasattr(edge_path, "controlPoints"):
            try:
                waypoints = [(float(p.x()), float(p.y())) for p in edge_path.controlPoints()]
            except Exception:
                waypoints = []

        return EdgeGeometry(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_port=source_port_name,
            target_port=target_port_name,
            flow_type=flow_type,
            color=color,
            width=width,
            style=style,
            selected=selected,
            waypoints=waypoints,
        )

    def _extract_canvas_state(self, node_graph) -> CanvasState:
        """Extract current canvas/viewport state."""
        widget = self._get_graph_widget(node_graph)
        zoom = self._safe_float(self._read_attr_or_call(widget, "zoom", 1.0), 1.0)
        if zoom == 1.0:
            zoom = self._safe_float(self._read_attr_or_call(widget, "get_zoom", 1.0), 1.0)

        pan_x = pan_y = 0.0
        if widget is not None and hasattr(widget, "transform"):
            try:
                transform = widget.transform()
                pan_x = float(transform.dx())
                pan_y = float(transform.dy())
            except Exception:
                pan_x = pan_y = 0.0

        grid_enabled = bool(
            self._read_attr_or_call(
                widget,
                "grid_mode",
                self._read_attr_or_call(widget, "grid_enabled", True),
            )
        )
        grid_size = int(self._safe_float(self._read_attr_or_call(widget, "grid_size", 20), 20))
        snap_to_grid = bool(
            self._read_attr_or_call(
                widget,
                "snap_to_grid",
                self._read_attr_or_call(widget, "grid_snap", True),
            )
        )

        return CanvasState(
            zoom_level=zoom,
            pan_x=pan_x,
            pan_y=pan_y,
            grid_enabled=grid_enabled,
            grid_size=grid_size,
            snap_to_grid=snap_to_grid,
        )

    def _apply_node_geometry(self, node, geometry: NodeGeometry, stats: Optional[Dict[str, int]] = None) -> None:
        """Apply geometry to a node."""
        self._apply_setter(
            targets=[node],
            candidates=[("set_pos", (geometry.x, geometry.y))],
            stats=stats,
            category="node",
            field="position",
        )
        self._apply_setter(
            targets=[node, getattr(node, "view", None)],
            candidates=[
                ("set_size", (geometry.width, geometry.height)),
                ("set_dimensions", (geometry.width, geometry.height)),
                ("set_width", (geometry.width,)),
                ("set_height", (geometry.height,)),
            ],
            stats=stats,
            category="node",
            field="dimensions",
        )
        self._apply_setter(
            targets=[node],
            candidates=[("set_color", tuple(geometry.color))],
            stats=stats,
            category="node",
            field="color",
        )
        self._apply_setter(
            targets=[node],
            candidates=[("set_border_color", tuple(geometry.border_color))],
            stats=stats,
            category="node",
            field="border_color",
        )
        self._apply_setter(
            targets=[node, getattr(node, "view", None)],
            candidates=[("set_text_color", tuple(geometry.text_color))],
            stats=stats,
            category="node",
            field="text_color",
        )
        self._apply_setter(
            targets=[node, getattr(node, "view", None)],
            candidates=[("set_selected", (bool(geometry.selected),))],
            stats=stats,
            category="node",
            field="selected",
        )
        self._apply_setter(
            targets=[node, getattr(node, "view", None)],
            candidates=[("set_disabled", (bool(geometry.disabled),))],
            stats=stats,
            category="node",
            field="disabled",
        )
        collapse_candidates = [("set_collapsed", (bool(geometry.collapsed),))]
        if geometry.collapsed:
            collapse_candidates.append(("collapse", tuple()))
        else:
            collapse_candidates.append(("expand", tuple()))
        self._apply_setter(
            targets=[node, getattr(node, "view", None)],
            candidates=collapse_candidates,
            stats=stats,
            category="node",
            field="collapsed",
        )

    def _apply_edge_geometry(self, edge, geometry: EdgeGeometry, stats: Optional[Dict[str, int]] = None) -> None:
        """Best-effort edge style restoration."""
        self._apply_setter(
            targets=[edge],
            candidates=[("set_color", tuple(geometry.color))],
            stats=stats,
            category="edge",
            field="color",
        )
        self._apply_setter(
            targets=[edge],
            candidates=[("set_width", (geometry.width,))],
            stats=stats,
            category="edge",
            field="width",
        )
        self._apply_setter(
            targets=[edge],
            candidates=[("set_style", (geometry.style,))],
            stats=stats,
            category="edge",
            field="style",
        )
        self._apply_setter(
            targets=[edge],
            candidates=[("set_selected", (bool(geometry.selected),))],
            stats=stats,
            category="edge",
            field="selected",
        )
        if geometry.waypoints:
            self._apply_setter(
                targets=[edge],
                candidates=[
                    ("set_waypoints", (list(geometry.waypoints),)),
                    ("set_path_points", (list(geometry.waypoints),)),
                ],
                stats=stats,
                category="edge",
                field="waypoints",
            )

    def _apply_canvas_state(self, node_graph, state: CanvasState, stats: Optional[Dict[str, int]] = None) -> None:
        """Apply canvas state to graph."""
        widget = self._get_graph_widget(node_graph)
        if widget is None:
            self._count_stat(stats, "canvas", applied=False)
            return

        self._apply_setter(
            targets=[widget],
            candidates=[("set_zoom", (state.zoom_level,))],
            stats=stats,
            category="canvas",
            field="zoom",
        )
        self._apply_setter(
            targets=[widget],
            candidates=[("set_pan", (state.pan_x, state.pan_y))],
            stats=stats,
            category="canvas",
            field="pan",
        )
        self._apply_setter(
            targets=[widget],
            candidates=[("set_grid_mode", (bool(state.grid_enabled),)), ("set_grid_enabled", (bool(state.grid_enabled),))],
            stats=stats,
            category="canvas",
            field="grid_enabled",
        )
        self._apply_setter(
            targets=[widget],
            candidates=[("set_grid_size", (int(state.grid_size),))],
            stats=stats,
            category="canvas",
            field="grid_size",
        )
        self._apply_setter(
            targets=[widget],
            candidates=[("set_snap_to_grid", (bool(state.snap_to_grid),)), ("set_grid_snap", (bool(state.snap_to_grid),))],
            stats=stats,
            category="canvas",
            field="snap_to_grid",
        )

    def _coerce_node_geometry(self, data: Dict[str, Any]) -> NodeGeometry:
        payload = dict(data or {})
        return NodeGeometry(
            x=self._safe_float(payload.get("x", 0.0), 0.0),
            y=self._safe_float(payload.get("y", 0.0), 0.0),
            width=self._safe_float(payload.get("width", 100.0), 100.0),
            height=self._safe_float(payload.get("height", 100.0), 100.0),
            color=self._to_rgb_tuple(payload.get("color", (100, 100, 100))),
            border_color=self._to_rgb_tuple(payload.get("border_color", (50, 50, 50))),
            text_color=self._to_rgb_tuple(payload.get("text_color", (255, 255, 255))),
            selected=bool(payload.get("selected", False)),
            disabled=bool(payload.get("disabled", False)),
            collapsed=bool(payload.get("collapsed", False)),
        )

    def _coerce_edge_geometry(self, edge_data: Dict[str, Any], geometry_data: Dict[str, Any]) -> EdgeGeometry:
        payload = dict(geometry_data or {})
        source_node_id = str(payload.get("source_node_id", edge_data.get("source_node_id", "")))
        target_node_id = str(payload.get("target_node_id", edge_data.get("target_node_id", "")))
        source_port = str(payload.get("source_port", edge_data.get("source_port", "")))
        target_port = str(payload.get("target_port", edge_data.get("target_port", "")))
        flow_type = str(payload.get("flow_type", edge_data.get("flow_type", "default")))
        waypoints = payload.get("waypoints", [])
        if not isinstance(waypoints, list):
            waypoints = []
        normalized_waypoints = []
        for point in waypoints:
            if isinstance(point, (tuple, list)) and len(point) >= 2:
                normalized_waypoints.append((self._safe_float(point[0], 0.0), self._safe_float(point[1], 0.0)))
        return EdgeGeometry(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_port=source_port,
            target_port=target_port,
            flow_type=flow_type,
            color=self._to_rgb_tuple(payload.get("color", (255, 255, 255))),
            width=self._safe_float(payload.get("width", 2.0), 2.0),
            style=str(payload.get("style", "solid")),
            selected=bool(payload.get("selected", False)),
            waypoints=normalized_waypoints,
        )

    @staticmethod
    def _get_graph_widget(node_graph):
        widget_ref = getattr(node_graph, "widget", None)
        if callable(widget_ref):
            try:
                return widget_ref()
            except Exception:
                return None
        return widget_ref

    def _apply_setter(
        self,
        targets: List[Any],
        candidates: List[Tuple[str, Tuple[Any, ...]]],
        stats: Optional[Dict[str, int]],
        category: str,
        field: str,
    ) -> bool:
        for target in targets:
            if target is None:
                continue
            for method_name, args in candidates:
                method = getattr(target, method_name, None)
                if not callable(method):
                    continue
                try:
                    method(*args)
                    self._count_stat(stats, category, applied=True)
                    return True
                except Exception as exc:
                    logger.debug("Skipped %s.%s(%s): %s", type(target).__name__, method_name, field, exc)
                    continue
        self._count_stat(stats, category, applied=False)
        return False

    @staticmethod
    def _count_stat(stats: Optional[Dict[str, int]], category: str, applied: bool) -> None:
        if stats is None:
            return
        key = f"{category}_{'applied' if applied else 'skipped'}"
        if key in stats:
            stats[key] += 1

    def _restore_dynamic_ports(self, node, node_properties: Dict[str, Any]) -> None:
        """Restore dynamic ports encoded in node properties (scenario imports)."""
        input_ports = self._coerce_port_list(
            node_properties.get("__scenario_inputs", node_properties.get("_scenario_inputs"))
        )
        output_ports = self._coerce_port_list(
            node_properties.get("__scenario_outputs", node_properties.get("_scenario_outputs"))
        )

        if not input_ports and not output_ports:
            return

        # Prefer node-specific implementation when available.
        if hasattr(node, "configure_scenario_ports"):
            try:
                node.configure_scenario_ports(input_ports, output_ports)
                return
            except Exception as exc:
                logger.warning("Failed using configure_scenario_ports on %s: %s", node, exc)

        existing_inputs = {p.name() for p in node.input_ports()} if hasattr(node, "input_ports") else set()
        existing_outputs = {p.name() for p in node.output_ports()} if hasattr(node, "output_ports") else set()

        for port_name in input_ports:
            if port_name in existing_inputs:
                continue
            try:
                if hasattr(node, "add_input"):
                    try:
                        node.add_input(port_name, flow_type=self._infer_flow_type_from_port(port_name), multi_input=True)
                    except TypeError:
                        node.add_input(port_name)
            except Exception as exc:
                logger.warning("Failed restoring input port '%s' on %s: %s", port_name, node, exc)

        for port_name in output_ports:
            if port_name in existing_outputs:
                continue
            try:
                if hasattr(node, "add_output"):
                    try:
                        node.add_output(port_name, flow_type=self._infer_flow_type_from_port(port_name), multi_output=True)
                    except TypeError:
                        node.add_output(port_name)
            except Exception as exc:
                logger.warning("Failed restoring output port '%s' on %s: %s", port_name, node, exc)

    def _restore_ports_from_edges(self, snapshot: GraphSnapshot, node_map: Dict[str, Any]) -> None:
        """
        Restore missing ports by scanning edge definitions.

        This is a compatibility path for legacy snapshots that don't include
        explicit scenario port metadata.
        """
        for edge_data in snapshot.edges:
            src_id = edge_data.get("source_node_id")
            tgt_id = edge_data.get("target_node_id")
            src_port = edge_data.get("source_port")
            tgt_port = edge_data.get("target_port")

            if src_id not in node_map or tgt_id not in node_map:
                continue
            if not src_port or not tgt_port:
                continue

            src_node = node_map[src_id]
            tgt_node = node_map[tgt_id]

            self._ensure_port(
                node=src_node,
                port_name=str(src_port),
                direction="output",
                node_id=str(src_id),
            )
            self._ensure_port(
                node=tgt_node,
                port_name=str(tgt_port),
                direction="input",
                node_id=str(tgt_id),
            )

    def _apply_node_properties_safe(
        self,
        node,
        node_id: str,
        node_type: str,
        node_properties: Dict[str, Any],
    ) -> None:
        """
        Restore node properties permissively.

        If a property doesn't exist on the node, attempt to create it as hidden
        metadata first. Failures are logged per-property and never abort node
        creation.
        """
        if not hasattr(node, "set_property"):
            return

        for prop_name, raw_value in (node_properties or {}).items():
            if prop_name in self.IGNORED_RESTORE_KEYS:
                logger.debug(
                    "Skipping UI-only restore key: node_type=%s node_id=%s property=%s",
                    node_type,
                    node_id,
                    prop_name,
                )
                continue

            value = self._normalize_property_value(raw_value)

            # Authoritative path: try set first, then create+retry set.
            if self._set_node_property(node, prop_name, value):
                continue

            created = self._create_hidden_property(node, prop_name, raw_value)
            if created and self._set_node_property(node, prop_name, value):
                continue

            # One bad property should not stop node restore.
            if created:
                logger.warning(
                    "Property restore skipped (set failed after create): node_type=%s node_id=%s property=%s",
                    node_type,
                    node_id,
                    prop_name,
                )
            else:
                logger.warning(
                    "Property restore skipped (create failed): node_type=%s node_id=%s property=%s",
                    node_type,
                    node_id,
                    prop_name,
                )

    def _hydrate_typed_scenario_properties(
        self,
        node,
        node_type: str,
        node_properties: Dict[str, Any],
        default_props: Set[str],
    ) -> None:
        """
        Hydrate typed imported nodes from canonical __scenario_params for legacy snapshots.

        Legacy prebuilt files carried only hidden canonical params on typed nodes. This
        method surfaces mapped values to typed GUI properties when missing.
        """
        if node_type == "nodes.Scenario.ScenarioComponentNode":
            return

        params = node_properties.get("__scenario_params")
        if not isinstance(params, dict) or not params:
            return

        backend_type = str(
            node_properties.get("__scenario_backend_type")
            or node_properties.get("backend_type")
            or self._legacy_gui_class_to_backend_type(node_type)
            or ""
        ).strip()
        if not backend_type:
            return

        available_props = {str(key) for key in default_props if str(key) and not str(key).startswith("__")}
        mapped_props, unmapped = backend_to_gui_props(
            backend_type=backend_type,
            backend_params=params,
            available_props=available_props,
        )

        has_surface_props = any(prop_name in node_properties for prop_name in mapped_props.keys())
        if not has_surface_props:
            for prop_name, raw_value in mapped_props.items():
                if prop_name in {"component_id"} or prop_name.startswith("__"):
                    continue
                value = self._normalize_property_value(raw_value)
                self._set_node_property(node, prop_name, value)

        # Keep imported unmapped backend params available for read-only inspection.
        if "__scenario_unmapped_params" not in node_properties:
            if self._create_hidden_property(node, "__scenario_unmapped_params", unmapped):
                self._set_node_property(node, "__scenario_unmapped_params", unmapped)

        if "__scenario_backend_type" not in node_properties:
            if self._create_hidden_property(node, "__scenario_backend_type", backend_type):
                self._set_node_property(node, "__scenario_backend_type", backend_type)

    def _ensure_port(self, node, port_name: str, direction: str, node_id: str) -> None:
        """Ensure a specific input/output port exists on node."""
        try:
            if direction == "output":
                exists = hasattr(node, "get_output") and node.get_output(port_name) is not None
            else:
                exists = hasattr(node, "get_input") and node.get_input(port_name) is not None

            if exists:
                return

            flow_type = self._infer_flow_type_from_port(port_name)
            if direction == "output" and hasattr(node, "add_output"):
                try:
                    node.add_output(port_name, flow_type=flow_type, multi_output=True)
                except TypeError:
                    node.add_output(port_name)
            elif direction == "input" and hasattr(node, "add_input"):
                try:
                    node.add_input(port_name, flow_type=flow_type, multi_input=True)
                except TypeError:
                    node.add_input(port_name)
        except Exception as exc:
            logger.warning(
                "Failed restoring %s port from edge metadata: node_id=%s port=%s error=%s",
                direction,
                node_id,
                port_name,
                exc,
            )

    @staticmethod
    def _normalize_property_value(value: Any) -> Any:
        """
        Normalize snapshot values before restoring to node properties.

        NodeGraphQt text inputs can raise if receiving numeric values directly,
        so numeric scalars are converted to strings.
        """
        if isinstance(value, (int, float)):
            return str(value)
        return value

    def _build_fallback_scenario_properties(
        self,
        node_id: str,
        node_type: str,
        node_properties: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build ScenarioComponent-compatible property set for removed/unknown nodes.

        Preserves identity keys and stores legacy class information for round-trip
        fidelity.
        """
        props = dict(node_properties or {})
        component_id = str(
            props.get("component_id")
            or props.get("__scenario_component_id")
            or node_id
        ).strip()

        backend_type_raw = (
            props.get("__scenario_backend_type")
            or props.get("backend_type")
            or ""
        )
        backend_type = str(backend_type_raw).strip()
        if not backend_type:
            backend_type = self._legacy_gui_class_to_backend_type(node_type)

        params = props.get("__scenario_params")
        if not isinstance(params, dict):
            params = {}

        # Preserve original visible properties as scenario params when missing,
        # so legacy layouts can be re-exported without silent data loss.
        if not params:
            params = {
                k: v
                for k, v in props.items()
                if not str(k).startswith("__scenario_")
            }

        props["component_id"] = component_id
        props["backend_type"] = backend_type
        props["__scenario_component_id"] = component_id
        props["__scenario_backend_type"] = backend_type
        props["__scenario_params"] = params
        props["__scenario_unmapped_params"] = props.get("__scenario_unmapped_params", {})
        props["__legacy_removed_node_type"] = str(node_type)

        return props

    @staticmethod
    def _legacy_gui_class_to_backend_type(node_type: str) -> str:
        """Best-effort mapping from legacy GUI class names to backend type IDs."""
        class_name = str(node_type).split(".")[-1]
        mapping = {
            "PEMStackNode": "PEM",
            "SOECStackNode": "SOEC",
            "RectifierNode": "PowerTransformer",
            "ChillerNode": "Chiller",
            "DryCoolerNode": "DryCooler",
            "CoalescerNode": "Coalescer",
            "KnockOutDrumNode": "KnockOutDrum",
            "PSAUnitNode": "PSA Unit",
            "DeoxoReactorNode": "DeoxoReactor",
            "MixerNode": "Mixer",
            "ValveNode": "Valve",
            "WaterPurifierNode": "WaterPurifier",
            "UltraPureWaterTankNode": "UltraPureWaterTank",
        }
        if class_name in mapping:
            return mapping[class_name]

        lower = class_name.lower()
        if "tank" in lower:
            return "DetailedTank"
        if "compressor" in lower:
            return "CompressorSingle"
        if "pump" in lower:
            return "WaterPumpThermodynamic"
        if "mixer" in lower:
            return "SignalMakeupMixer"
        if "consumer" in lower:
            return "DischargeStation"
        if "grid" in lower and "connection" in lower:
            return "GridConnection"
        if "water" in lower and "supply" in lower:
            return "ExternalWaterSource"
        if "wind" in lower:
            return "WindEnergySource"
        if "demand" in lower:
            return "DemandScheduler"
        if "energy" in lower and "price" in lower:
            return "EnergyPrice"
        if "arbitrage" in lower:
            return "Arbitrage"
        if "battery" in lower:
            return "Battery"
        if "atr" in lower:
            return "IntegratedATRPlant"
        if "electrolyzer" in lower:
            return "PEM"
        if "heat" in lower and "exchanger" in lower:
            return "HeatExchanger"
        if "steam" in lower and "generator" in lower:
            return "ElectricBoiler"
        if "tsa" in lower:
            return "TSA Unit"
        if "separation" in lower and "tank" in lower:
            return "SeparationTank"
        if "water" in lower and "treatment" in lower:
            return "WaterTreatment"
        if "oxygen" in lower and "source" in lower:
            return "ExternalOxygenSource"
        if "heat" in lower and "source" in lower:
            return "HeatSource"
        if "natural" in lower and "gas" in lower:
            return "BiogasSource"
        if "ambient" in lower and "heat" in lower:
            return "AmbientHeat"
        if "wgs" in lower:
            return "WGSReactor"
        return "PassiveComponent"

    @staticmethod
    def _create_hidden_property(node, prop_name: str, value: Any) -> bool:
        """Try to create a hidden property for unknown snapshot keys."""
        if not hasattr(node, "create_property"):
            return False

        try:
            node.create_property(prop_name, value=value, widget_type=0)
            return True
        except TypeError:
            try:
                node.create_property(prop_name, value)
                return True
            except Exception as exc:
                if "already exists" in str(exc).lower():
                    return True
                return False
        except Exception as exc:
            # If property already exists in underlying model, treat as soft success.
            if "already exists" in str(exc).lower():
                return True
            return False

    @staticmethod
    def _set_node_property(node, prop_name: str, value: Any) -> bool:
        """Set property with compatibility for different set_property signatures."""
        try:
            node.set_property(prop_name, value, push_undo=False)
            return True
        except TypeError:
            try:
                node.set_property(prop_name, value)
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _serialize_snapshot(self, snapshot: GraphSnapshot) -> Dict[str, Any]:
        """Convert snapshot to serializable dict."""
        return {
            "format": "h2plant",
            "schema_version": self.SCHEMA_VERSION,
            "visual_fidelity_schema_version": self.VISUAL_FIDELITY_SCHEMA_VERSION,
            "metadata": asdict(snapshot.metadata),
            "canvas_state": asdict(snapshot.canvas_state),
            "nodes": snapshot.nodes,
            "edges": snapshot.edges,
            "validation_result": snapshot.validation_result,
            "topology_analysis": snapshot.topology_analysis,
        }

    def _deserialize_snapshot(self, data: Dict[str, Any]) -> GraphSnapshot:
        """Convert dict back to snapshot."""
        try:
            self._last_visual_fidelity_schema_version = int(data.get("visual_fidelity_schema_version") or 0)
        except Exception:
            self._last_visual_fidelity_schema_version = 0

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

    @staticmethod
    def _infer_flow_type_from_port(port_name: str) -> str:
        """Infer flow type string from port name for persistence."""
        name = port_name.lower()
        if any(k in name for k in ('power', 'electric', 'energy', 'p_')):
            return 'electricity'
        if any(k in name for k in ('o2', 'oxygen')):
            return 'oxygen'
        if any(k in name for k in ('water', 'h2o', 'ultrapure', 'raw_water')):
            return 'water'
        if any(k in name for k in ('h2', 'hydrogen', 'purified', 'compressed_h2')):
            return 'hydrogen'
        if any(k in name for k in ('gas', 'inlet', 'feed', 'syngas', 'tail')):
            return 'gas'
        if any(k in name for k in ('heat', 'thermal', 'duty', 'cooling')):
            return 'heat'
        if any(k in name for k in ('signal', 'control', 'demand')):
            return 'signal'
        return 'stream'  # Safe default

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
            return "unknown"

    @staticmethod
    def _to_rgb_tuple(color) -> Tuple[int, int, int]:
        """Convert various color formats to RGB tuple."""
        if isinstance(color, (tuple, list)):
            if len(color) >= 3:
                return (int(color[0]), int(color[1]), int(color[2]))

        if isinstance(color, str):
            parts = [part.strip() for part in color.split(",")]
            if len(parts) >= 3:
                try:
                    return (int(parts[0]), int(parts[1]), int(parts[2]))
                except (TypeError, ValueError):
                    pass
        
        # Handle QColor from PySide6
        if hasattr(color, 'red') and hasattr(color, 'green') and hasattr(color, 'blue'):
            return (color.red(), color.green(), color.blue())
        
        # Default
        return (100, 100, 100)

    @staticmethod
    def _coerce_port_list(value: Any) -> List[str]:
        """Normalize serialized port lists from snapshot properties."""
        if value is None:
            return []
        if isinstance(value, list):
            ports = [str(item).strip() for item in value]
        else:
            ports = [token.strip() for token in str(value).split(",")]
        return [port for port in ports if port]


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
