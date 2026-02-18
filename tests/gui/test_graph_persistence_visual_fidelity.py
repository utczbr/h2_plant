"""Visual fidelity round-trip tests for GraphPersistenceManager."""

from __future__ import annotations

import json

from h2_plant.gui.core.graph_persistence import (
    CanvasState,
    GraphPersistenceManager,
    GraphSnapshot,
    ProjectMetadata,
)


class _Point:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y


class _Path:
    def __init__(self, points):
        self._points = points

    def controlPoints(self):
        return list(self._points)


class _Transform:
    def __init__(self, dx: float, dy: float):
        self._dx = dx
        self._dy = dy

    def dx(self):
        return self._dx

    def dy(self):
        return self._dy


class _CanvasWidget:
    def __init__(self):
        self.zoom = 1.25
        self._transform = _Transform(12.0, -8.0)
        self.grid_mode = True
        self.grid_size = 32
        self.snap_to_grid = False
        self.calls = []

    def transform(self):
        return self._transform

    def set_zoom(self, value):
        self.calls.append(("zoom", value))

    def set_pan(self, x, y):
        self.calls.append(("pan", x, y))

    def set_grid_mode(self, enabled):
        self.calls.append(("grid_mode", enabled))

    def set_grid_size(self, size):
        self.calls.append(("grid_size", size))

    def set_snap_to_grid(self, enabled):
        self.calls.append(("snap_to_grid", enabled))


class _View:
    def __init__(self, width: float = 320.0, height: float = 140.0, collapsed: bool = False):
        self._width = width
        self._height = height
        self._collapsed = collapsed

    def width(self):
        return self._width

    def height(self):
        return self._height

    def is_collapsed(self):
        return self._collapsed


class _PortName:
    def __init__(self, name: str, node=None):
        self._name = name
        self._node = node
        self.last_edge = None

    def name(self):
        return self._name

    def node(self):
        return self._node

    def connected_ports(self):
        return []

    def connect_to(self, _other, push_undo=False):
        self.last_edge = _ConnectedEdge()
        return self.last_edge


class _ConnectedEdge:
    def __init__(self):
        self.data = {}

    def set_color(self, r, g, b):
        self.data["color"] = (r, g, b)

    def set_width(self, width):
        self.data["width"] = width

    def set_style(self, style):
        self.data["style"] = style

    def set_selected(self, selected):
        self.data["selected"] = selected

    def set_waypoints(self, waypoints):
        self.data["waypoints"] = list(waypoints)


class _SnapshotNode:
    def __init__(self, node_id: str, display_name: str, color=(100, 110, 120)):
        self.id = node_id
        self.type_ = "AnyNode"
        self._display_name = display_name
        self._props = {"component_id": node_id}
        self._color = color
        self._border = (10, 20, 30)
        self._text = (210, 220, 230)
        self.view = _View(width=280.0, height=160.0, collapsed=True)

    def name(self):
        return self._display_name

    def get_properties(self):
        return dict(self._props)

    def pos(self):
        return (15.5, 42.0)

    def color(self):
        return self._color

    def border_color(self):
        return self._border

    def text_color(self):
        return self._text

    def selected(self):
        return True

    def disabled(self):
        return True

    def output_ports(self):
        return []


class _EdgeObj:
    def __init__(self, source_node, target_node):
        self.source_node = source_node
        self.target_node = target_node
        self.source_port = _PortName("h2_out", node=source_node)
        self.target_port = _PortName("h2_in", node=target_node)
        self.flow_type = "oxygen"
        self.width = 3.5
        self.style = "dashed"
        self.selected = True
        self.path = _Path([_Point(10, 20), _Point(30, 40)])

    def color(self):
        return (9, 8, 7)


class _SnapshotGraph:
    def __init__(self, nodes, edges):
        self._nodes = list(nodes)
        self._edges = list(edges)
        self.widget = _CanvasWidget()

    def all_nodes(self):
        return list(self._nodes)

    def all_edges(self):
        return list(self._edges)


class _RestoreNode:
    def __init__(self, name: str):
        self._name = name
        self._props = {"component_id": ""}
        self._inputs = {}
        self._outputs = {}
        self.state = {}
        self.view = _View()

    def name(self):
        return self._name

    def properties(self):
        return dict(self._props)

    def create_property(self, name, value=None, widget_type=0):
        self._props.setdefault(name, value)

    def set_property(self, name, value, push_undo=True):
        self._props[name] = value

    def input_ports(self):
        return list(self._inputs.values())

    def output_ports(self):
        return list(self._outputs.values())

    def add_input(self, name, flow_type=None, multi_input=True):
        self._inputs.setdefault(name, _PortName(name, node=self))

    def add_output(self, name, flow_type=None, multi_output=True):
        self._outputs.setdefault(name, _PortName(name, node=self))

    def get_input(self, name):
        return self._inputs.get(name)

    def get_output(self, name):
        return self._outputs.get(name)

    def set_pos(self, x, y):
        self.state["pos"] = (x, y)

    def set_size(self, w, h):
        self.state["size"] = (w, h)

    def set_color(self, r, g, b):
        self.state["color"] = (r, g, b)

    def set_border_color(self, r, g, b):
        self.state["border"] = (r, g, b)

    def set_text_color(self, r, g, b):
        self.state["text"] = (r, g, b)

    def set_selected(self, selected):
        self.state["selected"] = selected

    def set_disabled(self, disabled):
        self.state["disabled"] = disabled

    def set_collapsed(self, collapsed):
        self.state["collapsed"] = collapsed


class _RestoreGraph:
    def __init__(self):
        self.created = []
        self.widget = _CanvasWidget()

    def begin_undo(self, _label):
        return None

    def end_undo(self):
        return None

    def create_node(self, node_type, name=None, push_undo=False):
        node = _RestoreNode(name or node_type)
        self.created.append(node)
        return node


def test_create_snapshot_extracts_full_node_geometry_and_canvas_state():
    manager = GraphPersistenceManager()
    node = _SnapshotNode("N1", "Node 1")
    graph = _SnapshotGraph(nodes=[node], edges=[])

    snapshot = manager.create_snapshot(graph, {})
    node_geom = snapshot.nodes["N1"]["geometry"]

    assert node_geom["x"] == 15.5
    assert node_geom["y"] == 42.0
    assert node_geom["width"] == 280.0
    assert node_geom["height"] == 160.0
    assert node_geom["color"] == (100, 110, 120)
    assert node_geom["border_color"] == (10, 20, 30)
    assert node_geom["text_color"] == (210, 220, 230)
    assert node_geom["selected"] is True
    assert node_geom["disabled"] is True
    assert node_geom["collapsed"] is True

    assert snapshot.canvas_state.zoom_level == 1.25
    assert snapshot.canvas_state.pan_x == 12.0
    assert snapshot.canvas_state.pan_y == -8.0
    assert snapshot.canvas_state.grid_enabled is True
    assert snapshot.canvas_state.grid_size == 32
    assert snapshot.canvas_state.snap_to_grid is False


def test_create_snapshot_uses_edge_objects_with_visual_geometry():
    manager = GraphPersistenceManager()
    src = _SnapshotNode("SRC", "Source")
    dst = _SnapshotNode("DST", "Target")
    edge = _EdgeObj(src, dst)
    graph = _SnapshotGraph(nodes=[src, dst], edges=[edge])

    snapshot = manager.create_snapshot(graph, {})
    assert len(snapshot.edges) == 1
    edge_data = snapshot.edges[0]
    edge_geom = edge_data["geometry"]

    assert edge_data["source_node_id"] == "SRC"
    assert edge_data["target_node_id"] == "DST"
    assert edge_data["source_port"] == "h2_out"
    assert edge_data["target_port"] == "h2_in"
    assert edge_data["flow_type"] == "oxygen"
    assert tuple(edge_geom["color"]) == (9, 8, 7)
    assert edge_geom["width"] == 3.5
    assert edge_geom["style"] == "dashed"
    assert edge_geom["selected"] is True
    assert edge_geom["waypoints"] == [(10.0, 20.0), (30.0, 40.0)]


def test_restore_applies_extended_node_edge_and_canvas_visual_state():
    manager = GraphPersistenceManager()
    graph = _RestoreGraph()
    snapshot = GraphSnapshot(
        metadata=ProjectMetadata(name="visual-restore"),
        canvas_state=CanvasState(
            zoom_level=2.0,
            pan_x=100.0,
            pan_y=-50.0,
            grid_enabled=False,
            grid_size=48,
            snap_to_grid=True,
        ),
        nodes={
            "A": {
                "type": "AnyNode",
                "display_name": "Node A",
                "properties": {"component_id": "A"},
                "geometry": {
                    "x": 1.0,
                    "y": 2.0,
                    "width": 300.0,
                    "height": 200.0,
                    "color": [1, 2, 3],
                    "border_color": [4, 5, 6],
                    "text_color": [7, 8, 9],
                    "selected": True,
                    "disabled": True,
                    "collapsed": True,
                },
            },
            "B": {
                "type": "AnyNode",
                "display_name": "Node B",
                "properties": {"component_id": "B"},
                "geometry": {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 120.0,
                    "height": 80.0,
                    "color": [10, 20, 30],
                    "border_color": [40, 50, 60],
                    "text_color": [70, 80, 90],
                    "selected": False,
                    "disabled": False,
                    "collapsed": False,
                },
            },
        },
        edges=[
            {
                "source_node_id": "A",
                "target_node_id": "B",
                "source_port": "h2_out",
                "target_port": "h2_in",
                "flow_type": "hydrogen",
                "geometry": {
                    "source_node_id": "A",
                    "target_node_id": "B",
                    "source_port": "h2_out",
                    "target_port": "h2_in",
                    "flow_type": "hydrogen",
                    "color": [10, 11, 12],
                    "width": 5.0,
                    "style": "dotted",
                    "selected": True,
                    "waypoints": [[1.0, 2.0], [3.0, 4.0]],
                },
            }
        ],
    )

    manager.restore_to_graph(graph, snapshot)

    node_a = graph.created[0]
    node_b = graph.created[1]
    assert node_a.state["pos"] == (1.0, 2.0)
    assert node_a.state["size"] == (300.0, 200.0)
    assert node_a.state["color"] == (1, 2, 3)
    assert node_a.state["border"] == (4, 5, 6)
    assert node_a.state["text"] == (7, 8, 9)
    assert node_a.state["selected"] is True
    assert node_a.state["disabled"] is True
    assert node_a.state["collapsed"] is True

    edge_obj = node_a.get_output("h2_out").last_edge
    assert edge_obj is not None
    assert edge_obj.data["color"] == (10, 11, 12)
    assert edge_obj.data["width"] == 5.0
    assert edge_obj.data["style"] == "dotted"
    assert edge_obj.data["selected"] is True
    assert edge_obj.data["waypoints"] == [(1.0, 2.0), (3.0, 4.0)]

    assert ("zoom", 2.0) in graph.widget.calls
    assert ("pan", 100.0, -50.0) in graph.widget.calls
    assert ("grid_mode", False) in graph.widget.calls
    assert ("grid_size", 48) in graph.widget.calls
    assert ("snap_to_grid", True) in graph.widget.calls
    assert node_b.state["collapsed"] is False


def test_serialize_snapshot_includes_visual_fidelity_schema_marker():
    manager = GraphPersistenceManager()
    snapshot = GraphSnapshot(
        metadata=ProjectMetadata(name="schema"),
        canvas_state=CanvasState(),
        nodes={},
        edges=[],
    )
    data = manager._serialize_snapshot(snapshot)  # pylint: disable=protected-access
    assert data["visual_fidelity_schema_version"] == 1


def test_load_legacy_file_without_visual_fidelity_marker(tmp_path):
    manager = GraphPersistenceManager()
    legacy_data = {
        "format": "h2plant",
        "schema_version": "1.0",
        "metadata": {"name": "legacy"},
        "canvas_state": {"zoom_level": 1.0, "pan_x": 0.0, "pan_y": 0.0},
        "nodes": {},
        "edges": [],
    }
    file_path = tmp_path / "legacy.h2plant"
    file_path.write_text(json.dumps(legacy_data), encoding="utf-8")

    snapshot = manager.load(str(file_path))
    assert isinstance(snapshot, GraphSnapshot)
    assert manager._last_visual_fidelity_schema_version == 0  # pylint: disable=protected-access

