"""Regression tests for removed-node fallback during graph restore."""

from __future__ import annotations

from h2_plant.gui.core.graph_persistence import (
    CanvasState,
    GraphPersistenceManager,
    GraphSnapshot,
    ProjectMetadata,
)


class _FakePort:
    def __init__(self, node, name: str):
        self._node = node
        self._name = name
        self._connected = []

    def name(self):
        return self._name

    def node(self):
        return self._node

    def connected_ports(self):
        return list(self._connected)

    def connect_to(self, other, push_undo=False):
        if other not in self._connected:
            self._connected.append(other)
        if self not in other._connected:
            other._connected.append(self)


class _FakeNode:
    _seq = 0

    def __init__(self, node_type: str, name: str):
        _FakeNode._seq += 1
        self.id = f"fake_{_FakeNode._seq}"
        self.type_ = node_type
        self._name = name
        self._props = {}
        self._inputs = {}
        self._outputs = {}

    def name(self):
        return self._name

    def input_ports(self):
        return list(self._inputs.values())

    def output_ports(self):
        return list(self._outputs.values())

    def add_input(self, name, flow_type=None, multi_input=True):
        self._inputs.setdefault(name, _FakePort(self, name))

    def add_output(self, name, flow_type=None, multi_output=True):
        self._outputs.setdefault(name, _FakePort(self, name))

    def get_input(self, name):
        return self._inputs.get(name)

    def get_output(self, name):
        return self._outputs.get(name)

    def create_property(self, name, value=None, widget_type=0):
        self._props.setdefault(name, value)

    def set_property(self, name, value, push_undo=True):
        self._props[name] = value

    def get_property(self, name):
        return self._props.get(name)

    def properties(self):
        return dict(self._props)

    def set_pos(self, x, y):
        self._props["_pos"] = (x, y)

    def set_color(self, r, g, b):
        self._props["_color"] = (r, g, b)

    def set_border_color(self, r, g, b):
        self._props["_border_color"] = (r, g, b)


class _FakeGraph:
    def __init__(self):
        self.nodes = []

    def begin_undo(self, label):
        return None

    def end_undo(self):
        return None

    def create_node(self, node_type, name, push_undo=False):
        if node_type != "nodes.Scenario.ScenarioComponentNode":
            raise ValueError(f"Unsupported node type in fake graph: {node_type}")
        node = _FakeNode(node_type=node_type, name=name)
        self.nodes.append(node)
        return node


def test_restore_unknown_types_falls_back_to_scenario_component_and_keeps_edges():
    manager = GraphPersistenceManager()
    graph = _FakeGraph()

    snapshot = GraphSnapshot(
        metadata=ProjectMetadata(name="legacy_layout"),
        canvas_state=CanvasState(),
        nodes={
            "legacy_1": {
                "type": "nodes.storage.LPTankNode",
                "display_name": "Legacy Tank",
                "properties": {
                    "component_id": "HP_Compressor_S4",
                    "capacity_per_tank_kg": 120.0,
                },
                "geometry": {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 100.0,
                    "height": 100.0,
                    "color": (100, 100, 100),
                    "border_color": (50, 50, 50),
                    "text_color": (255, 255, 255),
                    "selected": False,
                    "disabled": False,
                    "collapsed": False,
                },
            },
            "legacy_2": {
                "type": "nodes.legacy.FillingCompressorNode",
                "display_name": "Legacy Comp",
                "properties": {
                    "component_id": "Fill_1",
                    "efficiency": 75.0,
                },
                "geometry": {
                    "x": 200.0,
                    "y": 20.0,
                    "width": 100.0,
                    "height": 100.0,
                    "color": (100, 100, 100),
                    "border_color": (50, 50, 50),
                    "text_color": (255, 255, 255),
                    "selected": False,
                    "disabled": False,
                    "collapsed": False,
                },
            },
        },
        edges=[
            {
                "source_node_id": "legacy_1",
                "target_node_id": "legacy_2",
                "source_port": "h2_out",
                "target_port": "h2_in",
                "flow_type": "hydrogen",
                "geometry": {},
            }
        ],
    )

    manager.restore_to_graph(graph, snapshot)

    assert len(graph.nodes) == 2

    node_a = graph.nodes[0]
    node_b = graph.nodes[1]

    assert node_a.type_ == "nodes.Scenario.ScenarioComponentNode"
    assert node_b.type_ == "nodes.Scenario.ScenarioComponentNode"

    assert node_a.get_property("component_id") == "HP_Compressor_S4"
    assert node_a.get_property("__scenario_component_id") == "HP_Compressor_S4"
    assert node_a.get_property("__scenario_backend_type") == "DetailedTank"
    assert node_a.get_property("__legacy_removed_node_type") == "nodes.storage.LPTankNode"

    assert node_b.get_property("__scenario_backend_type") == "CompressorSingle"
    assert node_b.get_property("__legacy_removed_node_type") == "nodes.legacy.FillingCompressorNode"

    source_port = node_a.get_output("h2_out")
    target_port = node_b.get_input("h2_in")
    assert source_port is not None
    assert target_port is not None
    assert target_port in source_port.connected_ports()
