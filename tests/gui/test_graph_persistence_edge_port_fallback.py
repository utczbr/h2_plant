"""
Legacy edge-driven port reconstruction tests for GraphPersistenceManager.
"""

from h2_plant.gui.core.graph_persistence import (
    CanvasState,
    GraphPersistenceManager,
    GraphSnapshot,
    ProjectMetadata,
)


class _Port:
    def __init__(self, node, name, direction):
        self._node = node
        self._name = name
        self._direction = direction
        self._connected = []

    def name(self):
        return self._name

    def node(self):
        return self._node

    def connected_ports(self):
        return list(self._connected)

    def connect_to(self, other, push_undo=False):
        self._connected.append(other)


class _NodeWithDynamicPorts:
    def __init__(self):
        self._props = {"component_id": ""}
        self._inputs = {}
        self._outputs = {}

    def properties(self):
        return dict(self._props)

    def create_property(self, name, value=None, widget_type=0):
        self._props[name] = value

    def set_property(self, name, value, push_undo=True):
        if name not in self._props:
            raise RuntimeError(f'No property "{name}"')
        self._props[name] = value

    def add_input(self, name, flow_type=None, multi_input=False):
        self._inputs[name] = _Port(self, name, "input")

    def add_output(self, name, flow_type=None, multi_output=True):
        self._outputs[name] = _Port(self, name, "output")

    def get_input(self, name):
        return self._inputs.get(name)

    def get_output(self, name):
        return self._outputs.get(name)

    def input_ports(self):
        return list(self._inputs.values())

    def output_ports(self):
        return list(self._outputs.values())

    def set_pos(self, x, y):
        return None

    def set_color(self, *args):
        return None

    def set_border_color(self, *args):
        return None

    def name(self):
        return "DynamicNode"


class _FakeGraph:
    def __init__(self):
        self.nodes = []

    def begin_undo(self, _label):
        return None

    def end_undo(self):
        return None

    def create_node(self, node_type, name=None, push_undo=False):
        node = _NodeWithDynamicPorts()
        self.nodes.append(node)
        return node


def test_restore_reconstructs_ports_from_edges_when_metadata_missing():
    geometry = {
        "x": 0.0,
        "y": 0.0,
        "width": 100.0,
        "height": 100.0,
        "color": [100, 100, 100],
        "border_color": [50, 50, 50],
        "text_color": [255, 255, 255],
        "selected": False,
        "disabled": False,
        "collapsed": False,
    }
    snapshot = GraphSnapshot(
        metadata=ProjectMetadata(name="legacy-ports"),
        canvas_state=CanvasState(),
        nodes={
            "a": {
                "type": "AnyNode",
                "display_name": "A",
                "properties": {"component_id": "A"},
                "geometry": geometry,
            },
            "b": {
                "type": "AnyNode",
                "display_name": "B",
                "properties": {"component_id": "B"},
                "geometry": geometry,
            },
        },
        edges=[
            {
                "source_node_id": "a",
                "target_node_id": "b",
                "source_port": "water_out",
                "target_port": "raw_water_in",
                "flow_type": "stream",
            }
        ],
    )

    manager = GraphPersistenceManager()
    graph = _FakeGraph()
    manager.restore_to_graph(graph, snapshot)

    src, dst = graph.nodes
    assert src.get_output("water_out") is not None
    assert dst.get_input("raw_water_in") is not None
    assert dst.get_input("raw_water_in") in src.get_output("water_out").connected_ports()
