"""
Regression tests for permissive property restore in GraphPersistenceManager.
"""

from h2_plant.gui.core.graph_persistence import (
    CanvasState,
    GraphPersistenceManager,
    GraphSnapshot,
    ProjectMetadata,
)


class _StrictNode:
    """Node stub that raises on unknown set_property unless property exists."""

    def __init__(self, fail_create_for=None):
        self._props = {"component_id": ""}
        self._fail_create_for = set(fail_create_for or [])
        self._x = 0.0
        self._y = 0.0

    def properties(self):
        return dict(self._props)

    def create_property(self, name, value=None, widget_type=0):
        if name in self._fail_create_for:
            raise RuntimeError("create_property failed")
        self._props[name] = value

    def set_property(self, name, value, push_undo=True):
        if name not in self._props:
            raise RuntimeError(f'No property "{name}"')
        self._props[name] = value

    def input_ports(self):
        return []

    def output_ports(self):
        return []

    def get_input(self, _name):
        return None

    def get_output(self, _name):
        return None

    def set_pos(self, x, y):
        self._x = x
        self._y = y

    def set_color(self, *_args):
        return None

    def set_border_color(self, *_args):
        return None

    def name(self):
        return "StrictNode"


class _LegacyTypedNode(_StrictNode):
    """Typed-node stub exposing a fixed set of GUI properties."""

    def __init__(self):
        super().__init__()
        self._props = {
            "component_id": "",
            "max_power_kw": "0",
            "conversion_efficiency": "0",
            "system_group": "",
        }


class _SettableButHiddenNode(_StrictNode):
    """
    Node stub where set_property accepts keys not listed in properties().
    Simulates frameworks where settable fields are not fully reflected by
    `properties()` at restore time.
    """

    def __init__(self):
        super().__init__()
        self._settable_keys = {"component_id", "__scenario_backend_type"}

    def set_property(self, name, value, push_undo=True):
        if name not in self._settable_keys and name not in self._props:
            raise RuntimeError(f'No property "{name}"')
        # Keep hidden settable values outside properties() visibility.
        if name in self._settable_keys and name not in self._props:
            return
        self._props[name] = value


class _DuplicateCreateNode(_StrictNode):
    """Node stub where create_property raises duplicate-style error."""

    def __init__(self):
        super().__init__()
        self._settable_keys = {"component_id", "dup_meta"}

    def create_property(self, name, value=None, widget_type=0):
        if name == "dup_meta":
            raise RuntimeError("property already exists")
        return super().create_property(name, value=value, widget_type=widget_type)

    def set_property(self, name, value, push_undo=True):
        if name not in self._settable_keys and name not in self._props:
            raise RuntimeError(f'No property "{name}"')
        if name in self._settable_keys and name not in self._props:
            return
        self._props[name] = value


class _FakeGraph:
    def __init__(self):
        self.created = []

    def begin_undo(self, _label):
        return None

    def end_undo(self):
        return None

    def create_node(self, node_type, name=None, push_undo=False):
        if node_type == "BadNode":
            node = _StrictNode(fail_create_for={"bad_meta"})
        elif node_type == "SettableButHiddenNode":
            node = _SettableButHiddenNode()
        elif node_type == "DuplicateCreateNode":
            node = _DuplicateCreateNode()
        elif node_type == "RectifierNode":
            node = _LegacyTypedNode()
        else:
            node = _StrictNode()
        self.created.append((node_type, name, node))
        return node


def _snapshot_with_unknown_properties():
    geometry = {
        "x": 10.0,
        "y": 20.0,
        "width": 100.0,
        "height": 100.0,
        "color": [100, 100, 100],
        "border_color": [50, 50, 50],
        "text_color": [255, 255, 255],
        "selected": False,
        "disabled": False,
        "collapsed": False,
    }
    return GraphSnapshot(
        metadata=ProjectMetadata(name="restore-test"),
        canvas_state=CanvasState(),
        nodes={
            "ok_1": {
                "type": "GoodNode",
                "display_name": "Good Node",
                "properties": {
                    "component_id": "GOOD_1",
                    "unknown_meta": "abc",
                    "layout_direction": 0,
                    "inputs": {"inlet": {}},
                    "outputs": {"outlet": {}},
                },
                "geometry": geometry,
            },
            "bad_1": {
                "type": "BadNode",
                "display_name": "Bad Node",
                "properties": {
                    "component_id": "BAD_1",
                    "bad_meta": "should_warn_and_continue",
                },
                "geometry": geometry,
            },
        },
        edges=[],
    )


def test_restore_to_graph_is_permissive_for_unknown_properties(caplog):
    manager = GraphPersistenceManager()
    graph = _FakeGraph()
    snapshot = _snapshot_with_unknown_properties()

    with caplog.at_level("WARNING"):
        manager.restore_to_graph(graph, snapshot)

    assert len(graph.created) == 2
    good_node = graph.created[0][2]
    bad_node = graph.created[1][2]

    # Unknown property should be auto-created and restored.
    assert good_node.properties()["unknown_meta"] == "abc"
    # UI-only properties are intentionally ignored.
    assert "layout_direction" not in good_node.properties()
    assert "inputs" not in good_node.properties()
    assert "outputs" not in good_node.properties()
    # Unrecoverable property should be skipped without aborting node restore.
    assert "bad_meta" not in bad_node.properties()
    assert any("bad_meta" in rec.message for rec in caplog.records)
    assert not any("layout_direction" in rec.message for rec in caplog.records)
    assert not any("inputs" in rec.message for rec in caplog.records)
    assert not any("outputs" in rec.message for rec in caplog.records)


def test_restore_prefers_set_before_create_for_hidden_settable_properties(caplog):
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
        metadata=ProjectMetadata(name="set-first"),
        canvas_state=CanvasState(),
        nodes={
            "n1": {
                "type": "SettableButHiddenNode",
                "display_name": "Node 1",
                "properties": {"component_id": "N1", "__scenario_backend_type": "CompressorSingle"},
                "geometry": geometry,
            }
        },
        edges=[],
    )

    manager = GraphPersistenceManager()
    graph = _FakeGraph()
    with caplog.at_level("WARNING"):
        manager.restore_to_graph(graph, snapshot)

    assert len(graph.created) == 1
    # Key behavior: no warning about __scenario_backend_type create/set failure.
    assert not any("__scenario_backend_type" in rec.message for rec in caplog.records)


def test_restore_tolerates_duplicate_create_error_and_retries_set(caplog):
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
        metadata=ProjectMetadata(name="dup-create"),
        canvas_state=CanvasState(),
        nodes={
            "n1": {
                "type": "DuplicateCreateNode",
                "display_name": "Node 1",
                "properties": {"component_id": "N1", "dup_meta": "v"},
                "geometry": geometry,
            }
        },
        edges=[],
    )

    manager = GraphPersistenceManager()
    graph = _FakeGraph()
    with caplog.at_level("WARNING"):
        manager.restore_to_graph(graph, snapshot)

    assert len(graph.created) == 1
    # Duplicate create should be treated as soft-success, no warning expected.
    assert not any("dup_meta" in rec.message for rec in caplog.records)


def test_restore_hydrates_typed_nodes_from_hidden_scenario_params():
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
        metadata=ProjectMetadata(name="legacy-typed"),
        canvas_state=CanvasState(),
        nodes={
            "n1": {
                "type": "RectifierNode",
                "display_name": "PowerTransformer: SOEC_Transformer",
                "properties": {
                    "component_id": "SOEC_Transformer",
                    "__scenario_backend_type": "PowerTransformer",
                    "__scenario_params": {
                        "rated_power_mw": 15.25,
                        "efficiency": 0.95,
                        "system_group": "SOEC",
                        "process_step": 8,
                    },
                },
                "geometry": geometry,
            }
        },
        edges=[],
    )

    manager = GraphPersistenceManager()
    graph = _FakeGraph()
    manager.restore_to_graph(graph, snapshot)

    restored = graph.created[0][2]
    assert float(restored.properties()["max_power_kw"]) == 15250.0
    assert float(restored.properties()["conversion_efficiency"]) == 95.0
    assert restored.properties()["system_group"] == "SOEC"
    assert restored.properties()["__scenario_unmapped_params"] == {"process_step": 8}
