"""
Tests for YAML type parsing in ScenarioComponentNode param sync.

Covers Fix 4: set_property should parse string values with yaml.safe_load
before storing them in __scenario_params, preserving numeric/bool/list types.
"""

import pytest

pytest.importorskip("NodeGraphQt")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication

from h2_plant.gui.nodes.scenario_component import ScenarioComponentNode


# ---------------------------------------------------------------------------
# Fixture: QApplication
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_configured_node(params: dict) -> ScenarioComponentNode:
    """Create a ScenarioComponentNode with surfaced parameter properties."""
    node = ScenarioComponentNode()
    node.configure_from_scenario(
        component_id="test_node",
        backend_type="TestType",
        input_ports=["in_1"],
        output_ports=["out_1"],
        params=params,
    )
    return node


# ---------------------------------------------------------------------------
# Fix 4: YAML type parsing on param sync
# ---------------------------------------------------------------------------

class TestParamTypeParsing:
    """Verify set_property parses string values to typed values in __scenario_params."""

    def test_numeric_string_parsed_to_float(self):
        node = _make_configured_node({"flow_rate": 10.0})
        # Simulate UI edit: text widget sends string
        node.set_property("flow_rate", "3.14")
        params = node.get_property("__scenario_params")
        assert params["flow_rate"] == 3.14
        assert isinstance(params["flow_rate"], float)

    def test_integer_string_parsed_to_int(self):
        node = _make_configured_node({"count": 5})
        node.set_property("count", "42")
        params = node.get_property("__scenario_params")
        assert params["count"] == 42
        assert isinstance(params["count"], int)

    def test_bool_string_parsed_to_bool(self):
        node = _make_configured_node({"enabled": True})
        node.set_property("enabled", "true")
        params = node.get_property("__scenario_params")
        assert params["enabled"] is True
        assert isinstance(params["enabled"], bool)

    def test_bool_false_string_parsed(self):
        node = _make_configured_node({"enabled": True})
        node.set_property("enabled", "false")
        params = node.get_property("__scenario_params")
        assert params["enabled"] is False

    def test_list_string_parsed_to_list(self):
        node = _make_configured_node({"stages": [1, 2]})
        node.set_property("stages", "[1, 2, 3]")
        params = node.get_property("__scenario_params")
        assert params["stages"] == [1, 2, 3]
        assert isinstance(params["stages"], list)

    def test_plain_string_stays_string(self):
        node = _make_configured_node({"label": "default"})
        node.set_property("label", "hello world")
        params = node.get_property("__scenario_params")
        assert params["label"] == "hello world"
        assert isinstance(params["label"], str)

    def test_non_string_value_passes_through(self):
        """Non-string values (e.g. from programmatic set) are stored as-is."""
        node = _make_configured_node({"flow_rate": 10.0})
        node.set_property("flow_rate", 25.5)
        params = node.get_property("__scenario_params")
        assert params["flow_rate"] == 25.5
        assert isinstance(params["flow_rate"], float)

    def test_malformed_yaml_stays_as_string(self):
        """Invalid YAML should fall back to storing the raw string."""
        node = _make_configured_node({"config": "default"})
        node.set_property("config", "[unclosed bracket")
        params = node.get_property("__scenario_params")
        assert params["config"] == "[unclosed bracket"
        assert isinstance(params["config"], str)
