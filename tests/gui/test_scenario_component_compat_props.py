"""
Compatibility property checks for ScenarioComponentNode.
"""

import pytest

pytest.importorskip("NodeGraphQt")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication

from h2_plant.gui.nodes.scenario_component import ScenarioComponentNode


def test_scenario_component_has_compat_properties_and_populates_them():
    app = QApplication.instance() or QApplication([])
    assert app is not None

    node = ScenarioComponentNode()
    props = node.properties()

    assert "__scenario_component_id" in props
    assert "__scenario_backend_type" in props

    node.configure_from_scenario(
        component_id="SOEC_H2_Interchanger_1",
        backend_type="Interchanger",
        input_ports=["hot_in"],
        output_ports=["hot_out"],
        params={"min_approach_temp_k": 10.0},
    )

    assert node.get_property("__scenario_component_id") == "SOEC_H2_Interchanger_1"
    assert node.get_property("__scenario_backend_type") == "Interchanger"
