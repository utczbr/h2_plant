"""
Tests for scenario-run lock behavior on imported layouts.
"""

from h2_plant.gui.core.scenario_visual_importer import resolve_simulation_source


def test_imported_layout_forces_scenario_mode_from_manifest():
    manifest = {
        "scenarios_dir": "/tmp/my_scenarios",
        "topology_file": "/tmp/my_scenarios/plant_topology.yaml",
    }

    scenarios_dir, topology_file, forced = resolve_simulation_source(
        scenario_manifest=manifest,
        requested_scenarios_dir=None,
    )

    assert forced is True
    assert scenarios_dir == "/tmp/my_scenarios"
    assert topology_file == "/tmp/my_scenarios/plant_topology.yaml"


def test_non_imported_layout_keeps_requested_mode():
    scenarios_dir, topology_file, forced = resolve_simulation_source(
        scenario_manifest=None,
        requested_scenarios_dir="scenarios",
    )

    assert forced is False
    assert scenarios_dir == "scenarios"
    assert topology_file is None
