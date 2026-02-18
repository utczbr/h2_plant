"""
Round-trip persistence tests for scenario import metadata in .h2plant snapshots.
"""

from h2_plant.gui.core.graph_persistence import (
    CanvasState,
    GraphPersistenceManager,
    GraphSnapshot,
    ProjectMetadata,
)
from h2_plant.gui.core.scenario_visual_importer import ScenarioVisualImporter


def test_scenario_manifest_persists_in_h2plant_roundtrip(tmp_path):
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )

    topology_analysis = {
        "scenario_manifest": dict(model.metadata["scenario_manifest"]),
        "scenario_economics": dict(model.metadata["economics"]),
        "scenario_equipment_entries": list(model.metadata["equipment_entries"]),
        "scenario_equipment_index": dict(model.metadata["equipment_index"]),
    }

    snapshot = GraphSnapshot(
        metadata=ProjectMetadata(name="Scenario Import Test"),
        canvas_state=CanvasState(),
        nodes={},
        edges=[],
        topology_analysis=topology_analysis,
    )

    manager = GraphPersistenceManager()
    out_file = tmp_path / "scenario_import.h2plant"
    manager.save(str(out_file), snapshot, create_backup=False)

    loaded = manager.load(str(out_file))

    assert loaded.topology_analysis is not None
    assert loaded.topology_analysis["scenario_manifest"]["topology_file_name"] == "plant_topology.yaml"
    assert loaded.topology_analysis["scenario_manifest"]["physics_file"] == "physics_parameters.yaml"
    assert loaded.topology_analysis["scenario_manifest"]["simulation_config_file"] == "simulation_config.yaml"
    assert loaded.topology_analysis["scenario_economics"]["h2_price_eur_kg"] == 9.6
    assert "SOEC_H2_PSA_1" in loaded.topology_analysis["scenario_equipment_index"]
