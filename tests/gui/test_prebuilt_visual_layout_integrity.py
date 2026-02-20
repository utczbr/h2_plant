"""
Integrity checks for committed prebuilt visual twin artifact.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from h2_plant.gui.core.graph_persistence import GraphPersistenceManager
from h2_plant.gui.core.prebuilt_visual_layout import (
    generate_layout_file,
    prebuilt_layout_needs_regeneration,
)


def _assert_snapshot_integrity(snapshot):
    assert len(snapshot.nodes) == 123
    assert len(snapshot.edges) == 143
    assert snapshot.topology_analysis is not None
    assert snapshot.topology_analysis.get("scenario_manifest") is not None
    assert snapshot.topology_analysis.get("import_surface_schema_version") == 2

    typed_count = 0
    surfaced_count = 0
    for node_id, node_data in snapshot.nodes.items():
        assert not str(node_id).startswith("0x")
        if node_data["type"] == "nodes.Scenario.ScenarioComponentNode":
            continue
        typed_count += 1
        props = node_data.get("properties", {})
        assert "component_id" in props
        assert "__scenario_inputs" in props
        assert "__scenario_outputs" in props
        assert "__scenario_backend_type" in props
        assert "__scenario_unmapped_params" in props
        surfaced_keys = {
            k for k in props.keys()
            if not k.startswith("__")
            and k not in {"component_id", "backend_type"}
        }
        if surfaced_keys:
            surfaced_count += 1

    assert typed_count > 0
    assert surfaced_count > 0


def test_prebuilt_visual_layout_integrity_and_clean_typed_metadata():
    manager = GraphPersistenceManager()
    canonical_path = Path("h2_plant/gui/layouts/plant_topology_visual.h2plant")

    if canonical_path.exists():
        needs_regen, _reason = prebuilt_layout_needs_regeneration(
            canonical_path=canonical_path,
            scenarios_dir="scenarios",
            topology_file="plant_topology.yaml",
        )
        if not needs_regen:
            snapshot = manager.load(str(canonical_path))
            _assert_snapshot_integrity(snapshot)
            return

    with TemporaryDirectory() as tmp_dir:
        generated_path, _, _ = generate_layout_file(
            output_path=Path(tmp_dir) / "plant_topology_visual.h2plant",
            scenarios_dir="scenarios",
            topology_file="plant_topology.yaml",
            project_name="Plant Topology Visual Twin",
        )
        snapshot = manager.load(str(generated_path))
        _assert_snapshot_integrity(snapshot)
