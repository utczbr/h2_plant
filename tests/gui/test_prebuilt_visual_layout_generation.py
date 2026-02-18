"""
Tests for prebuilt visual layout generation helpers.
"""

import json
from pathlib import Path

from h2_plant.gui.core.graph_persistence import GraphPersistenceManager
from h2_plant.gui.core.prebuilt_visual_layout import (
    build_snapshot,
    ensure_prebuilt_layout_file,
    generate_layout_file,
    prebuilt_layout_needs_regeneration,
)
from h2_plant.gui.core.scenario_visual_importer import ScenarioVisualImporter


def test_generate_layout_file_creates_valid_snapshot(tmp_path):
    output_path = tmp_path / "plant_topology_visual.h2plant"

    saved_path, node_count, edge_count = generate_layout_file(
        output_path=output_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Plant Topology Visual Twin",
    )

    assert saved_path == output_path
    assert saved_path.exists()
    assert node_count == 123
    assert edge_count == 143

    snapshot = GraphPersistenceManager().load(str(saved_path))
    assert len(snapshot.nodes) == node_count
    assert len(snapshot.edges) == edge_count
    assert snapshot.topology_analysis is not None
    assert snapshot.topology_analysis.get("scenario_manifest") is not None


def test_build_snapshot_preserves_visual_model_metadata():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )

    snapshot = build_snapshot(project_name="Plant Topology Visual Twin", model=model)

    assert len(snapshot.nodes) == len(model.nodes)
    assert len(snapshot.edges) == len(model.edges)
    assert snapshot.topology_analysis is not None
    assert snapshot.topology_analysis.get("scenario_manifest") is not None
    assert snapshot.topology_analysis.get("scenario_economics") is not None
    assert snapshot.topology_analysis.get("import_surface_schema_version") == 1

    pem_node = snapshot.nodes.get("PEM_Electrolyzer", {})
    pem_props = pem_node.get("properties", {})
    assert pem_props.get("__scenario_backend_type") == "PEM"
    assert float(pem_props.get("rated_power_kw")) == 5350.0


def test_ensure_prebuilt_layout_file_falls_back_to_temp_on_canonical_failure(tmp_path, monkeypatch):
    canonical_path = tmp_path / "layouts" / "plant_topology_visual.h2plant"
    temp_dir = tmp_path / "prebuilt_temp"
    calls = []

    from h2_plant.gui.core import prebuilt_visual_layout as prebuilt_module

    original_generate = prebuilt_module.generate_layout_file

    def fake_generate_layout_file(*, output_path, scenarios_dir, topology_file, project_name):
        output_path = Path(output_path)
        calls.append(output_path)

        if output_path == canonical_path:
            raise PermissionError("simulated canonical write failure")

        return original_generate(
            output_path=output_path,
            scenarios_dir=scenarios_dir,
            topology_file=topology_file,
            project_name=project_name,
        )

    monkeypatch.setattr(prebuilt_module, "generate_layout_file", fake_generate_layout_file)

    saved_path, was_generated, used_temp_fallback, node_count, edge_count = ensure_prebuilt_layout_file(
        canonical_path=canonical_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Plant Topology Visual Twin",
        temp_dir=temp_dir,
    )

    assert was_generated is True
    assert used_temp_fallback is True
    assert node_count == 123
    assert edge_count == 143
    assert len(calls) == 2
    assert calls[0] == canonical_path
    assert calls[1] == saved_path
    assert saved_path.parent == temp_dir
    assert saved_path.exists()


def test_ensure_prebuilt_layout_file_force_regenerate_overwrites_existing(tmp_path):
    canonical_path = tmp_path / "layouts" / "plant_topology_visual.h2plant"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text("not a snapshot", encoding="utf-8")

    saved_path, was_generated, used_temp_fallback, node_count, edge_count = ensure_prebuilt_layout_file(
        canonical_path=canonical_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Plant Topology Visual Twin",
        force_regenerate=True,
    )

    assert saved_path == canonical_path
    assert was_generated is True
    assert used_temp_fallback is False
    assert node_count == 123
    assert edge_count == 143

    snapshot = GraphPersistenceManager().load(str(canonical_path))
    assert len(snapshot.nodes) == 123


def test_prebuilt_layout_needs_regeneration_detects_legacy_surface_schema(tmp_path):
    canonical_path = tmp_path / "plant_topology_visual.h2plant"
    generate_layout_file(
        output_path=canonical_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Plant Topology Visual Twin",
    )

    raw = json.loads(canonical_path.read_text(encoding="utf-8"))
    raw.setdefault("topology_analysis", {}).pop("import_surface_schema_version", None)
    canonical_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    needs, reason = prebuilt_layout_needs_regeneration(
        canonical_path=canonical_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    assert needs is True
    assert reason == "legacy_surface_schema"


def test_prebuilt_layout_needs_regeneration_detects_hash_drift(tmp_path):
    canonical_path = tmp_path / "plant_topology_visual.h2plant"
    generate_layout_file(
        output_path=canonical_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Plant Topology Visual Twin",
    )

    raw = json.loads(canonical_path.read_text(encoding="utf-8"))
    manifest = raw.setdefault("topology_analysis", {}).setdefault("scenario_manifest", {})
    hashes = manifest.setdefault("file_hashes", {})
    if hashes:
        first_key = sorted(hashes.keys())[0]
        hashes[first_key] = "0000000000000000000000000000000000000000000000000000000000000000"
    canonical_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    needs, reason = prebuilt_layout_needs_regeneration(
        canonical_path=canonical_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    assert needs is True
    assert reason.startswith("hash_drift:")
