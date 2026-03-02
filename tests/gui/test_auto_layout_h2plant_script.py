"""
Regression checks for standalone .h2plant industrial relayout script.
"""

import json
from pathlib import Path
import subprocess
import sys

from h2_plant.gui.core.industrial_layout_engine import X_SPACING
from h2_plant.gui.core.prebuilt_visual_layout import generate_layout_file


def _flatten_node_positions(layout_path: Path) -> None:
    raw = json.loads(layout_path.read_text(encoding="utf-8"))
    for node_data in raw.get("nodes", {}).values():
        geometry = dict(node_data.get("geometry", {}) or {})
        geometry["x"] = 0.0
        geometry["y"] = 0.0
        node_data["geometry"] = geometry
    layout_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def _intra_group_max_gaps(snapshot: dict) -> dict:
    nodes = dict(snapshot.get("nodes", {}) or {})
    visual_layout = dict(snapshot.get("topology_analysis", {}).get("visual_layout", {}) or {})
    group_map = dict(visual_layout.get("node_group_map") or {})

    x_by_group = {}
    for node_id, node_data in nodes.items():
        group_name = group_map.get(node_id)
        if not group_name:
            continue
        x = float(dict(node_data.get("geometry", {}) or {}).get("x", 0.0))
        x_by_group.setdefault(group_name, []).append(x)

    gaps = {}
    for group_name, xs in x_by_group.items():
        if len(xs) < 2:
            continue
        xs_sorted = sorted(xs)
        max_gap = max(xs_sorted[idx + 1] - xs_sorted[idx] for idx in range(len(xs_sorted) - 1))
        gaps[group_name] = max_gap
    return gaps


def test_auto_layout_h2plant_script_writes_organized_output(tmp_path):
    input_path = tmp_path / "plant_topology_visual.h2plant"
    output_path = tmp_path / "plant_topology_visual_organized.h2plant"
    generate_layout_file(
        output_path=input_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Script Layout Input",
    )
    _flatten_node_positions(input_path)

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/auto_layout_h2plant.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=str(Path.cwd()),
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    assert "Industrial layout completed" in proc.stdout
    organized = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(organized.get("nodes", {})) == 123
    assert len(organized.get("edges", [])) == 143

    coords = {
        (
            float(node_data.get("geometry", {}).get("x", 0.0)),
            float(node_data.get("geometry", {}).get("y", 0.0)),
        )
        for node_data in organized.get("nodes", {}).values()
    }
    assert len(coords) > 40

    visual_layout = organized.get("topology_analysis", {}).get("visual_layout", {})
    assert visual_layout.get("layout_mode") == "industrial_pfd_v1"
    assert int(visual_layout.get("layout_schema_version", 0)) >= 2
    assert visual_layout.get("spacing_policy") == "group_local_rank"
    assert isinstance(visual_layout.get("node_group_map"), dict)
    assert isinstance(visual_layout.get("node_zone_map"), dict)
    gaps = _intra_group_max_gaps(organized)
    assert gaps
    assert all(max_gap <= (2.0 * X_SPACING) for max_gap in gaps.values())


def test_auto_layout_h2plant_script_supports_in_place_mode(tmp_path):
    input_path = tmp_path / "plant_topology_visual.h2plant"
    generate_layout_file(
        output_path=input_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Script InPlace Input",
    )
    _flatten_node_positions(input_path)

    subprocess.run(
        [
            sys.executable,
            "scripts/auto_layout_h2plant.py",
            "--input",
            str(input_path),
            "--in-place",
        ],
        cwd=str(Path.cwd()),
        check=True,
        capture_output=True,
        text=True,
    )

    organized = json.loads(input_path.read_text(encoding="utf-8"))
    visual_layout = organized.get("topology_analysis", {}).get("visual_layout", {})
    assert visual_layout.get("layout_mode") == "industrial_pfd_v1"
    assert int(visual_layout.get("layout_schema_version", 0)) >= 2
    assert visual_layout.get("spacing_policy") == "group_local_rank"
