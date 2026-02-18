from pathlib import Path

from h2_plant.gui.core.node_visual_semantic_audit import PHASE_DEFINITIONS, run_audit
from h2_plant.gui.core.prebuilt_visual_layout import generate_layout_file


def test_full_backend_mirror_audit_all_nodes(tmp_path):
    layout_path = tmp_path / "visual_layout.h2plant"
    generate_layout_file(
        output_path=layout_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Audit Test Layout",
    )

    output_dir = tmp_path / "qa_output"
    report = run_audit(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        layout_path=str(layout_path),
        output_dir=str(output_dir),
    )

    assert report["summary"]["total_backend_nodes"] == 123
    assert report["summary"]["total_snapshot_nodes"] == 123
    assert report["summary"]["severity_counts"]["critical"] == 0
    assert report["summary"]["severity_counts"]["major"] == 0
    assert report["summary"]["verdict"] in {"PASS", "PASS with legacy notes"}

    total_checked = sum(report["phase_summary"][phase]["checked_nodes"] for phase, _ in PHASE_DEFINITIONS)
    total_passed = sum(report["phase_summary"][phase]["passed_nodes"] for phase, _ in PHASE_DEFINITIONS)
    assert total_checked == 123
    assert total_passed == 123

    assert Path(report["artifacts"]["audit_report"]).exists()
    assert Path(report["artifacts"]["phase_summary"]).exists()

