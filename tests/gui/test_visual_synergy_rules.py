from h2_plant.gui.core.node_visual_semantic_audit import _contrast_ratio, _overlap_ratio, run_audit
from h2_plant.gui.core.prebuilt_visual_layout import generate_layout_file


def test_contrast_ratio_heuristic_behaves_as_expected():
    assert _contrast_ratio((255, 255, 255), (0, 0, 0)) > 10.0
    assert _contrast_ratio((120, 120, 120), (130, 130, 130)) < 1.2


def test_overlap_ratio_heuristic_behaves_as_expected():
    assert _overlap_ratio((0, 0, 100, 100), (200, 200, 100, 100)) == 0.0
    assert _overlap_ratio((0, 0, 100, 100), (10, 10, 100, 100)) > 0.5


def test_visual_synergy_audit_no_blocking_contrast_or_overlap_findings(tmp_path):
    layout_path = tmp_path / "visual_layout.h2plant"
    generate_layout_file(
        output_path=layout_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Visual Synergy Layout",
    )

    report = run_audit(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        layout_path=str(layout_path),
        output_dir=str(tmp_path / "qa_output"),
    )

    categories = {"low_text_contrast", "severe_node_overlap"}
    blocking = [
        finding
        for finding in report["findings"]
        if finding["category"] in categories and finding["severity"] in {"critical", "major"}
    ]
    assert blocking == []

