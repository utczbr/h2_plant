from h2_plant.gui.core.node_visual_semantic_audit import run_audit
from h2_plant.gui.core.prebuilt_visual_layout import generate_layout_file


PORT_EDGE_CATEGORIES = {
    "missing_input_port",
    "unexpected_input_port",
    "missing_output_port",
    "unexpected_output_port",
    "missing_outgoing_edge",
    "unexpected_outgoing_edge",
    "self_loop_edge",
}


def test_full_port_integrity_audit_no_critical_major_port_findings(tmp_path):
    layout_path = tmp_path / "visual_layout.h2plant"
    generate_layout_file(
        output_path=layout_path,
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        project_name="Port Integrity Layout",
    )

    report = run_audit(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
        layout_path=str(layout_path),
        output_dir=str(tmp_path / "qa_output"),
    )

    blocking_findings = [
        finding
        for finding in report["findings"]
        if finding["category"] in PORT_EDGE_CATEGORIES and finding["severity"] in {"critical", "major"}
    ]
    assert blocking_findings == []

