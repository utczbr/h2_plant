"""
Source-level regression checks for prebuilt visual twin menu variants.
"""

from pathlib import Path


def _source() -> str:
    return Path("h2_plant/gui/ui/main_window.py").read_text(encoding="utf-8")


def test_prebuilt_visual_twin_menu_is_a_submenu_with_both_variants():
    source = _source()
    assert 'open_prebuilt_menu = file_menu.addMenu("Open Prebuilt Visual Twin")' in source
    assert '"SOEC + PEM + ATR"' in source
    assert '"SOEC + PEM"' in source
    assert "partial(" in source
    assert "self.open_prebuilt_visual_twin" in source


def test_prebuilt_visual_twin_menu_wires_pem_soec_variant_files():
    source = _source()
    assert '"plant_topology_PEM+SOEC.yaml"' in source
    assert '"Economics/equipment_mappings_PEM+SOEC.yaml"' in source
    assert '"plant_topology_visual_pem_soec.h2plant"' in source


def test_view_menu_uses_industrial_auto_layout_label_and_status_tip():
    source = _source()
    assert 'auto_layout_action = QAction("Auto-Layout (Industrial PFD)", self)' in source
    assert (
        'auto_layout_action.setStatusTip("Recalculate industrial plant layout and redraw grouped overlays")'
        in source
    )


def test_import_scenario_visual_applies_angle_pipe_style_and_equipment_inference():
    source = _source()
    assert "equipment_file = infer_equipment_file_for_topology(topology_file)" in source
    assert "self.graph.set_pipe_style(PipeLayoutEnum.ANGLE)" in source
    assert "Could not apply ANGLE pipe layout after import" in source
