"""
Tests for equipment component-id resolution fallback from node display name.
"""

from h2_plant.gui.core.scenario_visual_importer import resolve_component_id_for_equipment


def test_component_id_resolution_falls_back_to_display_name_suffix():
    resolved = resolve_component_id_for_equipment(
        component_id=None,
        legacy_component_id=None,
        node_name="PSA Unit: SOEC_H2_PSA_1",
    )
    assert resolved == "SOEC_H2_PSA_1"


def test_component_id_resolution_prefers_component_id():
    resolved = resolve_component_id_for_equipment(
        component_id="PEM_Cluster",
        legacy_component_id="LEGACY_ID",
        node_name="PEM: OTHER",
    )
    assert resolved == "PEM_Cluster"
