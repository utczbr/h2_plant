"""
Tests for equipment mapping reverse index used by scenario visual import.
"""

from h2_plant.gui.core.scenario_visual_importer import (
    build_equipment_index,
    load_scenario_bundle,
    normalize_equipment_entries,
)


def test_equipment_index_references_all_resolve_to_topology_ids():
    bundle = load_scenario_bundle("scenarios", "plant_topology.yaml")
    topology_ids = {node["id"] for node in bundle["topology"]["nodes"]}

    equipment_entries = normalize_equipment_entries(
        bundle["equipment"].get("equipment", []) or []
    )
    equipment_index = build_equipment_index(equipment_entries)

    referenced_ids = set(equipment_index.keys())
    missing_ids = referenced_ids - topology_ids

    assert len(referenced_ids) == 108
    assert missing_ids == set()


def test_equipment_index_known_nodes_return_expected_tags():
    bundle = load_scenario_bundle("scenarios", "plant_topology.yaml")
    equipment_entries = normalize_equipment_entries(
        bundle["equipment"].get("equipment", []) or []
    )
    equipment_index = build_equipment_index(equipment_entries)

    def tags_for(topology_id: str):
        indices = equipment_index.get(topology_id, [])
        return {equipment_entries[idx].get("tag") for idx in indices}

    assert "PSA-H2" in tags_for("SOEC_H2_PSA_1")
    assert "KOT-3" in tags_for("SOEC_H2_KOD_1")
    assert "HP-COMP" in tags_for("HP_Compressor_S2")


def test_equipment_index_pem_soec_variant_references_only_existing_topology_ids():
    bundle = load_scenario_bundle(
        "scenarios",
        topology_file="plant_topology_PEM+SOEC.yaml",
        equipment_file="Economics/equipment_mappings_PEM+SOEC.yaml",
    )
    topology_ids = {node["id"] for node in bundle["topology"]["nodes"]}

    equipment_entries = normalize_equipment_entries(
        bundle["equipment"].get("equipment", []) or []
    )
    equipment_index = build_equipment_index(equipment_entries)

    referenced_ids = set(equipment_index.keys())
    missing_ids = referenced_ids - topology_ids

    assert len(referenced_ids) == 85
    assert missing_ids == set()
