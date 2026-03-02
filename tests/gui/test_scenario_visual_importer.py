"""
Tests for scenario visual importer model generation.
"""

import yaml

from h2_plant.gui.core.industrial_layout_engine import X_SPACING
from h2_plant.gui.core.scenario_visual_importer import ScenarioVisualImporter


def _topology_edge_tuples(topology_nodes):
    edges = set()
    for source_node in topology_nodes:
        source_id = source_node["id"]
        for conn in source_node.get("connections", []) or []:
            edges.add(
                (
                    source_id,
                    conn["source_port"],
                    conn["target_name"],
                    conn["target_port"],
                    conn["resource_type"],
                )
            )
    return edges


def _intra_group_max_gaps(model):
    visual_layout = dict(model.metadata.get("visual_layout") or {})
    group_map = dict(visual_layout.get("node_group_map") or {})
    x_by_group = {}
    for node in model.nodes:
        group_name = group_map.get(node.id)
        if not group_name:
            continue
        x_by_group.setdefault(group_name, []).append(float(node.x))

    gaps = {}
    for group_name, xs in x_by_group.items():
        if len(xs) < 2:
            continue
        xs_sorted = sorted(xs)
        max_gap = max(xs_sorted[idx + 1] - xs_sorted[idx] for idx in range(len(xs_sorted) - 1))
        gaps[group_name] = max_gap
    return gaps


def test_scenario_visual_importer_loads_all_nodes_and_edges():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )

    assert len(model.nodes) == 123
    assert len(model.edges) == 143


def test_scenario_visual_importer_preserves_topology_ids_and_ports():
    with open("scenarios/plant_topology.yaml", "r", encoding="utf-8") as handle:
        topology = yaml.safe_load(handle)
    topology_nodes = topology["nodes"]

    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )

    expected_node_ids = {node["id"] for node in topology_nodes}
    imported_node_ids = {node.id for node in model.nodes}
    assert imported_node_ids == expected_node_ids

    expected_edges = _topology_edge_tuples(topology_nodes)
    imported_edges = {
        (edge.source_id, edge.source_port, edge.target_id, edge.target_port, edge.resource_type)
        for edge in model.edges
    }
    assert imported_edges == expected_edges


def test_scenario_visual_importer_manifest_includes_optional_file_refs():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    manifest = model.metadata["scenario_manifest"]

    assert manifest["physics_file"] == "physics_parameters.yaml"
    assert manifest["simulation_config_file"] == "simulation_config.yaml"
    assert manifest["equipment_file"] == "Economics/equipment_mappings.yaml"
    # Optional in some scenarios, present in repo default.
    assert manifest.get("opex_file") == "Economics/opex_config.yaml"

    file_hashes = manifest.get("file_hashes", {})
    assert "plant_topology.yaml" in file_hashes
    assert "economics_parameters.yaml" in file_hashes
    assert "Economics/equipment_mappings.yaml" in file_hashes
    visual_layout = model.metadata.get("visual_layout", {})
    assert visual_layout.get("layout_mode") == "industrial_pfd_v1"
    assert int(visual_layout.get("layout_schema_version", 0)) >= 2
    assert visual_layout.get("spacing_policy") == "group_local_rank"


def test_scenario_visual_importer_supports_explicit_pem_soec_variant_files():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology_PEM+SOEC.yaml",
        equipment_file="Economics/equipment_mappings_PEM+SOEC.yaml",
    )

    assert len(model.nodes) == 95
    assert len(model.edges) == 106

    manifest = model.metadata["scenario_manifest"]
    assert manifest["topology_file"] == "plant_topology_PEM+SOEC.yaml"
    assert manifest["equipment_file"] == "Economics/equipment_mappings_PEM+SOEC.yaml"
    assert "Economics/equipment_mappings_PEM+SOEC.yaml" in manifest.get("file_hashes", {})


def test_scenario_visual_importer_infers_variant_equipment_file_from_topology():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology_PEM+SOEC.yaml",
    )
    manifest = model.metadata["scenario_manifest"]
    assert manifest["equipment_file"] == "Economics/equipment_mappings_PEM+SOEC.yaml"


def test_scenario_visual_importer_applies_industrial_row_order_for_anchor_nodes():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    node_pos = {node.id: (node.x, node.y) for node in model.nodes}

    assert node_pos["cooling_manager"][1] < node_pos["SOEC_Transformer"][1]
    assert node_pos["SOEC_Transformer"][1] < node_pos["PEM_Transformer"][1]
    assert node_pos["PEM_Transformer"][1] < node_pos["ATR_Feed_Pump"][1]


def test_scenario_visual_importer_left_to_right_flow_ordering_for_known_chains():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    node_pos = {node.id: (node.x, node.y) for node in model.nodes}

    assert node_pos["SOEC_Transformer"][0] < node_pos["SOEC_H2_Interchanger_1"][0]
    assert node_pos["SOEC_H2_Interchanger_1"][0] < node_pos["SOEC_H2_DryCooler_1"][0]

    assert node_pos["H2_Production_Mixer"][0] < node_pos["LP_Storage_Tank"][0]
    assert node_pos["LP_Storage_Tank"][0] < node_pos["Truck_Station_1"][0]


def test_scenario_visual_importer_keeps_intra_group_spacing_compact():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    gaps = _intra_group_max_gaps(model)

    assert gaps
    for group_name, max_gap in gaps.items():
        assert max_gap <= (2.0 * X_SPACING), f"group {group_name} has oversized max gap {max_gap}"


def test_scenario_visual_importer_bounds_known_problematic_pem_gaps():
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    node_pos = {node.id: (node.x, node.y) for node in model.nodes}
    node_group_map = dict(model.metadata.get("visual_layout", {}).get("node_group_map") or {})

    assert node_group_map.get("PEM_Water_Return_Valve_2") == "PEM Water/O2"
    assert node_group_map.get("PEM_O2_ElectricBoiler") == "PEM Water/O2"
    assert abs(
        node_pos["PEM_O2_ElectricBoiler"][0] - node_pos["PEM_Water_Return_Valve_2"][0]
    ) <= (2.0 * X_SPACING)

    assert node_group_map.get("PEM_H2_PSA_1") == "PEM Train"
    assert node_group_map.get("PEM_Transformer") == "PEM Train"
    assert abs(node_pos["PEM_Transformer"][0] - node_pos["PEM_H2_PSA_1"][0]) <= (2.0 * X_SPACING)
