"""
Tests for scenario visual importer model generation.
"""

import yaml

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
