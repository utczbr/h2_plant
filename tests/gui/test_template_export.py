"""
Focused regression tests for scenario_bundle_exporter.

Tests cover the critical paths identified in code review:
- Backend type resolution (typed nodes, fallback nodes)
- Safety guard self-blocking prevention
- Param resolution with live property overlay
- Gas resource type inference
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
import yaml

from h2_plant.gui.core.scenario_bundle_exporter import (
    _resolve_backend_type,
    _resolve_params,
    _resolve_component_id,
    _infer_resource_type,
    _assert_not_inside_template,
    TemplateSafetyError,
    export_bundle,
)
from h2_plant.gui.core.scenario_workspace import create_workspace_from_sources
from h2_plant.gui.core.scenario_visual_importer import ScenarioVisualImporter


# ---------------------------------------------------------------------------
# Helpers: lightweight node mocks
# ---------------------------------------------------------------------------

def _make_node(
    name="TestNode",
    class_name="ScenarioComponentNode",
    properties=None,
):
    """Create a mock node with the given properties."""
    node = MagicMock()
    node.name.return_value = name
    type(node).__name__ = class_name
    props = properties or {}
    node.properties.return_value = props
    # get_properties alias
    node.get_properties = node.properties
    return node


class _PortStub:
    def __init__(self, name, node=None, connected=None):
        self._name = name
        self._node = node
        self._connected = connected or []

    def name(self):
        return self._name

    def node(self):
        return self._node

    def connected_ports(self):
        return list(self._connected)


class _GraphNodeStub:
    def __init__(self, display_name, props):
        self._display_name = display_name
        self._props = props
        self._outputs = []

    def name(self):
        return self._display_name

    def properties(self):
        return self._props

    def get_properties(self):
        return self._props

    def output_ports(self):
        return list(self._outputs)


class _GraphStub:
    def __init__(self, nodes):
        self._nodes = nodes

    def all_nodes(self):
        return list(self._nodes)


def _build_graph_stub_from_visual_model(model):
    node_by_id = {}
    for visual_node in model.nodes:
        custom_props = {
            "component_id": visual_node.id,
            "__scenario_component_id": visual_node.id,
            "__scenario_backend_type": visual_node.backend_type,
            "__scenario_inputs": list(visual_node.incoming_ports),
            "__scenario_outputs": list(visual_node.outgoing_ports),
            "__scenario_params": dict(visual_node.params),
            # Keep a tuple field to validate YAML-safe serialization.
            "node_color": (100, 180, 220),
        }
        props = {
            "type_": f"nodes.{visual_node.backend_type}",
            "visible": True,
            "width": 250.0,
            "height": 100.0,
            "custom": custom_props,
        }

        # Simulate NodeGraphQt surface values that are often strings in the UI.
        for key, value in visual_node.params.items():
            if isinstance(value, (int, float, bool)):
                props[key] = str(value)
            else:
                props[key] = value

        node_by_id[visual_node.id] = _GraphNodeStub(
            display_name=f"{visual_node.backend_type}: {visual_node.id}",
            props=props,
        )

    edges_by_source_port = {}
    for edge in model.edges:
        edges_by_source_port.setdefault((edge.source_id, edge.source_port), []).append(edge)

    for (source_id, source_port), edges in edges_by_source_port.items():
        source_node = node_by_id[source_id]
        connected = []
        for edge in edges:
            target_node = node_by_id[edge.target_id]
            connected.append(_PortStub(edge.target_port, node=target_node))
        source_node._outputs.append(_PortStub(source_port, node=source_node, connected=connected))

    return _GraphStub(list(node_by_id.values()))


def _canonicalize_topology(topology_dict):
    normalized = {
        "scenario_name": topology_dict.get("scenario_name"),
        "nodes": [],
    }
    for node in topology_dict.get("nodes", []) or []:
        node_copy = dict(node)
        node_copy["connections"] = sorted(
            node_copy.get("connections", []) or [],
            key=lambda c: (
                c.get("source_port", ""),
                c.get("target_name", ""),
                c.get("target_port", ""),
                c.get("resource_type", ""),
            ),
        )
        normalized["nodes"].append(node_copy)
    normalized["nodes"] = sorted(normalized["nodes"], key=lambda n: n.get("id", ""))
    return normalized


# ---------------------------------------------------------------------------
# Fix 1: Backend type resolution
# ---------------------------------------------------------------------------

class TestBackendTypeResolution:
    """Verify _resolve_backend_type uses correct priority chain."""

    def test_explicit_scenario_backend_type(self):
        """Hidden __scenario_backend_type should be first priority."""
        node = _make_node(
            name="PowerTransformer: SOEC_Rectifier_1",
            class_name="RectifierNode",
            properties={"__scenario_backend_type": "PowerTransformer"},
        )
        assert _resolve_backend_type(node) == "PowerTransformer"

    def test_backend_type_from_custom_payload(self):
        node = _make_node(
            class_name="RectifierNode",
            properties={"custom": {"__scenario_backend_type": "PowerTransformer"}},
        )
        assert _resolve_backend_type(node) == "PowerTransformer"

    def test_class_based_reverse_lookup(self):
        """Without hidden property, class name should resolve correctly."""
        node = _make_node(
            name="PowerTransformer: SOEC_Rectifier_1",
            class_name="RectifierNode",
            properties={},
        )
        assert _resolve_backend_type(node) == "PowerTransformer"

    def test_soec_typed_node(self):
        node = _make_node(name="SOEC: SOEC_Stack_1", class_name="SOECStackNode", properties={})
        assert _resolve_backend_type(node) == "SOEC"

    def test_pem_typed_node(self):
        node = _make_node(name="PEM: PEM_Stack_1", class_name="PEMStackNode", properties={})
        assert _resolve_backend_type(node) == "PEM"

    def test_psa_typed_node(self):
        node = _make_node(name="PSA Unit: PSA_1", class_name="PSAUnitNode", properties={})
        assert _resolve_backend_type(node) == "PSA Unit"

    def test_fallback_never_splits_name(self):
        """Unknown class should return 'Unknown', not name.split('_')[0]."""
        node = _make_node(
            name="SomeWeird: Name_Here",
            class_name="SomeUnknownNodeClass",
            properties={},
        )
        result = _resolve_backend_type(node)
        # Must NOT be "SomeWeird: Name" or any split artifact
        assert result == "Unknown"
        assert ":" not in result

    def test_scenario_component_fallback(self):
        node = _make_node(name="Fallback_Node", class_name="ScenarioComponentNode", properties={})
        assert _resolve_backend_type(node) == "ScenarioComponent"

    @pytest.mark.parametrize("class_name,expected_type", [
        # Electrolysis / Production (3)
        ("PEMStackNode", "PEM"),
        ("SOECStackNode", "SOEC"),
        ("RectifierNode", "PowerTransformer"),
        # Thermal (6)
        ("ChillerNode", "Chiller"),
        ("DryCoolerNode", "DryCooler"),
        ("InterchangerNode", "Interchanger"),
        ("ElectricBoilerNode", "ElectricBoiler"),
        ("AttemperatorNode", "Attemperator"),
        ("CoolingManagerNode", "CoolingManager"),
        # Separation (7)
        ("CoalescerNode", "Coalescer"),
        ("KnockOutDrumNode", "KnockOutDrum"),
        ("PSAUnitNode", "PSA Unit"),
        ("DeoxoReactorNode", "DeoxoReactor"),
        ("HydrogenMultiCycloneNode", "HydrogenMultiCyclone"),
        ("SeparationTankNode", "SeparationTank"),
        ("SyngasPSANode", "SyngasPSA"),
        # Mixing / Flow (7)
        ("MixerNode", "Mixer"),
        ("ValveNode", "Valve"),
        ("StreamSplitterNode", "StreamSplitter"),
        ("DrainRecorderMixerNode", "DrainRecorderMixer"),
        ("SignalMakeupMixerNode", "SignalMakeupMixer"),
        ("ProportionalMakeupMixerNode", "ProportionalMakeupMixer"),
        ("OxygenMakeupNode", "OxygenMakeupNode"),
        # Water (4)
        ("WaterPurifierNode", "WaterPurifier"),
        ("UltraPureWaterTankNode", "UltraPureWaterTank"),
        ("ExternalWaterSourceNode", "ExternalWaterSource"),
        ("WaterPumpThermodynamicNode", "WaterPumpThermodynamic"),
        # Storage / Delivery (3)
        ("DetailedTankNode", "DetailedTank"),
        ("DischargeStationNode", "DischargeStation"),
        ("CompressorSingleNode", "CompressorSingle"),
        # Reforming (3)
        ("IntegratedATRPlantNode", "IntegratedATRPlant"),
        ("ATRBoilerNode", "ATR_Boiler"),
        ("BiogasSourceNode", "BiogasSource"),
    ])
    def test_all_typed_class_mappings(self, class_name, expected_type):
        """All 33 typed palette node classes must resolve to correct backend types."""
        node = _make_node(name="Test", class_name=class_name, properties={})
        assert _resolve_backend_type(node) == expected_type


# ---------------------------------------------------------------------------
# Fix 2: Safety guard self-blocking
# ---------------------------------------------------------------------------

class TestSafetyGuard:
    """Verify safety guard allows re-export but blocks inside template."""

    def test_same_dir_allowed(self, tmp_path):
        """Re-exporting to the same dir should NOT raise."""
        _assert_not_inside_template(tmp_path, tmp_path)  # should not raise

    def test_different_dir_allowed(self, tmp_path):
        output = tmp_path / "generated"
        output.mkdir()
        template = tmp_path / "scenarios"
        template.mkdir()
        _assert_not_inside_template(output, template)  # should not raise

    def test_inside_template_blocked(self, tmp_path):
        template = tmp_path / "scenarios"
        template.mkdir()
        inside = template / "sub_bundle"
        inside.mkdir()
        with pytest.raises(TemplateSafetyError):
            _assert_not_inside_template(inside, template)


# ---------------------------------------------------------------------------
# Fix 3: Param resolution with live overlay
# ---------------------------------------------------------------------------

class TestParamResolution:
    """Verify _resolve_params merges hidden dict with live properties."""

    def test_hidden_params_only(self):
        """When no overlay, hidden params are returned as-is."""
        node = _make_node(properties={
            "__scenario_params": {"capacity_mw": 10.0, "efficiency": 0.65},
        })
        result = _resolve_params(node)
        assert result == {"capacity_mw": 10.0, "efficiency": 0.65}

    def test_live_overlay_captures_edits(self):
        """Visible property edits overlay the hidden snapshot."""
        node = _make_node(properties={
            "__scenario_params": {"capacity_mw": 10.0, "efficiency": 0.65},
            "capacity_mw": 25.0,  # user edited this in the UI
        })
        result = _resolve_params(node)
        assert result["capacity_mw"] == 25.0
        assert result["efficiency"] == 0.65

    def test_live_overlay_with_equivalent_string_keeps_original_type(self):
        node = _make_node(properties={
            "__scenario_params": {"capacity_mw": 10.0},
            "capacity_mw": "10.0",
        })
        result = _resolve_params(node)
        assert isinstance(result["capacity_mw"], float)
        assert result["capacity_mw"] == 10.0

    def test_live_overlay_maps_gui_keys_back_to_backend_keys(self):
        node = _make_node(
            class_name="RectifierNode",
            properties={
                "__scenario_backend_type": "PowerTransformer",
                "__scenario_params": {
                    "rated_power_mw": 15.25,
                    "efficiency": 0.95,
                    "system_group": "SOEC",
                    "process_step": 8,
                },
                "max_power_kw": "16000",
                "conversion_efficiency": "98",
                "system_group": "PEM",
            },
        )
        result = _resolve_params(node)
        assert result["rated_power_mw"] == pytest.approx(16.0)
        assert result["efficiency"] == pytest.approx(0.98)
        assert result["system_group"] == "PEM"
        assert result["process_step"] == 8

    def test_empty_params_with_no_visible_props(self):
        """Node with no __scenario_params and no visible props returns {}."""
        node = _make_node(properties={})
        assert _resolve_params(node) == {}

    def test_fallback_collects_visible_properties(self):
        """Without __scenario_params, visible non-framework properties are collected."""
        node = _make_node(properties={
            "capacity_mw": 10.0,
            "efficiency": 0.65,
            "name": "should_be_excluded",
            "color": (255, 0, 0),
            "node_color": (255, 0, 0),
        })
        result = _resolve_params(node)
        assert result == {"capacity_mw": 10.0, "efficiency": 0.65}
        assert "name" not in result
        assert "color" not in result
        assert "node_color" not in result

    def test_fallback_excludes_hidden_properties(self):
        """Properties starting with _ or __ should be excluded in fallback."""
        node = _make_node(properties={
            "capacity_mw": 10.0,
            "__scenario_backend_type": "SOEC",
            "_internal_state": "active",
        })
        result = _resolve_params(node)
        assert result == {"capacity_mw": 10.0}

    def test_fallback_excludes_none_values(self):
        """None values should be excluded in fallback."""
        node = _make_node(properties={
            "capacity_mw": 10.0,
            "unused_param": None,
        })
        result = _resolve_params(node)
        assert result == {"capacity_mw": 10.0}


# ---------------------------------------------------------------------------
# Fix 7: Gas resource type inference
# ---------------------------------------------------------------------------

class TestResourceTypeInference:
    """Verify _infer_resource_type handles all expected port names."""

    @pytest.mark.parametrize("port_name,expected", [
        ("gas_out", "gas"),
        ("tail_gas_out", "gas"),
        ("syngas_output", "gas"),
        ("feed_gas", "gas"),
        ("h2_out", "hydrogen"),
        ("power_in", "electricity"),
        ("water_in", "water"),
        ("steam_out", "water"),
        ("heat_out", "heat"),
        ("o2_vent", "oxygen"),
        ("control_signal", "signal"),
        ("unknown_port", "stream"),
    ])
    def test_inference(self, port_name, expected):
        assert _infer_resource_type(port_name) == expected


# ---------------------------------------------------------------------------
# Component ID resolution
# ---------------------------------------------------------------------------

class TestComponentIdResolution:
    def test_explicit_component_id(self):
        node = _make_node(properties={"component_id": "SOEC_Stack_1"})
        assert _resolve_component_id(node) == "SOEC_Stack_1"

    def test_scenario_component_id(self):
        node = _make_node(properties={"__scenario_component_id": "ATR_Unit_1"})
        assert _resolve_component_id(node) == "ATR_Unit_1"

    def test_component_id_from_custom_payload(self):
        node = _make_node(properties={"custom": {"component_id": "Cooling_Manager_1"}})
        assert _resolve_component_id(node) == "Cooling_Manager_1"

    def test_fallback_to_name(self):
        node = _make_node(name="MyNode", properties={})
        assert _resolve_component_id(node) == "MyNode"


# ---------------------------------------------------------------------------
# Integration: export_bundle minimal roundtrip
# ---------------------------------------------------------------------------

class TestExportBundleRoundtrip:
    """Verify export_bundle generates expected files without crashing."""

    def _make_graph_with_nodes(self, tmp_path):
        """Create a mock graph with two connected nodes."""
        # Template dir with required files
        template_dir = tmp_path / "template"
        template_dir.mkdir()
        (template_dir / "physics_parameters.yaml").write_text("dt: 1.0\n")
        (template_dir / "simulation_config.yaml").write_text("hours: 24\n")

        # Nodes
        soec = _make_node(
            name="SOEC: SOEC_Stack_1",
            class_name="SOECStackNode",
            properties={
                "__scenario_backend_type": "SOEC",
                "component_id": "SOEC_Stack_1",
                "__scenario_params": {"capacity_mw": 10.0},
            },
        )
        psa = _make_node(
            name="PSA Unit: PSA_1",
            class_name="PSAUnitNode",
            properties={
                "__scenario_backend_type": "PSA Unit",
                "component_id": "PSA_1",
                "__scenario_params": {"recovery": 0.85},
            },
        )

        # Port connections: SOEC h2_out → PSA h2_in
        out_port = MagicMock()
        out_port.name.return_value = "h2_out"
        in_port = MagicMock()
        in_port.name.return_value = "h2_in"
        in_port.node.return_value = psa
        out_port.connected_ports.return_value = [in_port]
        soec.output_ports.return_value = [out_port]
        psa.output_ports.return_value = []

        graph = MagicMock()
        graph.all_nodes.return_value = [soec, psa]
        return graph, template_dir

    def test_bundle_files_created(self, tmp_path):
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"

        manifest = export_bundle(
            graph=graph,
            template_manifest={"scenarios_dir": str(template_dir)},
            economics={"h2_price_eur_kg": 9.6},
            output_dir=output_dir,
        )

        assert (output_dir / "plant_topology.yaml").exists()
        assert (output_dir / "economics_parameters.yaml").exists()
        assert (output_dir / "physics_parameters.yaml").exists()
        assert (output_dir / "simulation_config.yaml").exists()
        assert (output_dir / "bundle_manifest.json").exists()
        assert manifest["bundle_dir"] == str(output_dir)

    def test_topology_contains_correct_types(self, tmp_path):
        """Exported topology YAML must have correct backend types."""
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"

        export_bundle(
            graph=graph,
            template_manifest={"scenarios_dir": str(template_dir)},
            economics={},
            output_dir=output_dir,
        )

        with open(output_dir / "plant_topology.yaml") as f:
            topo = yaml.safe_load(f)

        nodes_by_id = {n["id"]: n for n in topo["nodes"]}
        assert nodes_by_id["SOEC_Stack_1"]["type"] == "SOEC"
        assert nodes_by_id["PSA_1"]["type"] == "PSA Unit"

    def test_topology_contains_connections(self, tmp_path):
        """Exported topology YAML must have connections."""
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"

        export_bundle(
            graph=graph,
            template_manifest={"scenarios_dir": str(template_dir)},
            economics={},
            output_dir=output_dir,
        )

        with open(output_dir / "plant_topology.yaml") as f:
            topo = yaml.safe_load(f)

        soec_node = [n for n in topo["nodes"] if n["id"] == "SOEC_Stack_1"][0]
        assert len(soec_node["connections"]) == 1
        conn = soec_node["connections"][0]
        assert conn["source_port"] == "h2_out"
        assert conn["target_name"] == "PSA_1"
        assert conn["target_port"] == "h2_in"
        assert conn["resource_type"] == "hydrogen"

    def test_exported_topology_uses_safe_yaml_without_python_tags(self, tmp_path):
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"

        # Inject tuple in custom payload to verify no !!python/tuple appears.
        graph.all_nodes.return_value[0].properties.return_value["custom"] = {
            "__scenario_backend_type": "SOEC",
            "component_id": "SOEC_Stack_1",
            "__scenario_params": {"capacity_mw": 10.0},
            "node_color": (100, 180, 220),
        }

        export_bundle(
            graph=graph,
            template_manifest={"scenarios_dir": str(template_dir)},
            economics={},
            output_dir=output_dir,
        )

        topology_text = (output_dir / "plant_topology.yaml").read_text(encoding="utf-8")
        assert "!!python/" not in topology_text
        with open(output_dir / "plant_topology.yaml", "r", encoding="utf-8") as handle:
            yaml.safe_load(handle)

    def test_resource_type_preserved_from_source_topology(self, tmp_path):
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"
        source_topology = {
            "scenario_name": "Template Scenario",
            "nodes": [
                {
                    "id": "SOEC_Stack_1",
                    "type": "SOEC",
                    "connections": [
                        {
                            "source_port": "h2_out",
                            "target_name": "PSA_1",
                            "target_port": "h2_in",
                            "resource_type": "stream",
                        }
                    ],
                },
                {"id": "PSA_1", "type": "PSA Unit"},
            ],
        }
        with open(template_dir / "plant_topology.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(source_topology, handle, sort_keys=False)

        export_bundle(
            graph=graph,
            template_manifest={
                "scenarios_dir": str(template_dir),
                "topology_file": "plant_topology.yaml",
            },
            economics={},
            output_dir=output_dir,
            scenario_name=None,
        )

        with open(output_dir / "plant_topology.yaml", "r", encoding="utf-8") as handle:
            topo = yaml.safe_load(handle)

        soec_node = [n for n in topo["nodes"] if n["id"] == "SOEC_Stack_1"][0]
        assert soec_node["connections"][0]["resource_type"] == "stream"
        assert topo["scenario_name"] == "Template Scenario"

    def test_re_export_to_same_dir_succeeds(self, tmp_path):
        """Re-exporting to the generated bundle dir should not raise."""
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"

        # First export
        export_bundle(
            graph=graph,
            template_manifest={"scenarios_dir": str(template_dir)},
            economics={},
            output_dir=output_dir,
        )
        # Second export to same dir (re-export after save)
        manifest = export_bundle(
            graph=graph,
            template_manifest={"scenarios_dir": str(output_dir)},  # now pointing at itself
            economics={},
            output_dir=output_dir,
        )
        assert manifest["bundle_dir"] == str(output_dir)

    def test_duplicate_component_id_raises(self, tmp_path):
        """Duplicate component_id should raise ValueError."""
        node1 = _make_node(properties={"component_id": "DuplicateID"})
        node1.output_ports.return_value = []
        node2 = _make_node(properties={"component_id": "DuplicateID"})
        node2.output_ports.return_value = []

        graph = MagicMock()
        graph.all_nodes.return_value = [node1, node2]

        # Use separate template and output dirs to avoid safety guard interference
        template_dir = tmp_path / "template"
        template_dir.mkdir()
        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="Duplicate component_id"):
            export_bundle(
                graph=graph,
                template_manifest={"scenarios_dir": str(template_dir)},
                economics={},
                output_dir=output_dir,
            )

    def test_export_localizes_simulation_data_paths_and_copies_csvs(self, tmp_path):
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"

        (template_dir / "prices.csv").write_text("t,price\n0,50\n1,51\n", encoding="utf-8")
        (template_dir / "wind.csv").write_text("t,wind\n0,1\n1,2\n", encoding="utf-8")
        (template_dir / "simulation_config.yaml").write_text(
            "timestep_hours: 1\n"
            "duration_hours: 24\n"
            "start_hour: 0\n"
            "checkpoint_interval_hours: 24\n"
            "energy_price_file: prices.csv\n"
            "wind_data_file: wind.csv\n"
            "dispatch_strategy: ECONOMIC_SPOT\n"
            "storage_control_mode: SCHMITT_TRIGGER\n",
            encoding="utf-8",
        )

        export_bundle(
            graph=graph,
            template_manifest={
                "scenarios_dir": str(template_dir),
                "simulation_config_file": "simulation_config.yaml",
            },
            economics={},
            output_dir=output_dir,
        )

        exported_sim = yaml.safe_load((output_dir / "simulation_config.yaml").read_text(encoding="utf-8"))
        assert exported_sim["energy_price_file"] == "data/prices.csv"
        assert exported_sim["wind_data_file"] == "data/wind.csv"
        assert (output_dir / "data" / "prices.csv").exists()
        assert (output_dir / "data" / "wind.csv").exists()

    def test_export_resolves_data_files_via_source_scenarios_dir(self, tmp_path):
        graph, _ = self._make_graph_with_nodes(tmp_path)
        project_root = tmp_path / "project"
        source_dir = project_root / "scenarios"
        source_dir.mkdir(parents=True, exist_ok=True)

        data_dir = project_root / "h2_plant" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "NL_Prices_2024_15min.csv").write_text("t,price\n0,50\n1,51\n", encoding="utf-8")
        (data_dir / "producao_horaria_turbina.csv").write_text("t,wind\n0,1\n1,2\n", encoding="utf-8")

        (source_dir / "plant_topology.yaml").write_text(
            "scenario_name: Source Scenario\nnodes: []\n",
            encoding="utf-8",
        )
        (source_dir / "physics_parameters.yaml").write_text("dt: 1.0\n", encoding="utf-8")
        (source_dir / "economics_parameters.yaml").write_text("h2_price_eur_kg: 9.6\n", encoding="utf-8")
        (source_dir / "simulation_config.yaml").write_text(
            "timestep_hours: 1\n"
            "duration_hours: 24\n"
            "start_hour: 0\n"
            "checkpoint_interval_hours: 24\n"
            "energy_price_file: ../h2_plant/data/NL_Prices_2024_15min.csv\n"
            "wind_data_file: ../h2_plant/data/producao_horaria_turbina.csv\n"
            "dispatch_strategy: ECONOMIC_SPOT\n"
            "storage_control_mode: SCHMITT_TRIGGER\n",
            encoding="utf-8",
        )

        staged_manifest = create_workspace_from_sources(
            {"scenarios_dir": str(source_dir)},
            workspace_root=tmp_path / "generated",
        )
        output_dir = tmp_path / "output_bundle_from_workspace"
        export_bundle(
            graph=graph,
            template_manifest=staged_manifest,
            economics={},
            output_dir=output_dir,
        )

        exported_sim = yaml.safe_load((output_dir / "simulation_config.yaml").read_text(encoding="utf-8"))
        assert exported_sim["energy_price_file"] == "data/NL_Prices_2024_15min.csv"
        assert exported_sim["wind_data_file"] == "data/producao_horaria_turbina.csv"
        assert (output_dir / "data" / "NL_Prices_2024_15min.csv").exists()
        assert (output_dir / "data" / "producao_horaria_turbina.csv").exists()

    def test_export_legacy_fallback_resolves_data_files_without_source_scenarios_dir(
        self, tmp_path
    ):
        graph, _ = self._make_graph_with_nodes(tmp_path)
        project_root = tmp_path / "legacy_project"
        (project_root / "scenarios").mkdir(parents=True, exist_ok=True)

        data_dir = project_root / "h2_plant" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "NL_Prices_2024_15min.csv").write_text("t,price\n0,50\n", encoding="utf-8")
        (data_dir / "producao_horaria_turbina.csv").write_text("t,wind\n0,1\n", encoding="utf-8")

        session_dir = project_root / "h2_plant" / "gui" / "layouts" / "generated" / "session_legacy"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "physics_parameters.yaml").write_text("dt: 1.0\n", encoding="utf-8")
        (session_dir / "simulation_config.yaml").write_text(
            "timestep_hours: 1\n"
            "duration_hours: 24\n"
            "start_hour: 0\n"
            "checkpoint_interval_hours: 24\n"
            "energy_price_file: ../h2_plant/data/NL_Prices_2024_15min.csv\n"
            "wind_data_file: ../h2_plant/data/producao_horaria_turbina.csv\n"
            "dispatch_strategy: ECONOMIC_SPOT\n"
            "storage_control_mode: SCHMITT_TRIGGER\n",
            encoding="utf-8",
        )

        output_dir = tmp_path / "legacy_output_bundle"
        export_bundle(
            graph=graph,
            template_manifest={"scenarios_dir": str(session_dir)},
            economics={},
            output_dir=output_dir,
        )

        exported_sim = yaml.safe_load((output_dir / "simulation_config.yaml").read_text(encoding="utf-8"))
        assert exported_sim["energy_price_file"] == "data/NL_Prices_2024_15min.csv"
        assert exported_sim["wind_data_file"] == "data/producao_horaria_turbina.csv"
        assert (output_dir / "data" / "NL_Prices_2024_15min.csv").exists()
        assert (output_dir / "data" / "producao_horaria_turbina.csv").exists()

    def test_export_copies_selected_opex_file(self, tmp_path):
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"
        econ_dir = template_dir / "Economics"
        econ_dir.mkdir(exist_ok=True)
        (econ_dir / "opex_config.yaml").write_text(
            "scenario_name: test\nopex_items: []\n",
            encoding="utf-8",
        )
        (template_dir / "prices.csv").write_text("t,price\n0,50\n", encoding="utf-8")
        (template_dir / "wind.csv").write_text("t,wind\n0,1\n", encoding="utf-8")
        (template_dir / "simulation_config.yaml").write_text(
            "timestep_hours: 1\n"
            "duration_hours: 24\n"
            "start_hour: 0\n"
            "checkpoint_interval_hours: 24\n"
            "energy_price_file: prices.csv\n"
            "wind_data_file: wind.csv\n"
            "dispatch_strategy: ECONOMIC_SPOT\n"
            "storage_control_mode: SCHMITT_TRIGGER\n",
            encoding="utf-8",
        )

        manifest = export_bundle(
            graph=graph,
            template_manifest={
                "scenarios_dir": str(template_dir),
                "opex_file": "Economics/opex_config.yaml",
            },
            economics={},
            output_dir=output_dir,
        )

        assert (output_dir / "Economics" / "opex_config.yaml").exists()
        assert "Economics/opex_config.yaml" in manifest["files"]["copied"]

    def test_export_raises_when_simulation_data_reference_missing(self, tmp_path):
        graph, template_dir = self._make_graph_with_nodes(tmp_path)
        output_dir = tmp_path / "output_bundle"
        (template_dir / "simulation_config.yaml").write_text(
            "timestep_hours: 1\n"
            "duration_hours: 24\n"
            "start_hour: 0\n"
            "checkpoint_interval_hours: 24\n"
            "energy_price_file: missing_prices.csv\n"
            "wind_data_file: missing_wind.csv\n"
            "dispatch_strategy: ECONOMIC_SPOT\n"
            "storage_control_mode: SCHMITT_TRIGGER\n",
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="references missing file"):
            export_bundle(
                graph=graph,
                template_manifest={"scenarios_dir": str(template_dir)},
                economics={},
                output_dir=output_dir,
            )


def test_semantic_parity_no_edit_prebuilt_export(tmp_path):
    baseline_path = Path("scenarios/plant_topology.yaml")
    with open(baseline_path, "r", encoding="utf-8") as handle:
        baseline_topology = yaml.safe_load(handle)

    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    graph = _build_graph_stub_from_visual_model(model)

    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "physics_parameters.yaml").write_text("pem_system: {}\nsoec_cluster: {}\n", encoding="utf-8")
    (template_dir / "prices.csv").write_text("t,price\n0,50\n", encoding="utf-8")
    (template_dir / "wind.csv").write_text("t,wind\n0,1\n", encoding="utf-8")
    (template_dir / "simulation_config.yaml").write_text("timestep_hours: 1\nduration_hours: 24\nenergy_price_file: prices.csv\nwind_data_file: wind.csv\n", encoding="utf-8")
    (template_dir / "plant_topology.yaml").write_text(
        baseline_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    econ_dir = template_dir / "Economics"
    econ_dir.mkdir()
    (econ_dir / "equipment_mappings.yaml").write_text("equipment: []\n", encoding="utf-8")

    output_dir = tmp_path / "output_bundle"
    export_bundle(
        graph=graph,
        template_manifest={
            "scenarios_dir": str(template_dir),
            "topology_file": "plant_topology.yaml",
        },
        economics={},
        output_dir=output_dir,
        scenario_name=None,
    )

    with open(output_dir / "plant_topology.yaml", "r", encoding="utf-8") as handle:
        exported_topology = yaml.safe_load(handle)

    assert _canonicalize_topology(exported_topology) == _canonicalize_topology(baseline_topology)


def test_semantic_parity_physics_config_no_edit(tmp_path):
    """Physics config must be copied verbatim (semantic equality) on no-edit round-trip."""
    physics_content = "pem_system:\n  max_power_mw: 10\nsoec_cluster:\n  max_power_mw: 5\n"

    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    graph = _build_graph_stub_from_visual_model(model)

    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "plant_topology.yaml").write_text(
        Path("scenarios/plant_topology.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (template_dir / "physics_parameters.yaml").write_text(physics_content, encoding="utf-8")
    (template_dir / "prices.csv").write_text("t,price\n0,50\n", encoding="utf-8")
    (template_dir / "wind.csv").write_text("t,wind\n0,1\n", encoding="utf-8")
    (template_dir / "simulation_config.yaml").write_text(
        "timestep_hours: 1\nduration_hours: 24\nenergy_price_file: prices.csv\nwind_data_file: wind.csv\n",
        encoding="utf-8",
    )
    econ_dir = template_dir / "Economics"
    econ_dir.mkdir()
    (econ_dir / "equipment_mappings.yaml").write_text("equipment: []\n", encoding="utf-8")

    output_dir = tmp_path / "bundle"
    export_bundle(
        graph=graph,
        template_manifest={"scenarios_dir": str(template_dir), "topology_file": "plant_topology.yaml"},
        economics={},
        output_dir=output_dir,
        scenario_name=None,
    )

    exported_physics = yaml.safe_load((output_dir / "physics_parameters.yaml").read_text(encoding="utf-8"))
    baseline_physics = yaml.safe_load(physics_content)
    assert exported_physics == baseline_physics, "physics_parameters.yaml must be semantically identical after no-edit round-trip"


def test_semantic_parity_simulation_config_no_edit_option_a(tmp_path):
    """Simulation config non-path keys must be identical after no-edit round-trip.

    Data file paths are rewritten (Option A) to relative bundle paths; the test
    verifies only that the file basenames are preserved and all other keys match.
    """
    sim_content = "timestep_hours: 1\nduration_hours: 24\nenergy_price_file: prices.csv\nwind_data_file: wind.csv\nsome_flag: true\n"

    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir="scenarios",
        topology_file="plant_topology.yaml",
    )
    graph = _build_graph_stub_from_visual_model(model)

    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "plant_topology.yaml").write_text(
        Path("scenarios/plant_topology.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (template_dir / "physics_parameters.yaml").write_text("pem_system: {}\nsoec_cluster: {}\n", encoding="utf-8")
    (template_dir / "prices.csv").write_text("t,price\n0,50\n", encoding="utf-8")
    (template_dir / "wind.csv").write_text("t,wind\n0,1\n", encoding="utf-8")
    (template_dir / "simulation_config.yaml").write_text(sim_content, encoding="utf-8")
    econ_dir = template_dir / "Economics"
    econ_dir.mkdir()
    (econ_dir / "equipment_mappings.yaml").write_text("equipment: []\n", encoding="utf-8")

    output_dir = tmp_path / "bundle"
    export_bundle(
        graph=graph,
        template_manifest={"scenarios_dir": str(template_dir), "topology_file": "plant_topology.yaml"},
        economics={},
        output_dir=output_dir,
        scenario_name=None,
    )

    baseline_sim = yaml.safe_load(sim_content)
    exported_sim = yaml.safe_load((output_dir / "simulation_config.yaml").read_text(encoding="utf-8"))

    # Non-path keys must be identical
    data_file_keys = {"energy_price_file", "wind_data_file"}
    for key in baseline_sim:
        if key in data_file_keys:
            continue
        assert exported_sim.get(key) == baseline_sim[key], f"simulation_config key '{key}' changed in no-edit round-trip"

    # Data file basenames must be preserved (paths are localised to data/ but names kept)
    for file_key in data_file_keys:
        if baseline_sim.get(file_key):
            baseline_name = Path(str(baseline_sim[file_key])).name
            exported_path = str(exported_sim.get(file_key, ""))
            assert exported_path.endswith(baseline_name), (
                f"Data file basename for '{file_key}' changed: {baseline_sim[file_key]} → {exported_path}"
            )


def test_no_injection_of_absent_keys_on_no_edit_roundtrip(tmp_path):
    """Keys absent from the canonical __scenario_params must not appear in export.

    Regression guard for base_efficiency, volume_m3, and similar keys that
    typed GUI nodes expose as defaults but that were not in the source topology.
    """
    from h2_plant.gui.core.scenario_bundle_exporter import _resolve_params

    # Simulate an imported PEM node whose canonical params do NOT include base_efficiency
    canonical_params = {"max_power_mw": 5.0, "lifecycle": 87600}
    node = _make_node(
        class_name="PEMStackNode",
        properties={
            "__scenario_params": canonical_params,
            "__scenario_backend_type": "PEM",
            # GUI shows rated_power_kw (maps to max_power_mw) and efficiency_rated
            # (maps to base_efficiency — which is NOT in canonical_params)
            "rated_power_kw": 5000.0,   # == round-trip of 5.0 MW
            "efficiency_rated": 75.0,   # maps to base_efficiency, absent in canonical
        },
    )
    type(node).__name__ = "PEMStackNode"

    result = _resolve_params(node)

    assert "max_power_mw" in result, "canonical key must be present"
    assert "base_efficiency" not in result, (
        "base_efficiency must NOT be injected when absent from canonical __scenario_params"
    )
    assert "lifecycle" in result, "other canonical keys must be preserved"


def test_no_injection_of_volume_m3_when_absent_from_canonical(tmp_path):
    """Regression: volume_m3 must not appear in export if absent from source topology."""
    from h2_plant.gui.core.scenario_bundle_exporter import _resolve_params

    # SeparationTankNode exposes volume_m3 as a default GUI property, but
    # the canonical params may not include it (older topology without it).
    canonical_params = {"efficiency": 1.0}
    node = _make_node(
        class_name="SeparationTankNode",
        properties={
            "__scenario_params": canonical_params,
            "__scenario_backend_type": "SeparationTank",
            "volume_m3": 2.0,   # GUI default — NOT a real user edit if absent from canonical
            "efficiency": 100.0,  # round-trip of 1.0 fraction × 100
        },
    )
    type(node).__name__ = "SeparationTankNode"

    result = _resolve_params(node)

    assert "efficiency" in result
    assert "volume_m3" not in result, (
        "volume_m3 must NOT be injected when absent from canonical __scenario_params"
    )
