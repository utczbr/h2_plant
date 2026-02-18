"""
Utilities to generate a prebuilt `.h2plant` visual layout from scenario YAML files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import gettempdir
from typing import Optional, Tuple

from h2_plant.gui.core.graph_persistence import (
    CanvasState,
    GraphPersistenceManager,
    GraphSnapshot,
    ProjectMetadata,
)
from h2_plant.gui.core.scenario_param_mapper import (
    IMPORT_SURFACE_SCHEMA_VERSION,
    backend_to_gui_props,
)
from h2_plant.gui.core.scenario_visual_importer import ScenarioVisualImporter


def _typed_backend_map() -> dict[str, str]:
    return {
        # Electrolysis / Production
        "PEM": "nodes.Production.PEMStackNode",
        "SOEC": "nodes.Production.SOECStackNode",
        "PowerTransformer": "nodes.Production.RectifierNode",
        # Thermal
        "Chiller": "nodes.Thermal.ChillerNode",
        "DryCooler": "nodes.Thermal.DryCoolerNode",
        "Interchanger": "nodes.Thermal.InterchangerNode",
        "ElectricBoiler": "nodes.Thermal.ElectricBoilerNode",
        "Attemperator": "nodes.Thermal.AttemperatorNode",
        "CoolingManager": "nodes.Thermal.CoolingManagerNode",
        # Separation
        "Coalescer": "nodes.Separation.CoalescerNode",
        "KnockOutDrum": "nodes.Separation.KnockOutDrumNode",
        "PSA Unit": "nodes.Separation.PSAUnitNode",
        "DeoxoReactor": "nodes.Separation.DeoxoReactorNode",
        "HydrogenMultiCyclone": "nodes.Separation.HydrogenMultiCycloneNode",
        "SeparationTank": "nodes.Separation.SeparationTankNode",
        "SyngasPSA": "nodes.Separation.SyngasPSANode",
        # Mixing / Flow
        "Mixer": "nodes.Flow.MixerNode",
        "Valve": "nodes.Flow.ValveNode",
        "StreamSplitter": "nodes.Flow.StreamSplitterNode",
        "DrainRecorderMixer": "nodes.Flow.DrainRecorderMixerNode",
        "SignalMakeupMixer": "nodes.Flow.SignalMakeupMixerNode",
        "ProportionalMakeupMixer": "nodes.Flow.ProportionalMakeupMixerNode",
        "OxygenMakeupNode": "nodes.Flow.OxygenMakeupNode",
        # Water
        "WaterPurifier": "h2_plant.water.WaterPurifierNode",
        "UltraPureWaterTank": "h2_plant.water_tank.UltraPureWaterTankNode",
        "ExternalWaterSource": "h2_plant.water.ExternalWaterSourceNode",
        "WaterPumpThermodynamic": "h2_plant.water.WaterPumpThermodynamicNode",
        # Storage / Delivery
        "DetailedTank": "nodes.Storage.DetailedTankNode",
        "DischargeStation": "nodes.Storage.DischargeStationNode",
        "CompressorSingle": "nodes.Storage.CompressorSingleNode",
        # Reforming
        "IntegratedATRPlant": "nodes.Reforming.IntegratedATRPlantNode",
        "ATR_Boiler": "nodes.Reforming.ATRBoilerNode",
        "BiogasSource": "nodes.Reforming.BiogasSourceNode",
    }


def build_snapshot(project_name: str, model) -> GraphSnapshot:
    """Build a GraphSnapshot from a ScenarioVisualImporter model."""
    typed_map = _typed_backend_map()
    fallback_type = "nodes.Scenario.ScenarioComponentNode"

    nodes = {}
    for node in model.nodes:
        node_type = typed_map.get(node.backend_type, fallback_type)
        display_name = f"{node.backend_type}: {node.id}"
        properties = {
            "component_id": node.id,
            "__scenario_inputs": list(node.incoming_ports),
            "__scenario_outputs": list(node.outgoing_ports),
            "__scenario_params": dict(node.params),
            "__scenario_unmapped_params": {},
        }
        if node_type == fallback_type:
            properties["backend_type"] = node.backend_type
            properties["__scenario_component_id"] = node.id
            properties["__scenario_backend_type"] = node.backend_type
        else:
            mapped_props, unmapped = backend_to_gui_props(
                backend_type=node.backend_type,
                backend_params=dict(node.params),
                available_props=None,
            )
            for prop_name, prop_value in mapped_props.items():
                if prop_name.startswith("__") or prop_name == "component_id":
                    continue
                properties[prop_name] = prop_value
            properties["__scenario_backend_type"] = node.backend_type
            properties["__scenario_unmapped_params"] = unmapped

        nodes[node.id] = {
            "type": node_type,
            "display_name": display_name,
            "properties": properties,
            "geometry": {
                "x": float(node.x),
                "y": float(node.y),
                "width": 100.0,
                "height": 100.0,
                "color": [100, 100, 100],
                "border_color": [50, 50, 50],
                "text_color": [255, 255, 255],
                "selected": False,
                "disabled": False,
                "collapsed": False,
            },
        }

    edges = []
    for edge in model.edges:
        edges.append(
            {
                "source_node_id": edge.source_id,
                "target_node_id": edge.target_id,
                "source_port": edge.source_port,
                "target_port": edge.target_port,
                "flow_type": edge.resource_type,
                "geometry": {
                    "source_node_id": edge.source_id,
                    "target_node_id": edge.target_id,
                    "source_port": edge.source_port,
                    "target_port": edge.target_port,
                    "flow_type": edge.resource_type,
                    "color": [255, 255, 255],
                    "width": 2.0,
                    "style": "solid",
                    "selected": False,
                    "waypoints": [],
                },
            }
        )

    metadata = ProjectMetadata(
        name=project_name,
        version=GraphPersistenceManager.SCHEMA_VERSION,
        created=datetime.now().isoformat(),
        modified=datetime.now().isoformat(),
        author="Scenario Visual Importer",
        description=f"Visual twin generated from scenario: {project_name}",
    )

    topology_analysis = {
        "scenario_manifest": dict(model.metadata.get("scenario_manifest", {})),
        "scenario_economics": dict(model.metadata.get("economics", {})),
        "scenario_equipment_entries": list(model.metadata.get("equipment_entries", [])),
        "scenario_equipment_index": dict(model.metadata.get("equipment_index", {})),
        "import_surface_schema_version": IMPORT_SURFACE_SCHEMA_VERSION,
    }

    return GraphSnapshot(
        metadata=metadata,
        canvas_state=CanvasState(),
        nodes=nodes,
        edges=edges,
        validation_result=None,
        topology_analysis=topology_analysis,
    )


def generate_layout_file(
    output_path: Path,
    scenarios_dir: str,
    topology_file: str,
    project_name: str,
) -> Tuple[Path, int, int]:
    """Generate and persist a prebuilt visual layout file."""
    model = ScenarioVisualImporter.build_visual_model(
        scenarios_dir=scenarios_dir,
        topology_file=topology_file,
    )

    snapshot = build_snapshot(project_name=project_name, model=model)
    manager = GraphPersistenceManager()
    snapshot.metadata.checksum = manager._calculate_checksum(snapshot)  # pylint: disable=protected-access
    saved_path = manager.save(str(output_path), snapshot)

    return saved_path, len(model.nodes), len(model.edges)


def ensure_prebuilt_layout_file(
    canonical_path: Path,
    scenarios_dir: str,
    topology_file: str = "plant_topology.yaml",
    project_name: str = "Plant Topology Visual Twin",
    temp_dir: Optional[Path] = None,
    force_regenerate: bool = False,
) -> Tuple[Path, bool, bool, int, int]:
    """
    Ensure a prebuilt layout file exists.

    Returns:
        (layout_path, was_generated, used_temp_fallback, node_count, edge_count)
    """
    if canonical_path.exists() and not force_regenerate:
        return canonical_path, False, False, 0, 0

    try:
        saved_path, node_count, edge_count = generate_layout_file(
            output_path=canonical_path,
            scenarios_dir=scenarios_dir,
            topology_file=topology_file,
            project_name=project_name,
        )
        return saved_path, True, False, node_count, edge_count
    except Exception as canonical_exc:
        fallback_dir = Path(gettempdir()) if temp_dir is None else Path(temp_dir)
        fallback_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = canonical_path.suffix or ".h2plant"
        fallback_path = fallback_dir / f"{canonical_path.stem}_{stamp}{suffix}"

        try:
            saved_path, node_count, edge_count = generate_layout_file(
                output_path=fallback_path,
                scenarios_dir=scenarios_dir,
                topology_file=topology_file,
                project_name=project_name,
            )
            return saved_path, True, True, node_count, edge_count
        except Exception as temp_exc:
            raise RuntimeError(
                "Failed to generate prebuilt visual twin. "
                f"Canonical write error: {canonical_exc}; "
                f"Temporary write error: {temp_exc}"
            ) from temp_exc


def prebuilt_layout_needs_regeneration(
    canonical_path: Path,
    scenarios_dir: str,
    topology_file: str = "plant_topology.yaml",
    required_surface_schema_version: int = IMPORT_SURFACE_SCHEMA_VERSION,
) -> Tuple[bool, str]:
    """
    Decide if the canonical prebuilt layout should be regenerated.

    Returns:
        (needs_regeneration, reason)
    """
    if not canonical_path.exists():
        return True, "missing"

    manager = GraphPersistenceManager()
    try:
        snapshot = manager.load(str(canonical_path))
    except Exception as exc:
        return True, f"unreadable:{exc}"

    topology_analysis = dict(snapshot.topology_analysis or {})
    surface_version = int(topology_analysis.get("import_surface_schema_version") or 0)
    if surface_version < int(required_surface_schema_version):
        return True, "legacy_surface_schema"

    stored_manifest = dict(topology_analysis.get("scenario_manifest") or {})
    stored_hashes = dict(stored_manifest.get("file_hashes") or {})
    if not stored_hashes:
        return True, "missing_file_hashes"

    try:
        current_model = ScenarioVisualImporter.build_visual_model(
            scenarios_dir=scenarios_dir,
            topology_file=topology_file,
        )
    except Exception as exc:
        # If source is unavailable, preserve current prebuilt instead of blocking load.
        return False, f"source_unavailable:{exc}"

    current_manifest = dict(current_model.metadata.get("scenario_manifest") or {})
    current_hashes = dict(current_manifest.get("file_hashes") or {})

    for rel_path, current_hash in current_hashes.items():
        if stored_hashes.get(rel_path) != current_hash:
            return True, f"hash_drift:{rel_path}"

    return False, "up_to_date"
