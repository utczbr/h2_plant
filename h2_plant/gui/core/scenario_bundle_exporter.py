"""
Scenario Bundle Exporter — generates a self-contained scenario bundle from the
live NodeGraph state so that ConfigLoader can consume it directly.

The exported bundle never modifies the template scenarios/ directory.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from h2_plant.gui.core.scenario_param_mapper import (
    backend_to_gui_props,
    effective_backend_key,
    gui_to_backend_overlay,
)
from h2_plant.gui.core.scenario_workspace import (
    DEFAULT_EQUIPMENT_FILE,
    DEFAULT_OPEX_FILE,
    DEFAULT_PHYSICS_FILE,
    DEFAULT_SIMULATION_FILE,
    resolve_manifest_file,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Safety guard
# ---------------------------------------------------------------------------

class TemplateSafetyError(Exception):
    """Raised when the exporter would write into a protected template directory."""


def _assert_not_inside_template(output_dir: Path, template_dir: Path) -> None:
    """Refuse to write if output_dir is inside the template scenarios/ folder.
    
    Overwriting the *same* output dir on re-export is intentional and allowed.
    We only block writing *inside* the original template source.
    """
    try:
        resolved_out = output_dir.resolve()
        resolved_tmpl = template_dir.resolve()
        # Same dir is fine (re-export overwrites)
        if resolved_out == resolved_tmpl:
            return
        resolved_out.relative_to(resolved_tmpl)
        raise TemplateSafetyError(
            f"Refusing to write generated bundle inside template directory: "
            f"{template_dir}"
        )
    except ValueError:
        pass  # not a sub-path — safe


# ---------------------------------------------------------------------------
#  Identity resolution helpers
# ---------------------------------------------------------------------------

_DEFAULT_SCENARIO_NAME = "GUI-Authored Scenario"
_FRAMEWORK_KEYS = frozenset(
    {
        "name",
        "color",
        "disabled",
        "selected",
        "pos",
        "id",
        "type_",
        "type",
        "visible",
        "width",
        "height",
        "layout_direction",
        "port_deletion_allowed",
        "subgraph_session",
        "inputs",
        "outputs",
        "custom",
        "node_color",
        "custom_label",
        "collapse_spacer",
        "text_color",
        "border_color",
    }
)
_PARAM_OVERLAY_EXCLUDED_KEYS = frozenset(
    {
        "component_id",
        "backend_type",
        "__scenario_component_id",
        "__scenario_backend_type",
        "__scenario_inputs",
        "__scenario_outputs",
        "__scenario_params",
        "__scenario_unmapped_params",
    }
)


def _extract_export_props(node) -> Dict[str, Any]:
    """Normalize node properties from NodeGraphQt flat/custom shapes."""
    raw_props = node.get_properties() if hasattr(node, "get_properties") else node.properties()
    if not isinstance(raw_props, dict):
        return {}

    normalized = dict(raw_props)
    custom_props = raw_props.get("custom")
    if isinstance(custom_props, dict):
        # Prefer custom payload values when both representations exist.
        normalized.update(custom_props)
    return normalized


def _to_yaml_safe(value: Any) -> Any:
    """Recursively coerce values to YAML-safe builtins."""
    if isinstance(value, dict):
        return {k: _to_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_to_yaml_safe(v) for v in value]
    return value


def _parse_live_value(value: Any) -> Any:
    """Parse string editor values to preserve scalar types when possible."""
    if not isinstance(value, str):
        return value
    if value == "":
        return value
    try:
        parsed = yaml.safe_load(value)
    except Exception:
        return value
    if parsed is None and value.strip() not in {"null", "Null", "NULL", "~"}:
        return value
    return parsed


def _resolve_source_topology_path(template_manifest: Dict[str, Any]) -> Optional[Path]:
    topology_ref_raw = template_manifest.get("topology_file") or template_manifest.get("topology_file_name")
    return resolve_manifest_file(
        template_manifest,
        "topology_file",
        str(topology_ref_raw or "plant_topology.yaml"),
    )


def _load_source_topology_data(template_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort load of source topology YAML for parity-sensitive export."""
    topology_path = _resolve_source_topology_path(template_manifest)
    if not topology_path or not topology_path.exists():
        return {}

    try:
        with open(topology_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(f"Failed to load source topology '{topology_path}': {exc}")
    return {}


def _build_source_resource_type_map(
    source_topology: Dict[str, Any],
) -> Dict[Tuple[str, str, str, str], str]:
    """Map source connection identity to resource_type for no-edit parity."""
    mapping: Dict[Tuple[str, str, str, str], str] = {}
    for source_node in source_topology.get("nodes", []) or []:
        source_id = str(source_node.get("id", "")).strip()
        if not source_id:
            continue
        for conn in source_node.get("connections", []) or []:
            source_port = str(conn.get("source_port", "")).strip()
            target_id = str(conn.get("target_name", "")).strip()
            target_port = str(conn.get("target_port", "")).strip()
            if not source_port or not target_id or not target_port:
                continue
            resource_type = str(conn.get("resource_type", "")).strip()
            if not resource_type:
                resource_type = _infer_resource_type(source_port)
            mapping[(source_id, source_port, target_id, target_port)] = resource_type
    return mapping


def _resolve_scenario_name(
    explicit_name: Optional[str],
    template_manifest: Dict[str, Any],
    source_topology: Dict[str, Any],
) -> str:
    """Resolve scenario name with source-preserving precedence."""
    explicit = str(explicit_name or "").strip()
    if explicit and explicit != _DEFAULT_SCENARIO_NAME:
        return explicit

    manifest_name = str(template_manifest.get("scenario_name") or "").strip()
    if manifest_name:
        return manifest_name

    source_name = str(source_topology.get("scenario_name") or "").strip()
    if source_name:
        return source_name

    if explicit:
        return explicit
    return _DEFAULT_SCENARIO_NAME


def _resolve_component_id(node) -> str:
    """Resolve canonical component_id from a graph node."""
    props = _extract_export_props(node)
    # Priority: explicit component_id > __scenario_component_id > node name
    cid = props.get("component_id") or props.get("__scenario_component_id") or ""
    return str(cid).strip() or node.name()


def _resolve_backend_type(node) -> str:
    """Resolve backend type string from a graph node."""
    props = _extract_export_props(node)
    # Priority: __scenario_backend_type > backend_type property > class-based reverse lookup
    bt = (
        props.get("__scenario_backend_type")
        or props.get("backend_type")
        or ""
    )
    if str(bt).strip():
        return str(bt).strip()
    # Class-based reverse lookup for typed nodes
    node_class_name = type(node).__name__
    class_to_backend = {
        # Electrolysis
        "PEMStackNode": "PEM",
        "SOECStackNode": "SOEC",
        "RectifierNode": "PowerTransformer",
        # Flow Control
        "ValveNode": "Valve",
        "MixerNode": "Mixer",
        "StreamSplitterNode": "StreamSplitter",
        "DrainRecorderMixerNode": "DrainRecorderMixer",
        "SignalMakeupMixerNode": "SignalMakeupMixer",
        "ProportionalMakeupMixerNode": "ProportionalMakeupMixer",
        "OxygenMakeupNode": "OxygenMakeupNode",
        # Thermal
        "ChillerNode": "Chiller",
        "DryCoolerNode": "DryCooler",
        "InterchangerNode": "Interchanger",
        "ElectricBoilerNode": "ElectricBoiler",
        "AttemperatorNode": "Attemperator",
        "CoolingManagerNode": "CoolingManager",
        # Separation
        "CoalescerNode": "Coalescer",
        "KnockOutDrumNode": "KnockOutDrum",
        "PSAUnitNode": "PSA Unit",
        "DeoxoReactorNode": "DeoxoReactor",
        "HydrogenMultiCycloneNode": "HydrogenMultiCyclone",
        "SeparationTankNode": "SeparationTank",
        "SyngasPSANode": "SyngasPSA",
        # Water
        "WaterPurifierNode": "WaterPurifier",
        "UltraPureWaterTankNode": "UltraPureWaterTank",
        "ExternalWaterSourceNode": "ExternalWaterSource",
        "WaterPumpThermodynamicNode": "WaterPumpThermodynamic",
        # Storage / Delivery
        "DetailedTankNode": "DetailedTank",
        "DischargeStationNode": "DischargeStation",
        "CompressorSingleNode": "CompressorSingle",
        # Reforming
        "IntegratedATRPlantNode": "IntegratedATRPlant",
        "ATRBoilerNode": "ATR_Boiler",
        "BiogasSourceNode": "BiogasSource",
        # Scenario fallback
        "ScenarioComponentNode": "ScenarioComponent",
    }
    return class_to_backend.get(node_class_name, "Unknown")


def _values_effectively_equal(a: Any, b: Any) -> bool:
    """Return True if two GUI-level values are semantically equivalent.

    Handles float round-trip tolerance so that, e.g., 97.0 and 97 are equal
    and a canonical 0.97 fraction converted to 97.0 GUI percent compares
    correctly against a textbox that parsed to 97.0.
    """
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def _resolve_params(node) -> Dict[str, Any]:
    """Extract canonical params dict, merging hidden snapshot with live property edits.

    For ScenarioComponentNode: __scenario_params is kept in sync by set_property override.
    For typed nodes: __scenario_params is a frozen import-time snapshot. We must overlay
    any visible properties that match param keys to capture UI edits.
    For manually added typed nodes (no __scenario_params): collect visible configuration
    properties as a fallback so configured values are not silently dropped.
    """
    props = _extract_export_props(node)
    params = props.get("__scenario_params")
    if not isinstance(params, dict):
        # Fallback: collect visible configuration properties from typed nodes
        return {
            k: _to_yaml_safe(_parse_live_value(v))
            for k, v in props.items()
            if not k.startswith("_") and k not in _FRAMEWORK_KEYS and v is not None
        }

    backend_type = _resolve_backend_type(node)

    # Back-convert the canonical snapshot to GUI-facing values so we can compare
    # them against what the live widget shows and detect only genuine user edits.
    canonical_as_gui, _ = backend_to_gui_props(backend_type, params)

    visible_props = {
        key: _parse_live_value(value)
        for key, value in props.items()
        if not key.startswith("_")
        and key not in _FRAMEWORK_KEYS
        and key not in _PARAM_OVERLAY_EXCLUDED_KEYS
        and value is not None
    }

    # Collect only effective edits — keys whose live value genuinely differs
    # from the canonical round-trip value.  This prevents float-drift and
    # uninitialised GUI defaults from silently injecting or overwriting params.
    effective_edits: Dict[str, Any] = {}
    for gui_key, gui_val in visible_props.items():
        canonical_gui_val = canonical_as_gui.get(gui_key)
        if canonical_gui_val is None:
            # GUI key absent from canonical round-trip.  Find the backend key this
            # GUI key would write to (via explicit mapping or direct key).  If that
            # backend key is also absent from canonical params, this is an
            # uninitialised GUI default — not a real user edit; skip it.
            bk = effective_backend_key(backend_type, gui_key)
            if bk not in params:
                continue
            # Backend key exists but canonical_as_gui doesn't have a GUI mapping
            # for it; treat as a new user-added entry.
            effective_edits[gui_key] = gui_val
            continue
        if not _values_effectively_equal(gui_val, canonical_gui_val):
            effective_edits[gui_key] = gui_val

    merged = gui_to_backend_overlay(
        backend_type=backend_type,
        gui_props=effective_edits,
        base_backend_params=dict(params),
    )
    return _to_yaml_safe(merged)


def _has_scenario_params_payload(node) -> bool:
    props = _extract_export_props(node)
    return isinstance(props.get("__scenario_params"), dict)


def _infer_resource_type(port_name: str) -> str:
    """Infer resource_type from port name using standard naming conventions."""
    name = str(port_name).lower()
    if any(tok in name for tok in ("signal", "control", "demand")):
        return "signal"
    if any(tok in name for tok in ("water", "steam", "drain", "makeup", "ultrapure")):
        return "water"
    if any(tok in name for tok in ("o2", "oxygen")):
        return "oxygen"
    if any(tok in name for tok in ("power", "electric", "grid")):
        return "electricity"
    if any(tok in name for tok in ("h2", "hydrogen", "purified", "compressed")):
        return "hydrogen"
    if any(tok in name for tok in ("heat", "thermal", "cooling", "duty")):
        return "heat"
    if any(tok in name for tok in ("gas", "syngas", "tail", "feed", "inlet")):
        return "gas"
    return "stream"


def _resolve_template_file(
    template_manifest: Dict[str, Any],
    key: str,
    default_reference: str,
) -> Optional[Path]:
    return resolve_manifest_file(template_manifest, key, default_reference)


def _copy_optional_file(
    source_path: Optional[Path],
    destination_path: Path,
    copied_files: List[str],
    copied_entry: str,
) -> None:
    if not source_path or not source_path.exists():
        return

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
    copied_files.append(copied_entry)


def _find_nearest_project_root(start_path: Path) -> Optional[Path]:
    """Find nearest ancestor containing both project-standard folders."""
    resolved_start = start_path.resolve()
    for candidate in (resolved_start, *resolved_start.parents):
        if (candidate / "scenarios").is_dir() and (candidate / "h2_plant").is_dir():
            return candidate
    return None


def _append_candidate(
    candidates: List[Path],
    seen: Set[Path],
    candidate: Path,
) -> None:
    resolved = candidate.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    candidates.append(resolved)


def _resolve_simulation_data_path(
    raw_value: Any,
    source_config_path: Path,
    template_manifest: Dict[str, Any],
) -> Tuple[Optional[Path], List[Path]]:
    raw_path = Path(str(raw_value))
    candidates: List[Path] = []
    seen: Set[Path] = set()

    if raw_path.is_absolute():
        _append_candidate(candidates, seen, raw_path)
        resolved = candidates[0]
        return (resolved if resolved.exists() else None), candidates

    # 1) Relative to simulation_config location.
    _append_candidate(candidates, seen, source_config_path.parent / raw_path)

    # 2) Relative to template manifest scenarios_dir, when present.
    scenarios_dir = template_manifest.get("scenarios_dir")
    if scenarios_dir:
        _append_candidate(candidates, seen, Path(str(scenarios_dir)) / raw_path)

    # 3) Relative to original source scenarios dir, when present.
    source_scenarios_dir = template_manifest.get("source_scenarios_dir")
    if source_scenarios_dir:
        _append_candidate(candidates, seen, Path(str(source_scenarios_dir)) / raw_path)

    # 4) Legacy fallback: nearest project root with scenarios/ + h2_plant/.
    project_root = _find_nearest_project_root(source_config_path.parent)
    if not project_root and scenarios_dir:
        project_root = _find_nearest_project_root(Path(str(scenarios_dir)))
    if not project_root and source_scenarios_dir:
        project_root = _find_nearest_project_root(Path(str(source_scenarios_dir)))
    if project_root:
        _append_candidate(candidates, seen, project_root / "scenarios" / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return None, candidates


def _copy_simulation_config_with_localized_data(
    template_manifest: Dict[str, Any],
    output_dir: Path,
    copied_files: List[str],
) -> None:
    source_config_path = _resolve_template_file(
        template_manifest,
        "simulation_config_file",
        DEFAULT_SIMULATION_FILE,
    )
    if not source_config_path or not source_config_path.exists():
        raise FileNotFoundError(
            f"Template simulation config not found: {source_config_path}"
        )

    with open(source_config_path, "r", encoding="utf-8") as handle:
        sim_data = yaml.safe_load(handle) or {}
    if not isinstance(sim_data, dict):
        raise ValueError(
            f"simulation_config.yaml root must be a mapping: {source_config_path}"
        )

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    copied_data_names: Dict[str, Path] = {}

    for key in ("energy_price_file", "wind_data_file"):
        raw_value = sim_data.get(key)
        if not raw_value:
            continue

        source_data_path, attempted_candidates = _resolve_simulation_data_path(
            raw_value,
            source_config_path,
            template_manifest,
        )

        if not source_data_path:
            attempted = ", ".join(str(path) for path in attempted_candidates) or str(raw_value)
            missing_reference = (
                str(attempted_candidates[0]) if attempted_candidates else str(raw_value)
            )
            raise FileNotFoundError(
                f"simulation_config '{key}' references missing file: "
                f"{missing_reference} (from {source_config_path}); attempted: {attempted}"
            )

        candidate_name = source_data_path.name
        if candidate_name in copied_data_names and copied_data_names[candidate_name] != source_data_path:
            stem = source_data_path.stem
            suffix = source_data_path.suffix
            idx = 2
            while True:
                new_name = f"{stem}_{idx}{suffix}"
                if new_name not in copied_data_names:
                    candidate_name = new_name
                    break
                idx += 1

        destination = data_dir / candidate_name
        copied_data_names[candidate_name] = source_data_path
        if source_data_path.resolve() != destination.resolve():
            shutil.copy2(source_data_path, destination)

        relative_entry = f"data/{candidate_name}"
        if relative_entry not in copied_files:
            copied_files.append(relative_entry)
        sim_data[key] = relative_entry

    sim_dst = output_dir / DEFAULT_SIMULATION_FILE
    with open(sim_dst, "w", encoding="utf-8") as handle:
        yaml.safe_dump(sim_data, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)
    copied_files.append(DEFAULT_SIMULATION_FILE)


# ---------------------------------------------------------------------------
#  Core export API
# ---------------------------------------------------------------------------

def export_bundle(
    graph,
    template_manifest: Dict[str, Any],
    economics: Dict[str, Any],
    output_dir: Path,
    *,
    scenario_name: Optional[str] = _DEFAULT_SCENARIO_NAME,
    topology_filename: str = "plant_topology.yaml",
) -> Dict[str, Any]:
    """
    Export a self-contained scenario bundle from the live graph.

    Parameters
    ----------
    graph : NodeGraph
        The live NodeGraphQt graph instance.
    template_manifest : dict
        The original scenario manifest (contains ``scenarios_dir`` for template files).
    economics : dict
        Flat economics parameter dict (keys must match ``EconomicsConfig`` fields).
    output_dir : Path
        Directory to write the bundle into (created if needed).
    scenario_name : str
        Human-readable name for the generated scenario.
    topology_filename : str
        Filename for the generated topology YAML.

    Returns
    -------
    dict
        Bundle manifest with file paths, timestamps, and provenance info.

    Raises
    ------
    TemplateSafetyError
        If ``output_dir`` is inside the template ``scenarios/`` directory.
    ValueError
        If duplicate ``component_id`` values are found.
    """
    output_dir = Path(output_dir)
    template_dir_raw = template_manifest.get("scenarios_dir")
    template_dir = Path(str(template_dir_raw)) if template_dir_raw else Path("")

    # --- Safety guard ---
    if template_dir_raw and template_dir.exists():
        _assert_not_inside_template(output_dir, template_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Resolve identity for each node ---
    all_nodes = graph.all_nodes()
    node_ids: Dict[str, Any] = {}  # component_id -> node
    for node in all_nodes:
        cid = _resolve_component_id(node)
        if cid in node_ids:
            raise ValueError(
                f"Duplicate component_id '{cid}' found on nodes: "
                f"'{node_ids[cid].name()}' and '{node.name()}'. "
                f"Edit one of them before exporting."
            )
        node_ids[cid] = node

    source_topology = _load_source_topology_data(template_manifest)
    resolved_scenario_name = _resolve_scenario_name(
        explicit_name=scenario_name,
        template_manifest=template_manifest,
        source_topology=source_topology,
    )
    source_resource_types = _build_source_resource_type_map(source_topology)

    # --- 2. Build topology YAML ---
    topology_data = _build_topology_yaml(
        all_nodes,
        node_ids,
        resolved_scenario_name,
        source_resource_types,
        source_topology=source_topology,
    )
    topology_path = output_dir / topology_filename
    with open(topology_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(topology_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info(f"Generated topology: {topology_path}")

    # --- 3. Generate economics YAML ---
    economics_path = output_dir / "economics_parameters.yaml"
    with open(economics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(economics), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info(f"Generated economics: {economics_path}")

    # --- 4. Copy template files (manifest-aware) ---
    copied_files: List[str] = []

    physics_src = _resolve_template_file(
        template_manifest,
        "physics_file",
        DEFAULT_PHYSICS_FILE,
    )
    physics_dst = output_dir / DEFAULT_PHYSICS_FILE
    if not physics_src or not physics_src.exists():
        raise FileNotFoundError(f"Template physics config not found: {physics_src}")
    _copy_optional_file(
        physics_src,
        physics_dst,
        copied_files,
        DEFAULT_PHYSICS_FILE,
    )

    # Copy simulation config and rewrite referenced data files to bundle-local data/.
    _copy_simulation_config_with_localized_data(template_manifest, output_dir, copied_files)

    equip_src = _resolve_template_file(
        template_manifest,
        "equipment_file",
        DEFAULT_EQUIPMENT_FILE,
    )
    equip_dst = output_dir / DEFAULT_EQUIPMENT_FILE
    _copy_optional_file(
        equip_src,
        equip_dst,
        copied_files,
        DEFAULT_EQUIPMENT_FILE,
    )

    opex_src = _resolve_template_file(
        template_manifest,
        "opex_file",
        DEFAULT_OPEX_FILE,
    )
    opex_dst = output_dir / DEFAULT_OPEX_FILE
    _copy_optional_file(
        opex_src,
        opex_dst,
        copied_files,
        DEFAULT_OPEX_FILE,
    )

    # --- 5. Build bundle manifest ---
    manifest = {
        "bundle_dir": str(output_dir),
        "source_template": str(template_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "topology": topology_filename,
            "economics": "economics_parameters.yaml",
            "copied": copied_files,
        },
        "scenario_name": resolved_scenario_name,
    }

    # Write manifest as JSON sidecar
    import json
    manifest_path = output_dir / "bundle_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Bundle exported: {len(node_ids)} nodes, "
        f"{len(copied_files)} copied files → {output_dir}"
    )
    return manifest


# ---------------------------------------------------------------------------
#  Topology YAML builder
# ---------------------------------------------------------------------------

def _build_topology_yaml(
    all_nodes,
    node_ids: Dict[str, Any],
    scenario_name: str,
    source_resource_types: Optional[Dict[Tuple[str, str, str, str], str]] = None,
    source_topology: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the topology YAML dict from the live graph.

    Schema matches ConfigLoader expectations:
    ```yaml
    scenario_name: "..."
    nodes:
      - id: "component_id"
        type: "BackendType"
        params: {...}
        connections:
          - source_port: "..."
            target_name: "..."
            target_port: "..."
            resource_type: "..."
    ```

    When ``source_topology`` is provided, node order and per-node connection
    order are anchored to the source so that a no-edit round-trip produces
    semantically identical YAML (modulo comments and whitespace).
    """
    # Reverse lookup: node object -> component_id
    node_to_cid: Dict[int, str] = {}
    for cid, node in node_ids.items():
        node_to_cid[id(node)] = cid

    # Build source-anchored ordering structures when source is available.
    source_nodes_list = (source_topology or {}).get("nodes", []) or []

    # Source node order: list of component_ids from source YAML
    source_node_order: List[str] = [
        str(n.get("id", "")).strip()
        for n in source_nodes_list
        if str(n.get("id", "")).strip()
    ]

    # Per-node source connection order: cid -> [(source_port, target_id, target_port)]
    source_conn_order: Dict[str, List[Tuple[str, str, str]]] = {}
    for src_node in source_nodes_list:
        node_id = str(src_node.get("id", "")).strip()
        if not node_id:
            continue
        source_conn_order[node_id] = [
            (
                str(conn.get("source_port", "")),
                str(conn.get("target_name", "")),
                str(conn.get("target_port", "")),
            )
            for conn in (src_node.get("connections") or [])
        ]

    # Determine output node order: source-ordered nodes first (if still live),
    # then any live nodes not present in source (new additions) appended.
    seen_cids: Set[str] = set()
    ordered_cids: List[str] = []
    for src_cid in source_node_order:
        if src_cid in node_ids:
            ordered_cids.append(src_cid)
            seen_cids.add(src_cid)
    for cid in node_ids:
        if cid not in seen_cids:
            ordered_cids.append(cid)

    nodes_list: List[Dict[str, Any]] = []

    for cid in ordered_cids:
        node = node_ids[cid]
        backend_type = _resolve_backend_type(node)
        params = _resolve_params(node)

        # Build live connection map: (source_port, target_cid, target_port) -> entry dict
        live_conn_map: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        for output_port in node.output_ports():
            for connected_port in output_port.connected_ports():
                target_node = connected_port.node()
                target_cid = node_to_cid.get(id(target_node), target_node.name())
                source_port = output_port.name()
                target_port = connected_port.name()
                source_key = (cid, source_port, target_cid, target_port)
                resource_type = (
                    source_resource_types.get(source_key)
                    if source_resource_types
                    else None
                )
                if not resource_type:
                    resource_type = _infer_resource_type(source_port)
                live_conn_map[(source_port, target_cid, target_port)] = {
                    "source_port": source_port,
                    "target_name": target_cid,
                    "target_port": target_port,
                    "resource_type": resource_type,
                }

        # Order connections: source order first, then live-only additions appended.
        connections: List[Dict[str, str]] = []
        seen_conn_keys: Set[Tuple[str, str, str]] = set()
        for conn_key in (source_conn_order.get(cid) or []):
            if conn_key in live_conn_map:
                connections.append(live_conn_map[conn_key])
                seen_conn_keys.add(conn_key)
        for conn_key, conn_entry in live_conn_map.items():
            if conn_key not in seen_conn_keys:
                connections.append(conn_entry)

        node_entry: Dict[str, Any] = {
            "id": cid,
            "type": backend_type,
        }
        if params or _has_scenario_params_payload(node):
            node_entry["params"] = params
        if connections:
            node_entry["connections"] = connections

        nodes_list.append(node_entry)

    return {
        "scenario_name": scenario_name,
        "nodes": nodes_list,
    }
