"""
Scenario visual importer for rendering backend-authored YAML scenarios in the GUI.

This module intentionally keeps parsing and transformation logic independent from Qt,
so it can be used both by the GUI and headless tooling/tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import hashlib

import time

import yaml

from h2_plant.gui.core.visual_layout_policy import (
    BLOCK_COLORS,
    BLOCK_GAP,
    BLOCK_ORDER,
    SYSTEM_GROUP_TO_BLOCK,
    ZONE_MARGIN,
    assign_zone,
)


def _to_str(value: Any) -> str:
    """Convert any value to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _resolve_topology_path(scenarios_dir: Path, topology_file: str) -> Path:
    """Resolve topology path from relative or absolute input."""
    candidate = Path(topology_file)
    if candidate.is_absolute():
        return candidate
    return (scenarios_dir / candidate).resolve()


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """Read a YAML file as a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_scenario_bundle(scenarios_dir: str, topology_file: str = "plant_topology.yaml") -> Dict[str, Any]:
    """
    Load topology/economics/equipment files for scenario visual import.

    Returns:
        A dictionary containing loaded YAML content and resolved source paths.
    """
    scenario_path = Path(scenarios_dir).resolve()
    topology_path = _resolve_topology_path(scenario_path, topology_file)
    economics_path = scenario_path / "economics_parameters.yaml"
    equipment_path = scenario_path / "Economics" / "equipment_mappings.yaml"

    topology_data = _load_yaml_file(topology_path)
    economics_data = _load_yaml_file(economics_path)
    equipment_data = _load_yaml_file(equipment_path)

    return {
        "scenarios_dir": str(scenario_path),
        "topology_path": str(topology_path),
        "economics_path": str(economics_path.resolve()),
        "equipment_path": str(equipment_path.resolve()),
        "topology": topology_data,
        "economics": economics_data,
        "equipment": equipment_data,
    }


def resolve_simulation_source(
    scenario_manifest: Optional[Dict[str, Any]],
    requested_scenarios_dir: Optional[str],
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Resolve simulation source mode.

    Handles both v1 (absolute paths) and v2 (relative paths) manifests.

    Returns:
        (scenarios_dir, topology_file, forced_scenario_mode)
    """
    if scenario_manifest:
        scenarios_dir = scenario_manifest.get("scenarios_dir")
        topology_file = scenario_manifest.get("topology_file")
        # v2 manifests store relative paths; resolve against scenarios_dir
        if scenarios_dir and topology_file and not Path(topology_file).is_absolute():
            topology_file = str(Path(scenarios_dir) / topology_file)
        return scenarios_dir, topology_file, True
    return requested_scenarios_dir, None, False


def resolve_component_id_for_equipment(
    component_id: Optional[Any],
    legacy_component_id: Optional[Any],
    node_name: Any,
) -> str:
    """
    Resolve component ID for equipment-index lookup.

    Fallback order:
    1) explicit component_id
    2) legacy __scenario_component_id
    3) node display name suffix after first ":" (e.g. "Type: ID")
    4) full node name
    """
    if component_id:
        return str(component_id)
    if legacy_component_id:
        return str(legacy_component_id)

    label = str(node_name)
    if ":" in label:
        return label.split(":", 1)[1].strip()
    return label


def _normalize_topology_ids(raw_topology_ids: Any) -> List[str]:
    """Normalize topology_ids values from equipment mappings."""
    normalized: List[str] = []
    values: Iterable[Any]

    if raw_topology_ids is None:
        values = []
    elif isinstance(raw_topology_ids, list):
        values = raw_topology_ids
    else:
        values = [raw_topology_ids]

    for raw in values:
        text = _to_str(raw)
        if not text:
            continue
        # YAML entries may include comma-separated IDs in one scalar.
        for token in text.split(","):
            topology_id = token.strip()
            if topology_id:
                normalized.append(topology_id)

    return normalized


def normalize_equipment_entries(equipment_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return equipment entries with normalized topology_ids lists."""
    normalized: List[Dict[str, Any]] = []
    for entry in equipment_entries:
        entry_copy = dict(entry)
        entry_copy["topology_ids"] = _normalize_topology_ids(entry_copy.get("topology_ids"))
        normalized.append(entry_copy)
    return normalized


def build_incoming_port_index(topology_nodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build target-port lookup: node_id -> incoming ports."""
    incoming: Dict[str, Set[str]] = {}

    for node in topology_nodes:
        node_id = _to_str(node.get("id"))
        incoming.setdefault(node_id, set())

    for source_node in topology_nodes:
        for conn in source_node.get("connections", []) or []:
            target_name = _to_str(conn.get("target_name"))
            target_port = _to_str(conn.get("target_port"))
            if not target_name or not target_port:
                continue
            incoming.setdefault(target_name, set()).add(target_port)

    return {node_id: sorted(ports) for node_id, ports in incoming.items()}


def build_equipment_index(equipment_entries: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Build reverse map topology_id -> list[equipment_index]."""
    index: Dict[str, List[int]] = {}
    for entry_idx, entry in enumerate(equipment_entries):
        topology_ids = _normalize_topology_ids(entry.get("topology_ids"))
        for topology_id in topology_ids:
            index.setdefault(topology_id, []).append(entry_idx)
    return {topology_id: sorted(indices) for topology_id, indices in index.items()}


@dataclass(frozen=True)
class ScenarioVisualNode:
    """Normalized node record used by GUI/scenario layout generation."""

    id: str
    backend_type: str
    params: Dict[str, Any]
    connections: List[Dict[str, Any]]
    incoming_ports: List[str]
    outgoing_ports: List[str]
    x: float
    y: float
    typed_candidate: bool


@dataclass(frozen=True)
class ScenarioVisualEdge:
    """Normalized edge record with explicit ports and resource type."""

    source_id: str
    source_port: str
    target_id: str
    target_port: str
    resource_type: str


@dataclass(frozen=True)
class ScenarioVisualModel:
    """Complete visual import model."""

    nodes: List[ScenarioVisualNode]
    edges: List[ScenarioVisualEdge]
    metadata: Dict[str, Any]


class ScenarioVisualImporter:
    """
    Scenario visual importer service.

    Produces deterministic node/edge models and metadata from scenario YAML files.
    """

    COLUMN_SPACING = 220.0
    ROW_SPACING = 140.0  # Must exceed node height (~100-120px) to prevent overlap
    LANE_GAP = 150.0

    TYPED_BACKEND_TYPES = {
        # Electrolysis / Production
        "PEM",
        "SOEC",
        "PowerTransformer",
        # Thermal
        "Chiller",
        "DryCooler",
        "Interchanger",
        "ElectricBoiler",
        "Attemperator",
        "CoolingManager",
        # Separation
        "Coalescer",
        "KnockOutDrum",
        "PSA Unit",
        "DeoxoReactor",
        "HydrogenMultiCyclone",
        "SeparationTank",
        "SyngasPSA",
        # Mixing / Flow
        "Mixer",
        "Valve",
        "StreamSplitter",
        "DrainRecorderMixer",
        "SignalMakeupMixer",
        "ProportionalMakeupMixer",
        "OxygenMakeupNode",
        # Water
        "WaterPurifier",
        "UltraPureWaterTank",
        "ExternalWaterSource",
        "WaterPumpThermodynamic",
        # Storage / Delivery
        "DetailedTank",
        "DischargeStation",
        "CompressorSingle",
        # Reforming
        "IntegratedATRPlant",
        "ATR_Boiler",
        "BiogasSource",
    }

    @classmethod
    def build_visual_model(
        cls,
        scenarios_dir: str,
        topology_file: str = "plant_topology.yaml",
        layout_mode: str = "report_v1",
    ) -> ScenarioVisualModel:
        """Build normalized visual model from scenario files.

        Args:
            scenarios_dir: Path to the scenarios directory.
            topology_file: Topology YAML filename (relative to scenarios_dir).
            layout_mode: ``"report_v1"`` (zone-based industrial layout, default)
                         or ``"legacy"`` (original lane-based layout).
        """
        bundle = load_scenario_bundle(scenarios_dir=scenarios_dir, topology_file=topology_file)

        topology_data = bundle["topology"]
        topology_nodes = topology_data.get("nodes", []) or []
        if not isinstance(topology_nodes, list):
            raise ValueError("Invalid topology: 'nodes' must be a list")

        incoming_index = build_incoming_port_index(topology_nodes)
        edges = cls._extract_edges(topology_nodes)

        # Load equipment early so block assignments are available for layout computation.
        normalized_equipment = normalize_equipment_entries(
            bundle["equipment"].get("equipment", []) or []
        )
        equipment_index = build_equipment_index(normalized_equipment)

        visual_layout: Optional[Dict[str, Any]] = None
        if layout_mode == "report_v1":
            layout, visual_layout = cls._compute_report_layout(
                topology_nodes, edges, normalized_equipment
            )
        else:
            layout = cls._compute_layout(topology_nodes, edges)

        nodes: List[ScenarioVisualNode] = []
        for node in topology_nodes:
            node_id = _to_str(node.get("id"))
            backend_type = _to_str(node.get("type")) or "PassiveComponent"
            params = dict(node.get("params", {}) or {})
            connections = [dict(conn) for conn in (node.get("connections", []) or [])]

            outgoing_ports: Set[str] = set()
            for conn in connections:
                source_port = _to_str(conn.get("source_port"))
                if source_port:
                    outgoing_ports.add(source_port)

            x, y = layout.get(node_id, (0.0, 0.0))

            nodes.append(
                ScenarioVisualNode(
                    id=node_id,
                    backend_type=backend_type,
                    params=params,
                    connections=connections,
                    incoming_ports=incoming_index.get(node_id, []),
                    outgoing_ports=sorted(outgoing_ports),
                    x=x,
                    y=y,
                    typed_candidate=backend_type in cls.TYPED_BACKEND_TYPES,
                )
            )

        topology_path = Path(bundle["topology_path"])
        economics_path = Path(bundle["economics_path"])
        equipment_path = Path(bundle["equipment_path"])
        scenarios_dir = Path(bundle["scenarios_dir"])
        physics_path = scenarios_dir / "physics_parameters.yaml"
        simulation_config_path = scenarios_dir / "simulation_config.yaml"
        opex_path = scenarios_dir / "Economics" / "opex_config.yaml"

        # Store paths relative to scenarios_dir for portability.
        # Absolute paths break when projects are moved between machines.
        def _relpath(p: Path) -> str:
            try:
                return str(p.relative_to(scenarios_dir))
            except ValueError:
                return str(p)

        file_hashes = {
            _relpath(topology_path): _sha256_file(topology_path),
            _relpath(economics_path): _sha256_file(economics_path),
            _relpath(equipment_path): _sha256_file(equipment_path),
        }
        if physics_path.exists():
            file_hashes[_relpath(physics_path)] = _sha256_file(physics_path)
        if simulation_config_path.exists():
            file_hashes[_relpath(simulation_config_path)] = _sha256_file(simulation_config_path)
        if opex_path.exists():
            file_hashes[_relpath(opex_path)] = _sha256_file(opex_path)

        manifest = {
            "kind": "scenario_visual_manifest",
            "version": 2,
            "scenarios_dir": bundle["scenarios_dir"],
            "topology_file": _relpath(topology_path),
            "topology_file_name": topology_path.name,
            "physics_file": _relpath(physics_path),
            "economics_file": _relpath(economics_path),
            "simulation_config_file": _relpath(simulation_config_path),
            "equipment_file": _relpath(equipment_path),
            "file_hashes": file_hashes,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "topology_node_count": len(nodes),
            "topology_edge_count": len(edges),
        }
        if opex_path.exists():
            manifest["opex_file"] = _relpath(opex_path)

        metadata: Dict[str, Any] = {
            "scenario_manifest": manifest,
            "economics": dict(bundle["economics"]),
            "equipment_entries": normalized_equipment,
            "equipment_index": equipment_index,
        }
        if visual_layout is not None:
            metadata["visual_layout"] = visual_layout

        return ScenarioVisualModel(nodes=nodes, edges=edges, metadata=metadata)

    @classmethod
    def _extract_edges(cls, topology_nodes: List[Dict[str, Any]]) -> List[ScenarioVisualEdge]:
        edges: List[ScenarioVisualEdge] = []
        for source_node in topology_nodes:
            source_id = _to_str(source_node.get("id"))
            for conn in source_node.get("connections", []) or []:
                source_port = _to_str(conn.get("source_port"))
                target_id = _to_str(conn.get("target_name"))
                target_port = _to_str(conn.get("target_port"))
                resource_type = _to_str(conn.get("resource_type")) or "stream"
                if not source_id or not source_port or not target_id or not target_port:
                    continue
                edges.append(
                    ScenarioVisualEdge(
                        source_id=source_id,
                        source_port=source_port,
                        target_id=target_id,
                        target_port=target_port,
                        resource_type=resource_type,
                    )
                )
        return edges

    @classmethod
    def _compute_layout(
        cls,
        topology_nodes: List[Dict[str, Any]],
        edges: List[ScenarioVisualEdge],
    ) -> Dict[str, Tuple[float, float]]:
        """Deterministic layout with non-overlapping lanes and process-flow ordering."""
        node_ids = [_to_str(node.get("id")) for node in topology_nodes]
        depth = cls._topological_depth(node_ids, edges)

        raw_x_order: Dict[str, float] = {}
        lane_key: Dict[str, str] = {}
        for node in topology_nodes:
            node_id = _to_str(node.get("id"))
            params = dict(node.get("params", {}) or {})
            backend_type = _to_str(node.get("type")) or "PassiveComponent"

            process_step_raw = params.get("process_step")
            process_step: Optional[float] = None
            if process_step_raw is not None:
                try:
                    process_step = float(process_step_raw)
                except (TypeError, ValueError):
                    process_step = None

            raw_x_order[node_id] = process_step if process_step is not None else float(depth.get(node_id, 0))
            lane = _to_str(params.get("system_group")) or backend_type
            lane_key[node_id] = lane

        unique_x_values = sorted(set(raw_x_order.values()))
        x_rank = {value: idx for idx, value in enumerate(unique_x_values)}

        nodes_by_lane: Dict[str, List[str]] = {}
        for node_id in node_ids:
            lane = lane_key[node_id]
            nodes_by_lane.setdefault(lane, []).append(node_id)

        lane_order = sorted(
            nodes_by_lane.keys(),
            key=lambda lane: (
                min(x_rank[raw_x_order[nid]] for nid in nodes_by_lane[lane]),
                lane,
            ),
        )

        positions: Dict[str, Tuple[float, float]] = {}
        y_cursor = 0.0
        for lane in lane_order:
            lane_nodes = sorted(
                nodes_by_lane[lane],
                key=lambda nid: (x_rank[raw_x_order[nid]], nid),
            )
            for row_idx, node_id in enumerate(lane_nodes):
                x = float(x_rank[raw_x_order[node_id]]) * cls.COLUMN_SPACING
                y = y_cursor + float(row_idx) * cls.ROW_SPACING
                positions[node_id] = (x, y)
            lane_height = max(len(lane_nodes) - 1, 0) * cls.ROW_SPACING
            y_cursor += lane_height + cls.LANE_GAP

        return positions

    @staticmethod
    def _build_block_index(equipment_entries: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build {node_id: block_name} from equipment entries.

        Priority: first entry that maps a topology_id to a block wins.
        """
        block_by_node: Dict[str, str] = {}
        for entry in equipment_entries:
            block = str(entry.get("block") or "").strip()
            if not block:
                continue
            for tid in _normalize_topology_ids(entry.get("topology_ids")):
                if tid and tid not in block_by_node:
                    block_by_node[tid] = block
        return block_by_node

    @classmethod
    def _compute_report_layout(
        cls,
        topology_nodes: List[Dict[str, Any]],
        edges: List[ScenarioVisualEdge],
        equipment_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
        """
        Equipment-block-based deterministic layout for report/prebuilt mode.

        Algorithm:
        1. Assign each node to a block using equipment_entries topology_ids → block.
           Fallback: system_group param → "Uncategorised".
        2. Within each block, determine sub-lanes by system_group (or backend_type).
        3. Within each sub-lane, rank nodes horizontally by process_step (or topo depth).
        4. Lay out blocks left-to-right (by BLOCK_ORDER); sub-lanes stacked top-to-bottom
           within each block; nodes placed in columns within each sub-lane.
        5. Compute per-block bounding boxes for overlay rendering.

        Returns:
            (positions, visual_layout)
            - positions: {node_id: (x, y)}
            - visual_layout: metadata dict with zones (blocks), node_zone_map, layout_constants
        """
        import logging as _logging

        t0 = time.perf_counter()

        node_ids = [_to_str(node.get("id")) for node in topology_nodes]
        depth = cls._topological_depth(node_ids, edges)

        # Step 1: Block assignment + per-node metadata
        block_by_node = cls._build_block_index(equipment_entries or [])

        raw_x_order: Dict[str, float] = {}
        block_key: Dict[str, str] = {}
        sub_lane_key: Dict[str, str] = {}   # sub-lane within block

        for node in topology_nodes:
            node_id = _to_str(node.get("id"))
            params = dict(node.get("params", {}) or {})
            backend_type = _to_str(node.get("type")) or "PassiveComponent"

            # Block: from equipment entries → system_group mapped to parent block →
            #        node ID prefix heuristic → "Uncategorised"
            block = block_by_node.get(node_id)
            if not block:
                sys_grp = str(params.get("system_group") or "").strip()
                block = SYSTEM_GROUP_TO_BLOCK.get(sys_grp) or sys_grp or ""
            if not block:
                # Last-resort: derive block from node ID prefix (SOEC_*, PEM_*, ATR_*, etc.)
                upper = node_id.upper()
                for candidate in ("SOEC", "PEM", "ATR", "STORAGE", "UTILITIES"):
                    if upper.startswith(candidate + "_") or upper == candidate:
                        block = candidate.title() if candidate == "STORAGE" else candidate
                        break
            block_key[node_id] = block or "Uncategorised"

            # Sub-lane within block: system_group if set, else backend_type
            sub_lane = str(params.get("system_group") or "").strip() or backend_type
            sub_lane_key[node_id] = sub_lane

            # Horizontal ordering: process_step first, else global topo depth
            process_step_raw = params.get("process_step")
            process_step: Optional[float] = None
            if process_step_raw is not None:
                try:
                    process_step = float(process_step_raw)
                except (TypeError, ValueError):
                    process_step = None
            raw_x_order[node_id] = (
                process_step if process_step is not None else float(depth.get(node_id, 0))
            )

        # Step 2: Group nodes by block
        nodes_by_block: Dict[str, List[str]] = {}
        for node_id in node_ids:
            blk = block_key[node_id]
            nodes_by_block.setdefault(blk, []).append(node_id)

        # Sort blocks by BLOCK_ORDER, then alphabetically for any extras
        block_order_index = {b: i for i, b in enumerate(BLOCK_ORDER)}
        blocks_present = sorted(
            nodes_by_block.keys(),
            key=lambda b: (block_order_index.get(b, len(BLOCK_ORDER)), b),
        )

        # Step 3–4: Place nodes block-by-block, left-to-right
        positions: Dict[str, Tuple[float, float]] = {}
        block_min_max: Dict[str, Dict[str, float]] = {}
        block_x_offset = 0.0     # left edge of current block

        for block_name in blocks_present:
            block_node_ids = nodes_by_block[block_name]

            # Compute local x-ranks within this block
            block_raw_x = {nid: raw_x_order[nid] for nid in block_node_ids}
            unique_x_local = sorted(set(block_raw_x.values()))
            local_x_rank = {v: i for i, v in enumerate(unique_x_local)}

            # Group block nodes by sub-lane
            sub_lanes_in_block: Dict[str, List[str]] = {}
            for nid in block_node_ids:
                sl = sub_lane_key[nid]
                sub_lanes_in_block.setdefault(sl, []).append(nid)

            # Sort sub-lanes by their minimum local x-rank, then name
            sub_lane_order = sorted(
                sub_lanes_in_block.keys(),
                key=lambda sl: (
                    min(local_x_rank[block_raw_x[nid]] for nid in sub_lanes_in_block[sl]),
                    sl,
                ),
            )

            bm: Dict[str, float] = {
                "min_x": float("inf"), "max_x": float("-inf"),
                "min_y": float("inf"), "max_y": float("-inf"),
            }
            block_min_max[block_name] = bm

            y_cursor = 0.0
            for sl in sub_lane_order:
                sl_nodes = sorted(
                    sub_lanes_in_block[sl],
                    key=lambda nid: (local_x_rank[block_raw_x[nid]], nid),
                )
                for row_idx, nid in enumerate(sl_nodes):
                    x = block_x_offset + float(local_x_rank[block_raw_x[nid]]) * cls.COLUMN_SPACING
                    y = y_cursor + float(row_idx) * cls.ROW_SPACING
                    positions[nid] = (x, y)
                    bm["min_x"] = min(bm["min_x"], x)
                    bm["max_x"] = max(bm["max_x"], x)
                    bm["min_y"] = min(bm["min_y"], y)
                    bm["max_y"] = max(bm["max_y"], y)

                lane_height = max(len(sl_nodes) - 1, 0) * cls.ROW_SPACING
                y_cursor += lane_height + cls.LANE_GAP

            # Next block starts after this block's right edge + BLOCK_GAP
            block_width = (len(unique_x_local) - 1) * cls.COLUMN_SPACING
            block_x_offset += block_width + cls.COLUMN_SPACING + BLOCK_GAP

        # Step 5: Compute per-block bounding boxes with margin
        zones_meta: Dict[str, Any] = {}
        fallback_color = list(BLOCK_COLORS.get("Uncategorised", (210, 210, 210)))
        for block_name in blocks_present:
            bm = block_min_max[block_name]
            color = list(BLOCK_COLORS.get(block_name, fallback_color))
            zones_meta[block_name] = {
                "x": bm["min_x"] - ZONE_MARGIN,
                "y": bm["min_y"] - ZONE_MARGIN,
                "w": (bm["max_x"] - bm["min_x"]) + cls.COLUMN_SPACING + 2 * ZONE_MARGIN,
                "h": (bm["max_y"] - bm["min_y"]) + cls.ROW_SPACING + 2 * ZONE_MARGIN,
                "color": color,
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _logging.getLogger(__name__).debug(
            "report_layout completed in %.1f ms for %d nodes across %d blocks",
            elapsed_ms, len(node_ids), len(blocks_present),
        )

        visual_layout = {
            "layout_mode": "report_v1",
            "zones": zones_meta,
            "node_zone_map": dict(block_key),
            "layout_constants": {
                "COLUMN_SPACING": cls.COLUMN_SPACING,
                "ROW_SPACING": cls.ROW_SPACING,
                "LANE_GAP": cls.LANE_GAP,
                "BLOCK_GAP": BLOCK_GAP,
                "ZONE_MARGIN": ZONE_MARGIN,
            },
        }
        return positions, visual_layout

    @staticmethod
    def _topological_depth(node_ids: List[str], edges: List[ScenarioVisualEdge]) -> Dict[str, int]:
        """Compute topological depth for fallback horizontal ordering."""
        indegree = {node_id: 0 for node_id in node_ids}
        adjacency: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}

        for edge in edges:
            if edge.source_id not in indegree or edge.target_id not in indegree:
                continue
            adjacency[edge.source_id].append(edge.target_id)
            indegree[edge.target_id] += 1

        for source_id in adjacency:
            adjacency[source_id] = sorted(adjacency[source_id])

        queue = sorted([node_id for node_id, degree in indegree.items() if degree == 0])
        depth = {node_id: 0 for node_id in queue}

        while queue:
            current = queue.pop(0)
            current_depth = depth.get(current, 0)

            for target in adjacency.get(current, []):
                next_depth = current_depth + 1
                if next_depth > depth.get(target, 0):
                    depth[target] = next_depth

                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
                    queue.sort()

        # Cycles/unreached nodes get zero depth fallback.
        for node_id in node_ids:
            depth.setdefault(node_id, 0)

        return depth
