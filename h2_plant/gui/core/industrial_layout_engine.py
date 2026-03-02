"""
Industrial PFD layout engine for scenario visual twins.

Pure layout helpers (no Qt/YAML dependencies) used by importer and tooling.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from h2_plant.gui.core.visual_layout_policy import SYSTEM_GROUP_TO_BLOCK


INDUSTRIAL_LAYOUT_MODE = "industrial_pfd_v1"
INDUSTRIAL_LAYOUT_SCHEMA_VERSION = 2

ROW_1_UTILITIES = "Row 1: Utilities"
ROW_2_SOEC = "Row 2: SOEC Train"
ROW_3_PEM_STORAGE = "Row 3: PEM + Manifold + Storage/Dispatch"
ROW_4_ATR = "Row 4: ATR/Biogas Train"

ROW_ORDER: List[str] = [
    ROW_1_UTILITIES,
    ROW_2_SOEC,
    ROW_3_PEM_STORAGE,
    ROW_4_ATR,
]
ROW_ORDER_INDEX = {name: idx for idx, name in enumerate(ROW_ORDER)}

ROW_COLORS: Dict[str, Tuple[int, int, int]] = {
    ROW_1_UTILITIES: (150, 190, 235),
    ROW_2_SOEC: (170, 215, 245),
    ROW_3_PEM_STORAGE: (175, 225, 175),
    ROW_4_ATR: (245, 195, 155),
}

GROUP_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Core Utilities": (130, 170, 215),
    "Water Utilities": (130, 190, 235),
    "O2 Header": (175, 165, 220),
    "SOEC H2 Train": (155, 205, 240),
    "SOEC Water/Steam": (135, 190, 225),
    "SOEC Aux": (180, 220, 245),
    "PEM Train": (155, 210, 155),
    "PEM Water/O2": (130, 190, 130),
    "Storage & Dispatch": (220, 210, 150),
    "Manifold": (170, 205, 170),
    "ATR Process": (235, 180, 140),
    "Biogas Feed": (220, 165, 125),
    "ATR Aux": (235, 190, 160),
}

START_X = 120.0
START_Y = 120.0
X_SPACING = 260.0
INTER_GROUP_GAP = 320.0
ROW_GAP = 560.0
LANE_GAP = 120.0
ROW_SPACING = 120.0
ZONE_MARGIN = 50.0
GROUP_MARGIN = 30.0


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_topology_ids(raw_topology_ids: Any) -> List[str]:
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
        for token in text.split(","):
            topology_id = token.strip()
            if topology_id:
                normalized.append(topology_id)

    return normalized


def _build_block_index(equipment_entries: List[Dict[str, Any]]) -> Dict[str, str]:
    block_by_node: Dict[str, str] = {}
    for entry in equipment_entries:
        block = _to_str(entry.get("block"))
        if not block:
            continue
        for topology_id in _normalize_topology_ids(entry.get("topology_ids")):
            if topology_id and topology_id not in block_by_node:
                block_by_node[topology_id] = block
    return block_by_node


def _system_group_to_block(system_group: str) -> str:
    return _to_str(SYSTEM_GROUP_TO_BLOCK.get(system_group)) or system_group


def _fallback_block_from_id(node_id: str) -> str:
    upper = node_id.upper()
    if upper.startswith("SOEC_"):
        return "SOEC"
    if upper.startswith("PEM_"):
        return "PEM"
    if upper.startswith("ATR_") or upper.startswith("BIOGAS_"):
        return "ATR"
    if upper.startswith(("LP_", "HP_", "TRUCK_", "H2_")):
        return "Storage"
    if upper.startswith(("WATER_", "ULTRAPURE_", "BOP_", "O2_")) or upper == "COOLING_MANAGER":
        return "Utilities"
    return ""


def _assign_row(
    node_id: str,
    backend_type: str,
    params: Dict[str, Any],
    block_by_node: Dict[str, str],
) -> str:
    upper = node_id.upper()
    if upper.startswith("SOEC_"):
        return ROW_2_SOEC
    if upper.startswith("PEM_"):
        return ROW_3_PEM_STORAGE
    if upper.startswith("ATR_") or upper.startswith("BIOGAS_"):
        return ROW_4_ATR
    if upper.startswith(("LP_", "HP_", "TRUCK_", "H2_")):
        return ROW_3_PEM_STORAGE

    block = _to_str(block_by_node.get(node_id))
    if not block:
        block = _system_group_to_block(_to_str(params.get("system_group")))
    if not block:
        block = _fallback_block_from_id(node_id)
    block_upper = block.upper()

    if block_upper in {"SOEC"}:
        return ROW_2_SOEC
    if block_upper in {"PEM", "STORAGE", "GENERAL", "OTHER"}:
        return ROW_3_PEM_STORAGE
    if block_upper in {"ATR"}:
        return ROW_4_ATR
    if block_upper in {"UTILITIES", "WATER_TREATMENT"}:
        return ROW_1_UTILITIES

    if backend_type in {"IntegratedATRPlant", "ATR_Boiler", "BiogasSource", "SyngasPSA"}:
        return ROW_4_ATR
    if backend_type in {"DetailedTank", "CompressorSingle", "DischargeStation", "Mixer", "StreamSplitter"}:
        return ROW_3_PEM_STORAGE
    if backend_type in {
        "ExternalWaterSource",
        "WaterPurifier",
        "UltraPureWaterTank",
        "WaterPumpThermodynamic",
        "CoolingManager",
    }:
        return ROW_1_UTILITIES

    return ROW_3_PEM_STORAGE


def _assign_group(row_name: str, node_id: str, params: Dict[str, Any]) -> str:
    upper = node_id.upper()
    system_group = _to_str(params.get("system_group")).upper()

    if row_name == ROW_1_UTILITIES:
        if upper.startswith(("WATER_", "ULTRAPURE_")) or "WATER" in system_group:
            return "Water Utilities"
        if upper.startswith("O2_") or "O2" in system_group:
            return "O2 Header"
        return "Core Utilities"

    if row_name == ROW_2_SOEC:
        if "WATER" in system_group or "STEAM" in system_group or "WATER" in upper or "STEAM" in upper:
            return "SOEC Water/Steam"
        if "_H2_" in upper or "H2" in upper:
            return "SOEC H2 Train"
        return "SOEC Aux"

    if row_name == ROW_3_PEM_STORAGE:
        if upper.startswith(("LP_", "HP_", "TRUCK_", "H2_")) or upper in {
            "LP_STORAGE_TANK",
            "TRUCK_STATION_1",
        }:
            return "Storage & Dispatch"
        if upper.startswith("PEM_"):
            if "WATER" in system_group or "O2" in system_group or "WATER" in upper or "_O2_" in upper:
                return "PEM Water/O2"
            return "PEM Train"
        return "Manifold"

    if upper.startswith("BIOGAS_"):
        return "Biogas Feed"
    if upper.startswith("ATR_"):
        return "ATR Process"
    return "ATR Aux"


def _extract_order_value(node_params: Dict[str, Any], node_depth: int) -> float:
    process_step_raw = node_params.get("process_step")
    if process_step_raw is not None:
        try:
            return float(process_step_raw)
        except (TypeError, ValueError):
            pass
    return float(node_depth)


def _has_valid_process_step(node_params: Dict[str, Any]) -> bool:
    process_step_raw = node_params.get("process_step")
    if process_step_raw is None:
        return False
    try:
        float(process_step_raw)
        return True
    except (TypeError, ValueError):
        return False


def _minmax() -> Dict[str, float]:
    return {
        "min_x": float("inf"),
        "max_x": float("-inf"),
        "min_y": float("inf"),
        "max_y": float("-inf"),
    }


def _update_minmax(bounds: Dict[str, float], x: float, y: float) -> None:
    bounds["min_x"] = min(bounds["min_x"], x)
    bounds["max_x"] = max(bounds["max_x"], x)
    bounds["min_y"] = min(bounds["min_y"], y)
    bounds["max_y"] = max(bounds["max_y"], y)


def _safe_rect(bounds: Dict[str, float]) -> Optional[Tuple[float, float, float, float]]:
    if bounds["min_x"] == float("inf"):
        return None
    x = bounds["min_x"]
    y = bounds["min_y"]
    w = (bounds["max_x"] - bounds["min_x"]) + X_SPACING
    h = (bounds["max_y"] - bounds["min_y"]) + ROW_SPACING
    return x, y, w, h


def compute_industrial_layout(
    topology_nodes: List[Dict[str, Any]],
    depth: Dict[str, int],
    equipment_entries: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    """
    Compute deterministic industrial PFD layout.

    Returns:
        (positions, visual_layout_metadata)
    """
    block_by_node = _build_block_index(equipment_entries or [])

    order_values: Dict[str, float] = {}
    row_by_node: Dict[str, str] = {}
    group_by_node: Dict[str, str] = {}
    nodes_by_row: Dict[str, List[str]] = {row: [] for row in ROW_ORDER}

    for node in topology_nodes:
        node_id = _to_str(node.get("id"))
        params = dict(node.get("params", {}) or {})
        backend_type = _to_str(node.get("type")) or "PassiveComponent"

        row_name = _assign_row(node_id, backend_type, params, block_by_node)
        group_name = _assign_group(row_name, node_id, params)
        order_value = _extract_order_value(params, int(depth.get(node_id, 0)))
        if (
            row_name == ROW_3_PEM_STORAGE
            and group_name == "Storage & Dispatch"
            and not _has_valid_process_step(params)
        ):
            # Keep storage/offtake hardware at the far right in row 3.
            order_value += 900.0
        if row_name == ROW_3_PEM_STORAGE and group_name == "Storage & Dispatch":
            upper = node_id.upper()
            if upper.startswith("TRUCK_"):
                order_value += 1500.0
            elif upper.startswith("HP_"):
                order_value += 1300.0
            elif upper.startswith("LP_"):
                order_value += 1100.0
            elif upper.startswith("H2_"):
                order_value += 900.0

        order_values[node_id] = order_value
        row_by_node[node_id] = row_name
        group_by_node[node_id] = group_name
        nodes_by_row.setdefault(row_name, []).append(node_id)

    positions: Dict[str, Tuple[float, float]] = {}
    row_bounds: Dict[str, Dict[str, float]] = {row: _minmax() for row in ROW_ORDER}
    group_bounds: Dict[str, Dict[str, float]] = {}

    for row_name in ROW_ORDER:
        row_nodes = nodes_by_row.get(row_name, [])
        if not row_nodes:
            continue

        groups_in_row: Dict[str, List[str]] = {}
        for node_id in row_nodes:
            groups_in_row.setdefault(group_by_node[node_id], []).append(node_id)

        group_order = sorted(
            groups_in_row.keys(),
            key=lambda group_name: (
                min(order_values[nid] for nid in groups_in_row[group_name]),
                group_name,
            ),
        )

        row_base_y = START_Y + ROW_ORDER_INDEX[row_name] * ROW_GAP
        group_start_x = START_X

        for lane_idx, group_name in enumerate(group_order):
            lane_nodes = sorted(
                groups_in_row[group_name],
                key=lambda nid: (order_values[nid], nid),
            )
            lane_y = row_base_y + lane_idx * LANE_GAP
            bounds = group_bounds.setdefault(group_name, _minmax())

            for local_idx, node_id in enumerate(lane_nodes):
                x = group_start_x + float(local_idx) * X_SPACING
                y = lane_y

                positions[node_id] = (x, y)
                _update_minmax(row_bounds[row_name], x, y)
                _update_minmax(bounds, x, y)

            group_width = max(len(lane_nodes) - 1, 0) * X_SPACING
            group_start_x += group_width + INTER_GROUP_GAP

    zones_meta: Dict[str, Any] = {}
    for row_name in ROW_ORDER:
        rect = _safe_rect(row_bounds[row_name])
        if rect is None:
            continue
        x, y, w, h = rect
        color = list(ROW_COLORS.get(row_name, (200, 200, 200)))
        zones_meta[row_name] = {
            "x": x - ZONE_MARGIN,
            "y": y - ZONE_MARGIN,
            "w": w + 2 * ZONE_MARGIN,
            "h": h + 2 * ZONE_MARGIN,
            "color": color,
        }

    groups_meta: Dict[str, Any] = {}
    for group_name, bounds in sorted(group_bounds.items()):
        rect = _safe_rect(bounds)
        if rect is None:
            continue
        x, y, w, h = rect
        color = list(GROUP_COLORS.get(group_name, (210, 210, 210)))
        groups_meta[group_name] = {
            "x": x - GROUP_MARGIN,
            "y": y - GROUP_MARGIN,
            "w": w + 2 * GROUP_MARGIN,
            "h": h + 2 * GROUP_MARGIN,
            "color": color,
            "row": min(
                (row_by_node[nid] for nid, g in group_by_node.items() if g == group_name),
                key=lambda row: ROW_ORDER_INDEX.get(row, 999),
            ),
        }

    visual_layout = {
        "layout_mode": INDUSTRIAL_LAYOUT_MODE,
        "layout_schema_version": INDUSTRIAL_LAYOUT_SCHEMA_VERSION,
        "zones": zones_meta,
        "groups": groups_meta,
        "row_order": list(ROW_ORDER),
        "node_zone_map": dict(row_by_node),
        "node_group_map": dict(group_by_node),
        "layout_constants": {
            "START_X": START_X,
            "START_Y": START_Y,
            "X_SPACING": X_SPACING,
            "INTER_GROUP_GAP": INTER_GROUP_GAP,
            "ROW_GAP": ROW_GAP,
            "LANE_GAP": LANE_GAP,
            "ROW_SPACING": ROW_SPACING,
            "ZONE_MARGIN": ZONE_MARGIN,
            "GROUP_MARGIN": GROUP_MARGIN,
        },
        "spacing_policy": "group_local_rank",
    }
    return positions, visual_layout
