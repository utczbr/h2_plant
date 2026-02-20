"""
Phased visual + semantic audit harness for scenario mirrors.

This module compares GUI/persisted layout state against backend scenario YAML
and produces deterministic reports for node-level findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from h2_plant.config.loader import ConfigLoader
from h2_plant.gui.core.graph_persistence import GraphPersistenceManager
from h2_plant.gui.core.scenario_param_mapper import backend_to_gui_props
from h2_plant.gui.core.visual_layout_policy import PHASE_DEFINITIONS

PHASE_ORDER = {name: idx for idx, (name, _) in enumerate(PHASE_DEFINITIONS)}
PHASE_BY_TYPE = {backend_type: phase for phase, types in PHASE_DEFINITIONS for backend_type in types}

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}

REQUIRED_NODE_GEOMETRY_FIELDS = (
    "x",
    "y",
    "width",
    "height",
    "color",
    "border_color",
    "text_color",
    "selected",
    "disabled",
    "collapsed",
)
REQUIRED_EDGE_GEOMETRY_FIELDS = (
    "source_node_id",
    "target_node_id",
    "source_port",
    "target_port",
    "flow_type",
    "color",
    "width",
    "style",
    "selected",
    "waypoints",
)

_REAL_RUN_REQUIRED_PEM_KEYS = {"max_power_mw", "base_efficiency", "kwh_per_kg"}
_REAL_RUN_REQUIRED_SOEC_KEYS = {"max_power_nominal_mw", "optimal_limit"}


@dataclass(frozen=True)
class AuditFinding:
    phase: str
    node_id: str
    backend_type: str
    severity: str
    category: str
    expected: Any
    observed: Any
    fix_hint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "node_id": self.node_id,
            "backend_type": self.backend_type,
            "severity": self.severity,
            "category": self.category,
            "expected": self.expected,
            "observed": self.observed,
            "fix_hint": self.fix_hint,
        }


def _phase_for_type(backend_type: str) -> str:
    return PHASE_BY_TYPE.get(str(backend_type), "Phase 8: Unassigned")


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _safe_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return value
    try:
        parsed = yaml.safe_load(text)
    except Exception:
        return value
    if parsed is None and text.lower() not in {"null", "~"}:
        return value
    return parsed


def _semantic(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _semantic(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_semantic(v) for v in value]
    if isinstance(value, tuple):
        return [_semantic(v) for v in value]
    return _safe_scalar(value)


def _coerce_port_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = [str(item).strip() for item in value]
    else:
        out = [token.strip() for token in str(value).split(",")]
    return [token for token in out if token]


def _coerce_rgb(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return int(value[0]), int(value[1]), int(value[2])
        except (TypeError, ValueError):
            return default
    return default


def _contrast_ratio(rgb_a: Tuple[int, int, int], rgb_b: Tuple[int, int, int]) -> float:
    def channel(c: int) -> float:
        v = max(0.0, min(255.0, float(c))) / 255.0
        if v <= 0.03928:
            return v / 12.92
        return ((v + 0.055) / 1.055) ** 2.4

    def luminance(rgb: Tuple[int, int, int]) -> float:
        r, g, b = rgb
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

    la = luminance(rgb_a)
    lb = luminance(rgb_b)
    high = max(la, lb)
    low = min(la, lb)
    return (high + 0.05) / (low + 0.05)


def _overlap_ratio(rect_a: Tuple[float, float, float, float], rect_b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, aw * ah)
    area_b = max(1.0, bw * bh)
    return inter / min(area_a, area_b)


def _build_backend_index(topology_nodes: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    backend_by_id: Dict[str, Dict[str, Any]] = {}
    outgoing_edges: Dict[str, List[Dict[str, Any]]] = {}
    incoming_edges: Dict[str, List[Dict[str, Any]]] = {}

    for idx, node in enumerate(topology_nodes):
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        backend_type = str(node.get("type", "")).strip()
        params = dict(node.get("params", {}) or {})
        conns = list(node.get("connections", []) or [])
        backend_by_id[node_id] = {
            "order": idx,
            "id": node_id,
            "type": backend_type,
            "params": params,
            "connections": conns,
            "phase": _phase_for_type(backend_type),
        }
        outgoing_edges.setdefault(node_id, [])
        incoming_edges.setdefault(node_id, [])

    for source_id, node in backend_by_id.items():
        for conn in node["connections"]:
            source_port = str(conn.get("source_port", "")).strip()
            target_id = str(conn.get("target_name", "")).strip()
            target_port = str(conn.get("target_port", "")).strip()
            resource_type = str(conn.get("resource_type", "")).strip() or "stream"
            if not source_port or not target_id or not target_port:
                continue
            edge = {
                "source_id": source_id,
                "source_port": source_port,
                "target_id": target_id,
                "target_port": target_port,
                "resource_type": resource_type,
            }
            outgoing_edges[source_id].append(edge)
            incoming_edges.setdefault(target_id, []).append(edge)

    return backend_by_id, outgoing_edges, incoming_edges


def _build_snapshot_edge_index(snapshot_edges: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    outgoing: Dict[str, List[Dict[str, Any]]] = {}
    incoming: Dict[str, List[Dict[str, Any]]] = {}
    for edge in snapshot_edges:
        source_id = str(edge.get("source_node_id", "")).strip()
        target_id = str(edge.get("target_node_id", "")).strip()
        source_port = str(edge.get("source_port", "")).strip()
        target_port = str(edge.get("target_port", "")).strip()
        flow_type = str(edge.get("flow_type", "")).strip() or "stream"
        if not source_id or not target_id or not source_port or not target_port:
            continue
        packed = {
            "source_id": source_id,
            "source_port": source_port,
            "target_id": target_id,
            "target_port": target_port,
            "resource_type": flow_type,
        }
        outgoing.setdefault(source_id, []).append(packed)
        incoming.setdefault(target_id, []).append(packed)
    return outgoing, incoming


def _classify_runs(generated_root: Path) -> List[Dict[str, Any]]:
    runs = sorted([p for p in generated_root.iterdir() if p.is_dir() and p.name.startswith("run_")])
    results: List[Dict[str, Any]] = []
    for run_dir in runs:
        physics = run_dir / "physics_parameters.yaml"
        classification = "synthetic_or_legacy"
        reason = "missing_physics_file"
        if physics.exists():
            try:
                pdata = _safe_load_yaml(physics)
                pem = pdata.get("pem_system") or {}
                soec = pdata.get("soec_cluster") or {}
                pem_ok = isinstance(pem, dict) and _REAL_RUN_REQUIRED_PEM_KEYS.issubset(set(pem.keys()))
                soec_ok = isinstance(soec, dict) and _REAL_RUN_REQUIRED_SOEC_KEYS.issubset(set(soec.keys()))
                if pem_ok and soec_ok:
                    classification = "real"
                    reason = "physics_has_required_pem_soec_fields"
                else:
                    classification = "synthetic_or_legacy"
                    reason = "physics_missing_required_pem_or_soec_fields"
            except Exception as exc:
                reason = f"physics_unreadable:{type(exc).__name__}"
        results.append({"run_dir": str(run_dir), "classification": classification, "reason": reason})
    return results


def _add_finding(
    findings: List[AuditFinding],
    phase: str,
    node_id: str,
    backend_type: str,
    severity: str,
    category: str,
    expected: Any,
    observed: Any,
    fix_hint: str,
) -> None:
    findings.append(
        AuditFinding(
            phase=phase,
            node_id=node_id,
            backend_type=backend_type,
            severity=severity,
            category=category,
            expected=expected,
            observed=observed,
            fix_hint=fix_hint,
        )
    )


def _phase_status(critical_count: int, major_count: int) -> str:
    if critical_count > 0:
        return "blocked_critical"
    if major_count > 0:
        return "blocked_major"
    return "pass"


def _report_sort_key(finding: AuditFinding, backend_order: Dict[str, int]) -> Tuple[int, int, int, str]:
    return (
        PHASE_ORDER.get(finding.phase, 999),
        backend_order.get(finding.node_id, 999999),
        SEVERITY_ORDER.get(finding.severity, 99),
        finding.category,
    )


def run_audit(
    scenarios_dir: str = "scenarios",
    topology_file: str = "plant_topology.yaml",
    layout_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full phased node visual-semantic audit and write JSON reports."""
    scenarios_root = Path(scenarios_dir).resolve()
    topology_path = (Path(topology_file) if Path(topology_file).is_absolute() else scenarios_root / topology_file).resolve()
    if layout_path:
        layout_file = Path(layout_path).resolve()
    else:
        layout_file = (Path(__file__).resolve().parents[1] / "layouts" / "plant_topology_visual.h2plant").resolve()

    topology_data = _safe_load_yaml(topology_path)
    topology_nodes = list(topology_data.get("nodes", []) or [])
    backend_by_id, expected_outgoing, expected_incoming = _build_backend_index(topology_nodes)
    backend_order = {node_id: info["order"] for node_id, info in backend_by_id.items()}

    manager = GraphPersistenceManager()
    raw_layout_data = json.loads(layout_file.read_text(encoding="utf-8"))
    snapshot = manager.load(str(layout_file))
    actual_outgoing, actual_incoming = _build_snapshot_edge_index(snapshot.edges)

    baseline_notes: List[Dict[str, Any]] = []
    if raw_layout_data.get("visual_fidelity_schema_version") is None:
        baseline_notes.append(
            {
                "severity": "minor",
                "category": "legacy_layout_schema_marker_missing",
                "detail": "visual_fidelity_schema_version is missing. Layout remains loadable.",
                "path": str(layout_file),
            }
        )

    runs = _classify_runs((Path(__file__).resolve().parents[1] / "layouts" / "generated").resolve())
    legacy_runs = [run for run in runs if run["classification"] != "real"]
    if legacy_runs:
        baseline_notes.append(
            {
                "severity": "minor",
                "category": "legacy_or_synthetic_run_artifacts_present",
                "detail": f"{len(legacy_runs)} run artifact(s) classified as synthetic/legacy (non-blocking).",
            }
        )

    findings: List[AuditFinding] = []
    node_results: List[Dict[str, Any]] = []

    # Per-node mirror checks.
    for node_id, backend in sorted(backend_by_id.items(), key=lambda item: item[1]["order"]):
        phase = backend["phase"]
        backend_type = backend["type"]
        snapshot_node = snapshot.nodes.get(node_id)

        critical_before = sum(1 for f in findings if f.severity == "critical")
        major_before = sum(1 for f in findings if f.severity == "major")
        minor_before = sum(1 for f in findings if f.severity == "minor")

        if snapshot_node is None:
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "critical",
                "missing_node",
                expected="node present in layout snapshot",
                observed="node missing",
                fix_hint="Ensure import/prebuilt generation includes this backend node.",
            )
            node_results.append(
                {
                    "phase": phase,
                    "node_id": node_id,
                    "backend_type": backend_type,
                    "status": "blocked_critical",
                    "critical": 1,
                    "major": 0,
                    "minor": 0,
                }
            )
            continue

        properties = dict(snapshot_node.get("properties", {}) or {})
        geometry = dict(snapshot_node.get("geometry", {}) or {})
        display_name = str(snapshot_node.get("display_name", ""))

        component_id = str(properties.get("component_id") or properties.get("__scenario_component_id") or "").strip()
        if component_id != node_id:
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "critical",
                "component_id_mismatch",
                expected=node_id,
                observed=component_id,
                fix_hint="Persist canonical component_id on node properties.",
            )

        actual_backend_type = str(properties.get("__scenario_backend_type") or properties.get("backend_type") or "").strip()
        if actual_backend_type != backend_type:
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "critical",
                "backend_type_mismatch",
                expected=backend_type,
                observed=actual_backend_type,
                fix_hint="Persist __scenario_backend_type from backend mirror.",
            )

        expected_display = f"{backend_type}: {node_id}"
        if display_name != expected_display:
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "minor",
                "display_name_convention",
                expected=expected_display,
                observed=display_name,
                fix_hint="Use '<backend_type>: <component_id>' display naming convention.",
            )

        for field_name in REQUIRED_NODE_GEOMETRY_FIELDS:
            if field_name not in geometry:
                _add_finding(
                    findings,
                    phase,
                    node_id,
                    backend_type,
                    "major",
                    "missing_node_geometry_field",
                    expected=field_name,
                    observed=None,
                    fix_hint="Ensure full NodeGeometry payload is persisted.",
                )

        # Port checks against backend mirror.
        expected_in_ports = sorted({edge["target_port"] for edge in expected_incoming.get(node_id, [])})
        expected_out_ports = sorted({edge["source_port"] for edge in expected_outgoing.get(node_id, [])})
        actual_in_ports = sorted(_coerce_port_list(properties.get("__scenario_inputs")))
        actual_out_ports = sorted(_coerce_port_list(properties.get("__scenario_outputs")))

        for missing_port in sorted(set(expected_in_ports) - set(actual_in_ports)):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "major",
                "missing_input_port",
                expected=missing_port,
                observed=actual_in_ports,
                fix_hint="Restore/import scenario input ports from backend mirror.",
            )
        for extra_port in sorted(set(actual_in_ports) - set(expected_in_ports)):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "minor",
                "unexpected_input_port",
                expected=expected_in_ports,
                observed=extra_port,
                fix_hint="Remove stale input ports not present in backend mirror.",
            )
        for missing_port in sorted(set(expected_out_ports) - set(actual_out_ports)):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "major",
                "missing_output_port",
                expected=missing_port,
                observed=actual_out_ports,
                fix_hint="Restore/import scenario output ports from backend mirror.",
            )
        for extra_port in sorted(set(actual_out_ports) - set(expected_out_ports)):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "minor",
                "unexpected_output_port",
                expected=expected_out_ports,
                observed=extra_port,
                fix_hint="Remove stale output ports not present in backend mirror.",
            )

        # Edge checks.
        expected_edge_set = sorted(
            {
                (
                    edge["source_port"],
                    edge["target_id"],
                    edge["target_port"],
                    edge["resource_type"],
                )
                for edge in expected_outgoing.get(node_id, [])
            }
        )
        actual_edge_set = sorted(
            {
                (
                    edge["source_port"],
                    edge["target_id"],
                    edge["target_port"],
                    edge["resource_type"],
                )
                for edge in actual_outgoing.get(node_id, [])
            }
        )

        for missing_edge in sorted(set(expected_edge_set) - set(actual_edge_set)):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "critical",
                "missing_outgoing_edge",
                expected=missing_edge,
                observed=actual_edge_set,
                fix_hint="Rebuild missing edge from backend mirror topology.",
            )
        for extra_edge in sorted(set(actual_edge_set) - set(expected_edge_set)):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "major",
                "unexpected_outgoing_edge",
                expected=expected_edge_set,
                observed=extra_edge,
                fix_hint="Remove edge not mirrored in backend topology.",
            )

        # Value checks: canonical + mapped visible + unmapped payload.
        scenario_params = properties.get("__scenario_params")
        if not isinstance(scenario_params, dict):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "critical",
                "missing_scenario_params",
                expected="dict",
                observed=type(scenario_params).__name__,
                fix_hint="Persist canonical backend params in __scenario_params.",
            )
        elif _semantic(scenario_params) != _semantic(backend["params"]):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "major",
                "scenario_params_mismatch",
                expected=_semantic(backend["params"]),
                observed=_semantic(scenario_params),
                fix_hint="Synchronize canonical params with backend mirror before save.",
            )

        visible_props = {k: v for k, v in properties.items() if not str(k).startswith("__") and k != "component_id"}
        expected_visible, expected_unmapped = backend_to_gui_props(
            backend_type=backend_type,
            backend_params=dict(backend["params"]),
            available_props=set(visible_props.keys()),
        )
        expected_visible = dict(expected_visible)
        expected_unmapped = dict(expected_unmapped)
        # component_id is an identity field handled by dedicated checks above.
        expected_visible.pop("component_id", None)
        expected_unmapped.pop("component_id", None)

        for key, expected_value in sorted(expected_visible.items()):
            if key not in visible_props:
                _add_finding(
                    findings,
                    phase,
                    node_id,
                    backend_type,
                    "major",
                    "missing_visible_property",
                    expected={key: _semantic(expected_value)},
                    observed=sorted(visible_props.keys()),
                    fix_hint="Expose mapped backend parameter in node UI properties.",
                )
                continue
            observed_value = visible_props[key]
            if _semantic(observed_value) != _semantic(expected_value):
                _add_finding(
                    findings,
                    phase,
                    node_id,
                    backend_type,
                    "major",
                    "visible_property_mismatch",
                    expected={key: _semantic(expected_value)},
                    observed={key: _semantic(observed_value)},
                    fix_hint="Fix unit conversion/type parsing for mapped GUI property.",
                )

        observed_unmapped = properties.get("__scenario_unmapped_params", {})
        if not isinstance(observed_unmapped, dict):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "major",
                "unmapped_payload_type_invalid",
                expected="dict",
                observed=type(observed_unmapped).__name__,
                fix_hint="Persist unmapped params as dict in __scenario_unmapped_params.",
            )
        elif _semantic(observed_unmapped) != _semantic(expected_unmapped):
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "major",
                "unmapped_payload_mismatch",
                expected=_semantic(expected_unmapped),
                observed=_semantic(observed_unmapped),
                fix_hint="Keep unmapped backend params synchronized with mapper output.",
            )

        # Visual readability check.
        bg = _coerce_rgb(geometry.get("color"), (100, 100, 100))
        fg = _coerce_rgb(geometry.get("text_color"), (255, 255, 255))
        contrast = _contrast_ratio(bg, fg)
        if contrast < 2.5:
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "major",
                "low_text_contrast",
                expected="contrast_ratio >= 2.5",
                observed=round(contrast, 3),
                fix_hint="Adjust node/text colors to improve readability.",
            )
        elif contrast < 3.0:
            _add_finding(
                findings,
                phase,
                node_id,
                backend_type,
                "minor",
                "low_text_contrast_warning",
                expected="contrast_ratio >= 3.0",
                observed=round(contrast, 3),
                fix_hint="Increase contrast margin for better readability.",
            )

        critical_after = sum(1 for f in findings if f.severity == "critical")
        major_after = sum(1 for f in findings if f.severity == "major")
        minor_after = sum(1 for f in findings if f.severity == "minor")
        critical_count = critical_after - critical_before
        major_count = major_after - major_before
        minor_count = minor_after - minor_before

        node_results.append(
            {
                "phase": phase,
                "node_id": node_id,
                "backend_type": backend_type,
                "status": _phase_status(critical_count, major_count),
                "critical": critical_count,
                "major": major_count,
                "minor": minor_count,
            }
        )

    # Global overlap and edge readability checks.
    node_rects = []
    for node_id, backend in sorted(backend_by_id.items(), key=lambda item: item[1]["order"]):
        snapshot_node = snapshot.nodes.get(node_id)
        if snapshot_node is None:
            continue
        geometry = dict(snapshot_node.get("geometry", {}) or {})
        x = float(geometry.get("x", 0.0))
        y = float(geometry.get("y", 0.0))
        w = float(geometry.get("width", 100.0))
        h = float(geometry.get("height", 100.0))
        node_rects.append((node_id, backend["phase"], backend["type"], (x, y, w, h)))

    for i in range(len(node_rects)):
        node_i, phase_i, type_i, rect_i = node_rects[i]
        for j in range(i + 1, len(node_rects)):
            node_j, phase_j, type_j, rect_j = node_rects[j]
            ratio = _overlap_ratio(rect_i, rect_j)
            if ratio > 0.25:
                _add_finding(
                    findings,
                    phase_i,
                    node_i,
                    type_i,
                    "major",
                    "severe_node_overlap",
                    expected="overlap_ratio <= 0.25",
                    observed={"peer_node": node_j, "ratio": round(ratio, 4)},
                    fix_hint="Adjust node positions/layout spacing to avoid severe overlap.",
                )
                _add_finding(
                    findings,
                    phase_j,
                    node_j,
                    type_j,
                    "major",
                    "severe_node_overlap",
                    expected="overlap_ratio <= 0.25",
                    observed={"peer_node": node_i, "ratio": round(ratio, 4)},
                    fix_hint="Adjust node positions/layout spacing to avoid severe overlap.",
                )

    for edge in snapshot.edges:
        source = str(edge.get("source_node_id", ""))
        target = str(edge.get("target_node_id", ""))
        flow_type = str(edge.get("flow_type", "")).strip()
        if source and source == target:
            backend_type = backend_by_id.get(source, {}).get("type", "")
            _add_finding(
                findings,
                _phase_for_type(backend_type),
                source,
                backend_type,
                "critical",
                "self_loop_edge",
                expected="source_node_id != target_node_id",
                observed=edge,
                fix_hint="Remove unintended self-loop or split with explicit intermediate node.",
            )
        if flow_type in {"", "default"}:
            backend_type = backend_by_id.get(source, {}).get("type", "")
            _add_finding(
                findings,
                _phase_for_type(backend_type),
                source,
                backend_type,
                "minor",
                "default_edge_flow_type",
                expected="explicit resource flow type",
                observed=flow_type or "<empty>",
                fix_hint="Persist explicit flow/resource type for edge readability.",
            )

        geometry = dict(edge.get("geometry", {}) or {})
        missing_edge_geometry = [field for field in REQUIRED_EDGE_GEOMETRY_FIELDS if field not in geometry]
        if missing_edge_geometry:
            backend_type = backend_by_id.get(source, {}).get("type", "")
            _add_finding(
                findings,
                _phase_for_type(backend_type),
                source,
                backend_type,
                "major",
                "missing_edge_geometry_fields",
                expected=REQUIRED_EDGE_GEOMETRY_FIELDS,
                observed=missing_edge_geometry,
                fix_hint="Persist full edge geometry/style payload for visual fidelity.",
            )

    # Zone readability checks (only when visual_layout metadata is present).
    topology_analysis = dict(snapshot.topology_analysis or {})
    visual_layout = topology_analysis.get("visual_layout")
    if visual_layout and isinstance(visual_layout, dict) and visual_layout.get("zones"):
        from h2_plant.gui.core.visual_layout_policy import ZONE_BY_BACKEND_TYPE as _ZONE_BY_BACKEND_TYPE
        node_zone_map = visual_layout.get("node_zone_map", {})
        zones_meta = visual_layout.get("zones", {})
        for node_id, backend in sorted(backend_by_id.items(), key=lambda item: item[1]["order"]):
            backend_type = backend["type"]
            phase = backend["phase"]
            zone = node_zone_map.get(node_id)

            # Check 1: node should have a valid zone assignment
            if not zone or zone == "Uncategorised":
                expected_zone = _ZONE_BY_BACKEND_TYPE.get(backend_type, "")
                _add_finding(
                    findings,
                    phase,
                    node_id,
                    backend_type,
                    "minor",
                    "missing_visual_zone",
                    expected=expected_zone or "a named process zone",
                    observed=zone or "<absent>",
                    fix_hint=(
                        "Add this backend_type to PHASE_DEFINITIONS in visual_layout_policy.py "
                        "or set system_group in the topology YAML."
                    ),
                )
                continue

            # Check 2: node position should fall within its assigned zone bounding box
            snapshot_node = snapshot.nodes.get(node_id)
            if snapshot_node is None:
                continue
            geometry = dict(snapshot_node.get("geometry", {}) or {})
            node_x = float(geometry.get("x", 0.0))
            node_y = float(geometry.get("y", 0.0))
            zone_rect = zones_meta.get(zone)
            if zone_rect:
                zx = float(zone_rect.get("x", 0.0))
                zy = float(zone_rect.get("y", 0.0))
                zw = float(zone_rect.get("w", 0.0))
                zh = float(zone_rect.get("h", 0.0))
                if not (zx <= node_x <= zx + zw and zy <= node_y <= zy + zh):
                    _add_finding(
                        findings,
                        phase,
                        node_id,
                        backend_type,
                        "minor",
                        "node_outside_zone",
                        expected=f"position within zone '{zone}' bbox ({zx:.0f},{zy:.0f},{zw:.0f}x{zh:.0f})",
                        observed=f"node at ({node_x:.0f},{node_y:.0f})",
                        fix_hint=(
                            "Re-run auto-layout (View → Auto-Layout Report Mode) "
                            "or adjust node position manually."
                        ),
                    )

    # Phase summaries.
    findings.sort(key=lambda f: _report_sort_key(f, backend_order))
    findings_dict = [f.to_dict() for f in findings]

    phase_summary: Dict[str, Dict[str, Any]] = {}
    for phase_name, types in PHASE_DEFINITIONS:
        phase_nodes = [node for node in node_results if node["phase"] == phase_name]
        phase_findings = [f for f in findings if f.phase == phase_name]
        critical_count = sum(1 for finding in phase_findings if finding.severity == "critical")
        major_count = sum(1 for finding in phase_findings if finding.severity == "major")
        minor_count = sum(1 for finding in phase_findings if finding.severity == "minor")
        phase_summary[phase_name] = {
            "types": list(types),
            "checked_nodes": len(phase_nodes),
            "passed_nodes": sum(1 for node in phase_nodes if node["status"] == "pass"),
            "blocked_nodes": sum(1 for node in phase_nodes if node["status"] != "pass"),
            "severity_counts": {
                "critical": critical_count,
                "major": major_count,
                "minor": minor_count,
            },
            "gate_status": _phase_status(critical_count, major_count),
        }

    global_critical = sum(1 for finding in findings if finding.severity == "critical")
    global_major = sum(1 for finding in findings if finding.severity == "major")
    global_minor = sum(1 for finding in findings if finding.severity == "minor")
    if global_critical > 0:
        verdict = "FAIL"
    elif global_major > 0:
        verdict = "FAIL"
    elif baseline_notes:
        verdict = "PASS with legacy notes"
    else:
        verdict = "PASS"

    report = {
        "generated_at": datetime.now().isoformat(),
        "inputs": {
            "scenarios_dir": str(scenarios_root),
            "topology_file": str(topology_path),
            "layout_path": str(layout_file),
        },
        "baseline": {
            "layout_loadable": True,
            "schema_version": raw_layout_data.get("schema_version"),
            "visual_fidelity_schema_version": raw_layout_data.get("visual_fidelity_schema_version"),
            "run_artifacts": runs,
            "notes": baseline_notes,
        },
        "summary": {
            "total_backend_nodes": len(backend_by_id),
            "total_snapshot_nodes": len(snapshot.nodes),
            "total_snapshot_edges": len(snapshot.edges),
            "severity_counts": {
                "critical": global_critical,
                "major": global_major,
                "minor": global_minor,
            },
            "verdict": verdict,
        },
        "phase_summary": phase_summary,
        "node_results": node_results,
        "findings": findings_dict,
    }

    if output_dir:
        target_dir = Path(output_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = (Path(__file__).resolve().parents[1] / "layouts" / "generated" / f"qa_{stamp}").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    audit_path = target_dir / "audit_report.json"
    phase_path = target_dir / "phase_summary.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    phase_path.write_text(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "summary": report["summary"],
                "phase_summary": report["phase_summary"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    report["artifacts"] = {
        "output_dir": str(target_dir),
        "audit_report": str(audit_path),
        "phase_summary": str(phase_path),
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run phased visual-semantic node audit.")
    parser.add_argument("--scenarios-dir", default="scenarios", help="Scenario directory containing topology YAML.")
    parser.add_argument("--topology-file", default="plant_topology.yaml", help="Topology YAML filename or absolute path.")
    parser.add_argument("--layout-path", default=None, help="Optional .h2plant layout path (defaults to canonical prebuilt).")
    parser.add_argument("--output-dir", default=None, help="Optional output dir for reports.")
    args = parser.parse_args(argv)

    report = run_audit(
        scenarios_dir=args.scenarios_dir,
        topology_file=args.topology_file,
        layout_path=args.layout_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Audit report: {report['artifacts']['audit_report']}")
    print(f"Phase summary: {report['artifacts']['phase_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
