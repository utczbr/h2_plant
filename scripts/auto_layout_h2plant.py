import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Allow running as standalone script from repo root or scripts/ folder.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from h2_plant.gui.core.industrial_layout_engine import compute_industrial_layout


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_backend_type(node_data: Dict[str, Any]) -> str:
    props = dict(node_data.get("properties", {}) or {})
    backend = _to_str(props.get("__scenario_backend_type"))
    if backend:
        return backend
    backend = _to_str(props.get("backend_type"))
    if backend:
        return backend
    display_name = _to_str(node_data.get("display_name"))
    if ":" in display_name:
        return display_name.split(":", 1)[0].strip()
    return "PassiveComponent"


def _extract_params(node_data: Dict[str, Any]) -> Dict[str, Any]:
    props = dict(node_data.get("properties", {}) or {})
    params = dict(props.get("__scenario_params", {}) or {})

    # Keep common surfaced fields available for ordering/grouping fallbacks.
    if "process_step" not in params and props.get("process_step") is not None:
        params["process_step"] = props.get("process_step")
    if "system_group" not in params and props.get("system_group") is not None:
        params["system_group"] = props.get("system_group")
    return params


def _build_topology_nodes(nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
    topology_nodes: List[Dict[str, Any]] = []
    for node_id in sorted(nodes.keys()):
        node_data = dict(nodes.get(node_id, {}) or {})
        topology_nodes.append(
            {
                "id": node_id,
                "type": _extract_backend_type(node_data),
                "params": _extract_params(node_data),
                "connections": [],
            }
        )
    return topology_nodes


def _build_edge_records(edges: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    edge_records: List[Dict[str, str]] = []
    for edge in edges:
        source_id = _to_str(edge.get("source_node_id"))
        target_id = _to_str(edge.get("target_node_id"))
        source_port = _to_str(edge.get("source_port"))
        target_port = _to_str(edge.get("target_port"))
        if not source_id or not target_id or not source_port or not target_port:
            continue
        edge_records.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "source_port": source_port,
                "target_port": target_port,
            }
        )
    return edge_records


def _topological_depth(node_ids: List[str], edges: List[Dict[str, str]]) -> Dict[str, int]:
    indegree = {node_id: 0 for node_id in node_ids}
    adjacency: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}

    for edge in edges:
        source_id = edge["source_id"]
        target_id = edge["target_id"]
        if source_id not in indegree or target_id not in indegree:
            continue
        adjacency[source_id].append(target_id)
        indegree[target_id] += 1

    for source_id in adjacency:
        adjacency[source_id] = sorted(adjacency[source_id])

    queue = sorted([node_id for node_id, degree in indegree.items() if degree == 0])
    depth = {node_id: 0 for node_id in queue}

    while queue:
        current = queue.pop(0)
        current_depth = depth.get(current, 0)
        for target_id in adjacency.get(current, []):
            next_depth = current_depth + 1
            if next_depth > depth.get(target_id, 0):
                depth[target_id] = next_depth
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                queue.append(target_id)
                queue.sort()

    for node_id in node_ids:
        depth.setdefault(node_id, 0)
    return depth


def auto_layout_h2plant(input_file: Path, output_file: Path) -> Dict[str, Any]:
    data = json.loads(Path(input_file).read_text(encoding="utf-8"))
    nodes = dict(data.get("nodes", {}) or {})
    edges = list(data.get("edges", []) or [])

    topology_nodes = _build_topology_nodes(nodes)
    edge_records = _build_edge_records(edges)
    node_ids = [node["id"] for node in topology_nodes]
    depth = _topological_depth(node_ids, edge_records)
    positions, visual_layout = compute_industrial_layout(topology_nodes, depth, equipment_entries=[])

    updated = 0
    for node_id, (x, y) in positions.items():
        node_data = nodes.get(node_id)
        if not isinstance(node_data, dict):
            continue
        geometry = dict(node_data.get("geometry", {}) or {})
        geometry["x"] = float(x)
        geometry["y"] = float(y)
        node_data["geometry"] = geometry
        nodes[node_id] = node_data
        updated += 1

    data["nodes"] = nodes
    topology_analysis = dict(data.get("topology_analysis", {}) or {})
    topology_analysis["visual_layout"] = visual_layout
    data["topology_analysis"] = topology_analysis

    Path(output_file).write_text(json.dumps(data, indent=2), encoding="utf-8")

    row_counts: Dict[str, int] = {}
    for row in visual_layout.get("node_zone_map", {}).values():
        row_counts[row] = row_counts.get(row, 0) + 1
    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "total_nodes": len(nodes),
        "updated_nodes": updated,
        "row_counts": row_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-layout a .h2plant file into industrial PFD grouping.")
    parser.add_argument("--input", required=True, help="Input .h2plant file path")
    parser.add_argument("--output", help="Output .h2plant file path (default: <input>_organized.h2plant)")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a new file",
    )
    args = parser.parse_args()

    input_file = Path(args.input).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if args.in_place:
        output_file = input_file
    elif args.output:
        output_file = Path(args.output).resolve()
    else:
        output_file = input_file.with_name(f"{input_file.stem}_organized{input_file.suffix}")

    summary = auto_layout_h2plant(input_file=input_file, output_file=output_file)
    print(
        "Industrial layout completed: "
        f"{summary['updated_nodes']}/{summary['total_nodes']} nodes updated."
    )
    print(f"Output: {summary['output_file']}")
    print(f"Row counts: {summary['row_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
