#!/usr/bin/env python3
"""
Generate a prebuilt `.h2plant` visual layout from scenario YAML files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from h2_plant.gui.core.prebuilt_visual_layout import generate_layout_file


def main():
    parser = argparse.ArgumentParser(description="Generate scenario visual .h2plant layout")
    parser.add_argument(
        "--scenarios-dir",
        default="scenarios",
        help="Scenario directory containing topology/economics files",
    )
    parser.add_argument(
        "--topology-file",
        default="plant_topology.yaml",
        help="Topology YAML filename (relative to scenarios-dir) or absolute path",
    )
    parser.add_argument(
        "--output",
        default="h2_plant/gui/layouts/plant_topology_visual.h2plant",
        help="Output .h2plant file path",
    )
    parser.add_argument(
        "--project-name",
        default="Plant Topology Visual Twin",
        help="Project metadata name",
    )
    args = parser.parse_args()

    output_path, node_count, edge_count = generate_layout_file(
        output_path=Path(args.output),
        scenarios_dir=args.scenarios_dir,
        topology_file=args.topology_file,
        project_name=args.project_name,
    )
    print(f"Generated {output_path} with {node_count} nodes and {edge_count} edges.")


if __name__ == "__main__":
    main()
