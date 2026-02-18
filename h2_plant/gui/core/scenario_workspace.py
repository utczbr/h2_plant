"""
Scenario workspace helpers.

This module creates copy-only scenario workspaces under GUI generated folders so
the original YAML sources are never modified in place.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_TOPOLOGY_FILE = "plant_topology.yaml"
DEFAULT_PHYSICS_FILE = "physics_parameters.yaml"
DEFAULT_ECONOMICS_FILE = "economics_parameters.yaml"
DEFAULT_SIMULATION_FILE = "simulation_config.yaml"
DEFAULT_EQUIPMENT_FILE = "Economics/equipment_mappings.yaml"
DEFAULT_OPEX_FILE = "Economics/opex_config.yaml"


def _default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[1] / "layouts" / "generated"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def resolve_manifest_file(
    manifest: Dict[str, Any],
    key: str,
    default_reference: Optional[str] = None,
) -> Optional[Path]:
    """
    Resolve a file path reference from a scenario manifest.

    Relative references are resolved against ``manifest['scenarios_dir']``.
    """
    reference = manifest.get(key, default_reference)
    if not reference:
        return None

    ref_path = Path(str(reference))
    if ref_path.is_absolute():
        return ref_path

    scenarios_dir = manifest.get("scenarios_dir")
    if scenarios_dir:
        return Path(str(scenarios_dir)) / ref_path
    return ref_path


def copy_into_workspace(
    source_path: Path,
    workspace_dir: Path,
    canonical_relative_path: str,
    *,
    required: bool = True,
) -> Optional[Path]:
    """
    Copy a single file into a workspace canonical location.

    Returns destination path, or ``None`` when source is optional and missing.
    """
    source = Path(source_path)
    if not source.exists():
        if required:
            raise FileNotFoundError(f"Required scenario file not found: {source}")
        return None

    destination = workspace_dir / canonical_relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def create_workspace_from_sources(
    source_manifest: Dict[str, Any],
    *,
    workspace_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Create a copy-only workspace and return an updated manifest pointing to it.
    """
    source_manifest = dict(source_manifest or {})

    source_dir = source_manifest.get("scenarios_dir")
    if not source_dir:
        raise ValueError("source_manifest missing 'scenarios_dir'")
    source_dir_path = Path(str(source_dir))
    if not source_dir_path.exists():
        raise FileNotFoundError(f"Source scenarios directory not found: {source_dir_path}")

    root = Path(workspace_root) if workspace_root else _default_workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    workspace_dir = root / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    src_topology = resolve_manifest_file(source_manifest, "topology_file", DEFAULT_TOPOLOGY_FILE)
    src_physics = resolve_manifest_file(source_manifest, "physics_file", DEFAULT_PHYSICS_FILE)
    src_economics = resolve_manifest_file(source_manifest, "economics_file", DEFAULT_ECONOMICS_FILE)
    src_sim = resolve_manifest_file(
        source_manifest,
        "simulation_config_file",
        DEFAULT_SIMULATION_FILE,
    )
    src_equipment = resolve_manifest_file(source_manifest, "equipment_file", DEFAULT_EQUIPMENT_FILE)
    src_opex = resolve_manifest_file(source_manifest, "opex_file", DEFAULT_OPEX_FILE)

    copied: Dict[str, Path] = {}
    copied[DEFAULT_TOPOLOGY_FILE] = copy_into_workspace(
        src_topology,
        workspace_dir,
        DEFAULT_TOPOLOGY_FILE,
        required=True,
    )
    copied[DEFAULT_PHYSICS_FILE] = copy_into_workspace(
        src_physics,
        workspace_dir,
        DEFAULT_PHYSICS_FILE,
        required=True,
    )
    copied[DEFAULT_ECONOMICS_FILE] = copy_into_workspace(
        src_economics,
        workspace_dir,
        DEFAULT_ECONOMICS_FILE,
        required=True,
    )
    copied[DEFAULT_SIMULATION_FILE] = copy_into_workspace(
        src_sim,
        workspace_dir,
        DEFAULT_SIMULATION_FILE,
        required=True,
    )

    equipment_dst = copy_into_workspace(
        src_equipment,
        workspace_dir,
        DEFAULT_EQUIPMENT_FILE,
        required=False,
    )
    if equipment_dst is not None:
        copied[DEFAULT_EQUIPMENT_FILE] = equipment_dst

    opex_dst = copy_into_workspace(
        src_opex,
        workspace_dir,
        DEFAULT_OPEX_FILE,
        required=False,
    )
    if opex_dst is not None:
        copied[DEFAULT_OPEX_FILE] = opex_dst

    manifest = dict(source_manifest)
    manifest["scenarios_dir"] = str(workspace_dir)
    source_scenarios_dir = source_manifest.get("source_scenarios_dir") or str(source_dir_path)
    manifest["source_scenarios_dir"] = str(source_scenarios_dir)
    manifest["topology_file"] = DEFAULT_TOPOLOGY_FILE
    manifest["topology_file_name"] = DEFAULT_TOPOLOGY_FILE
    manifest["physics_file"] = DEFAULT_PHYSICS_FILE
    manifest["economics_file"] = DEFAULT_ECONOMICS_FILE
    manifest["simulation_config_file"] = DEFAULT_SIMULATION_FILE
    manifest["workspace_generated_at"] = datetime.now(timezone.utc).isoformat()

    if DEFAULT_EQUIPMENT_FILE in copied:
        manifest["equipment_file"] = DEFAULT_EQUIPMENT_FILE
    else:
        manifest.pop("equipment_file", None)

    if DEFAULT_OPEX_FILE in copied:
        manifest["opex_file"] = DEFAULT_OPEX_FILE
    else:
        manifest.pop("opex_file", None)

    manifest["file_hashes"] = {
        rel_path: _sha256_file(dst_path)
        for rel_path, dst_path in copied.items()
    }
    return manifest


def refresh_manifest_file_hashes(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recompute manifest file hashes for known manifest file references.
    """
    updated = dict(manifest or {})

    hash_refs = [
        ("topology_file", DEFAULT_TOPOLOGY_FILE),
        ("physics_file", DEFAULT_PHYSICS_FILE),
        ("economics_file", DEFAULT_ECONOMICS_FILE),
        ("simulation_config_file", DEFAULT_SIMULATION_FILE),
        ("equipment_file", DEFAULT_EQUIPMENT_FILE),
        ("opex_file", DEFAULT_OPEX_FILE),
    ]

    hashes: Dict[str, str] = {}
    for key, default_ref in hash_refs:
        path = resolve_manifest_file(updated, key, default_ref)
        reference = updated.get(key, default_ref)
        if path and path.exists():
            hashes[str(reference)] = _sha256_file(path)
    updated["file_hashes"] = hashes
    return updated


def load_yaml_preview(path: Path) -> Dict[str, Any]:
    """
    Load YAML for preview/validation screens.
    """
    target = Path(path)
    with open(target, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {target}")
    return data
