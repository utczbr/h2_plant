"""
Pure helpers for CAPEX/OPEX YAML editing in the GUI economics tab.

These functions do not depend on Qt so they are easy to unit-test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from h2_plant.economics.models import CEPCIData, EquipmentMapping
from h2_plant.economics.opex_models import OpexItemConfig


_CAPACITY_MODES = {"design", "history"}
_LEGACY_NON_TURTON_COEFF_METHODS = {"power_law_scaling"}


def _load_yaml_mapping(text: str) -> Dict[str, Any]:
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping.")
    return data


def load_yaml_text(path: Path) -> str:
    """Read full YAML text from disk."""
    target = Path(path)
    with open(target, "r", encoding="utf-8") as handle:
        return handle.read()


def _prepare_equipment_entry_for_validation(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize legacy/non-Turton coefficient payloads for schema validation.

    Some legacy vendor-quote entries store scaling metadata in ``coefficients``
    using ``cost_method: power_law_scaling`` and do not provide Turton ``K1/K2``.
    ``EquipmentMapping`` expects Turton coefficients, so we validate a copy with
    ``coefficients`` cleared for those explicit methods.
    """
    prepared = dict(entry)
    coeffs = prepared.get("coefficients")
    if not isinstance(coeffs, dict):
        return prepared

    method = str(coeffs.get("cost_method", "")).strip().lower()
    if method in _LEGACY_NON_TURTON_COEFF_METHODS:
        prepared["coefficients"] = None
    return prepared


def validate_capex_yaml_text(text: str) -> Dict[str, Any]:
    """
    Validate CAPEX/equipment-mappings YAML.

    Expected root: mapping.
    Optional validations:
    - ``cepci`` validated by ``CEPCIData``
    - ``capacity_mode`` constrained to design|history
    - ``equipment`` items validated by ``EquipmentMapping``
    """
    data = _load_yaml_mapping(text)

    if "cepci" in data:
        cepci_value = data.get("cepci")
        if not isinstance(cepci_value, dict):
            raise ValueError("'cepci' must be a mapping.")
        CEPCIData.model_validate(cepci_value)

    if "capacity_mode" in data:
        mode = str(data.get("capacity_mode", "")).strip().lower()
        if mode not in _CAPACITY_MODES:
            raise ValueError("capacity_mode must be 'design' or 'history'.")
        data["capacity_mode"] = mode

    if "equipment" in data:
        entries = data.get("equipment")
        if not isinstance(entries, list):
            raise ValueError("'equipment' must be a list.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each 'equipment' item must be a mapping.")
            EquipmentMapping.model_validate(_prepare_equipment_entry_for_validation(entry))

    return data


def validate_opex_yaml_text(text: str) -> Dict[str, Any]:
    """
    Validate OPEX YAML.

    Expected root: mapping with ``opex_items`` list.
    Each item is validated by ``OpexItemConfig``.
    """
    data = _load_yaml_mapping(text)
    items = data.get("opex_items")
    if not isinstance(items, list):
        raise ValueError("Missing or invalid 'opex_items' list in OPEX file.")

    for entry in items:
        if not isinstance(entry, dict):
            raise ValueError("Each 'opex_items' entry must be a mapping.")
        OpexItemConfig.model_validate(entry)
    return data


def extract_general_econ_info(capex_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract general economic fields used by the CAPEX-linked form.

    Returned keys:
    - ``base_year`` (int)
    - ``base_index`` (float)
    - ``current_year`` (int)
    - ``current_index`` (float)
    - ``capacity_mode`` (design|history)
    """
    payload = dict(capex_data or {})
    cepci_raw = payload.get("cepci")

    if isinstance(cepci_raw, dict):
        cepci = CEPCIData.model_validate(cepci_raw)
    else:
        cepci = CEPCIData()

    mode = str(payload.get("capacity_mode", "history")).strip().lower()
    if mode not in _CAPACITY_MODES:
        mode = "history"

    return {
        "base_year": int(cepci.base_year),
        "base_index": float(cepci.base_index),
        "current_year": int(cepci.current_year),
        "current_index": float(cepci.current_index),
        "capacity_mode": mode,
    }


def apply_general_econ_info(capex_data: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply general economic form fields into a CAPEX mapping and return a copy.
    """
    data = dict(capex_data or {})
    info = dict(info or {})

    mode = str(info.get("capacity_mode", data.get("capacity_mode", "history"))).strip().lower()
    if mode not in _CAPACITY_MODES:
        raise ValueError("capacity_mode must be 'design' or 'history'.")

    cepci_payload = {
        "base_year": int(info.get("base_year")),
        "base_index": float(info.get("base_index")),
        "current_year": int(info.get("current_year")),
        "current_index": float(info.get("current_index")),
    }
    cepci = CEPCIData.model_validate(cepci_payload)

    data["cepci"] = {
        "base_year": int(cepci.base_year),
        "base_index": float(cepci.base_index),
        "current_year": int(cepci.current_year),
        "current_index": float(cepci.current_index),
    }
    data["capacity_mode"] = mode
    return data
