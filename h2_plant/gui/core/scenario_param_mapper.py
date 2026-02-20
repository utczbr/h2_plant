"""
Shared backend<->GUI parameter mapping helpers for scenario imports/exports.

The canonical scenario model uses backend keys (topology YAML). GUI nodes expose
typed property names that may differ and may use different units.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import yaml


IMPORT_SURFACE_SCHEMA_VERSION = 2


_FRACTION_KEYS = {"base_efficiency", "efficiency", "isentropic_efficiency", "eta_is", "eta_m"}


# GUI property name -> backend key, by backend component type.
_GUI_TO_BACKEND_BY_TYPE: Dict[str, Dict[str, str]] = {
    "PEM": {
        "rated_power_kw": "max_power_mw",
        "efficiency_rated": "base_efficiency",
    },
    "SOEC": {
        "rated_power_kw": "max_power_nominal_mw",
    },
    "PowerTransformer": {
        "max_power_kw": "rated_power_mw",
        "conversion_efficiency": "efficiency",
        "system_group": "system_group",
    },
    "Valve": {
        "outlet_pressure_bar": "P_out_pa",
        "fluid_type": "fluid",
    },
    "WaterPurifier": {
        "output_flow_kgh": "max_flow_kg_h",
    },
    "UltraPureWaterTank": {
        "capacity_m3": "volume_m3",
    },
    "Chiller": {
        "target_temp_c": "target_temp_k",
    },
}


def _parse_scalar(value: Any) -> Any:
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


def _to_float(value: Any) -> Tuple[bool, float]:
    if isinstance(value, bool):
        return False, 0.0
    try:
        return True, float(value)
    except (TypeError, ValueError):
        return False, 0.0


def _backend_to_gui_key_map(backend_type: str) -> Dict[str, str]:
    gui_to_backend = _GUI_TO_BACKEND_BY_TYPE.get(str(backend_type), {})
    return {backend_key: gui_key for gui_key, backend_key in gui_to_backend.items()}


def _gui_to_backend_key_map(backend_type: str) -> Dict[str, str]:
    return dict(_GUI_TO_BACKEND_BY_TYPE.get(str(backend_type), {}))


def effective_backend_key(backend_type: str, gui_key: str) -> str:
    """Return the backend key that *gui_key* would write to for the given type.

    If an explicit mapping exists, returns the mapped backend key.
    Otherwise returns *gui_key* unchanged (direct-key overlay).
    """
    return _GUI_TO_BACKEND_BY_TYPE.get(str(backend_type), {}).get(str(gui_key), str(gui_key))


def _convert_backend_to_gui(
    backend_key: str,
    gui_key: str,
    value: Any,
) -> Any:
    ok, num = _to_float(value)

    if ok and backend_key in {"max_power_mw", "max_power_nominal_mw"} and gui_key == "rated_power_kw":
        return num * 1000.0
    if ok and backend_key == "rated_power_mw" and gui_key == "max_power_kw":
        return num * 1000.0
    if ok and backend_key == "target_temp_k" and gui_key == "target_temp_c":
        return num - 273.15
    if ok and backend_key == "P_out_pa" and gui_key == "outlet_pressure_bar":
        return num / 1e5
    if ok and backend_key in _FRACTION_KEYS and gui_key in {
        "efficiency_rated",
        "conversion_efficiency",
        "efficiency",
        "isentropic_efficiency",
        "eta_is",
        "eta_m",
    }:
        if num <= 1.0:
            return num * 100.0
        return num

    return value


def _convert_gui_to_backend(
    gui_key: str,
    backend_key: str,
    value: Any,
) -> Any:
    ok, num = _to_float(value)

    if ok and backend_key in {"max_power_mw", "max_power_nominal_mw"} and gui_key == "rated_power_kw":
        return num / 1000.0
    if ok and backend_key == "rated_power_mw" and gui_key == "max_power_kw":
        return num / 1000.0
    if ok and backend_key == "target_temp_k" and gui_key == "target_temp_c":
        return num + 273.15
    if ok and backend_key == "P_out_pa" and gui_key == "outlet_pressure_bar":
        return num * 1e5
    if ok and backend_key in _FRACTION_KEYS:
        if num > 1.0:
            return num / 100.0
        return num

    return value


def backend_to_gui_props(
    backend_type: str,
    backend_params: Dict[str, Any],
    available_props: Iterable[str] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Map canonical backend params to GUI-facing properties.

    Returns:
        (mapped_gui_props, unmapped_backend_params)
    """
    params = dict(backend_params or {})
    available = {str(key) for key in (available_props or [])}
    enforce_available = bool(available)

    backend_to_gui = _backend_to_gui_key_map(backend_type)

    mapped: Dict[str, Any] = {}
    unmapped: Dict[str, Any] = {}

    for backend_key, raw_value in params.items():
        gui_key = backend_to_gui.get(str(backend_key), str(backend_key))
        gui_value = _convert_backend_to_gui(str(backend_key), gui_key, raw_value)

        if enforce_available and gui_key not in available:
            unmapped[str(backend_key)] = raw_value
            continue

        mapped[gui_key] = gui_value

    return mapped, unmapped


def gui_to_backend_overlay(
    backend_type: str,
    gui_props: Dict[str, Any],
    base_backend_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Overlay GUI property edits onto canonical backend params.

    Rules:
    - Mapped GUI keys always update their canonical backend key.
    - Direct key overlays only apply to keys already present in canonical params.
    - Values are scalar-parsed and unit-converted where needed.
    """
    merged = dict(base_backend_params or {})
    gui_to_backend = _gui_to_backend_key_map(backend_type)

    for gui_key, raw_value in (gui_props or {}).items():
        key = str(gui_key)
        if key.startswith("_"):
            continue

        backend_key = gui_to_backend.get(key)
        if backend_key is None:
            if key not in merged:
                continue
            backend_key = key

        parsed = _parse_scalar(raw_value)
        converted = _convert_gui_to_backend(key, backend_key, parsed)

        # Never overwrite a non-empty canonical string with an empty GUI value.
        # This prevents uninitialised enum widgets from blanking fields like 'fluid'.
        canonical = merged.get(backend_key)
        if isinstance(converted, str) and converted == "" and isinstance(canonical, str) and canonical != "":
            continue

        merged[backend_key] = converted

    return merged
