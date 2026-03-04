"""
H2 Plant Economics Package

Professional CAPEX configuration generator with:
- Type-safe Pydantic models
- Multiple cost estimation strategies (Turton, DACE, vendor quotes)
- External YAML configuration for equipment mappings
- AACE cost class metadata and uncertainty propagation
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CapexGenerator": ("h2_plant.economics.capex_generator", "CapexGenerator"),
    "EquipmentMapping": ("h2_plant.economics.models", "EquipmentMapping"),
    "CostCoefficients": ("h2_plant.economics.models", "CostCoefficients"),
    "CapexEntry": ("h2_plant.economics.models", "CapexEntry"),
    "CapexReport": ("h2_plant.economics.models", "CapexReport"),
    "AACECostClass": ("h2_plant.economics.models", "AACECostClass"),
    "CostStrategy": ("h2_plant.economics.cost_strategies", "CostStrategy"),
    "TurtonStrategy": ("h2_plant.economics.cost_strategies", "TurtonStrategy"),
    "VendorQuoteStrategy": ("h2_plant.economics.cost_strategies", "VendorQuoteStrategy"),
    "LcohCalculator": ("h2_plant.economics.lcoh_calculator", "LcohCalculator"),
    "LcohReport": ("h2_plant.economics.lcoh_models", "LcohReport"),
    "LcohVariantsReport": ("h2_plant.economics.lcoh_models", "LcohVariantsReport"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
