"""
H2 Plant Economics Package

Professional CAPEX configuration generator with:
- Type-safe Pydantic models
- Multiple cost estimation strategies (Turton, DACE, vendor quotes)
- External YAML configuration for equipment mappings
- AACE cost class metadata and uncertainty propagation
"""

from h2_plant.economics.capex_generator import CapexGenerator
from h2_plant.economics.models import (
    EquipmentMapping,
    CostCoefficients,
    CapexEntry,
    CapexReport,
    AACECostClass,
)
from h2_plant.economics.cost_strategies import (
    CostStrategy,
    TurtonStrategy,
    VendorQuoteStrategy,
)

__all__ = [
    "CapexGenerator",
    "EquipmentMapping",
    "CostCoefficients",
    "CapexEntry",
    "CapexReport",
    "AACECostClass",
    "CostStrategy",
    "TurtonStrategy",
    "VendorQuoteStrategy",
]
