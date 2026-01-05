"""
Cooling components for the H2 Plant simulation.

Components:
- DryCooler: Two-stage indirect cooling system (Gas → Glycol → Air)
- DryCoolerSimplified: Simplified enthalpy-based cooling with flash calculations
"""

from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.cooling.dry_cooler_simplified import DryCoolerSimplified

__all__ = [
    'DryCooler',
    'DryCoolerSimplified',
]
