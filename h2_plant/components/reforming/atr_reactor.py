"""
ATR Reactor Module.

This module provides backward compatibility for the ATRReactor class.
For new implementations, use IntegratedATRPlant from integrated_atr_plant.py.

The IntegratedATRPlant replaces the granular ATR_Reactor -> HTWGS -> LTWGS chain
with a single black-box model driven directly by the ATR_linear_regressions.csv data.
"""

# Re-export IntegratedATRPlant as ATRReactor for backward compatibility
from h2_plant.components.reforming.integrated_atr_plant import IntegratedATRPlant

# Alias for backward compatibility
ATRReactor = IntegratedATRPlant
