"""
Cost Estimation Strategies

Strategy pattern implementation for flexible cost estimation methods:
- TurtonStrategy: Turton correlation (2018 edition)
- VendorQuoteStrategy: Direct vendor quotes
- PercentageStrategy: Percentage of other costs (for indirect items)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np
import logging

from h2_plant.economics.models import (
    CostCoefficients,
    CapexEntry,
    AACECostClass,
    CEPCIData,
)

logger = logging.getLogger(__name__)


class CostStrategy(ABC):
    """Abstract base class for cost estimation strategies."""
    
    @abstractmethod
    def calculate(
        self,
        design_capacity: float,
        coefficients: Optional[CostCoefficients],
        cepci: CEPCIData,
        **kwargs
    ) -> Tuple[Optional[float], Optional[float], str, AACECostClass]:
        """
        Calculate bare module cost.
        
        Args:
            design_capacity: Design capacity in appropriate units
            coefficients: Cost coefficients (may be None for some strategies)
            cepci: CEPCI data for inflation adjustment
            **kwargs: Strategy-specific parameters
            
        Returns:
            Tuple of (C_p0, C_BM, formula_string, cost_class)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass


class TurtonStrategy(CostStrategy):
    """
    Turton correlation cost estimation (2018 edition).
    
    Formula: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2
    
    Module cost calculation:
    - B-factor method: C_BM = Cp0 * (B1 + B2*F_m*F_p) * F_multi
    - Simple method: C_BM = Cp0 * F_BM * F_m
    """
    
    @property
    def name(self) -> str:
        return "Turton Correlation (2018)"
    
    def calculate(
        self,
        design_capacity: float,
        coefficients: Optional[CostCoefficients],
        cepci: CEPCIData,
        **kwargs
    ) -> Tuple[Optional[float], Optional[float], str, AACECostClass]:
        """Calculate cost using Turton correlation."""
        
        if coefficients is None:
            return None, None, "Missing coefficients", AACECostClass.CLASS_5
        
        if design_capacity <= 0:
            return None, None, "Invalid capacity (≤0)", AACECostClass.CLASS_5
        
        # Check bounds
        out_of_bounds = False
        if coefficients.capacity_min is not None and design_capacity < coefficients.capacity_min:
            out_of_bounds = True
            logger.warning(f"Capacity {design_capacity} below minimum {coefficients.capacity_min}")
        if coefficients.capacity_max is not None and design_capacity > coefficients.capacity_max:
            out_of_bounds = True
            logger.warning(f"Capacity {design_capacity} above maximum {coefficients.capacity_max}")
        
        # Base cost calculation
        log_A = np.log10(design_capacity)
        log_Cp0 = coefficients.K1 + coefficients.K2 * log_A + coefficients.K3 * (log_A ** 2)
        Cp0 = 10 ** log_Cp0
        
        # Inflate to current year
        Cp_current = Cp0 * cepci.inflation_factor
        
        # Module cost calculation
        if coefficients.uses_b_factors:
            # B-factor method (vessels, heat exchangers)
            F_m = coefficients.F_m
            F_p = coefficients.F_p
            B1 = coefficients.B1
            B2 = coefficients.B2
            F_multi = coefficients.F_multi
            
            C_BM = Cp_current * (B1 + B2 * F_m * F_p) * F_multi
            
            formula = (
                f"log10(Cp0) = {coefficients.K1} + {coefficients.K2}*log10({design_capacity:.2f}) "
                f"+ {coefficients.K3}*[log10({design_capacity:.2f})]²; "
                f"Cp0 = ${Cp0:,.0f}; "
                f"C_BM = Cp0 × ({B1} + {B2}×{F_m}×{F_p}) × {F_multi} × CEPCI({cepci.inflation_factor:.4f})"
            )
        else:
            # Simple F_BM method
            F_BM = coefficients.F_BM or 1.0
            F_m = coefficients.F_m
            
            C_BM = Cp_current * F_BM * F_m
            
            formula = (
                f"log10(Cp0) = {coefficients.K1} + {coefficients.K2}*log10({design_capacity:.2f}) "
                f"+ {coefficients.K3}*[log10({design_capacity:.2f})]²; "
                f"Cp0 = ${Cp0:,.0f}; "
                f"C_BM = Cp0 × {F_BM} × {F_m} × CEPCI({cepci.inflation_factor:.4f})"
            )
        
        # Determine cost class based on method quality
        if out_of_bounds:
            cost_class = AACECostClass.CLASS_5  # Extrapolation = lower confidence
        else:
            cost_class = AACECostClass.CLASS_4  # Standard factored estimate
        
        return round(Cp0, 2), round(C_BM, 2), formula, cost_class


class VendorQuoteStrategy(CostStrategy):
    """
    Direct vendor quote strategy.
    
    Uses provided vendor quote with inflation adjustment if needed.
    Higher accuracy than correlations (Class 2-3).
    """
    
    @property
    def name(self) -> str:
        return "Vendor Quote"
    
    def calculate(
        self,
        design_capacity: float,
        coefficients: Optional[CostCoefficients],
        cepci: CEPCIData,
        vendor_quote_usd: Optional[float] = None,
        quote_year: int = 2024,
        quote_cepci: float = 800.0,
        **kwargs
    ) -> Tuple[Optional[float], Optional[float], str, AACECostClass]:
        """Calculate cost from vendor quote."""
        
        if vendor_quote_usd is None or vendor_quote_usd <= 0:
            return None, None, "No valid vendor quote provided", AACECostClass.CLASS_5
        
        # Inflate quote to current year if needed
        if quote_year != cepci.current_year:
            inflation = cepci.current_index / quote_cepci
            C_BM = vendor_quote_usd * inflation
            formula = f"Vendor Quote ${vendor_quote_usd:,.0f} × CEPCI({inflation:.4f}) to {cepci.current_year}"
        else:
            C_BM = vendor_quote_usd
            formula = f"Vendor Quote ${vendor_quote_usd:,.0f} (direct, {quote_year})"
        
        # Vendor quotes have higher accuracy
        cost_class = AACECostClass.CLASS_3
        
        return vendor_quote_usd, round(C_BM, 2), formula, cost_class


class PercentageStrategy(CostStrategy):
    """
    Percentage of capital cost strategy.
    
    For indirect costs, auxiliary equipment, installation, etc.
    """
    
    @property
    def name(self) -> str:
        return "Percentage of Capital"
    
    def calculate(
        self,
        design_capacity: float,
        coefficients: Optional[CostCoefficients],
        cepci: CEPCIData,
        base_cost: float = 0.0,
        percentage: float = 0.0,
        **kwargs
    ) -> Tuple[Optional[float], Optional[float], str, AACECostClass]:
        """Calculate cost as percentage of base cost."""
        
        if base_cost <= 0 or percentage <= 0:
            return None, None, "Invalid base cost or percentage", AACECostClass.CLASS_5
        
        C_BM = base_cost * (percentage / 100.0)
        formula = f"{percentage}% of ${base_cost:,.0f}"
        
        return None, round(C_BM, 2), formula, AACECostClass.CLASS_4


class ScalingStrategy(CostStrategy):
    """
    Six-tenths rule scaling strategy.
    
    Scales known cost to new capacity using power law:
    C2 = C1 * (S2/S1)^n where n ≈ 0.6
    """
    
    def __init__(self, scaling_exponent: float = 0.6):
        self.scaling_exponent = scaling_exponent
    
    @property
    def name(self) -> str:
        return f"Six-Tenths Rule (n={self.scaling_exponent})"
    
    def calculate(
        self,
        design_capacity: float,
        coefficients: Optional[CostCoefficients],
        cepci: CEPCIData,
        reference_capacity: float = 0.0,
        reference_cost: float = 0.0,
        reference_year: int = 2020,
        reference_cepci: float = 596.2,
        **kwargs
    ) -> Tuple[Optional[float], Optional[float], str, AACECostClass]:
        """Calculate cost using six-tenths rule scaling."""
        
        if reference_capacity <= 0 or reference_cost <= 0 or design_capacity <= 0:
            return None, None, "Invalid reference data or capacity", AACECostClass.CLASS_5
        
        # Scale to new capacity
        scaling_factor = (design_capacity / reference_capacity) ** self.scaling_exponent
        scaled_cost = reference_cost * scaling_factor
        
        # Inflate to current year
        inflation = cepci.current_index / reference_cepci
        C_BM = scaled_cost * inflation
        
        formula = (
            f"${reference_cost:,.0f} × ({design_capacity:.1f}/{reference_capacity:.1f})^{self.scaling_exponent} "
            f"× CEPCI({inflation:.4f})"
        )
        
        return round(scaled_cost, 2), round(C_BM, 2), formula, AACECostClass.CLASS_4


# Strategy registry for factory pattern
STRATEGY_REGISTRY: Dict[str, CostStrategy] = {
    "turton": TurtonStrategy(),
    "vendor_quote": VendorQuoteStrategy(),
    "percentage": PercentageStrategy(),
    "scaling": ScalingStrategy(),
}


def get_strategy(name: str) -> CostStrategy:
    """
    Get cost estimation strategy by name.
    
    Args:
        name: Strategy name (turton, vendor_quote, percentage, scaling)
        
    Returns:
        CostStrategy instance
        
    Raises:
        ValueError: If strategy not found
    """
    if name.lower() not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown cost strategy '{name}'. Available: {available}")
    
    return STRATEGY_REGISTRY[name.lower()]
