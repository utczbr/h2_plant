"""
OPEX Estimation Strategies

Strategy pattern for Operating Expense calculations:
- VariableStrategy: Price * Quantity (from simulation history)
- FixedStrategy: Manual fixed annual cost (e.g., insurance, admin)
- FactorStrategy: % of CAPEX or Labor (e.g., Maintenance = 7% FCI)
- TurtonLaborStrategy: Estimating operators based on process steps
"""

from abc import ABC, abstractmethod
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class OpexStrategy(ABC):
    """Abstract base class for OPEX strategies."""
    
    @abstractmethod
    def calculate(
        self,
        quantity: float,
        price: float,
        base_cost: float = 0.0,
        **kwargs
    ) -> Tuple[float, str]:
        """
        Calculate annual OPEX cost.
        
        Args:
            quantity: Annualized quantity (kWh, kg, hours, or 1.0 for fixed)
            price: Unit price or fixed sum or factor percentage
            base_cost: Base for factor calculations (e.g., FCI or Labor Cost)
            **kwargs: Strategy-specific parameters
            
        Returns:
            Tuple of (Annual Cost, Formula String)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass


class VariableStrategy(OpexStrategy):
    """
    Direct variable cost: Cost = Annual Quantity × Unit Price
    
    Used for: Electricity, Water, Raw Materials, Biogas.
    """
    
    @property
    def name(self) -> str:
        return "Variable"
    
    def calculate(
        self,
        quantity: float,
        price: float,
        base_cost: float = 0.0,
        **kwargs
    ) -> Tuple[float, str]:
        cost = quantity * price
        unit = kwargs.get('unit', 'units')
        formula = f"{quantity:,.2f} {unit} × ${price:.4f}/{unit}"
        return round(cost, 2), formula


class FixedStrategy(OpexStrategy):
    """
    Fixed annual cost.
    
    Used for: Administrative overhead, specific contracts, insurance.
    """
    
    @property
    def name(self) -> str:
        return "Fixed"
    
    def calculate(
        self,
        quantity: float,
        price: float,
        base_cost: float = 0.0,
        **kwargs
    ) -> Tuple[float, str]:
        # Price is treated as the fixed annual sum
        formula = f"Fixed annual cost: ${price:,.2f}"
        return round(price, 2), formula


class FactorStrategy(OpexStrategy):
    """
    Percentage of a base value (usually FCI or Labor).
    
    Used for: Maintenance (7% FCI), Taxes, Insurance, Overhead.
    """
    
    @property
    def name(self) -> str:
        return "Factor"
    
    def calculate(
        self,
        quantity: float,
        price: float,
        base_cost: float = 0.0,
        **kwargs
    ) -> Tuple[float, str]:
        # 'price' is treated as the percentage factor (e.g., 0.07 for 7%)
        cost = base_cost * price
        base_ref = kwargs.get('base_reference', 'Base')
        formula = f"{price*100:.1f}% of {base_ref} (${base_cost:,.0f})"
        return round(cost, 2), formula


class TurtonLaborStrategy(OpexStrategy):
    """
    Turton formula for operating labor estimation.
    
    N = (6.29 + 31.7*P² + 0.23*Nnp)^0.5
    
    Where:
        P = Number of particulate processing steps
        Nnp = Number of non-particulate processing steps
    """
    
    @property
    def name(self) -> str:
        return "Turton Labor"
    
    def calculate(
        self,
        quantity: float,
        price: float,
        base_cost: float = 0.0,
        **kwargs
    ) -> Tuple[float, str]:
        # Extract Turton parameters
        P = kwargs.get('P', 0)
        Nnp = kwargs.get('Nnp', 75)
        shifts = kwargs.get('shifts', 4.8)
        hours_per_year = kwargs.get('hours_per_year', 2080)
        
        # Calculate operators per shift
        N_shift = (6.29 + 31.7 * (P ** 2) + 0.23 * Nnp) ** 0.5
        
        # Total operators needed (accounting for shifts)
        total_operators = N_shift * shifts
        
        # Total annual labor cost
        # price = hourly wage, hours_per_year = hours per operator per year
        cost = total_operators * hours_per_year * price
        
        formula = (
            f"N_shift = (6.29 + 31.7×{P}² + 0.23×{Nnp})^0.5 = {N_shift:.2f}; "
            f"Total = {N_shift:.2f} × {shifts} shifts = {total_operators:.1f} operators × "
            f"{hours_per_year}h × ${price:.2f}/h"
        )
        
        return round(cost, 2), formula


class ScalingStrategy(OpexStrategy):
    """
    Scaling cost based on production quantity.
    
    Cost = Base Cost × (Actual Production / Reference Production)^n
    
    Used for: Scaled maintenance, variable overheads.
    """
    
    @property
    def name(self) -> str:
        return "Scaling"
    
    def calculate(
        self,
        quantity: float,
        price: float,
        base_cost: float = 0.0,
        **kwargs
    ) -> Tuple[float, str]:
        ref_production = kwargs.get('ref_production', 1.0)
        scaling_exponent = kwargs.get('scaling_exponent', 0.6)
        
        if ref_production > 0:
            scaling_factor = (quantity / ref_production) ** scaling_exponent
        else:
            scaling_factor = 1.0
        
        cost = base_cost * scaling_factor
        
        formula = (
            f"${base_cost:,.0f} × ({quantity:,.0f}/{ref_production:,.0f})^{scaling_exponent} "
            f"= ${cost:,.0f}"
        )
        
        return round(cost, 2), formula


def get_opex_strategy(strategy_name: str) -> OpexStrategy:
    """
    Factory function to get strategy by name.
    
    Args:
        strategy_name: One of 'variable', 'fixed', 'factor', 'turton_labor', 'scaling'
        
    Returns:
        Appropriate OpexStrategy instance
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    strategies = {
        'variable': VariableStrategy(),
        'fixed': FixedStrategy(),
        'factor': FactorStrategy(),
        'turton_labor': TurtonLaborStrategy(),
        'scaling': ScalingStrategy(),
    }
    
    strategy = strategies.get(strategy_name.lower())
    if strategy is None:
        available = ', '.join(strategies.keys())
        raise ValueError(f"Unknown OPEX strategy '{strategy_name}'. Available: {available}")
    
    return strategy
