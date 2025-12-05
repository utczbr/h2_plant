"""
Allocation strategies for dual-path hydrogen production.

Implements various algorithms for splitting demand across production pathways:
- Cost-optimal: Minimize total production cost
- Priority-based: Maximize usage of preferred source
- Balanced: Equal utilization
- Emissions-optimal: Minimize CO2 emissions
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

from h2_plant.core.enums import AllocationStrategy


class BaseAllocationStrategy(ABC):
    """Base class for allocation strategies."""
    
    @abstractmethod
    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """
        Allocate demand across pathways.
        
        Args:
            total_demand_kg: Total hydrogen demand for this timestep (kg)
            pathway_states: Current state of each pathway
            current_hour: Current simulation hour
            
        Returns:
            Dictionary mapping pathway_id to allocated demand (kg)
        """
        pass


class CostOptimalStrategy(BaseAllocationStrategy):
    """
    Minimize total production cost.
    
    Allocates demand to the pathway with lowest marginal cost,
    considering current energy prices and production efficiency.
    """
    
    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """Allocate demand to minimize cost."""
        
        # Calculate marginal cost for each pathway
        costs = {}
        for pathway_id, state in pathway_states.items():
            costs[pathway_id] = self._calculate_marginal_cost(state)
        
        # Sort pathways by cost (lowest first)
        sorted_pathways = sorted(costs.items(), key=lambda x: x[1])
        
        # Allocate demand starting with cheapest pathway
        allocation = {pathway_id: 0.0 for pathway_id in pathway_states.keys()}
        remaining_demand = total_demand_kg
        
        for pathway_id, cost in sorted_pathways:
            if remaining_demand <= 0:
                break
            
            # Get pathway capacity
            max_production = pathway_states[pathway_id].get('max_production_kg', 0.0)
            
            # Allocate up to capacity
            allocated = min(remaining_demand, max_production)
            allocation[pathway_id] = allocated
            remaining_demand -= allocated
        
        return allocation
    
    def _calculate_marginal_cost(self, pathway_state: Dict) -> float:
        """
        Calculate marginal cost of production ($/kg H2).
        
        Args:
            pathway_state: Current pathway state
            
        Returns:
            Marginal cost in $/kg
        """
        # Use historical cost if available
        if 'specific_cost_per_kg' in pathway_state and pathway_state['specific_cost_per_kg'] > 0:
            return pathway_state['specific_cost_per_kg']
        
        # Otherwise, use default estimates
        pathway_type = pathway_state.get('pathway_id', '')
        
        if 'electrolyzer' in pathway_type.lower():
            # Electrolyzer: ~$3-6/kg depending on electricity price
            return 4.50  # Default
        elif 'atr' in pathway_type.lower():
            # ATR: ~$1.5-2.5/kg (natural gas)
            return 2.00  # Default
        else:
            return 10.0  # Unknown pathway - penalize


class PriorityGridStrategy(BaseAllocationStrategy):
    """
    Maximize electrolyzer (grid-powered) usage.
    
    Prioritizes green hydrogen production from renewable electricity.
    Falls back to ATR only when electrolyzer cannot meet demand.
    """
    
    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """Allocate demand prioritizing electrolyzer."""
        
        allocation = {pathway_id: 0.0 for pathway_id in pathway_states.keys()}
        remaining_demand = total_demand_kg
        
        # Step 1: Allocate to electrolyzer first
        for pathway_id, state in pathway_states.items():
            if 'electrolyzer' in pathway_id.lower():
                max_production = state.get('max_production_kg', 0.0)
                allocated = min(remaining_demand, max_production)
                allocation[pathway_id] = allocated
                remaining_demand -= allocated
        
        # Step 2: Allocate remainder to ATR
        if remaining_demand > 0:
            for pathway_id, state in pathway_states.items():
                if 'atr' in pathway_id.lower():
                    max_production = state.get('max_production_kg', 0.0)
                    allocated = min(remaining_demand, max_production)
                    allocation[pathway_id] = allocated
                    remaining_demand -= allocated
        
        return allocation


class PriorityATRStrategy(BaseAllocationStrategy):
    """
    Maximize ATR usage.
    
    Prioritizes ATR for baseload production (more stable, lower cost).
    Uses electrolyzer for peak demand or when ATR insufficient.
    """
    
    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """Allocate demand prioritizing ATR."""
        
        allocation = {pathway_id: 0.0 for pathway_id in pathway_states.keys()}
        remaining_demand = total_demand_kg
        
        # Step 1: Allocate to ATR first
        for pathway_id, state in pathway_states.items():
            if 'atr' in pathway_id.lower():
                max_production = state.get('max_production_kg', 0.0)
                allocated = min(remaining_demand, max_production)
                allocation[pathway_id] = allocated
                remaining_demand -= allocated
        
        # Step 2: Allocate remainder to electrolyzer
        if remaining_demand > 0:
            for pathway_id, state in pathway_states.items():
                if 'electrolyzer' in pathway_id.lower():
                    max_production = state.get('max_production_kg', 0.0)
                    allocated = min(remaining_demand, max_production)
                    allocation[pathway_id] = allocated
                    remaining_demand -= allocated
        
        return allocation


class BalancedStrategy(BaseAllocationStrategy):
    """
    Balance demand equally across pathways.
    
    Splits demand 50/50 between available pathways,
    useful for testing or maintaining even utilization.
    """
    
    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """Allocate demand equally across pathways."""
        
        num_pathways = len(pathway_states)
        if num_pathways == 0:
            return {}
        
        target_per_pathway = total_demand_kg / num_pathways
        allocation = {}
        
        for pathway_id, state in pathway_states.items():
            max_production = state.get('max_production_kg', 0.0)
            allocated = min(target_per_pathway, max_production)
            allocation[pathway_id] = allocated
        
        return allocation


class EmissionsOptimalStrategy(BaseAllocationStrategy):
    """
    Minimize CO2 emissions.
    
    Prioritizes pathways with lowest emissions factor.
    For dual-path with electrolyzer + ATR, heavily favors electrolyzer.
    """
    
    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """Allocate demand to minimize emissions."""
        
        # Calculate emissions factor for each pathway
        emissions = {}
        for pathway_id, state in pathway_states.items():
            emissions[pathway_id] = self._get_emissions_factor(pathway_id, state)
        
        # Sort by emissions (lowest first)
        sorted_pathways = sorted(emissions.items(), key=lambda x: x[1])
        
        # Allocate demand starting with lowest emissions
        allocation = {pathway_id: 0.0 for pathway_id in pathway_states.keys()}
        remaining_demand = total_demand_kg
        
        for pathway_id, emissions_factor in sorted_pathways:
            if remaining_demand <= 0:
                break
            
            max_production = pathway_states[pathway_id].get('max_production_kg', 0.0)
            allocated = min(remaining_demand, max_production)
            allocation[pathway_id] = allocated
            remaining_demand -= allocated
        
        return allocation
    
    def _get_emissions_factor(self, pathway_id: str, pathway_state: Dict) -> float:
        """
        Get CO2 emissions factor (kg CO2 per kg H2).
        
        Args:
            pathway_id: Pathway identifier
            pathway_state: Current pathway state
            
        Returns:
            Emissions factor
        """
        # Check if state contains emissions info
        if 'emissions_factor' in pathway_state:
            return pathway_state['emissions_factor']
        
        # Otherwise, use defaults based on pathway type
        if 'electrolyzer' in pathway_id.lower():
            return 0.0  # Green hydrogen (assuming renewable grid)
        elif 'atr' in pathway_id.lower():
            return 10.5  # Grey hydrogen (~10.5 kg CO2 per kg H2)
        else:
            return 50.0  # Unknown - heavily penalize


# Strategy factory
ALLOCATION_STRATEGIES = {
    AllocationStrategy.COST_OPTIMAL: CostOptimalStrategy(),
    AllocationStrategy.PRIORITY_GRID: PriorityGridStrategy(),
    AllocationStrategy.PRIORITY_ATR: PriorityATRStrategy(),
    AllocationStrategy.BALANCED: BalancedStrategy(),
    AllocationStrategy.EMISSIONS_OPTIMAL: EmissionsOptimalStrategy()
}


def get_allocation_strategy(strategy: AllocationStrategy) -> BaseAllocationStrategy:
    """
    Get allocation strategy instance.
    
    Args:
        strategy: AllocationStrategy enum value
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy not implemented
    """
    if strategy not in ALLOCATION_STRATEGIES:
        raise ValueError(f"Unknown allocation strategy: {strategy}")
    
    return ALLOCATION_STRATEGIES[strategy]
