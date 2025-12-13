"""
Allocation Strategies for Dual-Path Hydrogen Production.

This module implements demand allocation algorithms for splitting hydrogen
demand across multiple production pathways (e.g., electrolyzer + ATR).

Available Strategies:
    - **COST_OPTIMAL**: Minimize total production cost.
    - **PRIORITY_GRID**: Maximize electrolyzer (green H₂) usage.
    - **PRIORITY_ATR**: Maximize ATR (baseload) usage.
    - **BALANCED**: Equal utilization across pathways.
    - **EMISSIONS_OPTIMAL**: Minimize CO₂ emissions.

Strategy Selection:
    Use `get_allocation_strategy(AllocationStrategy.XXX)` to obtain
    an instance of the desired strategy.

Allocation Logic:
    Each strategy implements `allocate()` which returns a dictionary
    mapping pathway_id to allocated demand (kg). The sum of allocations
    may be less than total_demand_kg if pathways have insufficient capacity.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np

from h2_plant.core.enums import AllocationStrategy


class BaseAllocationStrategy(ABC):
    """
    Abstract base class for allocation strategies.

    Subclasses must implement `allocate()` to define the specific
    demand distribution logic.
    """

    @abstractmethod
    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """
        Allocate demand across available pathways.

        Args:
            total_demand_kg (float): Total hydrogen demand for timestep (kg).
            pathway_states (Dict[str, Dict]): Current state of each pathway,
                containing 'max_production_kg', 'specific_cost_per_kg', etc.
            current_hour (int): Current simulation hour.

        Returns:
            Dict[str, float]: Dictionary mapping pathway_id to allocated demand (kg).
        """
        pass


class CostOptimalStrategy(BaseAllocationStrategy):
    """
    Minimize total production cost.

    Allocates demand to the pathway with lowest marginal cost,
    considering current energy prices and production efficiency.
    Uses merit-order dispatch: cheapest first.

    Typical Costs:
        - Electrolyzer: $3-6/kg (depends on electricity price).
        - ATR: $1.5-2.5/kg (natural gas-based).
    """

    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """
        Allocate demand to minimize cost using merit-order dispatch.

        Args:
            total_demand_kg (float): Total demand (kg).
            pathway_states (Dict[str, Dict]): Pathway states with cost info.
            current_hour (int): Current hour.

        Returns:
            Dict[str, float]: Allocation per pathway.
        """
        # Calculate marginal cost for each pathway
        costs = {}
        for pathway_id, state in pathway_states.items():
            costs[pathway_id] = self._calculate_marginal_cost(state)

        # Sort by cost (lowest first)
        sorted_pathways = sorted(costs.items(), key=lambda x: x[1])

        # Allocate starting with cheapest
        allocation = {pathway_id: 0.0 for pathway_id in pathway_states.keys()}
        remaining_demand = total_demand_kg

        for pathway_id, cost in sorted_pathways:
            if remaining_demand <= 0:
                break

            max_production = pathway_states[pathway_id].get('max_production_kg', 0.0)
            allocated = min(remaining_demand, max_production)
            allocation[pathway_id] = allocated
            remaining_demand -= allocated

        return allocation

    def _calculate_marginal_cost(self, pathway_state: Dict) -> float:
        """
        Calculate marginal cost of production.

        Uses historical cost if available, otherwise applies
        technology-specific defaults.

        Args:
            pathway_state (Dict): Current pathway state.

        Returns:
            float: Marginal cost in $/kg H₂.
        """
        if 'specific_cost_per_kg' in pathway_state and pathway_state['specific_cost_per_kg'] > 0:
            return pathway_state['specific_cost_per_kg']

        pathway_type = pathway_state.get('pathway_id', '')

        if 'electrolyzer' in pathway_type.lower():
            return 4.50
        elif 'atr' in pathway_type.lower():
            return 2.00
        else:
            return 10.0


class PriorityGridStrategy(BaseAllocationStrategy):
    """
    Maximize electrolyzer (grid-powered) usage.

    Prioritizes green hydrogen production from renewable electricity.
    Falls back to ATR only when electrolyzer cannot meet demand.

    Use Case:
        Maximize renewable H₂ fraction for green certification.
    """

    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """
        Allocate demand prioritizing electrolyzer.

        Args:
            total_demand_kg (float): Total demand (kg).
            pathway_states (Dict[str, Dict]): Pathway states.
            current_hour (int): Current hour.

        Returns:
            Dict[str, float]: Allocation per pathway.
        """
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

    Prioritizes ATR for baseload production (more stable operation,
    lower marginal cost). Uses electrolyzer for peak demand or
    when ATR capacity is insufficient.

    Use Case:
        Minimize operating cost when green certification not required.
    """

    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """
        Allocate demand prioritizing ATR.

        Args:
            total_demand_kg (float): Total demand (kg).
            pathway_states (Dict[str, Dict]): Pathway states.
            current_hour (int): Current hour.

        Returns:
            Dict[str, float]: Allocation per pathway.
        """
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

    Splits demand 50/50 (or equally among N pathways), useful for
    testing or maintaining even utilization of all production assets.

    Use Case:
        Ensure all pathways remain operational for flexibility.
    """

    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """
        Allocate demand equally across pathways.

        Args:
            total_demand_kg (float): Total demand (kg).
            pathway_states (Dict[str, Dict]): Pathway states.
            current_hour (int): Current hour.

        Returns:
            Dict[str, float]: Equal allocation per pathway.
        """
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
    Minimize CO₂ emissions.

    Prioritizes pathways with lowest emissions factor.
    For dual-path with electrolyzer + ATR, heavily favors electrolyzer.

    Emissions Factors:
        - Electrolyzer (green grid): ~0 kg CO₂/kg H₂.
        - ATR (grey hydrogen): ~10.5 kg CO₂/kg H₂.

    Use Case:
        Minimize carbon footprint for ESG reporting.
    """

    def allocate(
        self,
        total_demand_kg: float,
        pathway_states: Dict[str, Dict],
        current_hour: int
    ) -> Dict[str, float]:
        """
        Allocate demand to minimize emissions.

        Args:
            total_demand_kg (float): Total demand (kg).
            pathway_states (Dict[str, Dict]): Pathway states.
            current_hour (int): Current hour.

        Returns:
            Dict[str, float]: Allocation per pathway.
        """
        # Calculate emissions factor for each pathway
        emissions = {}
        for pathway_id, state in pathway_states.items():
            emissions[pathway_id] = self._get_emissions_factor(pathway_id, state)

        # Sort by emissions (lowest first)
        sorted_pathways = sorted(emissions.items(), key=lambda x: x[1])

        # Allocate starting with lowest emissions
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
        Get CO₂ emissions factor.

        Args:
            pathway_id (str): Pathway identifier.
            pathway_state (Dict): Current pathway state.

        Returns:
            float: Emissions factor (kg CO₂/kg H₂).
        """
        if 'emissions_factor' in pathway_state:
            return pathway_state['emissions_factor']

        if 'electrolyzer' in pathway_id.lower():
            return 0.0
        elif 'atr' in pathway_id.lower():
            return 10.5
        else:
            return 50.0


# =============================================================================
# STRATEGY FACTORY
# =============================================================================

ALLOCATION_STRATEGIES = {
    AllocationStrategy.COST_OPTIMAL: CostOptimalStrategy(),
    AllocationStrategy.PRIORITY_GRID: PriorityGridStrategy(),
    AllocationStrategy.PRIORITY_ATR: PriorityATRStrategy(),
    AllocationStrategy.BALANCED: BalancedStrategy(),
    AllocationStrategy.EMISSIONS_OPTIMAL: EmissionsOptimalStrategy()
}


def get_allocation_strategy(strategy: AllocationStrategy) -> BaseAllocationStrategy:
    """
    Get allocation strategy instance by enum.

    Args:
        strategy (AllocationStrategy): Strategy enum value.

    Returns:
        BaseAllocationStrategy: Strategy instance.

    Raises:
        ValueError: If strategy not implemented.

    Example:
        >>> strategy = get_allocation_strategy(AllocationStrategy.COST_OPTIMAL)
        >>> allocation = strategy.allocate(100.0, pathway_states, current_hour)
    """
    if strategy not in ALLOCATION_STRATEGIES:
        raise ValueError(f"Unknown allocation strategy: {strategy}")

    return ALLOCATION_STRATEGIES[strategy]
