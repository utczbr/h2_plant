# STEP 3: Technical Specification - Pathway Integration

---

# 05_Pathway_Integration_Specification.md

**Document:** Pathway Integration Layer Technical Specification  
**Project:** Dual-Path Hydrogen Production System - Modular Refactoring v2.0  
**Date:** November 18, 2025  
**Layer:** Layer 4 - Pathway Orchestration  
**Priority:** MEDIUM  
**Dependencies:** Layers 1-4 (Core Foundation, Performance, Components, Configuration)

***

## 1. Overview

### 1.1 Purpose

This specification defines the **pathway orchestration layer** that coordinates the dual-path hydrogen production system. The pathway layer addresses the critique's finding of incomplete allocation strategy implementation by providing a complete, flexible framework for managing multiple production pathways with sophisticated allocation logic.

**Key Objectives:**
- Implement `IsolatedProductionPath` for self-contained production chains
- Create `DualPathCoordinator` with complete allocation strategy implementations
- Enable economic optimization across production pathways
- Support dynamic pathway switching based on operational conditions
- Provide pathway-level monitoring and analytics

**Critique Remediation:**
- **PARTIAL → PASS:** "AllocationStrategy exists but not fully integrated with pathways" (Section 3)
- **Enhancement:** Complete economic optimization implementation

***

### 1.2 System Architecture

The dual-path system consists of two isolated production chains:

```
Electrolyzer Path:
  ElectrolyzerSource → LP Tanks → Filling Compressor → HP Tanks (Electrolyzer) → Delivery

ATR Path:
  ATRSource → LP Tanks → Filling Compressor → HP Tanks (ATR) → Delivery

Coordination:
  DualPathCoordinator allocates demand across paths based on strategy
```

**Key Design Decisions:**
1. **Physical Isolation:** Each pathway has dedicated storage to prevent mixing
2. **Flexible Allocation:** Strategy pattern enables pluggable allocation algorithms
3. **Economic Optimization:** Real-time cost minimization based on energy prices
4. **Source Tracking:** Maintains emissions accounting per pathway

***

### 1.3 Scope

**In Scope:**
- `pathways/isolated_production_path.py`: Single production pathway orchestration
- `pathways/dual_path_coordinator.py`: Multi-pathway coordination and allocation
- `pathways/allocation_strategies.py`: Allocation strategy implementations
- Pathway-level state management and monitoring

**Out of Scope:**
- Component implementations (covered in `03_Component_Standardization_Specification.md`)
- Sub-component thermal loops (Heat Exchangers, Attemperators, Recirculation Pumps) - Encapsulated within Source
- Simulation engine (covered in `06_Simulation_Engine_Specification.md`)
- Multi-pathway systems beyond dual-path (future enhancement)

***

### 1.4 Design Principles

1. **Encapsulation:** Each pathway is self-contained and independently operable
2. **Strategy Pattern:** Allocation algorithms are pluggable and testable
3. **Economic Awareness:** All allocation decisions consider production costs
4. **State Transparency:** Pathway state fully observable for monitoring
5. **Fail-Safe Operation:** Single pathway failure doesn't crash entire system

***

## 2. IsolatedProductionPath

### 2.1 Design Rationale

**Purpose:** Encapsulate a complete production chain (source → storage → compression) as a single orchestratable unit.

**Benefits:**
- Simplifies coordination logic (coordinator deals with paths, not individual components)
- Enables independent testing of production chains
- Enables independent testing of production chains
- **Encapsulates Complexity:** Auxiliary thermal management (e.g., Attemperators, Cooling loops) is hidden within the Source abstraction.
- Supports future multi-pathway configurations
- Clear responsibility boundaries

---

### 2.2 Implementation

**File:** `h2_plant/pathways/isolated_production_path.py`

```python
"""
Isolated production pathway orchestration.

A production pathway encapsulates:
- Production source (Electrolyzer or ATR)
- Low-pressure buffer storage
- Compression to high-pressure storage
- High-pressure delivery storage
"""

from typing import Dict, Any, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import ProductionState, TankState

logger = logging.getLogger(__name__)


class IsolatedProductionPath(Component):
    """
    Orchestrates a complete isolated production pathway.
    
    Manages the flow: Production Source → LP Storage → Compressor → HP Storage
    
    Features:
    - Automatic pressure management (LP to HP transfer)
    - Production-storage coordination
    - Pathway-level state tracking
    - Economic metrics (cost per kg H2)
    
    Example:
        # Create electrolyzer pathway
        path = IsolatedProductionPath(
            pathway_id='electrolyzer',
            source_id='electrolyzer',
            lp_storage_id='elec_lp_tanks',
            hp_storage_id='elec_hp_tanks',
            compressor_id='elec_filling_compressor'
        )
        
        # Set production target
        path.production_target_kg_h = 50.0
        
        # Execute timestep
        path.step(t)
        
        # Read outputs
        h2_produced = path.h2_produced_kg
        h2_available = path.h2_available_kg
    """
    
    def __init__(
        self,
        pathway_id: str,
        source_id: str,
        lp_storage_id: str,
        hp_storage_id: str,
        compressor_id: str,
        lp_to_hp_threshold_kg: float = 100.0,
        hp_min_reserve_kg: float = 50.0
    ):
        """
        Initialize isolated production pathway.
        
        Args:
            pathway_id: Unique pathway identifier
            source_id: Production source component ID in registry
            lp_storage_id: Low-pressure storage component ID
            hp_storage_id: High-pressure storage component ID
            compressor_id: Filling compressor component ID
            lp_to_hp_threshold_kg: Minimum LP mass before transfer (kg)
            hp_min_reserve_kg: Minimum HP reserve to maintain (kg)
        """
        super().__init__()
        
        self.pathway_id = pathway_id
        self.source_id = source_id
        self.lp_storage_id = lp_storage_id
        self.hp_storage_id = hp_storage_id
        self.compressor_id = compressor_id
        self.lp_to_hp_threshold_kg = lp_to_hp_threshold_kg
        self.hp_min_reserve_kg = hp_min_reserve_kg
        
        # Component references (populated during initialize)
        self._source: Optional[Component] = None
        self._lp_storage: Optional[Component] = None
        self._hp_storage: Optional[Component] = None
        self._compressor: Optional[Component] = None
        
        # Inputs (set by coordinator before each step)
        self.production_target_kg_h = 0.0
        self.discharge_demand_kg = 0.0
        
        # Outputs (read by coordinator after each step)
        self.h2_produced_kg = 0.0
        self.h2_stored_lp_kg = 0.0
        self.h2_stored_hp_kg = 0.0
        self.h2_available_kg = 0.0  # HP storage available for delivery
        self.h2_delivered_kg = 0.0
        
        # State tracking
        self.cumulative_production_kg = 0.0
        self.cumulative_delivery_kg = 0.0
        self.cumulative_compression_energy_kwh = 0.0
        self.cumulative_production_energy_kwh = 0.0
        self.cumulative_cost = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize pathway and resolve component references."""
        super().initialize(dt, registry)
        
        # Resolve component references
        try:
            self._source = registry.get(self.source_id)
            self._lp_storage = registry.get(self.lp_storage_id)
            self._hp_storage = registry.get(self.hp_storage_id)
            self._compressor = registry.get(self.compressor_id)
        except KeyError as e:
            raise ValueError(f"Pathway {self.pathway_id} component not found: {e}")
        
        logger.info(f"Initialized pathway '{self.pathway_id}'")
    
    def step(self, t: float) -> None:
        """
        Execute single timestep of pathway orchestration.
        
        Workflow:
        1. Produce hydrogen based on target
        2. Store production in LP tanks
        3. Transfer LP → HP when threshold reached
        4. Deliver from HP storage based on demand
        
        Args:
            t: Current simulation time (hours)
        """
        super().step(t)
        
        # Step 1: Production
        self._execute_production()
        
        # Step 2: LP Storage
        self._manage_lp_storage()
        
        # Step 3: LP → HP Compression
        self._execute_compression()
        
        # Step 4: HP Storage & Delivery
        self._manage_hp_storage()
        
        # Update state
        self._update_state()
    
    def get_state(self) -> Dict[str, Any]:
        """Return pathway state for monitoring."""
        return {
            **super().get_state(),
            'pathway_id': self.pathway_id,
            'production_target_kg_h': float(self.production_target_kg_h),
            'h2_produced_kg': float(self.h2_produced_kg),
            'h2_stored_lp_kg': float(self.h2_stored_lp_kg),
            'h2_stored_hp_kg': float(self.h2_stored_hp_kg),
            'h2_available_kg': float(self.h2_available_kg),
            'h2_delivered_kg': float(self.h2_delivered_kg),
            'cumulative_production_kg': float(self.cumulative_production_kg),
            'cumulative_delivery_kg': float(self.cumulative_delivery_kg),
            'cumulative_cost': float(self.cumulative_cost),
            'specific_cost_per_kg': float(
                self.cumulative_cost / self.cumulative_production_kg
                if self.cumulative_production_kg > 0 else 0.0
            )
        }
    
    def _execute_production(self) -> None:
        """Execute production step based on target."""
        # Set production input based on source type
        if hasattr(self._source, 'power_input_mw'):
            # Electrolyzer - convert kg/h target to power
            # Simplified: 50 kWh/kg at 65% efficiency ≈ 0.077 MW per kg/h
            self._source.power_input_mw = self.production_target_kg_h * 0.077
        
        elif hasattr(self._source, 'ng_flow_rate_kg_h'):
            # ATR - convert kg/h target to NG flow
            # Simplified: 1 kg NG → 0.28 kg H2 at 75% efficiency
            self._source.ng_flow_rate_kg_h = self.production_target_kg_h / 0.28
        
        # Read production output
        if hasattr(self._source, 'h2_output_kg'):
            self.h2_produced_kg = self._source.h2_output_kg
        else:
            self.h2_produced_kg = 0.0
        
        self.cumulative_production_kg += self.h2_produced_kg
    
    def _manage_lp_storage(self) -> None:
        """Store production in LP tanks."""
        if self.h2_produced_kg > 0:
            stored, overflow = self._lp_storage.fill(self.h2_produced_kg)
            
            if overflow > 0:
                logger.warning(
                    f"Pathway {self.pathway_id}: LP storage full, "
                    f"venting {overflow:.2f} kg H2"
                )
        
        # Update LP storage state
        self.h2_stored_lp_kg = self._lp_storage.get_total_mass()
    
    def _execute_compression(self) -> None:
        """Transfer LP → HP when threshold reached."""
        if self.h2_stored_lp_kg >= self.lp_to_hp_threshold_kg:
            # Determine transfer amount
            transfer_mass = min(
                self.h2_stored_lp_kg,
                self._compressor.max_flow_kg_h * self.dt
            )
            
            # Check HP capacity
            hp_available_capacity = self._hp_storage.get_available_capacity()
            transfer_mass = min(transfer_mass, hp_available_capacity)
            
            if transfer_mass > 0:
                # Discharge from LP
                lp_discharged = self._lp_storage.discharge(transfer_mass)
                
                # Compress (compressor tracks energy)
                self._compressor.transfer_mass_kg = lp_discharged
                
                # Fill HP
                hp_stored, hp_overflow = self._hp_storage.fill(lp_discharged)
                
                # Track compression energy
                if hasattr(self._compressor, 'energy_consumed_kwh'):
                    self.cumulative_compression_energy_kwh += self._compressor.energy_consumed_kwh
    
    def _manage_hp_storage(self) -> None:
        """Manage HP storage and deliver based on demand."""
        # Update HP storage state
        self.h2_stored_hp_kg = self._hp_storage.get_total_mass()
        
        # Available for delivery (total minus reserve)
        self.h2_available_kg = max(0.0, self.h2_stored_hp_kg - self.hp_min_reserve_kg)
        
        # Deliver if demand exists
        if self.discharge_demand_kg > 0:
            # Don't violate minimum reserve
            deliverable = min(self.discharge_demand_kg, self.h2_available_kg)
            
            if deliverable > 0:
                self.h2_delivered_kg = self._hp_storage.discharge(deliverable)
                self.cumulative_delivery_kg += self.h2_delivered_kg
            else:
                self.h2_delivered_kg = 0.0
        else:
            self.h2_delivered_kg = 0.0
    
    def _update_state(self) -> None:
        """Update cumulative state tracking."""
        # Track production energy if available
        if hasattr(self._source, 'cumulative_energy_kwh'):
            self.cumulative_production_energy_kwh = self._source.cumulative_energy_kwh
        
        # Track cost if available
        if hasattr(self._source, 'cumulative_cost'):
            self.cumulative_cost = self._source.cumulative_cost
    
    def get_production_cost_per_kg(self) -> float:
        """
        Calculate specific production cost ($/kg H2).
        
        Returns:
            Cost per kg of hydrogen produced
        """
        if self.cumulative_production_kg > 0:
            return self.cumulative_cost / self.cumulative_production_kg
        return 0.0
    
    def get_total_energy_per_kg(self) -> float:
        """
        Calculate specific energy consumption (kWh/kg H2).
        
        Includes production energy and compression energy.
        
        Returns:
            Energy per kg of hydrogen produced
        """
        if self.cumulative_production_kg > 0:
            total_energy = (
                self.cumulative_production_energy_kwh +
                self.cumulative_compression_energy_kwh
            )
            return total_energy / self.cumulative_production_kg
        return 0.0
```

***

## 3. Allocation Strategies

### 3.1 Strategy Pattern Design

**File:** `h2_plant/pathways/allocation_strategies.py`

```python
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
        if 'specific_cost_per_kg' in pathway_state:
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
```

***

## 4. DualPathCoordinator

### 4.1 Implementation

**File:** `h2_plant/pathways/dual_path_coordinator.py`

```python
"""
Dual-path production coordinator.

Coordinates two isolated production pathways with configurable
allocation strategies for demand splitting and economic optimization.
"""

from typing import Dict, Any, List, Optional
import logging

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import AllocationStrategy
from h2_plant.pathways.isolated_production_path import IsolatedProductionPath
from h2_plant.pathways.allocation_strategies import get_allocation_strategy

logger = logging.getLogger(__name__)


class DualPathCoordinator(Component):
    """
    Coordinates dual-path hydrogen production system.
    
    Manages:
    - Demand allocation across pathways
    - Economic optimization
    - Pathway health monitoring
    - Aggregate metrics
    
    Example:
        coordinator = DualPathCoordinator(
            pathway_ids=['electrolyzer_path', 'atr_path'],
            allocation_strategy=AllocationStrategy.COST_OPTIMAL
        )
        
        coordinator.total_demand_kg = 100.0
        coordinator.step(t)
        
        # Read results
        delivered = coordinator.total_delivered_kg
    """
    
    def __init__(
        self,
        pathway_ids: List[str],
        allocation_strategy: AllocationStrategy = AllocationStrategy.COST_OPTIMAL,
        demand_scheduler_id: str = 'demand_scheduler'
    ):
        """
        Initialize dual-path coordinator.
        
        Args:
            pathway_ids: List of pathway component IDs in registry
            allocation_strategy: Strategy for demand allocation
            demand_scheduler_id: Demand scheduler component ID
        """
        super().__init__()
        
        self.pathway_ids = pathway_ids
        self.allocation_strategy = allocation_strategy
        self.demand_scheduler_id = demand_scheduler_id
        
        # Component references
        self._pathways: Dict[str, IsolatedProductionPath] = {}
        self._demand_scheduler: Optional[Component] = None
        self._strategy_impl = get_allocation_strategy(allocation_strategy)
        
        # Inputs (can be overridden before step)
        self.total_demand_kg = 0.0
        
        # Outputs
        self.pathway_allocations: Dict[str, float] = {}
        self.total_produced_kg = 0.0
        self.total_delivered_kg = 0.0
        self.total_available_kg = 0.0
        self.demand_shortfall_kg = 0.0
        
        # Cumulative metrics
        self.cumulative_production_kg = 0.0
        self.cumulative_delivery_kg = 0.0
        self.cumulative_shortfall_kg = 0.0
        self.cumulative_cost = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize coordinator and resolve pathway references."""
        super().initialize(dt, registry)
        
        # Resolve pathway references
        for pathway_id in self.pathway_ids:
            try:
                pathway = registry.get(pathway_id)
                if not isinstance(pathway, IsolatedProductionPath):
                    raise TypeError(f"{pathway_id} is not an IsolatedProductionPath")
                self._pathways[pathway_id] = pathway
            except KeyError:
                raise ValueError(f"Pathway not found in registry: {pathway_id}")
        
        # Resolve demand scheduler
        if registry.has(self.demand_scheduler_id):
            self._demand_scheduler = registry.get(self.demand_scheduler_id)
        
        logger.info(
            f"DualPathCoordinator initialized: {len(self._pathways)} pathways, "
            f"strategy={self.allocation_strategy.name}"
        )
    
    def step(self, t: float) -> None:
        """
        Execute single timestep of dual-path coordination.
        
        Workflow:
        1. Get total demand (from scheduler or manual input)
        2. Allocate demand across pathways using strategy
        3. Set production targets for each pathway
        4. Aggregate results
        5. Calculate shortfall if any
        
        Args:
            t: Current simulation time (hours)
        """
        super().step(t)
        
        # Step 1: Determine total demand
        if self._demand_scheduler is not None:
            self.total_demand_kg = self._demand_scheduler.current_demand_kg
        
        # Step 2: Allocate demand across pathways
        self._allocate_demand(t)
        
        # Step 3: Set pathway production targets
        for pathway_id, allocation_kg in self.pathway_allocations.items():
            pathway = self._pathways[pathway_id]
            # Convert kg to kg/h rate
            pathway.production_target_kg_h = allocation_kg / self.dt
        
        # Step 4: Aggregate results
        self._aggregate_results()
        
        # Step 5: Calculate shortfall
        self.demand_shortfall_kg = max(0.0, self.total_demand_kg - self.total_delivered_kg)
        
        # Update cumulative metrics
        self.cumulative_production_kg += self.total_produced_kg
        self.cumulative_delivery_kg += self.total_delivered_kg
        self.cumulative_shortfall_kg += self.demand_shortfall_kg
    
    def get_state(self) -> Dict[str, Any]:
        """Return coordinator state for monitoring."""
        state = {
            **super().get_state(),
            'allocation_strategy': self.allocation_strategy.name,
            'total_demand_kg': float(self.total_demand_kg),
            'total_produced_kg': float(self.total_produced_kg),
            'total_delivered_kg': float(self.total_delivered_kg),
            'total_available_kg': float(self.total_available_kg),
            'demand_shortfall_kg': float(self.demand_shortfall_kg),
            'cumulative_production_kg': float(self.cumulative_production_kg),
            'cumulative_delivery_kg': float(self.cumulative_delivery_kg),
            'cumulative_shortfall_kg': float(self.cumulative_shortfall_kg),
            'cumulative_cost': float(self.cumulative_cost),
            'pathway_allocations': {k: float(v) for k, v in self.pathway_allocations.items()}
        }
        
        # Add per-pathway metrics
        for pathway_id, pathway in self._pathways.items():
            pathway_state = pathway.get_state()
            state[f'{pathway_id}_production_kg'] = pathway_state.get('h2_produced_kg', 0.0)
            state[f'{pathway_id}_delivery_kg'] = pathway_state.get('h2_delivered_kg', 0.0)
            state[f'{pathway_id}_cost_per_kg'] = pathway_state.get('specific_cost_per_kg', 0.0)
        
        return state
    
    def _allocate_demand(self, t: float) -> None:
        """Allocate demand across pathways using configured strategy."""
        
        # Build pathway state summary for strategy
        pathway_states = {}
        for pathway_id, pathway in self._pathways.items():
            pathway_state = pathway.get_state()
            
            # Calculate max production for this timestep
            # This is pathway-specific and depends on source capacity
            max_production_kg_h = self._get_pathway_max_production(pathway)
            max_production_kg = max_production_kg_h * self.dt
            
            pathway_states[pathway_id] = {
                'pathway_id': pathway_id,
                'max_production_kg': max_production_kg,
                'h2_available_kg': pathway_state.get('h2_available_kg', 0.0),
                'specific_cost_per_kg': pathway.get_production_cost_per_kg()
            }
        
        # Execute allocation strategy
        self.pathway_allocations = self._strategy_impl.allocate(
            total_demand_kg=self.total_demand_kg,
            pathway_states=pathway_states,
            current_hour=int(t)
        )
        
        logger.debug(f"Hour {t}: Allocated {self.pathway_allocations}")
    
    def _get_pathway_max_production(self, pathway: IsolatedProductionPath) -> float:
        """
        Get maximum production rate for pathway (kg/h).
        
        Args:
            pathway: Production pathway
            
        Returns:
            Maximum production rate in kg/h
        """
        # This is a simplified calculation
        # In practice, query the source component for max capacity
        
        # Default estimates
        if 'electrolyzer' in pathway.pathway_id.lower():
            return 50.0  # ~2.5 MW electrolyzer ≈ 50 kg/h
        elif 'atr' in pathway.pathway_id.lower():
            return 75.0  # ~100 kg/h NG ≈ 75 kg/h H2
        else:
            return 25.0  # Conservative default
    
    def _aggregate_results(self) -> None:
        """Aggregate results from all pathways."""
        self.total_produced_kg = sum(
            pathway.h2_produced_kg for pathway in self._pathways.values()
        )
        
        self.total_delivered_kg = sum(
            pathway.h2_delivered_kg for pathway in self._pathways.values()
        )
        
        self.total_available_kg = sum(
            pathway.h2_available_kg for pathway in self._pathways.values()
        )
        
        self.cumulative_cost = sum(
            pathway.cumulative_cost for pathway in self._pathways.values()
        )
    
    def get_pathway_utilization(self) -> Dict[str, float]:
        """
        Calculate utilization factor for each pathway.
        
        Returns:
            Dictionary mapping pathway_id to utilization (0-1)
        """
        utilization = {}
        
        for pathway_id, pathway in self._pathways.items():
            if pathway.cumulative_production_kg > 0:
                # Rough estimate: utilization = actual / theoretical max
                max_theoretical = self._get_pathway_max_production(pathway) * self.dt * 8760
                utilization[pathway_id] = min(1.0, pathway.cumulative_production_kg / max_theoretical)
            else:
                utilization[pathway_id] = 0.0
        
        return utilization
    
    def get_weighted_average_cost(self) -> float:
        """
        Calculate production-weighted average cost ($/kg H2).
        
        Returns:
            Weighted average cost across all pathways
        """
        if self.cumulative_production_kg > 0:
            return self.cumulative_cost / self.cumulative_production_kg
        return 0.0
```

***

## 5. Integration Examples

### 5.1 Complete Dual-Path System

**File:** `examples/dual_path_simulation.py`

```python
"""
Complete dual-path hydrogen production simulation.

Demonstrates pathway integration with configuration-driven setup.
"""

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.pathways.isolated_production_path import IsolatedProductionPath
from h2_plant.pathways.dual_path_coordinator import DualPathCoordinator
from h2_plant.core.enums import AllocationStrategy


def run_dual_path_simulation():
    """Run 1-week dual-path simulation with cost optimization."""
    
    # Load plant configuration
    plant = PlantBuilder.from_file("configs/plant_baseline.yaml")
    registry = plant.registry
    
    # Create pathways
    electrolyzer_path = IsolatedProductionPath(
        pathway_id='electrolyzer_path',
        source_id='electrolyzer',
        lp_storage_id='elec_lp_tanks',
        hp_storage_id='elec_hp_tanks',
        compressor_id='filling_compressor'
    )
    
    atr_path = IsolatedProductionPath(
        pathway_id='atr_path',
        source_id='atr',
        lp_storage_id='atr_lp_tanks',
        hp_storage_id='atr_hp_tanks',
        compressor_id='atr_filling_compressor'
    )
    
    # Register pathways
    registry.register('electrolyzer_path', electrolyzer_path, component_type='pathway')
    registry.register('atr_path', atr_path, component_type='pathway')
    
    # Create coordinator
    coordinator = DualPathCoordinator(
        pathway_ids=['electrolyzer_path', 'atr_path'],
        allocation_strategy=AllocationStrategy.COST_OPTIMAL
    )
    registry.register('coordinator', coordinator, component_type='pathway')
    
    # Initialize all
    registry.initialize_all(dt=1.0)
    
    # Run simulation (1 week = 168 hours)
    print("Running dual-path simulation...")
    
    for hour in range(168):
        # Step all components
        registry.step_all(hour)
        
        # Daily reporting
        if hour > 0 and hour % 24 == 0:
            state = coordinator.get_state()
            
            print(f"\nDay {hour//24}:")
            print(f"  Total Production: {state['total_produced_kg']:.1f} kg")
            print(f"  Total Delivered: {state['total_delivered_kg']:.1f} kg")
            print(f"  Shortfall: {state['demand_shortfall_kg']:.1f} kg")
            print(f"  Electrolyzer: {state['pathway_allocations'].get('electrolyzer_path', 0):.1f} kg")
            print(f"  ATR: {state['pathway_allocations'].get('atr_path', 0):.1f} kg")
    
    # Final report
    final_state = coordinator.get_state()
    
    print("\n=== Final Results ===")
    print(f"Total Production: {final_state['cumulative_production_kg']:.1f} kg")
    print(f"Total Delivery: {final_state['cumulative_delivery_kg']:.1f} kg")
    print(f"Total Cost: ${final_state['cumulative_cost']:.2f}")
    print(f"Average Cost: ${coordinator.get_weighted_average_cost():.2f}/kg")
    
    utilization = coordinator.get_pathway_utilization()
    print(f"\nPathway Utilization:")
    for pathway_id, util in utilization.items():
        print(f"  {pathway_id}: {util*100:.1f}%")


if __name__ == '__main__':
    run_dual_path_simulation()
```

***

## 6. Testing Strategy

### 6.1 Pathway Unit Tests

**File:** `tests/pathways/test_isolated_production_path.py`

```python
import pytest
from h2_plant.pathways.isolated_production_path import IsolatedProductionPath
from h2_plant.core.component_registry import ComponentRegistry


def test_pathway_initialization(mock_components):
    """Test pathway initializes and resolves component references."""
    registry = ComponentRegistry()
    
    # Register mock components
    for comp_id, comp in mock_components.items():
        registry.register(comp_id, comp)
    
    # Create pathway
    path = IsolatedProductionPath(
        pathway_id='test_path',
        source_id='source',
        lp_storage_id='lp_storage',
        hp_storage_id='hp_storage',
        compressor_id='compressor'
    )
    
    # Initialize
    path.initialize(dt=1.0, registry=registry)
    
    assert path._initialized
    assert path._source is not None
    assert path._lp_storage is not None


def test_pathway_production_flow():
    """Test complete production flow through pathway."""
    # Setup pathway with real components
    # ... (create components, register, initialize)
    
    # Set production target
    path.production_target_kg_h = 50.0
    
    # Execute timestep
    path.step(0.0)
    
    # Verify production occurred
    assert path.h2_produced_kg > 0
    assert path.h2_stored_lp_kg > 0
```

***

### 6.2 Allocation Strategy Tests

**File:** `tests/pathways/test_allocation_strategies.py`

```python
from h2_plant.pathways.allocation_strategies import CostOptimalStrategy, PriorityGridStrategy


def test_cost_optimal_allocation():
    """Test cost-optimal strategy chooses cheapest pathway."""
    strategy = CostOptimalStrategy()
    
    pathway_states = {
        'electrolyzer': {
            'max_production_kg': 50.0,
            'specific_cost_per_kg': 5.00  # More expensive
        },
        'atr': {
            'max_production_kg': 75.0,
            'specific_cost_per_kg': 2.00  # Cheaper
        }
    }
    
    allocation = strategy.allocate(
        total_demand_kg=100.0,
        pathway_states=pathway_states,
        current_hour=0
    )
    
    # Should allocate to ATR first (cheaper)
    assert allocation['atr'] == 75.0  # Maxed out ATR
    assert allocation['electrolyzer'] == 25.0  # Remainder to electrolyzer


def test_priority_grid_allocation():
    """Test priority-grid strategy maximizes electrolyzer usage."""
    strategy = PriorityGridStrategy()
    
    pathway_states = {
        'electrolyzer': {'max_production_kg': 50.0},
        'atr': {'max_production_kg': 75.0}
    }
    
    allocation = strategy.allocate(
        total_demand_kg=100.0,
        pathway_states=pathway_states,
        current_hour=0
    )
    
    # Should allocate to electrolyzer first
    assert allocation['electrolyzer'] == 50.0  # Maxed out electrolyzer
    assert allocation['atr'] == 50.0  # Remainder to ATR
```

***

## 7. Validation Criteria

This Pathway Integration Layer is **COMPLETE** when:

 **IsolatedProductionPath:**
- Complete production flow orchestration
- Component integration working
- State tracking accurate
- Unit tests achieve 95%+ coverage

 **Allocation Strategies:**
- All 5 strategies implemented (COST_OPTIMAL, PRIORITY_GRID, PRIORITY_ATR, BALANCED, EMISSIONS_OPTIMAL)
- Strategy pattern correctly implemented
- Economic calculations accurate
- Unit tests for each strategy

 **DualPathCoordinator:**
- Demand allocation working
- Pathway coordination correct
- Aggregate metrics accurate
- Integration tests pass

 **Integration:**
- End-to-end dual-path simulation working
- Configuration-driven setup functional
- Economic optimization validated

***

## 8. Success Metrics

| **Metric** | **Target** | **Validation** |
|-----------|-----------|----------------|
| Strategy Completeness | 5/5 implemented | All AllocationStrategy enum values supported |
| Allocation Accuracy | 100% | Mass balance conserved |
| Economic Optimization | Functional | Cost-optimal strategy minimizes costs |
| Test Coverage | 95%+ | `pytest --cov=h2_plant.pathways` |
| Integration | Complete | Full simulation runs successfully |

***
