# STEP 3: Technical Specification - Component Standardization

---

# 03_Component_Standardization_Specification.md

**Document:** Component Standardization Layer Technical Specification  
**Project:** Dual-Path Hydrogen Production System - Modular Refactoring v2.0  
**Date:** November 18, 2025  
**Layer:** Layer 3 - Component Implementations  
**Priority:** HIGH  
**Dependencies:** Layer 1 (Core Foundation), Layer 2 (Performance Optimization)

---

## 1. Overview

### 1.1 Purpose

This specification defines the **standardized component implementations** that comprise the hydrogen production system. All components inherit from the `Component` ABC defined in Layer 1 and leverage the performance optimizations from Layer 2, ensuring uniform interfaces, predictable behavior, and high performance.

**Key Objectives:**
- Refactor existing components to inherit from `Component` base class
- Standardize lifecycle methods (`initialize()`, `step()`, `get_state()`)
- Integrate LUT Manager and Numba operations for performance
- Eliminate interface inconsistencies identified in critique
- Provide comprehensive component catalog with clear responsibilities

**Critique Remediation:**
- **PARTIAL → PASS:** "HydrogenProductionSource uses calculate_production() instead of step()" (Section 2)
- **PARTIAL → PASS:** "SourceTaggedTank doesn't inherit from Component ABC" (Section 3)
- **PARTIAL → PASS:** "Inconsistent component interfaces" (Sections 2-6)

***

### 1.2 Component Categories

The system consists of five component categories:

1. **Production Components:** Hydrogen generation (Electrolyzer, ATR)
2. **Storage Components:** Hydrogen containment (TankArray, SourceIsolatedTanks, OxygenBuffer)
3. **Compression Components:** Pressure elevation (FillingCompressor, OutgoingCompressor)
4. **Utility Components:** Supporting functions (DemandScheduler, HeatManager, EnergyPriceTracker)
5. **Legacy Adapters:** Backward compatibility wrappers (HPTanks adapter)

***

### 1.3 Scope

**In Scope:**
- Production: `ElectrolyzerProductionSource`, `ATRProductionSource`
- Storage: `TankArray`, `SourceIsolatedTanks`, `OxygenBuffer`
- Compression: `FillingCompressor`, `OutgoingCompressor`
- Utility: `DemandScheduler`, `EnergyPriceTracker`, `HeatManager`
- Legacy: Adapters for backward compatibility

**Out of Scope:**
- Pathway orchestration (covered in `05_Pathway_Integration_Specification.md`)
- Simulation engine (covered in `06_Simulation_Engine_Specification.md`)
- Configuration loading (covered in `04_Configuration_System_Specification.md`)

***

### 1.4 Design Principles

1. **Component ABC Compliance:** All components implement `initialize()`, `step()`, `get_state()`
2. **Single Responsibility:** Each component has one clear purpose
3. **Loose Coupling:** Components communicate via registry, not direct references
4. **Performance First:** Integrate LUT Manager and Numba operations where applicable
5. **State Transparency:** All internal state exposed via `get_state()` for checkpointing

***

## 2. Production Components

### 2.1 ElectrolyzerProductionSource

**Purpose:** Grid-powered water electrolysis hydrogen production with economic optimization.

**Design Rationale:**

**Problem (from critique):**
```python
# Old interface - inconsistent with Component ABC
h2_mass = electrolyzer.calculate_production(power_mw, dt)
```

**Solution:**
```python
# New interface - Component ABC compliant
electrolyzer.power_input_mw = power_mw  # Set input
electrolyzer.step(t)                     # Execute timestep
h2_mass = electrolyzer.h2_output_kg     # Read output
```

***

**File:** `h2_plant/components/production/electrolyzer_source.py`

```python
"""
Electrolyzer-based hydrogen production component.

Implements grid-powered water electrolysis with:
- Efficiency curves based on load factor
- Economic tracking (energy costs)
- Oxygen byproduct quantification
- Component ABC compliance
"""

import numpy as np
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import ProductionState
from h2_plant.core.constants import ProductionConstants, ConversionFactors


class ElectrolyzerProductionSource(Component):
    """
    Grid-powered electrolyzer hydrogen production.
    
    Converts electrical energy to hydrogen via water electrolysis:
    2H2O + electrical energy → 2H2 + O2
    
    Features:
    - Load-dependent efficiency (higher efficiency at higher loads)
    - Minimum stable load constraint (typically 20%)
    - Oxygen byproduct tracking
    - Energy cost tracking
    
    Example:
        electrolyzer = ElectrolyzerProductionSource(
            max_power_mw=2.5,
            base_efficiency=0.65,
            min_load_factor=0.20
        )
        
        # Initialize
        electrolyzer.initialize(dt=1.0, registry)
        
        # Set power input
        electrolyzer.power_input_mw = 2.0  # 80% load
        
        # Execute timestep
        electrolyzer.step(t)
        
        # Read outputs
        h2_kg = electrolyzer.h2_output_kg
        o2_kg = electrolyzer.o2_output_kg
    """
    
    def __init__(
        self,
        max_power_mw: float,
        base_efficiency: float = ProductionConstants.ELECTROLYSIS_TYPICAL_EFFICIENCY,
        min_load_factor: float = 0.20,
        startup_time_hours: float = 0.1
    ):
        """
        Initialize electrolyzer production source.
        
        Args:
            max_power_mw: Maximum electrical power input (MW)
            base_efficiency: Efficiency at rated load (0-1, LHV basis)
            min_load_factor: Minimum stable operating load (0-1)
            startup_time_hours: Time to reach full operation
        """
        super().__init__()
        
        # Configuration
        self.max_power_mw = max_power_mw
        self.base_efficiency = base_efficiency
        self.min_load_factor = min_load_factor
        self.startup_time_hours = startup_time_hours
        
        # Inputs (set before each step)
        self.power_input_mw = 0.0
        
        # Outputs (read after each step)
        self.h2_output_kg = 0.0
        self.o2_output_kg = 0.0
        self.actual_efficiency = 0.0
        
        # State
        self.state = ProductionState.OFFLINE
        self.cumulative_h2_kg = 0.0
        self.cumulative_energy_kwh = 0.0
        self.cumulative_cost = 0.0
        
        # Dependencies (populated during initialize)
        self._energy_price_tracker: Optional[Component] = None
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize electrolyzer component."""
        super().initialize(dt, registry)
        
        # Get energy price tracker if available
        if registry.has('energy_price_tracker'):
            self._energy_price_tracker = registry.get('energy_price_tracker')
        
        # Start in offline state
        self.state = ProductionState.OFFLINE
    
    def step(self, t: float) -> None:
        """
        Execute single timestep of electrolyzer operation.
        
        Args:
            t: Current simulation time (hours)
        """
        super().step(t)
        
        # Determine operating state
        load_factor = self.power_input_mw / self.max_power_mw
        
        if load_factor < self.min_load_factor:
            # Below minimum load - shut down
            self.state = ProductionState.OFFLINE
            self.h2_output_kg = 0.0
            self.o2_output_kg = 0.0
            self.actual_efficiency = 0.0
            return
        
        # Operating state
        self.state = ProductionState.RUNNING
        
        # Calculate efficiency based on load factor
        # Efficiency increases with load (typical electrolyzer behavior)
        self.actual_efficiency = self._calculate_efficiency(load_factor)
        
        # Calculate hydrogen production
        # Energy required: 33 kWh/kg at 100% efficiency (LHV)
        energy_input_kwh = self.power_input_mw * 1000 * self.dt
        theoretical_h2_kg = energy_input_kwh / ProductionConstants.H2_ENERGY_CONTENT_LHV_KWH_PER_KG
        self.h2_output_kg = theoretical_h2_kg * self.actual_efficiency
        
        # Calculate oxygen byproduct (stoichiometry: 8 kg O2 per kg H2)
        self.o2_output_kg = self.h2_output_kg * ProductionConstants.O2_TO_H2_MASS_RATIO
        
        # Update cumulative statistics
        self.cumulative_h2_kg += self.h2_output_kg
        self.cumulative_energy_kwh += energy_input_kwh
        
        # Track energy cost if price tracker available
        if self._energy_price_tracker is not None:
            energy_cost = energy_input_kwh * self._energy_price_tracker.current_price_per_kwh
            self.cumulative_cost += energy_cost
    
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            'power_input_mw': float(self.power_input_mw),
            'h2_output_kg': float(self.h2_output_kg),
            'o2_output_kg': float(self.o2_output_kg),
            'efficiency': float(self.actual_efficiency),
            'state': int(self.state),
            'load_factor': float(self.power_input_mw / self.max_power_mw),
            'cumulative_h2_kg': float(self.cumulative_h2_kg),
            'cumulative_energy_kwh': float(self.cumulative_energy_kwh),
            'cumulative_cost': float(self.cumulative_cost)
        }
    
    def _calculate_efficiency(self, load_factor: float) -> float:
        """
        Calculate efficiency based on load factor.
        
        Electrolyzer efficiency typically increases with load:
        - At min load (20%): ~90% of base efficiency
        - At rated load (100%): base efficiency
        - Non-linear relationship
        
        Args:
            load_factor: Current load / max load (0-1)
            
        Returns:
            Operating efficiency (0-1)
        """
        # Efficiency curve: η(L) = η_base * (0.9 + 0.1 * L)
        # Where L is load factor (0-1)
        efficiency = self.base_efficiency * (0.9 + 0.1 * load_factor)
        return min(efficiency, 1.0)  # Cap at 100%
    
    def get_specific_energy_consumption(self) -> float:
        """
        Calculate specific energy consumption (kWh/kg H2).
        
        Returns:
            Average energy consumption per kg H2 produced
        """
        if self.cumulative_h2_kg > 0:
            return self.cumulative_energy_kwh / self.cumulative_h2_kg
        return 0.0
    
    # Legacy adapter (deprecated)
    def calculate_production(self, power_mw: float, dt: float) -> float:
        """
        DEPRECATED: Legacy interface for backward compatibility.
        
        Use power_input + step() instead.
        """
        import warnings
        warnings.warn(
            "calculate_production() is deprecated. "
            "Use electrolyzer.power_input_mw = power; electrolyzer.step(t) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.power_input_mw = power_mw
        self.step(0.0)
        return self.h2_output_kg
```

***

### 2.2 ATRProductionSource

**Purpose:** Auto-Thermal Reforming (ATR) hydrogen production from natural gas with Numba optimization.

**File:** `h2_plant/components/production/atr_source.py`

```python
"""
ATR (Auto-Thermal Reforming) hydrogen production component.

Implements natural gas reforming with:
- Numba JIT-compiled reaction kinetics (from existing ATR_model.py)
- Startup/shutdown state management
- Waste heat recovery tracking
- CO2 emissions quantification
"""

import numpy as np
from typing import Dict, Any, Optional
from numba import njit

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import ProductionState
from h2_plant.core.constants import ProductionConstants, GasConstants


@njit
def calculate_atr_production(
    ng_flow_rate: float,
    temperature_k: float,
    pressure_pa: float,
    efficiency: float,
    dt: float
) -> tuple[float, float, float]:
    """
    Numba-compiled ATR reaction calculation.
    
    Simplified ATR reaction: CH4 + H2O → 3H2 + CO
    
    Args:
        ng_flow_rate: Natural gas flow rate (kg/h)
        temperature_k: Reactor temperature (K)
        pressure_pa: Reactor pressure (Pa)
        efficiency: Process efficiency (0-1)
        dt: Timestep (hours)
        
    Returns:
        Tuple of (h2_output_kg, co2_output_kg, waste_heat_kwh)
    """
    # Stoichiometry: 1 kg CH4 → 0.375 kg H2 (theoretical)
    theoretical_h2_yield = 0.375  # kg H2 per kg CH4
    
    ng_consumed = ng_flow_rate * dt
    h2_output = ng_consumed * theoretical_h2_yield * efficiency
    
    # CO2 emissions: 1 kg CH4 → 2.75 kg CO2
    co2_output = ng_consumed * 2.75
    
    # Waste heat (simplified): ~30% of NG energy content
    ng_energy_kwh = ng_consumed * 13.9  # kWh/kg NG (LHV)
    waste_heat = ng_energy_kwh * 0.30
    
    return h2_output, co2_output, waste_heat


class ATRProductionSource(Component):
    """
    Auto-Thermal Reforming hydrogen production from natural gas.
    
    Features:
    - Numba JIT-compiled reaction kinetics
    - Startup/shutdown state management
    - Waste heat recovery potential
    - CO2 emissions tracking
    
    Example:
        atr = ATRProductionSource(
            max_ng_flow_kg_h=100.0,
            efficiency=0.75
        )
        
        # Initialize
        atr.initialize(dt=1.0, registry)
        
        # Set natural gas flow
        atr.ng_flow_rate_kg_h = 80.0  # 80% capacity
        
        # Execute timestep
        atr.step(t)
        
        # Read outputs
        h2_kg = atr.h2_output_kg
        co2_kg = atr.co2_emissions_kg
    """
    
    def __init__(
        self,
        max_ng_flow_kg_h: float,
        efficiency: float = ProductionConstants.ATR_TYPICAL_EFFICIENCY,
        reactor_temperature_k: float = 1200.0,
        reactor_pressure_bar: float = 25.0,
        startup_time_hours: float = 1.0,
        cooldown_time_hours: float = 0.5
    ):
        """
        Initialize ATR production source.
        
        Args:
            max_ng_flow_kg_h: Maximum natural gas flow rate (kg/h)
            efficiency: Process efficiency (0-1)
            reactor_temperature_k: Operating temperature (K)
            reactor_pressure_bar: Operating pressure (bar)
            startup_time_hours: Time to reach operating conditions
            cooldown_time_hours: Time to safely shut down
        """
        super().__init__()
        
        # Configuration
        self.max_ng_flow_kg_h = max_ng_flow_kg_h
        self.efficiency = efficiency
        self.reactor_temperature_k = reactor_temperature_k
        self.reactor_pressure_pa = reactor_pressure_bar * 1e5
        self.startup_time_hours = startup_time_hours
        self.cooldown_time_hours = cooldown_time_hours
        
        # Inputs
        self.ng_flow_rate_kg_h = 0.0
        
        # Outputs
        self.h2_output_kg = 0.0
        self.co2_emissions_kg = 0.0
        self.waste_heat_kwh = 0.0
        
        # State
        self.state = ProductionState.OFFLINE
        self.startup_progress = 0.0  # 0-1, tracks startup completion
        self.cumulative_h2_kg = 0.0
        self.cumulative_co2_kg = 0.0
        self.cumulative_ng_kg = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize ATR component."""
        super().initialize(dt, registry)
        self.state = ProductionState.OFFLINE
    
    def step(self, t: float) -> None:
        """Execute single timestep of ATR operation."""
        super().step(t)
        
        # Handle state transitions
        if self.ng_flow_rate_kg_h > 0 and self.state == ProductionState.OFFLINE:
            self.state = ProductionState.STARTING
            self.startup_progress = 0.0
        
        if self.ng_flow_rate_kg_h == 0 and self.state == ProductionState.RUNNING:
            self.state = ProductionState.SHUTTING_DOWN
        
        # Process based on state
        if self.state == ProductionState.STARTING:
            # Startup phase
            self.startup_progress += self.dt / self.startup_time_hours
            
            if self.startup_progress >= 1.0:
                self.state = ProductionState.RUNNING
                self.startup_progress = 1.0
            
            # Reduced output during startup
            effective_flow = self.ng_flow_rate_kg_h * self.startup_progress
            self._calculate_production(effective_flow)
        
        elif self.state == ProductionState.RUNNING:
            # Normal operation
            self._calculate_production(self.ng_flow_rate_kg_h)
        
        elif self.state == ProductionState.SHUTTING_DOWN:
            # Cooldown phase
            self.h2_output_kg = 0.0
            self.co2_emissions_kg = 0.0
            self.waste_heat_kwh = 0.0
            self.state = ProductionState.OFFLINE
        
        else:  # OFFLINE
            self.h2_output_kg = 0.0
            self.co2_emissions_kg = 0.0
            self.waste_heat_kwh = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Return current component state."""
        return {
            **super().get_state(),
            'ng_flow_rate_kg_h': float(self.ng_flow_rate_kg_h),
            'h2_output_kg': float(self.h2_output_kg),
            'co2_emissions_kg': float(self.co2_emissions_kg),
            'waste_heat_kwh': float(self.waste_heat_kwh),
            'state': int(self.state),
            'startup_progress': float(self.startup_progress),
            'cumulative_h2_kg': float(self.cumulative_h2_kg),
            'cumulative_co2_kg': float(self.cumulative_co2_kg),
            'cumulative_ng_kg': float(self.cumulative_ng_kg)
        }
    
    def _calculate_production(self, effective_flow: float) -> None:
        """Calculate production using Numba-compiled function."""
        self.h2_output_kg, self.co2_emissions_kg, self.waste_heat_kwh = \
            calculate_atr_production(
                effective_flow,
                self.reactor_temperature_k,
                self.reactor_pressure_pa,
                self.efficiency,
                self.dt
            )
        
        # Update cumulative statistics
        self.cumulative_h2_kg += self.h2_output_kg
        self.cumulative_co2_kg += self.co2_emissions_kg
        self.cumulative_ng_kg += effective_flow * self.dt
```

***

## 3. Storage Components

### 3.1 SourceIsolatedTanks

**Purpose:** Physically separated storage for different production pathways (electrolyzer vs ATR) with source tagging.

**Design Rationale:**

The dual-path system requires **physical separation** between hydrogen from different sources:
- Electrolyzer path: "Green" hydrogen (renewable energy)
- ATR path: "Grey" hydrogen (natural gas)

This enables source-specific tracking for emissions accounting and compliance.

**File:** `h2_plant/components/storage/source_isolated_tanks.py`

```python
"""
Source-isolated tank storage system.

Maintains physical separation between hydrogen from different production
sources (e.g., electrolyzer vs ATR) for emissions tracking and compliance.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import TankState
from h2_plant.components.storage.tank_array import TankArray


@dataclass
class SourceTag:
    """Tag identifying hydrogen source."""
    source_id: str          # e.g., "electrolyzer_1", "atr_1"
    source_type: str        # e.g., "electrolyzer", "atr"
    emissions_factor: float # kg CO2 per kg H2


class SourceIsolatedTanks(Component):
    """
    Storage system maintaining physical separation by production source.
    
    Uses multiple TankArray instances, each dedicated to a specific source.
    Ensures no mixing between sources for emissions accounting.
    
    Example:
        storage = SourceIsolatedTanks(
            sources={
                'electrolyzer': SourceTag('elec_1', 'electrolyzer', 0.0),
                'atr': SourceTag('atr_1', 'atr', 10.5)
            },
            tanks_per_source=4,
            capacity_kg=200.0,
            pressure_bar=350
        )
        
        # Fill from electrolyzer
        storage.fill('electrolyzer', 150.0)
        
        # Discharge (prioritizes lowest emissions)
        mass, source = storage.discharge(100.0)
    """
    
    def __init__(
        self,
        sources: Dict[str, SourceTag],
        tanks_per_source: int,
        capacity_kg: float,
        pressure_bar: float
    ):
        """
        Initialize source-isolated tank system.
        
        Args:
            sources: Dictionary mapping source names to SourceTag metadata
            tanks_per_source: Number of tanks allocated to each source
            capacity_kg: Capacity of each tank (kg)
            pressure_bar: Operating pressure (bar)
        """
        super().__init__()
        
        self.sources = sources
        self.tanks_per_source = tanks_per_source
        self.capacity_kg = capacity_kg
        self.pressure_bar = pressure_bar
        
        # Create TankArray for each source
        self._tank_arrays: Dict[str, TankArray] = {}
        for source_name in sources.keys():
            self._tank_arrays[source_name] = TankArray(
                n_tanks=tanks_per_source,
                capacity_kg=capacity_kg,
                pressure_bar=pressure_bar
            )
        
        # Tracking
        self.fills_by_source: Dict[str, float] = {s: 0.0 for s in sources.keys()}
        self.discharges_by_source: Dict[str, float] = {s: 0.0 for s in sources.keys()}
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize all tank arrays."""
        super().initialize(dt, registry)
        
        for tank_array in self._tank_arrays.values():
            tank_array.initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep on all tank arrays."""
        super().step(t)
        
        for tank_array in self._tank_arrays.values():
            tank_array.step(t)
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        state = {**super().get_state()}
        
        # Add per-source tank states
        for source_name, tank_array in self._tank_arrays.items():
            state[f'{source_name}_mass_kg'] = tank_array.get_total_mass()
            state[f'{source_name}_capacity_kg'] = tank_array.get_available_capacity()
            state[f'{source_name}_fills_kg'] = self.fills_by_source[source_name]
            state[f'{source_name}_discharges_kg'] = self.discharges_by_source[source_name]
        
        # Add aggregate metrics
        state['total_mass_kg'] = self.get_total_mass()
        state['total_capacity_kg'] = self.get_total_capacity()
        
        return state
    
    def fill(self, source_name: str, mass_kg: float) -> tuple[float, float]:
        """
        Fill tanks from specific source.
        
        Args:
            source_name: Source identifier (must exist in sources dict)
            mass_kg: Mass to store (kg)
            
        Returns:
            Tuple of (mass_stored, mass_overflow)
            
        Raises:
            ValueError: If source_name not recognized
        """
        if source_name not in self._tank_arrays:
            raise ValueError(
                f"Unknown source '{source_name}'. Available: {list(self._tank_arrays.keys())}"
            )
        
        stored, overflow = self._tank_arrays[source_name].fill(mass_kg)
        self.fills_by_source[source_name] += stored
        
        return stored, overflow
    
    def discharge(
        self,
        mass_kg: float,
        priority_source: Optional[str] = None
    ) -> tuple[float, str]:
        """
        Discharge hydrogen, optionally prioritizing a specific source.
        
        Args:
            mass_kg: Mass to discharge (kg)
            priority_source: Source to discharge from first (None = lowest emissions)
            
        Returns:
            Tuple of (mass_discharged, source_name)
        """
        if priority_source is None:
            # Default: discharge from lowest emissions source first
            priority_source = self._get_lowest_emissions_source()
        
        # Try priority source first
        discharged = self._tank_arrays[priority_source].discharge(mass_kg)
        self.discharges_by_source[priority_source] += discharged
        
        # If insufficient, try other sources
        remaining = mass_kg - discharged
        if remaining > 0.01:
            for source_name in self._tank_arrays.keys():
                if source_name == priority_source:
                    continue
                
                additional = self._tank_arrays[source_name].discharge(remaining)
                self.discharges_by_source[source_name] += additional
                discharged += additional
                remaining -= additional
                
                if remaining < 0.01:
                    break
        
        return discharged, priority_source
    
    def get_total_mass(self) -> float:
        """Return total mass across all sources (kg)."""
        return sum(ta.get_total_mass() for ta in self._tank_arrays.values())
    
    def get_total_capacity(self) -> float:
        """Return total available capacity across all sources (kg)."""
        return sum(ta.get_available_capacity() for ta in self._tank_arrays.values())
    
    def get_mass_by_source(self, source_name: str) -> float:
        """Return mass stored from specific source (kg)."""
        return self._tank_arrays[source_name].get_total_mass()
    
    def _get_lowest_emissions_source(self) -> str:
        """Return source with lowest emissions factor."""
        return min(self.sources.items(), key=lambda x: x[1].emissions_factor)[0]
    
    def get_weighted_emissions_factor(self) -> float:
        """
        Calculate mass-weighted average emissions factor for stored hydrogen.
        
        Returns:
            Weighted average kg CO2 per kg H2
        """
        total_mass = 0.0
        weighted_emissions = 0.0
        
        for source_name, source_tag in self.sources.items():
            mass = self._tank_arrays[source_name].get_total_mass()
            total_mass += mass
            weighted_emissions += mass * source_tag.emissions_factor
        
        if total_mass > 0:
            return weighted_emissions / total_mass
        return 0.0
```

***

### 3.2 OxygenBuffer

**Purpose:** Byproduct oxygen storage from electrolysis with overflow management.

**File:** `h2_plant/components/storage/oxygen_buffer.py`

```python
"""
Oxygen buffer storage for electrolyzer byproduct.

Simple buffer with overflow venting when capacity exceeded.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class OxygenBuffer(Component):
    """
    Buffer storage for oxygen byproduct from electrolysis.
    
    Features:
    - Simple mass balance tracking
    - Overflow venting (no storage limit violation)
    - Usage tracking (if oxygen monetized)
    
    Example:
        o2_buffer = OxygenBuffer(capacity_kg=500.0)
        
        # Add oxygen from electrolyzer
        o2_buffer.add_oxygen(63.5)  # 8 kg O2 per kg H2
        
        # Remove for industrial use
        o2_buffer.remove_oxygen(50.0)
    """
    
    def __init__(self, capacity_kg: float):
        """
        Initialize oxygen buffer.
        
        Args:
            capacity_kg: Maximum oxygen storage capacity (kg)
        """
        super().__init__()
        
        self.capacity_kg = capacity_kg
        self.mass_kg = 0.0
        
        # Tracking
        self.cumulative_added_kg = 0.0
        self.cumulative_removed_kg = 0.0
        self.cumulative_vented_kg = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize buffer."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep."""
        super().step(t)
        # No per-timestep logic needed
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'mass_kg': float(self.mass_kg),
            'capacity_kg': float(self.capacity_kg),
            'fill_percentage': float(self.mass_kg / self.capacity_kg * 100),
            'cumulative_added_kg': float(self.cumulative_added_kg),
            'cumulative_removed_kg': float(self.cumulative_removed_kg),
            'cumulative_vented_kg': float(self.cumulative_vented_kg)
        }
    
    def add_oxygen(self, mass_kg: float) -> float:
        """
        Add oxygen to buffer.
        
        Args:
            mass_kg: Mass to add (kg)
            
        Returns:
            Mass vented due to overflow (kg)
        """
        available_capacity = self.capacity_kg - self.mass_kg
        
        if mass_kg <= available_capacity:
            self.mass_kg += mass_kg
            self.cumulative_added_kg += mass_kg
            return 0.0
        else:
            # Partial storage + venting
            stored = available_capacity
            vented = mass_kg - stored
            
            self.mass_kg = self.capacity_kg
            self.cumulative_added_kg += stored
            self.cumulative_vented_kg += vented
            
            return vented
    
    def remove_oxygen(self, mass_kg: float) -> float:
        """
        Remove oxygen from buffer (for sale or industrial use).
        
        Args:
            mass_kg: Mass to remove (kg)
            
        Returns:
            Actual mass removed (may be less if insufficient stored)
        """
        removed = min(mass_kg, self.mass_kg)
        self.mass_kg -= removed
        self.cumulative_removed_kg += removed
        
        return removed
```

***

## 4. Compression Components

### 4.1 FillingCompressor

**Purpose:** Multi-stage compression for LP → HP tank transfer with energy tracking.

**File:** `h2_plant/components/compression/filling_compressor.py`

```python
"""
Filling compressor for LP to HP storage transfer.

Implements multi-stage compression with:
- Energy consumption calculation using Numba
- Inter-stage cooling
- Efficiency degradation at part-load
"""

import numpy as np
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import CompressorMode
from h2_plant.core.constants import CompressionConstants, GasConstants
from h2_plant.optimization.numba_ops import calculate_compression_work


class FillingCompressor(Component):
    """
    Multi-stage compressor for LP→HP storage transfer.
    
    Compresses hydrogen from low-pressure storage (~30 bar) to
    high-pressure storage (~350 bar) using multi-stage compression
    with inter-stage cooling.
    
    Example:
        compressor = FillingCompressor(
            max_flow_kg_h=100.0,
            inlet_pressure_bar=30,
            outlet_pressure_bar=350,
            num_stages=3
        )
        
        # Transfer mass
        compressor.transfer_mass_kg = 50.0
        compressor.step(t)
        
        # Read energy consumption
        energy_kwh = compressor.energy_consumed_kwh
    """
    
    def __init__(
        self,
        max_flow_kg_h: float,
        inlet_pressure_bar: float = 30.0,
        outlet_pressure_bar: float = 350.0,
        num_stages: int = 3,
        efficiency: float = CompressionConstants.ISENTROPIC_EFFICIENCY
    ):
        """
        Initialize filling compressor.
        
        Args:
            max_flow_kg_h: Maximum flow rate (kg/h)
            inlet_pressure_bar: Inlet pressure (bar)
            outlet_pressure_bar: Outlet pressure (bar)
            num_stages: Number of compression stages
            efficiency: Isentropic efficiency (0-1)
        """
        super().__init__()
        
        self.max_flow_kg_h = max_flow_kg_h
        self.inlet_pressure_pa = inlet_pressure_bar * 1e5
        self.outlet_pressure_pa = outlet_pressure_bar * 1e5
        self.num_stages = num_stages
        self.efficiency = efficiency
        
        # Calculate inter-stage pressures
        pressure_ratio_total = outlet_pressure_bar / inlet_pressure_bar
        self.stage_pressure_ratio = pressure_ratio_total ** (1.0 / num_stages)
        
        # Inputs
        self.transfer_mass_kg = 0.0
        
        # Outputs
        self.energy_consumed_kwh = 0.0
        self.actual_mass_transferred_kg = 0.0
        
        # State
        self.mode = CompressorMode.IDLE
        self.cumulative_energy_kwh = 0.0
        self.cumulative_mass_kg = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize compressor."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep."""
        super().step(t)
        
        if self.transfer_mass_kg > 0:
            self.mode = CompressorMode.LP_TO_HP
            
            # Clamp to max flow rate
            max_transfer = self.max_flow_kg_h * self.dt
            self.actual_mass_transferred_kg = min(self.transfer_mass_kg, max_transfer)
            
            # Calculate energy consumption using Numba
            self.energy_consumed_kwh = self._calculate_energy()
            
            # Update cumulative statistics
            self.cumulative_energy_kwh += self.energy_consumed_kwh
            self.cumulative_mass_kg += self.actual_mass_transferred_kg
        else:
            self.mode = CompressorMode.IDLE
            self.actual_mass_transferred_kg = 0.0
            self.energy_consumed_kwh = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'transfer_mass_kg': float(self.transfer_mass_kg),
            'actual_mass_transferred_kg': float(self.actual_mass_transferred_kg),
            'energy_consumed_kwh': float(self.energy_consumed_kwh),
            'mode': int(self.mode),
            'cumulative_energy_kwh': float(self.cumulative_energy_kwh),
            'cumulative_mass_kg': float(self.cumulative_mass_kg),
            'specific_energy_kwh_per_kg': float(
                self.cumulative_energy_kwh / self.cumulative_mass_kg
                if self.cumulative_mass_kg > 0 else 0.0
            )
        }
    
    def _calculate_energy(self) -> float:
        """
        Calculate compression energy using Numba-optimized function.
        
        Multi-stage compression with inter-stage cooling to minimize work.
        
        Returns:
            Energy consumption in kWh
        """
        total_work_j = 0.0
        current_pressure = self.inlet_pressure_pa
        
        # Calculate work for each stage
        for stage in range(self.num_stages):
            next_pressure = current_pressure * self.stage_pressure_ratio
            
            stage_work_j = calculate_compression_work(
                p1=current_pressure,
                p2=next_pressure,
                mass=self.actual_mass_transferred_kg,
                temperature=298.15,  # Assume inter-stage cooling to ambient
                efficiency=self.efficiency,
                gamma=GasConstants.GAMMA_H2,
                gas_constant=GasConstants.R_H2
            )
            
            total_work_j += stage_work_j
            current_pressure = next_pressure
        
        # Convert J to kWh
        return total_work_j / 3.6e6
```

***

### 4.2 OutgoingCompressor

**Purpose:** Boost HP storage to delivery pressure (900 bar) for customer dispensing.

**File:** `h2_plant/components/compression/outgoing_compressor.py`

```python
"""
Outgoing compressor for delivery pressure boosting.

Boosts hydrogen from HP storage (~350 bar) to delivery pressure (~900 bar)
for customer dispensing or transport.
"""

from typing import Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import CompressorMode
from h2_plant.core.constants import CompressionConstants, StorageConstants
from h2_plant.optimization.numba_ops import calculate_compression_work


class OutgoingCompressor(Component):
    """
    Compressor for boosting to delivery pressure.
    
    Single-stage or two-stage compression from HP storage to
    delivery pressure for customer dispensing.
    
    Example:
        compressor = OutgoingCompressor(
            max_flow_kg_h=200.0,
            inlet_pressure_bar=350,
            outlet_pressure_bar=900
        )
    """
    
    def __init__(
        self,
        max_flow_kg_h: float,
        inlet_pressure_bar: float = 350.0,
        outlet_pressure_bar: float = 900.0,
        efficiency: float = CompressionConstants.ISENTROPIC_EFFICIENCY
    ):
        """Initialize outgoing compressor."""
        super().__init__()
        
        self.max_flow_kg_h = max_flow_kg_h
        self.inlet_pressure_pa = inlet_pressure_bar * 1e5
        self.outlet_pressure_pa = outlet_pressure_bar * 1e5
        self.efficiency = efficiency
        
        # Inputs
        self.delivery_mass_kg = 0.0
        
        # Outputs
        self.energy_consumed_kwh = 0.0
        self.actual_mass_delivered_kg = 0.0
        
        # State
        self.mode = CompressorMode.IDLE
        self.cumulative_energy_kwh = 0.0
        self.cumulative_mass_kg = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize compressor."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep."""
        super().step(t)
        
        if self.delivery_mass_kg > 0:
            self.mode = CompressorMode.HP_TO_DELIVERY
            
            # Clamp to max flow rate
            max_delivery = self.max_flow_kg_h * self.dt
            self.actual_mass_delivered_kg = min(self.delivery_mass_kg, max_delivery)
            
            # Calculate energy
            work_j = calculate_compression_work(
                p1=self.inlet_pressure_pa,
                p2=self.outlet_pressure_pa,
                mass=self.actual_mass_delivered_kg,
                temperature=298.15,
                efficiency=self.efficiency
            )
            
            self.energy_consumed_kwh = work_j / 3.6e6
            
            # Update cumulative
            self.cumulative_energy_kwh += self.energy_consumed_kwh
            self.cumulative_mass_kg += self.actual_mass_delivered_kg
        else:
            self.mode = CompressorMode.IDLE
            self.actual_mass_delivered_kg = 0.0
            self.energy_consumed_kwh = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'delivery_mass_kg': float(self.delivery_mass_kg),
            'actual_mass_delivered_kg': float(self.actual_mass_delivered_kg),
            'energy_consumed_kwh': float(self.energy_consumed_kwh),
            'mode': int(self.mode),
            'cumulative_energy_kwh': float(self.cumulative_energy_kwh),
            'cumulative_mass_kg': float(self.cumulative_mass_kg)
        }
```

***

## 5. Utility Components

### 5.1 DemandScheduler

**Purpose:** Time-based hydrogen demand profile with configurable patterns.

**File:** `h2_plant/components/utility/demand_scheduler.py`

```python
"""
Hydrogen demand scheduler with time-based profiles.

Supports:
- Constant demand
- Time-of-day patterns (day/night shifts)
- Weekly patterns
- Custom profiles from arrays
"""

import numpy as np
from typing import Dict, Any, Optional, Literal

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


DemandPattern = Literal['constant', 'day_night', 'weekly', 'custom']


class DemandScheduler(Component):
    """
    Time-based hydrogen demand scheduler.
    
    Example:
        # Day/night pattern
        scheduler = DemandScheduler(
            pattern='day_night',
            day_demand_kg_h=80.0,
            night_demand_kg_h=20.0,
            day_start_hour=6,
            night_start_hour=22
        )
        
        scheduler.step(t=14)  # 2 PM
        demand = scheduler.current_demand_kg_h  # 80.0 (daytime)
    """
    
    def __init__(
        self,
        pattern: DemandPattern = 'constant',
        base_demand_kg_h: float = 50.0,
        day_demand_kg_h: Optional[float] = None,
        night_demand_kg_h: Optional[float] = None,
        day_start_hour: int = 6,
        night_start_hour: int = 22,
        custom_profile: Optional[np.ndarray] = None
    ):
        """
        Initialize demand scheduler.
        
        Args:
            pattern: Demand pattern type
            base_demand_kg_h: Baseline demand for 'constant' pattern
            day_demand_kg_h: Daytime demand for 'day_night' pattern
            night_demand_kg_h: Nighttime demand for 'day_night' pattern
            day_start_hour: Hour when day shift starts (0-23)
            night_start_hour: Hour when night shift starts (0-23)
            custom_profile: 8760-element array for 'custom' pattern
        """
        super().__init__()
        
        self.pattern = pattern
        self.base_demand_kg_h = base_demand_kg_h
        self.day_demand_kg_h = day_demand_kg_h or base_demand_kg_h
        self.night_demand_kg_h = night_demand_kg_h or base_demand_kg_h * 0.3
        self.day_start_hour = day_start_hour
        self.night_start_hour = night_start_hour
        self.custom_profile = custom_profile
        
        # Output
        self.current_demand_kg_h = 0.0
        self.current_demand_kg = 0.0  # For this timestep (demand * dt)
        
        # Tracking
        self.cumulative_demand_kg = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize scheduler."""
        super().initialize(dt, registry)
        
        if self.pattern == 'custom' and self.custom_profile is None:
            raise ValueError("custom_profile required for 'custom' pattern")
    
    def step(self, t: float) -> None:
        """Update demand based on current time."""
        super().step(t)
        
        if self.pattern == 'constant':
            self.current_demand_kg_h = self.base_demand_kg_h
        
        elif self.pattern == 'day_night':
            hour_of_day = int(t) % 24
            
            if self.day_start_hour <= hour_of_day < self.night_start_hour:
                self.current_demand_kg_h = self.day_demand_kg_h
            else:
                self.current_demand_kg_h = self.night_demand_kg_h
        
        elif self.pattern == 'custom':
            hour_index = int(t) % len(self.custom_profile)
            self.current_demand_kg_h = self.custom_profile[hour_index]
        
        # Calculate demand for this timestep
        self.current_demand_kg = self.current_demand_kg_h * self.dt
        self.cumulative_demand_kg += self.current_demand_kg
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'pattern': self.pattern,
            'current_demand_kg_h': float(self.current_demand_kg_h),
            'current_demand_kg': float(self.current_demand_kg),
            'cumulative_demand_kg': float(self.cumulative_demand_kg)
        }
```

***

### 5.2 EnergyPriceTracker

**Purpose:** Time-based energy pricing with real-time cost tracking.

**File:** `h2_plant/components/utility/energy_price_tracker.py`

```python
"""
Energy price tracker for economic optimization.

Tracks time-varying energy prices (e.g., day-ahead market, real-time pricing)
and provides current price for production cost calculations.
"""

import numpy as np
from typing import Dict, Any, Optional

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class EnergyPriceTracker(Component):
    """
    Time-based energy price tracker.
    
    Example:
        # Load hourly prices
        prices_mwh = np.array([...])  # 8760 hourly prices
        
        tracker = EnergyPriceTracker(prices_per_mwh=prices_mwh)
        tracker.step(t=15)
        
        price = tracker.current_price_per_kwh  # $/kWh at hour 15
    """
    
    def __init__(
        self,
        prices_per_mwh: np.ndarray,
        default_price_per_mwh: float = 60.0
    ):
        """
        Initialize energy price tracker.
        
        Args:
            prices_per_mwh: Array of hourly prices ($/MWh)
            default_price_per_mwh: Default price if array exhausted
        """
        super().__init__()
        
        self.prices_per_mwh = prices_per_mwh
        self.default_price_per_mwh = default_price_per_mwh
        
        # Outputs
        self.current_price_per_mwh = default_price_per_mwh
        self.current_price_per_kwh = default_price_per_mwh / 1000.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize tracker."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Update current price based on time."""
        super().step(t)
        
        hour_index = int(t)
        
        if hour_index < len(self.prices_per_mwh):
            self.current_price_per_mwh = float(self.prices_per_mwh[hour_index])
        else:
            self.current_price_per_mwh = self.default_price_per_mwh
        
        self.current_price_per_kwh = self.current_price_per_mwh / 1000.0
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'current_price_per_mwh': float(self.current_price_per_mwh),
            'current_price_per_kwh': float(self.current_price_per_kwh)
        }
```

***

## 6. Testing Strategy

### 6.1 Component Unit Tests

**File:** `tests/components/test_electrolyzer_source.py`

```python
import pytest
import numpy as np
from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import ProductionState


def test_electrolyzer_initialization():
    """Test electrolyzer initializes correctly."""
    elec = ElectrolyzerProductionSource(max_power_mw=2.5, base_efficiency=0.65)
    registry = ComponentRegistry()
    
    elec.initialize(dt=1.0, registry=registry)
    
    assert elec._initialized
    assert elec.state == ProductionState.OFFLINE


def test_electrolyzer_production_calculation():
    """Test hydrogen production calculation."""
    elec = ElectrolyzerProductionSource(max_power_mw=2.5, base_efficiency=0.65)
    registry = ComponentRegistry()
    elec.initialize(dt=1.0, registry=registry)
    
    # Set 2 MW input (80% load)
    elec.power_input_mw = 2.0
    elec.step(0.0)
    
    # Check production
    assert elec.h2_output_kg > 0
    assert elec.state == ProductionState.RUNNING
    
    # Check oxygen byproduct
    expected_o2 = elec.h2_output_kg * 7.94
    assert abs(elec.o2_output_kg - expected_o2) < 0.01


def test_electrolyzer_below_min_load():
    """Test electrolyzer shuts down below minimum load."""
    elec = ElectrolyzerProductionSource(
        max_power_mw=2.5,
        min_load_factor=0.20
    )
    registry = ComponentRegistry()
    elec.initialize(dt=1.0, registry=registry)
    
    # Set below min load (10% = 0.25 MW < 20% min)
    elec.power_input_mw = 0.25
    elec.step(0.0)
    
    assert elec.state == ProductionState.OFFLINE
    assert elec.h2_output_kg == 0.0


def test_electrolyzer_state_serialization():
    """Test get_state returns complete state."""
    elec = ElectrolyzerProductionSource(max_power_mw=2.5)
    registry = ComponentRegistry()
    elec.initialize(dt=1.0, registry=registry)
    
    elec.power_input_mw = 2.0
    elec.step(0.0)
    
    state = elec.get_state()
    
    assert 'h2_output_kg' in state
    assert 'efficiency' in state
    assert 'state' in state
    assert 'cumulative_h2_kg' in state
```

***

### 6.2 Integration Tests

**File:** `tests/integration/test_production_to_storage.py`

```python
def test_electrolyzer_to_tank_array():
    """Test electrolyzer filling tank array."""
    from h2_plant.components.production.electrolyzer_source import ElectrolyzerProductionSource
    from h2_plant.components.storage.tank_array import TankArray
    from h2_plant.core.component_registry import ComponentRegistry
    
    # Setup
    registry = ComponentRegistry()
    
    electrolyzer = ElectrolyzerProductionSource(max_power_mw=2.5)
    tanks = TankArray(n_tanks=4, capacity_kg=200.0, pressure_bar=350)
    
    registry.register('electrolyzer', electrolyzer, component_type='production')
    registry.register('tanks', tanks, component_type='storage')
    
    registry.initialize_all(dt=1.0)
    
    # Simulate 10 hours of production
    for hour in range(10):
        # Produce hydrogen
        electrolyzer.power_input_mw = 2.0
        electrolyzer.step(hour)
        
        # Store in tanks
        h2_produced = electrolyzer.h2_output_kg
        stored, overflow = tanks.fill(h2_produced)
        
        # Verify no overflow
        assert overflow == 0.0
        
        # Step tanks
        tanks.step(hour)
    
    # Verify mass balance
    total_produced = electrolyzer.cumulative_h2_kg
    total_stored = tanks.get_total_mass()
    
    assert abs(total_produced - total_stored) < 0.01  # Mass conserved
```

***

## 7. Validation Criteria

This Component Standardization Layer is **COMPLETE** when:

 **Production Components:**
- `ElectrolyzerProductionSource` and `ATRProductionSource` implemented
- Both inherit from `Component` ABC
- Legacy `calculate_production()` deprecated with warnings
- Numba integration working (ATR)
- Unit tests achieve 95%+ coverage

 **Storage Components:**
- `TankArray` uses NumPy vectorization
- `SourceIsolatedTanks` maintains physical separation
- `OxygenBuffer` handles overflow correctly
- All inherit from `Component` ABC

 **Compression Components:**
- `FillingCompressor` and `OutgoingCompressor` use Numba energy calculations
- Multi-stage compression implemented
- Energy tracking accurate

 **Utility Components:**
- `DemandScheduler` supports multiple patterns
- `EnergyPriceTracker` provides time-varying prices

 **Integration:**
- All components work together in test scenarios
- Registry manages all components
- State serialization complete

***

## 8. Success Metrics

| **Metric** | **Target** | **Validation** |
|-----------|-----------|----------------|
| Component ABC Compliance | 100% | All components inherit from `Component` |
| Interface Standardization | 100% | All use `initialize/step/get_state` |
| Test Coverage | 95%+ | `pytest --cov=h2_plant.components` |
| Performance Integration | 100% | LUT/Numba used where applicable |
| Legacy Compatibility | 100% | Deprecated methods work with warnings |

***
