# Extension Analysis: External Inputs, Mixer, and Battery Components

## Executive Summary

**Required Work:**
1.  Create 3 new component types (External Inputs, Mixer, Battery)
2.  Add optional component configuration support
3.  Define component interfaces and thermodynamic skeletons
4.  Minor change to configuration schema

**Architecture Compatibility: 100%**   
**Implementation Effort: 2-3 days per component**

***

## 1. Component Analysis

### 1.1 External Input Components

#### **A. External Oxygen Source**

**Purpose:** Provide supplemental oxygen from external suppliers (e.g., purchased O₂, industrial surplus).

**Component Specification:**

```python
"""
External oxygen source component.

Provides oxygen input from external suppliers, configurable by flow rate
or pressure-driven delivery.
"""

from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class ExternalOxygenSource(Component):
    """
    External oxygen supply source.
    
    Can operate in two modes:
    1. Fixed Flow Rate: Constant O₂ delivery (kg/h)
    2. Pressure-Driven: Delivers O₂ to maintain target pressure
    
    Example Configuration:
        oxygen_source:
          mode: "fixed_flow"
          flow_rate_kg_h: 50.0
          pressure_bar: 5.0
          cost_per_kg: 0.15
    
    Integration Points:
        - Connects to: OxygenBuffer or OxygenMixer
        - Provides: O₂ mass flow output
    """
    
    def __init__(
        self,
        mode: str = "fixed_flow",  # "fixed_flow" or "pressure_driven"
        flow_rate_kg_h: float = 0.0,
        pressure_bar: float = 5.0,
        cost_per_kg: float = 0.15,
        max_capacity_kg_h: float = 100.0
    ):
        super().__init__()
        
        self.mode = mode
        self.flow_rate_kg_h = flow_rate_kg_h
        self.pressure_bar = pressure_bar
        self.cost_per_kg = cost_per_kg
        self.max_capacity_kg_h = max_capacity_kg_h
        
        # State
        self.o2_output_kg = 0.0
        self.cumulative_o2_kg = 0.0
        self.cumulative_cost = 0.0
        
        # Connections (populated during initialize)
        self._target_component: Optional[Component] = None
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize external oxygen source."""
        super().initialize(dt, registry)
        
        # Optional: Connect to target component
        if registry.has('oxygen_mixer'):
            self._target_component = registry.get('oxygen_mixer')
        elif registry.has('oxygen_buffer'):
            self._target_component = registry.get('oxygen_buffer')
    
    def step(self, t: float) -> None:
        """Execute timestep - deliver oxygen."""
        super().step(t)
        
        if self.mode == "fixed_flow":
            # Fixed flow rate mode
            self.o2_output_kg = self.flow_rate_kg_h * self.dt
        
        elif self.mode == "pressure_driven":
            # Pressure-driven mode (simplified - needs thermodynamic model)
            if self._target_component:
                # TODO: Calculate required flow based on target pressure
                # For now, use placeholder
                required_flow = self._estimate_required_flow()
                self.o2_output_kg = min(required_flow, self.max_capacity_kg_h * self.dt)
            else:
                self.o2_output_kg = 0.0
        
        # Update cumulative metrics
        self.cumulative_o2_kg += self.o2_output_kg
        self.cumulative_cost += self.o2_output_kg * self.cost_per_kg
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'mode': self.mode,
            'flow_rate_kg_h': float(self.flow_rate_kg_h),
            'o2_output_kg': float(self.o2_output_kg),
            'pressure_bar': float(self.pressure_bar),
            'cumulative_o2_kg': float(self.cumulative_o2_kg),
            'cumulative_cost': float(self.cumulative_cost),
            
            # Flow metadata for dashboards
            'flows': {
                'outputs': {
                    'oxygen': {
                        'value': float(self.o2_output_kg),
                        'unit': 'kg',
                        'destination': 'oxygen_mixer' if self._target_component else 'oxygen_buffer'
                    }
                }
            }
        }
    
    def _estimate_required_flow(self) -> float:
        """
        Estimate required O₂ flow to maintain target pressure.
        
        TODO: Implement thermodynamic calculation:
        - Get current pressure from target component
        - Calculate pressure deficit
        - Convert to mass flow requirement
        - Account for temperature effects
        
        Skeleton for thermodynamic model:
            P_target = self.pressure_bar * 1e5  # Pa
            P_current = self._target_component.get_current_pressure()
            delta_P = P_target - P_current
            
            # Ideal gas law: PV = mRT
            # Required mass: m = (delta_P * V) / (R * T)
            volume = self._target_component.get_volume()
            temperature = self._target_component.get_temperature()
            required_mass = (delta_P * volume) / (GasConstants.R_O2 * temperature)
            
            return required_mass / self.dt  # Convert to kg/h
        """
        # Placeholder
        return 10.0  # kg/h
```

***

#### **B. External Waste Heat Source**

**Purpose:** Provide thermal energy input (e.g., industrial waste heat recovery, solar thermal).

**Component Specification:**

```python
"""
External waste heat source component.

Provides thermal energy from external sources for process heating,
water preheating, or power generation (future).
"""

from typing import Dict, Any, Optional
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class ExternalHeatSource(Component):
    """
    External waste heat supply source.
    
    Provides thermal energy for:
    - Water preheating (electrolyzer feed)
    - Process heating (ATR preheating)
    - Space heating
    - Future: ORC power generation
    
    Example Configuration:
        heat_source:
          thermal_power_kw: 500.0
          temperature_c: 150.0
          availability_factor: 0.85  # 85% uptime
          cost_per_kwh: 0.02  # If purchased heat
    
    Integration Points:
        - Connects to: HeatExchanger, WaterPreheater, ATRPreheater
        - Provides: Thermal power (kW) at specific temperature
    """
    
    def __init__(
        self,
        thermal_power_kw: float = 500.0,
        temperature_c: float = 150.0,
        availability_factor: float = 1.0,
        cost_per_kwh: float = 0.0,  # 0 if waste heat is free
        min_output_fraction: float = 0.2
    ):
        super().__init__()
        
        self.thermal_power_kw = thermal_power_kw
        self.temperature_k = temperature_c + 273.15
        self.temperature_c = temperature_c
        self.availability_factor = availability_factor
        self.cost_per_kwh = cost_per_kwh
        self.min_output_fraction = min_output_fraction
        
        # State
        self.available = True
        self.heat_output_kwh = 0.0
        self.current_power_kw = 0.0
        self.cumulative_heat_kwh = 0.0
        self.cumulative_cost = 0.0
        
        # Demand-driven output
        self.heat_demand_kw = 0.0  # Set by consuming components
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize heat source."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """Execute timestep - provide heat."""
        super().step(t)
        
        # Simulate availability (random outages based on availability factor)
        import random
        self.available = random.random() < self.availability_factor
        
        if not self.available:
            self.heat_output_kwh = 0.0
            self.current_power_kw = 0.0
            return
        
        # Deliver heat based on demand (clamped to capacity)
        if self.heat_demand_kw > 0:
            # Check minimum output constraint
            if self.heat_demand_kw < self.thermal_power_kw * self.min_output_fraction:
                # Below minimum - shut down (some heat sources can't operate at low load)
                self.current_power_kw = 0.0
            else:
                # Normal operation
                self.current_power_kw = min(self.heat_demand_kw, self.thermal_power_kw)
        else:
            # No demand - idle
            self.current_power_kw = 0.0
        
        # Calculate heat delivered this timestep
        self.heat_output_kwh = self.current_power_kw * self.dt
        
        # Update cumulative metrics
        self.cumulative_heat_kwh += self.heat_output_kwh
        self.cumulative_cost += self.heat_output_kwh * self.cost_per_kwh
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state."""
        return {
            **super().get_state(),
            'available': self.available,
            'thermal_power_kw': float(self.thermal_power_kw),
            'temperature_c': float(self.temperature_c),
            'temperature_k': float(self.temperature_k),
            'current_power_kw': float(self.current_power_kw),
            'heat_output_kwh': float(self.heat_output_kwh),
            'heat_demand_kw': float(self.heat_demand_kw),
            'cumulative_heat_kwh': float(self.cumulative_heat_kwh),
            'cumulative_cost': float(self.cumulative_cost),
            'utilization': float(self.current_power_kw / self.thermal_power_kw) if self.thermal_power_kw > 0 else 0.0,
            
            # Flow metadata
            'flows': {
                'outputs': {
                    'thermal_energy': {
                        'value': float(self.heat_output_kwh),
                        'unit': 'kWh',
                        'temperature_c': float(self.temperature_c),
                        'destination': 'heat_consumers'
                    }
                }
            }
        }
    
    def set_demand(self, demand_kw: float) -> None:
        """
        Set heat demand from consuming components.
        
        Called by components that need heat (water preheater, ATR, etc.)
        """
        self.heat_demand_kw = demand_kw
```

***

### 1.2 Oxygen Mixer Component

**Purpose:** Aggregate oxygen flows from multiple sources (electrolyzers + external) with thermodynamic mixing.

**Component Specification:**

```python
"""
Oxygen mixer component.

Mixes oxygen from multiple sources (electrolyzers, external suppliers)
with thermodynamic calculations for temperature and pressure equilibration.

Replaces OxygenBuffer when multiple O₂ sources exist.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.constants import GasConstants


class OxygenMixer(Component):
    """
    Multi-source oxygen mixing component.
    
    Features:
    - Accepts oxygen from multiple sources
    - Performs thermodynamic mixing calculations
    - Maintains pressure equilibration
    - Tracks source contributions
    
    Example Configuration:
        oxygen_mixer:
          capacity_kg: 1000.0
          target_pressure_bar: 5.0
          target_temperature_c: 25.0
          input_sources:
            - electrolyzer_1
            - electrolyzer_2
            - external_oxygen_source
    
    Thermodynamic Model (Skeleton):
        - Mass balance: m_total = Σ m_input_i
        - Energy balance: h_total = Σ (m_i * h_i) / m_total
        - Pressure equilibration: P_final from ideal gas law
        - Temperature mixing: T_final from energy balance
    """
    
    def __init__(
        self,
        capacity_kg: float = 1000.0,
        target_pressure_bar: float = 5.0,
        target_temperature_c: float = 25.0,
        input_source_ids: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.capacity_kg = capacity_kg
        self.target_pressure_pa = target_pressure_bar * 1e5
        self.target_pressure_bar = target_pressure_bar
        self.target_temperature_k = target_temperature_c + 273.15
        self.input_source_ids = input_source_ids or []
        
        # State
        self.mass_kg = 0.0
        self.pressure_pa = 1e5  # Start at atmospheric
        self.temperature_k = 298.15  # Start at 25°C
        
        # Input tracking
        self.input_flows: Dict[str, float] = {}  # source_id -> mass this timestep
        self.input_temperatures: Dict[str, float] = {}  # source_id -> temperature
        self.input_pressures: Dict[str, float] = {}  # source_id -> pressure
        
        # Output tracking
        self.output_mass_kg = 0.0
        self.cumulative_input_kg = 0.0
        self.cumulative_output_kg = 0.0
        self.cumulative_vented_kg = 0.0
        
        # Component references
        self._input_sources: List[Component] = []
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize mixer and resolve input sources."""
        super().initialize(dt, registry)
        
        # Resolve input source references
        for source_id in self.input_source_ids:
            if registry.has(source_id):
                self._input_sources.append(registry.get(source_id))
            else:
                logger.warning(f"Input source '{source_id}' not found in registry")
    
    def step(self, t: float) -> None:
        """
        Execute timestep - mix oxygen from all sources.
        
        Workflow:
        1. Collect O₂ from all input sources
        2. Perform thermodynamic mixing
        3. Update pressure and temperature
        4. Handle overflow if capacity exceeded
        """
        super().step(t)
        
        # Step 1: Collect inputs
        total_input_mass = 0.0
        self.input_flows.clear()
        self.input_temperatures.clear()
        self.input_pressures.clear()
        
        for source in self._input_sources:
            source_state = source.get_state()
            
            # Extract O₂ output (component-specific field names)
            o2_output = self._extract_o2_output(source_state)
            
            if o2_output > 0:
                source_id = source.component_id
                self.input_flows[source_id] = o2_output
                self.input_temperatures[source_id] = self._extract_temperature(source_state)
                self.input_pressures[source_id] = self._extract_pressure(source_state)
                total_input_mass += o2_output
        
        # Step 2: Thermodynamic mixing
        if total_input_mass > 0:
            mixed_temperature, mixed_pressure = self._perform_mixing(
                total_input_mass,
                self.input_flows,
                self.input_temperatures,
                self.input_pressures
            )
            
            # Update mixer state
            self.temperature_k = mixed_temperature
            self.pressure_pa = mixed_pressure
        
        # Step 3: Add to mixer inventory
        available_capacity = self.capacity_kg - self.mass_kg
        
        if total_input_mass <= available_capacity:
            # All fits
            self.mass_kg += total_input_mass
            self.cumulative_input_kg += total_input_mass
            vented = 0.0
        else:
            # Overflow - vent excess
            stored = available_capacity
            vented = total_input_mass - stored
            self.mass_kg = self.capacity_kg
            self.cumulative_input_kg += stored
            self.cumulative_vented_kg += vented
            
            logger.warning(f"Oxygen mixer overflow: {vented:.2f} kg vented")
    
    def get_state(self) -> Dict[str, Any]:
        """Return mixer state."""
        return {
            **super().get_state(),
            'mass_kg': float(self.mass_kg),
            'capacity_kg': float(self.capacity_kg),
            'pressure_bar': float(self.pressure_pa / 1e5),
            'pressure_pa': float(self.pressure_pa),
            'temperature_c': float(self.temperature_k - 273.15),
            'temperature_k': float(self.temperature_k),
            'fill_percentage': float(self.mass_kg / self.capacity_kg * 100),
            'cumulative_input_kg': float(self.cumulative_input_kg),
            'cumulative_output_kg': float(self.cumulative_output_kg),
            'cumulative_vented_kg': float(self.cumulative_vented_kg),
            
            # Input source breakdown
            'input_sources': {
                source_id: {
                    'mass_kg': float(mass),
                    'temperature_k': float(self.input_temperatures.get(source_id, 0)),
                    'pressure_pa': float(self.input_pressures.get(source_id, 0))
                }
                for source_id, mass in self.input_flows.items()
            },
            
            # Flow metadata
            'flows': {
                'inputs': {
                    source_id: {
                        'value': float(mass),
                        'unit': 'kg',
                        'source': source_id
                    }
                    for source_id, mass in self.input_flows.items()
                },
                'outputs': {
                    'mixed_oxygen': {
                        'value': float(self.output_mass_kg),
                        'unit': 'kg',
                        'temperature_k': float(self.temperature_k),
                        'pressure_pa': float(self.pressure_pa)
                    }
                }
            }
        }
    
    def remove_oxygen(self, mass_kg: float) -> float:
        """
        Remove oxygen from mixer (for consumption or sale).
        
        Args:
            mass_kg: Mass to remove
            
        Returns:
            Actual mass removed
        """
        removed = min(mass_kg, self.mass_kg)
        self.mass_kg -= removed
        self.output_mass_kg = removed
        self.cumulative_output_kg += removed
        
        return removed
    
    def _perform_mixing(
        self,
        total_mass: float,
        input_flows: Dict[str, float],
        input_temperatures: Dict[str, float],
        input_pressures: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Perform thermodynamic mixing calculation.
        
        This is a SKELETON - thermodynamic model needs implementation.
        
        TODO: Implement full thermodynamic mixing:
        
        1. Mass Balance:
           m_total = Σ m_i
        
        2. Energy Balance (constant pressure mixing):
           H_total = Σ (m_i * h_i)
           h_final = H_total / m_total
           T_final = f(h_final, P_final)  # From enthalpy-temperature correlation
        
        3. Pressure Equilibration:
           Option A: Constant volume mixing
               P_final * V = Σ (m_i * R * T_i)
           
           Option B: Constant pressure mixing
               P_final = P_mixer (assume mixer maintains constant pressure)
        
        4. Account for mixing entropy (if precision required):
           ΔS_mix = -R * Σ (x_i * ln(x_i))  where x_i = mole fraction
        
        Required inputs from LUT Manager:
           - h(T, P) - specific enthalpy
           - T(h, P) - temperature from enthalpy
           - Cp(T, P) - specific heat capacity
        
        Args:
            total_mass: Total mass being mixed (kg)
            input_flows: Mass from each source (kg)
            input_temperatures: Temperature of each input (K)
            input_pressures: Pressure of each input (Pa)
            
        Returns:
            Tuple of (final_temperature_K, final_pressure_Pa)
        """
        
        # PLACEHOLDER IMPLEMENTATION (simplified)
        # Replace with full thermodynamic model
        
        # Mass-weighted average temperature (simplified, ignores enthalpy)
        weighted_temp_sum = sum(
            input_flows[source_id] * input_temperatures[source_id]
            for source_id in input_flows.keys()
        )
        final_temperature = weighted_temp_sum / total_mass if total_mass > 0 else 298.15
        
        # Assume mixer maintains target pressure (constant pressure mixing)
        final_pressure = self.target_pressure_pa
        
        # TODO: Replace with:
        # final_temperature, final_pressure = self._thermodynamic_mixing(
        #     total_mass, input_flows, input_temperatures, input_pressures,
        #     current_mixer_mass=self.mass_kg,
        #     current_mixer_temp=self.temperature_k,
        #     current_mixer_pressure=self.pressure_pa
        # )
        
        return final_temperature, final_pressure
    
    def _extract_o2_output(self, state: Dict[str, Any]) -> float:
        """Extract O₂ output from component state (handles different field names)."""
        # Try common field names
        if 'o2_output_kg' in state:
            return state['o2_output_kg']
        elif 'oxygen_output_kg' in state:
            return state['oxygen_output_kg']
        elif 'flows' in state and 'outputs' in state['flows']:
            if 'oxygen' in state['flows']['outputs']:
                return state['flows']['outputs']['oxygen']['value']
        
        return 0.0
    
    def _extract_temperature(self, state: Dict[str, Any]) -> float:
        """Extract temperature from component state."""
        if 'temperature_k' in state:
            return state['temperature_k']
        elif 'temperature_c' in state:
            return state['temperature_c'] + 273.15
        return 298.15  # Default
    
    def _extract_pressure(self, state: Dict[str, Any]) -> float:
        """Extract pressure from component state."""
        if 'pressure_pa' in state:
            return state['pressure_pa']
        elif 'pressure_bar' in state:
            return state['pressure_bar'] * 1e5
        return 1e5  # Default (atmospheric)
```

***

### 1.3 Battery Component

**Purpose:** Energy storage for grid backup and load leveling.

**Component Specification:**

```python
"""
Battery energy storage system (BESS) component.

Provides backup power for electrolyzers during grid outages
and enables load leveling for cost optimization.
"""

from typing import Dict, Any, Optional
from enum import IntEnum
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry


class BatteryMode(IntEnum):
    """Battery operating modes."""
    IDLE = 0
    CHARGING = 1
    DISCHARGING = 2
    FULL = 3
    EMPTY = 4
    FAULT = 5


class BatteryStorage(Component):
    """
    Battery energy storage system for electrolyzer backup.
    
    Features:
    - Parallel connection to grid (charges when grid available)
    - Automatic switchover on grid failure
    - Configurable capacity and power limits
    - State-of-charge (SOC) tracking
    - Cycle counting and degradation (future)
    
    Example Configuration:
        battery:
          enabled: true
          capacity_kwh: 1000.0
          max_charge_power_kw: 500.0
          max_discharge_power_kw: 500.0
          charge_efficiency: 0.95
          discharge_efficiency: 0.95
          min_soc: 0.20  # 20% minimum SOC
          max_soc: 0.95  # 95% maximum SOC
    
    Integration:
        Grid → Battery (charging when available)
             ↓
        Battery → Electrolyzer (backup when grid fails)
    """
    
    def __init__(
        self,
        capacity_kwh: float = 1000.0,
        max_charge_power_kw: float = 500.0,
        max_discharge_power_kw: float = 500.0,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        min_soc: float = 0.20,  # 20%
        max_soc: float = 0.95,  # 95%
        initial_soc: float = 0.50  # Start at 50%
    ):
        super().__init__()
        
        self.capacity_kwh = capacity_kwh
        self.max_charge_power_kw = max_charge_power_kw
        self.max_discharge_power_kw = max_discharge_power_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        
        # State
        self.energy_kwh = capacity_kwh * initial_soc
        self.soc = initial_soc  # State of charge (0-1)
        self.mode = BatteryMode.IDLE
        
        # Power flows
        self.charge_power_kw = 0.0
        self.discharge_power_kw = 0.0
        
        # Cumulative tracking
        self.cumulative_charged_kwh = 0.0
        self.cumulative_discharged_kwh = 0.0
        self.charge_cycles = 0.0  # Fractional cycles
        
        # Grid status (set by external logic)
        self.grid_available = True
        self.grid_power_available_kw = 0.0
        
        # Load demand (electrolyzer power requirement)
        self.load_demand_kw = 0.0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize battery system."""
        super().initialize(dt, registry)
    
    def step(self, t: float) -> None:
        """
        Execute timestep - manage charging/discharging.
        
        Logic:
        1. If grid available:
           - Charge battery (if not full)
           - Supply load from grid
        2. If grid unavailable:
           - Supply load from battery (if sufficient SOC)
           - Track discharge
        """
        super().step(t)
        
        # Determine operating mode
        if self.grid_available:
            # Grid available - charge battery and supply load from grid
            self._charge_from_grid()
        else:
            # Grid failure - supply load from battery
            self._discharge_to_load()
        
        # Update state of charge
        self.soc = self.energy_kwh / self.capacity_kwh
        
        # Update mode status
        self._update_mode()
    
    def get_state(self) -> Dict[str, Any]:
        """Return battery state."""
        return {
            **super().get_state(),
            'capacity_kwh': float(self.capacity_kwh),
            'energy_kwh': float(self.energy_kwh),
            'soc': float(self.soc),
            'soc_percentage': float(self.soc * 100),
            'mode': int(self.mode),
            'mode_name': self.mode.name,
            'charge_power_kw': float(self.charge_power_kw),
            'discharge_power_kw': float(self.discharge_power_kw),
            'grid_available': self.grid_available,
            'cumulative_charged_kwh': float(self.cumulative_charged_kwh),
            'cumulative_discharged_kwh': float(self.cumulative_discharged_kwh),
            'charge_cycles': float(self.charge_cycles),
            
            # Flow metadata
            'flows': {
                'inputs': {
                    'grid_charging': {
                        'value': float(self.charge_power_kw * self.dt),
                        'unit': 'kWh',
                        'source': 'grid'
                    }
                } if self.grid_available else {},
                'outputs': {
                    'load_supply': {
                        'value': float(self.discharge_power_kw * self.dt),
                        'unit': 'kWh',
                        'destination': 'electrolyzer'
                    }
                } if not self.grid_available else {}
            }
        }
    
    def _charge_from_grid(self) -> None:
        """Charge battery from grid when available."""
        # Calculate available charging capacity
        available_capacity_kwh = (self.max_soc - self.soc) * self.capacity_kwh
        
        if available_capacity_kwh > 0.01:  # Not full
            # Determine charge power
            max_charge_energy = self.max_charge_power_kw * self.dt
            
            # Available grid power after supplying load
            available_grid_power = self.grid_power_available_kw - self.load_demand_kw
            available_grid_energy = max(0, available_grid_power * self.dt)
            
            # Charge amount (limited by capacity, max power, and available grid)
            charge_energy = min(
                available_capacity_kwh,
                max_charge_energy,
                available_grid_energy
            )
            
            if charge_energy > 0:
                # Apply charging efficiency
                energy_stored = charge_energy * self.charge_efficiency
                
                self.energy_kwh += energy_stored
                self.charge_power_kw = charge_energy / self.dt
                self.cumulative_charged_kwh += charge_energy
                
                # Track cycles (1 cycle = charge full capacity once)
                self.charge_cycles += charge_energy / self.capacity_kwh
                
                self.mode = BatteryMode.CHARGING
            else:
                self.charge_power_kw = 0.0
                self.mode = BatteryMode.IDLE
        else:
            # Battery full
            self.charge_power_kw = 0.0
            self.mode = BatteryMode.FULL
    
    def _discharge_to_load(self) -> None:
        """Discharge battery to supply load when grid unavailable."""
        # Calculate available discharge capacity
        available_discharge_kwh = (self.soc - self.min_soc) * self.capacity_kwh
        
        if available_discharge_kwh > 0.01:  # Above minimum SOC
            # Determine discharge power needed
            required_energy = self.load_demand_kw * self.dt
            max_discharge_energy = self.max_discharge_power_kw * self.dt
            
            # Discharge amount (limited by available energy, max power, and load)
            discharge_energy = min(
                available_discharge_kwh,
                max_discharge_energy,
                required_energy / self.discharge_efficiency  # Account for efficiency
            )
            
            if discharge_energy > 0:
                # Energy delivered to load (after efficiency loss)
                energy_delivered = discharge_energy * self.discharge_efficiency
                
                self.energy_kwh -= discharge_energy
                self.discharge_power_kw = energy_delivered / self.dt
                self.cumulative_discharged_kwh += discharge_energy
                
                self.mode = BatteryMode.DISCHARGING
            else:
                self.discharge_power_kw = 0.0
                self.mode = BatteryMode.IDLE
        else:
            # Battery depleted
            self.discharge_power_kw = 0.0
            self.mode = BatteryMode.EMPTY
            logger.warning("Battery depleted - cannot supply load")
    
    def _update_mode(self) -> None:
        """Update battery operating mode based on SOC."""
        if self.soc >= self.max_soc:
            if self.mode != BatteryMode.CHARGING:
                self.mode = BatteryMode.FULL
        elif self.soc <= self.min_soc:
            if self.mode != BatteryMode.DISCHARGING:
                self.mode = BatteryMode.EMPTY
    
    def set_grid_status(self, available: bool, power_kw: float = 0.0) -> None:
        """
        Set grid availability status.
        
        Called by grid/power management component or simulation event.
        
        Args:
            available: True if grid power available
            power_kw: Available grid power (kW)
        """
        self.grid_available = available
        self.grid_power_available_kw = power_kw
    
    def set_load_demand(self, demand_kw: float) -> None:
        """
        Set load demand (electrolyzer power requirement).
        
        Args:
            demand_kw: Power demand from electrolyzer (kW)
        """
        self.load_demand_kw = demand_kw
    
    def get_available_power(self) -> float:
        """
        Get available power for load (kW).
        
        Returns grid power if available, battery power otherwise.
        """
        if self.grid_available:
            return self.grid_power_available_kw
        else:
            # Calculate max discharge power given current SOC
            available_energy = (self.soc - self.min_soc) * self.capacity_kwh
            max_power = min(
                self.max_discharge_power_kw,
                available_energy / self.dt
            ) * self.discharge_efficiency
            
            return max_power
```

***

## 2. Configuration Schema Updates

### 2.1 YAML Configuration Examples

**File:** `configs/plant_with_battery_and_external.yaml`

```yaml
name: "Advanced Dual-Path Plant with Battery & External Inputs"
version: "2.1"
description: >
  Enhanced plant configuration with:
  - Battery backup for electrolyzers
  - External oxygen source
  - External waste heat recovery
  - Oxygen mixer for multiple O₂ sources

# Production (unchanged)
production:
  electrolyzer_1:
    max_power_mw: 1.5
    efficiency: 0.67
  
  electrolyzer_2:
    max_power_mw: 1.0
    efficiency: 0.65
  
  atr:
    max_ng_flow_kg_h: 100.0
    efficiency: 0.75

# NEW: External inputs
external_inputs:
  oxygen_source:
    enabled: true
    mode: "fixed_flow"
    flow_rate_kg_h: 50.0
    pressure_bar: 5.0
    cost_per_kg: 0.15
  
  heat_source:
    enabled: true
    thermal_power_kw: 500.0
    temperature_c: 150.0
    availability_factor: 0.85
    cost_per_kwh: 0.02

# NEW: Oxygen management (replaces simple buffer)
oxygen_management:
  use_mixer: true  # If true, use OxygenMixer; if false, use OxygenBuffer
  
  mixer:
    capacity_kg: 1000.0
    target_pressure_bar: 5.0
    target_temperature_c: 25.0
    input_sources:
      - electrolyzer_1
      - electrolyzer_2
      - external_oxygen_source

# NEW: Battery storage
battery:
  enabled: true  # Optional component
  capacity_kwh: 1000.0
  max_charge_power_kw: 500.0
  max_discharge_power_kw: 500.0
  charge_efficiency: 0.95
  discharge_efficiency: 0.95
  min_soc: 0.20
  max_soc: 0.95
  initial_soc: 0.50

# Storage (unchanged)
storage:
  lp_tanks:
    count: 4
    capacity_kg: 50.0
    pressure_bar: 30.0
  
  hp_tanks:
    count: 8
    capacity_kg: 200.0
    pressure_bar: 350.0

# ... rest of configuration
```

***

### 2.2 PlantBuilder Enhancements

**File:** `h2_plant/config/plant_builder.py` (additions)

```python
class PlantBuilder:
    """Enhanced with external inputs, mixer, and battery support."""
    
    def build(self) -> None:
        """Build complete plant system and populate registry."""
        logger.info(f"Building plant: {self.config.name}")
        
        # Existing layers
        self._build_lut_manager()
        self._build_production()
        self._build_storage()
        self._build_compression()
        self._build_utilities()
        
        # NEW: Build external inputs
        self._build_external_inputs()
        
        # NEW: Build oxygen management (mixer or buffer)
        self._build_oxygen_management()
        
        # NEW: Build battery (if enabled)
        self._build_battery()
        
        logger.info(f"Plant built successfully: {self.registry.get_component_count()} components")
    
    def _build_external_inputs(self) -> None:
        """Build external input components."""
        if not hasattr(self.config, 'external_inputs'):
            return
        
        ext_inputs = self.config.external_inputs
        
        # External oxygen source
        if hasattr(ext_inputs, 'oxygen_source') and ext_inputs.oxygen_source.enabled:
            from h2_plant.components.external.oxygen_source import ExternalOxygenSource
            
            o2_source = ExternalOxygenSource(
                mode=ext_inputs.oxygen_source.mode,
                flow_rate_kg_h=ext_inputs.oxygen_source.flow_rate_kg_h,
                pressure_bar=ext_inputs.oxygen_source.pressure_bar,
                cost_per_kg=ext_inputs.oxygen_source.cost_per_kg
            )
            self.registry.register('external_oxygen_source', o2_source, component_type='external_input')
        
        # External heat source
        if hasattr(ext_inputs, 'heat_source') and ext_inputs.heat_source.enabled:
            from h2_plant.components.external.heat_source import ExternalHeatSource
            
            heat_source = ExternalHeatSource(
                thermal_power_kw=ext_inputs.heat_source.thermal_power_kw,
                temperature_c=ext_inputs.heat_source.temperature_c,
                availability_factor=ext_inputs.heat_source.availability_factor,
                cost_per_kwh=ext_inputs.heat_source.cost_per_kwh
            )
            self.registry.register('external_heat_source', heat_source, component_type='external_input')
    
    def _build_oxygen_management(self) -> None:
        """Build oxygen management (mixer or buffer)."""
        if not hasattr(self.config, 'oxygen_management'):
            # Default to simple buffer
            self._build_oxygen_buffer()
            return
        
        o2_mgmt = self.config.oxygen_management
        
        if o2_mgmt.use_mixer:
            # Use mixer for multiple sources
            from h2_plant.components.mixing.oxygen_mixer import OxygenMixer
            
            mixer = OxygenMixer(
                capacity_kg=o2_mgmt.mixer.capacity_kg,
                target_pressure_bar=o2_mgmt.mixer.target_pressure_bar,
                target_temperature_c=o2_mgmt.mixer.target_temperature_c,
                input_source_ids=o2_mgmt.mixer.input_sources
            )
            self.registry.register('oxygen_mixer', mixer, component_type='oxygen_management')
        else:
            # Use simple buffer
            self._build_oxygen_buffer()
    
    def _build_battery(self) -> None:
        """Build battery storage (if enabled)."""
        if not hasattr(self.config, 'battery') or not self.config.battery.enabled:
            return
        
        from h2_plant.components.storage.battery_storage import BatteryStorage
        
        battery = BatteryStorage(
            capacity_kwh=self.config.battery.capacity_kwh,
            max_charge_power_kw=self.config.battery.max_charge_power_kw,
            max_discharge_power_kw=self.config.battery.max_discharge_power_kw,
            charge_efficiency=self.config.battery.charge_efficiency,
            discharge_efficiency=self.config.battery.discharge_efficiency,
            min_soc=self.config.battery.min_soc,
            max_soc=self.config.battery.max_soc,
            initial_soc=self.config.battery.initial_soc
        )
        self.registry.register('battery', battery, component_type='energy_storage')
```

***

## 3. Directory Structure Updates

```
h2_plant/
├── components/
│   ├── production/      (existing)
│   ├── storage/         (existing)
│   │   └── battery_storage.py    # NEW
│   ├── compression/     (existing)
│   ├── utility/         (existing)
│   │
│   ├── external/        # NEW DIRECTORY
│   │   ├── __init__.py
│   │   ├── oxygen_source.py
│   │   └── heat_source.py
│   │
│   └── mixing/          # NEW DIRECTORY
│       ├── __init__.py
│       └── oxygen_mixer.py
```

***

## 4. Integration Examples

### 4.1 Complete System with All New Components

```python
"""
Example: Running plant with battery, external inputs, and mixer.
"""

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine

# Load configuration with new components
plant = PlantBuilder.from_file("configs/plant_with_battery_and_external.yaml")
registry = plant.registry

# Verify new components registered
print("Registered components:")
for comp_id in registry.get_all_ids():
    print(f"  - {comp_id}")

# Initialize
registry.initialize_all(dt=1.0)

# Create simulation engine
engine = SimulationEngine(registry, plant.config.simulation)

# Simulate grid outage event at hour 1000
from h2_plant.simulation.event_scheduler import Event

def grid_outage_handler(registry):
    """Simulate grid failure - battery takes over."""
    if registry.has('battery'):
        battery = registry.get('battery')
        battery.set_grid_status(available=False, power_kw=0.0)
        print(f"Grid outage! Battery SOC: {battery.soc*100:.1f}%")

def grid_restore_handler(registry):
    """Restore grid power."""
    if registry.has('battery'):
        battery = registry.get('battery')
        battery.set_grid_status(available=True, power_kw=3000.0)
        print(f"Grid restored. Battery SOC: {battery.soc*100:.1f}%")

# Schedule events
outage_event = Event(
    hour=1000,
    event_type="grid_outage",
    handler=grid_outage_handler
)
restore_event = Event(
    hour=1024,  # 24-hour outage
    event_type="grid_restore",
    handler=grid_restore_handler
)

engine.schedule_event(outage_event)
engine.schedule_event(restore_event)

# Run simulation
results = engine.run()

# Analyze battery performance
battery_state = results['final_states']['battery']
print(f"\nBattery Performance:")
print(f"  Final SOC: {battery_state['soc_percentage']:.1f}%")
print(f"  Charge Cycles: {battery_state['charge_cycles']:.2f}")
print(f"  Energy Discharged: {battery_state['cumulative_discharged_kwh']:.1f} kWh")
```

***

## 5. Summary & Recommendations

###  **Architecture Compatibility: 100%**

All requested components fit **perfectly** into the existing architecture:

| **New Component** | **Layer** | **Directory** | **Dependencies** | **Status** |
|------------------|-----------|---------------|------------------|------------|
| External Oxygen Source | Layer 3 | `components/external/` | None |  Ready to implement |
| External Heat Source | Layer 3 | `components/external/` | None |  Ready to implement |
| Oxygen Mixer | Layer 3 | `components/mixing/` | GasConstants, LUT Manager |  Ready to implement |
| Battery Storage | Layer 3 | `components/storage/` | None |  Ready to implement |

---

### **Implementation Checklist**

**Week 1: External Inputs**
- [ ] Create `components/external/` directory
- [ ] Implement `ExternalOxygenSource` component
- [ ] Implement `ExternalHeatSource` component
- [ ] Add configuration schema extensions
- [ ] Write unit tests

**Week 2: Oxygen Mixer**
- [ ] Create `components/mixing/` directory
- [ ] Implement `OxygenMixer` component with thermodynamic skeleton
- [ ] Add multi-source input logic
- [ ] Integrate with PlantBuilder
- [ ] Write integration tests

**Week 3: Battery Storage**
- [ ] Implement `BatteryStorage` component
- [ ] Add grid management logic
- [ ] Create grid outage event handlers
- [ ] Test charging/discharging cycles
- [ ] Validate SOC tracking

**Week 4: Integration & Testing**
- [ ] Update PlantBuilder for all new components
- [ ] Create example configuration file
- [ ] Write end-to-end test scenarios
- [ ] Document integration points
- [ ] Performance validation

***

###  **Key Design Decisions Made**

1. **Battery is Optional**   
   - Configuration flag: `battery.enabled: true/false`
   - Components gracefully handle battery absence

2. **Mixer Replaces Buffer**   
   - Configuration flag: `oxygen_management.use_mixer: true/false`
   - Backward compatible with simple OxygenBuffer

3. **Thermodynamic Skeleton Ready**   
   - Placeholder mixing calculations
   - Clear TODO markers for full thermodynamic model
   - Integration points with LUT Manager defined

4. **Plug-and-Play Architecture**   
   - All components follow Component ABC
   - No architectural modifications needed
   - Clean separation of concerns

***