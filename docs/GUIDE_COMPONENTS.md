# Component Development Guide

This guide describes how to implement new components in the Dual-Path Hydrogen Production System using four core patterns: Standard, Composite, Physics, and Flow-aware components.[file:2][file:11]  
All components must comply with the Component ABC lifecycle, integrate cleanly with the ComponentRegistry, and be buildable via the configuration-driven PlantBuilder.[file:2][file:10][file:12]

---

## 1. Core Lifecycle and Rules

Every simulation entity must inherit from `h2_plant.core.component.Component` and implement `initialize(dt, registry)`, `step(t)`, and `get_state()`.[file:2][file:10]  
The base class sets `dt`, stores the `ComponentRegistry` reference, guards against use before initialization, and returns a minimal default state dictionary.[file:2][file:10]

Key lifecycle expectations:[file:2][file:10]  
- `initialize(dt, registry)`: Validate configuration, allocate arrays, resolve dependencies from the registry.[file:2][file:10]  
- `step(t)`: Read upstream states via the registry, run physics or logic, and update internal state for this timestep.[file:2][file:11]  
- `get_state()`: Return a JSON-serializable dict including scalar state, any arrays in list form if needed, and optional `flows` metadata for FlowTracker.[file:2][file:4][file:6]

Minimal skeleton:

```
from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

class MyComponent(Component):
    def __init__(self, some_param: float):
        super().__init__()
        self.some_param = some_param
        self.internal_state = 0.0

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        # Validate and set up
        if self.some_param <= 0:
            raise ValueError("some_param must be positive")

    def step(self, t: float) -> None:
        super().step(t)
        # Update internal_state
        self.internal_state += self.some_param * self.dt

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "internal_state": float(self.internal_state),
        }
```

---

## 2. Standard Pattern: Simple Components

The Standard pattern covers atomic components such as simple tanks, pumps without subsystems, or scalar controllers that do not contain child components.[file:10][file:11]  
These components encapsulate their own logic and state but do not step or aggregate other Components internally.[file:2][file:11]

### 2.1 Design Guidelines

For a Standard component:[file:2][file:10]  
- Inherit directly from `Component` and keep constructor arguments simple and serializable.[file:2][file:11]  
- Use primitive attributes or NumPy arrays for state, and ensure everything in `get_state()` can be serialized without custom encoders.[file:2][file:11]  
- Use enums from `core.enums` as `IntEnum` and convert to `int` in `get_state()`.[file:10][file:11]

### 2.2 Example: SimpleTank Pattern

This is analogous to the SimpleTank example in the Core Foundation specification but extended slightly to show a fully self-contained Standard component.[file:10][file:11]

```
from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import TankState

class SimpleTank(Component):
    """Basic hydrogen storage tank with mass tracking."""

    def __init__(self, capacity_kg: float, pressure_bar: float):
        super().__init__()
        self.capacity_kg = capacity_kg
        self.pressure_bar = pressure_bar
        self.mass_kg = 0.0
        self.state = TankState.EMPTY

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if self.capacity_kg <= 0:
            raise ValueError("Tank capacity must be positive")

    def step(self, t: float) -> None:
        super().step(t)
        fill_ratio = self.mass_kg / self.capacity_kg if self.capacity_kg > 0 else 0.0
        if fill_ratio >= 0.99:
            self.state = TankState.FULL
        elif fill_ratio <= 0.01:
            self.state = TankState.EMPTY
        else:
            self.state = TankState.IDLE

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "mass_kg": float(self.mass_kg),
            "capacity_kg": float(self.capacity_kg),
            "pressure_bar": float(self.pressure_bar),
            "state": int(self.state),
            "fill_ratio": float(self.mass_kg / self.capacity_kg if self.capacity_kg else 0.0),
        }

    def fill(self, mass_kg: float) -> float:
        available = max(self.capacity_kg - self.mass_kg, 0.0)
        actual = min(mass_kg, available)
        self.mass_kg += actual
        return actual
```

---

## 3. Composite Pattern: Hierarchical Subsystems

The Composite pattern is used when a parent component such as `DetailedPEMElectrolyzer` needs to own and orchestrate a set of child Components that each have their own lifecycle.[file:6][file:8]  
This is implemented via `h2_plant.core.composite_component.CompositeComponent`, which extends `Component` and manages a list of subsystems.[file:8][file:6]

Composite behavior:[file:6][file:8]  
- The parent calls `subsystem.initialize(dt, registry)` on all children inside its own `initialize` implementation.[file:8][file:6]  
- The parent’s `step(t)` can both call `subsystem.step(t)` and implement extra coupling logic before or after stepping children.[file:6][file:8]  
- `get_state()` returns the parent state plus a nested `subsystems` mapping keyed by child `component_id` or a generated name.[file:8][file:6]

### 3.1 Composite Base Class

`CompositeComponent` encapsulates the repetitive wiring and is the recommended base for any hierarchical component.[file:8][file:6]

```
from typing import Dict, Any, List
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

class CompositeComponent(Component):
    def __init__(self) -> None:
        super().__init__()
        self._subsystems: List[Component] = []

    def add_subsystem(self, name: str, component: Component) -> None:
        self._subsystems.append(component)
        setattr(self, name, component)

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        for subsystem in self._subsystems:
            subsystem.initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        for subsystem in self._subsystems:
            subsystem.step(t)

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "subsystems": {
                getattr(subsystem, "component_id", f"subsystem_{i}"): subsystem.get_state()
                for i, subsystem in enumerate(self._subsystems)
            },
        }
```

### 3.2 Example: DetailedPEMElectrolyzer Wrapper

In practice, the `pem_electrolyzer_detailed.py` file defines many subsystems such as `FeedwaterInlet`, `RecirculationPump`, and `PEMStackArray`, each inheriting from `Component`.[file:6]
A top-level composite wrapper, `DetailedPEMElectrolyzer`, encapsulates these building blocks and presents a clean interface to upstream coordinators.[file:6][file:8]

```python
from h2_plant.core.composite_component import CompositeComponent
from h2_plant.components.production.pem_electrolyzer_detailed import (
    FeedwaterInlet,
    RecirculationPump,
    PEMStackArray
)

class DetailedPEMElectrolyzer(CompositeComponent):
    """
    Simplified example of the detailed electrolyzer composite.
    It owns and orchestrates its subsystems.
    """
    def __init__(self, max_power_kw: float):
        super().__init__()
        # Add subsystems
        self.add_subsystem('feedwater', FeedwaterInlet(max_flow_kg_h=1000.0))
        self.add_subsystem('pump1', RecirculationPump("P1", 1000.0, 5.0))
        self.add_subsystem('stacks', PEMStackArray(num_stacks=10, stack_config={'max_current_a': 500.0}))

    def step(self, t: float) -> None:
        # Custom orchestration logic before stepping children
        self.pump1.flow_rate_kg_h = self.feedwater.current_flow_kg_h
        
        # The base CompositeComponent step will call step() on all children
        super().step(t)
        
        # Post-step aggregation
        self.h2_product_kg_h = self.stacks.get_state()['total_h2_output_kg_h']

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["h2_product_kg_h"] = float(self.h2_product_kg_h)
        return base
```

---

## 4. Physics Pattern: Using MultiComponentMixer and LUTManager

The Physics pattern is for components that must perform real thermodynamics beyond simple algebra, often by integrating with `MultiComponentMixer` and `LUTManager`.[file:5][file:7][file:9]  
This pattern appears in the `MultiComponentMixer`, which handles gas mixing, UV-flash calculations, phase equilibrium, and real-gas property evaluation for H₂/O₂/CO₂/CH₄/H₂O mixtures.[file:5][file:7]

### 4.1 Using MultiComponentMixer as a Downstream Physics Engine

`MultiComponentMixer` is itself a Component that connects to upstream sources via the registry and consumes their states as input streams.[file:7][file:5]  
Upstream components that feed the mixer must expose a state schema compatible with `_extract_stream_from_state`, typically including `flow_kmol_hr`, `temperature_k`, `pressure_pa`, and a `composition` dict.[file:7][file:5]

Typical pattern for an upstream gas-producing component:

```
from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

class SimpleGasSource(Component):
    def __init__(self, species: str, flow_kmol_hr: float):
        super().__init__()
        self.species = species
        self.flow_kmol_hr = flow_kmol_hr
        self.temperature_k = 298.15
        self.pressure_pa = 1e5

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)

    def step(self, t: float) -> None:
        super().step(t)
        # For now, constant flow; advanced logic can vary by time

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "flow_kmol_hr": float(self.flow_kmol_hr),
            "temperature_k": float(self.temperature_k),
            "pressure_pa": float(self.pressure_pa),
            "composition": {self.species: 1.0},
        }
```

The mixer then collects and aggregates streams via `_collect_input_streams()` and runs `_perform_uv_flash()` to solve for temperature, pressure, and phase split from internal energy, volume, and composition.[file:7][file:5]  

### 4.2 Accessing LUTManager in Physics-Heavy Components

For components that do not need full UV-flash but still require fast property calls, use `LUTManager` from the registry.[file:9][file:11]  
`LUTManager` is registered under the id `lut_manager` by PlantBuilder and supports scalar and batch lookups for density, enthalpy, entropy, heat capacity, and viscosity.[file:9][file:12]

Pattern for using LUTManager:

```
from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager

class ThermoAwareTank(Component):
    def __init__(self, fluid: str, pressure_bar: float, temperature_k: float):
        super().__init__()
        self.fluid = fluid
        self.pressure_pa = pressure_bar * 1e5
        self.temperature_k = temperature_k
        self.density_kg_m3 = 0.0
        self._lut: LUTManager | None = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        if registry.has("lut_manager"):
            self._lut = registry.get("lut_manager")
        else:
            raise RuntimeError("LUTManager is required for ThermoAwareTank")

    def step(self, t: float) -> None:
        super().step(t)
        if self._lut is not None:
            self.density_kg_m3 = self._lut.lookup(
                self.fluid, "D", self.pressure_pa, self.temperature_k
            )

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "fluid": self.fluid,
            "pressure_pa": float(self.pressure_pa),
            "temperature_k": float(self.temperature_k),
            "density_kg_m3": float(self.density_kg_m3),
        }
```

For heavy math loops such as repeated flash calculations or large array operations, follow the optimization guide by implementing the loops in `numba_ops.py` under `@njit` and calling them from your component.[file:9][file:11]  

---

## 5. Flow Pattern: Making Components Visible to FlowTracker

The Flow pattern defines how components should annotate their `get_state()` output so that the FlowTracker can reconstruct a topology-aware picture of mass, energy, and cost flows.[file:4][file:6]  
Flow metadata is added under a `flows` key containing `inputs` and `outputs`, which are mappings of named streams to structured descriptors.[file:4][file:6]

### 5.1 Flow State Schema

The recommended schema is as follows:[file:4][file:6]  

```
{
    ...
    "flows": {
        "inputs": {
            "<stream_name>": {
                "value": <float>,
                "unit": "<unit_str>",
                "source": "<component_id_or_label>",
                # optional:
                "flow_type": "HYDROGEN_MASS",   # matches FlowType enum
                "metadata": {...}
            },
            ...
        },
        "outputs": {
            "<stream_name>": {
                "value": <float>,
                "unit": "<unit_str>",
                "destination": "<component_id_or_label>",
                # optional:
                "flow_type": "THERMAL_ENERGY",
                "metadata": {...}
            },
            ...
        }
    }
}
```

FlowTracker uses these fields together with the `FlowType` enum and `Flow` dataclass in `flow_tracker.py` to build Sankey-ready link records between source and destination nodes.[file:4]  

### 5.2 Example: NewFilterUnit with Flows

`NewFilterUnit` is a simple component that removes a fraction of impurities from an incoming gas stream while reporting both mass flows and compression work to FlowTracker.[file:4][file:6][file:11]

```
from typing import Dict, Any
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry

class NewFilterUnit(Component):
    """Generic gas filter component with flow-aware state."""

    def __init__(self, removal_fraction: float = 0.95, pressure_drop_bar: float = 1.0):
        super().__init__()
        self.removal_fraction = removal_fraction
        self.pressure_drop_bar = pressure_drop_bar

        self.inlet_kg_h = 0.0
        self.filtered_kg_h = 0.0
        self.waste_kg_h = 0.0

        self.inlet_component_id: str | None = None
        self.outlet_component_id: str | None = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        # Optionally resolve upstream/downstream IDs from configuration or registry tags

    def step(self, t: float) -> None:
        super().step(t)
        # For now assume inlet_kg_h is set externally (e.g., by coordinator)
        self.filtered_kg_h = self.inlet_kg_h * self.removal_fraction
        self.waste_kg_h = self.inlet_kg_h - self.filtered_kg_h

    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "inlet_kg_h": float(self.inlet_kg_h),
            "filtered_kg_h": float(self.filtered_kg_h),
            "waste_kg_h": float(self.waste_kg_h),
            "pressure_drop_bar": float(self.pressure_drop_bar),
            "flows": {
                "inputs": {
                    "raw_gas": {
                        "value": self.inlet_kg_h,
                        "unit": "kg/h",
                        "source": self.inlet_component_id or "upstream",
                        "flow_type": "HYDROGEN_MASS",
                    }
                },
                "outputs": {
                    "clean_gas": {
                        "value": self.filtered_kg_h,
                        "unit": "kg/h",
                        "destination": self.outlet_component_id or "downstream",
                        "flow_type": "HYDROGEN_MASS",
                    },
                    "waste_stream": {
                        "value": self.waste_kg_h,
                        "unit": "kg/h",
                        "destination": "vent_or_recycle",
                        "flow_type": "HYDROGEN_MASS",
                    },
                },
            },
        }
```

By following this schema, FlowTracker can convert these per-timestep states into `Flow` records and aggregate them into Sankey link tables for dashboards.[file:4]  

---

## 6. Registration and Configuration (ComponentRegistry + PlantBuilder)

To participate in a plant, a component must be registered with the `ComponentRegistry` and optionally made constructible via PlantBuilder from YAML configuration.[file:3][file:10][file:12]  

### 6.1 Manual Registration

Manual registration is straightforward and useful for tests or ad hoc setups.[file:3][file:10]  

```
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.utility.new_filter_unit import NewFilterUnit

registry = ComponentRegistry()
filter_unit = NewFilterUnit(removal_fraction=0.98, pressure_drop_bar=0.5)
registry.register("h2_filter", filter_unit, component_type="utility")
registry.initialize_all(dt=1.0)
```

This ensures `component_id` is assigned and `initialize()` is called for all registered components before the SimulationEngine starts stepping them.[file:3][file:10]  

### 6.2 Making a Component YAML-Constructible

To have PlantBuilder create your component from configuration, follow these steps:[file:11][file:12]  

1. **Add a config dataclass** in `plant_config.py`, for example:[file:12]  

```
from dataclasses import dataclass
from h2_plant.core.enums import AllocationStrategy  # if needed

@dataclass
class NewFilterUnitConfig:
    enabled: bool = True
    removal_fraction: float = 0.95
    pressure_drop_bar: float = 1.0

    def validate(self) -> None:
        if not 0.0 < self.removal_fraction <= 1.0:
            raise ValueError(f"removal_fraction must be in (0, 1], got {self.removal_fraction}")
        if self.pressure_drop_bar < 0.0:
            raise ValueError(f"pressure_drop_bar must be non-negative, got {self.pressure_drop_bar}")
```

2. **Wire the config into the main `PlantConfig`** so it appears under an appropriate section, for example `utility` or `compression`, and update `validate()` accordingly.[file:12]  

3. **Update the JSON schema** (`plant_schema_v1.json`) so that YAML files describing `new_filter_unit` parameters are validated at load time.[file:12]  

4. **Extend PlantBuilder** in `plant_builder.py` to instantiate and register the component during `build()`.[file:12]  

```
from h2_plant.components.utility.new_filter_unit import NewFilterUnit

class PlantBuilder:
    ...

    def _build_utilities(self) -> None:
        # Existing utilities
        ...
        # New filter
        if getattr(self.config.utility, "new_filter_unit", None) and \
           self.config.utility.new_filter_unit.enabled:
            cfg = self.config.utility.new_filter_unit
            new_filter = NewFilterUnit(
                removal_fraction=cfg.removal_fraction,
                pressure_drop_bar=cfg.pressure_drop_bar,
            )
            self.registry.register("new_filter_unit", new_filter, component_type="utility")
```

5. **Use YAML to enable and configure the component**, for example:[file:12]  

```
utility:
  new_filter_unit:
    enabled: true
    removal_fraction: 0.97
    pressure_drop_bar: 0.5
```

After these steps, the same `PlantBuilder.from_file("configs/plant_variant.yaml")` call will create and register `NewFilterUnit` alongside the existing production, storage, and compression components.[file:12]  

---

## 7. Pattern Selection Cheat Sheet

- Use the Standard pattern for atomic components with local logic only, such as simple tanks, pumps, or controllers without children.[file:10][file:11]  
- Use the Composite pattern for systems like detailed electrolyzers that are naturally decomposed into many reusable subsystems that must each be initialized, stepped, and monitored.[file:6][file:8]  
- Use the Physics pattern when you need rigorous thermodynamics via `MultiComponentMixer` or high-performance property calls via `LUTManager` and `numba_ops`.[file:5][file:7][file:9]  
- Use the Flow pattern whenever your component moves mass, energy, or cost between conceptual nodes and you want that movement visible in Sankey diagrams and dashboard analytics.[file:4][file:6]  

```