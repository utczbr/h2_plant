# Dual-Path Hydrogen Production System v2.0 — System Architecture

**Document Version:** 2.0
**Last Updated:** March 2026
**Target Audience:** Senior Engineers, System Architects, New Maintainers

> **Scope:** Conceptual system overview, layered design, and authoritative package map.
> For detailed execution-flow sequence diagrams, see [BACKEND_DIAGRAM.md](BACKEND_DIAGRAM.md).

---

## Purpose

This document is the **single canonical architecture reference** for the Dual-Path Hydrogen Production System. It covers the intended layering, system responsibilities, and major subsystem boundaries so that new contributors can form a correct mental model before navigating code.

It is intentionally *not* an exhaustive file map. Treat the Package Map section as the authoritative directory reference; for individual module internals, rely on package-level `__init__.py` files and inline docstrings.

---

## System Overview

The system is a **modular, high-performance simulation framework** for modeling industrial-scale hydrogen production. It combines rigorous thermodynamic physics with event-driven orchestration to model:

- **Dual Pathways:** Grid-powered electrolysis and natural gas autothermal reforming (ATR).
- **Complex Physics:** Real-gas mixtures (H₂, O₂, CO₂, H₂O) with phase equilibrium.
- **Deep Composition:** Recursive subsystems (e.g., electrolyzers containing pumps, heat exchangers).
- **HPC Performance:** Sub-millisecond timesteps via LUT caching and Numba JIT.
- **Interactive GUI:** Node-based visual editor for plant configuration.
- **Rich Visualization:** Automated generation of interactive dashboards and reports.

**Scale:** 8,760 hours (525,600 minutes) per annual simulation cycle.

---

## Layered Architecture (6 Layers)

The codebase is organized into **six distinct layers**, each with a single well-defined responsibility. Layer numbers are stable identifiers; they do not imply a call stack.

### Layer 1 — Core Foundation (`h2_plant/core/`)

Establishes standardized interfaces and shared abstractions used by every other layer:

- **`Component` ABC** — defines the strict `initialize / step / get_state` lifecycle contract.
- **`ComponentRegistry`** — central orchestrator for dependency injection and component management.
- **`Integer Enums`** — Numba-compatible state definitions (`TankState`, `FlowType`).

### Layer 2 — Performance Optimization (`h2_plant/optimization/`)

Achieves 50–200× speedup on computational bottlenecks:

- **`LUTManager`** — 3D lookup tables for pure fluids and mixture properties (H₂/O₂/CO₂/H₂O).
- **`numba_ops`** — JIT-compiled hot paths for flash calculations, tank array operations, and solvers.
- **`TankArray`** — vectorized storage logic using NumPy.

### Layer 3 — Component Implementations (`h2_plant/components/`)

Standardized simulation entities organized into 20 subdirectories:

`electrolysis` · `reforming` · `compression` · `separation` · `purification` · `carbon` · `thermal` · `cooling` · `water` · `storage` · `mixing` · `power` · `control` · `coordination` · `environment` · `external` · `logistics` · `utility` · `balance_of_plant`

> **Distinction:** `h2_plant/components/storage/` contains component-level tank classes (e.g., `H2StorageTankEnhanced`, `TankArray`). The top-level `h2_plant/storage/` package (see Package Map) is a separate, higher-level storage-management layer added after the original component taxonomy was established.

### Layer 4 — Pathway Orchestration (`h2_plant/pathways/`)

Coordinates production and demand-allocation strategies across the full plant:

- **`DualPathCoordinator`** — economic optimization and demand allocation logic.
- **`IsolatedProductionPath`** — encapsulates source → storage → compression chains.
- **`IntegratedPlant`** — full plant coordinator wiring all subsystems.
- **`AllocationStrategies`** — algorithmic strategies (Cost, Emissions, Balanced) for demand splitting.

### Layer 5 — Simulation Engine (`h2_plant/simulation/`)

Execution and monitoring. The `SimulationEngine` runs the main registry-driven event loop. Dispatch logic is decoupled into `h2_plant/control/dispatch.py` (pure intent layer) and bound to the engine at runtime via `HybridArbitrageEngineStrategy`.

Key submodules: `engine.py`, `event_scheduler.py`, `state_manager.py`, `flow_network.py`, `flow_tracker.py`, `monitoring.py`, `runner.py`.

> For the full timestep sequence diagram see [BACKEND_DIAGRAM.md](BACKEND_DIAGRAM.md).

### Layer 6 — User Interface & Reporting (`h2_plant/gui/`, `h2_plant/reporting/`, `h2_plant/visualization/`)

Visual configuration, interaction, and output generation:

- **`gui/`** — PySide6 node-based visual editor (`PlantEditorWindow`, `NodeEditor`, backend–GUI bridge).
- **`reporting/`** — Markdown report generator (`markdown_report.py`), stream summary tables (`stream_table.py`), detailed report engine (`report_generator.py`).
- **`visualization/`** — Matplotlib static graphs, Plotly interactive graphs, config-driven graph orchestrator, HTML dashboard generator.

---

## Integrated Control Architecture

The system uses a **Split-Layer Control Architecture** that separates economic decision-making from physical execution:

- **Logic Layer (`h2_plant/control/dispatch.py`)** — pure Python strategies (e.g., `ReferenceHybridStrategy`, `SoecOnlyStrategy`) that determine *intent*: they process market signals and output power setpoints (MW to SOEC, PEM, Grid). Stateless with respect to physics.
- **Engine Integration (`HybridArbitrageEngineStrategy`, in `h2_plant/run_integrated_simulation.py`)** — binds dispatch logic to the `SimulationEngine`, pre-allocates NumPy history arrays for the entire simulation duration, and records actual post-step outcomes.

The execution cycle per timestep: **Decide & Apply → Physics Execution → Record**. Full sequence diagram in [BACKEND_DIAGRAM.md](BACKEND_DIAGRAM.md).

---

## Component Lifecycle Contract

Every component, atomic or composite, follows a strict three-phase lifecycle:

1. **`initialize(dt, registry)`** — allocate memory, pre-compute constants, resolve registry dependencies, initialize child components if composite.
2. **`step(t)`** — execute physics, mixing logic, and reactions; causal order is enforced by the engine (inputs → processing → outputs).
3. **`get_state()`** — return a JSON-serializable dictionary of the full internal state; composites recursively include nested sub-component states.

---

## Package Map

Authoritative directory listing as of version 2.0. All paths are relative to the repository root.

```
h2_plant/
├── components/       # Layer 3: All simulation entities (20 subdirectories)
│   ├── balance_of_plant/
│   ├── carbon/
│   ├── compression/
│   ├── control/          # Component-level control (valves)
│   ├── cooling/
│   ├── coordination/
│   ├── electrolysis/
│   ├── environment/
│   ├── external/
│   ├── logistics/
│   ├── mixing/
│   ├── power/
│   ├── purification/
│   ├── reforming/
│   ├── separation/
│   ├── storage/          # Component-level tank classes (distinct from top-level storage/)
│   ├── thermal/
│   ├── utility/
│   └── water/
│
├── config/           # YAML loaders, Pydantic models, physics constants
│   ├── loader.py         #   ConfigLoader — YAML to SimulationContext
│   ├── models.py         #   Pydantic models (SimulationContext, ComponentNode)
│   ├── plant_config.py   #   ConnectionConfig, SimulationConfig
│   └── constants_physics.py  #   Physical constants (SI units)
│
├── control/          # Dispatch logic — intent layer (stateless re: physics)
│   └── dispatch.py       #   ReferenceHybridStrategy, SoecOnlyStrategy, DispatchStrategy ABC
│
├── core/             # Layer 1: Component ABC, registry, enums, types
│   ├── component.py
│   ├── component_ids.py
│   ├── component_registry.py
│   ├── composite_component.py
│   ├── graph_builder.py  #   PlantGraphBuilder — topology to components
│   ├── stream.py
│   ├── constants.py
│   ├── enums.py
│   ├── exceptions.py
│   └── types.py
│
├── data/             # Data loaders and pre-fitted model files
│   ├── price_loader.py
│   └── ATR_model_functions.pkl
│
├── economics/        # Economic analysis: LCOH, OPEX/CAPEX, revenue models
│
├── gui/              # Layer 6: PySide6 visual editor
│   ├── main_window.py
│   ├── node_editor/
│   └── core/             #   Backend–GUI bridge
│
├── models/           # Trained ML/regression models and utilities
│
├── optimization/     # Layer 2: LUT caching, Numba JIT ops
│   ├── lut_manager.py
│   └── coolprop_lut.py
│
├── pathways/         # Layer 4: Plant-level orchestration strategies
│   ├── integrated_plant.py
│   ├── isolated_production_path.py
│   └── allocation_strategies.py
│
├── reporting/        # Post-simulation reports
│   ├── markdown_report.py
│   ├── report_generator.py
│   └── stream_table.py
│
├── scripts/          # Standalone analysis and utility scripts
│
├── simulation/       # Layer 5: Engine, scheduler, flow network
│   ├── engine.py
│   ├── event_scheduler.py
│   ├── state_manager.py
│   ├── flow_network.py
│   ├── flow_tracker.py
│   ├── monitoring.py
│   └── runner.py
│
├── storage/          # Top-level storage-management layer (higher-level than components/storage/)
│
├── tests/            # Test suite
│
├── utils/            # Shared utilities
│   └── henry_solubility.py
│
├── visualization/    # Graphs and dashboards
│   ├── static_graphs.py
│   ├── graph_orchestrator.py
│   ├── graph_catalog.py
│   ├── plotly_graphs.py
│   ├── dashboard_generator.py
│   ├── metrics_collector.py
│   └── graphs/
│
├── orchestrator.py   # Legacy orchestrator (deprecated — do not use in new code)
└── run_integrated_simulation.py  # CLI entry point
```

---

## Data Flow Patterns

### Stream Propagation (Push Architecture)
Components push data downstream via `receive_input()`:
```
Producer.step() → Downstream.receive_input(port, stream, resource_type)
```

### Control Flow (Pull Architecture)
Dispatch queries component state after physics:
```
Engine → dispatch.record_post_step() → component.get_state()['actual_power']
```

### Registry Pattern
Components resolve dependencies at initialization time:
```python
def initialize(self, dt, registry):
    self._lut = registry.get(ComponentID.LUT_MANAGER)
```

---

## Configuration & Units

All configuration lives in `h2_plant/config/`. Simulation parameters (`timestep_hours`, `duration_hours`, `checkpoint_interval_hours`) are in `plant_config.py`. Physical constants use SI units internally and are converted to engineering units only at GUI/report boundaries.

| Domain      | Internal (SI) | Display (Engineering) |
|-------------|---------------|-----------------------|
| Pressure    | Pa            | bar                   |
| Temperature | K             | °C                    |
| Mass Flow   | kg/s          | kg/h                  |
| Power       | W             | MW                    |

---

## Quick Navigation

| Task | Where to look |
|------|---------------|
| Add a new component | `h2_plant/components/<category>/` → inherit `Component` → implement lifecycle → register in `h2_plant/core/graph_builder.py` |
| Change dispatch / control logic | `h2_plant/control/dispatch.py` — modify or subclass `DispatchStrategy` |
| Change history recording | `HybridArbitrageEngineStrategy` in `h2_plant/run_integrated_simulation.py` |
| Modify the simulation loop | `h2_plant/simulation/engine.py` — `SimulationEngine._execute_timestep()` |
| Change thermodynamic properties | `h2_plant/optimization/lut_manager.py` and `h2_plant/config/constants_physics.py` |
| Add economic analysis | `h2_plant/economics/` |
| Add a GUI element | `h2_plant/gui/widgets/` and `h2_plant/gui/core/` |

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [BACKEND_DIAGRAM.md](BACKEND_DIAGRAM.md) | Detailed execution-flow sequence diagrams (SimulationEngine loop, dispatch cycle) |
| [developer_guide_component.md](developer_guide_component.md) | How to implement new components |
| [docs/diagrams/](diagrams/) | Component-level architecture diagrams |
| [README.md](../README.md) | Project overview and quick start |
