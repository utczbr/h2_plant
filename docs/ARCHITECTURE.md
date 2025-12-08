# Dual-Path Hydrogen Production System v2.0 - System Architecture

**Document Version:** 1.1  
**Last Updated:** November 20, 2025  
**Target Audience:** Senior Engineers, System Architects, New Maintainers

---

## Purpose

This document provides a **comprehensive architectural overview** of the Dual-Path Hydrogen Production System v2.0. It details the execution flow, layered design, and critical advanced capabilities that elevate the system from a simple simulator to a high-fidelity engineering tool.

---

## System Overview

The Dual-Path Hydrogen Production System is a **modular, high-performance simulation framework** for modeling industrial-scale hydrogen production. It combines rigorous thermodynamic physics with event-driven orchestration to model:

- **Dual Pathways:** Grid-powered electrolysis and natural gas autothermal reforming (ATR).
- **Complex Physics:** Real-gas mixtures (H₂, O₂, CO₂, H₂O) with phase equilibrium.
- **Deep Composition:** Recursive subsystems (e.g., electrolyzers containing pumps, heat exchangers).
- **HPC Performance:** Sub-millisecond timesteps via LUT caching and Numba JIT.
- **Interactive GUI:** Node-based visual editor for plant configuration.
- **Rich Visualization:** Automated generation of interactive dashboards and reports.

**Scale:** 8,760 minutes timesteps per annual simulation cycle[file:1].

---

## Execution Flow: SimulationEngine Loop

The core execution pattern follows a **registry-driven event loop** where the `SimulationEngine` orchestrates component lifecycle management through the `ComponentRegistry`[file:3].

### Sequence Diagram

```
sequenceDiagram
    participant User
    participant SimEngine as SimulationEngine
    participant EventSched as EventScheduler
    participant Registry as ComponentRegistry
    participant Components as Component[]
    participant FlowTracker
    participant StateManager

    User->>SimEngine: run(start_hour=0, end_hour=8760)
    SimEngine->>Registry: initialize_all(dt=1.0)
    Registry->>Components: initialize(dt, registry)
    Components-->>Registry: ✓ Initialized

    loop Every Minute (t = 0..518400)
        SimEngine->>EventSched: process_events(t)
        
        SimEngine->>Registry: step_all(t)
        Registry->>Components: step(t)
        Components->>Components: Internal Physics & Logic
        Components-->>Registry: ✓ Timestep complete
        
        SimEngine->>FlowTracker: record_flows(t)
        
        alt Checkpoint Interval (t % 168 == 0)
            SimEngine->>Registry: get_all_states()
            Registry->>Components: get_state()
            Components-->>Registry: Deep State Dicts
            SimEngine->>StateManager: save_checkpoint(t, state)
        end
    end

    SimEngine-->>User: Simulation complete
```

---

## Layered Architecture (6 Layers)

The system is organized into **six distinct layers**, plus a set of **Advanced Capabilities** that cut across layers[file:1].

### Layer 1: Core Foundation
**Purpose:** Establish standardized interfaces and shared abstractions.
- **`Component` ABC:** Defines the strict `initialize`, `step`, `get_state` lifecycle[file:2].
- **`ComponentRegistry`:** Central orchestrator for dependency injection.
- **`Integer Enums`:** Numba-compatible state definitions (`TankState`, `FlowType`).

### Layer 2: Performance Optimization
**Purpose:** Achieve 50-200x speedup on computational bottlenecks.
- **`LUTManager`:** 3D lookup tables for pure fluids and **mixture properties** (H₂/O₂/CO₂/H₂O).
- **`numba_ops`:** JIT-compiled hot paths for flash calculations and array operations.
- **`TankArray`:** Vectorized storage logic using NumPy.

### Layer 3: Component Implementations
**Purpose:** Standardized simulation entities.
- **Production:** `DetailedPEMElectrolyzer`, `SOECOperator`, `ATRProductionSource`.
- **Storage:** `TankArray`, `SourceIsolatedTanks`, `OxygenBuffer`, `BatteryStorage`, `H2StorageTankEnhanced`.
- **Compression:** `FillingCompressor`, `OutgoingCompressor`, `CompressorStorage`.
- **Separation:** `KnockOutDrum`, `Coalescer`, `OxygenMixer`, `MultiComponentMixer`.
- **Thermal:** `Chiller`, `HeatExchanger`.
- **External:** `ExternalOxygenSource`, `ExternalHeatSource`.
- **Water:** `WaterQualityTestBlock`, `WaterTreatmentBlock`, `UltrapureWaterStorageTank`, `WaterPumpThermodynamic`.
- **Utility:** `DemandScheduler`, `EnergyPriceTracker`.

### Layer 4: Pathway Orchestration
**Purpose:** Coordinate production and allocation strategies.
- **`DualPathCoordinator`:** Economic optimization and demand allocation.
- **`IsolatedProductionPath`:** Encapsulates source → storage → compression chains.
- **`PlantBuilder`:** YAML-driven component instantiation.

### Layer 5: Simulation Engine
**Purpose:** Execution and monitoring.
- **`SimulationEngine`:** Main loop and event scheduling.
- **`StateManager`:** Checkpoint persistence (JSON/Pickle).
- **`MonitoringSystem`:** Real-time metrics.
- **`FlowTracker`:** Topology-aware flow tracking for Sankey diagrams.
- **`MetricsCollector`:** Centralized data gathering for the visualization system.

### Layer 6: User Interface
**Purpose:** Visual configuration and interaction.
- **`PlantEditorWindow`:** Main GUI entry point (PySide6).
- **`NodeEditor`:** Visual programming interface for connecting components.
- **`GraphGenerator`:** Post-simulation reporting engine.

---

## Advanced Capabilities

These features distinguish the system as a high-fidelity engineering tool.

### 1. High-Fidelity Physics & Mixing[file:5]
Unlike simple mass-balance simulators, this system implements rigorous thermodynamics:
- **`MultiComponentMixer`:** Handles real-gas mixtures (H₂, O₂, CO₂, H₂O) using **UV-flash calculations**.
- **Phase Equilibrium:** Solves Rachford-Rice equations to detect water condensation and phase splitting.
- **LUT Integration:** The `LUTManager` supports mixture property caching to maintain performance (<2ms/step) despite physical complexity.

### 2. Recursive Subsystem Decomposition[file:6]
Components are not always atomic. The architecture supports **Composite Components**:
- **Pattern:** A parent component (e.g., `DetailedPEMElectrolyzer`) manages child components.
- **Example:** An electrolyzer contains:
  - `RecirculationPump` (P-1, P-2)
  - `HeatExchanger` (HX-1, HX-2)
  - `SeparationTank` (ST-1, ST-2)
- **Lifecycle:** The parent's `step()` orchestrates the children, and `get_state()` aggregates their data into a nested structure.

### 3. Topology-Aware Flow Tracking[file:4]
The **`FlowTracker`** (in Layer 5) provides "Flow Intelligence":
- **Graph Topology:** Automatically maps connections between components.
- **Sankey Generation:** Captures mass, energy, and cost flows for visualization.
- **Enhanced States:** Component state dictionaries include flow metadata (source, destination, flow type) to enable rich dashboards.

### 4. External Interfaces[file:11]
The system models boundaries with the outside world:
- **`ExternalOxygenSource`:** For importing O₂ when internal production is insufficient.
- **`ExternalWasteHeatSource`:** For integrating industrial heat streams.
- **Interface:** These components behave like standard sources but track "imported" vs "produced" resources for economic analysis.

### 5. Integrated Visualization & Reporting[file:visualization/README.md]
The system includes a comprehensive reporting engine:
- **`MetricsCollector`:** Hooks into the simulation loop to capture time-series data.
- **`GraphGenerator`:** Produces interactive Plotly graphs (HTML) and static images (PNG/PDF).
- **`Dashboard`:** Aggregates graphs into a unified HTML report.
- **Configurable:** Users can enable/disable specific graph categories (Production, Economics, etc.) via YAML.

---

## Directory Structure

The folder structure reflects the layered design and advanced capabilities[file:1]:

```
h2_plant/
├── core/                          # Layer 1: Foundation
│   ├── component.py               # Component ABC
│   ├── component_registry.py      # Registry
│   └── ...
├── optimization/                  # Layer 2: Performance
│   ├── lut_manager.py             # Thermodynamics & Mixtures
│   └── numba_ops.py               # JIT Flash Calculations
├── models/                        # Physics Models (Separated from Components)
│   ├── pem_physics.py
│   ├── soec_operation.py
│   └── ...
├── components/                    # Layer 3: Components
│   ├── production/
│   │   ├── pem_electrolyzer_detailed.py  # Uses models/pem_physics.py
│   │   └── ...
│   ├── mixing/
│   │   ├── multicomponent_mixer.py       # Physics Engine
│   │   └── oxygen_mixer.py
│   ├── external/
│   │   ├── oxygen_source.py
│   │   └── heat_source.py
│   ├── water/
│   │   ├── quality_test.py
│   │   ├── treatment.py
│   │   ├── storage.py
│   │   └── pump.py
│   └── ...
├── pathways/                      # Layer 4: Orchestration
├── config/                        # YAML Builders
├── simulation/                    # Layer 5: Execution
│   ├── engine.py
│   ├── flow_tracker.py            # Flow Intelligence
│   └── ...
├── visualization/                 # Reporting Engine
│   ├── metrics_collector.py
│   ├── graph_generator.py
│   └── ...
├── gui/                           # Layer 6: User Interface
│   ├── main.py
│   ├── nodes/                     # Visual Nodes
│   └── ...
├── data/                          # Static Data Assets
└── utils/                         # Shared Utilities
```

---

## Component Lifecycle Contract

Every component, whether atomic or composite, follows the strict three-phase lifecycle[file:2]:

1. **`initialize(dt, registry)`**: Allocate memory, resolve dependencies (including child components).
2. **`step(t)`**: Execute physics, mixing logic, and update state.
3. **`get_state()`**: Return a JSON-serializable dict. For composites, this includes nested states of all sub-components.

---

***
