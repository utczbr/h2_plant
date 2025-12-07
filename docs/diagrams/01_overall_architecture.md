# H2 Plant - Overall System Architecture

A comprehensive view of the Dual-Path Hydrogen Production System's 6-layer architecture.

## High-Level Architecture

```mermaid
%%{init: {
  "theme": "neutral",
  "flowchart": { "nodeSpacing": 50, "rankSpacing": 60, "curve": "basis" }
}}%%
graph TB
    classDef ui fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef engine fill:#ede7f6,stroke:#5e35b1,stroke-width:2px
    classDef orch fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef comp fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef core fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef ext fill:#eceff1,stroke:#546e7a,stroke-width:2px

    UI["User Interface\n(GUI + Reports)"]:::ui
    Engine["Simulation Engine\n(Execution + Monitoring)"]:::engine
    Orch["Orchestration\n(Dispatch + Flow)"]:::orch
    Components["Components\n(SOEC, PEM, Tank, ...)"]:::comp
    Core["Core Foundation\n(ABC + Registry)"]:::core
    External["External Data\n(YAML, Prices, Wind)"]:::ext

    UI -->|configures| Orch
    External -->|inputs| Orch
    Orch -->|coordinates| Engine
    Engine -->|steps| Components
    Components -->|inherits| Core
    Engine -.->|reports| UI
```

---

## Detailed System Layers

```mermaid
%%{init: {
  "theme": "neutral",
  "flowchart": { "nodeSpacing": 40, "rankSpacing": 70, "curve": "linear" }
}}%%
graph TB
    classDef ui fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    classDef engine fill:#ede7f6,stroke:#5e35b1,stroke-width:1px
    classDef orch fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    classDef comp fill:#fff3e0,stroke:#ef6c00,stroke-width:1px
    classDef perf fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    classDef core fill:#fce4ec,stroke:#c2185b,stroke-width:1px
    classDef ext fill:#eceff1,stroke:#546e7a,stroke-width:1px

    %% Layer 6: UI
    subgraph L6["Layer 6 · User Interface"]
        GUI["PlantEditorWindow"]:::ui
        NodeEditor["NodeEditor"]:::ui
        SimReport["SimReportWidget"]:::ui
    end

    %% Layer 5: Simulation Engine
    subgraph L5["Layer 5 · Simulation Engine"]
        Engine["SimulationEngine"]:::engine
        FlowNet["FlowNetwork"]:::engine
        Monitor["MonitoringSystem"]:::engine
        StateMan["StateManager"]:::engine
    end

    %% Layer 4: Orchestration
    subgraph L4["Layer 4 · Orchestration"]
        Orch["Orchestrator"]:::orch
        Dispatch["DispatchStrategy"]:::orch
        Adapter["GraphToConfigAdapter"]:::orch
    end

    %% Layer 3: Components
    subgraph L3["Layer 3 · Components"]
        SOEC["SOEC"]:::comp
        PEM["PEM"]:::comp
        Tank["Tank"]:::comp
        Comp["Compressor"]:::comp
        Pump["Pump"]:::comp
        Mixer["Mixer"]:::comp
    end

    %% Layer 2: Performance
    subgraph L2["Layer 2 · Performance"]
        LUT["LUTManager"]:::perf
        Numba["numba_ops"]:::perf
    end

    %% Layer 1: Core
    subgraph L1["Layer 1 · Core Foundation"]
        CompABC["Component ABC"]:::core
        Registry["ComponentRegistry"]:::core
        Stream["Stream"]:::core
    end

    %% External
    subgraph Ext["External Data"]
        YAML["YAML Configs"]:::ext
        Prices["Energy Prices"]:::ext
        Wind["Wind Data"]:::ext
    end

    %% Top-to-bottom flow
    GUI --> NodeEditor
    NodeEditor -->|builds| Adapter
    SimReport -->|shows| Monitor

    Adapter -->|configures| Orch
    Orch --> Dispatch
    Orch -->|uses| Registry
    Ext -->|inputs| Orch

    Engine --> FlowNet
    Engine --> Monitor
    Engine --> StateMan
    Engine -->|steps| Registry

    Registry -->|manages| CompABC
    SOEC --> CompABC
    PEM --> CompABC
    Tank --> CompABC

    SOEC -->|uses| LUT
    PEM -->|uses| LUT
    FlowNet -->|uses| Stream
```

### Technology Stack

| Layer | Technologies |
|-------|--------------|
| UI | PySide6, NodeGraphQt, Matplotlib |
| Engine | NumPy, async workers |
| Orchestration | YAML loader, Pydantic |
| Components | CoolProp, SciPy |
| Performance | Numba JIT, LUT caching |
| Core | Python ABC, dataclasses |


## Component Lifecycle Flow

```mermaid
sequenceDiagram
    participant User
    participant GUI as PlantEditorWindow
    participant Adapter as GraphToConfigAdapter
    participant Orch as Orchestrator
    participant Builder as PlantGraphBuilder
    participant Registry as ComponentRegistry
    participant Comp as Components

    User->>GUI: Design Plant Topology
    User->>GUI: Click "Run Simulation"
    GUI->>Adapter: Export Graph to YAML
    Adapter->>Orch: Create with Context
    Orch->>Builder: Build Components
    Builder->>Comp: Instantiate (PEM, SOEC, Tank, etc.)
    Comp-->>Builder: Component Instances
    Builder-->>Orch: Component Dictionary
    Orch->>Registry: Register All
    Registry->>Comp: initialize(dt, registry)
    
    loop Every Timestep (t)
        Orch->>Registry: step_all(t)
        Registry->>Comp: step(t)
        Comp-->>Registry: Updated State
        Orch->>Orch: Log History
    end
    
    Orch-->>GUI: Simulation Results
    GUI->>User: Display Reports
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph Inputs
        Wind["Wind Power<br/>(MW)"]
        Price["Spot Price<br/>(EUR/MWh)"]
        Config["Plant Config<br/>(YAML)"]
    end

    subgraph Processing
        Dispatch["Dispatch Strategy"]
        SOEC["SOEC Cluster"]
        PEM["PEM Stack"]
        BoP["Balance of Plant"]
    end

    subgraph Outputs
        H2["Hydrogen<br/>(kg/h)"]
        O2["Oxygen<br/>(kg/h)"]
        Revenue["Grid Revenue<br/>(EUR)"]
        Reports["Graphs/Dashboard"]
    end

    Wind --> Dispatch
    Price --> Dispatch
    Config --> SOEC
    Config --> PEM

    Dispatch -->|P_soec| SOEC
    Dispatch -->|P_pem| PEM
    Dispatch -->|P_sold| Revenue

    SOEC --> H2
    SOEC --> BoP
    PEM --> H2
    PEM --> O2
    PEM --> BoP

    H2 --> Reports
    Revenue --> Reports
```

## Directory Structure Mapping

| Layer | Directory | Key Files |
|-------|-----------|-----------|
| 6 - UI | `h2_plant/gui/` | `main_window.py`, `nodes/*.py` |
| 5 - Engine | `h2_plant/simulation/` | `engine.py`, `flow_network.py` |
| 4 - Orchestration | `h2_plant/` | `orchestrator.py` |
| 4 - Orchestration | `h2_plant/control/` | `dispatch.py` |
| 3 - Components | `h2_plant/components/` | `electrolysis/`, `storage/`, etc. |
| 2 - Optimization | `h2_plant/optimization/` | `lut_manager.py`, `numba_ops.py` |
| 1 - Core | `h2_plant/core/` | `component.py`, `component_registry.py` |
