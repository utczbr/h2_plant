# H2 Plant - Overall System Architecture

A comprehensive view of the Dual-Path Hydrogen Production System's 6-layer architecture.

## System Layers Overview

```mermaid
graph TB
    subgraph L6["Layer 6: User Interface"]
        GUI["PlantEditorWindow<br/>(PySide6)"]
        NodeEditor["NodeEditor<br/>(NodeGraphQt)"]
        SimReport["SimulationReportWidget"]
    end

    subgraph L5["Layer 5: Simulation Engine"]
        Engine["SimulationEngine"]
        EventSched["EventScheduler"]
        FlowNet["FlowNetwork"]
        StateMan["StateManager"]
        Monitor["MonitoringSystem"]
    end

    subgraph L4["Layer 4: Pathway Orchestration"]
        Orch["Orchestrator"]
        Dispatch["DispatchStrategy"]
        GraphAdapter["GraphToConfigAdapter"]
    end

    subgraph L3["Layer 3: Component Implementations"]
        subgraph Production
            SOEC["SOECOperator"]
            PEM["DetailedPEMElectrolyzer"]
        end
        subgraph Storage
            Tank["Tank"]
            Battery["BatteryStorage"]
        end
        subgraph BoP["Balance of Plant"]
            Comp["Compressor"]
            Pump["Pump"]
            Mixer["MultiComponentMixer"]
        end
        subgraph Thermal
            Chiller["Chiller"]
            HX["HeatExchanger"]
        end
    end

    subgraph L2["Layer 2: Performance Optimization"]
        LUT["LUTManager<br/>(Thermodynamics)"]
        NumbaOps["numba_ops<br/>(JIT Flash Calcs)"]
    end

    subgraph L1["Layer 1: Core Foundation"]
        CompABC["Component ABC"]
        Registry["ComponentRegistry"]
        Stream["Stream/MassStream"]
        Enums["IntEnums<br/>(TankState, FlowType)"]
    end

    subgraph External["External Data"]
        YAML["YAML Configs"]
        PriceData["Energy Prices"]
        WindData["Wind Data"]
    end

    %% Connections
    GUI --> NodeEditor
    GUI --> SimReport
    NodeEditor --> GraphAdapter
    SimReport --> Monitor

    Engine --> Registry
    Engine --> EventSched
    Engine --> FlowNet
    Engine --> StateMan
    Engine --> Monitor

    Orch --> Registry
    Orch --> Dispatch
    GraphAdapter --> Orch

    SOEC --> CompABC
    PEM --> CompABC
    Tank --> CompABC
    Comp --> CompABC
    Pump --> CompABC
    Mixer --> CompABC

    SOEC --> LUT
    PEM --> LUT
    Mixer --> LUT

    Registry --> CompABC
    FlowNet --> Stream

    External --> Orch
```

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
