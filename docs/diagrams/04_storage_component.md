# Storage Component Diagram

Detailed architecture of the Tank/Storage component for hydrogen storage.

## Component Overview

```mermaid
classDiagram
    class Tank {
        -float capacity_kg
        -float initial_level_kg
        -float min_level_ratio
        -float max_pressure_bar
        -float current_level_kg
        -float pressure_bar
        +initialize(dt, registry)
        +step(t, flow_in_kg_h, flow_out_kg_h)
        +get_state()
    }
    
    class Component {
        <<abstract>>
        +initialize(dt, registry)
        +step(t)
        +get_state()
    }
    
    Tank --|> Component
```

## Mass Balance Model

```mermaid
flowchart LR
    subgraph Inputs
        FlowIn["flow_in_kg_h<br/>(H₂ inflow)"]
    end
    
    subgraph Tank["Tank Storage"]
        Level["current_level_kg<br/>Mass Balance"]
        Pressure["pressure_bar<br/>P = (m/m_max) × P_max"]
    end
    
    subgraph Outputs
        FlowOut["flow_out_kg_h<br/>(H₂ delivery)"]
    end
    
    FlowIn --> Level
    Level --> Pressure
    Level --> FlowOut
```

## Step Execution Flow

```mermaid
flowchart TD
    Start([step called]) --> CalcDelta["Δm = (flow_in - flow_out) × dt"]
    CalcDelta --> Update["current_level += Δm"]
    Update --> Clamp{"Clamp to bounds?"}
    
    Clamp -->|Level < 0| SetZero["current_level = 0"]
    Clamp -->|Level > capacity| SetMax["current_level = capacity"]
    Clamp -->|In bounds| Skip["No change"]
    
    SetZero --> CalcP
    SetMax --> CalcP
    Skip --> CalcP
    
    CalcP["pressure = (level/capacity) × max_pressure"]
    CalcP --> End([Return])
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity_kg` | 1000.0 | Maximum storage capacity |
| `initial_level_kg` | 0.0 | Starting fill level |
| `min_level_ratio` | 0.05 | Minimum operational level (5%) |
| `max_pressure_bar` | 200.0 | Maximum pressure at full capacity |

## State Output

```mermaid
flowchart TB
    subgraph get_state["get_state() Returns"]
        level_kg["level_kg: float"]
        fill_pct["fill_percentage: float"]
        pressure["pressure_bar: float"]
    end
```

## Integration Example

```mermaid
sequenceDiagram
    participant Comp as Compressor
    participant Tank
    participant Demand as Demand Scheduler
    
    Comp->>Tank: step(flow_in=X kg/h)
    Tank-->>Tank: Update level
    Tank-->>Tank: Update pressure
    
    Demand->>Tank: step(flow_out=Y kg/h)
    Tank-->>Tank: Reduce level
    
    Note over Tank: Level = max(0, min(capacity, level + Δm))
```
