# Storage Component Diagram

Detailed architecture of the Tank/Storage component for hydrogen storage.

## Component Overview

```mermaid
    class TankArray {
        -int n_tanks
        -float capacity_kg (per tank)
        -float[] masses
        -float[] pressures
        -int[] states
        +initialize(dt, registry)
        +step(t)
        +fill(mass_kg)
        +discharge(mass_kg)
        +get_state()
    }
    
    class Component {
        <<abstract>>
        +initialize(dt, registry)
        +step(t)
        +get_state()
    }
    
    TankArray --|> Component
```

## Mass Balance Model

```mermaid
flowchart LR
    subgraph Inputs
        H2In["h2_in<br/>(H₂ inflow)"]
    end
    
    subgraph Tank["TankArray (Vectorized)"]
        Level["masses[]<br/>Mass Balance"]
        Pressure["pressures[]<br/>P = f(m, V, T)"]
    end
    
    subgraph Outputs
        H2Out["h2_out<br/>(H₂ delivery)"]
    end
    
    H2In --> Level
    Level --> Pressure
    Level --> H2Out
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
    participant Tank as TankArray
    participant Demand as Demand Scheduler
    
    Comp->>Tank: receive_input("h2_in", Stream)
    Tank-->>Tank: fill(mass) [Vectorized]
    
    Tank->>Tank: step(t)
    Tank-->>Tank: update_pressures() [Numba]
    
    Demand->>Tank: get_output("h2_out")
    Tank-->>Demand: Returns buffered new mass
```
