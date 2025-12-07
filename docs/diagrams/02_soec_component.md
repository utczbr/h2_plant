# SOEC Electrolyzer Component Diagram

Detailed architecture of the Solid Oxide Electrolyzer Cell (SOEC) Operator component.

## Component Overview

```mermaid
classDiagram
    class SOECOperator {
        -int num_modules
        -float max_power_nominal_mw
        -float optimal_limit
        -float h2_kwh_kg
        -ndarray real_powers
        -ndarray real_states
        -ndarray degradation
        +initialize(dt, registry)
        +step(reference_power_mw, t)
        +receive_input(port_name, value, resource_type)
        +get_output(port_name)
        +get_ports()
        +get_state()
        +get_status()
    }
    
    class Component {
        <<abstract>>
        +initialize(dt, registry)
        +step(t)
        +get_state()
    }
    
    SOECOperator --|> Component
```

## Module State Machine

```mermaid
stateDiagram-v2
    [*] --> OFF: Initial
    OFF --> RAMPING: Power > 0
    RAMPING --> NOMINAL: Ramp Complete
    NOMINAL --> RAMPING: Power Change
    NOMINAL --> OFF: Power = 0
    RAMPING --> OFF: Power = 0
    
    note right of NOMINAL
        Active production
        Degradation accumulates
    end note
    
    note right of RAMPING
        15-minute transition
        Gradual power change
    end note
```

## Step Execution Flow

```mermaid
flowchart TD
    Start([step called]) --> CheckPower{reference_power > 0?}
    
    CheckPower -->|No| Shutdown[Set all modules OFF]
    CheckPower -->|Yes| Allocate[Allocate power to modules]
    
    Allocate --> Loop{For each module}
    Loop --> CalcPower[Calculate module power]
    CalcPower --> UpdateState[Update module state]
    UpdateState --> CheckRamp{Ramping?}
    
    CheckRamp -->|Yes| ApplyRamp[Apply 15-min ramp rate]
    CheckRamp -->|No| Direct[Use direct setpoint]
    
    ApplyRamp --> Degrade
    Direct --> Degrade[Apply degradation factor]
    
    Degrade --> CalcH2[Calculate H₂ production]
    CalcH2 --> CalcSteam[Calculate steam consumption]
    
    CalcSteam --> Loop
    Loop -->|Done| Aggregate[Aggregate totals]
    
    Aggregate --> Output[Return: P_actual, H₂_kg, Steam_kg]
    Shutdown --> Output
    Output --> End([End])
```

## Input/Output Ports

```mermaid
flowchart LR
    subgraph Inputs
        power_in["power_in<br/>(MW)"]
        steam_in["water_in/steam_in<br/>(kg/h)"]
    end
    
    subgraph SOEC["SOECOperator"]
        direction TB
        Modules["6 Modules<br/>(State + Degradation)"]
    end
    
    subgraph Outputs
        h2_out["h2_out<br/>(Stream - kg/h)"]
        water_out["water_out<br/>(kg/h)"]
    end
    
    power_in --> SOEC
    steam_in --> SOEC
    SOEC --> h2_out
    SOEC --> water_out
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_modules` | 6 | Number of SOEC modules |
| `max_power_nominal_mw` | 8.0 | Max power per module (MW) |
| `optimal_limit` | 0.8 | Operating limit factor |
| `h2_kwh_kg` | 37.5 | Energy per kg H₂ produced |
| `ramp_time_min` | 15 | Time to ramp up/down |

## Integration with Orchestrator

```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant Dispatch as DispatchStrategy
    participant SOEC as SOECOperator
    participant Down as Downstream (Chiller/Tank)
    
    Orch->>Dispatch: Get P_soec setpoint
    Dispatch-->>Orch: P_soec = X MW
    
    Orch->>SOEC: step(P_soec)
    SOEC-->>SOEC: Allocate to modules
    SOEC-->>SOEC: Apply ramp/degradation
    SOEC-->>Orch: (P_actual, H₂_kg, Steam_kg)
    
    Orch->>SOEC: get_output("h2_out")
    SOEC-->>Orch: H₂ Stream
    Orch->>Down: _step_downstream(h2_stream)
```
