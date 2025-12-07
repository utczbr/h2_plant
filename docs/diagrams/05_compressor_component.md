# Compressor Component Diagram

Detailed architecture of the multi-stage hydrogen compressor component.

## Component Overview

```mermaid
classDiagram
    class CompressorStorage {
        -float max_flow_kg_h
        -float inlet_pressure_bar
        -float outlet_pressure_bar
        -float inlet_temperature_c
        -float max_temperature_c
        -float isentropic_efficiency
        -float chiller_cop
        -int num_stages
        -float power_kw
        -float outlet_temp_c
        +initialize(dt, registry)
        +step(t)
        +receive_input(port_name, value, resource_type)
        +get_output(port_name)
        +get_state()
    }
    
    class Component {
        <<abstract>>
        +initialize(dt, registry)
        +step(t)
        +get_state()
    }
    
    CompressorStorage --|> Component
```

## Multi-Stage Compression Model

```mermaid
flowchart TD
    subgraph Stage1["Stage 1"]
        direction TB
        C1["Compressor<br/>T₁ → T₂"]
        IC1["Intercooler<br/>T₂ → T₁"]
        C1 --> IC1
    end
    
    subgraph Stage2["Stage 2"]
        direction TB
        C2["Compressor<br/>T₁ → T₂"]
        IC2["Intercooler<br/>T₂ → T₁"]
        C2 --> IC2
    end
    
    subgraph StageN["Stage N"]
        direction TB
        CN["Compressor<br/>T₁ → T₂"]
        ICN["Aftercooler<br/>T₂ → T₁"]
        CN --> ICN
    end
    
    Inlet["Inlet<br/>P_in, T_in"] --> Stage1
    Stage1 --> Stage2
    Stage2 --> StageN
    StageN --> Outlet["Outlet<br/>P_out, T_out"]
```

## Stage Configuration Logic

```mermaid
flowchart TD
    Start([Initialize]) --> CalcRatio["Calculate pressure ratio<br/>r = P_out / P_in"]
    CalcRatio --> Loop{For each stage count}
    
    Loop --> CalcStageR["Stage ratio = r^(1/n)"]
    CalcStageR --> CalcTout["T_out = T_in × (r_stage)^((γ-1)/γ)"]
    CalcTout --> Check{T_out > T_max?}
    
    Check -->|Yes| MoreStages["n = n + 1"]
    MoreStages --> Loop
    
    Check -->|No| Done["num_stages = n"]
    Done --> End([Configuration Complete])
```

## Input/Output Ports

```mermaid
flowchart LR
    subgraph Inputs
        gas_in["gas_in<br/>(H₂ Stream)"]
        power_in["power_in<br/>(Electricity)"]
    end
    
    subgraph Compressor["CompressorStorage"]
        direction TB
        Stages["Multi-Stage<br/>Compression"]
        Coolers["Intercoolers"]
    end
    
    subgraph Outputs
        gas_out["gas_out<br/>(Compressed H₂ Stream)"]
        heat_out["heat_out<br/>(Rejected Heat)"]
    end
    
    gas_in --> Compressor
    power_in --> Compressor
    Compressor --> gas_out
    Compressor --> heat_out
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_flow_kg_h` | - | Maximum mass flow rate |
| `inlet_pressure_bar` | 1.0 | Suction pressure |
| `outlet_pressure_bar` | 200.0 | Discharge pressure |
| `inlet_temperature_c` | 10.0 | Suction temperature |
| `max_temperature_c` | 85.0 | Max discharge temp per stage |
| `isentropic_efficiency` | 0.65 | Compression efficiency |
| `chiller_cop` | 3.0 | Intercooler COP |

## Thermodynamic Calculations

```mermaid
flowchart LR
    subgraph Inputs
        h1["h₁ = f(P₁, T₁)"]
        s1["s₁ = f(P₁, T₁)"]
    end
    
    subgraph Isentropic
        h2s["h₂ₛ = f(P₂, s₁)<br/>Isentropic outlet"]
    end
    
    subgraph Real
        ws["w_s = h₂ₛ - h₁"]
        wreal["w_real = w_s / η_is"]
        h2["h₂ = h₁ + w_real"]
    end
    
    subgraph Power
        P["P = ṁ × w_real / η_m"]
    end
    
    Inputs --> Isentropic
    Isentropic --> Real
    Real --> Power
```

## State Output

```mermaid
flowchart TB
    subgraph get_state["get_state() Returns"]
        power["power_kw: float"]
        stages["num_stages: int"]
        flow["flow_rate_kg_h: float"]
        temp_out["outlet_temp_c: float"]
        stage_work["stage_work_specific: list"]
        cooling["cooling_power_kw: float"]
    end
```
