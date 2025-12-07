# Pump Component Diagram

Detailed architecture of the water pump component with rigorous thermodynamics.

## Component Overview

```mermaid
classDiagram
    class Pump {
        -float target_pressure_pa
        -float eta_is
        -float eta_m
        -float capacity_kg_h
        -Stream outlet_stream
        -float power_kw
        -List input_buffer
        +initialize(dt, registry)
        +receive_input(port_name, value, resource_type)
        +step(t)
        +get_output(port_name)
        +get_state()
    }
    
    class Component {
        <<abstract>>
        +initialize(dt, registry)
        +step(t)
        +get_state()
    }
    
    Pump --|> Component
```

## Push Architecture Flow

```mermaid
sequenceDiagram
    participant Up as Upstream
    participant Pump
    participant Down as Downstream
    
    Up->>Pump: receive_input(inlet, Stream)
    Note over Pump: Buffer input
    
    Pump->>Pump: step(t)
    Note over Pump: Process buffer
    Note over Pump: Calculate pumping work
    Note over Pump: Create outlet stream
    
    Down->>Pump: get_output(outlet)
    Pump-->>Down: outlet_stream
```

## Step Execution Flow

```mermaid
flowchart TD
    Start([step called]) --> CheckBuffer{Input buffer empty?}
    
    CheckBuffer -->|Yes| NoFlow["power_kw = 0<br/>outlet = None"]
    CheckBuffer -->|No| Aggregate["Combine input streams"]
    
    Aggregate --> CheckPressure{"P_in >= P_target?"}
    
    CheckPressure -->|Yes| Passthrough["Pass through<br/>power_kw = 0"]
    CheckPressure -->|No| CoolProp{CoolProp available?}
    
    CoolProp -->|Yes| Rigorous["Rigorous calculation"]
    CoolProp -->|No| Simplified["Simplified hydraulic"]
    
    Rigorous --> CreateStream
    Simplified --> CreateStream["Create outlet stream"]
    Passthrough --> CreateStream
    
    NoFlow --> End([Return])
    CreateStream --> End
```

## Thermodynamic Calculation (CoolProp)

```mermaid
flowchart TD
    subgraph Inlet["Inlet State"]
        P1["P₁ (Pa)"]
        T1["T₁ (K)"]
        h1["h₁ = H(P₁, T₁)"]
        s1["s₁ = S(P₁, T₁)"]
    end
    
    subgraph Isentropic["Isentropic Process"]
        P2["P₂ (target)"]
        h2s["h₂ₛ = H(P₂, s₁)"]
    end
    
    subgraph Real["Real Process"]
        ws["w_s = h₂ₛ - h₁"]
        wreal["w_real = w_s / η_is"]
        h2["h₂ = h₁ + w_real"]
        T2["T₂ = T(P₂, h₂)"]
    end
    
    subgraph Power["Power Calculation"]
        P["P = ṁ × w_real / η_m"]
    end
    
    Inlet --> Isentropic
    Isentropic --> Real
    Real --> Power
```

## Input/Output Ports

```mermaid
flowchart LR
    subgraph Inputs
        inlet["inlet / water_in<br/>(Stream)"]
    end
    
    subgraph Pump["Pump"]
        direction TB
        Buffer["Input Buffer"]
        Calc["Pump Calculation"]
    end
    
    subgraph Outputs
        outlet["outlet / water_out<br/>(Stream @ P_target)"]
    end
    
    inlet --> Pump
    Pump --> outlet
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_pressure_bar` | - | Target outlet pressure |
| `eta_is` | 0.82 | Isentropic efficiency |
| `eta_m` | 0.96 | Mechanical efficiency |
| `capacity_kg_h` | 1000.0 | Maximum flow capacity |

## State Output

```mermaid
flowchart TB
    subgraph get_state["get_state() Returns"]
        power["power_kw: float"]
        flow["flow_rate_kg_h: float"]
        temp["outlet_temp_c: float"]
        target["target_pressure_bar: float"]
    end
```

## Simplified Fallback

When CoolProp is unavailable:

```
Hydraulic Power = V̇ × ΔP
Shaft Power = Hydraulic Power / (η_is × η_m)

where:
  V̇ = mass flow / density (ρ ≈ 1000 kg/m³)
  ΔP = P_target - P_inlet
```
