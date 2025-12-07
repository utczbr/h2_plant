# PEM Electrolyzer Component Diagram

Detailed architecture of the Proton Exchange Membrane (PEM) Electrolyzer component.

## Component Overview

```mermaid
classDiagram
    class DetailedPEMElectrolyzer {
        -int n_cells
        -float A_cell_m2
        -float max_power_mw
        -float T_op_C
        -float operating_hours
        -float V_cell
        -float h2_output_kg
        -float water_consumption_kg
        +initialize(dt, registry)
        +step(t)
        +set_power_input_mw(P_mw)
        +shutdown()
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
    
    DetailedPEMElectrolyzer --|> Component
    
    class RecirculationPump {
        +flow_rate_kg_h
    }
    
    class HeatExchanger {
        +inlet_flow_kg_h
    }
    
    DetailedPEMElectrolyzer ..> RecirculationPump : uses
    DetailedPEMElectrolyzer ..> HeatExchanger : uses
```

## Electrochemistry Model

```mermaid
flowchart TD
    subgraph Voltage["Cell Voltage Calculation"]
        Urev["U_rev (Nernst)<br/>1.23V base"]
        eta_act["η_act (Activation)<br/>Tafel equation"]
        eta_ohm["η_ohm (Ohmic)<br/>Membrane resistance"]
        eta_conc["η_conc (Concentration)<br/>Mass transport"]
        U_deg["U_deg (Degradation)<br/>Time-based"]
        
        Urev --> Ucell
        eta_act --> Ucell
        eta_ohm --> Ucell
        eta_conc --> Ucell
        U_deg --> Ucell["U_cell = U_rev + η_act + η_ohm + η_conc + U_deg"]
    end
    
    subgraph Production["H₂ Production"]
        j_op["Current Density<br/>j (A/cm²)"]
        Faraday["Faraday's Law<br/>n_H2 = j·A / (2F)"]
        eta_F["Faradaic Efficiency<br/>η_F ≈ 0.99"]
        
        j_op --> Faraday
        Faraday --> H2out["H₂ Output (kg/h)"]
        eta_F --> H2out
    end
```

## Step Execution Flow

```mermaid
flowchart TD
    Start([step called]) --> GetPower{P_input > 0?}
    
    GetPower -->|No| Idle[H₂ = 0, O₂ = 0]
    GetPower -->|Yes| CalcJ[Solve for current density j]
    
    CalcJ --> CalcV[Calculate V_cell]
    CalcV --> CalcPower[P_consumed = V_cell × j × A × n_cells]
    
    CalcPower --> CalcH2["H₂ = j × A × n_cells / (2F) × M_H2 × η_F"]
    CalcH2 --> CalcO2["O₂ = H₂ × 8 (stoichiometric)"]
    CalcO2 --> CalcH2O["H₂O = H₂ × 9 × 1.02 (excess)"]
    
    CalcH2O --> UpdateDeg[Update degradation hours]
    UpdateDeg --> UpdateState[Store outputs]
    
    Idle --> End([End])
    UpdateState --> End
```

## Input/Output Ports

```mermaid
flowchart LR
    subgraph Inputs
        power_in["power_in<br/>(MW)"]
        water_in["water_in<br/>(kg/h)"]
    end
    
    subgraph PEM["DetailedPEMElectrolyzer"]
        direction TB
        Stack["Cell Stack<br/>(n_cells × A_cell)"]
        BoP["Balance of Plant<br/>(Pumps, HX)"]
    end
    
    subgraph Outputs
        h2_out["h2_out<br/>(Stream - kg/h)"]
        o2_out["o2_out<br/>(Stream - kg/h)"]
        heat_out["heat_out<br/>(kW)"]
    end
    
    power_in --> PEM
    water_in --> PEM
    PEM --> h2_out
    PEM --> o2_out
    PEM --> heat_out
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_stacks` | 35 | Number of stacks |
| `N_cell_per_stack` | 85 | Cells per stack |
| `A_cell` | 300 cm² | Active cell area |
| `max_power_mw` | 5.0 | Maximum input power |
| `T_op` | 333.15 K (60°C) | Operating temperature |
| `P_op` | 40 bar | Operating pressure |


## Degradation Model

```mermaid
flowchart LR
    Hours["Operating Hours<br/>t_op"] --> Poly["Polynomial Model<br/>ΔV = f(t_op)"]
    Poly --> DeltaV["Voltage Increase<br/>ΔV (mV)"]
    DeltaV --> Efficiency["↓ Efficiency<br/>↑ Power Demand"]
```

| Time (hours) | Degradation (mV) |
|--------------|------------------|
| 0 | 0 |
| 5,000 | ~5 |
| 10,000 | ~15 |
| 20,000 | ~40 |

## Integration with Orchestrator

```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant Dispatch as DispatchStrategy
    participant PEM as DetailedPEMElectrolyzer
    
    Orch->>Dispatch: Get P_pem setpoint
    Dispatch-->>Orch: P_pem = X MW
    
    Orch->>PEM: set_power_input_mw(P_pem)
    Orch->>PEM: step(t)
    PEM-->>PEM: Solve j, V_cell
    PEM-->>PEM: Calculate H₂, O₂, H₂O
    
    Orch->>PEM: Access h2_output_kg
    Orch->>Orch: Log to history
```
