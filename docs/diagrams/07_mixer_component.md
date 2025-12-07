# Mixer Component Diagram

Detailed architecture of the multi-component gas mixer with thermodynamic calculations.

## Component Overview

```mermaid
classDiagram
    class MultiComponentMixer {
        -float volume_m3
        -bool enable_phase_equilibrium
        -float heat_loss_coeff
        -float pressure_relief_pa
        -Dict moles_stored
        -float total_internal_energy_J
        -float temperature_k
        -float pressure_pa
        -float vapor_fraction
        -List input_buffer
        +initialize(dt, registry)
        +receive_input(port_name, value, resource_type)
        +step(t)
        +get_state()
    }
    
    class Component {
        <<abstract>>
        +initialize(dt, registry)
        +step(t)
        +get_state()
    }
    
    MultiComponentMixer --|> Component
```

## Supported Species

```mermaid
flowchart LR
    subgraph Species["Tracked Species"]
        H2["H₂<br/>Hydrogen"]
        O2["O₂<br/>Oxygen"]
        CO2["CO₂<br/>Carbon Dioxide"]
        H2O["H₂O<br/>Water"]
        CH4["CH₄<br/>Methane"]
        N2["N₂<br/>Nitrogen"]
    end
```

## Step Execution Flow

```mermaid
flowchart TD
    Start([step called]) --> ProcessBuffer["Process input buffer"]
    ProcessBuffer --> CalcEnthalpy["Calculate inlet enthalpy<br/>H_in = ṁ × h_specific"]
    CalcEnthalpy --> AccumMoles["Accumulate moles per species"]
    AccumMoles --> HeatLoss{"Heat loss enabled?"}
    
    HeatLoss -->|Yes| ApplyLoss["Q_loss = -UA × (T - T_amb)"]
    HeatLoss -->|No| Skip["Skip heat loss"]
    
    ApplyLoss --> UpdateU["U_total += H_in + Q"]
    Skip --> UpdateU
    
    UpdateU --> Flash["UV-Flash Calculation"]
    Flash --> Relief{"P > P_relief?"}
    
    Relief -->|Yes| Vent["Activate pressure relief"]
    Relief -->|No| Done
    
    Vent --> Done([End])
```

## UV-Flash Algorithm

```mermaid
flowchart TD
    subgraph Flash["UV-Flash (Internal Energy + Volume)"]
        Target["u_target = U_total / n_total"]
        Guess["Initial guess: T_current"]
        
        subgraph Iteration["Brent's Method"]
            CalcP["P = nRT/V (Ideal Gas)"]
            CalcU["u_calc = H_mix - RT"]
            Residual["residual = u_calc - u_target"]
        end
        
        Solve["Solve for T where residual = 0"]
        Result["T_solution, P_solution"]
    end
    
    Target --> Guess
    Guess --> Iteration
    Iteration --> Solve
    Solve --> Result
```

## Enthalpy Calculation

```mermaid
flowchart LR
    subgraph Species["Per Species"]
        hf["h_formation"]
        Cp["∫Cp dT from 298.15K to T"]
    end
    
    subgraph Mixture["Mixture"]
        h_mix["H_mix = Σ xᵢ × (h_f,i + ΔH_i)"]
    end
    
    Species --> Mixture
```

**Cp Integration (Shomate Equation):**
```
Cp = A + BT + CT² + DT³ + E/T²
∫Cp dT = AT + ½BT² + ⅓CT³ + ¼DT⁴ - E/T
```

## Input/Output Ports

```mermaid
flowchart LR
    subgraph Inputs
        inlet1["inlet/gas_in<br/>(Stream 1)"]
        inlet2["inlet/gas_in<br/>(Stream 2)"]
        inletN["inlet/gas_in<br/>(Stream N)"]
    end
    
    subgraph Mixer["MultiComponentMixer"]
        direction TB
        Accumulation["Mole/Energy<br/>Accumulation"]
        Flash["UV-Flash<br/>Equilibrium"]
    end
    
    subgraph State["Internal State"]
        T["Temperature (K)"]
        P["Pressure (Pa)"]
        n["Moles (per species)"]
    end
    
    inlet1 --> Mixer
    inlet2 --> Mixer
    inletN --> Mixer
    Mixer --> State
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `volume_m3` | - | Mixer vessel volume |
| `enable_phase_equilibrium` | true | Enable VLE calculations |
| `heat_loss_coeff_W_per_K` | 0.0 | UA for heat loss |
| `pressure_relief_threshold_bar` | 50.0 | Relief valve setpoint |
| `initial_temperature_k` | 298.15 | Starting temperature |

## State Output

```mermaid
flowchart TB
    subgraph get_state["get_state() Returns"]
        temp["temperature_k: float"]
        press["pressure_pa: float"]
        moles["total_moles: float"]
        vf["vapor_fraction: float"]
    end
```

## Future Enhancements

```mermaid
flowchart LR
    subgraph Planned["Future Features"]
        VLE["Rachford-Rice VLE<br/>(Vapor-Liquid Equilibrium)"]
        EOS["Peng-Robinson EOS<br/>(Non-ideal Mixing)"]
        Relief["Pressure Relief<br/>Mass Venting"]
    end
```
