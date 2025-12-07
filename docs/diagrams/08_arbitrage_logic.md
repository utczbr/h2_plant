# Arbitration Logic Diagrams

Detailed flowcharts for the dispatch strategies that control power allocation between electrolyzers and grid sales.

## Overview

```mermaid
classDiagram
    class DispatchStrategy {
        <<abstract>>
        +decide(inputs, state) DispatchResult
    }
    
    class ReferenceHybridStrategy {
        +decide(inputs, state) DispatchResult
    }
    
    class SoecOnlyStrategy {
        +decide(inputs, state) DispatchResult
    }
    
    class DispatchInput {
        +int minute
        +float P_offer
        +float P_future_offer
        +float current_price
        +float soec_capacity_mw
        +float pem_max_power_mw
    }
    
    class DispatchResult {
        +float P_soec
        +float P_pem
        +float P_sold
        +Dict state_update
    }
    
    DispatchStrategy <|-- ReferenceHybridStrategy
    DispatchStrategy <|-- SoecOnlyStrategy
```

---

## ReferenceHybridStrategy

The main arbitration logic for hybrid SOEC+PEM systems.

### Decision Flowchart

```mermaid
flowchart TD
    Start([Dispatch Decision]) --> GetInputs["Get: P_offer, Price, P_soec_prev, force_sell"]
    GetInputs --> CalcThreshold["Calculate arbitrage threshold<br/>h2_eq = (1000/h2_kwh_kg) × H2_PRICE<br/>limit = PPA_PRICE + h2_eq"]
    
    CalcThreshold --> CheckMinute{Minute of hour = 0?}
    
    CheckMinute -->|Yes| CheckSurplus{P_offer - P_soec_prev > 0?}
    CheckMinute -->|No| Continuous
    
    CheckSurplus -->|Yes| CalcProfit["Calculate profits:<br/>sale_profit = surplus × 0.25 × (price - PPA)<br/>h2_profit = surplus × 0.25 × (1000/kwh_kg) × H2_PRICE"]
    CheckSurplus -->|No| Continuous
    
    CalcProfit --> CompareProfit{sale_profit > h2_profit?}
    CompareProfit -->|Yes| SetForce["force_sell = True"]
    CompareProfit -->|No| ClearForce["force_sell = False"]
    
    SetForce --> Continuous
    ClearForce --> Continuous
    
    Continuous["Continuous Check:<br/>if force_sell AND price <= limit:<br/>  force_sell = False"]
    
    Continuous --> CheckRamp{Minute = 45?}
    CheckRamp -->|Yes| RampCheck["Check future power<br/>if P_soec_prev > P_future:<br/>  force_sell = False"]
    CheckRamp -->|No| Dispatch
    RampCheck --> Dispatch
    
    Dispatch{force_sell?}
    Dispatch -->|Yes| SellMode
    Dispatch -->|No| NormalMode
```

### Force Sell Mode

```mermaid
flowchart LR
    ForceSell["force_sell = True"] --> Maintain["P_soec = P_soec_prev<br/>(Maintain previous)"]
    Maintain --> NoPem["P_pem = 0"]
    NoPem --> Sell["P_sold = P_offer - P_soec"]
```

### Normal Operation Mode

```mermaid
flowchart TD
    Normal["force_sell = False"] --> CalcSoec["P_soec_target = min(P_offer, SOEC_capacity)"]
    
    CalcSoec --> CheckRampDown{Minute >= 45?}
    CheckRampDown -->|Yes| RampLogic["P_soec_fut = min(P_future, capacity)<br/>if P_soec_target > P_soec_fut:<br/>  P_soec_target = P_soec_fut"]
    CheckRampDown -->|No| SetSoec
    RampLogic --> SetSoec
    
    SetSoec["P_soec = P_soec_target"]
    SetSoec --> CalcSurplus["surplus = P_offer - P_soec"]
    
    CalcSurplus --> CheckSurplus2{surplus > 0?}
    CheckSurplus2 -->|No| NoSurplus["P_pem = 0<br/>P_sold = 0"]
    CheckSurplus2 -->|Yes| CheckArbitrage{price > arbitrage_limit?}
    
    CheckArbitrage -->|Yes| SellSurplus["P_sold = surplus<br/>P_pem = 0"]
    CheckArbitrage -->|No| UsePem["P_pem = min(surplus, PEM_max)<br/>P_sold = surplus - P_pem"]
```

---

## SoecOnlyStrategy

Simplified strategy for SOEC-only topologies.

### Decision Flowchart

```mermaid
flowchart TD
    Start([SOEC-Only Decision]) --> GetInputs["Get: P_offer, Price, P_soec_prev, force_sell"]
    GetInputs --> CalcThreshold["Calculate arbitrage threshold"]
    
    CalcThreshold --> CheckMinute{Minute = 0?}
    CheckMinute -->|Yes| ArbitrageCheck["Same arbitrage logic as Hybrid"]
    CheckMinute -->|No| Continuous
    ArbitrageCheck --> Continuous
    
    Continuous --> CheckRamp{Minute = 45?}
    CheckRamp -->|Yes| RampCheck["Ramp anticipation"]
    CheckRamp -->|No| Dispatch
    RampCheck --> Dispatch
    
    Dispatch{force_sell?}
    Dispatch -->|Yes| SellMode["P_soec = P_soec_prev<br/>P_pem = 0<br/>P_sold = P_offer - P_soec"]
    Dispatch -->|No| NormalMode["P_soec = min(P_offer, capacity)<br/>P_pem = 0<br/>P_sold = surplus"]
```

---

## Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PPA_PRICE` | 50.0 EUR/MWh | Power Purchase Agreement price |
| `H2_PRICE_KG` | 9.6 EUR/kg | Hydrogen selling price |
| `SOEC_H2_KWH_KG` | 37.5 kWh/kg | SOEC energy per kg H₂ |
| `PEM_H2_KWH_KG` | 50.0 kWh/kg | PEM energy per kg H₂ |

## Arbitrage Threshold Calculation

```
h2_equivalent_price = (1000 kWh/MWh) / (h2_kwh_kg) × H2_PRICE_KG
                    = (1000 / 37.5) × 9.6
                    = 256 EUR/MWh

arbitrage_limit = PPA_PRICE + h2_equivalent_price
                = 50 + 256
                = 306 EUR/MWh
```

**Interpretation:** If spot price exceeds 306 EUR/MWh, it's more profitable to sell electricity than to produce hydrogen.

## Integration with Orchestrator

```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant Strategy as DispatchStrategy
    participant SOEC
    participant PEM
    participant Grid
    
    Orch->>Orch: Get P_offer, current_price
    Orch->>Strategy: decide(DispatchInput, DispatchState)
    Strategy-->>Orch: DispatchResult(P_soec, P_pem, P_sold)
    
    alt P_soec > 0
        Orch->>SOEC: step(P_soec)
    end
    
    alt P_pem > 0
        Orch->>PEM: set_power_input_mw(P_pem)
        Orch->>PEM: step(t)
    end
    
    Orch->>Orch: Log P_sold to history (revenue)
```
