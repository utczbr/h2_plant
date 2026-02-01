# Backend Architecture Diagram

This document provides a visual representation of the backend execution flow for the Hydrogen Production System, including the Integrated Control Architecture.

---

## Simulation Execution Flow (Legacy View)

```mermaid
sequenceDiagram
    participant User
    participant Engine as SimulationEngine
    participant Scheduler as EventScheduler
    participant Registry as ComponentRegistry
    participant Components as Component[]
    participant Monitor as MonitoringSystem
    participant Metrics as MetricsCollector
    participant State as StateManager

    Note over User, State: Initialization Phase
    User->>Engine: run(start_hour, end_hour)
    Engine->>Registry: initialize_all(dt)
    Registry->>Components: initialize(dt, registry)
    Components-->>Registry: initialized
    Engine->>Monitor: initialize(registry)

    Note over User, State: Main Simulation Loop
    loop For each timestep (t)
        Engine->>Engine: _execute_timestep(t)
        
        rect rgb(240, 248, 255)
            Note right of Engine: 1. Event Processing
            Engine->>Scheduler: process_events(t)
            Scheduler->>Registry: (Trigger Component Actions)
        end

        rect rgb(255, 250, 240)
            Note right of Engine: 2. Physics & Logic Step
            Engine->>Registry: step_all(t)
            Registry->>Components: step(t)
            Components->>Components: Calculate Physics/Flows
            Components-->>Registry: step complete
        end

        rect rgb(240, 255, 240)
            Note right of Engine: 3. Monitoring & Metrics
            Engine->>Monitor: collect(t, registry)
            Engine->>Metrics: collect_step(t, states)
        end

        rect rgb(255, 240, 245)
            Note right of Engine: 4. Checkpointing (Optional)
            alt Every N hours
                Engine->>Registry: get_all_states()
                Registry->>Components: get_state()
                Components-->>Registry: state dicts
                Engine->>State: save_checkpoint(t, states)
            end
        end
    end

    Note over User, State: Finalization Phase
    Engine->>Registry: get_all_states()
    Engine->>Monitor: get_summary()
    Engine->>State: save_results(results)
    Engine->>Metrics: (Generate Graphs & Dashboard)
    Engine-->>User: Return Results Dict
```

---

## Integrated Control Architecture (New)

The Integrated Control Architecture separates **Intent** (what we want to happen) from **Outcome** (what physics allows). This enables high-frequency optimization without coupling control logic to physics calculations.

### Execution Cycle Diagram

```mermaid
sequenceDiagram
    participant Engine as SimulationEngine
    participant Dispatch as DispatchStrategy
    participant PEM as PEMElectrolyzer
    participant SOEC as SOECOperator
    participant Registry as ComponentRegistry
    participant History as NumPy Arrays

    Note over Engine, History: Per-Timestep Execution (1-minute resolution)

    rect rgb(230, 245, 255)
        Note right of Engine: Phase 1: DECIDE & APPLY (Pre-Step)
        Engine->>Dispatch: decide_and_apply(t, prices, wind)
        Dispatch->>Dispatch: Calculate optimal split (Arbitrage Logic)
        Dispatch->>PEM: receive_input('power_kw', setpoint_pem)
        Dispatch->>SOEC: receive_input('power_kw', setpoint_soec)
        Note right of Dispatch: Intent: "Use 3 MW for PEM, 2 MW for SOEC"
    end

    rect rgb(255, 245, 230)
        Note right of Engine: Phase 2: PHYSICS (Step)
        Engine->>Registry: step_all(t)
        Registry->>PEM: step(t)
        PEM->>PEM: Consume power, produce H2, update thermal state
        Registry->>SOEC: step(t)
        SOEC->>SOEC: Consume steam, produce H2
        Note right of PEM: Outcome: "Actually consumed 2.8 MW (thermal limit)"
    end

    rect rgb(245, 255, 230)
        Note right of Engine: Phase 3: RECORD (Post-Step)
        Engine->>Dispatch: record_post_step()
        Dispatch->>PEM: get_state()
        PEM-->>Dispatch: {P_consumed_W, H2_produced_kg, T_stack}
        Dispatch->>SOEC: get_state()
        SOEC-->>Dispatch: {P_consumed_W, H2_produced_kg}
        Dispatch->>History: Write actual values to pre-allocated arrays
    end
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Intent** | Power setpoints calculated by `dispatch.py` based on prices and availability. |
| **Outcome** | Actual consumption/production after physics constraints (temperature limits, ramp rates). |
| **Pre-allocation** | `HybridArbitrageEngineStrategy` creates NumPy arrays for entire simulation duration at initialization. |
| **Arbitrage Threshold** | `P_threshold = P_PPA + (1000/η) × P_H2`. Below threshold, produce H₂; above, sell to grid. |

---

## Component Lifecycle Flow

Every component follows this three-phase lifecycle:

```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> Ready: initialize(dt, registry)
    Ready --> Stepping: step(t)
    Stepping --> Ready: step complete
    Ready --> [*]: finalize()
    
    note right of Ready
        Component can:
        - receive_input()
        - get_output()
        - get_state()
    end note
    
    note right of Stepping
        Component performs:
        - Mass balance
        - Energy balance
        - State update
    end note
```

---

## Data Flow Patterns

### Stream Propagation (Push Architecture)

```mermaid
flowchart LR
    A[Producer] -->|step()| B[output_stream]
    B -->|receive_input| C[Consumer.inlet]
    C -->|step()| D[Consumer Output]
    
    style A fill:#e1f5fe
    style C fill:#fff3e0
```

### Control Flow (Pull Architecture)

```mermaid
flowchart LR
    E[Engine] -->|record_post_step| F[Dispatch]
    F -->|get_state| G[Component]
    G -->|actual_power| F
    F -->|write| H[History Arrays]
    
    style E fill:#f3e5f5
    style H fill:#e8f5e9
```

---

## Key Components

| Component | Role | File |
|-----------|------|------|
| **SimulationEngine** | Main loop, timestep management | `simulation/engine.py` |
| **ComponentRegistry** | Central component directory | `core/component_registry.py` |
| **DispatchStrategy** | Power allocation logic (Intent) | `control/dispatch.py` |
| **HybridArbitrageEngineStrategy** | Engine binding (Outcome recording) | Imported in `run_integrated_simulation.py` |
| **EventScheduler** | Time-based event injection | `simulation/event_scheduler.py` |
| **MonitoringSystem** | Real-time metrics collection | `simulation/monitoring.py` |
| **FlowNetwork** | Topology-aware flow routing | `simulation/flow_network.py` |
| **StateManager** | Checkpoint persistence | `simulation/state_manager.py` |

---

## Performance Optimizations

The Integrated Control Architecture achieves 10-50x speedup through:

1. **Pre-allocated NumPy Arrays**: History arrays created once at initialization for entire simulation duration.
2. **Vectorized Operations**: Batch calculations where possible (e.g., efficiency curves).
3. **LUT Manager**: Thermodynamic lookups via bilinear interpolation instead of CoolProp calls.
4. **Numba JIT**: Hot paths (PFR solver, flash equilibrium) compiled to machine code.

```python
# Example: Pre-allocation in HybridArbitrageEngineStrategy
def initialize(self, registry, context, total_steps):
    # Pre-allocate arrays for entire simulation duration
    self._history = {
        'minute': np.zeros(total_steps, dtype=np.int32),
        'P_soec_actual': np.zeros(total_steps, dtype=np.float32),
        'H2_soec_kg': np.zeros(total_steps, dtype=np.float32),
        'spot_price': np.zeros(total_steps, dtype=np.float32),
    }
```

