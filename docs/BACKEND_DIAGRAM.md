# Backend Architecture Diagram

This document provides a visual representation of the backend execution flow for the Dual-Path Hydrogen Production System.

## Simulation Execution Flow

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

    Note over User, State: Main Simulation Loop (Hourly)
    loop For each hour (t)
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

## Key Components

1.  **SimulationEngine**: The central conductor. It manages the clock, triggers events, steps components, and handles data persistence.
2.  **ComponentRegistry**: A central directory of all system components (Electrolyzers, Tanks, etc.). It abstracts the complexity of iterating over hundreds of components.
3.  **EventScheduler**: Handles dynamic events (e.g., "Grid Power Outage at hour 500"). It injects changes into components at specific times.
4.  **Components**: The actual physical models. They implement the `step(t)` method to calculate their internal physics (mass balance, energy balance) for one timestep.
5.  **MonitoringSystem**: Observes the system. It collects time-series data (e.g., "H2 Production Rate", "Tank Level") for analysis and dashboard generation.
6.  **MetricsCollector**: Specialized data collector for the Visualization system. It feeds data into the GraphGenerator.
7.  **StateManager**: Handles saving/loading simulation state to disk, allowing for checkpoints and resuming long simulations.
