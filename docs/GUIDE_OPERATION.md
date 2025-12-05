# Configuration & Simulation Operation Guide

**Document Version:** 1.0  
**Last Updated:** November 20, 2025  
**Target Audience:** Plant Operators, Simulation Engineers, Configuration Managers

---

## Purpose

This guide explains how to **configure, run, and manage** hydrogen production plant simulations using the zero-code configuration system and modular simulation engine[file:12][file:15][file:16][file:17]. You will learn to create plant configurations in YAML, launch simulations, manage checkpoints, and resume interrupted runs without writing Python code.

---

## Quick Start: Running Your First Simulation

### Minimal Example

```
# 1. Create a simple configuration file
cat > configs/my_plant.yaml << 'EOF'
name: "My First H2 Plant"
production:
  electrolyzer:
    max_power_mw: 2.5
    base_efficiency: 0.65
storage:
  lp_tanks:
    count: 4
    capacity_kg: 50
    pressure_bar: 30
  hp_tanks:
    count: 8
    capacity_kg: 200
    pressure_bar: 350
compression:
  filling_compressor:
    max_flow_kg_h: 100
EOF

# 2. Run the simulation
python -m h2_plant.simulation.runner configs/my_plant.yaml

# 3. Results appear in ./simulation_output/
ls simulation_output/
# simulation_results.json
# checkpoints/
# metrics/
```

**Expected Performance:** 8,760-hour simulation completes in <90 seconds on a typical workstation[file:16][file:17].

---

## Zero-Code Configuration System

### Philosophy: Declarative Over Imperative

The system uses **dataclass-based configuration** validated against JSON schemas to enable plant reconfiguration without code changes[file:12][file:15].

**Before (Hardcoded):**
```
# Requires editing Python source code
electrolyzer = HydrogenProductionSource(max_power_mw=2.5, efficiency=0.65)
hp_tanks = [Tank(capacity_kg=200.0, pressure_bar=350) for _ in range(8)]
```

**After (Configuration-Driven):**
```
# Edit YAML, no Python changes needed
production:
  electrolyzer:
    max_power_mw: 2.5
    base_efficiency: 0.65
storage:
  hp_tanks:
    count: 8
    capacity_kg: 200
    pressure_bar: 350
```

---

## Configuration Structure

### Top-Level Schema

All configurations follow this hierarchy[file:12][file:15]:

```
# Plant Metadata
name: "Plant Name"
version: "1.0"
description: "Optional description"

# System Configurations
production: {...}    # Electrolyzer and/or ATR settings
storage: {...}       # Tank arrays and isolation config
compression: {...}   # Compressor stages
demand: {...}        # Demand profile
energy_price: {...}  # Energy pricing source
pathway: {...}       # Dual-path allocation strategy
simulation: {...}    # Timestep, duration, checkpoints
```

### Configuration Sections

#### 1. Production Configuration

Define hydrogen production sources[file:12][file:15]:

```
production:
  # Grid-powered electrolysis
  electrolyzer:
    enabled: true
    max_power_mw: 5.0
    base_efficiency: 0.68
    min_load_factor: 0.15
    startup_time_hours: 0.1
  
  # Natural gas reforming (optional)
  atr:
    enabled: true
    max_ng_flow_kg_h: 100.0
    efficiency: 0.75
    reactor_temperature_k: 1200.0
    reactor_pressure_bar: 25.0
    startup_time_hours: 1.0
```

**Validation Rules:**
- At least one production source must be enabled
- Efficiency must be in (0, 1]
- Max power/flow must be positive

#### 2. Storage Configuration

Configure tank arrays with optional source isolation[file:12][file:15]:

```
storage:
  # Standard storage (no source tracking)
  lp_tanks:
    count: 4
    capacity_kg: 50.0
    pressure_bar: 30.0
  
  hp_tanks:
    count: 8
    capacity_kg: 200.0
    pressure_bar: 350.0
  
  # Source-isolated storage (tracks electrolyzer vs ATR H2)
  source_isolated: true
  isolated_config:
    electrolyzer_tanks:
      count: 4
      capacity_kg: 200.0
      pressure_bar: 350.0
    atr_tanks:
      count: 4
      capacity_kg: 200.0
      pressure_bar: 350.0
    oxygen_buffer_capacity_kg: 500.0
```

#### 3. Compression Configuration

Multi-stage compressor settings[file:12][file:15]:

```
compression:
  filling_compressor:
    max_flow_kg_h: 100.0
    inlet_pressure_bar: 30.0
    outlet_pressure_bar: 350.0
    num_stages: 3
    efficiency: 0.75
  
  outgoing_compressor:
    max_flow_kg_h: 200.0
    inlet_pressure_bar: 350.0
    outlet_pressure_bar: 900.0
    num_stages: 2
    efficiency: 0.75
```

#### 4. Demand Profile

Choose from predefined patterns or custom profiles[file:12][file:15]:

```
# Option A: Constant demand
demand:
  pattern: "constant"
  base_demand_kg_h: 50.0

# Option B: Day/night cycling
demand:
  pattern: "day_night"
  day_demand_kg_h: 80.0
  night_demand_kg_h: 20.0
  day_start_hour: 6
  night_start_hour: 22

# Option C: Custom profile from file
demand:
  pattern: "custom"
  custom_profile_file: "data/demand_profile_2025.csv"
```

#### 5. Energy Pricing

Configure grid electricity costs[file:12][file:15]:

```
# Option A: Fixed price
energy_price:
  source: "constant"
  constant_price_per_mwh: 60.0

# Option B: Hourly pricing from file
energy_price:
  source: "file"
  price_file: "data/energy_prices_2025.csv"
  constant_price_per_mwh: 60.0  # Fallback
```

**File Format (CSV):**
```
hour,price_per_mwh
0,55.2
1,52.8
2,50.1
...
```

#### 6. Pathway Coordination

Allocation strategy for dual-path plants[file:12][file:15]:

```
pathway:
  allocation_strategy: "COST_OPTIMAL"  # Options: COST_OPTIMAL, PRIORITY_GRID, PRIORITY_ATR, BALANCED
  priority_source: null  # Override: 'electrolyzer' or 'atr'
```

#### 7. External Inputs (Optional)

Configure external sources for oxygen and heat.

```yaml
external_inputs:
  oxygen_source:
    enabled: true
    mode: "fixed_flow"  # or "pressure_driven"
    flow_rate_kg_h: 50.0
    cost_per_kg: 0.15
  
  heat_source:
    enabled: true
    thermal_power_kw: 500.0
    temperature_c: 150.0
    availability_factor: 0.9
```

#### 8. Oxygen Management (Optional)

Configure a simple buffer or an advanced thermodynamic mixer.

```yaml
oxygen_management:
  use_mixer: true
  mixer:
    capacity_kg: 1000.0
    input_sources: ["electrolyzer", "external_oxygen_source"]
```

#### 9. Battery Storage (Optional)

Configure a battery energy storage system (BESS) for grid backup and load leveling.

```yaml
battery:
  enabled: true
  capacity_kwh: 1000.0
  max_charge_power_kw: 500.0
  max_discharge_power_kw: 500.0
  charge_efficiency: 0.95
  discharge_efficiency: 0.95
  min_soc: 0.20
  initial_soc: 0.50
```

#### 10. Water Treatment System (Optional)

Configure the ultrapure water treatment and distribution system.

```yaml
water_treatment:
  quality_test:
    enabled: true
    sample_interval_hours: 1.0
    
  treatment_block:
    enabled: true
    max_flow_m3h: 10.0
    power_consumption_kw: 20.0
    
  ultrapure_storage:
    capacity_l: 5000.0
    initial_fill_ratio: 0.5
      
  pumps:
    pump_a:
      enabled: true
      power_kw: 0.75
      power_source: grid_or_battery
      outlet_pressure_bar: 5.0
      
    pump_b:
      enabled: true
      power_kw: 1.5
      power_source: grid
      outlet_pressure_bar: 8.0
```

#### 11. Simulation Parameters

Control simulation execution[file:12][file:15]:

```
simulation:
  timestep_hours: 1.0
  duration_hours: 8760  # Full year
  start_hour: 0
  checkpoint_interval_hours: 168  # Weekly
```

---

## JSON Schema Validation

### Automatic Validation

All configurations are validated against `h2_plant/config/schemas/plant_schema_v1.json` during load[file:12][file:15]:

```
from h2_plant.config.loaders import ConfigLoader

loader = ConfigLoader()
try:
    config = loader.load_yaml("configs/my_plant.yaml")
    print("✓ Configuration valid")
except ConfigurationError as e:
    print(f"✗ Configuration error: {e}")
```

### Common Validation Errors

| **Error** | **Cause** | **Fix** |
|-----------|-----------|---------|
| `Schema validation failed: 'production' is a required property` | Missing production section | Add `production:` with at least one source |
| `Efficiency must be in (0, 1], got 1.5` | Invalid efficiency value | Set `base_efficiency: 0.68` (not 1.5) |
| `Tank count must be positive, got 0` | Invalid tank count | Set `count: 4` or higher |
| `isolated_config required when source_isolated=True` | Missing isolation config | Add `isolated_config:` section |

---

## The Builder Pattern: YAML → Python Objects

### How PlantBuilder Works

`PlantBuilder` converts validated YAML configurations into a fully-wired `ComponentRegistry`[file:12][file:15]:

```
graph LR
    A[YAML File] --> B[ConfigLoader]
    B --> C[PlantConfig Dataclass]
    C --> D[PlantBuilder]
    D --> E[ComponentRegistry]
    E --> F[SimulationEngine]
```

### Builder Workflow

```
from h2_plant.config.plant_builder import PlantBuilder

# 1. Load and build plant from YAML
plant = PlantBuilder.from_file("configs/plant_baseline.yaml")

# 2. Access components via registry
registry = plant.registry

# Registry is now populated with:
# - electrolyzer (if enabled)
# - atr_source (if enabled)
# - lp_tank_array
# - hp_tank_array
# - filling_compressor
# - outgoing_compressor
# - lut_manager (thermodynamics)
# - demand_scheduler
# - energy_price_tracker

# 3. Verify components
assert registry.has("electrolyzer")
assert registry.has("lp_tank_array")
print(f"Total components: {len(registry.get_all())}")
```

### Component Registration

Each configuration section maps to components[file:12][file:15]:

```
production:
  electrolyzer:
    max_power_mw: 2.5
```

**Becomes:**

```
electrolyzer = ElectrolyzerSource(
    max_power_mw=2.5,
    base_efficiency=0.65,  # From config
    ...
)
registry.register("electrolyzer", electrolyzer, component_type="production")
```

---

## Running Simulations

### Command-Line Interface

```
# Basic run
python -m h2_plant.simulation.runner configs/plant_baseline.yaml

# Specify output directory
python -m h2_plant.simulation.runner configs/plant_baseline.yaml --output-dir results/baseline_run1

# Resume from checkpoint
python -m h2_plant.simulation.runner configs/plant_baseline.yaml --resume checkpoints/checkpoint_hour_168.json
```

### Programmatic API

```
from h2_plant.simulation.runner import run_simulation_from_config

# Run simulation
results = run_simulation_from_config(
    config_path="configs/plant_baseline.yaml",
    output_dir="results/my_run"
)

# Access results
print(f"Total H2 produced: {results['metrics']['total_h2_produced_kg']:.2f} kg")
print(f"Simulation time: {results['simulation']['execution_time_seconds']:.2f}s")
```

### Multi-Scenario Comparison

```
from h2_plant.simulation.runner import run_scenario_comparison

scenarios = [
    "configs/plant_baseline.yaml",
    "configs/plant_grid_only.yaml",
    "configs/plant_high_capacity.yaml"
]

results = run_scenario_comparison(scenarios, output_dir="results/comparison")

# Results saved to:
# results/comparison/
#   plant_baseline/
#   plant_grid_only/
#   plant_high_capacity/
#   comparison_report.json
```

---

## State Management & Checkpointing

### Automatic Checkpointing

Checkpoints are saved automatically during simulation based on `checkpoint_interval_hours`[file:16][file:17]:

```
simulation:
  checkpoint_interval_hours: 168  # Save every week
```

**Checkpoint Storage:**
```
simulation_output/
└── checkpoints/
    ├── checkpoint_hour_168.json
    ├── checkpoint_hour_336.json
    ├── checkpoint_hour_504.json
    └── ...
```

### Checkpoint Formats

`StateManager` supports multiple formats[file:16][file:17]:

| **Format** | **Speed** | **Size** | **Human-Readable** | **Use Case** |
|------------|-----------|----------|-------------------|--------------|
| JSON | Moderate | Large | Yes | Debugging, inspection |
| Pickle | Fast | Small | No | Production, speed-critical |
| HDF5 | Fast | Smallest | No | Large arrays, time-series |

**Example: Force HDF5 checkpoints**

```
from h2_plant.simulation.engine import SimulationEngine

engine = SimulationEngine(registry, config)
engine.state_manager.checkpoint_format = "hdf5"
engine.run()
```

### Resume from Checkpoint

**Scenario:** Simulation crashed at hour 1500 due to power outage. Resume from last checkpoint (hour 1344).

```
from h2_plant.simulation.runner import run_simulation_from_config

results = run_simulation_from_config(
    config_path="configs/plant_baseline.yaml",
    resume_from="simulation_output/checkpoints/checkpoint_hour_1344.json"
)

# Simulation resumes from hour 1344 → 8760
print(f"Resumed from hour {results['simulation']['start_hour']}")
```

### Emergency Checkpoints

If simulation is interrupted (Ctrl+C, crash), an emergency checkpoint is saved automatically[file:16][file:17]:

```
python -m h2_plant.simulation.runner configs/plant.yaml
# ... simulation running ...
# ^C (Ctrl+C pressed)

# Output:
# WARNING: Simulation interrupted by user
# INFO: Saving emergency checkpoint at hour 2543
# Checkpoint saved to: simulation_output/checkpoints/checkpoint_hour_2543.json
```

---

## StateManager API

### Manual Checkpoint Management

```
from pathlib import Path
from h2_plant.simulation.state_manager import StateManager

manager = StateManager(output_dir=Path("simulation_output"))

# Save checkpoint manually
manager.save_checkpoint(
    hour=500,
    component_states=registry.get_all_states(),
    metadata={"note": "Pre-maintenance backup"},
    format="json"
)

# Load checkpoint
checkpoint_data = manager.load_checkpoint("simulation_output/checkpoints/checkpoint_hour_500.json")
print(f"Checkpoint from hour: {checkpoint_data['hour']}")
print(f"Timestamp: {checkpoint_data['timestamp']}")

# List all checkpoints
checkpoints = manager.list_checkpoints()
for cp in checkpoints:
    print(f"  - {cp.name}")
```

### Checkpoint Data Structure

```
{
  "hour": 168,
  "timestamp": "2025-11-20T14:30:00",
  "component_states": {
    "electrolyzer": {
      "h2_output_kg_h": 45.2,
      "power_consumption_mw": 2.3,
      "state": 1
    },
    "hp_tank_array": {
      "masses": [120.5, 180.2, 0.0, 0.0, ...],
      "states": [2, 2, 0, 0, ...],
      "total_inventory_kg": 300.7
    }
  },
  "metadata": {
    "checkpoint_type": "regular",
    "simulation_config": {
      "timestep_hours": 1.0,
      "total_duration_hours": 8760
    }
  }
}
```

---

## Simulation Lifecycle

### Complete Execution Flow

```
sequenceDiagram
    participant User
    participant Runner
    participant PlantBuilder
    participant Registry
    participant SimEngine
    participant StateManager

    User->>Runner: run_simulation_from_config("plant.yaml")
    Runner->>PlantBuilder: from_file("plant.yaml")
    PlantBuilder->>PlantBuilder: Load & validate YAML
    PlantBuilder->>Registry: Register components
    PlantBuilder-->>Runner: Plant object

    Runner->>SimEngine: SimulationEngine(registry, config)
    SimEngine->>Registry: initialize_all(dt=1.0)
    
    loop Every timestep (t=0..8759)
        SimEngine->>Registry: step_all(t)
        
        alt Checkpoint interval
            SimEngine->>Registry: get_all_states()
            SimEngine->>StateManager: save_checkpoint(t, states)
        end
    end

    SimEngine->>StateManager: save_results(final_results)
    SimEngine-->>Runner: Results dict
    Runner-->>User: Return results
```

### Execution Phases

1. **Configuration Loading** (0.1-0.5s)
   - YAML parsing and JSON schema validation
   - PlantConfig dataclass construction

2. **Plant Assembly** (0.2-1.0s)
   - Component instantiation via PlantBuilder
   - ComponentRegistry population
   - Dependency resolution

3. **Initialization** (0.5-2.0s)
   - `registry.initialize_all(dt)`
   - LUT Manager table loading
   - Component state allocation

4. **Simulation Loop** (30-90s for 8760 hours)
   - Event processing
   - Component stepping
   - Monitoring data collection
   - Periodic checkpointing

5. **Finalization** (0.5-1.0s)
   - Final state aggregation
   - Metrics export (JSON, CSV)
   - Results persistence

---

## Output Files

### Directory Structure

After a simulation run[file:16][file:17]:

```
simulation_output/
├── simulation_results.json       # Final aggregated results
├── checkpoints/                  # State snapshots
│   ├── checkpoint_hour_168.json
│   ├── checkpoint_hour_336.json
│   └── ...
├── metrics/                      # Time-series data
│   ├── production_timeseries.csv
│   ├── storage_levels.csv
│   └── energy_consumption.csv
└── logs/
    └── simulation.log            # Execution log
```

### Results JSON Schema

```
{
  "simulation": {
    "start_hour": 0,
    "end_hour": 8760,
    "duration_hours": 8760,
    "timestep_hours": 1.0,
    "execution_time_seconds": 45.3
  },
  "final_states": {
    "electrolyzer": {...},
    "hp_tank_array": {...},
    ...
  },
  "metrics": {
    "total_h2_produced_kg": 125403.5,
    "total_energy_consumed_mwh": 5234.2,
    "average_efficiency": 0.66,
    "total_cost_usd": 314052.0
  }
}
```

---

## Configuration Best Practices

### 1. Version Your Configurations

```
name: "Production Plant - Winter 2025"
version: "2.1"
description: "Updated for new electrolyzer efficiency post-maintenance"
```

Commit YAML files to version control alongside code.

### 2. Use Sensible Defaults

The system provides defaults for optional parameters[file:12][file:15]:

```
# Minimal config (uses defaults for most settings)
production:
  electrolyzer:
    max_power_mw: 2.5  # Only override what matters
```

### 3. Validate Before Running

```
from h2_plant.config.loaders import ConfigLoader

loader = ConfigLoader()
config = loader.load_yaml("configs/new_plant.yaml")
config.validate()  # Raises exception if invalid
print("✓ Configuration valid")
```

### 4. Separate Concerns

```
configs/
├── baseline/
│   ├── production.yaml        # Only production settings
│   ├── storage.yaml           # Only storage settings
│   └── full_plant.yaml        # Combines via includes
```

### 5. Document Custom Profiles

```
demand:
  pattern: "custom"
  custom_profile_file: "data/winter_2025_demand.csv"

# Add metadata in description
description: >
  Winter 2025 demand profile based on historical data.
  Peak demand: 120 kg/h at hour 1500.
  Source: operations_team@h2plant.com
```

---

## Troubleshooting

### Issue: Configuration Validation Fails

**Symptom:**
```
ConfigurationError: Schema validation failed: 'electrolyzer' is not of type 'object'
```

**Solution:** Check YAML indentation and structure:

```
# ❌ Wrong
production:
electrolyzer:
  max_power_mw: 2.5

# ✓ Correct
production:
  electrolyzer:
    max_power_mw: 2.5
```

### Issue: Simulation Hangs at Initialization

**Symptom:** "Initializing simulation engine..." with no progress.

**Cause:** LUT Manager loading large thermodynamic tables.

**Solution:** Wait 5-10 seconds on first run. Subsequent runs use cached tables.

### Issue: Checkpoint Resume Fails

**Symptom:**
```
SimulationError: Failed to resume from checkpoint: component 'electrolyzer' not found
```

**Cause:** Configuration changed between checkpoint save and resume.

**Solution:** Use the same configuration file that created the checkpoint, or disable components gracefully:

```
production:
  electrolyzer:
    enabled: false  # Disable instead of removing
```

### Issue: Out of Memory During Simulation

**Symptom:** Simulation crashes with `MemoryError` after several thousand timesteps.

**Cause:** Monitoring system accumulating too much time-series data.

**Solution:** Reduce checkpoint frequency or disable detailed metrics:

```
simulation:
  checkpoint_interval_hours: 336  # Less frequent (every 2 weeks)
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 20, 2025 | Initial operational guide for configuration and simulation lifecycle |
```

***
