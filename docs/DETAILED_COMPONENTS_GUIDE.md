# Component Reference Guide

**Version**: 3.0  
**Updated**: December 28, 2025

This guide provides complete documentation for every component type supported by the `PlantGraphBuilder`. For each component you will find:

- **Topology Usage**: YAML syntax for including the component
- **Parameters**: All configurable parameters with types and defaults
- **Ports**: Input/output port names and resource types for connections
- **Recorded History**: Keys tracked by `engine_dispatch.py` for graphs and analysis

---

## Table of Contents

1. [Electrolysis](#1-electrolysis)
2. [Thermal Management](#2-thermal-management)
3. [Separation & Purification](#3-separation--purification)
4. [Compression](#4-compression)
5. [Water Management](#5-water-management)
6. [Flow Control](#6-flow-control)

---

## 1. Electrolysis

### 1.1 SOEC (Solid Oxide Electrolyzer Cluster)

**Class**: `SOECOperator`  
**Module**: `h2_plant.components.electrolysis.soec_operator`

#### Topology Usage

```yaml
- id: "SOEC_Cluster"
  type: "SOEC"
  params:
    num_modules: 6
    max_power_nominal_mw: 2.4
    optimal_limit: 0.80
    steam_input_ratio_kg_per_kg_h2: 10.5
    ramp_step_mw: 0.24
    process_step: 10
  connections:
    - source_port: "h2_out"
      target_name: "Interchanger_1"
      target_port: "hot_in"
      resource_type: "stream"
    - source_port: "o2_out"
      target_name: "O2_DryCooler"
      target_port: "fluid_in"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_modules` | int | 6 | Number of SOEC modules in the cluster |
| `max_power_nominal_mw` | float | 2.4 | Nominal power capacity per module (MW) |
| `optimal_limit` | float | 0.80 | Efficiency ceiling (fraction of nominal) |
| `steam_input_ratio_kg_per_kg_h2` | float | 10.5 | Steam stoichiometry |
| `ramp_step_mw` | float | 0.24 | Power change per timestep (MW) |
| `process_step` | int | 10 | Execution order priority |

#### Ports

| Port | Direction | Resource Type | Description |
|------|-----------|---------------|-------------|
| `power_in` | Input | electricity | DC power setpoint (MW) |
| `steam_in` | Input | stream | Superheated steam feed |
| `h2_out` | Output | stream | Wet hydrogen (H₂ + H₂O + trace O₂) |
| `o2_out` | Output | stream | Anode exhaust (O₂ + H₂O) |

#### Recorded History

```
P_soec_actual, H2_soec_kg, steam_soec_kg, H2O_soec_out_kg
soec_active_modules, soec_module_powers_{1..N}
```

---

### 1.2 PEM Electrolyzer

**Class**: `DetailedPEMElectrolyzer`  
**Module**: `h2_plant.components.electrolysis.pem_electrolyzer`

#### Topology Usage

```yaml
- id: "PEM_Stack_1"
  type: "PEM"
  params:
    max_power_mw: 5.0
    base_efficiency: 0.65
    use_polynomials: false
  connections:
    - source_port: "h2_out"
      target_name: "KOD_1"
      target_port: "gas_inlet"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_power_mw` | float | 5.0 | Maximum electrical input (MW) |
| `base_efficiency` | float | 0.65 | Baseline Faradaic efficiency |
| `use_polynomials` | bool | false | Use polynomial efficiency curves |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `power_in` | Input | electricity |
| `water_in` | Input | stream |
| `h2_out` | Output | stream |
| `o2_out` | Output | stream |

#### Recorded History

```
P_pem, H2_pem_kg, H2O_pem_kg, O2_pem_kg, pem_V_cell
{component_id}_o2_impurity_ppm_mol
```

---

## 2. Thermal Management

### 2.1 DryCooler

**Class**: `DryCooler`  
**Module**: `h2_plant.components.cooling.dry_cooler`

Air-cooled heat exchanger for process gas cooling. Uses ε-NTU method with approach temperature constraint.

#### Topology Usage

```yaml
- id: "DryCooler_HX"
  type: "DryCooler"
  params:
    target_temp_c: 90.0
    process_step: 30
  connections:
    - source_port: "fluid_out"
      target_name: "KOD_1"
      target_port: "gas_inlet"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_temp_c` | float | 40.0 | Target outlet temperature (°C) |
| `ambient_temp_c` | float | 25.0 | Ambient air temperature (°C) |
| `approach_temp_c` | float | 5.0 | Minimum approach ΔT (°C) |
| `fan_power_kw` | float | 10.0 | Parasitic fan power (kW) |
| `process_step` | int | — | Execution order |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `fluid_in` | Input | stream |
| `fluid_out` | Output | stream |

#### Recorded History

```
{id}_outlet_o2_ppm_mol, {id}_heat_rejected_kw, {id}_tqc_duty_kw
{id}_dc_duty_kw, {id}_fan_power_kw, {id}_outlet_temp_c
{id}_outlet_pressure_bar, {id}_outlet_h2o_frac, {id}_outlet_enthalpy_kj_kg
```

---

### 2.2 Chiller

**Class**: `Chiller`  
**Module**: `h2_plant.components.thermal.chiller`

Active refrigeration unit with COP-based electrical model. Handles phase change (condensation) via enthalpy calculations.

#### Topology Usage

```yaml
- id: "Chiller_1"
  type: "Chiller"
  params:
    target_temp_k: 277.15  # 4°C
    cooling_capacity_kw: 200.0
    cop: 4.0
    pressure_drop_bar: 0.2
    process_step: 50
  connections:
    - source_port: "fluid_out"
      target_name: "KOD_2"
      target_port: "gas_inlet"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_temp_k` | float | 298.15 | Target outlet temperature (K) |
| `cooling_capacity_kw` | float | 100.0 | Maximum cooling capacity (kW) |
| `cop` | float | 4.0 | Coefficient of Performance |
| `pressure_drop_bar` | float | 0.2 | Pressure loss across unit (bar) |
| `process_step` | int | — | Execution order |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `fluid_in` | Input | stream |
| `fluid_out` | Output | stream |

#### Recorded History

```
{id}_cooling_load_kw, {id}_electrical_power_kw, {id}_outlet_o2_ppm_mol
{id}_outlet_temp_c, {id}_outlet_pressure_bar
{id}_outlet_h2o_frac, {id}_outlet_enthalpy_kj_kg
```

---

### 2.3 ElectricBoiler

**Class**: `ElectricBoiler`  
**Module**: `h2_plant.components.thermal.electric_boiler`

Isobaric enthalpy addition device for heating liquids/gases. Automatically handles phase transitions (subcooled → two-phase → superheated).

#### Topology Usage

```yaml
- id: "Steam_Generator"
  type: "ElectricBoiler"
  params:
    max_power_kw: 3000.0
    target_temp_c: 152.0
    efficiency: 0.99
  connections:
    - source_port: "fluid_out"
      target_name: "SOEC_Cluster"
      target_port: "steam_in"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_power_kw` | float | 1000.0 | Maximum electrical input (kW) |
| `target_temp_c` | float | — | Target outlet temperature (°C) |
| `efficiency` | float | 0.99 | Thermal conversion efficiency |
| `design_pressure_bar` | float | 10.0 | Operating pressure limit (bar) |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `fluid_in` | Input | stream |
| `power_in` | Input | electricity |
| `fluid_out` | Output | stream |

---

### 2.4 Interchanger

**Class**: `Interchanger`  
**Module**: `h2_plant.components.thermal.interchanger`

Counter-flow heat exchanger for waste heat recovery. Constrained by minimum approach temperature (pinch point).

#### Topology Usage

```yaml
- id: "Interchanger_1"
  type: "Interchanger"
  params:
    min_approach_temp_k: 10.0
    target_cold_out_temp_c: 95.0
    efficiency: 0.95
  connections:
    - source_port: "hot_out"
      target_name: "DryCooler_HX"
      target_port: "fluid_in"
      resource_type: "stream"
    - source_port: "cold_out"
      target_name: "Steam_Generator"
      target_port: "water_in"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_approach_temp_k` | float | 10.0 | Minimum ΔT at pinch (K) |
| `target_cold_out_temp_c` | float | 95.0 | Cold stream target temp (°C) |
| `efficiency` | float | 0.95 | Adiabatic efficiency |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `hot_in` | Input | stream |
| `cold_in` | Input | stream |
| `hot_out` | Output | stream |
| `cold_out` | Output | stream |

---

## 3. Separation & Purification

### 3.1 KnockOutDrum

**Class**: `KnockOutDrum`  
**Module**: `h2_plant.components.separation.knock_out_drum`

Gravity-based bulk liquid separator using Souders-Brown velocity limit. Performs isoenthalpic flash calculation for VLE.

> **Design Constraint**: For high-temperature streams (e.g., SOEC exhaust at 800°C), active cooling via `DryCooler` is required **upstream**. Direct connection of superheated vapor yields zero separation efficiency.

#### Topology Usage

```yaml
- id: "KOD_1"
  type: "KnockOutDrum"
  params:
    diameter_m: 0.8
    delta_p_bar: 0.05
    gas_species: "H2"  # Optional, defaults to H2
    process_step: 40
  connections:
    - source_port: "gas_outlet"
      target_name: "Chiller_1"
      target_port: "fluid_in"
      resource_type: "stream"
    - source_port: "liquid_drain"
      target_name: "Drain_Mixer"
      target_port: "KOD_1"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diameter_m` | float | 0.5 | Vessel diameter (m) |
| `delta_p_bar` | float | 0.05 | Pressure drop (bar) |
| `gas_species` | str | "H2" | Dominant gas species |
| `process_step` | int | — | Execution order |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `gas_inlet` | Input | stream |
| `gas_outlet` | Output | stream |
| `liquid_drain` | Output | stream |

#### Recorded History

```
{id}_outlet_o2_ppm_mol, {id}_water_removed_kg_h
{id}_drain_temp_k, {id}_drain_pressure_bar, {id}_dissolved_gas_ppm
{id}_dissolved_gas_in_kg_h, {id}_dissolved_gas_out_kg_h
{id}_outlet_temp_c, {id}_outlet_pressure_bar, {id}_outlet_h2o_frac
{id}_outlet_enthalpy_kj_kg
```

---

### 3.2 HydrogenMultiCyclone

**Class**: `HydrogenMultiCyclone`  
**Module**: `h2_plant.components.separation.hydrogen_cyclone`

Centrifugal separation of fine droplets using Barth/Muschelknautz model. Euler number correlation for pressure drop.

#### Topology Usage

```yaml
- id: "Cyclone_1"
  type: "HydrogenMultiCyclone"
  params:
    element_diameter_mm: 50.0
    vane_angle_deg: 45.0
    target_velocity_ms: 22.0
    process_step: 70
  connections:
    - source_port: "outlet"
      target_name: "Compressor_S1"
      target_port: "h2_in"
      resource_type: "stream"
    - source_port: "drain"
      target_name: "Drain_Mixer"
      target_port: "Cyclone_1"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `element_diameter_mm` | float | 50.0 | Cyclone element ID (mm) |
| `vane_angle_deg` | float | 45.0 | Inlet vane angle (°) |
| `target_velocity_ms` | float | 22.0 | Design inlet velocity (m/s) |
| `process_step` | int | — | Execution order |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `inlet` | Input | stream |
| `outlet` | Output | stream |
| `drain` | Output | stream |

#### Recorded History

```
{id}_outlet_o2_ppm_mol, {id}_water_removed_kg_h
{id}_dissolved_gas_ppm, {id}_pressure_drop_mbar
{id}_outlet_temp_c, {id}_outlet_pressure_bar
{id}_outlet_h2o_frac, {id}_outlet_enthalpy_kj_kg
```

---

### 3.3 Coalescer

**Class**: `Coalescer`  
**Module**: `h2_plant.components.separation.coalescer`

Fibrous media aerosol separator. Uses Carman-Kozeny for pressure drop. Tracks dissolved gas via Henry's Law.

#### Topology Usage

```yaml
- id: "Coalescer_2"
  type: "Coalescer"
  params:
    d_shell: 0.3
    gas_type: "H2"
    process_step: 220
  connections:
    - source_port: "outlet"
      target_name: "ElectricBoiler_PSA"
      target_port: "fluid_in"
      resource_type: "stream"
    - source_port: "drain"
      target_name: "Drain_Mixer"
      target_port: "Coalescer_2"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_shell` | float | 0.3 | Shell diameter (m) |
| `gas_type` | str | "H2" | Dominant gas species |
| `process_step` | int | — | Execution order |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `inlet` | Input | stream |
| `outlet` | Output | stream |
| `drain` | Output | stream |

#### Recorded History

```
{id}_delta_p_bar, {id}_drain_flow_kg_h, {id}_outlet_o2_ppm_mol
{id}_dissolved_gas_ppm, {id}_dissolved_gas_in_kg_h, {id}_dissolved_gas_out_kg_h
{id}_outlet_temp_c, {id}_outlet_pressure_bar
{id}_outlet_h2o_frac, {id}_outlet_enthalpy_kj_kg
```

---

### 3.4 DeoxoReactor

**Class**: `DeoxoReactor`  
**Module**: `h2_plant.components.purification.deoxo_reactor`

Catalytic Plug Flow Reactor (PFR) for oxygen removal. Coupled mass/energy balance with JIT-compiled solver.

#### Topology Usage

```yaml
- id: "Deoxo_1"
  type: "DeoxoReactor"
  params:
    component_id: "Deoxo_1"
    process_step: 200
  connections:
    - source_port: "outlet"
      target_name: "Chiller_2"
      target_port: "fluid_in"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `component_id` | str | — | Unique identifier |
| `process_step` | int | — | Execution order |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `inlet` | Input | stream |
| `outlet` | Output | stream |

#### Recorded History

```
{id}_outlet_o2_ppm_mol, {id}_inlet_temp_c, {id}_inlet_pressure_bar
{id}_o2_in_kg_h, {id}_peak_temp_c, {id}_conversion_percent, {id}_mass_flow_kg_h
{id}_outlet_temp_c, {id}_outlet_pressure_bar
{id}_outlet_h2o_frac, {id}_outlet_enthalpy_kj_kg
```

---

### 3.5 PSA Unit

**Class**: `PSA`  
**Module**: `h2_plant.components.separation.psa`

Pressure Swing Adsorption for final H₂ purification (99.999%). Uses Ergun equation for bed pressure drop.

#### Topology Usage

```yaml
- id: "PSA_1"
  type: "PSA Unit"
  params:
    purity_target: 0.99995
    recovery_rate: 0.85
    process_step: 300
  connections: []  # Final product
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `purity_target` | float | 0.99995 | Product purity (fraction) |
| `recovery_rate` | float | 0.85 | H₂ recovery efficiency |
| `process_step` | int | — | Execution order |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `gas_in` | Input | stream |
| `purified_gas_out` | Output | stream |
| `tail_gas_out` | Output | stream |

#### Recorded History

```
{id}_outlet_o2_ppm_mol
```

---

## 4. Compression

### 4.1 CompressorSingle

**Class**: `CompressorSingle`  
**Module**: `h2_plant.components.compression.compressor_single`

Single-stage adiabatic compressor. Supports temperature-limited mode (auto-selects max pressure ratio to stay under T_max).

#### Topology Usage

```yaml
- id: "Compressor_S1"
  type: "CompressorSingle"
  params:
    max_flow_kg_h: 400.0
    max_temp_c: 135.0
    temperature_limited: true
    outlet_pressure_bar: 200.0  # Target (capped by temp limit)
    isentropic_efficiency: 0.75
    process_step: 100
  connections:
    - source_port: "outlet"
      target_name: "Intercooler_1"
      target_port: "fluid_in"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_flow_kg_h` | float | 400.0 | Design capacity (kg/h) |
| `max_temp_c` | float | 135.0 | Max outlet temp (°C) |
| `temperature_limited` | bool | true | Enable T-limited mode |
| `outlet_pressure_bar` | float | 200.0 | Target outlet pressure (bar) |
| `isentropic_efficiency` | float | 0.75 | η_is |
| `process_step` | int | — | Execution order |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `h2_in` | Input | stream |
| `outlet` | Output | stream |

#### Recorded History

```
{id}_outlet_o2_ppm_mol, {id}_power_kw
{id}_outlet_temp_c, {id}_outlet_pressure_bar
{id}_outlet_h2o_frac, {id}_outlet_enthalpy_kj_kg
```

---

## 5. Water Management

### 5.1 DrainRecorderMixer

**Class**: `DrainRecorderMixer`  
**Module**: `h2_plant.components.water.drain_recorder_mixer`

Collects drain streams from multiple separators. Records per-source contributions for water balance analytics.

#### Topology Usage

```yaml
- id: "Drain_Mixer"
  type: "DrainRecorderMixer"
  params:
    source_ids: ["KOD_1", "KOD_2", "Cyclone_1", "Coalescer_2"]
  connections:
    - source_port: "outlet"
      target_name: "Makeup_Mixer_1"
      target_port: "drain_in"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_ids` | list[str] | — | Port names matching upstream drain ports |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `{source_id}` | Input | stream | One per source_ids entry |
| `outlet` | Output | stream |

#### Recorded History

```
{id}_dissolved_gas_ppm, {id}_outlet_mass_flow_kg_h
{id}_outlet_temperature_c, {id}_outlet_pressure_kpa
```

---

### 5.2 MakeupMixer

**Class**: `MakeupMixer`  
**Module**: `h2_plant.components.water.makeup_mixer`

Active control node for water loop inventory. Injects fresh makeup water to maintain target flow rate.

#### Topology Usage

```yaml
- id: "Makeup_Mixer_1"
  type: "MakeupMixer"
  params:
    target_flow_kg_h: 3600.0
    makeup_temp_c: 20.0
    makeup_pressure_bar: 1.0
  connections:
    - source_port: "water_out"
      target_name: "Feed_Pump"
      target_port: "water_in"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_flow_kg_h` | float | — | Loop flow setpoint (kg/h) |
| `makeup_temp_c` | float | 20.0 | Fresh water temperature (°C) |
| `makeup_pressure_bar` | float | 1.0 | Fresh water pressure (bar) |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `drain_in` | Input | stream |
| `water_out` | Output | stream |

---

### 5.3 WaterPumpThermodynamic

**Class**: `WaterPumpThermodynamic`  
**Module**: `h2_plant.components.water.water_pump`

Pump with thermodynamic heating model. Calculates temperature rise from efficiency losses.

#### Topology Usage

```yaml
- id: "Feed_Pump"
  type: "WaterPumpThermodynamic"
  params:
    pump_id: "Feed_Pump"
    target_pressure_pa: 500000.0  # 5 bar
    eta_is: 0.82
    eta_m: 0.96
  connections:
    - source_port: "water_out"
      target_name: "Interchanger_1"
      target_port: "cold_in"
      resource_type: "stream"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pump_id` | str | — | Unique identifier |
| `target_pressure_pa` | float | — | Outlet pressure (Pa) |
| `eta_is` | float | 0.82 | Isentropic efficiency |
| `eta_m` | float | 0.96 | Mechanical efficiency |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `water_in` | Input | stream |
| `water_out` | Output | stream |

---

## 6. Flow Control

### 6.1 Valve (Throttling)

**Class**: `Valve`  
**Module**: `h2_plant.components.control.valve`

Isenthalpic expansion device (Joule-Thomson). Solves `H(P_in, T_in) = H(P_out, T_out)` via LUT bisection.

> **Note**: For H₂ at standard conditions, μ_JT < 0; expansion causes temperature **increase**.

#### Topology Usage

```yaml
- id: "O2_Vent_Valve"
  type: "Valve"
  params:
    P_out_pa: 101325.0  # Atmosphere
    fluid: "O2"
  connections: []
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `P_out_pa` | float | 101325.0 | Outlet pressure (Pa) |
| `fluid` | str | "H2" | Fluid type for LUT |

#### Ports

| Port | Direction | Resource Type |
|------|-----------|---------------|
| `inlet` | Input | stream |
| `outlet` | Output | stream |

---

## Connection Syntax

All connections in the topology follow this structure:

```yaml
connections:
  - source_port: "fluid_out"      # Port name on this component
    target_name: "Next_Component" # ID of the downstream component
    target_port: "fluid_in"       # Port name on downstream component
    resource_type: "stream"       # Resource classification
```

The `resource_type` can be: `stream`, `electricity`, `water`, `heat`.

---

## Execution Order

The `process_step` parameter controls execution order within each simulation timestep. Lower values execute first.

**Recommended ranges:**
- `1-50`: Sources and electrolyzers
- `51-100`: Primary separation
- `100-200`: Compression stages
- `200-300`: Purification
- `300+`: Final processing

---

## History Recording

The `HybridArbitrageEngineStrategy` in `engine_dispatch.py` pre-allocates NumPy arrays for all component metrics. After simulation, use:

```python
history = strategy.get_history()
# Returns Dict[str, np.ndarray] with all recorded metrics
```

To add new tracked metrics, register them in `initialize()` and populate in `record_post_step()`.
