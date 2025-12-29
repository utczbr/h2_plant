# Component Tracking Reference

This document describes the metrics tracked for each component type in `engine_dispatch.py`, explains how missing properties are handled, and how this affects graph visualization.

---

## Overview: Data Flow Architecture

```
Component.step()  →  Component.get_state()  →  engine_dispatch.record_post_step()  →  History DataFrame  →  Graph Functions
```

1. **Component Execution**: Each component's `step()` method executes physics calculations.
2. **State Exposure**: `get_state()` returns a dictionary of metrics.
3. **Recording**: `engine_dispatch.py` reads these metrics and stores them in pre-allocated NumPy arrays.
4. **Visualization**: Graph functions in `static_graphs.py` read columns from the history DataFrame.

---

## Global Metrics (System-Level)

These are tracked regardless of which components exist:

| Column Name | Description | Units |
|-------------|-------------|-------|
| `minute` | Simulation timestep index | - |
| `P_offer` | Available power offer (wind/grid) | MW |
| `P_soec_actual` | Actual SOEC power consumption | MW |
| `P_pem` | Actual PEM power consumption | MW |
| `P_sold` | Power sold to grid | MW |
| `spot_price` | Electricity spot price | EUR/MWh |
| `h2_kg` | Total H₂ produced this step | kg |
| `H2_soec_kg` | H₂ from SOEC this step | kg |
| `H2_pem_kg` | H₂ from PEM this step | kg |
| `cumulative_h2_kg` | Running total H₂ production | kg |
| `steam_soec_kg` | Steam consumed by SOEC | kg |
| `H2O_soec_out_kg` | Unreacted steam from SOEC | kg |
| `soec_active_modules` | Number of active SOEC modules | count |
| `H2O_pem_kg` | Water consumed by PEM | kg |
| `O2_pem_kg` | Oxygen from PEM | kg |
| `pem_V_cell` | PEM cell voltage | V |
| `P_bop_mw` | Balance of Plant power | MW |
| `tank_level_kg` | H₂ tank level | kg |
| `tank_pressure_bar` | H₂ tank pressure | bar |
| `compressor_power_kw` | Total compressor power | kW |
| `sell_decision` | Binary sell/don't sell flag | 0/1 |

---

## Per-Component Metrics

### SOEC Operator

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `soec_module_powers_{i}` | `real_powers[i]` | Power per module (1-indexed) |

### PEM Electrolyzer

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_o2_impurity_ppm_mol` | `get_state()['o2_impurity_ppm_mol']` | O₂ crossover impurity |

### Chiller

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_cooling_load_kw` | `cooling_load_kw` | Cooling duty |
| `{cid}_electrical_power_kw` | `electrical_power_kw` | Electrical consumption |
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity tracking |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` | Stream enthalpy | Specific enthalpy |

### Coalescer

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_delta_p_bar` | `delta_p_bar` | Pressure drop |
| `{cid}_drain_flow_kg_h` | `drain_flow_kg_h` | Drain water flow |
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_dissolved_gas_ppm` | `dissolved_gas_ppm` | Dissolved gas concentration |
| `{cid}_dissolved_gas_in_kg_h` | `dissolved_gas_in_kg_h` | Inlet dissolved gas load |
| `{cid}_dissolved_gas_out_kg_h` | `dissolved_gas_out_kg_h` | Outlet dissolved gas load |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` | Stream enthalpy | Specific enthalpy |

### Deoxo Reactor

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | Residual O₂ |
| `{cid}_inlet_temp_c` | `inlet_temp_c` | Inlet temperature |
| `{cid}_inlet_pressure_bar` | `inlet_pressure_bar` | Inlet pressure |
| `{cid}_o2_in_kg_h` | `o2_in_kg_h` | Inlet O₂ mass flow |
| `{cid}_peak_temp_c` | `peak_temp_c` | Reaction peak temp |
| `{cid}_conversion_percent` | `conversion_percent` | O₂ conversion efficiency |
| `{cid}_mass_flow_kg_h` | `mass_flow_kg_h` | Total mass flow |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` | Stream enthalpy | Specific enthalpy |

### PSA (Pressure Swing Adsorption)

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | Product O₂ impurity |

### Knock-Out Drum (KOD)

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_water_removed_kg_h` | `water_removed_kg_h` | Water separation rate |
| `{cid}_drain_temp_k` | `drain_temp_k` | Drain temperature |
| `{cid}_drain_pressure_bar` | `drain_pressure_bar` | Drain pressure |
| `{cid}_dissolved_gas_ppm` | `dissolved_gas_ppm` | Dissolved gas in drain |
| `{cid}_m_dot_H2O_liq_accomp_kg_s` | `m_dot_H2O_liq_accomp_kg_s` | Entrained liquid |
| `{cid}_dissolved_gas_in_kg_h` | `dissolved_gas_in_kg_h` | Inlet dissolved gas |
| `{cid}_dissolved_gas_out_kg_h` | `dissolved_gas_out_kg_h` | Outlet dissolved gas |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` | Stream enthalpy | Specific enthalpy |

### Hydrogen Multi-Cyclone

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_water_removed_kg_h` | `water_removed_kg_h` | Water separation rate |
| `{cid}_drain_temp_k` | `drain_temp_k` | Drain temperature |
| `{cid}_drain_pressure_bar` | `drain_pressure_bar` | Drain pressure |
| `{cid}_dissolved_gas_ppm` | `dissolved_gas_ppm` | Dissolved gas in drain |
| `{cid}_dissolved_gas_in_kg_h` | `dissolved_gas_in_kg_h` | Inlet dissolved gas |
| `{cid}_dissolved_gas_out_kg_h` | `dissolved_gas_out_kg_h` | Outlet dissolved gas |
| `{cid}_pressure_drop_mbar` | `pressure_drop_mbar` | Cyclone pressure drop |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` | Stream enthalpy | Specific enthalpy |

### Compressor

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_power_kw` | `power_kw` | Shaft power |
| `{cid}_outlet_temp_c` | `outlet_temp_c` | Discharge temperature |
| `{cid}_outlet_pressure_bar` | Stream/state | Discharge pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` | Stream enthalpy | Specific enthalpy |

### Dry Cooler

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_heat_rejected_kw` | `heat_rejected_kw` | Total heat rejected |
| `{cid}_tqc_duty_kw` | `tqc_duty_kw` | TQC section duty |
| `{cid}_dc_duty_kw` | `dc_duty_kw` | DC section duty |
| `{cid}_fan_power_kw` | `fan_power_kw` | Fan electrical power |
| `{cid}_outlet_temp_c` | `outlet_temp_c` | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` | Stream enthalpy | Specific enthalpy |

### Heat Exchanger

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_heat_removed_kw` | `heat_removed_kw` | Heat transfer rate |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` | Stream enthalpy | Specific enthalpy |

### Drain Recorder Mixer

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_dissolved_gas_ppm` | `dissolved_gas_ppm` | Mixed stream dissolved gas |
| `{cid}_outlet_mass_flow_kg_h` | `outlet_mass_flow_kg_h` | Total drain flow |
| `{cid}_outlet_temperature_c` | `outlet_temperature_c` | Mixed temperature |
| `{cid}_outlet_pressure_kpa` | `outlet_pressure_kpa` | Mixed pressure |

---

## What Happens When a Property is Missing?

### Recording Behavior

When `engine_dispatch.py` records a metric, it uses the pattern:

```python
self._history[f"{cid}_metric"][step_idx] = state.get('metric', 0.0)
```

The `state.get('metric', 0.0)` pattern means:
- **If the key exists**: The actual value is recorded.
- **If the key is missing**: `0.0` is recorded as the default.

### Pre-Allocation Behavior

All history arrays are pre-allocated with zeros:

```python
self._history[f"{cid}_metric"] = np.zeros(total_steps, dtype=np.float64)
```

This means:
- **If a component doesn't exist in the topology**: No array is allocated for it.
- **If a component exists but doesn't expose a metric**: The array contains zeros.

### Impact on Graphs

| Scenario | Array State | Graph Behavior |
|----------|-------------|----------------|
| Component missing from topology | No column in DataFrame | Graph shows "No data" or skips the series |
| Component exists, metric exposed | Actual values | Normal graph display |
| Component exists, metric NOT exposed | All zeros | Graph shows flat line at 0 |
| Transient issue (one timestep fails) | Single 0 in data | Spike/dip to 0 in graph |

---

## Common "Zero Data" Issues and Resolutions

### Issue: Graph shows constant 0

**Possible Causes:**
1. **Missing `get_state()` key**: Component doesn't expose the required metric.
2. **Wrong attribute name**: `engine_dispatch.py` reads wrong attribute (e.g., `last_steam_output_kg` vs `last_step_steam_input_kg`).
3. **Component not in topology**: The component type isn't instantiated.

**Resolution:**
1. Check component's `get_state()` method for the expected key.
2. Verify `engine_dispatch.py` uses the correct attribute name.
3. Confirm the component exists in `plant_topology.yaml`.

### Issue: Graph shows "No data available"

**Possible Causes:**
1. **Column not in DataFrame**: The history array was never allocated.
2. **Component list empty**: No instances of that component type exist.

**Resolution:**
1. Verify the component type is imported and detected in `engine_dispatch.__init__()`.
2. Check if the component is defined in the topology.

### Issue: Partial data (some timesteps missing)

**Possible Causes:**
1. **Component inactive**: Component only operates under certain conditions.
2. **Stream not connected**: Upstream component not providing input.

**Resolution:**
This is often expected behavior (e.g., PEM only runs when SOEC is at capacity).

---

## Adding New Tracked Metrics

To track a new metric for a component:

### Step 1: Expose in Component

```python
# In component's get_state() method
def get_state(self) -> Dict[str, Any]:
    state = super().get_state()
    state['new_metric'] = self.calculated_value
    return state
```

### Step 2: Allocate in engine_dispatch.py

```python
# In __init__, within the component loop
for comp in self._components:
    cid = comp.component_id
    self._history[f"{cid}_new_metric"] = np.zeros(total_steps, dtype=np.float64)
```

### Step 3: Record in engine_dispatch.py

```python
# In record_post_step(), within the component loop
for comp in self._components:
    cid = comp.component_id
    state = comp.get_state()
    self._history[f"{cid}_new_metric"][step_idx] = state.get('new_metric', 0.0)
```

### Step 4: Use in Graph

```python
# In static_graphs.py graph function
metric_col = _find_component_columns(df, 'ComponentID', 'new_metric')
if metric_col:
    ax.plot(df['minute'], df[metric_col], label='New Metric')
```

---

## Column Naming Conventions

| Pattern | Meaning | Example |
|---------|---------|---------|
| `{cid}_metric` | Per-component metric | `KOD_1_dissolved_gas_ppm` |
| `metric` (no prefix) | System-level aggregate | `cumulative_h2_kg` |
| `soec_module_powers_{i}` | Per-module SOEC data | `soec_module_powers_1` |

### Column Aliases

`run_integrated_simulation.py` applies aliases for legacy graph compatibility:

| Original Name | Alias |
|---------------|-------|
| `H2_soec_kg` | `H2_soec` |
| `H2_pem_kg` | `H2_pem` |
| `steam_soec_kg` | `Steam_soec` |
| `H2O_pem_kg` | `H2O_pem` |
| `spot_price` | `Spot` |

---

## Debugging Checklist

When a graph shows unexpected zeros:

- [ ] **Check component exists**: Is it in `plant_topology.yaml`?
- [ ] **Check detection**: Is the component type in `engine_dispatch._find_*()` or `isinstance()` checks?
- [ ] **Check allocation**: Is the history array allocated in `__init__`?
- [ ] **Check exposure**: Does `get_state()` return the required key?
- [ ] **Check attribute name**: Does `engine_dispatch.py` use the correct attribute/key name?
- [ ] **Check recording**: Is the value being recorded in `record_post_step()`?
- [ ] **Check aliases**: Is there a column alias defined for graph compatibility?
