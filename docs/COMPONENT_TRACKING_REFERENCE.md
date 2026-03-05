# Component Tracking Reference

> **⚠ Document Status — Structurally Stale (March 2026)**
>
> The high-level architecture described here (component execution → state exposure → `engine_dispatch.py` recording → history DataFrame → downstream graphs) remains correct. However, several specific sections no longer reflect the live implementation:
>
> - **Visualization pipeline** — updated below; `static_graphs.py` / `GRAPH_MAP` loop has been replaced.
> - **Global Metrics table** — known to be incomplete; many fields tracked in `HybridArbitrageEngineStrategy` are not listed here (see note in that section).
> - **Per-component metric tables** — some entries do not appear in the current `CONFIG_MAP`; treat as indicative, not authoritative.
> - **Recording fallback logic** — corrected below; the old description was an oversimplification.
> - **Column Aliases** — the table in this document was **never implemented** in the current codebase and has been retracted.
>
> For the authoritative dispatch-history schema, refer directly to `HybridArbitrageEngineStrategy` in `h2_plant/control/engine_dispatch.py`. This document is retained for its debugging guidance and workflow explanations.

This document describes the metrics tracked for each component type in `engine_dispatch.py`, explains how missing properties are handled, and how this affects graph visualization.

---

## Overview: Data Flow Architecture

```
Component.step()  →  Component.get_state()  →  engine_dispatch.record_post_step()  →  History DataFrame  →  GraphOrchestrator
```

1. **Component Execution**: Each component's `step()` method executes physics calculations.
2. **State Exposure**: `get_state()` returns a dictionary of metrics.
3. **Recording**: `engine_dispatch.py` reads these metrics and stores them in pre-allocated NumPy arrays.
4. **Visualization**: `GraphOrchestrator` drives post-run graph generation via `graph_catalog.py` and `UnifiedGraphExecutor`, reading columns from the history DataFrame. The legacy `GRAPH_MAP` loop and direct calls into `static_graphs.py` have been removed.

---

## Global Metrics (System-Level)

These are tracked regardless of which components exist.

> **⚠ Incomplete table.** The fields below are a subset of the columns actually written by `HybridArbitrageEngineStrategy`. Known omissions from the live codebase include (non-exhaustive): `H2_atr_kg`, `pem_current_density`, `pem_efficiency`, `storage_*` fields, RFNBO metrics, BOP cost fields, cooling-manager fields, grid-usage fields, and per-module SOEC degradation fields. Consult `engine_dispatch.py` for the full schema.

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

> **⚠ Partially stale.** Several entries in the tables below (especially for Chiller, Coalescer, Deoxo, Knock-Out Drum, Hydrogen Multi-Cyclone, and Heat Exchanger) do not appear in the current `CONFIG_MAP` or recording logic in `engine_dispatch.py`. Fields such as `enthalpy`, `dissolved_gas_*`, `conversion_percent`, `inlet_pressure_bar`, and `tqc_duty_kw` are flagged in analysis as absent from the active recorder. Verify against `CONFIG_MAP` in `engine_dispatch.py` before relying on specific field names.

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
| `{cid}_outlet_enthalpy_kj_kg` ⚠ | Stream enthalpy | Specific enthalpy — verify against CONFIG_MAP |

### Coalescer

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_delta_p_bar` | `delta_p_bar` | Pressure drop |
| `{cid}_drain_flow_kg_h` | `drain_flow_kg_h` | Drain water flow |
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_dissolved_gas_ppm` ⚠ | `dissolved_gas_ppm` | Dissolved gas — verify against CONFIG_MAP |
| `{cid}_dissolved_gas_in_kg_h` ⚠ | `dissolved_gas_in_kg_h` | Inlet dissolved gas load — verify against CONFIG_MAP |
| `{cid}_dissolved_gas_out_kg_h` ⚠ | `dissolved_gas_out_kg_h` | Outlet dissolved gas load — verify against CONFIG_MAP |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` ⚠ | Stream enthalpy | Specific enthalpy — verify against CONFIG_MAP |

### Deoxo Reactor

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | Residual O₂ |
| `{cid}_inlet_temp_c` | `inlet_temp_c` | Inlet temperature |
| `{cid}_inlet_pressure_bar` ⚠ | `inlet_pressure_bar` | Inlet pressure — verify against CONFIG_MAP |
| `{cid}_o2_in_kg_h` | `o2_in_kg_h` | Inlet O₂ mass flow |
| `{cid}_peak_temp_c` | `peak_temp_c` | Reaction peak temp |
| `{cid}_conversion_percent` ⚠ | `conversion_percent` | O₂ conversion efficiency — verify against CONFIG_MAP |
| `{cid}_mass_flow_kg_h` | `mass_flow_kg_h` | Total mass flow |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` ⚠ | Stream enthalpy | Specific enthalpy — verify against CONFIG_MAP |

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
| `{cid}_dissolved_gas_ppm` ⚠ | `dissolved_gas_ppm` | Dissolved gas in drain — verify against CONFIG_MAP |
| `{cid}_m_dot_H2O_liq_accomp_kg_s` | `m_dot_H2O_liq_accomp_kg_s` | Entrained liquid |
| `{cid}_dissolved_gas_in_kg_h` ⚠ | `dissolved_gas_in_kg_h` | Inlet dissolved gas — verify against CONFIG_MAP |
| `{cid}_dissolved_gas_out_kg_h` ⚠ | `dissolved_gas_out_kg_h` | Outlet dissolved gas — verify against CONFIG_MAP |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` ⚠ | Stream enthalpy | Specific enthalpy — verify against CONFIG_MAP |

### Hydrogen Multi-Cyclone

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_water_removed_kg_h` | `water_removed_kg_h` | Water separation rate |
| `{cid}_drain_temp_k` | `drain_temp_k` | Drain temperature |
| `{cid}_drain_pressure_bar` | `drain_pressure_bar` | Drain pressure |
| `{cid}_dissolved_gas_ppm` ⚠ | `dissolved_gas_ppm` | Dissolved gas in drain — verify against CONFIG_MAP |
| `{cid}_dissolved_gas_in_kg_h` ⚠ | `dissolved_gas_in_kg_h` | Inlet dissolved gas — verify against CONFIG_MAP |
| `{cid}_dissolved_gas_out_kg_h` ⚠ | `dissolved_gas_out_kg_h` | Outlet dissolved gas — verify against CONFIG_MAP |
| `{cid}_pressure_drop_mbar` | `pressure_drop_mbar` | Cyclone pressure drop |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` ⚠ | Stream enthalpy | Specific enthalpy — verify against CONFIG_MAP |

### Compressor

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_power_kw` | `power_kw` | Shaft power |
| `{cid}_outlet_temp_c` | `outlet_temp_c` | Discharge temperature |
| `{cid}_outlet_pressure_bar` | Stream/state | Discharge pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` ⚠ | Stream enthalpy | Specific enthalpy — verify against CONFIG_MAP |

### Dry Cooler

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_heat_rejected_kw` | `heat_rejected_kw` | Total heat rejected |
| `{cid}_tqc_duty_kw` ⚠ | `tqc_duty_kw` | TQC section duty — verify against CONFIG_MAP |
| `{cid}_dc_duty_kw` | `dc_duty_kw` | DC section duty |
| `{cid}_fan_power_kw` | `fan_power_kw` | Fan electrical power |
| `{cid}_outlet_temp_c` | `outlet_temp_c` | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` ⚠ | Stream enthalpy | Specific enthalpy — verify against CONFIG_MAP |

### Heat Exchanger

| Metric | Source Attribute | Description |
|--------|------------------|-------------|
| `{cid}_outlet_o2_ppm_mol` | `outlet_o2_ppm_mol` | O₂ impurity |
| `{cid}_heat_removed_kw` | `heat_removed_kw` | Heat transfer rate |
| `{cid}_outlet_temp_c` | Stream temperature | Outlet temperature |
| `{cid}_outlet_pressure_bar` | Stream pressure | Outlet pressure |
| `{cid}_outlet_h2o_frac` | Stream composition | Water fraction |
| `{cid}_outlet_enthalpy_kj_kg` ⚠ | Stream enthalpy | Specific enthalpy — verify against CONFIG_MAP |

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

When `engine_dispatch.py` records a per-component metric it uses a two-stage lookup — **not** a simple `state.get()` call:

```python
# Stage 1: try direct attribute access on the component object
value = getattr(rec.component, attr_name, None)

# Stage 2: fall back to get_state() if the attribute is absent
if value is None:
    value = rec.component.get_state().get(attr_name, 0.0)
```

The practical consequence is:
- If the attribute exists directly on the component (e.g. as a property or instance variable), that value is used and `get_state()` is **not** called for this field.
- Only when `getattr` returns `None` does the recorder fall back to the state dictionary, using `0.0` as the final default.

This distinction matters when debugging: a metric that always returns `0.0` may be failing at the `getattr` stage (wrong attribute name on the object), not necessarily a missing `get_state()` key.

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
1. **Wrong attribute name**: `engine_dispatch.py` records via `getattr(component, attr_name)` first; a mismatched name silently returns `None` and falls back to 0.
2. **Missing `get_state()` key**: If `getattr` falls through to `get_state()`, the key may still be absent.
3. **Component not in topology**: The component type isn't instantiated.

**Resolution:**
1. Check `CONFIG_MAP` in `engine_dispatch.py` for the exact `attr_name` used for this field.
2. Verify the attribute name matches what the component exposes (directly as an attribute, or via `get_state()`).
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
# In record_post_step(), add to CONFIG_MAP or the component's recording block.
# The recorder tries getattr(component, 'new_metric', None) first,
# then falls back to get_state().get('new_metric', 0.0).
```

### Step 4: Register in graph_catalog.py

Add the new column to the appropriate graph entry in `graph_catalog.py`. `GraphOrchestrator` uses `UnifiedGraphExecutor` to drive all post-run graphs from this catalog — there is no longer a `static_graphs.py` function to update.

```python
# In graph_catalog.py — add column to the relevant graph's series definition
{
    "graph_id": "my_graph",
    "series": [
        {"column": "{cid}_new_metric", "label": "New Metric", "unit": "kW"}
    ]
}
```

---

## Column Naming Conventions

| Pattern | Meaning | Example |
|---------|---------|---------|
| `{cid}_metric` | Per-component metric | `KOD_1_dissolved_gas_ppm` |
| `metric` (no prefix) | System-level aggregate | `cumulative_h2_kg` |
| `soec_module_powers_{i}` | Per-module SOEC data | `soec_module_powers_1` |

> **⚠ Column Aliases — Retracted.** A previous version of this document listed a set of column aliases applied by `run_integrated_simulation.py` (e.g. `H2_soec_kg` → `H2_soec`). **These aliases are not implemented in the current codebase.** The alias table has been removed to avoid misleading debugging. If you are chasing a missing column and suspect a rename is involved, check `run_integrated_simulation.py` directly.

---

## Debugging Checklist

When a graph shows unexpected zeros:

- [ ] **Check component exists**: Is it in `plant_topology.yaml`?
- [ ] **Check detection**: Is the component type in `engine_dispatch._find_*()` or `isinstance()` checks?
- [ ] **Check allocation**: Is the history array allocated in `__init__`?
- [ ] **Check `CONFIG_MAP`**: Is the field listed with the correct `attr_name`? (`getattr` is tried first — attribute name must match the component object, not just `get_state()`.)
- [ ] **Check exposure**: Does `get_state()` return the required key (used as fallback)?
- [ ] **Check recording**: Is the value being recorded in `record_post_step()`?
- [ ] **Check catalog**: Is the column referenced correctly in `graph_catalog.py`? (Column aliases from the old pipeline do not exist.)
