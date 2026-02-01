# Visualization System Gap Analysis Report

**Generated:** 2025-12-27  
**Scope:** Cross-reference of Configuration, Catalog, Orchestrator, and Implementation files

---

## Section 1: Migration Gap Matrix

### 1.1 Orchestrator Handler Coverage (visualization_config.yaml → graph_orchestrator.py)

| Config Graph ID | Orchestrator Handler | Status | Notes |
|-----------------|---------------------|--------|-------|
| `production_time_series` | `production.plot_time_series` | ✅ Migrated | |
| [production_stacked](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#135-176) | `production.plot_stacked` | ✅ Migrated | |
| `production_cumulative` | `production.plot_cumulative` | ✅ Migrated | |
| `performance_time_series` | `performance.plot_time_series` | ✅ Migrated | |
| `performance_scatter` | `performance.plot_scatter` | ✅ Migrated | |
| `economics_time_series` | `economics.plot_time_series` | ✅ Migrated | |
| `dispatch_stack` | `economics.plot_dispatch` | ✅ Migrated | |
| `economics_pie` | `economics.plot_pie` | ✅ Migrated | |
| `economics_scatter` | `economics.plot_arbitrage` | ✅ Migrated | |
| `soec_modules_time_series` | `soec.plot_active_modules` | ✅ Migrated | |
| [soec_heatmap](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#740-768) | `soec.plot_module_heatmap` | ✅ Migrated | |
| `soec_stats` | `soec.plot_module_stats` | ✅ Migrated | |
| `storage_levels` | `storage.plot_tank_levels` | ✅ Migrated | |
| [compressor_power](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/storage.py#49-86) | `storage.plot_compressor_power` | ✅ Migrated | |
| [thermal_load_breakdown](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1903-1983) | `thermal.plot_load_breakdown` | ✅ Migrated | |
| `water_removal_bar` | `separation.plot_water_removal` | ✅ Migrated | |
| [process_train_profile](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#3373-3464) | `profiles.plot_profile` | ✅ Migrated | |

### 1.2 Catalog Registry vs Implementation (graph_catalog.py → static_graphs.py/plotly_graphs.py)

| Catalog Graph ID | Function Reference | Implementation File | Status |
|-----------------|-------------------|---------------------|--------|
| `pem_h2_production_over_time` | `pg.plot_pem_production_timeline` | plotly_graphs.py:66 | ✅ Exists |
| `soec_h2_production_over_time` | `pg.plot_soec_production_timeline` | plotly_graphs.py:105 | ✅ Exists |
| `total_h2_production_stacked` | `pg.plot_total_production_stacked` | plotly_graphs.py:135 | ✅ Exists |
| `cumulative_h2_production` | `pg.plot_cumulative_production` | plotly_graphs.py:178 | ✅ Exists |
| `pem_cell_voltage_over_time` | `pg.plot_pem_voltage_timeline` | plotly_graphs.py:227 | ✅ Exists |
| `pem_efficiency_over_time` | `pg.plot_pem_efficiency_timeline` | plotly_graphs.py:262 | ✅ Exists |
| `energy_price_over_time` | `pg.plot_energy_price_timeline` | plotly_graphs.py:291 | ✅ Exists |
| `dispatch_strategy_stacked` | `pg.plot_dispatch_strategy` | plotly_graphs.py:324 | ✅ Exists |
| `power_consumption_breakdown_pie` | `pg.plot_power_breakdown_pie` | plotly_graphs.py:391 | ✅ Exists |
| `soec_active_modules_over_time` | `pg.plot_soec_modules_timeline` | plotly_graphs.py:415 | ✅ Exists |
| [storage_fatigue_cycling_3d](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#467-505) | `pg.plot_storage_fatigue_cycling_3d` | plotly_graphs.py:467 | ✅ Exists |
| [ramp_rate_stress_distribution](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#507-533) | `pg.plot_ramp_rate_stress_distribution` | plotly_graphs.py:507 | ✅ Exists |
| [wind_utilization_duration_curve](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#535-574) | `pg.plot_wind_utilization_duration_curve` | plotly_graphs.py:535 | ✅ Exists |
| [grid_interaction_phase_portrait](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#576-607) | `pg.plot_grid_interaction_phase_portrait` | plotly_graphs.py:576 | ✅ Exists |
| [lcoh_waterfall_breakdown](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#609-641) | `pg.plot_lcoh_waterfall_breakdown` | plotly_graphs.py:609 | ⚠️ Placeholder (no data) |
| [pem_performance_surface](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#643-684) | `pg.plot_pem_performance_surface` | plotly_graphs.py:643 | ✅ Exists |
| [tank_storage_timeline](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#446-465) | `pg.plot_tank_storage_timeline` | plotly_graphs.py:446 | ⚠️ Disabled by default |
| [dispatch](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/economics.py#13-65) | `sg.create_dispatch_figure` | static_graphs.py:223 | ✅ Exists |
| [arbitrage](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/economics.py#123-146) | `sg.create_arbitrage_figure` | static_graphs.py:253 | ✅ Exists |
| [h2_production](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#280-300) | `sg.create_h2_production_figure` | static_graphs.py:280 | ✅ Exists |
| `oxygen_production` | `sg.create_oxygen_figure` | static_graphs.py:301 | ✅ Exists |
| `water_consumption` | `sg.create_water_figure` | static_graphs.py:325 | ✅ Exists |
| [energy_pie](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#349-375) | `sg.create_energy_pie_figure` | static_graphs.py:349 | ✅ Exists |
| `price_histogram` | `sg.create_histogram_figure` | static_graphs.py:376 | ✅ Exists |
| [dispatch_curve](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#394-407) | `sg.create_dispatch_curve_figure` | static_graphs.py:394 | ✅ Exists |
| [cumulative_h2](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#408-432) | `sg.create_cumulative_h2_figure` | static_graphs.py:408 | ✅ Exists |
| [cumulative_energy](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#433-460) | `sg.create_cumulative_energy_figure` | static_graphs.py:433 | ✅ Exists |
| [efficiency_curve](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#461-490) | `sg.create_efficiency_curve_figure` | static_graphs.py:461 | ✅ Exists |
| [revenue_analysis](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#491-519) | `sg.create_revenue_analysis_figure` | static_graphs.py:491 | ✅ Exists |
| [temporal_averages](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#520-559) | `sg.create_temporal_averages_figure` | static_graphs.py:520 | ✅ Exists |
| [chiller_cooling](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#3163-3209) | `sg.create_chiller_cooling_figure` | static_graphs.py | ✅ Exists |
| [coalescer_separation](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#3211-3254) | `sg.create_coalescer_separation_figure` | static_graphs.py | ✅ Exists |
| [kod_separation](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#3256-3310) | `sg.create_kod_separation_figure` | static_graphs.py | ✅ Exists |
| `dry_cooler_performance` | `sg.create_dry_cooler_figure` | static_graphs.py | ✅ Exists |
| [water_removal_total](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1836-1923) | `sg.create_water_removal_total_figure` | static_graphs.py:1836 | ✅ Exists |
| [drains_discarded](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1924-1980) | `sg.create_drains_discarded_figure` | static_graphs.py:1924 | ✅ Exists |
| [individual_drains](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#2296-2368) | `sg.create_individual_drains_figure` | static_graphs.py:2296 | ✅ Exists |
| `dissolved_gas_concentration` | `sg.create_dissolved_gas_figure` | static_graphs.py:2369 | ✅ Exists |
| `dissolved_gas_efficiency` | `sg.create_drain_concentration_figure` | static_graphs.py:2412 | ✅ Exists |
| [crossover_impurities](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#2554-2760) | `sg.create_crossover_impurities_figure` | static_graphs.py:2554 | ✅ Exists |
| [process_train_profile](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#3373-3464) | `sg.create_process_train_profile_figure` | static_graphs.py | ✅ Exists |
| [energy_flows](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1986-2061) | `sg.create_energy_flows_figure` | static_graphs.py:1986 | ✅ Exists |
| [plant_balance](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#2063-2170) | `sg.create_plant_balance_schematic` | static_graphs.py:2063 | ✅ Exists |
| [q_breakdown](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#911-1037) | `sg.create_thermal_load_breakdown_figure` | static_graphs.py | ✅ Exists |
| [mixer_comparison](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#2171-2259) | `sg.create_mixer_comparison_figure` | static_graphs.py:2171 | ✅ Exists |
| [deoxo_profile](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#2761-2816) | `sg.create_deoxo_profile_figure` | static_graphs.py:2761 | ✅ Exists |
| [drain_line_properties](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1038-1177) | `sg.create_drain_line_properties_figure` | static_graphs.py:1038 | ✅ Exists |
| `drain_line_concentration` | `sg.create_drain_concentration_figure` | static_graphs.py:2412 | ⚠️ Duplicate ID |
| [recirculation_comparison](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#3006-3106) | `sg.create_recirculation_comparison_figure` | static_graphs.py | ✅ Exists |
| `entrained_liquid_flow` | `sg.create_entrained_liquid_figure` | static_graphs.py | ✅ Exists |
| [water_vapor_tracking](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#3466-3533) | `sg.create_water_vapor_tracking_figure` | static_graphs.py | ✅ Exists |
| [total_mass_flow](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#3535-3610) | `sg.create_total_mass_flow_figure` | static_graphs.py:800 | ✅ Exists |

### 1.3 Orphaned Implementations (Not Registered in Catalog)

The following functions exist in [static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py) but are **NOT registered** in [graph_catalog.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graph_catalog.py):

| Function Name | Location | Status |
|--------------|----------|--------|
| [create_monthly_performance_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#629-655) | static_graphs.py:629 | ❌ Not Registered |
| [create_monthly_efficiency_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#656-670) | static_graphs.py:656 | ❌ Not Registered |
| [create_monthly_capacity_factor_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#671-685) | static_graphs.py:671 | ❌ Not Registered |
| [create_soec_module_heatmap_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#686-705) | static_graphs.py:686 | ❌ Not Registered |
| [create_soec_module_power_stacked_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#769-834) | static_graphs.py:769 | ❌ Not Registered |
| [create_soec_module_wear_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#835-910) | static_graphs.py:835 | ❌ Not Registered |
| [create_q_breakdown_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#911-1037) | static_graphs.py:911 | Partial (registered as q_breakdown) |
| [create_drain_scheme_schematic](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1433-1572) | static_graphs.py:1433 | ❌ Not Registered |
| [create_energy_flow_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1573-1713) | static_graphs.py:1573 | ❌ Not Registered |
| [create_process_scheme_schematic](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1714-1832) | static_graphs.py:1714 | ❌ Not Registered |
| [create_drain_mixer_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#1313-1432) | static_graphs.py:1313 | ❌ Not Registered |

---

## Section 2: Data Dependency Failures

### 2.1 MetricsCollector Coverage Analysis

**Metrics Collected by [collect_step()](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/metrics_collector.py#100-214):**

| Category | Metric | Collected? |
|----------|--------|-----------|
| pem | h2_production_kg_h | ✅ |
| pem | voltage | ✅ |
| pem | efficiency | ✅ |
| pem | power_mw | ✅ |
| pem | cumulative_h2_kg | ✅ |
| pem | cumulative_energy_kwh | ✅ |
| soec | h2_production_kg_h | ✅ |
| soec | module_states | ⚠️ Declared but NOT collected |
| soec | active_modules | ✅ |
| soec | power_mw | ✅ |
| soec | ramp_rates | ✅ |
| tanks | lp_masses | ✅ (2D array) |
| tanks | hp_masses | ✅ (2D array) |
| tanks | hp_pressures | ✅ (2D array) |
| tanks | total_stored | ✅ |
| pricing | energy_price_eur_kwh | ✅ |
| pricing | wind_coefficient | ✅ |
| pricing | grid_exchange_mw | ✅ |
| economics | lcoh_cumulative | ⚠️ Declared but NOT collected |
| coordinator | pem_setpoint_mw | ✅ |
| coordinator | soec_setpoint_mw | ✅ |
| coordinator | sell_power_mw | ✅ |

### 2.2 Graphs with Missing Data Dependencies (Partially Resolved in Phase 7)
*Phase 7 Remediation: Updated static_graphs.py to prioritize sensor data over proxies for O2 and Water.*

| Graph ID | Required Data | MetricsCollector Status | Actual Data Source | Status |
|----------|---------------|------------------------|--------------------|--------|
| [lcoh_waterfall_breakdown](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#609-641) | `economics.lcoh_cumulative` | ❌ NOT COLLECTED | Placeholder only | Open |
| [storage_fatigue_cycling_3d](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py#467-505) | `tanks.hp_pressures` | ⚠️ 2D Array | Needs flattening logic | Open |
| [soec_heatmap](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#740-768) | `soec.module_states` | ❌ NOT COLLECTED | Uses `soec_module_powers_*` columns from history | Open |
| `oxygen_production` (Legacy) | `soec.o2_production`, `pem.o2_production` | ✅ | `o2_production` or `o2_out` columns (if available) | **Resolved** |
| `water_consumption` (Legacy) | `Water_Source.mass_flow` | ✅ | `Water_Source` columns (if available) | **Resolved** |

### 2.3 Data Source Mismatch

**CRITICAL:** The [graph_orchestrator.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graph_orchestrator.py) and `graphs/*.py` modules read data from **simulation history DataFrame** (from [run_integrated_simulation.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/run_integrated_simulation.py)), NOT from [MetricsCollector](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/metrics_collector.py#34-323).

The [MetricsCollector](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/metrics_collector.py#34-323) is a separate collection mechanism that is **NOT INTEGRATED** with the main simulation loop in [run_integrated_simulation.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/run_integrated_simulation.py).

**Implication:** All graphs in the Orchestrator architecture must rely on column names from [engine_dispatch.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/control/engine_dispatch.py) history recording, NOT [MetricsCollector](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/metrics_collector.py#34-323) structure.

---

## Section 3: Code Improvement Recommendations

### 3.1 Hardcoded Component IDs (Topology Coupling)

**Location:** [static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py), [graphs/production.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py), [graphs/profiles.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/profiles.py)

**Problem:** Multiple functions contain string literals like `"PEM"`, `"SOEC"`, `"KOD_1"`, `"H2_pem_kg"`.

**Affected Functions:**

| Function | Hardcoded IDs | Risk Level |
|----------|--------------|------------|
| `production.plot_time_series` | `"PEM"`, `"SOEC"`, `"H2_pem_kg"`, `"H2_soec_kg"` | HIGH |
| `production.plot_stacked` | Same as above | HIGH |
| `economics.plot_pie` | `"P_soec_actual"`, `"P_pem"`, `"compressor_power_kw"` | HIGH |
| `profiles.plot_profile` | Column suffix patterns | MEDIUM |
| `static_graphs.create_drain_concentration_figure` | `"KOD_"`, `"Coalescer_"`, `"O2_"` patterns | MEDIUM |

**Recommendation:**  
1. Move column name mappings to [visualization_config.yaml](file:///home/stuart/Documentos/Planta%20Hidrogenio/scenarios/visualization_config.yaml) under a `data_mappings:` section.
2. Create a `ColumnResolver` utility class that accepts component IDs and returns actual column names by probing the DataFrame.

### 3.2 Unit Inconsistencies

| Function | Issue | Location |
|----------|-------|----------|
| `profiles.plot_profile` | Assumes `_temp_c` but falls back to `_temp_k - 273.15` | profiles.py:36-39 |
| `profiles.plot_profile` | Assumes `_pressure_bar` but falls back to `_pa / 1e5` | profiles.py:41-44 |
| `static_graphs.create_drain_line_properties_figure` | Uses `_drain_temp_k` suffix, needs K→C conversion | static_graphs.py:1084 |
| `static_graphs.create_crossover_impurities_figure` | Uses ppm molar vs mass fraction confusion | static_graphs.py:2554+ |

**Recommendation:**  
1. Standardize on Celsius and Bar for all plotting functions.
2. Add unit conversion helpers in `h2_plant/visualization/utils.py`.

### 3.3 Downsampling Performance

**Location:** [plotly_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/plotly_graphs.py), [static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py)

**Issue:** Multiple functions implement ad-hoc downsampling with varying `stride` calculations.

**Examples:**
- `economics.plot_dispatch`: `stride = max(1, len(df) // 2000)`
- `plotly_graphs.plot_storage_fatigue_cycling_3d`: `stride = max(1, len(timestamps) // 1000)`

**Recommendation:**  
Create centralized `downsample_dataframe(df, max_points)` utility in `h2_plant/visualization/utils.py`.

### 3.4 Error Handling

**Issue:** The Orchestrator has a generic `try/except` block that logs errors but does not propagate failure information.

**Location:** `graph_orchestrator.py:108-125`

**Recommendation:**  
Add a `FailedGraphReport` dataclass to track:
- Graph ID
- Exception type
- Missing columns
- Generated partial output path (if any)

---

## Section 4: Redundancy Analysis

### 4.1 [production.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py) Module Assessment

**Files Under Review:**
- [h2_plant/visualization/graphs/production.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py) (133 lines)
- [h2_plant/visualization/static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py) (functions at lines 280-347)

**Function Comparison:**

| graphs/production.py | static_graphs.py | Semantic Match |
|---------------------|------------------|----------------|
| [plot_time_series](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/performance.py#13-44) | [create_h2_production_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#280-300) | 70% similar |
| [plot_stacked](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py#62-105) | [create_h2_production_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#280-300) (partial) | 50% similar |
| [plot_cumulative](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py#107-133) | [create_cumulative_h2_figure](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py#408-432) | 80% similar |

**Key Differences:**

1. **Interface:**
   - [production.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py): Signature [(df, component_ids, title, config)](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graph_catalog.py#134-137) - Orchestrator-compatible.
   - [static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py): Signature [(df, dpi)](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graph_catalog.py#134-137) - Legacy dispatcher-compatible.

2. **Component Resolution:**
   - [production.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py): Uses explicit `component_ids` from config (dynamic).
   - [static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py): Hardcodes column names like `H2_pem_kg`, `H2_soec_kg` (static).

3. **Styling:**
   - [production.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py): Minimal styling (Figure API).
   - [static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py): Full styling with color palette, axis formatting.

**Verdict: KEEP SEPARATE but REFACTOR**

[production.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py) is **NOT redundant**—it serves the new Orchestrator architecture with a different interface. However:

1. **Common Logic Extraction:**
   Move shared column-finding logic to `h2_plant/visualization/utils.py`:
   ```python
   def find_production_columns(df, component_id, variable):
       """Returns column name or None."""
   ```

2. **Style Consistency:**
   Import `COLORS` and styling constants from [static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py) into [production.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/graphs/production.py).

3. **Future Deprecation Path:**
   Legacy [static_graphs.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/static_graphs.py) functions should be marked `@deprecated` with pointer to Orchestrator equivalents.

### 4.2 Duplicate Registration in Catalog

| Graph ID | Registered Function | Issue |
|----------|---------------------|-------|
| `dissolved_gas_efficiency` | `sg.create_drain_concentration_figure` | Same function as `drain_line_concentration` |
| `drain_line_concentration` | `sg.create_drain_concentration_figure` | Duplicate registration |

**Recommendation:** Remove one of these registrations or differentiate functionality.

---

## Section 5: Architecture Reconciliation Matrix

### Current State (3 Parallel Systems)

```
┌──────────────────────────────────────────────────────────────────┐
│                       VISUALIZATION SYSTEMS                       │
├──────────────────┬──────────────────┬────────────────────────────┤
│ 1. Legacy        │ 2. Catalog       │ 3. Orchestrator            │
│ (run_integrated) │ (GRAPH_REGISTRY) │ (GraphOrchestrator)        │
├──────────────────┼──────────────────┼────────────────────────────┤
│ GRAPH_MAP dict   │ GraphCatalog     │ handlers dict              │
│ in run_integ.py  │ in graph_cat.py  │ in graph_orch.py           │
├──────────────────┼──────────────────┼────────────────────────────┤
│ 45 graphs        │ 67 graphs        │ 17 handlers                │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Uses: df         │ Uses: data dict  │ Uses: df + config          │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Output: PNG      │ Output: Plotly/  │ Output: PNG                │
│                  │ Matplotlib       │                            │
└──────────────────┴──────────────────┴────────────────────────────┘
```

### Recommended Target State

```
┌──────────────────────────────────────────────────────────────────┐
│                UNIFIED VISUALIZATION SYSTEM                       │
├──────────────────────────────────────────────────────────────────┤
│ GraphOrchestrator (Central Dispatcher)                           │
│   ├── visualization_config.yaml (defines what to plot)          │
│   ├── graphs/ package (modular handlers)                         │
│   │     ├── production.py                                        │
│   │     ├── economics.py                                         │
│   │     ├── thermal.py                                           │
│   │     ├── separation.py                                        │
│   │     ├── profiles.py                                          │
│   │     ├── soec.py                                              │
│   │     ├── storage.py                                           │
│   │     └── legacy.py (wraps static_graphs.py)                   │
│   └── utils.py (column resolution, downsampling, units)          │
├──────────────────────────────────────────────────────────────────┤
│ DEPRECATE:                                                        │
│   - graph_catalog.py GRAPH_REGISTRY (move metadata to YAML)      │
│   - GRAPH_MAP in run_integrated_simulation.py                    │
│   - MetricsCollector (unused parallel system)                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Section 6: Priority Action Items

| Priority | Action | Effort |
|----------|--------|--------|
| P0 | Delete [MetricsCollector](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/metrics_collector.py#34-323) (unused) or integrate into engine | 2h |
| P0 | Add missing catalog registrations for 11 orphaned functions | 1h |
| P1 | Create `utils.py` with `find_column`, [downsample](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/visualization/metrics_collector.py#215-242), `convert_units` | 3h |
| P1 | Remove duplicate `drain_line_concentration` registration | 15m |
| P2 | Deprecate `GRAPH_MAP` in [run_integrated_simulation.py](file:///home/stuart/Documentos/Planta%20Hidrogenio/h2_plant/run_integrated_simulation.py) | 4h |
| P2 | Add `legacy.py` wrapper module to Orchestrator | 2h |
| P3 | Migrate all 67 catalog graphs to Orchestrator pattern | 8h |
