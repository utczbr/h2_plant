# Detailed Component System Guide

## Overview
The Hydrogen Plant GUI now supports **component-level design**, allowing you to build plants using individual components (PEM stacks, heat exchangers, pumps, etc.) rather than just system-level nodes.

**Version**: 2.1  
**Updated**: December 27, 2025  
**Component Categories**: 20 subdirectories with 50+ component implementations

---

## Component Categories

### 1. Electrolysis Components
**Category**: `h2_plant.components.electrolysis`

- **PEM Stack** - Proton Exchange Membrane electrolyzer
  - Properties: `max_power_kw`, `cells_per_stack`, `parallel_stacks`
  - Ports: power_in, water_in → h2_out, o2_out, heat_out

- **SOEC Cluster** - Solid Oxide Electrolyzer Operator (Multi-module)
  - Properties: `num_modules` (default 6), `max_power_nominal_mw` (**Per Module**), `optimal_limit` (e.g., 0.8)
  - Capacity: Effective MW = `num_modules` * `max_power_nominal_mw` * `optimal_limit`
  - Ports: power_in, steam_in → h2_out, water_out

- **Rectifier/Transformer** - AC/DC power conversion
  - Properties: `max_power_kw`, `efficiency`, **`system` (PEM/SOEC)**
  - Ports: ac_power_in → dc_power_out

### 2. Reforming Components
**Category**: `h2_plant.reforming`

- **ATR Reactor** - Auto-Thermal Reforming with pickled model
  - Properties: `max_flow_kg_h`, `model_path`
  - Model: Uses pre-computed interpolation functions (3.5MB pickle file)
  - Ports: oxygen_in, biogas_in, steam_in → h2_out, heat_out

- **WGS Reactor** - Water-Gas Shift reactor
  - Properties: `conversion_rate` (0.0-1.0)
  - Ports: syngas_in → syngas_out

- **Steam Generator** - Produces steam for processes
  - Properties: `max_flow_kg_h`, **`system` (PEM/SOEC/ATR)**
  - Ports: water_in, heat_in → steam_out

### 3. Separation Components
**Category**: `h2_plant.separation`

- **PSA Unit** - Pressure Swing Adsorption (gas purification)
  - Properties: `gas_type` (H2/O2/Syngas), **`system` (PEM/SOEC/ATR)**
  - Recovery: ~90% (configurable)
  - Ports: feed_in → product_out, waste_out

- **Separation Tank** - Phase separation
  - Properties: `gas_type`, **`system` (PEM/SOEC/ATR)**
  - Ports: mixed_in → gas_out, liquid_out

- **Knock-Out Drum (KOD)** - Flash separation vessel
  - Properties: `diameter_m`
  - Physics: Rachford-Rice flash calculation for VLE
  - Ports: gas_inlet → gas_outlet, liquid_drain

- **Coalescer** - Aerosol mist removal
  - Properties: `design_flow_kg_h`
  - Ports: inlet → outlet, drain

- **TSA Unit** - Thermal Swing Adsorption
  - Properties: `bed_diameter_m`, `cycle_time_hours`, `regen_temp_k`
  - Physics: Ergun pressure drop, dynamic gas heating energy
  - Ports: wet_h2_in → dry_h2_out, water_out

- **Deoxo Reactor** - Catalytic Deoxidizer (PFR)
  - Properties: `component_id`
  - Physics: Coupled Mass/Energy PFR with JIT solver
  - Ports: inlet → outlet

- **Hydrogen Multi-Cyclone** - Centrifugal droplet separation
  - Properties: `design_flow_kg_h`, `num_cyclones`
  - Physics: Stokes law separation efficiency
  - Ports: inlet → gas_outlet, liquid_drain

### 4. Thermal Components
**Category**: `h2_plant.thermal`

- **Heat Exchanger** - Cooling/heating
  - Properties: `max_heat_removal_kw`, `target_outlet_temp_c`, **`system` (PEM/SOEC/ATR)**
  - Ports: hot_in, cold_in → hot_out, cold_out

- **Chiller** - Active refrigeration / Cooling
  - Properties: `cooling_capacity_kw`, `target_temp_k`
  - Ports: fluid_in → fluid_out

- **Electric Boiler** - Steam/water heating
  - Properties: `max_power_kw`, `target_temp_k`
  - Ports: water_in → steam_out

- **Interchanger** - Gas-to-gas heat exchange
  - Properties: `effectiveness`, `hot_side_id`
  - Ports: hot_in, cold_in → hot_out, cold_out

### 5. Compression Components
**Category**: `h2_plant.components.compression`

- **Compressor** - Multi-stage compression train
  - Properties: `num_stages`, `pressure_ratio_per_stage`, `intercooling`
  - Physics: Isentropic efficiency with interstage cooling
  - Ports: gas_in → gas_out

- **CompressorSingle** - Single-stage compressor unit
  - Properties: `pressure_ratio`, `isentropic_efficiency`
  - Ports: inlet → outlet

- **Process Compressor** - Gas compression
  - Properties: `max_flow_kg_h`, `pressure_ratio`, **`system` (PEM/SOEC/ATR)**
  - Ports: gas_in → gas_out

- **Recirculation Pump** - Liquid circulation
  - Properties: `max_flow_kg_h`, `pressure_bar`, **`system` (PEM/SOEC/ATR)**
  - Ports: fluid_in → fluid_out

### 6. External Resources
**Category**: `h2_plant.resources`

- **Grid Connection** - Electricity supply
  - Properties: `max_power_kw`, `price_per_kwh`
  
- **Water Supply** - Municipal water
  - Properties: `max_flow_m3h`, `pressure_bar`, `cost_per_m3`

- **Ambient Heat Source** - Environmental heat
  - Properties: `max_heat_kw`, `ambient_temp_c`

- **Natural Gas Supply** - Biogas/NG feed
  - Properties: `max_flow_kg_h`, `pressure_bar`, `cost_per_kg`

### 7. Storage Components
**Category**: `h2_plant.components.storage`

- **H2StorageTankEnhanced** - Pressurized hydrogen storage
  - Properties: `volume_m3`, `max_pressure_bar`, `initial_level_kg`
  - Ports: h2_in → h2_out

- **OxygenBuffer** - O₂ intermediate storage
  - Properties: `volume_m3`, `max_pressure_bar`

- **TankArray** - Multiple tank management
  - Properties: `num_tanks`, `tank_volume_m3`

### 8. Water System Components
**Category**: `h2_plant.components.water`

- **WaterPumpThermodynamic** - Circulation pump with thermodynamics
  - Properties: `max_flow_kg_h`, `pressure_rise_bar`
  - Ports: inlet → outlet

- **DrainRecorderMixer** - Drain collection and mixing
  - Properties: `num_inlets`
  - Ports: drain_in[] → mixed_out

- **UltrapureWaterStorageTank** - DI water storage
  - Properties: `volume_m3`, `max_level_m3`

### 9. Mixing Components
**Category**: `h2_plant.components.mixing`

- **MultiComponentMixer** - Gas stream mixing
  - Properties: `num_inlets`
  - Physics: Adiabatic mixing with composition tracking

- **WaterMixer** - Steam/water stream combining
- **OxygenMixer** - O₂ stream combining

---

## System Assignment

**Key Feature**: Context-dependent components can be assigned to specific systems.

**Components requiring system assignment:**
- Rectifier → PEM or SOEC
- Heat Exchanger → PEM, SOEC, or ATR
- Steam Generator → PEM, SOEC, or ATR
- PSA Unit → PEM, SOEC, or ATR
- Separation Tank → PEM, SOEC, or ATR
- Process Compressor → PEM, SOEC, or ATR
- Recirculation Pump → PEM, SOEC, or ATR

**How to assign:**
1. Add the component to canvas
2. Select it
3. In Properties panel, set **`system`** dropdown
4. Choose: PEM (0), SOEC (1), or ATR (2)

This tells the configuration generator which system this component belongs to.

---

## Configuration Structure

**Generated Config (v2.0)**:
```yaml
pem_system:
  stacks:
    - component_id: "PEM-Stack-1"
      max_power_kw: 2500.0
  rectifiers:
    - component_id: "RT-1"
      max_power_kw: 2500.0
  heat_exchangers:
    - component_id: "HX-1"
      max_heat_removal_kw: 500.0

soec_system:
  stacks:
    - component_id: "SOEC-Stack-1"
      max_power_kw: 1000.0

atr_system:
  reactors:
    - component_id: "ATR-Reactor"
      max_flow_kg_h: 1500.0
      model_path: "h2_plant/data/ATR_model_functions.pkl"

storage:
  # Same as v1.0

compression:
  # Same as v1.0
```

---

## Design Workflows

### Workflow 1: Design a PEM System
1. Add **PEM Stack** (production)
2. Add **Rectifier**, set `system=PEM`
3. Add **Recirculation Pump**, set `system=PEM`
4. Add **Heat Exchanger** (x3), set `system=PEM`
5. Add **Separation Tank** (x2), set `system=PEM`
6. Add **PSA Unit** (x2), set `system=PEM`
7. Add **LP Tank**, **HP Tank** (storage)
8. Connect: Grid → Rectifier → PEM Stack → Separation → PSA → Tanks

### Workflow 2: Design an ATR System
1. Add **ATR Reactor**
2. Add **WGS Reactor** (x2 - HT and LT)
3. Add **Steam Generator**, set `system=ATR`
4. Add **Heat Exchanger** (x3), set `system=ATR`
5. Add **Process Compressor** (x2), set `system=ATR`
6. Add **PSA Unit**, set `system=ATR`
7. Connect: NG Supply → Compressor → ATR → WGS → PSA → Tanks

### Workflow 3: Combined Plant
- Mix PEM, SOEC, and ATR components
- Assign each component to its system
- Share storage and compression
- Run simulation with all pathways active

---

## Environment Manager

**New Feature**: Time-series environmental data

**What it does:**
- Provides wind power availability (hourly, 8760 hours/year)
- Provides energy prices (day/night patterns)
- Automatically loaded by PlantBuilder
- Accessible to all components via `registry.get('environment_manager')`

**Data Files** (in `h2_plant/data/`):
- `wind_data.csv` - Wind power coefficients
- `EnergyPriceAverage2023-24.csv` - Historical pricing
- `ATR_model_functions.pkl` - ATR interpolation model

**Usage in simulation:**
```python
env_mgr = registry.get('environment_manager')
current_price = env_mgr.get_current_energy_price()  # EUR/kWh
wind_power = env_mgr.get_wind_power_availability(installed_kw=5000)
```

---

## Migration from v1.0 to v2.0

**Backward Compatibility**: Both v1.0 (system-level) and v2.0 (component-level) configs work.

**v1.0 (Legacy)**:
```yaml
production:
  electrolyzer:
    enabled: true
    max_power_mw: 2.5
```

**v2.0 (Detailed)**:
```yaml
pem_system:
  stacks:
    - component_id: "PEM-Stack-1"
      max_power_kw: 2500.0
```

**Which to use:**
- **v1.0**: Quick prototyping, simple plants
- **v2.0**: Detailed design, multiple systems, custom configurations

---

## Tips & Best Practices

1. **Start Simple**: Design one system at a time (PEM first, then add SOEC/ATR)
2. **Use Templates**: Check `configs/standard_plant_template.yaml` for reference
3. **System Assignment**: Always set the `system` property for shared components
4. **Component IDs**: Use descriptive IDs (e.g., "PEM-HX-1", "ATR-Compressor-1")
5. **Testing**: Run simulations frequently to catch configuration errors early

---

## Troubleshooting

**Error: "Unknown node type"**
- Cause: Node not registered in main_window.py
- Fix: Check that all imports are present

**Error: "Schema validation failed"**
- Cause: Missing required properties or invalid system assignment
- Fix: Check Properties panel, ensure all required fields are set

**Error: "KeyError: node_id"**
- Cause: NodeGraphQt internal issue when moving nodes
- Impact: Harmless, does not affect functionality
- Workaround: Ignore or restart GUI if persistent

**Slow simulation startup:**
- Cause: LUTManager building thermodynamic cache, ATR loading pickle model
- Expected: First run may take 5-10 seconds
- Subsequent runs: < 1 second

---

## Reference

**Standard Plant Template**: `configs/standard_plant_template.yaml`  
**User Guide**: `docs/GUI/GUI_USER_GUIDE.md`  
**Architecture**: `docs/ARCHITECTURE.md`  
**Walkthrough**: `walkthrough.md` (in artifacts)

---

## Summary

**Phase 4 adds:**
- ✅ 24 detailed component nodes
- ✅ System assignment (manual)
- ✅ Component aggregation
- ✅ EnvironmentManager (time-series data)
- ✅ Backward compatibility (v1.0 still works)
- ✅ Production-ready GUI

**Status**: Complete and tested
