# System Status & Investigation Log

**Status:** Investigation Complete

## Executive Summary
This document tracks the ongoing investigation into the H2 Plant Simulation codebase. It serves as a living record of findings, bugs, architectural notes, and suggested improvements.

## Investigation Schedule & Progress
| Module/Area | Status | Key Findings |
| :--- | :--- | :--- |
| **Core Architecture** | **Complete** | `Orchestrator` relies on legacy storage API (`current_level_kg`), breaking with new components. |
| **Configuration** | **Complete** | `PlantConfig` is robust but partially ignored by Orchestrator (hardcoded overrides). |
| **Components** | **Complete** | Storage/Compression mix "Push" vs "Pull" paradigms. SOEC/PEM pressure mismatch (1 vs 30 bar). |
| **GUI** | Pending | Not yet audited. |

## Detailed Investigation Log

### 4. GUI Integration (New)
- **New Nodes Added**:
    - `LPTankArrayNode` / `HPTankArrayNode`: Maps to `TankArray` (Vectorized).
    - `LPEnhancedTankNode` (Display: "LP Tank Dynamic") / `HPEnhancedTankNode` (Display: "HP Tank Dynamic"): Maps to `H2StorageTankEnhanced` (Stateful/Dynamic).
- **Graph Adapter**: Updated to support mapping these nodes to `PlantConfig` with a `model` discriminator ('array' vs 'enhanced').
- **PlantBuilder**: Updated to instantiate the correct storage class based on the `model` config field.

### 1. Core Architecture
- **Orchestrator (`orchestrator.py`)**:
    - **Hybrid Logic**: explicit handling for SOEC/PEM, but generic sweep for BoP.
    - **Hardcoded Values**: `num_modules = 6`, `soec_h2_kwh_kg = 37.5`.
    - **Flow Control**: `_step_downstream` implements a "push" model.
    - **State Management**: `simulation_state` dictionary tracks history.
- **Core Modules (`core/`)**:
    - **`Component`**: Solid ABC. `receive_input` defaults to rejecting flow (0.0). `step` must be called after `initialize`.
    - **`Stream`**: Handles thermodynamics (Enthalpy, Density).
        - Uses polynomial Cp (Shomate).
        - `mix_with` does adiabatic mixing.
        - `density_kg_m3` hardcodes liquid water density to 1000 kg/m³.
- **Physics Models (`models/`)**:
    - **`soec_operation.py`**: Advanced state machine (Ramp/Standby). Has degradation tables.
    - **`pem_physics.py`**: Voltage curves.
    - **`flow_dynamics.py`**: `GasAccumulatorDynamics` clamps max pressure to 350 bar hardcoded.

### 2. Components
- **Electrolysis**:
    - **`SOECOperator`**:
        - Explicitly clamps power if water is limited (but defaults to infinite).
        - Output H2 pressure is **1 bar** (101325 Pa).
        - Uses JIT for optimization.
    - **`DetailedPEMElectrolyzer`**:
        - Output H2 pressure is **30 bar** (30e5 Pa).
        - Implicit dependency on `Coordinator` in `step` (pulls setpoint).
        - Assumes infinite water (calculates consumption but doesn't clamp power).
        - Defines auxiliary stubs (PSA, Pump) at end of file.
- **Storage**:
    - **`TankArray` (`h2_tank.py`)**: Vectorized high-performance model. `get_output` returns *input buffer* (pass-through?), while `extract_output` discharges. Mismatch with Orchestrator expectation.
    - **`H2StorageTankEnhanced`**: Uses `GasAccumulatorDynamics`. Returns 0 flow stream in `get_output` (Passive).
    - **`Orchestrator` Mismatch**: Orchestrator looks for `current_level_kg` (legacy?) but `TankArray` uses arrays and `H2StorageTankEnhanced` uses `mass_kg`.
- **Compression**:
    - **`CompressorStorage`**:
        - Explicitly models steps (Inlet -> Isentropic Poly -> Intercool -> Next).
        - Logic copied from legacy with hard dependencies on `LUTManager` or `CoolProp`.
        - Calculates specific energy (kWh/kg) and updates cumulative stats.
        - **`get_output` returns cooled stream at outlet pressure.**
- **Power**:
    - **`Rectifier`**: Simple efficiency model (0.97).
- **Thermal**:
    - **`Chiller`**: Enthalpy-based with fallback to Cp. Caps capacity.
    - **`SteamGenerator`**: Simple pass-through mass balance.
- **Config**:
    - **`plant_config.py`**: Pydantic-style dataclasses. Good validation.
    - **`simulation_config.yaml`**: Defines simulation parameters. Hardcoded constraints like `max_flow_kg_h` are here.

- **Orchestrator (`orchestrator.py`)**:
    - **Critical Storage Mismatch**: Loops over components looking for `current_level_kg` to perform Balance of Plant (BoP) sweep. **New storage components (`TankArray`, `H2StorageTankEnhanced`) do NOT have this attribute**, so they are ignored, breaking the flow chain causing tanks to never discharge.
    - **Hardcoded Values**: `num_modules = 6` (line 121) and `soec_h2_kwh_kg = 37.5` (line 155) are hardcoded, overriding `PlantConfig`.
    - **Manual Stream Construction**: Manually creates `Stream` objects for BoP push, assuming Temp=298.15K and pure H2, bypassing component thermodynamics.
    - **Push Logic**: Implements a "Push" model (forcing mass to compressor), while `H2StorageTankEnhanced` expects a "Pull" (consumer calls `extract_output`).
    - **Power Accounting**: Correctly deducts `P_bop` (Compressors/Chillers) from `P_sold`.

## Identified Improvements & Issues
*(Prioritized list of bugs, hardcoded values, and architectural weaknesses)*

### Critical (Breaking Integration)
- [x] **[Orchestrator]** **Storage Mismatch**: ~~`Orchestrator` loops over components checking for `current_level_kg`.~~ **FIXED**: Orchestrator now detects both legacy (`current_level_kg`) and unified (`get_inventory_kg`/`withdraw_kg`) interfaces.
- [x] **[Orchestrator]** **Push vs Pull Conflict**: ~~Orchestrator uses "Push" while `H2StorageTankEnhanced` expects "Pull".~~ **FIXED**: Unified interface uses `withdraw_kg()` which is pull-compatible.
- [x] **[Orchestrator/Config]** **Hardcoded Overrides**: ~~`num_modules=6` and `soec_h2_kwh_kg=37.5` were hardcoded.~~ **FIXED**: Now reads from `self.context.physics.soec_cluster.*`.

### Major (Potential Bugs/Inconsistencies)
- [x] **[SOEC vs PEM]** **Pressure Inconsistency**: ~~SOEC 1 bar, PEM 30 bar.~~ **FIXED**: Both now use configurable `out_pressure_pa` with 30 bar default.
- [x] **[GasAccumulatorDynamics]** ~~Hardcoded 350 bar clamp.~~ **FIXED**: Now uses configurable `max_pressure_pa` param.
- [x] **[Stream]** ~~Liquid density hardcoded 1000 kg/m³.~~ **FIXED**: Delegates to LUTManager with fallback.
- [x] **[PEM]** ~~Assumes infinite water supply.~~ **FIXED**: Added `water_buffer_kg` tracking in `receive_input`.
- [x] **[TankArray]** ~~`get_output` returns input buffer.~~ **FIXED**: Configurable `output_mode` ('availability' default).

### Minor (Cleanup/Optimization)
- [x] **[Component]** ~~`receive_input` defaults to 0.0.~~ **FIXED**: Added DEBUG logging + `strict_inputs` config option.
- [x] **[Orchestrator]** ~~Manual Stream uses fixed 298.15K.~~ **ALREADY OK**: Uses `getattr(comp, 'temperature_k', 298.15)`.
- [x] **[PEM]** ~~Redundant Coordinator fetch.~~ **ALREADY OK**: Coordinator is fallback only when `_target_power_mw <= 0`.


