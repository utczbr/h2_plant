# Plant Component Specifications & Datasheets
**Project**: Green Hydrogen Plant Simulation  
**Revision**: 2.0 (Engineering Baseline)  
**Status**: As-Run Simulation Data

This document acts as the **Process Datasheet** for the hydrogen plant. It combines design-basis parameters (geometry, rated capacity) with effective operating points verified by dynamic simulation. Use this for equipment sizing, procurement, and HAZOP study.

---

## 1. Compressors (Centrifugal/Reciprocating)
**Service**: Hydrogen Gas Compression  
**Technology**: Multi-stage Polytropic/Adiabatic with Intercooling.

### 1.1 High Pressure (HP) H2 Train (Design vs. Actual)
*5-Stage Compression Train delivering H2 to Storage at 500 bar.*

| Stage ID | Suction P (bar) | Discharge P (bar) | Ratio ($R_c$) | Discharge T (°C) | Mass Flow (kg/h) | Shaft Power (est. kW) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HP_S1** | 40.0 $\to$ | **66.3** | 1.66 | 115.8 (Limit 135) | 276.6 | ~15-20 kW |
| **HP_S2** | 66.3 $\to$ | **109.9** | 1.65 | 103.7 (Limit 135) | 276.6 | ~15-20 kW |
| **HP_S3** | 109.9 $\to$ | **182.1** | 1.65 | 104.2 (Limit 135) | 276.6 | ~15-20 kW |
| **HP_S4** | 182.1 $\to$ | **301.8** | 1.65 | 105.2 (Limit 135) | 276.6 | ~15-20 kW |
| **HP_S5** | 301.8 $\to$ | **500.0** | 1.66 | 106.8 (Limit 135) | 276.6 | ~15-20 kW |

### 1.2 SOEC H2 Booster Train (Wet H2)
*Low-pressure booster train compressing wet H2 from SOEC (0.9 bar) to Purification (41 bar).*

| Stage ID | Suction P (bar) | Discharge P (bar) | Ratio ($R_c$) | Discharge T (°C) | Mass Flow (kg/h) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SOEC_H2_S1** | 0.9 $\to$ | **1.88** | 2.08 | 134.9 | 341.6 | T-Limited |
| **SOEC_H2_S2** | 1.8 $\to$ | **3.81** | 2.11 | 134.9 | 341.6 | T-Limited |
| **SOEC_H2_S3** | 3.8 $\to$ | **7.80** | 2.05 | 135.0 | 339.5 | T-Limited |
| **SOEC_H2_S4** | 7.8 $\to$ | **15.9** | 2.04 | 134.9 | 323.5 | T-Limited |
| **SOEC_H2_S5** | 15.9 $\to$ | **32.3** | 2.03 | 134.9 | 315.6 | T-Limited |
| **SOEC_H2_S6** | 32.3 $\to$ | **41.0** | 1.27 | 63.6 | 311.8 | Fixed P Cap |

### 1.3 SOEC O2 Compression Train
*4-Stage Oxygen Compression with Intercooling (Vent/Capture).*

| Stage ID | Suction P (bar) | Discharge P (bar) | Max Temp (°C) | Application |
| :--- | :--- | :--- | :--- | :--- |
| **SOEC_O2_S1** | ~1.0 | **15.0** | 135 (Limit) | Temp-Limited |
| **SOEC_O2_S2** | ~15.0 | **15.0** | 135 (Limit) | Temp-Limited |
| **SOEC_O2_S3** | ~15.0 | **15.0** | 135 (Limit) | Temp-Limited |
| **SOEC_O2_S4** | ~15.0 | **15.0** | 135 (Limit) | Final Stage |

*   ****Note**: All stages have `outlet_pressure_bar: 15.0` in topology, suggesting parallel boosting or typo. Review for cumulative cascade.
*   **Operating Flow**: ~2440 kg/h O2
*   **Intercooling**: 4x `SOEC_O2_Drycooler` units (Target 30°C)

### 1.4 Technical Specification (Class: `CompressorSingle`)
*   **Thermodynamic Model**: Real Gas (NIST equivalent LUTs).
*   **Efficiency**: $\eta_{is}=0.65$ (Isentropic), $\eta_{mech}=0.96$ (Mechanical).
*   **Control Strategy**:
    *   **Mode A (Fixed P)**: Compress to $P_{target}$.
    *   **Mode B (T-Limit)**: Reduce $P_{out}$ if $T_{discharge} > T_{max}$ (Active in SOEC trains).

---

## 2. Process Control Valves
**Service**: Pressure Reduction / Let-down  
**Type**: Joule-Thomson Throttling Valve

| Tag | Service | $P_{in}$ (bar) | $P_{out}$ (bar) | $T_{out}$ (°C) | $\Delta T_{JT}$ | Application |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PEM_Water_Return** | Liquid Water | 40.0 | 1.0 | 61.6 | +1.6°C | Water Recirculation |
| **PEM_O2_Valve** | Oxygen Gas | 40.0 | 15.0 | 12.2 | -8.0°C | Vent Pressure Control |

*   **Design Note**: Sizing should accommodate sonic flow (choked) conditions if $\Delta P > 50\% P_{in}$.

---

## 3. Dry Coolers (Air-Cooled Heat Exchangers)
**Service**: Process Gas Intercooling & Aftercooling  
**Type**: Fin-Fan Cooler (Indirect Glycol Loop or Direct Air)

### 3.1 Equipment Geometry (from `constants.py`)
These values dictate the physical footprint and heat transfer limits.

| Parameter | H2 Service Units | O2 Service Units | Unit | Source |
| :--- | :--- | :--- | :--- | :--- |
| **TQC Area (Gas→Glycol)** | 10.0 | 5.0 | $m^2$ | `DryCoolerIndirectConstants` |
| **DC Area (Glycol→Air)** | **453.62** | **92.95** | $m^2$ | `DryCoolerIndirectConstants` |
| **Design U (DC)** | 35.0 | 35.0 | $W/m^2K$ | Overall Coefficient |
| **Air Flow (Design)** | 25.9 | 3.55 | kg/s | Fan Capacity |



### 3.2 Operating Performance (Simulation Snapshot)
*Ambient Air Temperature: 25°C*

| Tag | Fluid | Load (Duty) | $T_{in}$ (°C) | $T_{out}$ (°C) | Approach $\Delta T$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SOEC_H2_Intercoolers** | H2 Mix | ~107 kW (avg) | 135.0 | ~34.9 | ~9.9°C |
| **SOEC_O2_Drycoolers** | O2 | ~45-53 kW | 135.0 | ~51.0 | ~26°C |
| **HP_DryCoolers** | H2 | (Varies) | ~105-115 | ~30.0 | ~5.0°C |

*   **Design Requirement**: All dry coolers usually sized for $T_{process} - T_{air} \approx 5-10^\circ C$ approach.
*   **Observation**: SOEC O2 coolers show a large approach (26°C), indicating potential undersizing or low effectiveness ($T_{out} = 51^\circ C$ vs Ambient $25^\circ C$).

---

## 4. PEM Electrolyzer
**Type**: `h2_plant.components.electrolysis.pem_electrolyzer.DetailedPEMElectrolyzer`  
**Description**: Detailed proton exchange membrane electrolyzer model with mechanistic electrochemistry, degradation tracking, and finite-volume thermal inertia.

### 4.1 Class Parameters
Initialized via `config` dict.

| Parameter | Type | Default | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| `max_power_mw` | float | `5.0` | MW | Rated power capacity. |
| `base_efficiency` | float | `0.65` | - | Nominal efficiency (reference). |
| `use_polynomials` | bool | `True` | - | O(log(n)) |
| `water_excess_factor` | float | `0.02` | - | Safety margin for water stoichiometry. |
| `out_pressure_pa` | float | `40e5` | Pa | H2 discharge pressure (40 bar). |

### 4.2 Physical Constants (PEMConstants)
Defined in `h2_plant.config.constants_physics.PEMConstants`.

| Constant | Value | Unit | Description |
| :--- | :--- | :--- | :--- |
| **Geometry** | | | |
| `N_stacks` | 35 | - | Number of stacks per 5 MW unit. |
| `N_cell_per_stack` | 85 | - | Cells per stack. |
| `A_cell_cm2` | 300.0 | cm² | Active area per cell. |
| **Electrochemistry** | | | |
| `delta_mem` | 100 μm | m | Membrane thickness. |
| `sigma_base` | 0.1 | S/cm | Proton conductivity. |
| `j0` | 1.0e-6 | A/cm² | Exchange current density. |
| `j_lim` | 4.0 | A/cm² | Limiting current density. |
| `j_nom` | 2.91 | A/cm² | Nominal current density. |
| **Operations** | | | |
| `T_default` | 333.15 | K | Operating temperature (60°C). |
| `P_op_default` | 40.0 | bar | Operating pressure. |
| `h2_purity_molar` | 0.9948 | - | ~99.5% H2 (saturated). |
| **Output Impurities** | | | |
| `h2o_vapor_ppm_molar` | **4986** | ppm | H2O in H2 (Design-Basis). |
| `o2_crossover_ppm_molar` | **200** | ppm | O2 in H2 (Crossover). |

### 4.3 Topology Instances
| Component ID | Power | Pressure | Design T | Design P | Design Production |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `PEM_Electrolyzer` | **5.0 MW** | 40 bar | 60.0°C | 40.00 bar | H2: **~100 kg/h**, O2: ~800 kg/h |

> **Note**: 7200h simulation ran PEM in standby (SOEC priority). Design values above reflect full-load capacity.


---



## 6. Separation & Purification
Components responsible for phase separation (Gas/Liquid) and impurity removal (DeOxo, PSA).

### 6.1 KnockOutDrum (KOD)
**Type**: `h2_plant.components.separation.knock_out_drum.KnockOutDrum`  
**Description**: Vertical gravity separator using Souders-Brown sizing and Rachford-Rice flash equilibrium.

#### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `diameter_m` | float | `1.0` | Vessel inner diameter (controls velocity). |
| `delta_p_bar` | float | `0.05` | Pressure drop. |
| `gas_species` | str | `'H2'` | Primary gas phase species. |

#### Physics & Implementation
*   **Flash**: Isothermal Rachford-Rice flash calculates VLE at inlet T.
*   **Sizing**: Checked against **Souders-Brown** limit ($K=0.08$). Status: `OK` or `UNDERSIZED`.
*   **Carryover**: Limited by mist entrainment (default 20 mg/Nm³) if sized correctly.

#### Topology Instances (7200h Simulation)
| Component ID | Diameter | Sim T | Sim P | Sim Drain | Cumulative |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SOEC_H2_KOD_1` | 0.8 m | 30.0°C | 0.90 bar | **2200 kg/h** | 11.2M kg |
| `SOEC_H2_KOD_2` | 0.6 m | 4.0°C | 0.65 bar | **49.8 kg/h** | 531.8k kg |
| `PEM_H2_KOD_1` | 0.5 m | 60.0°C | 39.95 bar | Trace | 268.2k kg |
| `PEM_H2_KOD_2` | 0.5 m | 4.0°C | 39.85 bar | 0.002 kg/h | 14.9k kg |
| `PEM_H2_KOD_3` | 0.5 m | 4.0°C | 39.80 bar | 0.0002 kg/h | 1.2k kg |
| `PEM_O2_KOD_1` | 0.5 m | 60.0°C | 39.95 bar | (Bulk) | 13.4M kg |
| `PEM_O2_KOD_2` | 0.5 m | 4.0°C | 39.65 bar | 0.48 kg/h | 7.7k kg |


### 6.2 HydrogenMultiCyclone
**Type**: `h2_plant.components.separation.hydrogen_cyclone.HydrogenMultiCyclone`  
**Description**: Axial multi-tube cyclone for high-efficiency mist removal using centrifugal force.

#### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `element_diameter_mm` | float | `50.0` | Diameter of single cyclone tube. |
| `vane_angle_deg` | float | `45.0` | Swirl vane angle. |
| `target_velocity_ms` | float | `20.0` | Design axial velocity (auto-scales tube count). |

#### Physics & Implementation
*   **Model**: **Barth/Muschelknautz** mechanics solved via Numba JIT.
*   **Cut-Size**: Calculates $d_{50}$ (critical particle size). Efficiency depends on $d_{50}$ (<5μm = 99%).
*   **Sizing**: Dynamic tube count allocation ($N_{tubes}$) to maintain target velocity.

#### Topology Instances
| Component ID | Target Vel | Sim T | Sim P | Sim Drain | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SOEC_H2_Cyclone_1` | 22 m/s | 4.0°C | 0.65 bar | 33.5 kg/h | 100% Eff |
| `SOEC_H2_Cyclone_2` | 22 m/s | 30.0°C | 3.76 bar | 31.4 kg/h | 99.9% Eff |
| `SOEC_H2_Cyclone_3` | 22 m/s | 30.0°C | 7.74 bar | 15.4 kg/h | 99.9% Eff |
| `SOEC_H2_Cyclone_4` | 22 m/s | 30.0°C | 15.83 bar | 7.5 kg/h | 99.9% Eff |
| `SOEC_H2_Cyclone_5` | 22 m/s | 30.0°C | 32.19 bar | 3.7 kg/h | 99.9% Eff |

### 6.3 Coalescer
**Type**: `h2_plant.components.separation.coalescer.Coalescer`  
**Description**: Fibrous cartridge filter for polishing final aerosols.

#### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `d_shell` | float | `0.3` | Vessel diameter (m). |
| `l_elem` | float | `0.5` | Element length (m). |
| `gas_type` | str | `'H2'` | Fluid species. |

#### Physics & Implementation
*   **Flow**: **Carman-Kozeny** equation for pressure drop in porous media.
*   **Efficiency**: Fixed high efficiency (99.99%) for liquid removal.
*   **Viscosity**: Sutherland power law ($\mu \propto T^{0.7}$).

#### Topology Instances
| Component ID | Shell Dia | Sim T | Sim P | Sim Drain |
| :--- | :--- | :--- | :--- | :--- |
| `SOEC_H2_Coalescer_2` | 0.3 m | 4.0°C | 39.87 bar | 0.64 kg/h |
| `PEM_H2_Coalescer_1` | (Default) | 4.0°C | 39.85 bar | 0.0006 kg/h |
| `PEM_O2_Coalescer_1` | (Default) | 4.0°C | 39.35 bar | 0.0205 kg/h |

### 6.4 DeoxoReactor
**Type**: `h2_plant.components.purification.deoxo_reactor.DeoxoReactor`  
**Description**: Catalytic fixed-bed reactor for O2 removal ($2H_2 + O_2 \to 2H_2O$).

#### Physics & Implementation
*   **Model**: Multi-zone **Plug Flow Reactor (PFR)** via RK4 integration.
*   **Solver**: Single-pass Numba JIT kernel for speed.
*   **Heat**: Tracks adiabatic temperature rise ($\Delta T_{ad}$) and catalyst limits.
*   **Pressure**: Ergun-type scaling from design point.

#### Topology Instances
| Component ID | Design | Sim T_in | Sim T_out | Sim O2 Out |
| :--- | :--- | :--- | :--- | :--- |
| `SOEC_H2_Deoxo_1` | - | 30.0°C | 32.0°C | 0 ppm |
| `PEM_H2_Deoxo_1` | - | 40.0°C | 50.0°C | 0 ppm |

#### 6.2 PSA (Pressure Swing Adsorption)
**Type**: `h2_plant.components.separation.psa.PSA`  
**Description**: Cycle-averaged adsorption unit for final H2 polishing.

#### Bed Geometry (Hardcoded in Logic)
These parameters drive the Ergun pressure drop calculation.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Bed Diameter** | 0.35 m | Per bed. |
| **Bed Length** | 1.00 m | Per bed. |
| **Particle Size** | 3.0 mm | Adsorbent pellet diameter. |
| **Porosity** | 0.40 | Bed void fraction. |

#### Parameters (Configuration)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `recovery_rate` | 0.90 | 90% H2 Recovery Target. |
| `purity_target` | 0.9999 | 99.99% Purity (Target). |

#### Performance
| Component ID | Product Flow | Tail Gas | Est. Power |
| :--- | :--- | :--- | :--- |
| `SOEC_H2_PSA_1` | 239.9 kg/h | 27.2 kg/h | ~84.9 kW |

## 4. Electrolyzer Specifications
**Service**: Hydrogen Production (Base Load & Dynamic)

### 4.1 PEM Electrolyzer (Detailed Mechanistic Model)
**Technology**: Proton Exchange Membrane (Low Temperature)  
**Model**: `DetailedPEMElectrolyzer`

#### Design Parameters (Stack Construction)
*Reference: `PEMConstants` in `config/constants_physics.py`*

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Active Structure** | 35 Stacks | Parallel arrangement. |
| **Cells per Stack** | 85 | Series arrangement. |
| **Active Area** | 300 $cm^2$ | Per cell. |
| **Membrane** | 100 $\mu m$ | Thickness ($N_{115}$ equiv). |
| **Rated Power** | 5.0 MW | Nominal system capacity. |
| **Op. Pressure** | 40.0 bar | Direct pressurized production. |
| **Op. Temperature** | 60°C | Nominal setpoint. |

#### Physics & Constraints (Hidden Parameters)
*   **Faraday Efficiency**: $\eta_F = j^2 / (j^2 + 0.02)$.
*   **Cathode Drag**: ~10% of reaction water dragged to H2 side.
*   **Impurity/Crossover**:
    *   $O_2$ in $H_2$: **200 ppm** (Safety Limit).
    *   $H_2$ in $O_2$: **4000 ppm** (Anode Safety).

#### Operational Performance (Simulation) (NOT ACTIVE IN CURRENT SIMULATION)
*Note: The current simulation run focused on the SOEC train. PEM train was idle or low-flow.*

---

### 4.2 SOEC Electrolyzer (Cluster)
**Technology**: Solid Oxide Electrolysis Cell (High Temperature)  
**Model**: `SOECOperator` (Cluster Manager)

#### Design Parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Modules** | 6 x 1.6 MW | Total ~10 MW Capacity. |
| **Steam Feed** | 11.0 kg/kg H2 | High steam/hydrogen ratio prevents Ni oxidation. |
| **Thermodynamics** | Endothermic | Requires heat input or high voltage to maintain T. |

#### Physics & Constraints (Hidden Parameters)
*   **Baseline Efficiency**: **37.5 kWh/kg** (LHV Basis).
*   **Degradation**: Voltage rises over time (lookup table).
*   **Crossover Leaks** (Hardcoded Constants):
    *   $O_2$ in $H_2$: **200 ppm**.
    *   $H_2$ in $O_2$: **4000 ppm**.

#### Operational Performance (7200h Simulation)
| Metric | Value | Unit | Notes |
| :--- | :--- | :--- | :--- |
| **Active Modules** | 3-6 | - | Dynamic load following. |
| **Power** | 5.6-11.5 | MW | Variable, avg ~217 kg/h H2. |
| **Stack Efficiency** | 63-65% | LHV | |
| **H2 Production** | **217.4** | kg/h | 7200h average. |
| **Cumulative H2** | **1,564,945** | kg | Total over 7200h run. |


---



## 6. Purification (Deoxo & PSA)
**Service**: Final Product Quality Upgrade

### 6.1 Deoxo Reactor (Catalytic)
**Type**: Fixed Bed Plug Flow Reactor
**Catalyst**: Pd/Al2O3 (Assumed)

#### Reactor Geometry (`DeoxoConstants`)
| Parameter | Value |
| :--- | :--- |
| **Length/Diameter** | $L=1.33m / D=0.324m$ |
| **Catalyst Volume** | ~0.11 $m^3$ |

#### Operating Performance
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Inlet** | 33.7°C / 41.0 bar | |
| **Conversion** | **99.99999%** | Efficiency ($O_2$ removal). |
| **Outlet O2** | $< 0.1$ ppm | High Purity. |
| **Temp Rise** | +1.9°C | Exothermic Reaction ($2H_2 + O_2 \to 2H_2O$). |
| **H2 Consumed** | ~0.1 kg/h | Negligible loss. |

### 6.2 PSA (Pressure Swing Adsorption)
**Type**: `h2_plant.components.separation.psa.PSA`
**Description**: Cycle-averaged adsorption unit for final H2 polishing.

#### Bed Geometry (Hardcoded in Logic)
These parameters drive the Ergun pressure drop calculation.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Bed Diameter** | 0.35 m | Per bed. |
| **Bed Length** | 1.00 m | Per bed. |
| **Particle Size** | 3.0 mm | Adsorbent pellet diameter. |
| **Porosity** | 0.40 | Bed void fraction. |

#### Parameters (Configuration)
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `recovery_rate` | 0.90 | 90% H2 Recovery Target. |
| `purity_target` | 0.9999 | 99.99% H2 Purity Target. |

#### Performance
| Component ID | Product Flow | Tail Gas | Power | Note |
| :--- | :--- | :--- | :--- | :--- |
| `SOEC_H2_PSA_1` | **134.3 kg/h** | 15.2 kg/h | **48.1 kW** | 7200h Sim |
| `PEM_H2_PSA_1` | **~90 kg/h** | ~10 kg/h | **10.0 kW** | Design Cap |



---

## 7. Storage & Balance of Plant
**Service**: Inventory Management and Utilities

### 7.1 Hydrogen Storage
**Type**: High Pressure Cascade (Type IV Tanks)

| Parameter | Value |
| :--- | :--- |
| **Pressure Rating** | **500 bar** |
| **Unit Capacity** | 200 kg H2 |
| **Array Size** | Configurable (Currently vector simulated). |

### 7.2 Electric Boilers
**Service**: Steam Generation (SOEC) & Gas Heating (PEM/SOEC).

| Tag | Duty (kW) | Target T (°C) | Eff. | Application | Design Capacity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SOEC_H2_ElectricBoiler_PSA** | 21.6 | 40 | 0.98 | Pre-heat H2 for PSA. | 50 kW |
| **SOEC_Steam_Generator** | **2714** | 152 | 0.98 | Steam from Water. | 3000 kW |
| **PEM_H2_ElectricBoiler_1** | **25.0** | 40 | 0.98 | Pre-heat H2 (Pre-Deoxo). | 25 kW |
| **PEM_H2_ElectricBoiler_2** | **25.0** | 30 | 0.98 | Pre-heat H2 (Pre-PSA). | 25 kW |
| **PEM_O2_ElectricBoiler** | **25.0** | 40 | 0.98 | Pre-heat O2. | 25 kW |

> **Note**: PEM boilers shown at design capacity (25 kW each). 7200h simulation ran PEM in standby.



### 7.3 Water Management
**Pumps**:
*   `WaterPumpThermodynamic`: Real pump curves with isentropic heating.
*   **Operating Point**: Pressurizes feed water to system pressure (e.g. 200+ bar for HP spray or system pressure).
*   **Constraint**: Checks NPSH margin ($P_{in} - P_{sat} > 0.3$ bar) to prevent cavitation.

#### Topology Instances (Pumps - 7200h Simulation)
| Component ID | Target P | Sim P_out | Sim T_out | Power (Shaft) | Cumulative |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SOEC_Feed_Pump` | 5.0 bar | 5.00 bar | 20.5°C | **0.41 kW** | 4.0k kWh |
| `PEM_Water_Pump` | 40.0 bar | 40.00 bar | 5.3°C | **5.7 kW** | 41.7k kWh |


**Mixers**:
*   `SOEC_Drain_Mixer`: Collects recovered water from SOEC separation drains.
*   `PEM_Drain_Mixer`: Collects recovered water from PEM separation drains.
*   `SOEC_Makeup_Mixer`: Target flow **3600 kg/h**. Blends drain + fresh makeup.
*   `PEM_Makeup_Mixer`: Target flow **4000 kg/h**. Blends drain + fresh makeup.
*   `H2_Production_Mixer`: Joins PSA product streams adiabatically.

**Water Balance Tracker** (`Water_Balance_Tracker`):
*   **Type**: Virtual Calculation Node (Not Physical Equipment).
*   **Function**: Calculates stoichiometric water consumption from H2 production rates.
*   **Purpose**: Sizing basis for Makeup Mixers and water balance closure.

### 7.4 HP Compression Train Dry Coolers
**Service**: Intercooling for 500 bar Storage Compression.
**Geometry**: Uses `DryCoolerIndirectConstants` defaults (U=35 W/m²K).

| Tag | Suction P | Discharge P | Duty | T_out |
| :--- | :--- | :--- | :--- | :--- |
| **HP_DryCooler_S1** | 66.3 bar | 66.25 bar | ~15 kW | 30°C |
| **HP_DryCooler_S2** | 109.9 bar | 109.85 bar | ~15 kW | 30°C |
| **HP_DryCooler_S3** | 182.1 bar | 182.05 bar | ~15 kW | 30°C |
| **HP_DryCooler_S4** | 301.8 bar | 301.75 bar | ~15 kW | 30°C |
| **HP_DryCooler_S5** | 500.0 bar | 499.95 bar | ~15 kW | 30°C |


---

## 8. Process Cooling & Heat Recovery (Thermal Machines)

### 8.1 Chillers (Active Cooling)
**Service**: Sub-ambient cooling (Water Knockout).

| Tag | Duty (kW) | $T_{out}$ Target | $T_{out}$ Actual | Design Capacity | Electrical Load |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SOEC_H2_Chiller_1** | **31.6 kW** | 4.0 °C | 4.0 °C (Met) | **500.0 kW** | ~7.9 kW (COP 4) |
| **SOEC_H2_Chiller_2** | **33.6 kW** | 4.0 °C | 4.0 °C (Met) | **500.0 kW** | ~8.4 kW (COP 4) |
| **PEM_H2_Chiller_1** | **~50 kW** | 4.0 °C | 4.0 °C | **500.0 kW** | ~12.5 kW (COP 4) |
| **PEM_H2_Chiller_2** | **~50 kW** | 4.0 °C | 4.0 °C | **500.0 kW** | ~12.5 kW (COP 4) |
| **PEM_O2_Chiller_1** | **~100 kW** | 4.0 °C | 4.0 °C | **500.0 kW** | ~25 kW (COP 4) |

*   **Engineering Note**: Design capacity (500 kW) provides significant headroom. PEM values shown at design-basis (full 5 MW load).


### 8.2 Interchanger (Heat Recovery)
**Service**: Waste Heat Recovery (Stack Gas $\to$ Feed Water).

| Tag | $Q_{transferred}$ | Hot Side $\Delta T$ | Cold Side $\Delta T$ | Effectiveness |
| :--- | :--- | :--- | :--- | :--- |
| **SOEC_H2_Interchanger_1** | **309.5 kW** | 152°C $\to$ 58°C | 10°C $\to$ 95°C | High |
