Based on the code provided, here is the detailed breakdown of data requirements for each graph.

The graphs are divided into **Static (Matplotlib)**, which typically rely on a flattened time-series history DataFrame (where column names contain component IDs), and **Interactive (Plotly)**, which rely on a structured dictionary of metrics.

### 1. Static Graphs (Matplotlib)

These graphs are defined in `static_graphs.py` and generally consume a simulation history DataFrame (`df`).

| Graph Name | Target Component(s) | Required Data Columns / Variables |
| --- | --- | --- |
| **Dispatch Strategy** | SOEC, PEM, Grid | • `minute` (Time)

`P_soec` or `P_soec_actual` (MW)

`P_pem` (MW)

`P_sold` (MW)

`P_offer` (MW) |
| **Price Scenario (Arbitrage)** | Market / Grid | • `minute` (Time)

`Spot` or `spot_price` (EUR/MWh)

*Config:* `ppa_price_eur_mwh` (optional limit)

*Config:* `h2_price_eur_kg` (for breakeven calc) |
| **H2 Production Rate** | SOEC, PEM | • `minute` (Time)

`H2_soec` or `H2_soec_kg` (kg/min)

`H2_pem` or `H2_pem_kg` (kg/min) |
| **O2 Production** | SOEC, PEM | • `minute` (Time)

`H2_soec` (derived from stoichiometry) 

`O2_pem_kg` (or derived from H2_pem) |
| **Water Consumption** | SOEC, PEM | • `minute` (Time)

`Steam_soec` or `steam_soec_kg`

`H2O_pem` or `H2O_pem_kg` |
| **Energy Distribution** | SOEC, PEM, Grid | • `P_soec` (MW)

`P_pem` (MW)

`P_sold` (MW) |
| **Dispatch Curve** | SOEC, PEM | • `P_soec` + `P_pem` (Total Power)

`H2_soec` + `H2_pem` (Total H2) |
| **Cumulative H2** | SOEC, PEM | • `H2_soec` (cumulative sum calculated internally)

`H2_pem` (cumulative sum calculated internally) |
| **Cumulative Energy** | SOEC, PEM, Grid | • `P_soec`, `P_pem`, `P_sold` (integrated over time internally) |
| **System Efficiency** | System (SOEC+PEM) | • `P_soec`, `P_pem` (Power Input)

`H2_soec`, `H2_pem` (Energy Output calculated via LHV) |
| **Revenue Analysis** | Market / System | • `P_sold` (Grid Revenue)

`Spot` (Price)

Total H2 (`H2_soec`+`H2_pem`) * Fixed H2 Price |
| **Water Removal (Total)** | Separators (KOD, Coalescer, DryCooler) | • Columns ending in `_water_removed_kg_h` or `_water_condensed_kg_h`

*Note:* Aggregates sum per component. |
| **Discarded Drains Overview** | Drains (KOD, etc.) | • Mass: `*_water_removed_kg_h`

Temp: `*_temp` or `*_temperature`

Pressure: `*_press` or `*_pressure` |
| **Individual Drain Props** | KOD, Coalescer | • Flow: `*_water_removed_kg_h` or `*_drain_flow_kg_h`

Temp: `*_drain_temp_k`

Pressure: `*_drain_pressure_bar` |
| **Dissolved Gas (Concentration)** | KOD, Coalescer, Drain | • `*_dissolved_gas_ppm` (mg/kg concentration in liquid) |
| **Dissolved Gas (Efficiency)** | Separators (KOD, Coalescer) | • Inlet: `*_dissolved_gas_in` or `*_inlet_dissolved_ppm`

Outlet: `*_dissolved_gas_out` or `*_outlet_dissolved_ppm` |
| **Crossover Impurities** | Deoxo, Sources, Separators | • H2 Stream: `*_o2_impurity_ppm_mol`, `*_outlet_o2_ppm_mol`, or `*_y_o2_ppm`

O2 Stream: `*_h2_impurity_ppm`, `*_outlet_h2_ppm` |
| **Deoxo Profile** | Deoxo Reactor | • `L` (Reactor Length array)

`T` (Temperature array)

`X` (Conversion array)

*(Requires profile data, not time-series)* |
| **Drain Line Properties** | Drain_Collector / WaterMixer | • `*_outlet_mass_flow_kg_h`

`*_outlet_temperature_c` (or K)

`*_outlet_pressure_kpa` |
| **Thermal Load Breakdown** | Chiller, DryCooler | • `*_cooling_load_kw`

`*_sensible_heat_kw`

`*_latent_heat_kw`

`*_tqc_duty_kw` |
| **Recirculation Comparison** | Drain_Collector vs Feed_Tank | • Recovered: `Drain_Collector_outlet_mass_flow_kg_h` & Temp

Recirculated: `WaterTank_mass_flow_out_kg_h` & Temp |
| **Entrained Liquid** | Gas Streams (Post-Separation) | • `*_m_dot_H2O_liq_accomp_kg_s` (Liquid water carried in gas) |
| **Chiller Cooling** | Chiller | • `*_cooling_load_kw`

`*_electrical_power_kw` |
| **Coalescer Separation** | Coalescer | • `*_delta_p_bar` (Pressure Drop)

`*_drain_flow_kg_h` |
| **KOD Separation** | Knock-Out Drum | • `*_rho_g` (Gas Density)

`*_v_real` (Gas Velocity)

`*_water_removed_kg_h` |
| **Dry Cooler Performance** | DryCooler | • `*_heat_rejected_kw`

`*_outlet_temp_k` |
| **Process Train Profile** | All Components in series | *Takes a separate DataFrame containing:*

`Component` (Name)

`T_c` (Temp)

`P_bar` (Pressure)

`H_kj_kg` (Enthalpy)

`S_kj_kgK` (Entropy)

`MassFrac_H2`, `MassFrac_O2`, `MassFrac_H2O` |
| **Water Vapor Tracking** | All Components | • `*_h2o_vapor_kg_h`

`*_molar_fraction_h2o` (used to calc ppm labels) |
| **Plant Balance Schematic** | Control Volume (Plant Wide) | • `*_h2_production_kg` or `*_h2_soec`/`pem`

`*_o2_production`

`*_p_soec`/`pem`

`*_cooling_load_kw`

`*_makeup`

`*_water_removed` |

---

### 2. Interactive Graphs (Plotly)

These graphs are defined in `plotly_graphs.py` and rely on a structured dictionary of data arrays (e.g., `data['pem']['voltage']`).

| Graph Name | Target Component(s) | Required Data Keys (in Metric Dictionary) |
| --- | --- | --- |
| **PEM H2 Production** | PEM Electrolyzer | • `timestamps`

`pem.h2_production_kg_h` |
| **SOEC H2 Production** | SOEC Electrolyzer | • `timestamps`

`soec.h2_production_kg_h` |
| **Total Production (Stacked)** | PEM + SOEC | • `timestamps`

`pem.h2_production_kg_h`

`soec.h2_production_kg_h` |
| **Cumulative Production** | PEM + SOEC | • `timestamps`

`pem.cumulative_h2_kg`

`soec.cumulative_h2_kg` |
| **PEM Voltage** | PEM Stack | • `timestamps`

`pem.voltage` |
| **PEM Efficiency** | PEM System | • `timestamps`

`pem.efficiency` |
| **Energy Price** | Market | • `timestamps`

`pricing.energy_price_eur_kwh` |
| **Dispatch Strategy** | Coordinator / Controller | • `timestamps`

`coordinator.pem_setpoint_mw`

`coordinator.soec_setpoint_mw`

`coordinator.sell_power_mw` |
| **Power Breakdown (Pie)** | PEM, SOEC | • `pem.cumulative_energy_kwh`

`soec.cumulative_energy_kwh` |
| **SOEC Active Modules** | SOEC Arrays | • `timestamps`

`soec.active_modules` (Count) |
| **Tank Storage (3D Fatigue)** | High Pressure Tanks | • `timestamps`

`tanks.hp_pressures` (2D Array: [Time][Tank_ID]) |
| **Ramp Rate Stress** | SOEC / PEM | • `soec.ramp_rates` (Array of MW/min values) |
| **Wind Utilization Curve** | Renewable Source | • `pricing.wind_coefficient`

`pem.power_mw`

`soec.power_mw` |
| **Grid Interaction Phase** | Grid Connection | • `pricing.wind_coefficient`

`pricing.grid_exchange_mw` |
| **LCOH Waterfall** | Economics | • `economics.lcoh_cumulative` (or component breakdown: energy, capex, opex, etc.) |
| **PEM Performance 3D** | PEM Electrolyzer | • `timestamps`

`pem.h2_production_kg_h`

`pem.power_mw` |