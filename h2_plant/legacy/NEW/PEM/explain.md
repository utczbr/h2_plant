Based on my analysis of the reference codes, I'll create a comprehensive, structured technical documentation. Given the extensive content, I'll organize it into a clear hierarchical structure.

# PEM Electrolyzer Gas Purification Simulator - Technical Documentation

## 1. System Architecture and Overview

The simulator implements a steady-state, component-by-component process model for H₂ and O₂ purification lines downstream of a PEM electrolyzer. The architecture follows a sequential execution pattern with three main phases:[1]

### 1.1 Execution Philosophy
- **Steady-state modeling**: No time-marching integration; all balances computed at equilibrium[1]
- **Sequential component resolution**: Each unit operation receives an inlet state, performs its closure, and passes outlet state to next component[1]
- **Two-line coordination**: H₂ line solved first, then O₂ line with demister capacity constraint from H₂[1]
- **Post-processing integration**: Drain recirculation and makeup water balance solved after main lines converge[6]

### 1.2 Core Data Structure
The central data structure is a **pandas DataFrame** with one row per component, containing thermodynamic state variables, mass/energy flows, and equipment-specific results. Each row stores:[1]

```python
# From process_execution.py lines defining history.append()
{
    'Componente': str,           # Unit operation name
    'TC': float,                 # Temperature [°C]
    'Pbar': float,              # Pressure [bar]
    'mdotgaskgs': float,        # Main gas mass flow [kg/s]
    'mdotmixkgs': float,        # Total mixture mass flow [kg/s]
    'mdotH2Ovapkgs': float,     # Water vapor flow [kg/s]
    'mdotH2Oliqaccompkgs': float, # Liquid water carryover [kg/s]
    'yH2O': float,              # Water molar fraction [-]
    'yO2': float,               # O₂ molar fraction [-]
    'yH2': float,               # H₂ molar fraction [-]
    'HmixJkg': float,           # Mixture specific enthalpy [J/kg]
    'QdotfluxoW': float,        # Thermal power exchanged [W]
    'WdotcompW': float,         # Shaft/electric power [W]
    'AguaCondensadakgs': float, # Water condensed/removed [kg/s]
    'AguaPuraRemovidaH2Okgs': float,  # Pure water drain [kg/s]
    'GasDissolvidoremovidokgs': float, # Dissolved gas loss [kg/s]
    # ... additional component-specific fields
}
```
[1]

***

## 2. Global Execution Sequence

### 2.1 Initialization and Configuration
**File**: `main_PEM_simulator.py`

**Step 1: Import Constants**
```python
from constants_and_config import (
    PINBAR,                    # Inlet pressure [bar]
    TINC,                      # Inlet temperature [°C]
    MDOTGH2, MDOTGO2,          # Gas mass flows [kg/s]
    TCHILLEROUTH2CC1, TCHILLEROUTO2C,  # Chiller setpoints [°C]
    MDOT_H2O_LIQ_IN_H2_TOTAL_KGS,      # Total H₂O for H₂ line [kg/s]
    MDOT_H2O_LIQ_IN_O2_TOTAL_KGS,      # Total H₂O for O₂ line [kg/s]
    YO2INH2, YH2INO2,          # Cross-over impurity fractions
    MODE_DEOXO_FINAL, LDEOXO_OTIMIZADO_M, DC2_MODE_FINAL
)
```


**Step 2: Import Process Simulator**
```python
from process_execution import simular_sistema
```


### 2.2 Single-Gas Simulation Wrapper
**Function**: `executar_simulacao_unica(gasfluido, mdotg_kgs, deoxo_mode, L_deoxo, dc2_mode, KOD, mdotH2O_total_fluxo_input, yO2_in, yH2_in, mdotliqmax_h2_ref)`

**Purpose**: Encapsulates call to `simular_sistema()` with pre/post-processing logic

**Returns**:
- `df_final`: DataFrame with all component states
- `result_real`: Dictionary with auxiliary data (deoxo profiles, etc.)
- `mdotliqmax_kgs`: Demister liquid capacity limit [kg/s]



### 2.3 Two-Gas Coordinated Simulation
**Location**: `if __name__ == "__main__":` block in `main_PEM_simulator.py`

**Sequence**:

1. **H₂ Line Simulation** (primary):
   ```python
   df_h2, result_h2, mdotliqmax_h2_ref = executar_simulacao_unica(
       "H2", MDOTGH2, MODE, LDEOXO_M, DC2_MODE,
       KOD=1,
       mdotH2O_total_fluxo_input=MDOT_H2O_LIQ_IN_H2_TOTAL_KGS,
       yO2_in=YO2INH2, yH2_in=0,
       mdotliqmax_h2_ref=None  # Free to compute
   )
   mdotliqmax_h2_ref = df_h2[df_h2['Componente']=='Entrada']['MDOTLIQMAXDEMISTERKGS'].iloc
   ```


2. **O₂ Line Simulation** (with H₂ constraint):
   ```python
   df_o2, result_o2, _ = executar_simulacao_unica(
       "O2", MDOTGO2, MODE, LDEOXO_M, dc2_mode=None,
       KOD=1,
       mdotH2O_total_fluxo_input=MDOT_H2O_LIQ_IN_O2_TOTAL_KGS,
       yO2_in=0, yH2_in=YH2INO2,
       mdotliqmax_h2_ref=mdotliqmax_h2_ref  # Force 1:2 ratio
   )
   # Inside simular_sistema, for O₂:
   if gasfluido == "O2" and mdotliqmax_h2_ref is not None:
       MDOTLIQMAXDEMISTERKGS = mdotliqmax_h2_ref / 2.0
   ```
[1]

3. **Drain Line and Recirculation** (optional):
   ```python
   if drain_mixer is not None:
       drenos_plot_data, _, estado_recirculacao_final = \
           drain_mixer.executar_simulacao_mixer(df_h2, df_o2, MOSTRAR_GRAFICOS)
   ```


### 2.4 Reporting and Visualization
After both simulations, sequential plot generation:
```python
plot_vazao_massica_total_eremovida(df_h2, "H2", ...)
plot_fluxos_energia(df_h2, df_o2, ...)
plot_agua_removida_total(df_h2, df_o2, ...)
exibir_balanco_agua_inicial(df_h2, df_o2)
```


***

## 3. Component-Level Mathematical Models

### 3.1 Dry Cooler + TQC (Two-Stage Cooling System)
**Files**: `modelo_dry_cooler.py`, called from `process_execution.py`[1]

**Physical Configuration**:
- **TQC (Trocador Quente)**: Gas-glycol counter-current heat exchanger
- **Dry Cooler**: Glycol-air heat rejection unit with fan

**TQC Energy Balance** (NTU-ε method):

1. **Heat Capacities**:
   \[
   \begin{aligned}
   C_{gas} &= \dot{m}_{gas} \, c_{p,gas} \quad [\text{W/K}] \\
   C_{liq} &= \dot{m}_{H_2O,liq} \, c_{p,H_2O,liq} \quad [\text{W/K}] \\
   C_{quente} &= C_{gas} + C_{liq} \quad [\text{hot side total}] \\
   C_{frio} &= \dot{m}_{ref} \, c_{p,ref} \quad [\text{glycol side}]
   \end{aligned}
   \]
   where \(c_{p,ref}\) for water-glycol mixture (40% glycol by mass):
   \[
   c_{p,ref} = 0.6 \times 4180 + 0.4 \times 2430 = 3480 \, \text{J/(kg·K)}
   \]


2. **Effectiveness Calculation**:
   \[
   \begin{aligned}
   C_{\min} &= \min(C_{quente}, C_{frio}) \\
   R &= \frac{C_{\min}}{C_{\max}} \\
   \text{NTU} &= \frac{U \cdot A}{C_{\min}} \\
   \varepsilon &= \begin{cases}
   \frac{1 - e^{-\text{NTU}(1-R)}}{1 - R \, e^{-\text{NTU}(1-R)}} & R \neq 1 \\
   \frac{\text{NTU}}{1+\text{NTU}} & R = 1
   \end{cases}
   \end{aligned}
   \]


3. **Heat Transfer and Outlet Temperatures**:
   \[
   \begin{aligned}
   Q_{max} &= C_{\min}(T_{hot,in} - T_{cold,in}) \\
   Q_{actual} &= \varepsilon \, Q_{max} \quad [\text{W}] \\
   T_{hot,out} &= T_{hot,in} - \frac{Q_{actual}}{C_{quente}} \\
   T_{cold,out} &= T_{cold,in} + \frac{Q_{actual}}{C_{frio}}
   \end{aligned}
   \]


**Dry Cooler Fan Power**:
\[
P_{fan} = \frac{\dot{V}_{air} \cdot \Delta p_{air}}{\eta_{fan}} = \frac{(\dot{m}_{air}/\rho_{air}) \cdot 500 \, \text{Pa}}{0.60} \quad [\text{W}]
\]


**Returned Values**:
```python
{
    'TC': T_hot_out,           # Gas outlet temp [°C]
    'Pbar': P_in - ΔP_loss,    # Pressure drop [bar]
    'QdotfluxoW': -Q_actual,   # Heat removed (negative)
    'WdotcompW': P_fan,        # Fan electric power [W]
    'TrefoutDCC': T_cold_out,  # Glycol return temp [°C]
}
```
[1]

***

### 3.2 Knock-Out Drum (KOD)
**File**: `modelo_kod.py`[2]

**Function**: Gravity separator for condensate removal with dissolved gas accounting

**Mass Balance**:

1. **Vapor Saturation Limit**:
   \[
   y_{H_2O,sat} = \frac{P_{sat,H_2O}(T)}{P_{total}}
   \]
   where \(P_{sat,H_2O}\) from `calcular_pressao_sublimacao_gelo(T_K)`[3]

2. **Condensate Formation**:
   \[
   \begin{aligned}
   \dot{F}_{H_2O,condense} &= \dot{F}_{gas} \cdot \max(0, \, y_{H_2O,in} - y_{H_2O,sat}) \quad [\text{kmol/s}] \\
   \dot{m}_{H_2O,liq,total} &= \dot{F}_{H_2O,condense} \cdot M_{H_2O} + \dot{m}_{H_2O,liq,in} \quad [\text{kg/s}]
   \end{aligned}
   \]
[2]

3. **Liquid Removal Efficiency**:
   \[
   \dot{m}_{H_2O,drain} = \eta_{removal} \cdot \dot{m}_{H_2O,liq,total} \quad (\eta_{removal} = 0.97)
   \]
[2]

4. **Dissolved Gas Loss** (Henry's Law):
   \[
   \begin{aligned}
   C_{gas,dissolved} &= \text{calcular\_solubilidade\_gas\_henry}(\text{gas}, T, P, y_{gas}) \quad [\text{mg/kg}] \\
   \dot{m}_{gas,diss} &= \dot{m}_{H_2O,drain} \cdot \frac{C_{gas,dissolved}}{10^6} \quad [\text{kg/s}]
   \end{aligned}
   \]
[2]

5. **Main Gas Outlet**:
   \[
   \dot{m}_{gas,out} = \dot{m}_{gas,in} - \dot{m}_{gas,diss}
   \]
[2]

**Pressure Drop and Dimensioning**:
- Fixed \(\Delta P = 0.05\) bar[2]
- Souders-Brown velocity check:
  \[
  V_{max} = K_{SB} \sqrt{\frac{\rho_{liquid} - \rho_{gas}}{\rho_{gas}}} \quad (K_{SB} = 0.08 \, \text{m/s})
  \]
[3]

**Returned Values**:
```python
{
    'TC': T_in,  # Isothermal
    'Pbar': P_in - 0.05,
    'yH2Ooutvap': y_H2O_sat,
    'mdotgasoutkgs': mdot_gas_in - mdot_gas_diss,
    'AguaCondensadaremovidakgs': mdot_H2O_drain + mdot_gas_diss,
    'AguaPuraRemovidaH2Okgs': mdot_H2O_drain,
    'GasDissolvidoremovidokgs': mdot_gas_diss,
    'QdotfluxoW': 0.0,
    'WdotcompW': V_gas_out * ΔP  # Pump surrogate
}
```
[3][1]

***

### 3.3 Coalescer
**File**: `modelo_coalescedor.py`

**Function**: Removes liquid water plus dissolved gas from main gas stream

**Mass Balance** (identical structure to KOD with higher efficiency):

1. **Liquid Removal**:
   \[
   \begin{aligned}
   \dot{m}_{H_2O,drain} &= \eta_{liq} \cdot \dot{m}_{H_2O,liq,in} \quad (\eta_{liq} = 0.98) \\
   \dot{m}_{H_2O,liq,out} &= (1 - \eta_{liq}) \cdot \dot{m}_{H_2O,liq,in}
   \end{aligned}
   \]


2. **Dissolved Gas** (same Henry's Law logic as KOD):
   \[
   \dot{m}_{gas,diss} = \dot{m}_{H_2O,drain} \cdot \frac{C_{gas,dissolved}(T,P)}{10^6}
   \]


3. **Gas Principal Correction**:
   \[
   \dot{m}_{gas,out} = \dot{m}_{gas,in} - \dot{m}_{gas,diss}
   \]


**Fixed Parameters**:
- \(\Delta P = 0.15\) bar (higher than KOD due to filter media)
- Isothermal: \(T_{out} = T_{in}\)
- No external heat exchange: \(Q_{dot,fluxoW} = 0\)

***

### 3.4 Deoxo Reactor (Catalytic O₂ Removal)
**File**: `modelo_deoxo.py`[3]

**Reaction**: 
\[
2H_2 + O_2 \rightarrow 2H_2O \quad \Delta H_{rxn} = -242 \, \text{kJ/mol}_{O_2}
\]
[3]

**Governing ODEs** (adiabatic PFR along length \(L\)):

1. **Mass Balance** (conversion \(X\)):
   \[
   \frac{dX}{dL} = \frac{A_c}{\ dot{F}_{O_2,in}} \cdot r_{O_2}(T, X)
   \]
   where:
   \[
   \begin{aligned}
   r_{O_2} &= k'_{eff}(T) \cdot C_{O_2} \quad [\text{mol/(m}^3\text{·s)}] \\
   k'_{eff} &= k_0 \exp\left(-\frac{E_a}{RT}\right) \quad (k_0 = 10^{10} \, \text{s}^{-1}, E_a = 55 \, \text{kJ/mol}) \\
   C_{O_2} &= \frac{P \cdot y_{O_2,in}(1-X)}{RT}
   \end{aligned}
   \]
[2]

2. **Energy Balance** (adiabatic):
   \[
   \frac{dT}{dL} = \frac{A_c}{\ dot{n}_{total} \, \overline{C_p}} \cdot (-\Delta H_{rxn}) \cdot r_{O_2}
   \]
   where \(\overline{C_p} = 29.5\) J/(mol·K) (mixture average)[2]

**Numerical Solution**:
```python
from scipy.integrate import solve_ivp
sol = solve_ivp(pfr_ode_adiabatico, [0, L_M], [X0, T0], t_eval=Lspan)
```
[3]

**Outlet Properties**:
\[
\begin{aligned}
X_{final} &= X(L) \quad \text{(O₂ conversion)} \\
\dot{F}_{O_2,consumed} &= \dot{F}_{O_2,in} \cdot X_{final} \\
\dot{m}_{H_2O,generated} &= \dot{F}_{O_2,consumed} \cdot M_{H_2O} \quad [\text{kg/s}] \\
y_{O_2,out} &= y_{O_2,in}(1 - X_{final})
\end{aligned}
\]
[3]

**Safety Check**:
\[
T_{max} = \max_{z \in [0,L]} T(z)
\]
Must satisfy: \(T_{max} < \text{LIMITES}[\text{"Deoxo"}]["TMAXC"]\)[2]

***

### 3.5 PSA (Pressure Swing Adsorption)
**File**: `modelo_psa.py`

**Function**: Ultra-dry H₂ purification via H₂O adsorption with H₂ recovery

**Species Mass Balances**:

1. **Molar Flow Decomposition**:
   \[
   \begin{aligned}
   \dot{F}_{mix,in} &= \frac{\dot{m}_{in}}{M_{mix}} \quad [\text{kmol/s}] \\
   \dot{m}_{H_2O,in} &= \dot{F}_{mix,in} \cdot y_{H_2O,in} \cdot M_{H_2O} \\
   \dot{m}_{H_2,in} &= \dot{F}_{mix,in} \cdot y_{H_2,in} \cdot M_{H_2}
   \end{aligned}
   \]


2. **Product and Purge Streams**:
   \[
   \begin{aligned}
   \dot{m}_{H_2,prod} &= \eta_{H_2} \cdot \dot{m}_{H_2,in} \quad (\eta_{H_2} = 0.90) \\
   \dot{m}_{H_2,purga} &= (1 - \eta_{H_2}) \cdot \dot{m}_{H_2,in} \\
   \dot{m}_{O_2,out} &= \dot{m}_{O_2,in} \quad \text{(not adsorbed)}
   \end{aligned}
   \]


3. **Water Removal**:
   \[
   \begin{aligned}
   \dot{m}_{H_2O,ads} &= \dot{m}_{H_2O,in} \quad \text{(complete removal)} \\
   y_{H_2O,out} &= 5 \times 10^{-6} \quad \text{(target purity, 5 ppm)}
   \end{aligned}
   \]


**Adsorbent Inventory**:
\[
\begin{aligned}
M_{H_2O,cycle} &= \dot{m}_{H_2O,in} \cdot T_{cycle} \quad (T_{cycle} = 250 \, \text{s}) \\
\Delta q_{H_2O,eff} &= \Delta q_{static} \cdot f_{MTZ} \quad (0.10 \cdot 0.70 = 0.07 \, \text{kg/kg}) \\
M_{ads,total} &= F_W \cdot \frac{M_{H_2O,cycle}}{\Delta q_{H_2O,eff}} \quad (F_W = 1.5)
\end{aligned}
\]


**Pressure Drop** (Ergun Equation):
\[
\frac{\Delta P}{L} = \frac{150(1-\epsilon)^2 \mu u}{\epsilon^3 d_p^2} + \frac{1.75(1-\epsilon) \rho u^2}{\epsilon^3 d_p}
\]
where \(u = \dot{V}/A_c\) (superficial velocity), \(\epsilon = 0.40\) (bed porosity)

**Compressor/Vacuum Power** (isentropic + efficiency):
\[
\dot{W}_{comp} \approx \frac{\dot{V}_{purga} P_{reg}}{\eta_{comp}} \left[\left(\frac{P_{ads}}{P_{reg}}\right)^{(\kappa-1)/\kappa} - 1\right] \quad (\kappa = 1.4, \eta = 0.75)
\]


***

### 3.6 VSA (Vacuum Swing Adsorption)
**File**: `modelo_vsa.py`

**Function**: Similar to PSA but with vacuum regeneration; used for coarser H₂ drying

**Key Differences from PSA**:
1. Inlet from volumetric flow with real-gas correction:
   \[
   \rho_{in} = \frac{P \cdot M_{mix}}{Z_{H_2,fixed} \cdot R \cdot T} \quad (Z_{H_2} = 1.050)
   \]


2. Compression and Vacuum Stages:
   - **Adsorption**: \(P_{ads} \approx 40\) bar, inlet compressed
   - **Regeneration**: \(P_{regen} \approx 1\) bar, vacuum pump
   
   Energy via enthalpy/entropy (isentropic + efficiency):
   \[
   \begin{aligned}
   w_{comp} &= \frac{h(P_{ads}, s_{in}) - h_{in}}{\eta_{comp}} \\
   w_{vac} &= \frac{h(P_{descarga}, s_{reg}) - h_{reg}}{\eta_{bomba}}
   \end{aligned}
   \]


3. Total Power:
   \[
   P_{total,kW} = \dot{m}_{in} w_{comp}/1000 + \dot{m}_{purga} |w_{vac}|/1000
   \]


***

## 4. Process Execution Flow (`simular_sistema`)

**File**: `process_execution.py`, function `simular_sistema(gasfluido, mdotg_kgs, ...)`[1]

### 4.1 Inlet Water Partitioning
**Critical Logic** (lines defining `MDOTVAPORENTRADAKGS`, `mdotH2Oliqinarraste`, `MDOTDRENOPEMKGS`):

```python
# 1. Saturation limit
y_H2O_sat = calcular_yH2O_inicial(TINC, PINBAR)
estado_sat = calcular_estado_termodinamico(gasfluido, TINC, PINBAR, MDOTGAS, y_H2O_sat, ...)
MDOTVAPORSATKGS = estado_sat['mdotH2Ovapkgs']  # Max vapor carrier capacity

# 2. Demister liquid capacity
MDOTLIQMAXDEMISTERKGS = calcular_vazao_demister(MDOTGAS, MMGAS)
# For O₂, forced to H₂_ref/2.0 if mdotliqmax_h2_ref provided

# 3. Partition total water input
MDOTVAPORENTRADAKGS = min(MH2OTOTALFLUXO, MDOTVAPORSATKGS)  # xy
MDOTLIQTOTALARRASKGS = max(0, MH2OTOTALFLUXO - MDOTVAPORENTRADAKGS)
mdotH2Oliqinarraste = min(MDOTLIQMAXDEMISTERKGS, MDOTLIQTOTALARRASKGS)  # zw
MDOTDRENOPEMKGS = max(0, MH2OTOTALFLUXO - MDOTVAPORENTRADAKGS - mdotH2Oliqinarraste)  # Drain
```
[1]

### 4.2 Component Loop Structure
```python
for comp in component_list_filtered:
    estado_in = history[-1].copy()
    
    # Call component model
    if comp == "KOD 1" or comp == "KOD 2":
        res = modelar_knockout_drum(gasfluido, ...)
    elif comp == "Coalescedor 1":
        res = modelar_coalescedor(gasfluido, ...)
    elif comp == "Dry Cooler 1":
        res = modelar_dry_cooler(gasfluido, ...)
    elif comp == "Chiller 1":
        res = modelar_chiller_gas(gasfluido, ...)
    elif comp == "Deoxo":
        res = modelar_deoxo(...)
    elif comp == "PSA":
        res = modelar_psa(...)
    # ... etc
    
    # Update thermodynamic state
    estado_atual = calcular_estado_termodinamico(
        gasfluido, res['TC'], res['Pbar'], mdot_gas_out, y_H2O_out, ...
    )
    
    # Append to history
    history.append({
        **estado_atual,
        'Componente': comp,
        'QdotfluxoW': res.get('QdotfluxoW', 0.0),
        'WdotcompW': res.get('WdotcompW', 0.0),
        'AguaCondensadakgs': ...,
        'AguaPuraRemovidaH2Okgs': ...,
        # ... etc
    })
```
[1]

### 4.3 Water Pool Tracking
Each component updates a "pool" variable representing available liquid water:

```python
mdotH2Oliquidapool_out = mdotH2Oliquidapool_in
# If component generates water (e.g., Deoxo):
mdotH2Oliquidapool_out += AguaGeradaDeoxo
# If component removes water (e.g., KOD, Coalescedor):
mdotH2Oliquidapool_out -= res['AguaPuraRemovidaH2Okgs']
# Ensure non-negative:
mdotH2Oliquidapool_out = max(0.0, mdotH2Oliquidapool_out)
```
[1]

***

## 5. DataFrame Column Mapping

### 5.1 Thermodynamic State Columns
From `aux_coolprop.calcular_estado_termodinamico()`:

| Column | Type | Units | Description | Source |
|--------|------|-------|-------------|--------|
| `TC` | float | °C | Bulk gas temperature | CoolProp or component model |
| `Pbar` | float | bar | Total pressure | Inlet - Σ(ΔP) |
| `gasfluido` | str | - | "H2" or "O2" | Config |
| `yH2O` | float | - | Water molar fraction | min(y_H2O_target, y_sat) |
| `yO2` | float | - | O₂ molar fraction | Deoxo reaction or inlet |
| `yH2` | float | - | H₂ molar fraction | 1 - yH2O - yO2 |
| `wH2O` | float | - | Water mass fraction | F_H2O·M_H2O / mdot_mix |
| `mdotgaskgs` | float | kg/s | Main gas (H₂ or O₂) flow | Inlet - Σ(dissolved losses) |
| `mdotmixkgs` | float | kg/s | Total mixture (gas+vapor) | mdot_gas + mdot_vapor |
| `mdotH2Ovapkgs` | float | kg/s | Water vapor flow | F_H2O_molar · M_H2O |
| `HmixJkg` | float | J/kg | Mixture specific enthalpy | Σ(F_i·M_i·h_i) / mdot_mix |

### 5.2 Water and Drain Columns
From component models:[11][2]

| Column | Type | Units | Component(s) | Description |
|--------|------|-------|--------------|-------------|
| `AguaPuraRemovidaH2Okgs` | float | kg/s | KOD, Coalescedor, PSA | Pure liquid water drain (no dissolved gas) |
| `AguaCondensadakgs` | float | kg/s | KOD, Coalescedor, Chiller, PSA | Total water removed (for plots) |
| `AguaLiquidaResidualoutkgs` | float | kg/s | Coalescedor | Residual liquid carryover |
| `mdotH2Oliqaccompkgs` | float | kg/s | All | Liquid water accompanying gas |
| `mdotH2Oliqpoolkgs` | float | kg/s | All | Accumulated liquid inventory |
| `GasDissolvidoremovidokgs` | float | kg/s | KOD, Coalescedor, PSA | Main gas lost to dissolution or purge |

### 5.3 Energy Columns
From component models:[8][11]

| Column | Type | Units | Component(s) | Sign Convention |
|--------|------|-------|--------------|----------------|
| `QdotfluxoW` | float | W | Dry Cooler, Chiller, Deoxo | Negative = heat removed from gas |
| `WdotcompW` | float | W | Dry Cooler (fan), PSA, VSA, KOD | Positive = shaft/electric power |
| `QdotH2Gas` | float | W | Dry Cooler, Chiller | Sensible heat of dry gas only |
| `QdotH2OTotal` | float | W | Dry Cooler, Chiller | Latent + sensible heat of H₂O |

### 5.4 Deoxo-Specific Columns
From `modelo_deoxo.py`:[3]

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `Lspan` | array | m | Axial position array [0...L] |
| `TprofileC` | array | °C | Temperature profile T(z) |
| `XO2` | float | - | Final O₂ conversion (0 to 1) |
| `Tmaxcalc` | float | °C | Peak reactor temperature |
| `AguaGeradakgs` | float | kg/s | Water generated by reaction |

***

## 6. Energy and Water Balance Visualization

### 6.1 Energy Flow Plot
**File**: `plot_fluxos_energia.py`[4]

**Data Extraction**:
```python
df_plot_h2 = df_h2[df_h2['Componente'] != 'Entrada'].copy()
comp_labels_h2 = df_plot_h2['Componente'].tolist()

# Thermal power per component
q_fluxo_h2 = df_plot_h2['QdotfluxoW'] / 1000  # Convert to kW

# Electric power per component
w_comp_h2 = df_plot_h2['WdotcompW'] / 1000  # Convert to kW
```
[4]

**Visualization**:
- **Top subplot**: Thermal power exchanged (Q̇) by component, stacked bars for H₂ vs O₂
- **Bottom subplot**: Electric power consumed (Ẇ) by component, stacked bars
- X-axis: Component names in sequence
- Color coding: Blue (H₂), Red (O₂)

[4]

### 6.2 Water Removal Plot
**File**: `plot_agua_removida_total.py`[5]

**Data Extraction**:
```python
comp_remover = ['KOD 1', 'KOD 2', 'Coalescedor 1', 'VSA', 'PSA']

remocao_h2 = {
    comp: df_h2[df_h2['Componente']==comp]['AguaCondensadakgs'].iloc[0] * 3600
    for comp in comp_remover if comp in df_h2['Componente'].values
}
```
[5]

**Visualization**:
- Grouped bar chart per component
- Y-axis: Water mass flow removed [kg/h]
- Conversion factor: `* 3600` to go from kg/s to kg/h

[5]

### 6.3 Water Balance Closure
**Function**: `exibir_balanco_agua_inicial(df_h2, df_o2)`

**Logic** (pseudo-code based on typical balance):
```python
# Inlet water
inlet_vapor = df['mdotH2Ovapkgs'].iloc[0]  # At 'Entrada' row
inlet_liquid = MDOT_H2O_LIQ_IN_..._TOTAL_KGS  # From constants
total_inlet = inlet_vapor + inlet_liquid

# Removed water
total_removed = df['AguaPuraRemovidaH2Okgs'].sum() + \
                df['AguaCondensadakgs'].sum()

# Residual water
final_vapor = df['mdotH2Ovapkgs'].iloc[-1]
final_liquid = df['mdotH2Oliqaccompkgs'].iloc[-1]
total_residual = final_vapor + final_liquid

# Check closure
closure_error = abs(total_inlet - total_removed - total_residual)
```


***

## 7. Drain Line and Recirculation

### 7.1 Drain Extraction
**Function**: `extrair_dados_dreno(df, gasfluido)` in `drain_mixer.py`

**Extraction Logic**:
```python
drenos_list = []
# PEM drain
mdot_dreno_pem = df[df['Componente']=='Entrada']['AguaPuraRemovidaH2Okgs'].iloc[0]
if mdot_dreno_pem > 0:
    drenos_list.append({
        'Componente': 'PEM Dreno Recirc.',
        'mdot': mdot_dreno_pem,  # kg/s
        'T': T_pem,
        'Pbar': P_pem,
        'GasDissolvido_in_mgkg': 0.0  # Assumed pure at PEM
    })

# KOD 1, KOD 2 drains
for comp in ['KOD 1', 'KOD 2']:
    if comp in df['Componente'].values:
        comp_data = df[df['Componente']==comp].iloc[0]
        mdot = comp_data['AguaPuraRemovidaH2Okgs']
        if mdot > 0:
            C_diss = calcular_solubilidade_gas_henry(gasfluido, T, Pbar, y_gas)
            drenos_list.append({
                'Componente': comp,
                'mdot': mdot,
                'T': comp_data['TC'],
                'Pbar': comp_data['Pbar'],
                'GasDissolvido_in_mgkg': C_diss
            })
```


### 7.2 Valve + Tank Vent (Flash Drum)
**Function**: `simular_linha_dreno(drenos_list, gasfluido, P_alvo_bar)`

**Steps**:

1. **Mass-Weighted Mixing**:
   \[
   \begin{aligned}
   \dot{m}_{tot} &= \sum_i \dot{m}_i \\
   T_{in} &= \frac{\sum_i \dot{m}_i T_i}{\dot{m}_{tot}} \\
   P_{in} &= \min_i P_i \\
   C_{in} &= \frac{\sum_i \dot{m}_i C_i}{\dot{m}_{tot}}
   \end{aligned}
   \]


2. **Valve** (isothermal throttling):
   \[
   T_{out} = T_{in}, \quad P_{out} = P_{alvo}
   \]


3. **Tank Vent** (degassing):
   - Uses `FlashDrumModel` to compute new dissolved concentration \(C_{out}\) based on lower pressure and residence time
   - Efficiency typically 95% removal of dissolved gas

**Outputs**:
- `entrada_dreno`: State before valve/drum (aggregated)
- `saida_dreno`: State after degassing



### 7.3 Final Mixer and Makeup Water
**Function**: `simular_reposicao_agua(dreno_final_state, ...)`

**Logic**:
```python
mdot_dreno = dreno_final_state['Mdot_H2Ofinal_kgs']
mdot_target = MDOT_H2O_RECIRC_TOTAL_KGS  # 69.444 kg/s (250 t/h)
mdot_makeup = mdot_target - mdot_dreno

if mdot_makeup > 0:
    # Mix drain @ T_dreno with makeup @ 20°C
    input_streams = [
        {'mdot': mdot_dreno, 'T': T_dreno, 'P': P_out},
        {'mdot': mdot_makeup, 'T': 20.0, 'P': P_out}
    ]
    _, output = modelo_mixer(input_streams, P_out)
    T_out = output['T3']
    H_out = output['h3']
else:
    # No makeup needed
    T_out = T_dreno
    H_out = H_dreno
```


**Final State**:
```python
estado_recirculacao_final = {
    'ToutC': T_out,
    'Poutbar': P_out,
    'Mdot_out_kgs': mdot_target,
    'Hout_Jkg': H_out,
    'Mdot_makeup_kgs': mdot_makeup
}
```


***

This comprehensive documentation maps the entire simulator structure based on the actual code. Each section references specific files and line logic from your codebase. Would you like me to expand any particular section further?
