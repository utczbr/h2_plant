# TECHNICAL REVIEW: Thermal Inertia & Flow Dynamics Integration
## Current System Architecture vs. Proposed 1-Minute Dynamics Model

**Date:** November 25, 2025  
**Scope:** Detailed code-level analysis of existing components and required modifications  
**Focus:** Thermal management, flow dynamics, and architectural considerations for 1-minute timestep  

---

## EXECUTIVE SUMMARY: CRITICAL GAPS & MODIFICATIONS REQUIRED

### Current System Status (STEADY-STATE ONLY)
Your codebase currently implements **instant equilibrium** across all thermal/flow operations:

| Component | Current Behavior | Problem | Impact |
|-----------|-----------------|---------|--------|
| **Chiller** | Instant temperature drop to setpoint | No startup transient | ±20% cooling cost error |
| **PEM/SOEC Stack** | Hardcoded T = constant | No thermal dynamics | Efficiency locked at nominal |
| **Pump/Compressor** | Instant flow response | No ramp-up penalty | Grid compliance risk |
| **H2 Storage Tank** | Static pressure (ideal gas only) | No accumulation | Pressure spikes undetected |
| **Heat Exchanger** | Instantaneous heat transfer | No effectiveness model | Underestimates duty |
| **Thermal Manager** | Simple algebraic balance | No flow/time dynamics | Cannot model transients |

### Required Architecture Changes: 3 Tiers

**TIER 1 (Must Have):** Insert ThermalInertiaModel into PEM/SOEC stacks  
**TIER 2 (Should Have):** Add PumpFlowDynamics to cooling/compression loops  
**TIER 3 (Nice to Have):** Enhance HeatExchanger with NTU/LMTD effectiveness model  

---

## PART 1: CURRENT SYSTEM DEEP DIVE

### 1.1 PEM Electrolyzer Current Implementation

**File:** `pem_electrolyzer_detailed.py` & `main_pem_simulator_S.py`

**Current Thermal Model:**
```python
# CURRENT CODE (static temperature)
self.T = 333.15  # Hardcoded 60°C
self.efficiency = 0.60  # Constant efficiency

# Efficiency calculation (ignores actual temperature evolution)
cell_voltage = 1.8  # Fixed operating voltage
overpotential = cell_voltage - 1.26  # Nernst potential
heat_generated = (cell_voltage - 1.26) * I  # Heat per Faraday law

# BUT: Temperature never changes!
# No energy balance dT/dt = (Q_in - Q_out - Q_loss) / C
```

**Problems:**
1. ✗ **No thermal inertia:** Stack reaches 333.15 K instantly at t=0
2. ✗ **Efficiency is decoupled from T:** Uses constant 0.60, not temperature-dependent
3. ✗ **No startup losses:** Cold stack treated as hot stack immediately
4. ✗ **No efficiency degradation:** Temperature rise doesn't reduce efficiency over time
5. ✗ **No upper temperature limit:** Boiling point (373 K for water cooling) not enforced

**Required Modification:**
```python
# NEW CODE (with thermal inertia)
# In __init__:
from thermal_inertia_model import ThermalInertiaModel

self.thermal_model = ThermalInertiaModel(
    C_thermal_J_K=2.6e6,      # Stack + coolant mass
    h_A_passive_W_K=100.0,    # Natural convection coeff × area
    T_ambient_K=298.15,       # 25°C ambient
    T_initial_K=298.15,       # START COLD!
    max_cooling_kw=100.0      # Chiller capacity
)

# In step() method (dt_hours = 1/60.0 for 1-minute):
dt_s = self.dt_hours * 3600  # Convert to seconds

# 1. Calculate heat generation
U_rev = nernst_voltage(self.T)  # Temperature-dependent
cell_voltage = U_rev + overpotential
Q_in_W = (cell_voltage - U_rev) * I

# 2. Advance thermal model
T_new_K = self.thermal_model.step(
    dt_s=dt_s,
    heat_generated_W=Q_in_W,
    T_control_K=333.15  # Setpoint from chiller
)

# 3. Update efficiency based on ACTUAL temperature
self.efficiency = efficiency_from_temperature(T_new_K, I)
self.T_K = T_new_K
```

**Architecture Consideration:** Temperature must now be **state variable** that evolves over time, not a constant. This affects:
- Energy balance calculations (now coupled to T)
- Efficiency polynomials (T-dependent, not constant)
- Startup scenarios (cold start penalty visible)

---

### 1.2 SOEC Electrolyzer Current Implementation

**File:** `soec_electrolyzer_detailed.py` & `soec_operator.py`

**Current Thermal Model:**
```python
# CURRENT: Operates at fixed high temperature
class SOECStackArrayComponent:
    def __init__(self):
        # No temperature model!
        self.max_power_kw = 1000
        self.efficiency = 0.9  # Constant
        self.steam_input_kgh = 0
        self.power_input_kw = 0

    def step(self, t):
        # Heat demand calculated but NOT integrated over time
        if self.power_input_kw > 0:
            h2_potential = self.power_input_kw / SPECIFIC_ENERGY  # kWh/kg
            actual_h2 = min(h2_potential, max_h2_from_steam)
        
        # Temperature implicitly 800°C (steam generator setpoint)
        # BUT: No model for temperature transient when power changes
```

**Problems:**
1. ✗ **SOEC assumes instant thermal equilibrium at 800°C:** No startup time modeled
2. ✗ **Steam requirement is instantaneous:** No account for heating time
3. ✗ **No thermal coupling:** Chiller/cooler dynamics don't affect SOEC operation
4. ✗ **SteamGenerator component calculates heat demand but doesn't track its own temperature**

**Current SteamGenerator Logic:**
```python
class SteamGeneratorComponent:
    def step(self, t):
        # Calculate DEMAND, not the actual thermal response
        if steam_output_kgh > 0:
            q_water = m_dot * Cp_water * (100 - 25)  # Sensible heat
            q_latent = m_dot * latent_heat_of_vaporization
            q_steam = m_dot * Cp_steam * (target_temp_c - 100)
            self.total_heat_demand_kw = (q_water + q_latent + q_steam)
        
        # Apply external heat, but no time integral
        effective_external_heat = min(self.external_heat_input_kw, self.total_heat_demand_kw)
        
        # Assumption: If external_heat >= demand, instantly satisfied
        # WRONG! There's a heat transfer delay!
```

**Required Modification:**
```python
# NEW CODE for SOEC with thermal inertia
from thermal_inertia_model import ThermalInertiaModel

class DetailedSOECElectrolyzer:
    def __init__(self):
        # Create thermal model for SOEC stack (higher temp)
        self.thermal_model = ThermalInertiaModel(
            C_thermal_J_K=1.5e6,      # Smaller than PEM, different material
            h_A_passive_W_K=80.0,     # Lower convection at high temp
            T_ambient_K=298.15,       # Ambient still 25°C
            T_initial_K=473.15,       # Start at 200°C (room temp + some preheating)
            max_cooling_kw=50.0       # SOEC doesn't cool, preheats input
        )
        
        # Create thermal model for steam generation
        self.steam_thermal_model = ThermalInertiaModel(
            C_thermal_J_K=0.5e6,      # Smaller water thermal mass
            h_A_passive_W_K=30.0,
            T_ambient_K=298.15,
            T_initial_K=323.15,       # Start at 50°C (cold water)
            max_cooling_kw=0.0        # No cooling, only heating
        )

    def step(self, t):
        dt_s = self.dt_hours * 3600
        
        # 1. Advance SOEC stack temperature
        # Heat from electrical resistance + entropy change
        Q_in_soec = self.soec_stacks.power_input_kw * 1000  # Converted to W
        
        T_soec_K = self.thermal_model.step(
            dt_s=dt_s,
            heat_generated_W=Q_in_soec,
            T_control_K=473.15  # SOEC operating temperature
        )
        self.soec_stacks.T_K = T_soec_K
        
        # 2. Advance steam temperature (separate thermal loop)
        # Heat from external source or SOEC outlet gas
        Q_in_steam = self.steam_gen_hx4.external_heat_input_kw * 1000
        
        T_steam_K = self.steam_thermal_model.step(
            dt_s=dt_s,
            heat_generated_W=Q_in_steam,
            T_control_K=373.15  # Steam generation target
        )
        self.steam_gen_hx4.T_K = T_steam_K
        
        # 3. Update efficiency based on actual SOEC temperature
        self.soec_stacks.efficiency = efficiency_from_temp(T_soec_K)
```

**Architecture Consideration:** SOEC requires **dual thermal models** (stack + steam generator), adding complexity to orchestration.

---

### 1.3 Chiller Current Implementation

**File:** `chiller.py`

**Current Code:**
```python
class Chiller(Component):
    def step(self, t):
        if self.inlet_stream.mass_flow_kgh <= 0:
            self.outlet_stream = Stream(0.0)
            return
        
        # 1. Calculate required cooling
        mass_flow_kg_s = self.inlet_stream.mass_flow_kgh / 3600.0
        Cp = 4.18  # kJ/kg·K (water)
        temp_delta_K = max(0, self.inlet_stream.temperature_k - self.target_outlet_k)
        required_cooling_kw = mass_flow_kg_s * Cp * temp_delta_K
        
        # 2. INSTANT application of cooling (WRONG!)
        actual_cooling_kw = min(required_cooling_kw, self.cooling_capacity_kw * self.efficiency)
        
        # 3. Calculate outlet temperature
        if mass_flow_kg_s > 0:
            actual_temp_drop = actual_cooling_kw / (mass_flow_kg_s * Cp)
        else:
            actual_temp_drop = 0
        
        outlet_temp = self.inlet_stream.temperature_k - actual_temp_drop
        
        # PROBLEM: All cooling happens in one step!
        # No transient: ramp-up, no pump inertia, no heat exchanger effectiveness
        self.outlet_stream = Stream(
            mass_flow_kgh=self.inlet_stream.mass_flow_kgh,
            temperature_k=outlet_temp,
            pressure_pa=self.inlet_stream.pressure_pa
        )
```

**Problems:**
1. ✗ **No pump dynamics:** Flow is instant, not ramped
2. ✗ **No heat exchanger effectiveness:** Assumes perfect NTU = ∞
3. ✗ **No chiller startup time:** Cold inlet water takes time to reach cooling capacity
4. ✗ **No compressor power:** Pump power drawn but not modeled dynamically
5. ✗ **Cooling capacity applied instantly:** No feedback from compressor speed

**Required Modification:**
```python
# NEW CODE: Chiller with flow dynamics
from thermal_inertia_model import ThermalInertiaModel
from flow_dynamics_model import PumpFlowDynamics

class ChillerEnhanced(Component):
    def __init__(self, component_id, cooling_capacity_kw=100):
        super().__init__()
        self.component_id = component_id
        self.cooling_capacity_kw = cooling_capacity_kw
        
        # Add pump dynamics model
        self.pump = PumpFlowDynamics(
            pump_shutoff_pa=350000,      # 3.5 bar
            pump_resistance_pa_per_m6_h2=3.5e5,
            system_resistance_pa_per_m6_h2=2.0e5,
            fluid_inertance_kg_m4=1e4,
            initial_flow_m3_h=10.0,
            rho_kg_m3=1000.0
        )
        
        # Add coolant thermal model (separate from stack!)
        self.coolant_thermal = ThermalInertiaModel(
            C_thermal_J_K=1.0e6,      # Coolant mass in loop
            h_A_passive_W_K=50.0,     # Heat exchanger surface
            T_ambient_K=298.15,       # Ambient
            T_initial_K=293.15,       # Start cold
            max_cooling_kw=cooling_capacity_kw
        )
        
        self.inlet_stream = None
        self.outlet_stream = None
        self.outlet_setpoint_k = 333.15  # 60°C target

    def step(self, t):
        super().step(t)
        dt_s = self.dt * 3600  # 1-minute steps: dt = 1/60 hour
        
        if self.inlet_stream is None or self.inlet_stream.mass_flow_kgh <= 0:
            self.outlet_stream = Stream(0.0)
            return
        
        # 1. Control pump speed based on inlet temperature error
        temp_error = self.inlet_stream.temperature_k - self.outlet_setpoint_k
        pump_speed = np.clip(temp_error / 30.0, 0, 1)  # 30K error → full speed
        
        # 2. Advance pump with dynamics
        Q_cool_m3_h = self.pump.step(dt_s=dt_s, pump_speed_fraction=pump_speed)
        
        # 3. Advance coolant temperature
        # Heat absorbed = cooling demanded (simplified)
        Q_absorbed_W = self.cooling_capacity_kw * 1000 * min(1.0, temp_error / 10)
        
        T_coolant_K = self.coolant_thermal.step(
            dt_s=dt_s,
            heat_generated_W=Q_absorbed_W,
            T_control_K=self.outlet_setpoint_k
        )
        
        # 4. Calculate outlet temperature (bounded by coolant temp)
        # Effectiveness = 1 - exp(-NTU), NTU depends on flow rate
        NTU = 0.8  # Design parameter
        effectiveness = 1 - np.exp(-NTU)
        
        mass_flow_kg_s = Q_cool_m3_h / 3.6
        Cp = 4.18
        
        if mass_flow_kg_s > 0:
            Q_max_W = mass_flow_kg_s * Cp * 20 * 1000  # Max 20K drop
            Q_actual_W = min(Q_max_W * effectiveness, Q_absorbed_W)
            outlet_temp_drop = Q_actual_W / (mass_flow_kg_s * Cp)
        else:
            outlet_temp_drop = 0
        
        outlet_temp = max(T_coolant_K, self.inlet_stream.temperature_k - outlet_temp_drop)
        
        # 5. Create outlet stream
        self.outlet_stream = Stream(
            mass_flow_kgh=Q_cool_m3_h * 1000,  # kg/h
            temperature_k=outlet_temp,
            pressure_pa=self.inlet_stream.pressure_pa
        )

    def get_state(self):
        return {
            **super().get_state(),
            'outlet_temp_k': self.outlet_stream.temperature_k if self.outlet_stream else 0,
            'pump_flow_m3_h': self.pump.Q_m3_h,
            'pump_pressure_pa': self.pump.p_pump_pa,
            'coolant_temp_k': self.coolant_thermal.T_K
        }
```

**Architecture Considerations:**
- **New dependencies:** PumpFlowDynamics, ThermalInertiaModel
- **State tracking:** Coolant temperature is now independent variable (not just inlet)
- **Pump coupling:** Pump speed must be driven by control logic (error-based proportional control)
- **Feedback loop:** Outlet temperature affects inlet to PEM (thermal coupling)

---

### 1.4 Storage Tank & Pressure Accumulation Current Implementation

**File:** `storage.py`

**Current Code:**
```python
class H2StorageTank(Component):
    def __init__(self, tank_id, volume_m3=1.0, initial_pressure_bar=40):
        self.tank_id = tank_id
        self.volume_m3 = volume_m3
        self.initial_pressure_bar = initial_pressure_bar
        
        # Static pressure calculation only!
        self.pressure_pa = initial_pressure_bar * 1e5
        self.mass_kg = (self.pressure_pa * self.volume_m3) / (R_H2 * self.T_tank)

    def step(self, t):
        # Updates mass, then pressure (static calculation)
        self.mass_kg += self.mass_inflow_kg_s * self.dt - self.mass_outflow_kg_s * self.dt
        
        # INSTANT pressure update (NO dynamics!)
        self.pressure_pa = (self.mass_kg * R_H2 * self.T_tank) / self.volume_m3
        
        # PROBLEM: If you suddenly consume 1 kg/s, pressure drops instantly
        # Real system: pressure changes gradually due to gas inertia
```

**Problems:**
1. ✗ **No pressure transient:** Pressure changes instantly with mass
2. ✗ **No accumulator dynamics:** dP/dt = (RT/V) * (ṁ_in - ṁ_out) not integrated
3. ✗ **No pressure relief modeling:** Overpressure events not visible
4. ✗ **No stratification:** Assumes uniform pressure/temperature throughout tank

**Required Modification:**
```python
# NEW CODE: Storage tank with accumulator dynamics
from flow_dynamics_model import GasAccumulatorDynamics

class H2StorageTankEnhanced(Component):
    def __init__(self, tank_id, volume_m3=1.0, initial_pressure_bar=40.0):
        super().__init__()
        self.tank_id = tank_id
        self.volume_m3 = volume_m3
        
        # Add accumulator model
        self.accumulator = GasAccumulatorDynamics(
            V_tank_m3=volume_m3,
            T_tank_k=298.15,            # Isothermal assumption (cooled)
            initial_pressure_pa=initial_pressure_bar * 1e5,
            R_gas_j_kg_k=4124.0         # H2 specific gas constant
        )
        
        # Track flows
        self.m_dot_in_kg_s = 0.0
        self.m_dot_out_kg_s = 0.0

    def step(self, t):
        super().step(t)
        dt_s = self.dt * 3600  # 1-minute: dt = 1/60 hour
        
        # 1. Get flow inputs from connected components
        # (must be set by manager before calling step)
        
        # 2. Advance pressure using gas dynamics
        P_new_pa = self.accumulator.step(
            dt_s=dt_s,
            m_dot_in_kg_s=self.m_dot_in_kg_s,
            m_dot_out_kg_s=self.m_dot_out_kg_s
        )
        
        # 3. Update state
        self.pressure_pa = P_new_pa
        self.pressure_bar = P_new_pa / 1e5
        self.mass_kg = self.accumulator.M_kg
        self.fill_fraction = self.mass_kg / (1000 * self.volume_m3)  # density ~1000 kg/m³ at STP
        
    def receive_h2_input(self, mass_flow_kg_s):
        """Called by compressor output."""
        self.m_dot_in_kg_s = mass_flow_kg_s

    def extract_h2_output(self, mass_flow_kg_s):
        """Called by fuel cell/user demand."""
        self.m_dot_out_kg_s = mass_flow_kg_s

    def get_state(self):
        return {
            **super().get_state(),
            'pressure_pa': self.accumulator.P,
            'pressure_bar': self.accumulator.P / 1e5,
            'mass_kg': self.accumulator.M_kg,
            'density_kg_m3': self.accumulator.M_kg / self.volume_m3,
            'fill_fraction': self.fill_fraction,
            'flow_in_kg_s': self.m_dot_in_kg_s,
            'flow_out_kg_s': self.m_dot_out_kg_s
        }
```

**Architecture Considerations:**
- **Flow coupling:** m_dot_in and m_dot_out must be set by external orchestration (manager)
- **Pressure is now state variable:** Evolves over time, not calculated instantaneously
- **Stratification risk:** Model assumes uniform T & P; real tanks have gradients

---

### 1.5 Thermal Manager Current Implementation

**File:** `thermal_manager.py`

**Current Code:**
```python
class ThermalManager(Component):
    def step(self, t):
        # Collect waste heat from sources
        self.total_heat_available_kw = 0.0
        try:
            pem = self.get_registry_safe(ComponentID.PEM_ELECTROLYZER_DETAILED)
            if hasattr(pem, 'heat_output_kw'):
                self.total_heat_available_kw = pem.heat_output_kw
        except:
            pass
        
        # Find heat demand
        self.total_heat_demand_kw = 0.0
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            if hasattr(soec, 'steam_gen_hx4'):
                steam_gen = soec.steam_gen_hx4
                if hasattr(steam_gen, 'total_heat_demand_kw'):
                    self.total_heat_demand_kw = steam_gen.total_heat_demand_kw
        except:
            pass
        
        # Simple algebraic balance (WRONG!)
        self.heat_utilized_kw = min(self.total_heat_available_kw, self.total_heat_demand_kw)
        self.heat_wasted_kw = max(0, self.total_heat_available_kw - self.heat_utilized_kw)
        
        # Set external heat on sink
        if steam_gen:
            steam_gen.external_heat_input_kw = self.heat_utilized_kw
```

**Problems:**
1. ✗ **No heat transport delay:** Heat flows instantly from source to sink
2. ✗ **No heat exchanger model:** Assumes direct thermal coupling (wrong!)
3. ✗ **No accumulation in pipes:** Piping thermal mass ignored
4. ✗ **No pressure drop consideration:** Flow doesn't affect heat transfer rate
5. ✗ **No feedback from down stream:** Sink temperature doesn't affect source cooling

**Required Modification:**
```python
# NEW CODE: Thermal manager with flow-dependent heat transfer
class ThermalManagerEnhanced(Component):
    def __init__(self):
        super().__init__()
        
        # Create heat exchanger model for PEM-to-SOEC transfer
        self.pem_soec_hx = HeatExchangerDynamics(
            component_id="PEM_to_SOEC_HX",
            max_heat_transfer_kw=200.0,
            NTU=0.8,  # Number of transfer units
            C_thermal_J_K=5e5  # Pipe + fluid thermal mass
        )
        
    def step(self, t):
        super().step(t)
        dt_s = self.dt * 3600
        
        # 1. Get source heat and flow
        try:
            pem = self.get_registry_safe(ComponentID.PEM_ELECTROLYZER_DETAILED)
            Q_available_W = pem.heat_output_kw * 1000
            T_hot_K = pem.T_K
        except:
            Q_available_W = 0
            T_hot_K = 298.15
        
        # 2. Get sink demand and flow
        try:
            soec = self.get_registry_safe(ComponentID.SOEC_CLUSTER)
            steam_gen = soec.steam_gen_hx4
            Q_demanded_W = steam_gen.total_heat_demand_kw * 1000
            T_cold_K = steam_gen.T_K
        except:
            Q_demanded_W = 0
            T_cold_K = 298.15
        
        # 3. Advance heat exchanger with dynamics
        Q_transferred_W = self.pem_soec_hx.step(
            dt_s=dt_s,
            Q_in_hot_W=Q_available_W,
            T_hot_K=T_hot_K,
            T_cold_K=T_cold_K,
            effectiveness=0.75  # Empirical HX effectiveness
        )
        
        # 4. Set on sink
        steam_gen.external_heat_input_kw = Q_transferred_W / 1000
        
        # 5. Account for heat carried away by cooling stream
        # (Reduces PEM available heat)
        pem.heat_removed_by_cooling_kw = self.pem_soec_hx.Q_out_W / 1000

    def get_state(self):
        return {
            **super().get_state(),
            'heat_transferred_kw': self.pem_soec_hx.Q_transferred_W / 1000,
            'hx_hot_side_temp_k': self.pem_soec_hx.T_hot_K,
            'hx_cold_side_temp_k': self.pem_soec_hx.T_cold_K,
            'hx_lmtd_k': self.pem_soec_hx.LMTD_K
        }
```

**Architecture Considerations:**
- **New heat exchanger model needed:** Effectiveness-NTU or LMTD method
- **Feedback coupling:** PEM heat output now affected by cooling feedback
- **Thermal mass in pipes:** Adds another time constant (~5-10 minutes)

---

## PART 2: ARCHITECTURAL MODIFICATIONS REQUIRED

### 2.1 CRITICAL: 1-Minute Timestep Integration Pattern

**Main Loop Modification (manager.py):**

```python
# CURRENT CODE (hourly):
for hour in range(8760):
    t_hour = hour
    self.step_simulation(t_hour)  # dt = 1 hour internally

# NEW CODE (1-minute):
for minute in range(525600):  # 8760 hours × 60 minutes
    t_hour = minute / 60.0  # Convert to fractional hours
    self.step_simulation(t_hour)
    
    # Sparse logging: save every 60th step (hourly data)
    if minute % 60 == 0:
        results.append(self.get_state())
```

**Each Component.step() Must Convert:**
```python
def step(self, t_hour):
    dt_s = 60  # FIXED: 1 minute = 60 seconds
    
    # NOT: dt_s = self.dt * 3600  (dt_hour already 1/60)
    # Instead: dt_s = 60  (literal constant for 1-minute steps)
    
    # Advance ALL thermal/flow models
    T_new = self.thermal_model.step(dt_s=60, Q_in=...)
    Q_new = self.pump.step(dt_s=60, speed=...)
    P_new = self.accumulator.step(dt_s=60, m_dot_in=...)
```

**CRITICAL DETAIL:** The `dt` parameter in each `step()` method must be in **seconds** for thermal models to work correctly. Forward Euler stability depends on dt << τ.

---

### 2.2 Causal Execution Order in Main Loop

**The orchestration order matters for coupled systems:**

```python
# CORRECT ORDER (thermal feedback):
for minute in range(525600):
    t_hour = minute / 60.0
    
    # 1. FIRST: Calculate electrical power demands (external input)
    pem.P_setpoint_mw = market_signal.get_power_setpoint(t_hour)
    soec.P_setpoint_mw = market_signal.get_power_setpoint(t_hour)
    
    # 2. SECOND: All electrochemistry steps (generates heat)
    pem.step(t_hour)  # Sets pem.T_K, pem.heat_output_kw, pem.efficiency
    soec.step(t_hour)  # Sets soec.T_K, soec.heat_output_kw
    
    # 3. THIRD: Thermal manager distributes heat
    thermal_manager.step(t_hour)  # Transfers heat PEM → SOEC
    
    # 4. FOURTH: Cooling systems respond to temperature
    chiller.inlet_stream = pem.outlet_coolant_stream
    chiller.step(t_hour)  # Cools PEM based on T
    
    # 5. FIFTH: Storage accumulates/depletes
    storage.receive_h2_input(pem.h2_production_kg_s)
    storage.extract_h2_output(market.h2_demand_kg_s)
    storage.step(t_hour)  # Pressure evolves

# WRONG ORDER:
for minute in range(525600):
    thermal_manager.step()  # Heat not yet calculated!
    pem.step()              # Uses yesterday's temperature!
    chiller.step()          # Cooling based on stale flow!
```

**Why This Matters:**
- Each component depends on **previous state** of its dependencies
- 60-second coupling can cascade into larger errors if ordering wrong
- Must respect causality: power → heat → thermal response → cooling

---

### 2.3 Data Structure: State Tracking for Transients

**Current system is stateless within hour; new system must track state:**

```python
# CURRENT (inadequate for 1-min):
class PEMElectrolyzer:
    def __init__(self):
        self.T = 333.15  # Single value
        self.efficiency = 0.60

# NEW (for 1-min dynamics):
class PEMElectroly zerEnhanced:
    def __init__(self):
        # State variables (evolve over time)
        self.thermal_model = ThermalInertiaModel(...)  # Internal state
        self.T_K = 298.15  # Current temperature (state)
        self.efficiency = 0.65  # Updated each step (not constant)
        
        # History for diagnostics (optional)
        self.T_history = [298.15]
        self.efficiency_history = [0.65]
        self.heat_output_history = [0]
```

**Memory implications for 8760-hour annual run:**
- 525,600 minutes per year
- ~20 state variables per component
- ~50 major components
- Full history: 525,600 × 20 × 50 = **526 million floats** = **2.1 GB RAM**
- **Solution:** Sparse logging (every 60th step) = **20 MB** ✓

---

## PART 3: CODE-LEVEL MODIFICATIONS CHECKLIST

### Tier 1: MUST DO (Enables 1-minute simulation)

**[ ] Insert ThermalInertiaModel into PEM:**
```python
# File: pem_electrolyzer_detailed.py
# Add to __init__:
self.thermal_model = ThermalInertiaModel(
    C_thermal_J_K=2.6e6,
    h_A_passive_W_K=100.0,
    T_initial_K=298.15,
    max_cooling_kw=100.0
)

# Modify step():
dt_s = 60  # Hardcoded for 1-minute
T_new_K = self.thermal_model.step(dt_s, heat_generated_W, 333.15)
self.T_K = T_new_K
self.efficiency = self._calc_efficiency_from_temp(T_new_K)
```

**[ ] Insert ThermalInertiaModel into SOEC:**
```python
# File: soec_electrolyzer_detailed.py
# Similar to PEM, but T_initial=473K (higher operating temp)
self.thermal_model_soec = ThermalInertiaModel(
    C_thermal_J_K=1.5e6,
    h_A_passive_W_K=80.0,
    T_initial_K=473.15  # 200°C initial
)

self.thermal_model_steam = ThermalInertiaModel(
    C_thermal_J_K=0.5e6,
    h_A_passive_W_K=30.0,
    T_initial_K=323.15  # 50°C water initial
)
```

**[ ] Insert PumpFlowDynamics into Chiller:**
```python
# File: chiller.py
self.pump = PumpFlowDynamics(
    pump_shutoff_pa=350000,
    fluid_inertance_kg_m4=1e4
)

# In step():
dt_s = 60
Q_cool = self.pump.step(dt_s, pump_speed_fraction)
```

**[ ] Insert GasAccumulatorDynamics into Storage:**
```python
# File: storage.py
self.accumulator = GasAccumulatorDynamics(
    V_tank_m3=1.0,
    initial_pressure_pa=40e5
)

# In step():
dt_s = 60
P_new = self.accumulator.step(dt_s, m_dot_in, m_dot_out)
```

**[ ] Modify Main Loop dt:**
```python
# File: manager.py (or simulation_engine.py)
# OLD: self.dt = 1.0  # 1 hour
# NEW:
self.dt = 1.0 / 60.0  # 1 minute in hour units
num_steps = 525600     # 8760 hours × 60 min/hour

for step in range(num_steps):
    t = step * self.dt
    self.step_simulation(t)
```

---

### Tier 2: SHOULD DO (Better accuracy)

**[ ] Add HeatExchangerDynamics to ThermalManager:**
- Replaces instant heat transfer with effectiveness-NTU model
- Adds ~50 lines of code
- Benefit: +5-10% accuracy for distributed systems

**[ ] Temperature-dependent efficiency polynomials:**
- PEM: efficiency drops ~2% per 10°C above nominal
- SOEC: efficiency changes with stack T and steam temp
- Replace hardcoded constants with f(T) functions

**[ ] Pump curve lookup tables:**
- Replace simple quadratic (p = p0 - k·Q²) with real pump datasheets
- More realistic flow-pressure relationship

---

### Tier 3: NICE TO HAVE (Advanced features)

**[ ] Thermal stratification in storage tank:**
- Model hot/cold layers separately
- Adds complexity; benefit for very large tanks only

**[ ] Compressor discharge temperature rise:**
- Account for adiabatic heating during compression
- Affects subsequent cooling duty

**[ ] Multi-phase piping models:**
- H2 can be liquid at high pressure (~2 phases)
- Adds significant complexity; questionable ROI

---

## PART 4: ARCHITECTURAL RISKS & MITIGATION

### Risk 1: Numerical Instability

**Problem:** Forward Euler can diverge if dt too large

**Mitigation:**
```python
# In main simulation __init__:
tau_thermal_min = 50 * 60  # 50 min in seconds = 3000 s
tau_flow_min = 5           # 5 seconds
tau_pressure_min = 30      # 30 seconds

dt_s = 60  # 1 minute

# Check Courant criterion: dt < 2 * τ_min
assert dt_s < 2 * min(tau_thermal_min, tau_flow_min, tau_pressure_min), \
    f"Timestep {dt_s} too large for tau_min={min(...)}"
```

✓ Passes: 60 < 2 × 5 = 10? NO! But 60 < 2 × 30 = 60? YES, marginal.
→ Flow dynamics (τ=5s) will be under-resolved at 60s timestep
→ Need to use smaller timestep or accept some aliasing

**Better approach: Use 30-second timestep (2 per minute):**
```python
# 2-per-minute resolution: more stable
self.dt_minutes = 0.5  # 30 seconds
num_steps = 1051200    # 8760 × 60 / 0.5
```

### Risk 2: Memory Explosion

**Problem:** 525,600 steps × 20 variables × 50 components = 526 MB/hour of history

**Mitigation:**
```python
# Sparse logging strategy
results = []
for step in range(525600):
    t = step / 60.0
    self.step_simulation(t)
    
    if step % 60 == 0:  # Log every hour only
        results.append({
            'time_hour': t,
            'pem_temp_c': pem.T_K - 273.15,
            'soec_temp_c': soec.T_K - 273.15,
            'storage_pressure_bar': storage.pressure_bar,
            'lcoh_eur_mwh': market.calculate_lcoh(...)
        })

# Result: ~200 MB for full year with hourly data
```

### Risk 3: Causal Coupling Errors

**Problem:** If component A depends on component B's output, must call B first

**Mitigation:**
```python
# Enforce dependency graph in orchestration:
STEP_ORDER = [
    'market',           # Inputs
    'pem',              # Primary generation (generates heat)
    'soec',             # Secondary generation
    'thermal_manager',  # Distributes heat
    'chiller',          # Responds to temperature
    'compressor',       # H2 pressurization
    'storage',          # Accumulates H2/pressure
    'metrics'           # Calculate LCOH, efficiency
]

for step_name in STEP_ORDER:
    component = system.get_component(step_name)
    component.step(t)
```

### Risk 4: Stale Parameter Values

**Problem:** Model parameter C_thermal might be ±20% wrong

**Mitigation:**
```python
# Calibration procedure (Week 3 task):
# 1. Run pilot plant for 1 week, log T(t) at 1-minute intervals
# 2. Sweep parameter space:
for C_candidate in [2.0e6, 2.2e6, 2.4e6, 2.6e6, 2.8e6]:
    for h_A_candidate in [80, 90, 100, 110, 120]:
        model = ThermalInertiaModel(C=C_candidate, h_A=h_A_candidate, ...)
        error = calculate_rmse(model.T_history, pilot_plant.T_history)
        best_params = minimize(error)

# 3. Use best_params for all future simulations
```

---

## PART 5: IMPLEMENTATION PRIORITY & EFFORT MATRIX

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| Insert ThermalInertiaModel (PEM) | 2 hours | ★★★★★ | **NOW** |
| Insert ThermalInertiaModel (SOEC) | 2 hours | ★★★★★ | **NOW** |
| Insert PumpFlowDynamics (Chiller) | 1.5 hours | ★★★★ | **NOW** |
| Insert GasAccumulatorDynamics (Storage) | 1 hour | ★★★★ | **NOW** |
| Fix main loop dt | 0.5 hours | ★★★★★ | **NOW** |
| HeatExchangerDynamics | 3 hours | ★★★ | Week 2 |
| Efficiency polynomial tuning | 2 hours | ★★★ | Week 2 |
| Calibration against pilot data | 8 hours | ★★★★ | Week 3 |
| Unit test suite | 4 hours | ★★★★ | Week 2 |
| Documentation | 3 hours | ★★ | Week 3 |

**Total Week 1:** ~7 hours (CRITICAL PATH)  
**Total Week 2-3:** ~20 hours (refinement)

---

## SUMMARY: GO/NO-GO FOR 1-MINUTE INTEGRATION

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Mathematical models available?** | ✓ YES | Provided in documents |
| **Python code templates provided?** | ✓ YES | thermal_dynamics_cookbook.md |
| **Numerical stability OK?** | ⚠ MARGINAL | Use 30-sec timestep if possible |
| **Memory feasible?** | ✓ YES | ~200 MB with sparse logging |
| **Integration complexity?** | ✓ LOW | 4 major insertions needed |
| **Risk mitigation clear?** | ✓ YES | 4 main risks documented |
| **Effort realistic?** | ✓ YES | 27 hours total (3 weeks) |

**RECOMMENDATION: GO** ✓

---

**Document Version:** 1.0  
**Date:** November 25, 2025  
**Status:** READY FOR DEVELOPMENT
