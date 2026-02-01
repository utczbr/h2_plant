# IMPLEMENTATION QUICK REFERENCE
## Thermal Inertia & Flow Dynamics - Code Integration Points

**Date:** November 25, 2025  
**Purpose:** Line-by-line modification checklist for developers  
**Scope:** 4 critical files requiring changes  

---

## FILE 1: pem_electrolyzer_detailed.py

### Location A: __init__ Method (ADD THERMAL MODEL)

**Current Code (Lines ~45-60):**
```python
class DetailedPEMElectrolyzer:
    def __init__(self):
        self.maxpower_kw = 5000
        self.efficiency = 0.60
        self.temperature_k = 333.15  # HARDCODED!
        # ... other init
```

**Required Modification:**
```python
class DetailedPEMElectrolyzer:
    def __init__(self):
        self.max_power_kw = 5000
        
        # ✓ INSERT HERE: Import and instantiate thermal model
        from thermal_inertia_model import ThermalInertiaModel
        
        self.thermal_model = ThermalInertiaModel(
            C_thermal_J_K=2.6e6,           # Thermal mass: stack + coolant
            h_A_passive_W_K=100.0,         # Passive convection
            T_ambient_K=298.15,            # 25°C ambient
            T_initial_K=298.15,            # START COLD (not hot!)
            max_cooling_kw=100.0           # Chiller capacity
        )
        
        self.T_K = self.thermal_model.T_K  # Now tracks model's T
        self.efficiency = 0.45             # Lower at cold start
        
        # Temperature-dependent parameters
        self.U_rev_base_V = 1.23           # Nernst base voltage
        self.dU_dT = -0.85e-3              # Temperature coefficient (V/K)
        
        # ... rest of init
```

**Why:** Thermal state must now evolve; efficiency must respond to T.

---

### Location B: step() Method (ADVANCE THERMAL MODEL)

**Current Code (Lines ~120-150):**
```python
def step(self, t):
    # Calculate efficiency at fixed temperature
    self.efficiency = 0.60  # WRONG: constant!
    
    # Voltage calculation
    U_rev = 1.26  # Hardcoded
    V_cell = 1.8
    I = (self.power_setpoint_mw * 1e6) / V_cell
    
    # Heat generation
    Q = (V_cell - U_rev) * I
    
    # Update production (no temperature feedback)
    self.h2_production_kg_s = (I / 96485) * 2 * 0.002  # kg/s
```

**Required Modification:**
```python
def step(self, t):
    import numpy as np
    
    # ✓ MODIFICATION 1: Get timestep in seconds
    dt_s = 60  # 1-minute timestep (FIXED, not self.dt!)
    
    # ✓ MODIFICATION 2: Temperature-dependent Nernst voltage
    T_current_K = self.thermal_model.T_K
    U_rev = self.U_rev_base_V + self.dU_dT * (T_current_K - 298.15)
    
    # ✓ MODIFICATION 3: Overpotential voltage (affected by current & temp)
    V_cell = U_rev + 0.54  # Operating voltage with overpotential
    I = (self.power_setpoint_mw * 1e6) / V_cell
    
    # ✓ MODIFICATION 4: Heat generation
    Q_in_W = (V_cell - U_rev) * I  # Joule heating from overpotential
    
    # ✓ MODIFICATION 5: ADVANCE THERMAL MODEL (KEY!)
    T_new_K = self.thermal_model.step(
        dt_s=dt_s,
        heat_generated_W=Q_in_W,
        T_control_K=333.15  # 60°C chiller setpoint
    )
    self.T_K = T_new_K
    
    # ✓ MODIFICATION 6: Update efficiency based on ACTUAL temperature
    # Efficiency drops ~2% per 10°C above 60°C nominal
    self.efficiency = self._calc_efficiency_from_temp(self.T_K, I)
    
    # ✓ MODIFICATION 7: H2 production now efficiency-dependent
    theoretical_h2 = (I / 96485) * 2 * 0.002  # Faraday's law
    actual_h2 = theoretical_h2 * self.efficiency
    self.h2_production_kg_s = actual_h2
    
    # ✓ MODIFICATION 8: Track heat output (for thermal manager)
    self.heat_output_kw = Q_in_W / 1000

def _calc_efficiency_from_temp(self, T_K, I):
    """Temperature-dependent efficiency polynomial."""
    T_nominal_K = 333.15  # 60°C
    
    # Base efficiency at nominal temperature
    eta_ref = 0.65  # Nominal efficiency
    
    # Temperature penalty: 0.2% per Kelvin above nominal
    temp_penalty = (T_K - T_nominal_K) * 0.002
    
    # Current density penalty (higher current → higher overpotential)
    j_A_cm2 = I / (self.stack_area_cm2)  # Current density
    current_penalty = max(0, (j_A_cm2 - 1.0) * 0.05)
    
    # Combine penalties
    eta = max(0.40, eta_ref - temp_penalty - current_penalty)
    return eta
```

**Why:** Thermal model is now internal ODE that must advance each step. Efficiency must respond dynamically.

**Critical Detail:** `dt_s = 60` is HARDCODED constant, not calculated from self.dt. This is because the main loop now runs at 1-minute resolution (self.dt = 1/60 hour).

---

## FILE 2: chiller.py

### Location A: __init__ Method (ADD PUMP & THERMAL MODELS)

**Current Code:**
```python
class Chiller(Component):
    def __init__(self, component_id="chiller", cooling_capacity_kw=100.0):
        self.component_id = component_id
        self.cooling_capacity_kw = cooling_capacity_kw
        self.efficiency = 0.95
        # ... no dynamics
```

**Required Modification:**
```python
class ChillerEnhanced(Component):
    def __init__(self, component_id="chiller", cooling_capacity_kw=100.0):
        super().__init__()
        self.component_id = component_id
        self.cooling_capacity_kw = cooling_capacity_kw
        
        # ✓ INSERT: Pump flow dynamics
        from flow_dynamics_model import PumpFlowDynamics
        
        self.pump = PumpFlowDynamics(
            pump_shutoff_pa=350000,             # 3.5 bar max head
            pump_resistance_pa_per_m6_h2=3.5e5, # Flow-pressure curve slope
            system_resistance_pa_per_m6_h2=2.0e5,
            fluid_inertance_kg_m4=1e4,          # Pipes + manifold mass
            initial_flow_m3_h=10.0,
            rho_kg_m3=1000.0                    # Water density
        )
        
        # ✓ INSERT: Coolant thermal model (separate from PEM!)
        from thermal_inertia_model import ThermalInertiaModel
        
        self.coolant_thermal = ThermalInertiaModel(
            C_thermal_J_K=1.0e6,        # Coolant loop thermal mass
            h_A_passive_W_K=50.0,       # HX surface area × coeff
            T_ambient_K=298.15,
            T_initial_K=293.15,         # Start cold
            max_cooling_kw=cooling_capacity_kw
        )
        
        self.outlet_setpoint_k = 333.15  # 60°C target
        self.inlet_stream = None
        self.outlet_stream = None
```

**Why:** Chiller now has TWO independent thermal dynamics: (1) pump ramp-up, (2) coolant temperature response.

---

### Location B: step() Method (ORCHESTRATE DYNAMICS)

**Current Code:**
```python
def step(self, t):
    if self.inlet_stream is None or self.inlet_stream.mass_flow_kgh <= 0:
        return
    
    # Instant temperature drop
    temp_drop = min(self.inlet_stream.temperature_k - 333.15, 
                    self.cooling_capacity_kw / (m_flow * Cp))
    self.outlet_stream.temperature_k = self.inlet_stream.temperature_k - temp_drop
```

**Required Modification:**
```python
def step(self, t):
    import numpy as np
    
    if self.inlet_stream is None or self.inlet_stream.mass_flow_kgh <= 0:
        return
    
    # ✓ MODIFICATION 1: Get timestep
    dt_s = 60  # 1-minute
    
    # ✓ MODIFICATION 2: Calculate pump speed from temperature error
    temp_error = self.inlet_stream.temperature_k - self.outlet_setpoint_k
    pump_speed = np.clip(temp_error / 30.0, 0, 1)  # 30K error → 100% speed
    
    # ✓ MODIFICATION 3: ADVANCE PUMP FLOW (transient response)
    Q_cool_m3_h = self.pump.step(dt_s=dt_s, pump_speed_fraction=pump_speed)
    
    # ✓ MODIFICATION 4: Estimate heat absorbed by coolant
    Q_absorbed_W = self.cooling_capacity_kw * 1000 * min(1.0, temp_error / 10)
    
    # ✓ MODIFICATION 5: ADVANCE COOLANT TEMPERATURE
    T_coolant_K = self.coolant_thermal.step(
        dt_s=dt_s,
        heat_generated_W=Q_absorbed_W,
        T_control_K=self.outlet_setpoint_k
    )
    
    # ✓ MODIFICATION 6: Calculate outlet temperature
    # Bounded by coolant temperature (can't cool below coolant temp!)
    mass_flow_kg_s = Q_cool_m3_h / 3.6  # m³/h → kg/s
    Cp = 4.18  # kJ/kg·K
    
    if mass_flow_kg_s > 0:
        # Effectiveness-NTU model: Q = effectiveness × m_dot × Cp × LMTD
        NTU = 0.8  # Number of transfer units (design parameter)
        effectiveness = 1 - np.exp(-NTU)
        
        Q_max_W = mass_flow_kg_s * Cp * 20 * 1000  # Max ΔT = 20K
        Q_actual_W = min(Q_max_W * effectiveness, Q_absorbed_W)
        
        outlet_temp_drop = Q_actual_W / (mass_flow_kg_s * Cp)
        outlet_temp = max(T_coolant_K, self.inlet_stream.temperature_k - outlet_temp_drop)
    else:
        outlet_temp = self.inlet_stream.temperature_k
    
    # ✓ MODIFICATION 7: Create outlet stream
    self.outlet_stream = Stream(
        mass_flow_kgh=Q_cool_m3_h * 1000,  # m³/h water → kg/h
        temperature_k=outlet_temp,
        pressure_pa=self.inlet_stream.pressure_pa
    )

def get_state(self):
    """Return diagnostic state."""
    return {
        **super().get_state(),
        'outlet_temp_c': self.outlet_stream.temperature_k - 273.15 if self.outlet_stream else 0,
        'pump_flow_m3_h': self.pump.Q_m3_h,
        'pump_pressure_bar': self.pump.p_pump_pa / 1e5,
        'coolant_temp_c': self.coolant_thermal.T_K - 273.15
    }
```

**Why:** Chiller now has 3-part orchestration: (1) pump speed control, (2) pump flow dynamics, (3) coolant temperature. Each has different timescale.

---

## FILE 3: storage.py

### Location A: __init__ Method (ADD ACCUMULATOR MODEL)

**Current Code:**
```python
class H2StorageTank(Component):
    def __init__(self, tank_id, volume_m3=1.0, initial_pressure_bar=40):
        self.volume_m3 = volume_m3
        self.initial_pressure_bar = initial_pressure_bar
        self.pressure_pa = initial_pressure_bar * 1e5
```

**Required Modification:**
```python
class H2StorageTankEnhanced(Component):
    def __init__(self, tank_id, volume_m3=1.0, initial_pressure_bar=40.0):
        super().__init__()
        self.tank_id = tank_id
        self.volume_m3 = volume_m3
        
        # ✓ INSERT: Gas accumulator model
        from flow_dynamics_model import GasAccumulatorDynamics
        
        self.accumulator = GasAccumulatorDynamics(
            V_tank_m3=volume_m3,
            T_tank_k=298.15,                    # Isothermal (cooled)
            initial_pressure_pa=initial_pressure_bar * 1e5,
            R_gas_j_kg_k=4124.0                 # H2 gas constant
        )
        
        # Track mass flows (set by manager before step)
        self.m_dot_in_kg_s = 0.0              # Compressor output
        self.m_dot_out_kg_s = 0.0             # User demand
```

**Why:** Pressure is now evolved state (not calculated algebraically).

---

### Location B: step() Method (ADVANCE ACCUMULATOR)

**Current Code:**
```python
def step(self, t):
    # Static calculation
    self.pressure_pa = (self.mass_kg * R * T) / self.volume_m3
```

**Required Modification:**
```python
def step(self, t):
    # ✓ MODIFICATION: Get timestep
    dt_s = 60  # 1-minute
    
    # ✓ CRITICAL: m_dot_in and m_dot_out must be set by manager before calling step!
    # They are set via:
    #   storage.m_dot_in_kg_s = compressor.h2_production_kg_s
    #   storage.m_dot_out_kg_s = market_demand_kg_s
    
    # ✓ ADVANCE ACCUMULATOR (KEY!)
    P_new_pa = self.accumulator.step(
        dt_s=dt_s,
        m_dot_in_kg_s=self.m_dot_in_kg_s,
        m_dot_out_kg_s=self.m_dot_out_kg_s
    )
    
    # ✓ UPDATE STATE
    self.pressure_pa = P_new_pa
    self.pressure_bar = P_new_pa / 1e5
    self.mass_kg = self.accumulator.M_kg

def receive_h2_input(self, mass_flow_kg_s):
    """Called by compressor to set input flow."""
    self.m_dot_in_kg_s = mass_flow_kg_s

def extract_h2_output(self, mass_flow_kg_s):
    """Called by demand to set output flow."""
    self.m_dot_out_kg_s = mass_flow_kg_s
```

**Why:** Pressure dynamics now depend on net flow (in - out). Manager must set flows before calling step().

---

## FILE 4: manager.py (MAIN ORCHESTRATOR)

### Location A: Main Loop (CHANGE TIMESTEP & ORDER)

**Current Code:**
```python
class SimulationEngine:
    def __init__(self):
        self.dt = 1.0  # 1 hour
        self.current_time = 0.0
    
    def run(self):
        for hour in range(8760):
            self.step_simulation(hour)
            
    def step_simulation(self, t_hour):
        # All components step
        pem.step(t_hour)
        chiller.step(t_hour)
        storage.step(t_hour)
```

**Required Modification:**
```python
class SimulationEngine:
    def __init__(self):
        # ✓ MODIFICATION 1: 1-minute timestep
        self.dt = 1.0 / 60.0  # 1-minute in hour units
        self.current_time = 0.0
        
        # ✓ MODIFICATION 2: Track components for causal ordering
        self.components_ordered = [
            self.market,           # 1. External inputs
            self.pem,              # 2. Primary generation → heat
            self.soec,             # 3. Secondary generation
            self.thermal_manager,  # 4. Distribute heat
            self.chiller,          # 5. Cool PEM
            self.compressor,       # 6. Compress H2
            self.storage,          # 7. Store H2 + pressure
            self.metrics           # 8. Calculate LCOH
        ]
    
    def run(self):
        # ✓ MODIFICATION 3: Run 1-minute steps for full year
        num_steps = 525600  # 8760 hours × 60 min/hour
        
        results = []
        for step in range(num_steps):
            t_hour = step * self.dt  # Fractional hours
            self.step_simulation(t_hour)
            
            # ✓ MODIFICATION 4: Sparse logging (save every hour)
            if step % 60 == 0:
                results.append(self.get_state())
                print(f"Hour {step // 60} / 8760")
        
        return results
    
    def step_simulation(self, t_hour):
        # ✓ MODIFICATION 5: CAUSAL EXECUTION ORDER (must not change!)
        
        # 1. Market signal (external input)
        price_eur_mwh = self.market.get_price(t_hour)
        pem_setpoint = self.market.calculate_pem_setpoint(price_eur_mwh)
        soec_setpoint = self.market.calculate_soec_setpoint(price_eur_mwh)
        
        # 2. PEM electrolyzer (generates heat)
        self.pem.power_setpoint_mw = pem_setpoint
        self.pem.step(t_hour)
        # State now: pem.T_K, pem.heat_output_kw, pem.efficiency (all dynamic)
        
        # 3. SOEC electrolyzer (generates more heat)
        self.soec.power_setpoint_mw = soec_setpoint
        self.soec.step(t_hour)
        
        # 4. Thermal manager (distribute heat PEM → SOEC)
        self.thermal_manager.step(t_hour)
        # Transfers heat based on: PEM.T, SOEC.T, flow rates
        
        # 5. Chiller (cool PEM based on its actual temperature)
        self.chiller.inlet_stream = self.pem.outlet_coolant_stream
        self.chiller.step(t_hour)
        # Pump speed responds to pem.T_K, flow ramps dynamically
        
        # 6. Compressor (compress H2 from PEM)
        self.compressor.inlet_h2_kg_s = self.pem.h2_production_kg_s
        self.compressor.step(t_hour)
        
        # 7. Storage (accumulate H2 & pressure)
        self.storage.receive_h2_input(self.compressor.h2_output_kg_s)
        self.storage.extract_h2_output(self.market.h2_demand_kg_s)
        self.storage.step(t_hour)
        # Pressure now evolves: dP/dt = (RT/V) * (in - out)
        
        # 8. Calculate metrics
        lcoh = self.calculate_lcoh(t_hour)
        
    def get_state(self):
        """Return diagnostic state for one time point."""
        return {
            'time_hour': self.current_time,
            'pem_temp_c': self.pem.T_K - 273.15,
            'soec_temp_c': self.soec.T_K - 273.15,
            'pem_efficiency': self.pem.efficiency,
            'chiller_outlet_c': self.chiller.outlet_stream.temperature_k - 273.15,
            'pump_flow_m3_h': self.chiller.pump.Q_m3_h,
            'storage_pressure_bar': self.storage.pressure_bar,
            'storage_fill_fraction': self.storage.mass_kg / (1000 * self.storage.volume_m3),
            'lcoh_eur_mwh': self.metrics.lcoh
        }
```

**Why:**
1. **dt = 1/60 hour = 1 minute** is now the main loop timestep
2. **Causal order must be preserved** (market → generation → cooling → storage → metrics)
3. **Sparse logging** saves memory (200 MB not 2 GB)
4. **All dt_s values in component steps are now hardcoded 60 seconds**, not calculated

---

## IMPLEMENTATION CHECKLIST (Use This!)

### Phase 1: Insert Models (Day 1)
- [ ] Import `ThermalInertiaModel` into pem_electrolyzer_detailed.py
- [ ] Create `self.thermal_model = ThermalInertiaModel(...)` in __init__
- [ ] Replace hardcoded `self.T = 333.15` with `self.T_K = self.thermal_model.T_K`
- [ ] Repeat for SOEC

- [ ] Import `PumpFlowDynamics` into chiller.py
- [ ] Create `self.pump = PumpFlowDynamics(...)` in __init__
- [ ] Create `self.coolant_thermal = ThermalInertiaModel(...)` in __init__

- [ ] Import `GasAccumulatorDynamics` into storage.py
- [ ] Create `self.accumulator = GasAccumulatorDynamics(...)` in __init__

### Phase 2: Advance Models (Day 2)
- [ ] In pem.step(), call `T_new = self.thermal_model.step(dt_s=60, Q_in=...)`
- [ ] Add temperature-dependent efficiency function `_calc_efficiency_from_temp()`
- [ ] Update `self.efficiency` based on actual T_K
- [ ] Update `self.heat_output_kw` for thermal manager

- [ ] In chiller.step(), call `Q_flow = self.pump.step(dt_s=60, speed=...)`
- [ ] Call `T_cool = self.coolant_thermal.step(dt_s=60, Q_in=...)`
- [ ] Calculate outlet T based on flow & coolant T

- [ ] In storage.step(), call `P_new = self.accumulator.step(dt_s=60, m_dot_in=..., m_dot_out=...)`
- [ ] Add `receive_h2_input()` and `extract_h2_output()` methods

### Phase 3: Orchestrate (Day 3)
- [ ] Change main loop: `self.dt = 1/60` (1-minute)
- [ ] Change loop: `for step in range(525600):` (not 8760)
- [ ] Define component execution order
- [ ] Set flows before calling storage.step()
- [ ] Add sparse logging (save every 60 steps)

---

**Status: READY FOR DEVELOPER IMPLEMENTATION**

Use this as copy-paste template for exact modifications needed.

