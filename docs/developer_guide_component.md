# Developer Guide: Component Implementation

**Version**: 2.0  
**Updated**: December 2025

This guide defines the comprehensive standards for implementing new components in the H2 Plant simulation. All components must adhere to these interfaces to ensure compatibility with the `SimulationEngine`, `EngineDispatchStrategy`, and `Stream` topology.

---

## 1. The Component Lifecycle Contract (Layer 1)

All components must inherit from `h2_plant.core.component.Component` and implement the three-phase lifecycle. This contract ensures deterministic execution and strictly defined state transitions.

### Phase 1: Initialization provided by `initialize()`

**Purpose:** Allocate memory, resolve dependencies via the `Registry`, and prepare internal state for the first timestep.
**Constraint:** No physics execution or time advancement. Do not call `CoolProp` here; use JIT-compiled LUTs if possible, or defer to `step()`.

**Arguments:**
*   `dt` (float): Simulation timestep in hours (e.g., 0.0166 for 1 minute).
*   `registry` (ComponentRegistry): Interface to access other system components (e.g., `lut_manager`).

```python
def initialize(self, dt: float, registry: ComponentRegistry) -> None:
    super().initialize(dt, registry)
    
    # 1. Validation
    if self.min_power_kw < 0:
        raise ValueError(f"Component {self.component_id}: min_power_kw must be positive.")

    # 2. Store timestep
    self.dt = dt
    
    # 3. Resolve dependencies (e.g., LUT Manager for fast thermodynamics)
    self.lut_manager = registry.get("lut_manager")
    
    # 4. Initialize State Dictionaries (for get_state())
    self.current_state = {
        "temperature_k": 298.15,
        "pressure_pa": 101325.0,
        "mass_flow_kg_h": 0.0
    }
```

### Phase 2: Execution provided by `step()`

**Purpose:** Advance component state by one timestep `dt`.
**Constraint:** Causal execution. Inputs (streams, power) must be available *before* processing.

**Arguments:**
*   `t` (float): Current simulation time in hours.

```python
def step(self, t: float) -> None:
    super().step(t)
    
    # 1. Read Inputs (from upstream or dispatch setpoints)
    in_stream = self.receive_input("inlet") 
    power_mw = self.current_input.get("power_in", 0.0)
    
    # 2. Guard Clauses (Idle State)
    if not in_stream or in_stream.mass_flow_kg_h <= 0:
        self._set_idle_state()
        return

    # 3. Execute Physics (The "Why")
    # Explain: "Compress adiabatic: H_out = H_in + Work/Efficiency"
    out_stream, work_kw = self._calculate_compression(in_stream, power_mw)
    
    # 4. Update Internal State
    self.current_state.update({
        "temperature_k": out_stream.temperature_k,
        "power_kw": work_kw,
        "efficiency": self._calculate_efficiency(work_kw)
    })
    
    # 5. Push/Buffer Outputs for Downstream
    self.output_buffer["outlet"] = out_stream
```

### Phase 3: State Reporting provided by `get_state()`

**Purpose:** Return a snapshot of the component's internal state for monitoring, history tracking (Dispatch), and checkpoints.
**Constraint:** Must return a JSON-serializable dictionary (no objects).

```python
def get_state(self) -> Dict[str, Any]:
    # Combine base state with component-specific metrics
    return {
        **super().get_state(),
        **self.current_state,
        "specific_metric": self.internal_value
    }
```

---

## 2. The Flow Interface (Layer 3)

Components interact via the `Stream` object and a Push/Pull flow mechanic.

### Flow Mechanics

1.  **Push (receive_input)**: Upstream components or the `FlowNetwork` push material into the component buffer.
2.  **Pull (get_output)**: Downstream components query `get_output()` to check availability.
3.  **Commit (extract_output)**: Downstream components call `extract_output()` to "take" the material, clearing the buffer.

### Working with Streams

The `Stream` object is the universal currency of the plant.

```python
from h2_plant.core.stream import Stream

# Creating a Stream (e.g., H2 Source)
h2_stream = Stream(
    mass_flow_kg_h=50.0,
    temperature_k=350.0,
    pressure_pa=30e5,
    composition={'H2': 0.99, 'H2O': 0.01}, # Mass Fractions
    phase='gas'
)

# Mixing Streams (e.g., Mixer)
# Handles Enthalpy balance: H_mix = (m1*h1 + m2*h2) / (m1+m2)
mixed_stream = stream_a.mix_with(stream_b) 
```

### Implementing `get_output`

Always return the stream currently in your output buffer. Do not calculate on the fly here; calculations belong in `step()`.

```python
def get_output(self, port_name: str) -> Optional[Stream]:
    return self.output_buffer.get(port_name)
```

---

## 3. Physical Modeling Best Practices

### 3.1 Thermodynamic Properties (LUTs)
**Critical**: Avoid calling `CoolProp.PropsSI` inside the `step()` loop. It is too slow (100ms/call). Use `LUTManager` (0.01ms/call).

```python
# Bad
rho = CoolProp.PropsSI('D', 'T', T, 'P', P, 'Hydrogen')

# Good
rho = self.lut_manager.lookup('H2', 'D', P, T)
```

### 3.2 Mass Conservation
**Rule**: $\sum \dot{m}_{in} = \sum \dot{m}_{out} + \frac{dm_{store}}{dt}$

*   **Steady State**: `inlet.mass_flow` must exactly equal `output.mass_flow` + `drain.mass_flow`.
*   **Dynamic**: Storage tanks must track inventory precisely.

### 3.3 Energy Conservation
**Rule**: $Q_{net} = \dot{m}(h_{in} - h_{out}) + W_{elec}$

*   **Compressors**: Work input raises enthalpy ($h_{out} > h_{in}$).
*   **Coolers**: Heat removal lowers enthalpy ($h_{out} < h_{in}$).
*   Always calculate $T_{out}$ from $H_{out}$ (Enthalpy) logic, not the other way around. $H$ is the conserved quantity; $T$ is a derived property.

### 3.4 Phase Equilibrium (Flash)
**Rule**: Never assume a stream is pure liquid/gas if P/T changes.

*   Use `solve_rachford_rice` (in `numba_ops`) for VLE calculations.
*   Check if partial pressure of water $P_{H2O} > P_{sat}(T)$. If so, condense liquid.

---

## 4. Integration with Dispatch Engine

The `EngineDispatchStrategy` records your component's history. To ensure your component is tracked:

### 1. Register History Arrays
In `engine_dispatch.py` -> `initialize()`:

```python
if self._my_components:
    for comp in self._my_components:
        cid = comp.component_id
        # Pre-allocate numpy arrays for performance
        self._history[f"{cid}_power_kw"] = np.zeros(total_steps, dtype=np.float64)
        self._history[f"{cid}_efficiency"] = np.zeros(total_steps, dtype=np.float64)
```

### 2. Record Pre-Step (Decision)
In `engine_dispatch.py` -> `decide_and_apply()`:

*   Send control signals (setpoints) to your component.
*   Do NOT record state here (physics hasn't run yet).

```python
for comp in self._my_components:
    target_power = calculate_optimal(prices)
    comp.receive_input('power_in', target_power, 'electricity')
```

### 3. Record Post-Step (Result)
In `engine_dispatch.py` -> `record_post_step()`:

*   Read the state *after* `step()` has completed.

```python
for comp in self._my_components:
    state = comp.get_state()
    # Direct fast write to numpy array
    self._history[f"{comp.component_id}_power_kw"][step_idx] = state.get('power_kw', 0.0)
```

---

## 5. Topology YAML Configuration

Your component must be instantiable from `plant_topology.yaml`.

### 1. Define Unique `type`
Map the string type to your Python class in `ComponentRegistry`.

### 2. Parameter Schema
Support all parameters defined in your `__init__`.

```yaml
- id: "My_New_Heater"
  type: "ElectricBoiler"
  params:
    max_power_kw: 500.0
    efficiency: 0.98
    process_step: 150  # Execution Order Priority
  connections:
    - source_port: "fluid_out"
      target_name: "Next_Component"
      target_port: "inlet"
      resource_type: "stream"
```

### 3. Process Step (Execution Order)
*   **Low (0-50)**: Sources, Mixers
*   **Medium (50-200)**: Electrolyzers, Processing
*   **High (200+)**: Storage, Sinks

---

## 6. Documentation Standards

Professional engineering documentation is mandatory.

### Google-Style Docstrings
Every class and method must have a docstring.

```python
def calculate_efficiency(self, load_fraction: float) -> float:
    """
    Calculate thermal efficiency based on current load.
    
    Uses a 3rd-order polynomial curve fit from experimental data
    to approximate efficiency at partial loads.
    
    Args:
        load_fraction (float): Operational load (0.0 to 1.0).
        
    Returns:
        float: Thermal efficiency (0.0 to 1.0).
        
    Raises:
        ValueError: If load_fraction is outside [0, 1].
    """
```

### Implementation Comments (The "Why")

*   **Bad**: `# Calculate enthalpy`
*   **Good**: `# Calculate isentropic enthalpy rise to determine theoretical T_out before efficiency losses`

---

## 7. Troubleshooting Common Issues

### "Property lookup out of LUT bounds"
*   **Cause**: Your component is generating P or T outside the table limits (e.g., T > 1200K or P < 0.1 bar).
*   **Fix**: Clamp inputs or check physical validity. Vacuum pressures often need explicit handling.

### "Mass Balance Violation"
*   **Cause**: `inlet != outlet + drain`. Often caused by forgetting to account for `Stream.extra` (entrained liquid).
*   **Fix**: Ensure `m_dot_total` includes vapor + liquid phases + any separated mass.

### "AttributeError: 'dict' object has no attribute..."
*   **Cause**: `get_state()` returned a `Stream` object instead of a primitive or dict.
*   **Fix**: Use `stream.to_dict()` or manually extract fields in `get_state()`.

### "Graph line is flat / zero"
*   **Cause**: History key in `engine_dispatch.py` matches `static_graphs.py` column name?
*   **Fix**: Check `COLUMN_ALIASES` in the plotting modules. Ensure `record_post_step` is actually writing data (not inside a conditional that fails).
