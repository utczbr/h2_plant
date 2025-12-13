# Developer Guide: Component Implementation

This guide defines the standards for implementing new components in the H2 Plant simulation. All components must adhere to these interfaces to ensure compatibility with the simulation engine, optimization layer, and visualization tools.

## 1. The Component Lifecycle Contract (Layer 1)

All components must inherit from `h2_plant.core.component.Component` and implement the three-phase lifecycle. This contract ensures deterministic execution and strictly defined state transitions.

### Phase 1: Initialization provided by `initialize()`
**Purpose:** Allocate memory, resolve dependencies, and prepare for the first timestep.
**Constraint:** No physics execution or time advancement.
**Args:**
*   `dt` (float): Simulation timestep in hours (e.g., 1/60 for 1 minute).
*   `registry` (ComponentRegistry): Access to other system components.

```python
def initialize(self, dt: float, registry: ComponentRegistry) -> None:
    super().initialize(dt, registry)
    # 1. Store timestep
    self.dt = dt
    
    # 2. Resolve dependencies
    self.other_component = registry.get("other_id")
    
    # 3. Initialize sub-components (if composite)
    self.child.initialize(dt, registry)
    
    # 4. Pre-allocate arrays (optimization)
    self.history = np.zeros(1000)
```

### Phase 2: Execution provided by `step()`
**Purpose:** Advance component state by one timestep `dt`.
**Constraint:** Causal execution. Inputs must be available before processing.
**Args:**
*   `t` (float): Current simulation time in hours.

```python
def step(self, t: float) -> None:
    super().step(t)
    
    # 1. Read Inputs (from upstream or setpoints)
    inflow = self.input_port.get_value()
    
    # 2. Execute Physics (Thermodynamics, Reactions)
    # Explain WHY: "Calculate equilibrium to determine condensation"
    outflow = self._calculate_physics(inflow)
    
    # 3. Update State
    self.state.update(outflow)
    
    # 4. Push/Buffer Outputs
    self.output_port.set_value(outflow)
```

### Phase 3: State Reporting provided by `get_state()`
**Purpose:** Return a snapshot of the component's internal state for monitoring and checkpoints.
**Constraint:** Must return a JSON-serializable dictionary.
**Args:** None.

```python
def get_state(self) -> Dict[str, Any]:
    return {
        **super().get_state(),
        "temperature_k": self.temp_k,
        "pressure_pa": self.pressure_pa,
        "efficiency": self.efficiency
        # Composite components include child states
        # "subsystem": self.child.get_state()
    }
```

---

## 2. Documentation Standards

Professional engineering documentation is mandatory. Code comments should explain **why** logic exists, not just what it does.

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

### Explaining the "Why"
*   **Bad:** `i += 1` (Increments i)
*   **Good:** `i += 1` (Advance to next integration node)

*   **Bad:** `if p > 100: p = 100` (Cap pressure)
*   **Good:** `if p > 100: p = 100` (Relief valve activation pressure limit per ASME safety standard)

---

## 3. Units and Standards

To ensure physical accuracy, the project follows a strict units convention.

### Internal Calculations (SI Units)
All internal physics, thermodynamics, and flow calculations use **SI Units**:
*   **Pressure:** Pascals (`Pa`)
*   **Temperature:** Kelvin (`K`)
*   **Mass:** Kilograms (`kg`)
*   **Energy:** Joules (`J`) or Watt-hours (`Wh`) for accumulated energy
*   **Power:** Watts (`W`)
*   **Time:** Hours (`h`) (Note: Flow rates are usually `kg/h`)

### Configuration & GUI (Engineering Units)
For specific user-facing configuration (YAML) and GUI displays:
*   **Pressure:** Bar (`bar`) -> Converted to Pa in `__init__`
*   **Temperature:** Celsius (`°C`) -> Converted to K in `__init__`
*   **Power:** Megawatts (`MW`) or Kilowatts (`kW`)

**Developer Pattern:**
```python
def __init__(self, pressure_bar: float):
    # Convert config input to internal SI units immediately
    self.pressure_pa = pressure_bar * 1e5
```

---

## 4. Flow Interface (Layer 3)

Components that transfer resources use the Flow/Port interface.

1.  **`get_ports()`**: Define metadata for inputs/outputs.
2.  **`receive_input()`**: Accept resources from upstream (Push).
3.  **`get_output()`**: Offer resources to downstream (Pull/Query).
4.  **`extract_output()`**: Finalize transfer and deduct resources (Transaction commit).
5.  **Output Buffering**: Accumulate production during `step()`, offer it in `get_output()`.

See the `FlowNetwork` documentation for detailed topology examples.

---

## 5. Performance Optimization

For computation-intensive components (thermodynamics, mixing, large arrays):

1.  **LUTManager**: Use `self.registry.get('lut_manager')` for property lookups instead of calling CoolProp directly.
2.  **Numba**: Move hot loops to `h2_plant/optimization/numba_ops.py` and decorate with `@njit`.
3.  **Vectorization**: Use NumPy arrays for collections (e.g., `TankArray`) instead of Python lists of objects.
