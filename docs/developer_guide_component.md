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

### Composition Convention
*   **Stream.composition:** Always **mass fractions** (x_i = m_i / Σm_j)
*   **YAML impurity params:** Typically **mole fractions** (converted in constructor)
*   **PPM values:** Use Stream.mole_fractions property for equilibrium calculations

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

---

## 6. Simulation Engine Architecture

The `SimulationEngine` (Layer 2) orchestrates all component execution, flow propagation, and monitoring.

### Engine Lifecycle

```python
from h2_plant.simulation.engine import SimulationEngine

# 1. Create engine with registry and topology
engine = SimulationEngine(
    registry=registry,
    config=config,
    topology=connections,
    dispatch_strategy=dispatch_strategy  # Optional
)

# 2. Initialize all components
engine.initialize()

# 3. Set dispatch data (optional)
engine.set_dispatch_data(prices=price_array, wind=wind_array)

# 4. Run simulation
results = engine.run(start_hour=0, end_hour=24)
```

### Timestep Execution Flow

Each timestep follows this sequence:

1. **Pre-step callback** (optional)
2. **Process scheduled events**
3. **Dispatch: decide_and_apply()** — Sets power inputs on electrolyzers
4. **Execute components** — Each component.step() called once in causal order
5. **Propagate flows** — FlowNetwork routes streams between components
6. **Dispatch: record_post_step()** — Captures physics results
7. **Post-step callback** (optional)

### Causal Execution Order

Components are executed in dependency order to ensure correct physics:
1. External inputs (price tracker, environment)
2. Production (electrolyzers)
3. Thermal management
4. Separation and processing
5. Compression and storage

---

## 7. Dispatch Strategy Integration

The `EngineDispatchStrategy` pattern separates **control logic** from **physics execution**.

### Pattern Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ decide_and_apply│ -> │ component.step() │ -> │ record_post_step│
│   (set power)   │    │ (run physics)    │    │ (capture result)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Implementing a Dispatch Strategy

```python
from h2_plant.control.engine_dispatch import EngineDispatchStrategy

class MyStrategy(EngineDispatchStrategy):
    def initialize(self, registry, context, total_steps):
        """Pre-allocate NumPy arrays for history."""
        self.registry = registry
        self.h2_production = np.zeros(total_steps)
        self.step_idx = 0
    
    def decide_and_apply(self, t: float, prices: np.ndarray, wind: np.ndarray):
        """Set power inputs BEFORE physics runs."""
        electrolyzer = self.registry.get('pem_electrolyzer')
        if electrolyzer:
            # Calculate setpoint from prices/wind
            power_mw = self._calculate_optimal_power(prices, wind, t)
            electrolyzer.receive_input('power_in', power_mw * 1e6, 'electricity')
    
    def record_post_step(self):
        """Capture results AFTER physics runs."""
        electrolyzer = self.registry.get('pem_electrolyzer')
        if electrolyzer:
            state = electrolyzer.get_state()
            self.h2_production[self.step_idx] = state.get('h2_production_kg_h', 0)
        self.step_idx += 1
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Return recorded metrics."""
        return {'h2_production': self.h2_production[:self.step_idx]}
```

---

## 8. Stream Class Reference

The `Stream` dataclass represents a material flow with thermodynamic properties.

### Constructor

```python
from h2_plant.core.stream import Stream

stream = Stream(
    mass_flow_kg_h=100.0,           # Required
    temperature_k=333.15,           # Default: 298.15 K
    pressure_pa=40e5,               # Default: 101325 Pa
    composition={'H2': 0.995, 'H2O': 0.005},  # Mass fractions (auto-normalized)
    phase='gas',                    # 'gas', 'liquid', 'mixed'
    extra={}                        # Additional metadata
)
```

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `mole_fractions` | Dict[str, float] | Converts mass fractions to mole fractions |
| `specific_enthalpy_j_kg` | float | Enthalpy relative to 298.15K (J/kg) |
| `specific_entropy_j_kgK` | float | Entropy with mixing term (J/kg·K) |
| `density_kg_m3` | float | Ideal gas law density |
| `volume_flow_m3_h` | float | Volumetric flow rate |
| `entrained_liq_kg_s` | float | Liquid carryover from extra dict |

### Stream Mixing

```python
# Conserves mass and enthalpy
mixed = stream_a.mix_with(stream_b)
# Temperature solved via Newton-Raphson iteration
# Pressure = min(stream_a.pressure_pa, stream_b.pressure_pa)
```

### Isentropic Compression

```python
compressed_stream, work_kwh = stream.compress_isentropic(
    outlet_pressure_pa=70e5,
    isentropic_efficiency=0.75
)
```

---

## 9. LUTManager Reference

The `LUTManager` provides 50-200x faster property lookups than CoolProp via pre-computed interpolation tables.

### Initialization

```python
# LUTManager is auto-registered during engine initialization
lut_manager = registry.get('lut_manager')
```

### Single-Point Lookup

```python
# lookup(fluid, property, pressure_pa, temperature_k)
rho = lut_manager.lookup('H2', 'D', 350e5, 298.15)  # Density (kg/m³)
h = lut_manager.lookup('H2', 'H', 350e5, 298.15)    # Enthalpy (J/kg)
s = lut_manager.lookup('H2', 'S', 350e5, 298.15)    # Entropy (J/kg·K)
z = lut_manager.lookup('H2', 'Z', 350e5, 298.15)    # Compressibility factor
cp = lut_manager.lookup('H2', 'C', 350e5, 298.15)   # Heat capacity (J/kg·K)
```

### Batch Lookups (Vectorized)

```python
# 10-50x faster than loops
pressures = np.array([10e5, 20e5, 30e5])
temps = np.array([300, 350, 400])
densities = lut_manager.lookup_batch('H2', 'D', pressures, temps)
```

### Isentropic Process Lookup

```python
# For compressor/expander calculations: H(P, S)
h_out = lut_manager.lookup_isentropic_enthalpy('H2', p_out, s_in)
```

---

## 10. Numba Operations Reference

Performance-critical functions are JIT-compiled in `h2_plant/optimization/numba_ops.py`.

### Available Functions

#### Tank Operations
```python
from h2_plant.optimization.numba_ops import (
    find_available_tank,      # Find idle tank with capacity
    find_fullest_tank,        # Find tank for discharge
    batch_pressure_update,    # Update all tank pressures (in-place)
    distribute_mass_to_tanks, # Sequential fill
)
```

#### Flash Equilibrium
```python
from h2_plant.optimization.numba_ops import solve_rachford_rice_single_condensable

# For single condensable (water) in inert gas (H2)
# K = P_sat / P_total
# z = mole fraction of water in feed
# Returns beta = vapor fraction (0-1)
beta = solve_rachford_rice_single_condensable(z_H2O, K_value)

# Condensation occurs when z > K (supersaturated)
# beta = (1 - z) / (1 - K) when z > K
```

#### Thermodynamic Calculations
```python
from h2_plant.optimization.numba_ops import (
    calculate_mixture_enthalpy,  # H_mix = Σ yᵢ × [H_f,i + ∫Cp dT]
    calculate_mixture_cp,        # Cp_mix = Σ yᵢ × Cp,i(T)
    calculate_compression_work,  # Polytropic work calculation
)
```

### Adding New Numba Functions

```python
from numba import njit
import numpy as np

@njit
def my_hot_loop(data: np.ndarray, param: float) -> float:
    """
    Brief description of the operation.
    
    Args:
        data: Input array.
        param: Scalar parameter.
    
    Returns:
        Result value.
    """
    total = 0.0
    for i in range(len(data)):
        total += data[i] * param
    return total
```

> [!TIP]
> Use `@njit(cache=True)` for production code to cache compiled bytecode.
> Clear `.h2_plant/__pycache__/` and Numba cache after modifying functions.
