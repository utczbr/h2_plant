# HPC & Thermodynamics Optimization Guide

**Document Version:** 1.0  
**Last Updated:** November 20, 2025  
**Target Audience:** Performance Engineers, Component Developers

---

## Purpose

This guide establishes **mandatory performance standards** for all computation-intensive code in the Dual-Path Hydrogen Production System[file:13][file:14]. Following these standards ensures the system maintains its **50-200x performance advantage** over naive implementations while preserving thermodynamic accuracy[file:13][file:14].

---

## Performance Baseline & Targets

The system achieves **30-90 second** simulations for 8,760-hour annual cycles through three complementary techniques[file:13][file:14]:

| **Technique** | **Target Speedup** | **Application** |
|--------------|-------------------|-----------------|
| LUT Manager | 50-200x | Thermodynamic property lookups |
| Numba JIT | 10-50x | Hot path math loops |
| NumPy Vectorization | 10-50x | Multi-tank operations |

**Critical Rule:** Any component performing >100 thermodynamic lookups per timestep or operating on arrays with >8 elements **must** use these optimizations[file:13][file:14].

---

## LUT Strategy: When to Cache vs Calculate

### Decision Tree

```
Is this a thermodynamic property call?
├─ NO → Use direct calculation
└─ YES → Is it in a hot path (called >100/timestep)?
    ├─ NO → CoolProp direct call acceptable
    └─ YES → Use LUTManager
        ├─ Is property cached? (H2/O2/N2 density/enthalpy/entropy/Cp)
        │   ├─ YES → lut_manager.lookup()
        │   └─ NO → Request LUT extension or fall back
        └─ Is query out of bounds? (P: 1-900 bar, T: 250-350 K)
            ├─ YES → Fallback to CoolProp (logged warning)
            └─ NO → Standard LUT lookup
```

### LUT Coverage & Accuracy

The `LUTManager` pre-caches properties for H₂, O₂, and N₂ over operational ranges[file:13][file:14]:

- **Pressure:** 1-900 bar (1e5 - 900e5 Pa)
- **Temperature:** 250-350 K (-23°C to 77°C)
- **Properties:** Density (D), Enthalpy (H), Entropy (S), Heat Capacity (C)
- **Accuracy:** <0.5% interpolation error vs CoolProp
- **Performance:** 0.05-0.1 ms vs 5-20 ms (CoolProp)

### Usage Pattern: Single Lookup

```
from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager

class OptimizedTank(Component):
    def __init__(self, capacity_kg: float, pressure_bar: float):
        super().__init__()
        self.capacity_kg = capacity_kg
        self.pressure_pa = pressure_bar * 1e5
        self.temperature_k = 298.15
        self._lut: LUTManager | None = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        # Resolve LUTManager from registry
        self._lut = registry.get("lut_manager")

    def step(self, t: float) -> None:
        super().step(t)
        # Fast density lookup (0.05-0.1 ms)
        density = self._lut.lookup('H2', 'D', self.pressure_pa, self.temperature_k)
        
        # Use density for volume calculation
        volume = self.mass_kg / density
```

### Usage Pattern: Batch Vectorized Lookup

For components operating on arrays (e.g., `TankArray` with 8-16 tanks), use `lookup_batch` for SIMD parallelization[file:13][file:14]:

```
import numpy as np

class TankArrayOptimized(Component):
    def __init__(self, n_tanks: int, pressure_bar: float):
        super().__init__()
        self.n_tanks = n_tanks
        self.masses = np.zeros(n_tanks, dtype=np.float64)
        self.pressures = np.full(n_tanks, pressure_bar * 1e5, dtype=np.float64)
        self.temperatures = np.full(n_tanks, 298.15, dtype=np.float64)
        self._lut: LUTManager | None = None

    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        super().initialize(dt, registry)
        self._lut = registry.get("lut_manager")

    def step(self, t: float) -> None:
        super().step(t)
        # Vectorized density lookup for all tanks
        densities = self._lut.lookup_batch('H2', 'D', self.pressures, self.temperatures)
        volumes = self.masses / densities  # Element-wise division
```

### Adding New Fluids to LUT

To extend LUT support for additional species (e.g., CO₂, CH₄, H₂O for mixing)[file:13][file:14]:

1. **Update `LUTConfig` in `lut_manager.py`:**

```
@dataclass
class LUTConfig:
    fluids: Tuple[str, ...] = ('H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O')
    properties: Tuple[PropertyType, ...] = ('D', 'H', 'S', 'C')
    # ... other config
```

2. **Regenerate LUT cache:**

```
from h2_plant.optimization.lut_manager import LUTManager, LUTConfig

config = LUTConfig(fluids=('H2', 'O2', 'N2', 'CO2', 'CH4', 'H2O'))
lut = LUTManager(config)
lut.initialize()  # Generates and caches new tables (~2-5 minutes)
```

3. **Validate accuracy:**

```
report = lut.get_accuracy_report('CO2', num_samples=1000)
print(f"CO2 accuracy: {report['D']['mean_rel_error_pct']:.3f}% error")
# Ensure <0.5% for all properties
```

---

## Thermodynamics: UV-Flash & Phase Equilibrium

### When to Use MultiComponentMixer

The `MultiComponentMixer` component implements rigorous thermodynamics for real-gas mixtures[file:5][file:7]:

- **Phase Equilibrium:** Solves Rachford-Rice equations to detect liquid water condensation
- **UV-Flash:** Determines temperature and pressure from internal energy (U) and volume (V)
- **Performance:** <2 ms per timestep via Numba-optimized flash calculations

**Use Cases:**
- Mixing H₂, O₂, CO₂, H₂O streams (e.g., ATR reforming products)
- Components where phase changes occur (water condensation)
- High-pressure systems where real-gas effects dominate

### Integration Pattern

Upstream components must expose compatible state dictionaries[file:7]:

```
class GasProducerForMixer(Component):
    def get_state(self) -> Dict[str, Any]:
        return {
            **super().get_state(),
            "flow_kmol_hr": float(self.flow_rate),
            "temperature_k": float(self.temperature),
            "pressure_pa": float(self.pressure),
            "composition": {
                "H2": 0.95,
                "H2O": 0.03,
                "CO2": 0.02,
            },
        }
```

The mixer then collects streams via registry and solves for equilibrium[file:7]:

```
mixer = MultiComponentMixer(
    volume_m3=10.0,
    input_source_ids=["electrolyzer", "atr_output"],
    enable_phase_equilibrium=True
)

# In step(t):
mixer.step(t)  # Internally calls _perform_uv_flash()
mixed_state = mixer.get_state()
# mixed_state contains: T, P, liquid_fraction, vapor_composition, etc.
```

### Simplified Mixing (No Phase Change)

For simple oxygen mixing without phase equilibrium[file:7]:

```
from h2_plant.components.utility.oxygen_mixer import OxygenMixer

o2_mixer = OxygenMixer(
    input_source_ids=["electrolyzer_o2", "external_o2"],
    target_flow_kg_h=50.0
)
```

---

## Numba/NumPy Guidelines

### Core Principles

1. **Hot Path Identification:** Profile code to find functions consuming >1% of total runtime[file:13][file:14].
2. **Numba for Loops:** If a hot path contains Python loops over arrays, move to `numba_ops.py` with `@njit`[file:13][file:14].
3. **NumPy for Bulk Operations:** Use vectorized operations (`np.sum`, `np.where`, boolean masks) instead of loops[file:13][file:14].
4. **IntEnum Compatibility:** All enums must be `IntEnum` for Numba (e.g., `TankState.IDLE == 0`)[file:13][file:14].

### Numba Do's and Don'ts

| **Do** | **Don't** |
|--------|-----------|
| Use NumPy arrays (`npt.NDArray`) | Use Python lists or dicts |
| Use integer enums (`TankState.IDLE`) | Use string-based enums |
| Pre-allocate arrays (`np.zeros`, `np.empty`) | Append to lists in loops |
| Return simple types (float, int, tuple) | Return complex Python objects |
| Keep functions pure (no side effects ideal) | Access class attributes (use arguments) |

### Example: Converting to Numba

**Before (Slow Python Loop):**

```
def find_available_tank(tanks: List[Tank]) -> int:
    for i, tank in enumerate(tanks):
        if tank.state == "IDLE" and tank.mass < tank.capacity * 0.99:
            return i
    return -1
```

**After (Fast Numba):**

```
# In numba_ops.py
from numba import njit
import numpy as np

@njit
def find_available_tank(
    states: np.ndarray,  # dtype=np.int32
    masses: np.ndarray,  # dtype=np.float64
    capacities: np.ndarray  # dtype=np.float64
) -> int:
    for i in range(len(states)):
        if states[i] == 0 and masses[i] < capacities[i] * 0.99:  # TankState.IDLE == 0
            return i
    return -1
```

**Usage in Component:**

```
from h2_plant.optimization.numba_ops import find_available_tank

class TankArray(Component):
    def find_fillable_tank(self) -> int:
        return find_available_tank(self.states, self.masses, self.capacities)
```

### NumPy Vectorization Examples

**Task: Update all tank pressures**

```
# ❌ Slow loop
for i in range(self.n_tanks):
    self.pressures[i] = (self.masses[i] / self.volumes[i]) * R * T

# ✓ Fast vectorized
densities = self.masses / self.volumes  # Element-wise
self.pressures = densities * R * T  # Broadcasting
```

**Task: Find tanks matching criteria**

```
# ❌ Slow loop
idle_indices = []
for i, state in enumerate(self.states):
    if state == TankState.IDLE:
        idle_indices.append(i)

# ✓ Fast boolean mask
idle_mask = (self.states == TankState.IDLE)
idle_indices = np.where(idle_mask)
```

**Task: Conditional updates**

```
# ❌ Slow loop
for i in range(len(self.masses)):
    if self.masses[i] >= self.capacities[i] * 0.99:
        self.states[i] = TankState.FULL

# ✓ Fast mask assignment
full_mask = (self.masses >= self.capacities * 0.99)
self.states[full_mask] = TankState.FULL
```

---

## Storage Array Performance: TankArray Pattern

All multi-tank storage components **must** use the `TankArray` pattern for >4 tanks[file:13][file:14]:

### Architecture

```
class TankArray(Component):
    def __init__(self, n_tanks: int, capacity_kg: float, pressure_bar: float):
        super().__init__()
        # Vectorized state arrays
        self.masses = np.zeros(n_tanks, dtype=np.float64)
        self.states = np.full(n_tanks, TankState.IDLE, dtype=np.int32)
        self.capacities = np.full(n_tanks, capacity_kg, dtype=np.float64)
        self.pressures = np.zeros(n_tanks, dtype=np.float64)
        
    def fill(self, mass_kg: float) -> Tuple[float, float]:
        # Use Numba-compiled distribution
        from h2_plant.optimization.numba_ops import distribute_mass_to_tanks
        updated_masses, overflow = distribute_mass_to_tanks(
            mass_kg, self.states, self.masses, self.capacities
        )
        self.masses = updated_masses
        return mass_kg - overflow, overflow
```

### Performance Benchmark

Expected performance on a typical workstation[file:13][file:14]:

| **Operation** | **TankArray (NumPy+Numba)** | **List[Tank] (Python)** | **Speedup** |
|--------------|------------------------------|-------------------------|-------------|
| Find available tank (8 tanks) | 2.35 μs | 23.4 μs | 10x |
| Batch pressure update (1000 tanks) | 0.157 ms | 1.85 ms | 12x |
| Fill operation (distribute mass) | 0.08 ms | 0.95 ms | 12x |

---

## Performance Checklist

Before merging performance-critical code, verify:

### LUT Usage
- [ ] Thermodynamic lookups use `LUTManager` if frequency >100/timestep
- [ ] LUT is resolved from registry in `initialize()`
- [ ] Batch lookups use `lookup_batch()` for array inputs
- [ ] Out-of-bounds cases handled gracefully (logged warnings)

### Numba Compliance
- [ ] Hot path loops moved to `numba_ops.py` with `@njit`
- [ ] All array arguments are NumPy with explicit dtypes
- [ ] Enums converted to integers (not strings)
- [ ] Function signatures use primitive types
- [ ] JIT compilation succeeds without warnings

### NumPy Vectorization
- [ ] Bulk operations use array syntax (no explicit loops)
- [ ] Boolean masks replace conditional loops
- [ ] Pre-allocation used for output arrays
- [ ] Broadcasting applied correctly

### Storage Arrays
- [ ] Multi-tank systems (>4 tanks) use `TankArray` pattern
- [ ] State represented as NumPy arrays (not lists)
- [ ] Numba ops integrated for search/distribution

---

## Troubleshooting

### Issue: LUT KeyError for Fluid

**Symptom:** `ValueError: Fluid 'CO2' not supported`

**Solution:** Extend `LUTConfig` fluids tuple and regenerate cache:

```
config = LUTConfig(fluids=('H2', 'O2', 'N2', 'CO2'))
lut = LUTManager(config)
lut.initialize()
```

### Issue: Numba Compilation Failure

**Symptom:** `TypingError: cannot determine Numba type of <class>`

**Common Causes:**
- Using Python objects (lists, dicts) instead of NumPy arrays
- String-based enums (use `IntEnum`)
- Complex return types (use tuples of primitives)

**Fix:** Convert to NumPy and integer enums:

```
# ❌ Numba can't compile this
@njit
def bad_function(states: list) -> dict:
    return {"result": states}

# ✓ Numba compiles this
@njit
def good_function(states: np.ndarray) -> int:
    return states
```

### Issue: LUT Out-of-Bounds Warnings

**Symptom:** `WARNING: Property lookup out of LUT bounds: P=950 bar`

**Solution:** Either expand LUT range or validate inputs:

```
# Option 1: Expand LUT
config = LUTConfig(pressure_max=1000e5)  # Extend to 1000 bar

# Option 2: Clamp inputs
pressure = np.clip(pressure, 1e5, 900e5)
density = lut.lookup('H2', 'D', pressure, temperature)
```

---

## Performance Profiling

To identify optimization opportunities:

```
# Profile a simulation
python -m cProfile -o profile.stats scripts/run_simulation.py

# Analyze results
python -m pstats profile.stats
> sort cumtime
> stats 20  # Show top 20 functions by cumulative time
```

**Optimization Priority:**
1. Functions consuming >5% of runtime
2. Functions called >10,000 times
3. Nested loops over large arrays

---

## Benchmarking Standards

All performance-critical code must include benchmarks in `tests/benchmarks/`:

```
# tests/benchmarks/bench_my_component.py
import time
import numpy as np
from h2_plant.components.my_component import MyComponent

def benchmark_hot_path(n_iterations=1000):
    component = MyComponent()
    component.initialize(dt=1.0, registry)
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        component.step(0.0)
    elapsed = time.perf_counter() - start
    
    print(f"MyComponent.step(): {elapsed/n_iterations*1000:.4f} ms/call")
    assert elapsed / n_iterations < 0.01, "Performance regression"
```

**Acceptance Criteria:**
- Full simulation (8760 steps) completes in <90 seconds
- Individual component `step()` completes in <0.1 ms average
- Thermodynamic lookups complete in <0.1 ms (LUT)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 20, 2025 | Initial HPC & thermodynamics standards |
```

***

**This completes Step 3.** The `GUIDE_OPTIMIZATION.md` document provides:

1. **LUT Strategy:** Decision tree for when to use cached lookups vs direct CoolProp, with extension instructions for new fluids.[2][1]
2. **Thermodynamics:** Explanation of MultiComponentMixer UV-flash approach and integration patterns for phase equilibrium.[3][4]
3. **Numba/NumPy Guidelines:** Do's and don'ts checklist, conversion examples, and vectorization patterns.[1][2]
4. **Performance Checklist:** Pre-merge verification steps covering LUT, Numba, NumPy, and storage arrays.[2][1]
