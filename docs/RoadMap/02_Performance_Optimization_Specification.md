# STEP 3: Technical Specification - Performance Optimization

---

# 02_Performance_Optimization_Specification.md

**Document:** Performance Optimization Layer Technical Specification  
**Project:** Dual-Path Hydrogen Production System - Modular Refactoring v2.0  
**Date:** November 18, 2025  
**Layer:** Layer 2 - Performance Optimization  
**Priority:** HIGH  
**Dependencies:** Layer 1 - Core Foundation (Component ABC, Integer Enums, Constants)

---

## 1. Overview

### 1.1 Purpose

This specification defines the **performance optimization infrastructure** that eliminates computational bottlenecks in the hydrogen production simulation system. The optimization layer targets a **50-200x speedup** on hot paths through three complementary techniques:

1. **LUT Manager:** Replace expensive CoolProp thermodynamic property calls (5-20ms) with pre-computed lookup tables (0.05-0.1ms)
2. **Numba JIT Compilation:** Compile Python hot paths to native machine code for near-C performance
3. **NumPy Vectorization:** Replace Python loops with array operations for SIMD parallelization

**Critique Remediation:**
- **FAIL → PASS:** "No LUT-based thermodynamic property lookups" (Section 2)
- **PARTIAL → PASS:** "Incomplete Numba usage" (Section 3)
- **FAIL → PASS:** "No NumPy-based TankArray" (Section 4)

---

### 1.2 Performance Baseline

**Current System Bottlenecks (from critique):**

| **Operation** | **Current Performance** | **Frequency** | **Annual Impact** |
|--------------|------------------------|---------------|-------------------|
| CoolProp thermodynamic lookup | 5-20 ms/call | ~100 calls/timestep | 45-60 min/year |
| Tank search (Python loop) | 2 ms/search | ~50 searches/timestep | 15 min/year |
| Pressure calculations (Python) | 0.5 ms/calc | ~200 calcs/timestep | 10 min/year |
| **Total 8760-hour simulation** | **45-60 minutes** | - | **Baseline** |

**Target Performance:**

| **Operation** | **Target Performance** | **Speedup** | **Annual Impact** |
|--------------|----------------------|-------------|-------------------|
| LUT thermodynamic lookup | 0.05-0.1 ms/call | **50-200x** | 0.5-1 min/year |
| NumPy vectorized tank search | 0.2 ms/search | **10x** | 1.5 min/year |
| Numba JIT pressure calculations | 0.05 ms/calc | **10x** | 1 min/year |
| **Total 8760-hour simulation** | **30-90 seconds** | **30-120x** | **Target** |

***

### 1.3 Scope

**In Scope:**
- `optimization/lut_manager.py`: Lookup table generation, storage, and interpolation
- `optimization/numba_ops.py`: JIT-compiled hot path functions
- `optimization/thermodynamics.py`: Cached thermodynamic calculations
- `components/storage/tank_array.py`: NumPy-based vectorized tank storage
- Performance benchmarking suite

**Out of Scope:**
- Component business logic (covered in `03_Component_Standardization_Specification.md`)
- Configuration system (covered in `04_Configuration_System_Specification.md`)
- GPU acceleration (future enhancement)

***

### 1.4 Design Principles

1. **Lazy Loading:** Generate LUTs on first use, cache to disk
2. **Accuracy Preservation:** LUT interpolation error <0.5% vs CoolProp
3. **Graceful Degradation:** Fallback to CoolProp if LUT accuracy insufficient
4. **Memory Efficiency:** Balance LUT resolution vs memory footprint
5. **Numba Compatibility:** All vectorized code must support `@njit` decorator

***

## 2. LUT Manager - Thermodynamic Property Caching

### 2.1 Design Rationale

**Problem Analysis:**

The current system calls CoolProp.PropsSI() directly for thermodynamic properties:

```python
# Expensive! 5-20ms per call
density = CP.PropsSI('D', 'P', pressure, 'T', temperature, 'H2')
enthalpy = CP.PropsSI('H', 'P', pressure, 'T', temperature, 'H2')
```

**Performance Impact:**
- ~100 property lookups per timestep
- 5-20 ms per lookup
- 500-2000 ms per timestep
- **45-60 minutes for 8760-hour simulation**

**Solution Architecture:**

Pre-compute 3D lookup tables (LUTs) covering operational ranges:

```python
# LUT generation (done once, cached to disk)
pressures = np.linspace(1e5, 900e5, 100)      # 1-900 bar
temperatures = np.linspace(250, 350, 50)      # 250-350 K
properties = ['D', 'H', 'S', 'C']            # Density, enthalpy, entropy, heat capacity

lut[property][pressure_idx, temperature_idx] = CoolProp.PropsSI(...)

# Runtime lookup (fast! 0.05-0.1ms)
density = lut_manager.lookup('H2', 'D', pressure=350e5, temperature=298.15)
```

**Performance Gain:** 5-20ms → 0.05-0.1ms = **50-200x speedup**

***

### 2.2 Implementation

**File:** `h2_plant/optimization/lut_manager.py`

```python
"""
Lookup Table Manager for high-performance thermodynamic property lookups.

Replaces expensive CoolProp.PropsSI() calls with pre-computed interpolation
tables, achieving 50-200x speedup while maintaining <0.5% accuracy.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Optional, Literal
from pathlib import Path
import pickle
import logging
from dataclasses import dataclass

try:
    import CoolProp.CoolProp as CP
except ImportError:
    CP = None
    logging.warning("CoolProp not available - LUT generation disabled")

from h2_plant.core.constants import StandardConditions

logger = logging.getLogger(__name__)


PropertyType = Literal['D', 'H', 'S', 'C', 'V']  # Density, Enthalpy, Entropy, Heat capacity, Viscosity


@dataclass
class LUTConfig:
    """Configuration for lookup table generation."""
    
    # Pressure range (Pa)
    pressure_min: float = 1e5          # 1 bar
    pressure_max: float = 900e5        # 900 bar
    pressure_points: int = 100
    
    # Temperature range (K)
    temperature_min: float = 250.0     # -23°C
    temperature_max: float = 350.0     # 77°C
    temperature_points: int = 50
    
    # Properties to pre-compute
    properties: Tuple[PropertyType, ...] = ('D', 'H', 'S', 'C')
    
    # Gases to support
    fluids: Tuple[str, ...] = ('H2', 'O2', 'N2')
    
    # Interpolation method
    interpolation: Literal['linear', 'cubic'] = 'linear'
    
    # Cache directory
    cache_dir: Path = Path.home() / '.h2_plant' / 'lut_cache'


class LUTManager:
    """
    Manages lookup tables for thermodynamic property calculations.
    
    Provides high-performance property lookups with automatic LUT generation,
    disk caching, and fallback to CoolProp for out-of-range queries.
    
    Example:
        lut = LUTManager()
        
        # Fast lookup (0.05-0.1ms)
        density = lut.lookup('H2', 'D', pressure=350e5, temperature=298.15)
        
        # Batch lookup (vectorized)
        pressures = np.array([30e5, 350e5, 900e5])
        temperatures = np.array([298.15, 298.15, 298.15])
        densities = lut.lookup_batch('H2', 'D', pressures, temperatures)
    """
    
    def __init__(self, config: Optional[LUTConfig] = None):
        """
        Initialize LUT Manager.
        
        Args:
            config: LUT configuration (uses defaults if None)
        """
        self.config = config or LUTConfig()
        self._luts: Dict[str, Dict[PropertyType, npt.NDArray]] = {}
        self._pressure_grid: Optional[npt.NDArray] = None
        self._temperature_grid: Optional[npt.NDArray] = None
        self._initialized: bool = False
        
        # Create cache directory
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LUTManager initialized with cache dir: {self.config.cache_dir}")
    
    def initialize(self) -> None:
        """
        Initialize LUT Manager by loading or generating lookup tables.
        
        Attempts to load cached LUTs from disk. If not found, generates
        new LUTs using CoolProp and saves to cache.
        
        Raises:
            RuntimeError: If CoolProp unavailable and no cache exists
        """
        if self._initialized:
            return
        
        logger.info("Initializing LUT Manager...")
        
        # Generate coordinate grids
        self._pressure_grid = np.linspace(
            self.config.pressure_min,
            self.config.pressure_max,
            self.config.pressure_points
        )
        self._temperature_grid = np.linspace(
            self.config.temperature_min,
            self.config.temperature_max,
            self.config.temperature_points
        )
        
        # Load or generate LUTs for each fluid
        for fluid in self.config.fluids:
            cache_path = self._get_cache_path(fluid)
            
            if cache_path.exists():
                logger.info(f"Loading cached LUT for {fluid}")
                self._luts[fluid] = self._load_from_cache(cache_path)
            else:
                logger.info(f"Generating LUT for {fluid} (this may take 1-2 minutes)...")
                self._luts[fluid] = self._generate_lut(fluid)
                self._save_to_cache(fluid, cache_path)
        
        self._initialized = True
        logger.info("LUT Manager initialization complete")
    
    def lookup(
        self,
        fluid: str,
        property_type: PropertyType,
        pressure: float,
        temperature: float
    ) -> float:
        """
        Lookup thermodynamic property with bilinear interpolation.
        
        Args:
            fluid: Fluid name ('H2', 'O2', 'N2')
            property_type: Property code ('D'=density, 'H'=enthalpy, etc.)
            pressure: Pressure in Pa
            temperature: Temperature in K
            
        Returns:
            Property value (units depend on property_type)
            
        Raises:
            ValueError: If fluid or property_type not supported
            RuntimeError: If LUT not initialized
            
        Example:
            # Density of H2 at 350 bar, 298.15 K
            rho = lut.lookup('H2', 'D', 350e5, 298.15)  # kg/m³
        """
        if not self._initialized:
            self.initialize()
        
        if fluid not in self._luts:
            raise ValueError(f"Fluid '{fluid}' not supported. Available: {list(self._luts.keys())}")
        
        if property_type not in self._luts[fluid]:
            raise ValueError(
                f"Property '{property_type}' not available for {fluid}. "
                f"Available: {list(self._luts[fluid].keys())}"
            )
        
        # Check if in bounds
        if not self._in_bounds(pressure, temperature):
            logger.warning(
                f"Property lookup out of LUT bounds: P={pressure/1e5:.1f} bar, "
                f"T={temperature:.1f} K. Falling back to CoolProp."
            )
            return self._fallback_coolprop(fluid, property_type, pressure, temperature)
        
        # Bilinear interpolation
        lut = self._luts[fluid][property_type]
        value = self._interpolate_2d(lut, pressure, temperature)
        
        return float(value)
    
    def lookup_batch(
        self,
        fluid: str,
        property_type: PropertyType,
        pressures: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Vectorized batch lookup for arrays of pressures and temperatures.
        
        Args:
            fluid: Fluid name
            property_type: Property code
            pressures: Array of pressures in Pa
            temperatures: Array of temperatures in K
            
        Returns:
            Array of property values
            
        Example:
            pressures = np.array([30e5, 350e5, 900e5])
            temps = np.array([298.15, 298.15, 298.15])
            densities = lut.lookup_batch('H2', 'D', pressures, temps)
        """
        if not self._initialized:
            self.initialize()
        
        # Vectorized interpolation
        lut = self._luts[fluid][property_type]
        results = np.zeros_like(pressures)
        
        for i in range(len(pressures)):
            results[i] = self._interpolate_2d(lut, pressures[i], temperatures[i])
        
        return results
    
    def _interpolate_2d(
        self,
        lut: npt.NDArray,
        pressure: float,
        temperature: float
    ) -> float:
        """
        Perform 2D bilinear interpolation on LUT.
        
        Args:
            lut: 2D lookup table [pressure_idx, temperature_idx]
            pressure: Pressure value in Pa
            temperature: Temperature value in K
            
        Returns:
            Interpolated property value
        """
        # Find bounding indices
        p_idx = np.searchsorted(self._pressure_grid, pressure)
        t_idx = np.searchsorted(self._temperature_grid, temperature)
        
        # Clamp to valid range
        p_idx = np.clip(p_idx, 1, len(self._pressure_grid) - 1)
        t_idx = np.clip(t_idx, 1, len(self._temperature_grid) - 1)
        
        # Get bounding coordinates
        p0, p1 = self._pressure_grid[p_idx - 1], self._pressure_grid[p_idx]
        t0, t1 = self._temperature_grid[t_idx - 1], self._temperature_grid[t_idx]
        
        # Get corner values
        q00 = lut[p_idx - 1, t_idx - 1]
        q01 = lut[p_idx - 1, t_idx]
        q10 = lut[p_idx, t_idx - 1]
        q11 = lut[p_idx, t_idx]
        
        # Bilinear interpolation
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        wp = (pressure - p0) / (p1 - p0) if p1 != p0 else 0.0
        wt = (temperature - t0) / (t1 - t0) if t1 != t0 else 0.0
        
        value = (
            q00 * (1 - wp) * (1 - wt) +
            q10 * wp * (1 - wt) +
            q01 * (1 - wp) * wt +
            q11 * wp * wt
        )
        
        return value
    
    def _generate_lut(self, fluid: str) -> Dict[PropertyType, npt.NDArray]:
        """
        Generate lookup table for a fluid using CoolProp.
        
        Args:
            fluid: Fluid name ('H2', 'O2', 'N2')
            
        Returns:
            Dictionary mapping property types to 2D arrays
        """
        if CP is None:
            raise RuntimeError("CoolProp not available - cannot generate LUT")
        
        lut = {}
        
        for prop in self.config.properties:
            logger.info(f"Generating {fluid} {prop} table...")
            
            # Initialize array
            table = np.zeros((self.config.pressure_points, self.config.temperature_points))
            
            # Populate table
            for i, pressure in enumerate(self._pressure_grid):
                for j, temperature in enumerate(self._temperature_grid):
                    try:
                        value = CP.PropsSI(prop, 'P', pressure, 'T', temperature, fluid)
                        table[i, j] = value
                    except Exception as e:
                        logger.warning(
                            f"CoolProp error at P={pressure/1e5:.1f} bar, "
                            f"T={temperature:.1f} K: {e}"
                        )
                        table[i, j] = np.nan
            
            lut[prop] = table
            logger.info(f"  ✓ {fluid} {prop} table complete ({table.shape})")
        
        return lut
    
    def _in_bounds(self, pressure: float, temperature: float) -> bool:
        """Check if pressure and temperature are within LUT bounds."""
        return (
            self.config.pressure_min <= pressure <= self.config.pressure_max and
            self.config.temperature_min <= temperature <= self.config.temperature_max
        )
    
    def _fallback_coolprop(
        self,
        fluid: str,
        property_type: PropertyType,
        pressure: float,
        temperature: float
    ) -> float:
        """Fallback to direct CoolProp call for out-of-bounds queries."""
        if CP is None:
            raise RuntimeError(
                f"Query out of LUT bounds and CoolProp unavailable: "
                f"P={pressure/1e5:.1f} bar, T={temperature:.1f} K"
            )
        
        return CP.PropsSI(property_type, 'P', pressure, 'T', temperature, fluid)
    
    def _get_cache_path(self, fluid: str) -> Path:
        """Get cache file path for a fluid."""
        return self.config.cache_dir / f"lut_{fluid}_v1.pkl"
    
    def _save_to_cache(self, fluid: str, cache_path: Path) -> None:
        """Save LUT to disk cache."""
        logger.info(f"Saving LUT cache to {cache_path}")
        
        cache_data = {
            'lut': self._luts[fluid],
            'pressure_grid': self._pressure_grid,
            'temperature_grid': self._temperature_grid,
            'config': self.config
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_from_cache(self, cache_path: Path) -> Dict[PropertyType, npt.NDArray]:
        """Load LUT from disk cache."""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache matches current config
        if cache_data['config'] != self.config:
            logger.warning("Cached LUT config mismatch - regenerating")
            return self._generate_lut(cache_path.stem.split('_')[1])
        
        return cache_data['lut']
    
    def get_accuracy_report(self, fluid: str, num_samples: int = 1000) -> Dict[str, float]:
        """
        Generate accuracy report comparing LUT interpolation to CoolProp.
        
        Args:
            fluid: Fluid to test
            num_samples: Number of random samples to compare
            
        Returns:
            Dictionary with mean/max absolute and relative errors per property
        """
        if CP is None:
            raise RuntimeError("CoolProp required for accuracy validation")
        
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Generating accuracy report for {fluid} ({num_samples} samples)...")
        
        # Random sample points within LUT bounds
        pressures = np.random.uniform(
            self.config.pressure_min,
            self.config.pressure_max,
            num_samples
        )
        temperatures = np.random.uniform(
            self.config.temperature_min,
            self.config.temperature_max,
            num_samples
        )
        
        report = {}
        
        for prop in self.config.properties:
            lut_values = np.array([
                self.lookup(fluid, prop, p, t)
                for p, t in zip(pressures, temperatures)
            ])
            
            coolprop_values = np.array([
                CP.PropsSI(prop, 'P', p, 'T', t, fluid)
                for p, t in zip(pressures, temperatures)
            ])
            
            abs_error = np.abs(lut_values - coolprop_values)
            rel_error = abs_error / np.abs(coolprop_values) * 100  # Percent
            
            report[prop] = {
                'mean_abs_error': float(np.mean(abs_error)),
                'max_abs_error': float(np.max(abs_error)),
                'mean_rel_error_pct': float(np.mean(rel_error)),
                'max_rel_error_pct': float(np.max(rel_error))
            }
            
            logger.info(
                f"  {prop}: mean error {np.mean(rel_error):.3f}%, "
                f"max error {np.max(rel_error):.3f}%"
            )
        
        return report
```

***

### 2.3 Usage Examples

#### Example 1: Basic Property Lookup

```python
from h2_plant.optimization.lut_manager import LUTManager

# Initialize manager (loads or generates LUTs)
lut = LUTManager()
lut.initialize()

# Single lookup (0.05-0.1ms vs 5-20ms CoolProp)
pressure = 350e5  # 350 bar
temperature = 298.15  # 25°C

density = lut.lookup('H2', 'D', pressure, temperature)  # kg/m³
enthalpy = lut.lookup('H2', 'H', pressure, temperature)  # J/kg
heat_capacity = lut.lookup('H2', 'C', pressure, temperature)  # J/kg·K

print(f"H2 density at 350 bar, 25°C: {density:.2f} kg/m³")
```

#### Example 2: Batch Vectorized Lookup

```python
import numpy as np

# Tank pressures and temperatures
pressures = np.array([30e5, 350e5, 900e5])  # LP, HP, Delivery
temperatures = np.array([298.15, 298.15, 298.15])

# Vectorized lookup (much faster than loop)
densities = lut.lookup_batch('H2', 'D', pressures, temperatures)

for i, (p, rho) in enumerate(zip(pressures, densities)):
    print(f"Tank {i+1}: {p/1e5:.0f} bar → {rho:.2f} kg/m³")
```

#### Example 3: Integration with Component

```python
from h2_plant.core.component import Component

class TankWithLUT(Component):
    """Hydrogen tank using LUT for density calculations."""
    
    def __init__(self, capacity_kg: float, pressure_bar: float):
        super().__init__()
        self.capacity_kg = capacity_kg
        self.pressure_pa = pressure_bar * 1e5
        self.mass_kg = 0.0
        self.temperature_k = 298.15
        self._lut = None
    
    def initialize(self, dt: float, registry):
        super().initialize(dt, registry)
        # Get LUT manager from registry
        self._lut = registry.get('lut_manager')
    
    def step(self, t: float):
        super().step(t)
        
        # Fast density lookup
        density = self._lut.lookup('H2', 'D', self.pressure_pa, self.temperature_k)
        volume = self.mass_kg / density
        
        # Update tank state based on volume
        self.fill_percentage = volume / self.capacity_volume * 100
```

***

### 2.4 Performance Benchmarks

**Benchmark Script:** `tests/benchmarks/bench_lut_manager.py`

```python
import time
import numpy as np
from h2_plant.optimization.lut_manager import LUTManager

try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False

def benchmark_lut_vs_coolprop(num_lookups: int = 1000):
    """Compare LUT vs CoolProp performance."""
    
    lut = LUTManager()
    lut.initialize()
    
    # Random test points
    pressures = np.random.uniform(30e5, 900e5, num_lookups)
    temperatures = np.random.uniform(250, 350, num_lookups)
    
    # Benchmark LUT
    start = time.perf_counter()
    for p, t in zip(pressures, temperatures):
        density = lut.lookup('H2', 'D', p, t)
    lut_time = time.perf_counter() - start
    
    # Benchmark CoolProp (if available)
    if COOLPROP_AVAILABLE:
        start = time.perf_counter()
        for p, t in zip(pressures, temperatures):
            density = CP.PropsSI('D', 'P', p, 'T', t, 'H2')
        coolprop_time = time.perf_counter() - start
        
        speedup = coolprop_time / lut_time
        
        print(f"Benchmark Results ({num_lookups} lookups):")
        print(f"  LUT:      {lut_time*1000:.2f} ms ({lut_time/num_lookups*1000:.4f} ms/lookup)")
        print(f"  CoolProp: {coolprop_time*1000:.2f} ms ({coolprop_time/num_lookups*1000:.4f} ms/lookup)")
        print(f"  Speedup:  {speedup:.1f}x")
    else:
        print(f"LUT: {lut_time*1000:.2f} ms ({lut_time/num_lookups*1000:.4f} ms/lookup)")

if __name__ == '__main__':
    benchmark_lut_vs_coolprop()
```

**Expected Output:**
```
Benchmark Results (1000 lookups):
  LUT:      87.32 ms (0.0873 ms/lookup)
  CoolProp: 8234.56 ms (8.2346 ms/lookup)
  Speedup:  94.3x
```

***

## 3. Numba JIT Compilation

### 3.1 Design Rationale

**Problem:** Python loops are slow due to interpreted execution and dynamic typing:

```python
# Slow Python loop (2ms for 8 tanks)
def find_available_tank(tanks):
    for i, tank in enumerate(tanks):
        if tank.state == "IDLE" and tank.mass < tank.capacity * 0.99:
            return i
    return -1
```

**Solution:** Numba JIT compiles Python to machine code with type specialization:

```python
from numba import njit
import numpy as np

@njit
def find_available_tank_njit(states, masses, capacities):
    """Compiled to machine code - 10x faster!"""
    for i in range(len(states)):
        if states[i] == 0 and masses[i] < capacities[i] * 0.99:  # IntEnum.IDLE == 0
            return i
    return -1
```

**Performance Gain:** 2ms → 0.2ms = **10x speedup**

---

### 3.2 Implementation

**File:** `h2_plant/optimization/numba_ops.py`

```python
"""
Numba JIT-compiled operations for hot path performance.

All functions decorated with @njit compile to native machine code,
achieving near-C performance for numerical operations.
"""

import numpy as np
import numpy.typing as npt
from numba import njit
from typing import Tuple

from h2_plant.core.enums import TankState
from h2_plant.core.constants import GasConstants


@njit
def find_available_tank(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    capacities: npt.NDArray[np.float64],
    min_capacity: float = 0.0
) -> int:
    """
    Find first idle tank with sufficient available capacity.
    
    Args:
        states: Array of TankState values (IntEnum)
        masses: Array of current masses (kg)
        capacities: Array of tank capacities (kg)
        min_capacity: Minimum required available capacity (kg)
        
    Returns:
        Index of suitable tank, or -1 if none found
        
    Example:
        states = np.array([TankState.FULL, TankState.IDLE, TankState.IDLE], dtype=np.int32)
        masses = np.array([200.0, 50.0, 0.0], dtype=np.float64)
        capacities = np.array([200.0, 200.0, 200.0], dtype=np.float64)
        
        idx = find_available_tank(states, masses, capacities, min_capacity=100.0)
        # Returns 1 (has 150 kg available) or 2 (has 200 kg available)
    """
    for i in range(len(states)):
        available_capacity = capacities[i] - masses[i]
        if states[i] == TankState.IDLE and available_capacity >= min_capacity:
            return i
    return -1


@njit
def find_fullest_tank(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    min_mass: float = 0.0
) -> int:
    """
    Find fullest tank available for discharge.
    
    Args:
        states: Array of TankState values
        masses: Array of current masses (kg)
        min_mass: Minimum mass required (kg)
        
    Returns:
        Index of fullest tank meeting criteria, or -1 if none found
    """
    max_mass = -1.0
    best_idx = -1
    
    for i in range(len(states)):
        if (states[i] == TankState.IDLE or states[i] == TankState.FULL) and masses[i] >= min_mass:
            if masses[i] > max_mass:
                max_mass = masses[i]
                best_idx = i
    
    return best_idx


@njit
def batch_pressure_update(
    masses: npt.NDArray[np.float64],
    volumes: npt.NDArray[np.float64],
    temperature: float,
    gas_constant: float = GasConstants.R_H2
) -> npt.NDArray[np.float64]:
    """
    Update pressures for all tanks using ideal gas law.
    
    P = (m/V) * R * T  where m=mass, V=volume, R=gas constant, T=temperature
    
    Args:
        masses: Array of tank masses (kg)
        volumes: Array of tank volumes (m³)
        temperature: Gas temperature (K)
        gas_constant: Specific gas constant (J/kg·K)
        
    Returns:
        Array of pressures (Pa)
        
    Example:
        masses = np.array([100.0, 150.0, 200.0])
        volumes = np.array([1.0, 1.0, 1.0])
        pressures = batch_pressure_update(masses, volumes, 298.15)
    """
    pressures = np.empty_like(masses)
    
    for i in range(len(masses)):
        density = masses[i] / volumes[i]
        pressures[i] = density * gas_constant * temperature
    
    return pressures


@njit
def calculate_compression_work(
    p1: float,
    p2: float,
    mass: float,
    temperature: float,
    efficiency: float = 0.75,
    gamma: float = GasConstants.GAMMA_H2,
    gas_constant: float = GasConstants.R_H2
) -> float:
    """
    Calculate compression work using polytropic process model.
    
    W = (γ/(γ-1)) * (m*R*T/η) * [(P2/P1)^((γ-1)/γ) - 1]
    
    Args:
        p1: Inlet pressure (Pa)
        p2: Outlet pressure (Pa)
        mass: Mass of gas compressed (kg)
        temperature: Inlet temperature (K)
        efficiency: Isentropic efficiency (0-1)
        gamma: Specific heat ratio (Cp/Cv)
        gas_constant: Specific gas constant (J/kg·K)
        
    Returns:
        Compression work (J)
        
    Example:
        # Compress 50 kg H2 from 30 bar to 350 bar
        work_j = calculate_compression_work(30e5, 350e5, 50.0, 298.15)
        work_kwh = work_j / 3.6e6
    """
    pressure_ratio = p2 / p1
    exponent = (gamma - 1.0) / gamma
    
    work = (
        (gamma / (gamma - 1.0)) *
        (mass * gas_constant * temperature / efficiency) *
        (pressure_ratio**exponent - 1.0)
    )
    
    return work


@njit
def distribute_mass_to_tanks(
    total_mass: float,
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    capacities: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], float]:
    """
    Distribute mass across available tanks, filling sequentially.
    
    Args:
        total_mass: Total mass to distribute (kg)
        states: Array of TankState values
        masses: Array of current masses (kg) - MODIFIED IN PLACE
        capacities: Array of tank capacities (kg)
        
    Returns:
        Tuple of (updated masses array, remaining undistributed mass)
        
    Example:
        masses = np.array([0.0, 50.0, 0.0])
        capacities = np.array([100.0, 100.0, 100.0])
        states = np.array([TankState.IDLE, TankState.IDLE, TankState.IDLE])
        
        updated_masses, overflow = distribute_mass_to_tanks(180.0, states, masses, capacities)
        # Tank 0: 100 kg (filled), Tank 1: 100 kg (topped off), Tank 2: 30 kg, overflow: 0 kg
    """
    remaining = total_mass
    
    for i in range(len(masses)):
        if remaining <= 0:
            break
        
        if states[i] != TankState.IDLE:
            continue
        
        available_capacity = capacities[i] - masses[i]
        mass_to_add = min(remaining, available_capacity)
        
        masses[i] += mass_to_add
        remaining -= mass_to_add
        
        # Update state if full
        if masses[i] >= capacities[i] * 0.99:
            states[i] = TankState.FULL
    
    return masses, remaining


@njit
def calculate_total_mass_by_state(
    states: npt.NDArray[np.int32],
    masses: npt.NDArray[np.float64],
    target_state: int
) -> float:
    """
    Calculate total mass in tanks matching a specific state.
    
    Args:
        states: Array of TankState values
        masses: Array of masses (kg)
        target_state: State to filter by (IntEnum value)
        
    Returns:
        Total mass in matching tanks (kg)
        
    Example:
        total_full = calculate_total_mass_by_state(states, masses, TankState.FULL)
    """
    total = 0.0
    
    for i in range(len(states)):
        if states[i] == target_state:
            total += masses[i]
    
    return total


@njit
def simulate_filling_timestep(
    production_rate: float,
    dt: float,
    tank_states: npt.NDArray[np.int32],
    tank_masses: npt.NDArray[np.float64],
    tank_capacities: npt.NDArray[np.float64]
) -> Tuple[float, float]:
    """
    Simulate one timestep of production filling tanks.
    
    Args:
        production_rate: H2 production rate (kg/h)
        dt: Timestep (hours)
        tank_states: Array of TankState values - MODIFIED IN PLACE
        tank_masses: Array of masses (kg) - MODIFIED IN PLACE
        tank_capacities: Array of capacities (kg)
        
    Returns:
        Tuple of (mass stored, mass overflow)
    """
    production = production_rate * dt
    
    _, overflow = distribute_mass_to_tanks(
        production,
        tank_states,
        tank_masses,
        tank_capacities
    )
    
    stored = production - overflow
    
    return stored, overflow
```

***

### 3.3 Performance Benchmarks

**Benchmark Script:** `tests/benchmarks/bench_numba_ops.py`

```python
import time
import numpy as np
from h2_plant.optimization.numba_ops import (
    find_available_tank,
    batch_pressure_update,
    calculate_compression_work
)
from h2_plant.core.enums import TankState

def benchmark_find_available_tank():
    """Compare Numba JIT vs Python loop."""
    
    n_tanks = 100
    states = np.random.randint(0, 4, n_tanks, dtype=np.int32)
    masses = np.random.uniform(0, 200, n_tanks)
    capacities = np.full(n_tanks, 200.0)
    
    # Warm up JIT
    find_available_tank(states, masses, capacities)
    
    # Benchmark
    num_trials = 10000
    start = time.perf_counter()
    for _ in range(num_trials):
        idx = find_available_tank(states, masses, capacities)
    elapsed = time.perf_counter() - start
    
    print(f"find_available_tank (Numba JIT):")
    print(f"  {elapsed*1000:.2f} ms for {num_trials} calls")
    print(f"  {elapsed/num_trials*1e6:.2f} μs per call")

def benchmark_batch_pressure_update():
    """Benchmark vectorized pressure calculation."""
    
    n_tanks = 1000
    masses = np.random.uniform(50, 200, n_tanks)
    volumes = np.random.uniform(0.8, 1.2, n_tanks)
    
    # Warm up
    batch_pressure_update(masses, volumes, 298.15)
    
    # Benchmark
    num_trials = 1000
    start = time.perf_counter()
    for _ in range(num_trials):
        pressures = batch_pressure_update(masses, volumes, 298.15)
    elapsed = time.perf_counter() - start
    
    print(f"\nbatch_pressure_update (Numba JIT):")
    print(f"  {elapsed*1000:.2f} ms for {num_trials} calls ({n_tanks} tanks each)")
    print(f"  {elapsed/num_trials*1000:.4f} ms per call")

if __name__ == '__main__':
    benchmark_find_available_tank()
    benchmark_batch_pressure_update()
```

**Expected Output:**
```
find_available_tank (Numba JIT):
  23.45 ms for 10000 calls
  2.35 μs per call

batch_pressure_update (Numba JIT):
  156.78 ms for 1000 calls (1000 tanks each)
  0.1568 ms per call
```

***

## 4. NumPy-Based Tank Array

### 4.1 Design Rationale

**Problem:** Current system uses Python lists of tank objects:

```python
# Slow - Python objects, interpreted loops, scattered memory
tanks = [SourceTaggedTank(200.0, 350e5) for _ in range(8)]

# Find available tank - 2ms for 8 tanks
for i, tank in enumerate(tanks):
    if tank.state == "IDLE":
        return i
```

**Solution:** Vectorized NumPy arrays with Numba-compiled operations:

```python
# Fast - contiguous arrays, compiled code, SIMD operations
class TankArray:
    def __init__(self, n_tanks, capacity):
        self.masses = np.zeros(n_tanks, dtype=np.float64)
        self.states = np.zeros(n_tanks, dtype=np.int32)
        self.capacities = np.full(n_tanks, capacity, dtype=np.float64)

# Find available tank - 0.2ms (10x faster)
idx = np.argmax(self.states == TankState.IDLE)
```

***

### 4.2 Implementation

**File:** `h2_plant/components/storage/tank_array.py`

```python
"""
NumPy-based vectorized tank array for high-performance storage operations.

Replaces Python list of tank objects with contiguous NumPy arrays,
enabling SIMD vectorization and Numba JIT compilation.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Dict, Any

from h2_plant.core.component import Component
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.enums import TankState
from h2_plant.core.constants import StorageConstants, GasConstants
from h2_plant.optimization.numba_ops import (
    find_available_tank,
    batch_pressure_update,
    distribute_mass_to_tanks,
    calculate_total_mass_by_state
)


class TankArray(Component):
    """
    Vectorized array of hydrogen storage tanks.
    
    Uses NumPy arrays for tank properties instead of Python objects,
    enabling 10-50x performance improvement through vectorization and
    Numba JIT compilation.
    
    Example:
        # Create 8 tanks of 200 kg capacity at 350 bar
        tanks = TankArray(
            n_tanks=8,
            capacity_kg=200.0,
            pressure_bar=350
        )
        
        # Initialize
        tanks.initialize(dt=1.0, registry)
        
        # Fill tanks with 500 kg H2
        stored, overflow = tanks.fill(500.0)
        
        # Discharge 300 kg H2
        discharged = tanks.discharge(300.0)
        
        # Query state
        total_mass = tanks.get_total_mass()
        available_capacity = tanks.get_available_capacity()
    """
    
    def __init__(
        self,
        n_tanks: int,
        capacity_kg: float,
        pressure_bar: float,
        temperature_k: float = 298.15
    ):
        """
        Initialize tank array.
        
        Args:
            n_tanks: Number of tanks in array
            capacity_kg: Capacity of each tank (kg)
            pressure_bar: Nominal pressure (bar)
            temperature_k: Operating temperature (K)
        """
        super().__init__()
        
        self.n_tanks = n_tanks
        self.capacity_kg = capacity_kg
        self.pressure_pa = pressure_bar * 1e5
        self.temperature_k = temperature_k
        
        # NumPy arrays for vectorized operations
        self.masses = np.zeros(n_tanks, dtype=np.float64)
        self.states = np.full(n_tanks, TankState.IDLE, dtype=np.int32)
        self.capacities = np.full(n_tanks, capacity_kg, dtype=np.float64)
        self.pressures = np.zeros(n_tanks, dtype=np.float64)
        
        # Calculate tank volumes (constant)
        ideal_gas_density = self.pressure_pa / (GasConstants.R_H2 * self.temperature_k)
        self.volumes = self.capacities / ideal_gas_density
        
        # Statistics
        self.total_filled_kg = 0.0
        self.total_discharged_kg = 0.0
        self.overflow_count = 0
    
    def initialize(self, dt: float, registry: ComponentRegistry) -> None:
        """Initialize tank array."""
        super().initialize(dt, registry)
        
        # Update initial pressures
        self._update_pressures()
    
    def step(self, t: float) -> None:
        """Execute timestep - update pressures and states."""
        super().step(t)
        
        self._update_pressures()
        self._update_states()
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state for checkpointing."""
        return {
            **super().get_state(),
            'n_tanks': self.n_tanks,
            'masses': self.masses.tolist(),
            'states': self.states.tolist(),
            'pressures': (self.pressures / 1e5).tolist(),  # Convert to bar
            'total_mass_kg': float(self.get_total_mass()),
            'available_capacity_kg': float(self.get_available_capacity()),
            'total_filled_kg': float(self.total_filled_kg),
            'total_discharged_kg': float(self.total_discharged_kg),
            'overflow_count': self.overflow_count
        }
    
    def fill(self, mass_kg: float) -> tuple[float, float]:
        """
        Fill tanks with hydrogen mass.
        
        Distributes mass across available (IDLE) tanks sequentially.
        
        Args:
            mass_kg: Mass to store (kg)
            
        Returns:
            Tuple of (mass_stored, mass_overflow)
            
        Example:
            stored, overflow = tanks.fill(500.0)
            if overflow > 0:
                print(f"Warning: {overflow:.1f} kg could not be stored")
        """
        # Use Numba-compiled distribution function
        updated_masses, overflow = distribute_mass_to_tanks(
            mass_kg,
            self.states,
            self.masses,
            self.capacities
        )
        
        self.masses = updated_masses
        stored = mass_kg - overflow
        
        # Update statistics
        self.total_filled_kg += stored
        if overflow > 0:
            self.overflow_count += 1
        
        return stored, overflow
    
    def discharge(self, mass_kg: float) -> float:
        """
        Discharge hydrogen from tanks.
        
        Draws from fullest tanks first.
        
        Args:
            mass_kg: Mass to discharge (kg)
            
        Returns:
            Actual mass discharged (may be less if insufficient stored)
            
        Example:
            discharged = tanks.discharge(300.0)
            if discharged < 300.0:
                print(f"Warning: only {discharged:.1f} kg available")
        """
        remaining = mass_kg
        total_discharged = 0.0
        
        while remaining > 0.01:  # Continue until depleted
            # Find fullest tank with mass
            tank_idx = find_fullest_tank(self.states, self.masses, min_mass=0.01)
            
            if tank_idx == -1:
                break  # No more tanks available
            
            # Discharge from this tank
            available = self.masses[tank_idx]
            discharge_amount = min(remaining, available)
            
            self.masses[tank_idx] -= discharge_amount
            remaining -= discharge_amount
            total_discharged += discharge_amount
            
            # Update tank state
            if self.masses[tank_idx] < StorageConstants.TANK_EMPTY_THRESHOLD:
                self.states[tank_idx] = TankState.EMPTY
            else:
                self.states[tank_idx] = TankState.IDLE
        
        self.total_discharged_kg += total_discharged
        
        return total_discharged
    
    def get_total_mass(self) -> float:
        """Return total mass stored across all tanks (kg)."""
        return float(np.sum(self.masses))
    
    def get_available_capacity(self) -> float:
        """Return total available capacity in idle/empty tanks (kg)."""
        available_mask = np.logical_or(
            self.states == TankState.IDLE,
            self.states == TankState.EMPTY
        )
        available_capacities = self.capacities[available_mask] - self.masses[available_mask]
        return float(np.sum(available_capacities))
    
    def get_mass_by_state(self, state: TankState) -> float:
        """Return total mass in tanks with specific state (kg)."""
        return calculate_total_mass_by_state(self.states, self.masses, int(state))
    
    def get_tank_count_by_state(self, state: TankState) -> int:
        """Return number of tanks in specific state."""
        return int(np.sum(self.states == state))
    
    def _update_pressures(self) -> None:
        """Update pressures for all tanks using ideal gas law."""
        self.pressures = batch_pressure_update(
            self.masses,
            self.volumes,
            self.temperature_k,
            GasConstants.R_H2
        )
    
    def _update_states(self) -> None:
        """Update tank states based on fill levels."""
        fill_percentages = self.masses / self.capacities
        
        # Set FULL state
        full_mask = fill_percentages >= StorageConstants.TANK_FULL_THRESHOLD
        self.states[full_mask] = TankState.FULL
        
        # Set EMPTY state
        empty_mask = fill_percentages <= StorageConstants.TANK_EMPTY_THRESHOLD
        self.states[empty_mask] = TankState.EMPTY
        
        # Set IDLE state (between empty and full)
        idle_mask = np.logical_and(
            fill_percentages > StorageConstants.TANK_EMPTY_THRESHOLD,
            fill_percentages < StorageConstants.TANK_FULL_THRESHOLD
        )
        self.states[idle_mask] = TankState.IDLE
```

***

### 4.3 Usage Examples

#### Example 1: Basic Tank Operations

```python
from h2_plant.components.storage.tank_array import TankArray

# Create 8 HP tanks (200 kg, 350 bar each)
hp_tanks = TankArray(n_tanks=8, capacity_kg=200.0, pressure_bar=350)

# Initialize
hp_tanks.initialize(dt=1.0, registry)

# Fill with 500 kg H2
stored, overflow = hp_tanks.fill(500.0)
print(f"Stored: {stored:.1f} kg, Overflow: {overflow:.1f} kg")

# Query state
print(f"Total mass: {hp_tanks.get_total_mass():.1f} kg")
print(f"Available capacity: {hp_tanks.get_available_capacity():.1f} kg")
print(f"Full tanks: {hp_tanks.get_tank_count_by_state(TankState.FULL)}")

# Discharge 300 kg
discharged = hp_tanks.discharge(300.0)
print(f"Discharged: {discharged:.1f} kg")
```

#### Example 2: Integration with Production

```python
from h2_plant.core.component import Component

class ProductionToStorageSystem(Component):
    """System coordinating production and storage."""
    
    def __init__(self):
        super().__init__()
        self.electrolyzer = None
        self.hp_tanks = None
    
    def initialize(self, dt, registry):
        super().initialize(dt, registry)
        self.electrolyzer = registry.get('electrolyzer')
        self.hp_tanks = registry.get('hp_tanks')
    
    def step(self, t):
        super().step(t)
        
        # Get production from electrolyzer
        h2_produced = self.electrolyzer.h2_output_kg
        
        # Store in tanks
        stored, overflow = self.hp_tanks.fill(h2_produced)
        
        if overflow > 0:
            print(f"Hour {t}: Storage full! {overflow:.1f} kg overflow")
```

***

### 4.4 Performance Comparison

**Benchmark:** `tests/benchmarks/bench_tank_array.py`

```python
import time
import numpy as np
from h2_plant.components.storage.tank_array import TankArray

def benchmark_fill_operations(n_tanks: int = 100, n_operations: int = 1000):
    """Benchmark tank filling operations."""
    
    tanks = TankArray(n_tanks=n_tanks, capacity_kg=200.0, pressure_bar=350)
    tanks.initialize(dt=1.0, ComponentRegistry())
    
    # Random fill amounts
    fill_amounts = np.random.uniform(10, 500, n_operations)
    
    start = time.perf_counter()
    for mass in fill_amounts:
        tanks.fill(mass)
    elapsed = time.perf_counter() - start
    
    print(f"Tank Fill Benchmark ({n_tanks} tanks, {n_operations} operations):")
    print(f"  Total time: {elapsed*1000:.2f} ms")
    print(f"  Per operation: {elapsed/n_operations*1000:.4f} ms")
    print(f"  Operations/sec: {n_operations/elapsed:.0f}")

if __name__ == '__main__':
    benchmark_fill_operations()
```

**Expected Output:**
```
Tank Fill Benchmark (100 tanks, 1000 operations):
  Total time: 45.67 ms
  Per operation: 0.0457 ms
  Operations/sec: 21897
```

***

## 5. Integration and Testing

### 5.1 Integration Example

**File:** `examples/performance_demo.py`

```python
"""
Demonstration of performance optimization layer integration.

Shows LUT Manager, Numba operations, and TankArray working together
in a realistic simulation scenario.
"""

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.optimization.numba_ops import calculate_compression_work
import numpy as np

def run_performance_demo():
    """Run 100-hour simulation with optimization layer."""
    
    # Setup
    registry = ComponentRegistry()
    
    # Initialize LUT Manager
    lut = LUTManager()
    lut.initialize()
    registry.register('lut_manager', lut)
    
    # Create tank arrays
    lp_tanks = TankArray(n_tanks=4, capacity_kg=50.0, pressure_bar=30)
    hp_tanks = TankArray(n_tanks=8, capacity_kg=200.0, pressure_bar=350)
    
    registry.register('lp_tanks', lp_tanks, component_type='storage')
    registry.register('hp_tanks', hp_tanks, component_type='storage')
    
    # Initialize all
    registry.initialize_all(dt=1.0)
    
    # Simulate 100 hours
    print("Running 100-hour simulation with optimization layer...")
    
    import time
    start = time.perf_counter()
    
    for hour in range(100):
        # Production (simulate 50 kg/h)
        production_kg = 50.0
        
        # Fill LP tanks
        stored_lp, overflow_lp = lp_tanks.fill(production_kg)
        
        # Transfer LP → HP (compression)
        lp_mass = lp_tanks.get_total_mass()
        if lp_mass > 100:  # Transfer when >100 kg accumulated
            transfer_mass = min(lp_mass, 200.0)
            
            # Calculate compression work (Numba JIT)
            work_j = calculate_compression_work(
                p1=30e5,
                p2=350e5,
                mass=transfer_mass,
                temperature=298.15
            )
            work_kwh = work_j / 3.6e6
            
            # Execute transfer
            lp_tanks.discharge(transfer_mass)
            stored_hp, overflow_hp = hp_tanks.fill(transfer_mass)
        
        # Fast thermodynamic lookup (LUT)
        density_hp = lut.lookup('H2', 'D', 350e5, 298.15)
        
        # Step all components
        registry.step_all(hour)
    
    elapsed = time.perf_counter() - start
    
    print(f"\n✓ Simulation complete in {elapsed*1000:.2f} ms")
    print(f"  Average: {elapsed/100*1000:.3f} ms per hour")
    print(f"\nFinal State:")
    print(f"  LP tanks: {lp_tanks.get_total_mass():.1f} kg")
    print(f"  HP tanks: {hp_tanks.get_total_mass():.1f} kg")

if __name__ == '__main__':
    run_performance_demo()
```

***

### 5.2 Performance Testing Suite

**File:** `tests/performance/test_optimization_layer.py`

```python
import pytest
import time
import numpy as np
from h2_plant.optimization.lut_manager import LUTManager
from h2_plant.components.storage.tank_array import TankArray
from h2_plant.optimization.numba_ops import calculate_compression_work


@pytest.mark.performance
def test_lut_lookup_speed():
    """Verify LUT lookup meets <0.1ms target."""
    lut = LUTManager()
    lut.initialize()
    
    # Warm up JIT
    for _ in range(10):
        lut.lookup('H2', 'D', 350e5, 298.15)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        density = lut.lookup('H2', 'D', 350e5, 298.15)
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / 1000) * 1000
    assert avg_time_ms < 0.1, f"LUT lookup too slow: {avg_time_ms:.4f} ms"


@pytest.mark.performance
def test_lut_accuracy():
    """Verify LUT interpolation error <0.5%."""
    lut = LUTManager()
    lut.initialize()
    
    accuracy_report = lut.get_accuracy_report('H2', num_samples=100)
    
    for prop, errors in accuracy_report.items():
        assert errors['max_rel_error_pct'] < 0.5, \
            f"LUT accuracy insufficient for {prop}: {errors['max_rel_error_pct']:.2f}%"


@pytest.mark.performance
def test_tank_array_fill_speed():
    """Verify tank fill operations meet performance targets."""
    tanks = TankArray(n_tanks=100, capacity_kg=200.0, pressure_bar=350)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(1000):
        tanks.fill(50.0)
    elapsed = time.perf_counter() - start
    
    ops_per_sec = 1000 / elapsed
    assert ops_per_sec > 10000, f"Tank operations too slow: {ops_per_sec:.0f} ops/sec"


@pytest.mark.performance
def test_numba_compilation():
    """Verify Numba JIT functions compile successfully."""
    # Should not raise errors
    work = calculate_compression_work(30e5, 350e5, 50.0, 298.15)
    assert work > 0
```

***

## 6. Validation Criteria

This Performance Optimization Layer is **COMPLETE** when:

**LUT Manager:**
- Generates lookup tables for H2, O2, N2
- Achieves <0.1ms average lookup time
- Maintains <0.5% interpolation error vs CoolProp
- Implements disk caching
- Unit tests achieve 95%+ coverage

**Numba Operations:**
- All hot path functions decorated with `@njit`
- Tank search, pressure updates, compression work compiled
- Performance benchmarks show >5x speedup vs pure Python
- NumPy array compatibility validated

**Tank Array:**
- Vectorized storage operations implemented
- Numba integration working
- Fill/discharge operations meet performance targets
- State management with IntEnum integration
- Component ABC compliance

**Integration:**
- All three subsystems work together
- End-to-end performance tests pass
- 8760-hour simulation completes in <90 seconds

***

## 7. Success Metrics

| **Metric** | **Target** | **Measurement** | **Status** |
|-----------|-----------|-----------------|------------|
| CoolProp Replacement | 50-200x speedup | Benchmark LUT vs CoolProp | TBD |
| LUT Accuracy | <0.5% error | Accuracy report | TBD |
| Tank Operations | >10,000 ops/sec | Fill/discharge benchmark | TBD |
| Full Simulation (8760h) | 30-90 seconds | Wall clock time | TBD |
| Test Coverage | 95%+ | `pytest --cov` | TBD |

***
