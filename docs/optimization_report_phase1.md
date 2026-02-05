# H2 Plant Simulation - Phase 1 Optimization Report

**Date:** 2026-02-04
**Baseline Runtime:** 622.6 seconds (30-day simulation, 43,200 timesteps)
**Final Result:** All optimizations reverted due to performance regression

---

## Executive Summary

Phase 1 optimizations were implemented with the goal of achieving 30-50% performance improvement. Instead, they caused a **+16.7% regression** (726.4s vs 622.6s baseline). After thorough profiling analysis, all changes were reverted. This report documents the optimizations attempted, measured results, and lessons learned.

---

## 1. Optimizations Implemented

### P1.4: Stream Object Reuse

**Goal:** Reduce the 8 million `Stream` object allocations per simulation by reusing stream instances in hot components.

**Changes Made:**
- Added `Stream.update()` method for in-place property updates
- Added `Stream._invalidate_cache()` for explicit cache management
- Modified `compressor_single.py`, `dry_cooler.py`, and `hydrogen_cyclone.py` to reuse cached stream objects instead of creating new ones

**Code Example (before):**
```python
def get_output(self, port_name):
    return Stream(
        mass_flow_kg_h=out_mass_flow,
        temperature_k=self.outlet_temperature_k,
        pressure_pa=self.actual_outlet_pressure_pa,
        composition=out_comp
    )
```

**Code Example (after):**
```python
def get_output(self, port_name):
    if self._output_stream is None:
        self._output_stream = Stream(...)
    else:
        self._output_stream.update(
            mass_flow_kg_h=out_mass_flow,
            temperature_k=self.outlet_temperature_k,
            ...
        )
    return self._output_stream
```

### P1.5: Numba Explicit Type Signatures

**Goal:** Eliminate Numba's runtime type inference overhead (~20s) by adding explicit type signatures.

**Changes Made:**
- Changed `@njit(cache=True)` to `@njit('Tuple((f8[:], f8, f8))(f8[:])', cache=True)`
- Applied to: `fast_composition_properties`, `bilinear_interp_jit`, `calculate_water_psat_jit`, `solve_rachford_rice_single_condensable`

### P1.6: Pre-bound Attribute Accessors

**Goal:** Replace dynamic `getattr()` calls with pre-bound `operator.attrgetter()` for faster metric recording.

**Changes Made:**
- Added `StreamRecorder.extra_metric_getters` list
- Added `bind_extra_metric_accessors()` method
- Modified `record_post_step()` to use pre-bound getters

### P1.2/P1.3: Flow Network Optimizations

**Goal:** Reduce flow tracking overhead and cache source port outputs.

**Changes Made:**
- Added `flow_tracking_mode` configuration ('per_step', 'hourly_aggregate', 'off')
- Grouped connections by source port for single `get_output()` call per port
- Added `_execute_single_flow_with_output()` method

---

## 2. Profiling Results

### 2.1 Overall Performance

| Metric | Baseline | After P1 | Change |
|--------|----------|----------|--------|
| **Total Runtime** | 622.6s | 726.4s | **+103.8s (+16.7%)** |
| Total Function Calls | 542M | 587M | +45M |
| Time per Step | 14.4ms | 16.8ms | +2.4ms |

### 2.2 Individual Optimization Results

#### P1.4 (Stream Reuse): NET REGRESSION +11.5s

| Metric | Baseline | After | Impact |
|--------|----------|-------|--------|
| `Stream.__post_init__` calls | 8,064,075 | 4,654,086 | -42% (good) |
| `Stream.__post_init__` time | 16.9s | 11.4s | -5.5s (savings) |
| `Stream.update()` calls | 0 | 3,409,943 | NEW |
| `Stream.update()` time | 0 | 17.0s | **+17.0s (overhead)** |
| **Net Effect** | - | - | **+11.5s worse** |

**Root Cause:** The `update()` method was slower than allocation because:
1. Always called `sum()` for composition normalization (3.4M calls)
2. Always called `_invalidate_cache()` (3.5s overhead)
3. Dict operations for setting composition values

#### P1.5 (Numba Signatures): REGRESSION +42s

| Metric | Baseline | After | Impact |
|--------|----------|-------|--------|
| `_compile_for_args` | 48.7s | 83.7s | **+35.0s** |
| `typeof_pyval` | ~20s | 27.3s | +7.3s |

**Root Cause:** The Numba cache was not cleared after adding explicit signatures. Stale cached functions caused signature mismatch detection and recompilation.

#### P1.6 (Pre-bound Accessors): REGRESSION +35s (fixed, then reverted)

| Metric | Baseline | After | Impact |
|--------|----------|-------|--------|
| `getter_fn` calls | 0 | 2,116,800 | NEW |
| `getter_fn` time | 0 | 35.0s | **+35s overhead** |

**Root Cause:** The fallback `getter_fn` always called `get_state()` instead of only when `getattr()` returned None. Most metrics come from `get_state()`, so the slow path was hit 100% of the time.

#### P1.2/P1.3 (Flow Network): REGRESSION +15s

| Metric | Baseline | After | Impact |
|--------|----------|-------|--------|
| `execute_flows` | 100.8s | 115.6s | +14.8s |
| `_execute_single_flow_with_output` | N/A | 94.5s | NEW function |

**Root Cause:** The source port grouping added overhead without providing benefits because most ports have only one outgoing connection.

### 2.3 Breakdown of Total Regression

| Source | Impact |
|--------|--------|
| P1.4 Stream.update() overhead | +11.5s |
| Numba stale cache | +42.0s |
| Flow network changes | +15.0s |
| Component step overhead | +20.0s |
| Other (composition arrays, recording) | +15.0s |
| **Total** | **+103.8s** |

---

## 3. Why We Reverted

### 3.1 Python's Allocator is Highly Optimized

The assumption that "object reuse saves allocation time" was incorrect. Python's memory allocator is extremely fast for small objects. The overhead of:
- Checking if object exists
- Calling `update()` method
- Normalizing composition with `sum()`
- Invalidating caches

...exceeded the cost of simply creating a new `Stream` object.

### 3.2 Defensive Code Has Hidden Costs

The `update()` method included defensive normalization:
```python
total_fraction = sum(composition.values())
if abs(total_fraction - 1.0) > 1e-3 and total_fraction > 0:
    for k in self.composition:
        self.composition[k] /= total_fraction
```

Called 3.4 million times, this added ~5 seconds of overhead. In the original code, normalization happens once at construction in `__post_init__`.

### 3.3 Numba Cache Management is Critical

Adding explicit type signatures should reduce dispatch overhead, but the old cached functions (.nbc files) were still being loaded. This caused Numba to:
1. Load the old cached function
2. Detect signature mismatch
3. Recompile with the new signature
4. Repeat on every invocation

**Lesson:** Always run `rm -rf ~/.cache/numba/` after changing Numba function signatures.

### 3.4 Profiling Masked Other Regressions

The P1.6 regression (+35s) was so large that it masked the P1.4 regression (+11.5s) in the first profiling run. After fixing P1.6, the total runtime got worse because the P1.4 regression was now visible.

**Lesson:** Profile after each individual change, not after batch implementation.

---

## 4. Files Modified and Reverted

| File | Changes Made | Status |
|------|--------------|--------|
| `h2_plant/core/stream.py` | Added `update()`, `_invalidate_cache()` | Reverted |
| `h2_plant/components/compression/compressor_single.py` | Added `_output_stream` caching | Reverted |
| `h2_plant/components/cooling/dry_cooler.py` | Added stream reuse in `step()` | Reverted |
| `h2_plant/components/separation/hydrogen_cyclone.py` | Added stream reuse in `step()` | Reverted |
| `h2_plant/optimization/numba_ops.py` | Added explicit type signatures | Reverted |
| `h2_plant/control/engine_dispatch.py` | Added pre-bound accessors | Reverted |
| `h2_plant/simulation/flow_network.py` | Added source port grouping | Reverted |
| `h2_plant/simulation/engine.py` | Added flow_tracking_mode config | Reverted |
| `h2_plant/config/models.py` | Added config fields | Reverted |
| `scenarios/simulation_config.yaml` | Added P1.1/P1.2 options | Reverted |
| `~/.cache/numba/` | - | Cleared |

---

## 5. Recommendations for Future Optimization

### 5.1 Config-Only Optimizations (Low Risk)

Focus on **P1.1 (history detail level)** and **P1.2 (flow tracking modes)** which only affect logging/recording, not physics:
- `history_detail_level: summary` - Skip per-stream mole fraction recording
- `flow_tracking_mode: off` - Disable flow tracking for long runs

### 5.2 Data-Oriented Composition (Phase 2)

Instead of reusing objects, change the data structure:
- Replace `Dict[str, float]` with fixed-size `np.ndarray` for composition
- Eliminate 110M `dict.get()` calls
- This addresses the root cause rather than the symptom

### 5.3 Profile-Driven Development

1. Profile baseline before any changes
2. Implement ONE optimization at a time
3. Profile immediately after each change
4. Only proceed if measurable improvement
5. Clear all caches (Numba, Python) between tests

### 5.4 Avoid Defensive Overhead in Hot Paths

For methods called millions of times:
- Skip validation that's already guaranteed by callers
- Use assertions (disabled in production) instead of runtime checks
- Trust internal code paths

---

## 6. Conclusion

Phase 1 optimizations failed due to incorrect assumptions about Python's performance characteristics:

1. **Object reuse is not always faster** - Python's allocator is highly optimized
2. **In-place updates have hidden costs** - Validation, cache invalidation, dict operations
3. **Cache management is critical** - Stale Numba cache caused +42s regression
4. **Batch implementation masks issues** - Individual profiling is essential

The simulation has been restored to baseline (622.6s). Future optimization efforts should focus on config-only changes and Phase 2 data-oriented design.

---

*Report generated after reverting all Phase 1 changes and clearing Numba cache.*
