# H2 Plant Simulation — Performance Optimization Plan v2

**Date:** 2026-02-01
**Revision:** 2 — incorporates peer review corrections from two developer reviews.
**Benchmark:** 168-hour simulation (10,080 steps at 1-min resolution)
**Baseline Metrics:**

| Metric | Value |
|---|---|
| Total Runtime | 218.51 s |
| Simulation Loop Time | 212.42 s |
| Avg Step Time | 21.07 ms |
| Max Step Time | 1,906 ms |
| Memory Start → End | 1,590 → 1,850 MB (+260 MB) |
| Function Calls (loop) | ~194M |

**Baseline source:** `benchmark_output/benchmark_report.txt` captured on **2026-02-01**. Re-baseline after the benchmark harness fix (see below); treat current savings estimates as upper bounds until then, and refresh per-function timings accordingly.

---

## Important: Benchmark vs Production Caveat

The benchmark harness (`benchmark_suite.py:166-180`) uses a **different execution path** than the production engine (`engine.py:266-324`):

| Aspect | Benchmark (`benchmark_suite.py`) | Production (`engine.run()`) |
|---|---|---|
| Component stepping | `registry.step_all()` with per-component try/except | Pre-resolved `execution_list` — bare loop, no try/except |
| Monitoring | `collect()` called **every step** (10,080 calls) | Gated to **hourly** (`step % steps_per_hour == 0`, 168 calls) |
| `get_all_states()` | Called 10,080 times (from monitoring) | Called 168 times |
| FlowTracker flush | Often disabled in benchmarks (accumulates flow records) — **must flush every 24h** to match production | Flushed to disk every 24 simulated hours (`engine.py:322-323`) |

**Consequence:** ~50–60 s of the 212 s loop time is benchmark-only overhead (`step_all` try/except: ~2.8 s, `collect` + `get_all_states`: ~31 s, extra `get_state()` on components: ~14 s). The estimated **production loop time** for this workload is ~150–165 s. Part of the +260 MB memory growth is also benchmark-specific (per-step monitoring appends and any unflushed FlowTracker buffer).

**Recommendation:** Update `benchmark_suite.py` to mirror `engine.run()` (use the pre-resolved execution list, gate monitoring hourly, flush FlowTracker periodically) so future profiles reflect production behavior. All optimizations below are evaluated against the **production path** unless noted otherwise.

---

## Implementation Phases

Optimizations are grouped into two phases with a mandatory re-profile gate between them. This avoids double-counting savings for items in shared call chains (notably P7 overlaps with P1/P2 callers).

### Phase 1 — Low-Risk, No Physics Changes

| ID | Target |
|---|---|
| P0 | JIT pre-warming |
| P2 | ATR `interp1d` replacement |
| P3 | CoolProp rounding (short-term) |
| P4 | `record_post_step` reflection |
| P9 | Flow network dispatch |
| P9.1 | FlowTracker flush/serialization overhead |
| P11 | Saturation properties caching |
| P12 | LUT lookup fast path (pre‑binding) |

**After Phase 1:** Fix the benchmark harness, re-profile on the production path. Use the new profile to validate Phase 2 savings estimates and re-scope P7.

### Phase 1 Completion Status (2026-02-02)

**Implemented:**
- **P0** JIT warm-up added in `benchmark_suite.py` (warm-up is optional in `engine.run()` and remains unimplemented there).
- **P2** ATR lookup now uses `np.interp` + explicit linear extrapolation (`atr_data_manager.py`).
- **P3a** CoolProp quantization replaced `_round_sig` with fixed‑resolution bins (`coolprop_lut.py`).
- **P4** `record_post_step` reflection reduced via pre‑resolved attributes and buffered writes (`engine_dispatch.py`).
- **P9.1** FlowTracker serialization optimized (manual dict, cached matrix keys, batch JSON, rolling flush).
- **P11** `LUTManager.get_saturation_properties()` now memoizes **exact** pressure (single‑entry cache).

**Partial / still to close:**
- **P9** pre‑binding of connection callables (`_resolved_flows`) not implemented; only signal/flow split and flow‑type pre‑mapping done. Either finish pre‑binding or adjust expected savings.
- **P12** `LUTManager.bind_lookup()` exists, but no hot components use it yet. Integrate into fixed‑fluid/property callers (compressors, valves, pumps) and re‑profile.
- **P11 (optional)** component‑level memoization not added; current cache is global last‑pressure only (safe but less effective).

**Re‑profile gate:** Benchmark harness now mirrors `engine.run()`; a fresh profile is still required before Phase 2.

### Phase 2 — Physics-Touching Refactors (Require Validation)

| ID | Target |
|---|---|
| P1 | Interchanger JIT flash |
| P1.5 | Wire `mixture_thermodynamics` to existing JIT kernels |
| P5 | Dry cooler fused JIT |
| P6 | Mixer PH-flash JIT |
| P7 | `numpy.interp` scalar overhead (re-scoped after Phase 1 re-profile) |
| P8 | Stream creation overhead (fast-path `from_arrays`) |
| P10 | LUT float64 → float32 |
| P3b | CoolProp → LUT routing (medium-term) |

---

## Priority-Ordered Optimization Tasks

### P0 — Pre-warm Numba JIT Kernels Before Timing

**Evidence:** `numba.core.dispatcher._compile_for_args` = 12.8 s, `compile` = 5.4 s in the profile. Total: **18.2 s** of JIT compilation happening inside the timed loop. The 1,906 ms max-step outlier is almost certainly a JIT compilation event.

**Root Cause:** Some Numba-decorated functions are compiled on first call with specific type signatures. If the benchmark (or production run) encounters a new type signature mid-loop, compilation is triggered.

**Clarification:** Pre-warming **shifts** compile time from the loop into initialization — it does not eliminate it from total wall-clock time. In production, this means the user pays the cost during startup instead of experiencing jitter mid-simulation. The primary value is eliminating the 1,906 ms max-step outlier and making step times predictable.

**Files:**
- `benchmark_suite.py` — add warm-up calls before the timing loop
- `h2_plant/optimization/numba_ops.py` — add a `warmup()` function

**Action:**
0. **Warm-up coverage audit (NEW):** Enumerate all `@njit` functions in `numba_ops.py` and identify their production call signatures (dtype, shape, contiguity). Use this list to drive warm-up calls so every runtime specialization is pre-compiled.
1. Create a `warmup_jit_kernels(lut_mgr)` function in `numba_ops.py` that invokes every exported `@njit` function once with **representative inputs matching production types**. This means:
   - Use the actual `lut_mgr.stacked_H` (float64, C-contiguous) and `lut_mgr._pressure_grid` arrays, not synthetic 1-element arrays. Numba compiles per type signature — a warm-up with different dtypes, contiguity, or array dimensions will compile a *different* specialization and miss the production path entirely.
   - Pass scalar float64 values for P, T, etc.
2. Call `warmup_jit_kernels()` after engine initialization but before the simulation loop in `benchmark_suite.py`. In production (`engine.py`), make this **optional** (e.g., `engine.run(warmup_jit=True)`) — do not unconditionally add 18s to every startup if the Numba cache is already populated.
3. Ensure all `@njit` functions use `cache=True` (most already do). On second and subsequent runs, cached `.nbi` files skip compilation entirely, making warm-up unnecessary. The warm-up function is primarily needed for first-run or after code changes.

**Expected Savings:** ~18 s shifted from loop to init. Max-step outlier drops from 1,906 ms to <50 ms. Step-time variance reduced.
**Effort:** Low.

---

### P1 — JIT-Compile the Interchanger Flash Bisection

**Evidence:** `interchanger.py:79 step()` = **36.3 s** cumtime (30,240 calls = 3 instances × 10,080 steps). This cost is identical in production and benchmark.

**Root Cause:** Each `step()` runs a Python-level bisection loop (up to 50 iterations) where every iteration:
- Computes Antoine equation in Python
- Calls `solve_rachford_rice_single_condensable` (JIT — fast, but Python→C boundary per call)
- Loops over a composition dict doing individual `lut_mgr.lookup()` calls (`interchanger.py:373`) — each lookup dispatches through Python validation, dict lookups, and bounds checks before reaching the JIT interpolator
- Performs dict copies, `sum()` on dict values, and string key comparisons
- Calls `lut_mgr.get_saturation_properties()` at line 273, whose return value `sat_props` is **never used** — dead work in every iteration

Additional issues:
- `from h2_plant.optimization.numba_ops import ...` is inside `step()` (line 229) — import machinery runs every call
- Composition arrays are rebuilt from dicts every step (lines 183-191) instead of using cached arrays

**Files:**
- `h2_plant/optimization/numba_ops.py` — new function
- `h2_plant/components/thermal/interchanger.py` — refactor `step()`

**Critical Implementation Detail — Species Order Mapping:**

Three different species orderings exist in the codebase:

| Context | Order | Length |
|---|---|---|
| `Stream` canonical (`constants.py:113`) | `(H2, O2, N2, CO2, CH4, H2O)` | 6 |
| `LUTConfig.fluids` (`lut_manager.py:95`) | `(H2, O2, N2, CO2, CH4, H2O, CO)` | 7 |
| Interchanger JIT prep (`interchanger.py:233`) | `[H2, O2, N2, H2O, CH4, CO2, CO]` | 7, different order |

`Stream.get_composition_arrays()` returns a 6-element array in canonical order and **omits CO and H2O_liq**. `lut_mgr.stacked_H` is indexed by `LUTConfig.fluids` order (7 fluids). Passing Stream arrays directly into LUT-indexed JIT functions will produce **wrong results** due to misaligned indices.

**Required:** Build an index-mapping array once at init time:
```python
# At interchanger.initialize() time:
canonical = StandardConditions.CANONICAL_FLUID_ORDER  # 6-elem
lut_order = lut_mgr.config.fluids                     # 7-elem

# Map: for each LUT fluid, which canonical index? (-1 if absent, e.g. CO)
self._canonical_to_lut = np.array([
    canonical.index(f) if f in canonical else -1
    for f in lut_order
], dtype=np.int32)
```
The JIT function must accept this mapping and reorder/zero-fill the mass fraction array internally.

Additionally, the interchanger currently folds `H2O_liq` into `H2O` at line 241. The JIT function must replicate this: accept the liquid water fraction as a separate scalar and add it to the H2O slot.

**Global Mapping Helper (Required for P1/P6/P1.5):**
Centralize the mapping logic so every JIT mixture path uses the **same** canonical→LUT alignment.
Add `LUTManager.get_species_map()` that returns a `np.int32` array sized to `len(lut_fluids)` with:
- Canonical index for each LUT fluid
- `-1` for missing species (e.g., CO)

All JIT mixture functions should accept this map and use it to remap 6‑element canonical arrays to 7‑element LUT arrays. The remap must:
- Zero-fill missing species (`-1` index)
- Add `H2O_liq` into the `H2O` slot (LUT index for H2O)
This avoids per-component bespoke mappings and prevents silent misalignment across Interchanger/Mixer/mixture_thermodynamics.

**Mapping Test Matrix (Required for P1/P6/P1.5):**
Add unit tests that validate the canonical→LUT remap (including `H2O_liq` folding and CO handling). Minimum cases:
- Canonical `[H2=1, O2=0, N2=0, CO2=0, CH4=0, H2O=0]`, `H2O_liq=0` → LUT `[H2=1, O2=0, N2=0, CO2=0, CH4=0, H2O=0, CO=0]`
- Canonical `[H2=0.5, H2O=0.5]`, `H2O_liq=0.1` → LUT `H2O=0.6`, all others `0`, `CO=0`
- Canonical all zeros, `H2O_liq>0` → LUT `H2O=H2O_liq`, others `0`
- LUT-only species (CO) always maps to `0` via `-1` index handling

**Out-of-Bounds Safety:** `lut_mgr.lookup()` falls back to CoolProp when P/T exceed LUT bounds (`lut_manager.py:469-474`). A pure JIT path loses this safety net. The JIT function must either:
- **Clamp** P and T to grid boundaries (acceptable if transient excursions are brief and small), or
- **Return a flag** indicating out-of-bounds, allowing the Python caller to fall back to CoolProp for that step

Recommended approach: clamp within the JIT function with a boolean return flag. The Python wrapper checks the flag and logs a warning if clamping occurred.
**Policy requirement:** If `clamped_flag=True`, the caller must **explicitly fall back** to the Python/CoolProp path for that step (or use the clamped value consistently and log a warning once per N steps). Do not silently mix clamped and unclamped paths.

**Solver Strategy — Hybrid Bisection/Newton:**

Pure Newton-Raphson for T-from-H can diverge when crossing the saturation boundary (dH/dT is discontinuous due to latent heat). Use a **bracketed Newton** approach:
1. Start with bisection bounds [T_min, T_max]
2. At each iteration, compute Newton step: `ΔT = (H_calc - H_target) / Cp`
3. If the Newton step stays within the current bracket, take it (fast convergence)
4. If it would escape the bracket, fall back to a bisection step (safe convergence)
5. Typical convergence: 5–8 iterations in single-phase, 10–15 near saturation

This replaces the current 50-iteration pure bisection and handles two-phase safely.

**Action:**
1. Remove the dead `lut_mgr.get_saturation_properties()` call at line 273 (immediate, zero-risk).
2. Move imports to module level (line 229).
3. Build the species index-mapping array at init time (see above).
4. Write `@njit(cache=True)` function `solve_interchanger_outlet_jit(...)` that takes:
   - `z_h2o` (float), `P_system` (float), `T_h_in` (float), `h_h_out_target` (float)
   - `mass_fracs_7` (float64[7] array in LUT order — remapped from canonical + H2O_liq by caller)
   - `stacked_H` (3D LUT array), `P_grid`, `T_grid` (1D arrays)
   - Uses hybrid bisection/Newton internally. Returns `(T_out, beta, clamped_flag)`.
5. Replace the Python bisection block (lines 259-448) with a single call.

**Before (current hot path, simplified):**
```python
# Inside step(), called 30,240 times
from h2_plant.optimization.numba_ops import ...  # import every call

for iter_idx in range(50):
    T_mid = 0.5 * (T_min + T_max)
    sat_props = lut_mgr.get_saturation_properties(...)  # DEAD — result unused
    P_sat = 10**(8.07 - 1730.63/(233.4 + T_mid - 273.15)) * 133.322
    K_w = P_sat / P_h_in
    beta = solve_rachford_rice_single_condensable(z_h2o, K_w)
    for s, mf in comp_copy.items():          # Python dict iteration
        h_s = lut_mgr.lookup(s, 'H', P, T)  # Python dispatch per species
    ...
```

**After:**
```python
# Module-level import (once)
from h2_plant.optimization.numba_ops import solve_interchanger_outlet_jit

# At init time (once):
self._lut_mass_fracs = np.zeros(7, dtype=np.float64)
self._canonical_to_lut = build_index_map(canonical_order, lut_mgr.config.fluids)

# Inside step():
canonical_fracs, _, _, _ = self.hot_stream.get_composition_arrays()
h2o_liq = self.hot_stream.composition.get('H2O_liq', 0.0)
remap_to_lut_order(canonical_fracs, h2o_liq, self._canonical_to_lut, self._lut_mass_fracs)

T_out, beta, was_clamped = solve_interchanger_outlet_jit(
    z_h2o, P_h_in, T_h_in, h_h_out_target,
    self._lut_mass_fracs,
    lut_mgr.stacked_H, lut_mgr._pressure_grid, lut_mgr._temperature_grid
)
if was_clamped:
    logger.warning(f"Interchanger flash clamped to LUT bounds at t={t}")
```

**Expected Savings:** 25–30 s (from 36 s to ~6–8 s).
**Effort:** Medium. Requires ~80–100 lines of Numba code (solver + remapping) and refactoring the interchanger step.
**Risk:** Medium — species mapping and two-phase convergence must be validated. See Validation section.

---

### P1.5 — Wire `mixture_thermodynamics` to Existing JIT Kernels (Phase 2)

**Evidence:** `mixture_thermodynamics.py` is used as a fallback path in multiple components (compressor, chiller, dry cooler, separators). It iterates Python dicts and calls `lut_manager.lookup()` per species — the slow path even when stacked LUTs and JIT kernels are available.

**Root Cause:** The module predates the stacked LUT/JIT mixture kernels in `numba_ops.py` and never adopted them. This leaves legacy components on a Python-heavy path.

**Files:**
- `h2_plant/optimization/mixture_thermodynamics.py`
- `h2_plant/optimization/numba_ops.py`
- `h2_plant/optimization/lut_manager.py` (new mapping helper)

**Action (Adoption, Not New Kernels):**
1. Add a fast-path inside `get_mixture_enthalpy`, `get_mixture_entropy`, and `get_mixture_density`:
   - If `lut_manager.stacked_*` and grids are available **and** `(P, T)` are in-bounds (`_p_min/_p_max`, `_t_min/_t_max`), use the existing JIT kernels:
     - `get_interp_weights_jit(...)`
     - `get_mix_enthalpy_fast_jit(...)` / `get_mix_entropy_fast_jit(...)` / `get_mix_density_jit(...)`
   - Remap canonical 6‑element mass arrays to LUT 7‑element arrays using the global mapping helper (see Global Mapping).
   - Fold `H2O_liq` into `H2O` before remapping.
2. If out-of-bounds or stacked LUTs unavailable, **fall back** to the current `lut_manager.lookup()` loop (preserves CoolProp fallback behavior).
3. For entropy, compute `mole_fracs`, `M_mix`, and `sum_ylny` via `numba_ops.fast_composition_properties()` (or existing code path) before calling `get_mix_entropy_fast_jit`.

**Benefit:** Speeds the “legacy thermodynamics” path without removing the robust Python fallback. This is wiring/adoption, not a new physics implementation.
**Expected Savings:** TBD (profile-dependent; only visible if `mixture_thermodynamics` is non-trivial in the post‑Phase‑1 profile).
**Effort:** Medium.
**Risk:** Medium — bounds handling and species mapping must be correct. Requires validation.

---

### P2 — Replace ATR `interp1d` with `np.interp` or Direct Coefficients

**Evidence:** `atr_data_manager.py:69 lookup()` = **10.9 s** cumtime (211,632 calls). The underlying `scipy.interpolate.interp1d.__call__` = 10.5 s, and `_call_linear` = 6.8 s.

**Root Cause:** `interp1d` is a general-purpose SciPy interpolator with significant per-call overhead for scalar inputs (object dispatch, input validation, array creation). For `kind='linear'` on a 1D table, `np.interp()` is 5–10x faster. If the data is truly piecewise-linear with few segments, direct `a*x + b` evaluation is ~100x faster.

**Extrapolation Behavior Change:** The current code uses `interp1d(..., fill_value="extrapolate")` which linearly extrapolates beyond the table bounds. `np.interp()` clamps to the boundary values instead. The `get_oxygen_flow()` method (`atr_data_manager.py:92+`) clips the F_O2 input to valid range before calling `lookup()`, so for that code path the difference is moot. However, **any direct caller of `lookup()` that passes unclipped values will see different behavior**. Audit all `lookup()` call sites before switching, or add explicit extrapolation logic:
```python
def lookup(self, func_name: str, f_o2_kmol_h: float) -> float:
    x, y = self._raw_data[func_name]
    if f_o2_kmol_h <= x[0]:
        # Linear extrapolation below range
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return float(y[0] + slope * (f_o2_kmol_h - x[0]))
    if f_o2_kmol_h >= x[-1]:
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return float(y[-1] + slope * (f_o2_kmol_h - x[-1]))
    return float(np.interp(f_o2_kmol_h, x, y))
```

**Files:**
- `h2_plant/components/reforming/atr_data_manager.py`

**Action:**
1. In `load_data()`, store raw `x` and `y` arrays as contiguous float64:
   ```python
   self._raw_data[col] = (np.ascontiguousarray(x), np.ascontiguousarray(df[col].values))
   ```
2. **Audit all call sites** of `ATRDataManager.lookup()` to confirm inputs are clipped to valid range (or implement explicit extrapolation logic in `lookup()` as shown).
3. Replace the `lookup()` method with `np.interp` + extrapolation guard as above.
4. For maximum speed, write a `@njit` wrapper or use `np.searchsorted` + linear formula.

**Expected Savings:** ~8–9 s.
**Effort:** Low.

---

### P3 — Eliminate CoolProp/Rounding Overhead in Hot Path

**Evidence:** `CoolPropLUT.PropsSI` = **8.6 s** (881,377 calls). `_round_sig` = **5.3 s** (1,762,754 calls — 2 rounding calls per lookup).

**Root Cause:** `_round_sig` uses `math.log10` + `math.floor` + `round` — three function calls per rounding. This runs twice per `PropsSI` call (once for each input value), even on cache hits. The rounding itself costs 62% of the wrapper's total time.

**Files:**
- `h2_plant/optimization/coolprop_lut.py`
- Callers in `h2_plant/components/cooling/dry_cooler.py`, `h2_plant/components/thermal/chiller.py`

**Action (two-pronged):**

**Short term (P3a) — faster rounding with per-variable resolution:**

Replace `_round_sig` with fixed-resolution quantization, using **separate resolutions for pressure and temperature**:

```python
@staticmethod
def PropsSI(output: str, name1: str, value1: float, name2: str, value2: float, fluid: str) -> float:
    # Per-variable quantization resolutions
    if name1 == 'P':
        v1_q = round(value1 / 500.0) * 500.0     # 500 Pa bins (~0.005 bar)
    elif name1 == 'T':
        v1_q = round(value1 * 10.0) / 10.0        # 0.1 K bins
    else:
        v1_q = round(value1 / 100.0) * 100.0      # generic

    if name2 == 'P':
        v2_q = round(value2 / 500.0) * 500.0
    elif name2 == 'T':
        v2_q = round(value2 * 10.0) / 10.0
    else:
        v2_q = round(value2 / 100.0) * 100.0

    key = (output, name1, v1_q, name2, v2_q, fluid)
    if key in CoolPropLUT._cache:
        return CoolPropLUT._cache[key]
    ...
```

A single global resolution (e.g., `100.0`) would either destroy cache hit rate for large pressures (1e7/100 = 100K distinct bins) or damage accuracy for small values (5000 Pa rounded to 5000 Pa — no binning benefit). Per-variable resolutions balance hit rate and accuracy.

**Medium term (P3b, Phase 2) — route through LUT system:**

The primary callers are dry cooler and chiller requesting saturation and latent-heat properties for water. Precompute a small 1D `h_vap(T)` lookup table at init (e.g., 200 points from 273–473 K) using CoolProp once, then interpolate with `np.interp` in the hot path. This eliminates CoolPropLUT calls entirely for these components.

**Memory note:** `CoolPropLUT._cache` is an unbounded class-level dict that grows monotonically. For annual (8,760-hour) runs the cache can accumulate tens of thousands of entries. Consider either:
- **Commit to an eviction policy** (recommended): `functools.lru_cache(maxsize=10_000)` or a capped dict.
- Or **enforce periodic clear** (e.g., every 1000 simulated hours) with a config flag.

**Expected Savings:** P3a: ~4 s. P3b: ~7–8 s total.
**Effort:** Low (P3a), Medium (P3b).

---

### P4 — Eliminate `hasattr`/`getattr` Reflection in `record_post_step()`

**Evidence:** `engine_dispatch.py:1125 record_post_step()` = **34.8 s** cumtime, **9.8 s** tottime (10,080 calls).

**Root Cause:** The function uses `hasattr()` and `getattr()` extensively to probe component attributes (lines 1149-1154, 1195-1198). `hasattr` internally calls `getattr` and catches `AttributeError`, making it expensive in CPython. This runs 10,080 times with ~20 probes per call = ~200,000 attribute lookups via exception handling.

**Files:**
- `h2_plant/control/engine_dispatch.py`

**Action:**
1. During `initialize()` or strategy setup, resolve all attribute paths once and store as direct references or flags:
   ```python
   # At init time:
   self._soec_has_real_powers = hasattr(self._soec, 'real_powers')
   # Resolve the h2 getter chain once:
   if hasattr(self._soec, 'last_step_h2_kg'):
       self._get_soec_h2 = lambda: self._soec.last_step_h2_kg
   elif hasattr(self._soec, 'last_h2_output_kg'):
       self._get_soec_h2 = lambda: self._soec.last_h2_output_kg
   else:
       self._get_soec_h2 = lambda: self._soec.h2_output_kg
   ```
2. Replace the `if not hasattr(self, '_compressors')` pattern (line 1195) with initialization in the setup method — move the list comprehension to `initialize()`.
3. Replace `getattr(obj, attr, default)` with direct attribute access where the attribute is guaranteed to exist after init.
4. **Chunked history safety:** any cached array references must be re‑bound on chunk rollover (`_rebind_recorders()`). Avoid storing raw array slices outside the rebind mechanism.

**Chunk-Rebinding Constraint:** `record_post_step()` uses `StreamRecorder` objects with pre-bound array references that point into the current history chunk (`engine_dispatch.py:1027-1034`). When chunked history mode flushes a full chunk and allocates fresh arrays (`history_manager.py:allocate_chunk()`), `_rebind_recorders()` is called at `local_idx == 0` to update all pointers. **Any pre-bound references introduced by P4 must also be re-bound per chunk.** If P4 caches attribute references (e.g., `self._get_soec_h2 = lambda: ...`), these lambdas capture object references (not array pointers) and are safe. However, if any optimization caches direct array slices or numpy views from `history_store`, those references will become stale after chunk flush and must be included in `_rebind_recorders()`.

**Expected Savings:** ~3–5 s of the 9.8 s tottime.
**Effort:** Low.

---

### P5 — Fuse Dry Cooler Numeric Kernel into Single JIT Function

**Evidence:** `dry_cooler.py:255 step()` = **17.6 s** cumtime, **10.5 s** tottime (231,840 calls = 23 instances × 10,080 steps).

**Root Cause:** Each step makes ~6 individual JIT calls from Python (Reynolds, Nusselt, ε-NTU for TQC, then again for DC, plus condensation). The per-call Python→JIT overhead is ~1-2 μs, but at 231,840 × 6 = 1.4M crossings, it accumulates. The 10.5 s tottime indicates significant Python-level orchestration between JIT calls.

**Scope Clarification:** A full JIT replacement of `step()` is **not feasible** without refactoring away Python objects (inlet Stream, composition dicts, optional CoolPropLUT for latent heat, `self.*` attributes). The realistic scope is a **fused numeric kernel** with Python pre/post handling:

- **Python (pre):** Extract scalars from Stream, resolve composition to array, read `self.*` state
- **JIT kernel:** Reynolds → Nusselt → U-value → ε-NTU (TQC) → Q_tqc → T_gas_mid → Reynolds → Nusselt → ε-NTU (DC) → Q_dc → T_glycol_new → condensation check
- **Python (post):** Write results back to `self.*`, construct output Stream

**Files:**
- `h2_plant/optimization/numba_ops.py` — new fused function
- `h2_plant/components/cooling/dry_cooler.py` — refactor `step()`

**Action:**
1. Create `@njit` function `solve_dry_cooler_thermal_jit(...)` that takes all scalar inlet conditions and geometry parameters, returns `(T_gas_out, T_glycol_hot, T_glycol_cold_new, Q_tqc_w, Q_dc_w, condensate_kg_s)`.
2. Internally fuse the two ε-NTU solvers and condensation check.
3. In `step()`, extract scalars → call JIT kernel → write results back.

**Dependency on P3b:** If dry cooler latent heat lookups are moved from CoolPropLUT to a precomputed 1D LUT (P3b), the JIT kernel can accept the LUT arrays directly, eliminating the last Python callback in the numeric path.

**Expected Savings:** ~5–7 s (reduced from original 8–10 s estimate after scoping down).
**Effort:** Medium.

---

### P6 — Fuse Mixer PH-Flash into JIT

**Evidence:** `multicomponent_mixer.py:202 step()` = **12.2 s** (50,400 calls). `_perform_ph_flash()` = **8.2 s**.

**Root Cause:** Same pattern as interchanger — Python-level bisection/iteration with per-iteration LUT lookups through Python dispatch.

**Files:**
- `h2_plant/optimization/numba_ops.py` — new function
- `h2_plant/components/mixing/multicomponent_mixer.py` — refactor

**Action:** Same approach as P1 — write a `@njit` PH-flash solver that takes stacked LUT arrays and mass fractions, returns (T_out, phase, composition_split). The same species-order mapping and out-of-bounds handling concerns from P1 apply here. Consider sharing the same JIT flash kernel between P1 and P6 if the physics is compatible.

**Solver:** Use the same hybrid bisection/Newton approach as P1.

**Expected Savings:** ~5–6 s.
**Effort:** Medium.

---

### P7 — Reduce `numpy.interp` Scalar Call Overhead

**Evidence:** `numpy.interp` = **7.3 s** (2,169,101 calls). Per-call: ~3.4 μs.

**Root Cause:** `np.interp` for scalar inputs has Python→C overhead (argument parsing, array creation for scalar, dispatching). Used in saturation property lookups and various 1D interpolations in the hot path.

**Important — Re-scope After Phase 1:** A significant fraction of the 2.17M `np.interp` calls likely originate from code paths that P1 (interchanger) and P2 (ATR) will subsume. After implementing Phase 1 and re-profiling, the residual `np.interp` call count may be substantially lower. **Do not invest in P7 until post-Phase 1 profiling confirms the residual cost justifies it.**

**Files:** Multiple callers — identify with `grep -rn "np.interp" h2_plant/`

**Action (after re-profile confirms need):**
1. For callers inside `@njit` functions: `np.interp` is already supported by Numba — no change needed.
2. For callers in Python-level hot paths: either move the calling code into `@njit`, or replace with a pre-computed coefficient array + direct `a*x + b` evaluation using `np.searchsorted`.

**Expected Savings:** ~2–4 s (reduced from original 4–5 s after accounting for P1/P2 overlap).
**Effort:** Medium (requires identifying and refactoring multiple call sites).

---

### P8 — Fast-Path Stream Creation (`from_arrays`)

**Evidence:** `Stream.__post_init__` = **4.4 s** (2,053,803 calls). `_cache_composition_arrays` = **11.1 s** (1,183,897 calls). `get_composition_arrays` = **11.7 s**.

**Root Cause:** ~2M Stream objects are created per 168-hour simulation. Each construction:
- Normalizes a dict (Python loop + sum)
- Invalidates cached arrays (new object = uncached)
- Triggers array rebuild downstream (dict → numpy) in hot paths

**Files:**
- `h2_plant/core/stream.py`

**Action (Boundary-Focused Fast Path):**
1. Add a classmethod `Stream.from_arrays(...)` that **bypasses `__post_init__`** and pre-populates cached arrays:
   - Accept canonical mass fractions (6‑element), `T`, `P`, `mass_flow`, optional `h2o_liq_frac`, `phase`, `extra`.
   - Use `numba_ops.fast_composition_properties()` to compute `mole_fracs`, `M_mix`, `sum_ylny`.
   - Set `_mass_fracs_arr`, `_mole_fracs_arr`, `_cached_M_mix`, `_cached_sum_ylny`, `_arrays_cached=True`.
   - Build a **minimal** `composition` dict (non‑zero species + `H2O_liq`) for downstream compatibility.
2. Use this fast path **only at component boundaries** where composition is already normalized and stable (e.g., output streams created after a JIT flash). Avoid in solver loops (already mostly avoided).
3. Add an explicit mutation guard for `from_arrays()` streams (e.g., `_frozen_composition=True` + reject `composition` mutation, or auto‑invalidate cached arrays on change). This must be enforced in code, not just documented.
4. Optional: add `_normalize=False` for conventional constructors to skip dict normalization when callers guarantee normalization.

**Mutation Safety Warning:** Many components read/modify `stream.composition`. If cached arrays and dict diverge, plots and physics will silently drift. **Choose and enforce** one of:
- **(a) Immutability contract** for streams created via `from_arrays()` (no dict mutation).
- **(b) Explicit cache invalidation** when composition changes.
- **(c) Copy‑on‑construct:** use `from_arrays()` only for fresh output streams that are not mutated later.  

Option (c) is the safest incremental approach, but still requires an explicit guard to prevent silent mutation.

**Expected Savings:** ~5–8 s across `__post_init__` + `_cache_composition_arrays`.
**Effort:** Medium. Requires auditing Stream construction sites for safe fast‑path usage.
**Risk:** Medium — stale cached arrays can corrupt composition‑dependent results.

---

### P9 — Optimize Flow Network Dispatch

**Evidence:** `flow_network.py:177 _execute_single_flow()` = **24.3 s** cumtime, **7.9 s** tottime (1,602,720 calls).

**Root Cause:** Each call performs:
- Dict lookups on `self._source_cache[conn.source_id]` and `self._target_cache[conn.target_id]`
- `isinstance()` type checks on output (Stream vs float vs None)
- String comparison `conn.resource_type == 'signal'` on every call
- `flow_tracker.record_flow()` with `resource_type.lower()` and dict `.get()` per flow

**Files:**
- `h2_plant/simulation/flow_network.py`

**Action:**
1. At init, split `self._connections` into typed lists: `_stream_connections`, `_signal_connections`, `_energy_connections`. The `execute_flows()` method iterates only `_stream_connections`; `execute_signals()` iterates only `_signal_connections`. Eliminates the per-call `resource_type == 'signal'` check.
2. Pre-bind connections as tuples of direct references:
   ```python
   # At init:
   self._resolved_flows = [
       (source_obj, source_obj.get_output, conn.source_port,
        target_obj, target_obj.receive_input, conn.target_port,
        source_obj.extract_output, flow_type_enum)
       for conn in self._stream_connections
   ]
   ```
3. Pre-map `resource_type` → `flow_type` string at init instead of calling `.lower()` and `.get()` per flow.
4. Add a guard that the topology is immutable after init (or rebuild `_resolved_flows` if connections change). If dynamic topologies are supported, skip pre-binding and fall back to the safe path.

**Expected Savings:** ~2–3 s of the 7.9 s tottime.
**Effort:** Low.

---

### P9.1 — FlowTracker Flush/Serialization Overhead

**Evidence:** `flow_tracker.flush` = **10.9 s** (7 calls), `record_flow` = **10.5 s** (1.35M calls). `json.dumps`/encoder = **~11 s** across 1.15M records. The `flows.jsonl` file reached ~280 MB for a 160-hour run after the FlowTracker bug fix.

**Root Cause:**
- `Flow.to_dict()` used `dataclasses.asdict()` (deep-copies every record)
- `record_flow()` formats `"src -> dest"` and `"type (unit)"` strings on every call
- `flush()` serializes per record and appends to a growing file

**Files:**
- `h2_plant/simulation/flow_tracker.py`
- `h2_plant/simulation/engine.py`
- `benchmark_suite.py`

**Action:**
1. Replace `dataclasses.asdict()` with a manual dict (no deep copy).
2. Cache matrix key strings (or use tuple keys) to avoid per-call formatting.
3. Batch-serialize buffer and write once per flush (JSON Lines).
4. Add **rolling flush** mode that overwrites the file each interval (last-interval only) and optionally resets aggregates to keep memory bounded.
5. If full history is needed, use per-chunk files or Parquet instead of a single ever-growing JSON file.
6. Optional (P13 folded): add `record_flow_enum()` (accepts `FlowType` directly) to skip enum lookup in hot paths.
7. Optional (P13 folded): convert `Flow` to `@dataclass(slots=True)` or bypass the dataclass entirely (write dicts directly). **Compatibility risk:** slots break dynamic attribute assignment—verify no consumers rely on it.

**Expected Savings:** ~8–12 s.
**Effort:** Low.
**Risk:** Low (format/compatibility — document JSONL + rolling behavior).

---

### P11 — Saturation Properties Caching (P‑sat, h_f, h_g)

**Evidence:** `lut_manager.get_saturation_properties()` = **10.8 s** cumtime in the profile. It is called by `electric_boiler.py`, `attemperator.py`, and (today) interchanger. The interchanger call is dead work and should be removed in P1; re‑profile after that change to confirm residual cost.

**Root Cause:** The function performs repeated Python‑level work (dict access, bounds checks, multiple `np.interp`) even when pressure is constant over a step. The math itself is cheap; the overhead dominates.

**Files:**
- `h2_plant/optimization/lut_manager.py`
- `h2_plant/components/thermal/electric_boiler.py`
- `h2_plant/components/thermal/attemperator.py`

**Action (low‑risk first pass):**
1. Add **component‑level memoization**: cache the last `(pressure_pa → sat_props)` per component and reuse when pressure is unchanged (or within a very small tolerance like 1–10 Pa).
2. Preserve **out‑of‑bounds behavior**: if pressure is outside LUT bounds, bypass cache and use the existing CoolProp fallback (do not clamp silently).
3. Optional: add a small **global cache** in `LUTManager` keyed by **exact** pressure (no quantization). If pressure quantization is introduced to boost reuse, treat it as a **Phase 2–style change** and validate accuracy.

**Expected Savings:** ~5–9 s (upper bound; depends on how much interchanger cleanup removes).
**Effort:** Low.
**Risk:** Low if exact‑pressure caching only; Medium if quantization is used (requires validation).

---

### P12 — LUT Lookup Fast Path (Pre‑binding + Direct Interp)

**Evidence:** `lut_manager.lookup()` = **20.5 s** cumtime and `_interpolate_2d` = **12.5 s**. `dict.get` and dispatch overhead are hot, indicating Python‑level selection and bounds checks are significant.

**Root Cause:** `lookup()` performs repeated fluid/property dict navigation and validation per call even when the caller always queries the same fluid/property. This overhead is amplified in tight loops.

**Files:**
- `h2_plant/optimization/lut_manager.py`
- Hot callers in `h2_plant/components/*` (compressors, valves, pumps)

**Action:**
1. Add a **pre‑binding helper** (e.g., `lut_mgr.bind_lookup(fluid, property)`), returning a lightweight callable or struct with:
   - `lut` array pointer
   - `pressure_grid`, `temperature_grid`
   - cached bounds (`_p_min/_p_max`, `_t_min/_t_max`)
2. Add `lookup_fast(pressure, temperature)` that:
   - checks bounds (same logic as `lookup`)
   - computes weights via `get_interp_weights_jit`
   - returns `interp_from_weights_jit(lut, ...)`
   - if out‑of‑bounds, falls back to the existing `lookup()` (CoolProp path preserved)
3. Use the bound helper in components with fixed fluid/property to avoid dict dispatch in inner loops.

**Expected Savings:** ~4–6 s (even 20–30% of the lookup cost is meaningful).
**Effort:** Low–Medium.
**Risk:** Low if bounds/fallback are preserved; ensure dtypes/contiguity match warm‑up signatures (P0).

---

### P10 — Downcast LUT Arrays from float64 to float32

**Evidence:** Memory footprint is 1.5+ GB. LUT grids are 2000×2000 × 5 properties × 7 fluids × 8 bytes ≈ 1.1 GB.

**Root Cause:** Thermodynamic properties are stored as float64 but engineering calculations require at most 6 significant digits. float32 provides ~7 significant digits.

**Files:**
- `h2_plant/optimization/lut_manager.py` — LUT generation and loading
- `h2_plant/optimization/numba_ops.py` — JIT interpolation functions (type signatures)

**Action:**
1. In `_generate_lut()` and/or the stacked cache save path, cast arrays to float32:
   ```python
   lut = lut.astype(np.float32)
   ```
2. Update JIT function signatures to accept float32 arrays. Numba recompiles per type signature — if `cache=True` is set, the new float32 specialization will be cached on first call. **Ensure P0 warm-up uses the new float32 arrays** to avoid runtime compilation.
3. Keep pressure/temperature grids as float64 (they're 1D, negligible memory).

**Caution:** Verify that float32 precision is sufficient for the enthalpy convergence criterion (100 J/kg tolerance with enthalpy values up to ~5 MJ/kg = relative precision of 2e-5, well within float32's ~1e-7). **Do NOT use float16** — it has only ~3 decimal digits of precision, which is insufficient.

**Expected Savings:** ~550 MB memory reduction.
**Effort:** Low, but requires validation testing.

---

## Summary Table

| ID | Phase | Target | Est. Savings | Effort | Risk |
|---|---|---|---|---|---|
| P0 | 1 | JIT pre-warming | 18 s shifted to init | Low | None |
| P2 | 1 | ATR `interp1d` → `np.interp` | 8–9 s | Low | Low (extrapolation) |
| P3a | 1 | CoolProp rounding fix | ~4 s | Low | None |
| P4 | 1 | `record_post_step` reflection | 3–5 s | Low | None |
| P9 | 1 | Flow network dispatch | 2–3 s | Low | None |
| P9.1 | 1 | FlowTracker flush/serialization | 8–12 s | Low | Low |
| P11 | 1 | Saturation properties caching | 5–9 s | Low | Low–Med (if quantized) |
| P12 | 1 | LUT lookup fast path | 4–6 s | Low–Med | Low |
| | | **Phase 1 subtotal** | **~34–48 s + jitter fix** | | |
| | | **--- RE-PROFILE GATE ---** | | | |
| P1 | 2 | Interchanger JIT flash | 25–30 s | Medium | Medium |
| P1.5 | 2 | Wire `mixture_thermodynamics` to JIT kernels | TBD (profile‑dependent) | Medium | Medium |
| P5 | 2 | Dry cooler fused JIT kernel | 5–7 s | Medium | Low |
| P6 | 2 | Mixer PH-flash JIT | 5–6 s | Medium | Medium |
| P7 | 2 | `numpy.interp` (re-scoped) | 2–4 s | Medium | Low |
| P8 | 2 | Stream creation overhead (fast-path `from_arrays`) | 5–8 s | Medium | Medium |
| P10 | 2 | LUT float64 → float32 | 550 MB memory | Low | Low |
| P3b | 2 | CoolProp → LUT routing | ~3–4 s | Medium | Low |
| | | **Phase 2 subtotal** | **~45–59 s + 550 MB** | | |

**Combined estimated production savings: ~79–107 s (from ~150–165 s to ~43–86 s) + 550 MB memory.**  
Note: P1.5 is not included in the subtotal; its impact is profile‑dependent and should be measured post‑Phase‑1. Re‑baseline after the benchmark harness fix; treat savings as upper bounds until re‑profiled.
Note: savings are not perfectly additive since some functions are in callee chains of others. The re-profile gate between phases ensures estimates stay grounded.

---

## Benchmark Harness Fix (Prerequisite)

Before measuring the impact of any optimization, update `benchmark_suite.py` to match the production execution path:

1. Replace `engine.registry.step_all(hour)` (line 167) with iteration over the engine's pre-resolved execution list (the same list `engine.run()` uses at `engine.py:290`).
2. Gate `engine.monitoring.collect()` (line 180) with `if current_step % steps_per_hour == 0:` to match `engine.py:307,319`.
3. Add FlowTracker flush calls (e.g., every 24 simulated hours) to match `engine.py:322-323`.
4. Add JIT warm-up calls between initialization and the timing loop start.

This ensures profile results reflect production performance and avoids wasting optimization effort on benchmark-only artifacts.

---

## Hypotheses Evaluated (from Literature Review)

| Hypothesis | Verdict | Rationale |
|---|---|---|
| **A: GPU / CUDA** | **Invalid** | No data-parallel workload. LUT lookups are scalar, simulation is sequential. Host-device transfer would make it slower. |
| **B: Solver replacement** | **Partially valid** | Rachford-Rice is already analytical (optimal). The bisection wrapper is the bottleneck — addressed by P1/P6. Hybrid bisection/Newton for T-from-H convergence is valid within those JIT functions, but pure Newton is unsafe near saturation. |
| **C: Advanced Numba** | **Partially valid** | `nopython=True` already used. Loop fusion (P1, P5, P6) is the key win. float32 LUTs valid (P10), float16 invalid. No evidence of cache-miss issues in the memory profile. |
| **D: Parallelism** | **Invalid** | 21 ms step time with Python GIL makes thread overhead prohibitive. Components have topological data dependencies. Shared mutable state in registry prevents safe concurrency. |

---

## Validation Requirements

**All Phase 2 items (P1, P1.5, P5, P6, P8, P10, P3b) change physics or data representations and require validation before merge.**

### Phase 2 Mathematical Validity & Code Implications (2026-02-02 Review)

**High‑risk correctness invariants (must hold before merging P1/P6/P1.5):**
1. **Species order consistency:** All JIT mixture paths must use the same canonical→LUT mapping. Interchanger currently builds LUT‑ordered arrays for `lookup_mixture_enthalpy`, then uses a different order for flash (`['H2','O2','N2','H2O','CH4','CO2','CO']`). This will produce wrong results unless the global mapping helper is enforced everywhere.  
   Files: `h2_plant/components/thermal/interchanger.py`, `h2_plant/optimization/numba_ops.py`.
2. **Basis consistency (mass vs molar):** Mixer PH flash currently targets **molar** enthalpy (J/mol). LUT JIT kernels operate in **mass** space (J/kg). Any P6 JIT solver must explicitly convert or the target enthalpy will be wrong.  
   Files: `h2_plant/components/mixing/multicomponent_mixer.py`, `h2_plant/optimization/numba_ops.py`.
3. **Two‑phase enthalpy handling:** The current interchanger model computes two‑phase enthalpy by splitting vapor/liquid and mixing with mass fractions (incl. partial‑pressure vapor enthalpy). A naive “mixture enthalpy from LUT” ignores latent heat unless explicitly modeled. P1 must reproduce the existing phase split or document/validate a physics change.  
   Files: `h2_plant/components/thermal/interchanger.py`.
4. **Out‑of‑bounds behavior:** `lut_manager.lookup()` falls back to CoolProp; JIT kernels clamp to bounds. Any new fast path must preserve OOB semantics (flag + fallback), not silently clamp.  
   Files: `h2_plant/optimization/lut_manager.py`, `h2_plant/optimization/numba_ops.py`.

**Medium‑risk implications:**
5. **P6 phase equilibrium:** Mixer PH flash is currently single‑phase; adding two‑phase JIT logic changes physics unless explicitly gated by `enable_phase_equilibrium`.  
6. **P1.5 H2O_liq handling:** `mixture_thermodynamics` currently skips `H2O_liq`. Folding liquid into vapor for a fast path changes results; if equivalence is required, preserve the skip or validate the change.  
7. **P8 cache coherence:** `Stream.from_arrays()` bypasses normalization and caches arrays; if `composition` is mutated later, cached arrays become stale. Enforce immutability or explicit cache invalidation.  
8. **P10 precision:** float32 LUTs can destabilize bisection/Newton near low‑enthalpy or saturation boundaries; validate with edge‑case tests before accepting.

**Bottleneck coverage check (current profile):**
- **P1** targets `interchanger.step` (~34.8 s cumtime). Largest expected gain if JIT flash is correct.
- **P6** targets `multicomponent_mixer.step` (~11.9 s) / `_perform_ph_flash` (~8.0 s).
- **P5** targets `dry_cooler.step` (~13.6 s), but only the numeric core is safe to fuse (condensation + clamps must stay consistent).
- **P8** targets `get_composition_arrays` (~10.4 s) and `_cache_composition_arrays` (~9.7 s) only if fast‑path adoption is broad.
- **P3b** targets `CoolPropLUT.PropsSI` (~7.5 s).
- **P7** targets `numpy.interp` (~7.4 s) but should be re‑scoped after Phase 1+P1/P2/P11.

### Regression Test Protocol

1. **Baseline capture:** Run a 168-hour simulation on the current (unoptimized) code. Record per-step outputs for key components:
   - Interchanger: `T_hot_out`, `T_cold_out`, `q_transferred_kw`, `outlet_H2O_molf`
   - Mixer: `T_out`, `phase`, per-species mass fractions
   - Dry cooler: `T_gas_out`, `Q_tqc`, `condensate_kg_s`
   - Compressor: `T_out`, `P_out`, `W_actual`
   - Storage: `total_mass_kg`, `pressure_Pa` per tank

2. **Post-optimization diff:** Run the same 168-hour scenario with optimized code. Compare outputs:
   - **Temperature:** absolute tolerance ≤ 0.1 K
   - **Mass fractions:** absolute tolerance ≤ 1e-6
   - **Energy (Q, W):** relative tolerance ≤ 0.1%
   - **Mass balance:** total plant mass in = mass out ± 1e-9 kg/step

3. **Edge-case tests for P1/P6:**
   - Flash near dew point (z_H2O ≈ K_value): verify no NaN or divergence
   - Flash at very low water content (z_H2O < 1e-6): verify β = 1.0
   - Out-of-bounds P/T: verify clamping flag is raised and results are physically reasonable
   - Two-phase crossing during cooling: verify smooth transition through saturation

4. **P10 precision test:** Compare float64 vs float32 LUT interpolation for 10,000 random (P,T) points across the grid. Verify max absolute enthalpy error < 50 J/kg (half the convergence tolerance).
5. **Species mapping test matrix (P1/P6/P1.5):** Validate canonical→LUT remap for `H2O_liq` folding and `CO=-1` handling (see Mapping Test Matrix above).
6. **P11/P12 correctness check:** For a random sample of pressures/temperatures, compare `get_saturation_properties()` and `lookup()` results against the fast‑path versions. Out‑of‑bounds queries must follow the same CoolProp fallback path; no silent clamping.
7. **Scripted equivalence checks (NEW, required for “physics‑neutral” changes):**
   - Implement a small validation script (e.g., `scripts/validate_equivalence.py`) that runs the **same scenario twice**: once on the *baseline* (pre‑change) and once on the *optimized* code.
   - The script must load the two result sets and compute **relative error** for matched fields. Use a strict threshold:
     - **diff < 0.01%** (relative) for scalar thermodynamic outputs (T, P, H, S, density, Cp).
     - **diff < 0.01%** for mass/energy totals and component KPIs (Q, W, production rates).
   - Where the value can be zero or near‑zero, use a **combined tolerance**:
     - `abs(diff) < 1e-6` **or** `rel(diff) < 1e-4` (0.01%) to avoid division blow‑ups.
   - The script must print a summary table of **max** and **p99** relative errors per metric and fail (non‑zero exit code) if any metric exceeds 0.01%.

8. **CoolProp vs LUT equivalence (NEW, required for LUT fast paths and P3b/P10):**
   - Add a standalone script (e.g., `scripts/validate_coolprop_vs_lut.py`) that samples `(P, T)` points across the LUT domain and compares:
     - CoolProp `PropsSI` (truth) vs LUT lookup (baseline path)
     - CoolProp `PropsSI` vs optimized fast‑path (JIT / cached)
   - Use at least **10,000 random points** + **edge cases** (min/max P/T, saturation boundary, near‑dewpoint, low‑enthalpy region).
   - Acceptance: **diff < 0.01%** for all properties, plus the existing **absolute** threshold (e.g., < 50 J/kg for enthalpy).
   - For any out‑of‑bounds sample, ensure the LUT path correctly falls back to CoolProp and the reported diff is exactly **0** (same path).

9. **Result file diffing (NEW):**
   - For each benchmark run, write a compact JSON/CSV “signature” file with key metrics at hourly resolution (or every N steps).
   - Provide a `diff_results.py` helper that compares two signatures and enforces **diff < 0.01%** globally.
   - Store signatures under `benchmark_output/baselines/<date>/` to enable regression tracking.

### Test Harness + CI Entry Point (NEW)
- Add a regression harness script (e.g., `scripts/run_regression_suite.py`) that runs baseline + optimized scenarios, emits a diff report (CSV/JSON), and exits non‑zero on tolerance failures.
- Add a CI entrypoint (e.g., `scripts/ci/run_regression.sh`) that runs a short 24‑hour scenario and archives artifacts; keep the full 168‑hour run manual or nightly.
- Store baselines under a dated folder (e.g., `benchmark_output/baselines/2026-02-01/`) and tag them by code version.

### Chunked History Smoke Test
- Run a **2-chunk simulation** (e.g., `2 × chunk_size` steps). Verify:
  - History length equals total steps (no truncation at chunk boundary).
  - All recorded columns have non-zero updates in **both** chunks (confirms `_rebind_recorders()` works after flush).
  - Parquet files on disk contain the expected number of rows per chunk.

### Memory Validation
- Run an 8,760-hour simulation. Verify `CoolPropLUT._cache` does not exceed a set memory cap.
- Verify FlowTracker memory is bounded (periodic flush works).
- Verify `flows.jsonl` size remains bounded when rolling flush is enabled (last-interval only).
- Verify post-P10 peak RSS is reduced by ≥400 MB.

---

## Pre-Existing Bug: Silent Flow Record Dropping (FlowTracker)

**Discovered during:** QA review of optimization plan (not caused by any optimization).

**Severity:** High — Sankey diagrams, flow matrix graphs, and all FlowTracker-derived analytics are missing the majority of flow data.
**Status:** Fixed in code on 2026-02-01 (FlowType.STREAM_MASS + resource map). Keep this section as a guardrail for P9.

### Root Cause

Two files interact to silently discard most flow records:

1. **`flow_network.py:90-98`** defines `_resource_map` with 7 entries (`hydrogen`, `water`, `oxygen`, `electricity`, `heat`, `natural_gas`, `work`). The fallback at **line 237** generates `f"{conn.resource_type.upper()}_MASS"` for any unmapped type.

2. **`flow_tracker.py:35-61`** defines `FlowType` as an `IntEnum` with 10 values. `"STREAM_MASS"` is **not** among them.

3. **`flow_tracker.py:159-162`** — `record_flow()` attempts `FlowType[flow_type.upper()]`, catches `KeyError`, and **silently returns** without recording.

4. **`plant_topology.yaml`** uses `resource_type: "stream"` for **~100 of ~122 connections** — the vast majority of the plant's flows.

**Result:** `"stream"` → fallback → `"STREAM_MASS"` → `KeyError` → silently dropped. Sankey/matrix graphs are near-empty in production today.

### Files

- `h2_plant/simulation/flow_network.py` (lines 90-98, 237)
- `h2_plant/simulation/flow_tracker.py` (lines 35-61, 159-162)
- `scenarios/plant_topology.yaml`, `scenarios/plant_topology_without_ATR.yaml`

### Fix Options

**Option A — Minimal, Safe:**

Add a generic enum entry and explicit mapping:

1. Append `STREAM_MASS = 10` to `FlowType` in `flow_tracker.py` (append to preserve existing numeric values).
2. Add `'stream': 'STREAM_MASS'` to `_resource_map` in `flow_network.py`.

Result: Sankey/matrix graphs immediately start showing flows as a mixed "Stream Mass" category. No topology file changes needed.

**Option B — Better Semantics, Still Local:**

Keep Option A, but add composition-based inference when `conn.resource_type == "stream"`:

```python
# In flow_network.py, when resource_type == "stream" and output is a Stream:
def _infer_flow_type(self, stream: Stream) -> str:
    comp = stream.composition
    dominant = max(comp, key=comp.get) if comp else None
    return {
        'H2': 'HYDROGEN_MASS',
        'O2': 'OXYGEN_MASS',
        'H2O': 'WATER_MASS',
        'H2O_liq': 'WATER_MASS',
        'CH4': 'NATURAL_GAS_MASS',
        'CO2': 'CO2_EMISSIONS',
    }.get(dominant, 'STREAM_MASS')
```

Result: Sankey splits by substance where dominant species is clear; mixed streams fall back to `STREAM_MASS`.


### Recommended Approach

**Implement Option A immediately** (2-line fix, zero risk). Then implement Option B as a follow-up — Option B is lower effort.

### Verification Steps

1. Run a short simulation (e.g., 1 hour), confirm `flows.jsonl` is non-empty.
2. Confirm `flow_tracker.get_sankey_data()` returns nodes and links.
3. Verify units are correct (`kg` for mass flows, `kWh` for energy flows).
4. Compare Sankey node/link counts before and after fix.

### Interaction with P9

P9 (flow network dispatch optimization) pre-maps `resource_type` → `flow_type` at init time. This bug **must be fixed before or alongside P9**, otherwise P9 will bake the broken mapping into its pre-resolved tuples. If Option B is chosen, the composition-based inference runs at flow time (not init time), so P9's pre-binding would need to store the raw `resource_type` and defer inference to execution.
