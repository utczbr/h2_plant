# OPTIMIZATION IMPLEMENTATION CHECKLIST
## Copy-Paste Deployment Guide for Performance Engineering

**Date:** November 25, 2025  
**Estimated Effort:** 28 hours across 3 weeks  
**Expected Outcome:** 6-24× speedup (24h → 1h-4h runtime)  

---

## PRE-IMPLEMENTATION CHECKLIST

- [ ] Read `optimization_strategy.md` (understand all 4 bottlenecks)
- [ ] Review `optimization_implementation.md` (understand code templates)
- [ ] Ensure Python 3.10+ with NumPy 1.24+, Numba 0.57+
- [ ] Verify Numba installed: `python -c "from numba import njit; print('✓')"`
- [ ] Create feature branch: `git checkout -b feat/performance-optimization`
- [ ] Set up profiling tools: `pip install py-spy memory-profiler`
- [ ] Baseline performance: Run full year, record time/memory

---

## WEEK 1: LUT + NUMBA FUNDAMENTALS (16 hours)

### Day 1-2: LUT Manager Optimization (6 hours)

#### Step 1.1: Copy LUTManagerOptimized class (1.5 hours)

- [ ] Create `h2plant/core/lut_manager_optimized.py`
- [ ] Copy entire `LUTManagerOptimized` class from `optimization_implementation.md`
- [ ] Verify no syntax errors: `python -m py_compile lut_manager_optimized.py`
- [ ] Add to imports in `__init__.py`

**Expected result:** New class with cache + quantization support

#### Step 1.2: Warm cache with common operating points (1 hour)

- [ ] Modify `_warm_cache()` method with PEM/SOEC typical conditions
- [ ] Test: `python -c "from lut_manager_optimized import LUTManagerOptimized; lut = LUTManagerOptimized(); lut._warm_cache(); print('✓')"`
- [ ] Verify cache size: `len(lut._lookup_cache)` should be ~100-200 entries

**Expected result:** Cache pre-populated with frequent queries

#### Step 1.3: Integrate into simulation manager (1.5 hours)

- [ ] In `manager.py`, replace: `self.lut = LUTManager()` → `self.lut = LUTManagerOptimized()`
- [ ] Test 8-hour scenario: Should run <1 minute
- [ ] Record cache stats: `lut.get_cache_stats()`

**Expected result:** 8-hour scenario <1 minute, cache hit rate >70%

**Performance checkpoint:** Cache should show ~80% hit rate

---

### Day 3-4: Numba Operations (4 hours)

#### Step 2.1: Create enhanced numba_ops module (2 hours)

- [ ] Create `h2plant/core/numba_ops_enhanced.py`
- [ ] Copy all Numba-compiled functions from `optimization_implementation.md`:
  - `thermal_step_compiled`
  - `batch_thermal_updates`
  - `pump_flow_step_compiled`
  - `pressure_step_compiled`
  - `batch_pressure_updates`
  - `batch_lookup_vectorized`
  - `calculate_efficiency_compiled`
  - `batch_efficiency_updates`

- [ ] Verify compilation: 
  ```python
  from numba_ops_enhanced import thermal_step_compiled
  T = thermal_step_compiled(333.15, 60, 700e3, 50e3, 100, 2.6e6)
  print(f"✓ Numba compiled, result T={T:.2f} K")
  ```

**Expected result:** All Numba functions compile without errors

#### Step 2.2: Replace component thermal calls (1.5 hours)

- [ ] In `pem_electrolyzer_detailed.py`, replace:
  ```python
  # OLD:
  T_new = self.thermal_model.step(60, Q_in, Q_out, Q_loss, 333.15)
  
  # NEW:
  from numba_ops_enhanced import thermal_step_compiled
  T_new = thermal_step_compiled(self.T_K, 60, Q_in, Q_out, Q_loss, 
                                self.C_thermal, 298.15, 373.15)
  ```

- [ ] Repeat for all thermal components (SOEC, steam generator, etc.)
- [ ] Verify 8-hour scenario still runs correctly
- [ ] Time individual thermal steps: should be <0.1 microseconds

**Expected result:** Thermal calculations 100× faster

**Performance checkpoint:** Profile should show 100× speedup on thermal ops

---

### Day 5: Integration & Vectorization (3 hours)

#### Step 3.1: Integrate batch LUT lookups (1.5 hours)

- [ ] Modify component properties that do multiple LUT lookups
- [ ] Use `batch_lookup_vectorized()` for arrays of (P,T) pairs
- [ ] Example: If component calculates properties for 10 pressures:
  ```python
  # OLD: 10 individual lookups
  for p in pressures:
      rho[i] = lut.lookup('H2O', 'D', p, temp)
  
  # NEW: 1 batch call
  from numba_ops_enhanced import batch_lookup_vectorized
  rho = batch_lookup_vectorized(lut_array, pressures, np.full_like(pressures, temp), ...)
  ```

**Expected result:** Batch lookups working for 50% of LUT calls

#### Step 3.2: Full integration testing (1.5 hours)

- [ ] Run 8-hour scenario: `python main.py test_8hour`
- [ ] Verify: Runtime <1 minute
- [ ] Verify: Results match baseline (within 0.1%)
- [ ] Profile with: `python -m cProfile -s cumtime main.py test_8hour`
- [ ] Record: LUT time, thermal time, total time

**Performance checkpoint:** 8-hour scenario <1 minute

---

### Day 5 Evening: Week 1 Exit Validation

**Execute tests:**
```bash
# Test 1: Cache performance
python test_optimization.py --test-lut-cache
# Expected: Hit rate >70%, avg lookup <1 μs

# Test 2: Thermal performance
python test_optimization.py --test-thermal
# Expected: Individual step <0.2 μs, batch <0.5 μs

# Test 3: 8-hour scenario
python main.py test_8hour
# Expected: Runtime <1 minute, results match baseline
```

**Record metrics:**
- [ ] LUT cache hit rate: _________ % (target: >70%)
- [ ] Cache lookup time: _________ μs (target: <1 μs)
- [ ] Thermal step time: _________ μs (target: <0.2 μs)
- [ ] 8-hour scenario runtime: _________ seconds (target: <60 sec)
- [ ] Speedup vs baseline: _________× (target: 6×)

---

## WEEK 2: ADVANCED VECTORIZATION (12 hours)

### Day 1-2: Manager Vectorization (4 hours)

#### Step 4.1: Modify main simulation loop (2 hours)

- [ ] In `manager.py`, create `SimulationEngineOptimized` class
- [ ] Copy template from `optimization_implementation.md` Section 3
- [ ] Implement `register_component()` to cache thermal/pressure/PEM components
- [ ] Build index maps: `self._T_indices`, `self._P_indices`

**Expected result:** Components cached and indexed for batch operations

#### Step 4.2: Implement vectorized step (2 hours)

- [ ] Replace component-by-component loop with batch operations:
  ```python
  # Extract state into arrays
  T_array = np.array([c.T_K for c in self.thermal_components])
  Q_in = np.array([c.Q_in_W for c in self.thermal_components])
  
  # Batch update (Numba-compiled)
  T_new = batch_thermal_updates(T_array, 60, Q_in, Q_out, Q_loss, C, 298, 373)
  
  # Write back
  for i, comp in enumerate(self.thermal_components):
      comp.T_K = T_new[i]
  ```

- [ ] Test: 8-hour scenario should run <30 seconds
- [ ] Profile: Should show dramatic reduction in thermal calculation time

**Performance checkpoint:** Component-level thermal ops <10 μs total per timestep

---

### Day 3-4: Full-year simulation & profiling (5 hours)

#### Step 5.1: Run full year simulation (2 hours)

- [ ] Execute: `python main.py test_full_year`
- [ ] Monitor: Runtime (target: <2.5 hours), memory (target: <300 MB)
- [ ] Use profiler: `python -m memory_profiler main.py test_full_year`
- [ ] Record peak memory usage

**Performance checkpoint:** Full year <2.5 hours, memory <300 MB

#### Step 5.2: Identify new bottlenecks (2 hours)

- [ ] Profile full year: `python -m cProfile -s cumtime main.py test_full_year > profile.txt`
- [ ] Analyze `profile.txt`, identify top 5 time consumers
- [ ] Look for: LUT lookups still prominent? Component steps? I/O?
- [ ] Document findings in `OPTIMIZATION_LOG.md`

**Expected findings:** LUT and thermal should be minor now, new bottleneck likely in:
- Pressure calculations
- Efficiency calculations
- Component-specific logic (steam generation, etc.)

#### Step 5.3: Optimize memory (1 hour)

- [ ] Replace full result history with structured NumPy array
- [ ] See template in `optimization_implementation.md` Section 4
- [ ] Log only hourly (every 60 steps)
- [ ] Memory should drop 500 MB → 280 MB

**Expected result:** Memory <300 MB

---

### Day 5: Regression testing (3 hours)

#### Step 6.1: Numerical accuracy validation (1.5 hours)

- [ ] Compare optimized vs baseline results on 1-year simulation
- [ ] Check: LCOH, H2 production, temperatures, pressures
- [ ] Acceptable error: ±0.1% (quantization tolerance is ±1%)
- [ ] Create report: `ACCURACY_VALIDATION.md`

**Expected result:** Results match within ±0.1%

#### Step 6.2: Automated test suite (1.5 hours)

- [ ] Create `tests/test_optimization.py` with:
  - Cache performance tests
  - Thermal step accuracy tests
  - Full-year determinism test (same results twice)
  - Memory usage tests
  - Runtime benchmarks

- [ ] All tests should pass: `pytest tests/test_optimization.py -v`

**Expected result:** All tests green ✓

---

### Day 5 Evening: Week 2 Exit Validation

**Execute comprehensive test:**
```bash
python main.py test_full_year --profile --validate-accuracy
```

**Record metrics:**
- [ ] Full year runtime: _________ hours (target: <2.5 hours)
- [ ] Peak memory: _________ MB (target: <300 MB)
- [ ] Accuracy vs baseline: _________ % error (target: <0.1%)
- [ ] Speedup vs baseline: _________× (target: 8-10×)
- [ ] Cache hit rate: _________ % (target: >70%)
- [ ] Tests passing: _________/ _________ (target: all)

---

## WEEK 3: PARALLELIZATION (OPTIONAL, 10 hours)

### Day 1-2: Multi-core Numba (3 hours)

#### Step 7.1: Enable OpenMP in Numba (1.5 hours)

- [ ] Modify Numba functions with `@njit(parallel=True)`
- [ ] Example:
  ```python
  @njit(parallel=True)
  def batch_thermal_updates_parallel(T_array, ...):
      for i in prange(len(T_array)):  # Use prange instead of range
          # ... calculation ...
  ```

- [ ] Verify: Test on 2-core, 4-core systems
- [ ] Record scaling: Speedup per core

**Expected result:** 3-4× speedup on 4-core CPU (75% efficiency)

#### Step 7.2: Benchmark scaling (1.5 hours)

- [ ] Run full year with: 1, 2, 4, 8 cores
- [ ] Plot: Speedup vs cores (should be ~3× for 4 cores)
- [ ] Create: `SCALING_REPORT.md` with benchmark results

---

### Day 3: Dask Parallelization (4 hours)

#### Step 8.1: Optional Dask for I/O-bound operations (2 hours)

- [ ] If system has separate data loading: use Dask
- [ ] Most likely: central processing is CPU-bound, not I/O-bound
- [ ] Skip unless profiling shows disk I/O bottleneck

#### Step 8.2: GPU acceleration (Optional, 2 hours)

- [ ] If NVIDIA GPU available: explore CuPy
- [ ] Can achieve 100-150× speedup (10-15 minute runtime)
- [ ] Not recommended unless running >10 simulations/day

---

### Day 4-5: Final Validation (3 hours)

#### Step 9.1: Production readiness (1.5 hours)

- [ ] Clean up temporary code
- [ ] Update documentation
- [ ] Code review: 2 team members
- [ ] Security check: No hardcoded credentials

#### Step 9.2: Release & monitoring (1.5 hours)

- [ ] Merge feature branch to main
- [ ] Create release notes: `OPTIMIZATION_RELEASE_NOTES.md`
- [ ] Set up performance monitoring dashboard
- [ ] Plan: Monitor first 10 runs, watch for regressions

---

## FINAL VALIDATION CHECKLIST

### Performance Targets Met?

- [ ] Week 1: 6× speedup (24h → 4h) ✓
- [ ] Week 2: 8-10× speedup (24h → 2.5-3h) ✓
- [ ] Week 3 (optional): 24× speedup (24h → 1h) ✓
- [ ] Memory: <300 MB ✓
- [ ] Accuracy: ±0.1% vs baseline ✓

### Code Quality?

- [ ] No syntax errors ✓
- [ ] All imports present ✓
- [ ] Type hints on new functions ✓
- [ ] Docstrings explain performance implications ✓
- [ ] Unit tests pass ✓
- [ ] Integration tests pass ✓

### Documentation?

- [ ] Updated README.md with performance notes ✓
- [ ] Profiling guide for future developers ✓
- [ ] Known limitations documented ✓
- [ ] Release notes written ✓

### Monitoring?

- [ ] Performance dashboard set up ✓
- [ ] Alert thresholds defined ✓
- [ ] Regression detection enabled ✓
- [ ] Metrics logged to system ✓

---

## ROLLBACK PROCEDURE (IF NEEDED)

```bash
# 1. Identify problematic commit
git log --oneline --grep="performance"

# 2. Create hotfix branch
git checkout -b hotfix/performance-revert

# 3. Revert specific commit
git revert <commit-hash>

# 4. Test baseline performance
python main.py test_full_year

# 5. Push hotfix
git push origin hotfix/performance-revert

# 6. Create pull request for review
```

---

## TROUBLESHOOTING

### Issue: Numba compilation fails

**Solution:**
```python
# Enable Numba debug mode
import os
os.environ['NUMBA_DEBUG_JIT'] = '1'

# Re-run to see compilation errors
python main.py test_full_year
```

### Issue: Cache hit rate too low (<50%)

**Solution:**
- Expand `_warm_cache()` with more operating points
- Increase `_lookup_cache_size` to 20000
- Profile to see what queries are frequent

### Issue: Accuracy loss >0.5%

**Solution:**
- Reduce quantization tolerance: `_snap_tolerance_pct = 0.5` (was 1.0)
- Reduce temperature tolerance: `_snap_tolerance_k = 1.0` (was 2.0)
- Or disable quantization entirely: `self._quantize_enabled = False`

### Issue: Memory usage still high

**Solution:**
- Use `memory_profiler`: `python -m memory_profiler main.py test_full_year`
- Identify which components allocate most memory
- Consider using generators for large datasets
- Pre-allocate result arrays instead of growing lists

---

## QUICK REFERENCE: KEY FILES

| File | Purpose | Status |
|------|---------|--------|
| `h2plant/core/lut_manager_optimized.py` | LUT with cache | NEW (Week 1) |
| `h2plant/core/numba_ops_enhanced.py` | Numba functions | NEW (Week 1) |
| `h2plant/simulation/manager_optimized.py` | Vectorized manager | MODIFIED (Week 2) |
| `tests/test_optimization.py` | Performance tests | NEW (Week 2) |
| `OPTIMIZATION_LOG.md` | Profiling notes | NEW (Week 2) |
| `ACCURACY_VALIDATION.md` | Numerical accuracy | NEW (Week 2) |
| `SCALING_REPORT.md` | Multi-core results | NEW (Week 3) |

---

## SUCCESS METRICS

**Print this, fill in as you progress:**

```
WEEK 1 PROGRESS
═══════════════════════════════════════════
LUT Cache Hit Rate:           ____%  (target: >70%)
Cache Lookup Time:            ____μs (target: <1 μs)
Thermal Step Time:            ____μs (target: <0.2 μs)
8-Hour Scenario Runtime:      ____sec (target: <60 sec)
Speedup vs Baseline:          ____× (target: 6×)

Status: _____ (On Track / At Risk / Behind)


WEEK 2 PROGRESS
═══════════════════════════════════════════
Full Year Runtime:            ____hrs (target: <2.5 hrs)
Peak Memory:                  ____MB (target: <300 MB)
Numerical Accuracy Error:     ____.___% (target: <0.1%)
Overall Speedup:              ____× (target: 8-10×)
Tests Passing:                ___/%  (target: 100%)

Status: _____ (On Track / At Risk / Behind)


WEEK 3 PROGRESS (Optional)
═══════════════════════════════════════════
Multi-core Runtime (4 cores): ____hrs (target: <1 hr)
Scaling Efficiency:           ___% (target: >75%)
Final Speedup:                ____× (target: 24×)
GPU Runtime (if applicable):  ____mins (target: <15 mins)

Status: _____ (Complete / Skipped / In Progress)
```

---

**Document Version:** 1.0  
**Status:** Ready for deployment  
**Last Updated:** November 25, 2025  
**Next Review:** After Week 1 completion
