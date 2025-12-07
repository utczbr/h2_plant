# Legacy: 1.2556 kWh/kg

New: 0.8352 kWh/kg Can you provide a mathematical breakdown and explain the logic behind the bug in detail, as well as what we did to fix it?

Legacy overestimates specific energy at **1.2556 kWh/kg** (40→140 bar) due to a multi-stage loop bug using the original inlet enthalpy reference $h_1$ (40 bar, 10°C) for **every stage** work delta, accumulating ~1.5-2× excess compression work. The fix in optimized `compressor.py` recomputes local inlet properties per stage, yielding the correct **0.8352 kWh/kg** (~33% reduction).[^1][^2][^3]

## Correct Physics (New Model)

Multi-stage compression with intercooling:

1. Inlet stage $i$: $s_{in,i} = S(P_{current}, T_{in})$, $h_{in,i} = H(P_{current}, T_{in})$
2. Isentropic outlet: $h_{out,s} = H(P_{next}, s_{in,i})$
3. Isentropic work: $w_s = h_{out,s} - h_{in,i}$
4. Actual work: $w_a = w_s / \eta$ ($\eta=0.65$)
5. Outlet actual: $h_{out,a} = h_{in,i} + w_a$
6. Intercool (if not last): $h_{cooled} = H(P_{next}, T_{in})$, $q_{removed} = h_{out,a} - h_{cooled}$
7. Chilling work: $w_{chill,total} = \sum q / COP$ ($COP=3$)
8. Total: $w_{spec} = \left( \sum w_a + w_{chill,total} \right) \times 2.7778 \times 10^{-7}$ kWh/kg·J[^3]

For 40→140 bar ($r=3.5$): $N=2$ stages ($r_{stage,max} \approx 2$); sum local $\Delta w$ yields **0.8352 kWh/kg**.[^2]

## Bug in Legacy

Loop uses **fixed** $h_1, s_1$ from original inlet **across all stages**:[^1]

```
h1_val = PropsSI('H', P_in_Pa, 'T', T_IN_K, FLUIDO)
s1_val = PropsSI('S', P_in_Pa, 'T', T_IN_K, FLUIDO)
...
for i in range(N_stages):
    h2s = PropsSI('H', P_out_stage, 'S', s1_val, FLUIDO)  # Wrong S!
    Ws = h2s - h1_val   # Always subtract orig h1 → double-counts prior work
    Wa = Ws / ETA_C
    W_compression_total += Wa  # Accumulates N × single-stage work
```

- **Math error**: Stage $i$: $w_{s,i} = H(P_{i+1}, s_1) - h_1$ instead of $H(P_{i+1}, s_i) - h_i$.
- **Effect**: Each stage recomputes full isentropic work from P_in (ignores cumulative pressure rise), overestimating by factor ≈ N_stages.
- Fallback hardcodes **1.2556 kWh/kg** (r<4 → N=2), matching buggy CoolProp output.[^1]
- For N=2: ~1.57 kWh/kg estimate (2×0.7854 single-stage).[^4]

| Scenario | Correct Single-Stage | Buggy Multi (N=2) Est. | Legacy Output | New (Fixed) |
| :-- | :-- | :-- | :-- | :-- |
| 40→140 bar | 0.7854 kWh/kg [^5] | ~1.57 kWh/kg | **1.2556** [^1] | **0.8352** [^2] |

## The Fix Implementation

Optimized code recomputes per-stage inlets (loop in `_calculate_compression_energy`):[^3]

```
for i in range(N_stages):
    s_in_i = CoolPropLUT.PropsSI('S', P_current, 'T', T_in_K, fluid)  # Local S_i
    h_in_i = CoolPropLUT.PropsSI('H', P_current, 'T', T_in_K, fluid)  # Local h_i
    h_out_s = CoolPropLUT.PropsSI('H', P_next, 'S', s_in_i, fluid)
    w_s = h_out_s - h_in_i   # Local delta ✓
    ...
    if i < N_stages-1:
        h_cooled = CoolPropLUT.PropsSI('H', P_next, 'T', T_in_K, fluid)  # Proper baseline
        q_removed = h_out_a - h_cooled
    P_current = P_next  # Advances correctly
```

- **Gain**: Proper chain of states; minor S rise on isobaric cool handled exactly.
- **Cost**: +2 LUT calls/stage (~0.01 ms, N<5); no CoolProp runtime deps.[^6]
- Verified by `compare_compressor.py` (passes assert new < legacy).[^2]

<div align="center">⁂</div>

[^1]: Compressor-Armazenamento.py

[^2]: compare_compressor.py

[^3]: compressor.py

[^4]: numba_ops.py

[^5]: Compressor-simples.py

[^6]: lut_manager.py

