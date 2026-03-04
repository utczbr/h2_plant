# Methodology for OPEX and LCOH under `--capacity-mode design`

## 1. Objective and scope

This chapter defines the computational methodology used to estimate annual operating expenditure (OPEX) and levelized cost of hydrogen (LCOH) for the PEM+SOEC+ATR hydrogen plant when CAPEX is generated with `--capacity-mode design`.

The workflow is implemented as a coupled pipeline:

1. CAPEX estimation (design-capacity based)
2. OPEX estimation (history-driven variable costs + CAPEX-coupled fixed/maintenance factors)
3. Discounted LCOH estimation (low/base/high variants)

The methodology is intentionally code-faithful to the current implementation and should be treated as the normative reference for reproducibility.

## 2. Software interfaces and data sources

### 2.1 Main execution interfaces

- CAPEX+OPEX driver: `tools/regenerate_capex.py`
- LCOH driver: `tools/regenerate_lcoh.py`

### 2.2 Core computational modules

- CAPEX sizing and costing: `h2_plant/economics/capex_generator.py`
- CAPEX cost strategies: `h2_plant/economics/cost_strategies.py`
- OPEX orchestration and streaming extraction: `h2_plant/economics/opex_generator.py`
- OPEX item strategies: `h2_plant/economics/opex_strategies.py`
- LCOH calculation and variant coupling: `h2_plant/economics/lcoh_calculator.py`

### 2.3 Configuration and schema

- CAPEX mappings: `scenarios/Economics/equipment_mappings.yaml`
- OPEX item configuration: `scenarios/Economics/opex_config.yaml`
- Economic defaults (discount rate, project years): `scenarios/economics_parameters.yaml`
- CAPEX/OPEX/LCOH models: `h2_plant/economics/models.py`, `h2_plant/economics/opex_models.py`, `h2_plant/economics/lcoh_models.py`

### 2.4 Primary simulation history signals

From `history_chunks/chunk_*.parquet` (or CSV fallback), the methodology uses, at minimum:

- `minute`
- `electricity_consumption_kwh_step`
- `ppa_price_effective_eur_mwh`
- `sold_energy_mwh_step`
- `spot_price`
- `water_makeup_kg_step`
- `biogas_feed_kg_step`
- `cooling_duty_kwh_th_step`
- `H2_pem_kg`, `H2_soec_kg`, `H2_atr_kg`

## 3. Notation

- `i`: equipment mapping index (CAPEX)
- `j`: OPEX item index
- `y`: simulation-relative year index (`y = 1..N` in reporting)
- `t`: time step index (typically 1-minute resolution)
- `S_i`: design capacity of equipment `i`
- `C_p0`: base purchased cost
- `C_BM`: bare module cost
- `FCI`: fixed capital investment (total installed CAPEX)
- `Q_{j,y}`: annual quantity of OPEX item `j` in year `y`
- `p_j`: configured unit price of item `j`
- `m_j`: configured cost multiplier of item `j`
- `r`: discount rate
- `DF_y`: yearly discount factor

## 4. CAPEX methodology under `--capacity-mode design`

## 4.1 Capacity extraction hierarchy

For each equipment mapping `i`, the design capacity `S_i` is extracted from the topology IDs using a fixed priority when mode is `design`:

1. Direct design/rated attributes (`max_power_kw`, `area_m2`, `volume_m3`, `capacity_*`, etc.)
2. Derived geometric or composite attributes (for example vessel volume from `D` and `L`, tank-array totals)
3. History fallback only if design information is not available

This ordering is enforced in `CapexGenerator._extract_capacity` and is central to the meaning of `design` mode: cost reflects installed/sized capability, not only realized dispatch.

## 4.2 Aggregation across parallel topology IDs

If multiple components map to one cost tag, capacities are aggregated by the configured operator:

- `sum`
- `max`
- `avg`

Modular entries may be split by module definitions and multiplied by number of units.

## 4.3 Cost strategy equations

For Turton-type correlations:

`log10(C_p0) = K1 + K2*log10(S_i) + K3*(log10(S_i))^2`

`C_p,current = C_p0 * (CEPCI_current / CEPCI_base)`

Simple module-factor form:

`C_BM,i = C_p,current * F_BM * F_m`

B-factor form:

`C_BM,i = C_p,current * (B1 + B2*F_m*F_p) * F_complex`

Additional strategies (vendor quote, scaling, percentage, fixed/excluded) are handled by strategy dispatch in `cost_strategies.py`.

## 4.4 Uncertainty and installed cost

For each CAPEX entry, low/high bands are derived from the AACE class range (`accuracy_range`). For Class 4, the factors are `(0.70, 1.30)`.

`C_BM,i_low = C_BM,i * f_low`

`C_BM,i_high = C_BM,i * f_high`

Block-level installation is computed by applying configured installation percentages `f_{b,k}` to block equipment totals.

`C_inst,b = C_BM,b * sum_k(f_{b,k})`

Total installed CAPEX (used as FCI):

`FCI = sum_i(C_BM,i) + sum_b(C_inst,b)`

Low/high installed totals are computed analogously.

## 5. OPEX methodology (yearly-variable with scalar compatibility)

## 5.1 Item taxonomy and strategy engine

OPEX items are declared in YAML and evaluated by strategy:

- `variable`
- `fixed`
- `factor`
- `turton_labor`
- `scaling`

For each item, the engine produces:

- annual quantity (or fixed base)
- annual cost
- formula trace
- optional causal pathway shares

## 5.2 Strict signal resolution

For streaming CSV/parquet extraction, required variable signals are resolved with strict matching policy. Missing required signals raise an exception, preventing silent zero-cost bias.

## 5.3 Time span and annualization

When `minute` is available, simulation span is inferred and annualization uses:

`AF = 8760 / H_sim`

where `H_sim` is inferred simulation hours (with timestep correction from positive minute differences). If inferred and configured hours differ, inferred hours are preferred.

## 5.4 Relative-year binning and partial-year coverage

Year binning uses simulation-relative windows:

`year_idx = floor((minute - minute_origin) / 525600)`

with fixed 8760 h windows and no leap-year adjustment.

Covered hours per year `h_y` are computed from overlap of observed minute span with each year window.

## 5.5 Variable-cost equations

### 5.5.1 Fixed-price variable item

For non-dynamic variable item `j`:

`C_{j,y} = Q_{j,y} * p_j * m_j`

`Q_{j,y}` is built from configured metric (`sum`, `avg`, `max`) over year `y`.

### 5.5.2 Dynamic-price variable item

For items with `resource_id` and `price_resource_id`:

`C_{j,y} = sum_{t in y}(q_{j,t} * pi_t) * p_j * m_j`

In current configuration, electricity import/export items set `p_j = 1.0`; conversion/sign are handled by `m_j` (for example `0.001` for kWh->MWh factor and `-1.0` for credit sign).

## 5.6 Fixed, factor, labor and scaling strategies

## 5.6.1 Fixed

`C_j = fixed_annual_value`

## 5.6.2 Factor

`C_j = base_reference * factor`

where base reference is typically `FCI` or `Labor`.

## 5.6.3 Turton labor

`N_shift = sqrt(6.29 + 31.7*P^2 + 0.23*Nnp)`

`C_labor = N_shift * shifts * hours_per_year * wage`

## 5.6.4 Scaling

`C_j = base_cost * (quantity / ref_production)^n`

## 5.6.5 Yearly distribution for non-variable strategies

When yearly arrays are produced, non-variable annual values are distributed by covered-year fraction:

`C_{j,y} = C_{j,annual} * (h_y / 8760)`

## 5.7 Category totals, credits and cashflow

For each year:

`OPEX_var,y`, `OPEX_fix,y`, `OPEX_maint,y`

`OPEX_tot,y = OPEX_var,y + OPEX_fix,y + OPEX_maint,y`

Credit terms are explicit (`is_credit: true`) and accumulated as `Credit_y`.

Cashflow OPEX is:

`OPEX_cash,y = OPEX_tot,y - Credit_y`

This preserves a clear distinction between gross operating cost and net cash outflow.

## 5.8 Backward-compatible scalar annual metrics

To preserve compatibility with legacy consumers, scalar annual totals are computed as equivalent annualized values from yearly vectors:

`OPEX_eq_annual = (sum_y OPEX_y / sum_y h_y) * 8760`

The same transformation is applied to variable/fixed/maintenance totals and to low/high/cashflow variants.

## 5.9 OPEX low/high variants

Preferred method:

1. Re-evaluate full OPEX model with `FCI_low`
2. Re-evaluate full OPEX model with `FCI_high`

Fallback if CAPEX low/high are unavailable:

- Apply AACE uncertainty factors to base OPEX and scale yearly vectors accordingly.

## 6. LCOH methodology (discounted yearly streams)

## 6.1 Hydrogen production extraction

Hydrogen production is read from history as step quantities:

- `H2_pem_kg`
- `H2_soec_kg`
- `H2_atr_kg`

Totals are accumulated per simulation-relative year with the same binning rule used by OPEX.

## 6.2 Discounting framework

For yearly index `y` (1-based in economics):

`DF_y = 1 / (1 + r)^y`

Present values:

`PV_OPEX = sum_{y=1..N}(OPEX_y * DF_y)`

`PV_H2 = sum_{y=1..N}(H2_y * DF_y)`

Plant-level LCOH:

`LCOH = (CAPEX + PV_OPEX) / PV_H2`

## 6.3 Variant coupling (low/base/high)

The LCOH generator builds variants with consistent pairing:

- CAPEX variant: `total_installed_cost_low/base/high`
- OPEX variant: `total_opex_low/base/high` and yearly arrays if available

If yearly low/high OPEX arrays are missing but base exists, scaled fallback is applied from base yearly series.

## 6.4 Pathway allocation for decomposition

Pathway CAPEX:

- Allocated from block summaries by block-name matching (`PEM`, `SOEC`, `ATR`)
- If unmatched, fallback to production-share allocation

Pathway OPEX:

- Variable item allocation uses configured causal drivers (`pathway_driver_resource_ids`)
- Non-variable and unresolved shares fallback to production shares

Pathway LCOH:

`LCOH_k = (CAPEX_k + PV_OPEX * s_k) / PV_H2,k`

where `s_k` is pathway OPEX share.

## 6.5 Horizon mismatch handling

If requested project years exceed available yearly history, discounted calculations are performed on available history horizon and a warning is emitted.

## 7. Explicit effect of `--capacity-mode design`

1. CAPEX sizing basis changes to design/rated capacities.
2. Therefore `FCI` changes relative to `history` mode.
3. OPEX variable terms remain history-driven (dispatch and consumption signals).
4. OPEX factor terms linked to `FCI` (insurance, maintenance percentages, reserves) respond directly to the design-mode `FCI`.
5. LCOH numerator changes through both CAPEX and CAPEX-coupled OPEX factors.

## 8. Reproducibility protocol

## 8.1 Minimal command sequence

```bash
python3 tools/regenerate_capex.py scenarios \
  --capacity-mode design \
  --history-dir scenarios/20_years_derived/history_chunks \
  --output-dir scenarios/20_years_derived

python3 tools/regenerate_lcoh.py scenarios/20_years_derived \
  --economics-dir scenarios/20_years_derived \
  --history-dir scenarios/20_years_derived/history_chunks \
  --output-dir scenarios/20_years_derived
```

## 8.2 Acceptance checks

1. `opex_report.json` contains yearly arrays and scalar totals.
2. `total_opex` equals equivalent annualization of `total_opex_by_year` within floating tolerance.
3. `lcoh_report.json` includes `discounted_opex_pv` and `discounted_h2_pv`.
4. Low/base/high variant consistency is preserved across CAPEX, OPEX and LCOH.

## 9. Assumptions and defaults

1. Currency: EUR.
2. Relative simulation years: fixed 8760 h windows from `minute_origin`.
3. Default economics from `scenarios/economics_parameters.yaml` (currently `discount_rate = 0.08`, `project_lifetime_years = 20`).
4. Dynamic electricity economics from history price signals (`ppa_price_effective_eur_mwh`, `spot_price`).
5. Credits are negative OPEX terms and are also reported through cashflow-adjusted OPEX fields.
6. Although YAML may specify another global capacity mode, this methodology applies to runs explicitly invoked with `--capacity-mode design`.

## 10. Limitations

1. Year binning is simulation-relative, not calendar-year aligned.
2. If required OPEX variable columns are absent, strict extraction raises an error and no variable OPEX is synthesized automatically.
3. Discounting assumes one cashflow point per simulation-relative year and does not model intra-year discounting.
4. Pathway allocation for non-causal items depends on fallback production-share logic when explicit drivers are unavailable.

