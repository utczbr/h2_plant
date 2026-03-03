# Forensic Timeline Audit: OPEX/LCOH Allegations (Historical vs Current)

Date: 2026-03-03

## Scope and Baselines
- Historical baseline (2026-03-02): `h2_plant/gui/layouts/generated/run_20260302_180628`
- Current baseline (2026-03-03): `scenarios/simulation_output` + current source/tests
- Currency labels are reported exactly as present in artifacts (no normalization applied).

## Normalized Allegation Buckets
- OPEX scaling
- Variable-cost integration
- LCOH completeness

Normalized matrix artifact:
- `docs/forensics/opex_lcoh_allegation_traceability_matrix.csv`

## Evidence Pack
### Historical baseline (2026-03-02)
- Config uses non-canonical variable resource IDs and no `lcoh_component` tags:
  - `h2_plant/gui/layouts/generated/run_20260302_180628/Economics/opex_config.yaml:22`
  - `h2_plant/gui/layouts/generated/run_20260302_180628/Economics/opex_config.yaml:33`
  - `h2_plant/gui/layouts/generated/run_20260302_180628/Economics/opex_config.yaml:43`
  - `h2_plant/gui/layouts/generated/run_20260302_180628/Economics/opex_config.yaml:54`
- OPEX report shows electricity/water/cooling as zero, biogas non-zero, and variant totals present:
  - `h2_plant/gui/layouts/generated/run_20260302_180628/simulation_output/opex_report.json:17`
  - `h2_plant/gui/layouts/generated/run_20260302_180628/simulation_output/opex_report.json:27`
  - `h2_plant/gui/layouts/generated/run_20260302_180628/simulation_output/opex_report.json:37`
  - `h2_plant/gui/layouts/generated/run_20260302_180628/simulation_output/opex_report.json:47`
  - `h2_plant/gui/layouts/generated/run_20260302_180628/simulation_output/opex_report.json:140`
- LCOH report has low/base/high variants, but base breakdown energy/water/compression are all zero:
  - `h2_plant/gui/layouts/generated/run_20260302_180628/simulation_output/lcoh_report.json:5`
  - `h2_plant/gui/layouts/generated/run_20260302_180628/simulation_output/lcoh_report.json:41`

### Current baseline (2026-03-03)
- Config uses canonical utility signals and includes `lcoh_component` tags:
  - `scenarios/Economics/opex_config.yaml:22`
  - `scenarios/Economics/opex_config.yaml:26`
  - `scenarios/Economics/opex_config.yaml:34`
  - `scenarios/Economics/opex_config.yaml:38`
  - `scenarios/Economics/opex_config.yaml:45`
  - `scenarios/Economics/opex_config.yaml:49`
  - `scenarios/Economics/opex_config.yaml:57`
  - `scenarios/Economics/opex_config.yaml:61`
- OPEX report shows non-zero variable quantities/costs for electricity, water, biogas, cooling and low/base/high totals:
  - `scenarios/simulation_output/opex_report.json:18`
  - `scenarios/simulation_output/opex_report.json:29`
  - `scenarios/simulation_output/opex_report.json:40`
  - `scenarios/simulation_output/opex_report.json:51`
  - `scenarios/simulation_output/opex_report.json:153`
- LCOH report has low/base/high variants and non-zero component breakdown including energy/water/compression:
  - `scenarios/simulation_output/lcoh_report.json:5`
  - `scenarios/simulation_output/lcoh_report.json:82`

## Implementation Path Verification
- OPEX variant recomputation against CAPEX low/high is implemented:
  - `h2_plant/economics/opex_generator.py:315`
  - `h2_plant/economics/opex_generator.py:333`
  - `h2_plant/economics/opex_generator.py:337`
  - `h2_plant/economics/opex_generator.py:342`
- Strict variable-signal resolution (fail-fast if configured signal missing) is implemented:
  - `h2_plant/economics/opex_generator.py:720`
  - `h2_plant/economics/opex_generator.py:724`
- LCOH pathway OPEX is currently proportional to production share, not causal allocation:
  - `h2_plant/economics/lcoh_calculator.py:347`
- Dispatch computes timestep effective dual-tier PPA and records it in history:
  - `h2_plant/control/engine_dispatch.py:993`
  - `h2_plant/control/engine_dispatch.py:1009`
  - `h2_plant/control/engine_dispatch.py:1011`
  - `h2_plant/control/engine_dispatch.py:1204`

## Reporting/Graph Reconciliation
- Integrated run generates OPEX/LCOH through shared economics path:
  - `h2_plant/run_integrated_simulation.py:333`
  - `h2_plant/run_integrated_simulation.py:344`
  - `h2_plant/run_integrated_simulation.py:400`
- Net-profit plotting uses paired CAPEX/OPEX low/base/high by default and fails fast on missing variants:
  - `tools/regenerate_net_profit_plotly.py:640`
  - `tools/regenerate_net_profit_plotly.py:644`
  - `tools/regenerate_net_profit_plotly.py:656`

## Quantitative Reconciliation (history_chunks vs reports)
Machine-readable evidence:
- `docs/forensics/opex_lcoh_reconciliation_evidence.json`

Current baseline (key checks):
- Annualized electricity consumption from history: 140,797,214.04 kWh/year
- OPEX electricity quantity: 140,797,214.04 kWh/year
- Annualized water: 74,472,512.59 kg/year (matches OPEX)
- Annualized biogas: 10,888,994.45 kg/year (matches OPEX)
- Annualized cooling: 13,159,320.91 kWh_th/year (matches OPEX)
- Estimated annual sale revenue from history (`P_sold * spot_price`): 5,258,589.97 EUR/year
- Electricity gross cost at 0.25 EUR/kWh: 35,199,303.51 EUR/year
- Net electricity cost if sale credit were applied: 29,940,713.54 EUR/year

Historical baseline (key checks):
- OPEX report includes biogas only among variable items (394,200 EUR/year)
- OPEX electricity/water/cooling are zero
- LCOH component breakdown energy/water/compression are zero

## Focused Regression Suite
Command executed:
- `pytest -q tests/test_opex_uncertainty_bands.py tests/test_regenerate_lcoh_variants.py tests/test_lcoh_breakdown_components.py tests/test_regenerate_net_profit_opex_variants.py tests/simulation/test_engine_dispatch_canonical_utility_columns.py`

Result:
- 23 passed
- Full log: `docs/forensics/regression_test_log.txt`

## Verdict Table
| Claim | Historical Evidence | Current Evidence | Status | Severity | Residual Risk | Recommended Action |
|---|---|---|---|---|---|---|
| OPEX does not scale with CAPEX scenario | Historical run has `total_opex_low/base/high` populated in OPEX and LCOH variant outputs (`.../opex_report.json:140`, `.../lcoh_report.json:5`) | Current run also has paired variants (`scenarios/simulation_output/opex_report.json:154`, `scenarios/simulation_output/lcoh_report.json:5`) | Not supported by baseline evidence | Low | Narrative confusion from older thesis text | Keep variant fields explicit in reporting tables |
| FCI-dependent OPEX uses central FCI only | Historical output already includes low/high OPEX totals and CAPEX variants | Current code recomputes low/high using CAPEX low/high (`opex_generator.py:333-347`) | Not supported by baseline evidence | Low | Reader may conflate different vintages | Keep explicit FCI used per scenario in report output |
| Variable costs missing from OPEX | Historical: electricity/water/cooling zero; biogas non-zero (`.../opex_report.json:17`, `:27`, `:37`, `:47`) | Current: all four variable components non-zero (`scenarios/simulation_output/opex_report.json:18`, `:29`, `:40`, `:51`) | True in historical baseline and fixed now (partial historical) | High | Legacy outputs can still circulate | Mark historical run superseded; keep canonical utility-column tests |
| Electricity sale credit not netted in OPEX/LCOH | Historical OPEX/LCOH do not model sale credit as an explicit negative OPEX term | Current OPEX still prices gross electricity consumption; sale revenue remains in cashflow path, not OPEX/LCOH | Still true (open defect) | High | Distorts cost decomposition and comparability | Add explicit electricity purchase and electricity sale-credit terms in OPEX/LCOH |
| LCOH uses incomplete OPEX componentization | Historical base breakdown energy/water/compression are zero (`.../lcoh_report.json:41`) | Current base breakdown includes energy/water/compression/non-zero (`scenarios/simulation_output/lcoh_report.json:82`) | True in historical baseline and fixed now | Medium | Legacy reports remain misleading | Keep `lcoh_component` tags mandatory in OPEX configs |
| LCOH only computed for central scenario | Historical and current outputs both include low/base/high variants | Confirmed by variant generation path (`lcoh_calculator.generate_variants`) | Not supported by baseline evidence | Low | Misread of stale snapshots | Keep fail-fast tests for missing variant fields |
| Pathway variable-cost allocation is causal | Historical pathway OPEX uses pro-rata structure | Current still uses production-share OPEX allocation (`lcoh_calculator.py:347`) | Still true (open defect) | Medium | Pathway ranking bias | Implement causal pathway allocation (PEM/SOEC/ATR-specific cost drivers) |
| Economics uses timestep dual-tier PPA pricing directly | Historical economics used flat variable price and mismatched signals | Current dispatch computes timestep `ppa_price_effective_eur_mwh`, but OPEX economics still uses static unit prices from config | Still true (open defect) | High | Electricity OPEX may diverge from dispatch economics | Integrate timestep price series (`ppa_price_effective_eur_mwh`) into economic aggregation |

## Public API / Interface Changes
- Investigation phase: none.
- Follow-up remediation (not implemented in this audit): define explicit pricing and pathway-allocation interfaces.

