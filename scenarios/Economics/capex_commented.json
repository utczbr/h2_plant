{
  "meta_info": {
    "description": "CORRECTED CAPEX calculation instructions with topology mapping and power consumption data",
    "methodology": "Turton et al. (Analysis, Synthesis and Design of Chemical Processes, 4th Edition) with CEPCI inflation adjustment",
    "global_constants": {
      "CEPCI_Base_Turton": 397,
      "CEPCI_Target_2026": 820,
      "inflation_factor": 2.0655,
      "inflation_formula": "CEPCI_2026 / CEPCI_Base = 820 / 397 = 2.0655"
    },
    "currency": "USD (converted from EUR where noted)",
    "data_sources": {
      "topology": "scenarios/plant_topology.yaml",
      "power_consumption": "scenarios/Economics/max_power_consumption.json",
      "capex_coefficients": "scenarios/Economics/capex_instructions_full.json"
    },
    "power_consumption_summary_kw": {
      "SOEC_System": 11520.0,
      "PEM_System": 5000.2,
      "SOEC_H2_Compression_Train": 868.7,
      "SOEC_H2_Intercooler_Fans": 1901.9,
      "SOEC_O2_Compression": 58.5,
      "SOEC_Steam_Compression": 518.0,
      "LP_HP_Storage_Compression": 642.2,
      "HP_Intercooler_Fans": 264.2,
      "ATR_Compression_Biogas": 167.6,
      "ATR_Compression_H2": 159.4,
      "ATR_O2_Compressor": 125.0,
      "Chillers_Total": 75.9,
      "Dry_Cooler_Fans_Other": 116.6,
      "Pumps_Total": 7.3,
      "ESTIMATED_TOTAL_MW": 21.4,
      "note": "Values are peak power consumption from 168-hour simulation"
    },
    "corrections_applied": [
      "Removed H-101 (Fired Heater) - not in source file",
      "Removed C-101 (Air Compressor) - not in source file",
      "Corrected F_m = 2.41 for PMP-2, PMP-3, PMP-4 (was 2.1)",
      "Corrected F_m = 2.41 for MCY-4 (was 2.1)",
      "Added specific F_p values for different dry cooler locations",
      "Clarified SOEC cost: \u20ac24.2M fixed (ROGER) vs 2750 USD/kW alternative",
      "Noted PEM cost source: No data in CSV, 1100 USD/kW is external estimate"
    ]
  },
  "blocks": {
    "SOEC": {
      "description": "Solid Oxide Electrolyzer Cell system - Primary hydrogen production pathway",
      "process_location": "Process Step 10 - Main gas train entry point",
      "components": [
        {
          "tag": "SOEC_Cluster",
          "topology_id": "SOEC_Cluster",
          "name": "SOEC Stack Module (6 modules)",
          "component_type": "Solid Oxide Electrolyzer Cell System",
          "process_description": "Electrochemical water splitting at high temperature (650-850\u00b0C). Produces hot wet H2 (152\u00b0C, cathode) and O2 (anode). Requires superheated steam feed at 156.9\u00b0C, 5 bar.",
          "power_consumption": {
            "reference": "SOEC_System",
            "value_w": 11520000.0,
            "value_mw": 11.52,
            "note": "Rated capacity - 6 modules \u00d7 2.4 MW each"
          },
          "method": "Direct cost from reliable source",
          "cost_basis": {
            "value_eur": 24197358.17,
            "value_usd": 26617093.99,
            "source": "ROGER study",
            "year": "2026",
            "alternative_method": {
              "specific_cost_usd_per_kw": 2750,
              "note": "Alternative linear estimation if fixed cost not used",
              "calculation": "P_total_kW \u00d7 2750 USD/kW"
            }
          },
          "calculation_steps": [
            "PRIMARY METHOD:",
            "1. Use fixed cost from ROGER: \u20ac24,197,358.17",
            "2. Convert to USD: \u20ac24,197,358.17 \u00d7 1.1 = $26,617,093.99",
            "3. NO CEPCI adjustment (already 2026 pricing)",
            "",
            "ALTERNATIVE METHOD (if capacity scaling needed):",
            "1. Determine SOEC electrical capacity: P_total (kW)",
            "2. Apply specific cost: CAPEX = P_total \u00d7 2750 USD/kW",
            "3. NO CEPCI adjustment (2026 market projection)"
          ],
          "cost_formula": "Formula not found",
          "C_BM": 26617093.99
        },
        {
          "tag": "MTC-1",
          "topology_id": "SOEC_Steam_Compressor_1 / SOEC_Steam_Compressor_2",
          "name": "Multistage Turbo-Centrifugal Steam Compressor (SOEC Feed)",
          "component_type": "Centrifugal Compressor",
          "process_description": "Two-stage steam compression: Stage 1 (1.0\u21925.0 bar), Stage 2 (2.3\u21925.0 bar). Pressurizes saturated steam before attemperator. Operates in parallel with attemperator bypass for precise superheat control.",
          "power_consumption": {
            "stage_1": {
              "reference": "SOEC_Steam_Compressor_1",
              "value_w": 302919.31,
              "value_kw": 302.9
            },
            "stage_2": {
              "reference": "SOEC_Steam_Compressor_2",
              "value_w": 215079.55,
              "value_kw": 215.1
            },
            "total_kw": 518.0
          },
          "capacity_variable": {
            "symbol": "W_shaft",
            "description": "Shaft power",
            "unit": "kW",
            "valid_range": "450 - 3000 kW",
            "design_capacity": 518.0
          },
          "coefficients": {
            "K1": 2.2891,
            "K2": 1.3604,
            "K3": -0.1027,
            "F_m": 2.5,
            "F_m_note": "Stainless Steel 316 or Alloy Steel required for 250\u00b0C steam",
            "F_BM": 2.15,
            "F_BM_note": "Package installation for centrifugal unit"
          },
          "calculation_steps": [
            "1. Verify: 450 \u2264 W_shaft \u2264 3000 kW",
            "2. Base cost: log10(Cp0) = 2.2891 + 1.3604\u00d7log10(W_shaft) - 0.1027\u00d7[log10(W_shaft)]\u00b2",
            "3. Convert: Cp0 = 10^(log10(Cp0))",
            "4. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "5. Apply factors: CAPEX = Cp_2026 \u00d7 F_BM \u00d7 F_m = Cp_2026 \u00d7 2.15 \u00d7 2.5 = Cp_2026 \u00d7 5.375"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * F_BM * F_p",
          "C_BM": 1863917.93
        },
        {
          "tag": "ATT-1",
          "topology_id": "SOEC_Feed_Attemperator",
          "name": "Attemperator (Venturi)",
          "component_type": "Spray Water Attemperator",
          "process_description": "Mixes compressed steam (from MTC-1) with bypass liquid water (7.88% of total flow) to achieve precise 156.9\u00b0C superheat at 5 bar. Critical for SOEC feed quality - prevents condensation while avoiding excessive temperature.",
          "power_consumption": {
            "note": "Passive device - no electrical consumption. Relies on pressure differential."
          },
          "capacity_variable": {
            "symbol": "Q_vol",
            "description": "Volumetric flow rate",
            "unit": "m\u00b3/s",
            "valid_range": "0.05 - 50 m\u00b3/s",
            "design_capacity": 0.05
          },
          "coefficients": {
            "K1": 3.6298,
            "K2": -0.4991,
            "K3": 0.0411,
            "F_m": 3.1,
            "F_p": 1.0,
            "B1": 2.25,
            "B2": 1.82
          },
          "cost_reference": {
            "value_eur": 19778.0,
            "note": "Fixed cost provided in source for reference"
          },
          "calculation_steps": [
            "1. Verify: 0.05 \u2264 Q_vol \u2264 50 m\u00b3/s",
            "2. Base cost: log10(Cp0) = 3.6298 - 0.4991\u00d7log10(Q_vol) + 0.0411\u00d7[log10(Q_vol)]\u00b2",
            "3. Convert: Cp0 = 10^(log10(Cp0))",
            "4. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "5. Bare module: CAPEX = Cp_2026 \u00d7 [2.25 + (1.82 \u00d7 3.1 \u00d7 1.0)] = Cp_2026 \u00d7 7.892"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * [B_1 + (B_2 * F_m * F_p)]",
          "C_BM": 363852.58
        },
        {
          "tag": "HTX-3 / EBL-1",
          "topology_id": "SOEC_H2_Boiler / SOEC_H2_ElectricBoiler_PSA / SOEC_Steam_Boiler",
          "name": "Electric Process Heater",
          "component_type": "Electrical Resistance Heater",
          "process_description": "Multiple electric heaters in SOEC train: (1) H2_Boiler - warms dried H2 to 15\u00b0C before compression, (2) ElectricBoiler_PSA - reheats to 40\u00b0C before PSA, (3) Steam_Boiler - generates saturated steam at 105\u00b0C from preheated water.",
          "power_consumption": {
            "note": "Varies by location - typically 25-3000 kW depending on duty. See SOEC_Steam_Boiler parameters (max_power_kw: 3000)."
          },
          "capacity_variable": {
            "symbol": "P_heat",
            "description": "Heating power",
            "unit": "kW",
            "valid_range": "10 - 1000 kW",
            "design_capacity": 3150.0
          },
          "coefficients": {
            "K1": 2.858,
            "K2": 0.8209,
            "K3": 0.0075,
            "F_BM": 1.3
          },
          "calculation_steps": [
            "1. Verify: 10 \u2264 P_heat \u2264 1000 kW",
            "2. Base cost: log10(Cp0) = 2.858 + 0.8209\u00d7log10(P_heat) + 0.0075\u00d7[log10(P_heat)]\u00b2",
            "3. Convert: Cp0 = 10^(log10(Cp0))",
            "4. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "5. Apply: CAPEX = Cp_2026 \u00d7 1.3"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * F_BM",
          "C_BM": 1946774.64,
          "cost_breakdown": "Sum of 2 units: $973,387 + $973,387"
        },
        {
          "tag": "DCL-7 / DCL-8 / DCL-9 / DCL-10 / DCL-11",
          "topology_id": "SOEC_H2_Intercooler_1 through SOEC_H2_Intercooler_6",
          "name": "Dry Cooler Heat Exchangers - SOEC H2 Train (Shell & Tube, Fixed)",
          "component_type": "Air-Cooled Heat Exchanger (Dry Cooler)",
          "process_description": "Six intercoolers in H2 compression train. Cool compressed gas from ~135\u00b0C to 40\u00b0C between stages (S1\u2192S6). Each cooler handles ~1800 kW design load. Critical for maintaining temperature limits and condensing residual moisture.",
          "power_consumption": {
            "per_unit": {
              "reference": "SOEC_H2_Intercooler_X_fan (X=1 to 6)",
              "value_w": 316983.67,
              "value_kw": 317.0,
              "note": "Fan power per intercooler - 6 units total"
            },
            "total_kw": 1901.9
          },
          "capacity_variable": {
            "symbol": "A_hx",
            "description": "Heat transfer area",
            "unit": "m\u00b2",
            "valid_range": "10 - 1000 m\u00b2",
            "design_capacity": 10.0
          },
          "coefficients": {
            "K1": 4.3247,
            "K2": -0.303,
            "K3": 0.1634,
            "F_m": 2.7,
            "F_m_note": "Stainless Steel 316SS",
            "F_p": 1.092,
            "F_p_note": "Specific to SOEC H2 compression train",
            "B1": 1.63,
            "B2": 1.66
          },
          "calculation_steps": [
            "1. Verify: 10 \u2264 A_hx \u2264 1000 m\u00b2",
            "2. Base cost: log10(Cp0) = 4.3247 - 0.303\u00d7log10(A_hx) + 0.1634\u00d7[log10(A_hx)]\u00b2",
            "3. Convert: Cp0 = 10^(log10(Cp0))",
            "4. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "5. Bare module: CAPEX = Cp_2026 \u00d7 [1.63 + (1.66 \u00d7 2.7 \u00d7 1.092)]",
            "6. Final: CAPEX = Cp_2026 \u00d7 6.521"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * [B_1 + (B_2 * F_m * F_p)]",
          "C_BM": 1193512.91,
          "note": "Missing specific data. Calculated using Minimum Valid Range as default design basis.",
          "cost_breakdown": "Sum of 5 units: $238,703 + $238,703 + $238,703 + $238,703 + $238,703"
        },
        {
          "tag": "DLC-4 / DLC-5 / DLC-6",
          "topology_id": "SOEC_O2_Drycooler_1 through SOEC_O2_Drycooler_4",
          "name": "Dry Cooler Heat Exchangers - SOEC O2 Train",
          "component_type": "Air-Cooled Heat Exchanger (Dry Cooler)",
          "process_description": "Four intercoolers in O2 compression train (4 stages to 15 bar). Cool oxygen from compression heat to 30\u00b0C. Lower duty than H2 train due to smaller mass flow.",
          "power_consumption": {
            "per_unit": {
              "reference": "SOEC_O2_Drycooler_X_fan (X=1 to 4)",
              "value_w": 2416.33,
              "value_kw": 2.4,
              "note": "Fan power per intercooler - 4 units total"
            },
            "total_kw": 9.7,
            "additional": {
              "reference": "SOEC_Steam_Drycooler_fan",
              "value_w": 2416.33,
              "value_kw": 2.4,
              "note": "Steam drycooler (150\u00b0C target)"
            }
          },
          "coefficients": {
            "K1": 4.3247,
            "K2": -0.303,
            "K3": 0.1634,
            "F_m": 2.7,
            "F_p": 1.092,
            "F_p_note": "Same as H2 train",
            "B1": 1.63,
            "B2": 1.66
          },
          "calculation_steps": [
            "Same as DCL-7/8/9/10/11"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * [B_1 + (B_2 * F_m * F_p)]",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "m2"
          }
        },
        {
          "tag": "KOT-3 / KOT-4",
          "topology_id": "SOEC_H2_KOD_1 / SOEC_H2_KOD_2",
          "name": "Knock-Out Drum Separator (with demister)",
          "component_type": "Vertical Pressure Vessel with Demister Pad",
          "process_description": "Two-stage water knockout: KOD_1 after DryCooler (40\u00b0C), KOD_2 after Chiller (4\u00b0C). Separate condensed water from H2 stream before compression. Essential for protecting compressors from liquid slugs.",
          "power_consumption": {
            "note": "Passive separation devices - no electrical consumption"
          },
          "capacity_variable": {
            "symbol": "V",
            "description": "Vessel volume",
            "unit": "m\u00b3",
            "valid_range": "0.3 - 520 m\u00b3",
            "design_capacity": 1.715
          },
          "coefficients": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 3.1,
            "F_p": 1.0,
            "B1": 2.25,
            "B2": 1.82,
            "F_multi": 1.2
          },
          "calculation_steps": [
            "1. Verify: 0.3 \u2264 V \u2264 520 m\u00b3",
            "2. Base cost: log10(Cp0) = 3.4974 + 0.4485\u00d7log10(V) + 0.1074\u00d7[log10(V)]\u00b2",
            "3. Convert: Cp0 = 10^(log10(Cp0))",
            "4. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "5. Bare module: C_BM = Cp_2026 \u00d7 [2.25 + (1.82 \u00d7 3.1 \u00d7 1.0)]",
            "6. Apply demister: CAPEX = C_BM \u00d7 1.2 = Cp_2026 \u00d7 9.470"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * [B_1 + (B_2 * F_m * F_p)] * F_multi",
          "C_BM": 114909.99,
          "cost_breakdown": "Sum of 2 units: $57,455 + $57,455"
        },
        {
          "tag": "MTC-2 / MTC-3",
          "topology_id": "SOEC_O2_compressor_1 through SOEC_O2_compressor_4",
          "name": "Multistage Centrifugal Oxygen Compressor (4 stages)",
          "component_type": "Centrifugal Compressor",
          "process_description": "Four-stage O2 compression from ~1 bar to 15 bar. Handles anode gas from SOEC. Requires nickel alloys/Monel for high-pressure oxygen service (fire/explosion hazard mitigation).",
          "power_consumption": {
            "stage_1": {
              "reference": "SOEC_O2_compressor_1",
              "value_w": 15239.79,
              "value_kw": 15.2
            },
            "stage_2": {
              "reference": "SOEC_O2_compressor_2",
              "value_w": 15233.56,
              "value_kw": 15.2
            },
            "stage_3": {
              "reference": "SOEC_O2_compressor_3",
              "value_w": 15222.71,
              "value_kw": 15.2
            },
            "stage_4": {
              "reference": "SOEC_O2_compressor_4",
              "value_w": 12915.42,
              "value_kw": 12.9
            },
            "total_kw": 58.5
          },
          "capacity_variable": {
            "symbol": "W_shaft",
            "description": "Shaft power",
            "unit": "kW",
            "valid_range": "450 - 3000 kW",
            "design_capacity": 58.61
          },
          "coefficients": {
            "K1": 2.2891,
            "K2": 1.3604,
            "K3": -0.1027,
            "F_m": 3.5,
            "F_m_note": "Nickel alloys/Monel for high-pressure oxygen service",
            "F_BM": 2.15
          },
          "calculation_steps": [
            "1. Verify: 450 \u2264 W_shaft \u2264 3000 kW",
            "2. Base cost: log10(Cp0) = 2.2891 + 1.3604\u00d7log10(W_shaft) - 0.1027\u00d7[log10(W_shaft)]\u00b2",
            "3. Convert: Cp0 = 10^(log10(Cp0))",
            "4. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "5. Apply: CAPEX = Cp_2026 \u00d7 2.15 \u00d7 3.5 = Cp_2026 \u00d7 7.525"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * F_BM * F_m",
          "C_BM": 359987.35,
          "cost_breakdown": "Sum of 2 units: $179,994 + $179,994"
        },
        {
          "tag": "MTC-4 / MTC-5",
          "topology_id": "SOEC_H2_Compressor_S1 through SOEC_H2_Compressor_S6",
          "name": "Multistage Reciprocating Hydrogen Compressor",
          "component_type": "Reciprocating Compressor",
          "process_description": "Six-stage H2 compression train (pressure ratio ~2.0x per stage) from ~1 bar to 41 bar. Compresses BEFORE purification (legacy design). Each stage temperature-limited to 135\u00b0C with intercooling.",
          "power_consumption": {
            "stage_1": {
              "reference": "SOEC_H2_Compressor_S1",
              "value_w": 194096.79,
              "value_kw": 194.1
            },
            "stage_2": {
              "reference": "SOEC_H2_Compressor_S2",
              "value_w": 135367.95,
              "value_kw": 135.4
            },
            "stage_3": {
              "reference": "SOEC_H2_Compressor_S3",
              "value_w": 135594.28,
              "value_kw": 135.6
            },
            "stage_4": {
              "reference": "SOEC_H2_Compressor_S4",
              "value_w": 135833.48,
              "value_kw": 135.8
            },
            "stage_5": {
              "reference": "SOEC_H2_Compressor_S5",
              "value_w": 134124.2,
              "value_kw": 134.1
            },
            "stage_6": {
              "reference": "SOEC_H2_Compressor_S6",
              "value_w": 133727.61,
              "value_kw": 133.7
            },
            "total_kw": 868.7
          },
          "capacity_variable": {
            "symbol": "W_shaft",
            "description": "Shaft power",
            "unit": "kW",
            "valid_range": "10 - 10000 kW",
            "design_capacity": 868.74
          },
          "coefficients": {
            "K1": 2.0309,
            "K2": 1.2524,
            "K3": -0.0638,
            "F_m": 2.1,
            "F_m_note": "Alloy Steel/Stainless for H2",
            "F_BM": 2.15
          },
          "calculation_steps": [
            "1. Verify: 10 \u2264 W_shaft \u2264 10000 kW",
            "2. Base cost: log10(Cp0) = 2.0309 + 1.2524\u00d7log10(W_shaft) - 0.0638\u00d7[log10(W_shaft)]\u00b2",
            "3. Convert: Cp0 = 10^(log10(Cp0))",
            "4. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "5. Apply: CAPEX = Cp_2026 \u00d7 2.15 \u00d7 2.1 = Cp_2026 \u00d7 4.515"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * F_BM * F_m",
          "C_BM": 1449869.01,
          "cost_breakdown": "Sum of 2 units: $724,935 + $724,935"
        },
        {
          "tag": "CYC-1 to CYC-5",
          "topology_id": "SOEC_H2_Cyclone_1 through SOEC_H2_Cyclone_6",
          "name": "Cyclone Separator (Multi-Cyclone Assembly)",
          "component_type": "Centrifugal Gas-Liquid Separator",
          "process_description": "Six cyclones in H2 train: (1) Post-KOD2, (2-5) Post-intercoolers S2-S5, (6) Post-Deoxo. Remove fine water mist via centrifugal force. Element diameter 50mm, vane angle 45\u00b0, target velocity 22 m/s.",
          "power_consumption": {
            "note": "Passive separation devices - pressure drop only (~0.05-0.1 bar per unit)"
          },
          "description": "Combined vessel + internals calculation",
          "vessel": {
            "coefficients": {
              "K1": 3.4974,
              "K2": 0.4485,
              "K3": 0.1074,
              "F_m": 1.0
            }
          },
          "internals": {
            "coefficients": {
              "K1": 3.349,
              "K2": 0.4669,
              "K3": 0.1243
            }
          },
          "calculation_steps": [
            "VESSEL: Use vertical vessel correlation with F_p based on pressure",
            "INTERNALS: Use cross-sectional area correlation",
            "TOTAL: Cp_2026_total = (Cp_2026_vessel \u00d7 F_p \u00d7 F_m_vessel) + (Cp_2026_internals \u00d7 F_m_internals)",
            "Note: F_m depends on service (2.41 for SS316, adjust for material)"
          ],
          "cost_formula": "Compound: Shell(V=0.68m3) + Internals(A=0.28m2); N=5",
          "C_BM": 41413.5,
          "capacity_variable": {
            "unit": "m3"
          },
          "cost_breakdown": "Compound Calculation: (5,495 [Shell] + 2,788 [Int]) x 5 units = 41,414"
        },
        {
          "tag": "PMP-1",
          "topology_id": "SOEC_Feed_Pump",
          "name": "Centrifugal Water Pump + Motor (SS316)",
          "component_type": "Centrifugal Pump with Electric Motor",
          "process_description": "Pressurizes bypass water (7.88% of total flow) from 1 bar to 5.5 bar before attemperator injection. Handles liquid water at ~25\u00b0C.",
          "power_consumption": {
            "reference": "SOEC_Feed_Pump",
            "value_w": 71.29,
            "value_kw": 0.071,
            "note": "Low power due to small flow fraction (bypass only)"
          },
          "capacity_variable": {
            "symbol": "W_pump",
            "description": "Pump power",
            "unit": "kW",
            "valid_range": "1 - 100 kW",
            "design_capacity": 0.07
          },
          "coefficients": {
            "pump": {
              "K1": 3.3892,
              "K2": 0.0536,
              "K3": 0.1538,
              "F_m": 2.41,
              "F_m_note": "SS316 material",
              "F_p": 1.0,
              "B1": 1.89,
              "B2": 1.35
            },
            "motor": {
              "K1_low": 3.3432,
              "K2_low": 0.2761,
              "K3_low": 0.0543,
              "threshold": "75 kW"
            }
          },
          "calculation_steps": [
            "PUMP: CAPEX_pump = Cp_2026_pump \u00d7 [1.89 + (1.35 \u00d7 2.1 \u00d7 1.0)] = Cp_2026 \u00d7 4.725",
            "MOTOR: CAPEX_motor = Cp_2026_motor (F_BM = 1.0)",
            "TOTAL: CAPEX = CAPEX_pump + CAPEX_motor"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0",
          "C_BM": 36200.9
        },
        {
          "tag": "PMP-2 / PMP-3 / PMP-4",
          "topology_id": "SOEC_DRAIN_PUMP_1 / SOEC_DRAIN_PUMP_2 / SOEC_Drain_Pump",
          "name": "Centrifugal Water Pump + Motor (CORRECTED F_m)",
          "component_type": "Centrifugal Pump with Electric Motor",
          "process_description": "Drain recovery pumps: (1) DRAIN_PUMP_1 from KOD_1, (2) DRAIN_PUMP_2 from KOD_2, (3) Drain_Pump (main) returns combined drains to heat recovery loop. Pressurize to 1.2 bar for circulation.",
          "power_consumption": {
            "drain_pump_1": {
              "reference": "SOEC_DRAIN_PUMP_1",
              "value_w": 2.2,
              "value_kw": 0.002
            },
            "drain_pump_2": {
              "reference": "SOEC_DRAIN_PUMP_2",
              "value_w": 2.49,
              "value_kw": 0.002
            },
            "main_drain": {
              "reference": "SOEC_Drain_Pump",
              "value_w": 36.6,
              "value_kw": 0.037
            },
            "note": "Low power due to minimal head and flow (gravity-assisted drainage)"
          },
          "capacity_variable": {
            "symbol": "W_pump",
            "description": "Pump power",
            "unit": "kW",
            "valid_range": "1 - 100 kW",
            "design_capacity": 0.05
          },
          "coefficients": {
            "pump": {
              "K1": 3.3892,
              "K2": 0.0536,
              "K3": 0.1538,
              "F_m": 2.41,
              "F_m_note": "SS316 material - CORRECTED from 2.1",
              "F_p": 1.0,
              "B1": 1.89,
              "B2": 1.35
            },
            "motor": {
              "K1_low": 3.3432,
              "K2_low": 0.2761,
              "K3_low": 0.0543,
              "threshold": "75 kW"
            }
          },
          "calculation_steps": [
            "PUMP: CAPEX_pump = Cp_2026_pump \u00d7 [1.89 + (1.35 \u00d7 2.41 \u00d7 1.0)] = Cp_2026 \u00d7 5.144",
            "MOTOR: CAPEX_motor = Cp_2026_motor",
            "TOTAL: CAPEX = CAPEX_pump + CAPEX_motor",
            "NOTE: F_m = 2.41 per source file, not 2.1"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0",
          "C_BM": 192125.27,
          "cost_breakdown": "Sum of 3 units: $64,042 + $64,042 + $64,042"
        },
        {
          "tag": "RCT-2",
          "topology_id": "Integrated with SOEC_Cluster",
          "name": "Rectifier and Transformer (SOEC Power Conditioning)",
          "component_type": "AC-DC Power Conversion System",
          "process_description": "Converts grid AC to DC for SOEC stack operation. Rated for 11.52 MW (6 modules \u00d7 2.4 MW). Includes transformer, rectifier bridge, and filtering. Integrated into SOEC cost calculation.",
          "power_consumption": {
            "note": "Power losses ~2-3% of throughput (included in SOEC_System total)"
          },
          "capacity_variable": {
            "symbol": "P_rated",
            "description": "Rated power",
            "unit": "kW"
          },
          "coefficients": {
            "Cp0_per_kW": 200,
            "B1": 1.28,
            "B2": 0.87,
            "F_m": 1.0,
            "F_p": 1.0
          },
          "calculation_steps": [
            "1. Base cost: Cp0 = P_rated \u00d7 200 USD/kW",
            "2. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "3. Bare module: CAPEX = Cp_2026 \u00d7 [1.28 + (0.87 \u00d7 1.0 \u00d7 1.0)] = Cp_2026 \u00d7 2.15"
          ],
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation"
        }
      ]
    },
    "PEM": {
      "description": "Proton Exchange Membrane electrolyzer - Secondary hydrogen production pathway",
      "process_location": "Parallel to SOEC - operates at 40 bar, ambient temperature",
      "components": [
        {
          "tag": "PEM_Electrolyzer",
          "topology_id": "PEM_Electrolyzer",
          "name": "PEM Stack Module",
          "component_type": "Proton Exchange Membrane Electrolyzer",
          "process_description": "Low-temperature (60-80\u00b0C) water electrolysis at 40 bar. Produces wet H2 and O2 directly at pressure. Simpler thermal management than SOEC but higher electrical consumption per kg H2.",
          "power_consumption": {
            "reference": "PEM_System",
            "value_w": 5000243.45,
            "value_mw": 5.0,
            "note": "Rated capacity - higher specific consumption than SOEC (~50 kWh/kg vs 40 kWh/kg)"
          },
          "method": "NO DATA IN SOURCE FILE - External estimate used",
          "cost_basis": {
            "specific_cost_usd": 1100,
            "unit": "USD/kW",
            "year": "2026 projection",
            "source_note": "NOT in provided CSV file - this is an external estimate/assumption",
            "csv_status": "Empty row in source file"
          },
          "calculation_steps": [
            "WARNING: Source CSV has no coefficients or costs for PEM Electrolyzer",
            "1. Determine PEM capacity: P_total (kW)",
            "2. Apply estimate: CAPEX = P_total \u00d7 1100 USD/kW",
            "3. NO CEPCI adjustment (estimate is 2026)",
            "Note: This value (1100 USD/kW) is NOT from the source file"
          ],
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "kW"
          }
        },
        {
          "tag": "RCT-1",
          "topology_id": "Integrated with PEM_Electrolyzer",
          "name": "Rectifier and Transformer (PEM)",
          "component_type": "AC-DC Power Conversion System",
          "process_description": "Converts grid AC to DC for PEM stack operation. Rated for 5.0 MW. Similar to SOEC rectifier but sized for PEM capacity.",
          "power_consumption": {
            "note": "Power losses ~2-3% of throughput (included in PEM_System total)"
          },
          "capacity_variable": {
            "symbol": "P_rated",
            "description": "Rated power",
            "unit": "kW"
          },
          "coefficients": {
            "Cp0_per_kW": 200,
            "B1": 1.28,
            "B2": 0.87,
            "F_m": 1.0,
            "F_p": 1.0
          },
          "calculation_steps": [
            "1. Base cost: Cp0 = P_rated \u00d7 200 USD/kW",
            "2. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "3. Apply: CAPEX = Cp_2026 \u00d7 2.15"
          ],
          "cost_formula": "C_BM=Cp0 * [B_1 + (B_2 * F_m * F_p)]",
          "C_BM": null,
          "error": "Missing parameters for calculation"
        },
        {
          "tag": "KOT-1 / KOT-2",
          "topology_id": "PEM_H2_KOD_1 / PEM_O2_KOD_1",
          "name": "Knock-Out Drum Separator",
          "component_type": "Vertical Pressure Vessel with Demister Pad",
          "process_description": "Primary water knockout for PEM: (1) H2_KOD_1 after electrolyzer, (2) O2_KOD_1 for oxygen stream. Operate at 40 bar (higher F_p than SOEC). Remove entrained liquid before drying.",
          "power_consumption": {
            "note": "Passive separation devices - no electrical consumption"
          },
          "coefficients": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 3.1,
            "F_p": 1.0,
            "B1": 2.25,
            "B2": 1.82,
            "F_multi": 1.2
          },
          "calculation_steps": [
            "Same as SOEC KOT calculation"
          ],
          "cost_formula": "Turton (V derived); Source: Topology (PEM_O2_KOD_1)",
          "C_BM": 84228.2,
          "capacity_variable": {
            "unit": "m3"
          },
          "cost_breakdown": "Sum: KOT-1: $42,114 (V=0.39m3 via Topology (PEM_H2_KOD_1)) + KOT-2: $42,114 (V=0.39m3 via Topology (PEM_O2_KOD_1))",
          "note": "Capacity based on Topology (PEM_O2_KOD_1)"
        },
        {
          "tag": "COA-1 / COA-2 / COA-3 / COA-4",
          "topology_id": "PEM_H2_Coalescer_1 / PEM_H2_Coalescer_2 / PEM_O2_Coalescer_1",
          "name": "Coalescer",
          "component_type": "Fiber-Bed Coalescing Filter",
          "process_description": "Fine mist removal after cyclones: (1-2) H2 coalescers before/after Deoxo, (3) O2 coalescer. Capture sub-micron droplets missed by cyclones. Critical for PSA adsorbent protection.",
          "power_consumption": {
            "note": "Passive separation - pressure drop ~0.2-0.5 bar through fiber bed"
          },
          "coefficients": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 2.1,
            "F_p": 1.38,
            "F_p_note": "Reference pressure 15 bar",
            "B1": 2.25,
            "B2": 1.82,
            "F_coalescer": 1.25
          },
          "calculation_steps": [
            "1. Base cost: log10(Cp0) = 3.4974 + 0.4485\u00d7log10(V) + 0.1074\u00d7[log10(V)]\u00b2",
            "2. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "3. Bare module: C_BM = Cp_2026 \u00d7 [2.25 + (1.82 \u00d7 2.1 \u00d7 1.38)]",
            "4. Apply coalescer: CAPEX = C_BM \u00d7 1.25"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM= 1,25 * {Cp0 * [B_1 + (B_2 * F_m * F_p)]}",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {}
        },
        {
          "tag": "HTR-1 / HTR-2 / HTR-3",
          "topology_id": "PEM_H2_ElectricBoiler_1 / PEM_H2_ElectricBoiler_2 / PEM_O2_ElectricBoiler",
          "name": "Electrical Heater",
          "component_type": "Electrical Resistance Heater",
          "process_description": "Temperature control heaters: (1) H2_Boiler_1 pre-Deoxo (40\u00b0C), (2) H2_Boiler_2 pre-PSA (30\u00b0C), (3) O2_Boiler (20\u00b0C). Prevent condensation and maintain optimal operating temperatures.",
          "power_consumption": {
            "note": "Typically 25 kW each - see topology max_power_kw parameters"
          },
          "coefficients": {
            "K1": 2.858,
            "K2": 0.8209,
            "K3": 0.0075,
            "F_BM": 1.3
          },
          "calculation_steps": [
            "Same as SOEC electric heaters"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0 * F_BM",
          "C_BM": 33600.77,
          "capacity_variable": {
            "unit": "kW",
            "design_capacity": 25.0
          },
          "note": "Design capacity fetched from Topology Node PEM_H2_ElectricBoiler_1",
          "cost_breakdown": "Sum of 3 units: $11,200 + $11,200 + $11,200"
        },
        {
          "tag": "CHL-1 / CHL-2 / CHL-3",
          "topology_id": "PEM_H2_Chiller_1 / PEM_H2_Chiller_2 / PEM_O2_Chiller_1",
          "name": "Electrical Chiller",
          "component_type": "Vapor Compression Refrigeration Unit",
          "process_description": "Deep cooling for water removal: (1) H2_Chiller_1 post-DryCooler (4\u00b0C, 500 kW), (2) H2_Chiller_2 post-Deoxo (4\u00b0C), (3) O2_Chiller_1 (4\u00b0C, 8 MW - upsized for bulk cooling). Target dewpoint depression.",
          "power_consumption": {
            "h2_chiller_1": {
              "reference": "PEM_H2_Chiller_1_electrical",
              "value_w": 2022.75,
              "value_kw": 2.0,
              "note": "COP ~4.0"
            },
            "h2_chiller_2": {
              "reference": "PEM_H2_Chiller_2_electrical",
              "value_w": 2843.25,
              "value_kw": 2.8
            },
            "o2_chiller_1": {
              "reference": "PEM_O2_Chiller_1_electrical",
              "value_w": 1127.05,
              "value_kw": 1.1
            }
          },
          "coefficients": {
            "K1": 2.858,
            "K2": 0.8209,
            "K3": 0.0075,
            "F_BM": 1.3
          },
          "calculation_steps": [
            "Same as electric heaters (same correlation)"
          ],
          "cost_formula": "Sum of Turton costs per sub-unit",
          "C_BM": 10043.92,
          "capacity_variable": {
            "unit": "kW"
          },
          "cost_breakdown": "Sum of 3 sub-units: h2_chiller_1 (2.0kW): $3,426 + h2_chiller_2 (2.8kW): $4,524 + o2_chiller_1 (1.1kW): $2,094",
          "note": "Cost calculated by summing 3 sub-units found in power_consumption block"
        },
        {
          "tag": "DCL-1 / DCL-2",
          "topology_id": "PEM_H2_DryCooler_1 / PEM_O2_Drycooler_1",
          "name": "Dry Cooler Heat Exchangers (PEM)",
          "component_type": "Air-Cooled Heat Exchanger (Dry Cooler)",
          "process_description": "Primary cooling for PEM outlet streams: (1) H2_DryCooler_1 after KOD, (2) O2_DryCooler_1. Cool from electrolyzer temperature (~60-80\u00b0C) to 30\u00b0C before chilling.",
          "power_consumption": {
            "h2_drycooler": {
              "reference": "PEM_H2_DryCooler_1_fan",
              "value_w": 17610.2,
              "value_kw": 17.6
            },
            "o2_drycooler": {
              "reference": "PEM_O2_Drycooler_1_fan",
              "value_w": 2416.33,
              "value_kw": 2.4
            }
          },
          "note": "Cost calculated by summing 2 sub-units found in power_consumption block",
          "coefficients": {
            "K1": 4.3247,
            "K2": -0.303,
            "K3": 0.1634,
            "F_m": 2.7,
            "F_p": 1.092,
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Sum of Turton costs per sub-unit",
          "C_BM": 444475.17,
          "capacity_variable": {
            "unit": "m2"
          },
          "cost_breakdown": "Sum of 2 sub-units: h2_drycooler (17.6kW): $213,971 + o2_drycooler (2.4kW): $230,504"
        },
        {
          "tag": "VAL-1 / VAL-2 / VAL-3",
          "topology_id": "PEM_Water_Return_Valve_1 / PEM_Water_Return_Valve_2 / PEM_O2_Valve",
          "name": "Control Pressure Valve (Ball)",
          "component_type": "Ball Valve with Actuator",
          "process_description": "Pressure control valves: (1-2) Water return valves (40\u21924 bar for degassing), (3) O2_Valve (40\u219215 bar for header). Throttle pressure while maintaining flow control.",
          "power_consumption": {
            "note": "Actuator power negligible (~50-100 W per valve during operation)"
          },
          "capacity_variable": {
            "symbol": "A",
            "description": "Nominal diameter (inches) or Cv",
            "unit": "inches or Cv"
          },
          "coefficients": {
            "K1": 3.1491,
            "K2": 0.5841,
            "K3": -0.0125,
            "F_m": 2.4,
            "F_p": 1.0,
            "B1": 1.63,
            "B2": 1.66
          },
          "calculation_steps": [
            "1. Base cost: log10(Cp0) = 3.1491 + 0.5841\u00d7log10(A) - 0.0125\u00d7[log10(A)]\u00b2",
            "2. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "3. Apply: CAPEX = Cp_2026 \u00d7 [1.63 + (1.66 \u00d7 2.4 \u00d7 1.0)] = Cp_2026 \u00d7 5.614"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM= {Cp0 * [B_1 + (B_2 * F_m * F_p)]}",
          "C_BM": null,
          "error": "Missing parameters for calculation"
        },
        {
          "tag": "DGS-1 / DGS-2",
          "topology_id": "PEM_Degasser_1 / PEM_Degasser_2",
          "name": "Degasser Tank",
          "component_type": "Atmospheric Separation Tank",
          "process_description": "Gas-water separation tanks: (1) Degasser_1 for H2 drain water (2 m\u00b3), (2) Degasser_2 for O2 drain water (1 m\u00b3). Allow dissolved gases to escape before water recirculation. Atmospheric pressure, passive separation.",
          "power_consumption": {
            "note": "Passive gravity separation - no electrical consumption"
          },
          "coefficients": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_BM": 1.0
          },
          "calculation_steps": [
            "1. Base cost: log10(Cp0) = 3.4974 + 0.4485\u00d7log10(V) + 0.1074\u00d7[log10(V)]\u00b2",
            "2. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "3. Simple module: CAPEX = Cp_2026 \u00d7 1.0"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM=Cp0",
          "C_BM": 12985.4,
          "capacity_variable": {
            "unit": "m3",
            "design_capacity": 2.0
          },
          "note": "Design capacity fetched from Topology Node PEM_Degasser_1",
          "cost_breakdown": "Sum of 2 units: $6,493 + $6,493"
        },
        {
          "tag": "MCY-1",
          "topology_id": "PEM_O2_Cyclone_1",
          "name": "Multicyclone Separator (O2)",
          "component_type": "Multi-Element Centrifugal Separator",
          "process_description": "Oxygen-service multicyclone after O2_Chiller. Special materials (F_m=3.0) required for oxygen-enriched environment. Higher pressure (40 bar, F_p=2.27). Element diameter 50mm, vane angle 45\u00b0.",
          "power_consumption": {
            "note": "Passive separation - pressure drop ~0.1 bar"
          },
          "vessel": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 3.0,
            "F_m_note": "Oxygen requires special materials + production methods",
            "F_p": 2.27,
            "F_p_note": "Reference pressure 40 bar"
          },
          "internals": {
            "K1": 3.349,
            "K2": 0.4669,
            "K3": 0.1243
          },
          "calculation_steps": [
            "VESSEL: Cost_vessel = Cp_2026_vessel \u00d7 F_p \u00d7 F_m",
            "INTERNALS: Cost_internals = Cp_2026_int \u00d7 F_m",
            "TOTAL: CAPEX = Cost_vessel + Cost_internals"
          ],
          "cost_formula": "Compound: Shell(V=0.68m3) + Internals(A=0.28m2); N=1",
          "C_BM": 8282.7,
          "capacity_variable": {
            "unit": "m3"
          },
          "cost_breakdown": "Compound Calculation: (5,495 [Shell] + 2,788 [Int]) x 1 units = 8,283"
        },
        {
          "tag": "MCY-2",
          "topology_id": "PEM_H2_Cyclone_1",
          "name": "Multicyclone Separator (H2)",
          "component_type": "Multi-Element Centrifugal Separator",
          "process_description": "Hydrogen-service multicyclone after H2_Chiller_1. SS316 construction (F_m=2.1). Operates at 40 bar (F_p=2.27). First-stage mist removal before coalescer.",
          "power_consumption": {
            "note": "Passive separation - pressure drop ~0.1 bar"
          },
          "vessel": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 2.1,
            "F_m_note": "SS316 for hydrogen",
            "F_p": 2.27,
            "F_p_note": "Reference pressure 40 bar"
          },
          "internals": {
            "K1": 3.349,
            "K2": 0.4669,
            "K3": 0.1243
          },
          "cost_formula": "Compound: Shell(V=0.68m3) + Internals(A=0.28m2); N=1",
          "C_BM": 8282.7,
          "capacity_variable": {
            "unit": "m3"
          },
          "cost_breakdown": "Compound Calculation: (5,495 [Shell] + 2,788 [Int]) x 1 units = 8,283"
        },
        {
          "tag": "MCY-4",
          "topology_id": "PEM_H2_Cyclone_2",
          "name": "Multicyclone Separator (H2)",
          "component_type": "Multi-Element Centrifugal Separator",
          "process_description": "Second H2 cyclone after H2_Chiller_2 (post-Deoxo). CORRECTED F_m=2.41 per source file (not 2.1). Lower pressure (F_p=1.0) than MCY-2. Final mist removal before PSA.",
          "power_consumption": {
            "note": "Passive separation - pressure drop ~0.1 bar"
          },
          "vessel": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 2.41,
            "F_m_note": "SS316 - CORRECTED from 2.1 per source file",
            "F_p": 1.0,
            "F_p_note": "Low pressure"
          },
          "internals": {
            "K1": 3.349,
            "K2": 0.4669,
            "K3": 0.1243
          },
          "note": "MCY-4 has different F_m (2.41) than MCY-2 (2.1) per source file",
          "cost_formula": "Compound: Shell(V=0.68m3) + Internals(A=0.28m2); N=1",
          "C_BM": 8282.7,
          "capacity_variable": {
            "unit": "m3"
          },
          "cost_breakdown": "Compound Calculation: (5,495 [Shell] + 2,788 [Int]) x 1 units = 8,283"
        },
        {
          "tag": "PSA-1 / PSA-2",
          "topology_id": "PEM_H2_PSA_1",
          "name": "Pressure Swing Adsorption Unit",
          "component_type": "Adsorbent Bed Gas Purification System",
          "process_description": "H2 purification to 99.995% purity. Removes O2, N2, H2O traces via zeolite/carbon molecular sieve adsorbents. Operates at 40 bar, 90% H2 recovery. Cycle time ~10 min (adsorption-desorption).",
          "power_consumption": {
            "note": "Valve actuation and instrumentation - typically 10-25 kW per unit"
          },
          "coefficients": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 3.1,
            "F_p": 1.2,
            "F_p_note": "Reference pressure 15 bar",
            "B1": 2.25,
            "B2": 1.82,
            "F_PSA": 1.25
          },
          "calculation_steps": [
            "1. Base cost: log10(Cp0) = 3.4974 + 0.4485\u00d7log10(V) + 0.1074\u00d7[log10(V)]\u00b2",
            "2. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "3. Bare module: C_BM = Cp_2026 \u00d7 [2.25 + (1.82 \u00d7 3.1 \u00d7 1.2)]",
            "4. Apply PSA: CAPEX = C_BM \u00d7 1.25"
          ],
          "cost_formula": "Base: log10(Cp0) = K1 + K2*log10(A) + K3*[log10(A)]^2; Module: C_BM= 1,25 * {Cp0 * [B_1 + (B_2 * F_m * F_p)]}",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {}
        }
      ]
    },
    "ATR": {
      "description": "Autothermal Reformer with Water-Gas Shift reactors - Biogas-to-hydrogen pathway",
      "process_location": "Parallel to SOEC/PEM - converts biogas (60% CH4) to syngas, then purifies H2",
      "components": [
        {
          "tag": "ATR_Reactor",
          "topology_id": "ATR_Plant (IntegratedATRPlant - Black Box)",
          "name": "Autothermal Reforming Reactor",
          "component_type": "Catalytic Partial Oxidation Reactor",
          "process_description": "Converts biogas + O2 + steam to syngas (H2 + CO) at 950-1100\u00b0C. Partial oxidation provides heat for endothermic reforming. Nickel-based catalyst. Integrated model includes HT-WGS and LT-WGS reactors.",
          "power_consumption": {
            "note": "Exothermic operation - no direct electrical consumption. Associated equipment power listed separately."
          },
          "method": "Direct reliable source cost",
          "cost_basis": {
            "value_eur": 3118657.28,
            "value_usd": 3430523.01,
            "source": "Carlini, G. (2025)",
            "year": "2026"
          },
          "calculation_steps": [
            "1. Use direct cost: \u20ac3,118,657.28",
            "2. Convert: \u20ac3,118,657.28 \u00d7 1.1 = $3,430,523.01",
            "3. NO CEPCI adjustment (already 2026)"
          ],
          "cost_formula": "Formula not found",
          "C_BM": 3430523.01
        },
        {
          "tag": "HT-WGS",
          "topology_id": "Integrated in ATR_Plant",
          "name": "High-Temperature Water-Gas Shift Reactor",
          "component_type": "Catalytic Shift Reactor",
          "process_description": "First WGS stage: CO + H2O \u2192 CO2 + H2 at 350-450\u00b0C. Iron-chromium oxide catalyst. Converts bulk CO from ATR. Integrated into ATR_Plant black box model.",
          "power_consumption": {
            "note": "Exothermic reaction - no electrical consumption"
          },
          "cost_basis": {
            "value_eur": 436612.02,
            "value_usd": 480273.22,
            "source": "Carlini, G. (2025)",
            "year": "2026"
          },
          "calculation_steps": [
            "1. Direct cost: \u20ac436,612.02 \u00d7 1.1 = $480,273.22",
            "2. NO CEPCI adjustment"
          ],
          "cost_formula": "Fixed Cost (Source: Direct/External)",
          "C_BM": 480273.22
        },
        {
          "tag": "LT-WGS",
          "topology_id": "Integrated in ATR_Plant",
          "name": "Low-Temperature Water-Gas Shift Reactor",
          "component_type": "Catalytic Shift Reactor",
          "process_description": "Second WGS stage: CO + H2O \u2192 CO2 + H2 at 200-250\u00b0C. Copper-zinc oxide catalyst. Polishes remaining CO to <1%. Integrated into ATR_Plant black box model.",
          "power_consumption": {
            "note": "Exothermic reaction - no electrical consumption"
          },
          "cost_basis": {
            "value_eur": 436612.02,
            "value_usd": 480273.22,
            "source": "Carlini, G. (2025)",
            "year": "2026"
          },
          "calculation_steps": [
            "Same as HT-WGS"
          ],
          "cost_formula": "Fixed Cost (Source: Direct/External)",
          "C_BM": 480273.22
        },
        {
          "tag": "Inter-reactor Heat Exchanges",
          "topology_id": "ATR_Syngas_Cooler",
          "name": "Inter-Reactor Heat Exchangers",
          "component_type": "Shell & Tube Heat Exchanger (Interchanger)",
          "process_description": "Heat recovery from hot syngas (~800\u00b0C post-ATR). Preheats SOEC makeup water from ~25\u00b0C to 105\u00b0C. Min approach temp 10K, efficiency 95%.",
          "power_consumption": {
            "note": "Passive heat transfer - no electrical consumption"
          },
          "cost_basis": {
            "value_eur": 399110.85,
            "value_usd": 439021.94,
            "source": "Carlini, G. (2025)",
            "year": "2026"
          },
          "alternative_calculation": {
            "note": "If area-based calculation needed (not in source)",
            "coefficients": {
              "K1": 4.8306,
              "K2": -0.8509,
              "K3": 0.3187,
              "F_m": 1.0,
              "F_p": 1.0,
              "B1": 1.63,
              "B2": 1.66
            }
          },
          "cost_formula": "Fixed Cost (Source: Direct/External)",
          "C_BM": 439021.94
        },
        {
          "tag": "Biogas_Compressor",
          "topology_id": "Biogas_Compressor_1 / Biogas_Compressor_2 / Biogas_Compressor_3",
          "name": "Multistage Alternating Compressor for Biogas",
          "component_type": "Reciprocating Compressor (3 Stages)",
          "process_description": "Three-stage biogas compression: 3.0\u21925.2\u21928.8\u219215.0 bar. Each stage intercooled. Handles CH4/CO2 mixture with moisture. F_m=1.5 minimum for corrosion resistance.",
          "power_consumption": {
            "stage_1": {
              "reference": "Biogas_Compressor_1",
              "value_w": 54513.86,
              "value_kw": 54.5
            },
            "stage_2": {
              "reference": "Biogas_Compressor_2",
              "value_w": 53647.22,
              "value_kw": 53.6
            },
            "stage_3": {
              "reference": "Biogas_Compressor_3",
              "value_w": 59455.19,
              "value_kw": 59.5
            },
            "total_kw": 167.6,
            "intercooler_fans": {
              "reference": "Biogas_Intercooler_X_fan (X=1,2)",
              "value_w_each": 2416.33,
              "value_kw_total": 4.8
            }
          },
          "capacity_variable": {
            "symbol": "W_shaft",
            "description": "Shaft power",
            "unit": "kW",
            "valid_range": "10 - 10000 kW",
            "design_capacity": 167.62
          },
          "coefficients": {
            "K1": 2.03,
            "K2": 1.25,
            "K3": -0.06,
            "F_m": 1.5,
            "F_m_note": "Minimum F_m=1.5 for biogas (moisture/contaminant risk)",
            "F_BM": 2.15
          },
          "calculation_steps": [
            "1. Base cost: log10(Cp0) = 2.03 + 1.25\u00d7log10(W_shaft) - 0.06\u00d7[log10(W_shaft)]\u00b2",
            "2. Inflate: Cp_2026 = Cp0 \u00d7 2.0655",
            "3. Apply: CAPEX = Cp_2026 \u00d7 2.15 \u00d7 1.5 = Cp_2026 \u00d7 3.225"
          ],
          "cost_formula": "Formula not found",
          "C_BM": 217322.5
        },
        {
          "tag": "ATR_H2_Compressor",
          "topology_id": "ATR_H2_Compressor_1 / ATR_H2_Compressor_2",
          "name": "Multistage Alternating Hydrogen Compressor",
          "component_type": "Reciprocating Compressor (2 Stages)",
          "process_description": "Two-stage compression of purified H2 from PSA: 15\u219239.64\u219239.64 bar (note: second stage likely typo, should be ~70 bar). Brings ATR H2 to common header pressure.",
          "power_consumption": {
            "stage_1": {
              "reference": "ATR_H2_Compressor_1",
              "value_w": 120894.15,
              "value_kw": 120.9
            },
            "stage_2": {
              "reference": "ATR_H2_Compressor_2",
              "value_w": 38535.42,
              "value_kw": 38.5
            },
            "total_kw": 159.4,
            "drycooler": {
              "reference": "ATR_H2_Drycooler_fan",
              "value_w": 17610.2,
              "value_kw": 17.6
            }
          },
          "coefficients": {
            "K1": 2.0309,
            "K2": 1.2524,
            "K3": -0.0638,
            "F_m": 2.1,
            "F_BM": 2.15
          },
          "calculation_steps": [
            "Same as SOEC MTC-4/MTC-5"
          ],
          "cost_formula": "Sum of Turton costs per sub-unit",
          "C_BM": 310659.19,
          "capacity_variable": {
            "unit": "kW"
          },
          "cost_breakdown": "Sum of 3 sub-units: stage_1 (120.9kW): $214,756 + stage_2 (38.5kW): $66,964 + drycooler (17.6kW): $28,939",
          "note": "Cost calculated by summing 3 sub-units found in power_consumption block"
        },
        {
          "tag": "Dry Cooler - Compressor Biogas (Fixed tube)",
          "topology_id": "Biogas_Intercooler_1 / Biogas_Intercooler_2",
          "name": "Dry Cooler - Biogas Compressor (Shell & Tube, Fixed)",
          "component_type": "Air-Cooled Heat Exchanger",
          "process_description": "Two intercoolers for biogas compression train. Cool from ~120\u00b0C (compression limit) to 30-62\u00b0C. Prevent thermal degradation of biogas components.",
          "power_consumption": {
            "per_unit": {
              "reference": "Biogas_Intercooler_X_fan",
              "value_w": 2416.33,
              "value_kw": 2.4
            }
          },
          "coefficients": {
            "K1": 4.3247,
            "K2": -0.303,
            "K3": 0.1634,
            "F_m": 2.1,
            "F_p": 1.0,
            "F_p_note": "Base case, no specific F_p in source",
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Sum of Turton costs per sub-unit",
          "C_BM": 180747.76,
          "capacity_variable": {
            "unit": "kW"
          },
          "cost_breakdown": "Sum of 1 sub-units: per_unit (2.4kW): $180,748",
          "note": "Cost calculated by summing 1 sub-units found in power_consumption block"
        },
        {
          "tag": "Dry Cooler - Compressor Biogas (with F_p)",
          "topology_id": "Alternative specification (not explicitly used)",
          "name": "Dry Cooler - Biogas Compressor (alternate spec)",
          "component_type": "Air-Cooled Heat Exchanger",
          "process_description": "Alternate specification with F_p=1.08 for specific operating conditions.",
          "coefficients": {
            "K1": 4.3247,
            "K2": -0.303,
            "K3": 0.1634,
            "F_m": 2.1,
            "F_p": 1.08,
            "F_p_note": "Specific F_p from source file",
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "kW"
          }
        },
        {
          "tag": "Dry Cooler - pr\u00e9 HTWGS (Floating Head)",
          "topology_id": "Integrated in ATR_Plant heat management",
          "name": "Pre-HTWGS Dry Cooler (Floating Head)",
          "component_type": "Shell & Tube Heat Exchanger (Floating Head)",
          "process_description": "Cools syngas before HT-WGS reactor to optimal catalyst temperature (350-450\u00b0C). Floating head design accommodates thermal expansion. Integrated into ATR_Plant model.",
          "coefficients": {
            "K1": 4.8229,
            "K2": -0.0922,
            "K3": 0.1258,
            "F_m": 2.1,
            "F_p": 1.04,
            "F_p_note": "Specific to pre-HTWGS location",
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "m2"
          }
        },
        {
          "tag": "Dry Cooler - pr\u00e9 LTWGS (Floating Head)",
          "topology_id": "Integrated in ATR_Plant heat management",
          "name": "Pre-LTWGS Dry Cooler",
          "component_type": "Shell & Tube Heat Exchanger (Floating Head)",
          "process_description": "Cools syngas before LT-WGS reactor to optimal catalyst temperature (200-250\u00b0C). Floating head design for thermal cycling. Integrated into ATR_Plant model.",
          "coefficients": {
            "K1": 4.8229,
            "K2": -0.0922,
            "K3": 0.1258,
            "F_m": 2.1,
            "F_p": 1.04,
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "m2"
          }
        },
        {
          "tag": "Dry Cooler - p\u00f3s LTWGS (Floating Head)",
          "topology_id": "ATR_Syngas_Cooler (partial) + ATR_DryCooler",
          "name": "Post-LTWGS Dry Cooler",
          "component_type": "Shell & Tube Heat Exchanger + Air Cooler",
          "process_description": "Two-stage cooling of shifted syngas: (1) Interchanger recovers heat to SOEC water (~800\u2192105\u00b0C), (2) DryCooler final cooling to 30\u00b0C before water knockout.",
          "power_consumption": {
            "drycooler": {
              "reference": "ATR_DryCooler_fan",
              "value_w": 17610.2,
              "value_kw": 17.6
            }
          },
          "coefficients": {
            "K1": 4.8229,
            "K2": -0.0922,
            "K3": 0.1258,
            "F_m": 2.1,
            "F_p": 1.04,
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Sum of Turton costs per sub-unit",
          "C_BM": 868663.91,
          "capacity_variable": {
            "unit": "m2"
          },
          "cost_breakdown": "Sum of 1 sub-units: drycooler (17.6kW): $868,664",
          "note": "Cost calculated by summing 1 sub-units found in power_consumption block"
        },
        {
          "tag": "Dry Cooler - Compressor H2 (Fixed)",
          "topology_id": "ATR_H2_Drycooler",
          "name": "H2 Compressor Dry Cooler",
          "component_type": "Air-Cooled Heat Exchanger",
          "process_description": "Intercooler between ATR H2 compression stages. Cools from ~135\u00b0C to 40\u00b0C. SS316 construction for hydrogen service.",
          "power_consumption": {
            "reference": "ATR_H2_Drycooler_fan",
            "value_w": 17610.2,
            "value_kw": 17.6
          },
          "coefficients": {
            "K1": 4.3247,
            "K2": -0.303,
            "K3": 0.1634,
            "F_m": 2.7,
            "F_m_note": "SS316",
            "F_p": 1.092,
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "kW"
          }
        },
        {
          "tag": "Trocador para Chiller El\u00e9trico",
          "topology_id": "ATR_Chiller_1 (heat exchanger component)",
          "name": "Heat Exchanger for Electric Chiller",
          "component_type": "Kettle Reboiler / Evaporator",
          "process_description": "Heat exchanger for ATR chiller system (5 MW capacity). Provides deep cooling to 4\u00b0C for water knockout. High F_p (1.02) for refrigerant-side pressure.",
          "power_consumption": {
            "chiller_system": {
              "reference": "ATR_Chiller_1_electrical",
              "value_w": 16296.87,
              "value_kw": 16.3,
              "note": "Chiller compressor power - COP ~4.0 implies 65 kW cooling"
            }
          },
          "coefficients": {
            "K1": 4.4646,
            "K2": -0.5302,
            "K3": 0.1644,
            "F_m": 2.7,
            "F_p": 1.02,
            "F_p_note": "Chiller service",
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Sum of Turton costs per sub-unit",
          "C_BM": 148251.48,
          "capacity_variable": {
            "unit": "kW"
          },
          "cost_breakdown": "Sum of 1 sub-units: chiller_system (16.3kW): $148,251",
          "note": "Cost calculated by summing 1 sub-units found in power_consumption block"
        },
        {
          "tag": "Trocador - Aquecimento SOEC (Kettle Reboiler)",
          "topology_id": "ATR_H01_Boiler / ATR_H02_Boiler / ATR_H04_Boiler (surrogate model)",
          "name": "SOEC Heating Heat Exchanger (Kettle Reboiler)",
          "component_type": "Electric Heater (Surrogate Model)",
          "process_description": "Three electric heaters in ATR surrogate model: H01 (biogas preheat), H02 (O2 preheat), H04 (steam/mixed feed heat). Mimic heat integration in actual ATR. Power consumption tracked via O2 flow signal.",
          "power_consumption": {
            "note": "Power calculated internally by ATR_Boiler surrogate model based on O2 flow rate (regression-based)"
          },
          "coefficients": {
            "K1": 4.4646,
            "K2": -0.5302,
            "K3": 0.1644,
            "F_m": 2.7,
            "F_p": 1.04,
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "kW"
          }
        },
        {
          "tag": "Coalescer (ATR)",
          "topology_id": "ATR_Coalescer_1",
          "name": "Coalescer Filter",
          "component_type": "Fiber-Bed Coalescing Filter",
          "process_description": "Fine mist removal from syngas after chilling. Protects PSA adsorbent from water damage. Shell diameter 2.0m (larger than PEM due to higher flow).",
          "power_consumption": {
            "note": "Passive separation - pressure drop ~0.2-0.5 bar"
          },
          "coefficients": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 2.1,
            "F_p": 1.2,
            "B1": 2.25,
            "B2": 1.82,
            "F_coalescer": 1.25
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {}
        },
        {
          "tag": "PSA-ATR",
          "topology_id": "ATR_PSA_1 (SyngasPSA)",
          "name": "PSA Unit for Syngas Purification",
          "component_type": "Pressure Swing Adsorption (Syngas Service)",
          "process_description": "H2 purification from shifted syngas. Removes CO2, CO, CH4, N2. Target 99.9% purity, 90% recovery. Operates at 15 bar, 10-min cycles. Bed: 2.5m length \u00d7 1.0m diameter.",
          "power_consumption": {
            "note": "Valve actuation and instrumentation - typically 25 kW"
          },
          "coefficients": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 3.1,
            "F_p": 1.2,
            "B1": 2.25,
            "B2": 1.82,
            "F_PSA": 1.25
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {}
        },
        {
          "tag": "Multicyclone Separator (Syngas) - Vessel",
          "topology_id": "ATR_Cyclone_1 / ATR_Cyclone_2",
          "name": "Multicyclone Separator (Syngas) - Vessel",
          "component_type": "Multi-Element Centrifugal Separator",
          "process_description": "Two cyclones in ATR train: (1) Post-DryCooler, (2) Post-Chiller. Remove condensed water from cooled syngas. SS316 construction, 15 bar operation (F_p=1.78).",
          "power_consumption": {
            "note": "Passive separation - pressure drop ~0.1 bar per stage"
          },
          "coefficients": {
            "K1": 3.4974,
            "K2": 0.4485,
            "K3": 0.1074,
            "F_m": 2.1,
            "F_m_note": "SS316",
            "F_p": 1.78,
            "F_p_note": "Reference pressure 15 bar"
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "m3"
          }
        },
        {
          "tag": "Multicyclone Separator (Syngas) - Internals",
          "topology_id": "Integrated with ATR_Cyclone_1 / ATR_Cyclone_2",
          "name": "Multicyclone Separator (Syngas) - Internals (Vane Pack)",
          "component_type": "Cyclone Internals",
          "process_description": "Vane pack internals for syngas cyclones. 50mm element diameter, 45\u00b0 vane angle. Designed for syngas mixture (H2, CO, CO2, H2O vapor).",
          "coefficients": {
            "K1": 3.349,
            "K2": 0.4669,
            "K3": 0.1243
          },
          "note": "Calculate combined vessel + internals cost",
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "m3"
          }
        }
      ]
    },
    "COMMON_COMPONENTS": {
      "description": "Components shared across multiple blocks or plant-wide systems",
      "components": [
        {
          "name": "Rectifier and Transformer (Balance of Plant)",
          "tag": "RCT-3",
          "topology_id": "Implicit in LP/HP compression train power",
          "component_type": "AC-DC Power Conversion System",
          "process_description": "Grid connection and power conditioning for balance-of-plant loads (pumps, fans, compressors, chillers, control systems). Separate from SOEC/PEM rectifiers.",
          "power_consumption": {
            "note": "Services total auxiliary load - see individual component consumptions in max_power_consumption.json"
          },
          "cost_basis": {
            "Cp0_per_kW": 200,
            "source": "IEA Global H2 Review",
            "F_BM": 2.15
          },
          "calculation": "CAPEX = P_rated \u00d7 200 \u00d7 2.0655 \u00d7 2.15",
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "kW"
          }
        },
        {
          "name": "Underground H2 Storage",
          "topology_id": "LP_Storage_Tank",
          "component_type": "High-Pressure Gas Storage (30 Tanks \u00d7 107.37 m\u00b3)",
          "process_description": "Buffer storage at 70 bar (max). 30 tanks, each 107.37 m\u00b3. Total capacity ~3222 m\u00b3 at 70 bar (~15,000 kg H2). Provides supply decoupling between production and truck loading.",
          "power_consumption": {
            "note": "Passive storage - no electrical consumption. Associated compression power listed under LP/HP compressors."
          },
          "cost_basis": {
            "value_eur": 6736268.15,
            "value_usd": 7409895.0,
            "source": "ROGER"
          },
          "cost_formula": "Formula not found",
          "C_BM": 7409895.0
        },
        {
          "name": "Filling/Loading Docks (5 stations)",
          "topology_id": "Truck_Station_1 (5 parallel stations)",
          "component_type": "Truck Dispensing Infrastructure",
          "process_description": "Five identical truck loading bays. Each handles 280 kg tube trailers. Fill rate 0.583-1.5 kg/min (35-90 kg/h). Max 16 hours/day operation. Includes pressure vessels, valving, metering, safety systems.",
          "power_consumption": {
            "compression_energy": {
              "note": "Isentropic compression from storage (30-70 bar) to truck (500 bar). Power calculated dynamically by DischargeStation model.",
              "efficiency": {
                "isentropic": 0.65,
                "mechanical": 0.9
              }
            }
          },
          "cost_basis": {
            "value_eur": 1931019.12,
            "value_usd": 2124121.0,
            "source": "ROGER"
          },
          "cost_formula": "Formula not found",
          "C_BM": 2124121.0
        },
        {
          "name": "Grid Connection Safeguarding",
          "topology_id": "Plant-wide electrical infrastructure",
          "component_type": "Electrical Protection and Control Systems",
          "process_description": "Grid interconnection equipment: circuit breakers, relays, metering, SCADA interfaces, power quality monitoring. Ensures safe connection of ~17 MW peak load (11.5 MW SOEC + 5 MW PEM + auxiliaries).",
          "cost_basis": {
            "value_eur": 439041.23,
            "value_usd": 482945.0
          },
          "cost_formula": "Formula not found",
          "C_BM": 482945.0
        }
      ]
    },
    "H2_STORAGE_DISTRIBUTION": {
      "description": "Hydrogen storage and truck distribution systems - Final compression and dispensing",
      "process_location": "Downstream of all production trains - accepts H2 from common header at ~40 bar",
      "components": [
        {
          "tag": "LP_Compressor",
          "topology_id": "LP_Compressor_S1",
          "name": "Low-Pressure Hydrogen Compressor",
          "component_type": "Reciprocating Compressor",
          "process_description": "First stage compression from production header (~40 bar) to mid-pressure (~70 bar). Handles combined H2 from SOEC, PEM, and ATR. Single-stage temperature-limited to 135\u00b0C.",
          "power_consumption": {
            "reference": "LP_Compressor_S1",
            "value_w": 208401.96,
            "value_kw": 208.4,
            "intercooler": {
              "reference": "LP_Intercooler_1_fan",
              "value_w": 88051.02,
              "value_kw": 88.1
            },
            "total_kw": 296.5
          },
          "coefficients": {
            "K1": 2.0309,
            "K2": 1.2524,
            "K3": -0.0638,
            "F_m": 2.1,
            "F_m_note": "Alloy Steel/Stainless for H2",
            "F_BM": 2.15
          },
          "calculation_steps": [
            "Same as SOEC MTC-4/MTC-5 (reciprocating H2 compressors)"
          ],
          "cost_formula": "Sum of Turton costs per sub-unit",
          "C_BM": 156718.32,
          "capacity_variable": {
            "unit": "kW"
          },
          "cost_breakdown": "Sum of 1 sub-units: intercooler (88.1kW): $156,718",
          "note": "Cost calculated by summing 1 sub-units found in power_consumption block"
        },
        {
          "tag": "HP_Compressor_Train",
          "topology_id": "HP_Compressor_S2 / HP_Compressor_S3 / HP_Compressor_S4 / HP_Compressor_S5",
          "name": "High-Pressure Hydrogen Compressor Train (4 Stages)",
          "component_type": "Reciprocating Compressor (Multi-Stage)",
          "process_description": "Four-stage compression from ~70 bar to storage pressure (500 bar). Each stage intercooled to 40\u00b0C. Temperature-limited to 135\u00b0C per stage. Fills HP tanks for truck dispensing.",
          "power_consumption": {
            "stage_2": {
              "reference": "HP_Compressor_S2",
              "value_w": 46161.95,
              "value_kw": 46.2
            },
            "stage_3": {
              "reference": "HP_Compressor_S3",
              "value_w": 45842.49,
              "value_kw": 45.8
            },
            "stage_4": {
              "reference": "HP_Compressor_S4",
              "value_w": 48524.78,
              "value_kw": 48.5
            },
            "stage_5": {
              "reference": "HP_Compressor_S5",
              "value_w": 48522.15,
              "value_kw": 48.5
            },
            "total_kw": 189.0,
            "intercooler_fans": {
              "hp_intercooler_2": {
                "reference": "HP_Intercooler_2_fan",
                "value_w": 88051.02,
                "value_kw": 88.1
              },
              "hp_intercooler_3": {
                "reference": "HP_Intercooler_3_fan",
                "value_w": 88051.02,
                "value_kw": 88.1
              },
              "hp_intercooler_4": {
                "reference": "HP_Intercooler_4_fan",
                "value_w": 88051.02,
                "value_kw": 88.1
              },
              "total_fans_kw": 264.2
            },
            "total_compression_system_kw": 453.2
          },
          "coefficients": {
            "K1": 2.0309,
            "K2": 1.2524,
            "K3": -0.0638,
            "F_m": 2.1,
            "F_m_note": "Alloy Steel/Stainless for H2",
            "F_BM": 2.15
          },
          "calculation_steps": [
            "Same as SOEC MTC-4/MTC-5 (reciprocating H2 compressors)",
            "Apply per-stage calculations, sum for total train cost"
          ],
          "cost_formula": "Sum of Turton costs per sub-unit",
          "C_BM": 801889.27,
          "capacity_variable": {
            "unit": "kW"
          },
          "cost_breakdown": "Sum of 7 sub-units: stage_2 (46.2kW): $81,019 + stage_3 (45.8kW): $80,290 + stage_4 (48.5kW): $85,213 + stage_5 (48.5kW): $85,213 + intercooler_fans_hp_intercooler_2 (88.1kW): $156,718 + intercooler_fans_hp_intercooler_3 (88.1kW): $156,718 + intercooler_fans_hp_intercooler_4 (88.1kW): $156,718",
          "note": "Cost calculated by summing 7 sub-units found in power_consumption block"
        },
        {
          "tag": "HP_Intercoolers",
          "topology_id": "HP_Intercooler_2 / HP_Intercooler_3 / HP_Intercooler_4",
          "name": "High-Pressure Intercooler Heat Exchangers",
          "component_type": "Air-Cooled Heat Exchanger (Dry Cooler)",
          "process_description": "Three intercoolers in HP compression train. Cool from ~135\u00b0C to 40\u00b0C between stages. Handle high-pressure H2 (up to 500 bar) - requires high F_p.",
          "power_consumption": {
            "per_unit": {
              "reference": "HP_Intercooler_X_fan",
              "value_w": 88051.02,
              "value_kw": 88.1
            },
            "total_kw": 264.2
          },
          "coefficients": {
            "K1": 4.3247,
            "K2": -0.303,
            "K3": 0.1634,
            "F_m": 2.7,
            "F_m_note": "SS316 for H2 service",
            "F_p": 2.5,
            "F_p_note": "High pressure (>200 bar stages)",
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "m2"
          }
        },
        {
          "tag": "H2_Production_Cooler",
          "topology_id": "H2_Production_Cooler",
          "name": "Production Header Cooler",
          "component_type": "Air-Cooled Heat Exchanger (Dry Cooler)",
          "process_description": "Final cooling of combined H2 stream before storage. Cools to 30\u00b0C to maximize storage density. Located after H2_Production_Mixer.",
          "power_consumption": {
            "reference": "H2_Production_Cooler_fan",
            "value_w": 17610.2,
            "value_kw": 17.6
          },
          "coefficients": {
            "K1": 4.3247,
            "K2": -0.303,
            "K3": 0.1634,
            "F_m": 2.7,
            "F_p": 1.092,
            "B1": 1.63,
            "B2": 1.66
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "m2"
          }
        },
        {
          "tag": "ATR_O2_Compressor",
          "topology_id": "ATR_O2_Compressor",
          "name": "Oxygen Compressor for ATR Feed",
          "component_type": "Centrifugal Compressor",
          "process_description": "Compresses combined O2 from SOEC + PEM trains (~15 bar) to ATR operating pressure. Feeds O2 to ATR for partial oxidation reaction. Nickel alloy construction for oxygen service.",
          "power_consumption": {
            "reference": "ATR_O2_Compressor",
            "value_w": 125039.73,
            "value_kw": 125.0
          },
          "coefficients": {
            "K1": 2.2891,
            "K2": 1.3604,
            "K3": -0.1027,
            "F_m": 3.5,
            "F_m_note": "Nickel alloys/Monel for O2 service",
            "F_BM": 2.15
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "kW"
          }
        },
        {
          "tag": "PEM_Water_Pump",
          "topology_id": "PEM_Water_Pump",
          "name": "PEM Water Feed Pump",
          "component_type": "Centrifugal Pump with Electric Motor",
          "process_description": "Pressurizes PEM feed water from makeup mixer (1 bar) to PEM operating pressure (40 bar). Handles ~4000 kg/h water flow. SS316 construction.",
          "power_consumption": {
            "reference": "PEM_Water_Pump",
            "value_w": 6011.59,
            "value_kw": 6.0
          },
          "coefficients": {
            "pump": {
              "K1": 3.3892,
              "K2": 0.0536,
              "K3": 0.1538,
              "F_m": 2.41,
              "F_p": 1.0,
              "B1": 1.89,
              "B2": 1.35
            },
            "motor": {
              "K1_low": 3.3432,
              "K2_low": 0.2761,
              "K3_low": 0.0543
            }
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "kW"
          }
        },
        {
          "tag": "ATR_Water_Pumps",
          "topology_id": "ATR_Feed_Pump / ATR_Drain_Pump_1 / ATR_Drain_Pump_2",
          "name": "ATR Water Feed and Drain Pumps",
          "component_type": "Centrifugal Pump with Electric Motor",
          "process_description": "Water management pumps for ATR: (1) Feed_Pump (15 bar for ATR steam), (2-3) Drain_Pumps for condensate recovery from cyclones.",
          "power_consumption": {
            "feed_pump": {
              "reference": "ATR_Feed_Pump",
              "value_w": 927.91,
              "value_kw": 0.93
            },
            "drain_pump_1": {
              "reference": "ATR_Drain_Pump_1",
              "value_w": 150.91,
              "value_kw": 0.15
            },
            "drain_pump_2": {
              "reference": "ATR_Drain_Pump_2",
              "value_w": 14.43,
              "value_kw": 0.01
            },
            "total_kw": 1.09
          },
          "coefficients": {
            "pump": {
              "K1": 3.3892,
              "K2": 0.0536,
              "K3": 0.1538,
              "F_m": 2.41,
              "F_p": 1.0,
              "B1": 1.89,
              "B2": 1.35
            }
          },
          "cost_formula": "Formula not found",
          "C_BM": null,
          "error": "Missing parameters for calculation",
          "capacity_variable": {
            "unit": "kW"
          }
        }
      ]
    }
  },
  "indirect_costs": {
    "description": "Applied as percentages of direct equipment cost",
    "multipliers": {
      "electrical_PEM": "45-55%",
      "electrical_SOEC": "45-55%",
      "instrumentation_PEM": "20%",
      "instrumentation_SOEC": "20%",
      "piping_PEM": "15%",
      "piping_SOEC": "15%",
      "piping_ATR_CCS": "60-70%",
      "civil_foundations_ATR_CCS": "25%",
      "insulation_ATR_CCS": "10%",
      "piping_insulation_SOEC_ATR": "20%",
      "engineering_supervision": "33%",
      "construction_expenses": "15%",
      "contingency": "15%"
    },
    "notes": {
      "electrical": "Includes wiring, conduit, panels, transformers, motor starters. Higher for SOEC/PEM due to high-power DC systems.",
      "instrumentation": "Sensors, analyzers, control valves, DCS hardware/software, safety interlocks.",
      "piping": "Process piping, valves, fittings, supports. ATR/CCS higher due to high-temperature materials and longer runs.",
      "civil_foundations": "Concrete pads, structural steel, building enclosures. ATR/CCS higher due to heavy reactor vessels.",
      "insulation": "Thermal insulation and cladding. ATR has extensive high-temp insulation (up to 1100\u00b0C).",
      "engineering": "FEED, detailed design, procurement, construction management, commissioning support.",
      "construction": "Labor, equipment rental, temporary facilities, QA/QC, safety programs.",
      "contingency": "Design allowance for scope uncertainty, regulatory changes, market escalation."
    }
  },
  "validation_checks": {
    "capacity_ranges": "Always verify input within valid range",
    "material_factors": "F_m must match service conditions",
    "pressure_factors": "Calculate F_p for vessels >5 bar",
    "inflation_check": "CEPCI Base=397, Target=820",
    "source_file_accuracy": "All values verified against source CSV"
  },
  "references": {
    "primary": "Turton, R., et al. (2012). Analysis, Synthesis and Design of Chemical Processes, 4th Edition",
    "CEPCI": "Chemical Engineering Plant Cost Index",
    "industry_sources": [
      "Carlini, G. (2025) - ATR/WGS cost data",
      "ROGER Study - Storage and infrastructure costs",
      "IEA Global Hydrogen Review - Transformer/Rectifier costs"
    ]
  }
}