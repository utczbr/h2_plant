"""
Scenario Summary Report Generator

Consolidates simulation metrics into a single CSV report.
Optimized for memory efficiency via streaming/chunked processing.
Handles H2/O2 production, energy consumption, thermal integration, 
and RFNBO certification analysis using authoritative tags from graph_catalog.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

logger = logging.getLogger(__name__)

# =============================================================================
# COLUMN DEFINITIONS (Aligned with graph_catalog.py)
# =============================================================================
# These patterns allow the UnifiedGraphExecutor to filter columns efficiently.

SUMMARY_COLUMNS = [
    # Time
    'minute',
    
    # Production (Mass per step)
    'H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg',
    'O2_soec_kg', 'O2_pem_kg',
    
    # Production (Rates for Integration)
    '*_outlet_mass_flow_kg_h',  # PSA outlets
    'ATR_Coalescer_1_outlet_mass_flow_kg_h',
    'ATR_PSA_1_outlet_mass_flow_kg_h',
    
    # RFNBO (Mass per step)
    'h2_rfnbo_kg', 'h2_non_rfnbo_kg',
    'cumulative_h2_rfnbo_kg', 'cumulative_h2_non_rfnbo_kg',
    'cumulative_h2_kg',
    
    # Power (MW/kW)
    'P_soec_actual', 'P_pem', 'P_bop_mw', 'P_sold', 'P_offer',
    'bop_grid_import_mw', 'compressor_power_kw',
    
    # Consumption (Rates)
    'H2O_pem_kg', 'steam_soec_kg', # Water (Mass per step)
    'ATR_Makeup_Mixer_outlet',     # Water (Rate?) check usage
    'Biogas_Source_out',           # Biogas (Rate or Mass?) -> Usually Rate
    'ATR_O2_Compressor_outlet',    # Oxygen (Rate)
    
    # Thermal (kW)
    '*heat_rejected_kw*', '*cooling_load_kw*', '*tqc_duty_kw*',
    '*boiler*_power_input*',
    'ATR_Syngas_Cooler*', 'Syngas_Cooler_q_transferred_kw'
]


class ScenarioSummaryAccumulator:
    """
    Accumulates metrics chunk-by-chunk to avoid loading full history into memory.
    """
    
    def __init__(self, guaranteed_power_mw: float = 10.0):
        self.guaranteed_power_mw = guaranteed_power_mw
        self.total_time_h = 0.0
        
        # Accumulators
        self.sums: Dict[str, float] = {}
        self.last_cumulative: Dict[str, float] = {} # For cumulative columns (take last value)
        
        # Specific specialized accumulators
        self.h2_purified_sum = 0.0
        self.offgas_atr_sum = 0.0
        self.biogas_mass_sum = 0.0
        self.water_atr_sum = 0.0
        self.heat_rejected_mwh = 0.0
        self.heat_recovery_atr_kwh = 0.0
        
    def update(self, df_chunk: pd.DataFrame):
        """
        Process a DataFrame chunk and update running totals.
        """
        if df_chunk.empty:
            return

        # 1. Time Delta Calculation
        if 'minute' in df_chunk.columns:
            # Calculate local dt
            dt_minutes = df_chunk['minute'].diff().fillna(0).clip(lower=0)
            
            # Handle chunk boundaries (first element diff is NaN/0, need estimate)
            # If we had statefulness we could track last minute of previous chunk.
            # For simplicity in stateless update, assume mean step for the first element
            # or usage of a uniform step if provided.
            mean_step = dt_minutes[dt_minutes > 0].mean()
            if pd.isna(mean_step) or mean_step == 0:
                mean_step = 60.0 # Default 1 hour
            
            # Replace the first 0 with mean_step (assuming continuous simulation)
            # note: this introduces small error at chunk boundaries if not careful, 
            # but is standard for summary approximations.
            dt_minutes.iloc[0] = mean_step 
            
            dt_hours = dt_minutes / 60.0
            self.total_time_h += dt_hours.sum()
        else:
            # Fallback
            dt_hours = pd.Series([1.0] * len(df_chunk), index=df_chunk.index)
            self.total_time_h += len(df_chunk)

        # Helper: Integrate Rate (Column * dt) -> e.g., MW to MWh, kg/h to kg
        def integrate(col_name: str, scaling: float = 1.0) -> float:
            if col_name in df_chunk.columns:
                return (df_chunk[col_name] * dt_hours).sum() * scaling
            return 0.0
            
        def integrate_pattern(pattern: str, scaling: float = 1.0) -> float:
            cols = [c for c in df_chunk.columns if pattern.lower() in c.lower()]
            total = 0.0
            for c in cols:
                total += (df_chunk[c] * dt_hours).sum() * scaling
            return total

        # Helper: Accumulate Quantity (Sum of Column) -> e.g., kg per step
        def accumulate(col_name: str, scaling: float = 1.0) -> float:
            if col_name in df_chunk.columns:
                return df_chunk[col_name].sum() * scaling
            return 0.0

        # --- 1. Production (Per-step Mass) ---
        for col in ['H2_soec_kg', 'H2_pem_kg', 'H2_atr_kg', 'O2_soec_kg', 'O2_pem_kg']:
            self.sums[col] = self.sums.get(col, 0.0) + accumulate(col)
            
        # --- 2. RFNBO (Per-step Mass) ---
        for col in ['h2_rfnbo_kg', 'h2_non_rfnbo_kg']:
            self.sums[col] = self.sums.get(col, 0.0) + accumulate(col)
            
        # Capture last values of cumulative columns
        for col in ['cumulative_h2_kg', 'cumulative_h2_rfnbo_kg', 'cumulative_h2_non_rfnbo_kg']:
            if col in df_chunk.columns:
                self.last_cumulative[col] = df_chunk[col].iloc[-1]

        # --- 3. H2 Purified (Integration of PSA Outlet Rates) ---
        psa_product_cols = [
            'SOEC_H2_PSA_1_outlet_mass_flow_kg_h',
            'PEM_H2_PSA_1_outlet_mass_flow_kg_h', 
            'ATR_PSA_1_outlet_mass_flow_kg_h'
        ]
        for c in psa_product_cols:
            self.h2_purified_sum += integrate(c)

        # --- 4. Offgas / Vent ---
        # Explicit tail gas integration
        tail_cols = [c for c in df_chunk.columns if 'PSA' in c and 'tail_gas' in c]
        if tail_cols:
            for c in tail_cols:
                self.offgas_atr_sum += integrate(c)
        else:
            # Fallback estimation per chunk (Inlet - Outlet)
            inlet = integrate('ATR_Coalescer_1_outlet_mass_flow_kg_h')
            outlet = integrate('ATR_PSA_1_outlet_mass_flow_kg_h')
            self.offgas_atr_sum += max(0.0, inlet - outlet)

        # --- 5. Consumption (Water/Biogas) ---
        # Water (per step mass)
        for col in ['H2O_pem_kg', 'steam_soec_kg']:
            self.sums[col] = self.sums.get(col, 0.0) + accumulate(col)
            
        # ATR Water (Mixer Rate)
        self.water_atr_sum += integrate_pattern('ATR_Makeup_Mixer_outlet')
        
        # Biogas (Rate)
        self.biogas_mass_sum += integrate_pattern('Biogas_Source_out')
        
        # O2 for ATR (Rate)
        self.sums['ATR_O2_in'] = self.sums.get('ATR_O2_in', 0.0) + integrate_pattern('ATR_O2_Compressor_outlet')

        # --- 6. Power & Energy (Integration) ---
        # MW -> MWh
        for col in ['P_soec_actual', 'P_pem', 'P_bop_mw', 'P_sold', 'bop_grid_import_mw', 'P_offer']:
            self.sums[f"{col}_MWh"] = self.sums.get(f"{col}_MWh", 0.0) + integrate(col)

        # --- 7. Thermal (Integration) ---
        # kW -> kWh (divide by 1000 later for MWh)
        cols_reject = [
            c for c in df_chunk.columns 
            if ('heat_rejected' in c or 'cooling_load' in c or 'tqc_duty' in c) 
            and 'boiler' not in c.lower()
        ]
        for c in cols_reject:
            self.heat_rejected_mwh += integrate(c) / 1000.0
            
        # ATR Heat Recovery
        self.heat_recovery_atr_kwh += integrate_pattern('ATR_Syngas_Cooler', scaling=1.0)
        self.heat_recovery_atr_kwh += integrate_pattern('Syngas_Cooler_q_transferred_kw', scaling=1.0)

    def finalize(self, scenario_name: str, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Compute final metrics and generate DataFrame.
        """
        metrics = []
        
        # --- 1. PRODUCTION ---
        h2_soec = self.sums.get('H2_soec_kg', 0.0)
        h2_pem = self.sums.get('H2_pem_kg', 0.0)
        h2_atr = self.sums.get('H2_atr_kg', 0.0)
        
        # Total H2: Use cumulative column if available (more accurate), else sum
        if 'cumulative_h2_kg' in self.last_cumulative:
            h2_total = self.last_cumulative['cumulative_h2_kg']
        else:
            h2_total = h2_soec + h2_pem + h2_atr

        metrics.append(('Production', 'H2 Total Generated', 'kg', h2_total))
        metrics.append(('Production', 'H2 SOEC', 'kg', h2_soec))
        metrics.append(('Production', 'H2 PEM', 'kg', h2_pem))
        metrics.append(('Production', 'H2 ATR', 'kg', h2_atr))
        metrics.append(('Production', 'H2 Total Purified', 'kg', self.h2_purified_sum))

        # O2
        o2_pem = self.sums.get('O2_pem_kg', 0.0)
        o2_soec = self.sums.get('O2_soec_kg', 0.0)
        if o2_soec == 0 and h2_soec > 0:
            o2_soec = h2_soec * 8.0 # Fallback stoichiometry
            
        metrics.append(('Production', 'O2 Total', 'kg', o2_pem + o2_soec))
        metrics.append(('Production', 'O2 SOEC', 'kg', o2_soec))
        metrics.append(('Production', 'O2 PEM', 'kg', o2_pem))
        metrics.append(('Production', 'Offgas/Vent', 'kg', self.offgas_atr_sum))

        # --- 2. CONSUMPTION ---
        water_pem = self.sums.get('H2O_pem_kg', 0.0)
        water_soec = self.sums.get('steam_soec_kg', 0.0)
        water_atr = self.water_atr_sum
        if water_atr == 0 and h2_atr > 0:
            water_atr = h2_atr * 9.0 # Fallback
            
        metrics.append(('Consumption', 'Water Total', 'kg', water_pem + water_soec + water_atr))
        metrics.append(('Consumption', 'Water ATR', 'kg', water_atr))
        
        biogas = self.biogas_mass_sum
        if biogas == 0 and h2_atr > 0:
            biogas = h2_atr * 5.5 # Fallback
            
        metrics.append(('Consumption', 'Biogas Mass', 'kg', biogas))
        # Energy (LHV approx 13.9 kWh/kg)
        biogas_energy_mwh = (biogas * 13.9) / 1000.0
        metrics.append(('Consumption', 'Biogas Energy', 'MWh', biogas_energy_mwh))
        metrics.append(('Consumption', 'O2 Consumption (ATR)', 'kg', self.sums.get('ATR_O2_in', 0.0)))

        # --- 3. ELECTRICAL ---
        e_soec = self.sums.get('P_soec_actual_MWh', 0.0)
        e_pem = self.sums.get('P_pem_MWh', 0.0)
        e_bop = self.sums.get('P_bop_mw_MWh', 0.0)
        
        metrics.append(('Electrical', 'Total Plant Load', 'MWh', e_soec + e_pem + e_bop))
        metrics.append(('Electrical', 'SOEC Load', 'MWh', e_soec))
        metrics.append(('Electrical', 'PEM Load', 'MWh', e_pem))
        metrics.append(('Electrical', 'Balance of Plant', 'MWh', e_bop))

        # --- 4. THERMAL ---
        metrics.append(('Thermal', 'Total Heat Rejected', 'MWh', self.heat_rejected_mwh))
        metrics.append(('Thermal', 'Heat Exchange ATR->SOEC', 'MWh', self.heat_recovery_atr_kwh / 1000.0))

        # --- 5. GRID ---
        e_exported = self.sums.get('P_sold_MWh', 0.0)
        e_imported = self.sums.get('bop_grid_import_mw_MWh', 0.0)
        metrics.append(('Grid', 'Energy Sold', 'MWh', e_exported))
        metrics.append(('Grid', 'Energy Purchased', 'MWh', e_imported))
        
        total_renewable = self.sums.get('P_offer_MWh', 0.0)
        e_guaranteed = self.guaranteed_power_mw * self.total_time_h
        e_variable = max(0.0, total_renewable - e_guaranteed)
        
        metrics.append(('Grid', 'Renewable Available (Total)', 'MWh', total_renewable))
        metrics.append(('Grid', 'Renewable (Guaranteed Base)', 'MWh', e_guaranteed))
        metrics.append(('Grid', 'Renewable (Variable Wind)', 'MWh', e_variable))

        # --- 6. RFNBO ---
        if 'cumulative_h2_rfnbo_kg' in self.last_cumulative:
            rfnbo_kg = self.last_cumulative['cumulative_h2_rfnbo_kg']
            non_rfnbo_kg = self.last_cumulative.get('cumulative_h2_non_rfnbo_kg', 0.0)
        else:
            rfnbo_kg = self.sums.get('h2_rfnbo_kg', 0.0)
            non_rfnbo_kg = self.sums.get('h2_non_rfnbo_kg', 0.0)

        metrics.append(('RFNBO', 'Compliant H2 (RFNBO)', 'kg', rfnbo_kg))
        metrics.append(('RFNBO', 'Non-Compliant H2', 'kg', non_rfnbo_kg))
        
        total_rfnbo_mass = rfnbo_kg + non_rfnbo_kg
        ratio = (rfnbo_kg / total_rfnbo_mass * 100.0) if total_rfnbo_mass > 0 else 0.0
        metrics.append(('RFNBO', 'Compliance Ratio', '%', ratio))

        # --- 7. EFFICIENCY ---
        h2_lhv_mwh = (h2_total * 33.33) / 1000.0 
        total_input_mwh = (e_soec + e_pem + e_bop) + biogas_energy_mwh
        global_eff = (h2_lhv_mwh / total_input_mwh * 100.0) if total_input_mwh > 0 else 0.0
        
        metrics.append(('Efficiency', 'Global Plant Efficiency (LHV)', '%', global_eff))
        
        sec_soec = (e_soec * 1000.0) / h2_soec if h2_soec > 0 else 0.0
        sec_pem = (e_pem * 1000.0) / h2_pem if h2_pem > 0 else 0.0
        
        metrics.append(('Efficiency', 'SEC SOEC', 'kWh/kg', sec_soec))
        metrics.append(('Efficiency', 'SEC PEM', 'kWh/kg', sec_pem))

        # Export
        summary_df = pd.DataFrame(metrics, columns=['Category', 'Metric', 'Unit', 'Value'])
        summary_df['Scenario'] = scenario_name
        summary_df = summary_df[['Scenario', 'Category', 'Metric', 'Unit', 'Value']]
        
        if output_path:
            mode = 'a' if output_path.exists() else 'w'
            header = not output_path.exists()
            try:
                summary_df.to_csv(output_path, mode=mode, header=header, index=False)
                logger.info(f"Summary report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save summary report: {e}")

        return summary_df


def generate_scenario_summary(
    df: pd.DataFrame,
    scenario_name: str,
    output_path: Path,
    guaranteed_power_mw: float = 10.0
) -> pd.DataFrame:
    """
    Process simulation history and generate a summarized CSV report.
    Compatible with both full DataFrames and chunked processing flows.

    Args:
        df: Simulation history DataFrame (full or partial)
        scenario_name: Label for the scenario (e.g., "Baseline_2025")
        output_path: Path to save the output CSV
        guaranteed_power_mw: The contract base power in MW.

    Returns:
        pd.DataFrame: The generated summary table in long format.
    """
    logger.info(f"Generating summary report for scenario: {scenario_name}")
    
    accumulator = ScenarioSummaryAccumulator(guaranteed_power_mw)
    
    # Process the dataframe (assume it's the full history in this backward-compatible call)
    accumulator.update(df)
    
    return accumulator.finalize(scenario_name, output_path)