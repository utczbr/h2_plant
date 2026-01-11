#!/usr/bin/env python3
"""
Daily Average Hydrogen Production Graph.

Generates a graph showing:
- Gross daily H2 production by source (PEM, SOEC, ATR) as stacked bars
- Total purified daily H2 production as a line overlay

The data is aggregated to daily averages across the simulation year.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

def generate_daily_h2_production_graph(
    csv_path: str = "scenarios/simulation_output/simulation_history.csv",
    output_path: str = "scenarios/simulation_output/daily_h2_production.png",
    config_path: str = None,
    nominal_daily_production_kg: float = None
):
    """
    Generate daily H2 production graph.
    
    Args:
        csv_path: Path to simulation history CSV
        output_path: Path for output PNG
        config_path: Optional path to visualization_config.yaml
        nominal_daily_production_kg: Override for 100% capacity (kg/day).
                                     If None, reads from config or uses 8500.
    """
    
    # Load nominal production from config if not provided
    if nominal_daily_production_kg is None:
        nominal_daily_production_kg = 8500.0  # Default: SOEC + PEM only
        
        if config_path:
            config_file = Path(config_path)
        else:
            # Try to find config relative to csv_path
            config_file = Path(csv_path).parent.parent / "visualization_config.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    viz_config = yaml.safe_load(f)
                daily_config = viz_config.get('visualization', {}).get('orchestrated_graphs', {}).get('daily_h2_production_average', {})
                nominal_daily_production_kg = daily_config.get('nominal_daily_production_kg', 8500.0)
                print(f"Loaded nominal production from config: {nominal_daily_production_kg} kg/day")
            except Exception as e:
                print(f"Warning: Could not load config ({e}), using default 8500 kg/day")
    
    print("Loading simulation data...")
    df = pd.read_csv(csv_path)
    
    dt_hours = 1/60  # 1 minute timestep
    
    # =========================================================================
    # GROSS PRODUCTION
    # H2 produced by each source BEFORE PSA purification
    # =========================================================================
    
    # SOEC: Direct electrolysis H2 (recorded per timestep)
    if 'H2_soec_kg' in df.columns:
        df['Gross_SOEC_kg'] = df['H2_soec_kg']
    else:
        df['Gross_SOEC_kg'] = 0.0
        
    # PEM: Back-calculate from PSA output (consistent 90%)
    pem_psa_col = 'PEM_H2_PSA_1_outlet_mass_flow_kg_h'
    if pem_psa_col in df.columns:
        df['Gross_PEM_kg'] = (df[pem_psa_col] / 0.90) * dt_hours
    elif 'H2_pem_kg' in df.columns:
        df['Gross_PEM_kg'] = df['H2_pem_kg']
    else:
        df['Gross_PEM_kg'] = 0.0
    
    # ATR: Post-WGS H2 entering the PSA (NOT reactor output, which excludes WGS gain)
    # The ATR reactor output goes through HTWGS + LTWGS which INCREASES H2 by converting CO
    # Correct "gross" is the H2 entering the PSA, calculated from PSA outlet / recovery_rate
    atr_psa_col = 'ATR_PSA_1_outlet_mass_flow_kg_h'
    if atr_psa_col in df.columns:
        # Calculate PSA input from output using recovery rate (90%)
        # Gross = Purified / 0.90
        df['Gross_ATR_kg'] = (df[atr_psa_col] / 0.90) * dt_hours
    elif 'H2_atr_kg' in df.columns:
        # Fallback: Use reactor output, but note this excludes WGS gain
        df['Gross_ATR_kg'] = df['H2_atr_kg']
        print("Warning: Using ATR reactor output (excludes WGS H2 gain)")
    else:
        df['Gross_ATR_kg'] = 0.0
    
    # =========================================================================
    # PURIFIED PRODUCTION  
    # H2 output from each PSA unit (high purity product)
    # =========================================================================
    
    # SOEC PSA: flow rate × timestep
    soec_psa_col = 'SOEC_H2_PSA_1_outlet_mass_flow_kg_h'
    if soec_psa_col in df.columns:
        df['Purified_SOEC_kg'] = df[soec_psa_col] * dt_hours
    else:
        df['Purified_SOEC_kg'] = df['Gross_SOEC_kg'] * 0.90  # Estimate
        print(f"Note: {soec_psa_col} not found, estimating 90% recovery")
    
    # PEM PSA: flow rate × timestep
    pem_psa_col = 'PEM_H2_PSA_1_outlet_mass_flow_kg_h'
    if pem_psa_col in df.columns:
        df['Purified_PEM_kg'] = df[pem_psa_col] * dt_hours
    else:
        df['Purified_PEM_kg'] = df['Gross_PEM_kg'] * 0.90  # Estimate
        print(f"Note: {pem_psa_col} not found, estimating 90% recovery")
    
    # ATR PSA: flow rate × timestep
    if atr_psa_col in df.columns:
        df['Purified_ATR_kg'] = df[atr_psa_col] * dt_hours
    else:
        df['Purified_ATR_kg'] = df['Gross_ATR_kg'] * 0.90  # Estimate
        print(f"Note: {atr_psa_col} not found, estimating 90% recovery")
    
    # =========================================================================
    # AGGREGATE BY DAY
    # =========================================================================
    
    # Convert minute to day (1 day = 1440 minutes)
    df['day'] = df['minute'] // 1440
    
    daily = df.groupby('day').agg({
        # Gross production
        'Gross_SOEC_kg': 'sum',
        'Gross_PEM_kg': 'sum', 
        'Gross_ATR_kg': 'sum',
        # Purified production
        'Purified_SOEC_kg': 'sum',
        'Purified_PEM_kg': 'sum',
        'Purified_ATR_kg': 'sum',
    }).reset_index()
    
    # Calculate totals
    daily['Gross_Total'] = daily['Gross_SOEC_kg'] + daily['Gross_PEM_kg'] + daily['Gross_ATR_kg']
    daily['Purified_Total'] = daily['Purified_SOEC_kg'] + daily['Purified_PEM_kg'] + daily['Purified_ATR_kg']
    
    print(f"\nLoaded {len(df)} timesteps, aggregated to {len(daily)} days")
    print(f"\nGross Production (pre-PSA):")
    print(f"  SOEC: {daily['Gross_SOEC_kg'].mean():,.0f} kg/day")
    print(f"  PEM:  {daily['Gross_PEM_kg'].mean():,.0f} kg/day")
    print(f"  ATR:  {daily['Gross_ATR_kg'].mean():,.0f} kg/day")
    print(f"  ─────────────────")
    print(f"  Total: {daily['Gross_Total'].mean():,.0f} kg/day")
    
    print(f"\nPurified Production (PSA output):")
    print(f"  SOEC: {daily['Purified_SOEC_kg'].mean():,.0f} kg/day")
    print(f"  PEM:  {daily['Purified_PEM_kg'].mean():,.0f} kg/day")
    print(f"  ATR:  {daily['Purified_ATR_kg'].mean():,.0f} kg/day")
    print(f"  ─────────────────")
    print(f"  Total: {daily['Purified_Total'].mean():,.0f} kg/day")
    
    # Calculate recovery rate
    recovery_rate = daily['Purified_Total'].sum() / daily['Gross_Total'].sum() * 100 if daily['Gross_Total'].sum() > 0 else 0
    print(f"\nOverall PSA Recovery Rate: {recovery_rate:.1f}%")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Color scheme
    colors = {
        'PEM': '#3498db',      # Blue
        'SOEC': '#2ecc71',     # Green
        'ATR': '#e74c3c',      # Red
        'Purified': '#9b59b6'  # Purple
    }
    
    # X-axis: days
    days = daily['day'].values
    
    # Stacked bar chart for gross production
    bar_width = 0.8
    
    # Bottom layer: SOEC
    bars_soec = ax.bar(days, daily['Gross_SOEC_kg'], 
                       width=bar_width, 
                       label='SOEC (Gross)', 
                       color=colors['SOEC'],
                       alpha=0.8)
    
    # Middle layer: PEM (stacked on SOEC)
    bars_pem = ax.bar(days, daily['Gross_PEM_kg'], 
                      width=bar_width, 
                      bottom=daily['Gross_SOEC_kg'],
                      label='PEM (Gross)', 
                      color=colors['PEM'],
                      alpha=0.8)
    
    # Top layer: ATR (stacked on SOEC + PEM)
    bars_atr = ax.bar(days, daily['Gross_ATR_kg'], 
                      width=bar_width, 
                      bottom=daily['Gross_SOEC_kg'] + daily['Gross_PEM_kg'],
                      label='ATR (Gross)', 
                      color=colors['ATR'],
                      alpha=0.8)
    
    # Line overlay for purified production
    ax.plot(days, daily['Purified_Total'], 
            color=colors['Purified'], 
            linewidth=2.5, 
            linestyle='-',
            marker='',
            label='Purified Total (PSA Output)',
            zorder=5)
    
    # Formatting
    ax.set_xlabel('Day of Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hydrogen Production (kg/day)', fontsize=12, fontweight='bold')
    ax.set_title('Daily Hydrogen Production by Source\n(Gross Production Breakdown + Purified Total)', 
                 fontsize=14, fontweight='bold')
    
    # Legend - positioned at lower right to avoid overlap with high purified values
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # X-axis ticks (show every 30 days for readability)
    max_day = int(days.max())
    ax.set_xticks(np.arange(0, max_day + 1, 30))
    ax.set_xlim(-1, max_day + 1)
    
    # Add summary statistics as text box
    avg_soec = daily['Gross_SOEC_kg'].mean()
    avg_pem = daily['Gross_PEM_kg'].mean()
    avg_atr = daily['Gross_ATR_kg'].mean()
    avg_gross = daily['Gross_Total'].mean()
    avg_purified = daily['Purified_Total'].mean()
    
    # Calculate equivalent nominal days (based on configured nominal capacity)
    total_purified_kg = daily['Purified_Total'].sum()
    equiv_days = total_purified_kg / nominal_daily_production_kg
    num_sim_days = len(daily)
    capacity_factor = (avg_purified / nominal_daily_production_kg) * 100 if avg_purified > 0 else 0
    
    stats_text = (
        f"Average Daily Production:\n"
        f"─────────────────────\n"
        f"SOEC:     {avg_soec:,.0f} kg/day\n"
        f"PEM:      {avg_pem:,.0f} kg/day\n"
        f"ATR:      {avg_atr:,.0f} kg/day\n"
        f"─────────────────────\n"
        f"Gross:    {avg_gross:,.0f} kg/day\n"
        f"Purified: {avg_purified:,.0f} kg/day\n"
        f"Recovery: {avg_purified/avg_gross*100:.1f}%\n"
        f"─────────────────────\n"
        f"Eq. Nominal Days: {equiv_days:.1f}\n"
        f"Capacity Factor:  {capacity_factor:.1f}%"
    )
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Graph saved to: {output_file}")
    
    # Also save as PDF for high quality
    pdf_path = output_file.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close()
    
    return daily

if __name__ == "__main__":
    generate_daily_h2_production_graph()
