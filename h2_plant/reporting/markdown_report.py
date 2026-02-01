"""
Structured Markdown Report Generator.

Generates a progressive 3-tier simulation report:
1. System Totals - Executive KPIs
2. Component Operational Logs - Performance by category
3. Detailed Stream Topology - Full thermodynamic state

Design Rationale:
    - Managers see KPIs immediately without scrolling
    - Engineers find performance warnings in Section 2
    - Thermodynamic details in Section 3 for debugging
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from h2_plant.reporting.stream_table import (
    mass_frac_to_mole_frac,
    _format_species_dual,
    _get_phase_abbrev,
    _get_section,
)

# Component type detection patterns
COMPONENT_PATTERNS = {
    'knock_out_drum': ['KOD', 'Knock'],
    'coalescer': ['Coalescer'],
    'chiller': ['Chiller'],
    'dry_cooler': ['DryCooler', 'Dry_Cooler', 'Intercooler'],
    'compressor': ['Compressor'],
    'deoxo': ['Deoxo'],
    'psa': ['PSA', 'VSA'],
    'soec': ['SOEC'],
    'pem': ['PEM'],
    'tank': ['Tank'],
    'steam_gen': ['Steam_Generator', 'SteamGen'],
}


def _categorize_components(components: Dict[str, Any]) -> Dict[str, List[tuple]]:
    """Categorize components by type for sectioned reporting."""
    categories = {k: [] for k in COMPONENT_PATTERNS.keys()}
    categories['other'] = []
    
    for comp_id, comp in components.items():
        matched = False
        for cat, patterns in COMPONENT_PATTERNS.items():
            if any(p.lower() in comp_id.lower() for p in patterns):
                categories[cat].append((comp_id, comp))
                matched = True
                break
        if not matched:
            categories['other'].append((comp_id, comp))
    
    return categories


def _section_system_totals(
    history: Dict[str, np.ndarray],
    duration_hours: float,
    components: Dict[str, Any]
) -> str:
    """Generate Section 1: System Totals."""
    lines = []
    lines.append("## 1. System Totals")
    lines.append("| Metric | Value | Unit |")
    lines.append("| :--- | ---: | :--- |")
    
    # Total H2 produced
    h2_total = np.sum(history.get('h2_kg', [0]))
    lines.append(f"| **Total H2 Produced** | **{h2_total:,.2f}** | kg |")
    
    # Total water removed (sum from KODs, Coalescers, Chillers, PSA)
    water_total = 0.0
    for comp_id, comp in components.items():
        if hasattr(comp, 'get_state'):
            state = comp.get_state()
            for key in ['water_removed_kg_h', 'drain_flow_kg_h', 'liquid_removed_kg_h']:
                if key in state:
                    water_total += state[key] * duration_hours
    lines.append(f"| Total Water Removed | {water_total:,.2f} | kg |")
    
    # Energy consumed (SOEC + PEM + BoP)
    p_soec = np.sum(history.get('P_soec_actual', [0])) * (duration_hours / len(history.get('P_soec_actual', [1])))
    p_pem = np.sum(history.get('P_pem', [0])) * (duration_hours / max(1, len(history.get('P_pem', [1]))))
    p_bop = np.sum(history.get('P_bop_mw', [0])) * (duration_hours / max(1, len(history.get('P_bop_mw', [1]))))
    energy_total = p_soec + p_pem + p_bop
    lines.append(f"| Total Energy Consumed | {energy_total:,.2f} | MWh |")
    
    # Energy sold
    p_sold = np.sum(history.get('P_sold', [0])) * (duration_hours / max(1, len(history.get('P_sold', [1]))))
    lines.append(f"| Energy Sold to Market | {p_sold:,.2f} | MWh |")
    
    lines.append("")
    return "\n".join(lines)


def _section_component_logs(
    components: Dict[str, Any],
    duration_hours: float
) -> str:
    """Generate Section 2: Component Operational Logs."""
    lines = []
    lines.append("## 2. Component Operational Logs")
    lines.append("*Performance metrics and physical deltas (Mass/Energy transfer).*")
    lines.append("")
    
    categories = _categorize_components(components)
    
    # 2.1 Separation Units
    sep_units = categories['knock_out_drum'] + categories['coalescer'] + categories['psa']
    if sep_units:
        lines.append("### 2.1 Separation Units (Water Knock-out)")
        lines.append("| Unit ID | Water Removed (kg) | Velocity Margin | Status |")
        lines.append("| :--- | ---: | :--- | :--- |")
        
        for comp_id, comp in sep_units:
            state = comp.get_state() if hasattr(comp, 'get_state') else {}
            water_removed = state.get('water_removed_kg_h', state.get('drain_flow_kg_h', 0.0)) * duration_hours
            v_real = state.get('v_real', 0.0)
            v_max = state.get('v_max', 1.0)
            margin = ((v_max - v_real) / v_max * 100) if v_max > 0 else 0
            margin_str = f"**{margin:.1f}%** ({v_real:.2f} < {v_max:.2f} m/s)" if v_max > 0.01 else "N/A"
            status = "✅ OK" if state.get('separation_status', 'OK') == 'OK' else "⚠️ WARN"
            lines.append(f"| **{comp_id}** | {water_removed:,.2f} | {margin_str} | {status} |")
        lines.append("")
    
    # 2.2 Compression & Purification
    comp_units = categories['compressor'] + categories['deoxo']
    if comp_units:
        lines.append("### 2.2 Compression & Purification Stats")
        lines.append("| Unit ID | Power (kW) | Energy (kWh) | Efficiency | Notes |")
        lines.append("| :--- | ---: | ---: | ---: | :--- |")
        
        for comp_id, comp in comp_units:
            state = comp.get_state() if hasattr(comp, 'get_state') else {}
            power_kw = state.get('power_kw', state.get('shaft_power_kw', 0.0))
            energy_kwh = power_kw * duration_hours
            eta = state.get('isentropic_efficiency', state.get('efficiency', 0.0)) * 100
            notes = ""
            if 'Deoxo' in comp_id:
                o2_conv = state.get('o2_conversion', 1.0) * 100
                notes = f"O2 Conv: {o2_conv:.0f}%"
            lines.append(f"| **{comp_id}** | {power_kw:.2f} | {energy_kwh:,.0f} | {eta:.1f}% | {notes} |")
        lines.append("")
    
    # 2.3 Thermal Management
    thermal_units = categories['chiller'] + categories['dry_cooler']
    if thermal_units:
        lines.append("### 2.3 Thermal Management")
        lines.append("| Unit ID | Duty (kW) | Elec. (kW) | Type |")
        lines.append("| :--- | ---: | ---: | :--- |")
        
        for comp_id, comp in thermal_units:
            state = comp.get_state() if hasattr(comp, 'get_state') else {}
            duty_kw = state.get('cooling_load_kw', state.get('tqc_duty_kw', 0.0))
            elec_kw = state.get('electrical_power_kw', state.get('fan_power_kw', 0.0))
            unit_type = "Active" if 'Chiller' in comp_id else "Passive"
            lines.append(f"| **{comp_id}** | {duty_kw:.2f} | {elec_kw:.2f} | {unit_type} |")
        lines.append("")
    
    return "\n".join(lines)


def _section_stream_topology(
    components: Dict[str, Any],
    topo_order: List[str],
    connection_map: Optional[Dict[str, List[str]]] = None
) -> str:
    """Generate Section 3: Detailed Stream Topology."""
    if connection_map is None:
        connection_map = {}
    
    lines = []
    lines.append("## 3. Detailed Stream Topology (Gas State)")
    lines.append("*Comprehensive thermodynamic state at the outlet of each component.*")
    lines.append("")
    lines.append("| Component | T (°C) | P (bar) | ṁ (kg/h) | Ph | H (kJ/kg) | ρ (kg/m³) | H2 (Mol%) | H2O | O2 |")
    lines.append("| :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |")
    
    for comp_id in topo_order:
        comp = components.get(comp_id)
        if not comp:
            continue
        
        # Get output stream
        stream = None
        for port in ['outlet', 'h2_out', 'o2_out', 'fluid_out', 'gas_outlet', 'purified_gas_out', 'hot_out']:
            try:
                stream = comp.get_output(port)
                if stream is not None and stream.mass_flow_kg_h > 1e-6:
                    break
                stream = None
            except:
                pass
        
        if stream is None:
            continue
        
        T_c = stream.temperature_k - 273.15
        P_bar = stream.pressure_pa / 1e5
        phase_str = _get_phase_abbrev(stream)
        mass_flow = stream.mass_flow_kg_h
        
        try:
            h_kj = stream.specific_enthalpy_j_kg / 1000.0
        except:
            h_kj = 0.0
        
        try:
            rho = stream.density_kg_m3
        except:
            rho = 0.0
        
        # Mole fractions
        mole_fracs = mass_frac_to_mole_frac(stream.composition)
        h2_mol = mole_fracs.get('H2', 0.0)
        h2o_mol = mole_fracs.get('H2O', 0.0) + mole_fracs.get('H2O_liq', 0.0)
        o2_mol = mole_fracs.get('O2', 0.0)
        
        # Format with dual-unit
        h2_frac = stream.composition.get('H2', 0.0)
        h2o_frac = stream.composition.get('H2O', 0.0) + stream.composition.get('H2O_liq', 0.0)
        o2_frac = stream.composition.get('O2', 0.0)
        
        h2_str = _format_species_dual(h2_mol, h2_frac * mass_flow, precision=3)
        h2o_str = _format_species_dual(h2o_mol, h2o_frac * mass_flow)
        o2_str = _format_species_dual(o2_mol, o2_frac * mass_flow)
        
        lines.append(f"| **{comp_id}** | {T_c:.1f} | {P_bar:.2f} | {mass_flow:.2f} | {phase_str} | {h_kj:.1f} | {rho:.3f} | {h2_str} | {h2o_str} | {o2_str} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_simulation_report(
    components: Dict[str, Any],
    topo_order: List[str],
    connection_map: Dict[str, List[str]],
    history: Dict[str, np.ndarray],
    duration_hours: float,
    output_path: Path,
    topology_name: str = "SOEC Hydrogen Production"
) -> Path:
    """
    Generate a structured markdown simulation report.
    
    Args:
        components: Dictionary of component_id -> component instance.
        topo_order: List of component IDs in topology order.
        connection_map: Dict of source_id -> list of target_ids.
        history: Simulation history arrays.
        duration_hours: Total simulation duration in hours.
        output_path: Path to save the report.
        topology_name: Name for the report header.
    
    Returns:
        Path to the generated report file.
    """
    lines = []
    
    # Header
    date_str = datetime.now().strftime("%Y-%m-%d")
    steps = len(history.get('minute', [0]))
    lines.append(f"# Simulation Report: {topology_name}")
    lines.append(f"**Date:** {date_str} | **Duration:** {duration_hours:.1f} hrs | **Steps:** {steps}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Section 1: System Totals
    lines.append(_section_system_totals(history, duration_hours, components))
    lines.append("---")
    lines.append("")
    
    # Section 2: Component Operational Logs
    lines.append(_section_component_logs(components, duration_hours))
    lines.append("---")
    lines.append("")
    
    # Section 3: Detailed Stream Topology
    lines.append(_section_stream_topology(components, topo_order, connection_map))
    
    # Write to file
    report_content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding='utf-8')
    
    return output_path
