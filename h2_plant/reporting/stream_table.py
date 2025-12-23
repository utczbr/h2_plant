"""
Stream Summary Table Generator.

Provides a reusable function for printing enhanced stream summary tables
with section grouping, connection arrows, phase indicators, and compact formatting.

Enhanced with:
- Dual-unit formatting: % (ppm) for trace species
- Enthalpy (H) and Density (ρ) columns
- 3-decimal precision for H2 purity
"""

from typing import Dict, List, Any, Optional


# Molecular weights (kg/mol)
MW_SPECIES = {
    'H2': 2.016e-3,
    'O2': 32.0e-3,
    'H2O': 18.015e-3,
    'H2O_liq': 18.015e-3,
    'N2': 28.0e-3,
    'CO2': 44.0e-3,
    'CH4': 16.0e-3,
}


def mass_frac_to_mole_frac(composition: dict) -> dict:
    """
    Convert mass fractions to mole fractions.
    
    Args:
        composition: Dictionary of {species: mass_fraction}
        
    Returns:
        Dictionary of {species: mole_fraction}
    """
    n_species = {}
    n_total = 0.0
    for species, mass_frac in composition.items():
        mw = MW_SPECIES.get(species, 28.0e-3)
        if mass_frac > 0 and mw > 0:
            n = mass_frac / mw
            n_species[species] = n
            n_total += n
    
    if n_total > 0:
        return {s: n / n_total for s, n in n_species.items()}
    return composition.copy()


def _format_species_dual(mole_frac: float, mass_kg_h: float, precision: int = 2) -> str:
    """
    Format species with dual-unit display (% and ppm for trace).
    
    - Major components (>=1%): "XX.XX%"
    - Trace components (<1%): "X.XXX% (XXXX ppm)"
    - Negligible (<0.1 ppm): "—"
    
    Args:
        mole_frac: Mole fraction (0-1)
        mass_kg_h: Mass flow rate for this species (kg/h)
        precision: Decimal places for percentage
        
    Returns:
        Formatted string with dual units for trace species.
    """
    if mole_frac < 1e-7:  # < 0.1 ppm
        return "—"
    
    ppm = mole_frac * 1e6
    pct = mole_frac * 100
    
    if mole_frac < 0.01:  # < 1% -> show dual format
        return f"{pct:.3f}% ({ppm:.0f} ppm)"
    else:
        return f"{pct:.{precision}f}%"


def _get_phase_abbrev(stream) -> str:
    """Determine abbreviated phase string (G/L/M) from stream composition."""
    liq_frac = stream.composition.get('H2O_liq', 0.0)
    if liq_frac < 0.01:
        return "G"
    elif liq_frac > 0.99:
        return "L"
    else:
        vap_pct = int((1.0 - liq_frac) * 100)
        return f"M{vap_pct}V"


# Section grouping patterns
SECTION_PATTERNS = [
    ("=== SOEC & Mixing ===", ["SOEC", "Mixer", "Interchanger"]),
    ("=== Separation Train ===", ["KOD", "Chiller", "DryCooler", "Dry_Cooler", "Coalescer"]),
    ("=== Purification ===", ["Deoxo", "PSA", "Boiler"]),
    ("=== Compression Train ===", ["Compressor", "Intercooler"]),
    ("=== Water Recirculation ===", ["Drain", "Makeup", "Pump", "Steam_Generator"]),
]


def _get_section(comp_id: str) -> str:
    """Get section name for a component based on naming patterns."""
    for section_name, patterns in SECTION_PATTERNS:
        if any(p.lower() in comp_id.lower() for p in patterns):
            return section_name
    return "=== Other ==="


def print_stream_summary_table(
    components: Dict[str, Any],
    topo_order: List[str],
    connection_map: Optional[Dict[str, List[str]]] = None
) -> List[Dict[str, Any]]:
    """
    Print an enhanced stream summary table with section grouping and connection arrows.
    
    Columns: Component, T(°C), P(bar), Ph, ṁ(kg/h), H(kJ/kg), ρ(kg/m³), H2(Mol%), H2O, O2, → Next
    
    Args:
        components: Dictionary of component_id -> component instance.
        topo_order: List of component IDs in topology order.
        connection_map: Optional dict of source_id -> list of target_ids.
            If None, connection arrows will show "—".
    
    Returns:
        List of profile data dictionaries for further processing (e.g., graphing).
    """
    if connection_map is None:
        connection_map = {}
    
    # Print header (160 chars wide to accommodate new columns)
    TABLE_WIDTH = 175
    print(f"\n### Stream Summary Table (Topology Order) - TOTAL MOLAR (Vapor + Liquid)")
    print("─" * TABLE_WIDTH)
    header = (
        f"{'Component':<20} │ {'T(°C)':>6} │ {'P(bar)':>6} │ {'Ph':>5} │ "
        f"{'ṁ(kg/h)':>8} │ {'H(kJ/kg)':>9} │ {'ρ(kg/m³)':>9} │ "
        f"{'H2 (Mol%)':>18} │ {'H2O':>18} │ {'O2':>18} │ {'→ Next':<15}"
    )
    print(header)
    print("─" * TABLE_WIDTH)
    
    profile_data = []
    current_section = None
    
    for comp_id in topo_order:
        comp = components.get(comp_id)
        if not comp:
            continue
        
        # Try to get output stream
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
        
        # Section header
        section = _get_section(comp_id)
        if section != current_section:
            current_section = section
            print(f"\n{section}")
            print("─" * TABLE_WIDTH)
        
        T_c = stream.temperature_k - 273.15
        P_bar = stream.pressure_pa / 1e5
        phase_str = _get_phase_abbrev(stream)
        mass_flow = stream.mass_flow_kg_h
        
        # Thermodynamic properties
        try:
            h_kj_kg = stream.specific_enthalpy_j_kg / 1000.0
        except:
            h_kj_kg = 0.0
        
        try:
            rho_kg_m3 = stream.density_kg_m3
        except:
            rho_kg_m3 = 0.0
        
        # Extract mass fractions
        h2_frac = stream.composition.get('H2', 0.0)
        h2o_frac = stream.composition.get('H2O', 0.0) + stream.composition.get('H2O_liq', 0.0)
        o2_frac = stream.composition.get('O2', 0.0)
        
        # Convert to mole fractions
        mole_fracs = mass_frac_to_mole_frac(stream.composition)
        h2_mol = mole_fracs.get('H2', 0.0)
        h2o_mol = mole_fracs.get('H2O', 0.0) + mole_fracs.get('H2O_liq', 0.0)
        o2_mol = mole_fracs.get('O2', 0.0)
        
        # Format species with dual-unit for trace, 3-decimal for H2
        h2_str = _format_species_dual(h2_mol, h2_frac * mass_flow, precision=3)
        h2o_str = _format_species_dual(h2o_mol, h2o_frac * mass_flow)
        o2_str = _format_species_dual(o2_mol, o2_frac * mass_flow)
        
        # Get connection target
        targets = connection_map.get(comp_id, [])
        next_comp = targets[0] if targets else "—"
        if len(next_comp) > 15:
            next_comp = next_comp[:12] + "..."
        
        # Store profile data for graphing
        try:
            s_val = stream.specific_entropy_j_kgK / 1000.0
        except:
            s_val = 0.0
        
        profile_data.append({
            'Component': comp_id,
            'T_c': T_c,
            'P_bar': P_bar,
            'H_kj_kg': h_kj_kg,
            'S_kj_kgK': s_val,
            'rho_kg_m3': rho_kg_m3,
            'MolFrac_H2': h2_mol,
            'MolFrac_O2': o2_mol,
            'MolFrac_H2O': h2o_mol,
            'MassFrac_H2': h2_frac,
            'MassFrac_O2': o2_frac,
            'MassFrac_H2O': h2o_frac
        })
        
        # Print row
        row = (
            f"{comp_id:<20} │ {T_c:>5.1f}° │ {P_bar:>6.2f} │ {phase_str:>5} │ "
            f"{mass_flow:>8.2f} │ {h_kj_kg:>9.2f} │ {rho_kg_m3:>9.4f} │ "
            f"{h2_str:>18} │ {h2o_str:>18} │ {o2_str:>18} │ → {next_comp:<13}"
        )
        print(row)
    
    print("─" * TABLE_WIDTH)
    
    return profile_data
