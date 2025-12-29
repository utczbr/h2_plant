"""
Stream Summary Table Generator.

Refactored to provide a single, high-density summary table grouped by plant topology sections.
Features:
- 6-Section Topology Grouping
- Dual-unit formatting (% and ppm)
- Total H2O (Vapor + Liquid) calculation
- Phase determination (%Liq, %Vap)
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
    """Convert mass fractions to mole fractions."""
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

def _format_ppm_pct(mole_frac: float, precision: int = 4) -> str:
    """
    Format as % if >= 0.01% (100 ppm), else as ppm.
    """
    if mole_frac < 1e-9:
        return "0 ppm"
    
    pct = mole_frac * 100.0
    if pct >= 0.01:
        return f"{pct:.{precision}f}%"
    else:
        ppm = mole_frac * 1e6
        return f"{ppm:.0f} ppm"

def _format_species_dual(mole_frac: float, mass_kg_h: float = 0.0, precision: int = 4) -> str:
    """
    Backward compatibility wrapper for _format_ppm_pct.
    Ignores mass_kg_h as the new table doesn't use it for formatting decisions.
    """
    return _format_ppm_pct(mole_frac, precision)

def _get_topology_section(comp_id: str, comp_type: str) -> int:
    """
    Determine the topology section index (1-6) for sorting.
    
    Section 1: Components upstream of the SOEC (Feed -> SOEC).
    Section 2: Components upstream of the PEM (Feed -> PEM).
    Section 3: SOEC H2 Output Train (Cathode side).
    Section 4: SOEC O2 Output Train (Anode side).
    Section 5: PEM H2 Output Train.
    Section 6: PEM O2 Output Train.
    """
    cid = comp_id.upper()
    ctype = comp_type.upper()
    
    # Storage & Distribution (Tanks + High Pressure Train)
    if any(k in cid for k in ["STORAGE", "TANK", "GRID", "CONSUMER"]) or "HP_" in cid:
        return 7
    if any(k in ctype for k in ["TANK", "STORAGE"]):
        return 7
    
    # Electrolyzers are the boundary lines
    if "SOEC" in cid:
        return 3 # Start of H2 Train (or section 1 end, but put in 3 for visibility)
    if "PEM" in cid:
        return 5

    # O2 Trains (Anode)
    if "O2_" in cid or ("O2" in cid and "COMPRESSOR" in cid):
        # Determine if SOEC or PEM O2
        # Heuristic: If we had a PEM named 'PEM_1', associated O2 might be 'PEM_O2_...'
        # For now, default O2 components to SOEC Anode (Section 4) unless marked PEM
        if "PEM" in cid:
            return 6
        return 4

    # Upstream / Feed
    if any(x in cid for x in ["FEED", "PUMP", "SOURCE", "WATER", "STEAM"]):
        # Heuristic: Steam/Boiler is usually SOEC upstream
        if "KAOWOOL" in cid or "PEM" in cid:
            return 2
        return 1

    # H2 Trains (Cathode) - Default for Compressors, KODs, Chillers not marked O2
    # Determine if PEM or SOEC H2
    if "PEM" in cid:
        return 5
    
    # Default to SOEC H2 Train (Section 3) for standard compressors/intercoolers
    return 3

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

def _get_section(comp_id: str) -> str:
    """
    Determine the display section string for a component.
    Backward compatibility wrapper for _get_topology_section.
    """
    # Heuristic type guess since we only have ID here
    # This is imperfect but satisfies the legacy interface
    comp_type = "Unknown"
    if "SOEC" in comp_id: comp_type = "SOEC"
    elif "PEM" in comp_id: comp_type = "PEM"
    
    sec_idx = _get_topology_section(comp_id, comp_type)
    return SECTION_HEADERS.get(sec_idx, "=== Other / General ===")

SECTION_HEADERS = {
    1: "=== Section 1: Feed & Upstream (SOEC) ===",
    2: "=== Section 2: Feed & Upstream (PEM) ===",
    3: "=== Section 3: SOEC H2 Train (Cathode) ===",
    4: "=== Section 4: SOEC O2 Train (Anode) ===",
    5: "=== Section 5: PEM H2 Train (Cathode) ===",
    6: "=== Section 6: PEM O2 Train (Anode) ===",
    7: "=== Section 7: Storage & Distribution ==="
}

def print_stream_summary_table(
    components: Dict[str, Any],
    topo_order: List[str]
) -> None:
    """
    Print consolidated stream summary table.
    """
    print("\n" + "="*145)
    print(f"{'Component':<18} | {'T_out':>7} | {'P_out':>9} | {'H2%':>9} | {'H2 kg/h':>8} | {'H2O':>9} | {'H2O kg/h':>9} | {'O2':>9} | {'O2 kg/h':>8} | {'Total':>9} | {'%Liq':>8} | {'%Vap':>8}")
    print("-" * 145)

    # Group components by section
    sections = {i: [] for i in range(1, 8)}
    
    # Sort components into sections based on ID/Type rules
    # We use the passed topo_order to maintain flow order *within* sections
    for cid in topo_order:
        comp = components.get(cid)
        if not comp: continue
        
        # Get component output stream (try common ports)
        stream = None
        for port in ['outlet', 'h2_out', 'o2_out', 'fluid_out', 'gas_outlet', 'purified_gas_out', 'hot_out', 'water_out', 'steam_out']:
            try:
                s = comp.get_output(port)
                if s and s.mass_flow_kg_h > 1e-6:
                    stream = s
                    break
            except: pass
            
        if not stream: continue # Skip components with no flow
        
        sec_idx = _get_topology_section(cid, type(comp).__name__)
        sections[sec_idx].append((cid, stream))

    # Print by section
    for i in range(1, 8):
        comps = sections[i]
        if not comps: continue
        
        print(f"{SECTION_HEADERS[i]}")
        print("-" * 145)
        
        for cid, stream in comps:
            # 1. Properties
            T_c = stream.temperature_k - 273.15
            P_bar = stream.pressure_pa / 1e5
            total_kg_h = stream.mass_flow_kg_h
            
            # 2. Composition (Mass)
            mass_h2o_vap = stream.composition.get('H2O', 0.0)
            mass_h2o_liq = stream.composition.get('H2O_liq', 0.0)
            mass_h2o_total = mass_h2o_vap + mass_h2o_liq
            
            mass_h2 = stream.composition.get('H2', 0.0)
            mass_o2 = stream.composition.get('O2', 0.0)
            
            kg_h_h2 = mass_h2 * total_kg_h
            kg_h_o2 = mass_o2 * total_kg_h
            kg_h_h2o = mass_h2o_total * total_kg_h
            
            # 3. Composition (Molar)
            mole_fracs = mass_frac_to_mole_frac(stream.composition)
            
            mol_h2 = mole_fracs.get('H2', 0.0)
            mol_o2 = mole_fracs.get('O2', 0.0)
            # Total molar water handles vapor+liquid as species
            mol_h2o_vap = mole_fracs.get('H2O', 0.0)
            mol_h2o_liq = mole_fracs.get('H2O_liq', 0.0)
            mol_h2o_total = mol_h2o_vap + mol_h2o_liq
            
            # 4. Phase Fractions (Mass based)
            # %Liq is mass fraction of H2O_liq in total stream (assuming only H2O condenses)
            pct_liq = mass_h2o_liq * 100.0
            pct_vap = 100.0 - pct_liq
            
            # 5. Formatting
            h2_str = _format_ppm_pct(mol_h2)
            h2o_str = _format_ppm_pct(mol_h2o_total)
            o2_str = _format_ppm_pct(mol_o2)
            
            print(f"{cid:<18} | {T_c:>5.1f}°C | {P_bar:>5.2f} bar | {h2_str:>9} | {kg_h_h2:>8.3f} | {h2o_str:>9} | {kg_h_h2o:>9.4f} | {o2_str:>9} | {kg_h_o2:>8.5f} | {total_kg_h:>9.2f} | {pct_liq:>7.3f}% | {pct_vap:>7.3f}%")
        
        print("-" * 145)
