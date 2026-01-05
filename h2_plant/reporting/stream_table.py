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
    'CH4': 16.04e-3,
    'CO': 28.01e-3,
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
    Determine the topology section index (1-7) for sorting.
    
    Section 1: Feed & Upstream (SOEC)
    Section 2: Feed & Upstream (PEM)
    Section 3: SOEC H2 Output Train (Cathode)
    Section 4: SOEC O2 Output Train (Anode)
    Section 5: PEM H2 Output Train (Cathode)
    Section 6: PEM O2 Output Train (Anode)
    Section 7: Storage & Distribution
    """
    cid = comp_id.upper()
    ctype = comp_type.upper()
    
    # 0. ATR / WGS / Reformer (Highest Priority for this unit)
    if any(k in cid for k in ["ATR", "WGS", "SHIFT", "REFORM", "BIOGAS"]): return 8
    
    # 1. Storage & Distribution (High Priority)
    if "HP_" in cid: return 7
    if any(k in cid for k in ["STORAGE", "TANK", "GRID", "CONSUMER"]): return 7
    if any(k in ctype for k in ["TANK", "STORAGE"]): return 7
    
    # 2. Specific Train Prefixes (Explicit O2/H2 Paths)
    # Check O2 first to distinguish from generic SOEC/PEM
    if "PEM_O2_" in cid: return 6
    if "SOEC_O2_" in cid: return 4
    if "O2_" in cid: 
        return 6 if "PEM" in cid else 4

    if "PEM_H2_" in cid: return 5
    if "SOEC_H2_" in cid: return 3

    # 3. Upstream / Feed / Recirculation
    # Matches Feed_Pump, Makeup_Mixer, Drain_Mixer, Steam_Generator
    upstream_keywords = ["FEED", "PUMP", "SOURCE", "WATER", "STEAM", "MAKEUP", "DRAIN", "MIXER"]
    if any(x in cid for x in upstream_keywords):
        if "PEM" in cid: return 2
        return 1

    # 4. Core Units & Fallbacks (Generic)
    if "PEM" in cid: return 5
    if "SOEC" in cid: return 3

    # Default to Section 1 if nothing matches (or 3 if it looks like main process)
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
    7: "=== Section 7: Storage & Distribution ===",
    8: "=== Section 8: ATR Reforming & WGS Unit ==="
}

def print_stream_summary_table(
    components: Dict[str, Any],
    topo_order: List[str]
) -> None:
    """
    Print consolidated stream summary table.
    """
    # Group components by section
    sections = {i: [] for i in range(1, 9)}
    
    # Sort components into sections based on ID/Type rules
    for cid in topo_order:
        comp = components.get(cid)
        if not comp: continue
        
        # Get component output stream - prioritize by gas species mass (H2 + O2 + CH4)
        candidate_streams = []
        priority_ports = ['h2_out', 'syngas_out', 'purified_gas_out', 'hydrogen']
        all_ports = ['outlet', 'h2_out', 'syngas_out', 'o2_out', 'fluid_out', 'gas_outlet', 'purified_gas_out', 'hot_out', 'water_out', 'steam_out', 'liquid_drain', 'drain', 'out', 'biogas_out', 'cold_out', 'hydrogen', 'offgas', 'water']
        
        for port in all_ports:
            try:
                s = comp.get_output(port)
                if s and s.mass_flow_kg_h > 1e-6:
                    gas_mass_frac = (
                        s.composition.get('H2', 0.0) + 
                        s.composition.get('O2', 0.0) + 
                        s.composition.get('CH4', 0.0) +
                        s.composition.get('CO', 0.0)
                    )
                    priority_boost = 1000.0 if port in priority_ports else 1.0
                    gas_mass_kg_h = s.mass_flow_kg_h * gas_mass_frac * priority_boost
                    candidate_streams.append((gas_mass_kg_h, s))
            except: pass
        
        # Select stream with highest gas species mass
        if candidate_streams:
            candidate_streams.sort(key=lambda x: x[0], reverse=True)
            stream = candidate_streams[0][1]
        else:
            stream = None
            
        if not stream: continue
        
        sec_idx = _get_topology_section(cid, type(comp).__name__)
        sections[sec_idx].append((cid, stream))

    # Print by section
    for i in range(1, 9):
        comps = sections[i]
        if not comps: continue
        
        print(f"\n{SECTION_HEADERS[i]}")
        
        # Determine format based on section
        # Format A: Sections 1-7 (Electroylsis/Compression)
        # Format B: Section 8 (ATR)
        is_atr = (i == 8)
        
        if is_atr:
            # Format B Header: Component | T_out | P_out | H2% | H2O% | O2% | CH4 | CO | CO2 | N2 | Total kg/h
            print("-" * 145)
            print(f"{'Component':<20} | {'T_out':>7} | {'P_out':>9} | {'H2%':>7} | {'H2O%':>7} | {'O2%':>7} | {'CH4':>7} | {'CO':>7} | {'CO2':>7} | {'N2':>7} | {'Total kg/h':>10}")
            print("-" * 145)
        else:
            # Format A Header: Component | T_out | P_out | H2% | H2 kg/h | H2O% | H2O kg/h | O2% | O2 kg/h | Total | H2O %Liq | H2O %Vap
            print("-" * 155)
            print(f"{'Component':<20} | {'T_out':>7} | {'P_out':>9} | {'H2%':>7} | {'H2 kg/h':>9} | {'H2O%':>7} | {'H2O kg/h':>9} | {'O2%':>7} | {'O2 kg/h':>9} | {'Total':>10} | {'%H2O(L)':>7} | {'%H2O(V)':>7}")
            print("-" * 155)
        
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
            mol_h2o_total = mole_fracs.get('H2O', 0.0) + mole_fracs.get('H2O_liq', 0.0)
            
            # 4. Phase Partitioning of Water
            # Calculate % of WATER that is Liquid vs Vapor
            if mass_h2o_total > 1e-9:
                pct_h2o_liq = (mass_h2o_liq / mass_h2o_total) * 100.0
            else:
                pct_h2o_liq = 0.0
            pct_h2o_vap = 100.0 - pct_h2o_liq
            
            # Formatting helpers
            h2_str = _format_ppm_pct(mol_h2)
            h2o_str = _format_ppm_pct(mol_h2o_total)
            o2_str = _format_ppm_pct(mol_o2)
            
            if is_atr:
                # Format B Rows
                ch4_str = _format_ppm_pct(mole_fracs.get('CH4', 0.0))
                co_str = _format_ppm_pct(mole_fracs.get('CO', 0.0))
                co2_str = _format_ppm_pct(mole_fracs.get('CO2', 0.0))
                n2_str = _format_ppm_pct(mole_fracs.get('N2', 0.0))
                
                print(f"{cid:<20} | {T_c:>5.1f}°C | {P_bar:>5.2f} bar | {h2_str:>7} | {h2o_str:>7} | {o2_str:>7} | {ch4_str:>7} | {co_str:>7} | {co2_str:>7} | {n2_str:>7} | {total_kg_h:>10.2f}")
            
            else:
                # Format A Rows
                # Component | T_out | P_out | H2% | H2 kg/h | H2O% | H2O kg/h | O2% | O2 kg/h | Total | H2O %Liq | H2O %Vap
                
                print(f"{cid:<20} | {T_c:>5.1f}°C | {P_bar:>5.2f} bar | {h2_str:>7} | {kg_h_h2:>9.2f} | {h2o_str:>7} | {kg_h_h2o:>9.2f} | {o2_str:>7} | {kg_h_o2:>9.2f} | {total_kg_h:>10.2f} | {pct_h2o_liq:>7.1f}% | {pct_h2o_vap:>7.1f}%")
        
        if is_atr:
             print("-" * 145)
        else:
             print("-" * 155)
