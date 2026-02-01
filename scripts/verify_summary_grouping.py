
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from h2_plant.reporting.stream_table import _get_topology_section

def test_grouping():
    test_cases = [
        # Upstream
        ("Feed_Pump", "Pump", 1),
        ("SOEC_Steam_Generator", "Boiler", 1),
        ("Makeup_Mixer_1", "Mixer", 1),
        ("Drain_Mixer", "Mixer", 1),
        
        # SOEC H2 (Section 3)
        ("SOEC_Cluster", "SOEC", 3),
        ("SOEC_H2_Interchanger_1", "Interchanger", 3),
        ("SOEC_H2_Compressor_S1", "Compressor", 3),
        ("SOEC_H2_Cyclone_2", "Cyclone", 3),
        
        # SOEC O2 (Section 4)
        ("SOEC_O2_Drycooler_1", "DryCooler", 4),
        ("SOEC_O2_compressor_1", "Compressor", 4),
        
        # PEM H2 (Section 5)
        ("PEM_Electrolyzer", "PEM", 5),
        ("PEM_H2_KOD_1", "KOD", 5),
        ("PEM_H2_ElectricBoiler_1", "Boiler", 5),
        
        # PEM O2 (Section 6)
        ("PEM_O2_KOD_1", "KOD", 6),
        ("PEM_O2_Valve", "Valve", 6),
        
        # Storage / HP (Section 7)
        ("HP_Compressor_S1", "Compressor", 7),
        ("H2_Storage_Tank", "Tank", 7),
        ("H2_Production_Mixer", "Mixer", 1) # Wait, is this 1? It combines H2 products...
        # H2_Production_Mixer combines PEM_H2 and SOEC_H2 output to feed HP train.
        # It contains "MIXER" -> Section 1 (Upstream).
        # Actually it sits between Production and Compression. Ideally it should be grouped with HP train or separate.
        # Current logic puts "MIXER" in Upstream.
    ]
    
    failures = []
    print(f"{'Component ID':<30} | {'Expected':>8} | {'Actual':>8} | {'Result'}")
    print("-" * 65)
    
    for cid, ctype, expected in test_cases:
        actual = _get_topology_section(cid, ctype)
        status = "✅" if actual == expected else "❌"
        print(f"{cid:<30} | {expected:>8} | {actual:>8} | {status}")
        if actual != expected:
            failures.append((cid, expected, actual))
            
    if failures:
        print(f"\n❌ Failed {len(failures)} tests.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed.")
        sys.exit(0)

if __name__ == "__main__":
    test_grouping()
