#!/usr/bin/env python3
"""Test script to verify stream table formatting with new dual-unit display."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from h2_plant.reporting.stream_table import print_stream_summary_table


class MockStream:
    """Mock Stream for testing."""
    def __init__(self, composition, mass_flow, t_k, p_pa, h_j_kg=0, rho=0.1):
        self.composition = composition
        self.mass_flow_kg_h = mass_flow
        self.temperature_k = t_k
        self.pressure_pa = p_pa
        self.specific_enthalpy_j_kg = h_j_kg
        self.specific_entropy_j_kgK = 0
        self._density = rho

    @property
    def density_kg_m3(self):
        return self._density


class MockComponent:
    """Mock Component for testing."""
    def __init__(self, output):
        self.output = output

    def get_output(self, port):
        return self.output


def main():
    """Create test streams matching user's example data and display table."""
    
    # Scenario 1: PSA output - ultra-high purity H2 (11 ppm H2O)
    # Mass fractions calculated from 11 molar ppm H2O
    # H2 = 99.9902% mass, H2O = 0.0098% mass
    psa_stream = MockStream(
        composition={'H2': 0.999902, 'H2O': 0.000098, 'O2': 0.0},
        mass_flow=214.09,
        t_k=312.95,  # 39.8°C
        p_pa=3766000,  # 37.66 bar
        h_j_kg=211573,  # ~212 kJ/kg
        rho=2.918
    )
    
    # Scenario 2: Coalescer_1 output - moderate purity (586 ppm H2O, 199 ppm O2)
    coalescer_stream = MockStream(
        composition={'H2': 0.9471, 'H2O': 0.0496, 'O2': 0.003007, 'H2O_liq': 0.000361},
        mass_flow=251.29,
        t_k=277.15,  # 4°C
        p_pa=107300,  # 1.07 bar
        h_j_kg=-286809,
        rho=0.0985
    )
    
    # Scenario 3: SOEC Cluster - high temp output (200 ppm O2)
    soec_stream = MockStream(
        composition={'H2': 0.9998, 'H2O': 0.0, 'O2': 0.0002},
        mass_flow=307.20,
        t_k=1073.15,  # 800°C
        p_pa=100000,  # 1 bar
        h_j_kg=3845200,
        rho=0.041
    )
    
    # Scenario 4: Cathode Mixer - significant H2O content
    mixer_stream = MockStream(
        composition={'H2': 0.3987, 'H2O': 0.6, 'O2': 0.001266},
        mass_flow=768.00,
        t_k=1073.15,  # 800°C
        p_pa=100000,
        h_j_kg=2105400,
        rho=0.038
    )
    
    # Scenario 5: Drain_Mixer - pure water
    drain_stream = MockStream(
        composition={'H2': 0.0, 'H2O': 1.0, 'O2': 0.0},
        mass_flow=422.78,
        t_k=296.95,  # 23.8°C
        p_pa=90000,
        h_j_kg=20000,
        rho=1000.0
    )
    
    components = {
        'SOEC_Cluster': MockComponent(soec_stream),
        'Cathode_Mixer': MockComponent(mixer_stream),
        'Coalescer_1': MockComponent(coalescer_stream),
        'PSA_1': MockComponent(psa_stream),
        'Drain_Mixer': MockComponent(drain_stream),
    }
    
    topo = ['SOEC_Cluster', 'Cathode_Mixer', 'Coalescer_1', 'PSA_1', 'Drain_Mixer']
    
    connection_map = {
        'SOEC_Cluster': ['Cathode_Mixer'],
        'Cathode_Mixer': ['Interchanger_1'],
        'Coalescer_1': ['Compressor_S1'],
        'PSA_1': [],
        'Drain_Mixer': ['Makeup_Mixer_1'],
    }
    
    print("=" * 80)
    print("STREAM TABLE VERIFICATION TEST")
    print("Expected: Dual-unit formatting, H/rho columns, abbreviated Phase")
    print("=" * 80)
    
    profile_data = print_stream_summary_table(components, topo, connection_map)
    
    print("\n" + "=" * 80)
    print("PROFILE DATA RETURNED (for graphing):")
    print("=" * 80)
    for row in profile_data:
        print(f"  {row['Component']}: H2={row['MolFrac_H2']*100:.3f}% | H={row['H_kj_kg']:.1f} kJ/kg | rho={row['rho_kg_m3']:.4f} kg/m³")


if __name__ == "__main__":
    main()
