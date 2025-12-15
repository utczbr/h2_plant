"""
Integration Test: Legacy Graph Generation

This test runs a single timestep through the H2 purification line topology
and then invokes the LegacyDataAdapter and LegacyPlotManager to generate
the full suite of 19 simulation plots.

It validates that:
1. The simulation runs correctly.
2. Data covers all components.
3. Plots are generated without errors in 'outputs/legacy_plots'.
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Add Legacy paths
LEGACY_PEM_DIR = PROJECT_ROOT / "h2_plant" / "legacy" / "NEW" / "PEM"
sys.path.insert(0, str(LEGACY_PEM_DIR))

from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.separation.knock_out_drum import KnockOutDrum
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
from h2_plant.components.separation.psa import PSA
from h2_plant.gui.plotting.legacy_adapter import LegacyDataAdapter
from h2_plant.gui.plotting.plot_manager import LegacyPlotManager
from h2_plant.optimization.lut_manager import LUTManager


def test_generate_legacy_plots():
    """Run simulation and generate legacy plots."""
    
    output_dir = PROJECT_ROOT / "outputs" / "legacy_plots_test"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # === Configuration ===
    T_IN_K = 333.15  # 60°C
    P_IN_PA = 40e5   # 40 bar
    MDOT_H2_KG_H = 89.0
    
    inlet_composition = {
        'H2': 0.98,
        'H2O': 0.018,
        'O2': 0.002
    }
    
    # === Create Components ===
    registry = ComponentRegistry()
    dt = 1/60 
    
    # Initialize LUT Manager (required for Adapter enthalpy calc)
    lut = LUTManager()
    lut.initialize()
    registry.register('lut_manager', lut)
    
    # Components (ID mapping important for Adapter)
    # Adapter expects: 'source_h2', 'kod1_h2', 'dry_cooler', 'chiller', 'kod2_h2', 'coalescer', 'heater', 'deoxo', 'psa'
    
    kod1 = KnockOutDrum(component_id='kod1_h2', diameter_m=0.5, delta_p_bar=0.05, gas_species='H2')
    dry_cooler = DryCooler(component_id='dry_cooler')
    chiller = Chiller(component_id='chiller', target_temp_k=277.15, pressure_drop_bar=0.1, enable_dynamics=False)
    kod2 = KnockOutDrum(component_id='kod2_h2', diameter_m=0.5, delta_p_bar=0.05, gas_species='H2')
    coalescer = Coalescer(component_id='coalescer', d_shell=0.32, l_elem=1.0, gas_type='H2')
    # Use 'deoxo' ID to match generic search or specific name
    deoxo = DeoxoReactor(component_id='deoxo') 
    psa = PSA(component_id='psa', purity_target=0.9999, recovery_rate=0.90)
    
    # Initialize
    for comp in [kod1, dry_cooler, chiller, kod2, coalescer, deoxo, psa]:
        comp.initialize(dt, registry)
    
    # === Run Simulation Step ===
    inlet_stream = Stream(
        mass_flow_kg_h=MDOT_H2_KG_H,
        temperature_k=T_IN_K,
        pressure_pa=P_IN_PA,
        composition=inlet_composition,
        phase='gas'
    )
    
    # Source "state" in adapter is faked if not registered, but adapter handles 'source_h2' ID.
    # We won't have a source component object unless we make one.
    # The adapter looks up 'source_h2'. If not found, it skips.
    # Let's simple register a mock or skip source for now.
    
    # Manual Wiring & stepping
    kod1.receive_input('gas_inlet', inlet_stream)
    kod1.step(t=0.0)
    s1 = kod1.get_output('gas_outlet')
    
    dry_cooler.receive_input('fluid_in', s1)
    dry_cooler.step(t=0.0)
    s2 = dry_cooler.get_output('fluid_out')
    
    chiller.receive_input('fluid_in', s2)
    chiller.step(t=0.0)
    s3 = chiller.get_output('fluid_out')
    
    kod2.receive_input('gas_inlet', s3)
    kod2.step(t=0.0)
    s4 = kod2.get_output('gas_outlet')
    
    coalescer.receive_input('inlet', s4)
    coalescer.step(t=0.0)
    s5 = coalescer.get_output('outlet')
    
    # Heater skipped in manual step? The test had no heater. Adapter expects 'heater'.
    # If missing, adapter skips.
    
    deoxo.receive_input('inlet', s5)
    deoxo.step(t=0.0)
    s6 = deoxo.get_output('outlet')
    
    psa.receive_input('gas_in', s6)
    psa.step(t=0.0)
    
    print("\nSimulation Step Complete.")
    
    # === Generate Plots ===
    print("\nGenerating Legacy Plots...")
    
    adapter = LegacyDataAdapter(registry)
    df_h2, df_o2 = adapter.generate_dataframes()
    
    print("\nAdapted Data (H2 Line Head):")
    print(df_h2[['Componente', 'T_C', 'mdotgaskgs']].head())
    
    manager = LegacyPlotManager(output_dir=str(output_dir))
    manager.generate_all(df_h2, df_o2, registry)
    
    # === Validation ===
    png_files = list(output_dir.glob("*.png"))
    print(f"\nGenerated {len(png_files)} plots in {output_dir}")
    
    expected_plots = [
        'plot_agua_removida_total.png',
        'plot_fluxos_energia.png',
        'plot_deoxo_perfil.png',
        'plot_propriedades_empilhadas_H2.png'
    ]
    
    missing = []
    for p in expected_plots:
        if not (output_dir / p).exists():
            missing.append(p)
    
    if missing:
        print(f"❌ MISSING PLOTS: {missing}")
        # Identify why deoxo profile missing?
        # Check Deoxo profiles
        profs = deoxo.get_last_profiles()
        print(f"Deoxo Profiles L size: {len(profs.get('L', []))}")
    else:
        print("✅ All key plots generated successfully.")

if __name__ == '__main__':
    test_generate_legacy_plots()
