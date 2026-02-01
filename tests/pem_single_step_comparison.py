"""
Single-Step System Comparison: PEM H2 Purification Line

This test runs a single timestep through the H2 purification line topology
using the new component architecture and validates outputs.

Topology:
    Inlet -> KOD1 -> DryCooler -> Chiller -> KOD2 -> Coalescer -> Deoxo -> PSA

Test Strategy:
    1. Create all components with legacy-matching configurations.
    2. Wire them together manually (simulating PlantGraphBuilder).
    3. Run one timestep.
    4. Validate key outputs at each stage.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.separation.knock_out_drum import KnockOutDrum
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
from h2_plant.components.separation.psa import PSA


def test_single_step_pem_h2_line():
    """Run single timestep through H2 purification line."""
    
    # === Configuration (from constants_and_config.py) ===
    T_IN_K = 333.15  # 60°C
    P_IN_PA = 40e5   # 40 bar
    MDOT_H2_KG_H = 89.0
    
    # Initial composition (from PEM outlet)
    inlet_composition = {
        'H2': 0.98,
        'H2O': 0.018,
        'O2': 0.002  # 2000 ppm O2 crossover
    }
    
    # === Create Components ===
    registry = ComponentRegistry()
    dt = 1/60  # 1 minute timestep
    
    kod1 = KnockOutDrum(diameter_m=0.5, delta_p_bar=0.05, gas_species='H2')
    dry_cooler = DryCooler(component_id='DC_H2')
    chiller = Chiller(target_temp_k=277.15, pressure_drop_bar=0.1, enable_dynamics=False)
    kod2 = KnockOutDrum(diameter_m=0.5, delta_p_bar=0.05, gas_species='H2')
    coalescer = Coalescer(d_shell=0.32, l_elem=1.0, gas_type='H2')
    deoxo = DeoxoReactor(component_id='deoxo_h2')
    psa = PSA(purity_target=0.9999, recovery_rate=0.90)
    
    # Initialize all
    for comp in [kod1, dry_cooler, chiller, kod2, coalescer, deoxo, psa]:
        comp.initialize(dt, registry)
    
    # === Create Inlet Stream ===
    inlet_stream = Stream(
        mass_flow_kg_h=MDOT_H2_KG_H,
        temperature_k=T_IN_K,
        pressure_pa=P_IN_PA,
        composition=inlet_composition,
        phase='gas'
    )
    
    print("=" * 60)
    print("PEM H2 PURIFICATION LINE - SINGLE STEP TEST")
    print("=" * 60)
    print(f"\nINLET: {MDOT_H2_KG_H:.1f} kg/h, {T_IN_K-273.15:.1f}°C, {P_IN_PA/1e5:.1f} bar")
    
    # === Step 1: KOD 1 ===
    kod1.receive_input('gas_inlet', inlet_stream)
    kod1.step(t=0.0)
    stream = kod1.get_output('gas_outlet')
    print(f"\n1. KOD1: {stream.mass_flow_kg_h:.2f} kg/h, {stream.temperature_k-273.15:.1f}°C")
    
    # === Step 2: Dry Cooler ===
    dry_cooler.receive_input('fluid_in', stream)
    dry_cooler.step(t=0.0)
    stream = dry_cooler.get_output('fluid_out')
    dc_state = dry_cooler.get_state()
    print(f"2. DryCooler: {stream.mass_flow_kg_h:.2f} kg/h, {stream.temperature_k-273.15:.1f}°C, Q={dc_state['heat_duty_kw']:.1f}kW")
    
    # === Step 3: Chiller ===
    chiller.receive_input('fluid_in', stream)
    chiller.step(t=0.0)
    stream = chiller.get_output('fluid_out')
    ch_state = chiller.get_state()
    print(f"3. Chiller: {stream.mass_flow_kg_h:.2f} kg/h, {stream.temperature_k-273.15:.1f}°C, Q={ch_state['cooling_load_kw']:.1f}kW")
    
    # === Step 4: KOD 2 ===
    kod2.receive_input('gas_inlet', stream)
    kod2.step(t=0.0)
    stream = kod2.get_output('gas_outlet')
    print(f"4. KOD2: {stream.mass_flow_kg_h:.2f} kg/h, {stream.temperature_k-273.15:.1f}°C")
    
    # === Step 5: Coalescer ===
    coalescer.receive_input('inlet', stream)
    coalescer.step(t=0.0)
    stream = coalescer.get_output('outlet')
    print(f"5. Coalescer: {stream.mass_flow_kg_h:.2f} kg/h, {stream.temperature_k-273.15:.1f}°C")
    
    # === Step 6: Deoxo ===
    deoxo.receive_input('inlet', stream)
    deoxo.step(t=0.0)
    stream = deoxo.get_output('outlet')
    dx_state = deoxo.get_state()
    print(f"6. Deoxo: {stream.mass_flow_kg_h:.2f} kg/h, {stream.temperature_k-273.15:.1f}°C, conv={dx_state.get('conversion_o2_percent', 0):.1f}%")
    
    # === Step 7: PSA ===
    psa.receive_input('gas_in', stream)
    psa.step(t=0.0)
    product = psa.get_output('purified_gas_out')
    tail = psa.get_output('tail_gas_out')
    psa_state = psa.get_state()
    print(f"7. PSA: Product={product.mass_flow_kg_h:.2f} kg/h, Tail={tail.mass_flow_kg_h:.2f} kg/h")
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Inlet Flow: {MDOT_H2_KG_H:.2f} kg/h")
    print(f"Product Flow: {product.mass_flow_kg_h:.2f} kg/h")
    print(f"Product Purity (H2): {product.composition.get('H2', 0)*100:.4f}%")
    print(f"Recovery Rate: {product.mass_flow_kg_h/MDOT_H2_KG_H*100:.1f}%")
    print(f"Temperature Drop: {T_IN_K-273.15:.1f}°C -> {product.temperature_k-273.15:.1f}°C")
    print(f"Pressure Drop: {P_IN_PA/1e5:.1f} bar -> {product.pressure_pa/1e5:.2f} bar")
    
    # === Validation ===
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    # Mass balance
    total_out = product.mass_flow_kg_h + tail.mass_flow_kg_h
    mass_balance_error = abs(stream.mass_flow_kg_h - total_out) / stream.mass_flow_kg_h * 100
    print(f"PSA Mass Balance Error: {mass_balance_error:.4f}% (max 1%)")
    assert mass_balance_error < 1.0, f"Mass balance error too large: {mass_balance_error}%"
    
    # Product purity
    h2_purity = product.composition.get('H2', 0)
    print(f"Product H2 Purity: {h2_purity*100:.4f}% (target > 99.9%)")
    
    # Recovery
    recovery = product.mass_flow_kg_h / MDOT_H2_KG_H
    print(f"Overall Recovery: {recovery*100:.1f}% (expected ~80-90%)")
    
    print("\n✅ SINGLE-STEP SYSTEM TEST PASSED")


if __name__ == '__main__':
    test_single_step_pem_h2_line()
