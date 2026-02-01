"""
Verification script for SyngasPSA component.

Tests:
1. Mass balance (inlet = product + tail gas)
2. Purity constraint enforcement
3. Recovery rate enforcement
4. Mole/Mass fraction conversion correctness
"""

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.components.separation.psa_syngas import SyngasPSA
from h2_plant.core.stream import Stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_syngas_psa():
    logger.info("=" * 60)
    logger.info("SyngasPSA Verification")
    logger.info("=" * 60)

    # Create PSA
    psa = SyngasPSA(
        component_id="Test_SyngasPSA",
        purity_target=0.9999,
        recovery_rate=0.90
    )
    psa.initialize(dt=1.0, registry=None)  # 1 hour timestep

    # Create test inlet stream (typical syngas composition in MOLE fractions)
    # Approx: 66% H2, 10% CO, 10% CO2, 10% H2O, 3% CH4, 1% N2
    inlet = Stream(
        mass_flow_kg_h=1000.0,
        temperature_k=313.15,  # 40°C
        pressure_pa=2500000.0,  # 25 bar
        composition={
            'H2': 0.66,
            'CO': 0.10,
            'CO2': 0.10,
            'H2O': 0.10,
            'CH4': 0.03,
            'N2': 0.01
        },
        phase='gas'
    )

    # Feed to PSA
    psa.receive_input('gas_in', inlet, 'gas')

    # Step
    psa.step(0.0)

    # Get outputs
    product = psa.product_outlet
    tailgas = psa.tail_gas_outlet

    # Verify Mass Balance
    mass_in = inlet.mass_flow_kg_h
    mass_out = product.mass_flow_kg_h + tailgas.mass_flow_kg_h
    mass_error = abs(mass_in - mass_out) / mass_in * 100

    logger.info(f"\n--- Mass Balance ---")
    logger.info(f"Inlet:    {mass_in:.2f} kg/h")
    logger.info(f"Product:  {product.mass_flow_kg_h:.2f} kg/h")
    logger.info(f"Tail Gas: {tailgas.mass_flow_kg_h:.2f} kg/h")
    logger.info(f"Error:    {mass_error:.4f}%")

    if mass_error < 0.01:
        logger.info("PASS: Mass Balance")
    else:
        logger.error(f"FAIL: Mass Balance Error = {mass_error:.4f}%")

    # Verify Purity
    h2_purity_mol = product.composition.get('H2', 0.0)
    logger.info(f"\n--- Purity ---")
    logger.info(f"H2 Mole Fraction in Product: {h2_purity_mol:.6f}")

    # Calculate mass purity for comparison
    mw = {'H2': 2.016, 'CO': 28.01, 'CO2': 44.01, 'CH4': 16.04, 'H2O': 18.015, 'N2': 28.014}
    comp = product.composition
    total_mw = sum(comp.get(s, 0) * mw.get(s, 28) for s in comp)
    h2_mass_frac = (comp.get('H2', 0) * 2.016) / total_mw if total_mw > 0 else 0

    logger.info(f"H2 Mass Fraction in Product: {h2_mass_frac:.6f}")
    logger.info(f"Target Purity (mass):        {psa.purity_target:.6f}")

    if h2_mass_frac >= psa.purity_target * 0.999:  # Allow 0.1% tolerance
        logger.info("PASS: Purity Target Met")
    else:
        logger.error(f"FAIL: Purity {h2_mass_frac:.6f} < Target {psa.purity_target:.6f}")

    # Verify Recovery
    # Calculate H2 input mass
    inlet_comp = inlet.composition
    inlet_mw = sum(inlet_comp.get(s, 0) * mw.get(s, 28) for s in inlet_comp)
    h2_in_mass = (inlet_comp.get('H2', 0) * 2.016 / inlet_mw) * mass_in if inlet_mw > 0 else 0
    
    # Calculate H2 output mass
    h2_out_mass = h2_mass_frac * product.mass_flow_kg_h
    
    actual_recovery = h2_out_mass / h2_in_mass if h2_in_mass > 0 else 0

    logger.info(f"\n--- Recovery ---")
    logger.info(f"H2 In:       {h2_in_mass:.2f} kg/h")
    logger.info(f"H2 Out:      {h2_out_mass:.2f} kg/h")
    logger.info(f"Actual:      {actual_recovery:.4f}")
    logger.info(f"Target:      {psa.recovery_rate:.4f}")

    if abs(actual_recovery - psa.recovery_rate) < 0.01:
        logger.info("PASS: Recovery Rate")
    else:
        logger.error(f"FAIL: Recovery {actual_recovery:.4f} != Target {psa.recovery_rate:.4f}")

    # Log Tail Gas Composition
    logger.info(f"\n--- Tail Gas Composition (mol%) ---")
    for s, y in tailgas.composition.items():
        logger.info(f"  {s}: {y*100:.2f}%")

    logger.info(f"\n--- Power Consumption ---")
    logger.info(f"  {psa.power_consumption_kw:.2f} kW")

    logger.info("\n" + "=" * 60)
    logger.info("Verification Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    verify_syngas_psa()
