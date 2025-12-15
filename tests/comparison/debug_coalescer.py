"""
Debug: Verify Coalescer Parity with Legacy Model
"""
import sys
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')

from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.core.stream import Stream
from h2_plant.core.constants import CoalescerConstants

# Constants
BAR_TO_PA = 1e5
Q_M = 80.46  # kg/h
P_IN_BAR = 35.73
T_IN_K = 277.15 # ~4 C

print("=" * 70)
print("VERIFICATION: Coalescer Component vs Legacy Spec")
print("=" * 70)
print(f"Loaded K_PERDA from constants: {CoalescerConstants.K_PERDA:.2e}")
print(f"Loaded ETA from constants:     {CoalescerConstants.ETA_LIQUID_REMOVAL:.4f}")

# 1. Instantiate Real Component
coal = Coalescer(gas_type='H2')
coal.initialize(dt=1/60, registry=None)

# 2. Create Input Stream
inlet = Stream(
    mass_flow_kg_h=Q_M,
    temperature_k=T_IN_K,
    pressure_pa=P_IN_BAR * BAR_TO_PA,
    composition={'H2': 1.0}
)

# 3. specific density check
print(f"\nConditions:")
print(f"  P_in: {P_IN_BAR:.2f} bar")
print(f"  T_in: {T_IN_K:.2f} K")
print(f"  Flow: {Q_M:.2f} kg/h")

# 4. Run Step
coal.receive_input('inlet', inlet)

# 5. Get Results
state = coal.get_state()
dp_bar = state['delta_p_bar']
liq_rem = state['drain_flow_kg_h']

print("\n--- RESULTS ---")
print(f"Delta P (Calculated): {dp_bar:.4f} bar")
print(f"Delta P (Legacy Target): ~0.1500 bar")

diff = abs(dp_bar - 0.15)
print(f"Error: {diff:.4f} bar")

if dp_bar > 0.1:
    print("\n[SUCCESS] Pressure drop is within expected magnitude of legacy model.")
else:
    print("\n[FAILURE] Pressure drop is too low. Check K_PERDA.")
