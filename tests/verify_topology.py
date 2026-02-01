import sys
from pathlib import Path

# Add project root to path
sys.path.append("/home/stuart/Documentos/Planta Hidrogenio")

from h2_plant.run_integrated_simulation import run_with_dispatch_strategy

scenarios_dir = "/home/stuart/Documentos/Planta Hidrogenio/scenarios"
try:
    print("Starting verification simulation (1 hour)...")
    run_with_dispatch_strategy(scenarios_dir, hours=1)
    print("Simulation verified successfully.")
except Exception as e:
    print(f"Simulation failed: {e}")
    sys.exit(1)
