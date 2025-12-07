
import pickle
from pathlib import Path
import sys
import numpy as np
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from h2_plant.optimization.lut_manager import LUTConfig

path = Path('/home/stuart/.h2_plant/lut_cache/lut_H2_v1.pkl')
print(f"Inspecting {path}...")
with open(path, 'rb') as f:
    data = pickle.load(f)

c = data['config']
print(f"Config in file:")
print(f"  Pressure: {c.pressure_points} pts ({c.pressure_min} - {c.pressure_max})")
print(f"  Temperature: {c.temperature_points} pts ({c.temperature_min} - {c.temperature_max})")
print(f"  Fluids: {c.fluids}")

lut = data['lut']
print(f"LUT Data Data Shapes:")
for prop, arr in lut.items():
    print(f"  {prop}: {arr.shape}")
