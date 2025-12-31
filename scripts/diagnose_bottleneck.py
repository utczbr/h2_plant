"""Test with realistic stream composition from simulation."""
import sys
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')
import time

# Track mix_thermo calls
from h2_plant.optimization import mixture_thermodynamics as mix_thermo

call_count = [0]
orig_enthalpy = mix_thermo.get_mixture_enthalpy
def tracked_enthalpy(*args, **kwargs):
    call_count[0] += 1
    return orig_enthalpy(*args, **kwargs)
mix_thermo.get_mixture_enthalpy = tracked_enthalpy

from h2_plant.core.stream import Stream
from h2_plant.components.cooling.dry_cooler import DryCooler  
from h2_plant.optimization.lut_manager import LUTManager

# Create component
dc = DryCooler(component_id='test_dc')

# Create a mock registry with LUT manager
class MockRegistry:
    def __init__(self):
        self.lut_mgr = None
        try:
            self.lut_mgr = LUTManager()
            self.lut_mgr.initialize()
        except Exception as e:
            print(f"LUT init failed: {e}")
            
    def get(self, name):
        if name == 'lut_manager':
            return self.lut_mgr
        return None

print("Loading LUTs...")
t0 = time.perf_counter()
reg = MockRegistry()
print(f"LUT load time: {time.perf_counter() - t0:.2f}s")

dc.initialize(dt=1/60, registry=reg)

# Test with wet H2 stream (95% H2, 5% H2O) - simulates real conditions
test_stream = Stream(
    mass_flow_kg_h=100.0,
    temperature_k=400.0,
    pressure_pa=1e6,
    composition={'H2': 0.95, 'H2O': 0.05},
    phase='gas'
)

N = 100
call_count[0] = 0
t0 = time.perf_counter()
for i in range(N):
    dc.receive_input('inlet', test_stream, 'gas')
    dc.step(i * 1/60)
elapsed = time.perf_counter() - t0

print(f"\n=== WITH LUT MANAGER + WET H2 (95% H2) ===")
print(f"{N} iterations in {elapsed:.3f}s = {N/elapsed:.1f} iter/sec")
print(f"mix_thermo calls per iteration: {call_count[0]/N:.1f}")
print(f"dominant_frac = 0.95 < 0.98 -> mix_thermo IS CALLED")

# Now test with dry H2 (99% H2)
test_stream_dry = Stream(
    mass_flow_kg_h=100.0,
    temperature_k=400.0,
    pressure_pa=1e6,
    composition={'H2': 0.99, 'H2O': 0.01},
    phase='gas'
)

call_count[0] = 0
t0 = time.perf_counter()
for i in range(N):
    dc.receive_input('inlet', test_stream_dry, 'gas')
    dc.step(i * 1/60)
elapsed_dry = time.perf_counter() - t0

print(f"\n=== WITH LUT MANAGER + DRY H2 (99% H2) ===")
print(f"{N} iterations in {elapsed_dry:.3f}s = {N/elapsed_dry:.1f} iter/sec")
print(f"mix_thermo calls: {call_count[0]} (should be 0)")
print(f"dominant_frac = 0.99 > 0.98 -> mix_thermo SKIPPED")
