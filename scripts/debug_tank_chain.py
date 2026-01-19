
import logging
import sys
import os

# Adjust path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from h2_plant.components.storage.detailed_tank import DetailedTankArray
from h2_plant.components.compression.compressor_single import CompressorSingle
from h2_plant.components.cooling.dry_cooler import DryCooler
from h2_plant.components.delivery.discharge_station import DischargeStation
from h2_plant.core.stream import Stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugChain")

def debug_chain():
    logger.info("Initializing partial chain: Tank -> HP Compressors -> Station")
    
    dt = 1/60.0 # 1 minute steps
    
    # 1. Tank (Start FULL)
    # 30 tanks, 107 m3, P_max=70 => Huge capacity
    tank = DetailedTankArray(
        n_tanks=30,
        volume_per_tank_m3=107.37,
        max_pressure_bar=70.0,
        initial_pressure_bar=60.0, # Start FULL (60 bar > 30 bar min)
        min_discharge_pressure_bar=30.0,
    )
    tank.component_id = "LP_Storage_Tank"
    
    # 2. HP Compressor S2 (First in chain)
    comp_s2 = CompressorSingle(
        max_flow_kg_h=1020.0,
        inlet_pressure_bar=30.0,
        outlet_pressure_bar=500.0, 
    )
    comp_s2.component_id = "HP_Compressor_S2"
    
    # ... Skipping intermediate stages for brevity, just simulating the ends ...
    
    # 3. Final Compressor S5
    comp_s5 = CompressorSingle(
        max_flow_kg_h=1020.0,
        inlet_pressure_bar=250.0,
        outlet_pressure_bar=500.0,
        temperature_limited=True,
        max_temp_c=135.0,
    )
    comp_s5.component_id = "HP_Compressor_S5"
    
    # 4. Discharge Station
    station = DischargeStation(
        n_stations=20,
        max_fill_rate_kg_min=1.5,
        h_in_day_max=16.0
    )
    station.component_id = "Truck_Station_1"
    
    class MockRegistry:
        def __init__(self):
            self.components = {}
        def has(self, key): return key in self.components
        def get(self, key): return self.components.get(key)
        def register(self, key, comp): self.components[key] = comp
        def list_components(self): return self.components.items()

    registry = MockRegistry()
    registry.register("LP_Storage_Tank", tank)
    registry.register("HP_Compressor_S2", comp_s2)
    registry.register("HP_Compressor_S5", comp_s5)
    registry.register("Truck_Station_1", station)
    
    tank.initialize(dt, registry)
    comp_s2.initialize(dt, registry)
    comp_s5.initialize(dt, registry)
    station.initialize(dt, registry)
    
    # Simulation Loop
    for step in range(10):
        t = step * dt
        print(f"\n--- Step {step} (t={t:.2f}h) ---")
        
        # A. Station Step (Calculates Demand)
        station.step(t)
        demand_stream = station.get_output('demand_signal')
        print(f"Station Demand Signal: {demand_stream.mass_flow_kg_h:.2f} kg/h")
        
        # B. Propagate Demand to Tank
        tank.receive_input('demand_signal', demand_stream)
        
        # C. Tank Step (Calculates Discharge)
        tank.step(t)
        discharge_stream = tank.get_output('h2_out')
        print(f"Tank Discharge: {discharge_stream.mass_flow_kg_h:.2f} kg/h, P_avg={tank.avg_pressure_bar:.1f} bar")
        
        # D. Propagate Flow: Tank -> Comp S2
        accepted_s2 = comp_s2.receive_input('inlet', discharge_stream, 'gas')
        tank.extract_output('h2_out', accepted_s2, 'gas')
        print(f"Comp S2 Accepted from Tank: {accepted_s2:.4f} kg (Rate: {accepted_s2/dt:.2f} kg/h)")
        
        # E. Comp S2 Step
        comp_s2.step(t)
        s2_out = comp_s2.get_output('outlet')
        print(f"Comp S2 Output: {s2_out.mass_flow_kg_h:.2f} kg/h")
        
        # ... Imagine flow goes through middle stages ...
        
        # F. Propagate Flow: Middle -> Comp S5
        # We manually just pass S2 output to S5 for testing blockage
        accepted_s5 = comp_s5.receive_input('inlet', s2_out, 'gas')
        print(f"Comp S5 Accepted from S2: {accepted_s5:.4f} kg (Rate: {accepted_s5/dt:.2f} kg/h)")
        
        # G. Comp S5 Step
        comp_s5.step(t)
        s5_out = comp_s5.get_output('outlet')
        print(f"Comp S5 Output: {s5_out.mass_flow_kg_h:.2f} kg/h")
        print(f"Comp S5 Temp Limited? {comp_s5.temperature_limited}, T_out={comp_s5.outlet_temperature_c:.1f}C")
        
        # H. Propagate Flow: S5 -> Station
        accepted_station = station.receive_input('h2_in', s5_out)
        print(f"Station Received: {accepted_station:.2f} kg/h")

if __name__ == "__main__":
    try:
        debug_chain()
    except Exception as e:
        logger.error(f"Debug crashed: {e}")
        import traceback
        traceback.print_exc()
