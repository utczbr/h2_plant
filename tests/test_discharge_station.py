
import unittest
import logging
from h2_plant.components.delivery.discharge_station import DischargeStation
from h2_plant.core.stream import Stream

# Configure logging to capture component output
logging.basicConfig(level=logging.DEBUG)

class MockRegistry:
    def __init__(self):
        self.components = {}
    def has(self, name):
        return name in self.components
    def get(self, name):
        return self.components.get(name)

class TestDischargeStation(unittest.TestCase):
    def setUp(self):
        self.registry = MockRegistry()
        # Configuration matching user's topology (approximate)
        self.config = {
            'n_stations': 5,
            'truck_capacity_kg': 280.0,
            'delivery_pressure_bar': 500.0,
            'max_fill_rate_kg_min': 1.5,       # 90 kg/h
            'min_fill_rate_kg_min': 0.583,     # ~35 kg/h
            'h_in_day_max': 16.0,              # 16 hours day shift
            'valves_close_time_min': 5.0,
            'arrival_probability': 0.3,
            'isen_efficiency': 0.65,
            'mech_efficiency': 0.90,
            'cooldown_minutes': 187.0
        }

    def test_scheduled_mode_demand(self):
        """Verify demand calculation in Scheduled Mode (Day vs Night)."""
        station = DischargeStation(station_id=1, **self.config)
        station.initialize(dt=1.0, registry=self.registry) # dt = 1 hour

        # --- Test Day Shift (Hour 10) ---
        station.step(t=10.0) 
        
        # Expected Demand: max_fill_rate * n_stations * 60 min
        expected_demand_day = 1.5 * 5 * 60.0 
        demand_signal_day = station.get_output('demand_signal').mass_flow_kg_h
        
        print(f"Day Shift (10h): Expected={expected_demand_day:.2f}, Actual={demand_signal_day:.2f}")
        self.assertAlmostEqual(demand_signal_day, expected_demand_day, places=2, 
                               msg="Day shift demand mismatch")

        # --- Test Night Shift (Hour 20) ---
        station.step(t=20.0)
        
        # Expected Demand: min_fill_rate * n_stations * 60 min
        expected_demand_night = 0.583 * 5 * 60.0
        demand_signal_night = station.get_output('demand_signal').mass_flow_kg_h
        
        print(f"Night Shift (20h): Expected={expected_demand_night:.2f}, Actual={demand_signal_night:.2f}")
        self.assertAlmostEqual(demand_signal_night, expected_demand_night, places=2, 
                               msg="Night shift demand mismatch")

    def test_mass_delivery(self):
        """Verify mass is accepted and accumulated."""
        station = DischargeStation(station_id=1, **self.config)
        station.initialize(dt=1.0, registry=self.registry)
        
        # Simulate receiving matching flow
        flow_rate = 100.0
        stream = Stream(mass_flow_kg_h=flow_rate, pressure_pa=40e5, temperature_k=300, composition={'H2':1.0})
        station.receive_input('h2_in', stream)
        
        station.step(t=1.0)
        
        self.assertEqual(station.total_h2_delivered_kg, 100.0, "Total delivered mass should match input")
        
        # Check output stream (for physics continuity, even if it leaves system)
        out_stream = station.get_output('h2_out')
        self.assertEqual(out_stream.mass_flow_kg_h, 100.0, "Output stream should pass through mass")
        self.assertEqual(out_stream.pressure_pa, 500e5, "Output pressure should be delivery pressure")

    def test_n_stations_scaling(self):
        """Verify that demand scales linearly with n_stations."""
        # 1 Station
        cfg_1 = self.config.copy()
        cfg_1['n_stations'] = 1
        station_1 = DischargeStation(station_id=1, **cfg_1)
        station_1.initialize(dt=1.0, registry=self.registry)
        station_1.step(t=12.0)
        demand_1 = station_1.get_output('demand_signal').mass_flow_kg_h
        
        # 10 Stations
        cfg_10 = self.config.copy()
        cfg_10['n_stations'] = 10
        station_10 = DischargeStation(station_id=2, **cfg_10)
        station_10.initialize(dt=1.0, registry=self.registry)
        station_10.step(t=12.0)
        demand_10 = station_10.get_output('demand_signal').mass_flow_kg_h
        
        print(f"Scaling Limit: 1 Station={demand_1:.2f}, 10 Stations={demand_10:.2f}")
        self.assertAlmostEqual(demand_10, demand_1 * 10, places=2, 
                               msg="Demand did not scale linearly with n_stations")

if __name__ == '__main__':
    unittest.main()
