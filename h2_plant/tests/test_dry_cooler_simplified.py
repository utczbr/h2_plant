"""
Unit tests for DryCoolerSimplified component.
Validates enthalpy-based cooling and condensation detection.
"""

import unittest
from h2_plant.components.cooling.dry_cooler_simplified import DryCoolerSimplified
from h2_plant.core.stream import Stream


class TestDryCoolerSimplifiedBasicCooling(unittest.TestCase):
    """Test basic cooling functionality."""
    
    def setUp(self):
        self.cooler = DryCoolerSimplified(
            component_id="test_cooler",
            target_temp_k=313.15,  # 40°C
            pressure_drop_bar=0.05,
            fan_specific_power_kw_per_mw=15.0
        )
        self.cooler.initialize(dt=1.0, registry=None)
    
    def test_h2_stream_cools_to_target(self):
        """Test cooling of H2 stream achieves target temperature."""
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=373.15,  # 100°C
            pressure_pa=5e5,       # 5 bar
            composition={'H2': 1.0}
        )
        
        self.cooler.receive_input('inlet', inlet)
        self.cooler.step(t=0)
        
        outlet = self.cooler.get_output('outlet')
        
        # Outlet should achieve target temperature
        self.assertAlmostEqual(outlet.temperature_k, 313.15, delta=0.1)
        
        # Heat should be rejected (positive value)
        self.assertGreater(self.cooler.heat_rejected_kw, 0)
    
    def test_o2_stream_cools_correctly(self):
        """Test cooling of O2 stream."""
        inlet = Stream(
            mass_flow_kg_h=500.0,
            temperature_k=353.15,  # 80°C
            pressure_pa=3e5,
            composition={'O2': 1.0}
        )
        
        self.cooler.receive_input('inlet', inlet)
        self.cooler.step(t=0)
        
        outlet = self.cooler.get_output('outlet')
        
        self.assertAlmostEqual(outlet.temperature_k, 313.15, delta=0.1)

    def test_bypass_when_inlet_cold(self):
        """Test stream passes through unchanged if already below target."""
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,  # 27°C (below 40°C target)
            pressure_pa=5e5,
            composition={'H2': 1.0}
        )
        
        self.cooler.receive_input('inlet', inlet)
        self.cooler.step(t=0)
        
        outlet = self.cooler.get_output('outlet')
        
        # Should not heat up - stays at inlet temperature
        self.assertAlmostEqual(outlet.temperature_k, 300.0, delta=0.1)


class TestDryCoolerSimplifiedPressureDrop(unittest.TestCase):
    """Test pressure drop implementation."""
    
    def test_pressure_drop_applied(self):
        """Verify P_out = P_in - ΔP."""
        cooler = DryCoolerSimplified(
            component_id="test_cooler",
            target_temp_k=313.15,
            pressure_drop_bar=0.1
        )
        cooler.initialize(dt=1.0, registry=None)
        
        inlet_pressure_pa = 5e5  # 5 bar
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=373.15,
            pressure_pa=inlet_pressure_pa,
            composition={'H2': 1.0}
        )
        
        cooler.receive_input('inlet', inlet)
        cooler.step(t=0)
        
        outlet = cooler.get_output('outlet')
        expected_pressure = inlet_pressure_pa - (0.1 * 1e5)  # 4.9 bar
        
        self.assertAlmostEqual(outlet.pressure_pa, expected_pressure, delta=100)


class TestDryCoolerSimplifiedMassConservation(unittest.TestCase):
    """Test that mass is conserved (liquid remains entrained)."""
    
    def test_mass_preserved(self):
        """Verify inlet mass equals outlet mass."""
        cooler = DryCoolerSimplified(
            component_id="test_cooler",
            target_temp_k=303.15  # 30°C - may cause condensation
        )
        cooler.initialize(dt=1.0, registry=None)
        
        inlet_mass = 200.0
        inlet = Stream(
            mass_flow_kg_h=inlet_mass,
            temperature_k=373.15,
            pressure_pa=3e5,
            composition={'H2': 0.9, 'H2O': 0.1}
        )
        
        cooler.receive_input('inlet', inlet)
        cooler.step(t=0)
        
        outlet = cooler.get_output('outlet')
        
        # Mass must be conserved
        self.assertAlmostEqual(outlet.mass_flow_kg_h, inlet_mass, delta=0.01)


class TestDryCoolerSimplifiedFanPower(unittest.TestCase):
    """Test fan power calculation."""
    
    def test_fan_power_scales_with_duty(self):
        """Verify fan power = specific_power * heat_duty_mw."""
        cooler = DryCoolerSimplified(
            component_id="test_cooler",
            target_temp_k=298.15,
            fan_specific_power_kw_per_mw=15.0
        )
        cooler.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=373.15,
            pressure_pa=5e5,
            composition={'H2': 1.0}
        )
        
        cooler.receive_input('inlet', inlet)
        cooler.step(t=0)
        
        # Fan power should be proportional to heat duty
        expected_fan_power = (cooler.heat_rejected_kw / 1000.0) * 15.0
        self.assertAlmostEqual(cooler.fan_power_kw, expected_fan_power, delta=0.01)


class TestDryCoolerSimplifiedIdleState(unittest.TestCase):
    """Test idle behavior with no flow."""
    
    def test_zero_flow_idle(self):
        """Verify all outputs zero when no inlet flow."""
        cooler = DryCoolerSimplified(component_id="test_cooler")
        cooler.initialize(dt=1.0, registry=None)
        
        # No inlet flow
        inlet = Stream(mass_flow_kg_h=0.0)
        cooler.receive_input('inlet', inlet)
        cooler.step(t=0)
        
        self.assertEqual(cooler.heat_rejected_kw, 0.0)
        self.assertEqual(cooler.fan_power_kw, 0.0)
        self.assertEqual(cooler.condensed_water_kg_h, 0.0)


class TestDryCoolerSimplifiedPorts(unittest.TestCase):
    """Test port definitions."""
    
    def test_ports_defined(self):
        """Verify required ports exist."""
        cooler = DryCoolerSimplified(component_id="test_cooler")
        ports = cooler.get_ports()
        
        self.assertIn('inlet', ports)
        self.assertIn('outlet', ports)
        self.assertEqual(ports['inlet']['type'], 'input')
        self.assertEqual(ports['outlet']['type'], 'output')


class TestDryCoolerSimplifiedState(unittest.TestCase):
    """Test state dictionary."""
    
    def test_state_includes_key_fields(self):
        """Verify state dict has required fields."""
        cooler = DryCoolerSimplified(
            component_id="test_cooler",
            target_temp_k=313.15,
            pressure_drop_bar=0.05
        )
        cooler.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=373.15,
            pressure_pa=5e5,
            composition={'H2': 1.0}
        )
        cooler.receive_input('inlet', inlet)
        cooler.step(t=0)
        
        state = cooler.get_state()
        
        self.assertIn('heat_rejected_kw', state)
        self.assertIn('fan_power_kw', state)
        self.assertIn('condensed_water_kg_h', state)
        self.assertIn('vapor_fraction', state)
        self.assertIn('outlet_temp_k', state)
        self.assertIn('pressure_drop_bar', state)


if __name__ == '__main__':
    unittest.main()
