"""
Unit tests for Chiller component.
Validates alignment with reference model (modelo_chiller.py).
"""

import unittest
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants


class TestChillerEnthalpyCalculation(unittest.TestCase):
    """Test enthalpy-based cooling calculation."""
    
    def setUp(self):
        self.chiller = Chiller(
            component_id="test_chiller",
            cop=4.0,
            pressure_drop_bar=0.2,
            target_temp_k=298.15,  # 25°C
            enable_dynamics=False
        )
        self.chiller.initialize(dt=1.0, registry=None)
    
    def test_h2_stream_cooling(self):
        """Test cooling of H2 stream uses enthalpy or gas-specific Cp."""
        inlet = Stream(
            mass_flow_kg_h=36.0,  # 0.01 kg/s
            temperature_k=353.15,  # 80°C
            pressure_pa=3e5,  # 3 bar
            composition={'H2': 1.0}
        )
        
        self.chiller.receive_input('fluid_in', inlet)
        self.chiller.step(t=0)
        
        outlet = self.chiller.get_output('fluid_out')
        
        # Outlet should achieve exact target temperature
        self.assertAlmostEqual(outlet.temperature_k, 298.15, delta=0.1)
        
        # Cooling load should be positive (heat removed)
        self.assertGreater(self.chiller.cooling_load_kw, 0)
    
    def test_o2_stream_cooling(self):
        """Test cooling of O2 stream uses correct Cp fallback."""
        inlet = Stream(
            mass_flow_kg_h=360.0,  # 0.1 kg/s
            temperature_k=363.15,  # 90°C
            pressure_pa=2e5,
            composition={'O2': 1.0}
        )
        
        self.chiller.receive_input('fluid_in', inlet)
        self.chiller.step(t=0)
        
        outlet = self.chiller.get_output('fluid_out')
        
        # Outlet should achieve exact target temperature
        self.assertAlmostEqual(outlet.temperature_k, 298.15, delta=0.1)


class TestChillerCOP(unittest.TestCase):
    """Test COP-based electrical power calculation."""
    
    def test_electrical_power_with_cop(self):
        """Verify W = |Q| / COP."""
        chiller = Chiller(
            cop=4.0,
            target_temp_k=298.15,
            enable_dynamics=False
        )
        chiller.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=36.0,
            temperature_k=353.15,  # 80°C
            pressure_pa=3e5,
            composition={'H2': 1.0}
        )
        
        chiller.receive_input('fluid_in', inlet)
        chiller.step(t=0)
        
        # Electrical power should be cooling load / COP
        expected_power = abs(chiller.cooling_load_kw) / 4.0
        self.assertAlmostEqual(
            chiller.electrical_power_kw, 
            expected_power, 
            delta=0.01
        )
    
    def test_electricity_port_output(self):
        """Verify electricity_in port returns electrical power."""
        chiller = Chiller(cop=4.0, enable_dynamics=False)
        chiller.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=36.0,
            temperature_k=353.15,
            pressure_pa=3e5,
            composition={'H2': 1.0}
        )
        
        chiller.receive_input('fluid_in', inlet)
        chiller.step(t=0)
        
        power = chiller.get_output('electricity_in')
        self.assertEqual(power, chiller.electrical_power_kw)


class TestChillerPressureDrop(unittest.TestCase):
    """Test pressure drop implementation."""
    
    def test_pressure_drop_applied(self):
        """Verify P_out = P_in - ΔP."""
        chiller = Chiller(
            pressure_drop_bar=0.2,
            target_temp_k=298.15,
            enable_dynamics=False
        )
        chiller.initialize(dt=1.0, registry=None)
        
        inlet_pressure_pa = 3e5  # 3 bar
        inlet = Stream(
            mass_flow_kg_h=36.0,
            temperature_k=353.15,
            pressure_pa=inlet_pressure_pa,
            composition={'H2': 1.0}
        )
        
        chiller.receive_input('fluid_in', inlet)
        chiller.step(t=0)
        
        outlet = chiller.get_output('fluid_out')
        expected_pressure = inlet_pressure_pa - (0.2 * 1e5)  # 2.8 bar
        
        self.assertAlmostEqual(
            outlet.pressure_pa, 
            expected_pressure, 
            delta=100
        )
    
    def test_pressure_drop_custom_value(self):
        """Verify custom pressure drop works."""
        chiller = Chiller(
            pressure_drop_bar=0.5,
            enable_dynamics=False
        )
        chiller.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=36.0,
            temperature_k=353.15,
            pressure_pa=5e5,  # 5 bar
            composition={'H2': 1.0}
        )
        
        chiller.receive_input('fluid_in', inlet)
        chiller.step(t=0)
        
        outlet = chiller.get_output('fluid_out')
        expected_pressure = 5e5 - (0.5 * 1e5)  # 4.5 bar
        
        self.assertAlmostEqual(outlet.pressure_pa, expected_pressure, delta=100)


class TestChillerExactTemperature(unittest.TestCase):
    """Test exact outlet temperature achievement."""
    
    def test_achieves_exact_target_temp(self):
        """Verify outlet achieves exact target (no capacity limiting)."""
        target_temp = 303.15  # 30°C
        chiller = Chiller(
            target_temp_k=target_temp,
            enable_dynamics=False
        )
        chiller.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=373.15,  # 100°C
            pressure_pa=2e5,
            composition={'H2': 0.8, 'O2': 0.2}
        )
        
        chiller.receive_input('fluid_in', inlet)
        chiller.step(t=0)
        
        outlet = chiller.get_output('fluid_out')
        
        # Should achieve exact target, not limited by capacity
        self.assertAlmostEqual(outlet.temperature_k, target_temp, delta=0.01)


class TestChillerDynamicsBypass(unittest.TestCase):
    """Test dynamics bypass functionality."""
    
    def test_no_dynamics_when_disabled(self):
        """Verify pump/thermal models not used when enable_dynamics=False."""
        chiller = Chiller(enable_dynamics=False)
        
        self.assertIsNone(chiller.pump)
        self.assertIsNone(chiller.coolant_thermal)
    
    def test_dynamics_enabled(self):
        """Verify pump/thermal models initialized when enable_dynamics=True."""
        chiller = Chiller(enable_dynamics=True)
        
        self.assertIsNotNone(chiller.pump)
        self.assertIsNotNone(chiller.coolant_thermal)


class TestChillerIdleState(unittest.TestCase):
    """Test chiller behavior with no flow."""
    
    def test_zero_flow_idle(self):
        """Verify all outputs zero when no inlet flow."""
        chiller = Chiller(enable_dynamics=False)
        chiller.initialize(dt=1.0, registry=None)
        
        # No inlet flow
        inlet = Stream(mass_flow_kg_h=0.0)
        chiller.receive_input('fluid_in', inlet)
        chiller.step(t=0)
        
        self.assertEqual(chiller.cooling_load_kw, 0.0)
        self.assertEqual(chiller.electrical_power_kw, 0.0)
        self.assertEqual(chiller.heat_rejected_kw, 0.0)


class TestChillerPortDefinitions(unittest.TestCase):
    """Test port definitions include electricity."""
    
    def test_electricity_port_defined(self):
        """Verify electricity_in port exists."""
        chiller = Chiller()
        ports = chiller.get_ports()
        
        self.assertIn('electricity_in', ports)
        self.assertEqual(ports['electricity_in']['type'], 'input')
        self.assertEqual(ports['electricity_in']['resource_type'], 'electricity')


class TestChillerState(unittest.TestCase):
    """Test state dictionary includes new fields."""
    
    def test_state_includes_cop_and_pressure(self):
        """Verify state dict has COP and pressure drop info."""
        chiller = Chiller(
            cop=4.0,
            pressure_drop_bar=0.2,
            enable_dynamics=False
        )
        chiller.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=36.0,
            temperature_k=353.15,
            pressure_pa=3e5,
            composition={'H2': 1.0}
        )
        chiller.receive_input('fluid_in', inlet)
        chiller.step(t=0)
        
        state = chiller.get_state()
        
        self.assertIn('electrical_power_kw', state)
        self.assertIn('outlet_pressure_bar', state)
        self.assertIn('cop', state)
        self.assertIn('pressure_drop_bar', state)
        self.assertEqual(state['cop'], 4.0)
        self.assertEqual(state['pressure_drop_bar'], 0.2)


if __name__ == '__main__':
    unittest.main()
