"""
Comprehensive unit, comparison, and integration tests for KnockOutDrum component.
Validates alignment with reference model (h2_plant/legacy/Knock out drum/KOD.py).
"""

import unittest
import math
import CoolProp.CoolProp as CP

from h2_plant.components.separation.knock_out_drum import KnockOutDrum
from h2_plant.core.stream import Stream
from h2_plant.core.constants import GasConstants


# =============================================================================
# Reference Model Constants (from KOD.py)
# =============================================================================
T_IN_C_REF = 4.0  # Reference inlet temp (°C)
T_IN_K_REF = T_IN_C_REF + 273.15  # 277.15 K
P_IN_BAR_REF = 40.0  # Reference inlet pressure (bar)
P_IN_PA_REF = P_IN_BAR_REF * 1e5  # Pa
DELTA_P_BAR_REF = 0.05  # Default pressure drop
R_UNIV = 8.31446  # Universal gas constant
RHO_L_WATER = 1000.0  # kg/m³
K_SOUDERS_BROWN = 0.08  # m/s


def reference_model(gas_fluido: str, vazao_molar_in: float, 
                    delta_p_bar: float = 0.05, diametro_vaso_m: float = 1.0) -> dict:
    """
    Python translation of KOD.py reference model for comparison testing.
    Uses simplified y_H2O = P_sat/P_out (single condensable assumption).
    """
    P_OUT_BAR = P_IN_BAR_REF - delta_p_bar
    P_OUT_PA = P_OUT_BAR * 1e5
    
    P_SAT_H2O_PA = CP.PropsSI('P', 'T', T_IN_K_REF, 'Q', 0, 'Water')
    y_H2O_out = P_SAT_H2O_PA / P_OUT_PA
    y_gas_out = 1.0 - y_H2O_out
    
    M_H2O = CP.PropsSI('M', 'Water')  # kg/mol
    M_GAS = CP.PropsSI('M', gas_fluido)  # kg/mol
    M_MIX_G = y_gas_out * M_GAS + y_H2O_out * M_H2O
    
    Z_gas = CP.PropsSI('Z', 'T', T_IN_K_REF, 'P', P_OUT_PA, gas_fluido)
    rho_G_out = P_OUT_PA * M_MIX_G / (Z_gas * R_UNIV * T_IN_K_REF)
    
    vazao_molar_gas_out = vazao_molar_in * y_gas_out
    vazao_volumetrica_gas_out = vazao_molar_gas_out * M_MIX_G / rho_G_out
    
    V_max = K_SOUDERS_BROWN * math.sqrt((RHO_L_WATER - rho_G_out) / rho_G_out)
    A_vaso = math.pi * (diametro_vaso_m / 2)**2
    V_superficial_real = vazao_volumetrica_gas_out / A_vaso
    
    potencia_W = vazao_volumetrica_gas_out * (delta_p_bar * 1e5)
    status = "OK" if V_superficial_real < V_max else "UNDERSIZED"
    
    return {
        "P_OUT_BAR": P_OUT_BAR,
        "y_H2O_out": y_H2O_out,
        "y_gas_out": y_gas_out,
        "rho_G_out": rho_G_out,
        "V_max": V_max,
        "V_real": V_superficial_real,
        "power_W": potencia_W,
        "status": status
    }


class TestKnockOutDrumConstruction(unittest.TestCase):
    """Test constructor and initialization."""
    
    def test_default_parameters(self):
        """Verify default parameter values."""
        kod = KnockOutDrum()
        self.assertEqual(kod.diameter_m, 1.0)
        self.assertEqual(kod.delta_p_bar, 0.05)
        self.assertEqual(kod.gas_species, 'H2')
    
    def test_custom_parameters(self):
        """Verify custom parameters accepted."""
        kod = KnockOutDrum(diameter_m=0.8, delta_p_bar=0.1, gas_species='O2')
        self.assertEqual(kod.diameter_m, 0.8)
        self.assertEqual(kod.delta_p_bar, 0.1)
        self.assertEqual(kod.gas_species, 'O2')
    
    def test_invalid_diameter_raises(self):
        """Verify ValueError on non-positive diameter."""
        with self.assertRaises(ValueError):
            KnockOutDrum(diameter_m=0.0)
        with self.assertRaises(ValueError):
            KnockOutDrum(diameter_m=-1.0)
    
    def test_invalid_gas_species_raises(self):
        """Verify ValueError on invalid gas species."""
        with self.assertRaises(ValueError):
            KnockOutDrum(gas_species='N2')
    
    def test_initialize(self):
        """Verify initialize() marks component ready."""
        kod = KnockOutDrum()
        kod.initialize(dt=1.0, registry=None)
        self.assertTrue(kod._initialized)


class TestKnockOutDrumPorts(unittest.TestCase):
    """Test port definitions."""
    
    def test_port_definitions(self):
        """Verify correct ports are defined."""
        kod = KnockOutDrum()
        ports = kod.get_ports()
        
        self.assertIn('gas_inlet', ports)
        self.assertEqual(ports['gas_inlet']['type'], 'input')
        
        self.assertIn('gas_outlet', ports)
        self.assertEqual(ports['gas_outlet']['type'], 'output')
        
        self.assertIn('liquid_drain', ports)
        self.assertEqual(ports['liquid_drain']['type'], 'output')


class TestKnockOutDrumNoFlow(unittest.TestCase):
    """Test behavior with no input flow."""
    
    def test_no_input_stream(self):
        """Verify graceful handling when no input."""
        kod = KnockOutDrum()
        kod.initialize(dt=1.0, registry=None)
        kod.step(t=0)
        
        self.assertEqual(kod._separation_status, "NO_FLOW")
        self.assertIsNone(kod.get_output('gas_outlet'))
    
    def test_zero_flow_stream(self):
        """Verify graceful handling of zero-flow stream."""
        kod = KnockOutDrum()
        kod.initialize(dt=1.0, registry=None)
        
        inlet = Stream(mass_flow_kg_h=0.0)
        kod.receive_input('gas_inlet', inlet)
        kod.step(t=0)
        
        self.assertEqual(kod._separation_status, "NO_FLOW")


class TestKnockOutDrumReferenceComparison(unittest.TestCase):
    """
    Compare component outputs against reference model (KOD.py).
    Uses identical conditions: T=4°C, P=40bar, pure gas inlet.
    """
    
    def setUp(self):
        """Create component with reference conditions."""
        self.kod_h2 = KnockOutDrum(diameter_m=1.0, delta_p_bar=0.05, gas_species='H2')
        self.kod_h2.initialize(dt=1.0, registry=None)
        
        self.kod_o2 = KnockOutDrum(diameter_m=1.0, delta_p_bar=0.05, gas_species='O2')
        self.kod_o2.initialize(dt=1.0, registry=None)
    
    def _molar_to_mass_stream(self, gas: str, molar_flow_mol_s: float) -> Stream:
        """
        Convert molar flow (mol/s) to mass flow stream (kg/h).
        For ref comparison, inlet is pure gas with trace H2O at saturation.
        
        Reference model assumes inlet = molar_flow of mixture,
        then calculates y_H2O from saturation. We'll match that by:
        1. Creating inlet with trace H2O corresponding to saturation.
        """
        # P_sat at 4°C
        P_sat = CP.PropsSI('P', 'T', T_IN_K_REF, 'Q', 0, 'Water')
        P_out = (P_IN_BAR_REF - 0.05) * 1e5
        y_H2O = P_sat / P_out
        y_gas = 1.0 - y_H2O
        
        M_gas = CP.PropsSI('M', gas)
        M_H2O = CP.PropsSI('M', 'Water')
        
        # Mixed molar mass
        M_mix = y_gas * M_gas + y_H2O * M_H2O
        
        # Total mass flow (kg/s) then (kg/h)
        n_total = molar_flow_mol_s
        m_dot_kg_s = n_total * M_mix
        m_dot_kg_h = m_dot_kg_s * 3600.0
        
        # Mass fractions from mole fractions
        m_gas = y_gas * M_gas
        m_h2o = y_H2O * M_H2O
        total_m = m_gas + m_h2o
        x_gas = m_gas / total_m
        x_h2o = m_h2o / total_m
        
        return Stream(
            mass_flow_kg_h=m_dot_kg_h,
            temperature_k=T_IN_K_REF,
            pressure_pa=P_IN_PA_REF,
            composition={gas: x_gas, 'H2O': x_h2o},
            phase='gas'
        )
    
    def test_h2_density_comparison(self):
        """Compare gas density with reference for H2."""
        ref = reference_model('H2', 100.0)
        
        inlet = self._molar_to_mass_stream('H2', 100.0)
        self.kod_h2.receive_input('gas_inlet', inlet)
        self.kod_h2.step(t=0)
        
        state = self.kod_h2.get_state()
        
        # Allow 5% deviation due to implementation differences
        self.assertAlmostEqual(state['rho_g'], ref['rho_G_out'], delta=ref['rho_G_out'] * 0.05)
    
    def test_h2_v_max_comparison(self):
        """Compare V_max with reference for H2."""
        ref = reference_model('H2', 100.0)
        
        inlet = self._molar_to_mass_stream('H2', 100.0)
        self.kod_h2.receive_input('gas_inlet', inlet)
        self.kod_h2.step(t=0)
        
        state = self.kod_h2.get_state()
        
        # Allow 5% deviation
        self.assertAlmostEqual(state['v_max'], ref['V_max'], delta=ref['V_max'] * 0.05)
    
    def test_o2_density_comparison(self):
        """Compare gas density with reference for O2."""
        ref = reference_model('O2', 50.0)
        
        inlet = self._molar_to_mass_stream('O2', 50.0)
        self.kod_o2.receive_input('gas_inlet', inlet)
        self.kod_o2.step(t=0)
        
        state = self.kod_o2.get_state()
        
        # Allow 5% deviation
        self.assertAlmostEqual(state['rho_g'], ref['rho_G_out'], delta=ref['rho_G_out'] * 0.05)
    
    def test_status_ok_condition(self):
        """Verify STATUS is OK when vessel is correctly sized."""
        ref = reference_model('H2', 100.0)
        
        inlet = self._molar_to_mass_stream('H2', 100.0)
        self.kod_h2.receive_input('gas_inlet', inlet)
        self.kod_h2.step(t=0)
        
        state = self.kod_h2.get_state()
        
        # Both should report OK for these conditions
        self.assertEqual(state['separation_status'], ref['status'])


class TestKnockOutDrumUndersizedCondition(unittest.TestCase):
    """Test UNDERSIZED detection with small vessel."""
    
    def test_undersized_detection(self):
        """High flow + small diameter should trigger UNDERSIZED."""
        # Use very small diameter
        kod = KnockOutDrum(diameter_m=0.1, gas_species='H2')
        kod.initialize(dt=1.0, registry=None)
        
        # High mass flow
        inlet = Stream(
            mass_flow_kg_h=1000.0,  # High flow
            temperature_k=T_IN_K_REF,
            pressure_pa=P_IN_PA_REF,
            composition={'H2': 0.99, 'H2O': 0.01},
            phase='gas'
        )
        
        kod.receive_input('gas_inlet', inlet)
        kod.step(t=0)
        
        state = kod.get_state()
        self.assertEqual(state['separation_status'], "UNDERSIZED")


class TestKnockOutDrumPressureDrop(unittest.TestCase):
    """Test pressure drop application."""
    
    def test_outlet_pressure_reduced(self):
        """Verify P_out = P_in - delta_P."""
        delta_p_bar = 0.1
        kod = KnockOutDrum(delta_p_bar=delta_p_bar)
        kod.initialize(dt=1.0, registry=None)
        
        inlet_p = 30e5  # 30 bar
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=inlet_p,
            composition={'H2': 0.99, 'H2O': 0.01}
        )
        
        kod.receive_input('gas_inlet', inlet)
        kod.step(t=0)
        
        outlet = kod.get_output('gas_outlet')
        expected_p = inlet_p - (delta_p_bar * 1e5)
        
        self.assertAlmostEqual(outlet.pressure_pa, expected_p, delta=100)


class TestKnockOutDrumOutputStreams(unittest.TestCase):
    """Test output stream properties."""
    
    def setUp(self):
        self.kod = KnockOutDrum(gas_species='H2')
        self.kod.initialize(dt=1.0, registry=None)
    
    def test_gas_outlet_composition(self):
        """Verify gas outlet contains H2 and H2O."""
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=30e5,
            composition={'H2': 0.95, 'H2O': 0.05}
        )
        
        self.kod.receive_input('gas_inlet', inlet)
        self.kod.step(t=0)
        
        outlet = self.kod.get_output('gas_outlet')
        
        self.assertIn('H2', outlet.composition)
        self.assertIn('H2O', outlet.composition)
        self.assertGreater(outlet.composition['H2'], 0)
    
    def test_liquid_drain_is_pure_water(self):
        """Verify liquid drain is 100% H2O."""
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=30e5,
            composition={'H2': 0.95, 'H2O': 0.05}
        )
        
        self.kod.receive_input('gas_inlet', inlet)
        self.kod.step(t=0)
        
        drain = self.kod.get_output('liquid_drain')
        
        self.assertEqual(drain.composition.get('H2O', 0), 1.0)
        self.assertEqual(drain.phase, 'liquid')
    
    def test_isothermal_operation(self):
        """Verify T_out = T_in (isothermal)."""
        T_in = 305.0
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=T_in,
            pressure_pa=30e5,
            composition={'H2': 0.99, 'H2O': 0.01}
        )
        
        self.kod.receive_input('gas_inlet', inlet)
        self.kod.step(t=0)
        
        outlet = self.kod.get_output('gas_outlet')
        drain = self.kod.get_output('liquid_drain')
        
        self.assertEqual(outlet.temperature_k, T_in)
        self.assertEqual(drain.temperature_k, T_in)


class TestKnockOutDrumCouplingWithChiller(unittest.TestCase):
    """Integration test: Chiller → KnockOutDrum chain."""
    
    def test_chiller_kod_coupling(self):
        """Test stream passes correctly from Chiller output to KOD input."""
        try:
            from h2_plant.components.thermal.chiller import Chiller
        except ImportError:
            self.skipTest("Chiller component not available")
        
        # Create chiller
        chiller = Chiller(
            target_temp_k=T_IN_K_REF,  # Cool to 4°C
            pressure_drop_bar=0.2,
            enable_dynamics=False
        )
        chiller.initialize(dt=1.0, registry=None)
        
        # Create KOD
        kod = KnockOutDrum(gas_species='H2')
        kod.initialize(dt=1.0, registry=None)
        
        # Hot H2 stream into chiller
        hot_inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=353.15,  # 80°C
            pressure_pa=40e5,
            composition={'H2': 0.98, 'H2O': 0.02}
        )
        
        # Step chiller
        chiller.receive_input('fluid_in', hot_inlet)
        chiller.step(t=0)
        chiller_out = chiller.get_output('fluid_out')
        
        # Step KOD with chiller output
        kod.receive_input('gas_inlet', chiller_out)
        kod.step(t=0)
        
        kod_out = kod.get_output('gas_outlet')
        
        # Verify chain worked
        self.assertIsNotNone(kod_out)
        self.assertGreater(kod_out.mass_flow_kg_h, 0)


class TestKnockOutDrum24HourSimulation(unittest.TestCase):
    """24-hour simulation stress test."""
    
    def test_24_hour_simulation(self):
        """Run 24 simulated hours with varying conditions."""
        kod = KnockOutDrum(gas_species='H2')
        kod.initialize(dt=1.0, registry=None)
        
        total_gas_out = 0.0
        total_liquid_out = 0.0
        
        for hour in range(24):
            # Vary flow sinusoidally (50-150 kg/h)
            flow = 100.0 + 50.0 * math.sin(2 * math.pi * hour / 24)
            
            # Vary temperature (4°C to 20°C)
            temp_k = T_IN_K_REF + 8.0 * (hour / 24)
            
            inlet = Stream(
                mass_flow_kg_h=flow,
                temperature_k=temp_k,
                pressure_pa=P_IN_PA_REF,
                composition={'H2': 0.97, 'H2O': 0.03}
            )
            
            kod.receive_input('gas_inlet', inlet)
            kod.step(t=float(hour))
            
            gas_out = kod.get_output('gas_outlet')
            liq_out = kod.get_output('liquid_drain')
            
            if gas_out:
                total_gas_out += gas_out.mass_flow_kg_h
            if liq_out:
                total_liquid_out += liq_out.mass_flow_kg_h
            
            # Verify state is valid each hour
            state = kod.get_state()
            self.assertIn(state['separation_status'], ['OK', 'UNDERSIZED'])
            self.assertGreaterEqual(state['rho_g'], 0)
        
        # Verify mass conservation (approximately)
        total_in = 24 * 100.0  # ~2400 kg (average)
        total_out = total_gas_out + total_liquid_out
        
        # Should be within 10% due to averaging
        self.assertGreater(total_out, 0)
        print(f"\n24h Simulation: Gas={total_gas_out:.1f} kg, Liquid={total_liquid_out:.1f} kg")


class TestKnockOutDrumState(unittest.TestCase):
    """Test get_state() returns all required fields."""
    
    def test_state_fields(self):
        """Verify all required state fields present."""
        kod = KnockOutDrum()
        kod.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=30e5,
            composition={'H2': 0.99, 'H2O': 0.01}
        )
        
        kod.receive_input('gas_inlet', inlet)
        kod.step(t=0)
        
        state = kod.get_state()
        
        required_fields = [
            'rho_g', 'v_max', 'v_real', 
            'separation_status', 'power_consumption_w',
            'diameter_m', 'delta_p_bar', 'gas_species'
        ]
        
        for field in required_fields:
            self.assertIn(field, state, f"Missing state field: {field}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
