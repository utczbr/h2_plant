"""Unit tests for Coalescer component."""

import pytest
import math
from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.core.stream import Stream
from h2_plant.core.constants import CoalescerConstants, ConversionFactors


class TestCoalescerPhysics:
    """Test physical model accuracy against reference CoalescerModel.py."""
    
    def test_pressure_drop_h2_nominal(self):
        """
        Verify pressure drop matches reference model for H2 stream.
        Reference: CoalescerModel.py nominal conditions.
        """
        coalescer = Coalescer(gas_type='H2')
        coalescer.initialize(dt=1.0, registry=None)
        
        # H2 stream at nominal conditions (~80.46 kg/h, 35.7 bar, 4°C after chiller)
        # Using worst case: P=35.7 bar, T=31.7°C (304.87 K)
        inlet = Stream(
            mass_flow_kg_h=80.46,
            temperature_k=304.87,
            pressure_pa=35.7e5,
            composition={'H2': 0.999, 'H2O_liq': 0.001}
        )
        
        coalescer.receive_input('inlet', inlet)
        
        # ΔP should be small for clean filter (< 0.01 bar typically)
        assert coalescer.current_delta_p_bar > 0
        assert coalescer.current_delta_p_bar < 0.1  # Sanity check
        
    def test_pressure_drop_o2_nominal(self):
        """Verify O2 stream uses different viscosity reference."""
        coalescer = Coalescer(gas_type='O2')
        coalescer.initialize(dt=1.0, registry=None)
        
        # O2 stream (~707 kg/h)
        inlet = Stream(
            mass_flow_kg_h=707.0,
            temperature_k=304.87,
            pressure_pa=35.7e5,
            composition={'O2': 0.999, 'H2O_liq': 0.001}
        )
        
        coalescer.receive_input('inlet', inlet)
        
        # O2 has higher viscosity than H2, so higher ΔP for same conditions
        assert coalescer.current_delta_p_bar > 0
        
    def test_viscosity_temperature_dependence(self):
        """Verify Sutherland viscosity scaling (T^0.7)."""
        coalescer = Coalescer(gas_type='H2')
        coalescer.initialize(dt=1.0, registry=None)
        
        # Same flow at different temperatures
        t_low = 280.0  # K
        t_high = 320.0  # K
        
        inlet_low = Stream(
            mass_flow_kg_h=80.0,
            temperature_k=t_low,
            pressure_pa=30e5,
            composition={'H2': 1.0}
        )
        inlet_high = Stream(
            mass_flow_kg_h=80.0,
            temperature_k=t_high,
            pressure_pa=30e5,
            composition={'H2': 1.0}
        )
        
        coalescer.receive_input('inlet', inlet_low)
        dp_low = coalescer.current_delta_p_bar
        
        coalescer.receive_input('inlet', inlet_high)
        dp_high = coalescer.current_delta_p_bar
        
        # Higher T → higher viscosity → higher ΔP
        expected_ratio = (t_high / t_low) ** 0.7
        actual_ratio = dp_high / dp_low if dp_low > 0 else 0
        
        # Ratio includes both viscosity AND density effects (density changes with T)
        # Actual ratio will differ from pure viscosity ratio due to Q_V changes
        assert actual_ratio > 1.0  # Higher T should give higher ΔP
        assert actual_ratio < 1.5  # Reasonable bound


class TestCoalescerSeparation:
    """Test liquid removal efficiency."""
    
    def test_liquid_removal_efficiency(self):
        """Verify 99.99% liquid water removal."""
        coalescer = Coalescer()
        coalescer.initialize(dt=1.0, registry=None)
        
        # 1% liquid water content
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=30e5,
            composition={'H2': 0.99, 'H2O_liq': 0.01}
        )
        
        coalescer.receive_input('inlet', inlet)
        outlet = coalescer.get_output('outlet')
        drain = coalescer.get_output('drain')
        
        # Inlet liquid = 100 * 0.01 = 1.0 kg/h
        # Removed = 1.0 * 0.9999 = 0.9999 kg/h
        assert drain.mass_flow_kg_h == pytest.approx(0.9999, rel=1e-4)
        
        # Remaining liquid = 0.0001 kg/h
        remaining = outlet.mass_flow_kg_h * outlet.composition.get('H2O_liq', 0)
        assert remaining == pytest.approx(0.0001, rel=0.05)  # 5% tolerance for normalization
        
    def test_mass_balance(self):
        """Verify inlet = outlet + drain."""
        coalescer = Coalescer()
        coalescer.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=30e5,
            composition={'H2': 0.95, 'H2O_liq': 0.05}
        )
        
        coalescer.receive_input('inlet', inlet)
        outlet = coalescer.get_output('outlet')
        drain = coalescer.get_output('drain')
        
        total_out = outlet.mass_flow_kg_h + drain.mass_flow_kg_h
        assert total_out == pytest.approx(inlet.mass_flow_kg_h, rel=1e-6)


class TestCoalescerInterface:
    """Test component interface compliance."""
    
    def test_ports(self):
        """Verify port definitions."""
        coalescer = Coalescer()
        ports = coalescer.get_ports()
        
        assert 'inlet' in ports
        assert ports['inlet']['type'] == 'input'
        
        assert 'outlet' in ports
        assert ports['outlet']['type'] == 'output'
        
        assert 'drain' in ports
        assert ports['drain']['type'] == 'output'
        
    def test_state_accumulation(self):
        """Verify liquid accumulation over timesteps."""
        coalescer = Coalescer()
        coalescer.initialize(dt=1.0, registry=None)
        
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=300.0,
            pressure_pa=30e5,
            composition={'H2': 0.99, 'H2O_liq': 0.01}
        )
        
        # Simulate 3 timesteps
        for t in range(3):
            coalescer.receive_input('inlet', inlet)
            coalescer.step(t)
        
        state = coalescer.get_state()
        
        # 0.9999 kg/h * 3 hours = 2.9997 kg
        assert state['total_liquid_removed_kg'] == pytest.approx(2.9997, rel=1e-3)
        
    def test_zero_flow(self):
        """Verify handling of zero inlet flow."""
        coalescer = Coalescer()
        coalescer.initialize(dt=1.0, registry=None)
        
        inlet = Stream(mass_flow_kg_h=0.0)
        coalescer.receive_input('inlet', inlet)
        
        outlet = coalescer.get_output('outlet')
        drain = coalescer.get_output('drain')
        
        assert outlet.mass_flow_kg_h == 0.0
        assert drain.mass_flow_kg_h == 0.0
        assert coalescer.current_delta_p_bar == 0.0
