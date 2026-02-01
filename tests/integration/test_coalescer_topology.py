"""
Integration test: Coalescer in topology with Chiller and Compressor.

Tests:
1. Stream propagation (chiller → coalescer → compressor)
2. Mass balance closure
3. Property conservation (temperature, pressure)
"""
import pytest
import sys
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')

from h2_plant.components.separation.coalescer import Coalescer
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.compression.compressor import CompressorStorage
from h2_plant.core.stream import Stream
from h2_plant.core.component_registry import ComponentRegistry


class TestCoalescerTopology:
    """Integration tests for Coalescer in a process flow."""
    
    def test_chiller_coalescer_stream_propagation(self):
        """
        Test stream passes correctly from Chiller to Coalescer.
        
        Topology: Chiller → Coalescer
        """
        # Setup components
        chiller = Chiller(
            component_id="chiller_h2",
            target_temp_k=280.0,  # Cool to 7°C
            cop=4.0
        )
        coalescer = Coalescer(gas_type='H2')
        
        # Initialize
        registry = ComponentRegistry()
        chiller.initialize(dt=1.0, registry=registry)
        coalescer.initialize(dt=1.0, registry=registry)
        
        # Create inlet stream (hot H2 with aerosols)
        inlet_stream = Stream(
            mass_flow_kg_h=80.0,
            temperature_k=350.0,  # 77°C hot
            pressure_pa=35e5,
            composition={'H2': 0.99, 'H2O_liq': 0.01}
        )
        
        # Flow: inlet → chiller
        chiller.receive_input('fluid_in', inlet_stream)
        chiller.step(0)
        chiller_out = chiller.get_output('fluid_out')
        
        # Verify chiller cooled the stream
        assert chiller_out.temperature_k == pytest.approx(280.0, abs=1.0)
        assert chiller_out.mass_flow_kg_h == pytest.approx(inlet_stream.mass_flow_kg_h, rel=1e-3)
        
        # Flow: chiller → coalescer
        coalescer.receive_input('inlet', chiller_out)
        coalescer.step(1)
        coalescer_out = coalescer.get_output('outlet')
        drain = coalescer.get_output('drain')
        
        # Verify coalescer separated aerosols
        assert coalescer_out.mass_flow_kg_h > 0
        assert drain.mass_flow_kg_h > 0
        
        # Mass balance
        total_out = coalescer_out.mass_flow_kg_h + drain.mass_flow_kg_h
        assert total_out == pytest.approx(chiller_out.mass_flow_kg_h, rel=1e-6)
        
    def test_full_chain_mass_balance(self):
        """
        Test mass closure through full chain.
        
        Topology: Source → Chiller → Coalescer
        """
        chiller = Chiller(component_id="hx1", target_temp_k=280.0)
        coalescer = Coalescer(gas_type='H2')
        
        registry = ComponentRegistry()
        chiller.initialize(dt=1.0, registry=registry)
        coalescer.initialize(dt=1.0, registry=registry)
        
        # Simulate 100 hours
        inlet = Stream(
            mass_flow_kg_h=100.0,
            temperature_k=320.0,
            pressure_pa=40e5,
            composition={'H2': 0.98, 'H2O': 0.01, 'H2O_liq': 0.01}
        )
        
        total_inlet_mass = 0.0
        total_outlet_mass = 0.0
        total_drain_mass = 0.0
        
        for t in range(100):
            # Chiller step
            chiller.receive_input('fluid_in', inlet)
            chiller.step(t)
            chiller_out = chiller.get_output('fluid_out')
            
            # Coalescer step
            coalescer.receive_input('inlet', chiller_out)
            coalescer.step(t)
            
            outlet = coalescer.get_output('outlet')
            drain = coalescer.get_output('drain')
            
            # Accumulate (mass = flow * dt, dt=1h)
            total_inlet_mass += inlet.mass_flow_kg_h
            total_outlet_mass += outlet.mass_flow_kg_h
            total_drain_mass += drain.mass_flow_kg_h
        
        # Mass closure check
        mass_closure_error = abs(total_inlet_mass - total_outlet_mass - total_drain_mass) / total_inlet_mass
        assert mass_closure_error < 1e-6, f"Mass closure error: {mass_closure_error*100:.4f}%"
        
        # Check accumulated liquid
        state = coalescer.get_state()
        expected_liquid = 100 * 100 * 0.01 * 0.9999  # 100h * 100kg/h * 1% liq * 99.99% eff
        assert state['total_liquid_removed_kg'] == pytest.approx(expected_liquid, rel=1e-3)
        
    def test_pressure_drop_propagation(self):
        """
        Test pressure correctly decreases through chain.
        
        Chiller: -0.2 bar
        Coalescer: ~0 bar (very small)
        """
        chiller = Chiller(
            component_id="hx",
            target_temp_k=280.0,
            pressure_drop_bar=0.2
        )
        coalescer = Coalescer(gas_type='H2')
        
        registry = ComponentRegistry()
        chiller.initialize(dt=1.0, registry=registry)
        coalescer.initialize(dt=1.0, registry=registry)
        
        inlet = Stream(
            mass_flow_kg_h=80.0,
            temperature_k=300.0,
            pressure_pa=40e5,  # 40 bar
            composition={'H2': 0.99, 'H2O_liq': 0.01}
        )
        
        chiller.receive_input('fluid_in', inlet)
        chiller.step(0)
        after_chiller = chiller.get_output('fluid_out')
        
        coalescer.receive_input('inlet', after_chiller)
        coalescer.step(0)
        after_coalescer = coalescer.get_output('outlet')
        
        # Verify pressure drops are cumulative
        expected_after_chiller = 40.0 - 0.2  # bar
        assert after_chiller.pressure_pa / 1e5 == pytest.approx(expected_after_chiller, abs=0.01)
        
        # Coalescer adds small ΔP
        assert after_coalescer.pressure_pa < after_chiller.pressure_pa
        coalescer_dp = (after_chiller.pressure_pa - after_coalescer.pressure_pa) / 1e5
        assert coalescer_dp > 0
        assert coalescer_dp < 0.01  # Very small for clean filter
        
    def test_port_compatibility(self):
        """
        Verify port types are compatible for connection.
        """
        chiller = Chiller()
        coalescer = Coalescer()
        
        chiller_ports = chiller.get_ports()
        coalescer_ports = coalescer.get_ports()
        
        # Chiller output should connect to coalescer input
        assert 'fluid_out' in chiller_ports
        assert chiller_ports['fluid_out']['type'] == 'output'
        
        assert 'inlet' in coalescer_ports
        assert coalescer_ports['inlet']['type'] == 'input'
        
        # Coalescer has two outputs
        assert 'outlet' in coalescer_ports
        assert 'drain' in coalescer_ports


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
