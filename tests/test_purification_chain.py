
import pytest
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.stream import Stream
from h2_plant.core.constants import DeoxoConstants, GasConstants

# Components
from h2_plant.components.purification.deoxo_reactor import DeoxoReactor
from h2_plant.components.thermal.chiller import Chiller
from h2_plant.components.separation.tsa_unit import TSAUnit

class TestPurificationChain:
    
    def test_deoxo_chiller_tsa_chain(self):
        """
        Verify the full purification chain:
        Wet/O2-rich H2 -> Deoxo (Remove O2, Heat up) -> Chiller (Cool down) -> TSA (Remove H2O) -> Pure H2
        """
        registry = ComponentRegistry()
        dt = 0.1 # hours
        
        # 1. Instantiate Components
        deoxo = DeoxoReactor("deoxo_1")
        chiller = Chiller(
            component_id="chiller_1",
            cooling_capacity_kw=100.0,
            target_temp_k=277.15 # 4 C
        )
        tsa = TSAUnit(
            component_id="tsa_1",
            cycle_time_hours=6.0,
            regen_temp_k=523.15
        )
        
        for c in [deoxo, chiller, tsa]:
            c.initialize(dt, registry)
            
        # 2. Define Input Stream (Worst Case from Deoxo Spec)
        # 4 C, 39.55 bar, 2% O2, Saturated H2O (approx 200ppm? or more?)
        # Let's use mass flow 0.02235 kg/s (~80 kg/h)
        mass_flow = 0.02235 * 3600.0 # ~80.46 kg/h
        
        inlet_stream = Stream(
            mass_flow_kg_h=mass_flow,
            temperature_k=277.15,
            pressure_pa=39.55e5,
            composition={'H2': 0.98, 'O2': 0.02, 'H2O': 0.0002}, # Simplified
            phase='gas'
        )
        
        print("\n--- SIMULATION START ---")
        print(f"Inlet: {inlet_stream.mass_flow_kg_h:.2f} kg/h, {inlet_stream.temperature_k:.2f} K")
        
        # 3. Step 1: Deoxo
        deoxo.receive_input("inlet", inlet_stream)
        deoxo.step(0.0)
        deoxo_out = deoxo.get_output("outlet")
        
        print("\n[Step 1] Deoxo Output:")
        print(f"  T: {deoxo_out.temperature_k:.2f} K (expect > 400 K)")
        print(f"  O2: {deoxo_out.composition.get('O2', 0.0):.6f} (expect ~0)")
        print(f"  H2O: {deoxo_out.composition.get('H2O', 0.0):.6f} (expect increased)")
        
        assert deoxo_out.composition.get('O2', 0.0) < 1e-5, "Deoxo failed to remove O2"
        assert deoxo_out.temperature_k > 373.15, "Deoxo failed to heat up (reaction inactive?)"
        
        # 4. Step 2: Chiller
        # Connect Deoxo -> Chiller
        chiller.receive_input("fluid_in", deoxo_out)
        chiller.step(0.0)
        chiller_out = chiller.get_output("fluid_out")
        
        print("\n[Step 2] Chiller Output:")
        print(f"  T: {chiller_out.temperature_k:.2f} K (expect ~277.15 K)")
        
        assert abs(chiller_out.temperature_k - 277.15) < 1.0, "Chiller failed to cool gas"
        
        # 5. Step 3: TSA
        # Connect Chiller -> TSA
        tsa.receive_input("wet_h2_in", chiller_out)
        tsa.step(0.0)
        tsa_out = tsa.get_output("dry_h2_out")
        water_out = tsa.get_output("water_out")
        
        print("\n[Step 3] TSA Output:")
        print(f"  H2 Flow: {tsa_out.mass_flow_kg_h:.4f} kg/h")
        print(f"  H2O Content: {tsa_out.composition.get('H2O',0.0)} (expect ~0)")
        print(f"  Press Drop: {(chiller_out.pressure_pa - tsa_out.pressure_pa)/100.0:.2f} mbar")
        
        # Verify Refinement
        assert tsa_out.composition.get('H2O', 0.0) == 0.0, "TSA should remove water"
        
        # 6. Verify Energy Usage
        tsa_state = tsa.get_state()
        print(f"\nTSA Regen Power: {tsa_state.get('regen_power_kw', 0.0):.2f} kW")
        
        assert tsa_state.get('regen_power_kw', 0.0) > 1.0, "TSA should consume significant power (purge heating included)"


if __name__ == "__main__":
    t = TestPurificationChain()
    t.test_deoxo_chiller_tsa_chain()
