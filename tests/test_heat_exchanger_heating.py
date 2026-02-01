"""
Test script for HeatExchanger heating functionality.

Verifies that the refactored HeatExchanger can heat a stream when
target_temp > inlet_temp, and respects capacity limits.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from h2_plant.components.thermal.heat_exchanger import HeatExchanger
from h2_plant.core.stream import Stream

def test_heating():
    print("\n=== HEAT EXCHANGER HEATING TEST ===")
    
    # Setup: Target 40°C, Max Capacity 100 kW
    hx = HeatExchanger(
        component_id="HX_Test",
        max_heat_removal_kw=100.0,
        target_outlet_temp_c=40.0
    )
    
    class MockRegistry:
        pass
    
    hx.initialize(dt=1/60, registry=MockRegistry())
    
    # Case 1: Unconstrained Heating
    # Input: 4°C, 150 kg/h
    # Expected: Output 40°C, Negative Heat Removed (Injection)
    print("\n[Case 1: Unconstrained Heating]")
    in_stream = Stream(
        mass_flow_kg_h=150.0,
        temperature_k=277.15, # 4°C
        pressure_pa=30e5,     # 30 bar
        composition={'H2': 1.0}
    )
    
    hx.receive_input('h2_in', in_stream, 'hydrogen')
    hx.step(t=0.0)
    
    state = hx.get_state()
    out_stream = hx.get_output('h2_out')
    
    print(f"Inlet Temp: {in_stream.temperature_k - 273.15:.2f} °C")
    print(f"Target Temp: {hx.target_outlet_temp_c:.2f} °C")
    print(f"Outlet Temp: {out_stream.temperature_k - 273.15:.2f} °C")
    print(f"Heat Transfer: {state['heat_removed_kw']:.4f} kW (Negative = Heating)")
    
    if abs(out_stream.temperature_k - 313.15) < 0.1:
        print("✓ Temperature reached target")
    else:
        print("✗ Temperature mismatch")
        
    if state['heat_removed_kw'] < 0:
        print("✓ Heat transfer is negative (Heating)")
    else:
        print("✗ Heat transfer sign incorrect")

    # Case 2: Constrained Heating
    # Reduce capacity to 1 kW (very low)
    print("\n[Case 2: Constrained Heating]")
    hx.max_heat_removal_kw = 1.0
    
    hx.step(t=0.1) # Next step
    state = hx.get_state()
    out_stream = hx.get_output('h2_out')
    
    print(f"Max Capacity: {hx.max_heat_removal_kw} kW")
    print(f"Outlet Temp: {out_stream.temperature_k - 273.15:.2f} °C")
    print(f"Heat Transfer: {state['heat_removed_kw']:.4f} kW")
    
    if out_stream.temperature_k < 313.15 and out_stream.temperature_k > 277.15:
        print("✓ Output temperature between inlet and target (Limited)")
    else:
        print("✗ Limiting logic failed")
        
    if abs(state['heat_removed_kw'] + 1.0) < 0.01:
         print("✓ Limit applied correctly (-1.0 kW)")
    else:
         print("✗ Limit value mismatch")

if __name__ == "__main__":
    test_heating()
