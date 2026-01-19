#!/usr/bin/env python3
"""
Diagnostic script to trace tank filling/draining behavior.
Runs a short simulation and logs:
- Tank mass and pressure
- Demand signal from DischargeStation  
- Actual discharge rate
- Production rate
"""
import sys
sys.path.insert(0, '/home/stuart/Documentos/Planta Hidrogenio')

from h2_plant.run_integrated_simulation import run_with_dispatch_strategy
import numpy as np

def main():
    print("=" * 60)
    print("TANK FLOW DIAGNOSTIC")
    print("=" * 60)
    
    # Run short simulation (48 hours to see full cycle)
    print("\nRunning 48-hour simulation...")
    history = run_with_dispatch_strategy(
        scenarios_dir="/home/stuart/Documentos/Planta Hidrogenio/scenarios",
        hours=48,
        strategy="REFERENCE_HYBRID"
    )
    
    # Extract key metrics
    minutes = history.get('minute', np.array([]))
    hours = minutes / 60.0
    
    # Tank metrics
    tank_mass = history.get('LP_Storage_Tank_total_mass_kg', np.zeros_like(hours))
    tank_pressure = history.get('LP_Storage_Tank_avg_pressure_bar', np.zeros_like(hours))
    
    # Flow metrics
    tank_discharged = history.get('LP_Storage_Tank_total_discharged_kg', np.zeros_like(hours))
    tank_filled = history.get('LP_Storage_Tank_total_filled_kg', np.zeros_like(hours))
    
    # Compressor flow
    comp_flow = history.get('HP_Compressor_S2_actual_mass_flow_kg_h', np.zeros_like(hours))
    
    # Station demand
    station_demand = history.get('Truck_Station_1_total_demand_signal_kg_h', np.zeros_like(hours))
    
    # Calculate net flow (production - discharge)
    production_rate = history.get('H2_Production_Mixer_mass_flow_kg_h', np.zeros_like(hours))
    
    print("\n" + "=" * 80)
    print(f"{'Hour':>6} {'Tank Mass':>12} {'Tank P':>10} {'Prod Rate':>12} {'Demand':>12} {'Comp Flow':>12} {'Net':>10}")
    print(f"{'':>6} {'(kg)':>12} {'(bar)':>10} {'(kg/h)':>12} {'(kg/h)':>12} {'(kg/h)':>12} {'(kg/h)':>10}")
    print("=" * 80)
    
    # Print every 4 hours
    for i, h in enumerate(hours):
        if i % int(4 * 60 / (hours[1] - hours[0]) if len(hours) > 1 else 1) == 0:
            net = production_rate[i] - comp_flow[i] if i < len(production_rate) else 0
            print(f"{h:>6.1f} {tank_mass[i]:>12.1f} {tank_pressure[i]:>10.1f} {production_rate[i]:>12.1f} {station_demand[i]:>12.1f} {comp_flow[i]:>12.1f} {net:>+10.1f}")
    
    print("=" * 80)
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Initial Tank Mass: {tank_mass[0]:.1f} kg at {tank_pressure[0]:.1f} bar")
    print(f"  Final Tank Mass:   {tank_mass[-1]:.1f} kg at {tank_pressure[-1]:.1f} bar")
    print(f"  Total Filled:      {tank_filled[-1] - tank_filled[0]:.1f} kg")
    print(f"  Total Discharged:  {tank_discharged[-1] - tank_discharged[0]:.1f} kg")
    print(f"  Net Change:        {tank_mass[-1] - tank_mass[0]:+.1f} kg")
    
    # Time to reach 30 bar
    idx_30bar = np.argmax(tank_pressure >= 30.0)
    if tank_pressure[idx_30bar] >= 30.0:
        print(f"\n  Time to reach 30 bar: {hours[idx_30bar]:.1f} hours (idx {idx_30bar})")
    else:
        print(f"\n  Tank never reached 30 bar (max: {tank_pressure.max():.1f} bar)")
    
    # Check if discharging ever happens
    if comp_flow.max() > 0:
        print(f"  Compressor flow started at: {hours[np.argmax(comp_flow > 0)]:.1f} hours")
        print(f"  Peak compressor flow: {comp_flow.max():.1f} kg/h")
    else:
        print(f"  Compressor flow: NEVER STARTED (always 0)")
        print(f"  Station demand max: {station_demand.max():.1f} kg/h")

if __name__ == "__main__":
    main()
