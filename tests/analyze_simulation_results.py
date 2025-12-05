import sys
import os
import pandas as pd

# Ensure project root is in path
sys.path.append(os.getcwd())

from h2_plant.orchestrator import Orchestrator

def test_fuller_tank():
    print("\n\n=== TEST 1: Fuller Tank (Start @ 500 kg / ~100 bar) ===")
    scenarios_dir = os.path.join(os.getcwd(), "scenarios")
    orchestrator = Orchestrator(scenarios_dir)
    
    # Manually set Tank Level
    tank = orchestrator.components.get("H2_Tank")
    if tank:
        tank.current_level_kg = 500.0
        # Force update pressure
        tank.step(0, 0, 0) 
        print(f"Initial Tank State: Level={tank.current_level_kg} kg, Pressure={tank.pressure_bar} bar")
    
    # Run for 24 hours
    history = orchestrator.run_simulation(hours=24)
    df = pd.DataFrame(history)
    
    # Check Compressor Power
    avg_power = df['compressor_power_kw'].mean()
    print(f"Average Compressor Power: {avg_power:.2f} kW")
    
    # Show first few hours
    print(df[['minute', 'tank_pressure_bar', 'compressor_power_kw']].head(5).to_markdown(index=False, floatfmt=".2f"))

def test_long_simulation():
    print("\n\n=== TEST 2: Long Simulation (30 Days / 720 Hours) ===")
    scenarios_dir = os.path.join(os.getcwd(), "scenarios")
    orchestrator = Orchestrator(scenarios_dir)
    
    # Run for 720 hours
    history = orchestrator.run_simulation(hours=720)
    df = pd.DataFrame(history)
    
    # Sample every 24 hours
    daily_df = df[df['minute'] % 1440 == 0][['minute', 'h2_kg', 'compressor_power_kw', 'tank_level_kg', 'tank_pressure_bar']].reset_index(drop=True)
    daily_df['Day'] = daily_df['minute'] / 1440
    
    print("\n--- Daily Summary ---")
    print(daily_df[['Day', 'tank_level_kg', 'tank_pressure_bar', 'compressor_power_kw']].to_markdown(index=False, floatfmt=".2f"))
    
    print(f"\nFinal Tank Level: {df['tank_level_kg'].iloc[-1]:.2f} kg")
    print(f"Max Compressor Power: {df['compressor_power_kw'].max():.2f} kW")

if __name__ == "__main__":
    # analyze_results() # Skip default
    test_fuller_tank()
    test_long_simulation()
