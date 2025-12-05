"""
Example: Running a more advanced plant simulation with a battery,
external inputs, and an oxygen mixer.
"""

import logging
from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.simulation.event_scheduler import Event

# Setup basic logging to see simulation events and progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_advanced_simulation():
    """
    Run a simulation with a battery and grid outage event.
    """
    try:
        # Load the advanced plant configuration
        plant = PlantBuilder.from_file("configs/plant_with_battery_and_external.yaml")
        registry = plant.registry

        # Verify new components were registered by the builder
        print("Registered components:")
        for comp_id in sorted(registry.get_all_ids()):
            print(f"  - {comp_id}")

        # Create simulation engine
        engine = SimulationEngine(registry, plant.config.simulation)

        # --- Schedule a grid outage event to test the battery ---
        def grid_outage_handler(reg):
            """Event handler to simulate a grid failure."""
            if reg.has('battery'):
                battery = reg.get('battery')
                battery.set_grid_status(available=False, power_kw=0.0)
                logging.info(f"--- Event: Grid Outage! Battery taking over. SOC: {battery.soc*100:.1f}% ---")

        def grid_restore_handler(reg):
            """Event handler to restore grid power."""
            if reg.has('battery'):
                battery = reg.get('battery')
                # Assume grid comes back with plenty of power
                battery.set_grid_status(available=True, power_kw=5000.0)
                logging.info(f"--- Event: Grid Restored. Battery can now recharge. SOC: {battery.soc*100:.1f}% ---")

        # Schedule a 10-hour outage starting at hour 20
        outage_event = Event(hour=20, event_type="grid_outage", handler=grid_outage_handler)
        restore_event = Event(hour=30, event_type="grid_restore", handler=grid_restore_handler)
        
        engine.schedule_event(outage_event)
        engine.schedule_event(restore_event)

        # In a real scenario, you would set demands on components before each step
        # For this example, we'll rely on the default behavior and coordinators
        if registry.has('external_heat_source'):
            registry.get('external_heat_source').set_demand(400) # set a heat demand

        # Run the simulation
        results = engine.run()

        # --- Analyze and print key results ---
        final_states = results.get('final_states', {})
        
        if 'battery' in final_states:
            battery_state = final_states['battery']
            print("\n--- Battery Performance ---")
            print(f"  Final SOC: {battery_state.get('soc_percentage', 0):.1f}%")
        
        if 'oxygen_mixer' in final_states:
            mixer_state = final_states['oxygen_mixer']
            print("\n--- Oxygen Mixer State ---")
            print(f"  Final Mass: {mixer_state.get('mass_kg', 0):.1f} kg")
            print(f"  Fill Percentage: {mixer_state.get('fill_percentage', 0):.1f}%")

        print("\nSimulation complete.")

    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}", exc_info=True)


if __name__ == '__main__':
    run_advanced_simulation()
