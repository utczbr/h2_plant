"""
Complete dual-path hydrogen production simulation.

Demonstrates pathway integration with configuration-driven setup.
"""

from h2_plant.config.plant_builder import PlantBuilder
from h2_plant.pathways.isolated_production_path import IsolatedProductionPath
from h2_plant.pathways.dual_path_coordinator import DualPathCoordinator
from h2_plant.core.enums import AllocationStrategy
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_dual_path_simulation():
    """Run 1-week dual-path simulation with cost optimization."""
    
    # Load plant configuration using the builder. This will create and register
    # all the base components like electrolyzer, ATR, tanks, etc.
    try:
        plant = PlantBuilder.from_file("configs/plant_baseline.yaml")
        registry = plant.registry
    except Exception as e:
        logging.error(f"Failed to build plant from configuration: {e}")
        return

    # Manually create the orchestration components (Pathways and Coordinator)
    # This shows how the higher-level logic is layered on top of the base components.
    
    # Create Electrolyzer Pathway
    if registry.has('electrolyzer') and registry.has('electrolyzer_hp_tanks') and registry.has('filling_compressor'):
        electrolyzer_path = IsolatedProductionPath(
            pathway_id='electrolyzer_path',
            source_id='electrolyzer',
            lp_storage_id='lp_tanks',  # Assuming a common lp_tanks for simplicity in this example
            hp_storage_id='electrolyzer_hp_tanks',
            compressor_id='filling_compressor'
        )
        registry.register('electrolyzer_path', electrolyzer_path, component_type='pathway')

    # Create ATR Pathway
    if registry.has('atr') and registry.has('atr_hp_tanks') and registry.has('filling_compressor'):
        atr_path = IsolatedProductionPath(
            pathway_id='atr_path',
            source_id='atr',
            lp_storage_id='lp_tanks', # Assuming a common lp_tanks
            hp_storage_id='atr_hp_tanks',
            compressor_id='filling_compressor'
        )
        registry.register('atr_path', atr_path, component_type='pathway')

    # Create Coordinator
    pathway_ids = [comp_id for comp_id, comp in registry._components.items() if isinstance(comp, IsolatedProductionPath)]
    if not pathway_ids:
        logging.error("No production pathways were created. Aborting simulation.")
        return
        
    coordinator = DualPathCoordinator(
        pathway_ids=pathway_ids,
        allocation_strategy=AllocationStrategy.COST_OPTIMAL,
        demand_scheduler_id='demand_scheduler'
    )
    registry.register('coordinator', coordinator, component_type='coordinator')
    
    # Initialize all components (including the new orchestration ones)
    registry.initialize_all(dt=1.0)
    
    # Run simulation (1 week = 168 hours)
    logging.info("Running dual-path simulation...")
    
    for hour in range(168):
        # The coordinator will orchestrate the pathways, which in turn orchestrate the components.
        # So we only need to step the coordinator. For simplicity, we step all components.
        # A more advanced engine would have a step order.
        registry.step_all(hour)
        
        # Daily reporting
        if hour > 0 and hour % 24 == 0:
            state = coordinator.get_state()
            
            logging.info(f"\nDay {hour//24}:")
            logging.info(f"  Total Demand: {state['total_demand_kg']:.1f} kg")
            logging.info(f"  Total Delivered: {state['total_delivered_kg']:.1f} kg")
            logging.info(f"  Shortfall: {state['demand_shortfall_kg']:.1f} kg")
            logging.info(f"  Electrolyzer Allocation: {state['pathway_allocations'].get('electrolyzer_path', 0.0):.1f} kg")
            logging.info(f"  ATR Allocation: {state['pathway_allocations'].get('atr_path', 0.0):.1f} kg")
    
    # Final report
    final_state = coordinator.get_state()
    
    logging.info("\n=== Final Results ===")
    logging.info(f"Total Production: {final_state['cumulative_production_kg']:.1f} kg")
    logging.info(f"Total Delivery: {final_state['cumulative_delivery_kg']:.1f} kg")
    logging.info(f"Total Cost: ${final_state['cumulative_cost']:.2f}")
    logging.info(f"Average Cost: ${coordinator.get_weighted_average_cost():.2f}/kg")
    
    utilization = coordinator.get_pathway_utilization()
    logging.info("\nPathway Utilization:")
    for pathway_id, util in utilization.items():
        logging.info(f"  {pathway_id}: {util*100:.1f}%")


if __name__ == '__main__':
    run_dual_path_simulation()
