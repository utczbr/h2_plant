
import logging
from pathlib import Path
import sys
import os

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.components.separation.hydrogen_cyclone import HydrogenMultiCyclone
from h2_plant.economics.capex_generator import CapexGenerator

# Mock objects
class MockMonitoring:
    def __init__(self):
        self.component_metrics = {}

def reproduce_issue():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Setup Component
    registry = ComponentRegistry()
    
    # Create Cyclone components as defined in topology
    # IDs: SOEC_H2_Cyclone_1 to SOEC_H2_Cyclone_6
    cyclone = HydrogenMultiCyclone(
        element_diameter_mm=50.0,
        volume_m3=0.15 # Explicit volume
    )
    registry.register("SOEC_H2_Cyclone_1", cyclone)

    # 2. Initialize Generator
    config_path = project_root / "scenarios" / "Economics" / "equipment_mappings.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    generator = CapexGenerator.from_yaml(config_path)
    
    # Filter mappings to only Cyclone for clarity
    generator.mappings = [m for m in generator.mappings if "CYC-SOEC" in m.tag]
    logger.info(f"Testing {len(generator.mappings)} mappings: {[m.tag for m in generator.mappings]}")

    # 3. Generate Report
    report = generator.generate(
        registry=registry, 
        monitoring=MockMonitoring(),
        simulation_hours=100
    )

    # 4. Inspect Results
    for entry in report.entries:
        print(f"\nTag: {entry.tag}")
        print(f"Capacity: {entry.design_capacity} {entry.capacity_unit} ({entry.capacity_source})")
        print(f"Cost: C_BM=${entry.C_BM} (Using {entry.cost_source})")
        print(f"Warnings: {entry.warnings}")
        print(f"Errors: {entry.errors}")
        print(f"Notes: {entry.notes}")

if __name__ == "__main__":
    reproduce_issue()
