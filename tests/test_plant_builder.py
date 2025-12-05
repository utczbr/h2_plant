"""Test PlantBuilder loading from h2plant.yaml"""

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from h2_plant.config.plant_builder import PlantBuilder

def test_plant_builder_yaml():
    """Test loading plant from YAML configuration."""
    print("Testing PlantBuilder with h2plant.yaml...")
    
    try:
        # Try to build from YAML
        plant = PlantBuilder.from_file("h2plant.yaml")
        # plant.build() is already called inside from_file
        
        print(f"✓ Plant loaded successfully!")
        print(f"  Registry has {len(plant.registry._components)} components")
        print(f"  Component types: {plant.registry.get_types()}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load plant: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plant_builder_yaml()
    sys.exit(0 if success else 1)
