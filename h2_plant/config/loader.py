import yaml
import os
from typing import Dict, Any
import logging
from h2_plant.config.models import (
    SimulationContext, PhysicsConfig, TopologyConfig, 
    SimulationConfig, EconomicsConfig
)

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads and validates configuration files into a SimulationContext.
    """
    def __init__(self, scenarios_dir: str):
        self.scenarios_dir = scenarios_dir

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.scenarios_dir, filename)
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise

    def load_context(self) -> SimulationContext:
        """
        Loads all required configs and returns a validated SimulationContext.
        """
        logger.info("Loading configuration context...")

        # 1. Load Raw YAMLs
        physics_data = self._load_yaml("physics_parameters.yaml")
        topology_data = self._load_yaml("plant_topology.yaml")
        sim_data = self._load_yaml("simulation_config.yaml")
        econ_data = self._load_yaml("economics_parameters.yaml")

        # 2. Validate & Create Models
        # Note: We might need to map raw yaml keys to model fields if they differ.
        # Assuming direct mapping for now.

        physics = PhysicsConfig(**physics_data)
        topology = TopologyConfig(**topology_data)
        simulation = SimulationConfig(**sim_data)
        economics = EconomicsConfig(**econ_data)

        # 3. Assemble Context
        context = SimulationContext(
            physics=physics,
            topology=topology,
            simulation=simulation,
            economics=economics
        )
        
        logger.info("Configuration context loaded and validated.")
        return context
