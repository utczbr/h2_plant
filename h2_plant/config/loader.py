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
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise

    def load_context(self, topology_file: str = "plant_topology.yaml") -> SimulationContext:
        """
        Loads all required configs and returns a validated SimulationContext.
        
        Args:
            topology_file (str): Topology file name or path. If absolute path,
                                 used directly. Otherwise, relative to scenarios_dir.
                                 Default: "plant_topology.yaml"
        """
        logger.info(f"Loading configuration context with topology: {topology_file}")

        # 1. Load Raw YAMLs
        physics_data = self._load_yaml("physics_parameters.yaml")
        
        # Handle topology file - support absolute or relative paths
        if os.path.isabs(topology_file):
            with open(topology_file, 'r', encoding='utf-8') as f:
                topology_data = yaml.safe_load(f)
        else:
            topology_data = self._load_yaml(topology_file)
        
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
