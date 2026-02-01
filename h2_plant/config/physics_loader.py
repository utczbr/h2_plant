"""
Physics Parameter Loader
Loads physical constants and engineering parameters from YAML configuration.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_physics_parameters(config_path: str = None) -> Dict[str, Any]:
    """
    Load physics parameters from YAML file.
    
    Args:
        config_path: Path to YAML file. If None, tries default locations.
        
    Returns:
        Dictionary containing physics parameters.
    """
    if config_path:
        paths_to_try = [Path(config_path)]
    else:
        # Default locations
        root_dir = Path(__file__).parent.parent.parent
        paths_to_try = [
            root_dir / "scenarios" / "physics_parameters.yaml",  # User's location
            root_dir / "configs" / "physics_parameters.yaml",
            Path("scenarios/physics_parameters.yaml"),
            Path("configs/physics_parameters.yaml"),
            Path("physics_parameters.yaml")
        ]
    
    for path in paths_to_try:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded physics parameters from {path}")
                return config
            except Exception as e:
                logger.error(f"Failed to load physics parameters from {path}: {e}")
                raise
    
    logger.warning("Physics parameters file not found. Using hardcoded defaults.")
    return {}
