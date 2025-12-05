"""
State management for simulation checkpointing and persistence.

Handles:
- Checkpoint creation and loading
- State serialization (HDF5, JSON, Parquet)
- Results export
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logging.warning("h5py not available - HDF5 checkpoints disabled")

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super(NpEncoder, self).default(obj)

class StateManager:
    """
    Manages simulation state persistence and checkpointing.
    
    Example:
        manager = StateManager(output_dir=Path("checkpoints"))
        
        # Save checkpoint
        manager.save_checkpoint(
            hour=168,
            component_states=registry.get_all_states()
        )
        
        # Load checkpoint
        data = manager.load_checkpoint("checkpoint_hour_168.json")
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize state manager.
        
        Args:
            output_dir: Directory for checkpoint storage
        """
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        hour: int,
        component_states: Dict[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "json"
    ) -> Path:
        """
        Save simulation checkpoint.
        
        Args:
            hour: Current simulation hour
            component_states: State from all components
            metadata: Additional metadata to store
            format: Storage format ('json', 'pickle', 'hdf5')
            
        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().isoformat()
        
        checkpoint_data = {
            'hour': hour,
            'timestamp': timestamp,
            'component_states': component_states,
            'metadata': metadata or {}
        }
        
        filename = f"checkpoint_hour_{hour}.{format}"
        checkpoint_path = self.checkpoint_dir / filename
        
        if format == "json":
            self._save_json(checkpoint_data, checkpoint_path)
        elif format == "pickle":
            self._save_pickle(checkpoint_data, checkpoint_path)
        elif format == "hdf5" and HDF5_AVAILABLE:
            self._save_hdf5(checkpoint_data, checkpoint_path)
        else:
            logger.warning(f"Unsupported checkpoint format '{format}' or h5py not available. Defaulting to json.")
            filename = f"checkpoint_hour_{hour}.json"
            checkpoint_path = self.checkpoint_dir / filename
            self._save_json(checkpoint_data, checkpoint_path)
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path | str) -> Dict[str, Any]:
        """
        Load simulation checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        format = checkpoint_path.suffix.lstrip('.')
        
        if format == "json":
            return self._load_json(checkpoint_path)
        elif format == "pickle":
            return self._load_pickle(checkpoint_path)
        elif format == "hdf5" and HDF5_AVAILABLE:
            return self._load_hdf5(checkpoint_path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {format}")
    
    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        Save final simulation results.
        
        Args:
            results: Results dictionary
            output_path: Path to save results
        """
        self._save_json(results, output_path)
        logger.info(f"Results saved to: {output_path}")
    
    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as JSON, handling numpy types."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=NpEncoder)
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load data from JSON."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _save_pickle(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as pickle."""
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_pickle(self, path: Path) -> Dict[str, Any]:
        """Load data from pickle."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _save_hdf5(self, data: Dict[str, Any], path: Path) -> None:
        """Save data as HDF5 (requires h5py)."""
        with h5py.File(path, 'w') as f:
            f.attrs['hour'] = data['hour']
            f.attrs['timestamp'] = data['timestamp']
            
            # Store component states as groups
            for comp_id, comp_state in data['component_states'].items():
                group = f.create_group(comp_id)
                for key, value in comp_state.items():
                    try:
                        group.attrs[key] = value
                    except TypeError:
                        # Handle non-scalar attributes like lists/arrays
                        # Convert lists to numpy arrays to store as datasets
                        try:
                            group.create_dataset(key, data=np.array(value))
                        except Exception as e:
                            logger.warning(f"Could not save key '{key}' for component '{comp_id}' to HDF5: {e}")

    def _load_hdf5(self, path: Path) -> Dict[str, Any]:
        """Load data from HDF5."""
        data = {'component_states': {}}
        
        with h5py.File(path, 'r') as f:
            data['hour'] = f.attrs['hour']
            data['timestamp'] = f.attrs['timestamp']
            
            for comp_id in f.keys():
                group = f[comp_id]
                comp_state = dict(group.attrs)
                for key, dset in group.items():
                    comp_state[key] = dset[()] # Read dataset
                data['component_states'][comp_id] = comp_state
        
        return data
    
    def list_checkpoints(self) -> list[Path]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint file paths
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_hour_*.json"))
        checkpoints.extend(sorted(self.checkpoint_dir.glob("checkpoint_hour_*.pickle")))
        checkpoints.extend(sorted(self.checkpoint_dir.glob("checkpoint_hour_*.hdf5")))
        
        return checkpoints
