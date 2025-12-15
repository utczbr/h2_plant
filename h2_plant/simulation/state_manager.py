"""
State Management for Simulation Checkpointing and Persistence.

This module handles simulation state persistence, enabling:
- Checkpoint creation at regular intervals.
- Recovery from interrupted simulations.
- Final results export in multiple formats.

Serialization Formats:
    - **JSON**: Human-readable, portable, handles NumPy via NpEncoder.
    - **Pickle**: Fast, compact, Python-native.
    - **HDF5**: Optimized for large arrays (requires h5py).

Checkpoint Structure:
    {
        'hour': int,           # Simulation hour at checkpoint
        'timestamp': str,      # ISO timestamp of checkpoint creation
        'component_states': {  # Per-component state dictionaries
            'component_id': {...},
            ...
        },
        'metadata': {...}      # Optional simulation metadata
    }
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


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types.

    Converts NumPy arrays and scalar types to JSON-serializable
    Python equivalents.
    """

    def default(self, obj):
        """
        Convert non-serializable objects to JSON-compatible types.

        Args:
            obj: Object to serialize.

        Returns:
            JSON-serializable equivalent.
        """
        if isinstance(obj, np.bool_):
            return bool(obj)
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

    Provides save/load functionality for simulation checkpoints in
    multiple formats with automatic format detection on load.

    Attributes:
        output_dir (Path): Base output directory.
        checkpoint_dir (Path): Subdirectory for checkpoint files.

    Example:
        >>> manager = StateManager(output_dir=Path("checkpoints"))
        >>> manager.save_checkpoint(
        ...     hour=168,
        ...     component_states=registry.get_all_states()
        ... )
        >>> data = manager.load_checkpoint("checkpoint_hour_168.json")
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the state manager.

        Args:
            output_dir (Path): Directory for checkpoint storage.
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
        Save simulation checkpoint to disk.

        Creates a snapshot of all component states for later recovery.
        Checkpoint filename includes the simulation hour for identification.

        Args:
            hour (int): Current simulation hour.
            component_states (Dict): State dictionaries from all components.
            metadata (Dict, optional): Additional metadata to store.
            format (str): Storage format ('json', 'pickle', 'hdf5').
                Default: 'json'.

        Returns:
            Path: Path to saved checkpoint file.
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
        Load simulation checkpoint from disk.

        Automatically detects format from file extension and loads
        checkpoint data for simulation recovery.

        Args:
            checkpoint_path (Path | str): Path to checkpoint file.

        Returns:
            Dict[str, Any]: Checkpoint data dictionary.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If checkpoint format is unsupported.
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
        Save final simulation results to JSON.

        Args:
            results (Dict): Results dictionary.
            output_path (Path): Destination file path.
        """
        self._save_json(results, output_path)
        logger.info(f"Results saved to: {output_path}")

    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """
        Save data as JSON with NumPy type handling.

        Args:
            data (Dict): Data to serialize.
            path (Path): Destination file path.
        """
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=NpEncoder)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """
        Load data from JSON file.

        Args:
            path (Path): Source file path.

        Returns:
            Dict[str, Any]: Loaded data.
        """
        with open(path, 'r') as f:
            return json.load(f)

    def _save_pickle(self, data: Dict[str, Any], path: Path) -> None:
        """
        Save data as pickle with highest protocol.

        Args:
            data (Dict): Data to serialize.
            path (Path): Destination file path.
        """
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, path: Path) -> Dict[str, Any]:
        """
        Load data from pickle file.

        Args:
            path (Path): Source file path.

        Returns:
            Dict[str, Any]: Loaded data.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_hdf5(self, data: Dict[str, Any], path: Path) -> None:
        """
        Save data as HDF5 for efficient large array storage.

        Component states are stored as HDF5 groups with attributes
        for scalar values and datasets for arrays.

        Args:
            data (Dict): Data to serialize.
            path (Path): Destination file path.
        """
        with h5py.File(path, 'w') as f:
            f.attrs['hour'] = data['hour']
            f.attrs['timestamp'] = data['timestamp']

            for comp_id, comp_state in data['component_states'].items():
                group = f.create_group(comp_id)
                for key, value in comp_state.items():
                    try:
                        group.attrs[key] = value
                    except TypeError:
                        try:
                            group.create_dataset(key, data=np.array(value))
                        except Exception as e:
                            logger.warning(f"Could not save key '{key}' for component '{comp_id}' to HDF5: {e}")

    def _load_hdf5(self, path: Path) -> Dict[str, Any]:
        """
        Load data from HDF5 file.

        Args:
            path (Path): Source file path.

        Returns:
            Dict[str, Any]: Loaded data.
        """
        data = {'component_states': {}}

        with h5py.File(path, 'r') as f:
            data['hour'] = f.attrs['hour']
            data['timestamp'] = f.attrs['timestamp']

            for comp_id in f.keys():
                group = f[comp_id]
                comp_state = dict(group.attrs)
                for key, dset in group.items():
                    comp_state[key] = dset[()]
                data['component_states'][comp_id] = comp_state

        return data

    def list_checkpoints(self) -> list[Path]:
        """
        List all available checkpoints.

        Scans checkpoint directory for files in all supported formats.

        Returns:
            list[Path]: Sorted list of checkpoint file paths.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_hour_*.json"))
        checkpoints.extend(sorted(self.checkpoint_dir.glob("checkpoint_hour_*.pickle")))
        checkpoints.extend(sorted(self.checkpoint_dir.glob("checkpoint_hour_*.hdf5")))

        return checkpoints
