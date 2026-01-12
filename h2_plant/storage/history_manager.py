"""
Chunked History Manager for memory-efficient simulation data storage.

This module provides streaming history storage that keeps memory usage constant
regardless of simulation length by periodically flushing data chunks to disk.

Architecture:
    - In-memory buffer for current chunk (configurable size, default 10,000 steps)
    - Periodic flush to Parquet files (10x smaller than CSV)
    - Checkpoint metadata for crash recovery
    - Lazy-load API for graph generation with column filtering

Memory Usage:
    - Constant ~100 MB regardless of simulation length
    - 1-year simulation: 4.4 GB → 100 MB
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import json
import gc
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a single history chunk."""
    chunk_index: int
    start_step: int
    end_step: int
    filepath: Path
    columns: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class CheckpointState:
    """Checkpoint state for crash recovery."""
    simulation_id: str
    total_steps_target: int
    total_steps_completed: int
    chunk_index: int
    step_in_chunk: int
    columns: List[str]
    chunks: List[Dict[str, Any]]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ChunkedHistoryManager:
    """
    Memory-efficient history storage with periodic disk flushing.
    
    Instead of keeping all history in RAM, this manager:
    1. Allocates a fixed-size buffer for the current chunk
    2. Flushes to Parquet when chunk is full
    3. Provides lazy-loading API for graph generation
    
    Usage:
        manager = ChunkedHistoryManager(output_dir, total_steps=525600)
        manager.register_column('P_offer', dtype=np.float64)
        manager.register_column('H2_kg', dtype=np.float64)
        
        for step in range(total_steps):
            manager.record(step, 'P_offer', value1)
            manager.record(step, 'H2_kg', value2)
        
        manager.finalize()
        df = manager.get_dataframe(columns=['P_offer', 'H2_kg'])
    """
    
    DEFAULT_CHUNK_SIZE = 10_000  # ~7 simulation days @ 1min/step
    
    def __init__(
        self,
        output_dir: Path,
        total_steps: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        simulation_id: Optional[str] = None
    ):
        """
        Initialize the chunked history manager.
        
        Args:
            output_dir: Directory for chunk files and checkpoints
            total_steps: Total expected simulation steps
            chunk_size: Steps per chunk (default 10,000)
            simulation_id: Unique ID for this simulation run
        """
        self.output_dir = Path(output_dir)
        self.chunks_dir = self.output_dir / "history_chunks"
        
        # Clean up old chunk files from previous runs to prevent stale data
        if self.chunks_dir.exists():
            old_chunks = list(self.chunks_dir.glob("chunk_*.parquet"))
            if old_chunks:
                logger.info(f"Cleaning {len(old_chunks)} old chunk files from previous run")
                for old_chunk in old_chunks:
                    try:
                        old_chunk.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete {old_chunk}: {e}")
        
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        self.total_steps = total_steps
        self.chunk_size = chunk_size
        self.simulation_id = simulation_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Column registry
        self._columns: Dict[str, np.dtype] = {}
        self._column_order: List[str] = []
        
        # Current chunk state
        self._current_chunk: Dict[str, np.ndarray] = {}
        self._chunk_index: int = 0
        self._step_in_chunk: int = 0
        self._total_steps_completed: int = 0
        
        # Chunk metadata
        self._chunk_metadata: List[ChunkMetadata] = []
        
        # Finalized flag
        self._finalized: bool = False
        
        logger.info(f"ChunkedHistoryManager initialized: {total_steps} steps, "
                   f"chunk_size={chunk_size}, ~{total_steps // chunk_size + 1} chunks")
    
    def register_column(self, name: str, dtype: np.dtype = np.float64) -> None:
        """
        Register a column for recording.
        
        Args:
            name: Column name
            dtype: NumPy dtype for the column
        """
        if name not in self._columns:
            self._columns[name] = dtype
            self._column_order.append(name)
            
            # Allocate in current chunk if already allocated
            if self._current_chunk:
                self._current_chunk[name] = np.zeros(self.chunk_size, dtype=dtype)
    
    def register_columns(self, columns: Dict[str, np.dtype]) -> None:
        """Register multiple columns at once."""
        for name, dtype in columns.items():
            self.register_column(name, dtype)
    
    def allocate_chunk(self) -> None:
        """Allocate memory for a new chunk."""
        self._current_chunk = {
            name: np.zeros(self.chunk_size, dtype=dtype)
            for name, dtype in self._columns.items()
        }
        self._step_in_chunk = 0
        
        chunk_memory_mb = sum(
            arr.nbytes for arr in self._current_chunk.values()
        ) / (1024 * 1024)
        logger.debug(f"Allocated chunk {self._chunk_index}: {chunk_memory_mb:.1f} MB, "
                    f"{len(self._columns)} columns")
    
    def record(self, step_idx: int, column: str, value: float) -> None:
        """
        Record a value at the given step. O(1) operation.
        
        Args:
            step_idx: Global step index
            column: Column name
            value: Value to record
        """
        if column not in self._current_chunk:
            # Auto-register column if not exists
            self.register_column(column)
            if column not in self._current_chunk:
                self._current_chunk[column] = np.zeros(self.chunk_size, dtype=np.float64)
        
        local_idx = step_idx % self.chunk_size
        self._current_chunk[column][local_idx] = value
    
    def step_complete(self, step_idx: int) -> None:
        """
        Mark a step as complete. Call this after recording all columns for a step.
        Triggers flush if chunk is full.
        
        Args:
            step_idx: Global step index that was just completed
        """
        self._step_in_chunk = (step_idx % self.chunk_size) + 1
        self._total_steps_completed = step_idx + 1
        
        # Check if chunk is full
        if self._step_in_chunk >= self.chunk_size:
            self.flush()
    
    def flush(self) -> Optional[Path]:
        """
        Flush current chunk to disk.
        
        Returns:
            Path to the written Parquet file, or None if chunk is empty
        """
        if self._step_in_chunk == 0:
            return None  # Empty chunk
        
        # Trim arrays to actual data
        chunk_data = {
            name: arr[:self._step_in_chunk].copy()
            for name, arr in self._current_chunk.items()
        }
        
        # Write to Parquet
        chunk_path = self.chunks_dir / f"chunk_{self._chunk_index:04d}.parquet"
        df = pd.DataFrame(chunk_data)
        df.to_parquet(chunk_path, compression='snappy', index=False)
        
        # Record metadata
        start_step = self._chunk_index * self.chunk_size
        end_step = start_step + self._step_in_chunk - 1
        
        metadata = ChunkMetadata(
            chunk_index=self._chunk_index,
            start_step=start_step,
            end_step=end_step,
            filepath=chunk_path,
            columns=list(self._columns.keys())
        )
        self._chunk_metadata.append(metadata)
        
        logger.info(f"Flushed chunk {self._chunk_index}: steps {start_step}-{end_step} "
                   f"({self._step_in_chunk} rows) -> {chunk_path.name}")
        
        # Save checkpoint
        self._save_checkpoint()
        
        # Advance to next chunk
        self._chunk_index += 1
        self.allocate_chunk()
        gc.collect()
        
        return chunk_path
    
    def _save_checkpoint(self) -> Path:
        """Save checkpoint state for crash recovery."""
        checkpoint = CheckpointState(
            simulation_id=self.simulation_id,
            total_steps_target=self.total_steps,
            total_steps_completed=self._total_steps_completed,
            chunk_index=self._chunk_index,
            step_in_chunk=self._step_in_chunk,
            columns=list(self._columns.keys()),
            chunks=[{
                'index': m.chunk_index,
                'start': m.start_step,
                'end': m.end_step,
                'file': str(m.filepath.name)
            } for m in self._chunk_metadata]
        )
        
        checkpoint_path = self.output_dir / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.__dict__, f, indent=2)
        
        return checkpoint_path
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> 'ChunkedHistoryManager':
        """
        Resume from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint.json
            
        Returns:
            ChunkedHistoryManager instance ready to resume
        """
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
        
        output_dir = checkpoint_path.parent
        manager = cls(
            output_dir=output_dir,
            total_steps=state['total_steps_target'],
            simulation_id=state['simulation_id']
        )
        
        # Restore column registry
        for col in state['columns']:
            manager.register_column(col)
        
        # Restore chunk metadata
        for chunk_info in state['chunks']:
            manager._chunk_metadata.append(ChunkMetadata(
                chunk_index=chunk_info['index'],
                start_step=chunk_info['start'],
                end_step=chunk_info['end'],
                filepath=manager.chunks_dir / chunk_info['file'],
                columns=state['columns']
            ))
        
        # Resume from last position
        manager._chunk_index = state['chunk_index']
        manager._total_steps_completed = state['total_steps_completed']
        manager.allocate_chunk()
        
        logger.info(f"Resumed from checkpoint: {state['total_steps_completed']} steps completed")
        
        return manager
    
    def finalize(self) -> Path:
        """
        Finalize the history - flush remaining data.
        
        Returns:
            Path to the chunks directory
        """
        if self._finalized:
            return self.chunks_dir
        
        # Flush any remaining data
        if self._step_in_chunk > 0:
            self.flush()
        
        self._finalized = True
        logger.info(f"History finalized: {len(self._chunk_metadata)} chunks, "
                   f"{self._total_steps_completed} total steps")
        
        return self.chunks_dir
    
    def get_dataframe(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load history as DataFrame with optional column filtering.
        
        This is the main API for graph generation. It lazy-loads only
        the requested columns from disk.
        
        Args:
            columns: List of columns to load (None = all)
            
        Returns:
            Concatenated DataFrame from all chunks
        """
        chunk_files = sorted(self.chunks_dir.glob("chunk_*.parquet"))
        
        if not chunk_files:
            logger.warning("No chunk files found")
            return pd.DataFrame()
        
        # Load with column filtering
        dfs = []
        for chunk_file in chunk_files:
            try:
                df = pd.read_parquet(chunk_file, columns=columns)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to read {chunk_file}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Loaded {len(result)} rows, {len(result.columns)} columns "
                   f"from {len(chunk_files)} chunks")
        
        return result
    
    def export_to_csv(self, output_path: Path) -> None:
        """
        Stream history to CSV file chunk by chunk to keep memory usage low.
        Replaces the need to load the full DataFrame into memory.
        
        Args:
            output_path: Path to the output CSV file
        """
        self.finalize()
        chunk_files = sorted(self.chunks_dir.glob("chunk_*.parquet"))
        
        if not chunk_files:
            logger.warning("No chunks to export")
            return

        logger.info(f"Streaming {len(chunk_files)} chunks to {output_path} (additive write)...")
        
        # Write first chunk with header
        first_chunk = True
        for chunk_file in chunk_files:
            try:
                df = pd.read_parquet(chunk_file)
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                
                df.to_csv(output_path, mode=mode, header=header, index=False)
                
                first_chunk = False
                del df
                gc.collect()
            except Exception as e:
                logger.error(f"Failed to export chunk {chunk_file}: {e}")
                raise
        
        logger.info(f"Streamed export complete: {output_path}")

    def get_column(self, column: str) -> np.ndarray:
        """
        Load a single column as NumPy array.
        
        Args:
            column: Column name
            
        Returns:
            NumPy array with all values
        """
        df = self.get_dataframe(columns=[column])
        return df[column].values if column in df.columns else np.array([])
    
    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            'simulation_id': self.simulation_id,
            'total_steps_completed': self._total_steps_completed,
            'total_steps_target': self.total_steps,
            'chunks_written': len(self._chunk_metadata),
            'columns': len(self._columns),
            'finalized': self._finalized,
            'chunk_size': self.chunk_size
        }


class HistoryArrayProxy:
    """
    Proxy that provides array-like access for recording while using chunked storage.
    
    This allows drop-in replacement of the old _history dict:
    
    Before: self._history['P_offer'][step_idx] = value
    After:  self._history['P_offer'][step_idx] = value  # Same syntax!
    
    The proxy intercepts __setitem__ and routes to the manager.
    """
    
    def __init__(self, manager: ChunkedHistoryManager, column: str):
        self._manager = manager
        self._column = column
    
    def __setitem__(self, step_idx: int, value: float) -> None:
        """Route assignment to manager."""
        self._manager.record(step_idx, self._column, value)
    
    def __getitem__(self, step_idx: int) -> float:
        """
        Get value from current chunk (limited to current chunk only).
        For historical data, use manager.get_column().
        """
        local_idx = step_idx % self._manager.chunk_size
        return self._manager._current_chunk.get(self._column, np.zeros(1))[local_idx]


class HistoryDictProxy:
    """
    Proxy that mimics dict[column][step] access pattern.
    
    This is a drop-in replacement for the _history dict:
    
    self._history = HistoryDictProxy(manager)
    self._history['P_offer'][step_idx] = value  # Works!
    """
    
    def __init__(self, manager: ChunkedHistoryManager):
        self._manager = manager
        self._column_proxies: Dict[str, HistoryArrayProxy] = {}
    
    def __getitem__(self, column: str) -> HistoryArrayProxy:
        """Get or create column proxy."""
        if column not in self._column_proxies:
            self._manager.register_column(column)
            self._column_proxies[column] = HistoryArrayProxy(self._manager, column)
        return self._column_proxies[column]
    
    def __setitem__(self, column: str, value: np.ndarray) -> None:
        """
        Support for direct array assignment (for pre-allocation compatibility).
        This registers the column but ignores the array value.
        """
        dtype = value.dtype if isinstance(value, np.ndarray) else np.float64
        self._manager.register_column(column, dtype)
    
    def __contains__(self, column: str) -> bool:
        return column in self._manager._columns
    
    def keys(self) -> List[str]:
        return list(self._manager._columns.keys())
    
    def items(self):
        """Compatibility method - returns column proxies."""
        return [(col, self[col]) for col in self._manager._columns.keys()]
