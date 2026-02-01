import gc
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import psutil

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Real-time memory monitoring with thresholds.
    
    Thresholds:
    - Normal: < 60% of max
    - Warning: 60-80% of max
    - Critical: > 80% of max
    """
    
    def __init__(self, max_memory_mb: int):
        self.max_memory_mb = max_memory_mb
        self._process = psutil.Process()
    
    def get_current_mb(self) -> float:
        """Get current process memory in MB."""
        return self._process.memory_info().rss / 1e6
    
    def get_pressure(self) -> float:
        """Get memory pressure as fraction (0.0 to 1.0+)."""
        if self.max_memory_mb <= 0:
            return 0.0
        return self.get_current_mb() / self.max_memory_mb
    
    def has_headroom(self, min_headroom_mb: float = 500.0) -> bool:
        """Check if there's at least min_headroom_mb available."""
        return (self.max_memory_mb - self.get_current_mb()) > min_headroom_mb
    
    def is_critical(self) -> bool:
        """Check if memory usage is in critical range."""
        return self.get_pressure() > 0.8
    
    def log_usage(self, context: str = ""):
        """Log current memory usage."""
        current = self.get_current_mb()
        pressure = self.get_pressure()
        level = "CRITICAL" if pressure > 0.8 else "WARNING" if pressure > 0.6 else "NORMAL"
        logger.info(f"[{level}] Memory: {current:.0f} MB / {self.max_memory_mb:.0f} MB ({pressure*100:.1f}%) {context}")


class StreamingDownsampler:
    """
    Memory-efficient chunk processing with configurable limits.
    
    Features:
    - Sequential chunk processing (never loads all at once)
    - Per-chunk downsampling before accumulation
    - Memory budget enforcement with automatic GC
    - Progress tracking with estimated memory usage
    - Fallback to aggressive downsampling if needed
    """
    
    def __init__(
        self,
        max_memory_mb: int = 2000, 
        target_resolution_minutes: int = 60,
        emergency_resolution_minutes: int = 360
    ):
        self.max_memory_mb = max_memory_mb
        self.target_resolution = target_resolution_minutes
        self.emergency_resolution = emergency_resolution_minutes
        self.accumulated_chunks: List[pd.DataFrame] = []
        self.current_memory_mb: float = 0.0
        self.monitor = MemoryMonitor(max_memory_mb)
        self.current_resolution = self.target_resolution
        self.is_emergency_mode = False
    
    def process_chunks_directory(
        self, 
        chunks_dir: Path,
        output_cache: Path,
        required_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Process all chunks with memory safety.
        """
        # Fix: Sort numerically by chunk index (chunk_1, chunk_2, ... chunk_10)
        # Using simple lambda that handles the filename format chunk_N.parquet
        try:
            chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"), 
                               key=lambda p: int(p.stem.split('_')[-1]))
        except Exception:
            # Fallback to lexical sort if naming convention doesn't match
            logger.warning("Could not sort chunks numerically, falling back to lexical sort")
            chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
        
        if not chunk_files:
            raise ValueError(f"No chunk files found in {chunks_dir}")
        
        logger.info(f"Processing {len(chunk_files)} chunks with memory budget: {self.max_memory_mb} MB")
        
        for i, chunk_file in enumerate(chunk_files):
            # Memory check before loading
            if self._check_memory_pressure():
                logger.warning(f"Memory pressure detected at chunk {i+1}/{len(chunk_files)}")
                self._emergency_consolidation()
            
            # Load and downsample chunk
            try:
                chunk_df = self._load_and_downsample_chunk(chunk_file, required_columns)
                
                # Accumulate
                self.accumulated_chunks.append(chunk_df)
                # Estimate size increase (approximate)
                self.current_memory_mb += chunk_df.memory_usage(deep=True).sum() / 1e6
                
                # Progress
                if (i + 1) % 10 == 0:
                    self.monitor.log_usage(f"after processing chunk {i+1}/{len(chunk_files)}")
                    
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_file}: {e}")
                continue
        
        # Final concatenation
        logger.info("Concatenating downsampled chunks...")
        if not self.accumulated_chunks:
             logger.warning("No data accumulated.")
             return pd.DataFrame()

        result_df = pd.concat(self.accumulated_chunks, ignore_index=True)
        
        # FIX: Ensure final result is sorted by time to prevent plotting issues
        if 'minute' in result_df.columns:
            result_df = result_df.sort_values('minute').reset_index(drop=True)
        
        # Clear accumulator
        self.accumulated_chunks.clear()
        gc.collect()
        
        # Save cache
        logger.info(f"Writing cache to {output_cache}...")
        # Ensure parent directory exists
        output_cache.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_cache, index=False, compression='snappy')
        
        return result_df
    
    def _load_and_downsample_chunk(self, chunk_file: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load single chunk and apply target downsampling.
        
        Args:
            chunk_file: Path to parquet chunk
            columns: Optional list of columns to load. If None, loads all.
        """
        if columns:
            # Ensure minute is always loaded if available (critical for time axis)
            if 'minute' not in columns:
                columns = columns + ['minute']
            
            try:
                # Fast path: try loading requested columns
                df = pd.read_parquet(chunk_file, columns=columns)
            except Exception:
                # Fallback: load all, then filter
                try:
                    import pyarrow.parquet as pq
                    schema = pq.read_schema(chunk_file)
                    valid_cols = [c for c in columns if c in schema.names]
                    df = pd.read_parquet(chunk_file, columns=valid_cols)
                    
                    # Add missing columns as NaN
                    for c in columns:
                        if c not in df.columns:
                            df[c] = float('nan')
                except ImportError:
                     df = pd.read_parquet(chunk_file)
                     df = df[[c for c in columns if c in df.columns]]
        else:
            df = pd.read_parquet(chunk_file)
        
        # Basic validation
        if df.empty:
            return df

        # FIX: Ensure DataFrame is sorted by time immediately after loading
        # This prevents negative diffs or weird stride calculations if rows are unsorted
        if 'minute' in df.columns:
            df = df.sort_values('minute').reset_index(drop=True)
            
        # Determine current resolution (seconds between rows)
        avg_step_minutes = 1.0
        if len(df) > 1 and 'minute' in df.columns:
            # We assume 'minute' column exists and is numeric, representing time in minutes
            try:
                # Sort just in case to get correct diff
                sorted_minutes = df['minute'].sort_values()
                start = sorted_minutes.iloc[0]
                end = sorted_minutes.iloc[-1]
                # logger.debug(f"Chunk minute range: {start} -> {end}")
                
                avg_step_minutes = sorted_minutes.diff().mean()
                if pd.isna(avg_step_minutes) or avg_step_minutes <= 0:
                    avg_step_minutes = 1.0 
            except Exception:
                pass
        
        # Calculate required stride
        # e.g. target 60 min, data 1 min -> stride 60
        stride = max(1, int(self.current_resolution / avg_step_minutes))
        
        if stride > 1:
            df = df.iloc[::stride].copy()
        
        return df
    
    def _check_memory_pressure(self) -> bool:
        """Check if approaching memory budget."""
        # Check both the calculated estimation and actual process memory
        # Being conservative: if real usage is > 80% OR our accumulator estimation is > 80%
        
        real_pressure = self.monitor.get_pressure()
        estimated_pressure = self.current_memory_mb / self.max_memory_mb if self.max_memory_mb > 0 else 0
        
        return real_pressure > 0.8 or estimated_pressure > 0.8
    
    def _emergency_consolidation(self):
        """
        Aggressive downsampling when approaching memory limit.
        """
        if self.is_emergency_mode:
            # Already in emergency mode, just force GC and maybe warn
            logger.warning("Memory pressure persists in emergency mode. Forcing GC.")
            gc.collect()
            return

        logger.warning(f"EMERGENCY: Switching to aggressive downsampling ({self.emergency_resolution} min)")
        self.is_emergency_mode = True
        self.current_resolution = self.emergency_resolution
        
        if not self.accumulated_chunks:
            return
        
        # Concatenate what we have
        temp_df = pd.concat(self.accumulated_chunks, ignore_index=True)
        
        # Apply consolidation to match NEW resolution
        # Old resolution was self.target_resolution
        # New is self.emergency_resolution
        # factor = new / old
        factor = max(1, int(self.emergency_resolution / self.target_resolution))
        
        if factor > 1:
            temp_df = temp_df.iloc[::factor].copy()
        
        # Replace accumulator with consolidated chunk
        self.accumulated_chunks = [temp_df]
        
        # Force cleanup
        gc.collect()
        
        self.current_memory_mb = temp_df.memory_usage(deep=True).sum() / 1e6
        logger.info(f"  Emergency consolidation complete: {len(temp_df)} rows, {self.current_memory_mb:.0f} MB")
