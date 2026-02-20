"""
UnifiedGraphExecutor: Central executor for all graph generation.

This module replaces the fragmented graph generation in:
- run_integrated_simulation.py (GRAPH_MAP loop)
- graph_orchestrator.py (YAML-driven handlers)

Features:
- YAML-driven enable/disable via GraphCatalog
- Column deduplication across all enabled graphs
- Priority-sorted execution with tqdm progress
- Per-graph timeout protection
- Library-specific export (PNG for Matplotlib, HTML for Plotly)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union
import logging
import fnmatch
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from h2_plant.visualization.streaming_downsampler import MemoryMonitor
from h2_plant.visualization.graph_catalog import GraphMetadata

logger = logging.getLogger(__name__)

# Optional imports with fallback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


class TimeoutException(Exception):
    """Raised when a graph generation times out."""
    pass


@contextmanager
def time_limit(seconds: int, graph_id: str):
    """
    Context manager for per-graph timeout protection.
    
    Uses SIGALRM on Unix systems; no-op on Windows.
    
    Args:
        seconds: Maximum execution time in seconds
        graph_id: Graph ID for error messages
        
    Yields:
        None
        
    Raises:
        TimeoutException: If execution exceeds time limit
    """
    if seconds <= 0:
        yield
        return
        
    def signal_handler(signum, frame):
        raise TimeoutException(f"Graph '{graph_id}' timed out after {seconds}s")
    
    # Only use SIGALRM on Unix
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows fallback - no timeout (threading approach is complex)
        yield


@dataclass
class GraphResult:
    """Result of a single graph generation."""
    graph_id: str
    status: str  # 'success', 'failed', 'timeout', 'skipped'
    path: Optional[Path] = None
    error: Optional[str] = None
    duration_ms: int = 0


def _infer_dt_seconds(df: pd.DataFrame) -> None:
    """Infer the row spacing and store it as df.attrs['row_dt_seconds'].

    'row_dt_seconds' is the time between consecutive rows in the cache (e.g. 3 600 s
    for an hourly-decimated cache).  It is intentionally stored under a DIFFERENT key
    from 'dt_seconds' because:

    - 'dt_seconds'     = original simulation timestep (60 s for 1-min simulation).
                         Used by all rate/efficiency calculations in plotly_graphs.py
                         (H2_kg / dt_h → kg/h).  Must stay at 60 s.
    - 'row_dt_seconds' = spacing between cache rows (3 600 s for hourly cache).
                         Used only for duration/energy calculations that treat each row
                         as representing a full interval (e.g. MW × row_dt → MWh).

    Mixing the two caused efficiency to read ~1.4 % instead of ~84-89 %.
    """
    if 'minute' not in df.columns or len(df) < 2:
        return
    diffs = np.diff(df['minute'].values)
    pos = diffs[diffs > 0]
    if not pos.size:
        return
    inferred = float(np.median(pos)) * 60.0
    df.attrs['row_dt_seconds'] = inferred
    horizon_yr = float(df['minute'].max()) / 60.0 / 8760.0
    logger.info(
        "Inferred row_dt_seconds=%.0fs (median minute diff=%.1f min, horizon≈%.1f yr)",
        inferred, float(np.median(pos)), horizon_yr,
    )


class UnifiedGraphExecutor:
    """
    Central executor for all graph generation.
    
    Replaces the legacy GRAPH_MAP loop and GraphOrchestrator with a single,
    optimized execution path that:
    
    1. Uses GraphCatalog as the sole source of truth for graph metadata
    2. Computes unique column requirements across all enabled graphs
    3. Loads a single optimized DataFrame (using ChunkedHistoryManager if available)
    4. Executes graphs in priority order with progress bar and timeout protection
    5. Exports to appropriate format based on library (PNG/HTML)
    
    Usage:
        from h2_plant.visualization.unified_executor import UnifiedGraphExecutor
        from h2_plant.visualization.graph_catalog import GRAPH_REGISTRY
        
        executor = UnifiedGraphExecutor(GRAPH_REGISTRY, output_dir)
        executor.configure_from_yaml(viz_config)
        df = executor.load_data(history=history_dict)
        results = executor.execute(df, timeout_seconds=60)
    """
    
    def __init__(self, catalog: 'GraphCatalog', output_dir: Union[str, Path]):
        """
        Initialize the executor.
        
        Args:
            catalog: GraphCatalog instance with registered graphs
            output_dir: Directory for graph output files
        """
        self.catalog = catalog
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._configured = False
        self._viz_config: Dict[str, Any] = {}
        
        # Initialize memory monitor with default 4GB limit (can be updated)
        self.memory_monitor = MemoryMonitor(max_memory_mb=4000)
        
        logger.info(f"UnifiedGraphExecutor initialized with output_dir={self.output_dir}")
    
    def configure_from_yaml(self, config: Dict[str, Any]) -> None:
        """
        Configure enabled graphs based on visualization_config.yaml.
        
        Respects:
        - visualization.categories.<category>: true/false
        - visualization.graphs.<graph_id>: true/false
        - visualization.skip_legacy_graphs: true/false
        
        Args:
            config: Parsed YAML configuration dict
        """
        self._viz_config = config
        viz = config.get('visualization', {})
        
        if not config:
            logger.info("Empty config provided to configure_from_yaml. Leaving defaults enabled.")
            return
        
        # First disable all, then selectively enable
        # self.catalog.disable_all()  <-- REMOVED: We want defaults to persist unless disabled

        
        # Enable by category
        categories_config = viz.get('categories', {})
        for category, enabled in categories_config.items():
            if enabled:
                self.catalog.enable_category(category)
        
        # Enable/disable individual graphs (overrides categories)
        graphs_config = viz.get('graphs', {})
        for graph_id, enabled in graphs_config.items():
            if enabled:
                self.catalog.enable(graph_id)
            else:
                self.catalog.disable(graph_id)
        
        # Handle skip_legacy_graphs
        if viz.get('skip_legacy_graphs', False):
            self.catalog.disable_category('legacy')
        
        # Handle orchestrated_graphs section (enable specific plots)
        self._register_orchestrated_graphs(viz.get('orchestrated_graphs', {}))
        
        # ----------------------------------------------------------------------
        # DUAL GENERATION LOGIC
        # Automatically enable/disable graphs based on library preferences
        # ----------------------------------------------------------------------
        from h2_plant.visualization.utils import get_library_preference
        
        # Snapshot of currently enabled graphs to iterate safely
        initial_enabled = list(self.catalog.get_enabled())
        
        for meta in initial_enabled:
            # We only process "primary" graphs (not the _plotly twins themselves)
            # to avoid infinite recursion or double handling.
            # Assuming primary graphs don't end in _plotly.
            if meta.graph_id.endswith('_plotly'):
                continue
                
            prefs = get_library_preference(meta.graph_id, meta.category)
            
            # Check for Plotly twin
            plotly_id = f"{meta.graph_id}_plotly"
            has_twin = plotly_id in self.catalog
            
            # 1. Handle Plotly preference
            if 'plotly' in prefs:
                if has_twin:
                    self.catalog.enable(plotly_id)
                elif meta.library.value == 'plotly':
                     # If the graph itself IS plotly (e.g. new orchestrated ones?), ensure enabled
                     pass
            
            # 2. Handle Matplotlib preference
            if 'matplotlib' not in prefs:
                # If matplotlib is EXPLICITLY excluded, disable this primary graph
                # (assuming primary is MPL)
                if meta.library.value == 'matplotlib':
                    self.catalog.disable(meta.graph_id)
        
        self._configured = True
        enabled = self.catalog.list_enabled()
        logger.info(f"Configured {len(enabled)} enabled graphs from YAML (Dual Mode Active)")

    def _infer_columns_for_graph_type(self, graph_type: str, components: List[str]) -> List[str]:
        """
        Infer required columns from graph type and component names.
        
        This avoids the ['history'] wildcard fallback that would load ALL columns.
        Instead, we derive specific patterns based on what the graph type needs.
        
        Args:
            graph_type: Type of graph (e.g., 'temperature_profile', 'thermal_load_breakdown')
            components: List of component names from YAML config
            
        Returns:
            List of column patterns to load
        """
        # Base patterns by graph type
        base_patterns: Dict[str, List[str]] = {
            # Profile graphs
            'temperature_profile': ['minute', '*_outlet_temp*'],
            'pressure_profile': ['minute', '*_outlet_pressure*'],
            'flow_profile': ['minute', '*_outlet_mass_flow*', '*_h2_*', '*_h2o_*'],
            'process_train_profile': ['minute', '*_outlet_temp*', '*_outlet_pressure*', '*_outlet_mass_flow*'],
            
            # Thermal graphs
            'thermal_load_breakdown': ['minute', '*_cooling*', '*_duty*', '*_thermal*', '*_q_transferred*', '*_u_value*', '*_effectiveness*'],
            'thermal_time_series': ['minute', '*_outlet_temp*', '*_cooling*', '*_duty*', '*_q_transferred*', '*_u_value*', '*_effectiveness*'],
            'central_cooling_performance': ['minute', '*cooling*', '*temperature*', '*duty*', '*_u_value*', '*_effectiveness*', '*_return_temp*'],
            'dry_cooler_performance': ['minute', '*DryCooler*', '*Drycooler*', '*_cooling*', '*_u_value*', '*_effectiveness*'],
            
            # Separation graphs
            'water_removal_bar': ['minute', '*_liquid_removed*', '*_water_removed*'],
            'crossover_impurities': ['minute', '*_o2_*', '*_h2_*', '*_impurity*'],
            
            # Economics graphs
            'dispatch_stack': ['minute', 'P_offer', 'P_soec*', 'P_pem', 'spot_price', 'P_sold'],
            'economics_time_series': ['minute', 'spot_price', 'ppa_*', 'P_*'],
            'economics_pie': ['minute', 'P_*', '*_power*'],
            'economics_scatter': ['minute', 'spot_price', 'P_offer', 'h2_kg'],
            'effective_ppa': ['minute', 'ppa_*', 'spot_price'],
            
            # Production graphs
            'production_time_series': ['minute', 'H2_*_kg', 'h2_*'],
            'production_stacked': ['minute', 'H2_*_kg', 'O2_*_kg', '*_h2o_*'],
            'production_cumulative': ['minute', 'H2_*_kg', 'cumulative_*'],
            
            # Performance graphs
            'performance_time_series': ['minute', '*_voltage*', '*_efficiency*', '*_power*'],
            'performance_scatter': ['minute', 'spot_price', 'P_*', '*_power*'],
            
            # SOEC graphs
            'soec_modules_time_series': ['minute', 'soec_*', 'P_soec*'],
            'soec_heatmap': ['minute', 'soec_module_*', 'soec_active*'],
            'soec_stats': ['minute', 'soec_module_*'],
            
            # Storage graphs
            'storage_levels': ['minute', '*Tank*', '*_level*', '*_pressure*', '*inventory*'],
            'compressor_power': ['minute', '*Compressor*_power*', 'compressor_*'],
            'storage_apc': ['minute', 'storage_*', '*_soc*', '*_zone*'],
            'storage_inventory': ['minute', '*inventory*', '*Tank*', '*_kg*'],
            'storage_pressure_heatmap': ['minute', '*Tank*pressure*', '*_bar'],
            'water_tank_inventory': ['minute', '*UltraPure*', '*mass_kg*', '*control_zone*'],
            
            # P2 FIX: Missing graph types (were causing empty DataFrames)
            'rfnbo_compliance': ['minute', 'h2_rfnbo_kg', 'h2_non_rfnbo_kg', 'purchase_threshold_eur_mwh', 'P_grid_renewable*'],
            'deoxo_profile': ['minute', '*Deoxo*', '*_inlet_temp*', '*_outlet_temp*', '*_o2_in*'],
            'drain_properties': ['minute', '*Drain*', '*_drain_temp*', '*_drain_pressure*', '*_liquid_removed*'],
            'energy_flow': ['minute', '*_power_kw*', '*_duty*', 'compressor_power_kw'],
        }
        
        patterns = base_patterns.get(graph_type, ['minute']).copy()
        
        # Add component-specific patterns
        for comp in components:
            if comp:  # Skip empty strings
                patterns.append(f'*{comp}*')
        
        return patterns
    
    def _register_orchestrated_graphs(self, orchestrated_config: Dict[str, Any]) -> None:
        """
        Dynamically register orchestrated graphs from YAML config.
        
        Args:
            orchestrated_config: 'orchestrated_graphs' section of YAML
        """
        try:
            from h2_plant.visualization.graphs.modular_handlers import (
                MODULAR_HANDLERS, create_modular_wrapper
            )
            from h2_plant.visualization.graph_catalog import (
                GraphMetadata, GraphPriority, GraphLibrary
            )
        except ImportError:
            logger.warning("Modular graph handlers not available. Skipping orchestrated graphs.")
            return

        count = 0
        for graph_type, settings in orchestrated_config.items():
            # Skip if disabled or boolean
            if isinstance(settings, bool):
                continue
            if not settings.get('enabled', False):
                continue
                
            handler = MODULAR_HANDLERS.get(graph_type)
            if not handler:
                continue
                
            plots = settings.get('plots', [])
            for i, plot_config in enumerate(plots):
                title = plot_config.get('title', f"{graph_type}_{i}")
                components = plot_config.get('components', [])
                
                # Create unique ID for this specific plot instance
                safe_title = "".join([c if c.isalnum() else "_" for c in title]).lower()
                instance_id = f"orch_{graph_type}_{safe_title}"
                
                # Create wrapper function
                wrapper = create_modular_wrapper(handler, components, title, plot_config)
                
                # Infer required columns from graph type and components
                # This avoids the ['history'] wildcard that loads ALL columns
                inferred_columns = self._infer_columns_for_graph_type(graph_type, components)
                
                meta = GraphMetadata(
                    graph_id=instance_id,
                    title=title,
                    description=f"Orchestrated {graph_type}: {title}",
                    function=wrapper,
                    library=GraphLibrary.MATPLOTLIB,
                    data_required=inferred_columns,
                    priority=GraphPriority.HIGH,
                    category='orchestrated',
                    enabled=True
                )
                
                # Register and enable
                self.catalog.register(meta)
                # Note: register() doesn't auto-enable unless enabled=True acts on it?
                # GraphCatalog.register sets enabled based on metadata.enabled, 
                # so it should be added to _enabled_graphs.
                
                count += 1
        
        if count > 0:
            logger.info(f"Registered {count} orchestrated graph instances")
    
    def get_required_columns(self) -> Set[str]:
        """
        Compute union of data_required across all enabled graphs.
        
        Returns:
            Set of unique column names/patterns needed for all enabled graphs.
            Always includes 'minute' as the base time column.
        """
        from h2_plant.visualization.graph_catalog import CORE_COLUMNS
        
        required: Set[str] = {'minute'}  # Always required
        history_fallback_graphs = []
        
        for meta in self.catalog.get_enabled():
            if meta.data_required:
                # Handle ['history'] fallback gracefully - use CORE_COLUMNS instead of wildcard
                if 'history' in meta.data_required:
                    history_fallback_graphs.append(meta.graph_id)
                    # Add core columns as fallback instead of returning {'*'}
                    required.update(CORE_COLUMNS)
                    continue
                    
                for col in meta.data_required:
                    required.add(col)
        
        if history_fallback_graphs:
            logger.warning(
                f"{len(history_fallback_graphs)} graphs use ['history'] fallback. "
                f"Consider declaring specific columns. Graphs: {history_fallback_graphs[:5]}..."
            )
        
        return required
    
    def _expand_patterns(self, patterns: Set[str], all_columns: List[str]) -> Set[str]:
        """
        Expand glob patterns against actual column names.
        
        Args:
            patterns: Set of column names and/or glob patterns (e.g., '*_outlet_*')
            all_columns: List of actual column names in the DataFrame
            
        Returns:
            Set of expanded column names that match patterns
        """
        expanded: Set[str] = set()
        
        for pattern in patterns:
            if '*' in pattern or '?' in pattern:
                # Glob pattern - match against all columns
                matched = [col for col in all_columns if fnmatch.fnmatch(col, pattern)]
                expanded.update(matched)
            else:
                # Exact column name
                if pattern in all_columns:
                    expanded.add(pattern)
        
        return expanded
    
    def load_data(
        self,
        history: Optional[Dict[str, np.ndarray]] = None,
        chunks_dir: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        downsample_factor: int = 60,
        cache_path: Optional[Path] = None,
        resample_freq: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load DataFrame with only required columns.
        
        Tries sources in order:
        1. Precomputed cache (if cache_path provided and exists)
        2. In-memory history dict (if provided)
        3. Chunked Parquet files (if chunks_dir exists)
        4. CSV file (if csv_path provided)
        
        Args:
            history: In-memory history dict from simulation
            chunks_dir: Path to history_chunks/ directory with Parquet files
            csv_path: Path to simulation_history.csv
            downsample_factor: Take every Nth row to reduce memory usage.
                Default: 60 (converts 1-minute data to hourly).
                Set to 1 for full resolution.
            cache_path: Path to precomputed downsampled parquet file.
            resample_freq: Pandas frequency string (e.g. '1D', '1H') for STREAMING aggregation.
            
        Returns:
            pd.DataFrame with required columns for enabled graphs
        """
        required_patterns = self.get_required_columns()
        
        # Log downsampling info
        if downsample_factor > 1:
            logger.info(f"Downsampling enabled: taking every {downsample_factor}th row (e.g., 1-min → hourly)")
        
        # Try precomputed cache first
        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading from downsampled cache: {cache_path}")
            try:
                # Resolve patterns against cache file schema
                import pyarrow.parquet as pq
                import gc
                pf = pq.ParquetFile(cache_path)
                schema = pf.schema_arrow  # pyarrow.Schema — same as pq.read_schema()
                all_columns = schema.names

                columns_to_load = list(self._expand_patterns(required_patterns, all_columns))
                logger.info(f"Filtered to {len(columns_to_load)} required columns from cache")

                if downsample_factor > 1:
                    # Stream row-group by row-group to avoid loading ALL rows before striding.
                    # Peak memory = one_row_group × cols × 8B instead of full_file × cols × 8B.
                    strided_chunks = []
                    stride_pos = 0  # cumulative row count across groups, used for offset alignment
                    for rg_idx in range(pf.metadata.num_row_groups):
                        rg_table = pf.read_row_group(rg_idx, columns=columns_to_load)
                        rg_df = rg_table.to_pandas()
                        del rg_table
                        # Compute which row within this group is the first stride-aligned row
                        # so that selected rows are exactly downsample_factor apart globally.
                        offset = (-stride_pos) % downsample_factor
                        strided_chunks.append(rg_df.iloc[offset::downsample_factor].copy())
                        stride_pos += len(rg_df)
                        del rg_df
                    df = pd.concat(strided_chunks, ignore_index=True)
                    del strided_chunks
                    gc.collect()
                    logger.info(
                        f"Streamed cache (stride={downsample_factor}x): "
                        f"{len(df)} rows x {len(df.columns)} cols"
                    )
                else:
                    # use_threads=False: sequential column decompression avoids parallel
                    # decompression buffer spikes (each column's tmp buffer freed before next).
                    # self_destruct=True: PyArrow frees each Arrow column buffer as it is
                    # converted to a numpy array, keeping peak at ~1× final size instead of
                    # ~2× (Arrow table + Pandas DataFrame overlapping in memory).
                    arrow_table = pf.read(columns=columns_to_load, use_threads=False)
                    df = arrow_table.to_pandas(self_destruct=True)
                    del arrow_table
                    gc.collect()
                    logger.info(f"Loaded cache: {len(df)} rows x {len(df.columns)} columns")

                # Normalize loaded data
                from h2_plant.visualization.static_graphs import normalize_history
                df = normalize_history(df, inplace=True)
                _infer_dt_seconds(df)
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Falling back to other sources.")
        
        
        # Try in-memory history first
        if history is not None:
            logger.info("Loading from in-memory history dict")
            all_columns = list(history.keys())
            columns_to_load = self._expand_patterns(required_patterns, all_columns)
            
            # Create DataFrame with only required columns, preserving matrix data
            from h2_plant.visualization.static_graphs import normalize_history
            
            data = {}
            matrix_attrs = {}
            
            for col in columns_to_load:
                if col in history:
                    val = history[col]
                    
                    # Store multi-dimensional data in matrix_attrs for heatmaps
                    if hasattr(val, 'ndim') and val.ndim > 1:
                        # Downsample matrix data along first axis
                        if downsample_factor > 1:
                            matrix_attrs[col] = val[::downsample_factor]
                        else:
                            matrix_attrs[col] = val
                        continue
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (list, tuple)):
                        matrix_attrs[col] = val
                        continue
                    
                    # Downsample 1D arrays
                    if downsample_factor > 1 and hasattr(val, '__getitem__'):
                        data[col] = val[::downsample_factor]
                    else:
                        data[col] = val
            
            # Use normalize_history to generate aliases (P_soec from P_soec_actual etc)
            # This is critical for legacy graphs that expect specific normalized names
            df = normalize_history(data)
            
            # Re-attach matrix data to dataframe attributes
            for k, v in matrix_attrs.items():
                df.attrs[k] = v

            _infer_dt_seconds(df)
            logger.info(f"Loaded in-memory history: {len(df)} rows (downsampled by {downsample_factor}x)")
            return df
        
        # Try chunked Parquet files
        if chunks_dir is not None:
            chunks_path = Path(chunks_dir)
            if chunks_path.exists():
                try:
                    chunk_files = sorted(chunks_path.glob('chunk_*.parquet'), 
                                       key=lambda p: int(p.stem.split('_')[-1]))
                except Exception:
                    chunk_files = sorted(chunks_path.glob('chunk_*.parquet'))
            else:
                chunk_files = []
            if chunk_files:
                logger.info(f"Loading from {len(chunk_files)} chunked Parquet files in {chunks_path}")
                try:
                    import gc
                    
                    # Read schema from first chunk to get available columns
                    try:
                        import pyarrow.parquet as pq
                        schema = pq.read_schema(chunk_files[0])
                        all_columns = schema.names
                    except ImportError:
                        # Fallback: read first row to get columns (slower but works)
                        logger.warning("pyarrow not available for schema inspection, using fallback")
                        sample_df = pd.read_parquet(chunk_files[0], nrows=1)
                        all_columns = list(sample_df.columns)
                        del sample_df
                    
                    # Expand patterns against actual columns
                    if '*' in required_patterns:
                        # Wildcard fallback - load all columns (but log warning)
                        logger.warning("Loading ALL columns due to '*' in required_patterns - consider declaring specific columns")
                        columns_to_load = None
                    else:
                        columns_to_load = list(self._expand_patterns(required_patterns, all_columns))
                        if not columns_to_load:
                            logger.warning("No columns matched patterns, falling back to all columns")
                            columns_to_load = None
                        else:
                            logger.info(f"Column filtering: loading {len(columns_to_load)}/{len(all_columns)} columns")
                    
                    # Stream chunks with column filtering, DOWNSAMPLING, and periodic GC
                    dfs = []
                    total_rows_original = 0
                    total_rows_downsampled = 0
                    
                    for i, chunk_file in enumerate(chunk_files):
                        try:
                            df_chunk = pd.read_parquet(chunk_file, columns=columns_to_load)
                        except Exception as e:
                            # Handle schema drift: some columns in columns_to_load might be missing in this chunk
                            logger.debug(f"Chunk {chunk_file.name} missing some columns: {e}. Attempting fallback.")
                            
                            # Get actual columns in this chunk
                            try:
                                import pyarrow.parquet as pq
                                chunk_schema = pq.read_schema(chunk_file)
                                valid_cols = [c for c in columns_to_load if c in chunk_schema.names]
                            except ImportError:
                                # Fallback if pyarrow not direct importable
                                valid_cols = list(pd.read_parquet(chunk_file).columns) # expensive but rare fallback
                                valid_cols = [c for c in columns_to_load if c in valid_cols]
                            
                            if not valid_cols:
                                continue # Skip if no relevant data
                                
                            df_chunk = pd.read_parquet(chunk_file, columns=valid_cols)
                            
                            
                            # Fill missing columns with NaN to ensure concatenation works
                            missing_cols = set(columns_to_load) - set(valid_cols)
                            for missing in missing_cols:
                                df_chunk[missing] = np.nan
                        
                        
                        # =========================================================
                        # STREAMING RESAMPLING (Aggregative)
                        # =========================================================
                        if resample_freq:
                            # 1. Convert to DatetimeIndex
                            if 'minute' in df_chunk.columns:
                                # Optimization: Vectorized timestamp creation
                                start_date = pd.Timestamp("2025-01-01")
                                timestamps = start_date + pd.to_timedelta(df_chunk['minute'], unit='m')
                                df_chunk.set_index(timestamps, inplace=True)
                                
                                # 2. Get Aggregation Rules (Once per load usually, but cheap)
                                agg_rules = self._get_aggregation_rules(df_chunk.columns)
                                
                                # 3. Resample Chunk
                                try:
                                    # Resample this individual chunk
                                    # Note: This might create partial intervals at boundaries.
                                    # We will fix this by grouping the final result.
                                    df_chunk = df_chunk.resample(resample_freq).agg(agg_rules)
                                    
                                    # Don't reset index yet, we need it for alignment
                                except Exception as e:
                                    logger.warning(f"Chunk resampling failed: {e}")
                                    
                            else:
                                logger.warning("No 'minute' column for resampling. Skipping.")
                        
                        # =========================================================
                        # SIMPLE DECIMATION (Legacy)
                        # =========================================================
                        elif downsample_factor > 1:
                            df_chunk = df_chunk.iloc[::downsample_factor]
                        
                        
                        total_rows_original += len(df_chunk) # This stat is a bit mixed now, but ok

                        
                        total_rows_downsampled += len(df_chunk)
                        dfs.append(df_chunk)
                        
                        # Periodic garbage collection to keep memory bounded
                        if (i + 1) % 4 == 0:
                            gc.collect()
                            logger.debug(f"Loaded {i + 1}/{len(chunk_files)} chunks, GC triggered")
                    
                    if dfs:
                        combined_df = pd.concat(dfs, ignore_index=False) # Keep index if resampling
                        del dfs
                        gc.collect()
                        
                        # Finalize Streaming Resampling
                        if resample_freq:
                             # Merge split intervals (e.g. Day 1 part A + Day 1 Part B)
                             # Group by Index (Datetime) and aggregate again
                             agg_rules = self._get_aggregation_rules(combined_df.columns)
                             combined_df = combined_df.groupby(level=0).agg(agg_rules)
                             
                             # Restore minute column
                             if not combined_df.empty:
                                start_time = combined_df.index[0]
                                combined_df['minute'] = (combined_df.index - start_time).total_seconds() / 60.0
                                
                             logger.info(f"Streaming Resampling complete. Final shape: {combined_df.shape}")
                        else:
                             # Reset index if it wasn't resampled (concat with ignore_index=False above kept it integer or whatever)
                             # Actually previous code had ignore_index=True for simple concat.
                             # If we didn't resample, index is RangeIndex likely from read_parquet usually range(0, N)?
                             # No, read_parquet chunks usually have RangeIndex 0..N.
                             # concat with ignore_index=False preserves 0..N, 0..N.
                             # We should reset index to be safe/clean 0..TotalN
                             combined_df.reset_index(drop=True, inplace=True)

                        
                        # Normalize loaded data (create aliases like P_soec)
                        from h2_plant.visualization.static_graphs import normalize_history
                        combined_df = normalize_history(combined_df)
                        _infer_dt_seconds(combined_df)

                        if downsample_factor > 1:
                            logger.info(
                                f"Loaded DataFrame: {total_rows_downsampled} rows "
                                f"(downsampled from {total_rows_original}, {downsample_factor}x reduction) "
                                f"x {len(combined_df.columns)} columns"
                            )
                        else:
                            logger.info(f"Loaded DataFrame: {len(combined_df)} rows x {len(combined_df.columns)} columns")

                        return combined_df
                except Exception as e:
                    logger.warning(f"Failed to load from chunks: {e}. Falling back to CSV.")
        
        # Try CSV file
        if csv_path is not None and csv_path.exists():
            logger.info(f"Loading from CSV: {csv_path}")
            
            # Read header first to get column names
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
            
            columns_to_load = list(self._expand_patterns(required_patterns, header))
            
            # Load CSV with optional downsampling via skiprows
            if downsample_factor > 1:
                # skiprows with lambda: skip rows where (row_num - 1) % factor != 0
                # Row 0 is header, so we keep it. Then keep row 1, skip 2-60, keep 61, etc.
                df_csv = pd.read_csv(
                    csv_path, 
                    usecols=columns_to_load,
                    skiprows=lambda x: x > 0 and (x - 1) % downsample_factor != 0
                )
                logger.info(f"Loaded CSV with {downsample_factor}x downsampling: {len(df_csv)} rows")
            else:
                df_csv = pd.read_csv(csv_path, usecols=columns_to_load)
            
            # Normalize loaded data (create aliases like P_soec)
            from h2_plant.visualization.static_graphs import normalize_history
            return normalize_history(df_csv)
        
        # Nothing to load
        logger.error("No data source available")
        return pd.DataFrame()

    def precompute_downsampled_cache(
        self,
        chunks_dir: Path,
        output_path: Path,
        downsample_factor: int = 60
    ) -> pd.DataFrame:
        """
        Scan all chunks, downsample them via STREAMING, and create a single master Parquet file.
        
        This method uses pyarrow to iterate over batches of rows, downsampling each batch
        IMMEDIATELY before memory accumulation. This ensures that even with 500MB+ chunks,
        we never hold more than a few MB in memory at once.
        
        Args:
            chunks_dir: Input Directory containing chunk_*.parquet files
            output_path: Output path for the aggregated parquet file
            downsample_factor: Factor to reduce rows by (default 60: 1 min -> 1 hour)
            
        Returns:
            The loaded aggregated DataFrame
        """
        import gc
        import pyarrow.parquet as pq
        
        try:
            chunk_files = sorted(chunks_dir.glob('chunk_*.parquet'), 
                               key=lambda p: int(p.stem.split('_')[-1]))
        except Exception:
            chunk_files = sorted(chunks_dir.glob('chunk_*.parquet'))
        if not chunk_files:
            logger.warning("No chunks found for precomputation")
            return pd.DataFrame()
            
        logger.info(f"PRE-COMPUTING: Streaming {len(chunk_files)} chunks with {downsample_factor}x downsampling...")
        
        downsampled_frames = []
        total_rows_original = 0
        total_rows_final = 0
        
        # Track global row index to ensure consistent downsampling across file boundaries
        current_global_row = 0
        
        for i, cf in enumerate(tqdm(chunk_files, desc="Streaming chunks")):
            try:
                # Open Parquet file without reading content
                parquet_file = pq.ParquetFile(cf)
                
                # Iterate over batches (e.g. default batch_size=65536 rows)
                for batch in parquet_file.iter_batches():
                    n_rows = batch.num_rows
                    total_rows_original += n_rows
                    
                    # Convert to pandas only for this small batch
                    df_batch = batch.to_pandas()
                    
                    # Calculate indices to keep: (global_index % factor == 0)
                    # We start at current_global_row
                    start_idx = current_global_row
                    end_idx = start_idx + n_rows
                    
                    # Vectorized boolean mask for downsampling
                    # kept_indices = (np.arange(start_idx, end_idx) % downsample_factor == 0)
                    # However, simple slicing is faster if we align it right.
                    
                    # Calculate offset for first row to keep in this batch
                    # First row index to keep >= start_idx is: start_idx + (factor - start_idx % factor) % factor
                    remainder = start_idx % downsample_factor
                    offset = (downsample_factor - remainder) % downsample_factor
                    
                    # Slice locally
                    df_small = df_batch.iloc[offset::downsample_factor]
                    
                    if not df_small.empty:
                        downsampled_frames.append(df_small)
                        total_rows_final += len(df_small)
                    
                    current_global_row += n_rows
                    
                    # Explicit cleanup per batch
                    del df_batch
                
                # Periodic GC
                if (i + 1) % 5 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Failed to process chunk {cf.name}: {e}")
        
        if not downsampled_frames:
            return pd.DataFrame()
            
        # Concatenate
        logger.info("Concatenating stream results...")
        full_df = pd.concat(downsampled_frames, ignore_index=True)
        del downsampled_frames
        gc.collect()
        
        # Save cache
        logger.info(f"Saving precomputed cache to {output_path}...")
        full_df.to_parquet(output_path, index=False)
        
        logger.info(
            f"Done. Cache built: {total_rows_final} rows "
            f"(was {total_rows_original}, {downsample_factor}x reduction). "
            f"Size: {full_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )
        
        # Normalize immediately so it's ready for use
        from h2_plant.visualization.static_graphs import normalize_history
        return normalize_history(full_df)
    
    def _is_figure_empty(self, fig) -> bool:
        """
        Check if matplotlib figure has any content.
        
        Args:
            fig: Matplotlib Figure object
            
        Returns:
            True if figure has no axes or no artists (lines, patches, etc.)
        """
        if not fig.axes:
            return True
        
        for ax in fig.axes:
            # Check for standard plotting elements
            if ax.lines or ax.collections or ax.patches or ax.images:
                return False
            # Also check for containers (e.g. bar plots sometimes use them)
            if hasattr(ax, 'containers') and ax.containers:
                return False
                
        return True
    
        return True
    
    def execute_batched(
        self,
        df: pd.DataFrame,
        timeout_seconds: int = 60,
        batch_size: int = 10
    ) -> Dict[str, GraphResult]:
        """
        Execute graphs in batches with memory monitoring.
        """
        import time
        import gc
        
        enabled_graphs = self.catalog.get_enabled()
        results = {}
        
        # Divide into batches
        batches = [
            enabled_graphs[i:i+batch_size] 
            for i in range(0, len(enabled_graphs), batch_size)
        ]
        
        logger.info(f"Executing {len(enabled_graphs)} graphs in {len(batches)} batches")
        
        for batch_num, batch in enumerate(batches, 1):
            # Pre-batch memory check
            if not self.memory_monitor.has_headroom():
                logger.error(f"Insufficient memory for batch {batch_num}. Skipping remaining graphs.")
                for meta in batch:
                    results[meta.graph_id] = GraphResult(
                        graph_id=meta.graph_id,
                        status='skipped',
                        error='Insufficient memory'
                    )
                # Skip remaining batches
                break
            
            logger.info(f"Batch {batch_num}/{len(batches)}: {len(batch)} graphs")
            
            # Execute batch
            # We reuse the existing execute method, passing specific target_graphs
            batch_results = self.execute(
                df, 
                timeout_seconds=timeout_seconds,
                target_graphs=batch
            )
            results.update(batch_results)
            
            # Post-batch cleanup
            self.memory_monitor.log_usage(f"After batch {batch_num}")
            plt.close('all')
            gc.collect()
            
            # Check pressure
            if self.memory_monitor.get_pressure() > 0.8:
                logger.warning("Critical memory pressure detected.")
        
        return results

    def execute_sequentially_by_category(
        self,
        history: Optional[Dict[str, np.ndarray]] = None,
        chunks_dir: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        downsample_factor: int = 60,
        timeout_seconds: int = 60,
        cache_path: Optional[Path] = None,
        cache_stride: int = 1,
        df_attrs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, GraphResult]:
        """
        Execute graphs sequentially by category to minimize RAM usage.
        
        Instead of loading all data at once, this method:
        1. Groups enabled graphs by category
        2. For each category:
           - Identifies required columns
           - Loads data ONLY for those columns
           - Executes graphs
           - Clears data from memory
           
        Args:
            Same as load_data + execute
            
        Returns:
            Combined results dictionary
        """
        import gc
        
        all_enabled = self.catalog.get_enabled()
        if not all_enabled:
            logger.warning("No graphs enabled for execution.")
            return {}
            
        # Group by category
        by_category: Dict[str, List] = {}
        for meta in all_enabled:
            cat = meta.category or 'uncategorized'
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(meta)
            
        logger.info(f"Starting SEQUENTIAL execution: {len(all_enabled)} graphs in {len(by_category)} categories")
        
        all_results = {}
        
        
        # Pre-fetch available columns and row count to correctly estimate batch sizes
        all_columns = []
        cache_row_count: Optional[int] = None
        if cache_path is not None and Path(cache_path).exists():
             try:
                 import pyarrow.parquet as pq
                 pf = pq.ParquetFile(cache_path)
                 all_columns = pf.schema_arrow.names
                 cache_row_count = pf.metadata.num_rows
                 if cache_stride > 1:
                     cache_row_count = max(1, cache_row_count // cache_stride)
             except Exception as e:
                 logger.warning(f"Could not read cache schema for column counting: {e}")
        elif chunks_dir is not None:
             chunks_path = Path(chunks_dir)
             chunk_files = sorted(chunks_path.glob('chunk_*.parquet'))
             if chunk_files:
                 try:
                     import pyarrow.parquet as pq
                     # Read schema from last chunk as it's most likely to have all columns
                     # (or first, but let's trust schema drift handling in load_data)
                     schema = pq.read_schema(chunk_files[-1])
                     all_columns = schema.names
                 except Exception as e:
                     logger.warning(f"Could not read schema for column counting: {e}")
        elif csv_path is not None and csv_path.exists():
             with open(csv_path, 'r') as f:
                 all_columns = f.readline().strip().split(',')

        # Compute memory-aware MAX_COLS for sub-batch sizing.
        # Formula: available_mb * 0.85 / (rows * 8 bytes/float * overhead 3) → column count.
        # Falls back to hard cap of 1000 when row count is unknown.
        try:
            import psutil as _psutil
            _avail_mb = _psutil.virtual_memory().available / 1e6
        except Exception:
            _avail_mb = 4000.0
        if cache_row_count and cache_row_count > 0:
            _projected_max_cols = max(50, int(_avail_mb * 0.85 / (cache_row_count * 8 * 3 / 1e6)))
            MAX_COLS = min(1000, _projected_max_cols)
            logger.info(
                "Sub-batch MAX_COLS=%d (projected from %d rows, %.0f MB available)",
                MAX_COLS, cache_row_count, _avail_mb,
            )
        else:
            MAX_COLS = 1000

        # Process each category
        for cat, graphs_in_cat in by_category.items():
            # Sub-batching to avoid massive column unions (like 'orchestrated' with 7000+ columns)
            batches = []
            current_batch = []
            current_cols_patterns = {'minute'}
            current_cols_expanded = set(['minute'])
            
            from h2_plant.visualization.graph_catalog import CORE_COLUMNS
            
            for meta in graphs_in_cat:
                # Determine pattern columns for this graph
                g_patterns = set()
                if meta.data_required:
                    if 'history' in meta.data_required:
                        g_patterns.update(CORE_COLUMNS)
                    else:
                        g_patterns.update(meta.data_required)
                
                # Expand to actual columns if possible
                if all_columns:
                    g_expanded = self._expand_patterns(g_patterns, all_columns)
                else:
                    g_expanded = g_patterns # Fallback if no schema access
                
                # Check if adding this graph exceeds the memory-projected column limit
                potential_expanded = current_cols_expanded | g_expanded
                
                if current_batch and len(potential_expanded) > MAX_COLS:
                    # Flush current batch
                    batches.append((current_batch, current_cols_patterns))
                    # Start new batch
                    current_batch = [meta]
                    current_cols_patterns = {'minute'} | g_patterns
                    current_cols_expanded = {'minute'} | g_expanded
                else:
                    current_batch.append(meta)
                    current_cols_patterns = current_cols_patterns | g_patterns
                    current_cols_expanded = potential_expanded
            
            if current_batch:
                batches.append((current_batch, current_cols_patterns))
            
            logger.info(">>> Starting category: %s (%d graphs, %d batches)", cat, len(graphs_in_cat), len(batches))
            
            for i, (graphs, required) in enumerate(batches):
                logger.debug(f"   Batch {i+1}/{len(batches)}: {len(graphs)} graphs, ~{len(required)} columns")
                
                # Monkey-patch get_required_columns to return ONLY this batch's requirements
                original_method = self.get_required_columns
                self.get_required_columns = lambda: required
                
                try:
                    # Load data for this batch
                    df = self.load_data(
                        history=history,
                        chunks_dir=chunks_dir,
                        csv_path=csv_path,
                        downsample_factor=cache_stride if cache_path is not None else downsample_factor,
                        cache_path=cache_path
                    )
                    
                    if df.empty:
                        logger.warning(f"No data loaded for batch {i+1} in {cat}")
                        continue

                    if df_attrs:
                        if 'config' in df_attrs and isinstance(df_attrs['config'], dict):
                            existing_cfg = df.attrs.get('config', {})
                            merged_cfg = dict(existing_cfg)
                            merged_cfg.update(df_attrs['config'])
                            df.attrs['config'] = merged_cfg
                        for attr_key, attr_value in df_attrs.items():
                            if attr_key == 'config':
                                continue
                            df.attrs[attr_key] = attr_value
                        
                    # Execute graphs in this batch; split once on MemoryError and retry
                    def _run_batch(target):
                        return self.execute(
                            df,
                            timeout_seconds=timeout_seconds,
                            target_graphs=target,
                        )
                    try:
                        results = _run_batch(graphs)
                    except MemoryError:
                        gc.collect()
                        mid = len(graphs) // 2
                        if mid < 1:
                            raise
                        logger.warning(
                            "MemoryError on batch %d/%d in %s; splitting into two halves and retrying",
                            i + 1, len(batches), cat,
                        )
                        try:
                            results = _run_batch(graphs[:mid])
                            results.update(_run_batch(graphs[mid:]))
                        except MemoryError:
                            logger.error(
                                "MemoryError persists after batch split in %s; skipping batch %d",
                                cat, i + 1,
                            )
                            results = {}
                    all_results.update(results)

                except Exception as e:
                    logger.error(f"Failed to execute batch {i+1} in {cat}: {e}", exc_info=True)
                finally:
                    # Restore original method
                    self.get_required_columns = original_method

                    # Force cleanup
                    if 'df' in locals():
                        del df
                    gc.collect()

            logger.info("<<< Completed category: %s", cat)

        logger.info(f"Sequential execution complete. Total results: {len(all_results)}")
        return all_results


    def _get_aggregation_rules(self, columns: List[str]) -> Dict[str, str]:
        """
        Define aggregation rules for columns based on naming conventions.
        
        Rules:
        - Mass/Energy totals (kg, MWh): SUM
        - Rates (kg/h, MW) and State (Price, Pressure, Temp, SOC): MEAN
        - Everything else: MEAN
        """
        agg_dict = {}
        for col in columns:
            c = col.lower()
            if any(x in c for x in ['_kg', '_mwh', 'energy', 'production', 'consumption']):
                # Differentiate between totals (sum) and rates (mean)
                if '_kg_h' in c or '_kg_per_h' in c or '_mw' in c or '_kw' in c:
                    # Note: Power (MW) often appears in 'energy' contexts, check explicit unit
                    agg_dict[col] = 'mean' 
                elif 'consumption' in c or 'production' in c or 'energy' in c:
                    # Totals
                    agg_dict[col] = 'sum'
                elif '_kg' in c:
                    # Mass totals vs rates
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'sum'
            elif any(x in c for x in ['price', 'pressure', 'temp', 'volt', 'power', '_mw', '_kw', 'eff', 'soc']):
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'mean' # Default
        return agg_dict

    def resample_data(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample DataFrame to a new time frequency with correct aggregation.
        
        Aggregation Rules:
        - Mass/Energy (kg, MWh): SUM
        - Rates/Intensive (MW, Price, Pressure, Temp): MEAN
        - Stock/Levels (Inventory): LAST or MEAN
        
        Args:
            df: Input DataFrame
            freq: Pandas frequency string (e.g., '1H', '1D')
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
            
        logger.info(f"Resampling data to frequency: {freq}")
        df_res = df.copy()
        
        # Ensure DatetimeIndex
        if not isinstance(df_res.index, pd.DatetimeIndex):
            if 'minute' in df_res.columns:
                start_date = pd.Timestamp("2025-01-01")
                timestamps = start_date + pd.to_timedelta(df_res['minute'], unit='m')
                df_res.set_index(timestamps, inplace=True)
            else:
                logger.warning("No 'minute' column or DatetimeIndex found for resampling. Skipping.")
                return df

        # Define aggregation logic
        agg_dict = self._get_aggregation_rules(df_res.columns)
        
        try:
            resampled = df_res.resample(freq).agg(agg_dict)
            # Restore minute column for X-axis
            start_time = resampled.index[0]
            resampled['minute'] = (resampled.index - start_time).total_seconds() / 60.0
            logger.info(f"Resampling complete. Shape: {df.shape} -> {resampled.shape}")
            return resampled
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return df

    def _add_metadata_stamp(self, fig, df: pd.DataFrame, sim_name: str = "Unknown Simulation", meta_info: str = ""):
        """
        Add standard metadata footer to the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from h2_plant.visualization import utils

        # Only for Matplotlib figures
        if not isinstance(fig, Figure):
            return

        dt_h = utils.get_dt_hours(df)
        dt_str = f"dt={dt_h*60:.1f}min"
        
        # Stamp text
        stamp = f"{sim_name} | {dt_str} | Generated by H2Plant OS"
        if meta_info:
            stamp += f" | {meta_info}"
            
        # Add text to bottom right
        fig.text(0.99, 0.01, stamp, ha='right', va='bottom', 
                 fontsize=6, color='gray', alpha=0.7)

    def execute(
        self,
        df: pd.DataFrame,
        timeout_seconds: int = 60,
        dpi: int = 100,
        target_graphs: Optional[List['GraphMetadata']] = None,
        resample_freq: Optional[str] = None
    ) -> Dict[str, GraphResult]:
        """
        Execute enabled graphs in priority order.
        
        Args:
            df: DataFrame with simulation history
            timeout_seconds: Maximum time per graph (0=no timeout)
            dpi: Resolution for Matplotlib figures
            target_graphs: Optional list of specific graphs to execute. 
                           If None, executes all enabled graphs from catalog.
            resample_freq: Optional frequency string (e.g., '1D') to resample data.
            
        Returns:
            Dict mapping graph_id to GraphResult
        """
        import time
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Apply Resampling if requested
        if resample_freq:
            df = self.resample_data(df, resample_freq)
        
        # PERFORMANCE FIX: Strip heavy attributes (large matrices) from DataFrame
        # to prevent expensive deepcopies during column access in graph functions.
        # We preserve the simulation name for stamping.
        sim_name = df.attrs.get('config', {}).get('simulation_name', 'Unknown Simulation')
        
        # Create lightweight shallow copy for graph functions
        df_light = df.copy(deep=False)
        
        # P0 FIX: Preserve essential config keys instead of stripping all attrs.
        # This prevents LCOH, Arbitrage, and other config-dependent graphs from failing.
        df_light.attrs = {
            'config': df.attrs.get('config', {}),
            'dt_seconds': df.attrs.get('dt_seconds', 60.0),
            'metrics': df.attrs.get('metrics', {}),
            'start_date': df.attrs.get('start_date', None),
            'scenario_name': df.attrs.get('scenario_name', 'Scenario'),
        }
        
        results: Dict[str, GraphResult] = {}
        
        # Use provided target_graphs or fall back to all enabled
        enabled_graphs = target_graphs if target_graphs is not None else self.catalog.get_enabled()
        
        logger.info(f"Executing {len(enabled_graphs)} graphs...")
        
        # Use tqdm if available
        iterator = tqdm(enabled_graphs, desc="Generating graphs") if TQDM_AVAILABLE else enabled_graphs
        
        for graph_num, meta in enumerate(iterator, start=1):
            graph_id = meta.graph_id
            start_time = time.time()
            fig = None
            
            try:
                with time_limit(timeout_seconds, graph_id):
                    # Call the graph function with lightweight DataFrame
                    fig = meta.function(df_light, dpi=dpi)
                    
                    if fig is None:
                        results[graph_id] = GraphResult(
                            graph_id=graph_id,
                            status='skipped',
                            error='Function returned None'
                        )
                        continue
                    
                    # Check for empty figure (Matplotlib only)
                    if meta.library.value == 'matplotlib' and self._is_figure_empty(fig):
                        # Use INFO level for expected empty figures (e.g. no drains active)
                        logger.info(f"Graph '{graph_id}' skipped: produced an empty figure (no data plotted).")
                        results[graph_id] = GraphResult(
                            graph_id=graph_id,
                            status='skipped',
                            error='Empty figure (no data plotted)'
                        )
                        plt.close(fig)
                        fig = None
                        continue
                        
                    # Add Metadata Stamp (pass sim_name explicitly)
                    # Pass the ORIGINAL dataframe length/info if resampled, to show full context?
                    # Or just show current context. Current implementation of stamp uses df attributes.
                    self._add_metadata_stamp(fig, df_light, sim_name=sim_name)

                    # Determine output path and format based on library
                    if meta.library.value == 'matplotlib':
                        filename = f"{meta.title.replace(' ', '_').replace('/', '_')}.png"
                        output_path = self.output_dir / filename
                        
                        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
                        plt.close(fig)
                        fig = None
                        
                    elif meta.library.value == 'plotly':
                        filename = f"{meta.title.replace(' ', '_').replace('/', '_')}.html"
                        output_path = self.output_dir / filename
                        # PERFORMANCE: Use CDN for plotly.js (reduces file size from ~5MB to ~500KB)
                        fig.write_html(
                            str(output_path),
                            include_plotlyjs='cdn',
                            full_html=True,
                            config={'displayModeBar': True, 'responsive': True, 'editable': True}
                        )
                        
                    else:
                        # Seaborn or unknown - treat as matplotlib
                        filename = f"{meta.title.replace(' ', '_').replace('/', '_')}.png"
                        output_path = self.output_dir / filename
                        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                        plt.close(fig)
                        fig = None
                    
                    duration_ms = int((time.time() - start_time) * 1000)
                    results[graph_id] = GraphResult(
                        graph_id=graph_id,
                        status='success',
                        path=output_path,
                        duration_ms=duration_ms
                    )
                    
            except TimeoutException as e:
                logger.warning(f"Graph '{graph_id}' timed out")
                results[graph_id] = GraphResult(
                    graph_id=graph_id,
                    status='timeout',
                    error=str(e),
                    duration_ms=timeout_seconds * 1000
                )
                # Clean up any open figures
                plt.close('all')
                
            except Exception as e:
                logger.error(f"Graph '{graph_id}' failed: {e}")
                results[graph_id] = GraphResult(
                    graph_id=graph_id,
                    status='failed',
                    error=str(e),
                    duration_ms=int((time.time() - start_time) * 1000)
                )
                plt.close('all')
            finally:
                # Explicitly drop figure references to release memory pressure
                if fig is not None:
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                    fig = None

                pressure = self.memory_monitor.get_pressure()
                if pressure > 0.8:
                    self.memory_monitor.log_usage(f"during graph {graph_id}")
                if graph_num % 5 == 0 or pressure > 0.8:
                    gc.collect()
        
        # Summary
        del df_light
        gc.collect()
        success = sum(1 for r in results.values() if r.status == 'success')
        failed = sum(1 for r in results.values() if r.status == 'failed')
        timeout = sum(1 for r in results.values() if r.status == 'timeout')
        
        logger.info(f"Graph generation complete: {success} success, {failed} failed, {timeout} timeout")
        
        return results
    
    def summary(self) -> Dict[str, Any]:
        """
        Return summary of executor state.
        
        Returns:
            Dict with configuration and catalog summary
        """
        return {
            'output_dir': str(self.output_dir),
            'configured': self._configured,
            'catalog': self.catalog.summary(),
            'required_columns': len(self.get_required_columns())
        }
