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
        
        self._configured = True
        enabled = self.catalog.list_enabled()
        logger.info(f"Configured {len(enabled)} enabled graphs from YAML")

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
            'thermal_load_breakdown': ['minute', '*_cooling*', '*_duty*', '*_thermal*', '*_q_transferred*'],
            'thermal_time_series': ['minute', '*_outlet_temp*', '*_cooling*', '*_duty*', '*_q_transferred*'],
            'central_cooling_performance': ['minute', '*cooling*', '*temperature*', '*duty*'],
            
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
        downsample_factor: int = 60
    ) -> pd.DataFrame:
        """
        Load DataFrame with only required columns.
        
        Tries sources in order:
        1. In-memory history dict (if provided)
        2. Chunked Parquet files (if chunks_dir exists)
        3. CSV file (if csv_path provided)
        
        Args:
            history: In-memory history dict from simulation
            chunks_dir: Path to history_chunks/ directory with Parquet files
            csv_path: Path to simulation_history.csv
            downsample_factor: Take every Nth row to reduce memory usage.
                Default: 60 (converts 1-minute data to hourly).
                Set to 1 for full resolution.
            
        Returns:
            pd.DataFrame with required columns for enabled graphs
        """
        required_patterns = self.get_required_columns()
        
        # Log downsampling info
        if downsample_factor > 1:
            logger.info(f"Downsampling enabled: taking every {downsample_factor}th row (e.g., 1-min → hourly)")
        
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
            
            logger.info(f"Loaded in-memory history: {len(df)} rows (downsampled by {downsample_factor}x)")
            return df
        
        # Try chunked Parquet files
        if chunks_dir is not None:
            chunks_path = Path(chunks_dir)
            chunk_files = sorted(chunks_path.glob('chunk_*.parquet')) if chunks_path.exists() else []
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
                        df_chunk = pd.read_parquet(chunk_file, columns=columns_to_load)
                        total_rows_original += len(df_chunk)
                        
                        # Apply downsampling to each chunk
                        if downsample_factor > 1:
                            df_chunk = df_chunk.iloc[::downsample_factor]
                        
                        total_rows_downsampled += len(df_chunk)
                        dfs.append(df_chunk)
                        
                        # Periodic garbage collection to keep memory bounded
                        if (i + 1) % 4 == 0:
                            gc.collect()
                            logger.debug(f"Loaded {i + 1}/{len(chunk_files)} chunks, GC triggered")
                    
                    if dfs:
                        combined_df = pd.concat(dfs, ignore_index=True)
                        del dfs
                        gc.collect()
                        
                        # Normalize loaded data (create aliases like P_soec)
                        from h2_plant.visualization.static_graphs import normalize_history
                        
                        if downsample_factor > 1:
                            logger.info(
                                f"Loaded DataFrame: {total_rows_downsampled} rows "
                                f"(downsampled from {total_rows_original}, {downsample_factor}x reduction) "
                                f"x {len(combined_df.columns)} columns"
                            )
                        else:
                            logger.info(f"Loaded DataFrame: {len(combined_df)} rows x {len(combined_df.columns)} columns")
                        
                        return normalize_history(combined_df)
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
        dpi: int = 100
    ) -> Dict[str, GraphResult]:
        """
        Execute all enabled graphs in priority order.
        
        Args:
            df: DataFrame with simulation history
            timeout_seconds: Maximum time per graph (0=no timeout)
            dpi: Resolution for Matplotlib figures
            
        Returns:
            Dict mapping graph_id to GraphResult
        """
        import time
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # PERFORMANCE FIX: Strip heavy attributes (large matrices) from DataFrame
        # to prevent expensive deepcopies during column access in graph functions.
        # We preserve the simulation name for stamping.
        sim_name = df.attrs.get('config', {}).get('simulation_name', 'Unknown Simulation')
        
        # Create lightweight shallow copy for graph functions
        df_light = df.copy(deep=False)
        df_light.attrs = {}
        
        results: Dict[str, GraphResult] = {}
        enabled_graphs = self.catalog.get_enabled()  # Already sorted by priority
        
        logger.info(f"Executing {len(enabled_graphs)} graphs...")
        
        # Use tqdm if available
        iterator = tqdm(enabled_graphs, desc="Generating graphs") if TQDM_AVAILABLE else enabled_graphs
        
        for meta in iterator:
            graph_id = meta.graph_id
            start_time = time.time()
            
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
                        continue
                        
                    # Add Metadata Stamp (pass sim_name explicitly)
                    self._add_metadata_stamp(fig, df_light, sim_name=sim_name)

                    # Determine output path and format based on library
                    if meta.library.value == 'matplotlib':
                        filename = f"{meta.title.replace(' ', '_').replace('/', '_')}.png"
                        output_path = self.output_dir / filename
                        
                        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
                        plt.close(fig)
                        
                    elif meta.library.value == 'plotly':
                        filename = f"{meta.title.replace(' ', '_').replace('/', '_')}.html"
                        output_path = self.output_dir / filename
                        fig.write_html(str(output_path))
                        
                    else:
                        # Seaborn or unknown - treat as matplotlib
                        filename = f"{meta.title.replace(' ', '_').replace('/', '_')}.png"
                        output_path = self.output_dir / filename
                        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
                        plt.close('all')
                    
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
        
        # Summary
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
