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
                
                # Create metadata
                # Note: 'data_required' should ideally be derived from components,
                # but for now we rely on the auto-resolution or pass ['history'] 
                # to ensure all data is loaded if we can't be specific.
                # Modular graphs usually need a wide range of columns.
                meta = GraphMetadata(
                    graph_id=instance_id,
                    title=title,
                    description=f"Orchestrated {graph_type}: {title}",
                    function=wrapper,
                    library=GraphLibrary.MATPLOTLIB,
                    data_required=['history'],  # Fallback to full load for safety
                    priority=GraphPriority.HIGH, # Default for these custom plots
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
        required: Set[str] = {'minute'}  # Always required
        
        for meta in self.catalog.get_enabled():
            if meta.data_required:
                # If any graph needs full history (placeholder), verify all columns are loaded
                if 'history' in meta.data_required:
                    return {'*'}
                    
                for col in meta.data_required:
                    required.add(col)
        
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
        csv_path: Optional[Path] = None
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
            
        Returns:
            pd.DataFrame with required columns for enabled graphs
        """
        required_patterns = self.get_required_columns()
        
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
                        matrix_attrs[col] = val
                        continue
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (list, tuple)):
                        matrix_attrs[col] = val
                        continue
                        
                    data[col] = val
            
            # Use normalize_history to generate aliases (P_soec from P_soec_actual etc)
            # This is critical for legacy graphs that expect specific normalized names
            df = normalize_history(data)
            
            # Re-attach matrix data to dataframe attributes
            for k, v in matrix_attrs.items():
                df.attrs[k] = v
                
            return df
        
        # Try chunked Parquet files
        if chunks_dir is not None:
            chunks_path = Path(chunks_dir)
            if chunks_path.exists() and list(chunks_path.glob('*.parquet')):
                logger.info(f"Loading from chunked Parquet files in {chunks_path}")
                try:
                    from h2_plant.storage.history_manager import ChunkedHistoryManager
                    
                    # Use lazy-loading with column filtering
                    manager = ChunkedHistoryManager.from_chunks(chunks_path)
                    all_columns = manager.get_columns()
                    columns_to_load = list(self._expand_patterns(required_patterns, all_columns))
                    
                    df_chunked = manager.get_dataframe(columns=columns_to_load)
                    
                    # Normalize loaded data (create aliases like P_soec)
                    from h2_plant.visualization.static_graphs import normalize_history
                    return normalize_history(df_chunked)
                except ImportError:
                    logger.warning("ChunkedHistoryManager not available, falling back to CSV")
        
        # Try CSV file
        if csv_path is not None and csv_path.exists():
            logger.info(f"Loading from CSV: {csv_path}")
            
            # Read header first to get column names
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
            
            columns_to_load = list(self._expand_patterns(required_patterns, header))
            
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
    
    def _add_metadata_stamp(self, fig, df: pd.DataFrame, meta_info: str = ""):
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
        
        # Try to get simulation name
        sim_name = df.attrs.get('config', {}).get('simulation_name', 'Unknown Simulation')
        
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
                    # Call the graph function
                    fig = meta.function(df, dpi=dpi)
                    
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
                        
                    # Add Metadata Stamp
                    self._add_metadata_stamp(fig, df)

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
