"""
GraphGenerator: Master orchestrator for visualization generation.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from h2_plant.visualization.metrics_collector import MetricsCollector
from h2_plant.visualization.graph_catalog import GraphCatalog, GraphMetadata, GRAPH_REGISTRY

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

logger = logging.getLogger(__name__)


class GraphGenerator:
    """
    Master orchestrator for graph generation.
    
    Responsibilities:
    - Generate individual graphs from collected metrics
    - Create multi-graph dashboards
    - Export graphs to various formats (HTML, PNG, PDF, WebP)
    - Manage graph enable/disable configuration
    - Parallel generation for performance
    """
    
    def __init__(self, metrics_collector: MetricsCollector, catalog: Optional[GraphCatalog] = None):
        """
        Initialize the graph generator.
        
        .. deprecated::
            GraphGenerator uses MetricsCollector which is not integrated with 
            the main simulation loop. Use GraphOrchestrator instead with the 
            history DataFrame from `run_integrated_simulation.py`.
        
        Args:
            metrics_collector: MetricsCollector instance with simulation data
            catalog: Optional GraphCatalog (uses global registry if None)
        """
        import warnings
        warnings.warn(
            "GraphGenerator is deprecated. Use GraphOrchestrator with "
            "visualization_config.yaml and the history DataFrame instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.collector = metrics_collector
        self.catalog = catalog or GRAPH_REGISTRY
        self._generated_graphs: Dict[str, Any] = {}
        
        # Performance settings
        self.use_webgl = True
        self.max_workers = 4
        
        logger.info(f"GraphGenerator initialized with {len(self.catalog.list_enabled())} enabled graphs")
    
    def generate(self, graph_id: str, **kwargs) -> Optional[Any]:
        """
        Generate a specific graph by ID.
        
        Args:
            graph_id: ID of the graph to generate
            **kwargs: Additional arguments to pass to the graph function
        
        Returns:
            Generated figure object (Plotly/Matplotlib) or None if disabled/failed
        """
        metadata = self.catalog.get(graph_id)
        
        if metadata is None:
            logger.error(f"Unknown graph ID: {graph_id}")
            return None
        
        if not metadata.enabled:
            logger.info(f"Graph '{graph_id}' is disabled, skipping")
            return None
        
        # Check data requirements
        missing_data = self._check_data_requirements(metadata.data_required)
        if missing_data:
            logger.warning(f"Cannot generate '{graph_id}': missing data {missing_data}")
            return None
        
        try:
            # Prepare data for the graph function
            data = self._prepare_data(metadata.data_required)
            
            # Merge kwargs with metadata kwargs
            plot_kwargs = {**metadata.kwargs, **kwargs}
            
            # Inject WebGL setting if applicable
            if self.use_webgl and len(data.get('timestamps', [])) > 10000:
                plot_kwargs['use_webgl'] = True
            
            # Call the graph function
            logger.info(f"Generating graph: {graph_id}")
            fig = metadata.function(data, **plot_kwargs)
            
            # Cache the generated graph
            self._generated_graphs[graph_id] = fig
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to generate graph '{graph_id}': {e}", exc_info=True)
            return None
    
    def generate_all_enabled(self, parallel: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Generate all enabled graphs.
        
        Args:
            parallel: Whether to use parallel execution
            **kwargs: Additional arguments to pass to all graph functions
        
        Returns:
            Dictionary mapping graph_id to figure objects
        """
        enabled_graphs = self.catalog.get_enabled()
        results = {}
        
        logger.info(f"Generating {len(enabled_graphs)} enabled graphs (parallel={parallel})...")
        
        if parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_id = {
                    executor.submit(self.generate, m.graph_id, **kwargs): m.graph_id 
                    for m in enabled_graphs
                }
                
                for future in as_completed(future_to_id):
                    graph_id = future_to_id[future]
                    try:
                        fig = future.result()
                        if fig is not None:
                            results[graph_id] = fig
                    except Exception as e:
                        logger.error(f"Error generating {graph_id}: {e}")
        else:
            for metadata in enabled_graphs:
                fig = self.generate(metadata.graph_id, **kwargs)
                if fig is not None:
                    results[metadata.graph_id] = fig
        
        logger.info(f"Successfully generated {len(results)}/{len(enabled_graphs)} graphs")
        return results
    
    def generate_category(self, category: str, **kwargs) -> Dict[str, Any]:
        """
        Generate all enabled graphs in a specific category.
        
        Args:
            category: Category name
            **kwargs: Additional arguments to pass to graph functions
        
        Returns:
            Dictionary mapping graph_id to figure objects
        """
        category_graphs = [m for m in self.catalog.get_by_category(category) if m.enabled]
        results = {}
        
        logger.info(f"Generating {len(category_graphs)} graphs in category '{category}'...")
        
        for metadata in category_graphs:
            fig = self.generate(metadata.graph_id, **kwargs)
            if fig is not None:
                results[metadata.graph_id] = fig
        
        return results
    
    def export(self, graph_id: str, output_path: Union[str, Path], 
               format: str = 'html', **kwargs) -> bool:
        """
        Export a generated graph to file.
        
        Args:
            graph_id: ID of the graph to export
            output_path: Path to save the file
            format: Export format ('html', 'png', 'pdf', 'svg', 'webp')
            **kwargs: Additional export options
        
        Returns:
            True if export succeeded, False otherwise
        """
        if graph_id not in self._generated_graphs:
            logger.warning(f"Graph '{graph_id}' not yet generated, generating now...")
            fig = self.generate(graph_id)
            if fig is None:
                return False
        else:
            fig = self._generated_graphs[graph_id]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'html':
                # PERFORMANCE: Use CDN for plotly.js (reduces file size)
                fig.write_html(str(output_path), include_plotlyjs='cdn', **kwargs)
            elif format in ['png', 'pdf', 'svg', 'webp']:
                # Requires kaleido
                fig.write_image(str(output_path), format=format, **kwargs)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported '{graph_id}' to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export '{graph_id}': {e}")
            return False
    
    def export_all(self, output_dir: Union[str, Path], format: str = 'html', **kwargs) -> int:
        """
        Export all generated graphs to a directory.
        
        Args:
            output_dir: Directory to save graphs
            format: Export format
            **kwargs: Additional export options
        
        Returns:
            Number of successfully exported graphs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for graph_id in self._generated_graphs:
            filename = f"{graph_id}.{format}"
            if self.export(graph_id, output_dir / filename, format=format, **kwargs):
                count += 1
        
        logger.info(f"Exported {count} graphs to {output_dir}")
        return count
    
    def create_dashboard(self, graph_ids: Optional[List[str]] = None, 
                        title: str = "H2 Plant Dashboard") -> Optional[Any]:
        """
        Create a multi-graph dashboard (placeholder for Dash integration).
        
        Args:
            graph_ids: List of graph IDs to include (None = all enabled)
            title: Dashboard title
        
        Returns:
            Dashboard object or HTML string
        """
        if graph_ids is None:
            graph_ids = self.catalog.list_enabled()
        
        # For now, create a simple HTML page with all graphs
        html_parts = [f"<html><head><title>{title}</title></head><body>"]
        html_parts.append(f"<h1>{title}</h1>")
        
        for graph_id in graph_ids:
            if graph_id not in self._generated_graphs:
                self.generate(graph_id)
            
            if graph_id in self._generated_graphs:
                fig = self._generated_graphs[graph_id]
                if hasattr(fig, 'to_html'):
                    html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        html_parts.append("</body></html>")
        
        return "\n".join(html_parts)
    
    def _check_data_requirements(self, requirements: List[str]) -> List[str]:
        """
        Check if all required data is available.
        
        Args:
            requirements: List of data requirements (e.g., 'pem.h2_production_kg_h')
        
        Returns:
            List of missing data items
        """
        missing = []
        
        for req in requirements:
            if '.' in req:
                category, field = req.split('.', 1)
                if category not in self.collector.timeseries:
                    missing.append(req)
                elif field not in self.collector.timeseries[category]:
                    missing.append(req)
                elif not self.collector.timeseries[category][field]:
                    missing.append(req)
            else:
                # Top-level requirement (e.g., 'timestamps')
                if req not in self.collector.timeseries:
                    missing.append(req)
                elif not self.collector.timeseries[req]:
                    missing.append(req)
        
        return missing
    
    def _prepare_data(self, requirements: List[str]) -> Dict[str, Any]:
        """
        Prepare data dictionary for graph functions.
        
        Args:
            requirements: List of data requirements
        
        Returns:
            Dictionary with nested structure matching requirements
        """
        data = {}
        
        # Always include timestamps
        data['timestamps'] = self.collector.timeseries['timestamps']
        
        # Organize by category
        for req in requirements:
            if '.' in req:
                category, field = req.split('.', 1)
                if category not in data:
                    data[category] = {}
                data[category][field] = self.collector.timeseries[category][field]
            elif req != 'timestamps':  # Don't duplicate timestamps
                data[req] = self.collector.timeseries[req]
        
        return data
    
    def summary(self) -> Dict[str, Any]:
        """Return summary of graph generator status."""
        return {
            'catalog_summary': self.catalog.summary(),
            'generated_graphs': len(self._generated_graphs),
            'cached_graphs': list(self._generated_graphs.keys()),
            'data_points': self.collector.summary()
        }
