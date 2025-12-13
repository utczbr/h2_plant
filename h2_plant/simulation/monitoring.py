"""
Monitoring and Metrics Collection System.

This module provides the central monitoring infrastructure for collecting,
aggregating, and exporting simulation metrics and KPIs.

Metrics Categories:
    - **Component metrics**: Per-component state tracking over time.
    - **System aggregates**: Total production, storage, demand.
    - **Time-series data**: Hourly snapshots for trend analysis.
    - **Flow tracking**: Mass and energy flow between components.
    - **KPIs**: Efficiency, cost, fulfillment rate.

Export Formats:
    - JSON: Dashboard data, summary statistics, raw metrics.
    - CSV: Time-series data for spreadsheet analysis.
    - Sankey: Flow visualization data.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import csv
import json
import numpy as np

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.flow_tracker import FlowTracker

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
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super(NpEncoder, self).default(obj)


class MonitoringSystem:
    """
    Central metrics collection and aggregation system.

    Collects component states at each timestep, aggregates system-level
    metrics, tracks flows, and exports data in multiple formats.

    Attributes:
        output_dir (Path): Base output directory.
        metrics_dir (Path): Subdirectory for metrics files.
        flow_tracker (FlowTracker): Flow tracking subsystem.
        timeseries (Dict): Time-series data arrays.
        component_metrics (Dict): Per-component metric history.
        total_production_kg (float): Cumulative H₂ production.
        total_demand_kg (float): Cumulative H₂ demand.
        total_cost (float): Cumulative operational cost.

    Example:
        >>> monitoring = MonitoringSystem(output_dir)
        >>> monitoring.initialize(registry)
        >>> for hour in simulation:
        ...     monitoring.collect(hour, registry)
        >>> monitoring.export_dashboard_data()
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the monitoring system.

        Args:
            output_dir (Path): Base directory for output files.
        """
        self.output_dir = output_dir
        self.metrics_dir = output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.flow_tracker = FlowTracker()

        self.timeseries: Dict[str, List[Any]] = {
            'hour': [],
            'total_production_kg': [],
            'total_storage_kg': [],
            'total_demand_kg': [],
            'energy_price_mwh': [],
            'total_cost': []
        }

        self.component_metrics: Dict[str, Dict[str, List]] = {}

        self.total_production_kg = 0.0
        self.total_demand_kg = 0.0
        self.total_energy_kwh = 0.0
        self.total_cost = 0.0

    def initialize(self, registry: ComponentRegistry) -> None:
        """
        Initialize monitoring for all registered components.

        Creates metric storage for each component in the registry.

        Args:
            registry (ComponentRegistry): Registry of system components.
        """
        for component_id in registry.get_all_ids():
            self.component_metrics[component_id] = {}
        logger.info("Monitoring system initialized")

    def collect(self, hour: int, registry: ComponentRegistry) -> None:
        """
        Collect metrics and flows for the current timestep.

        Aggregates system-level metrics (production, storage, demand)
        and records per-component state history.

        Args:
            hour (int): Current simulation hour.
            registry (ComponentRegistry): Component registry for state access.
        """
        self.flow_tracker.set_current_hour(hour)
        states = registry.get_all_states()

        self.timeseries['hour'].append(hour)

        # Filter None states
        valid_states = []
        for cid, s in states.items():
            if s is None:
                logger.warning(f"Component {cid} returned None state")
            else:
                valid_states.append(s)

        # Aggregate production
        total_production = sum(s.get('h2_output_kg', 0.0) for s in valid_states)
        self.timeseries['total_production_kg'].append(total_production)
        self.total_production_kg += total_production

        # Aggregate storage
        total_storage = sum(
            s.get('total_mass_kg', s.get('mass_kg', 0.0))
            for c, s in states.items()
            if ('storage' in c or 'tank' in c) and s is not None
        )
        self.timeseries['total_storage_kg'].append(total_storage)

        # Demand tracking
        if registry.has('demand_scheduler'):
            demand_state = registry.get('demand_scheduler').get_state()
            current_demand = demand_state.get('current_demand_kg', 0.0)
            self.timeseries['total_demand_kg'].append(current_demand)
            self.total_demand_kg += current_demand
        else:
            self.timeseries['total_demand_kg'].append(0.0)

        # Energy price
        if registry.has('energy_price_tracker'):
            price_state = registry.get('energy_price_tracker').get_state()
            self.timeseries['energy_price_mwh'].append(price_state.get('current_price_per_mwh', 0.0))
        else:
            self.timeseries['energy_price_mwh'].append(0.0)

        # Cost tracking
        total_cost = 0.0
        if registry.has('dual_path_coordinator'):
            coordinator = registry.get('dual_path_coordinator')
            total_cost = coordinator.get_state().get('cumulative_net_cost_eur', 0.0)
        elif registry.has('coordinator'):
            total_cost = registry.get('coordinator').get_state().get('cumulative_cost', 0.0)
        else:
            for state in valid_states:
                total_cost = max(total_cost, state.get('cumulative_cost', 0.0))
        self.timeseries['total_cost'].append(total_cost)
        self.total_cost = total_cost

        # Record flows and component-specific metrics
        for comp_id, state in states.items():
            if state is None:
                continue
            if 'flows' in state and isinstance(state['flows'], dict):
                for direction, flow_group in state['flows'].items():
                    if not isinstance(flow_group, dict):
                        continue
                    for flow_name, flow_data in flow_group.items():
                        if not isinstance(flow_data, dict):
                            continue

                        source = flow_data.get('source', comp_id) if direction == 'inputs' else comp_id
                        dest = flow_data.get('destination', comp_id) if direction == 'outputs' else comp_id

                        self.flow_tracker.record_flow(
                            source_component=source,
                            destination_component=dest,
                            flow_type=flow_name,
                            amount=flow_data.get('value', 0.0),
                            unit=flow_data.get('unit', '')
                        )

            if comp_id not in self.component_metrics:
                self.component_metrics[comp_id] = {}
            for metric_name, metric_value in state.items():
                if metric_name not in self.component_metrics[comp_id]:
                    self.component_metrics[comp_id][metric_name] = []
                self.component_metrics[comp_id][metric_name].append(metric_value)

    def _extract_component_metric(self, comp_id: str, metric: str) -> List[Any]:
        """
        Extract a specific metric time-series for a component.

        Args:
            comp_id (str): Component identifier.
            metric (str): Metric name.

        Returns:
            List[Any]: Metric values over time, or empty list if not found.
        """
        return self.component_metrics.get(comp_id, {}).get(metric, [])

    def export_dashboard_data(self, filename: str = "dashboard_data.json") -> Path:
        """
        Export data optimized for dashboard consumption.

        Generates a comprehensive JSON structure with time-series,
        flow visualization data, and KPIs.

        Args:
            filename (str): Output filename. Default: 'dashboard_data.json'.

        Returns:
            Path: Path to exported file.
        """
        dashboard_data = {
            'metadata': {
                'simulation_hours': len(self.timeseries['hour']),
                'start_date': '2025-01-01',
                'timestep_hours': 1.0
            },
            'timeseries': {
                'hour': self.timeseries['hour'],
                'production': {
                    'electrolyzer_kg': (
                        self._extract_component_metric('electrolyzer', 'h2_output_kg') or
                        self._extract_component_metric('pem_electrolyzer_detailed', 'h2_output_kg')
                    ),
                    'atr_kg': (
                        self._extract_component_metric('atr', 'h2_output_kg') or
                        self._extract_component_metric('atr_system', 'h2_output_kg')
                    ),
                    'total_kg': self.timeseries['total_production_kg']
                },
                'storage': {
                    'lp_level_kg': self._extract_component_metric('lp_tanks', 'total_mass_kg'),
                    'hp_level_kg': self._extract_component_metric('hp_tanks', 'total_mass_kg')
                },
                'energy': {
                    'electrolyzer_kwh': (
                        self._extract_component_metric('electrolyzer', 'cumulative_energy_kwh') or
                        self._extract_component_metric('pem_electrolyzer_detailed', 'cumulative_energy_kwh')
                    ),
                    'compression_kwh': self._extract_component_metric('filling_compressor', 'cumulative_energy_kwh'),
                },
                'demand': {
                    'requested_kg': self.timeseries['total_demand_kg'],
                },
                'price': {
                    'per_mwh': self.timeseries['energy_price_mwh']
                }
            },
            'flows': {
                'sankey': self.flow_tracker.get_sankey_data(),
                'matrix': self.flow_tracker.get_flow_matrix()
            },
            'kpis': self.get_summary()
        }

        output_path = self.metrics_dir / filename
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, cls=NpEncoder)

        logger.info(f"Dashboard data exported to: {output_path}")
        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics and KPIs.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_production_kg: Total H₂ produced.
                - total_demand_kg: Total H₂ demanded.
                - total_cost: Cumulative cost.
                - average_cost_per_kg: Production cost efficiency.
                - demand_fulfillment_rate: Demand satisfaction ratio.
        """
        summary = {
            'total_production_kg': self.total_production_kg,
            'total_demand_kg': self.total_demand_kg,
            'total_cost': self.total_cost,
            'average_cost_per_kg': (
                self.total_cost / self.total_production_kg
                if self.total_production_kg > 0 else 0.0
            ),
            'demand_fulfillment_rate': (
                self.total_production_kg / self.total_demand_kg
                if self.total_demand_kg > 0 else 0.0
            )
        }
        return summary

    def export_timeseries(self, filename: str = "timeseries.csv") -> Path:
        """
        Export time-series data to CSV format.

        Args:
            filename (str): Output filename. Default: 'timeseries.csv'.

        Returns:
            Path: Path to exported file.
        """
        output_path = self.metrics_dir / filename
        num_rows = len(self.timeseries['hour'])
        headers = list(self.timeseries.keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i in range(num_rows):
                row = [self.timeseries[header][i] for header in headers]
                writer.writerow(row)

        logger.info(f"Time-series data exported to: {output_path}")
        return output_path

    def export_summary(self, filename: str = "summary.json") -> Path:
        """
        Export summary statistics to JSON format.

        Args:
            filename (str): Output filename. Default: 'summary.json'.

        Returns:
            Path: Path to exported file.
        """
        output_path = self.metrics_dir / filename
        summary = self.get_summary()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NpEncoder)

        logger.info(f"Summary exported to: {output_path}")
        return output_path

    def export_raw_metrics(self, filename: str = "raw_metrics.json") -> Path:
        """
        Export full raw metrics for advanced analysis.

        Args:
            filename (str): Output filename. Default: 'raw_metrics.json'.

        Returns:
            Path: Path to exported file.
        """
        raw_data = {
            'timeseries': self.timeseries,
            'component_metrics': self.component_metrics
        }

        output_path = self.metrics_dir / filename
        with open(output_path, 'w') as f:
            json.dump(raw_data, f, indent=2, cls=NpEncoder)

        logger.info(f"Raw metrics exported to: {output_path}")
        return output_path
