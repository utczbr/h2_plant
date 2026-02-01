"""
Monitoring and Metrics Collection System.

This module provides the central monitoring infrastructure for collecting,
aggregating, and exporting simulation metrics and KPIs.

OPTIMIZED: This version uses Zero-Redundancy Storage.
Detailed component metrics are fetched directly from the Dispatch Strategy's
efficient ChunkedHistoryManager (Parquet/Arrow) instead of being duplicated in RAM.

Metrics Categories:
    - **Component metrics**: Per-component state tracking over time (via Dispatch History).
    - **System aggregates**: Total production, storage, demand (via Dispatch History).
    - **Time-series data**: Hourly snapshots for trend analysis.
    - **Flow tracking**: Mass and energy flow between components (Streamed to disk).
    - **KPIs**: Efficiency, cost, fulfillment rate.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import csv
import json
import numpy as np

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.flow_tracker import FlowTracker
from h2_plant.visualization import utils

logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
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


class MonitoringSystem:
    """
    Central metrics collection and aggregation system.
    
    Zero-Redundancy Architecture:
    - Detailed Metrics -> ChunkedHistoryManager (Disk/Parquet)
    - Flows -> FlowTracker (Streamed JSONL)
    - Scalars -> self.timeseries (RAM - Tiny overhead)
    """

    def __init__(self, output_dir: Path, lightweight_mode: bool = True):
        self.output_dir = output_dir
        self.metrics_dir = output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Enforce lightweight mode to prevent RAM leaks
        self._lightweight_mode = True 
        logger.info("MonitoringSystem: Lightweight mode ENFORCED (Zero-Redundancy Architecture)")

        self.flow_tracker = FlowTracker()

        # Keep lightweight scalars in memory
        self.timeseries: Dict[str, List[Any]] = {
            'hour': [],
            'total_production_kg': [],
            'total_storage_kg': [],
            'total_demand_kg': [],
            'energy_price_mwh': [],
            'total_cost': [],
            'total_compression_energy_kwh': []
        }

        # Legacy redundant storage - REMOVED
        self.component_metrics: Dict[str, Dict[str, List]] = {}

        self.total_production_kg = 0.0
        self.total_demand_kg = 0.0
        self.total_cost = 0.0
        self.total_compression_energy_kwh = 0.0
        
        # RFNBO Classification Metrics
        self.h2_rfnbo_total_kg = 0.0
        self.h2_non_rfnbo_total_kg = 0.0
        self.spot_purchases_mwh = 0.0

    def initialize(self, registry: ComponentRegistry) -> None:
        logger.info("Monitoring system initialized (Zero-Redundancy Mode)")

    def collect(self, hour: int, registry: ComponentRegistry) -> None:
        """
        Collect metrics and flows for the current timestep.
        """
        self.flow_tracker.set_current_hour(hour)
        states = registry.get_all_states()

        self.timeseries['hour'].append(hour)

        # Filter None states
        valid_states = []
        for cid, s in states.items():
            if s is not None:
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
        current_demand = 0.0
        if registry.has('demand_scheduler'):
            demand_state = registry.get('demand_scheduler').get_state()
            current_demand = demand_state.get('current_demand_kg', 0.0)
        self.timeseries['total_demand_kg'].append(current_demand)
        self.total_demand_kg += current_demand

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
        self.timeseries['total_cost'].append(total_cost)
        self.total_cost = total_cost
        
        # Aggregate Compression Energy
        comp_energy_step = sum(
            s.get('compression_work_kwh', 0.0) 
            for c, s in states.items() 
            if ('compressor' in c.lower() or 'compression' in c.lower()) and s is not None
        )
        self.total_compression_energy_kwh += comp_energy_step
        self.timeseries['total_compression_energy_kwh'].append(self.total_compression_energy_kwh)

        # Record flows (Streamed to disk)
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

    def export_dashboard_data(self, engine: Any = None, filename: str = "dashboard_data.json") -> Path:
        """
        Export data optimized for dashboard consumption.
        
        ZERO-REDUNDANCY IMPLEMENTATION:
        Extracts high-resolution data directly from the Dispatch Strategy's history
        (ChunkedHistoryManager) instead of local RAM.
        
        Args:
            engine: Reference to SimulationEngine (needed to access dispatch strategy).
            filename (str): Output filename.
        """
        # 1. Fetch History Accessor
        history = {}
        if engine and hasattr(engine, 'dispatch_strategy'):
            try:
                # This returns either the dict or the HistoryDictProxy (Parquet reader)
                history = engine.dispatch_strategy.get_history()
            except Exception as e:
                logger.error(f"Failed to retrieve dispatch history: {e}")
        
        if not history:
            logger.warning("No history available for dashboard export. Graphs may be empty.")

        # 2. Extract Columns Helper
        def get_col(col_name: str, default_val: float = 0.0) -> List[float]:
            if col_name in history:
                return utils.downsample_list(history[col_name].tolist())
            return []

        # 3. Construct Dashboard JSON
        dashboard_data = {
            'metadata': {
                'simulation_hours': len(self.timeseries['hour']),
                'start_date': '2025-01-01',
                'timestep_hours': 1.0
            },
            'timeseries': {
                'hour': utils.downsample_list(self.timeseries['hour']),
                # Map Dispatch History Columns to Dashboard Keys
                'production': {
                    'electrolyzer_kg': get_col('H2_soec_kg'),
                    'pem_kg': get_col('H2_pem_kg'),
                    'atr_kg': get_col('H2_atr_kg'),
                    'total_kg': get_col('h2_kg')
                },
                'storage': {
                    'level_kg': get_col('tank_level_kg'),
                    'pressure_bar': get_col('tank_pressure_bar')
                },
                'energy': {
                    'soec_power_mw': get_col('P_soec_actual'),
                    'pem_power_mw': get_col('P_pem'),
                    'bop_power_mw': get_col('P_bop_mw'),
                    'compressor_kw': get_col('compressor_power_kw'),
                },
                'demand': {
                    'requested_kg': utils.downsample_list(self.timeseries['total_demand_kg']),
                },
                'price': {
                    'per_mwh': get_col('spot_price'),
                    'cost_eur': get_col('cumulative_bop_cost_eur') # Example mapping
                }
            },
            'flows': {
                'sankey': self.flow_tracker.get_sankey_data(),
                'matrix': self.flow_tracker.get_flow_matrix()
            },
            'kpis': self.get_summary(history=history)
        }

        output_path = self.metrics_dir / filename
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, cls=NpEncoder)

        logger.info(f"Dashboard data exported to: {output_path}")
        return output_path

    def get_summary(self, history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get summary of monitoring metrics with optional history integration.
        
        Args:
            history (Dict[str, Any], optional): Simulation history dictionary or DataFrame.
                If provided, some metrics (like RFNBO totals) will be calculated
                by aggregating the history data to ensure consistency with dispatch results.

        Returns:
            Dict[str, Any]: Summary dictionary
        """
        if history is not None:
            # If history is provided, verify/update RFNBO totals from it
            # This handles the case where DispatchStrategy records to history directly
            # but MonitoringSystem.collect() didn't see the internal variables.
            
            # Check for standard numpy/list history or pandas DataFrame
            is_dataframe = hasattr(history, 'columns') and hasattr(history, 'sum')
            
            try:
                # RFNBO Total
                if is_dataframe and 'h2_rfnbo_kg' in history.columns:
                     self.h2_rfnbo_total_kg = float(history['h2_rfnbo_kg'].sum())
                elif not is_dataframe and 'h2_rfnbo_kg' in history:
                     self.h2_rfnbo_total_kg = float(np.sum(history['h2_rfnbo_kg']))
                
                # Non-RFNBO Total
                if is_dataframe and 'h2_non_rfnbo_kg' in history.columns:
                     self.h2_non_rfnbo_total_kg = float(history['h2_non_rfnbo_kg'].sum())
                elif not is_dataframe and 'h2_non_rfnbo_kg' in history:
                     self.h2_non_rfnbo_total_kg = float(np.sum(history['h2_non_rfnbo_kg']))
                     
            except Exception as e:
                logger.warning(f"Failed to aggregate RFNBO totals from history: {e}")

        # Calculate RFNBO Percentage
        total_h2_classified = self.h2_rfnbo_total_kg + self.h2_non_rfnbo_total_kg
        rfnbo_compliance = (
            (self.h2_rfnbo_total_kg / total_h2_classified * 100) 
            if total_h2_classified > 0 else 0.0
        )
        
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
            ),
            'h2_rfnbo_kg': self.h2_rfnbo_total_kg,
            'h2_non_rfnbo_kg': self.h2_non_rfnbo_total_kg,
            'rfnbo_compliance_pct': rfnbo_compliance,
            'spot_purchases_mwh': self.spot_purchases_mwh,
            'flow_stats': self.flow_tracker.get_summary_statistics()
        }
        return summary

    def export_timeseries(self, filename: str = "timeseries.csv") -> Path:
        """Export scalar time-series data to CSV (Lightweight metrics only)."""
        output_path = self.metrics_dir / filename
        
        if not self.timeseries['hour']:
            return output_path
            
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
        output_path = self.metrics_dir / filename
        summary = self.get_summary()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NpEncoder)
        return output_path

    def export_raw_metrics(self, filename: str = "raw_metrics.json") -> Path:
        """
        Export raw metrics.
        Note: detailed component_metrics are NO LONGER AVAILABLE in raw json
        as they are stored in Parquet files by the Dispatch Strategy.
        """
        raw_data = {
            'timeseries': self.timeseries,
            'note': "Detailed component metrics are stored in parquet history files."
        }

        output_path = self.metrics_dir / filename
        with open(output_path, 'w') as f:
            json.dump(raw_data, f, indent=2, cls=NpEncoder)
        return output_path
