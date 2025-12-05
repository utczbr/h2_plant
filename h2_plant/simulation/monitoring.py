"""
Monitoring and metrics collection system.

Tracks:
- Component-level metrics (production rates, storage levels, costs)
- System-level aggregates
- Time-series data
- Performance indicators (KPIs)
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

class MonitoringSystem:
    """
    Collects and aggregates simulation metrics.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize monitoring system.
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
        Initialize monitoring for registered components.
        """
        for component_id in registry.get_all_ids():
            self.component_metrics[component_id] = {}
        logger.info("Monitoring system initialized")
    
    def collect(self, hour: int, registry: ComponentRegistry) -> None:
        """
        Collect metrics and flows for the current timestep.
        """
        self.flow_tracker.set_current_hour(hour)
        states = registry.get_all_states()
        
        self.timeseries['hour'].append(hour)
        
        # --- Aggregate Time-series Data ---
        # Filter out None states and log warning
        valid_states = []
        for cid, s in states.items():
            if s is None:
                logger.warning(f"Component {cid} returned None state")
            else:
                valid_states.append(s)

        total_production = sum(s.get('h2_output_kg', 0.0) for s in valid_states)
        # Ensure we capture detailed components if they don't report h2_output_kg in the same way or if we need to be explicit
        # But 'h2_output_kg' is the standard interface key.
        # Let's verify if detailed components return it.
        # DetailedPEMElectrolyzer returns 'h2_output_kg'
        # SOECMultiModuleCluster returns 'h2_production_kg_h' (rate) and 'h2_output_kg' (mass)
        # So the sum above should work IF the components are in valid_states.
        # However, if 'electrolyzer' (legacy) is NOT there, but 'pem_electrolyzer_detailed' IS, we rely on the loop.
        # The loop iterates over ALL states.
        
        # Issue might be that 'h2_output_kg' is 0 for some reason, or key is missing.
        # Let's add a debug log or be more robust.
        # Also, we need to make sure we don't double count if both legacy and detailed are present (unlikely with current builder).
        
        self.timeseries['total_production_kg'].append(total_production)
        self.total_production_kg += total_production

        total_storage = sum(s.get('total_mass_kg', s.get('mass_kg', 0.0)) for c, s in states.items() if ('storage' in c or 'tank' in c) and s is not None)
        self.timeseries['total_storage_kg'].append(total_storage)
        
        if registry.has('demand_scheduler'):
            demand_state = registry.get('demand_scheduler').get_state()
            current_demand = demand_state.get('current_demand_kg', 0.0)
            self.timeseries['total_demand_kg'].append(current_demand)
            self.total_demand_kg += current_demand
        else:
            self.timeseries['total_demand_kg'].append(0.0)

        if registry.has('energy_price_tracker'):
            price_state = registry.get('energy_price_tracker').get_state()
            self.timeseries['energy_price_mwh'].append(price_state.get('current_price_per_mwh', 0.0))
        else:
            self.timeseries['energy_price_mwh'].append(0.0)

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
        
        # --- Record Flows and Component-specific Metrics ---
        for comp_id, state in states.items():
            if state is None: continue
            if 'flows' in state and isinstance(state['flows'], dict):
                for direction, flow_group in state['flows'].items():
                    if not isinstance(flow_group, dict): continue
                    for flow_name, flow_data in flow_group.items():
                        if not isinstance(flow_data, dict): continue
                        
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
        return self.component_metrics.get(comp_id, {}).get(metric, [])

    def export_dashboard_data(self, filename: str = "dashboard_data.json") -> Path:
        """
        Export data optimized for dashboard consumption.
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
        Get summary statistics.
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
        Export time-series data to CSV.
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
        Export summary statistics to JSON.
        """
        output_path = self.metrics_dir / filename
        summary = self.get_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NpEncoder)
        
        logger.info(f"Summary exported to: {output_path}")
        return output_path

    def export_raw_metrics(self, filename: str = "raw_metrics.json") -> Path:
        """
        Export full raw metrics for advanced visualization.
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
