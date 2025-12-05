"""
Flow tracking component for monitoring mass and energy flows between components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import IntEnum

if TYPE_CHECKING:
    from h2_plant.core.stream import Stream

class FlowType(IntEnum):
    """Types of flows in the system."""
    ELECTRICAL_ENERGY = 0   # kWh or MW
    HYDROGEN_MASS = 1       # kg
    OXYGEN_MASS = 2         # kg
    NATURAL_GAS_MASS = 3    # kg
    THERMAL_ENERGY = 4      # kWh
    WATER_MASS = 5          # kg
    COMPRESSION_WORK = 6    # kWh
    CO2_EMISSIONS = 7       # kg


@dataclass
class Flow:
    """Represents a flow between components."""
    hour: int
    flow_type: FlowType
    source_component: str
    destination_component: str
    amount: float
    unit: str
    temperature_k: Optional[float] = None
    pressure_pa: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class FlowTracker:
    """
    Tracks all flows (energy, mass, work) between components.
    
    Integrates with MonitoringSystem to provide flow analytics.
    """
    
    def __init__(self):
        self.flows: List[Flow] = []
        self.current_hour: int = 0
    
    def set_current_hour(self, hour: int):
        self.current_hour = hour

    def record_flow(
        self, 
        source_component: str, 
        destination_component: str, 
        flow_type: str, 
        amount: float, 
        unit: str,
        temperature_k: Optional[float] = None,
        pressure_pa: Optional[float] = None
    ) -> None:
        """Record a flow event."""
        try:
            flow_type_enum = FlowType[flow_type.upper()]
        except KeyError:
            # Handle cases where flow_type string doesn't match an enum member
            # For example, log a warning or skip recording the flow
            return
            
        flow = Flow(
            hour=self.current_hour,
            flow_type=flow_type_enum,
            source_component=source_component,
            destination_component=destination_component,
            amount=amount,
            unit=unit,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa
        )
        self.flows.append(flow)

    def record_stream(
        self,
        source_component: str,
        destination_component: str,
        stream: 'Stream',
        flow_type: str = 'HYDROGEN_MASS'
    ) -> None:
        """Record a Stream flow event."""
        if stream.mass_flow_kg_h <= 0:
            return
            
        self.record_flow(
            source_component=source_component,
            destination_component=destination_component,
            flow_type=flow_type,
            amount=stream.mass_flow_kg_h,
            unit='kg',
            temperature_k=stream.temperature_k,
            pressure_pa=stream.pressure_pa
        )
        
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of all recorded flows.
        """
        summary = {}
        for flow in self.flows:
            key = f"{flow.source_component}_to_{flow.destination_component}_{flow.flow_type.name}"
            if key not in summary:
                summary[key] = {'total_amount': 0.0, 'unit': flow.unit, 'count': 0}
            summary[key]['total_amount'] += flow.amount
            summary[key]['count'] += 1
        return summary

    def get_sankey_data(self) -> Dict[str, List]:
        """
        Generate Sankey diagram data structure.
        
        Format compatible with Plotly/D3.js Sankey diagrams.
        """
        nodes = []
        links = []
        node_indices: Dict[str, int] = {}

        def get_or_add_node(name: str):
            if name not in node_indices:
                node_indices[name] = len(nodes)
                nodes.append({'name': name})
            return node_indices[name]

        link_aggregates: Dict[tuple, float] = {}
        for flow in self.flows:
            source_idx = get_or_add_node(flow.source_component)
            target_idx = get_or_add_node(flow.destination_component)
            link_key = (source_idx, target_idx, flow.flow_type.name)
            
            link_aggregates[link_key] = link_aggregates.get(link_key, 0.0) + flow.amount
        
        for (source_idx, target_idx, flow_type_name), value in link_aggregates.items():
            if value > 1e-6: # Only add links with significant flow
                links.append({
                    'source': source_idx,
                    'target': target_idx,
                    'value': value,
                    'label': flow_type_name.replace('_', ' ').title()
                })
        
        return {
            'nodes': nodes,
            'links': links
        }
    
    def get_flow_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get flow matrix for Sankey diagram generation.
        """
        matrix: Dict[str, Dict[str, float]] = {}
        for flow in self.flows:
            key = f"{flow.source_component} -> {flow.destination_component}"
            sub_key = f"{flow.flow_type.name} ({flow.unit})"
            if key not in matrix:
                matrix[key] = {}
            matrix[key][sub_key] = matrix[key].get(sub_key, 0.0) + flow.amount
        return matrix
