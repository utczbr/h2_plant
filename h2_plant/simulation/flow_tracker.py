"""
Flow Tracker for Mass and Energy Flow Monitoring.

This module provides flow tracking infrastructure for recording and
analyzing mass and energy transfers between system components.

Flow Categories:
    - **Electrical energy**: Power consumption and distribution (kWh, MW).
    - **Hydrogen mass**: H₂ production and storage flows (kg).
    - **Oxygen mass**: O₂ byproduct flows (kg).
    - **Thermal energy**: Heat transfer between components (kWh).
    - **Water mass**: Feedwater and cooling flows (kg).
    - **Compression work**: Compressor energy consumption (kWh).

Visualization Support:
    The tracker generates data structures compatible with Sankey diagrams
    (Plotly/D3.js) for visualizing energy and mass flow distribution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import IntEnum

if TYPE_CHECKING:
    from h2_plant.core.stream import Stream


class FlowType(IntEnum):
    """
    Enumeration of tracked flow types in the system.

    Attributes:
        ELECTRICAL_ENERGY: Electrical power flows (kWh or MW).
        HYDROGEN_MASS: Hydrogen gas mass flows (kg).
        OXYGEN_MASS: Oxygen gas mass flows (kg).
        NATURAL_GAS_MASS: Natural gas fuel flows (kg).
        THERMAL_ENERGY: Heat energy flows (kWh).
        WATER_MASS: Water/steam mass flows (kg).
        COMPRESSION_WORK: Compression energy consumption (kWh).
        CO2_EMISSIONS: Carbon dioxide emissions (kg).
        HYDROGEN_RFNBO: Green certified hydrogen (kg).
        HYDROGEN_NON_RFNBO: Non-certified hydrogen (kg).
    """
    ELECTRICAL_ENERGY = 0
    HYDROGEN_MASS = 1
    OXYGEN_MASS = 2
    NATURAL_GAS_MASS = 3
    THERMAL_ENERGY = 4
    WATER_MASS = 5
    COMPRESSION_WORK = 6
    CO2_EMISSIONS = 7
    HYDROGEN_RFNBO = 8      # Green certified H2 (renewable powered)
    HYDROGEN_NON_RFNBO = 9  # Non-certified H2 (grid powered)


@dataclass
class Flow:
    """
    Represents a single flow record between components.

    Attributes:
        hour (int): Simulation hour when flow occurred.
        flow_type (FlowType): Category of flow.
        source_component (str): Source component ID.
        destination_component (str): Destination component ID.
        amount (float): Flow quantity.
        unit (str): Unit of measurement.
        temperature_k (float, optional): Stream temperature in K.
        pressure_pa (float, optional): Stream pressure in Pa.
        metadata (Dict): Additional flow properties.
    """
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

    Integrates with MonitoringSystem to provide flow analytics
    and visualization data for Sankey diagrams.

    Attributes:
        flows (List[Flow]): Recorded flow events.
        current_hour (int): Current simulation hour for new recordings.

    Example:
        >>> tracker = FlowTracker()
        >>> tracker.set_current_hour(100)
        >>> tracker.record_flow("electrolyzer", "storage", "HYDROGEN_MASS", 50.0, "kg")
        >>> sankey_data = tracker.get_sankey_data()
    """

    def __init__(self):
        """Initialize the flow tracker with empty flow list."""
        self.flows: List[Flow] = []
        self.current_hour: int = 0

    def set_current_hour(self, hour: int) -> None:
        """
        Set the current simulation hour for new recordings.

        Args:
            hour (int): Current simulation hour.
        """
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
        """
        Record a flow event between components.

        Args:
            source_component (str): Source component ID.
            destination_component (str): Destination component ID.
            flow_type (str): Flow type name (must match FlowType enum).
            amount (float): Flow quantity.
            unit (str): Unit of measurement.
            temperature_k (float, optional): Stream temperature in K.
            pressure_pa (float, optional): Stream pressure in Pa.
        """
        try:
            flow_type_enum = FlowType[flow_type.upper()]
        except KeyError:
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
        """
        Record a Stream object as a flow event.

        Convenience method that extracts flow properties from Stream.

        Args:
            source_component (str): Source component ID.
            destination_component (str): Destination component ID.
            stream (Stream): Stream object to record.
            flow_type (str): Flow type name. Default: 'HYDROGEN_MASS'.
        """
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

        Aggregates flows by source-destination-type combination.

        Returns:
            Dict[str, Any]: Dictionary with total amounts and counts per flow path.
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

        Format is compatible with Plotly and D3.js Sankey visualizations.
        Flows are aggregated by source-destination-type combination.

        Returns:
            Dict[str, List]: Dictionary with 'nodes' and 'links' arrays.

        Example:
            >>> data = tracker.get_sankey_data()
            >>> # data['nodes'] = [{'name': 'electrolyzer'}, ...]
            >>> # data['links'] = [{'source': 0, 'target': 1, 'value': 100}, ...]
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
            if value > 1e-6:
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
        Get flow matrix for tabular analysis.

        Groups flows by source→destination path with flow type breakdown.

        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary of flow totals.

        Example:
            >>> matrix = tracker.get_flow_matrix()
            >>> # matrix['electrolyzer -> storage']['HYDROGEN_MASS (kg)'] = 500.0
        """
        matrix: Dict[str, Dict[str, float]] = {}
        for flow in self.flows:
            key = f"{flow.source_component} -> {flow.destination_component}"
            sub_key = f"{flow.flow_type.name} ({flow.unit})"
            if key not in matrix:
                matrix[key] = {}
            matrix[key][sub_key] = matrix[key].get(sub_key, 0.0) + flow.amount
        return matrix
