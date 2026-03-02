"""
Flow Tracker for Mass and Energy Flow Monitoring - Streaming Implementation.

This module provides flow tracking infrastructure for recording and
analyzing mass and energy transfers between system components.

It implements a BUFFERED STREAMING architecture to prevent memory leaks
during long simulations.

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
import json
from pathlib import Path
import logging

if TYPE_CHECKING:
    from h2_plant.core.stream import Stream

logger = logging.getLogger(__name__)

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
        HYDROGEN_RFNBO: Green certified H2 (renewable powered)
        HYDROGEN_NON_RFNBO: Non-certified H2 (grid powered)
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
    STREAM_MASS = 10        # Generic multi-component stream (fallback for unmapped resource types)


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

    def to_dict(self):
        return {
            'hour': self.hour,
            'flow_type': self.flow_type.name,
            'source_component': self.source_component,
            'destination_component': self.destination_component,
            'amount': self.amount,
            'unit': self.unit,
            'temperature_k': self.temperature_k,
            'pressure_pa': self.pressure_pa,
            'metadata': self.metadata,
        }


class FlowTracker:
    """
    Tracks all flows (energy, mass, work) between components via Streaming.

    Integrates with MonitoringSystem to provide flow analytics
    and visualization data for Sankey diagrams.

    Start: In-Memory List (Leak) -> End: Buffered Stream (Constant RAM)

    Attributes:
        buffer (List[Flow]): Temporary buffer for flow records.
        current_hour (int): Current simulation hour for new recordings.
        aggregates (Dict): Running totals for O(1) Sankey generation.
    """

    def __init__(self, buffer_size: int = 1000):
        """
        Initialize the flow tracker.
        
        Args:
            buffer_size (int): Number of records to keep before auto-flush.
        """
        self.buffer: List[tuple] = []
        self.current_hour: int = 0
        self.buffer_size = buffer_size
        
        # O(1) Running Aggregates for Dashboard support without reading full history
        # Key: (source, dest, flow_type_name) -> Value: total_amount
        self.aggregates: Dict[tuple, float] = {}
        
        # Matrix Aggregate: Key: "src->dest", Subkey: "type (unit)" -> Value: amount
        self.matrix_aggregates: Dict[str, Dict[str, float]] = {}
        # Cache formatted matrix keys to avoid repeated f-string overhead
        self._matrix_key_cache: Dict[tuple, str] = {}
        self._matrix_subkey_cache: Dict[tuple, str] = {}
        
        self.log_interval: int = 1  # Will be controlled by UI in Phase 2
        self._total_buffer_appends: int = 0

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
        pressure_pa: Optional[float] = None,
        step_idx: int = 0
    ) -> None:
        """
        Record a flow event between components.
        """
        try:
            ft_name = flow_type.upper()
            FlowType[ft_name]  # validate enum membership
        except KeyError:
            return

        is_log_step = (self.log_interval <= 1) or (step_idx % self.log_interval == 0)

        # Store as lightweight tuple instead of dataclass (avoids __init__ overhead)
        if is_log_step:
            self.buffer.append((
                self.current_hour, ft_name, source_component,
                destination_component, amount, unit, temperature_k, pressure_pa
            ))
            self._total_buffer_appends += 1

        # Update aggregates
        s_key = (source_component, destination_component, ft_name)
        self.aggregates[s_key] = self.aggregates.get(s_key, 0.0) + amount

        # Matrix aggregates (cached keys)
        pair_key = (source_component, destination_component)
        m_key_main = self._matrix_key_cache.get(pair_key)
        if m_key_main is None:
            m_key_main = f"{source_component} -> {destination_component}"
            self._matrix_key_cache[pair_key] = m_key_main

        sub_key = (ft_name, unit)
        m_key_sub = self._matrix_subkey_cache.get(sub_key)
        if m_key_sub is None:
            m_key_sub = f"{ft_name} ({unit})"
            self._matrix_subkey_cache[sub_key] = m_key_sub
        if m_key_main not in self.matrix_aggregates:
            self.matrix_aggregates[m_key_main] = {}
        self.matrix_aggregates[m_key_main][m_key_sub] = \
            self.matrix_aggregates[m_key_main].get(m_key_sub, 0.0) + amount

    def record_stream(
        self,
        source_component: str,
        destination_component: str,
        stream: 'Stream',
        flow_type: str = 'HYDROGEN_MASS',
        step_idx: int = 0
    ) -> None:
        """
        Record a Stream object as a flow event.

        Convenience method that extracts flow properties from Stream.

        Args:
            source_component (str): Source component ID.
            destination_component (str): Destination component ID.
            stream (Stream): Stream object to record.
            flow_type (str): Flow type name. Default: 'HYDROGEN_MASS'.
            step_idx (int): Current simulation step index for throttling.
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
            pressure_pa=stream.pressure_pa,
            step_idx=step_idx
        )
    
    def flush(self, filepath: Path, *, rolling: bool = False, reset_aggregates: bool = False) -> None:
        """
        Flush buffered flows to disk (JSON Lines format).
        
        Args:
            filepath (Path): Output file path.
            rolling (bool): If True, overwrite file each flush (keeps only last interval).
            reset_aggregates (bool): If True, clear aggregates/matrix after flush.
        """
        if not self.buffer:
            return
            
        try:
            mode = 'w' if rolling else ('a' if filepath.exists() else 'w')
            if not filepath.parent.exists():
                filepath.parent.mkdir(parents=True, exist_ok=True)

            # Batch-serialize tuples directly (avoids to_dict + json.dumps overhead)
            parts = []
            for rec in self.buffer:
                hour, ft, src, dst, amt, unit, temp, pres = rec
                # Manual JSON construction for flat records (5-10x faster than json.dumps)
                line = (f'{{"hour":{hour},"flow_type":"{ft}",'
                        f'"source_component":"{src}","destination_component":"{dst}",'
                        f'"amount":{amt},"unit":"{unit}",'
                        f'"temperature_k":{temp if temp is not None else "null"},'
                        f'"pressure_pa":{pres if pres is not None else "null"}}}')
                parts.append(line)
            with open(filepath, mode) as f:
                f.write('\n'.join(parts))
                if parts:
                    f.write('\n')

            # Clear buffer after successful write
            self.buffer = []
            if reset_aggregates:
                self.aggregates = {}
                self.matrix_aggregates = {}

        except Exception as e:
            logger.error(f"Failed to flush flow tracker: {e}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of all recorded flows (from aggregators).

        Returns:
            Dict[str, Any]: Dictionary with total amounts and counts per flow path.
        """
        summary = {}
        # Reconstruct summary from matrix aggregates for O(1) access
        for path, types in self.matrix_aggregates.items():
            for type_unit, amount in types.items():
                key = f"{path}_{type_unit}"
                summary[key] = {'total_amount': amount}
        return summary

    def get_sankey_data(self) -> Dict[str, List]:
        """
        Generate Sankey diagram data structure from O(1) aggregates.

        Returns:
            Dict[str, List]: Dictionary with 'nodes' and 'links' arrays.
        """
        nodes = []
        links = []
        node_indices: Dict[str, int] = {}

        def get_or_add_node(name: str):
            if name not in node_indices:
                node_indices[name] = len(nodes)
                nodes.append({'name': name})
            return node_indices[name]

        for (source, dest, flow_name), value in self.aggregates.items():
            if value > 1e-6:
                source_idx = get_or_add_node(source)
                target_idx = get_or_add_node(dest)
                
                links.append({
                    'source': source_idx,
                    'target': target_idx,
                    'value': value,
                    'label': flow_name.replace('_', ' ').title()
                })

        return {
            'nodes': nodes,
            'links': links
        }

    def get_flow_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get flow matrix for tabular analysis from O(1) aggregates.

        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary of flow totals.
        """
        return self.matrix_aggregates
