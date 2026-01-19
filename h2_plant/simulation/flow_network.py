"""
Flow Network for Dynamic Resource Routing.

This module implements the flow network that routes mass and energy streams
between connected components based on the configured topology.

Network Architecture:
    The FlowNetwork operates on the configured topology (ConnectionConfig)
    to propagate streams between components:

    1. **Get output** from source component port.
    2. **Push input** to target component port.
    3. **Extract feedback** to source for mass balance.

    This pull-push-extract pattern ensures bidirectional consistency.

Port Interface:
    Components expose ports via `get_ports()`:
    - Input ports accept resources via `receive_input()`.
    - Output ports provide resources via `get_output()`.
    - Resources extracted via `extract_output()` after acceptance.

Connection Types:
    - **Legacy**: Direct component ID connections.
    - **Indexed**: Component name + index for array components.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import ConnectionConfig, IndexedConnectionConfig
from h2_plant.core.stream import Stream
from h2_plant.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class FlowNetwork:
    """
    Manages the network of resource connections between components.

    Validates topology at initialization and executes flows at each
    timestep by routing streams from source to target ports.

    Responsibilities:
        1. Validate connection topology (components and ports exist).
        2. Execute flows at each timestep (get → push → extract).
        3. Handle flow constraints and resource type compatibility.

    Attributes:
        registry (ComponentRegistry): Component registry for lookups.
        topology (List[ConnectionConfig]): Legacy connection definitions.
        indexed_topology (List[IndexedConnectionConfig]): Indexed connections.

    Example:
        >>> network = FlowNetwork(registry, topology, indexed_topology)
        >>> network.initialize()
        >>> for step in simulation:
        ...     network.execute_flows(t)
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        topology: List[ConnectionConfig],
        indexed_topology: List[IndexedConnectionConfig] = None
    ):
        """
        Initialize the FlowNetwork.

        Args:
            registry (ComponentRegistry): Registry containing all components.
            topology (List[ConnectionConfig]): Legacy connection configs.
            indexed_topology (List[IndexedConnectionConfig], optional):
                Indexed connection configs for array components.
        """
        self.registry = registry
        self.topology = topology
        self.indexed_topology = indexed_topology or []
        self._connections: List[ConnectionConfig] = topology
        self._indexed_connections: List[IndexedConnectionConfig] = self.indexed_topology

    def initialize(self) -> None:
        """
        Initialize and validate the flow network.

        Verifies that all source/target components exist in registry
        and that port definitions are compatible.

        Raises:
            ConfigurationError: If component or port validation fails.
        """
        logger.debug(
            f"Initializing FlowNetwork with {len(self._connections)} legacy "
            f"and {len(self._indexed_connections)} indexed connections"
        )
        self._validate_connections()
        
        # PERFORMANCE: Pre-resolve component objects for O(1) access during simulation
        # Eliminates ~2M registry.get() dict lookups per 480h simulation
        self._source_cache: Dict[str, Any] = {}
        self._target_cache: Dict[str, Any] = {}
        
        for conn in self._connections:
            self._source_cache[conn.source_id] = self.registry.get(conn.source_id)
            self._target_cache[conn.target_id] = self.registry.get(conn.target_id)
            
        for conn in self._indexed_connections:
            src_id = self._resolve_component_id(conn.source_name, conn.source_index)
            tgt_id = self._resolve_component_id(conn.target_name, conn.target_index)
            self._source_cache[src_id] = self.registry.get(src_id)
            self._target_cache[tgt_id] = self.registry.get(tgt_id)

    def execute_flows(self, t: float) -> None:
        """
        Execute all flows for the current timestep.

        Iterates through all connections, pulling output from source
        and pushing to target. Flow execution continues even if
        individual connections fail (error is logged).

        Args:
            t (float): Current simulation time in hours.
        """
        for conn in self._connections:
            try:
                self._execute_single_flow(conn, t)
            except Exception as e:
                logger.error(f"Flow execution failed for {conn.source_id}->{conn.target_id}: {e}")

        for conn in self._indexed_connections:
            try:
                self._execute_single_indexed_flow(conn, t)
            except Exception as e:
                logger.error(
                    f"Flow execution failed for "
                    f"{conn.source_name}_{conn.source_index}->{conn.target_name}_{conn.target_index}: {e}"
                )

    def execute_signals(self, t: float) -> None:
        """
        Execute ONLY signal connections (pre-physics pass).

        This method propagates control/demand signals between components
        BEFORE the physics step, ensuring that downstream components
        have current demand information for their step() calculation.
        
        Called from SimulationEngine BEFORE component.step() loop.

        Args:
            t (float): Current simulation time in hours.
        """
        for conn in self._connections:
            if conn.resource_type == 'signal':
                try:
                    self._execute_single_flow(conn, t)
                except Exception as e:
                    logger.error(f"Signal execution failed for {conn.source_id}->{conn.target_id}: {e}")

    def _execute_single_flow(self, conn: ConnectionConfig, t: float) -> None:
        """
        Execute a single legacy connection flow.

        Flow Pattern:
            1. Get output from source port.
            2. Check if flow exists (mass > 0 or energy > 0).
               EXCEPTION: Signals are always transferred regardless of value.
            3. Push to target via receive_input().
            4. Notify source of extraction via extract_output().

        Args:
            conn (ConnectionConfig): Connection configuration.
            t (float): Current simulation time.
        """
        source = self._source_cache[conn.source_id]
        target = self._target_cache[conn.target_id]

        output_value = source.get_output(conn.source_port)

        # Check for valid flow
        # IMPORTANT: Signal connections must ALWAYS be transferred, even with 0 value
        # (0 = "Stop" command from tank control zone C)
        is_signal = conn.resource_type == 'signal'
        
        if isinstance(output_value, Stream):
            # Skip physical streams with no flow, but NEVER skip signals
            if not is_signal and output_value.mass_flow_kg_h <= 0:
                return
        elif isinstance(output_value, (int, float)):
            if not is_signal and output_value <= 0:
                return
        elif output_value is None:
            # No output - skip
            return
        else:
            logger.warning(f"Unknown output type from {conn.source_id}:{conn.source_port}: {type(output_value)}")
            return

        # Push to target
        accepted_amount = target.receive_input(
            port_name=conn.target_port,
            value=output_value,
            resource_type=conn.resource_type
        )
        
        # DEBUG: Log signal transfers
        if is_signal:
            flow_val = output_value.mass_flow_kg_h if isinstance(output_value, Stream) else output_value
            logger.debug(f"SIGNAL TRANSFER: {conn.source_id}:{conn.source_port} -> {conn.target_id}:{conn.target_port} = {flow_val:.0f} kg/h")

        # Notify source of extraction (only for physical flows, not signals)
        if accepted_amount > 0 and not is_signal:
            source.extract_output(
                port_name=conn.source_port,
                amount=accepted_amount,
                resource_type=conn.resource_type
            )

    def _resolve_component_id(self, name: str, index: int) -> str:
        """
        Resolve indexed component name to actual registry ID.

        Resolution order:
            1. Try indexed format (e.g., "chiller_0").
            2. Try singleton name (handles index 0 for non-array components).
            3. Default to indexed format (will fail validation if not found).

        Args:
            name (str): Component base name.
            index (int): Component index.

        Returns:
            str: Resolved component ID.
        """
        indexed_id = f"{name}_{index}"
        if self.registry.has(indexed_id):
            return indexed_id

        if self.registry.has(name):
            return name

        return indexed_id

    def _execute_single_indexed_flow(self, conn: IndexedConnectionConfig, t: float) -> None:
        """
        Execute a single indexed connection flow.

        Same flow pattern as legacy connections but with indexed
        component resolution.

        Args:
            conn (IndexedConnectionConfig): Indexed connection config.
            t (float): Current simulation time.
        """
        source_id = self._resolve_component_id(conn.source_name, conn.source_index)
        target_id = self._resolve_component_id(conn.target_name, conn.target_index)

        source = self._source_cache[source_id]
        target = self._target_cache[target_id]

        output_value = source.get_output(conn.source_port)

        if isinstance(output_value, Stream):
            if output_value.mass_flow_kg_h <= 0:
                return
        elif isinstance(output_value, (int, float)):
            if output_value <= 0:
                return
        else:
            return

        accepted_amount = target.receive_input(
            port_name=conn.target_port,
            value=output_value,
            resource_type=conn.resource_type
        )

        if accepted_amount > 0:
            source.extract_output(
                port_name=conn.source_port,
                amount=accepted_amount,
                resource_type=conn.resource_type
            )

    def _validate_connections(self) -> None:
        """
        Validate that all connections reference existing components and ports.

        Raises:
            ConfigurationError: If source or target component not found.
        """
        for conn in self._connections:
            if not self.registry.has(conn.source_id):
                raise ConfigurationError(f"Source component not found: {conn.source_id}")
            if not self.registry.has(conn.target_id):
                raise ConfigurationError(f"Target component not found: {conn.target_id}")

            self._validate_ports(conn.source_id, conn.source_port, conn.target_id, conn.target_port, conn.resource_type)

        for conn in self._indexed_connections:
            source_id = self._resolve_component_id(conn.source_name, conn.source_index)
            target_id = self._resolve_component_id(conn.target_name, conn.target_index)

            if not self.registry.has(source_id):
                raise ConfigurationError(
                    f"Source component not found: {source_id} (from {conn.source_name} index {conn.source_index})"
                )
            if not self.registry.has(target_id):
                raise ConfigurationError(
                    f"Target component not found: {target_id} (from {conn.target_name} index {conn.target_index})"
                )

            self._validate_ports(source_id, conn.source_port, target_id, conn.target_port, conn.resource_type)

    def _validate_ports(
        self,
        source_id: str,
        source_port: str,
        target_id: str,
        target_port: str,
        resource_type: str
    ) -> None:
        """
        Validate port existence and compatibility between components.

        Args:
            source_id (str): Source component ID.
            source_port (str): Source port name.
            target_id (str): Target component ID.
            target_port (str): Target port name.
            resource_type (str): Expected resource type.

        Raises:
            ConfigurationError: If port direction is incorrect.
        """
        source = self.registry.get(source_id)
        target = self.registry.get(target_id)

        source_ports = source.get_ports()
        target_ports = target.get_ports()

        if source_port not in source_ports:
            logger.warning(f"Source port '{source_port}' not advertised by {source_id}")
        elif source_ports[source_port]['type'] != 'output':
            raise ConfigurationError(f"Port {source_port} on {source_id} is not an output")

        if target_port not in target_ports:
            logger.warning(f"Target port '{target_port}' not advertised by {target_id}")
        elif target_ports[target_port]['type'] != 'input':
            raise ConfigurationError(f"Port {target_port} on {target_id} is not an input")

        # Resource type compatibility check
        if source_port in source_ports and target_port in target_ports:
            src_res = source_ports[source_port].get('resource_type')
            tgt_res = target_ports[target_port].get('resource_type')
            if src_res != tgt_res and src_res is not None and tgt_res is not None:
                if src_res != resource_type:
                    logger.warning(f"Connection resource '{resource_type}' mismatches source port '{src_res}'")
