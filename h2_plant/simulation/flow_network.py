"""
FlowNetwork: Manages dynamic resource flows between components.

Executes the topology defined in configuration, moving mass and energy
between component ports using the standard Component interface.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import ConnectionConfig, IndexedConnectionConfig
from h2_plant.core.stream import Stream
from h2_plant.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class FlowNetwork:
    """
    Manages the network of connections between components.
    
    Responsible for:
    1. Validating the connection topology.
    2. Executing flows at each timestep (pull from source, push to target).
    3. Handling flow constraints and compatibility.
    """
    
    def __init__(self, registry: ComponentRegistry, topology: List[ConnectionConfig], indexed_topology: List[IndexedConnectionConfig] = None):
        """
        Initialize FlowNetwork.
        
        Args:
            registry: ComponentRegistry containing all system components.
            topology: List of connection configurations.
            indexed_topology: List of indexed connection configurations.
        """
        self.registry = registry
        self.topology = topology
        self.indexed_topology = indexed_topology or []
        self._connections: List[ConnectionConfig] = topology
        self._indexed_connections: List[IndexedConnectionConfig] = self.indexed_topology
        
    def initialize(self) -> None:
        """
        Initialize and validate the network.
        
        Verifies that all components and ports exist and are compatible.
        """
        logger.info(f"Initializing FlowNetwork with {len(self._connections)} legacy and {len(self._indexed_connections)} indexed connections")
        self._validate_connections()
        
    def execute_flows(self, t: float) -> None:
        """
        Execute all flows for the current timestep.
        
        Iterates through connections:
        1. Gets output from source component.
        2. Pushes input to target component.
        3. Logs flow details.
        
        Args:
            t: Current simulation time (hours).
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
                logger.error(f"Flow execution failed for {conn.source_name}_{conn.source_index}->{conn.target_name}_{conn.target_index}: {e}")

    def _execute_single_flow(self, conn: ConnectionConfig, t: float) -> None:
        """Execute a single connection flow."""
        source = self.registry.get(conn.source_id)
        target = self.registry.get(conn.target_id)
        
        # 1. Get output from source
        # This might return a Stream (for material) or float (for energy)
        output_value = source.get_output(conn.source_port)
        
        # 2. Check if there is anything to flow
        if isinstance(output_value, Stream):
            if output_value.mass_flow_kg_h <= 0:
                return
        elif isinstance(output_value, (int, float)):
            if output_value <= 0:
                return
        else:
             logger.warning(f"Unknown output type from {conn.source_id}:{conn.source_port}: {type(output_value)}")
             return

        # 3. Push to target
        # Target returns how much it accepted
        accepted_amount = target.receive_input(
            port_name=conn.target_port,
            value=output_value,
            resource_type=conn.resource_type
        )
        
        # 4. Feedback loop: Notify source of extraction
        if accepted_amount > 0:
            source.extract_output(
                port_name=conn.source_port,
                amount=accepted_amount,
                resource_type=conn.resource_type
            )
        
        # logger.debug(f"Flow {conn.source_id}->{conn.target_id}: {accepted_amount} {conn.resource_type}")

        # logger.debug(f"Flow {conn.source_id}->{conn.target_id}: {accepted_amount} {conn.resource_type}")

    def _resolve_component_id(self, name: str, index: int) -> str:
        """Resolve indexed name to actual component ID."""
        # 1. Try indexed name (e.g. "chiller_0")
        indexed_id = f"{name}_{index}"
        if self.registry.has(indexed_id):
            return indexed_id
            
        # 2. Try singleton name (e.g. "ultrapure_water_storage")
        # This handles components that are singletons but referenced with index 0
        if self.registry.has(name):
            return name
            
        # Default to indexed ID (will fail validation if not found)
        return indexed_id

    def _execute_single_indexed_flow(self, conn: IndexedConnectionConfig, t: float) -> None:
        """Execute a single indexed connection flow."""
        source_id = self._resolve_component_id(conn.source_name, conn.source_index)
        target_id = self._resolve_component_id(conn.target_name, conn.target_index)
        
        # Create a temporary ConnectionConfig to reuse logic
        # or just duplicate logic for now to avoid overhead
        source = self.registry.get(source_id)
        target = self.registry.get(target_id)
        
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
        """Validate that all connections are valid."""
        # Validate legacy connections
        for conn in self._connections:
            # Check components exist
            if not self.registry.has(conn.source_id):
                raise ConfigurationError(f"Source component not found: {conn.source_id}")
            if not self.registry.has(conn.target_id):
                raise ConfigurationError(f"Target component not found: {conn.target_id}")
            
            self._validate_ports(conn.source_id, conn.source_port, conn.target_id, conn.target_port, conn.resource_type)

        # Validate indexed connections
        for conn in self._indexed_connections:
            source_id = self._resolve_component_id(conn.source_name, conn.source_index)
            target_id = self._resolve_component_id(conn.target_name, conn.target_index)
            
            if not self.registry.has(source_id):
                raise ConfigurationError(f"Source component not found: {source_id} (from {conn.source_name} index {conn.source_index})")
            if not self.registry.has(target_id):
                raise ConfigurationError(f"Target component not found: {target_id} (from {conn.target_name} index {conn.target_index})")
                
            self._validate_ports(source_id, conn.source_port, target_id, conn.target_port, conn.resource_type)

    def _validate_ports(self, source_id: str, source_port: str, target_id: str, target_port: str, resource_type: str) -> None:
        """Helper to validate ports between two components."""
        source = self.registry.get(source_id)
        target = self.registry.get(target_id)
        
        # Check ports exist (using get_ports metadata)
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
        
        # Check resource compatibility
        if source_port in source_ports and target_port in target_ports:
            src_res = source_ports[source_port].get('resource_type')
            tgt_res = target_ports[target_port].get('resource_type')
            if src_res != tgt_res and src_res is not None and tgt_res is not None:
                 if src_res != resource_type:
                     logger.warning(f"Connection resource '{resource_type}' mismatches source port '{src_res}'")
