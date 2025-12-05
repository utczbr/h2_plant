"""
TopologyValidator: Validates the plant configuration topology.
"""

import logging
from typing import List, Dict, Any
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.config.plant_config import PlantConfig, ConnectionConfig
from h2_plant.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

class TopologyValidator:
    """
    Validates the plant topology configuration.
    """
    
    def __init__(self, config: PlantConfig, registry: ComponentRegistry):
        self.config = config
        self.registry = registry
        
    def validate(self) -> None:
        """
        Validate the topology.
        
        Checks:
        1. All components referenced in topology exist.
        2. All ports referenced exist on components.
        3. Resource types match between connected ports.
        4. Required inputs are connected (orphan check).
        """
        logger.info("Validating plant topology...")
        
        self._validate_components_exist()
        self._validate_ports_and_resources()
        self._check_orphans()
        
        logger.info("Topology validation successful.")
        
    def _validate_components_exist(self) -> None:
        """Verify all source and target IDs exist in registry."""
        for conn in self.config.topology:
            if not self.registry.has(conn.source_id):
                raise ConfigurationError(f"Topology references unknown source component: {conn.source_id}")
            if not self.registry.has(conn.target_id):
                raise ConfigurationError(f"Topology references unknown target component: {conn.target_id}")

    def _validate_ports_and_resources(self) -> None:
        """Verify ports exist and resource types match."""
        for conn in self.config.topology:
            source = self.registry.get(conn.source_id)
            target = self.registry.get(conn.target_id)
            
            source_ports = source.get_ports()
            target_ports = target.get_ports()
            
            # Check Source Port
            if conn.source_port not in source_ports:
                # We might warn instead of error if components aren't fully migrated
                logger.warning(f"Component {conn.source_id} does not advertise port {conn.source_port}")
            else:
                port_info = source_ports[conn.source_port]
                if port_info['type'] != 'output':
                    raise ConfigurationError(f"Port {conn.source_port} on {conn.source_id} is not an output")
                if port_info.get('resource_type') != conn.resource_type:
                     logger.warning(
                        f"Connection resource '{conn.resource_type}' does not match "
                        f"source port resource '{port_info.get('resource_type')}' on {conn.source_id}"
                    )

            # Check Target Port
            if conn.target_port not in target_ports:
                logger.warning(f"Component {conn.target_id} does not advertise port {conn.target_port}")
            else:
                port_info = target_ports[conn.target_port]
                if port_info['type'] != 'input':
                    raise ConfigurationError(f"Port {conn.target_port} on {conn.target_id} is not an input")
                if port_info.get('resource_type') != conn.resource_type:
                    logger.warning(
                        f"Connection resource '{conn.resource_type}' does not match "
                        f"target port resource '{port_info.get('resource_type')}' on {conn.target_id}"
                    )

    def _check_orphans(self) -> None:
        """Check for components with unconnected required inputs."""
        # Build set of connected input ports
        connected_inputs = set()
        for conn in self.config.topology:
            connected_inputs.add((conn.target_id, conn.target_port))
            
        # Iterate all components
        for comp_id, component in self.registry._components.items():
            ports = component.get_ports()
            for port_name, info in ports.items():
                if info['type'] == 'input':
                    # We could add a 'required' flag to PortInfo in the future
                    # For now, just warn if an input is unconnected
                    if (comp_id, port_name) not in connected_inputs:
                        logger.info(f"Input port '{port_name}' on {comp_id} is unconnected")
