"""
GUI adapter regression tests for graph -> SimulationContext conversion.
"""

import pytest

from h2_plant.gui.core.graph_adapter import (
    FlowType,
    GraphEdge,
    GraphNode,
    GraphToConfigAdapter,
)


def _make_node(node_id: str, node_type: str, display_name: str, **props) -> GraphNode:
    return GraphNode(
        id=node_id,
        type=node_type,
        display_name=display_name,
        x=0.0,
        y=0.0,
        properties=props,
        ports=[],
    )


def test_to_simulation_context_prefers_component_id_over_runtime_node_id():
    adapter = GraphToConfigAdapter()

    pem = _make_node(
        node_id="0xabc",
        node_type="PEMStackNode",
        display_name="PEM A",
        component_id="PEM_Main",
        rated_power_kw=2500.0,
        efficiency_rated=65.0,
    )
    rectifier = _make_node(
        node_id="0xdef",
        node_type="RectifierNode",
        display_name="Rectifier 1",
        component_id="RECT_A",
        max_power_kw=1000.0,
        conversion_efficiency=95.0,
    )

    adapter.add_node(pem)
    adapter.add_node(rectifier)
    adapter.add_edge(
        GraphEdge(
            source_node_id="0xabc",
            source_port="h2_out",
            target_node_id="0xdef",
            target_port="ac_power_in",
            flow_type=FlowType.HYDROGEN,
        )
    )

    context = adapter.to_simulation_context()
    topo_ids = {node.id for node in context.topology.nodes}

    assert "PEM_Main" in topo_ids
    assert "RECT_A" in topo_ids
    assert "0xabc" not in topo_ids
    assert "0xdef" not in topo_ids

    pem_node = next(node for node in context.topology.nodes if node.id == "PEM_Main")
    assert pem_node.connections[0].target_name == "RECT_A"


def test_to_simulation_context_rejects_duplicate_component_ids():
    adapter = GraphToConfigAdapter()
    adapter.add_node(
        _make_node(
            node_id="0x1",
            node_type="PEMStackNode",
            display_name="PEM A",
            component_id="DUPLICATE_ID",
            rated_power_kw=2500.0,
            efficiency_rated=65.0,
        )
    )
    adapter.add_node(
        _make_node(
            node_id="0x2",
            node_type="SOECStackNode",
            display_name="SOEC B",
            component_id="DUPLICATE_ID",
            rated_power_kw=1000.0,
            operating_temp_c=800.0,
        )
    )

    with pytest.raises(ValueError, match="Duplicate component IDs"):
        adapter.to_simulation_context()
