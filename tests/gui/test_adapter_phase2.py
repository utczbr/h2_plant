"""Sanitized-scope smoke tests for GraphToConfigAdapter."""

from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode


def _node(node_id, node_type, name, **props):
    return GraphNode(
        id=node_id,
        type=node_type,
        display_name=name,
        x=0.0,
        y=0.0,
        properties=props,
        ports=[],
    )


def test_adapter_to_config_dict_with_allowed_nodes():
    adapter = GraphToConfigAdapter()

    adapter.add_node(
        _node(
            "pem_1",
            "PEMStackNode",
            "PEM 1",
            rated_power_kw=2500.0,
            efficiency_rated=65.0,
        )
    )
    adapter.add_node(
        _node(
            "soec_1",
            "SOECStackNode",
            "SOEC 1",
            rated_power_kw=1200.0,
            operating_temp_c=780.0,
        )
    )
    adapter.add_node(
        _node(
            "rect_1",
            "RectifierNode",
            "Rectifier 1",
            max_power_kw=3000.0,
            conversion_efficiency=97.0,
        )
    )
    adapter.add_node(_node("ch_1", "ChillerNode", "Chiller 1"))
    adapter.add_node(_node("dc_1", "DryCoolerNode", "DryCooler 1"))

    config = adapter.to_config_dict()

    assert config["production"]["electrolyzer"]["max_power_mw"] == 2.5
    assert config["production"]["soec"]["max_power_nominal_mw"] == 1.2
    assert config["production"]["rectifier"]["rated_power_mw"] == 3.0
    assert config["thermal_components"]["chillers"] == 1
    assert config["thermal_components"]["dry_coolers"] == 1


def test_adapter_to_simulation_context_preserves_scenario_backend_type():
    adapter = GraphToConfigAdapter()

    adapter.add_node(
        _node(
            "fallback_1",
            "ScenarioComponentNode",
            "Fallback",
            component_id="Legacy_1",
            __scenario_backend_type="KnockOutDrum",
            __scenario_params={"residence_time_s": 15.0},
        )
    )

    context = adapter.to_simulation_context()
    node = context.topology.nodes[0]

    assert node.id == "Legacy_1"
    assert node.type == "KnockOutDrum"
    assert node.params["residence_time_s"] == 15.0
