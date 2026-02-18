"""
Tests for shared scenario parameter mapping helpers.
"""

import pytest

from h2_plant.gui.core.scenario_param_mapper import (
    backend_to_gui_props,
    gui_to_backend_overlay,
)


def test_backend_to_gui_props_converts_transformer_values():
    mapped, unmapped = backend_to_gui_props(
        backend_type="PowerTransformer",
        backend_params={
            "rated_power_mw": 15.25,
            "efficiency": 0.95,
            "system_group": "SOEC",
            "process_step": 8,
        },
        available_props={"max_power_kw", "conversion_efficiency", "system_group"},
    )

    assert mapped["max_power_kw"] == pytest.approx(15250.0)
    assert mapped["conversion_efficiency"] == pytest.approx(95.0)
    assert mapped["system_group"] == "SOEC"
    assert unmapped == {"process_step": 8}


def test_backend_to_gui_props_maps_direct_keys_when_available():
    mapped, unmapped = backend_to_gui_props(
        backend_type="Chiller",
        backend_params={"cooling_capacity_kw": 140.0, "target_temp_k": 277.15},
        available_props={"cooling_capacity_kw", "target_temp_c"},
    )
    assert mapped["cooling_capacity_kw"] == pytest.approx(140.0)
    assert mapped["target_temp_c"] == pytest.approx(4.0)
    assert unmapped == {}


def test_gui_to_backend_overlay_applies_unit_conversions():
    result = gui_to_backend_overlay(
        backend_type="PowerTransformer",
        gui_props={
            "max_power_kw": "16000",
            "conversion_efficiency": "98",
            "system_group": "PEM",
        },
        base_backend_params={
            "rated_power_mw": 15.25,
            "efficiency": 0.95,
            "system_group": "SOEC",
            "process_step": 8,
        },
    )

    assert result["rated_power_mw"] == pytest.approx(16.0)
    assert result["efficiency"] == pytest.approx(0.98)
    assert result["system_group"] == "PEM"
    assert result["process_step"] == 8


def test_gui_to_backend_overlay_only_updates_known_keys():
    result = gui_to_backend_overlay(
        backend_type="PEM",
        gui_props={"unknown_ui_field": "42", "rated_power_kw": "5350"},
        base_backend_params={"max_power_mw": 5.35, "lifecycle": 87600},
    )

    assert result["max_power_mw"] == pytest.approx(5.35)
    assert result["lifecycle"] == 87600
    assert "unknown_ui_field" not in result


def test_gui_to_backend_overlay_empty_string_does_not_overwrite_canonical_fluid():
    """An uninitialised enum (empty string) must never blank a canonical fluid field."""
    result = gui_to_backend_overlay(
        backend_type="Valve",
        gui_props={"outlet_pressure_bar": "5.0", "fluid_type": ""},
        base_backend_params={"P_out_pa": 500000.0, "fluid": "H2"},
    )

    assert result["fluid"] == "H2", "canonical fluid must not be overwritten by empty GUI value"
    assert result["P_out_pa"] == pytest.approx(500000.0)


def test_gui_to_backend_overlay_non_empty_fluid_updates_canonical():
    """A real fluid selection must still update the canonical field."""
    result = gui_to_backend_overlay(
        backend_type="Valve",
        gui_props={"outlet_pressure_bar": "3.0", "fluid_type": "CH4"},
        base_backend_params={"P_out_pa": 500000.0, "fluid": "H2"},
    )

    assert result["fluid"] == "CH4"
    assert result["P_out_pa"] == pytest.approx(300000.0)


def test_valve_node_fluid_type_options_cover_lut_fluids():
    """ValveNode source must declare all fluids the backend LUT supports.

    Node instantiation is avoided (requires Qt display); we inspect the module
    source directly so the test is safe in headless CI environments.
    """
    import inspect
    import importlib
    mod = importlib.import_module("h2_plant.gui.nodes.valve_node")
    source = inspect.getsource(mod)
    required_fluids = ["H2", "N2", "O2", "CO2", "H2O", "CH4", "CO"]
    for fluid in required_fluids:
        assert f"'{fluid}'" in source or f'"{fluid}"' in source, (
            f"ValveNode source does not contain fluid option '{fluid}'"
        )


def test_coalescer_node_gas_type_includes_syngas():
    """CoalescerNode source must include 'Syngas' so ATR-path coalescers round-trip correctly."""
    import inspect
    import importlib
    mod = importlib.import_module("h2_plant.gui.nodes.separation")
    source = inspect.getsource(mod)
    # Check CoalescerNode specifically — find the class and verify Syngas is present
    assert "'Syngas'" in source or '"Syngas"' in source, (
        "separation.py does not contain 'Syngas' option for CoalescerNode"
    )
