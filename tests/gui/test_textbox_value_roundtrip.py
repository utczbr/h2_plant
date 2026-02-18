import pytest

from h2_plant.gui.core.scenario_param_mapper import backend_to_gui_props, gui_to_backend_overlay


def test_textbox_roundtrip_power_transformer_units_and_types():
    base_backend = {
        "rated_power_mw": 15.25,
        "efficiency": 0.95,
        "system_group": "SOEC",
        "process_step": 8,
    }
    gui_props, _ = backend_to_gui_props(
        backend_type="PowerTransformer",
        backend_params=base_backend,
        available_props={"max_power_kw", "conversion_efficiency", "system_group"},
    )

    textbox_values = {
        "max_power_kw": str(gui_props["max_power_kw"]),
        "conversion_efficiency": str(gui_props["conversion_efficiency"]),
        "system_group": gui_props["system_group"],
    }
    merged = gui_to_backend_overlay(
        backend_type="PowerTransformer",
        gui_props=textbox_values,
        base_backend_params=base_backend,
    )

    assert merged["rated_power_mw"] == pytest.approx(15.25)
    assert merged["efficiency"] == pytest.approx(0.95)
    assert merged["system_group"] == "SOEC"
    assert merged["process_step"] == 8


def test_textbox_roundtrip_chiller_temperature_conversion():
    base_backend = {"target_temp_k": 277.15, "cooling_capacity_kw": 140.0}
    gui_props, _ = backend_to_gui_props(
        backend_type="Chiller",
        backend_params=base_backend,
        available_props={"target_temp_c", "cooling_capacity_kw"},
    )
    assert gui_props["target_temp_c"] == pytest.approx(4.0)

    textbox_values = {
        "target_temp_c": "6.0",
        "cooling_capacity_kw": "140.0",
    }
    merged = gui_to_backend_overlay(
        backend_type="Chiller",
        gui_props=textbox_values,
        base_backend_params=base_backend,
    )
    assert merged["target_temp_k"] == pytest.approx(279.15)
    assert merged["cooling_capacity_kw"] == pytest.approx(140.0)


def test_textbox_roundtrip_valve_pressure_conversion():
    base_backend = {"P_out_pa": 1_500_000.0, "fluid": "O2"}
    gui_props, _ = backend_to_gui_props(
        backend_type="Valve",
        backend_params=base_backend,
        available_props={"outlet_pressure_bar", "fluid_type"},
    )
    assert gui_props["outlet_pressure_bar"] == pytest.approx(15.0)

    textbox_values = {"outlet_pressure_bar": "20", "fluid_type": "O2"}
    merged = gui_to_backend_overlay(
        backend_type="Valve",
        gui_props=textbox_values,
        base_backend_params=base_backend,
    )
    assert merged["P_out_pa"] == pytest.approx(2_000_000.0)
    assert merged["fluid"] == "O2"

