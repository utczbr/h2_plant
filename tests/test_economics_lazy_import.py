import importlib
import sys


def test_economics_module_uses_lazy_exports():
    for key in list(sys.modules):
        if key == "h2_plant.economics" or key.startswith("h2_plant.economics."):
            sys.modules.pop(key, None)

    econ = importlib.import_module("h2_plant.economics")

    assert "h2_plant.economics.capex_generator" not in sys.modules
    assert "h2_plant.economics.lcoh_calculator" not in sys.modules

    _ = econ.LcohCalculator
    assert "h2_plant.economics.lcoh_calculator" in sys.modules
    assert "h2_plant.economics.capex_generator" not in sys.modules

    _ = econ.CapexGenerator
    assert "h2_plant.economics.capex_generator" in sys.modules
