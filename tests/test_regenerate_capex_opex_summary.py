import types

import h2_plant.economics as economics_pkg
import h2_plant.economics.opex_generator as opex_module
import tools.regenerate_capex as regen_capex


def _make_fake_capex_report():
    return types.SimpleNamespace(
        overall_cost_class=types.SimpleNamespace(value="Class 3"),
        entries_with_cost=1,
        entries=[types.SimpleNamespace(errors=[], warnings=[])],
        entries_out_of_bounds=0,
        total_C_BM=1_000.0,
        total_C_BM_low=800.0,
        total_C_BM_high=1_200.0,
        block_summaries=[],
        total_installation=0.0,
        total_installation_low=0.0,
        total_installation_high=0.0,
        total_installed_cost=1_000.0,
        total_installed_cost_low=800.0,
        total_installed_cost_high=1_200.0,
    )


def _prepare_scenario_tree(tmp_path):
    scenarios_dir = tmp_path / "scenarios"
    economics_dir = scenarios_dir / "Economics"
    economics_dir.mkdir(parents=True)
    (economics_dir / "equipment_mappings.yaml").write_text("mappings: []\n", encoding="utf-8")
    (economics_dir / "opex_config.yaml").write_text("opex_items: []\n", encoding="utf-8")
    output_dir = tmp_path / "simulation_output"
    output_dir.mkdir()
    return scenarios_dir, output_dir


def _install_fake_generators(monkeypatch, opex_report):
    fake_capex_report = _make_fake_capex_report()

    class FakeCapexGenerator:
        def __init__(self):
            self.capacity_mode = "design"

        @classmethod
        def from_yaml(cls, _path):
            return cls()

        def generate(self, **_kwargs):
            return fake_capex_report

    class FakeOpexGenerator:
        def generate(self, **_kwargs):
            return opex_report

        def generate_streaming(self, **_kwargs):
            return opex_report

        def generate_streaming_parquet(self, **_kwargs):
            return opex_report

    monkeypatch.setattr(economics_pkg, "CapexGenerator", FakeCapexGenerator)
    monkeypatch.setattr(opex_module, "OpexGenerator", FakeOpexGenerator)
    monkeypatch.setattr(regen_capex, "_build_registry", lambda _cfg_dir: (object(), object()))


def test_regenerate_capex_prints_opex_low_high_when_available(tmp_path, monkeypatch, capsys):
    scenarios_dir, output_dir = _prepare_scenario_tree(tmp_path)
    opex_report = types.SimpleNamespace(
        scenario_name="test",
        simulation_hours=8760.0,
        total_opex=1_000.0,
        total_opex_low=800.0,
        total_opex_high=1_200.0,
        total_variable_cost=300.0,
        total_fixed_cost=400.0,
        total_maintenance_cost=300.0,
        annual_h2_production_kg=0.0,
        opex_per_kg_h2=0.0,
    )
    _install_fake_generators(monkeypatch, opex_report)

    rc = regen_capex.regenerate_capex(
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
        config_dir=scenarios_dir,
        generate_opex=True,
    )
    captured = capsys.readouterr().out

    assert rc == 0
    assert "OPEX Low:" in captured
    assert "OPEX High:" in captured


def test_regenerate_capex_omits_opex_low_high_when_missing(tmp_path, monkeypatch, capsys):
    scenarios_dir, output_dir = _prepare_scenario_tree(tmp_path)
    opex_report = types.SimpleNamespace(
        scenario_name="test",
        simulation_hours=8760.0,
        total_opex=1_000.0,
        total_opex_low=None,
        total_opex_high=None,
        total_variable_cost=300.0,
        total_fixed_cost=400.0,
        total_maintenance_cost=300.0,
        annual_h2_production_kg=0.0,
        opex_per_kg_h2=0.0,
    )
    _install_fake_generators(monkeypatch, opex_report)

    rc = regen_capex.regenerate_capex(
        scenarios_dir=scenarios_dir,
        output_dir=output_dir,
        config_dir=scenarios_dir,
        generate_opex=True,
    )
    captured = capsys.readouterr().out

    assert rc == 0
    assert "OPEX Low:" not in captured
    assert "OPEX High:" not in captured
