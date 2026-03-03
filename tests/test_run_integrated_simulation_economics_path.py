from types import SimpleNamespace

import h2_plant.run_integrated_simulation as runner


class _FakeRegistry:
    def list_components(self):
        return []


class _FakeLoader:
    def __init__(self, scenarios_dir):
        self._scenarios_dir = scenarios_dir

    def load_context(self):
        return SimpleNamespace(
            topology=SimpleNamespace(
                nodes=[
                    SimpleNamespace(
                        id="Node_A",
                        connections=[SimpleNamespace(target_name="Node_B")],
                    )
                ]
            ),
            simulation=SimpleNamespace(
                duration_hours=24,
            ),
        )


def test_run_with_dispatch_strategy_uses_shared_economics_generation_path(monkeypatch, tmp_path):
    import h2_plant.config.loader as loader_mod
    import h2_plant.reporting.stream_table as stream_table_mod

    captured_kwargs = {}

    def _fake_run_with_dispatch_context(context, **kwargs):
        captured_kwargs.update(kwargs)
        return {"minute": [0.0]}, _FakeRegistry()

    monkeypatch.setattr(loader_mod, "ConfigLoader", _FakeLoader)
    monkeypatch.setattr(runner, "run_with_dispatch_context", _fake_run_with_dispatch_context)
    monkeypatch.setattr(stream_table_mod, "print_stream_summary_table", lambda *args, **kwargs: None)

    output_dir = tmp_path / "simulation_output"
    history = runner.run_with_dispatch_strategy(
        scenarios_dir=str(tmp_path),
        hours=4,
        output_dir=output_dir,
    )

    assert history == {"minute": [0.0]}
    assert captured_kwargs["generate_economics_reports"] is True
    assert captured_kwargs["reports_scenarios_dir"] == str(tmp_path)
    assert captured_kwargs["output_dir"] == output_dir
