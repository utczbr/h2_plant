"""
Source-level regression checks for simulation/cleanup wiring in main window.
"""

from pathlib import Path


def _source() -> str:
    return Path("h2_plant/gui/ui/main_window.py").read_text(encoding="utf-8")


def test_duplicate_selection_uses_supported_property_api():
    source = _source()
    assert "source_props = (" in source
    assert "node.get_properties()" in source
    assert "else node.properties()" in source
    assert "for prop_name, prop_value in dict(source_props).items():" in source
    assert "for prop_name, prop_value in node.properties.items():" not in source


def test_run_simulation_wires_progress_and_cancel_signals():
    source = _source()
    assert (
        'progress = QProgressDialog("Running simulation... (0%)", "Cancel", 0, 100, self)'
        in source
    )
    assert "progress.canceled.connect(on_cancel)" in source
    assert "self.worker.progress.connect(on_progress)" in source
    assert "self.worker.canceled.connect(on_canceled)" in source


def test_close_event_and_worker_stop_helper_present():
    source = _source()
    assert "def closeEvent(self, event):" in source
    assert "self._stop_active_simulation_worker(wait_ms=5000)" in source
    assert "def _stop_active_simulation_worker(self, wait_ms: int = 3000) -> bool:" in source
    assert "self.report_widget.cleanup()" in source
