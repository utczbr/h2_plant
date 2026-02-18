"""
Regression check for report tab selection API usage.
"""

from pathlib import Path


def test_run_simulation_uses_report_widget_selection():
    source = Path("h2_plant/gui/ui/main_window.py").read_text(encoding="utf-8")
    assert "self.central_tabs.setCurrentWidget(self.report_widget)" in source
