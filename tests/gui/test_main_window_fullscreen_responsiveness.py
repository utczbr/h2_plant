"""
Source-level checks for fullscreen coverage fallback and canvas reflow plumbing.
"""

from pathlib import Path


def _source() -> str:
    return Path("h2_plant/gui/ui/main_window.py").read_text(encoding="utf-8")


def test_fullscreen_reflow_state_fields_exist():
    source = _source()
    assert "self._fullscreen_reflow_timer = QTimer(self)" in source
    assert "self._fullscreen_reflow_timer.setSingleShot(True)" in source
    assert "self._fullscreen_reflow_timer.timeout.connect(self._apply_window_state_reflow)" in source


def test_resizable_window_constraints_are_enforced():
    source = _source()
    assert "def _enforce_resizable_window_constraints(self):" in source
    # Must NOT call setWindowFlag — it recreates the native window on X11.
    assert "self.setWindowFlag(Qt.MSWindowsFixedSizeDialogHint, False)" not in source
    assert "self.setMaximumSize(16777215, 16777215)" in source
    assert "self.setMinimumSize(0, 0)" in source
    assert "self._enforce_resizable_window_constraints()" in source


def test_central_widgets_are_hardened_for_expansion():
    source = _source()
    assert "self.central_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)" in source
    assert "self.central_tabs.setMinimumSize(0, 0)" in source
    assert "self.graph.widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)" in source
    assert "self.graph.widget.setMinimumSize(0, 0)" in source


def test_fullscreen_reflow_helpers_and_hooks_exist():
    source = _source()
    assert "def _schedule_window_state_reflow(self, delay_ms=0):" in source
    assert "def _apply_window_state_reflow(self):" in source
    assert "self._schedule_window_state_reflow(80)" in source
    assert "self._schedule_window_state_reflow()" in source
    assert "if self.isFullScreen():" in source


def test_strict_fullscreen_and_graph_viewer_refresh_logic_present():
    source = _source()
    assert "def _is_fullscreen_mode(self) -> bool:" not in source
    assert "self._ensure_fullscreen_coverage_or_fallback()" not in source
    assert "want_fullscreen = (not self.isFullScreen()) if checked is None else bool(checked)" in source
    assert "viewer = self.graph.viewer()" in source
    assert "viewport.updateGeometry()" in source
    assert "viewport.update()" in source


def test_fullscreen_hides_docks_and_restores_on_exit():
    source = _source()
    assert "self._saved_dock_visibility" in source
    assert "dock.hide()" in source
    assert "dock.setVisible(was_visible)" in source


def test_two_pass_reflow_exists():
    source = _source()
    assert "def _apply_reflow_pass(self):" in source
    assert "QTimer.singleShot(150, self._apply_reflow_pass)" in source
    assert "self.setGeometry(geo)" in source
