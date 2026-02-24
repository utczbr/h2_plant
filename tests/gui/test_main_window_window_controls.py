"""
Source-level regression checks for main window control wiring.
"""

from pathlib import Path


def _source() -> str:
    return Path("h2_plant/gui/ui/main_window.py").read_text(encoding="utf-8")


def test_fullscreen_action_is_present_and_uses_f11():
    source = _source()
    assert 'fullscreen_action = QAction("Full Screen", self)' in source
    assert 'fullscreen_action.setShortcut("F11")' in source
    assert "self._fullscreen_action = fullscreen_action" in source


def test_corner_window_buttons_are_wired():
    source = _source()
    assert "self.minimize_button = QToolButton(window_controls)" in source
    assert "self.minimize_button.clicked.connect(self.showMinimized)" in source
    assert "self.window_mode_button = QToolButton(window_controls)" in source
    assert "lambda _checked=False: self._toggle_fullscreen()" in source
    assert "def _sync_window_mode_button_state(self):" in source
    assert "QStyle.SP_TitleBarNormalButton" in source
    assert "QStyle.SP_TitleBarMaxButton" in source
    assert "menubar.setCornerWidget(window_controls, Qt.TopRightCorner)" in source
    assert "self.close_button = QToolButton(window_controls)" not in source


def test_fullscreen_shortcuts_and_state_sync_helpers_exist():
    source = _source()
    assert "QShortcut(QKeySequence(Qt.Key_F11), self, lambda: self._toggle_fullscreen())" in source
    assert "QShortcut(QKeySequence(Qt.Key_Escape), self, self._handle_escape_pressed)" in source
    assert "def _sync_fullscreen_action_state(self):" in source
    assert "def changeEvent(self, event):" in source
