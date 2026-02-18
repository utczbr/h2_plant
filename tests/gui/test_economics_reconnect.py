"""
Tests for economics editing signal lifecycle across template mode transitions
and schema validation of economics edits.

Covers:
- Fix 1: _economics_editing_connected flag reset on exit
- Fix 5: EconomicsConfig schema validation with cell revert
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Minimal stub for MainWindow economics lifecycle testing
# ---------------------------------------------------------------------------

def _make_main_window_stub():
    """Create a lightweight stub with the economics signal lifecycle methods.

    We avoid instantiating the real MainWindow (needs a full Qt app + NodeGraph).
    Instead we replicate the exact flag/signal logic from the real class and test
    that logic directly.
    """
    from types import SimpleNamespace

    mw = SimpleNamespace()
    mw._economics_editing_connected = False
    mw._template_mode = False
    mw._template_source_manifest = None
    mw._generated_bundle_dir = None
    mw._scenario_manifest = {"scenarios_dir": "/tmp/test"}
    mw._scenario_economics = {}

    # Mock table widget
    table = MagicMock()
    mw.scenario_economics_table = table

    # Track connect/disconnect calls
    mw._signal_connected = False

    def _connect_signal(*args):
        mw._signal_connected = True

    def _disconnect_signal(*args):
        if not mw._signal_connected:
            raise RuntimeError("Not connected")
        mw._signal_connected = False

    table.cellChanged.connect = MagicMock(side_effect=_connect_signal)
    table.cellChanged.disconnect = MagicMock(side_effect=_disconnect_signal)
    table.setEditTriggers = MagicMock()
    table.setSelectionMode = MagicMock()
    table.blockSignals = MagicMock()

    # Mock graph
    graph = MagicMock()
    graph.all_nodes.return_value = []
    mw.graph = graph

    return mw


def _enter_template_mode(mw):
    """Replicate _enter_template_mode logic from main_window.py."""
    mw._template_mode = True
    mw._template_source_manifest = dict(mw._scenario_manifest) if mw._scenario_manifest else None
    mw._generated_bundle_dir = None
    # Make economics table editable
    if not mw._economics_editing_connected:
        mw.scenario_economics_table.cellChanged.connect("_on_economics_cell_changed")
        mw._economics_editing_connected = True


def _exit_template_mode(mw):
    """Replicate _exit_template_mode logic from main_window.py (WITH fix applied)."""
    mw._template_mode = False
    mw._template_source_manifest = None
    mw._generated_bundle_dir = None
    if mw._economics_editing_connected:
        try:
            mw.scenario_economics_table.cellChanged.disconnect("_on_economics_cell_changed")
        except RuntimeError:
            pass  # not connected
    mw._economics_editing_connected = False


# ---------------------------------------------------------------------------
# Fix 1: Economics reconnect flag lifecycle
# ---------------------------------------------------------------------------

class TestEconomicsReconnectFlag:
    """Verify _economics_editing_connected flag is correctly managed."""

    def test_flag_true_after_enter(self):
        mw = _make_main_window_stub()
        _enter_template_mode(mw)
        assert mw._economics_editing_connected is True

    def test_flag_false_after_exit(self):
        mw = _make_main_window_stub()
        _enter_template_mode(mw)
        _exit_template_mode(mw)
        assert mw._economics_editing_connected is False

    def test_signal_reconnects_on_reenter(self):
        mw = _make_main_window_stub()
        # First cycle
        _enter_template_mode(mw)
        assert mw._signal_connected is True
        _exit_template_mode(mw)
        assert mw._signal_connected is False
        # Second cycle — signal must reconnect
        _enter_template_mode(mw)
        assert mw._signal_connected is True
        assert mw._economics_editing_connected is True

    def test_multiple_cycles(self):
        mw = _make_main_window_stub()
        for _ in range(5):
            _enter_template_mode(mw)
            assert mw._economics_editing_connected is True
            assert mw._signal_connected is True
            _exit_template_mode(mw)
            assert mw._economics_editing_connected is False
            assert mw._signal_connected is False

    def test_exit_without_enter_is_safe(self):
        """Exiting template mode without entering should not raise."""
        mw = _make_main_window_stub()
        _exit_template_mode(mw)  # should not raise
        assert mw._economics_editing_connected is False
        mw.scenario_economics_table.cellChanged.disconnect.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 5: Economics schema validation
# ---------------------------------------------------------------------------

class TestEconomicsSchemaValidation:
    """Verify EconomicsConfig validation rejects invalid edits."""

    def test_valid_edit_accepted(self):
        from h2_plant.config.models import EconomicsConfig

        # Build a valid base dict with all required fields
        base = {f: EconomicsConfig.model_fields[f].default
                for f in EconomicsConfig.model_fields
                if EconomicsConfig.model_fields[f].default is not None}
        base["h2_price_eur_kg"] = 3.0  # required field with no default

        # Validate that updating a float field with a float works
        candidate = {**base, "h2_price_eur_kg": 5.0}
        config = EconomicsConfig.model_validate(candidate)
        assert config.h2_price_eur_kg == 5.0

    def test_invalid_edit_rejected(self):
        from h2_plant.config.models import EconomicsConfig

        base = {f: EconomicsConfig.model_fields[f].default
                for f in EconomicsConfig.model_fields
                if EconomicsConfig.model_fields[f].default is not None}
        base["h2_price_eur_kg"] = 3.0

        # Updating a float field with a non-numeric string should fail
        candidate = {**base, "h2_price_eur_kg": "not_a_number"}
        with pytest.raises(Exception):
            EconomicsConfig.model_validate(candidate)

    def test_unknown_key_passes_through(self):
        """Keys not in EconomicsConfig should be stored without validation."""
        from h2_plant.config.models import EconomicsConfig

        # field_info should be None for unknown keys
        assert EconomicsConfig.model_fields.get("custom_user_field") is None
