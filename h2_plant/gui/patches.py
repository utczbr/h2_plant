
"""
Runtime patches for third-party libraries.
"""
import logging
from NodeGraphQt.qgraphics.node_base import NodeItem

def apply_patches():
    """
    Apply runtime patches to fix known issues in dependencies.
    """
    _patch_nodegraphqt_paint_error()

def _patch_nodegraphqt_paint_error():
    """
    Fixes AttributeError in NodeItem.auto_switch_mode when viewer is None.
    See: https://github.com/jchanvfx/NodeGraphQt/issues/ (or similar tracking if exists)
    """
    original_auto_switch_mode = NodeItem.auto_switch_mode

    def fixed_auto_switch_mode(self):
        # PATCH: Check if viewer exists before using it
        viewer = self.viewer()
        if not viewer:
            return

        # Original logic (copied from library source, ensuring we use our local 'viewer' var)
        rect = self.sceneBoundingRect()
        l = viewer.mapToGlobal(
            viewer.mapFromScene(rect.topLeft()))
        r = viewer.mapToGlobal(
            viewer.mapFromScene(rect.topRight()))
        # width is the node width in screen
        width = r.x() - l.x()

        self.set_proxy_mode(width < self._proxy_mode_threshold)

    # Apply the patch
    NodeItem.auto_switch_mode = fixed_auto_switch_mode
    logging.info("Patched NodeGraphQt.NodeItem.auto_switch_mode to handle None viewer.")
