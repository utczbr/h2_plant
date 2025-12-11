"""
Abstract base class for all node types.

Every specific node (ElectrolyzerNode, TankNode, etc.) inherits from this.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from NodeGraphQt import BaseNode as QtBaseNode
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
from Qt import QtWidgets, QtCore, QtGui

class NodeSpacerWidget(NodeBaseWidget):
    """
    A transparent spacer widget to add vertical padding.
    """
    def __init__(self, parent=None, name='', label=''):
        super(NodeSpacerWidget, self).__init__(parent, name, label)
        self._my_widget = QtWidgets.QLabel('')
        self.set_custom_widget(self._my_widget)
        self._my_widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._my_widget.setStyleSheet("background: transparent; border: none;")
        self._my_widget.setFixedHeight(20) # Default height

    def set_height(self, height):
        self._my_widget.setFixedHeight(height)

    def get_value(self):
        return ""

    def set_value(self, value):
        pass

class CollapseButton(QtWidgets.QGraphicsItem):
    """
    A custom toggle button (arrow) displayed on the node to hide/show properties.
    """
    def __init__(self, parent=None, callback=None, start_collapsed=True):
        super(CollapseButton, self).__init__(parent)
        self._callback = callback
        self._collapsed = start_collapsed
        self._hovered = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        self.setZValue(2.0)  # Ensure it's drawn on top

        # Initial position
        self._update_pos()

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 16, 16)

    def _update_pos(self):
        if self.parentItem():
            rect = self.parentItem().boundingRect()
            self.setPos(rect.width() - 20, rect.height() - 20)

    def paint(self, painter, option, widget):
        painter.save()

        # Update position every paint to stay in corner if node resizes
        self._update_pos()

        # -------------------------------------------------------------------------
        # FIX: Enforce collapsed state to override NodeGraphQt's LOD auto-show behavior
        # -------------------------------------------------------------------------
        # When zooming in, NodeGraphQt detects the LOD change and sets all widgets 
        # to Visible=True. We must detect this and force them back to Hidden if 
        # we are currently in a collapsed state.
        if self._collapsed and self.parentItem() and hasattr(self.parentItem(), 'widgets'):
            for widget_item in self.parentItem().widgets.values():
                # We only want to hide standard widgets. Spacers (if used) 
                # might be intended to be visible in collapsed state.
                if not isinstance(widget_item, NodeSpacerWidget):
                    if widget_item.isVisible():
                        widget_item.setVisible(False)
        # -------------------------------------------------------------------------

        path = QtGui.QPainterPath()
        center_x = 8
        center_y = 8

        if self._collapsed:
            # Arrow points down (Show Properties)
            path.moveTo(center_x - 4, center_y - 2)
            path.lineTo(center_x, center_y + 3)
            path.lineTo(center_x + 4, center_y - 2)
        else:
             # Arrow points up (Hide)
            path.moveTo(center_x - 4, center_y + 2)
            path.lineTo(center_x, center_y - 3)
            path.lineTo(center_x + 4, center_y + 2)

        color = QtGui.QColor(200, 200, 200)
        if self._hovered:
            color = QtGui.QColor(255, 255, 255)

        painter.setPen(QtGui.QPen(color, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        painter.drawPath(path)

        painter.restore()

    def hoverEnterEvent(self, event):
        self._hovered = True
        if self._collapsed:
            self.setToolTip("Show Properties")
        else:
            self.setToolTip("Hide")
        self.update()
        super(CollapseButton, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super(CollapseButton, self).hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._collapsed = not self._collapsed
            if self._callback:
                self._callback(self._collapsed)
            self.update()


class ConfigurableNode(QtBaseNode):
    """
    Base class for all plant component nodes.
    
    Subclasses must define:
    - __identifier__: Unique identifier (e.g., 'h2_plant.production.electrolyzer')
    - NODE_NAME: Display name (e.g., 'Electrolyzer')
    - _init_ports(): Define input/output ports
    - _init_properties(): Define parameter fields
    """
    
    # Subclasses must override these
    __identifier__ = 'h2_plant.base'
    NODE_NAME = 'BaseNode'
    ICON_PATH = None  # Optional icon
    MIN_COLLAPSED_HEIGHT = 80
    
    PORT_COLORS = {
        "hydrogen": (0, 255, 255),      # Cyan
        "oxygen": (255, 200, 0),        # Orange
        "electricity": (255, 255, 0),   # Yellow
        "heat": (255, 100, 100),        # Red
        "water": (100, 150, 255),       # Blue
        "compressed_h2": (0, 200, 255), # Light cyan
        "gas": (200, 200, 200),         # Grey (Natural Gas / CO2)
    }
    
    def __init__(self):
        super().__init__()
        # self.set_name(self.NODE_NAME) # NodeGraphQt sets name from NODE_NAME automatically
        
        # Track property changes
        self._property_validators: Dict[str, Callable] = {}
        self._collapse_btn = None
        self._should_enable_collapse = False
        self._pending_spacers = []
        
        # Initialize ports and properties
        self._init_ports()
        self._init_properties()
        
        # Schedule post-creation hook
        QtCore.QTimer.singleShot(0, self.post_node_created)

    def post_node_created(self):
        """Hook called after NodeGraphQt fully creates the node view."""
        # Process pending spacers
        for name, height, tab in self._pending_spacers:
            self.add_spacer(name, height, tab)
        self._pending_spacers = []

        if self._should_enable_collapse:
            self.enable_collapse()

    def enable_collapse(self, start_collapsed=True):
        """
        Enables the collapse/expand functionality for this node.
        Safe to call anytime - defers if view is not ready.
        
        Args:
            start_collapsed: If True, node starts in collapsed state (default)
        """
        if not self.view:
            self._should_enable_collapse = True
            return
            
        if self.view:
            self._collapse_btn = CollapseButton(parent=self.view, callback=self._on_collapse_toggled, start_collapsed=start_collapsed)
            # Apply initial collapsed state to widgets
            self._on_collapse_toggled(start_collapsed)

    def add_spacer(self, name: str, height: int = 20, tab: str = None) -> None:
        """
        Add a spacer widget that is visible ONLY when collapsed.
        """
        if tab is None:
            tab = self.NODE_NAME

        if not self.view:
            self._pending_spacers.append((name, height, tab))
            return

        # We use create_property just to register it, but we don't need it in the properties bin really
        # NodeGraphQt requires create_property for widgets usually
        self.create_property(name, value="", widget_type=0, tab=tab) # 0 = HIDDEN
        
        widget = NodeSpacerWidget(self.view, name)
        widget.set_height(height)
        
        # NodeGraphQt's add_widget handles registration in view.widgets
        self.view.add_widget(widget)
        
        # Initially visible (since nodes start collapsed by default)
        widget.setVisible(True)

    def _on_collapse_toggled(self, collapsed):
        """
        Callback for when the collapse button is clicked.
        """
        view = self.view

        # Iterate over all properties to hide/show their widgets
        if hasattr(view, 'widgets'):
            for name, widget_item in view.widgets.items():
                if isinstance(widget_item, NodeSpacerWidget):
                    # Spacers are visible ONLY when collapsed
                    widget_item.setVisible(collapsed)
                else:
                    # Normal widgets are visible ONLY when expanded
                    widget_item.setVisible(not collapsed)

        # Force the node to redraw and recalculate size
        view.draw_node()

        # Trigger update on the button to ensure it moves
        if self._collapse_btn:
            self._collapse_btn.update()
    
    def _init_ports(self) -> None:
        """
        Define input/output ports.
        Override in subclasses.
        """
        pass
    
    def _init_properties(self) -> None:
        """
        Define configurable properties.
        Override in subclasses.
        """
        pass
    
    def add_input(self, 
                  name: str, 
                  flow_type: str = 'default',
                  multi_input: bool = False,
                  display_name: bool = True,
                  color: Tuple[int, int, int] = None,
                  locked: bool = False,
                  painter_func: Callable = None):
        """Add an input port."""
        if color is None:
            color = self.PORT_COLORS.get(flow_type, (128, 128, 128))
        
        super().add_input(name, multi_input, display_name, color, locked, painter_func)
        
        # Store metadata (NodeGraphQt doesn't support custom port data natively easily, 
        # so we might need to track it separately if needed, but for now color is enough visual cue)
        # We can access port by name later.
    
    def add_output(self, 
                   name: str, 
                   flow_type: str = 'default',
                   multi_output: bool = True,
                   display_name: bool = True,
                   color: Tuple[int, int, int] = None,
                   locked: bool = False,
                   painter_func: Callable = None):
        """Add an output port."""
        if color is None:
            color = self.PORT_COLORS.get(flow_type, (128, 128, 128))
            
        super().add_output(name, multi_output, display_name, color, locked, painter_func)

    # ===== Enhanced Property Methods with Tab Support and Units =====
    
    def add_float_property(self,
                          name: str,
                          default: float = 0.0,
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          unit: str = "",
                          tab: str = None) -> None:
        """
        Add a float property with validation, units, and tab organization.
        """
        if tab is None:
            tab = self.NODE_NAME
        
        # Create display label with unit
        label = f"{name} ({unit})" if unit else name
        
        # Call NodeGraphQt's add_text_input to create the widget
        # Note: NodeGraphQt doesn't have add_float_input, so we use text input
        super().add_text_input(name, label=label, text=str(default), tab=tab)
        
        # Store constraints for validation
        self._property_validators[name] = (
            lambda v: self._validate_float(v, min_val, max_val)
        )
    
    def add_float_input(self, 
                        name: str, 
                        default: float = 0.0,
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None) -> None:
        """Add a float property with optional constraints (legacy method)."""
        # Adapt legacy 'default' arg to NodeGraphQt 'value' arg
        self.create_property(name, default)
        
        # Store constraints for validation
        self._property_validators[name] = (
            lambda v: self._validate_float(v, min_val, max_val)
        )
    
    def add_integer_input(self, 
                         name: str, 
                         default: int = 0,
                         min_val: Optional[int] = None,
                         max_val: Optional[int] = None) -> None:
        """Add an integer property (legacy method)."""
        self.create_property(name, default)
        self._property_validators[name] = (
            lambda v: self._validate_integer(v, min_val, max_val)
        )
    
    def add_text_input(self, name: str, default: str = "") -> None:
        """Add a text property (legacy method)."""
        self.create_property(name, default)

    def add_enum_input(self, 
                      name: str, 
                      options: List[str],
                      default_index: int = 0) -> None:
        """Add an enum (dropdown) property (legacy method)."""
        default_val = options[default_index] if options else ""
        self.create_property(name, default_val, items=options)
        
    def add_integer_property(self,
                            name: str,
                            default: int = 0,
                            min_val: Optional[int] = None,
                            max_val: Optional[int] = None,
                            unit: str = "",
                            tab: str = None) -> None:
        """
        Add an integer property with validation, units, and tab organization.
        """
        if tab is None:
            tab = self.NODE_NAME
        
        label = f"{name} ({unit})" if unit else name
        
        # Call NodeGraphQt's add_text_input (no add_int_input available)
        super().add_text_input(name, label=label, text=str(default), tab=tab)
        
        self._property_validators[name] = (
            lambda v: self._validate_integer(v, min_val, max_val)
        )
    
    def add_percentage_property(self,
                               name: str,
                               default: float = 0.0,
                               min_val: float = 0.0,
                               max_val: float = 100.0,
                               tab: str = None) -> None:
        """
        Add a percentage property (0-100%) with validation.
        """
        self.add_float_property(name, default, min_val, max_val, unit="%", tab=tab)
    
    def add_text_property(self,
                         name: str,
                         default: str = "",
                         tab: str = None) -> None:
        """
        Add a text property with tab organization.
        """
        if tab is None:
            tab = self.NODE_NAME
        
        super().add_text_input(name, label=name, text=default, tab=tab)
    
    def add_enum_property(self,
                         name: str,
                         options: List[str],
                         default_index: int = 0,
                         tab: str = None) -> None:
        """
        Add an enum (dropdown) property with tab organization.
        """
        if tab is None:
            tab = self.NODE_NAME
        
        # NodeGraphQt uses add_combo_menu
        self.add_combo_menu(name, label=name, items=options, tab=tab)
        # Set default value
        if options:
            self.set_property(name, options[default_index])

    def add_color_property(self,
                          name: str,
                          default: Tuple[int, int, int] = (0, 0, 0),
                          tab: str = 'Custom') -> None:
        """
        Add a color picker property.
        """
        # NodePropWidgetEnum.COLOR_PICKER = 9
        self.create_property(name, default, widget_type=9, tab=tab)
    
    def get_properties(self) -> Dict[str, Any]:
        """Export node properties as a dict."""
        # NodeGraphQt stores properties in self.properties()
        # We exclude internal properties like 'id', 'name', 'color', 'border_color', etc if needed
        # But for now, let's just return custom properties.
        
        # We only want properties we explicitly added
        props = {}
        for name in self._property_validators.keys():
            props[name] = self.get_property(name)
        
        # Also include any text inputs or others we added without validators
        # A better way might be to track added properties
        # For now, let's just iterate all and filter standard ones
        
        all_props = self.properties()
        standard_props = {'id', 'name', 'color', 'border_color', 'text_color', 'type', 'selected', 'disabled'}
        
        for k, v in all_props.items():
            if k not in standard_props:
                props[k] = v
                
        return props
    
    @staticmethod
    def _validate_float(value: Any, 
                       min_val: Optional[float], 
                       max_val: Optional[float]) -> Tuple[bool, Optional[str]]:
        try:
            fval = float(value)
            if min_val is not None and fval < min_val:
                return False, f"Must be >= {min_val}"
            if max_val is not None and fval > max_val:
                return False, f"Must be <= {max_val}"
            return True, None
        except (ValueError, TypeError):
            return False, "Must be a number"
    
    @staticmethod
    def _validate_integer(value: Any, 
                         min_val: Optional[int], 
                         max_val: Optional[int]) -> Tuple[bool, Optional[str]]:
        try:
            ival = int(value)
            if min_val is not None and ival < min_val:
                return False, f"Must be >= {min_val}"
            if max_val is not None and ival > max_val:
                return False, f"Must be <= {max_val}"
            return True, None
        except (ValueError, TypeError):
            return False, "Must be an integer"
