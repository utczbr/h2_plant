"""
Theme Manager for H2 Plant GUI.

Manages application theme (Dark/Light), including QSS stylesheets,
QPalette for native widgets, and NodeGraphQt specific configurations.

Based on VS Code One Dark Pro theme.
"""

import textwrap
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt


class ThemeManager:
    """
    Manages the application theme (Dark/Light), including QSS stylesheets,
    QPalette for native widgets, and NodeGraphQt specific configurations.
    """
    
    # --- PALETTES ---
    
    # VS Code Dark One Pro (Refined: Less Blue, More Gray)
    DARK_PALETTE = {
        "background": "#282c34",
        "surface":    "#21252b",  # Inputs, Panels
        "foreground": "#abb2bf",  # Text
        "comment":    "#5c6370",  # Disabled / Secondary
        
        # Grays for UI Elements (Replacing excessive Blue)
        "border":     "#3e4451",  # Dividers
        "selection":  "#3e4451",  # Selected items background (Gray)
        "hover":      "#2c313a",  # Hover state (Lighter Gray)
        "button":     "#3b4048",  # Button background (Dark Gray)
        "button_hover": "#4b5263", # Button hover (Lighter Gray)
        
        # Accents (Used Sparingly)
        "blue":       "#61afef",  # Focus rings, Primary highlights
        "red":        "#e06c75",  # Errors
        "green":      "#98c379",  # Success
        "yellow":     "#e5c07b",  # Warnings
    }

    # Grid Colors
    DARK_GRID_COLOR = (60, 60, 60)  # Subtle gray grid on dark bg
    LIGHT_GRID_COLOR = (200, 200, 200) # Visible gray grid on light bg

    # Graph Backgrounds
    LIGHT_GRAPH_BG = (245, 245, 245) # #F5F5F5 - Light Gray for Node Graph
    DARK_GRAPH_BG  = (40, 44, 52)    # #282C34 - Matches Dark Background

    @staticmethod
    def apply_theme(window, app, theme="dark"):
        """
        Applies the specified theme to the window, application, and NodeGraph.
        
        Args:
            window: The MainWindow instance (must have .graph attribute)
            app: The QApplication instance (for QPalette)
            theme: 'dark' or 'light'
        """
        if theme == "dark":
            # 1. Apply QSS
            app.setStyleSheet(ThemeManager.get_dark_stylesheet())
            
            # 2. Apply QPalette (For deep integration: dialogs, title bars)
            app.setPalette(ThemeManager.get_dark_palette())
            
            # 3. Configure Node Graph (Dark)
            if hasattr(window, 'graph'):
                window.graph.set_background_color(*ThemeManager.DARK_GRAPH_BG)
                window.graph.set_grid_mode(1) # 0=None, 1=Dots, 2=Lines
                if hasattr(window.graph, 'set_grid_color'):
                    window.graph.set_grid_color(*ThemeManager.DARK_GRID_COLOR)
                
        else:
            # 1. Clear QSS (Revert to System defaults)
            app.setStyleSheet("") 
            
            # 2. Reset QPalette (Revert to System defaults)
            app.setPalette(QPalette())
            
            # 3. Configure Node Graph (Light - Custom Background)
            if hasattr(window, 'graph'):
                window.graph.set_background_color(*ThemeManager.LIGHT_GRAPH_BG)
                window.graph.set_grid_mode(1)
                if hasattr(window.graph, 'set_grid_color'):
                    window.graph.set_grid_color(*ThemeManager.LIGHT_GRID_COLOR)

    @staticmethod
    def get_dark_palette():
        """Creates a QPalette based on the Dark theme colors."""
        p = ThemeManager.DARK_PALETTE
        palette = QPalette()
        
        # Map custom colors to QPalette roles
        palette.setColor(QPalette.Window, QColor(p['background']))
        palette.setColor(QPalette.WindowText, QColor(p['foreground']))
        palette.setColor(QPalette.Base, QColor(p['surface']))
        palette.setColor(QPalette.AlternateBase, QColor(p['background']))
        palette.setColor(QPalette.ToolTipBase, QColor(p['surface']))
        palette.setColor(QPalette.ToolTipText, QColor(p['foreground']))
        palette.setColor(QPalette.Text, QColor(p['foreground']))
        
        # Button
        palette.setColor(QPalette.Button, QColor(p['button']))
        palette.setColor(QPalette.ButtonText, QColor(p['foreground']))
        
        # Link
        palette.setColor(QPalette.Link, QColor(p['blue']))
        palette.setColor(QPalette.Highlight, QColor(p['selection']))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # Disabled
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(p['comment']))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(p['comment']))
        
        return palette

    @staticmethod
    def get_dark_stylesheet():
        p = ThemeManager.DARK_PALETTE
        
        return textwrap.dedent(f"""
            /* --- GLOBAL --- */
            QMainWindow, QDialog, QMessageBox, QWidget {{
                background-color: {p['background']};
                color: {p['foreground']};
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }}

            /* --- MENUS --- */
            QMenuBar {{
                background-color: {p['background']};
                border-bottom: 1px solid {p['border']};
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 4px 8px;
            }}
            QMenuBar::item:selected {{
                background-color: {p['selection']}; /* Gray selection */
            }}
            QMenu {{
                background-color: {p['surface']};
                border: 1px solid {p['border']};
            }}
            QMenu::item {{
                padding: 4px 24px 4px 8px;
            }}
            QMenu::item:selected {{
                background-color: {p['selection']}; /* Gray selection */
                color: {p['foreground']}; 
            }}
            QMenu::separator {{
                background-color: {p['border']};
                height: 1px;
                margin: 4px 0px;
            }}

            /* --- DOCK WIDGETS --- */
            QDockWidget::title {{
                background-color: {p['surface']};
                text-align: left;
                padding: 6px;
                border-bottom: 1px solid {p['border']};
            }}
            QDockWidget::close-button, QDockWidget::float-button {{
                background: transparent;
                icon-size: 14px;
                border: none;
            }}
            QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
                background: {p['hover']};
            }}

            /* --- BUTTONS (Gray Style) --- */
            QPushButton {{
                background-color: {p['button']};     /* Dark Gray */
                color: {p['foreground']};
                border: 1px solid {p['border']};
                border-radius: 3px;
                padding: 5px 12px;
            }}
            QPushButton:hover {{
                background-color: {p['button_hover']}; /* Lighter Gray */
                border-color: {p['comment']};
            }}
            QPushButton:pressed {{
                background-color: {p['selection']};
            }}
            QPushButton:disabled {{
                background-color: {p['surface']};
                color: {p['comment']};
                border-color: {p['border']};
            }}
            
            /* --- INPUTS --- */
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {p['surface']};
                color: {p['foreground']};
                border: 1px solid {p['border']};
                border-radius: 3px;
                padding: 4px;
                selection-background-color: {p['selection']};
            }}
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {{
                border: 1px solid {p['blue']}; /* Blue only for Focus */
            }}

            /* --- COMBO BOX --- */
            QComboBox {{
                background-color: {p['surface']};
                border: 1px solid {p['border']};
                padding: 4px;
                border-radius: 3px;
            }}
            QComboBox:on {{ 
                background-color: {p['selection']};
            }}
            QComboBox::drop-down {{
                border: 0px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {p['surface']};
                selection-background-color: {p['selection']}; /* Gray Selection */
                border: 1px solid {p['border']};
            }}

            /* --- LISTS & TABLES --- */
            QListWidget, QTreeWidget, QTableWidget {{
                background-color: {p['surface']};
                border: 1px solid {p['border']};
                gridline-color: {p['border']};
                outline: none;
            }}
            QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
                background-color: {p['selection']}; /* Gray Selection */
                color: white;
                border-left: 2px solid {p['blue']}; /* Subtle Blue indicator */
            }}
            QListWidget::item:hover, QTreeWidget::item:hover, QTableWidget::item:hover {{
                background-color: {p['hover']};
            }}
            QHeaderView::section {{
                background-color: {p['background']};
                color: {p['foreground']};
                padding: 4px;
                border: 1px solid {p['border']};
            }}
            QTableCornerButton::section {{
                background-color: {p['background']};
                border: 1px solid {p['border']};
            }}

            /* --- TABS --- */
            QTabWidget::pane {{
                border: 1px solid {p['border']};
            }}
            QTabBar::tab {{
                background-color: {p['background']};
                color: {p['comment']};
                padding: 6px 12px;
                border: 1px solid transparent;
            }}
            QTabBar::tab:selected {{
                background-color: {p['surface']};
                color: {p['foreground']};
                border-top: 2px solid {p['blue']}; /* Subtle Blue Line */
            }}
            QTabBar::tab:hover {{
                background-color: {p['hover']};
            }}

            /* --- SCROLL BARS --- */
            QScrollBar:vertical {{
                background: {p['background']};
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {p['border']};
                min-height: 20px;
                border-radius: 6px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {p['comment']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background: {p['background']};
                height: 12px;
            }}
            QScrollBar::handle:horizontal {{
                background: {p['border']};
                min-width: 20px;
                border-radius: 6px;
                margin: 2px;
            }}

            /* --- SLIDERS --- */
            QSlider::groove:horizontal {{
                border: 1px solid {p['border']};
                height: 4px;
                background: {p['surface']};
                margin: 2px 0;
            }}
            QSlider::handle:horizontal {{
                background: {p['comment']}; 
                border: 1px solid {p['background']};
                width: 14px;
                height: 14px;
                margin: -6px 0;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {p['foreground']};
            }}

            /* --- MISC --- */
            QProgressBar {{
                border: 1px solid {p['border']};
                border-radius: 3px;
                text-align: center;
                background-color: {p['surface']};
            }}
            QProgressBar::chunk {{
                background-color: {p['blue']}; 
            }}
            QToolTip {{
                background-color: {p['background']};
                color: {p['foreground']};
                border: 1px solid {p['border']};
            }}
            QStatusBar {{
                background-color: {p['background']};
                color: {p['comment']};
                border-top: 1px solid {p['border']};
            }}
            QSplitter::handle {{
                background-color: {p['border']};
            }}
        """)
