"""
One Dark Pro Theme for PySide6

This module provides a complete One Dark Pro (VS Code) color theme for the application.
Colors based on https://github.com/Binaryify/OneDark-Pro/

Color Palette:
- Background: #282c34
- Foreground: #abb2bf
- Comment: #5c6370
- Red: #e06c75
- Orange: #d19a66
- Yellow: #e5c07b
- Green: #98c379
- Cyan: #56b6c2
- Blue: #61afef
- Purple: #c678dd
"""

# One Dark Pro Color Palette
COLORS = {
    # Base Colors
    "background": "#282c34",      # Main background
    "surface": "#21252b",         # Slightly darker for contrast
    "foreground": "#abb2bf",      # Main text color
    "comment": "#5c6370",         # Comments/disabled text
    
    # Accent Colors
    "red": "#e06c75",             # Errors, warnings
    "orange": "#d19a66",          # Warnings, secondary accent
    "yellow": "#e5c07b",          # Info, highlights
    "green": "#98c379",           # Success, valid
    "cyan": "#56b6c2",            # Secondary accent
    "blue": "#61afef",            # Primary accent
    "purple": "#c678dd",          # Secondary highlight
    
    # UI Colors
    "border": "#3e4451",          # Borders, dividers
    "line": "#1e222a",            # Line numbers, subtle lines
    "selection": "#3e4451",       # Selection background
    "hover": "#3e4451",           # Hover state
}

def get_stylesheet():
    """
    Generate complete One Dark Pro stylesheet for PySide6 application.
    
    Returns:
        str: Complete QSS stylesheet
    """
    return f"""
    /* One Dark Pro Theme */
    
    /* Main Window */
    QMainWindow {{
        background-color: {COLORS['background']};
        color: {COLORS['foreground']};
    }}
    
    /* Menus */
    QMenuBar {{
        background-color: {COLORS['background']};
        color: {COLORS['foreground']};
        border-bottom: 1px solid {COLORS['border']};
        padding: 2px;
    }}
    
    QMenuBar::item {{
        background-color: transparent;
        padding: 4px 12px;
        border-radius: 3px;
    }}
    
    QMenuBar::item:selected {{
        background-color: {COLORS['hover']};
    }}
    
    QMenuBar::item:pressed {{
        background-color: {COLORS['selection']};
    }}
    
    /* Menus Dropdown */
    QMenu {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px 0px;
    }}
    
    QMenu::item {{
        padding: 6px 20px;
        border-radius: 3px;
        margin: 2px 4px;
    }}
    
    QMenu::item:selected {{
        background-color: {COLORS['blue']};
        color: {COLORS['background']};
    }}
    
    QMenu::item:disabled {{
        color: {COLORS['comment']};
    }}
    
    QMenu::separator {{
        background-color: {COLORS['border']};
        height: 1px;
        margin: 4px 0px;
    }}
    
    /* Dock Widgets */
    QDockWidget {{
        background-color: {COLORS['background']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        titlebar-close-icon: none;
    }}
    
    QDockWidget::title {{
        background-color: {COLORS['surface']};
        padding: 6px;
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    QDockWidget::float-button {{
        background-color: transparent;
        border: none;
        padding: 2px;
    }}
    
    QDockWidget::float-button:hover {{
        background-color: {COLORS['hover']};
    }}
    
    QDockWidget::close-button {{
        background-color: transparent;
        border: none;
        padding: 2px;
    }}
    
    QDockWidget::close-button:hover {{
        background-color: {COLORS['red']};
        color: white;
    }}
    
    /* Toolbars */
    QToolBar {{
        background-color: {COLORS['background']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px;
        spacing: 4px;
    }}
    
    QToolBar::separator {{
        background-color: {COLORS['border']};
        width: 1px;
        margin: 4px;
    }}
    
    QToolButton {{
        background-color: transparent;
        color: {COLORS['foreground']};
        border: none;
        border-radius: 3px;
        padding: 4px 8px;
    }}
    
    QToolButton:hover {{
        background-color: {COLORS['hover']};
    }}
    
    QToolButton:pressed {{
        background-color: {COLORS['selection']};
    }}
    
    /* Buttons */
    QPushButton {{
        background-color: {COLORS['blue']};
        color: {COLORS['background']};
        border: none;
        border-radius: 4px;
        padding: 6px 16px;
        font-weight: bold;
        outline: none;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS['cyan']};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS['purple']};
    }}
    
    QPushButton:disabled {{
        background-color: {COLORS['comment']};
        color: {COLORS['surface']};
    }}
    
    QPushButton:focus {{
        outline: 2px solid {COLORS['blue']};
        outline-offset: 2px;
    }}
    
    /* Input Fields */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 6px;
        selection-background-color: {COLORS['blue']};
    }}
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 2px solid {COLORS['blue']};
    }}
    
    QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
        background-color: {COLORS['background']};
        color: {COLORS['comment']};
    }}
    
    /* Combo Boxes */
    QComboBox {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    QComboBox:focus {{
        border: 2px solid {COLORS['blue']};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    
    QComboBox::down-arrow {{
        image: none;
        width: 0px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        selection-background-color: {COLORS['blue']};
        border: 1px solid {COLORS['border']};
    }}
    
    /* Spinboxes */
    QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px;
    }}
    
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 2px solid {COLORS['blue']};
    }}
    
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background-color: {COLORS['hover']};
        border: 1px solid {COLORS['border']};
        width: 16px;
    }}
    
    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {COLORS['blue']};
    }}
    
    /* Sliders */
    QSlider::groove:horizontal {{
        background-color: {COLORS['border']};
        height: 4px;
        border-radius: 2px;
    }}
    
    QSlider::handle:horizontal {{
        background-color: {COLORS['blue']};
        border: none;
        width: 12px;
        margin: -4px 0;
        border-radius: 6px;
    }}
    
    QSlider::handle:horizontal:hover {{
        background-color: {COLORS['cyan']};
    }}
    
    /* List Widgets */
    QListWidget {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        outline: none;
    }}
    
    QListWidget::item {{
        padding: 4px;
        border-radius: 3px;
    }}
    
    QListWidget::item:selected {{
        background-color: {COLORS['blue']};
        color: {COLORS['background']};
    }}
    
    QListWidget::item:hover {{
        background-color: {COLORS['hover']};
    }}
    
    /* Table Widgets */
    QTableWidget {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        gridline-color: {COLORS['border']};
    }}
    
    QTableWidget::item {{
        padding: 4px;
        border-right: 1px solid {COLORS['border']};
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    QTableWidget::item:selected {{
        background-color: {COLORS['blue']};
        color: {COLORS['background']};
    }}
    
    QTableCornerButton::section {{
        background-color: {COLORS['background']};
        border: 1px solid {COLORS['border']};
    }}
    
    QHeaderView::section {{
        background-color: {COLORS['background']};
        color: {COLORS['foreground']};
        padding: 4px;
        border: 1px solid {COLORS['border']};
    }}
    
    /* Scroll Bars */
    QScrollBar:vertical {{
        background-color: {COLORS['surface']};
        width: 12px;
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {COLORS['comment']};
        border-radius: 5px;
        min-height: 20px;
        margin: 2px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['border']};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none;
    }}
    
    QScrollBar:horizontal {{
        background-color: {COLORS['surface']};
        height: 12px;
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
    }}
    
    QScrollBar::handle:horizontal {{
        background-color: {COLORS['comment']};
        border-radius: 5px;
        min-width: 20px;
        margin: 2px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background-color: {COLORS['border']};
    }}
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        border: none;
        background: none;
    }}
    
    /* Labels */
    QLabel {{
        color: {COLORS['foreground']};
        background-color: transparent;
    }}
    
    /* Tabs */
    QTabBar::tab {{
        background-color: {COLORS['background']};
        color: {COLORS['comment']};
        border: 1px solid {COLORS['border']};
        padding: 6px 16px;
        border-radius: 3px 3px 0 0;
        margin-right: 2px;
    }}
    
    QTabBar::tab:selected {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-bottom: 2px solid {COLORS['blue']};
    }}
    
    QTabBar::tab:hover {{
        background-color: {COLORS['hover']};
    }}
    
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
    }}
    
    /* Dialogs */
    QDialog {{
        background-color: {COLORS['background']};
        color: {COLORS['foreground']};
    }}
    
    /* Message Box */
    QMessageBox {{
        background-color: {COLORS['background']};
        color: {COLORS['foreground']};
    }}
    
    QMessageBox QLabel {{
        color: {COLORS['foreground']};
    }}
    
    QMessageBox QDialogButtonBox QPushButton {{
        min-width: 60px;
    }}
    
    /* Group Box */
    QGroupBox {{
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding-top: 10px;
        margin-top: 6px;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 3px;
        background-color: {COLORS['background']};
    }}
    
    /* Check Box and Radio Button */
    QCheckBox, QRadioButton {{
        color: {COLORS['foreground']};
        background-color: transparent;
        outline: none;
    }}
    
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {COLORS['border']};
        border-radius: 2px;
        background-color: {COLORS['surface']};
    }}
    
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border: 1px solid {COLORS['blue']};
    }}
    
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {COLORS['blue']};
        border: 1px solid {COLORS['blue']};
    }}
    
    QCheckBox::indicator:checked {{
        image: none;
    }}
    
    QRadioButton::indicator {{
        border-radius: 8px;
    }}
    
    QRadioButton::indicator:checked {{
        background: radial-gradient(circle, {COLORS['background']} 0%, {COLORS['background']} 30%, {COLORS['blue']} 30%, {COLORS['blue']} 100%);
    }}
    
    /* Splitter Handle */
    QSplitter::handle {{
        background-color: {COLORS['border']};
    }}
    
    QSplitter::handle:hover {{
        background-color: {COLORS['blue']};
    }}
    
    /* Status Bar */
    QStatusBar {{
        background-color: {COLORS['background']};
        color: {COLORS['foreground']};
        border-top: 1px solid {COLORS['border']};
    }}
    
    /* Tooltip */
    QToolTip {{
        background-color: {COLORS['surface']};
        color: {COLORS['foreground']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 4px;
    }}
    """

def get_palette_colors():
    """
    Get the color palette for programmatic use.
    
    Returns:
        dict: Color names and hex values
    """
    return COLORS.copy()
