"""
Theme system for H2 Plant GUI.

Provides theme management and color schemes for the application.
"""

from .theme_manager import ThemeManager
from .one_dark_pro_theme import get_stylesheet as get_one_dark_pro_stylesheet
from .one_dark_pro_theme import get_palette_colors

__all__ = [
    'ThemeManager',
    'get_one_dark_pro_stylesheet',
    'get_palette_colors',
]
