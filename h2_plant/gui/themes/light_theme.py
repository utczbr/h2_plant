"""
Light Theme for H2 Plant GUI.

Provides a minimal light theme that uses system defaults.
"""

def get_stylesheet():
    """
    Generate light theme stylesheet.
    
    Returns:
        str: Minimal QSS stylesheet for light theme
    """
    # Return empty string to use system defaults
    # This can be customized if specific light theme styling is needed
    return ""


def get_palette_colors():
    """
    Get the color palette for programmatic use.
    
    Returns:
        dict: Color names and values for light theme
    """
    return {
        "background": "#ffffff",
        "foreground": "#000000",
        "border": "#cccccc",
        "selection": "#0078d4",
    }
