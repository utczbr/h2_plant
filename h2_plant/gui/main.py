"""
Application Entry Point.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtWidgets import QApplication
from h2_plant.gui.ui.main_window import PlantEditorWindow
from h2_plant.gui import patches

def main():
    # Apply runtime patches before creating QApplication
    patches.apply_patches()
    
    app = QApplication(sys.argv)
    
    window = PlantEditorWindow()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
