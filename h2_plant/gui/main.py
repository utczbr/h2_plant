"""
Application Entry Point.
"""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    try:
        from PySide6.QtWidgets import QApplication
        from h2_plant.gui.ui.main_window import PlantEditorWindow
        from h2_plant.gui import patches
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(f"Missing GUI dependency: {missing}", file=sys.stderr)
        print('Install GUI dependencies with: pip install -e ".[gui]"', file=sys.stderr)
        print("Alternative: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from exc

    # Apply runtime patches before creating QApplication
    patches.apply_patches()

    app = QApplication(sys.argv)

    window = PlantEditorWindow()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
