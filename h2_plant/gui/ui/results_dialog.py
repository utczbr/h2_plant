"""
Dialog for displaying simulation results.
"""
import json
from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel

class ResultsDialog(QDialog):
    def __init__(self, results: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Results")
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Header
        layout.addWidget(QLabel("Simulation Completed Successfully"))
        
        # Results Text Area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        
        # Format results as JSON for now (can be improved later)
        formatted_results = json.dumps(results, indent=2)
        self.text_edit.setText(formatted_results)
        
        layout.addWidget(self.text_edit)
        
        # Close Button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
