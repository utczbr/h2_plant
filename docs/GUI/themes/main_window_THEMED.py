"""
H2 Plant Configuration Editor with One Dark Pro Theme

Main window with:
- One Dark Pro (VS Code) color scheme for all menus, dialogs, panels
- Working Delete/Duplicate functionality
- Advanced validation
- Topology inference
- Graph persistence

Node positions and graph visualization remain unchanged.
Only menu/dialog/panel colors are themed.
"""

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QMessageBox, QFileDialog, QDialog,
    QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QWidget, QListWidget, QListWidgetItem, QApplication
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from NodeGraphQt import NodeGraph, PropertiesBinWidget, NodesPaletteWidget
import copy
from pathlib import Path
import logging

# Import the One Dark Pro theme
from h2_plant.gui.themes.one_dark_pro_theme import get_stylesheet

# Core Managers
from h2_plant.gui.core.topology_inference import (
    TopologyInferenceEngine,
    extract_nodes_edges_from_graph
)
from h2_plant.gui.core.graph_persistence import GraphPersistenceManager
from h2_plant.gui.core.advanced_validation import AdvancedValidator, ValidationLevel

# Detailed component nodes
from h2_plant.gui.nodes.electrolysis import PEMStackNode, SOECStackNode, RectifierNode
from h2_plant.gui.nodes.reforming import ATRReactorNode, WGSReactorNode, SteamGeneratorNode
from h2_plant.gui.nodes.separation import PSAUnitNode, SeparationTankNode
from h2_plant.gui.nodes.thermal import HeatExchangerNode
from h2_plant.gui.nodes.fluid import ProcessCompressorNode, RecirculationPumpNode
from h2_plant.gui.nodes.logistics import ConsumerNode

# External resources
from h2_plant.gui.nodes.resources import GridConnectionNode, WaterSupplyNode, AmbientHeatNode, NaturalGasSupplyNode

# Storage nodes
from h2_plant.gui.nodes.storage import LPTankNode, HPTankNode, OxygenBufferNode

# Compression nodes
from h2_plant.gui.nodes.compression import FillingCompressorNode, OutgoingCompressorNode

# Logic nodes
from h2_plant.gui.nodes.logic import DemandSchedulerNode, EnergyPriceNode

# Utilities
from h2_plant.gui.nodes.utilities import BatteryNode

# Water Detail Components
from h2_plant.gui.nodes.water import WaterPurifierNode, UltraPureWaterTankNode

logger = logging.getLogger(__name__)


class PlantEditorWindow(QMainWindow):
    """
    H2 Plant Configuration Editor with One Dark Pro Theme.
    
    Features:
    - One Dark Pro color scheme for all UI elements
    - Delete/Duplicate node functionality
    - Topology auto-detection
    - Real-time validation
    - Layout persistence
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("H2 Plant Configuration Editor")
        self.setGeometry(100, 100, 1200, 800)

        # Apply One Dark Pro theme
        self.apply_theme()

        # Create node graph
        self.graph = NodeGraph()

        # Initialize Managers
        self.topology_engine = TopologyInferenceEngine()
        self.persistence_mgr = GraphPersistenceManager(backup_dir=Path("./backups"))
        self.validator = AdvancedValidator()

        # Validation Timer
        self.validation_timer = QTimer()
        self.validation_timer.timeout.connect(self.run_validation)
        # Uncomment to enable auto-validation every 2 seconds:
        # self.validation_timer.start(2000)

        # Create properties bin
        self.properties_bin = PropertiesBinWidget(node_graph=self.graph)
        self.properties_bin.setWindowFlags(self.properties_bin.windowFlags())

        # Create nodes palette
        self.nodes_palette = NodesPaletteWidget(node_graph=self.graph)
        self.nodes_palette.setWindowFlags(self.nodes_palette.windowFlags())

        # Set central widget
        self.setCentralWidget(self.graph.widget)

        # Add properties dock (Right)
        self.prop_dock = QDockWidget("Properties", self)
        self.prop_dock.setWidget(self.properties_bin)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop_dock)

        # Add palette dock (Left)
        self.palette_dock = QDockWidget("Nodes", self)
        self.palette_dock.setWidget(self.nodes_palette)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.palette_dock)

        # Register all component-level nodes
        self.graph.register_nodes([
            # Electrolysis Components
            PEMStackNode, SOECStackNode, RectifierNode,
            # Reforming Components
            ATRReactorNode, WGSReactorNode, SteamGeneratorNode,
            # Separation Components
            PSAUnitNode, SeparationTankNode,
            # Thermal Components
            HeatExchangerNode,
            # Fluid Handling Components
            ProcessCompressorNode, RecirculationPumpNode,
            # Storage Components
            LPTankNode, HPTankNode, OxygenBufferNode,
            # Compression Components
            FillingCompressorNode, OutgoingCompressorNode,
            # Logic Components
            DemandSchedulerNode, EnergyPriceNode,
            # Utilities
            BatteryNode,
            # Water Detail Components
            WaterPurifierNode, UltraPureWaterTankNode,
            # External Resources
            GridConnectionNode, WaterSupplyNode, AmbientHeatNode, NaturalGasSupplyNode,
            # Logistics
            ConsumerNode
        ])

        # Update palette
        self.nodes_palette.update()

        # Setup Context Menu and Keyboard Shortcuts
        self.setup_context_menu()
        self.setup_keyboard_shortcuts()

        # Setup Advanced Menus and Panels
        self.setup_advanced_menu()
        self.setup_validation_panel()

        # Toolbar
        self.toolbar = self.addToolBar("Simulation")
        self.run_action = self.toolbar.addAction("Run Simulation")
        self.run_action.triggered.connect(self.run_simulation)

        self.show()

    def apply_theme(self):
        """Apply One Dark Pro theme to the application."""
        try:
            stylesheet = get_stylesheet()
            self.setStyleSheet(stylesheet)
            logger.info("One Dark Pro theme applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply theme: {e}")

    def setup_context_menu(self):
        """Setup context menu commands for nodes and canvas."""
        try:
            graph_menu = self.graph.get_context_menu('graph')
            graph_menu.add_command('Fit Zoom', self.graph.fit_to_selection, 'F')
            graph_menu.add_command('Reset Zoom', self.graph.reset_zoom, 'H')
        except Exception as e:
            logger.warning(f"Could not setup graph context menu: {e}")

        try:
            nodes_menu = self.graph.get_context_menu('nodes')
            nodes_menu.add_command('Delete', self.delete_selection, 'Del')
            nodes_menu.add_command('Duplicate', self.duplicate_selection, 'Ctrl+D')
        except Exception as e:
            logger.warning(f"Could not setup nodes context menu: {e}")

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common operations."""
        from PySide6.QtGui import QShortcut, QKeySequence

        # Delete key (Del)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selection)

        # Duplicate (Ctrl+D)
        QShortcut(QKeySequence(Qt.CTRL | Qt.Key_D), self, self.duplicate_selection)

    def delete_selection(self):
        """
        Delete selected nodes.

        NodeGraphQt uses delete_node() (singular), not delete_nodes() (plural)
        """
        try:
            selected = self.graph.selected_nodes()

            if not selected:
                return

            # Confirm deletion for multiple nodes
            if len(selected) > 1:
                reply = QMessageBox.question(
                    self,
                    "Confirm Delete",
                    f"Delete {len(selected)} nodes?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

            # Delete each node individually
            for node in list(selected):
                try:
                    self.graph.delete_node(node)
                except Exception as e:
                    logger.error(f"Error deleting node {node}: {e}")

        except Exception as e:
            QMessageBox.warning(self, "Delete Error", f"Failed to delete nodes: {e}")

    def duplicate_selection(self):
        """
        Duplicate selected nodes with properties and connections.

        NodeGraphQt doesn't have duplicate_nodes(), so we implement manually:
        1. Create new nodes with offset position
        2. Copy all properties
        3. Copy styling (colors, borders)
        4. Recreate connections between duplicated nodes
        """
        try:
            selected = self.graph.selected_nodes()

            if not selected:
                return

            # Deselect all first
            for node in self.graph.all_nodes():
                node.set_selected(False)

            node_map = {}
            offset_x = 50
            offset_y = 50

            # Step 1: Create duplicate nodes with copied properties
            for old_node in selected:
                try:
                    node_type = old_node.type_
                    new_node = self.graph.create_node(
                        node_type,
                        pos=(old_node.pos()[0] + offset_x, old_node.pos()[1] + offset_y)
                    )

                    # Copy properties (except system ones)
                    old_props = old_node.properties()
                    for prop_name, prop_value in old_props.items():
                        if prop_name not in ['id', 'name', 'pos', 'selected', 'disabled', 'type_']:
                            try:
                                new_node.set_property(prop_name, prop_value)
                            except Exception as e:
                                logger.debug(f"Could not copy property {prop_name}: {e}")

                    # Copy node styling
                    try:
                        new_node.set_color(*old_node.color)
                    except:
                        pass

                    try:
                        new_node.set_border_color(*old_node.border_color)
                    except:
                        pass

                    try:
                        new_node.set_text_color(*old_node.text_color)
                    except:
                        pass

                    node_map[old_node] = new_node
                    new_node.set_selected(True)

                except Exception as e:
                    logger.error(f"Error creating duplicate node: {e}")
                    continue

            # Step 2: Duplicate connections between duplicated nodes
            try:
                for old_node in node_map.keys():
                    if old_node not in node_map:
                        continue

                    new_node = node_map[old_node]

                    # Check output connections
                    for output_port_name, output_port in old_node.output_ports.items():
                        for connected_port in output_port.connected_ports():
                            target_node = connected_port.node()

                            # Only duplicate connection if target is also duplicated
                            if target_node in node_map:
                                new_target = node_map[target_node]
                                target_port_name = connected_port.name()

                                try:
                                    new_node.output_ports[output_port_name].connect_to(
                                        new_target.input_ports[target_port_name]
                                    )
                                except Exception as e:
                                    logger.debug(f"Could not reconnect ports: {e}")

            except Exception as e:
                logger.debug(f"Could not duplicate connections: {e}")

        except Exception as e:
            QMessageBox.warning(self, "Duplicate Error", f"Failed to duplicate nodes: {e}")

    def run_simulation(self):
        """Run the simulation."""
        from h2_plant.gui.core.aggregation import aggregate_components_to_systems
        from h2_plant.gui.core.worker import SimulationWorker
        from PySide6.QtWidgets import QProgressDialog

        # 1. Generate Config from nodes
        try:
            nodes = self.graph.all_nodes()
            if not nodes:
                QMessageBox.warning(self, "Empty Graph", "Please add some components first!")
                return

            config_dict = aggregate_components_to_systems(nodes)
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", f"Failed to generate config: {e}")
            return

        # 2. Setup Worker
        self.worker = SimulationWorker(config_dict)

        # 3. Setup Progress Dialog
        self.progress = QProgressDialog("Running Simulation...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)

        # 4. Connect Signals
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.progress.canceled.connect(self.worker.stop)

        # 5. Start
        self.worker.start()
        self.progress.show()

    def on_simulation_finished(self, results):
        """Handle simulation completion."""
        from h2_plant.gui.ui.results_dialog import ResultsDialog

        self.progress.close()
        dialog = ResultsDialog(results, self)
        dialog.exec()

    def on_simulation_error(self, error_msg):
        """Handle simulation errors."""
        self.progress.close()
        QMessageBox.critical(self, "Simulation Failed", error_msg)

    def setup_advanced_menu(self):
        """Setup File, Edit, and Validation menus."""
        # File Menu
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Save Layout", self.save_layout, "Ctrl+S")
        file_menu.addAction("Load Layout", self.load_layout, "Ctrl+O")
        file_menu.addSeparator()
        file_menu.addAction("Export Config", self.export_config)

        # Edit Menu
        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction("Auto-Detect Topology", self.detect_topology)

        # Validation Menu
        validate_menu = self.menuBar().addMenu("Validation")
        validate_menu.addAction("Check All", self.run_validation)
        validate_menu.addAction("Show Report", self.show_last_report)

    def setup_validation_panel(self):
        """Setup real-time validation panel in dock."""
        self.validation_dock = QDockWidget("Validation", self)
        widget = QWidget()
        layout = QVBoxLayout()

        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #abb2bf; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.status_label)

        # Issues list
        self.issues_list = QListWidget()
        layout.addWidget(self.issues_list)

        # Validate button
        validate_btn = QPushButton("Validate Now")
        validate_btn.clicked.connect(self.run_validation)
        layout.addWidget(validate_btn)

        widget.setLayout(layout)
        self.validation_dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.validation_dock)

    # --- Persistence Methods ---

    def save_layout(self):
        """Save current layout to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plant Layout", "", "H2Plant (*.h2plant);;JSON (*.json)"
        )
        if file_path:
            try:
                config_dict = self.generate_config()
                snapshot = self.persistence_mgr.create_snapshot(
                    self.graph, config_dict, project_name="My Plant"
                )
                self.persistence_mgr.save(file_path, snapshot)
                QMessageBox.information(self, "Success", "Layout saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_layout(self):
        """Load layout from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Plant Layout", "", "H2Plant (*.h2plant);;JSON (*.json)"
        )
        if file_path:
            try:
                snapshot = self.persistence_mgr.load(file_path)
                self.graph.clear_session()

                # Node type mapping
                from h2_plant.gui.nodes.electrolysis import PEMStackNode, SOECStackNode, RectifierNode
                from h2_plant.gui.nodes.reforming import ATRReactorNode, WGSReactorNode, SteamGeneratorNode
                from h2_plant.gui.nodes.separation import PSAUnitNode, SeparationTankNode
                from h2_plant.gui.nodes.thermal import HeatExchangerNode
                from h2_plant.gui.nodes.fluid import ProcessCompressorNode, RecirculationPumpNode
                from h2_plant.gui.nodes.logistics import ConsumerNode
                from h2_plant.gui.nodes.resources import GridConnectionNode, WaterSupplyNode, AmbientHeatNode, NaturalGasSupplyNode
                from h2_plant.gui.nodes.storage import LPTankNode, HPTankNode, OxygenBufferNode
                from h2_plant.gui.nodes.compression import FillingCompressorNode, OutgoingCompressorNode
                from h2_plant.gui.nodes.logic import DemandSchedulerNode, EnergyPriceNode
                from h2_plant.gui.nodes.utilities import BatteryNode
                from h2_plant.gui.nodes.water import WaterPurifierNode, UltraPureWaterTankNode

                node_classes = {
                    cls.__name__: cls for cls in [
                        PEMStackNode, SOECStackNode, RectifierNode,
                        ATRReactorNode, WGSReactorNode, SteamGeneratorNode,
                        PSAUnitNode, SeparationTankNode, HeatExchangerNode,
                        ProcessCompressorNode, RecirculationPumpNode, ConsumerNode,
                        GridConnectionNode, WaterSupplyNode, AmbientHeatNode, NaturalGasSupplyNode,
                        LPTankNode, HPTankNode, OxygenBufferNode,
                        FillingCompressorNode, OutgoingCompressorNode,
                        DemandSchedulerNode, EnergyPriceNode, BatteryNode,
                        WaterPurifierNode, UltraPureWaterTankNode
                    ]
                }

                self.graph.clear_session()
                id_map = {}

                for old_id, node_data in snapshot.nodes.items():
                    node_type = node_data["type"]
                    try:
                        new_node = self.graph.create_node(node_type, pos=(0, 0))
                    except:
                        new_node = None

                    if not new_node and '.' not in node_type:
                        node_class = node_classes.get(node_type)
                        if node_class:
                            full_type = f"{node_class.__identifier__}.{node_class.__name__}"
                            try:
                                new_node = self.graph.create_node(full_type, pos=(0, 0))
                            except Exception as e:
                                logger.debug(f"Failed to create node {node_type}: {e}")

                    if new_node:
                        id_map[old_id] = new_node

                        for prop, val in node_data["properties"].items():
                            if prop not in ['id', 'name', 'pos', 'selected']:
                                try:
                                    new_node.set_property(prop, val)
                                except:
                                    pass

                        geom = node_data["geometry"]
                        new_node.set_pos(geom["x"], geom["y"])

                for edge_data in snapshot.edges:
                    source_node = id_map.get(edge_data["source_node_id"])
                    target_node = id_map.get(edge_data["target_node_id"])

                    if source_node and target_node:
                        try:
                            source_port_name = edge_data["source_port"]
                            target_port_name = edge_data["target_port"]
                            source_ports = source_node.output_ports()
                            target_ports = target_node.input_ports()

                            src_port = None
                            for p in source_ports:
                                if p.name() == source_port_name:
                                    src_port = p
                                    break

                            tgt_port = None
                            for p in target_ports:
                                if p.name() == target_port_name:
                                    tgt_port = p
                                    break

                            if src_port and tgt_port:
                                src_port.connect_to(tgt_port)
                        except Exception as e:
                            logger.debug(f"Failed to connect ports: {e}")

                QMessageBox.information(self, "Success", "Layout loaded successfully")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
                logger.exception("Load layout error")

    def export_config(self):
        """Export configuration to JSON."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration", "", "JSON (*.json)"
        )
        if file_path:
            try:
                config_dict = self.generate_config()
                import json
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                QMessageBox.information(self, "Success", "Configuration exported")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")

    def generate_config(self):
        """Generate configuration from current graph."""
        from h2_plant.gui.core.aggregation import aggregate_components_to_systems
        nodes = self.graph.all_nodes()
        return aggregate_components_to_systems(nodes)

    # --- Inference & Validation Methods ---

    def detect_topology(self):
        """Auto-detect topology from graph connections."""
        try:
            nodes, edges = extract_nodes_edges_from_graph(self.graph)
            analysis = self.topology_engine.infer(nodes, edges)
            self.show_topology_analysis(analysis)
        except Exception as e:
            logger.error(f"Topology detection failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to detect topology: {e}")

    def show_topology_analysis(self, analysis):
        """Display topology analysis results."""
        msg = f"""
Detected Pattern: {analysis.detected_pattern.value}
Confidence: {analysis.confidence_score:.1%}

Production Systems:
"""
        for prod_id, tanks in analysis.production_systems.items():
            msg += f"  {prod_id} → {len(tanks)} tank(s)\n"

        if analysis.has_recirculation:
            msg += "\nRecirculation: YES\n"

        msg += f"\nFlow Networks: {len(analysis.flow_networks)}\n"

        if analysis.warnings:
            msg += "\nWarnings:\n"
            for warning in analysis.warnings:
                msg += f"  - {warning}\n"

        QMessageBox.information(self, "Topology Analysis", msg)

    def run_validation(self):
        """Run validation and update UI."""
        try:
            report = self.validator.validate(self.graph)

            # Update status
            if report.is_valid:
                self.status_label.setText("✓ Valid")
                self.status_label.setStyleSheet("color: #98c379; font-weight: bold; font-size: 14px;")
            elif report.has_errors:
                self.status_label.setText("✗ Errors")
                self.status_label.setStyleSheet("color: #e06c75; font-weight: bold; font-size: 14px;")
            else:
                self.status_label.setText("⚠ Warnings")
                self.status_label.setStyleSheet("color: #e5c07b; font-weight: bold; font-size: 14px;")

            # Update issues list
            self.issues_list.clear()
            for issue in sorted(report.issues, key=lambda i: i.level.value):
                item_text = f"[{issue.level.value.upper()}] {issue.message}"
                item = QListWidgetItem(item_text)

                if issue.level == ValidationLevel.ERROR:
                    item.setForeground(QColor("#e06c75"))
                elif issue.level == ValidationLevel.WARNING:
                    item.setForeground(QColor("#e5c07b"))

                self.issues_list.addItem(item)

        except Exception as e:
            logger.error(f"Validation error: {e}")

    def show_last_report(self):
        """Show detailed validation report."""
        if hasattr(self.validator, 'last_report') and self.validator.last_report:
            self.run_validation()


# Entry point
if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = PlantEditorWindow()
    sys.exit(app.exec())
