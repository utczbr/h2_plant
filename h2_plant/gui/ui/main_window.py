import json
"""
H2 Plant Configuration Editor - Complete Version
Includes: File/Edit/View/Validation menus, Run Simulation, All Nodes tab with working drag-drop
"""
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QMessageBox, QFileDialog, QDialog,
    QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QWidget, QListWidget, QListWidgetItem, QApplication, QTabWidget,
    QProgressDialog, QCheckBox, QDialogButtonBox, QScrollArea, QGridLayout, QGroupBox, QHBoxLayout
)
from PySide6.QtCore import Qt, QTimer, QMimeData, QThread, Signal, QSettings
from PySide6.QtGui import QColor, QShortcut, QKeySequence, QDrag, QAction
from NodeGraphQt import NodeGraph, PropertiesBinWidget, NodesPaletteWidget
import copy
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path

# Core Managers
from h2_plant.gui.core.topology_inference import TopologyInferenceEngine, extract_nodes_edges_from_graph
from h2_plant.gui.core.graph_persistence import GraphPersistenceManager
from h2_plant.gui.core.advanced_validation import AdvancedValidator, ValidationLevel
from h2_plant.gui.core.worker import SimulationWorker
from h2_plant.gui.core.graph_adapter import GraphToConfigAdapter, GraphNode, GraphEdge, FlowType, Port

# Node imports
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
from h2_plant.gui.themes.theme_manager import ThemeManager
# New nodes
from h2_plant.gui.nodes.energy_source import WindEnergySourceNode
from h2_plant.gui.nodes.economics import ArbitrageNode
from h2_plant.gui.nodes.mixing import MixerNode


class AllNodesListWidget(QListWidget):
    """Enhanced list widget with proper drag-and-drop for NodeGraphQt."""
    
    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.graph = graph
        self.setDragEnabled(True) 
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setAcceptDrops(False)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item:
            return
            
        node_class = item.data(Qt.UserRole)
        if not node_class:
            return
            
        mimeData = QMimeData()
        node_identifier = node_class.__identifier__
        
        # 1. Set standard text (generic fallback)
        mimeData.setText(node_identifier)
        
        # 2. Set JSON data simulating NodeGraphQt's native palette
        # Many NodeGraphQt versions expect a JSON object with a 'nodes' list
        drag_data = {'nodes': [node_identifier]}
        try:
            json_data = json.dumps(drag_data).encode('utf-8')
            mimeData.setData('application/x-node-graph-qt', json_data)
        except Exception as e:
            print(f"Error encoding drag data: {e}")
        
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(supportedActions)


class SimulationReportWidget(QWidget):
    """Widget to display simulation reports with checkboxes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_canvases = {}  # Stores graph canvases
        self.graph_files = {
            'dispatch': 'report_dispatch.png',
            'arbitrage': 'report_arbitrage.png',
            'h2': 'report_h2_production.png',
            'oxygen': 'report_oxygen_production.png',
            'water': 'report_water_consumption.png',
            'pie': 'report_energy_pie.png',
            'histogram': 'report_price_histogram.png',
            'dispatchcurve': 'report_dispatch_curve.png'
        }
        
        self.graph_labels = {
            'dispatch': 'Power Dispatch',
            'arbitrage': 'Price Scenario',
            'h2': 'H2 Production',
            'oxygen': 'O2 Production',
            'water': 'Water Consumption',
            'pie': 'Energy Distribution',
            'histogram': 'Price Histogram',
            'dispatchcurve': 'Dispatch Curve'
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup widget UI."""
        main_layout = QVBoxLayout(self)
        
        # Control Area (Checkboxes)
        control_group = QGroupBox("Select Graphs to Display:")
        control_layout = QVBoxLayout()
        
        # Quick Select Buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        refresh_btn = QPushButton("Refresh Graphs")
        
        select_all_btn.clicked.connect(self.select_all_graphs)
        deselect_all_btn.clicked.connect(self.deselect_all_graphs)
        refresh_btn.clicked.connect(self.load_graphs)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addWidget(refresh_btn)
        button_layout.addStretch()
        
        control_layout.addLayout(button_layout)
        
        # Checkboxes
        checkbox_layout = QGridLayout()
        self.checkboxes = {}
        
        # Operational Graphs
        operational_graphs = list(self.graph_files.keys())
        for i, key in enumerate(operational_graphs):
            cb = QCheckBox(self.graph_labels[key])
            cb.setChecked(True)
            cb.stateChanged.connect(self.on_checkbox_changed)
            self.checkboxes[key] = cb
            checkbox_layout.addWidget(cb, i // 4, i % 4)
        
        control_layout.addLayout(checkbox_layout)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Graph Display Area (Scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.graphs_container = QWidget()
        self.graphs_layout = QGridLayout(self.graphs_container)
        self.graphs_layout.setSpacing(10)
        
        scroll_area.setWidget(self.graphs_container)
        main_layout.addWidget(scroll_area)
        
        # Initial Message
        self.no_data_label = QLabel("No graphs available. Run simulation first.")
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.setStyleSheet("color: gray; font-size: 14px;")
        self.graphs_layout.addWidget(self.no_data_label, 0, 0, 1, 2)
    
    def select_all_graphs(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)
    
    def deselect_all_graphs(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)
    
    def on_checkbox_changed(self):
        self.update_visible_graphs()
    
    def load_graphs(self):
        """Load graphs from disk."""
        # Clear previous
        for canvas in self.graph_canvases.values():
            canvas.setParent(None)
        self.graph_canvases.clear()
        
        if self.no_data_label:
            self.no_data_label.setParent(None)
            self.no_data_label = None
        
        row, col = 0, 0
        graphs_found = False
        
        # Assume graphs are in project root (CWD)
        project_root = Path("/home/stuart/Documentos/Planta Hidrogenio")
        
        for key, filename in self.graph_files.items():
            filepath = project_root / filename
            
            if filepath.exists():
                graphs_found = True
                
                # Create matplotlib figure
                fig = Figure(figsize=(6, 4), dpi=100)
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                
                # Load and display image
                img = plt.imread(str(filepath))
                ax.imshow(img)
                ax.axis('off')
                fig.tight_layout()
                
                canvas.setToolTip(self.graph_labels[key])
                self.graph_canvases[key] = canvas
                
                self.graphs_layout.addWidget(canvas, row, col)
                
                col += 1
                if col >= 2:
                    col = 0
                    row += 1
        
        if not graphs_found:
            self.no_data_label = QLabel("No graphs found. Run simulation first.")
            self.no_data_label.setAlignment(Qt.AlignCenter)
            self.no_data_label.setStyleSheet("color: gray; font-size: 14px;")
            self.graphs_layout.addWidget(self.no_data_label, 0, 0, 1, 2)
        else:
            self.update_visible_graphs()
    
    def update_visible_graphs(self):
        for key, canvas in self.graph_canvases.items():
            if key in self.checkboxes:
                is_checked = self.checkboxes[key].isChecked()
                canvas.setVisible(is_checked)




class PlantEditorWindow(QMainWindow):
    """H2 Plant Configuration Editor - Full Version."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H2 Plant Configuration Editor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create node graph
        self.graph = NodeGraph()
        
        # Initialize Managers
        self.topology_engine = TopologyInferenceEngine()
        self.persistence_mgr = GraphPersistenceManager(backup_dir=Path("./backups"))
        self.validator = AdvancedValidator()
        
        # Validation Timer
        self.validation_timer = QTimer()
        self.validation_timer.timeout.connect(self.run_validation_silent)
        
        # Set central widget with Tabs
        self.central_tabs = QTabWidget()
        
        # Tab 1: Graph Editor
        self.central_tabs.addTab(self.graph.widget, "Run Simulation")
        
        # Tab 2: Simulation Report
        self.report_widget = SimulationReportWidget()
        self.central_tabs.addTab(self.report_widget, "Simulation Report")
        
        self.setCentralWidget(self.central_tabs)
        
        # Register nodes FIRST
        self.register_all_nodes()
        
        # Setup UI components
        self.setup_docks()
        self.setup_menus()
        self.setup_toolbar()
        self.setup_context_menu()
        self.setup_keyboard_shortcuts()
        
        # Apply theme
        ThemeManager.apply_theme(self, QApplication.instance(), "dark")
        self.show()
    
    def register_all_nodes(self):
        """Register all node types."""
        self.node_classes = [
            # Electrolysis
            PEMStackNode, SOECStackNode,
            # Storage
            LPTankNode, HPTankNode,
            # Compression
            FillingCompressorNode, OutgoingCompressorNode,
            # Sources (NEW)
            WindEnergySourceNode,
            # Economics (NEW)
            ArbitrageNode,
            # Flow Control (NEW)
            MixerNode,
        ]
        self.graph.register_nodes(self.node_classes)
    
    def setup_docks(self):
        """Setup dock widgets."""
        # Properties dock (Right)
        self.properties_bin = PropertiesBinWidget(node_graph=self.graph)
        self.prop_dock = QDockWidget("Properties", self)
        self.prop_dock.setWidget(self.properties_bin)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop_dock)

        # Nodes palette dock
        self.nodes_palette = NodesPaletteWidget(node_graph=self.graph)
        self.all_nodes_list = AllNodesListWidget(self.graph)
        
        # Populate "All Nodes" tab
        for cls in self.node_classes:
            item = QListWidgetItem(cls.NODE_NAME)
            item.setData(Qt.UserRole, cls)
            item.setToolTip(f"Drag to canvas to create {cls.NODE_NAME}")
            self.all_nodes_list.addItem(item)

        self.palette_tabs = QTabWidget()
        self.palette_tabs.addTab(self.nodes_palette, "Categories")
        self.palette_tabs.addTab(self.all_nodes_list, "All Nodes")
        
        self.palette_dock = QDockWidget("Nodes", self)
        self.palette_dock.setWidget(self.palette_tabs)
        
        # [FIX] Reverted to LeftDockWidgetArea (Standard layout)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.palette_dock)
        
        # Update palette
        self.nodes_palette.update()
    
    def setup_menus(self):
        """Setup menu bar with File, Edit, View, Validation."""
        menubar = self.menuBar()
        
        # FILE MENU
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Layout", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_layout)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Layout...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_layout)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Layout", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_layout)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save Layout As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_layout_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # EDIT MENU
        edit_menu = menubar.addMenu("Edit")
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(lambda: self.graph.undo())
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(lambda: self.graph.redo())
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        delete_action = QAction("Delete", self)
        delete_action.setShortcut("Del")
        delete_action.triggered.connect(self.delete_selection)
        edit_menu.addAction(delete_action)
        
        duplicate_action = QAction("Duplicate", self)
        duplicate_action.setShortcut("Ctrl+D")
        duplicate_action.triggered.connect(self.duplicate_selection)
        edit_menu.addAction(duplicate_action)
        
        edit_menu.addSeparator()
        
        select_all_action = QAction("Select All", self)
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(lambda: self.graph.select_all())
        edit_menu.addAction(select_all_action)
        
        clear_selection_action = QAction("Clear Selection", self)
        clear_selection_action.setShortcut("Ctrl+Shift+A")
        clear_selection_action.triggered.connect(lambda: self.graph.clear_selection())
        edit_menu.addAction(clear_selection_action)
        
        # VIEW MENU
        view_menu = menubar.addMenu("View")
        
        fit_action = QAction("Fit to Selection", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(lambda: self.graph.fit_to_selection())
        view_menu.addAction(fit_action)
        
        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.setShortcut("H")
        reset_zoom_action.triggered.connect(lambda: self.graph.reset_zoom())
        view_menu.addAction(reset_zoom_action)
        
        view_menu.addSeparator()
        
        toggle_props_action = QAction("Toggle Properties Panel", self)
        toggle_props_action.triggered.connect(lambda: self.prop_dock.setVisible(not self.prop_dock.isVisible()))
        view_menu.addAction(toggle_props_action)
        
        toggle_palette_action = QAction("Toggle Nodes Panel", self)
        toggle_palette_action.triggered.connect(lambda: self.palette_dock.setVisible(not self.palette_dock.isVisible()))
        view_menu.addAction(toggle_palette_action)

        # VALIDATION MENU
        validation_menu = menubar.addMenu("Validation")
        
        validate_action = QAction("Run Validation", self)
        validate_action.setShortcut("Ctrl+V")
        validate_action.triggered.connect(self.run_validation)
        validation_menu.addAction(validate_action)
        
        validation_menu.addSeparator()
        
        auto_validate_action = QAction("Auto-Validate (every 2s)", self)
        auto_validate_action.setCheckable(True)
        auto_validate_action.toggled.connect(self.toggle_auto_validation)
        validation_menu.addAction(auto_validate_action)

        # --- NEW LOCATION FOR RUN SIMULATION ---
        # Added directly to menubar to appear right of "Validation"
        run_action = QAction("Run Simulation", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run_simulation)
        menubar.addAction(run_action)

    def setup_toolbar(self):
        """Toolbar removed as requested."""
        pass
    
    def setup_context_menu(self):
        """Setup context menus."""
        try:
            graph_menu = self.graph.get_context_menu('graph')
            graph_menu.add_command('Delete', self.delete_selection)
            graph_menu.add_command('Duplicate', self.duplicate_selection)
        except Exception as e:
            print(f"Warning: Could not setup context menu: {e}")
    
    def setup_keyboard_shortcuts(self):
        """Setup additional keyboard shortcuts."""
        QShortcut(QKeySequence(Qt.CTRL | Qt.Key_D), self, self.duplicate_selection)
    
    # ---- FILE OPERATIONS ----
    def new_layout(self):
        """Create a new empty layout."""
        reply = QMessageBox.question(self, "New Layout", 
                                     "Clear current layout?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.graph.clear_session()
    
    def save_layout(self):
        """Save current layout."""
        if not hasattr(self, 'current_file') or not self.current_file:
            self.save_layout_as()
        else:
            try:
                snapshot = self.persistence_mgr.create_snapshot(self.graph, {})
                self.persistence_mgr.save(self.current_file, snapshot)
                QMessageBox.information(self, "Success", "Layout saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")
    
    def save_layout_as(self):
        """Save layout with file dialog."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Layout", "", "H2_Plant Files (*.h2plant)"
        )
        if filepath:
            try:
                snapshot = self.persistence_mgr.create_snapshot(self.graph, {})
                self.persistence_mgr.save(filepath, snapshot)
                self.current_file = filepath
                QMessageBox.information(self, "Success", "Layout saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")
    
    def load_layout(self):
        """Load layout from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Layout", "", "H2_Plant Files (*.h2plant)"
        )
        if filepath:
            try:
                self.graph.clear_session()
                
                # DEBUG: Check registered nodes
                print(f"DEBUG: Registered nodes: {self.graph.registered_nodes()}")
                
                # Fix: load() returns snapshot, doesn't take graph
                snapshot = self.persistence_mgr.load(filepath)
                # Restore snapshot to graph
                self.persistence_mgr.restore_to_graph(self.graph, snapshot)
                
                self.current_file = filepath
                QMessageBox.information(self, "Success", "Layout loaded!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Load failed: {e}")
    
    # ---- EDIT OPERATIONS ----
    def delete_selection(self):
        """Delete selected nodes."""
        selected = self.graph.selected_nodes()
        if not selected:
            return
        for node in list(selected):
            try:
                self.graph.delete_node(node)
            except Exception as e:
                print(f"Error deleting node: {e}")
    
    def duplicate_selection(self):
        """Duplicate selected nodes."""
        selected = self.graph.selected_nodes()
        if not selected:
            return
        
        offset = 50
        new_nodes = []
        
        for node in selected:
            try:
                new_node = self.graph.create_node(
                    node.__class__,
                    name=f"{node.name()}_copy",
                    pos=[node.x_pos() + offset, node.y_pos() + offset]
                )
                for prop_name, prop_value in node.properties.items():
                    try:
                        new_node.set_property(prop_name, copy.deepcopy(prop_value))
                    except:
                        pass
                new_nodes.append(new_node)
            except Exception as e:
                print(f"Error duplicating node: {e}")
        
        self.graph.clear_selection()
        for node in new_nodes:
            node.set_selected(True)
    
    # ---- VALIDATION ----
    def run_validation(self):
        """Run validation and show results."""
        try:
            report = self.validator.validate(self.graph)
            if report.is_valid:
                QMessageBox.information(self, "Validation", "✓ Graph is valid!")
            else:
                issues_text = "\n".join([f"• {i.message}" for i in report.issues[:10]])
                if report.total_issues > 10:
                    issues_text += f"\n... and {report.total_issues - 10} more"
                QMessageBox.warning(self, "Validation Issues", issues_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Validation failed: {e}")
    
    def run_validation_silent(self):
        """Run validation without popup."""
        try:
            report = self.validator.validate(self.graph)
            # Update status bar or indicator
            if hasattr(self, 'statusBar'):
                status = "✓ Valid" if report.is_valid else f"⚠ {report.total_issues} issues"
                self.statusBar().showMessage(status)
        except Exception as e:
            print(f"Validation error: {e}")
    
    def toggle_auto_validation(self, enabled):
        """Toggle automatic validation."""
        if enabled:
            self.validation_timer.start(2000)
        else:
            self.validation_timer.stop()
    
    # ---- SIMULATION ----
    # ---- SIMULATION ----
    def run_simulation(self):
        """Run plant simulation using new backend architecture."""
        try:
            # 1. Create Adapter
            adapter = GraphToConfigAdapter()
            
            # 2. Extract Nodes
            for node in self.graph.all_nodes():
                # Extract ports
                ports = []
                for p in node.input_ports():
                    ports.append(Port(name=p.name(), flow_type=FlowType.HYDROGEN, direction="input")) # Defaulting to H2 for now
                for p in node.output_ports():
                    ports.append(Port(name=p.name(), flow_type=FlowType.HYDROGEN, direction="output"))

                # Create GraphNode
                graph_node = GraphNode(
                    id=node.id,
                    type=node.type_,
                    display_name=node.name(),
                    x=node.x_pos(),
                    y=node.y_pos(),
                    properties=node.properties(),
                    ports=ports
                )
                adapter.add_node(graph_node)
                
            # 3. Extract Edges
            for node in self.graph.all_nodes():
                for output_port in node.output_ports():
                    for target_port in output_port.connected_ports():
                        target_node = target_port.node()
                        
                        edge = GraphEdge(
                            source_node_id=node.id,
                            source_port=output_port.name(),
                            target_node_id=target_node.id,
                            target_port=target_port.name(),
                            flow_type=FlowType.HYDROGEN # Default, inference engine can refine this
                        )
                        adapter.add_edge(edge)
            
            # 4. Generate Context
            context = adapter.to_simulation_context()
            
            # 5. Create Worker
            self.worker = SimulationWorker(context)
            
            # 6. Setup Progress Dialog
            progress = QProgressDialog("Running Simulation... Please wait.", None, 0, 0, self)
            progress.setWindowTitle("Simulation in Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            
            def on_finished(history):
                progress.accept()
                
                # Generate Plots
                try:
                    from h2_plant.gui.core.plotter import generate_plots
                    generate_plots(history, output_dir="/home/stuart/Documentos/Planta Hidrogenio")
                except Exception as e:
                    print(f"Error generating plots: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Reload graphs from disk
                if hasattr(self, 'report_widget'):
                    self.report_widget.load_graphs()
                    self.central_tabs.setCurrentIndex(1)
                
                QMessageBox.information(self, "Simulation", "Simulation completed successfully!")
                self.worker = None
                
            def on_error(err_msg):
                progress.reject()
                QMessageBox.critical(self, "Simulation Error", f"Failed to run simulation: {err_msg}")
                self.worker = None
                
            self.worker.finished.connect(on_finished)
            self.worker.error.connect(on_error)
            
            self.worker.start()
            progress.exec_()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Simulation Error", f"Failed to start simulation: {e}")

