from PySide6.QtWidgets import QMainWindow, QDockWidget
from PySide6.QtCore import Qt
from NodeGraphQt import NodeGraph, PropertiesBinWidget, NodesPaletteWidget

# Detailed component nodes
from h2_plant.gui.nodes.electrolysis import PEMStackNode, SOECStackNode, RectifierNode
from h2_plant.gui.nodes.reforming import ATRReactorNode, WGSReactorNode, SteamGeneratorNode
from h2_plant.gui.nodes.separation import PSAUnitNode, SeparationTankNode
from h2_plant.gui.nodes.thermal import HeatExchangerNode
from h2_plant.gui.nodes.fluid import ProcessCompressorNode, RecirculationPumpNode
from h2_plant.gui.nodes.logistics import ConsumerNode

# External resources
from h2_plant.gui.nodes.resources import GridConnectionNode, WaterSupplyNode, AmbientHeatNode, NaturalGasSupplyNode

# Storage nodes (keep - already component-level)
from h2_plant.gui.nodes.storage import LPTankNode, HPTankNode, OxygenBufferNode

# Compression nodes (keep - already component-level)
from h2_plant.gui.nodes.compression import FillingCompressorNode, OutgoingCompressorNode

# Logic nodes (keep - already component-level)
from h2_plant.gui.nodes.logic import DemandSchedulerNode, EnergyPriceNode

# Utilities (keep selective component-level nodes)
from h2_plant.gui.nodes.utilities import BatteryNode

# Water Detail Components
from h2_plant.gui.nodes.water import WaterPurifierNode, UltraPureWaterTankNode

class PlantEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H2 Plant Configuration Editor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create node graph
        self.graph = NodeGraph()
        
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
        
        # Setup Context Menu
        self.setup_context_menu()
        
        # Toolbar
        self.toolbar = self.addToolBar("Simulation")
        self.run_action = self.toolbar.addAction("Run Simulation")
        self.run_action.triggered.connect(self.run_simulation)
        
        self.show()

    def setup_context_menu(self):
        """Setup context menu commands."""
        # Get the graph context menu (right-click on canvas)
        graph_menu = self.graph.get_context_menu('graph')
        graph_menu.add_command('Fit Zoom', self.graph.fit_to_selection, 'F')
        graph_menu.add_command('Reset Zoom', self.graph.reset_zoom, 'H')
        
        # Get the nodes context menu (right-click on nodes)
        nodes_menu = self.graph.get_context_menu('nodes')
        nodes_menu.add_command('Delete', self.delete_selection, 'Del')
        nodes_menu.add_command('Duplicate', self.duplicate_selection, 'Ctrl+D')

    def delete_selection(self):
        """Delete selected nodes."""
        selected = self.graph.selected_nodes()
        if selected:
            self.graph.delete_nodes(selected)
    
    def duplicate_selection(self):
        """Duplicate selected nodes."""
        selected = self.graph.selected_nodes()
        if selected:
            self.graph.duplicate_nodes(selected)

    def run_simulation(self):
        from h2_plant.gui.core.aggregation import aggregate_components_to_systems
        from h2_plant.gui.core.worker import SimulationWorker
        from PySide6.QtWidgets import QProgressDialog, QMessageBox
        
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
        from h2_plant.gui.ui.results_dialog import ResultsDialog
        self.progress.close()
        dialog = ResultsDialog(results, self)
        dialog.exec()
        
    def on_simulation_error(self, error_msg):
        from PySide6.QtWidgets import QMessageBox
        self.progress.close()
        QMessageBox.critical(self, "Simulation Failed", error_msg)
