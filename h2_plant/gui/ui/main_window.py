import json
"""
H2 Plant Configuration Editor - Complete Version
Includes: File/Edit/View/Validation menus, Run Simulation, All Nodes tab with working drag-drop
"""
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QMessageBox, QFileDialog, QDialog,
    QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QWidget, QListWidget, QListWidgetItem, QApplication, QTabWidget,
    QProgressDialog, QCheckBox, QDialogButtonBox, QScrollArea, QGridLayout, 
    QGroupBox, QHBoxLayout, QSplitter, QSizePolicy, QFrame, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QMimeData, QThread, Signal, QSettings, QRunnable, QThreadPool, QObject, Slot
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
from h2_plant.gui.nodes.pumping import PumpNode 
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
from h2_plant.gui.nodes.mixing import MixerNode, WaterMixerNode


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
        
        # NodeGraphQt expects the full node type: {__identifier__}.{NODE_NAME}
        # For example: "h2_plant.electrolysis.pem.PEM"
        node_identifier = node_class.__identifier__
        node_name = node_class.NODE_NAME
        full_node_type = f"{node_identifier}.{node_name}"
        
        # 1. Set standard text (generic fallback)
        mimeData.setText(full_node_type)
        
        # 2. Set JSON data using NodeGraphQt's native format
        # NodeGraphQt expects: {'nodes': [{'type': 'full.node.type'}]}
        drag_data = {'nodes': [{'type': full_node_type}]}
        try:
            json_data = json.dumps(drag_data).encode('utf-8')
            mimeData.setData('application/x-nodegraphqt', json_data)
        except Exception as e:
            print(f"Error encoding drag data: {e}")
        
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.exec_(supportedActions)


from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

# ==============================================================================
# GRAPH HIERARCHY - Extensible folder structure for graph categories
# ==============================================================================
# To add new folder groups:
# 1. Add a new key to GRAPH_HIERARCHY with a list of graph_ids from GRAPH_REGISTRY
# 2. Component-specific graphs can be added as new folders (e.g., "Compressor", "Tank")
#
# Available graph_ids are defined in plotter.py GRAPH_REGISTRY

GRAPH_HIERARCHY = {
    "Plant Overview": ["dispatch", "power_balance", "energy_pie"],
    "Production": ["h2_production", "oxygen_production", "cumulative_h2", "dispatch_curve"],
    "Economics": ["arbitrage", "price_histogram", "revenue_analysis"],
    "Resource Consumption": ["water_consumption", "cumulative_energy"],
    "Efficiency": ["efficiency_curve"],
    "Physics & Degradation": ["polarization", "degradation", "compressor_ts", "module_power"],
}


# ==============================================================================
# PERFORMANCE OPTIMIZATION - Caching and Lazy Loading
# ==============================================================================

import hashlib
from functools import lru_cache


class FigureCache:
    """
    LRU cache for generated matplotlib figures.
    
    Caches figures by (graph_id, data_hash) to avoid regeneration when:
    - User toggles checkbox off/on
    - User switches between tabs and returns
    
    The cache is invalidated when simulation data changes (new hash).
    """
    
    def __init__(self, max_size: int = 20):
        self._cache = {}  # (graph_id, data_hash) -> Figure
        self._access_order = []  # LRU tracking
        self._max_size = max_size
        self._current_data_hash = None
    
    def get_data_hash(self, simulation_data: dict) -> str:
        """Generate a hash from simulation data for cache invalidation."""
        # Use first and last values + length as a quick fingerprint
        try:
            keys = list(simulation_data.keys())[:5]
            sample = str([(k, len(simulation_data.get(k, []))) for k in keys])
            return hashlib.md5(sample.encode()).hexdigest()[:8]
        except:
            return "unknown"
    
    def set_data(self, simulation_data: dict):
        """Update the current data hash, clearing cache if data changed."""
        new_hash = self.get_data_hash(simulation_data)
        if new_hash != self._current_data_hash:
            self.clear()
            self._current_data_hash = new_hash
    
    def get(self, graph_id: str) -> object:
        """Get cached figure or None."""
        key = (graph_id, self._current_data_hash)
        if key in self._cache:
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, graph_id: str, figure: object):
        """Cache a figure."""
        key = (graph_id, self._current_data_hash)
        
        # Evict LRU if at capacity
        while len(self._cache) >= self._max_size and self._access_order:
            old_key = self._access_order.pop(0)
            old_fig = self._cache.pop(old_key, None)
            if old_fig:
                try:
                    old_fig.clear()
                    import matplotlib.pyplot as plt
                    plt.close(old_fig)
                except:
                    pass
        
        self._cache[key] = figure
        self._access_order.append(key)
    
    def clear(self):
        """Clear all cached figures."""
        for fig in self._cache.values():
            try:
                fig.clear()
                import matplotlib.pyplot as plt
                plt.close(fig)
            except:
                pass
        self._cache.clear()
        self._access_order.clear()


class GraphWorkerSignals(QObject):
    """Signals emitted by GraphWorker."""
    graph_ready = Signal(str, object)  # graph_id, Figure
    error = Signal(str, str)  # graph_id, error_message


class GraphWorker(QRunnable):
    """Worker for generating Matplotlib figures in a background thread."""
    
    def __init__(self, graph_id: str, simulation_data: dict, normalized_df=None):
        super().__init__()
        self.graph_id = graph_id
        self.simulation_data = simulation_data
        self.normalized_df = normalized_df
        self.signals = GraphWorkerSignals()
        self.setAutoDelete(True)
    
    @Slot()
    def run(self):
        """Generate the figure in background thread."""
        try:
            from h2_plant.gui.core.plotter import create_figure, normalize_history, GRAPH_REGISTRY
            
            # Use pre-normalized DataFrame if available
            if self.normalized_df is not None:
                df = self.normalized_df
                func = GRAPH_REGISTRY.get(self.graph_id, {}).get('func')
                if func:
                    fig = func(df)
                else:
                    fig = None
            else:
                fig = create_figure(self.graph_id, self.simulation_data)
            
            self.signals.graph_ready.emit(self.graph_id, fig)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.error.emit(self.graph_id, str(e))


class LazyGraphSlot(QFrame):
    """
    Lazy-loading placeholder that triggers graph generation when visible.
    
    Uses visibility detection to only generate graphs that are actually
    in the scroll viewport, dramatically reducing initial load time.
    """
    
    # Signal to request graph generation
    request_generation = Signal(str)  # graph_id
    
    def __init__(self, graph_id: str, graph_name: str, parent=None):
        super().__init__(parent)
        self.graph_id = graph_id
        self.graph_name = graph_name
        self._generation_requested = False
        self._is_loaded = False
        
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setMinimumHeight(200)
        self.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
            }
        """)
        
        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignCenter)
        
        # Graph name
        self._name_label = QLabel(graph_name)
        self._name_label.setStyleSheet("color: #888; font-size: 13px;")
        self._name_label.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._name_label)
        
        # Status label
        self._status_label = QLabel("Scroll to load")
        self._status_label.setStyleSheet("color: #555; font-size: 11px;")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._status_label)
    
    def check_visibility(self, viewport_rect):
        """Check if this slot is visible in the viewport and request generation."""
        if self._generation_requested or self._is_loaded:
            return
        
        # Get global position of this widget
        my_rect = self.rect()
        my_global = self.mapToGlobal(my_rect.topLeft())
        my_global_rect = my_rect.translated(my_global.x(), my_global.y())
        
        # Check intersection with viewport
        if viewport_rect.intersects(my_global_rect):
            self._request_graph()
    
    def _request_graph(self):
        """Request graph generation."""
        if self._generation_requested:
            return
        self._generation_requested = True
        self._status_label.setText("Loading...")
        self._status_label.setStyleSheet("color: #2196F3; font-size: 11px;")
        self.request_generation.emit(self.graph_id)
    
    def mark_loaded(self):
        """Mark this slot as having its graph loaded."""
        self._is_loaded = True


class SimulationReportWidget(QWidget):
    """
    Widget to display simulation reports with hierarchical tree selection.
    
    Features:
    - LAZY LOADING: Only generates graphs visible in viewport
    - FIGURE CACHING: Reuses already-generated figures on checkbox toggle
    - QSplitter with resizable sidebar and content panes
    - Async background generation via QThreadPool
    - Debounced rendering to prevent excessive reloads
    
    EXTENSIBILITY:
    - Add new graphs by registering in plotter.py GRAPH_REGISTRY
    - Add new folders by extending GRAPH_HIERARCHY above
    """
    
    # Graph display constants
    GRAPH_MIN_HEIGHT = 400
    DEBOUNCE_DELAY_MS = 300
    SIDEBAR_MIN_WIDTH = 180
    SIDEBAR_DEFAULT_WIDTH = 250
    VISIBILITY_CHECK_INTERVAL_MS = 150
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph_canvases = {}  # graph_id -> FigureCanvas
        self.lazy_slots = {}  # graph_id -> LazyGraphSlot
        self.simulation_data = None
        self.normalized_df = None  # Cached normalized DataFrame
        self.no_data_label = None
        self._tree_items = {}
        self._pending_graphs = set()
        
        # Performance: Figure cache
        self._figure_cache = FigureCache(max_size=15)
        
        # Thread pool for async graph generation
        self._thread_pool = QThreadPool.globalInstance()
        
        # Debounce timer for graph rendering
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._do_load_graphs)
        
        # Visibility check timer for lazy loading
        self._visibility_timer = QTimer(self)
        self._visibility_timer.timeout.connect(self._check_slot_visibility)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup widget UI with QSplitter layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter = QSplitter(Qt.Horizontal)
        
        # LEFT PANE: Sidebar
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        
        title_label = QLabel("Select Graphs")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        sidebar_layout.addWidget(title_label)
        
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("All")
        deselect_all_btn = QPushButton("None")
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(30)
        
        select_all_btn.clicked.connect(self.select_all_graphs)
        deselect_all_btn.clicked.connect(self.deselect_all_graphs)
        refresh_btn.clicked.connect(self._force_refresh)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addWidget(refresh_btn)
        sidebar_layout.addLayout(button_layout)
        
        self.graph_tree = QTreeWidget()
        self.graph_tree.setHeaderHidden(True)
        self.graph_tree.setIndentation(15)
        self.graph_tree.itemChanged.connect(self._on_tree_item_changed)
        self._populate_tree()
        
        sidebar_layout.addWidget(self.graph_tree, 1)
        sidebar_widget.setMinimumWidth(self.SIDEBAR_MIN_WIDTH)
        
        # RIGHT PANE: Graph display
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Connect scroll events for lazy loading
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll)
        
        self.graphs_container = QWidget()
        self.graphs_layout = QVBoxLayout(self.graphs_container)
        self.graphs_layout.setSpacing(12)
        self.graphs_layout.setContentsMargins(8, 8, 8, 8)
        
        self.scroll_area.setWidget(self.graphs_container)
        
        self.no_data_label = QLabel("No graphs available. Run simulation first.")
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.setStyleSheet("color: gray; font-size: 14px; padding: 50px;")
        self.graphs_layout.addWidget(self.no_data_label)
        self.graphs_layout.addStretch()
        
        self.splitter.addWidget(sidebar_widget)
        self.splitter.addWidget(self.scroll_area)
        self.splitter.setSizes([self.SIDEBAR_DEFAULT_WIDTH, 800])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(self.splitter)
    
    def _populate_tree(self):
        """Populate tree with folder hierarchy."""
        from h2_plant.gui.core.plotter import GRAPH_REGISTRY
        
        self.graph_tree.blockSignals(True)
        try:
            self.graph_tree.clear()
            self._tree_items = {}
            
            for folder_name, graph_ids in GRAPH_HIERARCHY.items():
                folder_item = QTreeWidgetItem([folder_name])
                folder_item.setFlags(folder_item.flags() | Qt.ItemIsUserCheckable)
                folder_item.setCheckState(0, Qt.Checked)
                
                for graph_id in graph_ids:
                    if graph_id in GRAPH_REGISTRY:
                        graph_info = GRAPH_REGISTRY[graph_id]
                        child_item = QTreeWidgetItem([graph_info['name']])
                        child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                        child_item.setCheckState(0, Qt.Checked)
                        child_item.setData(0, Qt.UserRole, graph_id)
                        child_item.setToolTip(0, graph_info.get('description', ''))
                        folder_item.addChild(child_item)
                        self._tree_items[graph_id] = child_item
                
                self.graph_tree.addTopLevelItem(folder_item)
                folder_item.setExpanded(True)
        finally:
            self.graph_tree.blockSignals(False)
    
    def _on_tree_item_changed(self, item, column):
        """Handle checkbox changes."""
        self.graph_tree.blockSignals(True)
        try:
            if item.childCount() > 0:
                new_state = item.checkState(column)
                for i in range(item.childCount()):
                    item.child(i).setCheckState(0, new_state)
            else:
                parent = item.parent()
                if parent:
                    self._update_parent_check_state(parent)
        finally:
            self.graph_tree.blockSignals(False)
        
        self._schedule_load_graphs()
    
    def _update_parent_check_state(self, parent):
        """Update parent check state."""
        checked = sum(1 for i in range(parent.childCount()) 
                      if parent.child(i).checkState(0) == Qt.Checked)
        total = parent.childCount()
        
        if checked == 0:
            parent.setCheckState(0, Qt.Unchecked)
        elif checked == total:
            parent.setCheckState(0, Qt.Checked)
        else:
            parent.setCheckState(0, Qt.PartiallyChecked)
    
    def select_all_graphs(self):
        self.graph_tree.blockSignals(True)
        try:
            for i in range(self.graph_tree.topLevelItemCount()):
                folder = self.graph_tree.topLevelItem(i)
                folder.setCheckState(0, Qt.Checked)
                for j in range(folder.childCount()):
                    folder.child(j).setCheckState(0, Qt.Checked)
        finally:
            self.graph_tree.blockSignals(False)
        self._schedule_load_graphs()
    
    def deselect_all_graphs(self):
        self.graph_tree.blockSignals(True)
        try:
            for i in range(self.graph_tree.topLevelItemCount()):
                folder = self.graph_tree.topLevelItem(i)
                folder.setCheckState(0, Qt.Unchecked)
                for j in range(folder.childCount()):
                    folder.child(j).setCheckState(0, Qt.Unchecked)
        finally:
            self.graph_tree.blockSignals(False)
        self._schedule_load_graphs()
    
    def set_simulation_data(self, history):
        """Set simulation history data."""
        self.simulation_data = history
        
        # Pre-normalize DataFrame once
        from h2_plant.gui.core.plotter import normalize_history
        self.normalized_df = normalize_history(history)
        
        # Update cache with new data hash
        self._figure_cache.set_data(history)
        
        self._do_load_graphs()
    
    def _get_checked_graph_ids(self):
        return [gid for gid, item in self._tree_items.items() 
                if item.checkState(0) == Qt.Checked]
    
    def _schedule_load_graphs(self):
        self._render_timer.start(self.DEBOUNCE_DELAY_MS)
    
    def load_graphs(self):
        self._schedule_load_graphs()
    
    def _force_refresh(self):
        """Force clear cache and reload."""
        self._figure_cache.clear()
        self._do_load_graphs()
    
    def _clear_layout(self):
        """Clear layout without clearing cache."""
        self._visibility_timer.stop()
        
        for canvas in self.graph_canvases.values():
            canvas.setParent(None)
            # Don't close figures - they're in cache
        self.graph_canvases.clear()
        
        for slot in self.lazy_slots.values():
            slot.setParent(None)
            slot.deleteLater()
        self.lazy_slots.clear()
        
        self._pending_graphs.clear()
        
        if self.no_data_label:
            self.no_data_label.setParent(None)
            self.no_data_label = None
        
        while self.graphs_layout.count():
            item = self.graphs_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
    
    def _do_load_graphs(self):
        """Create lazy slots for all selected graphs."""
        from h2_plant.gui.core.plotter import GRAPH_REGISTRY
        
        self._clear_layout()
        
        if self.simulation_data is None:
            self._show_message("No simulation data. Run simulation first.", "gray")
            return
        
        checked_ids = self._get_checked_graph_ids()
        
        if not checked_ids:
            self._show_message("No graphs selected.", "gray")
            return
        
        # Create lazy slots for all checked graphs
        for graph_id in checked_ids:
            # Check cache first
            cached_fig = self._figure_cache.get(graph_id)
            if cached_fig:
                # Instantly display cached figure
                self._display_figure(graph_id, cached_fig)
            else:
                # Create lazy slot
                graph_info = GRAPH_REGISTRY.get(graph_id, {})
                slot = LazyGraphSlot(graph_id, graph_info.get('name', graph_id))
                slot.request_generation.connect(self._on_slot_requests_generation)
                self.lazy_slots[graph_id] = slot
                self.graphs_layout.addWidget(slot)
        
        self.graphs_layout.addStretch()
        
        # Start visibility checking
        self._visibility_timer.start(self.VISIBILITY_CHECK_INTERVAL_MS)
        
        # Trigger initial visibility check
        QTimer.singleShot(50, self._check_slot_visibility)
    
    def _on_scroll(self):
        """Handle scroll to check visibility."""
        self._check_slot_visibility()
    
    def _check_slot_visibility(self):
        """Check which lazy slots are visible and trigger generation."""
        if not self.lazy_slots:
            self._visibility_timer.stop()
            return
        
        # Get viewport rectangle in global coordinates
        viewport = self.scroll_area.viewport()
        viewport_rect = viewport.rect()
        viewport_global = viewport.mapToGlobal(viewport_rect.topLeft())
        from PySide6.QtCore import QRect
        global_viewport = QRect(viewport_global.x(), viewport_global.y(),
                                viewport_rect.width(), viewport_rect.height())
        
        for slot in list(self.lazy_slots.values()):
            slot.check_visibility(global_viewport)
    
    @Slot(str)
    def _on_slot_requests_generation(self, graph_id: str):
        """Handle lazy slot requesting graph generation."""
        if graph_id in self._pending_graphs:
            return
        
        self._pending_graphs.add(graph_id)
        
        worker = GraphWorker(graph_id, self.simulation_data, self.normalized_df)
        worker.signals.graph_ready.connect(self._on_graph_generated)
        worker.signals.error.connect(self._on_graph_error)
        self._thread_pool.start(worker)
    
    @Slot(str, object)
    def _on_graph_generated(self, graph_id: str, figure):
        """Handle completed graph generation."""
        self._pending_graphs.discard(graph_id)
        
        if figure is None:
            return
        
        # Cache the figure
        self._figure_cache.put(graph_id, figure)
        
        # Replace lazy slot with canvas
        slot = self.lazy_slots.pop(graph_id, None)
        if slot:
            index = self.graphs_layout.indexOf(slot)
            slot.setParent(None)
            slot.deleteLater()
            
            self._display_figure(graph_id, figure, index)
    
    def _display_figure(self, graph_id: str, figure, index: int = -1):
        """Display a figure as a canvas."""
        from h2_plant.gui.core.plotter import GRAPH_REGISTRY
        
        canvas = FigureCanvas(figure)
        canvas.setMinimumHeight(self.GRAPH_MIN_HEIGHT)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        graph_info = GRAPH_REGISTRY.get(graph_id, {})
        canvas.setToolTip(graph_info.get('description', graph_id))
        
        self.graph_canvases[graph_id] = canvas
        
        if index >= 0:
            self.graphs_layout.insertWidget(index, canvas)
        else:
            insert_pos = max(0, self.graphs_layout.count() - 1)
            self.graphs_layout.insertWidget(insert_pos, canvas)
    
    @Slot(str, str)
    def _on_graph_error(self, graph_id: str, error_msg: str):
        """Handle graph generation error."""
        print(f"Error generating {graph_id}: {error_msg}")
        self._pending_graphs.discard(graph_id)
        
        slot = self.lazy_slots.pop(graph_id, None)
        if slot:
            slot._status_label.setText("Error")
            slot._status_label.setStyleSheet("color: #ff6b6b;")
    
    def _show_message(self, text, color):
        self.no_data_label = QLabel(text)
        self.no_data_label.setAlignment(Qt.AlignCenter)
        self.no_data_label.setStyleSheet(f"color: {color}; font-size: 14px; padding: 50px;")
        self.graphs_layout.addWidget(self.no_data_label)
        self.graphs_layout.addStretch()


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
            PEMStackNode, SOECStackNode, # RectifierNode (Implicit),
            # Storage
            LPTankNode, HPTankNode,
            # Compression
            FillingCompressorNode, OutgoingCompressorNode,
            # Fluid Machinery
            PumpNode, # RecirculationPumpNode (Use PumpNode), ProcessCompressorNode (Use specific compressors),
            # Sources (NEW)
            WindEnergySourceNode,
            # Economics (NEW)
            ArbitrageNode,
            # Flow Control (NEW)
            MixerNode, WaterMixerNode,
            # Logic
            DemandSchedulerNode, EnergyPriceNode,
            # Logistics
            ConsumerNode,
            # Resources
            GridConnectionNode, WaterSupplyNode, # AmbientHeatNode, NaturalGasSupplyNode
            # Other (Disabled - Future Phase)
            # ATRReactorNode, PSAUnitNode, SeparationTankNode,
            # BatteryNode, WaterPurifierNode, UltraPureWaterTankNode
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
                # Use get_properties() from ConfigurableNode for clean extraction
                node_props = node.get_properties() if hasattr(node, 'get_properties') else node.properties()
                
                graph_node = GraphNode(
                    id=node.id,
                    type=node.type_,
                    display_name=node.name(),
                    x=node.x_pos(),
                    y=node.y_pos(),
                    properties=node_props,
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
                
                # Pass simulation data directly to report widget (no disk I/O)
                if hasattr(self, 'report_widget'):
                    self.report_widget.set_simulation_data(history)
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
